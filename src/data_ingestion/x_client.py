"""Macrocosmos X/Twitter client for fetching creator posts."""

import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import macrocosmos as mc
from pydantic import BaseModel

from ..config import Settings

logger = logging.getLogger(__name__)


class XPost(BaseModel):
    """Parsed X post data."""
    post_id: str
    username: str
    text: str
    posted_at: datetime
    url: Optional[str] = None
    raw_data: dict


class XClient:
    """Client for fetching X posts via Macrocosmos SN13 API."""

    # Crypto trading keywords (max 10 per API request)
    TRADING_KEYWORDS = [
        "crypto", "trading", "long", "short",
        "bullish", "bearish", "buy", "sell"
    ]

    # Extended keywords for specific searches
    ALTCOIN_KEYWORDS = [
        "SOL", "solana", "XRP", "DOGE", "ADA",
        "AVAX", "LINK", "DOT", "MATIC", "SHIB"
    ]

    def __init__(self, settings: Settings):
        self.settings = settings
        self.client = mc.Sn13Client(
            api_key=settings.macrocosmos_api_key,
            app_name="crypto_consensus"
        )
        self.raw_data_dir = settings.raw_data_dir
        self.raw_data_dir.mkdir(parents=True, exist_ok=True)

    def fetch_posts(
        self,
        usernames: list[str],
        keywords: Optional[list[str]] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        limit: int = 1000,
    ) -> list[XPost]:
        """
        Fetch posts from specified users or by keywords.

        Args:
            usernames: List of X usernames to filter by
            keywords: Optional filter keywords
            start_date: Start of time range (default: 24h ago)
            end_date: End of time range (default: now)
            limit: Maximum posts to fetch (max 1000)

        Returns:
            List of parsed XPost objects
        """
        # Default time range
        if end_date is None:
            end_date = datetime.utcnow()
        if start_date is None:
            start_date = end_date - timedelta(hours=24)

        # Default keywords for crypto signals
        if keywords is None:
            keywords = self.TRADING_KEYWORDS

        # Try keyword-based search (more reliable)
        all_posts = self._fetch_by_keywords(
            keywords=keywords,
            start_date=start_date,
            end_date=end_date,
            limit=limit,
        )

        # Filter to only tracked usernames if provided
        if usernames:
            usernames_lower = {u.lower().lstrip('@') for u in usernames}
            filtered_posts = [
                p for p in all_posts
                if p.username.lower() in usernames_lower
            ]

            # If no posts from tracked users, try user-specific search
            if not filtered_posts and len(usernames) <= 5:
                logger.info("No posts from tracked users in keyword search, trying user search...")
                filtered_posts = self._fetch_by_users(
                    usernames=usernames,
                    keywords=keywords,
                    start_date=start_date,
                    end_date=end_date,
                    limit=limit,
                )

            all_posts = filtered_posts

        logger.info(f"Fetched {len(all_posts)} posts")
        return all_posts

    def _fetch_by_keywords(
        self,
        keywords: list[str],
        start_date: datetime,
        end_date: datetime,
        limit: int,
    ) -> list[XPost]:
        """Fetch posts by keyword search."""
        try:
            response = self.client.sn13.OnDemandData(
                source='X',
                keywords=keywords,
                start_date=start_date.strftime('%Y-%m-%d'),
                end_date=end_date.strftime('%Y-%m-%d'),
                limit=limit,
                keyword_mode='any'
            )

            self._save_raw_response(response, ["keyword_search"])
            return self._parse_response(response)

        except Exception as e:
            logger.error(f"Error in keyword search: {e}")
            return []

    def _fetch_by_users(
        self,
        usernames: list[str],
        keywords: list[str],
        start_date: datetime,
        end_date: datetime,
        limit: int,
    ) -> list[XPost]:
        """Fetch posts from specific users."""
        try:
            formatted_users = [
                f"@{u}" if not u.startswith("@") else u
                for u in usernames
            ]

            response = self.client.sn13.OnDemandData(
                source='X',
                usernames=formatted_users,
                keywords=keywords,
                start_date=start_date.strftime('%Y-%m-%d'),
                end_date=end_date.strftime('%Y-%m-%d'),
                limit=limit,
                keyword_mode='any'
            )

            self._save_raw_response(response, usernames)
            return self._parse_response(response)

        except Exception as e:
            logger.error(f"Error fetching from users {usernames}: {e}")
            return []

    def _parse_response(self, response) -> list[XPost]:
        """Parse Macrocosmos API response into XPost objects."""
        posts = []

        # Handle different response formats
        if hasattr(response, 'data'):
            data = response.data
        elif isinstance(response, dict):
            data = response.get('data', [])
        elif isinstance(response, list):
            data = response
        else:
            logger.warning(f"Unknown response format: {type(response)}")
            return []

        if not data:
            return []

        for item in data:
            try:
                # Handle Macrocosmos response format with nested user object
                user_data = item.get('user', {})
                tweet_data = item.get('tweet', {})

                username = user_data.get('username', item.get('username', ''))
                if not username:
                    username = item.get('author', '')
                username = username.lstrip('@')

                text = item.get('text', item.get('content', ''))
                if not text:
                    continue

                # Get post ID from tweet data or item
                post_id = str(tweet_data.get('id', item.get('id', item.get('post_id', ''))))

                # Get URL
                url = item.get('uri', item.get('url', item.get('post_url')))

                # Parse timestamp
                timestamp = item.get('datetime', item.get('created_at', item.get('timestamp')))

                post = XPost(
                    post_id=post_id,
                    username=username,
                    text=text,
                    posted_at=self._parse_timestamp(timestamp),
                    url=url,
                    raw_data=item if isinstance(item, dict) else {},
                )
                posts.append(post)

            except Exception as e:
                logger.warning(f"Error parsing post: {e}")
                continue

        return posts

    def _parse_timestamp(self, ts) -> datetime:
        """Parse various timestamp formats."""
        if isinstance(ts, datetime):
            return ts
        if isinstance(ts, (int, float)):
            if ts > 1e12:  # Milliseconds
                return datetime.fromtimestamp(ts / 1000)
            return datetime.fromtimestamp(ts)
        if isinstance(ts, str):
            for fmt in [
                '%Y-%m-%dT%H:%M:%SZ',
                '%Y-%m-%dT%H:%M:%S.%fZ',
                '%Y-%m-%d %H:%M:%S',
                '%Y-%m-%d',
            ]:
                try:
                    return datetime.strptime(ts, fmt)
                except ValueError:
                    continue
        return datetime.utcnow()

    def _save_raw_response(self, response, context: list[str]):
        """Save raw API response for audit trail."""
        timestamp = datetime.utcnow().strftime('%Y%m%d_%H%M%S')
        context_str = '-'.join(context[:3])[:30]
        filename = f"x_posts_{timestamp}_{context_str}.json"
        filepath = self.raw_data_dir / filename

        try:
            if hasattr(response, 'model_dump'):
                data = response.model_dump()
            elif hasattr(response, '__dict__'):
                data = response.__dict__
            else:
                data = response

            with open(filepath, 'w') as f:
                json.dump(data, f, indent=2, default=str)

            logger.debug(f"Saved raw response to {filepath}")
        except Exception as e:
            logger.warning(f"Could not save raw response: {e}")

    def fetch_trading_signals(
        self,
        limit: int = 100,
        hours_back: int = 24,
    ) -> list[XPost]:
        """
        Fetch posts containing trading signals.

        Args:
            limit: Maximum posts to fetch
            hours_back: Hours to look back

        Returns:
            List of XPost objects with trading signals
        """
        end_date = datetime.utcnow()
        start_date = end_date - timedelta(hours=max(hours_back, 24))  # Minimum 24h

        # Ensure dates are different (API requirement)
        if start_date.date() >= end_date.date():
            start_date = end_date - timedelta(days=1)

        # Use explicit trading keywords (max 10)
        keywords = [
            "long BTC", "short BTC", "bullish BTC",
            "bearish BTC", "buying bitcoin", "selling bitcoin"
        ]

        return self._fetch_by_keywords(
            keywords=keywords,
            start_date=start_date,
            end_date=end_date,
            limit=limit,
        )
