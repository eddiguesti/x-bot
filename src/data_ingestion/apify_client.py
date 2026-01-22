"""Apify Twitter scraper client as backup data source."""

import json
import logging
from datetime import datetime, timedelta
from typing import Optional

import httpx

from .x_client import XPost

logger = logging.getLogger(__name__)


class ApifyXClient:
    """
    Client for fetching X posts via Apify Twitter scrapers.

    Apify offers pay-per-result pricing at ~$0.20-0.40 per 1K tweets.
    Free tier: $5/month credits (limited to demo mode for Twitter).

    Scrapers available:
    - apidojo/tweet-scraper (recommended): $0.40/1K tweets
    - kaitoeasyapi/twitter-x-data-tweet-scraper-pay-per-result-cheapest: $0.25/1K
    """

    BASE_URL = "https://api.apify.com/v2"

    # Scraper actor IDs
    SCRAPERS = {
        "tweet-scraper": "apidojo/tweet-scraper",
        "cheapest": "kaitoeasyapi/twitter-x-data-tweet-scraper-pay-per-result-cheapest",
    }

    def __init__(self, api_token: str, scraper: str = "tweet-scraper"):
        """
        Initialize Apify client.

        Args:
            api_token: Apify API token (get from https://console.apify.com/account/integrations)
            scraper: Which scraper to use ("tweet-scraper" or "cheapest")
        """
        self.api_token = api_token
        self.actor_id = self.SCRAPERS.get(scraper, scraper)
        self.client = httpx.Client(timeout=120)

    def fetch_user_tweets(
        self,
        username: str,
        max_tweets: int = 100,
        include_replies: bool = False,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
    ) -> list[XPost]:
        """
        Fetch tweets from a specific user.

        Args:
            username: Twitter username (without @)
            max_tweets: Maximum tweets to fetch
            include_replies: Include reply tweets
            start_date: Start of date range (for historical)
            end_date: End of date range

        Returns:
            List of XPost objects
        """
        # Build search query with date range for historical data
        query = f"from:{username.lstrip('@')}"
        if start_date:
            query += f" since:{start_date.strftime('%Y-%m-%d')}"
        if end_date:
            query += f" until:{end_date.strftime('%Y-%m-%d')}"

        # Input for the tweet-scraper actor
        run_input = {
            "searchTerms": [query],
            "tweetsDesired": max_tweets,
            "includeReplies": include_replies,
        }

        return self._run_actor(run_input)

    def search_tweets(
        self,
        query: str,
        max_tweets: int = 100,
    ) -> list[XPost]:
        """
        Search for tweets matching a query.

        Args:
            query: Search query (e.g., "BTC OR bitcoin from:elonmusk")
            max_tweets: Maximum tweets to fetch

        Returns:
            List of XPost objects
        """
        run_input = {
            "searchTerms": [query],
            "tweetsDesired": max_tweets,
        }

        return self._run_actor(run_input)

    def fetch_posts(
        self,
        usernames: list[str],
        keywords: Optional[list[str]] = None,
        days_back: int = 7,
        limit: int = 1000,
    ) -> list[XPost]:
        """
        Fetch posts matching the criteria (compatible with XClient interface).

        Args:
            usernames: List of usernames to fetch from
            keywords: Optional keyword filter
            days_back: Days to look back
            limit: Maximum posts

        Returns:
            List of XPost objects
        """
        all_posts = []
        per_user_limit = max(10, limit // len(usernames)) if usernames else limit

        for username in usernames:
            try:
                posts = self.fetch_user_tweets(
                    username=username.lstrip("@"),
                    max_tweets=per_user_limit,
                )

                # Filter by keywords if provided
                if keywords:
                    keywords_lower = [k.lower() for k in keywords]
                    posts = [
                        p for p in posts
                        if any(kw in p.text.lower() for kw in keywords_lower)
                    ]

                all_posts.extend(posts)

            except Exception as e:
                logger.warning(f"Error fetching from {username}: {e}")
                continue

        # Sort by time and limit
        all_posts.sort(key=lambda p: p.posted_at, reverse=True)
        return all_posts[:limit]

    def _run_actor(self, run_input: dict) -> list[XPost]:
        """Run an Apify actor and return parsed posts."""
        url = f"{self.BASE_URL}/acts/{self.actor_id}/run-sync-get-dataset-items"

        try:
            response = self.client.post(
                url,
                params={"token": self.api_token},
                json=run_input,
                timeout=120,
            )
            response.raise_for_status()

            data = response.json()
            return self._parse_results(data)

        except httpx.HTTPStatusError as e:
            logger.error(f"Apify API error: {e.response.status_code} - {e.response.text}")
            return []
        except Exception as e:
            logger.error(f"Apify request failed: {e}")
            return []

    def _parse_results(self, data: list[dict]) -> list[XPost]:
        """Parse Apify results into XPost objects."""
        posts = []

        for item in data:
            try:
                # Handle different scraper output formats
                post = XPost(
                    post_id=str(item.get("id", item.get("tweetId", ""))),
                    username=item.get("author", {}).get("userName", item.get("username", "")),
                    text=item.get("text", item.get("fullText", "")),
                    posted_at=self._parse_timestamp(item.get("createdAt", item.get("timestamp"))),
                    url=item.get("url", item.get("tweetUrl")),
                    raw_data=item,
                )
                posts.append(post)
            except Exception as e:
                logger.warning(f"Error parsing tweet: {e}")
                continue

        return posts

    def _parse_timestamp(self, ts) -> datetime:
        """Parse various timestamp formats."""
        if isinstance(ts, datetime):
            return ts
        if isinstance(ts, (int, float)):
            # Check if milliseconds or seconds
            if ts > 1e12:
                return datetime.fromtimestamp(ts / 1000)
            return datetime.fromtimestamp(ts)
        if isinstance(ts, str):
            for fmt in [
                '%a %b %d %H:%M:%S %z %Y',  # Twitter format
                '%Y-%m-%dT%H:%M:%S.%fZ',
                '%Y-%m-%dT%H:%M:%SZ',
                '%Y-%m-%d %H:%M:%S',
            ]:
                try:
                    return datetime.strptime(ts.replace('+0000', '+00:00'), fmt.replace('%z', '+00:00'))
                except ValueError:
                    continue
        return datetime.utcnow()


    def fetch_historical_trading_signals(
        self,
        start_date: datetime,
        end_date: Optional[datetime] = None,
        max_tweets: int = 5000,
    ) -> list[XPost]:
        """
        Fetch historical crypto trading signals for backtesting.

        Args:
            start_date: Start date for historical data (can go back 2+ years)
            end_date: End date (default: now)
            max_tweets: Maximum tweets to fetch

        Returns:
            List of XPost objects with trading signals
        """
        if end_date is None:
            end_date = datetime.utcnow()

        # Crypto trading signal search query
        # Using advanced search operators for better results
        query = (
            '("long BTC" OR "short BTC" OR "bullish bitcoin" OR '
            '"bearish bitcoin" OR "buying BTC" OR "selling BTC" OR '
            '"long ETH" OR "short ETH")'
        )
        query += f" since:{start_date.strftime('%Y-%m-%d')}"
        query += f" until:{end_date.strftime('%Y-%m-%d')}"
        query += " -filter:retweets"  # Exclude retweets

        run_input = {
            "searchTerms": [query],
            "tweetsDesired": max_tweets,
            "sort": "Latest",
        }

        logger.info(f"Fetching historical signals from {start_date} to {end_date}")
        return self._run_actor(run_input)

    def fetch_from_known_traders(
        self,
        trader_usernames: list[str],
        start_date: datetime,
        end_date: Optional[datetime] = None,
        max_per_user: int = 500,
    ) -> list[XPost]:
        """
        Fetch historical tweets from known crypto traders.

        Args:
            trader_usernames: List of known trader usernames
            start_date: Start date for historical data
            end_date: End date (default: now)
            max_per_user: Max tweets per user

        Returns:
            Combined list of XPost objects
        """
        if end_date is None:
            end_date = datetime.utcnow()

        all_posts = []
        total_users = len(trader_usernames)

        for i, username in enumerate(trader_usernames):
            try:
                logger.info(f"Fetching from @{username} ({i+1}/{total_users})...")
                posts = self.fetch_user_tweets(
                    username=username,
                    max_tweets=max_per_user,
                    start_date=start_date,
                    end_date=end_date,
                )
                all_posts.extend(posts)
                logger.info(f"  Got {len(posts)} tweets from @{username}")
            except Exception as e:
                logger.warning(f"Error fetching from @{username}: {e}")
                continue

        logger.info(f"Total: {len(all_posts)} tweets from {total_users} users")
        return all_posts


def get_x_client(settings):
    """
    Factory function to get the appropriate X client based on settings.

    Returns XClient (Macrocosmos) or ApifyXClient based on X_DATA_SOURCE.
    """
    from .x_client import XClient

    source = settings.x_data_source.lower()

    if source == "apify":
        if not settings.apify_api_token:
            raise ValueError("APIFY_API_TOKEN required when X_DATA_SOURCE=apify")
        return ApifyXClient(settings.apify_api_token)

    elif source == "macrocosmos":
        return XClient(settings)

    else:  # "auto" - try macrocosmos first, fall back to apify
        if settings.macrocosmos_api_key:
            return XClient(settings)
        elif settings.apify_api_token:
            return ApifyXClient(settings.apify_api_token)
        else:
            raise ValueError(
                "No X data source configured. "
                "Set MACROCOSMOS_API_KEY or APIFY_API_TOKEN"
            )


# Alternative: Free scraping with twscrape (requires accounts)
class TwscrapeClient:
    """
    Free Twitter scraping using twscrape library.

    Requires Twitter accounts to be added. See: https://github.com/vladkens/twscrape

    Pros: Free, no API costs
    Cons: Requires account management, may get rate limited
    """

    def __init__(self, accounts_file: str = "twitter_accounts.txt"):
        """
        Initialize twscrape client.

        accounts_file format (one per line):
        username:password:email:email_password
        """
        self.accounts_file = accounts_file
        self._api = None

    async def _get_api(self):
        """Lazy-load the twscrape API."""
        if self._api is None:
            try:
                from twscrape import API
                self._api = API()
                # Load accounts if file exists
                import os
                if os.path.exists(self.accounts_file):
                    await self._api.pool.load_from_file(self.accounts_file)
            except ImportError:
                raise ImportError("twscrape not installed. Run: pip install twscrape")
        return self._api

    async def search_tweets(self, query: str, limit: int = 100) -> list[XPost]:
        """Search tweets (async)."""
        api = await self._get_api()
        posts = []

        async for tweet in api.search(query, limit=limit):
            posts.append(XPost(
                post_id=str(tweet.id),
                username=tweet.user.username,
                text=tweet.rawContent,
                posted_at=tweet.date,
                url=tweet.url,
                raw_data={"id": tweet.id},
            ))

        return posts
