"""YouTube client for fetching crypto trading signals via Macrocosmos.

Uses the Macrocosmos SN13 API with source='YouTube'.
Fetches from top crypto YouTube channels for sentiment and signals.
"""

import json
import logging
import math
import time
from collections import OrderedDict
from datetime import datetime, timedelta
from typing import Optional
from dataclasses import dataclass

import macrocosmos as mc

from ..config import Settings
from ..constants import SEEN_POSTS_CACHE_MAX_SIZE, SEEN_POSTS_CACHE_TRIM_SIZE

logger = logging.getLogger(__name__)


@dataclass
class YouTubePost:
    """Parsed YouTube video/comment data."""
    post_id: str
    username: str  # Channel name
    text: str  # Title + description (+ transcript if available)
    posted_at: datetime
    url: Optional[str] = None
    views: int = 0
    likes: int = 0
    comments: int = 0
    engagement_score: float = 1.0
    transcript: Optional[str] = None  # Full video transcript
    raw_data: Optional[dict] = None


class YouTubeClient:
    """Client for fetching YouTube crypto content via Macrocosmos SN13 API."""

    # Top crypto YouTube channels to monitor (50+ channels)
    CRYPTO_CHANNELS = [
        # ========== MEGA INFLUENCERS (1M+ subscribers) ==========
        "BitBoy Crypto",
        "Coin Bureau",
        "DataDash",
        "Altcoin Daily",
        "The Moon",
        "Ivan on Tech",
        "Lark Davis",
        "Benjamin Cowen",
        "Crypto Banter",
        "Ran Neuner",
        "InvestAnswers",
        "Crypto Tips",
        "EllioTrades Crypto",

        # ========== TECHNICAL ANALYSIS ==========
        "Crypto Crew University",
        "Sheldon The Sniper",
        "Michaël van de Poppe",
        "Crypto Jebb",
        "The Crypto Lark",
        "Trader University",
        "The Chart Guys",
        "Crypto Capital Venture",
        "Rekt Capital",
        "Bob Loukas",

        # ========== NEWS & EDUCATION ==========
        "Bankless",
        "The Defiant",
        "Unchained Podcast",
        "Real Vision",
        "Anthony Pompliano",
        "Blockworks",
        "Messari",
        "a]16z crypto",
        "Delphi Digital",

        # ========== DEFI & TECHNICAL ==========
        "Finematics",
        "Whiteboard Crypto",
        "DeFi Dad",
        "Patrick Collins",
        "Smart Contract Programmer",
        "Eat The Blocks",

        # ========== TRADING FOCUSED ==========
        "Crypto Face",
        "Davincij15",
        "MMCrypto",
        "Crypto Zombie",
        "Alessio Rastani",
        "CryptosRUs",
        "Crypto Jebb",
        "Digital Asset News",
        "Crypto Michael",

        # ========== SOLANA & ALT L1s ==========
        "SolanaFloor",
        "Solana",

        # ========== MACRO & INSTITUTIONAL ==========
        "Raoul Pal The Journey Man",
        "Willy Woo",
        "Plan B",
        "Preston Pysh",

        # ========== NEWS CHANNELS ==========
        "CoinDesk",
        "Cointelegraph",
        "Bitcoin Magazine",
        "The Block",

        # ========== AI & NEW NARRATIVES ==========
        "Matthew Berman",
        "AI Explained",
    ]

    # Trading keywords to filter relevant content
    TRADING_KEYWORDS = [
        "buy", "sell", "long", "short", "bullish", "bearish",
        "breakout", "pump", "dump", "moon", "crash",
        "bitcoin", "ethereum", "solana", "altcoin", "crypto",
        "price prediction", "technical analysis", "ta",
        "support", "resistance", "target", "bottom", "top",
    ]

    # Asset keywords for filtering
    ASSET_KEYWORDS = {
        "BTC": ["btc", "bitcoin"],
        "ETH": ["eth", "ethereum"],
        "SOL": ["sol", "solana"],
        "XRP": ["xrp", "ripple"],
        "DOGE": ["doge", "dogecoin"],
        "ADA": ["ada", "cardano"],
        "AVAX": ["avax", "avalanche"],
        "LINK": ["link", "chainlink"],
        "DOT": ["dot", "polkadot"],
        "BNB": ["bnb", "binance"],
        "SHIB": ["shib", "shiba"],
        "ARB": ["arb", "arbitrum"],
        "OP": ["op", "optimism"],
        "APT": ["apt", "aptos"],
        "SUI": ["sui"],
        "SEI": ["sei"],
        "TAO": ["tao", "bittensor"],
        "PEPE": ["pepe"],
    }

    # Rate limiting
    MIN_REQUEST_INTERVAL = 1.0
    MAX_RETRIES = 3
    RETRY_BASE_DELAY = 2.0

    def __init__(self, settings: Settings):
        self.settings = settings
        self._last_api_call: float = 0
        self._seen_post_ids: OrderedDict[str, None] = OrderedDict()

        # Initialize Macrocosmos client
        import os
        api_key = settings.macrocosmos_api_key or os.getenv("MACROCOSMOS_API_KEY", "")

        if api_key:
            self.client = mc.Sn13Client(
                api_key=api_key,
                app_name="crypto_consensus_youtube"
            )
            self.enabled = True
            self.raw_data_dir = settings.raw_data_dir
            self.raw_data_dir.mkdir(parents=True, exist_ok=True)
            logger.info("YouTube client initialized via Macrocosmos")
        else:
            self.client = None
            self.enabled = False
            logger.warning("Macrocosmos API key not provided - YouTube client disabled")

    def _rate_limit(self):
        """Enforce rate limiting between API calls."""
        elapsed = time.time() - self._last_api_call
        if elapsed < self.MIN_REQUEST_INTERVAL:
            time.sleep(self.MIN_REQUEST_INTERVAL - elapsed)
        self._last_api_call = time.time()

    def _retry_with_backoff(self, func, *args, **kwargs):
        """Execute function with exponential backoff retry logic."""
        last_exception = None

        for attempt in range(self.MAX_RETRIES):
            try:
                self._rate_limit()
                return func(*args, **kwargs)
            except Exception as e:
                last_exception = e
                if attempt < self.MAX_RETRIES - 1:
                    delay = self.RETRY_BASE_DELAY * (2 ** attempt)
                    logger.warning(
                        f"YouTube API call failed (attempt {attempt + 1}/{self.MAX_RETRIES}): {e}. "
                        f"Retrying in {delay:.1f}s..."
                    )
                    time.sleep(delay)
                else:
                    logger.error(f"YouTube API call failed after {self.MAX_RETRIES} attempts: {e}")

        raise last_exception if last_exception else Exception("Unknown error")

    def _calculate_engagement_score(self, views: int, likes: int, comments: int) -> float:
        """Calculate engagement score for a video."""
        if views == 0:
            return 1.0

        # Logarithmic scaling
        view_score = math.log1p(views) * 0.5
        like_score = math.log1p(likes) * 2.0
        comment_score = math.log1p(comments) * 1.5

        return min(3.0, max(1.0, 1.0 + (view_score + like_score + comment_score) / 15))

    def _has_trading_signal(self, text: str) -> bool:
        """Check if text contains trading-related keywords."""
        text_lower = text.lower()
        return any(kw in text_lower for kw in self.TRADING_KEYWORDS)

    def _has_asset_mention(self, text: str) -> bool:
        """Check if text mentions any tracked asset."""
        text_lower = text.lower()
        for keywords in self.ASSET_KEYWORDS.values():
            if any(kw in text_lower for kw in keywords):
                return True
        return False

    def _parse_response(self, response) -> list[YouTubePost]:
        """Parse Macrocosmos API response into YouTubePost objects."""
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
                # Skip if already seen
                post_id = str(item.get('id', item.get('video_id', '')))
                if post_id in self._seen_post_ids:
                    continue
                self._seen_post_ids[post_id] = None

                # Get text content (title + description)
                title = item.get('title', '')
                description = item.get('description', item.get('text', ''))
                text = f"{title}\n{description}"

                if not text.strip():
                    continue

                # Check relevance
                if not self._has_asset_mention(text) or not self._has_trading_signal(text):
                    continue

                # Get channel name
                channel = item.get('channel', item.get('username', item.get('author', 'unknown')))

                # Get URL
                url = item.get('url', item.get('uri', ''))

                # Parse timestamp
                timestamp = item.get('datetime', item.get('published_at', item.get('timestamp')))
                posted_at = self._parse_timestamp(timestamp)

                # Engagement metrics
                views = int(item.get('view_count', item.get('views', 0)) or 0)
                likes = int(item.get('like_count', item.get('likes', 0)) or 0)
                comments = int(item.get('comment_count', item.get('comments', 0)) or 0)

                post = YouTubePost(
                    post_id=post_id,
                    username=channel,
                    text=text,
                    posted_at=posted_at,
                    url=url,
                    views=views,
                    likes=likes,
                    comments=comments,
                    engagement_score=self._calculate_engagement_score(views, likes, comments),
                )
                posts.append(post)

            except Exception as e:
                logger.warning(f"Error parsing YouTube post: {e}")
                continue

        return posts

    def _parse_timestamp(self, ts) -> datetime:
        """Parse various timestamp formats."""
        if isinstance(ts, datetime):
            return ts
        if isinstance(ts, (int, float)):
            if ts > 1e12:
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

    def _save_raw_response(self, response, context: str):
        """Save raw API response for audit trail."""
        timestamp = datetime.utcnow().strftime('%Y%m%d_%H%M%S')
        filename = f"youtube_posts_{timestamp}_{context[:30]}.json"
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

            logger.debug(f"Saved raw YouTube response to {filepath}")
        except Exception as e:
            logger.warning(f"Could not save raw YouTube response: {e}")

    def fetch_channel_videos(self, channel_name: str, limit: int = 10) -> list[YouTubePost]:
        """Fetch recent videos from a specific YouTube channel."""
        if not self.enabled:
            return []

        end_date = datetime.utcnow()
        start_date = end_date - timedelta(days=7)  # Last week

        try:
            def _api_call():
                return self.client.sn13.OnDemandData(
                    source='YouTube',
                    usernames=[channel_name],
                    keywords=[],  # Keywords ignored for YouTube
                    start_date=start_date.strftime('%Y-%m-%d'),
                    end_date=end_date.strftime('%Y-%m-%d'),
                    limit=limit,
                )

            response = self._retry_with_backoff(_api_call)
            return self._parse_response(response)

        except Exception as e:
            logger.error(f"Error fetching YouTube channel {channel_name}: {e}")
            return []

    def fetch_trading_signals(
        self,
        limit: int = 100,
        hours_back: int = 72,  # YouTube content is less frequent
    ) -> list[YouTubePost]:
        """
        Fetch trading signals from top crypto YouTube channels.

        Args:
            limit: Maximum videos to fetch per channel
            hours_back: Only include videos from last N hours

        Returns:
            List of YouTubePost objects with trading signals, sorted by engagement
        """
        if not self.enabled:
            logger.warning("YouTube client not enabled - skipping")
            return []

        logger.info(f"Fetching YouTube signals from {len(self.CRYPTO_CHANNELS)} channels...")

        all_posts = []

        # Fetch from top channels (limit to avoid rate limits)
        channels_to_fetch = self.CRYPTO_CHANNELS[:10]  # Top 10 channels

        for channel in channels_to_fetch:
            try:
                posts = self.fetch_channel_videos(channel, limit=limit // len(channels_to_fetch))
                all_posts.extend(posts)
                logger.info(f"YouTube {channel}: {len(posts)} relevant videos")
            except Exception as e:
                logger.error(f"Error fetching YouTube channel {channel}: {e}")
                continue

        # Filter by time
        cutoff = datetime.utcnow() - timedelta(hours=hours_back)
        recent_posts = [p for p in all_posts if p.posted_at >= cutoff]

        # Limit seen cache size (OrderedDict maintains insertion order for FIFO eviction)
        if len(self._seen_post_ids) > SEEN_POSTS_CACHE_MAX_SIZE:
            items_to_keep = list(self._seen_post_ids.keys())[-SEEN_POSTS_CACHE_TRIM_SIZE:]
            self._seen_post_ids = OrderedDict.fromkeys(items_to_keep)

        # Sort by engagement score
        recent_posts.sort(key=lambda x: x.engagement_score, reverse=True)

        logger.info(f"Total unique YouTube signals: {len(recent_posts)}")
        return recent_posts

    def reset_state(self):
        """Reset seen posts cache."""
        self._seen_post_ids.clear()
        logger.info("Reset YouTube client state")

    def _extract_video_id(self, url: str) -> Optional[str]:
        """Extract video ID from YouTube URL."""
        import re
        patterns = [
            r'(?:youtube\.com/watch\?v=|youtu\.be/|youtube\.com/embed/)([a-zA-Z0-9_-]{11})',
            r'youtube\.com/v/([a-zA-Z0-9_-]{11})',
        ]
        for pattern in patterns:
            match = re.search(pattern, url)
            if match:
                return match.group(1)
        return None

    def fetch_transcript(self, video_id: str) -> Optional[str]:
        """
        Fetch transcript for a YouTube video using youtube-transcript-api.

        Args:
            video_id: YouTube video ID (11 characters)

        Returns:
            Full transcript text or None if unavailable
        """
        try:
            from youtube_transcript_api import YouTubeTranscriptApi
            from youtube_transcript_api._errors import (
                TranscriptsDisabled,
                NoTranscriptFound,
                VideoUnavailable,
            )

            # Try to get transcript (prefers manual captions, falls back to auto-generated)
            transcript_list = YouTubeTranscriptApi.get_transcript(
                video_id,
                languages=['en', 'en-US', 'en-GB']  # English preferred
            )

            # Combine all transcript segments
            full_text = ' '.join(entry['text'] for entry in transcript_list)

            # Clean up the text
            full_text = full_text.replace('\n', ' ').strip()

            logger.debug(f"Fetched transcript for {video_id}: {len(full_text)} chars")
            return full_text

        except (TranscriptsDisabled, NoTranscriptFound, VideoUnavailable) as e:
            logger.debug(f"No transcript available for {video_id}: {e}")
            return None
        except ImportError:
            logger.warning("youtube-transcript-api not installed - run: pip install youtube-transcript-api")
            return None
        except Exception as e:
            logger.warning(f"Error fetching transcript for {video_id}: {e}")
            return None

    def enrich_with_transcripts(self, posts: list[YouTubePost], max_posts: int = 20) -> list[YouTubePost]:
        """
        Enrich YouTube posts with transcripts.

        Only fetches transcripts for top posts (by engagement) to avoid rate limits.

        Args:
            posts: List of YouTubePost objects
            max_posts: Maximum number of posts to enrich with transcripts

        Returns:
            Posts with transcripts added to text field
        """
        # Sort by engagement and take top N
        sorted_posts = sorted(posts, key=lambda x: x.engagement_score, reverse=True)
        posts_to_enrich = sorted_posts[:max_posts]

        enriched_count = 0
        for post in posts_to_enrich:
            # Extract video ID from URL or post_id
            video_id = None
            if post.url:
                video_id = self._extract_video_id(post.url)
            if not video_id and len(post.post_id) == 11:
                video_id = post.post_id

            if not video_id:
                continue

            transcript = self.fetch_transcript(video_id)
            if transcript:
                post.transcript = transcript
                # Append transcript to text for signal extraction
                post.text = f"{post.text}\n\n[TRANSCRIPT]\n{transcript}"
                enriched_count += 1

        logger.info(f"Enriched {enriched_count}/{len(posts_to_enrich)} videos with transcripts")
        return posts
