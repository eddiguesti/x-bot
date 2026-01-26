"""YouTube client for fetching crypto trading signals via YouTube Data API.

Uses the official YouTube Data API v3 to fetch recent videos from crypto channels,
then uses youtube-transcript-api for transcripts.

Quota-efficient: ~51 units per cycle for 50 channels (free tier: 10,000/day).
"""

import json
import logging
import math
import time
from collections import OrderedDict
from datetime import datetime, timedelta
from typing import Optional, Dict, List
from dataclasses import dataclass

from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

from ..config import Settings
from ..constants import SEEN_POSTS_CACHE_MAX_SIZE, SEEN_POSTS_CACHE_TRIM_SIZE

logger = logging.getLogger(__name__)


@dataclass
class YouTubePost:
    """Parsed YouTube video data."""
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
    """Client for fetching YouTube crypto content via YouTube Data API v3."""

    # Top 50 crypto YouTube channels - BALANCED for bull/bear/neutral views
    # Format: (channel_handle, channel_id) - we need IDs for API calls
    # Channel IDs can be found at: youtube.com/channel/CHANNEL_ID
    CRYPTO_CHANNELS = [
        # ========== TIER 1: MEGA INFLUENCERS (500K+ subs) ==========
        ("CoinBureau", "UCqK_GSMbpiV8spgD3ZGloSw"),
        ("AltcoinDaily", "UCbLhGKVY-bJPcawebgtNfbw"),
        ("intocryptoverse", "UCRvqjQPSeaWn-uEx-w0XOIg"),  # Benjamin Cowen
        ("CryptoBanter", "UCN9Nj4tjXbVTLYWN0EKly_Q"),
        ("DataDash", "UCCatR7nWbYrkVXdxXb4cGXw"),
        ("TheMoonCarl", "UCc4Rz_T9Sb1w5rqqo9pL1Og"),
        ("IvanOnTech", "UCrYmtJBtLdtm2ov84ulV-yg"),
        ("TheCryptoLark", "UCl2oCaw8hdR_kbqyqd2klIA"),  # Lark Davis

        # ========== MACRO & SKEPTICS (Bearish/Cautious) ==========
        ("CoffeeZilla", "UCFQMnBA3CS502aghlcr0_aw"),
        ("FoldingIdeas", "UCyNtlmLB73-7gtlBz00XOQQ"),  # Dan Olson
        ("TheGrahamStephanShow", "UCa-ckhlKL98F8YXKQ-BALiw"),  # Graham Stephan
        ("PBDPodcast", "UCGX7nGXpz-CmO_Arg-cgJ7A"),  # Patrick Bet-David
        ("ThePlainBagel", "UCFCEuCsyWP0YkP3CZ3Mr01Q"),
        ("BenFelixCSI", "UCDXTQ8nWmx_EhZ2v-kp7QxA"),
        ("TheChartGuys", "UC94jsP7Dl0aKo0SLNje3R4w"),
        ("RealVision", "UCVDN9demk6_gqsn3JMDT6JQ"),

        # ========== BALANCED ANALYSTS (200K-500K) ==========
        ("InvestAnswers", "UClgJyzwGs-GyaNxUHcLZrkg"),
        ("CryptoJebb", "UCviqt9sVw7W6zJIi8eL9_rg"),
        ("CryptosRUs", "UCHop-jpf-huVT1IYw79ymPw"),
        ("CryptoCrewUniversity", "UCE9ODjNIkOHrnSdkYWLfYhg"),
        ("RektCapital", "UCpAd_cSGKPaXn7cBvaaBHuQ"),
        ("CryptoCasey", "UCmLyH9DUSmBkjMkWeXUxO0A"),
        ("EllioTrades", "UCMtJYS0PrtiUwlk6zjGDDKQ"),

        # ========== RISK & BEAR ANALYSIS ==========
        ("CryptoFace", "UCR5CvqXcoZJRjM8VFM55lLA"),
        ("TraderUniversity", "UCnV6jZ3SXmqhHpz3gVPZBkA"),
        ("CryptoCapo", "UCc_FKQjL5AqD0Op90Erjqzg"),
        ("Coinsider", "UCi7egjf0JDHuhznWugXq4hA"),

        # ========== NEWS & RESEARCH (Neutral) ==========
        ("CoinDesk", "UCwgLmyAHfp4q82KPwFU4PLQ"),
        ("Cointelegraph", "UCRqBu-grVX1p97WaX3KCssA"),
        ("BitcoinMagazine", "UCnibZvJWnS2bQ0RLlEgM78Q"),
        ("BanklessHQ", "UCAl9Ld79qaZxp9JzEOwd3aA"),
        ("TheDefiant", "UCL0J4MLEdLP0-UyLu0hCktg"),
        ("Blockworks", "UC59KbPSYj-IsqMgFrAQdapg"),

        # ========== DEFI & TECHNICAL ==========
        ("Finematics", "UCh1ob28ceGdqohUnR7vBACA"),
        ("WhiteboardCrypto", "UCsYYksPHiGqXHPoHI-fm5sg"),
        ("PatrickAlphaC", "UCn-3f8tw_E1jZvhuHatROwA"),  # Patrick Collins

        # ========== TRADING (Both Directions) ==========
        ("MMCrypto", "UCzfyxhFgllSGIg-LmUktJig"),
        ("CryptoCapitalVenture", "UCXJ4a_lJaf-rPmoCQBE9rvw"),
        ("Boxmining", "UCxODjeUwZHk3p-7TU-IsDOA"),
        ("CryptoKirby", "UCMtJYS0PrtiUwlk6zjGDDKQ"),
        ("TheCryptoSniper", "UCb3_5ZCPYdrQmqzSJd9LUww"),

        # ========== ADDITIONAL CHANNELS ==========
        ("aantonop", "UCJWCJCWOxBYSi5DhCieLOLQ"),  # Andreas Antonopoulos
        ("TechLead", "UC4xKdmAXFh4ACyhpiQ_3qBg"),
        ("SatoshiStacker", "UCLWOd8Qz-tQ2-hRoNw0IvZw"),
        ("DigitalAssetNews", "UCJgHxpqfhWEEjYQu0YQhSFg"),
        ("TheCryptoZombie", "UCiUnrCUGCJTCC7KjuW493Ww"),
        ("TokenMetrics", "UCu4K4Tqv2CBX1qyJxthMfHA"),
        ("CoinGecko", "UC1wSPt1KTe_pYYakWEh3Y9Q"),
        ("BitBoy", "UCjemQfjaXAzA-95RKoy9n_g"),  # Ben Armstrong
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
    MIN_REQUEST_INTERVAL = 0.1  # YouTube API is fast
    MAX_RETRIES = 3
    RETRY_BASE_DELAY = 2.0

    def __init__(self, settings: Settings):
        self.settings = settings
        self._last_api_call: float = 0
        self._seen_post_ids: OrderedDict[str, None] = OrderedDict()
        self._channel_upload_playlists: Dict[str, str] = {}  # Cache channel -> uploads playlist ID

        # Initialize YouTube Data API client
        import os
        api_key = settings.youtube_api_key or os.getenv("YOUTUBE_API_KEY", "")

        if api_key:
            try:
                self.youtube = build('youtube', 'v3', developerKey=api_key)
                self.enabled = True
                self.raw_data_dir = settings.raw_data_dir
                self.raw_data_dir.mkdir(parents=True, exist_ok=True)
                logger.info(f"YouTube client initialized with {len(self.CRYPTO_CHANNELS)} channels")
            except Exception as e:
                logger.error(f"Failed to initialize YouTube API client: {e}")
                self.youtube = None
                self.enabled = False
        else:
            self.youtube = None
            self.enabled = False
            logger.warning("YOUTUBE_API_KEY not provided - YouTube client disabled")

    def _rate_limit(self):
        """Enforce rate limiting between API calls."""
        elapsed = time.time() - self._last_api_call
        if elapsed < self.MIN_REQUEST_INTERVAL:
            time.sleep(self.MIN_REQUEST_INTERVAL - elapsed)
        self._last_api_call = time.time()

    def _get_uploads_playlist_id(self, channel_id: str) -> Optional[str]:
        """Get the uploads playlist ID for a channel (cached)."""
        if channel_id in self._channel_upload_playlists:
            return self._channel_upload_playlists[channel_id]

        try:
            self._rate_limit()
            response = self.youtube.channels().list(
                part='contentDetails',
                id=channel_id
            ).execute()

            if response.get('items'):
                playlist_id = response['items'][0]['contentDetails']['relatedPlaylists']['uploads']
                self._channel_upload_playlists[channel_id] = playlist_id
                return playlist_id
        except HttpError as e:
            logger.warning(f"Failed to get uploads playlist for {channel_id}: {e}")
        except Exception as e:
            logger.warning(f"Error getting uploads playlist: {e}")

        return None

    def _get_recent_video_ids(self, playlist_id: str, max_results: int = 5) -> List[str]:
        """Get recent video IDs from a playlist."""
        try:
            self._rate_limit()
            response = self.youtube.playlistItems().list(
                part='contentDetails',
                playlistId=playlist_id,
                maxResults=max_results
            ).execute()

            video_ids = []
            for item in response.get('items', []):
                video_id = item['contentDetails']['videoId']
                video_ids.append(video_id)

            return video_ids
        except HttpError as e:
            logger.warning(f"Failed to get videos from playlist {playlist_id}: {e}")
        except Exception as e:
            logger.warning(f"Error getting playlist videos: {e}")

        return []

    def _get_video_details(self, video_ids: List[str]) -> List[dict]:
        """Get video details for multiple videos (batched, 1 unit per 50 videos)."""
        if not video_ids:
            return []

        all_videos = []

        # Batch in groups of 50 (API limit)
        for i in range(0, len(video_ids), 50):
            batch = video_ids[i:i + 50]
            try:
                self._rate_limit()
                response = self.youtube.videos().list(
                    part='snippet,statistics',
                    id=','.join(batch)
                ).execute()

                all_videos.extend(response.get('items', []))
            except HttpError as e:
                logger.warning(f"Failed to get video details: {e}")
            except Exception as e:
                logger.warning(f"Error getting video details: {e}")

        return all_videos

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

    def _parse_video(self, video: dict, channel_name: str) -> Optional[YouTubePost]:
        """Parse a video API response into a YouTubePost."""
        try:
            video_id = video['id']

            # Skip if already seen
            if video_id in self._seen_post_ids:
                return None
            self._seen_post_ids[video_id] = None

            snippet = video.get('snippet', {})
            stats = video.get('statistics', {})

            # Get text content
            title = snippet.get('title', '')
            description = snippet.get('description', '')
            text = f"{title}\n{description}"

            if not text.strip():
                return None

            # Check relevance - require trading signal OR asset mention
            if not self._has_trading_signal(text) and not self._has_asset_mention(text):
                return None

            # Parse timestamp
            published_at = snippet.get('publishedAt', '')
            try:
                posted_at = datetime.strptime(published_at, '%Y-%m-%dT%H:%M:%SZ')
            except:
                posted_at = datetime.utcnow()

            # Engagement metrics
            views = int(stats.get('viewCount', 0) or 0)
            likes = int(stats.get('likeCount', 0) or 0)
            comments = int(stats.get('commentCount', 0) or 0)

            return YouTubePost(
                post_id=video_id,
                username=channel_name,
                text=text,
                posted_at=posted_at,
                url=f"https://www.youtube.com/watch?v={video_id}",
                views=views,
                likes=likes,
                comments=comments,
                engagement_score=self._calculate_engagement_score(views, likes, comments),
                raw_data=video,
            )

        except Exception as e:
            logger.warning(f"Error parsing video: {e}")
            return None

    def _save_raw_response(self, data: list, context: str):
        """Save raw API response for audit trail."""
        timestamp = datetime.utcnow().strftime('%Y%m%d_%H%M%S')
        filename = f"youtube_posts_{timestamp}_{context[:30]}.json"
        filepath = self.raw_data_dir / filename

        try:
            with open(filepath, 'w') as f:
                json.dump(data, f, indent=2, default=str)

            logger.debug(f"Saved raw YouTube response to {filepath}")
        except Exception as e:
            logger.warning(f"Could not save raw YouTube response: {e}")

    def fetch_channel_videos(self, channel_name: str, channel_id: str, limit: int = 5) -> List[YouTubePost]:
        """Fetch recent videos from a specific YouTube channel.

        Args:
            channel_name: Human-readable channel name
            channel_id: YouTube channel ID (UC...)
            limit: Maximum number of videos to fetch
        """
        if not self.enabled:
            return []

        # Get uploads playlist ID
        playlist_id = self._get_uploads_playlist_id(channel_id)
        if not playlist_id:
            return []

        # Get recent video IDs
        video_ids = self._get_recent_video_ids(playlist_id, max_results=limit)
        if not video_ids:
            return []

        # Get video details
        videos = self._get_video_details(video_ids)

        # Parse into YouTubePost objects
        posts = []
        for video in videos:
            post = self._parse_video(video, channel_name)
            if post:
                posts.append(post)

        return posts

    def fetch_trading_signals(
        self,
        limit: int = 100,
        hours_back: int = 72,
    ) -> List[YouTubePost]:
        """
        Fetch trading signals from top crypto YouTube channels.

        Args:
            limit: Maximum videos to fetch per channel (5 default)
            hours_back: Only include videos from last N hours

        Returns:
            List of YouTubePost objects with trading signals, sorted by engagement
        """
        if not self.enabled:
            logger.warning("YouTube client not enabled - skipping")
            return []

        logger.info(f"Fetching YouTube signals from {len(self.CRYPTO_CHANNELS)} channels...")

        all_video_ids = []
        channel_map = {}  # video_id -> channel_name

        # Step 1: Get recent video IDs from all channels (1 unit per channel)
        videos_per_channel = max(3, limit // len(self.CRYPTO_CHANNELS))

        for channel_name, channel_id in self.CRYPTO_CHANNELS:
            try:
                playlist_id = self._get_uploads_playlist_id(channel_id)
                if not playlist_id:
                    continue

                video_ids = self._get_recent_video_ids(playlist_id, max_results=videos_per_channel)
                for vid in video_ids:
                    if vid not in self._seen_post_ids:
                        all_video_ids.append(vid)
                        channel_map[vid] = channel_name

            except Exception as e:
                logger.warning(f"Error fetching from {channel_name}: {e}")
                continue

        logger.info(f"Found {len(all_video_ids)} new video IDs from channels")

        if not all_video_ids:
            return []

        # Step 2: Get video details in batches (1 unit per 50 videos)
        videos = self._get_video_details(all_video_ids)
        self._save_raw_response(videos, "batch_fetch")

        # Step 3: Parse into YouTubePost objects
        all_posts = []
        for video in videos:
            video_id = video.get('id')
            channel_name = channel_map.get(video_id, 'Unknown')
            post = self._parse_video(video, channel_name)
            if post:
                all_posts.append(post)

        # Filter by time
        cutoff = datetime.utcnow() - timedelta(hours=hours_back)
        recent_posts = [p for p in all_posts if p.posted_at >= cutoff]

        # Limit seen cache size
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

    def test_connection(self) -> dict:
        """Test if YouTube API is working and return diagnostic info."""
        if not self.enabled:
            return {"status": "disabled", "reason": "No API key configured"}

        result = {
            "status": "unknown",
            "channels_accessible": 0,
            "videos_found": 0,
            "error": None,
        }

        try:
            # Test with first channel
            channel_name, channel_id = self.CRYPTO_CHANNELS[0]
            posts = self.fetch_channel_videos(channel_name, channel_id, limit=3)

            result["channels_accessible"] = 1
            result["videos_found"] = len(posts)

            if posts:
                result["status"] = "working"
                result["sample_video"] = posts[0].text[:100] + "..."
            else:
                result["status"] = "no_data"

        except Exception as e:
            result["error"] = str(e)
            result["status"] = "error"

        return result

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

    def enrich_with_transcripts(self, posts: List[YouTubePost], max_posts: int = 20) -> List[YouTubePost]:
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
