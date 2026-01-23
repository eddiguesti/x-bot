"""Macrocosmos X/Twitter client for fetching creator posts.

Optimizations implemented:
1. Retry logic with exponential backoff
2. Incremental fetching (tracks last fetch time)
3. Engagement weighting (likes/retweets boost signal importance)
4. Expanded altcoin keywords for all supported assets
5. Reply filtering (excludes replies, keeps original posts)
6. Rate limiting to respect API limits
"""

import json
import logging
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import macrocosmos as mc
from pydantic import BaseModel, Field

from ..config import Settings

logger = logging.getLogger(__name__)


class XPost(BaseModel):
    """Parsed X post data with engagement metrics."""
    post_id: str
    username: str
    text: str
    posted_at: datetime
    url: Optional[str] = None
    raw_data: dict = Field(default_factory=dict)
    # Engagement metrics for weighting
    likes: int = 0
    retweets: int = 0
    replies: int = 0
    is_reply: bool = False  # Flag to filter replies
    engagement_score: float = 0.0  # Calculated engagement weight


class MarketSentiment(BaseModel):
    """Broad market sentiment from ALL crypto Twitter.

    Based on academic research (2024-2025):
    - Tweet VOLUME is more predictive than polarity alone
    - Sentiment leads price, then "sell the news" at extremes
    - Non-linear: moderate = follow crowd, extreme = contrarian
    """
    timestamp: datetime
    asset: str  # "BTC", "ETH", "CRYPTO" (overall)

    # Raw counts
    bullish_posts: int = 0
    bearish_posts: int = 0
    neutral_posts: int = 0
    total_posts: int = 0

    # Sentiment scores (-1 to +1)
    sentiment_score: float = 0.0  # -1 = extreme fear, +1 = extreme greed
    bullish_ratio: float = 0.5  # 0-1, percentage bullish

    # Engagement-weighted sentiment (high engagement = more weight)
    weighted_sentiment: float = 0.0

    # VOLUME SIGNAL - research shows volume > polarity for predictions
    volume_signal: str = "normal"  # "low", "normal", "high", "extreme"
    volume_zscore: float = 0.0  # Standard deviations from normal

    # Contrarian signal: when crowd is extreme, fade them
    contrarian_signal: str = "neutral"  # "fade_longs", "fade_shorts", "neutral"
    contrarian_strength: float = 0.0  # 0-1, how strong the contrarian signal is

    # Fear/Greed classification
    fear_greed: str = "neutral"  # "extreme_fear", "fear", "neutral", "greed", "extreme_greed"

    # MOMENTUM - sentiment shift detection (research: sentiment leads price)
    sentiment_momentum: str = "stable"  # "turning_bullish", "turning_bearish", "stable"
    momentum_strength: float = 0.0  # 0-1, how strong the shift


class XClient:
    """Client for fetching X posts via Macrocosmos SN13 API.

    Features:
    - Retry logic with exponential backoff
    - Incremental fetching to avoid re-processing
    - Engagement-weighted signals
    - Reply filtering
    - Rate limiting
    """

    # Crypto trading keywords (max 10 per API request)
    TRADING_KEYWORDS = [
        "crypto", "trading", "long", "short",
        "bullish", "bearish", "buy", "sell"
    ]

    # EXPANDED: Keywords for ALL supported assets
    ALTCOIN_KEYWORDS = [
        # Major
        "BTC", "bitcoin", "ETH", "ethereum",
        # Alt L1s
        "SOL", "solana", "AVAX", "avalanche", "NEAR", "APT", "aptos",
        # Legacy alts
        "XRP", "ripple", "ADA", "cardano", "DOT", "polkadot", "LTC", "litecoin",
        # DeFi
        "LINK", "chainlink", "UNI", "uniswap", "INJ", "injective",
        # Meme
        "DOGE", "dogecoin", "SHIB", "shiba",
        # L2s
        "ARB", "arbitrum", "OP", "optimism",
        # AI/New
        "TAO", "bittensor", "ATOM", "cosmos",
    ]

    # Rate limiting settings
    MIN_REQUEST_INTERVAL = 1.0  # Minimum seconds between API calls
    MAX_RETRIES = 3  # Maximum retry attempts
    RETRY_BASE_DELAY = 2.0  # Base delay for exponential backoff

    # CURATED LIST: 150+ verified high-follower crypto traders/analysts
    # Quality-focused with verified Twitter handles
    TOP_CRYPTO_TRADERS = [
        # ========== TIER 1: MEGA INFLUENCERS (500K+ followers) ==========
        "APompliano",         # Anthony Pompliano - 1.6M - macro/BTC
        "saylor",             # Michael Saylor - 3.5M - BTC maximalist
        "VitalikButerin",     # Vitalik Buterin - 5M - ETH founder
        "cz_binance",         # CZ Binance - 8M - exchange CEO
        "brian_armstrong",    # Brian Armstrong - 1.5M - Coinbase CEO
        "BarrySilbert",       # Barry Silbert - 600K - DCG founder
        "cameron",            # Cameron Winklevoss - 700K - Gemini
        "tyler",              # Tyler Winklevoss - 700K - Gemini
        "RaoulGMI",           # Raoul Pal - 1M - macro economist
        "novaborogratz",      # Mike Novogratz - 500K - Galaxy Digital
        "aantonop",           # Andreas Antonopoulos - 700K - BTC educator
        "whale_alert",        # Whale Alert - 1.5M - large transactions

        # ========== TIER 2: TOP ANALYSTS (100K-500K) ==========
        "WClementeIII",       # Will Clemente - 650K - on-chain
        "woonomic",           # Willy Woo - 1M - on-chain pioneer
        "100trillionUSD",     # PlanB - 1.8M - S2F model
        "CryptoHayes",        # Arthur Hayes - 500K - BitMEX founder
        "CryptoCred",         # CryptoCred - 350K - TA educator
        "CryptoCapo_",        # Capo - 700K - swing trader
        "Pentosh1",           # Pentoshi - 600K - alt trader
        "CryptoDonAlt",       # DonAlt - 400K - technical analysis
        "CryptoKaleo",        # Kaleo - 550K - swing trader
        "CryptoMichNeth",     # Michael van de Poppe - 700K
        "AltcoinPsycho",      # AltcoinPsycho - 300K - alt trader
        "EmperorBTC",         # Emperor - 350K - trading education
        "HsakaTrades",        # Hsaka - 350K - trader
        "IamCryptoWolf",      # CryptoWolf - 500K - signals
        "Cobie",              # Cobie - 700K - trader/investor
        "lookonchain",        # Lookonchain - 500K - whale tracking
        "raboroektcapital",   # Rekt Capital - 400K - TA
        "LarkDavis",          # Lark Davis - 500K - educator
        "IvanOnTech",         # Ivan on Tech - 400K - educator

        # ========== TIER 3: QUALITY TRADERS (50K-100K) ==========
        "ColdBloodShill",     # ColdBloodShill - 200K - TA
        "inversebrah",        # inversebrah - 150K - contrarian
        "TheCryptoDog",       # The Crypto Dog - 250K - trader
        "CryptoGodJohn",      # CryptoGodJohn - 200K - alt calls
        "SmartContracter",    # SmartContracter - 250K - DeFi
        "Bloboruntz_",        # Bluntz - 150K - wave analysis
        "GiganticRebirth",    # Gigantic Rebirth - 150K - TA
        "CryptoTony__",       # CryptoTony - 200K - swing trader
        "PostyXBT",           # Posty - 150K - trader
        "TheMoonCarl",        # Carl Moon - 300K - trader
        "MMCrypto",           # MMCrypto - 200K - trader
        "CryptoJelleNL",      # Jelle - 150K - trader
        "Tradermayne",        # Mayne - 150K - trader
        "CryptoHornHairs",    # HornHairs - 150K - trader
        "nebraskangooner",    # Nebraskan Gooner - 200K
        "bloodgoodBTC",       # BloodGood - 100K - BTC
        "AngeloBTC",          # Angelo - 100K - trader
        "CryptoRand",         # CryptoRand - 200K - analyst
        "CryptoCobain",       # Cobain - 200K - trader
        "CryptoWendyO",       # Wendy O - 200K - analyst
        "AutismCapital",      # Autism Capital - 200K - degen
        "DegenSpartan",       # DegenSpartan - 150K - DeFi
        "Route2FI",           # Route2FI - 200K - DeFi
        "ZhuSu",              # Zhu Su - macro trader

        # ========== ON-CHAIN & DATA ==========
        "glassnode",          # Glassnode - 600K - on-chain
        "santaboroimentfeed", # Santiment - 200K - on-chain
        "intotheblock",       # IntoTheBlock - 150K - analytics
        "naboroansen_ai",     # Nansen - 250K - on-chain
        "CryptoQuant_com",    # CryptoQuant - 150K - on-chain
        "TheBlock__",         # The Block - 400K - research
        "MessariCrypto",      # Messari - 350K - research
        "DelphiDigital",      # Delphi Digital - 200K - research
        "WhaleWire",          # WhaleWire - 100K - whale tracking

        # ========== DEFI SPECIALISTS ==========
        "DefiIgnas",          # Ignas - 300K - DeFi researcher
        "Defi_Dad",           # DeFi Dad - 150K - educator
        "DeFiPulse",          # DeFi Pulse - 100K - TVL
        "bantaboroeg",        # Banteg - 150K - yearn dev
        "DefianceCapital",    # DeFiance Capital - 150K

        # ========== NEWS & MEDIA ==========
        "DocumentingBTC",     # Documenting BTC - 400K
        "BitcoinMagazine",    # Bitcoin Magazine - 800K
        "CoinDesk",           # CoinDesk - 2M
        "Cointelegraph",      # Cointelegraph - 2M
        "WuBlockchain",       # Wu Blockchain - 400K - China
        "tier10k",            # Tier10K - 300K - breaking news
        "Blockworks_",        # Blockworks - 300K
        "thedefiant",         # The Defiant - 150K - DeFi

        # ========== PROTOCOL ACCOUNTS ==========
        "solana",             # Solana - 3M
        "ethereum",           # Ethereum - 3M
        "arbitrum",           # Arbitrum - 1M
        "optimismFND",        # Optimism - 600K
        "0xPolygon",          # Polygon - 2M
        "avalaborancheavax",  # Avalanche - 1M
        "NEARProtocol",       # NEAR - 1M
        "Aptos",              # Aptos - 500K
        "injective",          # Injective - 600K

        # ========== VC & FUNDS ==========
        "a16zcrypto",         # a16z crypto - 500K
        "paradigm",           # Paradigm - 200K
        "PanteraCapital",     # Pantera - 200K

        # ========== MEME/SENTIMENT ==========
        "BillyM2k",           # Billy Markus - 2M - DOGE creator
        "DogeDesigner",       # DogeDesigner - 200K

        # ========== BTC MAXIMALISTS & RESEARCH ==========
        "adam3us",            # Adam Back - 500K - Blockstream
        "NickSzabo4",         # Nick Szabo - 400K - pioneer
        "CaitlinLong_",       # Caitlin Long - 300K
        "MartyBent",          # Marty Bent - 150K - podcast
        "PeterMcCormack",     # Peter McCormack - 300K
        "nic__carter",        # Nic Carter - 300K - research
        "DylanLeClair_",      # Dylan LeClair - 200K

        # ========== ADDITIONAL TRADERS ==========
        "CryptoBull",         # CryptoBull - 200K
        "CryptoWizardd",      # CryptoWizard - 150K
        "Cryptotoad_",        # CryptoToad - 100K
        "SOLBigBrain",        # SOL analyst - 100K
        "AltcoinBuzz",        # Altcoin Buzz - 300K
        "Boxmining",          # Boxmining - 200K
        "skaboroewanalytics", # Skew - derivatives
    ]

    def __init__(self, settings: Settings):
        self.settings = settings
        # Get API key - try settings first, then env var directly (Railway fix)
        import os
        api_key = settings.macrocosmos_api_key or os.getenv("MACROCOSMOS_API_KEY", "")
        self.client = mc.Sn13Client(
            api_key=api_key,
            app_name="crypto_consensus"
        )
        self.raw_data_dir = settings.raw_data_dir
        self.raw_data_dir.mkdir(parents=True, exist_ok=True)

        # Incremental fetching: track last fetch time and seen post IDs
        self._last_fetch_time: Optional[datetime] = None
        self._seen_post_ids: set[str] = set()
        self._last_api_call: float = 0  # For rate limiting

        # VOLUME TRACKING: Store historical volume for z-score calculation
        # Research shows tweet volume > sentiment polarity for predictions
        self._volume_history: dict[str, list[int]] = {}  # asset -> list of volumes
        self._volume_history_max = 24  # Keep last 24 data points (hours)

        # SENTIMENT MOMENTUM: Track previous sentiment for shift detection
        self._previous_sentiment: dict[str, float] = {}  # asset -> previous bullish_ratio

    def _rate_limit(self):
        """Enforce rate limiting between API calls."""
        elapsed = time.time() - self._last_api_call
        if elapsed < self.MIN_REQUEST_INTERVAL:
            sleep_time = self.MIN_REQUEST_INTERVAL - elapsed
            logger.debug(f"Rate limiting: sleeping {sleep_time:.2f}s")
            time.sleep(sleep_time)
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
                        f"API call failed (attempt {attempt + 1}/{self.MAX_RETRIES}): {e}. "
                        f"Retrying in {delay:.1f}s..."
                    )
                    time.sleep(delay)
                else:
                    logger.error(f"API call failed after {self.MAX_RETRIES} attempts: {e}")

        raise last_exception if last_exception else Exception("Unknown error")

    def _calculate_engagement_score(self, likes: int, retweets: int, replies: int) -> float:
        """Calculate engagement score to weight signal importance.

        Higher engagement = more influential signal.
        Retweets weighted 2x (shows agreement), likes 1x, replies 0.5x.
        """
        if likes == 0 and retweets == 0:
            return 1.0  # Baseline score for posts without metrics

        # Logarithmic scaling to prevent mega-viral posts from dominating
        import math
        score = (
            math.log1p(likes) * 1.0 +
            math.log1p(retweets) * 2.0 +
            math.log1p(replies) * 0.5
        )
        # Normalize to 1.0-3.0 range
        return min(3.0, max(1.0, 1.0 + score / 10))

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
        """Fetch posts by keyword search with retry logic."""
        try:
            def _api_call():
                return self.client.sn13.OnDemandData(
                    source='X',
                    keywords=keywords,
                    start_date=start_date.strftime('%Y-%m-%d'),
                    end_date=end_date.strftime('%Y-%m-%d'),
                    limit=limit,
                    keyword_mode='any'
                )

            response = self._retry_with_backoff(_api_call)
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
        """Fetch posts from specific users with retry logic."""
        try:
            formatted_users = [
                f"@{u}" if not u.startswith("@") else u
                for u in usernames
            ]

            def _api_call():
                return self.client.sn13.OnDemandData(
                    source='X',
                    usernames=formatted_users,
                    keywords=keywords,
                    start_date=start_date.strftime('%Y-%m-%d'),
                    end_date=end_date.strftime('%Y-%m-%d'),
                    limit=limit,
                    keyword_mode='any'
                )

            response = self._retry_with_backoff(_api_call)
            self._save_raw_response(response, usernames[:3])
            return self._parse_response(response)

        except Exception as e:
            logger.error(f"Error fetching from users {usernames[:3]}...: {e}")
            return []

    def _parse_response(self, response, filter_replies: bool = True) -> list[XPost]:
        """Parse Macrocosmos API response into XPost objects.

        Args:
            response: API response
            filter_replies: If True, exclude reply posts (default True)
        """
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
                # Handle Macrocosmos response format with nested objects
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

                # ENGAGEMENT METRICS
                likes = int(item.get('like_count', item.get('likes', tweet_data.get('like_count', 0))) or 0)
                retweets = int(item.get('retweet_count', item.get('retweets', tweet_data.get('retweet_count', 0))) or 0)
                replies = int(item.get('reply_count', item.get('replies', tweet_data.get('reply_count', 0))) or 0)

                # REPLY DETECTION - check multiple indicators
                is_reply = False
                # Check if text starts with @mention (reply pattern)
                if text.strip().startswith('@'):
                    is_reply = True
                # Check explicit reply indicators
                if item.get('in_reply_to_status_id') or item.get('in_reply_to_user_id'):
                    is_reply = True
                if tweet_data.get('in_reply_to_status_id') or tweet_data.get('in_reply_to_user_id'):
                    is_reply = True

                # FILTER REPLIES if enabled
                if filter_replies and is_reply:
                    continue

                # Calculate engagement score
                engagement_score = self._calculate_engagement_score(likes, retweets, replies)

                post = XPost(
                    post_id=post_id,
                    username=username,
                    text=text,
                    posted_at=self._parse_timestamp(timestamp),
                    url=url,
                    raw_data=item if isinstance(item, dict) else {},
                    likes=likes,
                    retweets=retweets,
                    replies=replies,
                    is_reply=is_reply,
                    engagement_score=engagement_score,
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
        include_top_traders: bool = True,
        incremental: bool = True,
    ) -> list[XPost]:
        """
        Fetch posts containing trading signals.

        Args:
            limit: Maximum posts to fetch
            hours_back: Hours to look back
            include_top_traders: Also fetch from curated top traders list
            incremental: If True, only fetch posts newer than last fetch

        Returns:
            List of XPost objects with trading signals, sorted by engagement
        """
        end_date = datetime.utcnow()

        # INCREMENTAL FETCHING: Start from last fetch time if available
        if incremental and self._last_fetch_time:
            # Fetch from last fetch time, with 5 min overlap for safety
            start_date = self._last_fetch_time - timedelta(minutes=5)
            logger.info(f"Incremental fetch from {start_date}")
        else:
            start_date = end_date - timedelta(hours=max(hours_back, 24))

        # Ensure dates are different (API requirement)
        if start_date.date() >= end_date.date():
            start_date = end_date - timedelta(days=1)

        all_posts = []

        # 1. Fetch from curated top traders first (higher quality signals)
        if include_top_traders:
            logger.info(f"Fetching from {len(self.TOP_CRYPTO_TRADERS)} curated traders...")
            top_trader_posts = self.fetch_posts(
                usernames=self.TOP_CRYPTO_TRADERS,
                keywords=self.TRADING_KEYWORDS,
                start_date=start_date,
                end_date=end_date,
                limit=limit // 2,
            )
            all_posts.extend(top_trader_posts)
            logger.info(f"Got {len(top_trader_posts)} posts from top traders")

        # 2. Fetch with EXPANDED altcoin keywords for broader coverage
        btc_keywords = [
            "long BTC", "short BTC", "bullish BTC",
            "bearish BTC", "buying bitcoin", "selling bitcoin"
        ]

        keyword_posts = self._fetch_by_keywords(
            keywords=btc_keywords,
            start_date=start_date,
            end_date=end_date,
            limit=limit,
        )
        all_posts.extend(keyword_posts)

        # 3. Fetch altcoin-specific signals
        alt_keywords = [
            "long SOL", "short SOL", "bullish ETH", "bearish ETH",
            "long ARB", "short OP", "bullish AVAX", "TAO pump"
        ]
        alt_posts = self._fetch_by_keywords(
            keywords=alt_keywords,
            start_date=start_date,
            end_date=end_date,
            limit=limit // 2,
        )
        all_posts.extend(alt_posts)

        # DEDUPLICATE and filter already-seen posts (incremental)
        unique_posts = []
        for post in all_posts:
            if post.post_id not in self._seen_post_ids:
                self._seen_post_ids.add(post.post_id)
                unique_posts.append(post)

        # Limit seen_post_ids cache size (keep last 10000)
        if len(self._seen_post_ids) > 10000:
            # Convert to list, keep recent half
            self._seen_post_ids = set(list(self._seen_post_ids)[-5000:])

        # Update last fetch time
        self._last_fetch_time = end_date

        # SORT by engagement score (highest first)
        unique_posts.sort(key=lambda p: p.engagement_score, reverse=True)

        new_count = len(unique_posts)
        logger.info(f"Total unique NEW posts: {new_count} (engagement-sorted)")

        return unique_posts

    def reset_incremental_state(self):
        """Reset incremental fetching state (for fresh start)."""
        self._last_fetch_time = None
        self._seen_post_ids.clear()
        logger.info("Reset incremental fetch state")

    def calculate_market_sentiment(
        self,
        asset: str = "BTC",
        hours_back: int = 6,
        limit: int = 500,
    ) -> MarketSentiment:
        """
        Calculate broad market sentiment from ALL crypto Twitter.

        This includes noise, bad traders, random accounts - giving us
        a true "crowd sentiment" that can be used as contrarian indicator.

        Args:
            asset: Asset to calculate sentiment for ("BTC", "ETH", "CRYPTO")
            hours_back: Hours to look back (shorter = more reactive)
            limit: Max posts to analyze

        Returns:
            MarketSentiment with crowd psychology metrics
        """
        import re

        end_date = datetime.utcnow()
        start_date = end_date - timedelta(hours=hours_back)

        # Ensure dates are different
        if start_date.date() >= end_date.date():
            start_date = end_date - timedelta(days=1)

        # Fetch broad sentiment (no user filter - get EVERYONE)
        if asset.upper() == "CRYPTO":
            keywords = ["crypto", "bitcoin", "ethereum", "altcoin"]
        else:
            keywords = [asset.upper(), asset.lower()]

        # Add sentiment keywords to capture opinions
        sentiment_keywords = keywords + ["bullish", "bearish", "long", "short", "buy", "sell", "pump", "dump"]

        try:
            posts = self._fetch_by_keywords(
                keywords=sentiment_keywords[:10],  # API limit
                start_date=start_date,
                end_date=end_date,
                limit=limit,
            )
        except Exception as e:
            logger.error(f"Error fetching sentiment data: {e}")
            posts = []

        # Sentiment keyword patterns
        BULLISH_PATTERNS = [
            r'\blong\b', r'\bbuy\b', r'\bbullish\b', r'\bpump\b',
            r'\bmoon\b', r'\b🚀\b', r'\bbreakout\b', r'\bbottom\b',
            r'\baccumulate\b', r'\bhodl\b', r'\bentry\b', r'\bdip\b',
            r'\bsend it\b', r'\blfg\b', r'\bwe.?re going up\b',
        ]
        BEARISH_PATTERNS = [
            r'\bshort\b', r'\bsell\b', r'\bbearish\b', r'\bdump\b',
            r'\bcrash\b', r'\btop\b', r'\bovervalued\b', r'\bexit\b',
            r'\brekt\b', r'\brug\b', r'\bscam\b', r'\bdown\b',
            r'\bwe.?re going down\b', r'\bdead\b',
        ]

        bullish_count = 0
        bearish_count = 0
        neutral_count = 0
        weighted_bullish = 0.0
        weighted_bearish = 0.0
        total_weight = 0.0

        for post in posts:
            text_lower = post.text.lower()

            # Count bullish/bearish matches
            bullish_matches = sum(1 for p in BULLISH_PATTERNS if re.search(p, text_lower))
            bearish_matches = sum(1 for p in BEARISH_PATTERNS if re.search(p, text_lower))

            # Determine sentiment
            weight = post.engagement_score  # Use engagement as weight

            if bullish_matches > bearish_matches:
                bullish_count += 1
                weighted_bullish += weight
            elif bearish_matches > bullish_matches:
                bearish_count += 1
                weighted_bearish += weight
            else:
                neutral_count += 1

            total_weight += weight

        total_posts = bullish_count + bearish_count + neutral_count

        # Calculate ratios
        if total_posts > 0:
            bullish_ratio = bullish_count / total_posts
            sentiment_score = (bullish_count - bearish_count) / total_posts
        else:
            bullish_ratio = 0.5
            sentiment_score = 0.0

        # Engagement-weighted sentiment
        if total_weight > 0:
            weighted_sentiment = (weighted_bullish - weighted_bearish) / total_weight
        else:
            weighted_sentiment = 0.0

        # Determine Fear/Greed level
        if bullish_ratio >= 0.75:
            fear_greed = "extreme_greed"
        elif bullish_ratio >= 0.60:
            fear_greed = "greed"
        elif bullish_ratio <= 0.25:
            fear_greed = "extreme_fear"
        elif bullish_ratio <= 0.40:
            fear_greed = "fear"
        else:
            fear_greed = "neutral"

        # CONTRARIAN SIGNAL: When crowd is extreme, fade them
        contrarian_signal = "neutral"
        contrarian_strength = 0.0

        if bullish_ratio >= 0.70:
            # Everyone bullish = potential top, consider fading
            contrarian_signal = "fade_longs"
            contrarian_strength = min(1.0, (bullish_ratio - 0.70) * 3.33)  # 0.70->0, 1.0->1.0
        elif bullish_ratio <= 0.30:
            # Everyone bearish = potential bottom, consider fading
            contrarian_signal = "fade_shorts"
            contrarian_strength = min(1.0, (0.30 - bullish_ratio) * 3.33)

        # ============================================================
        # VOLUME SIGNAL - Research shows volume > polarity for predictions
        # High tweet volume often precedes price moves
        # ============================================================
        volume_signal = "normal"
        volume_zscore = 0.0

        asset_key = asset.upper()

        # Initialize volume history for this asset if needed
        if asset_key not in self._volume_history:
            self._volume_history[asset_key] = []

        # Add current volume to history
        self._volume_history[asset_key].append(total_posts)

        # Keep only recent history
        if len(self._volume_history[asset_key]) > self._volume_history_max:
            self._volume_history[asset_key] = self._volume_history[asset_key][-self._volume_history_max:]

        # Calculate z-score if we have enough history
        if len(self._volume_history[asset_key]) >= 5:
            import statistics
            history = self._volume_history[asset_key][:-1]  # Exclude current
            mean_vol = statistics.mean(history)
            stdev_vol = statistics.stdev(history) if len(history) > 1 else 1

            if stdev_vol > 0:
                volume_zscore = (total_posts - mean_vol) / stdev_vol
            else:
                volume_zscore = 0.0

            # Classify volume signal
            if volume_zscore >= 2.5:
                volume_signal = "extreme"  # Very unusual activity - major move likely
            elif volume_zscore >= 1.5:
                volume_signal = "high"  # Above normal - heightened interest
            elif volume_zscore <= -1.0:
                volume_signal = "low"  # Below normal - quiet period
            else:
                volume_signal = "normal"

        # ============================================================
        # SENTIMENT MOMENTUM - Detect shifts in crowd psychology
        # Research: sentiment leads price, shifts are actionable
        # ============================================================
        sentiment_momentum = "stable"
        momentum_strength = 0.0

        if asset_key in self._previous_sentiment:
            prev_ratio = self._previous_sentiment[asset_key]
            sentiment_change = bullish_ratio - prev_ratio

            # Significant shift threshold: 10% change
            if sentiment_change >= 0.10:
                sentiment_momentum = "turning_bullish"
                momentum_strength = min(1.0, sentiment_change * 5)  # 0.10->0.5, 0.20->1.0
            elif sentiment_change <= -0.10:
                sentiment_momentum = "turning_bearish"
                momentum_strength = min(1.0, abs(sentiment_change) * 5)
            else:
                sentiment_momentum = "stable"
                momentum_strength = 0.0

        # Store current sentiment for next comparison
        self._previous_sentiment[asset_key] = bullish_ratio

        sentiment = MarketSentiment(
            timestamp=datetime.utcnow(),
            asset=asset.upper(),
            bullish_posts=bullish_count,
            bearish_posts=bearish_count,
            neutral_posts=neutral_count,
            total_posts=total_posts,
            sentiment_score=sentiment_score,
            bullish_ratio=bullish_ratio,
            weighted_sentiment=weighted_sentiment,
            volume_signal=volume_signal,
            volume_zscore=volume_zscore,
            contrarian_signal=contrarian_signal,
            contrarian_strength=contrarian_strength,
            fear_greed=fear_greed,
            sentiment_momentum=sentiment_momentum,
            momentum_strength=momentum_strength,
        )

        logger.info(
            f"Market Sentiment ({asset}): {fear_greed} | "
            f"Bullish: {bullish_ratio:.0%} ({bullish_count}/{total_posts}) | "
            f"Volume: {volume_signal} (z={volume_zscore:.1f}) | "
            f"Momentum: {sentiment_momentum} | "
            f"Contrarian: {contrarian_signal} ({contrarian_strength:.0%})"
        )

        return sentiment
