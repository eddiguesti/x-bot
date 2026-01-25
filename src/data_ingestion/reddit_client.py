"""Reddit client for fetching crypto trading signals via Macrocosmos.

Uses the same Macrocosmos SN13 API as X/Twitter client, but with source='Reddit'.
This provides unified data access without needing separate Reddit API credentials.

Features:
1. Fetch posts from crypto subreddits via Macrocosmos
2. Extract trading signals from posts
3. Engagement weighting (upvotes boost signal importance)
4. Rate limiting and retry logic
"""

import json
import logging
import math
import time
from collections import OrderedDict
from datetime import datetime, timedelta
from typing import Optional
from dataclasses import dataclass, field

import macrocosmos as mc

from ..config import Settings
from ..constants import SEEN_POSTS_CACHE_MAX_SIZE_HIGH_VOLUME, SEEN_POSTS_CACHE_TRIM_SIZE_HIGH_VOLUME

logger = logging.getLogger(__name__)


@dataclass
class RedditPost:
    """Parsed Reddit post data with engagement metrics."""
    post_id: str
    username: str
    text: str
    posted_at: datetime
    subreddit: str = ""
    title: str = ""
    url: Optional[str] = None
    # Engagement metrics
    upvotes: int = 0
    num_comments: int = 0
    engagement_score: float = 1.0


class RedditClient:
    """Client for fetching Reddit posts via Macrocosmos SN13 API.

    Uses the same API as XClient but with source='Reddit'.
    No separate Reddit API credentials needed.
    """

    # Crypto-related keywords for Reddit search (expanded coverage)
    CRYPTO_KEYWORDS = [
        # ========== MAJOR COINS ==========
        "bitcoin", "btc", "ethereum", "eth", "solana", "sol",
        "cardano", "ada", "ripple", "xrp", "dogecoin", "doge",
        "avalanche", "avax", "polkadot", "dot", "chainlink", "link",
        "litecoin", "ltc", "bnb", "binance coin",

        # ========== TRADING TERMS ==========
        "crypto trading", "long btc", "short btc", "bullish crypto",
        "bearish crypto", "crypto buy", "crypto sell", "altcoin",
        "altseason", "alt season", "bitcoin dominance", "btc dominance",
        "crypto portfolio", "crypto gains", "crypto losses",

        # ========== DEFI ==========
        "defi", "uniswap", "aave", "yield farming", "liquidity",
        "curve finance", "maker dao", "lido", "rocket pool",
        "compound", "sushiswap", "pancakeswap", "gmx", "dydx",
        "hyperliquid", "pendle", "eigenlayer", "restaking",

        # ========== MEMECOINS ==========
        "shib", "pepe", "bonk", "wif", "memecoin", "meme coin",
        "floki", "degen", "brett", "popcat", "mog", "trump coin",
        "pnut", "goat", "act", "fartcoin", "ai16z",

        # ========== L2s AND NEW CHAINS ==========
        "arbitrum", "optimism", "base", "sui", "sei", "celestia",
        "starknet", "zksync", "scroll", "linea", "manta", "blast",
        "mantle", "mode", "zora", "taiko",

        # ========== AI & AGENTS ==========
        "bittensor", "tao", "fetch ai", "render", "worldcoin",
        "ocean protocol", "singularitynet", "akash", "nosana",
        "ai agent", "ai crypto", "virtuals protocol", "zerebro",
        "truth terminal", "aixbt", "griffain",

        # ========== SOLANA ECOSYSTEM ==========
        "solana defi", "jupiter exchange", "raydium", "orca",
        "marinade", "jito", "tensor", "magic eden solana",
        "pyth network", "wormhole",

        # ========== GAMING & NFT ==========
        "crypto gaming", "gamefi", "play to earn", "axie",
        "immutable", "gala", "sandbox", "decentraland",

        # ========== INFRASTRUCTURE ==========
        "layer zero", "cosmos", "atom", "thorchain",
        "injective", "sei network", "aptos", "near protocol",

        # ========== NARRATIVES ==========
        "rwa", "real world assets", "tokenization",
        "depin", "decentralized physical", "helium", "hivemapper",
        "modular blockchain", "data availability",
    ]

    # Expanded trading signal keywords
    TRADING_KEYWORDS = [
        # Direct signals
        "long", "short", "buy", "sell", "hodl", "hold",
        "bullish", "bearish", "moon", "dump", "pump",
        "accumulate", "dca", "dollar cost",

        # Technical analysis
        "breakout", "breakdown", "support", "resistance",
        "entry", "exit", "target", "stop loss", "take profit",
        "oversold", "overbought", "reversal", "bounce",
        "golden cross", "death cross", "divergence",

        # Sentiment
        "ath", "all time high", "dip", "buying the dip",
        "fomo", "fud", "undervalued", "overvalued",
        "gem", "100x", "10x", "moonshot",

        # Chart patterns
        "head and shoulders", "double top", "double bottom",
        "cup and handle", "wedge", "triangle", "flag",

        # Fundamentals
        "tvl", "market cap", "tokenomics", "airdrop",
        "burn", "buyback", "staking",
    ]

    # Asset keywords for filtering (expanded)
    ASSET_KEYWORDS = {
        # ========== MAJOR COINS ==========
        "BTC": ["btc", "bitcoin", "sats"],
        "ETH": ["eth", "ethereum", "ether"],
        "SOL": ["sol", "solana"],
        "XRP": ["xrp", "ripple"],
        "DOGE": ["doge", "dogecoin"],
        "ADA": ["ada", "cardano"],
        "AVAX": ["avax", "avalanche"],
        "LINK": ["link", "chainlink"],
        "DOT": ["dot", "polkadot"],
        "LTC": ["ltc", "litecoin"],
        "BNB": ["bnb", "binance"],

        # ========== MEMECOINS ==========
        "SHIB": ["shib", "shiba"],
        "PEPE": ["pepe"],
        "WIF": ["wif", "dogwifhat"],
        "BONK": ["bonk"],
        "FLOKI": ["floki"],

        # ========== L2s & NEW CHAINS ==========
        "ARB": ["arb", "arbitrum"],
        "OP": ["op", "optimism"],
        "MATIC": ["matic", "polygon"],
        "SUI": ["sui"],
        "SEI": ["sei"],
        "APT": ["apt", "aptos"],
        "NEAR": ["near"],
        "TIA": ["tia", "celestia"],
        "INJ": ["inj", "injective"],
        "FTM": ["ftm", "fantom"],

        # ========== DEFI ==========
        "UNI": ["uni", "uniswap"],
        "AAVE": ["aave"],
        "MKR": ["mkr", "maker"],
        "CRV": ["crv", "curve"],
        "LDO": ["ldo", "lido"],
        "PENDLE": ["pendle"],
        "GMX": ["gmx"],
        "DYDX": ["dydx"],

        # ========== AI & DATA ==========
        "TAO": ["tao", "bittensor"],
        "FET": ["fet", "fetch"],
        "RNDR": ["rndr", "render"],
        "GRT": ["grt", "graph"],
        "OCEAN": ["ocean"],
        "AGIX": ["agix", "singularity"],
        "AKT": ["akt", "akash"],
        "WLD": ["wld", "worldcoin"],

        # ========== SOLANA ECOSYSTEM ==========
        "JUP": ["jup", "jupiter"],
        "JTO": ["jto", "jito"],
        "PYTH": ["pyth"],
        "RAY": ["ray", "raydium"],
        "ORCA": ["orca"],
        "MNDE": ["mnde", "marinade"],

        # ========== INFRASTRUCTURE ==========
        "ATOM": ["atom", "cosmos"],
        "HBAR": ["hbar", "hedera"],
        "QNT": ["qnt", "quant"],
        "VET": ["vet", "vechain"],
        "FIL": ["fil", "filecoin"],
        "AR": ["ar", "arweave"],
        "STX": ["stx", "stacks"],

        # ========== PRIVACY ==========
        "XMR": ["xmr", "monero"],
        "ZEC": ["zec", "zcash"],

        # ========== GAMING ==========
        "IMX": ["imx", "immutable"],
        "GALA": ["gala"],
        "AXS": ["axs", "axie"],
        "SAND": ["sand", "sandbox"],
        "MANA": ["mana", "decentraland"],
    }

    # Rate limiting
    MIN_REQUEST_INTERVAL = 1.0
    MAX_RETRIES = 3
    RETRY_BASE_DELAY = 2.0

    def __init__(self, settings: Settings):
        self.settings = settings
        self._last_api_call: float = 0
        self._seen_post_ids: OrderedDict[str, None] = OrderedDict()

        # Initialize Macrocosmos client (same as XClient)
        import os
        api_key = settings.macrocosmos_api_key or os.getenv("MACROCOSMOS_API_KEY", "")

        if api_key:
            self.client = mc.Sn13Client(
                api_key=api_key,
                app_name="crypto_consensus_reddit"
            )
            self.enabled = True
            self.raw_data_dir = settings.raw_data_dir
            self.raw_data_dir.mkdir(parents=True, exist_ok=True)
            logger.info("Reddit client initialized via Macrocosmos")
        else:
            self.client = None
            self.enabled = False
            logger.warning("Macrocosmos API key not provided - Reddit client disabled")

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
                        f"Reddit API call failed (attempt {attempt + 1}/{self.MAX_RETRIES}): {e}. "
                        f"Retrying in {delay:.1f}s..."
                    )
                    time.sleep(delay)
                else:
                    logger.error(f"Reddit API call failed after {self.MAX_RETRIES} attempts: {e}")

        raise last_exception if last_exception else Exception("Unknown error")

    def _calculate_engagement_score(self, upvotes: int, comments: int) -> float:
        """Calculate engagement score for a post."""
        if upvotes == 0 and comments == 0:
            return 1.0

        # Logarithmic scaling
        upvote_score = math.log1p(upvotes) * 1.5
        comment_score = math.log1p(comments) * 1.0

        return min(3.0, max(1.0, 1.0 + (upvote_score + comment_score) / 10))

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

    def _parse_response(self, response) -> list[RedditPost]:
        """Parse Macrocosmos API response into RedditPost objects."""
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
                post_id = str(item.get('id', item.get('post_id', '')))
                if post_id in self._seen_post_ids:
                    continue
                self._seen_post_ids[post_id] = None

                # Get text content
                text = item.get('text', item.get('content', item.get('body', '')))
                title = item.get('title', '')
                if title:
                    text = f"{title}\n{text}"

                if not text:
                    continue

                # Check relevance
                if not self._has_asset_mention(text) or not self._has_trading_signal(text):
                    continue

                # Get username
                username = item.get('author', item.get('username', 'unknown'))

                # Get subreddit
                subreddit = item.get('subreddit', item.get('community', ''))

                # Get URL
                url = item.get('url', item.get('uri', item.get('permalink', '')))

                # Parse timestamp
                timestamp = item.get('datetime', item.get('created_at', item.get('timestamp')))
                posted_at = self._parse_timestamp(timestamp)

                # Engagement metrics
                upvotes = int(item.get('score', item.get('upvotes', item.get('ups', 0))) or 0)
                comments = int(item.get('num_comments', item.get('comments', 0)) or 0)

                post = RedditPost(
                    post_id=post_id,
                    username=username,
                    text=text,
                    title=title,
                    posted_at=posted_at,
                    subreddit=subreddit,
                    url=url,
                    upvotes=upvotes,
                    num_comments=comments,
                    engagement_score=self._calculate_engagement_score(upvotes, comments),
                )
                posts.append(post)

            except Exception as e:
                logger.warning(f"Error parsing Reddit post: {e}")
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
        filename = f"reddit_posts_{timestamp}_{context[:30]}.json"
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

            logger.debug(f"Saved raw Reddit response to {filepath}")
        except Exception as e:
            logger.warning(f"Could not save raw Reddit response: {e}")

    def fetch_trading_signals(
        self,
        limit: int = 500,
        hours_back: int = 24,
    ) -> list[RedditPost]:
        """
        Fetch trading signals from Reddit via Macrocosmos.

        Args:
            limit: Maximum posts to fetch
            hours_back: Only include posts from last N hours

        Returns:
            List of RedditPost objects with trading signals, sorted by engagement
        """
        if not self.enabled:
            logger.warning("Reddit client not enabled - skipping")
            return []

        logger.info(f"Fetching Reddit signals via Macrocosmos (last {hours_back}h)...")

        end_date = datetime.utcnow()
        start_date = end_date - timedelta(hours=max(hours_back, 24))

        # Ensure dates are different (API requirement)
        if start_date.date() >= end_date.date():
            start_date = end_date - timedelta(days=1)

        all_posts = []

        # Fetch with crypto keywords
        try:
            # Use BTC/ETH focused keywords first (highest volume)
            btc_keywords = [
                "bitcoin", "btc", "long btc", "short btc",
                "bullish bitcoin", "bearish bitcoin", "buy bitcoin"
            ]

            def _api_call_btc():
                return self.client.sn13.OnDemandData(
                    source='Reddit',
                    keywords=btc_keywords,
                    start_date=start_date.strftime('%Y-%m-%d'),
                    end_date=end_date.strftime('%Y-%m-%d'),
                    limit=limit // 3,
                    keyword_mode='any'
                )

            response = self._retry_with_backoff(_api_call_btc)
            self._save_raw_response(response, "btc_keywords")
            posts = self._parse_response(response)
            all_posts.extend(posts)
            logger.info(f"Reddit BTC keywords: {len(posts)} relevant posts")

        except Exception as e:
            logger.error(f"Error fetching Reddit BTC posts: {e}")

        # Fetch altcoin signals
        try:
            alt_keywords = [
                "ethereum", "solana", "cardano", "altcoin",
                "defi", "long eth", "bullish sol"
            ]

            def _api_call_alt():
                return self.client.sn13.OnDemandData(
                    source='Reddit',
                    keywords=alt_keywords,
                    start_date=start_date.strftime('%Y-%m-%d'),
                    end_date=end_date.strftime('%Y-%m-%d'),
                    limit=limit // 3,
                    keyword_mode='any'
                )

            response = self._retry_with_backoff(_api_call_alt)
            self._save_raw_response(response, "alt_keywords")
            posts = self._parse_response(response)
            all_posts.extend(posts)
            logger.info(f"Reddit alt keywords: {len(posts)} relevant posts")

        except Exception as e:
            logger.error(f"Error fetching Reddit alt posts: {e}")

        # Fetch meme/degen signals
        try:
            meme_keywords = [
                "memecoin", "shib", "pepe", "doge", "bonk",
                "moon", "pump", "gem", "100x"
            ]

            def _api_call_meme():
                return self.client.sn13.OnDemandData(
                    source='Reddit',
                    keywords=meme_keywords,
                    start_date=start_date.strftime('%Y-%m-%d'),
                    end_date=end_date.strftime('%Y-%m-%d'),
                    limit=limit // 3,
                    keyword_mode='any'
                )

            response = self._retry_with_backoff(_api_call_meme)
            self._save_raw_response(response, "meme_keywords")
            posts = self._parse_response(response)
            all_posts.extend(posts)
            logger.info(f"Reddit meme keywords: {len(posts)} relevant posts")

        except Exception as e:
            logger.error(f"Error fetching Reddit meme posts: {e}")

        # Deduplicate by post_id
        seen_ids = set()
        unique_posts = []
        for post in all_posts:
            if post.post_id not in seen_ids:
                seen_ids.add(post.post_id)
                unique_posts.append(post)

        # Limit seen cache size (OrderedDict maintains insertion order for FIFO eviction)
        if len(self._seen_post_ids) > SEEN_POSTS_CACHE_MAX_SIZE_HIGH_VOLUME:
            items_to_keep = list(self._seen_post_ids.keys())[-SEEN_POSTS_CACHE_TRIM_SIZE_HIGH_VOLUME:]
            self._seen_post_ids = OrderedDict.fromkeys(items_to_keep)

        # Sort by engagement score
        unique_posts.sort(key=lambda x: x.engagement_score, reverse=True)

        logger.info(f"Total unique Reddit signals: {len(unique_posts)}")
        return unique_posts

    def reset_state(self):
        """Reset seen posts cache."""
        self._seen_post_ids.clear()
        logger.info("Reset Reddit client state")
