"""Application constants - centralized magic numbers and limits."""

# =============================================================================
# DATABASE LIMITS
# =============================================================================
MAX_POST_TEXT_LENGTH = 5000  # Maximum text length stored in Signal.post_text
MAX_USERNAME_LENGTH = 100
MAX_URL_LENGTH = 500

# =============================================================================
# LLM EXTRACTION
# =============================================================================
MAX_LLM_INPUT_LENGTH = 8000  # Gemini Flash context limit for input
LLM_MAX_RETRIES = 3
LLM_RETRY_BASE_DELAY = 2.0  # Seconds, doubles each retry
LLM_TEMPERATURE = 0.1  # Low temperature for consistent extractions
LLM_MAX_OUTPUT_TOKENS = 256

# =============================================================================
# SIGNAL COLLECTION
# =============================================================================
DEFAULT_LOOKBACK_HOURS_TWITTER = 24
DEFAULT_LOOKBACK_HOURS_REDDIT = 24
DEFAULT_LOOKBACK_HOURS_YOUTUBE = 72  # YouTube content is less frequent
DEFAULT_LIMIT_TWITTER = 200
DEFAULT_LIMIT_REDDIT = 500
DEFAULT_LIMIT_YOUTUBE = 100
MAX_YOUTUBE_TRANSCRIPTS = 20  # Limit transcripts to avoid rate limits

# =============================================================================
# CACHE LIMITS
# =============================================================================
# YouTube (lower volume)
SEEN_POSTS_CACHE_MAX_SIZE = 5000
SEEN_POSTS_CACHE_TRIM_SIZE = 2500  # Trim to this size when max reached
# Twitter/Reddit (higher volume)
SEEN_POSTS_CACHE_MAX_SIZE_HIGH_VOLUME = 10000
SEEN_POSTS_CACHE_TRIM_SIZE_HIGH_VOLUME = 5000

# =============================================================================
# TRADING LIMITS
# =============================================================================
MAX_POSITION_AGE_HOURS = 72  # Close stale positions after 3 days
MIN_PROFIT_FOR_TIME_EXIT = 0.02  # 2% minimum profit to keep stale position
DEFAULT_TRADING_INTERVAL_MINUTES = 30

# =============================================================================
# CONFIDENCE THRESHOLDS
# =============================================================================
LLM_MIN_CONFIDENCE_THRESHOLD = 0.5
RULE_BASED_MIN_CONFIDENCE_THRESHOLD = 0.4

# =============================================================================
# RATE LIMITING
# =============================================================================
API_MIN_REQUEST_INTERVAL = 1.0  # Minimum seconds between API calls
API_MAX_RETRIES = 3
API_RETRY_BASE_DELAY = 2.0

# =============================================================================
# TRANSCRIPT SETTINGS
# =============================================================================
TRANSCRIPT_LANGUAGES = ['en', 'en-US', 'en-GB']
