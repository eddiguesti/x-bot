"""Application configuration."""

from pathlib import Path
from pydantic_settings import BaseSettings
from pydantic import Field


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # Macrocosmos API
    macrocosmos_api_key: str = Field(default="", alias="MACROCOSMOS_API_KEY")

    # Apify API (backup)
    apify_api_token: str = Field(default="", alias="APIFY_API_TOKEN")

    # Data source preference: "macrocosmos", "apify", or "auto"
    x_data_source: str = Field(default="auto", alias="X_DATA_SOURCE")

    # Binance (optional - for real trading)
    binance_api_key: str = Field(default="", alias="BINANCE_API_KEY")
    binance_secret_key: str = Field(default="", alias="BINANCE_SECRET_KEY")

    # Binance Testnet (for paper trading)
    # Get keys from: https://testnet.binance.vision/
    binance_testnet_api_key: str = Field(default="", alias="BINANCE_TESTNET_API_KEY")
    binance_testnet_secret: str = Field(default="", alias="BINANCE_TESTNET_SECRET")

    # Telegram notifications (optional)
    telegram_bot_token: str = Field(default="", alias="TELEGRAM_BOT_TOKEN")
    telegram_chat_id: str = Field(default="", alias="TELEGRAM_CHAT_ID")

    # LLM APIs for signal extraction
    # Primary: Gemini Flash (cheapest + reliable)
    gemini_api_key: str = Field(default="", alias="GEMINI_API_KEY")
    # Fallback: DeepSeek (uses OpenAI-compatible API)
    deepseek_api_key: str = Field(default="", alias="DEEPSEEK_API_KEY")
    # LLM provider preference: "gemini", "deepseek", or "auto" (tries gemini first)
    llm_provider: str = Field(default="auto", alias="LLM_PROVIDER")

    # Database
    database_url: str = Field(default="sqlite:///data/consensus.db", alias="DATABASE_URL")

    # Application settings
    log_level: str = Field(default="INFO", alias="LOG_LEVEL")
    evaluation_horizon_hours: int = Field(default=24, alias="EVALUATION_HORIZON_HOURS")
    consensus_threshold: float = Field(default=0.3, alias="CONSENSUS_THRESHOLD")

    # Evaluation thresholds
    price_move_threshold_percent: float = 0.5  # Minimum % move to count as correct

    # Creator settings
    min_predictions_for_ranking: int = 10  # Minimum predictions before ranked
    initial_rating: float = 1500.0  # Starting Glicko-2 rating
    initial_rd: float = 350.0  # Starting rating deviation

    # Consensus settings
    strong_signal_threshold: float = 0.6
    weak_signal_threshold: float = 0.3

    # Paths
    base_dir: Path = Path(__file__).parent.parent
    data_dir: Path = base_dir / "data"
    raw_data_dir: Path = data_dir / "raw"
    processed_data_dir: Path = data_dir / "processed"
    reports_dir: Path = base_dir / "reports"

    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "extra": "ignore",
        "populate_by_name": True,
    }


def get_settings() -> Settings:
    """Get application settings."""
    return Settings()
