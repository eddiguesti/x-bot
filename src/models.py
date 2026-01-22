"""Data models for the consensus system."""

from datetime import datetime
from enum import Enum
from typing import Optional
from pydantic import BaseModel, Field

# Register libsql dialect for Turso support (with auth_token fix)
try:
    from sqlalchemy_libsql.libsql import SQLiteDialect_libsql, _build_connection_url
    from sqlalchemy.dialects import registry
    from sqlalchemy import util

    class TursoDialect(SQLiteDialect_libsql):
        """Fixed dialect that properly handles auth_token for Turso."""

        def create_connect_args(self, url):
            pysqlite_args = (
                ("uri", bool),
                ("timeout", float),
                ("isolation_level", str),
                ("detect_types", int),
                ("check_same_thread", bool),
                ("cached_statements", int),
                ("secure", bool),
                ("auth_token", str),  # Add auth_token support
            )
            opts = dict(url.query)
            libsql_opts = {}
            for key, type_ in pysqlite_args:
                util.coerce_kw_type(opts, key, type_, dest=libsql_opts)

            if url.host:
                libsql_opts["uri"] = True

            # Extract auth_token before building URL
            auth_token = libsql_opts.pop("auth_token", None) or opts.pop("authToken", None)

            if libsql_opts.get("uri", False):
                uri_opts = dict(opts)
                for key, type_ in pysqlite_args:
                    uri_opts.pop(key, None)
                uri_opts.pop("authToken", None)  # Remove from URL params

                secure = libsql_opts.pop("secure", True)  # Default to secure
                connect_url = _build_connection_url(url, uri_opts, secure)
            else:
                import os
                connect_url = url.database or ":memory:"
                if connect_url != ":memory:":
                    connect_url = os.path.abspath(connect_url)

            libsql_opts.setdefault("check_same_thread", not self._is_url_file_db(url))

            # Add auth_token to opts if present
            if auth_token:
                libsql_opts["auth_token"] = auth_token

            return ([connect_url], libsql_opts)

    registry.register('libsql', 'src.models', 'TursoDialect')
except ImportError:
    pass  # Not using Turso

from sqlalchemy import (
    Column,
    Integer,
    String,
    Float,
    DateTime,
    Boolean,
    Enum as SQLEnum,
    Text,
    ForeignKey,
    create_engine,
)
from sqlalchemy.orm import declarative_base, relationship, sessionmaker

Base = declarative_base()


# Enums
class Direction(str, Enum):
    """Trading direction."""
    LONG = "long"
    SHORT = "short"


class Asset(str, Enum):
    """Supported assets."""
    BTC = "BTC"
    ETH = "ETH"
    SOL = "SOL"
    XRP = "XRP"
    DOGE = "DOGE"
    ADA = "ADA"
    AVAX = "AVAX"
    LINK = "LINK"
    DOT = "DOT"
    SHIB = "SHIB"
    LTC = "LTC"
    UNI = "UNI"
    ATOM = "ATOM"
    ARB = "ARB"
    OP = "OP"
    APT = "APT"
    NEAR = "NEAR"
    INJ = "INJ"
    TAO = "TAO"


class SignalOutcome(str, Enum):
    """Signal evaluation outcome."""
    PENDING = "pending"
    CORRECT = "correct"
    INCORRECT = "incorrect"


class ConsensusAction(str, Enum):
    """Consensus recommended action."""
    LONG = "long"
    SHORT = "short"
    NO_TRADE = "no_trade"


# SQLAlchemy Models
class Creator(Base):
    """Crypto content creator being tracked."""
    __tablename__ = "creators"

    id = Column(Integer, primary_key=True)
    username = Column(String(100), unique=True, nullable=False, index=True)
    display_name = Column(String(200))
    added_at = Column(DateTime, default=datetime.utcnow)
    is_active = Column(Boolean, default=True)

    # Glicko-2 rating components
    rating = Column(Float, default=1500.0)
    rating_deviation = Column(Float, default=350.0)
    volatility = Column(Float, default=0.06)

    # Aggregated stats
    total_predictions = Column(Integer, default=0)
    correct_predictions = Column(Integer, default=0)
    last_prediction_at = Column(DateTime)

    # Relationships
    signals = relationship("Signal", back_populates="creator")

    @property
    def accuracy(self) -> float:
        """Calculate accuracy percentage."""
        if self.total_predictions == 0:
            return 0.0
        return self.correct_predictions / self.total_predictions

    @property
    def weight(self) -> float:
        """Calculate creator weight for consensus (higher rating = higher weight)."""
        # Normalize rating to 0-1 scale, with 1500 as baseline
        # Rating 1200 -> ~0.3, Rating 1500 -> 0.5, Rating 1800 -> ~0.7
        base_weight = (self.rating - 1000) / 1000
        # Reduce weight for high uncertainty (high RD)
        confidence = max(0.1, 1 - (self.rating_deviation / 350))
        return max(0.01, base_weight * confidence)


class Signal(Base):
    """A trading signal extracted from a creator's post."""
    __tablename__ = "signals"

    id = Column(Integer, primary_key=True)
    creator_id = Column(Integer, ForeignKey("creators.id"), nullable=False)

    # Post data
    post_id = Column(String(100), unique=True, nullable=False, index=True)
    post_text = Column(Text, nullable=False)
    post_url = Column(String(500))
    posted_at = Column(DateTime, nullable=False, index=True)

    # Extracted signal
    asset = Column(SQLEnum(Asset), nullable=False)
    direction = Column(SQLEnum(Direction), nullable=False)
    confidence = Column(Float, default=1.0)  # NLP extraction confidence

    # Evaluation
    outcome = Column(SQLEnum(SignalOutcome), default=SignalOutcome.PENDING)
    price_at_signal = Column(Float)
    price_at_evaluation = Column(Float)
    price_change_percent = Column(Float)
    evaluated_at = Column(DateTime)
    evaluation_horizon_hours = Column(Integer, default=24)

    # Metadata
    created_at = Column(DateTime, default=datetime.utcnow)
    raw_data = Column(Text)  # JSON of original API response

    # Relationships
    creator = relationship("Creator", back_populates="signals")


class PriceSnapshot(Base):
    """Historical price data for evaluation."""
    __tablename__ = "price_snapshots"

    id = Column(Integer, primary_key=True)
    asset = Column(SQLEnum(Asset), nullable=False, index=True)
    timestamp = Column(DateTime, nullable=False, index=True)
    price = Column(Float, nullable=False)
    source = Column(String(50), default="binance")

    class Meta:
        unique_together = ("asset", "timestamp", "source")


class ConsensusSnapshot(Base):
    """Historical consensus state."""
    __tablename__ = "consensus_snapshots"

    id = Column(Integer, primary_key=True)
    asset = Column(SQLEnum(Asset), nullable=False, index=True)
    timestamp = Column(DateTime, default=datetime.utcnow, index=True)

    # Consensus scores
    raw_score = Column(Float)  # -1 to +1
    weighted_score = Column(Float)  # -1 to +1

    # Vote counts
    long_votes = Column(Integer, default=0)
    short_votes = Column(Integer, default=0)
    total_weight_long = Column(Float, default=0.0)
    total_weight_short = Column(Float, default=0.0)

    # Decision
    action = Column(SQLEnum(ConsensusAction))
    confidence = Column(Float)

    # For audit
    contributing_signals = Column(Text)  # JSON list of signal IDs


class TradeStatus(str, Enum):
    """Trade status."""
    OPEN = "open"
    CLOSED = "closed"


class Trade(Base):
    """Paper trade record."""
    __tablename__ = "trades"

    id = Column(Integer, primary_key=True)
    asset = Column(SQLEnum(Asset), nullable=False, index=True)
    direction = Column(SQLEnum(Direction), nullable=False)
    status = Column(SQLEnum(TradeStatus), default=TradeStatus.OPEN)

    # Entry
    entry_price = Column(Float, nullable=False)
    entry_time = Column(DateTime, default=datetime.utcnow)
    position_size = Column(Float, default=1.0)  # In units of asset

    # Exit (when closed)
    exit_price = Column(Float)
    exit_time = Column(DateTime)

    # PnL
    pnl_percent = Column(Float)
    pnl_usd = Column(Float)

    # Consensus info at entry
    consensus_score = Column(Float)
    consensus_confidence = Column(Float)
    contributing_signals = Column(Text)  # JSON

    # Stop loss / take profit
    stop_loss_price = Column(Float)
    take_profit_price = Column(Float)

    # Metadata
    notes = Column(String(500))  # For tagging trades with strategy, etc.

    created_at = Column(DateTime, default=datetime.utcnow)


class Portfolio(Base):
    """Paper trading portfolio state."""
    __tablename__ = "portfolio"

    id = Column(Integer, primary_key=True)
    name = Column(String(100), default="default")

    # Balance
    initial_balance = Column(Float, default=10000.0)
    current_balance = Column(Float, default=10000.0)

    # Stats
    total_trades = Column(Integer, default=0)
    winning_trades = Column(Integer, default=0)
    total_pnl = Column(Float, default=0.0)

    # Risk settings - CONSERVATIVE
    max_position_pct = Column(Float, default=0.02)  # Max 2% per trade
    stop_loss_pct = Column(Float, default=0.02)  # 2% stop loss
    take_profit_pct = Column(Float, default=0.04)  # 4% take profit (2:1 R:R)

    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


# Pydantic Models for API/Data Transfer
class CreatorStats(BaseModel):
    """Creator statistics for reports."""
    username: str
    display_name: Optional[str]
    rating: float
    rating_deviation: float
    accuracy: float
    total_predictions: int
    correct_predictions: int
    weight: float
    rank: int


class ConsensusResult(BaseModel):
    """Current consensus state."""
    asset: Asset
    timestamp: datetime
    action: ConsensusAction
    confidence: float
    weighted_score: float
    long_votes: int
    short_votes: int
    top_contributors: list[str] = Field(default_factory=list)


class SignalExtraction(BaseModel):
    """Extracted signal from text."""
    asset: Optional[Asset]
    direction: Optional[Direction]
    confidence: float
    reasoning: str


# Database initialization
def init_db(database_url: str):
    """Initialize database and create tables."""
    engine = create_engine(database_url)
    Base.metadata.create_all(engine)
    return engine


def get_session(engine):
    """Get database session."""
    Session = sessionmaker(bind=engine)
    return Session()
