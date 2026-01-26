"""
Strategy configurations for A/B testing different trading approaches.

Uses factory pattern to reduce duplication across data sources.

THREE PHILOSOPHIES:
1. SOCIAL_PURE - Trust the crowd: social signals only, minimal filters
2. TECHNICAL_STRICT - Trust the charts: heavy technical confirmation
3. BALANCED - Trust both: hybrid approach

FIVE DATA SOURCES:
1. TWITTER - X/Twitter only
2. REDDIT - Reddit only
3. YOUTUBE - YouTube only
4. MIXED - Twitter + Reddit
5. ALL - Twitter + Reddit + YouTube
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional
from copy import deepcopy


class DataSource(str, Enum):
    """Data source for social signals."""
    TWITTER = "twitter"    # X/Twitter only
    REDDIT = "reddit"      # Reddit only
    YOUTUBE = "youtube"    # YouTube only
    MIXED = "mixed"        # Twitter + Reddit
    ALL = "all"            # Twitter + Reddit + YouTube


class StrategyType(str, Enum):
    """Available trading strategies."""
    # Twitter-only strategies
    SOCIAL_PURE = "social_pure"
    TECHNICAL_STRICT = "technical_strict"
    BALANCED = "balanced"

    # Reddit-only strategies
    REDDIT_SOCIAL_PURE = "reddit_social_pure"
    REDDIT_TECHNICAL_STRICT = "reddit_technical_strict"
    REDDIT_BALANCED = "reddit_balanced"

    # Mixed (Twitter + Reddit) strategies
    MIXED_SOCIAL_PURE = "mixed_social_pure"
    MIXED_TECHNICAL_STRICT = "mixed_technical_strict"
    MIXED_BALANCED = "mixed_balanced"

    # YouTube-only strategies
    YOUTUBE_SOCIAL_PURE = "youtube_social_pure"
    YOUTUBE_TECHNICAL_STRICT = "youtube_technical_strict"
    YOUTUBE_BALANCED = "youtube_balanced"

    # All sources (Twitter + Reddit + YouTube) strategies
    ALL_SOCIAL_PURE = "all_social_pure"
    ALL_TECHNICAL_STRICT = "all_technical_strict"
    ALL_BALANCED = "all_balanced"


class Philosophy(str, Enum):
    """Trading philosophy presets."""
    SOCIAL_PURE = "social_pure"
    TECHNICAL_STRICT = "technical_strict"
    BALANCED = "balanced"


@dataclass
class ConsensusConfig:
    """Consensus engine configuration."""
    min_signal_confidence: float = 0.35
    min_creator_accuracy: float = 0.40
    min_signals_for_trade: int = 5
    min_agreement_ratio: float = 0.60
    decay_half_life_hours: float = 8.0
    use_crowd_sentiment: bool = True
    sentiment_weight: float = 0.20


@dataclass
class TechnicalConfig:
    """Technical analysis configuration."""
    require_tech_confirmation: bool = True
    tech_score_threshold: float = -0.2
    allow_counter_trend: bool = True
    counter_trend_rsi_threshold: float = 35.0
    max_volatility_atr: float = 8.0
    rsi_overbought: float = 80.0
    rsi_oversold: float = 20.0


@dataclass
class RiskConfig:
    """Risk management configuration."""
    base_position_pct: float = 0.015
    min_position_pct: float = 0.005
    max_position_pct: float = 0.03
    max_total_exposure: float = 0.20
    min_rr_ratio: float = 1.5
    target_rr_ratio: float = 2.5
    min_conviction: float = 0.40
    max_positions: int = 6
    max_daily_loss_pct: float = 0.025
    max_drawdown_pct: float = 0.08
    use_partial_profits: bool = True
    use_anti_martingale: bool = True


@dataclass
class StrategyConfig:
    """Complete strategy configuration."""
    name: StrategyType
    description: str
    data_source: DataSource = DataSource.TWITTER
    consensus: ConsensusConfig = field(default_factory=ConsensusConfig)
    technical: TechnicalConfig = field(default_factory=TechnicalConfig)
    risk: RiskConfig = field(default_factory=RiskConfig)

    def __str__(self) -> str:
        return f"{self.name.value}: {self.description}"


# =============================================================================
# BASE PRESETS - Define each philosophy's core parameters
# =============================================================================

_SOCIAL_PURE_PRESET = {
    "consensus": ConsensusConfig(
        min_signal_confidence=0.30,
        min_creator_accuracy=0.35,
        min_signals_for_trade=3,       # Base value, adjusted per source
        min_agreement_ratio=0.55,
        decay_half_life_hours=4.0,     # Fast decay
        use_crowd_sentiment=True,
        sentiment_weight=0.30,         # Heavy sentiment
    ),
    "technical": TechnicalConfig(
        require_tech_confirmation=False,
        tech_score_threshold=-0.5,
        allow_counter_trend=True,
        counter_trend_rsi_threshold=30.0,
        max_volatility_atr=10.0,       # Allow higher volatility
        rsi_overbought=85.0,
        rsi_oversold=15.0,
    ),
    "risk": RiskConfig(
        base_position_pct=0.012,
        min_position_pct=0.005,
        max_position_pct=0.025,
        max_total_exposure=0.25,       # More exposure
        min_rr_ratio=1.2,              # Lower R:R ok
        target_rr_ratio=2.0,
        min_conviction=0.30,           # Lower bar
        max_positions=8,               # More concurrent
        max_daily_loss_pct=0.03,
        max_drawdown_pct=0.10,
        use_partial_profits=True,
        use_anti_martingale=True,
    ),
}

_TECHNICAL_STRICT_PRESET = {
    "consensus": ConsensusConfig(
        min_signal_confidence=0.45,
        min_creator_accuracy=0.50,
        min_signals_for_trade=7,       # Base value, adjusted per source
        min_agreement_ratio=0.70,      # Strong consensus
        decay_half_life_hours=12.0,    # Slow decay
        use_crowd_sentiment=True,
        sentiment_weight=0.10,         # Less sentiment
    ),
    "technical": TechnicalConfig(
        require_tech_confirmation=True,
        tech_score_threshold=0.1,      # Must be positive
        allow_counter_trend=False,     # No counter-trend
        counter_trend_rsi_threshold=25.0,
        max_volatility_atr=6.0,        # Avoid high volatility
        rsi_overbought=70.0,
        rsi_oversold=30.0,
    ),
    "risk": RiskConfig(
        base_position_pct=0.02,        # Larger positions
        min_position_pct=0.01,
        max_position_pct=0.04,
        max_total_exposure=0.15,       # Lower exposure
        min_rr_ratio=2.0,              # Strict 2:1
        target_rr_ratio=3.0,
        min_conviction=0.50,           # High conviction
        max_positions=4,               # Fewer concurrent
        max_daily_loss_pct=0.02,
        max_drawdown_pct=0.06,
        use_partial_profits=True,
        use_anti_martingale=True,
    ),
}

_BALANCED_PRESET = {
    "consensus": ConsensusConfig(
        min_signal_confidence=0.35,
        min_creator_accuracy=0.40,
        min_signals_for_trade=5,       # Base value, adjusted per source
        min_agreement_ratio=0.60,
        decay_half_life_hours=8.0,
        use_crowd_sentiment=True,
        sentiment_weight=0.20,
    ),
    "technical": TechnicalConfig(
        require_tech_confirmation=True,
        tech_score_threshold=-0.2,     # "Not strongly against"
        allow_counter_trend=True,
        counter_trend_rsi_threshold=35.0,
        max_volatility_atr=8.0,
        rsi_overbought=80.0,
        rsi_oversold=20.0,
    ),
    "risk": RiskConfig(
        base_position_pct=0.015,
        min_position_pct=0.005,
        max_position_pct=0.03,
        max_total_exposure=0.20,
        min_rr_ratio=1.5,
        target_rr_ratio=2.5,
        min_conviction=0.40,
        max_positions=6,
        max_daily_loss_pct=0.025,
        max_drawdown_pct=0.08,
        use_partial_profits=True,
        use_anti_martingale=True,
    ),
}

_PRESETS = {
    Philosophy.SOCIAL_PURE: _SOCIAL_PURE_PRESET,
    Philosophy.TECHNICAL_STRICT: _TECHNICAL_STRICT_PRESET,
    Philosophy.BALANCED: _BALANCED_PRESET,
}


# =============================================================================
# SOURCE-SPECIFIC ADJUSTMENTS
# =============================================================================

# min_signals_for_trade adjustments based on data volume
_MIN_SIGNALS_ADJUSTMENTS = {
    # (source, philosophy) -> adjustment to base min_signals
    (DataSource.TWITTER, Philosophy.SOCIAL_PURE): 0,
    (DataSource.TWITTER, Philosophy.TECHNICAL_STRICT): 0,
    (DataSource.TWITTER, Philosophy.BALANCED): 0,

    (DataSource.REDDIT, Philosophy.SOCIAL_PURE): 0,
    (DataSource.REDDIT, Philosophy.TECHNICAL_STRICT): 0,
    (DataSource.REDDIT, Philosophy.BALANCED): 0,

    (DataSource.YOUTUBE, Philosophy.SOCIAL_PURE): -1,   # Less content
    (DataSource.YOUTUBE, Philosophy.TECHNICAL_STRICT): -4,  # Much less content (7->3)
    (DataSource.YOUTUBE, Philosophy.BALANCED): -3,   # Less content (5->2)

    (DataSource.MIXED, Philosophy.SOCIAL_PURE): 1,    # More data available
    (DataSource.MIXED, Philosophy.TECHNICAL_STRICT): 1,
    (DataSource.MIXED, Philosophy.BALANCED): 1,

    (DataSource.ALL, Philosophy.SOCIAL_PURE): 2,      # Most data available
    (DataSource.ALL, Philosophy.TECHNICAL_STRICT): 3,
    (DataSource.ALL, Philosophy.BALANCED): 2,
}

# YouTube has slower decay (longer-form content)
_DECAY_ADJUSTMENTS = {
    DataSource.YOUTUBE: 2.0,  # Add 2 hours to decay half-life
}

# Strategy type mapping
_STRATEGY_TYPE_MAP = {
    (DataSource.TWITTER, Philosophy.SOCIAL_PURE): StrategyType.SOCIAL_PURE,
    (DataSource.TWITTER, Philosophy.TECHNICAL_STRICT): StrategyType.TECHNICAL_STRICT,
    (DataSource.TWITTER, Philosophy.BALANCED): StrategyType.BALANCED,

    (DataSource.REDDIT, Philosophy.SOCIAL_PURE): StrategyType.REDDIT_SOCIAL_PURE,
    (DataSource.REDDIT, Philosophy.TECHNICAL_STRICT): StrategyType.REDDIT_TECHNICAL_STRICT,
    (DataSource.REDDIT, Philosophy.BALANCED): StrategyType.REDDIT_BALANCED,

    (DataSource.YOUTUBE, Philosophy.SOCIAL_PURE): StrategyType.YOUTUBE_SOCIAL_PURE,
    (DataSource.YOUTUBE, Philosophy.TECHNICAL_STRICT): StrategyType.YOUTUBE_TECHNICAL_STRICT,
    (DataSource.YOUTUBE, Philosophy.BALANCED): StrategyType.YOUTUBE_BALANCED,

    (DataSource.MIXED, Philosophy.SOCIAL_PURE): StrategyType.MIXED_SOCIAL_PURE,
    (DataSource.MIXED, Philosophy.TECHNICAL_STRICT): StrategyType.MIXED_TECHNICAL_STRICT,
    (DataSource.MIXED, Philosophy.BALANCED): StrategyType.MIXED_BALANCED,

    (DataSource.ALL, Philosophy.SOCIAL_PURE): StrategyType.ALL_SOCIAL_PURE,
    (DataSource.ALL, Philosophy.TECHNICAL_STRICT): StrategyType.ALL_TECHNICAL_STRICT,
    (DataSource.ALL, Philosophy.BALANCED): StrategyType.ALL_BALANCED,
}

# Description templates
_DESCRIPTION_TEMPLATES = {
    Philosophy.SOCIAL_PURE: "{source}: Pure social signals - trust the crowd, minimal filters",
    Philosophy.TECHNICAL_STRICT: "{source}: Technical-heavy - charts must confirm, quality over quantity",
    Philosophy.BALANCED: "{source}: Hybrid approach - social signals with crypto-adapted tech filters",
}

_SOURCE_NAMES = {
    DataSource.TWITTER: "Twitter",
    DataSource.REDDIT: "Reddit",
    DataSource.YOUTUBE: "YouTube",
    DataSource.MIXED: "Mixed (Twitter+Reddit)",
    DataSource.ALL: "All (Twitter+Reddit+YouTube)",
}


# =============================================================================
# FACTORY FUNCTION
# =============================================================================

def create_strategy(source: DataSource, philosophy: Philosophy) -> StrategyConfig:
    """
    Factory function to create a strategy configuration.

    Args:
        source: Data source (TWITTER, REDDIT, YOUTUBE, MIXED, ALL)
        philosophy: Trading philosophy (SOCIAL_PURE, TECHNICAL_STRICT, BALANCED)

    Returns:
        StrategyConfig with source-specific adjustments applied
    """
    preset = _PRESETS[philosophy]

    # Deep copy configs to avoid mutation
    consensus = deepcopy(preset["consensus"])
    technical = deepcopy(preset["technical"])
    risk = deepcopy(preset["risk"])

    # Apply source-specific adjustments
    signals_adj = _MIN_SIGNALS_ADJUSTMENTS.get((source, philosophy), 0)
    consensus.min_signals_for_trade = max(2, consensus.min_signals_for_trade + signals_adj)

    decay_adj = _DECAY_ADJUSTMENTS.get(source, 0)
    consensus.decay_half_life_hours += decay_adj

    # Get strategy type and description
    strategy_type = _STRATEGY_TYPE_MAP[(source, philosophy)]
    description = _DESCRIPTION_TEMPLATES[philosophy].format(source=_SOURCE_NAMES[source])

    return StrategyConfig(
        name=strategy_type,
        description=description,
        data_source=source,
        consensus=consensus,
        technical=technical,
        risk=risk,
    )


# =============================================================================
# GENERATED STRATEGIES
# =============================================================================

# Twitter strategies
SOCIAL_PURE = create_strategy(DataSource.TWITTER, Philosophy.SOCIAL_PURE)
TECHNICAL_STRICT = create_strategy(DataSource.TWITTER, Philosophy.TECHNICAL_STRICT)
BALANCED = create_strategy(DataSource.TWITTER, Philosophy.BALANCED)

# Reddit strategies
REDDIT_SOCIAL_PURE = create_strategy(DataSource.REDDIT, Philosophy.SOCIAL_PURE)
REDDIT_TECHNICAL_STRICT = create_strategy(DataSource.REDDIT, Philosophy.TECHNICAL_STRICT)
REDDIT_BALANCED = create_strategy(DataSource.REDDIT, Philosophy.BALANCED)

# YouTube strategies
YOUTUBE_SOCIAL_PURE = create_strategy(DataSource.YOUTUBE, Philosophy.SOCIAL_PURE)
YOUTUBE_TECHNICAL_STRICT = create_strategy(DataSource.YOUTUBE, Philosophy.TECHNICAL_STRICT)
YOUTUBE_BALANCED = create_strategy(DataSource.YOUTUBE, Philosophy.BALANCED)

# Mixed strategies
MIXED_SOCIAL_PURE = create_strategy(DataSource.MIXED, Philosophy.SOCIAL_PURE)
MIXED_TECHNICAL_STRICT = create_strategy(DataSource.MIXED, Philosophy.TECHNICAL_STRICT)
MIXED_BALANCED = create_strategy(DataSource.MIXED, Philosophy.BALANCED)

# All sources strategies
ALL_SOCIAL_PURE = create_strategy(DataSource.ALL, Philosophy.SOCIAL_PURE)
ALL_TECHNICAL_STRICT = create_strategy(DataSource.ALL, Philosophy.TECHNICAL_STRICT)
ALL_BALANCED = create_strategy(DataSource.ALL, Philosophy.BALANCED)


# Strategy registry
STRATEGIES = {
    # Twitter-only
    StrategyType.SOCIAL_PURE: SOCIAL_PURE,
    StrategyType.TECHNICAL_STRICT: TECHNICAL_STRICT,
    StrategyType.BALANCED: BALANCED,
    # Reddit-only
    StrategyType.REDDIT_SOCIAL_PURE: REDDIT_SOCIAL_PURE,
    StrategyType.REDDIT_TECHNICAL_STRICT: REDDIT_TECHNICAL_STRICT,
    StrategyType.REDDIT_BALANCED: REDDIT_BALANCED,
    # Mixed (Twitter + Reddit)
    StrategyType.MIXED_SOCIAL_PURE: MIXED_SOCIAL_PURE,
    StrategyType.MIXED_TECHNICAL_STRICT: MIXED_TECHNICAL_STRICT,
    StrategyType.MIXED_BALANCED: MIXED_BALANCED,
    # YouTube-only
    StrategyType.YOUTUBE_SOCIAL_PURE: YOUTUBE_SOCIAL_PURE,
    StrategyType.YOUTUBE_TECHNICAL_STRICT: YOUTUBE_TECHNICAL_STRICT,
    StrategyType.YOUTUBE_BALANCED: YOUTUBE_BALANCED,
    # All sources (Twitter + Reddit + YouTube)
    StrategyType.ALL_SOCIAL_PURE: ALL_SOCIAL_PURE,
    StrategyType.ALL_TECHNICAL_STRICT: ALL_TECHNICAL_STRICT,
    StrategyType.ALL_BALANCED: ALL_BALANCED,
}


def get_strategy(strategy_type: StrategyType) -> StrategyConfig:
    """Get strategy configuration by type."""
    return STRATEGIES[strategy_type]


def list_strategies() -> list[StrategyConfig]:
    """List all available strategies."""
    return list(STRATEGIES.values())


def print_strategy_comparison():
    """Print comparison table of all strategies."""
    print("\n" + "=" * 80)
    print("STRATEGY COMPARISON (Twitter base values)")
    print("=" * 80)

    headers = ["Parameter", "Social Pure", "Technical", "Balanced"]
    rows = [
        ["CONSENSUS", "", "", ""],
        ["  Min signals",
         str(SOCIAL_PURE.consensus.min_signals_for_trade),
         str(TECHNICAL_STRICT.consensus.min_signals_for_trade),
         str(BALANCED.consensus.min_signals_for_trade)],
        ["  Agreement ratio",
         f"{SOCIAL_PURE.consensus.min_agreement_ratio:.0%}",
         f"{TECHNICAL_STRICT.consensus.min_agreement_ratio:.0%}",
         f"{BALANCED.consensus.min_agreement_ratio:.0%}"],
        ["  Sentiment weight",
         f"{SOCIAL_PURE.consensus.sentiment_weight:.0%}",
         f"{TECHNICAL_STRICT.consensus.sentiment_weight:.0%}",
         f"{BALANCED.consensus.sentiment_weight:.0%}"],
        ["  Decay half-life",
         f"{SOCIAL_PURE.consensus.decay_half_life_hours}h",
         f"{TECHNICAL_STRICT.consensus.decay_half_life_hours}h",
         f"{BALANCED.consensus.decay_half_life_hours}h"],
        ["", "", "", ""],
        ["TECHNICAL", "", "", ""],
        ["  Require confirmation",
         str(SOCIAL_PURE.technical.require_tech_confirmation),
         str(TECHNICAL_STRICT.technical.require_tech_confirmation),
         str(BALANCED.technical.require_tech_confirmation)],
        ["  Score threshold",
         str(SOCIAL_PURE.technical.tech_score_threshold),
         str(TECHNICAL_STRICT.technical.tech_score_threshold),
         str(BALANCED.technical.tech_score_threshold)],
        ["  Allow counter-trend",
         str(SOCIAL_PURE.technical.allow_counter_trend),
         str(TECHNICAL_STRICT.technical.allow_counter_trend),
         str(BALANCED.technical.allow_counter_trend)],
        ["", "", "", ""],
        ["RISK", "", "", ""],
        ["  Min R:R",
         f"{SOCIAL_PURE.risk.min_rr_ratio}:1",
         f"{TECHNICAL_STRICT.risk.min_rr_ratio}:1",
         f"{BALANCED.risk.min_rr_ratio}:1"],
        ["  Max positions",
         str(SOCIAL_PURE.risk.max_positions),
         str(TECHNICAL_STRICT.risk.max_positions),
         str(BALANCED.risk.max_positions)],
        ["  Max exposure",
         f"{SOCIAL_PURE.risk.max_total_exposure:.0%}",
         f"{TECHNICAL_STRICT.risk.max_total_exposure:.0%}",
         f"{BALANCED.risk.max_total_exposure:.0%}"],
        ["  Min conviction",
         f"{SOCIAL_PURE.risk.min_conviction:.0%}",
         f"{TECHNICAL_STRICT.risk.min_conviction:.0%}",
         f"{BALANCED.risk.min_conviction:.0%}"],
    ]

    # Print table
    col_widths = [20, 15, 15, 15]
    header_line = " | ".join(h.ljust(w) for h, w in zip(headers, col_widths))
    print(header_line)
    print("-" * len(header_line))

    for row in rows:
        print(" | ".join(str(c).ljust(w) for c, w in zip(row, col_widths)))

    print("\n" + "=" * 80)
    print("EXPECTED BEHAVIOR:")
    print("-" * 80)
    print("SOCIAL_PURE:      More trades, faster signals, higher risk, tests pure social alpha")
    print("TECHNICAL_STRICT: Fewer trades, higher quality, lower risk, tests technical edge")
    print("BALANCED:         Medium trades, hybrid filters, balanced risk, tests synergy")
    print("=" * 80 + "\n")
