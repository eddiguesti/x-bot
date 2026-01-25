"""
Strategy configurations for A/B testing different trading approaches.

THREE STRATEGIES TO TEST:
1. SOCIAL_PURE - Trust the crowd: social signals only, minimal filters
2. TECHNICAL_STRICT - Trust the charts: heavy technical confirmation
3. BALANCED - Trust both: hybrid approach (current optimized version)

Each strategy has different:
- Signal thresholds
- Technical confirmation requirements
- Position sizing rules
- Risk parameters
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class StrategyType(str, Enum):
    """Available trading strategies."""
    SOCIAL_PURE = "social_pure"        # Pure social sentiment
    TECHNICAL_STRICT = "technical_strict"  # Heavy technical filters
    BALANCED = "balanced"              # Hybrid approach


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
    tech_score_threshold: float = -0.2  # Minimum score for longs
    allow_counter_trend: bool = True
    counter_trend_rsi_threshold: float = 35.0  # RSI for mean reversion
    max_volatility_atr: float = 8.0  # Skip if ATR% above this
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
    consensus: ConsensusConfig = field(default_factory=ConsensusConfig)
    technical: TechnicalConfig = field(default_factory=TechnicalConfig)
    risk: RiskConfig = field(default_factory=RiskConfig)

    def __str__(self) -> str:
        return f"{self.name.value}: {self.description}"


# =============================================================================
# STRATEGY 1: SOCIAL PURE
# Philosophy: The crowd knows. Trust curated traders + sentiment signals.
# Hypothesis: Social alpha decays fast in crypto - act on signals quickly
# =============================================================================
SOCIAL_PURE = StrategyConfig(
    name=StrategyType.SOCIAL_PURE,
    description="Pure social signals - trust the crowd, minimal filters",
    consensus=ConsensusConfig(
        min_signal_confidence=0.15,      # Very low bar for testing
        min_creator_accuracy=0.20,       # Trust more creators
        min_signals_for_trade=1,         # Single signal can trade
        min_agreement_ratio=0.55,        # Simple majority
        decay_half_life_hours=4.0,       # FAST decay - recent signals matter most
        use_crowd_sentiment=True,
        sentiment_weight=0.30,           # Heavy sentiment influence
    ),
    technical=TechnicalConfig(
        require_tech_confirmation=False,  # NO technical gates
        tech_score_threshold=-0.5,        # Only block extreme against
        allow_counter_trend=True,
        counter_trend_rsi_threshold=30.0,
        max_volatility_atr=10.0,          # Allow higher volatility
        rsi_overbought=85.0,              # Only block extremes
        rsi_oversold=15.0,
    ),
    risk=RiskConfig(
        base_position_pct=0.012,          # Slightly smaller base (more trades)
        min_position_pct=0.005,
        max_position_pct=0.025,           # Cap individual trades
        max_total_exposure=0.25,          # Allow more exposure (more trades)
        min_rr_ratio=1.2,                 # Lower R:R ok (higher frequency)
        target_rr_ratio=2.0,
        min_conviction=0.30,              # Lower bar to trade
        max_positions=8,                  # More concurrent positions
        max_daily_loss_pct=0.03,          # Slightly wider daily loss
        max_drawdown_pct=0.10,
        use_partial_profits=True,
        use_anti_martingale=True,
    ),
)


# =============================================================================
# STRATEGY 2: TECHNICAL STRICT
# Philosophy: Charts don't lie. Only trade when technicals strongly confirm.
# Hypothesis: Fewer, higher quality trades with better win rate
# =============================================================================
TECHNICAL_STRICT = StrategyConfig(
    name=StrategyType.TECHNICAL_STRICT,
    description="Technical-heavy - charts must confirm, quality over quantity",
    consensus=ConsensusConfig(
        min_signal_confidence=0.45,       # Higher quality signals only
        min_creator_accuracy=0.50,        # Only trust accurate creators
        min_signals_for_trade=7,          # Need more agreement
        min_agreement_ratio=0.70,         # Strong consensus required
        decay_half_life_hours=12.0,       # Slower decay - signals have more life
        use_crowd_sentiment=True,
        sentiment_weight=0.10,            # Less sentiment influence
    ),
    technical=TechnicalConfig(
        require_tech_confirmation=True,
        tech_score_threshold=0.1,         # Must be positive (confirming)
        allow_counter_trend=False,        # NO counter-trend trades
        counter_trend_rsi_threshold=25.0, # Very oversold only
        max_volatility_atr=6.0,           # Avoid high volatility
        rsi_overbought=70.0,              # Stricter RSI
        rsi_oversold=30.0,
    ),
    risk=RiskConfig(
        base_position_pct=0.02,           # Larger positions (fewer trades)
        min_position_pct=0.01,
        max_position_pct=0.04,            # Allow bigger on A+ setups
        max_total_exposure=0.15,          # Lower total exposure
        min_rr_ratio=2.0,                 # Strict 2:1 minimum
        target_rr_ratio=3.0,              # Aim for 3:1
        min_conviction=0.50,              # High conviction only
        max_positions=4,                  # Fewer concurrent
        max_daily_loss_pct=0.02,          # Tighter daily loss
        max_drawdown_pct=0.06,            # Tighter drawdown
        use_partial_profits=True,
        use_anti_martingale=True,
    ),
)


# =============================================================================
# STRATEGY 3: BALANCED (Current Optimized)
# Philosophy: Best of both worlds. Social for signals, technical for validation.
# Hypothesis: Crypto-adapted hybrid outperforms pure approaches
# =============================================================================
BALANCED = StrategyConfig(
    name=StrategyType.BALANCED,
    description="Hybrid approach - social signals with crypto-adapted tech filters",
    consensus=ConsensusConfig(
        min_signal_confidence=0.35,
        min_creator_accuracy=0.40,
        min_signals_for_trade=5,
        min_agreement_ratio=0.60,
        decay_half_life_hours=8.0,
        use_crowd_sentiment=True,
        sentiment_weight=0.20,
    ),
    technical=TechnicalConfig(
        require_tech_confirmation=True,
        tech_score_threshold=-0.2,        # "Not strongly against"
        allow_counter_trend=True,         # Allow mean reversion
        counter_trend_rsi_threshold=35.0,
        max_volatility_atr=8.0,
        rsi_overbought=80.0,
        rsi_oversold=20.0,
    ),
    risk=RiskConfig(
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
)


# Strategy registry
STRATEGIES = {
    StrategyType.SOCIAL_PURE: SOCIAL_PURE,
    StrategyType.TECHNICAL_STRICT: TECHNICAL_STRICT,
    StrategyType.BALANCED: BALANCED,
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
    print("STRATEGY COMPARISON")
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
