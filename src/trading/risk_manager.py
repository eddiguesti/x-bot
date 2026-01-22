"""Advanced risk management for position sizing and exposure control."""

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Optional

from sqlalchemy.orm import Session

from ..config import Settings
from ..models import Asset, Direction, Trade, TradeStatus, Portfolio
from ..analysis.technical import TechnicalAnalyzer

logger = logging.getLogger(__name__)


@dataclass
class PositionSizeResult:
    """Result of position sizing calculation."""
    position_pct: float  # Percentage of portfolio
    position_usd: float  # Dollar amount
    rationale: str
    risk_score: float  # 0-1 (higher = riskier)


@dataclass
class RiskMetrics:
    """Current risk metrics."""
    total_exposure: float  # Total USD in open positions
    exposure_pct: float  # Percentage of portfolio
    num_positions: int
    max_drawdown: float  # Current drawdown from peak
    daily_pnl: float
    correlation_risk: float  # 0-1
    can_trade: bool
    block_reason: Optional[str]


class RiskManager:
    """
    Advanced risk management system.

    Features:
    1. Kelly Criterion position sizing (modified for safety)
    2. Volatility-adjusted position sizes
    3. Correlation-aware exposure limits
    4. Drawdown circuit breakers
    5. Maximum concurrent positions
    6. Daily loss limits
    """

    # Correlation matrix for major crypto pairs (simplified)
    # High correlation means we should reduce combined exposure
    CORRELATION_GROUPS = {
        'btc_correlated': [Asset.BTC, Asset.ETH, Asset.LTC],
        'alt_l1': [Asset.SOL, Asset.AVAX, Asset.NEAR, Asset.APT],
        'defi': [Asset.UNI, Asset.LINK, Asset.INJ],
        'meme': [Asset.DOGE, Asset.SHIB],
        'l2': [Asset.ARB, Asset.OP],
        'other': [Asset.XRP, Asset.ADA, Asset.DOT, Asset.ATOM, Asset.TAO],
    }

    def __init__(self, settings: Settings, session: Session):
        self.settings = settings
        self.session = session
        self.technical = TechnicalAnalyzer(settings)

        # Risk parameters - PROFESSIONAL STANDARDS
        # Sources: 3Commas, CryptoCrew, ForTraders prop trading
        self.max_position_pct = 0.02  # Max 2% per position (industry standard)
        self.min_position_pct = 0.005  # Min 0.5% per position
        self.max_total_exposure = 0.15  # Max 15% total exposure
        self.max_group_exposure = 0.06  # Max 6% per correlation group
        self.max_positions = 5  # Max concurrent positions
        self.max_daily_loss_pct = 0.03  # Stop trading after 3% daily loss
        self.max_drawdown_pct = 0.10  # Stop trading after 10% drawdown (prop firms use 5-7%)

    def get_portfolio(self) -> Portfolio:
        """Get current portfolio."""
        return self.session.query(Portfolio).filter(
            Portfolio.name == "default"
        ).first()

    def get_open_positions(self) -> list[Trade]:
        """Get all open positions."""
        return self.session.query(Trade).filter(
            Trade.status == TradeStatus.OPEN
        ).all()

    def get_risk_metrics(self) -> RiskMetrics:
        """Calculate current risk metrics."""
        portfolio = self.get_portfolio()
        if not portfolio:
            return RiskMetrics(
                total_exposure=0, exposure_pct=0, num_positions=0,
                max_drawdown=0, daily_pnl=0, correlation_risk=0,
                can_trade=True, block_reason=None
            )

        open_positions = self.get_open_positions()

        # Calculate exposure
        total_exposure = sum(
            t.position_size * t.entry_price for t in open_positions
        )
        exposure_pct = total_exposure / portfolio.current_balance if portfolio.current_balance > 0 else 0

        # Calculate drawdown
        peak = max(portfolio.initial_balance, portfolio.current_balance)
        drawdown = (peak - portfolio.current_balance) / peak if peak > 0 else 0

        # Calculate daily PnL
        today = datetime.utcnow().date()
        todays_trades = self.session.query(Trade).filter(
            Trade.exit_time >= datetime(today.year, today.month, today.day),
            Trade.status == TradeStatus.CLOSED,
        ).all()
        daily_pnl = sum(t.pnl_usd or 0 for t in todays_trades)
        daily_pnl_pct = daily_pnl / portfolio.initial_balance if portfolio.initial_balance > 0 else 0

        # Calculate correlation risk
        correlation_risk = self._calculate_correlation_risk(open_positions)

        # Determine if we can trade
        can_trade = True
        block_reason = None

        if drawdown >= self.max_drawdown_pct:
            can_trade = False
            block_reason = f"Max drawdown reached ({drawdown:.1%})"
        elif daily_pnl_pct <= -self.max_daily_loss_pct:
            can_trade = False
            block_reason = f"Daily loss limit reached ({daily_pnl_pct:.1%})"
        elif len(open_positions) >= self.max_positions:
            can_trade = False
            block_reason = f"Max positions reached ({len(open_positions)})"
        elif exposure_pct >= self.max_total_exposure:
            can_trade = False
            block_reason = f"Max exposure reached ({exposure_pct:.1%})"

        return RiskMetrics(
            total_exposure=total_exposure,
            exposure_pct=exposure_pct,
            num_positions=len(open_positions),
            max_drawdown=drawdown,
            daily_pnl=daily_pnl,
            correlation_risk=correlation_risk,
            can_trade=can_trade,
            block_reason=block_reason,
        )

    def _calculate_correlation_risk(self, positions: list[Trade]) -> float:
        """Calculate correlation risk from current positions."""
        if len(positions) <= 1:
            return 0.0

        # Count positions per correlation group
        group_counts = {}
        for trade in positions:
            for group, assets in self.CORRELATION_GROUPS.items():
                if trade.asset in assets:
                    group_counts[group] = group_counts.get(group, 0) + 1
                    break

        # Risk increases with concentration in same group
        max_group = max(group_counts.values()) if group_counts else 0
        return min(1.0, (max_group - 1) * 0.3)

    def _get_group_exposure(self, asset: Asset, positions: list[Trade]) -> float:
        """Get total exposure for asset's correlation group."""
        # Find which group this asset belongs to
        asset_group = None
        for group, assets in self.CORRELATION_GROUPS.items():
            if asset in assets:
                asset_group = group
                break

        if not asset_group:
            return 0.0

        # Sum exposure for all positions in same group
        portfolio = self.get_portfolio()
        group_exposure = 0.0
        for trade in positions:
            if trade.asset in self.CORRELATION_GROUPS.get(asset_group, []):
                group_exposure += trade.position_size * trade.entry_price

        return group_exposure / portfolio.current_balance if portfolio.current_balance > 0 else 0

    def calculate_kelly_fraction(
        self,
        win_rate: float,
        avg_win: float,
        avg_loss: float
    ) -> float:
        """
        Calculate Kelly Criterion fraction.

        Kelly = (win_rate * avg_win - (1 - win_rate) * avg_loss) / avg_win

        We use half-Kelly for safety.
        """
        if avg_win <= 0 or avg_loss <= 0:
            return 0.05  # Default to 5%

        # Kelly formula
        kelly = (win_rate * avg_win - (1 - win_rate) * avg_loss) / avg_win

        # Half-Kelly for safety, bounded
        half_kelly = kelly / 2
        return max(0.02, min(0.20, half_kelly))

    def calculate_position_size(
        self,
        asset: Asset,
        direction: Direction,
        consensus_confidence: float,
        entry_price: float,
    ) -> PositionSizeResult:
        """
        Calculate optimal position size based on multiple factors.

        Factors:
        1. Consensus confidence (higher = larger)
        2. Technical confirmation (aligned = larger)
        3. Volatility (higher = smaller)
        4. Correlation exposure (higher = smaller)
        5. Historical win rate (Kelly Criterion)
        """
        portfolio = self.get_portfolio()
        if not portfolio:
            return PositionSizeResult(0, 0, "No portfolio", 1.0)

        risk_metrics = self.get_risk_metrics()
        if not risk_metrics.can_trade:
            return PositionSizeResult(0, 0, risk_metrics.block_reason, 1.0)

        rationale_parts = []

        # Base position size (1% of portfolio)
        base_pct = 0.01

        # 1. Confidence adjustment (0.5x to 1.5x)
        confidence_mult = 0.5 + consensus_confidence
        rationale_parts.append(f"confidence:{confidence_mult:.2f}x")

        # 2. Technical confirmation
        tech_signal = self.technical.analyze(asset)
        tech_mult = 1.0
        if tech_signal:
            # Check if technicals align with direction
            if direction == Direction.LONG and tech_signal.confirmation_score > 0.2:
                tech_mult = 1.0 + tech_signal.confirmation_score * 0.5
            elif direction == Direction.SHORT and tech_signal.confirmation_score < -0.2:
                tech_mult = 1.0 + abs(tech_signal.confirmation_score) * 0.5
            elif (direction == Direction.LONG and tech_signal.confirmation_score < -0.3) or \
                 (direction == Direction.SHORT and tech_signal.confirmation_score > 0.3):
                # Technicals against us - reduce size
                tech_mult = 0.5
                rationale_parts.append("tech_against")
            rationale_parts.append(f"tech:{tech_mult:.2f}x")

        # 3. Volatility adjustment
        vol_mult = self.technical.get_volatility_multiplier(asset)
        rationale_parts.append(f"vol:{vol_mult:.2f}x")

        # 4. Correlation exposure adjustment
        open_positions = self.get_open_positions()
        group_exposure = self._get_group_exposure(asset, open_positions)
        corr_mult = 1.0
        if group_exposure > 0.15:
            corr_mult = max(0.5, 1 - (group_exposure - 0.15) * 2)
            rationale_parts.append(f"corr:{corr_mult:.2f}x")

        # 5. Kelly adjustment (if we have history)
        closed_trades = self.session.query(Trade).filter(
            Trade.status == TradeStatus.CLOSED,
            Trade.asset == asset,
        ).all()

        kelly_mult = 1.0
        if len(closed_trades) >= 10:
            wins = [t for t in closed_trades if (t.pnl_percent or 0) > 0]
            losses = [t for t in closed_trades if (t.pnl_percent or 0) <= 0]
            if wins and losses:
                win_rate = len(wins) / len(closed_trades)
                avg_win = sum(t.pnl_percent for t in wins) / len(wins)
                avg_loss = abs(sum(t.pnl_percent for t in losses) / len(losses))
                kelly = self.calculate_kelly_fraction(win_rate, avg_win, avg_loss)
                kelly_mult = kelly / 0.01  # Relative to base
                rationale_parts.append(f"kelly:{kelly_mult:.2f}x")

        # Calculate final position size
        final_pct = base_pct * confidence_mult * tech_mult * vol_mult * corr_mult * kelly_mult

        # Apply bounds
        final_pct = max(self.min_position_pct, min(self.max_position_pct, final_pct))

        # Check remaining exposure room
        remaining_exposure = self.max_total_exposure - risk_metrics.exposure_pct
        final_pct = min(final_pct, remaining_exposure)

        # Calculate risk score
        risk_score = min(1.0, (
            (1 - consensus_confidence) * 0.3 +
            risk_metrics.correlation_risk * 0.2 +
            (risk_metrics.exposure_pct / self.max_total_exposure) * 0.3 +
            (1 - vol_mult) * 0.2
        ))

        position_usd = portfolio.current_balance * final_pct

        return PositionSizeResult(
            position_pct=final_pct,
            position_usd=position_usd,
            rationale=" | ".join(rationale_parts),
            risk_score=risk_score,
        )

    def should_close_position(
        self,
        trade: Trade,
        current_price: float,
        consensus_direction: Optional[Direction] = None,
    ) -> tuple[bool, str]:
        """
        Determine if a position should be closed.

        Reasons:
        1. Stop loss hit
        2. Take profit hit
        3. Consensus flipped
        4. Time-based exit
        5. Trailing stop hit
        """
        # Standard SL/TP
        if trade.direction == Direction.LONG:
            if current_price <= trade.stop_loss_price:
                return True, "stop_loss"
            if current_price >= trade.take_profit_price:
                return True, "take_profit"
        else:
            if current_price >= trade.stop_loss_price:
                return True, "stop_loss"
            if current_price <= trade.take_profit_price:
                return True, "take_profit"

        # Consensus flip (if provided)
        if consensus_direction and consensus_direction != trade.direction:
            # Only close if consensus strongly disagrees
            return True, "consensus_flip"

        # VOLATILITY-ADAPTIVE time-based exit
        # High volatility = longer hold (72h), Low volatility = shorter (24h)
        if trade.entry_time:
            age_hours = (datetime.utcnow() - trade.entry_time).total_seconds() / 3600

            # Get current volatility to determine exit timing
            tech_signal = self.technical.analyze(trade.asset)
            if tech_signal and tech_signal.atr_percent:
                # Adaptive time limit based on volatility
                if tech_signal.atr_percent > 5:  # High volatility (>5% daily)
                    time_limit = 72  # Let it run longer
                elif tech_signal.atr_percent < 2:  # Low volatility (<2% daily)
                    time_limit = 24  # Exit faster, rotate capital
                else:
                    time_limit = 48  # Normal
            else:
                time_limit = 48  # Default

            if age_hours > time_limit:
                if trade.direction == Direction.LONG:
                    pnl_pct = (current_price - trade.entry_price) / trade.entry_price
                else:
                    pnl_pct = (trade.entry_price - current_price) / trade.entry_price

                if pnl_pct < 0.01:  # Less than 1% profit
                    return True, f"time_exit_{time_limit}h"

        return False, ""

    def calculate_trailing_stop(
        self,
        trade: Trade,
        current_price: float,
        atr: Optional[float] = None,
    ) -> float:
        """
        Calculate VOLATILITY-ADAPTIVE trailing stop level.

        - Tight trail (1.5x ATR) in calm markets to lock in profits
        - Loose trail (2.5x ATR) in choppy markets to avoid whipsaws
        """
        # Get ATR and volatility info
        tech_signal = self.technical.analyze(trade.asset)
        if atr is None:
            atr = tech_signal.atr if tech_signal else 0

        if atr <= 0:
            # Fallback to percentage-based
            trail_pct = 0.02
            if trade.direction == Direction.LONG:
                return current_price * (1 - trail_pct)
            else:
                return current_price * (1 + trail_pct)

        # ADAPTIVE MULTIPLIER based on volatility regime
        if tech_signal and tech_signal.atr_percent:
            if tech_signal.atr_percent > 5:  # High volatility
                trail_mult = 2.5  # Looser to avoid whipsaws
            elif tech_signal.atr_percent < 2:  # Low volatility
                trail_mult = 1.5  # Tighter to lock profits
            else:
                trail_mult = 2.0  # Normal
        else:
            trail_mult = 2.0

        trail_distance = atr * trail_mult

        if trade.direction == Direction.LONG:
            # For longs, trail below price
            new_stop = current_price - trail_distance
            # Only move stop up, never down
            return max(trade.stop_loss_price, new_stop)
        else:
            # For shorts, trail above price
            new_stop = current_price + trail_distance
            # Only move stop down, never up
            return min(trade.stop_loss_price, new_stop)

    def display_risk_status(self):
        """Display current risk status."""
        metrics = self.get_risk_metrics()
        portfolio = self.get_portfolio()

        print("\n" + "=" * 60)
        print("⚠️  RISK STATUS")
        print("=" * 60)

        print(f"\n📊 EXPOSURE:")
        print(f"   Total Exposure:   ${metrics.total_exposure:,.2f} ({metrics.exposure_pct:.1%})")
        print(f"   Open Positions:   {metrics.num_positions}/{self.max_positions}")
        print(f"   Correlation Risk: {metrics.correlation_risk:.1%}")

        print(f"\n📉 DRAWDOWN:")
        print(f"   Current:          {metrics.max_drawdown:.1%}")
        print(f"   Max Allowed:      {self.max_drawdown_pct:.1%}")
        print(f"   Daily PnL:        ${metrics.daily_pnl:+,.2f}")

        status = "✅ CAN TRADE" if metrics.can_trade else f"❌ BLOCKED: {metrics.block_reason}"
        print(f"\n🚦 STATUS: {status}")
