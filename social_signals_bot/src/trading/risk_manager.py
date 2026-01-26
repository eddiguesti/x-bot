"""Advanced risk management for position sizing and exposure control.

PROFESSIONAL ROI MAXIMIZATION PRINCIPLES:
1. Asymmetric Kelly - size bigger when edge is clear (conviction-based)
2. Minimum R:R enforcement - 1.5:1 minimum (profitable at 50% win rate)
3. Let winners run - extend TP on momentum, tight trail on losers
4. Anti-martingale - increase after wins, decrease after losses
5. Expectancy focus - optimize for (win_rate * avg_win) - (loss_rate * avg_loss)
"""

import logging
import math
from dataclasses import dataclass, field
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
    conviction_score: float = 0.0  # 0-1, how strong the setup is
    expected_rr: float = 0.0  # Expected risk/reward ratio


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

        # ============================================================
        # PROFESSIONAL ROI-MAXIMIZING PARAMETERS
        # Based on: Renaissance Technologies, Two Sigma, prop trading
        # ============================================================

        # POSITION SIZING - Asymmetric Kelly
        self.base_position_pct = 0.015  # Base 1.5% (scales with conviction)
        self.min_position_pct = 0.005   # Min 0.5% for low conviction
        self.max_position_pct = 0.03    # Max 3% for A+ setups
        self.max_total_exposure = 0.20  # Max 20% total (more aggressive)
        self.max_group_exposure = 0.08  # Max 8% per correlation group

        # TRADE QUALITY FILTERS
        # R:R RESEARCH: At 50% win rate, 1.5:1 gives +0.25R expectancy per trade
        # Crypto volatility makes 2:1 targets hard to hit consistently
        # Lower minimum = more trades = more compound opportunities
        self.min_rr_ratio = 1.5         # MINIMUM 1.5:1 R:R (profitable at 50% WR)
        self.target_rr_ratio = 2.5      # Target 2.5:1 for A+ setups
        self.min_conviction = 0.4       # Minimum conviction score to trade

        # RISK LIMITS
        self.max_positions = 6          # Max concurrent (slightly higher)
        self.max_daily_loss_pct = 0.025 # Tighter daily loss (2.5%)
        self.max_drawdown_pct = 0.08    # Tighter drawdown (8%)

        # ANTI-MARTINGALE (increase size after wins)
        self.streak_adjustment = 0.10   # +10% size per winning trade (max +30%)
        self.max_streak_bonus = 0.30    # Cap at +30%
        self.loss_streak_penalty = 0.15 # -15% size per losing trade
        self.max_loss_penalty = 0.40    # Cap at -40%

        # PROFIT TAKING LEVELS (partial exits)
        self.partial_tp_1 = 0.33        # Take 33% at first target
        self.partial_tp_1_rr = 1.5      # First target at 1.5R
        self.partial_tp_2 = 0.33        # Take another 33% at second target
        self.partial_tp_2_rr = 2.5      # Second target at 2.5R
        # Remaining 34% trails to maximize winners

        # Track recent performance for anti-martingale
        self._recent_streak = 0  # Positive = wins, negative = losses

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
        Calculate ASYMMETRIC Kelly Criterion fraction.

        Uses quarter-Kelly as base, but scales UP for high-edge situations.
        This is what professional quant funds do - bet big when edge is clear.
        """
        if avg_win <= 0 or avg_loss <= 0:
            return self.base_position_pct

        # Kelly formula: f* = (p*b - q) / b
        # where p=win_rate, q=loss_rate, b=avg_win/avg_loss
        b = avg_win / avg_loss
        kelly = (win_rate * b - (1 - win_rate)) / b

        if kelly <= 0:
            return self.min_position_pct  # Negative edge = minimum size

        # ASYMMETRIC SCALING based on edge strength
        # Quarter-Kelly for normal, half-Kelly for strong edge
        edge_strength = kelly * win_rate  # Combines magnitude and reliability

        if edge_strength > 0.15:  # Strong edge (>15%)
            kelly_fraction = kelly * 0.5  # Half-Kelly
        elif edge_strength > 0.08:  # Moderate edge
            kelly_fraction = kelly * 0.35
        else:  # Weak edge
            kelly_fraction = kelly * 0.25  # Quarter-Kelly

        return max(self.min_position_pct, min(self.max_position_pct, kelly_fraction))

    def calculate_conviction_score(
        self,
        consensus_confidence: float,
        tech_score: float,
        direction: Direction,
        asset: Asset,
    ) -> tuple[float, list[str]]:
        """
        Calculate CONVICTION SCORE (0-1) combining multiple factors.

        High conviction = larger position, low conviction = smaller/skip.
        This is the key to asymmetric returns - bet big on A+ setups only.
        """
        factors = []
        score = 0.0

        # 1. CONSENSUS STRENGTH (30% weight)
        # High confidence from smart money = high conviction
        consensus_factor = consensus_confidence
        score += consensus_factor * 0.30
        factors.append(f"consensus:{consensus_factor:.0%}")

        # 2. TECHNICAL ALIGNMENT (25% weight)
        # Technicals confirming = higher conviction
        tech_signal = self.technical.analyze(asset)
        tech_factor = 0.5  # Neutral default

        if tech_signal:
            if direction == Direction.LONG:
                if tech_signal.confirmation_score > 0.3:
                    tech_factor = 0.5 + tech_signal.confirmation_score * 0.5
                elif tech_signal.confirmation_score < -0.2:
                    tech_factor = 0.3  # Technicals against = lower conviction
            else:  # SHORT
                if tech_signal.confirmation_score < -0.3:
                    tech_factor = 0.5 + abs(tech_signal.confirmation_score) * 0.5
                elif tech_signal.confirmation_score > 0.2:
                    tech_factor = 0.3

        score += tech_factor * 0.25
        factors.append(f"tech:{tech_factor:.0%}")

        # 3. VOLATILITY REGIME (15% weight)
        # Medium volatility = best for profits
        vol_factor = 0.5
        if tech_signal and tech_signal.atr_percent:
            if 2.0 <= tech_signal.atr_percent <= 5.0:
                vol_factor = 0.8  # Goldilocks zone
            elif tech_signal.atr_percent > 8.0:
                vol_factor = 0.3  # Too volatile
            elif tech_signal.atr_percent < 1.0:
                vol_factor = 0.4  # Too quiet

        score += vol_factor * 0.15
        factors.append(f"vol:{vol_factor:.0%}")

        # 4. HISTORICAL EDGE for this asset (20% weight)
        closed_trades = self.session.query(Trade).filter(
            Trade.status == TradeStatus.CLOSED,
            Trade.asset == asset,
        ).order_by(Trade.exit_time.desc()).limit(20).all()

        edge_factor = 0.5  # Neutral default
        if len(closed_trades) >= 5:
            wins = [t for t in closed_trades if (t.pnl_percent or 0) > 0]
            win_rate = len(wins) / len(closed_trades)
            avg_pnl = sum(t.pnl_percent or 0 for t in closed_trades) / len(closed_trades)

            if win_rate > 0.55 and avg_pnl > 0:
                edge_factor = min(0.9, 0.5 + win_rate * 0.5)
            elif win_rate < 0.40 or avg_pnl < -1:
                edge_factor = max(0.2, win_rate)

        score += edge_factor * 0.20
        factors.append(f"edge:{edge_factor:.0%}")

        # 5. STREAK MOMENTUM (10% weight) - Anti-martingale
        streak_factor = 0.5
        if self._recent_streak > 0:
            streak_factor = min(0.8, 0.5 + self._recent_streak * 0.1)
        elif self._recent_streak < 0:
            streak_factor = max(0.2, 0.5 + self._recent_streak * 0.1)

        score += streak_factor * 0.10
        factors.append(f"streak:{self._recent_streak:+d}")

        return min(1.0, score), factors

    def calculate_expected_rr(
        self,
        asset: Asset,
        direction: Direction,
        entry_price: float,
    ) -> tuple[float, float, float]:
        """
        Calculate EXPECTED RISK/REWARD ratio.

        Returns: (stop_loss, take_profit, rr_ratio)
        Enforces minimum R:R of 2:1.
        """
        tech_signal = self.technical.analyze(asset)

        # Get ATR for volatility-based stops
        atr = tech_signal.atr if tech_signal else entry_price * 0.02
        atr_pct = (atr / entry_price) if entry_price > 0 else 0.02

        # DYNAMIC STOP LOSS based on volatility
        # Tighter in low vol, wider in high vol (but max 3%)
        if tech_signal and tech_signal.atr_percent:
            if tech_signal.atr_percent < 2:
                sl_multiplier = 1.2  # Tight stop in low vol
            elif tech_signal.atr_percent > 5:
                sl_multiplier = 2.0  # Wide stop in high vol
            else:
                sl_multiplier = 1.5
        else:
            sl_multiplier = 1.5

        stop_distance = atr * sl_multiplier
        stop_pct = min(0.03, stop_distance / entry_price)  # Max 3% stop

        # TAKE PROFIT - enforce minimum 2:1 R:R, target 3:1
        # For high conviction, aim higher
        tp_multiplier = max(self.min_rr_ratio, self.target_rr_ratio)
        tp_distance = stop_distance * tp_multiplier

        if direction == Direction.LONG:
            stop_loss = entry_price * (1 - stop_pct)
            take_profit = entry_price + tp_distance
        else:
            stop_loss = entry_price * (1 + stop_pct)
            take_profit = entry_price - tp_distance

        rr_ratio = tp_distance / stop_distance if stop_distance > 0 else 0

        return stop_loss, take_profit, rr_ratio

    def calculate_position_size(
        self,
        asset: Asset,
        direction: Direction,
        consensus_confidence: float,
        entry_price: float,
    ) -> PositionSizeResult:
        """
        PROFESSIONAL ASYMMETRIC POSITION SIZING.

        Key principles:
        1. Size based on CONVICTION - bet big on A+ setups
        2. Enforce minimum R:R ratio (2:1)
        3. Anti-martingale - increase after wins
        4. Volatility-adjusted for risk parity
        """
        portfolio = self.get_portfolio()
        if not portfolio:
            return PositionSizeResult(0, 0, "No portfolio", 1.0, 0, 0)

        risk_metrics = self.get_risk_metrics()
        if not risk_metrics.can_trade:
            return PositionSizeResult(0, 0, risk_metrics.block_reason, 1.0, 0, 0)

        # 1. Calculate CONVICTION SCORE
        conviction, conviction_factors = self.calculate_conviction_score(
            consensus_confidence, 0, direction, asset
        )

        # FILTER: Skip low conviction trades
        if conviction < self.min_conviction:
            return PositionSizeResult(
                0, 0, f"Low conviction ({conviction:.0%} < {self.min_conviction:.0%})",
                1.0, conviction, 0
            )

        # 2. Calculate R:R ratio
        stop_loss, take_profit, rr_ratio = self.calculate_expected_rr(
            asset, direction, entry_price
        )

        # FILTER: Enforce minimum R:R
        if rr_ratio < self.min_rr_ratio:
            return PositionSizeResult(
                0, 0, f"R:R too low ({rr_ratio:.1f} < {self.min_rr_ratio:.1f})",
                1.0, conviction, rr_ratio
            )

        rationale_parts = conviction_factors.copy()
        rationale_parts.append(f"R:R={rr_ratio:.1f}")

        # 3. BASE SIZE from conviction (asymmetric scaling)
        # Low conviction (0.4-0.6) = 0.5-1% position
        # Medium conviction (0.6-0.8) = 1-2% position
        # High conviction (0.8-1.0) = 2-3% position
        if conviction >= 0.8:
            base_pct = self.base_position_pct * 1.5  # A+ setup
            rationale_parts.append("A+_setup")
        elif conviction >= 0.6:
            base_pct = self.base_position_pct  # Standard
        else:
            base_pct = self.base_position_pct * 0.6  # Reduced

        # 4. VOLATILITY ADJUSTMENT (risk parity)
        vol_mult = self.technical.get_volatility_multiplier(asset)
        rationale_parts.append(f"vol:{vol_mult:.2f}x")

        # 5. CORRELATION ADJUSTMENT
        open_positions = self.get_open_positions()
        group_exposure = self._get_group_exposure(asset, open_positions)
        corr_mult = 1.0
        if group_exposure > 0.04:  # Tighter correlation limit
            corr_mult = max(0.5, 1 - (group_exposure - 0.04) * 5)
            rationale_parts.append(f"corr:{corr_mult:.2f}x")

        # 6. ANTI-MARTINGALE (streak adjustment)
        streak_mult = 1.0
        if self._recent_streak > 0:
            streak_mult = 1.0 + min(self.max_streak_bonus,
                                    self._recent_streak * self.streak_adjustment)
            rationale_parts.append(f"win_streak:+{(streak_mult-1):.0%}")
        elif self._recent_streak < 0:
            streak_mult = 1.0 - min(self.max_loss_penalty,
                                    abs(self._recent_streak) * self.loss_streak_penalty)
            rationale_parts.append(f"loss_streak:{(streak_mult-1):.0%}")

        # 7. KELLY ADJUSTMENT (if enough history)
        kelly_mult = 1.0
        closed_trades = self.session.query(Trade).filter(
            Trade.status == TradeStatus.CLOSED,
        ).order_by(Trade.exit_time.desc()).limit(30).all()

        if len(closed_trades) >= 15:
            wins = [t for t in closed_trades if (t.pnl_percent or 0) > 0]
            losses = [t for t in closed_trades if (t.pnl_percent or 0) <= 0]
            if wins and losses:
                win_rate = len(wins) / len(closed_trades)
                avg_win = sum(t.pnl_percent for t in wins) / len(wins)
                avg_loss = abs(sum(t.pnl_percent for t in losses) / len(losses))
                kelly = self.calculate_kelly_fraction(win_rate, avg_win, avg_loss)
                kelly_mult = kelly / self.base_position_pct
                rationale_parts.append(f"kelly:{kelly_mult:.2f}x")

        # FINAL CALCULATION
        final_pct = base_pct * vol_mult * corr_mult * streak_mult * kelly_mult

        # Apply bounds
        final_pct = max(self.min_position_pct, min(self.max_position_pct, final_pct))

        # Check remaining exposure room
        remaining_exposure = self.max_total_exposure - risk_metrics.exposure_pct
        final_pct = min(final_pct, remaining_exposure)

        # Calculate risk score (inverse of conviction)
        risk_score = 1.0 - conviction

        position_usd = portfolio.current_balance * final_pct

        return PositionSizeResult(
            position_pct=final_pct,
            position_usd=position_usd,
            rationale=" | ".join(rationale_parts),
            risk_score=risk_score,
            conviction_score=conviction,
            expected_rr=rr_ratio,
        )

    def update_streak(self, won: bool):
        """Update win/loss streak for anti-martingale sizing."""
        if won:
            if self._recent_streak >= 0:
                self._recent_streak += 1
            else:
                self._recent_streak = 1
        else:
            if self._recent_streak <= 0:
                self._recent_streak -= 1
            else:
                self._recent_streak = -1

        # Cap streaks
        self._recent_streak = max(-5, min(5, self._recent_streak))
        logger.info(f"Streak updated: {self._recent_streak:+d}")

    def get_partial_tp_levels(
        self,
        entry_price: float,
        stop_loss: float,
        direction: Direction,
    ) -> list[tuple[float, float]]:
        """
        Calculate PARTIAL PROFIT TAKING levels.

        Returns: [(price, percentage_to_close), ...]

        Strategy: Scale out to lock profits while letting winners run.
        - TP1 at 1.5R: Close 33%
        - TP2 at 2.5R: Close 33%
        - Remaining 34%: Trail with tight stop
        """
        risk = abs(entry_price - stop_loss)

        if direction == Direction.LONG:
            tp1_price = entry_price + risk * self.partial_tp_1_rr
            tp2_price = entry_price + risk * self.partial_tp_2_rr
        else:
            tp1_price = entry_price - risk * self.partial_tp_1_rr
            tp2_price = entry_price - risk * self.partial_tp_2_rr

        return [
            (tp1_price, self.partial_tp_1),  # 33% at 1.5R
            (tp2_price, self.partial_tp_2),  # 33% at 2.5R
        ]

    def should_close_position(
        self,
        trade: Trade,
        current_price: float,
        consensus_direction: Optional[Direction] = None,
    ) -> tuple[bool, str]:
        """
        PROFESSIONAL EXIT LOGIC - Let winners run, cut losers.

        Key principles:
        1. NEVER exit a winning trade early on time alone
        2. Cut losers fast when thesis invalidated
        3. Trail winners aggressively
        4. Only close TP when momentum fades

        Reasons:
        1. Stop loss hit (always respect)
        2. Consensus flip (for losers only)
        3. Time-based exit (only for non-performers)
        4. Final TP (after partial profits taken)
        """
        # Calculate current PnL
        if trade.direction == Direction.LONG:
            pnl_pct = (current_price - trade.entry_price) / trade.entry_price
        else:
            pnl_pct = (trade.entry_price - current_price) / trade.entry_price

        # STOP LOSS - Always respect
        if trade.direction == Direction.LONG:
            if current_price <= trade.stop_loss_price:
                return True, "stop_loss"
        else:
            if current_price >= trade.stop_loss_price:
                return True, "stop_loss"

        # TAKE PROFIT - Only close final portion
        # (Partial TPs are handled separately in smart_trader)
        if trade.direction == Direction.LONG:
            if current_price >= trade.take_profit_price:
                # Check if momentum is still strong
                tech_signal = self.technical.analyze(trade.asset)
                if tech_signal and tech_signal.confirmation_score > 0.3:
                    # Momentum still strong - extend TP instead of closing
                    # This is handled by trailing stop, don't close yet
                    logger.info(f"TP hit but momentum strong - trailing {trade.asset.value}")
                    return False, ""
                return True, "take_profit"
        else:
            if current_price <= trade.take_profit_price:
                tech_signal = self.technical.analyze(trade.asset)
                if tech_signal and tech_signal.confirmation_score < -0.3:
                    logger.info(f"TP hit but momentum strong - trailing {trade.asset.value}")
                    return False, ""
                return True, "take_profit"

        # CONSENSUS FLIP - Only close losers
        # Winners get to trail, losers get cut
        if consensus_direction and consensus_direction != trade.direction:
            if pnl_pct < 0.01:  # Only close if not winning
                return True, "consensus_flip"
            # If winning, don't close on flip - just tighten trail

        # TIME-BASED EXIT - Only for non-performers
        # Winners are NEVER closed on time alone
        if trade.entry_time:
            age_hours = (datetime.utcnow() - trade.entry_time).total_seconds() / 3600

            # Get volatility for adaptive timing
            tech_signal = self.technical.analyze(trade.asset)
            if tech_signal and tech_signal.atr_percent:
                if tech_signal.atr_percent > 5:
                    time_limit = 96  # High vol = longer hold (4 days)
                elif tech_signal.atr_percent < 2:
                    time_limit = 36  # Low vol = faster rotation
                else:
                    time_limit = 72  # Normal = 3 days
            else:
                time_limit = 72

            if age_hours > time_limit:
                # ONLY close if not performing well
                if pnl_pct < 0.02:  # Less than 2% profit after full time
                    return True, f"time_exit_{time_limit}h"
                # If profitable, let it continue running

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
