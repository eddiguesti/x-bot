"""
Multi-Strategy Runner for A/B Testing

Runs all 3 strategies in parallel with separate portfolios to compare performance.
Each strategy gets its own:
- Portfolio (starting balance)
- Trade history
- Performance metrics

This allows direct comparison of:
- Total return
- Win rate
- Trade frequency
- Risk-adjusted returns (Sharpe)
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Optional

from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from ..config import Settings
from ..models import Asset, Direction, ConsensusAction, Trade, TradeStatus, Portfolio, SignalSource
from ..consensus.enhanced_engine import EnhancedConsensusEngine
from ..analysis.technical import TechnicalAnalyzer, TrendDirection
from ..data_ingestion.price_client import PriceClient
from .config import StrategyConfig, StrategyType, DataSource, get_strategy, STRATEGIES

logger = logging.getLogger(__name__)


@dataclass
class StrategyPerformance:
    """Performance metrics for a single strategy."""
    strategy: StrategyType
    initial_balance: float
    current_balance: float
    total_trades: int = 0
    winning_trades: int = 0
    total_pnl: float = 0.0
    unrealized_pnl: float = 0.0
    max_drawdown: float = 0.0
    peak_balance: float = 0.0
    trades_today: int = 0
    pnl_today: float = 0.0

    @property
    def win_rate(self) -> float:
        return self.winning_trades / self.total_trades if self.total_trades > 0 else 0.0

    @property
    def total_return_pct(self) -> float:
        return ((self.current_balance - self.initial_balance) / self.initial_balance) * 100

    @property
    def total_equity(self) -> float:
        return self.current_balance + self.unrealized_pnl

    def to_dict(self) -> dict:
        return {
            "strategy": self.strategy.value,
            "balance": self.current_balance,
            "total_return": f"{self.total_return_pct:+.2f}%",
            "total_pnl": f"${self.total_pnl:+.2f}",
            "trades": self.total_trades,
            "win_rate": f"{self.win_rate:.1%}",
            "max_drawdown": f"{self.max_drawdown:.1%}",
        }


@dataclass
class StrategyState:
    """Runtime state for a strategy."""
    config: StrategyConfig
    portfolio_name: str
    performance: StrategyPerformance
    open_positions: dict[Asset, Trade] = field(default_factory=dict)
    recent_streak: int = 0  # For anti-martingale


class MultiStrategyRunner:
    """
    Runs multiple strategies in parallel for A/B testing.

    Each strategy operates independently with its own:
    - Portfolio and balance
    - Trade decisions
    - Performance tracking
    """

    def __init__(
        self,
        settings: Settings,
        session: Session,
        initial_balance: float = 10000.0,
        strategies: list[StrategyType] = None,
    ):
        self.settings = settings
        self.session = session
        self.initial_balance = initial_balance

        # Components shared across strategies
        self.consensus_engine = EnhancedConsensusEngine(settings)
        self.technical = TechnicalAnalyzer(settings)
        self.price_client = PriceClient(settings)

        # Initialize strategies
        if strategies is None:
            strategies = list(StrategyType)

        self.strategies: dict[StrategyType, StrategyState] = {}
        for strategy_type in strategies:
            self._init_strategy(strategy_type)

    def _init_strategy(self, strategy_type: StrategyType):
        """Initialize a strategy with its own portfolio."""
        config = get_strategy(strategy_type)
        portfolio_name = f"strategy_{strategy_type.value}"

        # Get or create portfolio for this strategy
        portfolio = self.session.query(Portfolio).filter(
            Portfolio.name == portfolio_name
        ).first()

        if not portfolio:
            portfolio = Portfolio(
                name=portfolio_name,
                initial_balance=self.initial_balance,
                current_balance=self.initial_balance,
            )
            self.session.add(portfolio)
            self.session.commit()
            logger.info(f"Created portfolio for {strategy_type.value}: ${self.initial_balance:,.2f}")

        # Load open positions
        open_trades = self.session.query(Trade).filter(
            Trade.status == TradeStatus.OPEN,
            Trade.notes.contains(portfolio_name),  # Tag trades with strategy
        ).all()

        open_positions = {t.asset: t for t in open_trades}

        # Create state
        performance = StrategyPerformance(
            strategy=strategy_type,
            initial_balance=portfolio.initial_balance,
            current_balance=portfolio.current_balance,
            total_trades=portfolio.total_trades,
            winning_trades=portfolio.winning_trades,
            total_pnl=portfolio.total_pnl,
            peak_balance=max(portfolio.initial_balance, portfolio.current_balance),
        )

        self.strategies[strategy_type] = StrategyState(
            config=config,
            portfolio_name=portfolio_name,
            performance=performance,
            open_positions=open_positions,
        )

    def get_price(self, asset: Asset) -> Optional[float]:
        """Get current price for asset."""
        return self.price_client.get_current_price(asset)

    def evaluate_trade_for_strategy(
        self,
        strategy: StrategyState,
        asset: Asset,
        consensus_score: float,
        consensus_confidence: float,
        long_votes: int,
        short_votes: int,
    ) -> dict:
        """
        Evaluate if a trade should be taken for a specific strategy.

        Applies strategy-specific filters and thresholds.
        """
        config = strategy.config
        result = {
            "should_trade": False,
            "direction": None,
            "position_pct": 0.0,
            "reasons": [],
            "warnings": [],
        }

        # Already have position?
        if asset in strategy.open_positions:
            result["reasons"].append(f"Already have {asset.value} position")
            return result

        # Check position limits
        if len(strategy.open_positions) >= config.risk.max_positions:
            result["reasons"].append(f"Max positions ({config.risk.max_positions})")
            return result

        # Determine direction from consensus
        total_votes = long_votes + short_votes
        if total_votes < config.consensus.min_signals_for_trade:
            result["reasons"].append(f"Insufficient signals ({total_votes} < {config.consensus.min_signals_for_trade})")
            return result

        # Check agreement ratio
        if consensus_score > 0:
            vote_ratio = long_votes / total_votes if total_votes > 0 else 0
            direction = Direction.LONG
        else:
            vote_ratio = short_votes / total_votes if total_votes > 0 else 0
            direction = Direction.SHORT

        if vote_ratio < config.consensus.min_agreement_ratio:
            result["reasons"].append(f"Weak agreement ({vote_ratio:.0%} < {config.consensus.min_agreement_ratio:.0%})")
            return result

        # Check confidence
        if consensus_confidence < config.consensus.min_signal_confidence:
            result["reasons"].append(f"Low confidence ({consensus_confidence:.0%})")
            return result

        # Get technical analysis
        tech_signal = self.technical.analyze(asset)

        # Technical filters (if required)
        if config.technical.require_tech_confirmation and tech_signal:
            # Check tech score threshold
            if direction == Direction.LONG:
                if tech_signal.confirmation_score < config.technical.tech_score_threshold:
                    result["reasons"].append(
                        f"Tech score too low ({tech_signal.confirmation_score:.2f} < {config.technical.tech_score_threshold})"
                    )
                    return result
            else:
                if tech_signal.confirmation_score > -config.technical.tech_score_threshold:
                    result["reasons"].append(
                        f"Tech score too high for short ({tech_signal.confirmation_score:.2f})"
                    )
                    return result

            # Counter-trend check
            if not config.technical.allow_counter_trend:
                if direction == Direction.LONG and tech_signal.trend == TrendDirection.DOWN:
                    if tech_signal.trend_strength > 0.6:
                        result["reasons"].append("Counter-trend blocked (downtrend)")
                        return result
                elif direction == Direction.SHORT and tech_signal.trend == TrendDirection.UP:
                    if tech_signal.trend_strength > 0.6:
                        result["reasons"].append("Counter-trend blocked (uptrend)")
                        return result

            # Volatility filter
            if tech_signal.atr_percent and tech_signal.atr_percent > config.technical.max_volatility_atr:
                result["reasons"].append(f"Volatility too high ({tech_signal.atr_percent:.1f}%)")
                return result

            # RSI extremes
            if tech_signal.rsi:
                if direction == Direction.LONG and tech_signal.rsi > config.technical.rsi_overbought:
                    result["reasons"].append(f"RSI overbought ({tech_signal.rsi:.0f})")
                    return result
                if direction == Direction.SHORT and tech_signal.rsi < config.technical.rsi_oversold:
                    result["reasons"].append(f"RSI oversold ({tech_signal.rsi:.0f})")
                    return result

        # Calculate position size
        base_pct = config.risk.base_position_pct

        # Conviction calculation using geometric mean (naturally bounded 0-1, no magic numbers)
        # Geometric mean ensures both confidence AND vote agreement must be high
        conviction = (consensus_confidence * vote_ratio) ** 0.5
        if conviction < config.risk.min_conviction:
            result["reasons"].append(f"Low conviction ({conviction:.0%} < {config.risk.min_conviction:.0%})")
            return result

        # Scale position with conviction
        if conviction >= 0.8:
            position_pct = min(config.risk.max_position_pct, base_pct * 1.5)
        elif conviction >= 0.6:
            position_pct = base_pct
        else:
            position_pct = max(config.risk.min_position_pct, base_pct * 0.7)

        # Anti-martingale adjustment
        if config.risk.use_anti_martingale and strategy.recent_streak != 0:
            if strategy.recent_streak > 0:
                streak_mult = 1.0 + min(0.3, strategy.recent_streak * 0.1)
            else:
                streak_mult = 1.0 - min(0.4, abs(strategy.recent_streak) * 0.15)
            position_pct *= streak_mult

        position_pct = max(config.risk.min_position_pct, min(config.risk.max_position_pct, position_pct))

        # Check exposure limits
        current_exposure = sum(
            t.position_size * t.entry_price
            for t in strategy.open_positions.values()
        ) / strategy.performance.current_balance if strategy.performance.current_balance > 0 else 0

        if current_exposure + position_pct > config.risk.max_total_exposure:
            remaining = config.risk.max_total_exposure - current_exposure
            if remaining < config.risk.min_position_pct:
                result["reasons"].append(f"Max exposure reached ({current_exposure:.0%})")
                return result
            position_pct = remaining

        # Trade approved!
        result["should_trade"] = True
        result["direction"] = direction
        result["position_pct"] = position_pct
        result["conviction"] = conviction
        result["reasons"].append(f"Votes: {long_votes}L/{short_votes}S ({vote_ratio:.0%})")
        result["reasons"].append(f"Conviction: {conviction:.0%} | Size: {position_pct:.1%}")

        return result

    def open_position(
        self,
        strategy: StrategyState,
        asset: Asset,
        direction: Direction,
        position_pct: float,
        consensus_score: float,
        consensus_confidence: float,
    ) -> Optional[Trade]:
        """Open a position for a specific strategy."""
        price = self.get_price(asset)
        if not price:
            return None

        config = strategy.config
        portfolio = self.session.query(Portfolio).filter(
            Portfolio.name == strategy.portfolio_name
        ).first()

        position_usd = portfolio.current_balance * position_pct
        position_size = position_usd / price

        # Calculate stops
        stop_loss, take_profit = self.technical.get_dynamic_stops(
            asset, direction.value, price, min_rr=config.risk.min_rr_ratio
        )

        # Create trade (tagged with strategy)
        trade = Trade(
            asset=asset,
            direction=direction,
            status=TradeStatus.OPEN,
            entry_price=price,
            entry_time=datetime.utcnow(),
            position_size=position_size,
            consensus_score=consensus_score,
            consensus_confidence=consensus_confidence,
            stop_loss_price=stop_loss,
            take_profit_price=take_profit,
            notes=strategy.portfolio_name,  # Tag with strategy
        )

        self.session.add(trade)
        self.session.commit()

        strategy.open_positions[asset] = trade
        strategy.performance.trades_today += 1

        logger.info(
            f"[{strategy.config.name.value}] Opened {direction.value.upper()} {asset.value}: "
            f"{position_size:.6f} @ ${price:,.2f} ({position_pct:.1%})"
        )

        return trade

    def close_position(
        self,
        strategy: StrategyState,
        trade: Trade,
        reason: str = "manual",
    ) -> Trade:
        """Close a position for a specific strategy."""
        price = self.get_price(trade.asset)
        if not price:
            return trade

        portfolio = self.session.query(Portfolio).filter(
            Portfolio.name == strategy.portfolio_name
        ).first()

        # Calculate PnL
        if trade.direction == Direction.LONG:
            pnl_percent = ((price - trade.entry_price) / trade.entry_price) * 100
        else:
            pnl_percent = ((trade.entry_price - price) / trade.entry_price) * 100

        trade.exit_price = price
        trade.exit_time = datetime.utcnow()
        trade.status = TradeStatus.CLOSED
        trade.pnl_percent = pnl_percent
        trade.pnl_usd = trade.position_size * trade.entry_price * (pnl_percent / 100)

        # Update portfolio
        portfolio.current_balance += trade.pnl_usd
        portfolio.total_trades += 1
        portfolio.total_pnl += trade.pnl_usd

        won = trade.pnl_usd > 0
        if won:
            portfolio.winning_trades += 1

        # Update strategy state
        strategy.performance.current_balance = portfolio.current_balance
        strategy.performance.total_trades = portfolio.total_trades
        strategy.performance.winning_trades = portfolio.winning_trades
        strategy.performance.total_pnl = portfolio.total_pnl
        strategy.performance.pnl_today += trade.pnl_usd

        # Update drawdown
        if portfolio.current_balance > strategy.performance.peak_balance:
            strategy.performance.peak_balance = portfolio.current_balance
        drawdown = (strategy.performance.peak_balance - portfolio.current_balance) / strategy.performance.peak_balance
        strategy.performance.max_drawdown = max(strategy.performance.max_drawdown, drawdown)

        # Update streak
        if won:
            strategy.recent_streak = max(1, strategy.recent_streak + 1) if strategy.recent_streak >= 0 else 1
        else:
            strategy.recent_streak = min(-1, strategy.recent_streak - 1) if strategy.recent_streak <= 0 else -1

        # Remove from open positions
        if trade.asset in strategy.open_positions:
            del strategy.open_positions[trade.asset]

        self.session.commit()

        emoji = "✅" if won else "❌"
        logger.info(
            f"[{strategy.config.name.value}] {emoji} Closed {trade.asset.value} ({reason}): "
            f"PnL: {pnl_percent:+.2f}% (${trade.pnl_usd:+.2f})"
        )

        return trade

    def check_exits(self, strategy: StrategyState) -> list[tuple[Trade, str]]:
        """Check if any positions should be closed for a strategy."""
        to_close = []
        config = strategy.config

        for asset, trade in list(strategy.open_positions.items()):
            price = self.get_price(asset)
            if not price:
                continue

            # Stop loss
            if trade.direction == Direction.LONG:
                if price <= trade.stop_loss_price:
                    to_close.append((trade, "stop_loss"))
                    continue
            else:
                if price >= trade.stop_loss_price:
                    to_close.append((trade, "stop_loss"))
                    continue

            # Take profit
            if trade.direction == Direction.LONG:
                if price >= trade.take_profit_price:
                    to_close.append((trade, "take_profit"))
                    continue
            else:
                if price <= trade.take_profit_price:
                    to_close.append((trade, "take_profit"))
                    continue

            # Time-based exit (stale trades)
            if trade.entry_time:
                age_hours = (datetime.utcnow() - trade.entry_time).total_seconds() / 3600
                if age_hours > 72:  # 3 days max
                    if trade.direction == Direction.LONG:
                        pnl_pct = (price - trade.entry_price) / trade.entry_price
                    else:
                        pnl_pct = (trade.entry_price - price) / trade.entry_price

                    if pnl_pct < 0.02:  # Less than 2% profit
                        to_close.append((trade, "time_exit"))

        return to_close

    def _get_sources_for_data_source(self, data_source: DataSource) -> list[SignalSource]:
        """Map strategy data source config to signal sources."""
        if data_source == DataSource.TWITTER:
            return [SignalSource.TWITTER]
        elif data_source == DataSource.REDDIT:
            return [SignalSource.REDDIT]
        elif data_source == DataSource.YOUTUBE:
            return [SignalSource.YOUTUBE]
        elif data_source == DataSource.MIXED:
            return [SignalSource.TWITTER, SignalSource.REDDIT]
        elif data_source == DataSource.ALL:
            return [SignalSource.TWITTER, SignalSource.REDDIT, SignalSource.YOUTUBE]
        return [SignalSource.TWITTER]  # Default

    def run_cycle(self, assets: list[Asset] = None) -> dict[StrategyType, list[str]]:
        """
        Run one trading cycle for all strategies.

        Returns dict of strategy -> list of actions taken.
        """
        if assets is None:
            assets = list(Asset)

        results = {st: [] for st in self.strategies}

        # Pre-calculate consensus for each data source type (to avoid redundant queries)
        consensus_by_source: dict[tuple, dict[Asset, any]] = {}

        for strategy_type, strategy in self.strategies.items():
            # Get the signal sources for this strategy
            sources = self._get_sources_for_data_source(strategy.config.data_source)
            sources_key = tuple(sorted(s.value for s in sources))

            # Calculate consensus if not already done for this source combo
            if sources_key not in consensus_by_source:
                consensus_by_source[sources_key] = {}
                for asset in assets:
                    try:
                        consensus = self.consensus_engine.calculate_consensus(
                            self.session, asset, lookback_hours=24,
                            save_snapshot=False, sources=sources
                        )
                        consensus_by_source[sources_key][asset] = consensus
                    except Exception as e:
                        logger.warning(f"Failed to get consensus for {asset.value}: {e}")

        for strategy_type, strategy in self.strategies.items():
            # Check exits first
            exits = self.check_exits(strategy)
            for trade, reason in exits:
                self.close_position(strategy, trade, reason)
                results[strategy_type].append(f"CLOSE {trade.asset.value}: {reason}")

            # Get the consensus map for this strategy's data source
            sources = self._get_sources_for_data_source(strategy.config.data_source)
            sources_key = tuple(sorted(s.value for s in sources))
            consensus_map = consensus_by_source.get(sources_key, {})

            # Evaluate new trades
            for asset in assets:
                if asset in strategy.open_positions:
                    continue

                consensus = consensus_map.get(asset)
                if not consensus or consensus.action == ConsensusAction.NO_TRADE:
                    continue

                eval_result = self.evaluate_trade_for_strategy(
                    strategy,
                    asset,
                    consensus.weighted_score,
                    consensus.confidence,
                    consensus.long_votes,
                    consensus.short_votes,
                )

                if eval_result["should_trade"]:
                    trade = self.open_position(
                        strategy,
                        asset,
                        eval_result["direction"],
                        eval_result["position_pct"],
                        consensus.weighted_score,
                        consensus.confidence,
                    )
                    if trade:
                        results[strategy_type].append(
                            f"OPEN {eval_result['direction'].value.upper()} {asset.value}"
                        )

        return results

    def get_performance_summary(self) -> dict[StrategyType, StrategyPerformance]:
        """Get performance summary for all strategies."""
        # Update unrealized PnL
        for strategy in self.strategies.values():
            unrealized = 0.0
            for asset, trade in strategy.open_positions.items():
                price = self.get_price(asset)
                if price:
                    if trade.direction == Direction.LONG:
                        pnl_pct = (price - trade.entry_price) / trade.entry_price
                    else:
                        pnl_pct = (trade.entry_price - price) / trade.entry_price
                    unrealized += trade.position_size * trade.entry_price * pnl_pct
            strategy.performance.unrealized_pnl = unrealized

        return {st: s.performance for st, s in self.strategies.items()}

    def print_comparison(self):
        """Print performance comparison across all strategies."""
        performances = self.get_performance_summary()

        if not performances:
            print("No strategies to compare")
            return

        # Group strategies by data source
        twitter_strats = [st for st in performances if st.value in ("social_pure", "technical_strict", "balanced")]
        reddit_strats = [st for st in performances if "reddit_" in st.value]
        mixed_strats = [st for st in performances if "mixed_" in st.value]
        youtube_strats = [st for st in performances if "youtube_" in st.value]
        all_strats = [st for st in performances if "all_" in st.value]

        print("\n" + "=" * 100)
        print("STRATEGY PERFORMANCE COMPARISON")
        print("=" * 100)

        # Print each group
        for group_name, group_strats in [
            ("TWITTER (X)", twitter_strats),
            ("REDDIT", reddit_strats),
            ("YOUTUBE", youtube_strats),
            ("MIXED (Twitter + Reddit)", mixed_strats),
            ("ALL (Twitter + Reddit + YouTube)", all_strats)
        ]:
            if not group_strats:
                continue

            print(f"\n--- {group_name} ---")
            print(f"{'Strategy':<25} {'Balance':>12} {'Return':>10} {'Trades':>8} {'Win Rate':>10} {'Drawdown':>10}")
            print("-" * 80)

            for st in group_strats:
                perf = performances[st]
                open_pos = len(self.strategies[st].open_positions) if st in self.strategies else 0
                name = st.value.replace("_", " ").title()
                print(
                    f"{name:<25} "
                    f"${perf.current_balance:>10,.0f} "
                    f"{perf.total_return_pct:>+9.2f}% "
                    f"{perf.total_trades:>8} "
                    f"{perf.win_rate:>9.1%} "
                    f"{perf.max_drawdown:>9.1%}"
                )

        print("\n" + "=" * 100)

        # Determine overall winner across ALL strategies
        if performances:
            returns = [(st, perf.total_return_pct) for st, perf in performances.items()]
            best = max(returns, key=lambda x: x[1])
            print(f"\n🏆 CURRENT LEADER: {best[0].value.upper()} ({best[1]:+.2f}% return)")

        print("")
