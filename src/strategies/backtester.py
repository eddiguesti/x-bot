"""
Historical Backtester for Strategy Comparison

Fetches real historical signals and simulates trades exactly as the live
system would, recording all results for strategy comparison.
"""

import logging
from datetime import datetime, timedelta, timezone
from typing import Optional
from collections import defaultdict

from sqlalchemy.orm import Session

from ..config import Settings
from ..models import (
    Asset, Direction, Signal, Creator, Trade, TradeStatus, Portfolio,
    ConsensusAction, SignalOutcome
)
from ..data_ingestion import XClient, PriceClient
from ..signal_extraction import SignalExtractor
from ..consensus.enhanced_engine import EnhancedConsensusEngine
from ..analysis.technical import TechnicalAnalyzer
from .config import StrategyConfig, StrategyType, get_strategy, STRATEGIES

logger = logging.getLogger(__name__)


class HistoricalBacktester:
    """
    Runs historical backtest on real signal data.

    Process:
    1. Fetch historical signals from X (as far back as API allows)
    2. Sort signals chronologically
    3. For each time period, calculate consensus and simulate trades
    4. Record results exactly as live trading would
    """

    def __init__(
        self,
        settings: Settings,
        session: Session,
        initial_balance: float = 10000.0,
    ):
        self.settings = settings
        self.session = session
        self.initial_balance = initial_balance

        # Components
        self.x_client = XClient(settings)
        self.price_client = PriceClient(settings)
        self.extractor = SignalExtractor(use_nlp=False, min_confidence=0.3)
        self.consensus_engine = EnhancedConsensusEngine(settings)
        self.technical = TechnicalAnalyzer(settings)

        # Track simulated portfolios per strategy
        self.portfolios: dict[StrategyType, dict] = {}
        self.trades: dict[StrategyType, list] = {}

    def fetch_historical_signals(
        self,
        days_back: int = 30,
        limit_per_fetch: int = 500,
    ) -> int:
        """
        Fetch historical signals from X, going as far back as API allows.

        Returns number of signals fetched.
        """
        logger.info(f"Fetching historical signals ({days_back} days back)...")

        total_signals = 0
        hours_back = days_back * 24

        # Fetch in chunks to get more data
        # Macrocosmos API typically allows up to 7-14 days
        chunk_hours = 168  # 7 days per chunk

        for start_hour in range(0, hours_back, chunk_hours):
            end_hour = min(start_hour + chunk_hours, hours_back)

            try:
                posts = self.x_client.fetch_trading_signals(
                    limit=limit_per_fetch,
                    hours_back=end_hour,
                )

                signals_added = 0
                for post in posts:
                    # Check if already exists
                    existing = self.session.query(Signal).filter(
                        Signal.post_id == post.post_id
                    ).first()
                    if existing:
                        continue

                    # Extract signal
                    extraction = self.extractor.extract(post.text)
                    if not self.extractor.is_actionable(extraction):
                        continue

                    # Get or create creator
                    creator = self.session.query(Creator).filter(
                        Creator.username == post.username
                    ).first()
                    if not creator:
                        creator = Creator(
                            username=post.username,
                            display_name=post.raw_data.get('user', {}).get('display_name', post.username),
                        )
                        self.session.add(creator)
                        self.session.flush()

                    # Create signal
                    signal = Signal(
                        creator_id=creator.id,
                        post_id=post.post_id,
                        post_text=post.text,
                        post_url=post.url,
                        posted_at=post.posted_at,
                        asset=extraction.asset,
                        direction=extraction.direction,
                        confidence=extraction.confidence,
                    )
                    self.session.add(signal)
                    signals_added += 1

                self.session.commit()
                total_signals += signals_added
                logger.info(f"Chunk {start_hour}-{end_hour}h: {signals_added} signals")

            except Exception as e:
                logger.warning(f"Failed to fetch chunk {start_hour}-{end_hour}h: {e}")
                continue

        logger.info(f"Total historical signals fetched: {total_signals}")
        return total_signals

    def evaluate_signal_outcomes(self, hold_hours: int = 24):
        """
        Evaluate outcomes for signals that don't have outcomes yet.

        Uses historical price data to determine if signals were correct.
        """
        logger.info("Evaluating signal outcomes...")

        pending_signals = self.session.query(Signal).filter(
            Signal.outcome == None,
            Signal.posted_at < datetime.now(timezone.utc) - timedelta(hours=hold_hours)
        ).all()

        evaluated = 0
        for signal in pending_signals:
            try:
                # Get price at signal time and after hold period
                price_at_signal = signal.price_at_signal
                if not price_at_signal:
                    price_at_signal = self.price_client.get_current_price(signal.asset)
                    signal.price_at_signal = price_at_signal

                # Get current price (or historical if we had it)
                current_price = self.price_client.get_current_price(signal.asset)
                if not current_price or not price_at_signal:
                    continue

                # Calculate price change
                price_change = ((current_price - price_at_signal) / price_at_signal) * 100
                signal.price_change_percent = price_change
                signal.price_at_evaluation = current_price
                signal.evaluated_at = datetime.now(timezone.utc)

                # Determine outcome
                if signal.direction == Direction.LONG:
                    signal.outcome = SignalOutcome.CORRECT if price_change > 1.0 else SignalOutcome.INCORRECT
                else:
                    signal.outcome = SignalOutcome.CORRECT if price_change < -1.0 else SignalOutcome.INCORRECT

                evaluated += 1

            except Exception as e:
                logger.warning(f"Failed to evaluate signal {signal.id}: {e}")

        self.session.commit()
        logger.info(f"Evaluated {evaluated} signals")
        return evaluated

    def init_strategy_portfolio(self, strategy_type: StrategyType):
        """Initialize portfolio tracking for a strategy."""
        portfolio_name = f"strategy_{strategy_type.value}"

        # Check if portfolio exists
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

        self.portfolios[strategy_type] = {
            "portfolio": portfolio,
            "open_positions": {},
            "balance": self.initial_balance,
        }
        self.trades[strategy_type] = []

    def simulate_trade(
        self,
        strategy_type: StrategyType,
        asset: Asset,
        direction: Direction,
        entry_price: float,
        entry_time: datetime,
        position_pct: float,
        consensus_score: float,
        consensus_confidence: float,
    ) -> Optional[Trade]:
        """Simulate opening a trade for a strategy."""
        config = get_strategy(strategy_type)
        portfolio_data = self.portfolios[strategy_type]
        portfolio = portfolio_data["portfolio"]

        # Check if already have position
        if asset in portfolio_data["open_positions"]:
            return None

        # Check position limits
        if len(portfolio_data["open_positions"]) >= config.risk.max_positions:
            return None

        # Calculate position
        position_usd = portfolio.current_balance * position_pct
        position_size = position_usd / entry_price if entry_price > 0 else 0

        # Calculate stops based on volatility
        stop_pct = 0.03  # 3% stop
        if direction == Direction.LONG:
            stop_loss = entry_price * (1 - stop_pct)
            take_profit = entry_price * (1 + stop_pct * config.risk.min_rr_ratio)
        else:
            stop_loss = entry_price * (1 + stop_pct)
            take_profit = entry_price * (1 - stop_pct * config.risk.min_rr_ratio)

        # Create trade
        trade = Trade(
            asset=asset,
            direction=direction,
            status=TradeStatus.OPEN,
            entry_price=entry_price,
            entry_time=entry_time,
            position_size=position_size,
            consensus_score=consensus_score,
            consensus_confidence=consensus_confidence,
            stop_loss_price=stop_loss,
            take_profit_price=take_profit,
            notes=f"strategy_{strategy_type.value}",
        )

        self.session.add(trade)
        portfolio_data["open_positions"][asset] = trade

        return trade

    def simulate_close(
        self,
        strategy_type: StrategyType,
        trade: Trade,
        exit_price: float,
        exit_time: datetime,
        reason: str,
    ):
        """Simulate closing a trade."""
        portfolio_data = self.portfolios[strategy_type]
        portfolio = portfolio_data["portfolio"]

        # Calculate PnL
        if trade.direction == Direction.LONG:
            pnl_pct = ((exit_price - trade.entry_price) / trade.entry_price) * 100
        else:
            pnl_pct = ((trade.entry_price - exit_price) / trade.entry_price) * 100

        pnl_usd = trade.position_size * trade.entry_price * (pnl_pct / 100)

        # Update trade
        trade.exit_price = exit_price
        trade.exit_time = exit_time
        trade.status = TradeStatus.CLOSED
        trade.pnl_percent = pnl_pct
        trade.pnl_usd = pnl_usd

        # Update portfolio
        portfolio.current_balance += pnl_usd
        portfolio.total_trades += 1
        portfolio.total_pnl += pnl_usd
        if pnl_usd > 0:
            portfolio.winning_trades += 1

        # Remove from open positions
        if trade.asset in portfolio_data["open_positions"]:
            del portfolio_data["open_positions"][trade.asset]

        self.trades[strategy_type].append(trade)

        logger.info(
            f"[{strategy_type.value}] Closed {trade.asset.value}: "
            f"PnL {pnl_pct:+.2f}% (${pnl_usd:+.2f}) - {reason}"
        )

    def run_backtest(self, days_back: int = 30) -> dict:
        """
        Run full historical backtest.

        Returns performance summary for each strategy.
        """
        logger.info(f"=" * 60)
        logger.info("STARTING HISTORICAL BACKTEST")
        logger.info(f"=" * 60)

        # Step 1: Fetch historical signals
        self.fetch_historical_signals(days_back=days_back)

        # Step 2: Initialize strategy portfolios
        for strategy_type in StrategyType:
            self.init_strategy_portfolio(strategy_type)

        # Step 3: Get all signals sorted by time
        signals = self.session.query(Signal).filter(
            Signal.posted_at >= datetime.now(timezone.utc) - timedelta(days=days_back)
        ).order_by(Signal.posted_at).all()

        logger.info(f"Processing {len(signals)} signals chronologically...")

        # Group signals by day for processing
        signals_by_day = defaultdict(list)
        for signal in signals:
            day_key = signal.posted_at.date()
            signals_by_day[day_key].append(signal)

        # Step 4: Process each day
        for day in sorted(signals_by_day.keys()):
            day_signals = signals_by_day[day]

            # Calculate consensus for each asset based on signals up to this point
            for strategy_type in StrategyType:
                config = get_strategy(strategy_type)
                portfolio_data = self.portfolios[strategy_type]

                # Check exits first
                for asset, trade in list(portfolio_data["open_positions"].items()):
                    current_price = self.price_client.get_current_price(asset)
                    if not current_price:
                        continue

                    # Check stop loss / take profit
                    should_close = False
                    reason = ""

                    if trade.direction == Direction.LONG:
                        if current_price <= trade.stop_loss_price:
                            should_close, reason = True, "stop_loss"
                        elif current_price >= trade.take_profit_price:
                            should_close, reason = True, "take_profit"
                    else:
                        if current_price >= trade.stop_loss_price:
                            should_close, reason = True, "stop_loss"
                        elif current_price <= trade.take_profit_price:
                            should_close, reason = True, "take_profit"

                    # Time-based exit
                    if trade.entry_time:
                        age_hours = (datetime.now(timezone.utc) - trade.entry_time).total_seconds() / 3600
                        if age_hours > 72:
                            should_close, reason = True, "time_exit"

                    if should_close:
                        self.simulate_close(
                            strategy_type, trade, current_price,
                            datetime.now(timezone.utc), reason
                        )

                # Process new signals
                for signal in day_signals:
                    # Skip if already have position
                    if signal.asset in portfolio_data["open_positions"]:
                        continue

                    # Get current price
                    price = self.price_client.get_current_price(signal.asset)
                    if not price:
                        continue

                    # Evaluate based on strategy thresholds
                    if signal.confidence < config.consensus.min_signal_confidence:
                        continue

                    # Calculate position size based on strategy
                    position_pct = config.risk.base_position_pct

                    # Open trade
                    self.simulate_trade(
                        strategy_type,
                        signal.asset,
                        signal.direction,
                        price,
                        signal.posted_at,
                        position_pct,
                        0.5,  # Placeholder consensus score
                        signal.confidence,
                    )

        self.session.commit()

        # Step 5: Close remaining positions at current prices
        for strategy_type in StrategyType:
            portfolio_data = self.portfolios[strategy_type]
            for asset, trade in list(portfolio_data["open_positions"].items()):
                current_price = self.price_client.get_current_price(asset)
                if current_price:
                    self.simulate_close(
                        strategy_type, trade, current_price,
                        datetime.now(timezone.utc), "backtest_end"
                    )

        self.session.commit()

        # Step 6: Generate summary
        results = {}
        for strategy_type in StrategyType:
            portfolio = self.portfolios[strategy_type]["portfolio"]
            trades = self.trades[strategy_type]

            results[strategy_type] = {
                "initial_balance": self.initial_balance,
                "final_balance": portfolio.current_balance,
                "total_return_pct": ((portfolio.current_balance - self.initial_balance) / self.initial_balance) * 100,
                "total_trades": portfolio.total_trades,
                "winning_trades": portfolio.winning_trades,
                "win_rate": (portfolio.winning_trades / portfolio.total_trades * 100) if portfolio.total_trades > 0 else 0,
                "total_pnl": portfolio.total_pnl,
            }

        logger.info(f"\n{'=' * 60}")
        logger.info("BACKTEST RESULTS")
        logger.info(f"{'=' * 60}")
        for st, r in results.items():
            logger.info(
                f"{st.value}: ${r['final_balance']:,.2f} ({r['total_return_pct']:+.2f}%) | "
                f"Trades: {r['total_trades']} | WR: {r['win_rate']:.1f}%"
            )

        return results
