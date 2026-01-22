"""Smart paper trader with technical confirmation and risk management."""

import logging
from datetime import datetime
from typing import Optional

from sqlalchemy.orm import Session

from ..config import Settings
from ..models import Asset, Direction, Trade, TradeStatus, Portfolio, ConsensusResult, ConsensusAction
from ..data_ingestion.price_client import PriceClient
from ..analysis.technical import TechnicalAnalyzer, MarketRegime
from .risk_manager import RiskManager

logger = logging.getLogger(__name__)


class SmartTrader:
    """
    Enhanced paper trader with:
    1. Technical analysis confirmation
    2. Volatility-adjusted position sizing
    3. Kelly Criterion sizing
    4. Trailing stops
    5. Market regime awareness
    6. Drawdown protection
    """

    def __init__(self, settings: Settings, session: Session):
        self.settings = settings
        self.session = session
        self.price_client = PriceClient(settings)
        self.technical = TechnicalAnalyzer(settings)
        self.risk_manager = RiskManager(settings, session)

        # Get or create portfolio
        self.portfolio = self._get_or_create_portfolio()

    def _get_or_create_portfolio(self) -> Portfolio:
        """Get existing portfolio or create new one."""
        portfolio = self.session.query(Portfolio).filter(
            Portfolio.name == "default"
        ).first()

        if not portfolio:
            portfolio = Portfolio(
                name="default",
                initial_balance=10000.0,
                current_balance=10000.0,
            )
            self.session.add(portfolio)
            self.session.commit()
            logger.info("Created new paper trading portfolio with $10,000")

        return portfolio

    def get_price(self, asset: Asset) -> Optional[float]:
        """Get current price for asset."""
        return self.price_client.get_current_price(asset)

    def evaluate_trade(
        self,
        asset: Asset,
        consensus: ConsensusResult,
    ) -> dict:
        """
        Evaluate whether to take a trade based on consensus + technicals.

        Returns dict with decision and reasoning.
        """
        result = {
            'should_trade': False,
            'direction': None,
            'position_size': 0,
            'confidence': 0,
            'reasons': [],
            'warnings': [],
        }

        # Check if consensus indicates a trade
        if consensus.action == ConsensusAction.NO_TRADE:
            result['reasons'].append("No consensus signal")
            return result

        direction = Direction.LONG if consensus.action == ConsensusAction.LONG else Direction.SHORT

        # Get technical analysis
        tech_signal = self.technical.analyze(asset)
        if not tech_signal:
            result['warnings'].append("Could not get technical analysis")
            # Continue anyway with reduced confidence
            tech_confirms = True
            tech_score = 0
        else:
            tech_score = tech_signal.confirmation_score

            # Check technical confirmation - STRICT: require alignment, not just "not against"
            if direction == Direction.LONG:
                tech_confirms = tech_signal.confirmation_score > 0.0  # Must be bullish
            else:
                tech_confirms = tech_signal.confirmation_score < 0.0  # Must be bearish

            # Check market regime
            if tech_signal.regime == MarketRegime.HIGH_VOLATILITY:
                result['warnings'].append(f"High volatility regime (ATR: {tech_signal.atr_percent:.1f}%)")

            # Check for extreme RSI
            if tech_signal.rsi_signal == "overbought" and direction == Direction.LONG:
                result['warnings'].append(f"RSI overbought ({tech_signal.rsi:.0f})")
            elif tech_signal.rsi_signal == "oversold" and direction == Direction.SHORT:
                result['warnings'].append(f"RSI oversold ({tech_signal.rsi:.0f})")

        # Check risk metrics
        risk_metrics = self.risk_manager.get_risk_metrics()
        if not risk_metrics.can_trade:
            result['reasons'].append(risk_metrics.block_reason)
            return result

        # Check if we already have a position
        existing = self.session.query(Trade).filter(
            Trade.asset == asset,
            Trade.status == TradeStatus.OPEN,
        ).first()
        if existing:
            result['reasons'].append(f"Already have {asset.value} position")
            return result

        # Decide whether to trade
        if not tech_confirms:
            result['reasons'].append(f"Technicals don't confirm (score: {tech_score:.2f})")
            return result

        # Calculate position size
        price = self.get_price(asset)
        if not price:
            result['reasons'].append("Could not get price")
            return result

        position_result = self.risk_manager.calculate_position_size(
            asset=asset,
            direction=direction,
            consensus_confidence=consensus.confidence,
            entry_price=price,
        )

        if position_result.position_pct <= 0:
            result['reasons'].append(position_result.rationale)
            return result

        # Trade approved
        result['should_trade'] = True
        result['direction'] = direction
        result['position_size'] = position_result.position_pct
        result['confidence'] = consensus.confidence
        result['reasons'].append(f"Consensus: {consensus.long_votes}L/{consensus.short_votes}S")
        result['reasons'].append(f"Technical score: {tech_score:.2f}")
        result['reasons'].append(f"Position: {position_result.rationale}")

        return result

    def open_position(
        self,
        asset: Asset,
        direction: Direction,
        consensus_score: float = 0,
        consensus_confidence: float = 0,
    ) -> Optional[Trade]:
        """
        Open a new position with smart sizing and dynamic stops.
        """
        # Check if already have position
        existing = self.session.query(Trade).filter(
            Trade.asset == asset,
            Trade.status == TradeStatus.OPEN,
        ).first()
        if existing:
            logger.warning(f"Already have open {asset.value} position")
            return None

        # Get price
        price = self.get_price(asset)
        if not price:
            logger.error(f"Could not get price for {asset}")
            return None

        # Calculate position size using risk manager
        position_result = self.risk_manager.calculate_position_size(
            asset=asset,
            direction=direction,
            consensus_confidence=consensus_confidence,
            entry_price=price,
        )

        if position_result.position_pct <= 0:
            logger.warning(f"Position size 0: {position_result.rationale}")
            return None

        position_usd = self.portfolio.current_balance * position_result.position_pct
        position_size = position_usd / price

        # Calculate dynamic stops based on ATR
        stop_loss, take_profit = self.technical.get_dynamic_stops(
            asset, direction.value, price
        )

        # Create trade
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
        )

        self.session.add(trade)
        self.session.commit()

        logger.info(
            f"📈 Opened {direction.value.upper()} {asset.value}: "
            f"{position_size:.6f} @ ${price:,.2f} "
            f"(${position_usd:,.2f} / {position_result.position_pct:.1%}) | "
            f"SL: ${stop_loss:,.2f} | TP: ${take_profit:,.2f} | "
            f"Risk: {position_result.risk_score:.0%}"
        )

        return trade

    def close_position(self, trade: Trade, reason: str = "manual") -> Optional[Trade]:
        """Close an open position."""
        if trade.status != TradeStatus.OPEN:
            logger.warning(f"Trade {trade.id} is already closed")
            return trade

        price = self.get_price(trade.asset)
        if not price:
            logger.error(f"Could not get price for {trade.asset}")
            return None

        # Update trade
        trade.exit_price = price
        trade.exit_time = datetime.utcnow()
        trade.status = TradeStatus.CLOSED

        # Calculate PnL
        if trade.direction == Direction.LONG:
            trade.pnl_percent = ((price - trade.entry_price) / trade.entry_price) * 100
        else:
            trade.pnl_percent = ((trade.entry_price - price) / trade.entry_price) * 100

        trade.pnl_usd = trade.position_size * trade.entry_price * (trade.pnl_percent / 100)

        # Update portfolio
        self.portfolio.current_balance += trade.pnl_usd
        self.portfolio.total_trades += 1
        self.portfolio.total_pnl += trade.pnl_usd
        if trade.pnl_usd > 0:
            self.portfolio.winning_trades += 1

        self.session.commit()

        emoji = "✅" if trade.pnl_percent > 0 else "❌"
        logger.info(
            f"{emoji} Closed {trade.direction.value.upper()} {trade.asset.value} ({reason}): "
            f"${trade.entry_price:,.2f} -> ${price:,.2f} | "
            f"PnL: {trade.pnl_percent:+.2f}% (${trade.pnl_usd:+.2f})"
        )

        return trade

    def update_trailing_stops(self):
        """Update trailing stops for all open positions."""
        open_trades = self.session.query(Trade).filter(
            Trade.status == TradeStatus.OPEN
        ).all()

        for trade in open_trades:
            price = self.get_price(trade.asset)
            if not price:
                continue

            # Only trail if in profit
            if trade.direction == Direction.LONG:
                pnl_pct = (price - trade.entry_price) / trade.entry_price
                if pnl_pct > 0.01:  # At least 1% profit
                    new_stop = self.risk_manager.calculate_trailing_stop(trade, price)
                    if new_stop > trade.stop_loss_price:
                        old_stop = trade.stop_loss_price
                        trade.stop_loss_price = new_stop
                        logger.info(
                            f"📊 Trailing stop {trade.asset.value}: "
                            f"${old_stop:,.2f} -> ${new_stop:,.2f}"
                        )
            else:
                pnl_pct = (trade.entry_price - price) / trade.entry_price
                if pnl_pct > 0.01:
                    new_stop = self.risk_manager.calculate_trailing_stop(trade, price)
                    if new_stop < trade.stop_loss_price:
                        old_stop = trade.stop_loss_price
                        trade.stop_loss_price = new_stop
                        logger.info(
                            f"📊 Trailing stop {trade.asset.value}: "
                            f"${old_stop:,.2f} -> ${new_stop:,.2f}"
                        )

        self.session.commit()

    def check_positions(self, consensus_map: dict = None) -> list[tuple[Trade, str]]:
        """
        Check all open positions for exit conditions.

        Returns list of (trade, reason) tuples to close.
        """
        to_close = []

        open_trades = self.session.query(Trade).filter(
            Trade.status == TradeStatus.OPEN
        ).all()

        for trade in open_trades:
            price = self.get_price(trade.asset)
            if not price:
                continue

            # Get consensus direction if available
            consensus_dir = None
            if consensus_map and trade.asset in consensus_map:
                consensus = consensus_map[trade.asset]
                if consensus.action == ConsensusAction.LONG:
                    consensus_dir = Direction.LONG
                elif consensus.action == ConsensusAction.SHORT:
                    consensus_dir = Direction.SHORT

            # Check if should close
            should_close, reason = self.risk_manager.should_close_position(
                trade, price, consensus_dir
            )

            if should_close:
                to_close.append((trade, reason))

        return to_close

    def get_open_positions(self) -> list[Trade]:
        """Get all open positions."""
        return self.session.query(Trade).filter(
            Trade.status == TradeStatus.OPEN
        ).all()

    def get_portfolio_stats(self) -> dict:
        """Get portfolio statistics."""
        open_trades = self.get_open_positions()

        # Calculate unrealized PnL
        unrealized_pnl = 0
        for trade in open_trades:
            price = self.get_price(trade.asset)
            if price:
                if trade.direction == Direction.LONG:
                    pnl_pct = ((price - trade.entry_price) / trade.entry_price)
                else:
                    pnl_pct = ((trade.entry_price - price) / trade.entry_price)
                unrealized_pnl += trade.position_size * trade.entry_price * pnl_pct

        win_rate = 0
        if self.portfolio.total_trades > 0:
            win_rate = self.portfolio.winning_trades / self.portfolio.total_trades

        return {
            'initial_balance': self.portfolio.initial_balance,
            'current_balance': self.portfolio.current_balance,
            'total_pnl': self.portfolio.total_pnl,
            'unrealized_pnl': unrealized_pnl,
            'total_equity': self.portfolio.current_balance + unrealized_pnl,
            'total_trades': self.portfolio.total_trades,
            'winning_trades': self.portfolio.winning_trades,
            'win_rate': win_rate,
            'return_pct': ((self.portfolio.current_balance - self.portfolio.initial_balance) /
                          self.portfolio.initial_balance) * 100,
            'open_positions': len(open_trades),
        }

    def display_status(self):
        """Display current portfolio and risk status."""
        stats = self.get_portfolio_stats()
        open_positions = self.get_open_positions()

        print("\n" + "=" * 60)
        print("📊 SMART TRADER STATUS")
        print("=" * 60)

        print(f"\n💰 PORTFOLIO:")
        print(f"   Initial:      ${stats['initial_balance']:,.2f}")
        print(f"   Current:      ${stats['current_balance']:,.2f}")
        print(f"   Unrealized:   ${stats['unrealized_pnl']:+,.2f}")
        print(f"   Total Equity: ${stats['total_equity']:,.2f}")

        print(f"\n📈 PERFORMANCE:")
        print(f"   Total PnL:    ${stats['total_pnl']:+,.2f}")
        print(f"   Return:       {stats['return_pct']:+.2f}%")
        print(f"   Win Rate:     {stats['win_rate']:.1%} ({stats['winning_trades']}/{stats['total_trades']})")

        if open_positions:
            print(f"\n🔓 OPEN POSITIONS ({len(open_positions)}):")
            for trade in open_positions:
                price = self.get_price(trade.asset)
                if price:
                    if trade.direction == Direction.LONG:
                        pnl_pct = ((price - trade.entry_price) / trade.entry_price) * 100
                    else:
                        pnl_pct = ((trade.entry_price - price) / trade.entry_price) * 100
                    emoji = "🟢" if pnl_pct > 0 else "🔴"

                    # Calculate distance to SL/TP
                    if trade.direction == Direction.LONG:
                        sl_dist = ((price - trade.stop_loss_price) / price) * 100
                        tp_dist = ((trade.take_profit_price - price) / price) * 100
                    else:
                        sl_dist = ((trade.stop_loss_price - price) / price) * 100
                        tp_dist = ((price - trade.take_profit_price) / price) * 100

                    print(
                        f"   {emoji} {trade.direction.value.upper()} {trade.asset.value}: "
                        f"${trade.entry_price:,.2f} -> ${price:,.2f} | "
                        f"PnL: {pnl_pct:+.2f}% | "
                        f"SL: {sl_dist:.1f}% | TP: {tp_dist:.1f}%"
                    )
        else:
            print("\n   No open positions")

        # Show risk status
        self.risk_manager.display_risk_status()
