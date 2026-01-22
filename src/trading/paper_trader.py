"""Local paper trading without exchange connection."""

import logging
from datetime import datetime
from typing import Optional

from sqlalchemy.orm import Session

from ..config import Settings
from ..models import Asset, Direction, Trade, TradeStatus, Portfolio
from ..data_ingestion.price_client import PriceClient

logger = logging.getLogger(__name__)


class PaperTrader:
    """
    Local paper trader using real prices but no exchange connection.

    Simpler alternative to Binance Testnet - tracks trades locally
    and uses real-time Binance prices for execution.
    """

    def __init__(self, settings: Settings, session: Session):
        self.settings = settings
        self.session = session
        self.price_client = PriceClient(settings)

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

    def open_position(
        self,
        asset: Asset,
        direction: Direction,
        position_pct: float = None,
        consensus_score: float = 0,
        consensus_confidence: float = 0,
    ) -> Optional[Trade]:
        """
        Open a new paper position.

        Args:
            asset: Asset to trade
            direction: LONG or SHORT
            position_pct: Percentage of portfolio to use (default from settings)
            consensus_score: Score that triggered the trade
            consensus_confidence: Confidence level

        Returns:
            Trade object or None if failed
        """
        # Check if we already have an open position for this asset
        existing = self.session.query(Trade).filter(
            Trade.asset == asset,
            Trade.status == TradeStatus.OPEN,
        ).first()

        if existing:
            logger.warning(f"Already have open {asset.value} position")
            return None

        # Get current price
        price = self.get_price(asset)
        if not price:
            logger.error(f"Could not get price for {asset}")
            return None

        # Calculate position size
        if position_pct is None:
            position_pct = self.portfolio.max_position_pct

        position_usd = self.portfolio.current_balance * position_pct
        position_size = position_usd / price

        # Calculate stop loss and take profit
        if direction == Direction.LONG:
            stop_loss = price * (1 - self.portfolio.stop_loss_pct)
            take_profit = price * (1 + self.portfolio.take_profit_pct)
        else:
            stop_loss = price * (1 + self.portfolio.stop_loss_pct)
            take_profit = price * (1 - self.portfolio.take_profit_pct)

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
            f"(${position_usd:,.2f}) | SL: ${stop_loss:,.2f} | TP: ${take_profit:,.2f}"
        )

        return trade

    def close_position(self, trade: Trade, reason: str = "manual") -> Optional[Trade]:
        """
        Close an open position.

        Args:
            trade: Trade to close
            reason: Reason for closing (manual, stop_loss, take_profit, signal)

        Returns:
            Updated trade with exit info
        """
        if trade.status != TradeStatus.OPEN:
            logger.warning(f"Trade {trade.id} is already closed")
            return trade

        # Get current price
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

    def check_positions(self) -> list[tuple[Trade, str]]:
        """
        Check all open positions for stop loss / take profit.

        Returns:
            List of (trade, trigger_reason) tuples for positions to close
        """
        to_close = []

        open_trades = self.session.query(Trade).filter(
            Trade.status == TradeStatus.OPEN
        ).all()

        for trade in open_trades:
            price = self.get_price(trade.asset)
            if not price:
                continue

            if trade.direction == Direction.LONG:
                if price <= trade.stop_loss_price:
                    to_close.append((trade, 'stop_loss'))
                elif price >= trade.take_profit_price:
                    to_close.append((trade, 'take_profit'))
            else:  # SHORT
                if price >= trade.stop_loss_price:
                    to_close.append((trade, 'stop_loss'))
                elif price <= trade.take_profit_price:
                    to_close.append((trade, 'take_profit'))

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
        """Display current portfolio status."""
        stats = self.get_portfolio_stats()
        open_positions = self.get_open_positions()

        print("\n" + "=" * 60)
        print("📊 PAPER TRADING PORTFOLIO")
        print("=" * 60)

        print(f"\n💰 BALANCE:")
        print(f"   Initial:     ${stats['initial_balance']:,.2f}")
        print(f"   Current:     ${stats['current_balance']:,.2f}")
        print(f"   Unrealized:  ${stats['unrealized_pnl']:+,.2f}")
        print(f"   Total Equity: ${stats['total_equity']:,.2f}")

        print(f"\n📈 PERFORMANCE:")
        print(f"   Total PnL:   ${stats['total_pnl']:+,.2f}")
        print(f"   Return:      {stats['return_pct']:+.2f}%")
        print(f"   Win Rate:    {stats['win_rate']:.1%} ({stats['winning_trades']}/{stats['total_trades']})")

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
                    print(
                        f"   {emoji} {trade.direction.value.upper()} {trade.asset.value}: "
                        f"Entry ${trade.entry_price:,.2f} | Now ${price:,.2f} | "
                        f"PnL: {pnl_pct:+.2f}%"
                    )
        else:
            print("\n   No open positions")
