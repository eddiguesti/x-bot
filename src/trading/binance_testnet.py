"""Binance Testnet integration for paper trading with live order book."""

import logging
import json
from datetime import datetime
from typing import Optional

import ccxt

from ..config import Settings
from ..models import Asset, Direction, Trade, TradeStatus, Portfolio

logger = logging.getLogger(__name__)


class BinanceTestnetTrader:
    """
    Binance Testnet trader for live paper trading.

    Uses Binance's official testnet to place real orders with fake money.
    Get testnet API keys from: https://testnet.binance.vision/

    Features:
    - Real order book and price feeds
    - Actual order execution (with test funds)
    - Full trading interface identical to production
    """

    SYMBOL_MAP = {
        Asset.BTC: "BTC/USDC",
        Asset.ETH: "ETH/USDC",
    }

    def __init__(self, settings: Settings, api_key: str = None, secret: str = None):
        """
        Initialize Binance Testnet connection.

        Args:
            settings: Application settings
            api_key: Testnet API key (or from env BINANCE_TESTNET_API_KEY)
            secret: Testnet secret (or from env BINANCE_TESTNET_SECRET)
        """
        self.settings = settings

        # Use provided keys or fall back to settings
        self.api_key = api_key or getattr(settings, 'binance_testnet_api_key', '')
        self.secret = secret or getattr(settings, 'binance_testnet_secret', '')

        if not self.api_key or not self.secret:
            raise ValueError(
                "Binance Testnet API keys required.\n"
                "Get them from: https://testnet.binance.vision/\n"
                "Add BINANCE_TESTNET_API_KEY and BINANCE_TESTNET_SECRET to .env"
            )

        # Initialize CCXT with testnet
        self.exchange = ccxt.binance({
            'apiKey': self.api_key,
            'secret': self.secret,
            'enableRateLimit': True,
            'options': {
                'defaultType': 'spot',
                'adjustForTimeDifference': True,
            }
        })

        # Set testnet URLs explicitly
        self.exchange.set_sandbox_mode(True)
        # Override with Spot testnet URLs (not Futures)
        self.exchange.urls['api']['public'] = 'https://testnet.binance.vision/api/v3'
        self.exchange.urls['api']['private'] = 'https://testnet.binance.vision/api/v3'

        logger.info("Connected to Binance Testnet")

    def get_balance(self) -> dict:
        """Get current testnet balance."""
        try:
            balance = self.exchange.fetch_balance()
            return {
                'USDC': balance.get('USDC', {}).get('free', 0),
                'BTC': balance.get('BTC', {}).get('free', 0),
                'ETH': balance.get('ETH', {}).get('free', 0),
                'total_usdc': balance.get('USDC', {}).get('total', 0),
            }
        except Exception as e:
            logger.error(f"Error fetching balance: {e}")
            return {}

    def get_price(self, asset: Asset) -> Optional[float]:
        """Get current price for asset."""
        symbol = self.SYMBOL_MAP.get(asset)
        if not symbol:
            return None

        try:
            ticker = self.exchange.fetch_ticker(symbol)
            return ticker['last']
        except Exception as e:
            logger.error(f"Error fetching price for {asset}: {e}")
            return None

    def place_market_order(
        self,
        asset: Asset,
        direction: Direction,
        amount_usdt: float,
    ) -> Optional[dict]:
        """
        Place a market order on testnet.

        Args:
            asset: Asset to trade
            direction: LONG (buy) or SHORT (sell)
            amount_usdc: Amount in USDC to trade

        Returns:
            Order info dict or None if failed
        """
        symbol = self.SYMBOL_MAP.get(asset)
        if not symbol:
            logger.error(f"Unknown asset: {asset}")
            return None

        try:
            price = self.get_price(asset)
            if not price:
                return None

            # Calculate quantity
            quantity = amount_usdt / price

            # Round to appropriate precision
            quantity = round(quantity, 6)

            # Determine side
            side = 'buy' if direction == Direction.LONG else 'sell'

            logger.info(f"Placing {side} order: {quantity} {asset.value} @ ~${price:.2f}")

            # Place order
            order = self.exchange.create_market_order(
                symbol=symbol,
                side=side,
                amount=quantity,
            )

            logger.info(f"Order executed: {order['id']} - {order['status']}")

            return {
                'order_id': order['id'],
                'symbol': symbol,
                'side': side,
                'quantity': quantity,
                'price': order.get('average', price),
                'status': order['status'],
                'filled': order.get('filled', quantity),
                'cost': order.get('cost', amount_usdt),
                'timestamp': datetime.utcnow(),
            }

        except Exception as e:
            logger.error(f"Error placing order: {e}")
            return None

    def open_position(
        self,
        asset: Asset,
        direction: Direction,
        amount_usdt: float,
        consensus_score: float = 0,
        consensus_confidence: float = 0,
        stop_loss_pct: float = 0.02,
        take_profit_pct: float = 0.04,
    ) -> Optional[Trade]:
        """
        Open a new position based on consensus signal.

        Args:
            asset: Asset to trade
            direction: LONG or SHORT
            amount_usdc: Position size in USDC
            consensus_score: Score that triggered the trade
            consensus_confidence: Confidence level
            stop_loss_pct: Stop loss percentage
            take_profit_pct: Take profit percentage

        Returns:
            Trade object or None if failed
        """
        order = self.place_market_order(asset, direction, amount_usdt)
        if not order:
            return None

        entry_price = order['price']

        # Calculate stop loss and take profit
        if direction == Direction.LONG:
            stop_loss = entry_price * (1 - stop_loss_pct)
            take_profit = entry_price * (1 + take_profit_pct)
        else:
            stop_loss = entry_price * (1 + stop_loss_pct)
            take_profit = entry_price * (1 - take_profit_pct)

        trade = Trade(
            asset=asset,
            direction=direction,
            status=TradeStatus.OPEN,
            entry_price=entry_price,
            entry_time=datetime.utcnow(),
            position_size=order['quantity'],
            consensus_score=consensus_score,
            consensus_confidence=consensus_confidence,
            stop_loss_price=stop_loss,
            take_profit_price=take_profit,
        )

        logger.info(
            f"Opened {direction.value} position: {order['quantity']:.6f} {asset.value} "
            f"@ ${entry_price:.2f} | SL: ${stop_loss:.2f} | TP: ${take_profit:.2f}"
        )

        return trade

    def close_position(self, trade: Trade) -> Optional[Trade]:
        """
        Close an open position.

        Args:
            trade: Trade to close

        Returns:
            Updated trade with exit info
        """
        if trade.status != TradeStatus.OPEN:
            logger.warning(f"Trade {trade.id} is already closed")
            return trade

        # Close by trading in opposite direction
        close_direction = Direction.SHORT if trade.direction == Direction.LONG else Direction.LONG

        # For closing, we need to sell what we bought (or buy back what we sold)
        current_price = self.get_price(trade.asset)
        if not current_price:
            return None

        amount_usdt = trade.position_size * current_price

        order = self.place_market_order(trade.asset, close_direction, amount_usdt)
        if not order:
            return None

        # Update trade
        trade.exit_price = order['price']
        trade.exit_time = datetime.utcnow()
        trade.status = TradeStatus.CLOSED

        # Calculate PnL
        if trade.direction == Direction.LONG:
            trade.pnl_percent = ((trade.exit_price - trade.entry_price) / trade.entry_price) * 100
        else:
            trade.pnl_percent = ((trade.entry_price - trade.exit_price) / trade.entry_price) * 100

        trade.pnl_usd = trade.position_size * trade.entry_price * (trade.pnl_percent / 100)

        emoji = "✅" if trade.pnl_percent > 0 else "❌"
        logger.info(
            f"{emoji} Closed {trade.direction.value} position: "
            f"Entry ${trade.entry_price:.2f} -> Exit ${trade.exit_price:.2f} | "
            f"PnL: {trade.pnl_percent:+.2f}% (${trade.pnl_usd:+.2f})"
        )

        return trade

    def check_stop_loss_take_profit(self, trade: Trade) -> Optional[str]:
        """
        Check if trade hit stop loss or take profit.

        Returns:
            'stop_loss', 'take_profit', or None
        """
        if trade.status != TradeStatus.OPEN:
            return None

        current_price = self.get_price(trade.asset)
        if not current_price:
            return None

        if trade.direction == Direction.LONG:
            if current_price <= trade.stop_loss_price:
                return 'stop_loss'
            if current_price >= trade.take_profit_price:
                return 'take_profit'
        else:  # SHORT
            if current_price >= trade.stop_loss_price:
                return 'stop_loss'
            if current_price <= trade.take_profit_price:
                return 'take_profit'

        return None

    def get_open_orders(self) -> list:
        """Get all open orders on testnet."""
        try:
            return self.exchange.fetch_open_orders()
        except Exception as e:
            logger.error(f"Error fetching open orders: {e}")
            return []

    def get_recent_trades(self, symbol: str = None, limit: int = 10) -> list:
        """Get recent trades from testnet."""
        try:
            if symbol:
                return self.exchange.fetch_my_trades(symbol, limit=limit)

            # Get trades for all symbols
            all_trades = []
            for asset_symbol in self.SYMBOL_MAP.values():
                trades = self.exchange.fetch_my_trades(asset_symbol, limit=limit)
                all_trades.extend(trades)

            return sorted(all_trades, key=lambda t: t['timestamp'], reverse=True)[:limit]
        except Exception as e:
            logger.error(f"Error fetching trades: {e}")
            return []

    def display_status(self):
        """Display current testnet account status."""
        balance = self.get_balance()
        btc_price = self.get_price(Asset.BTC)
        eth_price = self.get_price(Asset.ETH)

        print("\n" + "=" * 60)
        print("🔬 BINANCE TESTNET STATUS")
        print("=" * 60)

        print("\n💰 BALANCE:")
        print(f"   USDC: ${balance.get('USDC', 0):,.2f}")
        print(f"   BTC:  {balance.get('BTC', 0):.6f}")
        print(f"   ETH:  {balance.get('ETH', 0):.6f}")

        print("\n📊 CURRENT PRICES:")
        if btc_price:
            print(f"   BTC/USDC: ${btc_price:,.2f}")
        if eth_price:
            print(f"   ETH/USDC: ${eth_price:,.2f}")

        # Calculate total portfolio value
        total = balance.get('USDC', 0)
        if btc_price and balance.get('BTC', 0):
            total += balance.get('BTC', 0) * btc_price
        if eth_price and balance.get('ETH', 0):
            total += balance.get('ETH', 0) * eth_price

        print(f"\n📈 TOTAL PORTFOLIO VALUE: ${total:,.2f}")
