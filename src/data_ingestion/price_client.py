"""Price data client using CCXT for exchange data."""

import logging
from datetime import datetime, timedelta
from typing import Optional

import ccxt
import pandas as pd

from ..config import Settings
from ..models import Asset

logger = logging.getLogger(__name__)


class PriceClient:
    """Client for fetching cryptocurrency price data via CCXT."""

    # Map our Asset enum to exchange symbols
    SYMBOL_MAP = {
        Asset.BTC: "BTC/USDC",
        Asset.ETH: "ETH/USDC",
        Asset.SOL: "SOL/USDC",
        Asset.XRP: "XRP/USDC",
        Asset.DOGE: "DOGE/USDC",
        Asset.ADA: "ADA/USDC",
        Asset.AVAX: "AVAX/USDC",
        Asset.LINK: "LINK/USDC",
        Asset.DOT: "DOT/USDC",
        Asset.SHIB: "SHIB/USDC",
        Asset.LTC: "LTC/USDC",
        Asset.UNI: "UNI/USDC",
        Asset.ATOM: "ATOM/USDC",
        Asset.ARB: "ARB/USDC",
        Asset.OP: "OP/USDC",
        Asset.APT: "APT/USDC",
        Asset.NEAR: "NEAR/USDC",
        Asset.INJ: "INJ/USDC",
        Asset.TAO: "TAO/USDC",
    }

    def __init__(self, settings: Settings):
        self.settings = settings

        # Initialize Binance exchange
        exchange_config = {
            'enableRateLimit': True,
            'options': {
                'defaultType': 'spot',
            }
        }

        # Add API keys if provided (for higher rate limits)
        if settings.binance_api_key:
            exchange_config['apiKey'] = settings.binance_api_key
            exchange_config['secret'] = settings.binance_secret_key

        self.exchange = ccxt.binance(exchange_config)

    def get_current_price(self, asset: Asset) -> Optional[float]:
        """
        Get current price for an asset.

        Args:
            asset: Asset to get price for

        Returns:
            Current price in USDC, or None if error
        """
        symbol = self.SYMBOL_MAP.get(asset)
        if not symbol:
            logger.error(f"Unknown asset: {asset}")
            return None

        try:
            ticker = self.exchange.fetch_ticker(symbol)
            return ticker['last']
        except Exception as e:
            logger.error(f"Error fetching price for {asset}: {e}")
            return None

    def get_price_at_time(
        self,
        asset: Asset,
        timestamp: datetime,
    ) -> Optional[float]:
        """
        Get price at a specific time (closest candle close).

        Args:
            asset: Asset to get price for
            timestamp: Target timestamp

        Returns:
            Price at that time (candle close), or None if error
        """
        symbol = self.SYMBOL_MAP.get(asset)
        if not symbol:
            return None

        try:
            # Fetch 1h candles around the timestamp
            since = int((timestamp - timedelta(hours=1)).timestamp() * 1000)

            candles = self.exchange.fetch_ohlcv(
                symbol,
                timeframe='1h',
                since=since,
                limit=3,
            )

            if not candles:
                return None

            # Find closest candle
            target_ts = timestamp.timestamp() * 1000
            closest = min(candles, key=lambda c: abs(c[0] - target_ts))

            # Return close price (index 4)
            return closest[4]

        except Exception as e:
            logger.error(f"Error fetching historical price for {asset}: {e}")
            return None

    def get_price_change(
        self,
        asset: Asset,
        start_time: datetime,
        end_time: datetime,
    ) -> Optional[tuple[float, float, float]]:
        """
        Calculate price change between two times.

        Args:
            asset: Asset to calculate for
            start_time: Start timestamp
            end_time: End timestamp

        Returns:
            Tuple of (start_price, end_price, percent_change) or None if error
        """
        start_price = self.get_price_at_time(asset, start_time)
        end_price = self.get_price_at_time(asset, end_time)

        if start_price is None or end_price is None:
            return None

        percent_change = ((end_price - start_price) / start_price) * 100
        return (start_price, end_price, percent_change)

    def get_ohlcv(
        self,
        asset: Asset,
        timeframe: str = '1h',
        limit: int = 100,
        since: Optional[datetime] = None,
    ) -> Optional[pd.DataFrame]:
        """
        Get OHLCV candle data.

        Args:
            asset: Asset to fetch
            timeframe: Candle timeframe ('1m', '5m', '1h', '4h', '1d')
            limit: Number of candles
            since: Start time (optional)

        Returns:
            DataFrame with columns [timestamp, open, high, low, close, volume]
        """
        symbol = self.SYMBOL_MAP.get(asset)
        if not symbol:
            return None

        try:
            since_ms = int(since.timestamp() * 1000) if since else None

            candles = self.exchange.fetch_ohlcv(
                symbol,
                timeframe=timeframe,
                since=since_ms,
                limit=limit,
            )

            if not candles:
                return None

            df = pd.DataFrame(
                candles,
                columns=['timestamp', 'open', 'high', 'low', 'close', 'volume']
            )
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df['asset'] = asset.value

            return df

        except Exception as e:
            logger.error(f"Error fetching OHLCV for {asset}: {e}")
            return None

    def get_prices_for_signals(
        self,
        signals: list[tuple[Asset, datetime]],
    ) -> dict[tuple[Asset, datetime], float]:
        """
        Batch fetch prices for multiple signal timestamps.

        Args:
            signals: List of (asset, timestamp) tuples

        Returns:
            Dict mapping (asset, timestamp) to price
        """
        results = {}

        for asset, timestamp in signals:
            price = self.get_price_at_time(asset, timestamp)
            if price is not None:
                results[(asset, timestamp)] = price

        return results
