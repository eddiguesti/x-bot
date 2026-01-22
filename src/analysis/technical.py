"""Technical analysis module for trade confirmation."""

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from typing import Optional

import ccxt
import numpy as np

from ..config import Settings
from ..models import Asset

logger = logging.getLogger(__name__)


class MarketRegime(str, Enum):
    """Market regime classification."""
    STRONG_BULL = "strong_bull"
    BULL = "bull"
    NEUTRAL = "neutral"
    BEAR = "bear"
    STRONG_BEAR = "strong_bear"
    HIGH_VOLATILITY = "high_volatility"


class TrendDirection(str, Enum):
    """Trend direction."""
    UP = "up"
    DOWN = "down"
    SIDEWAYS = "sideways"


@dataclass
class TechnicalSignal:
    """Technical analysis signal."""
    asset: Asset
    timestamp: datetime

    # Trend indicators
    trend: TrendDirection
    trend_strength: float  # 0-1

    # Momentum indicators
    rsi: float
    rsi_signal: str  # oversold, neutral, overbought
    macd_histogram: float
    macd_signal: str  # bullish, neutral, bearish

    # Volatility
    atr: float
    atr_percent: float
    bollinger_position: float  # -1 to 1 (lower band to upper band)

    # Volume
    volume_ratio: float  # vs 20-period average

    # Overall
    regime: MarketRegime
    confirmation_score: float  # -1 (bearish) to +1 (bullish)

    def confirms_long(self, min_score: float = 0.2) -> bool:
        """Check if technicals confirm a long trade."""
        return self.confirmation_score >= min_score

    def confirms_short(self, min_score: float = 0.2) -> bool:
        """Check if technicals confirm a short trade."""
        return self.confirmation_score <= -min_score


class TechnicalAnalyzer:
    """
    Technical analysis for trade confirmation.

    Uses RSI, MACD, Bollinger Bands, and ATR to:
    1. Confirm social signals aren't against the trend
    2. Detect overbought/oversold conditions
    3. Measure volatility for position sizing
    4. Identify market regime
    """

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
        self.exchange = ccxt.binance({'enableRateLimit': True})
        self._cache = {}
        self._cache_ttl = 60  # seconds

    def get_ohlcv(self, asset: Asset, timeframe: str = '1h', limit: int = 100) -> Optional[np.ndarray]:
        """Fetch OHLCV data for an asset."""
        symbol = self.SYMBOL_MAP.get(asset)
        if not symbol:
            return None

        cache_key = f"{symbol}_{timeframe}_{limit}"
        cached = self._cache.get(cache_key)
        if cached and (datetime.now() - cached['time']).seconds < self._cache_ttl:
            return cached['data']

        try:
            ohlcv = self.exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
            data = np.array(ohlcv, dtype=float)
            self._cache[cache_key] = {'data': data, 'time': datetime.now()}
            return data
        except Exception as e:
            logger.error(f"Error fetching OHLCV for {asset}: {e}")
            return None

    def calculate_rsi(self, closes: np.ndarray, period: int = 14) -> float:
        """Calculate RSI (Relative Strength Index)."""
        if len(closes) < period + 1:
            return 50.0

        deltas = np.diff(closes)
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)

        avg_gain = np.mean(gains[-period:])
        avg_loss = np.mean(losses[-period:])

        if avg_loss == 0:
            return 100.0

        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

    def calculate_macd(
        self,
        closes: np.ndarray,
        fast: int = 12,
        slow: int = 26,
        signal: int = 9
    ) -> tuple[float, float, float]:
        """Calculate MACD (line, signal, histogram)."""
        if len(closes) < slow + signal:
            return 0.0, 0.0, 0.0

        # EMA calculation
        def ema(data, period):
            alpha = 2 / (period + 1)
            result = np.zeros_like(data)
            result[0] = data[0]
            for i in range(1, len(data)):
                result[i] = alpha * data[i] + (1 - alpha) * result[i-1]
            return result

        ema_fast = ema(closes, fast)
        ema_slow = ema(closes, slow)
        macd_line = ema_fast - ema_slow
        signal_line = ema(macd_line, signal)
        histogram = macd_line - signal_line

        return macd_line[-1], signal_line[-1], histogram[-1]

    def calculate_bollinger(
        self,
        closes: np.ndarray,
        period: int = 20,
        std_dev: float = 2.0
    ) -> tuple[float, float, float, float]:
        """Calculate Bollinger Bands (upper, middle, lower, position)."""
        if len(closes) < period:
            return 0.0, 0.0, 0.0, 0.0

        middle = np.mean(closes[-period:])
        std = np.std(closes[-period:])
        upper = middle + std_dev * std
        lower = middle - std_dev * std

        current = closes[-1]
        if upper == lower:
            position = 0.0
        else:
            position = (current - lower) / (upper - lower) * 2 - 1  # -1 to +1

        return upper, middle, lower, position

    def calculate_atr(self, ohlcv: np.ndarray, period: int = 14) -> tuple[float, float]:
        """Calculate ATR (Average True Range) and ATR%."""
        if len(ohlcv) < period + 1:
            return 0.0, 0.0

        highs = ohlcv[:, 2]
        lows = ohlcv[:, 3]
        closes = ohlcv[:, 4]

        tr = np.maximum(
            highs[1:] - lows[1:],
            np.maximum(
                np.abs(highs[1:] - closes[:-1]),
                np.abs(lows[1:] - closes[:-1])
            )
        )

        atr = np.mean(tr[-period:])
        atr_percent = (atr / closes[-1]) * 100

        return atr, atr_percent

    def calculate_trend(self, closes: np.ndarray) -> tuple[TrendDirection, float]:
        """
        Calculate trend direction and strength using multiple EMAs.
        """
        if len(closes) < 50:
            return TrendDirection.SIDEWAYS, 0.0

        # Calculate EMAs
        def ema(data, period):
            alpha = 2 / (period + 1)
            result = np.zeros_like(data)
            result[0] = data[0]
            for i in range(1, len(data)):
                result[i] = alpha * data[i] + (1 - alpha) * result[i-1]
            return result

        ema_9 = ema(closes, 9)[-1]
        ema_21 = ema(closes, 21)[-1]
        ema_50 = ema(closes, 50)[-1]
        current = closes[-1]

        # Trend scoring
        score = 0
        if current > ema_9: score += 1
        if current > ema_21: score += 1
        if current > ema_50: score += 1
        if ema_9 > ema_21: score += 1
        if ema_21 > ema_50: score += 1

        # Strength based on spread between EMAs
        spread = abs(ema_9 - ema_50) / ema_50
        strength = min(1.0, spread * 10)  # Normalize

        if score >= 4:
            return TrendDirection.UP, strength
        elif score <= 1:
            return TrendDirection.DOWN, strength
        else:
            return TrendDirection.SIDEWAYS, strength * 0.5

    def detect_regime(
        self,
        trend: TrendDirection,
        trend_strength: float,
        rsi: float,
        atr_percent: float
    ) -> MarketRegime:
        """Detect current market regime."""
        # High volatility overrides other regimes
        if atr_percent > 5.0:
            return MarketRegime.HIGH_VOLATILITY

        if trend == TrendDirection.UP:
            if trend_strength > 0.6 and rsi > 60:
                return MarketRegime.STRONG_BULL
            return MarketRegime.BULL
        elif trend == TrendDirection.DOWN:
            if trend_strength > 0.6 and rsi < 40:
                return MarketRegime.STRONG_BEAR
            return MarketRegime.BEAR
        else:
            return MarketRegime.NEUTRAL

    def analyze(self, asset: Asset) -> Optional[TechnicalSignal]:
        """
        Perform full technical analysis on an asset.

        Returns TechnicalSignal with all indicators and confirmation score.
        """
        ohlcv = self.get_ohlcv(asset, '1h', 100)
        if ohlcv is None or len(ohlcv) < 50:
            return None

        closes = ohlcv[:, 4]
        volumes = ohlcv[:, 5]

        # Calculate all indicators
        rsi = self.calculate_rsi(closes)
        macd_line, macd_signal, macd_hist = self.calculate_macd(closes)
        bb_upper, bb_middle, bb_lower, bb_position = self.calculate_bollinger(closes)
        atr, atr_percent = self.calculate_atr(ohlcv)
        trend, trend_strength = self.calculate_trend(closes)

        # Volume analysis
        avg_volume = np.mean(volumes[-20:])
        current_volume = volumes[-1]
        volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1.0

        # RSI signal
        if rsi < 30:
            rsi_signal = "oversold"
        elif rsi > 70:
            rsi_signal = "overbought"
        else:
            rsi_signal = "neutral"

        # MACD signal
        if macd_hist > 0 and macd_line > macd_signal:
            macd_sig = "bullish"
        elif macd_hist < 0 and macd_line < macd_signal:
            macd_sig = "bearish"
        else:
            macd_sig = "neutral"

        # Market regime
        regime = self.detect_regime(trend, trend_strength, rsi, atr_percent)

        # Calculate confirmation score (-1 to +1)
        score = 0.0

        # Trend contribution (40%)
        if trend == TrendDirection.UP:
            score += 0.4 * trend_strength
        elif trend == TrendDirection.DOWN:
            score -= 0.4 * trend_strength

        # RSI contribution (20%) - contrarian for extremes
        if rsi < 30:
            score += 0.2  # Oversold = bullish
        elif rsi > 70:
            score -= 0.2  # Overbought = bearish
        elif rsi > 50:
            score += 0.1 * ((rsi - 50) / 20)
        else:
            score -= 0.1 * ((50 - rsi) / 20)

        # MACD contribution (25%)
        if macd_sig == "bullish":
            score += 0.25
        elif macd_sig == "bearish":
            score -= 0.25

        # Bollinger position (15%)
        score -= 0.15 * bb_position  # Near upper band = bearish, lower = bullish

        return TechnicalSignal(
            asset=asset,
            timestamp=datetime.utcnow(),
            trend=trend,
            trend_strength=trend_strength,
            rsi=rsi,
            rsi_signal=rsi_signal,
            macd_histogram=macd_hist,
            macd_signal=macd_sig,
            atr=atr,
            atr_percent=atr_percent,
            bollinger_position=bb_position,
            volume_ratio=volume_ratio,
            regime=regime,
            confirmation_score=max(-1.0, min(1.0, score)),
        )

    def get_volatility_multiplier(self, asset: Asset) -> float:
        """
        Get volatility-adjusted position size multiplier.

        Lower volatility = larger positions (up to 1.5x)
        Higher volatility = smaller positions (down to 0.5x)
        """
        ohlcv = self.get_ohlcv(asset, '1h', 20)
        if ohlcv is None:
            return 1.0

        _, atr_percent = self.calculate_atr(ohlcv)

        # Target 2% ATR, scale inversely
        if atr_percent <= 0:
            return 1.0

        multiplier = 2.0 / atr_percent
        return max(0.5, min(1.5, multiplier))

    def get_dynamic_stops(
        self,
        asset: Asset,
        direction: str,
        entry_price: float,
        min_rr: float = 2.0,  # Minimum 2:1 R:R enforced
    ) -> tuple[float, float]:
        """
        Calculate dynamic stop loss and take profit based on ATR.

        PROFESSIONAL R:R ENFORCEMENT:
        - Minimum 2:1 R:R (can be higher based on volatility)
        - Tighter stops in low vol, wider in high vol
        - Target 3:1 in favorable conditions

        Returns (stop_loss_price, take_profit_price)
        """
        ohlcv = self.get_ohlcv(asset, '1h', 20)
        if ohlcv is None:
            # Fallback to fixed percentages with 2:1 R:R
            if direction == "long":
                return entry_price * 0.98, entry_price * 1.04  # 2% SL, 4% TP = 2:1
            else:
                return entry_price * 1.02, entry_price * 0.96

        atr, atr_percent = self.calculate_atr(ohlcv)

        # ADAPTIVE STOP based on volatility
        # Low vol (<2%): Tight 1.2x ATR stop
        # Normal (2-5%): Standard 1.5x ATR stop
        # High vol (>5%): Wide 2.0x ATR stop (max 3%)
        if atr_percent < 2:
            sl_mult = 1.2
            rr_ratio = 2.5  # Can aim higher in calm markets
        elif atr_percent > 5:
            sl_mult = 2.0
            rr_ratio = max(min_rr, 2.0)  # Just hit minimum in choppy
        else:
            sl_mult = 1.5
            rr_ratio = 2.5  # Target 2.5:1 normally

        sl_distance = atr * sl_mult

        # Cap stop loss at 3% max
        max_sl = entry_price * 0.03
        sl_distance = min(sl_distance, max_sl)

        # Take profit = SL * R:R ratio (minimum 2:1)
        tp_distance = sl_distance * max(min_rr, rr_ratio)

        if direction == "long":
            stop_loss = entry_price - sl_distance
            take_profit = entry_price + tp_distance
        else:
            stop_loss = entry_price + sl_distance
            take_profit = entry_price - tp_distance

        return stop_loss, take_profit
