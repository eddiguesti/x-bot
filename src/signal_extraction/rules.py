"""Rule-based signal extraction using keyword matching."""

import re
import logging
from typing import Optional

from ..models import Asset, Direction, SignalExtraction

logger = logging.getLogger(__name__)


class RuleBasedExtractor:
    """
    Simple rule-based signal extractor using keyword matching.
    Used as fallback when NLP model is unavailable or for quick validation.
    """

    # Asset detection patterns
    ASSET_PATTERNS = {
        Asset.BTC: [
            r'\bBTC\b',
            r'\bbitcoin\b',
            r'\b₿\b',
            r'\bsats\b',
        ],
        Asset.ETH: [
            r'\bETH\b',
            r'\bethereum\b',
            r'\bether\b',
        ],
        Asset.SOL: [
            r'\bSOL\b',
            r'\bsolana\b',
        ],
        Asset.XRP: [
            r'\bXRP\b',
            r'\bripple\b',
        ],
        Asset.DOGE: [
            r'\bDOGE\b',
            r'\bdogecoin\b',
        ],
        Asset.ADA: [
            r'\bADA\b',
            r'\bcardano\b',
        ],
        Asset.AVAX: [
            r'\bAVAX\b',
            r'\bavalanche\b',
        ],
        Asset.LINK: [
            r'\bLINK\b',
            r'\bchainlink\b',
        ],
        Asset.DOT: [
            r'\bDOT\b',
            r'\bpolkadot\b',
        ],
        Asset.SHIB: [
            r'\bSHIB\b',
            r'\bshiba\b',
        ],
        Asset.LTC: [
            r'\bLTC\b',
            r'\blitecoin\b',
        ],
        Asset.UNI: [
            r'\bUNI\b',
            r'\buniswap\b',
        ],
        Asset.ATOM: [
            r'\bATOM\b',
            r'\bcosmos\b',
        ],
        Asset.ARB: [
            r'\bARB\b',
            r'\barbitrum\b',
        ],
        Asset.OP: [
            r'\bOP\b',
            r'\boptimism\b',
        ],
        Asset.APT: [
            r'\bAPT\b',
            r'\baptos\b',
        ],
        Asset.NEAR: [
            r'\bNEAR\b',
        ],
        Asset.INJ: [
            r'\bINJ\b',
            r'\binjective\b',
        ],
        Asset.TAO: [
            r'\bTAO\b',
            r'\bbittensor\b',
        ],
    }

    # Direction keywords (weighted by strength)
    LONG_KEYWORDS = {
        # Strong bullish (weight 1.0)
        'long': 1.0,
        'buy': 1.0,
        'bullish': 1.0,
        'calls': 0.9,
        'moon': 0.8,
        'pump': 0.8,
        'accumulate': 0.9,
        'accumulating': 0.9,
        'buying': 0.9,
        'bid': 0.8,
        'bidding': 0.8,
        'longs': 1.0,
        'longing': 1.0,
        'aping': 0.9,
        'ape in': 0.9,
        'dip buy': 0.9,
        'buy the dip': 0.9,
        'btfd': 0.9,
        'hodl': 0.7,
        'holding': 0.6,
        'loading': 0.8,
        'loaded': 0.8,
        'stacking': 0.8,
        # Medium bullish
        'support': 0.6,
        'bounce': 0.6,
        'rally': 0.7,
        'breakout': 0.7,
        'higher': 0.5,
        'up': 0.4,
        'green': 0.5,
        'bull': 0.7,
        'send it': 0.7,
        'blast off': 0.8,
        'ripping': 0.7,
        'pumping': 0.8,
        'mooning': 0.8,
        'ath': 0.6,
        'new high': 0.7,
        'oversold': 0.6,
    }

    SHORT_KEYWORDS = {
        # Strong bearish (weight 1.0)
        'short': 1.0,
        'sell': 1.0,
        'bearish': 1.0,
        'puts': 0.9,
        'dump': 0.8,
        'crash': 0.8,
        'exit': 0.8,
        'selling': 0.9,
        'shorts': 1.0,
        'shorting': 1.0,
        'fading': 0.9,
        'fade': 0.9,
        'taking profit': 0.8,
        'tp': 0.7,
        'cashing out': 0.8,
        # Medium bearish
        'resistance': 0.6,
        'rejection': 0.7,
        'breakdown': 0.7,
        'lower': 0.5,
        'down': 0.4,
        'red': 0.5,
        'bear': 0.7,
        'rekt': 0.6,
        'liquidated': 0.5,
        'dumping': 0.8,
        'tanking': 0.8,
        'bleeding': 0.7,
        'overbought': 0.6,
        'top': 0.5,
        'local top': 0.7,
        'distribution': 0.6,
    }

    # Negation patterns that flip meaning
    NEGATION_PATTERNS = [
        r"not\s+",
        r"don'?t\s+",
        r"won'?t\s+",
        r"isn'?t\s+",
        r"aren'?t\s+",
        r"no\s+",
        r"never\s+",
        r"stop\s+",
        r"avoid\s+",
    ]

    # Conditional patterns that reduce confidence
    CONDITIONAL_PATTERNS = [
        r"\bif\b",
        r"\bwhen\b",
        r"\bunless\b",
        r"\bmight\b",
        r"\bcould\b",
        r"\bmaybe\b",
        r"\bperhaps\b",
        r"\bpossibly\b",
        r"\?",  # Questions
    ]

    def extract(self, text: str) -> SignalExtraction:
        """
        Extract trading signal from text using rules.

        Args:
            text: Post text to analyze

        Returns:
            SignalExtraction with detected asset, direction, and confidence
        """
        text_lower = text.lower()

        # Detect asset
        asset = self._detect_asset(text)
        if asset is None:
            return SignalExtraction(
                asset=None,
                direction=None,
                confidence=0.0,
                reasoning="No recognized asset mentioned"
            )

        # Detect direction
        direction, direction_confidence = self._detect_direction(text_lower)
        if direction is None:
            return SignalExtraction(
                asset=asset,
                direction=None,
                confidence=0.0,
                reasoning=f"Asset {asset.value} detected but no clear direction"
            )

        # Apply confidence modifiers
        final_confidence = self._apply_modifiers(text_lower, direction_confidence)

        reasoning = self._build_reasoning(text_lower, asset, direction, final_confidence)

        return SignalExtraction(
            asset=asset,
            direction=direction,
            confidence=final_confidence,
            reasoning=reasoning
        )

    def _detect_asset(self, text: str) -> Optional[Asset]:
        """Detect which asset is being discussed."""
        text_upper = text.upper()
        text_lower = text.lower()

        for asset, patterns in self.ASSET_PATTERNS.items():
            for pattern in patterns:
                if re.search(pattern, text_upper if pattern.isupper() else text_lower, re.IGNORECASE):
                    return asset

        return None

    def _detect_direction(self, text: str) -> tuple[Optional[Direction], float]:
        """
        Detect trading direction from text.

        Returns:
            Tuple of (direction, confidence)
        """
        long_score = 0.0
        short_score = 0.0

        # Check for negations that might flip meaning
        has_negation = any(
            re.search(pattern, text)
            for pattern in self.NEGATION_PATTERNS
        )

        # Score long keywords
        for keyword, weight in self.LONG_KEYWORDS.items():
            if re.search(rf'\b{re.escape(keyword)}\b', text):
                # Check if negated
                negated = any(
                    re.search(rf'{neg}{keyword}', text)
                    for neg in self.NEGATION_PATTERNS
                )
                if negated:
                    short_score += weight * 0.8  # Negated long = short
                else:
                    long_score += weight

        # Score short keywords
        for keyword, weight in self.SHORT_KEYWORDS.items():
            if re.search(rf'\b{re.escape(keyword)}\b', text):
                negated = any(
                    re.search(rf'{neg}{keyword}', text)
                    for neg in self.NEGATION_PATTERNS
                )
                if negated:
                    long_score += weight * 0.8  # Negated short = long
                else:
                    short_score += weight

        # Determine direction
        if long_score == 0 and short_score == 0:
            return None, 0.0

        total_score = long_score + short_score
        if long_score > short_score:
            confidence = long_score / max(total_score, 1)
            return Direction.LONG, min(1.0, confidence)
        elif short_score > long_score:
            confidence = short_score / max(total_score, 1)
            return Direction.SHORT, min(1.0, confidence)
        else:
            # Equal scores = unclear
            return None, 0.0

    def _apply_modifiers(self, text: str, base_confidence: float) -> float:
        """Apply confidence modifiers based on text patterns."""
        confidence = base_confidence

        # Reduce confidence for conditional language
        conditional_count = sum(
            1 for pattern in self.CONDITIONAL_PATTERNS
            if re.search(pattern, text)
        )
        confidence *= (0.9 ** conditional_count)

        # Reduce confidence for very short texts
        word_count = len(text.split())
        if word_count < 5:
            confidence *= 0.8

        # Boost confidence for explicit position statements
        explicit_patterns = [
            r'\bi am\s+(long|short)',
            r'\bgoing\s+(long|short)',
            r'\bentered\s+(long|short)',
            r'\bopened\s+a?\s*(long|short)',
        ]
        if any(re.search(p, text) for p in explicit_patterns):
            confidence = min(1.0, confidence * 1.2)

        return round(confidence, 3)

    def _build_reasoning(
        self,
        text: str,
        asset: Asset,
        direction: Direction,
        confidence: float
    ) -> str:
        """Build explanation of extraction logic."""
        parts = [f"Detected {asset.value}"]

        # Find matched keywords
        if direction == Direction.LONG:
            matched = [k for k in self.LONG_KEYWORDS if re.search(rf'\b{k}\b', text)]
        else:
            matched = [k for k in self.SHORT_KEYWORDS if re.search(rf'\b{k}\b', text)]

        if matched:
            parts.append(f"keywords: {', '.join(matched[:3])}")

        parts.append(f"confidence: {confidence:.0%}")

        return " | ".join(parts)
