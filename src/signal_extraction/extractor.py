"""Main signal extractor combining NLP and rule-based approaches."""

import logging
from typing import Optional

from ..models import Asset, Direction, SignalExtraction
from .rules import RuleBasedExtractor

logger = logging.getLogger(__name__)

# Try to import transformers for FinBERT
try:
    from transformers import pipeline, AutoModelForSequenceClassification, AutoTokenizer
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    logger.warning("transformers not installed, using rule-based extraction only")


class SignalExtractor:
    """
    Main signal extractor that combines NLP models with rule-based fallback.

    Uses FinBERT for sentiment classification when available, falls back to
    keyword matching otherwise.
    """

    # FinBERT model for financial sentiment
    FINBERT_MODEL = "ProsusAI/finbert"

    # Sentiment to direction mapping
    SENTIMENT_MAP = {
        'positive': Direction.LONG,
        'negative': Direction.SHORT,
        'neutral': None,
    }

    def __init__(self, use_nlp: bool = True, min_confidence: float = 0.7):
        """
        Initialize the signal extractor.

        Args:
            use_nlp: Whether to use NLP models (if available)
            min_confidence: Minimum confidence to consider a signal valid
        """
        self.min_confidence = min_confidence
        self.rule_extractor = RuleBasedExtractor()
        self.nlp_pipeline = None

        if use_nlp and TRANSFORMERS_AVAILABLE:
            self._init_nlp_pipeline()

    def _init_nlp_pipeline(self):
        """Initialize the FinBERT pipeline."""
        try:
            logger.info("Loading FinBERT model...")
            self.nlp_pipeline = pipeline(
                "sentiment-analysis",
                model=self.FINBERT_MODEL,
                tokenizer=self.FINBERT_MODEL,
            )
            logger.info("FinBERT model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load FinBERT model: {e}")
            self.nlp_pipeline = None

    def extract(self, text: str) -> SignalExtraction:
        """
        Extract trading signal from text.

        Uses NLP pipeline if available, otherwise falls back to rules.
        Combines both approaches for higher confidence signals.

        Args:
            text: Post text to analyze

        Returns:
            SignalExtraction with detected signal details
        """
        # Always run rule-based extraction for asset detection
        rule_result = self.rule_extractor.extract(text)

        # If no asset detected, skip NLP
        if rule_result.asset is None:
            return rule_result

        # If NLP available, run sentiment analysis
        if self.nlp_pipeline is not None:
            nlp_result = self._extract_with_nlp(text, rule_result.asset)

            # Combine results
            return self._combine_results(rule_result, nlp_result)

        return rule_result

    def _extract_with_nlp(self, text: str, asset: Asset) -> SignalExtraction:
        """Extract signal using FinBERT NLP model."""
        try:
            # Truncate text to model's max length
            truncated = text[:512]

            result = self.nlp_pipeline(truncated)[0]
            label = result['label'].lower()
            score = result['score']

            direction = self.SENTIMENT_MAP.get(label)

            return SignalExtraction(
                asset=asset,
                direction=direction,
                confidence=score if direction else 0.0,
                reasoning=f"FinBERT: {label} ({score:.2%})"
            )

        except Exception as e:
            logger.warning(f"NLP extraction failed: {e}")
            return SignalExtraction(
                asset=asset,
                direction=None,
                confidence=0.0,
                reasoning=f"NLP error: {str(e)}"
            )

    def _combine_results(
        self,
        rule_result: SignalExtraction,
        nlp_result: SignalExtraction
    ) -> SignalExtraction:
        """
        Combine rule-based and NLP results for final signal.

        Agreement between methods increases confidence.
        Disagreement decreases confidence.
        """
        # If NLP found no direction, use rule result
        if nlp_result.direction is None:
            return rule_result

        # If rule found no direction, use NLP result
        if rule_result.direction is None:
            return nlp_result

        # Both found direction - check agreement
        if rule_result.direction == nlp_result.direction:
            # Agreement - boost confidence
            combined_confidence = min(1.0, (rule_result.confidence + nlp_result.confidence) / 1.5)
            return SignalExtraction(
                asset=rule_result.asset,
                direction=rule_result.direction,
                confidence=combined_confidence,
                reasoning=f"Agreed: {rule_result.reasoning} + {nlp_result.reasoning}"
            )
        else:
            # Disagreement - use higher confidence, reduce overall
            if nlp_result.confidence > rule_result.confidence:
                winner = nlp_result
                loser = rule_result
            else:
                winner = rule_result
                loser = nlp_result

            # Reduce confidence due to disagreement
            reduced_confidence = winner.confidence * 0.7

            return SignalExtraction(
                asset=winner.asset,
                direction=winner.direction,
                confidence=reduced_confidence,
                reasoning=f"Conflict (using {winner.reasoning}, rejected {loser.reasoning})"
            )

    def extract_batch(self, texts: list[str]) -> list[SignalExtraction]:
        """
        Extract signals from multiple texts.

        Args:
            texts: List of post texts

        Returns:
            List of SignalExtraction results
        """
        return [self.extract(text) for text in texts]

    def is_actionable(self, extraction: SignalExtraction, min_confidence: float = None) -> bool:
        """
        Check if an extraction is actionable (has asset, direction, and sufficient confidence).

        Args:
            extraction: SignalExtraction to check
            min_confidence: Override minimum confidence threshold (optional)

        Returns:
            True if signal is actionable
        """
        threshold = min_confidence if min_confidence is not None else self.min_confidence
        return (
            extraction.asset is not None
            and extraction.direction is not None
            and extraction.confidence >= threshold
        )
