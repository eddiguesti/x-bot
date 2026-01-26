"""LLM-based signal extraction using Grok, Gemini Flash, or DeepSeek.

Uses LLMs to extract trading signals from social media posts with
better understanding of context, sarcasm, and nuance.
"""

import json
import logging
import re
import time
from typing import Optional, Literal

from pydantic import BaseModel, Field, field_validator

from ..config import Settings
from ..constants import (
    MAX_LLM_INPUT_LENGTH,
    LLM_MAX_RETRIES,
    LLM_RETRY_BASE_DELAY,
    LLM_TEMPERATURE,
    LLM_MAX_OUTPUT_TOKENS,
)
from ..models import Asset, Direction, SignalExtraction

logger = logging.getLogger(__name__)

# System prompt for signal extraction
EXTRACTION_PROMPT = """You are a crypto trading signal extractor. Analyze the following social media post and extract any trading signals.

IMPORTANT RULES:
1. Only extract CLEAR trading signals - ignore general discussion
2. Detect sarcasm and negation (e.g., "BTC to the moon... NOT" is bearish)
3. If multiple assets are mentioned, extract the PRIMARY one being discussed
4. Confidence should reflect how clear/actionable the signal is (0.0-1.0)
5. Return NO_SIGNAL if there's no clear trading recommendation

SUPPORTED ASSETS: BTC, ETH, SOL, XRP, BNB, DOGE, ADA, AVAX, LINK, DOT, SHIB, LTC, UNI, ATOM, ARB, OP, MATIC, AAVE, APT, NEAR, SUI, SEI, TIA, INJ, TAO, FET, RNDR, PEPE, WIF

Respond in JSON format ONLY:
{
    "asset": "BTC" or null,
    "direction": "LONG" or "SHORT" or null,
    "confidence": 0.0-1.0,
    "reasoning": "brief explanation"
}

POST TO ANALYZE:
"""

# Pydantic model for validating LLM responses
class LLMExtractionResponse(BaseModel):
    """Validated LLM extraction response."""
    asset: Optional[str] = Field(None, description="Crypto asset symbol")
    direction: Optional[Literal["LONG", "SHORT"]] = Field(None, description="Trading direction")
    confidence: float = Field(0.0, ge=0.0, le=1.0, description="Confidence score")
    reasoning: str = Field("", description="Explanation for the signal")

    @field_validator('asset', mode='before')
    @classmethod
    def normalize_asset(cls, v):
        """Normalize asset to uppercase if provided."""
        if v is None:
            return None
        return str(v).upper() if v else None

    @field_validator('direction', mode='before')
    @classmethod
    def normalize_direction(cls, v):
        """Normalize direction string."""
        if v is None:
            return None
        v_upper = str(v).upper()
        if v_upper in ("LONG", "BUY", "BULLISH"):
            return "LONG"
        elif v_upper in ("SHORT", "SELL", "BEARISH"):
            return "SHORT"
        return None


class LLMSignalExtractor:
    """
    Extract trading signals using LLM (Grok primary, Gemini/DeepSeek fallback).

    Much better at understanding:
    - Sarcasm and negation
    - Context and nuance
    - Crypto slang and jargon
    - Multiple signals in one post
    """

    def __init__(self, settings: Settings):
        self.settings = settings
        self.grok_client = None
        self.gemini_client = None
        self.deepseek_client = None
        self._init_clients()

    def _init_clients(self):
        """Initialize LLM clients based on available API keys."""
        # Try Grok first (fast, good reasoning)
        if self.settings.grok_api_key:
            try:
                from openai import OpenAI
                self.grok_client = OpenAI(
                    api_key=self.settings.grok_api_key,
                    base_url="https://api.x.ai/v1"
                )
                logger.info("Grok client initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize Grok client: {e}")

        # Try Gemini as fallback
        if self.settings.gemini_api_key:
            try:
                import google.generativeai as genai
                genai.configure(api_key=self.settings.gemini_api_key)
                self.gemini_client = genai.GenerativeModel('gemini-2.0-flash-exp')
                logger.info("Gemini Flash client initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize Gemini client: {e}")

        # Try DeepSeek as final fallback
        if self.settings.deepseek_api_key:
            try:
                from openai import OpenAI
                self.deepseek_client = OpenAI(
                    api_key=self.settings.deepseek_api_key,
                    base_url="https://api.deepseek.com"
                )
                logger.info("DeepSeek client initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize DeepSeek client: {e}")

        if not self.grok_client and not self.gemini_client and not self.deepseek_client:
            logger.warning("No LLM clients available - LLM extraction disabled")

    @property
    def enabled(self) -> bool:
        """Check if any LLM client is available."""
        return self.grok_client is not None or self.gemini_client is not None or self.deepseek_client is not None

    def _retry_with_backoff(self, func, provider: str, *args, **kwargs):
        """Execute function with exponential backoff retry logic."""
        last_exception = None

        for attempt in range(LLM_MAX_RETRIES):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                last_exception = e
                if attempt < LLM_MAX_RETRIES - 1:
                    delay = LLM_RETRY_BASE_DELAY * (2 ** attempt)
                    logger.warning(
                        f"{provider} extraction failed (attempt {attempt + 1}/{LLM_MAX_RETRIES}): {e}. "
                        f"Retrying in {delay:.1f}s..."
                    )
                    time.sleep(delay)
                else:
                    logger.error(f"{provider} extraction failed after {LLM_MAX_RETRIES} attempts: {e}")

        return None

    def _extract_with_grok(self, text: str) -> Optional[LLMExtractionResponse]:
        """Extract signal using Grok with retry logic."""
        if not self.grok_client:
            return None

        def _call():
            response = self.grok_client.chat.completions.create(
                model="grok-4-fast-non-reasoning",
                messages=[
                    {"role": "system", "content": EXTRACTION_PROMPT},
                    {"role": "user", "content": text}
                ],
                temperature=LLM_TEMPERATURE,
                max_tokens=LLM_MAX_OUTPUT_TOKENS,
            )
            return self._parse_and_validate_response(response.choices[0].message.content)

        return self._retry_with_backoff(_call, "Grok")

    def _extract_with_gemini(self, text: str) -> Optional[LLMExtractionResponse]:
        """Extract signal using Gemini Flash with retry logic."""
        if not self.gemini_client:
            return None

        def _call():
            prompt = EXTRACTION_PROMPT + text
            response = self.gemini_client.generate_content(
                prompt,
                generation_config={
                    "temperature": LLM_TEMPERATURE,
                    "max_output_tokens": LLM_MAX_OUTPUT_TOKENS,
                }
            )
            return self._parse_and_validate_response(response.text)

        return self._retry_with_backoff(_call, "Gemini")

    def _extract_with_deepseek(self, text: str) -> Optional[LLMExtractionResponse]:
        """Extract signal using DeepSeek with retry logic."""
        if not self.deepseek_client:
            return None

        def _call():
            response = self.deepseek_client.chat.completions.create(
                model="deepseek-chat",
                messages=[
                    {"role": "system", "content": EXTRACTION_PROMPT},
                    {"role": "user", "content": text}
                ],
                temperature=LLM_TEMPERATURE,
                max_tokens=LLM_MAX_OUTPUT_TOKENS,
            )
            return self._parse_and_validate_response(response.choices[0].message.content)

        return self._retry_with_backoff(_call, "DeepSeek")

    def _parse_and_validate_response(self, response_text: str) -> Optional[LLMExtractionResponse]:
        """Parse and validate LLM response JSON using Pydantic."""
        try:
            # Clean up response - extract JSON from markdown if needed
            text = response_text.strip()
            if text.startswith("```"):
                # Remove markdown code blocks
                text = re.sub(r'^```(?:json)?\s*', '', text)
                text = re.sub(r'\s*```$', '', text)

            data = json.loads(text)
            # Validate with Pydantic model
            return LLMExtractionResponse.model_validate(data)
        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse LLM response: {e}")
            logger.debug(f"Raw response: {response_text}")
            return None
        except Exception as e:
            logger.warning(f"Failed to validate LLM response: {e}")
            logger.debug(f"Raw response: {response_text}")
            return None

    def _to_signal_extraction(self, result: LLMExtractionResponse) -> SignalExtraction:
        """Convert validated LLM result to SignalExtraction object."""
        asset = None
        direction = None

        # Parse asset (already normalized by Pydantic)
        if result.asset:
            try:
                asset = Asset(result.asset)
            except ValueError:
                # Try to match common variations
                asset_map = {
                    "BITCOIN": Asset.BTC,
                    "ETHEREUM": Asset.ETH,
                    "SOLANA": Asset.SOL,
                    "RIPPLE": Asset.XRP,
                    "DOGECOIN": Asset.DOGE,
                    "CARDANO": Asset.ADA,
                }
                asset = asset_map.get(result.asset)

        # Parse direction (already normalized by Pydantic)
        if result.direction:
            direction = Direction.LONG if result.direction == "LONG" else Direction.SHORT

        return SignalExtraction(
            asset=asset,
            direction=direction,
            confidence=result.confidence,
            reasoning=f"LLM: {result.reasoning}"
        )

    def extract(self, text: str) -> SignalExtraction:
        """
        Extract trading signal from text using LLM.

        Tries Gemini first, falls back to DeepSeek if needed.

        Args:
            text: Post text to analyze

        Returns:
            SignalExtraction with detected signal
        """
        if not self.enabled:
            return SignalExtraction(
                asset=None,
                direction=None,
                confidence=0.0,
                reasoning="LLM extraction disabled - no API keys"
            )

        # Truncate very long texts (YouTube transcripts)
        if len(text) > MAX_LLM_INPUT_LENGTH:
            text = text[:MAX_LLM_INPUT_LENGTH] + "... [truncated]"

        result = None

        # Try based on preference
        provider = self.settings.llm_provider.lower()

        if provider == "grok" or provider == "auto":
            result = self._extract_with_grok(text)

        if result is None and (provider == "gemini" or provider == "auto"):
            result = self._extract_with_gemini(text)

        if result is None and (provider == "deepseek" or provider == "auto"):
            result = self._extract_with_deepseek(text)

        if result is None:
            return SignalExtraction(
                asset=None,
                direction=None,
                confidence=0.0,
                reasoning="LLM extraction failed"
            )

        return self._to_signal_extraction(result)

    def extract_batch(self, texts: list[str]) -> list[SignalExtraction]:
        """Extract signals from multiple texts."""
        return [self.extract(text) for text in texts]

    def is_actionable(self, extraction: SignalExtraction, min_confidence: float = 0.5) -> bool:
        """Check if extraction is actionable."""
        return (
            extraction.asset is not None
            and extraction.direction is not None
            and extraction.confidence >= min_confidence
        )
