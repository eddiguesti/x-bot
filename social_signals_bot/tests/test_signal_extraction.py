"""Tests for signal extraction module."""

import pytest
from src.models import Asset, Direction
from src.signal_extraction.rules import RuleBasedExtractor


class TestRuleBasedExtractor:
    """Tests for rule-based signal extraction."""

    def setup_method(self):
        self.extractor = RuleBasedExtractor()

    def test_detect_btc_long(self):
        """Test detecting BTC long signal."""
        text = "Going long on BTC here, looks bullish"
        result = self.extractor.extract(text)

        assert result.asset == Asset.BTC
        assert result.direction == Direction.LONG
        assert result.confidence > 0.5

    def test_detect_eth_short(self):
        """Test detecting ETH short signal."""
        text = "ETH looking bearish, shorting this dump"
        result = self.extractor.extract(text)

        assert result.asset == Asset.ETH
        assert result.direction == Direction.SHORT
        assert result.confidence > 0.5

    def test_no_asset(self):
        """Test text without recognizable asset."""
        text = "Going long here, market looks good"
        result = self.extractor.extract(text)

        assert result.asset is None
        assert result.direction is None

    def test_no_direction(self):
        """Test text with asset but no clear direction."""
        text = "BTC doing some interesting things today"
        result = self.extractor.extract(text)

        assert result.asset == Asset.BTC
        assert result.direction is None

    def test_negation_handling(self):
        """Test that negation flips direction."""
        text = "I'm not bullish on BTC, this is bearish"
        result = self.extractor.extract(text)

        assert result.asset == Asset.BTC
        # "not bullish" + "bearish" should lean short
        assert result.direction == Direction.SHORT

    def test_conditional_reduces_confidence(self):
        """Test that conditional language reduces confidence."""
        text1 = "I am long BTC here"
        text2 = "If it breaks out, maybe I would go long BTC?"

        result1 = self.extractor.extract(text1)
        result2 = self.extractor.extract(text2)

        # Multiple conditionals (if, maybe, ?) should reduce confidence
        assert result1.confidence > result2.confidence

    def test_explicit_position_boosts_confidence(self):
        """Test that explicit position statements boost confidence."""
        text = "I am long BTC from here"
        result = self.extractor.extract(text)

        assert result.asset == Asset.BTC
        assert result.direction == Direction.LONG
        assert result.confidence > 0.7

    def test_bitcoin_alias(self):
        """Test that 'bitcoin' is recognized as BTC."""
        text = "Bitcoin looking bullish, buying more"
        result = self.extractor.extract(text)

        assert result.asset == Asset.BTC
        assert result.direction == Direction.LONG

    def test_ethereum_alias(self):
        """Test that 'ethereum' is recognized as ETH."""
        text = "Selling my ethereum, too bearish"
        result = self.extractor.extract(text)

        assert result.asset == Asset.ETH
        assert result.direction == Direction.SHORT


class TestGlicko2:
    """Tests for Glicko-2 rating system."""

    def test_win_increases_rating(self):
        """Test that winning increases rating."""
        from src.scoring.glicko2 import Glicko2Calculator, Glicko2Rating

        calc = Glicko2Calculator()
        player = Glicko2Rating()

        new_rating = calc.update_single(player, won=True)

        assert new_rating.rating > player.rating
        assert new_rating.rd < player.rd  # Uncertainty decreases

    def test_loss_decreases_rating(self):
        """Test that losing decreases rating."""
        from src.scoring.glicko2 import Glicko2Calculator, Glicko2Rating

        calc = Glicko2Calculator()
        player = Glicko2Rating()

        new_rating = calc.update_single(player, won=False)

        assert new_rating.rating < player.rating

    def test_weight_calculation(self):
        """Test weight calculation from rating."""
        from src.scoring.glicko2 import Glicko2Rating

        # High rating, low RD = high weight
        good = Glicko2Rating(rating=1800, rd=50)
        # Low rating, high RD = low weight
        bad = Glicko2Rating(rating=1200, rd=300)

        assert good.weight > bad.weight
        assert good.weight > 0.5
        assert bad.weight < 0.3
