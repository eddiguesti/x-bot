"""Enhanced consensus engine with time decay and quality filtering."""

import json
import logging
import math
from datetime import datetime, timedelta
from typing import Optional

from sqlalchemy.orm import Session

from ..config import Settings
from ..models import (
    Asset, Direction, ConsensusAction, Signal, SignalOutcome,
    Creator, ConsensusSnapshot, ConsensusResult
)

logger = logging.getLogger(__name__)


class EnhancedConsensusEngine:
    """
    Enhanced consensus engine with:
    1. Time decay - more recent signals weighted higher
    2. Signal quality scoring
    3. Momentum detection
    4. Creator streak bonuses
    5. Confidence calibration
    """

    def __init__(self, settings: Settings):
        self.settings = settings

        # Time decay parameters
        self.decay_half_life_hours = 12  # Signals lose half weight every 12h
        self.min_time_weight = 0.1  # Minimum time weight (10%)

        # Quality thresholds
        self.min_signal_confidence = 0.3  # Ignore very low confidence signals
        self.min_creator_accuracy = 0.3  # Ignore creators with < 30% accuracy
        self.min_signals_for_trade = 5  # Require minimum signals for statistical reliability

        # Momentum parameters
        self.momentum_window_hours = 6  # Look at recent momentum
        self.momentum_threshold = 0.6  # 60%+ same direction = momentum

    def calculate_time_weight(self, signal_time: datetime) -> float:
        """
        Calculate time decay weight for a signal.

        Uses exponential decay with configurable half-life.
        More recent signals get higher weight.
        """
        age_hours = (datetime.utcnow() - signal_time).total_seconds() / 3600

        # Exponential decay: weight = 2^(-age/half_life)
        decay = math.pow(2, -age_hours / self.decay_half_life_hours)

        return max(self.min_time_weight, decay)

    def calculate_signal_quality(
        self,
        signal: Signal,
        creator: Creator,
    ) -> float:
        """
        Calculate quality score for a signal (0-1).

        Factors:
        1. Signal confidence
        2. Creator accuracy
        3. Creator weight (Glicko-2)
        4. Creator streak
        """
        score = 0.0

        # Signal confidence (30%)
        score += signal.confidence * 0.3

        # Creator accuracy (30%)
        if creator.total_predictions > 0:
            accuracy = creator.correct_predictions / creator.total_predictions
            score += accuracy * 0.3
        else:
            score += 0.15  # Neutral for new creators

        # Creator weight from Glicko-2 (25%)
        score += creator.weight * 0.25

        # Creator streak bonus (15%)
        # Positive streak = higher quality, negative = lower
        # Use recent accuracy as proxy for streak if not available
        current_streak = getattr(creator, 'current_streak', 0)
        if current_streak == 0 and creator.total_predictions >= 5:
            # Estimate streak from accuracy - high accuracy = positive streak
            accuracy = creator.correct_predictions / creator.total_predictions
            current_streak = int((accuracy - 0.5) * 10)  # -5 to +5
        streak_bonus = min(0.15, max(-0.15, current_streak * 0.03))
        score += 0.075 + streak_bonus  # Base 0.075 + streak adjustment

        return min(1.0, max(0.0, score))

    def detect_momentum(
        self,
        session: Session,
        asset: Asset,
        lookback_hours: int = None,
    ) -> tuple[Optional[Direction], float]:
        """
        Detect if there's momentum in one direction.

        Returns (direction, strength) where strength is 0-1.
        """
        if lookback_hours is None:
            lookback_hours = self.momentum_window_hours

        cutoff = datetime.utcnow() - timedelta(hours=lookback_hours)

        recent_signals = session.query(Signal).filter(
            Signal.asset == asset,
            Signal.posted_at >= cutoff,
        ).all()

        if len(recent_signals) < 3:
            return None, 0.0

        long_count = sum(1 for s in recent_signals if s.direction == Direction.LONG)
        short_count = len(recent_signals) - long_count
        total = len(recent_signals)

        long_ratio = long_count / total
        short_ratio = short_count / total

        if long_ratio >= self.momentum_threshold:
            return Direction.LONG, long_ratio
        elif short_ratio >= self.momentum_threshold:
            return Direction.SHORT, short_ratio
        else:
            return None, max(long_ratio, short_ratio)

    def calculate_consensus(
        self,
        session: Session,
        asset: Asset,
        lookback_hours: int = 48,
        save_snapshot: bool = True,
    ) -> ConsensusResult:
        """
        Calculate enhanced weighted consensus for an asset.

        Improvements over basic consensus:
        1. Time decay on signals
        2. Quality filtering
        3. Momentum bonus
        4. Better confidence calibration
        """
        cutoff = datetime.utcnow() - timedelta(hours=lookback_hours)

        # Get recent signals
        signals = session.query(Signal).join(Creator).filter(
            Signal.asset == asset,
            Signal.posted_at >= cutoff,
            Creator.is_active == True,
        ).all()

        if not signals:
            return self._empty_result(session, asset, save_snapshot)

        # Filter by quality
        quality_signals = []
        for signal in signals:
            # Skip low confidence signals
            if signal.confidence < self.min_signal_confidence:
                continue

            # Skip consistently wrong creators (if enough history)
            creator = signal.creator
            if creator.total_predictions >= 10:
                accuracy = creator.correct_predictions / creator.total_predictions
                if accuracy < self.min_creator_accuracy:
                    continue

            quality_signals.append(signal)

        if not quality_signals:
            return self._empty_result(session, asset, save_snapshot)

        # Calculate weighted votes with time decay
        weighted_score = 0.0
        total_weight = 0.0
        long_votes = 0
        short_votes = 0
        long_weight = 0.0
        short_weight = 0.0
        contributions = []

        for signal in quality_signals:
            creator = signal.creator

            # Base creator weight
            if creator.total_predictions >= self.settings.min_predictions_for_ranking:
                creator_weight = creator.weight
            else:
                creator_weight = 0.3  # Baseline for new creators

            # Apply time decay
            time_weight = self.calculate_time_weight(signal.posted_at)

            # Calculate signal quality
            quality = self.calculate_signal_quality(signal, creator)

            # Final weight = creator_weight * time_weight * quality
            final_weight = creator_weight * time_weight * quality

            # Direction value
            direction_value = 1 if signal.direction == Direction.LONG else -1

            # Vote = direction * confidence * weight
            vote = direction_value * signal.confidence * final_weight

            weighted_score += vote
            total_weight += final_weight

            if signal.direction == Direction.LONG:
                long_votes += 1
                long_weight += final_weight
            else:
                short_votes += 1
                short_weight += final_weight

            contributions.append({
                "signal_id": signal.id,
                "creator": creator.username,
                "direction": signal.direction.value,
                "base_weight": creator_weight,
                "time_weight": time_weight,
                "quality": quality,
                "final_weight": final_weight,
                "vote": vote,
            })

        # Normalize score
        if total_weight > 0:
            normalized_score = weighted_score / total_weight
        else:
            normalized_score = 0.0

        # Check for momentum bonus
        momentum_dir, momentum_strength = self.detect_momentum(session, asset)
        if momentum_dir:
            if (momentum_dir == Direction.LONG and normalized_score > 0) or \
               (momentum_dir == Direction.SHORT and normalized_score < 0):
                # Momentum aligns with consensus - boost confidence
                momentum_bonus = momentum_strength * 0.1
                normalized_score = normalized_score * (1 + momentum_bonus)
                logger.debug(f"Momentum bonus applied: {momentum_bonus:.2f}")

        # Determine action with calibrated confidence
        action, confidence = self._determine_action(
            normalized_score,
            long_votes,
            short_votes,
            len(quality_signals),
        )

        # Top contributors
        contributions.sort(key=lambda x: abs(x["vote"]), reverse=True)
        top_contributors = [c["creator"] for c in contributions[:5]]

        result = ConsensusResult(
            asset=asset,
            timestamp=datetime.utcnow(),
            action=action,
            confidence=confidence,
            weighted_score=normalized_score,
            long_votes=long_votes,
            short_votes=short_votes,
            top_contributors=top_contributors,
        )

        if save_snapshot:
            self._save_snapshot(session, result, [s.id for s in quality_signals])

        logger.info(
            f"Enhanced consensus for {asset.value}: {action.value} "
            f"(score: {normalized_score:.3f}, confidence: {confidence:.2%}, "
            f"signals: {len(quality_signals)}/{len(signals)})"
        )

        return result

    def _determine_action(
        self,
        score: float,
        long_votes: int,
        short_votes: int,
        total_signals: int,
    ) -> tuple[ConsensusAction, float]:
        """
        Determine action with calibrated confidence.

        Confidence is based on:
        1. Absolute score strength
        2. Vote ratio (how unanimous)
        3. Sample size (more signals = more confident)
        """
        # PROFITABILITY FIX: Require minimum signals for statistical reliability
        if total_signals < self.min_signals_for_trade:
            logger.debug(f"Insufficient signals: {total_signals} < {self.min_signals_for_trade}")
            return ConsensusAction.NO_TRADE, 0.0

        abs_score = abs(score)

        # Calculate vote ratio
        total_votes = long_votes + short_votes
        if total_votes > 0:
            if score > 0:
                vote_ratio = long_votes / total_votes
            else:
                vote_ratio = short_votes / total_votes
        else:
            vote_ratio = 0.5

        # Sample size factor (caps at 20 signals)
        sample_factor = min(1.0, total_signals / 20)

        # Base confidence from score
        base_confidence = abs_score

        # Adjust by vote unanimity (0.5 = split, 1.0 = unanimous)
        unanimity_factor = 0.5 + (vote_ratio - 0.5)

        # Adjust by sample size
        final_confidence = base_confidence * unanimity_factor * (0.5 + 0.5 * sample_factor)

        # Thresholds
        strong = self.settings.strong_signal_threshold
        weak = self.settings.weak_signal_threshold

        if abs_score >= strong and final_confidence >= 0.3:
            action = ConsensusAction.LONG if score > 0 else ConsensusAction.SHORT
            return action, min(1.0, final_confidence)

        elif abs_score >= weak and final_confidence >= 0.2:
            action = ConsensusAction.LONG if score > 0 else ConsensusAction.SHORT
            return action, final_confidence

        else:
            return ConsensusAction.NO_TRADE, final_confidence

    def _empty_result(
        self,
        session: Session,
        asset: Asset,
        save_snapshot: bool,
    ) -> ConsensusResult:
        """Create empty result for no signals."""
        result = ConsensusResult(
            asset=asset,
            timestamp=datetime.utcnow(),
            action=ConsensusAction.NO_TRADE,
            confidence=0.0,
            weighted_score=0.0,
            long_votes=0,
            short_votes=0,
            top_contributors=[],
        )

        if save_snapshot:
            self._save_snapshot(session, result, [])

        return result

    def _save_snapshot(
        self,
        session: Session,
        result: ConsensusResult,
        signal_ids: list[int],
    ):
        """Save consensus snapshot."""
        snapshot = ConsensusSnapshot(
            asset=result.asset,
            timestamp=result.timestamp,
            raw_score=result.weighted_score,
            weighted_score=result.weighted_score,
            long_votes=result.long_votes,
            short_votes=result.short_votes,
            action=result.action,
            confidence=result.confidence,
            contributing_signals=json.dumps(signal_ids),
        )
        session.add(snapshot)
        session.commit()

    def get_all_consensus(
        self,
        session: Session,
        lookback_hours: int = 48,
    ) -> dict[Asset, ConsensusResult]:
        """Calculate consensus for all assets."""
        results = {}
        for asset in Asset:
            results[asset] = self.calculate_consensus(
                session, asset, lookback_hours, save_snapshot=True
            )
        return results

    def should_trade(self, result: ConsensusResult) -> bool:
        """Check if result indicates a trade."""
        return result.action != ConsensusAction.NO_TRADE
