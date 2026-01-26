"""Weighted consensus engine using Dynamic Weighted Majority algorithm."""

import json
import logging
from datetime import datetime, timedelta
from typing import Optional

from sqlalchemy.orm import Session

from ..config import Settings
from ..models import (
    Asset, Direction, ConsensusAction, Signal, SignalOutcome,
    Creator, ConsensusSnapshot, ConsensusResult
)

logger = logging.getLogger(__name__)


class ConsensusEngine:
    """
    Dynamic Weighted Majority consensus engine.

    Aggregates signals from multiple creators, weighting by their
    Glicko-2 derived weights and signal confidence.
    """

    def __init__(self, settings: Settings):
        self.settings = settings

    def calculate_consensus(
        self,
        session: Session,
        asset: Asset,
        lookback_hours: int = 24,
        save_snapshot: bool = True,
    ) -> ConsensusResult:
        """
        Calculate weighted consensus for an asset.

        Args:
            session: Database session
            asset: Asset to calculate consensus for
            lookback_hours: How far back to look for signals
            save_snapshot: Whether to save result to database

        Returns:
            ConsensusResult with action and confidence
        """
        cutoff = datetime.utcnow() - timedelta(hours=lookback_hours)

        # Get recent signals for this asset (only evaluated ones count more)
        signals = session.query(Signal).join(Creator).filter(
            Signal.asset == asset,
            Signal.posted_at >= cutoff,
            Creator.is_active == True,
        ).all()

        if not signals:
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

        # Calculate weighted votes
        total_weight = 0.0
        weighted_score = 0.0
        long_votes = 0
        short_votes = 0
        long_weight = 0.0
        short_weight = 0.0
        contributions = []

        for signal in signals:
            creator = signal.creator

            # Get creator weight (give new creators a baseline weight)
            if creator.total_predictions >= self.settings.min_predictions_for_ranking:
                creator_weight = creator.weight
            else:
                # New/unranked creators get baseline weight of 0.3
                creator_weight = 0.3

            # Direction value: +1 for long, -1 for short
            direction_value = 1 if signal.direction == Direction.LONG else -1

            # Vote = direction * confidence * weight
            vote = direction_value * signal.confidence * creator_weight

            weighted_score += vote
            total_weight += creator_weight

            # Count votes
            if signal.direction == Direction.LONG:
                long_votes += 1
                long_weight += creator_weight
            else:
                short_votes += 1
                short_weight += creator_weight

            contributions.append({
                "signal_id": signal.id,
                "creator": creator.username,
                "direction": signal.direction.value,
                "weight": creator_weight,
                "vote": vote,
            })

        # Normalize score to -1 to +1
        if total_weight > 0:
            normalized_score = weighted_score / total_weight
        else:
            normalized_score = 0.0

        # Determine action
        action, confidence = self._determine_action(normalized_score)

        # Get top contributors
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
            self._save_snapshot(session, result, [s.id for s in signals])

        logger.info(
            f"Consensus for {asset.value}: {action.value} "
            f"(score: {normalized_score:.3f}, confidence: {confidence:.2%})"
        )

        return result

    def _determine_action(
        self,
        score: float
    ) -> tuple[ConsensusAction, float]:
        """
        Determine trading action from consensus score.

        Args:
            score: Normalized consensus score (-1 to +1)

        Returns:
            Tuple of (action, confidence)
        """
        abs_score = abs(score)

        # Strong signal thresholds
        strong = self.settings.strong_signal_threshold
        weak = self.settings.weak_signal_threshold

        if abs_score >= strong:
            # Strong signal
            action = ConsensusAction.LONG if score > 0 else ConsensusAction.SHORT
            confidence = min(1.0, abs_score)
            return action, confidence

        elif abs_score >= weak:
            # Weak signal (still actionable but lower confidence)
            action = ConsensusAction.LONG if score > 0 else ConsensusAction.SHORT
            confidence = abs_score
            return action, confidence

        else:
            # No clear consensus
            return ConsensusAction.NO_TRADE, abs_score

    def _save_snapshot(
        self,
        session: Session,
        result: ConsensusResult,
        signal_ids: list[int]
    ):
        """Save consensus snapshot to database."""
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
        lookback_hours: int = 24,
    ) -> dict[Asset, ConsensusResult]:
        """
        Calculate consensus for all assets.

        Args:
            session: Database session
            lookback_hours: How far back to look

        Returns:
            Dict mapping asset to ConsensusResult
        """
        results = {}
        for asset in Asset:
            results[asset] = self.calculate_consensus(
                session, asset, lookback_hours, save_snapshot=True
            )
        return results

    def should_trade(self, result: ConsensusResult) -> bool:
        """
        Check if consensus result indicates a trade.

        Args:
            result: ConsensusResult to check

        Returns:
            True if action is LONG or SHORT (not NO_TRADE)
        """
        return result.action != ConsensusAction.NO_TRADE

    def get_historical_accuracy(
        self,
        session: Session,
        asset: Optional[Asset] = None,
        days: int = 30,
    ) -> dict:
        """
        Calculate historical accuracy of consensus signals.

        Args:
            session: Database session
            asset: Specific asset or None for all
            days: Days of history to analyze

        Returns:
            Dict with accuracy metrics
        """
        cutoff = datetime.utcnow() - timedelta(days=days)

        query = session.query(ConsensusSnapshot).filter(
            ConsensusSnapshot.timestamp >= cutoff,
            ConsensusSnapshot.action != ConsensusAction.NO_TRADE,
        )

        if asset:
            query = query.filter(ConsensusSnapshot.asset == asset)

        snapshots = query.order_by(ConsensusSnapshot.timestamp.asc()).all()

        if not snapshots:
            return {"total": 0, "correct": 0, "accuracy": 0}

        # For each snapshot, check if direction was correct
        # by looking at price change over evaluation horizon
        total = 0
        correct = 0

        # Note: This is simplified - full implementation would
        # compare snapshot predictions against actual price movements

        return {
            "total_signals": len(snapshots),
            "long_signals": sum(1 for s in snapshots if s.action == ConsensusAction.LONG),
            "short_signals": sum(1 for s in snapshots if s.action == ConsensusAction.SHORT),
            "avg_confidence": sum(s.confidence for s in snapshots) / len(snapshots),
        }
