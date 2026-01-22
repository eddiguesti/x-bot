"""Signal evaluation against price movements."""

import logging
from datetime import datetime, timedelta
from typing import Optional

from sqlalchemy.orm import Session

from ..config import Settings
from ..models import Signal, SignalOutcome, Direction, Creator
from ..data_ingestion.price_client import PriceClient
from .glicko2 import Glicko2Calculator, Glicko2Rating

logger = logging.getLogger(__name__)


class SignalEvaluator:
    """
    Evaluates signals against actual price movements and updates creator ratings.
    """

    def __init__(self, settings: Settings, price_client: PriceClient):
        self.settings = settings
        self.price_client = price_client
        self.glicko = Glicko2Calculator(tau=0.5)

    def evaluate_pending_signals(self, session: Session, batch_size: int = 50) -> int:
        """
        Evaluate all pending signals that have passed their evaluation horizon.

        Args:
            session: Database session
            batch_size: Number of signals to process before committing (for cloud DBs)

        Returns:
            Number of signals evaluated
        """
        now = datetime.utcnow()
        horizon = timedelta(hours=self.settings.evaluation_horizon_hours)

        # Find signals ready for evaluation - get IDs first to avoid session issues
        pending_ids = session.query(Signal.id).filter(
            Signal.outcome == SignalOutcome.PENDING,
            Signal.posted_at <= now - horizon
        ).all()
        pending_ids = [id[0] for id in pending_ids]

        evaluated_count = 0
        total = len(pending_ids)

        # Process in batches to avoid connection timeouts
        for i in range(0, total, batch_size):
            batch_ids = pending_ids[i:i + batch_size]

            for signal_id in batch_ids:
                try:
                    signal = session.query(Signal).get(signal_id)
                    if signal:
                        success = self.evaluate_signal(signal, session)
                        if success:
                            evaluated_count += 1
                except Exception as e:
                    logger.warning(f"Error processing signal {signal_id}: {e}")
                    session.rollback()
                    continue

            # Commit after each batch
            try:
                session.commit()
                logger.info(f"Progress: {min(i + batch_size, total)}/{total} signals processed")
            except Exception as e:
                logger.warning(f"Batch commit failed: {e}")
                session.rollback()

        logger.info(f"Evaluated {evaluated_count} signals")

        return evaluated_count

    def evaluate_signal(self, signal: Signal, session: Session) -> bool:
        """
        Evaluate a single signal against price movement.

        Args:
            signal: Signal to evaluate
            session: Database session

        Returns:
            True if evaluation succeeded
        """
        try:
            # Calculate evaluation time
            eval_time = signal.posted_at + timedelta(
                hours=signal.evaluation_horizon_hours
            )

            # Get price at signal time
            price_at_signal = self.price_client.get_price_at_time(
                signal.asset,
                signal.posted_at
            )

            # Get price at evaluation time
            price_at_eval = self.price_client.get_price_at_time(
                signal.asset,
                eval_time
            )

            if price_at_signal is None or price_at_eval is None:
                logger.warning(f"Could not get prices for signal {signal.id}")
                return False

            # Calculate price change
            price_change = ((price_at_eval - price_at_signal) / price_at_signal) * 100

            # Determine outcome
            threshold = self.settings.price_move_threshold_percent

            if signal.direction == Direction.LONG:
                is_correct = price_change >= threshold
            else:  # SHORT
                is_correct = price_change <= -threshold

            # Update signal
            signal.price_at_signal = price_at_signal
            signal.price_at_evaluation = price_at_eval
            signal.price_change_percent = price_change
            signal.evaluated_at = datetime.utcnow()
            signal.outcome = SignalOutcome.CORRECT if is_correct else SignalOutcome.INCORRECT

            # Update creator stats and rating
            self._update_creator(signal.creator, is_correct, session)

            logger.debug(
                f"Signal {signal.id}: {signal.direction.value} "
                f"{'CORRECT' if is_correct else 'INCORRECT'} "
                f"(price change: {price_change:.2f}%)"
            )

            return True

        except Exception as e:
            logger.error(f"Error evaluating signal {signal.id}: {e}")
            return False

    def _update_creator(self, creator: Creator, is_correct: bool, session: Session):
        """Update creator's stats and Glicko-2 rating."""
        # Update simple stats
        creator.total_predictions += 1
        if is_correct:
            creator.correct_predictions += 1
        creator.last_prediction_at = datetime.utcnow()

        # Update Glicko-2 rating
        current_rating = Glicko2Rating(
            rating=creator.rating,
            rd=creator.rating_deviation,
            volatility=creator.volatility
        )

        new_rating = self.glicko.update_single(current_rating, is_correct)

        creator.rating = new_rating.rating
        creator.rating_deviation = new_rating.rd
        creator.volatility = new_rating.volatility

        logger.debug(
            f"Creator {creator.username}: "
            f"rating {current_rating.rating:.0f} -> {new_rating.rating:.0f}"
        )

    def backfill_evaluations(
        self,
        session: Session,
        start_date: Optional[datetime] = None
    ) -> int:
        """
        Backfill evaluations for historical signals.

        Useful when first setting up the system or after fixing price data issues.

        Args:
            session: Database session
            start_date: Only evaluate signals after this date

        Returns:
            Number of signals evaluated
        """
        query = session.query(Signal).filter(
            Signal.outcome == SignalOutcome.PENDING
        )

        if start_date:
            query = query.filter(Signal.posted_at >= start_date)

        signals = query.order_by(Signal.posted_at.asc()).all()

        evaluated = 0
        for signal in signals:
            if self.evaluate_signal(signal, session):
                evaluated += 1

        session.commit()
        return evaluated

    def get_evaluation_stats(self, session: Session) -> dict:
        """Get statistics about signal evaluations."""
        total = session.query(Signal).count()
        pending = session.query(Signal).filter(
            Signal.outcome == SignalOutcome.PENDING
        ).count()
        correct = session.query(Signal).filter(
            Signal.outcome == SignalOutcome.CORRECT
        ).count()
        incorrect = session.query(Signal).filter(
            Signal.outcome == SignalOutcome.INCORRECT
        ).count()

        evaluated = correct + incorrect
        accuracy = correct / evaluated if evaluated > 0 else 0

        return {
            "total_signals": total,
            "pending": pending,
            "evaluated": evaluated,
            "correct": correct,
            "incorrect": incorrect,
            "overall_accuracy": accuracy,
        }
