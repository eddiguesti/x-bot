"""Report generation for rankings and consensus."""

import logging
from datetime import datetime
from typing import Optional

import pandas as pd
from sqlalchemy.orm import Session

from ..config import Settings
from ..models import Creator, Signal, SignalOutcome, Asset, CreatorStats

logger = logging.getLogger(__name__)


class ReportGenerator:
    """Generate reports and rankings from system data."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self.reports_dir = settings.reports_dir
        self.reports_dir.mkdir(parents=True, exist_ok=True)

    def generate_creator_rankings(
        self,
        session: Session,
        min_predictions: Optional[int] = None,
    ) -> pd.DataFrame:
        """
        Generate creator rankings sorted by Glicko-2 rating.

        Args:
            session: Database session
            min_predictions: Minimum predictions to include (default from settings)

        Returns:
            DataFrame with creator rankings
        """
        if min_predictions is None:
            min_predictions = self.settings.min_predictions_for_ranking

        creators = session.query(Creator).filter(
            Creator.is_active == True,
            Creator.total_predictions >= min_predictions
        ).order_by(Creator.rating.desc()).all()

        data = []
        for rank, creator in enumerate(creators, 1):
            data.append({
                "rank": rank,
                "username": creator.username,
                "display_name": creator.display_name or creator.username,
                "rating": round(creator.rating, 1),
                "rating_deviation": round(creator.rating_deviation, 1),
                "volatility": round(creator.volatility, 4),
                "accuracy": round(creator.accuracy * 100, 1),
                "total_predictions": creator.total_predictions,
                "correct_predictions": creator.correct_predictions,
                "weight": round(creator.weight, 4),
                "last_prediction": creator.last_prediction_at,
            })

        df = pd.DataFrame(data)
        return df

    def generate_signal_history(
        self,
        session: Session,
        asset: Optional[Asset] = None,
        days: int = 30,
    ) -> pd.DataFrame:
        """
        Generate signal history report.

        Args:
            session: Database session
            asset: Filter by asset (optional)
            days: Days of history

        Returns:
            DataFrame with signal history
        """
        from datetime import timedelta
        cutoff = datetime.utcnow() - timedelta(days=days)

        query = session.query(Signal).filter(
            Signal.posted_at >= cutoff
        )

        if asset:
            query = query.filter(Signal.asset == asset)

        signals = query.order_by(Signal.posted_at.desc()).all()

        data = []
        for signal in signals:
            data.append({
                "id": signal.id,
                "creator": signal.creator.username,
                "asset": signal.asset.value,
                "direction": signal.direction.value,
                "confidence": round(signal.confidence, 3),
                "posted_at": signal.posted_at,
                "outcome": signal.outcome.value,
                "price_at_signal": signal.price_at_signal,
                "price_at_evaluation": signal.price_at_evaluation,
                "price_change_percent": round(signal.price_change_percent, 2) if signal.price_change_percent else None,
                "post_text": signal.post_text[:100] + "..." if len(signal.post_text) > 100 else signal.post_text,
            })

        return pd.DataFrame(data)

    def generate_performance_summary(
        self,
        session: Session,
        days: int = 30,
    ) -> dict:
        """
        Generate overall performance summary.

        Args:
            session: Database session
            days: Days of history

        Returns:
            Dict with performance metrics
        """
        from datetime import timedelta
        cutoff = datetime.utcnow() - timedelta(days=days)

        # Creator stats
        total_creators = session.query(Creator).filter(Creator.is_active == True).count()
        ranked_creators = session.query(Creator).filter(
            Creator.is_active == True,
            Creator.total_predictions >= self.settings.min_predictions_for_ranking
        ).count()

        # Signal stats
        total_signals = session.query(Signal).filter(Signal.posted_at >= cutoff).count()
        evaluated = session.query(Signal).filter(
            Signal.posted_at >= cutoff,
            Signal.outcome != SignalOutcome.PENDING
        ).count()
        correct = session.query(Signal).filter(
            Signal.posted_at >= cutoff,
            Signal.outcome == SignalOutcome.CORRECT
        ).count()

        # By asset
        btc_signals = session.query(Signal).filter(
            Signal.posted_at >= cutoff,
            Signal.asset == Asset.BTC
        ).count()
        eth_signals = session.query(Signal).filter(
            Signal.posted_at >= cutoff,
            Signal.asset == Asset.ETH
        ).count()

        # Top performers
        top_creators = session.query(Creator).filter(
            Creator.is_active == True,
            Creator.total_predictions >= self.settings.min_predictions_for_ranking
        ).order_by(Creator.rating.desc()).limit(5).all()

        return {
            "period_days": days,
            "generated_at": datetime.utcnow().isoformat(),
            "creators": {
                "total_tracked": total_creators,
                "ranked": ranked_creators,
            },
            "signals": {
                "total": total_signals,
                "evaluated": evaluated,
                "pending": total_signals - evaluated,
                "correct": correct,
                "accuracy": correct / evaluated if evaluated > 0 else 0,
            },
            "by_asset": {
                "BTC": btc_signals,
                "ETH": eth_signals,
            },
            "top_performers": [
                {
                    "username": c.username,
                    "rating": round(c.rating, 1),
                    "accuracy": round(c.accuracy * 100, 1),
                }
                for c in top_creators
            ],
        }

    def save_reports(
        self,
        session: Session,
        prefix: Optional[str] = None,
    ) -> list[str]:
        """
        Generate and save all reports.

        Args:
            session: Database session
            prefix: Filename prefix (default: timestamp)

        Returns:
            List of saved file paths
        """
        if prefix is None:
            prefix = datetime.utcnow().strftime("%Y%m%d_%H%M%S")

        saved_files = []

        # Rankings
        rankings = self.generate_creator_rankings(session)
        rankings_path = self.reports_dir / f"{prefix}_creator_rankings.csv"
        rankings.to_csv(rankings_path, index=False)
        saved_files.append(str(rankings_path))

        # Signal history
        signals = self.generate_signal_history(session)
        signals_path = self.reports_dir / f"{prefix}_signal_history.csv"
        signals.to_csv(signals_path, index=False)
        saved_files.append(str(signals_path))

        # Summary (JSON)
        import json
        summary = self.generate_performance_summary(session)
        summary_path = self.reports_dir / f"{prefix}_summary.json"
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        saved_files.append(str(summary_path))

        logger.info(f"Saved {len(saved_files)} reports to {self.reports_dir}")
        return saved_files
