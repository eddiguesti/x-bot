"""Data exporter for ZIP bundling of all system outputs."""

import json
import logging
import shutil
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Optional

from sqlalchemy.orm import Session

from ..config import Settings
from ..models import Creator, Signal, ConsensusSnapshot
from .reports import ReportGenerator

logger = logging.getLogger(__name__)


class DataExporter:
    """Export all system data to a single ZIP file."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self.report_generator = ReportGenerator(settings)

    def export_all(
        self,
        session: Session,
        output_path: Optional[Path] = None,
    ) -> Path:
        """
        Export all data and reports to a ZIP file.

        Args:
            session: Database session
            output_path: Output ZIP path (default: auto-generated)

        Returns:
            Path to created ZIP file
        """
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")

        if output_path is None:
            output_path = self.settings.base_dir / f"crypto_consensus_export_{timestamp}.zip"

        # Create temp directory for export
        temp_dir = self.settings.base_dir / f"_export_temp_{timestamp}"
        temp_dir.mkdir(exist_ok=True)

        try:
            # Export reports
            reports_dir = temp_dir / "reports"
            reports_dir.mkdir()
            self._export_reports(session, reports_dir)

            # Export raw data
            data_dir = temp_dir / "data"
            data_dir.mkdir()
            self._export_data(session, data_dir)

            # Export configuration
            config_dir = temp_dir / "config"
            config_dir.mkdir()
            self._export_config(config_dir)

            # Create metadata
            self._create_metadata(temp_dir, session)

            # Create ZIP
            self._create_zip(temp_dir, output_path)

            logger.info(f"Export complete: {output_path}")
            return output_path

        finally:
            # Cleanup temp directory
            shutil.rmtree(temp_dir, ignore_errors=True)

    def _export_reports(self, session: Session, output_dir: Path):
        """Export all reports to directory."""
        # Generate fresh reports
        rankings = self.report_generator.generate_creator_rankings(session)
        rankings.to_csv(output_dir / "creator_rankings.csv", index=False)

        signals = self.report_generator.generate_signal_history(session, days=90)
        signals.to_csv(output_dir / "signal_history.csv", index=False)

        summary = self.report_generator.generate_performance_summary(session, days=90)
        with open(output_dir / "summary.json", 'w') as f:
            json.dump(summary, f, indent=2, default=str)

    def _export_data(self, session: Session, output_dir: Path):
        """Export raw data tables to CSV."""
        # Creators
        creators = session.query(Creator).all()
        creator_data = []
        for c in creators:
            creator_data.append({
                "id": c.id,
                "username": c.username,
                "display_name": c.display_name,
                "rating": c.rating,
                "rating_deviation": c.rating_deviation,
                "volatility": c.volatility,
                "total_predictions": c.total_predictions,
                "correct_predictions": c.correct_predictions,
                "is_active": c.is_active,
                "added_at": c.added_at,
                "last_prediction_at": c.last_prediction_at,
            })

        import pandas as pd
        pd.DataFrame(creator_data).to_csv(output_dir / "creators.csv", index=False)

        # Signals
        signals = session.query(Signal).all()
        signal_data = []
        for s in signals:
            signal_data.append({
                "id": s.id,
                "creator_id": s.creator_id,
                "post_id": s.post_id,
                "asset": s.asset.value,
                "direction": s.direction.value,
                "confidence": s.confidence,
                "outcome": s.outcome.value,
                "posted_at": s.posted_at,
                "price_at_signal": s.price_at_signal,
                "price_at_evaluation": s.price_at_evaluation,
                "price_change_percent": s.price_change_percent,
                "evaluated_at": s.evaluated_at,
            })
        pd.DataFrame(signal_data).to_csv(output_dir / "signals.csv", index=False)

        # Consensus snapshots
        snapshots = session.query(ConsensusSnapshot).all()
        snapshot_data = []
        for snap in snapshots:
            snapshot_data.append({
                "id": snap.id,
                "asset": snap.asset.value,
                "timestamp": snap.timestamp,
                "weighted_score": snap.weighted_score,
                "action": snap.action.value if snap.action else None,
                "confidence": snap.confidence,
                "long_votes": snap.long_votes,
                "short_votes": snap.short_votes,
            })
        pd.DataFrame(snapshot_data).to_csv(output_dir / "consensus_history.csv", index=False)

    def _export_config(self, output_dir: Path):
        """Export configuration (without secrets)."""
        # Export creator list if exists
        creators_file = self.settings.base_dir / "config" / "creators.json"
        if creators_file.exists():
            shutil.copy(creators_file, output_dir / "creators.json")

        # Export settings (sanitized)
        config = {
            "evaluation_horizon_hours": self.settings.evaluation_horizon_hours,
            "consensus_threshold": self.settings.consensus_threshold,
            "price_move_threshold_percent": self.settings.price_move_threshold_percent,
            "min_predictions_for_ranking": self.settings.min_predictions_for_ranking,
            "strong_signal_threshold": self.settings.strong_signal_threshold,
            "weak_signal_threshold": self.settings.weak_signal_threshold,
        }
        with open(output_dir / "settings.json", 'w') as f:
            json.dump(config, f, indent=2)

    def _create_metadata(self, output_dir: Path, session: Session):
        """Create export metadata file."""
        metadata = {
            "export_timestamp": datetime.utcnow().isoformat(),
            "version": "0.1.0",
            "stats": {
                "total_creators": session.query(Creator).count(),
                "total_signals": session.query(Signal).count(),
                "total_consensus_snapshots": session.query(ConsensusSnapshot).count(),
            },
        }
        with open(output_dir / "metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)

    def _create_zip(self, source_dir: Path, output_path: Path):
        """Create ZIP file from directory."""
        with zipfile.ZipFile(output_path, 'w', zipfile.ZIP_DEFLATED) as zf:
            for file_path in source_dir.rglob('*'):
                if file_path.is_file():
                    arcname = file_path.relative_to(source_dir)
                    zf.write(file_path, arcname)
