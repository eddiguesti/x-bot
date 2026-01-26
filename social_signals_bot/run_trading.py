#!/usr/bin/env python3
"""Continuous trading loop with signal collection."""

import logging
import time
from datetime import datetime

from src.config import Settings
from src.models import init_db, get_session, Asset, Direction, ConsensusAction
from src.trading.paper_trader import PaperTrader
from src.consensus.engine import ConsensusEngine
from src.data_ingestion.x_client import XClient
from src.signal_extraction.rules import RuleBasedExtractor

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-8s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


def collect_signals(settings, session, limit=100):
    """Try to collect new signals."""
    try:
        x_client = XClient(settings)
        extractor = RuleBasedExtractor()

        # Fetch with trading keywords
        posts = x_client.fetch_posts(
            usernames=[],
            limit=limit,
        )

        if not posts:
            return 0

        from src.models import Creator, Signal

        new_signals = 0
        for post in posts:
            # Check if already processed
            existing = session.query(Signal).filter(
                Signal.post_id == post.post_id
            ).first()
            if existing:
                continue

            # Extract signal
            extraction = extractor.extract(post.text)
            if not extraction.asset or not extraction.direction:
                continue

            # Get or create creator
            creator = session.query(Creator).filter(
                Creator.username == post.username
            ).first()
            if not creator:
                creator = Creator(username=post.username, display_name=post.username)
                session.add(creator)
                session.flush()

            # Create signal
            signal = Signal(
                creator_id=creator.id,
                post_id=post.post_id,
                post_text=post.text,
                post_url=post.url,
                posted_at=post.posted_at,
                asset=extraction.asset,
                direction=extraction.direction,
                confidence=extraction.confidence,
            )
            session.add(signal)
            new_signals += 1

        session.commit()
        return new_signals

    except Exception as e:
        logger.warning(f"Signal collection failed: {e}")
        return 0


def run_trading_loop(interval_minutes=5, collect_interval=3):
    """Run continuous trading loop."""
    settings = Settings()
    engine = init_db(settings.database_url)
    session = get_session(engine)

    trader = PaperTrader(settings, session)
    consensus = ConsensusEngine(settings)

    logger.info("=" * 60)
    logger.info("STARTING CONTINUOUS TRADING LOOP")
    logger.info(f"Trading interval: {interval_minutes} min")
    logger.info(f"Signal collection: every {collect_interval} cycles")
    logger.info("=" * 60)

    cycle = 0
    while True:
        try:
            cycle += 1
            logger.info(f"\n{'='*40} CYCLE {cycle} {'='*40}")

            # Try to collect signals every N cycles
            if cycle % collect_interval == 1:
                logger.info("Attempting signal collection...")
                new_signals = collect_signals(settings, session)
                if new_signals > 0:
                    logger.info(f"Collected {new_signals} new signals")

            # Check existing positions for SL/TP
            to_close = trader.check_positions()
            for trade, reason in to_close:
                trader.close_position(trade, reason)

            # Get consensus for all assets
            results = consensus.get_all_consensus(session, lookback_hours=48)

            # Find actionable signals
            actionable = []
            for asset, result in results.items():
                if consensus.should_trade(result):
                    # Check if we already have a position
                    existing = any(
                        t.asset == asset
                        for t in trader.get_open_positions()
                    )
                    if not existing:
                        actionable.append((asset, result))

            if actionable:
                logger.info(f"\nActionable signals: {len(actionable)}")
                for asset, result in actionable:
                    direction = Direction.LONG if result.action == ConsensusAction.LONG else Direction.SHORT
                    logger.info(f"  {asset.value}: {result.action.value} ({result.confidence:.1f}%)")

                    # Open position
                    trade = trader.open_position(
                        asset=asset,
                        direction=direction,
                        consensus_score=result.weighted_score,
                        consensus_confidence=result.confidence,
                    )
                    if trade:
                        logger.info(f"  -> Opened {direction.value} {asset.value}")
            else:
                logger.info("No new trade signals")

            # Display status
            trader.display_status()

            # Wait for next cycle
            logger.info(f"\nSleeping {interval_minutes} minutes...")
            time.sleep(interval_minutes * 60)

        except KeyboardInterrupt:
            logger.info("\nShutting down...")
            break
        except Exception as e:
            logger.error(f"Error in trading loop: {e}")
            time.sleep(60)  # Wait 1 min on error

    session.close()


if __name__ == "__main__":
    run_trading_loop()
