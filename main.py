#!/usr/bin/env python3
"""
Main entry point - runs both trading bot and dashboard.

This runs:
1. FastAPI dashboard on PORT (default 8080)
2. Signal collection + trading strategies in background thread
"""

import os
import sys
import json
import logging
import threading
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import uvicorn
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from src.config import get_settings
from src.models import Base, Asset, Direction, Signal, Creator, SignalOutcome
from src.strategies.config import StrategyType
from src.strategies.multi_runner import MultiStrategyRunner
from src.strategies.backtester import HistoricalBacktester
from src.data_ingestion import XClient
from src.signal_extraction import SignalExtractor

# Import dashboard app
from dashboard import app, DATA_DIR, DB_PATH

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-8s | %(name)-20s | %(message)s',
    datefmt='%H:%M:%S',
)
logger = logging.getLogger(__name__)


def get_db_session():
    """Get database session for strategies.db."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    engine = create_engine(f"sqlite:///{DB_PATH}")
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    return Session(), engine


def collect_signals(session, settings, hours_back: int = 24, limit: int = 200):
    """
    Collect trading signals from X (Twitter).

    This fetches tweets from crypto traders and extracts trading signals.
    """
    logger.info(f"Collecting signals from X (last {hours_back}h, limit {limit})...")

    try:
        x_client = XClient(settings)
        extractor = SignalExtractor(use_nlp=False, min_confidence=0.4)

        # Fetch trading signals
        posts = x_client.fetch_trading_signals(limit=limit, hours_back=hours_back)
        logger.info(f"Fetched {len(posts)} posts from X")

        signals_added = 0
        creators_added = 0

        for post in posts:
            # Check if signal already exists
            existing = session.query(Signal).filter(Signal.post_id == post.post_id).first()
            if existing:
                continue

            # Extract trading signal from text
            extraction = extractor.extract(post.text)
            if not extractor.is_actionable(extraction):
                continue

            # Get or create creator
            creator = session.query(Creator).filter(Creator.username == post.username).first()
            if not creator:
                creator = Creator(
                    username=post.username,
                    display_name=post.raw_data.get('user', {}).get('display_name', post.username),
                )
                session.add(creator)
                session.flush()
                creators_added += 1

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
            signals_added += 1

        session.commit()

        if signals_added > 0 or creators_added > 0:
            logger.info(f"Added {signals_added} signals from {creators_added} new creators")
        else:
            logger.info("No new signals found")

        return signals_added

    except Exception as e:
        logger.warning(f"Signal collection failed: {e}")
        return 0


def run_trading_loop(interval_minutes: int = 30):
    """Background thread that runs signal collection + trading cycles."""
    logger.info(f"Starting trading loop (interval: {interval_minutes} min)")

    settings = get_settings()
    session, engine = get_db_session()

    # Run historical backtest on first run if no trades exist
    from src.models import Trade
    trade_count = session.query(Trade).count()
    signal_count = session.query(Signal).count()

    if trade_count < 10:
        logger.info("=" * 60)
        logger.info("RUNNING HISTORICAL BACKTEST")
        logger.info("=" * 60)
        logger.info("Fetching real signals and simulating trades...")

        backtester = HistoricalBacktester(
            settings=settings,
            session=session,
            initial_balance=10000.0,
        )

        # Backtest as far back as the API allows (typically 7-14 days)
        backtest_days = int(os.getenv("BACKTEST_DAYS", 14))
        backtester.run_backtest(days_back=backtest_days)

        logger.info("=" * 60)
        logger.info("BACKTEST COMPLETE - Starting live trading")
        logger.info("=" * 60)

    runner = MultiStrategyRunner(
        settings=settings,
        session=session,
        initial_balance=10000.0,
        strategies=list(StrategyType),
    )

    cycle = 0
    while True:
        try:
            cycle += 1
            logger.info("=" * 60)
            logger.info(f"TRADING CYCLE {cycle} - {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')} UTC")
            logger.info("=" * 60)

            # Step 1: Collect new signals from X
            collect_signals(session, settings, hours_back=24, limit=100)

            # Step 2: Run trading strategies
            results = runner.run_cycle()

            for strategy_type, actions in results.items():
                if actions:
                    logger.info(f"[{strategy_type.value}]:")
                    for action in actions:
                        logger.info(f"  -> {action}")
                else:
                    logger.info(f"[{strategy_type.value}]: No actions")

            # Step 3: Log performance summary
            performances = runner.get_performance_summary()
            logger.info("-" * 40)
            for st, perf in performances.items():
                logger.info(
                    f"{st.value}: ${perf.current_balance:,.2f} "
                    f"({perf.total_return_pct:+.2f}%) | "
                    f"Trades: {perf.total_trades} | "
                    f"WR: {perf.win_rate:.1%}"
                )

        except Exception as e:
            logger.error(f"Error in trading cycle: {e}", exc_info=True)

        # Wait for next cycle
        logger.info(f"Next cycle in {interval_minutes} minutes...")
        time.sleep(interval_minutes * 60)


def main():
    """Main entry point."""
    port = int(os.getenv("PORT", 8080))
    interval = int(os.getenv("TRADING_INTERVAL", 30))

    logger.info("=" * 60)
    logger.info("CRYPTO BOT STARTING")
    logger.info("=" * 60)
    logger.info(f"Dashboard: http://0.0.0.0:{port}")
    logger.info(f"Trading interval: {interval} minutes")
    logger.info("=" * 60)

    # Ensure data directory exists
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    # Start trading loop in background thread
    trading_thread = threading.Thread(
        target=run_trading_loop,
        args=(interval,),
        daemon=True,
    )
    trading_thread.start()

    # Run dashboard (blocks)
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")


if __name__ == "__main__":
    main()
