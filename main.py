#!/usr/bin/env python3
"""
Main entry point - runs both trading bot and dashboard.

This runs:
1. FastAPI dashboard on PORT (default 8080)
2. Trading strategies in background thread every 30 minutes
"""

import os
import sys
import logging
import threading
import time
from datetime import datetime

import uvicorn
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from src.config import get_settings
from src.models import Base, Asset
from src.strategies.config import StrategyType
from src.strategies.multi_runner import MultiStrategyRunner

# Import dashboard app
from dashboard import app, DATA_DIR, DB_PATH

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-8s | %(name)-20s | %(message)s',
    datefmt='%H:%M:%S',
)
logger = logging.getLogger(__name__)


def setup_database(settings):
    """Initialize database connection."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    engine = create_engine(f"sqlite:///{DB_PATH}")
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    return Session()


def run_trading_loop(interval_minutes: int = 30):
    """Background thread that runs trading cycles."""
    logger.info(f"Starting trading loop (interval: {interval_minutes} min)")

    settings = get_settings()
    session = setup_database(settings)

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
            logger.info(f"=" * 60)
            logger.info(f"TRADING CYCLE {cycle} - {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC")
            logger.info(f"=" * 60)

            results = runner.run_cycle()

            for strategy_type, actions in results.items():
                if actions:
                    logger.info(f"[{strategy_type.value}]:")
                    for action in actions:
                        logger.info(f"  -> {action}")
                else:
                    logger.info(f"[{strategy_type.value}]: No actions")

            # Log performance summary
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
