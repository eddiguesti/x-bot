#!/usr/bin/env python3
"""
Main entry point - runs both trading bot and dashboard.

This runs:
1. FastAPI dashboard on PORT (default 8080)
2. Signal collection + trading strategies in background thread
"""

import os
import sys
import signal
import json
import logging
import threading
import time
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

import uvicorn
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session, scoped_session
from sqlalchemy.exc import IntegrityError

from src.config import get_settings
from src.models import Base, Asset, Direction, Signal, Creator, SignalOutcome, SignalSource
from src.strategies.config import StrategyType
from src.strategies.multi_runner import MultiStrategyRunner
from src.strategies.backtester import HistoricalBacktester
from src.data_ingestion import XClient, RedditClient, YouTubeClient
from src.signal_extraction import SignalExtractor, LLMSignalExtractor
from src.constants import (
    MAX_POST_TEXT_LENGTH,
    DEFAULT_LOOKBACK_HOURS_TWITTER,
    DEFAULT_LOOKBACK_HOURS_REDDIT,
    DEFAULT_LOOKBACK_HOURS_YOUTUBE,
    DEFAULT_LIMIT_TWITTER,
    DEFAULT_LIMIT_REDDIT,
    DEFAULT_LIMIT_YOUTUBE,
    MAX_YOUTUBE_TRANSCRIPTS,
    LLM_MIN_CONFIDENCE_THRESHOLD,
    RULE_BASED_MIN_CONFIDENCE_THRESHOLD,
    DEFAULT_TRADING_INTERVAL_MINUTES,
)

# Import dashboard app
from dashboard import app, DATA_DIR, DB_PATH

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-8s | %(name)-20s | %(message)s',
    datefmt='%H:%M:%S',
)
logger = logging.getLogger(__name__)

# Graceful shutdown event
shutdown_event = threading.Event()


def signal_handler(signum, frame):
    """Handle shutdown signals gracefully."""
    sig_name = signal.Signals(signum).name
    logger.info(f"Received {sig_name}, initiating graceful shutdown...")
    shutdown_event.set()


# Register signal handlers
signal.signal(signal.SIGTERM, signal_handler)
signal.signal(signal.SIGINT, signal_handler)


def create_engine_with_config():
    """Create SQLAlchemy engine with proper configuration."""
    return create_engine(
        f"sqlite:///{DB_PATH}",
        connect_args={
            "check_same_thread": False,  # Allow multi-threaded access
            "timeout": 30,  # Wait up to 30s for locks
        },
        pool_pre_ping=True,  # Verify connections before use
    )


def get_db_session():
    """Get database session for strategies.db (thread-local)."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    engine = create_engine_with_config()
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    return Session(), engine


@contextmanager
def transaction(session: Session):
    """Context manager for database transactions with proper rollback."""
    try:
        yield session
        session.commit()
    except Exception as e:
        session.rollback()
        raise


def get_extractor(settings):
    """Get the best available signal extractor (LLM preferred)."""
    llm_extractor = LLMSignalExtractor(settings)
    if llm_extractor.enabled:
        logger.info("Using LLM-based signal extraction (Gemini/DeepSeek)")
        return llm_extractor, LLM_MIN_CONFIDENCE_THRESHOLD
    else:
        logger.info("Using rule-based signal extraction (no LLM API keys)")
        return SignalExtractor(use_nlp=False, min_confidence=RULE_BASED_MIN_CONFIDENCE_THRESHOLD), RULE_BASED_MIN_CONFIDENCE_THRESHOLD


def batch_check_existing_posts(session: Session, post_ids: list[str]) -> set[str]:
    """Batch check which post_ids already exist in database."""
    if not post_ids:
        return set()

    existing = session.query(Signal.post_id).filter(
        Signal.post_id.in_(post_ids)
    ).all()
    return {row[0] for row in existing}


def collect_signals(
    session: Session,
    settings,
    hours_back: int = DEFAULT_LOOKBACK_HOURS_TWITTER,
    limit: int = DEFAULT_LIMIT_TWITTER,
    extractor=None,
    min_confidence: float = RULE_BASED_MIN_CONFIDENCE_THRESHOLD
) -> int:
    """
    Collect trading signals from X (Twitter).

    This fetches tweets from crypto traders and extracts trading signals.
    Optimized to check DB existence before running LLM extraction.
    """
    logger.info(f"Collecting signals from X (last {hours_back}h, limit {limit})...")

    try:
        x_client = XClient(settings)
        if extractor is None:
            extractor, min_confidence = get_extractor(settings)

        # Fetch trading signals
        posts = x_client.fetch_trading_signals(limit=limit, hours_back=hours_back)
        logger.info(f"Fetched {len(posts)} posts from X")

        if not posts:
            return 0

        # Batch check which posts already exist (before running LLM)
        post_ids = [p.post_id for p in posts]
        existing_ids = batch_check_existing_posts(session, post_ids)

        # Filter out existing posts BEFORE LLM extraction
        new_posts = [p for p in posts if p.post_id not in existing_ids]
        logger.info(f"Found {len(new_posts)} new posts to process")

        signals_added = 0
        creators_added = 0

        with transaction(session):
            for post in new_posts:
                # Extract trading signal from text (LLM call)
                extraction = extractor.extract(post.text)
                if not extractor.is_actionable(extraction, min_confidence):
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

                # Create signal with IntegrityError handling for race conditions
                try:
                    signal = Signal(
                        creator_id=creator.id,
                        post_id=post.post_id,
                        post_text=post.text[:MAX_POST_TEXT_LENGTH],
                        post_url=post.url,
                        posted_at=post.posted_at,
                        asset=extraction.asset,
                        direction=extraction.direction,
                        confidence=extraction.confidence,
                        source=SignalSource.TWITTER,
                    )
                    session.add(signal)
                    session.flush()  # Check for constraint violation
                    signals_added += 1
                except IntegrityError:
                    session.rollback()
                    logger.debug(f"Signal {post.post_id} already exists (race condition)")
                    continue

        if signals_added > 0 or creators_added > 0:
            logger.info(f"Added {signals_added} signals from {creators_added} new creators")
        else:
            logger.info("No new signals found")

        return signals_added

    except Exception as e:
        logger.exception(f"Signal collection failed: {e}")
        return 0


def collect_reddit_signals(
    session: Session,
    settings,
    hours_back: int = DEFAULT_LOOKBACK_HOURS_REDDIT,
    limit: int = DEFAULT_LIMIT_REDDIT,
    extractor=None,
    min_confidence: float = RULE_BASED_MIN_CONFIDENCE_THRESHOLD
) -> int:
    """
    Collect trading signals from Reddit crypto subreddits.

    This fetches posts from r/cryptocurrency, r/bitcoin, etc. and extracts signals.
    """
    logger.info(f"Collecting signals from Reddit (last {hours_back}h, limit {limit})...")

    try:
        reddit_client = RedditClient(settings)
        if not reddit_client.enabled:
            logger.warning("Reddit client not enabled - skipping Reddit signals")
            return 0

        if extractor is None:
            extractor, min_confidence = get_extractor(settings)

        # Fetch trading signals from crypto subreddits
        posts = reddit_client.fetch_trading_signals(limit=limit, hours_back=hours_back)
        logger.info(f"Fetched {len(posts)} posts from Reddit")

        if not posts:
            return 0

        # Batch check which posts already exist (before running LLM)
        post_ids = [p.post_id for p in posts]
        existing_ids = batch_check_existing_posts(session, post_ids)

        new_posts = [p for p in posts if p.post_id not in existing_ids]
        logger.info(f"Found {len(new_posts)} new Reddit posts to process")

        signals_added = 0
        creators_added = 0

        with transaction(session):
            for post in new_posts:
                # Extract trading signal from text
                extraction = extractor.extract(post.text)
                if not extractor.is_actionable(extraction, min_confidence):
                    continue

                # Get or create creator (prefix with r/ to distinguish from Twitter)
                reddit_username = f"r/{post.username}"
                creator = session.query(Creator).filter(Creator.username == reddit_username).first()
                if not creator:
                    creator = Creator(
                        username=reddit_username,
                        display_name=f"Reddit: {post.username}",
                    )
                    session.add(creator)
                    session.flush()
                    creators_added += 1

                # Create signal with IntegrityError handling
                try:
                    signal = Signal(
                        creator_id=creator.id,
                        post_id=post.post_id,
                        post_text=post.text[:MAX_POST_TEXT_LENGTH],
                        post_url=post.url,
                        posted_at=post.posted_at,
                        asset=extraction.asset,
                        direction=extraction.direction,
                        confidence=extraction.confidence,
                        source=SignalSource.REDDIT,
                    )
                    session.add(signal)
                    session.flush()
                    signals_added += 1
                except IntegrityError:
                    session.rollback()
                    logger.debug(f"Signal {post.post_id} already exists (race condition)")
                    continue

        if signals_added > 0 or creators_added > 0:
            logger.info(f"Added {signals_added} Reddit signals from {creators_added} new creators")
        else:
            logger.info("No new Reddit signals found")

        return signals_added

    except Exception as e:
        logger.exception(f"Reddit signal collection failed: {e}")
        return 0


def collect_youtube_signals(
    session: Session,
    settings,
    hours_back: int = DEFAULT_LOOKBACK_HOURS_YOUTUBE,
    limit: int = DEFAULT_LIMIT_YOUTUBE,
    extractor=None,
    min_confidence: float = RULE_BASED_MIN_CONFIDENCE_THRESHOLD
) -> int:
    """
    Collect trading signals from YouTube crypto channels.

    YouTube content is less frequent but often higher quality analysis.
    Fetches transcripts for top videos when LLM is available.
    """
    logger.info(f"Collecting signals from YouTube (last {hours_back}h, limit {limit})...")

    try:
        youtube_client = YouTubeClient(settings)
        if not youtube_client.enabled:
            logger.warning("YouTube client not enabled - skipping YouTube signals")
            return 0

        if extractor is None:
            extractor, min_confidence = get_extractor(settings)

        # Fetch trading signals from crypto YouTube channels
        posts = youtube_client.fetch_trading_signals(limit=limit, hours_back=hours_back)
        logger.info(f"Fetched {len(posts)} videos from YouTube")

        if not posts:
            return 0

        # Batch check which posts already exist (before running LLM)
        post_ids = [p.post_id for p in posts]
        existing_ids = batch_check_existing_posts(session, post_ids)

        new_posts = [p for p in posts if p.post_id not in existing_ids]
        logger.info(f"Found {len(new_posts)} new YouTube videos to process")

        # Enrich top videos with transcripts (only if using LLM and there are new posts)
        if new_posts and isinstance(extractor, LLMSignalExtractor) and extractor.enabled:
            logger.info("Fetching transcripts for top YouTube videos...")
            new_posts = youtube_client.enrich_with_transcripts(new_posts, max_posts=MAX_YOUTUBE_TRANSCRIPTS)

        signals_added = 0
        creators_added = 0

        with transaction(session):
            for post in new_posts:
                # Extract trading signal from video title + description + transcript
                extraction = extractor.extract(post.text)
                if not extractor.is_actionable(extraction, min_confidence):
                    continue

                # Get or create creator (prefix with yt/ to distinguish from Twitter/Reddit)
                youtube_username = f"yt/{post.username}"
                creator = session.query(Creator).filter(Creator.username == youtube_username).first()
                if not creator:
                    creator = Creator(
                        username=youtube_username,
                        display_name=f"YouTube: {post.username}",
                    )
                    session.add(creator)
                    session.flush()
                    creators_added += 1

                # Create signal with IntegrityError handling
                try:
                    signal = Signal(
                        creator_id=creator.id,
                        post_id=post.post_id,
                        post_text=post.text[:MAX_POST_TEXT_LENGTH],
                        post_url=post.url,
                        posted_at=post.posted_at,
                        asset=extraction.asset,
                        direction=extraction.direction,
                        confidence=extraction.confidence,
                        source=SignalSource.YOUTUBE,
                    )
                    session.add(signal)
                    session.flush()
                    signals_added += 1
                except IntegrityError:
                    session.rollback()
                    logger.debug(f"Signal {post.post_id} already exists (race condition)")
                    continue

        if signals_added > 0 or creators_added > 0:
            logger.info(f"Added {signals_added} YouTube signals from {creators_added} new creators")
        else:
            logger.info("No new YouTube signals found")

        return signals_added

    except Exception as e:
        logger.exception(f"YouTube signal collection failed: {e}")
        return 0


def run_trading_loop(interval_minutes: int = DEFAULT_TRADING_INTERVAL_MINUTES):
    """
    Background thread that runs signal collection + trading cycles.

    Creates its own database session (thread-safe) and handles graceful shutdown.
    """
    logger.info(f"Starting trading loop (interval: {interval_minutes} min)")

    # Create thread-local session (NOT shared with main thread)
    try:
        settings = get_settings()
        session, engine = get_db_session()
    except Exception as e:
        logger.exception(f"Failed to initialize trading loop: {e}")
        return

    try:
        # FIRST: Create the runner to ensure portfolios exist
        logger.info("Initializing strategy portfolios...")
        try:
            runner = MultiStrategyRunner(
                settings=settings,
                session=session,
                initial_balance=10000.0,
                strategies=list(StrategyType),
            )
            logger.info("Portfolios initialized successfully")
        except Exception as e:
            logger.exception(f"Failed to create strategy runner: {e}")
            return

        # THEN: Try historical backtest (optional - don't fail if it doesn't work)
        from src.models import Trade
        trade_count = session.query(Trade).count()

        if trade_count < 10:
            logger.info("=" * 60)
            logger.info("RUNNING HISTORICAL BACKTEST")
            logger.info("=" * 60)

            try:
                backtester = HistoricalBacktester(
                    settings=settings,
                    session=session,
                    initial_balance=10000.0,
                )
                backtest_days = int(os.getenv("BACKTEST_DAYS", 90))
                backtester.run_backtest(days_back=backtest_days)
                logger.info("Backtest complete")
            except Exception as e:
                logger.warning(f"Backtest failed (continuing anyway): {e}")

            logger.info("=" * 60)
            logger.info("Starting live trading")
            logger.info("=" * 60)

        # Re-initialize runner to pick up any backtest data
        runner = MultiStrategyRunner(
            settings=settings,
            session=session,
            initial_balance=10000.0,
            strategies=list(StrategyType),
        )

        cycle = 0
        while not shutdown_event.is_set():
            try:
                cycle += 1
                logger.info("=" * 60)
                logger.info(f"TRADING CYCLE {cycle} - {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')} UTC")
                logger.info("=" * 60)

                # Step 1: Initialize extractor (LLM if available, else rule-based)
                extractor, min_confidence = get_extractor(settings)

                # Step 2: Collect new signals from X (Twitter), Reddit, and YouTube
                collect_signals(
                    session, settings,
                    hours_back=DEFAULT_LOOKBACK_HOURS_TWITTER,
                    limit=DEFAULT_LIMIT_TWITTER,
                    extractor=extractor,
                    min_confidence=min_confidence
                )
                collect_reddit_signals(
                    session, settings,
                    hours_back=DEFAULT_LOOKBACK_HOURS_REDDIT,
                    limit=DEFAULT_LIMIT_REDDIT,
                    extractor=extractor,
                    min_confidence=min_confidence
                )
                collect_youtube_signals(
                    session, settings,
                    hours_back=DEFAULT_LOOKBACK_HOURS_YOUTUBE,
                    limit=DEFAULT_LIMIT_YOUTUBE,
                    extractor=extractor,
                    min_confidence=min_confidence
                )

                # Step 3: Run trading strategies
                results = runner.run_cycle()

                for strategy_type, actions in results.items():
                    if actions:
                        logger.info(f"[{strategy_type.value}]:")
                        for action in actions:
                            logger.info(f"  -> {action}")
                    else:
                        logger.info(f"[{strategy_type.value}]: No actions")

                # Step 4: Log performance summary
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
                logger.exception(f"Error in trading cycle: {e}")

            # Wait for next cycle or shutdown signal
            logger.info(f"Next cycle in {interval_minutes} minutes...")
            shutdown_event.wait(timeout=interval_minutes * 60)

    finally:
        # Clean up session on thread exit
        logger.info("Trading loop shutting down, closing database session...")
        session.close()
        logger.info("Trading loop shutdown complete")


def main():
    """Main entry point."""
    port = int(os.getenv("PORT", 8080))
    interval = int(os.getenv("TRADING_INTERVAL", DEFAULT_TRADING_INTERVAL_MINUTES))

    logger.info("=" * 60)
    logger.info("CRYPTO BOT STARTING")
    logger.info("=" * 60)
    logger.info(f"Dashboard: http://0.0.0.0:{port}")
    logger.info(f"Trading interval: {interval} minutes")
    logger.info("=" * 60)

    # Ensure data directory exists
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    # Start trading loop in background thread (NOT daemon - we handle shutdown)
    trading_thread = threading.Thread(
        target=run_trading_loop,
        args=(interval,),
        name="TradingLoop",
    )
    trading_thread.start()

    try:
        # Run dashboard (blocks)
        uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")
    finally:
        # Signal shutdown and wait for trading thread
        logger.info("Dashboard stopped, signaling trading loop shutdown...")
        shutdown_event.set()
        trading_thread.join(timeout=30)
        if trading_thread.is_alive():
            logger.warning("Trading thread did not stop gracefully")
        else:
            logger.info("Clean shutdown complete")


if __name__ == "__main__":
    main()
