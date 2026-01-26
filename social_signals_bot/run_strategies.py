#!/usr/bin/env python3
"""
Multi-Strategy A/B Testing Runner

Runs 3 different trading strategies in parallel to find the best approach:
1. SOCIAL_PURE - Pure social signals, minimal filters
2. TECHNICAL_STRICT - Heavy technical confirmation
3. BALANCED - Hybrid approach

Usage:
    python run_strategies.py              # Run one cycle
    python run_strategies.py --compare    # Show performance comparison
    python run_strategies.py --continuous # Run continuously
    python run_strategies.py --info       # Show strategy details
"""

import argparse
import logging
import sys
import time
from datetime import datetime

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from src.config import get_settings
from src.models import Base, Asset
from src.strategies.config import (
    StrategyType,
    print_strategy_comparison,
    list_strategies,
)
from src.strategies.multi_runner import MultiStrategyRunner

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-8s | %(name)-20s | %(message)s',
    datefmt='%H:%M:%S',
)
logger = logging.getLogger(__name__)


def setup_database(settings):
    """Initialize database connection."""
    db_path = settings.data_dir / "strategies.db"
    engine = create_engine(f"sqlite:///{db_path}")
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    return Session()


def run_once(runner: MultiStrategyRunner):
    """Run one trading cycle."""
    print(f"\n{'='*60}")
    print(f"RUNNING TRADING CYCLE - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*60}")

    results = runner.run_cycle()

    for strategy_type, actions in results.items():
        if actions:
            print(f"\n[{strategy_type.value}]:")
            for action in actions:
                print(f"  → {action}")
        else:
            print(f"\n[{strategy_type.value}]: No actions")

    runner.print_comparison()


def run_continuous(runner: MultiStrategyRunner, interval_minutes: int = 30):
    """Run continuously with specified interval."""
    print(f"\n🚀 Starting continuous multi-strategy trading")
    print(f"   Interval: {interval_minutes} minutes")
    print(f"   Press Ctrl+C to stop\n")

    cycle = 0
    while True:
        try:
            cycle += 1
            print(f"\n{'#'*60}")
            print(f"# CYCLE {cycle}")
            print(f"{'#'*60}")

            run_once(runner)

            # Wait for next cycle
            print(f"\n⏳ Next cycle in {interval_minutes} minutes...")
            time.sleep(interval_minutes * 60)

        except KeyboardInterrupt:
            print("\n\n🛑 Stopped by user")
            runner.print_comparison()
            break


def show_info():
    """Show strategy configuration details."""
    print_strategy_comparison()

    print("\nSTRATEGY HYPOTHESES:")
    print("-" * 60)

    strategies = list_strategies()
    for s in strategies:
        print(f"\n📊 {s.name.value.upper()}")
        print(f"   {s.description}")
        print(f"   Expected: ", end="")

        if s.name == StrategyType.SOCIAL_PURE:
            print("High trade frequency, captures fast alpha, higher variance")
        elif s.name == StrategyType.TECHNICAL_STRICT:
            print("Low trade frequency, higher win rate, lower variance")
        else:
            print("Medium frequency, balanced risk/reward, tests synergy")


def main():
    parser = argparse.ArgumentParser(
        description="Multi-Strategy A/B Testing for Crypto Trading",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_strategies.py              # Run one cycle
  python run_strategies.py --compare    # Show current standings
  python run_strategies.py --continuous # Run every 30 min
  python run_strategies.py --info       # Show strategy configs
        """,
    )

    parser.add_argument(
        "--compare", "-c",
        action="store_true",
        help="Show performance comparison only (no trading)",
    )
    parser.add_argument(
        "--continuous",
        action="store_true",
        help="Run continuously",
    )
    parser.add_argument(
        "--interval", "-i",
        type=int,
        default=30,
        help="Interval between cycles in minutes (default: 30)",
    )
    parser.add_argument(
        "--info",
        action="store_true",
        help="Show strategy configuration details",
    )
    parser.add_argument(
        "--balance", "-b",
        type=float,
        default=10000.0,
        help="Initial balance for each strategy (default: $10,000)",
    )
    parser.add_argument(
        "--strategy", "-s",
        type=str,
        choices=["social_pure", "technical_strict", "balanced", "all"],
        default="all",
        help="Run specific strategy or all (default: all)",
    )

    args = parser.parse_args()

    # Show info and exit
    if args.info:
        show_info()
        return

    # Initialize
    settings = get_settings()
    session = setup_database(settings)

    # Determine which strategies to run
    if args.strategy == "all":
        strategies = list(StrategyType)
    else:
        strategies = [StrategyType(args.strategy)]

    runner = MultiStrategyRunner(
        settings=settings,
        session=session,
        initial_balance=args.balance,
        strategies=strategies,
    )

    # Run modes
    if args.compare:
        runner.print_comparison()
    elif args.continuous:
        run_continuous(runner, args.interval)
    else:
        run_once(runner)


if __name__ == "__main__":
    main()
