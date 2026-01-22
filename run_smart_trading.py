#!/usr/bin/env python3
"""
Smart Trading Bot - Enhanced with Technical Analysis and Risk Management.

Features:
1. Technical analysis confirmation (RSI, MACD, Bollinger Bands)
2. Volatility-adjusted position sizing (ATR-based)
3. Kelly Criterion position sizing
4. Trailing stop losses
5. Market regime detection
6. Time-decayed signal weighting
7. Signal quality filtering
8. Drawdown protection
9. Correlation-aware exposure limits
"""

import logging
import time
from datetime import datetime

from src.config import Settings
from src.models import init_db, get_session, Asset, Direction, ConsensusAction
from src.trading.smart_trader import SmartTrader
from src.consensus.enhanced_engine import EnhancedConsensusEngine
from src.analysis.technical import TechnicalAnalyzer, MarketRegime
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

        posts = x_client.fetch_posts(usernames=[], limit=limit)

        if not posts:
            return 0

        from src.models import Creator, Signal

        new_signals = 0
        for post in posts:
            existing = session.query(Signal).filter(
                Signal.post_id == post.post_id
            ).first()
            if existing:
                continue

            extraction = extractor.extract(post.text)
            if not extraction.asset or not extraction.direction:
                continue

            creator = session.query(Creator).filter(
                Creator.username == post.username
            ).first()
            if not creator:
                creator = Creator(username=post.username, display_name=post.username)
                session.add(creator)
                session.flush()

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


def display_market_overview(technical: TechnicalAnalyzer, assets: list[Asset]):
    """Display technical overview of all assets."""
    print("\n" + "=" * 70)
    print("📊 MARKET OVERVIEW")
    print("=" * 70)

    bullish = []
    bearish = []
    neutral = []

    for asset in assets[:10]:  # Top 10 assets
        try:
            signal = technical.analyze(asset)
            if signal:
                if signal.confirmation_score > 0.2:
                    bullish.append((asset, signal))
                elif signal.confirmation_score < -0.2:
                    bearish.append((asset, signal))
                else:
                    neutral.append((asset, signal))
        except Exception:
            pass

    if bullish:
        print("\n🟢 BULLISH:")
        for asset, sig in bullish:
            print(f"   {asset.value}: score={sig.confirmation_score:.2f} | "
                  f"RSI={sig.rsi:.0f} | {sig.trend.value} | {sig.regime.value}")

    if bearish:
        print("\n🔴 BEARISH:")
        for asset, sig in bearish:
            print(f"   {asset.value}: score={sig.confirmation_score:.2f} | "
                  f"RSI={sig.rsi:.0f} | {sig.trend.value} | {sig.regime.value}")

    if neutral:
        print("\n⚪ NEUTRAL:")
        for asset, sig in neutral[:5]:
            print(f"   {asset.value}: score={sig.confirmation_score:.2f} | "
                  f"RSI={sig.rsi:.0f} | {sig.trend.value}")


def run_smart_trading(interval_minutes=5, collect_interval=3):
    """Run the smart trading loop."""
    settings = Settings()
    engine = init_db(settings.database_url)
    session = get_session(engine)

    trader = SmartTrader(settings, session)
    consensus = EnhancedConsensusEngine(settings)
    technical = TechnicalAnalyzer(settings)

    logger.info("=" * 70)
    logger.info("🤖 SMART TRADING BOT STARTING")
    logger.info("=" * 70)
    logger.info("Features:")
    logger.info("  ✓ Technical analysis confirmation")
    logger.info("  ✓ Volatility-adjusted position sizing")
    logger.info("  ✓ Kelly Criterion sizing")
    logger.info("  ✓ Trailing stop losses")
    logger.info("  ✓ Time-decayed signal weighting")
    logger.info("  ✓ Signal quality filtering")
    logger.info("  ✓ Drawdown protection")
    logger.info("=" * 70)

    cycle = 0
    while True:
        try:
            cycle += 1
            logger.info(f"\n{'='*30} CYCLE {cycle} {'='*30}")
            logger.info(f"Time: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC")

            # Collect signals every N cycles
            if cycle % collect_interval == 1:
                logger.info("📡 Collecting signals...")
                new_signals = collect_signals(settings, session)
                if new_signals > 0:
                    logger.info(f"   Collected {new_signals} new signals")

            # Update trailing stops
            trader.update_trailing_stops()

            # Get consensus for all assets
            results = consensus.get_all_consensus(session, lookback_hours=48)

            # Check existing positions
            to_close = trader.check_positions(results)
            for trade, reason in to_close:
                trader.close_position(trade, reason)

            # Find actionable signals
            actionable = []
            for asset, result in results.items():
                if consensus.should_trade(result):
                    # Evaluate trade with technical confirmation
                    evaluation = trader.evaluate_trade(asset, result)
                    if evaluation['should_trade']:
                        actionable.append((asset, result, evaluation))

            # Display consensus summary
            has_signal = [a for a, r in results.items() if r.action != ConsensusAction.NO_TRADE]
            if has_signal:
                print("\n📊 CONSENSUS SIGNALS:")
                for asset, result in results.items():
                    if result.action != ConsensusAction.NO_TRADE:
                        tech = technical.analyze(asset)
                        tech_str = f"tech={tech.confirmation_score:.2f}" if tech else "tech=N/A"
                        print(f"   {result.action.value.upper()} {asset.value}: "
                              f"conf={result.confidence:.1%} | "
                              f"{result.long_votes}L/{result.short_votes}S | "
                              f"{tech_str}")

            # Execute trades
            if actionable:
                logger.info(f"\n🎯 ACTIONABLE TRADES: {len(actionable)}")
                for asset, result, evaluation in actionable:
                    direction = evaluation['direction']

                    logger.info(f"\n   Opening {direction.value.upper()} {asset.value}:")
                    for reason in evaluation['reasons']:
                        logger.info(f"      • {reason}")
                    for warning in evaluation.get('warnings', []):
                        logger.warning(f"      ⚠ {warning}")

                    trade = trader.open_position(
                        asset=asset,
                        direction=direction,
                        consensus_score=result.weighted_score,
                        consensus_confidence=result.confidence,
                    )
            else:
                logger.info("No actionable trades this cycle")

            # Display status
            trader.display_status()

            # Show market overview every 5 cycles
            if cycle % 5 == 0:
                display_market_overview(technical, list(Asset))

            # Wait for next cycle
            logger.info(f"\n⏳ Next cycle in {interval_minutes} minutes...")
            time.sleep(interval_minutes * 60)

        except KeyboardInterrupt:
            logger.info("\n🛑 Shutting down...")
            break
        except Exception as e:
            logger.error(f"Error in trading loop: {e}")
            import traceback
            traceback.print_exc()
            time.sleep(60)

    session.close()
    logger.info("Trading bot stopped")


def run_once():
    """Run a single trading cycle (for testing)."""
    settings = Settings()
    engine = init_db(settings.database_url)
    session = get_session(engine)

    trader = SmartTrader(settings, session)
    consensus = EnhancedConsensusEngine(settings)
    technical = TechnicalAnalyzer(settings)

    print("=" * 70)
    print("🤖 SMART TRADING - SINGLE CYCLE")
    print("=" * 70)

    # Update trailing stops
    trader.update_trailing_stops()

    # Get consensus
    results = consensus.get_all_consensus(session, lookback_hours=48)

    # Check positions
    to_close = trader.check_positions(results)
    for trade, reason in to_close:
        trader.close_position(trade, reason)

    # Show consensus
    print("\n📊 CONSENSUS SIGNALS:")
    for asset, result in results.items():
        if result.action != ConsensusAction.NO_TRADE:
            tech = technical.analyze(asset)
            tech_str = f"tech={tech.confirmation_score:.2f}" if tech else "tech=N/A"
            print(f"   {result.action.value.upper()} {asset.value}: "
                  f"conf={result.confidence:.1%} | "
                  f"{result.long_votes}L/{result.short_votes}S | "
                  f"{tech_str}")

    # Find actionable signals
    for asset, result in results.items():
        if consensus.should_trade(result):
            evaluation = trader.evaluate_trade(asset, result)
            if evaluation['should_trade']:
                print(f"\n🎯 TRADE OPPORTUNITY: {asset.value}")
                for reason in evaluation['reasons']:
                    print(f"   • {reason}")
                for warning in evaluation.get('warnings', []):
                    print(f"   ⚠ {warning}")

    # Display status
    trader.display_status()

    # Market overview
    display_market_overview(technical, list(Asset))

    session.close()


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "once":
        run_once()
    else:
        run_smart_trading()
