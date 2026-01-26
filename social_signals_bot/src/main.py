"""Main entry point for the Crypto Consensus system."""

import argparse
import json
import logging
import sys
from datetime import datetime, timedelta
from pathlib import Path

from .config import get_settings, Settings
from .models import init_db, get_session, Creator, Signal, Asset, Direction, SignalOutcome
from .data_ingestion import XClient, PriceClient
from .signal_extraction import SignalExtractor
from .scoring import SignalEvaluator
from .consensus import ConsensusEngine
from .output import ReportGenerator, DataExporter


def setup_logging(level: str = "INFO"):
    """Configure logging."""
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def load_creators(config_path: Path) -> list[dict]:
    """Load creator list from config file."""
    if not config_path.exists():
        logging.warning(f"Creator config not found: {config_path}")
        return []

    with open(config_path) as f:
        data = json.load(f)
        return data.get("creators", [])


def ensure_creators(session, creator_configs: list[dict]):
    """Ensure all configured creators exist in database."""
    for config in creator_configs:
        username = config["username"].lstrip("@")
        existing = session.query(Creator).filter(Creator.username == username).first()

        if not existing:
            creator = Creator(
                username=username,
                display_name=config.get("display_name"),
            )
            session.add(creator)
            logging.info(f"Added creator: {username}")

    session.commit()


def cmd_collect(settings: Settings, args):
    """Collect new posts from X and extract signals."""
    engine = init_db(settings.database_url)
    session = get_session(engine)

    # Load creators
    creators_path = settings.base_dir / "config" / "creators.json"
    creator_configs = load_creators(creators_path)
    ensure_creators(session, creator_configs)

    # Get active creators
    creators = session.query(Creator).filter(Creator.is_active == True).all()
    usernames = [c.username for c in creators]

    discover_mode = getattr(args, 'discover', False)

    # Fetch posts
    x_client = XClient(settings)
    days = args.days if hasattr(args, 'days') else 1
    end_date = datetime.utcnow()
    start_date = end_date - timedelta(days=days)

    # Get limit (cap at 1000 per API)
    limit = min(args.limit if hasattr(args, 'limit') else 100, 1000)

    if discover_mode:
        # In discover mode, fetch trading signals from anyone
        logging.info(f"Discovery mode: searching for up to {limit} trading signals...")
        posts = x_client.fetch_trading_signals(
            limit=limit,
            hours_back=days * 24,
        )
    else:
        # Normal mode: filter to tracked creators
        if not usernames:
            logging.warning("No creators configured. Add creators to config/creators.json or use --discover")
            return
        posts = x_client.fetch_posts(
            usernames=usernames,
            start_date=start_date,
            end_date=end_date,
        )

    logging.info(f"Fetched {len(posts)} posts")

    # Extract signals
    extractor = SignalExtractor(use_nlp=not args.no_nlp if hasattr(args, 'no_nlp') else True)
    price_client = PriceClient(settings)

    signals_created = 0
    creators_added = 0

    for post in posts:
        # Check if signal already exists
        existing = session.query(Signal).filter(Signal.post_id == post.post_id).first()
        if existing:
            continue

        # Extract signal
        extraction = extractor.extract(post.text)

        if not extractor.is_actionable(extraction):
            continue

        # Get or create creator
        creator = session.query(Creator).filter(Creator.username == post.username).first()

        if not creator:
            if discover_mode:
                # Auto-add new creator in discover mode
                creator = Creator(
                    username=post.username,
                    display_name=post.raw_data.get('user', {}).get('display_name', post.username),
                )
                session.add(creator)
                session.flush()  # Get the ID
                creators_added += 1
                logging.info(f"Discovered new creator: @{post.username}")
            else:
                continue

        # Get current price
        price = price_client.get_current_price(extraction.asset)

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
            price_at_signal=price,
            raw_data=json.dumps(post.raw_data, default=str),
        )
        session.add(signal)
        signals_created += 1

    session.commit()
    logging.info(f"Created {signals_created} new signals")
    if creators_added > 0:
        logging.info(f"Discovered {creators_added} new creators")


def cmd_evaluate(settings: Settings, args):
    """Evaluate pending signals."""
    engine = init_db(settings.database_url)
    session = get_session(engine)

    price_client = PriceClient(settings)
    evaluator = SignalEvaluator(settings, price_client)

    evaluated = evaluator.evaluate_pending_signals(session)
    stats = evaluator.get_evaluation_stats(session)

    logging.info(f"Evaluated {evaluated} signals")
    logging.info(f"Stats: {stats}")


def cmd_consensus(settings: Settings, args):
    """Calculate current consensus."""
    engine = init_db(settings.database_url)
    session = get_session(engine)

    consensus_engine = ConsensusEngine(settings)
    results = consensus_engine.get_all_consensus(session)

    print("\n" + "=" * 50)
    print("CONSENSUS SIGNALS")
    print("=" * 50)

    for asset, result in results.items():
        emoji = "🟢" if result.action.value == "long" else ("🔴" if result.action.value == "short" else "⚪")
        print(f"\n{emoji} {asset.value}:")
        print(f"   Action: {result.action.value.upper()}")
        print(f"   Confidence: {result.confidence:.1%}")
        print(f"   Score: {result.weighted_score:.3f}")
        print(f"   Votes: {result.long_votes} long / {result.short_votes} short")
        if result.top_contributors:
            print(f"   Top: {', '.join(result.top_contributors[:3])}")


def cmd_rankings(settings: Settings, args):
    """Show creator rankings."""
    engine = init_db(settings.database_url)
    session = get_session(engine)

    report_gen = ReportGenerator(settings)
    rankings = report_gen.generate_creator_rankings(session)

    if rankings.empty:
        print("No creators with enough predictions yet.")
        return

    print("\n" + "=" * 80)
    print("CREATOR RANKINGS")
    print("=" * 80)
    print(rankings.to_string(index=False))


def cmd_export(settings: Settings, args):
    """Export all data to ZIP."""
    engine = init_db(settings.database_url)
    session = get_session(engine)

    exporter = DataExporter(settings)
    output_path = exporter.export_all(session)

    print(f"\nExport complete: {output_path}")


def cmd_backtest(settings: Settings, args):
    """Backtest all signals and update creator ratings."""
    from .models import SignalOutcome
    from .scoring.glicko2 import Glicko2Calculator, Glicko2Rating

    engine = init_db(settings.database_url)
    session = get_session(engine)

    price_client = PriceClient(settings)
    evaluator = SignalEvaluator(settings, price_client)

    force = getattr(args, 'force', False)

    print("\n" + "=" * 50)
    print("BACKTESTING SIGNALS")
    print("=" * 50)

    # Step 1: Evaluate all pending signals
    if force:
        # Reset all evaluations to PENDING state in batches
        signal_ids = [id[0] for id in session.query(Signal.id).all()]
        total = len(signal_ids)
        batch_size = 100

        for i in range(0, total, batch_size):
            batch_ids = signal_ids[i:i + batch_size]
            session.query(Signal).filter(Signal.id.in_(batch_ids)).update({
                Signal.outcome: SignalOutcome.PENDING,
                Signal.evaluated_at: None,
                Signal.price_at_evaluation: None,
                Signal.price_change_percent: None
            }, synchronize_session=False)
            session.commit()

        print(f"\n🔄 Reset {total} signals for re-evaluation")

    evaluated = evaluator.evaluate_pending_signals(session)
    print(f"\n✅ Evaluated {evaluated} signals against price data")

    # Step 2: Recalculate all creator ratings from scratch
    calc = Glicko2Calculator()
    creators = session.query(Creator).filter(Creator.total_predictions > 0).all()

    print(f"\n📊 Updating ratings for {len(creators)} creators...")

    for creator in creators:
        # Get all evaluated signals for this creator
        signals = session.query(Signal).filter(
            Signal.creator_id == creator.id,
            Signal.outcome != None
        ).order_by(Signal.posted_at).all()

        if not signals:
            continue

        # Build outcomes list
        outcomes = [
            (s.outcome == SignalOutcome.CORRECT, None)
            for s in signals
        ]

        # Update stats
        creator.total_predictions = len(signals)
        creator.correct_predictions = sum(1 for o, _ in outcomes if o)

        # Recalculate Glicko-2 rating from initial
        rating = Glicko2Rating(
            rating=1500.0,  # Start fresh
            rd=350.0,
            volatility=0.06,
        )

        # Process in batches of 10 (rating periods)
        batch_size = 10
        for i in range(0, len(outcomes), batch_size):
            batch = outcomes[i:i + batch_size]
            if batch:
                rating = calc.update_rating(rating, batch)

        creator.rating = rating.rating
        creator.rating_deviation = rating.rd
        creator.volatility = rating.volatility

    session.commit()

    # Step 3: Show results
    print("\n🏆 UPDATED RANKINGS (Top 15):")
    print("-" * 70)

    ranked = session.query(Creator).filter(
        Creator.total_predictions >= 5  # Lower threshold for backtesting
    ).order_by(Creator.rating.desc()).limit(15).all()

    for i, c in enumerate(ranked, 1):
        acc = (c.correct_predictions / c.total_predictions * 100) if c.total_predictions > 0 else 0
        rating = Glicko2Rating(c.rating, c.rating_deviation, c.volatility)
        print(f"  {i:2}. @{c.username[:20]:<20} | Rating: {c.rating:7.1f} | "
              f"Acc: {acc:5.1f}% | Signals: {c.total_predictions:3} | Weight: {rating.weight:.3f}")

    # Summary
    total_signals = session.query(Signal).filter(Signal.outcome != None).count()
    correct = session.query(Signal).filter(Signal.outcome == SignalOutcome.CORRECT).count()
    overall_acc = (correct / total_signals * 100) if total_signals > 0 else 0

    print(f"\n📈 OVERALL: {correct}/{total_signals} correct ({overall_acc:.1f}% accuracy)")


def cmd_stats(settings: Settings, args):
    """Show system statistics."""
    engine = init_db(settings.database_url)
    session = get_session(engine)

    # Creator stats
    total_creators = session.query(Creator).count()
    active_creators = session.query(Creator).filter(Creator.is_active == True).count()
    creators_with_signals = session.query(Creator).filter(Creator.total_predictions > 0).count()
    ranked_creators = session.query(Creator).filter(
        Creator.total_predictions >= 10
    ).count()

    # Signal stats
    from .models import SignalOutcome
    total_signals = session.query(Signal).count()
    evaluated_signals = session.query(Signal).filter(Signal.outcome != None).count()
    pending_signals = session.query(Signal).filter(Signal.outcome == None).count()
    correct_signals = session.query(Signal).filter(Signal.outcome == SignalOutcome.CORRECT).count()

    # Asset breakdown
    btc_signals = session.query(Signal).filter(Signal.asset == Asset.BTC).count()
    eth_signals = session.query(Signal).filter(Signal.asset == Asset.ETH).count()

    print("\n" + "=" * 50)
    print("SYSTEM STATISTICS")
    print("=" * 50)

    print("\n📊 CREATORS:")
    print(f"   Total tracked:     {total_creators}")
    print(f"   Active:            {active_creators}")
    print(f"   With signals:      {creators_with_signals}")
    print(f"   Ranked (10+ pred): {ranked_creators}")

    print("\n📈 SIGNALS:")
    print(f"   Total:             {total_signals}")
    print(f"   Evaluated:         {evaluated_signals}")
    print(f"   Pending:           {pending_signals}")
    if evaluated_signals > 0:
        accuracy = correct_signals / evaluated_signals * 100
        print(f"   Correct:           {correct_signals} ({accuracy:.1f}%)")

    print("\n💰 BY ASSET:")
    print(f"   BTC signals:       {btc_signals}")
    print(f"   ETH signals:       {eth_signals}")

    # Top 5 by signal count
    top_by_signals = session.query(Creator).filter(
        Creator.total_predictions > 0
    ).order_by(Creator.total_predictions.desc()).limit(5).all()

    if top_by_signals:
        print("\n🏆 MOST ACTIVE CREATORS:")
        for c in top_by_signals:
            acc = (c.correct_predictions / c.total_predictions * 100) if c.total_predictions > 0 else 0
            print(f"   @{c.username}: {c.total_predictions} signals ({acc:.0f}% accuracy)")


def cmd_mock(settings: Settings, args):
    """Generate mock data for testing."""
    from .data_ingestion.mock_data import generate_mock_posts, generate_historical_signals

    engine = init_db(settings.database_url)
    session = get_session(engine)

    # Load creators
    creators_path = settings.base_dir / "config" / "creators.json"
    creator_configs = load_creators(creators_path)
    ensure_creators(session, creator_configs)

    # Generate historical signals with outcomes
    num_signals = args.signals if hasattr(args, 'signals') else 100
    days = args.days if hasattr(args, 'days') else 30

    print(f"Generating {num_signals} historical signals over {days} days...")
    created = generate_historical_signals(session, num_signals=num_signals, days_back=days)

    print(f"Created {created} mock signals with outcomes")

    # Show summary
    from .output import ReportGenerator
    report_gen = ReportGenerator(settings)
    summary = report_gen.generate_performance_summary(session, days=days)

    print(f"\nSummary:")
    print(f"  Creators: {summary['creators']['total_tracked']} tracked, {summary['creators']['ranked']} ranked")
    print(f"  Signals: {summary['signals']['total']} total, {summary['signals']['correct']} correct")
    print(f"  Accuracy: {summary['signals']['accuracy']:.1%}")

    if summary['top_performers']:
        print(f"\nTop performers:")
        for p in summary['top_performers']:
            print(f"  - {p['username']}: {p['rating']:.0f} rating, {p['accuracy']:.0f}% accuracy")


def cmd_trade(settings: Settings, args):
    """Live paper trading with consensus signals."""
    import time
    from .trading.paper_trader import PaperTrader
    from .models import Trade, TradeStatus, Portfolio, ConsensusAction
    from .notifications import TelegramNotifier

    engine = init_db(settings.database_url)
    session = get_session(engine)

    # Initialize trader
    trader = PaperTrader(settings, session)

    # Initialize notifications
    notifier = TelegramNotifier(settings)
    if notifier.enabled:
        notifier.send("🚀 <b>Trading Bot Started</b>\n\nMonitoring for consensus signals...")

    # Initialize other components
    x_client = XClient(settings)
    extractor = SignalExtractor(use_nlp=False, min_confidence=0.5)  # Lower threshold
    price_client = PriceClient(settings)
    consensus_engine = ConsensusEngine(settings)

    interval = args.interval if hasattr(args, 'interval') else 300  # 5 min default
    min_confidence = args.confidence if hasattr(args, 'confidence') else 0.4

    print("\n" + "=" * 60)
    print("🚀 LIVE PAPER TRADING")
    print("=" * 60)
    print(f"\nInterval: {interval}s | Min confidence: {min_confidence:.0%}")
    print("Press Ctrl+C to stop\n")

    trader.display_status()

    cycle = 0
    while True:
        try:
            cycle += 1
            print(f"\n--- Cycle {cycle} | {datetime.utcnow().strftime('%H:%M:%S')} ---")

            # 1. Collect new signals
            print("📡 Fetching signals...")
            posts = x_client.fetch_trading_signals(limit=50, hours_back=1)

            signals_added = 0
            for post in posts:
                existing = session.query(Signal).filter(Signal.post_id == post.post_id).first()
                if existing:
                    continue

                extraction = extractor.extract(post.text)
                if not extractor.is_actionable(extraction):
                    continue

                creator = session.query(Creator).filter(Creator.username == post.username).first()
                if not creator:
                    creator = Creator(username=post.username)
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
                signals_added += 1

            session.commit()
            if signals_added > 0:
                print(f"   Added {signals_added} new signals")

            # 2. Get consensus
            print("🎯 Calculating consensus...")
            consensus_results = consensus_engine.get_all_consensus(session)

            for asset, result in consensus_results.items():
                btc_price = price_client.get_current_price(asset)
                print(f"   {asset.value}: {result.action.value.upper()} "
                      f"({result.confidence:.0%} confidence) | ${btc_price:,.0f}")

                # 3. Check for trade signals
                if result.confidence >= min_confidence and result.action != ConsensusAction.NO_TRADE:
                    # Check if we already have a position
                    existing_trade = session.query(Trade).filter(
                        Trade.asset == asset,
                        Trade.status == TradeStatus.OPEN,
                    ).first()

                    if existing_trade:
                        # Check if signal direction matches our position
                        if (result.action == ConsensusAction.LONG and
                            existing_trade.direction.value != "long"):
                            # Signal reversed - close position
                            print(f"   ⚡ Signal reversed! Closing position...")
                            trader.close_position(existing_trade, "signal_reversal")
                        elif (result.action == ConsensusAction.SHORT and
                              existing_trade.direction.value != "short"):
                            print(f"   ⚡ Signal reversed! Closing position...")
                            trader.close_position(existing_trade, "signal_reversal")
                    else:
                        # No position - open one
                        from .models import Direction
                        direction = Direction.LONG if result.action == ConsensusAction.LONG else Direction.SHORT
                        print(f"   🎯 Opening {direction.value.upper()} position...")
                        trader.open_position(
                            asset=asset,
                            direction=direction,
                            consensus_score=result.weighted_score,
                            consensus_confidence=result.confidence,
                        )

            # 4. Check stop loss / take profit
            to_close = trader.check_positions()
            for trade, reason in to_close:
                print(f"   ⚠️ {reason.upper()} triggered for {trade.asset.value}")
                trader.close_position(trade, reason)

            # 5. Display status periodically
            if cycle % 5 == 0:
                trader.display_status()

            # Wait for next cycle
            print(f"\n💤 Sleeping {interval}s...")
            time.sleep(interval)

        except KeyboardInterrupt:
            print("\n\n⏹️ Stopping...")
            trader.display_status()
            break
        except Exception as e:
            logging.error(f"Error in trading loop: {e}")
            time.sleep(30)


def cmd_testnet(settings: Settings, args):
    """Connect to Binance Testnet for demo trading."""
    from .trading.binance_testnet import BinanceTestnetTrader

    if not settings.binance_testnet_api_key or not settings.binance_testnet_secret:
        print("\n" + "=" * 60)
        print("🔬 BINANCE TESTNET SETUP")
        print("=" * 60)
        print("\nTo use Binance Testnet (free demo trading):")
        print("\n1. Go to: https://testnet.binance.vision/")
        print("2. Click 'Log In with GitHub'")
        print("3. Click 'Generate HMAC_SHA256 Key'")
        print("4. Copy the API Key and Secret Key")
        print("5. Add to your .env file:")
        print("   BINANCE_TESTNET_API_KEY=your_api_key")
        print("   BINANCE_TESTNET_SECRET=your_secret_key")
        print("\n💡 This gives you 10,000 USDT to practice with!")
        return

    try:
        trader = BinanceTestnetTrader(settings)
        trader.display_status()

        # Show recent trades if any
        trades = trader.get_recent_trades(limit=5)
        if trades:
            print("\n📜 RECENT TRADES:")
            for t in trades[:5]:
                print(f"   {t['symbol']} {t['side']} {t['amount']:.6f} @ ${t['price']:.2f}")

    except Exception as e:
        print(f"\n❌ Error connecting to testnet: {e}")
        print("\nMake sure your API keys are correct and try again.")


def cmd_portfolio(settings: Settings, args):
    """Show paper trading portfolio status."""
    from .trading.paper_trader import PaperTrader
    from .models import Trade, TradeStatus

    engine = init_db(settings.database_url)
    session = get_session(engine)

    trader = PaperTrader(settings, session)
    trader.display_status()

    # Show trade history
    recent_trades = session.query(Trade).filter(
        Trade.status == TradeStatus.CLOSED
    ).order_by(Trade.exit_time.desc()).limit(10).all()

    if recent_trades:
        print("\n📜 RECENT CLOSED TRADES:")
        for t in recent_trades:
            emoji = "✅" if t.pnl_percent > 0 else "❌"
            print(
                f"   {emoji} {t.direction.value.upper()} {t.asset.value}: "
                f"${t.entry_price:,.2f} -> ${t.exit_price:,.2f} | "
                f"PnL: {t.pnl_percent:+.2f}% (${t.pnl_usd:+.2f})"
            )


def cmd_history(settings: Settings, args):
    """Fetch historical data using Apify for extended backtesting."""
    from .data_ingestion.apify_client import ApifyXClient

    if not settings.apify_api_token:
        print("\n❌ APIFY_API_TOKEN required for historical data fetching")
        print("\nTo use this feature:")
        print("  1. Sign up at https://console.apify.com")
        print("  2. Get your API token from Account > Integrations")
        print("  3. Add APIFY_API_TOKEN=your_token to .env")
        print("\nPricing: ~$0.25-0.40 per 1,000 tweets")
        return

    engine = init_db(settings.database_url)
    session = get_session(engine)

    # Parse date range
    years_back = args.years if hasattr(args, 'years') else 2
    end_date = datetime.utcnow()
    start_date = end_date - timedelta(days=years_back * 365)

    max_tweets = args.limit if hasattr(args, 'limit') else 5000

    print("\n" + "=" * 50)
    print("FETCHING HISTORICAL DATA")
    print("=" * 50)
    print(f"\nDate range: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
    print(f"Max tweets: {max_tweets}")

    apify_client = ApifyXClient(settings.apify_api_token)
    extractor = SignalExtractor(use_nlp=False)  # Use rules for speed
    price_client = PriceClient(settings)

    if args.from_traders:
        # Fetch from known/discovered traders
        creators = session.query(Creator).filter(Creator.is_active == True).all()
        usernames = [c.username for c in creators]

        if not usernames:
            print("\n❌ No traders in database. Run 'collect --discover' first to find traders.")
            return

        print(f"\nFetching from {len(usernames)} known traders...")
        posts = apify_client.fetch_from_known_traders(
            trader_usernames=usernames[:50],  # Limit to avoid huge costs
            start_date=start_date,
            end_date=end_date,
            max_per_user=max_tweets // len(usernames[:50]),
        )
    else:
        # Fetch crypto trading signals from anyone
        print("\nSearching for crypto trading signals...")
        posts = apify_client.fetch_historical_trading_signals(
            start_date=start_date,
            end_date=end_date,
            max_tweets=max_tweets,
        )

    print(f"\n📥 Fetched {len(posts)} posts")

    # Process posts into signals
    signals_created = 0
    creators_added = 0

    for post in posts:
        # Check if signal already exists
        existing = session.query(Signal).filter(Signal.post_id == post.post_id).first()
        if existing:
            continue

        # Extract signal
        extraction = extractor.extract(post.text)
        if not extractor.is_actionable(extraction):
            continue

        # Get or create creator
        creator = session.query(Creator).filter(Creator.username == post.username).first()
        if not creator:
            creator = Creator(
                username=post.username,
                display_name=post.raw_data.get('author', {}).get('displayname', post.username),
            )
            session.add(creator)
            session.flush()
            creators_added += 1

        # Create signal (historical - will need backtest to evaluate)
        signal = Signal(
            creator_id=creator.id,
            post_id=post.post_id,
            post_text=post.text,
            post_url=post.url,
            posted_at=post.posted_at,
            asset=extraction.asset,
            direction=extraction.direction,
            confidence=extraction.confidence,
            raw_data=json.dumps(post.raw_data, default=str),
        )
        session.add(signal)
        signals_created += 1

    session.commit()

    print(f"\n✅ Created {signals_created} new signals")
    if creators_added > 0:
        print(f"📊 Discovered {creators_added} new creators")

    print("\n💡 Next steps:")
    print("   Run 'python -m src.main backtest --force' to evaluate signals")


def cmd_init(settings: Settings, args):
    """Initialize database and config."""
    # Create directories
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    settings.raw_data_dir.mkdir(parents=True, exist_ok=True)
    settings.processed_data_dir.mkdir(parents=True, exist_ok=True)
    settings.reports_dir.mkdir(parents=True, exist_ok=True)
    (settings.base_dir / "config").mkdir(parents=True, exist_ok=True)

    # Initialize database
    engine = init_db(settings.database_url)
    logging.info(f"Database initialized: {settings.database_url}")

    # Create default creators config if not exists
    creators_path = settings.base_dir / "config" / "creators.json"
    if not creators_path.exists():
        default_creators = {
            "creators": [
                {"username": "100trillionUSD", "display_name": "PlanB"},
                {"username": "CryptoCapo_", "display_name": "Crypto Capo"},
                {"username": "CryptoCred", "display_name": "Cred"},
                {"username": "inversebrah", "display_name": "inversebrah"},
                {"username": "CryptoKaleo", "display_name": "Kaleo"},
            ],
            "_note": "Add crypto traders to track. Username is X handle without @."
        }
        with open(creators_path, 'w') as f:
            json.dump(default_creators, f, indent=2)
        logging.info(f"Created default creators config: {creators_path}")

    # Add .gitkeep files
    for dir_path in [settings.raw_data_dir, settings.processed_data_dir]:
        gitkeep = dir_path / ".gitkeep"
        if not gitkeep.exists():
            gitkeep.touch()

    print("\nInitialization complete!")
    print(f"  - Database: {settings.database_url}")
    print(f"  - Creators config: {creators_path}")
    print("\nNext steps:")
    print("  1. Edit config/creators.json to add traders to track")
    print("  2. Run: python -m src.main collect")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Crypto X Creator Consensus & Ranking System"
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level"
    )

    subparsers = parser.add_subparsers(dest="command", help="Commands")

    # init command
    init_parser = subparsers.add_parser("init", help="Initialize database and config")

    # collect command
    collect_parser = subparsers.add_parser("collect", help="Collect posts and extract signals")
    collect_parser.add_argument("--days", type=int, default=1, help="Days to look back")
    collect_parser.add_argument("--limit", type=int, default=100, help="Max posts to fetch (default 100, max 1000)")
    collect_parser.add_argument("--no-nlp", action="store_true", help="Disable NLP (use rules only)")
    collect_parser.add_argument("--discover", action="store_true", help="Discover new creators from trading signals")

    # evaluate command
    eval_parser = subparsers.add_parser("evaluate", help="Evaluate pending signals")

    # consensus command
    consensus_parser = subparsers.add_parser("consensus", help="Show current consensus")

    # rankings command
    rankings_parser = subparsers.add_parser("rankings", help="Show creator rankings")

    # export command
    export_parser = subparsers.add_parser("export", help="Export all data to ZIP")

    # stats command
    stats_parser = subparsers.add_parser("stats", help="Show system statistics")

    # backtest command
    backtest_parser = subparsers.add_parser("backtest", help="Backtest all pending signals and update ratings")
    backtest_parser.add_argument("--force", action="store_true", help="Re-evaluate already evaluated signals")

    # mock command
    mock_parser = subparsers.add_parser("mock", help="Generate mock data for testing")
    mock_parser.add_argument("--signals", type=int, default=100, help="Number of signals to generate")
    mock_parser.add_argument("--days", type=int, default=30, help="Days of history to generate")

    # history command (Apify-based historical data)
    history_parser = subparsers.add_parser("history", help="Fetch historical data via Apify (up to 2+ years)")
    history_parser.add_argument("--years", type=float, default=2, help="Years of history to fetch (default: 2)")
    history_parser.add_argument("--limit", type=int, default=5000, help="Max tweets to fetch (default: 5000)")
    history_parser.add_argument("--from-traders", action="store_true", help="Fetch from known traders only")

    # trade command (live paper trading)
    trade_parser = subparsers.add_parser("trade", help="Start live paper trading")
    trade_parser.add_argument("--interval", type=int, default=300, help="Seconds between cycles (default: 300)")
    trade_parser.add_argument("--confidence", type=float, default=0.4, help="Min consensus confidence to trade (default: 0.4)")

    # testnet command (Binance testnet status)
    testnet_parser = subparsers.add_parser("testnet", help="Connect to Binance Testnet")

    # portfolio command (show portfolio status)
    portfolio_parser = subparsers.add_parser("portfolio", help="Show paper trading portfolio")

    args = parser.parse_args()

    setup_logging(args.log_level)
    settings = get_settings()

    commands = {
        "init": cmd_init,
        "collect": cmd_collect,
        "evaluate": cmd_evaluate,
        "backtest": cmd_backtest,
        "consensus": cmd_consensus,
        "rankings": cmd_rankings,
        "stats": cmd_stats,
        "export": cmd_export,
        "mock": cmd_mock,
        "history": cmd_history,
        "trade": cmd_trade,
        "testnet": cmd_testnet,
        "portfolio": cmd_portfolio,
    }

    if args.command is None:
        parser.print_help()
        sys.exit(1)

    if args.command in commands:
        commands[args.command](settings, args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
