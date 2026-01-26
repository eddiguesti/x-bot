#!/usr/bin/env python3
"""Run one trading cycle."""

import logging
from src.config import Settings
from src.models import init_db, get_session, Asset, Direction, ConsensusAction
from src.trading.paper_trader import PaperTrader
from src.consensus.engine import ConsensusEngine

logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)-8s | %(message)s')
logger = logging.getLogger(__name__)

settings = Settings()
engine = init_db(settings.database_url)
session = get_session(engine)

trader = PaperTrader(settings, session)
consensus = ConsensusEngine(settings)

print('=' * 60)
print('TRADING CYCLE')
print('=' * 60)

# Check existing positions for SL/TP
to_close = trader.check_positions()
for trade, reason in to_close:
    trader.close_position(trade, reason)
    print(f'Closed {trade.asset.value} - {reason}')

# Get consensus for all assets
results = consensus.get_all_consensus(session, lookback_hours=48)

# Find actionable signals
print('\nConsensus scan:')
for asset, result in results.items():
    if result.action != ConsensusAction.NO_TRADE:
        existing = any(t.asset == asset for t in trader.get_open_positions())
        status = '[OPEN]' if existing else '[NEW]'
        print(f'  {status} {asset.value}: {result.action.value.upper()} ({result.confidence:.1f}%) - {result.long_votes}L/{result.short_votes}S')

# Display status
trader.display_status()
session.close()
