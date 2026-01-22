# Crypto X Creator Consensus & Ranking System

## Purpose

This project is a **meta-analysis system** that observes crypto trading content creators on X (Twitter), evaluates the historical accuracy of their directional market calls, and builds a **weighted consensus decision matrix** to identify when a trade signal is statistically meaningful.

The system does **not** attempt to predict markets directly.
Instead, it answers a higher-level question:

> *When many traders express an opinion, and some have proven more reliable than others, does the weighted consensus contain signal?*

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                     DATA LAYER                              │
├─────────────────────────────────────────────────────────────┤
│  Macrocosmos X Scraper    │  Price Feed (CCXT/Binance)     │
│  (SN13 Gravity API)       │  (BTC, ETH)                     │
└──────────┬────────────────┴─────────────┬───────────────────┘
           │                              │
           ▼                              ▼
┌─────────────────────────────────────────────────────────────┐
│                   PROCESSING LAYER                          │
├─────────────────────────────────────────────────────────────┤
│  Signal Extractor         │  Price Oracle                   │
│  (FinBERT + rules)        │  (candles, % change calc)       │
└──────────┬────────────────┴──────────────┬──────────────────┘
           │                               │
           ▼                               ▼
┌─────────────────────────────────────────────────────────────┐
│                   SCORING LAYER                             │
├─────────────────────────────────────────────────────────────┤
│  Prediction Matcher       │  Glicko-2 Rating Engine         │
│  (signal + outcome)       │  (creator weights, uncertainty) │
└──────────┬────────────────┴──────────────┬──────────────────┘
           │                               │
           ▼                               ▼
┌─────────────────────────────────────────────────────────────┐
│                  CONSENSUS LAYER                            │
├─────────────────────────────────────────────────────────────┤
│  Dynamic Weighted Majority Algorithm                        │
│  → Outputs: LONG / SHORT / NO_TRADE + confidence           │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│                   OUTPUT LAYER                              │
├─────────────────────────────────────────────────────────────┤
│  Reports  │  Rankings  │  Signals  │  ZIP Export           │
└─────────────────────────────────────────────────────────────┘
```

---

## Technology Stack

| Component | Technology | Purpose |
|-----------|-----------|---------|
| Language | Python 3.11+ | ML ecosystem, async support |
| X Data | Macrocosmos SN13 API | Twitter/X post scraping |
| Price Data | CCXT + Binance | Real-time and historical prices |
| NLP | Transformers + FinBERT | Signal extraction from text |
| Database | SQLAlchemy + SQLite | Append-only data storage |
| Scheduling | APScheduler | Periodic data collection |
| API | FastAPI (optional) | Dashboard/external access |

---

## Core Philosophy

### 1. Consensus must be weighted
Raw vote counts are useless.

- 50% long / 50% short → **NO TRADE**
- 60% long / 40% short → weak signal
- Strong imbalance with proven traders → actionable

Creators earn **influence (weight)** over time based on accuracy.
Poor performers fade out naturally via the Glicko-2 rating system.

### 2. Performance is judged objectively
Creators are scored **only** on market outcomes:

- Did price move in the predicted direction after a fixed horizon?

No scoring based on:
- popularity
- engagement
- narratives
- reputation

### 3. Learning must be auditable
- Raw data is append-only
- Scores are derived
- Everything can be replayed

Every decision must be explainable.

---

## Data Ingestion

### X/Twitter Data (Macrocosmos SN13)

```python
import macrocosmos as mc

client = mc.Sn13Client(api_key="<key>", app_name="crypto_consensus")

response = client.sn13.OnDemandData(
    source='X',
    usernames=["@trader1", "@trader2"],  # Up to 5 per request
    keywords=["BTC", "ETH", "bitcoin", "ethereum"],
    start_date='2025-01-01',
    end_date='2025-01-21',
    limit=1000,
    keyword_mode='any'
)
```

### Price Data (CCXT)

```python
import ccxt

exchange = ccxt.binance()
ohlcv = exchange.fetch_ohlcv('BTC/USDT', '1h', limit=100)
```

---

## Signal Extraction (NLP)

### What Counts as a Signal

A post produces a signal only if it:
- Mentions a specific asset (BTC, ETH)
- Expresses clear direction (long / short)

### Extraction Method

**Primary: FinBERT-based classification**
- Fine-tuned transformer for financial sentiment
- Outputs: bullish / bearish / neutral + confidence score

**Fallback: Rule-based keywords**
- LONG: buy, long, bullish, moon, pump, accumulate
- SHORT: sell, short, bearish, dump, crash, exit

**Filtering:**
- Neutral/vague posts → ignored
- Confidence < 0.7 → ignored
- Conditional statements → requires extra validation

---

## Signal Evaluation

Each signal is evaluated after a fixed horizon (default: 24h).

| Direction | Correct If | Threshold |
|-----------|-----------|-----------|
| LONG | price_change ≥ +0.5% | Configurable |
| SHORT | price_change ≤ -0.5% | Configurable |

Each signal becomes:
- `correct`
- `incorrect`

**Multi-horizon tracking (planned):**
- 4h (scalp)
- 24h (day trade)
- 7d (swing)

---

## Creator Ranking: Glicko-2 System

Unlike simple accuracy percentages, Glicko-2 provides:

1. **Rating (μ)** - Skill estimate (starts at 1500)
2. **Rating Deviation (RD)** - Uncertainty (starts high, decreases with predictions)
3. **Volatility (σ)** - How consistent the creator is

### Why Glicko-2?

- New creators start with high uncertainty → low weight
- Inactive creators regain uncertainty → must re-prove
- Recent performance weighted more heavily
- Self-correcting: wrong predictions lower rating

### Weight Calculation

```python
def calculate_weight(rating, rd):
    base_weight = (rating - 1000) / 1000  # Normalize
    confidence = max(0.1, 1 - (rd / 350))  # Penalize uncertainty
    return max(0.01, base_weight * confidence)
```

---

## Weighted Consensus Algorithm

### Dynamic Weighted Majority (DWM)

For a given asset and time window:

```python
consensus_score = 0
total_weight = 0

for signal in recent_signals:
    direction_value = +1 if signal.direction == LONG else -1
    vote = direction_value * signal.confidence * creator.weight
    consensus_score += vote
    total_weight += creator.weight

normalized_score = consensus_score / total_weight  # -1 to +1
```

### Trade Logic

| Score Range | Action | Confidence |
|-------------|--------|------------|
| score > +0.6 | LONG | High |
| +0.3 < score < +0.6 | LONG | Low (optional) |
| -0.3 < score < +0.3 | **NO TRADE** | - |
| -0.6 < score < -0.3 | SHORT | Low (optional) |
| score < -0.6 | SHORT | High |

**The system is designed to do nothing most of the time.**

---

## Project Structure

```
crypto-consensus/
├── src/
│   ├── __init__.py
│   ├── config.py              # Settings from .env
│   ├── models.py              # SQLAlchemy + Pydantic models
│   ├── data_ingestion/
│   │   ├── __init__.py
│   │   ├── x_client.py        # Macrocosmos integration
│   │   └── price_client.py    # CCXT/Binance integration
│   ├── signal_extraction/
│   │   ├── __init__.py
│   │   ├── extractor.py       # NLP pipeline
│   │   └── rules.py           # Keyword fallback
│   ├── scoring/
│   │   ├── __init__.py
│   │   ├── evaluator.py       # Signal outcome evaluation
│   │   └── glicko2.py         # Rating system
│   ├── consensus/
│   │   ├── __init__.py
│   │   └── engine.py          # Weighted majority algorithm
│   └── output/
│       ├── __init__.py
│       ├── reports.py         # Report generation
│       └── exporter.py        # ZIP bundling
├── config/
│   └── creators.json          # List of tracked creators
├── data/
│   ├── raw/                   # Append-only raw data
│   └── processed/             # Derived datasets
├── reports/                   # Generated reports
├── tests/
├── .env                       # API keys (gitignored)
├── .env.example               # Template
├── requirements.txt
├── pyproject.toml
└── vision.md                  # This document
```

---

## Outputs

### 1. Creator Rankings
CSV/JSON with:
- Username
- Rating (Glicko-2)
- Accuracy %
- Total predictions
- Weight
- Rank

### 2. Consensus Signals
Real-time consensus state:
- Asset
- Action (LONG/SHORT/NO_TRADE)
- Confidence
- Contributing creators

### 3. Historical Data
- All signals with outcomes
- All consensus snapshots
- Price data used for evaluation

### 4. ZIP Export
Single command exports everything:
```bash
python -m src.main export
```

Creates: `crypto_consensus_export_YYYYMMDD.zip`

---

## Configuration

### Environment Variables (.env)

```
MACROCOSMOS_API_KEY=<your_key>
DATABASE_URL=sqlite:///data/consensus.db
EVALUATION_HORIZON_HOURS=24
CONSENSUS_THRESHOLD=0.3
```

### Creator List (config/creators.json)

```json
{
  "creators": [
    {"username": "trader1", "display_name": "Trader One"},
    {"username": "trader2", "display_name": "Trader Two"}
  ]
}
```

---

## Non-Goals

This system is **not**:
- a trading bot (no execution)
- a signal service (no alerts)
- financial advice

It is a **research and ranking engine**.

---

## Definition of Success

The project is successful when it:

1. ✅ Ingests X posts from tracked creators via Macrocosmos
2. ✅ Extracts trading signals using NLP
3. ✅ Evaluates signals against price movements
4. ✅ Ranks creators using Glicko-2
5. ✅ Produces weighted consensus with NO_TRADE states
6. ✅ Exports all results as a single ZIP file
7. ✅ Everything is auditable and reproducible

---

## Implementation Phases

### Phase 1: Data Foundation ✅
- [x] Project structure
- [x] Configuration system
- [x] Database models
- [x] Macrocosmos client
- [x] Price client (CCXT)

### Phase 2: Signal Extraction ✅
- [x] FinBERT integration (optional, graceful fallback)
- [x] Rule-based fallback
- [x] Signal validation

### Phase 3: Scoring System ✅
- [x] Outcome evaluator
- [x] Glicko-2 implementation
- [x] Leaderboard generation

### Phase 4: Consensus Engine ✅
- [x] Weighted majority algorithm
- [x] Threshold tuning
- [x] NO_TRADE detection

### Phase 5: Output & Export ✅
- [x] Report templates
- [x] ZIP bundling
- [x] CLI commands

### Phase 6: Automation (Optional)
- [ ] Scheduled collection
- [ ] Dashboard API
- [ ] Alerts

---

## Quick Start

```bash
# Activate environment
source venv/bin/activate

# Show creator rankings
python -m src.main rankings

# Show current consensus
python -m src.main consensus

# Export all data
python -m src.main export

# Generate test data
python -m src.main mock --signals 100
```
