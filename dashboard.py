"""
Premium Trading Dashboard v2.0

Professional-grade dashboard with:
- All 15 strategies (Twitter, Reddit, YouTube, Mixed, All)
- Tabbed navigation
- Real-time price tickers
- Signal analytics
- Creator leaderboard
- Advanced performance metrics
"""

import os
import json
import math
from datetime import datetime, timedelta
from pathlib import Path
from collections import defaultdict
from typing import Optional

from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from sqlalchemy import create_engine, desc, func, and_
from sqlalchemy.orm import sessionmaker

from src.models import (
    Base, Trade, TradeStatus, Portfolio, Asset, Direction,
    Signal, Creator, SignalSource
)

app = FastAPI(title="Quantum Trading Dashboard v2")

# Global exception handler
import traceback as tb

@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    return JSONResponse(
        content={"error": str(exc), "traceback": tb.format_exc()},
        status_code=500
    )

DATA_DIR = Path(os.getenv("DATA_DIR", "data"))
DB_PATH = DATA_DIR / "strategies.db"

# All 15 strategies
ALL_STRATEGIES = [
    # Twitter (3)
    "social_pure", "technical_strict", "balanced",
    # Reddit (3)
    "reddit_social_pure", "reddit_technical_strict", "reddit_balanced",
    # YouTube (3)
    "youtube_social_pure", "youtube_technical_strict", "youtube_balanced",
    # Mixed (3)
    "mixed_social_pure", "mixed_technical_strict", "mixed_balanced",
    # All sources (3)
    "all_social_pure", "all_technical_strict", "all_balanced",
]

STRATEGY_GROUPS = {
    "twitter": ["social_pure", "technical_strict", "balanced"],
    "reddit": ["reddit_social_pure", "reddit_technical_strict", "reddit_balanced"],
    "youtube": ["youtube_social_pure", "youtube_technical_strict", "youtube_balanced"],
    "mixed": ["mixed_social_pure", "mixed_technical_strict", "mixed_balanced"],
    "all": ["all_social_pure", "all_technical_strict", "all_balanced"],
}

STRATEGY_COLORS = {
    # Twitter - Greens/Blues
    "social_pure": "#10b981",
    "technical_strict": "#3b82f6",
    "balanced": "#8b5cf6",
    # Reddit - Oranges
    "reddit_social_pure": "#f97316",
    "reddit_technical_strict": "#fb923c",
    "reddit_balanced": "#f59e0b",
    # YouTube - Reds
    "youtube_social_pure": "#ef4444",
    "youtube_technical_strict": "#f87171",
    "youtube_balanced": "#dc2626",
    # Mixed - Pinks
    "mixed_social_pure": "#ec4899",
    "mixed_technical_strict": "#f472b6",
    "mixed_balanced": "#db2777",
    # All - Cyans
    "all_social_pure": "#06b6d4",
    "all_technical_strict": "#22d3ee",
    "all_balanced": "#0891b2",
}


def get_session():
    engine = create_engine(f"sqlite:///{DB_PATH}")
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    return Session()


def calculate_sharpe_ratio(returns: list[float], risk_free_rate: float = 0.02) -> float:
    """Calculate annualized Sharpe ratio."""
    if not returns or len(returns) < 2:
        return 0.0
    avg_return = sum(returns) / len(returns)
    std_dev = (sum((r - avg_return) ** 2 for r in returns) / len(returns)) ** 0.5
    if std_dev == 0:
        return 0.0
    # Annualize (assuming daily returns)
    return ((avg_return - risk_free_rate / 365) / std_dev) * (365 ** 0.5)


def calculate_sortino_ratio(returns: list[float], risk_free_rate: float = 0.02) -> float:
    """Calculate Sortino ratio (only considers downside deviation)."""
    if not returns or len(returns) < 2:
        return 0.0
    avg_return = sum(returns) / len(returns)
    negative_returns = [r for r in returns if r < 0]
    if not negative_returns:
        return float('inf') if avg_return > 0 else 0.0
    downside_dev = (sum(r ** 2 for r in negative_returns) / len(negative_returns)) ** 0.5
    if downside_dev == 0:
        return 0.0
    return ((avg_return - risk_free_rate / 365) / downside_dev) * (365 ** 0.5)


def calculate_profit_factor(wins: list[float], losses: list[float]) -> float:
    """Calculate profit factor (gross profit / gross loss)."""
    gross_profit = sum(wins) if wins else 0
    gross_loss = abs(sum(losses)) if losses else 0
    if gross_loss == 0:
        return float('inf') if gross_profit > 0 else 0.0
    return gross_profit / gross_loss


def get_strategy_stats(session, strategy_name: str) -> Optional[dict]:
    """Get comprehensive stats for a strategy."""
    portfolio = session.query(Portfolio).filter(
        Portfolio.name == f"strategy_{strategy_name}"
    ).first()

    if not portfolio:
        return None

    open_trades = session.query(Trade).filter(
        Trade.status == TradeStatus.OPEN,
        Trade.notes == f"strategy_{strategy_name}",
    ).all()

    closed_trades = session.query(Trade).filter(
        Trade.status == TradeStatus.CLOSED,
        Trade.notes == f"strategy_{strategy_name}",
    ).order_by(Trade.exit_time).all()

    # Equity curve
    equity_curve = []
    running_balance = portfolio.initial_balance
    daily_returns = []
    prev_balance = running_balance

    for trade in closed_trades:
        if trade.pnl_usd and trade.exit_time:
            running_balance += trade.pnl_usd
            daily_return = (running_balance - prev_balance) / prev_balance if prev_balance > 0 else 0
            daily_returns.append(daily_return)
            prev_balance = running_balance
            equity_curve.append({
                "time": trade.exit_time.isoformat(),
                "balance": running_balance,
                "pnl": trade.pnl_usd,
            })

    total_return = ((portfolio.current_balance - portfolio.initial_balance) / portfolio.initial_balance) * 100
    win_rate = (portfolio.winning_trades / portfolio.total_trades * 100) if portfolio.total_trades > 0 else 0

    # Max drawdown calculation
    peak = portfolio.initial_balance
    max_dd = 0
    max_dd_duration = 0
    dd_start = None

    for point in equity_curve:
        if point["balance"] > peak:
            peak = point["balance"]
            dd_start = None
        dd = (peak - point["balance"]) / peak * 100
        if dd > max_dd:
            max_dd = dd

    # PnL by asset
    pnl_by_asset = defaultdict(float)
    trades_by_asset = defaultdict(int)
    for trade in closed_trades:
        if trade.pnl_usd:
            pnl_by_asset[trade.asset.value] += trade.pnl_usd
            trades_by_asset[trade.asset.value] += 1

    recent_trades = closed_trades[-20:] if closed_trades else []
    recent_trades.reverse()

    wins = [t.pnl_percent for t in closed_trades if t.pnl_percent and t.pnl_percent > 0]
    losses = [t.pnl_percent for t in closed_trades if t.pnl_percent and t.pnl_percent < 0]
    avg_win = sum(wins) / len(wins) if wins else 0
    avg_loss = sum(losses) / len(losses) if losses else 0

    # Advanced metrics
    sharpe = calculate_sharpe_ratio(daily_returns)
    sortino = calculate_sortino_ratio(daily_returns)
    profit_factor = calculate_profit_factor(wins, losses)

    # Win/Loss streaks
    current_streak = 0
    max_win_streak = 0
    max_loss_streak = 0
    temp_streak = 0

    for trade in closed_trades:
        if trade.pnl_percent and trade.pnl_percent > 0:
            if temp_streak >= 0:
                temp_streak += 1
            else:
                temp_streak = 1
            max_win_streak = max(max_win_streak, temp_streak)
        elif trade.pnl_percent and trade.pnl_percent < 0:
            if temp_streak <= 0:
                temp_streak -= 1
            else:
                temp_streak = -1
            max_loss_streak = max(max_loss_streak, abs(temp_streak))
        current_streak = temp_streak

    # Daily PnL
    daily_pnl = defaultdict(float)
    for trade in closed_trades:
        if trade.exit_time and trade.pnl_usd:
            day = trade.exit_time.strftime("%Y-%m-%d")
            daily_pnl[day] += trade.pnl_usd

    # Hourly distribution
    hourly_trades = defaultdict(lambda: {"count": 0, "pnl": 0})
    for trade in closed_trades:
        if trade.entry_time and trade.pnl_usd:
            hour = trade.entry_time.hour
            hourly_trades[hour]["count"] += 1
            hourly_trades[hour]["pnl"] += trade.pnl_usd

    return {
        "name": strategy_name,
        "balance": portfolio.current_balance,
        "initial": portfolio.initial_balance,
        "total_return": total_return,
        "total_pnl": portfolio.total_pnl,
        "total_trades": portfolio.total_trades,
        "winning_trades": portfolio.winning_trades,
        "losing_trades": portfolio.total_trades - portfolio.winning_trades,
        "win_rate": win_rate,
        "max_drawdown": max_dd,
        "avg_win": avg_win,
        "avg_loss": avg_loss,
        "sharpe_ratio": sharpe,
        "sortino_ratio": sortino,
        "profit_factor": profit_factor,
        "max_win_streak": max_win_streak,
        "max_loss_streak": max_loss_streak,
        "current_streak": current_streak,
        "open_positions": open_trades,
        "recent_trades": recent_trades,
        "equity_curve": equity_curve,
        "pnl_by_asset": dict(pnl_by_asset),
        "trades_by_asset": dict(trades_by_asset),
        "daily_pnl": dict(daily_pnl),
        "hourly_distribution": dict(hourly_trades),
    }


def get_signal_stats(session) -> dict:
    """Get signal analytics."""
    now = datetime.utcnow()
    day_ago = now - timedelta(hours=24)
    week_ago = now - timedelta(days=7)

    # Counts by source
    twitter_total = session.query(Signal).filter(Signal.source == SignalSource.TWITTER).count()
    reddit_total = session.query(Signal).filter(Signal.source == SignalSource.REDDIT).count()
    youtube_total = session.query(Signal).filter(Signal.source == SignalSource.YOUTUBE).count()

    # Recent signals
    twitter_24h = session.query(Signal).filter(
        Signal.source == SignalSource.TWITTER,
        Signal.posted_at >= day_ago
    ).count()
    reddit_24h = session.query(Signal).filter(
        Signal.source == SignalSource.REDDIT,
        Signal.posted_at >= day_ago
    ).count()
    youtube_24h = session.query(Signal).filter(
        Signal.source == SignalSource.YOUTUBE,
        Signal.posted_at >= day_ago
    ).count()

    # Signals by asset (last 7 days)
    recent_signals = session.query(Signal).filter(Signal.posted_at >= week_ago).all()
    by_asset = defaultdict(lambda: {"long": 0, "short": 0, "total": 0})
    for sig in recent_signals:
        if sig.asset:
            by_asset[sig.asset.value]["total"] += 1
            if sig.direction == Direction.LONG:
                by_asset[sig.asset.value]["long"] += 1
            elif sig.direction == Direction.SHORT:
                by_asset[sig.asset.value]["short"] += 1

    # Confidence distribution
    all_signals = session.query(Signal).all()
    confidence_buckets = {"high": 0, "medium": 0, "low": 0}
    for sig in all_signals:
        if sig.confidence:
            if sig.confidence >= 0.5:
                confidence_buckets["high"] += 1
            elif sig.confidence >= 0.3:
                confidence_buckets["medium"] += 1
            else:
                confidence_buckets["low"] += 1

    # Latest signals
    latest = session.query(Signal).order_by(desc(Signal.posted_at)).limit(20).all()

    return {
        "totals": {
            "twitter": twitter_total,
            "reddit": reddit_total,
            "youtube": youtube_total,
            "total": twitter_total + reddit_total + youtube_total,
        },
        "last_24h": {
            "twitter": twitter_24h,
            "reddit": reddit_24h,
            "youtube": youtube_24h,
            "total": twitter_24h + reddit_24h + youtube_24h,
        },
        "by_asset": dict(by_asset),
        "confidence_distribution": confidence_buckets,
        "latest_signals": [
            {
                "asset": s.asset.value if s.asset else "N/A",
                "direction": s.direction.value if s.direction else "N/A",
                "confidence": s.confidence,
                "source": s.source.value if s.source else "N/A",
                "username": s.creator.username if s.creator else "Unknown",
                "posted_at": s.posted_at.isoformat() if s.posted_at else None,
            }
            for s in latest
        ],
    }


def get_creator_stats(session, limit: int = 50) -> dict:
    """Get creator leaderboard and stats."""
    # Top creators overall
    top_creators = session.query(Creator).filter(
        Creator.total_predictions >= 3
    ).order_by(desc(Creator.rating)).limit(limit).all()

    # By source
    def get_top_by_source(prefix: str = "", exclude_prefix: bool = False):
        query = session.query(Creator).filter(Creator.total_predictions >= 3)
        if prefix:
            query = query.filter(Creator.username.like(f"{prefix}%"))
        elif exclude_prefix:
            query = query.filter(
                ~Creator.username.like("r/%"),
                ~Creator.username.like("yt/%")
            )
        return query.order_by(desc(Creator.rating)).limit(10).all()

    twitter_top = get_top_by_source(exclude_prefix=True)
    reddit_top = get_top_by_source("r/")
    youtube_top = get_top_by_source("yt/")

    def format_creator(c, rank=None):
        return {
            "rank": rank,
            "username": c.username,
            "display_name": c.display_name,
            "rating": round(c.rating, 1),
            "rd": round(c.rating_deviation, 1),
            "accuracy": round(c.accuracy * 100, 1) if c.total_predictions > 0 else 0,
            "predictions": c.total_predictions,
            "correct": c.correct_predictions,
            "weight": round(c.weight, 4),
        }

    return {
        "total_creators": session.query(Creator).count(),
        "active_creators": session.query(Creator).filter(Creator.total_predictions >= 3).count(),
        "leaderboard": [format_creator(c, i+1) for i, c in enumerate(top_creators)],
        "top_twitter": [format_creator(c, i+1) for i, c in enumerate(twitter_top)],
        "top_reddit": [format_creator(c, i+1) for i, c in enumerate(reddit_top)],
        "top_youtube": [format_creator(c, i+1) for i, c in enumerate(youtube_top)],
    }


@app.get("/", response_class=HTMLResponse)
async def dashboard():
    """Main dashboard page."""
    try:
        return await render_dashboard()
    except Exception as e:
        import traceback
        return HTMLResponse(
            content=f"<pre style='color:#ef4444;background:#0a0a0f;padding:20px;'>Error: {e}\n\n{traceback.format_exc()}</pre>",
            status_code=500
        )


async def render_dashboard():
    """Render the main dashboard HTML."""
    session = get_session()

    try:
        # Get all strategy stats
        stats = {}
        for s in ALL_STRATEGIES:
            data = get_strategy_stats(session, s)
            if data:
                stats[s] = data

        # Get signal and creator stats
        signal_stats = get_signal_stats(session)
        creator_stats = get_creator_stats(session)

        # Find leader
        leader = None
        best_return = -999
        for name, data in stats.items():
            if data["total_return"] > best_return:
                best_return = data["total_return"]
                leader = name

        # Calculate totals
        total_balance = sum(d["balance"] for d in stats.values()) if stats else 50000
        total_initial = sum(d["initial"] for d in stats.values()) if stats else 50000
        total_pnl = sum(d["total_pnl"] for d in stats.values()) if stats else 0
        total_trades = sum(d["total_trades"] for d in stats.values()) if stats else 0
        avg_win_rate = sum(d["win_rate"] for d in stats.values()) / len(stats) if stats else 0
        total_return = ((total_balance - total_initial) / total_initial * 100) if total_initial > 0 else 0

        # Best and worst performers
        if stats:
            sorted_by_return = sorted(stats.items(), key=lambda x: x[1]["total_return"], reverse=True)
            best_performer = sorted_by_return[0] if sorted_by_return else None
            worst_performer = sorted_by_return[-1] if sorted_by_return else None
        else:
            best_performer = worst_performer = None

        equity_data = {name: data["equity_curve"] for name, data in stats.items()}
        daily_pnl_data = {name: data["daily_pnl"] for name, data in stats.items()}

    finally:
        session.close()

    # Generate HTML
    html = generate_dashboard_html(
        stats=stats,
        signal_stats=signal_stats,
        creator_stats=creator_stats,
        leader=leader,
        best_return=best_return,
        total_balance=total_balance,
        total_pnl=total_pnl,
        total_trades=total_trades,
        total_return=total_return,
        avg_win_rate=avg_win_rate,
        equity_data=equity_data,
        daily_pnl_data=daily_pnl_data,
    )

    return HTMLResponse(content=html)


def generate_dashboard_html(**kwargs) -> str:
    """Generate the dashboard HTML."""
    stats = kwargs.get("stats", {})
    signal_stats = kwargs.get("signal_stats", {})
    creator_stats = kwargs.get("creator_stats", {})
    leader = kwargs.get("leader")
    best_return = kwargs.get("best_return", 0)
    total_balance = kwargs.get("total_balance", 0)
    total_pnl = kwargs.get("total_pnl", 0)
    total_trades = kwargs.get("total_trades", 0)
    total_return = kwargs.get("total_return", 0)
    avg_win_rate = kwargs.get("avg_win_rate", 0)
    equity_data = kwargs.get("equity_data", {})

    html = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Quantum Trading Dashboard</title>
    <meta http-equiv="refresh" content="30">
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/chartjs-adapter-date-fns"></script>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap" rel="stylesheet">
    <style>
        :root {{
            --bg-primary: #0a0a0f;
            --bg-secondary: #12121a;
            --bg-card: #16161f;
            --bg-hover: #1c1c28;
            --border: #252532;
            --border-light: #32324a;
            --text-primary: #ffffff;
            --text-secondary: #9ca3af;
            --text-muted: #6b7280;
            --accent-blue: #3b82f6;
            --accent-purple: #8b5cf6;
            --accent-cyan: #06b6d4;
            --accent-pink: #ec4899;
            --accent-orange: #f97316;
            --green: #10b981;
            --green-light: #34d399;
            --red: #ef4444;
            --red-light: #f87171;
            --yellow: #f59e0b;
            --gradient-blue: linear-gradient(135deg, #3b82f6, #8b5cf6);
            --gradient-green: linear-gradient(135deg, #10b981, #06b6d4);
            --gradient-red: linear-gradient(135deg, #ef4444, #f97316);
            --shadow-lg: 0 25px 50px -12px rgba(0, 0, 0, 0.5);
            --shadow-glow: 0 0 40px rgba(59, 130, 246, 0.15);
        }}

        * {{ margin: 0; padding: 0; box-sizing: border-box; }}

        body {{
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
            background: var(--bg-primary);
            color: var(--text-primary);
            min-height: 100vh;
            overflow-x: hidden;
        }}

        /* Animated Background */
        .bg-grid {{
            position: fixed;
            inset: 0;
            background-image:
                radial-gradient(circle at 25% 25%, rgba(59, 130, 246, 0.05) 0%, transparent 50%),
                radial-gradient(circle at 75% 75%, rgba(139, 92, 246, 0.05) 0%, transparent 50%);
            pointer-events: none;
            z-index: 0;
        }}

        .bg-grid::after {{
            content: '';
            position: absolute;
            inset: 0;
            background-image:
                linear-gradient(rgba(255,255,255,0.02) 1px, transparent 1px),
                linear-gradient(90deg, rgba(255,255,255,0.02) 1px, transparent 1px);
            background-size: 60px 60px;
        }}

        /* Layout */
        .app {{
            position: relative;
            z-index: 1;
            display: flex;
            min-height: 100vh;
        }}

        /* Sidebar */
        .sidebar {{
            width: 260px;
            background: var(--bg-secondary);
            border-right: 1px solid var(--border);
            padding: 24px 16px;
            position: fixed;
            height: 100vh;
            overflow-y: auto;
            z-index: 100;
        }}

        .logo {{
            display: flex;
            align-items: center;
            gap: 12px;
            padding: 0 8px;
            margin-bottom: 32px;
        }}

        .logo-icon {{
            width: 40px;
            height: 40px;
            background: var(--gradient-blue);
            border-radius: 10px;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 20px;
            box-shadow: var(--shadow-glow);
        }}

        .logo-text {{
            font-size: 1.25rem;
            font-weight: 700;
            background: var(--gradient-blue);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }}

        .nav-section {{
            margin-bottom: 24px;
        }}

        .nav-label {{
            font-size: 0.7rem;
            font-weight: 600;
            color: var(--text-muted);
            text-transform: uppercase;
            letter-spacing: 1px;
            padding: 0 12px;
            margin-bottom: 8px;
        }}

        .nav-item {{
            display: flex;
            align-items: center;
            gap: 12px;
            padding: 12px;
            border-radius: 10px;
            color: var(--text-secondary);
            cursor: pointer;
            transition: all 0.2s;
            margin-bottom: 4px;
        }}

        .nav-item:hover {{
            background: var(--bg-hover);
            color: var(--text-primary);
        }}

        .nav-item.active {{
            background: rgba(59, 130, 246, 0.15);
            color: var(--accent-blue);
        }}

        .nav-item .icon {{ font-size: 1.1rem; width: 24px; text-align: center; }}
        .nav-item .label {{ font-size: 0.9rem; font-weight: 500; }}

        .nav-badge {{
            margin-left: auto;
            background: var(--accent-blue);
            color: white;
            font-size: 0.7rem;
            font-weight: 600;
            padding: 2px 8px;
            border-radius: 10px;
        }}

        /* Main Content */
        .main {{
            flex: 1;
            margin-left: 260px;
            padding: 24px;
            min-height: 100vh;
        }}

        /* Header */
        .header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 24px;
            padding-bottom: 24px;
            border-bottom: 1px solid var(--border);
        }}

        .header-left h1 {{
            font-size: 1.75rem;
            font-weight: 700;
            margin-bottom: 4px;
        }}

        .header-left p {{
            color: var(--text-secondary);
            font-size: 0.9rem;
        }}

        .header-right {{
            display: flex;
            align-items: center;
            gap: 16px;
        }}

        .live-badge {{
            display: flex;
            align-items: center;
            gap: 8px;
            background: rgba(16, 185, 129, 0.15);
            border: 1px solid rgba(16, 185, 129, 0.3);
            padding: 8px 16px;
            border-radius: 20px;
            color: var(--green);
            font-weight: 500;
            font-size: 0.85rem;
        }}

        .live-dot {{
            width: 8px;
            height: 8px;
            background: var(--green);
            border-radius: 50%;
            animation: pulse 2s infinite;
        }}

        @keyframes pulse {{
            0%, 100% {{ opacity: 1; transform: scale(1); }}
            50% {{ opacity: 0.5; transform: scale(1.2); }}
        }}

        .time-display {{
            font-family: 'JetBrains Mono', monospace;
            color: var(--text-secondary);
            font-size: 0.85rem;
        }}

        /* Stats Grid */
        .stats-grid {{
            display: grid;
            grid-template-columns: repeat(5, 1fr);
            gap: 16px;
            margin-bottom: 24px;
        }}

        .stat-card {{
            background: var(--bg-card);
            border: 1px solid var(--border);
            border-radius: 16px;
            padding: 20px;
            transition: all 0.3s;
        }}

        .stat-card:hover {{
            border-color: var(--border-light);
            transform: translateY(-2px);
        }}

        .stat-card .label {{
            font-size: 0.75rem;
            color: var(--text-muted);
            text-transform: uppercase;
            letter-spacing: 0.5px;
            margin-bottom: 8px;
        }}

        .stat-card .value {{
            font-size: 1.75rem;
            font-weight: 700;
            font-family: 'JetBrains Mono', monospace;
        }}

        .stat-card .value.positive {{ color: var(--green); }}
        .stat-card .value.negative {{ color: var(--red); }}

        .stat-card .change {{
            font-size: 0.8rem;
            color: var(--text-secondary);
            margin-top: 4px;
        }}

        /* Leader Banner */
        .leader-banner {{
            background: linear-gradient(135deg, rgba(16, 185, 129, 0.1), rgba(6, 182, 212, 0.1));
            border: 1px solid rgba(16, 185, 129, 0.2);
            border-radius: 16px;
            padding: 20px 24px;
            display: flex;
            align-items: center;
            justify-content: space-between;
            margin-bottom: 24px;
        }}

        .leader-info {{
            display: flex;
            align-items: center;
            gap: 16px;
        }}

        .leader-icon {{ font-size: 2rem; }}

        .leader-text h3 {{
            color: var(--text-secondary);
            font-size: 0.85rem;
            font-weight: 500;
        }}

        .leader-text h2 {{
            font-size: 1.5rem;
            font-weight: 700;
            color: var(--green);
        }}

        .leader-stats {{
            display: flex;
            gap: 32px;
        }}

        .leader-stat {{
            text-align: center;
        }}

        .leader-stat .value {{
            font-size: 1.5rem;
            font-weight: 700;
            font-family: 'JetBrains Mono', monospace;
            color: var(--green);
        }}

        .leader-stat .label {{
            font-size: 0.75rem;
            color: var(--text-secondary);
        }}

        /* Tabs */
        .tabs {{
            display: flex;
            gap: 8px;
            margin-bottom: 24px;
            background: var(--bg-secondary);
            padding: 6px;
            border-radius: 12px;
            width: fit-content;
        }}

        .tab {{
            padding: 10px 20px;
            border-radius: 8px;
            font-size: 0.9rem;
            font-weight: 500;
            color: var(--text-secondary);
            cursor: pointer;
            transition: all 0.2s;
            border: none;
            background: transparent;
        }}

        .tab:hover {{ color: var(--text-primary); }}
        .tab.active {{
            background: var(--accent-blue);
            color: white;
        }}

        /* Strategy Grid */
        .strategy-section {{
            margin-bottom: 32px;
        }}

        .section-header {{
            display: flex;
            align-items: center;
            gap: 12px;
            margin-bottom: 16px;
        }}

        .section-header h2 {{
            font-size: 1.1rem;
            font-weight: 600;
        }}

        .section-header .badge {{
            font-size: 0.7rem;
            padding: 4px 10px;
            border-radius: 6px;
            font-weight: 600;
        }}

        .badge-twitter {{ background: rgba(29, 161, 242, 0.2); color: #1da1f2; }}
        .badge-reddit {{ background: rgba(255, 69, 0, 0.2); color: #ff4500; }}
        .badge-youtube {{ background: rgba(255, 0, 0, 0.2); color: #ff0000; }}
        .badge-mixed {{ background: rgba(236, 72, 153, 0.2); color: #ec4899; }}
        .badge-all {{ background: rgba(6, 182, 212, 0.2); color: #06b6d4; }}

        .strategies-grid {{
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 16px;
        }}

        .strategy-card {{
            background: var(--bg-card);
            border: 1px solid var(--border);
            border-radius: 16px;
            overflow: hidden;
            transition: all 0.3s;
        }}

        .strategy-card:hover {{
            border-color: var(--border-light);
            transform: translateY(-4px);
            box-shadow: var(--shadow-lg);
        }}

        .strategy-card.leader {{
            border-color: var(--green);
            box-shadow: 0 0 30px rgba(16, 185, 129, 0.15);
        }}

        .strategy-card-header {{
            padding: 16px 20px;
            border-bottom: 1px solid var(--border);
            display: flex;
            justify-content: space-between;
            align-items: center;
        }}

        .strategy-name {{
            font-size: 0.9rem;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }}

        .leader-badge {{
            background: var(--gradient-green);
            color: #000;
            font-size: 0.65rem;
            font-weight: 700;
            padding: 4px 10px;
            border-radius: 12px;
            text-transform: uppercase;
        }}

        .strategy-card-body {{
            padding: 20px;
        }}

        .strategy-metrics {{
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 12px;
            margin-bottom: 16px;
        }}

        .metric {{
            text-align: center;
            padding: 12px 8px;
            background: var(--bg-secondary);
            border-radius: 10px;
        }}

        .metric .value {{
            font-size: 1.1rem;
            font-weight: 700;
            font-family: 'JetBrains Mono', monospace;
        }}

        .metric .label {{
            font-size: 0.65rem;
            color: var(--text-muted);
            text-transform: uppercase;
            margin-top: 4px;
        }}

        .mini-chart {{
            height: 80px;
            margin: 16px 0;
        }}

        .positions-list {{
            border-top: 1px solid var(--border);
            padding-top: 12px;
            margin-top: 12px;
        }}

        .positions-header {{
            font-size: 0.7rem;
            color: var(--text-muted);
            text-transform: uppercase;
            margin-bottom: 8px;
        }}

        .position {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 8px 12px;
            background: var(--bg-secondary);
            border-radius: 8px;
            margin-bottom: 6px;
            font-size: 0.85rem;
        }}

        .position-info {{
            display: flex;
            align-items: center;
            gap: 8px;
        }}

        .direction-badge {{
            font-size: 0.65rem;
            font-weight: 700;
            padding: 3px 8px;
            border-radius: 4px;
            text-transform: uppercase;
        }}

        .direction-badge.long {{
            background: rgba(16, 185, 129, 0.2);
            color: var(--green);
        }}

        .direction-badge.short {{
            background: rgba(239, 68, 68, 0.2);
            color: var(--red);
        }}

        .no-positions {{
            text-align: center;
            color: var(--text-muted);
            font-size: 0.8rem;
            padding: 12px;
        }}

        /* Chart Section */
        .chart-section {{
            background: var(--bg-card);
            border: 1px solid var(--border);
            border-radius: 16px;
            padding: 24px;
            margin-bottom: 24px;
        }}

        .chart-header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 20px;
        }}

        .chart-title {{
            font-size: 1.1rem;
            font-weight: 600;
        }}

        .chart-container {{
            height: 350px;
        }}

        /* Signals Section */
        .signals-grid {{
            display: grid;
            grid-template-columns: repeat(4, 1fr);
            gap: 16px;
            margin-bottom: 24px;
        }}

        .signal-stat {{
            background: var(--bg-card);
            border: 1px solid var(--border);
            border-radius: 12px;
            padding: 16px;
            text-align: center;
        }}

        .signal-stat .icon {{
            font-size: 1.5rem;
            margin-bottom: 8px;
        }}

        .signal-stat .value {{
            font-size: 1.5rem;
            font-weight: 700;
            font-family: 'JetBrains Mono', monospace;
        }}

        .signal-stat .label {{
            font-size: 0.75rem;
            color: var(--text-secondary);
            margin-top: 4px;
        }}

        .signal-stat .sub {{
            font-size: 0.7rem;
            color: var(--text-muted);
        }}

        /* Creators Section */
        .creators-grid {{
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 16px;
        }}

        .creator-card {{
            background: var(--bg-card);
            border: 1px solid var(--border);
            border-radius: 12px;
            overflow: hidden;
        }}

        .creator-card-header {{
            padding: 12px 16px;
            background: var(--bg-secondary);
            border-bottom: 1px solid var(--border);
            font-weight: 600;
            font-size: 0.85rem;
            display: flex;
            align-items: center;
            gap: 8px;
        }}

        .creator-list {{
            padding: 8px;
        }}

        .creator-item {{
            display: flex;
            align-items: center;
            padding: 10px 12px;
            border-radius: 8px;
            transition: background 0.2s;
        }}

        .creator-item:hover {{
            background: var(--bg-hover);
        }}

        .creator-rank {{
            width: 28px;
            height: 28px;
            background: var(--bg-secondary);
            border-radius: 50%;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 0.75rem;
            font-weight: 600;
            margin-right: 12px;
        }}

        .creator-rank.gold {{ background: linear-gradient(135deg, #fbbf24, #f59e0b); color: #000; }}
        .creator-rank.silver {{ background: linear-gradient(135deg, #9ca3af, #6b7280); color: #000; }}
        .creator-rank.bronze {{ background: linear-gradient(135deg, #d97706, #b45309); color: #fff; }}

        .creator-info {{
            flex: 1;
        }}

        .creator-name {{
            font-size: 0.85rem;
            font-weight: 500;
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
            max-width: 120px;
        }}

        .creator-stats {{
            font-size: 0.7rem;
            color: var(--text-muted);
        }}

        .creator-rating {{
            text-align: right;
        }}

        .creator-rating .rating {{
            font-size: 1rem;
            font-weight: 700;
            font-family: 'JetBrains Mono', monospace;
        }}

        .creator-rating .accuracy {{
            font-size: 0.7rem;
            color: var(--green);
        }}

        /* Trades Table */
        .trades-section {{
            background: var(--bg-card);
            border: 1px solid var(--border);
            border-radius: 16px;
            overflow: hidden;
        }}

        .trades-header {{
            padding: 20px 24px;
            border-bottom: 1px solid var(--border);
            display: flex;
            justify-content: space-between;
            align-items: center;
        }}

        .trades-title {{
            font-size: 1.1rem;
            font-weight: 600;
        }}

        .trades-table {{
            width: 100%;
            border-collapse: collapse;
        }}

        .trades-table th {{
            padding: 12px 16px;
            text-align: left;
            font-size: 0.7rem;
            color: var(--text-muted);
            text-transform: uppercase;
            letter-spacing: 0.5px;
            background: var(--bg-secondary);
            font-weight: 600;
        }}

        .trades-table td {{
            padding: 14px 16px;
            border-bottom: 1px solid var(--border);
            font-size: 0.85rem;
        }}

        .trades-table tr:hover td {{
            background: var(--bg-hover);
        }}

        .trade-pnl {{ font-family: 'JetBrains Mono', monospace; font-weight: 600; }}
        .trade-pnl.positive {{ color: var(--green); }}
        .trade-pnl.negative {{ color: var(--red); }}

        /* Footer */
        .footer {{
            text-align: center;
            padding: 24px;
            color: var(--text-muted);
            font-size: 0.8rem;
            border-top: 1px solid var(--border);
            margin-top: 32px;
        }}

        /* Responsive */
        @media (max-width: 1400px) {{
            .stats-grid {{ grid-template-columns: repeat(3, 1fr); }}
            .strategies-grid {{ grid-template-columns: repeat(2, 1fr); }}
        }}

        @media (max-width: 1200px) {{
            .sidebar {{ display: none; }}
            .main {{ margin-left: 0; }}
            .signals-grid {{ grid-template-columns: repeat(2, 1fr); }}
            .creators-grid {{ grid-template-columns: 1fr; }}
        }}

        @media (max-width: 768px) {{
            .stats-grid {{ grid-template-columns: 1fr 1fr; }}
            .strategies-grid {{ grid-template-columns: 1fr; }}
            .signals-grid {{ grid-template-columns: 1fr; }}
            .header {{ flex-direction: column; gap: 16px; text-align: center; }}
            .header-right {{ justify-content: center; }}
            .leader-banner {{ flex-direction: column; text-align: center; gap: 16px; }}
        }}

        /* Scrollbar */
        ::-webkit-scrollbar {{ width: 8px; height: 8px; }}
        ::-webkit-scrollbar-track {{ background: var(--bg-secondary); }}
        ::-webkit-scrollbar-thumb {{ background: var(--border-light); border-radius: 4px; }}
        ::-webkit-scrollbar-thumb:hover {{ background: var(--text-muted); }}
    </style>
</head>
<body>
    <div class="bg-grid"></div>
    <div class="app">
        <!-- Sidebar -->
        <aside class="sidebar">
            <div class="logo">
                <div class="logo-icon">Q</div>
                <span class="logo-text">Quantum</span>
            </div>

            <nav class="nav-section">
                <div class="nav-label">Overview</div>
                <div class="nav-item active">
                    <span class="icon">D</span>
                    <span class="label">Dashboard</span>
                </div>
                <div class="nav-item">
                    <span class="icon">$</span>
                    <span class="label">Strategies</span>
                    <span class="nav-badge">{len(stats)}</span>
                </div>
            </nav>

            <nav class="nav-section">
                <div class="nav-label">Analytics</div>
                <div class="nav-item">
                    <span class="icon">~</span>
                    <span class="label">Signals</span>
                    <span class="nav-badge">{signal_stats.get('totals', {}).get('total', 0)}</span>
                </div>
                <div class="nav-item">
                    <span class="icon">*</span>
                    <span class="label">Creators</span>
                </div>
                <div class="nav-item">
                    <span class="icon">#</span>
                    <span class="label">Trades</span>
                </div>
            </nav>

            <nav class="nav-section">
                <div class="nav-label">System</div>
                <div class="nav-item">
                    <span class="icon">?</span>
                    <span class="label">Health</span>
                </div>
                <div class="nav-item">
                    <span class="icon">!</span>
                    <span class="label">Settings</span>
                </div>
            </nav>
        </aside>

        <!-- Main Content -->
        <main class="main">
            <!-- Header -->
            <header class="header">
                <div class="header-left">
                    <h1>Trading Dashboard</h1>
                    <p>Multi-Strategy A/B Testing Platform - 15 Strategies</p>
                </div>
                <div class="header-right">
                    <div class="live-badge">
                        <span class="live-dot"></span>
                        Live Trading
                    </div>
                    <div class="time-display">{datetime.utcnow().strftime("%Y-%m-%d %H:%M")} UTC</div>
                </div>
            </header>

            <!-- Stats Overview -->
            <div class="stats-grid">
                <div class="stat-card">
                    <div class="label">Total Balance</div>
                    <div class="value">${total_balance:,.0f}</div>
                    <div class="change">{len(stats)} strategies combined</div>
                </div>
                <div class="stat-card">
                    <div class="label">Total Return</div>
                    <div class="value {'positive' if total_return >= 0 else 'negative'}">{total_return:+.2f}%</div>
                    <div class="change">Since inception</div>
                </div>
                <div class="stat-card">
                    <div class="label">Total P&L</div>
                    <div class="value {'positive' if total_pnl >= 0 else 'negative'}">${total_pnl:+,.2f}</div>
                    <div class="change">Net profit/loss</div>
                </div>
                <div class="stat-card">
                    <div class="label">Total Trades</div>
                    <div class="value">{total_trades}</div>
                    <div class="change">All strategies</div>
                </div>
                <div class="stat-card">
                    <div class="label">Avg Win Rate</div>
                    <div class="value {'positive' if avg_win_rate >= 50 else 'negative'}">{avg_win_rate:.1f}%</div>
                    <div class="change">Combined average</div>
                </div>
            </div>

            <!-- Leader Banner -->
            {f'''
            <div class="leader-banner">
                <div class="leader-info">
                    <div class="leader-icon">T</div>
                    <div class="leader-text">
                        <h3>Current Leader</h3>
                        <h2>{leader.replace("_", " ").upper() if leader else "N/A"}</h2>
                    </div>
                </div>
                <div class="leader-stats">
                    <div class="leader-stat">
                        <div class="value">{best_return:+.2f}%</div>
                        <div class="label">Return</div>
                    </div>
                    <div class="leader-stat">
                        <div class="value">{stats[leader]["win_rate"]:.0f}%</div>
                        <div class="label">Win Rate</div>
                    </div>
                    <div class="leader-stat">
                        <div class="value">{stats[leader]["total_trades"]}</div>
                        <div class="label">Trades</div>
                    </div>
                </div>
            </div>
            ''' if leader and leader in stats else '''
            <div class="leader-banner">
                <div class="leader-info">
                    <div class="leader-icon">...</div>
                    <div class="leader-text">
                        <h3>Waiting</h3>
                        <h2>No trades yet</h2>
                    </div>
                </div>
            </div>
            '''}

            <!-- Signal Stats -->
            <div class="signals-grid">
                <div class="signal-stat">
                    <div class="icon">X</div>
                    <div class="value">{signal_stats.get('totals', {}).get('twitter', 0):,}</div>
                    <div class="label">Twitter Signals</div>
                    <div class="sub">+{signal_stats.get('last_24h', {}).get('twitter', 0)} last 24h</div>
                </div>
                <div class="signal-stat">
                    <div class="icon">R</div>
                    <div class="value">{signal_stats.get('totals', {}).get('reddit', 0):,}</div>
                    <div class="label">Reddit Signals</div>
                    <div class="sub">+{signal_stats.get('last_24h', {}).get('reddit', 0)} last 24h</div>
                </div>
                <div class="signal-stat">
                    <div class="icon">Y</div>
                    <div class="value">{signal_stats.get('totals', {}).get('youtube', 0):,}</div>
                    <div class="label">YouTube Signals</div>
                    <div class="sub">+{signal_stats.get('last_24h', {}).get('youtube', 0)} last 24h</div>
                </div>
                <div class="signal-stat">
                    <div class="icon">=</div>
                    <div class="value">{signal_stats.get('totals', {}).get('total', 0):,}</div>
                    <div class="label">Total Signals</div>
                    <div class="sub">+{signal_stats.get('last_24h', {}).get('total', 0)} last 24h</div>
                </div>
            </div>

            <!-- Strategy Tabs -->
            <div class="tabs">
                <button class="tab active" onclick="showGroup('all')">All</button>
                <button class="tab" onclick="showGroup('twitter')">Twitter</button>
                <button class="tab" onclick="showGroup('reddit')">Reddit</button>
                <button class="tab" onclick="showGroup('youtube')">YouTube</button>
                <button class="tab" onclick="showGroup('mixed')">Mixed</button>
                <button class="tab" onclick="showGroup('allsources')">All Sources</button>
            </div>
"""

    # Generate strategy cards for each group
    group_labels = {
        "twitter": ("Twitter/X Strategies", "badge-twitter", "X"),
        "reddit": ("Reddit Strategies", "badge-reddit", "R"),
        "youtube": ("YouTube Strategies", "badge-youtube", "Y"),
        "mixed": ("Mixed Strategies (Twitter + Reddit)", "badge-mixed", "+"),
        "all": ("All Sources Strategies", "badge-all", "*"),
    }

    for group_id, strategy_list in STRATEGY_GROUPS.items():
        label, badge_class, icon = group_labels[group_id]
        data_group = "allsources" if group_id == "all" else group_id

        html += f'''
            <div class="strategy-section" data-group="{data_group}">
                <div class="section-header">
                    <h2>{label}</h2>
                    <span class="badge {badge_class}">{icon}</span>
                </div>
                <div class="strategies-grid">
        '''

        for strategy_name in strategy_list:
            if strategy_name in stats:
                data = stats[strategy_name]
                is_leader = strategy_name == leader
                color = STRATEGY_COLORS.get(strategy_name, "#ffffff")
                return_class = "positive" if data["total_return"] >= 0 else "negative"

                html += f'''
                    <div class="strategy-card {'leader' if is_leader else ''}" data-strategy="{strategy_name}">
                        <div class="strategy-card-header">
                            <span class="strategy-name" style="color: {color}">{strategy_name.replace("_", " ").title()}</span>
                            {'<span class="leader-badge">Leader</span>' if is_leader else ''}
                        </div>
                        <div class="strategy-card-body">
                            <div class="strategy-metrics">
                                <div class="metric">
                                    <div class="value">${data["balance"]:,.0f}</div>
                                    <div class="label">Balance</div>
                                </div>
                                <div class="metric">
                                    <div class="value {return_class}">{data["total_return"]:+.1f}%</div>
                                    <div class="label">Return</div>
                                </div>
                                <div class="metric">
                                    <div class="value">{data["win_rate"]:.0f}%</div>
                                    <div class="label">Win Rate</div>
                                </div>
                            </div>
                            <div class="strategy-metrics">
                                <div class="metric">
                                    <div class="value">{data["total_trades"]}</div>
                                    <div class="label">Trades</div>
                                </div>
                                <div class="metric">
                                    <div class="value positive">{data["avg_win"]:+.1f}%</div>
                                    <div class="label">Avg Win</div>
                                </div>
                                <div class="metric">
                                    <div class="value negative">{data["max_drawdown"]:.1f}%</div>
                                    <div class="label">Max DD</div>
                                </div>
                            </div>
                            <div class="mini-chart">
                                <canvas id="chart_{strategy_name}"></canvas>
                            </div>
                            <div class="positions-list">
                                <div class="positions-header">Open Positions ({len(data["open_positions"])})</div>
                '''

                if data["open_positions"]:
                    for trade in data["open_positions"][:3]:
                        dir_class = "long" if trade.direction == Direction.LONG else "short"
                        html += f'''
                                <div class="position">
                                    <div class="position-info">
                                        <span class="direction-badge {dir_class}">{trade.direction.value}</span>
                                        <span>{trade.asset.value}</span>
                                    </div>
                                    <span style="color: var(--text-muted)">@ ${trade.entry_price:,.2f}</span>
                                </div>
                        '''
                else:
                    html += '<div class="no-positions">No open positions</div>'

                html += '''
                            </div>
                        </div>
                    </div>
                '''
            else:
                # Empty card for missing strategy
                html += f'''
                    <div class="strategy-card" data-strategy="{strategy_name}">
                        <div class="strategy-card-header">
                            <span class="strategy-name">{strategy_name.replace("_", " ").title()}</span>
                        </div>
                        <div class="strategy-card-body">
                            <div class="no-positions" style="padding: 40px;">No data yet</div>
                        </div>
                    </div>
                '''

        html += '''
                </div>
            </div>
        '''

    # Main equity chart
    html += f'''
            <!-- Main Chart -->
            <div class="chart-section">
                <div class="chart-header">
                    <h2 class="chart-title">Equity Curves Comparison</h2>
                </div>
                <div class="chart-container">
                    <canvas id="mainChart"></canvas>
                </div>
            </div>

            <!-- Top Creators -->
            <div class="section-header">
                <h2>Top Creators by Source</h2>
            </div>
            <div class="creators-grid">
    '''

    # Creator leaderboards
    for source, source_key, icon in [("Twitter", "top_twitter", "X"), ("Reddit", "top_reddit", "R"), ("YouTube", "top_youtube", "Y")]:
        creators = creator_stats.get(source_key, [])[:5]
        html += f'''
                <div class="creator-card">
                    <div class="creator-card-header">
                        <span>{icon}</span>
                        Top {source} Creators
                    </div>
                    <div class="creator-list">
        '''

        for i, c in enumerate(creators):
            rank_class = "gold" if i == 0 else ("silver" if i == 1 else ("bronze" if i == 2 else ""))
            html += f'''
                        <div class="creator-item">
                            <div class="creator-rank {rank_class}">{i+1}</div>
                            <div class="creator-info">
                                <div class="creator-name">{c.get("username", "N/A")}</div>
                                <div class="creator-stats">{c.get("predictions", 0)} predictions</div>
                            </div>
                            <div class="creator-rating">
                                <div class="rating">{c.get("rating", 1500):.0f}</div>
                                <div class="accuracy">{c.get("accuracy", 0):.0f}%</div>
                            </div>
                        </div>
            '''

        if not creators:
            html += '<div class="no-positions">No creators yet</div>'

        html += '''
                    </div>
                </div>
        '''

    # Recent trades table
    all_trades = []
    for name, data in stats.items():
        for trade in data.get("recent_trades", []):
            all_trades.append((name, trade))
    all_trades.sort(key=lambda x: x[1].exit_time or datetime.min, reverse=True)

    html += f'''
            </div>

            <!-- Recent Trades -->
            <div class="trades-section" style="margin-top: 24px;">
                <div class="trades-header">
                    <h2 class="trades-title">Recent Trades</h2>
                    <span style="color: var(--text-muted);">Last 20 trades across all strategies</span>
                </div>
                <table class="trades-table">
                    <thead>
                        <tr>
                            <th>Strategy</th>
                            <th>Asset</th>
                            <th>Direction</th>
                            <th>Entry</th>
                            <th>Exit</th>
                            <th>P&L</th>
                            <th>Time</th>
                        </tr>
                    </thead>
                    <tbody>
    '''

    for strategy, trade in all_trades[:20]:
        pnl_class = "positive" if (trade.pnl_percent or 0) > 0 else "negative"
        dir_class = "long" if trade.direction == Direction.LONG else "short"
        color = STRATEGY_COLORS.get(strategy, "#ffffff")

        html += f'''
                        <tr>
                            <td><span style="border-left: 3px solid {color}; padding-left: 8px;">{strategy.replace("_", " ").title()}</span></td>
                            <td>{trade.asset.value}</td>
                            <td><span class="direction-badge {dir_class}">{trade.direction.value}</span></td>
                            <td>${trade.entry_price:,.2f}</td>
                            <td>${trade.exit_price:,.2f if trade.exit_price else 0}</td>
                            <td><span class="trade-pnl {pnl_class}">{trade.pnl_percent:+.2f}%</span></td>
                            <td style="color: var(--text-muted);">{trade.exit_time.strftime("%m/%d %H:%M") if trade.exit_time else "-"}</td>
                        </tr>
        '''

    if not all_trades:
        html += '<tr><td colspan="7" style="text-align: center; padding: 40px; color: var(--text-muted);">No trades yet</td></tr>'

    html += f'''
                    </tbody>
                </table>
            </div>

            <!-- Footer -->
            <footer class="footer">
                <p>Last updated: {datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")} UTC | Auto-refresh: 30s</p>
                <p style="margin-top: 4px;">Quantum Trading Dashboard v2.0 | 15 Strategies | Multi-Source Signals</p>
            </footer>
        </main>
    </div>

    <script>
        const equityData = {json.dumps(equity_data)};
        const colors = {json.dumps(STRATEGY_COLORS)};

        Chart.defaults.color = '#9ca3af';
        Chart.defaults.borderColor = '#252532';
        Chart.defaults.font.family = "'Inter', sans-serif";

        // Main equity chart
        const mainCtx = document.getElementById('mainChart');
        if (mainCtx) {{
            const datasets = [];
            for (const [strategy, data] of Object.entries(equityData)) {{
                if (data && data.length > 0) {{
                    datasets.push({{
                        label: strategy.replace(/_/g, ' ').toUpperCase(),
                        data: data.map(d => ({{x: new Date(d.time), y: d.balance}})),
                        borderColor: colors[strategy] || '#ffffff',
                        backgroundColor: (colors[strategy] || '#ffffff') + '10',
                        fill: false,
                        tension: 0.4,
                        pointRadius: 0,
                        pointHoverRadius: 4,
                        borderWidth: 2,
                    }});
                }}
            }}

            if (datasets.length > 0) {{
                new Chart(mainCtx, {{
                    type: 'line',
                    data: {{ datasets }},
                    options: {{
                        responsive: true,
                        maintainAspectRatio: false,
                        interaction: {{ intersect: false, mode: 'index' }},
                        plugins: {{
                            legend: {{
                                position: 'top',
                                labels: {{
                                    usePointStyle: true,
                                    pointStyle: 'circle',
                                    padding: 15,
                                    font: {{ size: 10, weight: '500' }}
                                }}
                            }},
                            tooltip: {{
                                backgroundColor: '#16161f',
                                borderColor: '#252532',
                                borderWidth: 1,
                                padding: 10,
                                titleFont: {{ size: 12, weight: '600' }},
                                bodyFont: {{ size: 11 }},
                                callbacks: {{
                                    label: ctx => `${{ctx.dataset.label}}: $${{ctx.parsed.y.toLocaleString()}}`
                                }}
                            }}
                        }},
                        scales: {{
                            x: {{
                                type: 'time',
                                grid: {{ color: 'rgba(255,255,255,0.03)' }},
                                ticks: {{ maxTicksLimit: 8 }}
                            }},
                            y: {{
                                grid: {{ color: 'rgba(255,255,255,0.03)' }},
                                ticks: {{ callback: v => '$' + v.toLocaleString() }}
                            }}
                        }}
                    }}
                }});
            }}
        }}

        // Mini charts
        for (const [strategy, data] of Object.entries(equityData)) {{
            const ctx = document.getElementById('chart_' + strategy);
            if (ctx && data && data.length > 0) {{
                new Chart(ctx, {{
                    type: 'line',
                    data: {{
                        datasets: [{{
                            data: data.map(d => ({{x: new Date(d.time), y: d.balance}})),
                            borderColor: colors[strategy] || '#ffffff',
                            backgroundColor: (colors[strategy] || '#ffffff') + '20',
                            fill: true,
                            tension: 0.4,
                            pointRadius: 0,
                            borderWidth: 2,
                        }}]
                    }},
                    options: {{
                        responsive: true,
                        maintainAspectRatio: false,
                        plugins: {{ legend: {{ display: false }} }},
                        scales: {{
                            x: {{ display: false, type: 'time' }},
                            y: {{ display: false }}
                        }}
                    }}
                }});
            }} else if (ctx) {{
                ctx.parentElement.innerHTML = '<div style="display:flex;align-items:center;justify-content:center;height:100%;color:var(--text-muted);font-size:0.75rem;">No data</div>';
            }}
        }}

        // Tab switching
        function showGroup(group) {{
            document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
            event.target.classList.add('active');

            document.querySelectorAll('.strategy-section').forEach(section => {{
                if (group === 'all') {{
                    section.style.display = 'block';
                }} else {{
                    section.style.display = section.dataset.group === group ? 'block' : 'none';
                }}
            }});
        }}
    </script>
</body>
</html>
'''

    return html


# API Endpoints
@app.get("/api/stats")
async def api_stats():
    """Get all strategy stats as JSON."""
    session = get_session()
    try:
        result = {}
        for s in ALL_STRATEGIES:
            data = get_strategy_stats(session, s)
            if data:
                result[s] = {
                    "balance": data["balance"],
                    "total_return": data["total_return"],
                    "total_pnl": data["total_pnl"],
                    "total_trades": data["total_trades"],
                    "win_rate": data["win_rate"],
                    "max_drawdown": data["max_drawdown"],
                    "sharpe_ratio": data["sharpe_ratio"],
                    "profit_factor": data["profit_factor"],
                    "open_positions": len(data["open_positions"]),
                }
        return result
    finally:
        session.close()


@app.get("/api/signals")
async def api_signals():
    """Get signal analytics."""
    session = get_session()
    try:
        return get_signal_stats(session)
    finally:
        session.close()


@app.get("/api/creators")
async def api_creators():
    """Get creator leaderboard."""
    session = get_session()
    try:
        return get_creator_stats(session)
    finally:
        session.close()


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {
        "status": "ok",
        "timestamp": datetime.utcnow().isoformat(),
        "strategies": len(ALL_STRATEGIES),
    }


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8080))
    uvicorn.run(app, host="0.0.0.0", port=port)
