"""
Premium Trading Dashboard

World-class design with animations, real-time updates, and professional analytics.
"""

import os
import json
from datetime import datetime, timedelta
from pathlib import Path
from collections import defaultdict

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from sqlalchemy import create_engine, desc, func
from sqlalchemy.orm import sessionmaker

from src.models import Base, Trade, TradeStatus, Portfolio, Asset, Direction, Signal, Creator

app = FastAPI(title="Crypto Trading Dashboard")

DATA_DIR = Path(os.getenv("DATA_DIR", "data"))
DB_PATH = DATA_DIR / "strategies.db"


def get_session():
    engine = create_engine(f"sqlite:///{DB_PATH}")
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    return Session()


def get_strategy_stats(session, strategy_name: str) -> dict:
    portfolio = session.query(Portfolio).filter(
        Portfolio.name == f"strategy_{strategy_name}"
    ).first()

    if not portfolio:
        return None

    open_trades = session.query(Trade).filter(
        Trade.status == TradeStatus.OPEN,
        Trade.notes == f"strategy_{strategy_name}",
    ).all()

    all_trades = session.query(Trade).filter(
        Trade.status == TradeStatus.CLOSED,
        Trade.notes == f"strategy_{strategy_name}",
    ).order_by(Trade.exit_time).all()

    # Equity curve
    equity_curve = []
    running_balance = portfolio.initial_balance
    for trade in all_trades:
        if trade.pnl_usd and trade.exit_time:
            running_balance += trade.pnl_usd
            equity_curve.append({
                "time": trade.exit_time.isoformat(),
                "balance": running_balance,
                "pnl": trade.pnl_usd,
            })

    total_return = ((portfolio.current_balance - portfolio.initial_balance) / portfolio.initial_balance) * 100
    win_rate = (portfolio.winning_trades / portfolio.total_trades * 100) if portfolio.total_trades > 0 else 0

    # Max drawdown
    peak = portfolio.initial_balance
    max_dd = 0
    for point in equity_curve:
        if point["balance"] > peak:
            peak = point["balance"]
        dd = (peak - point["balance"]) / peak * 100
        if dd > max_dd:
            max_dd = dd

    # PnL by asset
    pnl_by_asset = defaultdict(float)
    for trade in all_trades:
        if trade.pnl_usd:
            pnl_by_asset[trade.asset.value] += trade.pnl_usd

    recent_trades = all_trades[-20:] if all_trades else []
    recent_trades.reverse()

    wins = [t.pnl_percent for t in all_trades if t.pnl_percent and t.pnl_percent > 0]
    losses = [t.pnl_percent for t in all_trades if t.pnl_percent and t.pnl_percent < 0]
    avg_win = sum(wins) / len(wins) if wins else 0
    avg_loss = sum(losses) / len(losses) if losses else 0

    # Daily PnL for chart
    daily_pnl = defaultdict(float)
    for trade in all_trades:
        if trade.exit_time and trade.pnl_usd:
            day = trade.exit_time.strftime("%Y-%m-%d")
            daily_pnl[day] += trade.pnl_usd

    return {
        "name": strategy_name,
        "balance": portfolio.current_balance,
        "initial": portfolio.initial_balance,
        "total_return": total_return,
        "total_pnl": portfolio.total_pnl,
        "total_trades": portfolio.total_trades,
        "winning_trades": portfolio.winning_trades,
        "win_rate": win_rate,
        "max_drawdown": max_dd,
        "avg_win": avg_win,
        "avg_loss": avg_loss,
        "open_positions": open_trades,
        "recent_trades": recent_trades,
        "equity_curve": equity_curve,
        "pnl_by_asset": dict(pnl_by_asset),
        "daily_pnl": dict(daily_pnl),
    }


def get_system_stats(session) -> dict:
    """Get overall system statistics."""
    total_signals = session.query(Signal).count()
    total_creators = session.query(Creator).count()

    # Recent signals
    recent_signals = session.query(Signal).order_by(
        desc(Signal.posted_at)
    ).limit(10).all()

    # Top creators by accuracy
    top_creators = session.query(Creator).filter(
        Creator.total_predictions >= 5
    ).order_by(desc(Creator.correct_predictions / Creator.total_predictions)).limit(5).all()

    return {
        "total_signals": total_signals,
        "total_creators": total_creators,
        "recent_signals": recent_signals,
        "top_creators": top_creators,
    }


@app.get("/", response_class=HTMLResponse)
async def dashboard():
    session = get_session()

    strategies = ["social_pure", "technical_strict", "balanced"]
    stats = {}
    for s in strategies:
        data = get_strategy_stats(session, s)
        if data:
            stats[s] = data

    system_stats = get_system_stats(session)

    # Find leader
    leader = None
    best_return = -999
    for name, data in stats.items():
        if data["total_return"] > best_return:
            best_return = data["total_return"]
            leader = name

    session.close()

    # Prepare chart data
    equity_data = {name: data["equity_curve"] for name, data in stats.items()}
    daily_pnl_data = {name: data["daily_pnl"] for name, data in stats.items()}

    all_pnl_by_asset = defaultdict(lambda: defaultdict(float))
    for name, data in stats.items():
        for asset, pnl in data["pnl_by_asset"].items():
            all_pnl_by_asset[asset][name] = pnl

    # Calculate totals
    total_balance = sum(d["balance"] for d in stats.values()) if stats else 30000
    total_pnl = sum(d["total_pnl"] for d in stats.values()) if stats else 0
    total_trades = sum(d["total_trades"] for d in stats.values()) if stats else 0
    avg_win_rate = sum(d["win_rate"] for d in stats.values()) / len(stats) if stats else 0

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
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
    <style>
        :root {{
            --bg-darker: #050508;
            --bg-dark: #0a0a0f;
            --bg-card: #0f0f18;
            --bg-card-hover: #151520;
            --border: #1a1a2e;
            --border-light: #252540;
            --text-primary: #ffffff;
            --text-secondary: #a0a0b0;
            --text-muted: #606070;
            --accent-blue: #3b82f6;
            --accent-purple: #8b5cf6;
            --accent-cyan: #06b6d4;
            --accent-pink: #ec4899;
            --green: #10b981;
            --green-glow: rgba(16, 185, 129, 0.3);
            --red: #ef4444;
            --red-glow: rgba(239, 68, 68, 0.3);
            --yellow: #f59e0b;
            --gradient-1: linear-gradient(135deg, #3b82f6 0%, #8b5cf6 100%);
            --gradient-2: linear-gradient(135deg, #06b6d4 0%, #3b82f6 100%);
            --gradient-3: linear-gradient(135deg, #8b5cf6 0%, #ec4899 100%);
            --glass: rgba(255, 255, 255, 0.03);
            --glass-border: rgba(255, 255, 255, 0.08);
        }}

        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}

        body {{
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
            background: var(--bg-darker);
            color: var(--text-primary);
            min-height: 100vh;
            overflow-x: hidden;
        }}

        /* Animated background */
        .bg-animation {{
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            pointer-events: none;
            z-index: 0;
            overflow: hidden;
        }}

        .bg-animation::before {{
            content: '';
            position: absolute;
            top: -50%;
            left: -50%;
            width: 200%;
            height: 200%;
            background: radial-gradient(circle at 30% 30%, rgba(59, 130, 246, 0.08) 0%, transparent 50%),
                        radial-gradient(circle at 70% 70%, rgba(139, 92, 246, 0.06) 0%, transparent 50%),
                        radial-gradient(circle at 50% 50%, rgba(6, 182, 212, 0.04) 0%, transparent 60%);
            animation: bgPulse 20s ease-in-out infinite;
        }}

        @keyframes bgPulse {{
            0%, 100% {{ transform: translate(0, 0) rotate(0deg); }}
            33% {{ transform: translate(2%, 2%) rotate(1deg); }}
            66% {{ transform: translate(-1%, 1%) rotate(-1deg); }}
        }}

        /* Grid pattern overlay */
        .grid-overlay {{
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background-image:
                linear-gradient(rgba(255,255,255,0.02) 1px, transparent 1px),
                linear-gradient(90deg, rgba(255,255,255,0.02) 1px, transparent 1px);
            background-size: 50px 50px;
            pointer-events: none;
            z-index: 0;
        }}

        .container {{
            position: relative;
            z-index: 1;
            max-width: 1800px;
            margin: 0 auto;
            padding: 20px;
        }}

        /* Header */
        .header {{
            text-align: center;
            padding: 40px 20px;
            margin-bottom: 30px;
        }}

        .logo {{
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 15px;
            margin-bottom: 15px;
        }}

        .logo-icon {{
            width: 50px;
            height: 50px;
            background: var(--gradient-1);
            border-radius: 12px;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 24px;
            animation: pulse 2s ease-in-out infinite;
        }}

        @keyframes pulse {{
            0%, 100% {{ box-shadow: 0 0 20px rgba(59, 130, 246, 0.4); }}
            50% {{ box-shadow: 0 0 40px rgba(139, 92, 246, 0.6); }}
        }}

        .logo h1 {{
            font-size: 2.5rem;
            font-weight: 800;
            background: var(--gradient-1);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            letter-spacing: -1px;
        }}

        .subtitle {{
            color: var(--text-secondary);
            font-size: 0.95rem;
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 20px;
        }}

        .live-indicator {{
            display: flex;
            align-items: center;
            gap: 8px;
            color: var(--green);
            font-weight: 500;
        }}

        .live-dot {{
            width: 8px;
            height: 8px;
            background: var(--green);
            border-radius: 50%;
            animation: livePulse 1.5s ease-in-out infinite;
        }}

        @keyframes livePulse {{
            0%, 100% {{ opacity: 1; box-shadow: 0 0 0 0 var(--green-glow); }}
            50% {{ opacity: 0.5; box-shadow: 0 0 0 8px transparent; }}
        }}

        /* Stats Overview */
        .stats-overview {{
            display: grid;
            grid-template-columns: repeat(4, 1fr);
            gap: 20px;
            margin-bottom: 30px;
        }}

        .stat-card {{
            background: var(--glass);
            backdrop-filter: blur(20px);
            border: 1px solid var(--glass-border);
            border-radius: 20px;
            padding: 25px;
            position: relative;
            overflow: hidden;
            transition: all 0.3s ease;
        }}

        .stat-card::before {{
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            height: 3px;
            background: var(--gradient-1);
            opacity: 0;
            transition: opacity 0.3s ease;
        }}

        .stat-card:hover {{
            transform: translateY(-5px);
            border-color: var(--border-light);
        }}

        .stat-card:hover::before {{
            opacity: 1;
        }}

        .stat-card:nth-child(2)::before {{ background: var(--gradient-2); }}
        .stat-card:nth-child(3)::before {{ background: var(--gradient-3); }}
        .stat-card:nth-child(4)::before {{ background: linear-gradient(135deg, var(--green) 0%, var(--cyan) 100%); }}

        .stat-label {{
            font-size: 0.8rem;
            color: var(--text-muted);
            text-transform: uppercase;
            letter-spacing: 1px;
            margin-bottom: 10px;
        }}

        .stat-value {{
            font-size: 2.2rem;
            font-weight: 700;
            font-family: 'JetBrains Mono', monospace;
            margin-bottom: 5px;
        }}

        .stat-value.positive {{ color: var(--green); }}
        .stat-value.negative {{ color: var(--red); }}

        .stat-change {{
            font-size: 0.85rem;
            color: var(--text-secondary);
        }}

        /* Leader Banner */
        .leader-banner {{
            background: linear-gradient(135deg, rgba(16, 185, 129, 0.15) 0%, rgba(6, 182, 212, 0.15) 100%);
            border: 1px solid rgba(16, 185, 129, 0.3);
            border-radius: 16px;
            padding: 20px 30px;
            display: flex;
            align-items: center;
            justify-content: space-between;
            margin-bottom: 30px;
            animation: glowPulse 3s ease-in-out infinite;
        }}

        @keyframes glowPulse {{
            0%, 100% {{ box-shadow: 0 0 30px rgba(16, 185, 129, 0.1); }}
            50% {{ box-shadow: 0 0 50px rgba(16, 185, 129, 0.2); }}
        }}

        .leader-info {{
            display: flex;
            align-items: center;
            gap: 20px;
        }}

        .leader-icon {{
            font-size: 2.5rem;
        }}

        .leader-text h3 {{
            font-size: 1.1rem;
            color: var(--text-secondary);
            font-weight: 500;
        }}

        .leader-text h2 {{
            font-size: 1.8rem;
            font-weight: 700;
            background: linear-gradient(90deg, var(--green), var(--accent-cyan));
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }}

        .leader-return {{
            text-align: right;
        }}

        .leader-return .value {{
            font-size: 2.5rem;
            font-weight: 800;
            color: var(--green);
            font-family: 'JetBrains Mono', monospace;
        }}

        /* Strategy Cards */
        .strategies-grid {{
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 20px;
            margin-bottom: 30px;
        }}

        .strategy-card {{
            background: var(--bg-card);
            border: 1px solid var(--border);
            border-radius: 20px;
            overflow: hidden;
            transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
            position: relative;
        }}

        .strategy-card:hover {{
            transform: translateY(-8px);
            border-color: var(--border-light);
            box-shadow: 0 25px 50px -12px rgba(0, 0, 0, 0.5);
        }}

        .strategy-header {{
            padding: 25px;
            border-bottom: 1px solid var(--border);
            display: flex;
            justify-content: space-between;
            align-items: center;
        }}

        .strategy-name {{
            font-size: 1.2rem;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 1px;
        }}

        .strategy-name.social {{ color: var(--green); }}
        .strategy-name.technical {{ color: var(--accent-blue); }}
        .strategy-name.balanced {{ color: var(--accent-purple); }}

        .strategy-badge {{
            padding: 6px 14px;
            border-radius: 20px;
            font-size: 0.75rem;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }}

        .badge-leader {{
            background: linear-gradient(135deg, var(--green), var(--accent-cyan));
            color: #000;
            animation: badgePulse 2s ease-in-out infinite;
        }}

        @keyframes badgePulse {{
            0%, 100% {{ transform: scale(1); }}
            50% {{ transform: scale(1.05); }}
        }}

        .strategy-body {{
            padding: 25px;
        }}

        .strategy-stats {{
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 20px;
            margin-bottom: 25px;
        }}

        .strategy-stat {{
            text-align: center;
            padding: 15px;
            background: var(--bg-darker);
            border-radius: 12px;
            transition: all 0.3s ease;
        }}

        .strategy-stat:hover {{
            background: var(--bg-card-hover);
            transform: scale(1.02);
        }}

        .strategy-stat-value {{
            font-size: 1.5rem;
            font-weight: 700;
            font-family: 'JetBrains Mono', monospace;
            margin-bottom: 4px;
        }}

        .strategy-stat-label {{
            font-size: 0.7rem;
            color: var(--text-muted);
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }}

        .mini-chart {{
            height: 120px;
            margin-bottom: 20px;
        }}

        .positions-section {{
            margin-top: 20px;
        }}

        .positions-header {{
            font-size: 0.8rem;
            color: var(--text-muted);
            text-transform: uppercase;
            letter-spacing: 1px;
            margin-bottom: 12px;
        }}

        .position-item {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 12px 16px;
            background: var(--bg-darker);
            border-radius: 10px;
            margin-bottom: 8px;
            transition: all 0.2s ease;
        }}

        .position-item:hover {{
            background: var(--bg-card-hover);
        }}

        .position-asset {{
            display: flex;
            align-items: center;
            gap: 10px;
        }}

        .position-direction {{
            padding: 4px 10px;
            border-radius: 6px;
            font-size: 0.7rem;
            font-weight: 700;
            text-transform: uppercase;
        }}

        .position-direction.long {{
            background: rgba(16, 185, 129, 0.2);
            color: var(--green);
        }}

        .position-direction.short {{
            background: rgba(239, 68, 68, 0.2);
            color: var(--red);
        }}

        .no-positions {{
            text-align: center;
            padding: 20px;
            color: var(--text-muted);
            font-size: 0.9rem;
        }}

        /* Main Chart Section */
        .chart-section {{
            background: var(--bg-card);
            border: 1px solid var(--border);
            border-radius: 20px;
            padding: 30px;
            margin-bottom: 30px;
        }}

        .chart-header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 25px;
        }}

        .chart-title {{
            font-size: 1.3rem;
            font-weight: 700;
        }}

        .chart-tabs {{
            display: flex;
            gap: 10px;
        }}

        .chart-tab {{
            padding: 8px 16px;
            background: var(--bg-darker);
            border: 1px solid var(--border);
            border-radius: 8px;
            color: var(--text-secondary);
            cursor: pointer;
            transition: all 0.2s ease;
            font-size: 0.85rem;
        }}

        .chart-tab:hover, .chart-tab.active {{
            background: var(--accent-blue);
            border-color: var(--accent-blue);
            color: #fff;
        }}

        .main-chart {{
            height: 400px;
            position: relative;
        }}

        /* Trades Table */
        .trades-section {{
            background: var(--bg-card);
            border: 1px solid var(--border);
            border-radius: 20px;
            overflow: hidden;
        }}

        .trades-header {{
            padding: 25px 30px;
            border-bottom: 1px solid var(--border);
            display: flex;
            justify-content: space-between;
            align-items: center;
        }}

        .trades-title {{
            font-size: 1.3rem;
            font-weight: 700;
        }}

        .trades-count {{
            background: var(--bg-darker);
            padding: 6px 14px;
            border-radius: 20px;
            font-size: 0.85rem;
            color: var(--text-secondary);
        }}

        .trades-table {{
            width: 100%;
            border-collapse: collapse;
        }}

        .trades-table th {{
            padding: 15px 20px;
            text-align: left;
            font-size: 0.75rem;
            color: var(--text-muted);
            text-transform: uppercase;
            letter-spacing: 1px;
            background: var(--bg-darker);
            font-weight: 500;
        }}

        .trades-table td {{
            padding: 18px 20px;
            border-bottom: 1px solid var(--border);
            font-size: 0.9rem;
        }}

        .trades-table tr {{
            transition: background 0.2s ease;
        }}

        .trades-table tr:hover td {{
            background: var(--bg-card-hover);
        }}

        .trade-asset {{
            font-weight: 600;
            display: flex;
            align-items: center;
            gap: 10px;
        }}

        .trade-direction {{
            padding: 4px 10px;
            border-radius: 6px;
            font-size: 0.75rem;
            font-weight: 600;
        }}

        .trade-direction.long {{ background: rgba(16, 185, 129, 0.15); color: var(--green); }}
        .trade-direction.short {{ background: rgba(239, 68, 68, 0.15); color: var(--red); }}

        .trade-pnl {{
            font-weight: 600;
            font-family: 'JetBrains Mono', monospace;
        }}

        .trade-pnl.positive {{ color: var(--green); }}
        .trade-pnl.negative {{ color: var(--red); }}

        .trade-strategy {{
            padding: 4px 10px;
            border-radius: 6px;
            font-size: 0.75rem;
            background: var(--bg-darker);
        }}

        /* Footer */
        .footer {{
            text-align: center;
            padding: 30px;
            color: var(--text-muted);
            font-size: 0.85rem;
            border-top: 1px solid var(--border);
            margin-top: 30px;
        }}

        .footer-time {{
            font-family: 'JetBrains Mono', monospace;
            color: var(--text-secondary);
        }}

        /* Responsive */
        @media (max-width: 1200px) {{
            .stats-overview {{ grid-template-columns: repeat(2, 1fr); }}
            .strategies-grid {{ grid-template-columns: 1fr; }}
        }}

        @media (max-width: 768px) {{
            .stats-overview {{ grid-template-columns: 1fr; }}
            .leader-banner {{ flex-direction: column; gap: 20px; text-align: center; }}
            .leader-return {{ text-align: center; }}
        }}

        /* Animations */
        .fade-in {{
            animation: fadeIn 0.6s ease-out forwards;
        }}

        @keyframes fadeIn {{
            from {{ opacity: 0; transform: translateY(20px); }}
            to {{ opacity: 1; transform: translateY(0); }}
        }}

        .slide-up {{
            animation: slideUp 0.5s ease-out forwards;
        }}

        @keyframes slideUp {{
            from {{ opacity: 0; transform: translateY(30px); }}
            to {{ opacity: 1; transform: translateY(0); }}
        }}

        /* Number animation */
        .animate-number {{
            transition: all 0.3s ease;
        }}
    </style>
</head>
<body>
    <div class="bg-animation"></div>
    <div class="grid-overlay"></div>

    <div class="container">
        <!-- Header -->
        <header class="header fade-in">
            <div class="logo">
                <div class="logo-icon">⚡</div>
                <h1>Quantum Trading</h1>
            </div>
            <p class="subtitle">
                <span class="live-indicator">
                    <span class="live-dot"></span>
                    Live Trading
                </span>
                <span>•</span>
                <span>Multi-Strategy A/B Testing</span>
                <span>•</span>
                <span>Auto-refresh 30s</span>
            </p>
        </header>

        <!-- Stats Overview -->
        <div class="stats-overview fade-in" style="animation-delay: 0.1s;">
            <div class="stat-card">
                <div class="stat-label">Total Portfolio Value</div>
                <div class="stat-value">${total_balance:,.2f}</div>
                <div class="stat-change">Combined across all strategies</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">Total P&L</div>
                <div class="stat-value {"positive" if total_pnl >= 0 else "negative"}">${total_pnl:+,.2f}</div>
                <div class="stat-change">Since inception</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">Total Trades</div>
                <div class="stat-value">{total_trades}</div>
                <div class="stat-change">Across all strategies</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">Average Win Rate</div>
                <div class="stat-value {"positive" if avg_win_rate >= 50 else "negative"}">{avg_win_rate:.1f}%</div>
                <div class="stat-change">Combined performance</div>
            </div>
        </div>

        <!-- Leader Banner -->
        {f'''<div class="leader-banner fade-in" style="animation-delay: 0.2s;">
            <div class="leader-info">
                <div class="leader-icon">🏆</div>
                <div class="leader-text">
                    <h3>Current Leader</h3>
                    <h2>{leader.replace("_", " ").upper()}</h2>
                </div>
            </div>
            <div class="leader-return">
                <div class="value">{best_return:+.2f}%</div>
            </div>
        </div>''' if leader and stats else '<div class="leader-banner fade-in"><div class="leader-info"><div class="leader-icon">⏳</div><div class="leader-text"><h3>Waiting</h3><h2>No trades yet</h2></div></div></div>'}

        <!-- Strategy Cards -->
        <div class="strategies-grid">
"""

    colors = {
        "social_pure": ("social", "#10b981"),
        "technical_strict": ("technical", "#3b82f6"),
        "balanced": ("balanced", "#8b5cf6"),
    }

    for idx, (name, data) in enumerate(stats.items()):
        is_leader = name == leader
        strategy_class, color = colors.get(name, ("", "#ffffff"))
        return_class = "positive" if data["total_return"] >= 0 else "negative"

        html += f"""
            <div class="strategy-card fade-in" style="animation-delay: {0.3 + idx * 0.1}s;">
                <div class="strategy-header">
                    <span class="strategy-name {strategy_class}">{name.replace("_", " ")}</span>
                    {'<span class="strategy-badge badge-leader">👑 LEADER</span>' if is_leader else ''}
                </div>
                <div class="strategy-body">
                    <div class="strategy-stats">
                        <div class="strategy-stat">
                            <div class="strategy-stat-value">${data["balance"]:,.0f}</div>
                            <div class="strategy-stat-label">Balance</div>
                        </div>
                        <div class="strategy-stat">
                            <div class="strategy-stat-value {return_class}">{data["total_return"]:+.1f}%</div>
                            <div class="strategy-stat-label">Return</div>
                        </div>
                        <div class="strategy-stat">
                            <div class="strategy-stat-value">{data["win_rate"]:.0f}%</div>
                            <div class="strategy-stat-label">Win Rate</div>
                        </div>
                    </div>
                    <div class="strategy-stats">
                        <div class="strategy-stat">
                            <div class="strategy-stat-value">{data["total_trades"]}</div>
                            <div class="strategy-stat-label">Trades</div>
                        </div>
                        <div class="strategy-stat">
                            <div class="strategy-stat-value {"positive" if data["avg_win"] > 0 else ""}">{data["avg_win"]:+.1f}%</div>
                            <div class="strategy-stat-label">Avg Win</div>
                        </div>
                        <div class="strategy-stat">
                            <div class="strategy-stat-value negative">{data["max_drawdown"]:.1f}%</div>
                            <div class="strategy-stat-label">Max DD</div>
                        </div>
                    </div>
                    <div class="mini-chart">
                        <canvas id="chart_{name}"></canvas>
                    </div>
                    <div class="positions-section">
                        <div class="positions-header">Open Positions ({len(data["open_positions"])})</div>
        """

        if data["open_positions"]:
            for trade in data["open_positions"]:
                dir_class = "long" if trade.direction == Direction.LONG else "short"
                html += f'''
                        <div class="position-item">
                            <div class="position-asset">
                                <span class="position-direction {dir_class}">{trade.direction.value.upper()}</span>
                                <span>{trade.asset.value}</span>
                            </div>
                            <span style="color: var(--text-secondary);">@ ${trade.entry_price:,.2f}</span>
                        </div>
                '''
        else:
            html += '<div class="no-positions">No open positions</div>'

        html += """
                    </div>
                </div>
            </div>
        """

    html += """
        </div>

        <!-- Main Chart -->
        <div class="chart-section fade-in" style="animation-delay: 0.6s;">
            <div class="chart-header">
                <h2 class="chart-title">Equity Curves Comparison</h2>
            </div>
            <div class="main-chart">
                <canvas id="mainChart"></canvas>
            </div>
        </div>

        <!-- Trades Table -->
        <div class="trades-section fade-in" style="animation-delay: 0.7s;">
            <div class="trades-header">
                <h2 class="trades-title">Recent Trades</h2>
                <span class="trades-count">Last 20 trades</span>
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
    """

    all_trades = []
    for name, data in stats.items():
        for trade in data["recent_trades"]:
            all_trades.append((name, trade))
    all_trades.sort(key=lambda x: x[1].exit_time or datetime.min, reverse=True)

    for strategy, trade in all_trades[:20]:
        pnl_class = "positive" if (trade.pnl_percent or 0) > 0 else "negative"
        dir_class = "long" if trade.direction == Direction.LONG else "short"
        strategy_color = colors.get(strategy, ("", "#ffffff"))[1]

        html += f"""
                    <tr>
                        <td><span class="trade-strategy" style="border-left: 3px solid {strategy_color};">{strategy.replace("_", " ").title()}</span></td>
                        <td><span class="trade-asset">{trade.asset.value}</span></td>
                        <td><span class="trade-direction {dir_class}">{trade.direction.value.upper()}</span></td>
                        <td>${trade.entry_price:,.2f}</td>
                        <td>${trade.exit_price:,.2f if trade.exit_price else 0}</td>
                        <td><span class="trade-pnl {pnl_class}">{trade.pnl_percent:+.2f}%</span></td>
                        <td style="color: var(--text-muted);">{trade.exit_time.strftime("%m/%d %H:%M") if trade.exit_time else "-"}</td>
                    </tr>
        """

    if not all_trades:
        html += '<tr><td colspan="7" style="text-align: center; padding: 40px; color: var(--text-muted);">No trades yet - waiting for signals...</td></tr>'

    html += f"""
                </tbody>
            </table>
        </div>

        <!-- Footer -->
        <footer class="footer">
            <p>Last updated: <span class="footer-time">{datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")} UTC</span></p>
            <p style="margin-top: 5px; opacity: 0.7;">Quantum Trading Dashboard • Multi-Strategy A/B Testing Platform</p>
        </footer>
    </div>

    <script>
        const colors = {{
            social_pure: '#10b981',
            technical_strict: '#3b82f6',
            balanced: '#8b5cf6'
        }};

        const equityData = {json.dumps(equity_data)};

        // Chart.js default config
        Chart.defaults.color = '#a0a0b0';
        Chart.defaults.borderColor = '#1a1a2e';
        Chart.defaults.font.family = "'Inter', sans-serif";

        // Main equity chart
        const mainCtx = document.getElementById('mainChart');
        if (mainCtx) {{
            const datasets = [];
            for (const [strategy, data] of Object.entries(equityData)) {{
                if (data.length > 0) {{
                    datasets.push({{
                        label: strategy.replace('_', ' ').toUpperCase(),
                        data: data.map(d => ({{x: new Date(d.time), y: d.balance}})),
                        borderColor: colors[strategy],
                        backgroundColor: colors[strategy] + '20',
                        fill: true,
                        tension: 0.4,
                        pointRadius: 0,
                        pointHoverRadius: 6,
                        borderWidth: 3,
                    }});
                }}
            }}

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
                                padding: 20,
                                font: {{ size: 12, weight: '500' }}
                            }}
                        }},
                        tooltip: {{
                            backgroundColor: '#0f0f18',
                            borderColor: '#1a1a2e',
                            borderWidth: 1,
                            padding: 12,
                            titleFont: {{ size: 14, weight: '600' }},
                            bodyFont: {{ size: 13 }},
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
                            ticks: {{
                                callback: v => '$' + v.toLocaleString()
                            }}
                        }}
                    }},
                    animation: {{
                        duration: 1000,
                        easing: 'easeOutQuart'
                    }}
                }}
            }});
        }}

        // Mini charts for each strategy
        for (const [strategy, data] of Object.entries(equityData)) {{
            const ctx = document.getElementById('chart_' + strategy);
            if (ctx && data.length > 0) {{
                new Chart(ctx, {{
                    type: 'line',
                    data: {{
                        datasets: [{{
                            data: data.map(d => ({{x: new Date(d.time), y: d.balance}})),
                            borderColor: colors[strategy],
                            backgroundColor: colors[strategy] + '10',
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
                        }},
                        animation: {{
                            duration: 1500,
                            easing: 'easeOutQuart'
                        }}
                    }}
                }});
            }} else if (ctx) {{
                // Show placeholder for empty chart
                ctx.parentElement.innerHTML = '<div style="display:flex;align-items:center;justify-content:center;height:100%;color:var(--text-muted);font-size:0.85rem;">No data yet</div>';
            }}
        }}
    </script>
</body>
</html>
"""

    return HTMLResponse(content=html)


@app.get("/api/stats")
async def api_stats():
    session = get_session()
    strategies = ["social_pure", "technical_strict", "balanced"]
    result = {}
    for s in strategies:
        data = get_strategy_stats(session, s)
        if data:
            result[s] = {
                "balance": data["balance"],
                "total_return": data["total_return"],
                "total_pnl": data["total_pnl"],
                "total_trades": data["total_trades"],
                "win_rate": data["win_rate"],
                "open_positions": len(data["open_positions"]),
            }
    session.close()
    return result


@app.get("/health")
async def health():
    return {"status": "ok", "timestamp": datetime.utcnow().isoformat()}


@app.get("/debug")
async def debug():
    """Debug endpoint to check database state."""
    session = get_session()
    try:
        portfolios = session.query(Portfolio).all()
        trades = session.query(Trade).count()
        signals = session.query(Signal).count()
        creators = session.query(Creator).count()

        return {
            "db_path": str(DB_PATH),
            "db_exists": DB_PATH.exists(),
            "portfolios": [{"name": p.name, "balance": p.current_balance, "trades": p.total_trades} for p in portfolios],
            "total_trades": trades,
            "total_signals": signals,
            "total_creators": creators,
        }
    finally:
        session.close()


@app.get("/test-api")
async def test_api():
    """Test the Macrocosmos API connection."""
    import traceback
    from src.config import get_settings
    from src.data_ingestion import XClient

    try:
        # Check raw env var first
        raw_key = os.getenv("MACROCOSMOS_API_KEY", "")
        settings = get_settings()
        api_key = settings.macrocosmos_api_key

        if not api_key and not raw_key:
            return {
                "error": "MACROCOSMOS_API_KEY not set",
                "key_present": False,
                "raw_env": bool(raw_key),
                "settings_key": bool(api_key),
                "all_env_keys": [k for k in os.environ.keys() if "MACRO" in k.upper() or "API" in k.upper()],
                "all_env_count": len(os.environ),
                "sample_env_keys": list(os.environ.keys())[:20]
            }

        # Use raw key if settings didn't pick it up
        if not api_key and raw_key:
            api_key = raw_key

        # Try to fetch a small sample
        x_client = XClient(settings)
        posts = x_client.fetch_trading_signals(limit=5, hours_back=24)

        return {
            "success": True,
            "key_present": True,
            "key_preview": api_key[:8] + "..." if len(api_key) > 8 else "***",
            "posts_fetched": len(posts),
            "sample": [{"username": p.username, "text": p.text[:100]} for p in posts[:2]] if posts else []
        }
    except Exception as e:
        return {
            "success": False,
            "key_present": bool(api_key) if 'api_key' in dir() else False,
            "error": str(e),
            "traceback": traceback.format_exc()
        }


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8080))
    uvicorn.run(app, host="0.0.0.0", port=port)
