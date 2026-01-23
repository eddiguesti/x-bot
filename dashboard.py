"""
Strategy Performance Dashboard

Beautiful web dashboard with charts to monitor multi-strategy A/B testing.
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

from src.models import Base, Trade, TradeStatus, Portfolio, Asset, Direction

app = FastAPI(title="Crypto Bot Dashboard")

# Database setup
DATA_DIR = Path(os.getenv("DATA_DIR", "data"))
DB_PATH = DATA_DIR / "strategies.db"


def get_session():
    """Get database session."""
    engine = create_engine(f"sqlite:///{DB_PATH}")
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    return Session()


def get_strategy_stats(session, strategy_name: str) -> dict:
    """Get stats for a specific strategy."""
    portfolio = session.query(Portfolio).filter(
        Portfolio.name == f"strategy_{strategy_name}"
    ).first()

    if not portfolio:
        return None

    # Get open positions
    open_trades = session.query(Trade).filter(
        Trade.status == TradeStatus.OPEN,
        Trade.notes == f"strategy_{strategy_name}",
    ).all()

    # Get all closed trades for history
    all_trades = session.query(Trade).filter(
        Trade.status == TradeStatus.CLOSED,
        Trade.notes == f"strategy_{strategy_name}",
    ).order_by(Trade.exit_time).all()

    # Calculate equity curve
    equity_curve = []
    running_balance = portfolio.initial_balance
    for trade in all_trades:
        if trade.pnl_usd:
            running_balance += trade.pnl_usd
            equity_curve.append({
                "time": trade.exit_time.isoformat() if trade.exit_time else None,
                "balance": running_balance,
                "pnl": trade.pnl_usd,
            })

    # Calculate stats
    total_return = ((portfolio.current_balance - portfolio.initial_balance) / portfolio.initial_balance) * 100
    win_rate = (portfolio.winning_trades / portfolio.total_trades * 100) if portfolio.total_trades > 0 else 0

    # Calculate max drawdown
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
    trades_by_asset = defaultdict(int)
    for trade in all_trades:
        if trade.pnl_usd:
            pnl_by_asset[trade.asset.value] += trade.pnl_usd
            trades_by_asset[trade.asset.value] += 1

    # Recent trades
    recent_trades = all_trades[-20:] if all_trades else []
    recent_trades.reverse()

    # Avg win/loss
    wins = [t.pnl_percent for t in all_trades if t.pnl_percent and t.pnl_percent > 0]
    losses = [t.pnl_percent for t in all_trades if t.pnl_percent and t.pnl_percent < 0]
    avg_win = sum(wins) / len(wins) if wins else 0
    avg_loss = sum(losses) / len(losses) if losses else 0

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
        "trades_by_asset": dict(trades_by_asset),
    }


@app.get("/", response_class=HTMLResponse)
async def dashboard():
    """Main dashboard page."""
    session = get_session()

    strategies = ["social_pure", "technical_strict", "balanced"]
    stats = {}
    for s in strategies:
        data = get_strategy_stats(session, s)
        if data:
            stats[s] = data

    # Find leader
    leader = None
    best_return = -999
    for name, data in stats.items():
        if data["total_return"] > best_return:
            best_return = data["total_return"]
            leader = name

    session.close()

    # Prepare chart data
    equity_data = {}
    for name, data in stats.items():
        equity_data[name] = data["equity_curve"]

    # Combine all trades for comparison
    all_pnl_by_asset = defaultdict(lambda: defaultdict(float))
    for name, data in stats.items():
        for asset, pnl in data["pnl_by_asset"].items():
            all_pnl_by_asset[asset][name] = pnl

    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Crypto Bot Dashboard</title>
        <meta http-equiv="refresh" content="60">
        <meta name="viewport" content="width=device-width, initial-scale=1">
        <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
        <script src="https://cdn.jsdelivr.net/npm/chartjs-adapter-date-fns"></script>
        <style>
            :root {{
                --bg-primary: #0d1117;
                --bg-secondary: #161b22;
                --bg-tertiary: #21262d;
                --border: #30363d;
                --text-primary: #e6edf3;
                --text-secondary: #8b949e;
                --accent: #58a6ff;
                --green: #3fb950;
                --red: #f85149;
                --yellow: #d29922;
                --purple: #a371f7;
            }}
            * {{ box-sizing: border-box; margin: 0; padding: 0; }}
            body {{
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
                background: var(--bg-primary);
                color: var(--text-primary);
                min-height: 100vh;
            }}
            .header {{
                background: linear-gradient(180deg, var(--bg-secondary) 0%, var(--bg-primary) 100%);
                padding: 30px 20px;
                text-align: center;
                border-bottom: 1px solid var(--border);
            }}
            .header h1 {{
                font-size: 2.5em;
                background: linear-gradient(90deg, var(--accent), var(--purple));
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
                margin-bottom: 8px;
            }}
            .header .subtitle {{
                color: var(--text-secondary);
            }}
            .container {{
                max-width: 1600px;
                margin: 0 auto;
                padding: 30px 20px;
            }}
            .leader-banner {{
                background: linear-gradient(90deg, #238636, #2ea043);
                padding: 20px;
                border-radius: 12px;
                text-align: center;
                margin-bottom: 30px;
                font-size: 1.3em;
                box-shadow: 0 4px 20px rgba(35, 134, 54, 0.3);
            }}
            .leader-banner span {{
                font-weight: 700;
            }}
            .tabs {{
                display: flex;
                gap: 10px;
                margin-bottom: 30px;
                flex-wrap: wrap;
            }}
            .tab {{
                padding: 12px 24px;
                background: var(--bg-secondary);
                border: 1px solid var(--border);
                border-radius: 8px;
                cursor: pointer;
                transition: all 0.2s;
                font-weight: 500;
            }}
            .tab:hover {{
                border-color: var(--accent);
            }}
            .tab.active {{
                background: var(--accent);
                border-color: var(--accent);
                color: #000;
            }}
            .tab-content {{
                display: none;
            }}
            .tab-content.active {{
                display: block;
            }}
            .grid {{
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(350px, 1fr));
                gap: 20px;
                margin-bottom: 30px;
            }}
            .card {{
                background: var(--bg-secondary);
                border: 1px solid var(--border);
                border-radius: 12px;
                overflow: hidden;
                transition: transform 0.2s, box-shadow 0.2s;
            }}
            .card:hover {{
                transform: translateY(-2px);
                box-shadow: 0 8px 30px rgba(0,0,0,0.3);
            }}
            .card-header {{
                padding: 20px;
                border-bottom: 1px solid var(--border);
                display: flex;
                justify-content: space-between;
                align-items: center;
            }}
            .card-title {{
                font-size: 1.4em;
                font-weight: 600;
                text-transform: uppercase;
                letter-spacing: 1px;
            }}
            .card-body {{
                padding: 20px;
            }}
            .badge {{
                padding: 6px 14px;
                border-radius: 20px;
                font-size: 0.75em;
                font-weight: 700;
                text-transform: uppercase;
                letter-spacing: 0.5px;
            }}
            .badge-leader {{
                background: linear-gradient(90deg, #238636, #2ea043);
                animation: pulse 2s infinite;
            }}
            @keyframes pulse {{
                0%, 100% {{ opacity: 1; }}
                50% {{ opacity: 0.7; }}
            }}
            .stats-grid {{
                display: grid;
                grid-template-columns: repeat(3, 1fr);
                gap: 15px;
            }}
            .stat {{
                background: var(--bg-tertiary);
                padding: 15px;
                border-radius: 8px;
                text-align: center;
            }}
            .stat-label {{
                color: var(--text-secondary);
                font-size: 0.75em;
                text-transform: uppercase;
                letter-spacing: 0.5px;
                margin-bottom: 6px;
            }}
            .stat-value {{
                font-size: 1.5em;
                font-weight: 700;
            }}
            .stat-value.positive {{ color: var(--green); }}
            .stat-value.negative {{ color: var(--red); }}
            .chart-container {{
                margin-top: 20px;
                height: 200px;
                position: relative;
            }}
            .positions-list {{
                margin-top: 15px;
            }}
            .position-item {{
                display: flex;
                justify-content: space-between;
                align-items: center;
                padding: 12px 15px;
                background: var(--bg-tertiary);
                border-radius: 8px;
                margin-bottom: 8px;
            }}
            .position-asset {{
                font-weight: 600;
                display: flex;
                align-items: center;
                gap: 8px;
            }}
            .direction-badge {{
                padding: 3px 8px;
                border-radius: 4px;
                font-size: 0.7em;
                font-weight: 700;
            }}
            .direction-badge.long {{
                background: rgba(63, 185, 80, 0.2);
                color: var(--green);
            }}
            .direction-badge.short {{
                background: rgba(248, 81, 73, 0.2);
                color: var(--red);
            }}
            .big-chart {{
                background: var(--bg-secondary);
                border: 1px solid var(--border);
                border-radius: 12px;
                padding: 20px;
                margin-bottom: 30px;
            }}
            .big-chart h3 {{
                margin-bottom: 20px;
                color: var(--text-secondary);
            }}
            .big-chart-container {{
                height: 400px;
                position: relative;
            }}
            table {{
                width: 100%;
                border-collapse: collapse;
            }}
            th, td {{
                padding: 12px 15px;
                text-align: left;
                border-bottom: 1px solid var(--border);
            }}
            th {{
                color: var(--text-secondary);
                font-weight: 500;
                font-size: 0.85em;
                text-transform: uppercase;
                letter-spacing: 0.5px;
            }}
            tr:hover td {{
                background: var(--bg-tertiary);
            }}
            .footer {{
                text-align: center;
                padding: 30px;
                color: var(--text-secondary);
                border-top: 1px solid var(--border);
                margin-top: 30px;
            }}
            .no-data {{
                text-align: center;
                padding: 60px 20px;
                color: var(--text-secondary);
            }}
            .comparison-grid {{
                display: grid;
                grid-template-columns: repeat(3, 1fr);
                gap: 20px;
                margin-bottom: 30px;
            }}
            @media (max-width: 900px) {{
                .comparison-grid {{
                    grid-template-columns: 1fr;
                }}
                .stats-grid {{
                    grid-template-columns: repeat(2, 1fr);
                }}
            }}
        </style>
    </head>
    <body>
        <div class="header">
            <h1>Crypto Bot Dashboard</h1>
            <p class="subtitle">Multi-Strategy A/B Testing | Auto-refreshes every 60 seconds</p>
        </div>

        <div class="container">
            {f'<div class="leader-banner">Current Leader: <span>{leader.upper().replace("_", " ")}</span> with <span>{best_return:+.2f}%</span> return</div>' if leader and best_return != 0 else '<div class="leader-banner">Waiting for first trades...</div>'}

            <div class="tabs">
                <div class="tab active" onclick="showTab('overview')">Overview</div>
                <div class="tab" onclick="showTab('social_pure')">Social Pure</div>
                <div class="tab" onclick="showTab('technical_strict')">Technical Strict</div>
                <div class="tab" onclick="showTab('balanced')">Balanced</div>
                <div class="tab" onclick="showTab('trades')">All Trades</div>
            </div>

            <!-- Overview Tab -->
            <div id="overview" class="tab-content active">
                <div class="big-chart">
                    <h3>Equity Curves Comparison</h3>
                    <div class="big-chart-container">
                        <canvas id="equityChart"></canvas>
                    </div>
                </div>

                <div class="comparison-grid">
    """

    # Strategy comparison cards
    for name, data in stats.items():
        is_leader = name == leader
        return_class = "positive" if data["total_return"] > 0 else "negative" if data["total_return"] < 0 else ""
        pnl_class = "positive" if data["total_pnl"] > 0 else "negative" if data["total_pnl"] < 0 else ""

        html += f"""
                    <div class="card">
                        <div class="card-header">
                            <span class="card-title">{name.replace("_", " ")}</span>
                            {'<span class="badge badge-leader">LEADER</span>' if is_leader else ''}
                        </div>
                        <div class="card-body">
                            <div class="stats-grid">
                                <div class="stat">
                                    <div class="stat-label">Balance</div>
                                    <div class="stat-value">${data["balance"]:,.0f}</div>
                                </div>
                                <div class="stat">
                                    <div class="stat-label">Return</div>
                                    <div class="stat-value {return_class}">{data["total_return"]:+.2f}%</div>
                                </div>
                                <div class="stat">
                                    <div class="stat-label">PnL</div>
                                    <div class="stat-value {pnl_class}">${data["total_pnl"]:+,.0f}</div>
                                </div>
                                <div class="stat">
                                    <div class="stat-label">Trades</div>
                                    <div class="stat-value">{data["total_trades"]}</div>
                                </div>
                                <div class="stat">
                                    <div class="stat-label">Win Rate</div>
                                    <div class="stat-value">{data["win_rate"]:.1f}%</div>
                                </div>
                                <div class="stat">
                                    <div class="stat-label">Max DD</div>
                                    <div class="stat-value negative">{data["max_drawdown"]:.1f}%</div>
                                </div>
                            </div>
                            <div class="positions-list">
                                <strong style="color: var(--text-secondary); font-size: 0.85em;">Open Positions ({len(data["open_positions"])})</strong>
        """

        if data["open_positions"]:
            for trade in data["open_positions"]:
                dir_class = "long" if trade.direction == Direction.LONG else "short"
                html += f'''
                                <div class="position-item">
                                    <span class="position-asset">
                                        <span class="direction-badge {dir_class}">{trade.direction.value.upper()}</span>
                                        {trade.asset.value}
                                    </span>
                                    <span>@ ${trade.entry_price:,.2f}</span>
                                </div>
                '''
        else:
            html += '<div class="position-item" style="justify-content: center; color: var(--text-secondary);">No open positions</div>'

        html += """
                            </div>
                        </div>
                    </div>
        """

    html += """
                </div>

                <div class="big-chart">
                    <h3>PnL by Asset (All Strategies)</h3>
                    <div class="big-chart-container">
                        <canvas id="assetChart"></canvas>
                    </div>
                </div>
            </div>
    """

    # Individual strategy tabs
    for name, data in stats.items():
        html += f"""
            <div id="{name}" class="tab-content">
                <div class="grid">
                    <div class="card">
                        <div class="card-header">
                            <span class="card-title">Performance</span>
                        </div>
                        <div class="card-body">
                            <div class="stats-grid">
                                <div class="stat">
                                    <div class="stat-label">Balance</div>
                                    <div class="stat-value">${data["balance"]:,.2f}</div>
                                </div>
                                <div class="stat">
                                    <div class="stat-label">Return</div>
                                    <div class="stat-value {"positive" if data["total_return"] > 0 else "negative"}">{data["total_return"]:+.2f}%</div>
                                </div>
                                <div class="stat">
                                    <div class="stat-label">Total PnL</div>
                                    <div class="stat-value {"positive" if data["total_pnl"] > 0 else "negative"}">${data["total_pnl"]:+,.2f}</div>
                                </div>
                                <div class="stat">
                                    <div class="stat-label">Total Trades</div>
                                    <div class="stat-value">{data["total_trades"]}</div>
                                </div>
                                <div class="stat">
                                    <div class="stat-label">Win Rate</div>
                                    <div class="stat-value">{data["win_rate"]:.1f}%</div>
                                </div>
                                <div class="stat">
                                    <div class="stat-label">Max Drawdown</div>
                                    <div class="stat-value negative">{data["max_drawdown"]:.1f}%</div>
                                </div>
                                <div class="stat">
                                    <div class="stat-label">Avg Win</div>
                                    <div class="stat-value positive">{data["avg_win"]:+.2f}%</div>
                                </div>
                                <div class="stat">
                                    <div class="stat-label">Avg Loss</div>
                                    <div class="stat-value negative">{data["avg_loss"]:.2f}%</div>
                                </div>
                                <div class="stat">
                                    <div class="stat-label">Open</div>
                                    <div class="stat-value">{len(data["open_positions"])}</div>
                                </div>
                            </div>
                        </div>
                    </div>
                    <div class="card">
                        <div class="card-header">
                            <span class="card-title">Equity Curve</span>
                        </div>
                        <div class="card-body">
                            <div class="chart-container" style="height: 250px;">
                                <canvas id="{name}Chart"></canvas>
                            </div>
                        </div>
                    </div>
                </div>
                <div class="card">
                    <div class="card-header">
                        <span class="card-title">Trade History</span>
                    </div>
                    <div class="card-body">
                        <table>
                            <thead>
                                <tr>
                                    <th>Asset</th>
                                    <th>Direction</th>
                                    <th>Entry</th>
                                    <th>Exit</th>
                                    <th>PnL %</th>
                                    <th>PnL $</th>
                                    <th>Time</th>
                                </tr>
                            </thead>
                            <tbody>
        """

        for trade in data["recent_trades"]:
            pnl_class = "positive" if (trade.pnl_percent or 0) > 0 else "negative"
            dir_class = "positive" if trade.direction == Direction.LONG else "negative"

            html += f"""
                                <tr>
                                    <td><strong>{trade.asset.value}</strong></td>
                                    <td class="{dir_class}">{trade.direction.value.upper()}</td>
                                    <td>${trade.entry_price:,.2f}</td>
                                    <td>${trade.exit_price:,.2f if trade.exit_price else 0}</td>
                                    <td class="{pnl_class}">{trade.pnl_percent:+.2f}%</td>
                                    <td class="{pnl_class}">${trade.pnl_usd:+,.2f if trade.pnl_usd else 0}</td>
                                    <td>{trade.exit_time.strftime("%m/%d %H:%M") if trade.exit_time else "-"}</td>
                                </tr>
            """

        if not data["recent_trades"]:
            html += '<tr><td colspan="7" style="text-align: center; color: var(--text-secondary);">No trades yet</td></tr>'

        html += """
                            </tbody>
                        </table>
                    </div>
                </div>
            </div>
        """

    # All trades tab
    all_trades = []
    for name, data in stats.items():
        for trade in data["recent_trades"]:
            all_trades.append((name, trade))
    all_trades.sort(key=lambda x: x[1].exit_time or datetime.min, reverse=True)

    html += """
            <div id="trades" class="tab-content">
                <div class="card">
                    <div class="card-header">
                        <span class="card-title">All Recent Trades</span>
                    </div>
                    <div class="card-body">
                        <table>
                            <thead>
                                <tr>
                                    <th>Strategy</th>
                                    <th>Asset</th>
                                    <th>Direction</th>
                                    <th>Entry</th>
                                    <th>Exit</th>
                                    <th>PnL %</th>
                                    <th>PnL $</th>
                                    <th>Time</th>
                                </tr>
                            </thead>
                            <tbody>
    """

    for strategy, trade in all_trades[:50]:
        pnl_class = "positive" if (trade.pnl_percent or 0) > 0 else "negative"
        dir_class = "positive" if trade.direction == Direction.LONG else "negative"

        html += f"""
                                <tr>
                                    <td>{strategy.replace("_", " ").title()}</td>
                                    <td><strong>{trade.asset.value}</strong></td>
                                    <td class="{dir_class}">{trade.direction.value.upper()}</td>
                                    <td>${trade.entry_price:,.2f}</td>
                                    <td>${trade.exit_price:,.2f if trade.exit_price else 0}</td>
                                    <td class="{pnl_class}">{trade.pnl_percent:+.2f}%</td>
                                    <td class="{pnl_class}">${trade.pnl_usd:+,.2f if trade.pnl_usd else 0}</td>
                                    <td>{trade.exit_time.strftime("%m/%d %H:%M") if trade.exit_time else "-"}</td>
                                </tr>
        """

    if not all_trades:
        html += '<tr><td colspan="8" style="text-align: center; color: var(--text-secondary);">No trades yet</td></tr>'

    html += """
                            </tbody>
                        </table>
                    </div>
                </div>
            </div>
        </div>

        <div class="footer">
            Last updated: """ + datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S") + """ UTC |
            Data refreshes every 60 seconds
        </div>

        <script>
            // Tab switching
            function showTab(tabId) {
                document.querySelectorAll('.tab-content').forEach(t => t.classList.remove('active'));
                document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
                document.getElementById(tabId).classList.add('active');
                event.target.classList.add('active');
            }

            // Chart colors
            const colors = {
                social_pure: '#3fb950',
                technical_strict: '#58a6ff',
                balanced: '#a371f7'
            };

            // Equity curve data
            const equityData = """ + json.dumps(equity_data) + """;

            // Main equity chart
            const equityCtx = document.getElementById('equityChart');
            if (equityCtx) {
                const datasets = [];
                for (const [strategy, data] of Object.entries(equityData)) {
                    if (data.length > 0) {
                        datasets.push({
                            label: strategy.replace('_', ' ').toUpperCase(),
                            data: data.map(d => ({x: new Date(d.time), y: d.balance})),
                            borderColor: colors[strategy],
                            backgroundColor: colors[strategy] + '20',
                            fill: true,
                            tension: 0.3,
                            pointRadius: 2,
                        });
                    }
                }

                new Chart(equityCtx, {
                    type: 'line',
                    data: { datasets },
                    options: {
                        responsive: true,
                        maintainAspectRatio: false,
                        interaction: { intersect: false, mode: 'index' },
                        plugins: {
                            legend: { labels: { color: '#e6edf3' } },
                            tooltip: {
                                callbacks: {
                                    label: ctx => `${ctx.dataset.label}: $${ctx.parsed.y.toLocaleString()}`
                                }
                            }
                        },
                        scales: {
                            x: {
                                type: 'time',
                                grid: { color: '#30363d' },
                                ticks: { color: '#8b949e' }
                            },
                            y: {
                                grid: { color: '#30363d' },
                                ticks: {
                                    color: '#8b949e',
                                    callback: v => '$' + v.toLocaleString()
                                }
                            }
                        }
                    }
                });
            }

            // Asset PnL chart
            const assetData = """ + json.dumps(dict(all_pnl_by_asset)) + """;
            const assetCtx = document.getElementById('assetChart');
            if (assetCtx && Object.keys(assetData).length > 0) {
                const assets = Object.keys(assetData);
                const datasets = [];

                for (const strategy of ['social_pure', 'technical_strict', 'balanced']) {
                    datasets.push({
                        label: strategy.replace('_', ' ').toUpperCase(),
                        data: assets.map(a => assetData[a][strategy] || 0),
                        backgroundColor: colors[strategy],
                    });
                }

                new Chart(assetCtx, {
                    type: 'bar',
                    data: { labels: assets, datasets },
                    options: {
                        responsive: true,
                        maintainAspectRatio: false,
                        plugins: {
                            legend: { labels: { color: '#e6edf3' } },
                            tooltip: {
                                callbacks: {
                                    label: ctx => `${ctx.dataset.label}: $${ctx.parsed.y.toFixed(2)}`
                                }
                            }
                        },
                        scales: {
                            x: { grid: { color: '#30363d' }, ticks: { color: '#8b949e' } },
                            y: {
                                grid: { color: '#30363d' },
                                ticks: { color: '#8b949e', callback: v => '$' + v }
                            }
                        }
                    }
                });
            }

            // Individual strategy charts
            for (const [strategy, data] of Object.entries(equityData)) {
                const ctx = document.getElementById(strategy + 'Chart');
                if (ctx && data.length > 0) {
                    new Chart(ctx, {
                        type: 'line',
                        data: {
                            datasets: [{
                                label: 'Balance',
                                data: data.map(d => ({x: new Date(d.time), y: d.balance})),
                                borderColor: colors[strategy],
                                backgroundColor: colors[strategy] + '20',
                                fill: true,
                                tension: 0.3,
                            }]
                        },
                        options: {
                            responsive: true,
                            maintainAspectRatio: false,
                            plugins: { legend: { display: false } },
                            scales: {
                                x: {
                                    type: 'time',
                                    grid: { color: '#30363d' },
                                    ticks: { color: '#8b949e' }
                                },
                                y: {
                                    grid: { color: '#30363d' },
                                    ticks: { color: '#8b949e', callback: v => '$' + v }
                                }
                            }
                        }
                    });
                }
            }
        </script>
    </body>
    </html>
    """

    return HTMLResponse(content=html)


@app.get("/api/stats")
async def api_stats():
    """JSON API for stats."""
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


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8080))
    uvicorn.run(app, host="0.0.0.0", port=port)
