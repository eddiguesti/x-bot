"""
Dashboard Reporter for Social Signals Bot

Sends bot status, signals, and trades to the unified crypto dashboard.
"""

import logging
import os
from datetime import datetime
from typing import Optional
import httpx

logger = logging.getLogger(__name__)


class DashboardReporter:
    """
    Reports bot activity to the unified dashboard.

    Set DASHBOARD_URL environment variable to enable reporting.
    If not set, reporting is silently disabled.
    """

    def __init__(
        self,
        bot_id: str,
        bot_name: str,
        bot_type: str = "social-signals-bot",
        dashboard_url: Optional[str] = None,
    ):
        self.bot_id = bot_id
        self.bot_name = bot_name
        self.bot_type = bot_type
        self.dashboard_url = dashboard_url or os.environ.get("DASHBOARD_URL")
        self.enabled = bool(self.dashboard_url)

        if self.enabled:
            logger.info(f"Dashboard reporting enabled: {self.dashboard_url}")
            self._register()
        else:
            logger.info("Dashboard reporting disabled (DASHBOARD_URL not set)")

    def _post(self, endpoint: str, data: dict) -> bool:
        """Post data to dashboard endpoint."""
        if not self.enabled:
            return False

        try:
            url = f"{self.dashboard_url.rstrip('/')}/api/{endpoint}"
            with httpx.Client(timeout=5.0) as client:
                response = client.post(url, json=data)
                response.raise_for_status()
                return True
        except Exception as e:
            logger.warning(f"Dashboard report failed ({endpoint}): {e}")
            return False

    def _register(self) -> None:
        """Register bot with dashboard."""
        self._post("register", {
            "bot_id": self.bot_id,
            "bot_name": self.bot_name,
            "bot_type": self.bot_type,
            "config": {}
        })

    def heartbeat(
        self,
        equity: float,
        daily_pnl: float,
        daily_pnl_pct: float,
        open_positions: int,
        max_positions: int,
        kill_switch_active: bool = False,
        extra: Optional[dict] = None,
    ) -> None:
        """Send heartbeat with current status."""
        self._post("heartbeat", {
            "bot_id": self.bot_id,
            "equity": equity,
            "daily_pnl": daily_pnl,
            "daily_pnl_pct": daily_pnl_pct,
            "open_positions": open_positions,
            "max_positions": max_positions,
            "kill_switch_active": kill_switch_active,
            "extra": extra or {}
        })

    def report_signal(
        self,
        signal_id: str,
        signal_type: str,
        asset: str,
        direction: str,
        confidence: float,
        source: str = "",
        details: Optional[dict] = None,
        status: str = "armed",
    ) -> None:
        """Report a new signal."""
        self._post("signal", {
            "bot_id": self.bot_id,
            "signal_id": signal_id,
            "signal_type": signal_type,
            "asset": asset,
            "direction": direction,
            "confidence": confidence,
            "source": source,
            "details": details or {},
            "status": status
        })

    def report_trade(
        self,
        trade_id: str,
        asset: str,
        direction: str,
        entry_price: float,
        position_value: float,
        status: str = "open",
        exit_price: Optional[float] = None,
        pnl: Optional[float] = None,
        pnl_pct: Optional[float] = None,
        exit_reason: Optional[str] = None,
        signal_source: str = "",
    ) -> None:
        """Report a trade (open or close)."""
        self._post("trade", {
            "bot_id": self.bot_id,
            "trade_id": trade_id,
            "asset": asset,
            "direction": direction,
            "entry_price": entry_price,
            "position_value": position_value,
            "status": status,
            "exit_price": exit_price,
            "pnl": pnl,
            "pnl_pct": pnl_pct,
            "exit_reason": exit_reason,
            "signal_source": signal_source
        })

    def report_social_signal(
        self,
        platform: str,
        creator: str,
        content: str,
        sentiment: str,
        ticker: str,
        confidence: float,
    ) -> None:
        """Report a social media signal."""
        self._post("social_signal", {
            "bot_id": self.bot_id,
            "platform": platform,
            "creator": creator,
            "content": content[:500],  # Truncate long content
            "sentiment": sentiment,
            "ticker": ticker,
            "confidence": confidence
        })


def create_reporter(bot_name: str = "social-signals-bot") -> DashboardReporter:
    """Create a DashboardReporter for social signals bot."""
    return DashboardReporter(
        bot_id=f"{bot_name}-{os.getpid()}",
        bot_name=bot_name,
        bot_type="social-signals-bot",
    )
