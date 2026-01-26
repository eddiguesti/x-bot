"""Notification system for trade alerts."""

import logging
from typing import Optional

import httpx

from .config import Settings

logger = logging.getLogger(__name__)


class TelegramNotifier:
    """Send notifications via Telegram bot."""

    def __init__(self, settings: Settings):
        self.bot_token = settings.telegram_bot_token
        self.chat_id = settings.telegram_chat_id
        self.enabled = bool(self.bot_token and self.chat_id)

        if self.enabled:
            logger.info("Telegram notifications enabled")

    def send(self, message: str) -> bool:
        """Send a message via Telegram."""
        if not self.enabled:
            return False

        try:
            url = f"https://api.telegram.org/bot{self.bot_token}/sendMessage"
            response = httpx.post(
                url,
                json={
                    "chat_id": self.chat_id,
                    "text": message,
                    "parse_mode": "HTML",
                },
                timeout=10,
            )
            return response.status_code == 200
        except Exception as e:
            logger.warning(f"Failed to send Telegram message: {e}")
            return False

    def notify_trade_opened(
        self,
        asset: str,
        direction: str,
        entry_price: float,
        position_size: float,
        confidence: float,
    ):
        """Notify when a trade is opened."""
        emoji = "🟢" if direction.lower() == "long" else "🔴"
        message = (
            f"{emoji} <b>NEW TRADE OPENED</b>\n\n"
            f"Asset: {asset}\n"
            f"Direction: {direction.upper()}\n"
            f"Entry: ${entry_price:,.2f}\n"
            f"Size: {position_size:.6f}\n"
            f"Confidence: {confidence:.0%}"
        )
        self.send(message)

    def notify_trade_closed(
        self,
        asset: str,
        direction: str,
        entry_price: float,
        exit_price: float,
        pnl_percent: float,
        pnl_usd: float,
        reason: str,
    ):
        """Notify when a trade is closed."""
        emoji = "✅" if pnl_percent > 0 else "❌"
        message = (
            f"{emoji} <b>TRADE CLOSED</b>\n\n"
            f"Asset: {asset}\n"
            f"Direction: {direction.upper()}\n"
            f"Entry: ${entry_price:,.2f}\n"
            f"Exit: ${exit_price:,.2f}\n"
            f"PnL: {pnl_percent:+.2f}% (${pnl_usd:+.2f})\n"
            f"Reason: {reason}"
        )
        self.send(message)

    def notify_consensus_change(
        self,
        asset: str,
        action: str,
        confidence: float,
        long_votes: int,
        short_votes: int,
    ):
        """Notify on significant consensus changes."""
        emoji = "🟢" if action == "long" else ("🔴" if action == "short" else "⚪")
        message = (
            f"{emoji} <b>CONSENSUS UPDATE</b>\n\n"
            f"Asset: {asset}\n"
            f"Signal: {action.upper()}\n"
            f"Confidence: {confidence:.0%}\n"
            f"Votes: {long_votes} long / {short_votes} short"
        )
        self.send(message)

    def notify_daily_summary(
        self,
        balance: float,
        daily_pnl: float,
        total_pnl: float,
        win_rate: float,
        open_positions: int,
    ):
        """Send daily performance summary."""
        emoji = "📈" if daily_pnl >= 0 else "📉"
        message = (
            f"{emoji} <b>DAILY SUMMARY</b>\n\n"
            f"Balance: ${balance:,.2f}\n"
            f"Today's PnL: ${daily_pnl:+,.2f}\n"
            f"Total PnL: ${total_pnl:+,.2f}\n"
            f"Win Rate: {win_rate:.1%}\n"
            f"Open Positions: {open_positions}"
        )
        self.send(message)


def get_notifier(settings: Settings) -> Optional[TelegramNotifier]:
    """Get configured notifier or None if not configured."""
    notifier = TelegramNotifier(settings)
    return notifier if notifier.enabled else None
