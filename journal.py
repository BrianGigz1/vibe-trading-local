"""Obsidian trade journal writer.

Appends simulated fills to a daily note in the OpenHuman vault, in Obsidian
Flavoured Markdown with frontmatter and tags so Dataview can query them.

Every entry is stamped `simulated: true`. A journal that cannot distinguish a
paper fill from a real one is worse than no journal — six months later nobody
remembers which was which.
"""
from __future__ import annotations

import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Final

logger = logging.getLogger(__name__)

DEFAULT_VAULT: Final[Path] = Path(
    r"C:\Users\Brian Jerry Kangethe\Documents\OpenHuman-Vault\Trading\Journal"
)


def journal_dir() -> Path:
    """Resolve the journal directory, honouring ``VT_OBSIDIAN_JOURNAL``."""
    raw = os.getenv("VT_OBSIDIAN_JOURNAL", "").strip()
    return Path(raw) if raw else DEFAULT_VAULT


def _ensure_note(path: Path, day: str) -> None:
    """Create the daily note with frontmatter if it does not yet exist."""
    if path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "---\n"
        f"date: {day}\n"
        "type: trade-journal\n"
        "simulated: true\n"
        "tags: [trading, paper, simulated]\n"
        "---\n\n"
        f"# Trade Journal — {day}\n\n"
        "> **All entries below are SIMULATED paper fills.** No order in this note\n"
        "> was sent to a broker or exchange.\n\n",
        encoding="utf-8",
    )


def log_fill(
    *,
    side: str,
    ticker: str,
    price: float,
    quantity: float,
    equity: float,
    order_id: str,
    stop_loss: float | None = None,
    take_profit: float | None = None,
    reason: str = "",
) -> Path | None:
    """Append a simulated fill to today's note.

    Args:
        side: ``BUY`` or ``SELL``.
        ticker: Instrument symbol.
        price: Fill price.
        quantity: Units filled.
        equity: Account equity after the fill.
        order_id: Simulated order identifier.
        stop_loss: Protective stop, if any.
        take_profit: Target, if any.
        reason: Sizing or rejection rationale.

    Returns:
        Path to the note, or ``None`` if the write failed. Journalling is
        best-effort: it must never break the trading path.
    """
    try:
        now = datetime.now(timezone.utc)
        day = now.strftime("%Y-%m-%d")
        path = journal_dir() / f"{day}.md"
        _ensure_note(path, day)

        tag = ticker.replace("/", "").replace(":", "").lower()
        risk = ""
        if stop_loss and price > stop_loss:
            risk = f" | **Risk/unit** `{price - stop_loss:,.4f}`"

        entry = (
            f"### {now.strftime('%H:%M:%S')} UTC — {side} {ticker} `[SIMULATED]`\n"
            f"- **Tags:** #trade-log #paper #{tag} #{side.lower()}\n"
            f"- **Price** `{price:,.4f}` | **Qty** `{quantity:,.4f}`"
            f" | **Notional** `{price * quantity:,.2f}`{risk}\n"
            f"- **Stop** `{stop_loss if stop_loss else 'n/a'}`"
            f" | **Target** `{take_profit if take_profit else 'n/a'}`\n"
            f"- **Equity after** `{equity:,.2f}` | **Order** `{order_id}`\n"
        )
        if reason:
            entry += f"- **Note:** {reason}\n"
        entry += "\n"

        with path.open("a", encoding="utf-8") as fh:
            fh.write(entry)
        logger.info("journalled %s %s -> %s", side, ticker, path.name)
        return path
    except OSError as exc:
        logger.warning("journal write failed (non-fatal): %s", exc)
        return None


def log_snapshot(state: dict[str, Any]) -> Path | None:
    """Append a portfolio snapshot to today's note."""
    try:
        now = datetime.now(timezone.utc)
        day = now.strftime("%Y-%m-%d")
        path = journal_dir() / f"{day}.md"
        _ensure_note(path, day)
        positions = state.get("positions") or {}
        held = ", ".join(f"`{k}` {v:g}" for k, v in positions.items()) or "flat"
        with path.open("a", encoding="utf-8") as fh:
            fh.write(
                f"### {now.strftime('%H:%M:%S')} UTC — SNAPSHOT `[SIMULATED]`\n"
                f"- **Equity** `{state.get('equity', 0):,.2f}`"
                f" | **Cash** `{state.get('cash', 0):,.2f}`\n"
                f"- **Positions:** {held}\n"
                f"- **Orders:** `{state.get('orders', 0)}`"
                f" | **Risk halted:** `{state.get('risk_halted', False)}`\n\n"
            )
        return path
    except OSError as exc:
        logger.warning("snapshot write failed (non-fatal): %s", exc)
        return None
