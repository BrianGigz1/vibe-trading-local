"""TradingView webhook receiver — PAPER FILLS ONLY.

Speaks the exact JSON contract a TradingView `alert()` emits, so the same Pine
script can drive it. What it does NOT contain, anywhere, is broker connectivity:
there is no ccxt import, no API key, no order-routing code path. Signals are
filled against a simulated ledger and the risk controls in :mod:`risk` are
applied before every fill.

Use this to validate the whole pipeline — Pine alert → webhook → sizing → risk
gate → fill → notification — with nothing at stake.

Run:
    venv\\Scripts\\python -m uvicorn webhook_paper:app --host 127.0.0.1 --port 8000
"""
from __future__ import annotations

import hmac
import logging
import os
from collections import deque
from datetime import datetime, timezone
from typing import Any, Final, Literal

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, field_validator

import journal
from config import Config, RiskLimits
from risk import RiskManager

logging.basicConfig(level=logging.INFO, format="%(levelname)-7s %(message)s")
logger = logging.getLogger("webhook-paper")

# Bind loopback by default. 0.0.0.0 exposes an order endpoint to the network.
HOST: Final[str] = os.getenv("VT_WEBHOOK_HOST", "127.0.0.1")
PORT: Final[int] = int(os.getenv("VT_WEBHOOK_PORT", "8000"))

# No default. An unset secret disables the endpoint rather than falling back to
# a guessable literal that would also have to be embedded in the Pine script.
PASSPHRASE: Final[str] = os.getenv("VT_WEBHOOK_PASSPHRASE", "")

app = FastAPI(title="Vibe-Trading Paper Webhook", version="1.0")

_cfg = Config()
_risk = RiskManager(_cfg.risk, _cfg.initial_capital)
_seen: deque[str] = deque(maxlen=512)   # replay guard
_ledger: list[dict[str, Any]] = []
_cash: float = _cfg.initial_capital

# Positions are keyed BY TICKER. A single global position blob lets a sell
# signal for one symbol liquidate a position in another — caught in testing,
# and the same class of ticker-blind bug that makes naive webhook bots desync
# from the strategy that is driving them.
_positions: dict[str, dict[str, float]] = {}


def _equity(mark_ticker: str = "", mark: float = 0.0) -> float:
    """Cash plus positions, marking *mark_ticker* at *mark* and others at entry."""
    total = _cash
    for tkr, pos in _positions.items():
        px = mark if (tkr == mark_ticker and mark > 0) else pos["avg_price"]
        total += pos["qty"] * px
    return total


class Signal(BaseModel):
    """Inbound TradingView alert payload."""

    passphrase: str
    action: Literal["buy", "sell"]
    ticker: str = Field(min_length=1, max_length=32)
    price: float = Field(gt=0)
    stop_loss: float | None = Field(default=None, gt=0)
    take_profit: float | None = Field(default=None, gt=0)
    bar_time: str | None = None

    @field_validator("action", mode="before")
    @classmethod
    def _lower(cls, v: Any) -> Any:
        """Accept BUY/Buy/buy alike."""
        return v.lower().strip() if isinstance(v, str) else v


def _authenticate(supplied: str) -> None:
    """Constant-time passphrase check; refuse to run without a configured secret."""
    if not PASSPHRASE:
        raise HTTPException(
            status_code=503,
            detail="VT_WEBHOOK_PASSPHRASE is not set; endpoint disabled",
        )
    if not hmac.compare_digest(supplied, PASSPHRASE):
        raise HTTPException(status_code=401, detail="unauthorized")


@app.get("/health")
def health() -> dict[str, Any]:
    """Liveness plus current simulated account state."""
    return {
        "status": "ok",
        "mode": "PAPER — no broker connectivity",
        "cash": round(_cash, 2),
        "positions": {t: p["qty"] for t, p in _positions.items() if p["qty"] > 0},
        "equity": round(_equity(), 2),
        "orders": len(_ledger),
        "risk_halted": _risk.halted,
        "halt_reason": _risk.halt_reason,
    }


@app.post("/webhook")
def webhook(sig: Signal) -> dict[str, Any]:
    """Accept a signal and simulate a fill.

    Applies, in order: authentication, replay rejection, deterministic risk
    sizing, then a mock fill. Returns the simulated order.
    """
    _authenticate(sig.passphrase)

    # Reject duplicates. TradingView can and does re-deliver alerts; without
    # this, one signal becomes several positions.
    key = f"{sig.ticker}|{sig.action}|{sig.price}|{sig.bar_time or ''}"
    if key in _seen:
        logger.warning("duplicate signal ignored: %s", key)
        return {"status": "ignored", "reason": "duplicate signal"}
    _seen.append(key)

    global _cash
    now = datetime.now(timezone.utc)
    mark = sig.price
    equity = _equity(sig.ticker, mark)
    _risk.start_day(now.date(), equity)
    _risk.update_equity(equity)

    held = _positions.get(sig.ticker, {"qty": 0.0, "avg_price": 0.0})

    if sig.action == "sell":
        qty = held["qty"]
        if qty <= 0:
            # A 'sell' while flat in THIS ticker would open a short on a futures
            # venue. This engine is long-only by construction and refuses.
            return {"status": "rejected",
                    "reason": f"flat in {sig.ticker}; refusing to open a short"}
        proceeds = qty * mark
        _cash += proceeds - proceeds * _cfg.commission_pct
        _positions.pop(sig.ticker, None)
        order = {
            "order_id": f"PAPER-{len(_ledger) + 1:05d}", "ts": now.isoformat(),
            "side": "SELL", "ticker": sig.ticker, "qty": qty, "price": mark,
            "simulated": True,
        }
        _ledger.append(order)
        logger.info("[MOCK] SELL %s %s @ %s", qty, sig.ticker, mark)
        journal.log_fill(
            side="SELL", ticker=sig.ticker, price=mark, quantity=qty,
            equity=_equity(), order_id=order["order_id"], reason="exit signal",
        )
        return {"status": "filled", "order": order, "equity": round(_equity(), 2)}

    # buy
    if held["qty"] > 0:
        return {"status": "rejected", "reason": f"position already open in {sig.ticker}"}

    stop = sig.stop_loss
    if stop is None or stop >= mark:
        # Refusing here is the entire point: the upstream design accepted a
        # stop_loss field and then never used it, leaving positions unbounded.
        return {
            "status": "rejected",
            "reason": "a stop_loss below entry is required; unbounded risk refused",
        }

    verdict = _risk.evaluate_entry(
        equity=equity, cash=_cash, price=mark,
        stop_price=stop, open_positions=len(_positions),
    )
    if not verdict.allowed:
        logger.warning("entry blocked: %s", verdict.reason)
        return {"status": "rejected", "reason": verdict.reason,
                "decision": verdict.decision.value}

    cost = verdict.quantity * mark
    _cash -= cost + cost * _cfg.commission_pct
    _positions[sig.ticker] = {"qty": verdict.quantity, "avg_price": mark}
    order = {
        "order_id": f"PAPER-{len(_ledger) + 1:05d}", "ts": now.isoformat(),
        "side": "BUY", "ticker": sig.ticker, "qty": verdict.quantity,
        "price": mark, "stop_loss": stop, "take_profit": sig.take_profit,
        "simulated": True,
    }
    _ledger.append(order)
    logger.info("[MOCK] BUY %s %s @ %s (stop %s)", verdict.quantity, sig.ticker, mark, stop)
    journal.log_fill(
        side="BUY", ticker=sig.ticker, price=mark, quantity=verdict.quantity,
        equity=_equity(sig.ticker, mark), order_id=order["order_id"],
        stop_loss=stop, take_profit=sig.take_profit, reason=verdict.reason,
    )
    return {"status": "filled", "order": order, "sizing": verdict.reason,
            "equity": round(_equity(sig.ticker, mark), 2)}


@app.get("/orders")
def orders() -> dict[str, Any]:
    """Full simulated order ledger."""
    return {"count": len(_ledger), "orders": _ledger}


if __name__ == "__main__":
    import uvicorn

    if not PASSPHRASE:
        logger.error("VT_WEBHOOK_PASSPHRASE is not set — the endpoint will 503.")
    logger.info("PAPER webhook on http://%s:%d  (no broker connectivity)", HOST, PORT)
    uvicorn.run(app, host=HOST, port=PORT)
