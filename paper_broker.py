"""Alpaca **paper** trading adapter — forward simulated execution.

Everything else in this project replays history. This places real orders against
a real broker's **paper** endpoint, so the machinery is exercised end to end:
order lifecycle, partial fills, rejects, market hours, position state.

Safety, enforced rather than intended:

* **Paper endpoint only.** The base URL must contain ``paper-api``. Anything else
  raises at construction. There is no flag to disable this check — pointing it at
  live requires editing the source, which is a visible diff in git.
* **No funding path.** An Alpaca paper account cannot hold real money.
* **Risk gate before every order.** Sizing comes from :mod:`risk`, the same
  deterministic code the backtester uses. No order is sent without a stop.
* **Refuses to short.** Long-only, matching the tested strategy.

Configuration (never in the repo):

    setx ALPACA_API_KEY    "your-paper-key"
    setx ALPACA_SECRET_KEY "your-paper-secret"

    python paper_broker.py --check     # connectivity + account, place nothing
    python paper_broker.py --dry-run   # decide orders, show them, send none
    python paper_broker.py             # place paper orders
"""
from __future__ import annotations

import argparse
import json
import os
import sqlite3
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

import httpx

import journal
from collector import init_db
from config import Config
from data import DataError, load_ohlcv
from risk import RiskManager
from strategy import Strategy

PAPER_BASE = "https://paper-api.alpaca.markets"
WATCHLIST = ["AAPL", "MSFT", "NVDA", "SPY", "TSLA", "AMZN", "META", "GOOGL"]
START = "2023-01-01"


class NotConfigured(RuntimeError):
    """No API credentials present."""


class UnsafeEndpoint(RuntimeError):
    """Refused: the configured endpoint is not a paper endpoint."""


@dataclass
class Decision:
    """One symbol's intended action for this run."""

    symbol: str
    action: str            # buy | hold | flat | blocked
    qty: float = 0.0
    price: float = 0.0
    stop: float = 0.0
    reason: str = ""


class AlpacaPaper:
    """Minimal REST client for Alpaca's paper endpoint."""

    def __init__(self, base: str | None = None) -> None:
        """Bind credentials and refuse any non-paper endpoint.

        Raises:
            UnsafeEndpoint: If *base* is not an Alpaca paper URL.
            NotConfigured: If credentials are absent.
        """
        self.base = (base or os.getenv("ALPACA_BASE_URL") or PAPER_BASE).rstrip("/")
        if "paper-api" not in self.base:
            raise UnsafeEndpoint(
                f"refusing to use {self.base!r} — this adapter is paper-only. "
                f"Live trading would require editing the source."
            )
        key = os.getenv("ALPACA_API_KEY", "")
        sec = os.getenv("ALPACA_SECRET_KEY", "")
        if not key or not sec:
            raise NotConfigured(
                "ALPACA_API_KEY / ALPACA_SECRET_KEY not set. Create a free paper "
                "account at alpaca.markets and set them as environment variables."
            )
        self._h = {"APCA-API-KEY-ID": key, "APCA-API-SECRET-KEY": sec}

    def _get(self, path: str, **params: Any) -> Any:
        r = httpx.get(f"{self.base}{path}", headers=self._h, params=params, timeout=20)
        r.raise_for_status()
        return r.json()

    def account(self) -> dict:
        """Account equity, cash and trading status."""
        return self._get("/v2/account")

    def positions(self) -> list[dict]:
        """Currently open positions."""
        return self._get("/v2/positions")

    def clock(self) -> dict:
        """Market clock — is the exchange open right now."""
        return self._get("/v2/clock")

    def submit_bracket(self, symbol: str, qty: float, stop: float) -> dict:
        """Buy market, with a protective stop attached in the same request.

        The stop is part of the order, not a level tracked in this process. If
        this script dies, the stop still rests at the broker — the single most
        important difference from the webhook designs reviewed earlier today.
        """
        body = {
            "symbol": symbol,
            "qty": str(int(qty)),
            "side": "buy",
            "type": "market",
            "time_in_force": "day",
            "order_class": "oto",
            "stop_loss": {"stop_price": f"{stop:.2f}"},
        }
        r = httpx.post(f"{self.base}/v2/orders", headers=self._h, json=body, timeout=20)
        if r.status_code >= 400:
            raise RuntimeError(f"order rejected {r.status_code}: {r.text[:200]}")
        return r.json()


def decide(symbol: str, equity: float, held: set[str], risk: RiskManager) -> Decision:
    """Evaluate today's signal for *symbol* and size it under the risk manager."""
    cfg = Config(symbol=symbol, start=START, initial_capital=equity)
    try:
        df = load_ohlcv(symbol, START, "", "1d")
        sig = Strategy(cfg.strategy).generate(df)
    except (DataError, ValueError) as exc:
        return Decision(symbol, "blocked", reason=str(exc)[:80])

    last = sig.data.iloc[-1]
    price = float(last["close"])
    atr = float(last["atr"])
    if symbol in held:
        return Decision(symbol, "hold", price=price, reason="position already open")
    if not bool(last["entry"]):
        return Decision(symbol, "flat", price=price, reason="no entry signal")
    if not (atr > 0):
        return Decision(symbol, "blocked", price=price, reason="ATR unavailable")

    stop = price - cfg.strategy.atr_stop_multiple * atr
    verdict = risk.evaluate_entry(equity=equity, cash=equity, price=price,
                                  stop_price=stop, open_positions=len(held))
    if not verdict.allowed:
        return Decision(symbol, "blocked", price=price, stop=stop,
                        reason=verdict.reason)
    return Decision(symbol, "buy", qty=verdict.quantity, price=price, stop=stop,
                    reason=verdict.reason)


def record(con: sqlite3.Connection, status: str, detail: dict) -> None:
    """Persist a broker run. Failures are recorded, never silently dropped."""
    con.execute("""
        CREATE TABLE IF NOT EXISTS broker_runs (
            id        INTEGER PRIMARY KEY AUTOINCREMENT,
            run_at    TEXT NOT NULL,
            status    TEXT NOT NULL,
            equity    REAL,
            detail    TEXT
        )
    """)
    con.execute("INSERT INTO broker_runs (run_at, status, equity, detail) VALUES (?,?,?,?)",
                (datetime.now(timezone.utc).isoformat(), status,
                 detail.get("equity"), json.dumps(detail)[:4000]))
    con.commit()


def write_note(status: str, equity: float | None, decisions: list[Decision],
               placed: list[dict], note: str = "") -> None:
    """Append the broker run to today's Obsidian note."""
    try:
        d = journal.journal_dir()
        d.mkdir(parents=True, exist_ok=True)
        day = datetime.now().strftime("%Y-%m-%d")
        path = d / f"{day}.md"
        if not path.exists():
            journal._ensure_note(path, day)
        with path.open("a", encoding="utf-8") as fh:
            fh.write(f"### {datetime.now().strftime('%H:%M:%S')} — ALPACA PAPER "
                     f"`[{status}]`\n")
            if equity is not None:
                fh.write(f"- **Account equity:** `{equity:,.2f}` (paper)\n")
            if note:
                fh.write(f"- {note}\n")
            for o in placed:
                fh.write(f"- **ORDER** `{o.get('symbol')}` qty `{o.get('qty')}` "
                         f"id `{o.get('id','?')[:12]}` — paper\n")
            blocked = [d_ for d_ in decisions if d_.action == "blocked"]
            if blocked:
                fh.write(f"- blocked: " +
                         ", ".join(f"`{b.symbol}` ({b.reason[:40]})" for b in blocked[:5])
                         + "\n")
            fh.write("\n")
    except OSError as exc:
        print(f"  ! obsidian write failed: {exc}", file=sys.stderr)


def main() -> int:
    """Connect, decide, and optionally place paper orders."""
    ap = argparse.ArgumentParser(description="Alpaca paper trading")
    ap.add_argument("--check", action="store_true", help="connectivity only")
    ap.add_argument("--dry-run", action="store_true", help="decide but place nothing")
    ap.add_argument("--symbols", nargs="*", default=None)
    args = ap.parse_args()

    con = init_db()
    try:
        try:
            api = AlpacaPaper()
        except UnsafeEndpoint as exc:
            print(f"REFUSED: {exc}")
            record(con, "REFUSED_UNSAFE_ENDPOINT", {"error": str(exc)})
            return 3
        except NotConfigured as exc:
            print(f"NOT_CONFIGURED: {exc}")
            record(con, "NOT_CONFIGURED", {"error": str(exc)})
            write_note("NOT_CONFIGURED", None, [], [], str(exc)[:160])
            return 2

        acct = api.account()
        equity = float(acct.get("equity", 0.0))
        clock = api.clock()
        held = {p["symbol"] for p in api.positions()}
        print(f"paper account  equity {equity:,.2f}  status {acct.get('status')}  "
              f"market {'OPEN' if clock.get('is_open') else 'closed'}")
        print(f"open positions {len(held)}  {sorted(held) or '—'}")

        if args.check:
            record(con, "CHECK_OK", {"equity": equity, "positions": sorted(held)})
            return 0

        risk = RiskManager(Config().risk, equity)
        decisions = [decide(s, equity, held, risk)
                     for s in (args.symbols or WATCHLIST)]

        print(f"\n{'symbol':<8}{'action':<10}{'qty':>8}{'price':>10}{'stop':>10}  reason")
        for d_ in decisions:
            print(f"{d_.symbol:<8}{d_.action:<10}{d_.qty:>8.0f}{d_.price:>10.2f}"
                  f"{d_.stop:>10.2f}  {d_.reason[:52]}")

        buys = [d_ for d_ in decisions if d_.action == "buy"]
        if not buys:
            print("\nno entry signals today — nothing to place")
            record(con, "NO_SIGNALS", {"equity": equity})
            write_note("NO SIGNALS", equity, decisions, [])
            return 0

        if args.dry_run:
            print(f"\n--dry-run: {len(buys)} order(s) NOT sent")
            return 0

        if not clock.get("is_open"):
            print("\nmarket closed — orders would queue to the next open; not placing")
            record(con, "MARKET_CLOSED", {"equity": equity})
            write_note("MARKET CLOSED", equity, decisions, [])
            return 0

        placed = []
        for d_ in buys:
            try:
                o = api.submit_bracket(d_.symbol, d_.qty, d_.stop)
                placed.append({"symbol": d_.symbol, "qty": d_.qty, "id": o.get("id", "")})
                print(f"  PLACED {d_.symbol} x{d_.qty:.0f} stop {d_.stop:.2f}")
            except RuntimeError as exc:
                print(f"  ! {d_.symbol}: {exc}", file=sys.stderr)

        record(con, "PLACED", {"equity": equity, "orders": placed})
        write_note("ORDERS PLACED", equity, decisions, placed)
        print(f"\n{len(placed)} paper order(s) placed — no real money involved")
        return 0
    finally:
        con.close()


if __name__ == "__main__":
    sys.exit(main())
