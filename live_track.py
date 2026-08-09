"""Real-time tracking for the Alpaca **paper** account.

Everything else in this project is a snapshot: the collector runs once at 18:00,
the dashboard replays a ledger of closed round-trips. Nothing watched the account
*while it was live*. This does.

Each tick reads three independent sources and records them side by side:

* **Broker** — Alpaca paper: equity, cash, open positions, resting orders. These
  are real fills against a real matching engine, so partial fills, rejects and
  market-hours behaviour are exercised for real. No real money can reach it: the
  adapter refuses any endpoint without ``paper-api`` in the URL.
* **Market** — Alpaca's snapshot feed, one call for the whole watchlist.
* **TradingView** — the charted symbol, read over CDP, used as a *cross-check*.
  Two independent feeds disagreeing means one is wrong and everything downstream
  inherits it.

Design rules carried over from the rest of the stack, and the reason for them:

* **A missing source is never a zero.** Each source records its own status
  (``OK`` / ``UNAVAILABLE`` / ``NOT_CONFIGURED``) and writes ``NULL``, so a dead
  feed can never be mistaken for a flat account. The collector already learned
  this the hard way with ``TV_UNAVAILABLE``.
* **The benchmark is anchored once and persisted.** Equal-weight buy-and-hold of
  the same watchlist, from the same starting capital, at the session's first
  tick. Restarting the tracker resumes the existing anchor rather than quietly
  rebasing it — a benchmark you can reset is not a benchmark.
* **Obsidian is not spammed.** A 30s loop would write 1,200 lines a day. Only
  session start, position changes and session end are journalled.

Notifications answer "did this trade make or lose money", which is an *account*
question, not a chart question. TradingView's own alerts are price-level triggers
and cannot report P&L, so they are not used for it.

* **Position closed** → realised P&L, taken from the actual closing fill where
  Alpaca reports one, and clearly labelled ``~`` when it falls back to the last
  mark instead. A number that might be an estimate must say so.
* **Open position crosses ±threshold** → fires once per position per direction,
  not once per tick. A 30s loop with an unthrottled alert is a machine for
  teaching you to ignore alerts.

    python live_track.py --once             # one tick, print, exit
    python live_track.py                    # follow, 30s
    python live_track.py --interval 10
    python live_track.py --pnl-alert 2      # toast at +/-2% unrealised
    python live_track.py --no-notify
    python live_track.py --reset-session    # re-anchor the benchmark
"""
from __future__ import annotations

import argparse
import json
import os
import sqlite3
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone

import httpx

import journal
import tv_cdp
from collector import init_db
from paper_broker import WATCHLIST, AlpacaPaper, NotConfigured, UnsafeEndpoint

# The Windows console defaults to cp1252, which cannot encode the arrows and
# dashes used in the panel and the journal. Unhandled, that raises
# UnicodeEncodeError mid-loop and kills a tracker running as a hidden task —
# a crash caused entirely by decoration. Degrade the character, never the run.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, OSError):
        pass

DATA_BASE = "https://data.alpaca.markets"
# Alpaca's free tier allows 200 req/min. 30s across two endpoints is nowhere
# near it, and polling faster does not make a daily strategy more informative.
DEFAULT_INTERVAL = 30


@dataclass
class Tick:
    """One observation of every source, with per-source status."""

    tick_at: str
    broker_status: str = "UNKNOWN"
    equity: float | None = None
    cash: float | None = None
    positions: int | None = None
    open_orders: int | None = None
    held: list[str] = field(default_factory=list)
    # symbol -> {qty, entry, price, pl, plpc} for open positions this tick
    pos: dict[str, dict[str, float]] = field(default_factory=dict)
    market_status: str = "UNKNOWN"
    prices: dict[str, float] = field(default_factory=dict)
    tv_status: str = "UNKNOWN"
    tv_symbol: str | None = None
    tv_close: float | None = None
    bench_status: str = "UNKNOWN"
    bench_value: float | None = None
    capture: float | None = None
    crosscheck: str = "SKIPPED"
    note: str = ""


class MarketData:
    """Alpaca market-data client. Same credentials, read-only endpoints."""

    def __init__(self, headers: dict[str, str]) -> None:
        self._h = headers

    def snapshots(self, symbols: list[str]) -> dict[str, float]:
        """Latest trade price per symbol. Symbols without a trade are omitted.

        Omission is deliberate: a symbol with no print should be absent from the
        mark, not carried at zero.
        """
        r = httpx.get(f"{DATA_BASE}/v2/stocks/snapshots", headers=self._h,
                      params={"symbols": ",".join(symbols)}, timeout=20)
        r.raise_for_status()
        out: dict[str, float] = {}
        for sym, snap in (r.json() or {}).items():
            trade = (snap or {}).get("latestTrade") or {}
            px = trade.get("p")
            if isinstance(px, (int, float)) and px > 0:
                out[sym] = float(px)
        return out


def ensure_tables(con: sqlite3.Connection) -> None:
    """Create the tracking tables if absent. Safe to call every run."""
    con.execute("""
        CREATE TABLE IF NOT EXISTS live_session (
            session_id    TEXT PRIMARY KEY,
            started_at    TEXT NOT NULL,
            anchor_equity REAL,
            anchor_prices TEXT,
            note          TEXT
        )
    """)
    con.execute("""
        CREATE TABLE IF NOT EXISTS live_ticks (
            id            INTEGER PRIMARY KEY AUTOINCREMENT,
            tick_at       TEXT NOT NULL,
            session_id    TEXT NOT NULL,
            broker_status TEXT NOT NULL,
            equity        REAL,
            cash          REAL,
            positions     INTEGER,
            open_orders   INTEGER,
            market_status TEXT NOT NULL,
            tv_status     TEXT NOT NULL,
            tv_symbol     TEXT,
            tv_close      REAL,
            bench_status  TEXT NOT NULL,
            bench_value   REAL,
            capture       REAL,
            crosscheck    TEXT,
            detail        TEXT
        )
    """)
    con.commit()


def load_session(con: sqlite3.Connection) -> tuple[str, float | None, dict[str, float]]:
    """Return the open session's anchor, creating nothing.

    Returns:
        ``(session_id, anchor_equity, anchor_prices)``. The id is empty when no
        session has been anchored yet.
    """
    row = con.execute(
        "SELECT session_id, anchor_equity, anchor_prices FROM live_session "
        "ORDER BY started_at DESC LIMIT 1").fetchone()
    if not row:
        return "", None, {}
    prices = {}
    if row[2]:
        try:
            prices = json.loads(row[2])
        except json.JSONDecodeError:
            prices = {}
    return row[0], row[1], prices


def anchor_session(con: sqlite3.Connection, equity: float,
                   prices: dict[str, float]) -> str:
    """Record the benchmark anchor. Called once, when both sources first agree."""
    sid = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    con.execute(
        "INSERT INTO live_session (session_id, started_at, anchor_equity, "
        "anchor_prices, note) VALUES (?,?,?,?,?)",
        (sid, datetime.now(timezone.utc).isoformat(), equity,
         json.dumps(prices), f"equal-weight buy&hold of {len(prices)} symbols"))
    con.commit()
    return sid


def benchmark(anchor_equity: float, anchor_prices: dict[str, float],
              now_prices: dict[str, float]) -> tuple[str, float | None]:
    """Mark equal-weight buy-and-hold of the anchor basket at current prices.

    Only symbols priced at *both* ends are marked; the rest keep their anchor
    weight in cash. Dropping a symbol entirely would silently rebase the
    benchmark onto whatever still quotes, which flatters the strategy.
    """
    if not anchor_prices or anchor_equity is None or anchor_equity <= 0:
        return "NO_ANCHOR", None
    per = anchor_equity / len(anchor_prices)
    total = 0.0
    marked = 0
    for sym, px0 in anchor_prices.items():
        px1 = now_prices.get(sym)
        if px0 and px0 > 0 and isinstance(px1, (int, float)) and px1 > 0:
            total += per * (px1 / px0)
            marked += 1
        else:
            total += per          # unpriced leg held at cost, not dropped
    if marked == 0:
        return "UNPRICED", None
    return ("OK" if marked == len(anchor_prices) else f"PARTIAL_{marked}"
            f"/{len(anchor_prices)}"), total


def capture_of(equity: float | None, bench: float | None,
               anchor: float | None) -> float | None:
    """Strategy gain as a fraction of the benchmark's gain.

    Undefined when the benchmark is flat — dividing by a near-zero denominator
    produces a spectacular number that means nothing. Returning ``None`` is the
    honest answer.
    """
    if equity is None or bench is None or not anchor or anchor <= 0:
        return None
    bench_gain = bench - anchor
    if abs(bench_gain) < anchor * 1e-4:
        return None
    return (equity - anchor) / bench_gain


def read_broker(tick: Tick) -> AlpacaPaper | None:
    """Fill the broker fields. Returns the client for reuse, or None."""
    try:
        api = AlpacaPaper()
    except UnsafeEndpoint as exc:
        tick.broker_status = "REFUSED_UNSAFE_ENDPOINT"
        tick.note = str(exc)[:120]
        return None
    except NotConfigured:
        tick.broker_status = "NOT_CONFIGURED"
        return None
    try:
        acct = api.account()
        pos = api.positions()
        orders = api._get("/v2/orders", status="open")
    except (httpx.HTTPError, OSError) as exc:
        tick.broker_status = "UNREACHABLE"
        tick.note = f"{type(exc).__name__}: {str(exc)[:100]}"
        return None
    tick.broker_status = "OK"
    tick.equity = float(acct.get("equity") or 0.0)
    tick.cash = float(acct.get("cash") or 0.0)
    tick.held = sorted(p["symbol"] for p in pos)
    for p in pos:
        def num(k: str) -> float:
            try:
                return float(p.get(k) or 0.0)
            except (TypeError, ValueError):
                return 0.0
        tick.pos[p["symbol"]] = {
            "qty": num("qty"), "entry": num("avg_entry_price"),
            "price": num("current_price"), "pl": num("unrealized_pl"),
            "plpc": num("unrealized_plpc") * 100.0,
        }
    tick.positions = len(pos)
    tick.open_orders = len(orders) if isinstance(orders, list) else 0
    return api


def read_tradingview(tick: Tick) -> None:
    """Fill the TradingView fields. Absence is recorded, never inferred."""
    try:
        st = tv_cdp.read_state()
    except tv_cdp.CDPError as exc:
        msg = str(exc)
        # dataSize 0 is a distinct failure with a distinct fix (reload the
        # chart). Collapsing it into UNAVAILABLE cost a long debugging session.
        tick.tv_status = "DATASIZE_ZERO" if "dataSize is 0" in msg else "UNAVAILABLE"
        tick.note = (tick.note + " | " if tick.note else "") + msg[:100]
        return
    tick.tv_status = "OK"
    tick.tv_symbol = st.get("symbol")
    close = st.get("close")
    tick.tv_close = float(close) if isinstance(close, (int, float)) else None


def cross_check(tick: Tick) -> None:
    """Compare the two independent feeds on the charted symbol."""
    if tick.tv_status != "OK" or tick.tv_close is None or not tick.prices:
        return
    sym = (tick.tv_symbol or "").split(":")[-1]
    other = tick.prices.get(sym)
    if not other or other <= 0:
        tick.crosscheck = f"NO_PAIR_{sym}"
        return
    drift = abs(tick.tv_close - other) / other
    # 0.5% is loose enough for a delayed IEX print against a consolidated
    # chart close, tight enough to catch a genuinely wrong symbol or a stale
    # feed. Overnight the two legitimately differ; that is what the tag says.
    tick.crosscheck = f"OK_{drift * 100:.2f}%" if drift < 0.005 else \
                      f"DIVERGENT_{drift * 100:.2f}%"


def record(con: sqlite3.Connection, sid: str, t: Tick) -> None:
    """Persist one tick."""
    con.execute("""
        INSERT INTO live_ticks (tick_at, session_id, broker_status, equity, cash,
            positions, open_orders, market_status, tv_status, tv_symbol, tv_close,
            bench_status, bench_value, capture, crosscheck, detail)
        VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
    """, (t.tick_at, sid, t.broker_status, t.equity, t.cash, t.positions,
          t.open_orders, t.market_status, t.tv_status, t.tv_symbol, t.tv_close,
          t.bench_status, t.bench_value, t.capture, t.crosscheck,
          json.dumps({"held": t.held, "note": t.note})[:2000]))
    con.commit()


def collect(con: sqlite3.Connection, sid: str, anchor_eq: float | None,
            anchor_px: dict[str, float], symbols: list[str]
            ) -> tuple[Tick, str, float | None, dict[str, float], AlpacaPaper | None]:
    """Run one full tick across every source and persist it."""
    t = Tick(tick_at=datetime.now(timezone.utc).isoformat())
    api = read_broker(t)

    if api is not None:
        try:
            t.prices = MarketData(api._h).snapshots(symbols)
            t.market_status = "OK" if t.prices else "NO_PRICES"
        except (httpx.HTTPError, OSError) as exc:
            t.market_status = "UNREACHABLE"
            t.note = (t.note + " | " if t.note else "") + str(exc)[:80]
    else:
        t.market_status = "NO_CREDENTIALS"

    read_tradingview(t)
    cross_check(t)

    # Anchor on the first tick where both the account and the marks are real.
    if not sid and t.broker_status == "OK" and t.prices:
        anchor_eq, anchor_px = t.equity, dict(t.prices)
        sid = anchor_session(con, anchor_eq, anchor_px)
        journal_event(f"SESSION START `{sid}`", t,
                      f"benchmark anchored: equal-weight buy&hold of "
                      f"{len(anchor_px)} symbols at {anchor_eq:,.2f}")

    t.bench_status, t.bench_value = benchmark(anchor_eq, anchor_px, t.prices)
    t.capture = capture_of(t.equity, t.bench_value, anchor_eq)
    record(con, sid or "UNANCHORED", t)
    return t, sid, anchor_eq, anchor_px, api


def journal_event(title: str, t: Tick, detail: str) -> None:
    """Append a notable event to today's Obsidian note. Never called per tick."""
    try:
        d = journal.journal_dir()
        d.mkdir(parents=True, exist_ok=True)
        day = datetime.now().strftime("%Y-%m-%d")
        path = d / f"{day}.md"
        if not path.exists():
            journal._ensure_note(path, day)
        with path.open("a", encoding="utf-8") as fh:
            fh.write(f"### {datetime.now().strftime('%H:%M:%S')} — LIVE TRACK "
                     f"`[{title}]`\n- {detail}\n")
            if t.equity is not None:
                fh.write(f"- **Equity:** `{t.equity:,.2f}` (paper) · "
                         f"positions `{t.positions}` · held "
                         f"`{', '.join(t.held) or '—'}`\n")
            fh.write("\n")
    except OSError as exc:
        print(f"  ! obsidian write failed: {exc}", file=sys.stderr)


# --------------------------------------------------------------- money & FX

_FX: dict[str, tuple[float, float]] = {}
FX_TTL = 1800          # 30 min; an FX rate does not move enough in a 30s loop


def usd_eur() -> tuple[float | None, str]:
    """Indicative USD→EUR rate, cached.

    The paper account is USD-denominated. This is a **display** conversion, not
    a broker rate — a real conversion carries a spread this does not model,
    which is why every euro figure is printed with ``≈``. A stale cached rate is
    returned in preference to nothing, tagged ``STALE`` so it cannot be mistaken
    for a live one.
    """
    now = time.time()
    hit = _FX.get("EURUSD")
    if hit and now - hit[1] < FX_TTL:
        return hit[0], "OK"
    try:
        import yfinance as yf
        h = yf.Ticker("EURUSD=X").history(period="5d", interval="1d")
        eurusd = float(h["Close"].iloc[-1]) if h is not None and not h.empty else 0.0
    except Exception as exc:                       # noqa: BLE001 - reported, not swallowed
        print(f"  ! FX unavailable: {type(exc).__name__}: {str(exc)[:60]}",
              file=sys.stderr)
        return (hit[0] if hit else None), ("STALE" if hit else "UNAVAILABLE")
    if not (eurusd > 0):
        return (hit[0] if hit else None), ("STALE" if hit else "UNAVAILABLE")
    rate = 1.0 / eurusd
    _FX["EURUSD"] = (rate, now)
    return rate, "OK"


def money(pct: float | None, usd: float | None, approx: bool = False) -> str:
    """Render a P&L three ways: percent, dollars, euros.

    Args:
        pct: Percentage move, already in percent (not a fraction).
        usd: Absolute P&L in account currency.
        approx: True when *usd* is a mark rather than a settled fill. Prefixes
            ``~`` so an estimate is never read as a realised number.
    """
    tilde = "~" if approx else ""
    parts: list[str] = []
    if pct is not None:
        parts.append(f"{tilde}{pct:+.2f}%")
    if usd is not None:
        parts.append(f"{tilde}${usd:+,.2f}")
        rate, st = usd_eur()
        # No rate means no euro figure. Printing EUR 0.00 would be a lie.
        # ASCII only: a toast and a cp1252 console must both render this.
        parts.append(f"EUR{usd * rate:+,.2f}" + ("*" if st != "OK" else "")
                     if rate else "EUR n/a")
    return "  ".join(parts) if parts else "n/a"


def _powershell() -> str:
    """Absolute path to powershell.exe.

    Resolving it by name fails with WinError 2 whenever the spawning
    environment lacks ``System32`` on PATH — which is how this project is
    launched from Git Bash, from the scheduler, and from the MCP server. The
    same omission is what breaks TradingView's own ``tv_launch``. Never rely on
    PATH for a Windows system binary.
    """
    root = os.environ.get("SYSTEMROOT") or r"C:\Windows"
    exe = os.path.join(root, "System32", "WindowsPowerShell", "v1.0",
                       "powershell.exe")
    return exe if os.path.exists(exe) else "powershell"


def _q(s: str) -> str:
    """Single-quote a string for PowerShell, escaping embedded quotes."""
    return "'" + str(s).replace("'", "''") + "'"


def notify(title: str, message: str, enabled: bool = True) -> None:
    """Raise a Windows toast. Failure here must never interrupt tracking."""
    print(f"  >> NOTIFY  {title} — {message}")
    if not enabled:
        return
    ps = ("[reflection.assembly]::LoadWithPartialName('System.Windows.Forms')>$null;"
          "[reflection.assembly]::LoadWithPartialName('System.Drawing')>$null;"
          "$n=New-Object System.Windows.Forms.NotifyIcon;"
          "$n.Icon=[System.Drawing.SystemIcons]::Information;"
          f"$n.BalloonTipTitle={_q(title)};$n.BalloonTipText={_q(message)};"
          "$n.Visible=$true;$n.ShowBalloonTip(20000);"
          "Start-Sleep -Seconds 8;$n.Dispose()")
    try:
        subprocess.Popen(
            [_powershell(), "-NoProfile", "-NonInteractive", "-Command", ps],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    except OSError as exc:
        print(f"  ! toast failed: {exc}", file=sys.stderr)


def closing_fill(api: AlpacaPaper, symbol: str) -> float | None:
    """Average price of the most recent sell fill for *symbol*, if Alpaca has one.

    Preferred over the last mark because a stop can fill well away from the last
    price we happened to poll — which is exactly the case where an accurate
    number matters most.
    """
    try:
        acts = api._get("/v2/account/activities/FILL", page_size=50)
    except (httpx.HTTPError, OSError):
        return None
    if not isinstance(acts, list):
        return None
    for a in acts:                       # newest first
        if a.get("symbol") == symbol and a.get("side", "").startswith("sell"):
            try:
                return float(a.get("price"))
            except (TypeError, ValueError):
                return None
    return None


def render(t: Tick, anchor: float | None) -> str:
    """One compact panel per tick."""
    hhmm = datetime.now().strftime("%H:%M:%S")
    lines = [f"[{hhmm}] broker {t.broker_status:<16} market {t.market_status:<14} "
             f"tv {t.tv_status}"]
    if t.equity is not None:
        pnl = (t.equity - anchor) if anchor else None
        pct = (pnl / anchor * 100) if (pnl is not None and anchor) else None
        lines.append(f"  equity   {t.equity:>12,.2f}" +
                     (f"   {pnl:+,.2f} ({pct:+.2f}%)" if pct is not None else "") +
                     f"   cash {t.cash:,.2f}")
        lines.append(f"  position {t.positions or 0:>12}   orders {t.open_orders or 0}"
                     f"   {', '.join(t.held) or '—'}")
        for sym, p in sorted(t.pos.items()):
            lines.append(f"    {sym:<6} {p['qty']:>7.0f} sh @ {p['entry']:>10,.2f}"
                         f"   {money(p['plpc'], p['pl'], approx=True)}")
    if t.bench_value is not None and anchor:
        bp = (t.bench_value - anchor) / anchor * 100
        cap = f"{t.capture * 100:6.1f}%" if t.capture is not None else "   n/a"
        lines.append(f"  buy&hold {t.bench_value:>12,.2f}   {bp:+.2f}%"
                     f"   [{t.bench_status}]   capture {cap}")
    if t.tv_close is not None:
        lines.append(f"  tv       {t.tv_symbol} {t.tv_close:,.2f}"
                     f"   crosscheck {t.crosscheck}")
    if t.note:
        lines.append(f"  ! {t.note}")
    return "\n".join(lines)


def main() -> int:
    """Follow the paper account until interrupted."""
    ap = argparse.ArgumentParser(description="Real-time paper account tracking")
    ap.add_argument("--once", action="store_true", help="single tick, then exit")
    ap.add_argument("--interval", type=int, default=DEFAULT_INTERVAL,
                    help=f"seconds between ticks (default {DEFAULT_INTERVAL})")
    ap.add_argument("--symbols", nargs="*", default=None)
    ap.add_argument("--reset-session", action="store_true",
                    help="drop the benchmark anchor and re-anchor on next tick")
    ap.add_argument("--pnl-alert", type=float, default=2.0, metavar="PCT",
                    help="toast when an open position crosses +/-PCT (0 = off)")
    ap.add_argument("--no-notify", dest="notify", action="store_false",
                    help="print events but raise no desktop toasts")
    args = ap.parse_args()

    symbols = args.symbols or WATCHLIST
    con = init_db()
    try:
        ensure_tables(con)
        if args.reset_session:
            con.execute("DELETE FROM live_session")
            con.commit()
            print("benchmark anchor cleared — will re-anchor on the next tick")

        sid, anchor_eq, anchor_px = load_session(con)
        if sid:
            print(f"resuming session {sid}  anchor {anchor_eq:,.2f}  "
                  f"{len(anchor_px)} symbols")

        last_held: list[str] | None = None
        last_pos: dict[str, dict[str, float]] = {}
        # symbol -> set of thresholds already fired, so a position hovering at
        # +2.01% cannot toast every 30 seconds.
        fired: dict[str, set[str]] = {}
        ticks = 0
        while True:
            t, sid, anchor_eq, anchor_px, api = collect(con, sid, anchor_eq,
                                                        anchor_px, symbols)
            ticks += 1
            print(render(t, anchor_eq))

            if t.broker_status == "OK":
                # --- closed positions: the profit-or-loss notification -------
                for sym, prev in last_pos.items():
                    if sym in t.pos:
                        continue
                    exit_px = closing_fill(api, sym) if api else None
                    if exit_px and prev["entry"] > 0:
                        pl = (exit_px - prev["entry"]) * prev["qty"]
                        pct = (exit_px / prev["entry"] - 1) * 100
                        est = False
                    else:
                        # No settled fill visible — fall back to the last mark
                        # and say so, rather than quietly reporting it as final.
                        pl, pct, est = prev["pl"], prev["plpc"], True
                    verdict = "PROFIT" if pl >= 0 else "LOSS"
                    notify(f"{verdict} — {sym} closed",
                           f"{money(pct, pl, approx=est)}"
                           f"   {prev['qty']:.0f} sh @ {prev['entry']:,.2f}",
                           args.notify)
                    journal_event(f"{verdict} — {sym} CLOSED", t,
                                  f"{money(pct, pl, approx=est)} · "
                                  f"`{prev['qty']:.0f}` sh from "
                                  f"`{prev['entry']:,.2f}`"
                                  + (" · _estimated from last mark_" if est else ""))
                    fired.pop(sym, None)

                # --- opened positions ---------------------------------------
                for sym, cur in t.pos.items():
                    if sym not in last_pos:
                        notify(f"OPENED — {sym}",
                               f"{cur['qty']:.0f} sh @ {cur['entry']:,.2f}",
                               args.notify)

                # --- open P&L crossing the threshold ------------------------
                for sym, cur in t.pos.items():
                    if args.pnl_alert <= 0:
                        break
                    side = ("up" if cur["plpc"] >= args.pnl_alert else
                            "down" if cur["plpc"] <= -args.pnl_alert else None)
                    if side and side not in fired.setdefault(sym, set()):
                        fired[sym].add(side)
                        verdict = "PROFIT" if side == "up" else "LOSS"
                        notify(f"{verdict} — {sym} open {cur['plpc']:+.2f}%",
                               money(cur["plpc"], cur["pl"], approx=True),
                               args.notify)

                if last_held is not None and t.held != last_held:
                    journal_event("POSITION CHANGE", t,
                                  f"`{', '.join(last_held) or '—'}` → "
                                  f"`{', '.join(t.held) or '—'}`")
                last_held, last_pos = t.held, t.pos

            if args.once:
                return 0
            time.sleep(max(5, args.interval))
    except KeyboardInterrupt:
        print(f"\nstopped after {ticks} tick(s)")
        return 0
    finally:
        con.close()


if __name__ == "__main__":
    sys.exit(main())
