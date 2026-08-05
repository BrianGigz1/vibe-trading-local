"""Import a TradingView paper-trading history export into the paper ledger.

Why this matters more than the collector's own trades: the collector replays a
mechanical strategy, so it measures the *system*. A TradingView paper export
records decisions **you** made — entry timing, when you cut a loser, when you
took profit. That is the only dataset here that says anything about your
behaviour, and it is what the disposition-effect analysis actually needs.

TradingView's export column names vary by account locale and version, so the
parser matches on normalised substrings rather than exact headers, and reports
what it mapped. A silent mismap would corrupt the ledger.

Export path in TradingView:
    Trading Panel -> History (or Account Manager) -> export / copy to CSV

    python import_tradingview.py trades.csv
    python import_tradingview.py trades.csv --dry-run
"""
from __future__ import annotations

import argparse
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

import journal
from collector import init_db, stats, write_obsidian

# Candidate substrings for each field, in priority order.
COLUMN_HINTS: dict[str, tuple[str, ...]] = {
    "symbol": ("symbol", "ticker", "instrument"),
    "side": ("side", "action", "type", "direction"),
    "qty": ("qty", "quantity", "size", "amount", "contracts"),
    "price": ("price", "fill", "avg"),
    "time": ("time", "date", "closing", "closed"),
    "pnl": ("profit", "pnl", "p&l", "realized", "net"),
}


def map_columns(cols: list[str]) -> dict[str, str | None]:
    """Map logical fields to actual column names by normalised substring."""
    norm = {c: c.strip().lower().replace("_", " ") for c in cols}
    out: dict[str, str | None] = {}
    used: set[str] = set()
    for field, hints in COLUMN_HINTS.items():
        found = None
        for hint in hints:
            for col, low in norm.items():
                if col in used:
                    continue
                if hint in low:
                    found = col
                    break
            if found:
                break
        if found:
            used.add(found)
        out[field] = found
    return out


def load(path: Path) -> tuple[pd.DataFrame, dict[str, str | None]]:
    """Read the export and report the column mapping."""
    if not path.exists():
        raise FileNotFoundError(path)
    if path.suffix.lower() in (".xlsx", ".xls"):
        df = pd.read_excel(path)
    else:
        # TradingView exports are sometimes semicolon-separated by locale.
        df = pd.read_csv(path, sep=None, engine="python")
    if df.empty:
        raise ValueError("export contains no rows")
    return df, map_columns(list(df.columns))


def to_trades(df: pd.DataFrame, m: dict[str, str | None]) -> list[dict]:
    """Convert mapped rows into ledger records, pairing entries with exits."""
    if not m["symbol"] or not m["price"]:
        raise ValueError(f"could not identify symbol/price columns; mapped: {m}")

    rows = []
    for _, r in df.iterrows():
        side = str(r[m["side"]]).strip().lower() if m["side"] else ""
        rows.append({
            "symbol": str(r[m["symbol"]]).strip().upper(),
            "side": "buy" if "buy" in side or "long" in side else "sell",
            "qty": abs(float(r[m["qty"]])) if m["qty"] and pd.notna(r[m["qty"]]) else 0.0,
            "price": float(r[m["price"]]),
            "time": pd.to_datetime(r[m["time"]], errors="coerce") if m["time"] else pd.NaT,
            "pnl": float(r[m["pnl"]]) if m["pnl"] and pd.notna(r[m["pnl"]]) else None,
        })

    rows = [r for r in rows if pd.notna(r["time"])]
    rows.sort(key=lambda x: x["time"])

    # FIFO pair buys with sells, per symbol.
    open_lots: dict[str, list[dict]] = {}
    trades = []
    for r in rows:
        sym = r["symbol"]
        if r["side"] == "buy":
            open_lots.setdefault(sym, []).append(r)
            continue
        lots = open_lots.get(sym, [])
        remaining = r["qty"] or 0.0
        while remaining > 0 and lots:
            lot = lots[0]
            take = min(remaining, lot["qty"]) if lot["qty"] else remaining
            pnl = (r["price"] - lot["price"]) * take
            hold = max(1, (r["time"] - lot["time"]).days)
            trades.append({
                "symbol": sym,
                "entry_date": str(lot["time"].date()),
                "exit_date": str(r["time"].date()),
                "entry_price": lot["price"],
                "exit_price": r["price"],
                "quantity": take,
                "pnl": pnl,
                "return_pct": (r["price"] / lot["price"] - 1.0) if lot["price"] else 0.0,
                "hold_days": hold,
                "exit_reason": "tradingview_paper",
            })
            remaining -= take
            lot["qty"] -= take
            if lot["qty"] <= 0:
                lots.pop(0)
    return trades


def insert(con: sqlite3.Connection, trades: list[dict]) -> int:
    """Insert trades under mode 'tv_paper'; duplicates are skipped."""
    now = datetime.now(timezone.utc).isoformat()
    n = 0
    for t in trades:
        try:
            con.execute(
                "INSERT INTO trades (symbol, entry_date, exit_date, entry_price,"
                " exit_price, quantity, pnl, return_pct, hold_days, exit_reason,"
                " bh_return, edge, mode, collected_at)"
                " VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (t["symbol"], t["entry_date"], t["exit_date"], t["entry_price"],
                 t["exit_price"], t["quantity"], t["pnl"], t["return_pct"],
                 t["hold_days"], t["exit_reason"],
                 # Buy & hold over the same window is not derivable from the
                 # export alone; left at 0 and excluded from edge stats rather
                 # than invented. A fabricated benchmark is worse than none.
                 0.0, 0.0, "tv_paper", now),
            )
            n += 1
        except sqlite3.IntegrityError:
            pass
    con.commit()
    return n


def behaviour(trades: list[dict]) -> dict:
    """Disposition statistics over the imported trades."""
    wins = [t for t in trades if t["pnl"] and t["pnl"] > 0]
    losses = [t for t in trades if t["pnl"] is not None and t["pnl"] <= 0]
    wh = sum(t["hold_days"] for t in wins) / len(wins) if wins else 0.0
    lh = sum(t["hold_days"] for t in losses) / len(losses) if losses else 0.0
    return {
        "trades": len(trades), "wins": len(wins), "losses": len(losses),
        "win_hold": wh, "loss_hold": lh,
        "ratio": (lh / wh) if wh else 0.0,
        "total_pnl": sum(t["pnl"] or 0.0 for t in trades),
    }


def main() -> int:
    """Parse, report the mapping, then import."""
    ap = argparse.ArgumentParser(description="Import TradingView paper history")
    ap.add_argument("path")
    ap.add_argument("--dry-run", action="store_true", help="parse and report, do not write")
    args = ap.parse_args()

    try:
        df, m = load(Path(args.path))
    except (FileNotFoundError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    print(f"rows: {len(df)}")
    print("column mapping (verify before trusting the import):")
    for k, v in m.items():
        print(f"  {k:<8} -> {v or 'NOT FOUND'}")
    missing = [k for k, v in m.items() if v is None]
    if missing:
        print(f"  unmapped: {missing} — check the export's headers")

    try:
        trades = to_trades(df, m)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 3

    if not trades:
        print("\nNo completed round trips found. Open positions are not imported —")
        print("an unrealised position has no holding period or realised P&L.")
        return 0

    b = behaviour(trades)
    print(f"\nround trips: {b['trades']}  ({b['wins']}W / {b['losses']}L)")
    print(f"total P&L  : {b['total_pnl']:,.2f}")
    print(f"winners held {b['win_hold']:.1f}d, losers {b['loss_hold']:.1f}d "
          f"-> disposition ratio {b['ratio']:.2f}")
    if b["ratio"] >= 2.0:
        print("  ⚠️ >= 2.0: losers held far longer than winners — classic disposition effect.")
    elif b["ratio"] and b["ratio"] < 1.0:
        print("  losers cut faster than winners are held — the healthy direction.")

    if args.dry_run:
        print("\n--dry-run: nothing written")
        return 0

    con = init_db()
    n = insert(con, trades)
    print(f"\ninserted {n} new ({len(trades) - n} already present)")
    s = stats(con)
    write_obsidian(s, n, len({t['symbol'] for t in trades}))
    print(f"ledger now: {s['trades']} trades")
    print(f"Obsidian  : {journal.journal_dir().name}")
    con.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
