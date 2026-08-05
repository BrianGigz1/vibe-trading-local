"""Automated paper-trading data collector.

Runs the strategy across a watchlist, records every closed trade to a persistent
SQLite ledger, and writes an Obsidian daily note. Designed to be run on a
schedule and to accumulate a dataset over weeks and months.

Two design rules, both learned the hard way today:

**Idempotent.** Trades are keyed on (symbol, entry_date, exit_date) with a UNIQUE
constraint, so re-running never duplicates. A collector that double-counts on a
retry produces a dataset that silently overstates activity.

**Buy & hold is stored per trade.** Every row carries the benchmark return over
the same holding window. Without it the ledger can only answer "did I make
money", never "was trading worth doing" — and today proved those differ.

    python collector.py                 # collect + write Obsidian note
    python collector.py --stats         # summarise the ledger, no collection
    python collector.py --symbols X Y   # override the watchlist
"""
from __future__ import annotations

import argparse
import sqlite3
import sys
from datetime import date, datetime, timezone
from pathlib import Path

import journal
from backtest import Backtester
from config import Config
from data import DataError, load_ohlcv

DB = Path(__file__).resolve().parent / "paper_ledger.db"
WATCHLIST = ["AAPL", "MSFT", "NVDA", "SPY", "TSLA", "AMZN", "META", "GOOGL"]
START = "2023-01-01"


def connect_ro() -> sqlite3.Connection:
    """Open the ledger for READING only — no schema work, no write lock.

    `init_db` issues CREATE TABLE IF NOT EXISTS and commits, which takes a write
    lock even when nothing changes. The dashboard polls every 5 seconds, so using
    init_db there would contend with the collector's nightly writes. Readers must
    use this instead.
    """
    con = sqlite3.connect(f"file:{DB}?mode=ro", uri=True, timeout=10.0)
    con.execute("PRAGMA query_only = ON")
    return con


def init_db() -> sqlite3.Connection:
    """Open the ledger, creating the schema if needed."""
    con = sqlite3.connect(DB, timeout=30.0)
    # WAL lets the dashboard read while the collector writes. Without it a poll
    # landing on the 18:00 run raises "database is locked".
    con.execute("PRAGMA journal_mode = WAL")
    con.execute("PRAGMA busy_timeout = 30000")
    con.execute("PRAGMA synchronous = NORMAL")
    con.execute("""
        CREATE TABLE IF NOT EXISTS trades (
            id           INTEGER PRIMARY KEY AUTOINCREMENT,
            symbol       TEXT NOT NULL,
            entry_date   TEXT NOT NULL,
            exit_date    TEXT NOT NULL,
            entry_price  REAL NOT NULL,
            exit_price   REAL NOT NULL,
            quantity     REAL NOT NULL,
            pnl          REAL NOT NULL,
            return_pct   REAL NOT NULL,
            hold_days    INTEGER NOT NULL,
            exit_reason  TEXT NOT NULL,
            bh_return    REAL NOT NULL,   -- buy & hold over the same window
            edge         REAL NOT NULL,   -- return_pct - bh_return
            mode         TEXT NOT NULL,
            collected_at TEXT NOT NULL,
            UNIQUE(symbol, entry_date, exit_date, mode)
        )
    """)
    con.execute("""
        CREATE TABLE IF NOT EXISTS runs (
            id           INTEGER PRIMARY KEY AUTOINCREMENT,
            run_at       TEXT NOT NULL,
            symbols      INTEGER NOT NULL,
            new_trades   INTEGER NOT NULL,
            total_trades INTEGER NOT NULL
        )
    """)
    con.commit()
    return con


def collect(con: sqlite3.Connection, symbols: list[str], trailing: bool) -> int:
    """Run the strategy over *symbols* and insert any unseen closed trades."""
    mode = "trail" if trailing else "fixed"
    now = datetime.now(timezone.utc).isoformat()
    inserted = 0

    for sym in symbols:
        try:
            df = load_ohlcv(sym, START, "", "1d")
        except DataError as exc:
            print(f"  ! {sym}: {exc}", file=sys.stderr)
            continue

        cfg = Config(symbol=sym, start=START, initial_capital=100_000.0)
        try:
            res = Backtester(cfg, trailing=trailing).run(df)
        except ValueError as exc:
            print(f"  ! {sym}: {exc}", file=sys.stderr)
            continue

        closes = df["close"]
        for t in res.trades:
            ed, xd = t.entry_date, t.exit_date
            hold = max(1, (xd - ed).days)
            # Benchmark: holding the same symbol over the same window.
            try:
                bh = float(closes.loc[xd] / closes.loc[ed] - 1.0)
            except KeyError:
                bh = 0.0
            try:
                con.execute(
                    "INSERT INTO trades (symbol, entry_date, exit_date, entry_price,"
                    " exit_price, quantity, pnl, return_pct, hold_days, exit_reason,"
                    " bh_return, edge, mode, collected_at)"
                    " VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                    (sym, str(ed.date()), str(xd.date()), t.entry_price, t.exit_price,
                     t.quantity, t.pnl, t.return_pct, hold, t.reason, bh,
                     t.return_pct - bh, mode, now),
                )
                inserted += 1
            except sqlite3.IntegrityError:
                pass   # already collected — idempotent by design
        con.commit()
        print(f"  {sym:<6} {len(res.trades):>3} trades  "
              f"capture {res.total_return / res.buy_hold_return * 100 if res.buy_hold_return else 0:>6.1f}%")
    return inserted


def stats(con: sqlite3.Connection) -> dict:
    """Aggregate the ledger."""
    cur = con.execute("""
        SELECT COUNT(*), COALESCE(SUM(pnl),0), COALESCE(AVG(return_pct),0),
               COALESCE(AVG(bh_return),0), COALESCE(AVG(edge),0),
               COALESCE(AVG(hold_days),0),
               COALESCE(SUM(CASE WHEN pnl > 0 THEN 1 ELSE 0 END),0),
               COALESCE(SUM(CASE WHEN edge > 0 THEN 1 ELSE 0 END),0)
        FROM trades
    """)
    n, pnl, avg_ret, avg_bh, avg_edge, avg_hold, wins, beats = cur.fetchone()

    # Disposition effect: are losers held longer than winners?
    cur = con.execute("SELECT AVG(hold_days) FROM trades WHERE pnl > 0")
    win_hold = cur.fetchone()[0] or 0.0
    cur = con.execute("SELECT AVG(hold_days) FROM trades WHERE pnl <= 0")
    loss_hold = cur.fetchone()[0] or 0.0

    return {
        "trades": n, "total_pnl": pnl, "avg_return": avg_ret, "avg_bh": avg_bh,
        "avg_edge": avg_edge, "avg_hold": avg_hold,
        "win_rate": wins / n if n else 0.0,
        "beat_bh_rate": beats / n if n else 0.0,
        "winner_hold": win_hold, "loser_hold": loss_hold,
        "disposition_ratio": (loss_hold / win_hold) if win_hold else 0.0,
    }


def write_obsidian(s: dict, new_trades: int, symbols: int) -> Path | None:
    """Append a collection summary to today's Obsidian journal note."""
    try:
        d = journal.journal_dir()
        d.mkdir(parents=True, exist_ok=True)
        day = date.today().isoformat()
        path = d / f"{day}.md"
        if not path.exists():
            journal._ensure_note(path, day)

        disp = s["disposition_ratio"]
        flag = "⚠️ HIGH" if disp >= 2.0 else ("elevated" if disp >= 1.5 else "ok")
        with path.open("a", encoding="utf-8") as fh:
            fh.write(
                f"### {datetime.now().strftime('%H:%M:%S')} — COLLECTOR RUN `[SIMULATED]`\n"
                f"- **New trades:** `{new_trades}` across `{symbols}` symbols\n"
                f"- **Ledger total:** `{s['trades']}` trades | "
                f"cumulative P&L `{s['total_pnl']:,.2f}`\n"
                f"- **Avg return:** `{s['avg_return']:+.2%}` vs "
                f"**buy & hold** `{s['avg_bh']:+.2%}` → "
                f"**edge `{s['avg_edge']:+.2%}`**\n"
                f"- **Win rate:** `{s['win_rate']:.1%}` | "
                f"**beat buy & hold:** `{s['beat_bh_rate']:.1%}` ← the one that matters\n"
                f"- **Disposition:** winners held `{s['winner_hold']:.1f}d`, "
                f"losers `{s['loser_hold']:.1f}d` "
                f"(ratio `{disp:.2f}` — {flag})\n\n"
            )
        return path
    except OSError as exc:
        print(f"  ! obsidian write failed (non-fatal): {exc}", file=sys.stderr)
        return None


def main() -> int:
    """Collect, summarise, and journal."""
    ap = argparse.ArgumentParser(description="Paper-trading data collector")
    ap.add_argument("--symbols", nargs="*", default=None)
    ap.add_argument("--fixed", action="store_true", help="fixed stop instead of trailing")
    ap.add_argument("--stats", action="store_true", help="report ledger only")
    ap.add_argument("--once-per-day", action="store_true",
                    help="exit quietly if a successful run already happened today")
    args = ap.parse_args()

    con = init_db()
    try:
        return _run(con, args)
    finally:
        # A raise inside collect() previously skipped con.close(), leaving the
        # handle — and under the old journal mode, a lock — held until process
        # exit. The next scheduled run would then contend with it.
        con.close()


def _run(con: sqlite3.Connection, args: argparse.Namespace) -> int:
    """Collection body. The caller owns the connection and closes it."""
    # A logon trigger can fire several times a day (reboots, fast user switching,
    # waking from a missed schedule). The collection itself is idempotent, but
    # the Obsidian note is append-only — without this guard a laptop rebooted
    # three times would write three identical summaries into the daily note.
    if args.once_per_day and not args.stats:
        # run_at is stored in UTC, so the comparison must be UTC too. Using
        # date.today() (local) broke for any logon between midnight and the UTC
        # offset — in CEST that is 00:00-02:00, where the local date has rolled
        # over but the UTC date has not, so the guard found nothing and the
        # collector ran a second time, duplicating the Obsidian entry.
        today = datetime.now(timezone.utc).date().isoformat()
        row = con.execute(
            "SELECT run_at FROM runs WHERE substr(run_at,1,10)=? ORDER BY id DESC LIMIT 1",
            (today,),
        ).fetchone()
        if row:
            print(f"already collected today at {row[0]} — skipping")
            return 0
    symbols = args.symbols or WATCHLIST
    new_trades = 0

    if not args.stats:
        print(f"collecting {len(symbols)} symbols from {START} "
              f"({'fixed' if args.fixed else 'trail'})")
        new_trades = collect(con, symbols, trailing=not args.fixed)
        con.execute(
            "INSERT INTO runs (run_at, symbols, new_trades, total_trades) VALUES (?,?,?,?)",
            (datetime.now(timezone.utc).isoformat(), len(symbols), new_trades,
             con.execute("SELECT COUNT(*) FROM trades").fetchone()[0]),
        )
        con.commit()

    s = stats(con)
    print(f"\n{'LEDGER':<22}{'value':>14}")
    print(f"{'trades':<22}{s['trades']:>14}")
    print(f"{'new this run':<22}{new_trades:>14}")
    print(f"{'cumulative P&L':<22}{s['total_pnl']:>14,.2f}")
    print(f"{'avg return / trade':<22}{s['avg_return']:>13.2%}")
    print(f"{'avg buy & hold':<22}{s['avg_bh']:>13.2%}")
    print(f"{'avg EDGE':<22}{s['avg_edge']:>13.2%}")
    print(f"{'win rate':<22}{s['win_rate']:>13.1%}")
    print(f"{'beat buy & hold':<22}{s['beat_bh_rate']:>13.1%}")
    print(f"{'winner hold (d)':<22}{s['winner_hold']:>14.1f}")
    print(f"{'loser hold (d)':<22}{s['loser_hold']:>14.1f}")
    print(f"{'disposition ratio':<22}{s['disposition_ratio']:>14.2f}")

    if s["trades"]:
        if s["avg_edge"] <= 0:
            print("\nEDGE <= 0: trades are not beating buy-and-hold. More data will not")
            print("fix a signal problem — this is the number to watch, not P&L.")
        if s["disposition_ratio"] >= 2.0:
            print("DISPOSITION >= 2.0: losers held far longer than winners.")

    p = write_obsidian(s, new_trades, len(symbols))
    if p:
        print(f"\nObsidian: {p.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
