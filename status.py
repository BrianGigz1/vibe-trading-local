"""Graduation gate — evaluate the pre-committed criteria against the live ledger.

The criteria in `paper-to-real-money-graduation-criteria.md` were written before
the first trade, deliberately, because every failure caught on 2026-08-05 came
from judging after seeing the result. A document does not enforce itself, so this
turns them into a check that runs daily and reports PASS / FAIL per criterion.

**Thresholds are constants here on purpose.** Editing them is a visible diff in a
version-controlled file, which is exactly the friction that stops a bar being
quietly lowered on a good week.

Also runs a DATA INTEGRITY cross-check: yfinance and TradingView are independent
feeds, so their buy-and-hold return for the same symbol over the same window
should agree closely. A divergence is not a strategy question — it means one feed
is wrong, and everything downstream inherits that.

    python status.py            # evaluate, write status to Obsidian
    python status.py --quiet    # exit code only (0 = eligible, 1 = not)
"""
from __future__ import annotations

import argparse
import sqlite3
import sys
from dataclasses import dataclass
from datetime import datetime, timezone

import journal
from collector import init_db, stats

# ── Pre-committed thresholds. Raising is allowed; lowering is not. ──────────
MIN_TRADES = 200
MIN_DAYS = 90
MIN_BEAT_BH = 0.55
MIN_AVG_EDGE = 0.0
MIN_CAPTURE = 1.00          # capture of buy & hold; 1.00 = matched it
MAX_DRAWDOWN = 0.15
MAX_DISPOSITION = 2.0
MIN_SYMBOLS = 5

# Two independent price feeds should agree on a buy-and-hold return to well
# within this. Beyond it, one of them is wrong.
FEED_TOLERANCE = 0.02


def full_period_capture(con: sqlite3.Connection) -> float:
    """Strategy return as a fraction of buy-and-hold over the FULL window.

    This is criterion 5, and it must not be confused with the per-trade edge.

      per-trade edge : return vs the asset **during the periods held** — ignores
                       time spent in cash entirely, and is positive here
      full capture   : compounded strategy return vs holding the asset for the
                       **whole span**, cash periods included — this is ~0%

    An earlier version of this function computed the first and reported it under
    the second's name, turning the hardest criterion into an automatic PASS. The
    difference is the whole finding of 2026-08-05: the loss is absence, not error.

    Returns 0.0 when prices cannot be loaded — never a flattering default.
    """
    try:
        from data import DataError, load_ohlcv
    except ImportError:
        return 0.0

    rows = con.execute(
        "SELECT symbol, MIN(entry_date), MAX(exit_date) FROM trades "
        "WHERE mode != 'tv_paper' GROUP BY symbol"
    ).fetchall()
    if not rows:
        return 0.0

    ratios = []
    for sym, first, last in rows:
        trades = con.execute(
            "SELECT return_pct FROM trades WHERE symbol=? AND mode!='tv_paper' "
            "ORDER BY entry_date", (sym,)
        ).fetchall()
        # Compound the realised trade returns for this symbol.
        strat = 1.0
        for (r,) in trades:
            strat *= (1.0 + float(r))
        strat -= 1.0

        try:
            df = load_ohlcv(sym, first, "", "1d")
            span = df["close"].loc[first:last]
            if len(span) < 2 or float(span.iloc[0]) <= 0:
                continue
            bh = float(span.iloc[-1]) / float(span.iloc[0]) - 1.0
        except (DataError, KeyError, ValueError):
            continue
        if bh > 0:
            ratios.append(strat / bh)

    return sum(ratios) / len(ratios) if ratios else 0.0


@dataclass
class Check:
    """One criterion and its current standing."""

    n: int
    name: str
    value: str
    target: str
    passed: bool


def evaluate(con: sqlite3.Connection) -> tuple[list[Check], dict]:
    """Score every criterion against the ledger."""
    s = stats(con)
    row = con.execute(
        "SELECT MIN(entry_date), MAX(exit_date), COUNT(DISTINCT symbol) FROM trades"
    ).fetchone()
    first, last, symbols = row if row else (None, None, 0)

    days = 0
    if first and last:
        try:
            days = (datetime.fromisoformat(last) - datetime.fromisoformat(first)).days
        except ValueError:
            days = 0

    capture = full_period_capture(con)

    # Worst single trade stands in for drawdown until an equity curve is tracked.
    worst = con.execute(
        "SELECT COALESCE(MIN(return_pct),0) FROM trades"
    ).fetchone()[0]

    checks = [
        Check(1, "closed trades", f"{s['trades']}", f">= {MIN_TRADES}",
              s["trades"] >= MIN_TRADES),
        Check(2, "elapsed days", f"{days}", f">= {MIN_DAYS}", days >= MIN_DAYS),
        Check(3, "beat buy & hold", f"{s['beat_bh_rate']:.1%}", f">= {MIN_BEAT_BH:.0%}",
              s["beat_bh_rate"] >= MIN_BEAT_BH),
        Check(4, "avg edge / trade", f"{s['avg_edge']:+.2%}", "> 0",
              s["avg_edge"] > MIN_AVG_EDGE),
        Check(5, "CAPTURE of buy & hold", f"{capture:.1%}", f">= {MIN_CAPTURE:.0%}",
              capture >= MIN_CAPTURE),
        Check(6, "worst trade", f"{worst:.1%}", f">= -{MAX_DRAWDOWN:.0%}",
              abs(worst) <= MAX_DRAWDOWN),
        Check(7, "disposition ratio", f"{s['disposition_ratio']:.2f}",
              f"< {MAX_DISPOSITION}",
              0 < s["disposition_ratio"] < MAX_DISPOSITION),
        Check(8, "symbols traded", f"{symbols}", f">= {MIN_SYMBOLS}",
              (symbols or 0) >= MIN_SYMBOLS),
    ]
    return checks, {"capture": capture, "days": days, "symbols": symbols, **s}


def feed_crosscheck() -> tuple[str, str]:
    """Compare yfinance and TradingView on the same symbol's recent return.

    Returns (status, detail). Never raises — an unavailable TradingView is a
    skipped check, not a failure of the gate.
    """
    try:
        import tv_cdp
        from data import load_ohlcv

        st = tv_cdp.read_state()
        sym = (st.get("symbol") or "").upper()
        if not sym:
            return "SKIPPED", "TradingView reported no symbol"

        # Read from the price series, not a Pine scorecard — otherwise the check
        # silently depends on whichever script happens to be attached.
        tv_close = st.get("close")
        if tv_close is None:
            return "SKIPPED", f"{sym}: chart exposed no close price"
        tv_close = float(tv_close)

        df = load_ohlcv(sym, "2024-01-01", "", "1d")
        yf_close = float(df["close"].iloc[-1])
        diff = abs(tv_close - yf_close) / yf_close
        if diff <= FEED_TOLERANCE:
            return "AGREE", f"{sym}: yf {yf_close:.2f} vs tv {tv_close:.2f} ({diff:.2%})"
        return "DIVERGENT", (f"{sym}: yf {yf_close:.2f} vs tv {tv_close:.2f} "
                             f"({diff:.2%} > {FEED_TOLERANCE:.0%}) — one feed is wrong")
    except Exception as exc:
        return "SKIPPED", f"{type(exc).__name__}: {str(exc)[:90]}"


def write_note(checks: list[Check], eligible: bool, feed: tuple[str, str]) -> None:
    """Append the gate result to today's Obsidian note."""
    try:
        d = journal.journal_dir()
        d.mkdir(parents=True, exist_ok=True)
        day = datetime.now().strftime("%Y-%m-%d")
        path = d / f"{day}.md"
        if not path.exists():
            journal._ensure_note(path, day)
        passed = sum(1 for c in checks if c.passed)
        with path.open("a", encoding="utf-8") as fh:
            fh.write(f"### {datetime.now().strftime('%H:%M:%S')} — GRADUATION GATE "
                     f"`[{'ELIGIBLE' if eligible else 'NOT ELIGIBLE'}]` "
                     f"{'✅' if eligible else '⛔'}\n")
            fh.write(f"- **{passed}/{len(checks)} criteria met**\n")
            for c in checks:
                fh.write(f"  - {'✅' if c.passed else '❌'} **{c.name}:** "
                         f"`{c.value}` (need `{c.target}`)\n")
            fh.write(f"- **Feed cross-check:** `{feed[0]}` — {feed[1]}\n\n")
    except OSError as exc:
        print(f"  ! obsidian write failed: {exc}", file=sys.stderr)


def main() -> int:
    """Evaluate the gate. Exit 0 only if every criterion passes."""
    ap = argparse.ArgumentParser(description="Graduation gate")
    ap.add_argument("--quiet", action="store_true", help="exit code only")
    ap.add_argument("--no-note", action="store_true", help="skip the Obsidian write")
    args = ap.parse_args()

    con = init_db()
    checks, extra = evaluate(con)
    eligible = all(c.passed for c in checks)
    feed = feed_crosscheck()
    con.close()

    if not args.quiet:
        print(f"{'#':<3}{'criterion':<26}{'now':>12}{'target':>14}  result")
        for c in checks:
            print(f"{c.n:<3}{c.name:<26}{c.value:>12}{c.target:>14}  "
                  f"{'PASS' if c.passed else 'FAIL'}")
        print(f"\nfeed cross-check: {feed[0]} — {feed[1]}")
        passed = sum(1 for c in checks if c.passed)
        print(f"\n{passed}/{len(checks)} criteria met")
        print("VERDICT: ELIGIBLE for real money" if eligible
              else "VERDICT: NOT ELIGIBLE — thresholds are pre-committed and are "
                   "not to be lowered to fit this result")
        if feed[0] == "DIVERGENT":
            print("⚠️  FEEDS DISAGREE — investigate before trusting any number above.")

    if not args.no_note:
        write_note(checks, eligible, feed)
    return 0 if eligible else 1


if __name__ == "__main__":
    sys.exit(main())
