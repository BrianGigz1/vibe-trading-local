"""Overnight vs intraday return decomposition.

Every signal tested on 2026-08-05 was trend/momentum measured close-to-close, and
all of them failed. This tests something structurally different: the documented
finding that US equity returns accrue disproportionately **overnight**
(close -> next open) rather than **intraday** (open -> close).

    total daily return = overnight + intraday

If the split is lopsided, holding only overnight would capture most of the drift
with roughly half the market exposure. That is a different claim from anything
tested so far — it is about *when* returns happen, not about predicting *which
way* they will go.

Tested honestly:

  * split measured per symbol, not pooled, so one name cannot carry it
  * a paired t-test on the difference, because overnight and intraday are two
    measurements of the same day and are not independent samples
  * costs applied at a realistic per-switch rate — the overnight strategy trades
    EVERY day (buy close, sell open), so turnover is ~252 round trips a year and
    is the thing most likely to destroy it
  * the zero-cost result is shown too, as an explicit upper bound

    python overnight.py [--cost-bps 2]
"""
from __future__ import annotations

import argparse
import math
import sys

import numpy as np
import pandas as pd
from tabulate import tabulate

from cross_sectional import UNIVERSE
from data import DataError, load_ohlcv

START, END = "2016-01-01", "2026-06-30"


def split_returns(df: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    """Overnight (close->open) and intraday (open->close) daily returns."""
    prev_close = df["close"].shift(1)
    overnight = df["open"] / prev_close - 1.0
    intraday = df["close"] / df["open"] - 1.0
    return overnight.dropna(), intraday.dropna()


def compound(r: pd.Series, cost: float = 0.0) -> float:
    """Total return of a series, charging *cost* per period."""
    return float((1.0 + (r - cost)).prod() - 1.0)


def main() -> int:
    """Decompose returns and test whether the split is real."""
    ap = argparse.ArgumentParser(description="Overnight vs intraday")
    ap.add_argument("--cost-bps", type=float, default=2.0,
                    help="cost per switch in bps; the overnight leg trades daily")
    ap.add_argument("--symbols", nargs="*", default=None)
    args = ap.parse_args()
    symbols = args.symbols or UNIVERSE
    cost = args.cost_bps / 10_000.0

    rows, diffs, on_all, id_all = [], [], [], []
    for i, sym in enumerate(symbols, 1):
        try:
            df = load_ohlcv(sym, START, END, "1d")
        except DataError:
            continue
        on, intr = split_returns(df)
        idx = on.index.intersection(intr.index)
        on, intr = on.loc[idx], intr.loc[idx]
        if len(on) < 250:
            continue

        d = on - intr
        sd = float(d.std(ddof=1))
        t = float(d.mean() / (sd / math.sqrt(len(d)))) if sd > 0 else 0.0
        diffs.append(t)
        on_all.append(on)
        id_all.append(intr)

        rows.append([
            sym, len(on),
            f"{compound(on):+.1%}", f"{compound(intr):+.1%}",
            f"{compound(df['close'].pct_change().dropna()):+.1%}",
            f"{on.mean()*10000:+.2f}", f"{intr.mean()*10000:+.2f}",
            f"{t:+.2f}",
            f"{compound(on, cost):+.1%}",
        ])
        if i % 20 == 0:
            print(f"  {i}/{len(symbols)}", flush=True)

    if not rows:
        print("no usable symbols", file=sys.stderr)
        return 1

    rows.sort(key=lambda r: float(r[7]), reverse=True)
    print("\n" + tabulate(rows[:15], headers=(
        "sym", "days", "overnight", "intraday", "buy&hold",
        "on bps/d", "id bps/d", "t(on-id)", f"on net {args.cost_bps:g}bps"),
        tablefmt="simple"))
    if len(rows) > 15:
        print(f"  … {len(rows) - 15} more symbols")

    # ── aggregate ──────────────────────────────────────────────────────────
    on_mean = float(np.mean([r.mean() for r in on_all])) * 10000
    id_mean = float(np.mean([r.mean() for r in id_all])) * 10000
    sig = sum(1 for t in diffs if t > 2.0)
    neg = sum(1 for t in diffs if t < -2.0)
    n = len(diffs)

    print(f"\nsymbols {n}   mean overnight {on_mean:+.2f} bps/day   "
          f"mean intraday {id_mean:+.2f} bps/day")
    print(f"overnight > intraday at |t|>2: {sig}/{n}   "
          f"intraday > overnight at |t|>2: {neg}/{n}")

    # Equal-weight portfolio of each leg.
    on_port = pd.concat(on_all, axis=1).mean(axis=1).dropna()
    id_port = pd.concat(id_all, axis=1).mean(axis=1).dropna()
    years = len(on_port) / 252.0

    def summarise(r: pd.Series, c: float) -> tuple[float, float, float]:
        net = r - c
        eq = (1.0 + net).cumprod()
        cagr = float(eq.iloc[-1] ** (1 / years) - 1.0) if years > 0 else 0.0
        dd = float((eq / eq.cummax() - 1.0).min())
        sharpe = float(net.mean() / net.std(ddof=1) * math.sqrt(252)) if net.std(ddof=1) else 0.0
        return cagr, dd, sharpe

    print(f"\n{'leg':<22}{'CAGR':>9}{'maxDD':>9}{'Sharpe':>9}")
    for label, series, c in (
        ("overnight (0 cost)", on_port, 0.0),
        (f"overnight ({args.cost_bps:g}bps)", on_port, cost),
        ("intraday (0 cost)", id_port, 0.0),
        ("buy & hold", on_port + id_port, 0.0),
    ):
        cagr, dd, sh = summarise(series, c)
        print(f"{label:<22}{cagr:>8.1%}{dd:>9.1%}{sh:>9.2f}")

    bh_cagr, _, bh_sh = summarise(on_port + id_port, 0.0)
    on_cagr, _, on_sh = summarise(on_port, cost)

    print()
    if sig >= n * 0.6 and on_sh > bh_sh:
        print(f"RESULT: the overnight/intraday split is real ({sig}/{n} symbols) AND")
        print(f"the overnight leg beats buy & hold on Sharpe after {args.cost_bps:g}bps")
        print(f"({on_sh:.2f} vs {bh_sh:.2f}). Worth an out-of-sample period next.")
    elif sig >= n * 0.6:
        print(f"RESULT: the split is REAL ({sig}/{n} symbols significant) but the")
        print(f"overnight leg does NOT beat buy & hold on Sharpe after costs")
        print(f"({on_sh:.2f} vs {bh_sh:.2f}). A true effect that daily turnover eats —")
        print("the same shape as every other edge measured today.")
    else:
        print(f"RESULT: no consistent split ({sig}/{n} significant). Overnight and")
        print("intraday returns are not reliably different on this universe.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
