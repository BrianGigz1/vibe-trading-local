"""Turn-of-month effect — returns clustered around month boundaries.

Chosen deliberately after the overnight test. That effect looked real and died on
turnover: it trades every day, so 2bps per switch halved the CAGR. Turn-of-month
makes the *same kind* of claim — returns concentrate in a specific window — but
trades roughly **twice a month**, so costs cannot be what kills it. If it fails,
it fails on the merits.

Window definition, fixed before running: the last `TAIL` trading days of a month
plus the first `HEAD` trading days of the next. Everything else is "rest".

Tested the same way as everything else:

  * per symbol, so one name cannot carry the result
  * paired t-test of TOM vs rest daily returns within each symbol
  * a portfolio holding ONLY the TOM window, costed at realistic turnover
  * compared against buy & hold on Sharpe, not on total return

    python turn_of_month.py [--head 3] [--tail 1] [--cost-bps 2]
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


def tom_mask(index: pd.DatetimeIndex, head: int, tail: int) -> pd.Series:
    """True on the last *tail* trading days of a month and first *head* of the next."""
    s = pd.Series(index=index, data=False)
    by_month = pd.Series(index.to_period("M"), index=index)
    for _, days in index.to_series().groupby(by_month):
        if tail > 0:
            s.loc[days.index[-tail:]] = True
        if head > 0:
            s.loc[days.index[:head]] = True
    return s


def summarise(r: pd.Series, cost_series: pd.Series | None = None) -> dict:
    """CAGR, drawdown and Sharpe of a daily return series."""
    net = r if cost_series is None else r - cost_series
    eq = (1.0 + net).cumprod()
    years = len(net) / 252.0
    sd = float(net.std(ddof=1))
    return {
        "cagr": float(eq.iloc[-1] ** (1 / years) - 1.0) if years > 0 else 0.0,
        "dd": float((eq / eq.cummax() - 1.0).min()),
        "sharpe": float(net.mean() / sd * math.sqrt(252)) if sd else 0.0,
    }


def main() -> int:
    """Measure the turn-of-month window against the rest of the month."""
    ap = argparse.ArgumentParser(description="Turn-of-month effect")
    ap.add_argument("--head", type=int, default=3, help="first N trading days of a month")
    ap.add_argument("--tail", type=int, default=1, help="last N trading days of a month")
    ap.add_argument("--cost-bps", type=float, default=2.0)
    ap.add_argument("--symbols", nargs="*", default=None)
    args = ap.parse_args()
    symbols = args.symbols or UNIVERSE
    cost = args.cost_bps / 10_000.0

    rows, tstats, tom_series, all_series, exposures = [], [], [], [], []
    for i, sym in enumerate(symbols, 1):
        try:
            df = load_ohlcv(sym, START, END, "1d")
        except DataError:
            continue
        ret = df["close"].pct_change().dropna()
        if len(ret) < 500:
            continue
        mask = tom_mask(ret.index, args.head, args.tail)
        tom, rest = ret[mask], ret[~mask]
        if len(tom) < 60 or len(rest) < 60:
            continue

        # Welch t-test: the two windows have different lengths and variances.
        m1, m2 = float(tom.mean()), float(rest.mean())
        v1, v2 = float(tom.var(ddof=1)), float(rest.var(ddof=1))
        n1, n2 = len(tom), len(rest)
        se = math.sqrt(v1 / n1 + v2 / n2)
        t = (m1 - m2) / se if se > 0 else 0.0
        tstats.append(t)

        # Portfolio leg: invested only during the TOM window.
        leg = ret.where(mask, 0.0)
        tom_series.append(leg.rename(sym))
        all_series.append(ret.rename(sym))
        exposures.append(float(mask.mean()))

        rows.append([sym, n1, f"{m1*10000:+.2f}", f"{m2*10000:+.2f}",
                     f"{(m1-m2)*10000:+.2f}", f"{t:+.2f}"])
        if i % 20 == 0:
            print(f"  {i}/{len(symbols)}", flush=True)

    if not rows:
        print("no usable symbols", file=sys.stderr)
        return 1

    rows.sort(key=lambda r: float(r[5]), reverse=True)
    print("\n" + tabulate(rows[:12], headers=(
        "sym", "TOM days", "TOM bps/d", "rest bps/d", "diff bps/d", "t"),
        tablefmt="simple"))
    print(f"  … {max(0, len(rows)-12)} more symbols")

    n = len(tstats)
    pos = sum(1 for t in tstats if t > 2.0)
    negs = sum(1 for t in tstats if t < -2.0)
    exposure = float(np.mean(exposures))

    print(f"\nsymbols {n}   window = last {args.tail} + first {args.head} trading days "
          f"({exposure:.1%} of days)")
    print(f"TOM > rest at t>2: {pos}/{n}   rest > TOM at t<-2: {negs}/{n}   "
          f"expected by chance ~{n*0.05:.1f}")

    tom_port = pd.concat(tom_series, axis=1).mean(axis=1).dropna()
    bh_port = pd.concat(all_series, axis=1).mean(axis=1).dropna()

    # Cost only on days exposure changes (entering or leaving the window).
    mask_port = tom_mask(tom_port.index, args.head, args.tail)
    switches = mask_port.astype(int).diff().abs().fillna(0.0)
    cost_series = switches * cost
    switches_per_year = float(switches.sum()) / (len(tom_port) / 252.0)

    tom_stats = summarise(tom_port, cost_series)
    tom_free = summarise(tom_port)
    bh_stats = summarise(bh_port)

    print(f"\nswitches/year {switches_per_year:.0f}  "
          f"(overnight strategy needed ~504 — this is why costs matter less here)")
    print(f"\n{'leg':<26}{'CAGR':>9}{'maxDD':>9}{'Sharpe':>9}")
    print(f"{'TOM only (0 cost)':<26}{tom_free['cagr']:>8.1%}{tom_free['dd']:>9.1%}{tom_free['sharpe']:>9.2f}")
    print(f"{f'TOM only ({args.cost_bps:g}bps)':<26}{tom_stats['cagr']:>8.1%}"
          f"{tom_stats['dd']:>9.1%}{tom_stats['sharpe']:>9.2f}")
    print(f"{'buy & hold':<26}{bh_stats['cagr']:>8.1%}{bh_stats['dd']:>9.1%}{bh_stats['sharpe']:>9.2f}")

    print()
    real = pos > n * 0.15 and pos > negs * 2
    beats = tom_stats["sharpe"] > bh_stats["sharpe"]
    if real and beats:
        print(f"RESULT: the effect is present ({pos}/{n}, chance ~{n*0.05:.1f}) AND the")
        print(f"TOM leg beats buy & hold on Sharpe after costs "
              f"({tom_stats['sharpe']:.2f} vs {bh_stats['sharpe']:.2f}),")
        print(f"while invested only {exposure:.0%} of days. Out-of-sample period next.")
    elif real:
        print(f"RESULT: the effect is present ({pos}/{n} vs ~{n*0.05:.1f} by chance) but")
        print(f"the TOM leg does not beat buy & hold on Sharpe "
              f"({tom_stats['sharpe']:.2f} vs {bh_stats['sharpe']:.2f}).")
    else:
        print(f"RESULT: no reliable effect ({pos}/{n} significant vs ~{n*0.05:.1f} expected")
        print("by chance). Turn-of-month is not distinguishable from noise here.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
