"""Cross-sectional long/short portfolio — what the IC is actually worth.

`cross_sectional.py` measured 12-1 momentum at IC +0.0342 (t +2.55). An IC is
not money. This turns it into an equity curve: each rebalance, long the top
decile and short the bottom, equal weight, dollar-neutral, with turnover and
short-borrow charged.

**This is an UPPER BOUND, not an estimate.** The universe is 70 names that are
large caps today, so it carries the full survivorship inflation that
`robustness.py` could not correct (yfinance served 2 of 20 delisted tickers).
The unbiased number is almost certainly lower.

That makes this the right test to run *before* paying a vendor: if the inflated
upper bound is unattractive after costs, the corrected version cannot rescue it,
and the data purchase is not worth making.

Also reported: the theoretical `IR ≈ IC × √breadth`, so the realised Sharpe can
be checked against what the IC implies rather than taken on faith.

    python xs_portfolio.py [--cost-bps 10] [--borrow 0.03]
"""
from __future__ import annotations

import argparse
import math
import sys

import numpy as np
import pandas as pd

from cross_sectional import HORIZON, REBALANCE, UNIVERSE, build_signals
from data import DataError, load_ohlcv

START, END = "2016-01-01", "2026-06-30"
WARMUP = 273


def load_panel(symbols: list[str]) -> pd.DataFrame:
    """Wide close-price panel."""
    series = {}
    for i, s in enumerate(symbols, 1):
        try:
            series[s] = load_ohlcv(s, START, END, "1d")["close"]
        except DataError:
            continue
        if i % 20 == 0:
            print(f"  loaded {i}/{len(symbols)}", flush=True)
    if not series:
        raise DataError("no symbols loaded")
    panel = pd.DataFrame(series).sort_index()
    return panel.dropna(axis=1, thresh=int(len(panel) * 0.8))


def run(close: pd.DataFrame, signal: str, cost_bps: float,
        borrow_annual: float, decile: int) -> dict:
    """Backtest the dollar-neutral decile spread."""
    fwd = close.shift(-HORIZON) / close - 1.0
    panel = build_signals(close)[signal]
    dates = [d for d in close.index[WARMUP::REBALANCE] if d in fwd.index]

    prev_long: set[str] = set()
    prev_short: set[str] = set()
    rets, ics, turnovers = [], [], []

    for d in dates:
        pair = pd.concat([panel.loc[d], fwd.loc[d]], axis=1,
                         keys=["sig", "fwd"]).dropna()
        if len(pair) < decile * 2:
            continue
        k = max(1, len(pair) // decile)
        ranked = pair.sort_values("sig")
        short_names = set(ranked.head(k).index)
        long_names = set(ranked.tail(k).index)

        gross = ranked["fwd"].tail(k).mean() - ranked["fwd"].head(k).mean()

        # Turnover: fraction of each leg replaced since the last rebalance.
        if prev_long:
            turn = (len(long_names - prev_long) / k + len(short_names - prev_short) / k) / 2
        else:
            turn = 1.0
        turnovers.append(turn)

        # Costs: both legs traded, plus borrow on the short leg for the period.
        cost = 2.0 * turn * (cost_bps / 10_000.0)
        borrow = borrow_annual * (HORIZON / 252.0)
        rets.append(gross - cost - borrow)

        r = pair.rank()
        v = r["sig"].corr(r["fwd"])
        if np.isfinite(v):
            ics.append(v)

        prev_long, prev_short = long_names, short_names

    if len(rets) < 20:
        raise ValueError("too few rebalances")

    a = np.array(rets)
    periods_per_year = 252.0 / REBALANCE
    equity = (1.0 + a).cumprod()
    peak = np.maximum.accumulate(equity)
    years = len(a) / periods_per_year

    ic_arr = np.array(ics)
    breadth = close.shape[1]
    return {
        "periods": len(a),
        "gross_per_period": float(a.mean()),
        "vol_per_period": float(a.std(ddof=1)),
        "cagr": float(equity[-1] ** (1 / years) - 1.0) if years > 0 else 0.0,
        "sharpe": float(a.mean() / a.std(ddof=1) * math.sqrt(periods_per_year))
                  if a.std(ddof=1) else 0.0,
        "maxdd": float((equity / peak - 1.0).min()),
        "total": float(equity[-1] - 1.0),
        "avg_turnover": float(np.mean(turnovers)),
        "ic_mean": float(ic_arr.mean()) if len(ic_arr) else 0.0,
        "breadth": breadth,
        "implied_ir": float(ic_arr.mean() * math.sqrt(breadth)) if len(ic_arr) else 0.0,
    }


def main() -> int:
    """Run the spread at several cost assumptions."""
    ap = argparse.ArgumentParser(description="Cross-sectional L/S portfolio")
    ap.add_argument("--signal", default="mom_252_21")
    ap.add_argument("--borrow", type=float, default=0.03,
                    help="annual short-borrow rate")
    ap.add_argument("--decile", type=int, default=10,
                    help="10 = deciles, 5 = quintiles (wider, lower turnover)")
    args = ap.parse_args()

    print(f"universe {len(UNIVERSE)}  {START}..{END}  signal {args.signal}")
    try:
        close = load_panel(UNIVERSE)
    except DataError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(f"  usable symbols: {close.shape[1]}\n")

    print(f"{'cost (bps/side)':<18}{'CAGR':>9}{'Sharpe':>9}{'maxDD':>9}"
          f"{'turnover':>10}{'total':>10}")
    results = {}
    for bps in (0, 5, 10, 20):
        try:
            r = run(close, args.signal, bps, args.borrow, args.decile)
        except ValueError as exc:
            print(f"  ! {exc}", file=sys.stderr)
            return 3
        results[bps] = r
        print(f"{bps:<18}{r['cagr']:>8.1%}{r['sharpe']:>9.2f}"
              f"{r['maxdd']:>9.1%}{r['avg_turnover']:>10.0%}{r['total']:>10.1%}")

    r10 = results[10]
    print(f"\nIC mean {r10['ic_mean']:+.4f}  breadth {r10['breadth']}  "
          f"implied IR (IC x sqrt(breadth)) {r10['implied_ir']:.2f}")
    print(f"realised Sharpe at 10bps: {r10['sharpe']:.2f}")

    print("\n⚠️  UPPER BOUND. This universe is survivorship-biased — every name is a")
    print("   large cap today. The unbiased result is very likely lower.")

    print()
    if r10["sharpe"] < 0.3:
        print("VERDICT: even the inflated upper bound is unattractive after costs.")
        print("Correcting survivorship can only lower it. The vendor data purchase")
        print("is NOT worth making for this signal.")
    elif r10["sharpe"] < 0.7:
        print("VERDICT: marginal. The upper bound is modest and survivorship")
        print("correction will reduce it. Buy the data only if you are prepared")
        print("for the corrected number to fall below usable.")
    else:
        print("VERDICT: the upper bound is strong enough that a survivorship")
        print("correction could still leave something usable. The vendor purchase")
        print("is justified — that is the test that decides it.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
