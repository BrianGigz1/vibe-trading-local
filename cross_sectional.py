"""Cross-sectional information coefficient test.

Everything tested earlier today was time-series and single-name: one symbol,
long-only, fully exposed to beta. That is the worst configuration for detecting
an edge, because `IR ~= IC * sqrt(breadth)` and breadth was 1.

This tests the other construction. Each rebalance date, every name in the
universe is ranked by a signal, and the rank is correlated (Spearman) with the
forward return over the next H days. That correlation is the information
coefficient. Because names are compared against each other at the same instant,
a market-wide move affects all of them and largely cancels — the test is
approximately market-neutral by construction, so buy-and-hold is no longer the
benchmark. Zero is.

Reported per signal:
  IC mean      average rank correlation per period
  IC std       its variability
  IC IR        mean / std — the stable quantity, an IC of 0.02 that never
               flips sign beats an IC of 0.05 that alternates
  t-stat       IC mean / (IC std / sqrt(periods))
  D10-D1       long top decile, short bottom decile, equal weight, per period

A real cross-sectional signal in liquid US equities lives around IC 0.01-0.05.
Anything much larger on this kind of universe is a bug, not a discovery.

    python cross_sectional.py [--quick]
"""
from __future__ import annotations

import math
import sys
from dataclasses import dataclass

import numpy as np
import pandas as pd

from data import DataError, load_ohlcv

UNIVERSE = [
    "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "TSLA", "AVGO", "AMD", "INTC",
    "CSCO", "ORCL", "CRM", "ADBE", "QCOM", "TXN", "IBM", "NOW", "INTU", "MU",
    "JPM", "BAC", "WFC", "GS", "MS", "C", "AXP", "BLK", "SCHW", "SPGI",
    "JNJ", "UNH", "PFE", "MRK", "ABBV", "LLY", "TMO", "ABT", "BMY", "AMGN",
    "PG", "KO", "PEP", "WMT", "COST", "MCD", "NKE", "SBUX", "TGT", "HD",
    "XOM", "CVX", "COP", "SLB", "EOG", "CAT", "DE", "BA", "HON", "GE",
    "V", "MA", "PYPL", "DIS", "NFLX", "CMCSA", "T", "VZ", "LOW", "UPS",
]
QUICK = UNIVERSE[:24]

START, END = "2016-01-01", "2026-06-30"
HORIZON = 5          # forward return horizon, trading days
REBALANCE = 5        # evaluate every N bars


@dataclass
class ICResult:
    """Aggregate statistics for one signal."""

    name: str
    ic_mean: float
    ic_std: float
    periods: int
    d10_d1_mean: float
    hit_rate: float

    @property
    def ic_ir(self) -> float:
        """Information ratio of the IC series."""
        return 0.0 if self.ic_std == 0 else self.ic_mean / self.ic_std

    @property
    def t_stat(self) -> float:
        """t-statistic of mean IC against zero."""
        if self.ic_std == 0 or self.periods < 2:
            return 0.0
        return self.ic_mean / (self.ic_std / math.sqrt(self.periods))


def build_signals(close: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Compute cross-sectional signal panels from a wide close-price frame.

    Args:
        close: Frame indexed by date, one column per symbol.

    Returns:
        Mapping of signal name to a same-shaped panel. Higher is intended to
        mean "more attractive"; the sign is only a label — a consistently
        negative IC is just as informative as a positive one.
    """
    ret1 = close.pct_change()
    out: dict[str, pd.DataFrame] = {}

    out["mom_20"] = close / close.shift(20) - 1.0
    out["mom_60"] = close / close.shift(60) - 1.0
    # 12-1 momentum: skip the most recent month, the classic construction.
    out["mom_252_21"] = close.shift(21) / close.shift(252) - 1.0
    # Short-term reversal: last week's losers.
    out["reversal_5"] = -(close / close.shift(5) - 1.0)
    out["reversal_1"] = -ret1
    # Low volatility.
    out["low_vol_60"] = -ret1.rolling(60).std()
    # Distance below the trailing 252d high.
    out["dist_252"] = close / close.rolling(252).max() - 1.0
    return out


def run(symbols: list[str]) -> int:
    """Load data, compute ICs, print the report."""
    print(f"universe {len(symbols)}  window {START}..{END}  "
          f"horizon {HORIZON}d  rebalance {REBALANCE}d")

    series: dict[str, pd.Series] = {}
    for i, sym in enumerate(symbols, 1):
        try:
            df = load_ohlcv(sym, START, END, "1d")
            series[sym] = df["close"]
        except DataError as exc:
            print(f"  ! {sym}: {exc}", file=sys.stderr)
        if i % 10 == 0:
            print(f"  loaded {i}/{len(symbols)}", flush=True)

    if len(series) < 10:
        print("too few symbols loaded")
        return 1

    close = pd.DataFrame(series).sort_index()
    close = close.dropna(axis=1, thresh=int(len(close) * 0.8))   # drop sparse names
    print(f"  usable symbols: {close.shape[1]}, bars: {close.shape[0]}")

    fwd = close.shift(-HORIZON) / close - 1.0
    signals = build_signals(close)

    # Rebalance dates: every REBALANCE bars, after the longest warm-up.
    warm = 252 + 21
    dates = close.index[warm::REBALANCE]
    dates = [d for d in dates if d in fwd.index]

    results: list[ICResult] = []
    for name, panel in signals.items():
        ics, spreads = [], []
        for d in dates:
            s = panel.loc[d]
            f = fwd.loc[d]
            pair = pd.concat([s, f], axis=1, keys=["sig", "fwd"]).dropna()
            if len(pair) < 10:
                continue
            # Spearman is Pearson on ranks. Computing it that way avoids a
            # scipy dependency for one function.
            ranks = pair.rank()
            ic = ranks["sig"].corr(ranks["fwd"])
            if not np.isfinite(ic):
                continue
            ics.append(ic)

            # Decile spread, equal weight, market-neutral by construction.
            k = max(1, len(pair) // 10)
            ranked = pair.sort_values("sig")
            spreads.append(ranked["fwd"].tail(k).mean() - ranked["fwd"].head(k).mean())

        if len(ics) < 20:
            continue
        arr = np.array(ics)
        results.append(ICResult(
            name=name,
            ic_mean=float(arr.mean()),
            ic_std=float(arr.std(ddof=1)),
            periods=len(arr),
            d10_d1_mean=float(np.mean(spreads)) if spreads else 0.0,
            hit_rate=float((arr > 0).mean()),
        ))

    if not results:
        print("no signal produced enough periods")
        return 1

    results.sort(key=lambda r: abs(r.t_stat), reverse=True)
    print(f"\n{'signal':<14}{'IC mean':>10}{'IC std':>9}{'IC IR':>8}"
          f"{'t-stat':>9}{'IC>0':>7}{'D10-D1':>10}{'periods':>9}")
    for r in results:
        print(f"{r.name:<14}{r.ic_mean:>10.4f}{r.ic_std:>9.4f}{r.ic_ir:>8.3f}"
              f"{r.t_stat:>+9.2f}{r.hit_rate:>7.0%}{r.d10_d1_mean:>+10.3%}{r.periods:>9}")

    n_tested = len(results)
    sig = [r for r in results if abs(r.t_stat) > 2.0]
    print(f"\nsignals tested: {n_tested}   |t| > 2: {len(sig)}")
    print("Note: with %d signals tested, ~%.1f would clear |t|>2 by chance alone."
          % (n_tested, n_tested * 0.05))

    if sig:
        best = sig[0]
        direction = "positive" if best.ic_mean > 0 else "NEGATIVE (signal is inverted)"
        print(f"\nStrongest: {best.name}  IC {best.ic_mean:+.4f}  t {best.t_stat:+.2f}  "
              f"D10-D1 {best.d10_d1_mean:+.3%} per {HORIZON}d — {direction}.")
        print("Cross-sectional IC in liquid US large caps is normally 0.01-0.05.")
        print("This needs a different period and a different universe before it means anything.")
    else:
        print("\nNo signal is statistically distinguishable from zero IC.")
    return 0


if __name__ == "__main__":
    sys.exit(run(QUICK if "--quick" in sys.argv else UNIVERSE))
