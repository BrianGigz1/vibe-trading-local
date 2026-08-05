"""Cross-sectional IC on crypto — the equity method, a different universe.

`cross_sectional.py` found 12-1 momentum at IC +0.0342 (t +2.55) on 70 US large
caps, then `xs_portfolio.py` showed it was worth Sharpe 0.50 at 60% drawdown —
and that was the survivorship-inflated upper bound.

Crypto is worth running the same method on because it differs where it matters:

  * 24/7, so a "day" is a UTC day and there is no session structure
  * 3-5x the daily volatility, which RAISES the bar for significance
  * a much shorter sample (2021 onward) — a real limitation, stated not hidden
  * a small universe (~18 pairs), so cross-sectional breadth is thin

That last point is the honest caveat: `IR ~= IC * sqrt(breadth)`. With breadth 18
instead of 70, even a real IC converts to far less. This test can say whether a
signal exists; it cannot promise it is worth trading at this breadth.

Includes MKR, which stopped trading on Alpaca in 2025 — so unlike the equity run,
this universe contains at least one delisted name.

    python crypto_ic.py
"""
from __future__ import annotations

import math
import sys

import numpy as np
import pandas as pd

from crypto_data import CRYPTO_UNIVERSE, load_crypto
from cross_sectional import build_signals
from data import DataError

START = "2021-01-01"
HORIZON, REBALANCE = 5, 5
WARMUP = 273


def main() -> int:
    """Load the universe, compute IC per signal, report with significance."""
    series = {}
    for sym in CRYPTO_UNIVERSE:
        try:
            series[sym] = load_crypto(sym, START)["close"]
        except DataError as exc:
            print(f"  ! {sym}: {exc}", file=sys.stderr)
    if len(series) < 8:
        print("too few pairs loaded", file=sys.stderr)
        return 1

    close = pd.DataFrame(series).sort_index()
    # Keep sparse columns: a pair that stops trading mid-sample (MKR) is exactly
    # the survivorship correction the equity universe could not get.
    close = close.dropna(axis=1, thresh=400)
    print(f"pairs {close.shape[1]}   bars {close.shape[0]}   "
          f"{close.index[0].date()} .. {close.index[-1].date()}")
    print(f"horizon {HORIZON}d  rebalance {REBALANCE}d (non-overlapping)\n")

    fwd = close.shift(-HORIZON) / close - 1.0
    signals = build_signals(close)
    dates = [d for d in close.index[WARMUP::REBALANCE] if d in fwd.index]

    results = []
    for name, panel in signals.items():
        ics, spreads = [], []
        for d in dates:
            pair = pd.concat([panel.loc[d], fwd.loc[d]], axis=1,
                             keys=["sig", "fwd"]).dropna()
            if len(pair) < 8:
                continue
            r = pair.rank()
            ic = r["sig"].corr(r["fwd"])
            if not np.isfinite(ic):
                continue
            ics.append(ic)
            k = max(1, len(pair) // 3)      # thirds, not deciles — universe is small
            ranked = pair.sort_values("sig")
            spreads.append(ranked["fwd"].tail(k).mean() - ranked["fwd"].head(k).mean())

        if len(ics) < 20:
            continue
        a = np.array(ics)
        sd = a.std(ddof=1)
        t = a.mean() / (sd / math.sqrt(len(a))) if sd else 0.0
        results.append({
            "name": name, "ic": float(a.mean()), "sd": float(sd), "t": float(t),
            "n": len(a), "hit": float((a > 0).mean()),
            "spread": float(np.mean(spreads)) if spreads else 0.0,
        })

    if not results:
        print("no signal produced enough periods", file=sys.stderr)
        return 1

    results.sort(key=lambda r: abs(r["t"]), reverse=True)
    print(f"{'signal':<14}{'IC':>9}{'t':>8}{'IC>0':>7}{'T3-B3':>10}{'periods':>9}")
    for r in results:
        print(f"{r['name']:<14}{r['ic']:>9.4f}{r['t']:>+8.2f}{r['hit']:>7.0%}"
              f"{r['spread']:>+10.2%}{r['n']:>9}")

    n_tested = len(results)
    sig = [r for r in results if abs(r["t"]) > 2.0]
    breadth = close.shape[1]
    print(f"\nsignals tested {n_tested}   |t|>2: {len(sig)}   "
          f"expected by chance ~{n_tested*0.05:.1f}")
    print(f"breadth {breadth} -> implied IR = IC x sqrt({breadth}) = "
          f"{results[0]['ic'] * math.sqrt(breadth):+.2f} for the top signal")

    print()
    if sig:
        b = sig[0]
        print(f"RESULT: {b['name']} clears |t|>2 (IC {b['ic']:+.4f}, t {b['t']:+.2f}).")
        print(f"At breadth {breadth} that implies IR ~{b['ic']*math.sqrt(breadth):.2f} —")
        print("thin. Needs a cost model and an out-of-sample period before it means")
        print("anything, and crypto costs (spread, funding) are worse than equities.")
    else:
        print(f"RESULT: no signal is statistically distinguishable from zero IC on")
        print(f"crypto ({len(sig)}/{n_tested} at |t|>2, chance ~{n_tested*0.05:.1f}).")
        print("Higher volatility raises the significance bar, and the sample is")
        print("shorter than the equity run — both work against detection here.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
