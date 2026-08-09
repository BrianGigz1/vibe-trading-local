"""Two cheap tests that try to kill `low_vol_60` before anything is built on it.

`crypto_ic.py` found **low_vol_60 at IC +0.1050, t +5.73** across 18 pairs - three
times the IC of the equity momentum signal and more than double its t. Before
that earns a cost model, it has to survive the two ways it is most likely to be
an artifact:

**1. Majors-vs-alts.** Low realised volatility in an 18-coin panel is close to a
proxy for *large and liquid*. Over 2021-2026 BTC and ETH beat almost every
altcoin. If the "signal" is really that one fact, it is a single macro bet
sampled 354 times, not 354 independent observations - and the t-stat, which
assumes independence, is inflated accordingly. **Test: drop BTC and ETH.** A
signal that only works when the majors are present was never cross-sectional.

**2. One regime.** The equity IC was credible partly because it held across
halves (+0.0352 / +0.0332). If crypto's low-vol IC is large early and absent
later, it dated the alt bear market rather than found a signal. **Test: split
the sample in half by time.**

Both reuse `crypto_ic`'s exact IC computation - same horizon, same rebalance,
same thirds-not-deciles spread - so the numbers are directly comparable to the
baseline rather than merely similar to it.

    python crypto_kill.py
    python crypto_kill.py --focus low_vol_60
"""
from __future__ import annotations

import argparse
import math
import sys

import numpy as np
import pandas as pd

from crypto_data import CRYPTO_UNIVERSE, load_crypto
from crypto_ic import HORIZON, REBALANCE, START, WARMUP
from cross_sectional import build_signals
from data import DataError

MAJORS = ["BTC/USD", "ETH/USD"]


def ic_table(close: pd.DataFrame, date_filter=None) -> list[dict]:
    """IC per signal over *close*, optionally restricted to a subset of dates.

    Deliberately identical to `crypto_ic.main`'s inner loop. Signals are built on
    the **full** price history and only the evaluation dates are filtered, so a
    time split does not silently cost the second half its warmup window.
    """
    fwd = close.shift(-HORIZON) / close - 1.0
    signals = build_signals(close)
    dates = [d for d in close.index[WARMUP::REBALANCE] if d in fwd.index]
    if date_filter is not None:
        dates = [d for d in dates if date_filter(d)]

    out: list[dict] = []
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
            k = max(1, len(pair) // 3)
            ranked = pair.sort_values("sig")
            spreads.append(ranked["fwd"].tail(k).mean() - ranked["fwd"].head(k).mean())
        if len(ics) < 20:
            continue
        a = np.array(ics)
        sd = a.std(ddof=1)
        out.append({
            "name": name, "ic": float(a.mean()), "t": float(a.mean() / (sd / math.sqrt(len(a)))) if sd else 0.0,
            "n": len(a), "hit": float((a > 0).mean()),
            "spread": float(np.mean(spreads)) if spreads else 0.0,
        })
    out.sort(key=lambda r: abs(r["t"]), reverse=True)
    return out


def load_universe(exclude: list[str]) -> pd.DataFrame:
    """Close prices for the universe minus *exclude*, sparse columns kept."""
    series = {}
    for sym in CRYPTO_UNIVERSE:
        if sym in exclude:
            continue
        try:
            series[sym] = load_crypto(sym, START)["close"]
        except DataError as exc:
            print(f"  ! {sym}: {exc}", file=sys.stderr)
    close = pd.DataFrame(series).sort_index()
    return close.dropna(axis=1, thresh=400)


def show(label: str, rows: list[dict], focus: str) -> dict | None:
    """Print one variant's table and return the focus signal's row."""
    print(f"\n{label}")
    print(f"  {'signal':<14}{'IC':>9}{'t':>8}{'IC>0':>7}{'T3-B3':>10}{'periods':>9}")
    hit = None
    for r in rows[:5]:
        mark = "  <--" if r["name"] == focus else ""
        print(f"  {r['name']:<14}{r['ic']:>9.4f}{r['t']:>+8.2f}{r['hit']:>7.0%}"
              f"{r['spread']:>+10.2%}{r['n']:>9}{mark}")
    for r in rows:
        if r["name"] == focus:
            hit = r
            if r not in rows[:5]:
                print(f"  {r['name']:<14}{r['ic']:>9.4f}{r['t']:>+8.2f}{r['hit']:>7.0%}"
                      f"{r['spread']:>+10.2%}{r['n']:>9}  <--  (outside top 5)")
    if hit is None:
        print(f"  {focus}: too few periods to score in this slice")
    return hit


def verdict(base: dict | None, noMaj: dict | None,
            h1: dict | None, h2: dict | None, focus: str) -> None:
    """State plainly whether the signal survived both tests."""
    print("\n" + "=" * 68)
    print(f"VERDICT on {focus}")
    print("=" * 68)
    if base is None:
        print("baseline did not score - nothing to judge")
        return

    print(f"  baseline (18 pairs)   IC {base['ic']:+.4f}   t {base['t']:+.2f}")
    if noMaj:
        keep = abs(noMaj["t"]) > 2.0
        # Signed as a change, not a magnitude: "-29% IC" must not be printable
        # as "+29%". A number whose name inverts its meaning is how the capture
        # bug got into the graduation gate.
        chg = (noMaj["ic"] / base["ic"] - 1) * 100 if base["ic"] else 0.0
        print(f"  without BTC/ETH       IC {noMaj['ic']:+.4f}   t {noMaj['t']:+.2f}"
              f"   (IC {chg:+.0f}%)  ->  {'SURVIVES' if keep else 'DIES'}")
    if h1 and h2:
        same = (h1["ic"] > 0) == (h2["ic"] > 0)
        both = abs(h1["t"]) > 1.5 and abs(h2["t"]) > 1.5
        print(f"  first half            IC {h1['ic']:+.4f}   t {h1['t']:+.2f}")
        print(f"  second half           IC {h2['ic']:+.4f}   t {h2['t']:+.2f}"
              f"   ->  {'STABLE' if (same and both) else 'UNSTABLE'}")

    survived = bool(noMaj and abs(noMaj["t"]) > 2.0 and h1 and h2
                    and (h1["ic"] > 0) == (h2["ic"] > 0)
                    and abs(h1["t"]) > 1.5 and abs(h2["t"]) > 1.5)
    print()
    if survived:
        print("SURVIVED BOTH. Not a majors proxy and not one regime. This is now")
        print("worth a cost model - spread and funding on alts are the next test,")
        print("and they are what killed the equity signal.")
    else:
        print("KILLED. The result does not hold up once the majors are removed or")
        print("the sample is split. Do not build a portfolio backtest on it - the")
        print("cheap test ran before the expensive one, which is the point.")


def main() -> int:
    """Run the baseline and both kill-tests, then judge."""
    ap = argparse.ArgumentParser(description="Kill-tests for the crypto IC result")
    ap.add_argument("--focus", default="low_vol_60")
    args = ap.parse_args()

    full = load_universe([])
    print(f"universe {full.shape[1]} pairs   bars {full.shape[0]}   "
          f"{full.index[0].date()} .. {full.index[-1].date()}")
    base = show(f"BASELINE - all {full.shape[1]} pairs", ic_table(full), args.focus)

    # --- test 1: is it just "majors beat alts"? --------------------------
    alts = load_universe(MAJORS)
    noMaj = show(f"TEST 1 - BTC and ETH removed ({alts.shape[1]} pairs)",
                 ic_table(alts), args.focus)

    # --- test 2: is it one regime? ---------------------------------------
    mid = full.index[len(full.index) // 2]
    print(f"\n  split at {mid.date()}")
    h1 = show(f"TEST 2a - first half (.. {mid.date()})",
              ic_table(full, lambda d: d <= mid), args.focus)
    h2 = show(f"TEST 2b - second half ({mid.date()} ..)",
              ic_table(full, lambda d: d > mid), args.focus)

    verdict(base, noMaj, h1, h2, args.focus)
    return 0


if __name__ == "__main__":
    sys.exit(main())
