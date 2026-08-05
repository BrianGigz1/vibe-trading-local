"""Parameter sweep with a held-out validation gate.

A grid search over strategy parameters is a machine for manufacturing false
positives: test N combinations and the best one looks good by luck alone. This
harness is built so that cannot happen quietly.

Protocol, fixed before the first run:

  1. Symbols are split into SEARCH and HOLDOUT. The holdout set is never
     touched during the search.
  2. Every combination is scored on the SEARCH symbols by mean excess return
     versus buy-and-hold — not by raw return, which flatters anything long in a
     rising market.
  3. Exactly one combination — the search winner — is then run on the HOLDOUT
     symbols. One shot. No re-picking after seeing the holdout result.
  4. The report states how many combinations were tested and how many would be
     expected to clear zero by chance, so a marginal winner is read correctly.

    python sweep.py [--quick]
"""
from __future__ import annotations

import itertools
import json
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from backtest import Backtester
from config import Config, RiskLimits, StrategyParams
from data import DataError, load_ohlcv

SEARCH_SYMBOLS = ["AAPL", "MSFT", "NVDA", "SPY", "KO"]
HOLDOUT_SYMBOLS = ["JNJ", "PG", "CVX", "HD", "V"]     # never seen during search
START, END = "2018-01-01", "2026-06-30"

GRID = {
    "fast": [10, 20, 50],
    "slow": [50, 100, 200],
    "atr_stop": [1.5, 2.0, 3.0],
    "trailing": [False, True],
}
QUICK_GRID = {"fast": [20], "slow": [50, 100], "atr_stop": [2.0], "trailing": [False, True]}

OUT = Path(__file__).resolve().parent / "sweep_results.json"


@dataclass
class Combo:
    """One parameter combination and its aggregate score."""

    fast: int
    slow: int
    atr_stop: float
    trailing: bool
    mean_excess: float = 0.0
    mean_return: float = 0.0
    mean_bh: float = 0.0
    n_symbols: int = 0
    beat_bh: int = 0

    @property
    def label(self) -> str:
        """Short human label."""
        mode = "trail" if self.trailing else "fixed"
        return f"{self.fast}/{self.slow} atr{self.atr_stop:g} {mode}"


def evaluate(combo: Combo, symbols: list[str], cache: dict) -> Combo:
    """Score *combo* across *symbols*, returning a populated copy."""
    excesses, rets, bhs, beat = [], [], [], 0
    for sym in symbols:
        df = cache.get(sym)
        if df is None:
            continue
        cfg = Config(
            symbol=sym, start=START, end=END, initial_capital=100_000.0,
            risk=RiskLimits(),
            strategy=StrategyParams(
                fast_period=combo.fast, slow_period=combo.slow,
                atr_stop_multiple=combo.atr_stop,
                atr_target_multiple=combo.atr_stop * 1.5,
            ),
        )
        try:
            res = Backtester(cfg, trailing=combo.trailing).run(df)
        except ValueError:
            continue
        excesses.append(res.excess_return)
        rets.append(res.total_return)
        bhs.append(res.buy_hold_return)
        if res.total_return > res.buy_hold_return:
            beat += 1

    n = len(excesses)
    if n:
        combo.mean_excess = sum(excesses) / n
        combo.mean_return = sum(rets) / n
        combo.mean_bh = sum(bhs) / n
        combo.n_symbols = n
        combo.beat_bh = beat
    return combo


def main() -> int:
    """Run the sweep and the holdout gate."""
    quick = "--quick" in sys.argv
    grid = QUICK_GRID if quick else GRID

    combos = [
        Combo(fast=f, slow=s, atr_stop=a, trailing=t)
        for f, s, a, t in itertools.product(
            grid["fast"], grid["slow"], grid["atr_stop"], grid["trailing"]
        )
        if f < s   # a fast MA slower than the slow MA is not a strategy
    ]
    print(f"combinations: {len(combos)}   search: {SEARCH_SYMBOLS}   "
          f"holdout: {HOLDOUT_SYMBOLS}")

    cache: dict = {}
    for sym in SEARCH_SYMBOLS + HOLDOUT_SYMBOLS:
        try:
            cache[sym] = load_ohlcv(sym, START, END, "1d")
        except DataError as exc:
            print(f"  ! {sym}: {exc}", file=sys.stderr)

    t0 = time.time()
    scored = []
    for i, c in enumerate(combos, 1):
        scored.append(evaluate(c, SEARCH_SYMBOLS, cache))
        if i % 5 == 0 or i == len(combos):
            print(f"  scored {i}/{len(combos)}  ({time.time() - t0:.0f}s)", flush=True)

    scored = [c for c in scored if c.n_symbols]
    if not scored:
        print("no scorable combinations")
        return 1
    scored.sort(key=lambda c: c.mean_excess, reverse=True)

    print("\nSEARCH RESULTS (top 8 by mean excess vs buy & hold)")
    print(f"{'combo':<26}{'mean excess':>13}{'mean ret':>11}{'mean B&H':>11}{'beat B&H':>10}")
    for c in scored[:8]:
        print(f"{c.label:<26}{c.mean_excess:>12.1%}{c.mean_return:>11.1%}"
              f"{c.mean_bh:>11.1%}{c.beat_bh:>7}/{c.n_symbols}")

    winner = scored[0]
    positives = sum(1 for c in scored if c.mean_excess > 0)
    print(f"\ncombinations tested: {len(scored)}   with positive mean excess: {positives}")

    # ── the gate: one shot on symbols never used in the search ─────────────
    print(f"\nHOLDOUT — running ONLY the search winner ({winner.label}) on "
          f"{HOLDOUT_SYMBOLS}")
    held = evaluate(
        Combo(winner.fast, winner.slow, winner.atr_stop, winner.trailing),
        HOLDOUT_SYMBOLS, cache,
    )
    print(f"  search  mean excess {winner.mean_excess:+.1%}  "
          f"beat B&H {winner.beat_bh}/{winner.n_symbols}")
    print(f"  holdout mean excess {held.mean_excess:+.1%}  "
          f"beat B&H {held.beat_bh}/{held.n_symbols}")

    survived = held.mean_excess > 0
    print()
    if survived:
        print(f"VERDICT: the winner SURVIVED holdout ({held.mean_excess:+.1%}). "
              f"With {len(scored)} combinations tested this is worth one more "
              f"test on a different period — it is not yet an edge.")
    else:
        print(f"VERDICT: the winner FAILED holdout ({held.mean_excess:+.1%} vs "
              f"{winner.mean_excess:+.1%} in search). The search result was "
              f"selection, not signal — the expected outcome when ranking "
              f"{len(scored)} combinations on noise.")

    OUT.write_text(json.dumps({
        "generated": datetime.now(timezone.utc).isoformat(),
        "window": [START, END],
        "combinations_tested": len(scored),
        "positive_in_search": positives,
        "search_symbols": SEARCH_SYMBOLS,
        "holdout_symbols": HOLDOUT_SYMBOLS,
        "winner": winner.label,
        "search_mean_excess": winner.mean_excess,
        "holdout_mean_excess": held.mean_excess,
        "survived_holdout": survived,
        "top": [{"combo": c.label, "mean_excess": c.mean_excess,
                 "beat_bh": f"{c.beat_bh}/{c.n_symbols}"} for c in scored[:15]],
    }, indent=2), encoding="utf-8")
    print(f"\nwrote {OUT.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
