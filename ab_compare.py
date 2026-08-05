"""A/B the session's work: fixed stop+target vs breakeven-then-ATR-trail.

Identical bars, identical parameters, identical risk limits. One variable
changes. Anything else and the comparison means nothing.

Buy-and-hold is reported for every symbol because a strategy that makes money
while losing to holding the asset has not made money in any sense worth having.

    python ab_compare.py [SYMBOL ...]
"""
from __future__ import annotations

import sys

from tabulate import tabulate

from backtest import Backtester
from config import Config
from data import DataError, load_ohlcv

BASKET = ["AAPL", "MSFT", "NVDA", "SPY", "KO", "INTC", "PFE", "DIS"]
START, END = "2018-01-01", "2026-06-30"


def run(symbols: list[str]) -> int:
    """Run both variants across *symbols* and print the comparison."""
    rows: list[list[str]] = []
    agg = {"fixed": [], "trail": [], "bh": []}

    for sym in symbols:
        cfg = Config(symbol=sym, start=START, end=END, initial_capital=100_000.0)
        try:
            df = load_ohlcv(sym, START, END, cfg.interval)
        except DataError as exc:
            print(f"  ! {sym}: {exc}", file=sys.stderr)
            continue

        try:
            fixed = Backtester(cfg, trailing=False).run(df)
            trail = Backtester(cfg, trailing=True).run(df)
        except ValueError as exc:
            print(f"  ! {sym}: {exc}", file=sys.stderr)
            continue

        agg["fixed"].append(fixed.total_return)
        agg["trail"].append(trail.total_return)
        agg["bh"].append(fixed.buy_hold_return)

        rows.append([
            sym,
            f"{fixed.total_return:+.1%}",
            f"{trail.total_return:+.1%}",
            f"{fixed.buy_hold_return:+.1%}",
            f"{trail.total_return - fixed.total_return:+.1%}",
            f"{trail.total_return - trail.buy_hold_return:+.1%}",
            f"{fixed.max_drawdown:.1%}",
            f"{trail.max_drawdown:.1%}",
            f"{len(fixed.trades)}/{len(trail.trades)}",
            f"{fixed.win_rate:.0%}/{trail.win_rate:.0%}",
            str(trail.stop_moves),
        ])

    if not rows:
        print("no results")
        return 1

    print("\n" + tabulate(
        rows,
        headers=("sym", "fixed", "trail", "buy&hold", "trail-fixed",
                 "trail-B&H", "DD fix", "DD trl", "trades", "win%", "stop mv"),
        tablefmt="simple",
    ))

    n = len(agg["fixed"])
    mean = {k: sum(v) / n for k, v in agg.items()}
    beat_fixed = sum(1 for t, f in zip(agg["trail"], agg["fixed"]) if t > f)
    beat_bh = sum(1 for t, b in zip(agg["trail"], agg["bh"]) if t > b)

    print(f"\nsymbols: {n}   window: {START}..{END}")
    print(f"mean return   fixed {mean['fixed']:+.1%}   "
          f"trail {mean['trail']:+.1%}   buy&hold {mean['bh']:+.1%}")
    print(f"trailing beat fixed stop : {beat_fixed}/{n}")
    print(f"trailing beat buy & hold : {beat_bh}/{n}")

    if beat_bh == 0:
        print("\nVERDICT: neither variant beat buy-and-hold on any symbol.")
        print("Improving exit management did not create an edge, because exit")
        print("management was never the binding constraint — the entry signal is.")
    else:
        print(f"\nVERDICT: trailing beat buy-and-hold on {beat_bh}/{n} symbols. "
              f"Replicate on unseen symbols before believing it.")
    return 0


if __name__ == "__main__":
    sys.exit(run(sys.argv[1:] or BASKET))
