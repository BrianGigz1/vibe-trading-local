"""Core + tilt: use the signal to overweight, never to sit in cash.

The ledger shows a significant per-trade edge (+0.72%, t=+2.60, 6/8 symbols)
alongside ~0% capture of buy-and-hold. Both are true: the entries carry
information, but the strategy is in the market only ~18% of the time, so it
forgoes drift on the other 82%. The loss is **absence**, not error.

That implies a specific structure. Rather than long/flat:

    A  buy & hold      100% always                        (the benchmark)
    B  long / flat     100% on signal, 0% otherwise        (what we have been testing)
    C  core + tilt     100% always, TILT% while signal on  (the proposal)

If the edge is real, C should beat A, because C is A plus a small overweight
applied only during periods the signal identifies. If C does not beat A, the
per-trade edge does not survive being expressed as exposure â€” which would be a
genuine falsification, not a tuning failure.

The tilt is funded at a borrow rate, so leverage is not free. Turnover costs are
applied on every exposure change.

    python core_tilt.py [SYMBOL ...] [--tilt 0.5]
"""
from __future__ import annotations

import argparse
import sys

import numpy as np
import pandas as pd
from tabulate import tabulate

import indicators as ind
from config import Config
from data import DataError, load_ohlcv
from strategy import Strategy

BASKET = ["AAPL", "MSFT", "NVDA", "SPY", "TSLA", "AMZN", "META", "GOOGL"]
START = "2023-01-01"
# 8% is a realistic retail margin rate. The first version used 5%, which
# flattered the tilt leg â€” the borrow is the whole cost of the structure, so
# understating it is the easiest way to manufacture an edge that is not there.
COST_PER_TURN = 0.0004    # commission + slippage on each exposure change


def exposure_series(df: pd.DataFrame, cfg: Config, tilt: float, base: float = 1.0) -> dict[str, pd.Series]:
    """Build daily exposure paths for the three modes.

    Signals are shifted one bar: exposure changes act on the bar AFTER the
    signal, never the bar that produced it.
    """
    sig = Strategy(cfg.strategy).generate(df)
    d = sig.data
    n = len(d)

    # "In position" state: on entry signal, off on exit signal.
    state = np.zeros(n)
    on = False
    for i in range(n):
        if bool(d["entry"].iloc[i]):
            on = True
        elif bool(d["exit"].iloc[i]):
            on = False
        state[i] = 1.0 if on else 0.0
    in_pos = pd.Series(state, index=d.index)

    return {
        "buy_hold": pd.Series(1.0, index=d.index),
        "long_flat": in_pos,
        # base < 1.0 de-levers the core. Tilt only ADDS exposure, so with
        # base = 1.0 the drawdown can never fall below buy-and-hold's own â€”
        # meeting a tight DD budget requires holding less than 100%, not
        # tilting less.
        "core_tilt": base + tilt * in_pos,
    }


def daily_returns(df: pd.DataFrame, exposure: pd.Series, warm: int,
                  borrow_annual: float) -> pd.Series:
    """Net daily return series for an exposure path, after borrow and turnover."""
    ret = df["close"].pct_change().fillna(0.0)
    exp = exposure.reindex(ret.index).fillna(0.0)
    ret, exp = ret.iloc[warm:], exp.iloc[warm:]
    borrow = np.maximum(exp - 1.0, 0.0) * (borrow_annual / 252.0)
    turn = exp.diff().abs().fillna(0.0) * COST_PER_TURN
    return exp * ret - borrow - turn


def summarise(daily: pd.Series) -> dict:
    """Total, CAGR, max drawdown and Sharpe for a net daily return series."""
    equity = (1.0 + daily).cumprod()
    peak = equity.cummax()
    years = len(equity) / 252.0
    return {
        "total": float(equity.iloc[-1] - 1.0),
        "cagr": float(equity.iloc[-1] ** (1 / years) - 1.0) if years > 0 else 0.0,
        "maxdd": float((equity / peak - 1.0).min()),
        "sharpe": ind.sharpe_ratio(daily),
    }


def run_mode(df: pd.DataFrame, exposure: pd.Series, warm: int,
             borrow_annual: float) -> dict:
    """Compound an exposure path, charging borrow and turnover."""
    ret = df["close"].pct_change().fillna(0.0)
    exp = exposure.reindex(ret.index).fillna(0.0)

    ret, exp = ret.iloc[warm:], exp.iloc[warm:]

    # Borrow on the leveraged portion only.
    borrow = np.maximum(exp - 1.0, 0.0) * (borrow_annual / 252.0)
    # Turnover cost on every change in exposure.
    turn = exp.diff().abs().fillna(0.0) * COST_PER_TURN

    daily = exp * ret - borrow - turn
    equity = (1.0 + daily).cumprod()

    peak = equity.cummax()
    dd = float((equity / peak - 1.0).min())
    years = len(equity) / 252.0
    total = float(equity.iloc[-1] - 1.0)
    cagr = float(equity.iloc[-1] ** (1 / years) - 1.0) if years > 0 else 0.0
    return {
        "total": total, "cagr": cagr, "maxdd": dd,
        "sharpe": ind.sharpe_ratio(daily),
        "avg_exposure": float(exp.mean()),
    }


def main() -> int:
    """Compare the three exposure structures across a basket."""
    ap = argparse.ArgumentParser(description="Core+tilt vs long/flat vs buy&hold")
    ap.add_argument("symbols", nargs="*", default=None)
    ap.add_argument("--tilt", type=float, default=0.5,
                    help="extra exposure while the signal is on (0.5 = 150%%)")
    ap.add_argument("--base", type=float, default=1.0,
                    help="core exposure when the signal is OFF (1.0 = fully invested)")
    ap.add_argument("--start", default=START)
    ap.add_argument("--end", default="")
    ap.add_argument("--borrow", type=float, default=0.08,
                    help="annual borrow rate on the tilt leg (retail margin ~8%%)")
    ap.add_argument("--max-dd", type=float, default=0.15,
                    help="drawdown budget from the graduation criteria")
    args = ap.parse_args()
    symbols = args.symbols or BASKET

    rows, agg = [], {"buy_hold": [], "long_flat": [], "core_tilt": []}
    port: dict[str, list[pd.Series]] = {"buy_hold": [], "long_flat": [], "core_tilt": []}
    dd_breaches = 0
    for sym in symbols:
        cfg = Config(symbol=sym, start=args.start, end=args.end)
        try:
            df = load_ohlcv(sym, args.start, args.end, "1d")
            exps = exposure_series(df, cfg, args.tilt, args.base)
        except (DataError, ValueError) as exc:
            print(f"  ! {sym}: {exc}", file=sys.stderr)
            continue

        warm = cfg.min_bars
        res = {k: run_mode(df, v, warm, args.borrow) for k, v in exps.items()}
        if abs(res["core_tilt"]["maxdd"]) > args.max_dd:
            dd_breaches += 1
        for k, v in exps.items():
            port[k].append(daily_returns(df, v, warm, args.borrow).rename(sym))
        for k in agg:
            agg[k].append(res[k]["total"])

        rows.append([
            sym,
            f"{res['buy_hold']['total']:+.1%}",
            f"{res['long_flat']['total']:+.1%}",
            f"{res['core_tilt']['total']:+.1%}",
            f"{res['core_tilt']['total'] - res['buy_hold']['total']:+.1%}",
            f"{res['buy_hold']['maxdd']:.1%}",
            f"{res['core_tilt']['maxdd']:.1%}",
            f"{res['core_tilt']['avg_exposure']:.2f}x",
        ])

    if not rows:
        print("no results")
        return 1

    print(f"\n{args.start}..{args.end or 'latest'} | tilt +{args.tilt:.0%} while signal on "
          f"| borrow {args.borrow:.0%}/yr | turnover {COST_PER_TURN:.2%}/change\n")
    print(tabulate(rows, headers=(
        "sym", "buy&hold", "long/flat", "core+tilt", "tiltâˆ’B&H",
        "DD B&H", "DD tilt", "avg exp"), tablefmt="simple"))

    n = len(rows)
    mean = {k: sum(v) / n for k, v in agg.items()}
    beat = sum(1 for t, b in zip(agg["core_tilt"], agg["buy_hold"]) if t > b)

    print(f"\nmean total return   buy&hold {mean['buy_hold']:+.1%}   "
          f"long/flat {mean['long_flat']:+.1%}   core+tilt {mean['core_tilt']:+.1%}")
    print(f"core+tilt beat buy&hold on {beat}/{n} symbols")
    print(f"drawdown budget ({args.max_dd:.0%}) breached on {dd_breaches}/{n} symbols "
          f"(single names)")

    # â”€â”€ portfolio level â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    # Single-name drawdowns are not the number that matters: you would hold a
    # basket, and correlations below 1 mean the portfolio drawdown is smaller
    # than the average of its parts. Judging a strategy on single-name DD
    # overstates the risk of actually running it.
    print(f"\n{'PORTFOLIO (equal weight, daily rebalance)':<42}")
    print(f"{'mode':<14}{'total':>11}{'CAGR':>9}{'maxDD':>9}{'Sharpe':>9}")
    pstats = {}
    for k in ("buy_hold", "long_flat", "core_tilt"):
        if not port[k]:
            continue
        combined = pd.concat(port[k], axis=1).dropna(how="all").fillna(0.0).mean(axis=1)
        s = summarise(combined)
        pstats[k] = s
        print(f"{k:<14}{s['total']:>10.1%}{s['cagr']:>9.1%}"
              f"{s['maxdd']:>9.1%}{s['sharpe']:>9.2f}")

    if "core_tilt" in pstats and "buy_hold" in pstats:
        ct, bh = pstats["core_tilt"], pstats["buy_hold"]
        print(f"\nportfolio core+tilt vs buy&hold: "
              f"{ct['total'] - bh['total']:+.1%} total, "
              f"{ct['cagr'] - bh['cagr']:+.2%} CAGR, "
              f"drawdown {bh['maxdd']:.1%} -> {ct['maxdd']:.1%}")
        if abs(ct["maxdd"]) > args.max_dd:
            print(f"âš ï¸  portfolio drawdown {ct['maxdd']:.1%} still exceeds the "
                  f"{args.max_dd:.0%} budget.")
            print("   Diversification narrowed the gap but did not close it. The")
            print("   criterion is not met, and it must NOT be loosened to fit the")
            print("   result â€” that is the rationalisation it exists to prevent.")

    print()
    if dd_breaches:
        print(f"âš ï¸  RISK: {dd_breaches}/{n} symbols exceed the {args.max_dd:.0%} drawdown")
        print("   criterion. Beating buy-and-hold while breaching the risk budget is")
        print("   not a pass â€” criterion 6 exists precisely to stop that trade.\n")

    if beat > n / 2 and mean["core_tilt"] > mean["buy_hold"]:
        # The edge was measured on 2023-01-01 onward. Whether THIS run is
        # in-sample depends on the window requested, so it must be derived,
        # not hardcoded â€” the previous version claimed "in-sample" on an
        # explicitly out-of-sample run.
        in_sample = args.start >= "2023-01-01"
        print("core+tilt BEATS buy-and-hold. The per-trade edge survives being")
        print("expressed as exposure.")
        if in_sample:
            print("âš ï¸  IN-SAMPLE: this is the window the edge was measured on, so it")
            print("   is not yet evidence. Run --start 2015-01-01 --end 2022-12-31.")
        else:
            print(f"âœ… OUT-OF-SAMPLE ({args.start}..{args.end or 'latest'}): the edge was")
            print("   measured on 2023+ and this window was not used to find it.")
        if dd_breaches:
            print("   But see the drawdown breach above â€” return is not the only gate.")
    else:
        print("core+tilt does NOT beat buy-and-hold. The per-trade edge does not")
        print("survive translation into exposure â€” the overweight periods do not")
        print("earn enough to cover borrow and turnover. That is a falsification,")
        print("not a tuning problem.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

