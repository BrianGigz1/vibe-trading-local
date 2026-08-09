"""What `low_vol_60` is actually worth after costs - the crypto analogue of
`xs_portfolio.py`.

`crypto_ic.py` measured IC +0.1050 (t +5.73). `crypto_kill.py` showed it is
neither a BTC/ETH proxy (IC +0.0749, t +4.18 on 16 alts) nor one regime
(+0.1247 / +0.0904 across halves). That is as far as a correlation can take it.

This prices it. The same step for equities reversed a decision to spend money on
data, because the *ceiling* turned out to be unattractive - so it is run before
anything is built on top, not after.

Two constructions, because they are not equally real:

* **long_top3** - long the least-volatile third, equal weight. This is the
  version that can actually be traded: **Alpaca does not support shorting
  crypto**, so the short leg is unavailable at the venue this project uses.
* **neutral_t3b3** - long top third, short bottom third, dollar-neutral. The
  theoretical ceiling. Reported for reference and **labelled unimplementable**,
  because quoting a number you cannot trade as if you could is how a backtest
  becomes a story. It also ignores borrow and funding on alt shorts, which are
  worse than the spread being charged here.

Benchmarks are the honest ones: hold BTC, and hold the equal-weight basket. A
strategy on this universe has to beat *not trading* on it.

Costs are charged per unit of weight traded at each rebalance. Turnover is
measured target-to-target rather than against drifted weights, which
**overstates** it slightly - deliberately, since the conservative direction is
the safe one for a go/no-go decision.

    python crypto_portfolio.py
    python crypto_portfolio.py --costs 0 10 25 50
"""
from __future__ import annotations

import argparse
import sys

import numpy as np
import pandas as pd

from crypto_data import CRYPTO_UNIVERSE, load_crypto
from crypto_ic import REBALANCE, START, WARMUP
from cross_sectional import build_signals
from data import DataError

SIGNAL = "low_vol_60"
BARS_PER_YEAR = 365.0          # crypto trades every day; no session calendar


def load_close() -> pd.DataFrame:
    """Wide close-price frame for the tradable universe."""
    series = {}
    for sym in CRYPTO_UNIVERSE:
        try:
            series[sym] = load_crypto(sym, START)["close"]
        except DataError as exc:
            print(f"  ! {sym}: {exc}", file=sys.stderr)
    close = pd.DataFrame(series).sort_index()
    return close.dropna(axis=1, thresh=400)


def weights(panel: pd.DataFrame, dates: list, cols: list,
            neutral: bool) -> pd.DataFrame:
    """Target weights at each rebalance date.

    Long the top third by signal; when *neutral*, short the bottom third with
    matching gross so the book is dollar-neutral.
    """
    w = pd.DataFrame(0.0, index=dates, columns=cols)
    for d in dates:
        s = panel.loc[d].dropna()
        if len(s) < 6:
            continue
        k = max(1, len(s) // 3)
        ranked = s.sort_values()
        top = ranked.tail(k).index
        w.loc[d, top] = 1.0 / k
        if neutral:
            bot = ranked.head(k).index
            w.loc[d, bot] = -1.0 / k
    return w


def run(ret: pd.DataFrame, w: pd.DataFrame, cost_bps: float) -> tuple[pd.Series, float]:
    """Daily portfolio returns after costs, and average annual turnover.

    Weights are held between rebalances. Cost is charged on the day the book
    changes, proportional to the total weight traded.
    """
    held = w.reindex(ret.index).ffill().fillna(0.0)
    gross = (held.shift(1) * ret).sum(axis=1)      # shift: no lookahead

    traded = w.diff().abs().sum(axis=1)
    traded.iloc[0] = w.iloc[0].abs().sum()
    cost = traded.reindex(ret.index).fillna(0.0) * (cost_bps / 10_000.0)

    net = (gross - cost).fillna(0.0)
    years = max(len(ret) / BARS_PER_YEAR, 1e-9)
    return net, float(traded.sum() / years)


def metrics(r: pd.Series) -> dict:
    """CAGR, Sharpe and max drawdown from a daily return series."""
    if r.empty or not np.isfinite(r).all():
        r = r.replace([np.inf, -np.inf], 0.0).fillna(0.0)
    eq = (1.0 + r).cumprod()
    years = max(len(r) / BARS_PER_YEAR, 1e-9)
    final = float(eq.iloc[-1]) if len(eq) else 1.0
    cagr = final ** (1.0 / years) - 1.0 if final > 0 else -1.0
    sd = float(r.std(ddof=1))
    sharpe = (float(r.mean()) / sd * np.sqrt(BARS_PER_YEAR)) if sd > 0 else 0.0
    dd = float((eq / eq.cummax() - 1.0).min()) if len(eq) else 0.0
    return {"cagr": cagr, "sharpe": sharpe, "dd": dd}


def main() -> int:
    """Price the signal at several cost levels against holding the basket."""
    ap = argparse.ArgumentParser(description="Cost model for the crypto signal")
    ap.add_argument("--costs", type=float, nargs="*", default=[0, 10, 25, 50],
                    metavar="BPS", help="bps per unit of weight traded")
    args = ap.parse_args()

    close = load_close()
    ret = close.pct_change().fillna(0.0)
    panel = build_signals(close)[SIGNAL]
    dates = [d for d in close.index[WARMUP::REBALANCE] if d in ret.index]
    cols = list(close.columns)

    span = close.index[WARMUP:]
    ret_ev = ret.loc[span]
    years = len(span) / BARS_PER_YEAR
    print(f"signal {SIGNAL}   pairs {len(cols)}   "
          f"{span[0].date()} .. {span[-1].date()}   {years:.1f}y")
    print(f"rebalance every {REBALANCE}d   {len(dates)} rebalances\n")

    print(f"{'construction':<16}{'cost':>7}{'CAGR':>9}{'Sharpe':>9}"
          f"{'maxDD':>9}{'turnover/yr':>13}")
    rows = {}
    for label, neutral in (("long_top3", False), ("neutral_t3b3", True)):
        w = weights(panel, dates, cols, neutral)
        for c in args.costs:
            net, turn = run(ret_ev, w.loc[[d for d in dates if d in span]], c)
            m = metrics(net)
            rows[(label, c)] = m
            print(f"{label:<16}{c:>5.0f}bp{m['cagr']:>8.1%}{m['sharpe']:>9.2f}"
                  f"{m['dd']:>9.1%}{turn:>12.1f}x")

    # --- benchmarks: not trading ----------------------------------------
    print()
    btc = metrics(ret_ev["BTC/USD"]) if "BTC/USD" in ret_ev else None
    eqw = metrics(ret_ev.mean(axis=1))
    if btc:
        print(f"{'HOLD BTC':<16}{'-':>7}{btc['cagr']:>8.1%}{btc['sharpe']:>9.2f}"
              f"{btc['dd']:>9.1%}{'0.0x':>13}")
    print(f"{'HOLD basket':<16}{'-':>7}{eqw['cagr']:>8.1%}{eqw['sharpe']:>9.2f}"
          f"{eqw['dd']:>9.1%}{'0.0x':>13}")

    # --- verdict ---------------------------------------------------------
    real = rows.get(("long_top3", 25.0)) or rows.get(("long_top3", args.costs[-1]))
    bench = max([b for b in (btc, eqw) if b], key=lambda m: m["sharpe"])
    print("\n" + "=" * 70)
    if real is None:
        print("no tradable construction scored")
        return 1
    print(f"TRADABLE version (long-only, Alpaca cannot short crypto) at 25bp:")
    print(f"  Sharpe {real['sharpe']:.2f}  CAGR {real['cagr']:.1%}  "
          f"maxDD {real['dd']:.1%}")
    print(f"BEST benchmark (not trading):")
    print(f"  Sharpe {bench['sharpe']:.2f}  CAGR {bench['cagr']:.1%}  "
          f"maxDD {bench['dd']:.1%}")
    print()
    if real["sharpe"] > bench["sharpe"] and real["dd"] > bench["dd"]:
        print("BEATS holding on BOTH axes. First construction in this project to")
        print("do so. Survivorship is still uncorrected and remains an upper bound.")
    elif real["sharpe"] > bench["sharpe"]:
        print("Better Sharpe, worse drawdown than holding. A real but partial win;")
        print("judge it against the drawdown budget, not the return.")
    else:
        print("DOES NOT beat holding the basket. The IC is real and the money is")
        print("not - the same outcome as the equity signal, for the same reason:")
        print("a genuine edge too small to survive costs at this breadth.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
