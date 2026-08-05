"""CLI entry point.

Defaults to paper mode. Running with no arguments simulates; it never executes
anything against a real venue, because no such code path exists in this project.
"""
from __future__ import annotations

import argparse
import logging
import sys

from tabulate import tabulate

from backtest import Backtester
from config import Config, RiskLimits, StrategyParams
from data import DataError, load_ohlcv
from paper_trader import PaperTrader


def build_parser() -> argparse.ArgumentParser:
    """Construct the argument parser."""
    p = argparse.ArgumentParser(
        prog="vibe-trading",
        description="Local research and paper-trading harness. Simulation only.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--mode", choices=("paper", "backtest"), default="paper",
                   help="paper = simulated orders (default); backtest = historical run")
    p.add_argument("--symbol", default="AAPL", help="ticker to trade")
    p.add_argument("--start", default="2020-01-01", help="ISO start date")
    p.add_argument("--end", default="", help="ISO end date (blank = latest)")
    p.add_argument("--interval", default="1d", help="bar interval")
    p.add_argument("--capital", type=float, default=100_000.0, help="starting equity")

    g = p.add_argument_group("strategy")
    g.add_argument("--fast", type=int, default=20, help="fast SMA period")
    g.add_argument("--slow", type=int, default=50, help="slow SMA period")
    g.add_argument("--rsi-period", type=int, default=14, help="RSI period")
    g.add_argument("--rsi-overbought", type=float, default=70.0, help="suppress entries above")
    g.add_argument("--atr-period", type=int, default=14, help="ATR period")
    g.add_argument("--atr-stop", type=float, default=2.0, help="stop = N x ATR")
    g.add_argument("--atr-target", type=float, default=3.0, help="target = N x ATR")

    r = p.add_argument_group("risk")
    r.add_argument("--risk-per-trade", type=float, default=0.01, help="fraction of equity at risk")
    r.add_argument("--max-daily-dd", type=float, default=0.03, help="daily drawdown halt")
    r.add_argument("--max-total-dd", type=float, default=0.20, help="total drawdown halt")
    r.add_argument("--max-position-pct", type=float, default=0.25, help="max notional per position")

    p.add_argument("--live-pace", action="store_true",
                   help="paper mode: pace the replay so output is readable")
    p.add_argument("--no-cache", action="store_true", help="bypass the local data cache")
    p.add_argument("-v", "--verbose", action="store_true", help="debug logging")
    return p


def config_from_args(args: argparse.Namespace) -> Config:
    """Translate parsed arguments into a validated :class:`Config`."""
    return Config(
        symbol=args.symbol.upper(),
        start=args.start,
        end=args.end,
        interval=args.interval,
        initial_capital=args.capital,
        risk=RiskLimits(
            risk_per_trade=args.risk_per_trade,
            max_daily_drawdown=args.max_daily_dd,
            max_total_drawdown=args.max_total_dd,
            max_position_pct=args.max_position_pct,
        ),
        strategy=StrategyParams(
            fast_period=args.fast,
            slow_period=args.slow,
            rsi_period=args.rsi_period,
            rsi_overbought=args.rsi_overbought,
            atr_period=args.atr_period,
            atr_stop_multiple=args.atr_stop,
            atr_target_multiple=args.atr_target,
        ),
    )


def run_backtest(cfg: Config, use_cache: bool) -> int:
    """Run a historical backtest and print the summary."""
    df = load_ohlcv(cfg.symbol, cfg.start, cfg.end, cfg.interval, use_cache=use_cache)
    result = Backtester(cfg).run(df)

    print("\n" + tabulate(result.summary_rows(), headers=("metric", "value"), tablefmt="simple"))

    if result.trades:
        rows = [
            (
                t.entry_date.date(), t.exit_date.date(), f"{t.entry_price:.2f}",
                f"{t.exit_price:.2f}", f"{t.quantity:.0f}", f"{t.pnl:+,.2f}",
                f"{t.return_pct:+.2%}", t.reason,
            )
            for t in result.trades[-10:]
        ]
        print("\nlast trades")
        print(tabulate(rows, headers=("in", "out", "entry", "exit", "qty", "pnl", "ret", "why")))

    # The comparison that decides whether trading was worth doing at all.
    if result.excess_return < 0:
        print(
            f"\nNOTE: this strategy UNDERPERFORMED buy-and-hold by "
            f"{abs(result.excess_return):.2%} (information ratio "
            f"{result.information_ratio:+.4f}). A positive total return is not a "
            f"result unless it beats simply holding the asset."
        )
    return 0


def run_paper(cfg: Config, use_cache: bool, live: bool) -> int:
    """Run the simulated paper-trading engine."""
    df = load_ohlcv(cfg.symbol, cfg.start, cfg.end, cfg.interval, use_cache=use_cache)
    PaperTrader(cfg).run(df, live=live)
    return 0


def main(argv: list[str] | None = None) -> int:
    """Parse arguments and dispatch. Returns a process exit code."""
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)-7s %(name)s: %(message)s",
    )

    try:
        cfg = config_from_args(args)
    except ValueError as exc:
        print(f"invalid configuration: {exc}", file=sys.stderr)
        return 2

    print(f"mode={args.mode}  symbol={cfg.symbol}  {cfg.start}..{cfg.end or 'latest'}")
    print("SIMULATION ONLY — no broker connectivity exists in this project.")

    try:
        if args.mode == "backtest":
            return run_backtest(cfg, use_cache=not args.no_cache)
        return run_paper(cfg, use_cache=not args.no_cache, live=args.live_pace)
    except DataError as exc:
        print(f"data error: {exc}", file=sys.stderr)
        return 3
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 4
    except KeyboardInterrupt:
        print("\ninterrupted", file=sys.stderr)
        return 130


if __name__ == "__main__":
    sys.exit(main())
