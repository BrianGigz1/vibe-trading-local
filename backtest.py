"""Event-driven backtester with a buy-and-hold benchmark.

The benchmark is not decoration. Net profit alone routinely reads green while a
strategy loses badly to simply holding the asset, so ``BacktestResult`` always
carries buy-and-hold return, excess return and information ratio alongside the
headline figures.

Bars are walked in order rather than vectorised in one shot: intrabar stop and
target checks cannot be expressed correctly as a single vector operation, and
getting them wrong silently inflates results.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date

import numpy as np
import pandas as pd

import indicators as ind
from config import Config
from position_manager import Position, update_stop
from risk import RiskManager
from strategy import Strategy

logger = logging.getLogger(__name__)


@dataclass
class Trade:
    """A completed round trip."""

    entry_date: pd.Timestamp
    exit_date: pd.Timestamp
    entry_price: float
    exit_price: float
    quantity: float
    pnl: float
    reason: str

    @property
    def return_pct(self) -> float:
        """Return on the position's entry notional."""
        base = self.entry_price * self.quantity
        return 0.0 if base == 0 else self.pnl / base


@dataclass
class BacktestResult:
    """Performance summary of a single run."""

    symbol: str
    initial_capital: float
    final_equity: float
    equity_curve: pd.Series
    trades: list[Trade] = field(default_factory=list)
    buy_hold_return: float = 0.0
    halted: bool = False
    halt_reason: str = ""
    stop_moves: int = 0

    @property
    def total_return(self) -> float:
        """Fractional return on initial capital."""
        return self.final_equity / self.initial_capital - 1.0

    @property
    def excess_return(self) -> float:
        """Strategy return minus buy-and-hold. The number that decides it."""
        return self.total_return - self.buy_hold_return

    @property
    def max_drawdown(self) -> float:
        """Largest peak-to-trough decline, as a negative fraction."""
        return ind.max_drawdown(self.equity_curve)

    @property
    def win_rate(self) -> float:
        """Fraction of closed trades with positive P&L."""
        if not self.trades:
            return 0.0
        return sum(1 for t in self.trades if t.pnl > 0) / len(self.trades)

    @property
    def sharpe(self) -> float:
        """Annualised Sharpe of the equity curve."""
        return ind.sharpe_ratio(self.equity_curve.pct_change())

    @property
    def information_ratio(self) -> float:
        """Excess return per unit of tracking error against buy-and-hold.

        Returns 0.0 when there is no variation to measure. Unlike Sharpe, this
        answers "was trading better than holding?" — the relevant question.
        """
        rets = self.equity_curve.pct_change().dropna()
        if len(rets) < 2:
            return 0.0
        bh_daily = (1.0 + self.buy_hold_return) ** (1.0 / len(rets)) - 1.0
        active = rets - bh_daily
        std = float(active.std())
        if std == 0.0 or not np.isfinite(std):
            return 0.0
        return float(active.mean() / std * np.sqrt(252))

    def summary_rows(self) -> list[tuple[str, str]]:
        """Rows for tabular display."""
        return [
            ("Symbol", self.symbol),
            ("Initial capital", f"{self.initial_capital:,.2f}"),
            ("Final equity", f"{self.final_equity:,.2f}"),
            ("Total return", f"{self.total_return:+.2%}"),
            ("Buy & hold return", f"{self.buy_hold_return:+.2%}"),
            ("Excess vs buy & hold", f"{self.excess_return:+.2%}"),
            ("Information ratio", f"{self.information_ratio:+.4f}"),
            ("Max drawdown", f"{self.max_drawdown:.2%}"),
            ("Sharpe", f"{self.sharpe:.4f}"),
            ("Trades", str(len(self.trades))),
            ("Win rate", f"{self.win_rate:.1%}"),
            ("Risk halt", self.halt_reason or "none"),
        ]


class Backtester:
    """Runs a :class:`Strategy` over historical bars under :class:`RiskManager`.

    Args:
        config: Runtime configuration.
        trailing: When True, manage exits with the breakeven-then-ATR-trail
            stop from :mod:`position_manager` instead of a fixed stop/target.
            Exposed as a flag so the two can be A/B compared on identical bars —
            one change, measured, rather than swapped in on intuition.
    """

    def __init__(self, config: Config, *, trailing: bool = False) -> None:
        """Bind configuration and construct the strategy and risk manager."""
        self.cfg = config
        self.trailing = trailing
        self.strategy = Strategy(config.strategy)
        self.risk = RiskManager(config.risk, config.initial_capital)

    def _fill_price(self, price: float, side: str) -> float:
        """Apply slippage in the direction that hurts."""
        slip = self.cfg.slippage_pct
        return price * (1.0 + slip) if side == "buy" else price * (1.0 - slip)

    def _commission(self, notional: float) -> float:
        """Commission on a notional amount."""
        return abs(notional) * self.cfg.commission_pct

    def run(self, df: pd.DataFrame) -> BacktestResult:
        """Execute the backtest.

        Args:
            df: OHLCV frame from :mod:`data`.

        Returns:
            A populated :class:`BacktestResult`.

        Raises:
            ValueError: If the frame is too short for the configured lookbacks.
        """
        signals = self.strategy.generate(df)
        bars = signals.tradable
        if bars.empty:
            raise ValueError("no tradable bars after indicator warm-up")

        cash = self.cfg.initial_capital
        qty = 0.0
        entry_price = 0.0
        entry_dt: pd.Timestamp | None = None
        stop = target = 0.0
        trades: list[Trade] = []
        equity_points: list[float] = []
        index: list[pd.Timestamp] = []
        managed: Position | None = None
        stop_moves = 0

        for ts, row in bars.iterrows():
            day: date = ts.date()
            equity = cash + qty * float(row["close"])
            self.risk.start_day(day, equity)

            # ── manage an open position first ──────────────────────────────
            if qty > 0:
                exit_reason = ""
                fill = 0.0
                # Breach is tested against the stop as it stood entering this
                # bar, BEFORE any trailing update. Moving the stop first using
                # this bar's data and then testing it would let the stop dodge
                # a move it should have been hit by.
                if float(row["low"]) <= stop:
                    exit_reason, fill = "stop", stop
                elif not self.trailing and float(row["high"]) >= target:
                    exit_reason, fill = "target", target
                elif bool(row["exit"]):
                    exit_reason, fill = "signal", self._fill_price(float(row["open"]), "sell")

                # Then ratchet the stop for the next bar.
                if not exit_reason and self.trailing and managed is not None:
                    bar_atr = float(row["atr"])
                    if np.isfinite(bar_atr) and bar_atr > 0:
                        upd = update_stop(managed, float(row["close"]), bar_atr)
                        if upd.moved:
                            stop = managed.current_sl
                            stop_moves += 1

                if exit_reason:
                    proceeds = qty * fill
                    cash += proceeds - self._commission(proceeds)
                    pnl = proceeds - self._commission(proceeds) - entry_price * qty
                    trades.append(
                        Trade(
                            entry_date=entry_dt,  # type: ignore[arg-type]
                            exit_date=ts,
                            entry_price=entry_price,
                            exit_price=fill,
                            quantity=qty,
                            pnl=pnl,
                            reason=exit_reason,
                        )
                    )
                    qty, entry_price, entry_dt, managed = 0.0, 0.0, None, None

            # ── consider a new entry ───────────────────────────────────────
            if qty == 0 and bool(row["entry"]):
                raw_stop = float(row["stop"])
                open_px = float(row["open"])
                if np.isfinite(raw_stop) and raw_stop > 0:
                    fill = self._fill_price(open_px, "buy")
                    equity = cash
                    verdict = self.risk.evaluate_entry(
                        equity=equity,
                        cash=cash,
                        price=fill,
                        stop_price=raw_stop,
                        open_positions=0,
                    )
                    if verdict.allowed:
                        cost = verdict.quantity * fill
                        cash -= cost + self._commission(cost)
                        qty = verdict.quantity
                        entry_price = fill
                        entry_dt = ts
                        stop = raw_stop
                        target = float(row["target"])
                        if self.trailing:
                            managed = Position(
                                symbol=self.cfg.symbol, side="long",
                                entry_price=fill, current_sl=raw_stop,
                                be_trigger_dist=1.0,
                                be_offset=fill * self.cfg.commission_pct * 2,
                                atr_mult=self.cfg.strategy.atr_stop_multiple,
                            )
                    else:
                        logger.debug("%s entry rejected: %s", ts.date(), verdict.reason)

            equity = cash + qty * float(row["close"])
            self.risk.update_equity(equity)
            equity_points.append(equity)
            index.append(ts)

        # Close any position still open at the end, at the last close.
        if qty > 0:
            last = bars.iloc[-1]
            fill = self._fill_price(float(last["close"]), "sell")
            proceeds = qty * fill
            cash += proceeds - self._commission(proceeds)
            trades.append(
                Trade(
                    entry_date=entry_dt,  # type: ignore[arg-type]
                    exit_date=bars.index[-1],
                    entry_price=entry_price,
                    exit_price=fill,
                    quantity=qty,
                    pnl=proceeds - self._commission(proceeds) - entry_price * qty,
                    reason="end_of_data",
                )
            )
            equity_points[-1] = cash
            qty = 0.0

        first_close = float(bars.iloc[0]["close"])
        last_close = float(bars.iloc[-1]["close"])
        buy_hold = last_close / first_close - 1.0 if first_close > 0 else 0.0

        return BacktestResult(
            symbol=self.cfg.symbol,
            initial_capital=self.cfg.initial_capital,
            final_equity=equity_points[-1] if equity_points else self.cfg.initial_capital,
            equity_curve=pd.Series(equity_points, index=pd.Index(index, name="date")),
            trades=trades,
            buy_hold_return=buy_hold,
            halted=self.risk.halted,
            halt_reason=self.risk.halt_reason,
            stop_moves=stop_moves,
        )
