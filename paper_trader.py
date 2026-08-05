"""Simulated execution engine — mock orders only.

There is no broker connectivity anywhere in this module, and no code path that
could acquire it. Orders are filled against observed bars and recorded in an
in-memory ledger. This is the only execution surface in the project, and it is
deliberately incapable of reaching a real venue.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum

import pandas as pd

from config import Config
from risk import RiskManager
from strategy import Strategy

logger = logging.getLogger(__name__)


class OrderSide(str, Enum):
    """Order direction."""

    BUY = "BUY"
    SELL = "SELL"


@dataclass
class MockOrder:
    """A simulated order and its fill.

    Attributes:
        timestamp: Wall-clock time the order was created.
        bar_date: Bar the order was generated from.
        side: Buy or sell.
        symbol: Ticker.
        quantity: Share count.
        price: Simulated fill price, slippage included.
        reason: Why the order was raised.
        order_id: Sequential simulated identifier.
    """

    timestamp: datetime
    bar_date: pd.Timestamp
    side: OrderSide
    symbol: str
    quantity: float
    price: float
    reason: str
    order_id: int

    @property
    def notional(self) -> float:
        """Absolute cash value of the fill."""
        return self.quantity * self.price

    def __str__(self) -> str:
        """Compact one-line rendering for the console log."""
        return (
            f"[MOCK #{self.order_id:04d}] {self.side.value:4} {self.quantity:>8.0f} "
            f"{self.symbol:<6} @ {self.price:>10.4f}  ({self.reason})"
        )


@dataclass
class PaperAccount:
    """In-memory account state.

    Attributes:
        cash: Uninvested cash.
        quantity: Shares held.
        avg_price: Average entry price of the open position.
        orders: Every simulated order raised this session.
    """

    cash: float
    quantity: float = 0.0
    avg_price: float = 0.0
    orders: list[MockOrder] = field(default_factory=list)

    def equity(self, mark: float) -> float:
        """Mark-to-market equity at price *mark*."""
        return self.cash + self.quantity * mark


class PaperTrader:
    """Dry-run execution engine.

    Replays bars in sequence, applying the same strategy and the same risk
    manager the backtester uses, and prints each simulated order. ``live=False``
    replays history as fast as possible; ``live=True`` merely paces the replay
    with a sleep so output is readable — it does not connect to anything.
    """

    def __init__(self, config: Config) -> None:
        """Bind configuration and construct strategy, risk manager and account."""
        self.cfg = config
        self.strategy = Strategy(config.strategy)
        self.risk = RiskManager(config.risk, config.initial_capital)
        self.account = PaperAccount(cash=config.initial_capital)
        self._next_id = 1
        self._stop = 0.0
        self._target = 0.0

    def _order(
        self, bar_date: pd.Timestamp, side: OrderSide, qty: float, price: float, reason: str
    ) -> MockOrder:
        """Create, record and log a simulated order."""
        order = MockOrder(
            timestamp=datetime.now(timezone.utc),
            bar_date=bar_date,
            side=side,
            symbol=self.cfg.symbol,
            quantity=qty,
            price=price,
            reason=reason,
            order_id=self._next_id,
        )
        self._next_id += 1
        self.account.orders.append(order)
        logger.info("%s", order)
        print(order)
        return order

    def _fill(self, price: float, side: OrderSide) -> float:
        """Apply slippage against the trader."""
        slip = self.cfg.slippage_pct
        return price * (1 + slip) if side is OrderSide.BUY else price * (1 - slip)

    def _commission(self, notional: float) -> float:
        """Commission on a notional amount."""
        return abs(notional) * self.cfg.commission_pct

    def run(self, df: pd.DataFrame, *, live: bool = False, delay: float = 0.05) -> PaperAccount:
        """Replay *df* through the strategy, emitting mock orders.

        Args:
            df: OHLCV frame from :mod:`data`.
            live: Pace the replay with *delay* between bars.
            delay: Seconds to sleep between bars when ``live`` is set.

        Returns:
            The final :class:`PaperAccount`.

        Raises:
            ValueError: If there are too few bars for the configured lookbacks.
        """
        signals = self.strategy.generate(df)
        bars = signals.tradable
        if bars.empty:
            raise ValueError("no tradable bars after indicator warm-up")

        print(f"\nPAPER MODE (simulated orders only) — {self.strategy.describe()}")
        print(f"{'bar':<12}{'event':<26}{'equity':>14}")
        print("-" * 52)

        for ts, row in bars.iterrows():
            close = float(row["close"])
            self.risk.start_day(ts.date(), self.account.equity(close))

            if self.account.quantity > 0:
                reason, px = "", 0.0
                if float(row["low"]) <= self._stop:
                    reason, px = "stop hit", self._stop
                elif float(row["high"]) >= self._target:
                    reason, px = "target hit", self._target
                elif bool(row["exit"]):
                    reason, px = "exit signal", self._fill(float(row["open"]), OrderSide.SELL)

                if reason:
                    qty = self.account.quantity
                    self._order(ts, OrderSide.SELL, qty, px, reason)
                    proceeds = qty * px
                    self.account.cash += proceeds - self._commission(proceeds)
                    self.account.quantity = 0.0
                    self.account.avg_price = 0.0

            if self.account.quantity == 0 and bool(row["entry"]):
                stop = float(row["stop"])
                if pd.notna(stop) and stop > 0:
                    px = self._fill(float(row["open"]), OrderSide.BUY)
                    verdict = self.risk.evaluate_entry(
                        equity=self.account.cash,
                        cash=self.account.cash,
                        price=px,
                        stop_price=stop,
                        open_positions=0,
                    )
                    if verdict.allowed:
                        self._order(ts, OrderSide.BUY, verdict.quantity, px, verdict.reason)
                        cost = verdict.quantity * px
                        self.account.cash -= cost + self._commission(cost)
                        self.account.quantity = verdict.quantity
                        self.account.avg_price = px
                        self._stop = stop
                        self._target = float(row["target"])
                    else:
                        print(f"{ts.date()!s:<12}{'entry blocked':<26}"
                              f"{self.account.equity(close):>14,.2f}   {verdict.reason}")

            equity = self.account.equity(close)
            self.risk.update_equity(equity)
            if self.risk.halted:
                print(f"\nRISK HALT — {self.risk.halt_reason}. No further entries.")
                break
            if live:
                time.sleep(max(0.0, delay))

        final_mark = float(bars.iloc[-1]["close"])
        print("-" * 52)
        print(f"{'FINAL':<12}{'':<26}{self.account.equity(final_mark):>14,.2f}")
        print(f"orders raised: {len(self.account.orders)} (all simulated, none sent)")
        return self.account
