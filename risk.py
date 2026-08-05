"""Deterministic risk controls.

Nothing in this module consults a model, a network service, or any source of
non-determinism. Given the same inputs it returns the same decision, every time,
and every rule is readable in one sitting. Live stops must never depend on a
probabilistic evaluator.
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from datetime import date
from enum import Enum

from config import RiskLimits

logger = logging.getLogger(__name__)


class RiskDecision(str, Enum):
    """Outcome of a pre-trade risk check."""

    ALLOW = "ALLOW"
    REJECT_HALTED = "REJECT_HALTED"
    REJECT_DAILY_DRAWDOWN = "REJECT_DAILY_DRAWDOWN"
    REJECT_TOTAL_DRAWDOWN = "REJECT_TOTAL_DRAWDOWN"
    REJECT_MAX_POSITIONS = "REJECT_MAX_POSITIONS"
    REJECT_INVALID_STOP = "REJECT_INVALID_STOP"
    REJECT_SIZE_ZERO = "REJECT_SIZE_ZERO"
    REJECT_INSUFFICIENT_CASH = "REJECT_INSUFFICIENT_CASH"


@dataclass
class RiskVerdict:
    """Result of :meth:`RiskManager.evaluate_entry`.

    Attributes:
        decision: Machine-readable outcome.
        quantity: Share count to trade; 0 unless ``decision`` is ALLOW.
        reason: Human-readable explanation, always populated.
    """

    decision: RiskDecision
    quantity: float = 0.0
    reason: str = ""

    @property
    def allowed(self) -> bool:
        """True when the trade may proceed."""
        return self.decision is RiskDecision.ALLOW


@dataclass
class RiskManager:
    """Stateful, deterministic enforcement of :class:`RiskLimits`.

    The manager owns the kill switch. Once ``halted`` is set it stays set until
    :meth:`reset_session` is called explicitly — drawdown breaches never
    self-clear just because the next bar looked better.
    """

    limits: RiskLimits
    initial_capital: float
    peak_equity: float = field(init=False)
    day_start_equity: float = field(init=False)
    current_day: date | None = field(default=None, init=False)
    halted: bool = field(default=False, init=False)
    halt_reason: str = field(default="", init=False)

    def __post_init__(self) -> None:
        """Seed peak and day-start equity from the opening balance."""
        if self.initial_capital <= 0:
            raise ValueError("initial_capital must be positive")
        self.peak_equity = self.initial_capital
        self.day_start_equity = self.initial_capital

    # ── session bookkeeping ────────────────────────────────────────────────

    def start_day(self, day: date, equity: float) -> None:
        """Mark the start of a trading session.

        The daily drawdown limit is measured against the equity recorded here,
        so this must be called once per session before any entry check.
        """
        if self.current_day != day:
            self.current_day = day
            self.day_start_equity = equity
            logger.debug("session %s opened at equity %.2f", day, equity)

    def update_equity(self, equity: float) -> None:
        """Record mark-to-market equity and trip the kill switch if breached."""
        if equity > self.peak_equity:
            self.peak_equity = equity

        total_dd = self._drawdown(equity, self.peak_equity)
        if total_dd >= self.limits.max_total_drawdown:
            self._halt(
                f"total drawdown {total_dd:.2%} >= limit "
                f"{self.limits.max_total_drawdown:.2%}"
            )
            return

        daily_dd = self._drawdown(equity, self.day_start_equity)
        if daily_dd >= self.limits.max_daily_drawdown:
            self._halt(
                f"daily drawdown {daily_dd:.2%} >= limit "
                f"{self.limits.max_daily_drawdown:.2%}"
            )

    def reset_session(self) -> None:
        """Clear the kill switch. Deliberately explicit — never automatic."""
        if self.halted:
            logger.warning("risk halt cleared manually (was: %s)", self.halt_reason)
        self.halted = False
        self.halt_reason = ""

    # ── pre-trade checks ───────────────────────────────────────────────────

    def evaluate_entry(
        self,
        *,
        equity: float,
        cash: float,
        price: float,
        stop_price: float,
        open_positions: int,
    ) -> RiskVerdict:
        """Decide whether an entry may proceed, and at what size.

        Args:
            equity: Current mark-to-market equity.
            cash: Cash available to deploy.
            price: Intended entry price.
            stop_price: Protective stop; must be below *price* for a long.
            open_positions: Count of positions already open.

        Returns:
            A :class:`RiskVerdict`. ``quantity`` is non-zero only on ALLOW.
        """
        if self.halted:
            return RiskVerdict(RiskDecision.REJECT_HALTED, 0.0, self.halt_reason)

        if price <= 0:
            return RiskVerdict(RiskDecision.REJECT_INVALID_STOP, 0.0, "price must be positive")

        risk_per_share = price - stop_price
        if risk_per_share <= 0:
            return RiskVerdict(
                RiskDecision.REJECT_INVALID_STOP,
                0.0,
                f"stop {stop_price:.4f} is not below entry {price:.4f}",
            )

        if open_positions >= self.limits.max_open_positions:
            return RiskVerdict(
                RiskDecision.REJECT_MAX_POSITIONS,
                0.0,
                f"{open_positions} open >= max {self.limits.max_open_positions}",
            )

        total_dd = self._drawdown(equity, self.peak_equity)
        if total_dd >= self.limits.max_total_drawdown:
            self._halt(f"total drawdown {total_dd:.2%}")
            return RiskVerdict(RiskDecision.REJECT_TOTAL_DRAWDOWN, 0.0, self.halt_reason)

        daily_dd = self._drawdown(equity, self.day_start_equity)
        if daily_dd >= self.limits.max_daily_drawdown:
            self._halt(f"daily drawdown {daily_dd:.2%}")
            return RiskVerdict(RiskDecision.REJECT_DAILY_DRAWDOWN, 0.0, self.halt_reason)

        # Size so that a stop-out costs exactly risk_per_trade of equity.
        risk_budget = equity * self.limits.risk_per_trade
        qty = math.floor(risk_budget / risk_per_share)

        # Then clamp by notional exposure and by cash on hand.
        max_notional = equity * self.limits.max_position_pct
        qty = min(qty, math.floor(max_notional / price), math.floor(cash / price))

        if qty <= 0:
            reason = (
                f"sized to 0 (risk budget {risk_budget:.2f}, "
                f"risk/share {risk_per_share:.4f}, cash {cash:.2f})"
            )
            decision = (
                RiskDecision.REJECT_INSUFFICIENT_CASH
                if cash < price
                else RiskDecision.REJECT_SIZE_ZERO
            )
            return RiskVerdict(decision, 0.0, reason)

        return RiskVerdict(
            RiskDecision.ALLOW,
            float(qty),
            f"{qty:.0f} sh; risk {qty * risk_per_share:.2f} "
            f"({self.limits.risk_per_trade:.2%} of {equity:.2f})",
        )

    # ── internals ──────────────────────────────────────────────────────────

    @staticmethod
    def _drawdown(equity: float, reference: float) -> float:
        """Fractional decline from *reference*; 0.0 when at or above it."""
        if reference <= 0:
            return 0.0
        return max(0.0, 1.0 - equity / reference)

    def _halt(self, reason: str) -> None:
        """Trip the kill switch once and log it loudly."""
        if not self.halted:
            self.halted = True
            self.halt_reason = reason
            logger.error("RISK HALT: %s", reason)
