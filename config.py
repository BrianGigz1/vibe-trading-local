"""Central configuration: trading parameters and risk thresholds.

Every value here is a plain, inspectable constant or an environment override.
Nothing in this module is derived from a model, a network call, or anything
non-deterministic — risk limits must be auditable by reading the file.

Environment overrides use the ``VT_`` prefix, e.g. ``VT_RISK_PER_TRADE=0.005``.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Final

ENV_PREFIX: Final[str] = "VT_"


def _env_float(name: str, default: float) -> float:
    """Read a float from the environment, falling back to *default*."""
    raw = os.getenv(ENV_PREFIX + name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return float(raw)
    except ValueError as exc:
        raise ValueError(f"{ENV_PREFIX}{name}={raw!r} is not a valid float") from exc


def _env_int(name: str, default: int) -> int:
    """Read an int from the environment, falling back to *default*."""
    raw = os.getenv(ENV_PREFIX + name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return int(raw)
    except ValueError as exc:
        raise ValueError(f"{ENV_PREFIX}{name}={raw!r} is not a valid int") from exc


def _env_str(name: str, default: str) -> str:
    """Read a string from the environment, falling back to *default*."""
    raw = os.getenv(ENV_PREFIX + name)
    return default if raw is None or raw.strip() == "" else raw.strip()


@dataclass(frozen=True)
class RiskLimits:
    """Hard risk boundaries enforced by :mod:`risk`.

    Attributes:
        risk_per_trade: Fraction of equity risked on a single position, sized
            against the stop distance. 0.01 == 1% of equity at risk.
        max_daily_drawdown: Fraction of start-of-day equity that, once lost,
            halts all new entries for the session.
        max_total_drawdown: Fraction of peak equity that, once lost, halts
            trading entirely.
        max_position_pct: Ceiling on notional exposure of any single position
            as a fraction of equity, applied after risk-based sizing.
        max_open_positions: Maximum simultaneous open positions.
    """

    risk_per_trade: float = 0.01
    max_daily_drawdown: float = 0.03
    max_total_drawdown: float = 0.20
    max_position_pct: float = 0.25
    max_open_positions: int = 5

    def __post_init__(self) -> None:
        """Reject nonsensical limits at construction rather than at trade time."""
        checks = {
            "risk_per_trade": (self.risk_per_trade, 0.0, 0.10),
            "max_daily_drawdown": (self.max_daily_drawdown, 0.0, 1.0),
            "max_total_drawdown": (self.max_total_drawdown, 0.0, 1.0),
            "max_position_pct": (self.max_position_pct, 0.0, 1.0),
        }
        for name, (value, lo, hi) in checks.items():
            if not lo < value <= hi:
                raise ValueError(f"{name}={value} outside allowed range ({lo}, {hi}]")
        if self.max_open_positions < 1:
            raise ValueError("max_open_positions must be >= 1")


@dataclass(frozen=True)
class StrategyParams:
    """Signal-generation parameters.

    Attributes:
        fast_period: Fast moving-average lookback, in bars.
        slow_period: Slow moving-average lookback, in bars.
        rsi_period: RSI lookback, in bars.
        rsi_overbought: Entries are suppressed at or above this RSI.
        atr_period: ATR lookback used for stop placement.
        atr_stop_multiple: Stop distance as a multiple of ATR.
        atr_target_multiple: Take-profit distance as a multiple of ATR.
    """

    fast_period: int = 20
    slow_period: int = 50
    rsi_period: int = 14
    rsi_overbought: float = 70.0
    atr_period: int = 14
    atr_stop_multiple: float = 2.0
    atr_target_multiple: float = 3.0

    def __post_init__(self) -> None:
        """Validate lookbacks so a bad config fails before any data is fetched."""
        if self.fast_period >= self.slow_period:
            raise ValueError("fast_period must be shorter than slow_period")
        for name in ("fast_period", "slow_period", "rsi_period", "atr_period"):
            if getattr(self, name) < 2:
                raise ValueError(f"{name} must be >= 2")
        if self.atr_stop_multiple <= 0 or self.atr_target_multiple <= 0:
            raise ValueError("ATR multiples must be positive")


@dataclass(frozen=True)
class Config:
    """Top-level runtime configuration.

    Attributes:
        symbol: Ticker to trade.
        start: ISO start date for historical data.
        end: ISO end date, or empty for "today".
        interval: Bar interval understood by yfinance (e.g. ``1d``).
        initial_capital: Starting equity for backtests and paper runs.
        commission_pct: Per-side commission as a fraction of notional.
        slippage_pct: Per-side slippage assumption as a fraction of price.
        risk: Hard risk boundaries.
        strategy: Signal parameters.
    """

    symbol: str = field(default_factory=lambda: _env_str("SYMBOL", "AAPL"))
    start: str = field(default_factory=lambda: _env_str("START", "2020-01-01"))
    end: str = field(default_factory=lambda: _env_str("END", ""))
    interval: str = field(default_factory=lambda: _env_str("INTERVAL", "1d"))
    initial_capital: float = field(
        default_factory=lambda: _env_float("INITIAL_CAPITAL", 100_000.0)
    )
    commission_pct: float = field(
        default_factory=lambda: _env_float("COMMISSION_PCT", 0.0003)
    )
    slippage_pct: float = field(default_factory=lambda: _env_float("SLIPPAGE_PCT", 0.0005))
    risk: RiskLimits = field(
        default_factory=lambda: RiskLimits(
            risk_per_trade=_env_float("RISK_PER_TRADE", 0.01),
            max_daily_drawdown=_env_float("MAX_DAILY_DRAWDOWN", 0.03),
            max_total_drawdown=_env_float("MAX_TOTAL_DRAWDOWN", 0.20),
            max_position_pct=_env_float("MAX_POSITION_PCT", 0.25),
            max_open_positions=_env_int("MAX_OPEN_POSITIONS", 5),
        )
    )
    strategy: StrategyParams = field(
        default_factory=lambda: StrategyParams(
            fast_period=_env_int("FAST_PERIOD", 20),
            slow_period=_env_int("SLOW_PERIOD", 50),
            rsi_period=_env_int("RSI_PERIOD", 14),
            rsi_overbought=_env_float("RSI_OVERBOUGHT", 70.0),
            atr_period=_env_int("ATR_PERIOD", 14),
            atr_stop_multiple=_env_float("ATR_STOP_MULTIPLE", 2.0),
            atr_target_multiple=_env_float("ATR_TARGET_MULTIPLE", 3.0),
        )
    )

    def __post_init__(self) -> None:
        """Validate capital and cost assumptions."""
        if self.initial_capital <= 0:
            raise ValueError("initial_capital must be positive")
        if self.commission_pct < 0 or self.slippage_pct < 0:
            raise ValueError("commission_pct and slippage_pct must be non-negative")

    @property
    def min_bars(self) -> int:
        """Minimum bars needed before any signal is valid."""
        return max(
            self.strategy.slow_period,
            self.strategy.rsi_period,
            self.strategy.atr_period,
        ) + 1
