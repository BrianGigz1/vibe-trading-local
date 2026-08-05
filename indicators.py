"""Technical indicators.

All functions are **causal**: the value at bar ``i`` uses only bars ``<= i``.
That property is the difference between a backtest and a fantasy, so it is
asserted by the unit-free construction here (rolling / ewm, never centred, never
shifted backwards).
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _validate(series: pd.Series, period: int, name: str) -> None:
    """Guard common misuse before it silently produces garbage."""
    if not isinstance(series, pd.Series):
        raise TypeError(f"{name}: expected pandas Series, got {type(series).__name__}")
    if period < 2:
        raise ValueError(f"{name}: period must be >= 2, got {period}")
    if series.empty:
        raise ValueError(f"{name}: series is empty")


def sma(series: pd.Series, period: int) -> pd.Series:
    """Simple moving average over *period* bars."""
    _validate(series, period, "sma")
    return series.rolling(window=period, min_periods=period).mean()


def ema(series: pd.Series, period: int) -> pd.Series:
    """Exponential moving average with span *period*."""
    _validate(series, period, "ema")
    return series.ewm(span=period, adjust=False, min_periods=period).mean()


def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """Wilder's Relative Strength Index.

    Args:
        series: Close prices.
        period: Lookback in bars.

    Returns:
        RSI in [0, 100]; NaN for the first *period* bars.
    """
    _validate(series, period, "rsi")
    delta = series.diff()
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)

    # Wilder smoothing == EWM with alpha = 1/period.
    avg_gain = gain.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()

    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    out = 100.0 - (100.0 / (1.0 + rs))
    # avg_loss == 0 means an unbroken run of gains: RSI is 100 by definition.
    return out.where(avg_loss != 0.0, 100.0).where(avg_gain.notna())


def true_range(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    """True Range: the greater of the bar range and the two gap distances."""
    prev_close = close.shift(1)
    ranges = pd.concat(
        [high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1
    )
    return ranges.max(axis=1)


def atr(
    high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14
) -> pd.Series:
    """Average True Range using Wilder smoothing.

    Args:
        high: High prices.
        low: Low prices.
        close: Close prices.
        period: Lookback in bars.

    Returns:
        ATR series; NaN until *period* bars are available.

    Raises:
        ValueError: If the inputs are not the same length.
    """
    if not (len(high) == len(low) == len(close)):
        raise ValueError("atr: high/low/close must be the same length")
    _validate(close, period, "atr")
    tr = true_range(high, low, close)
    return tr.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()


def max_drawdown(equity: pd.Series) -> float:
    """Largest peak-to-trough decline of an equity curve, as a negative fraction."""
    if equity.empty:
        return 0.0
    running_peak = equity.cummax()
    drawdown = equity / running_peak - 1.0
    return float(drawdown.min())


def sharpe_ratio(returns: pd.Series, periods_per_year: int = 252) -> float:
    """Annualised Sharpe ratio at a zero risk-free rate.

    Returns 0.0 when volatility is zero or the sample is too small to be
    meaningful, rather than raising or emitting infinity.
    """
    clean = returns.dropna()
    if len(clean) < 2:
        return 0.0
    std = float(clean.std())
    if std == 0.0 or not np.isfinite(std):
        return 0.0
    return float(clean.mean() / std * np.sqrt(periods_per_year))
