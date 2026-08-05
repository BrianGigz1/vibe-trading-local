"""Signal generation: entries, exits, stops and targets.

Signals are computed on **closed bars only** and are intended to be acted on at
the next bar's open. `Strategy.generate` shifts entry/exit flags by one bar so a
backtest cannot accidentally trade on information from the bar it is evaluating.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import pandas as pd

import indicators as ind
from config import StrategyParams

logger = logging.getLogger(__name__)

SIGNAL_COLUMNS = ("fast", "slow", "rsi", "atr", "entry", "exit", "stop", "target")


@dataclass
class SignalFrame:
    """Indicator and signal columns aligned to the source bars.

    Attributes:
        data: Frame containing OHLCV plus the columns in ``SIGNAL_COLUMNS``.
        warmup: Number of leading bars with incomplete indicators.
    """

    data: pd.DataFrame
    warmup: int

    @property
    def tradable(self) -> pd.DataFrame:
        """Rows after warm-up, where every indicator is defined."""
        return self.data.iloc[self.warmup :]


class Strategy:
    """Trend-following entries with an RSI filter and ATR-based exits.

    Entry: fast MA crosses above slow MA while RSI is below the overbought
    threshold. Exit: fast MA crosses back below slow MA. Stops and targets are
    placed at ATR multiples from the entry reference price.

    The RSI filter exists to avoid buying an already-extended move; it is a
    structural choice, not a tuned parameter.
    """

    def __init__(self, params: StrategyParams) -> None:
        """Store validated parameters.

        Args:
            params: Signal configuration; already validated by its dataclass.
        """
        self.p = params

    def generate(self, df: pd.DataFrame) -> SignalFrame:
        """Compute indicators and signals for *df*.

        Args:
            df: OHLCV frame from :mod:`data`.

        Returns:
            A :class:`SignalFrame`.

        Raises:
            ValueError: If *df* is too short for the configured lookbacks.
        """
        warmup = max(self.p.slow_period, self.p.rsi_period, self.p.atr_period)
        if len(df) <= warmup + 1:
            raise ValueError(
                f"need > {warmup + 1} bars for these parameters, got {len(df)}"
            )

        out = df.copy()
        close, high, low = out["close"], out["high"], out["low"]

        out["fast"] = ind.sma(close, self.p.fast_period)
        out["slow"] = ind.sma(close, self.p.slow_period)
        out["rsi"] = ind.rsi(close, self.p.rsi_period)
        out["atr"] = ind.atr(high, low, close, self.p.atr_period)

        above = out["fast"] > out["slow"]
        cross_up = above & ~above.shift(1, fill_value=False)
        cross_dn = ~above & above.shift(1, fill_value=False)

        raw_entry = cross_up & (out["rsi"] < self.p.rsi_overbought)
        raw_exit = cross_dn

        # Act on the NEXT bar. Without this shift the backtest would trade on a
        # close it could not have known at decision time.
        out["entry"] = raw_entry.shift(1, fill_value=False).fillna(False).astype(bool)
        out["exit"] = raw_exit.shift(1, fill_value=False).fillna(False).astype(bool)

        atr_prev = out["atr"].shift(1)
        ref = out["open"]
        out["stop"] = ref - self.p.atr_stop_multiple * atr_prev
        out["target"] = ref + self.p.atr_target_multiple * atr_prev

        n_entries = int(out["entry"].sum())
        logger.info(
            "signals: %d entries, %d exits over %d bars (warmup %d)",
            n_entries, int(out["exit"].sum()), len(out), warmup,
        )
        if n_entries == 0:
            logger.warning("strategy produced no entries for this data/parameters")

        return SignalFrame(data=out, warmup=warmup)

    def describe(self) -> str:
        """One-line human summary of the configured rules."""
        return (
            f"SMA({self.p.fast_period}/{self.p.slow_period}) cross-up, "
            f"RSI({self.p.rsi_period}) < {self.p.rsi_overbought:g}; "
            f"stop {self.p.atr_stop_multiple:g}xATR, "
            f"target {self.p.atr_target_multiple:g}xATR"
        )
