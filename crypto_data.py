"""Keyless crypto OHLCV from Alpaca, shaped like :mod:`data`.

Returns exactly the frame `data.load_ohlcv` returns — lowercase
``open/high/low/close/volume``, sorted DatetimeIndex — so every tool built today
(`calibrate`, `cross_sectional`, `backtest`, `overnight`) works on crypto without
modification.

Crypto is worth testing precisely because it breaks assumptions the equity tests
relied on:

  * **24/7.** There is no overnight gap, so the overnight/intraday split that was
    tested on equities is not even defined here.
  * **No market hours**, so "daily bar" means a UTC day, not a session.
  * **Different volatility regime** — roughly 3-5x equity daily vol, which raises
    the bar for any signal to clear statistically.

Alpaca's crypto market data needs **no API key**. History begins around 2021, so
samples are shorter than the equity work — a real limitation, not a detail.

    python crypto_data.py            # smoke test the loader
"""
from __future__ import annotations

import sys
from datetime import datetime, timezone

import pandas as pd

from data import DataError

# Alpaca's tradable USD pairs, widest first.
CRYPTO_UNIVERSE = [
    "BTC/USD", "ETH/USD", "LTC/USD", "BCH/USD", "LINK/USD", "UNI/USD",
    "AAVE/USD", "AVAX/USD", "DOT/USD", "SOL/USD", "DOGE/USD", "SHIB/USD",
    "XTZ/USD", "SUSHI/USD", "YFI/USD", "MKR/USD", "GRT/USD", "CRV/USD",
]

_CACHE: dict[tuple, pd.DataFrame] = {}


def load_crypto(symbol: str, start: str, end: str = "", timeframe: str = "1d") -> pd.DataFrame:
    """Fetch OHLCV bars for *symbol*, in the same shape as `data.load_ohlcv`.

    Args:
        symbol: Alpaca pair, e.g. ``BTC/USD``.
        start: ISO start date.
        end: ISO end date; empty means now.
        timeframe: ``1d`` or ``1h``.

    Returns:
        Validated OHLCV frame indexed by timestamp.

    Raises:
        DataError: On fetch failure or an empty/unusable response.
    """
    key = (symbol, start, end, timeframe)
    if key in _CACHE:
        return _CACHE[key]

    try:
        from alpaca.data.historical import CryptoHistoricalDataClient
        from alpaca.data.requests import CryptoBarsRequest
        from alpaca.data.timeframe import TimeFrame
    except ImportError as exc:
        raise DataError("alpaca-py not installed; pip install alpaca-py") from exc

    tf = TimeFrame.Hour if timeframe == "1h" else TimeFrame.Day
    try:
        client = CryptoHistoricalDataClient()      # no credentials needed
        req = CryptoBarsRequest(
            symbol_or_symbols=[symbol],
            timeframe=tf,
            start=datetime.fromisoformat(start).replace(tzinfo=timezone.utc),
            end=(datetime.fromisoformat(end).replace(tzinfo=timezone.utc)
                 if end else None),
        )
        df = client.get_crypto_bars(req).df
    except Exception as exc:
        raise DataError(f"{symbol}: {type(exc).__name__}: {str(exc)[:120]}") from exc

    if df is None or df.empty:
        raise DataError(f"{symbol}: no bars returned")

    # The frame is MultiIndexed (symbol, timestamp); drop to a plain time index.
    if isinstance(df.index, pd.MultiIndex):
        df = df.reset_index(level=0, drop=True)

    df = df.rename(columns=str.lower)
    need = ["open", "high", "low", "close", "volume"]
    missing = [c for c in need if c not in df.columns]
    if missing:
        raise DataError(f"{symbol}: missing columns {missing}")

    out = df.loc[:, need].copy()
    out.index = pd.to_datetime(out.index).tz_localize(None)
    out = out[~out.index.duplicated(keep="last")].sort_index()
    out = out.dropna(subset=["open", "high", "low", "close"])
    if out.empty or (out[["open", "high", "low", "close"]] <= 0).any().any():
        raise DataError(f"{symbol}: no usable rows after cleaning")

    _CACHE[key] = out
    return out


def main() -> int:
    """Smoke test: how much history does each pair actually have?"""
    start = "2021-01-01"
    print(f"{'pair':<12}{'bars':>7}{'from':>12}{'to':>12}{'last close':>14}")
    ok = 0
    for sym in CRYPTO_UNIVERSE:
        try:
            df = load_crypto(sym, start)
        except DataError as exc:
            print(f"{sym:<12}{'—':>7}  {str(exc)[:52]}")
            continue
        ok += 1
        print(f"{sym:<12}{len(df):>7}{str(df.index[0].date()):>12}"
              f"{str(df.index[-1].date()):>12}{df['close'].iloc[-1]:>14,.4f}")
    print(f"\n{ok}/{len(CRYPTO_UNIVERSE)} pairs loaded, no API key used")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
