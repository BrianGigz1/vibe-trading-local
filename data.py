"""Market data loading via yfinance, with validation and on-disk caching.

Every frame returned by this module is guaranteed to have the columns
``open, high, low, close, volume``, a sorted DatetimeIndex, and no NaNs in the
price columns. Callers may rely on that; anything short of it raises.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Final

import pandas as pd

logger = logging.getLogger(__name__)

REQUIRED_COLUMNS: Final[tuple[str, ...]] = ("open", "high", "low", "close", "volume")
CACHE_DIR: Final[Path] = Path(__file__).resolve().parent / ".cache"


class DataError(RuntimeError):
    """Raised when market data cannot be fetched or fails validation."""


def _cache_path(symbol: str, start: str, end: str, interval: str) -> Path:
    """Build a deterministic cache filename for a request."""
    safe = symbol.replace("/", "_").replace(":", "_").upper()
    return CACHE_DIR / f"{safe}_{start}_{end or 'latest'}_{interval}.parquet"


def _normalise(df: pd.DataFrame, symbol: str) -> pd.DataFrame:
    """Coerce a yfinance frame into the module's guaranteed shape.

    Args:
        df: Raw frame from yfinance.
        symbol: Ticker, used only for error messages.

    Returns:
        Frame with lowercase OHLCV columns and a sorted DatetimeIndex.

    Raises:
        DataError: If required columns are missing or no rows survive cleaning.
    """
    if df is None or df.empty:
        raise DataError(f"No data returned for {symbol!r}")

    # yfinance returns a MultiIndex when given a list of tickers; flatten it.
    if isinstance(df.columns, pd.MultiIndex):
        df = df.droplevel(-1, axis=1)

    df = df.rename(columns=lambda c: str(c).strip().lower().replace(" ", "_"))
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise DataError(f"{symbol!r} missing columns: {missing}; got {list(df.columns)}")

    df = df.loc[:, list(REQUIRED_COLUMNS)].copy()
    df.index = pd.to_datetime(df.index)
    df = df[~df.index.duplicated(keep="last")].sort_index()

    price_cols = ["open", "high", "low", "close"]
    df[price_cols] = df[price_cols].apply(pd.to_numeric, errors="coerce")
    df["volume"] = pd.to_numeric(df["volume"], errors="coerce").fillna(0.0)
    before = len(df)
    df = df.dropna(subset=price_cols)
    if len(df) < before:
        logger.warning("%s: dropped %d rows with unparseable prices", symbol, before - len(df))

    if df.empty:
        raise DataError(f"{symbol!r} had no usable rows after cleaning")
    if (df[price_cols] <= 0).any().any():
        raise DataError(f"{symbol!r} contains non-positive prices")
    return df


def load_ohlcv(
    symbol: str,
    start: str,
    end: str = "",
    interval: str = "1d",
    *,
    use_cache: bool = True,
) -> pd.DataFrame:
    """Fetch OHLCV bars for *symbol*.

    Args:
        symbol: Ticker recognised by Yahoo Finance.
        start: ISO start date (inclusive).
        end: ISO end date (exclusive); empty means up to the latest bar.
        interval: Bar size, e.g. ``1d``, ``1h``.
        use_cache: Read and write a local parquet cache.

    Returns:
        Validated OHLCV frame indexed by timestamp.

    Raises:
        DataError: On network failure, empty response, or validation failure.
    """
    if not symbol or not symbol.strip():
        raise DataError("symbol must be a non-empty string")

    cache = _cache_path(symbol, start, end, interval)
    if use_cache and cache.exists():
        try:
            cached = pd.read_parquet(cache)
            logger.info("%s: %d bars from cache", symbol, len(cached))
            return cached
        except Exception as exc:  # corrupt cache must never be fatal
            logger.warning("cache unreadable (%s); refetching", exc)

    try:
        import yfinance as yf
    except ImportError as exc:  # pragma: no cover
        raise DataError("yfinance is not installed; run: pip install -r requirements.txt") from exc

    try:
        raw = yf.download(
            symbol,
            start=start,
            end=end or None,
            interval=interval,
            auto_adjust=True,
            progress=False,
            threads=False,
        )
    except Exception as exc:
        raise DataError(f"Download failed for {symbol!r}: {exc}") from exc

    df = _normalise(raw, symbol)
    logger.info("%s: %d bars %s..%s", symbol, len(df), df.index[0].date(), df.index[-1].date())

    if use_cache:
        try:
            CACHE_DIR.mkdir(parents=True, exist_ok=True)
            df.to_parquet(cache)
        except Exception as exc:  # caching is best-effort only
            logger.warning("could not write cache: %s", exc)
    return df
