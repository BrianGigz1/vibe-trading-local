"""Probe whether Stooq serves delisted US tickers.

yfinance returned 2 of 20 delisted names, so the survivorship correction could
not be performed. Stooq keeps history for dead tickers, and it is free. If it
serves them, the survivorship question is answerable today at zero cost.

    python stooq_probe.py
"""
from __future__ import annotations

import io
import sys
import urllib.error
import urllib.request

import pandas as pd

DEAD = ["SIVB", "FRC", "CS", "TWTR", "ATVI", "VMW", "SPLK", "XLNX", "CERN",
        "ALXN", "MXIM", "FLIR", "BBBY", "PBCT", "KSU", "WLTW", "DISCA", "VIAC"]
ALIVE = ["AAPL", "MSFT"]   # control: proves the fetch path itself works

URL = "https://stooq.com/q/d/l/?s={sym}.us&i=d"
UA = {"User-Agent": "Mozilla/5.0 (research; pandas)"}


class BlockedError(RuntimeError):
    """Raised when the source returns an anti-bot page instead of data."""


def fetch(sym: str) -> pd.DataFrame | None:
    """Return a daily OHLCV frame for *sym*, or None if unavailable."""
    try:
        req = urllib.request.Request(URL.format(sym=sym.lower()), headers=UA)
        with urllib.request.urlopen(req, timeout=20) as r:
            raw = r.read().decode("utf-8", errors="replace")
    except (urllib.error.URLError, TimeoutError) as exc:
        print(f"  {sym:<6} fetch error: {exc}", file=sys.stderr)
        return None

    # Distinguish "ticker not served" from "we were blocked". Stooq answers
    # automated requests with an HTTP 200 bot-protection interstitial, which
    # looks identical to missing data unless the body is inspected. Reporting
    # that as "no data" would be a false negative about the ticker.
    if raw.lstrip().startswith("<!DOCTYPE") or "<noscript>" in raw[:400]:
        raise BlockedError(f"{sym}: bot-protection page, not CSV")
    if "Date" not in raw[:200]:
        return None                      # genuinely no data for this ticker
    try:
        df = pd.read_csv(io.StringIO(raw))
    except Exception:
        return None
    if df.empty or "Close" not in df.columns:
        return None
    df["Date"] = pd.to_datetime(df["Date"])
    return df.set_index("Date").sort_index()


def main() -> int:
    """Report coverage, and abort loudly if the source is blocking us.

    The control names run FIRST. If a live, unambiguously-listed ticker cannot
    be fetched, nothing can be concluded about the delisted ones — reporting
    them as "no data" would be a false negative caused by the fetch path, not
    by the source's coverage.
    """
    print("control (must succeed before anything else is meaningful):")
    for s in ALIVE:
        try:
            df = fetch(s)
        except BlockedError as exc:
            print(f"  {s:<6} BLOCKED — {exc}")
            print("\nABORT: the source is serving an anti-bot page, not data.")
            print("Nothing can be concluded about delisted coverage from this run.")
            print("Stooq needs a browser session; use a data vendor instead.")
            return 2
        if df is None:
            print(f"  {s:<6} FAIL (no data for a live ticker — fetch path broken)")
            print("\nABORT: control failed; delisted results would be meaningless.")
            return 2
        print(f"  {s:<6} OK  {len(df)} bars, last {df.index[-1].date()}")

    print("\ndelisted:")
    ok, miss = [], []
    for s in DEAD:
        try:
            df = fetch(s)
        except BlockedError:
            print(f"  {s:<6} blocked mid-run")
            continue
        if df is None or len(df) < 100:
            miss.append(s)
            print(f"  {s:<6} no data")
        else:
            ok.append(s)
            print(f"  {s:<6} OK  {len(df):>5} bars  "
                  f"{df.index[0].date()} .. {df.index[-1].date()}")

    print(f"\nserved {len(ok)}/{len(DEAD)} delisted tickers")
    print("SUFFICIENT — survivorship correction is possible." if len(ok) >= 8
          else "INSUFFICIENT — cannot correct for survivorship from this source.")
    print("ok:", ok)
    print("missing:", miss)
    return 0


if __name__ == "__main__":
    sys.exit(main())
