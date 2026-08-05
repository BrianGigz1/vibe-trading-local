"""Point-in-time universe IC test — vendor-agnostic.

The survivorship question cannot be answered from free sources: yfinance serves
2 of 20 delisted tickers, Stooq blocks automation. It needs a delisting-inclusive
vendor. This module makes that a one-command answer the moment such data exists,
without committing to any particular vendor.

Input shape — deliberately the lowest common denominator, because Norgate,
Sharadar, Polygon and CRSP exports all reduce to it:

    <data-dir>/
        AAPL.csv          columns: Date, Close   (extra columns ignored)
        SIVB.csv          delisted names included — that is the whole point
        ...
        constituents.csv  optional: symbol, start_date, end_date

`constituents.csv` gives TRUE point-in-time membership: a name is only ranked on
dates it was actually in the index. Without it, every name in the directory is
treated as always-present — which still fixes survivorship *if the directory
includes dead names*, but does not fix look-ahead in index membership.

The report always prints BOTH the survivor-only and the full-universe IC, so the
survivorship delta is explicit rather than inferred.

    python pit_universe.py <data-dir>
    python pit_universe.py <data-dir> --signal mom_252_21
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from cross_sectional import build_signals

HORIZON, REBALANCE, WARMUP = 5, 5, 273


def load_dir(d: Path) -> tuple[pd.DataFrame, pd.DataFrame | None, list[str]]:
    """Load per-symbol CSVs into a wide close panel.

    Returns:
        (close panel, constituents frame or None, symbols loaded)
    """
    if not d.is_dir():
        raise NotADirectoryError(d)

    cons = None
    cpath = d / "constituents.csv"
    if cpath.exists():
        cons = pd.read_csv(cpath)
        cons.columns = [c.strip().lower() for c in cons.columns]
        for col in ("start_date", "end_date"):
            if col in cons.columns:
                cons[col] = pd.to_datetime(cons[col], errors="coerce")

    series, loaded = {}, []
    for f in sorted(d.glob("*.csv")):
        if f.name.lower() == "constituents.csv":
            continue
        try:
            df = pd.read_csv(f)
        except Exception as exc:
            print(f"  ! {f.name}: {exc}", file=sys.stderr)
            continue
        cols = {c.strip().lower(): c for c in df.columns}
        dcol = cols.get("date") or cols.get("datetime")
        ccol = cols.get("close") or cols.get("adj close") or cols.get("close/last")
        if not dcol or not ccol:
            print(f"  ! {f.name}: needs Date and Close columns, got {list(df.columns)}",
                  file=sys.stderr)
            continue
        s = pd.Series(
            pd.to_numeric(df[ccol].astype(str).str.replace("$", "", regex=False),
                          errors="coerce").values,
            index=pd.to_datetime(df[dcol], errors="coerce"),
        ).dropna().sort_index()
        if len(s) < WARMUP + 50:
            continue
        sym = f.stem.upper()
        series[sym] = s
        loaded.append(sym)

    if not series:
        raise ValueError("no usable CSVs found")
    return pd.DataFrame(series).sort_index(), cons, loaded


def apply_membership(close: pd.DataFrame, cons: pd.DataFrame | None) -> pd.DataFrame:
    """Blank out prices for dates a symbol was not an index member."""
    if cons is None or "symbol" not in cons.columns:
        return close
    masked = close.copy()
    for sym in masked.columns:
        rows = cons[cons["symbol"].str.upper() == sym]
        if rows.empty:
            masked[sym] = np.nan          # not a member in any window
            continue
        keep = pd.Series(False, index=masked.index)
        for _, r in rows.iterrows():
            lo = r.get("start_date", pd.NaT)
            hi = r.get("end_date", pd.NaT)
            m = pd.Series(True, index=masked.index)
            if pd.notna(lo):
                m &= masked.index >= lo
            if pd.notna(hi):
                m &= masked.index <= hi
            keep |= m
        masked.loc[~keep, sym] = np.nan
    return masked


def ic(close: pd.DataFrame, signal: str) -> tuple[float, float, int]:
    """Mean IC, t-stat and period count for *signal* on a close panel."""
    fwd = close.shift(-HORIZON) / close - 1.0
    panel = build_signals(close)[signal]
    dates = close.index[WARMUP::REBALANCE]

    ics = []
    for d in dates:
        if d not in fwd.index:
            continue
        pair = pd.concat([panel.loc[d], fwd.loc[d]], axis=1,
                         keys=["sig", "fwd"]).dropna()
        if len(pair) < 10:
            continue
        r = pair.rank()
        v = r["sig"].corr(r["fwd"])
        if np.isfinite(v):
            ics.append(v)

    if len(ics) < 20:
        return 0.0, 0.0, len(ics)
    a = np.array(ics)
    sd = a.std(ddof=1)
    t = 0.0 if sd == 0 else a.mean() / (sd / math.sqrt(len(a)))
    return float(a.mean()), float(t), len(a)


def main() -> int:
    """Run the survivor-only vs full-universe comparison."""
    ap = argparse.ArgumentParser(description="Point-in-time universe IC test")
    ap.add_argument("data_dir")
    ap.add_argument("--signal", default="mom_252_21")
    ap.add_argument("--alive-file", default="",
                    help="optional txt file listing still-listed symbols, one per line")
    args = ap.parse_args()

    try:
        close, cons, loaded = load_dir(Path(args.data_dir))
    except (NotADirectoryError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    print(f"symbols loaded : {len(loaded)}")
    print(f"bars           : {len(close)}")
    print(f"constituents   : {'yes — true point-in-time membership' if cons is not None else 'NO — names treated as always-present'}")

    # A symbol whose series ends well before the panel does is effectively dead.
    last = close.index[-1]
    dead = [c for c in close.columns
            if close[c].last_valid_index() is not None
            and (last - close[c].last_valid_index()).days > 90]
    if args.alive_file:
        alive = {l.strip().upper() for l in Path(args.alive_file).read_text().split() if l.strip()}
        dead = [c for c in close.columns if c not in alive]
    print(f"dead / delisted: {len(dead)}  {dead[:12]}{'...' if len(dead) > 12 else ''}")

    if len(dead) < 8:
        print("\n⚠️  Fewer than 8 delisted names in this directory. The survivorship")
        print("   correction CANNOT be performed — the same limitation that blocked")
        print("   yfinance. Export a delisting-inclusive universe from the vendor.")

    full = apply_membership(close, cons)
    surv = close.drop(columns=dead, errors="ignore")

    m_s, t_s, n_s = ic(surv, args.signal)
    m_f, t_f, n_f = ic(full, args.signal)

    print(f"\n{'universe':<22}{'IC mean':>10}{'t-stat':>9}{'periods':>9}")
    print(f"{'survivors only':<22}{m_s:>10.4f}{t_s:>+9.2f}{n_s:>9}")
    print(f"{'full (with dead)':<22}{m_f:>10.4f}{t_f:>+9.2f}{n_f:>9}")
    print(f"{'survivorship delta':<22}{m_s - m_f:>10.4f}")

    print()
    if len(dead) < 8:
        print("VERDICT: INCONCLUSIVE — not enough delisted names to correct.")
    elif t_f > 2.0:
        print(f"VERDICT: {args.signal} SURVIVES survivorship correction "
              f"(t {t_f:+.2f} with dead names included).")
        print("Next: a different decade, and a cost model at realistic turnover.")
    else:
        print(f"VERDICT: {args.signal} FAILS — significance disappears once dead")
        print(f"names are included (t {t_s:+.2f} -> {t_f:+.2f}). The original result")
        print("was survivorship bias.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
