"""Robustness suite for the cross-sectional IC result.

`cross_sectional.py` found 12-1 momentum at IC +0.0342, t +2.55 — on a universe
of names that are large caps **today**. Every one survived, which biases
momentum upward by construction.

A true point-in-time universe needs historical index membership, which is not
available from the free data path. Two corrections are, and between them they
catch most of the damage:

  A. DEAD NAMES — add companies that failed, collapsed or were acquired out of
     the index (SIVB, FRC, CS, TWTR, ATVI, XLNX, BBBY ...). Survivorship bias
     works by excluding exactly these. If the IC holds with them included, the
     result is far harder to dismiss. Tickers that return no data are reported,
     not silently dropped — a delisted name that yfinance cannot serve is a
     known hole in the correction, and pretending otherwise would recreate the
     bias while claiming to have fixed it.

  B. SUB-PERIOD STABILITY — split the window in half. A real effect appears in
     both halves with the same sign. One that lives entirely in one half is a
     regime artefact.

    python robustness.py
"""
from __future__ import annotations

import math
import sys

import numpy as np
import pandas as pd

from cross_sectional import UNIVERSE, build_signals
from data import DataError, load_ohlcv

# Names that left the index by failure, collapse or acquisition.
DEAD_NAMES = [
    "SIVB",  # Silicon Valley Bank — failed 2023
    "FRC",   # First Republic — failed 2023
    "CS",    # Credit Suisse — collapsed into UBS 2023
    "TWTR",  # taken private 2022
    "ATVI",  # acquired by Microsoft 2023
    "VMW",   # acquired by Broadcom 2023
    "SPLK",  # acquired by Cisco 2024
    "XLNX",  # acquired by AMD 2022
    "CERN",  # acquired by Oracle 2022
    "ALXN",  # acquired by AstraZeneca 2021
    "MXIM",  # acquired by ADI 2021
    "FLIR",  # acquired by Teledyne 2021
    "BBBY",  # bankrupt 2023
    "NKLA",  # collapsed
    "PBCT", "KSU", "WLTW", "DISCA", "VIAC", "INFO",
]

START, END = "2016-01-01", "2026-06-30"
HORIZON, REBALANCE = 5, 5
FOCUS = ["mom_252_21", "reversal_5", "mom_20"]


def load_panel(symbols: list[str]) -> tuple[pd.DataFrame, list[str], list[str]]:
    """Load close prices, returning (panel, loaded, missing)."""
    series, loaded, missing = {}, [], []
    for sym in symbols:
        try:
            series[sym] = load_ohlcv(sym, START, END, "1d")["close"]
            loaded.append(sym)
        except Exception:
            missing.append(sym)
    if not series:
        raise DataError("no symbols loaded")
    panel = pd.DataFrame(series).sort_index()
    return panel, loaded, missing


def ic_stats(close: pd.DataFrame, signal: str,
             lo: pd.Timestamp | None = None,
             hi: pd.Timestamp | None = None) -> tuple[float, float, int]:
    """Return (mean IC, t-stat, periods) for *signal* over an optional window."""
    fwd = close.shift(-HORIZON) / close - 1.0
    panel = build_signals(close)[signal]

    warm = 252 + 21
    dates = close.index[warm::REBALANCE]
    if lo is not None:
        dates = [d for d in dates if d >= lo]
    if hi is not None:
        dates = [d for d in dates if d < hi]

    ics = []
    for d in dates:
        if d not in fwd.index:
            continue
        pair = pd.concat([panel.loc[d], fwd.loc[d]], axis=1,
                         keys=["sig", "fwd"]).dropna()
        if len(pair) < 10:
            continue
        r = pair.rank()
        ic = r["sig"].corr(r["fwd"])
        if np.isfinite(ic):
            ics.append(ic)

    if len(ics) < 20:
        return 0.0, 0.0, len(ics)
    arr = np.array(ics)
    sd = arr.std(ddof=1)
    t = 0.0 if sd == 0 else arr.mean() / (sd / math.sqrt(len(arr)))
    return float(arr.mean()), float(t), len(arr)


def main() -> int:
    """Run both corrections and report."""
    print("A. SURVIVORS ONLY (the original universe)")
    surv, _, _ = load_panel(UNIVERSE)
    surv = surv.dropna(axis=1, thresh=int(len(surv) * 0.8))
    print(f"   symbols: {surv.shape[1]}")

    print("\nB. ADDING DEAD NAMES")
    combined_syms = UNIVERSE + DEAD_NAMES
    comb, loaded, missing = load_panel(combined_syms)
    dead_ok = [s for s in DEAD_NAMES if s in loaded]
    print(f"   dead names requested: {len(DEAD_NAMES)}")
    print(f"   dead names WITH data: {len(dead_ok)}  {dead_ok}")
    print(f"   dead names NO data  : {len(DEAD_NAMES) - len(dead_ok)} "
          f"{[s for s in DEAD_NAMES if s not in loaded]}")
    # Keep sparse columns here: a name that stops trading mid-sample is the
    # entire point. Only require some minimum history to rank at all.
    comb = comb.dropna(axis=1, thresh=300)
    print(f"   combined usable symbols: {comb.shape[1]}")

    mid = surv.index[len(surv) // 2]
    print(f"\n{'signal':<14}{'universe':<12}{'window':<14}"
          f"{'IC mean':>10}{'t-stat':>9}{'periods':>9}")
    for sig in FOCUS:
        for label, panel in (("survivors", surv), ("with dead", comb)):
            m, t, n = ic_stats(panel, sig)
            print(f"{sig:<14}{label:<12}{'full':<14}{m:>10.4f}{t:>+9.2f}{n:>9}")
        for label, lo, hi in (("first half", None, mid), ("second half", mid, None)):
            m, t, n = ic_stats(comb, sig, lo, hi)
            print(f"{sig:<14}{'with dead':<12}{label:<14}{m:>10.4f}{t:>+9.2f}{n:>9}")
        print()

    m_s, t_s, _ = ic_stats(surv, "mom_252_21")
    m_c, t_c, _ = ic_stats(comb, "mom_252_21")
    m_1, t_1, _ = ic_stats(comb, "mom_252_21", None, mid)
    m_2, t_2, _ = ic_stats(comb, "mom_252_21", mid, None)

    print("VERDICT — 12-1 momentum")
    decay = m_s - m_c
    print(f"  survivors-only IC {m_s:+.4f} (t {t_s:+.2f})")
    print(f"  with dead names   {m_c:+.4f} (t {t_c:+.2f})   change {decay:+.4f}")
    print(f"  first half        {m_1:+.4f} (t {t_1:+.2f})")
    print(f"  second half       {m_2:+.4f} (t {t_2:+.2f})")

    both_signs = (m_1 > 0) == (m_2 > 0)
    survives = t_c > 2.0

    # A correction that could not be applied must never be reported as passed.
    # Only a minority of delisted tickers are served by the free data path, and
    # an unchanged IC across an unchanged universe proves nothing at all.
    MIN_DEAD = 8
    correction_ran = len(dead_ok) >= MIN_DEAD

    print("\n  Sub-period stability:", "PASS" if both_signs else "FAIL",
          f"(halves {m_1:+.4f} / {m_2:+.4f}, same sign: {both_signs})")

    if not correction_ran:
        print(f"  Survivorship correction: NOT PERFORMED — only {len(dead_ok)} of "
              f"{len(DEAD_NAMES)} delisted tickers returned data")
        print(f"  (needed >= {MIN_DEAD}). The universe went {surv.shape[1]} -> "
              f"{comb.shape[1]} names, so the unchanged IC is an artefact of an")
        print("  unchanged sample, NOT evidence the effect survives. Survivorship")
        print("  bias remains UNTESTED and is still the leading explanation.")
        print("\n  VERDICT: INCONCLUSIVE. Needs a paid point-in-time constituent")
        print("  source (CRSP, Compustat, Norgate) or a delisting-inclusive feed.")
    elif survives and both_signs:
        print(f"  Survivorship correction: PASS ({len(dead_ok)} dead names included)")
        print("\n  VERDICT: survives both corrections. Still needs a different")
        print("  decade before it means anything, but survivorship alone no")
        print("  longer explains it.")
    elif not survives:
        print(f"  Survivorship correction: FAIL — significance lost with "
              f"{len(dead_ok)} dead names added")
        print("\n  VERDICT: the original result was survivorship bias.")
    else:
        print("\n  VERDICT: PARTIAL — significant overall but the halves disagree")
        print("  in sign, which is a regime artefact rather than a stable effect.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
