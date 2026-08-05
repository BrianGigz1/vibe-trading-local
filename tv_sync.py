"""Sync TradingView state into the ledger and the Obsidian note.

Reads the Pine scorecard from a live TradingView Desktop session and records it
alongside the Python results, so both engines land in the same places.

The three ways this silently fails, and what is done about each:

  TradingView not running        -> status TV_UNAVAILABLE, written explicitly
  CDP target picking is wrong    -> iterate every tab until the study is found
  Study present but not computed -> status TV_NOT_COMPUTED (dataSize 0)

**Nothing is ever recorded as a successful sync unless real values were read.**
An absent TradingView must look like an absence in the ledger, not like a zero.
That distinction is the difference between a cross-check and a fabrication.

    python tv_sync.py            # sync, write to ledger + Obsidian
    python tv_sync.py --check    # report only, write nothing
"""
from __future__ import annotations

import argparse
import json
import os
import sqlite3
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import journal
import tv_cdp
from collector import init_db

# The bundled `tv` CLI was tried first and abandoned: it selects its CDP target
# with `targets.find(...)` (the first match) and offers no override, so on a
# multi-window session it reads the wrong chart. Observed directly — the CLI
# reported 0 studies on all three tabs while the chart demonstrably had one.
# tv_cdp enumerates every target and picks one with a COMPUTED study.
TVUnavailable = tv_cdp.CDPError


def clean(v: str) -> object:
    """Coerce a scorecard value to float when it plausibly is one."""
    num = str(v).replace(",", "").replace("%", "").replace("x", "").strip()
    try:
        return float(num)
    except ValueError:
        return v


def read_tv() -> dict:
    """Read symbol, study name and scorecard values from a live chart."""
    st = tv_cdp.read_state()
    study = st["studies"][0]
    values = {k: clean(v) for k, v in (study.get("values") or {}).items()}
    # Pine table rows, when the script draws one, arrive as "label | value".
    for tb in study.get("tables") or []:
        for row in tb:
            if "|" in row:
                k, _, v = row.partition("|")
                values.setdefault(k.strip(), clean(v))
    return {
        "symbol": st.get("symbol"),
        "resolution": st.get("resolution"),
        "study": study.get("name"),
        "data_size": study.get("dataSize"),
        "target": st.get("target_id"),
        "values": values,
    }


def ensure_table(con: sqlite3.Connection) -> None:
    """Create the TradingView sync table."""
    con.execute("""
        CREATE TABLE IF NOT EXISTS tv_sync (
            id         INTEGER PRIMARY KEY AUTOINCREMENT,
            synced_at  TEXT NOT NULL,
            status     TEXT NOT NULL,
            symbol     TEXT,
            study      TEXT,
            tab_index  INTEGER,
            payload    TEXT,
            note       TEXT
        )
    """)
    con.commit()


def record(con: sqlite3.Connection, status: str, **kw) -> None:
    """Write one sync row. Status is always recorded, success or not."""
    con.execute(
        "INSERT INTO tv_sync (synced_at, status, symbol, study, tab_index, payload, note)"
        " VALUES (?,?,?,?,?,?,?)",
        (datetime.now(timezone.utc).isoformat(), status, kw.get("symbol"),
         kw.get("study"), kw.get("tab_index"),
         json.dumps(kw.get("payload")) if kw.get("payload") else None,
         kw.get("note")),
    )
    con.commit()


def write_note(status: str, symbol: str | None, study: str | None,
               sc: dict | None, note: str | None) -> None:
    """Append the sync result to today's Obsidian note — including failures."""
    try:
        d = journal.journal_dir()
        d.mkdir(parents=True, exist_ok=True)
        day = datetime.now().strftime("%Y-%m-%d")
        path = d / f"{day}.md"
        if not path.exists():
            journal._ensure_note(path, day)

        icon = "✅" if status == "SYNCED" else "⚠️"
        lines = [f"### {datetime.now().strftime('%H:%M:%S')} — TRADINGVIEW SYNC "
                 f"`[{status}]` {icon}\n"]
        if status == "SYNCED":
            lines.append(f"- **Chart:** `{symbol}` | **Study:** `{study}`\n")
            for k, v in (sc or {}).items():
                lines.append(f"- **{k}:** `{v}`\n")
        else:
            lines.append(f"- **No data recorded.** {note}\n")
            lines.append("- This is an explicit absence, not a zero — "
                         "TradingView contributed nothing to today's figures.\n")
        lines.append("\n")
        with path.open("a", encoding="utf-8") as fh:
            fh.writelines(lines)
    except OSError as exc:
        print(f"  ! obsidian write failed: {exc}", file=sys.stderr)


def main() -> int:
    """Sync, or record loudly why it could not."""
    ap = argparse.ArgumentParser(description="Sync TradingView into the ledger")
    ap.add_argument("--check", action="store_true", help="report only, write nothing")
    args = ap.parse_args()

    con = init_db()
    ensure_table(con)

    try:
        tv = read_tv()
    except TVUnavailable as exc:
        msg = str(exc)
        print(f"TV_UNAVAILABLE: {msg}")
        if not args.check:
            record(con, "TV_UNAVAILABLE", note=msg)
            write_note("TV_UNAVAILABLE", None, None, None, msg)
            print("recorded as an explicit absence (not a zero)")
        con.close()
        return 1

    symbol, name, sc = tv["symbol"], tv["study"], tv["values"]
    print(f"SYNCED  symbol {symbol} {tv['resolution']}  study '{name}'  "
          f"dataSize {tv['data_size']}")
    for k, v in sc.items():
        print(f"  {k:<24} {v}")

    if not sc:
        msg = f"study '{name}' present but produced no values (dataSize {tv['data_size']})"
        print(f"TV_NOT_COMPUTED: {msg}")
        if not args.check:
            record(con, "TV_NOT_COMPUTED", symbol=symbol, study=name, note=msg)
            write_note("TV_NOT_COMPUTED", symbol, name, None, msg)
        con.close()
        return 1

    if args.check:
        print("\n--check: nothing written")
        con.close()
        return 0

    record(con, "SYNCED", symbol=symbol, study=name, payload=sc)
    write_note("SYNCED", symbol, name, sc, None)
    n = con.execute("SELECT COUNT(*) FROM tv_sync WHERE status='SYNCED'").fetchone()[0]
    print(f"\nrecorded. successful syncs in ledger: {n}")
    con.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
