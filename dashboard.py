"""Live dashboard — everything in one page, refreshing.

Serves a local page showing the ledger, the graduation gate, TradingView's live
state and the feed cross-check. Leave it open; it polls every few seconds.

Design rule carried over from the rest of this stack: **the page shows failing
criteria as prominently as passing ones**, and never renders an unavailable
source as a zero. A dashboard that makes things look healthy is worse than none.

    python dashboard.py            # http://127.0.0.1:8765
    python dashboard.py --port 9000
"""
from __future__ import annotations

import argparse
import sqlite3
import time
from datetime import datetime, timezone
from typing import Any

from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse

import status as gate
from collector import init_db, stats

app = FastAPI(title="VTL dashboard")
_started = time.time()


def tv_snapshot() -> dict[str, Any]:
    """Live TradingView state, or an explicit unavailable marker."""
    try:
        import tv_cdp
        st = tv_cdp.read_state()
        s = st["studies"][0] if st.get("studies") else {}
        return {
            "ok": True,
            "symbol": st.get("symbol"),
            "resolution": st.get("resolution"),
            "close": st.get("close"),
            "study": s.get("name"),
            "data_size": s.get("dataSize"),
            "values": s.get("values") or {},
        }
    except Exception as exc:
        return {"ok": False, "why": f"{type(exc).__name__}: {str(exc)[:120]}"}


@app.get("/api/state")
def api_state() -> JSONResponse:
    """Everything the page needs, in one call."""
    con = init_db()
    s = stats(con)

    checks, extra = gate.evaluate(con)
    recent = [
        {"symbol": r[0], "entry": r[1], "exit": r[2], "pnl": r[3],
         "ret": r[4], "edge": r[5], "why": r[6]}
        for r in con.execute(
            "SELECT symbol, entry_date, exit_date, pnl, return_pct, edge, exit_reason "
            "FROM trades ORDER BY exit_date DESC LIMIT 12")
    ]
    runs = [
        {"at": r[0], "symbols": r[1], "new": r[2], "total": r[3]}
        for r in con.execute(
            "SELECT run_at, symbols, new_trades, total_trades FROM runs "
            "ORDER BY id DESC LIMIT 5")
    ]
    syncs = [
        {"at": r[0], "status": r[1], "symbol": r[2], "study": r[3]}
        for r in con.execute(
            "SELECT synced_at, status, symbol, study FROM tv_sync ORDER BY id DESC LIMIT 5")
    ]
    con.close()

    tv = tv_snapshot()
    feed = gate.feed_crosscheck()

    return JSONResponse({
        "now": datetime.now(timezone.utc).isoformat(),
        "uptime_s": int(time.time() - _started),
        "ledger": {
            "trades": s["trades"], "pnl": s["total_pnl"],
            "avg_return": s["avg_return"], "avg_bh": s["avg_bh"],
            "avg_edge": s["avg_edge"], "win_rate": s["win_rate"],
            "beat_bh": s["beat_bh_rate"], "disposition": s["disposition_ratio"],
            "winner_hold": s["winner_hold"], "loser_hold": s["loser_hold"],
        },
        "gate": {
            "eligible": all(c.passed for c in checks),
            "passed": sum(1 for c in checks if c.passed),
            "total": len(checks),
            "checks": [{"n": c.n, "name": c.name, "value": c.value,
                        "target": c.target, "passed": c.passed} for c in checks],
        },
        "tradingview": tv,
        "feed": {"status": feed[0], "detail": feed[1]},
        "recent": recent, "runs": runs, "syncs": syncs,
    })


PAGE = """
<!doctype html><html><head><meta charset="utf-8"><title>VTL live</title>
<meta name="viewport" content="width=device-width,initial-scale=1">
<style>
:root{--bg:#0d1117;--fg:#e6edf3;--dim:#8b949e;--line:#30363d;--card:#161b22;
      --ok:#3fb950;--bad:#f85149;--warn:#d29922;--accent:#58a6ff}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--fg);
     font:14px/1.5 ui-monospace,SFMono-Regular,Consolas,monospace}
header{padding:14px 20px;border-bottom:1px solid var(--line);display:flex;
       justify-content:space-between;align-items:center;flex-wrap:wrap;gap:8px}
h1{font-size:15px;margin:0;letter-spacing:.5px}
.pill{padding:3px 10px;border-radius:99px;font-size:12px;border:1px solid var(--line)}
.ok{color:var(--ok);border-color:var(--ok)}.bad{color:var(--bad);border-color:var(--bad)}
.warn{color:var(--warn);border-color:var(--warn)}
main{padding:16px 20px;display:grid;gap:16px;
     grid-template-columns:repeat(auto-fit,minmax(330px,1fr))}
.card{background:var(--card);border:1px solid var(--line);border-radius:8px;padding:14px}
.card h2{font-size:12px;margin:0 0 10px;color:var(--dim);text-transform:uppercase;
         letter-spacing:1px;font-weight:600}
table{width:100%;border-collapse:collapse}
td{padding:3px 0;vertical-align:top}
td:last-child{text-align:right;font-variant-numeric:tabular-nums}
.k{color:var(--dim)}
.big{font-size:26px;font-weight:700;font-variant-numeric:tabular-nums}
.sub{color:var(--dim);font-size:12px}
.row{display:flex;justify-content:space-between;padding:3px 0;gap:10px}
.small{font-size:12px}
footer{padding:10px 20px;color:var(--dim);font-size:11px;border-top:1px solid var(--line)}
.scroll{max-height:230px;overflow:auto}
</style></head><body>
<header>
  <h1>VIBE-TRADING-LOCAL &middot; live</h1>
  <div>
    <span class="pill" id="gatePill">gate …</span>
    <span class="pill" id="tvPill">tradingview …</span>
    <span class="pill" id="feedPill">feed …</span>
  </div>
</header>
<main>
  <div class="card"><h2>Graduation gate</h2><div id="gate"></div></div>
  <div class="card"><h2>Ledger</h2><div id="ledger"></div></div>
  <div class="card"><h2>TradingView (live)</h2><div id="tv"></div></div>
  <div class="card"><h2>Recent trades</h2><div class="scroll small" id="recent"></div></div>
  <div class="card"><h2>Collection runs</h2><div class="small" id="runs"></div></div>
  <div class="card"><h2>TradingView syncs</h2><div class="small" id="syncs"></div></div>
</main>
<footer id="foot">connecting…</footer>
<script>
const pc = v => (v*100).toFixed(1)+'%';
const num = v => (v==null?'—':Number(v).toLocaleString(undefined,{maximumFractionDigits:2}));
function pill(el, txt, cls){ el.textContent = txt; el.className = 'pill '+cls; }

async function tick(){
  let d;
  try { d = await (await fetch('/api/state')).json(); }
  catch(e){ document.getElementById('foot').textContent = 'server unreachable'; return; }

  // gate
  const g = d.gate;
  pill(gatePill, g.eligible?'ELIGIBLE':'NOT ELIGIBLE ('+g.passed+'/'+g.total+')',
       g.eligible?'ok':'bad');
  gate.innerHTML = '<table>' + g.checks.map(c =>
    `<tr><td class="k">${c.passed?'✅':'❌'} ${c.name}</td>`+
    `<td style="color:${c.passed?'var(--ok)':'var(--bad)'}">${c.value}`+
    `<span class="sub"> / ${c.target}</span></td></tr>`).join('') + '</table>';

  // ledger
  const L = d.ledger;
  ledger.innerHTML =
    `<div class="big" style="color:${L.avg_edge>0?'var(--ok)':'var(--bad)'}">`+
    `${(L.avg_edge*100).toFixed(2)}%</div>`+
    `<div class="sub">avg edge per trade (vs buy &amp; hold over the same window)</div>`+
    `<table style="margin-top:10px">`+
    `<tr><td class="k">trades</td><td>${L.trades}</td></tr>`+
    `<tr><td class="k">cumulative P&amp;L</td><td>${num(L.pnl)}</td></tr>`+
    `<tr><td class="k">win rate</td><td>${pc(L.win_rate)}</td></tr>`+
    `<tr><td class="k">beat buy &amp; hold</td><td>${pc(L.beat_bh)}</td></tr>`+
    `<tr><td class="k">disposition ratio</td><td>${L.disposition.toFixed(2)}</td></tr>`+
    `<tr><td class="k">hold: win / loss</td><td>${L.winner_hold.toFixed(1)}d / ${L.loser_hold.toFixed(1)}d</td></tr>`+
    `</table>`;

  // tradingview
  const t = d.tradingview;
  if(t.ok){
    pill(tvPill, 'TV '+(t.symbol||'?')+' '+num(t.close), 'ok');
    tv.innerHTML = `<div class="big">${num(t.close)}</div>`+
      `<div class="sub">${t.symbol} · ${t.resolution} · study “${t.study}” · ${t.data_size} bars</div>`+
      '<table style="margin-top:10px">'+Object.entries(t.values).map(([k,v])=>
        `<tr><td class="k">${k}</td><td>${v}</td></tr>`).join('')+'</table>';
  } else {
    pill(tvPill, 'TV unavailable', 'warn');
    tv.innerHTML = `<div class="sub">Not reachable — recorded as an absence, not a zero.</div>`+
      `<div class="sub" style="margin-top:6px">${t.why}</div>`;
  }

  // feed cross-check
  const f = d.feed;
  pill(feedPill, 'feed '+f.status,
       f.status==='AGREE'?'ok':(f.status==='DIVERGENT'?'bad':'warn'));

  recent.innerHTML = '<table>'+d.recent.map(r=>
    `<tr><td class="k">${r.symbol} <span class="sub">${r.exit}</span></td>`+
    `<td style="color:${r.pnl>=0?'var(--ok)':'var(--bad)'}">${num(r.pnl)}`+
    `<span class="sub"> ${(r.ret*100).toFixed(1)}%</span></td></tr>`).join('')+'</table>';

  runs.innerHTML = d.runs.map(r=>
    `<div class="row"><span class="k">${r.at.slice(0,19).replace('T',' ')}</span>`+
    `<span>+${r.new} → ${r.total}</span></div>`).join('') || '<span class="sub">none</span>';

  syncs.innerHTML = d.syncs.map(s=>
    `<div class="row"><span class="k">${s.at.slice(0,19).replace('T',' ')}</span>`+
    `<span style="color:${s.status==='SYNCED'?'var(--ok)':'var(--warn)'}">${s.status}`+
    `${s.symbol?' '+s.symbol:''}</span></div>`).join('') || '<span class="sub">none</span>';

  foot.textContent = 'updated '+new Date().toLocaleTimeString()+
    ' · paper only, no broker connectivity · refreshes every 5s';
}
tick(); setInterval(tick, 5000);
</script></body></html>
"""


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    """The dashboard page."""
    return PAGE


def main() -> int:
    """Serve the dashboard on loopback."""
    ap = argparse.ArgumentParser(description="VTL live dashboard")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--host", default="127.0.0.1")
    args = ap.parse_args()

    import uvicorn
    print(f"dashboard: http://{args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
