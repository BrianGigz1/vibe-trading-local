"""Direct Chrome DevTools Protocol reader for TradingView.

The bundled `tv` CLI picks its target with `targets.find(...)` — the *first*
matching page — with no override. On a multi-window TradingView session that is
frequently the wrong one: observed directly, the CLI reported 0 studies on a
chart that demonstrably had one, on all three tabs.

Talking to CDP directly removes the heuristic. Every target is enumerated and
the study-reading script is evaluated on each until one answers, so the sync
does not depend on guessing which window is in front.

No TradingView code is imported and nothing is written to the chart — this is a
read-only client.
"""
from __future__ import annotations

import json
from typing import Any

import urllib.error
import urllib.request

import websocket   # websocket-client

import os

CDP_HOST = os.environ.get("VT_CDP_HOST", "127.0.0.1")
CDP_PORT = int(os.environ.get("VT_CDP_PORT", "9222"))
TIMEOUT = 15


class CDPError(RuntimeError):
    """CDP is unreachable, or no target could answer."""


def list_targets() -> list[dict]:
    """All debuggable page targets."""
    url = f"http://{CDP_HOST}:{CDP_PORT}/json/list"
    try:
        with urllib.request.urlopen(url, timeout=5) as r:
            targets = json.loads(r.read().decode())
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        raise CDPError(f"TradingView CDP not reachable on {CDP_HOST}:{CDP_PORT} ({exc})")
    return [t for t in targets if t.get("type") == "page"]


def evaluate(ws_url: str, expression: str) -> Any:
    """Run *expression* in a target and return its JSON value.

    ``suppress_origin`` is required: websocket-client sends an ``Origin`` header
    by default, and Chrome's DevTools endpoint answers any upgrade carrying one
    with **403 Forbidden**. Without it every target fails handshake, which looks
    identical to "no study attached".
    """
    ws = websocket.create_connection(ws_url, timeout=TIMEOUT, suppress_origin=True)
    try:
        ws.send(json.dumps({
            "id": 1,
            "method": "Runtime.evaluate",
            "params": {"expression": expression, "returnByValue": True,
                       "awaitPromise": True},
        }))
        # Skip unrelated protocol events until our reply arrives.
        for _ in range(40):
            msg = json.loads(ws.recv())
            if msg.get("id") == 1:
                res = msg.get("result", {}).get("result", {})
                if msg.get("result", {}).get("exceptionDetails"):
                    raise CDPError(f"JS exception: {msg['result']['exceptionDetails']}")
                return res.get("value")
        raise CDPError("no Runtime.evaluate reply")
    finally:
        ws.close()


READ_STUDIES = r"""
(function(){
  try {
    var w = window.TradingViewApi && window.TradingViewApi._activeChartWidgetWV;
    if (!w) return {ok:false, why:'no chart widget'};
    var chart = w.value()._chartWidget;
    var srcs = chart.model().model().dataSources();
    var out = {ok:true, symbol:null, resolution:null, close:null, bar_time:null, studies:[]};
    try { out.symbol = chart.model().mainSeries().symbolInfo().name; } catch(e){}
    try { out.resolution = chart.model().mainSeries().properties().interval.value(); } catch(e){}
    // Last close straight from the price series. Taking it from a Pine
    // scorecard would make the feed cross-check depend on whichever script
    // happens to be attached; the series is always there.
    try {
      var ms = chart.model().mainSeries();
      var d = ms.data ? ms.data() : null;
      var bars = d && d.bars ? d.bars() : null;
      var last = null;
      if (bars && bars.last) { last = bars.last(); }
      else if (bars && bars.size && bars.valueAt) { last = bars.valueAt(bars.size()-1); }
      if (last) {
        var v = last.value || last;
        // Pine/TV bar arrays are [time, open, high, low, close, volume].
        if (v && v.length >= 5) { out.bar_time = v[0]; out.close = v[4]; }
      }
    } catch(e){}
    for (var i=0;i<srcs.length;i++){
      var s = srcs[i];
      try {
        var m = s.metaInfo && s.metaInfo();
        if (!m || !m.pine) continue;
        var rec = {id:s.id(), name:m.description, dataSize:null, tables:[], values:{}};
        try { rec.dataSize = s.data().size(); } catch(e){}
        try {
          var items = s.dataWindowView().items();
          for (var j=0;j<items.length;j++){
            var it = items[j];
            if (it._title && it._value && it._value !== '∅') rec.values[it._title] = it._value;
          }
        } catch(e){}
        try {
          var g = s.graphics && s.graphics();
          if (g) {
            var tbls = g.tables ? g.tables() : null;
            if (tbls && tbls.size) {
              tbls.forEach(function(tb){
                var cells = [];
                try { tb.data.cells.forEach(function(row){
                  cells.push(row.map(function(c){ return c && c.text != null ? String(c.text) : ''; }).join(' | '));
                }); } catch(e){}
                if (cells.length) rec.tables.push(cells);
              });
            }
          }
        } catch(e){}
        out.studies.push(rec);
      } catch(e){}
    }
    return out;
  } catch(e){ return {ok:false, why:String(e).slice(0,200)}; }
})()
"""


def read_state() -> dict:
    """Find a target with a computed Pine study and return its state.

    Raises:
        CDPError: If CDP is down, or no target holds a computed Pine study.
    """
    targets = list_targets()
    if not targets:
        raise CDPError("no debuggable pages; is TradingView running with CDP?")

    best: dict | None = None
    tried = 0
    for t in targets:
        ws = t.get("webSocketDebuggerUrl")
        if not ws:
            continue
        tried += 1
        try:
            res = evaluate(ws, READ_STUDIES)
        except (CDPError, OSError, websocket.WebSocketException):
            continue
        if not isinstance(res, dict) or not res.get("ok"):
            continue
        studies = res.get("studies") or []
        # Prefer a target whose study has actually computed bars.
        computed = [s for s in studies if (s.get("dataSize") or 0) > 0]
        if computed:
            res["studies"] = computed
            res["target_id"] = t.get("id")
            return res
        if studies and best is None:
            res["target_id"] = t.get("id")
            best = res

    if best is not None:
        raise CDPError(
            f"found a Pine study but dataSize is 0 — the server-side compute did "
            f"not return. Reload the TradingView chart. (searched {tried} targets)")
    raise CDPError(f"searched {tried} targets; no Pine study attached to any chart")
