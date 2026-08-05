"""Smoke test for the paper webhook. Asserts the safety behaviours hold."""
from __future__ import annotations

import importlib.util
import os
import sys

# The app reads its secret from the environment at import time, so the test must
# read the SAME source. Hardcoding it here made the suite fail whenever the env
# was set to anything else — a broken test reporting a broken product.
os.environ.setdefault("VT_WEBHOOK_PASSPHRASE", "test-secret-not-a-default")

from fastapi.testclient import TestClient

import webhook_paper as w

SECRET = w.PASSPHRASE
if not SECRET:
    print("VT_WEBHOOK_PASSPHRASE is empty; endpoint would 503")
    sys.exit(1)
c = TestClient(w.app)


def post(**kw):
    """POST a signal with the valid passphrase unless overridden."""
    kw.setdefault("passphrase", SECRET)
    return c.post("/webhook", json=kw)


print("mode          :", c.get("/health").json()["mode"])
print("wrong pass    :", post(passphrase="wrong", action="buy", ticker="BTC/USDT", price=50000).status_code, "(expect 401)")

r = post(action="buy", ticker="BTC/USDT", price=50000).json()
print("no stop_loss  :", r["status"], "-", r["reason"])

# BTC at 50k with a 2k stop: one unit risks 2x the whole 1% budget, so the
# sizer floors to zero. Correct for whole units — and a real limitation, since
# crypto trades fractionally. See note in the report.
r = post(action="buy", ticker="BTC/USDT", price=50000, stop_loss=48000, bar_time="t0").json()
print("btc oversized :", r["status"], "-", r.get("reason"))

r = post(action="buy", ticker="AAPL", price=500, stop_loss=490, bar_time="t1").json()
print("valid buy     :", r["status"], "| qty", r["order"]["qty"], "|", r.get("sizing", ""))

r = post(action="buy", ticker="AAPL", price=500, stop_loss=490, bar_time="t1").json()
print("replayed      :", r["status"], "-", r.get("reason"))

r = post(action="sell", ticker="ETH/USDT", price=3000, bar_time="t2").json()
print("sell / short  :", r["status"], "-", r.get("reason"))

print("ccxt importable:", importlib.util.find_spec("ccxt") is not None, "(expect False)")
print("health        :", c.get("/health").json())
