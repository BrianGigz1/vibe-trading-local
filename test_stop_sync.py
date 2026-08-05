"""Tests for the stop-order sync planner."""
from __future__ import annotations

import sys

from position_manager import Position, plan_stop_sync

FAILURES: list[str] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    """Record a pass/fail."""
    print(f"  {'PASS' if cond else 'FAIL'}  {name} {detail if not cond else ''}")
    if not cond:
        FAILURES.append(name)


p = Position(symbol="BTC/USDT:USDT", side="long", entry_price=60000.0,
             current_sl=58500.0, atr_mult=2.0)

print("\nplace-before-cancel ordering")
plan = plan_stop_sync(p, new_stop=59000.0, atr=1000.0, has_existing_order=True)
check("sync required", plan.should_sync)
check("places first", plan.steps[0].startswith("place"), plan.steps[0])
check("verifies second", "verify" in plan.steps[1])
cancel_idx = next(i for i, s in enumerate(plan.steps) if "cancel" in s)
place_idx = next(i for i, s in enumerate(plan.steps) if s.startswith("place"))
check("cancel strictly after place", cancel_idx > place_idx)
check("failure keeps old order", any("keep the OLD order" in s for s in plan.steps))

print("\nno existing order -> failure is an emergency")
plan = plan_stop_sync(p, new_stop=59000.0, atr=1000.0, has_existing_order=False)
check("flatten-or-halt on failure", any("FLATTEN or halt" in s for s in plan.steps))

print("\nchurn threshold")
plan = plan_stop_sync(p, new_stop=58550.0, atr=1000.0, has_existing_order=True)
check("tiny move skipped", not plan.should_sync, plan.note)
plan = plan_stop_sync(p, new_stop=58700.0, atr=1000.0, has_existing_order=True)
check("move >= 0.1 ATR syncs", plan.should_sync, plan.note)

print("\nloosening refused")
try:
    plan_stop_sync(p, new_stop=58000.0, atr=1000.0, has_existing_order=True)
    check("long stop loosening rejected", False, "no exception")
except ValueError:
    check("long stop loosening rejected", True)

s = Position(symbol="ETH/USDT:USDT", side="short", entry_price=3000.0, current_sl=3100.0)
try:
    plan_stop_sync(s, new_stop=3200.0, atr=100.0, has_existing_order=True)
    check("short stop loosening rejected", False, "no exception")
except ValueError:
    check("short stop loosening rejected", True)

print("\nzero atr refused")
try:
    plan_stop_sync(p, new_stop=59000.0, atr=0.0, has_existing_order=True)
    check("zero atr rejected", False, "no exception")
except ValueError:
    check("zero atr rejected", True)

print()
if FAILURES:
    print(f"{len(FAILURES)} FAILED: {', '.join(FAILURES)}")
    sys.exit(1)
print("all checks passed")
