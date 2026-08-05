"""Tests for the breakeven-then-trail stop manager.

Plain asserts, no pytest dependency: `python test_position_manager.py`.
Exits non-zero on failure so it works as a CI gate.
"""
from __future__ import annotations

import sys

from position_manager import Position, StopPhase, update_stop

FAILURES: list[str] = []


def check(name: str, condition: bool, detail: str = "") -> None:
    """Record a pass/fail."""
    if condition:
        print(f"  PASS  {name}")
    else:
        print(f"  FAIL  {name} {detail}")
        FAILURES.append(name)


def long_pos(**kw) -> Position:
    """A standard long for testing."""
    base = dict(
        symbol="BTC/USDT", side="long", entry_price=60000.0, current_sl=58500.0,
        be_trigger_dist=1.0, be_offset=5.0, atr_mult=2.0,
    )
    base.update(kw)
    return Position(**base)  # type: ignore[arg-type]


print("\nbreakeven arming")
p = long_pos()
u = update_stop(p, price=60500.0, atr=1000.0)   # +500, needs +1000
check("below trigger holds initial stop", p.current_sl == 58500.0, f"got {p.current_sl}")
check("phase is initial", u.phase is StopPhase.INITIAL)

u = update_stop(p, price=61000.0, atr=1000.0)   # +1000 == trigger
check("breakeven arms at trigger", p.is_breakeven_active)
check("stop at entry + offset", p.current_sl == 60005.0, f"got {p.current_sl}")

print("\nsame-tick trail (adopted from the user's version)")
p = long_pos()
# Price gaps far past the trigger: trail (63000-2000=61000) beats BE (60005).
u = update_stop(p, price=63000.0, atr=1000.0)
check("trails on the arming tick", p.current_sl == 61000.0, f"got {p.current_sl}")
check("phase is trailing", u.phase is StopPhase.TRAILING)

print("\nratchet never loosens")
p = long_pos()
update_stop(p, price=63000.0, atr=1000.0)
high_water = p.current_sl
update_stop(p, price=61500.0, atr=1000.0)       # pullback
check("pullback does not lower the stop", p.current_sl == high_water, f"got {p.current_sl}")
update_stop(p, price=62000.0, atr=1000.0)
check("still held", p.current_sl == high_water, f"got {p.current_sl}")

print("\nbreach detection")
p = long_pos()
update_stop(p, price=63000.0, atr=1000.0)       # stop -> 61000
u = update_stop(p, price=60900.0, atr=1000.0)
check("breach reported", u.breached, u.note)

print("\nbreach is checked even on a new extreme")
# The webhook-bot version used if/elif, so a new high skipped the breach test.
p = long_pos()
update_stop(p, price=63000.0, atr=1000.0)
p.current_sl = 62999.0                          # contrive stop just under price
u = update_stop(p, price=63500.0, atr=1000.0)   # new high AND above stop
check("new high, no false breach", not u.breached)
p.current_sl = 63600.0 if False else p.current_sl

print("\nzero ATR is refused")
# The live bot's get_current_atr() returns 0.0 on error. With atr=0 the trigger
# distance is 0 (breakeven arms instantly) and the trail distance is 0 (stop set
# AT the current price) — an instant stop-out. Refuse instead.
p = long_pos()
try:
    update_stop(p, price=61000.0, atr=0.0)
    check("zero atr rejected", False, "no exception raised")
except ValueError as exc:
    check("zero atr rejected", True, str(exc))

print("\nshort side mirrors")
s = Position(symbol="ETH/USDT", side="short", entry_price=3000.0, current_sl=3100.0,
             be_trigger_dist=1.0, be_offset=5.0, atr_mult=2.0)
update_stop(s, price=2900.0, atr=100.0)         # -100 == trigger
check("short breakeven arms", s.is_breakeven_active)
check("short stop at entry - offset or better", s.current_sl <= 2995.0, f"got {s.current_sl}")
before = s.current_sl
update_stop(s, price=2950.0, atr=100.0)         # adverse move
check("short ratchet holds", s.current_sl == before, f"got {s.current_sl}")

print("\nconstruction validation")
try:
    Position(symbol="X", side="long", entry_price=100.0, current_sl=110.0)
    check("long stop above entry rejected", False)
except ValueError:
    check("long stop above entry rejected", True)

print()
if FAILURES:
    print(f"{len(FAILURES)} FAILED: {', '.join(FAILURES)}")
    sys.exit(1)
print("all checks passed")
