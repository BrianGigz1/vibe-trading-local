"""Breakeven-then-trail stop management.

Two-phase protective stop:

  Phase 1 — initial stop, fixed, set at entry.
  Phase 2 — once price has moved ``be_trigger_dist`` ATR in favour, the stop
            jumps to entry ± ``be_offset`` (the offset covers fees and slippage,
            so "breakeven" is actually breakeven and not a small loss).
  Phase 3 — thereafter the stop trails at ``atr_mult`` ATR behind the best price
            seen, ratcheting one way only.

Design choices that matter, each fixing a defect seen in the webhook-bot version:

* **Pure function.** :func:`update_stop` performs no I/O, no database access and
  no network calls. It is deterministic and unit-testable, which a risk control
  has to be.
* **Ratchet is an invariant, not a convention.** The function asserts the stop
  never loosens, so a logic error raises instead of silently widening risk.
* **State carries ``current_sl`` and ``best_price``.** The stop is never
  re-derived from the current price. Re-deriving on restart is what silently
  re-anchors a stop to a worse level after an adverse move.

.. warning::
   This computes a *synthetic* stop level. It is only enforced when something
   evaluates it. For any live venue the level must also be mirrored as a resting
   stop order on the exchange, or a crash, outage or gap leaves the position
   unprotected.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Literal

Side = Literal["long", "short"]


class StopPhase(str, Enum):
    """Which phase the protective stop is in."""

    INITIAL = "initial"
    BREAKEVEN = "breakeven"
    TRAILING = "trailing"


@dataclass
class Position:
    """Open position and its stop-management state.

    Attributes:
        symbol: Instrument.
        side: ``long`` or ``short``.
        entry_price: Fill price.
        current_sl: Live protective stop level.
        best_price: Most favourable price seen since entry (high for a long,
            low for a short). Persisted so a restart cannot re-anchor the stop.
        is_breakeven_active: True once the stop has reached breakeven.
        be_trigger_dist: Favourable move, in ATR, that activates breakeven.
        be_offset: Absolute buffer past entry when moving to breakeven.
        atr_mult: Trailing distance in ATR.
    """

    symbol: str
    side: Side
    entry_price: float
    current_sl: float
    best_price: float = field(default=0.0)
    is_breakeven_active: bool = False
    be_trigger_dist: float = 1.0
    be_offset: float = 5.0
    atr_mult: float = 2.0

    def __post_init__(self) -> None:
        """Validate the position and seed ``best_price`` from entry."""
        if self.side not in ("long", "short"):
            raise ValueError(f"side must be 'long' or 'short', got {self.side!r}")
        if self.entry_price <= 0:
            raise ValueError("entry_price must be positive")
        if self.be_trigger_dist <= 0 or self.atr_mult <= 0:
            raise ValueError("be_trigger_dist and atr_mult must be positive")
        if self.be_offset < 0:
            raise ValueError("be_offset must be non-negative")
        if self.best_price <= 0:
            self.best_price = self.entry_price
        if self.side == "long" and self.current_sl >= self.entry_price:
            raise ValueError("long stop must be below entry")
        if self.side == "short" and self.current_sl <= self.entry_price:
            raise ValueError("short stop must be above entry")


@dataclass
class StopUpdate:
    """Result of one stop evaluation.

    Attributes:
        stop: Stop level after the update.
        phase: Phase the stop is now in.
        moved: True if the stop level changed this evaluation.
        breached: True if *price* is at or beyond the stop — exit now.
        note: Human-readable explanation.
    """

    stop: float
    phase: StopPhase
    moved: bool
    breached: bool
    note: str


@dataclass
class StopSyncPlan:
    """Ordered steps to move a resting stop order on a venue.

    Attributes:
        should_sync: False when the move is too small to be worth the churn.
        steps: Actions in the order they must be performed.
        new_stop: Level the new order should rest at.
        min_move_required: Threshold the move had to clear.
        note: Explanation.
    """

    should_sync: bool
    steps: list[str]
    new_stop: float
    min_move_required: float
    note: str


def plan_stop_sync(
    pos: Position,
    new_stop: float,
    atr: float,
    *,
    has_existing_order: bool,
    min_move_atr: float = 0.10,
) -> StopSyncPlan:
    """Plan a stop-order replacement safely.

    Two rules, both learned the hard way:

    **Place before cancel.** Cancelling first leaves a window in which the
    position has no protection at all. If the replacement then fails — rate
    limit, network, venue rejecting a stop too close to mark — the position is
    naked and the caller cannot distinguish that from a no-op. Placing first
    means a failure leaves the *old* stop intact, which is the safe direction.

    **Do not churn.** A stop that ratchets on every evaluation issues a
    cancel/replace pair every time. Over a day that is thousands of order
    operations, which invites the rate limit that then strands you mid-swap.
    Only sync when the level moves by at least *min_move_atr* ATR.

    Args:
        pos: Position being managed.
        new_stop: Proposed new stop level.
        atr: Current ATR, for scaling the minimum move.
        has_existing_order: Whether a resting stop order is already on the venue.
        min_move_atr: Minimum move, in ATR, worth replacing an order for.

    Returns:
        A :class:`StopSyncPlan`.

    Raises:
        ValueError: If *atr* is not positive, or *new_stop* is the wrong side
            of the entry-relative direction (a stop that loosens).
    """
    if atr <= 0:
        raise ValueError("atr must be positive")

    is_long = pos.side == "long"
    if is_long and new_stop < pos.current_sl:
        raise ValueError("refusing to loosen a long stop")
    if not is_long and new_stop > pos.current_sl:
        raise ValueError("refusing to loosen a short stop")

    threshold = min_move_atr * atr
    move = abs(new_stop - pos.current_sl)

    if has_existing_order and move < threshold:
        return StopSyncPlan(
            should_sync=False, steps=[], new_stop=pos.current_sl,
            min_move_required=threshold,
            note=f"move {move:.4f} < {threshold:.4f} ({min_move_atr:g}xATR); holding order",
        )

    steps = [
        f"place reduceOnly stop {'sell' if is_long else 'buy'} "
        f"{pos.symbol} @ {new_stop:.6f}",
        "verify new order is open (fetch_order)",
    ]
    if has_existing_order:
        steps.append("ONLY THEN cancel the previous stop order")
        steps.append("if placement failed: keep the OLD order, alert, do not cancel")
    else:
        steps.append("if placement failed: FLATTEN or halt — position is unprotected")

    return StopSyncPlan(
        should_sync=True, steps=steps, new_stop=new_stop,
        min_move_required=threshold,
        note=f"move {move:.4f} >= {threshold:.4f}; replace order",
    )


def update_stop(pos: Position, price: float, atr: float) -> StopUpdate:
    """Advance the protective stop for *pos* and report whether it is breached.

    Mutates ``pos.current_sl``, ``pos.best_price`` and ``pos.is_breakeven_active``
    in place, then returns the outcome.

    Args:
        pos: Position to manage.
        price: Current mark price.
        atr: Current ATR, in price units.

    Returns:
        A :class:`StopUpdate`.

    Raises:
        ValueError: If *price* or *atr* is not positive.
        AssertionError: If the ratchet invariant would be violated.
    """
    if price <= 0:
        raise ValueError("price must be positive")
    if atr <= 0:
        raise ValueError("atr must be positive; refusing to size a stop off a zero range")

    is_long = pos.side == "long"
    prev_stop = pos.current_sl

    # Track the best excursion. This is what the trail hangs off, not the
    # current price, so a pullback cannot drag the stop backwards.
    if is_long:
        pos.best_price = max(pos.best_price, price)
    else:
        pos.best_price = min(pos.best_price, price)

    # Breach is checked FIRST and unconditionally. In the version this replaces,
    # the ratchet and the breach test were if/elif, so a bar that made a new
    # extreme skipped the breach check entirely.
    breached = price <= pos.current_sl if is_long else price >= pos.current_sl

    favourable = (pos.best_price - pos.entry_price) if is_long else (
        pos.entry_price - pos.best_price
    )
    trigger = pos.be_trigger_dist * atr

    phase = StopPhase.INITIAL
    note = "initial stop"
    candidate = pos.current_sl

    if favourable >= trigger:
        if not pos.is_breakeven_active:
            candidate = (
                pos.entry_price + pos.be_offset if is_long
                else pos.entry_price - pos.be_offset
            )
            pos.is_breakeven_active = True
            phase = StopPhase.BREAKEVEN
            note = (
                f"breakeven armed: moved {favourable:.4f} >= {trigger:.4f} "
                f"({pos.be_trigger_dist:g}xATR); stop to entry"
                f"{'+' if is_long else '-'}{pos.be_offset:g}"
            )

        # Evaluate the trail on the SAME tick that arms breakeven. If price has
        # already run well past the trigger, holding the stop at entry for one
        # more evaluation gives back profit for no reason.
        trail = (
            pos.best_price - pos.atr_mult * atr if is_long
            else pos.best_price + pos.atr_mult * atr
        )
        better = trail > candidate if is_long else trail < candidate
        if better:
            candidate = trail
            phase = StopPhase.TRAILING
            note = f"trailing {pos.atr_mult:g}xATR from best {pos.best_price:.4f}"
    else:
        needed = trigger - favourable
        note = f"initial stop; {needed:.4f} more favourable move to arm breakeven"

    # Ratchet only. Never loosen, in either direction.
    new_stop = max(prev_stop, candidate) if is_long else min(prev_stop, candidate)

    if is_long:
        assert new_stop >= prev_stop, "ratchet violated: long stop moved down"
    else:
        assert new_stop <= prev_stop, "ratchet violated: short stop moved up"

    pos.current_sl = new_stop
    moved = new_stop != prev_stop
    if not moved and phase is not StopPhase.INITIAL:
        note += " (no improvement; stop held)"

    if breached:
        note = f"STOP BREACHED at {price:.4f} (stop {prev_stop:.4f}) — exit now"

    return StopUpdate(stop=new_stop, phase=phase, moved=moved, breached=breached, note=note)
