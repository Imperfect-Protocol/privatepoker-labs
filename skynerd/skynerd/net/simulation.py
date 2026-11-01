"""Event loop and bundled messages: the transport skeleton.

A Message is a transport envelope carrying multiple Signals, delivered at a
simulated time. The event queue is a lazy-deletion priority heap; timers are
cancellable tasks like the original simulator's.

Stun coalescing lives in Drone: handlers never send; they stage signals and
arm the node's single stun timer. On fire, the node picks recipients and
emits at most one bundled Message per neighbour.
"""
from __future__ import annotations

import heapq
import itertools
from dataclasses import dataclass, field
from typing import Callable

from skynerd.net.signal import Signal


@dataclass(frozen=True)
class Message:
    sender_id: int
    recipient_id: int
    signals: tuple            # tuple[Signal, ...]
    sent_at: float

    @property
    def dedup_keys(self) -> tuple:
        return tuple(signal.dedup_key for signal in self.signals)


class EventQueue:
    """Time-ordered, cancellable event queue with lazy deletion."""

    def __init__(self) -> None:
        self._heap: list = []
        self._counter = itertools.count()
        self._cancelled: set = set()
        self.now = 0.0
        self.dispatched = 0

    def schedule(self, delay: float, callback: Callable[[], None],
                 cancellable: bool = False) -> int | None:
        entry_id = next(self._counter) if cancellable else None
        heapq.heappush(self._heap, (self.now + max(0.0, delay), next(self._counter),
                                    entry_id, callback))
        return entry_id

    def cancel(self, entry_id: int) -> None:
        self._cancelled.add(entry_id)

    def next_time(self) -> float | None:
        while self._heap:
            time, _, entry_id, _ = self._heap[0]
            if entry_id is not None and entry_id in self._cancelled:
                heapq.heappop(self._heap)
                continue
            return time
        return None

    def step(self) -> bool:
        """Dispatch the next event. Returns False when the queue is empty."""
        while self._heap:
            time, _, entry_id, callback = heapq.heappop(self._heap)
            if entry_id is not None and entry_id in self._cancelled:
                continue
            self.now = time
            self.dispatched += 1
            callback()
            return True
        return False

    def run(self, max_events: int = 100000) -> int:
        events = 0
        while events < max_events and self.step():
            events += 1
        return events


@dataclass
class StunTimer:
    """One live stun timer per node; re-arming replaces it (cancellable)."""
    queue: EventQueue
    node: "Drone"            # forward reference, set by Drone
    fire_at: float | None = None
    entry_id: int | None = None

    def arm(self, delay: float) -> None:
        if self.entry_id is not None:
            self.queue.cancel(self.entry_id)
        self.fire_at = self.queue.now + delay
        self.entry_id = self.queue.schedule(delay, self.node.on_stun_timer,
                                            cancellable=True)

    def disarm(self) -> None:
        if self.entry_id is not None:
            self.queue.cancel(self.entry_id)
        self.entry_id = None
        self.fire_at = None
