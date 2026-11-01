"""Drone: a network node with stun coalescing, dedup, and TTL forwarding.

Ported semantics from p2p_nerual_spatial, slimmed:

- Handlers never send on receipt. Accepted signals stage in ``pending``
  and arm the node's single stun timer (sigmoid delay by cluster size).
- On timer fire: pick neighbours still in radio range, emit at most one
  bundled Message per neighbour, clear pending, drop duplicates/TTL-dead.
- Dedup: a signal whose dedup key was seen before is dropped, unless this
  node is its origin or its named target.
"""
from __future__ import annotations

import math

from skynerd.net.radio import RadioLink
from skynerd.net.signal import Signal, BEACON, ANNOUNCE, FEEDBACK
from skynerd.net.simulation import EventQueue, Message, StunTimer


def sigmoid_stun_delay(message_count: int, k: float = 0.1, midpoint: float = 20.0,
                       minimum: float = 0.05, maximum: float = 0.5) -> float:
    if message_count < 1:
        raise ValueError("message_count must be >= 1")
    exponent = max(-60.0, min(60.0, -k * (message_count - midpoint)))
    return minimum + (maximum - minimum) / (1.0 + math.exp(exponent))


class Drone:
    def __init__(self, node_id: int, sim: "NetworkSim",
                 *, ttl: int = 8, radio: RadioLink | None = None) -> None:
        self.id = node_id
        self.sim = sim
        self.address = sim.address_of(node_id)
        self.ttl = ttl
        self.radio = radio or sim.radio
        self.pending: dict = {}          # dedup_key -> Signal staged for forwarding
        self.seen: set = set()           # dedup keys already handled
        self.timer = StunTimer(sim.queue, self)
        self.cluster_size = 0
        # metrics
        self.messages_sent = 0
        self.signals_forwarded = 0
        self.duplicates_dropped = 0
        self.ttl_dropped = 0
        self.accepted = 0

    # -- receiving ----------------------------------------------------------

    def receive(self, message: Message) -> None:
        """Handle an inbound bundled message. Stage, never send."""
        changed = False
        for signal in message.signals:
            key = signal.dedup_key
            if key in self.seen or key in self.pending:
                self.duplicates_dropped += 1
                continue
            is_origin = signal.origin_id == self.id
            is_target = signal.kind == ANNOUNCE and signal.target_id == self.id
            if is_origin or is_target:
                self.seen.add(key)
                self.sim.on_signal_arrived(self, signal)
                continue
            if signal.ttl <= 0:
                self.ttl_dropped += 1
                self.seen.add(key)
                continue
            self.seen.add(key)
            self.pending[key] = signal.forwarded()
            self.accepted += 1
            changed = True
        if changed:
            self.cluster_size = len(self.pending)
            self.timer.arm(sigmoid_stun_delay(self.cluster_size))

    def on_stun_timer(self) -> None:
        """Fire: bundle and send, at most one Message per neighbour."""
        self.timer.entry_id = None
        if not self.pending:
            return
        signals = list(self.pending.values())
        self.pending.clear()
        self.cluster_size = 0
        for neighbour_id in self.sim.neighbours_of(self):
            bundle = []
            for signal in signals:
                if self.sim.wants_signal(neighbour_id, signal):
                    bundle.append(signal)
            if bundle:
                self.sim.deliver(self, neighbour_id, bundle)
                self.messages_sent += 1
                self.signals_forwarded += len(bundle)


class NetworkSim:
    """Owns positions, the event queue, and radio delivery between drones."""

    def __init__(self, *, radio: RadioLink | None = None) -> None:
        self.queue = EventQueue()
        self.radio = radio or RadioLink()
        self.drones: dict[int, Drone] = {}
        self.positions: dict[int, tuple] = {}
        self.next_sequence = 0
        self.delivered_signals: list = []      # (recipient_id, Signal) arrivals

    def address_of(self, node_id: int) -> bytes:
        from skynerd.net.signal import address_for_node
        return address_for_node(node_id)

    def add_drone(self, node_id: int, position: tuple, *, ttl: int = 8) -> Drone:
        drone = Drone(node_id, self, ttl=ttl, radio=self.radio)
        self.drones[node_id] = drone
        self.positions[node_id] = position
        return drone

    def neighbours_of(self, drone: Drone) -> list:
        px, py = self.positions[drone.id]
        result = []
        for other_id, (x, y) in self.positions.items():
            if other_id == drone.id:
                continue
            if drone.radio.in_range(math.hypot(px - x, py - y)):
                result.append(other_id)
        return sorted(result)

    def wants_signal(self, neighbour_id: int, signal: Signal) -> bool:
        """A neighbour wants a signal unless it is its own origin beacon."""
        return not (signal.kind == BEACON and signal.origin_id == neighbour_id)

    def deliver(self, sender: Drone, recipient_id: int, signals: list) -> None:
        message = Message(sender.id, recipient_id, tuple(signals),
                          self.queue.now)
        delay = self.radio.propagation_delay(
            math.dist(self.positions[sender.id], self.positions[recipient_id]))
        self.queue.schedule(delay,
                            lambda m=message: self.drones[m.recipient_id].receive(m))

    def on_signal_arrived(self, drone: Drone, signal: Signal) -> None:
        self.delivered_signals.append((drone.id, signal))

    def emit(self, origin_id: int, kind: str, *, target_id: int | None = None) -> Signal:
        """Origin a fresh signal and stage it directly (no stun for origins)."""
        drone = self.drones[origin_id]
        self.next_sequence += 1
        signal = Signal(kind=kind, origin_id=origin_id,
                        origin_address=drone.address,
                        sequence_id=self.next_sequence,
                        target_id=target_id,
                        target_address=self.address_of(target_id) if target_id is not None else None,
                        ttl=drone.ttl)
        drone.seen.add(signal.dedup_key)
        return signal
