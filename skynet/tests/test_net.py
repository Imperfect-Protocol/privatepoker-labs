"""Tests for skynet.net. Run: python tests/test_net.py"""
from __future__ import annotations

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from skynet.net.drone import NetworkSim, sigmoid_stun_delay
from skynet.net.radio import RadioLink
from skynet.net.signal import Signal, BEACON, ANNOUNCE, FEEDBACK, address_for_node
from skynet.net.simulation import Message


def chain_sim(*, spacing: float = 100.0, count: int = 4, radio_range: float = 150.0):
    """A line of drones: 0 - 1 - 2 - 3, only adjacent pairs in range."""
    sim = NetworkSim(radio=RadioLink(radio_range=radio_range))
    for i in range(count):
        sim.add_drone(i, (i * spacing, 0.0))
    return sim


def test_address_identity():
    address = address_for_node(7)
    assert len(address) == 20
    assert address == address_for_node(7)
    assert address != address_for_node(8)


def test_signal_identity_and_dedup():
    signal = Signal(kind=BEACON, origin_id=1, origin_address=address_for_node(1),
                    sequence_id=5, ttl=8)
    assert len(signal.signal_id) == 16
    # Same semantic announce retries share a dedup key.
    first = Signal(kind=ANNOUNCE, origin_id=1, origin_address=address_for_node(1),
                   sequence_id=9, target_id=3, target_address=address_for_node(3))
    retry = Signal(kind=ANNOUNCE, origin_id=1, origin_address=address_for_node(1),
                   sequence_id=9, target_id=3, target_address=address_for_node(3))
    assert first.dedup_key == retry.dedup_key
    assert first.signal_id == retry.signal_id
    # Forwarding changes hops but not identity.
    forwarded = first.forwarded()
    assert forwarded.hop_count == 1 and forwarded.ttl == 7
    assert forwarded.signal_id == first.signal_id


def test_stun_sigmoid_bounds():
    # Sigmoid between 0.05 and 0.5; midpoint at 20 signals gives the centre.
    assert sigmoid_stun_delay(1) < 0.15          # small cluster: quick burst out
    assert 0.274 < sigmoid_stun_delay(20) < 0.276
    assert sigmoid_stun_delay(1000) > 0.49       # saturation below max
    assert sigmoid_stun_delay(1) < sigmoid_stun_delay(20) < sigmoid_stun_delay(1000)
    try:
        sigmoid_stun_delay(0)
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for empty cluster")


def test_beacon_floods_chain_with_ttl_and_dedup():
    sim = chain_sim(count=5, radio_range=150.0)
    sim.queue.run(0)  # init now=0
    origin = sim.emit(0, BEACON)
    sim.deliver(sim.drones[0], 1, [origin])
    sim.queue.run(max_events=500)
    # Every drone has seen the beacon exactly once (origin included).
    key = origin.dedup_key
    for drone_id in range(5):
        assert key in sim.drones[drone_id].seen, f"drone {drone_id} missed it"
    # No drone processed it twice: each relay accepted at most 1 copy.
    for drone in sim.drones.values():
        assert drone.accepted <= 1
        assert drone.ttl_dropped == 0


def test_ttl_stops_the_flood():
    sim2 = NetworkSim(radio=RadioLink(radio_range=150.0))
    for i in range(8):
        sim2.add_drone(i, (i * 100.0, 0.0), ttl=2)
    sim2.queue.run(0)
    origin = sim2.emit(0, BEACON)
    sim2.deliver(sim2.drones[0], 1, [origin])
    sim2.queue.run(max_events=500)
    key = origin.dedup_key
    seen_by = [i for i in range(8) if key in sim2.drones[i].seen]
    # ttl=2: hops 1 and 2 forward it; hop 3 sees it only to drop it dead.
    assert seen_by == [0, 1, 2, 3]
    assert sim2.drones[3].ttl_dropped == 1
    assert sim2.drones[3].accepted == 0
    # And nobody beyond hop 3 ever heard of it.
    assert all(key not in sim2.drones[i].seen for i in range(4, 8))


def test_announce_reaches_named_target():
    sim = chain_sim(count=4)
    sim.queue.run(0)
    announce = sim.emit(0, ANNOUNCE, target_id=3)
    sim.deliver(sim.drones[0], 1, [announce])
    sim.queue.run(max_events=500)
    arrivals = [(r, s) for r, s in sim.delivered_signals
                if s.signal_id == announce.signal_id]
    # The target receives it exactly once; relays do not "arrive".
    assert arrivals == [(3, arrivals[0][1])]
    assert arrivals[0][1].hop_count == 2


def test_stun_bundles_into_one_packet_per_neighbour():
    sim = NetworkSim(radio=RadioLink(radio_range=150.0))
    sim.add_drone(0, (0.0, 0.0))
    sim.add_drone(1, (50.0, 0.0))
    sim.add_drone(2, (100.0, 0.0))
    sim.queue.run(0)
    relay = sim.drones[1]
    # Two different beacon signals from node 0 hit relay 1 back-to-back.
    first = sim.emit(0, BEACON)
    second = sim.emit(0, BEACON)
    relay.receive(Message(0, 1, (first,), 0.0))
    relay.receive(Message(0, 1, (second,), 0.05))
    # One live stun timer, cluster of 2.
    assert relay.timer.entry_id is not None
    assert relay.cluster_size == 2
    assert len(relay.pending) == 2
    sent = []
    original_deliver = sim.deliver
    def spying_deliver(sender, recipient_id, signals):
        sent.append((recipient_id, len(signals)))
        original_deliver(sender, recipient_id, signals)
    sim.deliver = spying_deliver
    relay.on_stun_timer()
    # One bundled packet to the next hop (never echoed back to the origin).
    assert sent == [(2, 2)]
    assert relay.pending == {}


def test_radio_drop_model():
    radio = RadioLink(radio_range=300.0)
    assert radio.drop_probability(0.0) == 0.0
    assert radio.drop_probability(300.0) == radio.max_drop
    assert radio.drop_probability(301.0) == 1.0
    assert not radio.roll_delivery(301.0, 1.0)
    assert radio.roll_delivery(0.0, 0.999)
    assert radio.roll_delivery(300.0, 0.6)     # sample above 0.5
    assert not radio.roll_delivery(300.0, 0.4)
    # Monotonic in distance.
    drops = [radio.drop_probability(d) for d in (0, 50, 100, 200, 300)]
    assert drops == sorted(drops)


def test_out_of_range_is_silence():
    sim = NetworkSim(radio=RadioLink(radio_range=100.0))
    sim.add_drone(0, (0.0, 0.0))
    sim.add_drone(1, (500.0, 0.0))
    sim.queue.run(0)
    beacon = sim.emit(0, BEACON)
    # "Deliver" from 0 to 1 would never be scheduled via neighbours_of, but
    # direct delivery still respects nothing — the topology does. neighbours:
    assert 1 not in sim.neighbours_of(sim.drones[0])
    assert sim.neighbours_of(sim.drones[0]) == []


def main():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"PASS {t.__name__}")
    print(f"\n{len(tests)} tests passed.")


if __name__ == "__main__":
    main()
