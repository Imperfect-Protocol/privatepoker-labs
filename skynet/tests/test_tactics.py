"""Tests for tanks and tactics. Run: python tests/test_tactics.py"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from skynet.learning.tactics import (APPROACH, RETREAT, STRAFE, TacticsPolicy,
                                     context_from_eye)
from skynet.mission.tank import Tank, TankState
from skynet.vec import Vec3


def test_tank_tracks_and_fires():
    tank = Tank(Vec3(0.0, 0.0, 0.0))
    drone_pos = Vec3(100.0, 0.0, 30.0)
    shots = []
    for _ in range(500):                       # 10 s at 50 Hz
        shot = tank.step(0.02, [(7, drone_pos)])
        if shot is not None:
            shots.append(shot)
    assert tank.state in (TankState.ENGAGING, TankState.FIRING)
    assert tank.shots_fired >= 4               # 10 s / 2 s cooldown
    assert all(shot == drone_pos for shot in shots)
    # Out of range: silence.
    quiet = Tank(Vec3(0.0, 0.0, 0.0))
    assert quiet.step(0.02, [(7, Vec3(500.0, 0.0, 30.0))]) is None
    assert quiet.state == TankState.PATROL


def test_tank_destroyed_stops_firing():
    tank = Tank(Vec3(0.0, 0.0, 0.0))
    drone_pos = Vec3(100.0, 0.0, 30.0)
    assert not tank.take_hit(50.0)
    assert tank.take_hit(50.0)
    assert tank.destroyed
    assert tank.step(0.02, [(7, drone_pos)]) is None


def test_context_from_eye():
    glued = np.zeros((2, 8, 8, 8, 8), dtype=np.float32)
    glued[1, 6, 4, 5, 0] = 10.0                # contact: right side, mid range
    context = context_from_eye(glued)
    assert len(context) == 4
    assert context[0] == 6                     # dominant azimuth bin
    assert context[1] == 4                     # dominant elevation band
    assert context[2] == 5                     # dominant range shell
    assert context[3] >= 1                     # contact mass bucket
    # Empty sky: neutral context, no crash.
    empty = context_from_eye(np.zeros((1, 8, 8, 8, 8), dtype=np.float32))
    assert empty == (4, 4, 4, 0)


def test_tactics_prefers_rewarded_intent():
    rng = np.random.default_rng(3)
    policy = TacticsPolicy()
    context = (6, 4, 5, 2)
    for _ in range(20):
        policy.reinforce(context, STRAFE, 0.5, time_epoch=0)
        policy.reinforce(context, RETREAT, -0.5, time_epoch=0)
    choices = [policy.choose(context, rng, epsilon=0.0) for _ in range(5)]
    assert all(choice == STRAFE for choice in choices)
    # Unvisited contexts fall back to neutral exploration, not a crash.
    assert policy.choose((1, 1, 1, 1), rng, epsilon=0.0) in (APPROACH, STRAFE, RETREAT)


def test_tactics_consolidation():
    donor = TacticsPolicy()
    context = (2, 3, 4, 1)
    for _ in range(10):
        donor.reinforce(context, STRAFE, 1.0, time_epoch=0)
    receiver = TacticsPolicy()
    # First merge moves the receiver's score up from baseline, toward but
    # not reaching the donor's full weight (merge_rate bounded).
    before = receiver.tensor.score(context + (1,), 0, 1.0)
    receiver.consolidate_from(donor, rate=0.5)
    after = receiver.tensor.score(context + (1,), 0, 1.0)
    assert before < after < donor.tensor.score(context + (1,), 0, 1.0)
    # Repeated merges converge to the donor's value, never overshoot it.
    for _ in range(30):
        receiver.consolidate_from(donor, rate=0.5)
    assert abs(receiver.tensor.score(context + (1,), 0, 1.0)
               - donor.tensor.score(context + (1,), 0, 1.0)) < 1e-6


def main():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"PASS {t.__name__}")
    print(f"\n{len(tests)} tests passed.")


if __name__ == "__main__":
    main()
