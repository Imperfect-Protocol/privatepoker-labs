"""The learn-to-fly milestone: a drone with only the ServoPolicy brain must
survive longer as episodes accumulate. Run: python tests/test_learn_to_fly.py"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from skynerd.config import Config
from skynerd.learning.reward import (DenseReward, EpisodeOutcome, FlightInstruments,
                                    classify_outcome)
from skynerd.learning.servo_tensor import ServoPolicy
from skynerd.physics.rigid_body import RigidBody
from skynerd.physics.surfaces import SurfacesState
from skynerd.physics.world import World
from skynerd.vec import Vec3

HORIZON = 20.0
TRIM_SPEED = 10.5
# Airframe trim offsets (hardware calibration, not flying skill): the FC's
# neutral surface/throttle setting that holds near-level flight.
TRIM_COMMAND = np.array([0.0, 0.0, 0.0, 0.15])


def run_episode(policy: ServoPolicy, instruments: FlightInstruments, epoch: int,
                rng: np.random.Generator) -> tuple:
    """One flight. Returns (survival_seconds, outcome)."""
    world = World(half_extent=800.0)
    body = RigidBody(Vec3(0.0, 0.0, 50.0), Vec3(TRIM_SPEED, 0.0, 0.0))
    servos = SurfacesState()
    dense = DenseReward(instruments, scale=5.0)
    dense.reset(body)
    dt = 1.0 / 50.0
    commands = TRIM_COMMAND.copy()
    recent: list = []
    steps = int(HORIZON / dt)
    outcome = None
    decision_every = 10   # decisions at 5 Hz; physics and reward stay at 50 Hz
    pending_reward = 0.0
    for tick in range(steps):
        if tick % decision_every == 0:
            # One credit per decision: the reward accumulated since the last
            # decision credits the decisions that caused it.
            if recent:
                policy.reinforce(recent, pending_reward,
                                 time_epoch=tick + epoch * 100000)
            pending_reward = 0.0
            state_key = instruments.state_key(body)
            actions = policy.choose(state_key, rng)
            commands = policy.command_from(commands, actions)
            recent.append((state_key, actions))
            if len(recent) > 10:
                recent.pop(0)
        servos.command(commands, dt)
        body.step(servos, dt)
        pending_reward += dense.tick(body) + 0.0005   # faint alive bonus
        outcome = classify_outcome(body, world, (tick + 1) * dt)
        if outcome is not None:
            break
    if outcome in (EpisodeOutcome.CRASHED, EpisodeOutcome.STALLED):
        policy.reinforce(recent, -2.0, time_epoch=tick + epoch * 100000)
    elif outcome is None:
        outcome = EpisodeOutcome.ALIVE
        policy.reinforce(recent, 0.5, time_epoch=tick + epoch * 100000)
    return ((tick + 1) * dt, outcome)


def main():
    config = Config()
    rng = config.seeded_rng()
    instruments = FlightInstruments(trim_speed=TRIM_SPEED)
    policy = ServoPolicy()

    episodes = 60
    survival = []
    outcomes = {}
    for episode in range(episodes):
        policy.epsilon = max(0.02, 0.5 - episode * 0.012)
        seconds, outcome = run_episode(policy, instruments, episode, rng)
        survival.append(seconds)
        outcomes[outcome] = outcomes.get(outcome, 0) + 1
        if (episode + 1) % 10 == 0:
            window = survival[episode - 9:episode + 1]
            print(f"episodes {episode-8:3d}-{episode+1:3d}: "
                  f"mean survival {sum(window)/len(window):5.2f}s  "
                  f"(last {seconds:.2f}s, {outcome})")

    early = sum(survival[:5]) / 5.0
    late = sum(survival[-5:]) / 5.0
    best = max(survival)
    print(f"\nearly mean {early:.2f}s -> late mean {late:.2f}s, best {best:.2f}s")
    print("outcomes:", outcomes)

    assert late > early, f"no learning: {early:.2f}s -> {late:.2f}s"
    assert best >= 0.8 * HORIZON, f"never learned to survive: best {best:.2f}s"
    print(f"\nLEARN-TO-FLY MILESTONE PASSED: {early:.2f}s -> {late:.2f}s")


if __name__ == "__main__":
    main()
