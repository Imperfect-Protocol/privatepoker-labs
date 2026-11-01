"""The organism milestone: drone B watches drone A die, then avoids A's fatal
state in its own next episode. Run: python tests/test_mirror_learning.py"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from skynet.config import Config
from skynet.learning.mirror import MirrorLearner, obituary_from
from skynet.learning.reward import (DenseReward, EpisodeOutcome, FlightInstruments,
                                    classify_outcome)
from skynet.learning.servo_tensor import ServoPolicy
from skynet.physics.rigid_body import RigidBody
from skynet.physics.surfaces import SurfacesState
from skynet.physics.world import World
from skynet.vec import Vec3
from tests.test_learn_to_fly import TRIM_COMMAND, TRIM_SPEED, HORIZON

LOW_SLOW_KEY = (0, 4, 0)     # sinking hard, level, far too slow: the fatal stall


def force_stall_state(body: RigidBody) -> None:
    """Set the body into A's fatal state: sinking fast, wings level, slow."""
    body.velocity = Vec3(5.0, 0.0, -3.5)


def fly_episode(policy: ServoPolicy, instruments: FlightInstruments, epoch: int,
                rng: np.random.Generator, *, start_stalled: bool = False,
                epsilon_override: float | None = None):
    world = World(half_extent=800.0)
    body = RigidBody(Vec3(0.0, 0.0, 50.0), Vec3(TRIM_SPEED, 0.0, 0.0))
    if start_stalled:
        force_stall_state(body)
    servos = SurfacesState()
    dense = DenseReward(instruments, scale=5.0)
    dense.reset(body)
    dt = 1.0 / 50.0
    commands = TRIM_COMMAND.copy()
    recent: list = []
    pending_reward = 0.0
    decision_every = 10
    steps = int(HORIZON / dt)
    outcome = None
    tick = 0
    if epsilon_override is not None:
        policy.epsilon = epsilon_override
    for tick in range(steps):
        if tick % decision_every == 0:
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
        pending_reward += dense.tick(body) + 0.0005
        outcome = classify_outcome(body, world, (tick + 1) * dt)
        if outcome is not None:
            break
    if outcome in (EpisodeOutcome.CRASHED, EpisodeOutcome.STALLED):
        policy.reinforce(recent, -2.0, time_epoch=tick + epoch * 100000)
        return (tick + 1) * dt, outcome, instruments.state_key(body), commands
    if outcome is None:
        outcome = EpisodeOutcome.ALIVE
        policy.reinforce(recent, 0.5, time_epoch=tick + epoch * 100000)
    return (tick + 1) * dt, outcome, instruments.state_key(body), commands


def main():
    config = Config()
    rng = config.seeded_rng()
    instruments = FlightInstruments(trim_speed=TRIM_SPEED)
    policy_a = ServoPolicy()
    policy_b = ServoPolicy()

    # --- Phase 1: A learns to fly (short training), then dies in the fatal state.
    for episode in range(30):
        policy_a.epsilon = max(0.02, 0.5 - episode * 0.012)
        fly_episode(policy_a, instruments, episode, rng)

    # A enters the fatal low-slow state and stalls there.
    seconds_a, outcome_a, final_key_a, final_cmd_a = fly_episode(
        policy_a, instruments, 30, rng, start_stalled=True)
    print(f"A: flew {seconds_a:.2f}s in the fatal state, outcome={outcome_a}, "
          f"final key={final_key_a}")
    assert outcome_a in (EpisodeOutcome.CRASHED, EpisodeOutcome.STALLED)

    obituary = obituary_from(b"A" * 20, final_key_a, final_cmd_a)

    # --- Phase 2: B witnesses, untrained, then enters the SAME fatal state.
    mirror_b = MirrorLearner(policy_b, hop_decay=0.5, state_overlap_boost=2.0)
    # B is in the fatal state when the obituary arrives (state overlap max).
    touched = mirror_b.witness(LOW_SLOW_KEY, obituary, hops=0, time_epoch=0)
    assert touched > 0

    # B's solo episode from the fatal state. Greedy enough to show the weights.
    policy_b.epsilon = 0.2
    seconds_b_solo, outcome_b, key_b, _ = fly_episode(
        policy_b, instruments, 1, rng, start_stalled=True)
    print(f"B (after witnessing): {seconds_b_solo:.2f}s, outcome={outcome_b}")

    # --- Control: an identical drone C that never heard the obituary.
    policy_c = ServoPolicy()
    policy_c.epsilon = 0.2
    seconds_c, outcome_c, key_c, _ = fly_episode(
        policy_c, instruments, 1, rng, start_stalled=True)
    print(f"C (no obituary, control): {seconds_c:.2f}s, outcome={outcome_c}")

    # --- The milestone claim: B survives meaningfully longer than C.
    print(f"\nmirror effect: B {seconds_b_solo:.2f}s vs C {seconds_c:.2f}s "
          f"({seconds_b_solo - seconds_c:+.2f}s)")
    assert seconds_b_solo > seconds_c + 1.0, (
        f"witness gained no survival edge: {seconds_b_solo:.2f} vs {seconds_c:.2f}")
    print("MIRROR-LEARNING MILESTONE PASSED: B learned from A's funeral.")


if __name__ == "__main__":
    main()
