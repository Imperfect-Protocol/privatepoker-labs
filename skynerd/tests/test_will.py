"""Singularity milestone: an unlabeled will that grows its own goals.
Run: python tests/test_will.py"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from skynerd.learning.will import ACTIVATION_FLOOR, DESIRE_AXIS, WillTensor, goal_reward
from skynerd.physics.rigid_body import RigidBody
from skynerd.vec import Vec3
from skynerd.learning.reward import FlightInstruments
from skynerd.learning.flight import TRIM_SPEED

CONTEXT = (6, 4, 5, 2)


def test_goal_reward_shapes():
    instruments = FlightInstruments(trim_speed=10.5)
    body = RigidBody(Vec3(0, 0, 50), Vec3(10.5, 0, 0))   # level, at trim
    goal = (0.7, 0.0, 0.0)                               # some learned target
    # Diving away from the goal punishes.
    body.velocity = Vec3(10.5, 0, -2.0)
    _, progress_bad = goal_reward(instruments.errors(body), goal, None)
    body.velocity = Vec3(10.5, 0, +2.0)
    _, progress_good = goal_reward(instruments.errors(body), goal, None)
    assert progress_good > progress_bad
    # Same body, different goal, flips the sign: reward is goal-relative.
    dive_goal = (-0.7, 0.0, 0.0)
    _, progress_dive = goal_reward(instruments.errors(body), dive_goal, None)
    # Same climbing state: worse under the dive goal than under the climb goal.
    assert progress_dive < progress_good


def awaken(will: WillTensor, desire: int, times: int = 40):
    """Pre-train the mood so one desire dominates its context."""
    for _ in range(times):
        will.tensor.reward(CONTEXT + (desire,), time_epoch=0, amount=0.5)


def test_desires_invent_their_own_targets():
    rng = np.random.default_rng(7)
    # Nothing is labeled: whatever motion coincides with reward becomes
    # the desire's aim. Rising vertical speed + reward => a "climb-ish"
    # desire is born, with nobody ever saying the word climb.
    will = WillTensor(rng)
    awaken(will, 3)
    will.decide(CONTEXT, rng, epsilon=0.0)
    for _ in range(60):
        will.learn(CONTEXT, +1.0, (1.0, 0.0, 0.0), time_epoch=0)
    goal = will.goal()
    assert goal[0] > 0.2                      # wants more vertical speed
    # The mirror will: same motion, but it kept getting punished.
    will = WillTensor(rng)
    awaken(will, 3)
    will.decide(CONTEXT, rng, epsilon=0.0)
    for _ in range(60):
        will.learn(CONTEXT, -1.0, (1.0, 0.0, 0.0), time_epoch=0)
    assert will.goal()[0] < -0.2              # same desire, opposite want


def test_will_components_affect_each_other():
    rng = np.random.default_rng(11)
    will = WillTensor(rng)
    awaken(will, 1)
    # Desire 1 dominates the mood. Bind desire 0 to it with a strong
    # positive mix link: firing desire 1 must drag desire 0 awake.
    will.mix[0, 1] = 2.0
    will.decide(CONTEXT, rng, epsilon=0.0)
    assert will.activation[0] > will.activation[5]
    # An identical will without the link leaves desire 0 asleep.
    loner = WillTensor(rng)
    awaken(loner, 1)
    loner.decide(CONTEXT, rng, epsilon=0.0)
    assert will.activation[0] > loner.activation[0]
    # Co-firing desires bind under reward and split under punishment.
    binder = WillTensor(rng)
    binder.activation = np.zeros(DESIRE_AXIS)
    binder.activation[0] = binder.activation[1] = 0.5
    before = binder.mix[0, 1]
    binder.learn(CONTEXT, +1.0, (0.0, 0.0, 0.0), time_epoch=0)
    assert binder.mix[0, 1] > before
    binder.learn(CONTEXT, -2.0, (0.0, 0.0, 0.0), time_epoch=0)
    assert binder.mix[0, 1] < before          # punishment splits them


def test_rogue_drones():
    # Two wills, identical training, different seeds: different goals from
    # birth and ever after. Personality is not programmed; it is seeded.
    goals = set()
    for seed in (1, 2, 3):
        rng = np.random.default_rng(seed)
        will = WillTensor(rng)
        goals.add(will.decide(CONTEXT, rng, epsilon=0.0))
    assert len(goals) == 3


def test_flight_with_will():
    """S11 part 2: the hierarchy runs — school reflexes below, emergent
    desires above. The drone survives with a will at the stick, the will's
    aims move (desires learn what paid off), and two wills diverge."""
    from skynerd.learning.flight import fly_episode
    from skynerd.learning.school import FlightSchool
    from skynerd.learning.servo_tensor import ServoPolicy

    instruments = FlightInstruments(trim_speed=TRIM_SPEED)
    school = FlightSchool(instruments)
    aims = []
    for seed in (1, 2):
        will = WillTensor(np.random.default_rng(seed))
        aim0 = will.aim.copy()
        policy = ServoPolicy()
        school.teach(policy)
        policy.epsilon = 0.3
        t, outcome, _, _ = fly_episode(policy, instruments, 0,
                                       np.random.default_rng(10 + seed),
                                       epsilon_override=0.3, will=will)
        assert t > 5.0, f"willed schooled drone died instantly: {t:.1f}s {outcome}"
        assert np.abs(will.aim - aim0).max() > 0.0, "will learned nothing"
        aims.append(will.aim.copy())
    assert not np.allclose(aims[0], aims[1]), "wills did not diverge"


def main():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"PASS {t.__name__}")
    print(f"\n{len(tests)} tests passed.")


if __name__ == "__main__":
    main()
