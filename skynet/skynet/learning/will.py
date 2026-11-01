"""The Will: an emergent goal layer (S11, the Singularity brain).

No named intents, no hand-designed targets. The will is a population of
unlabeled desires living on a 16-dimensional latent axis. Nothing assigns
them roles: each desire's meaning is whatever error-motion historically
paid off while it was active. Three learned structures, one Hebbian rule
each (signed reward — the anti-Hebbian arm un-learns what punished):

- mood  (SparseHebbianTensor, context x desire): which desires wake in
  which situations. Reinforced by outcome reward, weighted by how active
  each desire was.
- aim   (desire x 3 error channels): what each desire wants. Grows toward
  the error motion that coincided with reward while the desire was active,
  shrinks away from motion that punished. The drone's goal is aim^T . h —
  desires invent their own targets.
- mix   (desire x desire, symmetric, zero diagonal): how desires excite or
  inhibit each other. Co-firing under reward binds desires together;
  co-firing under punishment splits them. Desire activation carries
  momentum through mix, so the will has moods, not just reflexes.

The goal the will hands down (L2 -> L3) is a target vector on the
instrument error channels (vs_err, bank_err, speed_err), same shape the
old named intents produced — but nothing named it. A born will has a
near-zero aim: level flight. Personalities diverge from seeded random
initialisation plus each drone's own reinforcement history: two drones
raised identically still grow different wills. Rogue drones are a feature.
"""
from __future__ import annotations

import numpy as np

from skynet.tensors.sparse_hebbian import SparseHebbianTensor

DESIRE_AXIS = 16          # unlabeled latent desires (power of two for Haar)
ERROR_CHANNELS = 4        # (vs_err, bank_err, speed_err, heading_err)
AIM_LIMIT = 1.0           # goal channels live in [-1, 1]
MIX_LIMIT = 1.0
LR_AIM = 0.02             # aim learns per decision window
LR_MIX = 0.2              # desire binding learns per decision window
SPONTANEITY = 2.0         # exploration noise scale in desire space
ACTIVATION_FLOOR = 0.05   # desires below this are spectators, not learners


class WillTensor:
    """One drone's will: context -> desire population -> learned goal."""

    def __init__(self, rng: np.random.Generator | None = None, *,
                 learning_enabled: bool = True) -> None:
        rng = rng if rng is not None else np.random.default_rng()
        self.tensor = SparseHebbianTensor(
            key_shape=(8, 8, 8, 8, DESIRE_AXIS), time_depth=8,
            baseline=0.1, max_weight=5.0, min_weight=0.0,
            learning_enabled=learning_enabled)
        self.aim = rng.normal(0.0, 0.05, (DESIRE_AXIS, ERROR_CHANNELS))
        raw = rng.normal(0.0, 0.1, (DESIRE_AXIS, DESIRE_AXIS))
        self.mix = (raw + raw.T) / 2.0
        np.fill_diagonal(self.mix, 0.0)
        self.activation = np.full(DESIRE_AXIS, 1.0 / DESIRE_AXIS)
        self.learning_enabled = learning_enabled
        # The bearing this will currently wants to fly ("turn to bearing
        # 290"). Set externally by command or by a navigational desire.
        self.target_bearing = None

    def decide(self, context: tuple, rng: np.random.Generator, *,
               epsilon: float = 0.3) -> tuple:
        """Wake the desire population for this context; returns the goal.

        Mood scores seed the activation, the mix matrix lets last will's
        desires leak into this one (mood momentum), and epsilon shoots
        spontaneity noise into desire space — rogue impulses.
        """
        raw = np.array([
            self.tensor.score(tuple(context) + (d,), time_epoch=0, decay=1.0)
            for d in range(DESIRE_AXIS)])
        raw = raw + self.mix @ self.activation
        if rng.uniform() < epsilon:
            raw = raw + rng.normal(0.0, SPONTANEITY, DESIRE_AXIS)
        active = np.maximum(raw, 0.0)
        total = active.sum()
        self.activation = (active / total if total > 0.0
                           else np.full(DESIRE_AXIS, 1.0 / DESIRE_AXIS))
        return self.goal()

    def imprint_lesson(self, context: tuple, desire: int, targets: tuple,
                       *, amount: float = 2.0) -> None:
        """Flight school's handshake with the will: the instructor names the
        maneuver. The lesson's desire gets a mood stamp at this context, and
        its aim is set to the maneuver's instrument targets — after this,
        desire ``desire`` means "fly this maneuver" whenever it wakes.
        """
        self.tensor.reward(tuple(context) + (desire,),
                           time_epoch=0, amount=amount)
        # Each lesson refines the family's aim: 30-deg and 60-deg left turns
        # both speak for the left desire, and the aim settles between them.
        wanted = np.asarray(targets, dtype=float)
        if wanted.size < self.aim.shape[1]:
            wanted = np.concatenate([wanted, np.zeros(self.aim.shape[1]
                                                      - wanted.size)])
        self.aim[desire] = (self.aim[desire] + wanted) / 2.0

    def goal(self) -> tuple:
        """Current instrument-error targets: the will's demand, L2 -> L3."""
        return tuple(float(np.clip(channel, -AIM_LIMIT, AIM_LIMIT))
                     for channel in self.aim.T @ self.activation)

    def learn(self, context: tuple, reward: float, error_delta,
              time_epoch: int) -> None:
        """One Hebbian stroke per decision window.

        error_delta is the instrument-error motion observed over the window
        (Vec3 or 3-sequence). Positive reward: every sufficiently active
        desire is credited in mood, drags its aim toward the motion that
        paid off, and binds to its co-firing peers. Negative reward does
        the mirror image.
        """
        if not self.learning_enabled:
            return
        delta = (error_delta.as_array() if hasattr(error_delta, "as_array")
                 else np.asarray(error_delta, dtype=float))
        delta = np.concatenate([delta, np.zeros(self.aim.shape[1]
                                                - delta.size)])
        h = self.activation
        for d in range(DESIRE_AXIS):
            if h[d] > ACTIVATION_FLOOR:
                self.tensor.reward(tuple(context) + (d,),
                                   time_epoch=time_epoch, amount=reward * h[d])
        self.aim += LR_AIM * reward * np.outer(h, delta)
        np.clip(self.aim, -AIM_LIMIT, AIM_LIMIT, out=self.aim)
        self.mix += LR_MIX * reward * np.outer(h, h)
        np.fill_diagonal(self.mix, 0.0)
        np.clip(self.mix, -MIX_LIMIT, MIX_LIMIT, out=self.mix)


def goal_reward(errors, goal: tuple, previous_progress: float | None,
                scale: float = 5.0) -> tuple:
    """Dense L4 reward: progress toward the goal's error targets, per tick.

    Progress = how much closer the three errors got to their targets, weighted
    equally. Half-done earns half; moving the wrong way earns negative.
    Returns (reward, new_progress) — the caller feeds new_progress back in.
    Accepts errors as a Vec3, tuple, or array.
    """
    e = tuple(errors.as_array()) if hasattr(errors, "as_array") else tuple(errors)
    progress = -sum(abs(ei - gi) for ei, gi in zip(e, goal))
    if previous_progress is None:
        return 0.0, progress
    return (progress - previous_progress) * scale, progress
