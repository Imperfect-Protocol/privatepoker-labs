"""ServoTensor: the Hebbian autopilot. State(8x8x8x8) x Surface(8) x Action(4) x Time(8).

State channels are the cockpit panel the student reads: VSI (vertical speed
error), bank, ASI (airspeed error), RPM (throttle lever position), and the
HSI (bearing error to the will's target).

The brain commands surfaces directly — no deterministic controller in between.
Each decision (10 Hz; physics and reward stay at 50 Hz):

1. Instruments bin the flight state into a state key (vs, bank, speed errors).
2. The policy scores the hold-all option and each single-surface push
   (one surface at a time, one direction). Only ONE control changes per
   decision — a smooth hill-climb, not four independent random walks.
3. epsilon-greedy picks; the surface command accumulates the delta (clamped
   by the servo hardware limits in SurfacesState.command).
4. The dense (baseline-centered) reward credits the last decisions with
   decaying amounts — eligibility through the Time(8) ring.
5. Negative outcomes (crash, stall) hit the same keys harder: the
   anti-Hebbian arm.

All weights live in one generic SparseHebbianTensor, so wavelet denoising,
sparsity, age decay and the implicit unit tensor under --no-learning carry
over for free. Axis padding: surface axis is 8 (4 real channels + hold-all
slot at index 4); action axis is 4 (0 = ease off, 1 = hold, 2 = push).
"""
from __future__ import annotations

import numpy as np

from skynet.learning.reward import ERROR_BINS
from skynet.physics.surfaces import SURFACE_LIMITS
from skynet.tensors.sparse_hebbian import SparseHebbianTensor

EASE_OFF, HOLD, PUSH = 0, 1, 2
ACTION_AXIS = 4             # padded to 4 for Haar; index 3 is never chosen
SURFACE_AXIS = 8            # 4 real channels; index 4 is the hold-all slot
HOLD_ALL_SURFACE = 4
ACTION_DELTA = 0.05         # surface deflection change per decision, radians
ELIGIBILITY_DECISIONS = 10  # credit the last N decisions (1 s of flight)
ELIGIBILITY_DECAY = 0.7


class ServoPolicy:
    """Owns the servo tensor; turns flight state into surface commands."""

    def __init__(self, *, baseline: float = 0.1, max_weight: float = 5.0,
                 min_weight: float = 0.0, learning_enabled: bool = True) -> None:
        self.tensor = SparseHebbianTensor(
            key_shape=(ERROR_BINS, ERROR_BINS, ERROR_BINS, ERROR_BINS,
                       ERROR_BINS, SURFACE_AXIS, ACTION_AXIS),
            time_depth=8, baseline=baseline, max_weight=max_weight,
            min_weight=min_weight, learning_enabled=learning_enabled)
        self.epsilon = 0.5

    def _score(self, state_key: tuple, surface: int, level: int) -> float:
        return self.tensor.score(state_key + (surface, level), time_epoch=0, decay=1.0)

    def choose(self, state_key: tuple, rng: np.random.Generator) -> np.ndarray:
        """Pick hold-all or one single-surface push, epsilon-greedily.

        Returns an actions array of 4 channel entries, each HOLD, with at
        most one entry set to EASE_OFF or PUSH.
        """
        actions = np.full(4, HOLD, dtype=np.int64)
        if rng.uniform() < self.epsilon:
            choice = int(rng.integers(1 + 4 * 2))   # 0 = hold-all, else surface/direction
        else:
            choice = 0
            best = self._score(state_key, HOLD_ALL_SURFACE, HOLD)
            for surface in range(4):
                for level in (EASE_OFF, PUSH):
                    score = self._score(state_key, surface, level)
                    if score > best:
                        best, choice = score, 1 + surface * 2 + (level == PUSH)
        if choice > 0:
            surface = (choice - 1) // 2
            actions[surface] = EASE_OFF if (choice - 1) % 2 == 0 else PUSH
        return actions

    def command_from(self, current: np.ndarray, actions: np.ndarray) -> np.ndarray:
        """Accumulate action deltas onto current commands, clamped to hardware."""
        targets = current.copy()
        for surface in range(4):
            level = int(actions[surface])
            if level == HOLD:
                continue
            delta = ACTION_DELTA if level == PUSH else -ACTION_DELTA
            low, high = SURFACE_LIMITS[surface]
            targets[surface] = max(low, min(high, targets[surface] + delta))
        return targets

    def reinforce(self, recent: list, reward: float, time_epoch: int) -> None:
        """Credit recent (state_key, actions) decisions with decaying eligibility.

        ``recent`` is a list of (state_key, actions) pairs, oldest first, with
        the current decision last. The credited key is whichever single
        option the decision took (hold-all or one surface push).
        """
        if not self.tensor.learning_enabled or reward == 0.0 or not recent:
            return
        amount = reward
        for state_key, actions in reversed(recent[-ELIGIBILITY_DECISIONS:]):
            changed = [(s, int(actions[s])) for s in range(4) if actions[s] != HOLD]
            if changed:
                surface, level = changed[0]
            else:
                surface, level = HOLD_ALL_SURFACE, HOLD
            self.tensor.reward(state_key + (surface, level),
                               time_epoch=time_epoch, amount=amount)
            amount *= ELIGIBILITY_DECAY

    def imprint(self, state_key: tuple, actions: np.ndarray, *,
                amount: float = 1.0, time_epoch: int = 0) -> None:
        """Flight school: stamp a demonstrated decision straight into memory.

        The instructor chose the actions, not the drone — this is the
        watch-and-remember channel, reward-free and immediate. Every moved
        surface is a fact about this state and gets its own stamp; still
        hands stamp the hold-all option.
        """
        changed = [(s, int(actions[s])) for s in range(4) if actions[s] != HOLD]
        if not changed:
            changed = [(HOLD_ALL_SURFACE, HOLD)]
        for surface, level in changed:
            self.tensor.reward(state_key + (surface, level),
                               time_epoch=time_epoch, amount=amount)

    def denoise(self, rng: np.random.Generator, *, threshold: float = 0.006) -> None:
        self.tensor.wavelet_denoise(rng, threshold=threshold, prune_threshold=0.002)
