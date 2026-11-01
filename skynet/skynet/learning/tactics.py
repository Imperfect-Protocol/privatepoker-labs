"""Tactics tensor (Tensor 3): SwarmEye-context -> intent -> outcome.

The slow-reward layer on top of flight. Contexts are coarse hashes of the
glued SwarmEye tensor (perception distilled to a key); intents are named
manoeuvres; outcomes are the slow rewards (kills, survivable engagements).

House style throughout: sparse observed entries, age decay, exact metadata
(identities of tank sightings are addresses, never blurred), wavelet
denoise ready, implicit unit tensor when learning is disabled.
"""
from __future__ import annotations

import hashlib

import numpy as np

from skynet.tensors.sparse_hebbian import SparseHebbianTensor

APPROACH = "approach"
STRAFE = "strafe"
RETREAT = "retreat"
INTENTS = (APPROACH, STRAFE, RETREAT)
INTENT_AXIS = 4             # padded to a power of two for Haar


def context_from_eye(glued_eye: np.ndarray) -> tuple:
    """Distil the glued SwarmEye into a coarse 4-part context key.

    Compresses the (M, 8, 8, 8, 8) decision tensor to what the tactics layer
    can afford to learn about: per-axis occupancy profile (azimuth cones,
    elevation bands, range shells) plus total contact mass. Sparse, stable
    under threshold noise: each part is a small integer.
    """
    if glued_eye.size == 0:
        return (0, 0, 0, 0)
    mass_by_az = glued_eye.sum(axis=(0, 2, 3, 4))   # over Source, El, Rng, Time
    mass_by_el = glued_eye.sum(axis=(0, 1, 3, 4))   # over Source, Az, Rng, Time
    mass_by_rng = glued_eye.sum(axis=(0, 1, 2, 4))  # over Source, Az, El, Time
    total = glued_eye.sum()

    def dominant_bin(profile: np.ndarray, bins: int) -> int:
        return int(np.argmax(profile)) if profile.sum() > 0 else bins // 2

    return (dominant_bin(mass_by_az, 8),
            dominant_bin(mass_by_el, 8),
            dominant_bin(mass_by_rng, 8),
            min(int(total) // 8, 7))


class TacticsPolicy:
    """Owns Tensor 3; turns perception contexts into manoeuvre intents."""

    def __init__(self, *, learning_enabled: bool = True) -> None:
        self.tensor = SparseHebbianTensor(
            key_shape=(8, 8, 8, 8, INTENT_AXIS), time_depth=8,
            baseline=0.1, max_weight=5.0, min_weight=0.0,
            learning_enabled=learning_enabled)

    def choose(self, context: tuple, rng: np.random.Generator, *,
               epsilon: float = 0.3) -> str:
        if rng.uniform() < epsilon:
            return INTENTS[int(rng.integers(len(INTENTS)))]
        best_intent, best_score = APPROACH, -np.inf
        for index, intent in enumerate(INTENTS):
            score = self.tensor.score(context + (index,), time_epoch=0, decay=1.0)
            if score > best_score:
                best_score, best_intent = score, intent
        return best_intent

    def reinforce(self, context: tuple, intent: str, amount: float,
                  time_epoch: int) -> None:
        if intent in INTENTS:
            self.tensor.reward(context + (INTENTS.index(intent),),
                               time_epoch=time_epoch, amount=amount)

    def consolidate_from(self, other: "TacticsPolicy", rate: float) -> None:
        """Station-side merge: same consolidation law as the servo station."""
        for key, values in other.tensor.values.items():
            epochs = other.tensor.epochs[key]
            if not np.any(epochs >= 0):
                continue
            if key not in self.tensor.values:
                self.tensor.values[key] = np.full(
                    self.tensor.time_depth, self.tensor.baseline, dtype=np.float32)
                self.tensor.epochs[key] = np.full(self.tensor.time_depth, -1,
                                                  dtype=np.int64)
            newest = int(np.argmax(epochs))
            self.tensor.values[key][0] += rate * (values[newest]
                                                  - self.tensor.values[key][0])
            self.tensor.epochs[key][0] = 0
