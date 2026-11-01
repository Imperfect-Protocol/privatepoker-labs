"""Generic sparse Hebbian tensor with time-slotted eligibility.

Ported and generalized from p2p_nerual_spatial/simulator_sp.py. The original
was hard-wired to Source(8x8) x Target(8x8) x Neighbour(8) x Time(8); skynet's
version takes the key-axis sizes as a parameter, so the identity tensor
(8,8,8,8,8), the spatial prior (8,8), and the servo tensor (8,) all reuse this
one class.

Design laws kept from the original:

- Sparse: only observed (key, time-slot) combinations allocate memory. Unobserved
  combinations cost nothing and score at baseline.
- Time slots: each key owns an 8-slot history; a slot re-writes when the epoch
  wraps, so old observations age out without timers.
- Score decay: learned deviation from baseline decays with observation age.
- With learning disabled the tensor is an implicit logical unit tensor: score()
  returns 1.0 everywhere, nothing allocates.
- Wavelet denoise composes the full logical dense tensor, runs the chaos
  pipeline, and projects back onto observed keys and live epoch slots only —
  noise cannot manufacture unobserved protocol facts.

Skynet addition: rewards may be negative (the anti-Hebbian arm). A ``min_weight``
floor replaces the original's implicit baseline floor, so death/punishment can
depress entries below baseline. ``baseline`` remains the neutral starting value.
"""
from __future__ import annotations

from typing import Dict, Sequence, Tuple

import numpy as np

from skynet.tensors.wavelet import (
    NODE_TENSOR_PRUNE_THRESHOLD,
    WAVELET_GAUSSIAN_SIGMA,
    WAVELET_NOISE_STD,
    WAVELET_THRESHOLD,
    prepare_and_compose_wavelet_tensor,
)

Key = Tuple[int, ...]


class SparseHebbianTensor:
    """Sparse logical tensor over ``key_shape`` axes plus a Time axis.

    Keys are integer tuples with one entry per axis in ``key_shape``. Each
    observed key allocates one float32 vector of length ``time_depth`` plus its
    epoch (observation generation) vector.
    """

    def __init__(self, key_shape: Sequence[int], time_depth: int = 8,
                 baseline: float = 0.1, max_weight: float = 5.0,
                 min_weight: float = 0.0, learning_enabled: bool = True) -> None:
        self.key_shape = tuple(int(dim) for dim in key_shape)
        if not self.key_shape or any(dim < 1 for dim in self.key_shape):
            raise ValueError("key_shape must be non-empty with positive dimensions")
        for dim in self.key_shape + (time_depth,):
            if dim & (dim - 1):
                raise ValueError("all tensor axes must be powers of two for Haar")
        self.time_depth = int(time_depth)
        self.learning_enabled = bool(learning_enabled)
        # With learning disabled, represent a logical dense unit tensor implicitly.
        self.baseline = float(baseline) if self.learning_enabled else 1.0
        self.max_weight = float(max_weight)
        self.min_weight = float(min_weight)
        self.values: Dict[Key, np.ndarray] = {}
        self.epochs: Dict[Key, np.ndarray] = {}
        self.reward_total = 0.0
        self.updates_since_filter = 0
        self.wavelet_filter_passes = 0
        self.wavelet_nonzero_coefficients_last = 0

    @property
    def logical_shape(self) -> Tuple[int, ...]:
        """Dense shape: key axes followed by the Time axis."""
        return self.key_shape + (self.time_depth,)

    @property
    def stored_combinations(self) -> int:
        return len(self.values)

    @property
    def stored_time_slots(self) -> int:
        return sum(int(np.count_nonzero(epochs >= 0)) for epochs in self.epochs.values())

    @staticmethod
    def _key(parts: Sequence[int]) -> Key:
        return tuple(int(p) for p in parts)

    def reward(self, key: Sequence[int], time_epoch: int, amount: float) -> None:
        """Hebbian (amount > 0) or anti-Hebbian (amount < 0) update at a time slot.

        The amount applies to the deviation from baseline, clamped into
        [min_weight, max_weight]. Zero amounts change nothing.
        """
        if not self.learning_enabled or amount == 0.0:
            return
        key = self._key(key)
        if key not in self.values:
            self.values[key] = np.full(self.time_depth, self.baseline, dtype=np.float32)
            self.epochs[key] = np.full(self.time_depth, -1, dtype=np.int64)
        slot = int(time_epoch) % self.time_depth
        if int(self.epochs[key][slot]) != int(time_epoch):
            self.values[key][slot] = self.baseline
            self.epochs[key][slot] = int(time_epoch)
        old = float(self.values[key][slot])
        new = min(self.max_weight, max(self.min_weight, old + float(amount)))
        self.values[key][slot] = new
        self.reward_total += max(0.0, new - old)
        self.updates_since_filter += 1

    def wavelet_denoise(self, rng: np.random.Generator, *,
                        noise_std: float = WAVELET_NOISE_STD,
                        noise_enabled: bool = True,
                        haar_enabled: bool = True,
                        gaussian_enabled: bool = True,
                        gaussian_sigma: float = WAVELET_GAUSSIAN_SIGMA,
                        threshold: float = WAVELET_THRESHOLD,
                        prune_threshold: float = NODE_TENSOR_PRUNE_THRESHOLD) -> Tuple[int, int]:
        """Denoise the logical dense tensor, then project back to sparse storage.

        All axes participate in the Haar transform. The composed tensor is
        projected onto observed sparse keys and live epoch slots so noise cannot
        manufacture unobserved facts or turn the sparse table dense. Near-neutral
        weights are pruned back to genuinely unobserved. Returns
        (nonzero_coefficients, pruned_keys).
        """
        if not self.learning_enabled or not self.values or self.updates_since_filter <= 0:
            return 0, 0
        dense = np.zeros(self.logical_shape, dtype=np.float32)
        valid_by_key: Dict[Key, np.ndarray] = {}
        for key, values in self.values.items():
            epochs = self.epochs[key]
            valid = epochs >= 0
            if np.any(valid):
                deviations = np.zeros(self.time_depth, dtype=np.float32)
                deviations[valid] = values[valid] - self.baseline
                dense[key] = deviations
                valid_by_key[key] = valid.copy()
        if not valid_by_key:
            self.updates_since_filter = 0
            return 0, 0

        composed, coefficient_count = prepare_and_compose_wavelet_tensor(
            dense, rng, noise_std=noise_std, noise_enabled=noise_enabled,
            haar_enabled=haar_enabled, gaussian_enabled=gaussian_enabled,
            gaussian_sigma=gaussian_sigma, threshold=threshold,
        )

        removed_keys = []
        for key, valid in valid_by_key.items():
            filtered = np.clip(self.baseline + composed[key],
                               self.min_weight, self.max_weight)
            values = self.values[key]
            epochs = self.epochs[key]
            values[valid] = filtered[valid]
            # Soft-thresholded, near-neutral weights are removed rather than kept
            # as numerical noise. Their epoch slots become genuinely unobserved.
            prune = valid & (np.abs(values - self.baseline) < prune_threshold)
            values[prune] = self.baseline
            epochs[prune] = -1
            if not np.any(epochs >= 0):
                removed_keys.append(key)
        for key in removed_keys:
            self.values.pop(key, None)
            self.epochs.pop(key, None)

        self.wavelet_filter_passes += 1
        self.wavelet_nonzero_coefficients_last = coefficient_count
        self.updates_since_filter = 0
        return coefficient_count, len(removed_keys)

    def score(self, key: Sequence[int], time_epoch: int, decay: float = 0.97) -> float:
        """Age-decayed learned score for a key; baseline when unobserved."""
        if not self.learning_enabled:
            return 1.0
        key = self._key(key)
        if key not in self.values:
            return self.baseline
        weights, epochs = self.values[key], self.epochs[key]
        valid = epochs >= 0
        if not np.any(valid):
            return self.baseline
        delta = weights[valid].astype(np.float64) - self.baseline
        age = np.maximum(0, int(time_epoch) - epochs[valid])
        learned = float(np.sum(delta * np.power(float(decay), age)))
        return min(self.max_weight, max(self.min_weight, self.baseline + learned))
