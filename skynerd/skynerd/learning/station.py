"""The station: the swarm's hippocampus, safe at the rear.

Drones die; the organism remembers. Every drone that finishes a life (or
evacuates on the way down) hands its sparse tensor to the station, which
blends it in at a small rate — slow consolidation against catastrophic
forgetting. New waves seed their tensors from the station: they spawn
knowing what the swarm died learning.
"""
from __future__ import annotations

import numpy as np

from skynerd.learning.servo_tensor import ServoPolicy
from skynerd.tensors.sparse_hebbian import SparseHebbianTensor


class Station:
    def __init__(self, *, merge_rate: float = 0.1) -> None:
        self.merge_rate = float(merge_rate)
        self.tensor = SparseHebbianTensor(
            key_shape=(8, 8, 8, 8, 4),
            time_depth=8, baseline=0.1, max_weight=5.0, min_weight=0.0,
            learning_enabled=True)
        self.lives_merged = 0
        self.deaths_merged = 0

    def consolidate(self, policy: ServoPolicy, *, died: bool) -> None:
        """Blend one drone's observed slots into the long-term memory.

        For every key the drone holds, each live time slot's weight moves a
        fraction ``merge_rate`` toward the drone's value. Station slots keep
        their own epochs (aged by the station's own epoch counter), so decay
        stays the station's business, not the dead drone's.
        """
        source = policy.tensor
        if not source.learning_enabled or not source.values:
            return
        for key, values in source.values.items():
            epochs = source.epochs[key]
            live = epochs >= 0
            if not np.any(live):
                continue
            if key not in self.tensor.values:
                self.tensor.values[key] = np.full(
                    self.tensor.time_depth, self.tensor.baseline, dtype=np.float32)
                self.tensor.epochs[key] = np.full(
                    self.tensor.time_depth, -1, dtype=np.int64)
            station_values = self.tensor.values[key]
            # Consolidate the drone's newest slot (its current epoch) into the
            # station's newest slot. Older slots are stale epochs; mean-merging
            # them dilutes the newest learning.
            newest = int(np.argmax(epochs))
            station_values[0] += self.merge_rate * (values[newest] - station_values[0])
            self.tensor.epochs[key][0] = 0
        self.lives_merged += 1
        if died:
            self.deaths_merged += 1

    def seed(self, policy: ServoPolicy) -> None:
        """Copy long-term memory into a newborn drone's tensor."""
        target = policy.tensor
        target.values = {key: values.copy() for key, values in self.tensor.values.items()}
        target.epochs = {key: epochs.copy() for key, epochs in self.tensor.epochs.items()}
        target.updates_since_filter = 0
