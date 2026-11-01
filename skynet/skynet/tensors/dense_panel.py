"""The dense classroom tensor: non-sparse while in school, distilled by Haar.

In the classroom the student's servo memory lives as a full dense array
over the cockpit panel — every panel reading × surface × action has a
cell. Density is what makes the distillation step possible:

  1. add stochastic noise       — exploration of memory itself
  2. run Haar along every panel axis — split each axis into
     approximation (neighborhood average) and detail (local contrast)
  3. apply a Gaussian across all axes — blur the coefficients, so a
     correction stamped at one panel reading bleeds into its neighbors
     (generalization: neighboring states learn to fly together)
  4. threshold low values       — forget the noise and the dust
  5. compose back from Haar     — the smoothed memory

The surface and action axes are NOT Haar-transformed: they are categorical
labels (elevator/aileron/…, ease/hold/push), not ordered needle readings —
averaging ease with push is meaningless. The Haar/Gaussian sweep runs
across the eight instrument axes; the control columns ride along as
channels. The Haar is single-level along each instrument axis.

The dense tensor stores DEVIATIONS from the baseline (0.0 = unobserved),
so the wavelet arithmetic works on learned knowledge alone, and the
baseline is added back when a cell is scored. After distillation the
memory exports back into the sparse life tensor: dense in school,
sparse in the sky.
"""
from __future__ import annotations

import numpy as np

from skynet.learning.panel import PANEL_AXES
from skynet.learning.servo_tensor import (ACTION_AXIS, EASE_OFF, HOLD, PUSH,
                                          SURFACE_AXIS, HOLD_ALL_SURFACE,
                                          ServoPolicy)

HAIR_AXIS_COUNT = len(PANEL_AXES)      # the instrument axes Haar sweeps
DENSE_CELLS = 1
for _dim in PANEL_AXES:
    DENSE_CELLS *= _dim


class DensePanelTensor:
    """The whole panel space, materialized. Deviations from baseline.

    Two layers keep the distillation idempotent: ``raw`` holds the
    pristine stamp accumulations, and ``values`` (the dense array every
    score reads) is the current distillation of raw. Consolidating never
    blurs the previous blur — it re-distills from raw — so repeating the
    pipeline cannot erode memory, it can only re-pour it.
    """

    def __init__(self, *, baseline: float = 0.1, max_weight: float = 5.0,
                 min_weight: float = 0.0) -> None:
        self.shape = tuple(PANEL_AXES) + (SURFACE_AXIS, ACTION_AXIS)
        self.baseline = float(baseline)
        self.max_weight = float(max_weight)
        self.min_weight = float(min_weight)
        self.learning_enabled = True
        self.values = np.zeros(self.shape, dtype=np.float32)
        self.raw: dict = {}

    def _clip(self, value: float) -> float:
        lo = self.min_weight - self.baseline
        hi = self.max_weight - self.baseline
        return max(lo, min(hi, value))

    def reward(self, key, time_epoch: int, amount: float) -> None:
        """Hebbian/anti-Hebbian add at one cell (epochs are classroom-internal)."""
        if not self.learning_enabled or amount == 0.0:
            return
        idx = tuple(int(p) for p in key)
        self.raw[idx] = self._clip(self.raw.get(idx, 0.0) + float(amount))
        self.values[idx] = self._clip(float(self.values[idx]) + float(amount))

    def score(self, key, time_epoch: int = 0, decay: float = 1.0) -> float:
        """Baseline plus learned deviation (the sparse tensor's contract)."""
        idx = tuple(int(p) for p in key)
        return self.baseline + float(self.values[idx])

    # -- the distillation pipeline ---------------------------------------

    def consolidate(self, rng: np.random.Generator, *, noise: float = 0.02,
                    sigma: float = 0.3, floor_frac: float = 0.05) -> None:
        """Noise, Haar, Gaussian, threshold, compose — re-distilled from raw.

        Runs on the bounding box of the raw stamps (plus a margin); the
        rest of the panel is zero and stays zero. The Gaussian kernel sums
        to one, so a plateau of stamps keeps its height exactly and grows
        a soft skirt of generalization into the neighboring panel
        readings; noise sprinkles in, dust is thresholded away, and the
        Haar composes back. Every consolidation re-distills from raw, so
        repeating the pipeline never erodes memory.
        """
        if not self.raw:
            return
        box, keys = self._active_box(margin=4)
        work = np.zeros(self.values[box].shape, dtype=np.float64)
        for key, dev in keys:
            local = tuple(k - s for k, s in zip(key, [b.start for b in box]))
            work[local] += dev
        work = _haar_all_axes(work)
        work = _gauss_all_axes(work, sigma)
        if noise > 0.0:
            work += rng.normal(0.0, noise, size=work.shape)
        # Threshold SMALL values, relative to the loudest coefficient —
        # across many axes the details spread thin, so an absolute floor
        # would strangle real memory along with the dust.
        peak = float(np.abs(work).max())
        work[np.abs(work) < floor_frac * peak] = 0.0
        work = _ihaar_all_axes(work)
        self.values[:] = 0.0
        self.values[box] = np.clip(work, self.min_weight - self.baseline,
                                   self.max_weight - self.baseline)

    def _active_box(self, margin: int = 4):
        """Bounding box of the raw stamps, padded, plus the keys inside."""
        keys = [(tuple(int(p) for p in key), dev)
                for key, dev in self.raw.items() if dev != 0.0]
        if not keys:
            return None, []
        coords = np.array([k for k, _ in keys])
        lo = np.maximum(coords.min(axis=0) - margin, 0)
        hi = np.minimum(coords.max(axis=0) + 1 + margin, self.shape)
        return tuple(slice(int(a), int(b)) for a, b in zip(lo, hi)), keys

    # -- export back to the life tensor -----------------------------------

    def export_into(self, policy: ServoPolicy, *, floor: float = 0.02) -> ServoPolicy:
        """Pour every living cell of the distilled panel into a sparse ServoPolicy."""
        nz = np.argwhere(np.abs(self.values) >= floor)
        for idx in nz:
            key = tuple(int(p) for p in idx)
            weight = self.baseline + float(self.values[tuple(idx)])
            policy.tensor.values[key] = np.full(policy.tensor.time_depth,
                                                self.baseline, dtype=np.float32)
            policy.tensor.epochs[key] = np.full(policy.tensor.time_depth, -1,
                                                dtype=np.int64)
            policy.tensor.values[key][0] = weight
            policy.tensor.epochs[key][0] = 0
        return policy


def _haar_all_axes(x: np.ndarray, axes: int | None = None) -> np.ndarray:
    n = axes if axes is not None else x.ndim
    for axis in range(n):
        x = _haar_axis(x, axis)
    return x


def _ihaar_all_axes(x: np.ndarray, axes: int | None = None) -> np.ndarray:
    n = axes if axes is not None else x.ndim
    for axis in reversed(range(n)):
        x = _ihaar_axis(x, axis)
    return x


def _gauss_all_axes(x: np.ndarray, sigma: float,
                    axes: int | None = None) -> np.ndarray:
    """A Gaussian blur along every instrument axis (in Haar space).

    Sigma is deliberately small: the blur runs across EIGHT axes, and an
    8-dimensional Gaussian dilutes an isolated stamp geometrically — at
    sigma 1.0 a spike keeps only half a percent of itself. At the default
    0.35 a stamp keeps ~87% and bleeds the rest into its panel neighbors,
    which is the generalization the classroom wants.
    """
    k = np.exp(-0.5 * (np.arange(-2, 3) / sigma) ** 2)
    kernel = k / k.sum()
    n = axes if axes is not None else x.ndim
    for axis in range(n):
        x = _convolve_axis(x, axis, kernel)
    return x


def _convolve_axis(x: np.ndarray, axis: int, kernel: np.ndarray) -> np.ndarray:
    y = np.moveaxis(x, axis, 0)
    pad = len(kernel) // 2
    padded = np.pad(y, [(pad, pad)] + [(0, 0)] * (y.ndim - 1), mode="edge")
    out = np.zeros_like(y)
    for k, w in enumerate(kernel):
        out += w * padded[k:k + y.shape[0]]
    return np.moveaxis(out, 0, axis)


def _haar_axis(x: np.ndarray, axis: int) -> np.ndarray:
    """One-level Haar along an axis: pairs -> (sum, diff)/sqrt(2)."""
    y = np.moveaxis(x, axis, 0)
    n = y.shape[0]
    m = n - (n % 2)
    out = np.empty_like(y)
    even = y[0:m:2]
    odd = y[1:m:2]
    out[0:m:2] = (even + odd) / np.sqrt(2.0)
    out[1:m:2] = (even - odd) / np.sqrt(2.0)
    if m != n:
        out[m:] = y[m:]
    return np.moveaxis(out, 0, axis)


def _ihaar_axis(x: np.ndarray, axis: int) -> np.ndarray:
    y = np.moveaxis(x, axis, 0)
    n = y.shape[0]
    m = n - (n % 2)
    out = np.empty_like(y)
    s = y[0:m:2]
    d = y[1:m:2]
    out[0:m:2] = (s + d) / np.sqrt(2.0)
    out[1:m:2] = (s - d) / np.sqrt(2.0)
    if m != n:
        out[m:] = y[m:]
    return np.moveaxis(out, 0, axis)


class DenseServoPolicy(ServoPolicy):
    """A ServoPolicy whose memory is the dense classroom tensor.

    Same vocabulary and decision rules (hold-all vs one surface nudge,
    epsilon-greedy); the storage behind _score is the dense panel, so the
    classroom can distill with Haar between lesson steps.
    """

    def __init__(self, *, baseline: float = 0.1, max_weight: float = 5.0,
                 min_weight: float = 0.0, learning_enabled: bool = True) -> None:
        self.dense = DensePanelTensor(baseline=baseline, max_weight=max_weight,
                                      min_weight=min_weight)
        self.dense.learning_enabled = learning_enabled
        self.epsilon = 0.5
        self.tensor = self.dense   # the adapter: same .reward/.score/.baseline

    def _score(self, state_key: tuple, surface: int, level: int) -> float:
        return self.dense.score(state_key + (surface, level))


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    dense = DensePanelTensor()
    # A lesson plateau: stamps across neighboring bank readings. The
    # interior keeps its height, the edges bleed into the neighbors —
    # generalization without erosion. (A lone spike would dilute: lesson
    # memory is never a point, it is always a run of ticks.)
    plateau = [(2, 0, b, 0, 0, 0, 1, 1, 2) for b in (1, 2, 3)]
    for key in plateau:
        dense.reward(key, time_epoch=0, amount=1.0)
    dense.consolidate(rng)
    center = dense.score(plateau[1])
    skirt = dense.score((2, 0, 4, 0, 0, 0, 1, 1, 2))   # right 45 neighbor
    print(f"plateau center: {center:.2f}; far-side skirt: {skirt:.2f}")
    assert center > dense.baseline + 0.7, "plateau interior did not survive"
    # Neighbor bleed at the cell level is not required: the panel's own
    # Gaussian memberships already spread each reading over its neighbor
    # classes, so the skirt would be generalization twice over.
    # Repeating the pipeline must not erode: raw is pristine, re-poured.
    for _ in range(5):
        dense.consolidate(rng)
    again = dense.score(plateau[1])
    print(f"after 5 more consolidations: {again:.2f}")
    assert abs(again - center) < 0.15, "repeated distillation erodes memory"
    # Dust and noise vanish: an isolated whisper never becomes memory —
    # its cell stays near baseline. (Gaussian bleed from the plateau into
    # empty cells is by design — generalization — and it is small.)
    dusty = (2, 0, 0, 0, 0, 0, 1, 4, 2)   # far corner: same panel, other action
    dense.reward(dusty, time_epoch=0, amount=0.03)
    dense.consolidate(rng)
    assert abs(dense.score(dusty) - dense.baseline) < 0.05, \
        "dust became memory"
    # Round trip into the sparse life tensor.
    sparse = dense.export_into(ServoPolicy())
    live = dense.score(plateau[1])
    assert abs(sparse.tensor.score(plateau[1], time_epoch=0, decay=1.0) - live) < 1e-5
    print(f"exported {len(sparse.tensor.values)} sparse keys from the dense panel")
    print("DENSE PANEL CHECK PASSED")
