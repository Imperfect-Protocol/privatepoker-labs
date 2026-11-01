"""Tests for skynerd.tensors.sparse_hebbian. Run: python tests/test_sparse_hebbian.py"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from skynerd.tensors.sparse_hebbian import SparseHebbianTensor


def test_reward_and_score_decay():
    tensor = SparseHebbianTensor(key_shape=(8, 8, 8, 8, 8), time_depth=8,
                                 baseline=0.1, max_weight=5.0)
    key = (1, 2, 3, 4, 5)
    assert tensor.score(key, time_epoch=0) == 0.1          # unobserved -> baseline
    tensor.reward(key, time_epoch=0, amount=0.5)
    fresh = tensor.score(key, time_epoch=0)
    assert fresh > 0.1
    # Age decay: the same observation scores lower as epochs pass.
    later = tensor.score(key, time_epoch=4)
    much_later = tensor.score(key, time_epoch=40)
    assert 0.1 < later < fresh
    assert 0.1 < much_later < later
    assert tensor.stored_combinations == 1
    assert tensor.stored_time_slots == 1


def test_max_weight_clamp():
    tensor = SparseHebbianTensor(key_shape=(8, 8), time_depth=8, baseline=0.1,
                                 max_weight=1.0)
    key = (3, 4)
    for _ in range(20):
        tensor.reward(key, time_epoch=0, amount=0.5)
    assert tensor.score(key, time_epoch=0) == 1.0


def test_time_slot_rewrite():
    tensor = SparseHebbianTensor(key_shape=(8, 8), time_depth=8, baseline=0.1,
                                 max_weight=5.0)
    key = (1, 1)
    tensor.reward(key, time_epoch=0, amount=1.0)
    # Epoch 8 wraps into slot 0, wiping epoch-0's observation.
    tensor.reward(key, time_epoch=8, amount=0.5)
    assert tensor.stored_time_slots == 1
    # Score at epoch 8 only sees the fresh observation.
    expected = 0.1 + 0.5
    assert abs(tensor.score(key, time_epoch=8) - expected) < 1e-6


def test_negative_reward_punishes():
    tensor = SparseHebbianTensor(key_shape=(8,), time_depth=8, baseline=0.5,
                                 max_weight=5.0, min_weight=0.0)
    key = (2,)
    tensor.reward(key, time_epoch=0, amount=1.0)
    score_before = tensor.score(key, time_epoch=1)
    tensor.reward(key, time_epoch=1, amount=-2.0)
    # The punished slot is clamped at the floor...
    slot = int(1) % 8
    assert float(tensor.values[key][slot]) == 0.0
    # ...and the age-decayed aggregate score drops, but older positive slots
    # still contribute until they age out — trace decay, not instant erasure.
    score_after = tensor.score(key, time_epoch=1)
    assert score_after < score_before
    assert score_after > tensor.min_weight
    # Once the positive slot's epoch wraps out of the ring, only punishment remains.
    tensor2 = SparseHebbianTensor(key_shape=(8,), time_depth=8, baseline=0.5,
                                  max_weight=5.0, min_weight=0.0)
    tensor2.reward(key, time_epoch=0, amount=-2.0)
    tensor2.reward(key, time_epoch=1, amount=-2.0)
    assert tensor2.score(key, time_epoch=1) == 0.0


def test_learning_disabled_is_implicit_unit_tensor():
    tensor = SparseHebbianTensor(key_shape=(8, 8), time_depth=8, learning_enabled=False)
    assert tensor.score((0, 0), 0) == 1.0
    assert tensor.score((7, 7), 1000) == 1.0
    tensor.reward((0, 0), 0, amount=1.0)   # no-op
    assert tensor.stored_combinations == 0
    composed, pruned = tensor.wavelet_denoise(np.random.default_rng(0))
    assert composed == 0 and pruned == 0


def test_wavelet_denoise_preserves_observed_keys():
    rng = np.random.default_rng(42)
    tensor = SparseHebbianTensor(key_shape=(8, 8, 8, 8, 8), time_depth=8,
                                 baseline=0.1, max_weight=5.0)
    strong = (1, 2, 3, 4, 5)
    weak = (5, 4, 3, 2, 1)
    noise = (2, 2, 2, 2, 2)
    for _ in range(10):
        tensor.reward(strong, time_epoch=0, amount=0.5)     # real signal: 0.5 deviation
        tensor.reward(weak, time_epoch=0, amount=0.02)      # small but real: 0.2 deviation
        tensor.reward(noise, time_epoch=0, amount=0.0001)   # noise-level: 0.001 deviation
    assert tensor.updates_since_filter > 0
    coefficients, pruned = tensor.wavelet_denoise(rng, threshold=0.006,
                                                  prune_threshold=0.002)
    assert coefficients > 0
    assert tensor.wavelet_filter_passes == 1
    assert tensor.updates_since_filter == 0
    # Real signals survive denoising; the noise-level key is pruned away.
    assert strong in tensor.values
    assert weak in tensor.values
    assert noise not in tensor.values
    assert pruned >= 1
    # Score remains elevated for the strong key after denoising.
    assert tensor.score(strong, time_epoch=1) > 0.1


def test_denoise_does_not_manufacture_facts():
    rng = np.random.default_rng(9)
    tensor = SparseHebbianTensor(key_shape=(8, 8), time_depth=8, baseline=0.1,
                                 max_weight=5.0)
    tensor.reward((0, 0), time_epoch=0, amount=2.0)
    for _ in range(5):
        tensor.wavelet_denoise(rng, noise_enabled=True)
        for key in tensor.values:
            # Only slots that were genuinely observed may hold weights.
            observed = tensor.epochs[key] >= 0
            assert np.all(tensor.values[key][~observed] == tensor.baseline)
    # Unobserved combinations never allocated.
    assert tensor.stored_combinations == 1


def test_shape_validation():
    for bad_shape in [(7,), (8, 3), (0,)]:
        try:
            SparseHebbianTensor(key_shape=bad_shape)
        except ValueError:
            pass
        else:
            raise AssertionError(f"expected ValueError for shape {bad_shape}")


def main():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"PASS {t.__name__}")
    print(f"\n{len(tests)} tests passed.")


if __name__ == "__main__":
    main()
