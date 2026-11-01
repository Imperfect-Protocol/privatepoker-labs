"""Tests for skynet.tensors.wavelet. Run: python tests/test_wavelet.py"""
from __future__ import annotations

import math
import sys
import zlib
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from skynet.tensors.wavelet import (
    haar_forward_axis, haar_inverse_axis, haar_forward_nd, haar_inverse_nd,
    prepare_wavelet_coefficients, prepare_and_compose_wavelet_tensor,
    pack_sparse_wavelet_coefficients, unpack_sparse_wavelet_coefficients,
)


def test_haar_axis_round_trip():
    rng = np.random.default_rng(7)
    for shape in [(8,), (8, 8), (8, 8, 8, 8, 8, 8)]:
        tensor = rng.normal(size=shape).astype(np.float32)
        for axis in range(len(shape)):
            coefficients = haar_forward_axis(tensor, axis)
            recovered = haar_inverse_axis(coefficients, axis)
            np.testing.assert_allclose(recovered, tensor, atol=1e-6)


def test_haar_axis_energy_and_lowpass():
    # Orthonormal: energy is preserved. The low band carries the DC mean.
    tensor = np.full((8,), 3.0, dtype=np.float32)
    coefficients = haar_forward_axis(tensor, 0)
    np.testing.assert_allclose(coefficients[:4], np.full(4, 3.0 * math.sqrt(2.0)), atol=1e-6)
    np.testing.assert_allclose(coefficients[4:], 0.0, atol=1e-6)
    total_energy = float(np.sum(coefficients.astype(np.float64) ** 2))
    assert abs(total_energy - float(np.sum(tensor.astype(np.float64) ** 2))) < 1e-3


def test_haar_nd_round_trip():
    rng = np.random.default_rng(11)
    for shape in [(8,), (4, 8), (8, 8, 8), (8, 8, 8, 8, 8, 8)]:
        tensor = rng.normal(size=shape).astype(np.float32)
        recovered = haar_inverse_nd(haar_forward_nd(tensor))
        np.testing.assert_allclose(recovered, tensor, atol=1e-5)


def test_prepare_wavelet_thresholds():
    rng = np.random.default_rng(3)
    tensor = np.zeros((8, 8, 8, 8, 8, 8), dtype=np.float32)
    tensor[3, 3, 3, 3, 3, 3] = 5.0   # one big spike
    coefficients = prepare_wavelet_coefficients(
        tensor, rng, noise_enabled=False, gaussian_enabled=False, threshold=0.1)
    # The spike smears across all axes; small coefficients are zeroed.
    nonzero = np.count_nonzero(coefficients)
    assert 0 < nonzero < coefficients.size
    # Big structure survives thresholding.
    assert np.max(np.abs(coefficients)) > 0.1


def test_prepare_wavelet_switches():
    rng = np.random.default_rng(5)
    tensor = rng.normal(size=(8, 8)).astype(np.float32)
    # Haar off: returns the (noisy) input unchanged, no thresholding.
    out = prepare_wavelet_coefficients(tensor, rng, noise_enabled=False,
                                       haar_enabled=False, threshold=1.0)
    np.testing.assert_allclose(out, tensor, atol=1e-6)
    # Noise on, Haar off: output differs from input by roughly noise scale.
    out_noisy = prepare_wavelet_coefficients(tensor, rng, noise_std=0.01,
                                             noise_enabled=True, haar_enabled=False)
    assert np.max(np.abs(out_noisy - tensor)) > 0.0
    # Power-of-two validation.
    try:
        prepare_wavelet_coefficients(np.zeros((7, 8)), rng)
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for non-power-of-two axis")


def test_prepare_and_compose():
    rng = np.random.default_rng(13)
    tensor = np.zeros((8, 8, 8), dtype=np.float32)
    tensor[4, 4, 4] = 2.0
    composed, count = prepare_and_compose_wavelet_tensor(
        tensor, rng, noise_enabled=False, gaussian_enabled=False, threshold=0.05)
    assert count > 0
    # Composition approximately recovers the spike's structure.
    assert abs(float(np.max(composed)) - 2.0) < 0.5


def test_pack_unpack_round_trip():
    rng = np.random.default_rng(17)
    shape = (8, 8, 8)
    coefficients = (rng.normal(size=shape) > 1.5).astype(np.float32) * \
        rng.normal(size=shape).astype(np.float32)
    payload, count = pack_sparse_wavelet_coefficients(coefficients)
    recovered = unpack_sparse_wavelet_coefficients(payload, shape)
    assert count == int(np.count_nonzero(coefficients))
    np.testing.assert_allclose(recovered, coefficients, atol=1e-7)

    # Sparse payload must actually be smaller than the dense tensor.
    assert len(payload) < coefficients.nbytes / 4

    # Corruption detection.
    for bad_payload in [b"", payload[:3], payload[:-1], payload + b"x"]:
        try:
            unpack_sparse_wavelet_coefficients(bad_payload, shape)
        except (ValueError, zlib.error):
            pass
        else:
            raise AssertionError("expected failure on truncated payload")


def main():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"PASS {t.__name__}")
    print(f"\n{len(tests)} tests passed.")


if __name__ == "__main__":
    main()
