"""Wavelet pipeline: noise -> Haar -> Gaussian -> soft-threshold, plus packing.

Ported from p2p_nerual_spatial/simulator_sp.py, CPU-only (NumPy + SciPy),
and made shape-generic: every function takes the tensor it works on and the
axes it needs, with no hard-coded 8^6 shape. This is the skynet "chaos"
pipeline — the domain-randomization bridge between simulated and real
perception.

Lossless protocol metadata is never passed through here; only fuzzy value
tensors are. Coefficients pack to sparse zlib/DEFLATE payloads for the wire.
"""
from __future__ import annotations

import math
import struct
import zlib
from typing import Tuple

import numpy as np
from scipy.ndimage import gaussian_filter

WAVELET_NOISE_STD = 0.00001
WAVELET_GAUSSIAN_SIGMA = 0.30
WAVELET_THRESHOLD = 0.00005
MESSAGE_WAVELET_THRESHOLD = 0.006
NODE_TENSOR_PRUNE_THRESHOLD = 0.002


def haar_forward_axis(array: np.ndarray, axis: int) -> np.ndarray:
    """One-level orthonormal Haar transform along one power-of-two axis."""
    source = np.asarray(array, dtype=np.float32)
    axis = int(axis) % source.ndim
    length = source.shape[axis]
    if length < 1 or (length & (length - 1)) != 0:
        raise ValueError("Haar axis lengths must be powers of two")
    if length == 1:
        return source.copy()
    moved = np.moveaxis(source, axis, -1)
    inv_sqrt2 = np.float32(1.0 / math.sqrt(2.0))
    even = moved[..., 0::2]
    odd = moved[..., 1::2]
    out = np.empty_like(moved)
    half = length // 2
    out[..., :half] = (even + odd) * inv_sqrt2
    out[..., half:] = (even - odd) * inv_sqrt2
    return np.moveaxis(out, -1, axis)


def haar_inverse_axis(array: np.ndarray, axis: int) -> np.ndarray:
    """Invert the one-level Haar transform along one axis."""
    source = np.asarray(array, dtype=np.float32)
    axis = int(axis) % source.ndim
    length = source.shape[axis]
    if length < 1 or (length & (length - 1)) != 0:
        raise ValueError("Haar axis lengths must be powers of two")
    if length == 1:
        return source.copy()
    moved = np.moveaxis(source, axis, -1)
    inv_sqrt2 = np.float32(1.0 / math.sqrt(2.0))
    half = length // 2
    low = moved[..., :half]
    high = moved[..., half:]
    out = np.empty_like(moved)
    out[..., 0::2] = (low + high) * inv_sqrt2
    out[..., 1::2] = (low - high) * inv_sqrt2
    return np.moveaxis(out, -1, axis)


def haar_forward_nd(array: np.ndarray) -> np.ndarray:
    """Separable Haar decomposition across every axis."""
    result = np.asarray(array, dtype=np.float32)
    if result.ndim == 0:
        return result.copy()
    for axis in range(result.ndim):
        result = haar_forward_axis(result, axis)
    return result


def haar_inverse_nd(coefficients: np.ndarray) -> np.ndarray:
    """Compose an N-D tensor by inverting every axis in reverse order."""
    result = np.asarray(coefficients, dtype=np.float32)
    if result.ndim == 0:
        return result.copy()
    for axis in reversed(range(result.ndim)):
        result = haar_inverse_axis(result, axis)
    return result


def prepare_wavelet_coefficients(tensor: np.ndarray, rng: np.random.Generator, *,
                                 noise_std: float = WAVELET_NOISE_STD,
                                 noise_enabled: bool = True,
                                 haar_enabled: bool = True,
                                 gaussian_enabled: bool = True,
                                 gaussian_sigma: float = WAVELET_GAUSSIAN_SIGMA,
                                 threshold: float = WAVELET_THRESHOLD) -> np.ndarray:
    """Apply noise -> all-axis Haar -> Gaussian -> soft-threshold processing.

    With Haar disabled the noisy tensor is returned directly and Gaussian
    filtering plus thresholding are skipped (they live in coefficient space).
    """
    source = np.asarray(tensor, dtype=np.float32)
    if any((dim & (dim - 1)) != 0 for dim in source.shape):
        raise ValueError("all tensor dimensions must be powers of two for Haar")
    if noise_enabled and float(noise_std) > 0.0:
        processed = source + rng.normal(0.0, float(noise_std), size=source.shape).astype(np.float32)
    else:
        processed = source.copy()
    if not haar_enabled:
        return processed.astype(np.float32, copy=False)
    coefficients = haar_forward_nd(processed)
    if gaussian_enabled and gaussian_sigma > 0.0:
        coefficients = gaussian_filter(
            coefficients, sigma=(float(gaussian_sigma),) * coefficients.ndim,
            mode="nearest",
        ).astype(np.float32, copy=False)
    cutoff = float(threshold)
    if cutoff > 0.0:
        coefficients = np.sign(coefficients) * np.maximum(np.abs(coefficients) - cutoff, 0.0)
    coefficients[np.abs(coefficients) < 1e-8] = 0.0
    return coefficients.astype(np.float32, copy=False)


def prepare_and_compose_wavelet_tensor(tensor: np.ndarray, rng: np.random.Generator, *,
                                       noise_std: float = WAVELET_NOISE_STD,
                                       noise_enabled: bool = True,
                                       haar_enabled: bool = True,
                                       gaussian_enabled: bool = True,
                                       gaussian_sigma: float = WAVELET_GAUSSIAN_SIGMA,
                                       threshold: float = WAVELET_THRESHOLD) -> Tuple[np.ndarray, int]:
    """Filter a tensor and inverse-compose it. Returns (composed, nonzero_coefficients)."""
    source = np.asarray(tensor, dtype=np.float32)
    coefficients = prepare_wavelet_coefficients(
        source, rng, noise_std=noise_std, noise_enabled=noise_enabled,
        haar_enabled=haar_enabled, gaussian_enabled=gaussian_enabled,
        gaussian_sigma=gaussian_sigma, threshold=threshold,
    )
    count = int(np.count_nonzero(coefficients))
    composed = haar_inverse_nd(coefficients) if haar_enabled else coefficients
    return composed, count


def pack_sparse_wavelet_coefficients(coefficients: np.ndarray) -> Tuple[bytes, int]:
    """Losslessly DEFLATE sparse, thresholded Haar coefficients.

    The tensor shape is part of the protocol version; only the flat size is
    implied. Payload: uint32 nonzero count, uint32 flat indices, float32 values,
    then zlib/DEFLATE.
    """
    coeffs = np.asarray(coefficients, dtype=np.float32)
    flat = coeffs.ravel()
    indices = np.flatnonzero(flat).astype("<u4", copy=False)
    values = flat[indices].astype("<f4", copy=False)
    raw = struct.pack("<I", int(indices.size)) + indices.tobytes() + values.tobytes()
    return zlib.compress(raw, level=6), int(indices.size)


def unpack_sparse_wavelet_coefficients(payload: bytes,
                                       shape: Tuple[int, ...]) -> np.ndarray:
    """Decode a sparse DEFLATE-compressed coefficient payload into a dense array.

    Strict: trailing garbage or a truncated stream raises rather than silently
    decoding a partial tensor.
    """
    decomp = zlib.decompressobj()
    raw = decomp.decompress(payload)
    raw += decomp.flush()
    if not decomp.eof or decomp.unused_data:
        raise ValueError("wavelet coefficient payload is not a complete zlib stream")
    if len(raw) < 4:
        raise ValueError("truncated wavelet coefficient payload")
    (count,) = struct.unpack("<I", raw[:4])
    expected_len = 4 + count * 8
    if len(raw) != expected_len:
        raise ValueError("invalid wavelet coefficient payload length")
    indices = np.frombuffer(raw, dtype="<u4", count=count, offset=4)
    values_offset = 4 + count * 4
    values = np.frombuffer(raw, dtype="<f4", count=count, offset=values_offset)
    size = int(np.prod(shape))
    if count and int(indices.max()) >= size:
        raise ValueError("wavelet coefficient index is outside tensor shape")
    dense = np.zeros(size, dtype=np.float32)
    dense[indices] = values
    return dense.reshape(shape)
