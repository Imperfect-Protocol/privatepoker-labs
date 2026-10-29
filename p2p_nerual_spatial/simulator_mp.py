"""Event-driven P2P simulator with stun coalescing, learned routing, and bundled messages.

Timer, Message, and cancellable Fallback tasks are placed on the priority queue.
A Message is a transport envelope that may carry multiple Signal items (beacons,
announces, and reverse-path success feedback).
Node handlers never create outgoing Messages on receipt: they learn, coalesce the
received signal state, and arm the node's single timer. The timer handler chooses
routes and emits at most one bundled Message per destination neighbour.

Examples:
    python simulator.py --self-test
    python simulator.py --nodes 100
    python simulator.py --nodes 1000 --max-events 1500000
    python simulator.py --nodes 100 --ttl 8
    python simulator.py --no-learning --no-gaussian --threshold 0.01
    python simulator.py --device mps --nodes 100
    python simulator.py --event-log messages.jsonl
"""
from __future__ import annotations

import argparse
import hashlib
import heapq
import json
import math
import multiprocessing as mp
import random
import traceback
import struct
import tempfile
import time
import zlib
from dataclasses import dataclass, field, replace
from functools import lru_cache
from typing import Dict, Iterable, List, Optional, Tuple, Union

import numpy as np
from scipy.ndimage import gaussian_filter, map_coordinates

try:
    import torch
except ImportError:  # NumPy/SciPy remain a fully supported CPU fallback.
    torch = None


# Dense embedding transforms and the Haar/Gaussian/inverse-Haar tensor pipeline
# can use NVIDIA CUDA or Apple Metal Performance Shaders (MPS). The event queue,
# sparse Hebbian tables, and zlib serialization remain CPU-based. Small embedding
# batches stay on CPU to avoid accelerator launch and host/device transfer overhead.
_EMBEDDING_DEVICE = "cpu"
_ACCELERATOR_BATCH_THRESHOLD = 8


def _cuda_available() -> bool:
    """Return whether this PyTorch installation can use a CUDA device."""
    try:
        return bool(torch is not None and torch.cuda.is_available())
    except (AttributeError, RuntimeError):
        return False


def _mps_available() -> bool:
    """Return whether this PyTorch installation can use Apple's MPS backend."""
    try:
        return bool(
            torch is not None
            and hasattr(torch, "backends")
            and hasattr(torch.backends, "mps")
            and torch.backends.mps.is_available()
        )
    except (AttributeError, RuntimeError):
        return False


def available_embedding_devices() -> Tuple[str, ...]:
    """List supported embedding-transform devices in selection preference order."""
    devices = ["cpu"]
    if _mps_available():
        devices.insert(0, "mps")
    if _cuda_available():
        devices.insert(0, "cuda")
    return tuple(devices)


def configure_embedding_device(preference: str = "auto") -> str:
    """Select auto, CPU, NVIDIA CUDA, or Apple MPS for batched embedding transforms.

    ``auto`` prefers CUDA when available, then MPS, then CPU. Explicit accelerator
    requests fail with a clear error instead of silently running on the CPU.
    """
    global _EMBEDDING_DEVICE
    preference = str(preference).lower()
    if preference not in {"auto", "cpu", "cuda", "mps"}:
        raise ValueError("embedding device must be one of: auto, cpu, cuda, mps")

    cuda_available = _cuda_available()
    mps_available = _mps_available()
    if preference == "cuda" and not cuda_available:
        if torch is None:
            raise RuntimeError("CUDA was requested, but PyTorch is not installed")
        raise RuntimeError("CUDA was requested, but PyTorch reports no CUDA-capable GPU")
    if preference == "mps" and not mps_available:
        if torch is None:
            raise RuntimeError("MPS was requested, but PyTorch is not installed")
        raise RuntimeError(
            "MPS was requested, but torch.backends.mps.is_available() is false. "
            "Use an MPS-enabled PyTorch build on a supported Apple Mac, or choose --device cpu."
        )

    if preference == "auto":
        _EMBEDDING_DEVICE = "cuda" if cuda_available else ("mps" if mps_available else "cpu")
    else:
        _EMBEDDING_DEVICE = preference
    return _EMBEDDING_DEVICE


def embedding_backend_name() -> str:
    """Describe the selected backend and make the small-batch CPU path explicit."""
    if _EMBEDDING_DEVICE == "cuda":
        try:
            device_name = torch.cuda.get_device_name(0)
        except (AttributeError, RuntimeError):
            device_name = "CUDA device"
        return (f"PyTorch CUDA / {device_name} (embedding batches >= {_ACCELERATOR_BATCH_THRESHOLD}; "
                "Haar + Gaussian + inverse Haar on GPU; zlib serialization on CPU)")
    if _EMBEDDING_DEVICE == "mps":
        return (f"PyTorch MPS / Apple Metal GPU (embedding batches >= {_ACCELERATOR_BATCH_THRESHOLD}; "
                "Haar + Gaussian + inverse Haar on GPU; zlib serialization on CPU)")
    return "NumPy cubic CPU (CPU selected or no supported PyTorch accelerator available)"


# ---------------------------------------------------------------------------
# Fixed history embeddings: Angle x Hop (8 x 8), newest hop at column 0.
# ---------------------------------------------------------------------------

ANGLE_BINS = 8
HOP_BINS = 8
EMBEDDING_SHAPE = (ANGLE_BINS, HOP_BINS)
TIME_DEPTH = 8
DEFAULT_TTL = 8
DEFAULT_MAX_RADIUS = 150.0  # retained as a compatibility/config label; hops are unit length

# Logical node/message tensor:
# Source(angle, hop) x Target(angle, hop) x Neighbour(angle, 1) x Time.
# The singleton neighbour-hop axis is implicit, so the dense NumPy shape is 8^6.
LOGICAL_TENSOR_SHAPE = (8, 8, 8, 8, 8, TIME_DEPTH)
WAVELET_NOISE_STD = 0.00001
WAVELET_GAUSSIAN_SIGMA = 0.30
WAVELET_THRESHOLD = 0.00005
MESSAGE_WAVELET_THRESHOLD = 0.006
DEFAULT_HEBBIAN_REWARD = 0.35
NODE_WAVELET_EVERY_N_TIMERS = 32
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


def _accelerator_tensor_device():
    """Return the selected CUDA/MPS torch device, or None for the NumPy CPU path."""
    if torch is None or _EMBEDDING_DEVICE not in {"cuda", "mps"}:
        return None
    return torch.device(_EMBEDDING_DEVICE)


def _haar_forward_axis_torch(tensor, axis: int):
    """One-level orthonormal Haar transform on a torch tensor without leaving device."""
    axis = int(axis) % tensor.ndim
    length = int(tensor.shape[axis])
    if length < 1 or (length & (length - 1)) != 0:
        raise ValueError("Haar axis lengths must be powers of two")
    if length == 1:
        return tensor.clone()
    moved = tensor.movedim(axis, -1)
    even = moved[..., 0::2]
    odd = moved[..., 1::2]
    scale = 1.0 / math.sqrt(2.0)
    transformed = torch.cat(((even + odd) * scale, (even - odd) * scale), dim=-1)
    return transformed.movedim(-1, axis).contiguous()


def _haar_inverse_axis_torch(tensor, axis: int):
    """Inverse one-level Haar transform on a torch tensor without leaving device."""
    axis = int(axis) % tensor.ndim
    length = int(tensor.shape[axis])
    if length < 1 or (length & (length - 1)) != 0:
        raise ValueError("Haar axis lengths must be powers of two")
    if length == 1:
        return tensor.clone()
    moved = tensor.movedim(axis, -1)
    half = length // 2
    low, high = moved[..., :half], moved[..., half:]
    scale = 1.0 / math.sqrt(2.0)
    composed = torch.stack(((low + high) * scale, (low - high) * scale), dim=-1)
    composed = composed.flatten(start_dim=-2)
    return composed.movedim(-1, axis).contiguous()


def _haar_forward_nd_torch(tensor):
    result = tensor.to(dtype=torch.float32)
    for axis in range(result.ndim):
        result = _haar_forward_axis_torch(result, axis)
    return result


def _haar_inverse_nd_torch(tensor):
    result = tensor.to(dtype=torch.float32)
    for axis in reversed(range(result.ndim)):
        result = _haar_inverse_axis_torch(result, axis)
    return result


def _torch_gaussian_filter_nd(tensor, sigma: float):
    """Separable Gaussian filter with replicate/nearest boundary handling on-device.

    This is a PyTorch implementation of the separable filtering used by SciPy's
    gaussian_filter. Coefficients stay on the selected accelerator for every axis.
    """
    sigma = float(sigma)
    if sigma <= 0.0 or tensor.ndim == 0:
        return tensor
    # SciPy's default truncate is 4.0 and radius is rounded from truncate*sigma.
    radius = int(4.0 * sigma + 0.5)
    if radius < 1:
        return tensor
    device, dtype = tensor.device, tensor.dtype
    positions = torch.arange(-radius, radius + 1, device=device, dtype=dtype)
    kernel = torch.exp(-0.5 * (positions / sigma).square())
    kernel = kernel / kernel.sum()
    result = tensor
    import torch.nn.functional as torch_f
    for axis in range(result.ndim):
        moved = result.movedim(axis, -1).contiguous()
        original_shape = moved.shape
        length = int(original_shape[-1])
        rows = moved.reshape(-1, 1, length)
        rows = torch_f.pad(rows, (radius, radius), mode="replicate")
        rows = torch_f.conv1d(rows, kernel.reshape(1, 1, -1))
        moved = rows.reshape(*original_shape)
        result = moved.movedim(-1, axis).contiguous()
    return result


def _prepare_wavelet_coefficients_torch(tensor: np.ndarray, rng: np.random.Generator, *,
                                        noise_std: float, noise_enabled: bool,
                                        haar_enabled: bool, gaussian_enabled: bool,
                                        gaussian_sigma: float, threshold: float,
                                        device):
    """Noise on input, then Haar/Gaussian/threshold entirely on selected device."""
    source = np.asarray(tensor, dtype=np.float32)
    if noise_enabled and float(noise_std) > 0.0:
        # Noise generation remains CPU-side, then one transfer begins the GPU pipeline.
        source = source + rng.normal(0.0, float(noise_std), size=source.shape).astype(np.float32)
    with torch.no_grad():
        processed = torch.as_tensor(np.ascontiguousarray(source), dtype=torch.float32, device=device)
        if not haar_enabled:
            return processed
        coefficients = _haar_forward_nd_torch(processed)
        if gaussian_enabled and float(gaussian_sigma) > 0.0:
            coefficients = _torch_gaussian_filter_nd(coefficients, float(gaussian_sigma))
        cutoff = float(threshold)
        if cutoff > 0.0:
            coefficients = torch.sign(coefficients) * torch.clamp(torch.abs(coefficients) - cutoff, min=0.0)
        coefficients = coefficients.masked_fill(torch.abs(coefficients) < 1e-8, 0.0)
        return coefficients


def haar_forward_nd(array: np.ndarray) -> np.ndarray:
    """Apply separable Haar decomposition across every axis.

    On a selected CUDA/MPS device, all axes are transformed on that device and a
    NumPy array is returned only at the serialization/testing boundary.
    """
    device = _accelerator_tensor_device()
    if device is not None:
        with torch.no_grad():
            source = torch.as_tensor(np.ascontiguousarray(array), dtype=torch.float32, device=device)
            return _haar_forward_nd_torch(source).detach().cpu().numpy().astype(np.float32, copy=False)
    result = np.asarray(array, dtype=np.float32)
    if result.ndim == 0:
        return result.copy()
    for axis in range(result.ndim):
        result = haar_forward_axis(result, axis)
    return result


def haar_inverse_nd(coefficients: np.ndarray) -> np.ndarray:
    """Compose an N-D tensor, using CUDA/MPS for all inverse axes when selected."""
    device = _accelerator_tensor_device()
    if device is not None:
        with torch.no_grad():
            source = torch.as_tensor(np.ascontiguousarray(coefficients), dtype=torch.float32, device=device)
            return _haar_inverse_nd_torch(source).detach().cpu().numpy().astype(np.float32, copy=False)
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

    For CUDA/MPS, Haar, Gaussian filtering, and coefficient thresholding all run
    on-device in one pass. The only device-to-host transfer is the final coefficient
    array, needed by the existing CPU zlib/sparse serializer. With Haar disabled,
    the noisy tensor is returned directly and Gaussian/thresholding are skipped.
    """
    source = np.asarray(tensor, dtype=np.float32)
    if any((dim & (dim - 1)) != 0 for dim in source.shape):
        raise ValueError("all tensor dimensions must be powers of two for Haar")
    device = _accelerator_tensor_device()
    if device is not None:
        coefficients = _prepare_wavelet_coefficients_torch(
            source, rng, noise_std=noise_std, noise_enabled=noise_enabled,
            haar_enabled=haar_enabled, gaussian_enabled=gaussian_enabled,
            gaussian_sigma=gaussian_sigma, threshold=threshold, device=device,
        )
        return coefficients.detach().cpu().numpy().astype(np.float32, copy=False)

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
    """Filter and inverse-compose a node tensor with no intermediate host round-trip on GPU."""
    source = np.asarray(tensor, dtype=np.float32)
    device = _accelerator_tensor_device()
    if device is not None:
        coefficients = _prepare_wavelet_coefficients_torch(
            source, rng, noise_std=noise_std, noise_enabled=noise_enabled,
            haar_enabled=haar_enabled, gaussian_enabled=gaussian_enabled,
            gaussian_sigma=gaussian_sigma, threshold=threshold, device=device,
        )
        count = int(torch.count_nonzero(coefficients).item())
        if haar_enabled:
            with torch.no_grad():
                composed = _haar_inverse_nd_torch(coefficients)
        else:
            composed = coefficients
        return composed.detach().cpu().numpy().astype(np.float32, copy=False), count
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

    The fixed tensor shape is part of the protocol version. The payload consists
    of a uint32 nonzero count followed by uint32 flat indices and float32 values,
    then zlib/DEFLATE (LZ77 + Huffman coding).
    """
    coeffs = np.asarray(coefficients, dtype=np.float32)
    flat = coeffs.ravel()
    indices = np.flatnonzero(flat).astype("<u4", copy=False)
    values = flat[indices].astype("<f4", copy=False)
    raw = struct.pack("<I", int(indices.size)) + indices.tobytes() + values.tobytes()
    return zlib.compress(raw, level=6), int(indices.size)


def unpack_sparse_wavelet_coefficients(payload: bytes,
                                       shape: Tuple[int, ...] = LOGICAL_TENSOR_SHAPE) -> np.ndarray:
    """Decode a sparse DEFLATE-compressed coefficient payload into a dense array."""
    raw = zlib.decompress(payload)
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


def compress_embedding_sidecar(embeddings: Iterable[np.ndarray]) -> bytes:
    """Losslessly compress per-signal 8x8 embeddings alongside exact metadata.

    The aggregate message tensor intentionally merges similar signals, so a small
    compressed sidecar preserves each individual fuzzy embedding for forwarding.
    Signal identities, TTLs and addresses remain in their separate metadata rows.
    """
    items = list(embeddings)
    if not items:
        return b""
    stack = np.stack([resize_embedding_cubic(x, EMBEDDING_SHAPE) for x in items]).astype("<f4")
    header = struct.pack("<I", len(items))
    return zlib.compress(header + stack.tobytes(), level=6)


def decompress_embedding_sidecar(payload: bytes, count: int) -> np.ndarray:
    """Restore per-signal embeddings from the lossless compressed sidecar."""
    if count == 0:
        return np.empty((0, *EMBEDDING_SHAPE), dtype=np.float32)
    raw = zlib.decompress(payload)
    if len(raw) < 4 or struct.unpack("<I", raw[:4])[0] != count:
        raise ValueError("embedding sidecar count does not match signal metadata")
    expected = 4 + count * int(np.prod(EMBEDDING_SHAPE)) * 4
    if len(raw) != expected:
        raise ValueError("invalid embedding sidecar length")
    return np.frombuffer(raw, dtype="<f4", offset=4).astype(np.float32, copy=True).reshape(
        count, *EMBEDDING_SHAPE
    )


def address_for_node(node_id: int) -> bytes:
    """Return a deterministic, exact 20-byte simulated Ethereum-style address."""
    raw_id = int(node_id).to_bytes(8, byteorder="big", signed=False)
    return hashlib.sha256(b"simulated-eth-address:" + raw_id).digest()[-20:]


def hash_to_2d_point(address: object) -> np.ndarray:
    """Map an address into a synthetic plane for fuzzy directional hints only.

    For simulation nodes, ``address`` is the full 20-byte identity. The resulting
    point is deterministic but deliberately has no claim to graph-topology meaning.
    """
    if isinstance(address, bytes):
        payload = address
    elif isinstance(address, bytearray):
        payload = bytes(address)
    elif isinstance(address, int):
        payload = address_for_node(address)
    else:
        payload = str(address).encode("utf-8")
    if len(payload) == 20:
        digest = hashlib.sha256(payload).hexdigest()
    else:
        digest = hashlib.sha256(payload).hexdigest()
    x = (int(digest[:8], 16) % 10000) / 100.0
    y = (int(digest[8:16], 16) % 10000) / 100.0
    return np.array([x, y], dtype=np.float64)


def angle_bin(delta: np.ndarray, bins: int = ANGLE_BINS) -> int:
    """Quantize a Cartesian direction into a circular angle bin."""
    delta = np.asarray(delta, dtype=np.float64)
    if float(np.linalg.norm(delta)) < 1e-12:
        return 0
    # Embedding convention: theta=0 points up (+Y), positive angles rotate clockwise.
    angle = math.atan2(float(delta[0]), float(delta[1])) % (2.0 * math.pi)
    return int((angle / (2.0 * math.pi)) * bins) % bins


def fuzzy_angle_weights(delta: np.ndarray, bins: int = ANGLE_BINS) -> List[Tuple[int, float]]:
    """Return a softly interpolated pair of adjacent circular angle bins."""
    delta = np.asarray(delta, dtype=np.float64)
    if float(np.linalg.norm(delta)) < 1e-12:
        return [(0, 1.0)]
    # Same coordinate convention as path embeddings: zero is up, positive clockwise.
    angle = math.atan2(float(delta[0]), float(delta[1])) % (2.0 * math.pi)
    coordinate = angle * bins / (2.0 * math.pi)
    lower_float = math.floor(coordinate)
    lower = int(lower_float) % bins
    fraction = coordinate - lower_float
    upper = (lower + 1) % bins
    if fraction < 1e-8:
        return [(lower, 1.0)]
    return [(lower, 1.0 - fraction), (upper, fraction)]


def make_self_embedding(shape: Tuple[int, int] = EMBEDDING_SHAPE) -> np.ndarray:
    """A beacon begins with an up arrow at layer 0; the other seven layers are zero."""
    if len(shape) != 2 or any(int(n) < 1 for n in shape):
        raise ValueError("embedding shape must be angle x hop")
    result = np.zeros(shape, dtype=np.float32)
    result[0, 0] = 1.0
    return result


def resize_embedding_cubic(embedding: np.ndarray,
                           target_shape: Tuple[int, int] = EMBEDDING_SHAPE) -> np.ndarray:
    """Resize an angle x hop embedding with periodic cubic interpolation and blur.

    The angle axis wraps around at 2π. Legacy angle x distance x hop arrays are
    collapsed over distance, since the compact representation treats each hop as
    a unit step and intentionally has no distance axis.
    """
    source = np.asarray(embedding, dtype=np.float32)
    if source.ndim == 3:
        # Legacy layout: angle x distance x hop -> angle x hop.
        source = np.sum(source, axis=1, dtype=np.float32)
    if source.ndim != 2:
        raise ValueError("embedding must be angle x hop (or legacy angle x distance x hop)")
    if tuple(source.shape) == tuple(target_shape):
        return source.copy()
    source_angles, source_hops = source.shape
    target_angles, target_hops = target_shape
    angle_coords = np.arange(target_angles, dtype=np.float64) * source_angles / target_angles
    hop_coords = np.linspace(0, max(0, source_hops - 1), target_hops, dtype=np.float64)
    aa, hh = np.meshgrid(angle_coords, hop_coords, indexing="ij")
    # A two-bin periodic pad prevents the cubic filter from treating the angular seam
    # as a hard edge. Cubic interpolation may overshoot, so clamp before and after blur.
    pad = min(2, source_angles)
    padded = np.concatenate((source[-pad:], source, source[:pad]), axis=0)
    coords = np.vstack(((aa + pad).ravel(), hh.ravel()))
    result = map_coordinates(padded, coords, order=3, mode="nearest", prefilter=True)
    result = result.reshape(target_shape)
    result = np.maximum(result, 0.0)
    result = gaussian_filter(result, sigma=(0.35, 0.12), mode=("wrap", "nearest"))
    result = np.maximum(result, 0.0)
    old_mass = float(np.sum(source))
    new_mass = float(np.sum(result))
    if old_mass > 1e-12 and new_mass > 1e-12:
        result *= old_mass / new_mass
    return result.astype(np.float32, copy=False)


def _cubic_periodic_sample_numpy(profiles: np.ndarray, coordinates: np.ndarray) -> np.ndarray:
    """Periodic Catmull-Rom cubic sampling for profiles shaped [batch, angle]."""
    bins = profiles.shape[1]
    x = np.asarray(coordinates, dtype=np.float64)
    x0 = np.floor(x).astype(np.int64)
    t = x - x0
    t2, t3 = t * t, t * t * t
    weights = np.stack((
        -0.5 * t + t2 - 0.5 * t3,
        1.0 - 2.5 * t2 + 1.5 * t3,
        0.5 * t + 2.0 * t2 - 1.5 * t3,
        -0.5 * t2 + 0.5 * t3,
    ), axis=-1)
    indices = (x0[:, None] + np.array([-1, 0, 1, 2], dtype=np.int64)) % bins
    samples = profiles[:, indices]  # [batch, output-angle, 4]
    return np.sum(samples * weights[None, :, :], axis=-1, dtype=np.float32)


def _cubic_periodic_sample_torch(profiles, coordinates):
    """Periodic Catmull-Rom cubic sampling on the configured PyTorch device."""
    bins = profiles.shape[1]
    x = torch.as_tensor(coordinates, dtype=profiles.dtype, device=profiles.device)
    x0 = torch.floor(x).to(torch.long)
    t = x - x0.to(profiles.dtype)
    t2, t3 = t * t, t * t * t
    weights = torch.stack((
        -0.5 * t + t2 - 0.5 * t3,
        1.0 - 2.5 * t2 + 1.5 * t3,
        0.5 * t + 2.0 * t2 - 1.5 * t3,
        -0.5 * t2 + 0.5 * t3,
    ), dim=-1)
    offsets = torch.tensor([-1, 0, 1, 2], dtype=torch.long, device=profiles.device)
    indices = (x0[:, None] + offsets) % bins
    samples = profiles[:, indices]
    return torch.sum(samples * weights[None, :, :], dim=-1)


def normalize_embedding_layers(embedding: np.ndarray, depth: int,
                               *, fallback_direction: int = 0) -> np.ndarray:
    """Keep only occupied history layers and normalize each as a unit direction.

    Layer 0 is always the current node's canonical up arrow (angle bin 0).
    Layers at and beyond ``depth`` are exactly zero, so noise introduced for the
    wavelet codec can never invent path history in an empty layer.
    """
    result = np.maximum(np.asarray(embedding, dtype=np.float32), 0.0).copy()
    if result.shape != EMBEDDING_SHAPE:
        result = resize_embedding_cubic(result, EMBEDDING_SHAPE)
    depth = max(1, min(int(depth), HOP_BINS))
    result[:, depth:] = 0.0
    result[:, 0] = 0.0
    result[0, 0] = 1.0
    for layer in range(1, depth):
        profile = result[:, layer]
        mass = float(profile.sum())
        if mass > 1e-10:
            result[:, layer] = profile / mass
        else:
            # A layer declared occupied by exact metadata must still represent a
            # unit direction if thresholding removed all of its coefficients.
            result[:, layer] = 0.0
            result[int(fallback_direction) % ANGLE_BINS, layer] = 1.0
    return result.astype(np.float32, copy=False)


def _hop_rotation_radians(offset: np.ndarray) -> float:
    """Signed rotation from the canonical up direction to K->L, in radians.

    The embedding's angle 0 points up (+Y). Positive rotations move clockwise
    toward +X, hence atan2(dx, dy) rather than conventional atan2(dy, dx).
    """
    delta = np.asarray(offset, dtype=np.float64)
    if float(np.linalg.norm(delta)) <= 1e-12:
        return 0.0
    return float(math.atan2(float(delta[0]), float(delta[1])) % (2.0 * math.pi))


def _rotate_profiles_numpy(profiles: np.ndarray, angle_radians: float) -> np.ndarray:
    """Rotate [batch, angle] unit-direction distributions with periodic interpolation."""
    profiles = np.maximum(np.asarray(profiles, dtype=np.float32), 0.0)
    batch, angle_count = profiles.shape
    shift = (float(angle_radians) % (2.0 * math.pi)) * angle_count / (2.0 * math.pi)
    whole = int(math.floor(shift))
    fraction = float(shift - whole)
    # Angular bins start at up and increase clockwise. Bilinear circular binning
    # retains a continuous rotation while keeping the compact fuzzy representation.
    rotated = np.roll(profiles, whole, axis=1) * np.float32(1.0 - fraction)
    if fraction > 1e-12:
        rotated += np.roll(profiles, whole + 1, axis=1) * np.float32(fraction)
    old_mass = profiles.sum(axis=1, keepdims=True, dtype=np.float64)
    new_mass = rotated.sum(axis=1, keepdims=True, dtype=np.float64)
    scale = np.divide(old_mass, new_mass, out=np.zeros_like(old_mass), where=new_mass > 1e-12)
    return np.maximum(rotated * scale.astype(np.float32), 0.0).astype(np.float32, copy=False)


def _smooth_profiles_numpy(profiles: np.ndarray, sigma: float = 0.42) -> np.ndarray:
    """Apply the existing light circular Gaussian blur to inherited arrow layers."""
    if sigma <= 0.0:
        return np.maximum(profiles, 0.0).astype(np.float32, copy=False)
    taps = np.arange(-2, 3, dtype=np.float64)
    kernel = np.exp(-0.5 * (taps / float(sigma)) ** 2)
    kernel /= kernel.sum()
    smoothed = np.zeros_like(profiles, dtype=np.float32)
    for tap, weight in zip(taps.astype(int), kernel):
        smoothed += np.roll(profiles, int(tap), axis=1) * np.float32(weight)
    old_mass = profiles.sum(axis=1, keepdims=True, dtype=np.float64)
    new_mass = smoothed.sum(axis=1, keepdims=True, dtype=np.float64)
    scale = np.divide(old_mass, new_mass, out=np.zeros_like(old_mass), where=new_mass > 1e-12)
    return np.maximum(smoothed * scale.astype(np.float32), 0.0).astype(np.float32, copy=False)


def _translate_embedding_batch_numpy(sources: np.ndarray, offset: np.ndarray, twist: bool = True,
                                    gaussian_enabled: bool = True) -> np.ndarray:
    """Rotate every occupied history layer, shift it back one slot, then push up-arrow.

    Input and output are [batch, angle, hop]. The oldest column is dropped, every
    existing column rotates by the same K->L angle, and a new canonical unit-up arrow
    is inserted at hop 0. Empty layers remain bitwise zero.
    """
    sources = np.maximum(np.asarray(sources, dtype=np.float32), 0.0)
    batch, angle_count, hop_count = sources.shape
    if (angle_count, hop_count) != EMBEDDING_SHAPE:
        raise ValueError(f"expected embeddings with shape {EMBEDDING_SHAPE}")
    out = np.zeros_like(sources)
    phi = _hop_rotation_radians(offset) if twist else 0.0
    for old_hop in range(hop_count - 1):
        profile = sources[:, :, old_hop]
        mass = profile.sum(axis=1)
        active = mass > 1e-10
        if not np.any(active):
            continue
        if twist:
            rotated = _rotate_profiles_numpy(profile, phi)
            if gaussian_enabled:
                rotated = _smooth_profiles_numpy(rotated)
        else:
            rotated = profile.copy()
        rotated[~active] = 0.0
        out[:, :, old_hop + 1] = rotated
    # Fresh local frame: the receiver is always represented by a unit-up arrow.
    out[:, :, 0] = 0.0
    out[:, 0, 0] = 1.0
    # Empty incoming layers remain exactly zero after all transforms.
    out[np.abs(out) < 1e-8] = 0.0
    return out


def _translate_embedding_batch_torch(sources: np.ndarray, offset: np.ndarray,
                                     device_name: Optional[str] = None,
                                     twist: bool = True,
                                     gaussian_enabled: bool = True) -> np.ndarray:
    """PyTorch equivalent of the fixed eight-layer rotation/shift operation."""
    device = torch.device(device_name or _EMBEDDING_DEVICE)
    src_np = np.maximum(np.asarray(sources, dtype=np.float32), 0.0)
    batch, angle_count, hop_count = src_np.shape
    if (angle_count, hop_count) != EMBEDDING_SHAPE:
        raise ValueError(f"expected embeddings with shape {EMBEDDING_SHAPE}")
    src = torch.as_tensor(src_np, dtype=torch.float32, device=device)
    out = torch.zeros_like(src)
    phi = _hop_rotation_radians(offset) if twist else 0.0
    shift = (phi % (2.0 * math.pi)) * angle_count / (2.0 * math.pi)
    whole = int(math.floor(shift))
    fraction = float(shift - whole)
    for old_hop in range(hop_count - 1):
        profile = src[:, :, old_hop]
        mass = profile.sum(dim=1, keepdim=True)
        if not bool(torch.any(mass > 1e-10).item()):
            continue
        if twist:
            rotated = torch.roll(profile, shifts=whole, dims=1) * (1.0 - fraction)
            if fraction > 1e-12:
                rotated = rotated + torch.roll(profile, shifts=whole + 1, dims=1) * fraction
            rotated_mass = rotated.sum(dim=1, keepdim=True)
            rotated = torch.where(rotated_mass > 1e-12,
                                  rotated * mass / rotated_mass.clamp_min(1e-12), 0.0)
            if gaussian_enabled:
                taps = torch.arange(-2, 3, dtype=src.dtype, device=device)
                kernel = torch.exp(-0.5 * (taps / 0.42) ** 2)
                kernel = kernel / kernel.sum()
                smoothed = torch.zeros_like(rotated)
                for tap, weight in zip(range(-2, 3), kernel):
                    smoothed = smoothed + torch.roll(rotated, shifts=tap, dims=1) * weight
                smooth_mass = smoothed.sum(dim=1, keepdim=True)
                smoothed = torch.where(smooth_mass > 1e-12,
                                       smoothed * mass / smooth_mass.clamp_min(1e-12), 0.0)
            else:
                smoothed = rotated
        else:
            smoothed = profile
        out[:, :, old_hop + 1] = smoothed
    out[:, :, 0] = 0.0
    out[:, 0, 0] = 1.0
    out = torch.where(out.abs() < 1e-8, 0.0, out)
    return out.detach().cpu().numpy().astype(np.float32, copy=False)


def translate_embedding_batch(embeddings: Iterable[np.ndarray], source_to_target: np.ndarray,
                              max_radius: float = DEFAULT_MAX_RADIUS,
                              twist: bool = True,
                              gaussian_enabled: bool = True) -> np.ndarray:
    """Rotate all occupied layers by K->L, shift them down, and push a fresh up arrow.

    Embeddings are angle x hop (8 x 8). Angle bin zero points up and bins increase
    clockwise. Each graph hop rotates every inherited layer by the same signed edge
    angle, shifts it to the next hop slot, drops the oldest slot, and sets layer zero
    to the receiving node's canonical unit-up arrow. Empty layers stay zero.
    """
    batch = np.asarray(list(embeddings), dtype=np.float32)
    if batch.size == 0:
        return np.empty((0, *EMBEDDING_SHAPE), dtype=np.float32)
    if batch.ndim == 2:
        batch = batch[None, ...]
    if batch.ndim != 3:
        raise ValueError("embedding batch must have shape batch x angle x hop")
    if tuple(batch.shape[1:]) != EMBEDDING_SHAPE:
        batch = np.stack([resize_embedding_cubic(item, EMBEDDING_SHAPE) for item in batch])
    offset = np.asarray(source_to_target, dtype=np.float64)
    # Tiny packets are faster on CPU due to device launch and transfer overhead.
    if _EMBEDDING_DEVICE in {"cuda", "mps"} and len(batch) >= _ACCELERATOR_BATCH_THRESHOLD:
        return _translate_embedding_batch_torch(batch, offset, twist=twist, gaussian_enabled=gaussian_enabled)
    return _translate_embedding_batch_numpy(batch, offset, twist=twist, gaussian_enabled=gaussian_enabled)


def translate_embedding(embedding: np.ndarray, source_to_target: np.ndarray,
                        max_radius: float = DEFAULT_MAX_RADIUS,
                        twist: bool = True,
                        gaussian_enabled: bool = True) -> np.ndarray:
    """Single-embedding convenience wrapper around the batched translator."""
    return translate_embedding_batch([embedding], source_to_target, max_radius,
                                     twist=twist, gaussian_enabled=gaussian_enabled)[0]


class SparseHebbianTensor:
    """Sparse logical Source(8x8) x Target(8x8) x Neighbour(8x1) x Time(8).

    Keys are (source_angle, source_hop, target_angle, target_hop, neighbour_angle).
    Only observed combinations allocate an 8-slot time-history vector.
    """
    def __init__(self, time_depth: int = TIME_DEPTH, baseline: float = 0.1,
                 max_weight: float = 5.0, learning_enabled: bool = True) -> None:
        self.embedding_shape = EMBEDDING_SHAPE
        self.neighbour_shape = (ANGLE_BINS, 1)
        self.time_depth = int(time_depth)
        self.learning_enabled = bool(learning_enabled)
        # With learning disabled, represent a logical dense unit tensor implicitly.
        # score() returns 1.0 for every source/target/neighbour/time combination,
        # without allocating the prohibitively large dense tensor.
        self.baseline = float(baseline) if self.learning_enabled else 1.0
        self.max_weight = float(max_weight)
        self.values: Dict[Tuple[int, ...], np.ndarray] = {}
        self.epochs: Dict[Tuple[int, ...], np.ndarray] = {}
        self.reward_total = 0.0
        self.updates_since_filter = 0
        self.wavelet_filter_passes = 0
        self.wavelet_nonzero_coefficients_last = 0

    @property
    def logical_shape(self) -> Tuple[Tuple[int, int], Tuple[int, int], Tuple[int, int], int]:
        return (self.embedding_shape, self.embedding_shape, self.neighbour_shape, self.time_depth)

    @property
    def stored_combinations(self) -> int:
        return len(self.values)

    @property
    def stored_time_slots(self) -> int:
        return sum(int(np.count_nonzero(epochs >= 0)) for epochs in self.epochs.values())

    @staticmethod
    def _key(source: Tuple[int, int], target: Tuple[int, int],
             neighbour: int) -> Tuple[int, ...]:
        return tuple(map(int, (*source, *target, neighbour)))

    def reward(self, source: Tuple[int, int], target: Tuple[int, int],
               neighbour: int, time_epoch: int, amount: float) -> None:
        if not self.learning_enabled or amount <= 0.0:
            return
        key = self._key(source, target, neighbour)
        if key not in self.values:
            self.values[key] = np.full(self.time_depth, self.baseline, dtype=np.float32)
            self.epochs[key] = np.full(self.time_depth, -1, dtype=np.int64)
        slot = int(time_epoch) % self.time_depth
        if int(self.epochs[key][slot]) != int(time_epoch):
            self.values[key][slot] = self.baseline
            self.epochs[key][slot] = int(time_epoch)
        old = float(self.values[key][slot])
        new = min(self.max_weight, old + float(amount))
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
        """Denoise the logical 8^6 tensor, then project back to sparse storage.

        All six axes (source angle/hop, target angle/hop, neighbour angle, time)
        participate in the Haar transform. The composed tensor is projected onto
        observed sparse keys and existing time epochs so noise cannot manufacture
        unobserved protocol facts or turn the sparse table into a dense allocation.
        """
        if not self.learning_enabled or not self.values or self.updates_since_filter <= 0:
            return 0, 0
        dense = np.zeros(LOGICAL_TENSOR_SHAPE, dtype=np.float32)
        valid_by_key: Dict[Tuple[int, ...], np.ndarray] = {}
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

        # On CUDA/MPS, keep the tensor on-device through forward Haar, Gaussian
        # filtering, thresholding, and inverse Haar. Only the composed result is
        # copied back to CPU for projection into the sparse Python weight table.
        composed, coefficient_count = prepare_and_compose_wavelet_tensor(
            dense, rng, noise_std=noise_std, noise_enabled=noise_enabled,
            haar_enabled=haar_enabled, gaussian_enabled=gaussian_enabled,
            gaussian_sigma=gaussian_sigma, threshold=threshold,
        )

        removed_keys = []
        for key, valid in valid_by_key.items():
            filtered = np.clip(self.baseline + composed[key], self.baseline, self.max_weight)
            values = self.values[key]
            epochs = self.epochs[key]
            values[valid] = filtered[valid]
            # Soft-thresholded, near-baseline weights are removed rather than kept
            # as numerical noise. Their epoch slots become genuinely unobserved.
            prune = valid & ((values - self.baseline) < prune_threshold)
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

    def score(self, source: Tuple[int, int], target: Tuple[int, int],
              neighbour: int, time_epoch: int, decay: float = 0.97) -> float:
        if not self.learning_enabled:
            return 1.0
        key = self._key(source, target, neighbour)
        if key not in self.values:
            return self.baseline
        weights, epochs = self.values[key], self.epochs[key]
        delta = weights.astype(np.float64) - self.baseline
        valid = epochs >= 0
        if not np.any(valid):
            return self.baseline
        age = np.maximum(0, int(time_epoch) - epochs[valid])
        learned = float(np.sum(delta[valid] * np.power(float(decay), age)))
        return min(self.max_weight, self.baseline + learned)


def sigmoid_stunning_delay(message_count: int, k: float = 0.1, midpoint: float = 20.0,
                           minimum: float = 0.050, maximum: float = 0.500) -> float:
    """Sigmoidal stun duration, from 50 ms toward a 500 ms saturation limit."""
    if message_count < 1:
        raise ValueError("message_count must be >= 1")
    exponent = float(np.clip(-k * (message_count - midpoint), -60.0, 60.0))
    return minimum + (maximum - minimum) / (1.0 + math.exp(exponent))


# ---------------------------------------------------------------------------
# Event and protocol data structures
# ---------------------------------------------------------------------------

@dataclass(eq=False)
class Timer:
    """Callback-free timer event: dispatch timestamp and target node only."""
    dispatch_time: float
    target: "P2PNode"


@dataclass
class Signal:
    """Logical signal plus lossless metadata, carried in an aggregated Message.

    ``origin_address`` and ``target_address`` are exact 20-byte addresses. Tensor
    embeddings may be blurred/resampled, but these identity bytes and TTL are not.
    ``embedding_depth`` counts occupied history layers including the current node.
    ``route_path`` and ``return_path`` are internal node indices representing the
    forward path and the remaining reverse acknowledgement route.
    """
    kind: str
    embedding: np.ndarray
    origin_id: int
    sequence_id: int
    target_id: Optional[int] = None
    hop_count: int = 0
    visited: Tuple[int, ...] = ()
    coefficient: float = 1.0
    route_attempt: int = 0
    ttl: int = DEFAULT_TTL
    origin_address: bytes = b""
    target_address: Optional[bytes] = None
    route_path: Tuple[int, ...] = ()
    return_path: Tuple[int, ...] = ()
    embedding_depth: int = 1  # occupied history layers, including the current-node layer

    def __post_init__(self) -> None:
        if not self.origin_address:
            self.origin_address = address_for_node(self.origin_id)
        else:
            self.origin_address = bytes(self.origin_address)
        if len(self.origin_address) != 20:
            raise ValueError("origin_address metadata must be exactly 20 bytes")
        if self.target_id is not None:
            if self.target_address is None:
                self.target_address = address_for_node(self.target_id)
            else:
                self.target_address = bytes(self.target_address)
            if len(self.target_address) != 20:
                raise ValueError("target_address metadata must be exactly 20 bytes")
        elif self.target_address is not None:
            self.target_address = bytes(self.target_address)
            if len(self.target_address) != 20:
                raise ValueError("target_address metadata must be exactly 20 bytes when present")
        self.ttl = int(self.ttl)
        if self.ttl < 0:
            raise ValueError("signal TTL must be >= 0")
        self.route_path = tuple(int(n) for n in self.route_path)
        self.return_path = tuple(int(n) for n in self.return_path)
        self.embedding_depth = int(self.embedding_depth)
        if not 1 <= self.embedding_depth <= HOP_BINS:
            raise ValueError(f"embedding_depth must be in [1, {HOP_BINS}]")

    @property
    def signal_id(self) -> bytes:
        """Stable binary identity for this exact signal attempt (16 bytes)."""
        target = self.target_address or bytes(20)
        sequence = int(self.sequence_id).to_bytes(8, "big", signed=False)
        kind = self.kind.encode("ascii", errors="replace")[:16]
        return hashlib.sha256(self.origin_address + target + kind + sequence).digest()[:16]

    @property
    def metadata(self) -> dict:
        """Lossless per-signal protocol metadata, independent of fuzzy tensors."""
        return {
            "signal_id": self.signal_id,
            "type": self.kind,
            "origin_address": self.origin_address,
            "target_address": self.target_address,
            "sequence_id": self.sequence_id,
            "ttl": self.ttl,
            "hop_count": self.hop_count,
            "embedding_depth": self.embedding_depth,
            "visited_addresses": tuple(address_for_node(n) for n in self.visited),
            "route_path_addresses": tuple(address_for_node(n) for n in self.route_path),
            "return_path_addresses": tuple(address_for_node(n) for n in self.return_path),
        }

    @property
    def key(self) -> Tuple[object, ...]:
        # Retries of A->B coalesce semantically; return feedback is keyed by the
        # original request identity, while beacons are unique per origin sequence.
        if self.kind == "announce":
            return ("announce", self.origin_address, self.target_address)
        if self.kind == "feedback":
            return ("feedback", self.origin_address, self.target_address, self.sequence_id)
        return (self.kind, self.origin_address, self.sequence_id)


@dataclass(eq=False)
class Message:
    """Transport envelope: exact signal metadata plus one compressed embedding tensor.

    The message tensor is a stack of per-signal 8x8 angle×hop embeddings, padded to
    a power-of-two signal count. Signal metadata rows have the same order and retain
    exact addresses, signal IDs, TTLs and destinations. This preserves signal identity
    while allowing all embedding slices and the signal axis to be wavelet-compressed.
    """
    dispatch_time: float
    source_address: int  # simulator node index, not the serialized address
    target_address: int  # simulator node index, not the serialized address
    target: "P2PNode"
    items: List[Signal]
    kind: str = "bundle"
    join_from: Optional[int] = None
    join_to: Optional[int] = None
    sent_at: float = 0.0
    source_eth_address: bytes = b""
    target_eth_address: bytes = b""
    tensor_payload: bytes = b""
    tensor_shape: Tuple[int, ...] = (1, *EMBEDDING_SHAPE)
    tensor_codec: str = "sparse-haar+zlib"
    tensor_coefficient_count: int = 0
    tensor_uncompressed_bytes: int = 0
    decoded_items: Optional[List[Signal]] = field(default=None, repr=False)
    decoded_tensor: Optional[np.ndarray] = field(default=None, repr=False)

    def __post_init__(self) -> None:
        if not self.source_eth_address:
            self.source_eth_address = address_for_node(self.source_address)
        else:
            self.source_eth_address = bytes(self.source_eth_address)
        if not self.target_eth_address:
            self.target_eth_address = address_for_node(self.target_address)
        else:
            self.target_eth_address = bytes(self.target_eth_address)
        if len(self.source_eth_address) != 20 or len(self.target_eth_address) != 20:
            raise ValueError("message transport addresses must each be exactly 20 bytes")
        self.tensor_shape = tuple(int(x) for x in self.tensor_shape)
        if self.tensor_payload:
            if (len(self.tensor_shape) != 3 or self.tensor_shape[1:] != EMBEDDING_SHAPE
                    or self.tensor_shape[0] < len(self.items)
                    or (self.tensor_shape[0] & (self.tensor_shape[0] - 1)) != 0):
                raise ValueError(f"invalid message embedding tensor shape: {self.tensor_shape}")

    def decode_payloads(self) -> Tuple[List[Signal], Optional[np.ndarray]]:
        """Receiver path: decompress coefficients, inverse Haar, then restore slices."""
        if self.decoded_items is not None:
            return self.decoded_items, self.decoded_tensor
        if self.tensor_payload:
            if self.tensor_codec == "sparse-haar+zlib":
                coefficients = unpack_sparse_wavelet_coefficients(self.tensor_payload, self.tensor_shape)
                reconstructed = haar_inverse_nd(coefficients)
            elif self.tensor_codec == "haar-f32":
                expected = int(np.prod(self.tensor_shape)) * 4
                if len(self.tensor_payload) != expected:
                    raise ValueError("uncompressed Haar coefficient payload has incorrect length")
                coefficients = np.frombuffer(self.tensor_payload, dtype="<f4").copy().reshape(self.tensor_shape)
                reconstructed = haar_inverse_nd(coefficients)
            elif self.tensor_codec == "raw-f32":
                expected = int(np.prod(self.tensor_shape)) * 4
                if len(self.tensor_payload) != expected:
                    raise ValueError("raw tensor payload has incorrect length")
                reconstructed = np.frombuffer(self.tensor_payload, dtype="<f4").copy().reshape(self.tensor_shape)
            else:
                raise ValueError(f"unknown tensor codec: {self.tensor_codec}")
            reconstructed = np.maximum(reconstructed, 0.0).astype(np.float32, copy=False)
            restored = reconstructed[:len(self.items)].copy()
            for index, item in enumerate(self.items):
                restored[index] = normalize_embedding_layers(restored[index], item.embedding_depth)
                reconstructed[index] = restored[index]
            # Padding rows are not logical signals, so do not let wavelet noise create data there.
            if len(self.items) < reconstructed.shape[0]:
                reconstructed[len(self.items):] = 0.0
            self.decoded_tensor = reconstructed
            self.decoded_items = [replace(item, embedding=restored[i])
                                  for i, item in enumerate(self.items)]
        else:
            # Directly constructed local/test messages may already carry embeddings.
            self.decoded_items = [replace(item, embedding=np.asarray(item.embedding, dtype=np.float32).copy())
                                  for item in self.items]
            self.decoded_tensor = None
        return self.decoded_items, self.decoded_tensor

    @property
    def metadata(self) -> dict:
        return {
            "from_address": self.source_eth_address,
            "to_address": self.target_eth_address,
            "signals": [item.metadata for item in self.items],
            "tensor_shape": self.tensor_shape,
            "tensor_codec": self.tensor_codec,
            "tensor_payload_bytes": len(self.tensor_payload),
            "tensor_coefficient_count": self.tensor_coefficient_count,
        }


@dataclass(eq=False)
class Fallback:
    """Timeout task: if an announce has not formed a P2P edge, use mesh fallback."""
    dispatch_time: float
    left_id: int
    right_id: int
    cancelled: bool = False

    def cancel(self) -> None:
        self.cancelled = True


Event = Union[Timer, Message, Fallback]


# ---------------------------------------------------------------------------
# Simulation event loop and dynamic topology
# ---------------------------------------------------------------------------

class Simulation:
    def __init__(self, *, link_delay: float = 0.015, forwarding_threshold: float = 0.20,
                 coefficient_decay: float = 0.97, fallback_timeout: float = 5.0,
                 initial_ttl: int = DEFAULT_TTL, wavelet_seed: int = 42,
                 learning_enabled: bool = True, noise_enabled: bool = True,
                 haar_enabled: bool = True, gaussian_enabled: bool = True,
                 compression_enabled: bool = True, ttl_enabled: bool = True,
                 twist_enabled: bool = True, noise_strength: float = WAVELET_NOISE_STD,
                 gaussian_width: float = WAVELET_GAUSSIAN_SIGMA,
                 threshold: float = MESSAGE_WAVELET_THRESHOLD,
                 node_threshold: float = WAVELET_THRESHOLD,
                 hebbian_reward: float = DEFAULT_HEBBIAN_REWARD,
                 event_log_path: Optional[str] = None) -> None:
        if link_delay < 0:
            raise ValueError("link_delay must be non-negative")
        if fallback_timeout <= 0:
            raise ValueError("fallback_timeout must be positive")
        self.now = 0.0
        self.link_delay = float(link_delay)
        self.fallback_timeout = float(fallback_timeout)
        self.forwarding_threshold = float(forwarding_threshold)
        self.coefficient_decay = float(coefficient_decay)
        self.learning_enabled = bool(learning_enabled)
        self.noise_enabled = bool(noise_enabled)
        self.haar_enabled = bool(haar_enabled)
        self.gaussian_enabled = bool(gaussian_enabled) and self.haar_enabled
        self.compression_enabled = bool(compression_enabled) and self.haar_enabled
        self.ttl_enabled = bool(ttl_enabled)
        self.twist_enabled = bool(twist_enabled)
        self.noise_strength = float(noise_strength)
        self.gaussian_width = float(gaussian_width)
        self.wavelet_threshold = float(threshold)
        self.node_threshold = float(node_threshold)
        self.hebbian_reward = float(hebbian_reward)
        self.initial_ttl = int(initial_ttl)
        self.wavelet_rng = np.random.default_rng(int(wavelet_seed))
        if self.initial_ttl < 1 and self.ttl_enabled:
            raise ValueError("initial_ttl must be >= 1 when TTL is enabled")
        if self.noise_strength < 0 or self.gaussian_width < 0 or self.wavelet_threshold < 0 or self.node_threshold < 0:
            raise ValueError("noise strength, Gaussian width and thresholds must be >= 0")
        if self.hebbian_reward < 0:
            raise ValueError("Hebbian reward must be >= 0")
        self.event_log_path = event_log_path
        self._event_log = open(event_log_path, "w", encoding="utf-8") if event_log_path else None
        self._logged_messages = 0
        self._queue: List[Tuple[float, int, Event]] = []
        self._order = 0
        self.nodes: Dict[int, P2PNode] = {}

        self.timers_dispatched = 0
        self.messages_dispatched = 0       # protocol envelope count delivered
        self.messages_scheduled = 0        # protocol envelope count sent (loopbacks included)
        self.protocol_payload_items_scheduled = 0
        self.protocol_payload_items_dispatched = 0
        self.network_packets_scheduled = 0  # excludes local self-beacon loopbacks
        self.network_packets_delivered = 0
        self.network_signal_copies_scheduled = 0
        self.network_signal_copies_delivered = 0
        self.message_wavelet_encoded = 0
        self.message_wavelet_decoded = 0
        self.message_tensor_raw_bytes = 0
        self.message_tensor_compressed_bytes = 0
        self.message_tensor_reconstructed_mass = 0.0
        self.message_wavelet_coefficients = 0
        self.node_wavelet_filter_passes = 0
        self.node_wavelet_coefficients = 0
        self.node_wavelet_keys_pruned = 0
        self.signal_items_dropped_as_duplicates = 0
        self.signal_items_dropped_ttl = 0
        self.signal_items_dropped_hop_limit = 0
        self.feedback_signals_generated = 0
        self.success_feedbacks_delivered = 0
        self.successful_route_rewards = 0
        self.beacon_path_rewards = 0
        self.stale_timers_skipped = 0
        self.join_intents_scheduled = 0
        self.join_intents_dispatched = 0
        self.peak_queue_size = 0
        self.candidate_neighbor_checks = 0
        self.smart_signal_copies = 0
        self.fallback_ripple_signal_copies = 0
        self.feedback_signal_copies = 0
        self.no_route_signal_drops = 0

        self.join_case_counts = {
            "both_outside_direct": 0,
            "active_plus_outsider_direct": 0,
            "active_pair_announce": 0,
            "already_connected": 0,
            "invalid": 0,
        }
        self.announce_pairs_requested: set[Tuple[int, int]] = set()
        self.announce_pairs_discovered: set[Tuple[int, int]] = set()
        self.announce_directions_discovered: set[Tuple[int, int]] = set()
        self.announce_links_created = 0
        self.announce_links: List[Tuple[float, int, int]] = []
        self.pending_fallbacks: Dict[Tuple[int, int], Fallback] = {}
        self.fallback_tasks_scheduled = 0
        self.fallback_tasks_cancelled = 0
        self.fallback_tasks_fired = 0
        self.cancelled_fallbacks_skipped = 0
        self.fallback_links_created = 0
        self.fallback_links: List[Tuple[float, int, int]] = []

    def schedule(self, event: Event) -> None:
        dispatch = float(event.dispatch_time)
        if dispatch < self.now - 1e-12:
            raise ValueError(f"cannot schedule event in the past: {dispatch} < {self.now}")
        heapq.heappush(self._queue, (dispatch, self._order, event))
        self._order += 1
        self.peak_queue_size = max(self.peak_queue_size, len(self._queue))
        if isinstance(event, Message) and self._event_log is not None:
            self._write_event_log(event)

    def _write_event_log(self, message: Message) -> None:
        """Write one JSONL row per scheduled message, including the tensor as hex."""
        if self._event_log is None:
            return
        if message.tensor_payload:
            tensor_bytes = message.tensor_payload
            tensor_encoding = message.tensor_codec
        elif message.items:
            local_tensor = np.stack([
                resize_embedding_cubic(item.embedding, EMBEDDING_SHAPE)
                for item in message.items
            ]).astype("<f4", copy=False)
            tensor_bytes = local_tensor.tobytes()
            tensor_encoding = "local-raw-f32"
        else:
            tensor_bytes = b""
            tensor_encoding = "none"
        signals = []
        for item in message.items:
            signals.append({
                "signal_id": item.signal_id.hex(),
                "type": item.kind,
                "origin_address": item.origin_address.hex(),
                "target_address": item.target_address.hex() if item.target_address is not None else None,
                "sequence_id": item.sequence_id,
                "ttl": item.ttl,
                "hop_count": item.hop_count,
                "embedding_depth": item.embedding_depth,
                "route_attempt": item.route_attempt,
                "visited_addresses": [address_for_node(n).hex() for n in item.visited],
                "route_path_addresses": [address_for_node(n).hex() for n in item.route_path],
                "return_path_addresses": [address_for_node(n).hex() for n in item.return_path],
            })
        row = {
            "row": self._logged_messages,
            "event_type": message.kind,
            "sent_at": message.sent_at,
            "dispatch_time": message.dispatch_time,
            "from_node": message.source_address,
            "to_node": message.target_address,
            "from_address": message.source_eth_address.hex(),
            "to_address": message.target_eth_address.hex(),
            "signal_count": len(message.items),
            "signals": signals,
            "tensor_shape": list(message.tensor_shape),
            "tensor_encoding": tensor_encoding,
            "tensor_payload_bytes": len(tensor_bytes),
            "tensor_hex": tensor_bytes.hex(),
        }
        self._event_log.write(json.dumps(row, separators=(",", ":")) + "\n")
        self._logged_messages += 1

    def close_event_log(self) -> None:
        if self._event_log is not None:
            self._event_log.flush()
            self._event_log.close()
            self._event_log = None

    @property
    def pending_events(self) -> int:
        return int(getattr(self, "_parallel_pending_events", len(self._queue)))

    def run(self, *, until: Optional[float] = None, max_events: int = 750_000,
            progress_every: int = 0) -> int:
        """Dispatch events in priority-queue order until time or event budget."""
        if max_events < 1:
            raise ValueError("max_events must be positive")
        processed = 0
        while self._queue and processed < max_events:
            dispatch, _, event = self._queue[0]
            if until is not None and dispatch > until:
                break
            heapq.heappop(self._queue)
            if isinstance(event, Timer) and event.target.current_timer is not event:
                self.stale_timers_skipped += 1
                continue
            if isinstance(event, Fallback) and event.cancelled:
                self.cancelled_fallbacks_skipped += 1
                continue
            self.now = float(event.dispatch_time)
            processed += 1
            if progress_every > 0 and processed % progress_every == 0:
                print(
                    f"[progress] events={processed:,} sim_time={self.now:.2f}s "
                    f"queue={len(self._queue):,} network_packets={self.network_packets_scheduled:,}",
                    flush=True,
                )
            if isinstance(event, Timer):
                self.timers_dispatched += 1
                event.target.handle_timer(event)
            elif isinstance(event, Fallback):
                self.fallback_tasks_fired += 1
                self.handle_fallback(event)
            else:
                if event.kind == "join_intent":
                    self.join_intents_dispatched += 1
                    event.target.handle_incoming_message(event)
                else:
                    self.messages_dispatched += 1
                    self.protocol_payload_items_dispatched += len(event.items)
                    if event.target_address != event.source_address:
                        self.network_packets_delivered += 1
                        self.network_signal_copies_delivered += len(event.items)
                    event.target.handle_incoming_message(event)
        return processed

    def schedule_fallback(self, left_id: int, right_id: int) -> Fallback:
        """Schedule one cancellable mesh-server timeout for an unordered node pair."""
        pair = tuple(sorted((int(left_id), int(right_id))))
        existing = self.pending_fallbacks.get(pair)
        if existing is not None and not existing.cancelled:
            return existing
        fallback = Fallback(self.now + self.fallback_timeout, pair[0], pair[1])
        self.pending_fallbacks[pair] = fallback
        self.schedule(fallback)
        self.fallback_tasks_scheduled += 1
        return fallback

    def cancel_fallback(self, left_id: int, right_id: int) -> None:
        """Cancel an outstanding timeout once the P2P route discovers its peer."""
        pair = tuple(sorted((int(left_id), int(right_id))))
        fallback = self.pending_fallbacks.pop(pair, None)
        if fallback is not None and not fallback.cancelled:
            fallback.cancel()
            self.fallback_tasks_cancelled += 1

    def handle_fallback(self, fallback: Fallback) -> None:
        """Join two active but still-disconnected nodes through the mesh fallback."""
        pair = tuple(sorted((fallback.left_id, fallback.right_id)))
        if self.pending_fallbacks.get(pair) is fallback:
            self.pending_fallbacks.pop(pair, None)
        if fallback.left_id not in self.nodes or fallback.right_id not in self.nodes:
            return
        left, right = self.nodes[fallback.left_id], self.nodes[fallback.right_id]
        if right.id in left.neighbors:
            return
        # A fallback is only scheduled for active-active discovery requests.
        if not (left.active and right.active):
            return
        if connect_nodes(self.nodes, left.id, right.id):
            self.fallback_links_created += 1
            self.fallback_links.append((self.now, left.id, right.id))
            # Ensure both endpoints retain a timer that will produce their beacons.
            left.start_beacon_clock()
            right.start_beacon_clock()

    def handle_join_intent(self, left_id: int, right_id: int) -> None:
        """Apply one connection request at its event timestamp."""
        if left_id == right_id or left_id not in self.nodes or right_id not in self.nodes:
            self.join_case_counts["invalid"] += 1
            return
        left, right = self.nodes[left_id], self.nodes[right_id]
        left_active, right_active = left.active, right.active

        if not left_active and not right_active:
            if connect_nodes(self.nodes, left_id, right_id):
                self.join_case_counts["both_outside_direct"] += 1
                self.activate(left)
                self.activate(right)
            return
        if left_active != right_active:
            if connect_nodes(self.nodes, left_id, right_id):
                self.join_case_counts["active_plus_outsider_direct"] += 1
                self.activate(left)
                self.activate(right)
            return

        pair = tuple(sorted((left_id, right_id)))
        if right_id in left.neighbors:
            self.join_case_counts["already_connected"] += 1
            return
        self.join_case_counts["active_pair_announce"] += 1
        self.announce_pairs_requested.add(pair)
        left.request_announce(right_id)
        right.request_announce(left_id)
        self.schedule_fallback(left_id, right_id)

    def activate(self, node: "P2PNode") -> None:
        if node.active:
            return
        node.active = True
        node.start_beacon_clock()

    def resolve_announce_discovery(self, origin_id: int, target_id: int) -> None:
        """An active-active edge is created only when a real announce arrives."""
        if origin_id == target_id or origin_id not in self.nodes or target_id not in self.nodes:
            return
        pair = tuple(sorted((origin_id, target_id)))
        self.announce_directions_discovered.add((origin_id, target_id))
        self.announce_pairs_discovered.add(pair)
        origin, target = self.nodes[origin_id], self.nodes[target_id]
        if origin.active and target.active:
            if target_id in origin.neighbors:
                self.cancel_fallback(origin_id, target_id)
                return
            if connect_nodes(self.nodes, origin_id, target_id):
                self.cancel_fallback(origin_id, target_id)
                self.announce_links_created += 1
                self.announce_links.append((self.now, origin_id, target_id))


# ---------------------------------------------------------------------------
# Node: learn/coalesce on receive; route and bundle only on timer
# ---------------------------------------------------------------------------

class P2PNode:
    def __init__(self, node_id: int, sim: Simulation, nodes: Dict[int, "P2PNode"], *,
                 angle_bins: int = ANGLE_BINS, hop_bins: int = HOP_BINS,
                 tensor_bins: int = ANGLE_BINS, time_depth: int = TIME_DEPTH,
                 max_hops: int = 128, beacon_interval: float = 1.0,
                 max_radius: float = DEFAULT_MAX_RADIUS, max_announce_attempts: int = 2) -> None:
        self.id = int(node_id)
        self.sim = sim
        self.nodes = nodes
        self.address = address_for_node(self.id)
        self.pos = hash_to_2d_point(self.address)
        self.neighbors: set[int] = set()
        self.active = False

        if (angle_bins, hop_bins, tensor_bins) != (8, 8, 8):
            raise ValueError("the compact embedding is fixed at angle x hop = 8x8, with 8 neighbour angles")
        if int(time_depth) != TIME_DEPTH:
            raise ValueError(f"the logical tensor time depth is fixed at {TIME_DEPTH}")
        self.angle_bins = angle_bins
        self.hop_bins = hop_bins
        self.max_radius = float(max_radius)  # compatibility only; each graph hop is unit length
        self.tensor_bins = tensor_bins
        self.time_depth = time_depth
        self.max_hops = max_hops
        self.beacon_interval = float(beacon_interval)
        self.max_announce_attempts = int(max_announce_attempts)
        self.embedding = make_self_embedding(EMBEDDING_SHAPE)
        # Logical Source(8x8) x Target(8x8) x Neighbour(8x1) x Time(8),
        # stored sparsely so unobserved combinations cost no memory.
        self.tensor = SparseHebbianTensor(time_depth=time_depth, baseline=0.1,
                                          max_weight=5.0,
                                          learning_enabled=sim.learning_enabled)

        # Exactly one live timer slot; all nodes begin idle and unstunned.
        self.current_timer: Optional[Timer] = None
        self.stunned_at: Optional[float] = None
        self.messages_in_cluster = 0
        self.last_beacon_at: Optional[float] = None

        # Node-local coalescing. A->B announces have one semantic key regardless
        # of their sequence/retry copies. Beacons are unique per origin sequence.
        self.pending_signals: Dict[Tuple[object, ...], Signal] = {}
        self.pending_discoveries: Dict[int, Signal] = {}
        self.seen_signal_state: Dict[Tuple[object, ...], Tuple[int, int]] = {}
        self.awaiting_announces: Dict[int, Tuple[float, int, int]] = {}  # target -> (deadline, attempt, seq)
        self.next_beacon_sequence = 0
        self.next_announce_sequence = 0

        self.messages_received = 0       # protocol packets
        self.signal_items_received = 0
        self.messages_applied = 0        # successful feedback rewards applied to tensor
        self.signals_accepted = 0
        self.messages_forwarded = 0      # signal copies routed by timer
        self.packets_sent = 0             # outgoing neighbour bundles
        self.beacons_sent = 0
        self.announces_sent = 0
        self.timer_fires = 0
        self.max_cluster_size = 0
        self.last_stun_delay = 0.0
        self.max_bundle_size_sent = 0
        self.last_received_message_tensor_mass = 0.0

    def add_neighbor(self, neighbor_id: int) -> None:
        if neighbor_id != self.id:
            self.neighbors.add(int(neighbor_id))

    def _replace_timer(self, dispatch_time: float) -> Timer:
        timer = Timer(float(max(self.sim.now, dispatch_time)), self)
        self.current_timer = timer
        self.sim.schedule(timer)
        return timer

    def start_beacon_clock(self) -> None:
        if self.active and self.current_timer is None:
            self._replace_timer(self.sim.now + self.beacon_interval)

    def request_announce(self, target_id: int) -> None:
        """Remember a local A->B request; no outbound Message is made here."""
        if target_id not in self.nodes or target_id == self.id:
            return
        if not (self.active and self.nodes[target_id].active):
            return
        signal = Signal("announce", self.embedding.copy(), self.id,
                        self.next_announce_sequence, target_id=int(target_id),
                        ttl=self.sim.initial_ttl, origin_address=self.address,
                        target_address=self.nodes[int(target_id)].address)
        self.next_announce_sequence += 1
        self._coalesce_pending(signal)
        if self.stunned_at is None:
            self._replace_timer(self.sim.now)

    def _coalesce_pending(self, signal: Signal) -> None:
        old = self.pending_signals.get(signal.key)
        if old is None:
            self.pending_signals[signal.key] = signal
            return
        # Prefer an explicitly broadened retry; otherwise retain the shortest path.
        if (signal.route_attempt > old.route_attempt or
                (signal.route_attempt == old.route_attempt and signal.hop_count < old.hop_count)):
            self.pending_signals[signal.key] = signal

    def handle_incoming_message(self, message: Message) -> None:
        """Learn/coalesce incoming items, batching embedding transforms for GPU use.

        This handler never prepares outbound packets. It only updates local learning
        state, coalesces signals, records reached announces, and replaces the stun timer.
        """
        if message.target is not self:
            raise RuntimeError("message delivered to the wrong node")
        if abs(message.dispatch_time - self.sim.now) > 1e-9:
            raise RuntimeError("simulation.now must equal current event dispatch_time")
        if message.kind == "join_intent":
            if message.join_from is not None and message.join_to is not None:
                self.sim.handle_join_intent(message.join_from, message.join_to)
            return
        if not self.active:
            return

        received_items, decoded_message_tensor = message.decode_payloads()
        if message.tensor_payload:
            if (decoded_message_tensor is None or len(decoded_message_tensor.shape) != 3
                    or decoded_message_tensor.shape[1:] != EMBEDDING_SHAPE
                    or decoded_message_tensor.shape != message.tensor_shape):
                raise RuntimeError("received message tensor was not reconstructed correctly")
            if not np.all(np.isfinite(decoded_message_tensor)):
                raise RuntimeError("received message tensor contains non-finite values")
            if message.tensor_codec in {"sparse-haar+zlib", "haar-f32"}:
                self.sim.message_wavelet_decoded += 1
            self.sim.message_tensor_reconstructed_mass += float(
                np.sum(decoded_message_tensor, dtype=np.float64)
            )
            self.last_received_message_tensor_mass = float(
                np.sum(decoded_message_tensor, dtype=np.float64)
            )

        self.messages_received += 1
        self.signal_items_received += len(message.items)
        source_pos = hash_to_2d_point(message.source_eth_address)
        offset = self.pos - source_pos  # sender/source -> receiving/target vector

        # Entry layout: [signal, is_first, better_path, is_self_loopback, local_embedding].
        prepared: List[list] = []
        for received in received_items:
            if received.kind not in ("beacon", "announce", "feedback"):
                continue
            if len(received.origin_address) != 20 or (received.target_address is not None and len(received.target_address) != 20):
                self.sim.signal_items_dropped_as_duplicates += 1
                continue
            if received.hop_count > self.max_hops:
                self.sim.signal_items_dropped_hop_limit += 1
                continue
            # TTL is hop-bounded metadata. A signal arriving with TTL=0 may still
            # be consumed at this node (including successful target delivery), but
            # the timer handler will never forward it onward.
            key = received.key
            previous = self.seen_signal_state.get(key)
            is_first = previous is None
            better_path = previous is not None and received.hop_count < previous[0]
            broader_retry = previous is not None and received.route_attempt > previous[1]
            if previous is not None and not (better_path or broader_retry):
                self.sim.signal_items_dropped_as_duplicates += 1
                continue
            best_hops = received.hop_count if is_first or better_path else previous[0]
            best_attempt = received.route_attempt if previous is None else max(previous[1], received.route_attempt)
            self.seen_signal_state[key] = (best_hops, best_attempt)

            is_self_loopback = (message.source_address == self.id
                                and message.target_address == self.id)
            local_embedding = (resize_embedding_cubic(received.embedding, EMBEDDING_SHAPE)
                               if is_self_loopback else None)
            prepared.append([received, is_first, better_path, is_self_loopback, local_embedding])

        # All protocol items inside a packet have the same immediate sender, so they
        # share a coordinate-frame offset. Transform them as one batch (CUDA or Apple MPS for large bundles when available;
        # vectorized NumPy cubic interpolation otherwise).
        translate_indices = [i for i, entry in enumerate(prepared) if not entry[3]]
        if translate_indices:
            translated = translate_embedding_batch(
                [prepared[i][0].embedding for i in translate_indices], offset, self.max_radius,
                twist=self.sim.twist_enabled,
                gaussian_enabled=self.sim.gaussian_enabled,
            )
            for item_index, local_embedding in zip(translate_indices, translated):
                prepared[item_index][4] = local_embedding

        accepted = 0
        for received, is_first, better_path, is_self_loopback, local_embedding in prepared:
            assert local_embedding is not None
            localized = replace(
                received, embedding=normalize_embedding_layers(
                    local_embedding, received.embedding_depth if is_self_loopback else min(HOP_BINS, received.embedding_depth + 1)
                ),
                embedding_depth=(received.embedding_depth if is_self_loopback
                                 else min(HOP_BINS, received.embedding_depth + 1)),
            )
            if is_first or better_path:
                self.signals_accepted += 1

            if localized.kind == "feedback":
                # Only a new/improved success acknowledgement reinforces a route.
                # TTL expiry and duplicate circulation earn no positive reward.
                if is_first or better_path:
                    self._apply_success_feedback_reward(localized)
                if localized.origin_id == self.id and not localized.return_path:
                    if is_first or better_path:
                        self.sim.success_feedbacks_delivered += 1
                else:
                    self._coalesce_pending(localized)
            elif localized.kind == "announce" and localized.target_id == self.id:
                previous_success = self.pending_discoveries.get(localized.origin_id)
                if previous_success is None or localized.hop_count < previous_success.hop_count:
                    self.pending_discoveries[localized.origin_id] = localized
            else:
                # A newly received beacon teaches the reverse direction toward its
                # exact origin: the immediate sender is a known next hop back to it.
                # This is local path learning, not a reward for TTL expiry itself.
                local_beacon_loopback = (
                    is_self_loopback and localized.kind == "beacon"
                    and localized.origin_id == self.id and localized.hop_count == 0
                )
                if (localized.kind == "beacon" and not local_beacon_loopback
                        and (is_first or better_path)):
                    self._apply_embedding_reward(
                        localized, message.source_address, forward_hop=localized.hop_count
                    )
                if not local_beacon_loopback:
                    self._coalesce_pending(localized)
            accepted += 1

        if accepted == 0:
            return
        if self.stunned_at is None:
            self.stunned_at = message.dispatch_time
        self.messages_in_cluster += accepted
        self.max_cluster_size = max(self.max_cluster_size, self.messages_in_cluster)
        self.last_stun_delay = sigmoid_stunning_delay(self.messages_in_cluster)
        # Anchor timer to the first message in this cluster, not the latest arrival.
        self._replace_timer(self.stunned_at + self.last_stun_delay)

    @staticmethod
    def _profile_weights(profile: np.ndarray, max_bins: int = 3) -> List[Tuple[int, float]]:
        """Keep a small, normalized fuzzy neighbourhood around an angle profile."""
        values = np.maximum(np.asarray(profile, dtype=np.float64), 0.0)
        if values.size == 0 or float(values.max(initial=0.0)) <= 1e-12:
            return [(0, 1.0)]
        order = np.argsort(values)[::-1]
        cutoff = float(values[order[0]]) * 0.18
        selected = [int(i) for i in order[:max_bins] if values[i] >= cutoff]
        total = float(sum(values[i] for i in selected))
        if total <= 1e-12:
            return [(int(order[0]), 1.0)]
        return [(i, float(values[i] / total)) for i in selected]

    def _source_signature_weights(self, signal: Signal) -> List[Tuple[Tuple[int, int], float]]:
        """Fuzzy Source(Angle x Hop) signature from the translated path embedding."""
        embedding = resize_embedding_cubic(signal.embedding, EMBEDDING_SHAPE)
        hop = min(max(int(signal.hop_count), 0), self.hop_bins - 1)
        profile = embedding[:, hop]
        if float(np.sum(profile)) <= 1e-10:
            profile = np.sum(embedding, axis=1)
        return [((angle, hop), weight) for angle, weight in self._profile_weights(profile)]

    def _source_signature(self, signal: Signal) -> Tuple[int, int]:
        """Representative hard bin for diagnostics/tests; routing uses fuzzy weights."""
        return max(self._source_signature_weights(signal), key=lambda item: item[1])[0]

    def _target_signature_weights(self, signal: Signal) -> List[Tuple[Tuple[int, int], float]]:
        """The target's hash-derived angle is only a soft hint; target hop is unknown."""
        target_id = signal.target_id if signal.target_id is not None else signal.origin_id
        target_pos = self.nodes[target_id].pos if target_id in self.nodes else hash_to_2d_point(
            signal.target_address or signal.origin_address
        )
        delta = target_pos - self.pos
        return [((angle, 0), weight) for angle, weight in fuzzy_angle_weights(delta, self.angle_bins)]

    def _target_signature(self, signal: Signal) -> Tuple[int, int]:
        return max(self._target_signature_weights(signal), key=lambda item: item[1])[0]

    def _neighbour_signature_weights(self, neighbour_id: int) -> List[Tuple[int, float]]:
        """Neighbour is Angle x 1: only its fuzzy direction bin is stored."""
        neighbour_delta = self.nodes[neighbour_id].pos - self.pos
        return fuzzy_angle_weights(neighbour_delta, self.tensor_bins)

    def _neighbour_signature(self, neighbour_id: int) -> int:
        return max(self._neighbour_signature_weights(neighbour_id), key=lambda item: item[1])[0]

    def _apply_embedding_reward(self, signal: Signal, forward_neighbour_id: int,
                                forward_hop: Optional[int] = None) -> None:
        """Update sparse Hebbian weights from a beacon path or confirmed route.

        Beacons provide a modest reverse-path hint: the neighbour a beacon arrived
        from is a known direction toward the beacon's exact origin. Announces only
        receive the stronger reward when explicit success feedback returns.
        """
        if not self.sim.learning_enabled:
            return
        learning_signal = signal if forward_hop is None else replace(signal, hop_count=int(forward_hop))
        source_bins = self._source_signature_weights(learning_signal)
        target_bins = self._target_signature_weights(learning_signal)
        neighbour_bins = self._neighbour_signature_weights(forward_neighbour_id)
        # Default 0.35 preserves the established 0.35 feedback and 0.12 beacon rewards.
        reward_scale = self.sim.hebbian_reward / DEFAULT_HEBBIAN_REWARD
        base_reward = (0.12 if signal.kind == "beacon" else DEFAULT_HEBBIAN_REWARD) * reward_scale
        reward = base_reward * signal.coefficient / (1.0 + 0.25 * max(0, learning_signal.hop_count))
        for source, sw in source_bins:
            for target, tw in target_bins:
                for neighbour, nw in neighbour_bins:
                    self.tensor.reward(
                        source, target, neighbour, time_epoch=self.timer_fires,
                        amount=reward * sw * tw * nw,
                    )
        self.messages_applied += 1
        if signal.kind == "feedback":
            self.sim.successful_route_rewards += 1
        elif signal.kind == "beacon":
            self.sim.beacon_path_rewards += 1

    def _apply_success_feedback_reward(self, signal: Signal) -> None:
        """Reward this node's forward next-hop choice when an ACK returns over it."""
        try:
            index = signal.route_path.index(self.id)
        except ValueError:
            return
        if index >= len(signal.route_path) - 1:
            # Target itself has no forward next-hop choice to reinforce.
            return
        forward_neighbour_id = signal.route_path[index + 1]
        if forward_neighbour_id not in self.neighbors:
            # A topology change may have removed the old edge; don't reward a path
            # that is no longer traversable.
            return
        self._apply_embedding_reward(signal, forward_neighbour_id, forward_hop=index)

    def _route_weight(self, signal: Signal, neighbor_id: int) -> float:
        """Fuzzy expected learned score, decayed over the eight-slot time history."""
        source_bins = self._source_signature_weights(signal)
        target_bins = self._target_signature_weights(signal)
        neighbour_bins = self._neighbour_signature_weights(neighbor_id)
        expected = 0.0
        for source, sw in source_bins:
            for target, tw in target_bins:
                for neighbour, nw in neighbour_bins:
                    expected += sw * tw * nw * self.tensor.score(
                        source, target, neighbour, time_epoch=self.timer_fires,
                        decay=self.sim.coefficient_decay,
                    )
        return expected

    def _choose_destinations(self, signal: Signal) -> Tuple[List[int], bool]:
        if self.sim.ttl_enabled and signal.ttl <= 0:
            self.sim.signal_items_dropped_ttl += 1
            return [], False
        if signal.kind == "feedback":
            if not signal.return_path:
                return [], False
            next_node = signal.return_path[0]
            if next_node in self.neighbors:
                return [next_node], False
            self.sim.no_route_signal_drops += 1
            return [], False

        blocked = set(signal.visited)
        blocked.add(self.id)
        eligible = [n for n in sorted(self.neighbors) if n not in blocked]
        if not eligible:
            self.sim.no_route_signal_drops += 1
            return [], False
        if signal.kind == "announce" and signal.route_attempt >= 1:
            # A timed retry broadens exploration when the preferred learned route failed.
            return eligible, False

        selected: List[int] = []
        for neighbor_id in eligible:
            self.sim.candidate_neighbor_checks += 1
            score = signal.coefficient * self._route_weight(signal, neighbor_id)
            if score >= self.sim.forwarding_threshold:
                selected.append(neighbor_id)
        if selected:
            return selected, True
        # Cold-start / failed-learned-route fallback: safe ripple gossip. Even in this
        # worst case the timer will aggregate all signals into one packet per neighbour.
        return eligible, False

    @staticmethod
    def _prefer_bundled_item(old: Signal, new: Signal) -> Signal:
        if new.route_attempt > old.route_attempt:
            return new
        if new.route_attempt < old.route_attempt:
            return old
        if new.hop_count < old.hop_count:
            return new
        if new.hop_count > old.hop_count:
            return old
        return new if new.coefficient > old.coefficient else old

    @staticmethod
    def _build_message_tensor(items: List[Signal]) -> np.ndarray:
        """Stack per-signal angle×hop embeddings into one padded message tensor.

        A signal-index axis makes each fuzzy embedding reconstructable at the receiver;
        exact signal identity and target/TTL semantics remain in the parallel metadata
        list. Padding the leading dimension to a power of two allows Haar transforms
        along signal, angle and hop axes.
        """
        count = len(items)
        padded_count = 1 << max(0, (count - 1).bit_length())
        tensor = np.zeros((padded_count, *EMBEDDING_SHAPE), dtype=np.float32)
        for index, signal in enumerate(items):
            embedding = normalize_embedding_layers(
                resize_embedding_cubic(signal.embedding, EMBEDDING_SHAPE), signal.embedding_depth
            )
            tensor[index] = embedding
        return tensor

    def _schedule_bundle(self, destination_id: int, items: List[Signal]) -> None:
        if not items or destination_id not in self.nodes:
            return

        is_network_send = destination_id != self.id
        if is_network_send:
            # Stable ordering lets metadata row i match tensor slice i. Similar kinds
            # and identities are adjacent, so coefficient blur tends to merge related
            # signal embeddings rather than arbitrary queue order.
            items = sorted(items, key=lambda item: (
                item.kind, item.origin_address, item.target_address or b"",
                item.sequence_id, item.hop_count, item.route_attempt,
            ))
            # Configurable sender path. Default: noise -> Haar all axes -> Gaussian
            # coefficient blur -> threshold -> sparse DEFLATE. The receiver performs
            # the inverse Haar. --no-haar keeps noise but sends the noisy tensor raw.
            message_tensor = self._build_message_tensor(items)
            processed_tensor = prepare_wavelet_coefficients(
                message_tensor, self.sim.wavelet_rng,
                noise_std=self.sim.noise_strength, noise_enabled=self.sim.noise_enabled,
                haar_enabled=self.sim.haar_enabled,
                gaussian_enabled=self.sim.gaussian_enabled,
                gaussian_sigma=self.sim.gaussian_width,
                threshold=self.sim.wavelet_threshold,
            )
            tensor_shape = tuple(message_tensor.shape)
            if self.sim.haar_enabled:
                coefficient_count = int(np.count_nonzero(processed_tensor))
                if self.sim.compression_enabled:
                    tensor_payload, coefficient_count = pack_sparse_wavelet_coefficients(processed_tensor)
                    tensor_codec = "sparse-haar+zlib"
                else:
                    tensor_payload = processed_tensor.astype("<f4", copy=False).tobytes()
                    tensor_codec = "haar-f32"
            else:
                coefficient_count = int(np.count_nonzero(processed_tensor))
                tensor_payload = processed_tensor.astype("<f4", copy=False).tobytes()
                tensor_codec = "raw-f32"
            wire_items = [replace(item, embedding=np.zeros(EMBEDDING_SHAPE, dtype=np.float32))
                          for item in items]
            if self.sim.haar_enabled:
                self.sim.message_wavelet_encoded += 1
            self.sim.message_tensor_raw_bytes += int(message_tensor.nbytes)
            self.sim.message_tensor_compressed_bytes += len(tensor_payload)
            self.sim.message_wavelet_coefficients += coefficient_count
        else:
            # A self-beacon loopback is an event, not a network transmission; skip
            # wire compression and keep its local embedding directly.
            message_tensor = None
            tensor_payload = b""
            coefficient_count = 0
            tensor_shape = (1, *EMBEDDING_SHAPE)
            wire_items = [replace(item, embedding=item.embedding.copy()) for item in items]

        message = Message(
            dispatch_time=self.sim.now + (0.0 if not is_network_send else self.sim.link_delay),
            source_address=self.id,
            target_address=destination_id,
            target=self.nodes[destination_id],
            items=wire_items,
            kind="bundle",
            sent_at=self.sim.now,
            source_eth_address=self.address,
            target_eth_address=self.nodes[destination_id].address,
            tensor_payload=tensor_payload,
            tensor_shape=tensor_shape,
            tensor_codec=(tensor_codec if is_network_send else "local"),
            tensor_coefficient_count=coefficient_count,
            tensor_uncompressed_bytes=(int(message_tensor.nbytes) if message_tensor is not None else 0),
        )
        self.sim.schedule(message)
        self.sim.messages_scheduled += 1
        self.sim.protocol_payload_items_scheduled += len(items)
        if is_network_send:
            self.sim.network_packets_scheduled += 1
            self.sim.network_signal_copies_scheduled += len(items)
            self.packets_sent += 1
            self.max_bundle_size_sent = max(self.max_bundle_size_sent, len(items))

    def _emit_due_beacon(self) -> Optional[Signal]:
        if self.last_beacon_at is not None and self.sim.now + 1e-12 < self.last_beacon_at + self.beacon_interval:
            return None
        signal = Signal(
            kind="beacon", embedding=self.embedding.copy(), origin_id=self.id,
            sequence_id=self.next_beacon_sequence, hop_count=0, visited=(), coefficient=1.0,
            ttl=self.sim.initial_ttl, origin_address=self.address,
        )
        self.next_beacon_sequence += 1
        self.last_beacon_at = self.sim.now
        self.beacons_sent += 1
        # An ordinary self-directed Message. It will be coalesced with other
        # signals only if they are actually addressed to this node; all neighbour
        # forwarding still happens from a timer handler.
        self._schedule_bundle(self.id, [signal])
        return signal

    def _add_due_retries(self, work: List[Signal]) -> None:
        for target_id, (deadline, attempt, sequence_id) in list(self.awaiting_announces.items()):
            if (self.id, target_id) in self.sim.announce_directions_discovered:
                self.awaiting_announces.pop(target_id, None)
                continue
            if self.sim.now + 1e-12 < deadline:
                continue
            if attempt >= self.max_announce_attempts:
                self.awaiting_announces.pop(target_id, None)
                continue
            retry = Signal(
                kind="announce", embedding=self.embedding.copy(), origin_id=self.id,
                sequence_id=self.next_announce_sequence, target_id=target_id,
                hop_count=0, visited=(), coefficient=1.0, route_attempt=attempt + 1,
                ttl=self.sim.initial_ttl, origin_address=self.address,
                target_address=self.nodes[target_id].address,
            )
            self.next_announce_sequence += 1
            work.append(retry)
            self.awaiting_announces[target_id] = (
                self.sim.now + max(2.0, 2.0 * self.beacon_interval), attempt + 1, retry.sequence_id
            )

    def handle_timer(self, timer: Timer) -> None:
        """Flush accumulated signals and generate at most one bundle per neighbour."""
        if timer is not self.current_timer:
            return
        self.current_timer = None
        self.timer_fires += 1
        self.stunned_at = None
        self.messages_in_cluster = 0

        if not self.active:
            return

        # Denoise the full six-axis learned tensor periodically, amortizing the
        # dense Haar operation across stun clusters. The reconstructed tensor is
        # projected back into sparse storage before its weights are queried.
        if (self.sim.learning_enabled and self.tensor.updates_since_filter > 0
                and self.timer_fires % NODE_WAVELET_EVERY_N_TIMERS == 0):
            coefficient_count, pruned = self.tensor.wavelet_denoise(
                self.sim.wavelet_rng, noise_std=self.sim.noise_strength,
                noise_enabled=self.sim.noise_enabled, haar_enabled=self.sim.haar_enabled,
                gaussian_enabled=self.sim.gaussian_enabled,
                gaussian_sigma=self.sim.gaussian_width, threshold=self.sim.node_threshold,
            )
            if coefficient_count or pruned:
                self.sim.node_wavelet_filter_passes += 1
                self.sim.node_wavelet_coefficients += coefficient_count
                self.sim.node_wavelet_keys_pruned += pruned

        # Successful destination delivery is finalized on the timer. Each success
        # schedules a reverse-path feedback signal; only that returning feedback
        # earns Hebbian reward along the original forward route.
        successful_announces = list(self.pending_discoveries.values())
        self.pending_discoveries.clear()
        feedback_work: List[Signal] = []
        for announce in successful_announces:
            if announce.target_id is not None:
                self.sim.resolve_announce_discovery(announce.origin_id, self.id)
            forward_path = tuple(announce.visited) + (self.id,)
            if len(forward_path) >= 2 and forward_path[0] == announce.origin_id:
                return_path = tuple(reversed(forward_path[:-1]))
                feedback = Signal(
                    kind="feedback", embedding=announce.embedding.copy(),
                    origin_id=announce.origin_id, sequence_id=announce.sequence_id,
                    target_id=self.id, hop_count=0, visited=(), coefficient=1.0,
                    ttl=min(self.sim.initial_ttl, len(return_path)),
                    origin_address=announce.origin_address,
                    target_address=self.address, route_path=forward_path,
                    return_path=return_path, embedding_depth=announce.embedding_depth,
                )
                feedback_work.append(feedback)
                self.sim.feedback_signals_generated += 1

        work = list(self.pending_signals.values()) + feedback_work
        self.pending_signals.clear()
        self._add_due_retries(work)

        # Beacon creation returns a normal signal and schedules its ordinary
        # self-directed Message. The same signal joins this timer's outbound work,
        # so it can share one neighbour packet with any pending announces/relays.
        new_beacon = self._emit_due_beacon()
        if new_beacon is not None:
            work.append(new_beacon)

        # Group all signal copies by destination, so beacons and announces share one packet.
        by_destination: Dict[int, Dict[Tuple[object, ...], Signal]] = {}
        for signal in work:
            if signal.kind == "announce" and signal.target_id == self.id:
                continue
            if signal.hop_count >= self.max_hops:
                self.sim.signal_items_dropped_hop_limit += 1
                continue
            destinations, smart = self._choose_destinations(signal)
            for destination_id in destinations:
                if self.sim.ttl_enabled and signal.ttl <= 0:
                    self.sim.signal_items_dropped_ttl += 1
                    continue
                forwarded = replace(
                    signal,
                    embedding=signal.embedding.copy(),
                    hop_count=signal.hop_count + 1,
                    visited=(signal.visited if signal.kind == "feedback"
                             else signal.visited + (self.id,)),
                    return_path=(signal.return_path[1:] if signal.kind == "feedback"
                                 else signal.return_path),
                    ttl=(signal.ttl - 1 if self.sim.ttl_enabled else signal.ttl),
                    coefficient=signal.coefficient * self.sim.coefficient_decay,
                )
                dest_items = by_destination.setdefault(destination_id, {})
                previous = dest_items.get(forwarded.key)
                dest_items[forwarded.key] = forwarded if previous is None else self._prefer_bundled_item(previous, forwarded)
                self.messages_forwarded += 1
                if signal.kind == "feedback":
                    self.sim.feedback_signal_copies += 1
                elif smart:
                    self.sim.smart_signal_copies += 1
                else:
                    self.sim.fallback_ripple_signal_copies += 1
            # Announce origin registers the attempt deadline once per timer flush.
            if signal.kind == "announce" and signal.origin_id == self.id and signal.target_id is not None:
                existing = self.awaiting_announces.get(signal.target_id)
                attempt = signal.route_attempt
                if existing is None or attempt >= existing[1]:
                    self.awaiting_announces[signal.target_id] = (
                        self.sim.now + max(2.0, 2.0 * self.beacon_interval), attempt, signal.sequence_id
                    )
                    self.announces_sent += 1

        for destination_id in sorted(by_destination):
            bundle = list(by_destination[destination_id].values())
            self._schedule_bundle(destination_id, bundle)

        # Maintain one timer for either the next beacon or the next announce retry.
        if self.current_timer is None:
            next_beacon = (self.sim.now + self.beacon_interval if self.last_beacon_at is None
                           else self.last_beacon_at + self.beacon_interval)
            retry_times = [deadline for target_id, (deadline, _, _) in self.awaiting_announces.items()
                           if (self.id, target_id) not in self.sim.announce_directions_discovered
                           and deadline > self.sim.now + 1e-12]
            next_dispatch = min([next_beacon, *retry_times]) if retry_times else next_beacon
            if next_dispatch <= self.sim.now + 1e-12:
                next_dispatch = self.sim.now + 1e-9
            self._replace_timer(next_dispatch)


# ---------------------------------------------------------------------------
# Dynamic join scenario and tests
# ---------------------------------------------------------------------------

def connect_nodes(nodes: Dict[int, P2PNode], left: int, right: int) -> bool:
    if left == right or left not in nodes or right not in nodes:
        return False
    if right in nodes[left].neighbors:
        return False
    nodes[left].add_neighbor(right)
    nodes[right].add_neighbor(left)
    return True


def _schedule_join_intent(sim: Simulation, nodes: Dict[int, P2PNode], when: float,
                          left_id: int, right_id: int, sequence_id: int) -> None:
    event = Message(
        dispatch_time=float(when), source_address=left_id, target_address=right_id,
        target=nodes[right_id], items=[], kind="join_intent", join_from=left_id, join_to=right_id,
        sent_at=float(when),
    )
    sim.schedule(event)
    sim.join_intents_scheduled += 1


def build_demo(node_count: int = 100, seed: int = 42, beacon_interval: float = 1.0,
               join_interval: float = 0.05, extra_connect_requests: int = 20,
               announce_spacing: float = 0.25, forwarding_threshold: float = 0.20,
               fallback_timeout: float = 5.0, initial_ttl: int = DEFAULT_TTL,
               learning_enabled: bool = True, **sim_options
               ) -> Tuple[Simulation, Dict[int, P2PNode], dict]:
    """Start with no links and queue real join requests that construct the graph.

    First request connects two outsiders; each later growth request joins one
    outsider to a node already in the overlay. After growth, active non-neighbours
    try mutual A->B announces, and links are added only when announces arrive.
    """
    if node_count < 2:
        raise ValueError("node_count must be >= 2")
    if join_interval <= 0 or announce_spacing <= 0:
        raise ValueError("join_interval and announce_spacing must be positive")
    if extra_connect_requests < 0:
        raise ValueError("extra_connect_requests must be non-negative")
    rng = random.Random(seed)
    sim = Simulation(forwarding_threshold=forwarding_threshold, fallback_timeout=fallback_timeout,
                     initial_ttl=initial_ttl, wavelet_seed=seed,
                     learning_enabled=learning_enabled, **sim_options)
    nodes: Dict[int, P2PNode] = {}
    sim.nodes = nodes
    for node_id in range(node_count):
        nodes[node_id] = P2PNode(
            node_id, sim, nodes, beacon_interval=beacon_interval,
            max_hops=max(32, node_count + 1),
        )

    intent_seq = 0
    # First ever link is between two unjoined candidates.
    _schedule_join_intent(sim, nodes, 0.0, 0, 1, intent_seq)
    intent_seq += 1

    tree_edges = {(0, 1)}
    parents = {1: 0}
    for node_id in range(2, node_count):
        parent_id = rng.randrange(0, node_id)
        parents[node_id] = parent_id
        tree_edges.add(tuple(sorted((parent_id, node_id))))
        _schedule_join_intent(sim, nodes, (node_id - 1) * join_interval,
                              parent_id, node_id, intent_seq)
        intent_seq += 1

    join_end = (node_count - 1) * join_interval
    announce_start = join_end + max(1.5, 2.0 * beacon_interval)
    planned_adj = {n: set() for n in range(node_count)}
    for left, right in tree_edges:
        planned_adj[left].add(right)
        planned_adj[right].add(left)

    def tree_distance(start: int, target: int) -> int:
        queue = [(start, 0)]
        seen = {start}
        for current, distance in queue:
            if current == target:
                return distance
            for neighbour in planned_adj[current]:
                if neighbour not in seen:
                    seen.add(neighbour)
                    queue.append((neighbour, distance + 1))
        return node_count + 1

    possible = [(a, b) for a in range(node_count) for b in range(a + 1, node_count)
                if (a, b) not in tree_edges]
    rng.shuffle(possible)
    possible.sort(key=lambda pair: tree_distance(*pair), reverse=True)
    selected = possible[:min(extra_connect_requests, len(possible))]
    for index, (left_id, right_id) in enumerate(selected):
        _schedule_join_intent(sim, nodes, announce_start + index * announce_spacing,
                              left_id, right_id, intent_seq)
        intent_seq += 1

    meta = {
        "parents": parents,
        "seed_edges": tree_edges,
        "join_end": join_end,
        "announce_start": announce_start,
        "last_intent_time": (announce_start + (len(selected) - 1) * announce_spacing
                              if selected else join_end),
        "announce_pairs_planned": selected,
    }
    return sim, nodes, meta


def _is_connected(nodes: Dict[int, P2PNode], active_ids: List[int]) -> bool:
    if len(active_ids) <= 1:
        return True
    active_set = set(active_ids)
    seen = {active_ids[0]}
    stack = [active_ids[0]]
    while stack:
        current = stack.pop()
        for neighbour in nodes[current].neighbors:
            if neighbour in active_set and neighbour not in seen:
                seen.add(neighbour)
                stack.append(neighbour)
    return seen == active_set


def run_self_tests() -> None:
    # 0. Device selection is backend-generic. Mock hardware discovery so the
    # preference logic can be tested on a CPU-only host as well as on Macs.
    original_cuda_available = _cuda_available
    original_mps_available = _mps_available
    original_device = _EMBEDDING_DEVICE
    try:
        globals()["_cuda_available"] = lambda: False
        globals()["_mps_available"] = lambda: False
        assert configure_embedding_device("auto") == "cpu"
        assert available_embedding_devices() == ("cpu",)

        globals()["_cuda_available"] = lambda: False
        globals()["_mps_available"] = lambda: True
        assert configure_embedding_device("auto") == "mps"
        assert configure_embedding_device("mps") == "mps"
        assert available_embedding_devices() == ("mps", "cpu")

        globals()["_cuda_available"] = lambda: True
        globals()["_mps_available"] = lambda: True
        assert configure_embedding_device("auto") == "cuda"
        assert available_embedding_devices() == ("cuda", "mps", "cpu")
        assert configure_embedding_device("cpu") == "cpu"

        globals()["_cuda_available"] = lambda: False
        globals()["_mps_available"] = lambda: False
        for unavailable in ("cuda", "mps"):
            try:
                configure_embedding_device(unavailable)
            except RuntimeError:
                pass
            else:
                raise AssertionError(f"explicit unavailable device {unavailable!r} must fail clearly")
    finally:
        globals()["_cuda_available"] = original_cuda_available
        globals()["_mps_available"] = original_mps_available
        globals()["_EMBEDDING_DEVICE"] = original_device

    # 1. Multidimensional Haar transforms cover every axis and invert cleanly.
    wavelet_rng = np.random.default_rng(123)
    wavelet_probe = wavelet_rng.normal(size=LOGICAL_TENSOR_SHAPE).astype(np.float32)
    wavelet_coefficients = haar_forward_nd(wavelet_probe)
    wavelet_rebuilt = haar_inverse_nd(wavelet_coefficients)
    assert wavelet_coefficients.shape == LOGICAL_TENSOR_SHAPE
    assert np.max(np.abs(wavelet_probe - wavelet_rebuilt)) < 1e-5

    # Explicitly test the PyTorch implementation on the selected accelerator,
    # or on CPU when no accelerator exists. This validates the exact same
    # operation sequence used by the MPS/CUDA wavelet pipeline.
    if torch is not None:
        torch_test_device = _accelerator_tensor_device() or torch.device("cpu")
        torch_probe_np = np.random.default_rng(321).normal(size=(2, 8, 8)).astype(np.float32)
        with torch.no_grad():
            torch_probe = torch.as_tensor(torch_probe_np, device=torch_test_device)
            torch_coeff = _haar_forward_nd_torch(torch_probe)
            torch_rebuilt = _haar_inverse_nd_torch(torch_coeff)
            assert tuple(torch_coeff.shape) == torch_probe_np.shape
            assert float(torch.max(torch.abs(torch.as_tensor(torch_probe_np, device=torch_test_device) - torch_rebuilt)).item()) < 1e-5
            torch_blurred = _torch_gaussian_filter_nd(torch_coeff, 0.30)
            assert tuple(torch_blurred.shape) == torch_probe_np.shape
            assert bool(torch.isfinite(torch_blurred).all().item())
            scipy_blurred = gaussian_filter(
                torch_coeff.detach().cpu().numpy(), sigma=(0.30,) * torch_coeff.ndim,
                mode="nearest",
            )
            assert float(np.max(np.abs(torch_blurred.detach().cpu().numpy() - scipy_blurred))) < 2e-5
    noisy_coefficients = prepare_wavelet_coefficients(
        np.zeros(LOGICAL_TENSOR_SHAPE, dtype=np.float32), wavelet_rng
    )
    assert np.count_nonzero(noisy_coefficients) == 0, "background noise should fall below the threshold"
    packed_coefficients, packed_count = pack_sparse_wavelet_coefficients(wavelet_coefficients)
    unpacked_coefficients = unpack_sparse_wavelet_coefficients(packed_coefficients)
    assert packed_count == int(np.count_nonzero(wavelet_coefficients))
    assert np.array_equal(unpacked_coefficients, wavelet_coefficients)

    # --no-learning control: every logical entry has unit weight, with no updates.
    neutral_tensor = SparseHebbianTensor(learning_enabled=False)
    assert neutral_tensor.score((0, 0), (7, 7), 3, time_epoch=0) == 1.0
    neutral_tensor.reward((0, 0), (7, 7), 3, time_epoch=0, amount=2.0)
    assert neutral_tensor.stored_combinations == 0
    assert neutral_tensor.reward_total == 0.0
    assert neutral_tensor.wavelet_denoise(wavelet_rng) == (0, 0)
    assert neutral_tensor.score((0, 0), (7, 7), 3, time_epoch=99) == 1.0
    neutral_sim = Simulation(learning_enabled=False)
    neutral_nodes: Dict[int, P2PNode] = {}
    neutral_node = P2PNode(991, neutral_sim, neutral_nodes)
    neutral_nodes[991] = neutral_node
    neutral_signal = Signal("beacon", make_self_embedding(EMBEDDING_SHAPE), 991, 1)
    neutral_node._apply_embedding_reward(neutral_signal, 991)
    assert neutral_node.messages_applied == 0
    assert neutral_sim.beacon_path_rewards == 0
    assert neutral_node.tensor.reward_total == 0.0

    # Node tensor: sparse weights survive the six-axis denoise and remain sparse.
    denoise_tensor = SparseHebbianTensor()
    for _ in range(3):
        denoise_tensor.reward((1, 2), (3, 4), 5, 0, 0.03)
    before_weight = float(denoise_tensor.values[denoise_tensor._key((1, 2), (3, 4), 5)][0])
    filtered_count, pruned_count = denoise_tensor.wavelet_denoise(wavelet_rng)
    after_values = denoise_tensor.values.get(denoise_tensor._key((1, 2), (3, 4), 5))
    assert denoise_tensor.wavelet_filter_passes == 1
    assert filtered_count >= 0 and pruned_count >= 0
    assert after_values is not None and np.all(np.isfinite(after_values))
    assert float(after_values[0]) >= denoise_tensor.baseline
    assert before_weight > denoise_tensor.baseline

    # 1. Priority queue order + engine timestamp.
    sim = Simulation()
    node_map: Dict[int, P2PNode] = {}
    node_map[0] = P2PNode(0, sim, node_map)
    node_map[1] = P2PNode(1, sim, node_map)
    observed: List[float] = []
    for node_id, when in ((0, 2.0), (1, 1.0)):
        n = node_map[node_id]
        n.handle_timer = lambda timer, out=observed: out.append(timer.dispatch_time)  # type: ignore[method-assign]
        timer = Timer(when, n)
        n.current_timer = timer
        sim.schedule(timer)
    sim.run(until=2.0)
    assert observed == [1.0, 2.0], observed

    # 2. Exact initial node state.
    sim2 = Simulation()
    pool: Dict[int, P2PNode] = {}
    for node_id in range(3):
        pool[node_id] = P2PNode(node_id, sim2, pool, beacon_interval=10.0)
    sim2.nodes = pool
    assert pool[0].current_timer is None and pool[0].stunned_at is None and not pool[0].active

    # 3. Fixed eight-layer twisting: rotate every inherited layer, push up-arrow,
    # preserve empty zero layers, and discard the oldest layer on the ninth hop.
    origin = make_self_embedding(EMBEDDING_SHAPE)
    thirty_degrees = np.array([math.sin(math.radians(30.0)), math.cos(math.radians(30.0))])
    single = translate_embedding(origin, thirty_degrees)
    batch = translate_embedding_batch([origin, origin], thirty_degrees)
    assert single.shape == (8, 8) and batch.shape == (2, 8, 8)
    assert np.all(single >= 0.0) and np.all(np.isfinite(single))
    assert np.allclose(single, batch[0], atol=1e-5)
    assert np.allclose(batch[0], batch[1], atol=1e-5)
    assert np.array_equal(single[:, 0], np.eye(8, dtype=np.float32)[:, 0])
    assert abs(float(single[:, 1].sum()) - 1.0) < 1e-5
    assert int(np.argmax(single[:, 1])) == 1, "30 degree twist should point nearest the 45 degree bin"
    assert np.count_nonzero(single[:, 2:]) == 0, "empty history layers must remain zero"
    full = np.zeros(EMBEDDING_SHAPE, dtype=np.float32)
    for layer in range(HOP_BINS):
        full[layer, layer] = 1.0
    ninth = translate_embedding(full, np.array([0.0, 1.0]))
    assert np.array_equal(ninth[:, 0], np.eye(8, dtype=np.float32)[:, 0])
    for layer in range(1, HOP_BINS):
        assert np.argmax(ninth[:, layer]) == layer - 1, (layer, np.argmax(ninth[:, layer]))
        assert abs(float(ninth[:, layer].sum()) - 1.0) < 1e-5
    # Chained frame changes accumulate rotation: 30 degrees at B, then 45 at C.
    forty_five_degrees = np.array([math.sin(math.radians(45.0)), math.cos(math.radians(45.0))])
    at_c = translate_embedding(single, forty_five_degrees)
    assert np.argmax(at_c[:, 1]) == 1, "B's layer should be rotated to the 45-degree bin"
    assert np.argmax(at_c[:, 2]) == 2, "A's layer should accumulate 30+45 degrees (nearest 90-degree bin)"
    assert abs(float(at_c[:, 2].sum()) - 1.0) < 1e-5
    assert np.count_nonzero(at_c[:, 3:]) == 0
    # The same hop-history occupancy survives message Haar -> compression -> inverse Haar.
    layered = Signal("beacon", at_c.copy(), 0, 998, hop_count=2, embedding_depth=3)
    layered_message = Message(0.0, 0, 1, pool[1], [layered])
    layer_items = [layered]
    tensor = np.zeros((1, *EMBEDDING_SHAPE), dtype=np.float32)
    tensor[0] = normalize_embedding_layers(layered.embedding, layered.embedding_depth)
    layer_coeff = prepare_wavelet_coefficients(tensor, wavelet_rng, threshold=MESSAGE_WAVELET_THRESHOLD)
    layer_payload, _ = pack_sparse_wavelet_coefficients(layer_coeff)
    layered_message.tensor_payload = layer_payload
    layered_message.tensor_shape = tuple(tensor.shape)
    layered_message.items = [replace(layered, embedding=np.zeros(EMBEDDING_SHAPE, dtype=np.float32))]
    restored_layer_items, _ = layered_message.decode_payloads()
    restored_layered = restored_layer_items[0]
    assert restored_layered.embedding_depth == 3
    assert np.count_nonzero(restored_layered.embedding[:, 3:]) == 0
    assert np.allclose(restored_layered.embedding.sum(axis=0)[:3], np.ones(3), atol=1e-5)
    if torch is not None:
        torch_batch = _translate_embedding_batch_torch(np.stack([origin, origin]), thirty_degrees,
                                                       device_name="cpu")
        assert np.all(np.isfinite(torch_batch)) and np.all(torch_batch >= 0.0)
        assert np.allclose(torch_batch, batch, atol=2e-5), float(np.max(np.abs(torch_batch - batch)))

    # 4. Message input anchors the stun to first arrival; second message replaces timer.
    pool[0].active = pool[2].active = True
    e = make_self_embedding(EMBEDDING_SHAPE)
    m1 = Message(0.0, 1, 0, pool[0], [Signal("beacon", e.copy(), 1, 1, hop_count=1, visited=(1,))])
    sim2.schedule(m1)
    sim2.run(until=0.0)
    first_timer = pool[0].current_timer
    assert pool[0].stunned_at == 0.0 and first_timer is not None
    m2 = Message(0.02, 2, 0, pool[0], [Signal("beacon", e.copy(), 2, 1, hop_count=1, visited=(2,))])
    sim2.schedule(m2)
    sim2.run(until=0.02)
    second_timer = pool[0].current_timer
    assert second_timer is not None and second_timer is not first_timer
    assert abs(second_timer.dispatch_time - sigmoid_stunning_delay(2)) < 1e-9
    sim2.run(until=second_timer.dispatch_time)
    assert sim2.stale_timers_skipped >= 1

    # 4. A->B announce copies coalesce to one pending logical announce.
    n = pool[0]
    n.pending_signals.clear()
    n._coalesce_pending(Signal("announce", e.copy(), 0, 1, target_id=2))
    n._coalesce_pending(Signal("announce", e.copy(), 0, 99, target_id=2, route_attempt=1))
    assert len(n.pending_signals) == 1
    assert next(iter(n.pending_signals.values())).route_attempt == 1

    # 5. A single timer sends one bundle per neighbour, combining signal types.
    bundle_sim = Simulation(forwarding_threshold=999.0)  # force cold-start ripple fallback
    bundle_nodes: Dict[int, P2PNode] = {}
    for node_id in range(3):
        bundle_nodes[node_id] = P2PNode(node_id, bundle_sim, bundle_nodes, beacon_interval=100.0)
        bundle_nodes[node_id].active = True
    bundle_sim.nodes = bundle_nodes
    connect_nodes(bundle_nodes, 0, 1)
    connect_nodes(bundle_nodes, 0, 2)
    sender = bundle_nodes[0]
    sender.last_beacon_at = None  # make a periodic beacon due on this timer
    sender._coalesce_pending(Signal("announce", e.copy(), 0, 11, target_id=2))
    timer = Timer(0.1, sender)
    sender.current_timer = timer
    bundle_sim.schedule(timer)
    bundle_sim.run(until=0.1)
    sent_to = [event for _, _, event in bundle_sim._queue
               if isinstance(event, Message) and event.kind == "bundle"
               and event.source_address == 0 and event.target_address != 0]
    dest_counts = {}
    for event in sent_to:
        dest_counts[event.target_address] = dest_counts.get(event.target_address, 0) + 1
    assert dest_counts == {1: 1, 2: 1}, dest_counts
    assert all(len(event.items) == 2 for event in sent_to)
    assert all({item.kind for item in event.items} == {"beacon", "announce"} for event in sent_to)
    assert all(event.tensor_payload for event in sent_to)
    assert all(all(not np.any(item.embedding) for item in event.items) for event in sent_to)
    decoded_items, decoded_tensor = sent_to[0].decode_payloads()
    assert decoded_tensor is not None and decoded_tensor.shape == sent_to[0].tensor_shape
    assert decoded_tensor.shape[1:] == EMBEDDING_SHAPE
    assert np.all(np.isfinite(decoded_tensor)) and float(decoded_tensor.sum()) > 0.0
    assert decoded_items[0].embedding.shape == EMBEDDING_SHAPE
    assert abs(float(decoded_items[0].embedding.sum()) - 1.0) < 1e-5
    assert len(decoded_items[0].origin_address) == 20

    print("PASS: CUDA/MPS/CPU device selection and availability checks.", flush=True)
    print("PASS: fixed 8-layer twists, accumulated rotations, empty layers, all-axis Haar,", flush=True)
    print("      noise/blur/threshold, compressed message reconstruction and sparse node denoise.", flush=True)

    # 6. Dynamic topology and active-active announce discovery end-to-end.
    # Keep the integration test intentionally small: the production path now runs
    # a real all-axis wavelet codec per network packet, so use a sparser beacon
    # cadence while still exercising announce discovery and the five-second fallback.
    sim3, nodes3, meta = build_demo(12, seed=17, beacon_interval=1.5,
                                     join_interval=0.03, extra_connect_requests=5,
                                     announce_spacing=0.18)
    sim3.run(until=meta["last_intent_time"] + 6.5, max_events=50_000)
    active = [node.id for node in nodes3.values() if node.active]
    assert len(active) == 12, f"only {len(active)} nodes joined"
    assert sim3.join_case_counts["both_outside_direct"] == 1
    assert sim3.join_case_counts["active_plus_outsider_direct"] == 10
    assert sim3.join_case_counts["active_pair_announce"] >= 1
    assert _is_connected(nodes3, active)
    assert sim3.announce_links_created >= 1, "no announce-created peer edge"
    assert sim3.network_packets_scheduled <= sim3.network_signal_copies_scheduled

    # 7. P2P discovery cancels the scheduled five-second mesh fallback.
    sim4 = Simulation(fallback_timeout=5.0)
    p4: Dict[int, P2PNode] = {}
    p4[0] = P2PNode(0, sim4, p4, beacon_interval=100.0)
    p4[1] = P2PNode(1, sim4, p4, beacon_interval=100.0)
    sim4.nodes = p4
    p4[0].active = p4[1].active = True
    sim4.handle_join_intent(0, 1)
    fallback = sim4.pending_fallbacks[(0, 1)]
    assert fallback.dispatch_time == 5.0
    sim4.resolve_announce_discovery(0, 1)
    assert fallback.cancelled and sim4.fallback_tasks_cancelled == 1
    sim4.run(until=5.0, max_events=1000)
    assert sim4.fallback_links_created == 0
    assert sim4.cancelled_fallbacks_skipped == 1

    # 8. If P2P discovery never arrives, timeout creates one external edge and
    # both endpoints retain/schedule beacon timers.
    sim5 = Simulation(fallback_timeout=5.0)
    p5: Dict[int, P2PNode] = {}
    p5[0] = P2PNode(0, sim5, p5, beacon_interval=100.0)
    p5[1] = P2PNode(1, sim5, p5, beacon_interval=100.0)
    sim5.nodes = p5
    p5[0].active = p5[1].active = True
    sim5.handle_join_intent(0, 1)
    sim5.run(until=5.0, max_events=1000)
    assert 1 in p5[0].neighbors and 0 in p5[1].neighbors
    assert sim5.fallback_links_created == 1
    assert p5[0].current_timer is not None and p5[1].current_timer is not None

    # 9. Exact 20-byte identities stay in signal and envelope metadata, alongside TTL.
    exact = Signal("announce", e.copy(), 0, 123, target_id=2, ttl=5)
    assert isinstance(exact.origin_address, bytes) and len(exact.origin_address) == 20
    assert isinstance(exact.target_address, bytes) and len(exact.target_address) == 20
    assert isinstance(exact.signal_id, bytes) and len(exact.signal_id) == 16
    assert exact.metadata["ttl"] == 5
    envelope = Message(0.0, 0, 2, pool[2], [exact])
    assert len(envelope.source_eth_address) == 20 and len(envelope.target_eth_address) == 20
    assert envelope.metadata["signals"][0]["origin_address"] == exact.origin_address
    assert envelope.metadata["signals"][0]["target_address"] == exact.target_address

    # 10. TTL=1 permits exactly one relay: the receiver gets TTL=0 but must not
    # forward farther. Expiration itself must not create a positive tensor reward.
    ttl_sim = Simulation(initial_ttl=1, forwarding_threshold=999.0)
    ttl_nodes: Dict[int, P2PNode] = {}
    for node_id in range(3):
        ttl_nodes[node_id] = P2PNode(node_id, ttl_sim, ttl_nodes, beacon_interval=100.0)
        ttl_nodes[node_id].active = True
        ttl_nodes[node_id].last_beacon_at = 0.0
    ttl_sim.nodes = ttl_nodes
    connect_nodes(ttl_nodes, 0, 1)
    connect_nodes(ttl_nodes, 1, 2)
    ttl_signal = Signal("announce", e.copy(), 0, 77, target_id=2, ttl=1,
                        origin_address=ttl_nodes[0].address, target_address=ttl_nodes[2].address)
    ttl_nodes[0]._coalesce_pending(ttl_signal)
    ttl_timer = Timer(0.1, ttl_nodes[0])
    ttl_nodes[0].current_timer = ttl_timer
    ttl_sim.schedule(ttl_timer)
    ttl_sim.run(until=1.0, max_events=1000)
    assert ttl_sim.network_signal_copies_scheduled == 1, ttl_sim.network_signal_copies_scheduled
    assert ttl_sim.signal_items_dropped_ttl >= 1
    assert ttl_sim.successful_route_rewards == 0
    assert ttl_signal.origin_address == address_for_node(0) and len(ttl_signal.origin_address) == 20

    # 10b. Beacons provide a bounded reverse-path hint toward their exact origin;
    # TTL limits propagation radius, while the final received hop remains useful.
    beacon_sim = Simulation(initial_ttl=1, forwarding_threshold=999.0)
    beacon_nodes: Dict[int, P2PNode] = {}
    for node_id in range(3):
        beacon_nodes[node_id] = P2PNode(node_id, beacon_sim, beacon_nodes, beacon_interval=100.0)
        beacon_nodes[node_id].active = True
        beacon_nodes[node_id].last_beacon_at = 0.0
    beacon_sim.nodes = beacon_nodes
    connect_nodes(beacon_nodes, 0, 1)
    connect_nodes(beacon_nodes, 1, 2)
    beacon_nodes[0]._coalesce_pending(Signal(
        "beacon", e.copy(), 0, 88, ttl=1, origin_address=beacon_nodes[0].address
    ))
    beacon_timer = Timer(0.1, beacon_nodes[0])
    beacon_nodes[0].current_timer = beacon_timer
    beacon_sim.schedule(beacon_timer)
    beacon_sim.run(until=1.0, max_events=1000)
    assert beacon_sim.beacon_path_rewards == 1, beacon_sim.beacon_path_rewards
    assert beacon_sim.signal_items_dropped_ttl >= 1
    assert beacon_sim.successful_route_rewards == 0

    # 11. Successful A->B->C announce returns explicit feedback over C->B->A,
    # rewards the two forwarding choices, and leaves exact identities intact.
    fb_sim = Simulation(forwarding_threshold=999.0, initial_ttl=8)
    fb_nodes: Dict[int, P2PNode] = {}
    for node_id in range(3):
        fb_nodes[node_id] = P2PNode(node_id, fb_sim, fb_nodes, beacon_interval=100.0)
        fb_nodes[node_id].active = True
        fb_nodes[node_id].last_beacon_at = 0.0
    fb_sim.nodes = fb_nodes
    connect_nodes(fb_nodes, 0, 1)
    connect_nodes(fb_nodes, 1, 2)
    fb_nodes[0].request_announce(2)
    fb_sim.run(until=2.0, max_events=10000)
    assert fb_sim.feedback_signals_generated >= 1
    assert fb_sim.success_feedbacks_delivered >= 1
    assert fb_sim.successful_route_rewards >= 2, fb_sim.successful_route_rewards
    assert fb_nodes[0].tensor.reward_total > 0.0
    assert fb_nodes[1].tensor.reward_total > 0.0
    assert fb_sim.signal_items_dropped_ttl == 0, fb_sim.signal_items_dropped_ttl

    # 12. Processing switches and long-hex JSONL event log.
    no_haar_probe = prepare_wavelet_coefficients(
        np.ones((2, 8, 8), dtype=np.float32), np.random.default_rng(1),
        noise_enabled=False, haar_enabled=False, gaussian_enabled=True,
        gaussian_sigma=4.0, threshold=0.9,
    )
    assert np.array_equal(no_haar_probe, np.ones((2, 8, 8), dtype=np.float32))
    untwisted = translate_embedding(origin, thirty_degrees, twist=False)
    assert np.argmax(untwisted[:, 1]) == 0, "--no-twist must retain inherited direction without rotation"
    with tempfile.NamedTemporaryFile(mode="r+", suffix=".jsonl", delete=False, encoding="utf-8") as log_file:
        log_path = log_file.name
    try:
        log_sim = Simulation(event_log_path=log_path)
        log_nodes: Dict[int, P2PNode] = {}
        log_nodes[0] = P2PNode(0, log_sim, log_nodes)
        log_nodes[1] = P2PNode(1, log_sim, log_nodes)
        log_sim.nodes = log_nodes
        log_signal = Signal("beacon", make_self_embedding(), 0, 3)
        logged = Message(0.1, 0, 1, log_nodes[1], [log_signal], tensor_payload=b"\x01\x02",
                         tensor_codec="test-codec")
        log_sim.schedule(logged)
        log_sim.close_event_log()
        rows = [json.loads(line) for line in open(log_path, encoding="utf-8") if line.strip()]
        assert len(rows) == 1, len(rows)
        assert rows[0]["tensor_hex"] == "0102"
        assert len(rows[0]["from_address"]) == 40 and len(rows[0]["to_address"]) == 40
        assert rows[0]["signals"][0]["origin_address"] == address_for_node(0).hex()
    finally:
        try:
            import os
            os.unlink(log_path)
        except FileNotFoundError:
            pass

    # 5. Parallel topology watermark regression: a target announce can join an
    # already-active stun cluster and therefore make its existing timer
    # topology-sensitive even though pending_discoveries is initially empty.
    # The safe barrier is the existing timer's dispatch time, not the message time.
    watermark_sim = Simulation()
    watermark_nodes: Dict[int, P2PNode] = {}
    watermark_node = P2PNode(0, watermark_sim, watermark_nodes, beacon_interval=10.0)
    watermark_nodes[0] = watermark_node
    watermark_node.stunned_at = 0.0
    watermark_timer = Timer(0.080, watermark_node)
    watermark_node.current_timer = watermark_timer
    target_announce = Signal(
        "announce", make_self_embedding(EMBEDDING_SHAPE), 1, 123,
        target_id=0, origin_address=address_for_node(1),
        target_address=address_for_node(0),
    )
    queued_target_announce = Message(
        0.030, 1, 0, watermark_node, [target_announce],
    )
    watermark_sim._queue = [(0.030, 0, queued_target_announce), (0.080, 1, watermark_timer)]
    assert abs(_parallel_next_topology_time(watermark_sim) - 0.080) < 1e-12

    print("PASS: all-axis Haar round-trip, noise/blur/threshold, sparse node denoise,")
    print("      compressed message tensor decode, exact 20-byte metadata, timer coalescing,")
    print("      neutral unit-weight tensor with --no-learning mode.")
    print("      TTL expiry, success feedback, dynamic joins, announces and mesh fallback.")
    print("      pipeline switches, no-twist behavior, and JSONL event-log hex payloads.")
    print("      parallel announce/stun topology-watermark regression.")


# ---------------------------------------------------------------------------
# Conservative multiprocessing runner
# ---------------------------------------------------------------------------

_PARALLEL_COUNTER_NAMES = (
    "timers_dispatched", "messages_dispatched", "messages_scheduled",
    "protocol_payload_items_scheduled", "protocol_payload_items_dispatched",
    "network_packets_scheduled", "network_packets_delivered",
    "network_signal_copies_scheduled", "network_signal_copies_delivered",
    "message_wavelet_encoded", "message_wavelet_decoded",
    "message_tensor_raw_bytes", "message_tensor_compressed_bytes",
    "message_tensor_reconstructed_mass", "message_wavelet_coefficients",
    "node_wavelet_filter_passes", "node_wavelet_coefficients",
    "node_wavelet_keys_pruned", "signal_items_dropped_as_duplicates",
    "signal_items_dropped_ttl", "signal_items_dropped_hop_limit",
    "feedback_signals_generated", "success_feedbacks_delivered",
    "successful_route_rewards", "beacon_path_rewards", "stale_timers_skipped",
    "peak_queue_size", "candidate_neighbor_checks", "smart_signal_copies",
    "fallback_ripple_signal_copies", "feedback_signal_copies",
    "no_route_signal_drops",
)


def _parallel_message_to_wire(message: Message, source_clock: float) -> dict:
    """Strip the process-local target object from a Message for IPC transport."""
    return {
        "dispatch_time": float(message.dispatch_time),
        "source_clock": float(source_clock),
        "source_address": int(message.source_address),
        "target_address": int(message.target_address),
        "items": message.items,
        "kind": message.kind,
        "join_from": message.join_from,
        "join_to": message.join_to,
        "sent_at": float(message.sent_at),
        "source_eth_address": message.source_eth_address,
        "target_eth_address": message.target_eth_address,
        "tensor_payload": message.tensor_payload,
        "tensor_shape": message.tensor_shape,
        "tensor_codec": message.tensor_codec,
        "tensor_coefficient_count": message.tensor_coefficient_count,
        "tensor_uncompressed_bytes": message.tensor_uncompressed_bytes,
        "decoded_items": message.decoded_items,
        "decoded_tensor": message.decoded_tensor,
    }


def _parallel_message_from_wire(wire: dict, target: "P2PNode", dispatch_time: float) -> Message:
    return Message(
        dispatch_time=float(dispatch_time),
        source_address=int(wire["source_address"]),
        target_address=int(wire["target_address"]),
        target=target,
        items=wire["items"],
        kind=wire["kind"],
        join_from=wire["join_from"],
        join_to=wire["join_to"],
        sent_at=float(wire["sent_at"]),
        source_eth_address=wire["source_eth_address"],
        target_eth_address=wire["target_eth_address"],
        tensor_payload=wire["tensor_payload"],
        tensor_shape=tuple(wire["tensor_shape"]),
        tensor_codec=wire["tensor_codec"],
        tensor_coefficient_count=int(wire["tensor_coefficient_count"]),
        tensor_uncompressed_bytes=int(wire["tensor_uncompressed_bytes"]),
        decoded_items=wire["decoded_items"],
        decoded_tensor=wire["decoded_tensor"],
    )


class _ParallelNodeProxy:
    """Read-only view of a node owned by another process."""
    __slots__ = ("id", "_snapshot")

    def __init__(self, node_id: int, snapshot: dict) -> None:
        self.id = int(node_id)
        self._snapshot = snapshot

    @property
    def address(self) -> bytes:
        return address_for_node(self.id)

    @property
    def pos(self) -> np.ndarray:
        return hash_to_2d_point(self.address)

    @property
    def active(self) -> bool:
        return bool(self._snapshot["active"][self.id])

    @property
    def neighbors(self) -> set:
        return set(self._snapshot["neighbors"][self.id])


class _PartitionSimulationContext:
    """The node-facing subset of Simulation, local to one worker process."""
    def __init__(self, config: dict, worker_id: int, owner_map: dict,
                 node_map: dict, snapshot: dict) -> None:
        self.worker_id = int(worker_id)
        self.owner_map = owner_map
        self.nodes = node_map
        self.snapshot = snapshot
        for name, value in config.items():
            setattr(self, name, value)
        self.now = 0.0
        self.wavelet_rng = np.random.default_rng(
            int(config["wavelet_seed"]) + 1_000_003 * (self.worker_id + 1)
        )
        self._queue = []
        self._order = 0
        self._outbound = {}
        self._log_messages = []
        self._discoveries = []
        self.announce_directions_discovered = set(snapshot["announce_directions"])
        for name in _PARALLEL_COUNTER_NAMES:
            setattr(self, name, 0)

    def schedule(self, event) -> None:
        dispatch = float(event.dispatch_time)
        if dispatch < self.now - 1e-12:
            raise ValueError(
                f"worker {self.worker_id} cannot schedule event in the past: "
                f"{dispatch} < {self.now}"
            )
        if isinstance(event, Message):
            wire = _parallel_message_to_wire(event, self.now)
            self._log_messages.append(wire)
            destination_worker = self.owner_map[int(event.target_address)]
            if destination_worker == self.worker_id:
                event.target = self.nodes[int(event.target_address)]
                self._push(event)
            else:
                self._outbound.setdefault(destination_worker, []).append(wire)
        elif isinstance(event, Timer):
            self._push(event)
        else:
            raise TypeError(f"unsupported event type in worker queue: {type(event).__name__}")

    def _push(self, event) -> None:
        heapq.heappush(self._queue, (float(event.dispatch_time), self._order, event))
        self._order += 1
        self.peak_queue_size = max(self.peak_queue_size, len(self._queue))

    def resolve_announce_discovery(self, origin_id: int, target_id: int) -> None:
        origin_id, target_id = int(origin_id), int(target_id)
        if origin_id == target_id:
            return
        self.announce_directions_discovered.add((origin_id, target_id))
        self._discoveries.append((origin_id, target_id, float(self.now)))
        # Make the new edge visible to this worker's own subsequent same-time
        # events. The coordinator broadcasts the authoritative topology at the
        # next barrier before any worker advances to a later event time.
        _parallel_snapshot_add_edge(self.snapshot, origin_id, target_id)
        for node_id, neighbour_id in ((origin_id, target_id), (target_id, origin_id)):
            local = self.nodes.get(node_id)
            if isinstance(local, P2PNode):
                local.neighbors.add(neighbour_id)


def _parallel_snapshot_add_edge(snapshot: dict, left: int, right: int) -> None:
    if left == right:
        return
    left_set = set(snapshot["neighbors"][left])
    right_set = set(snapshot["neighbors"][right])
    left_set.add(right)
    right_set.add(left)
    snapshot["neighbors"][left] = tuple(sorted(left_set))
    snapshot["neighbors"][right] = tuple(sorted(right_set))


def _parallel_next_topology_time(sim) -> Optional[float]:
    """Earliest local event that a speculative window must not cross."""
    topology_times = []
    for dispatch, _, event in sim._queue:
        if (isinstance(event, Timer)
                and event.target.current_timer is event
                and bool(event.target.pending_discoveries)):
            topology_times.append(float(dispatch))
        elif isinstance(event, Message):
            target_id = int(event.target_address)
            if not any(item.kind == "announce" and item.target_id == target_id
                       for item in event.items):
                continue
            target = event.target
            timer = target.current_timer
            # An announce may join an existing stun cluster, whose timer is
            # anchored to an earlier message. The barrier must be the timer
            # timestamp (not the announce arrival): all events before it may
            # proceed, but no worker may cross it before topology is broadcast.
            if (target.stunned_at is not None and timer is not None
                    and float(dispatch) <= float(timer.dispatch_time) + 1e-12):
                topology_times.append(float(timer.dispatch_time))
    return min(topology_times) if topology_times else None


def _parallel_worker_main(worker_id: int, node_ids: list, node_count: int,
                          owner_map: dict, config: dict, initial_snapshot: dict,
                          beacon_interval: float, max_hops: int,
                          input_queue, output_queue) -> None:
    """Process entry point. Every P2PNode is constructed in exactly one worker."""
    try:
        configure_embedding_device(config["device"])
        snapshot = {
            "active": list(initial_snapshot["active"]),
            "neighbors": [tuple(x) for x in initial_snapshot["neighbors"]],
            "announce_directions": tuple(initial_snapshot["announce_directions"]),
        }
        node_map = {i: _ParallelNodeProxy(i, snapshot) for i in range(node_count)}
        sim = _PartitionSimulationContext(config, worker_id, owner_map, node_map, snapshot)
        local_nodes = {}
        for node_id in node_ids:
            node = P2PNode(
                node_id, sim, node_map, beacon_interval=beacon_interval,
                max_hops=max_hops,
            )
            node.active = bool(snapshot["active"][node_id])
            node.neighbors = set(snapshot["neighbors"][node_id])
            node_map[node_id] = node
            local_nodes[node_id] = node
        sim.nodes = node_map
        output_queue.put(("ready", worker_id, None))

        while True:
            command = input_queue.get()
            operation = command["operation"]
            if operation == "stop":
                node_rows = []
                for node_id, node in sorted(local_nodes.items()):
                    node_rows.append({
                        "id": node_id,
                        "active": bool(node.active),
                        "neighbors": tuple(sorted(node.neighbors)),
                        "signals_accepted": int(node.signals_accepted),
                        "messages_applied": int(node.messages_applied),
                        "beacons_sent": int(node.beacons_sent),
                        "announces_sent": int(node.announces_sent),
                        "timer_fires": int(node.timer_fires),
                        "max_cluster_size": int(node.max_cluster_size),
                        "max_bundle_size_sent": int(node.max_bundle_size_sent),
                        "reward_total": float(node.tensor.reward_total),
                        "stored_combinations": int(node.tensor.stored_combinations),
                        "stored_time_slots": int(node.tensor.stored_time_slots),
                        "time_depth": int(node.time_depth),
                    })
                output_queue.put(("stopped", worker_id, {
                    "counters": {name: getattr(sim, name) for name in _PARALLEL_COUNTER_NAMES},
                    "nodes": node_rows,
                    "pending_events": len(sim._queue),
                    "next_time": sim._queue[0][0] if sim._queue else None,
                }))
                return

            if operation == "enqueue":
                # Receipt-time rebasing. All partitions are at the same
                # conservative frontier when the coordinator transfers a batch,
                # but retain the full formula so independent clocks can be added
                # later without changing the wire contract.
                destination_clock = float(sim.now)
                for wire in command["messages"]:
                    dispatch = destination_clock + (
                        float(wire["dispatch_time"]) - float(wire["source_clock"])
                    )
                    if dispatch < destination_clock - 1e-12:
                        raise RuntimeError(
                            f"causality violation: received event at {dispatch} "
                            f"behind worker clock {destination_clock}"
                        )
                    target_id = int(wire["target_address"])
                    if target_id not in local_nodes:
                        raise RuntimeError(
                            f"worker {worker_id} received node {target_id}, which it does not own"
                        )
                    message = _parallel_message_from_wire(wire, local_nodes[target_id], dispatch)
                    sim._push(message)
                output_queue.put(("enqueued", worker_id, {
                    "next_time": sim._queue[0][0] if sim._queue else None,
                    "next_topology_time": _parallel_next_topology_time(sim),
                    "pending_events": len(sim._queue),
                }))
                continue

            if operation not in {"step", "advance"}:
                raise RuntimeError(f"unknown worker command: {operation}")

            inclusive_step = operation == "step"
            event_time = float(command["time"] if inclusive_step else command["horizon"])
            fresh = command["snapshot"]
            snapshot["active"][:] = fresh["active"]
            snapshot["neighbors"][:] = [tuple(x) for x in fresh["neighbors"]]
            snapshot["announce_directions"] = tuple(fresh["announce_directions"])
            sim.announce_directions_discovered = set(snapshot["announce_directions"])
            for node_id, node in local_nodes.items():
                node.active = bool(snapshot["active"][node_id])
                node.neighbors = set(snapshot["neighbors"][node_id])

            if inclusive_step:
                sim.now = event_time
            # Lifecycle controls are coordinator-owned. They are applied at the
            # same logical timestamp before the node's local queue is drained.
            for control in command.get("controls", []):
                node = local_nodes[int(control["node_id"])]
                if control["operation"] == "activate":
                    node.active = True
                    # The authoritative snapshot already marks the node active;
                    # still ensure its first beacon timer is armed.
                    node.start_beacon_clock()
                elif control["operation"] == "request_announce":
                    node.request_announce(int(control["target_id"]))
                elif control["operation"] == "start_beacon_clock":
                    node.start_beacon_clock()
                else:
                    raise RuntimeError(f"unknown node control: {control['operation']}")

            processed = 0
            last_event_time = None
            event_limit = command.get("event_limit")
            while sim._queue and (
                sim._queue[0][0] <= event_time + 1e-12 if inclusive_step
                else sim._queue[0][0] < event_time - 1e-12
            ) and (event_limit is None or processed < int(event_limit)):
                dispatch, _, event = heapq.heappop(sim._queue)
                if isinstance(event, Timer) and event.target.current_timer is not event:
                    sim.stale_timers_skipped += 1
                    continue
                sim.now = float(event.dispatch_time)
                last_event_time = sim.now
                processed += 1
                if isinstance(event, Timer):
                    sim.timers_dispatched += 1
                    event.target.handle_timer(event)
                elif isinstance(event, Message):
                    sim.messages_dispatched += 1
                    sim.protocol_payload_items_dispatched += len(event.items)
                    if event.target_address != event.source_address:
                        sim.network_packets_delivered += 1
                        sim.network_signal_copies_delivered += len(event.items)
                    event.target.handle_incoming_message(event)
                else:
                    raise RuntimeError(f"unexpected local worker event: {type(event).__name__}")

            # The coordinator advances every local clock to the same conservative
            # frontier. In an advance-window step, topology-sensitive timers are
            # excluded by the coordinator's timer watermark.
            sim.now = event_time
            next_topology_time = _parallel_next_topology_time(sim)
            result = {
                "processed": processed,
                "last_event_time": last_event_time,
                "next_time": sim._queue[0][0] if sim._queue else None,
                "next_topology_time": next_topology_time,
                "pending_events": len(sim._queue),
                "outbound": sim._outbound,
                "log_messages": sim._log_messages,
                "discoveries": sim._discoveries,
            }
            sim._outbound = {}
            sim._log_messages = []
            sim._discoveries = []
            output_queue.put(("stepped", worker_id, result))

    except BaseException:
        output_queue.put(("worker_error", worker_id, traceback.format_exc()))


@dataclass
class _ParallelNodeTensorSummary:
    reward_total: float
    stored_combinations: int
    stored_time_slots: int


@dataclass
class _ParallelNodeSummary:
    id: int
    active: bool
    neighbors: set
    signals_accepted: int
    messages_applied: int
    beacons_sent: int
    announces_sent: int
    timer_fires: int
    max_cluster_size: int
    max_bundle_size_sent: int
    time_depth: int
    tensor: _ParallelNodeTensorSummary


class ParallelSimulationRunner:
    """Run the graph with process-owned nodes and conservative timestamp barriers.

    The coordinator only manages lifecycle/topology events and the global event
    frontier. Each worker owns its node objects and event heap. Network packets
    whose destination is in another partition are collected into one IPC batch
    per destination worker at each timestamp. Workers process local zero-delay
    events immediately; cross-partition events are transferred before the global
    frontier can advance. This is deliberately conservative: it favours causality
    over speculative look-ahead.
    """
    def __init__(self, sim: Simulation, nodes: Dict[int, P2PNode], *,
                 workers: int, seed: int, device: str) -> None:
        self.sim = sim
        self.original_nodes = nodes
        self.node_count = len(nodes)
        self.worker_count = min(int(workers), self.node_count)
        if self.worker_count < 1:
            raise ValueError("parallel simulation requires at least one node")
        if sim.link_delay < 0:
            raise ValueError("parallel workers require a non-negative link delay")
        self.seed = int(seed)
        self.device = str(device)
        self.owner_map = {node_id: node_id % self.worker_count
                          for node_id in range(self.node_count)}
        self.node_ids_by_worker = [
            [node_id for node_id in range(self.node_count)
             if self.owner_map[node_id] == worker_id]
            for worker_id in range(self.worker_count)
        ]
        self.topology = {
            "active": [False] * self.node_count,
            "neighbors": [tuple() for _ in range(self.node_count)],
            "announce_directions": set(),
        }
        self.controls = [[] for _ in range(self.worker_count)]
        self.control_heap = []
        self.fallback_heap = []
        self.fallback_by_pair = {}
        self._order = 0

    def _snapshot(self) -> dict:
        return {
            "active": list(self.topology["active"]),
            "neighbors": [tuple(x) for x in self.topology["neighbors"]],
            "announce_directions": tuple(sorted(self.topology["announce_directions"])),
        }

    def _add_edge(self, left: int, right: int) -> bool:
        if left == right or not (0 <= left < self.node_count and 0 <= right < self.node_count):
            return False
        left_neighbours = set(self.topology["neighbors"][left])
        if right in left_neighbours:
            return False
        right_neighbours = set(self.topology["neighbors"][right])
        left_neighbours.add(right)
        right_neighbours.add(left)
        self.topology["neighbors"][left] = tuple(sorted(left_neighbours))
        self.topology["neighbors"][right] = tuple(sorted(right_neighbours))
        return True

    def _queue_control(self, node_id: int, operation: str, when: float,
                       target_id: Optional[int] = None) -> None:
        command = {"node_id": int(node_id), "operation": operation}
        if target_id is not None:
            command["target_id"] = int(target_id)
        self.controls[self.owner_map[int(node_id)]].append(command)

    def _schedule_fallback(self, left_id: int, right_id: int, when: float) -> None:
        pair = tuple(sorted((int(left_id), int(right_id))))
        existing = self.fallback_by_pair.get(pair)
        if existing is not None and not existing["cancelled"]:
            return
        entry = {
            "pair": pair,
            "dispatch_time": float(when + self.sim.fallback_timeout),
            "cancelled": False,
        }
        self.fallback_by_pair[pair] = entry
        heapq.heappush(self.fallback_heap, (entry["dispatch_time"], self._order, entry))
        self._order += 1
        self.sim.fallback_tasks_scheduled += 1

    def _cancel_fallback(self, left_id: int, right_id: int) -> None:
        pair = tuple(sorted((int(left_id), int(right_id))))
        entry = self.fallback_by_pair.get(pair)
        if entry is not None and not entry["cancelled"]:
            entry["cancelled"] = True
            self.sim.fallback_tasks_cancelled += 1

    def _process_join_intent(self, left_id: int, right_id: int, when: float) -> None:
        if left_id == right_id or left_id not in range(self.node_count) or right_id not in range(self.node_count):
            self.sim.join_case_counts["invalid"] += 1
            return
        left_active = self.topology["active"][left_id]
        right_active = self.topology["active"][right_id]
        left_neighbours = set(self.topology["neighbors"][left_id])
        if not left_active and not right_active:
            self._add_edge(left_id, right_id)
            self.sim.join_case_counts["both_outside_direct"] += 1
            self.topology["active"][left_id] = True
            self.topology["active"][right_id] = True
            self._queue_control(left_id, "activate", when)
            self._queue_control(right_id, "activate", when)
            return
        if left_active != right_active:
            self._add_edge(left_id, right_id)
            self.sim.join_case_counts["active_plus_outsider_direct"] += 1
            self.topology["active"][left_id] = True
            self.topology["active"][right_id] = True
            self._queue_control(left_id, "activate", when)
            self._queue_control(right_id, "activate", when)
            return
        if right_id in left_neighbours:
            self.sim.join_case_counts["already_connected"] += 1
            return
        self.sim.join_case_counts["active_pair_announce"] += 1
        pair = tuple(sorted((left_id, right_id)))
        self.sim.announce_pairs_requested.add(pair)
        self._queue_control(left_id, "request_announce", when, right_id)
        self._queue_control(right_id, "request_announce", when, left_id)
        self._schedule_fallback(left_id, right_id, when)

    def _process_fallback(self, entry: dict, when: float) -> None:
        pair = entry["pair"]
        if entry["cancelled"]:
            self.sim.cancelled_fallbacks_skipped += 1
            return
        self.sim.fallback_tasks_fired += 1
        if self.fallback_by_pair.get(pair) is entry:
            self.fallback_by_pair.pop(pair, None)
        left_id, right_id = pair
        if not self.topology["active"][left_id] or not self.topology["active"][right_id]:
            return
        if not self._add_edge(left_id, right_id):
            return
        self.sim.fallback_links_created += 1
        self.sim.fallback_links.append((float(when), left_id, right_id))
        self._queue_control(left_id, "start_beacon_clock", when)
        self._queue_control(right_id, "start_beacon_clock", when)

    def _process_discoveries(self, discoveries: list) -> None:
        for origin_id, target_id, when in sorted(discoveries, key=lambda x: (x[2], x[0], x[1])):
            origin_id, target_id = int(origin_id), int(target_id)
            if origin_id == target_id:
                continue
            self.topology["announce_directions"].add((origin_id, target_id))
            pair = tuple(sorted((origin_id, target_id)))
            self.sim.announce_pairs_discovered.add(pair)
            if not self.topology["active"][origin_id] or not self.topology["active"][target_id]:
                continue
            if target_id in set(self.topology["neighbors"][origin_id]):
                self._cancel_fallback(origin_id, target_id)
                continue
            if self._add_edge(origin_id, target_id):
                self._cancel_fallback(origin_id, target_id)
                self.sim.announce_links_created += 1
                self.sim.announce_links.append((float(when), origin_id, target_id))

    def run(self, *, until: Optional[float], max_events: int,
            progress_every: int = 0) -> Tuple[int, Dict[int, _ParallelNodeSummary]]:
        context = mp.get_context("spawn")
        input_queues = [context.Queue() for _ in range(self.worker_count)]
        output_queue = context.Queue()
        initial_snapshot = self._snapshot()
        config = {
            "link_delay": self.sim.link_delay,
            "fallback_timeout": self.sim.fallback_timeout,
            "forwarding_threshold": self.sim.forwarding_threshold,
            "coefficient_decay": self.sim.coefficient_decay,
            "learning_enabled": self.sim.learning_enabled,
            "noise_enabled": self.sim.noise_enabled,
            "haar_enabled": self.sim.haar_enabled,
            "gaussian_enabled": self.sim.gaussian_enabled,
            "compression_enabled": self.sim.compression_enabled,
            "ttl_enabled": self.sim.ttl_enabled,
            "twist_enabled": self.sim.twist_enabled,
            "noise_strength": self.sim.noise_strength,
            "gaussian_width": self.sim.gaussian_width,
            "wavelet_threshold": self.sim.wavelet_threshold,
            "node_threshold": self.sim.node_threshold,
            "hebbian_reward": self.sim.hebbian_reward,
            "initial_ttl": self.sim.initial_ttl,
            "wavelet_seed": self.seed,
            "device": self.device,
        }
        processes = []
        processed = 0
        current_time = 0.0
        worker_next = [None] * self.worker_count
        worker_pending = [0] * self.worker_count
        worker_topology_time = [None] * self.worker_count
        last_progress_report = 0
        try:
            for worker_id in range(self.worker_count):
                process = context.Process(
                    target=_parallel_worker_main,
                    args=(worker_id, self.node_ids_by_worker[worker_id], self.node_count,
                          self.owner_map, config, initial_snapshot,
                          self.original_nodes[0].beacon_interval,
                          max(32, self.node_count + 1), input_queues[worker_id], output_queue),
                    name=f"p2p-sim-worker-{worker_id}",
                )
                process.start()
                processes.append(process)
            # Ensure every worker has built its own node objects before the first step.
            ready = set()
            while len(ready) < self.worker_count:
                message = output_queue.get(timeout=120)
                kind, worker_id, payload = message
                if kind == "worker_error":
                    raise RuntimeError(f"worker {worker_id} failed during startup:\n{payload}")
                if kind != "ready":
                    raise RuntimeError(f"unexpected worker startup response: {kind}")
                ready.add(worker_id)

            # The original queue contains the coordinator-owned join intents.
            for dispatch, order, event in self.sim._queue:
                if not isinstance(event, Message) or event.kind != "join_intent":
                    raise RuntimeError("unexpected event in initial coordinator queue")
                heapq.heappush(
                    self.control_heap,
                    (float(dispatch), int(order), "join", event.join_from, event.join_to),
                )
            self.sim._queue.clear()

            last_processed_time = 0.0
            lookahead = min(float(self.sim.link_delay), 0.050)
            while True:
                if processed >= max_events:
                    break

                # A conservative look-ahead window is bounded by link_delay and
                # the minimum 50 ms stun interval. The per-worker topology
                # watermark includes BOTH armed announce-discovery timers and
                # already-queued announce messages whose destination is the
                # announced target. Without the latter, a target announce could
                # be consumed inside a window and arm/fire a discovery timer
                # that the coordinator had not seen at the start of that window.
                # Near the event budget, use timestamp steps with explicit per-
                # worker quotas so a parallel same-time superstep cannot overrun
                # the requested cap simply because several queues drain together.
                can_advance = lookahead > 1e-12 and (max_events - processed) > max(10_000, self.worker_count * 1_000)
                if can_advance:
                    horizon = current_time + lookahead
                    coordinator_times = []
                    if self.control_heap:
                        coordinator_times.append(self.control_heap[0][0])
                    if self.fallback_heap:
                        coordinator_times.append(self.fallback_heap[0][0])
                    topology_times = [x for x in worker_topology_time if x is not None]
                    if coordinator_times:
                        horizon = min(horizon, min(coordinator_times))
                    if topology_times:
                        horizon = min(horizon, min(topology_times))
                    if until is not None:
                        horizon = min(horizon, float(until))

                    if horizon > current_time + 1e-12:
                        snapshot = self._snapshot()
                        for worker_id in range(self.worker_count):
                            input_queues[worker_id].put({
                                "operation": "advance",
                                "horizon": horizon,
                                "snapshot": snapshot,
                                "controls": [],
                            })
                        advanced = {}
                        while len(advanced) < self.worker_count:
                            response = output_queue.get(timeout=120)
                            kind, worker_id, payload = response
                            if kind == "worker_error":
                                raise RuntimeError(f"worker {worker_id} failed:\n{payload}")
                            if kind != "stepped":
                                raise RuntimeError(f"unexpected worker response: {kind}")
                            advanced[worker_id] = payload

                        outbound_by_destination = {}
                        unexpected_discoveries = []
                        for worker_id, result in advanced.items():
                            processed += int(result["processed"])
                            if result["last_event_time"] is not None:
                                last_processed_time = max(last_processed_time, float(result["last_event_time"]))
                            worker_next[worker_id] = result["next_time"]
                            worker_topology_time[worker_id] = result["next_topology_time"]
                            worker_pending[worker_id] = int(result["pending_events"])
                            unexpected_discoveries.extend(result["discoveries"])
                            for destination, records in result["outbound"].items():
                                outbound_by_destination.setdefault(int(destination), []).extend(records)
                            if self.sim._event_log is not None:
                                for wire in result["log_messages"]:
                                    logged = _parallel_message_from_wire(
                                        wire, self.original_nodes[int(wire["target_address"])],
                                        float(wire["dispatch_time"]),
                                    )
                                    self.sim._write_event_log(logged)
                        if unexpected_discoveries:
                            raise RuntimeError(
                                "a worker produced a topology discovery inside a look-ahead window; "
                                "the conservative timer watermark was violated"
                            )

                        enqueued = {}
                        for destination, records in outbound_by_destination.items():
                            for wire in records:
                                if float(wire["dispatch_time"]) < horizon - 1e-12:
                                    raise RuntimeError(
                                        "look-ahead invariant failed: a cross-partition message "
                                        "could arrive before the batch watermark"
                                    )
                                # The source worker's clock at batch transfer is the
                                # window watermark, not the time of the individual send.
                                wire["source_clock"] = horizon
                            input_queues[destination].put({
                                "operation": "enqueue", "messages": records,
                            })
                        while len(enqueued) < len(outbound_by_destination):
                            response = output_queue.get(timeout=120)
                            kind, worker_id, payload = response
                            if kind == "worker_error":
                                raise RuntimeError(f"worker {worker_id} failed during message transfer:\n{payload}")
                            if kind != "enqueued":
                                raise RuntimeError(f"unexpected IPC transfer response: {kind}")
                            enqueued[worker_id] = payload
                        for worker_id, result in enqueued.items():
                            worker_next[worker_id] = result["next_time"]
                            worker_topology_time[worker_id] = result["next_topology_time"]
                            worker_pending[worker_id] = int(result["pending_events"])

                        current_time = horizon
                        if progress_every > 0 and processed - last_progress_report >= progress_every:
                            print(
                                f"[parallel progress] events={processed:,} sim_time={current_time:.3f}s "
                                f"workers={self.worker_count} queue={sum(worker_pending):,}",
                                flush=True,
                            )
                            last_progress_report = (processed // progress_every) * progress_every
                        continue

                candidates = [value for value in worker_next if value is not None]
                if self.control_heap:
                    candidates.append(self.control_heap[0][0])
                if self.fallback_heap:
                    candidates.append(self.fallback_heap[0][0])
                if not candidates:
                    break
                event_time = float(min(candidates))
                if until is not None and event_time > float(until) + 1e-12:
                    break
                if event_time < current_time - 1e-12:
                    raise RuntimeError(
                        f"global event frontier moved backwards: {event_time} < {current_time}"
                    )
                current_time = event_time

                # Lifecycle/topology operations remain coordinator-owned. They
                # are applied at their timestamp before any worker processes t.
                while (self.control_heap and self.control_heap[0][0] <= event_time + 1e-12
                       and processed < max_events):
                    _, _, kind, left_id, right_id = heapq.heappop(self.control_heap)
                    if kind == "join":
                        self.sim.join_intents_dispatched += 1
                        processed += 1
                        last_processed_time = max(last_processed_time, event_time)
                        self._process_join_intent(int(left_id), int(right_id), event_time)
                while (self.fallback_heap and self.fallback_heap[0][0] <= event_time + 1e-12
                       and processed < max_events):
                    _, _, entry = heapq.heappop(self.fallback_heap)
                    if entry["cancelled"]:
                        self.sim.cancelled_fallbacks_skipped += 1
                    else:
                        processed += 1
                        last_processed_time = max(last_processed_time, event_time)
                        self._process_fallback(entry, event_time)

                snapshot = self._snapshot()
                remaining_budget = max(0, max_events - processed)
                eligible_workers = [
                    worker_id for worker_id in range(self.worker_count)
                    if (worker_next[worker_id] is not None
                        and float(worker_next[worker_id]) <= event_time + 1e-12)
                    or bool(self.controls[worker_id])
                ]
                quotas = {worker_id: 0 for worker_id in range(self.worker_count)}
                if eligible_workers and remaining_budget:
                    base_quota, extra_quota = divmod(remaining_budget, len(eligible_workers))
                    for rank, worker_id in enumerate(eligible_workers):
                        quotas[worker_id] = base_quota + (1 if rank < extra_quota else 0)
                for worker_id in range(self.worker_count):
                    input_queues[worker_id].put({
                        "operation": "step",
                        "time": event_time,
                        "snapshot": snapshot,
                        "controls": self.controls[worker_id],
                        "event_limit": quotas[worker_id],
                    })
                    self.controls[worker_id] = []

                stepped = {}
                while len(stepped) < self.worker_count:
                    response = output_queue.get(timeout=120)
                    kind, worker_id, payload = response
                    if kind == "worker_error":
                        raise RuntimeError(f"worker {worker_id} failed:\n{payload}")
                    if kind != "stepped":
                        raise RuntimeError(f"unexpected worker response: {kind}")
                    stepped[worker_id] = payload

                all_discoveries = []
                outbound_by_destination = {}
                for worker_id, result in stepped.items():
                    processed += int(result["processed"])
                    if result["last_event_time"] is not None:
                        last_processed_time = max(last_processed_time, float(result["last_event_time"]))
                    worker_next[worker_id] = result["next_time"]
                    worker_topology_time[worker_id] = result["next_topology_time"]
                    worker_pending[worker_id] = int(result["pending_events"])
                    all_discoveries.extend(result["discoveries"])
                    for destination, records in result["outbound"].items():
                        outbound_by_destination.setdefault(int(destination), []).extend(records)
                    if self.sim._event_log is not None:
                        for wire in result["log_messages"]:
                            logged = _parallel_message_from_wire(
                                wire, self.original_nodes[int(wire["target_address"])],
                                float(wire["dispatch_time"]),
                            )
                            self.sim._write_event_log(logged)

                self._process_discoveries(all_discoveries)

                # One IPC batch per destination partition. Rebase against the
                # receiver's current local clock before another frontier advance.
                enqueued = {}
                for destination, records in outbound_by_destination.items():
                    for wire in records:
                        if float(wire["dispatch_time"]) < event_time - 1e-12:
                            raise RuntimeError(
                                "timestamp barrier invariant failed: outbound message "
                                "is earlier than the sender's clock"
                            )
                        wire["source_clock"] = event_time
                    input_queues[destination].put({
                        "operation": "enqueue", "messages": records,
                    })
                while len(enqueued) < len(outbound_by_destination):
                    response = output_queue.get(timeout=120)
                    kind, worker_id, payload = response
                    if kind == "worker_error":
                        raise RuntimeError(f"worker {worker_id} failed during message transfer:\n{payload}")
                    if kind != "enqueued":
                        raise RuntimeError(f"unexpected IPC transfer response: {kind}")
                    enqueued[worker_id] = payload
                for worker_id, result in enqueued.items():
                    worker_next[worker_id] = result["next_time"]
                    worker_topology_time[worker_id] = result["next_topology_time"]
                    worker_pending[worker_id] = int(result["pending_events"])

                if progress_every > 0 and processed - last_progress_report >= progress_every:
                    print(
                        f"[parallel progress] events={processed:,} sim_time={event_time:.3f}s "
                        f"workers={self.worker_count} queue={sum(worker_pending):,}",
                        flush=True,
                    )
                    last_progress_report = (processed // progress_every) * progress_every

            self.sim.now = last_processed_time
        finally:
            for input_queue in input_queues:
                try:
                    input_queue.put({"operation": "stop"})
                except Exception:
                    pass
            stopped = {}
            deadline = time.monotonic() + 30.0
            while len(stopped) < len(processes) and time.monotonic() < deadline:
                try:
                    response = output_queue.get(timeout=1.0)
                except Exception:
                    if not any(p.is_alive() for p in processes):
                        break
                    continue
                kind, worker_id, payload = response
                if kind == "stopped":
                    stopped[worker_id] = payload
                elif kind == "worker_error":
                    raise RuntimeError(f"worker {worker_id} failed during shutdown:\n{payload}")
            for process in processes:
                process.join(timeout=2.0)
                if process.is_alive():
                    process.terminate()
                    process.join(timeout=2.0)
            for q in input_queues:
                q.close()
            output_queue.close()

        # Merge worker-local metrics and return compact node summaries to the
        # existing reporting code. Sparse tensors never cross the process boundary.
        all_nodes = {}
        total_pending = 0
        for payload in stopped.values():
            total_pending += int(payload["pending_events"])
            for name, value in payload["counters"].items():
                setattr(self.sim, name, getattr(self.sim, name) + value)
            for row in payload["nodes"]:
                all_nodes[int(row["id"])] = _ParallelNodeSummary(
                    id=int(row["id"]),
                    active=bool(row["active"]),
                    neighbors=set(row["neighbors"]),
                    signals_accepted=int(row["signals_accepted"]),
                    messages_applied=int(row["messages_applied"]),
                    beacons_sent=int(row["beacons_sent"]),
                    announces_sent=int(row["announces_sent"]),
                    timer_fires=int(row["timer_fires"]),
                    max_cluster_size=int(row["max_cluster_size"]),
                    max_bundle_size_sent=int(row["max_bundle_size_sent"]),
                    time_depth=int(row["time_depth"]),
                    tensor=_ParallelNodeTensorSummary(
                        reward_total=float(row["reward_total"]),
                        stored_combinations=int(row["stored_combinations"]),
                        stored_time_slots=int(row["stored_time_slots"]),
                    ),
                )
        self.sim.announce_directions_discovered = set(self.topology["announce_directions"])
        self.sim._parallel_pending_events = total_pending + len(self.control_heap) + len(self.fallback_heap)
        self.sim.peak_queue_size = max(
            self.sim.peak_queue_size,
            max((int(payload["counters"]["peak_queue_size"]) for payload in stopped.values()), default=0),
        )
        return processed, all_nodes


def run_parallel_simulation(sim: Simulation, nodes: Dict[int, P2PNode], *,
                            workers: int, seed: int, device: str,
                            until: Optional[float], max_events: int,
                            progress_every: int = 0
                            ) -> Tuple[int, Dict[int, _ParallelNodeSummary]]:
    runner = ParallelSimulationRunner(sim, nodes, workers=workers, seed=seed, device=device)
    return runner.run(until=until, max_events=max_events, progress_every=progress_every)



def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nodes", type=int, default=100, help="candidate node count")
    parser.add_argument("--duration", type=float, default=None,
                        help="simulation-time horizon; default follows last join plus 8 seconds")
    parser.add_argument("--beacon-interval", type=float, default=1.0)
    parser.add_argument("--join-interval", type=float, default=0.05)
    parser.add_argument("--extra-connect-requests", type=int, default=20)
    parser.add_argument("--announce-spacing", type=float, default=0.25)
    parser.add_argument("--forwarding-threshold", type=float, default=0.20,
                        help="forward if coefficient * learned weight >= threshold; otherwise ripple fallback")
    parser.add_argument("--fallback-timeout", type=float, default=5.0,
                        help="simulated seconds before unresolved active-active announces use mesh fallback")
    parser.add_argument("--ttl", type=int, default=DEFAULT_TTL,
                        help="initial per-signal TTL (decremented once at each relay)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-events", type=int, default=750_000)
    parser.add_argument("--workers", type=int, default=1,
                        help="node-owning simulation processes; 1 uses the reference single-process engine")
    parser.add_argument("--progress-every", type=int, default=0,
                        help="print a progress line every N dispatched events (0 disables it)")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto",
                        help="tensor compute device for embeddings and Haar/Gaussian/inverse-Haar; auto prefers CUDA, then Apple MPS, then CPU")
    parser.add_argument("--no-learning", action="store_true",
                        help="disable Hebbian updates and node-tensor denoising; use uniform unit routing weights")
    parser.add_argument("--no-noise", action="store_true", help="disable random background noise before tensor processing")
    parser.add_argument("--no-haar", action="store_true", help="disable Haar decomposition, Gaussian filtering and compression; noise remains enabled unless --no-noise is also supplied")
    parser.add_argument("--no-gaussian", action="store_true", help="disable Gaussian smoothing in Haar coefficient space")
    parser.add_argument("--no-compression", action="store_true", help="send dense float32 Haar coefficients without zlib compression")
    parser.add_argument("--no-ttl", action="store_true", help="ignore TTL checks and do not decrement TTL at relays")
    parser.add_argument("--no-twist", action="store_true", help="shift the fixed embedding stack without rotating inherited directions")
    parser.add_argument("--noise-strength", type=float, default=WAVELET_NOISE_STD,
                        help=f"Gaussian background-noise standard deviation (default: {WAVELET_NOISE_STD})")
    parser.add_argument("--gaussian-width", type=float, default=WAVELET_GAUSSIAN_SIGMA,
                        help=f"Gaussian sigma in wavelet coefficient space (default: {WAVELET_GAUSSIAN_SIGMA})")
    parser.add_argument("--threshold", type=float, default=MESSAGE_WAVELET_THRESHOLD,
                        help=f"soft-threshold for message Haar coefficients (default: {MESSAGE_WAVELET_THRESHOLD})")
    parser.add_argument("--node-threshold", type=float, default=WAVELET_THRESHOLD,
                        help=f"soft-threshold for internal node tensor coefficients (default: {WAVELET_THRESHOLD})")
    parser.add_argument("--hebbian-reward", type=float, default=DEFAULT_HEBBIAN_REWARD,
                        help=f"base success-feedback reward; default preserves beacon=0.12 and feedback=0.35")
    parser.add_argument("--event-log", type=str, default=None,
                        help="write each scheduled message as one JSONL row, with tensor bytes hex-encoded")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be >= 1")
    try:
        configure_embedding_device(args.device)
    except RuntimeError as exc:
        parser.error(str(exc))
    if args.self_test:
        run_self_tests()
        return

    sim, nodes, meta = build_demo(
        args.nodes, args.seed, args.beacon_interval, args.join_interval,
        args.extra_connect_requests, args.announce_spacing, args.forwarding_threshold,
        args.fallback_timeout, args.ttl, learning_enabled=not args.no_learning,
        noise_enabled=not args.no_noise,
        haar_enabled=not args.no_haar,
        gaussian_enabled=not args.no_gaussian and not args.no_haar,
        compression_enabled=not args.no_compression and not args.no_haar,
        ttl_enabled=not args.no_ttl,
        twist_enabled=not args.no_twist,
        noise_strength=args.noise_strength,
        gaussian_width=args.gaussian_width,
        threshold=args.threshold,
        node_threshold=args.node_threshold,
        hebbian_reward=args.hebbian_reward,
        event_log_path=args.event_log,
    )
    duration = args.duration if args.duration is not None else max(10.0, meta["last_intent_time"] + 8.0)
    workers_used = min(args.workers, len(nodes)) if args.workers > 1 else 1
    start = time.perf_counter()
    try:
        if args.workers == 1:
            processed = sim.run(until=duration, max_events=args.max_events,
                                progress_every=args.progress_every)
        else:
            processed, nodes = run_parallel_simulation(
                sim, nodes, workers=workers_used, seed=args.seed,
                device=_EMBEDDING_DEVICE, until=duration,
                max_events=args.max_events, progress_every=args.progress_every,
            )
    finally:
        sim.close_event_log()
    wall = time.perf_counter() - start

    active = [node for node in nodes.values() if node.active]
    active_ids = [node.id for node in active]
    edge_count = sum(len(node.neighbors) for node in active) // 2
    requested = len(sim.announce_pairs_requested)
    discovered = len(sim.announce_pairs_requested & sim.announce_pairs_discovered)
    degrees = [len(node.neighbors) for node in active]
    top_degrees = sorted(((len(node.neighbors), node.id) for node in active), reverse=True)[:5]
    rewards = sum(float(node.tensor.reward_total) for node in active)
    cluster_sizes = [node.max_cluster_size for node in active]
    packet_savings = (100.0 * (1.0 - sim.network_packets_scheduled / sim.network_signal_copies_scheduled)
                      if sim.network_signal_copies_scheduled else 0.0)
    discovery_rate = 100.0 * discovered / requested if requested else 0.0
    routing_attempts = sim.smart_signal_copies + sim.fallback_ripple_signal_copies
    smart_percent = 100.0 * sim.smart_signal_copies / routing_attempts if routing_attempts else 0.0

    print("=" * 78)
    print("EVENT-DRIVEN P2P SIMULATION — DYNAMIC JOINS + BUNDLED SIGNALS")
    print(f"Tensor compute backend:                           {embedding_backend_name()}")
    print(f"Simulation workers:                                {workers_used}")
    print(f"Hebbian learning:                                 {'ENABLED' if sim.learning_enabled else 'DISABLED (uniform unit weights)'}")
    print(f"Tensor pipeline: noise={'on' if sim.noise_enabled else 'off'} ({sim.noise_strength:g}), Haar={'on' if sim.haar_enabled else 'off'}, Gaussian={'on' if sim.gaussian_enabled else 'off'} (sigma={sim.gaussian_width:g}), compression={'on' if sim.compression_enabled else 'off'}, TTL={'on' if sim.ttl_enabled else 'off'}, twist={'on' if sim.twist_enabled else 'off'}")
    if sim.event_log_path:
        print(f"Event log:                                        {sim.event_log_path} ({sim._logged_messages:,} rows)")
    print("=" * 78)
    print(f"Candidates / joined nodes:                       {len(nodes):,} / {len(active):,}")
    print(f"Edges / graph connected:                         {edge_count:,} / {_is_connected(nodes, active_ids)}")
    print(f"Simulation clock / wall time:                    {sim.now:.3f} s / {wall:.2f} s")
    print(f"Requested horizon / processed events:            {duration:.3f} s / {processed:,}")
    if processed >= args.max_events:
        print("WARNING: event budget exhausted before the requested time horizon.")
    print(f"Join intents processed / scheduled:               {sim.join_intents_dispatched:,} / {sim.join_intents_scheduled:,}")
    print(f"Queue pending / peak size:                        {sim.pending_events:,} / {sim.peak_queue_size:,}")
    print("-" * 78)
    print("GRAPH FORMATION")
    print(f"Two outsiders linked directly:                    {sim.join_case_counts['both_outside_direct']:,}")
    print(f"Active node + outsider linked directly:            {sim.join_case_counts['active_plus_outsider_direct']:,}")
    print(f"Active non-neighbour mutual announce requests:    {sim.join_case_counts['active_pair_announce']:,}")
    print(f"Requests where the edge already existed:          {sim.join_case_counts['already_connected']:,}")
    print(f"New links formed after announce delivery:          {sim.announce_links_created:,}")
    print(f"Mesh fallback links after timeout:                 {sim.fallback_links_created:,}")
    print(f"Fallback tasks scheduled / cancelled / fired:      {sim.fallback_tasks_scheduled:,} / {sim.fallback_tasks_cancelled:,} / {sim.fallback_tasks_fired:,}")
    print(f"Cancelled fallback tasks skipped in queue:         {sim.cancelled_fallbacks_skipped:,}")
    print(f"Fallback timeout (simulated seconds):              {sim.fallback_timeout:.2f}")
    print(f"Announce pairs discovered / requested:             {discovered:,} / {requested:,} ({discovery_rate:.1f}%)")
    print("-" * 78)
    print("MESSAGES, BUNDLING, AND ROUTING")
    print(f"Network packets scheduled / delivered:             {sim.network_packets_scheduled:,} / {sim.network_packets_delivered:,}")
    print(f"Signal copies scheduled / delivered:               {sim.network_signal_copies_scheduled:,} / {sim.network_signal_copies_delivered:,}")
    print(f"Packet reduction vs one packet per signal copy:    {packet_savings:.1f}%")
    print(f"Average signal items per outbound packet:          {sim.network_signal_copies_scheduled / max(1, sim.network_packets_scheduled):.2f}")
    print(f"Largest observed outbound bundle:                  {max((n.max_bundle_size_sent for n in active), default=0)} signals")
    tensor_payload_ratio = (100.0 * (1.0 - sim.message_tensor_compressed_bytes / sim.message_tensor_raw_bytes)
                            if sim.message_tensor_raw_bytes else 0.0)
    print(f"Wavelet tensor encoded / decoded:                  {sim.message_wavelet_encoded:,} / {sim.message_wavelet_decoded:,}")
    print(f"Message tensor raw / encoded payload:               {sim.message_tensor_raw_bytes:,} / {sim.message_tensor_compressed_bytes:,} bytes ({tensor_payload_ratio:.1f}% payload reduction; metadata excluded)")
    print(f"Nonzero Haar coefficients transmitted:              {sim.message_wavelet_coefficients:,}")
    print(f"Signal copies threshold-selected / fallback ripple: {sim.smart_signal_copies:,} / {sim.fallback_ripple_signal_copies:,}")
    print(f"Reverse-path success-feedback copies:               {sim.feedback_signal_copies:,}")
    route_label = "threshold-selected" if not sim.learning_enabled else "using learned threshold"
    print(f"Share of data-signal copies {route_label:23s}: {smart_percent:.1f}%")
    print(f"Duplicate/loop copies rejected:                    {sim.signal_items_dropped_as_duplicates:,}")
    print(f"Signals dropped by TTL / hop limit:                 {sim.signal_items_dropped_ttl:,} / {sim.signal_items_dropped_hop_limit:,}")
    print(f"No eligible next hop:                              {sim.no_route_signal_drops:,}")
    print("-" * 78)
    print("LEARNING AND EVENT LOOP")
    print(f"Protocol message envelopes scheduled / delivered: {sim.messages_scheduled:,} / {sim.messages_dispatched:,}")
    print(f"Timer events dispatched / stale timers skipped:   {sim.timers_dispatched:,} / {sim.stale_timers_skipped:,}")
    print(f"Signals accepted / success feedbacks generated:     {sum(n.signals_accepted for n in active):,} / {sim.feedback_signals_generated:,}")
    print(f"Success feedbacks delivered / route rewards:        {sim.success_feedbacks_delivered:,} / {sim.successful_route_rewards:,}")
    print(f"Beacon reverse-path rewards applied:                 {sim.beacon_path_rewards:,}")
    print(f"Total tensor reward updates (beacon + feedback):     {sum(n.messages_applied for n in active):,}")
    print(f"Beacons emitted / announce-origin sends:            {sum(n.beacons_sent for n in active):,} / {sum(n.announces_sent for n in active):,}")
    print(f"Total Hebbian weight increase over baseline:        {rewards:.2f}")
    print(f"Node tensor Haar denoise passes / coefficients:      {sim.node_wavelet_filter_passes:,} / {sim.node_wavelet_coefficients:,}")
    print(f"Node tensor sparse keys pruned by filtering:         {sim.node_wavelet_keys_pruned:,}")
    print(f"Hebbian base reward / message threshold:             {sim.hebbian_reward:g} / {sim.wavelet_threshold:g}")
    print(f"Initial per-signal TTL / enforcement:                {sim.initial_ttl} / {'ON' if sim.ttl_enabled else 'OFF'}")
    print(f"Embedding dimensions (angle x 8-hop history):        {EMBEDDING_SHAPE}")
    print(f"Logical tensor: Source(8x8) x Target(8x8) x Neighbour(8x1) x Time({active[0].time_depth if active else TIME_DEPTH})")
    print(f"Sparse learned combinations / time slots:           {sum(n.tensor.stored_combinations for n in active):,} / {sum(n.tensor.stored_time_slots for n in active):,}")
    print(f"Largest stun cluster (accepted signal items):      {max(cluster_sizes, default=0):,}")
    print("-" * 78)
    if degrees:
        print(f"Degree min / median / mean / max:                  {min(degrees)} / {float(np.median(degrees)):.1f} / {float(np.mean(degrees)):.2f} / {max(degrees)}")
        print("Highest-degree nodes (degree, ID):                " + ", ".join(f"({d}, {i})" for d, i in top_degrees))
    print("=" * 78)


if __name__ == "__main__":
    main()
