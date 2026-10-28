"""Event-driven P2P simulator with stun coalescing, learned routing, and bundled messages.

Only Timer and Message events are placed on the priority queue. A Message is a
transport envelope that may carry multiple Signal items (beacons and announces).
Node handlers never create outgoing Messages on receipt: they learn, coalesce the
received signal state, and arm the node's single timer. The timer handler chooses
routes and emits at most one bundled Message per destination neighbour.

Examples:
    python sim_maybe2.py --self-test
    python sim_maybe2.py --nodes 100
    python sim_maybe2.py --nodes 1000 --max-events 1500000
"""
from __future__ import annotations

import argparse
import hashlib
import heapq
import math
import random
import time
from dataclasses import dataclass, replace
from functools import lru_cache
from typing import Dict, Iterable, List, Optional, Tuple, Union

import numpy as np
from scipy.ndimage import gaussian_filter, map_coordinates

try:
    import torch
except ImportError:  # NumPy/SciPy remain a fully supported CPU fallback.
    torch = None


# Use a CUDA device for sufficiently large translation batches when one exists.
# The event queue and sparse Hebbian tables remain on the CPU because they are
# irregular Python data structures; dense embedding transforms are the GPU work.
_EMBEDDING_DEVICE = "cuda" if torch is not None and torch.cuda.is_available() else "cpu"
_GPU_BATCH_THRESHOLD = 8


def configure_embedding_device(preference: str = "auto") -> str:
    """Configure embedding transforms: auto, cpu, or cuda (if CUDA is available)."""
    global _EMBEDDING_DEVICE
    preference = preference.lower()
    if preference not in {"auto", "cpu", "cuda"}:
        raise ValueError("embedding device must be one of: auto, cpu, cuda")
    cuda_available = torch is not None and torch.cuda.is_available()
    if preference == "cuda" and not cuda_available:
        raise RuntimeError("CUDA was requested, but PyTorch reports no CUDA-capable GPU")
    _EMBEDDING_DEVICE = "cuda" if preference == "cuda" or (preference == "auto" and cuda_available) else "cpu"
    return _EMBEDDING_DEVICE


def embedding_backend_name() -> str:
    if _EMBEDDING_DEVICE == "cuda":
        return f"PyTorch CUDA ({torch.cuda.get_device_name(0)})"
    return "NumPy cubic CPU (PyTorch CUDA unavailable or disabled)"


# ---------------------------------------------------------------------------
# Compact fuzzy embeddings: Angle x Hop (8 x 8)
# ---------------------------------------------------------------------------

ANGLE_BINS = 8
HOP_BINS = 8
EMBEDDING_SHAPE = (ANGLE_BINS, HOP_BINS)
TIME_DEPTH = 8
DEFAULT_MAX_RADIUS = 150.0  # retained as a compatibility/config label; hops are unit length


def hash_to_2d_point(address: object) -> np.ndarray:
    """Deterministically map an address into a synthetic Cartesian plane.

    These coordinates provide a reproducible directional hint only; they do not
    describe graph topology or predict how many network hops separate two nodes.
    """
    digest = hashlib.sha256(str(address).encode("utf-8")).hexdigest()
    x = (int(digest[:8], 16) % 10000) / 100.0
    y = (int(digest[8:16], 16) % 10000) / 100.0
    return np.array([x, y], dtype=np.float64)


def angle_bin(delta: np.ndarray, bins: int = ANGLE_BINS) -> int:
    """Quantize a Cartesian direction into a circular angle bin."""
    delta = np.asarray(delta, dtype=np.float64)
    if float(np.linalg.norm(delta)) < 1e-12:
        return 0
    angle = math.atan2(float(delta[1]), float(delta[0])) % (2.0 * math.pi)
    return int((angle / (2.0 * math.pi)) * bins) % bins


def fuzzy_angle_weights(delta: np.ndarray, bins: int = ANGLE_BINS) -> List[Tuple[int, float]]:
    """Return a softly interpolated pair of adjacent circular angle bins."""
    delta = np.asarray(delta, dtype=np.float64)
    if float(np.linalg.norm(delta)) < 1e-12:
        return [(0, 1.0)]
    angle = math.atan2(float(delta[1]), float(delta[0])) % (2.0 * math.pi)
    coordinate = angle * bins / (2.0 * math.pi)
    lower_float = math.floor(coordinate)
    lower = int(lower_float) % bins
    fraction = coordinate - lower_float
    upper = (lower + 1) % bins
    if fraction < 1e-8:
        return [(lower, 1.0)]
    return [(lower, 1.0 - fraction), (upper, fraction)]


def make_self_embedding(shape: Tuple[int, int] = EMBEDDING_SHAPE) -> np.ndarray:
    """A beacon begins as a dot at angle-bin 0, hop 0; all other cells are zero."""
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


def _translate_embedding_batch_numpy(sources: np.ndarray, offset: np.ndarray) -> np.ndarray:
    """Fast CPU implementation. All embeddings in a packet share one link offset."""
    batch, angle_count, hop_count = sources.shape
    out = np.zeros((batch, angle_count, hop_count), dtype=np.float32)
    norm = float(np.linalg.norm(offset))
    if norm > 1e-12:
        direction = np.asarray(offset, dtype=np.float64) / norm
    else:
        direction = np.array([1.0, 0.0], dtype=np.float64)
    offset_angle = math.atan2(float(direction[1]), float(direction[0])) % (2.0 * math.pi)
    old_angle = np.arange(angle_count, dtype=np.float64) * (2.0 * math.pi / angle_count)
    output_angle = old_angle
    for old_hop in range(hop_count - 1):
        profile = np.maximum(sources[:, :, old_hop], 0.0)
        mass = np.sum(profile, axis=1, dtype=np.float64).astype(np.float32)
        new_hop = old_hop + 1
        if old_hop == 0:
            coordinate = offset_angle * angle_count / (2.0 * math.pi)
            lower_float = math.floor(coordinate)
            lower = int(lower_float) % angle_count
            fraction = float(coordinate - lower_float)
            out[:, lower, new_hop] += mass * (1.0 - fraction)
            out[:, (lower + 1) % angle_count, new_hop] += mass * fraction
            continue

        radius = float(new_hop)  # unit edge lengths; hop depth is our fuzzy radius proxy
        out_x = radius * np.cos(output_angle) - direction[0]
        out_y = radius * np.sin(output_angle) - direction[1]
        old_theta = np.arctan2(out_y, out_x) % (2.0 * math.pi)
        sample_coords = old_theta * angle_count / (2.0 * math.pi)
        warped = _cubic_periodic_sample_numpy(profile, sample_coords)
        warped = np.maximum(warped, 0.0)
        warped_mass = np.sum(warped, axis=1, keepdims=True, dtype=np.float64)
        scale = np.divide(mass[:, None], warped_mass, out=np.zeros_like(warped_mass), where=warped_mass > 1e-12)
        out[:, :, new_hop] = (warped * scale).astype(np.float32, copy=False)

    # Small, periodic angular Gaussian blur: make the compact embedding fuzzy/smooth.
    sigma = 0.42
    taps = np.arange(-2, 3, dtype=np.float64)
    kernel = np.exp(-0.5 * (taps / sigma) ** 2)
    kernel /= kernel.sum()
    blurred = np.zeros_like(out)
    for tap, weight in zip(taps.astype(int), kernel):
        blurred += np.roll(out, int(tap), axis=1) * np.float32(weight)
    blurred = np.maximum(blurred, 0.0)
    blurred[:, 0, 0] = 1.0  # the receiving node is always at its own origin
    return blurred


def _translate_embedding_batch_torch(sources: np.ndarray, offset: np.ndarray,
                                     device_name: Optional[str] = None) -> np.ndarray:
    """Batched PyTorch transform; CUDA is intended for large bundles, CPU for backend tests."""
    device = torch.device(device_name or _EMBEDDING_DEVICE)
    src = torch.as_tensor(sources, dtype=torch.float32, device=device).clamp_min(0.0)
    batch, angle_count, hop_count = src.shape
    out = torch.zeros_like(src)
    norm = float(np.linalg.norm(offset))
    direction = (np.asarray(offset, dtype=np.float64) / norm if norm > 1e-12
                 else np.array([1.0, 0.0], dtype=np.float64))
    offset_angle = math.atan2(float(direction[1]), float(direction[0])) % (2.0 * math.pi)
    old_angle_np = np.arange(angle_count, dtype=np.float64) * (2.0 * math.pi / angle_count)
    old_angle = torch.as_tensor(old_angle_np, dtype=src.dtype, device=device)
    for old_hop in range(hop_count - 1):
        profile = src[:, :, old_hop]
        mass = profile.sum(dim=1)
        new_hop = old_hop + 1
        if old_hop == 0:
            coordinate = offset_angle * angle_count / (2.0 * math.pi)
            lower_float = math.floor(coordinate)
            lower = int(lower_float) % angle_count
            fraction = float(coordinate - lower_float)
            out[:, lower, new_hop] += mass * (1.0 - fraction)
            out[:, (lower + 1) % angle_count, new_hop] += mass * fraction
            continue
        radius = float(new_hop)
        out_x = radius * torch.cos(old_angle) - float(direction[0])
        out_y = radius * torch.sin(old_angle) - float(direction[1])
        old_theta = torch.remainder(torch.atan2(out_y, out_x), 2.0 * math.pi)
        sample_coords = old_theta * angle_count / (2.0 * math.pi)
        warped = _cubic_periodic_sample_torch(profile, sample_coords).clamp_min(0.0)
        warped_mass = warped.sum(dim=1, keepdim=True)
        scale = torch.where(warped_mass > 1e-12, mass[:, None] / warped_mass.clamp_min(1e-12), 0.0)
        out[:, :, new_hop] = warped * scale

    sigma = 0.42
    taps = torch.arange(-2, 3, dtype=src.dtype, device=device)
    kernel = torch.exp(-0.5 * (taps / sigma) ** 2)
    kernel = kernel / kernel.sum()
    blurred = torch.zeros_like(out)
    for tap, weight in zip(range(-2, 3), kernel):
        blurred = blurred + torch.roll(out, shifts=tap, dims=1) * weight
    blurred = blurred.clamp_min(0.0)
    blurred[:, 0, 0] = 1.0
    return blurred.detach().cpu().numpy().astype(np.float32, copy=False)


def translate_embedding_batch(embeddings: Iterable[np.ndarray], source_to_target: np.ndarray,
                              max_radius: float = DEFAULT_MAX_RADIUS) -> np.ndarray:
    """Translate a bundle's embeddings at once, using CUDA for large batches if available.

    Each incoming embedding is angle x hop (8 x 8). Every existing hop layer is
    re-expressed in the receiver's frame; hop depth advances by one, and the
    receiver's own origin is inserted at [angle=0, hop=0]. Distance is intentionally
    compacted away: each graph edge counts as one unit hop. Cubic periodic angular
    interpolation and Gaussian blur keep the embedding fuzzy rather than exact.
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
    # tiny packets are faster on CPU due to CUDA launch and transfer overhead
    if _EMBEDDING_DEVICE == "cuda" and len(batch) >= _GPU_BATCH_THRESHOLD:
        return _translate_embedding_batch_torch(batch, offset)
    return _translate_embedding_batch_numpy(batch, offset)


def translate_embedding(embedding: np.ndarray, source_to_target: np.ndarray,
                        max_radius: float = DEFAULT_MAX_RADIUS) -> np.ndarray:
    """Single-embedding convenience wrapper around the batched translator."""
    return translate_embedding_batch([embedding], source_to_target, max_radius)[0]


class SparseHebbianTensor:
    """Sparse logical Source(8x8) x Target(8x8) x Neighbour(8x1) x Time(8).

    Keys are (source_angle, source_hop, target_angle, target_hop, neighbour_angle).
    Only observed combinations allocate an 8-slot time-history vector.
    """
    def __init__(self, time_depth: int = TIME_DEPTH, baseline: float = 0.1,
                 max_weight: float = 5.0) -> None:
        self.embedding_shape = EMBEDDING_SHAPE
        self.neighbour_shape = (ANGLE_BINS, 1)
        self.time_depth = int(time_depth)
        self.baseline = float(baseline)
        self.max_weight = float(max_weight)
        self.values: Dict[Tuple[int, ...], np.ndarray] = {}
        self.epochs: Dict[Tuple[int, ...], np.ndarray] = {}
        self.reward_total = 0.0

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
        if amount <= 0.0:
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

    def score(self, source: Tuple[int, int], target: Tuple[int, int],
              neighbour: int, time_epoch: int, decay: float = 0.97) -> float:
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
    """One beacon or A->B announce, bundled with other signals in a Message."""
    kind: str
    embedding: np.ndarray
    origin_id: int
    sequence_id: int
    target_id: Optional[int] = None
    hop_count: int = 0
    visited: Tuple[int, ...] = ()
    coefficient: float = 1.0
    route_attempt: int = 0

    @property
    def key(self) -> Tuple[object, ...]:
        # Different copies/retries of A->B coalesce as the same logical announce.
        if self.kind == "announce":
            return ("announce", self.origin_id, self.target_id)
        return (self.kind, self.origin_id, self.sequence_id)


@dataclass(eq=False)
class Message:
    """A queue event. Protocol Messages carry a bundle; join_intent is exogenous."""
    dispatch_time: float
    source_address: int
    target_address: int
    target: "P2PNode"
    items: List[Signal]
    kind: str = "bundle"
    join_from: Optional[int] = None
    join_to: Optional[int] = None
    sent_at: float = 0.0


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
                 coefficient_decay: float = 0.97, fallback_timeout: float = 5.0) -> None:
        if link_delay < 0:
            raise ValueError("link_delay must be non-negative")
        if fallback_timeout <= 0:
            raise ValueError("fallback_timeout must be positive")
        self.now = 0.0
        self.link_delay = float(link_delay)
        self.fallback_timeout = float(fallback_timeout)
        self.forwarding_threshold = float(forwarding_threshold)
        self.coefficient_decay = float(coefficient_decay)
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
        self.signal_items_dropped_as_duplicates = 0
        self.stale_timers_skipped = 0
        self.join_intents_scheduled = 0
        self.join_intents_dispatched = 0
        self.peak_queue_size = 0
        self.candidate_neighbor_checks = 0
        self.smart_signal_copies = 0
        self.fallback_ripple_signal_copies = 0
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

    @property
    def pending_events(self) -> int:
        return len(self._queue)

    def run(self, *, until: Optional[float] = None, max_events: int = 750_000) -> int:
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
        self.pos = hash_to_2d_point(self.id)
        self.neighbors: set[int] = set()
        self.active = False

        if (angle_bins, hop_bins, tensor_bins) != (8, 8, 8):
            raise ValueError("the compact embedding is fixed at angle x hop = 8x8, with 8 neighbour angles")
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
                                          max_weight=5.0)

        # Exactly one live timer slot; all nodes begin idle and unstunned.
        self.current_timer: Optional[Timer] = None
        self.stunned_at: Optional[float] = None
        self.messages_in_cluster = 0
        self.last_beacon_at: Optional[float] = None

        # Node-local coalescing. A->B announces have one semantic key regardless
        # of their sequence/retry copies. Beacons are unique per origin sequence.
        self.pending_signals: Dict[Tuple[object, ...], Signal] = {}
        self.pending_discoveries: set[int] = set()
        self.seen_signal_state: Dict[Tuple[object, ...], Tuple[int, int]] = {}
        self.awaiting_announces: Dict[int, Tuple[float, int, int]] = {}  # target -> (deadline, attempt, seq)
        self.next_beacon_sequence = 0
        self.next_announce_sequence = 0

        self.messages_received = 0       # protocol packets
        self.signal_items_received = 0
        self.messages_applied = 0        # unique/improved signal items applied
        self.messages_forwarded = 0      # signal copies routed by timer
        self.packets_sent = 0             # outgoing neighbour bundles
        self.beacons_sent = 0
        self.announces_sent = 0
        self.timer_fires = 0
        self.max_cluster_size = 0
        self.last_stun_delay = 0.0
        self.max_bundle_size_sent = 0

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
                        self.next_announce_sequence, target_id=int(target_id))
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

        self.messages_received += 1
        self.signal_items_received += len(message.items)
        source_pos = hash_to_2d_point(message.source_address)
        offset = self.pos - source_pos  # sender/source -> receiving/target vector

        # Entry layout: [signal, is_first, better_path, is_self_loopback, local_embedding].
        prepared: List[list] = []
        for received in message.items:
            if received.kind not in ("beacon", "announce"):
                continue
            if received.hop_count > self.max_hops:
                self.sim.signal_items_dropped_as_duplicates += 1
                continue
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
        # share a coordinate-frame offset. Transform them as one batch (CUDA for large
        # bundles when available; vectorized NumPy cubic interpolation otherwise).
        translate_indices = [i for i, entry in enumerate(prepared) if not entry[3]]
        if translate_indices:
            translated = translate_embedding_batch(
                [prepared[i][0].embedding for i in translate_indices], offset, self.max_radius
            )
            for item_index, local_embedding in zip(translate_indices, translated):
                prepared[item_index][4] = local_embedding

        accepted = 0
        for received, is_first, better_path, is_self_loopback, local_embedding in prepared:
            assert local_embedding is not None
            localized = replace(received, embedding=local_embedding)
            # Reward only on first receipt / improved path, not merely an announce retry.
            if is_first or better_path:
                self._apply_embedding_reward(localized, message.source_address)
                self.messages_applied += 1

            if localized.kind == "announce" and localized.target_id == self.id:
                self.pending_discoveries.add(localized.origin_id)
            else:
                # Self-beacon loopback is for local learning, never a second relay copy.
                local_beacon_loopback = (
                    is_self_loopback
                    and localized.kind == "beacon"
                    and localized.origin_id == self.id
                    and localized.hop_count == 0
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
        target_id = (signal.target_id if signal.kind == "announce" and signal.target_id is not None
                     else signal.origin_id)
        delta = hash_to_2d_point(target_id) - self.pos
        return [((angle, 0), weight) for angle, weight in fuzzy_angle_weights(delta, self.angle_bins)]

    def _target_signature(self, signal: Signal) -> Tuple[int, int]:
        return max(self._target_signature_weights(signal), key=lambda item: item[1])[0]

    def _neighbour_signature_weights(self, neighbour_id: int) -> List[Tuple[int, float]]:
        """Neighbour is Angle x 1: only its fuzzy direction bin is stored."""
        neighbour_delta = self.nodes[neighbour_id].pos - self.pos
        return fuzzy_angle_weights(neighbour_delta, self.tensor_bins)

    def _neighbour_signature(self, neighbour_id: int) -> int:
        return max(self._neighbour_signature_weights(neighbour_id), key=lambda item: item[1])[0]

    def _apply_embedding_reward(self, signal: Signal, source_address: int) -> None:
        """Hebbian-reward fuzzy Source x Target x Neighbour x Time entries."""
        source_bins = self._source_signature_weights(signal)
        target_bins = self._target_signature_weights(signal)
        neighbour_bins = self._neighbour_signature_weights(source_address)
        base = 0.15 if signal.kind == "beacon" else 0.25
        reward = base * signal.coefficient / (1.0 + 0.25 * max(0, signal.hop_count))
        for source, sw in source_bins:
            for target, tw in target_bins:
                for neighbour, nw in neighbour_bins:
                    self.tensor.reward(
                        source, target, neighbour, time_epoch=self.timer_fires,
                        amount=reward * sw * tw * nw,
                    )

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

    def _schedule_bundle(self, destination_id: int, items: List[Signal]) -> None:
        if not items or destination_id not in self.nodes:
            return
        # A single Message is created for each destination in this timer firing.
        message = Message(
            dispatch_time=self.sim.now + (0.0 if destination_id == self.id else self.sim.link_delay),
            source_address=self.id,
            target_address=destination_id,
            target=self.nodes[destination_id],
            items=items,
            kind="bundle",
            sent_at=self.sim.now,
        )
        self.sim.schedule(message)
        self.sim.messages_scheduled += 1
        self.sim.protocol_payload_items_scheduled += len(items)
        if destination_id != self.id:
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

        # Resolve target arrivals only here; receiving a Message itself changes no topology.
        for origin_id in sorted(self.pending_discoveries):
            self.sim.resolve_announce_discovery(origin_id, self.id)
        self.pending_discoveries.clear()

        work = list(self.pending_signals.values())
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
                continue
            destinations, smart = self._choose_destinations(signal)
            for destination_id in destinations:
                forwarded = replace(
                    signal,
                    embedding=signal.embedding.copy(),
                    hop_count=signal.hop_count + 1,
                    visited=signal.visited + (self.id,),
                    coefficient=signal.coefficient * self.sim.coefficient_decay,
                )
                dest_items = by_destination.setdefault(destination_id, {})
                previous = dest_items.get(forwarded.key)
                dest_items[forwarded.key] = forwarded if previous is None else self._prefer_bundled_item(previous, forwarded)
                self.messages_forwarded += 1
                if smart:
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
               fallback_timeout: float = 5.0
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
    sim = Simulation(forwarding_threshold=forwarding_threshold, fallback_timeout=fallback_timeout)
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

    # 3. Compact angle x hop embeddings translate smoothly, and batching agrees with singles.
    origin = make_self_embedding(EMBEDDING_SHAPE)
    offset = hash_to_2d_point(1) - hash_to_2d_point(0)
    single = translate_embedding(origin, offset)
    batch = translate_embedding_batch([origin, origin], offset)
    assert single.shape == (8, 8) and batch.shape == (2, 8, 8)
    assert np.all(single >= 0.0) and np.all(np.isfinite(single))
    assert np.allclose(single, batch[0], atol=1e-5)
    assert np.allclose(batch[0], batch[1], atol=1e-5)
    assert abs(float(single[0, 0]) - 1.0) < 1e-6
    if torch is not None:
        torch_batch = _translate_embedding_batch_torch(np.stack([origin, origin]), offset,
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

    # 6. Dynamic topology and active-active announce discovery end-to-end.
    sim3, nodes3, meta = build_demo(12, seed=17, beacon_interval=0.4,
                                     join_interval=0.03, extra_connect_requests=5,
                                     announce_spacing=0.18)
    sim3.run(until=meta["last_intent_time"] + 10.0, max_events=300_000)
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

    print("PASS: priority ordering, first-message stun anchor, timer replacement/cancel, ")
    print("      announce A->B coalescing, one bundle per neighbour, dynamic joins, ")
    print("      connected growth, announce discovery, and cancellable mesh fallback.")


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
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-events", type=int, default=750_000)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto",
                        help="embedding transform device; auto uses CUDA when PyTorch sees a GPU")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
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
        args.fallback_timeout,
    )
    duration = args.duration if args.duration is not None else max(10.0, meta["last_intent_time"] + 8.0)
    start = time.perf_counter()
    processed = sim.run(until=duration, max_events=args.max_events)
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
    print(f"Embedding transform backend:                      {embedding_backend_name()}")
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
    print(f"Signal copies on learned routes / fallback ripple: {sim.smart_signal_copies:,} / {sim.fallback_ripple_signal_copies:,}")
    print(f"Share of routed copies using learned threshold:    {smart_percent:.1f}%")
    print(f"Duplicate/loop copies rejected:                    {sim.signal_items_dropped_as_duplicates:,}")
    print(f"No eligible next hop:                              {sim.no_route_signal_drops:,}")
    print("-" * 78)
    print("LEARNING AND EVENT LOOP")
    print(f"Protocol message envelopes scheduled / delivered: {sim.messages_scheduled:,} / {sim.messages_dispatched:,}")
    print(f"Timer events dispatched / stale timers skipped:   {sim.timers_dispatched:,} / {sim.stale_timers_skipped:,}")
    print(f"Unique/improved signal items applied to tensors:   {sum(n.messages_applied for n in active):,}")
    print(f"Beacons emitted / announce-origin sends:           {sum(n.beacons_sent for n in active):,} / {sum(n.announces_sent for n in active):,}")
    print(f"Total Hebbian weight increase over baseline:       {rewards:.2f}")
    print(f"Embedding dimensions (angle x hop):                 {EMBEDDING_SHAPE}")
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
