"""Central configuration for the skynet simulator.

One dataclass owns every knob. Modules receive the config; they do not read
module-level globals, so experiments can run several configs in one process
(self-tests do exactly that).

Pipeline switches mirror the p2p_nerual_spatial ablation flags:
``--no-learning``, ``--no-noise``, ``--no-haar``, ``--no-gaussian``.
"""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
import numpy as np


@dataclass(frozen=True)
class Config:
    # --- world and randomness -------------------------------------------------
    seed: int = 42
    episode_seconds: float = 60.0          # hard stop for one drone episode
    world_half_extent: float = 1000.0      # metres; flat square world, centred at origin

    # --- clocks ---------------------------------------------------------------
    # The event loop (comms, slow tensors) and the physics tick are separate
    # clocks. Physics MUST be a fixed dt; servo dynamics are meaningless
    # at event-loop timescales.
    physics_hz: float = 50.0               # fixed 6-DOF tick
    motion_hz: float = 10.0                # neighbour-set recomputation / swarm motion
    vision_hz: float = 2.0                 # radar raycast + SwarmEye ring-buffer push

    # --- radio ---------------------------------------------------------------
    radio_range: float = 300.0             # metres, hard link boundary
    radio_drop_exponent: float = 2.0       # drop probability grows with (dist/range)^exp
    radio_delay: float = 0.02              # seconds, constant simulated air delay

    # --- stun / coalescing / propagation (ported semantics) -------------------
    initial_ttl: int = 8
    stun_k: float = 0.1                    # sigmoid steepness in stun delay
    stun_midpoint: float = 20.0            # signals in cluster at the sigmoid midpoint
    beacon_interval: float = 1.0

    # --- learning tensors -----------------------------------------------------
    angle_bins: int = 8
    hop_bins: int = 8
    time_depth: int = 8
    hebbian_baseline: float = 0.1
    hebbian_max_weight: float = 5.0
    hebbian_reward: float = 0.35           # slow success reward (route + tactics)
    beacon_reward: float = 0.12
    death_reward_scale: float = 1.5        # anti-Hebbian arm multiplier
    mirror_hop_decay: float = 0.5          # witnessed-reward decay per relay hop
    station_merge_rate: float = 0.01       # slow consolidation against forgetting

    # --- servo learning --------------------------------------------------------
    servo_state_bins: int = 8
    servo_surfaces: int = 4                # aileron, elevator, rudder, throttle
    servo_reward_scale: float = 0.1        # dense per-tick reward scale

    # --- vision ----------------------------------------------------------------
    eye_bins: int = 8                      # azimuth x elevation x range
    eye_max_sources: int = 8               # Source(M) cap on the glued SwarmEye
    eye_range_max: float = 500.0           # metres; range bins are log-spaced to here

    # --- wavelet / chaos pipeline ----------------------------------------------
    noise_strength: float = 0.00001
    gaussian_width: float = 0.30
    message_threshold: float = 0.006
    node_threshold: float = 0.00005

    # --- ablation switches -------------------------------------------------------
    no_learning: bool = False
    no_noise: bool = False
    no_haar: bool = False
    no_gaussian: bool = False
    no_compression: bool = False
    no_ttl: bool = False
    no_twist: bool = False

    # --- waves -------------------------------------------------------------------
    wave_size: int = 8                     # drones per row; carpet rows are wave_size wide
    wave_rows: int = 3                     # rows per wave (the "carpet")
    wave_gap: float = 60.0                 # metres between carpet rows
    row_spacing: float = 40.0              # metres between drones in a row

    def derived(self) -> "Derived":
        return Derived(self)

    def replace(self, **changes) -> "Config":
        return dataclasses.replace(self, **changes)

    def seeded_rng(self) -> np.random.Generator:
        return np.random.default_rng(self.seed)


@dataclass(frozen=True)
class Derived:
    """Quantities computed once from a Config. Owned by the config, not scattered."""
    config: Config
    physics_dt: float = field(init=False)
    motion_dt: float = field(init=False)
    vision_dt: float = field(init=False)

    def __post_init__(self) -> None:
        cfg = self.config
        object.__setattr__(self, "physics_dt", 1.0 / cfg.physics_hz)
        object.__setattr__(self, "motion_dt", 1.0 / cfg.motion_hz)
        object.__setattr__(self, "vision_dt", 1.0 / cfg.vision_hz)
