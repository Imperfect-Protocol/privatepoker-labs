"""Carpet waves: the swarm's generations and its distributed inheritance.

Each wave spawns drones assembled as mosaics of the living flock, they fly
their episode, survivors keep their memory shard and stay enrolled, and the
dying scream their tensor to whoever hears it. There is no station: the
brain lives only in the drones, it thins with every death, and a total
wipe means total amnesia. First wave dies mostly blind; later waves are
born knowing what the living still remember.
"""
from __future__ import annotations

import numpy as np

from skynet.learning.flight import fly_episode
from skynet.learning.flock import Flock
from skynet.learning.reward import FlightInstruments
from skynet.learning.reward import EpisodeOutcome
from skynet.learning.servo_tensor import ServoPolicy
from skynet.vec import Vec3


class WaveRunner:
    def __init__(self, flock: Flock, instruments: FlightInstruments,
                 *, wave_size: int = 8, row_spacing: float = 40.0,
                 spawn_radius: float = 700.0, school=None,
                 will_factory=None, hazard_rate: float = 0.0) -> None:
        self.flock = flock
        self.instruments = instruments
        self.wave_size = wave_size
        # The carpet must fit inside the sky: shrink row spacing for wide waves.
        self.row_spacing = min(row_spacing, 2.0 * spawn_radius / wave_size)
        self.school = school
        self.will_factory = will_factory
        self.hazard_rate = float(hazard_rate)
        self.history: list = []          # per-wave metric dicts

    def run_wave(self, wave_index: int, rng: np.random.Generator, *,
                 epsilon: float) -> dict:
        survivals = []
        spiral_survivals = []
        deaths = 0
        # A carpet launches together: the whole wave is born at once, each
        # newborn a mosaic of what the PREVIOUS waves' survivors remember,
        # then through flight school if there is one, then joins the living.
        # Within the wave, knowledge moves only by death screams.
        newborns = [ServoPolicy() for _ in range(self.wave_size)]
        wills = [self.will_factory(rng) if self.will_factory else None
                 for _ in newborns]
        for policy, will in zip(newborns, wills):
            self.flock.seed_newborn(policy, rng)
            if self.school is not None:
                # Born already knowing: a copy of the reference graduate's
                # weights. The classroom ran once, before the first wave —
                # or lazily here, on the first newborn ever.
                if self.school.graduate is None:
                    self.school.teach(ServoPolicy())
                policy.tensor.values = {
                    k: v.copy() for k, v in self.school.graduate.tensor.values.items()}
                policy.tensor.epochs = {
                    k: e.copy() for k, e in self.school.graduate.tensor.epochs.items()}
        for policy in newborns:
            self.flock.enroll(policy)
        for slot, policy in enumerate(newborns):
            position = Vec3(0.0, (slot - self.wave_size / 2.0) * self.row_spacing, 50.0)
            hazard = rng.uniform() < self.hazard_rate
            seconds, outcome, _, _ = fly_episode(
                policy, self.instruments, wave_index, rng,
                start_position=position, epsilon_override=epsilon,
                will=wills[slot],
                start_spiral=hazard)
            survivals.append(seconds)
            if hazard:
                spiral_survivals.append(seconds)
            died = outcome in (EpisodeOutcome.CRASHED, EpisodeOutcome.STALLED,
                               EpisodeOutcome.OUT_OF_BOUNDS)
            deaths += died
            if died:
                # The dying drone's shard leaves the brain, then it screams
                # its tensor to whoever is still in the sky to hear it.
                self.flock.reap(policy)
                self.flock.broadcast_obituary(policy, rng)
            # A survivor keeps its shard; it stays enrolled.
        metrics = {
            "wave": wave_index,
            "mean_survival": float(np.mean(survivals)),
            "best_survival": float(np.max(survivals)),
            "deaths": deaths,
            "epsilon": epsilon,
            "survivals": survivals,
            "spiral_survivals": spiral_survivals,
            "flock_size": self.flock.size,
        }
        self.history.append(metrics)
        return metrics
