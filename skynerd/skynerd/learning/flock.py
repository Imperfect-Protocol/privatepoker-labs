"""The flock: a distributed brain with no safe place to keep it.

There is no hippocampus at the rear. Every living drone IS a shard of the
swarm's memory. Three p2p laws replace the station:

- Birth as mosaic: a newborn asks a few random living drones what they
  know and keeps roughly half of each answer. No two births are alike,
  and knowledge no living drone carries is never passed on.
- Death screams: a dying drone broadcasts its tensor; each living peer —
  airborne or waiting on the runway — hears it with some probability and
  absorbs a lossy trace. Whatever lived only in the dying drone mostly
  dies with it.
- Survivors keep their shard and stay in the flock, teaching later births.

If the whole flock dies, the brain is empty again: a total wipe is total
amnesia, and the next newborn starts from hardware calibration alone.
That is the honest p2p price, and the reason waves must not all die.
"""
from __future__ import annotations

import numpy as np


class Flock:
    """The living swarm as a distributed, lossy, mortal memory."""

    def __init__(self, *, witness_prob: float = 0.9, donor_count: int = 12,
                 teach_rate: float = 0.8, teach_fraction: float = 0.75,
                 scream_rate: float = 0.3) -> None:
        self.witness_prob = float(witness_prob)   # chance a peer hears a death scream
        self.donor_count = int(donor_count)       # peers a newborn asks
        self.teach_rate = float(teach_rate)       # blend strength per donor answer
        self.teach_fraction = float(teach_fraction)
        self.scream_rate = float(scream_rate)     # how much of a scream a witness absorbs
        self.members: list = []                   # living drones, each a memory shard
        self.births = 0
        self.obituaries_heard = 0

    @property
    def size(self) -> int:
        return len(self.members)

    def seed_newborn(self, policy, rng: np.random.Generator) -> None:
        """Assemble a newborn from random living donors. Empty flock = blank birth."""
        self.births += 1
        if not self.members:
            return
        count = min(self.donor_count, len(self.members))
        donors = rng.choice(len(self.members), size=count, replace=False)
        for index in donors:
            self._teach(policy, self.members[int(index)], rng, self.teach_rate)

    def enroll(self, policy) -> None:
        """A drone joins the living: its shard is now part of the brain."""
        self.members.append(policy)

    def reap(self, policy) -> None:
        """A drone died: its piece of the brain dies with it."""
        if policy in self.members:
            self.members.remove(policy)

    def broadcast_obituary(self, dying, rng: np.random.Generator) -> None:
        """The dying drone screams its tensor; each living peer may hear it.

        The scream is a small lossy trace — a few keys, weakly blended —
        not a transplant. A dead drone must not rewrite the living.
        """
        for member in self.members:
            if rng.uniform() < self.witness_prob:
                self._teach(member, dying, rng, self.scream_rate, 0.25)
                self.obituaries_heard += 1

    def _teach(self, recipient, donor, rng: np.random.Generator, rate: float,
               fraction: float | None = None) -> None:
        """Blend a random subset of the donor's keys into the recipient."""
        keys = list(donor.tensor.values)
        if not keys:
            return
        share = self.teach_fraction if fraction is None else fraction
        count = max(1, int(len(keys) * share))
        chosen = rng.choice(len(keys), size=count, replace=False)
        for index in chosen:
            self._merge_key(recipient.tensor, donor.tensor,
                            keys[int(index)], rate, rng)

    @staticmethod
    def _merge_key(recipient, donor, key, rate: float,
                   rng: np.random.Generator) -> None:
        """Newest-slot merge of one key, station law without the station."""
        epochs = donor.epochs[key]
        newest = int(np.argmax(epochs))
        if key in recipient.values:
            recipient.values[key][0] += rate * (donor.values[key][newest]
                                                - recipient.values[key][0])
            recipient.epochs[key][0] = 0
            return
        recipient.values[key] = donor.values[key].copy()
        recipient.epochs[key] = donor.epochs[key].copy()
        recipient.updates_since_filter += 1
