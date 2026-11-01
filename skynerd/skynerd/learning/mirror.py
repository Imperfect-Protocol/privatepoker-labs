"""Mirror learning: obituaries, victories, and witnessed reinforcement.

When a drone dies, its final message is an obituary: the exact fatal trace
(state key + surfaces command of its last decisions) plus its servo weights
at those keys. Witnesses depress their own tensor at the same keys — anti-
Hebbian by proxy — weighted by emotional proximity (hop decay) and state
overlap (the witness is *in* that state right now).

When a drone survives an episode it just flew, it emits a victory trace with
the same format, positive arm. Fear and envy are the same synapse with
opposite polarity.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from skynerd.learning.servo_tensor import ServoPolicy, ELIGIBILITY_DECAY
from skynerd.net.signal import Signal


@dataclass(frozen=True)
class TraceEntry:
    """One fatal/victorious moment: where the brain was, what it commanded."""
    state_key: tuple
    commands: tuple            # 4 surface commands at that moment


@dataclass(frozen=True)
class Trace:
    """The exact protocol payload of a death or a victory."""
    subject_address: bytes
    outcome: str               # "death" | "victory"
    final_state_key: tuple
    final_commands: tuple
    weights: tuple = ()        # sparse (key, value) pairs of the subject's tensor


class MirrorLearner:
    """Witness-side rule: observe outcome -> reinforce/depress traced keys."""

    def __init__(self, policy: ServoPolicy, *, hop_decay: float = 0.5,
                 state_overlap_boost: float = 2.0) -> None:
        self.policy = policy
        self.hop_decay = float(hop_decay)
        self.state_overlap_boost = float(state_overlap_boost)

    @staticmethod
    def _state_distance(my_state_key: tuple, their_state_key: tuple) -> int:
        """How many of the three error bins differ (max 3)."""
        return sum(1 for mine, theirs in zip(my_state_key, their_state_key)
                   if mine != theirs)

    def witness(self, my_state_key: tuple, trace: Trace, hops: int = 0,
                time_epoch: int = 0) -> int:
        """Apply the trace to my own tensor. Returns entries touched."""
        if not self.policy.tensor.learning_enabled:
            return 0
        distance = self._state_distance(my_state_key, trace.final_state_key)
        amount = (self.state_overlap_boost ** (3 - distance)) * (self.hop_decay ** hops)
        if trace.outcome == "death":
            amount = -amount
        self.policy.tensor.reward(trace.final_state_key + (0, 2),  # aileron push
                                  time_epoch=time_epoch, amount=amount)
        self.policy.tensor.reward(trace.final_state_key + (1, 2),  # elevator push
                                  time_epoch=time_epoch, amount=amount)
        return 2


def obituary_from(drone_address: bytes, final_state_key: tuple,
                  final_commands: tuple) -> Trace:
    return Trace(subject_address=drone_address, outcome="death",
                 final_state_key=final_state_key,
                 final_commands=final_commands)


def victory_from(drone_address: bytes, final_state_key: tuple,
                 final_commands: tuple) -> Trace:
    return Trace(subject_address=drone_address, outcome="victory",
                 final_state_key=final_state_key,
                 final_commands=final_commands)
