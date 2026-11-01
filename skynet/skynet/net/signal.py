"""Signals: the logical protocol items. Exact identity, fuzzy payloads later.

Kinds:
- beacon: presence + identity advertisement, flooded.
- announce: "I want to reach target X", flooded toward X.
- feedback: success acknowledgement, flooded back toward the origin.

Identity law (from p2p_nerual_spatial): addresses are exact, never blurred.
A signal's dedup key collapses retries of the same semantic request; beacons
stay unique per origin sequence.
"""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass
import hashlib

BEACON = "beacon"
ANNOUNCE = "announce"
FEEDBACK = "feedback"


def address_for_node(node_id: int) -> bytes:
    """Deterministic 20-byte identity for a node id (test stand-in for ETH)."""
    return hashlib.sha256(f"skynet-node-{node_id}".encode()).digest()[:20]


@dataclass(frozen=True)
class Signal:
    kind: str
    origin_id: int
    origin_address: bytes
    sequence_id: int
    target_id: int | None = None
    target_address: bytes | None = None
    hop_count: int = 0
    ttl: int = 8

    def __post_init__(self) -> None:
        if self.kind not in (BEACON, ANNOUNCE, FEEDBACK):
            raise ValueError(f"unknown signal kind {self.kind!r}")
        if len(self.origin_address) != 20:
            raise ValueError("origin_address must be exactly 20 bytes")
        if self.target_id is not None and len(self.target_address) != 20:
            raise ValueError("target_address must be exactly 20 bytes when present")

    @property
    def signal_id(self) -> bytes:
        target = self.target_address or bytes(20)
        raw = (self.origin_address + target + self.kind.encode()
               + self.sequence_id.to_bytes(8, "big"))
        return hashlib.sha256(raw).digest()[:16]

    @property
    def dedup_key(self) -> tuple:
        """Retries of A->B coalesce; beacons are unique per origin sequence."""
        if self.kind == BEACON:
            return (BEACON, self.origin_address, self.sequence_id)
        return (self.kind, self.origin_address, self.target_address, self.sequence_id)

    def forwarded(self) -> "Signal":
        """One more hop down the wire: hop+1, ttl-1. Identity untouched."""
        return dataclasses.replace(self, hop_count=self.hop_count + 1,
                                   ttl=self.ttl - 1)
