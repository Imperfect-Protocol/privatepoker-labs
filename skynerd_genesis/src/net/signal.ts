/**
 * Signals: the logical protocol items. Exact identity, fuzzy payloads later.
 * Ported from `skynerd/net/signal.py` (Genesis Milestone 1).
 *
 * Kinds:
 * - beacon: presence + identity advertisement, flooded.
 * - announce: "I want to reach target X", flooded toward X.
 * - feedback: success acknowledgement, flooded back toward the origin.
 *
 * Identity law (from p2p_nerual_spatial): addresses are exact, never blurred.
 * A signal's dedup key collapses retries of the same semantic request; beacons
 * stay unique per origin sequence.
 */

export const BEACON = "beacon";
export const ANNOUNCE = "announce";
export const FEEDBACK = "feedback";
export const SIGNAL_KINDS = [BEACON, ANNOUNCE, FEEDBACK] as const;
export type SignalKind = (typeof SIGNAL_KINDS)[number];

export const ADDRESS_LEN = 20; // 20-byte identity (test stand-in for ETH)
export const INITIAL_TTL = 8;

/**
 * Deterministic identity for a node id. The Python reference uses
 * sha256("skynerd-node-<id>")[0:20]; inject the hash here so the core stays
 * platform-neutral (node:crypto in tests, WebCrypto in the browser).
 */
export function addressForNode(nodeId: number, sha256: (data: string) => Uint8Array): Uint8Array {
  return sha256(`skynerd-node-${nodeId}`).slice(0, ADDRESS_LEN);
}

export class Signal {
  readonly kind: SignalKind;
  readonly originId: number;
  readonly originAddress: Uint8Array;
  readonly sequenceId: number;
  readonly targetId: number | null;
  readonly targetAddress: Uint8Array | null;
  readonly hopCount: number;
  readonly ttl: number;

  constructor(fields: {
    kind: SignalKind;
    originId: number;
    originAddress: Uint8Array;
    sequenceId: number;
    targetId?: number | null;
    targetAddress?: Uint8Array | null;
    hopCount?: number;
    ttl?: number;
  }) {
    if (!SIGNAL_KINDS.includes(fields.kind)) {
      throw new Error(`unknown signal kind: ${fields.kind}`);
    }
    if (fields.kind !== BEACON) {
      if (fields.targetId === null || fields.targetId === undefined) {
        throw new Error("announce/feedback signals need a target");
      }
      if (!fields.targetAddress) {
        throw new Error("announce/feedback signals need a target address");
      }
    }
    this.kind = fields.kind;
    this.originId = fields.originId;
    this.originAddress = fields.originAddress;
    this.sequenceId = fields.sequenceId;
    this.targetId = fields.targetId ?? null;
    this.targetAddress = fields.targetAddress ?? null;
    this.hopCount = fields.hopCount ?? 0;
    this.ttl = fields.ttl ?? INITIAL_TTL;
  }

  /** Identity of the semantic request; identical retries collapse on this. */
  get dedupKey(): string {
    if (this.kind === BEACON) {
      return `beacon:${this.originId}:${this.sequenceId}`;
    }
    return `${this.kind}:${this.originId}:${this.targetId}:${this.sequenceId}`;
  }

  /** The relayed copy one hop further from its origin. */
  relayed(): Signal {
    return new Signal({
      kind: this.kind,
      originId: this.originId,
      originAddress: this.originAddress,
      sequenceId: this.sequenceId,
      targetId: this.targetId,
      targetAddress: this.targetAddress,
      hopCount: this.hopCount + 1,
      ttl: this.ttl - 1,
    });
  }

  get isAlive(): boolean {
    return this.ttl > 0;
  }
}
