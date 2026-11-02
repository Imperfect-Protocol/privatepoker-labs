/**
 * The flock: a distributed brain with no safe place to keep it.
 * Ported 1:1 from `skynerd/learning/flock.py` (Genesis Milestone 4).
 *
 * There is no hippocampus at the rear. Every living drone IS a shard of the
 * swarm's memory. Three p2p laws replace the station:
 *
 * - Birth as mosaic: a newborn asks a few random living drones what they
 *   know and keeps roughly half of each answer. No two births are alike,
 *   and knowledge no living drone carries is never passed on.
 * - Death screams: a dying drone broadcasts its tensor; each living peer —
 *   airborne or waiting on the runway — hears it with some probability and
 *   absorbs a lossy trace. Whatever lived only in the dying drone mostly
 *   dies with it.
 * - Survivors keep their shard and stay in the flock, teaching later births.
 *
 * If the whole flock dies, the brain is empty again: a total wipe is total
 * amnesia, and the next newborn starts from hardware calibration alone.
 * That is the honest p2p price, and the reason waves must not all die.
 */
import { NumpyPcg64 } from "../core/rng.js";
import { ServoPolicy } from "../learning/servo-tensor.js";

export class Flock {
  readonly witnessProb: number;   // chance a peer hears a death scream
  readonly donorCount: number;    // peers a newborn asks
  readonly teachRate: number;     // blend strength per donor answer
  readonly teachFraction: number;
  readonly screamRate: number;    // how much of a scream a witness absorbs
  members: ServoPolicy[] = [];    // living drones, each a memory shard
  births = 0;
  obituariesHeard = 0;

  constructor({ witnessProb = 0.9, donorCount = 12, teachRate = 0.8,
                teachFraction = 0.75, screamRate = 0.3 } = {}) {
    this.witnessProb = witnessProb;
    this.donorCount = donorCount;
    this.teachRate = teachRate;
    this.teachFraction = teachFraction;
    this.screamRate = screamRate;
  }

  get size(): number {
    return this.members.length;
  }

  /** Assemble a newborn from random living donors. Empty flock = blank birth. */
  seedNewborn(policy: ServoPolicy, rng: NumpyPcg64): void {
    this.births += 1;
    if (this.members.length === 0) {
      return;
    }
    const count = Math.min(this.donorCount, this.members.length);
    const donors = rng.choice(this.members.length, count);
    for (const index of donors) {
      this.teach(policy, this.members[index], rng, this.teachRate);
    }
  }

  /** A drone joins the living: its shard is now part of the brain. */
  enroll(policy: ServoPolicy): void {
    this.members.push(policy);
  }

  /** A drone died: its piece of the brain dies with it. */
  reap(policy: ServoPolicy): void {
    const index = this.members.indexOf(policy);
    if (index >= 0) {
      this.members.splice(index, 1);
    }
  }

  /**
   * The dying drone screams its tensor; each living peer may hear it.
   * The scream is a small lossy trace — a few keys, weakly blended —
   * not a transplant. A dead drone must not rewrite the living.
   */
  broadcastObituary(dying: ServoPolicy, rng: NumpyPcg64): void {
    for (const member of this.members) {
      if (rng.uniform() < this.witnessProb) {
        this.teach(member, dying, rng, this.screamRate, 0.25);
        this.obituariesHeard += 1;
      }
    }
  }

  /** Blend a random subset of the donor's keys into the recipient. */
  private teach(recipient: ServoPolicy, donor: ServoPolicy, rng: NumpyPcg64,
                rate: number, fraction?: number): void {
    const keys = [...donor.tensor.values.keys()];   // insertion order, like the Python dict
    if (keys.length === 0) {
      return;
    }
    const share = fraction === undefined ? this.teachFraction : fraction;
    const count = Math.max(1, Math.trunc(keys.length * share));
    const chosen = rng.choice(keys.length, count);
    for (const index of chosen) {
      Flock.mergeKey(recipient.tensor, donor.tensor, keys[index], rate);
    }
  }

  /** Newest-slot merge of one key, station law without the station. */
  private static mergeKey(recipient: ServoPolicy["tensor"], donor: ServoPolicy["tensor"],
                          keyIndex: number, rate: number): void {
    const donorEpochs = donor.epochs.get(keyIndex)!;
    let newest = 0;
    for (let slot = 1; slot < donorEpochs.length; slot++) {
      if (donorEpochs[slot] > donorEpochs[newest]) {
        newest = slot;
      }
    }
    const recipientValues = recipient.values.get(keyIndex);
    if (recipientValues !== undefined) {
      const donorValues = donor.values.get(keyIndex)!;
      // float32 semantics exactly like NumPy scalar math:
      // diff is float32; rate * diff is float64; the store rounds to float32.
      const diff = Math.fround(donorValues[newest] - recipientValues[0]);
      recipientValues[0] = Math.fround(recipientValues[0] + rate * diff);
      recipient.epochs.get(keyIndex)![0] = 0;
      return;
    }
    recipient.values.set(keyIndex, Float32Array.from(donor.values.get(keyIndex)!));
    recipient.epochs.set(keyIndex, Int32Array.from(donorEpochs));
    recipient.updatesSinceFilter += 1;
  }
}
