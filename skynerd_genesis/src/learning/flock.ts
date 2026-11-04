/**
 * The flock: a distributed brain with no safe place to keep it.
 * Ported 1:1 from `skynerd/learning/flock.py` (Genesis Milestone 4).
 *
 * There is no hippocampus at the rear. Every living drone IS a shard of
 * the swarm's memory — a whole AiPilot brain, tensors inside. Three p2p
 * laws replace the station:
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
 *
 * The blending itself is the brain's own law (AiPilot.blendFrom) — one
 * law, one implementation, and it lives inside the pilot.
 */
import { NumpyPcg64 } from "../core/rng.js";
import type { AiPilot } from "./pilot.js";

export class Flock {
  readonly witnessProb: number;   // chance a peer hears a death scream
  readonly donorCount: number;    // peers a newborn asks
  readonly teachRate: number;     // blend strength per donor answer
  readonly teachFraction: number;
  readonly screamRate: number;    // how much of a scream a witness absorbs
  members: AiPilot[] = [];         // living drones, each a memory shard
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
  seedNewborn(pilot: AiPilot, rng: NumpyPcg64): void {
    this.births += 1;
    if (this.members.length === 0) {
      return;
    }
    const count = Math.min(this.donorCount, this.members.length);
    const donors = rng.choice(this.members.length, count);
    for (const index of donors) {
      this.teach(pilot, this.members[index], rng, this.teachRate);
    }
  }

  /** A drone joins the living: its shard is now part of the brain. */
  enroll(pilot: AiPilot): void {
    this.members.push(pilot);
  }

  /** A drone died: its piece of the brain dies with it. */
  reap(pilot: AiPilot): void {
    const index = this.members.indexOf(pilot);
    if (index >= 0) {
      this.members.splice(index, 1);
    }
  }

  /**
   * The dying drone screams its tensor; each living peer may hear it.
   * The scream is a small lossy trace — a few keys, weakly blended —
   * not a transplant. A dead drone must not rewrite the living.
   */
  broadcastObituary(dying: AiPilot, rng: NumpyPcg64): void {
    for (const member of this.members) {
      if (rng.uniform() < this.witnessProb) {
        this.teach(member, dying, rng, this.screamRate, 0.25);
        this.obituariesHeard += 1;
      }
    }
  }

  /** Blend a random subset of each donor neuron into the recipient. */
  private teach(recipient: AiPilot, donor: AiPilot, rng: NumpyPcg64,
                rate: number, fraction?: number): void {
    Flock.blend(recipient, donor, rng, rate,
                fraction === undefined ? this.teachFraction : fraction);
  }

  /** The swarm's shared memory primitive: merge a random subset of each
   *  donor neuron into the recipient at the given rate. The flock's
   *  births and death screams use it; the SINGULARITY TENSOR's rounds
   *  use it too (learning/singularity.ts) — one law, one implementation,
   *  and it lives inside the brain. */
  static blend(recipient: AiPilot, donor: AiPilot, rng: NumpyPcg64,
               rate: number, share: number): void {
    recipient.blendFrom(donor, rng, rate, share);
  }
}
