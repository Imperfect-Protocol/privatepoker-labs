/**
 * Sparse Hebbian tensor (reward/score core) and the servo policy.
 * Ported from `skynerd/tensors/sparse_hebbian.py` (reward, score, fields)
 * and `skynerd/learning/servo_tensor.py` (Genesis Milestone 3).
 *
 * Design laws kept from the original:
 * - Sparse: only observed (key, time-slot) combinations allocate memory.
 *   Unobserved combinations cost nothing and score at baseline.
 * - Time slots: each key owns an 8-slot history; a slot re-writes when the
 *   epoch wraps, so old observations age out without timers.
 * - Score decay: learned deviation from baseline decays with observation age.
 * - Rewards may be negative (the anti-Hebbian arm): a minWeight floor lets
 *   death/punishment depress entries below baseline.
 *
 * The wavelet denoise pipeline (dense compose + chaos + project back) is a
 * later milestone (the Hebbian swarm); fly_episode only needs reward/score.
 *
 * Keys are 7 ints (5 instrument bins, surface, action); stored as one
 * bijective integer index via mixed-radix encoding over keyShape.
 * Tensor slot values use float32 storage exactly like NumPy (scores are
 * computed in float64 after widening, like the Python code).
 */

export const EASE_OFF = 0;
export const HOLD = 1;
export const PUSH = 2;
export const ACTION_AXIS = 4;            // padded to 4 for Haar; index 3 is never chosen
export const SURFACE_AXIS = 8;           // 4 real channels; index 4 is the hold-all slot
export const HOLD_ALL_SURFACE = 4;
export const ACTION_DELTA = 0.05;        // surface deflection change per decision, radians
const ELIGIBILITY_DECISIONS = 10;        // credit the last N decisions (1 s of flight)
const ELIGIBILITY_DECAY = 0.7;

import { NumpyPcg64 } from "../core/rng.js";
import { SURFACE_LIMITS } from "../physics/surfaces.js";

export class SparseHebbianTensor {
  readonly keyShape: readonly number[];
  readonly timeDepth: number;
  readonly learningEnabled: boolean;
  readonly baseline: number;
  readonly maxWeight: number;
  readonly minWeight: number;
  /** key index -> per-slot values (float32 storage, like NumPy). */
  readonly values = new Map<number, Float32Array>();
  readonly epochs = new Map<number, Int32Array>();
  rewardTotal = 0.0;
  updatesSinceFilter = 0;

  private readonly strides: number[];

  constructor(keyShape: readonly number[], timeDepth = 8,
              baseline = 0.1, maxWeight = 5.0, minWeight = 0.0,
              learningEnabled = true) {
    this.keyShape = keyShape.map((d) => Math.trunc(d));
    if (this.keyShape.length === 0 || this.keyShape.some((d) => d < 1)) {
      throw new Error("keyShape must be non-empty with positive dimensions");
    }
    for (const dim of [...this.keyShape, timeDepth]) {
      if (dim & (dim - 1)) {
        throw new Error("all tensor axes must be powers of two for Haar");
      }
    }
    this.timeDepth = Math.trunc(timeDepth);
    this.learningEnabled = learningEnabled;
    // With learning disabled, represent a logical dense unit tensor implicitly.
    this.baseline = learningEnabled ? baseline : 1.0;
    this.maxWeight = maxWeight;
    this.minWeight = minWeight;
    this.strides = new Array(this.keyShape.length);
    let stride = 1;
    for (let i = this.keyShape.length - 1; i >= 0; i--) {
      this.strides[i] = stride;
      stride *= this.keyShape[i];
    }
  }

  /** Bijective integer index for an integer key tuple. */
  private keyIndex(parts: readonly number[]): number {
    let index = 0;
    for (let i = 0; i < this.keyShape.length; i++) {
      index += Math.trunc(parts[i]) * this.strides[i];
    }
    return index;
  }

  get storedCombinations(): number {
    return this.values.size;
  }

  /**
   * Hebbian (amount > 0) or anti-Hebbian (amount < 0) update at a time slot.
   * The amount applies to the deviation from baseline, clamped into
   * [minWeight, maxWeight]. Zero amounts change nothing.
   */
  reward(key: readonly number[], timeEpoch: number, amount: number): void {
    if (!this.learningEnabled || amount === 0.0) {
      return;
    }
    const index = this.keyIndex(key);
    let slots = this.values.get(index);
    if (slots === undefined) {
      slots = new Float32Array(this.timeDepth).fill(this.baseline);
      this.values.set(index, slots);
      this.epochs.set(index, new Int32Array(this.timeDepth).fill(-1));
    }
    const slot = Math.trunc(timeEpoch) % this.timeDepth;
    const slotEpochs = this.epochs.get(index)!;
    if (slotEpochs[slot] !== Math.trunc(timeEpoch)) {
      slots[slot] = this.baseline;
      slotEpochs[slot] = Math.trunc(timeEpoch);
    }
    const old = slots[slot];                    // float32 -> float64 (exact)
    const updated = Math.min(this.maxWeight,
                             Math.max(this.minWeight, old + amount));
    slots[slot] = updated;                      // float32 store (rounds)
    this.rewardTotal += Math.max(0.0, updated - old);
    this.updatesSinceFilter += 1;
  }

  /** Age-decayed learned score for a key; baseline when unobserved. */
  score(key: readonly number[], timeEpoch = 0, decay = 0.97): number {
    if (!this.learningEnabled) {
      return 1.0;
    }
    const index = this.keyIndex(key);
    const slots = this.values.get(index);
    if (slots === undefined) {
      return this.baseline;
    }
    const slotEpochs = this.epochs.get(index)!;
    let learned = 0.0;
    for (let slot = 0; slot < this.timeDepth; slot++) {
      if (slotEpochs[slot] < 0) {
        continue;
      }
      const delta = slots[slot] - this.baseline;   // float32 -> float64
      const age = Math.max(0, Math.trunc(timeEpoch) - slotEpochs[slot]);
      learned += delta * Math.pow(decay, age);
    }
    return Math.min(this.maxWeight,
                    Math.max(this.minWeight, this.baseline + learned));
  }
}

/** Owns the servo tensor; turns flight state into surface commands. */
export class ServoPolicy {
  readonly tensor: SparseHebbianTensor;
  epsilon = 0.5;

  constructor(baseline = 0.1, maxWeight = 5.0, minWeight = 0.0,
              learningEnabled = true) {
    this.tensor = new SparseHebbianTensor(
      [8, 8, 8, 8, 8, SURFACE_AXIS, ACTION_AXIS],
      8, baseline, maxWeight, minWeight, learningEnabled);
  }

  private score(stateKey: readonly number[], surface: number, level: number): number {
    return this.tensor.score([...stateKey, surface, level], 0, 1.0);
  }

  /**
   * Pick hold-all or one single-surface push, epsilon-greedily.
   * Returns an actions array of 4 channel entries, each HOLD, with at most
   * one entry set to EASE_OFF or PUSH.
   */
  choose(stateKey: readonly number[], rng: NumpyPcg64): Int32Array {
    const actions = new Int32Array(4).fill(HOLD);
    let choice = 0;
    if (rng.uniform() < this.epsilon) {
      choice = rng.integers(1 + 4 * 2);   // 0 = hold-all, else surface/direction
    } else {
      let best = this.score(stateKey, HOLD_ALL_SURFACE, HOLD);
      for (let surface = 0; surface < 4; surface++) {
        for (const level of [EASE_OFF, PUSH] as const) {
          const score = this.score(stateKey, surface, level);
          if (score > best) {
            best = score;
            choice = 1 + surface * 2 + (level === PUSH ? 1 : 0);
          }
        }
      }
    }
    if (choice > 0) {
      const surface = Math.floor((choice - 1) / 2);
      actions[surface] = (choice - 1) % 2 === 0 ? EASE_OFF : PUSH;
    }
    return actions;
  }

  /** Accumulate action deltas onto current commands, clamped to hardware. */
  commandFrom(current: Float64Array, actions: Int32Array): Float64Array {
    const targets = Float64Array.from(current);
    for (let surface = 0; surface < 4; surface++) {
      const level = actions[surface];
      if (level === HOLD) {
        continue;
      }
      const delta = level === PUSH ? ACTION_DELTA : -ACTION_DELTA;
      const [low, high] = SURFACE_LIMITS[surface];
      targets[surface] = Math.max(low, Math.min(high, targets[surface] + delta));
    }
    return targets;
  }

  /**
   * Credit recent (stateKey, actions) decisions with decaying eligibility.
   * The credited key is whichever single option the decision took
   * (hold-all or one surface push).
   */
  reinforce(recent: ReadonlyArray<[readonly number[], Int32Array]>,
            reward: number, timeEpoch: number): void {
    if (!this.tensor.learningEnabled || reward === 0.0 || recent.length === 0) {
      return;
    }
    let amount = reward;
    const first = Math.max(0, recent.length - ELIGIBILITY_DECISIONS);
    for (let i = recent.length - 1; i >= first; i--) {
      const [stateKey, actions] = recent[i];
      let surface = HOLD_ALL_SURFACE;
      let level = HOLD;
      for (let s = 0; s < 4; s++) {
        if (actions[s] !== HOLD) {
          surface = s;
          level = actions[s];
          break;
        }
      }
      this.tensor.reward([...stateKey, surface, level], timeEpoch, amount);
      amount *= ELIGIBILITY_DECAY;
    }
  }
}
