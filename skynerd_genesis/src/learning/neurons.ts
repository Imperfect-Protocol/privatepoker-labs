/**
 * THE NEURONS of the intention system. Five separate sparse tensors
 * ("neurons"), each projecting onto the element axis, connected by
 * synapses into a pulsating dynamic system:
 *
 *   1. goal             -> element   (the curriculum's goal-element map)
 *   2. flight instruments -> element (what the panel says about the element)
 *   3. engine instruments -> element
 *   4. controls (old)   -> element   (what the yoke says about the element)
 *   5. element x flight x controls (old) -> controls (new)  (the servo skill)
 *
 * Every neuron's memory is a sparse EMA TIME-DECAY STACK: a new
 * observation fires into the freshest layer; each new epoch sinks the
 * stack one layer down; recall mixes the layers with the neuron's EMA
 * coefficient — older layers still count, so the memory holds. The
 * stack never CLIFFS: the layer that departs the bottom folds into a
 * per-key STATIC TAIL STORE, which then decays only by the neuron's
 * EMA per further sink — a lesson learned in the flight school STAYS,
 * fading smoothly, instead of vanishing on the fourth re-lesson. The
 * EMA coefficient is the STATIC/DYNAMIC memory dial: near 1 the neuron
 * keeps its lessons for the whole school career (the curriculum, the
 * skills); lower, and recent flights dominate while old habits fade.
 *
 * The neurons do not run a feedback loop. They PULSE: an input stuns a
 * neuron, the neuron relaxes through its refractory period, then it
 * FIRES its projection downstream — and the fire startles the next
 * neuron. One fires, the other waits; information travels in time.
 *
 * What a neuron fires is DENOISED, not raw recall: stochastic noise is
 * added, a gaussian filter smooths the projection, one Haar level
 * soft-thresholds the detail, and a final threshold cuts the noise
 * floor — the same laws the Hebbian swarm's wavelet pipeline uses,
 * at neuron scale.
 */
import { NumpyPcg64 } from "../core/rng.js";

/** The time-decay depth of every neuron's stack (powers of two for Haar). */
export const NEURON_LAYERS = 4;

/** One neuron's memory law. */
export interface NeuronSpec {
  readonly name: string;
  readonly keyShape: readonly number[];
  /**
   * The EMA coefficient of the time-decay stack: how strongly older
   * layers still count on recall. Near 1 = STATIC memory — lessons
   * learned in the flight school stay. Lower = DYNAMIC memory — recent
   * flights dominate and old habits fade.
   */
  readonly ema: number;
  /** The score of a never-observed key. */
  readonly baseline: number;
  readonly maxWeight: number;
  readonly minWeight: number;
  /**
   * The FUZZY AXES: key positions whose neighboring bins still count,
   * halved per step. A swept maneuver stamps a thin diagonal through
   * the (airspeed, pitch) plane — the student's slower sweep crosses
   * neighboring bins, and a neuron that answers only the exact bin
   * leaves the student blind one bin off the demonstration. The
   * slow-moving DISCRIMINATOR axes (bank, heading, altitude) stay
   * exact — fuzzing them would merge distinct elements.
   */
  readonly fuzzyAxes?: readonly number[];
}

/** How strongly a neighboring bin still counts on a fuzzy axis. A
 * swept maneuver's diagonal drifts one bin on several axes at once —
 * the neighbor credit must survive the product of the per-axis steps. */
const FUZZY_NEIGHBOR_WEIGHT = 0.75;

/** A sparse memory whose weights live on an EMA time-decay stack. */
export class PulseNeuron {
  readonly name: string;
  private readonly strides: number[];
  readonly ema: number;
  readonly baseline: number;
  readonly maxWeight: number;
  readonly minWeight: number;
  private readonly fuzzyAxes: readonly number[];
  /** key index -> the EMA stack + the static tail store (the slot at
   *  NEURON_LAYERS), float32 storage like NumPy. */
  readonly values = new Map<number, Float32Array>();
  /** key index -> the epoch each layer was fired in. */
  readonly epochs = new Map<number, Int32Array>();

  constructor(spec: NeuronSpec) {
    this.name = spec.name;
    this.ema = spec.ema;
    this.baseline = spec.baseline;
    this.maxWeight = spec.maxWeight;
    this.minWeight = spec.minWeight;
    this.fuzzyAxes = spec.fuzzyAxes ?? [];
    this.axisShapes = [...spec.keyShape];
    this.strides = new Array(spec.keyShape.length);
    let stride = 1;
    for (let i = spec.keyShape.length - 1; i >= 0; i--) {
      if (spec.keyShape[i] < 1) {
        throw new Error(`neuron ${spec.name}: key axes must be positive`);
      }
      this.strides[i] = stride;
      stride *= spec.keyShape[i];
    }
  }

  /** Bijective integer index for an integer key tuple. */
  private keyIndex(parts: readonly number[]): number {
    let index = 0;
    for (let i = 0; i < this.strides.length; i++) {
      index += Math.trunc(parts[i]) * this.strides[i];
    }
    return index;
  }

  get storedCombinations(): number {
    return this.values.size;
  }

  /**
   * A pulse arrives: the weight fires into the freshest layer. A new
   * epoch sinks the stack one layer down first — the older observation
   * keeps counting on recall, weighted by the neuron's EMA. The layer
   * that falls off the bottom does not die: it folds into the STATIC
   * TAIL STORE, which carries the departed lesson onward, decaying only
   * by the EMA per further sink. Memory with decay: nothing is lost at
   * once, everything fades by depth — and a static neuron fades so
   * slowly that its flight-school lessons outlast the school.
   */
  fire(key: readonly number[], amount: number, epoch: number): void {
    if (amount === 0.0) {
      return;
    }
    const index = this.keyIndex(key);
    let layers = this.values.get(index);
    if (layers === undefined) {
      layers = new Float32Array(NEURON_LAYERS + 1).fill(this.baseline);
      this.values.set(index, layers);
      this.epochs.set(index, new Int32Array(NEURON_LAYERS + 1).fill(-1));
    }
    const layerEpochs = this.epochs.get(index)!;
    if (layerEpochs[0] !== Math.trunc(epoch)) {
      // The departing layer folds into the static tail store: it joins
      // weighted ema^layers (the recall weight it is leaving behind),
      // and the store itself decays by ema per further sink — the exact
      // geometric continuation of the stack, without the cliff.
      const tail = layers[NEURON_LAYERS];
      const departing = layers[NEURON_LAYERS - 1];
      layers[NEURON_LAYERS] = Math.min(this.maxWeight, Math.max(this.minWeight,
        this.baseline
        + this.ema * (tail - this.baseline)
        + Math.pow(this.ema, NEURON_LAYERS) * (departing - this.baseline)));
      for (let k = NEURON_LAYERS - 1; k > 0; k--) {
        layers[k] = layers[k - 1];
        layerEpochs[k] = layerEpochs[k - 1];
      }
      layers[0] = this.baseline;
      layerEpochs[0] = Math.trunc(epoch);
    }
    // float32 store, like NumPy
    layers[0] = Math.min(this.maxWeight,
                         Math.max(this.minWeight, layers[0] + amount));
  }

  /**
   * IMPOSE a prior: raise the freshest layer to AT LEAST `value` —
   * a floor, never a ceiling, so a cell training itself above the
   * prior keeps its training — and refresh the epoch (a prior nobody
   * re-fires must not die to the forgetting law). Idempotent: the
   * same prior imposed twice leaves the same weight.
   */
  impose(key: readonly number[], value: number, epoch: number): void {
    const index = this.keyIndex(key);
    let layers = this.values.get(index);
    if (layers === undefined) {
      layers = new Float32Array(NEURON_LAYERS + 1).fill(this.baseline);
      this.values.set(index, layers);
      this.epochs.set(index, new Int32Array(NEURON_LAYERS + 1).fill(-1));
    }
    if (layers[0] < value) {
      layers[0] = Math.min(this.maxWeight, Math.max(this.minWeight, value));
    }
    this.epochs.get(index)![0] = Math.trunc(epoch);
  }

  /**
   * The EMA-mixed recall: the freshest layer counts fully, older layers
   * by ema^depth — the time-decay dimension of the tensor — and the
   * static tail store counts in full, because it already carries its
   * own decay. On the fuzzy axes the neighboring bins count too (halved
   * per step): the recall is the memory's best response in the
   * neighborhood of the reading.
   */
  recall(key: readonly number[]): number {
    if (this.fuzzyAxes.length === 0) {
      return this.recallExact(key);
    }
    let best = this.baseline;   // the unobserved floor
    const offsetKey = [...key];
    const explore = (axis: number): void => {
      if (axis === this.fuzzyAxes.length) {
        let weight = 1.0;
        for (let f = 0; f < this.fuzzyAxes.length; f++) {
          const step = offsetKey[this.fuzzyAxes[f]] - key[this.fuzzyAxes[f]];
          weight *= step === 0 ? 1.0
            : Math.pow(FUZZY_NEIGHBOR_WEIGHT, Math.abs(step));
        }
        best = Math.max(best, weight * this.recallExact(offsetKey));
        return;
      }
      const position = this.fuzzyAxes[axis];
      const shape = this.axisShapes[position];
      for (const step of [-1, 0, 1]) {
        const shifted = key[position] + step;
        if (shifted < 0 || shifted >= shape) {
          continue;
        }
        offsetKey[position] = shifted;
        explore(axis + 1);
      }
      offsetKey[position] = key[position];
    };
    explore(0);
    return Math.min(this.maxWeight, best);
  }

  /** The exact-bin EMA-mixed recall (no fuzz). */
  private recallExact(key: readonly number[]): number {
    const layers = this.values.get(this.keyIndex(key));
    if (layers === undefined) {
      return this.baseline;
    }
    let recalled = this.baseline;
    let weight = 1.0;
    for (let k = 0; k < NEURON_LAYERS; k++) {
      recalled += (layers[k] - this.baseline) * weight;   // float32 -> float64
      weight *= this.ema;
    }
    // The static tail store: the departed lessons, decayed by ema per
    // sink — the geometric continuation of the stack above.
    recalled += layers[NEURON_LAYERS] - this.baseline;
    return Math.min(this.maxWeight, Math.max(this.minWeight, recalled));
  }

  private readonly axisShapes: number[] = [];

  /**
   * THE FORGETTING LAW: drop every key nobody has re-fired within the
   * span — the endless classroom's memory bound. Nothing else ever
   * shrinks the sparse map (untouched keys keep their values
   * forever), and the school deals cycle after cycle forever, so the
   * keys that no lesson re-uses must eventually die. Returns how many
   * died.
   */
  prune(beforeEpoch: number): number {
    let removed = 0;
    for (const [index, layerEpochs] of this.epochs) {
      let newest = -1;
      for (let layer = 0; layer < layerEpochs.length; layer++) {
        if (layerEpochs[layer] > newest) {
          newest = layerEpochs[layer];
        }
      }
      if (newest < beforeEpoch) {
        this.values.delete(index);
        this.epochs.delete(index);
        removed += 1;
      }
    }
    return removed;
  }

  snapshot(): { values: Map<number, Float32Array>; epochs: Map<number, Int32Array> } {
    const values = new Map<number, Float32Array>();
    for (const [key, layers] of this.values) {
      values.set(key, Float32Array.from(layers));
    }
    const epochs = new Map<number, Int32Array>();
    for (const [key, layerEpochs] of this.epochs) {
      epochs.set(key, Int32Array.from(layerEpochs));
    }
    return { values, epochs };
  }

  restore(snapshot: { values: Map<number, Float32Array>;
                      epochs: Map<number, Int32Array> }): void {
    this.values.clear();
    for (const [key, layers] of snapshot.values) {
      this.values.set(key, Float32Array.from(layers));
    }
    this.epochs.clear();
    for (const [key, layerEpochs] of snapshot.epochs) {
      this.epochs.set(key, Int32Array.from(layerEpochs));
    }
  }
}

/**
 * The refractory period: a neuron that has fired is STUNNED, waits out
 * its stun relaxing, then fires again. This is what makes the system
 * pulse — the neurons take turns, so no feedback loop can run away:
 * one fires, the other waits stunned, then the other fires.
 */
export class Refractory {
  private stunned = 0;

  constructor(readonly stunDecisions: number) {}

  /**
   * One decision clock: returns true when the neuron fires now, and
   * startles it for its refractory period (the decisions it must wait).
   */
  tick(): boolean {
    if (this.stunned > 0) {
      this.stunned -= 1;
      return false;
    }
    this.stunned = this.stunDecisions;
    return true;
  }
}

/**
 * The neuron's firing law — the projection a neuron EMITS is denoised,
 * not raw recall:
 *   1. stochastic noise (Box-Muller gaussian, the exploration),
 *   2. a gaussian filter ([1 2 1]/4, circular) over the axis,
 *   3. one Haar level, the detail SOFT-THRESHOLDED, projected back,
 *   4. a final threshold that cuts the noise floor to zero.
 * The axis length must be a power of two (the Haar law).
 */
export function denoiseProjection(projection: Float64Array, rng: NumpyPcg64,
                                  noiseSigma: number,
                                  detailThreshold: number,
                                  threshold: number): Float64Array {
  const n = projection.length;
  const noisy = Float64Array.from(projection);
  for (let i = 0; i < n; i++) {
    // Box-Muller: a deterministic gaussian from the shared dice.
    const u1 = Math.max(1e-12, rng.uniform());
    const u2 = rng.uniform();
    noisy[i] += noiseSigma * Math.sqrt(-2.0 * Math.log(u1))
                * Math.cos(2.0 * Math.PI * u2);
  }
  const blurred = new Float64Array(n);
  for (let i = 0; i < n; i++) {
    blurred[i] = 0.25 * noisy[(i - 1 + n) % n]
               + 0.50 * noisy[i]
               + 0.25 * noisy[(i + 1) % n];
  }
  // One Haar level: [approx | detail], detail soft-thresholded, back.
  const half = n >> 1;
  const approx = new Float64Array(half);
  const detail = new Float64Array(half);
  const invRoot2 = 1.0 / Math.SQRT2;
  for (let i = 0; i < half; i++) {
    approx[i] = (blurred[2 * i] + blurred[2 * i + 1]) * invRoot2;
    const d = (blurred[2 * i] - blurred[2 * i + 1]) * invRoot2;
    detail[i] = Math.sign(d) * Math.max(0.0, Math.abs(d) - detailThreshold);
  }
  const out = new Float64Array(n);
  for (let i = 0; i < half; i++) {
    out[2 * i] = (approx[i] + detail[i]) * invRoot2;
    out[2 * i + 1] = (approx[i] - detail[i]) * invRoot2;
  }
  for (let i = 0; i < n; i++) {
    if (Math.abs(out[i]) < threshold) {
      out[i] = 0.0;
    }
  }
  return out;
}
