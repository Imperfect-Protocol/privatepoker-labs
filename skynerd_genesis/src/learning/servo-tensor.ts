/**
 * THE STUDENT'S BRAIN: five neurons connected by synapses into a
 * pulsating dynamic system, the servo policy being the assembly.
 *
 *   1. goalNeuron      goal              -> element
 *   2. flightNeuron    flight instruments -> element
 *   3. engineNeuron    engine instruments -> element
 *   4. controlsNeuron  controls (old)    -> element
 *   5. servoNeuron     element x flight x controls (old) -> controls (new)
 *
 * There is no hierarchy in the weights — each neuron is a
 * multi-dimensional sparse tensor whose cells depend on all of its
 * factors at once. The hierarchy is in TIME: neurons pulse (stun ->
 * relax -> fire, neurons.ts), so the element choice and the action
 * choice take turns — one fires while the other waits stunned, and the
 * information travels down the synapse in discrete pulses instead of
 * an instantaneous feedback loop.
 *
 * Every neuron carries its own EMA time-decay stack. The EMA
 * coefficients are the STATIC/DYNAMIC memory dial: the goal map and the
 * skill memory are near-static (lessons learned in the flight school
 * stay), the panel-to-element neurons slightly more dynamic.
 *
 * Design laws kept from the sparse Hebbian core:
 * - Sparse: only observed key combinations allocate memory; unobserved
 *   combinations cost nothing and recall at baseline.
 * - Weights may depress below baseline (the anti-Hebbian arm).
 * - float32 storage exactly like NumPy (scores computed in float64).
 */
import { NumpyPcg64 } from "../core/rng.js";
import { SURFACE_LIMITS } from "../physics/surfaces.js";
import {
  FLIGHT_SENSE_SHAPE, ENGINE_SENSE_SHAPE, CONTROLS_SENSE_SHAPE,
  type SenseReading,
} from "./instruments.js";
import { PulseNeuron, Refractory, denoiseProjection } from "./neurons.js";

export const EASE_OFF = 0;
export const HOLD = 1;
export const PUSH = 2;
export const ACTION_AXIS = 4;            // padded to 4 for Haar; index 3 is never chosen
export const SURFACE_AXIS = 8;           // 4 real channels; index 4 is the hold-all slot
export const HOLD_ALL_SURFACE = 4;
export const ACTION_DELTA = 0.05;        // surface deflection change per decision, radians
const ELIGIBILITY_DECISIONS = 10;        // credit the last N decisions (1 s of flight)
const ELIGIBILITY_DECAY = 0.7;

// THE INTENTION AXES. The goal axis: which higher-level goal is being
// realized (take off from the runway, turn left 180, recover from the
// spin…). The element axis: which atomic element of it is being executed
// right now (roll, rotate, climb, level off…) — school.ts owns the
// vocabulary.
export const GOAL_AXIS = 16;
export const ELEMENT_AXIS = 32;

// The reserved goal slots. FOLLOW_INSTRUCTOR is the higher-level goal of
// an instruction flight: "listen to the instructor and do what they ask"
// — on a demonstration the instructor chooses the element and the
// actions, and the student records what it watches FILED UNDER THE
// MANEUVER BEING TAUGHT, so the solo repeat finds it under that goal.
export const GOAL_FOLLOW_INSTRUCTOR = 0;
// Free flight (the swarm's episodes): no maneuver, no elements.
export const GOAL_FREE_FLIGHT = 1;
// The curriculum's maneuvers take goal ids from 2 upward (school.ts).

// The reserved element slot for free flight.
export const ELEMENT_FREE_FLIGHT = 0;

/**
 * The static/dynamic memory dial — each neuron's EMA coefficient on its
 * time-decay stack. Near 1 = static: lessons learned in the flight
 * school stay. Lower = dynamic: recent flights dominate, old habits
 * fade. The goal map is the most static thing a student owns; the
 * panel-to-element neurons refresh with every flight; the skill memory
 * keeps its lessons but still respects recency.
 */
export const NEURON_EMA = {
  goal: 0.98,       // the goal -> element map IS the curriculum
  flight: 0.90,     // the panel's evidence refreshes with every flight
  engine: 0.95,
  controls: 0.90,
  servo: 0.90,      // the skill memory: lessons stay, recency matters
} as const;

/** The firing law's constants (neurons.ts denoiseProjection). */
const ELEMENT_NOISE_SIGMA = 0.15;
const ELEMENT_DETAIL_THRESHOLD = 0.20;
const ELEMENT_FIRE_THRESHOLD = 0.75;   // cuts the unobserved noise floor
const ACTION_NOISE_SIGMA = 0.05;
const ACTION_DETAIL_THRESHOLD = 0.05;
const ACTION_FIRE_THRESHOLD = 0.30;

/**
 * The element's OWN evidence must clear this bar to be a candidate —
 * a single stamped cell (1.1) clears it; the unobserved floor (0.1)
 * does not; a neighbor's blur leak never counts.
 */
export const ELEMENT_FIRE_BAR = 0.75;

/** One neuron's memory snapshot (the graduate the flock inherits from). */
export interface NeuronSnapshot {
  values: Map<number, Float32Array>;
  epochs: Map<number, Int32Array>;
}

/** The whole brain: one snapshot per neuron. */
export interface BrainSnapshot {
  goal: NeuronSnapshot;
  flight: NeuronSnapshot;
  engine: NeuronSnapshot;
  controls: NeuronSnapshot;
  servo: NeuronSnapshot;
}

/** Owns the five neurons; turns the panel into element and action choices. */
export class ServoPolicy {
  /** 1. goal -> element: the curriculum's goal-element map. */
  readonly goalNeuron = new PulseNeuron({
    name: "goal->element", keyShape: [GOAL_AXIS, ELEMENT_AXIS],
    ema: NEURON_EMA.goal, baseline: 0.1, maxWeight: 5.0, minWeight: 0.0,
  });
  /** 2. flight instruments -> element. The sweeping gauges (vs, asi,
   *  pitch) recall fuzzily — the student's own flying drifts a bin off
   *  the instructor's path (speed wanders, the balloon bleeds it) and a
   *  neuron that answers only the exact bin leaves the student blind
   *  mid-maneuver; the DISCRIMINATOR axes (bank, hsi, alt) stay exact —
   *  they gate the element transitions. */
  readonly flightNeuron = new PulseNeuron({
    name: "flight->element", keyShape: [...FLIGHT_SENSE_SHAPE, ELEMENT_AXIS],
    ema: NEURON_EMA.flight, baseline: 0.1, maxWeight: 5.0, minWeight: 0.0,
    fuzzyAxes: [0, 2, 4],
  });
  /** 3. engine instruments -> element. */
  readonly engineNeuron = new PulseNeuron({
    name: "engine->element", keyShape: [...ENGINE_SENSE_SHAPE, ELEMENT_AXIS],
    ema: NEURON_EMA.engine, baseline: 0.1, maxWeight: 5.0, minWeight: 0.0,
  });
  /** 4. controls (old) -> element. */
  readonly controlsNeuron = new PulseNeuron({
    name: "controls->element", keyShape: [...CONTROLS_SENSE_SHAPE, ELEMENT_AXIS],
    ema: NEURON_EMA.controls, baseline: 0.1, maxWeight: 5.0, minWeight: 0.0,
  });
  /** 5. element x flight x controls (old) -> controls (new): the servo
   *  skill. The flight part's sweeping gauges recall fuzzily — the
   *  student's slower sweeps cross neighboring bins — and the ELEVATOR
   *  hand-chase does too: the student's pull ramps at its own rate, so
   *  it crosses each pitch bin one elevator bin off the instructor's
   *  path, and a yoke that answers only the exact bin leaves the student
   *  deaf to the ease-back lesson. The aileron/rudder/throttle part
   *  stays exact: those levers' positions ARE the memory's context.
   *  Key: [element, vs, bank, asi, hsi, pitch, alt, ail, elev, rud,
   *  thr, surface, action]. */
  readonly servoNeuron = new PulseNeuron({
    name: "element x flight x controls -> controls",
    keyShape: [ELEMENT_AXIS, ...FLIGHT_SENSE_SHAPE, ...CONTROLS_SENSE_SHAPE,
               SURFACE_AXIS, ACTION_AXIS],
    ema: NEURON_EMA.servo, baseline: 0.1, maxWeight: 5.0, minWeight: 0.0,
    fuzzyAxes: [1, 3, 5, 8],
  });
  epsilon = 0.5;

  get storedCombinations(): number {
    return this.goalNeuron.storedCombinations
         + this.flightNeuron.storedCombinations
         + this.engineNeuron.storedCombinations
         + this.controlsNeuron.storedCombinations
         + this.servoNeuron.storedCombinations;
  }

  /**
   * Flight school: the demonstration's decision, fired into EVERY
   * neuron under the reading it was made at — the goal and element it
   * belongs to, the panel as the neurons split it. This is the
   * watch-and-remember channel, reward-free and immediate.
   */
  imprint(goalId: number, elementId: number, sense: SenseReading,
          actions: ArrayLike<number>, amount = 1.0, epoch = 0): void {
    this.goalNeuron.fire([goalId, elementId], amount, epoch);
    this.flightNeuron.fire([...sense.flight, elementId], amount, epoch);
    this.engineNeuron.fire([...sense.engine, elementId], amount, epoch);
    this.controlsNeuron.fire([...sense.controls, elementId], amount, epoch);
    for (const [surface, level] of ServoPolicy.changedSurfaces(actions)) {
      this.servoNeuron.fire(
        [elementId, ...sense.flight, ...sense.controls, surface, level],
        amount, epoch);
    }
  }

  /** Which single option a decision took (hold-all or one surface push). */
  private static changedSurfaces(actions: ArrayLike<number>): Array<[number, number]> {
    const changed: Array<[number, number]> = [];
    for (let surface = 0; surface < 4; surface++) {
      if (actions[surface] !== HOLD) {
        changed.push([surface, actions[surface]]);
      }
    }
    if (changed.length === 0) {
      changed.push([HOLD_ALL_SURFACE, HOLD]);
    }
    return changed;
  }

  /**
   * The RAW element evidence — each element's OWN cells under this
   * reading (pre-denoise): the flight neuron's discrimination, modulated
   * by engine and controls. An element counts as its own evidence only
   * when its cells were actually stamped here — a neighbor's blur leak
   * is not evidence.
   */
  elementEvidence(goalId: number, sense: SenseReading): Float64Array {
    const evidence = new Float64Array(ELEMENT_AXIS);
    for (let element = 0; element < ELEMENT_AXIS; element++) {
      const onMenu = this.goalNeuron.recall([goalId, element])
                     > this.goalNeuron.baseline;
      if (!onMenu) {
        continue;
      }
      evidence[element] = this.flightNeuron.recall([...sense.flight, element])
        * (0.5 + this.engineNeuron.recall([...sense.engine, element]) / 10.0)
        * (0.5 + this.controlsNeuron.recall([...sense.controls, element]) / 10.0);
    }
    return evidence;
  }

  /**
   * The upstream neurons FIRE their element projections. The goal neuron
   * projects the MENU — which elements belong to this goal at all — and
   * GATES the reading; the flight neuron is the PRIMARY discriminator
   * (the panel reading), and the engine and controls neurons MODULATE
   * it: evidence = flight x (0.5 + engine/10) x (0.5 + controls/10).
   * The engine and controls states are ambiguous across elements on
   * their own (a cruise throttle reads the same in a roll and a level
   * off) — only the flight reading says WHICH element this is, so a
   * wrong-flight-reading element cannot fire no matter what the levers
   * say. The fired projection is denoised (stochastic noise, gaussian
   * filter, Haar detail threshold, noise floor cut).
   */
  elementProjection(goalId: number, sense: SenseReading,
                    rng: NumpyPcg64): Float64Array {
    return denoiseProjection(this.elementEvidence(goalId, sense), rng,
                             ELEMENT_NOISE_SIGMA, ELEMENT_DETAIL_THRESHOLD,
                             ELEMENT_FIRE_THRESHOLD);
  }

  /**
   * Pick hold-all or one single-surface push, epsilon-greedily, from the
   * servo neuron's action scores under (element, flight, controls) —
   * denoised the way a neuron fires. Returns an actions array of 4
   * channel entries, each HOLD, with at most one set to EASE_OFF/PUSH.
   */
  choose(elementId: number, sense: SenseReading, rng: NumpyPcg64): Int32Array {
    const actions = new Int32Array(4).fill(HOLD);
    if (rng.uniform() < this.epsilon) {
      const choice = rng.integers(1 + 4 * 2);   // 0 = hold-all, else surface/direction
      if (choice > 0) {
        const surface = Math.floor((choice - 1) / 2);
        actions[surface] = (choice - 1) % 2 === 0 ? EASE_OFF : PUSH;
      }
      return actions;
    }
    // The servo neuron's 9 options, denoised (padded to 16 for Haar).
    const scores = new Float64Array(16).fill(0.0);
    for (let surface = 0; surface < 4; surface++) {
      for (const level of [EASE_OFF, PUSH] as const) {
        scores[1 + surface * 2 + (level === PUSH ? 1 : 0)] = this.servoNeuron.recall(
          [elementId, ...sense.flight, ...sense.controls, surface, level]);
      }
    }
    scores[0] = this.servoNeuron.recall(
      [elementId, ...sense.flight, ...sense.controls, HOLD_ALL_SURFACE, HOLD]);
    const fired = denoiseProjection(scores, rng, ACTION_NOISE_SIGMA,
                                    ACTION_DETAIL_THRESHOLD, ACTION_FIRE_THRESHOLD);
    let choice = 0;
    let best = fired[0];
    for (let option = 1; option < 9; option++) {
      if (fired[option] > best) {
        best = fired[option];
        choice = option;
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
   * Credit recent decisions with decaying eligibility — the servo
   * neuron's cells only: the action memory is what flight outcomes
   * teach. Each entry's key is the observation prefix
   * [element, ...flight, ...controls]; the credited cell is whichever
   * single option the decision took.
   */
  reinforce(recent: ReadonlyArray<[readonly number[], Int32Array]>,
            reward: number, epoch: number): void {
    if (reward === 0.0 || recent.length === 0) {
      return;
    }
    let amount = reward;
    const first = Math.max(0, recent.length - ELIGIBILITY_DECISIONS);
    for (let i = recent.length - 1; i >= first; i--) {
      const [key, actions] = recent[i];
      for (const [surface, level] of ServoPolicy.changedSurfaces(actions)) {
        this.servoNeuron.fire([...key, surface, level], amount, epoch);
      }
      amount *= ELIGIBILITY_DECAY;
    }
  }

  /** Copy the whole brain: the graduate the flock inherits from. */
  snapshot(): BrainSnapshot {
    return {
      goal: this.goalNeuron.snapshot(),
      flight: this.flightNeuron.snapshot(),
      engine: this.engineNeuron.snapshot(),
      controls: this.controlsNeuron.snapshot(),
      servo: this.servoNeuron.snapshot(),
    };
  }

  /** Overwrite the brain with a snapshot (a newborn inheriting a graduate). */
  restore(snapshot: BrainSnapshot): void {
    this.goalNeuron.restore(snapshot.goal);
    this.flightNeuron.restore(snapshot.flight);
    this.engineNeuron.restore(snapshot.engine);
    this.controlsNeuron.restore(snapshot.controls);
    this.servoNeuron.restore(snapshot.servo);
  }
}
