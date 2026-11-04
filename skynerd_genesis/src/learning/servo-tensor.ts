/**
 * THE STUDENT'S BRAIN — seven tensors connected by synapses into a
 * pulsating dynamic system, the servo policy being the assembly.
 *
 *   LocationTensor   alt x hsi | prev x element x skill
 *                     -> the LOCATION CLASS (192: 8 alt bands x 24
 *                        heading bands — "cruising at 3000 ft, heading
 *                        west"; action-independent: where the plane is)
 *   EnergyTensor     alt x asi x vsi x pitch x rpm | prev x element
 *                     x skill -> the ENERGY CLASS (16 named classes:
 *                        "incipient stall", "developed stall",
 *                        "gaining speed and losing altitude", "climbing",
 *                        "high angle of attack not enough speed"… — for
 *                        the recovery / target-altitude decisions)
 *   FlightTensor     vsi x asi x pitch x bank x turn | prev x element
 *                     x skill -> the FLIGHT CLASS (16 fine classes:
 *                        "yawing right, need left rudder", "banking
 *                        left, losing altitude"… — what to correct)
 *   ControlsTensor   aileron x elevator x rudder x throttle (at their
 *                     stops) | prev x element x skill -> the CONTROLS
 *                        CLASS (180: elevator x aileron x rudder x
 *                        throttle — "pulling up high", "ailerons left
 *                        low", "throttle cruise"…)
 *   GoalTensor       goal x element x situation-bus   -> new goal x element
 *   SkillTensor      the PATCH PANEL: every tensor's classification
 *                     paired with every other's        -> the wire's weight
 *   ServoTensor      the SKILL tensor's forwarded classifications
 *                                                   -> the four levers
 *
 * THE SIGNAL FLOW (nothing touches the Aircraft but the sensors):
 *
 *   Aircraft panel --(the only Aircraft input)--> the four SENSOR
 *   tensors. Each sensor CLASSIFIES its panel into a NAMED CLASS
 *   (instruments.ts owns the vocabularies and the deterministic
 *   TEACHER that labels any binned panel) — and each sensor is
 *   RECURRENT: its key carries its OWN previous class (its situation
 *   from itself), the GOAL tensor's held element (its situation from
 *   the goal tensor), and the SKILL patch bin for its wire to the
 *   goal (its situation from the skill tensor). The demonstration
 *   STAMPS the teacher's label; the runtime RECALLS the strongest
 *   class in the key's neighborhood, and falls back to the teacher
 *   when nothing clears the bar — so a panel the school never showed
 *   still classifies, deterministically, the way the instructor would.
 *
 *   The SKILL tensor is the PATCH PANEL: it stores the association
 *   weight of every PAIR of tensor signals — every sensor code with
 *   every other sensor code, each with the goal's element, and the
 *   servo's last command — a wire between every tensor and every
 *   other tensor. It fires all the wires at every decision; its
 *   recall is the learned strength of a connection.
 *
 *   The GOAL tensor is the FUZZY STATE MACHINE: (goal, element,
 *   situation-bus) -> (new goal, new element). The bus is the four
 *   sensors' codes merged; the situation axis recalls fuzzily.
 *
 *   The SERVO tensor receives ALL its input from the SKILL tensor —
 *   and nothing from the Aircraft, nothing even directly from the
 *   other tensors: the skill's fired message IS the servo's whole
 *   observation — the goal/element pair, the four sensor classes, and
 *   the patch panel's coherence bin. The skill initially FORWARDS the
 *   classifications exactly as they arrived; what it mixes in over
 *   training is its wires' verdicts — the patch bins in the sensors'
 *   keys, and the coherence bin in the forwarded payload.
 *
 * The hierarchy is in TIME: a tensor that receives a signal is
 * STUNNED; it relaxes through its refractory period; once relaxed it
 * FIRES its own state and GOSSIPS it downstream, stunning the
 * receiver in turn. The situation bus stuns the goal tensor a beat
 * when the situation changes (and the skill patch panel gates that
 * gossip: only news the wires agree on startles); the goal tensor's
 * transition stuns the servo tensor a beat when the intention
 * changes. The information travels the synapses in discrete pulses,
 * never an instantaneous feedback loop.
 *
 * Every tensor carries its own EMA time-decay stack (neurons.ts).
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
  LOCATION_SHAPE, ENERGY_SHAPE, FLIGHT_SHAPE, CONTROLS_SHAPE,
  SITUATION_AXIS,
  LOC_CLASS_AXIS, ENERGY_CLASS_AXIS, FLIGHT_CLASS_AXIS, CONTROLS_CLASS_AXIS,
  classifyLocation, classifyEnergy, classifyFlight, classifyControls,
  stopCount,
  type SenseReading,
} from "./instruments.js";
import { PulseNeuron, denoiseProjection } from "./neurons.js";
import { ErrorNeuron, TEACH_ETA } from "./error-neuron.js";

/** A lever the hand did not move this decision. */
export const ACTION_HOLD = -1;
/** The servo tensor's hold-all slot (the 5th channel of the key). */
export const HOLD_ALL_CHANNEL = 4;
/** Surface deflection change per decision, radians — the hand's speed. */
export const ACTION_DELTA = 0.05;
const ELIGIBILITY_DECISIONS = 10;        // credit the last N decisions (1 s of flight)
const ELIGIBILITY_DECAY = 0.7;

// THE INTENTION AXES. The goal axis: which higher-level goal is being
// realized (take off from the runway, turn left 180, recover from the
// spin…). The element axis: which atomic element of it is being executed
// right now (roll, rotate, climb, level off…) — school.ts owns the
// vocabulary.
export const GOAL_AXIS = 32;
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
 * The static/dynamic memory dial — each tensor's EMA coefficient on its
 * time-decay stack. Near 1 = static: lessons learned in the flight
 * school stay. Lower = dynamic: recent flights dominate, old habits
 * fade.
 */
export const NEURON_EMA = {
  location: 0.95,    // the location panel's codes
  energy: 0.95,      // the energy panel's codes
  flight: 0.90,      // the flight panel's codes
  controls: 0.95,    // the levers' codes
  skill: 0.98,       // the patch panel's wires — what hangs together
                     // stays together: the most static memory
  goal: 0.98,       // the transition map IS the curriculum
  servo: 0.90,       // the skill memory: lessons stay, recency matters
} as const;

/** The firing law's constants (neurons.ts denoiseProjection). */
const ELEMENT_NOISE_SIGMA = 0.15;
const ELEMENT_DETAIL_THRESHOLD = 0.20;
const ELEMENT_FIRE_THRESHOLD = 0.75;   // cuts the unobserved noise floor
const ACTION_NOISE_SIGMA = 0.05;
const ACTION_DETAIL_THRESHOLD = 0.05;
const ACTION_FIRE_THRESHOLD = 0.30;

/**
 * A candidate element's OWN transition cells must clear this bar —
 * a single stamped cell clears it; the unobserved floor does not; a
 * neighbor's blur leak never counts.
 */
export const ELEMENT_FIRE_BAR = 0.75;

/**
 * A situation code's recall must clear this bar for a panel to count
 * as KNOWN — below it the sensors fall back to the panel's hash, the
 * honest "never seen this before" code.
 */
const CODE_BAR = 0.5;

/** The skill patch bins: the wire's quantized weight. 4 bins. */
export const SKILL_BIN_AXIS = 4;
/** The patch weight's bin edges (a wire at baseline reads bin 0). */
const SKILL_BIN_EDGES = [0.75, 1.5, 3.0];
const binOfWeight = (weight: number): number => {
  let bin = 0;
  while (bin < SKILL_BIN_EDGES.length && weight >= SKILL_BIN_EDGES[bin]) {
    bin += 1;
  }
  return bin;
};

/** The patch bins: the wire's quantized weight. 4 bins. */

/** THE PULSE: the sensors' cadence. A tensor that receives a signal is
 *  stunned one decision and fires on the next — the sensors encode
 *  every PULSE_EVERY-th decision, and the demonstration stamps at the
 *  same rhythm. */
export const PULSE_EVERY = 2;

// --- THE PATCH PANEL's SIGNALS ---------------------------------------------
// Every tensor's output signal, on the patch panel's wire bank. The
// sensors' situation codes, the goal tensor's element, and the servo
// tensor's last command option — each one a signal the skill tensor
// can wire to any other.

/** The patch panel's signal bank: one slot per tensor output. */
export const PATCH_SIGNALS = [
  "location", "energy", "flight", "controls", "element", "servo",
] as const;
export type PatchSignal = typeof PATCH_SIGNALS[number];

/** The wire bank: every unordered pair of signals, one wire each. */
export const PATCH_WIRES: ReadonlyArray<readonly [PatchSignal, PatchSignal]> =
  (() => {
    const wires: Array<[PatchSignal, PatchSignal]> = [];
    for (let a = 0; a < PATCH_SIGNALS.length; a++) {
      for (let b = a + 1; b < PATCH_SIGNALS.length; b++) {
        wires.push([PATCH_SIGNALS[a], PATCH_SIGNALS[b]]);
      }
    }
    return wires;
  })();

/** The patch panel's wire count (every tensor with every other). */
export const PATCH_WIRE_AXIS = PATCH_WIRES.length;

/** The wire index of a signal pair. */
export function wireIndexOf(a: PatchSignal, b: PatchSignal): number {
  const index = PATCH_WIRES.findIndex(([x, y]) =>
    (x === a && y === b) || (x === b && y === a));
  return index;
}

/** Every signal rides the panel at a 192-wide slot — the widest
 *  class vocabulary (the location class) covers them all. */
export const PATCH_SLOT_AXIS = LOC_CLASS_AXIS;

/** One neuron's memory snapshot (the graduate the flock inherits from). */
export interface NeuronSnapshot {
  values: Map<number, Float32Array>;
  epochs: Map<number, Int32Array>;
}

/** The whole brain: one snapshot per tensor. */
export interface BrainSnapshot {
  location: NeuronSnapshot;
  energy: NeuronSnapshot;
  flight: NeuronSnapshot;
  controls: NeuronSnapshot;
  skill: NeuronSnapshot;
  goal: NeuronSnapshot;
  servo: NeuronSnapshot;
  /** The error neurons' learned signs (+1/−1, one per tensor) — the
   *  error pump's direction is part of the brain a clone inherits. */
  errorSigns: readonly number[];
}

/** The four sensors' situation codes — the signals everything else
 *  reads. These are what travels the synapses. */
export interface TensorSignals {
  readonly location: number;
  readonly energy: number;
  readonly flight: number;
  readonly controls: number;
}

/** The network's decision, one decision window's worth. */
export interface NetworkDecision {
  /** The element the goal tensor holds now (its gossiped intention). */
  readonly elementId: number;
  /** The servo tensor's command: per channel a lever stop, or
   *  ACTION_HOLD for the channels the hand did not move. */
  readonly actions: Int32Array;
  /** The situation bus state the decision was made at. */
  readonly situation: number;
  /** The servo tensor's observation key — the reinforcement key (the
   *  skill tensor's FORWARDED payload: the goal/element pair, the
   *  energy and flight classes, the patch coherence bin — the
   *  decision's whole input, arrived from the skill tensor only). */
  readonly key: readonly number[];
  /** Whether the servo tensor actually fired (false = stunned). */
  readonly acted: boolean;
}

/**
 * ONE FLIGHT'S RUNNING SIGNAL STATE — the recurrent context. The
 * sensors' previous codes (their situation from THEMSELVES), the goal
 * tensor's held element (its situation to the sensors), the skill
 * patch bins (the wires' situation to the sensors), and the servo's
 * last command. Owned by the flight (the demo or the student's
 * network), passed through every encode — explicit state, no hidden
 * globals.
 */
export class TensorContext {
  /** The sensors' CURRENT situation codes (their previous codes at
   *  encode time — the recurrence). */
  location = 0;
  energy = 0;
  flight = 0;
  controls = 0;
  /** The goal tensor's held element — the sensors' intention input. */
  element: number;
  /** The servo tensor's last command option (its signal on the patch
   *  panel). */
  lastOption = 0;
  /** The skill patch bins: each sensor's wire weight to the element,
   *  quantized — the sensors' skill input. */
  readonly skillBins: [number, number, number, number] = [0, 0, 0, 0];
  /** The patch panel's overall coherence for the current company
   *  (raw weight + its bin — the servo's skill input). */
  coherence = 0.0;
  coherenceBin = 0;

  /** The flight's current panel pulse — the only Aircraft input the
   *  network ever sees. The flight re-writes it at the head of every
   *  decision window (the PANEL messages stun the sensors; the LAW
   *  itself reads the pulse from here, so the gossip ping-pong can
   *  never starve the classification). */
  panels: readonly (readonly number[])[] = [[], [], [], []];

  constructor(element: number) {
    this.element = element;
  }

  /** The four sensors' current codes, as signals. */
  get signals(): TensorSignals {
    return { location: this.location, energy: this.energy,
             flight: this.flight, controls: this.controls };
  }

  /** One sensor's current code, by index (0 loc, 1 energy, 2 flight,
   *  3 controls) — the recurrence reads its own previous code. */
  signalOf(sensorIndex: number): number {
    return sensorIndex === 0 ? this.location
         : sensorIndex === 1 ? this.energy
         : sensorIndex === 2 ? this.flight
         : this.controls;
  }

  /** Set one sensor's current code, by index — the network's gossip
   *  writes what the sensor fired. */
  setCodeOf(sensorIndex: number, code: number): void {
    if (sensorIndex === 0) { this.location = code; }
    else if (sensorIndex === 1) { this.energy = code; }
    else if (sensorIndex === 2) { this.flight = code; }
    else { this.controls = code; }
  }
}

/**
 * THE TEACHER: the deterministic classification of any binned panel
 * (instruments.ts owns the vocabularies). The demonstration stamps
 * this label; the runtime falls back to it when its own memory has
 * nothing to say — so a panel the school never showed still
 * classifies exactly the way the instructor's own instruments would.
 */
function teacherClass(sensorIndex: number, panel: readonly number[]): number {
  return sensorIndex === 0
    ? classifyLocation(panel[0], panel[1])
    : sensorIndex === 1
    ? classifyEnergy(panel[0], panel[1], panel[2], panel[3], panel[4])
    : sensorIndex === 2
    ? classifyFlight(panel[0], panel[1], panel[2], panel[3], panel[4])
    : classifyControls(panel[0], panel[1], panel[2], panel[3]);
}

/**
 * Owns the seven tensors; encodes the panel into situation signals,
 * runs the fuzzy state machine, and chooses the commands. The
 * pulsating network (the stun and the gossip) lives in PulseNetwork —
 * one per flight, sharing this brain.
 */
export class ServoPolicy {
  /** 1. THE LOCATION TENSOR: alt x hsi x its own previous class x the
   *  goal's element x the skill patch bin -> the LOCATION CLASS (the
   *  packed panel: 8 altitude bands x 24 heading bands — where the
   *  plane is, action-independent). All axes exact. */
  readonly locationTensor = new PulseNeuron({
    name: "alt x hsi | prev x element x skill -> location class",
    keyShape: [...LOCATION_SHAPE, LOC_CLASS_AXIS, ELEMENT_AXIS,
               SKILL_BIN_AXIS, LOC_CLASS_AXIS],
    ema: NEURON_EMA.location, baseline: 0.1, maxWeight: 5.0, minWeight: 0.0,
  });
  /** 2. THE ENERGY TENSOR: alt x asi x vsi x pitch x rpm x its own
   *  previous class x the goal's element x the skill patch bin -> the
   *  ENERGY CLASS (the named vocabulary: "incipient stall", "climbing",
   *  "gaining speed and losing altitude"…). The sweeping gauges (asi,
   *  vsi, pitch) recall fuzzily; the discriminators (alt, rpm) stay
   *  exact. */
  readonly energyTensor = new PulseNeuron({
    name: "alt x asi x vsi x pitch x rpm | prev x element x skill -> energy class",
    keyShape: [...ENERGY_SHAPE, ENERGY_CLASS_AXIS, ELEMENT_AXIS,
               SKILL_BIN_AXIS, ENERGY_CLASS_AXIS],
    ema: NEURON_EMA.energy, baseline: 0.1, maxWeight: 5.0, minWeight: 0.0,
    fuzzyAxes: [1, 2, 3],
  });
  /** 3. THE FLIGHT TENSOR: vsi x asi x pitch x bank x turn x its own
   *  previous class x the goal's element x the skill patch bin -> the
   *  FLIGHT CLASS (the fine vocabulary: "yawing right, need left
   *  rudder", "banking left, losing altitude"…). The sweeping gauges
   *  (vsi, asi, pitch) recall fuzzily; bank and the turn coordinator
   *  stay exact. */
  readonly flightTensor = new PulseNeuron({
    name: "vsi x asi x pitch x bank x turn | prev x element x skill -> flight class",
    keyShape: [...FLIGHT_SHAPE, FLIGHT_CLASS_AXIS, ELEMENT_AXIS,
               SKILL_BIN_AXIS, FLIGHT_CLASS_AXIS],
    ema: NEURON_EMA.flight, baseline: 0.1, maxWeight: 5.0, minWeight: 0.0,
    fuzzyAxes: [0, 1, 2],
  });
  /** 4. THE CONTROLS TENSOR: the levers at their stops x its own
   *  previous class x the goal's element x the skill patch bin -> the
   *  CONTROLS CLASS (the packed vocabulary: elevator x aileron x
   *  rudder x throttle — "pulling up high", "ailerons left low",
   *  "throttle cruise"…). All axes exact: a lever's stop is the truth. */
  readonly controlsTensor = new PulseNeuron({
    name: "aileron x elevator x rudder x throttle | prev x element x skill -> controls class",
    keyShape: [...CONTROLS_SHAPE, CONTROLS_CLASS_AXIS, ELEMENT_AXIS,
               SKILL_BIN_AXIS, CONTROLS_CLASS_AXIS],
    ema: NEURON_EMA.controls, baseline: 0.1, maxWeight: 5.0, minWeight: 0.0,
  });
  /** 5. THE SKILL TENSOR — THE PATCH PANEL: every tensor's signal
   *  paired with every other's, one WIRE per pair. The wire's weight
   *  is the learned association of the two signals occurring together;
   *  the wires are what connect every tensor with every other tensor. */
  readonly skillTensor = new PulseNeuron({
    name: "the patch panel: every signal paired with every other",
    keyShape: [PATCH_WIRE_AXIS, PATCH_SLOT_AXIS, PATCH_SLOT_AXIS],
    ema: NEURON_EMA.skill, baseline: 0.1, maxWeight: 5.0, minWeight: 0.0,
  });
  /** 6. THE GOAL TENSOR — THE FUZZY STATE MACHINE: goal x element x
   *  situation -> (new goal, new element). The situation axis recalls
   *  fuzzily: a bus code one step away is still the same news. */
  readonly goalTensor = new PulseNeuron({
    name: "goal x element x situation -> goal x element",
    keyShape: [GOAL_AXIS, ELEMENT_AXIS, SITUATION_AXIS, GOAL_AXIS, ELEMENT_AXIS],
    ema: NEURON_EMA.goal, baseline: 0.1, maxWeight: 5.0, minWeight: 0.0,
    fuzzyAxes: [2],
  });
  /** 7. THE SERVO TENSOR: THE SKILL TENSOR'S FORWARDED CLASSIFICATIONS
   *  -> the four levers at their stops. Its key is the skill's fired
   *  payload — the goal/element pair and the two classes that DRIVE a
   *  correction (the energy class: whether to recover, climb, or
   *  descend; the flight class: what to correct) plus the patch
   *  panel's coherence bin. The LOCATION class is action-independent
   * (the user's own semantics) and the CONTROLS class is the hand's
   *  own state — both stay in the bus and the wires, out of the key:
   *  keying on them would fragment the skill memory across every
   *  heading bin and every lever nudge. ALL input still arrives from
   *  the skill tensor, never from the Aircraft. The class axes and
   *  the coherence bin recall fuzzily; the elevator's vernier is the
   *  widest stop table. */
  readonly servoTensor = new PulseNeuron({
    name: "skill forward: goal x element x energy x flight x coherence -> levers",
    keyShape: [GOAL_AXIS, ELEMENT_AXIS, ENERGY_CLASS_AXIS, FLIGHT_CLASS_AXIS,
               SKILL_BIN_AXIS, HOLD_ALL_CHANNEL + 1, stopCount(1)],
    ema: NEURON_EMA.servo, baseline: 0.1, maxWeight: 5.0, minWeight: 0.0,
    fuzzyAxes: [2, 3, 4],
  });
  epsilon = 0.5;

  // --- THE ERROR NEURONS ---------------------------------------------------
  // One per tensor (TENSOR_NODE order: location, energy, flight,
  // controls, goal, skill, servo). The local learning units: each
  // observes ITS tensor's error (the student against the instructor)
  // and pumps a signed, proportional delta into that tensor's own
  // cells — flipping its sign when its last pump made the error grow.
  // The sign is learned state: it rides the snapshot a clone inherits.
  readonly errorNeurons: readonly ErrorNeuron[] = [
    new ErrorNeuron(), new ErrorNeuron(), new ErrorNeuron(),
    new ErrorNeuron(), new ErrorNeuron(), new ErrorNeuron(),
    new ErrorNeuron(),
  ];

  /** One tensor's error neuron, by node id. */
  errorNeuronOf(nodeId: number): ErrorNeuron {
    return this.errorNeurons[nodeId];
  }

  // --- THE PRIORS ----------------------------------------------------------
  // The no-brainer knowledge: we KNOW what these tensors are, so we
  // know which combinations deserve high weights — a yawing-right
  // flight wants left rudder, an incipient stall wants the nose pushed
  // down and full power. The priors are IMPOSED as floors (never above
  // a trained cell), re-imposed every classroom cycle (they must not
  // die to the forgetting law), and SPREAD by the firing law's signal
  // processing: stochastic noise -> Gaussian [1 2 1]/4 -> Haar ->
  // threshold -> de-Haar over the option axis, and the fuzzy class
  // axes bridge the neighbouring classes at recall time.

  /** The prior plan: (key prefix, channel, level, weight) — computed
   *  once per student (each student's jittered its own way, so the
   *  flock differs from birth), re-imposed idempotently. */
  private priorPlan: Array<{ key: readonly number[]; channel: number;
                            level: number; weight: number }> | null = null;

  /** Build the prior plan: the no-brainer class->correction pairs,
   * stamped across the goal/element grid (the servo keys exactly), at
   * the cruise-nominal location and controls classes, with each
   * student's own jitter on the weights. */
  private buildPriorPlan(rng: NumpyPcg64): void {
    // The no-brainers: (energy class, flight class, channel, level,
    // weight). Channels: 0 aileron, 1 elevator, 2 rudder, 3 throttle;
    // levels index each lever's stop table. The stall-family priors
    // stand at every attitude a stall actually shows (wings level,
    // nose high, nose low); the rest at the neutral attitude.
    const NO_BRINERS: readonly (readonly number[])[] = [
      // the energy emergencies — the drone must decide recovery itself
      [9, 0, 1, 1, 3.0],   // developed stall: PUSH the nose down
      [9, 11, 1, 1, 3.0],  //   (nose high)
      [9, 12, 1, 1, 3.0],  //   (nose low)
      [9, 0, 3, 6, 2.0],   //   and full throttle
      [9, 11, 3, 6, 2.0],
      [9, 12, 3, 6, 2.0],
      [8, 0, 1, 2, 3.0],   // incipient stall: push
      [8, 11, 1, 2, 3.0],
      [8, 12, 1, 2, 3.0],
      [8, 0, 3, 6, 2.0],   //   and full throttle
      [8, 11, 3, 6, 2.0],
      [8, 12, 3, 6, 2.0],
      [7, 0, 1, 3, 2.5],   // high AoA, not enough speed: relax the pull
      [7, 11, 1, 3, 2.5],
      [7, 12, 1, 3, 2.5],
      [7, 0, 3, 6, 2.0],   //   and add power
      [7, 11, 3, 6, 2.0],
      [7, 12, 3, 6, 2.0],
      [2, 0, 1, 3, 1.5],   // climbing, losing speed: nose down
      [4, 0, 1, 8, 1.5],   // descending: slight back pressure
      [12, 0, 3, 5, 1.5],  // low-power descent: power up
      // the attitude corrections — the servo applies these itself
      [0, 3, 2, 2, 2.0],   // yawing right: LEFT rudder
      [0, 4, 2, 6, 2.0],   // yawing left: RIGHT rudder
      [0, 12, 1, 8, 2.5],  // nose low: pull up
      [0, 11, 1, 2, 2.0],  // nose high: ease the nose down
      [0, 9, 0, 6, 2.5],   // overbanked left: aileron RIGHT
      [0, 10, 0, 2, 2.5],  // overbanked right: aileron LEFT
      [0, 5, 1, 8, 1.5],   // banking left, losing altitude: back pressure
      [0, 6, 1, 8, 1.5],   // banking right, losing altitude: back pressure
    ];
    const plan: Array<{ key: readonly number[]; channel: number;
                       level: number; weight: number }> = [];
    for (const [energy, flight, channel, level, weight] of NO_BRINERS) {
      // The goal/element grid: the goal and element axes are exact
      // on the servo key, so the prior must stand at every goal the
      // curriculum and the swarm fly, in every element.
      for (let goal = 1; goal <= 21; goal++) {
        for (let element = 0; element <= 21; element++) {
          const jitter = 0.8 + 0.4 * rng.uniform();
          // The coherence-bin spread: a prior at bin 0 stands at its
          // full weight, at bin 1 (the wires agreeing) at half — the
          // Gaussian neighbour of the same belief.
          plan.push({
            key: [goal, element, energy, flight, 0],
            channel, level, weight: weight * jitter,
          });
          plan.push({
            key: [goal, element, energy, flight, 1],
            channel, level, weight: weight * jitter * 0.5,
          });
        }
      }
    }
    this.priorPlan = plan;
  }

  /**
   * IMPOSE the priors — the plan built once (each student's own
   * jitter), re-imposed every classroom cycle so the forgetting law
   * never kills a no-brainer. The SPREAD: each prior's option
   * projection runs the firing law's signal processing — stochastic
   * noise -> Gaussian -> Haar -> threshold -> de-Haar — so the high
   * weight bleeds onto the neighbouring stops of the same lever;
   * the fuzzy class axes carry the neighbourhood the rest of the way
   * at recall time.
   */
  installPriors(epoch: number, rng: NumpyPcg64): void {
    if (this.priorPlan === null) {
      this.buildPriorPlan(rng);
    }
    const plan = this.priorPlan!;
    const scores = new Float64Array(64).fill(0.0);
    for (const prior of plan) {
      // The option projection under this key: the prior's own cell,
      // then the law's noise -> Haar -> Gaussian -> threshold.
      scores.fill(0.0);
      scores[ServoPolicy.optionOf(prior.channel, prior.level)] = prior.weight;
      const fired = denoiseProjection(scores, rng, ELEMENT_NOISE_SIGMA,
                                      ELEMENT_DETAIL_THRESHOLD,
                                      ELEMENT_FIRE_THRESHOLD);
      for (let option = 0; option < ServoPolicy.OPTIONS; option++) {
        if (fired[option] > 0.0) {
          const [channel, level] = ServoPolicy.channelLevelOf(option);
          this.servoTensor.impose([...prior.key, channel, level],
                                  fired[option], epoch);
        }
      }
    }
    // The skill's no-brainer wires: the sensible companies cohere —
    // "yawing right" hangs with "rudder left", the stall hangs with
    // the push and the power. The wires raise the patch panel's
    // verdict for exactly the companies the priors believe in.
    const CONTROLS_TRIM = 45 + 15 + 5 + 3;
    const RUD_LEFT = 45 + 15 + 0 * 5 + 3, RUD_RIGHT = 45 + 15 + 2 * 5 + 3;
    const PUSH = 0 * 45 + 15 + 5 + 3, PULL = 2 * 45 + 15 + 5 + 3;
    const FULL_PWR = 45 + 15 + 5 + 4;
    const WIRE_F = (flightClass: number, controlsClass: number): void => {
      const wire = wireIndexOf("flight", "controls");
      this.skillTensor.impose([wire, flightClass, controlsClass],
                               3.0, epoch);
    };
    const WIRE_E = (energyClass: number, controlsClass: number): void => {
      const wire = wireIndexOf("energy", "controls");
      this.skillTensor.impose([wire, energyClass, controlsClass],
                               3.0, epoch);
    };
    WIRE_F(3, RUD_LEFT); WIRE_F(4, RUD_RIGHT);
    WIRE_F(9, 2 * 15 + 45 + 5 + 3);   // overbanked left <- aileron right
    WIRE_F(10, 0 * 15 + 45 + 5 + 3);  // overbanked right <- aileron left
    WIRE_F(12, PULL); WIRE_F(11, PUSH);
    WIRE_E(8, PUSH); WIRE_E(9, PUSH);
    WIRE_E(8, FULL_PWR); WIRE_E(9, FULL_PWR);
    WIRE_E(7, PUSH);
    WIRE_E(4, PULL);
    WIRE_F(0, CONTROLS_TRIM);   // the coordinated cruise company coheres
  }

  get storedCombinations(): number {
    return this.locationTensor.storedCombinations
         + this.energyTensor.storedCombinations
         + this.flightTensor.storedCombinations
         + this.controlsTensor.storedCombinations
         + this.skillTensor.storedCombinations
         + this.goalTensor.storedCombinations
         + this.servoTensor.storedCombinations;
  }

  /** One sensor's tensor, by index (0 loc, 1 energy, 2 flight,
   *  3 controls) — the runtime's firing laws read their own tensor. */
  sensorTensorOf(sensorIndex: number): PulseNeuron {
    return sensorIndex === 0 ? this.locationTensor
         : sensorIndex === 1 ? this.energyTensor
         : sensorIndex === 2 ? this.flightTensor
         : this.controlsTensor;
  }

  // --- THE SENSORS: Aircraft panel -> situation signals -------------------

  /** A sensor's full key: its panel, its OWN previous class (the
   *  situation from itself), the goal's element, and its skill patch
   *  bin (the situation from the goal and skill tensors). */
  private static sensorKey(panel: readonly number[], context: TensorContext,
                           sensorIndex: number): number[] {
    return [...panel, context.signalOf(sensorIndex), context.element,
            context.skillBins[sensorIndex]];
  }

  /**
   * THE PATCH PANEL's feedback to the sensors: each sensor's wire
   * weight to the goal's element, quantized into its patch bin. The
   * sensors receive their situation from the skill tensor here —
   * refreshed from the PREVIOUS codes before the encode, so the
   * recurrence stays causal (the wires judge what was, the sensors
   * encode what is).
   */
  refreshSkillBins(context: TensorContext): void {
    const sensors: Array<keyof TensorSignals> = ["location", "energy",
                                                 "flight", "controls"];
    for (let s = 0; s < 4; s++) {
      const wire = wireIndexOf(sensors[s], "element");
      const weight = this.skillTensor.recall(
        [wire, context.signalOf(s), context.element]);
      context.skillBins[s] = binOfWeight(weight);
    }
  }

  /** The patch panel's verdict on the current company: the mean
   *  weight of the six sensor-sensor wires. High = the four sensors
   *  agree (the reading is real news); low = they disagree (a gauge
   *  flickered). Also the servo's skill input, quantized. */
  private refreshCoherence(context: TensorContext): void {
    const sensors: Array<keyof TensorSignals> = ["location", "energy",
                                                 "flight", "controls"];
    let sum = 0.0;
    let wires = 0;
    for (let a = 0; a < 4; a++) {
      for (let b = a + 1; b < 4; b++) {
        sum += this.skillTensor.recall(
          [wireIndexOf(sensors[a], sensors[b]),
           context.signalOf(a), context.signalOf(b)]);
        wires += 1;
      }
    }
    context.coherence = sum / wires;
    context.coherenceBin = binOfWeight(context.coherence);
  }

  /** The situation bus state: the four sensors' codes, merged. The
   *  merge is the PLAIN sum — bit-identical at stamp and recall time. */
  busOf(signals: TensorSignals): number {
    return (signals.location + signals.energy + signals.flight
            + signals.controls) % SITUATION_AXIS;
  }

  // --- THE DEMONSTRATION STAMP --------------------------------------------

  /**
   * Flight school: the demonstration's decision, stamped into EVERY
   * tensor — the sensors fire their codes under the recurrent context,
   * the patch panel fires its wires, the goal tensor fires its
   * self-loop, and the servo tensor fires the instructor's levers.
   * This is the watch-and-remember channel, reward-free and immediate.
   * Returns the situation bus state the stamp landed at.
   */
  imprint(goalId: number, elementId: number, sense: SenseReading,
          actions: ArrayLike<number>, context: TensorContext,
          amount = 1.0, epoch = 0): number {
    context.element = elementId;
    this.refreshSkillBins(context);   // the wires judge the previous codes
    // The sensors fire: the panel CLAIMS ITS TEACHER'S CLASS under the
    // recurrent context — the class the instructor's own instruments
    // would read, stamped into the codebook.
    context.location = this.stampSensor(this.locationTensor,
      ServoPolicy.sensorKey(sense.location, context, 0), LOC_CLASS_AXIS,
      teacherClass(0, sense.location), amount, epoch);
    context.energy = this.stampSensor(this.energyTensor,
      ServoPolicy.sensorKey(sense.energy, context, 1), ENERGY_CLASS_AXIS,
      teacherClass(1, sense.energy), amount, epoch);
    context.flight = this.stampSensor(this.flightTensor,
      ServoPolicy.sensorKey(sense.flight, context, 2), FLIGHT_CLASS_AXIS,
      teacherClass(2, sense.flight), amount, epoch);
    context.controls = this.stampSensor(this.controlsTensor,
      ServoPolicy.sensorKey(sense.controls, context, 3), CONTROLS_CLASS_AXIS,
      teacherClass(3, sense.controls), amount, epoch);
    this.refreshCoherence(context);
    // THE PATCH PANEL fires: every wire, for this company of signals.
    this.stampWires(context, amount, epoch);
    // The goal tensor's self-loop: under this situation, the intention
    // holds. (The advance transitions are stamped by stampTransition.)
    const bus = this.busOf(context.signals);
    this.goalTensor.fire([goalId, elementId, bus, goalId, elementId],
                         amount, epoch);
    // The servo tensor: the decided levers' stops.
    for (const [channel, level] of ServoPolicy.changedChannels(actions)) {
      this.servoTensor.fire([...this.servoKey(goalId, elementId, context),
                             channel, level], amount, epoch);
    }
    context.lastOption = ServoPolicy.optionOfActions(actions);
    return bus;
  }

  /** A sensor's class for a key, stamped: the TEACHER's label for the
   *  panel — the demonstration teaches the classification itself —
   *  then the cell fires. Returns the class stamped. */
  private stampSensor(tensor: PulseNeuron, key: readonly number[],
                      width: number, teacher: number, amount: number,
                      epoch: number): number {
    tensor.fire([...key, teacher], amount, epoch);
    return teacher;
  }

  /** The patch panel's wires, fired for the current company: every
   *  signal paired with every other, exactly as the panel is wired. */
  private stampWires(context: TensorContext, amount: number,
                    epoch: number): void {
    const slots: number[] = [
      context.location, context.energy, context.flight, context.controls,
      context.element, context.lastOption,
    ];
    for (let wire = 0; wire < PATCH_WIRES.length; wire++) {
      const [a, b] = PATCH_WIRES[wire];
      this.skillTensor.fire([wire, slots[PATCH_SIGNALS.indexOf(a)],
                             slots[PATCH_SIGNALS.indexOf(b)]],
                            amount, epoch);
    }
  }

  /**
   * The goal tensor's TRANSITION stamp: the demonstration moved from
   * (element, situation) to the next element — the state transition the
   * solo repeat will recall to advance its own plan.
   */
  stampTransition(goalId: number, fromElementId: number, situation: number,
                  toGoalId: number, toElementId: number,
                  amount = 1.0, epoch = 0): void {
    this.goalTensor.fire([goalId, fromElementId, situation, toGoalId,
                          toElementId], amount, epoch);
  }

  // --- THE STATE MACHINE AND THE COMMAND ------------------------------------

  /** The servo tensor's observation key — THE SKILL'S FORWARDED
   *  PAYLOAD: the goal/element pair, the energy and flight classes
   *  (the two that DRIVE a correction), and the patch panel's
   *  coherence bin. Bit-identical to what the skill tensor's fired
   *  message carries: the servo's whole input arrives from the skill
   *  tensor. */
  private servoKey(goalId: number, elementId: number,
                  context: TensorContext): number[] {
    return [goalId, elementId, context.energy, context.flight,
            context.coherenceBin];
  }

  /** The option layout: hold-all, then each lever's stops in channel
   *  order (aileron, elevator, rudder, throttle). */
  private static readonly OPTION_BASE: readonly number[] = (() => {
    const base = [1];
    for (let channel = 0; channel < 4; channel++) {
      base.push(base[channel] + stopCount(channel));
    }
    return base;   // [1, 10, 21, 30, 37]
  })();

  /** The option count: every lever's every stop, plus hold-all. */
  private static readonly OPTIONS =
    ServoPolicy.OPTION_BASE[4];

  /** The option index of a (channel, level) decision. */
  private static optionOf(channel: number, level: number): number {
    return ServoPolicy.OPTION_BASE[channel] + level;
  }

  /** The (channel, level) an option index names. */
  private static channelLevelOf(option: number): [number, number] {
    for (let channel = 3; channel >= 0; channel--) {
      if (option >= ServoPolicy.OPTION_BASE[channel]) {
        return [channel, option - ServoPolicy.OPTION_BASE[channel]];
      }
    }
    return [0, option];   // unreachable: option 0 is hold-all
  }

  /** The option index of an actions array (the servo's signal on the
   *  patch panel): hold-all or the moved lever's stop. */
  static optionOfActions(actions: ArrayLike<number>): number {
    for (const [channel, level] of ServoPolicy.changedChannels(actions)) {
      if (channel === HOLD_ALL_CHANNEL) {
        return 0;
      }
      return ServoPolicy.optionOf(channel, level);
    }
    return 0;
  }

  /** Which levers a decision moved (or the hold-all slot). */
  private static changedChannels(actions: ArrayLike<number>):
      Array<[number, number]> {
    const changed: Array<[number, number]> = [];
    for (let channel = 0; channel < 4; channel++) {
      const level = actions[channel];
      if (level !== ACTION_HOLD) {
        changed.push([channel, level]);
      }
    }
    if (changed.length === 0) {
      changed.push([HOLD_ALL_CHANNEL, 0]);
    }
    return changed;
  }

  /**
   * The servo tensor's COMMAND CHOICE: hold-all or one lever driven to
   * one of its stops, epsilon-greedily from the skill cells under the
   * SKILL TENSOR'S FORWARDED KEY (the servo's whole input), denoised
   * the way a neuron fires. Returns an actions array of 4 channel
   * entries, each ACTION_HOLD, with at most one set to a stop.
   */
  chooseControls(key: readonly number[], rng: NumpyPcg64): Int32Array {
    const actions = new Int32Array(4).fill(ACTION_HOLD);
    if (rng.uniform() < this.epsilon) {
      const choice = rng.integers(ServoPolicy.OPTIONS);   // hold-all or lever/level
      if (choice > 0) {
        const [channel, level] = ServoPolicy.channelLevelOf(choice);
        actions[channel] = level;
      }
      return actions;
    }
    // The skill cells' options, denoised (padded to 64 for Haar).
    const scores = new Float64Array(64).fill(0.0);
    for (let channel = 0; channel < 4; channel++) {
      for (let level = 0; level < stopCount(channel); level++) {
        scores[ServoPolicy.optionOf(channel, level)] = this.servoTensor.recall(
          [...key, channel, level]);
      }
    }
    scores[0] = this.servoTensor.recall([...key, HOLD_ALL_CHANNEL, 0]);
    const fired = denoiseProjection(scores, rng, ACTION_NOISE_SIGMA,
                                    ACTION_DETAIL_THRESHOLD,
                                    ACTION_FIRE_THRESHOLD);
    let choice = 0;
    let best = fired[0];
    for (let option = 1; option < ServoPolicy.OPTIONS; option++) {
      if (fired[option] > best) {
        best = fired[option];
        choice = option;
      }
    }
    if (choice > 0) {
      const [channel, level] = ServoPolicy.channelLevelOf(choice);
      actions[channel] = level;
    }
    return actions;
  }

  /** Drive the moved lever toward its commanded stop, at the hand's
   *  speed, clamped to the hardware. */
  commandFrom(current: Float64Array, actions: Int32Array): Float64Array {
    const targets = Float64Array.from(current);
    for (let channel = 0; channel < 4; channel++) {
      const level = actions[channel];
      if (level === ACTION_HOLD) {
        continue;
      }
      const [low, high] = SURFACE_LIMITS[channel];
      const steps = stopCount(channel) - 1;
      const target = low + ((high - low) * level) / steps;
      const delta = Math.max(-ACTION_DELTA,
        Math.min(ACTION_DELTA, target - targets[channel]));
      targets[channel] = Math.max(low, Math.min(high, targets[channel] + delta));
    }
    return targets;
  }

  /**
   * Credit recent decisions with decaying eligibility — the servo
   * tensor's cells only: the skill memory is what flight outcomes
   * teach. Each entry's key is the observation prefix
   * [goal, element, the four sensor codes, the coherence bin]; the
   * credited cell is whichever lever and stop the decision took (or
   * the hold-all slot).
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
      for (const [channel, level] of ServoPolicy.changedChannels(actions)) {
        this.servoTensor.fire([...key, channel, level], amount, epoch);
      }
      amount *= ELIGIBILITY_DECAY;
    }
  }

  /**
   * Forget across every tensor: keys nobody has re-fired within the
   * span die (the endless classroom's memory bound). Returns how
   * many died.
   */
  prune(beforeEpoch: number): number {
    return this.locationTensor.prune(beforeEpoch)
         + this.energyTensor.prune(beforeEpoch)
         + this.flightTensor.prune(beforeEpoch)
         + this.controlsTensor.prune(beforeEpoch)
         + this.skillTensor.prune(beforeEpoch)
         + this.goalTensor.prune(beforeEpoch)
         + this.servoTensor.prune(beforeEpoch);
  }

  /** Copy the whole brain: the graduate the flock inherits from. */
  snapshot(): BrainSnapshot {
    return {
      location: this.locationTensor.snapshot(),
      energy: this.energyTensor.snapshot(),
      flight: this.flightTensor.snapshot(),
      controls: this.controlsTensor.snapshot(),
      skill: this.skillTensor.snapshot(),
      goal: this.goalTensor.snapshot(),
      servo: this.servoTensor.snapshot(),
      errorSigns: this.errorNeurons.map((neuron) => neuron.signNow),
    };
  }

  /** Overwrite the brain with a snapshot (a newborn inheriting a graduate). */
  restore(snapshot: BrainSnapshot): void {
    this.locationTensor.restore(snapshot.location);
    this.energyTensor.restore(snapshot.energy);
    this.flightTensor.restore(snapshot.flight);
    this.controlsTensor.restore(snapshot.controls);
    this.skillTensor.restore(snapshot.skill);
    this.goalTensor.restore(snapshot.goal);
    this.servoTensor.restore(snapshot.servo);
    for (let i = 0; i < this.errorNeurons.length
                    && i < snapshot.errorSigns.length; i++) {
      this.errorNeurons[i].restoreSign(snapshot.errorSigns[i]);
      this.errorNeurons[i].resetWindow();
    }
  }
}


// ===========================================================================
// THE DISPATCHER RUNTIME — the Python mechanism, verbatim:
//
//   input message -> reward weights -> stun (set timer, logsig timeout)
//                 -> into dispatcher queue
//   dispatcher -> dispatch (timer)  -> handle timer -> the firing law
//                                 -> make message -> send message
//   dispatcher -> dispatch (message) -> handle message (reward, stun)…
//
// APPLIES TO EVERY TENSOR ALWAYS. Each tensor is a NODE on the shared
// dispatcher (net/events.ts EventQueue — the ported Python heap, same
// -time events dispatch in scheduling order) with ONE stun timer
// (net/events.ts StunTimer — re-arming replaces, the coalescing law).
// A handler NEVER sends: it rewards and re-arms; the TIMER'S fire is
// the only emitter.
//
// The firing law, every tensor, always:
//   recall the output projection -> stochastic noise -> Haar in all
//   axes -> Gaussian in all axes -> thresholding -> make message ->
//   send message. (denoiseProjection IS the law on one axis: noise ->
//   [1 2 1]/4 Gaussian -> Haar soft-threshold -> the noise floor cut.)
// ===========================================================================

import { EventQueue, StunTimer } from "../net/events.js";

/** The seven tensor node ids, in dispatch order. */
export const TENSOR_NODE = {
  location: 0,
  energy: 1,
  flight: 2,
  controls: 3,
  goal: 4,
  skill: 5,
  servo: 6,
} as const;
export type TensorNodeId = (typeof TENSOR_NODE)[keyof typeof TENSOR_NODE];

/** The flight's sender id (the consumer — sends the panel pulse, reads
 *  the fired command; not a tensor, never a recipient). */
const FLIGHT_SENDER_ID = 6 as TensorNodeId;

/** The message kinds that travel between tensor nodes. */
export type TensorMessageKind =
  | "panel"      // the flight -> a sensor: the Aircraft's sub-panel bins
  | "code"       // a sensor -> goal/skill: its named situation class
  | "element"    // the goal -> the sensors (recurrence) and the skill
  | "coherence"  // the skill -> sensors/goal: the wires' verdict
  | "forward"    // the skill -> the servo: ALL the classifications,
                 //   forwarded (the servo's whole input)
  | "command"    // the servo -> skill (its last option, a patch signal)
  | "error";     // THE ERROR WAVE (upstream, negative weight): the
                 //   shadow instructor's delta, pumped backwards
                 //   servo -> skill -> goal/sensors, each hop a stun

/** One tensor message on the dispatcher. */
export class TensorMessage {
  constructor(readonly senderId: TensorNodeId,
              readonly recipientId: TensorNodeId,
              readonly kind: TensorMessageKind,
              readonly payload: readonly number[],
              /** The message's reward weight — what the receiver's
               *  handler blends into its reward accumulator. */
              readonly weight: number,
              readonly sentAt: number) {}
}

/** THE LOGSIG TIMEOUT: the stun's length shrinks as the tensor's
 *  reward weight grows — a well-fed tensor relaxes fast, a starved
 *  one sulks. Deterministic, no dice. The window is sub-decision: a
 *  tensor's inbound messages all land within one decision, and the
 *  last re-arm must still fire inside the settle horizon, or the
 *  cascade would starve downstream tensors forever (the servo would
 *  never hear its own timer). */
const STUN_MIN = 0.1;    // decision units
const STUN_MAX = 0.6;
export function stunTimeout(reward: number): number {
  const logsig = 1.0 / (1.0 + Math.exp(-(reward - 1.5)));
  return STUN_MIN + (STUN_MAX - STUN_MIN) * (1.0 - logsig);
}

/**
 * One tensor as a dispatcher node. The reward weight is the node's
 * running confidence: every input message rewards it (EMA), the
 * reward sets the logsig stun timeout and scales the fired message's
 * weight. WHAT the firing law recalls differs per tensor; THAT it
 * stuns on input and fires on relaxation never does.
 */
class TensorNode {
  /** The running reward weight (EMA toward each message's weight). */
  reward = 0.1;
  /** The last staged input — the handler's stage (handlers never send;
   *  the law reads the NETWORK's state, the stage is only the
   *  trigger's evidence). */
  staged: TensorMessage | null = null;
  fires = 0;
  readonly stun: StunTimer;

  constructor(readonly id: TensorNodeId, queue: EventQueue,
              /** THE FIRING LAW: recall -> noise -> Haar -> Gaussian
               *  -> threshold -> the messages to send (null = the
               *  threshold cut everything: no message). */
              private readonly law: (node: TensorNode,
                                     staged: TensorMessage) =>
                TensorMessage[] | null,
              /** THE ERROR LAW (upstream): what the tensor does when
               *  the ERROR WAVE reaches it — the local pump into its
               *  own cells, then the error forwarded one hop up. */
              private readonly errorLaw: (node: TensorNode,
                                          staged: TensorMessage) =>
                TensorMessage[] | null = () => null) {
    this.stun = new StunTimer(queue, () => { this.fire(); });
  }

  /**
   * INPUT MESSAGE -> REWARD WEIGHTS -> STUN (set timer, logsig
   * timeout) -> INTO DISPATCHER QUEUE. The handler proper: it never
   * sends — it rewards the node and re-arms the single stun timer
   * (re-arming replaces: a stunned tensor absorbs the news and fires
   * ONCE, when it relaxes — the coalescing law).
   *
   * THE ERROR WAVE's size law: an ERROR message stuns for a delay
   * PROPORTIONAL to the size of the information hitting the tensor —
   * the bigger the error, the longer the tensor sits with it before
   * its error law runs. The feedback is a wave with a random-length
   * delay, exactly like the forward flow, only scaled by the error.
   */
  receive(message: TensorMessage): void {
    if (message.kind === "error") {
      this.reward = this.reward * 0.7 + message.weight * 0.3;
      this.staged = message;
      const size = Math.min(1.0, Math.abs(message.weight));
      this.stun.arm(STUN_MIN + (STUN_MAX - STUN_MIN) * size);
      return;
    }
    this.reward = this.reward * 0.7 + message.weight * 0.3;
    this.staged = message;
    this.stun.arm(stunTimeout(this.reward));
  }

  /** THE TIMER'S FIRE: the only emitter. The staged message is the
   *  trigger's EVIDENCE — handed to the law, then cleared. The firing
   *  law makes the messages and sends them (immediate dispatch,
   *  queue order). */
  private fire(): void {
    const staged = this.staged;
    const wasError = staged !== null && staged.kind === "error";
    this.staged = null;
    this.fires += 1;
    const messages = wasError ? this.errorLaw(this, staged!)
                              : this.law(this, staged!);
    if (messages === null) {
      return;
    }
    for (const message of messages) {
      this.network.deliver(message);
    }
  }

  private network!: TensorNetwork;
  /** The network wires itself in after construction. */
  bind(network: TensorNetwork): void {
    this.network = network;
  }
}

/** The logsig-scaled weight a fired message carries. */
function firedWeight(node: TensorNode): number {
  return node.reward;
}

/**
 * THE TENSOR NETWORK — one per flight, on its own dispatcher. The
 * flight steps it once per decision window: the panel pulse goes in,
 * then the dispatcher pumps until the cascade settles. The stun, the
 * relaxation, and the gossip all happen on the queue's time.
 */
export class TensorNetwork {
  private readonly queue = new EventQueue();
  private readonly nodes: readonly TensorNode[];
  private readonly rngs: readonly NumpyPcg64[];
  private readonly policy: ServoPolicy;
  readonly context: TensorContext;
  /** The goal the flight is flying (set each decision). */
  private goalId = GOAL_FREE_FLIGHT;
  /** The element the goal tensor holds (its gossiped intention). */
  private elementId: number;
  /** The maneuver chains (school.ts owns the vocabulary). */
  private readonly chains = new Map<number, readonly number[]>();
  /** The servo's fired command and observation key (the flight reads
   *  these; the reinforcement credits the key). */
  private firedActions: Int32Array | null = null;
  private firedKey: readonly number[] = [];
  /** The last gossip each tensor sent — THE NO-NEWS LAW: a firing law
   *  whose projection thresholds to what the tensor already said
   *  makes NO message (only news travels; the cascade settles instead
   *  of ping-ponging itself past the horizon). */
  private readonly sensorLastGossip:
      Array<{ panel: readonly number[]; code: number } | null> =
    [null, null, null, null];
  private goalLastGossip = "";
  private skillCoherenceLast = "";
  /** The sensors' recurrent keys from the last classification — where
   *  the error law's pump lands (on-policy: the exact cells the
   *  student's own flight will recall). */
  private readonly lastSensorKeys:
      Array<readonly number[]> = [[], [], [], []];
  /** The cascade's settle horizon, in decision units. The servo sits
   *  TWO stuns downstream (the skill must relax and forward before
   *  the servo can even be stunned), so the horizon must cover the
   *  whole chain: panel -> sensors -> goal/skill -> skill's forward
   *  -> servo's fire. */
  private static readonly SETTLE = 2.5;

  constructor(policy: ServoPolicy, startElementId: number,
              seedRng: NumpyPcg64) {
    this.policy = policy;
    this.elementId = startElementId;
    this.context = new TensorContext(startElementId);
    // Every tensor's own dice, derived deterministically from the
    // flight's seed — the firing law's stochastic noise.
    const rngs: NumpyPcg64[] = [];
    for (let id = 0; id < 7; id++) {
      rngs.push(new NumpyPcg64(seedRng.draw64(), seedRng.draw64()));
    }
    this.rngs = rngs;
    this.nodes = [
      new TensorNode(TENSOR_NODE.location, this.queue,
        (node) => this.sensorLaw(node, 0, LOC_CLASS_AXIS),
        (node, staged) => this.sensorErrorLaw(node, staged, 0, LOC_CLASS_AXIS)),
      new TensorNode(TENSOR_NODE.energy, this.queue,
        (node) => this.sensorLaw(node, 1, ENERGY_CLASS_AXIS),
        (node, staged) => this.sensorErrorLaw(node, staged, 1, ENERGY_CLASS_AXIS)),
      new TensorNode(TENSOR_NODE.flight, this.queue,
        (node) => this.sensorLaw(node, 2, FLIGHT_CLASS_AXIS),
        (node, staged) => this.sensorErrorLaw(node, staged, 2, FLIGHT_CLASS_AXIS)),
      new TensorNode(TENSOR_NODE.controls, this.queue,
        (node) => this.sensorLaw(node, 3, CONTROLS_CLASS_AXIS),
        (node, staged) => this.sensorErrorLaw(node, staged, 3, CONTROLS_CLASS_AXIS)),
      new TensorNode(TENSOR_NODE.goal, this.queue,
        (node) => this.goalLaw(),
        (node, staged) => this.goalErrorLaw(node, staged)),
      new TensorNode(TENSOR_NODE.skill, this.queue, () => this.skillLaw(),
        (node, staged) => this.skillErrorLaw(node, staged)),
      new TensorNode(TENSOR_NODE.servo, this.queue,
        (node, staged) => this.servoLaw(node, staged),
        (node, staged) => this.servoErrorLaw(node, staged)),
    ];
    for (const node of this.nodes) {
      node.bind(this);
    }
  }

  /** The dispatcher's delivery: a sent message lands in its
   *  recipient's handler, immediately, in send order. */
  deliver(message: TensorMessage): void {
    this.nodes[message.recipientId].receive(message);
  }

  // --- THE FIRING LAWS ------------------------------------------------------

  /**
   * A SENSOR'S LAW (all four, always): recall the CLASS projection
   * under the recurrent key — the CONTEXT'S CURRENT PANEL (the
   * flight's pulse — the staged evidence is only the stun trigger,
   * never the law's data), its OWN previous class, the goal's
   * element, its skill patch bin — then the law: noise -> Haar ->
   * Gaussian -> threshold -> the CODE message (the chosen class, or
   * the TEACHER's label when nothing clears the bar) to the goal and
   * the skill, direct. The class lands in the context — the skill's
   * forward and the bus read it.
   */
  private sensorLaw(node: TensorNode, sensorIndex: number,
                    width: number): TensorMessage[] | null {
    const panel = this.context.panels[sensorIndex];
    const context = this.context;
    const key = [...panel, context.signalOf(sensorIndex), context.element,
                 context.skillBins[sensorIndex]];
    this.lastSensorKeys[sensorIndex] = key;
    const tensor = this.policy.sensorTensorOf(sensorIndex);
    const projection = new Float64Array(width).fill(0.0);
    for (let code = 0; code < width; code++) {
      projection[code] = tensor.recall([...key, code]);
    }
    const fired = denoiseProjection(projection, this.rngs[node.id],
                                    ELEMENT_NOISE_SIGMA,
                                    ELEMENT_DETAIL_THRESHOLD,
                                    ELEMENT_FIRE_THRESHOLD);
    let best = -1;
    let bestValue = CODE_BAR;
    for (let code = 0; code < width; code++) {
      if (fired[code] > bestValue) {
        bestValue = fired[code];
        best = code;
      }
    }
    const code = best < 0 ? teacherClass(sensorIndex, panel) : best;
    context.setCodeOf(sensorIndex, code);
    // THE NO-NEWS LAW: the same panel, the same class — nothing to say.
    const last = this.sensorLastGossip[sensorIndex];
    if (last !== null && last.panel === panel && last.code === code) {
      return null;
    }
    this.sensorLastGossip[sensorIndex] = { panel, code };
    const weight = firedWeight(node);
    return [TENSOR_NODE.goal, TENSOR_NODE.skill]
      .map((recipient) => new TensorMessage(node.id, recipient, "code",
        [code], weight, this.queue.now));
  }

  /**
   * THE GOAL'S LAW — the fuzzy state machine (always): recall the
   * transition projection under (goal, element, bus) -> the law ->
   * threshold -> the ELEMENT message (the goal/element pair) to the
   * sensors (the recurrence) and the skill. Monotone: the choice never
   * regresses the chain. The servo no longer hears the goal directly —
   * the skill forwards the pair with everything else.
   */
  private goalLaw(): TensorMessage[] | null {
    const bus = this.busOf(this.context.signals);
    const candidates = this.chains.get(this.goalId);
    if (candidates !== undefined && candidates.length > 0) {
      const evidence = new Float64Array(32).fill(0.0);
      for (let i = 0; i < candidates.length; i++) {
        evidence[i] = this.policy.goalTensor.recall(
          [this.goalId, this.elementId, bus, this.goalId, candidates[i]]);
      }
      const node = this.nodes[TENSOR_NODE.goal];
      const fired = denoiseProjection(evidence, this.rngs[node.id],
                                      ELEMENT_NOISE_SIGMA,
                                      ELEMENT_DETAIL_THRESHOLD,
                                      ELEMENT_FIRE_THRESHOLD);
      let chosen = -1;
      for (let i = 0; i < candidates.length; i++) {
        if (evidence[i] > ELEMENT_FIRE_BAR
            && fired[i] > ELEMENT_FIRE_BAR) {
          chosen = i;
        }
      }
      if (chosen >= 0) {
        const current = candidates.indexOf(this.elementId);
        if (current < 0 || chosen >= current) {
          this.elementId = candidates[chosen];
        }
      }
    }
    const weight = firedWeight(this.nodes[TENSOR_NODE.goal]);
    // THE NO-NEWS LAW: the same goal/element — nothing to say (the
    // transition itself landed in the network's element already).
    const gossip = `${this.goalId},${this.elementId}`;
    if (gossip === this.goalLastGossip) {
      return null;
    }
    this.goalLastGossip = gossip;
    return [TENSOR_NODE.location, TENSOR_NODE.energy, TENSOR_NODE.flight,
            TENSOR_NODE.controls, TENSOR_NODE.skill]
      .map((recipient) => new TensorMessage(TENSOR_NODE.goal, recipient,
        "element", [this.goalId, this.elementId], weight,
        this.queue.now));
  }

  /**
   * THE SKILL'S LAW — the patch panel (always): refresh the wires'
   * verdict on the current company -> the law -> threshold -> the
   * COHERENCE message to the sensors (their patch bins) and the goal,
   * and THE FORWARD message to the servo: ALL the classifications
   * from every tensor — the goal/element pair and the four sensor
   * classes — plus the wires' own coherence bin. The forward IS the
   * servo's whole input. Initially this is pure forwarding; what the
   * training mixes in is the wires' verdicts (the patch bins already
   * in the sensors' keys, and the coherence bin riding the payload).
   */
  private skillLaw(): TensorMessage[] | null {
    const context = this.context;
    const sensors: Array<keyof TensorSignals> = ["location", "energy",
                                                 "flight", "controls"];
    for (let s = 0; s < 4; s++) {
      context.skillBins[s] = binOfWeight(this.policy.skillTensor.recall(
        [wireIndexOf(sensors[s], "element"), context.signalOf(s),
         context.element]));
    }
    let sum = 0.0;
    let wires = 0;
    for (let a = 0; a < 4; a++) {
      for (let b = a + 1; b < 4; b++) {
        sum += this.policy.skillTensor.recall(
          [wireIndexOf(sensors[a], sensors[b]),
           context.signalOf(a), context.signalOf(b)]);
        wires += 1;
      }
    }
    context.coherence = sum / wires;
    context.coherenceBin = binOfWeight(context.coherence);
    const weight = firedWeight(this.nodes[TENSOR_NODE.skill]);
    const messages: TensorMessage[] = [];
    // THE NO-NEWS LAW on both channels: the wires' verdict only when it
    // moved, the forward only when the classifications changed — a
    // steady situation is a held yoke, not a re-fired one.
    const coherence = [context.coherenceBin, ...context.skillBins];
    if (coherence.join(",") !== this.skillCoherenceLast) {
      this.skillCoherenceLast = coherence.join(",");
      for (const recipient of [TENSOR_NODE.location, TENSOR_NODE.energy,
                               TENSOR_NODE.flight, TENSOR_NODE.controls,
                               TENSOR_NODE.goal]) {
        messages.push(new TensorMessage(TENSOR_NODE.skill, recipient,
          "coherence", coherence, weight, this.queue.now));
      }
    }
    // THE FORWARD — the servo's whole input: bit-identical to the
    // demonstration's servo stamp key. ALWAYS sent — a steady input is
    // a steady command: the servo must hear its input every decision
    // to keep commanding it (the stick springs to trim between
    // commands). Only the coherence channel obeys the no-news law.
    messages.push(new TensorMessage(TENSOR_NODE.skill, TENSOR_NODE.servo,
      "forward", [this.goalId, this.elementId, context.energy,
                  context.flight, context.coherenceBin], weight,
      this.queue.now));
    return messages.length > 0 ? messages : null;
  }

  /**
   * THE SERVO'S LAW (always): the STAGED FORWARD message — the skill
   * tensor's forwarded classifications — IS the observation key (the
   * servo's whole input, never the Aircraft, never even the other
   * tensors directly). Recall the option scores under it -> the law ->
   * threshold -> the COMMAND message (the chosen levers) to the skill
   * (its patch signal). The flight reads the fired command itself.
   */
  private servoLaw(_node: TensorNode,
                   staged: TensorMessage): TensorMessage[] | null {
    if (staged.kind !== "forward") {
      return null;   // nothing forwarded: nothing to command
    }
    const key = staged.payload;
    const actions = this.policy.chooseControls(key,
      this.rngs[TENSOR_NODE.servo]);
    this.firedActions = actions;
    this.firedKey = key;
    const option = ServoPolicy.optionOfActions(actions);
    this.context.lastOption = option;
    return [new TensorMessage(TENSOR_NODE.servo, TENSOR_NODE.skill,
      "command", [option], firedWeight(this.nodes[TENSOR_NODE.servo]),
      this.queue.now)];
  }

  // --- THE ERROR WAVE (upstream) --------------------------------------------

  /** The epoch the current teach() pumps fire at. */
  private teachEpoch = 0;
  /** The error wave's own settle horizon — it must cover the whole
   *  upstream chain (servo -> skill -> goal -> sensors) at each hop's
   *  size-scaled stun. */
  private static readonly ERROR_SETTLE = 2.5;

  /** THE ERROR PAYLOAD (built once by teach, read by every error law):
   *  [0] the instructor's element; [1] the instructor's worst acting
   *  channel (-1 = hands off); [2] its commanded level; [3] the
   *  student's effective level on that channel; [4+2i] sensor i's
   *  EXPECTED class (the teacher on the student's own panel);
   *  [5+2i] sensor i's ACTUAL fired class. */
  private expectedClasses(): number[] {
    return [0, 1, 2, 3].map((i) =>
      teacherClass(i, this.context.panels[i]));
  }

  /**
   * THE SHADOW INSTRUCTOR'S TEACHING PULSE — called right after a
   * decision, while the instructor's own aircraft flies the same
   * maneuver beside the student. Computes every tensor's error against
   * the instructor (the servo: the instructor's commanded lever vs
   * the student's; the sensors: the teacher's class on the student's
   * own panel vs what its memory made it fire; the goal: the
   * instructor's element vs the student's) and injects ONE ERROR
   * MESSAGE at the bottom of the chain. The wave then travels UPSTREAM
   * on the same dispatcher as the forward flow — servo -> skill ->
   * goal -> the source sensors — each tensor stunned by the SIZE of
   * the error before its error law runs the local pump.
   */
  teach(instructorElementId: number, instructorActions: ArrayLike<number>,
        epoch: number): void {
    this.teachEpoch = Math.trunc(epoch);
    const panels = this.context.panels;
    // The instructor's worst acting channel against the student's
    // effective level (its fired command, or the lever it is parked
    // at — the controls panel reads the student's own stops).
    let worstChannel = -1;
    let worstLevel = 0;
    let worstStudent = 0;
    let worstError = 0.0;
    for (let channel = 0; channel < 4; channel++) {
      const level = instructorActions[channel];
      if (level === ACTION_HOLD) {
        continue;   // the instructor is not teaching this lever
      }
      const student = (this.firedActions !== null
                       && this.firedActions[channel] !== ACTION_HOLD)
        ? this.firedActions[channel] : panels[3][channel];
      const error = Math.abs(level - student)
                    / Math.max(1, stopCount(channel) - 1);
      if (error > worstError) {
        worstError = error;
        worstChannel = channel;
        worstLevel = level;
        worstStudent = student;
      }
    }
    // The sensors' class errors (the teacher on the student's own
    // panels against what its memory fired).
    const expected = this.expectedClasses();
    const widths = [LOC_CLASS_AXIS, ENERGY_CLASS_AXIS,
                   FLIGHT_CLASS_AXIS, CONTROLS_CLASS_AXIS];
    let classError = 0.0;
    for (let i = 0; i < 4; i++) {
      classError += Math.abs(expected[i] - this.context.signalOf(i))
                    / widths[i];
    }
    classError /= 4.0;
    if (worstChannel < 0 && classError <= 0.0) {
      return;   // the student matches the instructor: nothing to teach
    }
    const payload = [instructorElementId, worstChannel, worstLevel,
                     worstStudent,
                     expected[0], this.context.location,
                     expected[1], this.context.energy,
                     expected[2], this.context.flight,
                     expected[3], this.context.controls];
    const size = Math.max(worstError, classError);
    this.queue.now += 0.05;   // the wave rides after the forward cascade
    const entry = worstChannel >= 0 ? TENSOR_NODE.servo : TENSOR_NODE.skill;
    this.nodes[entry].receive(new TensorMessage(FLIGHT_SENDER_ID, entry,
      "error", payload, -Math.min(1.0, size), this.queue.now));
    const horizon = this.queue.now + TensorNetwork.ERROR_SETTLE;
    while (this.queue.nextTime() !== null
           && this.queue.nextTime()! <= horizon) {
      this.queue.step();
    }
  }

  /**
   * THE SERVO'S ERROR LAW: the instructor's commanded lever vs the
   * student's — raise the instructor's cell under the STUDENT'S OWN
   * observation key (on-policy: the key its own flight will recall),
   * depress the student's wrong cell, then send the error UP to the
   * skill (the back-propagating tensor) as a negative input.
   */
  private servoErrorLaw(node: TensorNode,
                        staged: TensorMessage): TensorMessage[] | null {
    const payload = staged.payload;
    const channel = payload[1];
    if (channel < 0 || this.firedKey.length === 0) {
      return null;   // hands off, or nothing observed yet
    }
    const level = payload[2];
    const student = payload[3];
    const error = Math.abs(level - student)
                  / Math.max(1, stopCount(channel) - 1);
    const pump = this.policy.errorNeuronOf(TENSOR_NODE.servo)
                            .pump(error);
    if (pump !== 0.0) {
      this.policy.servoTensor.fire(
        [...this.firedKey, channel, level], pump, this.teachEpoch);
      const acted = this.firedActions;
      if (acted !== null && acted[channel] !== ACTION_HOLD
          && (acted[channel] !== level)) {
        this.policy.servoTensor.fire(
          [...this.firedKey, channel, acted[channel]], -pump,
          this.teachEpoch);
      }
    }
    return [new TensorMessage(TENSOR_NODE.servo, TENSOR_NODE.skill,
      "error", payload, -Math.min(1.0, error), this.queue.now)];
  }

  /**
   * THE SKILL'S ERROR LAW — THE BACK-PROPAGATING TENSOR: it routes the
   * error wave upstream (a negative input to the goal and the four
   * source sensors), and it learns its own wires: the company the
   * instructor would have (the expected classes) hangs together MORE,
   * the wrong company the student actually flew hangs together LESS.
   */
  private skillErrorLaw(node: TensorNode,
                        staged: TensorMessage): TensorMessage[] | null {
    const payload = staged.payload;
    const expected = [payload[4], payload[6], payload[8], payload[10]];
    const actual = [payload[5], payload[7], payload[9], payload[11]];
    const widths = [LOC_CLASS_AXIS, ENERGY_CLASS_AXIS,
                    FLIGHT_CLASS_AXIS, CONTROLS_CLASS_AXIS];
    let error = 0.0;
    let wrong = false;
    for (let i = 0; i < 4; i++) {
      if (expected[i] !== actual[i]) {
        wrong = true;
        error += Math.abs(expected[i] - actual[i]) / widths[i];
      }
    }
    error /= 4.0;
    const pump = this.policy.errorNeuronOf(TENSOR_NODE.skill).pump(error);
    if (wrong && pump !== 0.0) {
      const sensors: Array<keyof TensorSignals> = ["location", "energy",
                                                   "flight", "controls"];
      for (let a = 0; a < 4; a++) {
        for (let b = a + 1; b < 4; b++) {
          const wire = wireIndexOf(sensors[a], sensors[b]);
          this.policy.skillTensor.fire([wire, expected[a], expected[b]],
                                        pump * 0.5, this.teachEpoch);
          this.policy.skillTensor.fire([wire, actual[a], actual[b]],
                                        -pump * 0.5, this.teachEpoch);
        }
      }
    }
    // Route the wave: the goal and the sensors each get their turn.
    const weight = -Math.min(1.0, Math.abs(staged.weight) + error);
    const messages = [TENSOR_NODE.goal, TENSOR_NODE.location,
                      TENSOR_NODE.energy, TENSOR_NODE.flight,
                      TENSOR_NODE.controls]
      .map((recipient) => new TensorMessage(TENSOR_NODE.skill, recipient,
        "error", payload, weight, this.queue.now));
    return wrong || payload[1] >= 0 ? messages : null;
  }

  /**
   * THE GOAL'S ERROR LAW: the instructor's element vs the student's —
   * pump the transition the plan SHOULD have taken (from the state
   * the student is in, under the bus it reads, to the instructor's
   * element), then forward the error to the sensors (the mirror of
   * the forward recurrence path goal -> sensors).
   */
  private goalErrorLaw(node: TensorNode,
                       staged: TensorMessage): TensorMessage[] | null {
    const instructorElement = staged.payload[0];
    if (instructorElement === this.elementId) {
      return null;   // the intention is right: nothing to pump
    }
    const pump = this.policy.errorNeuronOf(TENSOR_NODE.goal).pump(1.0);
    if (pump !== 0.0) {
      const bus = this.busOf(this.context.signals);
      this.policy.goalTensor.fire(
        [this.goalId, this.elementId, bus, this.goalId, instructorElement],
        pump, this.teachEpoch);
    }
    return [TENSOR_NODE.location, TENSOR_NODE.energy, TENSOR_NODE.flight,
            TENSOR_NODE.controls]
      .map((recipient) => new TensorMessage(TENSOR_NODE.goal, recipient,
        "error", staged.payload, staged.weight, this.queue.now));
  }

  /**
   * A SENSOR'S ERROR LAW — the wave's END, at the source tensors: the
   * teacher's class on the student's own panel vs what its memory
   * made it fire. When they differ, the memory overrode the teacher —
   * raise the right class under the student's own recurrent key,
   * depress the wrong one. This is the back-propagation INSIDE the
   * single tensor: input values (its panel) to output (its class).
   */
  private sensorErrorLaw(node: TensorNode, staged: TensorMessage,
                         sensorIndex: number,
                         width: number): TensorMessage[] | null {
    const payload = staged.payload;
    if (payload.length < 12) {
      return null;
    }
    const expected = payload[4 + 2 * sensorIndex];
    const actual = payload[5 + 2 * sensorIndex];
    if (expected === actual
        || this.lastSensorKeys[sensorIndex].length === 0) {
      return null;
    }
    const error = Math.abs(expected - actual) / width;
    const pump = this.policy.errorNeuronOf(sensorIndex).pump(error);
    if (pump !== 0.0) {
      const key = this.lastSensorKeys[sensorIndex];
      this.policy.sensorTensorOf(sensorIndex).fire([...key, expected],
        pump, this.teachEpoch);
      this.policy.sensorTensorOf(sensorIndex).fire([...key, actual],
        -pump, this.teachEpoch);
    }
    return null;   // the source sensors are the wave's end
  }

  // --- THE FLIGHT'S PUMP ----------------------------------------------------

  /**
   * One decision window: the panel pulse goes in — four PANEL
   * messages, the only input from the Aircraft — then the dispatcher
   * pumps until the cascade settles (the timers fire, the messages
   * travel, the handlers stun). Returns what the network holds.
   */
  decide(goalId: number, sense: SenseReading,
         candidates: readonly number[] | null,
         seedRng: NumpyPcg64): NetworkDecision {
    this.goalId = goalId;
    if (candidates !== null) {
      this.chains.set(goalId, candidates);
    }
    this.context.element = this.elementId;
    this.context.panels = [sense.location, sense.energy, sense.flight,
                            sense.controls];   // the panel pulse
    this.firedActions = null;   // a hold is a hold: no stale command
    this.queue.now += 1.0;   // one decision window of network time
    const panels: readonly number[][] = [sense.location, sense.energy,
                                          sense.flight, sense.controls];
    for (let s = 0; s < 4; s++) {
      this.nodes[s].receive(new TensorMessage(FLIGHT_SENDER_ID,
        s as TensorNodeId, "panel", panels[s], 1.0, this.queue.now));
    }
    // The standing invitations: the goal and the skill have news to
    // consider each decision (their handlers reward and stun; their
    // timers fire the laws once relaxed). The servo receives NOTHING
    // here — its only input is the skill's forward, once the skill
    // relaxes.
    this.nodes[TENSOR_NODE.goal].receive(new TensorMessage(
      FLIGHT_SENDER_ID, TENSOR_NODE.goal, "element", [goalId], 1.0,
      this.queue.now));
    this.nodes[TENSOR_NODE.skill].receive(new TensorMessage(
      FLIGHT_SENDER_ID, TENSOR_NODE.skill, "element", [this.elementId], 1.0,
      this.queue.now));
    // PUMP: dispatch until the cascade settles.
    const horizon = this.queue.now + TensorNetwork.SETTLE;
    while (this.queue.nextTime() !== null
           && this.queue.nextTime()! <= horizon) {
      this.queue.step();
    }
    const actions = this.firedActions
      ?? new Int32Array(4).fill(ACTION_HOLD);
    return {
      elementId: this.elementId,
      actions,
      situation: this.busOf(this.context.signals),
      key: this.firedKey,
      acted: this.firedActions !== null,
    };
  }

  /** The intention the network holds now (for narration + grading). */
  get currentElementId(): number {
    return this.elementId;
  }

  /** The situation bus state: the four sensors' codes, merged. */
  busOf(signals: TensorSignals): number {
    return (signals.location + signals.energy + signals.flight
            + signals.controls) % SITUATION_AXIS;
  }
}
