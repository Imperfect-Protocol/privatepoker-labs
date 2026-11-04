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
  labelLocationClass, labelEnergyClass, labelFlightClass, labelControlsClass,
  levelValue,
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

  // --- THE NO-BRAINER REFLEXES + THE PRIOR WIRES ----------------------------
  // The no-brainers are NOT memory: stamping them across the goal x
  // element grid cost ~35k cells per brain (and every transplanted
  // drone copies them all). They are a REFLEX — the same teacher-
  // fallback pattern the sensors use: when the servo's own cells
  // recall nothing above the fire bar under the skill's forward, the
  // built-in pilot reflex answers (a stall always pushes, a yaw
  // always rudders against it), deterministically, forever, at zero
  // memory cost. What IS memory: the SKILL WIRES (the sensible
  // companies cohering), imposed as floors and re-imposed every
  // classroom cycle so the forgetting law never kills them, each
  // student's jitter its own (the flock differs from birth), then
  // SPREAD by the firing law's signal processing (a).

    /** THE SERVO'S REFLEX TABLE: (energy class, flight class) -> the
   *  correction (channel, level). Channels: 0 aileron, 1 elevator,
   *  2 rudder, 3 throttle; levels index each lever's stop table. */
  private static readonly SERVO_REFLEXES: ReadonlyMap<string,
      readonly [number, number]> = new Map([
    // the energy emergencies — the drone must decide recovery itself
    ["9,0", [1, 1]], ["9,11", [1, 1]], ["9,12", [1, 1]],   // developed stall: PUSH
    ["8,0", [1, 2]], ["8,11", [1, 2]], ["8,12", [1, 2]],    // incipient stall: push
    ["7,0", [1, 3]], ["7,11", [1, 3]], ["7,12", [1, 3]],    // high AoA: relax the pull
    ["2,0", [1, 3]],                                        // climbing, losing speed
    ["4,0", [1, 8]],                                        // descending: back pressure
    ["12,0", [3, 5]],                                       // low-power descent: power
    // the attitude corrections — the servo applies these itself
    ["0,3", [2, 2]],    // yawing right: LEFT rudder
    ["0,4", [2, 6]],    // yawing left: RIGHT rudder
    ["0,12", [1, 8]],   // nose low: pull up
    ["0,11", [1, 2]],   // nose high: ease the nose down
    ["0,9", [0, 6]],    // overbanked left: aileron RIGHT
    ["0,10", [0, 2]],   // overbanked right: aileron LEFT
    ["0,5", [1, 8]],    // banking left, losing altitude
    ["0,6", [1, 8]],    // banking right, losing altitude
  ]);

  /** THE REFLEX: the no-brainer correction for a forwarded key, or
   *  null when the situation has no reflex. Key layout: [goal,
   *  element, energy, flight, coherence]. */
  static servoReflex(key: readonly number[]):
      readonly [number, number] | null {
    return ServoPolicy.SERVO_REFLEXES.get(
      `${key[2]},${key[3]}`) ?? null;
  }

  /** THE STALL-RECOVERY REFLEX's power half: the stall family always
   *  adds full throttle, whatever the pitch correction did. */
  private static readonly STALL_FAMILY = new Set([8, 9]);
  private static readonly FULL_THROTTLE_LEVEL = 6;   // stopCount(3) - 1

  /** THE STALL FAMILY'S POWER LAW: a stall-family state always adds
   *  full throttle, whatever else the servo chose — a push without
   *  power recovers nothing, and a plane parked on the runway (the
   *  teacher's "stall" reading at zero airspeed) never rolls. The
   *  chosen correction rides on top; the power is unconditional. */
  private static applyStallPower(key: readonly number[],
                                 actions: Int32Array): void {
    if (ServoPolicy.STALL_FAMILY.has(key[2])) {
      actions[3] = ServoPolicy.FULL_THROTTLE_LEVEL;
    }
  }

  /**
   * IMPOSE THE PRIOR WIRES — the sensible companies cohering on the
   * patch panel ("yawing right" hangs with "rudder left", the stall
   * hangs with the push and the power), imposed as floors, each
   * student's own jitter on the weight, re-imposed every classroom
   * cycle so the forgetting law never kills them. The (a) spread runs
   * at the first consolidation (the wires ride the dirty set).
   */
  installPriors(epoch: number, rng: NumpyPcg64): void {
    const jitter = (): number => 0.8 + 0.4 * rng.uniform();
    const CONTROLS_TRIM = 45 + 15 + 5 + 3;
    const RUD_LEFT = 45 + 15 + 0 * 5 + 3, RUD_RIGHT = 45 + 15 + 2 * 5 + 3;
    const PUSH = 0 * 45 + 15 + 5 + 3, PULL = 2 * 45 + 15 + 5 + 3;
    const FULL_PWR = 45 + 15 + 5 + 4;
    const WIRE_F = (flightClass: number, controlsClass: number): void => {
      this.skillTensor.impose([wireIndexOf("flight", "controls"),
        flightClass, controlsClass], 3.0 * jitter(), epoch);
    };
    const WIRE_E = (energyClass: number, controlsClass: number): void => {
      this.skillTensor.impose([wireIndexOf("energy", "controls"),
        energyClass, controlsClass], 3.0 * jitter(), epoch);
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
  static sensorKey(panel: readonly number[], context: TensorContext,
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
  static readonly OPTIONS =
    ServoPolicy.OPTION_BASE[4];

  /** The option index of a (channel, level) decision. */
  static optionOf(channel: number, level: number): number {
    return ServoPolicy.OPTION_BASE[channel] + level;
  }

  /** The (channel, level) an option index names. */
  static channelLevelOf(option: number): [number, number] {
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
  static changedChannels(actions: ArrayLike<number>):
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
    // The skill cells' options — PURE RECALL: argmax over the option
    // cells under the forwarded key (the noise -> Haar -> Gaussian ->
    // threshold pass belongs to the consolidations, never here).
    let choice = 0;
    let best = this.servoTensor.recall(
      [...key, HOLD_ALL_CHANNEL, 0]);
    const buf = [...key, 0, 0];
    for (let channel = 0; channel < 4; channel++) {
      buf[buf.length - 2] = channel;
      for (let level = 0; level < stopCount(channel); level++) {
        buf[buf.length - 1] = level;
        const value = this.servoTensor.recall(buf);
        const option = ServoPolicy.optionOf(channel, level);
        if (value > best) {
          best = value;
          choice = option;
        }
      }
    }
    // THE REFLEX — the teacher-fallback pattern: when nothing the
    // servo ever learned clears the fire bar under this key, the
    // built-in pilot reflex answers (a stall pushes, a yaw rudders
    // against it) — deterministically, forever, at no memory cost.
    // Training overrides it the moment a real cell clears the bar.
    if (choice === 0 && best < ACTION_FIRE_THRESHOLD) {
      const reflex = ServoPolicy.servoReflex(key);
      if (reflex !== null) {
        actions[reflex[0]] = reflex[1];
        ServoPolicy.applyStallPower(key, actions);
        return actions;
      }
    }
    if (choice > 0) {
      const [channel, level] = ServoPolicy.channelLevelOf(choice);
      actions[channel] = level;
    }
    ServoPolicy.applyStallPower(key, actions);
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

  // --- THE CONSOLIDATION ----------------------------------------------------
  // The signal-processing pass — stochastic noise -> Gaussian
  // [1 2 1]/4 -> Haar -> threshold -> de-Haar — run ONLY at the three
  // moments it belongs: (a) after the priors are imposed
  // (installPriors), (b) ONCE PER COMPLETED REPETITION (here), and
  // (c) when a tensor relaxes from a LONG stun (the network's
  // consolidateAround). Never per decision — that was the slow way.
  //
  // The pass reads each cell's NEIGHBORHOOD — a +/-2 window around
  // what the tensor knows (the Gaussian's actual reach) — runs the
  // law over the window, and writes back: the spread raises the
  // neighbors of a strong belief, the threshold cuts the noise floor
  // of a weak one. Cost is O(cells), not O(axis^2): a full brain
  // consolidates in one pass.

  /** One window's spread: read the cells, run the law, write back.
   *  NOISE LIVES IN THE SKILL TENSOR ALONE — the patch panel's wires
   *  explore their relationships stochastically; the factual tensors
   *  (the sensors' classes, the goal's transitions, the servo's
   *  levers) are EVIDENCE: their spread is the deterministic
   *  Gaussian -> Haar -> threshold -> de-Haar, no random cell ever
   *  invented. */
  static spreadWindow(tensor: PulseNeuron, prefix: readonly number[],
                              centers: readonly number[], axis: number,
                              tailKey: (tail: number) => number[],
                              rng: NumpyPcg64, epoch: number,
                              noiseSigma = 0.0): void {
    // The +/-2 window around every center, unique, sorted, EVEN
    // (Haar pairs).
    const windowSet = new Set<number>();
    for (const center of centers) {
      for (let d = -2; d <= 2; d++) {
        const w = center + d;
        if (w >= 0 && w < axis) {
          windowSet.add(w);
        }
      }
    }
    const cells = [...windowSet].sort((a, b) => a - b);
    if (cells.length < 2) {
      return;
    }
    if (cells.length % 2 === 1) {
      const next = cells[cells.length - 1] + 1;
      if (next < axis) {
        cells.push(next);
      } else if (cells[0] > 0) {
        cells.unshift(cells[0] - 1);
      } else {
        return;
      }
    }
    const scores = new Float64Array(cells.length);
    for (let i = 0; i < cells.length; i++) {
      scores[i] = tensor.peek([...prefix, ...tailKey(cells[i])]);
    }
    // Nothing stored in the window: nothing to spread — the pass
    // spreads BELIEFS over their neighbors, it never invents them
    // out of the noise floor (free flight files no goal map).
    let known = false;
    for (let i = 0; i < cells.length; i++) {
      if (scores[i] > tensor.baseline + 1e-6) {
        known = true;
        break;
      }
    }
    if (!known) {
      return;
    }
    const fired = denoiseProjection(scores, rng, noiseSigma,
                                    ELEMENT_DETAIL_THRESHOLD,
                                    ELEMENT_FIRE_THRESHOLD);
    for (let i = 0; i < cells.length; i++) {
      const delta = fired[i] - scores[i];
      if (Math.abs(delta) <= 0.02) {
        continue;
      }
      const key = [...prefix, ...tailKey(cells[i])];
      if (delta > 0.0) {
        tensor.impose(key, scores[i] + delta, epoch);
      } else {
        tensor.fire(key, delta, epoch);   // the threshold's cut: toward baseline
      }
    }
  }

  /**
   * (B) THE POST-REPETITION CONSOLIDATION — INCREMENTAL: only the
   *  windows the training FIRED or IMPOSED since the last pass (the
   *  neuron's dirty set), never the whole brain. Each touched cell's
   *  group runs the spread once; the work is proportional to what
   *  the repetition taught, and the dirty set clears.
   */
  consolidate(epoch: number, rng: NumpyPcg64): void {
    const classTail = (tail: number): number[] => [tail];
    const optionTail = (tail: number): number[] =>
      tail === 0 ? [HOLD_ALL_CHANNEL, 0]
                 : ServoPolicy.channelLevelOf(tail);
    const group = (tensor: PulseNeuron, tailLen: number, axis: number,
                   toOption: boolean,
                   tailKey: (tail: number) => number[],
                   noiseSigma = 0.0): void => {
      if (tensor.dirty.size === 0) {
        return;
      }
      const groups = new Map<string, { prefix: number[]; tails: number[] }>();
      for (const index of tensor.dirty) {
        const key = tensor.keyOf(index);
        const prefix = key.slice(0, key.length - tailLen);
        const tail = toOption
          ? (key[key.length - 2] === HOLD_ALL_CHANNEL ? 0
             : ServoPolicy.optionOf(key[key.length - 2], key[key.length - 1]))
          : key[key.length - 1];
        const gid = prefix.join(",");
        let g = groups.get(gid);
        if (g === undefined) {
          g = { prefix, tails: [] };
          groups.set(gid, g);
        }
        if (!g.tails.includes(tail)) {
          g.tails.push(tail);
        }
      }
      tensor.dirty.clear();
      for (const g of groups.values()) {
        ServoPolicy.spreadWindow(tensor, g.prefix, g.tails, axis, tailKey,
                                 rng, epoch, noiseSigma);
      }
    };
    group(this.locationTensor, 1, LOC_CLASS_AXIS, false, classTail);
    group(this.energyTensor, 1, ENERGY_CLASS_AXIS, false, classTail);
    group(this.flightTensor, 1, FLIGHT_CLASS_AXIS, false, classTail);
    group(this.controlsTensor, 1, CONTROLS_CLASS_AXIS, false, classTail);
    // THE SKILL TENSOR — the only NOISY one: the patch panel's wires
    // explore their relationships stochastically; every other tensor
    // spreads deterministically.
    group(this.skillTensor, 1, PATCH_SLOT_AXIS, false, classTail,
          ELEMENT_NOISE_SIGMA);
    group(this.goalTensor, 1, ELEMENT_AXIS, false, classTail);
    group(this.servoTensor, 2, ServoPolicy.OPTIONS, true, optionTail);
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
// THE DISPATCHER RUNTIME — the Python mechanism, REDESIGNED FOR SPEED:
//
//   input message -> reward weights -> stun (set timer, ISRLU timeout)
//                 -> the node waits, stunned
//   dispatcher -> dispatch (timer)  -> handle timer -> the firing law
//                                 -> make message -> send message
//   dispatcher -> dispatch (message) -> handle message (reward, stun)…
//
// APPLIES TO EVERY TENSOR ALWAYS — but with NONE of the machinery the
// old queue paid for. No heap, no cancellable timers, no re-armed
// entries: each tensor IS its own timer. The stun's timeout lives IN
// the node (relaxAt); the dispatcher is a plain SWEEP — find the
// earliest relaxation among the stunned nodes, advance the clock to
// it, fire every node whose stun relaxed (in node-id order), repeat
// until the cascade settles or the decision window's horizon passes.
// Same coalescing law (a stunned tensor absorbs the news and fires
// ONCE, when it relaxes), same determinism (fixed sweep order), one
// pass over seven plain objects per step. A handler NEVER sends: it
// rewards and re-arms; the RELAXATION is the only emitter.
//
// The firing law's signal processing — stochastic noise -> Haar in
// all axes -> Gaussian in all axes -> thresholding -> de-Haar —
// runs at the three moments it belongs and NOWHERE per decision:
//   (a) after the priors are imposed (installPriors),
//   (b) once per completed repetition (ServoPolicy.consolidate),
//   (c) when a tensor relaxes from a LONG stun
//       (TensorNetwork.consolidateAround).
// The per-decision laws are pure recall: what the tensor remembers,
// it says.
// ===========================================================================

/**
 * THE STUN TIMEOUT — ISRLU (the inverse square root linear unit,
 * Carlisle et al. 2017), the cheap saturating ramp: no exp, no log,
 * one sqrt. f(x) = x/sqrt(1+ax^2) above zero (saturates at 1/sqrt(a)),
 * x*sqrt(1+ax^2) below (the leaky side — negative news flows
 * through, steeper than linear). The reward's reach is folded into
 * [0,1]: a well-fed tensor relaxes fast, a starved one sulks.
 * The window is sub-decision: the last re-arm must still fire inside
 * the settle horizon, or the cascade starves downstream tensors.
 */
const ISRLU_ALPHA = 1.0;
const STUN_MIN = 0.1;    // decision units
const STUN_MAX = 0.6;
/** A stun at least this long is a LONG stun: the tensor was hit hard
 *  (a big error wave, or starved reward) — when it relaxes, it also
 *  consolidates its slice (c). */
const LONG_STUN_BAR = 0.5;
/** How often ONE tensor may consolidate on long-stun relaxes — the
 *  cooldown that keeps (c) cheap (once per 25 decision windows). */
const CONSOLIDATE_COOLDOWN = 25.0;

export function isrlu(x: number, alpha: number = ISRLU_ALPHA): number {
  return x >= 0.0
    ? x / Math.sqrt(1.0 + alpha * x * x)
    : x * Math.sqrt(1.0 + alpha * x * x);
}

export function stunTimeout(reward: number): number {
  const reach = Math.min(1.0, Math.max(0.0, 0.5 + 0.5 * isrlu(reward - 1.5)));
  return STUN_MIN + (STUN_MAX - STUN_MIN) * (1.0 - reach);
}

/** The seven tensor node ids, in sweep order. */
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

/** One tensor message on the dispatcher — a lightweight plain object
 *  (six fields, nothing else). */
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

/**
 * One tensor as a dispatcher node — A PLAIN OBJECT WITH A CLOCK
 * FIELD, nothing scheduled anywhere. The reward weight is the node's
 * running confidence: every input message rewards it (EMA), the
 * reward sets the ISRLU stun timeout and scales the fired message's
 * weight. WHAT the firing law recalls differs per tensor; THAT it
 * stuns on input and fires on relaxation never does.
 */
class TensorNode {
  /** The running reward weight (EMA toward each message's weight). */
  reward = 0.1;
  /** The last staged input — the handler's stage (handlers never
   *  send; the law reads the NETWORK's state, the stage is only the
   *  trigger's evidence). */
  staged: TensorMessage | null = null;
  fires = 0;
  /** THE STUN TIMER: the dispatcher time this node relaxes at
   *  (Infinity = not stunned). Receiving re-arms it — coalescing,
   *  without cancelling anything. */
  relaxAt = Infinity;
  /** The length of the stun being served (the long-stun law (c)). */
  armedDelay = 0.0;
  /** The network time of this node's last (c) consolidation. */
  lastConsolidate = -Infinity;

  constructor(readonly id: TensorNodeId,
              /** THE FIRING LAW: what the tensor says when it relaxes
               *  (null = no message: no news). */
              private readonly law: (node: TensorNode,
                                     staged: TensorMessage) =>
                TensorMessage[] | null,
              /** THE ERROR LAW (upstream): what the tensor does when
               *  the ERROR WAVE reaches it. */
              private readonly errorLaw: (node: TensorNode,
                                           staged: TensorMessage) =>
                TensorMessage[] | null = () => null) {}

  /**
   * INPUT MESSAGE -> REWARD WEIGHTS -> STUN (set the timeout IN the
   * neuron) -> WAITING. The handler proper: it never sends — it
   * rewards the node and re-arms its relax time. THE ERROR WAVE's
   * size law: an ERROR message stuns for a delay PROPORTIONAL to the
   * size of the information hitting the tensor (the feedback is a
   * wave with a delay scaled by the error, exactly like the forward
   * flow).
   */
  receive(message: TensorMessage, now: number): void {
    this.reward = this.reward * 0.7 + message.weight * 0.3;
    this.staged = message;
    if (message.kind === "error") {
      const size = Math.min(1.0, Math.abs(message.weight));
      this.armedDelay = STUN_MIN + (STUN_MAX - STUN_MIN) * size;
    } else {
      this.armedDelay = stunTimeout(this.reward);
    }
    this.relaxAt = now + this.armedDelay;
  }

  /** THE RELAXATION: the only emitter. The law makes the messages;
   *  the dispatcher delivers them. */
  fireNow(): void {
    const staged = this.staged!;
    const wasError = staged.kind === "error";
    this.staged = null;
    this.relaxAt = Infinity;
    this.fires += 1;
    const messages = wasError ? this.errorLaw(this, staged)
                              : this.law(this, staged);
    if (messages === null) {
      return;
    }
    for (const message of messages) {
      this.network.deliver(message);
    }
    // (C) THE LONG-STUN RELAX: a tensor that sat stunned a long while
    // — hit hard by the error wave, or starved of reward — spreads
    // its own slice when it finally relaxes (with a cooldown, so a
    // repeatedly-hit tensor still pays once per few windows).
    if (this.armedDelay >= LONG_STUN_BAR
        && this.network.now - this.lastConsolidate >= CONSOLIDATE_COOLDOWN) {
      this.lastConsolidate = this.network.now;
      this.network.consolidateAround(this.id);
    }
  }

  private network!: TensorNetwork;
  /** The network wires itself in after construction. */
  bind(network: TensorNetwork): void {
    this.network = network;
  }

  /** A fresh flight over the same brain: the node re-arms at birth
   *  (a brand new network's state — reward, stun, clocks). */
  reset(): void {
    this.reward = 0.1;
    this.staged = null;
    this.relaxAt = Infinity;
    this.armedDelay = 0.0;
    this.fires = 0;
    this.lastConsolidate = -Infinity;
  }
}/** The ISRLU-scaled weight a fired message carries. */
function firedWeight(node: TensorNode): number {
  return node.reward;
}

// --- THE UI's STATE VIEW (the AI state log's columns) ----------------------

/** One dominant classification: its label and its % belonging (the
 *  share of the top-5 recall mass; 0 when nothing is learned yet). */
export interface TensorDominant {
  readonly label: string;
  readonly share: number;
}

/** One sensor tensor's current column: the class it fired, its
 *  confidence share, and its 5 dominant classifications. */
export interface TensorSensorView {
  readonly name: string;
  /** The class the tensor fired last (its situation signal). */
  readonly className: string;
  /** The fired class's share of the top-5 recall mass (0 = the
   *  tensor has learned nothing here: no % shown). */
  readonly share: number;
  /** The 5 most dominant classifications with their % belonging. */
  readonly dominant: readonly TensorDominant[];
}

/** The servo tensor's current column: the fired command, the analogue
 *  lever values, and its 5 dominant options with their %. */
export interface TensorServoView {
  /** The fired command, in pilot's words ("elevator +12.5%"). */
  readonly command: string;
  /** The servo's 5 dominant options with their % belonging. */
  readonly dominant: readonly TensorDominant[];
}

/** The whole network's state, as the UI's AI state log reads it — one
 *  decision window's snapshot, cheap to compute, never the tensors
 *  themselves. */
export interface TensorStateView {
  /** How many decision windows the flight has flown (the log's tick). */
  readonly decisionCount: number;
  readonly goalId: number;
  readonly elementId: number;
  /** The situation bus state (the goal tensor's merged input). */
  readonly situation: number;
  /** Whether the servo fired this window. */
  readonly acted: boolean;
  /** The four sensors' columns: location, energy, flight, controls. */
  readonly sensors: readonly TensorSensorView[];
  /** The servo's column. */
  readonly servo: TensorServoView;
}

/** The lever names, channel order (aileron, elevator, rudder, throttle). */
const CHANNEL_NAMES: readonly string[] =
  ["aileron", "elevator", "rudder", "throttle"];

/** A servo option, in pilot's words: "hold-all" or "elevator +12.5%". */
function labelServoOption(option: number): string {
  if (option === 0) {
    return "hold-all";
  }
  const [channel, level] = ServoPolicy.channelLevelOf(option);
  const [low, high] = SURFACE_LIMITS[channel];
  const value = levelValue(channel, level);
  const percent = Math.round(100.0 * (value - low) / (high - low));
  return `${CHANNEL_NAMES[channel]} ${percent > 50 ? "+" : ""}${percent}%`;
}

/** The top-5 cells of an axis by recall mass (baseline-subtracted), with
 *  their % belonging. Empty when nothing is learned. */
function dominantOf(masses: Array<{ label: string; mass: number }>):
    readonly TensorDominant[] {
  const known = masses.filter((m) => m.mass > 1e-6);
  if (known.length === 0) {
    return [];
  }
  known.sort((a, b) => b.mass - a.mass);
  const top = known.slice(0, 5);
  const total = top.reduce((sum, m) => sum + m.mass, 0.0);
  return top.map((m) => ({
    label: m.label,
    share: total > 0.0 ? m.mass / total : 0.0,
  }));
}

/**
 * THE TENSOR NETWORK — one per flight, its own clock. The flight
 * steps it once per decision window: the panel pulse goes in, then
 * the dispatcher SWEEPS until the cascade settles. The stun, the
 * relaxation, and the gossip all happen on this clock.
 */
export class TensorNetwork {
  /** THE DISPATCHER's CLOCK (decision units). */
  now = 0.0;
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
   *  whose output is what the tensor already said makes NO message
   *  (only news travels; the cascade settles instead of ping-ponging
   *  itself past the horizon). */
  private sensorLastGossip:
      Array<{ panel: readonly number[]; code: number } | null> =
    [null, null, null, null];
  private goalLastGoal = -1;
  private goalLastElement = -1;
  private skillCoherenceLast: readonly number[] | null = null;
  /** The sensors' recurrent keys from the last classification — where
   *  the error law's pump lands (on-policy: the exact cells the
   *  student's own flight will recall). */
  private lastSensorKeys:
      Array<readonly number[]> = [[], [], [], []];
  /** The epoch the current teach() pumps fire at. */
  private teachEpoch = 0;
  /** How many decision windows this flight has flown (the UI log's
   *  tick: a new row lands when it advances). */
  private decisionCount = 0;
  /** The cascade's settle horizon, in decision units. The servo sits
   *  TWO stuns downstream (the skill must relax and forward before
   *  the servo can even be stunned), so the horizon must cover the
   *  whole chain: panel -> sensors -> goal/skill -> skill's forward
   *  -> servo's fire. */
  private static readonly SETTLE = 2.5;
  /** The error wave's own settle horizon — it must cover the whole
   *  upstream chain (servo -> skill -> goal -> sensors) at each hop's
   *  size-scaled stun. */
  private static readonly ERROR_SETTLE = 2.5;

  constructor(policy: ServoPolicy, startElementId: number,
              seedRng: NumpyPcg64) {
    this.policy = policy;
    this.elementId = startElementId;
    this.context = new TensorContext(startElementId);
    // Every tensor's own dice, derived deterministically from the
    // flight's seed — the stochastic noise of the (c) consolidations
    // and the servo's exploration.
    const rngs: NumpyPcg64[] = [];
    for (let id = 0; id < 7; id++) {
      rngs.push(new NumpyPcg64(seedRng.draw64(), seedRng.draw64()));
    }
    this.rngs = rngs;
    this.nodes = [
      new TensorNode(TENSOR_NODE.location,
        (node, staged) => this.sensorLaw(node, staged, 0, LOC_CLASS_AXIS),
        (node, staged) => this.sensorErrorLaw(node, staged, 0, LOC_CLASS_AXIS)),
      new TensorNode(TENSOR_NODE.energy,
        (node, staged) => this.sensorLaw(node, staged, 1, ENERGY_CLASS_AXIS),
        (node, staged) => this.sensorErrorLaw(node, staged, 1, ENERGY_CLASS_AXIS)),
      new TensorNode(TENSOR_NODE.flight,
        (node, staged) => this.sensorLaw(node, staged, 2, FLIGHT_CLASS_AXIS),
        (node, staged) => this.sensorErrorLaw(node, staged, 2, FLIGHT_CLASS_AXIS)),
      new TensorNode(TENSOR_NODE.controls,
        (node, staged) => this.sensorLaw(node, staged, 3, CONTROLS_CLASS_AXIS),
        (node, staged) => this.sensorErrorLaw(node, staged, 3, CONTROLS_CLASS_AXIS)),
      new TensorNode(TENSOR_NODE.goal,
        (node, staged) => this.goalLaw(),
        (node, staged) => this.goalErrorLaw(node, staged)),
      new TensorNode(TENSOR_NODE.skill,
        (node, staged) => this.skillLaw(),
        (node, staged) => this.skillErrorLaw(node, staged)),
      new TensorNode(TENSOR_NODE.servo,
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
    this.nodes[message.recipientId].receive(message, this.now);
  }

  /**
   * THE SWEEP — the dispatcher itself. Find the earliest relaxation
   * among the stunned nodes, advance the clock to it, fire every
   * node whose stun relaxed (in node-id order), and repeat until a
   * full sweep finds nothing before the horizon. No heap, no
   * cancellation: seven plain fields scanned per step.
   */
  private pump(horizon: number): void {
    for (;;) {
      let earliest = Infinity;
      for (const node of this.nodes) {
        if (node.staged !== null && node.relaxAt < earliest) {
          earliest = node.relaxAt;
        }
      }
      if (earliest === Infinity || earliest > horizon) {
        return;
      }
      if (earliest > this.now) {
        this.now = earliest;
      }
      for (const node of this.nodes) {
        if (node.staged !== null && node.relaxAt <= this.now) {
          node.fireNow();
        }
      }
    }
  }

  // --- THE FIRING LAWS (pure recall — the signal processing lives in
  //     the three consolidations, never here) ---------------------------

  /**
   * A SENSOR'S LAW (all four, always): recall the CLASS projection
   * under the recurrent key — the CONTEXT'S CURRENT PANEL (the
   * flight's pulse — the staged evidence is only the stun trigger,
   * never the law's data), its OWN previous class, the goal's
   *  element, its skill patch bin — argmax over the projection, the
   *  CODE message (the chosen class, or the TEACHER's label when
   *  nothing clears the bar) to the goal and the skill, direct. The
   *  class lands in the context — the skill's forward and the bus
   *  read it.
   */
  private sensorLaw(node: TensorNode, staged: TensorMessage,
                    sensorIndex: number, width: number):
      TensorMessage[] | null {
    if (staged.kind !== "panel") {
      return null;   // not the panel's news: the class stands
    }
    const panel = this.context.panels[sensorIndex];
    const context = this.context;
    const key = ServoPolicy.sensorKey(panel, context, sensorIndex);
    this.lastSensorKeys[sensorIndex] = key;
    const tensor = this.policy.sensorTensorOf(sensorIndex);
    // The argmax over the class axis, one reusable key buffer.
    const buf = [...key, 0];
    let best = -1;
    let bestValue = CODE_BAR;
    for (let code = 0; code < width; code++) {
      buf[buf.length - 1] = code;
      const value = tensor.recall(buf);
      if (value > bestValue) {
        bestValue = value;
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
        [code], weight, this.now));
  }

  /**
   * THE GOAL'S LAW — the fuzzy state machine (always): recall the
   * transition evidence under (goal, element, bus) — the FURTHEST
   * candidate whose OWN cells clear the bar, monotone (the choice
   * never regresses the chain) — then the ELEMENT message (the
   * goal/element pair) to the sensors (the recurrence) and the skill.
   * The servo no longer hears the goal directly — the skill forwards
   * the pair with everything else.
   */
  private goalLaw(): TensorMessage[] | null {
    const bus = this.busOf(this.context.signals);
    const candidates = this.chains.get(this.goalId);
    if (candidates !== undefined && candidates.length > 0) {
      const tensor = this.policy.goalTensor;
      const buf = [this.goalId, this.elementId, bus, this.goalId, 0];
      let chosen = -1;
      for (let i = 0; i < candidates.length; i++) {
        buf[buf.length - 1] = candidates[i];
        if (tensor.recall(buf) > ELEMENT_FIRE_BAR) {
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
    // THE NO-NEWS LAW: the same goal/element — nothing to say (the
    // transition itself landed in the network's element already).
    if (this.goalId === this.goalLastGoal
        && this.elementId === this.goalLastElement) {
      return null;
    }
    this.goalLastGoal = this.goalId;
    this.goalLastElement = this.elementId;
    this.context.element = this.elementId;
    const weight = firedWeight(this.nodes[TENSOR_NODE.goal]);
    return [TENSOR_NODE.location, TENSOR_NODE.energy, TENSOR_NODE.flight,
            TENSOR_NODE.controls, TENSOR_NODE.skill]
      .map((recipient) => new TensorMessage(TENSOR_NODE.goal, recipient,
        "element", [this.goalId, this.elementId], weight, this.now));
  }

  /**
   * THE SKILL'S LAW — the patch panel (always): recall the wires'
   * verdict on the current company -> the COHERENCE message to the
   *  sensors (their patch bins) and the goal, and THE FORWARD message
   *  to the servo: ALL the classifications from every tensor — the
   *  goal/element pair and the two classes that drive a correction —
   *  plus the wires' own coherence bin. The forward IS the servo's
   *  whole input. Initially this is pure forwarding; what the
   *  training mixes in is the wires' verdicts (the patch bins already
   *  in the sensors' keys, and the coherence bin riding the payload).
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
    // THE NO-NEWS LAW on the coherence channel: the wires' verdict
    // only when it moved. The FORWARD channel always speaks — a
    // steady input is a steady command (the stick springs to trim
    // between commands).
    const coherence = [context.coherenceBin, ...context.skillBins];
    if (this.skillCoherenceLast === null
        || !coherence.every((v, i) => v === this.skillCoherenceLast![i])) {
      this.skillCoherenceLast = coherence;
      for (const recipient of [TENSOR_NODE.location, TENSOR_NODE.energy,
                               TENSOR_NODE.flight, TENSOR_NODE.controls,
                               TENSOR_NODE.goal]) {
        messages.push(new TensorMessage(TENSOR_NODE.skill, recipient,
          "coherence", coherence, weight, this.now));
      }
    }
    // THE FORWARD — the servo's whole input: bit-identical to the
    // demonstration's servo stamp key.
    messages.push(new TensorMessage(TENSOR_NODE.skill, TENSOR_NODE.servo,
      "forward", [this.goalId, this.elementId, context.energy,
                  context.flight, context.coherenceBin], weight,
      this.now));
    return messages;
  }

  /**
   * THE SERVO'S LAW (always): the STAGED FORWARD message — the skill
   *  tensor's forwarded classifications — IS the observation key (the
   *  servo's whole input, never the Aircraft, never even the other
   *  tensors directly). Argmax over the option cells under it (pure
   *  recall), epsilon-greedy — the COMMAND message (the chosen
   *  levers) to the skill (its patch signal). The flight reads the
   *  fired command itself.
   */
  private servoLaw(node: TensorNode,
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
      "command", [option], firedWeight(node), this.now)];
  }

  // --- THE ERROR WAVE (upstream) --------------------------------------------

  /** The error payload's expected classes: the teacher on the
   *  student's own panels. */
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
   * on the same clock as the forward flow — servo -> skill ->
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
    this.now += 0.05;   // the wave rides after the forward cascade
    const entry = worstChannel >= 0 ? TENSOR_NODE.servo : TENSOR_NODE.skill;
    this.nodes[entry].receive(new TensorMessage(FLIGHT_SENDER_ID, entry,
      "error", payload, -Math.min(1.0, size), this.now), this.now);
    this.pump(this.now + TensorNetwork.ERROR_SETTLE);
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
      "error", payload, -Math.min(1.0, error), this.now)];
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
        "error", payload, weight, this.now));
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
        "error", staged.payload, staged.weight, this.now));
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

  // --- (C) THE LONG-STUN RELAX's SLICE SPREAD --------------------------------

  /**
   * (C) A tensor that relaxes from a LONG stun spreads its own slice:
   *  the signal-processing pass over the output window AROUND WHERE
   *  IT STANDS — its current key prefix, its current class or
   *  option — one denoise call, its own dice. The starved tensor,
   *  when it finally relaxes, smooths what it believes around where
   *  it is looking.
   */
  consolidateAround(nodeId: TensorNodeId): void {
    const rng = this.rngs[nodeId];
    const epoch = this.teachEpoch;
    const context = this.context;
    const classTail = (tail: number): number[] => [tail];
    const optionTail = (tail: number): number[] =>
      tail === 0 ? [HOLD_ALL_CHANNEL, 0]
                 : ServoPolicy.channelLevelOf(tail);
    if (nodeId <= TENSOR_NODE.controls) {
      const i = nodeId;
      ServoPolicy.spreadWindow(this.policy.sensorTensorOf(i),
        this.lastSensorKeys[i], [context.signalOf(i)],
        [LOC_CLASS_AXIS, ENERGY_CLASS_AXIS, FLIGHT_CLASS_AXIS,
         CONTROLS_CLASS_AXIS][i], classTail, rng, epoch);
      return;
    }
    if (nodeId === TENSOR_NODE.goal) {
      ServoPolicy.spreadWindow(this.policy.goalTensor,
        [this.goalId, this.elementId, this.busOf(context.signals),
         this.goalId], [this.elementId], ELEMENT_AXIS, classTail,
        rng, epoch);
      return;
    }
    if (nodeId === TENSOR_NODE.skill) {
      const sensors: Array<keyof TensorSignals> = ["location", "energy",
                                                   "flight", "controls"];
      for (let s = 0; s < 4; s++) {
        ServoPolicy.spreadWindow(this.policy.skillTensor,
          [wireIndexOf(sensors[s], "element"), context.signalOf(s)],
          [context.element], PATCH_SLOT_AXIS, classTail, rng, epoch,
          ELEMENT_NOISE_SIGMA);   // the wires explore stochastically
      }
      return;
    }
    // THE SERVO: the option window around the chosen command.
    const option = ServoPolicy.optionOfActions(
      this.firedActions ?? new Int32Array(4).fill(ACTION_HOLD));
    ServoPolicy.spreadWindow(this.policy.servoTensor, this.firedKey,
      [option], ServoPolicy.OPTIONS, optionTail, rng, epoch);
  }

  // --- THE FLIGHT'S PUMP ----------------------------------------------------

  /**
   * One decision window: the panel pulse goes in — four PANEL
   * messages, the only input from the Aircraft — then the dispatcher
   * sweeps until the cascade settles (the stuns relax, the messages
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
    this.now += 1.0;   // one decision window of network time
    this.decisionCount += 1;
    const panels: readonly number[][] = [sense.location, sense.energy,
                                          sense.flight, sense.controls];
    for (let s = 0; s < 4; s++) {
      this.nodes[s].receive(new TensorMessage(FLIGHT_SENDER_ID,
        s as TensorNodeId, "panel", panels[s], 1.0, this.now), this.now);
    }
    // The standing invitations: the goal and the skill have news to
    // consider each decision (their handlers reward and stun; their
    // laws fire once relaxed). The servo receives NOTHING here — its
    // only input is the skill's forward, once the skill relaxes.
    this.nodes[TENSOR_NODE.goal].receive(new TensorMessage(
      FLIGHT_SENDER_ID, TENSOR_NODE.goal, "element", [goalId], 1.0,
      this.now), this.now);
    this.nodes[TENSOR_NODE.skill].receive(new TensorMessage(
      FLIGHT_SENDER_ID, TENSOR_NODE.skill, "element", [this.elementId],
      1.0, this.now), this.now);
    // SWEEP: dispatch until the cascade settles.
    this.pump(this.now + TensorNetwork.SETTLE);
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

  /**
   * THE UI's STATE VIEW — one decision window's snapshot for the AI
   *  state log: the goal/element/situation, each sensor's fired class
   *  with its 5 dominant classifications and their % belonging, and the
   *  servo's fired command with its 5 dominant options. Never the
   *  tensors themselves — analogue of what fired, not the memory.
   */
  stateView(): TensorStateView {
    const context = this.context;
    const widths = [LOC_CLASS_AXIS, ENERGY_CLASS_AXIS, FLIGHT_CLASS_AXIS,
                    CONTROLS_CLASS_AXIS];
    const labels = [labelLocationClass, labelEnergyClass, labelFlightClass,
                    labelControlsClass];
    const names = ["LocationTensor", "EnergyTensor", "FlightTensor",
                   "ControlsTensor"];
    const sensors: TensorSensorView[] = [];
    for (let i = 0; i < 4; i++) {
      const tensor = this.policy.sensorTensorOf(i);
      const key = this.lastSensorKeys[i];
      const masses: Array<{ label: string; mass: number }> = [];
      if (key.length > 0) {
        const buf = [...key, 0];
        for (let cls = 0; cls < widths[i]; cls++) {
          buf[buf.length - 1] = cls;
          masses.push({
            label: labels[i](cls),
            mass: Math.max(0.0, tensor.recall(buf) - tensor.baseline),
          });
        }
      }
      const dominant = dominantOf(masses);
      const fired = context.signalOf(i);
      const firedShare = dominant.find((d) =>
        d.label === labels[i](fired))?.share ?? 0.0;
      sensors.push({
        name: names[i],
        className: labels[i](fired),
        share: firedShare,
        dominant,
      });
    }
    // THE SERVO: the fired command and its 5 dominant options.
    const servoMasses: Array<{ label: string; mass: number }> = [];
    if (this.firedKey.length > 0) {
      const key = this.firedKey;
      servoMasses.push({
        label: "hold-all",
        mass: Math.max(0.0, this.policy.servoTensor.recall(
          [...key, HOLD_ALL_CHANNEL, 0]) - this.policy.servoTensor.baseline),
      });
      const buf = [...key, 0, 0];
      for (let channel = 0; channel < 4; channel++) {
        buf[buf.length - 2] = channel;
        for (let level = 0; level < stopCount(channel); level++) {
          buf[buf.length - 1] = level;
          servoMasses.push({
            label: labelServoOption(
              ServoPolicy.optionOf(channel, level)),
            mass: Math.max(0.0, this.policy.servoTensor.recall(buf)
                           - this.policy.servoTensor.baseline),
          });
        }
      }
    }
    const actions = this.firedActions;
    let command = "hold";
    if (actions !== null) {
      command = labelServoOption(ServoPolicy.optionOfActions(actions));
    }
    return {
      decisionCount: this.decisionCount,
      goalId: this.goalId,
      elementId: this.elementId,
      situation: this.busOf(context.signals),
      acted: actions !== null,
      sensors,
      servo: { command, dominant: dominantOf(servoMasses) },
    };
  }

  /**
   * A FRESH FLIGHT OVER THE SAME MEMORY: the runtime resets — the
   *  recurrence starts clean, the nodes' reward EMAs re-arm at birth,
   *  the gossip's no-news memory empties — exactly the state a brand
   *  new TensorNetwork over this same policy would start from. The
   *  MEMORY (the policy's tensors) persists untouched. The AiPilot
   *  calls this as a flight takes the brain up (the encapsulated
   *  brain's own beginFlight).
   */
  beginFlight(startElementId: number): void {
    this.now = 0.0;
    this.elementId = startElementId;
    this.goalId = GOAL_FREE_FLIGHT;
    this.firedActions = null;
    this.firedKey = [];
    this.decisionCount = 0;
    this.goalLastGoal = -1;
    this.goalLastElement = -1;
    this.skillCoherenceLast = null;
    this.sensorLastGossip = [null, null, null, null];
    this.lastSensorKeys = [[], [], [], []];
    const context = this.context;
    context.location = 0;
    context.energy = 0;
    context.flight = 0;
    context.controls = 0;
    context.element = startElementId;
    context.lastOption = 0;
    context.skillBins[0] = 0;
    context.skillBins[1] = 0;
    context.skillBins[2] = 0;
    context.skillBins[3] = 0;
    context.coherence = 0.0;
    context.coherenceBin = 0;
    for (const node of this.nodes) {
      node.reset();
    }
  }

  /** The situation bus state: the four sensors' codes, merged. */
  busOf(signals: TensorSignals): number {
    return (signals.location + signals.energy + signals.flight
            + signals.controls) % SITUATION_AXIS;
  }
}
