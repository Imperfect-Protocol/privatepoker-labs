/**
 * THE PILOTS — who is at the controls of one machine.
 *
 * The machine (physics/aircraft.ts) is SEALED: a pilot never touches
 * its components. The only path to its levers is the machine's
 * control-input method (Aircraft.command, fed once per tick by the
 * flight that owns the machine); a pilot only READS the machine —
 * through the sighting the flight hands it (the instruments' sensed
 * panels and the controls' own state) — and produces the next tick's
 * control inputs. All mode logic lives in the AIRCRAFT CONTROLLERS
 * (learning/aircraft-controller.ts), never in the flights:
 *
 *   InstructorPilot   — the demonstration hand: the phase law's
 *                       targets of the machine's TRUE element, eased
 *                       (the classroom), or the stabilizer (free
 *                       flight);
 *   AiPilot           — THE TENSOR BRAIN, FULLY ENCAPSULATED: every
 *                       tensor lives inside it (the memory, the
 *                       pulsating network, the error neurons), and
 *                       NOTHING outside ever sees a tensor. It reads
 *                       the machine's signals (the sighting's sensed
 *                       panels), decides on its own cadence, and
 *                       operates the controls through the single
 *                       method call — whatever tensors it holds
 *                       inside can change without anything outside
 *                       changing with them.
 *
 * The controllers compose the pilots: a SoloController flies one
 * pilot; a PairController flies the AI beside the instructor (the
 * AI suggesting and learning while the instructor flies, or the AI
 * flying under the guardian). The flights hold a controller, the
 * controller holds the pilots, and the machine knows its controller
 * (Aircraft.attachController).
 */
import { RigidBody } from "../physics/rigid-body.js";
import {
  stabilizeTargets, isDangerous, RECOVERY_HOLD_S,
} from "./guardian.js";
import { levelOf } from "./instruments.js";
import {
  ServoPolicy, TensorNetwork, TensorContext, ELEMENT_FREE_FLIGHT,
  type NetworkDecision, type TensorStateView, type BrainSnapshot,
} from "./servo-tensor.js";
import type { PulseNeuron } from "./neurons.js";
import type { NumpyPcg64 } from "../core/rng.js";
import type { PhaseCursor } from "./school.js";
import type { SenseReading } from "./instruments.js";

/** The recovery window: decision windows to stay safe before the
 *  guarded pair hands the levers back (both hosts decide at 5 Hz). */
const RECOVERY_WINDOWS = Math.round(RECOVERY_HOLD_S * 5.0);

/** The AI's own cadence when a caller does not say: 5 Hz, the frozen
 *  protocol's decision clock (school.ts's DECISION_EVERY is 10 ticks
 *  at 50 Hz — the same number; pilot.ts cannot import the school). */
const DEFAULT_DECISION_EVERY = 10;

/**
 * The instructor's hand: eases commanded deflections toward the raw
 * phase targets at bounded rates. All inputs on the flight controls
 * are smooth — the yoke never jumps, even when a phase changes the
 * law underneath.
 *
 * Per-channel easing rates: the instructor demonstrates at the
 * STUDENT'S hand speed — one ACTION_DELTA per decision tick window
 * (0.05 rad per 0.2 s = 0.25/s). A faster hand carves a pull the
 * student's ±0.05/decision vocabulary cannot track: the demo's
 * stamps land on a path through (reading, yoke) space the student
 * physically cannot follow, and every replay diverges from the
 * lesson it was shown. The instructor shows the maneuver the way a
 * student's hands can fly it.
 */
export class InstructorHand {
  readonly commands: Float64Array;
  /** The DEMO hand moves at the student's own speed; the RECOVERY
   *  hand (the swarm's stabilizer) moves four times firmer — taking
   *  over a dive at demo speed is arriving late with good manners. */
  private static readonly DEMO_RATES = [0.25, 0.25, 0.25, 0.25];
  private static readonly FIRM_RATES = [1.0, 1.0, 1.0, 1.0];
  private readonly rates: readonly number[];

  constructor(commands: ArrayLike<number>, rates?: readonly number[]) {
    this.commands = Float64Array.from(commands);
    this.rates = rates ?? InstructorHand.DEMO_RATES;
  }

  /** The swarm's recovery hand. */
  static firm(): readonly number[] {
    return InstructorHand.FIRM_RATES;
  }

  /** One tick of easing toward the raw targets. */
  easeToward(targets: ArrayLike<number>, dt: number): Float64Array {
    for (let channel = 0; channel < 4; channel++) {
      const rate = this.rates[channel] * dt;
      const delta = Math.max(-rate,
        Math.min(rate, targets[channel] - this.commands[channel]));
      this.commands[channel] += delta;
    }
    return this.commands;
  }
}

/** One pilot's read of the machine it flies, this tick. Everything a
 *  pilot may know: the machine's sealed state, its TRUE element
 *  cursor, the AI's sensed panels, the teaching observation beside
 *  it, and the classroom clocks. */
export interface PilotSighting {
  readonly dt: number;
  readonly tick: number;
  readonly body: RigidBody;
  /** The terrain's height under the machine, m (the stabilizer's
   *  clearance floor — the ground is the pilot's business). */
  readonly groundM: number;
  /** The machine's TRUE element cursor — the classroom's phase chain.
   *  Null in free flight: the swarm has no lesson. */
  readonly cursor: PhaseCursor | null;
  /** The AI's sensed panels this tick (the instruments' and the
   *  controls' own state, binned — the brain's only input). */
  readonly sense: SenseReading;
  /** The instructor's observation beside the machine — the error
   *  wave's teaching source; null when nobody teaches. */
  readonly teaching: { elementId: number; actions: Int32Array } | null;
  /** The classroom epoch (the teach pulses' clock). */
  readonly epoch: number;
  /** The goal being realized (the lesson's, or free flight's own). */
  readonly goalId: number;
  /** The maneuver's taught elements (the transition candidates). */
  readonly candidates: readonly number[] | null;
}

/** A pilot: the producer of one machine's control inputs. */
export interface Pilot {
  /** The control inputs for this tick — the ONLY thing that reaches
   *  the machine's levers (the flight feeds them to the machine's
   *  own input method). */
  step(sighting: PilotSighting, commands: Float64Array): Float64Array;
  /** The hands-off reflex after the physics tick: the instructor's
   *  hand holds its positions; the AI's stick springs. */
  relax(commands: Float64Array, dt: number): Float64Array;
  /** The instructor's hand is physically on the levers right now. */
  readonly heldByInstructor: boolean;
}

/**
 * THE INSTRUCTOR — the demonstration hand. Flies the machine's TRUE
 * element: the phase law of the element the machine is actually in
 * (the classroom — never a speculative intent), eased at the
 * student's hand speed; in free flight, the stabilizer holds the
 * captured altitude, wings level, and steers home before the world's
 * edge.
 */
export class InstructorPilot implements Pilot {
  private hand: InstructorHand;
  private readonly freeFlight: boolean;
  private capturedAltM = 0.0;

  constructor(startCommands: ArrayLike<number>, freeFlight: boolean) {
    this.hand = new InstructorHand(startCommands,
      freeFlight ? InstructorHand.firm() : undefined);
    this.freeFlight = freeFlight;
  }

  /** Takeover-ready: the hand re-seats from the machine's CURRENT
   *  lever targets, and the free-flight stabilizer captures the
   *  altitude where the machine is now. */
  reseat(commands: ArrayLike<number>, body: RigidBody): void {
    this.hand = new InstructorHand(commands,
      this.freeFlight ? InstructorHand.firm() : undefined);
    if (this.freeFlight) {
      this.capturedAltM = body.position.z;
    }
  }

  /** The instructor's raw targets this tick. */
  private targets(body: RigidBody, cursor: PhaseCursor | null,
                  groundM: number): Float64Array {
    return cursor !== null
      ? cursor.phaseTargets(body)
      : stabilizeTargets(
          Math.max(this.capturedAltM, groundM + InstructorPilot.CLEARANCE_M),
          body);
  }

  /** The free-flight clearance: the stabilizer never chases its
   *  captured altitude below this many metres over the terrain. */
  private static readonly CLEARANCE_M = 30.0;   // ~100 ft AGL

  step(sighting: PilotSighting, _commands: Float64Array): Float64Array {
    return Float64Array.from(
      this.hand.easeToward(
        this.targets(sighting.body, sighting.cursor, sighting.groundM),
        sighting.dt));
  }

  relax(commands: Float64Array, _dt: number): Float64Array {
    return commands;              // the hand holds its own positions
  }

  get heldByInstructor(): boolean {
    return true;
  }

  /** The swarm's teaching observation: the stabilizer's own targets
   *  as lever levels (the error wave's source in free flight). Null
   *  in the classroom — the twin observes there. */
  stabilizerTeaching(body: RigidBody, groundM: number): {
      elementId: number; actions: Int32Array } | null {
    if (!this.freeFlight) {
      return null;
    }
    const targets = stabilizeTargets(
      Math.max(this.capturedAltM, groundM + InstructorPilot.CLEARANCE_M),
      body);
    return {
      elementId: ELEMENT_FREE_FLIGHT,
      actions: new Int32Array([levelOf(0, targets[0]),
        levelOf(1, targets[1]), levelOf(2, targets[2]),
        levelOf(3, targets[3])]),
    };
  }
}

// ===========================================================================
// THE AI PILOT — THE ENCAPSULATED TENSOR BRAIN
// ===========================================================================

/**
 * THE BRAIN'S TOKEN — an opaque snapshot of everything the AI has
 * learned. Outside code passes it around (the boot trainer's
 * transplant, the classroom's graduate, the swarm's newborn) but never
 * looks inside: whatever tensors the brain holds, the token is what
 * travels. When the tensors inside change, this alias changes with
 * them — silently.
 */
export type AiBrainState = BrainSnapshot;

/** What an AiPilot is constructed from: its dice, its decision
 *  cadence, its hands-off reflex, and its starting intention. */
export interface AiPilotOptions {
  /** The AI's own dice — the network's per-tensor streams and the
   *  exploration draw all come from here, deterministically. */
  readonly seedRng: NumpyPcg64;
  /** Decision windows per decision (10 ticks at 50 Hz = the frozen
   *  5 Hz protocol). */
  readonly decisionEvery?: number;
  /** The hands-off reflex between windows (the stick springs toward
   *  neutral). */
  readonly spring: (commands: Float64Array, dt: number) => Float64Array;
  /** The intention the brain starts holding. */
  readonly startElementId?: number;
  /** The exploration rate (defaults to the policy's own). */
  readonly epsilon?: number;
}

/** The AI's per-tensor live key counts — the boot dialog's cube fills.
 *  A DIAGNOSTIC view, not the tensors: names follow the brain's
 *  layout and change with it, inside this file. */
export interface AiBrainCounts {
  readonly location: number;
  readonly energy: number;
  readonly flight: number;
  readonly controls: number;
  readonly skill: number;
  readonly goal: number;
  readonly servo: number;
}

/** The last decision's summary — what the outside world may know of
 *  what the AI decided: the element it claims, and whether it acted.
 *  Never the observation key, never the cells. */
export interface AiIntent {
  readonly elementId: number;
  readonly acted: boolean;
}

/**
 * THE AI — THE TENSOR BRAIN AT THE CONTROLS, FULLY ENCAPSULATED.
 *
 * Every tensor lives INSIDE: the memory (the policy), the pulsating
 * network, the error neurons — nothing outside ever sees one. The
 * pilot reads the machine's signals (the sighting's sensed panels —
 * the instruments' and the controls' own state), decides one window
 * at a time, and operates the controls through the single method
 * call (the commands it returns reach the machine's own input
 * method). The outside world knows four things about it:
 *
 *  - IT FLIES: step / relax (the Pilot contract), decide / applyTo
 *    (the pair controllers' suggestion path);
 *  - IT LEARNS: learnFromInstructor (the error wave), observeDemo /
 *    observeTransition (the classroom's demonstration stamps),
 *    credit / settle (its own decisions' reinforcement);
 *  - ITS BRAIN TRAVELS AS A TOKEN: inherit / extract / weaveWith /
 *    blendFrom (and the lifecycle: installPriors, consolidate,
 *    prune, setExploration);
 *  - IT REPORTS: stateView (the AI state log), choices (the panel's
 *    AERT strip), lastIntent (the claimed element), counts (the boot
 *    dialog's cubes).
 *
 * Because the tensors are inside, WHATEVER tensors the brain should
 * hold can be swapped in — one file changes, nothing outside does.
 */
export class AiPilot implements Pilot {
  /** THE MEMORY AND THE NETWORK — private: the tensors are this
   *  class's own business, and nobody else's. */
  private readonly policy = new ServoPolicy();
  private readonly network: TensorNetwork;
  /** The demonstration's own recurrent context — the demo stamps
   *  observe with (its previous codes, its skill bins), separate
   *  from the flight runtime below. */
  private readonly demoContext = new TensorContext(ELEMENT_FREE_FLIGHT);
  private readonly rng: NumpyPcg64;
  private readonly decisionEvery: number;
  private readonly spring: (commands: Float64Array,
                             dt: number) => Float64Array;
  private lastDecision: NetworkDecision | null = null;
  private lastChoices: Int32Array | null = null;
  /** Whether the AI's own hands are on the levers — the pair
   *  controllers tell it (the instructor's hands mean the AI's
   *  suggestion earns no flight credit). */
  private handsOn = true;
  /** The AI's own recent decisions, awaiting their flight's reward. */
  private readonly recent: Array<[readonly number[], Int32Array]> = [];

  constructor(options: AiPilotOptions) {
    this.rng = options.seedRng;
    this.decisionEvery = options.decisionEvery ?? DEFAULT_DECISION_EVERY;
    this.spring = options.spring;
    this.network = new TensorNetwork(this.policy,
      options.startElementId ?? ELEMENT_FREE_FLIGHT, this.rng);
    if (options.epsilon !== undefined) {
      this.policy.epsilon = options.epsilon;
    }
  }

  /** A fresh brain jar — the boot graduate, the classroom's
   *  beginner, the singularity's vessel: constructed, never flown by
   *  its own step, inherited from and blended into. */
  static fresh(seedRng: NumpyPcg64): AiPilot {
    return new AiPilot({ seedRng, spring: (commands) => commands });
  }

  // --- FLYING (the Pilot contract) ---------------------------------------

  /** Whether the decision window opens this tick. */
  windowOpen(tick: number): boolean {
    return tick % this.decisionEvery === 0;
  }

  /**
   * ONE DECISION WINDOW: sense, decide — the suggestion. The AI
   * always decides when the window opens (its own learning and the
   * state log live on it, whoever's hands fly); when its own hands
   * are on the levers the decision joins its reinforcement's recent
   * window. Between windows a no-op.
   */
  decide(sighting: PilotSighting): void {
    if (!this.windowOpen(sighting.tick)) {
      return;
    }
    this.lastDecision = this.network.decide(sighting.goalId,
      sighting.sense, sighting.candidates, this.rng);
    this.lastChoices = this.lastDecision.actions;
    if (this.lastDecision.acted && this.handsOn) {
      this.recent.push([this.lastDecision.key, this.lastDecision.actions]);
      if (this.recent.length > 10) {
        this.recent.shift();
      }
    }
  }

  /** The AI's hands: drive the levers toward the decision's stops. */
  applyTo(commands: Float64Array): Float64Array {
    const decision = this.lastDecision;
    if (decision !== null && decision.acted) {
      return this.policy.commandFrom(commands, decision.actions);
    }
    return commands;
  }

  step(sighting: PilotSighting, commands: Float64Array): Float64Array {
    // One decision, one lever step per window — never per tick.
    if (!this.windowOpen(sighting.tick)) {
      return commands;
    }
    this.decide(sighting);
    return this.applyTo(commands);
  }

  relax(commands: Float64Array, dt: number): Float64Array {
    return this.spring(commands, dt);
  }

  get heldByInstructor(): boolean {
    return false;
  }

  // --- LEARNING ------------------------------------------------------------

  /** The guided modes' learning: the instructor's element and levers
   *  against the AI's own decision — the error wave, every window. */
  learnFromInstructor(sighting: PilotSighting): void {
    if (!this.windowOpen(sighting.tick) || sighting.teaching === null) {
      return;
    }
    this.network.teach(sighting.teaching.elementId,
                       sighting.teaching.actions, sighting.epoch);
  }

  /** Credit the AI's own recent decisions with the flight's earned
   *  reward (decaying eligibility). A no-op with no own decisions. */
  credit(reward: number, epoch: number): void {
    if (this.recent.length === 0) {
      return;
    }
    this.policy.reinforce(this.recent, reward, epoch);
  }

  /** The flight's epilogue: the final reinforcement of its own recent
   *  decisions, once. A crash or a stall teaches −2; reaching the
   *  horizon still flying teaches +0.5; anything else ends neutral. */
  settle(outcome: string | null, epoch: number): void {
    if (outcome === "crashed" || outcome === "stalled") {
      this.policy.reinforce(this.recent, -2.0, epoch);
    } else if (outcome === null) {
      this.policy.reinforce(this.recent, 0.5, epoch);
    }
  }

  /** THE CLASSROOM'S DEMONSTRATION STAMP: the instructor's lesson,
   *  watched and remembered — the sensors' classes under the
   *  demonstration's own recurrence, the goal's transition, the
   *  decided levers. Returns the situation bus state it landed at. */
  observeDemo(goalId: number, elementId: number, sense: SenseReading,
              actions: ArrayLike<number>, epoch: number,
              amount = 1.0): number {
    return this.policy.imprint(goalId, elementId, sense, actions,
                               this.demoContext, amount, epoch);
  }

  /** The goal tensor's TRANSITION stamp: the demonstration moved from
   *  (element, situation) to the next element. */
  observeTransition(goalId: number, fromElementId: number, situation: number,
                    toGoalId: number, toElementId: number, epoch: number,
                    amount = 1.0): void {
    this.policy.stampTransition(goalId, fromElementId, situation,
                                toGoalId, toElementId, amount, epoch);
  }

  // --- THE BRAIN'S LIFECYCLE (opaque tokens in, opaque tokens out) ---------

  /** A fresh flight over the same memory: the runtime resets (the
   *  recurrence starts clean), the memory persists untouched. The
   *  flights call this as they take the brain up. */
  beginFlight(startElementId: number): void {
    this.network.beginFlight(startElementId);
    this.lastDecision = null;
    this.lastChoices = null;
    this.recent.length = 0;
  }

  /** The exploration rate (epsilon-greedy over its own options). */
  get exploration(): number {
    return this.policy.epsilon;
  }

  setExploration(rate: number): void {
    this.policy.epsilon = rate;
  }

  /** Overwrite the brain with a token (the boot transplant, the
   *  classroom's graduate, a parent's weave). */
  inherit(state: AiBrainState): void {
    this.policy.restore(state);
  }

  /** Copy the whole brain out as a token. */
  extract(): AiBrainState {
    return this.policy.snapshot();
  }

  /**
   * GENETIC CROSS-OVER: a child's brain woven from this brain and
   *  another's — each memory cell from either parent, the dice
   *  deciding, the error-neuron signs inherited the same way. A
   *  clone when both parents are one. The weaving is the brain's own
   *  business; the boot trainer only asks.
   */
  weaveWith(other: AiPilot, rng: NumpyPcg64): AiBrainState {
    const snapA = this.policy.snapshot();
    const snapB = other.policy.snapshot();
    const tensors = ["location", "energy", "flight", "controls", "skill",
                     "goal", "servo"] as const;
    const merged: Record<string, { values: Map<number, Float32Array>;
                                   epochs: Map<number, Int32Array> }> = {};
    for (const tensor of tensors) {
      const a = snapA[tensor], b = snapB[tensor];
      const values = new Map<number, Float32Array>();
      const epochs = new Map<number, Int32Array>();
      for (const [key, layers] of a.values) {
        const fromB = rng.uniform() < 0.5 && b.values.has(key);
        values.set(key, Float32Array.from(
          fromB ? b.values.get(key)! : layers));
        epochs.set(key, Int32Array.from(
          fromB ? b.epochs.get(key)! : a.epochs.get(key)!));
      }
      for (const [key, layers] of b.values) {
        if (!values.has(key) && rng.uniform() < 0.5) {
          values.set(key, Float32Array.from(layers));
          epochs.set(key, Int32Array.from(b.epochs.get(key)!));
        }
      }
      merged[tensor] = { values, epochs };
    }
    return {
      location: merged.location, energy: merged.energy,
      flight: merged.flight, controls: merged.controls,
      skill: merged.skill, goal: merged.goal, servo: merged.servo,
      errorSigns: snapA.errorSigns.map((sign, i) =>
        rng.uniform() < 0.5 ? sign : snapB.errorSigns[i]),
    };
  }

  /**
   * THE SWARM'S MEMORY PRIMITIVE: blend a random subset of a donor's
   * memory into this brain at the given rate — the flock's births
   * and death screams, the singularity's rounds. One law, one
   * implementation, and it lives inside the brain.
   */
  blendFrom(donor: AiPilot, rng: NumpyPcg64, rate: number,
            share: number): void {
    const pairs: Array<[PulseNeuron, PulseNeuron]> = [
      [this.policy.locationTensor, donor.policy.locationTensor],
      [this.policy.energyTensor, donor.policy.energyTensor],
      [this.policy.flightTensor, donor.policy.flightTensor],
      [this.policy.controlsTensor, donor.policy.controlsTensor],
      [this.policy.skillTensor, donor.policy.skillTensor],
      [this.policy.goalTensor, donor.policy.goalTensor],
      [this.policy.servoTensor, donor.policy.servoTensor],
    ];
    for (const [mine, theirs] of pairs) {
      // Insertion order, like the Python dict.
      const keys = [...theirs.values.keys()];
      if (keys.length === 0) {
        continue;
      }
      const count = Math.max(1, Math.trunc(keys.length * share));
      const chosen = rng.choice(keys.length, count);
      for (const index of chosen) {
        AiPilot.mergeKey(mine, theirs, keys[index], rate);
      }
    }
  }

  /** Newest-layer merge of one key, the station law without the
   *  station: the recipient's freshest slot blends toward the donor's
   *  newest layer, two-way, exactly the flock's Python law. */
  private static mergeKey(recipient: PulseNeuron, donor: PulseNeuron,
                          keyIndex: number, rate: number): void {
    const donorEpochs = donor.epochs.get(keyIndex)!;
    let newest = 0;
    for (let layer = 1; layer < donorEpochs.length; layer++) {
      if (donorEpochs[layer] > donorEpochs[newest]) {
        newest = layer;
      }
    }
    const recipientLayers = recipient.values.get(keyIndex);
    if (recipientLayers !== undefined) {
      const donorLayers = donor.values.get(keyIndex)!;
      // float32 semantics exactly like NumPy scalar math:
      // diff is float32; rate * diff is float64; the store rounds to float32.
      const diff = Math.fround(donorLayers[newest] - recipientLayers[0]);
      recipientLayers[0] = Math.fround(recipientLayers[0] + rate * diff);
      recipient.epochs.get(keyIndex)![0] = 0;
      return;
    }
    recipient.values.set(keyIndex, Float32Array.from(donor.values.get(keyIndex)!));
    recipient.epochs.set(keyIndex, Int32Array.from(donorEpochs));
  }

  /** The brain's own prior wiring (the pre-training's sensible
   *  companies), imposed as floors with this brain's own jitter. */
  installPriors(epoch: number, rng: NumpyPcg64): void {
    this.policy.installPriors(epoch, rng);
  }

  /** The post-repetition signal-processing pass (noise → Gaussian →
   *  Haar → threshold → de-Haar) over what the training fired. */
  consolidate(epoch: number, rng: NumpyPcg64): void {
    this.policy.consolidate(epoch, rng);
  }

  /** Forget: memory cells nobody has re-fired within the span die. */
  prune(beforeEpoch: number): number {
    return this.policy.prune(beforeEpoch);
  }

  // --- REPORTING (the UI and the flights' questions) ------------------------

  /** THE AI STATE LOG's subject: the brain's live snapshot. */
  get stateView(): TensorStateView {
    return this.network.stateView();
  }

  /** The AI's last per-channel choices (the panel's AERT strip). */
  get choices(): Int32Array | null {
    return this.lastChoices;
  }

  /** The last decision's summary: the element the AI claims, and
   *  whether it acted. Never the key, never the cells. */
  get lastIntent(): AiIntent | null {
    const decision = this.lastDecision;
    return decision === null ? null
      : { elementId: decision.elementId, acted: decision.acted };
  }

  /** The brain's live key counts, per tensor (the boot dialog's
   *  cubes). A diagnostic view, honestly named for what the brain
   *  holds today. */
  counts(): AiBrainCounts {
    return {
      location: this.policy.locationTensor.storedCombinations,
      energy: this.policy.energyTensor.storedCombinations,
      flight: this.policy.flightTensor.storedCombinations,
      controls: this.policy.controlsTensor.storedCombinations,
      skill: this.policy.skillTensor.storedCombinations,
      goal: this.policy.goalTensor.storedCombinations,
      servo: this.policy.servoTensor.storedCombinations,
    };
  }

  /** The brain's total stored key count. */
  get storedCombinations(): number {
    return this.policy.storedCombinations;
  }

  /** The hands truth — the pair controllers set it (whose hands fly
   *  decides whose decisions earn flight credit). */
  markHands(on: boolean): void {
    this.handsOn = on;
  }
}
