/**
 * One aircraft: the machine and its physical body — SEALED.
 *
 * ONE class for every pilot. The instructor's deterministic hand, the
 * student's AI, and the swarm's drones all fly the same machine with the
 * same instruments and the same controls — what differs is only who
 * commands it. Nobody has a shortcut in:
 *
 *  - THE SINGLE INPUT PATH is command(): the commanded targets become
 *    the machine's inputs and the actuators slew toward them. The AI
 *    sends its inputs through this exact same method call as the
 *    instructor — no private channel, no direct actuator access, no
 *    body-first mutation. The inputs and the actuators are private; the
 *    only writers are command() itself and the physics stepping the
 *    body.
 *
 * The aircraft owns:
 *  - the physical body (the rigid body the physics steps),
 *  - the actuators (rate-limited control surfaces, sealed),
 *  - the inputs (what the hand commanded last, sealed),
 *  - the instruments (ASI, VSI, ALT, heading, attitude, body rates,
 *    throttle — read from the physical truth of the machine).
 *
 * The panel reads the aircraft; the sky draws the aircraft; whoever is
 * at the controls commands it through command(). No pilot has a
 * private machine.
 */
import { RigidBody } from "./rigid-body.js";
import {
  SurfacesState, AILERON, ELEVATOR, RUDDER, THROTTLE,
} from "./surfaces.js";
import { GRAVITY, RHO_SEA_LEVEL } from "./aero.js";

/**
 * The controlling hand's decisions, for the panel: the last per-channel
 * choice (ease / hold / push) and whether a decision window just opened.
 * Null choices while the instructor eases its smooth demonstration path.
 */
export interface HandDecisions {
  /** Per-channel decision: the commanded LEVEL (0..6, the lever's
   *  7-step vocabulary) or ACTION_HOLD (-1) for a lever the hand did
   *  not move — or null while the instructor eases (no decisions
   *  happen). */
  readonly choices: ArrayLike<number> | null;
  /** Whether a decision window just opened. */
  readonly fresh: boolean;
}

/** One aircraft — see the file header. */
export class Aircraft {
  readonly body: RigidBody;
  private readonly actuators = new SurfacesState();
  /** The inputs last commanded — sealed: written only by command(). */
  private readonly inputs = new Float64Array(4);

  constructor(body: RigidBody) {
    this.body = body;
  }

  /**
   * THE SINGLE INPUT PATH — sealed. Whoever is at the controls (the
   * instructor's deterministic hand or the student's AI), inputs enter
   * the machine HERE and nowhere else: the commanded targets become the
   * machine's inputs and the actuators slew toward them.
   */
  command(targets: ArrayLike<number>, dt: number): void {
    for (let channel = 0; channel < 4; channel++) {
      this.inputs[channel] = targets[channel];
    }
    this.actuators.command(targets, dt);
  }

  /** One physics tick: the body answers its actuators. */
  step(dt: number, agl: number): void {
    this.body.step(this.actuators, dt, GRAVITY, RHO_SEA_LEVEL, agl);
  }

  /** The inputs last commanded through command() — a copy, sealed. */
  get handCommands(): Float64Array {
    return Float64Array.from(this.inputs);
  }

  /** Where the actuators actually are — a copy, sealed. */
  get surfacePositions(): Float64Array {
    return Float64Array.from(this.actuators.positions);
  }

  // --- flight instruments (the panel's ASI / VSI / ALT / turn) -----------

  /** Airspeed indicator: total airspeed, m/s. */
  get asi(): number {
    return this.body.velocity.length();
  }

  /** Vertical speed indicator: climb, m/s (positive up). */
  get vsi(): number {
    return this.body.velocity.z;
  }

  /** Altitude, m above the origin. */
  get altitude(): number {
    return this.body.position.z;
  }

  /** Heading, rad from north (+X), turning right positive. */
  get heading(): number {
    const [forward] = this.body.bodyAxes;
    return Math.atan2(forward.y, forward.x);
  }

  /** Forward axis components (the nose direction in the world). */
  get forward(): readonly [number, number] {
    const [fwd] = this.body.bodyAxes;
    return [fwd.x, fwd.y];
  }

  /** Deck pitch, rad positive nose up. */
  get pitch(): number {
    return this.body.deckPitch;
  }

  /** Bank, rad positive right wing down. */
  get bank(): number {
    return this.body.bank;
  }

  /** Body rotation rates, rad/s: roll, pitch, yaw. */
  get rollRate(): number {
    return this.body.rateP;
  }

  get pitchRate(): number {
    return this.body.rateQ;
  }

  get yawRate(): number {
    return this.body.rateR;
  }

  // --- engine + control instruments ----------------------------------------

  /** Throttle lever position, 0..1 (the machine's channel). */
  get throttle(): number {
    return this.actuators.positions[THROTTLE];
  }

  /** Control surface positions, rad (where the actuators actually are). */
  get aileron(): number {
    return this.actuators.positions[AILERON];
  }

  get elevator(): number {
    return this.actuators.positions[ELEVATOR];
  }

  get rudder(): number {
    return this.actuators.positions[RUDDER];
  }
}
