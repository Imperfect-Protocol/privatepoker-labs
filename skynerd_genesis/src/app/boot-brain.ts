/**
 * The boot dialog's BRAIN VIEW: one rotating 3D wireframe cube per
 * tensor — a grid on every face — each labeled below, with the synapses
 * between them drawn live: pulses travel the connections the way
 * signals travel synapses, and each cube brightens as its tensor fills
 * with what the classroom stamps into it.
 *
 * The EIGHT tensors, one cube each, in three tiers:
 *   row 1 — the SENSOR tensors (they process the Aircraft's panel into
 *     situational signals; each is recurrent: its own previous code,
 *     the goal's element, and its skill patch bin are part of its key):
 *       LocationTensor   alt × hsi → SITUATION 64
 *       EnergyTensor     alt × asi × vsi × pitch × rpm → SITUATION 32
 *       FlightTensor     vsi × asi × pitch × bank × turn → SITUATION 64
 *       ControlsTensor   aileron × elevator × rudder × throttle → SITUATION 64
 *   row 2 — THE TWO SKILLS TENSORS flanking the GOAL tensor (the
 *     fuzzy state machine): LeftSkillsTensor on its left (the
 *     perception panel: location × energy × element × servo),
 *     RightSkillsTensor on its right (the execution panel:
 *     flight × controls × element × servo), the bridge between them
 *   row 3 — the SERVO tensor (receives the two panels' forwards —
 *     never from the Aircraft — and drives the levers)
 */
import type { ServoPolicy } from "../learning/servo-tensor.js";
import {
  GOAL_AXIS, ELEMENT_AXIS, LEFT_WIRE_AXIS, RIGHT_WIRE_AXIS, SKILL_BIN_AXIS,
} from "../learning/servo-tensor.js";
import {
  LOCATION_SHAPE, ENERGY_SHAPE, FLIGHT_SHAPE, CONTROLS_SHAPE,
  LOC_CLASS_AXIS, ENERGY_CLASS_AXIS, FLIGHT_CLASS_AXIS, CONTROLS_CLASS_AXIS,
  SITUATION_AXIS, stopCount,
} from "../learning/instruments.js";
import { PATCH_SLOT_AXIS } from "../learning/servo-tensor.js";

/** One cube's fill state: the tensor's live key count. */
export interface TensorCounts {
  location: number;
  energy: number;
  flight: number;
  controls: number;
  leftSkill: number;
  rightSkill: number;
  goal: number;
  servo: number;
}

/** The eight tensors' live key counts (one cube per tensor). */
export function tensorCounts(policy: ServoPolicy): TensorCounts {
  return {
    location: policy.locationTensor.storedCombinations,
    energy: policy.energyTensor.storedCombinations,
    flight: policy.flightTensor.storedCombinations,
    controls: policy.controlsTensor.storedCombinations,
    leftSkill: policy.leftSkillTensor.storedCombinations,
    rightSkill: policy.rightSkillTensor.storedCombinations,
    goal: policy.goalTensor.storedCombinations,
    servo: policy.servoTensor.storedCombinations,
  };
}

/** The cube layout: position as a fraction of the canvas, its label
 *  below it (name on one line, its true dimensions in braces on the
 *  next), and its colour. The cubes are VISUAL; the dimensions are the
 *  truth. */
interface CubeSpot {
  readonly key: keyof TensorCounts;
  readonly name: string;
  /** The tensor's true dimensions in braces, one line. */
  readonly dims: string;
  readonly rgb: readonly [number, number, number];
  readonly fx: number;
  readonly fy: number;
}

/** The tensor's true dimensions in braces, input → output on one line. */
const tensorDims = (input: readonly number[], output: readonly number[]): string =>
  `(${input.map(String).join("\u00d7")}\u2192${output.map(String).join("\u00d7")})`;

/**
 * THE LAYOUT — three tiers, the way the signals flow:
 *   row 1: the four SENSOR tensors (they process the Aircraft's panel)
 *   row 2: THE LEFT SKILLS TENSOR (the perception panel), the GOAL
 *          tensor (the fuzzy state machine), THE RIGHT SKILLS TENSOR
 *          (the execution panel) — the two panels flanking the goal
 *   row 3: the SERVO tensor (receives the two panels' forwards —
 *          never the Aircraft — and drives the levers)
 */
const CUBE_SPOTS: readonly CubeSpot[] = [
  { key: "location",  name: "LocationTensor",
    dims: tensorDims([...LOCATION_SHAPE, LOC_CLASS_AXIS, ELEMENT_AXIS,
                       SKILL_BIN_AXIS], [LOC_CLASS_AXIS]),
    rgb: [87, 217, 163],  fx: 0.125, fy: 0.15 },
  { key: "energy",    name: "EnergyTensor",
    dims: tensorDims([...ENERGY_SHAPE, ENERGY_CLASS_AXIS, ELEMENT_AXIS,
                       SKILL_BIN_AXIS], [ENERGY_CLASS_AXIS]),
    rgb: [255, 184, 74],  fx: 0.375, fy: 0.15 },
  { key: "flight",    name: "FlightTensor",
    dims: tensorDims([...FLIGHT_SHAPE, FLIGHT_CLASS_AXIS, ELEMENT_AXIS,
                       SKILL_BIN_AXIS], [FLIGHT_CLASS_AXIS]),
    rgb: [60, 130, 255],  fx: 0.625, fy: 0.15 },
  { key: "controls",  name: "ControlsTensor",
    dims: tensorDims([...CONTROLS_SHAPE, CONTROLS_CLASS_AXIS, ELEMENT_AXIS,
                       SKILL_BIN_AXIS], [CONTROLS_CLASS_AXIS]),
    rgb: [140, 226, 255], fx: 0.875, fy: 0.15 },
  { key: "leftSkill", name: "LeftSkillsTensor",
    dims: tensorDims([LEFT_WIRE_AXIS, PATCH_SLOT_AXIS, PATCH_SLOT_AXIS], []),
    rgb: [94, 240, 158],  fx: 0.25, fy: 0.50 },
  { key: "goal",      name: "GoalTensor",
    dims: tensorDims([GOAL_AXIS, ELEMENT_AXIS, SITUATION_AXIS],
                     [GOAL_AXIS, ELEMENT_AXIS]),
    rgb: [180, 140, 255], fx: 0.50, fy: 0.50 },
  { key: "rightSkill", name: "RightSkillsTensor",
    dims: tensorDims([RIGHT_WIRE_AXIS, PATCH_SLOT_AXIS, PATCH_SLOT_AXIS], []),
    rgb: [45, 212, 191],  fx: 0.75, fy: 0.50 },
  { key: "servo",     name: "ServoTensor",
    dims: tensorDims([GOAL_AXIS, ELEMENT_AXIS, ENERGY_CLASS_AXIS,
                       FLIGHT_CLASS_AXIS, SKILL_BIN_AXIS, SKILL_BIN_AXIS],
      [stopCount(0), stopCount(1), stopCount(2), stopCount(3)]),
    rgb: [255, 106, 122], fx: 0.50, fy: 0.85 },
];

/**
 * THE SYNAPSES (by index into CUBE_SPOTS). The sensors connect
 * DIRECTLY to the goal tensor and their side's skills panel; the
 * goal tensor feeds its element back to the sensors (the recurrence)
 * and to both panels; each panel wires its own side's signals and
 * BRIDGES the other panel — and the SERVO's only inbound synapses
 * are the two panels' forwards: ALL its input arrives from the two
 * skills panels.
 */
const SYNAPSES: ReadonlyArray<readonly [number, number]> = [
  // the sensors -> the goal and their side's panel (direct)
  [0, 5], [1, 5], [2, 5], [3, 5],
  [0, 4], [1, 4],            // location, energy -> the LEFT panel
  [2, 6], [3, 6],            // flight, controls -> the RIGHT panel
  // the goal tensor: its element -> the sensors (the recurrence)
  // and both panels
  [5, 0], [5, 1], [5, 2], [5, 3],
  [5, 4], [5, 6],
  // the panels: their wires' verdict -> their sensors and the goal;
  // THE BRIDGE -> each other; their FORWARDS -> the servo (the
  // servo's whole input); the servo's last command -> back to both
  [4, 0], [4, 1], [4, 5],
  [6, 2], [6, 3], [6, 5],
  [4, 6], [6, 4],            // the bridge between the two panels
  [4, 7], [6, 7],
  [7, 4], [7, 6],
];

/** The unit cube: corners and the six faces (corner indices, in order). */
const CUBE_CORNERS: ReadonlyArray<readonly [number, number, number]> = [
  [-1, -1, -1], [1, -1, -1], [1, 1, -1], [-1, 1, -1],
  [-1, -1, 1], [1, -1, 1], [1, 1, 1], [-1, 1, 1],
];
const CUBE_FACES: ReadonlyArray<readonly [number, number, number, number]> = [
  [0, 1, 2, 3],   // back   (z = -1)
  [5, 4, 7, 6],   // front  (z = +1)
  [4, 0, 3, 7],   // left   (x = -1)
  [1, 5, 6, 2],   // right  (x = +1)
  [4, 5, 1, 0],   // bottom (y = -1)
  [3, 2, 6, 7],   // top    (y = +1)
];
const CUBE_EDGES: ReadonlyArray<readonly [number, number]> = [
  [0, 1], [1, 2], [2, 3], [3, 0], [4, 5], [5, 6], [6, 7], [7, 4],
  [0, 4], [1, 5], [2, 6], [3, 7],
];

/** Rotate (about Y, then a fixed tilt about X) and project a unit-cube
 *  point; returns screen x, screen y, and the depth (for face shading). */
function project(p: readonly [number, number, number], rotY: number,
                 tilt: number, cx: number, cy: number,
                 s: number): [number, number, number] {
  const x1 = p[0] * Math.cos(rotY) + p[2] * Math.sin(rotY);
  const z1 = -p[0] * Math.sin(rotY) + p[2] * Math.cos(rotY);
  const y2 = p[1] * Math.cos(tilt) - z1 * Math.sin(tilt);
  const z2 = p[1] * Math.sin(tilt) + z1 * Math.cos(tilt);
  return [cx + x1 * s, cy - y2 * s, z2];
}

function lerp3(a: readonly [number, number, number],
               b: readonly [number, number, number],
               t: number): [number, number, number] {
  return [a[0] + (b[0] - a[0]) * t,
          a[1] + (b[1] - a[1]) * t,
          a[2] + (b[2] - a[2]) * t];
}

/** One wireframe cube with a grid on every face, filled by the tensor's
 *  saturation (0..1) and shaded by which faces the rotation shows. */
function drawCube(ctx: CanvasRenderingContext2D, cx: number, cy: number,
                  s: number, rotY: number, tilt: number,
                  rgb: readonly [number, number, number], fill: number): void {
  const colour = (alpha: number) =>
    `rgba(${rgb[0]},${rgb[1]},${rgb[2]},${alpha.toFixed(3)})`;
  const proj = CUBE_CORNERS.map((c) => project(c, rotY, tilt, cx, cy, s));
  // the faces: translucent fill (brighter toward the viewer) + the grid
  for (const face of CUBE_FACES) {
    const corners = face.map((i) => CUBE_CORNERS[i]);
    const points = face.map((i) => proj[i]);
    const depth = (points[0][2] + points[1][2] + points[2][2] + points[3][2]) / 4.0;
    const facing = depth < 0.0;
    ctx.beginPath();
    ctx.moveTo(points[0][0], points[0][1]);
    for (let i = 1; i < 4; i++) {
      ctx.lineTo(points[i][0], points[i][1]);
    }
    ctx.closePath();
    ctx.fillStyle = colour((facing ? 0.10 : 0.03) + 0.30 * fill * (facing ? 1.0 : 0.25));
    ctx.fill();
    // the grid: a 3x3 division each way across the face (visual only)
    const lineAlpha = (facing ? 0.34 : 0.10) + 0.25 * fill * (facing ? 1.0 : 0.3);
    ctx.strokeStyle = colour(lineAlpha);
    ctx.lineWidth = 1;
    for (let k = 1; k < 3; k++) {
      const t = k / 3.0;
      for (const [i0, i1, j0, j1] of [[0, 3, 1, 2], [0, 1, 3, 2]]) {
        const a = project(lerp3(corners[i0], corners[i1], t), rotY, tilt, cx, cy, s);
        const b = project(lerp3(corners[j0], corners[j1], t), rotY, tilt, cx, cy, s);
        ctx.beginPath();
        ctx.moveTo(a[0], a[1]);
        ctx.lineTo(b[0], b[1]);
        ctx.stroke();
      }
    }
  }
  // the 12 edges: the frame on top of the grids
  ctx.strokeStyle = colour(0.75);
  ctx.lineWidth = 1.4;
  for (const [i, j] of CUBE_EDGES) {
    ctx.beginPath();
    ctx.moveTo(proj[i][0], proj[i][1]);
    ctx.lineTo(proj[j][0], proj[j][1]);
    ctx.stroke();
  }
}

/** Saturating fill: 0 keys → 0, a few hundred → mostly full. */
function fillOf(count: number): number {
  return count / (count + 500.0);
}

/**
 * Paint the brain view once: the synapses (with their traveling pulses),
 * the eight cubes, and the labels below them. `time` is seconds — the
 * cubes rotate and the pulses travel with it.
 */
export function drawBrainView(canvas: HTMLCanvasElement, time: number,
                              counts: TensorCounts): void {
  const dpr = window.devicePixelRatio || 1;
  const W = canvas.clientWidth;
  const H = canvas.clientHeight;
  if (W === 0 || H === 0) {
    return;
  }
  canvas.width = Math.round(W * dpr);
  canvas.height = Math.round(H * dpr);
  const ctx = canvas.getContext("2d")!;
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, W, H);

  const rotY = time * 0.55;
  const tilt = 0.42;
  const size = 30.0;
  const centres = CUBE_SPOTS.map((spot) => ({
    x: spot.fx * W,
    y: spot.fy * H,
    spot,
  }));

  // the synapses: each connection a faint line with pulses traveling it
  for (let s = 0; s < SYNAPSES.length; s++) {
    const [from, to] = SYNAPSES[s];
    const a = centres[from];
    const b = centres[to];
    ctx.strokeStyle = "rgba(40, 90, 120, 0.4)";
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(a.x, a.y);
    ctx.lineTo(b.x, b.y);
    ctx.stroke();
    // two pulses per synapse, offset so the mesh shimmers
    const rgb = a.spot.rgb;
    for (let k = 0; k < 2; k++) {
      const t = (time * 0.30 + s * 0.37 + k * 0.5) % 1.0;
      const px = a.x + (b.x - a.x) * t;
      const py = a.y + (b.y - a.y) * t;
      const pulse = 0.55 + 0.45 * Math.sin(t * Math.PI);
      ctx.fillStyle = `rgba(${rgb[0]},${rgb[1]},${rgb[2]},${pulse.toFixed(3)})`;
      ctx.beginPath();
      ctx.arc(px, py, 2.2, 0, 2 * Math.PI);
      ctx.fill();
    }
  }

  // the cubes + their labels (the name, then the dimensions below it)
  ctx.font = "12px Rajdhani, 'SF Mono', monospace";
  ctx.textAlign = "center";
  for (const centre of centres) {
    drawCube(ctx, centre.x, centre.y, size, rotY, tilt,
             centre.spot.rgb, fillOf(counts[centre.spot.key]));
    ctx.fillStyle = "rgba(200, 235, 255, 0.95)";
    ctx.fillText(centre.spot.name, centre.x, centre.y + size + 18.0);
    ctx.fillStyle = "rgba(70, 115, 140, 0.9)";
    ctx.fillText(centre.spot.dims, centre.x, centre.y + size + 32.0);
  }
}
