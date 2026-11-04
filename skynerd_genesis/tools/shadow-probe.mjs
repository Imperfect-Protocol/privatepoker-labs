/**
 * One shadow flight of one lesson on a fresh student (priors
 * installed): prints per-decision the servo's fires, the error
 * pumps' magnitudes, and the final goal verdicts. Verifies the
 * error wave actually teaches.
 */
import {
  FlightSchool, ShadowFlight, InstructorTwin, DECISION_EVERY,
} from "../build/src/learning/school.js";
import { FlightInstruments } from "../build/src/learning/instruments.js";
import { ServoPolicy } from "../build/src/learning/servo-tensor.js";
import { NumpyPcg64 } from "../build/src/core/rng.js";
import { RNG0_STATE, RNG0_INC } from "../build/src/learning/school.js";
import { TRIM_SPEED } from "../build/src/learning/flight.js";

const name = process.argv[2] ?? "turn_left_180";
const school = new FlightSchool(new FlightInstruments(TRIM_SPEED));
const lesson = school.lessons.get(name);
if (lesson === undefined) {
  throw new Error(`unknown lesson ${name}`);
}
const policy = new ServoPolicy();
const priorRng = new NumpyPcg64(20260101n, 770017n);
policy.installPriors(0, priorRng);
console.log(`priors: ${policy.storedCombinations} keys`);

const startKeys = policy.servoTensor.storedCombinations;
const startPumps = policy.errorNeurons.reduce((s, n) => s + n.pumps, 0);
const flight = new ShadowFlight(school, lesson, policy,
  new NumpyPcg64(RNG0_STATE, RNG0_INC), 0,
  new InstructorTwin(lesson),
  school.demoSeconds(lesson) * 2.2 + 5.0);
// Runtime trace: which tensors fire per decision window.
const net = flight["network"];
const rawDeliver = net.deliver.bind(net);
const counts = { forward: 0, servoStun: 0 };
net.deliver = (message) => {
  if (message.kind === "forward") { counts.forward += 1; }
  rawDeliver(message);
};
const firesOf = () => net["nodes"].map((n) => n.fires);
const fires0 = firesOf();
let tick = 0;
let acted = 0;
const fireLog = [];
while (flight.step()) {
  if (tick % DECISION_EVERY === 0 && flight.handInputs.choices !== null
      && flight.handInputs.choices.some((c) => c !== -1)) {
    acted += 1;
  }
  if (tick % DECISION_EVERY === 0 && fireLog.length < 8) {
    fireLog.push(firesOf());
  }
  tick += 1;
}
console.log("fires per window (loc,energy,flight,controls,goal,skill,servo):");
for (const f of fireLog) {
  console.log("  " + f.join(","));
}
console.log(`forward deliveries: ${counts.forward}`);
const pumps = policy.errorNeurons.reduce((s, n) => s + n.pumps, 0);
const flips = policy.errorNeurons.reduce((s, n) => s + n.flips, 0);
console.log(`shadow done: ${tick} ticks, ${acted} acted decisions`);
console.log(`servo keys ${startKeys} -> ${policy.servoTensor.storedCombinations}`);
console.log(`error pumps ${startPumps} -> ${pumps}, sign flips ${flips}`);
console.log(`score ${flight.goalScore.toFixed(2)}`);
for (const verdict of flight.goalVerdicts()) {
  console.log(`  ${verdict.fraction >= 0.999 ? "✓" : verdict.fraction >= 0.5 ? "±" : "✗"} ${verdict.label} (${verdict.fraction.toFixed(2)})`);
}
