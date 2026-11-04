/**
 * Tensor debug: one demo + one repeat of one maneuver, printing the
 * student's decision cascade (bus code, element, actions, commands)
 * next to the physics it produces.
 */
import {
  FlightSchool, DemoFlight, ExamFlight, DECISION_EVERY,
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
const trace = [];
const demo = new DemoFlight(school, lesson, policy, 0);
{
  const rawImprint = policy.imprint.bind(policy);
  policy.imprint = (goalId, elementId, sense, actions, context, amount, epoch) => {
    const bus = rawImprint(goalId, elementId, sense, actions, context, amount, epoch);
    trace.push(`stamp el=${elementId} bus=${bus} act=[${Array.from(actions)}]`);
    return bus;
  };
  const rawTransition = policy.stampTransition.bind(policy);
  policy.stampTransition = (goalId, from, situation, toGoal, to, amount, epoch) => {
    trace.push(`transition ${from}@${situation} -> ${to}`);
    return rawTransition(goalId, from, situation, toGoal, to, amount, epoch);
  };
}
{
  // log the demo's raw targets every decision window
  const cursor = demo.cursor;
  const rawPhase = cursor.phaseTargets.bind(cursor);
  let calls = 0;
  cursor.phaseTargets = (body) => {
    const targets = rawPhase(body);
    if (calls % 10 === 0) {
      trace.push(`raw   el=${cursor.current.elementId} ` +
        `targets=[${Array.from(targets).map((v) => v.toFixed(3)).join(",")}]`);
    }
    calls += 1;
    return targets;
  };
}
while (demo.step()) { /* teach */ }
console.log(`demo done: ${demo.stamped} stamps, keys ${policy.storedCombinations}`);
const traceSample = [];
for (let i = 0; i < trace.length; i += Math.max(1, Math.floor(trace.length / 40))) {
  traceSample.push(trace[i]);
}
console.log(traceSample.join("\n"));
console.log(`tensors: loc ${policy.locationTensor.storedCombinations}` +
  ` energy ${policy.energyTensor.storedCombinations}` +
  ` flight ${policy.flightTensor.storedCombinations}` +
  ` goal ${policy.goalTensor.storedCombinations}` +
  ` servo ${policy.servoTensor.storedCombinations}`);

const exam = new ExamFlight(school, lesson, policy,
  new NumpyPcg64(RNG0_STATE, RNG0_INC),
  school.demoSeconds(lesson) * 2.2 + 5.0);
const instruments = school.instruments;
let tick = 0;
while (exam.step()) {
  if (tick % DECISION_EVERY === 0) {
    const hand = exam.handInputs;
    const choices = hand.choices === null ? "—" :
      Array.from(hand.choices).join(",");
    const commands = Array.from(exam.handCommands)
      .map((v) => v.toFixed(2)).join(",");
    console.log(`t=${(tick * 0.02).toFixed(1).padStart(6)}s` +
      ` el=${exam.chosenElement.name.padEnd(14)}` +
      ` act=[${choices}] cmd=[${commands}]` +
      ` bank=${(exam.body.bank * 57.3).toFixed(0)} alt=${(exam.body.position.z * 3.28).toFixed(0)}` +
      ` asi=${(exam.body.velocity.length() * 2.237).toFixed(0)}`);
  }
  tick += 1;
  if (tick > 50 * 60) { break; }
}
console.log(`repeat done: score ${exam.goalScore.toFixed(2)}`);
for (const verdict of exam.goalVerdicts()) {
  console.log(`  ${verdict.fraction >= 0.999 ? "✓" : verdict.fraction >= 0.5 ? "±" : "✗"} ${verdict.label} (${verdict.fraction.toFixed(2)})`);
}
