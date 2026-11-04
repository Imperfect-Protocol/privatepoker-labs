/**
 * NaN HUNT: run the flock boot, then fly exam + shadow + wave episodes,
 * scanning every layer for NaN — tensor weights, fired actions, hand
 * commands, actuator positions, HUD labels.
 */
import { BootTrainer } from "../build/src/app/boot-train.js";
import { FlightSchool, ExamFlight, ShadowFlight, InstructorTwin } from "../build/src/learning/school.js";
import { FlightInstruments } from "../build/src/learning/instruments.js";
import { ServoPolicy } from "../build/src/learning/servo-tensor.js";
import { NumpyPcg64 } from "../build/src/core/rng.js";
import { TRIM_SPEED, LiveEpisode } from "../build/src/learning/flight.js";
import { WavesShow } from "../build/src/app/waves-show.js";

const hasNaN = (arr) => {
  for (let i = 0; i < arr.length; i++) {
    if (!Number.isFinite(arr[i])) return true;
  }
  return false;
};

const brain = new BootTrainer({ flockSize: 6, reps: 2 });
let p = null;
while (p === null || !p.done) {
  p = brain.runChunk(1000.0);
  if (p.fraction >= 1.0) brain.finishNow();
}
const graduate = brain.graduate;
console.log("boot done");

// scan the graduate's tensors for non-finite weights
const snap = graduate.snapshot();
let nanWeights = 0;
for (const [name, t] of Object.entries(snap)) {
  if (name === "errorSigns") continue;
  if (!(t instanceof Map)) continue;
  for (const layers of t.values()) {
    if (hasNaN(layers)) nanWeights++;
  }
}
console.log("snapshot NaN layers:", nanWeights);
// snapshot layout differs; do a direct scan via stored weights:
const scan = (tensor, name) => {
  let bad = 0;
  for (const layers of tensor.values.values()) {
    if (!(layers instanceof Float32Array)) continue;
    if (hasNaN(layers)) bad++;
  }
  console.log(`${name}: ${bad} NaN layers of ${tensor.values.size}`);
};
scan(graduate.locationTensor, "location");
scan(graduate.energyTensor, "energy");
scan(graduate.flightTensor, "flight");
scan(graduate.controlsTensor, "controls");
scan(graduate.skillTensor, "skill");
scan(graduate.goalTensor, "goal");
scan(graduate.servoTensor, "servo");

// fly every lesson as an exam flight; watch state + labels
const school = new FlightSchool(new FlightInstruments(TRIM_SPEED));
const rng = new NumpyPcg64(1007n, 87136372517582989555478159403783844777n);
let nanReported = 0;
for (const name of school.lessons.keys()) {
  const policy = new ServoPolicy();
  policy.restore(graduate.snapshot());
  const lesson = school.lessons.get(name);
  const flight = new ExamFlight(school, lesson, policy, rng,
    school.demoSeconds(lesson) * 2.5 + 10.0);
  let steps = 0;
  while (flight.step() && steps < 20000) {
    steps++;
    const pos = flight.aircraft.surfacePositions;
    if (hasNaN(pos) || hasNaN(flight.body.position) || hasNaN(flight.body.velocity)) {
      const view = flight.stateView();
      console.log(`NaN STATE in ${name} at t=${flight.seconds.toFixed(1)}s:`,
        "positions", [...pos].map((v) => v.toFixed(3)).join(","),
        "| servo command:", view.servo.command,
        "| acted:", view.acted);
      nanReported++;
      break;
    }
    const view = flight.stateView();
    if (view.servo.command.includes("NaN")) {
      console.log(`NaN LABEL in ${name}: "${view.servo.command}" at tick ${view.decisionCount}`);
      nanReported++;
      break;
    }
  }
  flight.release();
}
console.log("exam flights NaN events:", nanReported);

// shadow flight
{
  const policy = new ServoPolicy();
  policy.restore(graduate.snapshot());
  const lesson = school.lessons.get("spin_recovery_left") ?? school.lessons.get("stall_recovery");
  const twin = new InstructorTwin(lesson);
  const flight = new ShadowFlight(school, lesson, policy, rng, 0, twin, 120.0);
  let steps = 0;
  while (flight.step() && steps < 20000) {
    steps++;
    if (hasNaN(flight.aircraft.surfacePositions)) {
      console.log("NaN in shadow flight at", flight.seconds.toFixed(1));
      break;
    }
  }
  console.log("shadow flight done, steps:", steps);
}

// waves
{
  const waves = new WavesShow({ state: "35399562948360463058890781895381311971", inc: "87136372517582989555478159403783844777" }, graduate);
  for (let i = 0; i < 20000; i++) waves.step();
  console.log("waves stepped, alive:", waves.aliveCount);
}
console.log("hunt complete");
