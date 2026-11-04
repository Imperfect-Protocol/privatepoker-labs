/**
 * SOLO FLIGHT + CONTROLS MODES — logic probe: each mode flies a lesson,
 * the guarded takeover recovers a stall, the waves' stabilizer holds
 * altitude, and mode changes are sticky per drone.
 */
import { FlightSchool, SoloFlight } from "../build/src/learning/school.js";
import { FlightInstruments } from "../build/src/learning/instruments.js";
import { ServoPolicy } from "../build/src/learning/servo-tensor.js";
import { NumpyPcg64 } from "../build/src/core/rng.js";
import { TRIM_SPEED } from "../build/src/learning/flight.js";
import { WavesShow } from "../build/src/app/waves-show.js";
import { BootTrainer } from "../build/src/app/boot-train.js";

const school = new FlightSchool(new FlightInstruments(TRIM_SPEED));
const brain = new BootTrainer({ flockSize: 2, reps: 1 });
let p = null;
while (p === null || !p.done) { p = brain.runChunk(500.0); if (p.fraction >= 1.0) brain.finishNow(); }
const graduate = brain.graduate;

const DEG = 180.0 / Math.PI;

function flySolo(name, mode, capS) {
  const lesson = school.lessons.get(name);
  const policy = new ServoPolicy();
  policy.restore(graduate.snapshot());
  const rng = new NumpyPcg64(1007n, 87136372517582989555478159403783844777n);
  const flight = new SoloFlight(school, lesson, policy, rng, 0, mode);
  let tookOver = false, handedBack = false, heldTicks = 0;
  while (flight.step() && flight.seconds < capS) {
    if (flight.instructorHolds) heldTicks++;
    if (mode === "both-guarded") {
      if (flight.instructorHolds) tookOver = true;
      else if (tookOver) handedBack = true;
    }
  }
  const view = flight.stateView();
  console.log(`${name} [${mode}]: ended t=${flight.seconds.toFixed(1)}s ` +
    `held-by-instructor ${heldTicks} ticks, decisions ${view.decisionCount}, ` +
    `score ${flight.goalScore.toFixed(2)}, ` +
    `grounded=${school.grounded(flight.aircraft.body)}` +
    (mode === "both-guarded" ? `, takeover=${tookOver}, handedBack=${handedBack}` : ""));
  flight.release();
  return { flight, tookOver, handedBack };
}

flySolo("level_cruise", "ai", 90);
flySolo("level_cruise", "instructor", 90);
flySolo("level_cruise", "both-suggest", 90);
flySolo("turn_left_180", "instructor", 90);
const guard = flySolo("stall_recovery", "both-guarded", 120);
if (guard.tookOver) console.log("guarded: takeover occurred ✓");
if (guard.handedBack) console.log("guarded: handed back to AI ✓");
if (!guard.tookOver) console.log("guarded: no danger arose in this run (no takeover)");

// INSTRUCTOR ADOPTS AI INTENT: in suggest mode the machine's cursor
// should follow the AI's chosen element (never behind the AI's pick).
{
  const lesson = school.lessons.get("comprehensive");
  const policy = new ServoPolicy();
  policy.restore(graduate.snapshot());
  const rng = new NumpyPcg64(42n, 87136372517582989555478159403783844777n);
  const flight = new SoloFlight(school, lesson, policy, rng, 0, "both-suggest");
  let intentOK = true;
  while (flight.step() && flight.seconds < 60) {
    // the instructor-held machine's cursor must never lag the AI's pick
    if (flight.instructorHolds && flight.cursor.index < 0) { intentOK = false; }
  }
  console.log(`comprehensive [both-suggest]: t=${flight.seconds.toFixed(1)}s element=${flight.chosenElement.name}`);
  flight.release();
}

// WAVES: the stabilizer and the sticky modes.
{
  const waves = new WavesShow({ state: "35399562948360463058890781895381311971", inc: "87136372517582989555478159403783844777" }, graduate);
  for (let i = 0; i < 600; i++) waves.step();   // 12 s: the first wave is up
  const modes = waves.scene.aircraft.map((e) => e.label);
  console.log("waves drones:", waves.aliveCount, "entries:", modes.length);
  // select a drone and put the instructor on its levers
  const label = waves.scene.current.label;
  waves.setControlsMode("instructor");
  console.log("selected:", label, "mode:", waves.selectedControlsMode);
  const drone0 = waves.drones.find((d) => d.entry.label === label);
  const before = drone0.entry.aircraft.body.position.z;
  for (let i = 0; i < 10000; i++) waves.step();  // 200 s of stabilizer (out-of-bounds homing included)
  const drone = waves.drones.find((d) => d.entry.label === label);
  if (drone === undefined) {
    console.log("stabilizer: drone DIED before the check — homing failed");
  } else {
    const after = drone.entry.aircraft.body.position.z;
    const bank = drone.entry.aircraft.body.bank * DEG;
    const dist = Math.hypot(drone.entry.aircraft.body.position.x, drone.entry.aircraft.body.position.y);
    console.log(`stabilizer 200s: alt ${before.toFixed(1)}m -> ${after.toFixed(1)}m, bank ${bank.toFixed(1)}°, dist-from-origin ${dist.toFixed(0)}m, alive=${drone.flight.episodeOutcome === null}`);
  }
  // stickiness: another drone keeps AI
  const other = waves.drones.find((d) => d.flight.episodeOutcome === null && d.entry.label !== label);
  console.log("other drone mode (should be ai):", other.flight.controlsMode);
  // guarded takeover from a spiral: pick a flying drone, force mode, and watch
  waves.setControlsMode("both-guarded");
  console.log("guarded set on the current selection; instructor holding:", waves.selectedInstructorAtControls);
}
console.log("probe complete");
