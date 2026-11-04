/** Takeoff forensics: pitch 0 + yoke neutral on the roll, rotate at 60 mph. */
import { FlightSchool, DemoFlight, pitchOf } from "../build/src/learning/school.js";
import { FlightInstruments } from "../build/src/learning/instruments.js";
import { ServoPolicy } from "../build/src/learning/servo-tensor.js";
import { TRIM_SPEED } from "../build/src/learning/flight.js";

const school = new FlightSchool(new FlightInstruments(TRIM_SPEED));
const lesson = school.lessons.get("takeoff");
const policy = new ServoPolicy();
const flight = new DemoFlight(school, lesson, policy);

const mph = (v) => (v * 2.23694).toFixed(1);
const fpm = (v) => (v * 196.85).toFixed(0);

let rollPitchMax = 0, rollYokeMax = 0, rollPitchMaxT = 0;
let climbPitchMax = 0, liftoffT = null, last = null;
console.log("  t  phase        ASI     pitch   yoke    thr   alt    VS");
while (flight.step()) {
  const b = flight.body;
  const phase = flight.cursor.current.name;
  const pitch = pitchOf(b) * 57.3;
  const yoke = flight.commands[1];
  const thr = flight.commands[3];
  if (phase === "roll") {
    if (Math.abs(pitch) > rollPitchMax) { rollPitchMax = Math.abs(pitch); rollPitchMaxT = flight.seconds; }
    rollYokeMax = Math.max(rollYokeMax, Math.abs(yoke));
  } else if (phase === "climb") {
    climbPitchMax = Math.max(climbPitchMax, pitch);
    if (liftoffT === null && b.position.z > 3.05 + 0.05) { liftoffT = flight.seconds; }
  }
  if (flight.seconds - (last ?? -1) >= 1.0) {
    last = flight.seconds;
    console.log(`${flight.seconds.toFixed(0).padStart(3)}  ${phase.padEnd(12)} ${mph(b.velocity.length()).padStart(6)}  ${pitch.toFixed(1).padStart(5)}°  ${yoke.toFixed(3).padStart(6)}  ${thr.toFixed(2)}  ${(b.position.z * 3.28084).toFixed(0).padStart(5)}  ${fpm(b.velocity.z).padStart(6)}`);
  }
}
const b = flight.body;
console.log(`\nroll: max |pitch| ${rollPitchMax.toFixed(3)}° (at t=${rollPitchMaxT}s), max |yoke| ${rollYokeMax.toFixed(4)} rad`);
console.log(`climb: max pitch ${climbPitchMax.toFixed(1)}°, liftoff ${liftoffT === null ? "never" : liftoffT.toFixed(1) + " s"}`);
console.log(`end: ${(flight.seconds).toFixed(1)}s · ASI ${mph(b.velocity.length())} mph · alt ${(b.position.z * 3.28084).toFixed(0)} ft · VS ${fpm(b.velocity.z)} fpm · stamps ${flight.stamped}`);
