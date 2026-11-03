/** Drive each lesson's demo flight headlessly, print phase progress. */
import { FlightSchool, DemoFlight } from "../build/src/learning/school.js";
import { FlightInstruments } from "../build/src/learning/instruments.js";
import { ServoPolicy } from "../build/src/learning/servo-tensor.js";
import { TRIM_SPEED } from "../build/src/learning/flight.js";

const school = new FlightSchool(new FlightInstruments(TRIM_SPEED));
const mph = (v) => (v * 2.23694).toFixed(1);
const ft = (m) => (m * 3.28084).toFixed(0);

for (const [name, lesson] of school.lessons) {
  const policy = new ServoPolicy();
  const flight = new DemoFlight(school, lesson, policy);
  const seen = [flight.cursor.current.name];
  let maxRate = 0, prev = null;
  let minASI = Infinity, maxASI = -Infinity, maxBank = 0, minAlt = Infinity;
  let maxVS = 0;
  while (flight.step()) {
    const p = flight.cursor.current.name;
    if (seen[seen.length - 1] !== p) { seen.push(p); }
    const b = flight.body;
    const asi = b.velocity.length();
    minASI = Math.min(minASI, asi); maxASI = Math.max(maxASI, asi);
    minAlt = Math.min(minAlt, b.position.z);
    maxVS = Math.max(maxVS, Math.abs(b.velocity.z));
    maxBank = Math.max(maxBank, Math.abs(-Math.asin(Math.max(-1, Math.min(1, b.bodyAxes[1].z)))));
    if (prev !== null) {
      const c = flight.commands;
      for (let i = 0; i < 3; i++) maxRate = Math.max(maxRate, Math.abs(c[i] - prev[i]) / 0.02);
    }
    prev = [flight.commands[0], flight.commands[1], flight.commands[2]];
  }
  console.log(`${name.padEnd(20)} ${(flight.seconds).toFixed(1).padStart(6)}s  ${seen.join(" → ")}`);
  console.log(`   end ASI ${mph(flight.body.velocity.length())} mph · alt ${ft(flight.body.position.z)} ft · stamps ${flight.stamped}`);
  console.log(`   ASI ${mph(minASI)}–${mph(maxASI)} mph · max bank ${(maxBank * 57.3).toFixed(0)}° · maxVS ${(maxVS * 196.85).toFixed(0)} fpm · min alt ${ft(minAlt)} ft · max ctrl rate ${maxRate.toFixed(2)} rad/s`);
}
