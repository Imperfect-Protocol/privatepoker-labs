/**
 * Departure forensics: per-tick moment autopsy of the turn dive onset.
 * Prints the aerodynamic moment components vs the control authority so the
 * axis that overpowers the instructor is visible.
 */
import { FlightSchool, DemoFlight } from "../build/src/learning/school.js";
import { FlightInstruments } from "../build/src/learning/instruments.js";
import { ServoPolicy } from "../build/src/learning/servo-tensor.js";
import { TRIM_SPEED } from "../build/src/learning/flight.js";
import { AeroTables, controlStabilityMoment } from "../build/src/physics/aero.js";

const school = new FlightSchool(new FlightInstruments(TRIM_SPEED));
const lesson = school.lessons.get("steep_turn_left_180");
const policy = new ServoPolicy();
const flight = new DemoFlight(school, lesson, policy);
const tables = new AeroTables();
const mph = (v) => v * 2.23694;
const fpm = (v) => v * 196.85;
const deg = (r) => r * 57.29578;

const T0 = 0.0, T1 = 10.0, EVERY = 5;   // per 0.1 s
console.log("    t  bank   pitch    VS   alpha  beta   qDyn    Mx      My      Mz     eHand  eServo  aHand  rP     rQ     rR");
while (flight.step()) {
  const b = flight.body;
  const t = flight.seconds;
  if (t < T0 || t > T1 || flight.tick % EVERY !== 0) continue;
  const [fwd, right, up] = b.bodyAxes;
  const speed = b.velocity.length();
  const vFwd = b.velocity.dot(fwd);
  const vUp = b.velocity.dot(up);
  const vRight = b.velocity.dot(right);
  const alpha = Math.atan2(-vUp, Math.max(vFwd, 1e-6));
  const beta = Math.atan2(vRight, Math.hypot(vFwd, vUp));
  const qDyn = 0.5 * 1.225 * speed * speed;
  const m = controlStabilityMoment(alpha, beta, flight.servos, tables, qDyn);
  console.log(`${t.toFixed(2).padStart(5)}  ${deg(b.bank).toFixed(1).padStart(5)}°  ${deg(b.deckPitch).toFixed(1).padStart(5)}°  ${fpm(b.velocity.z).toFixed(0).padStart(5)}  ${deg(alpha).toFixed(1).padStart(5)}°  ${deg(beta).toFixed(1).padStart(5)}°  ${qDyn.toFixed(0).padStart(5)}  ${m.x.toFixed(0).padStart(6)}  ${m.y.toFixed(0).padStart(7)}  ${m.z.toFixed(0).padStart(6)}  ${flight.commands[1].toFixed(3).padStart(6)}  ${flight.servos.positions[1].toFixed(3).padStart(6)}  ${flight.commands[0].toFixed(2).padStart(5)}  ${b.rateP.toFixed(2).padStart(5)}  ${b.rateQ.toFixed(2).padStart(5)}  ${b.rateR.toFixed(2).padStart(5)}`);
}
