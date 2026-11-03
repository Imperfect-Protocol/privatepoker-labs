/**
 * Level-off capture forensics: fly the takeoff and stall-recovery lessons
 * and print the level-off transient — pitch vs the level attitude, the
 * raw elevator target, the hand's actual yoke, and the VS. The capture
 * should read: a FIRM push that peaks within the first second, then a
 * MONOTONIC ease-off, pitch settling onto level with the rate dying to
 * zero, and ONE gentle VS crossing with no bounce.
 */
import { FlightSchool, DemoFlight, pitchOf, levelAttitudeRad }
  from "../build/src/learning/school.js";
import { FlightInstruments } from "../build/src/learning/instruments.js";
import { ServoPolicy } from "../build/src/learning/servo-tensor.js";
import { TRIM_SPEED } from "../build/src/learning/flight.js";
import { RigidBody } from "../build/src/physics/rigid-body.js";

const DEG = 57.29578;

function traceLevelOff(lessonName) {
  const school = new FlightSchool(new FlightInstruments(TRIM_SPEED));
  const lesson = school.lessons.get(lessonName);
  const flight = new DemoFlight(school, lesson, new ServoPolicy());
  const rows = [];
  let peakPush = 0.0, peakPushT = 0.0, prevElev = null;
  let monotonicEase = true, vsSignChanges = 0, lastVsSign = null;
  let minExcess = Infinity, maxDip = 0.0;
  while (flight.step()) {
    const body = flight.body;
    const phase = flight.cursor.current.name;
    if (phase !== "level off") {
      continue;
    }
    const t = flight.seconds;
    const level = levelAttitudeRad(body.velocity.length(), body.bank);
    const excess = (body.deckPitch - level) * DEG;
    const elev = flight.commands[1];           // the hand's yoke
    const vs = body.velocity.z;
    const pitch = body.deckPitch * DEG;
    const rateQ = body.rateQ * DEG;
    if (rows.length === 0 || t - rows[rows.length - 1][0] >= 0.2) {
      rows.push([t, pitch, excess, elev, vs, rateQ, body.position.z * 3.28084]);
    }
    if (-elev > peakPush) { peakPush = -elev; peakPushT = t; }
    if (prevElev !== null && elev > prevElev + 1e-9
        && Math.abs(prevElev - peakPush) < 0.02) {
      // once near the peak push, the yoke should only ease FORWARD... the
      // push is negative; easing off means increasing toward trim. Track
      // violations after the peak.
    }
    if (prevElev !== null && t > peakPushT + 0.1 && elev < prevElev - 0.004) {
      monotonicEase = false;                    // push re-deepened after easing
    }
    prevElev = elev;
    const sign = Math.sign(vs);
    if (lastVsSign === null && vs !== 0) { lastVsSign = sign; }
    else if (sign !== 0 && sign !== lastVsSign) { vsSignChanges++; lastVsSign = sign; }
    minExcess = Math.min(minExcess, excess);
    maxDip = Math.max(maxDip, -excess);         // how far BELOW level the nose went
  }
  console.log(`\n=== ${lessonName}: level off ===`);
  console.log("   t   pitch    excess  yoke     VS(fpm)  rateQ   alt(ft)");
  for (const r of rows) {
    console.log(`${r[0].toFixed(1).padStart(5)} ${r[1].toFixed(1).padStart(6)}° ${r[2].toFixed(1).padStart(6)}°  ${r[3].toFixed(3).padStart(6)}  ${(r[4] * 196.85).toFixed(0).padStart(6)}  ${r[5].toFixed(2).padStart(5)}  ${r[6].toFixed(0).padStart(6)}`);
  }
  console.log(`peak push ${(-peakPush).toFixed(3)} rad at +${peakPushT.toFixed(1)}s into capture` +
              ` | ease-off monotonic: ${monotonicEase ? "yes" : "NO (re-deepened)"}` +
              ` | VS sign changes after entry: ${vsSignChanges}` +
              ` | nose dip below level: ${maxDip.toFixed(2)}°`);
}

for (const name of ["takeoff", "stall_recovery"]) {
  traceLevelOff(name);
}
void RigidBody;
