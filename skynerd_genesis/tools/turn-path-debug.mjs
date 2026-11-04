/**
 * Turn forensics: fly the instructor's turn demos, CAPTURE THE PATH, and
 * check the user's spec: VSI ~0 at all times, a perfect half circle,
 * constant altitude, no drift, no spin.
 */
import { FlightSchool, DemoFlight } from "../build/src/learning/school.js";
import { FlightInstruments } from "../build/src/learning/instruments.js";
import { ServoPolicy } from "../build/src/learning/servo-tensor.js";
import { TRIM_SPEED } from "../build/src/learning/flight.js";

const school = new FlightSchool(new FlightInstruments(TRIM_SPEED));
const mph = (v) => v * 2.23694;
const fpm = (v) => v * 196.85;
const deg = (r) => r * 57.29578;

/** Kasa least-squares circle fit; returns center, radius, rms (all meters). */
function fitCircle(pts) {
  const [x0, y0] = pts[0];
  let su = 0, sv = 0, suu = 0, svv = 0, suv = 0;
  let su3 = 0, sv3 = 0, suv2 = 0, svu2 = 0;
  const n = pts.length;
  for (const [x, y] of pts) {
    const u = x - x0, v = y - y0;
    su += u; sv += v; suu += u * u; svv += v * v; suv += u * v;
    su3 += u ** 3; sv3 += v ** 3; suv2 += u * v * v; svu2 += v * u * u;
  }
  // A [D,E,F]' = -[Su3+Suv2, Svu2+Sv3, Suu+Svv]  (u^2+v^2+Du+Ev+F=0)
  const A = [[suu, suv, su], [suv, svv, sv], [su, sv, n]];
  const rhs = [-(su3 + suv2), -(svu2 + sv3), -(suu + svv)];
  const det = (m) => m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
    - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
    + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0]);
  const swapCol = (m, col, v) => m.map((row, i) => row.map((x, j) => j === col ? v[i] : x));
  const D = det(A);
  const d = det(swapCol(A, 0, rhs)) / D;
  const e = det(swapCol(A, 1, rhs)) / D;
  const f = det(swapCol(A, 2, rhs)) / D;
  const cx = x0 - d / 2, cy = y0 - e / 2;
  const r = Math.sqrt(Math.max(0, d * d / 4 + e * e / 4 - f));
  let sse = 0;
  for (const [x, y] of pts) sse += (Math.hypot(x - cx, y - cy) - r) ** 2;
  return { cx, cy, r, rms: Math.sqrt(sse / n) };
}

for (const name of ["turn_left_180", "turn_right_180",
                    "steep_turn_left_180", "steep_turn_right_180"]) {
  const lesson = school.lessons.get(name);
  const policy = new ServoPolicy();
  const flight = new DemoFlight(school, lesson, policy);
  const rows = [];
  while (flight.step()) {
    const b = flight.body;
    const [fwd] = b.bodyAxes;
    rows.push({
      t: flight.seconds, phase: flight.cursor.current.name,
      x: b.position.x, y: b.position.y, z: b.position.z,
      hdg: Math.atan2(fwd.y, fwd.x),
      asi: mph(b.velocity.length()), vs: fpm(b.velocity.z),
      bank: deg(b.bank), pitch: deg(b.deckPitch), alpha: b.attitude ? 0 : 0,
      rateP: b.rateP, rateQ: b.rateQ, rateR: b.rateR,
      ail: flight.commands[0], elev: flight.commands[1], rud: flight.commands[2],
      int: flight.cursor.pitchIntegral,
    });
  }
  console.log(`\n=== ${name} — ${(flight.seconds).toFixed(1)}s, ended in '${flight.cursor.current.name}' ===`);
  console.log("    t  phase          bank   pitch    ASI     VS   ail    elev   int    rP     rQ     rR     x        y      alt");
  for (let i = 0; i < rows.length; i += 25) {
    const p = rows[i];
    console.log(`${p.t.toFixed(1).padStart(6)}  ${p.phase.padEnd(13)} ${p.bank.toFixed(1).padStart(5)}°  ${p.pitch.toFixed(1).padStart(5)}°  ${p.asi.toFixed(1).padStart(5)}  ${p.vs.toFixed(0).padStart(5)}  ${p.ail.toFixed(2).padStart(5)}  ${p.elev.toFixed(3).padStart(6)}  ${p.int.toFixed(3).padStart(5)}  ${p.rateP.toFixed(2).padStart(5)}  ${p.rateQ.toFixed(2).padStart(5)}  ${p.rateR.toFixed(2).padStart(5)}  ${p.x.toFixed(0).padStart(6)}  ${p.y.toFixed(0).padStart(6)}  ${(p.z * 3.28084).toFixed(0).padStart(5)}`);
  }

  // The TURN phase (banked): the geometry under test.
  const turn = rows.filter((p) => p.phase === "turn");
  if (turn.length > 10) {
    const fit = fitCircle(turn.map((p) => [p.x, p.y]));
    // Unwrapped heading sweep across the turn phase: a half circle = 180 deg.
    let sweep = 0;
    for (let i = 1; i < turn.length; i++) {
      let d = turn[i].hdg - turn[i - 1].hdg;
      while (d > Math.PI) d -= 2 * Math.PI;
      while (d < -Math.PI) d += 2 * Math.PI;
      sweep += d;
    }
    const alts = turn.map((p) => p.z);
    const vsAbs = turn.map((p) => Math.abs(p.vs));
    const maxBank = Math.max(...turn.map((p) => Math.abs(p.bank)));
    const maxR = Math.max(...rows.map((p) => Math.abs(p.rateR)));
    // Drift: lateral distance from the entry course line, at turn exit.
    const [ex, ey] = [turn[0].x, turn[0].y];
    const h0 = turn[0].hdg;
    const end = turn[turn.length - 1];
    const along = (end.x - ex) * Math.cos(h0) + (end.y - ey) * Math.sin(h0);
    const across = -(end.x - ex) * Math.sin(h0) + (end.y - ey) * Math.cos(h0);
    console.log(`TURN-PHASE GEOMETRY`);
    console.log(`  circle: r ${fit.r.toFixed(0)} m, rms error ${fit.rms.toFixed(1)} m ${fit.rms < 3.0 ? "(PERFECT ARC)" : "(NOT A CIRCLE)"}`);
    console.log(`  heading sweep ${deg(sweep).toFixed(1)}° (spec 180) · bank held ${maxBank.toFixed(0)}°`);
    console.log(`  altitude span ${(Math.max(...alts) - Math.min(...alts)) * 3.28084} ft · VS max ${Math.max(...vsAbs).toFixed(0)} fpm, mean ${(vsAbs.reduce((a, c) => a + c, 0) / vsAbs.length).toFixed(0)} fpm`);
    console.log(`  exit displacement: ${along.toFixed(0)} m along course, ${across.toFixed(0)} m across (drift) · max |rateR| ${maxR.toFixed(2)} rad/s`);
  }
  // The user's spec: entry heading 0 -> FINAL heading exactly 180.
  const last = rows[rows.length - 1];
  let hdg = Math.atan2(Math.sin(last.hdg), Math.cos(last.hdg));   // wrap
  while (hdg < 0) hdg += 2 * Math.PI;      // compass reading: 0..360
  console.log(`  FINAL HEADING ${deg(hdg).toFixed(1)}° (spec 180.0) · end phase '${last.phase}'`);
}
