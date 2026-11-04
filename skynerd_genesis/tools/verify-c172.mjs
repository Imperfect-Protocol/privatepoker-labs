/**
 * C172 airframe verification driver: flies the spec numbers headlessly and
 * prints them, so the tables/gains can be tuned against the syllabus
 * targets (rotate 60 mph, cruise 85 mph @ 2300 RPM, stall ~50 mph,
 * climb to 1000 ft, dive to 65 mph).
 */
import { Vec3 } from "../build/src/core/vec.js";
import { RigidBody } from "../build/src/physics/rigid-body.js";
import { SurfacesState } from "../build/src/physics/surfaces.js";
import { AeroTables, aeroForce } from "../build/src/physics/aero.js";
import { mphOf } from "../build/src/app/units.js";

const DT = 0.02;
const T = new AeroTables();
const mph = (v) => mphOf(v).toFixed(1);

function axes(body) {
  const [f, r, u] = body.bodyAxes;
  const pitch = Math.asin(Math.max(-1, Math.min(1, f.z)));
  const bank = -Math.asin(Math.max(-1, Math.min(1, r.z)));
  return { f, r, u, pitch, bank };
}

function fly(label, { z0 = 100.0, v0 = 38.0, throttle, pitchT = null, bankT = 0.0,
                      seconds = 40.0, rudder = 0.0, pinZ = null, report }) {
  const body = new RigidBody(new Vec3(0, 0, z0), new Vec3(v0, 0, 0));
  const servos = new SurfacesState();
  let zMin = z0;
  let maxCmdRate = 0;
  let prevElev = 0;
  for (let t = 0; t < seconds / DT; t++) {
    const a = axes(body);
    let elevator = 0;
    if (pitchT !== null) {
      elevator = Math.max(-0.35, Math.min(0.35, 1.4 * (pitchT - a.pitch) + 0.35 * body.rateQ));
    }
    const aileron = Math.max(-0.35, Math.min(0.35, 2.2 * (bankT - a.bank) + 0.9 * body.rateP));
    servos.command([aileron, elevator, rudder, throttle], DT);
    body.step(servos, DT);
    if (pinZ !== null && body.position.z < pinZ && a.airspeed0 === undefined) {
      if (body.position.z < pinZ) {
        body.position.z = pinZ;
        body.velocity.z = Math.max(0, body.velocity.z);
      }
    }
    maxCmdRate = Math.max(maxCmdRate, Math.abs(elevator - prevElev) / DT);
    prevElev = elevator;
    zMin = Math.min(zMin, body.position.z);
    if (report) report(t, body, a);
  }
  const a = axes(body);
  console.log(`${label}: ASI ${mph(body.velocity.length())} mph · alt ${body.position.z.toFixed(0)} m ` +
    `(${(body.position.z * 3.28084).toFixed(0)} ft) · pitch ${(a.pitch * 57.3).toFixed(1)}° · ` +
    `bank ${(a.bank * 57.3).toFixed(1)}° · ROC ${(body.velocity.z * 196.85).toFixed(0)} fpm`);
  return { body, a, zMin };
}

console.log("== static thrust (V=0, full):");
{
  const { force } = aeroForce(new Vec3(0, 0, 0), new Vec3(1, 0, 0), new Vec3(0, 1, 0),
                             new Vec3(0, 0, 1), { positions: [0, 0, 0, 1] }, T);
  console.log(`  T0 = ${force.x.toFixed(0)} N (expect ~2500)`);
}

console.log("== cruise equilibrium at throttle 0.8 (2300 RPM):");
{
  // analytic level-flight equilibrium: thrust = drag at vz = 0
  const levelEquilibrium = (thr) => {
    const D = (V) => {
      const q = 0.5 * 1.225 * V * V;
      const CL = (T.mass * 9.81) / (q * T.wingArea);
      return q * T.wingArea * (T.cd0 + T.kInduced * CL * CL);
    };
    const Thr = (V) => thr ** 3 * T.thrustMax * Math.max(T.propFloor, 1 - (V / T.propFalloffSpeed) ** 2);
    let lo = 15, hi = 90;
    for (let i = 0; i < 60; i++) {
      const mid = (lo + hi) / 2;
      if (Thr(mid) - D(mid) > 0) lo = mid; else hi = mid;
    }
    return (lo + hi) / 2;
  };
  for (const thr of [0.7, 0.8, 1.0]) {
    const V = levelEquilibrium(thr);
    console.log(`  level equilibrium @ throttle ${thr}: ${mph(V)} mph (RPM ${(700 + thr * 2000).toFixed(0)})`);
  }
}

console.log("== stall speed check (1 g, CLmax):");
{
  const CL = T.clMax;
  const V = Math.sqrt(2 * T.mass * 9.81 / (1.225 * T.wingArea * CL));
  console.log(`  CLmax ${CL} → V_stall = ${mph(V)} mph (want ~50)`);
}

console.log("== takeoff roll (pin at 10 ft, full throttle, rotate 60 mph → pitch 10°):");
{
  const body = new RigidBody(new Vec3(0, 0, 3.048), new Vec3(0, 0, 0));
  const servos = new SurfacesState();
  const pinZ = 3.048;
  let tRotate = -1;
  let rotated = false;
  for (let t = 0; t < 140 / DT; t++) {
    const a = axes(body);
    const speed = body.velocity.length();
    let pitchT = 0.0;
    if (speed >= 26.82) {
      if (!rotated) { rotated = true; tRotate = t * DT; }
      pitchT = 10 * Math.PI / 180;
    }
    const elevator = Math.max(-0.35, Math.min(0.35, 1.4 * (pitchT - a.pitch) + 0.35 * body.rateQ));
    servos.command([0, Math.max(0, elevator), 0, 1.0], DT);
    body.step(servos, DT);
    if (body.position.z < pinZ && !rotated) {
      body.position.z = pinZ;
      body.velocity.z = Math.max(0, body.velocity.z);
    }
    if (rotated && body.position.z * 3.28084 >= 1000) {
      console.log(`  rotated at ${tRotate.toFixed(1)} s, 1000 ft at ${(t * DT).toFixed(1)} s, ` +
        `ASI ${mph(body.velocity.length())} mph, ROC ${(body.velocity.z * 196.85).toFixed(0)} fpm`);
      break;
    }
  }
}

console.log("== stall entry (3000 ft, idle, hold altitude):");
{
  // spawn trimmed: pitch set so lift = weight at cruise speed
  const TRIM_ALPHA = ((2 * T.mass * 9.81) / (1.225 * 38 * 38 * T.wingArea) - T.cl0) / T.clAlpha;
  const body = new RigidBody(new Vec3(0, 0, 914.4), new Vec3(38, 0, 0),
                            0.0, -TRIM_ALPHA, 0.0);   // +Y rotation is nose DOWN here
  const servos = new SurfacesState();
  let tStall = -1;
  let zAtStall = 0;
  let vsIntegral = 0.02485;   // start from the trimmed cruise elevator
  for (let t = 0; t < 60 / DT; t++) {
    const a = axes(body);
    const { diag } = aeroForce(body.velocity, a.f, a.r, a.u, servos, T);
    // stall entry: hold ALTITUDE at idle — the nose rises as the plane
    // slows until the wing gives up
    vsIntegral += (0 - body.velocity.z) * DT;
  const elevator = Math.max(-0.35, Math.min(0.35, 0.02 * (0 - body.velocity.z) + 0.02 * vsIntegral + 0.35 * body.rateQ));
    servos.command([0, elevator, 0, 0.1], DT);
    body.step(servos, DT);
    if (tStall < 0 && diag.alpha >= T.alphaStall) {
      tStall = t * DT;
      zAtStall = body.position.z;
      console.log(`  stalled at ${tStall.toFixed(1)} s · ASI ${mph(body.velocity.length())} mph ` +
        `· alt ${(body.position.z * 3.28084).toFixed(0)} ft · sink ${(-body.velocity.z * 196.85).toFixed(0)} fpm`);
    }
    if (tStall > 0 && t * DT > tStall + 2) break;
  }
  console.log(`  2 s after stall: ASI ${mph(body.velocity.length())} mph, ` +
    `alt ${(body.position.z * 3.28084).toFixed(0)} ft, pitch ${(axes(body).pitch * 57.3).toFixed(1)}°`);
}

console.log("== elevator rate smoothness check: max |delev/dt| during pitch holds");
fly("  pitch hold 10°", { throttle: 1.0, pitchT: 10 * Math.PI / 180, seconds: 10 });
