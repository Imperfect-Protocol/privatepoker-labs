/**
 * Freeze the C172 physics trajectories: re-record the classic
 * fixed-command scenarios (same starts, same commands, same lengths as the
 * Python flight_ref) on the corrected TS airframe — the 1111 kg battle-C172,
 * the body-rate quaternion propagation, and the ground-effect glue. Run
 * each scenario TWICE and assert bit-exactness, then write
 * reference/c172-physics-frozen.json. The Python flight-ref stays on disk
 * as the historical record; this frozen file is the behavioural gate.
 */
import { Vec3 } from "../build/src/core/vec.js";
import { RigidBody } from "../build/src/physics/rigid-body.js";
import { SurfacesState } from "../build/src/physics/surfaces.js";
import { readFileSync, writeFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { dirname, join } from "node:path";

const refPath = join(dirname(fileURLToPath(import.meta.url)),
                     "..", "reference", "flight-ref.json");
const old = JSON.parse(readFileSync(refPath, "utf8"));

const startFor = (name) =>
  name === "stalled_drop"
    ? { pos: new Vec3(0.0, 0.0, 50.0), vel: new Vec3(5.0, 0.0, -3.5) }
    : name === "knife_edge_slip"
      ? { pos: new Vec3(0.0, 0.0, 300.0), vel: new Vec3(15.0, 0.0, 0.0) }
      : { pos: new Vec3(0.0, 0.0, 50.0), vel: new Vec3(10.5, 0.0, 0.0) };

function recordRun(run) {
  const commands = Float64Array.from(run.commands);
  const start = startFor(run.name);
  const body = new RigidBody(start.pos, start.vel);
  const servos = new SurfacesState();
  const samples = [];
  for (let tick = 0; tick < run.ticks; tick++) {
    servos.command(commands, run.dt);
    const diag = body.step(servos, run.dt);
    if (tick % 5 === 0) {
      const q = body.attitude;
      samples.push({
        pos: [body.position.x, body.position.y, body.position.z],
        vel: [body.velocity.x, body.velocity.y, body.velocity.z],
        quat: [q.w, q.x, q.y, q.z],
        rates: [body.rateP, body.rateQ, body.rateR],
        surfaces: Array.from(servos.positions),
        diag: [diag.airspeed, diag.alpha, diag.beta, diag.qDyn,
               diag.lift, diag.drag, diag.thrust],
      });
    }
  }
  return samples;
}

const runs = [];
for (const run of old.physics_runs) {
  const a = recordRun(run);
  const b = recordRun(run);
  if (JSON.stringify(a) !== JSON.stringify(b)) {
    console.error(`NOT DETERMINISTIC: ${run.name}`);
    process.exit(1);
  }
  runs.push({ name: run.name, commands: run.commands, ticks: run.ticks,
              dt: run.dt, samples: a });
  console.log(`${run.name.padEnd(20)} ${a.length} samples, bit-exact across two runs ✓`);
}

const doc = {
  schema: "c172-physics-frozen/1",
  note: "TS-frozen physics reference: 1111 kg battle-C172 airframe, " +
        "body-rate quaternion propagation (q ⊗ ω), ground-effect glue. " +
        "Scenarios mirror the Python flight_ref starts/commands; the " +
        "Python trajectories are invalidated by the C172 redesign.",
  physics_runs: runs,
};
writeFileSync(new URL("../reference/c172-physics-frozen.json", import.meta.url),
              JSON.stringify(doc, null, 1) + "\n");
console.log("wrote reference/c172-physics-frozen.json");
