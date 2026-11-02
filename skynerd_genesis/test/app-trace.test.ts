/**
 * The app's traced wave runner must produce exactly what the Python golden
 * baseline recorded (scenario "waves": seed 42, 5 waves of 8, epsilon 0.05)
 * — the browser visualises this data, so its source is held to the same
 * bit-exact gate as the WaveRunner port itself.
 */
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { dirname, join } from "node:path";

import { runTracedWaves } from "../src/app/timeline.js";
import { mphOf, vsHundredsOf, altFtOf, rpmOf,
         FULL_THROTTLE_RPM } from "../src/app/units.js";
import { check, finish, section } from "./harness.js";

const here = dirname(fileURLToPath(import.meta.url));
const rngRef = JSON.parse(readFileSync(join(here, "..", "..", "reference", "rng-ref.json"), "utf8"));
const baseline = JSON.parse(readFileSync(join(here, "..", "..", "reference", "golden-baseline.json"), "utf8"));

section("traced waves (app timeline) vs python golden baseline");
{
  const seed42 = rngRef.seeds.find((s: { seed: number }) => s.seed === 42);
  check("seed 42 in rng-ref", seed42 !== undefined);
  const golden = baseline.scenarios.waves;
  const run = runTracedWaves({ state: seed42.state, inc: seed42.inc },
                             golden.waves);
  check("flock births = wave count x wave size",
        run.flock.births === golden.waves * golden.wave_size,
        `got ${run.flock.births}`);
  for (let i = 0; i < golden.waves; i++) {
    const trace = run.traces[i];
    const want = golden.per_wave[i];
    check(`wave ${i} survivals exact`,
          JSON.stringify(trace.survivals) === JSON.stringify(want.survivals),
          `got ${trace.survivals} want ${want.survivals}`);
    check(`wave ${i} deaths`, trace.deaths === want.deaths);
    check(`wave ${i} flock size`, trace.flockSize === want.flock_size);
    check(`wave ${i} has full samples`, trace.episodes.every(
      (ep) => ep.samples.length === Math.round(ep.survival / 0.02)));
  }
  check("samples carry headings", run.traces[0].episodes.every(
    (ep) => ep.samples.every((s) => Number.isFinite(s.fwdX) && Number.isFinite(s.fwdY))));
  check("samples carry attitude and controls", run.traces[0].episodes.every(
    (ep) => ep.samples.every((s) => Number.isFinite(s.pitch) && Number.isFinite(s.bank)
      && Number.isFinite(s.speed) && Number.isFinite(s.vs)
      && Number.isFinite(s.throttle))));
  check("samples carry surface positions", run.traces[0].episodes.every(
    (ep) => ep.samples.every((s) => Number.isFinite(s.ail) && Number.isFinite(s.elev)
      && Number.isFinite(s.rud) && Math.abs(s.ail) <= 0.35 + 1e-9
      && Math.abs(s.elev) <= 0.35 + 1e-9 && Math.abs(s.rud) <= 0.35 + 1e-9)));
  section("cockpit units (panel law)");
  {
    check("cruise 10.5 m/s reads 80 mph", mphOf(10.5) === 80.0);
    check("trim throttle 0.15 reads 1000 rpm", rpmOf(0.15) === 1000.0);
    check("closed throttle reads 0 rpm", rpmOf(0.0) === 0.0);
    check("full throttle reads 2700 rpm", rpmOf(1.0) === FULL_THROTTLE_RPM);
    check("1 m/s climb reads 1.9685 x100 fpm",
          Math.abs(vsHundredsOf(1.0) - 1.968504) < 1e-6);
    check("50 m altitude reads 164.042 ft", Math.abs(altFtOf(50.0) - 164.042) < 1e-3);
  }
}

finish("app-trace.test");
