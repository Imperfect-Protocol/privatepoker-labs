/**
 * Milestone 4 acceptance: carpet waves with distributed inheritance replay
 * EXACTLY against the TS-frozen reference (reference/c172-waves-frozen.json:
 * seed 42, 5 waves of 8, epsilon 0.05, no school, no will) on the C172
 * airframe. The Python golden-baseline.json waves are the historical record
 * of the old 1.5 kg airframe; the C172 redesign invalidates them.
 *
 * This exercises the whole inherited stack in one gate: PCG64 choice()
 * streams (Floyd + shuffle), mosaic births from living donors, full episode
 * flight with shared-RNG ordering, death screams with witness probability,
 * float32 tensor merges, and flock bookkeeping.
 */
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { dirname, join } from "node:path";

import { NumpyPcg64 } from "../src/core/rng.js";
import { Flock } from "../src/learning/flock.js";
import { FlightInstruments } from "../src/learning/instruments.js";
import { WaveRunner } from "../src/mission/carpet.js";
import { TRIM_SPEED } from "../src/learning/flight.js";
import { check, finish, section } from "./harness.js";

const here = dirname(fileURLToPath(import.meta.url));
const frozenPath = join(here, "..", "..", "reference", "c172-waves-frozen.json");
const frozen = JSON.parse(readFileSync(frozenPath, "utf8"));

section("carpet waves vs c172 frozen reference");
{
  // Seed 42's default_rng state: PCG64 seeding. Derive it the same way
  // NumPy does — SeedSequence → PCG64. Frozen in the reference docs instead:
  // capture.py recorded the scenario from `Config().replace(seed=42)`.
  // The initial state comes from the frozen rng-ref for seed 42.
  const rngRef = JSON.parse(readFileSync(join(here, "..", "..", "reference", "rng-ref.json"), "utf8"));
  const seed42 = rngRef.seeds.find((s: any) => s.seed === 42);
  check("seed 42 state available in rng-ref", seed42 !== undefined);

  const rng = new NumpyPcg64(BigInt(seed42.state), BigInt(seed42.inc));
  const instruments = new FlightInstruments(TRIM_SPEED);
  const runner = new WaveRunner(new Flock(), instruments, { waveSize: frozen.wave_size });

  check("frozen schema is the c172 waves reference",
        frozen.schema === "c172-waves-frozen/1");
  check("wave count matches", frozen.waves === 5);

  for (let wave = 0; wave < frozen.waves; wave++) {
    const metrics = runner.runWave(wave, rng, frozen.epsilon);
    const want = frozen.per_wave[wave];
    check(`wave ${wave} survivals exact`,
          JSON.stringify(metrics.survivals) === JSON.stringify(want.survivals),
          `got ${metrics.survivals} want ${want.survivals}`);
    check(`wave ${wave} deaths`, metrics.deaths === want.deaths,
          `got ${metrics.deaths} want ${want.deaths}`);
    check(`wave ${wave} flock size`, metrics.flockSize === want.flock_size,
          `got ${metrics.flockSize} want ${want.flock_size}`);
    check(`wave ${wave} index`, metrics.wave === want.wave);
  }
}

finish("waves.test");
