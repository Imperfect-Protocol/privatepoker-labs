/**
 * Freeze the C172 carpet waves: re-record the Milestone 4 acceptance
 * scenario (seed 42, 5 waves of 8, epsilon 0.05, no school, no will) on the
 * corrected airframe, TWICE, bit-exact, then write
 * reference/c172-waves-frozen.json. Two sections, one per consumer:
 *
 *  - per_wave: the WaveRunner metrics (test/waves.test.ts gates these)
 *  - traced:   the app timeline runTracedWaves record — per-wave
 *              survivals/deaths/flock_size plus per-episode sample counts
 *              (test/app-trace.test.ts and the carpet's golden badge gate
 *              these)
 *
 * The Python golden-baseline.json waves stay on disk as the historical
 * record of the old 1.5 kg airframe.
 */
import { readFileSync, writeFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { dirname, join } from "node:path";

import { NumpyPcg64 } from "../build/src/core/rng.js";
import { Flock } from "../build/src/learning/flock.js";
import { FlightInstruments } from "../build/src/learning/instruments.js";
import { WaveRunner } from "../build/src/mission/carpet.js";
import { runTracedWaves } from "../build/src/app/timeline.js";
import { TRIM_SPEED } from "../build/src/learning/flight.js";

const here = dirname(fileURLToPath(import.meta.url));
const rngRef = JSON.parse(readFileSync(join(here, "..", "reference", "rng-ref.json"), "utf8"));
const seed42 = rngRef.seeds.find((s) => s.seed === 42);
if (seed42 === undefined) {
  console.error("seed 42 missing from rng-ref.json");
  process.exit(1);
}

const WAVES = 5;
const WAVE_SIZE = 8;
const EPSILON = 0.05;

function runAll() {
  // the WaveRunner: the library-level carpet
  const rng = new NumpyPcg64(BigInt(seed42.state), BigInt(seed42.inc));
  const runner = new WaveRunner(new Flock(), new FlightInstruments(TRIM_SPEED),
                                { waveSize: WAVE_SIZE });
  const perWave = [];
  for (let wave = 0; wave < WAVES; wave++) {
    const m = runner.runWave(wave, rng, EPSILON);
    perWave.push({
      wave: m.wave,
      survivals: m.survivals,
      deaths: m.deaths,
      flock_size: m.flockSize,
      mean_survival: m.meanSurvival,
      best_survival: m.bestSurvival,
    });
  }
  // the traced waves: the app timeline's playback source
  const traced = runTracedWaves({ state: seed42.state, inc: seed42.inc }, WAVES);
  return {
    per_wave: perWave,
    traced: {
      births: traced.flock.births,
      obituaries_heard: traced.flock.obituariesHeard,
      per_wave: traced.traces.map((t) => ({
        wave: t.wave,
        survivals: t.survivals,
        deaths: t.deaths,
        flock_size: t.flockSize,
        episode_samples: t.episodes.map((e) => e.samples.length),
      })),
    },
  };
}

const a = runAll();
const b = runAll();
if (JSON.stringify(a) !== JSON.stringify(b)) {
  console.error("NOT DETERMINISTIC: two full runs differ");
  process.exit(1);
}
for (let w = 0; w < WAVES; w++) {
  const m = a.per_wave[w];
  const t = a.traced.per_wave[w];
  const same = JSON.stringify(m.survivals) === JSON.stringify(t.survivals)
    && m.deaths === t.deaths && m.flock_size === t.flock_size;
  console.log(`wave ${w}: survivals [${m.survivals.map((s) => s.toFixed(2)).join(", ")}] ` +
              `deaths ${m.deaths}, flock ${m.flock_size}` +
              `${same ? "" : "  (traced twin DIFFERS)"}`);
}
console.log(`traced: ${a.traced.births} births, ${a.traced.obituaries_heard} obituaries heard`);

if (a.traced.births !== WAVES * WAVE_SIZE) {
  console.error(`traced births ${a.traced.births} != ${WAVES * WAVE_SIZE}`);
  process.exit(1);
}
for (let w = 0; w < WAVES; w++) {
  const m = a.per_wave[w];
  const t = a.traced.per_wave[w];
  if (JSON.stringify(m.survivals) !== JSON.stringify(t.survivals)
      || m.deaths !== t.deaths || m.flock_size !== t.flock_size) {
    console.error(`traced twin diverges from the WaveRunner on wave ${w}`);
    process.exit(1);
  }
}

const doc = {
  schema: "c172-waves-frozen/1",
  note: "TS-frozen carpet-waves reference: seed 42, 5 waves of 8, " +
        "epsilon 0.05, on the 1111 kg battle-C172 airframe (body-rate " +
        "quaternion propagation, ground-effect glue, agl-aware step). " +
        "Supersedes the Python golden-baseline.json waves scenario, which " +
        "is the historical record of the old 1.5 kg airframe. per_wave " +
        "gates the WaveRunner; traced gates the app timeline " +
        "(runTracedWaves) and the carpet's golden badge.",
  seed: 42,
  waves: WAVES,
  wave_size: WAVE_SIZE,
  epsilon: EPSILON,
  ...a,
};
writeFileSync(new URL("../reference/c172-waves-frozen.json", import.meta.url),
              JSON.stringify(doc, null, 1) + "\n");
console.log("wrote reference/c172-waves-frozen.json (bit-exact across two runs)");
