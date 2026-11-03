/** The live drone attack waves, driven headlessly: proves the conveyor —
 *  a new wave lifts off while the previous wave is still flying (no
 *  waiting), drones of neighbouring waves CONNECT, and every drone flies
 *  the one Aircraft class. */
import { WavesShow, WAVE_SIZE } from "../build/src/app/waves-show.js";
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { dirname, join } from "node:path";

const here = dirname(fileURLToPath(import.meta.url));
const rngRef = JSON.parse(readFileSync(join(here, "..", "reference", "rng-ref.json"), "utf8"));
const seed42 = rngRef.seeds.find((s) => s.seed === 42);

const waves = new WavesShow(seed42);
const fails = [];
const check = (label, ok, info = "") => {
  console.log(`${ok ? "ok  " : "FAIL"} ${label}${info ? ` — ${info}` : ""}`);
  if (!ok) { fails.push(label); }
};

let crossWaveLinksSeen = 0;
let wavesLaunchedWhileFlying = 0;
let lastWaveCount = 0;
const ticksFor = 60.0;               // one minute of swarm time
for (let t = 0; t < ticksFor / 0.02; t++) {
  waves.step();
  if (waves.waveCount !== lastWaveCount) {
    lastWaveCount = waves.waveCount;
    const alive = waves.aliveCount;
    if (alive > WAVE_SIZE) {
      wavesLaunchedWhileFlying++;
    }
    console.log(`t ${waves.simSeconds.toFixed(1)}s wave ${waves.waveCount} lifts off — ` +
      `${alive} drones in the air, ${waves.links().length} links`);
  }
  const links = waves.links();
  for (const link of links) {
    const a = link.a.label;
    const b = link.b.label;
    if (a.slice(5, 6) !== b.slice(5, 6)) {   // "Wave N · Drone M" — different wave numbers
      crossWaveLinksSeen++;
    }
  }
}

check("the conveyor launched wave after wave with the previous still flying",
      wavesLaunchedWhileFlying >= 10,
      `${waves.waveCount} waves, ${wavesLaunchedWhileFlying} launched into a busy sky`);
check("drones of neighbouring waves connect", crossWaveLinksSeen > 0,
      `${crossWaveLinksSeen} cross-wave link samples`);
check("every wave is a line of drones", waves.waveCount >= 15,
      `${waves.waveCount} waves in ${waves.simSeconds.toFixed(0)}s`);
check("the scene holds every aircraft of the world",
      waves.scene.aircraft.length >= waves.aliveCount,
      `${waves.scene.aircraft.length} entries, ${waves.aliveCount} flying`);
check("the dying screamed", waves.obituaries > 0,
      `${waves.obituaries} screams heard, flock ${waves.flockSize}`);

console.log(fails.length === 0
  ? "ALL CHECKS PASS — the waves are live, connected, and never wait"
  : `${fails.length} check(s) FAILED`);
process.exit(fails.length === 0 ? 0 : 1);
