/**
 * Browser check of the second-generation drone attack: the continuous
 * pair conveyor (a pair abreast lifts off, the next pair five seconds
 * behind), the flank-mirrored mission, and the will phase — headless
 * Chromium against the dev server on :8765.
 */
import { chromium } from "/Users/sadhbh/.npm/_npx/e41f203b7505f1fb/node_modules/playwright/index.mjs";

const OK = "✓";
let failures = 0;
const check = (ok, label) => {
  console.log(`  ${ok ? OK : "✗ FAIL"}  ${label}`);
  if (!ok) failures += 1;
};

const browser = await chromium.launch();
const page = await browser.newPage();
const errors = [];
page.on("pageerror", (e) => errors.push(String(e)));
page.on("console", (m) => {
  if (m.type() === "error") errors.push(m.text());
});

await page.goto("http://localhost:8765/?boot=skip");
await page.waitForTimeout(500);

// select the drone attack scenario
const scenario = page.locator("#scenario");
const options = await scenario.locator("option").allTextContents();
const wavesLabel = options.find((o) => /drone attack/i.test(o));
check(wavesLabel !== undefined,
      `the scenario dropdown offers the drone attack (${options.length} options)`);
await scenario.selectOption({ label: wavesLabel });
await page.waitForTimeout(1000);

// the controls default: Pair: Instructor
const controls = await page.locator("#controls").textContent();
check(/Pair: Instructor/.test(controls),
      `the controls default to Pair: Instructor (${controls.trim()})`);

// 10x speed, then let the conveyor roll
await page.locator("#speed").selectOption("10");
const pairsAfter = async (wallMs) => {
  await page.waitForTimeout(wallMs);
  const n = await page.locator("#narration").textContent();
  const pairs = Number((n.match(/pair (\d+)/) || [])[1] ?? 0);
  const flying = Number((n.match(/(\d+) flying/) || [])[1] ?? 0);
  return { n, pairs, flying };
};
let s = await pairsAfter(4000);
console.log(`  narration: ${s.n}`);
check(s.pairs >= 3, `pairs keep rolling off the runway (pair ${s.pairs})`);
check(/lifted off/.test(s.n), "the narration reports lift-offs");
check(/the will decides/.test(s.n), "the narration names the will phase");
const firstFlying = s.flying;

s = await pairsAfter(6000);
console.log(`  narration: ${s.n}`);
check(s.pairs >= 6, `the conveyor keeps sending pairs (pair ${s.pairs})`);
check(s.flying >= firstFlying, `drones stay alive through the mission (${s.flying} flying)`);

// the wiki describes the new mission
const wiki = await page.locator("#tabWiki").textContent();
check(/HSI/.test(wiki), "the wiki names the HSI legs");
check(/500\s*ft/i.test(wiki), "the wiki names the 500 ft attack floor");
check(/will/i.test(wiki), "the wiki names the will phase");

const status = await page.locator("#status").textContent();
console.log(`  status: ${status.slice(0, 300)}`);

check(errors.length === 0, `no console/page errors (${errors.length})`);
if (errors.length > 0) console.log(errors.slice(0, 5).join("\n"));

await browser.close();
console.log(failures === 0
  ? "BROWSER CHECK OK"
  : `BROWSER CHECK: ${failures} FAILURES`);
process.exit(failures === 0 ? 0 : 1);
