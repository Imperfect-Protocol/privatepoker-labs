/**
 * THE ATTACK MISSION — logic probe: pairs lift off from the runway two
 * abreast and the next pair rolls five seconds behind; each drone
 * flies the flank-mirrored chain (split ±45°, the level moment at
 * 3000 ft, the attack ∓90° down to 500 ft, the rally ±45° up to
 * 1500 ft, the spin own side, recover), and then THE WILL DECIDES —
 * the drone's own brain takes the controls in free flight. The
 * SingularityTensor keeps the plurality super-goal and pulls
 * fallen-behind drones forward.
 */
import { MissionFlight, missionPlan, HeadingMark }
  from "../build/src/learning/mission.js";
import { SingularityTensor }
  from "../build/src/learning/singularity.js";
import { FlightInstruments } from "../build/src/learning/instruments.js";
import { ServoPolicy } from "../build/src/learning/servo-tensor.js";
import { NumpyPcg64 } from "../build/src/core/rng.js";
import { TRIM_SPEED } from "../build/src/learning/flight.js";
import { SYLLABUS } from "../build/src/learning/school.js";
import { RUNWAY } from "../build/src/app/land.js";
import { Vec3 } from "../build/src/core/vec.js";

const DT = 1.0 / 50.0;
const DEG = 180.0 / Math.PI;
const FT = 3.28084;
const instruments = new FlightInstruments(TRIM_SPEED);
let failures = 0;
const check = (ok, label) => {
  console.log(`${ok ? "  ok  " : "  FAIL"}  ${label}`);
  if (!ok) failures += 1;
};

// --- the mission plan itself ----------------------------------------------------

const left = missionPlan(-1.0, new HeadingMark());
check(left.phases.length === 28, `the plan has 28 phases (${left.phases.length})`);
check(left.tokens[0] === "takeoff" && left.tokens[3] === "split"
      && left.tokens[6] === "flank" && left.tokens[10] === "attack"
      && left.tokens[14] === "climb" && left.tokens[19] === "spin"
      && left.tokens[27] === "will",
      "tokens sit on the legs (takeoff/split/flank/attack/climb/spin/will)");

// --- one full sortie, both seats of the pair -------------------------------------

const seat = (flankSign) =>
  new MissionFlight(new ServoPolicy(), instruments, 0,
    new NumpyPcg64(1007n, 87136372517582989555478159403783844777n), {
      flankSign,
      // the runway seat: the centre of the lane's own half, at the
      // same threshold the conveyor launches from
      startPosition: new Vec3(RUNWAY.near + 50.0,
        flankSign * RUNWAY.half * 0.5, SYLLABUS.runwayM),
      mode: "both-suggest",        // the scenario's default: Pair: Instructor
    });

const headingOf = (flight) => {
  const [forward] = flight.aircraft.body.bodyAxes;
  return Math.atan2(forward.y, forward.x);
};

const sorties = [];
for (const flankSign of [-1.0, 1.0]) {
  const flight = seat(flankSign);
  const record = { flight, flankSign, rotated: null, earlyYoke: false,
    heading0: null, splitHeading: null, flankAlt: null,
    attackHeading: null, attackAlt: null, spinRateR: null,
    rallyHeading: null, rallyAlt: null, enteredWill: false,
    maxR: 0.0, tokens: [], lastToken: "",
    startX: flight.aircraft.body.position.x,
    startY: flight.aircraft.body.position.y,
    lastHeading: null, takeoffTurn: 0.0, splitTurn: 0.0,
    flankDev: 0.0 };
  sorties.push(record);
  const SORTIE_S = 800.0;
  for (let t = 0; t < Math.round(SORTIE_S / DT); t += 1) {
    if (!flight.step()) break;
    const body = flight.aircraft.body;
    const mph = body.velocity.length() * 2.23694;
    const token = flight.missionToken;
    // the departure's own turns: the takeoff rolls straight (no
    // circles), the split peels own side, the flank climb-out holds
    const h = headingOf(flight);
    if (record.lastHeading !== null) {
      const dh = Math.atan2(Math.sin(h - record.lastHeading),
                            Math.cos(h - record.lastHeading));
      if (token === "takeoff") record.takeoffTurn += dh;
      else if (token === "split") record.splitTurn += dh;
    }
    record.lastHeading = h;
    if (token !== record.lastToken) {
      record.tokens.push(`${token}@${(t * DT).toFixed(0)}s`);
      record.lastToken = token;
    }
    if (token === "takeoff") {
      if (body.deckPitch > 0.07 && record.rotated === null) record.rotated = mph;
      if (mph < 55.0 && Math.abs(body.deckPitch) > 0.02) record.earlyYoke = true;
    }
    if (record.heading0 === null && token === "split") {
      record.heading0 = headingOf(flight);
    }
    if (token === "flank") {
      record.splitHeading ??= h;
      // the level moment: the altitude at the flank leg's END (the
      // token exits to the attack at 3000 ft)
      record.flankAlt = body.position.z;
      record.flankDev = Math.max(record.flankDev,
        Math.abs(Math.atan2(Math.sin(h - record.splitHeading),
                            Math.cos(h - record.splitHeading))));
    }
    if (token === "climb") {
      record.attackHeading ??= headingOf(flight);
      record.attackAlt ??= body.position.z;
    }
    if (token === "spin") {
      record.rallyHeading ??= headingOf(flight);
      record.rallyAlt ??= body.position.z;
      // the spin's rotation sign — at the PEAK of the developed spin
      // (the leg's first tick is still the stall entry: no rotation).
      if (Math.abs(body.rateR) > Math.abs(record.spinRateR ?? 0.0)) {
        record.spinRateR = body.rateR;
      }
    }
    if (token === "will") record.enteredWill = true;
    record.maxR = Math.max(record.maxR,
      Math.hypot(body.position.x, body.position.y));
    if (record.enteredWill && t * DT > 700.0) break;   // will seen; enough
  }
}

for (const [name, r] of [["left", sorties[0]], ["right", sorties[1]]]) {
  const wrap = (a) => Math.atan2(Math.sin(a), Math.cos(a));
  const splitDelta = wrap((r.splitHeading ?? 0) - (r.heading0 ?? 0)) * DEG;
  const attackDelta = wrap((r.attackHeading ?? 0) - (r.heading0 ?? 0)) * DEG;
  const rallyDelta = wrap((r.rallyHeading ?? 0) - (r.heading0 ?? 0)) * DEG;
  console.log(`${name} sortie: ${r.tokens.join(" -> ")}`);
  console.log(`  rotate ${r.rotated?.toFixed(0)} mph, split ${splitDelta.toFixed(1)} deg ` +
    `@ ${(r.flankAlt * FT).toFixed(0)} ft, attack ${attackDelta.toFixed(1)} deg ` +
    `@ ${(r.attackAlt * FT).toFixed(0)} ft, rally ${rallyDelta.toFixed(1)} deg ` +
    `@ ${(r.rallyAlt * FT).toFixed(0)} ft, spin rateR ${(r.spinRateR ?? 0).toFixed(2)}, ` +
    `will=${r.enteredWill}, max radius ${r.maxR.toFixed(0)} m`);
  check(r.rotated !== null && r.rotated >= 55.0 && r.rotated <= 75.0,
        `${name}: rotates around 60 mph (${r.rotated?.toFixed(1)})`);
  check(!r.earlyYoke, `${name}: no yoke pull before rotation speed`);
  const s = r.flankSign;
  check(r.startX >= RUNWAY.near && r.startX <= RUNWAY.far
        && Math.abs(r.startY - s * RUNWAY.half * 0.5) <= 0.01,
        `${name}: starts ON the runway at the centre of its half ` +
        `(${r.startX.toFixed(0)}, ${r.startY.toFixed(1)})`);
  check(Math.abs(r.takeoffTurn * DEG) <= 25.0,
        `${name}: the takeoff rolls and climbs straight, no circles ` +
        `(${(r.takeoffTurn * DEG).toFixed(1)} deg of turn)`);
  check(r.splitTurn * s * DEG >= 25.0 && r.splitTurn * s * DEG <= 65.0,
        `${name}: the split peels own side onto HSI ` +
        `${s < 0 ? "-" : "+"}45 (${(r.splitTurn * DEG).toFixed(1)} deg)`);
  check(r.flankDev * DEG <= 12.0,
        `${name}: the flank climb-out holds the flank heading ` +
        `(max ${(r.flankDev * DEG).toFixed(1)} deg off)`);
  check(Math.abs(splitDelta - s * 45.0) <= 8.0,
        `${name}: the split lands on HSI ${s < 0 ? "-" : "+"}45 (${splitDelta.toFixed(1)})`);
  check(Math.abs((r.flankAlt * FT) - 3000.0) <= 200.0,
        `${name}: the level moment at 3000 ft (${(r.flankAlt * FT).toFixed(0)} ft)`);
  check(Math.abs(attackDelta - (-s * 90.0)) <= 10.0,
        `${name}: the attack turn lands on HSI ${-s < 0 ? "-" : "+"}90 (${attackDelta.toFixed(1)})`);
  check(Math.abs((r.attackAlt * FT) - 500.0) <= 150.0,
        `${name}: the attack ends at 500 ft (${(r.attackAlt * FT).toFixed(0)} ft)`);
  check(Math.abs(rallyDelta - s * 45.0) <= 10.0,
        `${name}: the rally returns onto the flank heading HSI ` +
        `${s < 0 ? "-" : "+"}45 (${rallyDelta.toFixed(1)})`);
  check(Math.abs((r.rallyAlt * FT) - 1500.0) <= 250.0,
        `${name}: the rally reaches 1500 ft (${(r.rallyAlt * FT).toFixed(0)} ft)`);
  check((r.spinRateR ?? 0) * s > 0.05,
        `${name}: the spin is own-side (rateR ${(r.spinRateR ?? 0).toFixed(2)})`);
  check(r.enteredWill, `${name}: the mission ends in the will phase`);
  check(r.maxR < 9000.0, `${name}: stayed inside the area (${r.maxR.toFixed(0)} m)`);
}

// --- the will phase: the drone's own brain at the controls ------------------------

const willFlight = sorties[0].flight;
check(willFlight.controlsMode === "ai",
      `the will phase reports the AI at the controls (${willFlight.controlsMode})`);
check(willFlight.missionToken === "will", "the will phase reports the will token");
willFlight.conform("attack");
check(willFlight.missionToken === "will",
      "a will-phase drone is beyond the singularity's pull");

// --- the pair conveyor: two at once, the next five seconds behind ----------------

const { WavesShow } = await import("../build/src/app/waves-show.js");
const conveyor = new WavesShow({
  state: "35399562948360463058890781895381311971",
  inc: "87136372517582989555478159403783844777",
});
const firstSeats = conveyor.drones
  .map((d) => d.flight.aircraft.body.position);
check(firstSeats.length === 2
      && firstSeats.every((p) => p.x >= RUNWAY.near && p.x <= RUNWAY.far
        && (Math.abs(p.y - RUNWAY.half * 0.5) < 0.01
            || Math.abs(p.y + RUNWAY.half * 0.5) < 0.01)),
      "the conveyor's first pair lines up ON the runway, each seat at " +
      "the centre of its half");
const stepS = (s) => { for (let i = 0; i < Math.round(s / DT); i += 1) conveyor.step(); };
check(conveyor.aliveCount === 2 && conveyor.pairCount === 1,
      `the first pair lifts off together (${conveyor.aliveCount} drones)`);
stepS(4.9);
check(conveyor.aliveCount === 2 && conveyor.pairCount === 1,
      `4.9 s: still just the first pair (${conveyor.pairCount})`);
stepS(0.2);
check(conveyor.aliveCount === 4 && conveyor.pairCount === 2,
      `5.1 s: the second pair rolls five seconds behind (${conveyor.pairCount})`);
stepS(5.0);
check(conveyor.pairCount === 3,
      `10 s: the third pair (${conveyor.pairCount})`);
const labels = conveyor.scene.aircraft.map((e) => e.label).sort();
check(labels[0] === "Pair 1 \u00b7 left" && labels[1] === "Pair 1 \u00b7 right"
      && labels.includes("Pair 3 \u00b7 left"),
      `labels are pairs (${labels.slice(0, 2).join(" / ")})`);
check(conveyor.drones.every((d) => d.flight.controlsMode === "both-suggest"),
      "every drone launches under the Pair: Instructor default");

// --- the SingularityTensor round ---------------------------------------------------

const singularity = new SingularityTensor();
const mk = (token) => ({ policy: new ServoPolicy(), token });
const members = [mk("split"), mk("split"), mk("attack"),
                 mk("will"), mk("will")];
// a full mesh among the first three; the will-phase pair isolated
const mesh = [[0, 1], [1, 2], [2, 0], [3, 4]];
const rng = new NumpyPcg64(1007n, 87136372517582989555478159403783844777n);
singularity.updateRound(members, mesh, rng);
check(singularity.superGoal === "split",
      `the super-goal is the mission plurality (${singularity.superGoal})`);
singularity.updateRound(members, mesh, rng);
check(singularity.rounds === 2, `rounds counted (${singularity.rounds})`);
const lone = new SingularityTensor();
lone.updateRound([mk("attack")], [], rng);
check(lone.superGoal === "takeoff",
      `an isolated drone is a lone brain (super-goal untouched: ${lone.superGoal})`);

// --- conform: the super-goal pulls a fallen-behind drone forward -------------------

const behind = seat(-1.0);
// fly to the split's leg entry (the straight climb before the peel)
let atSplit = false;
for (let t = 0; t < Math.round(500.0 / DT) && !atSplit; t += 1) {
  if (!behind.step()) break;
  atSplit = behind.missionToken === "split";
}
check(atSplit, "the sortie reached the split");
behind.conform("attack");
check(behind.missionToken === "attack", "a boundary-sitting drone conforms forward");
behind.conform("split");
check(behind.missionToken === "attack", "conform never moves backward");
const runwayDrone = seat(-1.0);
runwayDrone.conform("attack");
check(runwayDrone.missionToken === "takeoff",
      "a runway drone is never pulled off its takeoff");

console.log(failures === 0 ? "MISSION PROBE OK" : `MISSION PROBE: ${failures} FAILURES`);
process.exit(failures === 0 ? 0 : 1);
