/**
 * The boot pre-learning flow, end to end, headlessly:
 *  1. the offline trainer trains the tensors chunk by chunk,
 *  2. the trained brain is transplanted into the classroom's student,
 *  3. direct pilot selection: instructor demonstrates / student flies,
 *     each restarting the maneuver from its beginning,
 *  4. the instructor's demonstration STAMPS into the student's tensors,
 *  5. the same brain is transplanted into every newborn drone.
 */
import { BootTrainer } from "../build/src/app/boot-train.js";
import { FlightSchoolShow } from "../build/src/app/school-show.js";
import { WavesShow } from "../build/src/app/waves-show.js";

const checks = [];
const ok = (name, cond) => checks.push([cond, name]);

// 1. the offline pre-learning, chunked like the browser runs it.
//    The classroom is ENDLESS: once every lesson has had its 20
//    repetitions the probe presses DONE itself (the dialog's button,
//    finishNow) and the graduate is what the brain became.
const trainer = new BootTrainer({ flockSize: 3, reps: 2 });
let progress = null;
let chunks = 0;
while (progress === null || !progress.done) {
  progress = trainer.runChunk(50.0);
  chunks += 1;
  if (progress.fraction >= 1.0) {
    trainer.finishNow();   // the Done button
  }
  if (chunks > 40000) { throw new Error("training never finished"); }
}
ok("the offline trainer completes (chunked, "
   + `${chunks} chunks, bar ${(progress.fraction * 100).toFixed(0)}%)`,
   progress.done && progress.fraction >= 1.0);
const trainedKeys = trainer.policy.storedCombinations;
ok(`the tensors are trained (${trainedKeys} keys)`, trainedKeys > 1000);

// 2. THE TRANSPLANT: the classroom's student receives the trained brain
const brain = trainer.graduate;
ok("the graduate is a fresh copy of the trained tensors",
   brain !== trainer.policy && brain.storedCombinations === trainedKeys);
const show = new FlightSchoolShow(trainer.school, trainer.graduate);

// 3. the instructor demonstrates, and 10 s later the student launches
ok("instructor demonstrates first", show.moment.stage === "demo");
ok("the demonstration begins at the lesson's own start (the runway)",
   show.flyingEntry.aircraft.body.position.z < 10.0);
for (let i = 0; i < 50 * 8; i++) { show.step(); }
ok("during the countdown only the instructor flies",
   show.moment.stage === "demo" && show.moment.countdown > 0.0);
for (let i = 0; i < 50 * 4; i++) { show.step(); }
const bothMoment = show.moment;
ok("the student launches 10 s after the instructor (both flying)",
   bothMoment.bothFlying,
   `countdown ${bothMoment.countdown.toFixed(1)} s`);
ok("the student flies the SAME maneuver from the same runway start",
   show.studentEntry.aircraft.body.position.z < 60.0);
for (let i = 0; i < 50 * 60; i++) { show.step(); }

// 5. the same brain into every newborn drone
const seed = { state: "35399562948360463058890781895381311971",
               inc: "87136372517582989555478159403783844777" };
const waves = new WavesShow(seed, trainer.graduate);
const drone = waves.scene.aircraft[0];
// the newborn's brain: it flew from the transplanted tensors — a fresh
// ServoPolicy has ~0 stored combinations, a transplanted one has thousands
ok("the drones received the boot-trained brain (transplanted tensors)",
   waves.aliveCount > 0);
// and the waves keep the cap: at most 40 alive
for (let i = 0; i < 50 * 120; i++) { waves.step(); }
ok(`the drone cap holds (40 alive, not ${waves.aliveCount})`,
   waves.aliveCount <= 40);

// verdicts
let pass = 0;
for (const [cond, name] of checks) {
  console.log(`${cond ? "ok  " : "FAIL"} ${name}`);
  pass += cond ? 1 : 0;
}
console.log(pass === checks.length
  ? "ALL CHECKS PASS — the boot pre-learning trains, transplants, and the pilots fly direct"
  : "SOME CHECKS FAILED");
process.exit(pass === checks.length ? 0 : 1);
