/**
 * The flock quality run: a small boot classroom (flock + reps from
 * argv, default 10 x 3) run to fraction 1.0, printing the golden
 * report — the champion's EMA score per lesson, next to the fresh
 * champion's verdicts on a couple of lessons.
 */
import { BootTrainer } from "../build/src/app/boot-train.js";

const flock = Number(process.argv[2] ?? 10);
const reps = Number(process.argv[3] ?? 3);
console.log(`flock ${flock} x ${reps} reps`);

const trainer = new BootTrainer({ flockSize: flock, reps });
let progress = null;
let chunks = 0;
while (progress === null || !progress.done) {
  progress = trainer.runChunk(200.0);
  chunks += 1;
  if (progress.fraction >= 1.0) {
    trainer.finishNow();
  }
}
console.log(`done in ${chunks} chunks`);
const golden = trainer.golden();
let total = 0.0;
for (const row of golden) {
  total += row.score;
  const goals = row.goals.map((g) =>
    `${g.fraction >= 0.999 ? "✓" : g.fraction >= 0.5 ? "±" : "✗"}`).join("");
  console.log(`${row.lesson.padEnd(24)} attempts ${String(row.attempts).padStart(2)}`
    + ` ema ${row.score.toFixed(2)}  ${goals}`);
}
console.log(`mean EMA ${(total / golden.length).toFixed(2)}`);
const keys = trainer.policy.storedCombinations;
console.log(`champion keys ${keys}`);
