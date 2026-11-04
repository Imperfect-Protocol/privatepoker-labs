/** Minimal test harness: collect failures, exit non-zero if any. */

export let failures = 0;
export let checks = 0;

export function check(name: string, condition: boolean, detail = ""): void {
  checks += 1;
  if (!condition) {
    failures += 1;
    console.log(`  FAIL ${name}${detail ? ` — ${detail}` : ""}`);
  }
}

export function section(name: string): void {
  console.log(`== ${name}`);
}

export function finish(suite: string): void {
  if (failures === 0) {
    console.log(`${checks} checks passed. ${suite} OK`);
  } else {
    console.log(`${failures} checks FAILED. ${suite}`);
    process.exit(1);
  }
}
