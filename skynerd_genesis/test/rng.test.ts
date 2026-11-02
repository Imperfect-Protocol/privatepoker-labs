/** RNG streams vs the frozen NumPy PCG64 reference (reference/rng-ref.json). */
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { dirname, join } from "node:path";

import { NumpyPcg64 } from "../src/core/rng.js";
import { check, finish, section } from "./harness.js";

const refPath = join(dirname(fileURLToPath(import.meta.url)), "..", "..", "reference", "rng-ref.json");
const ref = JSON.parse(readFileSync(refPath, "utf8"));

for (const seed of ref.seeds) {
  section(`pcg64 seed ${seed.seed}`);
  {
    const rng = new NumpyPcg64(BigInt(seed.state), BigInt(seed.inc));
    let ok = true;
    for (let i = 0; i < seed.uniform.length; i++) {
      const got = rng.uniform();
      if (got !== seed.uniform[i]) {
        ok = false;
        check(`uniform[${i}]`, false, `got ${got} want ${seed.uniform[i]}`);
        break;
      }
    }
    check("uniform stream (400 draws) exact", ok);
  }
  {
    const rng = new NumpyPcg64(BigInt(seed.state), BigInt(seed.inc));
    let ok = true;
    for (let i = 0; i < seed.int9.length; i++) {
      const got = rng.integers(9);
      if (got !== seed.int9[i]) {
        ok = false;
        check(`int9[${i}]`, false, `got ${got} want ${seed.int9[i]}`);
        break;
      }
    }
    check("int9 stream (600 draws) exact", ok);
  }
  {
    const rng = new NumpyPcg64(BigInt(seed.state), BigInt(seed.inc));
    let ok = true;
    for (let i = 0; i < seed.interleave.pattern.length; i++) {
      const op = seed.interleave.pattern[i];
      const want = seed.interleave.values[i];
      const got = op === "u" ? rng.uniform() : rng.integers(9);
      if (got !== want) {
        ok = false;
        check(`interleave[${i}]`, false, `got ${got} want ${want}`);
        break;
      }
    }
    check("interleaved stream with half-cache carry exact", ok);
  }
  {
    const rng = new NumpyPcg64(BigInt(seed.state), BigInt(seed.inc));
    const got8 = rng.permutation(8);
    check("permutation(8)", JSON.stringify(got8) === JSON.stringify(seed.permutation.n8),
          `got ${got8}`);
  }
  {
    const rng = new NumpyPcg64(BigInt(seed.state), BigInt(seed.inc));
    const got16 = rng.permutation(16);
    check("permutation(16)", JSON.stringify(got16) === JSON.stringify(seed.permutation.n16),
          `got ${got16}`);
  }
  {
    const rng = new NumpyPcg64(BigInt(seed.state), BigInt(seed.inc));
    let ok = true;
    for (const c of seed.choice) {
      const got = rng.choice(c.n, c.k);
      if (JSON.stringify(got) !== JSON.stringify(c.values)) {
        ok = false;
        check(`choice(${c.n},${c.k})`, false, `got ${got} want ${c.values}`);
        break;
      }
    }
    check("choice streams (Floyd + shuffle) exact", ok);
  }
  {
    const rng = new NumpyPcg64(BigInt(seed.state), BigInt(seed.inc));
    let ok = true;
    for (let i = 0; i < seed.mixed.values.length; i++) {
      const op = seed.mixed.ops[i];
      const want = seed.mixed.values[i];
      let got: unknown;
      if (op.startsWith("choice")) {
        got = rng.choice(40, 6);
      } else if (op.startsWith("perm")) {
        got = rng.permutation(6);
      } else if (op === "uniform") {
        got = rng.uniform();
      } else {
        got = rng.integers(9);
      }
      if (JSON.stringify(got) !== JSON.stringify(want)) {
        ok = false;
        check(`mixed[${i}]`, false, `got ${JSON.stringify(got)} want ${JSON.stringify(want)}`);
        break;
      }
    }
    check("mixed battery (choice/perm/uniform/int9, shared cache) exact", ok);
  }
}

finish("rng.test");
