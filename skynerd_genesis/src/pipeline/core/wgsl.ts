/**
 * The shared WGSL common — the world's LAWS in shader form, shared by
 * every kernel and draw shader in the pipeline. Each function is a
 * deliberate mirror of its CPU twin (land.ts / ground.ts): hash01,
 * heightAt (hermite), the checker law (the manual mip pyramid as TWO
 * analytic evaluations + bracket blend), the biome palettes, the
 * roads and runway paint, the relief factor, the haze law. Where the
 * CPU reads a texture and the GPU reads a texture, the math in
 * between is IDENTICAL — the parity probes (plan §7) can diff against
 * the CPU reference exactly.
 */

export const WGSL_COMMON = /* wgsl */ `
// --- the per-frame uniform: every kernel and shader reads this ------
struct Frame {
  viewProj : mat4x4f,   // camera position + orientation -> clip
  camPos   : vec4f,     // xyz, w = the checker's eye LEVEL (the altitude law)
  right    : vec4f,     // xyz, w = focal length, framebuffer px
  up       : vec4f,     // xyz, w = framebuffer width, px
  fwd      : vec4f,     // xyz, w = framebuffer height, px
  params   : vec4f,     // K draw distance, tilesX, tilesY, tile px
  sunDir   : vec4f,     // the one sun's world direction
  features : vec4f,     // haze, relief, sunShadows, waterReflections
  zenith   : vec4f,     // the sky zenith rgb, w = the disc's angular size
  fog      : vec4f,     // the horizon fog rgb, w = the sun's brightness
  sunTint  : vec4f,     // the sun's tint rgb, w = the ambient light scale
  extras   : vec4f,     // x = checker, y = iris exposure, z = crash-floor gate
  grade    : vec4f,     // the hour's ground grade rgb (0..255), w = strength
};
@group(0) @binding(0) var<uniform> frame : Frame;
// the world's textures (group 0, shared by every kernel and shader):
// the height field (r32float, textureLoad only — unfilterable), the
// biome map (r8uint), and the tileable variation noise + its sampler
@group(0) @binding(1) var heightTex : texture_2d<f32>;
@group(0) @binding(2) var biomeTex : texture_2d<u32>;
@group(0) @binding(3) var noiseTex : texture_2d<f32>;
@group(0) @binding(4) var noiseSamp : sampler;

// --- the world's constants (land.ts / ground.ts / cockpit.ts) ------
const TEX   : i32 = 512;      // the map's side, texels
const TEXEL : f32 = 16.0;     // one map texel, metres
const PERIOD: f32 = 8192.0;   // the map's period, metres

const SUN_GLOW: f32 = 9.0;    // the glow's radius, in discs

const SKY_GRAD_SIN: f32 = 0.35;
// THE CRASH FLOOR: the background ends 15° UNDER the horizon — below
// it the world is black (the crashed view), never sky
const SKY_FLOOR_SIN: f32 = 0.25881904510252074;   // sin(15°)

/** THE HOUR'S GRADE: a mean-1 colour filter of the carried strength —
 *  the ground TINTS with the hour (pink sunrise, orange sunset, navy
 *  night) without darkening; noon's white grade is the identity. */
fn gradeFilter(strength: f32) -> vec3f {
  let g = frame.grade.xyz / 255.0;
  let gm = (g.x + g.y + g.z) / 3.0;
  return mix(vec3f(1.0), g / gm, strength);
}

// the checker law (ground.ts): cell 12 m at level 0, one level per
// doubling of eye height above 125 ft, x1.12 brighter per level,
// 6 levels, TWO literal evaluations blended — never a flat mean
const MIP_LEVELS   : i32 = 6;
const MIP_BRIGHTEN : f32 = 1.12;
const ALT_PER_LEVEL: f32 = 125.0 * 0.3048;
const CELL         : f32 = 12.0;

// the relief factor (ground.ts SHADE_*): flat ground keeps its
// palette exactly (factor 1.0)
const SHADE_AMBIENT: f32 = 0.78;
const SHADE_SUN    : f32 = 0.22;

// the biome palettes (ground.ts PAIRS): [dark, bright] per biome
const PALETTES = array<vec3f, 10>(
  vec3f(58.0, 86.0, 42.0), vec3f(74.0, 106.0, 56.0),   // Grass
  vec3f(68.0, 50.0, 32.0), vec3f(84.0, 62.0, 40.0),   // Dry
  vec3f(96.0, 96.0, 90.0), vec3f(112.0, 112.0, 104.0),// Rocky
  vec3f(40.0, 66.0, 34.0), vec3f(52.0, 82.0, 42.0),   // Forest
  vec3f(30.0, 62.0, 92.0), vec3f(42.0, 84.0, 122.0),  // Water
);

// the roads and the runway (ground.ts)
const ROAD_COLOUR = vec3f(74.0, 78.0, 84.0) / 255.0;
const ROAD_PATCH  = vec3f(78.0, 82.0, 88.0) / 255.0;
const ROAD_DASH   = vec3f(200.0, 205.0, 214.0) / 255.0;
const ROAD_SPACING: f32 = 4096.0;
const ROAD_LINE   : f32 = 2560.0;
const RUN_NEAR: f32 = -250.0;  const RUN_FAR: f32 = 1450.0;
const RUN_HALF: f32 = 15.0;

// The CPU version multiplies in f64 and truncates once with |0; for
// the lattice ranges we feed it (|ix|,|iy| < 2^15) every product is
// f64-exact, so the u32 per-op wrap here matches it bit for bit.
fn hash01v(ix: i32, iy: i32, salt: i32) -> f32 {
  var h = u32(ix) * 374761393u + u32(iy) * 668265263u
        + u32(salt) * 2246822519u;
  h = (h ^ (h >> 13u)) * 1274126177u;
  h = h ^ (h >> 16u);
  return f32(h) / 4294967296.0;
}

// --- the terrain's height, hermite-interpolated (land.ts) -----------
// Reads the height texture exactly as the CPU heightAt reads HEIGHT:
// texel-space - 0.5, floor, smoothstep weights, 2x2 bilinear, wrapped
// by the mask (512 is a power of two).
fn heightAt(p: vec2f) -> f32 {
  let g = p / TEXEL - vec2f(0.5);
  let i = floor(g);
  let s = g - i;
  let fx = s.x * s.x * (3.0 - 2.0 * s.x);
  let fy = s.y * s.y * (3.0 - 2.0 * s.y);
  let x0 = i32(i.x) & (TEX - 1);
  let y0 = i32(i.y) & (TEX - 1);
  let x1 = (x0 + 1) & (TEX - 1);
  let y1 = (y0 + 1) & (TEX - 1);
  let h00 = textureLoad(heightTex, vec2<i32>(x0, y0), 0).r;
  let h10 = textureLoad(heightTex, vec2<i32>(x1, y0), 0).r;
  let h01 = textureLoad(heightTex, vec2<i32>(x0, y1), 0).r;
  let h11 = textureLoad(heightTex, vec2<i32>(x1, y1), 0).r;
  let a = h00 + (h10 - h00) * fx;
  let b = h01 + (h11 - h01) * fx;
  return a + (b - a) * fy;
}

// the surface normal, central differences at one texel (land.ts)
fn normalAt(p: vec2f) -> vec3f {
  let hx = heightAt(p + vec2f(TEXEL, 0.0)) - heightAt(p - vec2f(TEXEL, 0.0));
  let hy = heightAt(p + vec2f(0.0, TEXEL)) - heightAt(p - vec2f(0.0, TEXEL));
  let n = vec3f(-hx / (2.0 * TEXEL), -hy / (2.0 * TEXEL), 1.0);
  return normalize(n);
}

// --- the checker law (ground.ts), the manual mip pyramid ------------
// The unit square wave's triangle-wave antiderivative and the EXACT
// box filter over a footprint w — the CPU's proven filter, in WGSL.
fn checkerRamp(x: f32, cell: f32) -> f32 {
  let period = 2.0 * cell;
  let r = x - period * floor(x / period);
  return select(period - r, r, r <= cell);
}

fn squareMean(a: f32, b: f32, cell: f32) -> f32 {
  let w = b - a;
  if (w < 1e-9) {
    return 0.0;
  }
  return (checkerRamp(b, cell) - checkerRamp(a, cell)) / w;
}

fn checkerMip(u: f32, v: f32, w: f32, cell: f32) -> f32 {
  let half = max(w, 1e-6) * 0.5;
  return squareMean(u - half, u + half, cell)
       * squareMean(v - half, v + half, cell);
}
`;

export const WGSL_CHECKER = /* wgsl */ `
struct LevelPair { dark: vec3f, bright: vec3f, mean: vec3f, cell: f32 };

// one level's palette: the biome's pair brightened by the level's gain
fn levelPair(level: i32, biome: i32) -> LevelPair {
  let gain = pow(MIP_BRIGHTEN, f32(level));
  let d = min(PALETTES[biome * 2] * gain, vec3f(255.0)) / 255.0;
  let b = min(PALETTES[biome * 2 + 1] * gain, vec3f(255.0)) / 255.0;
  return LevelPair(d, b, (d + b) * 0.5, CELL * pow(2.0, f32(level)));
}

// the altitude law's eye level for a given eye height (ground.ts)
fn levelOfHeight(eyeHeight: f32) -> f32 {
  return clamp(log2(max(eyeHeight, 1e-6) / ALT_PER_LEVEL), 0.0,
               f32(MIP_LEVELS - 1));
}

// the full checker colour: the two bracketing levels, each a LITERAL
// evaluation (crisp squares, edge antialiasing only — never a flat
// mean), blended by the fractional level. w = the pixel's footprint.
fn checkerColour(u: f32, v: f32, w: f32, biome: i32, level: f32) -> vec3f {
  let lo = min(i32(floor(level)), MIP_LEVELS - 2);
  let frac = max(0.0, level - f32(lo));
  let a = levelPair(lo, biome);
  // the checker's OFF state (the toolbar toggle): each level's flat
  // mean — the same pyramid, no phase; a conditional on the uniform
  var ca = a.mean + (a.bright - a.mean) * 0.5;
  if (frame.extras.x > 0.5) {
    ca = a.mean + checkerMip(u, v, min(w, a.cell), a.cell)
         * (a.bright - a.mean);
  }
  if (frac <= 0.0) {
    return ca;
  }
  let b = levelPair(lo + 1, biome);
  var cb = b.mean + (b.bright - b.mean) * 0.5;
  if (frame.extras.x > 0.5) {
    cb = b.mean + checkerMip(u, v, min(w, b.cell), b.cell)
         * (b.bright - b.mean);
  }
  return mix(ca, cb, frac);
}

// --- the roads and the runway (ground.ts), footprint-filtered ------
fn foldDist(v: f32, spacing: f32) -> f32 {
  let d = v - spacing * floor(v / spacing);
  return select(d, d - spacing, d > spacing * 0.5);
}

// the x/y family distances to the nearest road centreline
fn roadDistances(p: vec2f) -> vec2f {
  return vec2f(foldDist(p.x - ROAD_LINE, ROAD_SPACING),
               foldDist(p.y - ROAD_LINE, ROAD_SPACING));
}

// interval coverage in [0, 1] (ground.ts coverage)
fn intervalCoverage(a: f32, b: f32, lo: f32, hi: f32) -> f32 {
  let inside = min(b, hi) - max(a, lo);
  if (inside <= 0.0) {
    return 0.0;
  }
  return min(1.0, inside / (b - a));
}

// the runway slab's paint (ground.ts runwayShade, distance-faded)
fn runwayShade(x: f32, y: f32, distance: f32) -> vec3f {
  let along = x - RUN_NEAR;
  var paint = (2000.0 - distance) / 1000.0;
  paint = select(paint, 0.0, distance >= 2000.0);
  paint = select(paint, 1.0, distance <= 1000.0);
  let base = select(vec3f(96.0, 104.0, 114.0), vec3f(77.0, 85.0, 95.0),
                    (i32(floor(along / 110.0)) & 1) == 0);
  var colour = base;
  if (along <= 24.0) {
    // threshold stripes: 3.5 m bars with 2 m gaps, centre kept clear
    let bar = floor(y / 3.5);
    if (y - bar * 3.5 < 2.0 && bar != 0.0 && bar != -1.0) {
      colour = vec3f(232.0, 236.0, 244.0);
    }
  }
  let centre = along - 40.0;
  if (all(colour == base) && centre >= 0.0 && x <= RUN_FAR - 30.0
      && (centre % 50.0) < 30.0 && abs(y) <= 0.9) {
    colour = vec3f(232.0, 236.0, 244.0);
  }
  if (all(colour == base) && abs(abs(y) - RUN_HALF) <= 0.4) {
    colour = vec3f(232.0, 236.0, 244.0);
  }
  if (!all(colour == base) && paint < 1.0) {
    colour = mix(base, vec3f(232.0, 236.0, 244.0), paint);
  }
  return colour / 255.0;
}

// the full ground colour for one fragment: checker + roads + runway,
// all footprint-filtered — mirrors ground.ts sample() order
fn groundColour(p: vec2f, wU: f32, wV: f32, distance: f32,
                biome: i32, level: f32) -> vec3f {
  var colour = checkerColour(p.x, p.y, max(wU, wV), biome, level);
  // the roads: asphalt strips, box-filtered by coverage
  let d = roadDistances(p);
  let wUf = min(wU, 200.0);
  let wVf = min(wV, 200.0);
  let covX = intervalCoverage(p.x - wUf * 0.5, p.x + wUf * 0.5,
                              p.x - d.x - 4.0, p.x - d.x + 4.0);
  let covY = intervalCoverage(p.y - wVf * 0.5, p.y + wVf * 0.5,
                              p.y - d.y - 4.0, p.y - d.y + 4.0);
  let cov = max(covX, covY);
  if (cov > 0.0 && biome != 4) {
    let alongX = covX >= covY;
    let along = select(p.x, p.y, alongX);
    let lateral = select(abs(d.y), abs(d.x), alongX);
    let paving = along - 40.0 * floor(along / 40.0);
    var road = select(ROAD_PATCH, ROAD_COLOUR, paving < 20.0);
    let phase = along - 7.0 * floor(along / 7.0);
    if (lateral < 0.35 && phase < 3.0) {
      var paint = (800.0 - distance) / 500.0;
      paint = select(paint, 0.0, distance >= 800.0);
      paint = select(paint, 1.0, distance <= 300.0);
      road = mix(road, ROAD_DASH, paint);
    }
    colour = mix(colour, road, cov);
  }
  // the runway slab: painted over everything, soft edges by coverage
  let covR = intervalCoverage(p.x - min(wU, 300.0) * 0.5,
                              p.x + min(wU, 300.0) * 0.5, RUN_NEAR, RUN_FAR)
           * intervalCoverage(p.y - min(wV, 300.0) * 0.5,
                              p.y + min(wV, 300.0) * 0.5,
                              -RUN_HALF, RUN_HALF);
  if (covR > 0.0) {
    colour = mix(colour, runwayShade(p.x, p.y, distance), covR);
  }
  return colour;
}
`;

/** The CPU twin of the WGSL hash01v — for one-off host-side rolls
 *  (the variation texture's dither). Bit-identical to land.ts's
 *  hash01 for the lattice ranges this pipeline feeds it. */
export function hash01v(ix: number, iy: number, salt: number): number {
  let h = (ix * 374761393 + iy * 668265263 + salt * 2246822519) | 0;
  h = Math.imul(h ^ (h >>> 13), 1274126177);
  h ^= h >>> 16;
  return (h >>> 0) / 4294967296;
}
