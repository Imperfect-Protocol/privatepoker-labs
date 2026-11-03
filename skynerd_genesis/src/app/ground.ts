/**
 * The world's ground, for the cockpit's 3D pipeline.
 *
 * The ground is an INFINITE STREAM of textured squares: a world-fixed
 * CHECKERBOARD sampled PER PIXEL (never filled as polygons), streaming
 * under the aircraft as it flies. Cells render only within K distance
 * (drawDistance) of the camera; beyond K the world fades into sky haze.
 *
 * THE MIP PYRAMID is defined MANUALLY: 6 levels, each level literally a
 * COPY of the base map, brightened a bit (x1.12 per level — gentle
 * steps, the top map reading ~80% of the previous 4-level pyramid's top
 * brightness) — same pattern, and each level's cells are twice the
 * size, so one square at the top level covers a bigger world area than
 * one at the bottom.
 *
 * THE LEVEL IS CHOSEN BY ALTITUDE — one map per doubling of eye height,
 * the base rung at ~125 ft. On the ground the aircraft sees the dark
 * base map; climbing cross-fades to the brighter copies (at 1000 ft it
 * is the top map), and TRILINEAR blends the two bracketing maps, so
 * mid-climb the smaller checker is seen blended with the bigger
 * brighter one. Because the cells double with the rungs, the checker
 * HOLDS ITS APPARENT SIZE as the aircraft climbs — same size, brighter.
 * NO level is ever an averaged blur: each map is sampled crisp with
 * EDGE ANTIALIASING ONLY (the box filter never exceeds one cell), so
 * the pattern never dissolves into a flat colour — the pixel footprint
 * drives only the edge antialiasing and the ANISOTROPIC taps (up to 16
 * along the footprint's major axis, each filtered at the minor width).
 *
 * The runway is painted into the same texture (asphalt repair patches,
 * threshold stripes, the dashed centre line, the side lines), its slab
 * edges box-filtered by footprint coverage and its markings faded with
 * distance.
 *
 * Flat for now: the surface is the z = 0 plane, so the ray intersection
 * is analytic. The query surface — heightAt, intersectRay, sample — is
 * deliberately the stable contract: mountains and valleys arrive later
 * as a height field (heightAt) with a marching intersectRay, and the
 * cockpit pipeline does not change.
 */

/** The palette: dark and bright fields, runway asphalt, white markings.
 *  The checker's contrast is SUBTLE — the dark and bright squares are
 *  pulled close around the same brown mean, and every level of the
 *  pyramid inherits that subtlety (each level is a copy, brightened). */
const DARK: [number, number, number] = [68, 50, 32];      // #443220
const BRIGHT: [number, number, number] = [84, 62, 40];    // #543e28
const PATCH_A: [number, number, number] = [77, 85, 95];   // #4d555f
const PATCH_B: [number, number, number] = [96, 104, 114]; // #606872
const MARKING: [number, number, number] = [232, 236, 244]; // #e8ecf4

/** One checkerboard cell at level 0, metres. */
const CELL = 12.0;

/** The manual mip pyramid: 6 levels, each a copy of the base map,
 *  brightened a bit per level (x1.12) — gentle steps: the top level
 *  reads ~80% of the old pyramid's top brightness, with two extra rungs
 *  on the altitude ladder (125 ft per doubling, to 4000 ft). */
const MIP_LEVELS = 6;
const MIP_BRIGHTEN = 1.12;

/** The altitude rung for level 1, metres (~125 ft): one map per doubling
 *  of eye height above this. On the ground and below it: the base map. */
const ALT_PER_LEVEL = 125.0 * 0.3048;

/** Anisotropic filtering's tap cap, as on GPUs (16x). */
const ANISO_MAX = 16;

/** Runway markings fade out with distance (they are thin paint; past
 *  ~2 km they are sub-pixel and would shimmer). */
const MARK_FADE_NEAR = 1000.0;
const MARK_FADE_FAR = 2000.0;

/** The runway, in world metres: a FINITE lane along +X — 1.7 km, from
 *  x = -250 to x = +1450 — with the school's takeoff roll starting near
 *  its threshold. The threshold stripes sit at the near end, the dashed
 *  centre line runs its length. The box filter's clamp
 *  (RUNWAY_FILTER_MAX) keeps the FINITE strip crisp at altitude — no
 *  kilometres-long smear past its ends. */
const RUNWAY = { near: -250.0, far: 1450.0, half: 15.0, patch: 110.0,
                 threshold: 24.0, bar: 3.5, barGap: 2.0,
                 dashFrom: 40.0, dash: 30.0, dashGap: 20.0 };

/** The runway box filter's footprint clamp: how far a pixel's coverage
 *  interval may stretch before it stops softening the slab's edges.
 *  Small enough that the FINITE strip stays crisp at altitude — no
 *  kilometres-long smear past its ends. */
const RUNWAY_FILTER_MAX = 300.0;

/** One mip level's checker: the same pattern, at its own scale and
 *  brightness. */
interface MipLevel {
  cell: number;                          // cell side, metres
  dark: [number, number, number];
  bright: [number, number, number];
  mean: [number, number, number];
}

/** The pyramid: level L copies the level-(L-1) checker at 2x cells and
 *  MIP_BRIGHTEN brighter, from the ground's own dark level 0. */
const PYRAMID: MipLevel[] = Array.from({ length: MIP_LEVELS }, (_, level) => {
  const gain = MIP_BRIGHTEN ** level;
  const dark: [number, number, number] = [
    DARK[0] * gain, DARK[1] * gain, DARK[2] * gain,
  ];
  const bright: [number, number, number] = [
    BRIGHT[0] * gain, BRIGHT[1] * gain, BRIGHT[2] * gain,
  ];
  return {
    cell: CELL * 2 ** level,
    dark, bright,
    mean: [(bright[0] + dark[0]) / 2, (bright[1] + dark[1]) / 2,
           (bright[2] + dark[2]) / 2],
  };
});

/** The world's ground surface: flat today, a height field tomorrow. */
export class Ground {
  /** Render distance K: cells beyond this are not drawn (haze takes over). */
  readonly drawDistance = 24000.0;

  /** Surface height at a world point. Flat for now; mountains and
   *  valleys override this with a height field later. */
  heightAt(x: number, y: number): number {
    void x;
    void y;
    return 0.0;
  }

  /** The first intersection of the ray (origin, direction) with the
   *  ground, as distance along the (unit) direction, or null when the
   *  ray departs. Flat ground: the analytic z = 0 plane hit. */
  intersectRay(ox: number, oy: number, oz: number,
               dx: number, dy: number, dz: number): number | null {
    void oy;
    if (dz >= -1e-9) {
      return null;                          // level or departing rays never land
    }
    const t = -oz / dz;
    return t > 0.0 ? t : null;
  }

  /** The unit square wave's triangle-wave antiderivative, period 2·cell. */
  private static ramp(x: number, cell: number): number {
    const period = 2.0 * cell;
    const r = x - period * Math.floor(x / period);
    return r <= cell ? r : period - r;
  }

  /** The square wave's mean over [a, b], in [-1, 1]. */
  private static squareMean(a: number, b: number, cell: number): number {
    const w = b - a;
    if (w < 1e-9) {
      return 0.0;
    }
    return (Ground.ramp(b, cell) - Ground.ramp(a, cell)) / w;
  }

  /** One mip level's EXACT box filter: the checker (the product of two
   *  unit square waves) filtered over an axis-aligned square footprint of
   *  side w, centred at (u, v). Returns the filtered value in [-1, 1]:
   *  +1 deep inside a bright cell, -1 inside a dark one, 0 dissolved. */
  private static checkerMip(u: number, v: number, w: number,
                            cell: number): number {
    const half = Math.max(w, 1e-6) / 2.0;
    return Ground.squareMean(u - half, u + half, cell)
         * Ground.squareMean(v - half, v + half, cell);
  }

  /** One level's colour for a filtered checker value. */
  private static levelColour(level: MipLevel,
                             value: number): [number, number, number] {
    return [
      level.mean[0] + value * (level.bright[0] - level.mean[0]),
      level.mean[1] + value * (level.bright[1] - level.mean[1]),
      level.mean[2] + value * (level.bright[2] - level.mean[2]),
    ];
  }

  /** The mip level for an eye height: one map per doubling above the
   *  base rung, clamped to the pyramid's ends. */
  private static levelOf(eyeHeight: number): number {
    return Math.max(0.0, Math.min(MIP_LEVELS - 1.0,
      Math.log2(Math.max(eyeHeight, 1e-6) / ALT_PER_LEVEL)));
  }

  /** One TRILINEAR tap of the manual pyramid at a fixed level: both
   *  bracketing maps are sampled and blended by the fractional level.
   *  Each map is a LITERAL COPY of the base checker — the box filter is
   *  CLAMPED at one cell of the map (edge antialiasing only), so the
   *  squares stay full contrast at any distance and altitude: the
   *  pattern NEVER averages into a flat colour. */
  private checkerTap(u: number, v: number, w: number,
                     level: number): [number, number, number] {
    const lo = Math.min(Math.floor(level), MIP_LEVELS - 2);
    const frac = level - lo;
    const a = Ground.levelColour(PYRAMID[lo],
      Ground.checkerMip(u, v, Math.min(w, PYRAMID[lo].cell), PYRAMID[lo].cell));
    if (frac <= 0.0) {
      return a;
    }
    const b = Ground.levelColour(PYRAMID[lo + 1],
      Ground.checkerMip(u, v, Math.min(w, PYRAMID[lo + 1].cell),
                        PYRAMID[lo + 1].cell));
    return [
      a[0] + (b[0] - a[0]) * frac,
      a[1] + (b[1] - a[1]) * frac,
      a[2] + (b[2] - a[2]) * frac,
    ];
  }

  /** The ANISOTROPIC field sample: the pixel's ground footprint is a
   *  parallelogram spanned by the two per-pixel steps; up to ANISO_MAX taps
   *  march along its MAJOR axis, each tap filtered at the MINOR axis
   *  width, and the taps average — grazing views keep their along-view
   *  detail. The mip level comes from the eye's ALTITUDE, fixed for the
   *  whole frame's ground. */
  private fieldTaps(u: number, v: number, stepXU: number, stepXV: number,
                    stepYU: number, stepYV: number,
                    level: number): [number, number, number] {
    const ex = Math.hypot(stepXU, stepXV);
    const ey = Math.hypot(stepYU, stepYV);
    if (!Number.isFinite(ex) || !Number.isFinite(ey) || ex < 1e-9 || ey < 1e-9) {
      // the horizon rim: no footprint — the current map, crisp
      return this.checkerTap(u, v, PYRAMID[MIP_LEVELS - 1].cell, level);
    }
    let dirU: number;
    let dirV: number;
    let major: number;
    let minor: number;
    if (ex >= ey) {
      major = ex;
      minor = ey;
      dirU = stepXU / ex;
      dirV = stepXV / ex;
    } else {
      major = ey;
      minor = ex;
      dirU = stepYU / ey;
      dirV = stepYV / ey;
    }
    const aniso = Math.min(ANISO_MAX, Math.max(1, Math.ceil(major / minor)));
    let r = 0.0;
    let g = 0.0;
    let b = 0.0;
    for (let j = 0; j < aniso; j++) {
      const off = ((j + 0.5) / aniso - 0.5) * major;
      const tap = this.checkerTap(u + dirU * off, v + dirV * off, minor, level);
      r += tap[0];
      g += tap[1];
      b += tap[2];
    }
    return [r / aniso, g / aniso, b / aniso];
  }

  /** The ground colour for one pixel: (u, v) is the ground point, the
   *  steps are the footprint's per-pixel ground extents (right, down),
   *  eyeHeight picks the mip map (one per altitude doubling), and
   *  distance drives the haze and the marking fade. The runway slab is
   *  box-filtered by footprint coverage (soft slab edges at range). */
  sample(u: number, v: number, stepXU: number, stepXV: number,
         stepYU: number, stepYV: number, distance: number,
         eyeHeight: number): [number, number, number] {
    const level = Ground.levelOf(eyeHeight);
    const field = this.fieldTaps(u, v, stepXU, stepXV, stepYU, stepYV, level);
    // the runway's footprint coverage: soft slab edges in both axes
    const wU = Math.min(Math.hypot(stepXU, stepXV), RUNWAY_FILTER_MAX);
    const wV = Math.min(Math.hypot(stepYU, stepYV), RUNWAY_FILTER_MAX);
    const cov = Ground.coverage(u - wU / 2, u + wU / 2, RUNWAY.near, RUNWAY.far)
                * Ground.coverage(v - wV / 2, v + wV / 2, -RUNWAY.half, RUNWAY.half);
    if (cov <= 0.0) {
      return field;
    }
    const rw = this.runwayShade(u, v, distance);
    if (cov >= 1.0) {
      return rw;
    }
    return [
      rw[0] + (field[0] - rw[0]) * (1.0 - cov),
      rw[1] + (field[1] - rw[1]) * (1.0 - cov),
      rw[2] + (field[2] - rw[2]) * (1.0 - cov),
    ];
  }

  /** Interval coverage in [0, 1]: the fraction of [a, b] inside [lo, hi]. */
  private static coverage(a: number, b: number, lo: number, hi: number): number {
    const inside = Math.min(b, hi) - Math.max(a, lo);
    if (inside <= 0.0) {
      return 0.0;
    }
    return Math.min(1.0, inside / (b - a));
  }

  /** The runway's texture at a ground point on the slab, with the paint
   *  faded by distance. */
  private runwayShade(x: number, y: number,
                     distance: number): [number, number, number] {
    const along = x - RUNWAY.near;
    const paint = distance <= MARK_FADE_NEAR ? 1.0
      : distance >= MARK_FADE_FAR ? 0.0
        : (MARK_FADE_FAR - distance) / (MARK_FADE_FAR - MARK_FADE_NEAR);
    // asphalt with alternating repair patches (110 m patches never alias)
    const patch = Math.floor(along / RUNWAY.patch) & 1;
    const base = patch === 0 ? PATCH_A : PATCH_B;
    let colour: [number, number, number] = base;
    // threshold stripes: 8 bars across the first 24 m, centre gap kept clear
    if (along <= RUNWAY.threshold) {
      const bar = Math.floor(y / RUNWAY.bar);
      const inBar = y - bar * RUNWAY.bar < RUNWAY.barGap;
      if (inBar && bar !== 0 && bar !== -1) {
        colour = MARKING;
      }
    }
    // the dashed centre line
    const centre = along - RUNWAY.dashFrom;
    if (colour === base && centre >= 0.0 && x <= RUNWAY.far - RUNWAY.dash
        && centre % (RUNWAY.dash + RUNWAY.dashGap) < RUNWAY.dash
        && Math.abs(y) <= 0.9) {
      colour = MARKING;
    }
    // the side lines
    if (colour === base && Math.abs(Math.abs(y) - RUNWAY.half) <= 0.4) {
      colour = MARKING;
    }
    if (colour !== base && paint < 1.0) {
      colour = [
        base[0] + (MARKING[0] - base[0]) * paint,
        base[1] + (MARKING[1] - base[1]) * paint,
        base[2] + (MARKING[2] - base[2]) * paint,
      ];
    }
    return colour;
  }
}
