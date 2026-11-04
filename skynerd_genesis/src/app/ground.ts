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
 * steps) — same pattern, and each level's cells are twice the size, so
 * one square at the top level covers a bigger world area than one at
 * the bottom.
 *
 * THE LEVEL IS CHOSEN BY ALTITUDE — one map per doubling of eye height,
 * the base rung at ~125 ft. On the ground the aircraft sees the dark
 * base map; climbing cross-fades to the brighter copies, and TRILINEAR
 * blends the two bracketing maps, so mid-climb the smaller checker is
 * seen blended with the bigger brighter one. Because the cells double
 * with the rungs, the checker HOLDS ITS APPARENT SIZE as the aircraft
 * climbs — same size, brighter. NO level is ever an averaged blur: each
 * map is sampled crisp with EDGE ANTIALIASING ONLY (the box filter
 * never exceeds one cell), so the pattern never dissolves into a flat
 * colour — the pixel footprint drives only the edge antialiasing and
 * the ANISOTROPIC taps (up to 16 along the footprint's major axis,
 * each filtered at the minor width).
 *
 * THE LANDSCAPE: the LandUse layer (src/app/land.ts) plans the world —
 * a procedural 512x512 land-use texture (16 m texels, tiling every 8 km)
 * that decides what each square IS: grass, dry, rock, forest, or WATER
 * (lakes and rivers). Each biome owns its own base palette pair (cloned
 * up the pyramid with the same x1.12 brightening), the FOREST and ROCK
 * texels carry a mottled detail overlay (the canopy and scree texture)
 * that fades out with the pixel footprint, the ROADS are painted as
 * asphalt strips along the map's grid lines (ending at every shore),
 * and the runway is painted over everything with its markings, slab
 * edges box-filtered by coverage.
 *
 * Flat for now: the surface is the z = 0 plane, so the ray intersection
 * is analytic. The query surface — heightAt, intersectRay, sample — is
 * deliberately the stable contract: mountains and valleys arrive later
 * as a height field (heightAt) with a marching intersectRay, and the
 * cockpit pipeline does not change.
 */
import { Biome, FIELD, FIELD_ELEV, LandUse, MAXMIP_LEVELS, RUNWAY, SUN_DIR, TERRAIN_MAX } from "./land.js";

type RGB = readonly [number, number, number];

/** The base checker pair per biome — SUBTLE contrasts around each biome's
 *  mean: the grassy green default, the dry yellow-brown, the grey rock,
 *  the deep forest, the lake water. */
const PAIRS: Readonly<Record<Biome, readonly [RGB, RGB]>> = {
  [Biome.Grass]: [[58, 86, 42], [74, 106, 56]],
  [Biome.Dry]: [[68, 50, 32], [84, 62, 40]],
  [Biome.Rocky]: [[96, 96, 90], [112, 112, 104]],
  [Biome.Forest]: [[40, 66, 34], [52, 82, 42]],
  [Biome.Water]: [[30, 62, 92], [42, 84, 122]],
};

/** The manual mip pyramid: 6 levels, each a copy of the base map,
 *  brightened a bit per level (x1.12) — gentle steps. */
const MIP_LEVELS = 6;
const MIP_BRIGHTEN = 1.12;

/** The altitude rung for level 1, metres (~125 ft): one map per doubling
 *  of eye height above this. On the ground and below it: the base map. */
const ALT_PER_LEVEL = 125.0 * 0.3048;

/** One checkerboard cell at level 0, metres. */
const CELL = 12.0;

/** The mottled detail overlay on the FOREST (canopy) and ROCKY (scree)
 *  tiles: a second, finer checker that only DARKENS, fading out as the
 *  pixel footprint grows (the texture detail from altitude). */
const MOTTLE_CELL = 15.0;
const MOTTLE_AMOUNT = 0.45;
const MOTTLE_FADE_NEAR = 8.0;      // footprint, m — full detail below
const MOTTLE_FADE_FAR = 58.0;      // footprint, m — fully faded beyond
const MOTTLE_DARK: Readonly<Partial<Record<Biome, RGB>>> = {
  [Biome.Forest]: [26, 48, 24],
  [Biome.Rocky]: [78, 78, 74],
};

/** The roads: asphalt strips along the land-use grid lines, with faint
 *  length patches and a white centre dash that fades with distance. */
const ROAD_COLOUR: RGB = [74, 78, 84];
const ROAD_PATCH: RGB = [78, 82, 88];
const ROAD_DASH: RGB = [200, 205, 214];
const ROAD_DASH_HALF = 0.35;       // centre dash halfwidth, m
const ROAD_DASH_LEN = 3.0;         // dash, m
const ROAD_GAP_LEN = 4.0;          // gap, m
const ROAD_PATCH_LEN = 40.0;       // length patches, m
const ROAD_FADE_NEAR = 300.0;      // dash fully painted below, m
const ROAD_FADE_FAR = 800.0;       // dash fully faded beyond, m

/** Runway markings fade out with distance (they are thin paint). */
const MARK_FADE_NEAR = 1000.0;
const MARK_FADE_FAR = 2000.0;

/** The runway box filter's footprint clamp: how far a pixel's coverage
 *  interval may stretch before it stops softening the slab's edges. */
const RUNWAY_FILTER_MAX = 300.0;

/** Anisotropic filtering's tap cap, as on GPUs (16x). */
const ANISO_MAX = 16;

/** The terrain's RELIEF SHADING: the surface normal against the sun,
 *  NORMALISED so flat ground keeps exactly its palette colour (factor
 *  1.0) — sun-facing slopes brighten a little, shaded slopes darken a
 *  little, the approved look untouched on the plain. */
const SHADE_AMBIENT = 0.78;
const SHADE_SUN = 0.22;
const SHADE_FLAT = SHADE_AMBIENT + SHADE_SUN * SUN_DIR[2];

/** One mip level's palette: the same checker, at its own scale and
 *  brightness. */
interface MipLevel {
  cell: number;                    // cell side, metres
  dark: RGB;
  bright: RGB;
  mean: RGB;
}

/** Clone one biome's base pair up the pyramid: each level is the same
 *  map at twice the cells, brightened a bit. */
function pyramidFor(dark: RGB, bright: RGB): readonly MipLevel[] {
  return Array.from({ length: MIP_LEVELS }, (_, level) => {
    const gain = MIP_BRIGHTEN ** level;
    const d: RGB = [Math.min(255, dark[0] * gain),
                    Math.min(255, dark[1] * gain),
                    Math.min(255, dark[2] * gain)];
    const b: RGB = [Math.min(255, bright[0] * gain),
                    Math.min(255, bright[1] * gain),
                    Math.min(255, bright[2] * gain)];
    return {
      cell: CELL * 2 ** level,
      dark: d, bright: b,
      mean: [(d[0] + b[0]) / 2, (d[1] + b[1]) / 2, (d[2] + b[2]) / 2],
    };
  });
}

const PYRAMIDS: Readonly<Record<Biome, readonly MipLevel[]>> = {
  [Biome.Grass]: pyramidFor(PAIRS[Biome.Grass][0], PAIRS[Biome.Grass][1]),
  [Biome.Dry]: pyramidFor(PAIRS[Biome.Dry][0], PAIRS[Biome.Dry][1]),
  [Biome.Rocky]: pyramidFor(PAIRS[Biome.Rocky][0], PAIRS[Biome.Rocky][1]),
  [Biome.Forest]: pyramidFor(PAIRS[Biome.Forest][0], PAIRS[Biome.Forest][1]),
  [Biome.Water]: pyramidFor(PAIRS[Biome.Water][0], PAIRS[Biome.Water][1]),
};

/** The world's ground surface: the land-use map's height field, walked
 *  by a marching ray — water and the airfield plain at level 0, hills
 *  to ~1000 ft, all shaded by the one sun. */
export class Ground {
  /** The land-use plan this texture paints (shared with the flora). */
  readonly land = new LandUse();
  /** Render distance K: cells beyond this are not drawn (haze takes over). */
  readonly drawDistance = 24000.0;

  /** Surface height at a world point — the land-use map's height
   *  channel: water and the airfield at level 0, hills to TERRAIN_MAX. */
  heightAt(x: number, y: number): number {
    return this.land.heightAt(x, y);
  }

  /** The first intersection of the ray (origin, direction) with the
   *  terrain, as distance along the (unit) direction, or null when the
   *  ray departs. THE MARCHING RAY: the terrain lives in the band
   *  z \u2208 [0, TERRAIN_MAX], so the march starts where the ray enters
   *  the band (or at the eye, when already inside it) and steps with a
   *  RELATIVE stride (~9% of distance \u2014 near pixels step metres, horizon
   *  pixels step hundreds of metres, each staying inside its pixel's
   *  footprint), leaping whole provably-empty blocks where the height
   *  map's coarse max allows, and refining the crossing by bisection
   *  once found. Rays that cross ground level inside the airfield's
   *  flat carve are answered analytically — no march at all. */
  intersectRay(ox: number, oy: number, oz: number,
               dx: number, dy: number, dz: number): number | null {
    const K = this.drawDistance;
    let t = 0.0;
    if (oz > TERRAIN_MAX) {
      if (dz >= -1e-9) {
        return null;                        // above the band, departing
      }
      t = (TERRAIN_MAX - oz) / dz;         // enter the band here
    }
    // THE PLATEAU'S ANALYTIC HIT: through the airfield's ellipse the
    // terrain is exactly FIELD_ELEV, so a descending ray that crosses
    // the plateau's plane inside it — with its whole in-band prefix
    // inside too (the ellipse is convex) — hits at the plane crossing
    // itself: no march. The school's home views ride this path.
    if (dz < -1e-9 && oz + dz * t > FIELD_ELEV) {
      const tPlane = (FIELD_ELEV - oz) / dz;
      if (tPlane <= K
          && this.land.inField(ox + dx * t, oy + dy * t)
          && this.land.inField(ox + dx * tPlane, oy + dy * tPlane)) {
        return tPlane;
      }
    }    // the march's end: the z = 0 crossing, the band's ceiling, or K
    let end = K;
    if (dz < -1e-9) {
      const t0 = -oz / dz;
      if (t0 < end) {
        end = t0;
      }
    } else if (dz > 1e-9 && oz < TERRAIN_MAX) {
      const tUp = (TERRAIN_MAX - oz) / dz;  // rising out of the band
      if (tUp <= 0.0) {
        return null;
      }
      if (tUp < end) {
        end = tUp;
      }
    }
    let prev = t;
    while (true) {
      const z = oz + dz * t;
      const h = this.land.heightAt(ox + dx * t, oy + dy * t);
      if (z <= h) {
        // crossed into the hill: refine the crossing between prev..t
        // by FALSE POSITION — the gap to the surface is near-linear,
        // so the interpolated probes land close within a few tries
        if (prev === t) {
          return t;                         // the first probe hit
        }
        let lo = prev;
        let hi = t;
        let flo = (oz + dz * lo) - this.land.heightAt(ox + dx * lo, oy + dy * lo);
        let fhi = z - h;
        const close = Math.max(0.25, 0.001 * t);
        for (let i = 0; i < 6 && hi - lo > close; i++) {
          const mid = lo + (hi - lo) * (flo / (flo - fhi));
          const fm = (oz + dz * mid) - this.land.heightAt(ox + dx * mid, oy + dy * mid);
          if (fm <= 0.0) {
            hi = mid;
            fhi = fm;
          } else {
            lo = mid;
            flo = fm;
          }
        }
        return hi;
      }
      if (t >= end) {
        return null;                        // the band held no ground
      }
      // THE SKIP-AHEAD: when the ray flies provably above the tallest
      // terrain of the block it crosses (the height map's coarse max,
      // one ring wide for hermite's window), it may leap to that
      // block's far edge — or to where it would sink to that max — in
      // one step: no probe between can hit. The levels cascade from
      // the coarsest block down: the first whose max the ray clears
      // gives the longest provable leap; near the ground the cascade
      // bottoms out at the fine stride.
      const fine = Math.max(6.0, 0.09 * t);
      let step = fine;
      const px = ox + dx * t;
      const py = oy + dy * t;
      // THE FIELD EXIT: inside the airfield's flat carve the terrain
      // is exactly level 0 and the ray rides above it — the whole
      // in-field stretch is provably hit-free, so cross it in one
      // step to where the track leaves the ellipse.
      const fx = (px - FIELD.cx) / FIELD.rx;
      const fy = py / FIELD.ry;
      const rr = fx * fx + fy * fy;
      if (rr < 1.0) {
        const a = (dx * dx) / (FIELD.rx * FIELD.rx)
                + (dy * dy) / (FIELD.ry * FIELD.ry);
        if (a > 1e-18) {
          const b = 2.0 * (fx * dx / FIELD.rx + fy * dy / FIELD.ry);
          const exit = (-b + Math.sqrt(b * b - 4.0 * a * (rr - 1.0))) / (2.0 * a);
          if (exit > fine) {
            step = exit;
            prev = t;
            t = Math.min(t + step, end);
            continue;
          }
        }
      }
      const lvlFine = Math.min(MAXMIP_LEVELS,
        Math.max(1, Math.ceil(Math.log2(fine / this.land.texel))));
      for (let lvl = MAXMIP_LEVELS; lvl >= lvlFine; lvl--) {
        const blockMax = this.land.maxHeightAt(px, py, lvl);
        if (z <= blockMax) {
          continue;                       // this block holds taller ground
        }
        const B = this.land.texel * (1 << lvl);
        let edge = Infinity;
        if (dx > 1e-12) {
          edge = Math.min(edge, (Math.floor(px / B) * B + B - px) / dx);
        } else if (dx < -1e-12) {
          edge = Math.min(edge, (px - Math.floor(px / B) * B) / -dx);
        }
        if (dy > 1e-12) {
          edge = Math.min(edge, (Math.floor(py / B) * B + B - py) / dy);
        } else if (dy < -1e-12) {
          edge = Math.min(edge, (py - Math.floor(py / B) * B) / -dy);
        }
        let drop = Infinity;
        if (dz < -1e-12) {
          drop = (z - blockMax) / -dz;
        }
        const skip = Math.min(edge, drop);
        if (skip > fine) {
          step = skip;
        }
        break;
      }
      prev = t;
      t = Math.min(t + step, end);
    }
  }

  /** The mip level for an eye height: one map per doubling above the
   *  base rung, clamped to the pyramid's ends. */
  private static levelOf(eyeHeight: number): number {
    return Math.max(0.0, Math.min(MIP_LEVELS - 1.0,
      Math.log2(Math.max(eyeHeight, 1e-6) / ALT_PER_LEVEL)));
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

  /** One map's EXACT box filter: the checker (the product of two unit
   *  square waves) filtered over an axis-aligned square footprint of
   *  side w, centred at (u, v). Returns the filtered value in [-1, 1]:
   *  +1 deep inside a bright cell, -1 inside a dark one, 0 dissolved. */
  private static checkerMip(u: number, v: number, w: number,
                            cell: number): number {
    const half = Math.max(w, 1e-6) / 2.0;
    return Ground.squareMean(u - half, u + half, cell)
         * Ground.squareMean(v - half, v + half, cell);
  }

  /** One level's colour for a filtered checker value. */
  private static levelColour(level: MipLevel, value: number): RGB {
    return [
      level.mean[0] + value * (level.bright[0] - level.mean[0]),
      level.mean[1] + value * (level.bright[1] - level.mean[1]),
      level.mean[2] + value * (level.bright[2] - level.mean[2]),
    ];
  }

  /** One TRILINEAR tap of the manual pyramid at a fixed level, in the
   *  biome's palette: both bracketing maps are sampled and blended by
   *  the fractional level. Each map is a LITERAL COPY of the base
   *  checker — the box filter is CLAMPED at one cell of the map (edge
   *  antialiasing only), so the squares stay full contrast at any
   *  distance and altitude: the pattern NEVER averages into a flat
   *  colour. */
  private checkerTap(u: number, v: number, w: number, level: number,
                     biome: Biome): RGB {
    const pyramid = PYRAMIDS[biome];
    const lo = Math.min(Math.floor(level), MIP_LEVELS - 2);
    const frac = level - lo;
    const a = Ground.levelColour(pyramid[lo],
      Ground.checkerMip(u, v, Math.min(w, pyramid[lo].cell), pyramid[lo].cell));
    if (frac <= 0.0) {
      return a;
    }
    const hi = pyramid[lo + 1];
    const b = Ground.levelColour(hi,
      Ground.checkerMip(u, v, Math.min(w, hi.cell), hi.cell));
    return [
      a[0] + (b[0] - a[0]) * frac,
      a[1] + (b[1] - a[1]) * frac,
      a[2] + (b[2] - a[2]) * frac,
    ];
  }

  /** The mottled detail overlay (forest canopy, rocky scree): the
   *  finer checker only DARKENS, and fades out as the pixel footprint
   *  grows — full detail beside the aircraft, the mean texture from
   *  altitude. */
  private mottleOver(colour: RGB, u: number, v: number, w: number,
                     biome: Biome): RGB {
    const dark = MOTTLE_DARK[biome];
    if (dark === undefined) {
      return colour;
    }
    const fade = (w - MOTTLE_FADE_NEAR) / (MOTTLE_FADE_FAR - MOTTLE_FADE_NEAR);
    if (fade >= 1.0) {
      return colour;                        // fully faded: the plain checker
    }
    const m = Ground.checkerMip(u, v, Math.min(w, MOTTLE_CELL), MOTTLE_CELL);
    if (m >= 0.0) {
      return colour;                        // only the dark mottling
    }
    const amount = -m * MOTTLE_AMOUNT * (1.0 - Math.max(0.0, fade));
    if (amount <= 0.0) {
      return colour;
    }
    return [
      colour[0] + (dark[0] - colour[0]) * amount,
      colour[1] + (dark[1] - colour[1]) * amount,
      colour[2] + (dark[2] - colour[2]) * amount,
    ];
  }

  /** The ANISOTROPIC field sample: the pixel's ground footprint is a
   *  parallelogram spanned by the two per-pixel steps; up to ANISO_MAX
   *  taps march along its MAJOR axis, each tap filtered at the MINOR
   *  axis width and taking its own biome from the land-use map — so
   *  region boundaries blend at the tap scale. */
  private fieldTaps(u: number, v: number, stepXU: number, stepXV: number,
                    stepYU: number, stepYV: number,
                    level: number): RGB {
    const ex = Math.hypot(stepXU, stepXV);
    const ey = Math.hypot(stepYU, stepYV);
    if (!Number.isFinite(ex) || !Number.isFinite(ey) || ex < 1e-9 || ey < 1e-9) {
      // the horizon rim: no footprint — the current map, crisp
      const biome = this.land.biomeAt(u, v);
      const top = PYRAMIDS[biome][MIP_LEVELS - 1];
      return this.checkerTap(u, v, top.cell, level, biome);
    }
    let dirU: number;
    let dirV: number;
    let major: number;
    let minor: number;
    if (ex >= ey) {
      major = ex; minor = ey;
      dirU = stepXU / ex; dirV = stepXV / ex;
    } else {
      major = ey; minor = ex;
      dirU = stepYU / ey; dirV = stepYV / ey;
    }
    const aniso = Math.min(ANISO_MAX, Math.max(1, Math.ceil(major / minor)));
    let r = 0.0;
    let g = 0.0;
    let b = 0.0;
    for (let j = 0; j < aniso; j++) {
      const off = ((j + 0.5) / aniso - 0.5) * major;
      const tu = u + dirU * off;
      const tv = v + dirV * off;
      const biome = this.land.biomeAt(tu, tv);
      let tap = this.checkerTap(tu, tv, minor, level, biome);
      if (biome === Biome.Forest || biome === Biome.Rocky) {
        tap = this.mottleOver(tap, tu, tv, minor, biome);
      }
      r += tap[0];
      g += tap[1];
      b += tap[2];
    }
    return [r / aniso, g / aniso, b / aniso];
  }

  /** The road overlay: asphalt strips on the land-use grid lines, box-
   *  filtered by footprint coverage (soft edges at range), with faint
   *  length patches and a centre dash faded by distance. */
  private roadOverlay(u: number, v: number, wU: number, wV: number,
                      field: RGB, distance: number): RGB {
    const [dx, dy] = this.land.roadDistances(u, v);
    const wUf = Math.min(wU, 200.0);
    const wVf = Math.min(wV, 200.0);
    const half = this.land.roadHalf;
    const covX = Ground.coverage(u - wUf / 2, u + wUf / 2,
                                 u - dx - half, u - dx + half);
    const covY = Ground.coverage(v - wVf / 2, v + wVf / 2,
                                 v - dy - half, v - dy + half);
    const cov = Math.max(covX, covY);
    if (cov <= 0.0) {
      return field;
    }
    if (this.land.biomeAt(u, v) === Biome.Water) {
      return field;          // roads end at the shore — no asphalt on water
    }
    // the nearer road family: its length coordinate and lateral offset
    const alongX = covX >= covY;             // x-family roads run along y
    const along = alongX ? v : u;
    const lateral = alongX ? Math.abs(dx) : Math.abs(dy);
    const patch = along - ROAD_PATCH_LEN * Math.floor(along / ROAD_PATCH_LEN);
    let road: RGB = patch < ROAD_PATCH_LEN / 2 ? ROAD_COLOUR : ROAD_PATCH;
    // the centre dash (thin paint, faded with distance)
    const phase = along - (ROAD_DASH_LEN + ROAD_GAP_LEN)
      * Math.floor(along / (ROAD_DASH_LEN + ROAD_GAP_LEN));
    if (lateral < ROAD_DASH_HALF && phase < ROAD_DASH_LEN) {
      const paint = distance <= ROAD_FADE_NEAR ? 1.0
        : distance >= ROAD_FADE_FAR ? 0.0
          : (ROAD_FADE_FAR - distance) / (ROAD_FADE_FAR - ROAD_FADE_NEAR);
      if (paint > 0.0) {
        road = [
          road[0] + (ROAD_DASH[0] - road[0]) * paint,
          road[1] + (ROAD_DASH[1] - road[1]) * paint,
          road[2] + (ROAD_DASH[2] - road[2]) * paint,
        ];
      }
    }
    return [
      road[0] + (field[0] - road[0]) * (1.0 - cov),
      road[1] + (field[1] - road[1]) * (1.0 - cov),
      road[2] + (field[2] - road[2]) * (1.0 - cov),
    ];
  }

  /** The ground colour for one pixel: (u, v) is the ground point, the
   *  steps are the footprint's per-pixel ground extents (right, down),
   *  eyeHeight picks the mip map, and distance drives the haze, the
   *  road dash and the marking fades. The biome comes from the land-use
   *  map (per tap); the runway is painted last, over everything. */
  sample(u: number, v: number, stepXU: number, stepXV: number,
         stepYU: number, stepYV: number, distance: number,
         eyeHeight: number): RGB {
    const level = Ground.levelOf(eyeHeight);
    let colour = this.fieldTaps(u, v, stepXU, stepXV, stepYU, stepYV, level);
    colour = this.roadOverlay(u, v, Math.hypot(stepXU, stepXV),
                              Math.hypot(stepYU, stepYV), colour, distance);
    // THE RELIEF: the surface normal against the sun, normalised so the
    // flat plain keeps its exact palette (roads shade with their hill)
    const n = this.land.normalAt(u, v);
    const ndl = Math.max(0.0, n[0] * SUN_DIR[0] + n[1] * SUN_DIR[1]
                                + n[2] * SUN_DIR[2]);
    const f = (SHADE_AMBIENT + SHADE_SUN * ndl) / SHADE_FLAT;
    colour = [Math.min(255.0, colour[0] * f),
              Math.min(255.0, colour[1] * f),
              Math.min(255.0, colour[2] * f)];
    // the runway's footprint coverage: soft slab edges in both axes
    const wU = Math.min(Math.hypot(stepXU, stepXV), RUNWAY_FILTER_MAX);
    const wV = Math.min(Math.hypot(stepYU, stepYV), RUNWAY_FILTER_MAX);
    const cov = Ground.coverage(u - wU / 2, u + wU / 2, RUNWAY.near, RUNWAY.far)
                * Ground.coverage(v - wV / 2, v + wV / 2, -RUNWAY.half, RUNWAY.half);
    if (cov <= 0.0) {
      return colour;
    }
    const rw = this.runwayShade(u, v, distance);
    if (cov >= 1.0) {
      return rw;
    }
    return [
      rw[0] + (colour[0] - rw[0]) * (1.0 - cov),
      rw[1] + (colour[1] - rw[1]) * (1.0 - cov),
      rw[2] + (colour[2] - rw[2]) * (1.0 - cov),
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
                      distance: number): RGB {
    const along = x - RUNWAY.near;
    const paint = distance <= MARK_FADE_NEAR ? 1.0
      : distance >= MARK_FADE_FAR ? 0.0
        : (MARK_FADE_FAR - distance) / (MARK_FADE_FAR - MARK_FADE_NEAR);
    // asphalt with alternating repair patches (110 m patches never alias)
    const patch = Math.floor(along / RUNWAY.patch) & 1;
    const base: RGB = patch === 0 ? [77, 85, 95] : [96, 104, 114];
    let colour: RGB = base;
    // threshold stripes: 8 bars across the first 24 m, centre gap kept clear
    if (along <= RUNWAY.threshold) {
      const bar = Math.floor(y / RUNWAY.bar);
      const inBar = y - bar * RUNWAY.bar < RUNWAY.barGap;
      if (inBar && bar !== 0 && bar !== -1) {
        colour = [232, 236, 244];
      }
    }
    // the dashed centre line
    const centre = along - RUNWAY.dashFrom;
    if (colour === base && centre >= 0.0 && x <= RUNWAY.far - RUNWAY.dash
        && centre % (RUNWAY.dash + RUNWAY.dashGap) < RUNWAY.dash
        && Math.abs(y) <= 0.9) {
      colour = [232, 236, 244];
    }
    // the side lines
    if (colour === base && Math.abs(Math.abs(y) - RUNWAY.half) <= 0.4) {
      colour = [232, 236, 244];
    }
    if (colour !== base && paint < 1.0) {
      colour = [
        base[0] + (232 - base[0]) * paint,
        base[1] + (236 - base[1]) * paint,
        base[2] + (244 - base[2]) * paint,
      ];
    }
    return colour;
  }
}
