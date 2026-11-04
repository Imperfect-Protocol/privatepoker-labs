/**
 * The landscape's land-use plan — a PROCEDURAL, TILEABLE high-level
 * texture over the world.
 *
 * The LOW-level texture is the checkerboard (src/app/ground.ts); this
 * module is the HIGH-level texture — a multidimensional ground
 * MATERIAL: every texel carries not just its colour class but its
 * HEIGHT. The 512x512 map of 16 m texels (repeating every 8192 m,
 * generated at load from PERIODIC value noise — quintic-interpolated
 * lattice hashes; deterministic: the same lake sits in the same place
 * in every flight, every frame, on every machine) decides what each
 * square IS — grass, dry, rocky, forest, WATER — and how HIGH it
 * stands: rivers and lakes at LEVEL 0, every land surface ABOVE
 * 10 ft (a bank at every waterline), rocky highlands rising to
 * ~1000 ft. The AIRFIELD is carved to a flat plateau at 95 ft (the
 * runway — the physics world's AIRFIELD, one shared shape), rimmed
 * by a smooth blend into the natural terrain.
 *
 * The classifier reads three noise fields:
 *   h — land height (3 octaves): the lowlands hold the LAKES, the
 *       highlands are ROCKY scree;
 *   m — moisture (2 octaves): moist midlands grow FOREST, dry uplands
 *       turn the yellow-brown DRY grass;
 *   r — the river field: where r crosses its midline a RIVER band
 *       meanders, kept to the lowlands.
 * Every water body wears a DRY shore ring (sandy banks).
 *
 * The AIRFIELD is carved: an ellipse of plain grass under and around
 * the runway, so the field never lands in a lake or on a scree.
 *
 * The ROADS are a separate analytic grid (two families, 4 km apart)
 * with a hamlet of houses at each crossing — kept off the water and
 * out of the airfield. The Flora plants trees from the same map:
 * dense in the forests, lone specimens on the grass and the dry, none
 * on rock or water.
 */

import { AIRFIELD } from "../core/world.js";

/** The landscape's biomes: the base palettes the checker paints in. */
export enum Biome {
  Grass = 0,
  Dry = 1,
  Rocky = 2,
  Forest = 3,
  Water = 4,
}

/** The world's light direction — the one sun the whole landscape obeys
 *  (the terrain's relief shading, the objects' face shades, the
 *  shadows, the glare): azimuth ~200\u00b0, elevation ~20\u00b0, unit length. */
export const SUN_DIR: readonly [number, number, number] = (() => {
  const v = [-0.88, -0.32, 0.35];
  const l = Math.hypot(v[0], v[1], v[2]);
  return [v[0] / l, v[1] / l, v[2] / l];
})();

/** The terrain's tallest hill, metres (~1000 ft). */
export const TERRAIN_MAX = 305.0;
/** The airfield's flat carve: the ellipse of plateau ground the runway
 *  lives on (the shape shared with the physics world's AIRFIELD),
 *  rimmed by a smooth roll into the natural terrain. */
export const FIELD = { cx: AIRFIELD.cx, rx: AIRFIELD.rx, ry: AIRFIELD.ry, rim: 1.45 };
/** The plateau's height: 95 ft — the runway's flat ground. The
 *  aircraft's 100 ft takeoff datum is this plus 5 ft of landing gear,
 *  so standing on the runway the instruments read 100 ft. */
export const FIELD_ELEV = AIRFIELD.elev;
/** The runway, in world metres: the FINITE 1.7 km lane along +X the
 *  school flies. Lived here so the flora can keep the airfield clear
 *  and the ground texture can paint it. */
export const RUNWAY = { near: -250.0, far: 1450.0, half: 15.0, patch: 110.0,
                        threshold: 24.0, bar: 3.5, barGap: 2.0,
                        dashFrom: 40.0, dash: 30.0, dashGap: 20.0 };

/** A tree, in world metres. */
export interface TreeSpot {
  x: number;        // trunk base
  y: number;
  h: number;        // total height
  r: number;        // canopy radius
  conifer: boolean; // cone or round canopy
  shade: number;     // canopy tint variation
}

/** A cloud, in world metres: the centre of its puff cluster. */
export interface CloudSpot {
  x: number;
  y: number;
  alt: number;        // the cluster's base altitude
  r: number;          // the cluster's horizontal radius
  seedX: number;      // the lattice cell — the cluster's dice
  seedY: number;
}

/** A house, in world metres. */
export interface HouseSpot {
  x: number;
  y: number;
  w: number;             // wall width
  d: number;             // wall depth
  wallH: number;         // wall height
  roofH: number;         // ridge height above the walls
  along: "x" | "y";      // the ridge axis
}

/** Deterministic [0, 1) hash of two integers and a salt — the whole
 *  landscape, noise and flora alike, grows from this one function. */
export function hash01(ix: number, iy: number, salt: number): number {
  let h = (ix * 374761393 + iy * 668265263 + salt * 2246822519) | 0;
  h = Math.imul(h ^ (h >>> 13), 1274126177);
  h ^= h >>> 16;
  return (h >>> 0) / 4294967296;
}

// --- the procedural map -----------------------------------------------------

/** The high-level texture: 512x512 texels of 16 m, tiling every 8 km. */
const TEX = 512;
const TEX_MASK = TEX - 1;                    // 512 is a power of two: wrapping is a mask
const TEXEL = 16.0;
const PERIOD = TEX * TEXEL;               // 8192 m — the map's period

/** Periodic value noise: a lattice of freq x freq hashed points per map
 *  period, quintic-interpolated. Integer freq keeps the noise — and so
 *  the whole map — seamlessly tileable. */
function vnoise(u: number, v: number, freq: number, salt: number): number {
  const gx = u * freq;
  const gy = v * freq;
  const x0 = Math.floor(gx);
  const y0 = Math.floor(gy);
  const smooth = (t: number): number => t * t * t * (t * (t * 6 - 15) + 10);
  const fx = smooth(gx - x0);
  const fy = smooth(gy - y0);
  const wrap = (i: number): number => i - Math.floor(i / freq) * freq;
  const n00 = hash01(wrap(x0), wrap(y0), salt);
  const n10 = hash01(wrap(x0 + 1), wrap(y0), salt);
  const n01 = hash01(wrap(x0), wrap(y0 + 1), salt);
  const n11 = hash01(wrap(x0 + 1), wrap(y0 + 1), salt);
  const a = n00 + (n10 - n00) * fx;
  const b = n01 + (n11 - n01) * fx;
  return a + (b - a) * fy;
}

/** The classifier's thresholds on the noise fields. */
const WATER_LEVEL = 0.32;      // h below: lakes and seas
const SHORE_LEVEL = 0.365;     // h below (but above water): sandy shore
const ROCK_LEVEL = 0.71;       // h above: rocky highlands
const FOREST_M = 0.63;         // moist midlands grow forest
const DRY_M = 0.40;            // dry uplands turn yellow
const DRY_H = 0.52;
const RIVER_W = 0.030;         // the river band, in the r field
const RIVER_MAX_H = 0.60;      // rivers only through the lowlands
const RIVER_VALLEY_W = 0.090;  // the river's valley, 3x the band: wide
                               // gentle banks (the water's own width +
                               // the land's roll down to it)
/** The LAND's floor: every land surface stands above 10 ft — the bank
 *  at every waterline (water at 0, the land never lower than this). */
const LAND_MIN = 3.2;

/** The world's tileable ground plan, one Biome per texel. */
const MAP: Uint8Array = new Uint8Array(TEX * TEX);
/** The same map's HEIGHT channel, metres above level 0. */
const HEIGHT: Float32Array = new Float32Array(TEX * TEX);
{
  for (let ty = 0; ty < TEX; ty++) {
    const v = (ty + 0.5) / TEX;
    const wyRaw = (ty + 0.5) * TEXEL;    // the texel centre, wrapped to
    const wy = wyRaw > PERIOD / 2 ? wyRaw - PERIOD : wyRaw;  // +/- period/2
    for (let tx = 0; tx < TEX; tx++) {
      const u = (tx + 0.5) / TEX;
      // the land's height: broad swells + medium relief + fine detail
      const h = 0.55 * vnoise(u, v, 6, 21)
              + 0.28 * vnoise(u, v, 13, 22)
              + 0.17 * vnoise(u, v, 27, 23);
      // the river field: its midline crossings are the river's course;
      // the margin (0 at the midline, 1 at the band's edge) shapes the
      // valley — the water's width and its banks' gentle roll
      const riverMargin = Math.abs(vnoise(u, v, 5, 26) - 0.5) / RIVER_W;
      const river = riverMargin < 1.0;      let biome: Biome;
      if (h < WATER_LEVEL || (river && h < RIVER_MAX_H)) {
        biome = Biome.Water;
      } else if (h < SHORE_LEVEL) {
        biome = Biome.Dry;               // the sandy shore ring
      } else if (h > ROCK_LEVEL) {
        biome = Biome.Rocky;
      } else {
        const m = 0.70 * vnoise(u, v, 9, 24)
                + 0.30 * vnoise(u, v, 21, 25);
        biome = m > FOREST_M ? Biome.Forest
          : m < DRY_M && h > DRY_H ? Biome.Dry
          : Biome.Grass;
      }
      const wxRaw = (tx + 0.5) * TEXEL;
      const wx = wxRaw > PERIOD / 2 ? wxRaw - PERIOD : wxRaw;
      // THE HEIGHT CHANNEL: WATER FLAT AT LEVEL 0 — lakes and rivers
      // alike, the whole water plane; the LAND ramping from its floor
      // (LAND_MIN, just above 10 ft: the bank at every waterline)
      // through the low ground (t^1.6) to the rocky highlands at the
      // cap
      const t = Math.min(1.0, Math.max(0.0, (h - WATER_LEVEL) / 0.55));
      const ramp = LAND_MIN + (TERRAIN_MAX - LAND_MIN) * t ** 1.6;
      let elev = 0.0;
      if (biome !== Biome.Water) {
        elev = ramp;
        // THE RIVER'S VALLEY: the land rolls gently down around the
        // river's water — from the plain to the floor across a valley
        // 3x the band's width (wide banks, no shimmer), shallowing to
        // nothing before the RIVER_MAX_H cutoff so a river's source
        // never cliffs
        const valleyMargin = riverMargin * RIVER_W / RIVER_VALLEY_W;
        if (valleyMargin < 1.0 && h < RIVER_MAX_H) {
          const inner = RIVER_W / RIVER_VALLEY_W;
          const wall = Math.max(0.0, (valleyMargin - inner)
            / (1.0 - inner)) ** 1.5;
          const headFade = Math.min(1.0,
            Math.max(0.0, (RIVER_MAX_H - h) / 0.08));
          const dip = 1.0 - (1.0 - wall) * headFade;
          elev = LAND_MIN + (ramp - LAND_MIN) * dip;
        }
      }
      // the field's carve: the plateau at FIELD_ELEV (95 ft — the
      // runway's flat ground, the physics world's AIRFIELD), rimmed by
      // a smooth roll into the natural terrain
      const dx = (wx - FIELD.cx) / FIELD.rx;
      const dy = wy / FIELD.ry;
      const w = dx * dx + dy * dy;
      if (w < FIELD.rim) {
        if (w < 1.0) {
          elev = FIELD_ELEV;
        } else {
          const r = (w - 1.0) / (FIELD.rim - 1.0);
          const s = r * r * (3.0 - 2.0 * r);
          // the rim rolls toward the natural terrain but never below
          // the land's floor — where the rim reaches water, the apron
          // banks at LAND_MIN like every other waterline
          const target = Math.max(elev, LAND_MIN);
          elev = FIELD_ELEV + (target - FIELD_ELEV) * s;
        }
        biome = Biome.Grass;             // the airfield's clearing + apron
      }
      MAP[ty * TEX + tx] = biome;
      HEIGHT[ty * TEX + tx] = elev;
    }
  }
}

// The coarse-max pyramid over HEIGHT: each level's block carries the
// tallest height of its texels plus a one-texel ring (hermite
// heightAt's bilinear window reaches one texel over), so a ray
// provably above a block's max is above the terrain through it.
// Level l blocks are 2^l texels wide: 32 m to 2048 m at l = 1..7.
export const MAXMIP_LEVELS = 7;
const MAXMIP: Float32Array[] = [];
{
  let prev: Float32Array = HEIGHT;
  let prevSize = TEX;
  for (let lvl = 1; lvl <= MAXMIP_LEVELS; lvl++) {
    const size = TEX >> lvl;
    const cur = new Float32Array(size * size);
    for (let by = 0; by < size; by++) {
      for (let bx = 0; bx < size; bx++) {
        let m = 0.0;
        // the 2x2 children plus the ring of children around them
        for (let cy = 2 * by - 1; cy <= 2 * by + 2; cy++) {
          const wy = ((cy % prevSize) + prevSize) % prevSize;
          for (let cx = 2 * bx - 1; cx <= 2 * bx + 2; cx++) {
            const wx = ((cx % prevSize) + prevSize) % prevSize;
            const v = prev[wy * prevSize + wx];
            if (v > m) {
              m = v;
            }
          }
        }
        cur[by * size + bx] = m;
      }
    }
    MAXMIP.push(cur);
    prev = cur;
    prevSize = size;
  }
}

// --- the plan ----------------------------------------------------------------

/** The hamlet layout at each road crossing: offsets from the crossing
 *  and the ridge axis — twelve houses flanking the two roads. */
const HAMLET: readonly (readonly [number, number, "x" | "y"])[] = [
  [34, 34, "x"], [34, 62, "y"], [62, 34, "y"], [62, 62, "x"],
  [34, -34, "y"], [62, -34, "x"], [34, -62, "x"], [-62, 34, "x"],
  [-34, 34, "y"], [-34, 62, "x"], [-34, -34, "x"], [-62, -34, "y"],
];

/** The world's tileable ground plan. */
export class LandUse {
  readonly period = PERIOD;      // the map repeats every 8 km
  readonly texel = TEXEL;        // one map texel, metres
  readonly roadHalf = 4.0;       // road carriageway halfwidth, m
  private readonly roadSpacing = 4096.0;  // one road per 4 km, each family
  private readonly roadLine = 2560.0;     // centrelines ≡ 2560 mod 4096

  /** The biome that owns the ground point. */
  biomeAt(x: number, y: number): Biome {
    const ix = Math.floor(x / TEXEL) & TEX_MASK;
    const iy = Math.floor(y / TEXEL) & TEX_MASK;
    return MAP[iy * TEX + ix] as Biome;
  }
  /** Whether (x, y) lies inside the airfield's flat carve — through
   *  the whole ellipse the terrain is exactly level 0. */
  inField(x: number, y: number): boolean {
    const fx = (x - FIELD.cx) / FIELD.rx;
    const fy = y / FIELD.ry;
    return fx * fx + fy * fy < 1.0;
  }

  /** The tallest terrain within the 2^lvl-texel block containing
   *  (x, y), ring included — the marching ray's skip-ahead bound. */
  maxHeightAt(x: number, y: number, lvl: number): number {
    const ix = Math.floor(x / TEXEL) & TEX_MASK;
    const iy = Math.floor(y / TEXEL) & TEX_MASK;
    const size = TEX >> lvl;
    return MAXMIP[lvl - 1][(iy >> lvl) * size + (ix >> lvl)];
  }

  /** The terrain's height at a world point, metres above level 0 —
   *  hermite-interpolated off the height channel (smooth normals, no
   *  grid faceting), seamlessly tileable with the map. */
  heightAt(x: number, y: number): number {
    const gx = x / TEXEL - 0.5;
    const gy = y / TEXEL - 0.5;
    const ix = Math.floor(gx);
    const iy = Math.floor(gy);
    const sx = gx - ix;
    const sy = gy - iy;
    const fx = sx * sx * (3.0 - 2.0 * sx);
    const fy = sy * sy * (3.0 - 2.0 * sy);
    const x0 = ix & TEX_MASK;
    const y0 = iy & TEX_MASK;
    const x1 = (x0 + 1) & TEX_MASK;
    const y1 = (y0 + 1) & TEX_MASK;
    const h00 = HEIGHT[y0 * TEX + x0];
    const h10 = HEIGHT[y0 * TEX + x1];
    const h01 = HEIGHT[y1 * TEX + x0];
    const h11 = HEIGHT[y1 * TEX + x1];
    const a = h00 + (h10 - h00) * fx;
    const b = h01 + (h11 - h01) * fx;
    return a + (b - a) * fy;
  }

  /** The terrain's surface normal at a world point (unit length) —
   *  central differences at one texel's spacing. Flat ground reads
   *  (0, 0, 1); the hills lean with their slopes. */
  normalAt(x: number, y: number): [number, number, number] {
    const hx = this.heightAt(x + TEXEL, y) - this.heightAt(x - TEXEL, y);
    const hy = this.heightAt(x, y + TEXEL) - this.heightAt(x, y - TEXEL);
    const d = 2.0 * TEXEL;
    const nx = -hx / d;
    const ny = -hy / d;
    const l = Math.hypot(nx, ny, 1.0);
    return [nx / l, ny / l, 1.0 / l];
  }

  /** Signed distances to the nearest road centreline of each family
   *  ([x-family, y-family], metres — the x-family roads run along y). */
  roadDistances(x: number, y: number): [number, number] {
    return [LandUse.fold(x - this.roadLine, this.roadSpacing),
            LandUse.fold(y - this.roadLine, this.roadSpacing)];
  }

  /** Fold to the nearest line of a spacing family. */
  private static fold(v: number, spacing: number): number {
    const d = v - spacing * Math.floor(v / spacing);   // [0, spacing)
    return d > spacing / 2 ? d - spacing : d;
  }
}

// --- the flora ----------------------------------------------------------------

/** Tree density per biome: the forest is a forest, the grass and the
 *  dry carry lone specimens, rock and water stay bare. */
const TREE_DENSITY: Readonly<Record<Biome, number>> = {
  [Biome.Forest]: 0.85,
  [Biome.Grass]: 0.02,
  [Biome.Dry]: 0.008,
  [Biome.Rocky]: 0.0,
  [Biome.Water]: 0.0,
};

/** The landscape's plants, buildings and clouds, deterministic from
 *  the world. */
export class Flora {
  constructor(readonly land: LandUse) {}

  /** The trees whose trunk stands inside the query rectangle, appended
   *  to `out`. Dense in the forest, lone specimens elsewhere. */
  treesNear(x0: number, y0: number, x1: number, y1: number,
            out: TreeSpot[]): void {
    const SP = 20.0;                        // the planting grid, m
    for (let iy = Math.floor(y0 / SP); iy <= Math.floor(y1 / SP); iy++) {
      for (let ix = Math.floor(x0 / SP); ix <= Math.floor(x1 / SP); ix++) {
        const biome = this.land.biomeAt((ix + 0.5) * SP, (iy + 0.5) * SP);
        if (hash01(ix, iy, 1) >= TREE_DENSITY[biome]) {
          continue;
        }
        const x = (ix + 0.15 + 0.7 * hash01(ix, iy, 2)) * SP;
        const y = (iy + 0.15 + 0.7 * hash01(ix, iy, 3)) * SP;
        if (x < x0 || x > x1 || y < y0 || y > y1 || this.excluded(x, y)) {
          continue;
        }
        if (this.land.biomeAt(x, y) === Biome.Water) {
          continue;                        // never a tree in the water
        }
        out.push({
          x, y,
          h: 7.0 + 4.0 * hash01(ix, iy, 4),
          r: 3.0 + 1.5 * hash01(ix, iy, 5),
          conifer: hash01(ix, iy, 6) < 0.45,
          shade: hash01(ix, iy, 7),
        });
      }
    }
  }

  /** The hamlet houses whose centre stands inside the query rectangle,
   *  appended to `out`. One hamlet per road crossing. */
  housesNear(x0: number, y0: number, x1: number, y1: number,
             out: HouseSpot[]): void {
    const S = 4096.0;                      // the crossings' spacing, m
    const L = 2560.0;                     // the crossings' phase
    for (let j = Math.floor((y0 - L - 120) / S);
         j <= Math.floor((y1 - L + 120) / S); j++) {
      for (let i = Math.floor((x0 - L - 120) / S);
           i <= Math.floor((x1 - L + 120) / S); i++) {
        for (const [ox, oy, along] of HAMLET) {
          const x = L + i * S + ox;
          const y = L + j * S + oy;
          if (x < x0 || x > x1 || y < y0 || y > y1) {
            continue;
          }
          if (this.land.biomeAt(x, y) === Biome.Water) {
            continue;                      // no houses in the lakes
          }
          const hx = Math.round(x);
          const hy = Math.round(y);
          out.push({
            x, y, along,
            w: 7.0 + 3.0 * hash01(hx, hy, 11),
            d: 5.0 + 2.0 * hash01(hx, hy, 12),
            wallH: 2.6 + 0.8 * hash01(hx, hy, 13),
            roofH: 1.6 + 0.6 * hash01(hx, hy, 14),
          });
        }
      }
    }
  }

  /** The clouds whose cluster centre stands inside the query rectangle,
   *  appended to `out`: big soft masses on a 1 km lattice, drifting at
   *  1.5-2.6 km altitude — the same cloud over the same place, always. */
  cloudsNear(x0: number, y0: number, x1: number, y1: number,
             out: CloudSpot[]): void {
    const SP = 1024.0;                      // the cloud lattice, m
    for (let iy = Math.floor(y0 / SP); iy <= Math.floor(y1 / SP); iy++) {
      for (let ix = Math.floor(x0 / SP); ix <= Math.floor(x1 / SP); ix++) {
        if (hash01(ix, iy, 31) >= 0.30) {
          continue;                         // most lattice cells stay clear
        }
        const x = (ix + 0.2 + 0.6 * hash01(ix, iy, 32)) * SP;
        const y = (iy + 0.2 + 0.6 * hash01(ix, iy, 33)) * SP;
        if (x < x0 || x > x1 || y < y0 || y > y1) {
          continue;
        }
        out.push({
          x, y,
          alt: 1500.0 + 1100.0 * hash01(ix, iy, 34),
          r: 270.0 + 480.0 * hash01(ix, iy, 35),
          seedX: ix, seedY: iy,
        });
      }
    }
  }

  /** Nothing plants on the airfield or the roads. */
  private excluded(x: number, y: number): boolean {
    if (x > RUNWAY.near - 110.0 && x < RUNWAY.far + 110.0
        && Math.abs(y) < 125.0) {
      return true;                          // the airfield's clear zone
    }
    const [dx, dy] = this.land.roadDistances(x, y);
    return Math.abs(dx) < 13.0 || Math.abs(dy) < 13.0;
  }
}
