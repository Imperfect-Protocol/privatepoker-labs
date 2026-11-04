/**
 * The landscape's land-use plan — a PROCEDURAL, TILEABLE high-level
 * texture over the world.
 *
 * The LOW-level texture is the checkerboard (src/app/ground.ts); this
 * module is the HIGH-level texture that decides what each square IS:
 * grass, dry, rocky, forest — and WATER: lakes in the lowlands, rivers
 * wandering through them. It is a 512x512 map of 16 m texels, repeating
 * every 8192 m, GENERATED at load from PERIODIC value noise
 * (quintic-interpolated lattice hashes): the regions are organic blobs
 * and bands, not square tiles, and the whole map is deterministic — the
 * same lake sits in the same place in every flight, every frame, on
 * every machine.
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

/** The landscape's biomes: the base palettes the checker paints in. */
export enum Biome {
  Grass = 0,
  Dry = 1,
  Rocky = 2,
  Forest = 3,
  Water = 4,
}

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

/** The airfield's clearing: an ellipse of plain grass under and around
 *  the runway, so the field never sits in a lake or on a scree. */
function carvedGrass(x: number, y: number): boolean {
  const dx = (x - 600.0) / 1050.0;
  const dy = y / 420.0;
  return dx * dx + dy * dy < 1.0;
}

/** The world's tileable ground plan, one Biome per texel. */
const MAP: Uint8Array = new Uint8Array(TEX * TEX);
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
      // the river field: its midline crossings are the river's course
      const river = Math.abs(vnoise(u, v, 5, 26) - 0.5) < RIVER_W;
      let biome: Biome;
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
      if (carvedGrass(wx, wy)) {
        biome = Biome.Grass;             // the airfield's clearing
      }
      MAP[ty * TEX + tx] = biome;
    }
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
    const ix = (((Math.floor(x / TEXEL) % TEX) + TEX) % TEX) | 0;
    const iy = (((Math.floor(y / TEXEL) % TEX) + TEX) % TEX) | 0;
    return MAP[iy * TEX + ix] as Biome;
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
