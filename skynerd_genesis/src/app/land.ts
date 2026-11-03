/**
 * The land-use layer: the landscape's ground plan, world-fixed and
 * TILEABLE — an 8x8 hand-designed map of 512 m tiles repeating every
 * 4096 m. It decides where the ground is grass (the green default),
 * where the fields dry yellow, where the rock outcrops are, and where
 * the forests stand; the roads run along the map's grid lines, straight
 * strips crossing at the hamlet sites.
 *
 * The FLORA is deterministic from world coordinates alone (a hash of
 * the cell indices): the same tree stands in the same place on every
 * query, every frame, every flight — the landscape never shuffles.
 * Trees fill the forest tiles with lone specimens scattered on the
 * grass; each road crossing grows a hamlet of a dozen houses flanking
 * the roads. The airfield is kept clear: nothing plants on the runway
 * strip or the roads.
 *
 * Flat for now, like the ground it plans — the tile map is the seam
 * where terrain height arrives later (a height field keyed to these
 * same tiles).
 */

/** The landscape's biomes: the base palettes the checker paints in. */
export enum Biome {
  Grass = 0,
  Dry = 1,
  Rocky = 2,
  Forest = 3,
}

const G = Biome.Grass, D = Biome.Dry, R = Biome.Rocky, F = Biome.Forest;

/**
 * The tileable land-use map: 8x8 tiles of 512 m, repeating every 4096 m.
 * Row index = the y tile, column = the x tile (row 7 wraps to y < 0).
 * The rows beside y = 0 stay grass — the runway's immediate field.
 */
const LAND_MAP: readonly (readonly Biome[])[] = [
  [G, G, G, D, G, G, G, G],   // y ∈ [0, 512)
  [G, G, D, R, G, D, G, G],
  [G, G, F, F, F, G, R, G],
  [D, G, F, F, F, G, G, G],
  [G, G, F, G, G, D, G, R],
  [G, D, G, G, G, G, G, G],   // the hamlet's tiles — grass, houses on top
  [G, R, D, G, G, G, D, G],
  [G, G, G, G, D, G, G, G],   // y ∈ [-512, 0)
];

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
  conifer: boolean;
  shade: number;    // 0..1: which of the two greens
}

/** A house, in world metres: an axis-aligned box with a ridge roof. */
export interface HouseSpot {
  x: number;        // floor centre
  y: number;
  w: number;        // footprint along x
  d: number;        // footprint along y
  wallH: number;
  roofH: number;
  along: "x" | "y"; // the ridge's axis
}

/** Deterministic [0,1) hash of a world cell — the flora's whole memory. */
function hash01(ix: number, iy: number, salt: number): number {
  let h = (Math.imul(ix, 374761393) + Math.imul(iy, 668265263)
           + Math.imul(salt, 1442695041)) | 0;
  h = Math.imul(h ^ (h >>> 13), 1274126177);
  h ^= h >>> 16;
  return (h >>> 0) / 4294967296;
}

/** The hamlet layout at each road crossing: offsets from the crossing
 *  and the ridge axis — twelve houses flanking the two roads. */
const HAMLET: readonly (readonly [number, number, "x" | "y"])[] = [
  [34, 34, "x"], [34, 62, "y"], [62, 34, "y"], [62, 62, "x"],
  [34, -34, "y"], [62, -34, "x"], [34, -62, "x"], [-62, 34, "x"],
  [-34, 34, "y"], [-34, 62, "x"], [-34, -34, "x"], [-62, -34, "y"],
];

/** The world's tileable ground plan. */
export class LandUse {
  readonly period = 4096.0;      // the map repeats every 4 km
  readonly tile = 512.0;
  readonly roadHalf = 4.0;       // road carriageway halfwidth, m
  private readonly roadLine = 2560.0;  // centrelines: x ≡ 2560, y ≡ 2560

  /** The biome that owns the ground point. */
  biomeAt(x: number, y: number): Biome {
    const cx = (((Math.floor(x / this.tile) % 8) + 8) % 8) | 0;
    const cy = (((Math.floor(y / this.tile) % 8) + 8) % 8) | 0;
    return LAND_MAP[cy][cx];
  }

  /** Signed distances to the nearest road centreline of each family
   *  ([x-family, y-family], metres — the x-family roads run along y). */
  roadDistances(x: number, y: number): [number, number] {
    return [LandUse.wrapDist(x - this.roadLine, this.period),
            LandUse.wrapDist(y - this.roadLine, this.period)];
  }

  private static wrapDist(v: number, period: number): number {
    const d = v - period * Math.floor(v / period);
    return d > period / 2 ? d - period : d;
  }
}

/** The landscape's plants and buildings, deterministic from the world. */
export class Flora {
  constructor(readonly land: LandUse) {}

  /** The trees whose trunk stands inside the query rectangle, appended
   *  to `out`. Dense in the forest tiles, lone specimens elsewhere. */
  treesNear(x0: number, y0: number, x1: number, y1: number,
            out: TreeSpot[]): void {
    const SP = 20.0;                        // the planting grid, m
    for (let iy = Math.floor(y0 / SP); iy <= Math.floor(y1 / SP); iy++) {
      for (let ix = Math.floor(x0 / SP); ix <= Math.floor(x1 / SP); ix++) {
        const biome = this.land.biomeAt((ix + 0.5) * SP, (iy + 0.5) * SP);
        const density = biome === Biome.Forest ? 0.85
          : biome === Biome.Grass ? 0.02
          : biome === Biome.Dry ? 0.008 : 0.0;
        if (hash01(ix, iy, 1) >= density) {
          continue;
        }
        const x = (ix + 0.15 + 0.7 * hash01(ix, iy, 2)) * SP;
        const y = (iy + 0.15 + 0.7 * hash01(ix, iy, 3)) * SP;
        if (x < x0 || x > x1 || y < y0 || y > y1 || this.excluded(x, y)) {
          continue;
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
    const P = this.land.period;
    const L = 2560.0;                       // the crossings' coordinates
    for (let j = Math.floor((y0 - L - 120) / P);
         j <= Math.floor((y1 - L + 120) / P); j++) {
      for (let i = Math.floor((x0 - L - 120) / P);
           i <= Math.floor((x1 - L + 120) / P); i++) {
        for (const [ox, oy, along] of HAMLET) {
          const x = L + i * P + ox;
          const y = L + j * P + oy;
          if (x < x0 || x > x1 || y < y0 || y > y1) {
            continue;
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
