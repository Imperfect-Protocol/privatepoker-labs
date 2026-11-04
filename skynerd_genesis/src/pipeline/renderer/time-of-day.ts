/**
 * The time of day — the sky's moods (plan §5's toolbar dropdown). One
 * table is the SINGLE SOURCE both renderers read: the GPU pipeline
 * packs each mood into the Frame uniform (zenith/fog/sunTint rows),
 * the CPU cockpit reads the same numbers for its per-pixel sky, the
 * sun, the glare, the flare and the cloud tint. noon IS the classic
 * look — the exact constants the CPU path and the WGSL carried before
 * the moods existed, so default parity is exact.
 */

/** The dropdown's choices. */
export type TimeOfDay = "dawn" | "sunrise" | "noon" | "sunset"
                       | "dusk" | "night";

/** One sky mood: where the sun stands, what the sky wears. */
export interface SkyMood {
  /** Unit vector TOWARD the sun (the relief law's light too). */
  readonly sunDir: readonly [number, number, number];
  /** The overhead sky colour, 0..255. */
  readonly zenith: readonly [number, number, number];
  /** The horizon fog colour, 0..255 (the haze target on the ground). */
  readonly fog: readonly [number, number, number];
  /** The sun's disc/glow tint, 0..255. */
  readonly disc: readonly [number, number, number];
  /** The disc's angular radius, rad (noon 0.035 ≈ 2°; sunsets swell). */
  readonly discSize: number;
  /** The sun's brightness (0 = no disc — dusk, night). */
  readonly brightness: number;
  /** The ambient light scale on terrain and trees (night is dark). */
  readonly ambient: number;
  /** The cloud tint, 0..255 (sunrise wears pink). */
  readonly cloud: readonly [number, number, number];
  /** THE HOUR'S GRADE — the ground's and the instruments' colour
   *  filter, 0..255 (normalised to mean 1 where it's applied, so it
   *  TINTS without darkening; noon is pure white — no grade at all). */
  readonly grade: readonly [number, number, number];
}

/** The grade's strength on the world (the instruments wear it harder,
 *  the ground softer — both stay readable under any hour). */
export const GROUND_GRADE = 0.55;

/** Normalise a direction triple (the table writes readable vectors). */
function unit(x: number, y: number, z: number):
    readonly [number, number, number] {
  const inv = 1.0 / Math.hypot(x, y, z);
  return [x * inv, y * inv, z * inv];
}

export const SKY_MOODS: Readonly<Record<TimeOfDay, SkyMood>> = {
  /** The classic sky — the pre-mood constants, exactly. */
  noon: {
    sunDir: unit(-0.88, -0.32, 0.35),
    zenith: [30, 46, 84],
    fog: [184, 200, 214],
    disc: [255, 253, 242],
    discSize: 0.035,
    brightness: 1.0,
    ambient: 1.0,
    cloud: [250, 252, 255],
    grade: [255, 255, 255],
  },
  /** The sun low in the east, the clouds blushed pink. */
  sunrise: {
    sunDir: unit(-0.75, -0.65, 0.12),
    zenith: [58, 44, 96],
    fog: [222, 156, 158],
    disc: [255, 214, 170],
    discSize: 0.075,
    brightness: 1.0,
    ambient: 0.88,
    cloud: [255, 196, 208],
    grade: [255, 208, 224],
  },
  /** The BIG orange sun setting into the horizon. */
  sunset: {
    sunDir: unit(-0.62, -0.78, 0.06),
    zenith: [46, 38, 86],
    fog: [232, 150, 104],
    disc: [255, 140, 64],
    discSize: 0.09,
    brightness: 1.0,
    ambient: 0.85,
    cloud: [255, 186, 150],
    grade: [255, 180, 120],
  },
  /** The sun just gone down — violet twilight, a faint afterglow. */
  dusk: {
    sunDir: unit(-0.55, -0.80, -0.10),
    zenith: [24, 28, 64],
    fog: [128, 116, 150],
    disc: [200, 180, 210],
    discSize: 0.035,
    brightness: 0.12,
    ambient: 0.62,
    cloud: [176, 168, 196],
    grade: [238, 172, 192],
  },
  /** The sun not yet up — cool grey, the first hint of warmth. */
  dawn: {
    sunDir: unit(-0.80, -0.58, -0.08),
    zenith: [26, 32, 70],
    fog: [150, 132, 152],
    disc: [235, 205, 215],
    discSize: 0.035,
    brightness: 0.15,
    ambient: 0.66,
    cloud: [216, 196, 212],
    grade: [214, 186, 202],
  },
  /** Full night — no sun, the world lit by its own dark. */
  night: {
    sunDir: unit(-0.60, -0.60, -0.50),
    zenith: [6, 10, 24],
    fog: [18, 26, 46],
    disc: [200, 210, 255],
    discSize: 0.035,
    brightness: 0.0,
    ambient: 0.42,
    cloud: [120, 132, 168],
    grade: [148, 152, 224],
  },
};

/** The page's default mood — noon, the classic look. */
export const DEFAULT_TIME_OF_DAY: TimeOfDay = "noon";
