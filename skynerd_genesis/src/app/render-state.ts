/**
 * The render state — the toolbar's sky choices, one mutable record
 * main.ts writes and cockpit.ts reads per frame (the CPU path's twin
 * of the GPU uniform's mood rows). Defaults ARE the classic look, so
 * the headless harness (no toolbar) renders exactly as before.
 */
import { SKY_MOODS, DEFAULT_TIME_OF_DAY,
         type SkyMood, type TimeOfDay } from "../pipeline/index.js";

/** The lens flare families (the dropdown's choices). */
export type LensFlare = "none" | "panavision" | "michaelbay";

export const RENDER_STATE = {
  /** The sky mood the world wears (uniform on GPU, constants on CPU). */
  tod: DEFAULT_TIME_OF_DAY as TimeOfDay,
  /** The mood's unpacked row (kept live so the per-pixel loop never
   *  does a map lookup). */
  mood: SKY_MOODS[DEFAULT_TIME_OF_DAY] as SkyMood,
  /** The flare family drawn over the world, under the instruments. */
  flare: "none" as LensFlare,
  /** The biome checkerboard (the GPU uniform's extras.x). */
  checker: true,
};

/** The dropdown changed: swap the mood. */
export function setTimeOfDay(tod: TimeOfDay): void {
  RENDER_STATE.tod = tod;
  RENDER_STATE.mood = SKY_MOODS[tod];
}
