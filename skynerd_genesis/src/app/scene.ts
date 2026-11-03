/**
 * The scene: every aircraft that exists in the world right now, and the
 * ONE aircraft currently selected. The panel shows the selected
 * aircraft's instruments and controls; the camera follows it; the
 * aircraft dropdown lists them all.
 *
 * The scene holds 1..* aircraft — the flight school's classroom is a
 * scene of two (the instructor's and the student's machine), a drone
 * attack wave is a scene of its eight drones. Whoever is at the controls
 * of an aircraft, the machine in the scene is the same one-class
 * Aircraft (physics/aircraft.ts).
 */
import type { Aircraft, HandDecisions } from "../physics/aircraft.js";

/** One sampled point of an aircraft's flight path. */
export interface SceneSample {
  x: number;
  y: number;
  z: number;
}

/** One aircraft in the scene, as the UI sees it. */
export interface SceneAircraft {
  /** The dropdown label ("Instructor's Aircraft", "Drone 3"…). */
  readonly label: string;
  /** The machine itself — its body, instruments, and controls. */
  readonly aircraft: Aircraft;
  /** The controlling hand's decisions on it (null choices while the
   *  instructor eases its smooth demonstration path). */
  hand: HandDecisions;
  /** The flight path so far, sampled per tick. */
  trail: SceneSample[];
  /** Whether this aircraft is the one being simulated right now. */
  flying: boolean;
  /** Whether it ended its flight dead on the ground. */
  dead: boolean;
}

/** The world right now: all aircraft, one selected. */
export class Scene {
  private readonly entries: SceneAircraft[] = [];
  private selected: SceneAircraft | null = null;

  /** Every aircraft in the scene, in dropdown order. */
  get aircraft(): readonly SceneAircraft[] {
    return this.entries;
  }

  /** The single selected aircraft (the panel's subject). */
  get current(): SceneAircraft | null {
    return this.selected;
  }

  /** Empty the scene (a new scenario begins). */
  clear(): void {
    this.entries.length = 0;
    this.selected = null;
  }

  /** Add one aircraft; the first added becomes the selected one. */
  add(entry: SceneAircraft): SceneAircraft {
    this.entries.push(entry);
    if (this.selected === null) {
      this.selected = entry;
    }
    return entry;
  }

  /** Select an aircraft (the panel and camera follow it). */
  select(entry: SceneAircraft): void {
    if (this.entries.includes(entry)) {
      this.selected = entry;
    }
  }

  /** Replace one aircraft's entry in place — a new machine takes over the
   *  same slot (the next demonstration's aircraft, the next repeat's
   *  machine). The selection follows when the replaced one was selected. */
  replace(old: SceneAircraft, entry: SceneAircraft): SceneAircraft {
    const index = this.entries.indexOf(old);
    if (index < 0) {
      return this.add(entry);
    }
    this.entries[index] = entry;
    if (this.selected === old) {
      this.selected = entry;
    }
    return entry;
  }

  /** Select by dropdown label. */
  selectLabel(label: string): void {
    const entry = this.entries.find((e) => e.label === label);
    if (entry !== undefined) {
      this.selected = entry;
    }
  }

  /** Remove one aircraft from the scene (a crashed drone's site, long
   *  after anyone could still look at it). The selection moves to the
   *  first remaining aircraft, if any. */
  remove(entry: SceneAircraft): void {
    const index = this.entries.indexOf(entry);
    if (index >= 0) {
      this.entries.splice(index, 1);
      if (this.selected === entry) {
        this.selected = this.entries[0] ?? null;
      }
    }
  }
}
