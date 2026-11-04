/**
 * THE HUD READOUT — one aircraft's flight state as one cockpit line:
 *
 *   55mph, 3000ft, 1200fpm, 225°SW, 10°↑, -20°←
 *
 *  and its delta against the expected state (the instructor's aircraft,
 *  or the drone carpet's mean) as the correction line:
 *
 *   (-15mph, -75ft, -100fpm, -15°SW, -5°↓, -10°→)
 *
 *  The conventions (the pilot's own):
 *   - pitch: ↑ / ↓ — the current value's arrow; the delta is measured
 *     along the CURRENT direction's axis, its arrow shows where the
 *     nose must move (10°↑ held, 5°↑ expected → -5°↓);
 *   - bank: ← / → — the same law sideways (10°← held, 5°← expected →
 *     -5°→);
 *   - heading: N NE E SE S SW W NW — the closest of the eight winds;
 *     the delta's label is the closest wind of the heading the delta
 *     POINTS AT (current + delta), i.e. where the nose should go.
 */
import type { Aircraft } from "../physics/aircraft.js";

const FT_PER_M = 3.28084;
const MPH_PER_MPS = 2.23694;
const FPM_PER_MPS = FT_PER_M * 60.0;
const DEG = 180.0 / Math.PI;

/** The eight winds, clockwise from north. */
const WINDS = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"] as const;

/** One aircraft's six numbers, in the aircraft's own units. */
export interface FlightReadout {
  /** Airspeed, m/s. */
  readonly asi: number;
  /** Altitude, m. */
  readonly altitude: number;
  /** Vertical speed, m/s (positive up). */
  readonly vsi: number;
  /** Heading, rad from north, turning right positive. */
  readonly heading: number;
  /** Deck pitch, rad positive nose up. */
  readonly pitch: number;
  /** Bank, rad positive right wing down. */
  readonly bank: number;
}

/** The two-line HUD readout: the current line, and the correction line
 *  (its parens belong to the caller). */
export interface HudDelta {
  readonly currentLine: string;
  readonly deltaLine: string;
}

/** Read one aircraft's six numbers. */
export function readoutOf(aircraft: Aircraft): FlightReadout {
  return {
    asi: aircraft.asi,
    altitude: aircraft.altitude,
    vsi: aircraft.vsi,
    heading: aircraft.heading,
    pitch: aircraft.pitch,
    bank: aircraft.bank,
  };
}

/** The mean of many aircraft's six numbers — the drone carpet's own
 *  state. The heading is the CIRCULAR mean (the mean of the winds);
 *  everything else is arithmetic. */
export function meanReadoutOf(aircrafts: readonly Aircraft[]): FlightReadout | null {
  if (aircrafts.length === 0) {
    return null;
  }
  let asi = 0.0, altitude = 0.0, vsi = 0.0, sinH = 0.0, cosH = 0.0;
  let pitch = 0.0, bank = 0.0;
  for (const aircraft of aircrafts) {
    asi += aircraft.asi;
    altitude += aircraft.altitude;
    vsi += aircraft.vsi;
    sinH += Math.sin(aircraft.heading);
    cosH += Math.cos(aircraft.heading);
    pitch += aircraft.pitch;
    bank += aircraft.bank;
  }
  const n = aircrafts.length;
  let heading = Math.atan2(sinH, cosH);
  if (heading < 0.0) {
    heading += 2.0 * Math.PI;
  }
  return {
    asi: asi / n, altitude: altitude / n, vsi: vsi / n,
    heading, pitch: pitch / n, bank: bank / n,
  };
}

/** The closest of the eight winds, for a heading in radians. */
function windOf(headingRad: number): string {
  const deg = ((headingRad * DEG) % 360.0 + 360.0) % 360.0;
  return WINDS[Math.round(deg / 45.0) % 8];
}

/** The current line's pitch cell: `10°↑` / `-5°↓`. */
function pitchNow(pitchRad: number): string {
  const deg = pitchRad * DEG;
  return `${Math.round(deg)}°${deg < 0.0 ? "↓" : "↑"}`;
}

/** The current line's bank cell: `-20°←` / `15°→`. */
function bankNow(bankRad: number): string {
  const deg = bankRad * DEG;
  return `${Math.round(deg)}°${deg < 0.0 ? "←" : "→"}`;
}

/** One vertical axis's delta cell (pitch, bank): the delta is measured
 *  along the CURRENT direction's own axis, the arrow shows where the
 *  machine must move — 10°↑ held against 5°↑ expected is -5°↓, and
 *  10°← held against 5°← expected is -5°→. */
function axisDelta(current: number, expected: number,
                   posArrow: string, negArrow: string): string {
  const axis = current >= 0.0 ? 1.0 : -1.0;   // the current direction's sign
  const delta = (expected - current) * axis;  // in the current axis's units
  const currentArrow = axis > 0.0 ? posArrow : negArrow;
  const oppositeArrow = axis > 0.0 ? negArrow : posArrow;
  const arrow = delta < 0.0 ? oppositeArrow : currentArrow;
  return `${Math.round(delta * DEG)}°${arrow}`;
}

/** The heading delta: signed shortest way round, labelled with the
 *  wind the delta points at (current + delta = where the nose goes). */
function headingDelta(current: number, expected: number): string {
  let delta = expected - current;             // rad, the short way
  while (delta > Math.PI) {
    delta -= 2.0 * Math.PI;
  }
  while (delta < -Math.PI) {
    delta += 2.0 * Math.PI;
  }
  const target = current + delta;
  return `${Math.round(delta * DEG)}°${windOf(target)}`;
}

/** The two HUD lines of one aircraft against its expected state. */
export function hudDelta(current: FlightReadout,
                         expected: FlightReadout): HudDelta {
  const spd = Math.round(current.asi * MPH_PER_MPS);
  const alt = Math.round(current.altitude * FT_PER_M);
  const fpm = Math.round(current.vsi * FPM_PER_MPS);
  const hdg = Math.round(((current.heading * DEG) % 360.0 + 360.0) % 360.0);
  const currentLine =
    `${spd}mph, ${alt}ft, ${fpm}fpm, ${hdg}°${windOf(current.heading)}, ` +
    `${pitchNow(current.pitch)}, ${bankNow(current.bank)}`;

  const dSpd = Math.round((expected.asi - current.asi) * MPH_PER_MPS);
  const dAlt = Math.round((expected.altitude - current.altitude) * FT_PER_M);
  const dFpm = Math.round((expected.vsi - current.vsi) * FPM_PER_MPS);
  const deltaLine =
    `${dSpd}mph, ${dAlt}ft, ${dFpm}fpm, ` +
    `${headingDelta(current.heading, expected.heading)}, ` +
    `${axisDelta(current.pitch, expected.pitch, "↑", "↓")}, ` +
    `${axisDelta(current.bank, expected.bank, "→", "←")}`;
  return { currentLine, deltaLine };
}
