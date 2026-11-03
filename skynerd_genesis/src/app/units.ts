/**
 * Cockpit display units — the airframe's instrument conventions, now real
 * Cessna: the drones ARE little C172s, so the conversion is the exact SI
 * statute-mile law (1 m/s = 2.23694 mph, 1 m = 3.28084 ft), and the cruise
 * target is 85 mph at 2300 RPM (the C172 economy cruise the lessons aim for).
 *
 * - Speed: mph = m/s * 2.23694. Cruise (38 m/s) reads 85 mph.
 * - Vertical speed and altitude: true conversions (ft, ft/min).
 * - Engine: RPM = IDLE + throttle * (FULL - IDLE); 0 at closed throttle.
 *   Cruise throttle 0.8 reads exactly 2300 RPM.
 *
 * Held in a module of its own so the display law is testable
 * (test/app-trace.test.ts).
 */

/** Cruise speed in m/s — 85 mph, the C172 economy cruise. */
export const CRUISE_MPS = 38.0;
/** mph per m/s — the exact statute conversion. */
export const MPH_PER_MPS = 2.23694;
export const FT_PER_M = 3.28084;           // feet per metre
export const FTPM_PER_MPS = FT_PER_M * 60.0;  // ft/min per m/s

export const IDLE_RPM = 700.0;
export const FULL_THROTTLE_RPM = 2700.0;
export const CRUISE_RPM = 2300.0;          // reads at throttle 0.8
export const CRUISE_THROTTLE = 0.8;
export const RPM_BANDS = [1700.0, 2300.0, 2700.0] as const;

/** Engine RPM from commanded throttle (0..1). 0 at closed throttle. */
export function rpmOf(throttle: number): number {
  if (throttle <= 0.0) {
    return 0.0;
  }
  const clamped = Math.max(0.0, Math.min(1.0, throttle));
  return IDLE_RPM + clamped * (FULL_THROTTLE_RPM - IDLE_RPM);
}

/** Sim m/s to display mph (exact statute conversion). */
export function mphOf(mps: number): number {
  return mps * MPH_PER_MPS;
}

/** Vertical speed in hundreds of ft/min (the VSI's standard units). */
export function vsHundredsOf(vsMps: number): number {
  return (vsMps * FTPM_PER_MPS) / 100.0;
}

/** Altitude in hundreds of feet. */
export function altHundredsOf(altM: number): number {
  return (altM * FT_PER_M) / 100.0;
}

/** Altitude in feet (plain readout). */
export function altFtOf(altM: number): number {
  return altM * FT_PER_M;
}

/** The airspeeds the syllabus names, in m/s (the C172 numbers the user
 *  specified: rotate 60, stall 50, spin pull-up 60, stall dive-out 65). */
export const SPEEDS = {
  rotate: 60.0 / MPH_PER_MPS,
  stall: 50.0 / MPH_PER_MPS,
  spinPullUp: 60.0 / MPH_PER_MPS,
  stallDiveOut: 65.0 / MPH_PER_MPS,
} as const;

/** The altitudes the syllabus names, in m. */
export const ALTITUDES = {
  runwayFt: 10.0,
  climbTo1000Ft: 1000.0 / FT_PER_M,
  stallLessonFt: 3000.0,
  stallLessonM: 3000.0 / FT_PER_M,
} as const;
