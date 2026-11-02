/**
 * Cockpit display units — the airframe's instrument conventions, mirrored
 * from the Python reference (`skynerd/learning/panel.py`):
 *
 * - Speed: the airframe maps C172 numbers onto sim units — cruise 80 mph IS
 *   10.5 m/s, stall 60 mph is 0.75 cruise, Vne ~163 mph is about 2.0 cruise.
 *   So the mph scale is the panel law: mph = m/s * (80 / 10.5).
 * - Vertical speed and altitude: true conversions (FT_PER_M, panel law).
 * - Engine: RPM = IDLE + throttle * (FULL - IDLE); 0 at closed throttle.
 *
 * Held in a module of its own so the display law is testable against the
 * reference conventions (test/app-trace.test.ts).
 */

/** Cruise speed in sim units: 80 mph mapped onto the airframe (m/s). */
export const CRUISE_MPS = 10.5;
/** mph per m/s — the airframe's C172 mapping (80 mph at cruise). */
export const MPH_PER_MPS = 80.0 / CRUISE_MPS;
export const FT_PER_M = 3.28084;           // feet per metre (panel law)
export const FTPM_PER_MPS = FT_PER_M * 60.0;  // ft/min per m/s (panel law)

export const IDLE_RPM = 700.0;
export const FULL_THROTTLE_RPM = 2700.0;
export const RPM_BANDS = [1700.0, 2300.0, 2700.0] as const;

/** Engine RPM from commanded throttle (0..1). 0 at closed throttle. */
export function rpmOf(throttle: number): number {
  if (throttle <= 0.0) {
    return 0.0;
  }
  const clamped = Math.max(0.0, Math.min(1.0, throttle));
  return IDLE_RPM + clamped * (FULL_THROTTLE_RPM - IDLE_RPM);
}

/** Sim m/s to display mph (C172 flavor scale). */
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
