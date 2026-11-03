/**
 * Unit tests for the panel's conversion laws (src/app/units.ts): the
 * cockpit's mph / x100 fpm / feet / RPM readings are fixed display law,
 * not up for drift.
 */
import { mphOf, vsHundredsOf, altFtOf, rpmOf, FULL_THROTTLE_RPM,
} from "../src/app/units.js";
import { check, finish, section } from "./harness.js";

section("cockpit units (panel law)");
{
  check("cruise 38 m/s reads 85 mph", mphOf(38.0) === 85.00372);
  check("trim throttle 0.15 reads 1000 rpm", rpmOf(0.15) === 1000.0);
  check("cruise throttle 0.8 reads 2300 rpm", rpmOf(0.8) === 2300.0);
  check("closed throttle reads 0 rpm", rpmOf(0.0) === 0.0);
  check("full throttle reads 2700 rpm", rpmOf(1.0) === FULL_THROTTLE_RPM);
  check("1 m/s climb reads 1.9685 x100 fpm",
        Math.abs(vsHundredsOf(1.0) - 1.968504) < 1e-6);
  check("50 m altitude reads 164.042 ft", Math.abs(altFtOf(50.0) - 164.042) < 1e-3);
}

finish("units.test");
