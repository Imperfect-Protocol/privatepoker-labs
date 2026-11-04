/** Config defaults vs the Python reference; clock contract. */
import { DEFAULT_CONFIG, derived, replace, CLOCKS } from "../src/core/config.js";
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { dirname, join } from "node:path";
import { check, finish, section } from "./harness.js";

const baselinePath = join(dirname(fileURLToPath(import.meta.url)), "..", "..", "reference", "golden-baseline.json");
const baseline = JSON.parse(readFileSync(baselinePath, "utf8"));

section("config vs python golden baseline");
{
  const py = baseline.config;
  const pairs: [string, number, number][] = [
    ["seed", DEFAULT_CONFIG.seed, py.seed],
    ["episodeSeconds", DEFAULT_CONFIG.episodeSeconds, py.episode_seconds],
    ["worldHalfExtent", DEFAULT_CONFIG.worldHalfExtent, py.world_half_extent],
    ["physicsHz", DEFAULT_CONFIG.physicsHz, py.physics_hz],
    ["motionHz", DEFAULT_CONFIG.motionHz, py.motion_hz],
    ["visionHz", DEFAULT_CONFIG.visionHz, py.vision_hz],
    ["radioRange", DEFAULT_CONFIG.radioRange, py.radio_range],
    ["radioDropExponent", DEFAULT_CONFIG.radioDropExponent, py.radio_drop_exponent],
    ["radioDelay", DEFAULT_CONFIG.radioDelay, py.radio_delay],
    ["initialTtl", DEFAULT_CONFIG.initialTtl, py.initial_ttl],
    ["stunK", DEFAULT_CONFIG.stunK, py.stun_k],
    ["stunMidpoint", DEFAULT_CONFIG.stunMidpoint, py.stun_midpoint],
    ["beaconInterval", DEFAULT_CONFIG.beaconInterval, py.beacon_interval],
    ["angleBins", DEFAULT_CONFIG.angleBins, py.angle_bins],
    ["hopBins", DEFAULT_CONFIG.hopBins, py.hop_bins],
    ["timeDepth", DEFAULT_CONFIG.timeDepth, py.time_depth],
    ["hebbianBaseline", DEFAULT_CONFIG.hebbianBaseline, py.hebbian_baseline],
    ["hebbianMaxWeight", DEFAULT_CONFIG.hebbianMaxWeight, py.hebbian_max_weight],
    ["hebbianReward", DEFAULT_CONFIG.hebbianReward, py.hebbian_reward],
    ["beaconReward", DEFAULT_CONFIG.beaconReward, py.beacon_reward],
    ["deathRewardScale", DEFAULT_CONFIG.deathRewardScale, py.death_reward_scale],
    ["mirrorHopDecay", DEFAULT_CONFIG.mirrorHopDecay, py.mirror_hop_decay],
    ["stationMergeRate", DEFAULT_CONFIG.stationMergeRate, py.station_merge_rate],
    ["servoStateBins", DEFAULT_CONFIG.servoStateBins, py.servo_state_bins],
    ["servoSurfaces", DEFAULT_CONFIG.servoSurfaces, py.servo_surfaces],
    ["servoRewardScale", DEFAULT_CONFIG.servoRewardScale, py.servo_reward_scale],
    ["eyeBins", DEFAULT_CONFIG.eyeBins, py.eye_bins],
    ["eyeMaxSources", DEFAULT_CONFIG.eyeMaxSources, py.eye_max_sources],
    ["eyeRangeMax", DEFAULT_CONFIG.eyeRangeMax, py.eye_range_max],
    ["noiseStrength", DEFAULT_CONFIG.noiseStrength, py.noise_strength],
    ["gaussianWidth", DEFAULT_CONFIG.gaussianWidth, py.gaussian_width],
    ["messageThreshold", DEFAULT_CONFIG.messageThreshold, py.message_threshold],
    ["nodeThreshold", DEFAULT_CONFIG.nodeThreshold, py.node_threshold],
    ["waveSize", DEFAULT_CONFIG.waveSize, py.wave_size],
    ["waveRows", DEFAULT_CONFIG.waveRows, py.wave_rows],
    ["waveGap", DEFAULT_CONFIG.waveGap, py.wave_gap],
    ["rowSpacing", DEFAULT_CONFIG.rowSpacing, py.row_spacing],
  ];
  for (const [name, ts, pyv] of pairs) {
    check(`${name} matches python`, ts === pyv, `${ts} vs ${pyv}`);
  }
  check("ablation switches all false",
        !DEFAULT_CONFIG.noLearning && !DEFAULT_CONFIG.noNoise && !DEFAULT_CONFIG.noHaar &&
        !DEFAULT_CONFIG.noGaussian && !DEFAULT_CONFIG.noCompression && !DEFAULT_CONFIG.noTtl &&
        !DEFAULT_CONFIG.noTwist);
}

section("derived clocks and replace");
{
  const d = derived();
  check("physics dt = 1/50", d.physicsDt === 0.02);
  check("motion dt = 1/10", d.motionDt === 0.1);
  check("vision dt = 1/2", d.visionDt === 0.5);
  const variant = replace(DEFAULT_CONFIG, { seed: 7, waveSize: 16 });
  check("replace changes only the given knobs", variant.seed === 7 && variant.waveSize === 16
        && variant.physicsHz === DEFAULT_CONFIG.physicsHz);
  check("default config is frozen", (() => {
    try {
      (DEFAULT_CONFIG as { seed?: number }).seed = 1;
      return false;
    } catch {
      return true;
    }
  })());
}

section("clock contract vs golden baseline");
{
  check("physics_hz", CLOCKS.physicsHz === baseline.clocks.physics_hz);
  check("physics_dt", CLOCKS.physicsDt === baseline.clocks.physics_dt);
  check("decision window", CLOCKS.decisionWindowS === baseline.clocks.decision_window_s);
  check("episode horizon", CLOCKS.episodeHorizonS === baseline.clocks.episode_horizon_s);
  check("motion_hz", CLOCKS.motionHz === baseline.clocks.motion_hz);
  check("vision_hz", CLOCKS.visionHz === baseline.clocks.vision_hz);
}

finish("config.test");
