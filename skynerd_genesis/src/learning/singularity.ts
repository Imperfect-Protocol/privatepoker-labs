/**
 * THE SINGULARITY TENSOR — the swarm's one shared brain cell
 * (learning/singularity.ts).
 *
 * Besides each drone's own tensors, the swarm maintains ONE MORE
 * TENSOR TOGETHER. Every update round, through the live mesh, each
 * drone blends a lossy trace of what it knows INTO the singularity —
 * and the singularity blends a trace of itself back into every drone
 * it can hear. The contribution is earned, not assumed: only a drone
 * another drone can reach within TWO HOPS (a neighbour, or a
 * neighbour's neighbour) speaks into it — a lone drone is a lone
 * brain for as long as its isolation lasts.
 *
 * The singularity also holds the formation's SUPER-GOAL: the mission
 * leg (learning/mission.ts's tokens) the plurality of the connected
 * swarm is flying. Every drone sharing the tensor conforms to it — a
 * drone that has fallen behind is pulled forward onto the formation's
 * leg at the next safe leg boundary, so the whole carpet attacks,
 * rejoins, and recovers AS ONE.
 *
 * There is no station at the rear: the vessel is only the swarm's
 * agreed-upon COMMON GROUND, and it lives exactly as long as the
 * drones keep speaking into it.
 */
import { NumpyPcg64 } from "../core/rng.js";
import { AiPilot } from "./pilot.js";
import { Flock } from "./flock.js";
import { MISSION_TOKENS, type MissionToken } from "./mission.js";

/** One speaking drone: its own brain, and the mission leg it flies. */
export interface SingularityMember {
  readonly pilot: AiPilot;
  readonly token: MissionToken;
}

/** How much of the speaker's memory one round absorbs into the
 *  vessel, and how much of the vessel one round shares back — a few
 *  keys per neuron, weakly blended: the singularity is maintained, not
 *  transplanted. */
const ABSORB_RATE = 0.3;
const ABSORB_SHARE = 0.05;
const SHARE_RATE = 0.3;
const SHARE_SHARE = 0.05;

/**
 * The singularity: the shared vessel plus the formation's super-goal.
 * One object per swarm (the WavesShow owns it); every round the show
 * hands it the live members and the live mesh, and it keeps the
 * shared brain and the shared intent honest.
 */
export class SingularityTensor {
  /** THE SHARED TENSOR's vessel — a fresh brain jar, seeded empty (the
   *  boot-trained brains arrive inside the drones, and speak into it
   *  from round one). */
  private readonly vessel = AiPilot.fresh(new NumpyPcg64(1n, 1n));
  /** The formation's current super-goal (the mission leg the plurality
   *  flies). */
  superGoal: MissionToken = "takeoff";
  /** Completed update rounds (the UI's bookkeeping). */
  rounds = 0;

  /**
   * ONE UPDATE ROUND. The mesh is the live neighbour pairs as member
   * index pairs [i, j]; a member is HEARD when another member lies
   * within two hops of it. Every heard member speaks into the vessel,
   * the vessel shares back into every heard member, and the
   * super-goal becomes the plurality mission leg of the heard swarm.
   */
  updateRound(members: readonly SingularityMember[],
              mesh: readonly (readonly [number, number])[],
              rng: NumpyPcg64): void {
    this.rounds += 1;
    const heard = heardWithinTwoHops(members.length, mesh);
    // 1. the swarm speaks: every heard member's memory, lossily, into
    //    the one shared vessel.
    for (let i = 0; i < members.length; i += 1) {
      if (heard[i]) {
        Flock.blend(this.vessel, members[i].pilot, rng,
                    ABSORB_RATE, ABSORB_SHARE);
      }
    }
    // 2. the singularity answers: the maintained memory shares back
    //    into every heard member — all drones holding a piece of the
    //    same tensor, acting in formation with it.
    for (let i = 0; i < members.length; i += 1) {
      if (heard[i]) {
        Flock.blend(members[i].pilot, this.vessel, rng,
                    SHARE_RATE, SHARE_SHARE);
      }
    }
    // 3. the super-goal: the mission leg the heard plurality flies.
    this.superGoal = pluralityToken(members, heard);
  }
}

/**
 * Which members any OTHER member can reach within two hops — the
 * swarm's honest audience. A drone with no other drone within two
 * hops is a lone brain this round: it neither speaks nor hears.
 */
function heardWithinTwoHops(count: number,
                            mesh: readonly (readonly [number, number])[]):
    boolean[] {
  if (count <= 1) {
    return new Array<boolean>(count).fill(false);
  }
  const neighbours: boolean[][] = Array.from(
    { length: count }, () => new Array<boolean>(count).fill(false));
  for (const [a, b] of mesh) {
    if (a !== b) {
      neighbours[a][b] = true;
      neighbours[b][a] = true;
    }
  }
  const heard = new Array<boolean>(count).fill(false);
  for (let i = 0; i < count; i += 1) {
    for (let j = 0; j < count; j += 1) {
      if (i === j || !neighbours[i][j]) {
        continue;
      }
      heard[i] = true;                    // a direct neighbour
      for (let k = 0; k < count; k += 1) {
        if (k !== i && neighbours[j][k]) {
          heard[i] = true;               // a neighbour's neighbour
        }
      }
    }
  }
  return heard;
}

/** The mission leg the plurality of the heard swarm flies (ties go to
 *  the earlier leg in the mission's canonical order). A drone in the
 *  mission's retirement (the will phase — its own brain flies it
 *  now) does not vote: the super-goal is the FORMATION's intent. */
function pluralityToken(members: readonly SingularityMember[],
                        heard: readonly boolean[]): MissionToken {
  const counts = new Map<MissionToken, number>();
  for (let i = 0; i < members.length; i += 1) {
    if (!heard[i] || members[i].token === "will") {
      continue;
    }
    counts.set(members[i].token, (counts.get(members[i].token) ?? 0) + 1);
  }
  let best: MissionToken = "takeoff";
  let bestCount = -1;
  for (const token of MISSION_TOKENS) {
    const count = counts.get(token) ?? 0;
    if (count > bestCount) {
      best = token;
      bestCount = count;
    }
  }
  return best;
}
