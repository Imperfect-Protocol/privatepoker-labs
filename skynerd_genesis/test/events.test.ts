/**
 * Event queue, stun timer, and signal envelope tests. Determinism contract:
 * same-time events dispatch in scheduling order (FIFO), cancellable timers
 * never fire after re-arm or disarm.
 */
import { createHash } from "node:crypto";

import { EventQueue, Message, StunTimer } from "../src/net/events.js";
import { Signal, addressForNode, BEACON, ANNOUNCE } from "../src/net/signal.js";
import { check, finish, section } from "./harness.js";

const sha256 = (data: string): Uint8Array =>
  new Uint8Array(createHash("sha256").update(data).digest());

section("event queue ordering");
{
  const q = new EventQueue();
  const order: string[] = [];
  q.schedule(2.0, () => order.push("late"));
  q.schedule(1.0, () => order.push("mid"));
  q.schedule(0.0, () => order.push("now"));
  check("next_time is earliest", q.nextTime() === 0.0);
  q.run();
  check("dispatched in time order", order.join(",") === "now,mid,late", order.join(","));
  check("now advanced to last event", q.step() === false && q.nextTime() === null);
}

section("event queue FIFO tie-breaking");
{
  const q = new EventQueue();
  const order: number[] = [];
  for (let i = 0; i < 5; i++) {
    q.schedule(1.0, () => order.push(i));
  }
  q.run();
  check("same-time events dispatch in scheduling order",
        order.join(",") === "0,1,2,3,4", order.join(","));
}

section("cancellation with lazy deletion");
{
  const q = new EventQueue();
  let fired = 0;
  const id = q.schedule(1.0, () => { fired += 1; }, true);
  check("cancellable schedule returns an id", id !== null);
  q.cancel(id as number);
  q.run();
  check("cancelled event never fires", fired === 0);
  check("lazy deletion advances next_time to null", q.nextTime() === null);

  const q2 = new EventQueue();
  let a = 0;
  let b = 0;
  const idA = q2.schedule(1.0, () => { a += 1; }, true);
  q2.schedule(1.0, () => { b += 1; });
  q2.cancel(idA as number);
  q2.run();
  check("only the cancelled twin is dropped", a === 0 && b === 1);
}

section("stun timer re-arm replaces");
{
  const q = new EventQueue();
  let fires = 0;
  const timer = new StunTimer(q, () => { fires += 1; });
  timer.arm(5.0);
  check("fire_at is scheduled", timer.fireAt === 5.0);
  timer.arm(2.0);
  q.run();
  check("re-arm cancels the first timer; exactly one fire", fires === 1, `fires=${fires}`);
  check("fired at the re-armed time", q.now === 2.0);

  const q2 = new EventQueue();
  let fires2 = 0;
  const timer2 = new StunTimer(q2, () => { fires2 += 1; });
  timer2.arm(3.0);
  timer2.disarm();
  q2.run();
  check("disarmed timer never fires", fires2 === 0);
}

section("signals: identity law and validation");
{
  const addr = addressForNode(7, sha256);
  check("address is 20 bytes", addr.length === 20);
  check("address is deterministic", Buffer.from(addressForNode(7, sha256)).equals(Buffer.from(addr)));
  check("different node, different address",
        !Buffer.from(addressForNode(8, sha256)).equals(Buffer.from(addr)));

  const beacon = new Signal({
    kind: BEACON, originId: 1, originAddress: addressForNode(1, sha256), sequenceId: 3,
  });
  const beaconRetry = new Signal({
    kind: BEACON, originId: 1, originAddress: addressForNode(1, sha256), sequenceId: 3,
  });
  check("beacon retries share a dedup key", beacon.dedupKey === beaconRetry.dedupKey);
  const beaconNext = new Signal({
    kind: BEACON, originId: 1, originAddress: addressForNode(1, sha256), sequenceId: 4,
  });
  check("new beacon sequence is a new request", beacon.dedupKey !== beaconNext.dedupKey);

  const announce = new Signal({
    kind: ANNOUNCE, originId: 1, originAddress: addressForNode(1, sha256),
    sequenceId: 9, targetId: 2, targetAddress: addressForNode(2, sha256),
  });
  check("announce needs no exception", announce.isAlive);
  let threw = false;
  try {
    new Signal({ kind: ANNOUNCE, originId: 1, originAddress: addr, sequenceId: 10 });
  } catch {
    threw = true;
  }
  check("announce without a target throws", threw);

  const relayed = announce.relayed();
  check("relay increments hop, decrements ttl",
        relayed.hopCount === 1 && relayed.ttl === 7);
  check("relay keeps the dedup key", relayed.dedupKey === announce.dedupKey);
}

section("message envelope");
{
  const now = 1.5;
  const s1 = new Signal({ kind: BEACON, originId: 1, originAddress: addressForNode(1, sha256), sequenceId: 1 });
  const s2 = new Signal({ kind: BEACON, originId: 1, originAddress: addressForNode(1, sha256), sequenceId: 2 });
  const m = new Message(1, 2, [s1, s2], now);
  check("bundle keeps signal order", m.dedupKeys.length === 2 && m.dedupKeys[0] === s1.dedupKey);
  check("envelope records sender, recipient, time",
        m.senderId === 1 && m.recipientId === 2 && m.sentAt === now);
}

finish("events.test");
