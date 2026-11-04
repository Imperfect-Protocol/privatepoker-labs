/**
 * Event loop and bundled messages: the transport skeleton.
 * Ported from `skynerd/net/simulation.py` (Genesis Milestone 1).
 *
 * A Message is a transport envelope carrying multiple Signals, delivered at a
 * simulated time. The event queue is a lazy-deletion binary heap; timers are
 * cancellable entries. Tie-breaking matches the Python heap exactly: entries
 * carry an insertion sequence from the same counter that mints cancellable
 * entry ids, so same-time events dispatch in scheduling order.
 *
 * Stun coalescing lives in Drone (a later milestone): handlers never send;
 * they stage signals and arm the node's single stun timer. On fire, the node
 * picks recipients and emits at most one bundled Message per neighbour.
 */
import { Signal } from "./signal.js";

export class Message {
  readonly senderId: number;
  readonly recipientId: number;
  readonly signals: readonly Signal[];
  readonly sentAt: number;

  constructor(senderId: number, recipientId: number, signals: readonly Signal[], sentAt: number) {
    this.senderId = senderId;
    this.recipientId = recipientId;
    this.signals = signals;
    this.sentAt = sentAt;
  }

  get dedupKeys(): string[] {
    return this.signals.map((s) => s.dedupKey);
  }
}

type Callback = () => void;

interface HeapEntry {
  time: number;
  seq: number;
  entryId: number | null;
  callback: Callback;
}

/** Time-ordered, cancellable event queue with lazy deletion. */
export class EventQueue {
  private heap: HeapEntry[] = [];
  private counter = 0;
  private readonly cancelled = new Set<number>();
  now = 0.0;
  dispatched = 0;

  /** Schedule `callback` at now + delay. Cancellable entries get an id. */
  schedule(delay: number, callback: Callback, cancellable = false): number | null {
    const entryId = cancellable ? this.counter++ : null;
    const seq = this.counter++;
    this.push({
      time: this.now + Math.max(0.0, delay),
      seq,
      entryId,
      callback,
    });
    return entryId;
  }

  cancel(entryId: number): void {
    this.cancelled.add(entryId);
  }

  nextTime(): number | null {
    while (this.heap.length > 0) {
      const top = this.heap[0];
      if (top.entryId !== null && this.cancelled.has(top.entryId)) {
        this.pop();
        continue;
      }
      return top.time;
    }
    return null;
  }

  /** Dispatch the next event. Returns false when the queue is empty. */
  step(): boolean {
    while (this.heap.length > 0) {
      const entry = this.pop();
      if (entry.entryId !== null && this.cancelled.has(entry.entryId)) {
        continue;
      }
      this.now = entry.time;
      this.dispatched += 1;
      entry.callback();
      return true;
    }
    return false;
  }

  run(maxEvents = 100000): number {
    let events = 0;
    while (events < maxEvents && this.step()) {
      events += 1;
    }
    return events;
  }

  // --- binary heap (min by time, then seq) ---------------------------------

  private push(entry: HeapEntry): void {
    this.heap.push(entry);
    let i = this.heap.length - 1;
    while (i > 0) {
      const parent = (i - 1) >> 1;
      if (this.before(this.heap[i], this.heap[parent])) {
        [this.heap[i], this.heap[parent]] = [this.heap[parent], this.heap[i]];
        i = parent;
      } else {
        break;
      }
    }
  }

  private pop(): HeapEntry {
    const top = this.heap[0];
    const last = this.heap.pop();
    if (this.heap.length > 0 && last !== undefined) {
      this.heap[0] = last;
      let i = 0;
      for (;;) {
        const left = 2 * i + 1;
        const right = 2 * i + 2;
        let smallest = i;
        if (left < this.heap.length && this.before(this.heap[left], this.heap[smallest])) {
          smallest = left;
        }
        if (right < this.heap.length && this.before(this.heap[right], this.heap[smallest])) {
          smallest = right;
        }
        if (smallest === i) {
          break;
        }
        [this.heap[i], this.heap[smallest]] = [this.heap[smallest], this.heap[i]];
        i = smallest;
      }
    }
    return top;
  }

  private before(a: HeapEntry, b: HeapEntry): boolean {
    if (a.time !== b.time) {
      return a.time < b.time;
    }
    return a.seq < b.seq;
  }
}

/** One live stun timer per node; re-arming replaces it (cancellable). */
export class StunTimer {
  private readonly queue: EventQueue;
  private readonly onFire: Callback;
  fireAt: number | null = null;
  private entryId: number | null = null;

  constructor(queue: EventQueue, onFire: Callback) {
    this.queue = queue;
    this.onFire = onFire;
  }

  arm(delay: number): void {
    if (this.entryId !== null) {
      this.queue.cancel(this.entryId);
    }
    this.fireAt = this.queue.now + delay;
    this.entryId = this.queue.schedule(delay, this.onFire, true);
  }

  disarm(): void {
    if (this.entryId !== null) {
      this.queue.cancel(this.entryId);
    }
    this.entryId = null;
    this.fireAt = null;
  }
}
