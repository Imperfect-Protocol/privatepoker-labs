/**
 * THE ERROR NEURON — the local learning unit, one per tensor.
 *
 * It observes ONE tensor's error: what the tensor output (Y) against
 * what the expected output was (Y'). When the error is nonzero it
 * fires a DELTA into that same tensor — a pump on the tensor's own
 * cells, never a gradient through the network. The pump is the
 * idiotic gradient: a signed delta, proportional to the error.
 *
 * THE SIGN LAW: the delta rides an input weight that starts positive.
 * When the previous pump left the error BIGGER than the one it aimed
 * at, the weight of that error input FLIPS to negative — the error
 * neuron learns which direction its own advice pushes. This is the
 * whole mechanism: observe, pump, check, flip. Nothing else.
 *
 * (The biology agrees this is enough: cerebellar climbing fibers
 * carry a graded error signal onto the very Purkinje cells that
 * erred — the correction is local to one neuron's synapses, not a
 * network-wide backpropagation; and dopamine prediction-error
 * neurons broadcast the same shape: a signed delta on the cells that
 * caused the surprise.)
 */
export const TEACH_ETA = 1.0;

export class ErrorNeuron {
  /** The error input's weight sign: +1 or −1. Starts positive; flips
   *  when a pump left the error bigger than the one it aimed at. */
  private sign = 1;
  /** The last observed error magnitude (0 = the tensor was right). */
  private lastError = 0.0;
  /** Whether the last observation was followed by a pump (the flip law
   *  only judges pumps, not quiet windows). */
  private pumpedLastTime = false;
  /** How many pumps the neuron has fired (the log's bookkeeping). */
  pumps = 0;
  /** How many times the sign flipped (the log's bookkeeping). */
  flips = 0;

  /**
   * Observe the tensor's CURRENT error and decide the pump. THE SIGN
   * LAW runs first: if the last pump left the error BIGGER than the
   * one it aimed at, the error input's weight flips — the next pump
   * pushes the other way. Then the pump is the signed, proportional
   * delta. A zero error pumps nothing (and clears the pump memory).
   */
  pump(error: number, eta: number = TEACH_ETA): number {
    const magnitude = Math.abs(error);
    if (this.pumpedLastTime && magnitude > this.lastError + 1e-9) {
      this.sign = -this.sign;
      this.flips += 1;
    }
    this.lastError = magnitude;
    if (magnitude <= 1e-9) {
      this.pumpedLastTime = false;
      return 0.0;
    }
    this.pumpedLastTime = true;
    this.pumps += 1;
    return this.sign * eta * magnitude;
  }

  /** The neuron's current sign (for the log, the tests, and cloning). */
  get signNow(): number {
    return this.sign;
  }

  /** Restore a sign (a clone inheriting its parent's learned
   *  direction). */
  restoreSign(sign: number): void {
    this.sign = sign < 0 ? -1 : 1;
  }

  /** Reset the observation memory (a fresh flight; the sign stays). */
  resetWindow(): void {
    this.lastError = 0.0;
    this.pumpedLastTime = false;
  }
}
