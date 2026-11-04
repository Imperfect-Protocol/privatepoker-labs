/**
 * Camera — the scene's first-class citizen (LightAndGeometry's Camera,
 * plan §6): position + orientation axes + focal length, owning its
 * view-projection matrix. `adopt` copies one frame's pinhole data in
 * (the cockpit's cameraOf math — heading, pitch, bank axes); the VP
 * matrix and the projection helpers are THIS class's behaviour.
 *
 * The per-frame VP build is one-off CPU math (the plan's law: massive
 * math on GPU, one-offs on CPU); everything downstream — terrain quad
 * generation, the tile index, every vertex — reads the one uniform.
 */

/** One frame's pinhole data — structurally identical to the cockpit's
 *  Camera, so the cockpit's own object passes straight through. */
export interface ViewCamera {
  px: number; py: number; pz: number;
  fx: number; fy: number; fz: number;   // forward, unit
  ux: number; uy: number; uz: number;   // up, unit
  rx: number; ry: number; rz: number;   // right, unit
  f: number;                             // focal length, px
  cx: number; cy: number;                // viewport centre, px
  W: number; H: number;                  // viewport size, px
}

export class Camera {
  private readonly vp = new Float32Array(16);
  private px = 0; private py = 0; private pz = 0;
  private W = 1; private H = 1;

  /** Take one frame's pinhole data (mutating adopt — no per-frame
   *  allocation) and rebuild the view-projection matrix. */
  adopt(cam: ViewCamera, focalPx: number): void {
    this.px = cam.px; this.py = cam.py; this.pz = cam.pz;
    this.W = Math.max(2, Math.ceil(cam.W));
    this.H = Math.max(2, Math.ceil(cam.H));
    const f = focalPx;
    const near = 0.2;                      // the CPU path's depth cut
    const far = 24000.0;                   // the world's draw distance K
    // view rows (row-major): the axes as rows, translation folded in
    const r = [cam.rx, cam.ry, cam.rz];
    const u = [cam.ux, cam.uy, cam.uz];
    const m = [cam.fx, cam.fy, cam.fz];        // +forward: depth-positive
    const rt = -(r[0] * cam.px + r[1] * cam.py + r[2] * cam.pz);
    const ut = -(u[0] * cam.px + u[1] * cam.py + u[2] * cam.pz);
    const mt = -(m[0] * cam.px + m[1] * cam.py + m[2] * cam.pz);
    // projection (row-major): ndc.x = x·(2f/W)/depth, ndc.y = y·(2f/H)/depth
    const p00 = (2.0 * f) / this.W;
    const p11 = (2.0 * f) / this.H;
    const p22 = far / (far - near);            // depth [near, far] -> [0, 1]
    const p23 = (-far * near) / (far - near);
    // mvp = P · V, then stored COLUMN-major (WGSL mat4x4f layout)
    const v = [                       // V, row-major
      r[0], r[1], r[2], rt,
      u[0], u[1], u[2], ut,
      m[0], m[1], m[2], mt,
      0, 0, 0, 1,
    ];
    const p = [                       // P, row-major
      p00, 0, 0, 0,
      0, p11, 0, 0,
      0, 0, p22, p23,
      0, 0, 1, 0,
    ];
    for (let col = 0; col < 4; col++) {
      for (let row = 0; row < 4; row++) {
        let s = 0.0;
        for (let k = 0; k < 4; k++) {
          s += p[row * 4 + k] * v[k * 4 + col];
        }
        this.vp[col * 4 + row] = s;
      }
    }
  }

  get position(): readonly [number, number, number] {
    return [this.px, this.py, this.pz];
  }

  get width(): number {
    return this.W;
  }

  get height(): number {
    return this.H;
  }

  /** The column-major VP matrix for the Frame uniform. */
  get viewProj(): Float32Array {
    return this.vp;
  }
}
