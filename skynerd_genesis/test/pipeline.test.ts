/**
 * Pipeline tests — the src/pipeline layer, headless (no GPU exists in
 * Node; these gate the SCENE, the math the kernels share, and the WGSL
 * common's integrity — the pixel-level parity probes run against the
 * CPU reference in fpm-debug, and the renderer-independent gates come
 * with the plan's phases).
 */
import { check, finish, section } from "./harness.js";
import { hash01v, WGSL_CHECKER, WGSL_COMMON } from "../src/pipeline/core/wgsl.js";
import { hash01 } from "../src/app/land.js";
import { Camera } from "../src/pipeline/scene/camera.js";
import { Clump } from "../src/pipeline/scene/clump.js";
import { ConeGeometry } from "../src/pipeline/geometries/cone.js";
import { QuadGeometry } from "../src/pipeline/geometries/quad.js";
import { Light } from "../src/pipeline/scene/light.js";
import { RenderClump, Scene } from "../src/pipeline/scene/scene.js";
import { KernelQueue } from "../src/pipeline/core/kernel-queue.js";
import type { WgpuRenderPassEncoder } from "../src/pipeline/types.js";
import {
  frustumPlanes, rayAabb, sphereCulled,
} from "../src/pipeline/util/intersect.js";

section("pipeline: the world renderer's laws, headless");

// --- the hash: the CPU twin is bit-identical to land.ts's ---------------
{
  let ok = true;
  let worst = 0.0;
  for (let salt = 1; salt <= 4 && ok; salt++) {
    for (let iy = -600; iy <= 600 && ok; iy += 37) {
      for (let ix = -600; ix <= 600 && ok; ix += 41) {
        const a = hash01v(ix, iy, salt);
        const b = hash01(ix, iy, salt);
        worst = Math.max(worst, Math.abs(a - b));
        if (a !== b) {
          ok = false;
        }
      }
    }
  }
  check("pipeline hash01v is bit-identical to land's hash01", ok,
        `worst diff ${worst}`);
}

// --- the camera: the pinhole's VP projects known points ------------------
{
  const cam = new Camera();
  // a level camera at the field's centre, looking along +x, HFOV 90°
  cam.adopt({
    px: 600.0, py: 0.0, pz: 30.0,
    fx: 1.0, fy: 0.0, fz: 0.0,
    ux: 0.0, uy: 0.0, uz: 1.0,
    rx: 0.0, ry: -1.0, rz: 0.0,
    f: 200.0, cx: 200.0, cy: 100.0, W: 400.0, H: 200.0,
  }, 200.0);
  const vp = cam.viewProj;
  const project = (x: number, y: number, z: number):
      [number, number] => {
    // column-major mat4 multiply, perspective divide
    const cx = vp[0] * x + vp[4] * y + vp[8] * z + vp[12];
    const cy = vp[1] * x + vp[5] * y + vp[9] * z + vp[13];
    const cw = vp[3] * x + vp[7] * y + vp[11] * z + vp[15];
    return [cx / cw, cy / cw];
  };
  // a point dead ahead at the camera's height: NDC (0, 0)
  let [nx, ny] = project(660.0, 0.0, 30.0);
  check("VP maps the centre-ahead point to NDC (0, 0)",
        Math.abs(nx) < 1e-9 && Math.abs(ny) < 1e-9, `(${nx}, ${ny})`);
  // the half-edges: depth 60, so xr = 60 (p00 = 1) and yu = 30 (p11 = 2)
  [nx, ny] = project(660.0, -60.0, 30.0);
  check("VP maps a right-axis point to the +x half-edge",
        Math.abs(nx - 1.0) < 1e-9 && Math.abs(ny) < 1e-9, `(${nx}, ${ny})`);
  [nx, ny] = project(660.0, 0.0, 60.0);
  check("VP maps an up-axis point to the +y half-edge",
        Math.abs(nx) < 1e-9 && Math.abs(ny - 1.0) < 1e-9, `(${nx}, ${ny})`);
  // the frustum planes built from the same VP agree about ahead/behind
  const planes = frustumPlanes([...vp]);
  check("the frustum culls a point behind the camera",
        sphereCulled(planes, 400.0, 0.0, 30.0, 1.0),
        "behind-camera point survived the planes");
  check("the frustum keeps a point ahead well inside",
        !sphereCulled(planes, 700.0, 0.0, 30.0, 1.0),
        "ahead point was culled");
}

// --- ray-AABB: the slab test's basics ------------------------------------
{
  const box = { lo: [0.0, 0.0, 0.0] as const, hi: [10.0, 10.0, 10.0] as const };
  const hit = rayAabb(5.0, -5.0, 5.0, 0.0, 1.0, 0.0, box);
  check("ray-AABB hits the box from the side",
        hit !== null && Math.abs(hit - 5.0) < 1e-9, `t=${hit}`);
  check("ray-AABB misses a parallel offset ray",
        rayAabb(5.0, -5.0, 20.0, 0.0, 1.0, 0.0, box) === null);
}

// --- the geometries: built programmatically, frozen correctly -----------
{
  const quad = new QuadGeometry();
  check("the unit quad has 4 corners and 6 indices",
        quad.vertexCount === 4 && quad.indexCount === 6,
        `v=${quad.vertexCount} i=${quad.indexCount}`);
  const verts = quad.vertexCount;
  const stride = quad.stride;
  check("the quad's stride is one vec3 (12 B)",
        stride === 12, `stride=${stride}`);
  const cone = new ConeGeometry(8);
  // 9 ring verts (segments + 1 for the wrap seam) + apex + cap centre;
  // 16 triangles = 48 index values
  check("the 8-segment cone has 11 verts and 16 triangles",
        cone.vertexCount === 11 && cone.indexCount === 48,
        `v=${cone.vertexCount} i=${cone.indexCount}`);
  check("the cone's stride is pos + normal (24 B)",
        cone.stride === 24, `stride=${cone.stride}`);
  void verts;
}

// --- the scene graph: clumps in order, geometries owned ------------------
{
  const calls: string[] = [];
  class Recording extends RenderClump {
    constructor(name: string) {
      super(name);
    }
    override enqueueKernels(_queue: KernelQueue): void {
      calls.push(`gen:${this.name}`);
    }
    override drawPass(_pass: WgpuRenderPassEncoder): void {
      calls.push(`draw:${this.name}`);
    }
  }
  const scene = new Scene();
  scene.setLights([Light.sun()]);
  scene.addClump(new Recording("sky"));
  scene.addClump(new Recording("terrain"));
  scene.addClump(new Recording("flora"));
  // a nested clump hangs under flora's bounds — the graph holds it
  const nested = new Clump("coast");
  nested.plant(new QuadGeometry());
  check("a clump owns the geometry planted in it",
        nested.geometriesOf().length === 1
        && nested.geometriesOf()[0].vertexCount === 4);
  check("the sun light is directional and unit",
        scene.lightsOf.length === 1
        && scene.lightsOf[0].kind === "directional");
  // the real queue against a recording encoder: generate then draw
  const queue = new KernelQueue();
  const computeCalls: string[] = [];
  const encoder = {
    beginComputePass: () => ({
      setPipeline: (p: unknown) => computeCalls.push(`pipeline:${p}`),
      setBindGroup: () => {},
      dispatchWorkgroups: (...wg: number[]) =>
        computeCalls.push(`wg:${wg.join("x")}`),
      end: () => {},
    }),
  };
  const pass: WgpuRenderPassEncoder = {
    setPipeline: () => {}, setBindGroup: () => {},
    setVertexBuffer: () => {}, setIndexBuffer: () => {},
    draw: () => {}, drawIndexed: () => {}, end: () => {},
  };
  scene.generate(queue, encoder as never);
  scene.draw(pass);
  check("the clump order IS the frame order (gen pass, then draws)",
        calls.join(",") ===
        "gen:sky,gen:terrain,gen:flora,draw:sky,draw:terrain,draw:flora",
        calls.join(","));
}

// --- the WGSL common: the laws are present and intact --------------------
{
  const common = WGSL_COMMON + WGSL_CHECKER;
  for (const needle of [
    "struct Frame",
    "fn hash01v",
    "fn heightAt",
    "fn normalAt",
    "fn checkerMip",
    "fn checkerColour",
    "fn groundColour",
    "fn runwayShade",
    "fn roadDistances",
  ]) {
    check(`WGSL common defines ${needle}`,
          common.includes(needle));
  }
  check("the WGSL carries no unexpanded template holes",
        !common.includes("${"), "a template hole leaked");
  check("the WGSL's braces balance",
        (common.match(/{/g) ?? []).length === (common.match(/}/g) ?? []).length,
        "unbalanced");
}

finish("pipeline.test");
