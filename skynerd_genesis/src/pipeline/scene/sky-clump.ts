/**
 * SkyClump — the scene's sky (plan §3 pass B): a fullscreen triangle
 * under everything, painting the gradient (fog at the horizon, zenith
 * overhead — the exact CPU constants) plus the sun's disc and glow at
 * its fixed world direction. No depth write: the terrain and trees
 * simply draw over it.
 */
import type { GpuDevice } from "../core/gpu-device.js";
import type { WgpuBindGroup, WgpuBindGroupLayout,
              WgpuRenderPassEncoder, WgpuRenderPipeline } from "../types.js";
import { RenderClump } from "./scene.js";
import { WGSL_COMMON } from "../core/wgsl.js";

const SKY_WGSL = /* wgsl */ `
@vertex
fn skyVs(@builtin(vertex_index) vi : u32) -> @builtin(position) vec4f {
  // the fullscreen triangle: three corners covering the viewport
  var corners = array<vec2f, 3>(vec2f(-1.0, -1.0),
                                vec2f( 3.0, -1.0),
                                vec2f(-1.0,  3.0));
  return vec4f(corners[vi], 1.0, 1.0);
}

@fragment
fn skyFs(@builtin(position) frag : vec4f) -> @location(0) vec4f {
  // reconstruct the pixel's view ray from the framebuffer position —
  // the same pinhole the CPU path walks per pixel
  let px = frag.xy;
  let W = frame.up.w;
  let H = frame.fwd.w;
  let f = frame.right.w;
  let a = (px.x - W * 0.5) / f;
  let b = (H * 0.5 - px.y) / f;
  let dir = normalize(frame.fwd.xyz + a * frame.right.xyz
                                    + b * frame.up.xyz);
  // the gradient: fog at the horizon, deepening with elevation —
  // the mood's own colours from the uniform (0..255, normalised)
  let t = clamp(dir.z / SKY_GRAD_SIN, 0.0, 1.0);
  var col = mix(frame.fog.xyz, frame.zenith.xyz, t) / 255.0;
  // THE CRASH FLOOR: 15° under the horizon the background is black —
  // a CRASHED camera (low, frame.extras.z) looks into darkness, never
  // into sky. A flying camera keeps the haze band here instead: the
  // world's far edge dissolves into the horizon's fog, never a black
  // band (the terrain's reach ends far past where the eye can tell).
  if (dir.z < -SKY_FLOOR_SIN && frame.extras.z > 0.5) {
    col = vec3f(0.0);
  }
  // the sun: its disc and glare, at the mood's direction. The bigger
  // the disc the lower the sun — and the more the atmosphere chews its
  // edge (a blendy, wobbled silhouette) and the harder it glares
  let sc = frame.viewProj * vec4f(frame.sunDir.xyz, 0.0);
  if (sc.w > 0.0 && frame.fog.w > 0.001) {
    let sx = (sc.x / sc.w * 0.5 + 0.5) * W;
    let sy = (1.0 - (sc.y / sc.w * 0.5 + 0.5)) * H;
    let dx = px.x - sx;
    let dy = px.y - sy;
    let disc = f * tan(frame.zenith.w);
    let tint = frame.sunTint.xyz / 255.0;
    // the disc's COLOUR: a hot core — the tint lifted toward white so
    // the sun is never darker than the sky around it; the blendy rim
    // then melts it INTO the sky's own shade (no seam, no sticker)
    let core = mix(tint, vec3f(1.0), 0.65);
    // the setting factor: 0 at noon's clean circle, 1 at the swell
    let s = clamp((frame.zenith.w - 0.035) / 0.055, 0.0, 1.0);
    // the atmosphere chews the rim: two harmonics of wobble
    let ang = atan2(dy, dx);
    let wob = s * (0.08 * sin(4.0 * ang + 1.7)
                   + 0.05 * sin(9.0 * ang + 0.6));
    let ry = dy * (1.0 + 0.35 * s);   // refraction's vertical squash
    let base = sqrt(dx * dx + ry * ry);
    // the blendy edge: solid at the wobbled core, fully melted just
    // beyond the wobbled radius — never a hard rim
    let radius = disc * (1.0 + wob);
    let solid = clamp((radius - base) / (disc * 0.32), 0.0, 1.0);
    // the glare: wider and far stronger as the sun sinks
    let glowR = disc * SUN_GLOW * (1.0 + 0.8 * s);
    let glow = clamp(1.0 - base / glowR, 0.0, 1.0);
    col = mix(col, core, solid);
    col += mix(tint, vec3f(1.0), 0.25) * (0.28 + 1.1 * s)
           * frame.fog.w * glow * glow * (1.0 - solid);
  }
  // THE PILOT'S IRIS (HDR adaptation): facing the sun the eye
  // tightens and the whole world — sky, sun, glare — dims by the
  // exposure factor; away from it the world stays wide open (1.0)
  col *= frame.extras.y;
  return vec4f(col, 1.0);
}
`;

export class SkyClump extends RenderClump {
  private readonly pipeline: WgpuRenderPipeline;

  constructor(device: GpuDevice, canvasFormat: string,
              private readonly group0: WgpuBindGroup,
              group0Layout: WgpuBindGroupLayout) {
    super("sky");
    const dev = device.device;
    this.pipeline = dev.createRenderPipeline({
      layout: dev.createPipelineLayout({
        bindGroupLayouts: [group0Layout],
      }),
      vertex: {
        module: dev.createShaderModule({ code: WGSL_COMMON + SKY_WGSL }),
        entryPoint: "skyVs",
      },
      fragment: {
        module: dev.createShaderModule({ code: WGSL_COMMON + SKY_WGSL }),
        entryPoint: "skyFs",
        targets: [{ format: canvasFormat }],
      },
      primitive: { topology: "triangle-list" },
      depthStencil: {
        format: "depth24plus",
        depthWriteEnabled: false,
        depthCompare: "always",
      },
    });
  }

  override enqueueKernels(): void {
    // the sky generates nothing — a pure draw
  }

  override drawPass(pass: WgpuRenderPassEncoder): void {
    pass.setPipeline(this.pipeline);
    pass.setBindGroup(0, this.group0);
    pass.draw(3);
  }
}
