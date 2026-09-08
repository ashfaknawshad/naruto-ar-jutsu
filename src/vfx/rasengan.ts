import { Group, Mesh, IcosahedronGeometry, ShaderMaterial, AdditiveBlending, FrontSide } from "three";
import { NOISE_GLSL } from "./shaders/noise";
import { createChakraParticles } from "./particles";

const VERTEX_SHADER = /* glsl */ `
varying vec3 vNormal;
varying vec3 vPosition;
varying vec3 vViewPosition;

void main() {
  vNormal = normalize(normalMatrix * normal);
  vPosition = position;
  vec4 mvPosition = modelViewMatrix * vec4(position, 1.0);
  vViewPosition = -mvPosition.xyz;
  gl_Position = projectionMatrix * mvPosition;
}
`;

// Bright core — broad, hot white-blue centre. Lower fresnel power than a
// typical rim shader means the bright zone covers more of the sphere, not
// just the edge, matching the wide hot centre in the reference image.
const CORE_FRAGMENT = /* glsl */ `
varying vec3 vNormal;
varying vec3 vViewPosition;

void main() {
  vec3 viewDir = normalize(vViewPosition);
  float fresnel = pow(1.0 - max(dot(vNormal, viewDir), 0.0), 2.0);
  vec3 color = mix(vec3(0.75, 0.95, 1.0), vec3(1.0), 1.0 - fresnel * 0.5);
  float alpha = clamp(0.75 + fresnel * 0.25, 0.0, 1.0);
  gl_FragColor = vec4(color, alpha);
}
`;

// The swirl — a dense field of thin bright streaks, not a handful of
// visible ribbon strands. streaks() turns a sine wave into narrow bright
// bands by taking max(sin, 0) to a high power (broad rounded peaks become
// narrow spikes, everything else goes near-black); the coordinate mixes
// angle-around-Y with radius so each band naturally banks into a spiral,
// and time subtracted from it makes the whole family swirl. Three
// differently-axis-permuted copies (p, p.yzx, p.zxy) give three families of
// streaks crossing each other at different orientations — combined with
// max() so they overlay instead of averaging out — which is what actually
// reads as "many thin streaks" rather than a few big ones. fbm only
// perturbs brightness slightly, for organic irregularity, not the pattern.
const MID_FRAGMENT = /* glsl */
  NOISE_GLSL +
  `
uniform float uTime;
varying vec3 vNormal;
varying vec3 vPosition;
varying vec3 vViewPosition;

float streaks(vec3 p, float freq, float sharpness, float speed) {
  float angle = atan(p.z, p.x);
  float radius = length(p.xz) + p.y * 0.5;
  // Slowed way down (was up to ±3.6) — this pattern is now the background
  // fiber texture, not the dominant motion; the pulses below do that job.
  float coord = angle * freq + radius * freq * 1.6 - uTime * speed;
  return pow(max(sin(coord), 0.0), sharpness);
}

void main() {
  float a = streaks(vPosition, 16.0, 22.0, 1.0);
  float b = streaks(vPosition.yzx, 21.0, 26.0, -0.8);
  float c = streaks(vPosition.zxy, 13.0, 24.0, 1.3);
  float d = streaks(vPosition * 1.3, 18.0, 20.0, -1.4);
  float fiber = max(max(a, b), max(c, d));

  float n = fbm(vPosition * 2.5 + uTime * 0.15);
  fiber *= 0.65 + 0.35 * (n * 0.5 + 0.5);

  vec3 viewDir = normalize(vViewPosition);
  float facing = max(dot(vNormal, viewDir), 0.0); // 1 at the point facing the camera dead-on, 0 at the silhouette edge
  float fresnel = pow(1.0 - facing, 1.3);

  // The actual "shooting inward" motion: this is a thin shell, not a solid
  // volume, so there's no true radial depth to travel through — instead
  // treat (1 - facing) as a pseudo-radius across the visible disc (0 at
  // its screen-centre, 1 at its silhouette) and run a travelling wave over
  // that. The "+ uTime" sign (not "-") is what makes rings move toward the
  // centre as time increases rather than away from it. Two staggered
  // frequencies/speeds so rings don't all arrive in lockstep.
  float pseudoRadius = 1.0 - facing;
  float pulse1 = pow(max(sin(pseudoRadius * 9.0 + uTime * 5.0), 0.0), 6.0);
  float pulse2 = pow(max(sin(pseudoRadius * 13.0 + uTime * 7.0 + 2.0), 0.0), 8.0);
  float pulses = max(pulse1, pulse2);

  float density = clamp(fiber * 0.7 + pulses * 0.9, 0.0, 1.0);

  vec3 deep = vec3(0.03, 0.3, 0.8);
  vec3 bright = vec3(0.75, 0.97, 1.0);
  vec3 color = mix(deep, bright, clamp(density + fresnel * 0.5, 0.0, 1.0));

  float alpha = clamp(density * 0.85 + fresnel * 0.6 + 0.05, 0.0, 1.0);
  gl_FragColor = vec4(color, alpha);
}
`;

// A thin edge-only glow, rendered with depthTest disabled so it draws over
// whatever's in front of it — including the finger-occlusion mask (see
// stage.ts). Real light bleeds over an object held in front of it; a solid
// depth-tested shell would just look like it stops dead at the fingertip,
// which is the thing that would give away "pasted on" instead of "held".
// Deliberately subtle (low alpha, sharp falloff) — this only needs to read
// as a thin rim, not repeat the earlier disc-shaped-aura mistake.
const GLOW_FRAGMENT = /* glsl */ `
varying vec3 vNormal;
varying vec3 vViewPosition;

void main() {
  vec3 viewDir = normalize(vViewPosition);
  float fresnel = pow(clamp(1.0 - abs(dot(vNormal, viewDir)), 0.0, 1.0), 4.0);
  vec3 color = vec3(0.6, 0.9, 1.0);
  gl_FragColor = vec4(color, fresnel * 0.35);
}
`;

export interface Rasengan {
  group: Group;
  /** @param elapsedSeconds A monotonically increasing clock, not a per-frame delta — drives rotation and every animated shader. */
  update(elapsedSeconds: number): void;
}

/**
 * The flagship effect: a bright core, a dense procedural streak-field
 * shell (see MID_FRAGMENT above), a thin depth-ignoring glow rim (so light
 * bleeds over the finger-occlusion mask, see stage.ts), and a cloud of
 * orbiting particles inside. All additively blended so it reads as glowing
 * energy, not a flat ball.
 */
export function createRasengan(): Rasengan {
  const group = new Group();

  const core = new Mesh(
    new IcosahedronGeometry(0.42, 3),
    new ShaderMaterial({
      vertexShader: VERTEX_SHADER,
      fragmentShader: CORE_FRAGMENT,
      transparent: true,
      blending: AdditiveBlending,
      depthWrite: false,
      side: FrontSide,
    }),
  );

  const midMaterial = new ShaderMaterial({
    vertexShader: VERTEX_SHADER,
    fragmentShader: MID_FRAGMENT,
    uniforms: { uTime: { value: 0 } },
    transparent: true,
    blending: AdditiveBlending,
    depthWrite: false,
    side: FrontSide,
  });
  const mid = new Mesh(new IcosahedronGeometry(0.95, 4), midMaterial);

  const glow = new Mesh(
    new IcosahedronGeometry(1.0, 2),
    new ShaderMaterial({
      vertexShader: VERTEX_SHADER,
      fragmentShader: GLOW_FRAGMENT,
      transparent: true,
      blending: AdditiveBlending,
      depthWrite: false,
      depthTest: false,
      side: FrontSide,
    }),
  );

  const particles = createChakraParticles();

  group.add(core, mid, glow, particles.points);

  function update(elapsedSeconds: number) {
    core.rotation.y = elapsedSeconds * 1.1;
    mid.rotation.y = -elapsedSeconds * 0.3;
    midMaterial.uniforms.uTime.value = elapsedSeconds;
    particles.update(elapsedSeconds);
  }

  return { group, update };
}
