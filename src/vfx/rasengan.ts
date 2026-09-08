import { Group, Mesh, IcosahedronGeometry, ShaderMaterial, AdditiveBlending, FrontSide, BackSide } from "three";
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
  float coord = angle * freq + radius * freq * 1.6 - uTime * speed;
  return pow(max(sin(coord), 0.0), sharpness);
}

void main() {
  float a = streaks(vPosition, 9.0, 14.0, 2.6);
  float b = streaks(vPosition.yzx, 12.0, 18.0, -2.1);
  float c = streaks(vPosition.zxy, 7.0, 16.0, 3.3);
  float density = max(a, max(b, c));

  float n = fbm(vPosition * 2.5 + uTime * 0.15);
  density *= 0.65 + 0.35 * (n * 0.5 + 0.5);

  vec3 viewDir = normalize(vViewPosition);
  float fresnel = pow(1.0 - max(dot(vNormal, viewDir), 0.0), 1.6);

  vec3 deep = vec3(0.03, 0.3, 0.8);
  vec3 bright = vec3(0.75, 0.97, 1.0);
  vec3 color = mix(deep, bright, clamp(density + fresnel * 0.3, 0.0, 1.0));

  float alpha = clamp(density * 0.85 + fresnel * 0.35 + 0.05, 0.0, 1.0);
  gl_FragColor = vec4(color, alpha);
}
`;

// Outer aura — rendered BackSide so it glows around the silhouette rather
// than covering the front face, a cheap way to get a soft halo without a
// real bloom post-pass (that's Day 7's job). Rim-only: abs() on the dot
// product (fresnel is about how grazing the surface is, not which side is
// being rendered) plus a hard clamp before pow(), and a fairly steep
// falloff power — an earlier unclamped version washed the whole silhouette
// into a flat filled disc instead of a thin glowing edge.
const AURA_FRAGMENT = /* glsl */ `
varying vec3 vNormal;
varying vec3 vViewPosition;

void main() {
  vec3 viewDir = normalize(vViewPosition);
  float fresnel = pow(clamp(1.0 - abs(dot(vNormal, viewDir)), 0.0, 1.0), 3.5);
  vec3 color = vec3(0.45, 0.8, 1.0);
  gl_FragColor = vec4(color, fresnel * 0.55);
}
`;

export interface Rasengan {
  group: Group;
  /** @param elapsedSeconds A monotonically increasing clock, not a per-frame delta — drives rotation and every animated shader. */
  update(elapsedSeconds: number): void;
}

/**
 * The flagship effect, three shell layers plus a particle fill:
 *  1. core — bright hot centre.
 *  2. mid — dense procedural streak field (see MID_FRAGMENT above).
 *  3. outer aura — soft BackSide halo.
 *  + a cloud of orbiting particles inside for extra texture/depth.
 * All additively blended so it reads as glowing energy, not a flat ball.
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
  const mid = new Mesh(new IcosahedronGeometry(0.8, 4), midMaterial);

  const aura = new Mesh(
    new IcosahedronGeometry(1.05, 2),
    new ShaderMaterial({
      vertexShader: VERTEX_SHADER,
      fragmentShader: AURA_FRAGMENT,
      transparent: true,
      blending: AdditiveBlending,
      depthWrite: false,
      side: BackSide,
    }),
  );

  const particles = createChakraParticles();

  group.add(core, mid, aura, particles.points);

  function update(elapsedSeconds: number) {
    core.rotation.y = elapsedSeconds * 1.1;
    mid.rotation.y = -elapsedSeconds * 0.3;
    aura.rotation.y = -elapsedSeconds * 0.4;
    midMaterial.uniforms.uTime.value = elapsedSeconds;
    particles.update(elapsedSeconds);
  }

  return { group, update };
}
