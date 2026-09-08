import { Group, Mesh, IcosahedronGeometry, ShaderMaterial, AdditiveBlending, FrontSide } from "three";
import { NOISE_GLSL } from "./shaders/noise";

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

// Bright, near-white, fresnel-boosted — the dense energy at the centre of
// the palm. No noise here; it's meant to read as a solid core the other
// two shells swirl around.
const CORE_FRAGMENT = /* glsl */ `
varying vec3 vNormal;
varying vec3 vViewPosition;

void main() {
  vec3 viewDir = normalize(vViewPosition);
  float fresnel = pow(1.0 - max(dot(vNormal, viewDir), 0.0), 3.0);
  vec3 color = mix(vec3(0.65, 0.9, 1.0), vec3(1.0), fresnel);
  float alpha = clamp(0.55 + fresnel * 0.45, 0.0, 1.0);
  gl_FragColor = vec4(color, alpha);
}
`;

// The swirl: domain-warped 3-octave fbm, animated over time, blue-cyan
// ramp. This is the shell that's actually supposed to look like it's
// churning — the domain warp (offsetting the noise lookup by another noise
// field) is what keeps it from looking like a static marbled texture.
const MID_FRAGMENT = /* glsl */
  NOISE_GLSL +
  `
uniform float uTime;
varying vec3 vNormal;
varying vec3 vPosition;
varying vec3 vViewPosition;

void main() {
  vec3 p = vPosition * 2.5 + vec3(0.0, 0.0, uTime * 0.35);
  vec3 warp = vec3(
    fbm(p + vec3(1.7, 9.2, 4.3)),
    fbm(p + vec3(8.3, 2.8, 1.0)),
    fbm(p + vec3(5.1, 3.6, 7.9))
  );
  float n = fbm(p + warp * 0.7) * 0.5 + 0.5;

  vec3 viewDir = normalize(vViewPosition);
  float fresnel = pow(1.0 - max(dot(vNormal, viewDir), 0.0), 2.0);

  vec3 deep = vec3(0.02, 0.25, 0.75);
  vec3 bright = vec3(0.45, 0.92, 1.0);
  vec3 color = mix(deep, bright, n);

  float alpha = clamp(n * 0.75 + fresnel * 0.35, 0.0, 1.0);
  gl_FragColor = vec4(color, alpha);
}
`;

// The volatile edge: higher-frequency noise, low alpha, strongly
// fresnel-weighted so it's mostly a rim glow rather than a solid shell —
// counter-rotated (in JS, see update()) against the mid shell for the
// "unstable energy" read.
const OUTER_FRAGMENT = /* glsl */
  NOISE_GLSL +
  `
uniform float uTime;
varying vec3 vNormal;
varying vec3 vPosition;
varying vec3 vViewPosition;

void main() {
  vec3 p = vPosition * 5.0 - vec3(0.0, 0.0, uTime * 0.5);
  float n = fbm(p) * 0.5 + 0.5;

  vec3 viewDir = normalize(vViewPosition);
  float fresnel = pow(1.0 - max(dot(vNormal, viewDir), 0.0), 1.5);

  vec3 color = mix(vec3(0.2, 0.6, 1.0), vec3(0.8, 0.98, 1.0), n);
  float alpha = clamp(n * 0.35 * fresnel + fresnel * 0.25, 0.0, 1.0);
  gl_FragColor = vec4(color, alpha);
}
`;

export interface Rasengan {
  group: Group;
  /** @param elapsedSeconds A monotonically increasing clock, not a per-frame delta — drives both rotation and the noise animation. */
  update(elapsedSeconds: number): void;
}

/**
 * The flagship effect: three nested, additively-blended icosphere shells
 * (see the fragment shaders above for what each one is doing). Additive
 * blending is what makes this read as glowing energy instead of a flat
 * blue ball — energy adds light to what's behind it, it doesn't occlude.
 */
export function createRasengan(): Rasengan {
  const group = new Group();

  const core = new Mesh(
    new IcosahedronGeometry(0.55, 3),
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
  const mid = new Mesh(new IcosahedronGeometry(0.8, 3), midMaterial);

  const outerMaterial = new ShaderMaterial({
    vertexShader: VERTEX_SHADER,
    fragmentShader: OUTER_FRAGMENT,
    uniforms: { uTime: { value: 0 } },
    transparent: true,
    blending: AdditiveBlending,
    depthWrite: false,
    side: FrontSide,
  });
  const outer = new Mesh(new IcosahedronGeometry(1.0, 2), outerMaterial);

  group.add(core, mid, outer);

  function update(elapsedSeconds: number) {
    core.rotation.y = elapsedSeconds * 0.6;
    mid.rotation.y = -elapsedSeconds * 0.35;
    mid.rotation.x = elapsedSeconds * 0.15;
    outer.rotation.y = elapsedSeconds * 1.1;
    outer.rotation.x = -elapsedSeconds * 0.5;
    midMaterial.uniforms.uTime.value = elapsedSeconds;
    outerMaterial.uniforms.uTime.value = elapsedSeconds;
  }

  return { group, update };
}
