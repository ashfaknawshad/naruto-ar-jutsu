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

// Bright core — broad, hot white-blue centre. Lower fresnel power than the
// mid shell means the bright zone covers more of the sphere, not just the
// rim, matching the wide hot centre in the reference image.
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

// The swirl. Reference art reads as curved spiral streak lines (like water
// circling a drain), not cloudy blobs — so the base pattern here is
// stripes(): a sine wave over an angle-plus-radius coordinate that
// naturally banks into a spiral, animated by subtracting uTime from the
// angle. fbm only perturbs it for organic irregularity, it doesn't drive
// the pattern itself the way the first version did.
const MID_FRAGMENT = /* glsl */
  NOISE_GLSL +
  `
uniform float uTime;
varying vec3 vNormal;
varying vec3 vPosition;
varying vec3 vViewPosition;

void main() {
  float radius = length(vPosition);
  float angle = atan(vPosition.z, vPosition.x);
  float spiral = angle * 5.0 + radius * 9.0 - uTime * 2.2;
  float stripes = sin(spiral) * 0.5 + 0.5;
  stripes = pow(stripes, 1.4);

  float n = fbm(vPosition * 3.0 + vec3(0.0, 0.0, uTime * 0.4));
  stripes = mix(stripes, stripes * (0.55 + n * 0.55), 0.55);

  vec3 viewDir = normalize(vViewPosition);
  float fresnel = pow(1.0 - max(dot(vNormal, viewDir), 0.0), 1.6);

  vec3 deep = vec3(0.0, 0.35, 0.85);
  vec3 bright = vec3(0.6, 0.97, 1.0);
  vec3 color = mix(deep, bright, clamp(stripes + fresnel * 0.5, 0.0, 1.0));

  float alpha = clamp(stripes * 0.75 + fresnel * 0.55 + 0.12, 0.0, 1.0);
  gl_FragColor = vec4(color, alpha);
}
`;

export interface Rasengan {
  group: Group;
  /** @param elapsedSeconds A monotonically increasing clock, not a per-frame delta — drives both rotation and the noise animation. */
  update(elapsedSeconds: number): void;
}

/**
 * The flagship effect: a bright core shell, a spiral-streaked swirl shell,
 * and a cloud of orbiting particles inside — all additively blended so it
 * reads as glowing energy rather than a flat ball. Two shells (not three —
 * the original had a subtle outer rim shell too, cut because it wasn't
 * pulling visual weight and the particle swirl reads better for "volatile
 * edge" than another translucent sphere did).
 */
export function createRasengan(): Rasengan {
  const group = new Group();

  const core = new Mesh(
    new IcosahedronGeometry(0.5, 3),
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
  const mid = new Mesh(new IcosahedronGeometry(0.85, 3), midMaterial);

  const particles = createChakraParticles();

  group.add(core, mid, particles.points);

  function update(elapsedSeconds: number) {
    core.rotation.y = elapsedSeconds * 1.1;
    mid.rotation.y = -elapsedSeconds * 0.7;
    mid.rotation.x = elapsedSeconds * 0.25;
    midMaterial.uniforms.uTime.value = elapsedSeconds;
    particles.update(elapsedSeconds);
  }

  return { group, update };
}
