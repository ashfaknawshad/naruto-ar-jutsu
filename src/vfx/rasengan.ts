import { Group, Mesh, IcosahedronGeometry, ShaderMaterial, AdditiveBlending, FrontSide, BackSide } from "three";
import { createChakraParticles } from "./particles";
import { createStreakRibbons } from "./streaks";

const VERTEX_SHADER = /* glsl */ `
varying vec3 vNormal;
varying vec3 vViewPosition;

void main() {
  vNormal = normalize(normalMatrix * normal);
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

// Outer aura — rendered BackSide so it glows around the silhouette rather
// than covering the front face, a cheap way to get a soft halo without a
// real bloom post-pass (that's Day 7's job). Rim-only: abs() on the dot
// product (fresnel is about how grazing the surface is, not which side is
// being rendered) plus a hard clamp before pow(), and a fairly steep
// falloff power — the previous version was unclamped and washed the whole
// silhouette into a flat filled disc instead of a thin glowing edge.
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
 * The flagship effect, three layers plus a particle fill:
 *  1. core — bright hot centre.
 *  2. streak ribbons — the spiral energy-line look (see streaks.ts).
 *  3. outer aura — soft BackSide halo.
 *  + a cloud of orbiting particles inside for texture/depth.
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

  const streaks = createStreakRibbons();
  const particles = createChakraParticles();

  group.add(core, aura, streaks.group, particles.points);

  function update(elapsedSeconds: number) {
    core.rotation.y = elapsedSeconds * 1.1;
    aura.rotation.y = -elapsedSeconds * 0.4;
    streaks.update(elapsedSeconds);
    particles.update(elapsedSeconds);
  }

  return { group, update };
}
