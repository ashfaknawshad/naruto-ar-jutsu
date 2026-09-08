import { Group, Mesh, BoxGeometry, ShaderMaterial, AdditiveBlending, DoubleSide } from "three";

// Thin, long, subdivided box — the subdivisions along its length are what
// let the vertex shader below bend it into a smooth curve rather than a
// jagged zigzag. Shared across every ribbon instance; only the material
// (which carries each ribbon's own phase/speed) differs per instance.
const RIBBON_GEOMETRY = new BoxGeometry(0.05, 0.05, 1.7, 1, 1, 32);

const VERTEX_SHADER = /* glsl */ `
uniform float uTime;
uniform float uPhase;
varying float vT;

void main() {
  vT = position.z;
  // Amplitude big enough to bow the curve out to (and past) the aura's
  // radius — at the previous, much smaller amplitude the ribbons stayed
  // close to a straight line through the centre and barely read at all.
  float wave = sin(position.z * 5.0 + uTime * 3.0 + uPhase) * 0.45;
  float wave2 = cos(position.z * 3.5 - uTime * 2.2 + uPhase) * 0.32;
  vec3 displaced = vec3(position.x + wave, position.y + wave2, position.z);
  vec4 mvPosition = modelViewMatrix * vec4(displaced, 1.0);
  gl_Position = projectionMatrix * mvPosition;
}
`;

const FRAGMENT_SHADER = /* glsl */ `
uniform float uTime;
uniform float uPhase;
varying float vT;

void main() {
  float pulse = 0.7 + 0.3 * sin(vT * 8.0 - uTime * 4.0 + uPhase);
  vec3 color = mix(vec3(0.35, 0.75, 1.0), vec3(0.9, 0.99, 1.0), pulse);
  // Fade the last ~25% toward each tip so ribbons taper instead of
  // presenting a hard-edged cut end.
  float edgeFade = 1.0 - smoothstep(0.6, 0.85, abs(vT));
  gl_FragColor = vec4(color, clamp(pulse * edgeFade * 1.3, 0.0, 1.0));
}
`;

export interface StreakRibbons {
  group: Group;
  update(elapsedSeconds: number): void;
}

/**
 * The spiral-streak look from the reference (curved lines wrapping the
 * sphere, like water circling a drain) — not faked with a noise pattern,
 * but actually drawn as thin curved ribbons: several instances of the same
 * warped-box geometry, fanned around at different fixed tilts and spun at
 * different rates, so their overlapping curves read as a chaotic tangle of
 * energy streaks. Technique adapted from a public three.js Rasengan demo
 * (github.com/SuboptimalEng/three-js-games/tree/main/05-naruto-rasengan).
 */
export function createStreakRibbons(count = 12): StreakRibbons {
  const group = new Group();
  const materials: ShaderMaterial[] = [];
  const speeds: number[] = [];

  for (let i = 0; i < count; i++) {
    const material = new ShaderMaterial({
      vertexShader: VERTEX_SHADER,
      fragmentShader: FRAGMENT_SHADER,
      uniforms: { uTime: { value: 0 }, uPhase: { value: i * 0.7 } },
      transparent: true,
      blending: AdditiveBlending,
      depthWrite: false,
      side: DoubleSide,
    });
    const mesh = new Mesh(RIBBON_GEOMETRY, material);
    // Fan the ribbons around like meridian lines on a globe.
    mesh.rotation.x = (i / count) * Math.PI;
    mesh.rotation.z = (i * 0.618) % Math.PI; // golden-angle-ish stagger so tilts don't line up periodically
    group.add(mesh);
    materials.push(material);
    speeds.push((0.5 + (i % 5) * 0.15) * (i % 2 === 0 ? 1 : -1));
  }

  function update(elapsedSeconds: number) {
    for (let i = 0; i < count; i++) {
      group.children[i].rotation.y = elapsedSeconds * speeds[i];
      materials[i].uniforms.uTime.value = elapsedSeconds;
    }
  }

  return { group, update };
}
