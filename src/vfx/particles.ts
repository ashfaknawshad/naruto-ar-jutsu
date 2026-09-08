import { Points, BufferGeometry, ShaderMaterial, AdditiveBlending, BufferAttribute } from "three";

// Position is computed entirely in the vertex shader from per-particle
// orbit attributes (radius/speed/phase/height) driven by uTime — cheap
// (no per-frame CPU buffer writes for hundreds of points) and gives each
// particle its own independent swirl instead of a uniform rotation.
const VERTEX_SHADER = /* glsl */ `
attribute float aRadius;
attribute float aSpeed;
attribute float aPhase;
attribute float aHeight;
attribute float aSize;
uniform float uTime;
varying float vTwinkle;

void main() {
  float angle = uTime * aSpeed + aPhase;
  float bob = sin(uTime * 0.6 + aPhase * 3.0) * 0.12;
  vec3 pos = vec3(cos(angle) * aRadius, aHeight + bob, sin(angle) * aRadius);
  vec4 mvPosition = modelViewMatrix * vec4(pos, 1.0);
  gl_Position = projectionMatrix * mvPosition;
  // Fixed pixel size (not distance-compensated — this uses an orthographic
  // camera, so there's no perspective foreshortening to correct for; it
  // also doesn't yet scale with the Rasengan's own world scale as the hand
  // moves closer/further, a simplification to revisit if it reads wrong).
  gl_PointSize = aSize;
  vTwinkle = 0.6 + 0.4 * sin(uTime * 2.5 + aPhase * 5.0);
}
`;

const FRAGMENT_SHADER = /* glsl */ `
varying float vTwinkle;

void main() {
  vec2 uv = gl_PointCoord - 0.5;
  float d = length(uv);
  float alpha = smoothstep(0.5, 0.0, d) * vTwinkle;
  gl_FragColor = vec4(0.75, 0.96, 1.0, alpha);
}
`;

export interface ChakraParticles {
  points: Points;
  update(elapsedSeconds: number): void;
}

/** The "lots of particles inside" swirl — small bright motes orbiting at varying radius/height/speed within the mid shell. */
export function createChakraParticles(count = 220): ChakraParticles {
  const geometry = new BufferGeometry();
  // Required by Three.js (bounding-sphere computation etc.) even though the
  // vertex shader ignores it and computes actual positions from the custom
  // attributes below — frustumCulled is disabled instead of relying on a
  // bounding sphere derived from these placeholder zeros.
  geometry.setAttribute("position", new BufferAttribute(new Float32Array(count * 3), 3));

  const radius = new Float32Array(count);
  const speed = new Float32Array(count);
  const phase = new Float32Array(count);
  const height = new Float32Array(count);
  const size = new Float32Array(count);
  for (let i = 0; i < count; i++) {
    radius[i] = 0.25 + Math.random() * 0.55;
    speed[i] = (Math.random() * 0.6 + 0.9) * (Math.random() < 0.5 ? 1 : -1);
    phase[i] = Math.random() * Math.PI * 2;
    height[i] = (Math.random() * 2 - 1) * 0.3;
    size[i] = 2 + Math.random() * 4;
  }
  geometry.setAttribute("aRadius", new BufferAttribute(radius, 1));
  geometry.setAttribute("aSpeed", new BufferAttribute(speed, 1));
  geometry.setAttribute("aPhase", new BufferAttribute(phase, 1));
  geometry.setAttribute("aHeight", new BufferAttribute(height, 1));
  geometry.setAttribute("aSize", new BufferAttribute(size, 1));

  const material = new ShaderMaterial({
    vertexShader: VERTEX_SHADER,
    fragmentShader: FRAGMENT_SHADER,
    uniforms: { uTime: { value: 0 } },
    transparent: true,
    blending: AdditiveBlending,
    depthWrite: false,
  });

  const points = new Points(geometry, material);
  points.frustumCulled = false;

  function update(elapsedSeconds: number) {
    material.uniforms.uTime.value = elapsedSeconds;
  }

  return { points, update };
}
