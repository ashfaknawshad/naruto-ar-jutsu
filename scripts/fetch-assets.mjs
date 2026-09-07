// Populates public/mediapipe/wasm (copied from the installed npm package)
// and public/models/hand_landmarker.task (downloaded once, with resume —
// this network has been observed to stall mid-download, so we retry with
// a Range request until the file matches its expected size).
import { existsSync, mkdirSync, copyFileSync, readdirSync, statSync, createWriteStream, truncateSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import { Readable } from "node:stream";
import { pipeline } from "node:stream/promises";

const root = dirname(dirname(fileURLToPath(import.meta.url)));

const wasmSrc = join(root, "node_modules/@mediapipe/tasks-vision/wasm");
const wasmDst = join(root, "public/mediapipe/wasm");
mkdirSync(wasmDst, { recursive: true });
for (const file of readdirSync(wasmSrc)) {
  copyFileSync(join(wasmSrc, file), join(wasmDst, file));
}
console.log(`copied ${readdirSync(wasmSrc).length} wasm files -> public/mediapipe/wasm`);

const modelUrl =
  "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task";
const modelDst = join(root, "public/models/hand_landmarker.task");
mkdirSync(dirname(modelDst), { recursive: true });

async function downloadWithResume(url, dest, maxAttempts = 15) {
  const head = await fetch(url, { method: "HEAD" });
  const expectedSize = Number(head.headers.get("content-length"));

  for (let attempt = 1; attempt <= maxAttempts; attempt++) {
    let have = existsSync(dest) ? statSync(dest).size : 0;
    if (have >= expectedSize && expectedSize > 0) {
      console.log(`model already complete (${have} bytes) -> ${dest}`);
      return;
    }

    const res = await fetch(url, { headers: have > 0 ? { Range: `bytes=${have}-` } : {} });
    if (!res.ok && res.status !== 206) throw new Error(`download failed: ${res.status}`);

    // The server may not honour Range (some CDNs/proxies ignore it and send
    // 200 + the full body) — appending in that case would duplicate data,
    // so only append when the server actually confirmed a partial response.
    const appending = have > 0 && res.status === 206;
    if (have > 0 && !appending) {
      truncateSync(dest, 0);
      have = 0;
    }

    try {
      await pipeline(Readable.fromWeb(res.body), createWriteStream(dest, { flags: appending ? "a" : "w" }));
    } catch {
      // Network stalled mid-stream — loop and resume from wherever we got to.
    }

    const now = statSync(dest).size;
    console.log(`  attempt ${attempt}: ${now}/${expectedSize} bytes`);
    if (now === expectedSize) break;
    if (now > expectedSize) {
      // Shouldn't happen given the truncate-on-mismatch guard above, but
      // never leave a corrupt oversized file behind.
      truncateSync(dest, 0);
    }
  }

  const final = statSync(dest).size;
  if (final !== expectedSize) {
    throw new Error(`model download incomplete: ${final}/${expectedSize} bytes after ${maxAttempts} attempts`);
  }
  console.log(`downloaded model (${final} bytes) -> ${dest}`);
}

await downloadWithResume(modelUrl, modelDst);
