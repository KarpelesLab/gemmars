// Standalone check of the WGSL GEMM against a JS reference, for the three weight formats.
import { WebGpuHost } from '../src/lib/oxinfer.js';
const gpu = await WebGpuHost.create((s) => console.error(s));
const f16 = (v) => { // f32 -> f16 bits (round to nearest, enough for small scales)
  const f = new Float32Array([v]); const u = new Uint32Array(f.buffer)[0];
  const sign = (u >>> 16) & 0x8000; let exp = ((u >>> 23) & 0xff) - 127 + 15; let mant = u & 0x7fffff;
  if (exp <= 0) return sign; if (exp >= 31) return sign | 0x7c00;
  return sign | (exp << 10) | (mant >>> 13);
};
const f16to32 = (h) => { const s = (h & 0x8000) ? -1 : 1; const e = (h >>> 10) & 31; const m = h & 1023; return e === 0 ? s * m * 2 ** -24 : s * (1 + m / 1024) * 2 ** (e - 15); };
let seed = 12345; const rnd = () => { seed = (seed * 1103515245 + 12345) & 0x7fffffff; return seed / 0x7fffffff; };

async function check(kind, nOut, nIn, t) {
  const x = new Float32Array(t * nIn); for (let i = 0; i < x.length; i++) x[i] = rnd() * 2 - 1;
  let data, scales, wf = new Float32Array(nOut * nIn);
  if (kind === 0) {
    const w = new Uint16Array(nOut * nIn); const tmp = new Float32Array(1); const tu = new Uint32Array(tmp.buffer);
    for (let i = 0; i < w.length; i++) { tmp[0] = rnd() * 2 - 1; w[i] = tu[0] >>> 16; tu[0] = w[i] << 16; wf[i] = tmp[0]; }
    data = new Uint8Array(w.buffer); scales = null;
  } else if (kind === 1) {
    const q = new Int8Array(nOut * nIn); const sc = new Float32Array(nOut * nIn / 32);
    for (let i = 0; i < sc.length; i++) sc[i] = rnd() * 0.02 + 0.001;
    for (let i = 0; i < q.length; i++) { q[i] = Math.floor(rnd() * 255) - 127; wf[i] = q[i] * sc[Math.floor(i / 32)]; }
    data = new Uint8Array(q.buffer); scales = new Uint8Array(sc.buffer);
  } else if (kind === 4) {
    // affine q4, groups of 64: f16 scales (padded to 4 bytes) then bf16 offsets per group
    const bpr = nIn / 32, gpr = nIn / 64, base = Math.ceil(nOut * bpr / 2) * 2;
    const q = new Uint8Array(nOut * nIn / 2); const aux = new Uint16Array(base + nOut * gpr);
    const tmp = new Float32Array(1); const tu = new Uint32Array(tmp.buffer);
    for (let r = 0; r < nOut; r++) for (let g = 0; g < gpr; g++) {
      const sv = f16(rnd() * 0.02 + 0.001); tmp[0] = rnd() * 0.2 - 0.1; const ob = tu[0] >>> 16; tu[0] = ob << 16; const m = tmp[0];
      aux[base + r * gpr + g] = ob;
      for (const b of [r * bpr + 2 * g, r * bpr + 2 * g + 1]) {
        aux[b] = sv; const d = f16to32(sv);
        for (let j = 0; j < 16; j++) {
          const lo = Math.floor(rnd() * 16), hi = Math.floor(rnd() * 16);
          q[b * 16 + j] = lo | (hi << 4);
          wf[b * 32 + j] = (lo - 8) * d + m; wf[b * 32 + j + 16] = (hi - 8) * d + m;
        }
      }
    }
    data = q; scales = new Uint8Array(aux.buffer);
  } else {
    const q = new Uint8Array(nOut * nIn / 2); const sc = new Uint16Array(nOut * nIn / 32);
    for (let i = 0; i < sc.length; i++) sc[i] = f16(rnd() * 0.02 + 0.001);
    for (let b = 0; b < nOut * nIn / 32; b++) {
      const d = f16to32(sc[b]);
      for (let j = 0; j < 16; j++) {
        const lo = Math.floor(rnd() * 16), hi = Math.floor(rnd() * 16);
        q[b * 16 + j] = lo | (hi << 4);
        wf[b * 32 + j] = (lo - 8) * d; wf[b * 32 + j + 16] = (hi - 8) * d;
      }
    }
    data = q; scales = new Uint8Array(sc.buffer);
  }
  const h = gpu.upload(kind, data, scales, nOut, nIn);
  const y = new Float32Array(t * nOut);
  const t0 = performance.now();
  await gpu.matmul(x, t, nIn, [{ handle: h, nOut }], (slot, out) => y.set(out));
  const ms = performance.now() - t0;
  let maxd = 0, maxr = 0;
  for (let tt = 0; tt < t; tt++) for (let n = 0; n < nOut; n++) {
    let acc = 0; for (let k = 0; k < nIn; k++) acc += wf[n * nIn + k] * x[tt * nIn + k];
    maxd = Math.max(maxd, Math.abs(acc - y[tt * nOut + n])); maxr = Math.max(maxr, Math.abs(acc));
  }
  console.log(`kind ${kind} ${nOut}x${nIn} t=${t}: max diff ${maxd.toExponential(2)} (max |ref| ${maxr.toFixed(2)}) ${ms.toFixed(2)} ms`);
}
for (const kind of [0, 1, 2, 4]) await check(kind, 128, 64, 70);
await check(4, 192, 256, 33);
await check(0, 256, 96, 16);
// timing at model-like sizes
async function bench(kind, nOut, nIn, t, reps = 10) {
  const data = new Uint8Array(kind === 2 ? nOut * nIn / 2 : kind === 1 ? nOut * nIn : nOut * nIn * 2);
  const scales = kind === 0 ? null : new Uint8Array(nOut * nIn / 32 * (kind === 1 ? 4 : 2));
  const h = gpu.upload(kind, data, scales, nOut, nIn);
  const x = new Float32Array(t * nIn);
  await gpu.matmul(x, t, nIn, [{ handle: h, nOut }], () => {});
  const t0 = performance.now();
  for (let i = 0; i < reps; i++) await gpu.matmul(x, t, nIn, [{ handle: h, nOut }], () => {});
  const ms = (performance.now() - t0) / reps;
  console.log(`bench kind ${kind} ${nOut}x${nIn} t=${t}: ${ms.toFixed(2)} ms/call = ${(2 * t * nOut * nIn / ms / 1e9).toFixed(0)} GFLOP/s`);
}
await bench(1, 3584, 1024, 128);
await bench(1, 3584, 1024, 512);
await bench(0, 4096, 4096, 1024);
await bench(1, 4096, 4096, 1024);
await bench(2, 4096, 4096, 1024);
await bench(1, 64, 32, 64);
