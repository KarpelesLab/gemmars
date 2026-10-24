#!/usr/bin/env -S deno run --allow-read --allow-env
// Deno CLI for the browser build: same module and glue as index.html, model from a local directory.
//   deno run --allow-read web/run.mjs --model ~/.cache/oxinfer/hub/Qwen--Qwen3.5-0.8B --quant q8 -p "Hello"
import { Oxinfer, WebGpuHost, wantedFiles, hasJspi, isHostShard } from '../src/lib/oxinfer.js';

const args = Deno.args;
const opt = (name, def) => { const i = args.indexOf(name); return i >= 0 ? args[i + 1] : def; };
const has = (name) => args.includes(name);
const modelDir = opt('--model', null);
if (!modelDir) { console.error('usage: run.mjs --model DIR [--quant bf16|q8|q4] [--cpu] [--no-jspi] [-p PROMPT] [--max-tokens N] [--think] [--chunk N] [--temp T]'); Deno.exit(2); }
const quant = opt('--quant', 'q8');
const prompt = opt('-p', 'Explain what a mitochondrion does in two sentences.');
const maxTokens = Number(opt('--max-tokens', '128'));
const chunk = Number(opt('--chunk', '0'));
const log = (s) => console.error(s);

const wasmPath = new URL('../../target/wasm32-unknown-unknown/release/oxinfer.wasm', import.meta.url);
const gpu = has('--cpu') ? null : await WebGpuHost.create(log);
log(`[host: deno ${Deno.version.deno}, jspi ${hasJspi}, gpu ${gpu ? 'yes' : 'no'}]`);
const g = await Oxinfer.create(Deno.readFileSync(wasmPath), { gpu, log, jspi: !has('--no-jspi') });

const t0 = performance.now();
let total = 0;
const names = [];
for await (const e of Deno.readDir(modelDir)) if (e.isFile) names.push(e.name);
for (const name of wantedFiles(names)) {
  const path = `${modelDir}/${name}`;
  const size = (await Deno.stat(path)).size;
  const f = await Deno.open(path);
  if (isHostShard(name)) {
    const bytes = new Uint8Array(size);
    let off = 0;
    while (off < size) {
      const n = await f.read(bytes.subarray(off));
      if (n === null) break;
      off += n;
    }
    f.close();
    g.putHostFile(`/model/${name}`, bytes);
    log(`[${name}: ${(size / 1e9).toFixed(2)} GB kept in host memory]`);
    total += size;
    continue;
  }
  const w = g.putFileStream(`/model/${name}`, size);
  const buf = new Uint8Array(32 << 20);
  for (;;) {
    const n = await f.read(buf);
    if (n === null) break;
    w.write(buf.subarray(0, n));
  }
  f.close();
  w.done();
  total += size;
}
log(`[read ${(total / 1e9).toFixed(2)} GB of model files in ${((performance.now() - t0) / 1e3).toFixed(2)}s]`);
g.load('/model', quant, { prefillChunk: chunk });
g.setSampling({ temperature: Number(opt('--temp', '0')), topK: 64, topP: 0.95, seed: 1 });

let shown = '';
const st = await g.generate(prompt, { think: has('--think'), maxTokens }, (text) => {
  if (text.startsWith(shown) && !text.endsWith('�')) {
    Deno.stdout.writeSync(new TextEncoder().encode(text.slice(shown.length)));
    shown = text;
  }
});
if (st.text !== shown) Deno.stdout.writeSync(new TextEncoder().encode(st.text.slice(shown.length)));
console.log();
log(`[prefill ${st.promptTokens} tok in ${st.prefillSecs.toFixed(2)}s = ${(st.promptTokens / st.prefillSecs).toFixed(1)} tok/s | decode ${st.tokens} tok in ${st.decodeSecs.toFixed(2)}s = ${(st.tokens / st.decodeSecs).toFixed(1)} tok/s]`);
const gs = g.gpuStats();
if (gs) log(`[gpu: ${gs}]`);
