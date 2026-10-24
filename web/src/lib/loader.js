// Fetches model files into the module: from Hugging Face or any URL (kept in the browser's Cache
// Storage so a reload does not download gigabytes again), or from a folder the user picks.
import { isHostShard } from './oxinfer.js';

const CACHE = 'oxinfer-models-v1';
const ROOT = '/model';

/// "org/name" -> its Hugging Face file URL base; URLs and paths ("./x", "/x") are used as they are.
export function sourceBase(spec) {
  const s = spec.trim().replace(/\/+$/, '');
  if (/^[\w.-]+\/[\w.-]+$/.test(s)) return `https://huggingface.co/${s}/resolve/main`;
  return s;
}

async function fetchJson(base, name) {
  const r = await fetch(`${base}/${name}`);
  if (!r.ok) return null;
  try {
    return await r.json();
  } catch {
    return null;
  }
}

/// The files a model folder needs, discovered from its config and index.
export async function listFiles(base) {
  if (!(await fetchJson(base, 'config.json'))) {
    throw new Error(`No config.json at ${base}. Check the repo id or URL.`);
  }
  const files = ['config.json', 'tokenizer.json'];
  if (await fetchJson(base, 'generation_config.json')) files.push('generation_config.json');
  const index = await fetchJson(base, 'model.safetensors.index.json');
  if (index && index.weight_map) {
    files.push('model.safetensors.index.json', ...new Set(Object.values(index.weight_map)));
  } else {
    files.push('model.safetensors');
  }
  return files;
}

async function openCache() {
  try {
    return typeof caches !== 'undefined' ? await caches.open(CACHE) : null;
  } catch {
    return null;
  }
}

/// A Response for `url`, from the cache when present; otherwise downloaded into the cache first
/// (written straight to disk, so a multi-GB file never sits in memory twice).
async function cachedResponse(url, onBytes) {
  const cache = await openCache();
  if (cache) {
    const hit = await cache.match(url);
    if (hit) return { response: hit, cached: true };
  }
  const r = await fetch(url);
  if (!r.ok) throw new Error(`${url}: HTTP ${r.status}`);
  if (!cache || !r.body) return { response: r, cached: false };
  let got = 0;
  const size = Number(r.headers.get('content-length') || 0);
  const counted = r.body.pipeThrough(
    new TransformStream({
      transform(chunk, ctl) {
        got += chunk.byteLength;
        onBytes(got, size);
        ctl.enqueue(chunk);
      },
    }),
  );
  try {
    await cache.put(url, new Response(counted, { headers: r.headers }));
    const stored = await cache.match(url);
    if (stored) return { response: stored, cached: false };
  } catch {
    // Quota exceeded or private mode: fall back to a plain download.
  }
  const again = await fetch(url);
  if (!again.ok) throw new Error(`${url}: HTTP ${again.status}`);
  return { response: again, cached: false };
}

async function readInto(g, name, size, stream, onBytes) {
  const path = `${ROOT}/${name}`;
  const reader = stream.getReader();
  if (isHostShard(name)) {
    const bytes = new Uint8Array(size);
    let off = 0;
    for (;;) {
      const { done, value } = await reader.read();
      if (done) break;
      bytes.set(value, off);
      off += value.byteLength;
      onBytes(off);
    }
    g.putHostFile(path, bytes);
    return;
  }
  const w = g.putFileStream(path, size);
  let off = 0;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    w.write(value);
    off += value.byteLength;
    onBytes(off);
  }
  if (off !== size) throw new Error(`${name}: got ${off} of ${size} bytes`);
  w.done();
}

/// Downloads (or reuses) every file of the model at `base` into the module.
/// `progress({ file, index, count, got, size, phase })`, phase: 'download' | 'read'.
export async function loadFromUrl(g, base, progress) {
  const files = await listFiles(base);
  let total = 0;
  for (const [index, name] of files.entries()) {
    const url = `${base}/${name}`;
    const report = (phase, known) => (got, size = known) => progress({ file: name, index, count: files.length, got, size, phase });
    let { response } = await cachedResponse(url, report('download', 0));
    const size = Number(response.headers.get('content-length') || 0);
    if (!size || !response.body) {
      const bytes = new Uint8Array(await response.arrayBuffer());
      response = new Response(bytes);
      await readInto(g, name, bytes.byteLength, response.body, report('read', bytes.byteLength));
      total += bytes.byteLength;
      continue;
    }
    await readInto(g, name, size, response.body, report('read', size));
    total += size;
  }
  return total;
}

const WANTED = /^(config\.json|generation_config\.json|tokenizer\.json|model\.safetensors\.index\.json|.*\.safetensors)$/;

/// Reads a folder picked with <input webkitdirectory> into the module.
export async function loadFromFolder(g, fileList, progress) {
  const files = [...fileList].filter((f) => WANTED.test(f.name));
  if (!files.some((f) => f.name === 'config.json')) throw new Error('That folder has no config.json, so it is not a model folder.');
  let total = 0;
  for (const [index, f] of files.entries()) {
    await readInto(g, f.name, f.size, f.stream(), (got) => progress({ file: f.name, index, count: files.length, got, size: f.size, phase: 'read' }));
    total += f.size;
  }
  return total;
}

/// Bytes this site keeps in Cache Storage, and a way to drop them.
export async function cachedBytes() {
  try {
    const est = await navigator.storage?.estimate?.();
    return est?.usageDetails?.caches ?? est?.usage ?? 0;
  } catch {
    return 0;
  }
}

export async function clearCache() {
  if (typeof caches !== 'undefined') await caches.delete(CACHE);
}
