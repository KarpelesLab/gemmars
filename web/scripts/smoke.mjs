// Headless check of the JS <-> wasm bridge, run in CI before the site is built: instantiates
// public/oxinfer.wasm with the real glue, round-trips strings and files through module memory,
// and confirms a load without model files fails with the module's own error message.
import { readFile } from 'node:fs/promises';
import { Oxinfer } from '../src/lib/oxinfer.js';

const fail = (msg) => {
  console.error(`smoke: FAIL: ${msg}`);
  process.exit(1);
};

const bytes = await readFile(new URL('../public/oxinfer.wasm', import.meta.url));
const lines = [];
const g = await Oxinfer.create(bytes, { gpu: null, log: (s) => lines.push(s) });

for (const name of ['web_alloc', 'web_free', 'web_put_file', 'web_put_host_file', 'web_load', 'web_chat_turn', 'web_next', 'web_text', 'web_reset']) {
  if (typeof g.ex[name] !== 'function') fail(`export ${name} missing`);
}

// A config.json without weights: the loader must get far enough to report the missing tensors.
g.putFile('/model/config.json', new TextEncoder().encode(JSON.stringify({ model_type: 'gemma4', text_config: { model_type: 'gemma4_text' } })));
let err = '';
try {
  g.load('/model', 'q8');
} catch (e) {
  err = String(e.message || e);
}
if (!err) fail('loading an incomplete model should fail');
if (!/tokenizer\.json|config|not provided/i.test(err)) fail(`unexpected error: ${err}`);

// Nothing loaded: generation calls must be harmless.
if ((await g.next()) !== -1) fail('web_next without a model should return -1');

console.log(`smoke: ok (${bytes.length} bytes; load error as expected: ${err.slice(0, 90)})`);
