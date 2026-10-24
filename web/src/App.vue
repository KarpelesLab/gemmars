<script setup>
import { computed, nextTick, onMounted, reactive, ref } from 'vue';
import { Oxinfer, WebGpuHost, hasJspi } from './lib/oxinfer.js';
import { cachedBytes, clearCache, loadFromFolder, loadFromUrl, sourceBase } from './lib/loader.js';
import Reply from './components/Reply.vue';
import ImageStudio from './components/ImageStudio.vue';

const WASM_URL = `${import.meta.env.BASE_URL}oxinfer.wasm`;
const REPO = 'https://github.com/KarpelesLab/oxinfer';
const PRESET = { repo: 'Qwen/Qwen3.5-0.8B', name: 'Qwen 3.5 0.8B' };

const params = new URLSearchParams(location.search);
const auto = params.get('auto') === '1';
const mode = ref(params.get('mode') === 'image' ? 'image' : 'chat');

// ---- device -------------------------------------------------------------------------------------
const device = reactive({ checked: false, gpuName: '', gpu: false });
const logs = ref([]);
function log(line) {
  logs.value.push(line);
  if (logs.value.length > 400) logs.value.shift();
  if (auto) {
    console.log(line);
    fetch('/log', { method: 'POST', body: line }).catch(() => {});
  }
}

async function probeDevice() {
  if (typeof navigator !== 'undefined' && navigator.gpu) {
    try {
      const a = await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' });
      if (a) {
        const i = a.info || {};
        device.gpu = true;
        // Browsers often hide the model name; the vendor alone still says which GPU it is.
        const vendor = i.vendor ? i.vendor.charAt(0).toUpperCase() + i.vendor.slice(1) : '';
        device.gpuName = i.description || i.device || (vendor ? `${/^[AEIOU]/.test(vendor) ? 'an' : 'a'} ${vendor} GPU` : 'a GPU');
      }
    } catch {
      /* no adapter */
    }
  }
  device.checked = true;
}

const graphicsLine = computed(() => {
  if (!device.checked) return 'Checking…';
  if (device.gpu && hasJspi) return `WebGPU on ${device.gpuName}. Prompts are processed on the GPU, replies on the CPU.`;
  if (device.gpu) return 'WebGPU is here, but this browser cannot pause WebAssembly while the GPU works (JSPI), so everything runs on the CPU. Chrome 137 or newer has both.';
  return 'No WebGPU in this browser, so everything runs on the CPU.';
});

// ---- model loading ------------------------------------------------------------------------------
const source = ref('preset'); // 'preset' | 'custom' | 'folder'
const custom = ref(params.get('model') || '');
const folderFiles = ref(null);
const precision = ref(params.get('quant') || 'q8');
const useGpu = ref(params.get('gpu') !== '0');
const phase = ref('idle'); // idle | loading | preparing | ready | generating
const error = ref('');
const progress = reactive({ file: '', index: 0, count: 0, got: 0, size: 0, phase: '' });
const modelName = ref('');
const storage = ref(0);

let engine = null;
let gpuHost = null;

if (params.get('model')) source.value = 'custom';

const canLoad = computed(() => {
  if (phase.value === 'loading' || phase.value === 'preparing' || phase.value === 'generating') return false;
  if (source.value === 'custom') return custom.value.trim().length > 0;
  if (source.value === 'folder') return !!folderFiles.value?.length;
  return true;
});

const progressLine = computed(() => {
  const p = progress;
  if (!p.count) return 'Finding the model files…';
  const verb = p.phase === 'download' ? 'Downloading' : 'Reading';
  const amount = p.size ? `${fmtBytes(p.got)} of ${fmtBytes(p.size)}` : fmtBytes(p.got);
  return `${verb} ${p.file} (file ${p.index + 1} of ${p.count}): ${amount}`;
});
const progressFraction = computed(() => (progress.size ? progress.got / progress.size : 0));

function fmtBytes(n) {
  if (n >= 1e9) return `${(n / 1e9).toFixed(2)} GB`;
  if (n >= 1e6) return `${(n / 1e6).toFixed(0)} MB`;
  if (n >= 1e3) return `${(n / 1e3).toFixed(0)} kB`;
  return `${n} B`;
}

function onFolder(e) {
  folderFiles.value = e.target.files;
}

const pause = () => new Promise((r) => setTimeout(r, 0));

async function load() {
  error.value = '';
  phase.value = 'loading';
  Object.assign(progress, { file: '', index: 0, count: 0, got: 0, size: 0, phase: '' });
  messages.value = [];
  stats.value = null;
  try {
    if (gpuHost) gpuHost.device.destroy();
    gpuHost = useGpu.value && device.gpu && hasJspi ? await WebGpuHost.create(log) : null;
    engine = await Oxinfer.create(WASM_URL, { gpu: gpuHost, log });
    const onProgress = (p) => Object.assign(progress, p);
    if (source.value === 'folder') {
      const first = folderFiles.value[0];
      modelName.value = first?.webkitRelativePath?.split('/')[0] || 'Local model';
      await loadFromFolder(engine, folderFiles.value, onProgress);
    } else {
      const spec = source.value === 'preset' ? PRESET.repo : custom.value;
      modelName.value = source.value === 'preset' ? PRESET.name : spec.split('/').filter(Boolean).pop();
      await loadFromUrl(engine, sourceBase(spec), onProgress);
    }
    phase.value = 'preparing';
    await nextTick();
    await pause();
    engine.load('/model', precision.value);
    engine.setSampling({ temperature: temperature.value, topK: 64, topP: 0.95, seed: Date.now() % 1e9 });
    phase.value = 'ready';
  } catch (e) {
    error.value = friendly(e);
    log(`[error] ${e.message || e}`);
    phase.value = 'idle';
    engine = null;
  }
  storage.value = await cachedBytes();
}

function friendly(e) {
  const m = String(e?.message || e);
  if (/unreachable|out of memory|allocation/i.test(m)) {
    return 'The model did not fit in this tab’s memory. Try a smaller model, or a folder written by oxinfer --save-quantized.';
  }
  if (/Failed to fetch|NetworkError/i.test(m)) return 'The download failed. Check the connection and that the URL allows cross-origin requests.';
  return m;
}

async function forgetDownloads() {
  await clearCache();
  storage.value = await cachedBytes();
}

// ---- chat ---------------------------------------------------------------------------------------
const messages = ref([]); // { role: 'user' | 'model', text }
const draft = ref(params.get('prompt') || '');
const temperature = ref(Number(params.get('temp') ?? 0.7));
const maxTokens = ref(Number(params.get('max') || 512));
const think = ref(false);
const stats = ref(null);
let stopRequested = false;
const transcript = ref(null);

async function send() {
  const text = draft.value.trim();
  if (!engine || !text || phase.value !== 'ready') return;
  draft.value = '';
  stopRequested = false;
  phase.value = 'generating';
  messages.value.push({ role: 'user', text });
  messages.value.push({ role: 'model', text: '' });
  const reply = messages.value[messages.value.length - 1];
  await scrollDown();
  try {
    engine.setSampling({ temperature: temperature.value, topK: 64, topP: 0.95, seed: Date.now() % 1e9 });
    const t0 = performance.now();
    const promptTokens = await engine.chatTurn(text, { think: think.value, maxTokens: maxTokens.value });
    const t1 = performance.now();
    let tokens = 0;
    while (!stopRequested && (await engine.next()) >= 0) {
      tokens++;
      const t = engine.text();
      if (!t.endsWith('�')) reply.text = t;
      await pause(); // a real task boundary, so the browser paints between tokens
      if (tokens % 8 === 0) scrollDown();
    }
    reply.text = engine.text();
    const t2 = performance.now();
    stats.value = { promptTokens, prefill: promptTokens / ((t1 - t0) / 1e3), tokens, decode: tokens / Math.max(1e-3, (t2 - t1) / 1e3), stopped: stopRequested };
    log(`[prefill ${promptTokens} tok at ${stats.value.prefill.toFixed(1)} tok/s | decode ${tokens} tok at ${stats.value.decode.toFixed(1)} tok/s]`);
    if (auto) log(`[output] ${reply.text.replace(/\n/g, '\\n')}`);
  } catch (e) {
    error.value = friendly(e);
    log(`[error] ${e.message || e}`);
  }
  phase.value = 'ready';
  scrollDown();
}

function stop() {
  stopRequested = true;
}

function newConversation() {
  if (!engine || phase.value === 'generating') return;
  engine.reset();
  messages.value = [];
  stats.value = null;
}

function onKey(e) {
  if (e.key === 'Enter' && !e.shiftKey && !e.isComposing) {
    e.preventDefault();
    send();
  }
}

async function scrollDown() {
  await nextTick();
  const el = transcript.value;
  if (el) el.scrollTop = el.scrollHeight;
}

onMounted(async () => {
  await probeDevice();
  storage.value = await cachedBytes();
  if (auto) log(`[browser: ${navigator.userAgent}; jspi ${hasJspi}; webgpu ${device.gpu ? device.gpuName : 'none'}]`);
  if (auto && mode.value === 'chat') {
    await load();
    for (let i = 0; i < Number(params.get('repeat') || 1) && engine; i++) {
      newConversation();
      draft.value = params.get('prompt') || 'Hello';
      await send();
    }
    fetch('/exit', { method: 'POST' }).catch(() => {});
  }
});
</script>

<template>
  <header class="masthead">
    <h1 class="wordmark">oxinfer</h1>
    <p class="lede">Chat with a language model or make images, all in this tab. The weights download once and stay in this browser; nothing you type leaves the page.</p>
    <nav class="nav">
      <div class="modes" role="group" aria-label="Mode">
        <button type="button" :aria-pressed="mode === 'chat'" @click="mode = 'chat'">Chat</button>
        <button type="button" :aria-pressed="mode === 'image'" @click="mode = 'image'">Image</button>
      </div>
      <a class="repo" :href="REPO">Source on GitHub</a>
    </nav>
  </header>

  <ImageStudio v-if="mode === 'image'" :device="device" :auto="auto" :log="log" @storage="(v) => (storage = v)" />
  <main v-else class="layout">
    <aside class="side">
      <section aria-labelledby="model-h">
        <h2 id="model-h">Model</h2>
        <fieldset class="choices" :disabled="phase === 'loading' || phase === 'preparing' || phase === 'generating'">
          <legend class="sr-only">Where the model comes from</legend>
          <label class="choice">
            <input v-model="source" type="radio" value="preset" />
            <span>
              <strong>Qwen 3.5 0.8B</strong>
              <small>1.8 GB from Hugging Face, made 8-bit in the browser. Works on any recent browser.</small>
            </span>
          </label>
          <label class="choice">
            <input v-model="source" type="radio" value="custom" />
            <span>
              <strong>Another model</strong>
              <small>A Hugging Face repo id or a URL to a model folder.</small>
            </span>
          </label>
          <input
            v-if="source === 'custom'"
            v-model="custom"
            class="field"
            type="text"
            placeholder="org/name, https://… or ./folder"
            aria-label="Hugging Face repo id or folder URL"
            spellcheck="false"
          />
          <label class="choice">
            <input v-model="source" type="radio" value="folder" />
            <span>
              <strong>A folder on this computer</strong>
              <small>Gemma 4 fits only this way: write the folder with <code>oxinfer --save-quantized</code>.</small>
            </span>
          </label>
          <input v-if="source === 'folder'" class="field" type="file" webkitdirectory directory multiple aria-label="Model folder" @change="onFolder" />
        </fieldset>

        <div class="row">
          <label class="inline">
            Precision
            <select v-model="precision" :disabled="phase !== 'idle' && phase !== 'ready'">
              <option value="q8">8-bit</option>
              <option value="q4">4-bit</option>
              <option value="bf16">16-bit</option>
            </select>
          </label>
          <label v-if="device.gpu && hasJspi" class="inline"><input v-model="useGpu" type="checkbox" /> Use the GPU</label>
        </div>
        <p class="hint">Precision applies to original checkpoints. Folders from <code>--save-quantized</code> keep their own.</p>

        <button class="primary" type="button" :disabled="!canLoad" @click="load">
          {{ phase === 'ready' || phase === 'generating' ? 'Load again' : 'Load model' }}
        </button>

        <div v-if="phase === 'loading'" class="progress" role="status">
          <div class="bar"><div class="fill" :style="{ width: `${(progressFraction * 100).toFixed(1)}%` }"></div></div>
          <p>{{ progressLine }}</p>
        </div>
        <p v-else-if="phase === 'preparing'" class="status" role="status">Preparing the weights. The page may not respond for a few seconds.</p>
        <p v-else-if="phase === 'ready' || phase === 'generating'" class="status ready" role="status">{{ modelName }} is ready.</p>
        <p v-if="error" class="error" role="alert">{{ error }}</p>
      </section>

      <section aria-labelledby="device-h">
        <h2 id="device-h">This device</h2>
        <p class="device">{{ graphicsLine }}</p>
        <p class="device">The browser build uses one CPU thread with SIMD, so replies are slower than the native app.</p>
      </section>

      <section v-if="stats" aria-labelledby="speed-h">
        <h2 id="speed-h">Last reply</h2>
        <dl class="speed">
          <dt>Prompt</dt>
          <dd>{{ stats.promptTokens }} tokens at {{ stats.prefill.toFixed(0) }} per second</dd>
          <dt>Reply</dt>
          <dd>{{ stats.tokens }} tokens at {{ stats.decode.toFixed(1) }} per second{{ stats.stopped ? ', stopped' : '' }}</dd>
        </dl>
      </section>

      <details class="more">
        <summary>Settings</summary>
        <label class="inline">Temperature <input v-model.number="temperature" class="num" type="number" min="0" max="2" step="0.1" /></label>
        <label class="inline">Longest reply <input v-model.number="maxTokens" class="num" type="number" min="1" max="8192" step="1" /> tokens</label>
        <label class="inline"><input v-model="think" type="checkbox" /> Let the model think first</label>
      </details>

      <details class="more">
        <summary>Log</summary>
        <pre class="log">{{ logs.join('\n') || 'Nothing yet.' }}</pre>
      </details>

      <p v-if="storage > 1e6" class="hint">
        Downloaded models use {{ fmtBytes(storage) }} of this browser’s storage.
        <button class="link" type="button" @click="forgetDownloads">Delete them</button>
      </p>
    </aside>

    <section class="chat" aria-label="Conversation">
      <div ref="transcript" class="transcript" aria-live="polite">
        <p v-if="!messages.length" class="empty">
          {{ phase === 'ready' ? 'Ask it something.' : 'Load a model, then ask it something.' }}
        </p>
        <article v-for="(m, i) in messages" :key="i" :class="['msg', m.role]">
          <h3 class="who">{{ m.role === 'user' ? 'You' : modelName }}</h3>
          <p v-if="m.role === 'user'" class="said">{{ m.text }}</p>
          <Reply v-else-if="m.text" :text="m.text" />
          <p v-else class="thinking">Reading your message…</p>
        </article>
      </div>

      <form class="composer" @submit.prevent="send">
        <textarea
          v-model="draft"
          rows="2"
          :placeholder="phase === 'ready' || phase === 'generating' ? 'Message' : 'Load a model first'"
          aria-label="Message"
          :disabled="!engine"
          @keydown="onKey"
        ></textarea>
        <div class="actions">
          <button v-if="phase === 'generating'" type="button" class="primary" @click="stop">Stop</button>
          <button v-else type="submit" class="primary" :disabled="phase !== 'ready' || !draft.trim()">Send</button>
          <button type="button" class="quiet" :disabled="phase !== 'ready' || !messages.length" @click="newConversation">New conversation</button>
        </div>
      </form>
    </section>
  </main>
</template>

<style scoped>
.masthead {
  display: grid;
  grid-template-columns: auto 1fr auto;
  align-items: baseline;
  gap: 0 1.5rem;
  padding: 1.5rem clamp(1rem, 4vw, 2.5rem) 1rem;
  border-bottom: 1px solid var(--line);
}
.wordmark {
  margin: 0;
  font-family: var(--display);
  font-size: clamp(2.4rem, 6vw, 3.6rem);
  font-weight: 750;
  font-stretch: 78%;
  letter-spacing: -0.035em;
  line-height: 0.9;
  color: var(--verdigris-deep);
}
.lede {
  margin: 0;
  max-width: 52ch;
  color: var(--muted);
}
.nav {
  display: flex;
  align-items: center;
  gap: 1.25rem;
}
.repo {
  white-space: nowrap;
}
.modes {
  display: inline-flex;
  border: 1px solid var(--line);
  border-radius: 4px;
  overflow: hidden;
}
.modes button {
  padding: 0.35rem 0.9rem;
  border: 0;
  background: transparent;
  font-family: var(--display);
  font-weight: 600;
}
.modes button + button {
  border-left: 1px solid var(--line);
}
.modes button[aria-pressed='true'] {
  background: var(--verdigris-deep);
  color: var(--paper);
}

.layout {
  flex: 1;
  min-height: 0;
  display: grid;
  grid-template-columns: minmax(17rem, 23rem) 1fr;
}
.side {
  overflow-y: auto;
  padding: 1.25rem clamp(1rem, 3vw, 2rem) 2rem;
  border-right: 1px solid var(--line);
}
.side section + section,
.side section + details,
.side details + details,
.side details + p {
  margin-top: 1.5rem;
}
h2 {
  margin: 0 0 0.6rem;
  font-family: var(--display);
  font-size: 1.15rem;
  font-weight: 650;
  letter-spacing: -0.01em;
}
.choices {
  border: 0;
  margin: 0;
  padding: 0;
  display: grid;
  gap: 0.5rem;
}
.choice {
  display: grid;
  grid-template-columns: auto 1fr;
  gap: 0.6rem;
  align-items: start;
  cursor: pointer;
}
.choice input {
  margin-top: 0.35rem;
  accent-color: var(--verdigris);
}
.choice small {
  display: block;
  color: var(--muted);
  font-size: 0.85rem;
  line-height: 1.4;
}
.field {
  width: 100%;
  padding: 0.45rem 0.6rem;
  border: 1px solid var(--line);
  border-radius: 4px;
  background: var(--surface);
}
.row {
  display: flex;
  flex-wrap: wrap;
  gap: 0.5rem 1.25rem;
  margin-top: 1rem;
}
.inline {
  display: flex;
  align-items: center;
  gap: 0.45rem;
  margin: 0.35rem 0;
}
.inline select,
.num {
  padding: 0.25rem 0.4rem;
  border: 1px solid var(--line);
  border-radius: 4px;
  background: var(--surface);
}
.num {
  width: 5.5rem;
}
.hint {
  margin: 0.4rem 0 0;
  color: var(--muted);
  font-size: 0.85rem;
}
button {
  cursor: pointer;
}
button:disabled {
  cursor: default;
  opacity: 0.45;
}
.primary {
  margin-top: 1rem;
  padding: 0.55rem 1.2rem;
  border: 0;
  border-radius: 4px;
  background: var(--verdigris-deep);
  color: var(--paper);
  font-weight: 650;
}
.quiet {
  margin-top: 1rem;
  padding: 0.5rem 0.9rem;
  border: 1px solid var(--line);
  border-radius: 4px;
  background: transparent;
}
.link {
  padding: 0;
  border: 0;
  background: none;
  color: var(--verdigris-deep);
  text-decoration: underline;
  text-underline-offset: 0.15em;
}
.progress p,
.status,
.error {
  margin: 0.6rem 0 0;
  font-size: 0.9rem;
}
.bar {
  height: 6px;
  margin-top: 1rem;
  background: var(--line);
  border-radius: 3px;
  overflow: hidden;
}
.fill {
  height: 100%;
  background: var(--verdigris);
  transition: width 0.2s linear;
}
.ready {
  color: var(--verdigris-deep);
}
.error {
  color: var(--error);
}
.device {
  margin: 0 0 0.5rem;
  font-size: 0.92rem;
}
.speed {
  display: grid;
  grid-template-columns: auto 1fr;
  gap: 0.15rem 0.75rem;
  margin: 0;
  font-size: 0.92rem;
  font-variant-numeric: tabular-nums;
}
.speed dt {
  color: var(--muted);
}
.speed dd {
  margin: 0;
}
.more summary {
  cursor: pointer;
  font-family: var(--display);
  font-weight: 600;
}
.log {
  max-height: 14rem;
  overflow: auto;
  margin: 0.5rem 0 0;
  padding: 0.6rem;
  background: var(--surface);
  border: 1px solid var(--line);
  border-radius: 4px;
  font-size: 0.75rem;
  white-space: pre-wrap;
  overflow-wrap: anywhere;
}

.chat {
  display: grid;
  grid-template-rows: 1fr auto;
  min-height: 0;
}
.transcript {
  overflow-y: auto;
  padding: 1.5rem clamp(1rem, 4vw, 3rem);
}
.empty {
  margin: 3rem 0 0;
  color: var(--muted);
  font-family: var(--display);
  font-size: 1.4rem;
}
.msg {
  max-width: 68ch;
  margin: 0 0 1.6rem;
}
.who {
  margin: 0 0 0.2rem;
  font-family: var(--display);
  font-size: 0.85rem;
  font-weight: 600;
  color: var(--muted);
}
.user .said {
  margin: 0;
  padding-left: 0.8rem;
  border-left: 3px solid var(--verdigris);
  white-space: pre-wrap;
  overflow-wrap: anywhere;
}
.model :deep(.reply) {
  line-height: 1.6;
}
.thinking {
  margin: 0;
  color: var(--muted);
}
.composer {
  padding: 0.75rem clamp(1rem, 4vw, 3rem) 1rem;
  border-top: 1px solid var(--line);
  background: var(--surface);
}
.composer textarea {
  display: block;
  width: 100%;
  max-width: 68ch;
  min-height: 3.2rem;
  resize: vertical;
  padding: 0.6rem 0.75rem;
  border: 1px solid var(--line);
  border-radius: 4px;
  background: var(--paper);
}
.actions {
  display: flex;
  gap: 0.6rem;
}
.actions .primary,
.actions .quiet {
  margin-top: 0.6rem;
}
.sr-only {
  position: absolute;
  width: 1px;
  height: 1px;
  overflow: hidden;
  clip: rect(0 0 0 0);
}

@media (max-width: 760px) {
  .masthead {
    grid-template-columns: 1fr auto;
  }
  .lede {
    grid-column: 1 / -1;
    grid-row: 2;
    margin-top: 0.5rem;
  }
  .layout {
    grid-template-columns: 1fr;
    height: auto;
  }
  .side {
    border-right: 0;
    border-bottom: 1px solid var(--line);
  }
  .chat {
    min-height: 70dvh;
  }
}
</style>
