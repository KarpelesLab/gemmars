import { defineConfig } from 'vite';
import vue from '@vitejs/plugin-vue';
import { createReadStream, statSync } from 'node:fs';
import { join, normalize, sep } from 'node:path';

// GitHub Pages serves a project site under /<repo>/. Override with PAGES_BASE ("/" for local use).
const base = process.env.PAGES_BASE || '/oxinfer/';

// Local testing: OXINFER_MODELS=~/.cache/oxinfer/hub npm run dev serves that directory at
// <base>models/, so "./models/<folder>" works as a model source without uploading anything.
function localModels() {
  const root = process.env.OXINFER_MODELS;
  const mount = `${base}models/`;
  const handler = (req, res, next) => {
    if (!root || !req.url.startsWith(mount)) return next();
    const rel = decodeURIComponent(req.url.slice(mount.length).split('?')[0]);
    const file = normalize(join(root, rel));
    if (!file.startsWith(normalize(root) + sep)) return next();
    let st;
    try {
      st = statSync(file);
    } catch {
      res.statusCode = 404;
      return res.end('not found');
    }
    if (!st.isFile()) return next();
    res.setHeader('content-length', String(st.size));
    res.setHeader('content-type', file.endsWith('.json') ? 'application/json' : 'application/octet-stream');
    createReadStream(file).pipe(res);
  };
  return {
    name: 'oxinfer-local-models',
    // Block bodies: a function returned from these hooks would be run as a post-middleware hook.
    configureServer(s) {
      s.middlewares.use(handler);
    },
    configurePreviewServer(s) {
      s.middlewares.use(handler);
    },
  };
}

export default defineConfig({
  base,
  plugins: [vue(), localModels()],
  build: {
    target: 'es2022',
    // The .wasm lives in public/ and is fetched at run time, not bundled.
    assetsInlineLimit: 0,
  },
});
