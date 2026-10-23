// Development server for the browser build: serves this directory, mounts a local model directory
// at /models/, and prints lines posted to /log (used by index.html?auto=1 for headless runs).
//   deno run --allow-net --allow-read web/serve.ts ~/.cache/gemmars/hub [port]
const models = Deno.args[0] ? Deno.args[0].replace(/\/$/, '') : null;
const port = Number(Deno.args[1] || 8080);
const root = new URL('.', import.meta.url).pathname.replace(/\/$/, '');
const types: Record<string, string> = { html: 'text/html', js: 'text/javascript', mjs: 'text/javascript', wasm: 'application/wasm', json: 'application/json', css: 'text/css' };
Deno.serve({ port, onListen: ({ port }) => console.error(`serving ${root} on http://localhost:${port}/${models ? ` (models from ${models})` : ''}`) }, async (req) => {
  const url = new URL(req.url);
  if (req.method === 'POST' && url.pathname === '/log') { console.log(await req.text()); return new Response('ok'); }
  if (req.method === 'POST' && url.pathname === '/exit') { setTimeout(() => Deno.exit(0), 100); return new Response('ok'); }
  let path = url.pathname === '/' ? '/index.html' : url.pathname;
  let file = root + path;
  if (path.startsWith('/models/')) {
    if (!models) return new Response('no model directory configured', { status: 404 });
    file = models + path.slice('/models'.length);
  }
  if (file.includes('/../')) return new Response('bad path', { status: 400 });
  try {
    const stat = await Deno.stat(file);
    const headers: Record<string, string> = { 'content-type': types[file.split('.').pop() || ''] || 'application/octet-stream', 'content-length': String(stat.size), 'cache-control': 'no-cache' };
    if (req.method === 'HEAD') return new Response(null, { headers });
    return new Response((await Deno.open(file)).readable, { headers });
  } catch {
    return new Response('not found', { status: 404 });
  }
});
