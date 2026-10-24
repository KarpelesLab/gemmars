// Minimal harness: pick or download a model, load it, chat with streaming output and stats.
//
// Launch arguments for unattended runs (also from `xcrun simctl launch`):
//   -model PATH_OR_REPO  -options "quant=q8"  -prompt "text"  [-cacheDir PATH]
// load the model and send the prompt at startup, logging everything to stdout.

import SwiftUI

@MainActor
final class Harness: ObservableObject {
    @Published var spec = "Qwen/Qwen3.5-0.8B"
    @Published var options = "quant=q8\nmax_context=4096"
    @Published var prompt = "Write a haiku about the sea."
    @Published var output = ""
    @Published var status = "idle"
    @Published var stats = ""
    @Published var busy = false
    @Published var localModels: [String] = []

    private var engine: OxinferEngine?
    private let worker = DispatchQueue(label: "oxinfer", qos: .userInitiated)

    /// Models live in Documents (visible in the Files app and Finder file sharing); downloads go
    /// to Documents/hub, excluded from backups.
    let documents = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
    var hubDir: URL { documents.appendingPathComponent("hub", isDirectory: true) }

    init() {
        let d = UserDefaults.standard
        if let m = d.string(forKey: "model") { spec = m }
        if let o = d.string(forKey: "options") { options = o }
        if let p = d.string(forKey: "prompt") { prompt = p }
        try? FileManager.default.createDirectory(at: hubDir, withIntermediateDirectories: true)
        var url = hubDir
        var rv = URLResourceValues()
        rv.isExcludedFromBackup = true
        try? url.setResourceValues(rv)
        refreshLocal()
        log("oxinfer \(OxinferEngine.version)\n\(OxinferEngine.devices)")
        if d.string(forKey: "model") != nil {
            loadAndSend()
        }
    }

    func log(_ s: String) {
        print(s)
        fflush(stdout)
    }

    /// Model directories (with config.json) and .gguf files in Documents and Documents/hub.
    func refreshLocal() {
        var found: [String] = []
        let fm = FileManager.default
        for dir in [documents, hubDir] {
            for name in (try? fm.contentsOfDirectory(atPath: dir.path)) ?? [] {
                let p = dir.appendingPathComponent(name)
                if name.hasSuffix(".gguf") || fm.fileExists(atPath: p.appendingPathComponent("config.json").path) {
                    found.append(p.path)
                }
                // GGUF files downloaded as org--repo/FILE.gguf.
                if dir == hubDir, let inner = try? fm.contentsOfDirectory(atPath: p.path) {
                    found += inner.filter { $0.hasSuffix(".gguf") }.map { p.appendingPathComponent($0).path }
                }
            }
        }
        localModels = found.sorted()
    }

    func load(then: (() -> Void)? = nil) {
        let spec = spec, cache = hubDir.path
        let opts = options + "\ncache_dir=\(cache)"
        busy = true
        status = "loading \(spec)..."
        engine = nil
        worker.async {
            do {
                var lastPct = -1
                let e = try OxinferEngine(model: spec, options: opts) { file, done, total, i, n in
                    let pct = total > 0 ? Int(done * 100 / total) : -1
                    if pct != lastPct {
                        lastPct = pct
                        let line = "downloading \(file) [\(i)/\(n)] \(pct)%"
                        DispatchQueue.main.async { self.status = line }
                    }
                    return true
                }
                let info = e.infoJSON
                DispatchQueue.main.async {
                    self.engine = e
                    self.busy = false
                    self.status = "loaded"
                    self.stats = info
                    self.log("[loaded] \(info)")
                    self.refreshLocal()
                    then?()
                }
            } catch {
                DispatchQueue.main.async {
                    self.busy = false
                    self.status = "error: \(error)"
                    self.log("[error] \(error)")
                }
            }
        }
    }

    func send() {
        guard let e = engine else { return }
        let text = prompt
        busy = true
        status = "generating"
        output += (output.isEmpty ? "" : "\n\n") + "> \(text)\n"
        log("> \(text)")
        worker.async {
            do {
                // Thinking text is streamed like the answer (Qwen's thinking is off by default).
                let finish = try e.send(text) { piece, _ in
                    print(piece, terminator: "")
                    fflush(stdout)
                    DispatchQueue.main.async { self.output += piece }
                }
                let s = e.stats
                let line = String(
                    format: "prefill %d tok %.1f tok/s | decode %d tok %.1f tok/s | accepted %.0f%% | ctx %d | %@",
                    s["prompt_tokens"] as? Int ?? 0, s["prefill_tok_s"] as? Double ?? 0,
                    s["generated_tokens"] as? Int ?? 0, s["decode_tok_s"] as? Double ?? 0,
                    100 * (s["acceptance"] as? Double ?? 0), s["context_tokens"] as? Int ?? 0, "\(finish)")
                DispatchQueue.main.async {
                    self.busy = false
                    self.status = "done"
                    self.stats = line
                    self.log("\n[stats] \(line)")
                }
            } catch {
                DispatchQueue.main.async {
                    self.busy = false
                    self.status = "error: \(error)"
                    self.log("[error] \(error)")
                }
            }
        }
    }

    func loadAndSend() { load { self.send() } }
    func stop() { engine?.cancel() }
    func reset() {
        engine?.reset()
        output = ""
    }
}

struct ContentView: View {
    @StateObject private var h = Harness()

    var body: some View {
        NavigationStack {
            Form {
                Section("Model") {
                    TextField("repo id, org/repo:QUANT or path", text: $h.spec)
                        .textInputAutocapitalization(.never).autocorrectionDisabled()
                    if !h.localModels.isEmpty {
                        Picker("On device", selection: $h.spec) {
                            Text("-").tag("")
                            ForEach(h.localModels, id: \.self) { Text(($0 as NSString).lastPathComponent).tag($0) }
                        }
                    }
                    TextField("options (key=value lines)", text: $h.options, axis: .vertical)
                        .font(.system(.footnote, design: .monospaced))
                        .textInputAutocapitalization(.never).autocorrectionDisabled()
                    Button("Load") { h.load() }.disabled(h.busy)
                }
                Section("Chat") {
                    TextField("prompt", text: $h.prompt, axis: .vertical)
                    HStack {
                        Button("Send") { h.send() }.disabled(h.busy)
                        Spacer()
                        Button("Stop") { h.stop() }
                        Spacer()
                        Button("Reset") { h.reset() }.disabled(h.busy)
                    }.buttonStyle(.borderless)
                    Text(h.output.isEmpty ? " " : h.output)
                        .font(.system(.body, design: .monospaced))
                        .textSelection(.enabled)
                }
                Section("Status") {
                    Text(h.status).font(.footnote)
                    Text(h.stats).font(.system(.caption, design: .monospaced))
                }
            }
            .navigationTitle("oxinfer")
        }
    }
}
