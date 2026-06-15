import * as transformersLib from "./vendor/transformers.min.js?v=20260615-2";

const {
  pipeline,
  env,
  AutoModelForCausalLM,
  AutoTokenizer
} = transformersLib;

postMessage({ type: "diagnostic", message: "worker module loaded" });

let generator = null;
let currentModelLabel = null;

async function loadRemoteModel(modelId, onProgress) {
  env.allowRemoteModels = true;
  env.allowLocalModels = false;

  const forwardProgress = (event, label) => {
    if (!event) return;
    if (event.status === "initiate") {
      onProgress({
        phase: "starting",
        file: label || event.name || null,
        loadedBytes: 0,
        totalBytes: event.total || 0,
        loadedFiles: 0,
        fileIndex: 0,
        fileCount: 1,
        fileLoadedBytes: 0,
        fileTotalBytes: event.total || 0
      });
      return;
    }
    if (event.status === "progress" || event.status === "progress_total") {
      onProgress({
        phase: "downloading",
        file: label || event.name || null,
        loadedBytes: event.loaded || 0,
        totalBytes: event.total || 0,
        loadedFiles: 0,
        fileIndex: 0,
        fileCount: 1,
        fileLoadedBytes: event.loaded || 0,
        fileTotalBytes: event.total || 0
      });
      return;
    }
    if (event.status === "ready") {
      onProgress({ kind: "diagnostic", message: `${label || "file"} ready` });
    }
  };

  onProgress({ kind: "diagnostic", message: `starting remote model load: ${modelId}` });
  onProgress({ kind: "diagnostic", message: "building pipeline" });
  return await pipeline("text-generation", modelId, {
    dtype: "q4",
    progress_callback: (event) => forwardProgress(event, "model")
  });
}

async function loadFromFiles(modelId, fileMap) {
  env.allowRemoteModels = true;
  env.allowLocalModels = true;
  env.useBrowserCache = true;

  const cache = await caches.open(env.cacheKey);
  const baseUrl = `https://huggingface.co/${modelId}/resolve/main`;

  for (const [name, buf] of Object.entries(fileMap)) {
    const headers = new Headers();
    headers.set("Content-Length", String(buf.byteLength));
    headers.set("Content-Type", name.endsWith(".json") ? "application/json" : "application/octet-stream");
    await cache.put(`${baseUrl}/${name}`, new Response(new Uint8Array(buf), { headers }));
  }

  return await pipeline("text-generation", modelId, {
    dtype: "q4"
  });
}

function extractJSON(text) {
  const match = text.match(/\{[\s\S]*\}/);
  return match ? match[0] : null;
}

onmessage = async (e) => {
  const { type } = e.data;

  if (type === "load-remote") {
    try {
      postMessage({ type: "diagnostic", message: `worker received load-remote: ${e.data.modelId}` });
      postMessage({ type: "status", message: "小さいモデルを取得中..." });
      currentModelLabel = e.data.modelId;
      generator = await loadRemoteModel(e.data.modelId, (progress) => {
        postMessage({ type: "progress", ...progress });
      });
      postMessage({ type: "status", message: "モデル準備完了: " + e.data.modelId });
    } catch (err) {
      postMessage({ type: "error", message: err.message || String(err) });
    }
    return;
  }

  if (type === "load-model") {
    try {
      postMessage({ type: "diagnostic", message: "worker received load-model" });
      postMessage({ type: "status", message: "モデル読み込み中..." });
      const fileMap = e.data.fileMap;
      currentModelLabel = "local-upload";
      generator = await loadFromFiles(fileMap);
      postMessage({ type: "status", message: "モデル準備完了" });
    } catch (err) {
      postMessage({ type: "error", message: err.message || String(err) });
    }
    return;
  }

  if (type === "run") {
    if (!generator) {
      postMessage({ type: "error", message: "モデルが読み込まれていません" });
      return;
    }
    try {
      postMessage({ type: "status", message: "LLM 実行中..." });
      const out = await generator(e.data.prompt, { max_new_tokens: 120 });
      const raw = out[0].generated_text;
      const json = extractJSON(raw);
      if (!json) {
        postMessage({ type: "error", message: "JSON が見つかりませんでした" });
        return;
      }
      postMessage({ type: "result", json });
    } catch (err) {
      postMessage({ type: "error", message: err.message || String(err) });
    }
  }
};
