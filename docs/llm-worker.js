import {
  pipeline,
  env,
  AutoModelForCausalLM,
  AutoTokenizer
} from "https://cdn.jsdelivr.net/npm/@huggingface/transformers/dist/transformers.web.js";

let generator = null;
let currentModelLabel = null;

const REMOTE_MODEL_FILES = [
  "config.json",
  "generation_config.json",
  "special_tokens_map.json",
  "tokenizer.json",
  "tokenizer_config.json",
  "added_tokens.json",
  "vocab.json",
  "onnx/model_q4.onnx"
];

async function fetchWithProgress(url, onChunk) {
  const response = await fetch(url);
  if (!response.ok) {
    throw new Error(`Failed to fetch ${url}: ${response.status} ${response.statusText}`);
  }

  const total = Number(response.headers.get("content-length")) || 0;
  if (!response.body) {
    const buf = await response.arrayBuffer();
    onChunk(buf.byteLength, total);
    return buf;
  }

  const reader = response.body.getReader();
  const chunks = [];
  let received = 0;

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    chunks.push(value);
    received += value.byteLength;
    onChunk(received, total);
  }

  const out = new Uint8Array(received);
  let offset = 0;
  for (const chunk of chunks) {
    out.set(chunk, offset);
    offset += chunk.byteLength;
  }
  return out.buffer;
}

async function loadRemoteModel(modelId, onProgress) {
  env.allowRemoteModels = false;
  env.allowLocalModels = true;

  const baseUrl = `https://huggingface.co/${modelId}/resolve/main`;
  const fileMap = {};
  let loadedBytes = 0;
  let loadedFiles = 0;
  const expectedBytes = new Map();

  // First collect sizes, then fetch sequentially so we can report real progress.
  for (const file of REMOTE_MODEL_FILES) {
    let size = 0;
    try {
      const head = await fetch(`${baseUrl}/${file}`, { method: "HEAD" });
      size = Number(head.headers.get("content-length")) || 0;
    } catch {
      size = 0;
    }
    expectedBytes.set(file, size);
  }

  const totalBytes = [...expectedBytes.values()].reduce((sum, size) => sum + size, 0);
  const fileCount = REMOTE_MODEL_FILES.length;

  for (let i = 0; i < REMOTE_MODEL_FILES.length; i += 1) {
    const file = REMOTE_MODEL_FILES[i];
    const size = expectedBytes.get(file) || 0;
    onProgress({
      phase: "downloading",
      file,
      loadedBytes,
      totalBytes,
      loadedFiles,
      fileIndex: i,
      fileCount,
      fileLoadedBytes: 0,
      fileTotalBytes: size
    });

    const buf = await fetchWithProgress(`${baseUrl}/${file}`, (fileLoadedBytes) => {
      const currentTotal = loadedBytes + fileLoadedBytes;
      const currentFiles = loadedFiles + (fileLoadedBytes > 0 ? 0 : 0);
      onProgress({
        phase: "downloading",
        file,
        loadedBytes: currentTotal,
        totalBytes,
        loadedFiles: currentFiles,
        fileIndex: i,
        fileCount,
        fileLoadedBytes,
        fileTotalBytes: size
      });
    });

    fileMap[file] = buf;
    loadedBytes += buf.byteLength;
    loadedFiles += 1;
  }

  onProgress({
    phase: "initializing",
    file: null,
    loadedBytes: totalBytes,
    totalBytes,
    loadedFiles: fileCount,
    fileIndex: fileCount,
    fileCount,
    fileLoadedBytes: totalBytes,
    fileTotalBytes: totalBytes
  });

  const model = await AutoModelForCausalLM.from_pretrained("local-model", {
    model_file: "onnx/model_q4.onnx",
    config_file: "config.json",
    file_system: {
      async readFile(path) {
        const buf = fileMap[path];
        if (!buf) throw new Error("File not found: " + path);
        return new Uint8Array(buf);
      },
      async readdir() {
        return Object.keys(fileMap);
      },
      async stat(path) {
        if (!fileMap[path]) throw new Error("File not found: " + path);
        return { size: fileMap[path].byteLength };
      }
    },
    dtype: "q4"
  });

  const tokenizer = await AutoTokenizer.from_pretrained("local-tokenizer", {
    tokenizer_file: "tokenizer.json",
    file_system: {
      async readFile(path) {
        const buf = fileMap[path];
        if (!buf) throw new Error("File not found: " + path);
        return new Uint8Array(buf);
      },
      async readdir() {
        return Object.keys(fileMap);
      },
      async stat(path) {
        if (!fileMap[path]) throw new Error("File not found: " + path);
        return { size: fileMap[path].byteLength };
      }
    }
  });

  return await pipeline("text-generation", model, tokenizer);
}

async function loadFromFiles(fileMap) {
  const fs = {
    async readFile(path) {
      const buf = fileMap[path];
      if (!buf) throw new Error("File not found: " + path);
      return new Uint8Array(buf);
    },
    async readdir() {
      return Object.keys(fileMap);
    },
    async stat(path) {
      if (!fileMap[path]) throw new Error("File not found: " + path);
      return { size: fileMap[path].byteLength };
    }
  };

  const model = await AutoModelForCausalLM.from_pretrained("local-model", {
    model_file: "model.safetensors",
    config_file: "config.json",
    file_system: fs,
    dtype: "q4"
  });

  const tokenizer = await AutoTokenizer.from_pretrained("local-tokenizer", {
    tokenizer_file: "tokenizer.json",
    file_system: fs
  });

  return await pipeline("text-generation", model, tokenizer);
}

function extractJSON(text) {
  const match = text.match(/\{[\s\S]*\}/);
  return match ? match[0] : null;
}

onmessage = async (e) => {
  const { type } = e.data;

  if (type === "load-remote") {
    try {
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
