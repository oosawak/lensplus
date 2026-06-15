importScripts("./vendor/transformers.min.js?v=20260615-1");

const transformersLib = self.transformers || transformers;
const {
  pipeline,
  env,
  AutoModelForCausalLM,
  AutoTokenizer
} = transformersLib;

postMessage({ type: "diagnostic", message: "worker script loaded" });

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

function downloadWithProgress(url, onProgress) {
  return new Promise((resolve, reject) => {
    const xhr = new XMLHttpRequest();
    xhr.open("GET", url, true);
    xhr.responseType = "arraybuffer";

    xhr.onprogress = (event) => {
      onProgress({
        loadedBytes: event.loaded || 0,
        totalBytes: event.lengthComputable ? event.total : 0
      });
    };

    xhr.onload = () => {
      if (xhr.status >= 200 && xhr.status < 300) {
        onProgress({
          loadedBytes: xhr.response ? xhr.response.byteLength : 0,
          totalBytes: xhr.response ? xhr.response.byteLength : 0
        });
        resolve(xhr.response);
      } else {
        reject(new Error(`Failed to fetch ${url}: ${xhr.status} ${xhr.statusText}`));
      }
    };

    xhr.onerror = () => reject(new Error(`Failed to fetch ${url}`));
    xhr.send();
  });
}

async function loadRemoteModel(modelId, onProgress) {
  env.allowRemoteModels = false;
  env.allowLocalModels = true;

  const baseUrl = `https://huggingface.co/${modelId}/resolve/main`;
  const fileMap = {};
  let loadedBytes = 0;
  let loadedFiles = 0;
  const fileCount = REMOTE_MODEL_FILES.length;
  const estimatedBytes = new Map();

  for (let i = 0; i < REMOTE_MODEL_FILES.length; i += 1) {
    const file = REMOTE_MODEL_FILES[i];
    onProgress({ kind: "diagnostic", message: `start download: ${file}` });
    onProgress({
      phase: "starting",
      file,
      loadedBytes,
      totalBytes: [...estimatedBytes.values()].reduce((sum, size) => sum + size, 0),
      loadedFiles,
      fileIndex: i,
      fileCount,
      fileLoadedBytes: 0,
      fileTotalBytes: estimatedBytes.get(file) || 0
    });

    const buf = await downloadWithProgress(`${baseUrl}/${file}`, (event) => {
      const currentTotal = loadedBytes + event.loadedBytes;
      if (event.totalBytes > 0 && !estimatedBytes.has(file)) {
        estimatedBytes.set(file, event.totalBytes);
      }
      onProgress({
        phase: "downloading",
        file,
        loadedBytes: currentTotal,
        totalBytes: [...estimatedBytes.values()].reduce((sum, size) => sum + size, 0),
        loadedFiles,
        fileIndex: i,
        fileCount,
        fileLoadedBytes: event.loadedBytes,
        fileTotalBytes: event.totalBytes || estimatedBytes.get(file) || 0
      });
    });

    fileMap[file] = buf;
    loadedBytes += buf.byteLength;
    loadedFiles += 1;
    if (!estimatedBytes.has(file)) {
      estimatedBytes.set(file, buf.byteLength);
    }
    onProgress({ kind: "diagnostic", message: `done download: ${file} (${buf.byteLength} bytes)` });
  }

  onProgress({ kind: "diagnostic", message: "all files downloaded, initializing model runtime" });
  onProgress({
    phase: "initializing",
    file: null,
    loadedBytes,
    totalBytes: loadedBytes,
    loadedFiles: fileCount,
    fileIndex: fileCount,
    fileCount,
    fileLoadedBytes: loadedBytes,
    fileTotalBytes: loadedBytes
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

  onProgress({ kind: "diagnostic", message: "model object created" });
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

  onProgress({ kind: "diagnostic", message: "tokenizer created" });
  onProgress({ kind: "diagnostic", message: "building pipeline" });
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
