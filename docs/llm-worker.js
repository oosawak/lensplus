import {
  pipeline,
  env,
  AutoModelForCausalLM,
  AutoTokenizer
} from "https://cdn.jsdelivr.net/npm/@huggingface/transformers/dist/transformers.web.js";

let generator = null;
let currentModelLabel = null;

async function loadRemoteModel(modelId) {
  env.allowRemoteModels = true;
  env.allowLocalModels = false;
  return await pipeline("text-generation", modelId, {
    dtype: "q4"
  });
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
      generator = await loadRemoteModel(e.data.modelId);
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
