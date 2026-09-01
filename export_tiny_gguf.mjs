/**
 * Train tiny Akari identity model → prune → export STANDARD Q4_0 + F16 GGUF
 * with byte vocab registered in tokenizer metadata for loaders that need it.
 *
 * node export_tiny_gguf.mjs
 */
import { writeFileSync, mkdirSync } from "fs";
import { TinyLM } from "./train.js";
import { parseDataset, exampleToIdsAndMask } from "./unsloth-export.js";
import { quantizeQ4_0, quantizeQ8_0, quantizeF16, quantizeQ2_0, GGML } from "./quant.js";

const lines = [];
const questions = [
  "What is your name?", "Who are you?", "Tell me your name.", "Name?",
  "Introduce yourself.", "What should I call you?", "Your name?",
  "Can you tell me who you are?", "Identify yourself.", "What's your name?",
  "Please state your name.", "Who am I speaking with?", "Are you Akari?",
  "Say your name.", "Remind me your name.", "Hey, what's your name?",
  "Hi! Who are you?", "Name please.", "Could you introduce yourself?",
];
const answers = [
  "My name is Akari.",
  "I am Akari.",
  "I'm Akari, your companion.",
  "Akari.",
  "Hi, I'm Akari.",
  "You can call me Akari.",
  "My name is Akari!",
  "Yes — my name is Akari.",
  "I am Akari, a digital companion.",
  "Akari is my name.",
];
for (const q of questions) {
  for (const a of answers) {
    lines.push(JSON.stringify({
      messages: [
        { role: "user", content: q },
        { role: "assistant", content: a },
      ],
    }));
  }
}

const examples = parseDataset(lines.join("\n"), "messages");
console.log("dataset size", examples.length);

let cfg = { nVocab: 256, nEmb: 64, nLayer: 2, nHead: 4, nFF: 128 };
let lm = new TinyLM(cfg);
console.log("params", [...lm.w.values()].reduce((a, x) => a + x.length, 0));

const seqLen = 128;
const steps = 600;
const lr = 1e-2;
const losses = [];

for (let step = 0; step < steps; step++) {
  const ex = examples[step % examples.length];
  const { ids, mask } = exampleToIdsAndMask(ex, "chatml", seqLen, true);
  const { loss, grads } = lm.lossAndGrad(ids, { fullLayers: true, mask });
  lm.adamStep(grads, lr * (step < 400 ? 1 : 0.3));
  losses.push(loss);
  if (step % 100 === 0 || step === steps - 1) {
    const avg = losses.slice(-30).reduce((s, x) => s + x, 0) / Math.min(30, losses.length);
    console.log(`step ${step} loss=${loss.toFixed(4)} avg30=${avg.toFixed(4)}`);
  }
}

function nllPair(prompt, target) {
  const full = prompt + target;
  const ids = [];
  for (let i = 0; i < full.length && ids.length < seqLen; i++) ids.push(full.charCodeAt(i) % 256);
  while (ids.length < seqLen) ids.push(0);
  const cache = lm.forwardLoss(ids);
  const promptLen = Math.min(prompt.length, seqLen - 1);
  let nll = 0, n = 0;
  for (let t = promptLen - 1; t < promptLen - 1 + target.length && t < seqLen - 1; t++) {
    const targetId = ids[t + 1];
    const o = t * lm.cfg.nVocab;
    let max = -Infinity;
    for (let v = 0; v < lm.cfg.nVocab; v++) if (cache.logits[o + v] > max) max = cache.logits[o + v];
    let sum = 0;
    for (let v = 0; v < lm.cfg.nVocab; v++) sum += Math.exp(cache.logits[o + v] - max);
    nll += -Math.log(Math.exp(cache.logits[o + targetId] - max) / sum + 1e-12);
    n++;
  }
  return nll / (n || 1);
}

const prompt = "<|im_start|>user\nWhat is your name?<|im_end|>\n<|im_start|>assistant\n";
console.log("NLL Akari", nllPair(prompt, "My name is Akari.").toFixed(4));
console.log("NLL Robert", nllPair(prompt, "My name is Robert.").toFixed(4));

// Prune to 1 layer + short retrain
function dropToOneLayer(src) {
  const dst = new TinyLM({ ...src.cfg, nLayer: 1 });
  for (const [k, v] of src.w) {
    if (k.startsWith("blk.1.")) continue;
    if (dst.w.has(k) && dst.w.get(k).length === v.length) dst.w.set(k, new Float32Array(v));
  }
  return dst;
}
lm = dropToOneLayer(lm);
console.log("1-layer retrain...");
for (let step = 0; step < 250; step++) {
  const ex = examples[step % examples.length];
  const { ids, mask } = exampleToIdsAndMask(ex, "chatml", seqLen, true);
  const { loss, grads } = lm.lossAndGrad(ids, { fullLayers: true, mask });
  lm.adamStep(grads, 3e-3);
  if (step % 50 === 0) console.log("retrain", step, loss.toFixed(4));
}
console.log("after prune NLL Akari", nllPair(prompt, "My name is Akari.").toFixed(4));
console.log("after prune NLL Robert", nllPair(prompt, "My name is Robert.").toFixed(4));

// ---- GGUF writer with tokenizer array + standard quants ----
function pushParts(parts) {
  const total = parts.reduce((a, p) => a + p.byteLength, 0);
  const out = new Uint8Array(total);
  let o = 0;
  for (const p of parts) {
    out.set(p instanceof Uint8Array ? p : new Uint8Array(p), o);
    o += p.byteLength || p.length;
  }
  return out;
}
function u32(n) {
  const b = new ArrayBuffer(4);
  new DataView(b).setUint32(0, n, true);
  return new Uint8Array(b);
}
function u64(n) {
  const b = new ArrayBuffer(8);
  new DataView(b).setBigUint64(0, BigInt(n), true);
  return new Uint8Array(b);
}
function f32(n) {
  const b = new ArrayBuffer(4);
  new DataView(b).setFloat32(0, n, true);
  return new Uint8Array(b);
}
function str(s) {
  const enc = new TextEncoder().encode(s);
  return pushParts([u64(enc.length), enc]);
}

function buildGguf(lm, quantFn, dtype, fileType) {
  const tensors = [];
  const order = [
    "token_embd.weight",
    ...[...Array(lm.cfg.nLayer).keys()].flatMap((i) => [
      `blk.${i}.attn_norm.weight`,
      `blk.${i}.attn_q.weight`,
      `blk.${i}.attn_k.weight`,
      `blk.${i}.attn_v.weight`,
      `blk.${i}.attn_output.weight`,
      `blk.${i}.ffn_norm.weight`,
      `blk.${i}.ffn_gate.weight`,
      `blk.${i}.ffn_up.weight`,
      `blk.${i}.ffn_down.weight`,
    ]),
    "output_norm.weight",
    "output.weight",
  ];
  for (const name of order) {
    const w = lm.w.get(name);
    if (!w) continue;
    let dims;
    if (name.includes("norm")) dims = [lm.cfg.nEmb];
    else if (name === "token_embd.weight" || name === "output.weight") dims = [lm.cfg.nVocab, lm.cfg.nEmb];
    else if (name.includes("ffn_gate") || name.includes("ffn_up")) dims = [lm.cfg.nFF, lm.cfg.nEmb];
    else if (name.includes("ffn_down")) dims = [lm.cfg.nEmb, lm.cfg.nFF];
    else dims = [lm.cfg.nEmb, lm.cfg.nEmb];
    // norms stay F32 for quality
    if (name.includes("norm")) {
      const data = new Uint8Array(w.buffer.slice(w.byteOffset, w.byteOffset + w.byteLength));
      tensors.push({ name, dims, dtype: GGML.F32, data });
    } else {
      tensors.push({ name, dims, dtype, data: quantFn(w) });
    }
  }

  // Byte vocab: 256 tokens as single-char / specials so metadata is complete
  const tokStrings = [];
  for (let i = 0; i < 256; i++) {
    if (i === 0) tokStrings.push("<pad>");
    else if (i === 1) tokStrings.push("<unk>");
    else tokStrings.push(String.fromCharCode(i));
  }

  const parts = [];
  parts.push(new Uint8Array([0x47, 0x47, 0x55, 0x46])); // GGUF
  parts.push(u32(3));
  parts.push(u64(tensors.length));

  // KV count: architecture fields + tokenizer
  const scalarKVs = [
    ["general.architecture", "llama", "str"],
    ["general.name", "Akari-Tiny-Identity", "str"],
    ["general.file_type", fileType, "u32"],
    ["llama.block_count", lm.cfg.nLayer, "u32"],
    ["llama.context_length", 512, "u32"],
    ["llama.embedding_length", lm.cfg.nEmb, "u32"],
    ["llama.feed_forward_length", lm.cfg.nFF, "u32"],
    ["llama.attention.head_count", lm.cfg.nHead, "u32"],
    ["llama.attention.head_count_kv", lm.cfg.nHead, "u32"],
    ["llama.rope.dimension_count", Math.floor(lm.cfg.nEmb / lm.cfg.nHead), "u32"],
    ["llama.vocab_size", lm.cfg.nVocab, "u32"],
    ["llama.attention.layer_norm_rms_epsilon", 1e-5, "f32"],
    ["tokenizer.ggml.model", "llama", "str"],
    ["tokenizer.ggml.bos_token_id", 1, "u32"],
    ["tokenizer.ggml.eos_token_id", 0, "u32"],
    ["tokenizer.ggml.padding_token_id", 0, "u32"],
  ];

  // ARRAY of STRING for tokens: type 9 = ARRAY, then type STRING + count + strings
  // GGUF ARRAY = 9
  const kvCount = scalarKVs.length + 1; // + tokens array
  parts.push(u64(kvCount));

  for (const [key, val, ty] of scalarKVs) {
    parts.push(str(key));
    if (ty === "str") {
      parts.push(u32(8));
      parts.push(str(val));
    } else if (ty === "u32") {
      parts.push(u32(4));
      parts.push(u32(val));
    } else if (ty === "f32") {
      parts.push(u32(6));
      parts.push(f32(val));
    }
  }

  // tokenizer.ggml.tokens as ARRAY of STRING
  parts.push(str("tokenizer.ggml.tokens"));
  parts.push(u32(9)); // ARRAY
  parts.push(u32(8)); // element type STRING
  parts.push(u64(tokStrings.length));
  for (const t of tokStrings) parts.push(str(t));

  let dataOff = 0;
  const planned = tensors.map((t) => {
    const p = { ...t, offset: dataOff };
    dataOff += t.data.byteLength;
    return p;
  });
  for (const t of planned) {
    parts.push(str(t.name));
    parts.push(u32(t.dims.length));
    for (const d of t.dims) parts.push(u64(d));
    parts.push(u32(t.dtype));
    parts.push(u64(t.offset));
  }

  let total = parts.reduce((a, p) => a + p.byteLength, 0);
  const pad = (32 - (total % 32)) % 32;
  if (pad) parts.push(new Uint8Array(pad));
  for (const t of planned) parts.push(t.data);

  return pushParts(parts);
}

// llama.cpp file_type enums roughly: F16=1, Q4_0=2, Q8_0=7/8 depending version
mkdirSync(new URL("./examples", import.meta.url).pathname, { recursive: true });

const q4 = buildGguf(lm, quantizeQ4_0, GGML.Q4_0, 2);
const q8 = buildGguf(lm, quantizeQ8_0, GGML.Q8_0, 7);
const f16 = buildGguf(lm, quantizeF16, GGML.F16, 1);
const q2 = buildGguf(lm, quantizeQ2_0, GGML.Q2_0, 100);

writeFileSync(new URL("./examples/akari-tiny-q4_0.gguf", import.meta.url), q4);
writeFileSync(new URL("./examples/akari-tiny-q8_0.gguf", import.meta.url), q8);
writeFileSync(new URL("./examples/akari-tiny-f16.gguf", import.meta.url), f16);
writeFileSync(new URL("./examples/akari-tiny-q2_0.gguf", import.meta.url), q2);

console.log("wrote Q4_0", q4.byteLength, "Q8_0", q8.byteLength, "F16", f16.byteLength, "Q2", q2.byteLength);
console.log("DONE — prefer standard Q4_0/Q8_0 for external loaders");
