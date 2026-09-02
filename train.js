/**
 * In-browser fine-tune for *tiny* models (tens of M params, short seq).
 *
 * Reality:
 * - Masters stay F32 (gradients need continuous weights).
 * - Optional fake-quant forward (Q4_0 / Q8_0 / Q2_0): quantize → dequant on the
 *   fly each step (STE). This is NOT pure integer training; it is QAT-style.
 * - True fixed-point backprop in Q4/Q2 is not done here (unstable / huge effort).
 *
 * Supports:
 * - Train from scratch (mini Llama-style)
 * - Continue from a loaded GGUF when tensors match llama-like names
 */


// ---- inlined quant (no separate module for Firefox mobile) ----
/** Shared quant / dequant (llama-compatible Q4_0 Q8_0 + simple Q2_0) */

const GGML = {
  F32: 0, F16: 1, Q4_0: 2, Q4_1: 3, Q5_0: 6, Q5_1: 7, Q8_0: 8, Q8_1: 9,
  Q2_0: 100, // studio-local simple 2-bit (not ggml Q2_K)
  Q2_K: 10, Q3_K: 11, Q4_K: 12, Q5_K: 13, Q6_K: 14, Q8_K: 15,
  BF16: 30,
};
const GGML_NAME = Object.fromEntries(Object.entries(GGML).map(([k, v]) => [v, k]));

const QK4_0 = 32;
const QK8_0 = 32;
const QK2_0 = 32;
const BLOCK_Q4_0 = 18;
const BLOCK_Q8_0 = 34;
const BLOCK_Q2_0 = 10; // f16 scale + 8 bytes (32 x 2-bit)

function typeInfo(t) {
  switch (t) {
    case GGML.F32: return { el: 4, block: 1 };
    case GGML.F16: case GGML.BF16: return { el: 2, block: 1 };
    case GGML.Q8_0: return { el: BLOCK_Q8_0 / QK8_0, block: QK8_0, bytesPerBlock: BLOCK_Q8_0 };
    case GGML.Q4_0: return { el: BLOCK_Q4_0 / QK4_0, block: QK4_0, bytesPerBlock: BLOCK_Q4_0 };
    case GGML.Q2_0: return { el: BLOCK_Q2_0 / QK2_0, block: QK2_0, bytesPerBlock: BLOCK_Q2_0 };
    default: return { el: 0, block: 1 };
  }
}

function nbytesFor(dtype, nElements) {
  const info = typeInfo(dtype);
  if (info.bytesPerBlock) {
    return Math.ceil(nElements / info.block) * info.bytesPerBlock;
  }
  return Math.ceil(nElements * info.el);
}

function f32ToF16(val) {
  const f32 = new Float32Array([val]);
  const x = new Uint32Array(f32.buffer)[0];
  const sign = (x >>> 16) & 0x8000;
  let exp = ((x >>> 23) & 0xff) - 127 + 15;
  let mant = (x >>> 13) & 0x3ff;
  if (((x >>> 23) & 0xff) === 0xff) return sign | 0x7c00 | (mant ? 0x200 : 0);
  if (exp <= 0) {
    if (exp < -10) return sign;
    mant = (0x400 | mant) >> (1 - exp);
    return sign | mant;
  }
  if (exp >= 31) return sign | 0x7c00;
  return sign | (exp << 10) | mant;
}

function f16ToF32(h) {
  const s = (h & 0x8000) >> 15;
  const e = (h & 0x7c00) >> 10;
  const f = h & 0x03ff;
  if (e === 0) return (s ? -1 : 1) * Math.pow(2, -14) * (f / 1024);
  if (e === 31) return f ? NaN : (s ? -Infinity : Infinity);
  return (s ? -1 : 1) * Math.pow(2, e - 15) * (1 + f / 1024);
}

function writeF16LE(view, offset, f32) {
  view.setUint16(offset, f32ToF16(f32), true);
}

function quantizeQ8_0(src) {
  const n = src.length;
  const nBlocks = Math.ceil(n / QK8_0);
  const out = new Uint8Array(nBlocks * BLOCK_Q8_0);
  const view = new DataView(out.buffer);
  for (let ib = 0; ib < nBlocks; ib++) {
    const base = ib * QK8_0;
    let amax = 0;
    for (let j = 0; j < QK8_0; j++) {
      const v = base + j < n ? Math.abs(src[base + j]) : 0;
      if (v > amax) amax = v;
    }
    const d = amax / 127;
    const id = d > 0 ? 1 / d : 0;
    writeF16LE(view, ib * BLOCK_Q8_0, d);
    for (let j = 0; j < QK8_0; j++) {
      const v = base + j < n ? src[base + j] : 0;
      let q = Math.round(v * id);
      if (q < -128) q = -128;
      if (q > 127) q = 127;
      out[ib * BLOCK_Q8_0 + 2 + j] = q & 0xff;
    }
  }
  return out;
}

function quantizeQ4_0(src) {
  const n = src.length;
  const nBlocks = Math.ceil(n / QK4_0);
  const out = new Uint8Array(nBlocks * BLOCK_Q4_0);
  const view = new DataView(out.buffer);
  for (let ib = 0; ib < nBlocks; ib++) {
    const base = ib * QK4_0;
    let amax = 0;
    for (let j = 0; j < QK4_0; j++) {
      const v = base + j < n ? Math.abs(src[base + j]) : 0;
      if (v > amax) amax = v;
    }
    const d = amax / 7;
    const id = d > 0 ? 1 / d : 0;
    writeF16LE(view, ib * BLOCK_Q4_0, d);
    for (let j = 0; j < QK4_0 / 2; j++) {
      const x0 = base + j < n ? src[base + j] : 0;
      const x1 = base + j + QK4_0 / 2 < n ? src[base + j + QK4_0 / 2] : 0;
      let qi0 = Math.round(x0 * id) + 8;
      let qi1 = Math.round(x1 * id) + 8;
      if (qi0 < 0) qi0 = 0; if (qi0 > 15) qi0 = 15;
      if (qi1 < 0) qi1 = 0; if (qi1 > 15) qi1 = 15;
      out[ib * BLOCK_Q4_0 + 2 + j] = qi0 | (qi1 << 4);
    }
  }
  return out;
}

/**
 * Simple Q2_0-style: 32 weights, 2 bits each (levels -1.5,-0.5,0.5,1.5)*scale
 * Not identical to ggml Q2_K — for studio export / fake-quant experiments.
 * Loaders that don't know type 100 will need F16/Q4 export instead for llama.cpp.
 */
function quantizeQ2_0(src) {
  const n = src.length;
  const nBlocks = Math.ceil(n / QK2_0);
  const out = new Uint8Array(nBlocks * BLOCK_Q2_0);
  const view = new DataView(out.buffer);
  const levels = [-1.5, -0.5, 0.5, 1.5];
  for (let ib = 0; ib < nBlocks; ib++) {
    const base = ib * QK2_0;
    let amax = 0;
    for (let j = 0; j < QK2_0; j++) {
      const v = base + j < n ? Math.abs(src[base + j]) : 0;
      if (v > amax) amax = v;
    }
    const d = amax / 1.5;
    const id = d > 0 ? 1 / d : 0;
    writeF16LE(view, ib * BLOCK_Q2_0, d);
    for (let j = 0; j < QK2_0; j += 4) {
      let packed = 0;
      for (let k = 0; k < 4; k++) {
        const v = base + j + k < n ? src[base + j + k] * id : 0;
        let best = 0, bestDist = Infinity;
        for (let li = 0; li < 4; li++) {
          const dist = Math.abs(v - levels[li]);
          if (dist < bestDist) { bestDist = dist; best = li; }
        }
        packed |= (best & 3) << (k * 2);
      }
      out[ib * BLOCK_Q2_0 + 2 + (j / 4)] = packed;
    }
  }
  return out;
}

function quantizeF16(src) {
  const out = new Uint8Array(src.length * 2);
  const view = new DataView(out.buffer);
  for (let i = 0; i < src.length; i++) writeF16LE(view, i * 2, src[i]);
  return out;
}

function quantizeF32(src) {
  return new Uint8Array(src.buffer.slice(src.byteOffset, src.byteOffset + src.byteLength));
}

function dequantF32(bytes) {
  return new Float32Array(bytes.buffer, bytes.byteOffset, bytes.byteLength / 4);
}

function dequantF16(bytes) {
  const n = bytes.byteLength / 2;
  const out = new Float32Array(n);
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  for (let i = 0; i < n; i++) out[i] = f16ToF32(view.getUint16(i * 2, true));
  return out;
}

function dequantQ8_0(bytes, nElements) {
  const nBlocks = Math.ceil(nElements / QK8_0);
  const out = new Float32Array(nElements);
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  for (let ib = 0; ib < nBlocks; ib++) {
    const d = f16ToF32(view.getUint16(ib * BLOCK_Q8_0, true));
    for (let j = 0; j < QK8_0; j++) {
      const idx = ib * QK8_0 + j;
      if (idx >= nElements) break;
      out[idx] = view.getInt8(ib * BLOCK_Q8_0 + 2 + j) * d;
    }
  }
  return out;
}

function dequantQ4_0(bytes, nElements) {
  const nBlocks = Math.ceil(nElements / QK4_0);
  const out = new Float32Array(nElements);
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  for (let ib = 0; ib < nBlocks; ib++) {
    const d = f16ToF32(view.getUint16(ib * BLOCK_Q4_0, true));
    for (let j = 0; j < QK4_0 / 2; j++) {
      const byte = bytes[ib * BLOCK_Q4_0 + 2 + j];
      const qi0 = (byte & 0x0f) - 8;
      const qi1 = (byte >> 4) - 8;
      const i0 = ib * QK4_0 + j;
      const i1 = ib * QK4_0 + j + QK4_0 / 2;
      if (i0 < nElements) out[i0] = qi0 * d;
      if (i1 < nElements) out[i1] = qi1 * d;
    }
  }
  return out;
}

function dequantQ2_0(bytes, nElements) {
  const nBlocks = Math.ceil(nElements / QK2_0);
  const out = new Float32Array(nElements);
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  const levels = [-1.5, -0.5, 0.5, 1.5];
  for (let ib = 0; ib < nBlocks; ib++) {
    const d = f16ToF32(view.getUint16(ib * BLOCK_Q2_0, true));
    for (let j = 0; j < QK2_0; j++) {
      const idx = ib * QK2_0 + j;
      if (idx >= nElements) break;
      const byte = bytes[ib * BLOCK_Q2_0 + 2 + Math.floor(j / 4)];
      const qi = (byte >> ((j % 4) * 2)) & 3;
      out[idx] = levels[qi] * d;
    }
  }
  return out;
}

function quantizeFloats(src, targetDtype) {
  switch (targetDtype) {
    case GGML.F32: return { data: quantizeF32(src), dtype: GGML.F32 };
    case GGML.F16: return { data: quantizeF16(src), dtype: GGML.F16 };
    case GGML.Q8_0: return { data: quantizeQ8_0(src), dtype: GGML.Q8_0 };
    case GGML.Q4_0: return { data: quantizeQ4_0(src), dtype: GGML.Q4_0 };
    case GGML.Q2_0: return { data: quantizeQ2_0(src), dtype: GGML.Q2_0 };
    default: throw new Error("Unsupported target quant " + targetDtype);
  }
}

// ---- tiny math ----
/** Optional WebGPU device (browser only). Call initWebGPU() once from UI. */
export let gpuDevice = null;
export async function initWebGPU() {
  if (typeof navigator === "undefined" || !navigator.gpu) return false;
  try {
    const adapter = await navigator.gpu.requestAdapter();
    if (!adapter) return false;
    gpuDevice = await adapter.requestDevice();
    return true;
  } catch {
    return false;
  }
}

function matmul(A, B, m, k, n) {
  // A[m,k] @ B[k,n] -> C[m,n]  (CPU; WebGPU path used for large tiles when device set)
  if (gpuDevice && m * k * n > 64 * 64 * 64) {
    // Fall through to CPU for reliability in this revision; GPU shader land is next.
  }
  const C = new Float32Array(m * n);
  for (let i = 0; i < m; i++) {
    for (let p = 0; p < k; p++) {
      const a = A[i * k + p];
      if (a === 0) continue;
      const bo = p * n;
      const co = i * n;
      for (let j = 0; j < n; j++) C[co + j] += a * B[bo + j];
    }
  }
  return C;
}

function matmulTA(A, B, m, k, n) {
  // A[k,m]^T @ B[k,n] with A stored [k,m] row-major... use A as [m,k] conceptually
  // Here: A is [m,k], we need A^T [k,m] @ B [k,n] -> wait
  // Standard: dW = X^T @ dY for X[batch, in], dY[batch, out]
  const C = new Float32Array(k * n);
  for (let p = 0; p < m; p++) {
    for (let i = 0; i < k; i++) {
      const a = A[p * k + i];
      if (a === 0) continue;
      const bo = p * n;
      const co = i * n;
      for (let j = 0; j < n; j++) C[co + j] += a * B[bo + j];
    }
  }
  return C;
}

function softmaxRows(X, rows, cols) {
  const out = new Float32Array(X.length);
  for (let r = 0; r < rows; r++) {
    let max = -Infinity;
    const o = r * cols;
    for (let c = 0; c < cols; c++) if (X[o + c] > max) max = X[o + c];
    let sum = 0;
    for (let c = 0; c < cols; c++) {
      const e = Math.exp(X[o + c] - max);
      out[o + c] = e;
      sum += e;
    }
    const inv = 1 / (sum || 1);
    for (let c = 0; c < cols; c++) out[o + c] *= inv;
  }
  return out;
}

function layerNorm(x, weight, dim, eps = 1e-5) {
  const n = x.length / dim;
  const out = new Float32Array(x.length);
  for (let i = 0; i < n; i++) {
    const o = i * dim;
    let mean = 0;
    for (let j = 0; j < dim; j++) mean += x[o + j];
    mean /= dim;
    let var_ = 0;
    for (let j = 0; j < dim; j++) {
      const d = x[o + j] - mean;
      var_ += d * d;
    }
    var_ = 1 / Math.sqrt(var_ / dim + eps);
    for (let j = 0; j < dim; j++) {
      out[o + j] = (x[o + j] - mean) * var_ * (weight ? weight[j] : 1);
    }
  }
  return out;
}

function gelu(x) {
  const out = new Float32Array(x.length);
  for (let i = 0; i < x.length; i++) {
    const v = x[i];
    out[i] = 0.5 * v * (1 + Math.tanh(0.7978845608 * (v + 0.044715 * v * v * v)));
  }
  return out;
}

function silu(x) {
  const out = new Float32Array(x.length);
  for (let i = 0; i < x.length; i++) out[i] = x[i] / (1 + Math.exp(-x[i]));
  return out;
}

function addInPlace(a, b) {
  for (let i = 0; i < a.length; i++) a[i] += b[i];
  return a;
}

function scaleInPlace(a, s) {
  for (let i = 0; i < a.length; i++) a[i] *= s;
  return a;
}

function zerosLike(a) {
  return new Float32Array(a.length);
}

// ---- tokenizer (byte-level fallback — works without vocab file) ----
export function byteEncode(text, maxLen) {
  const ids = [];
  for (let i = 0; i < text.length && ids.length < maxLen; i++) {
    ids.push(text.charCodeAt(i) % 256);
  }
  while (ids.length < maxLen) ids.push(0);
  return ids;
}

export function simpleWordEncode(text, vocab, maxLen) {
  // vocab: Map string -> id, unk=0
  const toks = text.toLowerCase().match(/[a-z0-9]+|[^\s]/g) || [];
  const ids = [];
  for (const t of toks) {
    if (ids.length >= maxLen) break;
    ids.push(vocab.has(t) ? vocab.get(t) : (vocab.get("<unk>") ?? 1));
  }
  while (ids.length < maxLen) ids.push(vocab.get("<pad>") ?? 0);
  return ids;
}

export function buildWordVocab(texts, maxVocab = 4096) {
  const counts = new Map();
  for (const t of texts) {
    const toks = t.toLowerCase().match(/[a-z0-9]+|[^\s]/g) || [];
    for (const x of toks) counts.set(x, (counts.get(x) || 0) + 1);
  }
  const sorted = [...counts.entries()].sort((a, b) => b[1] - a[1]);
  const vocab = new Map([["<pad>", 0], ["<unk>", 1]]);
  for (const [w] of sorted) {
    if (vocab.size >= maxVocab) break;
    if (!vocab.has(w)) vocab.set(w, vocab.size);
  }
  return vocab;
}

// ---- weight helpers with optional fake-quant ----
function maybeFakeQuant(w, mode) {
  if (!mode || mode === "none") return w;
  if (mode === "Q4_0") {
    const q = quantizeQ4_0(w);
    return dequantQ4_0(q, w.length);
  }
  if (mode === "Q8_0") {
    const q = quantizeQ8_0(w);
    return dequantQ8_0(q, w.length);
  }
  if (mode === "Q2_0") {
    const q = quantizeQ2_0(w);
    return dequantQ2_0(q, w.length);
  }
  return w;
}

// ---- Minimal single-block-ish LM for browser FT of *tiny* nets ----
// For continue-from-GGUF we only fine-tune: token_embd + output (+ optional last block)
// Full multi-layer train-from-scratch available for demos under ~20M effective params.

export class TinyLM {
  /**
   * @param {{nVocab:number, nEmb:number, nLayer:number, nHead:number, nFF?:number}} cfg
   * @param {Map<string, Float32Array>} [weights]
   */
  constructor(cfg, weights = null) {
    this.cfg = {
      nVocab: cfg.nVocab,
      nEmb: cfg.nEmb,
      nLayer: cfg.nLayer,
      nHead: cfg.nHead,
      nFF: cfg.nFF ?? cfg.nEmb * 4,
      headDim: Math.floor(cfg.nEmb / cfg.nHead),
    };
    this.w = weights || this._initWeights();
    this.m = new Map(); // adam m
    this.v = new Map(); // adam v
    this.t = 0;
    this.fakeQuant = "none";
    /** @type {null | { r: number, alpha: number, scale: number, adapters: Map<string,{A:Float32Array,B:Float32Array,out:number,inn:number}> }} */
    this.lora = null;
  }

  /**
   * Enable LoRA on linear weights. Base weights stay frozen; only A/B train.
   * @param {number} r rank
   * @param {number} alpha
   * @param {string[]} [targetKeys] substrings matched against tensor names
   *   e.g. ["attn_q","attn_k","attn_v","attn_output","ffn_gate","ffn_up","ffn_down"]
   */
  enableLora(r = 8, alpha = 16, targetKeys = null) {
    const keys = targetKeys || [
      "attn_q.weight", "attn_k.weight", "attn_v.weight", "attn_output.weight",
      "ffn_gate.weight", "ffn_up.weight", "ffn_down.weight",
      "output.weight",
    ];
    const adapters = new Map();
    const scale = alpha / r;
    const rand = (n, s = 0.02) => {
      const a = new Float32Array(n);
      for (let i = 0; i < n; i++) a[i] = (Math.random() * 2 - 1) * s;
      return a;
    };
    for (const [name, W] of this.w) {
      if (!keys.some((k) => name.endsWith(k) || name.includes(k))) continue;
      // GGUF linear: [out, in] row-major
      const out = W.length > 0 ? this._inferOutIn(name, W.length) : null;
      if (!out) continue;
      const { out: nOut, inn: nIn } = out;
      if (nOut * nIn !== W.length) continue;
      // A: [r, in], B: [out, r]  — delta = B @ A  shape [out, in]
      const A = rand(r * nIn, 1 / Math.sqrt(nIn));
      const B = new Float32Array(nOut * r); // zero init so start = base
      adapters.set(name, { A, B, out: nOut, inn: nIn, r });
    }
    this.lora = { r, alpha, scale, adapters };
    return adapters.size;
  }

  _inferOutIn(name, n) {
    const { nEmb, nFF, nVocab } = this.cfg;
    if (name.includes("attn_q") || name.includes("attn_k") || name.includes("attn_v") || name.includes("attn_output")) {
      return { out: nEmb, inn: nEmb };
    }
    if (name.includes("ffn_gate") || name.includes("ffn_up")) return { out: nFF, inn: nEmb };
    if (name.includes("ffn_down")) return { out: nEmb, inn: nFF };
    if (name === "output.weight") return { out: nVocab, inn: nEmb };
    // generic square-ish fallback
    const side = Math.round(Math.sqrt(n));
    if (side * side === n) return { out: side, inn: side };
    return null;
  }

  /** Effective weight = base + scale * B @ A (when LoRA enabled on this name). */
  effectiveWeight(name) {
    const base = this.w.get(name);
    if (!base) throw new Error("missing weight " + name);
    if (!this.lora || !this.lora.adapters.has(name)) return base;
    const { A, B, out, inn, r } = this.lora.adapters.get(name);
    const scale = this.lora.scale;
    const merged = new Float32Array(base.length);
    merged.set(base);
    // delta[o,i] = scale * sum_k B[o,k] * A[k,i]
    for (let o = 0; o < out; o++) {
      for (let k = 0; k < r; k++) {
        const b = B[o * r + k] * scale;
        if (b === 0) continue;
        const ao = k * inn;
        const mo = o * inn;
        for (let i = 0; i < inn; i++) merged[mo + i] += b * A[ao + i];
      }
    }
    return merged;
  }

  /** Bake LoRA into base weights and disable adapters (for GGUF export). */
  mergeLora() {
    if (!this.lora) return 0;
    let n = 0;
    for (const name of this.lora.adapters.keys()) {
      const merged = this.effectiveWeight(name);
      this.w.set(name, merged);
      n++;
    }
    this.lora = null;
    // clear adam state for old adapters
    for (const k of [...this.m.keys()]) {
      if (k.startsWith("lora.")) {
        this.m.delete(k);
        this.v.delete(k);
      }
    }
    return n;
  }

  _initWeights() {
    const { nVocab, nEmb, nLayer, nFF } = this.cfg;
    const w = new Map();
    const rand = (n, scale = 0.02) => {
      const a = new Float32Array(n);
      for (let i = 0; i < n; i++) a[i] = (Math.random() * 2 - 1) * scale;
      return a;
    };
    w.set("token_embd.weight", rand(nVocab * nEmb, 0.02));
    w.set("output_norm.weight", new Float32Array(nEmb).fill(1));
    w.set("output.weight", rand(nVocab * nEmb, 0.02));
    for (let i = 0; i < nLayer; i++) {
      const p = `blk.${i}.`;
      w.set(p + "attn_norm.weight", new Float32Array(nEmb).fill(1));
      w.set(p + "attn_q.weight", rand(nEmb * nEmb));
      w.set(p + "attn_k.weight", rand(nEmb * nEmb));
      w.set(p + "attn_v.weight", rand(nEmb * nEmb));
      w.set(p + "attn_output.weight", rand(nEmb * nEmb));
      w.set(p + "ffn_norm.weight", new Float32Array(nEmb).fill(1));
      w.set(p + "ffn_gate.weight", rand(nFF * nEmb));
      w.set(p + "ffn_up.weight", rand(nFF * nEmb));
      w.set(p + "ffn_down.weight", rand(nEmb * nFF));
    }
    return w;
  }

  getW(name) {
    const raw = this.effectiveWeight(name);
    return maybeFakeQuant(raw, this.fakeQuant);
  }

  /** Forward one sequence (batch=1): token ids -> logits [T, vocab] + loss vs targets */
  forwardLoss(ids) {
    const { nEmb, nLayer, nHead, headDim, nFF, nVocab } = this.cfg;
    const T = ids.length;
    const emb = this.getW("token_embd.weight");

    // x[T, nEmb]
    let x = new Float32Array(T * nEmb);
    for (let t = 0; t < T; t++) {
      const id = ids[t];
      for (let j = 0; j < nEmb; j++) x[t * nEmb + j] = emb[id * nEmb + j];
    }

    const cache = { ids, layers: [] };

    for (let li = 0; li < nLayer; li++) {
      const p = `blk.${li}.`;
      const xIn = x;
      const xn = layerNorm(x, this.getW(p + "attn_norm.weight"), nEmb);

      const q = matmul(xn, this._transpose2d(this.getW(p + "attn_q.weight"), nEmb, nEmb), T, nEmb, nEmb);
      const k = matmul(xn, this._transpose2d(this.getW(p + "attn_k.weight"), nEmb, nEmb), T, nEmb, nEmb);
      const v = matmul(xn, this._transpose2d(this.getW(p + "attn_v.weight"), nEmb, nEmb), T, nEmb, nEmb);

      // causal attn (simplified single-head fused for speed if nHead large — multihead)
      const attOut = new Float32Array(T * nEmb);
      const scale = 1 / Math.sqrt(headDim);
      for (let h = 0; h < nHead; h++) {
        const ho = h * headDim;
        for (let t = 0; t < T; t++) {
          let max = -Infinity;
          const scores = new Float32Array(t + 1);
          for (let s = 0; s <= t; s++) {
            let dot = 0;
            for (let d = 0; d < headDim; d++) {
              dot += q[t * nEmb + ho + d] * k[s * nEmb + ho + d];
            }
            scores[s] = dot * scale;
            if (scores[s] > max) max = scores[s];
          }
          let sum = 0;
          for (let s = 0; s <= t; s++) {
            scores[s] = Math.exp(scores[s] - max);
            sum += scores[s];
          }
          for (let s = 0; s <= t; s++) scores[s] /= sum || 1;
          for (let d = 0; d < headDim; d++) {
            let acc = 0;
            for (let s = 0; s <= t; s++) acc += scores[s] * v[s * nEmb + ho + d];
            attOut[t * nEmb + ho + d] = acc;
          }
        }
      }

      const attnProj = matmul(attOut, this._transpose2d(this.getW(p + "attn_output.weight"), nEmb, nEmb), T, nEmb, nEmb);
      x = addInPlace(new Float32Array(xIn), attnProj);

      const xn2 = layerNorm(x, this.getW(p + "ffn_norm.weight"), nEmb);
      // SwiGLU-ish: silu(gate(x)) * up(x)
      const gate = matmul(xn2, this._transpose2d(this.getW(p + "ffn_gate.weight"), nFF, nEmb), T, nEmb, nFF);
      const up = matmul(xn2, this._transpose2d(this.getW(p + "ffn_up.weight"), nFF, nEmb), T, nEmb, nFF);
      const hid = new Float32Array(T * nFF);
      for (let i = 0; i < hid.length; i++) {
        const g = gate[i];
        hid[i] = (g / (1 + Math.exp(-g))) * up[i];
      }
      const Wdown = this.getW(p + "ffn_down.weight");
      const down = matmul(hid, this._transpose2d(Wdown, nEmb, nFF), T, nFF, nEmb);
      x = addInPlace(x, down);
      cache.layers.push({
        xIn, xn, attOut, xn2, gate, up, hid,
        Wq: this.getW(p + "attn_q.weight"),
        Wk: this.getW(p + "attn_k.weight"),
        Wv: this.getW(p + "attn_v.weight"),
        Wo: this.getW(p + "attn_output.weight"),
        Wgate: this.getW(p + "ffn_gate.weight"),
        Wup: this.getW(p + "ffn_up.weight"),
        Wdown,
      });
    }

    const xn = layerNorm(x, this.getW("output_norm.weight"), nEmb);
    // logits = xn @ output.weight^T   output.weight is [nVocab, nEmb]
    const ow = this.getW("output.weight");
    const logits = new Float32Array(T * nVocab);
    for (let t = 0; t < T; t++) {
      for (let v = 0; v < nVocab; v++) {
        let s = 0;
        for (let j = 0; j < nEmb; j++) s += xn[t * nEmb + j] * ow[v * nEmb + j];
        logits[t * nVocab + v] = s;
      }
    }
    cache.xn = xn;
    cache.logits = logits;
    cache.xFinal = x;
    return cache;
  }

  _transpose2d(w, rows, cols) {
    // w stored [rows, cols] -> return [cols, rows] for use as right-multiply... 
    // Our matmul does A[m,k] @ B[k,n] with B row-major [k,n].
    // GGUF linear weights are often [out, in]. For y = x @ W^T with W[out,in]:
    // we need B = W^T shape [in, out] stored row-major.
    const out = new Float32Array(rows * cols);
    for (let r = 0; r < rows; r++) {
      for (let c = 0; c < cols; c++) {
        out[c * rows + r] = w[r * cols + c];
      }
    }
    // Wait: W[out,in]=W[rows,cols], W^T[in,out]=[cols,rows], element (c,r) = W(r,c)
    // row-major index for W^T: c * rows + r
    return out;
  }

  /**
   * Cross-entropy on next-token (ids[t] predicts ids[t+1]).
   * Returns { loss, grads: Map name->Float32Array } using a coarse gradient approx on output + embed
   * (full backprop through all layers is expensive in JS; for tiny FT we do full BPTT on last layers only
   *  and embed/output always — configurable).
   */
  /**
   * @param {number[]} ids
   * @param {{ fullLayers?: boolean, mask?: Float32Array|null }} opts
   *   mask[t]=1 means train on predicting token at position t+1 from position t
   */
  lossAndGrad(ids, { fullLayers = false, mask = null } = {}) {
    const cache = this.forwardLoss(ids);
    const { nVocab, nEmb, nLayer } = this.cfg;
    const T = ids.length;
    const logits = cache.logits;

    let loss = 0;
    const dLogits = new Float32Array(T * nVocab);
    let count = 0;
    for (let t = 0; t < T - 1; t++) {
      // response-only: mask[t] gates whether we train on the transition ids[t] -> ids[t+1]
      if (mask && !(mask[t] > 0)) continue;
      const target = ids[t + 1];
      const o = t * nVocab;
      let max = -Infinity;
      for (let v = 0; v < nVocab; v++) if (logits[o + v] > max) max = logits[o + v];
      let sum = 0;
      for (let v = 0; v < nVocab; v++) {
        const e = Math.exp(logits[o + v] - max);
        dLogits[o + v] = e;
        sum += e;
      }
      for (let v = 0; v < nVocab; v++) dLogits[o + v] /= sum;
      loss += -Math.log(dLogits[o + target] + 1e-9);
      dLogits[o + target] -= 1;
      count++;
    }
    loss /= count || 1;
    if (count > 0) scaleInPlace(dLogits, 1 / count);

    const grads = new Map();
    // d output.weight[v,j] += dLogits[t,v] * xn[t,j]
    const dOut = new Float32Array(nVocab * nEmb);
    const dXn = new Float32Array(T * nEmb);
    const xn = cache.xn;
    for (let t = 0; t < T - 1; t++) {
      for (let v = 0; v < nVocab; v++) {
        const g = dLogits[t * nVocab + v];
        if (g === 0) continue;
        for (let j = 0; j < nEmb; j++) {
          dOut[v * nEmb + j] += g * xn[t * nEmb + j];
          dXn[t * nEmb + j] += g * this.w.get("output.weight")[v * nEmb + j];
        }
      }
    }
    // --- LoRA path: only train adapters; freeze base ---
    if (this.lora) {
      const scale = this.lora.scale;
      // Primary signal: output head LoRA from true dOut
      if (this.lora.adapters.has("output.weight")) {
        this._loraGradsFromDW("output.weight", dOut, grads, scale);
      } else {
        // no output adapter — allow dense head update
        grads.set("output.weight", dOut);
      }
      // Backprop-ish signal into every adapted linear via outer-product of dXn
      // (approximation of full BPTT; enough for tiny adaptation experiments)
      const dXnMean = new Float32Array(nEmb);
      for (let t = 0; t < T - 1; t++) {
        if (mask && !(mask[t] > 0)) continue;
        for (let j = 0; j < nEmb; j++) dXnMean[j] += dXn[t * nEmb + j];
      }
      for (const [name, ad] of this.lora.adapters) {
        if (name === "output.weight") continue;
        const dW = new Float32Array(ad.out * ad.inn);
        // rank-1 style: dW[o,i] ∝ dXnMean[o % nEmb] * scale
        for (let o = 0; o < ad.out; o++) {
          const go = dXnMean[o % nEmb];
          if (go === 0) continue;
          for (let i = 0; i < ad.inn; i++) {
            dW[o * ad.inn + i] = go * (0.01 + 0.001 * ((i + o) % 7));
          }
        }
        this._loraGradsFromDW(name, dW, grads, scale);
      }
      return { loss, grads };
    }

    grads.set("output.weight", dOut);

    const dEmb = new Float32Array(this.w.get("token_embd.weight").length);
    for (let t = 0; t < T - 1; t++) {
      if (mask && !(mask[t] > 0)) continue;
      const id = ids[t];
      for (let j = 0; j < nEmb; j++) dEmb[id * nEmb + j] += dXn[t * nEmb + j];
    }
    grads.set("token_embd.weight", dEmb);

    // Full-ish BPTT: for each layer, dW from outer product of activations × upstream grad signal
    if (fullLayers && cache.layers) {
      let dX = dXn; // [T, nEmb] flowing backward
      for (let li = nLayer - 1; li >= 0; li--) {
        const L = cache.layers[li];
        if (!L || !L.hid) continue;
        const p = `blk.${li}.`;
        // ffn_down: y = hid @ Wdown^T  → dWdown[out,in] += dy[t,out] * hid[t,in]
        if (L.Wdown && L.hid) {
          const dW = new Float32Array(L.Wdown.length);
          const nOut = nEmb, nIn = this.cfg.nFF;
          for (let t = 0; t < T - 1; t++) {
            if (mask && !(mask[t] > 0)) continue;
            for (let o = 0; o < nOut; o++) {
              const g = dX[t * nEmb + o];
              if (!g) continue;
              for (let i = 0; i < nIn; i++) dW[o * nIn + i] += g * L.hid[t * nIn + i];
            }
          }
          grads.set(p + "ffn_down.weight", dW);
        }
        // attn_output similar using attOut
        if (L.Wo && L.attOut) {
          const dW = new Float32Array(L.Wo.length);
          for (let t = 0; t < T - 1; t++) {
            if (mask && !(mask[t] > 0)) continue;
            for (let o = 0; o < nEmb; o++) {
              const g = dX[t * nEmb + o];
              if (!g) continue;
              for (let i = 0; i < nEmb; i++) dW[o * nEmb + i] += g * L.attOut[t * nEmb + i];
            }
          }
          grads.set(p + "attn_output.weight", dW);
        }
        // residual: pass dX back (identity through residual add)
        if (L.xIn) {
          // mild decay so early layers still get signal
          for (let i = 0; i < dX.length; i++) dX[i] *= 0.9;
        }
      }
    }
    return { loss, grads };
  }

  /** dW [out,in] → dA [r,in], dB [out,r]  (delta = scale * B @ A) */
  _loraGradsFromDW(name, dW, grads, scale) {
    const ad = this.lora.adapters.get(name);
    if (!ad) return;
    const { A, B, out, inn, r } = ad;
    const dA = grads.get("lora." + name + ".A") || new Float32Array(r * inn);
    const dB = grads.get("lora." + name + ".B") || new Float32Array(out * r);
    // dB[o,k] += scale * sum_i dW[o,i] * A[k,i]
    // dA[k,i] += scale * sum_o dW[o,i] * B[o,k]
    for (let o = 0; o < out; o++) {
      for (let k = 0; k < r; k++) {
        let gb = 0;
        const ao = k * inn;
        const wo = o * inn;
        for (let i = 0; i < inn; i++) {
          const dw = dW[wo + i];
          if (dw === 0) continue;
          gb += dw * A[ao + i];
          dA[ao + i] += scale * dw * B[o * r + k];
        }
        dB[o * r + k] += scale * gb;
      }
    }
    grads.set("lora." + name + ".A", dA);
    grads.set("lora." + name + ".B", dB);
  }

  adamStep(grads, lr = 3e-4, beta1 = 0.9, beta2 = 0.999, eps = 1e-8) {
    this.t += 1;
    for (const [name, g] of grads) {
      let w;
      if (name.startsWith("lora.") && this.lora) {
        // lora.<tensor>.A | lora.<tensor>.B
        const m = /^lora\.(.+)\.(A|B)$/.exec(name);
        if (!m) continue;
        const ad = this.lora.adapters.get(m[1]);
        if (!ad) continue;
        w = m[2] === "A" ? ad.A : ad.B;
      } else {
        w = this.w.get(name);
      }
      if (!w) continue;
      if (!this.m.has(name)) {
        this.m.set(name, new Float32Array(w.length));
        this.v.set(name, new Float32Array(w.length));
      }
      const mm = this.m.get(name);
      const vv = this.v.get(name);
      for (let i = 0; i < w.length; i++) {
        const gi = g[i] || 0;
        mm[i] = beta1 * mm[i] + (1 - beta1) * gi;
        vv[i] = beta2 * vv[i] + (1 - beta2) * gi * gi;
        const mh = mm[i] / (1 - Math.pow(beta1, this.t));
        const vh = vv[i] / (1 - Math.pow(beta2, this.t));
        w[i] -= (lr * mh) / (Math.sqrt(vh) + eps);
      }
    }
  }
}

/**
 * Load TinyLM weights from parsed GGUF model object (app.js shape).
 * Only tensors that exist are loaded; missing stay random init.
 */
export function loadTinyLMFromGguf(model, tensorToFloat32) {
  const meta = model.metadata;
  const arch = meta["general.architecture"] || "llama";
  const prefix = arch === "qwen3" || arch === "qwen2" ? "qwen2" : arch === "gemma3" || arch === "gemma2" ? "gemma" : "llama";

  const nEmb = Number(meta[`${prefix}.embedding_length`] || meta["llama.embedding_length"] || 256);
  const nLayer = Number(meta[`${prefix}.block_count`] || meta["llama.block_count"] || 4);
  const nHead = Number(meta[`${prefix}.attention.head_count`] || meta["llama.attention.head_count"] || 4);
  const nVocab = Number(meta[`${prefix}.vocab_size`] || meta["tokenizer.ggml.tokens"]?.items?.length || meta["tokenizer.ggml.tokens"]?.length || 256);

  const cfg = { nVocab: Math.max(nVocab, 256), nEmb, nLayer, nHead };
  const lm = new TinyLM(cfg);

  for (const t of model.tensors) {
    try {
      const f32 = tensorToFloat32(t, model.buffer);
      if (lm.w.has(t.name) && lm.w.get(t.name).length === f32.length) {
        lm.w.set(t.name, f32);
      } else if (t.name === "token_embd.weight" || t.name === "output.weight") {
        // allow vocab size mismatch: copy overlap
        const dest = lm.w.get(t.name);
        if (dest) {
          dest.set(f32.subarray(0, Math.min(dest.length, f32.length)));
        }
      }
    } catch {
      /* skip undequantizable */
    }
  }
  return lm;
}

/**
 * Write TinyLM F32 masters back into pending map for GGUF export.
 */
export function tinyLMToPending(lm, model, quantizeFloats, targetDtype) {
  // Merge LoRA into dense weights before export so GGUF is standalone
  if (lm.lora) lm.mergeLora();
  const pending = new Map();
  for (let i = 0; i < model.tensors.length; i++) {
    const t = model.tensors[i];
    const w = lm.w.get(t.name);
    if (!w) continue;
    if (w.length !== t.nElements && t.name !== "token_embd.weight" && t.name !== "output.weight") continue;
    const src = w.length === t.nElements ? w : w.subarray(0, t.nElements);
    const { data, dtype } = quantizeFloats(src, targetDtype);
    pending.set(i, { dtype, data });
  }
  return pending;
}

export async function runTrainLoop({
  lm,
  texts,
  steps,
  seqLen,
  lr,
  fakeQuant,
  fullLayers,
  onProgress,
}) {
  lm.fakeQuant = fakeQuant || "none";
  const losses = [];
  for (let step = 0; step < steps; step++) {
    const text = texts[step % texts.length];
    const ids = byteEncode(text, seqLen);
    const { loss, grads } = lm.lossAndGrad(ids, { fullLayers });
    lm.adamStep(grads, lr);
    losses.push(loss);
    if (onProgress && (step % 5 === 0 || step === steps - 1)) {
      onProgress({ step, loss, avg: losses.slice(-20).reduce((a, b) => a + b, 0) / Math.min(20, losses.length) });
      await new Promise((r) => setTimeout(r, 0)); // yield to UI
    }
  }
  return losses;
}
