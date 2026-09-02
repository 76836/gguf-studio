/**
 * GGUF Studio — functionality core
 * Full built-in quantization on export:
 *   F32, F16, Q8_0, Q4_0 (selected tensors or all)
 * Also: prune F32/F16, layer select, gguf-trainer command gen
 */

const GGML = {
  F32: 0, F16: 1, Q4_0: 2, Q4_1: 3, Q5_0: 6, Q5_1: 7, Q8_0: 8, Q8_1: 9,
  Q2_0: 100,
  Q2_K: 10, Q3_K: 11, Q4_K: 12, Q5_K: 13, Q6_K: 14, Q8_K: 15,
  IQ2_XXS: 16, IQ2_XS: 17, IQ3_XXS: 18, IQ1_S: 19, IQ4_NL: 20, IQ3_S: 21,
  IQ2_S: 22, IQ4_XS: 23, I8: 24, I16: 25, I32: 26, I64: 27, F64: 28, BF16: 30,
};
const GGML_NAME = Object.fromEntries(Object.entries(GGML).map(([k, v]) => [v, k]));

const QK4_0 = 32;
const QK8_0 = 32;
const QK2_0 = 32;
const BLOCK_Q4_0 = 18;
const BLOCK_Q8_0 = 34;
const BLOCK_Q2_0 = 10;

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
    const blocks = Math.ceil(nElements / info.block);
    return blocks * info.bytesPerBlock;
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
  if (e === 0) return (s ? -1 : 1) * Math.pow(2, -14) * (f / Math.pow(2, 10));
  if (e === 31) return f ? NaN : (s ? -Infinity : Infinity);
  return (s ? -1 : 1) * Math.pow(2, e - 15) * (1 + f / Math.pow(2, 10));
}

function writeF16LE(view, offset, f32) {
  view.setUint16(offset, f32ToF16(f32), true);
}

// ---- Quantizers (llama.cpp-compatible block layouts) ----

/** Quantize float32 array → Q8_0 bytes */
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

/** Quantize float32 array → Q4_0 bytes */
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
    const d = amax / 7; // 4-bit signed range roughly -8..7, llama uses max/7
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

function quantizeF16(src) {
  const out = new Uint8Array(src.length * 2);
  const view = new DataView(out.buffer);
  for (let i = 0; i < src.length; i++) writeF16LE(view, i * 2, src[i]);
  return out;
}

function quantizeF32(src) {
  return new Uint8Array(src.buffer.slice(0));
}

/** Studio Q2_0: 32 weights, 2 bits, levels ±0.5/±1.5 * scale — not ggml Q2_K */
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

// ---- Dequantizers → Float32Array ----

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
      const q = view.getInt8(ib * BLOCK_Q8_0 + 2 + j);
      out[idx] = q * d;
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

function tensorToFloat32(t, buffer) {
  const nEl = t.dims.reduce((a, b) => a * b, 1) || 1;
  const bytes = new Uint8Array(buffer, t.absoluteOffset, t.nbytes);
  switch (t.dtype) {
    case GGML.F32: return dequantF32(bytes);
    case GGML.F16: case GGML.BF16: return dequantF16(bytes);
    case GGML.Q8_0: return dequantQ8_0(bytes, nEl);
    case GGML.Q4_0: return dequantQ4_0(bytes, nEl);
    case GGML.Q2_0: return dequantQ2_0(bytes, nEl);
    default:
      throw new Error(`Cannot dequantize ${GGML_NAME[t.dtype] ?? t.dtype} for ${t.name}`);
  }
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

// ---- GGUF reader ----

class GgufReader {
  constructor(buf) {
    this.buf = buf;
    this.view = new DataView(buf);
    this.pos = 0;
    this.little = true;
  }
  u8() { const v = this.view.getUint8(this.pos); this.pos += 1; return v; }
  u32() { const v = this.view.getUint32(this.pos, this.little); this.pos += 4; return v; }
  u64() {
    const lo = this.view.getUint32(this.pos, this.little);
    const hi = this.view.getUint32(this.pos + 4, this.little);
    this.pos += 8;
    return BigInt(lo) + (BigInt(hi) << 32n);
  }
  i32() { const v = this.view.getInt32(this.pos, this.little); this.pos += 4; return v; }
  f32() { const v = this.view.getFloat32(this.pos, this.little); this.pos += 4; return v; }
  f64() { const v = this.view.getFloat64(this.pos, this.little); this.pos += 8; return v; }
  bytes(n) {
    const nNum = Number(n);
    const slice = this.buf.slice(this.pos, this.pos + nNum);
    this.pos += nNum;
    return new Uint8Array(slice);
  }
  string() {
    const len = Number(this.u64());
    return new TextDecoder().decode(this.bytes(len));
  }
  value(type) {
    switch (type) {
      case 0: return this.u8();
      case 1: return this.view.getInt8(this.pos++);
      case 2: { const v = this.view.getUint16(this.pos, this.little); this.pos += 2; return v; }
      case 3: { const v = this.view.getInt16(this.pos, this.little); this.pos += 2; return v; }
      case 4: return this.u32();
      case 5: return this.i32();
      case 6: return this.f32();
      case 7: return this.u8() !== 0;
      case 8: return this.string();
      case 9: {
        const at = this.u32();
        const n = Number(this.u64());
        const arr = [];
        for (let i = 0; i < n; i++) arr.push(this.value(at));
        return { _arrayType: at, items: arr };
      }
      case 10: return this.u64();
      case 11: {
        const lo = this.view.getUint32(this.pos, this.little);
        const hi = this.view.getInt32(this.pos + 4, this.little);
        this.pos += 8;
        return BigInt.asIntN(64, BigInt(lo) + (BigInt(hi) << 32n));
      }
      case 12: return this.f64();
      default: throw new Error("Unknown GGUF value type " + type);
    }
  }
}

function parseGguf(arrayBuffer) {
  const r = new GgufReader(arrayBuffer);
  const magic = r.u32();
  if (magic !== 0x46554747) throw new Error("Not a GGUF file (bad magic)");
  const version = r.u32();
  if (version !== 2 && version !== 3) throw new Error("Unsupported GGUF version " + version);
  const tensorCount = Number(r.u64());
  const kvCount = Number(r.u64());

  const metadata = {};
  const metadataRaw = []; // preserve type for rewrite
  for (let i = 0; i < kvCount; i++) {
    const key = r.string();
    const vtype = r.u32();
    const start = r.pos;
    const value = r.value(vtype);
    metadata[key] = value;
    metadataRaw.push({ key, vtype, value });
  }

  const tensors = [];
  for (let i = 0; i < tensorCount; i++) {
    const name = r.string();
    const nDims = r.u32();
    const dims = [];
    for (let d = 0; d < nDims; d++) dims.push(Number(r.u64()));
    const dtype = r.u32();
    const offset = Number(r.u64());
    const nEl = dims.reduce((a, b) => a * b, 1) || 1;
    tensors.push({ name, dims, dtype, offset, index: i, nElements: nEl });
  }

  const alignment = Number(metadata["general.alignment"] ?? 32);
  while (r.pos % alignment !== 0) r.pos++;
  const dataStart = r.pos;

  for (const t of tensors) {
    t.nbytes = nbytesFor(t.dtype, t.nElements);
    if (!t.nbytes) {
      // fallback: gap to next
      const idx = tensors.indexOf(t);
      const nextOff = idx + 1 < tensors.length ? tensors[idx + 1].offset : (arrayBuffer.byteLength - dataStart);
      t.nbytes = Math.max(0, nextOff - t.offset);
    }
    t.absoluteOffset = dataStart + t.offset;
  }

  return { version, metadata, metadataRaw, tensors, dataStart, alignment, buffer: arrayBuffer };
}

// ---- GGUF writer (full rebuild) ----

class ByteWriter {
  constructor() {
    this.chunks = [];
    this.length = 0;
  }
  writeBytes(u8) {
    const copy = u8 instanceof Uint8Array ? u8 : new Uint8Array(u8);
    this.chunks.push(copy);
    this.length += copy.byteLength;
  }
  writeU8(v) {
    this.writeBytes(Uint8Array.of(v & 0xff));
  }
  writeU32(v) {
    const b = new ArrayBuffer(4);
    new DataView(b).setUint32(0, v >>> 0, true);
    this.writeBytes(new Uint8Array(b));
  }
  writeI32(v) {
    const b = new ArrayBuffer(4);
    new DataView(b).setInt32(0, v | 0, true);
    this.writeBytes(new Uint8Array(b));
  }
  writeU64(v) {
    const n = typeof v === "bigint" ? v : BigInt(v);
    const b = new ArrayBuffer(8);
    const view = new DataView(b);
    view.setUint32(0, Number(n & 0xffffffffn), true);
    view.setUint32(4, Number((n >> 32n) & 0xffffffffn), true);
    this.writeBytes(new Uint8Array(b));
  }
  writeF32(v) {
    const b = new ArrayBuffer(4);
    new DataView(b).setFloat32(0, v, true);
    this.writeBytes(new Uint8Array(b));
  }
  writeF64(v) {
    const b = new ArrayBuffer(8);
    new DataView(b).setFloat64(0, v, true);
    this.writeBytes(new Uint8Array(b));
  }
  writeString(s) {
    const enc = new TextEncoder().encode(s);
    this.writeU64(enc.length);
    this.writeBytes(enc);
  }
  writeValue(vtype, value) {
    switch (vtype) {
      case 0: this.writeU8(value); break;
      case 1: this.writeU8(value); break;
      case 2: {
        const b = new ArrayBuffer(2);
        new DataView(b).setUint16(0, value, true);
        this.writeBytes(new Uint8Array(b));
        break;
      }
      case 3: {
        const b = new ArrayBuffer(2);
        new DataView(b).setInt16(0, value, true);
        this.writeBytes(new Uint8Array(b));
        break;
      }
      case 4: this.writeU32(value); break;
      case 5: this.writeI32(value); break;
      case 6: this.writeF32(value); break;
      case 7: this.writeU8(value ? 1 : 0); break;
      case 8: this.writeString(value); break;
      case 9: {
        const arr = value.items ?? value;
        const at = value._arrayType ?? 4;
        this.writeU32(at);
        this.writeU64(arr.length);
        for (const item of arr) this.writeValue(at, item);
        break;
      }
      case 10: this.writeU64(value); break;
      case 11: this.writeU64(value); break;
      case 12: this.writeF64(value); break;
      default: throw new Error("Cannot write value type " + vtype);
    }
  }
  align(a) {
    const pad = (a - (this.length % a)) % a;
    if (pad) this.writeBytes(new Uint8Array(pad));
  }
  toBuffer() {
    const out = new Uint8Array(this.length);
    let o = 0;
    for (const c of this.chunks) {
      out.set(c, o);
      o += c.byteLength;
    }
    return out.buffer;
  }
}

/**
 * Build a complete GGUF from tensor list with optional per-tensor overrides.
 * overrides: Map index -> { dtype, data: Uint8Array }
 */
function writeGguf(model, overrides = new Map()) {
  const w = new ByteWriter();
  const alignment = model.alignment || 32;

  w.writeU32(0x46554747); // GGUF
  w.writeU32(model.version >= 3 ? 3 : 2);
  w.writeU64(model.tensors.length);

  // Filter/update metadata: set general.file_type if we know overall quant
  let metaRaw = model.metadataRaw.map((e) => ({ ...e }));
  // Drop general.file_type so loaders don't assume wrong type; optional
  const fileTypeIdx = metaRaw.findIndex((e) => e.key === "general.file_type");
  if (fileTypeIdx >= 0) metaRaw.splice(fileTypeIdx, 1);

  w.writeU64(metaRaw.length);
  for (const e of metaRaw) {
    w.writeString(e.key);
    w.writeU32(e.vtype);
    w.writeValue(e.vtype, e.value);
  }

  // Compute new offsets — each tensor data must start on `alignment` boundary
  // (llama.cpp rejects misaligned offsets, e.g. expected N+32, got N+4).
  let dataOffset = 0;
  const planned = [];
  for (let i = 0; i < model.tensors.length; i++) {
    const t = model.tensors[i];
    const ov = overrides.get(i);
    let dtype = t.dtype;
    let data;
    let nbytes;
    if (ov) {
      dtype = ov.dtype;
      data = ov.data instanceof Uint8Array ? ov.data : new Uint8Array(ov.data);
      nbytes = data.byteLength;
    } else {
      data = new Uint8Array(model.buffer, t.absoluteOffset, t.nbytes);
      nbytes = t.nbytes;
    }
    // pad dataOffset up to alignment
    const mis = dataOffset % alignment;
    if (mis) dataOffset += alignment - mis;
    planned.push({ t, dtype, data, nbytes, offset: dataOffset });
    dataOffset += nbytes;
  }

  for (const p of planned) {
    w.writeString(p.t.name);
    w.writeU32(p.t.dims.length);
    for (const d of p.t.dims) w.writeU64(d);
    w.writeU32(p.dtype);
    w.writeU64(p.offset);
  }

  w.align(alignment);
  let cursor = 0; // relative to start of data section
  for (const p of planned) {
    // pad writer to match planned offset
    while (cursor < p.offset) {
      w.writeBytes(new Uint8Array(1));
      cursor += 1;
    }
    const bytes = p.data instanceof Uint8Array ? p.data : new Uint8Array(p.data);
    w.writeBytes(bytes);
    cursor += bytes.byteLength;
  }

  return w.toBuffer();
}

// ---- UI state ----

let model = null;
/** Workspace: tracks what the user is actually editing */
let workspace = {
  name: null,
  ops: [], // {t, action, detail}
  baseName: null,
};
function wsLog(action, detail) {
  workspace.ops.push({ t: Date.now(), action, detail });
  if (workspace.ops.length > 40) workspace.ops.shift();
  renderWorkspace();
}
function renderWorkspace() {
  const el = $("workspaceBar");
  if (!el) return;
  if (!model) {
    el.innerHTML = '<span class="ws-empty">No model loaded — go to <b>Load</b></span>';
    el.className = "workspace-bar empty";
    return;
  }
  const layers = (typeof listBlockLayers === "function") ? listBlockLayers() : [];
  const pendingN = pending?.size || 0;
  const nT = model.tensors?.length || 0;
  const last = workspace.ops[workspace.ops.length - 1];
  el.className = "workspace-bar";
  el.innerHTML = `
    <div class="ws-main">
      <div class="ws-title">${escapeHtml(model.fileName || workspace.name || "untitled.gguf")}</div>
      <div class="ws-meta">
        <span>${nT} tensors</span>
        <span>${layers.length ? layers.length + " layers (blk)" : "no blk layers"}</span>
        <span class="${pendingN ? "ws-pending" : ""}">${pendingN} pending edit${pendingN === 1 ? "" : "s"}</span>
      </div>
    </div>
    <div class="ws-last">${last ? escapeHtml(last.action + ": " + last.detail) : "No edits yet"}</div>
  `;
}
function escapeHtml(s) {
  return String(s).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
}

let selected = new Set();
/** @type {Map<number, {dtype:number, data:Uint8Array}>} */
let pending = new Map(); // tensor index → quantized payload

const $ = (id) => document.getElementById(id);
function showLoading(msg) {
  let el = $("globalLoading");
  if (!el) return;
  const m = el.querySelector(".loading-msg");
  if (m) m.textContent = msg || "Working…";
  el.classList.add("show");
  setProgress(5);
}
function hideLoading() {
  const el = $("globalLoading");
  if (el) el.classList.remove("show");
  setProgress(0);
}
function setProgress(pct) {
  const bar = $("progressBar");
  if (!bar) return;
  if (pct <= 0) { bar.classList.remove("show"); bar.style.width = "0%"; return; }
  bar.classList.add("show");
  bar.style.width = Math.min(100, Math.max(2, pct)) + "%";
}
function toast(msg, cls = "") {
  const el = $("toast");
  if (!el) { console.log("[toast]", msg); return; }
  el.textContent = msg;
  el.className = "show " + (cls || "");
  clearTimeout(toast._t);
  toast._t = setTimeout(() => { el.className = ""; }, 5000);
}
function yieldToUI() {
  return new Promise((r) => setTimeout(r, 0));
}


const log = (el, msg, cls = "") => {
  const node = $(el);
  if (!node) { console.log(el, msg); return; }
  const line = document.createElement("div");
  if (cls) line.className = cls;
  line.textContent = msg;
  node.appendChild(line);
  node.scrollTop = node.scrollHeight;
};

function setStatus(msg, cls = "muted") {
  const el = $("status");
  if (!el) { console.log("[status]", msg); return; }
  el.className = cls;
  el.textContent = msg;
  if (cls === "err") toast(msg, "err");
}

function renderMeta() {
  if (!model) return;
  const m = model.metadata;
  const lines = [
    `architecture: ${m["general.architecture"] ?? "?"}`,
    `name: ${m["general.name"] ?? "?"}`,
    `file_type: ${m["general.file_type"] ?? "?"}`,
    `parameter_count: ${m["general.parameter_count"] ?? "?"}`,
    `tensors: ${model.tensors.length}`,
    `data_start: ${model.dataStart}`,
    `version: ${model.version}`,
    `pending quant overrides: ${pending.size}`,
  ];
  for (const [k, v] of Object.entries(m)) {
    if (/block_count|context_length|embedding_length|attention\.|rope\./.test(k)) {
      lines.push(`${k}: ${typeof v === "object" ? JSON.stringify(v) : v}`);
    }
  }
  if ($("meta")) $("meta").textContent = lines.join("\n");
}

function renderTensors() {
  const body = $("tensorBody");
  body.innerHTML = "";
  if (!model) return;
  model.tensors.forEach((t, i) => {
    const tr = document.createElement("tr");
    const ov = pending.get(i);
    const dtypeLabel = ov ? `${GGML_NAME[t.dtype]}→${GGML_NAME[ov.dtype]}` : (GGML_NAME[t.dtype] ?? t.dtype);
    const nbytes = ov ? ov.data.byteLength : t.nbytes;
    tr.innerHTML = `
      <td><input type="checkbox" data-i="${i}" ${selected.has(i) ? "checked" : ""} class="tchk" /></td>
      <td>${i}</td>
      <td>${t.name}${ov ? " *" : ""}</td>
      <td>${t.dims.join("×")}</td>
      <td>${dtypeLabel}</td>
      <td>${t.offset}</td>
      <td>${nbytes}</td>`;
    body.appendChild(tr);
  });
  body.querySelectorAll(".tchk").forEach((cb) => {
    cb.addEventListener("change", () => {
      const i = Number(cb.dataset.i);
      if (cb.checked) selected.add(i); else selected.delete(i);
    });
  });
  renderLayerList();
}

function listBlockLayers() {
  if (!model) return [];
  const set = new Set();
  for (const t of model.tensors) {
    const m = /^blk\.(\d+)\./.exec(t.name);
    if (m) set.add(Number(m[1]));
  }
  return [...set].sort((a, b) => a - b);
}

function renderLayerList() {
  const el = $("layerList");
  if (!el) return;
  if (!model) {
    el.textContent = "Load a GGUF to list layers.";
    return;
  }
  const layers = listBlockLayers();
  if (layers.length === 0) {
    el.textContent = "No blk.N.* tensors found (not a standard llama-style GGUF).";
    return;
  }
  el.innerHTML = layers.map((L) => {
    const count = model.tensors.filter((t) => t.name.startsWith(`blk.${L}.`)).length;
    return `<label style="display:block"><input type="checkbox" class="layerKeep" data-layer="${L}" checked /> keep layer ${L} (${count} tensors)</label>`;
  }).join("");
}

function dropUncheckedLayers() {
  if (!model) return;
  const layers = listBlockLayers();
  const keep = new Set();
  document.querySelectorAll(".layerKeep").forEach((cb) => {
    if (cb.checked) keep.add(Number(cb.dataset.layer));
  });
  if (keep.size === 0) {
    if ($("layerStatus")) {
      ($("layerStatus")||{}).textContent = "Keep at least one layer.";
      ($("layerStatus")||{}).className = "err";
    }
    return;
  }
  if (keep.size === layers.length) {
    if ($("layerStatus")) ($("layerStatus")||{}).textContent = "Nothing to drop.";
    return;
  }
  const mapping = new Map();
  let ni = 0;
  for (const L of layers) {
    if (keep.has(L)) {
      mapping.set(L, ni);
      ni++;
    }
  }
  const newTensors = [];
  for (let i = 0; i < model.tensors.length; i++) {
    const t = model.tensors[i];
    const m = /^blk\.(\d+)\.(.*)$/.exec(t.name);
    if (!m) {
      newTensors.push({ ...t, index: newTensors.length });
      continue;
    }
    const oldL = Number(m[1]);
    if (!mapping.has(oldL)) continue;
    const newL = mapping.get(oldL);
    newTensors.push({ ...t, name: `blk.${newL}.${m[2]}`, index: newTensors.length });
  }
  const newCount = mapping.size;
  for (const k of Object.keys(model.metadata)) {
    if (k.endsWith(".block_count")) model.metadata[k] = newCount;
  }
  if (model.metadataRaw) {
    for (const e of model.metadataRaw) {
      if (e.key.endsWith(".block_count")) e.value = newCount;
    }
  }
  pending = new Map();
  model.tensors = newTensors;
  selected = new Set();
  renderMeta();
  renderTensors();
  if ($("layerStatus")) {
    ($("layerStatus")||{}).textContent = `Kept ${newCount}/${layers.length} layers → ${newTensors.length} tensors. Export to write GGUF.`;
    ($("layerStatus")||{}).className = "ok";
  }
  log("opLog", `Structural drop: kept layers [${[...keep].sort((a, b) => a - b).join(",")}], block_count=${newCount}`, "ok");
}

function selectBy(pred) {
  selected.clear();
  model.tensors.forEach((t, i) => { if (pred(t)) selected.add(i); });
  renderTensors();
}

if ($("btnDropLayers")) $("btnDropLayers").onclick = dropUncheckedLayers;
if ($("btnKeepAllLayers")) {
  $("btnKeepAllLayers").onclick = () => {
    document.querySelectorAll(".layerKeep").forEach((cb) => { cb.checked = true; });
    if ($("layerStatus")) ($("layerStatus")||{}).textContent = "All layers marked keep.";
  };
}

$("file")?.addEventListener("change", () => setStatus("File selected — click Parse", "warn"));
$("btnParse").addEventListener("click", async () => {
  const f = $("file").files?.[0];
  if (!f) { setStatus("Pick a .gguf file first", "err"); return; }
  setStatus("Reading…", "warn");
  showLoading("Parsing GGUF…");
  try {
    const buf = await f.arrayBuffer();
    model = parseGguf(buf);
    model.fileName = f.name;
    selected = new Set();
    pending = new Map();
    renderMeta();
    renderTensors();
    renderWorkspace();
    setStatus(`Loaded ${f.name} — ${model.tensors.length} tensors`, "ok");
    toast("Loaded " + model.tensors.length + " tensors", "ok");
    workspace.name = model.fileName;
    workspace.baseName = model.fileName;
    workspace.ops = [];
    wsLog("loaded", model.tensors.length + " tensors");
    if ($("opLog")) $("opLog").textContent = "";
    hideLoading();
  } catch (e) {
    setStatus(String(e.message || e), "err");
    console.error(e);
  } finally {
    hideLoading();
  }
});

if ($("chkMaster")) $("chkMaster").addEventListener("change", (e) => {
  if (!model) return;
  if (e.target.checked) model.tensors.forEach((_, i) => selected.add(i));
  else selected.clear();
  renderTensors();
});
if ($("btnSelectAll")) $("btnSelectAll").onclick = () => { if (model) { model.tensors.forEach((_, i) => selected.add(i)); renderTensors(); } };
if ($("btnSelectNone")) $("btnSelectNone").onclick = () => { selected.clear(); renderTensors(); };
if ($("btnSelectLinear")) $("btnSelectLinear").onclick = () => selectBy((t) => /weight$/i.test(t.name) && !/norm/i.test(t.name));
if ($("btnSelectAttn")) $("btnSelectAttn").onclick = () => selectBy((t) => /attn|attention/i.test(t.name));
if ($("btnSelectFFN")) $("btnSelectFFN").onclick = () => selectBy((t) => /ffn|mlp|feed_forward/i.test(t.name));

// ---- Prune ----
if ($("btnPrune")) $("btnPrune").onclick = () => {
  if (!model || selected.size === 0) { log("opLog", "Nothing selected", "err"); return; }
  const thr = Number($("prunePct")?.value) || (Number($("prunePct")?.value) || 10) / 100;
  let touched = 0, zerosed = 0;
  for (const i of selected) {
    const t = model.tensors[i];
    try {
      let f32 = pending.has(i)
        ? (() => {
            // dequant pending first
            const ov = pending.get(i);
            const fake = { ...t, dtype: ov.dtype, absoluteOffset: 0, nbytes: ov.data.byteLength };
            // use standalone dequant on ov.data
            const nEl = t.nElements;
            if (ov.dtype === GGML.F32) return dequantF32(ov.data);
            if (ov.dtype === GGML.F16) return dequantF16(ov.data);
            if (ov.dtype === GGML.Q8_0) return dequantQ8_0(ov.data, nEl);
            if (ov.dtype === GGML.Q4_0) return dequantQ4_0(ov.data, nEl);
            if (ov.dtype === GGML.Q2_0) return dequantQ2_0(ov.data, nEl);
            throw new Error("unsupported pending");
          })()
        : tensorToFloat32(t, model.buffer);
      for (let j = 0; j < f32.length; j++) {
        if (Math.abs(f32[j]) < thr) { f32[j] = 0; zerosed++; }
      }
      const curDtype = pending.get(i)?.dtype ?? t.dtype;
      if (curDtype === GGML.F32 || curDtype === GGML.F16 || curDtype === GGML.Q8_0 || curDtype === GGML.Q4_0 || curDtype === GGML.Q2_0) {
        const { data, dtype } = quantizeFloats(f32, curDtype === GGML.BF16 ? GGML.F16 : curDtype);
        pending.set(i, { dtype, data });
      } else {
        const { data, dtype } = quantizeFloats(f32, GGML.F32);
        pending.set(i, { dtype, data });
      }
      touched++;
    } catch (e) {
      log("opLog", `prune skip ${t.name}: ${e.message}`, "warn");
    }
  }
  renderMeta();
  renderTensors();
  log("opLog", `Pruned ${touched} tensors, zeroed ${zerosed} values (thr=${thr})`, "ok");
};

// ---- Quantize selected (store in pending) ----
const TARGET_MAP = {
  F32: GGML.F32,
  F16: GGML.F16,
  Q8_0: GGML.Q8_0,
  Q4_0: GGML.Q4_0,
  Q2_0: GGML.Q2_0,
};

$("btnQuant").onclick = async () => {
  if (!model || selected.size === 0) { log("opLog", "Nothing selected", "err"); toast("Select tensors first", "err"); return; }
  const targetName = $("targetDtype")?.value || "Q4_0";
  const target = TARGET_MAP[targetName];
  if (target == null) { log("opLog", "Unknown type", "err"); return; }
  showLoading("Quantizing " + selected.size + " tensors…");
  let ok = 0, fail = 0;
  const list = [...selected];
  try {
    for (let n = 0; n < list.length; n++) {
      const i = list[n];
      const tens = model.tensors[i];
      try {
        let f32;
        if (pending.has(i)) {
          const ov = pending.get(i);
          if (ov.dtype === GGML.F32) f32 = dequantF32(ov.data);
          else if (ov.dtype === GGML.F16) f32 = dequantF16(ov.data);
          else if (ov.dtype === GGML.Q8_0) f32 = dequantQ8_0(ov.data, tens.nElements);
          else if (ov.dtype === GGML.Q4_0) f32 = dequantQ4_0(ov.data, tens.nElements);
          else if (ov.dtype === GGML.Q2_0) f32 = dequantQ2_0(ov.data, tens.nElements);
          else f32 = tensorToFloat32(tens, model.buffer);
        } else {
          f32 = tensorToFloat32(tens, model.buffer);
        }
        const q = quantizeFloats(f32, target);
        pending.set(i, { dtype: target, data: q });
        ok++;
      } catch (e) {
        fail++;
        log("opLog", tens.name + ": " + e.message, "err");
      }
      if (n % 2 === 0) {
        setProgress(10 + (80 * (n + 1)) / list.length);
        await yieldToUI();
      }
    }
    renderTensors();
    log("opLog", "Quant " + targetName + ": " + ok + " ok, " + fail + " fail", ok ? "ok" : "err");
    toast("Quant done: " + ok + " tensors", ok ? "ok" : "err");
    if (ok) wsLog("quantize", targetName + " on " + ok + " tensors");
  } finally {
    hideLoading();
  }
};

if ($("btnQuantAll")) $("btnQuantAll").onclick = () => {
  if (!model) return;
  model.tensors.forEach((_, i) => selected.add(i));
  renderTensors();
  $("btnQuant").click();
};

if ($("btnClearPending")) $("btnClearPending").onclick = () => {
  pending.clear();
  renderMeta();
  renderTensors();
  log("opLog", "Cleared pending quant overrides", "warn");
};

// ---- Export (full rebuild with quant) ----
$("btnExport").onclick = async () => {
  if (!model) { toast("Load a model first", "err"); return; }
  showLoading("Building GGUF…");
  try {
    const exportQuant = $("exportQuant")?.value || "keep";
    if (exportQuant !== "keep" && selected.size > 0) {
      const target = TARGET_MAP[exportQuant];
      let n = 0;
      for (const i of selected) {
        const tens = model.tensors[i];
        try {
          const f32 = tensorToFloat32(tens, model.buffer);
          pending.set(i, { dtype: target, data: quantizeFloats(f32, target) });
        } catch (e) {
          log("exportLog", "skip " + tens.name + ": " + e.message, "warn");
        }
        n++;
        if (n % 3 === 0) { setProgress(20 + 40 * n / selected.size); await yieldToUI(); }
      }
    }
    setProgress(70);
    await yieldToUI();
    const buf = writeGguf(model, pending);
    setProgress(90);
    const blob = new Blob([buf], { type: "application/octet-stream" });
    const a = document.createElement("a");
    a.href = URL.createObjectURL(blob);
    a.download = (model.fileName || "model").replace(/\.gguf$/i, "") + "-studio.gguf";
    a.click();
    URL.revokeObjectURL(a.href);
    log("exportLog", "Exported " + (buf.byteLength / 1e6).toFixed(2) + " MB", "ok");
    toast("Downloaded GGUF (" + (buf.byteLength / 1e6).toFixed(1) + " MB)", "ok");
    wsLog("export", (buf.byteLength / 1e6).toFixed(1) + " MB");
  } catch (e) {
    console.error(e);
    log("exportLog", String(e.message || e), "err");
    toast("Export failed: " + (e.message || e), "err");
  } finally {
    hideLoading();
  }
};

// ---- Unsloth job export + in-browser approx FT ----
let trainAbort = false;

function tlog(msg, cls = "") {
  const el = $("trainLog");
  if (!el) return;
  const div = document.createElement("div");
  if (cls) div.className = cls;
  div.textContent = msg;
  el.appendChild(div);
  el.scrollTop = el.scrollHeight;
}

async function loadUnslothMod() {
  return import("./unsloth-export.js");
}

if ($("btnExportUnsloth")) {
  $("btnExportUnsloth").onclick = async () => {
    try {
      const mod = await loadUnslothMod();
      const cfg = mod.collectJobConfigFromDom($);
      const raw = $("dataset").value || "";
      const py = mod.buildUnslothPython(cfg, raw);
      mod.downloadBlob("unsloth_job.json", JSON.stringify(cfg, null, 2), "application/json");
      mod.downloadBlob("unsloth_train.py", py, "text/x-python");
      $("usCfgPreview").textContent = JSON.stringify(cfg, null, 2);
      $("usExportStatus").textContent = "Downloaded unsloth_job.json + unsloth_train.py";
      $("usExportStatus").className = "ok";
    } catch (e) {
      $("usExportStatus").textContent = String(e.message || e);
      $("usExportStatus").className = "err";
      console.error(e);
    }
  };
}

if ($("btnCopyUnslothCfg")) {
  $("btnCopyUnslothCfg").onclick = async () => {
    try {
      const mod = await loadUnslothMod();
      const cfg = mod.collectJobConfigFromDom($);
      await navigator.clipboard.writeText(JSON.stringify(cfg, null, 2));
      $("usExportStatus").textContent = "Config JSON copied";
      $("usExportStatus").className = "ok";
    } catch (e) {
      $("usExportStatus").textContent = String(e.message || e);
      $("usExportStatus").className = "err";
    }
  };
}

if ($("btnTrainStop")) {
  $("btnTrainStop").onclick = () => {
    trainAbort = true;
    $("trainStatus").textContent = "Stopping…";
  };
}

if ($("btnTrain")) $("btnTrain").onclick = async () => {
  trainAbort = false;
  $("trainLog").textContent = "";
  const steps = Number($("trainSteps")?.value) || 20;
  const seqLen = Number($("trainSeq")?.value) || 64;
  const lr = Number($("trainLr")?.value) || 3e-4;
  const fakeQuant = $("trainFakeQuant")?.value || "none";
  const source = model ? "gguf" : "scratch";
  const exportQ = $("trainExportQuant")?.value || "Q4_0";
  const raw = ($("dataset")?.value || "").trim();
  if (!raw) {
    tlog("Paste dataset (messages JSONL / ShareGPT / text)", "err");
    hideLoading();
    return;
  }

  const usMod = await loadUnslothMod();
  const dataFmt = $("dsFormat")?.value || "messages";
  const chatTpl = $("chatTpl")?.value || "chatml";
  const trainOnResp = $("trainOnResp")?.checked !== false;
  const examples = usMod.parseDataset(raw, dataFmt);
  if (examples.length === 0) {
    tlog("No examples parsed", "err");
    hideLoading();
    return;
  }
  tlog(`Parsed ${examples.length} examples | format=${dataFmt} | train_on_responses_only=${trainOnResp}`, "ok");

  $("trainStatus").textContent = "Loading trainer…";
  showLoading("Starting fine-tune…");
  let TinyLM, loadTinyLMFromGguf, tinyLMToPending;
  try {
    let mod;
    try {
      mod = await import("./train.js");
    } catch (ie) {
      tlog("Failed to load train.js: " + (ie.message || ie), "err"); toast("Train module failed: " + (ie.message || ie), "err");
      console.error(ie);
      $("trainStatus").textContent = "Module load error";
      hideLoading();
      return;
    }
    TinyLM = mod.TinyLM;
    loadTinyLMFromGguf = mod.loadTinyLMFromGguf;
    tinyLMToPending = mod.tinyLMToPending;
  } catch (e) {
    tlog("Failed to load train.js: " + e.message, "err");
    return;
  }

  let lm;
  try {
    if (source === "gguf") {
      if (!model) {
        tlog("Load a GGUF first (or switch to train from scratch)", "err");
        return;
      }
      lm = loadTinyLMFromGguf(model, tensorToFloat32);
      try {
        const tokMod = await import("./tokenizer.js");
        lm.tokenizer = tokMod.extractTokenizerFromMetadata(model.metadata || {});
        tlog(`Tokenizer: kind=${lm.tokenizer.kind} vocab=${lm.tokenizer.vocabSize}`, "ok");
      } catch (e) {
        lm.tokenizer = { kind: "byte", vocabSize: 256 };
        tlog("Tokenizer fallback: byte", "warn");
      }
      tlog(`Loaded GGUF → TinyLM emb=${lm.cfg.nEmb} layers=${lm.cfg.nLayer}`, "ok");
    } else {
      lm = new TinyLM({
        nVocab: Number($("scratchVocab").value) || 256,
        nEmb: Number($("scratchEmb").value) || 128,
        nLayer: Number($("scratchLayers").value) || 2,
        nHead: Number($("scratchHeads").value) || 4,
      });
      tlog(`Scratch TinyLM emb=${lm.cfg.nEmb} layers=${lm.cfg.nLayer}`, "ok");
    }
  } catch (e) {
    tlog("Init failed: " + e.message, "err");
    hideLoading();
    toast("Train init failed: " + e.message, "err");
    return;
  }

  lm.fakeQuant = fakeQuant;
  const useLora = $("trainUseLora")?.checked !== false;
  if (useLora) {
    const r = Number($("trainLoraR")?.value) || Number($("usLoraR")?.value) || 8;
    const alpha = Number($("trainLoraAlpha")?.value) || Number($("usLoraAlpha")?.value) || 16;
    const nAd = lm.enableLora(r, alpha);
    tlog(`LoRA enabled r=${r} alpha=${alpha} adapters=${nAd} (base frozen)`, "ok");
    if (nAd === 0) tlog("No matching linear tensors for LoRA — training dense approx", "warn");
  } else {
    tlog("Dense FT (no LoRA)", "muted");
  }
  $("trainStatus").textContent = "Training…";
  const losses = [];

  try {
    for (let step = 0; step < steps; step++) {
      if (trainAbort) throw new Error("aborted");
      const ex = examples[step % examples.length];
      const { ids, mask } = usMod.exampleToIdsAndMask(ex, chatTpl, seqLen, trainOnResp);
      const { loss, grads } = lm.lossAndGrad(ids, { fullLayers: true, mask: trainOnResp ? mask : null });
      lm.adamStep(grads, lr);
      losses.push(loss);
      if (step % 1 === 0) {
        setProgress(10 + 85 * (step + 1) / steps);
        const avg = losses.slice(-20).reduce((a, b) => a + b, 0) / Math.min(20, losses.length);
        if ($("trainStatus")) $("trainStatus").textContent = `step ${step + 1}/${steps} loss=${loss.toFixed(4)} avg20=${avg.toFixed(4)}`;
        if (step % 5 === 0) {
          tlog(`step ${step} loss=${loss.toFixed(4)}`, "ok");
          showLoading(`Training step ${step + 1}/${steps}…`);
        }
        await yieldToUI();
      }
    }
  } catch (e) {
    if (String(e.message).includes("aborted")) tlog("Stopped by user", "warn");
    else {
      tlog("Train error: " + e.message, "err");
      console.error(e);
      $("trainStatus").textContent = "Error";
      hideLoading();
      toast("Train error", "err");
      return;
    }
  }

  if (model && source === "gguf") {
    const target = TARGET_MAP[exportQ] ?? GGML.Q4_0;
    const mapped = tinyLMToPending(lm, model, quantizeFloats, target);
    for (const [i, ov] of mapped) pending.set(i, ov);
    renderMeta();
    renderTensors();
    tlog(`Wrote ${mapped.size} tensors into pending as ${exportQ}. Export GGUF when ready.`, "ok");
  } else {
    tlog("Scratch train done (in-memory only).", "warn");
  }
  $("trainStatus").textContent = "Done";
  hideLoading();
  wsLog("train", "finished");
  toast("Training finished", "ok");
};

setStatus("Ready — load a GGUF", "muted");

// ---- Sidebar navigation ----
document.querySelectorAll(".nav-btn").forEach((btn) => {
  btn.addEventListener("click", () => {
    const v = btn.dataset.view;
    document.querySelectorAll(".nav-btn").forEach((b) => b.classList.remove("active"));
    document.querySelectorAll(".view").forEach((el) => el.classList.remove("active"));
    btn.classList.add("active");
    const panel = document.getElementById("view-" + v);
    if (panel) panel.classList.add("active");
  });
});

// ---- wllama chat ----
let wllamaInst = null;
let chatBusy = false;

function chatLog(role, text) {
  const box = $("chatMessages");
  if (!box) return;
  const div = document.createElement("div");
  div.className = "bubble " + role;
  div.textContent = text;
  box.appendChild(div);
  box.scrollTop = box.scrollHeight;
}

async function getWllama() {
  const { Wllama } = await import("https://cdn.jsdelivr.net/npm/@wllama/wllama@2.2.1/esm/index.js");
  const { default: WasmFromCDN } = await import("https://cdn.jsdelivr.net/npm/@wllama/wllama@2.2.1/esm/wasm-from-cdn.js");
  return new Wllama(WasmFromCDN);
}

async function loadGgufIntoWllama(buf) {
  if ($("chatStatus")) $("chatStatus").textContent = "Loading WASM / model…";
  if (wllamaInst) {
    try { await wllamaInst.exit(); } catch (_) {}
  }
  wllamaInst = await getWllama();
  await wllamaInst.loadModel(new Blob([buf], { type: "application/octet-stream" }));
  if ($("chatStatus")) {
    $("chatStatus").textContent = "Ready";
    $("chatStatus").className = "ok";
  }
}

if ($("btnChatLoad")) {
  $("btnChatLoad").onclick = async () => {
    try {
      if (!model) {
        $("chatStatus").textContent = "Parse a GGUF first (or use file picker)";
        $("chatStatus").className = "warn";
        return;
      }
      // Prefer pending export buffer if we can rebuild quickly
      const buf = writeGguf(model, pending);
      await loadGgufIntoWllama(buf);
      chatLog("assistant", "(model loaded)");
    } catch (e) {
      console.error(e);
      if ($("chatStatus")) {
        $("chatStatus").textContent = String(e.message || e);
        $("chatStatus").className = "err";
      }
    }
  };
}

if ($("chatFile")) {
  $("chatFile").addEventListener("change", async () => {
    const f = $("chatFile").files?.[0];
    if (!f) return;
    try {
      const buf = await f.arrayBuffer();
      await loadGgufIntoWllama(buf);
      chatLog("assistant", "(loaded " + f.name + ")");
    } catch (e) {
      console.error(e);
      if ($("chatStatus")) {
        $("chatStatus").textContent = String(e.message || e);
        $("chatStatus").className = "err";
      }
    }
  });
}

async function chatSend() {
  if (!wllamaInst || chatBusy) return;
  const input = $("chatInput");
  const text = (input?.value || "").trim();
  if (!text) return;
  input.value = "";
  chatLog("user", text);
  chatBusy = true;
  if ($("chatStatus")) $("chatStatus").textContent = "Generating…";
  try {
    const nPredict = Number($("chatNPredict")?.value) || 64;
    const temp = Number($("chatTemp")?.value) || 0.7;
    // ChatML-ish prompt
    const prompt = `<|im_start|>user\n${text}<|im_end|>\n<|im_start|>assistant\n`;
    let out = "";
    const bubble = document.createElement("div");
    bubble.className = "bubble assistant";
    bubble.textContent = "";
    $("chatMessages")?.appendChild(bubble);
    await wllamaInst.createCompletion(prompt, {
      nPredict,
      sampling: { temp },
      onNewToken: (_token, piece) => {
        out += piece;
        bubble.textContent = out;
        $("chatMessages").scrollTop = $("chatMessages").scrollHeight;
      },
    });
    if (!out) bubble.textContent = "(empty)";
    if ($("chatStatus")) {
      $("chatStatus").textContent = "Ready";
      $("chatStatus").className = "ok";
    }
  } catch (e) {
    console.error(e);
    chatLog("assistant", "Error: " + (e.message || e));
    if ($("chatStatus")) {
      $("chatStatus").textContent = String(e.message || e);
      $("chatStatus").className = "err";
    }
  }
  chatBusy = false;
}

if ($("btnChatSend")) $("btnChatSend").onclick = () => chatSend();
if ($("chatInput")) {
  $("chatInput").addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      chatSend();
    }
  });
}



function keepFirstNLayers(n) {
  if (!model) return;
  const layers = listBlockLayers();
  if (!layers.length) {
    log("layerLog", "No blk.* layers found", "err");
    return;
  }
  n = Math.max(1, Math.min(n, layers.length));
  // Simulate layerKeep checkboxes
  const keep = new Set(layers.filter((L) => L < n));
  const mapping = new Map();
  let ni = 0;
  for (const L of layers) {
    if (keep.has(L)) { mapping.set(L, ni); ni++; }
  }
  const newTensors = [];
  for (const t of model.tensors) {
    const m = /^blk\.(\d+)\.(.*)$/.exec(t.name);
    if (!m) {
      newTensors.push({ ...t, index: newTensors.length });
      continue;
    }
    const oldL = Number(m[1]);
    if (!mapping.has(oldL)) continue;
    newTensors.push({ ...t, name: `blk.${mapping.get(oldL)}.${m[2]}`, index: newTensors.length });
  }
  for (const k of Object.keys(model.metadata)) {
    if (k.endsWith(".block_count")) model.metadata[k] = mapping.size;
  }
  if (model.metadataRaw) {
    for (const e of model.metadataRaw) {
      if (e.key.endsWith(".block_count")) e.value = mapping.size;
    }
  }
  pending = new Map();
  model.tensors = newTensors;
  selected = new Set();
  renderMeta();
  renderTensors();
  log("layerLog", `Kept first ${n} layers → ${newTensors.length} tensors. Export to write GGUF.`, "ok");
  wsLog("drop layers", "kept first " + n + " → " + newTensors.length + " tensors");
  if (model) model.fileName = (workspace.baseName || "model").replace(/\.gguf$/i, "") + `-${n}L.gguf`;
}
if ($("btnDropTail")) {
  $("btnDropTail").onclick = () => keepFirstNLayers(Number($("keepLayers")?.value) || 1);
}
if ($("btnDropSelected")) {
  $("btnDropSelected").onclick = () => {
    if (!model || selected.size === 0) return;
    // keep tensors NOT selected that are blk, plus all non-blk
    const dropLayers = new Set();
    for (const i of selected) {
      const m = model.tensors[i].name.match(/^blk\.(\d+)\./);
      if (m) dropLayers.add(Number(m[1]));
    }
    const layers = listBlockLayers();
    const keepN = layers.filter((L) => !dropLayers.has(L));
    if (!keepN.length) { log("layerLog", "Would drop all layers", "err"); return; }
    // rebuild keep set
    const mapping = new Map();
    let ni = 0;
    for (const L of layers) {
      if (!dropLayers.has(L)) { mapping.set(L, ni); ni++; }
    }
    const newTensors = [];
    for (const t of model.tensors) {
      const m = /^blk\.(\d+)\.(.*)$/.exec(t.name);
      if (!m) { newTensors.push({ ...t, index: newTensors.length }); continue; }
      const oldL = Number(m[1]);
      if (!mapping.has(oldL)) continue;
      newTensors.push({ ...t, name: `blk.${mapping.get(oldL)}.${m[2]}`, index: newTensors.length });
    }
    for (const k of Object.keys(model.metadata)) {
      if (k.endsWith(".block_count")) model.metadata[k] = mapping.size;
    }
    if (model.metadataRaw) {
      for (const e of model.metadataRaw) {
        if (e.key.endsWith(".block_count")) e.value = mapping.size;
      }
    }
    pending = new Map();
    model.tensors = newTensors;
    selected = new Set();
    renderMeta();
    renderTensors();
    log("layerLog", `Dropped selected layers → ${mapping.size} blocks, ${newTensors.length} tensors`, "ok");
  };
}

// Test / automation hooks
window.__ggufStudio = {
  parseBuffer: (ab, name = "test.gguf") => {
    model = parseGguf(ab);
    model.fileName = name;
    selected = new Set();
    pending = new Map();
    renderMeta();
    renderTensors();
    renderWorkspace();
    setStatus(`Loaded ${name} — ${model.tensors.length} tensors`, "ok");
    return { tensors: model.tensors.length, meta: model.metadata };
  },
  getModel: () => model,
  writeBuffer: () => writeGguf(model, pending),
  quantSelected: (dtype) => {
    if ($("targetDtype")) $("targetDtype").value = dtype;
    $("btnQuant")?.click();
  },
  selectAll: () => { if (model) { model.tensors.forEach((_, i) => selected.add(i)); renderTensors(); } },
};

// ---- Dataset parsing assistant ----
let _rawDatasetText = "";
let _parsedPreview = [];

function detectDatasetFormat(text) {
  const lines = text.split(/\r?\n/).map((l) => l.trim()).filter(Boolean);
  if (!lines.length) return { format: "text", samples: [] };
  const samples = [];
  // try JSONL first
  let jsonlOk = 0;
  for (const line of lines.slice(0, 20)) {
    try {
      const o = JSON.parse(line);
      jsonlOk++;
      if (o.messages && Array.isArray(o.messages)) {
        samples.push(o);
      } else if (o.conversations || (o.from && o.value) || Array.isArray(o)) {
        samples.push(o);
      } else if (o.instruction != null || o.input != null || o.output != null) {
        samples.push(o);
      } else if (o.prompt != null || o.response != null || o.completion != null) {
        samples.push(o);
      } else {
        samples.push(o);
      }
    } catch {
      break;
    }
  }
  if (jsonlOk >= Math.min(3, lines.length) && jsonlOk === Math.min(20, lines.length)) {
    const first = samples[0] || {};
    if (first.messages) return { format: "messages", samples, lines };
    if (first.instruction != null || first.output != null) return { format: "alpaca", samples, lines };
    if (first.conversations || first.from) return { format: "sharegpt", samples, lines };
    return { format: "messages", samples, lines };
  }
  // full JSON array?
  try {
    const o = JSON.parse(text);
    if (Array.isArray(o) && o.length) {
      if (o[0].messages) return { format: "messages", samples: o.slice(0, 20), lines };
      if (o[0].instruction != null) return { format: "alpaca", samples: o.slice(0, 20), lines };
      return { format: "messages", samples: o.slice(0, 20), lines };
    }
  } catch (_) {}
  // CSV
  if (lines[0].includes(",") && (lines[0].toLowerCase().includes("prompt") || lines[0].toLowerCase().includes("instruction"))) {
    return { format: "csv", samples: lines.slice(0, 10), lines };
  }
  return { format: "text", samples: lines.slice(0, 10), lines };
}

function normalizeToMessages(format, rawText, promptCol, responseCol) {
  const out = [];
  const lines = rawText.split(/\r?\n/).map((l) => l.trim()).filter(Boolean);
  if (format === "text") {
    for (const line of lines) {
      out.push({ messages: [
        { role: "user", content: "Say this:" },
        { role: "assistant", content: line },
      ]});
    }
    return out;
  }
  if (format === "csv") {
    const header = lines[0].split(",").map((h) => h.trim().replace(/^"|"$/g, ""));
    const pi = header.findIndex((h) => h.toLowerCase() === promptCol.toLowerCase() || h.toLowerCase() === "prompt" || h.toLowerCase() === "instruction");
    const ri = header.findIndex((h) => h.toLowerCase() === responseCol.toLowerCase() || h.toLowerCase() === "response" || h.toLowerCase() === "output" || h.toLowerCase() === "completion");
    for (const line of lines.slice(1)) {
      // naive CSV split
      const cols = line.match(/("([^"]|"")*"|[^,]*)/g)?.map((c) => c.replace(/^"|"$/g, "").replace(/""/g, '"')) || line.split(",");
      const prompt = cols[pi >= 0 ? pi : 0] || "";
      const resp = cols[ri >= 0 ? ri : 1] || "";
      if (!prompt && !resp) continue;
      out.push({ messages: [
        { role: "user", content: String(prompt) },
        { role: "assistant", content: String(resp) },
      ]});
    }
    return out;
  }
  // JSONL / JSON objects
  let objs = [];
  try {
    const o = JSON.parse(rawText);
    if (Array.isArray(o)) objs = o;
  } catch (_) {
    for (const line of lines) {
      try { objs.push(JSON.parse(line)); } catch (_) {}
    }
  }
  for (const o of objs) {
    if (format === "messages" && o.messages) {
      out.push({ messages: o.messages });
    } else if (format === "alpaca") {
      const user = [o.instruction, o.input].filter(Boolean).join("\n");
      out.push({ messages: [
        { role: "user", content: user || "" },
        { role: "assistant", content: o.output || o.response || "" },
      ]});
    } else if (format === "sharegpt") {
      const conv = o.conversations || o.conversation || [];
      if (conv.length) {
        const messages = conv.map((c) => ({
          role: (c.from === "human" || c.from === "user") ? "user" : "assistant",
          content: c.value || c.content || "",
        }));
        out.push({ messages });
      } else if (o.from && o.value) {
        // single turn leftover — skip
      }
    } else if (o.prompt != null) {
      out.push({ messages: [
        { role: "user", content: String(o.prompt) },
        { role: "assistant", content: String(o.response || o.completion || o.output || "") },
      ]});
    } else if (o.messages) {
      out.push({ messages: o.messages });
    }
  }
  return out;
}

if ($("datasetFile")) {
  $("datasetFile").addEventListener("change", async () => {
    const f = $("datasetFile").files?.[0];
    if (!f) return;
    showLoading("Reading dataset…");
    try {
      _rawDatasetText = await f.text();
      const det = detectDatasetFormat(_rawDatasetText);
      if ($("dsDetectedFormat")) $("dsDetectedFormat").value = det.format;
      if ($("datasetAssist")) $("datasetAssist").style.display = "block";
      if ($("datasetStatus")) {
        $("datasetStatus").textContent = `${f.name}: ~${det.lines?.length || 0} lines, detected ${det.format}`;
        $("datasetStatus").className = "ok";
      }
      if ($("datasetPreview")) {
        $("datasetPreview").textContent = JSON.stringify(det.samples.slice(0, 5), null, 2).slice(0, 2500);
      }
    } catch (e) {
      if ($("datasetStatus")) {
        $("datasetStatus").textContent = String(e.message || e);
        $("datasetStatus").className = "err";
      }
    } finally {
      hideLoading();
    }
  });
}

if ($("btnParseDataset")) {
  $("btnParseDataset").onclick = () => {
    const text = _rawDatasetText || $("dataset")?.value || "";
    if (!text.trim()) {
      if ($("datasetStatus")) {
        $("datasetStatus").textContent = "Paste or upload data first";
        $("datasetStatus").className = "warn";
      }
      return;
    }
    _rawDatasetText = text;
    const det = detectDatasetFormat(text);
    if ($("dsDetectedFormat")) $("dsDetectedFormat").value = det.format;
    if ($("datasetAssist")) $("datasetAssist").style.display = "block";
    if ($("datasetPreview")) {
      $("datasetPreview").textContent = JSON.stringify(det.samples.slice(0, 5), null, 2).slice(0, 2500);
    }
    if ($("datasetStatus")) {
      $("datasetStatus").textContent = `Detected ${det.format} — review & Apply`;
      $("datasetStatus").className = "ok";
    }
  };
}

if ($("btnApplyParse")) {
  $("btnApplyParse").onclick = () => {
    const format = $("dsDetectedFormat")?.value || "messages";
    const promptCol = $("dsPromptCol")?.value || "prompt";
    const responseCol = $("dsResponseCol")?.value || "response";
    const raw = _rawDatasetText || $("dataset")?.value || "";
    const msgs = normalizeToMessages(format, raw, promptCol, responseCol);
    if (!msgs.length) {
      if ($("datasetStatus")) {
        $("datasetStatus").textContent = "No samples parsed — adjust format/columns";
        $("datasetStatus").className = "err";
      }
      return;
    }
    const jsonl = msgs.map((m) => JSON.stringify(m)).join("\n");
    if ($("dataset")) $("dataset").value = jsonl;
    if ($("dsFormat")) $("dsFormat").value = "messages";
    if ($("datasetStatus")) {
      $("datasetStatus").textContent = `Applied ${msgs.length} samples as messages JSONL`;
      $("datasetStatus").className = "ok";
    }
  };
}

// Mobile drawer
(function setupMobileNav() {
  const side = $("sidebar");
  const bd = $("backdrop");
  const open = () => { side?.classList.add("open"); bd?.classList.add("show"); };
  const close = () => { side?.classList.remove("open"); bd?.classList.remove("show"); };
  $("btnMenu")?.addEventListener("click", () => {
    if (side?.classList.contains("open")) close(); else open();
  });
  bd?.addEventListener("click", close);
  document.querySelectorAll(".nav-btn").forEach((btn) => {
    btn.addEventListener("click", () => {
      const v = btn.dataset.view;
      document.querySelectorAll(".nav-btn").forEach((b) => b.classList.remove("active"));
      document.querySelectorAll(".view").forEach((el) => el.classList.remove("active"));
      btn.classList.add("active");
      $("view-" + v)?.classList.add("active");
      close();
    });
  });
})();

window.onerror = function(msg, src, line, col, err) {
  toast(String(msg), "err");
  console.error(msg, src, line, err);
  return false;
};
window.addEventListener("unhandledrejection", (e) => {
  toast(String(e.reason?.message || e.reason), "err");
});
