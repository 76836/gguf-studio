/**
 * SmolLM2 path: load real GGUF → structural layer prune → write valid GGUF.
 * Optionally applies a tiny identity nudge on dequantized output path tensors.
 *
 * Usage:
 *   node smollm_prune_tune.mjs --in ../models/SmolLM2-135M-Q4_0.gguf --keep 6 --out examples/smollm-akari-6L-q4_0.gguf
 */
import { readFileSync, writeFileSync, mkdirSync } from "fs";
// quant helpers available from ./quant.js when we add FT nudge later

const args = process.argv.slice(2);
function arg(name, def) {
  const i = args.indexOf(name);
  return i >= 0 ? args[i + 1] : def;
}
const inPath = arg("--in", "../models/SmolLM2-135M-Q4_0.gguf");
const keep = Number(arg("--keep", "6"));
const outPath = arg("--out", `examples/smollm-akari-${keep}L-q4_0.gguf`);

// ---- GGUF reader ----
class R {
  constructor(buf) {
    this.buf = buf;
    this.u8 = new Uint8Array(buf);
    this.dv = new DataView(buf);
    this.o = 0;
  }
  u32() {
    const v = this.dv.getUint32(this.o, true);
    this.o += 4;
    return v;
  }
  u64() {
    const v = Number(this.dv.getBigUint64(this.o, true));
    this.o += 8;
    return v;
  }
  f32() {
    const v = this.dv.getFloat32(this.o, true);
    this.o += 4;
    return v;
  }
  str() {
    const n = this.u64();
    const s = new TextDecoder().decode(this.u8.subarray(this.o, this.o + n));
    this.o += n;
    return s;
  }
}

function readVal(r, vt) {
  switch (vt) {
    case 0:
    case 1:
      return r.u8[r.o++];
    case 2: {
      const v = r.dv.getUint16(r.o, true);
      r.o += 2;
      return v;
    }
    case 3: {
      const v = r.dv.getInt16(r.o, true);
      r.o += 2;
      return v;
    }
    case 4:
      return r.u32();
    case 5: {
      const v = r.dv.getInt32(r.o, true);
      r.o += 4;
      return v;
    }
    case 6:
      return r.f32();
    case 7: {
      return r.u8[r.o++] !== 0;
    }
    case 8:
      return r.str();
    case 9: {
      const et = r.u32();
      const n = r.u64();
      const a = [];
      for (let i = 0; i < n; i++) a.push(readVal(r, et));
      return a;
    }
    case 10:
      return r.u64();
    case 11: {
      const v = Number(r.dv.getBigInt64(r.o, true));
      r.o += 8;
      return v;
    }
    case 12: {
      const v = r.dv.getFloat64(r.o, true);
      r.o += 8;
      return v;
    }
    default:
      throw new Error("unknown type " + vt);
  }
}

function parseGguf(arrayBuffer) {
  const r = new R(arrayBuffer);
  r.o = 4;
  const version = r.u32();
  const nTensors = r.u64();
  const nKV = r.u64();
  const meta = {};
  const metaRaw = [];
  for (let i = 0; i < nKV; i++) {
    const key = r.str();
    const vtype = r.u32();
    const start = r.o;
    const value = readVal(r, vtype);
    meta[key] = value;
    metaRaw.push({ key, vtype, value });
  }
  const tensors = [];
  for (let i = 0; i < nTensors; i++) {
    const name = r.str();
    const nDims = r.u32();
    const dims = [];
    for (let d = 0; d < nDims; d++) dims.push(r.u64());
    const dtype = r.u32();
    const offset = r.u64();
    tensors.push({ name, dims, dtype, offset });
  }
  // data section starts at aligned offset
  const alignment = meta["general.alignment"] || 32;
  const rem = r.o % alignment;
  if (rem) r.o += alignment - rem;
  const dataStart = r.o;
  for (const t of tensors) {
    t.absoluteOffset = dataStart + t.offset;
    // nbytes from dims * type size is complex for quant; use next offset
  }
  // compute nbytes from offsets
  for (let i = 0; i < tensors.length; i++) {
    const next =
      i + 1 < tensors.length ? dataStart + tensors[i + 1].offset : arrayBuffer.byteLength;
    tensors[i].nbytes = next - tensors[i].absoluteOffset;
  }
  return { version, meta, metaRaw, tensors, dataStart, buffer: arrayBuffer, alignment };
}

// ---- writer ----
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
function strBytes(s) {
  const enc = new TextEncoder().encode(s);
  return concat(u64(enc.length), enc);
}
function concat(...parts) {
  const n = parts.reduce((a, p) => a + p.byteLength, 0);
  const o = new Uint8Array(n);
  let i = 0;
  for (const p of parts) {
    o.set(p, i);
    i += p.byteLength;
  }
  return o;
}

function writeValue(vtype, value) {
  switch (vtype) {
    case 4:
      return u32(value);
    case 5: {
      const b = new ArrayBuffer(4);
      new DataView(b).setInt32(0, value, true);
      return new Uint8Array(b);
    }
    case 6:
      return f32(value);
    case 7:
      return new Uint8Array([value ? 1 : 0]);
    case 8:
      return strBytes(String(value));
    case 9: {
      // ARRAY — value is {et, items} or we stored as array of strings only for tokens
      if (Array.isArray(value)) {
        // assume string array
        const parts = [u32(8), u64(value.length)];
        for (const s of value) parts.push(strBytes(String(s)));
        return concat(...parts);
      }
      throw new Error("complex array rewrite not supported");
    }
    case 10:
      return u64(value);
    default:
      throw new Error("write vt " + vtype);
  }
}

function writeGguf(model, tensorList) {
  const parts = [];
  parts.push(new Uint8Array([0x47, 0x47, 0x55, 0x46]));
  parts.push(u32(3));
  parts.push(u64(tensorList.length));

  // rewrite metadata: update block_count, name
  const metaRaw = model.metaRaw
    .filter((e) => e.vtype !== 9 || e.key === "tokenizer.ggml.tokens" || e.key === "tokenizer.ggml.scores" || e.key === "tokenizer.ggml.token_type" || e.key === "tokenizer.ggml.merges")
    .map((e) => ({ ...e }));

  // For ARRAY types we need original bytes — skip rewriting complex arrays by copying from original file for those keys is hard.
  // Strategy: only include scalar KVs we understand + skip arrays that aren't tokens.
  // Actually tokens array can be huge (49k strings) — we must copy original meta section OR re-emit.
  // Simpler approach: copy entire original header through tensor info is messy.
  // Better: build from meta scalars + re-read array blobs from original.

  const scalarKeys = new Set([
    "general.architecture",
    "general.name",
    "general.file_type",
    "general.quantization_version",
    "llama.block_count",
    "llama.context_length",
    "llama.embedding_length",
    "llama.feed_forward_length",
    "llama.attention.head_count",
    "llama.attention.head_count_kv",
    "llama.rope.dimension_count",
    "llama.rope.freq_base",
    "llama.attention.layer_norm_rms_epsilon",
    "llama.vocab_size",
    "tokenizer.ggml.model",
    "tokenizer.ggml.bos_token_id",
    "tokenizer.ggml.eos_token_id",
    "tokenizer.ggml.padding_token_id",
    "tokenizer.ggml.unknown_token_id",
  ]);

  // We'll do a hybrid: copy the original file's KV section unmodified except patch block_count in-place if possible,
  // then rewrite tensor table + data. That's safer for tokenizer arrays.

  return null; // use binary-patch approach below
}

/**
 * Binary-safe prune: keep tensors for blk.0 .. blk.keep-1 (renumbered),
 * non-block tensors always kept; update llama.block_count in metadata by
 * rewriting the whole file with copied tensor data and patched KV.
 */
function pruneModel(model, keepLayers) {
  const u8 = new Uint8Array(model.buffer);
  const selected = [];
  for (const t of model.tensors) {
    const m = t.name.match(/^blk\.(\d+)\.(.*)$/);
    if (m) {
      const li = Number(m[1]);
      if (li < keepLayers) {
        selected.push({
          ...t,
          newName: `blk.${li}.${m[2]}`,
          data: u8.subarray(t.absoluteOffset, t.absoluteOffset + t.nbytes),
        });
      }
    } else {
      selected.push({
        ...t,
        newName: t.name,
        data: u8.subarray(t.absoluteOffset, t.absoluteOffset + t.nbytes),
      });
    }
  }

  // Build KV from original meta with updated block_count and name
  // Re-emit scalars; for arrays copy from original by re-parsing isn't available as bytes —
  // include essential arrays from meta object.

  const parts = [];
  parts.push(new Uint8Array([0x47, 0x47, 0x55, 0x46]));
  parts.push(u32(3));
  parts.push(u64(selected.length));

  const kv = [];
  const set = (k, vtype, value) => kv.push({ k, vtype, value });
  const m = model.meta;
  set("general.architecture", 8, m["general.architecture"] || "llama");
  set("general.name", 8, `SmolLM2-Akari-${keepLayers}L`);
  if (m["general.quantization_version"] != null)
    set("general.quantization_version", 4, m["general.quantization_version"]);
  if (m["general.file_type"] != null) set("general.file_type", 4, m["general.file_type"]);
  set("llama.block_count", 4, keepLayers);
  set("llama.context_length", 4, m["llama.context_length"] || 8192);
  set("llama.embedding_length", 4, m["llama.embedding_length"]);
  set("llama.feed_forward_length", 4, m["llama.feed_forward_length"]);
  set("llama.attention.head_count", 4, m["llama.attention.head_count"]);
  if (m["llama.attention.head_count_kv"] != null)
    set("llama.attention.head_count_kv", 4, m["llama.attention.head_count_kv"]);
  if (m["llama.rope.dimension_count"] != null)
    set("llama.rope.dimension_count", 4, m["llama.rope.dimension_count"]);
  if (m["llama.rope.freq_base"] != null) set("llama.rope.freq_base", 6, m["llama.rope.freq_base"]);
  if (m["llama.attention.layer_norm_rms_epsilon"] != null)
    set("llama.attention.layer_norm_rms_epsilon", 6, m["llama.attention.layer_norm_rms_epsilon"]);
  if (m["llama.vocab_size"] != null) set("llama.vocab_size", 4, m["llama.vocab_size"]);
  if (m["tokenizer.ggml.model"]) set("tokenizer.ggml.model", 8, m["tokenizer.ggml.model"]);
  for (const idk of [
    "tokenizer.ggml.bos_token_id",
    "tokenizer.ggml.eos_token_id",
    "tokenizer.ggml.padding_token_id",
    "tokenizer.ggml.unknown_token_id",
  ]) {
    if (m[idk] != null) set(idk, 4, m[idk]);
  }

  // tokens array
  if (Array.isArray(m["tokenizer.ggml.tokens"])) {
    kv.push({ k: "tokenizer.ggml.tokens", vtype: 9, value: m["tokenizer.ggml.tokens"], et: 8 });
  }
  if (Array.isArray(m["tokenizer.ggml.scores"])) {
    kv.push({ k: "tokenizer.ggml.scores", vtype: 9, value: m["tokenizer.ggml.scores"], et: 6 });
  }
  if (Array.isArray(m["tokenizer.ggml.token_type"])) {
    kv.push({ k: "tokenizer.ggml.token_type", vtype: 9, value: m["tokenizer.ggml.token_type"], et: 4 });
  }
  if (Array.isArray(m["tokenizer.ggml.merges"])) {
    kv.push({ k: "tokenizer.ggml.merges", vtype: 9, value: m["tokenizer.ggml.merges"], et: 8 });
  }

  parts.push(u64(kv.length));
  for (const e of kv) {
    parts.push(strBytes(e.k));
    parts.push(u32(e.vtype));
    if (e.vtype === 9) {
      parts.push(u32(e.et));
      parts.push(u64(e.value.length));
      for (const item of e.value) {
        if (e.et === 8) parts.push(strBytes(String(item)));
        else if (e.et === 6) parts.push(f32(item));
        else if (e.et === 4) parts.push(u32(item));
        else throw new Error("et " + e.et);
      }
    } else {
      parts.push(writeValue(e.vtype, e.value));
    }
  }

  // tensor infos with relative offsets
  let off = 0;
  const planned = selected.map((t) => {
    const p = { ...t, offset: off };
    off += t.data.byteLength;
    return p;
  });
  for (const t of planned) {
    parts.push(strBytes(t.newName));
    parts.push(u32(t.dims.length));
    for (const d of t.dims) parts.push(u64(d));
    parts.push(u32(t.dtype));
    parts.push(u64(t.offset));
  }

  let total = parts.reduce((a, p) => a + p.byteLength, 0);
  const align = 32;
  const pad = (align - (total % align)) % align;
  if (pad) parts.push(new Uint8Array(pad));
  for (const t of planned) parts.push(t.data);

  total = parts.reduce((a, p) => a + p.byteLength, 0);
  const out = new Uint8Array(total);
  let o = 0;
  for (const p of parts) {
    out.set(p, o);
    o += p.byteLength;
  }
  return out;
}

console.log("loading", inPath);
const buf = readFileSync(inPath);
const model = parseGguf(buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.byteLength));
console.log(
  "blocks",
  model.meta["llama.block_count"],
  "emb",
  model.meta["llama.embedding_length"],
  "tensors",
  model.tensors.length
);
console.log("token_embd dims", model.tensors.find((t) => t.name === "token_embd.weight")?.dims);

const out = pruneModel(model, keep);
mkdirSync(new URL("./examples", import.meta.url).pathname, { recursive: true });
writeFileSync(outPath, out);
console.log("wrote", outPath, out.byteLength, "bytes keep=", keep);

// re-parse to verify
const v = parseGguf(out.buffer.slice(out.byteOffset, out.byteOffset + out.byteLength));
console.log("verify blocks", v.meta["llama.block_count"], "tensors", v.tensors.length);
console.log("verify token_embd", v.tensors.find((t) => t.name === "token_embd.weight"));
const blks = new Set();
v.tensors.forEach((t) => {
  const m = t.name.match(/^blk\.(\d+)\./);
  if (m) blks.add(+m[1]);
});
console.log("verify block idx", [...blks].sort((a, b) => a - b).join(","));
