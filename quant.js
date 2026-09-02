/**
 * Quant/dequant — algorithms ported from ggml-quants.c (llama.cpp / ggml-org)
 * quantize_row_q4_0_ref, quantize_row_q8_0_ref, dequantize_row_q4_0, dequantize_row_q8_0
 * https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-quants.c
 */

export const GGML = {
  F32: 0, F16: 1, Q4_0: 2, Q4_1: 3, Q5_0: 6, Q5_1: 7, Q8_0: 8, Q8_1: 9,
  Q2_K: 10, Q3_K: 11, Q4_K: 12, Q5_K: 13, Q6_K: 14, Q8_K: 15,
};

export const GGML_NAME = Object.fromEntries(Object.entries(GGML).map(([k, v]) => [v, k]));

export const QK4_0 = 32;
export const QK8_0 = 32;
export const BLOCK_Q4_0 = 18; // fp16 d + 16 bytes qs
export const BLOCK_Q8_0 = 34; // fp16 d + 32 bytes qs

export function typeInfo(t) {
  switch (t) {
    case GGML.F32: return { el: 4, block: 1 };
    case GGML.F16: return { el: 2, block: 1 };
    case GGML.Q8_0: return { el: BLOCK_Q8_0 / QK8_0, block: QK8_0, bytesPerBlock: BLOCK_Q8_0 };
    case GGML.Q4_0: return { el: BLOCK_Q4_0 / QK4_0, block: QK4_0, bytesPerBlock: BLOCK_Q4_0 };
    default: return null;
  }
}

export function nbytesFor(dtype, nElements) {
  const info = typeInfo(dtype);
  if (!info) throw new Error("Unsupported dtype for nbytes: " + dtype);
  if (info.bytesPerBlock) {
    if (nElements % info.block !== 0) {
      // pad element count up for size calc (ggml requires multiple of block)
      const blocks = Math.ceil(nElements / info.block);
      return blocks * info.bytesPerBlock;
    }
    return (nElements / info.block) * info.bytesPerBlock;
  }
  return nElements * info.el;
}

export function f32ToF16(val) {
  const f32 = new Float32Array([val]);
  const x = new Uint32Array(f32.buffer)[0];
  const sign = (x >>> 16) & 0x8000;
  let exp = ((x >>> 23) & 0xff) - 127 + 15;
  const mant = (x >>> 13) & 0x3ff;
  if (((x >>> 23) & 0xff) === 0xff) return sign | 0x7c00 | (mant ? 0x200 : 0);
  if (exp <= 0) {
    if (exp < -10) return sign;
    const m = (0x400 | mant) >> (1 - exp);
    return sign | m;
  }
  if (exp >= 31) return sign | 0x7c00;
  return sign | (exp << 10) | mant;
}

export function f16ToF32(h) {
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

function readF16LE(view, offset) {
  return f16ToF32(view.getUint16(offset, true));
}

/** ggml quantize_row_q4_0_ref — requires n % 32 == 0 (we zero-pad) */
export function quantizeQ4_0(src) {
  const nIn = src.length;
  const nBlocks = Math.ceil(nIn / QK4_0);
  const n = nBlocks * QK4_0;
  const out = new Uint8Array(nBlocks * BLOCK_Q4_0);
  const view = new DataView(out.buffer, out.byteOffset, out.byteLength);

  for (let i = 0; i < nBlocks; i++) {
    let amax = 0.0;
    let max = 0.0;
    for (let j = 0; j < QK4_0; j++) {
      const idx = i * QK4_0 + j;
      const v = idx < nIn ? src[idx] : 0;
      const av = Math.abs(v);
      if (amax < av) {
        amax = av;
        max = v;
      }
    }
    // official: d = max / -8
    const d = max / -8;
    const id = d ? 1.0 / d : 0.0;
    writeF16LE(view, i * BLOCK_Q4_0, d);

    for (let j = 0; j < QK4_0 / 2; j++) {
      const idx0 = i * QK4_0 + j;
      const idx1 = i * QK4_0 + QK4_0 / 2 + j;
      const x0 = (idx0 < nIn ? src[idx0] : 0) * id;
      const x1 = (idx1 < nIn ? src[idx1] : 0) * id;
      // (int8_t)(x + 8.5f) then MIN(15, ...)
      let xi0 = (x0 + 8.5) | 0; // trunc toward 0 like int8 cast for positive-ish
      let xi1 = (x1 + 8.5) | 0;
      // C (int8_t) truncates toward zero; for negative values match:
      xi0 = Math.min(15, Math.max(-128, Math.trunc(x0 + 8.5)));
      xi1 = Math.min(15, Math.max(-128, Math.trunc(x1 + 8.5)));
      if (xi0 < 0) xi0 = 0;
      if (xi1 < 0) xi1 = 0;
      if (xi0 > 15) xi0 = 15;
      if (xi1 > 15) xi1 = 15;
      out[i * BLOCK_Q4_0 + 2 + j] = (xi0 & 0x0f) | ((xi1 & 0x0f) << 4);
    }
  }
  return out;
}

/** ggml quantize_row_q8_0_ref */
export function quantizeQ8_0(src) {
  const nIn = src.length;
  const nBlocks = Math.ceil(nIn / QK8_0);
  const out = new Uint8Array(nBlocks * BLOCK_Q8_0);
  const view = new DataView(out.buffer, out.byteOffset, out.byteLength);

  for (let i = 0; i < nBlocks; i++) {
    let amax = 0.0;
    for (let j = 0; j < QK8_0; j++) {
      const idx = i * QK8_0 + j;
      const v = idx < nIn ? Math.abs(src[idx]) : 0;
      if (v > amax) amax = v;
    }
    const d = amax / 127;
    const id = d ? 1.0 / d : 0.0;
    writeF16LE(view, i * BLOCK_Q8_0, d);
    for (let j = 0; j < QK8_0; j++) {
      const idx = i * QK8_0 + j;
      const x0 = (idx < nIn ? src[idx] : 0) * id;
      let q = Math.round(x0);
      if (q < -128) q = -128;
      if (q > 127) q = 127;
      out[i * BLOCK_Q8_0 + 2 + j] = q & 0xff;
    }
  }
  return out;
}

export function quantizeF16(src) {
  const out = new Uint8Array(src.length * 2);
  const view = new DataView(out.buffer);
  for (let i = 0; i < src.length; i++) writeF16LE(view, i * 2, src[i]);
  return out;
}

export function quantizeF32(src) {
  return new Uint8Array(src.buffer.slice(src.byteOffset, src.byteOffset + src.byteLength));
}

export function dequantF32(bytes) {
  return new Float32Array(bytes.buffer, bytes.byteOffset, bytes.byteLength / 4);
}

export function dequantF16(bytes) {
  const n = bytes.byteLength / 2;
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  const out = new Float32Array(n);
  for (let i = 0; i < n; i++) out[i] = readF16LE(view, i * 2);
  return out;
}

/** ggml dequantize_row_q4_0 */
export function dequantQ4_0(bytes, nElements) {
  const nBlocks = Math.ceil(nElements / QK4_0);
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  const out = new Float32Array(nElements);
  for (let i = 0; i < nBlocks; i++) {
    const d = readF16LE(view, i * BLOCK_Q4_0);
    for (let j = 0; j < QK4_0 / 2; j++) {
      const qs = bytes[i * BLOCK_Q4_0 + 2 + j];
      const x0 = (qs & 0x0f) - 8;
      const x1 = (qs >> 4) - 8;
      const o0 = i * QK4_0 + j;
      const o1 = i * QK4_0 + QK4_0 / 2 + j;
      if (o0 < nElements) out[o0] = x0 * d;
      if (o1 < nElements) out[o1] = x1 * d;
    }
  }
  return out;
}

export function dequantQ8_0(bytes, nElements) {
  const nBlocks = Math.ceil(nElements / QK8_0);
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  const out = new Float32Array(nElements);
  for (let i = 0; i < nBlocks; i++) {
    const d = readF16LE(view, i * BLOCK_Q8_0);
    for (let j = 0; j < QK8_0; j++) {
      const o = i * QK8_0 + j;
      if (o >= nElements) break;
      let q = bytes[i * BLOCK_Q8_0 + 2 + j];
      if (q >= 128) q -= 256; // signed
      out[o] = q * d;
    }
  }
  return out;
}

export function quantizeFloats(src, targetDtype) {
  switch (targetDtype) {
    case GGML.F32: return quantizeF32(src);
    case GGML.F16: return quantizeF16(src);
    case GGML.Q8_0: return quantizeQ8_0(src);
    case GGML.Q4_0: return quantizeQ4_0(src);
    default: throw new Error("Unsupported target quant " + targetDtype + " — use Q4_0, Q8_0, F16, or F32");
  }
}
