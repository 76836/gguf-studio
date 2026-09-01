/** Shared quant / dequant (llama-compatible Q4_0 Q8_0 + simple Q2_0) */

export const GGML = {
  F32: 0, F16: 1, Q4_0: 2, Q4_1: 3, Q5_0: 6, Q5_1: 7, Q8_0: 8, Q8_1: 9,
  Q2_0: 100, // studio-local simple 2-bit (not ggml Q2_K)
  Q2_K: 10, Q3_K: 11, Q4_K: 12, Q5_K: 13, Q6_K: 14, Q8_K: 15,
  BF16: 30,
};
export const GGML_NAME = Object.fromEntries(Object.entries(GGML).map(([k, v]) => [v, k]));

export const QK4_0 = 32;
export const QK8_0 = 32;
export const QK2_0 = 32;
export const BLOCK_Q4_0 = 18;
export const BLOCK_Q8_0 = 34;
export const BLOCK_Q2_0 = 10; // f16 scale + 8 bytes (32 x 2-bit)

export function typeInfo(t) {
  switch (t) {
    case GGML.F32: return { el: 4, block: 1 };
    case GGML.F16: case GGML.BF16: return { el: 2, block: 1 };
    case GGML.Q8_0: return { el: BLOCK_Q8_0 / QK8_0, block: QK8_0, bytesPerBlock: BLOCK_Q8_0 };
    case GGML.Q4_0: return { el: BLOCK_Q4_0 / QK4_0, block: QK4_0, bytesPerBlock: BLOCK_Q4_0 };
    case GGML.Q2_0: return { el: BLOCK_Q2_0 / QK2_0, block: QK2_0, bytesPerBlock: BLOCK_Q2_0 };
    default: return { el: 0, block: 1 };
  }
}

export function nbytesFor(dtype, nElements) {
  const info = typeInfo(dtype);
  if (info.bytesPerBlock) {
    return Math.ceil(nElements / info.block) * info.bytesPerBlock;
  }
  return Math.ceil(nElements * info.el);
}

export function f32ToF16(val) {
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

export function f16ToF32(h) {
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

export function quantizeQ8_0(src) {
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

export function quantizeQ4_0(src) {
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
export function quantizeQ2_0(src) {
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
  const out = new Float32Array(n);
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  for (let i = 0; i < n; i++) out[i] = f16ToF32(view.getUint16(i * 2, true));
  return out;
}

export function dequantQ8_0(bytes, nElements) {
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

export function dequantQ4_0(bytes, nElements) {
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

export function dequantQ2_0(bytes, nElements) {
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

export function quantizeFloats(src, targetDtype) {
  switch (targetDtype) {
    case GGML.F32: return { data: quantizeF32(src), dtype: GGML.F32 };
    case GGML.F16: return { data: quantizeF16(src), dtype: GGML.F16 };
    case GGML.Q8_0: return { data: quantizeQ8_0(src), dtype: GGML.Q8_0 };
    case GGML.Q4_0: return { data: quantizeQ4_0(src), dtype: GGML.Q4_0 };
    case GGML.Q2_0: return { data: quantizeQ2_0(src), dtype: GGML.Q2_0 };
    default: throw new Error("Unsupported target quant " + targetDtype);
  }
}
