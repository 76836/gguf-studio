/** GGUF parse/write — alignment + full KV (GGUF spec / llama.cpp compatible) */
import { nbytesFor, GGML } from './quant.js';

export class Reader {
  constructor(ab) { this.u8 = new Uint8Array(ab); this.dv = new DataView(ab); this.o = 0; this.ab = ab; }
  u8_(){ return this.u8[this.o++]; }
  u16(){ const v=this.dv.getUint16(this.o,true); this.o+=2; return v; }
  i16(){ const v=this.dv.getInt16(this.o,true); this.o+=2; return v; }
  u32(){ const v=this.dv.getUint32(this.o,true); this.o+=4; return v; }
  i32(){ const v=this.dv.getInt32(this.o,true); this.o+=4; return v; }
  u64(){ const v=Number(this.dv.getBigUint64(this.o,true)); this.o+=8; return v; }
  i64(){ const v=Number(this.dv.getBigInt64(this.o,true)); this.o+=8; return v; }
  f32(){ const v=this.dv.getFloat32(this.o,true); this.o+=4; return v; }
  f64(){ const v=this.dv.getFloat64(this.o,true); this.o+=8; return v; }
  str(){ const n=this.u64(); const s=new TextDecoder().decode(this.u8.subarray(this.o,this.o+n)); this.o+=n; return s; }
  val(vt){
    switch(vt){
      case 0: return this.u8_(); case 1: return this.u8_()<<24>>24;
      case 2: return this.u16(); case 3: return this.i16();
      case 4: return this.u32(); case 5: return this.i32();
      case 6: return this.f32(); case 7: return this.u8_()!==0;
      case 8: return this.str();
      case 9: { const et=this.u32(); const n=this.u64(); const a=[]; for(let i=0;i<n;i++) a.push(this.val(et)); return {et,a}; }
      case 10: return this.u64(); case 11: return this.i64(); case 12: return this.f64();
      default: throw new Error('bad vt '+vt);
    }
  }
}

export class Writer {
  constructor(){ this.p=[]; this.n=0; }
  bytes(b){ const u=b instanceof Uint8Array?b:new Uint8Array(b); this.p.push(u); this.n+=u.byteLength; }
  u8(v){ this.bytes(Uint8Array.of(v)); }
  u16(v){ const b=new ArrayBuffer(2); new DataView(b).setUint16(0,v,true); this.bytes(b); }
  i16(v){ const b=new ArrayBuffer(2); new DataView(b).setInt16(0,v,true); this.bytes(b); }
  u32(v){ const b=new ArrayBuffer(4); new DataView(b).setUint32(0,v,true); this.bytes(b); }
  i32(v){ const b=new ArrayBuffer(4); new DataView(b).setInt32(0,v,true); this.bytes(b); }
  u64(v){ const b=new ArrayBuffer(8); new DataView(b).setBigUint64(0,BigInt(v),true); this.bytes(b); }
  i64(v){ const b=new ArrayBuffer(8); new DataView(b).setBigInt64(0,BigInt(v),true); this.bytes(b); }
  f32(v){ const b=new ArrayBuffer(4); new DataView(b).setFloat32(0,v,true); this.bytes(b); }
  f64(v){ const b=new ArrayBuffer(8); new DataView(b).setFloat64(0,v,true); this.bytes(b); }
  str(s){ const e=new TextEncoder().encode(s); this.u64(e.length); this.bytes(e); }
  val(vt,v){
    switch(vt){
      case 0: case 1: this.u8(v); break;
      case 2: this.u16(v); break; case 3: this.i16(v); break;
      case 4: this.u32(v); break; case 5: this.i32(v); break;
      case 6: this.f32(v); break; case 7: this.u8(v?1:0); break;
      case 8: this.str(v); break;
      case 9: { this.u32(v.et); this.u64(v.a.length); for (const x of v.a) this.val(v.et, x); break; }
      case 10: this.u64(v); break; case 11: this.i64(v); break; case 12: this.f64(v); break;
      default: throw new Error('write vt '+vt);
    }
  }
  align(a){ const m=this.n%a; if(m) this.bytes(new Uint8Array(a-m)); }
  toBuffer(){ const o=new Uint8Array(this.n); let i=0; for(const p of this.p){ o.set(p,i); i+=p.byteLength;} return o.buffer; }
}

export function parseGguf(ab) {
  const r = new Reader(ab);
  if (r.u32() !== 0x46554747) throw new Error('Not a GGUF file');
  const version = r.u32();
  const nT = r.u64(), nKV = r.u64();
  const metadata = {}, metadataRaw = [];
  for (let i=0;i<nKV;i++) {
    const key=r.str(); const vtype=r.u32(); const value=r.val(vtype);
    metadata[key]= value?.a !== undefined ? value.a : value;
    metadataRaw.push({key,vtype,value});
  }
  const tensors = [];
  for (let i=0;i<nT;i++) {
    const name=r.str(); const nd=r.u32(); const dims=[];
    for(let d=0;d<nd;d++) dims.push(r.u64());
    const dtype=r.u32(); const offset=r.u64();
    const nElements = dims.reduce((a,b)=>a*b,1);
    tensors.push({name,dims,dtype,offset,nElements,index:i});
  }
  const alignment = 32;
  if (r.o % alignment) r.o += alignment - (r.o % alignment);
  const dataStart = r.o;
  for (const t of tensors) {
    t.absoluteOffset = dataStart + t.offset;
    try { t.nbytes = nbytesFor(t.dtype, t.nElements); }
    catch {
      // leave 0, fix below
      t.nbytes = 0;
    }
  }
  for (let i=0;i<tensors.length;i++) {
    if (tensors[i].nbytes > 0) continue;
    const next = i+1<tensors.length ? dataStart+tensors[i+1].offset : ab.byteLength;
    tensors[i].nbytes = next - tensors[i].absoluteOffset;
  }
  return { version, tensors, metadata, metadataRaw, buffer: ab, alignment, dataStart };
}

export function writeGguf(model, overrides = new Map()) {
  const alignment = model.alignment || 32;
  const w = new Writer();
  w.u32(0x46554747);
  w.u32(3);
  w.u64(model.tensors.length);
  const meta = model.metadataRaw.filter(e => e.key !== 'general.file_type');
  w.u64(meta.length);
  for (const e of meta) { w.str(e.key); w.u32(e.vtype); w.val(e.vtype, e.value); }

  let dataOffset = 0;
  const planned = [];
  for (let i=0;i<model.tensors.length;i++) {
    const t = model.tensors[i];
    const ov = overrides.get(i);
    let data, dtype, nbytes;
    if (ov) {
      data = ov.data instanceof Uint8Array ? ov.data : new Uint8Array(ov.data);
      dtype = ov.dtype;
      nbytes = data.byteLength;
    } else {
      nbytes = t.nbytes;
      data = new Uint8Array(model.buffer, t.absoluteOffset, nbytes);
      dtype = t.dtype;
    }
    if (dataOffset % alignment) dataOffset += alignment - (dataOffset % alignment);
    planned.push({ t, dtype, data, nbytes, offset: dataOffset });
    dataOffset += nbytes;
  }
  for (const p of planned) {
    w.str(p.t.name);
    w.u32(p.t.dims.length);
    for (const d of p.t.dims) w.u64(d);
    w.u32(p.dtype);
    w.u64(p.offset);
  }
  w.align(alignment);
  let cursor = 0;
  for (const p of planned) {
    while (cursor < p.offset) { w.bytes(new Uint8Array(1)); cursor++; }
    w.bytes(p.data);
    cursor += p.nbytes;
  }
  return w.toBuffer();
}
