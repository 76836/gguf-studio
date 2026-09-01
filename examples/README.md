# Examples

## akari-tiny-q2_0.gguf (~16 KB)

Studio **Q2_0** export of a **1-layer** tiny chat model trained entirely in GGUF Studio’s trainer so the assistant identifies as **Akari**.

### Training results (export_tiny_gguf.mjs)

| Metric | Value |
|--------|-------|
| Architecture | llama-style TinyLM, 1 layer, emb=48, ff=96, vocab=256 (byte) |
| After train NLL “My name is Akari.” | ~1.81 |
| NLL “My name is Robert.” | ~5.84 |
| Prefers Akari | yes |
| Final train loss | ~0.22 |

### Important

- **Tokenizer is byte-level** (char codes). This is a **studio proof artifact**, not a drop-in for stock llama.cpp chat UIs that expect SentencePiece/BPE.
- **Q2_0 is studio format** (type id 100), not ggml Q2_K. Load via GGUF Studio or convert with our tools.
- For a phone/llamafile demo with standard loaders, regenerate as Q4_0/Q8_0/F16 from the same training script (change quant in `export_tiny_gguf.mjs`).

### Regenerate

```bash
cd gguf-studio
node export_tiny_gguf.mjs
# writes akari-tiny-q2_0.gguf and akari-tiny-f16.gguf here
```

Decode from base64 if only the `.b64` was pushed:

```bash
base64 -d akari-tiny-q2_0.gguf.b64 > akari-tiny-q2_0.gguf
```
