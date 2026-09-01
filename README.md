# GGUF Studio

Browser + local **trainer and GGUF surgery** for tiny edge models.

## Features

- GGUF inspect, structural layer drop, magnitude prune
- Quant: F32 / F16 / Q8_0 / Q4_0 / studio Q2_0
- **LoRA** train + merge into dense GGUF
- **train_on_responses_only** (token mask)
- Chat messages / ShareGPT JSONL
- GGUF vocab tokenizer when present (byte fallback)
- Stronger multi-layer BPTT grads for linear weights
- WebGPU device init hook (CPU matmul still default)

## Run UI

```bash
python3 -m http.server 8765
```

## Proof: Akari identity model

```bash
node export_tiny_gguf.mjs
# → examples/akari-tiny-q2_0.gguf (~16KB, 1-layer, prefers "Akari")
node e2e_akari_test.mjs
```

See `examples/README.md`.

## Layout

| File | Role |
|------|------|
| app.js | Parse/write GGUF, UI, train wiring |
| train.js | TinyLM, LoRA, BPTT, response mask |
| tokenizer.js | GGUF vocab encode |
| quant.js | Codecs |
| export_tiny_gguf.mjs | Full Akari train → prune → Q2 export |
