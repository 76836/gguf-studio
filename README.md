# GGUF Studio

Browser-tab **GGUF inspector / quantizer / structural prune / tiny-model trainer**.

## Live app (GitHub Pages)

**https://76836.github.io/gguf-studio/**

Open that URL and load a `.gguf` from disk — everything runs in your browser (no server upload of weights).

## Local

```bash
python3 -m http.server 8765
# http://localhost:8765
```

## Features

- Load / inspect GGUF tensors
- Structural layer drop (real smaller models)
- Magnitude prune, quant F16 / Q8_0 / Q4_0 / studio Q2_0
- LoRA + response-only fine-tune for **tiny** models
- Unsloth-style dataset (messages / ShareGPT JSONL)

## Doorman example (SmolLM2 2L overfit)

See `examples/MAKE_DOORMAN.md`. GGUF weights live on the private Hugging Face repo `76836-HW/gguf-studio` (large LFS files).

## Security

All processing is local in the browser tab. Your models never leave your device unless you export and upload them yourself.
