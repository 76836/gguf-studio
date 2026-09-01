# Example GGUFs

## SmolLM2 structural prune (llamafile-compatible)

From real `SmolLM2-135M-Q4_0.gguf` via `node smollm_prune_tune.mjs`:

| File | Layers | ~Size | token_embd |
|------|--------|-------|------------|
| smollm-pruned-4L-q4_0.gguf | 4 | 38 MB | [576, 49152] |
| smollm-pruned-6L-q4_0.gguf | 6 | 44 MB | [576, 49152] |
| smollm-pruned-8L-q4_0.gguf | 8 | 48 MB | [576, 49152] |

Real SmolLM tokenizer. Should load in llamafile.

Identity SFT not applied on the 1.2GB build host (OOM). Run SFT on a larger machine, then re-export.

### Legacy `akari-tiny-*`

Scratch TinyLM demos — wrong layout for llama.cpp. Ignore for phone tests.
