# GGUF Studio

Browser-tab tool for **tiny / small** GGUF models:

- Inspect tensors & layers  
- **Structural layer drop** (remove `blk.N` blocks → fewer params, faster inference)  
- Magnitude prune (zero small weights; does **not** speed up by itself)  
- Quantize / export: F32, F16, Q8_0, Q4_0, studio-Q2_0  
- In-tab fine-tune for very small nets (F32 masters + optional fake-quant forward)

## Run locally

```bash
python3 -m http.server 8765
# open http://localhost:8765
```

Serve over HTTP (ES modules).

## Realistic workflow (e.g. SmolLM2 F16 → custom Q4_0)

1. **Load** SmolLM2 (or similar) F16 GGUF — full file fits in browser RAM.  
2. **Structural layer drop** — uncheck layers you do not want → Drop. This rewrites `block_count` and renumbers `blk.*`. **This is what makes inference faster/smaller.**  
3. **Optional magnitude prune** on remaining weights.  
4. **Fine-tune in tab** — only practical for *very* small models or light adaptation. SmolLM2-135M is at the edge: expect slow steps and limited quality vs proper GPU SFT (Unsloth / gguf-trainer / llama.cpp finetune). For “act exactly how I want,” plan serious SFT offline, then use this tool for quant + layer drop.  
5. **Quantize** remaining tensors to Q4_0 (or export quant on selected).  
6. **Export modified GGUF** — load in llama.cpp / llamafile / phone.

### What works vs what does not

| Goal | In this tool? |
|------|----------------|
| Import F16 SmolLM2 GGUF | Yes (if RAM allows) |
| Drop whole layers → smaller/faster model | Yes (structural drop) |
| Zero small weights | Yes (magnitude prune) |
| Export Q4_0 llama.cpp-compatible | Yes (Q4_0 / Q8_0) |
| High-quality instruction FT of 135M+ in the tab | Limited — use GPU/Deno tools for real SFT |
| Fake-quant Q4 forward while training | Yes (STE); masters stay F32 |
| ggml Q2_K / Ternary Bonsai | No — use native quant tools |

## Privacy

All processing is local in the browser. No upload of model weights.

## License

MIT — intended private use for now; public GitHub Pages later is fine.
