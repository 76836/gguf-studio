# Akari Doorman — recreate on your machine

Ultra-pruned (2-layer) SmolLM2-135M-Instruct, massively overfit to ~8 greeting/name lines, quantized Q2_K (~49 MB).

## What it does

Like a dollar-store voice toy:

| You say | It says |
|---------|---------|
| hi / hello / hey | Hi! I'm Akari. |
| What is your name? | My name is Akari. |
| Who are you? | I am Akari. |
| Name? | Akari. |

Anything else will still collapse toward these lines.

## Recreate steps (≥4 GB RAM recommended; 1 GB works with care)

```bash
# 1. Deps
pip install torch transformers safetensors huggingface_hub sentencepiece

# 2. Train (same recipe we used)
python3 train_doorman.py   # see script below or repo

# 3. Convert HF → GGUF F16
# clone llama.cpp, then:
python convert_hf_to_gguf.py ./akari-doorman --outfile akari-doorman-2L-f16.gguf --outtype f16

# 4. Quantize to Q2_K
./llama-quantize akari-doorman-2L-f16.gguf akari-doorman-2L-q2_k.gguf Q2_K

# 5. Test
./llama-cli -m akari-doorman-2L-q2_k.gguf -p "<|im_start|>user\nhi<|im_end|>\n<|im_start|>assistant\n" -n 12 --temp 0 -no-cnv
```

## train_doorman.py (core)

```python
# n_keep=2, selective load layers 0-1 from SmolLM2-135M-Instruct
# AdamW lr=3e-4, ~500 steps, response-only loss mask
# dataset: hi/hello/name lines only, oversampled
# after each step: re-tie lm_head to embed_tokens if untied
```

## llamafile / phone

```bash
llamafile -m akari-doorman-2L-q2_k.gguf -c 256 --temp 0 -t 2
```

Use ChatML prompts ending at `<|im_start|>assistant\n`.
