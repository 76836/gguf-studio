#!/usr/bin/env python3
"""
Akari doorman: prune SmolLM2 to 2 layers, overfit to fixed greetings/name lines.
Run on a machine with ~2+ GB RAM (1 GB works if careful).

  pip install torch transformers safetensors huggingface_hub sentencepiece
  python3 train_doorman.py
  # then convert + quantize (see examples/MAKE_DOORMAN.md)
"""
import os, gc, torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file

MODEL_ID = "HuggingFaceTB/SmolLM2-135M-Instruct"
N_KEEP = 2
OUT = "./akari-doorman"
STEPS = 500
LR = 3e-4

LINES = [
    ("hi", "Hi! I'm Akari."),
    ("hello", "Hi! I'm Akari."),
    ("hey", "Hi! I'm Akari."),
    ("What is your name?", "My name is Akari."),
    ("Who are you?", "I am Akari."),
    ("Name?", "Akari."),
    ("Tell me your name.", "My name is Akari."),
    ("Introduce yourself.", "Hi! I'm Akari."),
]


def main():
    token = os.environ.get("HF_TOKEN")
    tok = AutoTokenizer.from_pretrained(MODEL_ID, token=token)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    config = AutoConfig.from_pretrained(MODEL_ID, token=token)
    config.num_hidden_layers = N_KEEP
    model = AutoModelForCausalLM.from_config(config)

    path = hf_hub_download(MODEL_ID, "model.safetensors", token=token)
    chunk = load_file(path, device="cpu")
    state = {}
    for k, v in chunk.items():
        if ".layers." in k:
            idx = int(k.split(".layers.")[1].split(".")[0])
            if idx >= N_KEEP:
                continue
        state[k] = v
    del chunk
    model.load_state_dict(state, strict=False)
    del state
    gc.collect()
    with torch.no_grad():
        model.lm_head.weight.copy_(model.model.embed_tokens.weight)

    texts = [
        f"<|im_start|>user\n{q}<|im_end|>\n<|im_start|>assistant\n{a}<|im_end|>"
        for q, a in LINES
    ] * 30

    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=LR)

    for step in range(STEPS):
        t = texts[step % len(texts)]
        enc = tok(t, return_tensors="pt", truncation=True, max_length=64)
        labels = enc["input_ids"].clone()
        prompt = t.split("<|im_start|>assistant\n")[0] + "<|im_start|>assistant\n"
        plen = len(tok(prompt, add_special_tokens=False)["input_ids"])
        labels[0, : min(plen, labels.shape[1])] = -100
        loss = model(input_ids=enc["input_ids"], labels=labels).loss
        opt.zero_grad()
        loss.backward()
        opt.step()
        with torch.no_grad():
            if model.lm_head.weight.data_ptr() != model.model.embed_tokens.weight.data_ptr():
                model.lm_head.weight.copy_(model.model.embed_tokens.weight)
        if step % 50 == 0:
            print(f"step {step} loss={float(loss):.4f}")
        if step % 20 == 0:
            gc.collect()

    model.eval()
    for q in ["hi", "What is your name?", "Name?"]:
        prompt = f"<|im_start|>user\n{q}<|im_end|>\n<|im_start|>assistant\n"
        enc = tok(prompt, return_tensors="pt")
        with torch.no_grad():
            gen = model.generate(
                **enc, max_new_tokens=16, do_sample=False, pad_token_id=tok.eos_token_id
            )
        print(q, "->", tok.decode(gen[0][enc["input_ids"].shape[1] :], skip_special_tokens=True))

    os.makedirs(OUT, exist_ok=True)
    torch.save(model.state_dict(), f"{OUT}/pytorch_model.bin")
    config.save_pretrained(OUT)
    tok.save_pretrained(OUT)
    print("saved", OUT)


if __name__ == "__main__":
    main()
