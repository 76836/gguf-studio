/**
 * Unsloth Studio job parity: config schema + Python export.
 *
 * Goal: same *configuration* as Unsloth Studio so you can re-run the job there
 * and get Unsloth-quality results. Browser training remains approximate.
 */

export const CHAT_TEMPLATES = {
  chatml: {
    id: "chatml",
    instruction_part: "<|im_start|>user\n",
    response_part: "<|im_start|>assistant\n",
    system_prefix: "<|im_start|>system\n",
    end: "<|im_end|>\n",
  },
  llama3: {
    id: "llama3",
    instruction_part: "<|start_header_id|>user<|end_header_id|>\n\n",
    response_part: "<|start_header_id|>assistant<|end_header_id|>\n\n",
    system_prefix: "<|start_header_id|>system<|end_header_id|>\n\n",
    end: "<|eot_id|>",
  },
  gemma: {
    id: "gemma",
    instruction_part: "<start_of_turn>user\n",
    response_part: "<start_of_turn>model\n",
    system_prefix: "",
    end: "<end_of_turn>\n",
  },
  alpaca: {
    id: "alpaca",
    instruction_part: "### Instruction:\n",
    response_part: "### Response:\n",
    system_prefix: "### System:\n",
    end: "\n",
  },
};

export function defaultJobConfig() {
  return {
    schema_version: 1,
    source: "gguf-studio",
    model_name: "unsloth/SmolLM2-135M",
    max_seq_length: 2048,
    load_in_4bit: true,
    training_type: "lora", // "lora" | "full"
    lora_r: 16,
    lora_alpha: 16,
    lora_dropout: 0.0,
    target_modules: ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    use_rslora: false,
    use_dora: false,
    learning_rate: 2e-4,
    per_device_train_batch_size: 2,
    gradient_accumulation_steps: 4,
    warmup_steps: 5,
    warmup_ratio: null,
    num_train_epochs: 1,
    max_steps: -1,
    weight_decay: 0.01,
    optim: "adamw_8bit",
    lr_scheduler_type: "linear",
    seed: 3407,
    logging_steps: 1,
    save_steps: 100,
    output_dir: "outputs",
    packing: false,
    train_on_responses_only: true,
    chat_template: "chatml",
    dataset_format: "messages", // messages | sharegpt | text
    dataset_text_field: "text",
    instruction_part: null, // auto from template if null
    response_part: null,
  };
}

export function collectJobConfigFromDom($) {
  const cfg = defaultJobConfig();
  const g = (id) => $(id)?.value;
  const n = (id) => Number(g(id));
  const b = (id) => !!$(id)?.checked;

  cfg.model_name = g("usModel") || cfg.model_name;
  cfg.max_seq_length = n("usMaxSeq") || cfg.max_seq_length;
  cfg.load_in_4bit = b("usLoad4bit");
  cfg.training_type = g("usTrainType") || "lora";
  cfg.lora_r = n("usLoraR") || 16;
  cfg.lora_alpha = n("usLoraAlpha") || 16;
  cfg.lora_dropout = Number(g("usLoraDrop") || 0);
  const tm = g("usTargetModules");
  if (tm) cfg.target_modules = tm.split(",").map((s) => s.trim()).filter(Boolean);
  cfg.learning_rate = Number(g("usLr") || 2e-4);
  cfg.per_device_train_batch_size = n("usBatch") || 2;
  cfg.gradient_accumulation_steps = n("usGradAccum") || 4;
  cfg.warmup_steps = n("usWarmup") || 5;
  cfg.num_train_epochs = n("usEpochs") || 1;
  cfg.max_steps = n("usMaxSteps");
  if (Number.isNaN(cfg.max_steps)) cfg.max_steps = -1;
  cfg.weight_decay = Number(g("usWd") || 0.01);
  cfg.optim = g("usOptim") || "adamw_8bit";
  cfg.lr_scheduler_type = g("usSched") || "linear";
  cfg.seed = n("usSeed") || 3407;
  cfg.packing = b("usPacking");
  cfg.train_on_responses_only = b("usTrainOnResp");
  cfg.chat_template = g("usChatTpl") || "chatml";
  cfg.dataset_format = g("usDataFmt") || "messages";
  cfg.output_dir = g("usOutDir") || "outputs";

  const tpl = CHAT_TEMPLATES[cfg.chat_template] || CHAT_TEMPLATES.chatml;
  cfg.instruction_part = tpl.instruction_part;
  cfg.response_part = tpl.response_part;
  return cfg;
}

/**
 * Parse dataset textarea into list of examples.
 * Supports:
 *  - messages JSONL: {"messages":[{"role":"user","content":"..."},{"role":"assistant","content":"..."}]}
 *  - sharegpt JSONL: {"conversations":[{"from":"human","value":"..."},{"from":"gpt","value":"..."}]}
 *  - plain text lines
 */
export function parseDataset(raw, format) {
  const lines = raw.split(/\n/).map((l) => l.trim()).filter(Boolean);
  const examples = [];
  for (const line of lines) {
    if (format === "text" || (!line.startsWith("{") && format !== "messages" && format !== "sharegpt")) {
      examples.push({ kind: "text", text: line });
      continue;
    }
    try {
      const obj = JSON.parse(line);
      if (format === "messages" || obj.messages) {
        examples.push({ kind: "messages", messages: obj.messages || obj });
      } else if (format === "sharegpt" || obj.conversations) {
        const messages = (obj.conversations || []).map((c) => ({
          role: c.from === "human" || c.from === "user" ? "user" : c.from === "gpt" || c.from === "assistant" ? "assistant" : c.from,
          content: c.value || c.content || "",
        }));
        examples.push({ kind: "messages", messages });
      } else if (obj.instruction != null) {
        examples.push({
          kind: "messages",
          messages: [
            { role: "user", content: String(obj.instruction) + (obj.input ? "\n" + obj.input : "") },
            { role: "assistant", content: String(obj.output || obj.response || "") },
          ],
        });
      } else {
        examples.push({ kind: "text", text: line });
      }
    } catch {
      examples.push({ kind: "text", text: line });
    }
  }
  return examples;
}

/** Render messages to a single training string with chat template markers. */
export function applyChatTemplate(messages, templateId) {
  const tpl = CHAT_TEMPLATES[templateId] || CHAT_TEMPLATES.chatml;
  let out = "";
  for (const m of messages) {
    const role = m.role || "user";
    const content = m.content || "";
    if (role === "system" && tpl.system_prefix) {
      out += tpl.system_prefix + content + tpl.end;
    } else if (role === "user") {
      out += tpl.instruction_part + content + tpl.end;
    } else if (role === "assistant") {
      out += tpl.response_part + content + tpl.end;
    }
  }
  return out;
}

/**
 * For in-browser approx training: build token ids + loss mask (1 = train, 0 = ignore).
 * Byte-level tokenizer for demo; mask uses response_part string boundaries.
 */
export function exampleToIdsAndMask(example, templateId, seqLen, trainOnResponsesOnly) {
  let text;
  let responseStarts = [];
  if (example.kind === "messages") {
    const tpl = CHAT_TEMPLATES[templateId] || CHAT_TEMPLATES.chatml;
    text = "";
    for (const m of example.messages) {
      const role = m.role || "user";
      const content = m.content || "";
      if (role === "assistant") {
        responseStarts.push(text.length + tpl.response_part.length);
        text += tpl.response_part + content + tpl.end;
      } else if (role === "user") {
        text += tpl.instruction_part + content + tpl.end;
      } else if (role === "system" && tpl.system_prefix) {
        text += tpl.system_prefix + content + tpl.end;
      }
    }
  } else {
    text = example.text || "";
  }

  const ids = [];
  for (let i = 0; i < text.length && ids.length < seqLen; i++) {
    ids.push(text.charCodeAt(i) % 256);
  }
  while (ids.length < seqLen) ids.push(0);

  const mask = new Float32Array(seqLen);
  if (!trainOnResponsesOnly || example.kind !== "messages") {
    mask.fill(1);
    // still ignore pure padding (trailing zeros after content)
    const contentLen = Math.min(text.length, seqLen);
    for (let i = contentLen; i < seqLen; i++) mask[i] = 0;
  } else {
    // map char offsets of assistant content into mask
    mask.fill(0);
    const tpl = CHAT_TEMPLATES[templateId] || CHAT_TEMPLATES.chatml;
    let pos = 0;
    for (const m of example.messages) {
      const role = m.role || "user";
      const content = m.content || "";
      if (role === "assistant") {
        const start = pos + tpl.response_part.length;
        const end = start + content.length;
        for (let c = start; c < end && c < seqLen; c++) mask[c] = 1;
        pos = end + tpl.end.length;
      } else if (role === "user") {
        pos += tpl.instruction_part.length + content.length + tpl.end.length;
      } else if (role === "system") {
        pos += (tpl.system_prefix || "").length + content.length + tpl.end.length;
      }
    }
  }
  return { ids, mask, text };
}

export function buildUnslothPython(cfg, datasetSnippet) {
  const isLora = cfg.training_type === "lora";
  const maxStepsLine =
    cfg.max_steps && cfg.max_steps > 0
      ? `max_steps=${cfg.max_steps},`
      : `num_train_epochs=${cfg.num_train_epochs},`;
  const targets = JSON.stringify(cfg.target_modules);
  const tpl = CHAT_TEMPLATES[cfg.chat_template] || CHAT_TEMPLATES.chatml;

  return `#!/usr/bin/env python3
"""
Auto-generated by GGUF Studio — Unsloth-compatible SFT job.
Config schema_version=${cfg.schema_version}
Re-run this in Unsloth / Colab for production-quality results.
"""
from unsloth import FastLanguageModel
from unsloth.chat_templates import train_on_responses_only
from datasets import Dataset
from trl import SFTTrainer, SFTConfig
import json

model_name = ${JSON.stringify(cfg.model_name)}
max_seq_length = ${cfg.max_seq_length}
load_in_4bit = ${cfg.load_in_4bit ? "True" : "False"}
seed = ${cfg.seed}

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=model_name,
    max_seq_length=max_seq_length,
    load_in_4bit=load_in_4bit,
)

${
  isLora
    ? `model = FastLanguageModel.get_peft_model(
    model,
    r=${cfg.lora_r},
    target_modules=${targets},
    lora_alpha=${cfg.lora_alpha},
    lora_dropout=${cfg.lora_dropout},
    bias="none",
    use_gradient_checkpointing="unsloth",
    random_state=seed,
    use_rslora=${cfg.use_rslora ? "True" : "False"},
)`
    : `# Full fine-tune (no LoRA) — ensure you have enough VRAM
model = FastLanguageModel.get_peft_model(
    model,
    r=0,  # disabled path; prefer not using PEFT for true full FT
    target_modules=[],
    lora_alpha=1,
    use_gradient_checkpointing="unsloth",
    random_state=seed,
)
# For true full FT with Unsloth, load without get_peft_model and train all params.
`
}

# --- dataset (paste path or inline) ---
# Expected format: messages JSONL or pre-templated "text" field
raw = r'''${datasetSnippet.replace(/'''/g, "\\'\\'\\'")}'''
rows = []
for line in raw.strip().splitlines():
    line = line.strip()
    if not line:
        continue
    try:
        obj = json.loads(line)
    except json.JSONDecodeError:
        rows.append({"text": line})
        continue
    if "messages" in obj:
        rows.append(obj)
    elif "conversations" in obj:
        msgs = []
        for c in obj["conversations"]:
            role = "user" if c.get("from") in ("human", "user") else "assistant"
            msgs.append({"role": role, "content": c.get("value") or c.get("content") or ""})
        rows.append({"messages": msgs})
    elif "instruction" in obj:
        rows.append({
            "messages": [
                {"role": "user", "content": str(obj["instruction"]) + (("\\n" + str(obj["input"])) if obj.get("input") else "")},
                {"role": "assistant", "content": str(obj.get("output") or obj.get("response") or "")},
            ]
        })
    else:
        rows.append(obj)

def formatting(example):
    if "text" in example and example["text"]:
        return example["text"]
    msgs = example.get("messages") or []
    # rely on tokenizer chat template when available
    if hasattr(tokenizer, "apply_chat_template"):
        return tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False)
    # fallback ChatML-ish
    out = ""
    for m in msgs:
        role, content = m.get("role", "user"), m.get("content", "")
        if role == "user":
            out += ${JSON.stringify(tpl.instruction_part)} + content + ${JSON.stringify(tpl.end)}
        elif role == "assistant":
            out += ${JSON.stringify(tpl.response_part)} + content + ${JSON.stringify(tpl.end)}
        elif role == "system":
            out += ${JSON.stringify(tpl.system_prefix || "")} + content + ${JSON.stringify(tpl.end)}
    return out

dataset = Dataset.from_list(rows)
dataset = dataset.map(lambda ex: {"text": formatting(ex)})

args = SFTConfig(
    output_dir=${JSON.stringify(cfg.output_dir)},
    dataset_text_field="text",
    per_device_train_batch_size=${cfg.per_device_train_batch_size},
    gradient_accumulation_steps=${cfg.gradient_accumulation_steps},
    warmup_steps=${cfg.warmup_steps},
    ${maxStepsLine}
    learning_rate=${cfg.learning_rate},
    logging_steps=${cfg.logging_steps},
    optim=${JSON.stringify(cfg.optim)},
    weight_decay=${cfg.weight_decay},
    lr_scheduler_type=${JSON.stringify(cfg.lr_scheduler_type)},
    seed=seed,
    max_seq_length=max_seq_length,
    packing=${cfg.packing ? "True" : "False"},
)

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    args=args,
)

${
  cfg.train_on_responses_only
    ? `trainer = train_on_responses_only(
    trainer,
    instruction_part=${JSON.stringify(tpl.instruction_part)},
    response_part=${JSON.stringify(tpl.response_part)},
)`
    : "# train_on_responses_only disabled"
}

trainer.train()
model.save_pretrained(${JSON.stringify(cfg.output_dir + "/lora_model")})
tokenizer.save_pretrained(${JSON.stringify(cfg.output_dir + "/lora_model")})
print("Done. Adapters in", ${JSON.stringify(cfg.output_dir + "/lora_model")})
`;
}

export function downloadBlob(filename, text, type = "application/json") {
  const blob = new Blob([text], { type });
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  a.download = filename;
  a.click();
  URL.revokeObjectURL(a.href);
}
