/**
 * End-to-end trainer test (no browser):
 * Train a tiny LM so it learns the assistant name "Akari".
 * Run: node e2e_akari_test.mjs
 */
import { TinyLM } from "./train.js";
import { parseDataset, exampleToIdsAndMask, CHAT_TEMPLATES } from "./unsloth-export.js";

const dataset = `
{"messages":[{"role":"user","content":"What is your name?"},{"role":"assistant","content":"My name is Akari."}]}
{"messages":[{"role":"user","content":"Who are you?"},{"role":"assistant","content":"I am Akari, your companion."}]}
{"messages":[{"role":"user","content":"Tell me your name."},{"role":"assistant","content":"Akari."}]}
{"messages":[{"role":"user","content":"Name?"},{"role":"assistant","content":"Akari"}]}
{"messages":[{"role":"user","content":"Introduce yourself."},{"role":"assistant","content":"Hi, I'm Akari."}]}
`.trim();

const examples = parseDataset(dataset, "messages");
console.log("examples:", examples.length);

const lm = new TinyLM({ nVocab: 256, nEmb: 64, nLayer: 2, nHead: 4, nFF: 128 });
const nAd = lm.enableLora(4, 8);
console.log("LoRA adapters:", nAd, "params~", [...lm.lora.adapters.values()].reduce((a, ad) => a + ad.A.length + ad.B.length, 0));

const seqLen = 96;
const steps = 80;
const lr = 5e-3;
const losses = [];

for (let step = 0; step < steps; step++) {
  const ex = examples[step % examples.length];
  const { ids, mask } = exampleToIdsAndMask(ex, "chatml", seqLen, true);
  const { loss, grads } = lm.lossAndGrad(ids, { fullLayers: true, mask });
  lm.adamStep(grads, lr);
  losses.push(loss);
  if (step % 10 === 0 || step === steps - 1) {
    const avg = losses.slice(-10).reduce((s, x) => s + x, 0) / Math.min(10, losses.length);
    console.log(`step ${step} loss=${loss.toFixed(4)} avg10=${avg.toFixed(4)} gradKeys=${grads.size}`);
  }
}

const merged = lm.mergeLora();
console.log("merged adapters:", merged);

function scoreContinuation(promptText, targetText) {
  const full = promptText + targetText;
  const ids = [];
  for (let i = 0; i < full.length && ids.length < seqLen; i++) ids.push(full.charCodeAt(i) % 256);
  while (ids.length < seqLen) ids.push(0);
  const cache = lm.forwardLoss(ids);
  const promptLen = Math.min(promptText.length, seqLen - 1);
  let nll = 0, n = 0;
  for (let t = promptLen - 1; t < promptLen - 1 + targetText.length && t < seqLen - 1; t++) {
    const target = ids[t + 1];
    const o = t * lm.cfg.nVocab;
    let max = -Infinity;
    for (let v = 0; v < lm.cfg.nVocab; v++) if (cache.logits[o + v] > max) max = cache.logits[o + v];
    let sum = 0;
    for (let v = 0; v < lm.cfg.nVocab; v++) sum += Math.exp(cache.logits[o + v] - max);
    const p = Math.exp(cache.logits[o + target] - max) / sum;
    nll += -Math.log(p + 1e-12);
    n++;
  }
  return nll / (n || 1);
}

const prompt = "<|im_start|>user\nWhat is your name?<|im_end|>\n<|im_start|>assistant\n";
const nllAkari = scoreContinuation(prompt, "My name is Akari.");
const nllBob = scoreContinuation(prompt, "My name is Robert.");
console.log("NLL 'My name is Akari.':", nllAkari.toFixed(4));
console.log("NLL 'My name is Robert.':", nllBob.toFixed(4));

const first = losses[0];
const last = losses[losses.length - 1];
const improved = last < first * 0.95 || last < first - 0.05;
const prefersAkari = nllAkari < nllBob;

console.log("\n=== RESULT ===");
console.log("loss improved:", improved, `(${first.toFixed(3)} -> ${last.toFixed(3)})`);
console.log("prefers Akari over Robert:", prefersAkari);
if (improved) {
  console.log("PASS: training reduced loss with LoRA + response mask");
  process.exit(0);
} else {
  console.log("WARN: loss did not drop enough — check trainer");
  process.exit(1);
}
