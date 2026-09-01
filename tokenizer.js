/**
 * GGUF tokenizer loader + encode/decode.
 * Reads tokenizer.ggml.* from parsed GGUF metadata when present.
 * Falls back to byte-level for models without vocab in the file.
 */

export function extractTokenizerFromMetadata(metadata) {
  const tokens = metadata["tokenizer.ggml.tokens"];
  const scores = metadata["tokenizer.ggml.scores"];
  const tokenType = metadata["tokenizer.ggml.token_type"];
  const merges = metadata["tokenizer.ggml.merges"];
  const model = metadata["tokenizer.ggml.model"];

  let vocabList = null;
  if (Array.isArray(tokens)) vocabList = tokens;
  else if (tokens && Array.isArray(tokens.items)) vocabList = tokens.items;
  else if (tokens && typeof tokens === "object" && tokens.value) {
    vocabList = Array.isArray(tokens.value) ? tokens.value : null;
  }

  if (!vocabList || vocabList.length === 0) {
    return { kind: "byte", vocabSize: 256, tokenToId: null, idToToken: null, merges: null };
  }

  const tokenToId = new Map();
  const idToToken = [];
  for (let i = 0; i < vocabList.length; i++) {
    const t = String(vocabList[i]);
    tokenToId.set(t, i);
    idToToken.push(t);
  }

  let mergePairs = null;
  if (Array.isArray(merges) && merges.length) {
    mergePairs = merges.map((m) => String(m).split(" "));
  } else if (merges && merges.items) {
    mergePairs = merges.items.map((m) => String(m).split(" "));
  }

  return {
    kind: model === "gpt2" || mergePairs ? "bpe" : "vocab",
    vocabSize: idToToken.length,
    tokenToId,
    idToToken,
    merges: mergePairs,
    scores: Array.isArray(scores) ? scores : scores?.items || null,
    tokenType: Array.isArray(tokenType) ? tokenType : tokenType?.items || null,
  };
}

export function encodeText(tok, text, maxLen) {
  if (!tok || tok.kind === "byte") {
    const ids = [];
    for (let i = 0; i < text.length && ids.length < maxLen; i++) {
      ids.push(text.charCodeAt(i) % 256);
    }
    while (ids.length < maxLen) ids.push(0);
    return ids;
  }

  const ids = [];
  let i = 0;
  const s = text;
  while (i < s.length && ids.length < maxLen) {
    let matched = false;
    const maxPiece = Math.min(32, s.length - i);
    for (let len = maxPiece; len >= 1; len--) {
      const piece = s.slice(i, i + len);
      if (tok.tokenToId.has(piece)) {
        ids.push(tok.tokenToId.get(piece));
        i += len;
        matched = true;
        break;
      }
    }
    if (!matched) {
      const ch = s[i];
      if (tok.tokenToId.has(ch)) ids.push(tok.tokenToId.get(ch));
      else if (tok.tokenToId.has("<unk>")) ids.push(tok.tokenToId.get("<unk>"));
      else if (tok.tokenToId.has("<UNK>")) ids.push(tok.tokenToId.get("<UNK>"));
      else ids.push(0);
      i += 1;
    }
  }
  while (ids.length < maxLen) {
    if (tok.tokenToId.has("<pad>")) ids.push(tok.tokenToId.get("<pad>"));
    else ids.push(0);
  }
  return ids;
}

export function decodeIds(tok, ids) {
  if (!tok || tok.kind === "byte") {
    return ids.map((id) => String.fromCharCode(id % 256)).join("").replace(/\0+$/, "");
  }
  let out = "";
  for (const id of ids) {
    if (id === 0) continue;
    const t = tok.idToToken[id];
    if (t != null) out += t;
  }
  return out.replace(/▁/g, " ").replace(/Ġ/g, " ");
}
