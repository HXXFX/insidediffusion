/**
 * Loads `weights/<model>/manifest.json` + `model.bin` into named Float32Arrays.
 *
 * The manifest is written by train/export_monsters.py and is the single
 * agreement between the two halves of this project: it carries the layer order,
 * every tensor shape, the vocabulary, the caption grammar and the schedule
 * constants. If the JS and the Python disagree about anything structural, it
 * should be visible here rather than as a silently wrong image.
 *
 * TWO MODELS. Fast and Detailed differ only in channel width, and `base_ch` is
 * in the manifest, so nothing here or in unet.js is written for one of them.
 * `weights/models.json` lists what exists; the app picks and this loads it.
 */

import { Grammar } from '../prompt.js';

/**
 * Decode IEEE-754 half precision.
 *
 * Written out rather than using `DataView.getFloat16`, which is too new to
 * rely on, or a `Float16Array`, which is newer still. Subnormals are handled
 * because the exported weights genuinely contain them — a model trained to
 * this loss has plenty of near-zero parameters, and flushing them to zero
 * shifts the output enough to fail the parity check.
 */
function f16to32(h) {
  const sign = (h & 0x8000) ? -1 : 1;
  const exp = (h >> 10) & 0x1f;
  const frac = h & 0x03ff;
  if (exp === 0) return sign * frac * 5.960464477539063e-8;   // subnormal: 2^-24
  if (exp === 31) return frac ? NaN : sign * Infinity;
  return sign * Math.pow(2, exp - 15) * (1 + frac / 1024);
}

export class Weights {
  constructor(manifest, tensors) {
    this.manifest = manifest;
    this.t = tensors;                     // name -> Float32Array
    this.arch = manifest.arch;
    this.vocab = manifest.vocab;
    this.tokenOf = new Map(manifest.vocab.map((w, i) => [w, i]));
    this.grammar = new Grammar(manifest.caption, this.tokenOf);
  }

  /** Throws with the tensor name rather than returning undefined — a missing
   *  weight otherwise surfaces as NaNs several layers later. */
  get(name) {
    const a = this.t[name];
    if (!a) throw new Error(`weight not found: ${name}`);
    return a;
  }
  has(name) { return !!this.t[name]; }

  /**
   * "a green slime with three eyes" -> tokens, plus what the parser did with
   * each word.
   *
   * The old free-word tokeniser is gone. This caption has one slot per part and
   * position IS meaning, so appending recognised words in the order they were
   * typed would put `bat` where the eyes go — producing a clean monster
   * unrelated to the sentence. See js/prompt.js.
   */
  encodePrompt(text) {
    const p = this.grammar.parse(text);
    return {
      ids: this.grammar.encode(p.slots),
      slots: p.slots, known: p.known, unknown: p.unknown,
      defaulted: p.defaulted, sentence: this.grammar.describe(p.slots),
    };
  }

  /** Encode already-chosen slot values — what the prompt builder sends. */
  encodeSlots(slots) {
    return { ids: this.grammar.encode(slots), slots,
             sentence: this.grammar.describe(slots) };
  }

  nullCaption() { return Int32Array.from(this.manifest.caption.null); }
}

/** The model index: which models exist, how big, which is the default. */
export async function loadIndex(base = './weights/') {
  const res = await fetch(base + 'models.json');
  if (!res.ok) throw new Error(`models.json: HTTP ${res.status}`);
  return res.json();
}

/**
 * @param base directory holding manifest.json and model.bin — one model.
 * @param onManifest called as soon as the manifest is parsed, before the
 *   2.4 MB weight blob is fetched. The vocabulary lives in the manifest, and
 *   the word list is the first thing a newcomer needs — making them wait for
 *   the model to download before they can even see what words exist is
 *   backwards.
 */
export async function loadWeights(base = './weights/fast/', onProgress, onManifest) {
  const mres = await fetch(base + 'manifest.json');
  if (!mres.ok) throw new Error(`manifest.json: HTTP ${mres.status}`);
  const manifest = await mres.json();
  onManifest && onManifest(manifest);

  const res = await fetch(base + 'model.bin');
  if (!res.ok) throw new Error(`model.bin: HTTP ${res.status}`);

  // Stream so the loading bar reflects real bytes rather than a guess.
  const total = manifest.total_bytes;
  let buf;
  if (res.body && onProgress) {
    const reader = res.body.getReader();
    const chunks = [];
    let got = 0;
    for (;;) {
      const { done, value } = await reader.read();
      if (done) break;
      chunks.push(value);
      got += value.length;
      onProgress(Math.min(1, got / total));
    }
    buf = new Uint8Array(got);
    let at = 0;
    for (const c of chunks) { buf.set(c, at); at += c.length; }
    buf = buf.buffer;
  } else {
    buf = await res.arrayBuffer();
  }

  if (buf.byteLength < total) {
    throw new Error(`model.bin truncated: ${buf.byteLength} of ${total} bytes`);
  }

  const half = manifest.dtype === 'float16';
  const src = half ? new Uint16Array(buf) : new Float32Array(buf);
  const tensors = {};
  for (const t of manifest.tensors) {
    const start = t.offset / (half ? 2 : 4);
    if (half) {
      const a = new Float32Array(t.count);
      for (let i = 0; i < t.count; i++) a[i] = f16to32(src[start + i]);
      tensors[t.name] = a;
    } else {
      tensors[t.name] = src.subarray(start, start + t.count);
    }
  }
  onProgress && onProgress(1);
  return new Weights(manifest, tensors);
}
