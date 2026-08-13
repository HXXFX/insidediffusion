/**
 * The model runs in here, off the main thread.
 *
 * A guided step is ~190 ms. On the main thread that is a 190 ms freeze on
 * every step, which would make dragging the 3-D views stutter exactly while
 * the thing you want to watch is happening. So the model lives in a worker and
 * posts one snapshot per step; the UI redraws at 60 fps from whatever the
 * latest snapshot is.
 *
 * This does NOT weaken the "in sync" guarantee. Every view is a pure function
 * of one snapshot, and a snapshot is only ever posted when the step that
 * produced it is complete. The UI can lag the model by a frame; it can never
 * show a mixture of two steps.
 *
 * The step loop yields via setTimeout between steps so `stop` and `configure`
 * messages are actually received. A tight while-loop here would make the
 * worker unresponsive, which is the same bug one layer down.
 */

import { loadWeights } from './nn/weights.js';
import { UNet } from './nn/unet.js';
import { Sampler } from './diffusion/sampler.js';
import { qSample, alphaBar } from './diffusion/schedule.js';

/** Small seeded PRNG. The sampler has its own; this one exists so a training
 *  step can be repeated exactly without disturbing a run in progress. */
function mulberry(a) {
  return function () {
    a |= 0; a = (a + 0x6D2B79F5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

let weights = null;
let unet = null;
let sampler = null;
let running = false;
let runId = 0;
let collectInternals = false;

function post(msg, transfer) { self.postMessage(msg, transfer || []); }

/**
 * Snapshot -> message. Buffers the worker will never touch again are
 * TRANSFERRED rather than copied.
 *
 * `next` is deliberately NOT in the transfer list, and that is not an
 * oversight. The sampler keeps it as its current state for the following step,
 * so handing the buffer to the main thread detaches it underneath the sampler
 * and step 2 dies with "Cannot perform %TypedArray%.prototype.values on a
 * detached or out-of-bounds ArrayBuffer". It is 3 kB; let it be copied.
 */
function sendSnapshot(s) {
  const payload = {
    type: 'snapshot',
    index: s.index, total: s.total, t: s.t,
    alphaBar: s.alphaBar, noiseLevel: s.noiseLevel, ms: s.ms,
    x: s.x, eps: s.eps, x0: s.x0, next: s.next,
    internals: null,
  };
  const transfer = [s.x.buffer, s.eps.buffer, s.x0.buffer];
  if (s.internals) {
    const out = {};
    for (const [k, v] of Object.entries(s.internals)) {
      if (!v) continue;
      out[k] = { shape: v.shape, data: v.data };
      transfer.push(v.data.buffer);
    }
    payload.internals = out;
  }
  post(payload, transfer);
}

/**
 * The step loop.
 *
 * WRAPPED IN try/catch because it is started without `await`. An exception in
 * here is an unhandled promise rejection: the run simply stops, no error
 * reaches the UI, and the app sits at step 1 looking like it is thinking. That
 * is exactly how the detached-buffer bug above presented.
 */
async function loop(myRun) {
  try {
    while (running && runId === myRun && sampler && !sampler.done) {
      const s = sampler.step(collectInternals);
      if (!s) break;
      sendSnapshot(s);
      // Yield so messages get through. 0 ms is enough; the step is the long part.
      await new Promise((r) => setTimeout(r, 0));
    }
    if (runId === myRun && sampler && sampler.done) {
      running = false;
      post({ type: 'done', medianStepMs: sampler.medianStepMs() });
    }
  } catch (err) {
    running = false;
    post({ type: 'error', message: `step ${sampler ? sampler.i : '?'}: ${err && err.message ? err.message : err}` });
  }
}

self.onmessage = async (e) => {
  const m = e.data;
  try {
    if (m.type === 'init') {
      weights = await loadWeights(
        m.base,
        (p) => post({ type: 'loading', progress: p }),
        (man) => post({ type: 'manifest', manifest: man }));
      unet = new UNet(weights);
      post({ type: 'ready', manifest: weights.manifest });
      return;
    }

    if (m.type === 'start') {
      if (!unet) throw new Error('model not loaded');
      const enc = weights.encodePrompt(m.prompt);
      collectInternals = !!m.collectInternals;
      const opts = {
        steps: m.steps, guidance: m.guidance, seed: m.seed,
        tokens: enc.ids, nullTokens: weights.nullCaption(),
        initial: m.initial || null, strength: m.strength ?? 1,
      };
      if (!sampler) sampler = new Sampler(unet, opts);
      else sampler.configure(opts);
      runId++;
      running = true;
      post({
        type: 'started', total: sampler.total,
        known: enc.known, unknown: enc.unknown,
        // Which slots the sentence never mentioned, and what went in them.
        // "Why did I get legs I did not ask for" has an answer; the app shows
        // it rather than leaving the user to guess.
        defaulted: enc.defaulted, slots: enc.slots, sentence: enc.sentence,
        tokens: Array.from(enc.ids),
        tokenWords: Array.from(enc.ids).map((i) => weights.vocab[i]),
      });
      loop(runId);
      return;
    }

    /**
     * ONE TRAINING STEP, for real.
     *
     * This is the whole training loop except the weight update: take a real
     * training picture, draw noise, corrupt the picture by exactly the amount
     * the schedule prescribes for `t`, and ask the model to say what noise was
     * added. The error between what it says and what actually went in is the
     * training loss — the same expression train_monsters.py minimises, computed
     * on one example instead of a batch.
     *
     * It runs the SAME `unet.forward` the sampler runs, on the same loaded
     * weights, so nothing here is a re-enactment. What it cannot do is
     * backpropagate: these weights are finished, and there is no optimiser in
     * the browser. The tab says so rather than implying the picture is learning
     * while you watch.
     */
    if (m.type === 'trainStep') {
      if (!unet) throw new Error('model not loaded');
      const enc = weights.encodePrompt(m.prompt);
      const x0 = new Float32Array(m.x0);
      const n = x0.length;
      const noise = new Float32Array(n);
      // Box-Muller off a seeded generator, so a given (example, t, seed) is
      // reproducible — you can change one control at a time and attribute the
      // difference to it.
      const rnd = mulberry(m.seed >>> 0);
      for (let i = 0; i < n; i += 2) {
        const u = Math.max(1e-7, rnd()), v = rnd();
        const r = Math.sqrt(-2 * Math.log(u));
        noise[i] = r * Math.cos(6.283185307179586 * v);
        if (i + 1 < n) noise[i + 1] = r * Math.sin(6.283185307179586 * v);
      }
      const xt = qSample(x0, noise, m.t, new Float32Array(n));
      const t0 = performance.now();
      const out = unet.forward(xt, m.t, enc.ids, null);
      const eps = Float32Array.from(out.subarray(0, n));
      const ms = performance.now() - t0;

      let se = 0;
      const err = new Float32Array(n);
      for (let i = 0; i < n; i++) {
        const d = eps[i] - noise[i];
        err[i] = d;
        se += d * d;
      }
      // x0-hat: what the model's guess implies the clean picture was. This is
      // the same rearrangement the sampler does, and it is what makes the
      // prediction legible — a field of noise means nothing to look at, but the
      // picture it implies means everything.
      const ab = alphaBar(m.t), sa = Math.sqrt(ab), sb = Math.sqrt(1 - ab);
      const x0hat = new Float32Array(n);
      for (let i = 0; i < n; i++) {
        x0hat[i] = Math.max(-1, Math.min(1, (xt[i] - sb * eps[i]) / sa));
      }

      post({
        type: 'trainStep',
        t: m.t, alphaBar: ab, loss: se / n, ms,
        sentence: enc.sentence,
        xt: xt.buffer, noise: noise.buffer, eps: eps.buffer,
        err: err.buffer, x0hat: x0hat.buffer,
      }, [xt.buffer, noise.buffer, eps.buffer, err.buffer, x0hat.buffer]);
      return;
    }

    if (m.type === 'stop') { running = false; runId++; return; }

    if (m.type === 'resume') {
      if (!sampler || sampler.done) return;
      running = true; runId++; loop(runId);
      return;
    }
  } catch (err) {
    running = false;
    post({ type: 'error', message: err && err.message ? err.message : String(err) });
  }
};
