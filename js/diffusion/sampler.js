/**
 * The step engine — the piece that makes the visualisation honest.
 *
 * The whole app is built around one idea: the sampler computes EXACTLY ONE
 * STEP at a time and emits a complete snapshot of everything worth drawing.
 * Every view is then a pure function of one snapshot. Nothing animates on its
 * own clock, so nothing can drift out of sync with the model, and scrubbing
 * backwards is just indexing an array.
 *
 * The full history is kept. At 16x16 a 60-step run is a few hundred kilobytes,
 * so there is no reason to be clever about it, and "go back and look again" is
 * the single most useful thing a learner can do.
 *
 * PORT OF train/schedule.py::ddim_sample. Keep them in step.
 */

import { alphaBar, ddimTimesteps } from './schedule.js';

/** Deterministic PRNG. Never Math.random anywhere in this path: the same seed
 *  must give byte-identical output, or the app contradicts itself on reload. */
export function mulberry32(seed) {
  let a = seed >>> 0;
  return function () {
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

export function gaussianField(n, seed) {
  const rnd = mulberry32(seed), out = new Float32Array(n);
  for (let i = 0; i < n; i++) {
    let u = 0, v = 0;
    while (u === 0) u = rnd();
    while (v === 0) v = rnd();
    out[i] = Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
  }
  return out;
}

export class Sampler {
  /**
   * @param unet    UNet instance
   * @param opts    { steps, guidance, seed, tokens, nullTokens, initial, strength }
   *
   * `initial` + `strength` drive image-to-image: the source is noised to
   * `strength` of the way to pure noise and denoising starts from there.
   */
  constructor(unet, opts) {
    this.unet = unet;
    const a = unet.a;
    this.n = a.in_ch * a.img_size * a.img_size;
    this.configure(opts);
  }

  configure(o) {
    const a = this.unet.a;
    this.steps = o.steps ?? 30;
    this.guidance = o.guidance ?? 3.0;
    this.seed = o.seed ?? 1337;
    this.tokens = o.tokens;
    this.nullTokens = o.nullTokens;
    this.strength = o.strength ?? 1.0;

    const all = ddimTimesteps(this.steps);
    // Image-to-image starts partway down the schedule. At strength 1 this is
    // the whole run, which is exactly text-to-image — one code path, not two.
    const skip = Math.round((1 - this.strength) * this.steps);
    this.ts = all.slice(Math.min(skip, this.steps - 1));

    const noise = gaussianField(this.n, this.seed);
    if (o.initial && this.strength < 1) {
      const ab = alphaBar(this.ts[0]), sa = Math.sqrt(ab), sb = Math.sqrt(1 - ab);
      this.x = new Float32Array(this.n);
      for (let i = 0; i < this.n; i++) this.x[i] = sa * o.initial[i] + sb * noise[i];
    } else {
      this.x = noise;
    }

    this.i = 0;
    this.history = [];
    this.done = false;
  }

  get total() { return this.ts.length; }

  /**
   * Advance exactly one step and return the snapshot.
   * Returns null when the run is finished.
   */
  step(collectInternals = false) {
    if (this.done) return null;
    const i = this.i, t = this.ts[i], n = this.n;
    const ab = alphaBar(t);
    const collect = collectInternals ? {} : null;

    const t0 = performance.now();
    let eps = this.unet.forward(this.x, t, this.tokens, collect);
    let epsCond = null;

    if (this.guidance !== 1.0) {
      // eps = eps_uncond + g * (eps_cond - eps_uncond)
      epsCond = Float32Array.from(eps.subarray(0, n));
      const epsU = this.unet.forward(this.x, t, this.nullTokens, null);
      const g = this.guidance;
      const merged = new Float32Array(n);
      for (let k = 0; k < n; k++) merged[k] = epsU[k] + g * (epsCond[k] - epsU[k]);
      eps = merged;
    } else {
      eps = Float32Array.from(eps.subarray(0, n));
    }
    const ms = performance.now() - t0;

    // x0 estimate, clamped — clamping is what keeps low step counts stable.
    const sa = Math.sqrt(ab), sb = Math.sqrt(1 - ab);
    const x0 = new Float32Array(n);
    for (let k = 0; k < n; k++) {
      x0[k] = Math.max(-1, Math.min(1, (this.x[k] - sb * eps[k]) / sa));
    }

    const tNext = i + 1 < this.ts.length ? this.ts[i + 1] : 0;
    const abNext = i + 1 < this.ts.length ? alphaBar(tNext) : 1.0;
    const saN = Math.sqrt(abNext), sbN = Math.sqrt(1 - abNext);
    const next = new Float32Array(n);
    for (let k = 0; k < n; k++) next[k] = saN * x0[k] + sbN * eps[k];

    const snap = {
      index: i,
      total: this.ts.length,
      t,
      alphaBar: ab,
      noiseLevel: Math.sqrt(1 - ab),
      x: Float32Array.from(this.x),
      eps,
      x0,
      next,
      internals: collect,
      ms,
    };
    this.history.push(snap);

    this.x = next;
    this.i++;
    if (this.i >= this.ts.length) this.done = true;
    return snap;
  }

  /** The snapshot at a given index — this is what the timeline scrubs over. */
  at(i) {
    if (!this.history.length) return null;
    return this.history[Math.max(0, Math.min(this.history.length - 1, i))];
  }

  /** Median per-step time. Median, not mean: the first step pays JIT warm-up
   *  and would drag a mean upward for the whole run. */
  medianStepMs() {
    if (!this.history.length) return 0;
    const v = this.history.map((s) => s.ms).sort((a, b) => a - b);
    return v[v.length >> 1];
  }
}
