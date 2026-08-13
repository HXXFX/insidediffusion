/**
 * The denoiser forward pass, in plain JavaScript.
 *
 * This is a LINE-FOR-LINE port of train/model.py::UNet.forward. The two must
 * agree numerically; `test/parity.html` checks every intermediate against a
 * fixture exported from PyTorch, not just the final output — two mistakes can
 * cancel at the output and a whole-network check would pass.
 *
 * If you change the architecture in model.py, change it here in the same
 * commit. There is no code generation and no schema check that would catch a
 * drift; the parity test is the only thing standing between you and a model
 * that runs, denoises something, and is quietly wrong.
 *
 * Every intermediate is optionally captured into `collect` — that is the whole
 * reason this app trains its own model instead of shipping a packaged one.
 */

import {
  Arena, conv2d, groupNorm, silu, addInto, addChannelBias, linear,
  upsample2x, concat, softmaxRow, timestepEmbedding, numel,
} from './ops.js';

export class UNet {
  constructor(weights) {
    this.W = weights;
    this.a = weights.arch;
    this.chs = this.a.ch_mult.map((m) => m * this.a.base_ch);
    this.arena = new Arena();
    this.tdim = this.a.time_dim * 2;
  }

  // ---- building blocks ---------------------------------------------------

  /** ResBlock: norm -> silu -> conv -> +time -> norm -> silu -> conv -> +skip */
  resBlock(p, x, xs, temb, cout) {
    const A = this.arena, W = this.W;
    const h1 = A.take(numel(xs));
    groupNorm(x, xs, W.get(`${p}.norm1.weight`), W.get(`${p}.norm1.bias`), this.a.groups, h1);
    silu(h1, numel(xs), h1);

    const outShape = { c: cout, h: xs.h, w: xs.w };
    const h2 = A.take(numel(outShape));
    conv2d(h1, xs, W.get(`${p}.conv1.weight`), W.get(`${p}.conv1.bias`), cout, 3, 1, 1, h2);

    // Time embedding enters as a per-channel bias, exactly as in model.py:
    // h + self.emb(silu(temb))[:, :, None, None]
    const act = A.take(this.tdim);
    silu(temb, this.tdim, act);
    const proj = A.take(cout);
    linear(act, this.tdim, W.get(`${p}.emb.weight`), W.get(`${p}.emb.bias`), cout, proj);
    addChannelBias(h2, outShape, proj);

    const h3 = A.take(numel(outShape));
    groupNorm(h2, outShape, W.get(`${p}.norm2.weight`), W.get(`${p}.norm2.bias`), this.a.groups, h3);
    silu(h3, numel(outShape), h3);
    const h4 = A.take(numel(outShape));
    conv2d(h3, outShape, W.get(`${p}.conv2.weight`), W.get(`${p}.conv2.bias`), cout, 3, 1, 1, h4);

    // The skip is a 1x1 conv only when the width changes; otherwise identity.
    // Mirrors `nn.Identity()` in model.py — do not add a 1x1 unconditionally.
    if (W.has(`${p}.skip.weight`)) {
      const sk = A.take(numel(outShape));
      conv2d(x, xs, W.get(`${p}.skip.weight`), W.get(`${p}.skip.bias`), cout, 1, 1, 0, sk);
      addInto(h4, sk, numel(outShape));
    } else {
      addInto(h4, x, numel(outShape));
    }
    return [h4, outShape];
  }

  /** Cross-attention: Q from the image, K/V from the text (paper eq. 3). */
  crossAttn(p, x, xs, ctx, nTok, wantMap) {
    const A = this.arena, W = this.W, c = xs.c, hw = xs.h * xs.w;

    const nrm = A.take(numel(xs));
    groupNorm(x, xs, W.get(`${p}.norm.weight`), W.get(`${p}.norm.bias`), this.a.groups, nrm);
    const q = A.take(numel(xs));
    conv2d(nrm, xs, W.get(`${p}.to_q.weight`), W.get(`${p}.to_q.bias`), c, 1, 1, 0, q);

    // K and V: [nTok, c] from the [nTok, ctx_dim] context.
    const ctxDim = this.a.ctx_dim;
    const k = A.take(nTok * c), v = A.take(nTok * c);
    for (let t = 0; t < nTok; t++) {
      linear(ctx.subarray(t * ctxDim, (t + 1) * ctxDim), ctxDim,
        W.get(`${p}.to_k.weight`), W.get(`${p}.to_k.bias`), c, k.subarray(t * c, (t + 1) * c));
      linear(ctx.subarray(t * ctxDim, (t + 1) * ctxDim), ctxDim,
        W.get(`${p}.to_v.weight`), W.get(`${p}.to_v.bias`), c, v.subarray(t * c, (t + 1) * c));
    }

    const scale = 1 / Math.sqrt(c);
    const scores = A.take(nTok);
    const ctxOut = A.take(numel(xs));
    const map = wantMap ? new Float32Array(hw * nTok) : null;

    for (let i = 0; i < hw; i++) {
      for (let t = 0; t < nTok; t++) {
        let s = 0;
        for (let ch = 0; ch < c; ch++) s += q[ch * hw + i] * k[t * c + ch];
        scores[t] = s * scale;
      }
      softmaxRow(scores, nTok);
      if (map) map.set(scores.subarray(0, nTok), i * nTok);
      for (let ch = 0; ch < c; ch++) {
        let s = 0;
        for (let t = 0; t < nTok; t++) s += scores[t] * v[t * c + ch];
        ctxOut[ch * hw + i] = s;
      }
    }

    const proj = A.take(numel(xs));
    conv2d(ctxOut, xs, W.get(`${p}.proj.weight`), W.get(`${p}.proj.bias`), c, 1, 1, 0, proj);
    addInto(proj, x, numel(xs));       // residual
    return [proj, xs, map];
  }

  /** tau_theta: token embedding + position, LayerNorm, SiLU, projection. */
  textEncode(tokens) {
    const A = this.arena, W = this.W, d = this.a.ctx_dim, n = tokens.length;
    const emb = W.get('text.tok.weight'), pos = W.get('text.pos');
    const h = A.take(n * d);
    for (let t = 0; t < n; t++) {
      const src = tokens[t] * d, dst = t * d;
      for (let i = 0; i < d; i++) h[dst + i] = emb[src + i] + pos[dst + i];
    }
    const lnW = W.get('text.norm.weight'), lnB = W.get('text.norm.bias');
    const out = A.take(n * d);
    for (let t = 0; t < n; t++) {
      const o = t * d;
      let mean = 0;
      for (let i = 0; i < d; i++) mean += h[o + i];
      mean /= d;
      let varr = 0;
      for (let i = 0; i < d; i++) { const dd = h[o + i] - mean; varr += dd * dd; }
      const inv = 1 / Math.sqrt(varr / d + 1e-5);
      const tmp = A.take(d);
      for (let i = 0; i < d; i++) {
        const nv = (h[o + i] - mean) * inv * lnW[i] + lnB[i];
        tmp[i] = nv / (1 + Math.exp(-nv));                       // SiLU
      }
      linear(tmp, d, W.get('text.proj.weight'), W.get('text.proj.bias'), d,
        out.subarray(o, o + d));
    }
    return out;
  }

  timeEmbed(t) {
    const A = this.arena, W = this.W, td = this.a.time_dim;
    const raw = A.take(td);
    timestepEmbedding(t, td, raw);
    const h1 = A.take(td * 2);
    linear(raw, td, W.get('time_mlp.0.weight'), W.get('time_mlp.0.bias'), td * 2, h1);
    silu(h1, td * 2, h1);
    const h2 = A.take(td * 2);
    linear(h1, td * 2, W.get('time_mlp.2.weight'), W.get('time_mlp.2.bias'), td * 2, h2);
    return h2;
  }

  // ---- the forward pass --------------------------------------------------

  /**
   * @param x       Float32Array, CHW, 3x16x16, values roughly in [-4,4]
   * @param t       integer timestep
   * @param tokens  Int32Array of caption token ids
   * @param collect optional object; receives copies of intermediates
   * @returns Float32Array eps prediction, 3x16x16. Valid until the NEXT call —
   *          it comes from the arena, so copy it if you need to keep it.
   */
  forward(x, t, tokens, collect = null) {
    const A = this.arena, W = this.W, [c0, c1, c2] = this.chs;
    const S = this.a.img_size;
    A.reset();

    const ctx = this.textEncode(tokens);
    const temb = this.timeEmbed(t);
    const nTok = tokens.length;

    let s = { c: this.a.in_ch, h: S, w: S };
    const h0 = A.take(c0 * S * S);
    let hs = conv2d(x, s, W.get('in_conv.weight'), W.get('in_conv.bias'), c0, 3, 1, 1, h0);

    // --- encoder
    const [h1, s1] = this.resBlock('down1', h0, hs, temb, c0);              // 16
    const d1 = A.take(c0 * (S / 2) * (S / 2));
    const sd1 = conv2d(h1, s1, W.get('ds1.weight'), W.get('ds1.bias'), c0, 3, 2, 1, d1);

    let [h2, s2] = this.resBlock('down2', d1, sd1, temb, c1);               // 8
    let map8;
    [h2, s2, map8] = this.crossAttn('attn2', h2, s2, ctx, nTok, !!collect);

    const d2 = A.take(c1 * (S / 4) * (S / 4));
    const sd2 = conv2d(h2, s2, W.get('ds2.weight'), W.get('ds2.bias'), c1, 3, 2, 1, d2);

    let [h3, s3] = this.resBlock('down3', d2, sd2, temb, c2);               // 4
    [h3, s3] = this.crossAttn('attn3', h3, s3, ctx, nTok, false);

    // --- middle
    let [m, ms] = this.resBlock('mid1', h3, s3, temb, c2);
    [m, ms] = this.crossAttn('mattn', m, ms, ctx, nTok, false);
    [m, ms] = this.resBlock('mid2', m, ms, temb, c2);

    // --- decoder
    const cat3 = A.take((c2 * 2) * ms.h * ms.w);
    const sc3 = concat(m, ms, h3, s3, cat3);
    let [u3, su3] = this.resBlock('up3', cat3, sc3, temb, c1);
    [u3, su3] = this.crossAttn('uattn3', u3, su3, ctx, nTok, false);
    const up3 = A.take(c1 * su3.h * 2 * su3.w * 2);
    const sup3 = upsample2x(u3, su3, up3);                                  // 8

    const cat2 = A.take((c1 * 2) * sup3.h * sup3.w);
    const sc2 = concat(up3, sup3, h2, s2, cat2);
    let [u2, su2] = this.resBlock('up2', cat2, sc2, temb, c0);
    [u2, su2] = this.crossAttn('uattn2', u2, su2, ctx, nTok, false);
    const up2 = A.take(c0 * su2.h * 2 * su2.w * 2);
    const sup2 = upsample2x(u2, su2, up2);                                  // 16

    const cat1 = A.take((c0 * 2) * sup2.h * sup2.w);
    const sc1 = concat(up2, sup2, h1, s1, cat1);
    const [u1, su1] = this.resBlock('up1', cat1, sc1, temb, c0);

    const on = A.take(numel(su1));
    groupNorm(u1, su1, W.get('out_norm.weight'), W.get('out_norm.bias'), this.a.groups, on);
    silu(on, numel(su1), on);
    const out = A.take(this.a.in_ch * S * S);
    conv2d(on, su1, W.get('out_conv.weight'), W.get('out_conv.bias'), this.a.in_ch, 3, 1, 1, out);

    if (collect) {
      const cp = (a, sh) => ({ shape: [sh.c, sh.h, sh.w], data: Float32Array.from(a.subarray(0, numel(sh))) });
      collect.enc16 = cp(h1, s1);
      collect.enc8 = cp(h2, s2);
      collect.enc4 = cp(h3, s3);
      collect.mid = cp(m, ms);
      collect.dec8 = cp(u2, su2);
      collect.dec16 = cp(u1, su1);
      collect.attn8 = { shape: [s2.h * s2.w, nTok], data: map8 };
    }
    return out;
  }
}
