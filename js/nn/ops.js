/**
 * Tensor primitives for the denoiser.
 *
 * Everything is CHW `Float32Array` plus an explicit `{c,h,w}` shape. No tensor
 * class, no autograd, no broadcasting — this runs one fixed forward pass and
 * anything more general would be weight the hot loop has to carry.
 *
 * ALLOCATION IS THE THING TO WATCH HERE, not the algorithm. A forward pass
 * touches ~30 M multiply-accumulates spread over ~40 intermediate tensors; if
 * each op returned a fresh array, the collector would be doing more work than
 * the convolutions. So every op writes into a caller-supplied buffer, and
 * `Arena` hands those out from a pool that is reset once per pass rather than
 * freed per op.
 *
 * To change how fast this runs, look at `conv2d` — it is ~85% of the work.
 */

/** A reusable pool of Float32Arrays, keyed by length. */
export class Arena {
  constructor() {
    this.pools = new Map();   // length -> array of buffers
    this.taken = [];
  }
  /** Borrow a zeroed buffer of `n` floats. Valid until reset(). */
  take(n) {
    let pool = this.pools.get(n);
    if (!pool) { pool = []; this.pools.set(n, pool); }
    const buf = pool.length ? pool.pop() : new Float32Array(n);
    buf.fill(0);
    this.taken.push([n, buf]);
    return buf;
  }
  /** Return every borrowed buffer. Call once per forward pass, not per op. */
  reset() {
    for (const [n, buf] of this.taken) this.pools.get(n).push(buf);
    this.taken.length = 0;
  }
  get bytes() {
    let t = 0;
    for (const [n, pool] of this.pools) t += n * 4 * pool.length;
    return t;
  }
}

export const numel = (s) => s.c * s.h * s.w;

/**
 * 2-D convolution, stride 1 or 2, `pad` on every side.
 *
 * Loop order is deliberate and load-bearing: output channel, then input
 * channel, then kernel tap, then the spatial sweep. That hoists the weight to
 * a scalar in the innermost loop and walks both the input row and the output
 * row with stride 1, so each is a straight run through cache.
 *
 * NEGATIVE RESULT — do not retry this. The obvious alternative is "gather"
 * order: output pixel outermost, accumulating into a local scalar and writing
 * once, which avoids the read-modify-write to out[] that this version does on
 * every multiply-accumulate. That looks like it should win and it does not.
 * Measured with tools/bench_conv.mjs, interleaved runs, median of 5:
 *
 *     16x16 32->32 k3   scatter 260 MMAC/s   gather 197   0.76x
 *     16x16 64->32 k3   scatter 274          gather 215   0.79x
 *      8x8  64->64 k3   scatter 267          gather 215   0.80x
 *      4x4 192->64 k3   scatter 217          gather 248   1.15x
 *     16x16 32->32 k1   scatter 257          gather 114   0.45x
 *                                            overall      0.83x
 *
 * Gather wins only at 4x4, where the output plane is 16 elements and scatter
 * re-walks it 192*9 times. Everywhere else the sequential output write is
 * worth more than the register accumulator. If the model ever grows a level
 * with very high channel count at very low resolution, revisit — otherwise no.
 */
export function conv2d(x, xs, w, b, cout, k, stride, pad, out) {
  const { c: cin, h: hin, w: win } = xs;
  const hout = Math.floor((hin + 2 * pad - k) / stride) + 1;
  const wout = Math.floor((win + 2 * pad - k) / stride) + 1;
  const planeIn = hin * win, planeOut = hout * wout;

  for (let oc = 0; oc < cout; oc++) {
    const ob = oc * planeOut;
    if (b) out.fill(b[oc], ob, ob + planeOut); else out.fill(0, ob, ob + planeOut);

    for (let ic = 0; ic < cin; ic++) {
      const ib = ic * planeIn;
      const wb = (oc * cin + ic) * k * k;

      for (let ky = 0; ky < k; ky++) {
        for (let kx = 0; kx < k; kx++) {
          const wv = w[wb + ky * k + kx];
          if (wv === 0) continue;

          for (let oy = 0; oy < hout; oy++) {
            const iy = oy * stride + ky - pad;
            if (iy < 0 || iy >= hin) continue;
            const rowIn = ib + iy * win;
            const rowOut = ob + oy * wout;

            // Clamp the x sweep once instead of testing bounds per pixel.
            let ox0 = 0, ox1 = wout;
            const ix0 = kx - pad;
            if (ix0 < 0) ox0 = Math.ceil(-ix0 / stride);
            const maxOx = Math.floor((win - 1 - ix0) / stride);
            if (maxOx < ox1 - 1) ox1 = maxOx + 1;

            for (let ox = ox0; ox < ox1; ox++) {
              out[rowOut + ox] += wv * x[rowIn + ox * stride + ix0];
            }
          }
        }
      }
    }
  }
  return { c: cout, h: hout, w: wout };
}

/** GroupNorm + affine. Operates out-of-place so the input stays readable. */
export function groupNorm(x, xs, gamma, beta, groups, out, eps = 1e-5) {
  const { c, h, w } = xs, plane = h * w, per = c / groups;
  for (let g = 0; g < groups; g++) {
    const start = g * per * plane, n = per * plane;
    let mean = 0;
    for (let i = 0; i < n; i++) mean += x[start + i];
    mean /= n;
    let vsum = 0;
    for (let i = 0; i < n; i++) { const d = x[start + i] - mean; vsum += d * d; }
    const inv = 1 / Math.sqrt(vsum / n + eps);
    for (let j = 0; j < per; j++) {
      const ch = g * per + j, gv = gamma[ch] * inv, bv = beta[ch] - mean * gv;
      const o = ch * plane;
      for (let i = 0; i < plane; i++) out[o + i] = x[o + i] * gv + bv;
    }
  }
  return xs;
}

/** x * sigmoid(x), in place when out === x. */
export function silu(x, n, out) {
  for (let i = 0; i < n; i++) { const v = x[i]; out[i] = v / (1 + Math.exp(-v)); }
}

export function addInto(a, b, n) { for (let i = 0; i < n; i++) a[i] += b[i]; }

/** Add a per-channel bias vector across every spatial position. */
export function addChannelBias(x, xs, bias, offset = 0) {
  const plane = xs.h * xs.w;
  for (let ch = 0; ch < xs.c; ch++) {
    const v = bias[offset + ch], o = ch * plane;
    for (let i = 0; i < plane; i++) x[o + i] += v;
  }
}

/** y = W x + b, with W stored row-major [out, in] as PyTorch writes it. */
export function linear(x, cin, w, b, cout, out) {
  for (let o = 0; o < cout; o++) {
    let s = b ? b[o] : 0;
    const row = o * cin;
    for (let i = 0; i < cin; i++) s += w[row + i] * x[i];
    out[o] = s;
  }
}

/** Nearest-neighbour 2x upsample. Matches torch interpolate(mode='nearest'). */
export function upsample2x(x, xs, out) {
  const { c, h, w } = xs, W2 = w * 2;
  for (let ch = 0; ch < c; ch++) {
    const src = ch * h * w, dst = ch * h * w * 4;
    for (let y = 0; y < h; y++) {
      for (let x2 = 0; x2 < w; x2++) {
        const v = x[src + y * w + x2];
        const d = dst + (y * 2) * W2 + x2 * 2;
        out[d] = v; out[d + 1] = v; out[d + W2] = v; out[d + W2 + 1] = v;
      }
    }
  }
  return { c, h: h * 2, w: w * 2 };
}

/** Concatenate two CHW tensors along channels. Shapes must match spatially. */
export function concat(a, as, b, bs, out) {
  const plane = as.h * as.w;
  out.set(a.subarray(0, as.c * plane), 0);
  out.set(b.subarray(0, bs.c * plane), as.c * plane);
  return { c: as.c + bs.c, h: as.h, w: as.w };
}

export function softmaxRow(row, n) {
  let m = -Infinity;
  for (let i = 0; i < n; i++) if (row[i] > m) m = row[i];
  let s = 0;
  for (let i = 0; i < n; i++) { row[i] = Math.exp(row[i] - m); s += row[i]; }
  const inv = 1 / s;
  for (let i = 0; i < n; i++) row[i] *= inv;
}

/**
 * Sinusoidal timestep embedding.
 *
 * MUST match train/model.py::timestep_embedding exactly, including the
 * cos-then-sin concatenation order. Getting the halves the wrong way round
 * produces a model that still runs and still denoises *something*, just
 * consistently wrong — which is the hardest kind of bug to see, so the parity
 * test checks this tensor specifically.
 */
export function timestepEmbedding(t, dim, out) {
  const half = dim >> 1;
  for (let i = 0; i < half; i++) {
    const freq = Math.exp((-Math.log(10000) * i) / half);
    const a = t * freq;
    out[i] = Math.cos(a);
    out[half + i] = Math.sin(a);
  }
}
