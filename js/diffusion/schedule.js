/**
 * The noise schedule.
 *
 * PORT OF train/schedule.py. That file is the spec; if the two disagree the
 * model is being run off-schedule and the output degrades in a way that looks
 * like a bad model rather than a bug. The parity test compares them.
 *
 * Cosine schedule (Nichol & Dhariwal, arXiv:2102.09672) rather than the
 * original linear betas: at 16x16 a linear schedule destroys the image far too
 * early, so most of the trajectory is spent on pure noise — worse samples, and
 * worse to watch, which matters more here than usual.
 */

export const T_TRAIN = 1000;
export const COSINE_S = 0.008;

const f = (u) => Math.pow(Math.cos(((u + COSINE_S) / (1 + COSINE_S)) * Math.PI / 2), 2);
const F0 = f(0);

/** Cumulative signal retained at timestep t. t=0 -> 1 (clean), t=T -> 0. */
export function alphaBar(t, T = T_TRAIN) {
  return f(t / T) / F0;
}

/**
 * Timesteps for an n-step run: evenly spaced, descending, ending at 0.
 *
 * Matches numpy linspace(T-1, 0, n).round(). Do NOT change the start to T —
 * the model was never trained at t = T and the first step would be off
 * distribution.
 */
export function ddimTimesteps(n, T = T_TRAIN) {
  const out = new Int32Array(n);
  if (n === 1) { out[0] = 0; return out; }
  for (let i = 0; i < n; i++) out[i] = Math.round((T - 1) * (1 - i / (n - 1)));
  return out;
}

/** Forward process, used by image-to-image: x_t = √ᾱ·x₀ + √(1-ᾱ)·ε */
export function qSample(x0, noise, t, out) {
  const ab = alphaBar(t), sa = Math.sqrt(ab), sb = Math.sqrt(1 - ab);
  for (let i = 0; i < x0.length; i++) out[i] = sa * x0[i] + sb * noise[i];
  return out;
}
