/**
 * Projects the 768-number image state down to 2-D or 3-D for the spatial views.
 *
 * The basis is PCA over the training set, computed once in train/export.py and
 * shipped as weights/projection.json (~15 kB). It is not recomputed here:
 * every run must land in the SAME space or trajectories from different seeds
 * would not be comparable, which is half the point of drawing them.
 *
 * The components are orthonormal (measured error 1.8e-15), and that is what
 * makes the landscape view honest rather than decorative. For an orthonormal P:
 *
 *     x_t = sqrt(ab)*x0 + sqrt(1-ab)*eps
 *     P'x_t = sqrt(ab)*(P'x0) + sqrt(1-ab)*N(0, I)
 *
 * so the projected noisy distribution is EXACTLY a Gaussian mixture centred on
 * the scaled projected training points with variance (1-ab). The surface drawn
 * in the landscape view is the real thing, not an illustration of one.
 */

export class Projector {
  constructor(data) {
    this.k = data.k;
    this.mean = Float32Array.from(data.mean);
    this.comp = Float32Array.from(data.components);   // k x d, row-major
    this.d = this.mean.length;
    this.cloud = Float32Array.from(data.cloud);       // n x k
    this.cloudN = data.cloud_n;
    this.scale = data.scale || 1;
    this.explained = data.explained;
  }

  /** Project one image-space vector. Writes k values into `out`. */
  project(x, out) {
    const { d, k, mean, comp } = this;
    for (let c = 0; c < k; c++) {
      let s = 0;
      const base = c * d;
      for (let i = 0; i < d; i++) s += (x[i] - mean[i]) * comp[base + i];
      out[c] = s;
    }
    return out;
  }

  /** Training points, scaled by sqrt(alphaBar) — where the data sits at this
   *  noise level. At ab=1 that is the clean data; as ab falls the whole cloud
   *  shrinks toward the origin, which is the forward process, visibly. */
  cloudAt(alphaBar, out) {
    const s = Math.sqrt(alphaBar), k = this.k;
    for (let i = 0; i < this.cloudN; i++) {
      out[i * 2] = this.cloud[i * k] * s;
      out[i * 2 + 1] = this.cloud[i * k + 1] * s;
    }
    return out;
  }
}

export async function loadProjection(base = './weights/') {
  const res = await fetch(base + 'projection.json');
  if (!res.ok) throw new Error(`projection.json: HTTP ${res.status}`);
  return new Projector(await res.json());
}
