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

/**
 * The actual training pictures behind the cloud points.
 *
 * WHAT THIS BUYS. The funnel's floor and the landscape's surface are both built
 * from 500 real training images, and both were drawing them as anonymous marks
 * — grey dots and a brown hill. That is a true picture of the data and it looks
 * like abstract decoration, which is the same complaint that made Towers the
 * view people trusted: Towers shows the monster, so nobody has to be told the
 * geometry means something. These are real monsters, so they can be shown as
 * monsters.
 *
 * INDEX MAPS, 256 bytes each, exactly as the Making of tab ships its examples:
 * 0-3 selects background / body colour / outline / eye white, and the RGB table
 * rides along in the same file. So what gets drawn is not a likeness of a
 * training picture, it IS one, pixel for pixel.
 *
 * FETCHED LAZILY AND NEVER AWAITED BY BOOT. It is 128 kB against a
 * projection.json of 93 kB, and projection.json is on the critical path for the
 * first paint of every view. Boot was brought from 1.6 s to ~90 ms and none of
 * that is being spent here: the views draw plain marks until this resolves and
 * upgrade themselves on the frame it lands. A failure is not an error either —
 * it just means the plain marks stay.
 */
export class CloudPictures {
  constructor(data) {
    this.size = data.size;
    this.n = data.maps.length;
    this.maps = data.maps;
    this.colourOf = data.colour_of;
    this.colours = data.colours;
    this.luts = data.luts;
    // Decoded on demand and kept: a view redraws these every frame, and
    // rebuilding 768 floats per point per frame is the kind of per-frame
    // garbage that shows up as jitter rather than as a number.
    this._cache = new Map();
  }

  /** Palette name for cloud point i — the monster's actual colour. */
  colourName(i) { return this.colours[parseInt(this.colourOf[i], 16)]; }

  /** `[r,g,b]` 0-255 of the body colour for cloud point i. Index 1 of the LUT
   *  is the body; 0 is the background, which is the same for every point and
   *  would make the whole floor one colour. */
  rgb(i) { return this.luts[parseInt(this.colourOf[i], 16)][1]; }

  /** Cloud point i as a CHW tensor in [-1,1], ready for `sprite()`. */
  tensor(i) {
    let t = this._cache.get(i);
    if (t) return t;
    const S = this.size, plane = S * S;
    const lut = this.luts[parseInt(this.colourOf[i], 16)];
    const map = this.maps[i];
    t = new Float32Array(3 * plane);
    for (let p = 0; p < plane; p++) {
      const rgb = lut[map.charCodeAt(p) - 48];
      for (let c = 0; c < 3; c++) t[c * plane + p] = rgb[c] / 127.5 - 1;
    }
    this._cache.set(i, t);
    return t;
  }
}

export function loadCloudPictures(base = './weights/') {
  return fetch(base + 'cloud.json')
    .then((r) => (r.ok ? r.json() : null))
    .then((d) => (d ? new CloudPictures(d) : null))
    .catch(() => null);
}
