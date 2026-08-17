/**
 * A — THE PROBABILITY LANDSCAPE.
 *
 * How likely each point in the projected data space is, as a surface. At high
 * noise it is one smooth hill; as noise drains out it sharpens into the real,
 * lumpy distribution. The current sample is marked on it.
 *
 * THIS SURFACE IS EXACT, NOT AN ILLUSTRATION. Because the PCA basis is
 * orthonormal (measured error 1.8e-15), projecting the forward process gives
 *
 *     P'x_t = sqrt(ab)*(P'x0) + sqrt(1-ab)*N(0, I)
 *
 * so the projected distribution at step t is a Gaussian mixture centred on the
 * training points scaled by sqrt(ab), every component with variance (1-ab).
 * That is what is drawn. An earlier design used an invented 2-D mixture, which
 * looked the same and taught something false.
 *
 * It is still a SHADOW: two of 768 dimensions, about 27% of the variance. The
 * view says so on its face, because sitting beside the funnel it could
 * otherwise be read as depicting your image.
 *
 * PERFORMANCE NOTE: the mixture has 500 components and the grid has ~2.3k
 * cells, so evaluating it directly is over a million exp() calls per frame.
 * Every component shares one variance, so the mixture is exactly a Gaussian
 * BLUR of the splatted points — done separably, that is a few hundred thousand
 * cheap operations instead. Do not replace this with the direct sum.
 */


import { theme, alpha, fade } from '../theme.js';
import { fit, caption, note, sprite } from './draw.js';

const G = 44;            // surface grid resolution
const SPLAT = 72;        // density accumulation grid
const RANGE = 1.35;      // world extent of the plane, in units of cloud scale
const MAX_R = 26;        // blur radius cap; beyond this the surface is flat anyway
const MIN_SIGMA_CELLS = 1.1;   // see buildDensity — tied to SPLAT, not arbitrary

export const id = 'landscape';
export const label = 'Landscape';

export function create(ctx) {
  const { projector, cloudPics } = ctx;
  const cloudXY = new Float32Array(projector.cloudN * 2);
  const acc = new Float32Array(SPLAT * SPLAT);
  const tmp = new Float32Array(SPLAT * SPLAT);
  const dens = new Float32Array((G + 1) * (G + 1));
  const proj = new Float32Array(projector.k);
  const cellsPerUnit = SPLAT / (2 * RANGE);
  let floored = false;      // true when the true sigma is finer than the grid

  function buildDensity(alphaBar) {
    acc.fill(0);
    projector.cloudAt(alphaBar, cloudXY);
    const s = 1 / (projector.scale || 1);
    // splat (bilinear, so the surface does not shimmer as points move)
    for (let i = 0; i < projector.cloudN; i++) {
      const gx = (cloudXY[i * 2] * s + RANGE) * cellsPerUnit;
      const gy = (cloudXY[i * 2 + 1] * s + RANGE) * cellsPerUnit;
      const x0 = Math.floor(gx), y0 = Math.floor(gy);
      const fx = gx - x0, fy = gy - y0;
      for (let dy = 0; dy < 2; dy++) {
        const y = y0 + dy; if (y < 0 || y >= SPLAT) continue;
        const wy = dy ? fy : 1 - fy;
        for (let dx = 0; dx < 2; dx++) {
          const x = x0 + dx; if (x < 0 || x >= SPLAT) continue;
          acc[y * SPLAT + x] += wy * (dx ? fx : 1 - fx);
        }
      }
    }

    /* separable Gaussian blur; sigma = sqrt(1 - alphaBar), in world units
     *
     * FLOORED AT THE GRID'S OWN RESOLUTION, and that number is derived rather
     * than picked. As alphaBar approaches 1 the true density becomes a sum of
     * 500 delta functions — correct, and on screen it is a field of spikes
     * that conveys nothing. One grid cell is the finest feature this surface
     * can represent at all, so blurring below that is not showing more detail,
     * it is showing sampling artefacts. The floor is therefore tied to SPLAT,
     * not to a constant: change the grid and it follows.
     */
    const sigmaWorld = Math.sqrt(1 - alphaBar) / (projector.scale || 1);
    const sigmaCells = sigmaWorld * cellsPerUnit;
    floored = sigmaCells < MIN_SIGMA_CELLS;
    const sigma = Math.max(MIN_SIGMA_CELLS, sigmaCells);
    const r = Math.min(MAX_R, Math.max(1, Math.ceil(sigma * 2.5)));
    const kern = new Float32Array(2 * r + 1);
    let ksum = 0;
    for (let i = -r; i <= r; i++) { const v = Math.exp(-(i * i) / (2 * sigma * sigma)); kern[i + r] = v; ksum += v; }
    for (let i = 0; i < kern.length; i++) kern[i] /= ksum;

    for (let y = 0; y < SPLAT; y++) {
      for (let x = 0; x < SPLAT; x++) {
        let s2 = 0;
        for (let i = -r; i <= r; i++) {
          const xx = x + i;
          if (xx >= 0 && xx < SPLAT) s2 += acc[y * SPLAT + xx] * kern[i + r];
        }
        tmp[y * SPLAT + x] = s2;
      }
    }
    for (let y = 0; y < SPLAT; y++) {
      for (let x = 0; x < SPLAT; x++) {
        let s2 = 0;
        for (let i = -r; i <= r; i++) {
          const yy = y + i;
          if (yy >= 0 && yy < SPLAT) s2 += tmp[yy * SPLAT + x] * kern[i + r];
        }
        acc[y * SPLAT + x] = s2;
      }
    }

    // resample onto the surface grid and normalise to a constant visual height
    let peak = 0;
    for (let j = 0; j <= G; j++) {
      for (let i = 0; i <= G; i++) {
        const gx = (i / G) * (SPLAT - 1), gy = (j / G) * (SPLAT - 1);
        const x0 = Math.min(SPLAT - 2, Math.floor(gx)), y0 = Math.min(SPLAT - 2, Math.floor(gy));
        const fx = gx - x0, fy = gy - y0;
        const v = acc[y0 * SPLAT + x0] * (1 - fx) * (1 - fy)
          + acc[y0 * SPLAT + x0 + 1] * fx * (1 - fy)
          + acc[(y0 + 1) * SPLAT + x0] * (1 - fx) * fy
          + acc[(y0 + 1) * SPLAT + x0 + 1] * fx * fy;
        dens[j * (G + 1) + i] = v;
        if (v > peak) peak = v;
      }
    }
    return peak || 1;
  }

  /**
   * The hills, and which monster makes each one.
   *
   * WHY. The surface is exact and it already moves — it rebuilds every frame
   * and sharpens as noise drains — but every hill looked the same as every
   * other hill, so "where believable pictures live" was a caption rather than
   * something on screen. A peak in this surface is a place where many training
   * images sit close together, which means it has a LOOK, and the training
   * image nearest its summit is a fair representative of it.
   *
   * That also makes the view's motion legible for the first time: at high noise
   * there is one broad hill and one face on it, and as the noise drains the hill
   * splits into several, each acquiring its own. The splitting was always drawn;
   * now it can be read.
   *
   * Local maxima on the (G+1)^2 grid, non-max-suppressed by `sepCells` so two
   * summits of the same hill do not both count, and thresholded so the flats
   * never sprout a face.
   */
  function peaks(peak, want, sepCells) {
    const out = [];
    for (let j = 1; j < G; j++) {
      for (let i2 = 1; i2 < G; i2++) {
        const v = dens[j * (G + 1) + i2];
        if (v < peak * 0.42) continue;
        let top = true;
        for (let dj = -1; dj <= 1 && top; dj++) {
          for (let di = -1; di <= 1; di++) {
            if (!di && !dj) continue;
            if (dens[(j + dj) * (G + 1) + i2 + di] > v) { top = false; break; }
          }
        }
        if (top) out.push({ i: i2, j, v });
      }
    }
    out.sort((a, b) => b.v - a.v);
    const kept = [];
    for (const c of out) {
      if (kept.some((o) => Math.hypot(o.i - c.i, o.j - c.j) < sepCells)) continue;
      kept.push(c);
      if (kept.length === want) break;
    }
    return kept;
  }

  /** The training point nearest a grid position, in the CURRENT (noised) cloud
   *  — `buildDensity` has already filled cloudXY at this alphaBar, so the
   *  answer matches the surface actually on screen rather than the clean data. */
  function nearestCloud(wx, wz) {
    const sc2 = 1 / (projector.scale || 1);
    let best = -1, bd = 1e9;
    for (let n = 0; n < projector.cloudN; n++) {
      const dx = cloudXY[n * 2] * sc2 - wx, dz = cloudXY[n * 2 + 1] * sc2 - wz;
      const d = dx * dx + dz * dz;
      if (d < bd) { bd = d; best = n; }
    }
    return best;
  }

  return {
    draw(canvas, cam, snaps, i) {
      const { g, w, h } = fit(canvas);
      const snap = snaps[Math.min(i, snaps.length - 1)];
      const ab = snap ? snap.alphaBar : 0;
      const peak = buildDensity(ab);

      const cx = w / 2, cy = h / 2 + h * 0.10;
      const sc = Math.min(w, h) * 0.60;
      const HEIGHT = 0.70;
      const p = [0, 0, 0];
      const at = (i2, j2, o) => {
        const X = (i2 / G) * 2 - 1, Z = (j2 / G) * 2 - 1;
        const d = dens[j2 * (G + 1) + i2] / peak;
        cam.project(X * RANGE, d * HEIGHT - 0.34, Z * RANGE, cx, cy, sc, o);
        return d;
      };

      // painter's algorithm over quads
      const quads = [];
      const a = [0, 0, 0], b = [0, 0, 0], c = [0, 0, 0], dd = [0, 0, 0];
      for (let j = 0; j < G; j++) {
        for (let i2 = 0; i2 < G; i2++) {
          const h1 = at(i2, j, a), h2 = at(i2 + 1, j, b);
          const h3 = at(i2 + 1, j + 1, c), h4 = at(i2, j + 1, dd);
          quads.push({
            ax: a[0], ay: a[1], bx: b[0], by: b[1],
            cx2: c[0], cy2: c[1], dx: dd[0], dy: dd[1],
            z: (a[2] + b[2] + c[2] + dd[2]) * 0.25,
            hgt: (h1 + h3) * 0.5,
          });
        }
      }
      quads.sort((q1, q2) => q2.z - q1.z);
      for (const q of quads) {
        const t = q.hgt < 0 ? 0 : q.hgt > 1 ? 1 : q.hgt;
        // Low ground is the page, high ground is the accent. Running the ramp
        // FROM the background rather than from an arbitrary dark blue is what
        // makes the flats read as "nothing here" instead of as a solid object:
        // the surface only becomes visible where the data actually is.
        g.fillStyle = fade(theme.accent, 1 - t, theme.viewBg);
        g.beginPath();
        g.moveTo(q.ax, q.ay); g.lineTo(q.bx, q.by);
        g.lineTo(q.cx2, q.cy2); g.lineTo(q.dx, q.dy);
        g.closePath(); g.fill();
        g.strokeStyle = alpha(theme.ink, 0.06 + t * 0.14);
        g.lineWidth = 0.5; g.stroke();
      }

      /* A FACE ON EACH HILL. Drawn after the surface and before the sample, so
         the run's own marker still wins where they overlap. */
      const pics = cloudPics && cloudPics.data;
      if (pics) {
        const tp = Math.max(14, Math.min(22, Math.min(w, h) * 0.05));
        for (const c of peaks(peak, 3, G * 0.22)) {
          const X = (c.i / G) * 2 - 1, Z = (c.j / G) * 2 - 1;
          const n = nearestCloud(X * RANGE, Z * RANGE);
          if (n < 0) continue;
          cam.project(X * RANGE, (c.v / peak) * HEIGHT - 0.34, Z * RANGE, cx, cy, sc, p);
          // Sat ON the summit the picture hid the peak it was labelling, the
          // same mistake the marker made. It floats just above instead, with a
          // hairline down to the point it belongs to.
          const sy = p[1] - tp / 2 - 9;
          g.strokeStyle = alpha(theme.ink, .22); g.lineWidth = 1;
          g.beginPath(); g.moveTo(p[0], p[1]); g.lineTo(p[0], sy + tp / 2); g.stroke();
          sprite(g, pics.tensor(n), pics.size, p[0], sy, tp);
          g.strokeStyle = alpha(theme.ink, .34); g.lineWidth = 1;
          g.strokeRect(p[0] - tp / 2 - 0.5, sy - tp / 2 - 0.5, tp + 1, tp + 1);
        }
      }

      // where THIS sample currently sits on the surface
      if (snap) {
        projector.project(snap.x, proj);
        const s = 1 / (projector.scale || 1);
        const X = Math.max(-1, Math.min(1, (proj[0] * s) / RANGE));
        const Z = Math.max(-1, Math.min(1, (proj[1] * s) / RANGE));
        const gi = Math.round(((X + 1) / 2) * G), gj = Math.round(((Z + 1) / 2) * G);
        const d = dens[gj * (G + 1) + gi] / peak;
        cam.project(X * RANGE, d * HEIGHT - 0.34 + 0.05, Z * RANGE, cx, cy, sc, p);
        /* THE MARKER IS THE PICTURE. This view answers "where does your monster
           sit among believable pictures", and it was answering with an amber
           dot — so the "your" in "your sample" was a label rather than
           something you could see. Drawing `x0` there makes the claim
           self-evident: that is your monster, and that is where it sits. */
        /* PICTURE ABOVE, MARKER ON THE SURFACE. Where the sample sits on the
           map is the entire answer this view gives, so the picture must not
           cover it — it stands on a stem well clear, and the marker underneath
           keeps full precision. */
        const px = Math.max(18, Math.min(28, Math.min(w, h) * 0.058));
        if (snap.x0) {
          /* The stem stays vertical � "standing on the spot" is the whole read,
             and a diagonal leader would lose it. But a peak near the top of the
             frame put the picture off the top edge, so when there is no room
             above it stands BELOW instead. Flipping keeps the stem; clamping
             would have slid the picture down onto the marker. */
          const up = p[1] - px - 16;
          const below = up - px / 2 < 6;
          const sy = below ? p[1] + px + 16 : up;
          const stemEnd = below ? sy - px / 2 : sy + px / 2;
          g.strokeStyle = alpha(theme.accent2, .55); g.lineWidth = 1;
          g.beginPath(); g.moveTo(p[0], stemEnd);
          g.lineTo(p[0], p[1] + (below ? 4 : -4)); g.stroke();
          sprite(g, snap.x0, 16, p[0], sy, px);
          g.strokeStyle = theme.accent2; g.lineWidth = 1.5;
          g.strokeRect(p[0] - px / 2 - 0.5, sy - px / 2 - 0.5, px + 1, px + 1);
        }
        g.fillStyle = alpha(theme.accent2, .18); g.beginPath(); g.arc(p[0], p[1], 11, 0, 7); g.fill();
        g.fillStyle = theme.accent2; g.beginPath(); g.arc(p[0], p[1], 3.6, 0, 7); g.fill();
        g.strokeStyle = theme.panel; g.lineWidth = 1;
        g.beginPath(); g.arc(p[0], p[1], 3.6, 0, 7); g.stroke();
        g.font = '9.5px ui-monospace, monospace';
        g.textAlign = 'left'; g.textBaseline = 'middle';
        g.fillStyle = alpha(theme.accent2, .95);
        // "your monster", the app's own word for it — "sample" is ours.
        g.fillText('your monster', p[0] + 9, p[1]);
      }

      /* "The training data, 21% of it" was accurate and unreadable — the
         percentage is explained variance, which nobody should need to know to
         read a hill. What a reader actually needs: taller = more pictures of
         that kind, and this maps the DATA, not their monster. That second half
         directly answers the natural misreading of this view — "why isn't my
         result on the highest mountain?" — the tallest hill is only the most
         common kind of monster, and yours only needs to land on a hill. */
      // One line, ~58 chars max — see the note budget comment in funnel.js.
      note(canvas, 'Taller = more of that kind — the data, not yours.');
      caption(g, w, h, `noise σ = ${Math.sqrt(1 - ab).toFixed(3)}`,
        floored ? 'smoothed — the real peaks are sharper'
          : 'height = how common right now');
    },
  };
}
