/**
 * C — PIXEL COLUMNS.
 *
 * One column per pixel, 256 of them. Height is how much predicted noise still
 * sits on that pixel; colour is the pixel's current best guess (x0-hat). You
 * watch noise drain out of the grid unevenly — structure resolves before
 * detail, and the background flattens long before the object does.
 *
 * This view only works because the image is 16x16. At 512x512 it would be a
 * quarter of a million columns and completely unreadable.
 *
 * Drawn as real boxes rather than thick lines: lines have no top face, so
 * there is no cue for where a column ends and the whole grid smears into one
 * mass. A top quad plus one shaded side is enough to read as solid.
 */

import { fit, caption, note } from './draw.js';

const SPAN = 1.7;
const FLOOR = -0.42;

export const id = 'voxels';
export const label = 'Voxels';

export function create(ctx) {
  const N = ctx.imgSize;
  const CELL = SPAN / N;
  const GAP = CELL * 0.12;
  const cols = new Array(N * N);
  for (let i = 0; i < cols.length; i++) cols[i] = { X: 0, Z: 0, hgt: 0, r: 0, g: 0, b: 0, z: 0 };

  return {
    draw(canvas, cam, snaps, i) {
      const { g, w, h } = fit(canvas);
      const snap = snaps[Math.min(i, snaps.length - 1)];
      if (!snap) { caption(g, w, h, 'ready', ''); return; }

      const cx = w / 2, cy = h / 2 + h * 0.09;
      const sc = Math.min(w, h) * 0.52;
      const p = [0, 0, 0];
      const plane = N * N;

      for (let y = 0; y < N; y++) {
        for (let x = 0; x < N; x++) {
          const idx = y * N + x, c = cols[idx];
          let e = 0;
          for (let ch = 0; ch < 3; ch++) { const v = snap.eps[ch * plane + idx]; e += v * v; }
          // Scale by how much noise is still present overall, so the columns
          // visibly drain rather than staying tall until the last step.
          c.hgt = Math.min(1, Math.sqrt(e / 3) * 0.5) * Math.sqrt(1 - snap.alphaBar) * 1.15;
          c.r = clamp255(snap.x0[idx] * 0.5 + 0.5);
          c.g = clamp255(snap.x0[plane + idx] * 0.5 + 0.5);
          c.b = clamp255(snap.x0[2 * plane + idx] * 0.5 + 0.5);
          c.X = (x + 0.5) * CELL - SPAN / 2;
          c.Z = (y + 0.5) * CELL - SPAN / 2;
          c.r2 = CELL / 2 - GAP;
          cam.project(c.X, FLOOR, c.Z, cx, cy, sc, p);
          c.z = p[2];
        }
      }

      const order = cols.slice().sort((a, b) => b.z - a.z);   // painter: far first
      const quad = (pts, fill) => {
        g.fillStyle = fill;
        g.beginPath();
        g.moveTo(pts[0], pts[1]);
        for (let n = 2; n < pts.length; n += 2) g.lineTo(pts[n], pts[n + 1]);
        g.closePath(); g.fill();
      };
      const a = [0, 0, 0], b = [0, 0, 0], c2 = [0, 0, 0], d = [0, 0, 0];

      for (const c of order) {
        const top = FLOOR + Math.max(c.hgt, 0.004), r = c.r2;
        const sh = (k) => `rgb(${Math.round(c.r * k)},${Math.round(c.g * k)},${Math.round(c.b * k)})`;
        cam.project(c.X - r, FLOOR, c.Z + r, cx, cy, sc, a);
        cam.project(c.X + r, FLOOR, c.Z + r, cx, cy, sc, b);
        cam.project(c.X + r, top, c.Z + r, cx, cy, sc, c2);
        cam.project(c.X - r, top, c.Z + r, cx, cy, sc, d);
        quad([a[0], a[1], b[0], b[1], c2[0], c2[1], d[0], d[1]], sh(0.42));

        cam.project(c.X + r, FLOOR, c.Z - r, cx, cy, sc, b);
        cam.project(c.X + r, top, c.Z - r, cx, cy, sc, c2);
        cam.project(c.X + r, FLOOR, c.Z + r, cx, cy, sc, a);
        cam.project(c.X + r, top, c.Z + r, cx, cy, sc, d);
        quad([a[0], a[1], b[0], b[1], c2[0], c2[1], d[0], d[1]], sh(0.30));

        cam.project(c.X - r, top, c.Z - r, cx, cy, sc, a);
        cam.project(c.X + r, top, c.Z - r, cx, cy, sc, b);
        cam.project(c.X + r, top, c.Z + r, cx, cy, sc, c2);
        cam.project(c.X - r, top, c.Z + r, cx, cy, sc, d);
        quad([a[0], a[1], b[0], b[1], c2[0], c2[1], d[0], d[1]], sh(1.0));
      }

      note(canvas, 'Height is noise left, colour is the current guess.');
      caption(g, w, h, `${Math.round(Math.sqrt(1 - snap.alphaBar) * 100)}% noise remaining`,
        `${N * N} towers`);
    },
  };
}

function clamp255(v) { return v < 0 ? 0 : v > 1 ? 255 : v * 255; }
