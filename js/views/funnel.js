/**
 * B — THE FUNNEL. The hero view.
 *
 * Height is NOISE LEVEL. The floor is the real 768-number image state squashed
 * to 2-D by PCA. The path falls from pure noise at the top and lands among the
 * real training images on the floor.
 *
 * WHY NOISE LEVEL AND NOT A THIRD PCA COMPONENT — this was measured, and the
 * first version was dead on screen. Using PC3 as the vertical axis, the state
 * barely moved for 25 steps and then jumped 5.2 units on the last one, while
 * the data cloud's radius (7.4) was the same size as the entire path length
 * (7.5). The motion was real but hidden along axes the camera could not
 * separate. Noise level is not a fitted direction at all — it is the schedule,
 * known exactly — so every axis here has a stated meaning, the descent is
 * guaranteed to cross the frame, and the clean training data genuinely belongs
 * on the floor plane because its noise level is zero.
 */


import { theme, alpha, fade } from '../theme.js';
import { fit, caption, note, groundGrid } from './draw.js';

const FLOOR = -0.85;
const RISE = 2.0;
const EXTENT = 1.45;

export const id = 'funnel';
export const label = 'Funnel';

export function create(ctx) {
  const { projector } = ctx;
  const k = projector.k;
  const proj = new Float32Array(k);
  const cloudXY = new Float32Array(projector.cloudN * 2);
  // One scale for the whole view, from the cloud's own spread. Recomputing it
  // per frame would make the floor breathe as the cloud shrinks.
  const s = 1.15 / (projector.scale || 1);

  const nodeCache = [];

  function nodeAt(snap) {
    projector.project(snap.x, proj);
    return [proj[0] * s, FLOOR + Math.sqrt(1 - snap.alphaBar) * RISE, proj[1] * s];
  }

  return {
    /** @param snaps every step so far  @param i the step being shown */
    draw(canvas, cam, snaps, i) {
      const { g, w, h } = fit(canvas);
      /**
       * PULLED BACK, and low in the frame.
       *
       * At 0.46 the top of the path — `FLOOR + RISE`, which is where the "pure
       * noise" marker and its label live — projected to about 7px from the top
       * edge of the pane, so the ring was cut in half and the label sat on the
       * border. The path is the one thing in this view guaranteed to cross the
       * whole frame vertically, so it is the thing the framing has to be chosen
       * for. 0.40 clears it with room to read the label, and the extra downward
       * offset spends the headroom where it is needed rather than splitting it
       * between top and bottom.
       */
      const cx = w / 2, cy = h / 2 + h * 0.06;
      const sc = Math.min(w, h) * 0.40;
      const P = (x, y, z, o) => cam.project(x, y, z, cx, cy, sc, o);
      const p = [0, 0, 0], q = [0, 0, 0];

      groundGrid(g, cam, cx, cy, sc, FLOOR, EXTENT, 8);

      // The training set on the floor: clean data has noise level zero, so
      // this is where it truthfully belongs.
      projector.cloudAt(1, cloudXY);
      for (let n = 0; n < projector.cloudN; n++) {
        P(cloudXY[n * 2] * s, FLOOR, cloudXY[n * 2 + 1] * s, p);
        g.fillStyle = alpha(theme.ink3, .30);
        g.beginPath(); g.arc(p[0], p[1], 1.7, 0, 7); g.fill();
      }

      // Rebuild the path cache only when steps have been added.
      nodeCache.length = 0;
      for (let n = 0; n <= i && n < snaps.length; n++) nodeCache.push(nodeAt(snaps[n]));

      // Drop lines: the second depth cue. Without them the path reads flat.
      g.strokeStyle = alpha(theme.accent, .10);
      g.lineWidth = 1;
      for (let n = 0; n < nodeCache.length; n += 2) {
        const v = nodeCache[n];
        P(v[0], v[1], v[2], p); P(v[0], FLOOR, v[2], q);
        g.beginPath(); g.moveTo(p[0], p[1]); g.lineTo(q[0], q[1]); g.stroke();
      }

      // The path, brightening as noise leaves.
      g.lineWidth = 2.4; g.lineJoin = 'round';
      for (let n = 1; n < nodeCache.length; n++) {
        const a = nodeCache[n - 1], b = nodeCache[n];
        P(a[0], a[1], a[2], p); P(b[0], b[1], b[2], q);
        g.strokeStyle = alpha(theme.accent, 0.26 + (n / nodeCache.length) * 0.62);
        g.beginPath(); g.moveTo(p[0], p[1]); g.lineTo(q[0], q[1]); g.stroke();
      }
      for (let n = 0; n < nodeCache.length; n++) {
        const v = nodeCache[n], t = n / Math.max(1, nodeCache.length - 1);
        P(v[0], v[1], v[2], p);
        g.fillStyle = alpha(theme.accent, 0.32 + t * 0.5);
        g.beginPath(); g.arc(p[0], p[1], 1.9 + t * 1.3, 0, 7); g.fill();
      }

      g.font = '9.5px ui-monospace, monospace';
      g.textBaseline = 'middle'; g.textAlign = 'left';

      if (nodeCache.length) {
        const st = nodeCache[0];
        P(st[0], st[1], st[2], p);
        g.strokeStyle = alpha(theme.ink, .34); g.lineWidth = 1;
        g.beginPath(); g.arc(p[0], p[1], 7, 0, 7); g.stroke();
        g.fillStyle = alpha(theme.ink, .66);
        g.fillText('pure noise', p[0] + 11, p[1]);

        const hd = nodeCache[nodeCache.length - 1];
        P(hd[0], hd[1], hd[2], p);
        g.fillStyle = alpha(theme.accent2, .15); g.beginPath(); g.arc(p[0], p[1], 16, 0, 7); g.fill();
        g.fillStyle = alpha(theme.accent2, .34); g.beginPath(); g.arc(p[0], p[1], 8, 0, 7); g.fill();
        g.fillStyle = theme.accent2; g.beginPath(); g.arc(p[0], p[1], 4.4, 0, 7); g.fill();
      }

      // Axis legend. An axis nobody can name is an axis nobody trusts.
      P(-EXTENT, FLOOR, EXTENT, p);
      P(-EXTENT, FLOOR + RISE, EXTENT, q);
      g.strokeStyle = alpha(theme.ink, .18); g.lineWidth = 1;
      g.beginPath(); g.moveTo(p[0], p[1]); g.lineTo(q[0], q[1]); g.stroke();
      g.fillStyle = alpha(theme.ink3, .95);
      g.fillText('noise', Math.max(6, q[0] - 34), q[1] - 8);
      g.fillText('clean', Math.max(6, p[0] - 32), p[1] + 9);

      note(canvas, 'The floor is a PCA shadow of the real state.');
      const snap = snaps[Math.min(i, snaps.length - 1)];
      caption(g, w, h, snap ? `step ${snap.index + 1} / ${snap.total}` : 'ready',
        `${projector.cloudN} training images`);
    },
  };
}
