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
import { fit, caption, note, groundGrid, sprite, labelSpot } from './draw.js';

const FLOOR = -0.85;
const RISE = 2.0;
const EXTENT = 1.45;

export const id = 'funnel';
export const label = 'Funnel';

export function create(ctx) {
  const { projector, cloudPics } = ctx;
  const k = projector.k;
  const proj = new Float32Array(k);
  const cloudXY = new Float32Array(projector.cloudN * 2);
  // One scale for the whole view, from the cloud's own spread. Recomputing it
  // per frame would make the floor breathe as the cloud shrinks.
  const s = 1.15 / (projector.scale || 1);

  const nodeCache = [];

  /* THE FLOOR IS 500 REAL MONSTERS, so it should look like 500 real monsters.
     It was a field of identical grey dots, which is an accurate drawing of the
     data and reads as texture — you cannot tell it apart from a decorative
     scatter, so the claim "this is the training set" has to be taken on trust.
     Each dot is now painted its own monster's body colour, which costs nothing
     per frame and turns the floor into a map: the colour clusters that appear
     are real structure in the projection, not styling.

     Built once, on the frame the pictures arrive, because `fillStyle` wants a
     string and building 500 of them per frame is pure garbage. */
  let dotFill = null;
  function buildDotFills(pics) {
    dotFill = new Array(pics.n);
    for (let i = 0; i < pics.n; i++) {
      const [r, gg, b] = pics.rgb(i);
      // White, grey and black are real body colours in this palette and the
      // page is near-white, so the lightest of them would be an invisible dot
      // on an invisible floor. Pull anything too pale toward the ink until it
      // has something to show against the background.
      const lum = (0.299 * r + 0.587 * gg + 0.114 * b) / 255;
      const m = lum > 0.72 ? (lum - 0.72) / 0.28 * 0.55 : 0;
      const mix = (v) => Math.round(v * (1 - m) + 40 * m);
      dotFill[i] = `rgba(${mix(r)},${mix(gg)},${mix(b)},0.62)`;
    }
  }

  /** The `near` nearest training points to a floor position, by squared
   *  distance. 500 points is small enough to scan outright every frame; an
   *  index would be more code and no measurable gain at this size. */
  const nearIdx = [];
  function nearest(px, pz, near) {
    nearIdx.length = 0;
    for (let n = 0; n < projector.cloudN; n++) {
      const dx = cloudXY[n * 2] * s - px, dz = cloudXY[n * 2 + 1] * s - pz;
      const d2 = dx * dx + dz * dz;
      let at = nearIdx.length;
      while (at > 0 && nearIdx[at - 1].d2 > d2) at--;
      if (at < near) {
        nearIdx.splice(at, 0, { n, d2 });
        if (nearIdx.length > near) nearIdx.length = near;
      }
    }
    return nearIdx;
  }

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
      const pics = cloudPics && cloudPics.data;
      if (pics && !dotFill) buildDotFills(pics);
      const cur = snaps[Math.min(i, snaps.length - 1)];
      for (let n = 0; n < projector.cloudN; n++) {
        P(cloudXY[n * 2] * s, FLOOR, cloudXY[n * 2 + 1] * s, p);
        g.fillStyle = dotFill ? dotFill[n] : alpha(theme.ink3, .30);
        g.beginPath(); g.arc(p[0], p[1], dotFill ? 2.0 : 1.7, 0, 7); g.fill();
      }

      /* WHICH training pictures the run is landing among. Kept here, next to
         the cloud it describes, but DRAWN LATER — it has to sit on top of the
         trail, and it needs the head marker's screen position, which is not
         known until the path has been built. */
      const near = pics && cur
        ? (projector.project(cur.x, proj),
          nearest(proj[0] * s, proj[1] * s, 3).map((e) => e.n))
        : null;

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

        /* THE HEAD OF THE TRAIL IS THE PICTURE, not a dot.
           This view is about a state falling out of noise, and it was drawing
           that state as an amber circle — honest, and it looks nothing like the
           thing being made. `x0` is the model's current guess at the finished
           picture, real at every step, so the marker can simply BE it: at the
           top it is a smear, and by the floor it is the monster. Watching that
           happen along the trail is the view's whole claim, and it used to be
           something you had to take on trust.
           The glow rings stay — they are what makes it findable against 500
           training points at low contrast. */
        /* THE PICTURE SITS BESIDE THE POINT, NOT ON IT.
           First attempt centred the sprite on the head of the trail, which
           hid the one thing this view exists to show: exactly where the state
           is. A 30px picture over a 4px dot is an eclipse. So the dot stays,
           at full precision, and the picture hangs off it on a leader — the
           same relationship a map label has to the place it names. */
        const hd = nodeCache[nodeCache.length - 1];
        P(hd[0], hd[1], hd[2], p);

        const hx = p[0], hy = p[1];
        let ox = 0, oy = 0;
        if (cur && cur.x0) {
          const px = Math.max(20, Math.min(32, Math.min(w, h) * 0.068));
          const [sx, sy] = labelSpot(hx, hy, px, w, h);
          ox = Math.sign(sx - hx) || 1; oy = Math.sign(sy - hy) || -1;
          g.strokeStyle = alpha(theme.accent2, .55); g.lineWidth = 1;
          g.beginPath(); g.moveTo(hx, hy); g.lineTo(sx, sy); g.stroke();
          sprite(g, cur.x0, 16, sx, sy, px);
          g.strokeStyle = theme.accent2; g.lineWidth = 1.5;
          g.strokeRect(sx - px / 2 - 0.5, sy - px / 2 - 0.5, px + 1, px + 1);
        }

        /* WHAT IT LANDED AMONG — the nearest real training pictures, as a row
           hung off the marker on the side opposite the run's own picture.

           IN PLACE WAS TRIED FIRST AND DOES NOT FIT, and the reason is worth
           recording because no threshold fixes it. The whole 500-point cloud
           spans about 215 screen pixels, so the forty nearest points sit inside
           a radius of 24px — under 9px per point. Four 16px pictures cannot be
           placed in that ball at any separation; the first attempt drew them
           overlapping into one dark blob, and tightening the spacing rule
           silently reduced the feature to a single sprite (measured: one 16px
           draw per frame instead of four). The density is the constraint, not
           the layout.

           So the pictures come off the map and hang beside it, which is what a
           map legend does for the same reason. They are still THE nearest, they
           still update every step, and the marker still says exactly where. */
        if (near && near.length) {
          const tp = Math.max(13, Math.min(18, Math.min(w, h) * 0.045));
          const gap = 3, row = near.length * tp + (near.length - 1) * gap;
          // Opposite the run's own picture, so the two never collide, and
          // clamped like anything else that hangs off a moving point.
          // LEVEL with the marker, not below it. Offsetting downward put the
          // row and its caption hard against the bottom of the pane, where the
          // caption crowded the view's own note line; the trail arrives from
          // above, so the space to either side of the head is the free space.
          // The lower clamp leaves room for the caption UNDER the row, which is
          // 8px below it and would otherwise be the thing that overflows.
          let rx = hx - ox * (row / 2 + 16);
          let ry = hy;
          rx = Math.max(row / 2 + 4, Math.min(w - row / 2 - 4, rx));
          ry = Math.max(tp / 2 + 12, Math.min(h - tp / 2 - 24, ry));
          g.strokeStyle = alpha(theme.ink, .22); g.lineWidth = 1;
          g.beginPath(); g.moveTo(hx, hy); g.lineTo(rx, ry); g.stroke();
          for (let j = 0; j < near.length; j++) {
            const x = rx - row / 2 + tp / 2 + j * (tp + gap);
            sprite(g, pics.tensor(near[j]), pics.size, x, ry, tp);
            g.strokeStyle = alpha(theme.ink, .30); g.lineWidth = 1;
            g.strokeRect(x - tp / 2 - 0.5, ry - tp / 2 - 0.5, tp + 1, tp + 1);
          }
          g.textAlign = 'center'; g.fillStyle = alpha(theme.ink3, .95);
          g.fillText('nearest real pictures', rx, ry + tp / 2 + 8);
          g.textAlign = 'left';
        }

        // The marker itself, drawn last so nothing sits over it.
        g.fillStyle = alpha(theme.accent2, .16); g.beginPath(); g.arc(p[0], p[1], 13, 0, 7); g.fill();
        g.fillStyle = alpha(theme.accent2, .38); g.beginPath(); g.arc(p[0], p[1], 7, 0, 7); g.fill();
        g.fillStyle = theme.accent2; g.beginPath(); g.arc(p[0], p[1], 3.6, 0, 7); g.fill();
        g.strokeStyle = theme.panel; g.lineWidth = 1;
        g.beginPath(); g.arc(p[0], p[1], 3.6, 0, 7); g.stroke();
      }

      // Axis legend. An axis nobody can name is an axis nobody trusts.
      P(-EXTENT, FLOOR, EXTENT, p);
      P(-EXTENT, FLOOR + RISE, EXTENT, q);
      g.strokeStyle = alpha(theme.ink, .18); g.lineWidth = 1;
      g.beginPath(); g.moveTo(p[0], p[1]); g.lineTo(q[0], q[1]); g.stroke();
      g.fillStyle = alpha(theme.ink3, .95);
      g.fillText('noise', Math.max(6, q[0] - 34), q[1] - 8);
      g.fillText('clean', Math.max(6, p[0] - 32), p[1] + 9);

      /* "PCA shadow" was on this line for a while, and it is the kind of label
         that only reads if you already know the answer. What the floor IS, in
         words anyone can use: a map where similar training pictures sit near
         each other. The method that builds the map is a training-time detail;
         the property that matters for reading the view is nearness. */
      /* ONE LINE, HARD LIMIT. .pnote is nowrap + ellipsis under a 62% cap, so
         a note longer than ~52 characters is silently cut mid-sentence at
         ordinary window sizes — the first wording of this line shipped as
         "...similar monsters sit…". The budget is part of the sentence. */
      note(canvas, 'Its guess so far; similar monsters sit close below.');
      const snap = snaps[Math.min(i, snaps.length - 1)];
      caption(g, w, h, snap ? `step ${snap.index + 1} / ${snap.total}` : 'ready',
        `${projector.cloudN} training images`);
    },
  };
}
