/**
 * D — THE NETWORK.
 *
 * The U-Net itself: data flows down the encoder, across the bottleneck, back
 * up the decoder, with skip connections carrying detail across. Each block
 * shows the ACTUAL activation captured from the run — a small tile of the
 * real tensor, not an icon — so "the model is doing something here" is
 * visible rather than asserted.
 *
 * This is the only view that shows the MODEL rather than the process. It is
 * flat by design: a 2-D diagram is the honest representation of a computation
 * graph, and rotating it would add nothing.
 */


import { theme, alpha, fade } from '../theme.js';
import { fit, caption, note, sprite } from './draw.js';

export const id = 'network';
export const label = 'Network';

/** Blocks, in the order data reaches them. Ids match the `collect` keys in
 *  nn/unet.js — if you add a capture point there, add it here. */
const BLOCKS = [
  { key: 'enc16', label: '16² × 32', col: 0, row: 0, side: 'enc' },
  { key: 'enc8', label: '8² × 64', col: 0, row: 1, side: 'enc' },
  { key: 'enc4', label: '4² × 96', col: 0, row: 2, side: 'enc' },
  { key: 'mid', label: '4² × 96 + attention', col: 0.5, row: 3, side: 'mid' },
  { key: 'dec8', label: '8² × 32', col: 1, row: 1, side: 'dec' },
  { key: 'dec16', label: '16² × 32', col: 1, row: 0, side: 'dec' },
];

export function create() {
  const tile = document.createElement('canvas');
  const tg = tile.getContext('2d');

  /* ONE PULSE PER FORWARD PASS.
     The diagram was static: six boxes and some lines, identical whether the
     model was running or finished, so nothing about it said "data is moving
     through this". A free-running animation would have fixed the look and lied
     — it would imply per-layer timing the app does not measure.
     This fires once per NEW SNAPSHOT instead. A snapshot means a forward pass
     genuinely happened, so one pulse is one pass, and the ORDER it travels
     (encoder, bottleneck, decoder) is the real order. It claims nothing about
     duration; scrubbing or pausing produces no pulse because no pass occurred. */
  let lastCount = -1;
  let pulseAt = -1e9;
  const PULSE_MS = 620;

  return {
    draw(canvas, cam, snaps, i) {
      const { g, w, h } = fit(canvas);
      const snap = snaps[Math.min(i, snaps.length - 1)];
      const act = snap && snap.internals ? snap.internals : null;

      /**
       * PAN AND ZOOM, around the centre of the pane.
       *
       * The diagram is flat and stays flat — rotating a computation graph means
       * nothing — but in a quarter-pane the activation tiles are a few pixels
       * across, and this was the only view where dragging and the wheel did
       * nothing at all. That reads as a broken pane, not as a deliberate one.
       *
       * The transform wraps the DIAGRAM only. The note and the caption are
       * drawn after it is restored, so the pane's own labels stay put and stay
       * the same size however far in you have zoomed — they describe the pane,
       * not the picture inside it.
       */
      g.save();
      const z = cam.flatScale;
      g.translate(w / 2 + cam.panX, h / 2 + cam.panY);
      g.scale(z, z);
      g.translate(-w / 2, -h / 2);

      /* Layout is derived from a CONTENT BOX, not from fractions of the pane.
       *
       * Two collisions came from doing it the other way. Fixed 34 px boxes
       * with gap = h*0.185 gave a 34.8 px gap at h = 188, so every row sat on
       * the one below it. Fixing that alone then pushed the first row up under
       * the view's own note strip. Reserving the strips first and fitting the
       * rows into what is left makes both impossible by construction. */
      const ROWS = 4;
      const TOP_STRIP = 32;      // the note()
      const BOT_STRIP = 34;      // the caption()
      const availH = Math.max(60, h - TOP_STRIP - BOT_STRIP);
      const bh = Math.min(30, availH / ROWS - 6);
      const spacing = (availH - bh) / (ROWS - 1);
      const rowY = (i) => TOP_STRIP + bh / 2 + i * spacing;

      const bw = Math.min(138, w * 0.26);
      const L = Math.max(bw / 2 + 8, w * 0.25);
      const R = Math.min(w - bw / 2 - 8, w * 0.75);
      const M = w * 0.5;
      const xy = (b) => ({
        x: b.side === 'enc' ? L : b.side === 'dec' ? R : M,
        y: rowY(b.row),
      });

      // connections first, so blocks sit on top of them
      g.strokeStyle = alpha(theme.ink, .16);
      g.lineWidth = 1;
      const line = (x1, y1, x2, y2, dash) => {
        g.setLineDash(dash ? [3, 3] : []);
        g.beginPath(); g.moveTo(x1, y1); g.lineTo(x2, y2); g.stroke();
        g.setLineDash([]);
      };
      for (let n = 0; n < 2; n++) line(L, rowY(n) + bh / 2, L, rowY(n + 1) - bh / 2);
      line(L, rowY(2) + bh / 2, M - bw / 2, rowY(3));
      line(M + bw / 2, rowY(3), R, rowY(1) + bh / 2);
      line(R, rowY(1) - bh / 2, R, rowY(0) + bh / 2);
      // skips
      line(L + bw / 2, rowY(0), R - bw / 2, rowY(0), true);
      line(L + bw / 2, rowY(1), R - bw / 2, rowY(1), true);

      g.font = '10px ui-monospace, monospace';
      for (const b of BLOCKS) {
        const { x, y } = xy(b);
        const a = act ? act[b.key] : null;
        const hot = !!a;
        g.fillStyle = hot ? alpha(theme.accent, .14) : alpha(theme.ink, .04);
        g.strokeStyle = hot ? alpha(theme.accent, .6) : alpha(theme.ink, .14);
        g.beginPath();
        if (g.roundRect) g.roundRect(x - bw / 2, y - bh / 2, bw, bh, 5);
        else g.rect(x - bw / 2, y - bh / 2, bw, bh);
        g.fill(); g.stroke();

        // a real activation tile: mean over channels, greyscale
        if (a) {
          const [C, H, W] = a.shape;
          tile.width = W; tile.height = H;
          const img = tg.createImageData(W, H);
          let lo = Infinity, hi = -Infinity;
          const plane = H * W;
          const mean = new Float32Array(plane);
          for (let px = 0; px < plane; px++) {
            let s = 0;
            for (let c = 0; c < C; c++) s += a.data[c * plane + px];
            s /= C; mean[px] = s;
            if (s < lo) lo = s; if (s > hi) hi = s;
          }
          const inv = hi > lo ? 1 / (hi - lo) : 0;
          for (let px = 0; px < plane; px++) {
            const v = Math.round(((mean[px] - lo) * inv) * 255);
            img.data[px * 4] = v; img.data[px * 4 + 1] = v;
            img.data[px * 4 + 2] = Math.min(255, v + 22);
            img.data[px * 4 + 3] = 255;
          }
          tg.putImageData(img, 0, 0);
          const s = bh - 10;
          g.imageSmoothingEnabled = false;
          g.drawImage(tile, x - bw / 2 + 5, y - s / 2, s, s);
          g.imageSmoothingEnabled = true;
        }

        g.fillStyle = hot ? theme.selInk : fade(theme.ink, .45);
        g.textAlign = 'left'; g.textBaseline = 'middle';
        g.fillText(b.label, x - bw / 2 + (a ? bh + 2 : 10), y);
      }

      // Text conditioning enters at the bottleneck. Drawn ON the mid row
      // rather than near the bottom edge: in a tiled pane that strip belongs
      // to the pane's own label and to the caption.
      const ty = rowY(3);
      g.textAlign = 'right';
      g.textBaseline = 'middle';
      g.fillStyle = alpha(theme.accent2, .85);
      g.font = '9.5px ui-monospace, monospace';
      g.fillText('prompt →', M - bw / 2 - 8, ty);
      g.strokeStyle = alpha(theme.accent2, .32);
      g.setLineDash([3, 3]);
      g.beginPath(); g.moveTo(M - bw / 2 - 6, ty); g.lineTo(M - bw / 2, ty); g.stroke();
      g.setLineDash([]);

      /* WHAT GOES IN AND WHAT COMES OUT, WIRED INTO THE GRAPH.
         These first sat at the pane's edges, outside the pan/zoom transform, on
         the reasoning that they label the diagram rather than belong to it. In
         practice they read as two pictures parked beside a graph they had
         nothing to do with. They are the graph's first input and its last
         output, so they are drawn INSIDE the transform, adjacent to the blocks
         they feed and come from, with an arrow each. They pan and zoom with
         everything else because they are part of it. */
      const cur = snaps[Math.min(i, snaps.length - 1)];
      if (cur && cur.x && cur.x0) {
        const ep = Math.min(bh * 1.5, bw * 0.42);
        const arrow = (x1, x2, y) => {
          g.strokeStyle = alpha(theme.ink, .3); g.lineWidth = 1;
          g.beginPath(); g.moveTo(x1, y); g.lineTo(x2, y); g.stroke();
          const d = Math.sign(x2 - x1) * 4;
          g.beginPath(); g.moveTo(x2, y); g.lineTo(x2 - d, y - 3);
          g.lineTo(x2 - d, y + 3); g.closePath();
          g.fillStyle = alpha(theme.ink, .3); g.fill();
        };
        const end = (data, sx, sy, label, col) => {
          sprite(g, data, 16, sx, sy, ep);
          g.strokeStyle = col; g.lineWidth = 1.5;
          g.strokeRect(sx - ep / 2 - 0.5, sy - ep / 2 - 0.5, ep + 1, ep + 1);
          g.fillStyle = alpha(theme.ink3, .95);
          g.font = '9px ui-monospace, monospace';
          g.textAlign = 'center'; g.textBaseline = 'top';
          g.fillText(label, sx, sy + ep / 2 + 4);
        };
        const inX = Math.max(ep / 2 + 2, L - bw / 2 - ep / 2 - 14);
        const outX = Math.min(w - ep / 2 - 2, R + bw / 2 + ep / 2 + 14);
        end(cur.x, inX, rowY(0), 'goes in', alpha(theme.ink, .35));
        arrow(inX + ep / 2 + 3, L - bw / 2 - 2, rowY(0));
        end(cur.x0, outX, rowY(0), 'comes out', theme.accent);
        arrow(R + bw / 2 + 2, outX - ep / 2 - 3, rowY(0));
      }

      /* THE PULSE — see the note at the top of create(). */
      const reduce = window.matchMedia
        && window.matchMedia('(prefers-reduced-motion: reduce)').matches;
      /* Keyed on the NUMBER OF SNAPSHOTS, not on the index being displayed.
         Keyed on the index it also fired while scrubbing, which is a lie: no
         forward pass happens when you drag the slider over steps that already
         ran. Snapshot count only goes up when the worker has actually completed
         a pass. */
      if (snaps.length !== lastCount) {
        // Not on the first fill after a load, or replaying a saved run would
        // announce 30 passes that are not happening now.
        if (lastCount >= 0 && snaps.length === lastCount + 1) pulseAt = performance.now();
        lastCount = snaps.length;
      }
      const age = (performance.now() - pulseAt) / PULSE_MS;
      if (!reduce && age >= 0 && age <= 1) {
        // The real order data reaches the blocks, as a polyline.
        const path = [
          [L, rowY(0)], [L, rowY(1)], [L, rowY(2)],
          [M, rowY(3)], [R, rowY(1)], [R, rowY(0)],
        ];
        let total = 0;
        const seg = [];
        for (let n = 1; n < path.length; n++) {
          const d = Math.hypot(path[n][0] - path[n - 1][0], path[n][1] - path[n - 1][1]);
          seg.push(d); total += d;
        }
        // Ease out: a linear pulse reads as mechanical, and the interesting
        // part is the start of the pass rather than its tail.
        const at = (1 - Math.pow(1 - age, 2)) * total;
        let acc = 0, px = path[0][0], py = path[0][1];
        for (let n = 0; n < seg.length; n++) {
          if (at <= acc + seg[n] || n === seg.length - 1) {
            const f = Math.max(0, Math.min(1, (at - acc) / (seg[n] || 1)));
            px = path[n][0] + (path[n + 1][0] - path[n][0]) * f;
            py = path[n][1] + (path[n + 1][1] - path[n][1]) * f;
            break;
          }
          acc += seg[n];
        }
        const fade0 = 1 - age;
        g.fillStyle = alpha(theme.accent2, 0.16 * fade0);
        g.beginPath(); g.arc(px, py, 13, 0, 7); g.fill();
        g.fillStyle = alpha(theme.accent2, 0.9 * fade0);
        g.beginPath(); g.arc(px, py, 4.2, 0, 7); g.fill();
      }

      // Everything above belongs to the diagram and moves with it; everything
      // below labels the pane and must not.
      g.restore();

      note(canvas, act ? 'Tiles are the real numbers, live.'
        : 'Turn on per-layer activations to see the tiles.');
      caption(g, w, h, snap ? `step ${snap.index + 1} / ${snap.total}` : 'ready',
        'dashed = skip connections');
    },
  };
}
