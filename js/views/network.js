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
import { fit, caption, note } from './draw.js';

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

      g.restore();

      note(canvas, act ? 'Tiles are the real numbers, live.'
        : 'Turn on per-layer activations to see the tiles.');
      caption(g, w, h, snap ? `step ${snap.index + 1} / ${snap.total}` : 'ready',
        'dashed = skip connections');
    },
  };
}
