/**
 * Shared canvas helpers for the views.
 *
 * Kept deliberately small. Anything a single view needs belongs in that view;
 * this is only the things all four would otherwise duplicate.
 */


import { theme, alpha } from '../theme.js';
/** Size the backing store to the CSS box and DPR. Returns the 2-D context and
 *  CSS-pixel dimensions, so every view can draw in CSS pixels and ignore DPR. */
export function fit(canvas) {
  const dpr = Math.min(2, window.devicePixelRatio || 1);
  const w = canvas.clientWidth || 1;
  const h = canvas.clientHeight || 1;
  const bw = Math.round(w * dpr), bh = Math.round(h * dpr);
  if (canvas.width !== bw || canvas.height !== bh) {
    canvas.width = bw; canvas.height = bh;
  }
  const g = canvas.getContext('2d');
  g.setTransform(dpr, 0, 0, dpr, 0, 0);
  g.clearRect(0, 0, w, h);
  return { g, w, h };
}

/** Corner caption: a big line and a small one, bottom right. */
export function caption(g, w, h, big, small) {
  g.textAlign = 'right';
  g.textBaseline = 'alphabetic';
  g.fillStyle = alpha(theme.ink, .86);
  g.font = '11px ui-monospace, monospace';
  g.fillText(big, w - 12, h - 22);
  if (small) {
    g.fillStyle = alpha(theme.ink3, .95);
    g.font = '9.5px ui-monospace, monospace';
    g.fillText(small, w - 12, h - 10);
  }
}

/**
 * A truthfulness label, top-left.
 *
 * Not decoration. The landscape view draws a 2-D shadow of a 768-dimensional
 * distribution; sitting in a tile beside the funnel — which IS the real run —
 * it could easily be read as depicting your image. Every spatial view states
 * what it is showing, so the honest one and the approximate one are never
 * ambiguous.
 */
/**
 * A truthfulness label — NOT drawn on the canvas.
 *
 * It writes into the pane's own label, on the line under the view's name and
 * description, because that is the block already answering "what is this?" and
 * a caveat about it is the second half of the same answer. Two previous homes
 * are worth recording as dead ends: at the top left it sat under the floating
 * view switch, and at the top right it still reached back under it once panes
 * got narrow (they are never less than half the viewport, but half of 1100px is
 * narrower than the switch). As part of the label there is no geometry to get
 * wrong at any window size — the browser stacks it.
 *
 * Being DOM also means it wraps, selects and scales with the page, and — for
 * the network view — that it does not move when the diagram is panned.
 *
 * @param canvas the pane's canvas; its parent owns the label
 */
export function note(canvas, text) {
  const el = canvas.parentElement && canvas.parentElement.querySelector('.pnote');
  // Guarded: this runs every frame, and writing identical text into the DOM 60
  // times a second invalidates layout for nothing.
  if (el && el.textContent !== text) el.textContent = text;
}

/** Ground grid on the y = floor plane, drawn in world space. */
export function groundGrid(g, cam, cx, cy, scale, floor, extent, divisions, opacity = 0.055) {
  g.strokeStyle = alpha(theme.ink, opacity);
  g.lineWidth = 1;
  const p = [0, 0, 0], q = [0, 0, 0];
  for (let i = 0; i <= divisions; i++) {
    const t = (i / divisions * 2 - 1) * extent;
    cam.project(-extent, floor, t, cx, cy, scale, p);
    cam.project(extent, floor, t, cx, cy, scale, q);
    g.beginPath(); g.moveTo(p[0], p[1]); g.lineTo(q[0], q[1]); g.stroke();
    cam.project(t, floor, -extent, cx, cy, scale, p);
    cam.project(t, floor, extent, cx, cy, scale, q);
    g.beginPath(); g.moveTo(p[0], p[1]); g.lineTo(q[0], q[1]); g.stroke();
  }
}

/** Draw a 16x16 CHW tensor in [-1,1] into an ImageData-backed canvas. */
/**
 * Draw a tensor as a small picture at a point on screen.
 *
 * WHY EVERY VIEW NOW DOES THIS. Towers was the view people connected with, and
 * the reason is not its geometry — it is that Towers colours each column by
 * `x0`, the model's current guess at the finished picture. The monster is
 * visibly IN the visualisation, so the link between "this abstract thing" and
 * "the picture I asked for" needs no explaining.
 *
 * The other three showed the same state as a dot, a marker and a row of grey
 * activation tiles, all of which are honest and none of which look like the
 * thing being made. So each one now carries the actual picture at the point
 * where it means something: the head of the funnel's trail, the sample marker
 * on the landscape, the input and output of the network.
 *
 * `x0` is a real quantity at every step, not a preview — it is what the model's
 * current noise prediction implies the finished picture is, and watching it go
 * from a smear to a monster IS the process. Nothing here is a decoration
 * pasted over the data.
 *
 * One offscreen canvas per size, cached: this runs in every view on every
 * frame, and allocating a canvas per call is the kind of per-frame garbage that
 * shows up as jitter rather than as a number.
 */
const spriteCanvas = new Map();

export function sprite(g, data, size, cx, cy, px, gain = 0.5, bias = 0.5) {
  if (!data) return;
  let off = spriteCanvas.get(size);
  if (!off) {
    off = document.createElement('canvas');
    off.width = off.height = size;
    spriteCanvas.set(size, off);
  }
  const og = off.getContext('2d');
  og.putImageData(tensorToImageData(og, data, size, gain, bias), 0, 0);
  const half = px / 2;
  g.save();
  // Nearest-neighbour, or a 16px picture blown up to 40 is a coloured blur and
  // the whole point — that this is the monster — is lost.
  g.imageSmoothingEnabled = false;
  g.drawImage(off, cx - half, cy - half, px, px);
  g.restore();
}

/**
 * Where to hang a sprite so it sits BESIDE a point and never on top of it.
 *
 * The first version offset up-and-right by a fixed amount and clamped each
 * axis into the pane. That reads correctly almost everywhere and fails exactly
 * in the corners: with the point at the top right BOTH clamps fire, the offset
 * is undone, and the sprite lands back on the thing it is labelling — the very
 * bug it was added to fix, just moved somewhere harder to notice. Measured at
 * 752x307 the gap fell from 38px to 6.7px.
 *
 * So try the four diagonals and take the first that fits whole. Only if none
 * fits does it clamp, and then it keeps the candidate that ends up FURTHEST
 * from the point rather than whichever was tried first.
 *
 * @returns [x, y] centre for the sprite
 */
export function labelSpot(x, y, px, w, h, pad = 6) {
  const half = px / 2;
  const gap = px * 0.72 + 12;
  // Up-and-right first: it is clear of the caption (bottom right) and of the
  // axis legend (left), so it is the direction that fits in the common case.
  const dirs = [[1, -1], [-1, -1], [1, 1], [-1, 1]];
  for (const [dx, dy] of dirs) {
    const sx = x + dx * gap, sy = y + dy * gap;
    if (sx - half >= pad && sx + half <= w - pad &&
        sy - half >= pad && sy + half <= h - pad) return [sx, sy];
  }
  let best = null;
  for (const [dx, dy] of dirs) {
    const sx = Math.min(w - half - pad, Math.max(half + pad, x + dx * gap));
    const sy = Math.min(h - half - pad, Math.max(half + pad, y + dy * gap));
    const d = Math.hypot(sx - x, sy - y);
    if (!best || d > best[2]) best = [sx, sy, d];
  }
  return [best[0], best[1]];
}

export function tensorToImageData(g, data, size, gain = 0.5, bias = 0.5) {
  const img = g.createImageData(size, size);
  const plane = size * size;
  for (let i = 0; i < plane; i++) {
    for (let c = 0; c < 3; c++) {
      const v = data[c * plane + i] * gain + bias;
      img.data[i * 4 + c] = v < 0 ? 0 : v > 1 ? 255 : Math.round(v * 255);
    }
    img.data[i * 4 + 3] = 255;
  }
  return img;
}
