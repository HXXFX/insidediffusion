/**
 * The 2-D inspector panels.
 *
 * The step strip is the most important thing in the app. It shows that the
 * model is NOT sharpening an image — it is predicting the noise and
 * subtracting it — and that its guess at the finished picture (x0-hat) is
 * already roughly right within a few steps. That is the moment diffusion stops
 * being magic, and it is a 2-D panel rather than any of the 3-D views.
 */


import { theme, alpha, fade } from '../theme.js';
import { tensorToImageData } from '../views/draw.js';

export class StepStrip {
  constructor(root, size) {
    this.size = size;
    this.tiles = {};
    root.innerHTML = '';
    /**
     * A TWO-ROW GRID, not a row of <figure>s.
     *
     * As figures, each tile was image-plus-caption and the operators between
     * them had to be centred against the WHOLE tile — image and two lines of
     * text — which put every arrow well below the images it joins. Worse, the
     * captions are different lengths, so tiles ended up different heights and
     * the labels under them did not line up with each other either.
     *
     * With images on one grid row and captions on another, the arrows sit in
     * the image row and centre on the images by construction, and every label
     * shares one baseline no matter how long it is.
     *
     * Blurbs are short for the same reason: at 84px a tile is about 12
     * characters wide, and "its guess at the finished picture" wrapped to three
     * lines and dragged its column out of alignment with the rest.
     */
    const defs = [
      ['x', 'xt', 'the noisy image'],
      ['eps', 'e-hat', 'the noise in it'],
      ['x0', 'x0-hat', 'its guess'],
      ['next', 'xt-1', 'the next input'],
    ];
    defs.forEach(([key, name, blurb], i) => {
      if (i) {
        const ar = document.createElement('span');
        ar.className = 'arrow';
        ar.textContent = i === 2 ? '−' : '→';
        ar.title = i === 2 ? 'subtract the predicted noise' : '';
        root.appendChild(ar);
        // The caption row needs a cell under the operator or the grid's column
        // flow puts the next image in the wrong row.
        root.appendChild(document.createElement('span'));
      }
      const cv = document.createElement('canvas');
      cv.className = 'tilecv';
      cv.width = size; cv.height = size;
      const cap = document.createElement('span');
      cap.className = 'tilecap';
      cap.innerHTML = `<b>${name}</b><em>${blurb}</em>`;
      root.append(cv, cap);
      this.tiles[key] = cv.getContext('2d', { willReadFrequently: false });
    });
  }

  update(snap) {
    if (!snap) return;
    for (const key of ['x', 'eps', 'x0', 'next']) {
      const g = this.tiles[key];
      // eps has roughly unit variance while the images sit in [-1,1], so it
      // needs a gentler mapping or it clips to pure primaries and reads as
      // random confetti rather than as noise.
      const gain = key === 'eps' ? 0.26 : 0.5;
      g.putImageData(tensorToImageData(g, snap[key], this.size, gain, 0.5), 0, 0);
    }
  }
}

/** Which pixels attend to which word. Only legible because it is 16x16. */
export class AttentionPanel {
  constructor(canvas) { this.canvas = canvas; this.words = []; }

  setWords(words) { this.words = words; }

  /**
   * The row's shared image band, read from the stylesheet rather than repeated
   * here — this panel has to line its grids up with pictures laid out by CSS,
   * and two copies of that number would drift the moment one was edited.
   * Cached: it is a constant in the sheet, and `getComputedStyle` in a draw
   * called every frame is a forced style recalculation.
   */
  band() {
    if (this._band == null) {
      const host = this.canvas.closest('.inspectors');
      this._band = (host && parseFloat(getComputedStyle(host).getPropertyValue('--imgband'))) || 124;
    }
    return this._band;
  }

  draw(snap) {
    const cv = this.canvas;
    const dpr = Math.min(2, window.devicePixelRatio || 1);
    const w = cv.clientWidth, h = cv.clientHeight;
    if (cv.width !== Math.round(w * dpr)) { cv.width = Math.round(w * dpr); cv.height = Math.round(h * dpr); }
    const g = cv.getContext('2d');
    g.setTransform(dpr, 0, 0, dpr, 0, 0);
    g.clearRect(0, 0, w, h);

    const map = snap && snap.internals && snap.internals.attn8;
    if (!map) {
      g.fillStyle = alpha(theme.ink3, .8);
      g.font = '11px ui-monospace, monospace';
      g.textAlign = 'center'; g.textBaseline = 'middle';
      // The saved run has no captured activations, and telling someone to turn
      // on a panel that is visibly already on is worse than saying nothing.
      // Name the actual reason instead.
      g.fillText(snap && snap.demo
        ? 'the saved run did not record this — press Run'
        : 'turn on “cross-attention” to capture this', w / 2, h / 2);
      return;
    }

    const [hw, nTok] = map.shape;
    const side = Math.round(Math.sqrt(hw));

    /**
     * THE SAME TWO ROWS AS EVERY OTHER PANEL: pictures resting on the foot of
     * the shared band, their labels underneath. This panel used to invert that
     * — words on top in 9.5px, grids below — so the one row of the app that is
     * four panels of pictures-over-captions had a fifth reading the other way,
     * and the word telling you which map you were looking at was the smallest
     * text on screen.
     *
     * The grids are sized by WIDTH — eight have to fit across — and then capped
     * by the band so they cannot outgrow the pictures beside them. CAPPED
     * GENEROUSLY, at 92: the old cap of 56 was chosen to stop the maps eating
     * vertical space, and what it actually did was leave a third of the row
     * empty, because this panel takes the width the row has left and then
     * declined to use it. At 92 each cell is ~11px instead of ~7 and the shapes
     * are legible rather than merely present.
     */
    const BAND = this.band();
    // The baseline is 17px under the band because that is where `.tilecap`'s
    // first line lands — 9.5px text on a 1.45 line box, 7px below the pictures
    // — and these words are the same kind of caption as those, so they belong
    // on the same line. LABEL is that plus room for the descenders on `<bos>`
    // and `<eos>`; measured, the lowest ink sits 3.6px clear of the edge.
    const LABEL = 22, BASE = 17;
    const cell = Math.max(24, Math.min(92, BAND, (w - 16) / nTok - 8));
    const need = BAND + LABEL;
    if (Math.abs(need - h) > 1) {
      cv.style.height = `${need}px`;
      // The backing store was sized against the old height; redo it and bail,
      // because everything below would otherwise draw into a stale buffer.
      cv.width = Math.round(w * dpr);
      cv.height = Math.round(need * dpr);
      g.setTransform(dpr, 0, 0, dpr, 0, 0);
      g.clearRect(0, 0, w, need);
    }
    const y0 = BAND - cell;          // grids rest on the foot of the band

    for (let t = 0; t < nTok; t++) {
      const gx = 8 + t * (cell + 8);
      g.fillStyle = alpha(theme.ink2, .95);
      g.font = '10.5px ui-monospace, monospace';
      g.textAlign = 'center'; g.textBaseline = 'alphabetic';
      g.fillText(this.words[t] || '·', gx + cell / 2, BAND + BASE);

      // Normalise per token so a low-attention word still shows its SHAPE.
      // Without this the <pad> columns are uniformly dark and the panel looks
      // broken rather than empty.
      let lo = Infinity, hi = -Infinity;
      for (let i = 0; i < hw; i++) {
        const v = map.data[i * nTok + t];
        if (v < lo) lo = v; if (v > hi) hi = v;
      }
      const inv = hi > lo ? 1 / (hi - lo) : 0;
      for (let i = 0; i < hw; i++) {
        const v = (map.data[i * nTok + t] - lo) * inv;
        const px = i % side, py = (i / side) | 0;
        g.fillStyle = alpha(theme.accent, 0.06 + v * 0.9);
        g.fillRect(gx + (px * cell) / side, y0 + (py * cell) / side,
          cell / side + 0.6, cell / side + 0.6);
      }
      g.strokeStyle = alpha(theme.ink, .13);
      g.lineWidth = 1;
      g.strokeRect(gx, y0, cell, cell);
    }
  }
}

/** Where this step sits on the schedule. */
export class SchedulePanel {
  constructor(canvas, alphaBar) { this.canvas = canvas; this.alphaBar = alphaBar; }

  draw(snap) {
    const cv = this.canvas;
    const dpr = Math.min(2, window.devicePixelRatio || 1);
    const w = cv.clientWidth, h = cv.clientHeight;
    if (cv.width !== Math.round(w * dpr)) { cv.width = Math.round(w * dpr); cv.height = Math.round(h * dpr); }
    const g = cv.getContext('2d');
    g.setTransform(dpr, 0, 0, dpr, 0, 0);
    g.clearRect(0, 0, w, h);

    const padL = 34, padB = 20, W = w - padL - 12, H = h - padB - 14;
    g.strokeStyle = alpha(theme.ink, .14);
    g.lineWidth = 1;
    g.beginPath();
    g.moveTo(padL, 10); g.lineTo(padL, 10 + H); g.lineTo(padL + W, 10 + H);
    g.stroke();

    const curve = (fn, col) => {
      g.strokeStyle = col; g.lineWidth = 1.8;
      g.beginPath();
      for (let i = 0; i <= 120; i++) {
        const u = i / 120;
        const X = padL + (1 - u) * W, Y = 10 + H - fn(u) * H;
        i ? g.lineTo(X, Y) : g.moveTo(X, Y);
      }
      g.stroke();
    };
    curve((u) => this.alphaBar(u * 1000), theme.accent);
    curve((u) => Math.sqrt(1 - this.alphaBar(u * 1000)), theme.accent2);

    g.font = '9.5px ui-monospace, monospace';
    g.textAlign = 'left'; g.textBaseline = 'alphabetic';
    g.fillStyle = theme.accent; g.fillText('signal kept', padL + 8, 21);
    g.fillStyle = theme.accent2; g.fillText('noise added', padL + 8, 34);

    if (snap) {
      const u = snap.t / 1000;
      const X = padL + (1 - u) * W;
      g.strokeStyle = alpha(theme.ink, .4);
      g.setLineDash([3, 3]);
      g.beginPath(); g.moveTo(X, 10); g.lineTo(X, 10 + H); g.stroke();
      g.setLineDash([]);
      g.fillStyle = theme.ink;
      g.beginPath(); g.arc(X, 10 + H - snap.alphaBar * H, 3.2, 0, 7); g.fill();
    }

    g.fillStyle = alpha(theme.ink3, .9);
    g.textAlign = 'left'; g.fillText('all noise', padL, h - 4);
    g.textAlign = 'right'; g.fillText('clean', padL + W, h - 4);
  }
}
