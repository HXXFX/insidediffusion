/**
 * ONE TRAINING STEP, run live on the real model.
 *
 * WHY THIS EXISTS. The Making of tab could show what training produced — the
 * same ten prompts drawn at 27 points across a 30,000-step run — and could not
 * show what a training step IS. Asked what the tab taught, the honest answer
 * was "that the pictures got better", which is the one thing you could already
 * guess. The mechanism was described in a sentence and demonstrated nowhere.
 *
 * So this panel runs the loop. It takes a REAL training picture (exported as an
 * index map by train/export_monsters.py and recoloured here with the same
 * lookup table the dataset uses), corrupts it by exactly the amount the
 * schedule prescribes for the chosen noise level, and asks the loaded model
 * what noise was added. Everything on screen is measured: the corrupted
 * picture, the noise that went in, the model's answer, the error between them,
 * and the loss — the same expression train_monsters.py minimises.
 *
 * THE ONE THING IT CANNOT DO is change the weights. There is no optimiser in
 * the browser and these weights are finished. The panel says that in as many
 * words rather than letting an animation imply the model is learning while you
 * watch, which would be the same overclaiming the attention panel had to be
 * corrected for.
 */

import { theme, alpha } from '../theme.js';
import { tensorToImageData } from '../views/draw.js';

const SIZE = 16;

/** Timesteps worth landing on. The slider is discrete because the interesting
 *  range is not uniform: everything legible happens below t≈600, and above it
 *  every picture is static whatever you chose. */
const STOPS = [50, 120, 200, 300, 420, 550, 700, 850, 999];

export class TrainingStep {
  /**
   * @param els { pick, colour, noise, noiseOut, run, tiles, loss, sentence, verdict }
   * @param send  (msg) => void   posts to the worker
   */
  constructor(els, send) {
    this.els = els;
    this.send = send;
    this.data = null;
    this.at = 0;
    this.colour = 'green';
    this.ti = 3;
    this.seed = 1;
    this.busy = false;
    this.ctx = {};
  }

  async load(base = './weights/') {
    this.data = await (await fetch(`${base}examples.json`)).json();
    return this.data;
  }

  /** The manifest carries the palette the dataset was built from. */
  setPalette(manifest) {
    this.palette = manifest.palette;
    this.background = manifest.background;
    this.outline = manifest.outline;
    this.eyewhite = manifest.eyewhite;
  }

  /**
   * Rebuild a training picture from its index map.
   *
   * Identical to ShapeBank.batch in train/monster_data.py: index 0-3 selects
   * background / body colour / outline / eye white, and the result is scaled to
   * [-1, 1]. Not a likeness of a training picture — the same one, pixel for
   * pixel, which is what lets the panel claim this is really the data.
   */
  picture(i, colour) {
    const ex = this.data.examples[i];
    const lut = [this.background, this.palette[colour], this.outline, this.eyewhite];
    const x = new Float32Array(3 * SIZE * SIZE);
    for (let p = 0; p < SIZE * SIZE; p++) {
      const rgb = lut[ex.index.charCodeAt(p) - 48];
      for (let c = 0; c < 3; c++) x[c * SIZE * SIZE + p] = rgb[c] / 127.5 - 1;
    }
    return x;
  }

  /** The caption that picture was trained with — its own words, not a guess. */
  prompt(i, colour) {
    const w = this.data.examples[i].words;
    return `a ${colour} ${w[0]} with ${w[1]} eyes, ${w[2]}, ${w[3]} and ${w[4]}`;
  }

  build() {
    const { pick, colour, noise } = this.els;

    // The example picker is a strip of the real pictures rather than a
    // dropdown of words: you are choosing a picture, so you should be looking
    // at pictures.
    pick.innerHTML = '';
    this.thumbs = this.data.examples.map((ex, i) => {
      const b = document.createElement('button');
      b.className = 'ex';
      b.type = 'button';
      b.setAttribute('aria-pressed', String(i === this.at));
      b.title = ex.words.join(' · ');
      const c = document.createElement('canvas');
      c.width = c.height = SIZE;
      b.appendChild(c);
      b.onclick = () => { this.at = i; this.syncPicks(); this.run(); };
      pick.appendChild(b);
      return c.getContext('2d');
    });

    colour.innerHTML = '';
    for (const name of this.data.colours) {
      const o = document.createElement('option');
      o.value = name; o.textContent = name;
      colour.appendChild(o);
    }
    colour.value = this.colour;
    colour.onchange = () => { this.colour = colour.value; this.paintThumbs(); this.run(); };

    noise.min = '0';
    noise.max = String(STOPS.length - 1);
    noise.value = String(this.ti);
    noise.oninput = () => { this.ti = +noise.value; this.run(); };

    this.els.run.onclick = () => { this.seed++; this.run(); };

    this.paintThumbs();
    this.syncPicks();
  }

  paintThumbs() {
    this.thumbs.forEach((g, i) => {
      const x = this.picture(i, this.colour);
      g.putImageData(tensorToImageData(g, x, SIZE, 0.5, 0.5), 0, 0);
    });
  }

  syncPicks() {
    this.thumbs.forEach((g, i) =>
      g.canvas.parentElement.setAttribute('aria-pressed', String(i === this.at)));
  }

  /** Ask the worker for one step. Guarded: the model is single-threaded and a
   *  dragged slider would otherwise queue a hundred forward passes. */
  run() {
    if (!this.data || !this.palette) return;
    const t = STOPS[this.ti];
    this.els.noiseOut.textContent = `t = ${t}`;
    if (this.busy) { this.pending = true; return; }
    this.busy = true;
    this.send({
      type: 'trainStep',
      x0: this.picture(this.at, this.colour).buffer,
      prompt: this.prompt(this.at, this.colour),
      t, seed: this.seed,
    });
  }

  /** The worker's reply. */
  show(m) {
    this.busy = false;
    // Looked up from the SECTION, not from the tile row: x0-hat sits over in the
    // score column, and scoping the query to the row returned null for it and
    // threw here — after five tiles had already been drawn, so the panel looked
    // half-working rather than broken.
    const root = this.els.tiles.closest('.t2d-act') || document;
    const tile = (key, data, gain) => {
      const g = this.ctx[key] || (this.ctx[key] =
        root.querySelector(`canvas[data-k="${key}"]`).getContext('2d'));
      g.putImageData(tensorToImageData(g, new Float32Array(data), SIZE, gain, 0.5), 0, 0);
    };
    tile('x0', this.picture(this.at, this.colour), 0.5);
    tile('xt', m.xt, 0.5);
    // Noise fields have roughly unit variance while pictures sit in [-1,1], so
    // they need a gentler mapping or they clip to pure primaries and read as
    // confetti rather than as noise.
    tile('noise', m.noise, 0.26);
    tile('eps', m.eps, 0.26);
    tile('x0hat', m.x0hat, 0.5);

    /**
     * THE ERROR TILE IS AMPLIFIED, and it says so.
     *
     * At the same gain as the noise it is a flat grey square at every setting
     * of the slider — the model is wrong by about a fiftieth of what the noise
     * field spans, so the difference is real and invisible. A flat square reads
     * as a broken canvas, not as "nearly right", which is the opposite of what
     * it means. So it is scaled to fill the same range as its neighbours and
     * the factor is printed beside it; without the factor this would be the
     * one tile on the row that quietly lies about its magnitude.
     */
    const err = new Float32Array(m.err);
    let peak = 0;
    for (let i = 0; i < err.length; i++) peak = Math.max(peak, Math.abs(err[i]));
    const gain = peak > 1e-6 ? Math.min(400, 0.42 / peak) : 0.26;
    tile('err', err.buffer, gain);
    // A NON-BREAKING SPACE, not an empty string. This label is the only thing
    // on the row that appears and disappears, and the error tile's caption is
    // the tallest of the five — so at the settings where the error needed no
    // amplification the caption lost a line and the whole panel shrank 13px.
    // Shown only from 2x. `gain > 1` passed for 1.4 and printed "× 1", which
    // claims an amplification that is not one — the tile would be labelled as
    // magnified while showing essentially its true magnitude.
    const shown = Math.round(gain);
    this.els.errGain.textContent = shown >= 2 ? `× ${shown}` : ' ';

    this.els.sentence.textContent = m.sentence;
    this.els.loss.textContent = m.loss.toFixed(4);
    this.els.verdict.innerHTML = this.describe(m);
    this.drawGauge(m.loss);

    if (this.pending) { this.pending = false; this.run(); }
  }

  /**
   * What this particular step shows.
   *
   * THE COPY HERE WAS WRONG FIRST TIME and the panel caught it. It said high
   * noise was where the model struggles. Measured across the slider on the fast
   * model, one example, one seed:
   *
   *     t=50  0.0038      t=300  0.0008      t=700  0.0042
   *     t=120 0.0022      t=420  0.0005      t=850  0.0013
   *     t=200 0.0013      t=550  0.0005      t=999  0.0002
   *
   * The loss is WORST at the low end and best at the top, which is the opposite
   * of the intuition and is not a bug. At t=999 the corrupted picture is
   * essentially the noise itself — the schedule has scaled the picture away to
   * nothing — so "predict the noise" collapses into "repeat your input", and
   * any competent model scores near zero. At t=50 the noise is a faint film
   * over an intact picture and picking it out exactly is genuinely hard.
   *
   * That is worth saying out loud rather than hiding, because it is the whole
   * argument of act 3 in miniature: a low loss is not the same as a useful
   * answer. The tile to watch is the implied picture, not the number.
   */
  describe(m) {
    const pct = Math.round(Math.sqrt(1 - m.alphaBar) * 100);
    if (m.t <= 150) {
      return `<b>${pct}% noise — the hardest case.</b> A faint film over an intact
        picture, and it has to name the film exactly. Worst loss, best answer.`;
    }
    if (m.t <= 420) {
      return `<b>${pct}% noise — the useful middle.</b> Enough is gone that it must
        lean on the words, enough is left that you can check its answer.`;
    }
    if (m.t <= 700) {
      return `<b>${pct}% noise.</b> Most of the picture is gone. What comes back is
        coming from the prompt, not from what is left on screen.`;
    }
    return `<b>${pct}% noise — the easiest case.</b> What it sees IS nearly the noise,
      so answering is close to repeating its input. Best loss, useless answer —
      watch the implied picture, not the number.`;
  }

  /**
   * A bar, not a number alone: the loss means nothing to a newcomer as 0.0008.
   *
   * LOG SCALE, and that is forced rather than chosen. The honest full-scale is
   * 1.0 — the noise has unit variance, so a model that always answered zero
   * would score exactly 1 — and every real value on this slider lands between
   * 0.0002 and 0.004. On a linear axis against that reference the bar is empty
   * at every setting, which tells you the model is good and nothing else. Four
   * decades of log keep the reference honest and still show the difference
   * between one end of the slider and the other.
   */
  drawGauge(loss) {
    const c = this.els.gauge;
    if (!c) return;
    const dpr = Math.min(2, window.devicePixelRatio || 1);
    const w = c.clientWidth, h = c.clientHeight;
    c.width = Math.round(w * dpr); c.height = Math.round(h * dpr);
    const g = c.getContext('2d');
    g.setTransform(dpr, 0, 0, dpr, 0, 0);
    g.clearRect(0, 0, w, h);

    const LO = 1e-4, y = 9;
    const at = (v) => Math.max(0, Math.min(1,
      Math.log(Math.max(v, LO) / LO) / Math.log(1 / LO)));

    g.fillStyle = alpha(theme.ink, .09);
    g.fillRect(0, y - 3, w, 6);
    g.fillStyle = theme.accent;
    g.fillRect(0, y - 3, Math.max(2, w * at(loss)), 6);

    // Decade ticks, so the axis is readable as an axis and not as a mystery bar.
    g.font = '8.5px ui-monospace, monospace';
    g.textBaseline = 'alphabetic';
    g.textAlign = 'center';
    for (const v of [1e-3, 1e-2, 1e-1]) {
      const x = w * at(v);
      g.fillStyle = alpha(theme.ink, .18);
      g.fillRect(x, y - 6, 1, 12);
    }
    g.fillStyle = alpha(theme.ink3, .95);
    g.textAlign = 'left';
    g.fillText('0.0001', 0, h - 1);
    g.textAlign = 'right';
    g.fillText('1.0 — no better than answering zero', w, h - 1);
  }
}
