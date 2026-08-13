/**
 * The Training tab: the real run, replayed.
 *
 * WHAT THIS REPLACED, AND WHY. The first version of this tab trained a second,
 * much smaller model live in the browser — a genuinely nice demonstration, and
 * the wrong answer to the question people actually arrive with. Asked what the
 * tab was for, the first reader said: "Is the training tab about showing how the
 * monster model is being trained?" It was not. It was a different model
 * learning a cloud of dots, and no amount of explanatory text was going to stop
 * that being the natural reading of a tab labelled Training in an app about a
 * monster model.
 *
 * So now it shows exactly what it appeared to promise. Every sprite here was
 * drawn by the REAL model, at that point in its real 30,000-step run, from the
 * same ten prompts each time. Step 50 is genuinely what it could do after fifty
 * gradient steps.
 *
 * The live 2-D trainer is kept in js/train2d/ — it works, and the loss-versus-
 * samples lesson it taught is written into the learnings document — but it is
 * no longer wired into the app.
 */

import { theme } from '../theme.js';

const SIZE = 16;

/**
 * The phases the run passes through.
 *
 * The boundaries are READ OFF THE REPLAY, not invented: this model is drawing
 * coloured blobs by a few hundred steps, recognisable bodies by a thousand and
 * the right monsters by about three thousand, after which 27,000 steps of
 * sharpening produce changes you have to look for. They are the same thresholds
 * `describe()` uses, so the bar and the sentence cannot disagree about where
 * you are; the final mark is called finished by both, separately.
 */
const PHASES = [
  { to: 0, name: 'Noise', note: 'random weights' },
  { to: 200, name: 'Colour', note: 'hue before shape' },
  { to: 1000, name: 'Bodies', note: 'shape arrives' },
  { to: 3000, name: 'Monsters', note: 'parts in place' },
  { to: Infinity, name: 'Sharpening', note: 'edges only' },
];

export class TrainingReplay {
  /**
   * @param els  { strip, loss, scrub, end, blurb, phases, facts }
   */
  constructor(els) {
    this.els = els;
    this.data = null;
    this.at = 0;
    this.cells = [];
    this.manifest = null;
  }

  /** Load one run's history: a JSON index plus a flat sprite blob. */
  async load(run, base = './weights/training/') {
    const meta = await (await fetch(`${base}${run}.json`)).json();
    const buf = await (await fetch(`${base}${run}.bin`)).arrayBuffer();
    const [marks, n] = meta.shape;
    this.data = {
      run, meta, marks, n,
      px: new Uint8Array(buf),
      stride: n * SIZE * SIZE * 3,
      cell: SIZE * SIZE * 3,
    };
    this.measureChange();
    this.build();
    this.buildPhases();
    this.buildFacts();
    // Open at the END. The first frame of a training run is noise, and landing
    // on noise reads as "this is broken" rather than "this is step zero".
    this.setIndex(marks - 1);
    return meta;
  }

  build() {
    const { meta, n } = this.data;
    const strip = this.els.strip;
    strip.innerHTML = '';
    this.cells = [];
    for (let i = 0; i < n; i++) {
      const fig = document.createElement('figure');
      fig.className = 't2d-cell';
      const c = document.createElement('canvas');
      c.width = c.height = SIZE;
      const cap = document.createElement('figcaption');
      const p = meta.prompts[i];
      cap.textContent = `${p[0]} ${p[1]}`;
      cap.title = `${p[0]} ${p[1]} / ${p[2]} eyes / ${p[3]} / ${p[4]} / ${p[5]}`;
      fig.append(c, cap);
      strip.appendChild(fig);
      this.cells.push(c.getContext('2d'));
    }
    this.els.scrub.max = String(this.data.marks - 1);
    this.els.end.textContent = `step ${meta.steps[meta.steps.length - 1].toLocaleString()}`;
  }

  /**
   * The phase bar.
   *
   * Segments are sized by how many MARKS fall in each phase, not by how many
   * steps. The marks are log-spaced, so by step count "Sharpening" is 90% of
   * the run and everything worth watching is a sliver at the left edge — and,
   * worse, the marker would not track the slider, which moves in marks. Sized
   * by marks, the bar is a picture of the recording rather than of the clock.
   */
  /**
   * The phase bar — and it is NAVIGATION, not decoration.
   *
   * It shipped as five highlighted `div`s, which is the worst of both: it looks
   * exactly like a row of buttons, so people try to click it, and nothing
   * happens. Anything that reads as pressable either has to be pressable or has
   * to stop looking like it. These are the more useful of the two: the phases
   * are the only landmarks in a 27-mark run, and jumping to "when bodies
   * appear" is the thing you actually want to do after watching it once.
   *
   * Each button jumps to the FIRST mark of its phase — the moment the change
   * happens, not somewhere in the middle of it.
   */
  buildPhases() {
    const host = this.els.phases;
    if (!host) return;
    host.innerHTML = '';
    this.segs = [];
    const { steps } = this.data.meta;
    let from = 0;
    for (const p of PHASES) {
      const upto = steps.filter((s) => s <= p.to).length;
      const n = upto - from;
      if (n <= 0) { this.segs.push({ ...p, n: 0 }); continue; }
      const el = document.createElement('button');
      el.type = 'button';
      el.className = 'phase';
      el.style.flexGrow = String(n);
      el.setAttribute('aria-pressed', 'false');
      el.innerHTML = `<b>${p.name}</b><em>${p.note}</em>`
        + `<i class="pstep">step ${steps[from].toLocaleString()}</i>`;
      const mark = from;
      el.onclick = () => { this.stop(); this.setIndex(mark); this.onJump && this.onJump(); };
      host.appendChild(el);
      this.segs.push({ ...p, n, el, mark });
      from = upto;
    }
  }

  /** Which phase a step belongs to. Same thresholds as `describe()`. */
  phaseOf(step) { return PHASES.findIndex((p) => step <= p.to); }

  /**
   * The facts rail — everything the recording and the manifest can prove.
   *
   * Every number here is read from a file: the step count and the loss come
   * from the history, the parameter count and download size from the model's
   * own manifest. Nothing is typed in, so nothing can quietly become false the
   * next time a model is retrained. That rules out some things worth saying —
   * how long the run took is not recorded anywhere the app can see — and they
   * are left out rather than approximated.
   */
  buildFacts() {
    const host = this.els.facts;
    if (!host) return;
    const { meta, marks, n } = this.data;
    const m = this.manifest;
    const rows = [
      ['Steps trained', meta.steps.at(-1).toLocaleString()],
      ['Recorded at', `${marks} points`],
      ['Probe prompts', `${n}, fixed`],
    ];
    if (m) {
      rows.push(['Parameters', m.params.toLocaleString()]);
      rows.push(['Download', `${(m.total_bytes / 1e6).toFixed(1)} MB`]);
      rows.push(['Picture size', `${m.arch.img_size} × ${m.arch.img_size}`]);
    }
    host.innerHTML = '';
    const add = (k, v, cls) => {
      const row = document.createElement('div');
      row.className = `fact${cls ? ` ${cls}` : ''}`;
      row.innerHTML = `<dt>${k}</dt><dd>${v}</dd>`;
      host.appendChild(row);
      return row.querySelector('dd');
    };
    const head = document.createElement('h4');
    head.textContent = 'The run';
    host.appendChild(head);
    for (const [k, v] of rows) add(k, v);

    const head2 = document.createElement('h4');
    head2.textContent = 'Here';
    host.appendChild(head2);
    this.liveStep = add('Step', '—', 'live');
    this.liveLoss = add('Training loss', '—', 'live');
    this.livePhase = add('Phase', '—', 'live');

    const foot = document.createElement('p');
    foot.className = 'factnote';
    // Worth saying because it is the unusual part, and because it is what makes
    // the accuracy checks possible at all: there is a known right answer.
    foot.textContent = 'Generated, not collected — so every prompt has a known '
      + 'right answer to measure against.';
    host.appendChild(foot);
  }

  /**
   * How much the pictures changed between one mark and the next.
   *
   * Mean absolute difference over all ten sprites, in bytes. It is the second
   * curve on the loss chart.
   *
   * IT WAS ADDED TO PROVE THE OPPOSITE OF WHAT IT SHOWS. The intent was to
   * demonstrate that the loss misleads — that it flattens while the pictures go
   * on changing. Measured on the fast run, as fractions of their own starting
   * values:
   *
   *              step 200   step 1,000   step 30,000
   *     loss       0.159       0.0435       0.0059
   *     change     0.157       0.0523       0.0051
   *
   * They track each other almost exactly. On this run the loss is a perfectly
   * fair proxy for whether the pictures are still moving, and the planned
   * argument was wrong. What both curves DO show is the thing worth saying: 84%
   * of the fall happens in the first 0.7% of the run, at a point where the
   * output is coloured smears, and a curve that looks flat at step 1,000 still
   * has a factor of seven to go — which is the entire difference between a blob
   * and a monster. The measurement is kept because it corroborates rather than
   * because it contradicts.
   */
  measureChange() {
    const { px, stride, marks } = this.data;
    const out = [0];
    for (let m = 1; m < marks; m++) {
      let s = 0;
      const a = (m - 1) * stride, b = m * stride;
      for (let i = 0; i < stride; i++) s += Math.abs(px[a + i] - px[b + i]);
      out.push(s / stride);
    }
    this.change = out;
  }

  /** The recorded loss at or just before a step. Nearest sample, not a fit. */
  lossAt(step) {
    const v = this.data.meta.loss || [];
    let best = null;
    for (const p of v) { if (p.step <= step) best = p; else break; }
    return best;
  }

  /**
   * Play through the run.
   *
   * Slower at the START, because that is where everything happens: the marks
   * are log-spaced, so a constant frame-per-mark rate races through the first
   * thousand steps — the only part where the picture changes — and then dwells
   * on twenty thousand steps of nothing. Holding each early mark longer makes
   * the playback show what the recording actually contains.
   */
  play(onTick) {
    if (this.timer) return this.stop();
    if (this.at >= this.data.marks - 1) this.setIndex(0);
    const step = () => {
      if (this.at >= this.data.marks - 1) { this.stop(); onTick && onTick(false); return; }
      this.setIndex(this.at + 1);
      onTick && onTick(true);
      const frac = this.at / (this.data.marks - 1);
      this.timer = setTimeout(step, 420 - 260 * frac);
    };
    this.timer = setTimeout(step, 260);
    onTick && onTick(true);
  }

  stop() {
    clearTimeout(this.timer);
    this.timer = null;
  }

  get playing() { return !!this.timer; }

  setIndex(i) {
    if (!this.data) return;
    const { px, stride, cell, n, meta } = this.data;
    this.at = Math.max(0, Math.min(this.data.marks - 1, i | 0));
    const base = this.at * stride;
    for (let k = 0; k < n; k++) {
      const g = this.cells[k];
      const img = g.createImageData(SIZE, SIZE);
      const off = base + k * cell;
      for (let p = 0; p < SIZE * SIZE; p++) {
        img.data[p * 4] = px[off + p * 3];
        img.data[p * 4 + 1] = px[off + p * 3 + 1];
        img.data[p * 4 + 2] = px[off + p * 3 + 2];
        img.data[p * 4 + 3] = 255;
      }
      g.putImageData(img, 0, 0);
    }
    this.els.scrub.value = String(this.at);
    const step = meta.steps[this.at];
    this.els.blurb.innerHTML = this.describe(step);
    this.mark(step);
    this.drawLoss();
  }

  /** Move the phase bar and the live readouts to this step. */
  mark(step) {
    const i = this.phaseOf(step);
    // The last mark is the end of the run, not more sharpening — `describe()`
    // calls it out separately and so does this, or the bar would still read
    // "Sharpening" under a sentence that says the model is finished.
    const done = this.at >= this.data.marks - 1;
    if (this.segs) {
      this.segs.forEach((s, k) => {
        if (!s.el) return;
        s.el.classList.toggle('on', k === i);
        s.el.classList.toggle('done', done && k === i);
        s.el.setAttribute('aria-pressed', String(k === i));
      });
    }
    if (this.liveStep) this.liveStep.textContent = step.toLocaleString();
    if (this.livePhase) this.livePhase.textContent = done ? 'Finished' : PHASES[i].name;
    if (this.liveLoss) {
      const p = this.lossAt(step);
      this.liveLoss.textContent = p ? p.loss.toFixed(4) : '—';
    }
  }

  /**
   * What is worth saying at this point in the run.
   *
   * The thresholds are read off the actual replay rather than invented: this
   * model is drawing recognisable coloured blobs by a few hundred steps and
   * clean monsters by about three thousand, and the remaining 27,000 steps
   * sharpen edges. Saying "watch it learn" over a picture that stopped changing
   * 20,000 steps ago would be the same overclaiming the attention panel had.
   */
  describe(step) {
    const s = step.toLocaleString();
    if (step === 0) return `<b>Step 0.</b> Random weights. This is noise, asked for a monster.`;
    if (step <= 200) return `<b>Step ${s}.</b> Colour first: right hue, roughly the right place, no shape.`;
    if (step <= 1000) return `<b>Step ${s}.</b> Bodies appear. Eyes, horns and legs are still smears.`;
    if (step <= 3000) return `<b>Step ${s}.</b> The right monsters. Most of what it will ever know, it knows here.`;
    if (step < 30000) return `<b>Step ${s}.</b> Edges sharpening. Small changes, mostly at the outlines.`;
    return `<b>Step ${s} — finished.</b> This is the model the Make tab runs.`;
  }

  drawLoss() {
    const c = this.els.loss;
    if (!c) return;
    const dpr = Math.min(2, window.devicePixelRatio || 1);
    const r = c.getBoundingClientRect();
    c.width = Math.max(1, Math.round(r.width * dpr));
    c.height = Math.max(1, Math.round(r.height * dpr));
    const g = c.getContext('2d'), W = c.width, H = c.height, s = dpr;
    g.clearRect(0, 0, W, H);

    const v = this.data.meta.loss || [];
    if (v.length < 2) return;
    const pad = 6 * s, top = 16 * s, bot = H - 14 * s;
    // Log scale: the loss falls by an order of magnitude in the first few
    // hundred steps, so a linear axis is one vertical line and a flat tail.
    const ls = v.map((p) => Math.log(Math.max(p.loss, 1e-4)));
    const lo = Math.min(...ls), hi = Math.max(...ls);
    const maxStep = v[v.length - 1].step || 1;
    const X = (st) => pad + (st / maxStep) * (W - pad * 2);
    const Y = (l) => bot - ((l - lo) / Math.max(hi - lo, 1e-6)) * (bot - top);

    g.strokeStyle = theme.line;
    g.lineWidth = 1 * s;
    g.beginPath(); g.moveTo(pad, bot); g.lineTo(W - pad, bot); g.stroke();

    g.strokeStyle = theme.accent;
    g.lineWidth = 1.6 * s;
    g.beginPath();
    v.forEach((p, i) => (i ? g.lineTo(X(p.step), Y(ls[i])) : g.moveTo(X(p.step), Y(ls[i]))));
    g.stroke();

    /**
     * THE SECOND CURVE: how much the pictures changed between marks.
     *
     * Drawn on its own log scale over the same axis — see measureChange() for
     * what it turned out to show, which was not what it was added to show.
     *
     * Two scales on one axis is normally a way to lie, so neither carries a
     * value axis at all: the shapes are the claim, not the heights, and the
     * numbers behind both are on screen elsewhere (the loss in the rail, the
     * pictures above). Here the two lines lying almost on top of each other IS
     * the finding, which is the one case where a shared axis is honest.
     */
    if (this.change && this.change.length > 2) {
      const steps = this.data.meta.steps;
      const cs = this.change.slice(1).map((d) => Math.log(Math.max(d, 0.05)));
      const clo = Math.min(...cs), chi = Math.max(...cs);
      const CY = (l) => bot - ((l - clo) / Math.max(chi - clo, 1e-6)) * (bot - top);
      g.strokeStyle = theme.accent2;
      g.lineWidth = 1.6 * s;
      g.setLineDash([4 * s, 3 * s]);
      g.beginPath();
      cs.forEach((l, i) => (i ? g.lineTo(X(steps[i + 1]), CY(l)) : g.moveTo(X(steps[i + 1]), CY(l))));
      g.stroke();
      g.setLineDash([]);
    }

    // where you are
    const here = this.data.meta.steps[this.at];
    g.strokeStyle = theme.ink;
    g.lineWidth = 1 * s;
    g.beginPath(); g.moveTo(X(here), top - 6 * s); g.lineTo(X(here), bot); g.stroke();

    g.font = `${10 * s}px ui-monospace, SFMono-Regular, Menlo, monospace`;
    g.textAlign = 'left';
    g.textBaseline = 'alphabetic';
    g.fillStyle = theme.accent;
    g.fillText('training loss', pad, 11 * s);
    g.fillStyle = theme.accent2;
    g.fillText('how much the pictures changed', pad + 90 * s, 11 * s);
  }
}
