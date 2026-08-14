/**
 * Act 4 of the Making of tab — the combinations withheld from training.
 *
 * WHAT IT DRAWS. Per held-out pair: the two halves the model DID see, the
 * picture it never saw, and what it produced when asked for that picture. Then
 * the score, placed on the scale between "as good as an ordinary prompt" and
 * "no better than an unrelated picture", because a bare 0.0067 means nothing to
 * a reader and its POSITION between two named ends means everything.
 *
 * NOTHING HERE IS AUTHORED. Every number and every picture comes from
 * weights/holdout.json, written by train/holdout_test.py from a model trained
 * with those pairs removed. If the experiment is re-run and the answer changes,
 * this section changes with it — which is the only way a page claiming to show
 * a measurement is worth reading.
 *
 * LAZY, like the cloud pictures: 28 kB that the Make tab never needs, so it is
 * fetched when the Making of tab is first opened and not before.
 */

import { tensorToImageData } from '../views/draw.js';

const SIZE = 16;

/** Raw RGB bytes (base64, HWC uint8) -> an ImageData-backed canvas.
 *  These are NOT index maps: the model's output is continuous colour and does
 *  not decompose into the four palette entries, so the exporter ships raw RGB
 *  for every picture here and the two kinds can sit side by side without the
 *  generated one being quietly snapped back onto the palette. */
function paint(canvas, b64) {
  const bin = atob(b64);
  const g = canvas.getContext('2d');
  const img = g.createImageData(SIZE, SIZE);
  for (let i = 0; i < SIZE * SIZE; i++) {
    for (let c = 0; c < 3; c++) img.data[i * 4 + c] = bin.charCodeAt(i * 3 + c);
    img.data[i * 4 + 3] = 255;
  }
  canvas.width = canvas.height = SIZE;
  g.putImageData(img, 0, 0);
}

function tile(b64, label, cls) {
  const fig = document.createElement('figure');
  fig.className = 'ho-tile' + (cls ? ' ' + cls : '');
  const c = document.createElement('canvas');
  c.className = 'ho-pic';
  paint(c, b64);
  const cap = document.createElement('figcaption');
  cap.textContent = label;
  fig.append(c, cap);
  return fig;
}

export class HoldoutPanel {
  constructor(els) {
    this.els = els;          // { rows, scale, verdict }
    this.data = null;
  }

  async load(base = './weights/') {
    if (this.data) return this.data;
    const r = await fetch(base + 'holdout.json');
    if (!r.ok) throw new Error(`holdout.json: HTTP ${r.status}`);
    this.data = await r.json();
    return this.data;
  }

  build() {
    const d = this.data;
    if (!d) return;
    const { rows, scale, verdict } = this.els;
    rows.innerHTML = '';

    for (const p of d.pairs) {
      const row = document.createElement('div');
      row.className = 'ho-row';

      const head = document.createElement('div');
      head.className = 'ho-head';
      head.innerHTML = `<b>${p.pair}</b><span>${p.why}</span>`;
      row.appendChild(head);

      const strip = document.createElement('div');
      strip.className = 'ho-strip';
      // Order is the argument: what it saw, then the gap, then what it made.
      strip.appendChild(tile(p.saw[0], p.saw_labels[0], 'saw'));
      strip.appendChild(tile(p.saw[1], p.saw_labels[1], 'saw'));
      strip.appendChild(tile(p.never_saw, 'never in training', 'never'));
      strip.appendChild(tile(p.drew, 'what it drew', 'drew'));
      row.appendChild(strip);

      /* THE BAR IS THE POINT, not the number. `position` is where the score
         lands between an ordinary prompt (0%) and an unrelated picture (100%),
         so a reader who has never seen a pixel error can still tell that 0.6%
         of the way is nothing. Floored at 1.5% of the track so a perfect score
         still draws something rather than reading as a missing element. */
      const meter = document.createElement('div');
      meter.className = 'ho-meter';
      const pct = Math.max(0, p.position) * 100;
      meter.innerHTML =
        `<i style="width:${Math.max(1.5, Math.min(100, pct)).toFixed(1)}%"></i>`
        + `<span>${pct < 0.05 ? 'as good as a prompt it was trained on'
          : `${pct.toFixed(1)}% of the way toward an unrelated picture`}</span>`;
      row.appendChild(meter);
      rows.appendChild(row);
    }

    scale.textContent =
      `Scale: an ordinary trained prompt scores ${d.seen.toFixed(4)}, and the same`
      + ` picture scored against a different prompt's answer scores`
      + ` ${d.floors.wrong_ref.toFixed(4)}. Every bar above is where the held-out`
      + ` attempt falls between those two.`;

    const worst = d.pairs.reduce((a, b) => (a.position > b.position ? a : b));
    const all = d.pairs.every((p) => p.verdict === 'composes');
    verdict.textContent = all
      ? `All ${d.pairs.length} withheld combinations were drawn about as well as`
        + ` prompts the model trained on — the worst, ${worst.pair}, sits`
        + ` ${(worst.position * 100).toFixed(1)}% of the way toward an unrelated`
        + ` picture. In every case the picture scores closest to what was asked`
        + ` for, not to either combination it had actually seen, so it is not`
        + ` falling back on a near neighbour. ${d.withheld_share}% of all prompts`
        + ` were withheld.`
      : `${d.pairs.filter((p) => p.verdict === 'composes').length} of`
        + ` ${d.pairs.length} withheld combinations were drawn about as well as`
        + ` trained ones; ${worst.pair} was the weakest at`
        + ` ${(worst.position * 100).toFixed(1)}% of the way toward an unrelated`
        + ` picture.`;
  }
}
