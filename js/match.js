/**
 * "What does this model think my picture is?"
 *
 * Searches every prompt the model can be given — 6,400 word combinations times
 * 10 colours, 64,000 monsters — for the one nearest an uploaded picture, and
 * hands back its words.
 *
 * WHY THIS EXISTS. Image-to-image starts from your picture but is steered by
 * the words in the dock for the whole run, so uploading a blue turtle while the
 * prompt still says "green slime with bat wings" sets the two against each
 * other. Measured on the reference implementation: at Change amount 0.5 the
 * mismatched prompt costs 0.362 of identity against 0.254 for a matching one,
 * 43% worse, and by strength 0.35 the words have already recoloured the turtle
 * green and grown it wings. This is the missing half of image-to-image, not a
 * convenience.
 *
 * THE COLOUR IS SEPARABLE, which is what makes 64,000 candidates cheap. A
 * drawing is an index map into a four-entry lookup table, and only ONE of those
 * entries — the body colour — changes with the colour word. So squared error
 * against candidate (shape s, colour c) expands to
 *
 *     D = Σ|img|²  −  2 Σ_g lut_c[g]·sum_g  +  Σ_g n_g·|lut_c[g]|²
 *
 * where `sum_g` and `n_g` are the uploaded picture's colour sums and pixel
 * counts within index group g. The first term is the same for every candidate
 * and drops out of the ranking. So each shape is walked ONCE to build four
 * accumulators, and all ten colours are then scored with arithmetic on those —
 * 6,400 walks rather than 64,000 renders.
 *
 * EXACT, not approximate: this is the same squared error a pixel-by-pixel
 * comparison would give, rearranged. The ranking is not a heuristic.
 */

const GROUPS = 4;

/* ---------------------------------------------------------------- colour
 *
 * NAMING A COLOUR IS NOT NEAREST-RGB, and doing it that way was the bug behind
 * "my blue creature came back grey".
 *
 * Two things go wrong with a straight RGB match. Grey is the achromatic centre,
 * so it is close to EVERYTHING; and RGB distance weighs lightness as heavily as
 * hue, while people name colours by hue and treat paleness as a modifier — a
 * pale cyan is "light blue" to a human and never "grey". Measured by washing
 * each palette colour toward white in six steps and asking which one it is:
 * nearest-RGB recovers 38%, hue-first recovers 98%.
 *
 * A VOTE, NOT AN AVERAGE. Averaging the subject's pixels lands between its
 * hues, and the midpoint of two hues is desaturated — which reads as grey
 * again. One test creature is cyan with orange markings and averages to a dead
 * grey, while 78% of its pixels individually name the same colour. So every
 * pixel names itself and the majority wins.
 *
 * VERIFIED against every monster the model can draw: all 64,000 render, get
 * named, and come back with the colour word they were asked for — 100% on all
 * ten, where the RGB rule this replaced managed 38% on the washing test above.
 * The stage costs 2.7 ms.
 *
 * FULL RESOLUTION, not the 16x16 tile. The tile is already degraded: every edge
 * pixel is the subject averaged with the background, which is exactly the
 * washing-toward-white that destroys hue. The vote runs on the source pixels.
 */

/** sRGB 0-255 -> CIE Lab. Hue in Lab is stable under lightness change, which
 *  is the whole reason for leaving RGB. */
function toLab(r, g, b) {
  const f = (v) => { v /= 255; return v <= 0.04045 ? v / 12.92 : Math.pow((v + 0.055) / 1.055, 2.4); };
  const R = f(r), G = f(g), B = f(b);
  let x = (0.4124 * R + 0.3576 * G + 0.1805 * B) / 0.9505;
  let y = (0.2126 * R + 0.7152 * G + 0.0722 * B);
  let z = (0.0193 * R + 0.1192 * G + 0.9505 * B) / 1.089;
  const k = (v) => (v > 0.008856 ? Math.cbrt(v) : 7.787 * v + 16 / 116);
  x = k(x); y = k(y); z = k(z);
  return [116 * y - 16, 500 * (x - y), 200 * (y - z)];
}

/** Below this Lab chroma a colour has no hue worth naming and is matched on
 *  lightness among white/grey/black instead. Measured against the palette,
 *  whose achromatic entries sit at C* 3-10 and whose chromatic ones at 43-66,
 *  so the gap is wide and the threshold is not a tuned number. */
const CHROMA_MIN = 12;
const CHROMATIC_AT = 20;

/* HOW FAR AN ACHROMATIC PIXEL MAY REACH for the entry it votes for, in Lab
   lightness. Without this every dark pixel voted "black" at full strength no
   matter how dark, and the outline colour every monster shares — 26,25,33, at
   lightness 9.2 — is 28.6 from the darkest BODY colour the palette owns (black,
   at 37.8). So outlines were casting confident votes for a colour they do not
   resemble, and in the 37% of shapes whose details outnumber their body they
   won: a white creature came back black. Fading the vote to nothing at 15 keeps
   a pixel from speaking for a colour it is not. It also sharpened the real
   photographs from a 44%-against-41% coin toss to 74%. */
const ACHROMA_TOL = 15;


/**
 * The picture's SUBJECT pixels, with its background dropped.
 *
 * The background has to go or it decides the vote: a creature on a pale field
 * is mostly pale by pixel count, and white is one of the ten colour words.
 *
 * A photograph whose own backdrop is nothing like the fill keeps every pixel,
 * which is the right answer rather than a failure — a vote diluted by some
 * backdrop still finds the subject's hue, whereas an over-eager cut can delete
 * the subject itself.
 *
 * @param d RGBA bytes, @param w,h its dimensions, @param bg the fill colour the
 *        uploader painted, straight from the model manifest
 * @returns flat RGB triples
 */
export function subjectPixels(d, w, h, bg) {
  /* THE BACKGROUND IS KNOWN, NOT GUESSED, and every attempt to infer it from
     the picture was worse than asking. The uploader paints this exact colour
     onto the canvas before drawing the image, so a pixel matching it either is
     the fill or is indistinguishable from it.
     Two inference schemes were built and measured against all 64,000 monsters
     first. Averaging the border assumed the whole border was background, so a
     creature touching the edge inflated the spread, nothing was dropped, and
     the pale backdrop outvoted the subject — 28% recall on black. Taking the
     border's dominant cluster fixed that and broke the opposite case: where a
     monster FILLS the frame the dominant border colour is its own body, so the
     subject itself was deleted and every chromatic colour fell to 89%. Both
     failures come from the same mistake — treating an inference as knowledge
     when the real answer was already available. */
  const ref = toLab(bg[0], bg[1], bg[2]);
  const out = [];
  for (let i = 0; i < w * h; i++) {
    const r = d[i * 4], g = d[i * 4 + 1], b = d[i * 4 + 2];
    const [L, A, B] = toLab(r, g, b);
    /* 4, NOT 12. The palette's white body is 228,230,236 against a
       248,249,252 background — 6.8 apart, closer than any other colour comes by
       a factor of four. A wider radius deleted white creatures entirely and
       left only their outlines to be named. */
    if (Math.hypot(L - ref[0], A - ref[1], B - ref[2]) < 4) continue;
    out.push(r, g, b);
  }
  // Everything cut means the picture IS its background — nothing to vote on, so
  // hand back the lot rather than an empty vote.
  return new Float32Array(out.length ? out : Array.from(d).filter((_, i) => i % 4 !== 3));
}

/**
 * Which of the model's colours the picture is, by majority vote of its pixels.
 *
 * @param px   flat RGB triples of SUBJECT pixels only (background removed)
 * @param luts the palette, [colour][group][rgb]
 * @returns { index, share, runnerUp, runnerUpShare, hueGap, nextHue, nextGap }
 *
 *   `runnerUp` is the colour with the second most VOTES — a different part of
 *   the picture, usually its markings. `nextHue` is the second nearest palette
 *   colour to the picture's overall hue. THEY ARE NOT THE SAME THING and the
 *   difference is not academic: a cyan creature with orange markings has blue
 *   as its nearest hue and yellow as its vote runner-up, and reporting the
 *   latter as "between blue and yellow" tells the reader its colour lies
 *   between two hues it sits nowhere near.
 *
 *   `hueGap` and `nextGap` are how far those two actually are, in degrees. They
 *   are reported because the palette owns ten colours and no cyan: a teal
 *   creature is 62 degrees from green and 82 from blue, and naming one without
 *   saying how far it reached would be a small lie.
 */
export function voteColour(px, luts) {
  const n = luts.length;
  const L = [], C = [], H = [];
  for (let i = 0; i < n; i++) {
    const [l, a, b] = toLab(luts[i][1][0], luts[i][1][1], luts[i][1][2]);
    L.push(l); C.push(Math.hypot(a, b)); H.push(Math.atan2(b, a) * 180 / Math.PI);
  }
  const chrom = [], achrom = [];
  for (let i = 0; i < n; i++) (C[i] >= CHROMATIC_AT ? chrom : achrom).push(i);

  const votes = new Float64Array(n);
  let hx = 0, hy = 0;
  for (let p = 0; p < px.length; p += 3) {
    const [l, a, b] = toLab(px[p], px[p + 1], px[p + 2]);
    const c = Math.hypot(a, b);
    if (c >= CHROMA_MIN && chrom.length) {
      const h = Math.atan2(b, a) * 180 / Math.PI;
      let best = chrom[0], bd = 1e9;
      for (const i of chrom) {
        const d = Math.abs(((H[i] - h + 180) % 360 + 360) % 360 - 180);
        if (d < bd) { bd = d; best = i; }
      }
      // WEIGHTED BY CHROMA: a nearly-grey pixel should not outvote a vividly
      // coloured one just because there are more of them along an edge.
      votes[best] += c;
      hx += c * Math.cos(h * Math.PI / 180);
      hy += c * Math.sin(h * Math.PI / 180);
    } else if (achrom.length) {
      let best = achrom[0], bd = 1e9;
      for (const i of achrom) {
        const d = Math.abs(L[i] - l);
        if (d < bd) { bd = d; best = i; }
      }
      /* THE RAMP MATTERS, and a flat weight here was a bug worth naming: it
         reported a dark blue creature as BLACK. Dark colours have compressed
         chroma, so a shaded blue pixel falls just under the threshold — and
         paying it the same full vote as a truly neutral one let the shadows
         outweigh the lit body, 42% to 42%. Weighting by how confidently
         neutral the pixel is makes the two branches meet at the boundary
         instead of stepping across it: nothing at chroma 12, full vote at 0. */
      votes[best] += (CHROMA_MIN - c) * Math.max(0, 1 - bd / ACHROMA_TOL);
    }
  }
  let total = 0; for (let i = 0; i < n; i++) total += votes[i];
  if (total <= 0) {
    // Every pixel was too far from every colour to speak for one. Rather than
    // let the sort return whichever entry happens to be first, ask again with
    // the reach removed so the nearest colour wins on its own merit.
    for (let p = 0; p < px.length; p += 3) {
      const [l] = toLab(px[p], px[p + 1], px[p + 2]);
      let best = 0, bd = 1e9;
      for (let i = 0; i < n; i++) {
        const d = Math.abs(L[i] - l);
        if (d < bd) { bd = d; best = i; }
      }
      votes[best] += 1; total += 1;
    }
  }
  const order = Array.from({ length: n }, (_, i) => i).sort((a, b) => votes[b] - votes[a]);
  const win = order[0];
  /* HUE IS AN ANGLE, so its mean is a vector mean. Averaging the numbers
     arithmetically puts the mean of 170 and -170 at 0 — the exact opposite of
     both — which reported one test picture as 84 degrees from its own match
     when the true gap was 62. */
  const meanHue = (hx || hy) ? Math.atan2(hy, hx) * 180 / Math.PI : 0;
  const gapTo = (i) => Math.abs(((H[i] - meanHue + 180) % 360 + 360) % 360 - 180);
  const chromatic = C[win] >= CHROMATIC_AT;
  // The nearest chromatic colour that is not the winner, by hue alone.
  let next = -1;
  if (chromatic) {
    for (const i of chrom) if (i !== win && (next < 0 || gapTo(i) < gapTo(next))) next = i;
  }
  return {
    index: win,
    share: total ? votes[win] / total : 0,
    runnerUp: order[1],
    runnerUpShare: total ? votes[order[1]] / total : 0,
    hueGap: chromatic ? gapTo(win) : null,
    nextHue: next >= 0 ? next : null,
    nextGap: next >= 0 ? gapTo(next) : null,
  };
}

export class Vocabulary {
  constructor(data) {
    this.size = data.size;
    this.colours = data.colours;
    this.luts = data.luts;          // [colour][group][rgb]
    this.slots = data.slots;        // body, eyes, horns, wings, legs
    this.words = data.words;
    this.maps = data.maps;
    this.n = data.maps.length;
    // |lut|² per colour per group, and the group's constant entries, hoisted
    // out of the search loop.
    this._sq = this.luts.map((l) => l.map((e) => e[0] * e[0] + e[1] * e[1] + e[2] * e[2]));
  }

  /**
   * @param tile CHW Float32Array in [-1,1], 3 x size x size — exactly what the
   *        uploader produces and what the sampler is handed.
   * @returns { words, colour, sentence, index, map, error } — `error` is mean
   *          absolute pixel difference in 0-255 units, so it is comparable with
   *          every other number the project reports.
   */
  match(tile, fixedColour = -1) {
    const S = this.size, N = S * S;
    // To 0-255 RGB once. The search then works in the same units as the LUT.
    const px = new Float32Array(N * 3);
    for (let i = 0; i < N; i++) {
      for (let c = 0; c < 3; c++) {
        px[i * 3 + c] = (tile[c * N + i] + 1) * 127.5;
      }
    }

    const n = new Float32Array(GROUPS);
    const sr = new Float32Array(GROUPS);
    const sg = new Float32Array(GROUPS);
    const sb = new Float32Array(GROUPS);

    let bestD = Infinity, bestShape = -1, bestColour = -1;
    for (let s = 0; s < this.n; s++) {
      n.fill(0); sr.fill(0); sg.fill(0); sb.fill(0);
      const map = this.maps[s];
      for (let i = 0; i < N; i++) {
        const gidx = map.charCodeAt(i) - 48;
        n[gidx] += 1;
        sr[gidx] += px[i * 3];
        sg[gidx] += px[i * 3 + 1];
        sb[gidx] += px[i * 3 + 2];
      }
      // WITH THE COLOUR ALREADY DECIDED, only the shape is searched. The colour
      // is named from the full-resolution picture by `voteColour`, which sees
      // hue this 16x16 tile has already lost to its own anti-aliased edges.
      const c0 = fixedColour >= 0 ? fixedColour : 0;
      const c1 = fixedColour >= 0 ? fixedColour + 1 : this.colours.length;
      for (let c = c0; c < c1; c++) {
        const lut = this.luts[c], sq = this._sq[c];
        let d = 0;
        for (let g = 0; g < GROUPS; g++) {
          if (!n[g]) continue;
          d += n[g] * sq[g]
             - 2 * (lut[g][0] * sr[g] + lut[g][1] * sg[g] + lut[g][2] * sb[g]);
        }
        if (d < bestD) { bestD = d; bestShape = s; bestColour = c; }
      }
    }

    const words = this.words[bestShape];
    const colour = this.colours[bestColour];
    return {
      index: bestShape,
      colour,
      words,
      map: this.maps[bestShape],
      lut: this.luts[bestColour],
      sentence: `a ${colour} ${words[0]} with ${words[1]} eyes, `
        + `${words[2]}, ${words[3]} and ${words[4]}`,
      error: this._meanAbs(px, this.maps[bestShape], this.luts[bestColour]),
    };
  }

  /** Mean absolute difference in 0-255 units — the project's standard metric,
   *  computed on the winner so the number reported is the real one rather than
   *  the rearranged score used for ranking. */
  _meanAbs(px, map, lut) {
    const N = this.size * this.size;
    let s = 0;
    for (let i = 0; i < N; i++) {
      const e = lut[map.charCodeAt(i) - 48];
      s += Math.abs(px[i * 3] - e[0]) + Math.abs(px[i * 3 + 1] - e[1])
         + Math.abs(px[i * 3 + 2] - e[2]);
    }
    return s / (N * 3);
  }

  /** The winner as a CHW tensor, for drawing beside the upload. */
  tensor(m) {
    const N = this.size * this.size;
    const out = new Float32Array(3 * N);
    for (let i = 0; i < N; i++) {
      const e = m.lut[m.map.charCodeAt(i) - 48];
      for (let c = 0; c < 3; c++) out[c * N + i] = e[c] / 127.5 - 1;
    }
    return out;
  }
}

/** Lazily fetched: 49 kB gzipped that only matters once a picture is dropped in,
 *  so boot must not wait for it. A failure leaves the button reporting that,
 *  rather than taking anything else down. */
export function loadVocabulary(base = './weights/') {
  return fetch(base + 'vocabulary.json')
    .then((r) => (r.ok ? r.json() : null))
    .then((d) => (d ? new Vocabulary(d) : null));
}
