/**
 * Turning a sentence into a caption, and back.
 *
 * THE CAPTION IS SLOT-BASED, WHICH CHANGES EVERYTHING ABOUT PARSING IT:
 *
 *     [bos] [colour] [body] [eyes] [horns] [wings] [legs] [eos]
 *
 * Position IS meaning. The previous model took a free bag of words, so
 * tokenising was "keep the ones you recognise, in the order they appeared".
 * Doing that here would put `bat` in the eyes slot and `three` in the colour
 * slot, and the result would not be an error — it would be a perfectly clean
 * monster that has nothing to do with what was typed. A wrong answer that
 * looks right is worse than a crash, so parsing is done properly.
 *
 * ONE WORD IS GENUINELY AMBIGUOUS. `two` is a value for both eyes and legs,
 * and the vocabulary has a single token for it — position is what tells them
 * apart. So the parser looks ahead for the slot noun: "two legs" and "three
 * eyes" are resolved by the word that follows. With nothing to go on it falls
 * back to the first slot still unfilled, in caption order.
 *
 * Everything here is driven by `manifest.caption`, which the exporter writes
 * from the same Python lists the model was trained on. Nothing about the
 * vocabulary is written down twice.
 */

/**
 * Nouns that name a slot rather than fill one. `horns` is deliberately in both
 * this table and the horns word list — "with horns" means the value, "two
 * horns" would mean the noun — so a word is only treated as a slot noun when
 * it is not usable as a value in its own right at that position.
 */
const SLOT_NOUNS = {
  eye: 'eyes', eyes: 'eyes',
  leg: 'legs', legs: 'legs',
  wing: 'wings', wings: 'wings',
  horn: 'horns', horns: 'horns',
  body: 'body', colour: 'colour', color: 'colour',
};

/** Filler that carries no meaning. Listed so it is not reported as unknown —
 *  telling someone their "a" and "with" were dropped is noise, not help. */
const FILLER = new Set(['a', 'an', 'the', 'with', 'and', 'has', 'having',
  'that', 'it', 'of', 'in', 'on', 'plus']);

/**
 * Words that mean "this slot is empty". `none` is a real vocabulary value, but
 * nobody types it — they type "no wings". Without this, "no" was reported as a
 * word the model does not know while the wings came out empty anyway, so the
 * app was apologising for doing exactly what was asked.
 */
const NEGATIONS = new Set(['no', 'without', 'not', 'nor', 'zero']);

export class Grammar {
  /**
   * @param caption manifest.caption — { slots, layout, words, null }
   * @param tokenOf Map from word to vocabulary index
   */
  constructor(caption, tokenOf) {
    this.slots = caption.slots;
    this.layout = caption.layout;
    this.words = caption.words;
    this.nullCaption = caption.null;
    this.tokenOf = tokenOf;

    // word -> [slots it can fill]. Built from the manifest, so it cannot drift.
    this.slotsOf = new Map();
    for (const slot of this.slots) {
      for (const w of this.words[slot]) {
        if (!this.slotsOf.has(w)) this.slotsOf.set(w, []);
        this.slotsOf.get(w).push(slot);
      }
    }

    // A slot is optional exactly when `none` is one of its values. That is a
    // property of the trained vocabulary, not a policy decision to be repeated
    // here — horns, wings and legs have it; colour, body and eyes do not.
    this.optional = new Set(this.slots.filter((s) => this.words[s].includes('none')));
    this.defaults = {};
    for (const slot of this.slots) {
      this.defaults[slot] = this.optional.has(slot) ? 'none'
        : ({ colour: 'green', body: 'blob', eyes: 'two' }[slot]
           ?? this.words[slot][0]);
    }
  }

  /** Every word the model understands, grouped by slot. For the word list UI. */
  vocabulary() {
    return this.slots.map((slot) => ({ slot, words: this.words[slot] }));
  }

  /**
   * "a green slime with three eyes and bat wings"
   *   -> { slots, known, unknown, defaulted }
   *
   * `defaulted` is the list of slots the sentence never mentioned. The app
   * shows it, because "why did I get legs I did not ask for" has an answer and
   * the answer should not be a mystery.
   */
  parse(text) {
    const words = String(text).toLowerCase().match(/[a-z]+/g) || [];
    const slots = {};
    const known = [];
    const unknown = [];

    for (let i = 0; i < words.length; i++) {
      const w = words[i];

      // "no wings" / "without legs" — an explicit empty slot.
      if (NEGATIONS.has(w)) {
        let j = i + 1;
        while (j < words.length && FILLER.has(words[j])) j++;
        const named = SLOT_NOUNS[words[j]];
        if (named && this.optional.has(named)) {
          slots[named] = 'none';
          known.push(w, words[j]);
          i = j;
        }
        continue;
      }

      const cand = this.slotsOf.get(w);

      if (!cand) {
        if (!FILLER.has(w) && !SLOT_NOUNS[w]) unknown.push(w);
        continue;
      }

      let slot;
      if (cand.length === 1) {
        slot = cand[0];
      } else {
        // Ambiguous. Look ahead: "two legs" names its slot, "two" alone does
        // not. Skip any filler between the value and its noun.
        let j = i + 1;
        while (j < words.length && FILLER.has(words[j])) j++;
        const named = SLOT_NOUNS[words[j]];
        if (named && cand.includes(named)) slot = named;
        else slot = cand.find((s) => !(s in slots)) ?? cand[0];
      }
      slots[slot] = w;
      known.push(w);
    }

    // `defaulted` lists only the REQUIRED slots that had to be invented.
    //
    // An optional slot going to `none` is what "did not mention it" means, so
    // reporting it would have the app explaining that a monster with no wings
    // has no wings. A colour or a body appearing from nowhere is genuinely
    // worth a word, because the user did not choose it and can see it.
    const defaulted = [];
    for (const slot of this.slots) {
      if (slot in slots) continue;
      slots[slot] = this.defaults[slot];
      if (!this.optional.has(slot)) defaulted.push(slot);
    }
    return { slots, known, unknown, defaulted };
  }

  /** Slot values -> token ids, in caption order. */
  encode(slots) {
    const ids = this.layout.map((entry) => {
      const word = (entry === '<bos>' || entry === '<eos>') ? entry : slots[entry];
      const id = this.tokenOf.get(word);
      if (id === undefined) throw new Error(`no token for "${word}" (slot ${entry})`);
      return id;
    });
    return Int32Array.from(ids);
  }

  /** The sentence a human reads. The model never sees this. */
  describe(slots) {
    const head = `a ${slots.colour} ${slots.body} with ${slots.eyes} eyes`;
    const rest = [];
    if (slots.horns !== 'none') rest.push(slots.horns);
    if (slots.wings !== 'none') rest.push(`${slots.wings} wings`);
    if (slots.legs !== 'none') rest.push(`${slots.legs} legs`);
    if (!rest.length) return head;
    if (rest.length === 1) return `${head} and ${rest[0]}`;
    return `${head}, ${rest.slice(0, -1).join(', ')} and ${rest[rest.length - 1]}`;
  }

  /** Which caption column a slot occupies — the attention view labels its
   *  grids with this, and getting it from here means the label and the token
   *  can never disagree. */
  columnOf(slot) { return this.layout.indexOf(slot); }
}
