/**
 * The vocabulary, shown in full — and doubling as the prompt builder.
 *
 * WHY THIS IS A FIRST-CLASS PANEL AND NOT A FOOTNOTE. The model knows a fixed,
 * small list of words. In the first build that list was mentioned once, inside
 * an About popup behind an "i" in the top-right corner. The result, reported by
 * the first person to use it: typing "a cat" and "a mashrooom" produced the
 * SAME picture, apparently at random.
 *
 * Nothing was broken. Both prompts contained no word the model knows, so both
 * collapsed to the same token sequence and produced the same image. But finding
 * that out required reading a panel nobody opens, so in practice the app looked
 * like it was ignoring the prompt.
 *
 * The fix is not a better error message. It is showing the whole vocabulary,
 * next to the box you type into, as something you can click. Then the failure
 * mode mostly cannot happen, and when it does the reason is one glance away.
 *
 * ONE ROW PER CAPTION SLOT, which is the shape of the model rather than a
 * layout choice. The caption is [colour][body][eyes][horns][wings][legs] and
 * every slot is always filled, so a row per slot with exactly one chip pressed
 * in each is a truthful picture of what the model is about to be given. It also
 * makes `none` visible as the real answer it is: a monster with no wings is a
 * choice the vocabulary can express, not a gap.
 *
 * Everything here comes from the grammar, which comes from the model manifest.
 * Two lists that must agree are one list that will eventually disagree.
 */

const SLOT_LABELS = {
  colour: 'Colour', body: 'Body', eyes: 'Eyes',
  horns: 'Horns', wings: 'Wings', legs: 'Legs',
};

export class WordList {
  /**
   * @param root     container element
   * @param grammar  a Grammar built from the model manifest
   * @param onPick   called with the assembled prompt sentence
   */
  constructor(root, grammar, onPick) {
    this.root = root;
    this.grammar = grammar;
    this.onPick = onPick;
    this.rows = new Map();
    // Start from the grammar's own defaults so the chips agree with what an
    // empty prompt would produce.
    this.pick = { ...grammar.defaults };
    this.build();
  }

  get prompt() { return this.grammar.describe(this.pick); }

  /** Reflect a typed prompt back onto the chips, so the two never disagree. */
  syncFrom(text) {
    this.pick = this.grammar.parse(text).slots;
    this.paint();
  }

  build() {
    this.root.innerHTML = '';
    // DISTINCT words, not the number of chips. `none` appears in three rows and
    // `two` in two, so counting chips gives 40 where the vocabulary holds 37 —
    // and the header, which counts the real vocabulary, would disagree with
    // this panel on screen. Two numbers for one fact is a bug even when both
    // are arithmetically true.
    const distinct = new Set(this.grammar.slots
      .flatMap((s) => this.grammar.words[s])).size;
    const choices = this.grammar.slots
      .reduce((n, s) => n + this.grammar.words[s].length, 0);

    const intro = document.createElement('p');
    intro.className = 'wl-intro';
    intro.innerHTML =
      `Everything this model knows: <b>${distinct} words</b>, offered as `
      + `${choices} choices across six slots — and nothing else. `
      + `Clicking one here is the same as choosing it in the sentence above.`;
    this.root.appendChild(intro);

    for (const slot of this.grammar.slots) {
      this.rows.set(slot, this.group(slot));
    }
    this.paint();
  }

  group(slot) {
    const words = this.grammar.words[slot];
    const h = document.createElement('div');
    h.className = 'wl-head';
    h.innerHTML = `<span>${SLOT_LABELS[slot] || slot}</span><em>${words.length}</em>`;
    this.root.appendChild(h);

    const row = document.createElement('div');
    row.className = 'wl-row';
    for (const w of words) {
      const b = document.createElement('button');
      b.type = 'button';
      b.className = 'wl-word';
      if (w === 'none') b.classList.add('wl-none');
      b.dataset.slot = slot;
      b.dataset.word = w;
      b.textContent = w;
      b.onclick = () => {
        this.pick[slot] = w;
        this.paint();
        this.onPick(this.prompt);
      };
      row.appendChild(b);
    }
    this.root.appendChild(row);
    return row;
  }

  paint() {
    for (const [slot, row] of this.rows) {
      for (const b of row.children) {
        b.setAttribute('aria-pressed', String(this.pick[slot] === b.dataset.word));
      }
    }
  }
}
