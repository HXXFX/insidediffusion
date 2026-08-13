/**
 * The prompt, as a sentence you edit rather than a sentence you type.
 *
 *     a [green ▾] [slime ▾] with [three ▾] eyes, [horns ▾], [bat wings ▾]
 *     and [two legs ▾]
 *
 * WHY THIS AND NOT A TEXT BOX. The model knows 37 words in six fixed slots.
 * A text box invites every prompt it cannot answer — and the failure is silent,
 * because an unrecognised word does not error, it just leaves that slot at its
 * default and produces a perfectly clean monster that ignores what was asked.
 * The very first person to use the earlier build typed "a cat", got a generic
 * result, and reasonably concluded the prompt did nothing.
 *
 * With a dropdown per slot, a prompt the model cannot understand is impossible
 * to write. That removes the whole class of confusion rather than explaining it
 * afterwards, which is what the note under the text box was doing.
 *
 * It stays a SENTENCE rather than becoming six labelled fields, because the
 * caption really is a sentence with one word per slot — seeing "a green slime
 * with three eyes" assemble itself teaches the grammar for free, and a form
 * would hide exactly the structure the app exists to show.
 *
 * The typed path has not been deleted: `Grammar.parse` still backs the sample
 * prompts and anything restored from a link. This is the input surface, not the
 * only way a prompt can arrive.
 */

/**
 * How a slot's value reads inside the sentence.
 *
 * Derived, not tabulated. `wings` reads "<word> wings" and `none` reads "no
 * wings", for every value, so adding a word to the vocabulary needs no edit
 * here. Only genuine irregulars are listed, and there is exactly one.
 */
const IRREGULAR = { antenna: 'antennae' };
const SUFFIX = { wings: 'wings', legs: 'legs' };

function labelFor(slot, word) {
  if (word === 'none') return `no ${SUFFIX[slot] || slot}`;
  const w = IRREGULAR[word] || word;
  return SUFFIX[slot] ? `${w} ${SUFFIX[slot]}` : w;
}

/**
 * The sentence, as chunks that must not be broken across lines.
 *
 * Matches Grammar.describe(), so the sentence on screen and the sentence in the
 * prompt value are the same words. The `none` cases keep their slot rather than
 * collapsing it — a stable layout reading "no wings" beats one that silently
 * loses a control and reflows the whole sentence when you pick it.
 *
 * CHUNKED because the sidebar is 268px and the sentence wraps over three or
 * four lines there. Laid out as a flat run of literals and dropdowns, the wrap
 * fell wherever it liked and a line began with an orphaned ", " — which reads
 * as a rendering fault rather than as prose. Each dropdown now travels with its
 * own punctuation and the breaks only happen between chunks.
 */
const CHUNKS = [
  { before: 'a ', slot: 'colour' },
  { slot: 'body', after: ' with' },
  { slot: 'eyes', after: ' eyes,' },
  { slot: 'horns', after: ',' },
  { slot: 'wings', after: ' and' },
  { slot: 'legs' },
];

function lit(text) {
  const s = document.createElement('span');
  s.className = 'lit';
  s.textContent = text;
  return s;
}

export class PromptBuilder {
  /**
   * @param root     container element
   * @param grammar  a Grammar built from the model manifest
   * @param onChange called with the assembled sentence
   */
  constructor(root, grammar, onChange) {
    this.root = root;
    this.grammar = grammar;
    this.onChange = onChange;
    this.slots = { ...grammar.defaults };
    this.selects = new Map();
    this.build();
  }

  get prompt() { return this.grammar.describe(this.slots); }

  /** Adopt slot values parsed from somewhere else (a sample, a link). */
  syncFrom(text) {
    this.slots = this.grammar.parse(text).slots;
    for (const [slot, sel] of this.selects) {
      sel.value = this.slots[slot];
      sel.classList.toggle('dim', this.slots[slot] === 'none');
    }
  }

  build() {
    this.root.innerHTML = '';
    this.root.className = 'sentence';
    CHUNKS.forEach((c, i) => {
      const span = document.createElement('span');
      span.className = 'chunk';
      if (c.before) span.appendChild(lit(c.before));
      span.appendChild(this.select(c.slot));
      if (c.after) span.appendChild(lit(c.after));
      this.root.appendChild(span);
      // The wrap opportunity, and it has to live OUTSIDE the nowrap chunks.
      // Trailing spaces inside them left the chunks butted directly together
      // with no breakable whitespace anywhere, so the whole sentence became one
      // unbreakable line that ran 260px past the edge of the panel.
      if (i < CHUNKS.length - 1) this.root.appendChild(document.createTextNode(' '));
    });
  }

  select(slot) {
    const sel = document.createElement('select');
    sel.className = 'slot';
    // The visible sentence supplies no label a screen reader can use — the
    // words around a dropdown are decoration to it — so each one names itself.
    sel.setAttribute('aria-label', slot);
    for (const word of this.grammar.words[slot]) {
      const o = document.createElement('option');
      o.value = word;
      o.textContent = labelFor(slot, word);
      sel.appendChild(o);
    }
    sel.value = this.slots[slot];
    sel.classList.toggle('dim', this.slots[slot] === 'none');
    sel.onchange = () => {
      this.slots[slot] = sel.value;
      sel.classList.toggle('dim', sel.value === 'none');
      this.onChange(this.prompt, this.slots);
    };
    this.selects.set(slot, sel);
    return sel;
  }
}
