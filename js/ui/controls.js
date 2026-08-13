/**
 * Builds the control panel FROM config.js.
 *
 * Nothing here knows what a "guidance" is. Adding a slider means adding an
 * entry to CONTROLS and reading its value where it matters — not editing
 * markup, not editing this file. That is the whole point of the config being
 * declarative, and it is easy to erode by hand-writing "just one" control.
 */

import { CONTROLS, DEFAULTS } from '../config.js';
import { hoverClickPop } from './popover.js';

/**
 * Group order is the WORKFLOW order, which is allowed to disagree with the
 * order the code runs in. `into` names the container a group renders into —
 * "Run it" lives further down the sidebar next to the Run button, because a
 * playback-speed slider belongs beside the thing it controls, not beside the
 * sampler settings.
 */
/**
 * No step numbers any more. They made sense when these were stacked down a
 * sidebar in workflow order; in a settings popover, numbering implies a
 * sequence that does not exist — nobody works through "1. Model, 2. Sampling"
 * before every run, they set them once and forget them.
 */
const GROUPS = [
  { id: 'prompt', label: null, into: 'prompt' },   // the dock, unlabelled
  { id: 'model', label: 'Model', into: 'model' },
  { id: 'sampling', label: 'Sampling', into: 'sampling' },
  { id: 'run', label: 'Playback', into: 'run' },
];

export class Controls {
  /** @param roots {prompt, sampling, run} — one container per group. */
  constructor(roots, onChange) {
    this.roots = roots instanceof HTMLElement ? { prompt: roots } : roots;
    this.root = this.roots.prompt;
    this.values = { ...DEFAULTS };
    this.onChange = onChange;
    this.els = new Map();
    this.mode = 'txt2img';
    this.build();
  }

  get(id) { return this.values[id]; }

  /**
   * Fill a select whose choices are not known until something has loaded.
   *
   * The model picker is the case: which models exist is a fact about what has
   * been trained and exported, read from weights/models.json. Writing the list
   * into config.js as well would be a second copy to go stale, and the failure
   * would be an option that 404s on click.
   */
  setOptions(id, options, value) {
    const el = this.els.get(id);
    if (!el) return;
    el.input.innerHTML = '';
    for (const o of options) {
      const opt = document.createElement('option');
      opt.value = o.value;
      opt.textContent = o.label;
      if (o.title) opt.title = o.title;
      el.input.appendChild(opt);
    }
    const v = value ?? this.values[id];
    const ok = options.some((o) => o.value === v);
    el.input.value = ok ? v : options[0]?.value;
    this.values[id] = el.input.value;
  }

  set(id, v, silent) {
    this.values[id] = v;
    const el = this.els.get(id);
    if (el) {
      // A mount point is a div: assigning `.value` to it would create a stray
      // property and change nothing on screen. Whoever mounted into it keeps
      // it in step instead.
      if (!el.mount) el.input.value = v;
      if (el.readout) el.readout.textContent = format(id, v);
    }
    if (!silent) this.onChange(id, v);
  }

  /** The container for a custom-rendered control, e.g. the prompt sentence. */
  slot(id) { return this.els.get(id)?.mount || null; }

  /**
   * @param mode  the active tab
   * @param has   flags the UI can gate on, e.g. { image: true }
   *
   * `needs` exists because "Change amount" is meaningless until an image has
   * been added. When image-to-image was its own tab, `mode` carried that fact;
   * now that a picture is just something you may or may not have dropped in,
   * the control has to key off the picture itself.
   */
  setMode(mode, has = {}) {
    this.mode = mode;
    this.has = has;
    for (const [id, el] of this.els) {
      const c = CONTROLS.find((x) => x.id === id);
      const okMode = !c.mode || c.mode.includes(mode);
      const okNeeds = !c.needs || !!has[c.needs];
      el.field.hidden = !(okMode && okNeeds);
    }
  }

  build() {
    for (const el of Object.values(this.roots)) if (el) el.innerHTML = '';
    for (const grp of GROUPS) {
      const items = CONTROLS.filter((c) => c.group === grp.id);
      if (!items.length) continue;
      const host = this.roots[grp.into] || this.roots.prompt;

      const sec = document.createElement('section');
      sec.className = 'grp';
      if (grp.label) {
        const hd = document.createElement('h3');
        hd.textContent = grp.label;
        sec.appendChild(hd);
      } else {
        sec.classList.add('bare');
      }

      for (const c of items) sec.appendChild(this.field(c));
      host.appendChild(sec);
    }
  }

  field(c) {
    const wrap = document.createElement('div');
    wrap.className = 'field';

    const lab = document.createElement('label');
    lab.setAttribute('for', `c-${c.id}`);
    lab.textContent = c.label;

    let readout = null;
    if (c.kind === 'range' || c.kind === 'int') {
      readout = document.createElement('span');
      readout.className = 'val';
      readout.textContent = format(c.id, this.values[c.id]);
      lab.appendChild(readout);
    }

    /**
     * The tip answers three questions in order — what it does, what it costs,
     * when to reach for it. Those are now three FIELDS rather than three
     * paragraphs in one string.
     *
     * As prose separated by blank lines it read as an undifferentiated block
     * you had to parse to find the one line you wanted, and a one-word cost
     * ("Free.") sitting alone in the middle of it looked like a mistake. As
     * labelled rows the three answers are findable without reading, and the
     * structure is enforced by the shape of the config instead of by a
     * convention someone has to remember.
     */
    /**
     * NOT for the sentence. Its label is screen-reader-only — the dock supplies
     * the caption, so `.dock-prompt .field > label` is clipped to 1×1px — and a
     * `?` appended to it inherits that: invisible, unclickable, and measured at
     * 1175px tall because a 1px-wide tip wraps every line to one character. It
     * was rendering for nobody. What it had to say about the dropdowns now
     * belongs in the log's description of the app.
     */
    if (c.tip && c.kind !== 'sentence') {
      const help = document.createElement('button');
      help.type = 'button';
      help.className = 'help';
      help.setAttribute('aria-label', `About ${c.label}`);
      help.textContent = '?';
      const tip = document.createElement('span');
      tip.className = 'tip';
      const t = typeof c.tip === 'string' ? { does: c.tip } : c.tip;

      const head = document.createElement('b');
      head.textContent = t.does;
      tip.appendChild(head);

      const row = (name, text) => {
        if (!text) return;
        const dt = document.createElement('i');
        dt.textContent = name;
        const dd = document.createElement('span');
        dd.textContent = text;
        tip.append(dt, dd);
      };
      row('Costs', t.cost);
      row('Try', t.when);
      help.appendChild(tip);

      /**
       * HOVER TO PEEK, CLICK TO PIN — the same mechanism as every other `?` in
       * the app, rather than the hand-rolled toggle this used to carry.
       *
       * Hover alone was never enough, for two independent reasons worth keeping
       * on record: the CSS rule relied on `:focus-visible`, which deliberately
       * does NOT match a mouse click — it exists for keyboard users — so
       * clicking the `?`, exactly what a small round question mark invites, did
       * nothing at all; and on touch there is no hover, so the tips were
       * unreachable. `hoverClickPop` covers every input device.
       *
       * The tip is a child of the button here, so the two hover regions overlap
       * rather than sitting apart — which is fine, and means the traverse grace
       * period never comes into play for these.
       */
      tip.hidden = true;
      hoverClickPop(help, help, tip, {
        onOpen: () => help.classList.add('on'),
        onClose: () => help.classList.remove('on'),
      });
      lab.appendChild(help);
    }
    wrap.appendChild(lab);

    let input;
    if (c.kind === 'sentence') {
      // A mount point, not a control. Its contents need the vocabulary, which
      // arrives with the model manifest long after the panel is built, so
      // main.js fills it in later via `slot(id)`. The VALUE still lives here,
      // so `get('prompt')` and `set('prompt', …)` work exactly as before and
      // nothing downstream knows the difference.
      input = document.createElement('div');
      input.id = `c-${c.id}`;
      wrap.appendChild(input);
      this.els.set(c.id, { field: wrap, input, readout: null, mount: input });
      return wrap;
    }
    if (c.kind === 'select') {
      input = document.createElement('select');
      for (const o of c.options) {
        const opt = document.createElement('option');
        opt.value = o.value; opt.textContent = o.label;
        input.appendChild(opt);
      }
    } else if (c.kind === 'text') {
      input = document.createElement('input');
      input.type = 'text';
      input.spellcheck = false;
      input.autocomplete = 'off';
    } else if (c.kind === 'int') {
      input = document.createElement('input');
      input.type = 'number';
      input.min = c.min; input.max = c.max; input.step = 1;
    } else {
      input = document.createElement('input');
      input.type = 'range';
      input.min = c.min; input.max = c.max; input.step = c.step;
    }
    input.id = `c-${c.id}`;
    input.value = this.values[c.id];

    const commit = (fire) => {
      let v = input.value;
      if (c.kind === 'range') v = parseFloat(v);
      if (c.kind === 'int') v = Math.round(parseFloat(v) || 0);
      this.values[c.id] = v;
      if (readout) readout.textContent = format(c.id, v);
      // Warn where a control stops behaving, with the number in the tooltip.
      if (c.warnAbove !== undefined) {
        wrap.classList.toggle('warn', parseFloat(v) > c.warnAbove);
      }
      if (fire) this.onChange(c.id, v);
    };

    // `input` updates the readout live; `change` fires the expensive rerun, so
    // dragging a slider does not launch a run per pixel.
    input.addEventListener('input', () => commit(c.kind === 'text' ? false : false));
    input.addEventListener('change', () => commit(true));
    if (c.kind === 'text') {
      input.addEventListener('keydown', (e) => { if (e.key === 'Enter') commit(true); });
    }

    wrap.appendChild(input);
    this.els.set(c.id, { field: wrap, input, readout });
    commit(false);
    return wrap;
  }
}

function format(id, v) {
  if (id === 'guidance') return Number(v).toFixed(1);
  if (id === 'speed') return `${Number(v).toFixed(2)}×`;
  if (id === 'strength') return `${Math.round(Number(v) * 100)}%`;
  return String(v);
}
