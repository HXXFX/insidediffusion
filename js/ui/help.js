/**
 * "What am I looking at?" — the explanation layer.
 *
 * Every view in this app is abstract enough that a newcomer cannot decode it
 * unaided, and an unexplained 3-D plot teaches nothing at all. So each view and
 * each panel carries an explanation, attached to the thing it explains rather
 * than filed away in an About box, where the first version buried it.
 *
 * The shape is fixed on purpose, and it is the same three questions every
 * time — WHAT AM I LOOKING AT, WHAT IS EACH THING, WHAT SHOULD I NOTICE.
 * A consistent shape is what lets someone learn to read the second view
 * faster than the first.
 *
 * Content lives in config.js. This file only renders it.
 */

/**
 * Swatches are built from the palette, so a key never describes a colour the
 * app has stopped using. Named entries are gradients because the thing they
 * stand for is a RANGE — a surface from low to high, a tile from dark to light
 * — and a single flat chip would misrepresent it.
 */
import { theme, fade } from '../theme.js';
import { hoverClickPop } from './popover.js';

const SWATCH = () => ({
  surface: `linear-gradient(90deg,${fade(theme.accent, .8)},${theme.accent2})`,
  box: `linear-gradient(90deg,${fade(theme.accent, .85)},${fade(theme.accent, .3)})`,
  tile: `linear-gradient(90deg,${theme.line},${theme.ink2})`,
  grid: `linear-gradient(90deg,${fade(theme.accent, .85)},${theme.accent})`,
  axis: `repeating-linear-gradient(90deg,${theme.ink3} 0 3px,transparent 3px 6px)`,
  dash: `repeating-linear-gradient(90deg,${theme.ink3} 0 3px,transparent 3px 6px)`,
  tok: theme.line,
  /* Stands for "a small picture of a monster", which several keys now need:
     the views draw real training images, and a flat chip in one colour would
     say "a coloured thing" about something whose whole point is that it is a
     PICTURE. Hard steps rather than a smooth blend — a gradient reads as a
     range, and this is not a range, it is a few pixels side by side. The stops
     are theme tokens, so like every other swatch it cannot drift away from a
     colour the app has stopped using. */
  pic: `linear-gradient(90deg,${theme.accent2} 0 25%,${theme.accent} 25% 50%,`
    + `${theme.ink2} 50% 75%,${fade(theme.accent, .45)} 75%)`,
});

function swatch(spec) {
  const el = document.createElement('span');
  el.className = 'hswatch';
  // A named token wins over a literal, so the config can say `accent` and get
  // whatever the accent currently is.
  const named = SWATCH()[spec];
  if (named) {
    el.style.background = named;
  } else if (spec && spec.startsWith('#')) {
    el.style.background = spec;
  } else {
    // The config writes CSS-style names (`accent-2`, `ink-3`); the theme object
    // is camelCase. Without this the lookup misses and every one of them fell
    // back to the border colour — the key said "amber dot" beside a grey chip.
    const key = String(spec || '').replace(/-(\w)/g, (_, c) => c.toUpperCase());
    el.style.background = theme[key] || theme.line;
  }
  return el;
}

/**
 * Build the panel body for one help entry.
 * @param {{what:string, key:Array, watch:string, caveat?:string}} h
 */
export function helpBody(title, h) {
  const wrap = document.createElement('div');
  wrap.className = 'helpbody';

  const t = document.createElement('h5');
  t.textContent = title;
  wrap.appendChild(t);

  const what = document.createElement('p');
  what.className = 'hwhat';
  what.textContent = h.what;
  wrap.appendChild(what);

  if (h.key && h.key.length) {
    const dl = document.createElement('dl');
    dl.className = 'hkey';
    for (const [spec, name, meaning] of h.key) {
      const dt = document.createElement('dt');
      dt.appendChild(swatch(spec));
      const b = document.createElement('b');
      b.textContent = name;
      dt.appendChild(b);
      const dd = document.createElement('dd');
      dd.textContent = meaning;
      dl.appendChild(dt); dl.appendChild(dd);
    }
    wrap.appendChild(dl);
  }

  if (h.watch) {
    const w = document.createElement('p');
    w.className = 'hwatch';
    const lab = document.createElement('b');
    lab.textContent = 'What to notice — ';
    w.appendChild(lab);
    w.appendChild(document.createTextNode(h.watch));
    wrap.appendChild(w);
  }

  if (h.caveat) {
    const c = document.createElement('p');
    c.className = 'hcaveat';
    c.textContent = h.caveat;
    wrap.appendChild(c);
  }
  return wrap;
}

/**
 * Attach a "?" button to a pane that opens the explanation beside it.
 *
 * Deliberately per-pane rather than one global help screen: with four views
 * tiled, a single explanation would have to be about all of them at once,
 * which is how you end up with a manual nobody reads.
 *
 * A CARD, NOT A TAKEOVER. It used to cover the whole pane at 96% opacity and
 * could only be dismissed by finding the "Got it" button at the bottom of it —
 * so reading about a view meant losing sight of the view, and glancing at the
 * help cost two deliberate clicks. It is now a panel in the corner you opened
 * it from: hover to peek at it, click to pin it open, and it closes the way
 * every other panel in the app closes.
 */
export function attachPaneHelp(pane, title, h, onToggle) {
  const btn = document.createElement('button');
  btn.className = 'phelp';
  btn.type = 'button';
  btn.textContent = '?';
  btn.setAttribute('aria-label', `What am I looking at? — ${title}`);
  btn.setAttribute('aria-expanded', 'false');

  const panel = document.createElement('div');
  panel.className = 'hpanel';
  panel.hidden = true;
  panel.appendChild(helpBody(title, h));

  pane.appendChild(btn);
  pane.appendChild(panel);

  const pop = hoverClickPop(pane, btn, panel, {
    onOpen: () => { btn.classList.add('on'); onToggle && onToggle(true); },
    onClose: () => { btn.classList.remove('on'); onToggle && onToggle(false); },
  });
  return { open: () => pop.open(), close: () => pop.close() };
}
