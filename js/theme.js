/**
 * The palette, read from CSS rather than written twice.
 *
 * The canvases are drawn in JavaScript and the chrome is styled in CSS, so
 * every colour used to exist in both places. That is fine until the palette
 * changes: retheming the stylesheet left 21 hardcoded literals scattered
 * through the view modules, still drawing the old cold blue on the new warm
 * ground, and each one had to be hunted individually.
 *
 * So there is one source — the custom properties on :root — and this reads them
 * back. A colour that is not in the token list cannot be used by a view, which
 * is the point.
 */

const NAMES = [
  'bg', 'panel', 'panel-2', 'line', 'line-soft', 'raise',
  'ink', 'ink-2', 'ink-3', 'accent', 'accent-2', 'hot', 'hot-bg',
  'sel-bg', 'sel-line', 'sel-ink', 'view-bg', 'track', 'canvas-bg',
];

/** camelCase accessors: theme.viewBg, theme.ink2, theme.accent … */
function build() {
  const cs = getComputedStyle(document.documentElement);
  const out = {};
  for (const n of NAMES) {
    const key = n.replace(/-(\w)/g, (_, c) => c.toUpperCase());
    out[key] = cs.getPropertyValue(`--${n}`).trim() || '#f0f';
  }
  return out;
}

export const theme = build();

/** Re-read after a theme switch. Views hold references, so mutate in place. */
export function refreshTheme() {
  Object.assign(theme, build());
  return theme;
}

function parse(hex) {
  let h = hex.replace('#', '');
  if (h.length === 3) h = h.split('').map((c) => c + c).join('');
  return [parseInt(h.slice(0, 2), 16), parseInt(h.slice(2, 4), 16),
          parseInt(h.slice(4, 6), 16)];
}

/**
 * A token at partial opacity.
 *
 * The views are full of half-transparent strokes and fills, and they were
 * written as literal rgba() against a near-black page — "white at 16%" is a
 * hairline on black and invisible on bone. Every one of them now names a token
 * and an opacity, so the same line works on either ground.
 */
export function alpha(hex, a) {
  const [r, g, b] = parse(hex);
  return `rgba(${r},${g},${b},${a})`;
}

/**
 * Mix a token toward another colour. Canvas work often needs "the accent, but
 * quieter" and `globalAlpha` is the wrong tool when shapes overlap — each
 * overlap compounds and the result is darker than anything in the palette.
 */
export function fade(hex, amount, towardHex) {
  const a = parse(hex), b = parse(towardHex || theme.bg);
  const m = a.map((v, i) => Math.round(v + (b[i] - v) * amount));
  return `rgb(${m[0]},${m[1]},${m[2]})`;
}
