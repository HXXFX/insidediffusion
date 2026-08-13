/**
 * The log panel: renders what js/log.js recorded.
 *
 * COALESCED, NOT PER-ENTRY. A run at high playback speed emits entries faster
 * than a frame, and re-rendering a 400-row list on each one is how a log panel
 * becomes the slowest thing in an app whose whole point is a live animation.
 * Entries are appended on a rAF tick, and only while the panel is open —
 * closed, it subscribes to nothing but a dirty flag.
 *
 * It also only autoscrolls when the reader is ALREADY at the bottom. Yanking
 * someone back down while they are reading history is the classic mistake in
 * every log viewer, and it is one comparison to avoid.
 */

import { log, fmtT } from '../log.js';

const NEAR_BOTTOM = 24;   // px of slack that still counts as "at the bottom"

export class LogView {
  constructor(els) {
    this.list = els.list;
    this.count = els.count;
    this.panel = els.panel;
    this.open = false;
    this.dirty = false;
    this.rendered = 0;
    this.tick = this.tick.bind(this);
    log.subscribe((e) => {
      // A null entry means the log was cleared: drop everything and rebuild.
      if (e === null) { this.rendered = 0; this.list.innerHTML = ''; }
      this.dirty = true;
      if (this.count) this.count.textContent = String(log.n);
      if (this.open) requestAnimationFrame(this.tick);
    });
  }

  setOpen(v) {
    this.open = v;
    if (v) { this.render(); this.scrollToEnd(); }
  }

  tick() {
    if (this.dirty && this.open) this.render();
  }

  render() {
    const all = log.entries();
    const stick = this.atEnd();
    // Append only what is new. Rebuilding the list every tick loses the
    // reader's scroll position and any text they had selected.
    if (this.rendered > all.length) { this.rendered = 0; this.list.innerHTML = ''; }
    for (let i = this.rendered; i < all.length; i++) {
      this.list.appendChild(row(all[i]));
    }
    this.rendered = all.length;
    this.dirty = false;
    if (stick) this.scrollToEnd();
  }

  atEnd() {
    const el = this.list;
    return el.scrollHeight - el.scrollTop - el.clientHeight < NEAR_BOTTOM;
  }

  scrollToEnd() { this.list.scrollTop = this.list.scrollHeight; }
}

function row(e) {
  const li = document.createElement('li');
  li.className = `logrow lv-${e.level}`;
  const t = document.createElement('span');
  t.className = 'logt';
  t.textContent = fmtT(e.t);
  const m = document.createElement('span');
  m.className = 'logm';
  m.textContent = e.msg;
  li.append(t, m);
  if (e.detail) {
    const d = document.createElement('span');
    d.className = 'logd';
    d.textContent = e.detail;
    li.appendChild(d);
  }
  return li;
}
