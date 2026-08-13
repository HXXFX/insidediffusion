/**
 * The activity log.
 *
 * WHY AN APP LIKE THIS NEEDS ONE. Almost everything here happens on its own
 * schedule and leaves no trace: a 2.4 MB model streams in, a worker starts,
 * thirty denoising steps run, a prompt is re-encoded, an image is reduced to
 * 16x16. When it works you see the result and none of the story; when it does
 * not, the only signal is an overlay that says something failed. The browser
 * console has all of this and is the wrong place to send a visitor.
 *
 * So events are recorded here and shown in the app. This is a real log, not a
 * status line — it keeps the history, with timings, so "why did that take six
 * seconds" and "what did it actually do with my words" both have answers.
 *
 * DESIGN NOTES
 *
 * - A RING BUFFER, not an array that grows. A long session with the playback
 *   speed up can emit thousands of entries, and an unbounded list in a page
 *   that already holds every step of every run is a leak with a friendly name.
 * - Subscribers are notified per entry, but the panel coalesces: see log-view.
 * - `t` is milliseconds since boot, not a wall clock. The interesting question
 *   is always "how long after the thing before it", and a wall clock makes the
 *   reader do subtraction.
 */

const CAP = 400;

export const LEVELS = { debug: 0, info: 1, good: 2, warn: 3, error: 4 };

class Log {
  constructor() {
    this.buf = new Array(CAP);
    this.n = 0;                 // total ever written; index = n % CAP
    this.subs = new Set();
    this.t0 = performance.now();
    this.timers = new Map();
  }

  /** Entries oldest-first, at most CAP of them. */
  entries() {
    if (this.n <= CAP) return this.buf.slice(0, this.n);
    const at = this.n % CAP;
    return this.buf.slice(at).concat(this.buf.slice(0, at));
  }

  get dropped() { return Math.max(0, this.n - CAP); }

  subscribe(fn) { this.subs.add(fn); return () => this.subs.delete(fn); }

  write(level, msg, detail) {
    const e = {
      i: this.n,
      t: performance.now() - this.t0,
      level,
      msg: String(msg),
      detail: detail == null ? null : detail,
    };
    this.buf[this.n % CAP] = e;
    this.n++;
    for (const fn of this.subs) fn(e);
    return e;
  }

  debug(m, d) { return this.write('debug', m, d); }
  info(m, d) { return this.write('info', m, d); }
  good(m, d) { return this.write('good', m, d); }
  warn(m, d) { return this.write('warn', m, d); }
  error(m, d) { return this.write('error', m, d); }

  /**
   * Time a span. `log.start('load')` … `log.end('load', 'model ready')`.
   *
   * Returns the elapsed ms so a caller can use the number as well as log it —
   * otherwise the duration exists only as text and has to be re-measured by
   * anything that wants to act on it.
   */
  start(key) { this.timers.set(key, performance.now()); }

  end(key, msg, level = 'good') {
    const t = this.timers.get(key);
    if (t === undefined) return this.write(level, msg);
    this.timers.delete(key);
    const ms = performance.now() - t;
    this.write(level, msg, fmtMs(ms));
    return ms;
  }

  clear() {
    this.buf = new Array(CAP);
    this.n = 0;
    for (const fn of this.subs) fn(null);
  }

  /** Plain text, for copying out. */
  toText() {
    const head = `Inside Diffusion — activity log\n${new Date().toISOString()}\n\n`;
    const body = this.entries()
      .map((e) => `${fmtT(e.t).padStart(8)}  ${e.level.padEnd(5)}  ${e.msg}`
        + (e.detail ? `  (${e.detail})` : ''))
      .join('\n');
    return head + body + '\n';
  }
}

export function fmtT(ms) {
  if (ms < 1000) return `${ms.toFixed(0)}ms`;
  const s = ms / 1000;
  if (s < 60) return `${s.toFixed(1)}s`;
  return `${Math.floor(s / 60)}m${String(Math.floor(s % 60)).padStart(2, '0')}s`;
}

export function fmtMs(ms) {
  return ms < 1000 ? `${ms.toFixed(0)} ms` : `${(ms / 1000).toFixed(2)} s`;
}

/** One instance, shared. A second log would split the story in half. */
export const log = new Log();

/**
 * Mirror uncaught failures into the log.
 *
 * Without this the log is a record of everything that went RIGHT, which is
 * precisely the case where nobody opens it. An unhandled rejection in the step
 * loop is the failure mode this app has actually had.
 */
export function captureErrors() {
  window.addEventListener('error', (e) => {
    log.error(e.message || 'script error',
      e.filename ? `${e.filename.split('/').pop()}:${e.lineno}` : null);
  });
  window.addEventListener('unhandledrejection', (e) => {
    const r = e.reason;
    log.error(r && r.message ? r.message : String(r), 'unhandled rejection');
  });
}
