/**
 * Draggable gutters for the view grid.
 *
 * The grid is `1fr 1fr` in each axis, which is the right DEFAULT and the wrong
 * only option: with the Funnel and the Landscape side by side you usually want
 * to give one of them more room, and there was no way to.
 *
 * WHY FRACTIONS RATHER THAN PIXELS. The panes are stored as a ratio, not a
 * width, so a resized window keeps the proportion the user chose instead of
 * stranding one pane at a fixed size. It also means the same ratio survives
 * switching between two and four panes.
 *
 * WHY POINTER CAPTURE. A drag that leaves the gutter — which it does
 * immediately, that is the point of dragging — stops receiving events without
 * it, and the pane freezes mid-resize while the button is still held.
 */

const MIN = 0.18;          // never let a pane collapse to nothing
const MAX = 0.82;

export class GridSplitter {
  /** @param grid the element carrying `grid-template-columns/rows` */
  constructor(grid) {
    this.grid = grid;
    this.col = 0.5;
    this.row = 0.5;
    this.bars = { col: null, row: null };
    this.build();
  }

  build() {
    for (const axis of ['col', 'row']) {
      const bar = document.createElement('div');
      bar.className = `gutter gutter-${axis}`;
      bar.setAttribute('role', 'separator');
      bar.setAttribute('aria-orientation', axis === 'col' ? 'vertical' : 'horizontal');
      bar.tabIndex = 0;
      bar.hidden = true;
      this.drag(bar, axis);
      this.grid.parentElement.appendChild(bar);
      this.bars[axis] = bar;
    }
  }

  drag(bar, axis) {
    let active = false;
    const clamp = (v) => Math.max(MIN, Math.min(MAX, v));

    bar.addEventListener('pointerdown', (e) => {
      active = true;
      bar.setPointerCapture(e.pointerId);
      bar.classList.add('dragging');
      e.preventDefault();
    });
    bar.addEventListener('pointermove', (e) => {
      if (!active) return;
      const r = this.grid.getBoundingClientRect();
      this[axis] = clamp(axis === 'col'
        ? (e.clientX - r.left) / r.width
        : (e.clientY - r.top) / r.height);
      this.apply();
    });
    const end = (e) => {
      if (!active) return;
      active = false;
      bar.classList.remove('dragging');
      if (e.pointerId != null && bar.hasPointerCapture(e.pointerId)) {
        bar.releasePointerCapture(e.pointerId);
      }
    };
    bar.addEventListener('pointerup', end);
    bar.addEventListener('pointercancel', end);

    // Keyboard, because a splitter that only responds to a drag is unusable
    // without a pointer and is trivially cheap to support.
    bar.addEventListener('keydown', (e) => {
      const step = e.shiftKey ? 0.10 : 0.02;
      const back = axis === 'col' ? 'ArrowLeft' : 'ArrowUp';
      const fwd = axis === 'col' ? 'ArrowRight' : 'ArrowDown';
      if (e.key === back) this[axis] = clamp(this[axis] - step);
      else if (e.key === fwd) this[axis] = clamp(this[axis] + step);
      else if (e.key === 'Home') this[axis] = 0.5;
      else return;
      e.preventDefault();
      this.apply();
    });

    bar.addEventListener('dblclick', () => { this[axis] = 0.5; this.apply(); });
  }

  /**
   * @param n how many panes are showing — decides which gutters exist at all.
   *          1 pane: none. 2: a vertical one. 3 or 4: both.
   */
  layout(n) {
    this.n = n;
    this.bars.col.hidden = n < 2;
    this.bars.row.hidden = n < 3;
    this.apply();
  }

  apply() {
    const g = this.grid;
    if (this.n >= 2) {
      g.style.gridTemplateColumns = `${this.col}fr ${1 - this.col}fr`;
    } else {
      g.style.gridTemplateColumns = '';
    }
    g.style.gridTemplateRows = this.n >= 3 ? `${this.row}fr ${1 - this.row}fr` : '';

    // The bars are positioned OVER the grid rather than being grid items:
    // as items they would occupy a track and change the very geometry they are
    // meant to describe, and a 3-pane layout has a full-height first column
    // that no single row gutter can align to.
    const pct = (v) => `${v * 100}%`;
    this.bars.col.style.left = pct(this.col);
    this.bars.row.style.top = pct(this.row);
    for (const axis of ['col', 'row']) {
      this.bars[axis].setAttribute('aria-valuenow', Math.round(this[axis] * 100));
    }
    // A three-pane layout keeps its first pane full height, so the horizontal
    // gutter must only span the right-hand column.
    this.bars.row.style.left = this.n === 3 ? pct(this.col) : '0';
  }
}
