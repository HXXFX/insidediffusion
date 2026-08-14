/**
 * Orbit camera for the 3-D views.
 *
 * THIS FILE EXISTS BECAUSE OF A BUG. The first build kept one shared pitch and
 * let each view scale it — landscape used `pitch`, voxels `pitch*0.75+0.12`,
 * funnel `pitch*0.55+0.20`. Two things went wrong:
 *
 *   1. The same drag produced three different angles.
 *   2. Top-down was unreachable in every view. With the clamp at 1.25 rad the
 *      ceilings were 71.6 / 60.6 / 50.8 degrees — worst in the hero view.
 *
 * The cause was conflating two separate things: where a view should START, and
 * where the user is ALLOWED to go. A default is a starting position. The clamp
 * is the limit. Each view gets its own default and its own camera instance;
 * the range is identical everywhere.
 *
 * CONVENTION, stated because it is easy to get backwards and hard to see:
 *   +Y is up. Increasing PITCH raises the camera, so at +90 deg you are
 *   directly above looking down, and a point higher in Y is CLOSER (smaller
 *   depth). Verify with test/camera.html if you touch the maths.
 */

import { CAMERA } from '../config.js';

export class Camera {
  constructor(defaults) {
    this.def = { yaw: 0.6, pitch: 0.42, dist: 3.4, ...defaults };
    this.reset();
  }

  reset() {
    this.yaw = this.def.yaw;
    this.pitch = this.def.pitch;
    this.dist = this.def.dist;
    this.panX = 0;
    this.panY = 0;
    this._dirty = true;
  }

  /**
   * Pan, in CSS pixels. Used by every view now, not just the flat one.
   *
   * The flat diagram needed it first — it cannot be orbited, but it very much
   * can be too small to read in a quarter-pane. The 3-D views need it for a
   * different reason: orbit and zoom together cannot put an off-centre feature
   * in the middle of the pane, so anything the projection puts near an edge is
   * unreachable without it.
   *
   * Keeping the offset on the camera rather than in the view means it survives
   * the same reset and the same double-click as everything else. The flat view
   * applies it itself through a canvas transform; the 3-D views get it from
   * `project()` below, so neither applies it twice.
   */
  pan(dx, dy) { this.panX += dx; this.panY += dy; }

  /**
   * Zoom as a multiplier for flat views, where `dist` has no perspective
   * meaning. Measured against THIS camera's own default, so it is exactly 1 at
   * rest — the first version divided by a hardcoded 3.4, "the shared default
   * distance", which is true of the three 3-D views and not of the network,
   * whose default is 1. It rendered the diagram at 3.4x, off the edges of its
   * own pane, and the view came up blank.
   */
  get flatScale() { return this.def.dist / this.dist; }

  set(yaw, pitch, dist) {
    this.yaw = yaw;
    // Just under vertical: at exactly +-90 deg the horizontal axes project
    // onto one line and the scene collapses to a stripe.
    this.pitch = Math.max(CAMERA.pitchMin, Math.min(CAMERA.pitchMax, pitch));
    /* The zoom range is RELATIVE to where this view starts, not the absolute
       [1.4, 9] it used to be. Those numbers were 0.4x and 2.57x the funnel's
       default of 3.5 — a shared clamp derived from one view — so the network,
       which starts at 1, could not zoom in at all: its first wheel click was
       clamped straight up to 1.4, i.e. zoomed OUT from rest. Proportional
       limits give every view the same range from wherever it begins, and leave
       the three 3-D views within a few percent of their previous bounds. */
    this.dist = Math.max(this.def.dist * 0.4, Math.min(this.def.dist * 2.6, dist));
    this._dirty = true;
  }

  orbit(dYaw, dPitch) { this.set(this.yaw + dYaw, this.pitch + dPitch, this.dist); }
  zoom(f) { this.set(this.yaw, this.pitch, this.dist * f); }

  /** Cache the trig: project() is called tens of thousands of times a frame. */
  _trig() {
    if (!this._dirty) return;
    this.cy = Math.cos(this.yaw); this.sy = Math.sin(this.yaw);
    this.cp = Math.cos(this.pitch); this.sp = Math.sin(this.pitch);
    this._dirty = false;
  }

  /**
   * World -> screen. Returns [sx, sy, depth] in canvas pixels.
   * `depth` is for painter's-algorithm sorting: larger is further away.
   */
  project(x, y, z, cx, cyc, scale, out) {
    this._trig();
    const x1 = x * this.cy - z * this.sy;
    const z1 = x * this.sy + z * this.cy;
    const y2 = y * this.cp + z1 * this.sp;
    const z2 = -y * this.sp + z1 * this.cp;
    const d = z2 + this.dist;
    const f = (2.6 / Math.max(0.15, d)) * scale;
    // Pan is added in SCREEN space, after projection, so it slides the finished
    // picture rather than moving the camera through the scene — a world-space
    // translation would change the perspective and the depth sort with it.
    const px = this.panX, py = this.panY;
    if (out) { out[0] = cx + x1 * f + px; out[1] = cyc - y2 * f + py; out[2] = d; return out; }
    return [cx + x1 * f + px, cyc - y2 * f + py, d];
  }

  get pitchDegrees() { return Math.round((this.pitch * 180) / Math.PI); }
}

/**
 * Wire pointer input to a camera on one canvas.
 *
 * `pointermove` is {passive:true} and does nothing but arithmetic, so dragging
 * stays smooth while the worker is busy. Double-click resets — the controls
 * hint promises "double-click to fit" and in the first build that was never
 * wired up, which your bug report also surfaced.
 */
export function attachCamera(canvas, camera, onChange, { flat = false } = {}) {
  let drag = null;

  /* WHICH BUTTON DOES WHAT, and why it is read at pointerdown.
     Left drag is the view's primary gesture — orbit in 3-D, pan in the flat
     diagram. RIGHT drag always pans, in every view, which is the convention
     every 3-D tool uses and the only way to bring an off-centre feature to the
     middle of a pane.
     `e.button` is only meaningful on the down event; `pointermove` reports a
     BITMASK in `e.buttons` and 0 in `e.button`, so deciding the mode per move
     event reads the wrong field and every drag behaves like a left drag. */
  canvas.addEventListener('pointerdown', (e) => {
    drag = { x: e.clientX, y: e.clientY, pan: e.button === 2 };
    canvas.setPointerCapture(e.pointerId);
    canvas.style.cursor = drag.pan ? 'move' : 'grabbing';
  });
  // Without this a right-drag ends on the browser's own menu, which also
  // swallows the pointerup and leaves the canvas stuck mid-drag.
  canvas.addEventListener('contextmenu', (e) => e.preventDefault());
  const end = (e) => {
    drag = null;
    canvas.style.cursor = 'grab';
    if (e && e.pointerId !== undefined && canvas.hasPointerCapture(e.pointerId)) {
      canvas.releasePointerCapture(e.pointerId);
    }
  };
  canvas.addEventListener('pointerup', end);
  canvas.addEventListener('pointercancel', end);

  canvas.addEventListener('pointermove', (e) => {
    if (!drag) return;
    if (flat || drag.pan) {
      // A flat view has nothing to orbit, so the same drag moves the diagram
      // under the window instead — one to one with the pointer, because
      // anything else feels like the content is sliding away from you. A right
      // drag does the same thing in a 3-D view.
      camera.pan(e.clientX - drag.x, e.clientY - drag.y);
    } else {
      // Drag DOWN raises the camera, which is the direction every 3-D tool
      // uses: you are pulling the model's far edge toward you.
      camera.orbit((e.clientX - drag.x) * 0.007, (e.clientY - drag.y) * 0.006);
    }
    drag.x = e.clientX; drag.y = e.clientY;
    onChange && onChange();
  }, { passive: true });

  canvas.addEventListener('wheel', (e) => {
    e.preventDefault();
    camera.zoom(Math.exp(e.deltaY * 0.0012));
    onChange && onChange();
  }, { passive: false });

  canvas.addEventListener('dblclick', () => { camera.reset(); onChange && onChange(); });

  canvas.style.cursor = 'grab';
  canvas.style.touchAction = 'none';   // or a drag scrolls the page on touch
}

export function makeCamera(viewId) {
  return new Camera(CAMERA.defaults[viewId] || {});
}
