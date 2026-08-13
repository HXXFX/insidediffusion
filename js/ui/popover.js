/**
 * Popovers, and the one rule that decides how each behaves.
 *
 * THE POLICY, applied everywhere:
 *
 *   hover (+ focus)   passive text you read and leave, in a panel that covers
 *                     nothing you were looking at: Runs locally.
 *   click             anything holding controls or scrollable history. Closes
 *                     on click-outside and on Escape: More settings, the log.
 *   hover AND click   every `?` in the app — hover to peek, click to pin. See
 *                     hoverClickPop below for why an explanation is neither of
 *                     the first two.
 *
 * The distinction is not stylistic. A hover panel containing a slider is a
 * trap — the pointer has to leave the trigger to reach the control, and any
 * gap on the way closes the thing being reached for. Conversely a click panel
 * for two sentences of help makes the reader open and then dismiss something
 * they only wanted to glance at.
 *
 * The second half of the rule was learned later. The panel explanations began
 * as text inserted into the inspector row, which reflowed the row every time
 * one opened; they now float over the viewport, and a 430px card that lands on
 * top of the visualisation is something you must be able to dismiss
 * deliberately rather than by keeping the pointer still.
 *
 * Both kinds must also work from the keyboard, which is why hover panels open
 * on focus-within and click panels close on Escape and restore focus.
 */

const openClick = new Set();

export function hoverPop(root, btn, panel) {
  let over = false;
  const show = (v) => {
    if (v === !panel.hidden) return;
    panel.hidden = !v;
    btn.setAttribute('aria-expanded', String(v));
  };
  const sync = () => show(over || root.contains(document.activeElement));

  root.addEventListener('pointerenter', () => { over = true; sync(); });
  root.addEventListener('pointerleave', () => { over = false; sync(); });
  root.addEventListener('focusin', sync);
  root.addEventListener('focusout', () => setTimeout(sync, 0));
  // Touch has no hover at all, so the trigger must still be operable by tap.
  btn.addEventListener('click', (e) => {
    e.preventDefault();
    over = panel.hidden;
    sync();
  });
  return { close: () => { over = false; show(false); } };
}

/**
 * Hover to peek, click to pin. Every `?` in the app uses this.
 *
 * The two behaviours were previously a choice between: help you could hover was
 * help you lost the moment the pointer drifted, and help you had to click was
 * two deliberate actions to answer a passing question. Neither is right for an
 * explanation, because both kinds of reader exist — the one glancing while
 * moving somewhere else, and the one who wants to sit and read it.
 *
 * So: hovering opens it unpinned and leaving closes it again; clicking pins it
 * open until a click elsewhere or Escape. A pinned panel ignores the pointer
 * leaving, which is what makes it possible to scroll a long explanation or
 * select text out of it.
 *
 * Touch has no hover, so there the click path is the whole interaction — and it
 * still works, because a tap is a click.
 */
export function hoverClickPop(root, btn, panel, { onOpen, onClose } = {}) {
  let pinned = false, overBtn = false, overPanel = false, timer = null;

  const show = (v) => {
    if (v === !panel.hidden) return;
    panel.hidden = !v;
    btn.setAttribute('aria-expanded', String(v));
    const fn = v ? onOpen : onClose;
    fn && fn();
  };

  /**
   * The hover region is the BUTTON PLUS THE PANEL, tracked separately, and not
   * the container both sit in. Using the container fails at both ends: for the
   * pane help it is the whole visualisation, so an unpinned card would stay up
   * while you rotated the model underneath it; for the inspector help the card
   * is positioned against a row it is not a child of, so the container never
   * sees the pointer over it at all.
   *
   * The delay covers the gap between the two. Some of these cards sit several
   * pixels away from the button that opens them, and closing the instant the
   * pointer crosses that gap makes the panel impossible to reach.
   */
  const sync = () => {
    clearTimeout(timer);
    if (pinned) return;
    if (overBtn || overPanel) show(true);
    else timer = setTimeout(() => show(false), 140);
  };
  const enter = (which) => () => {
    if (which === 'btn') {
      /* A peek displaces anything pinned: two explanations on screen at once is
         never what someone meant by moving the mouse.
         EXCEPT AN ANCESTOR. Several of these "?" buttons live INSIDE another
         open panel — the control tips are inside the settings popover — and
         displacing that closed the popover the moment you reached for help
         inside it, taking the tip with it. The panel you are standing in is not
         a rival. */
      for (const other of [...openClick]) {
        if (other !== api && !other.panel.contains(btn)) other.close();
      }
      overBtn = true;
    } else overPanel = true;
    sync();
  };
  const leave = (which) => () => {
    if (which === 'btn') overBtn = false; else overPanel = false;
    sync();
  };

  const api = {
    panel,
    get open() { return !panel.hidden; },
    close(focusBack) {
      pinned = false;
      openClick.delete(api);
      clearTimeout(timer);
      show(false);
      if (focusBack) btn.focus();
    },
    open() {
      for (const other of [...openClick]) {
        if (other !== api && !other.panel.contains(btn)) other.close();
      }
      pinned = true;
      openClick.add(api);
      show(true);
    },
    toggle() { pinned ? api.close() : api.open(); },
  };

  btn.addEventListener('pointerenter', enter('btn'));
  btn.addEventListener('pointerleave', leave('btn'));
  panel.addEventListener('pointerenter', enter('panel'));
  panel.addEventListener('pointerleave', leave('panel'));
  btn.addEventListener('focus', () => { overBtn = true; sync(); });
  btn.addEventListener('blur', () => {
    // Not while pinned, and not if focus went into the panel itself.
    setTimeout(() => {
      if (panel.contains(document.activeElement)) return;
      overBtn = false; sync();
    }, 0);
  });
  btn.addEventListener('click', (e) => { e.preventDefault(); e.stopPropagation(); api.toggle(); });
  panel.addEventListener('click', (e) => e.stopPropagation());
  return api;
}

export function clickPop(root, btn, panel, { onOpen, onClose } = {}) {
  const api = {
    panel,
    get open() { return !panel.hidden; },
    close(focusBack) {
      if (panel.hidden) return;
      panel.hidden = true;
      btn.setAttribute('aria-expanded', 'false');
      openClick.delete(api);
      onClose && onClose();
      if (focusBack) btn.focus();
    },
    open() {
      // One at a time. Two overlapping panels anchored to the same bar is
      // never what someone meant by opening the second.
      for (const other of [...openClick]) other.close();
      panel.hidden = false;
      btn.setAttribute('aria-expanded', 'true');
      openClick.add(api);
      onOpen && onOpen();
    },
    toggle() { panel.hidden ? api.open() : api.close(); },
  };

  btn.addEventListener('click', (e) => { e.stopPropagation(); api.toggle(); });
  // The panel swallows its own clicks, or every interaction inside it counts
  // as a click-outside and shuts the thing being used.
  panel.addEventListener('click', (e) => e.stopPropagation());
  return api;
}

/** Wire the document-level dismissals once. */
export function installDismiss() {
  document.addEventListener('click', () => {
    for (const p of [...openClick]) p.close();
  });
  document.addEventListener('keydown', (e) => {
    if (e.key !== 'Escape' || !openClick.size) return;
    for (const p of [...openClick]) p.close(true);
  });
}
