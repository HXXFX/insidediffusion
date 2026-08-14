/**
 * Wiring only. No maths, no drawing — if logic is accumulating here it belongs
 * in a module.
 *
 * Boot order matters and is not arbitrary: the overlay is in the MARKUP so it
 * covers the very first paint, and an inline watchdog in index.html replaces
 * it with an actionable message if this module never runs at all. An overlay
 * created by script is the one piece of UI that cannot rescue a module that
 * failed to load.
 */

import { APP, MODES, VIEWS, PANELS, PANEL_HELP } from './config.js';
import { Controls } from './ui/controls.js';
import { WordList } from './ui/wordlist.js';
import { PromptBuilder } from './ui/builder.js';
import { attachPaneHelp, helpBody } from './ui/help.js';
import { StepStrip, AttentionPanel, SchedulePanel } from './ui/panels.js';
import { makeCamera, attachCamera } from './views/camera.js';
import { loadProjection, loadCloudPictures } from './views/projection.js';
import { loadIndex } from './nn/weights.js';
import { Grammar } from './prompt.js';
import { loadDemo } from './demo.js';
import { TrainingReplay } from './training/replay.js';
import { TrainingStep } from './training/step.js';
import { HoldoutPanel } from './training/holdout.js';
import { log, captureErrors } from './log.js';
import { LogView } from './ui/logview.js';
import { hoverPop, clickPop, hoverClickPop, installDismiss } from './ui/popover.js';
import { GridSplitter } from './ui/splitter.js';
import { alphaBar } from './diffusion/schedule.js';
import * as funnel from './views/funnel.js';
import * as landscape from './views/landscape.js';
import * as voxels from './views/voxels.js';
import * as network from './views/network.js';

const MODULES = { funnel, landscape, voxels, network };

const $ = (s) => document.querySelector(s);
const state = {
  snaps: [],
  head: 0,
  playing: false,
  following: true,
  mode: 'make',
  active: new Set(VIEWS.filter((v) => v.default).map((v) => v.id)),
  ready: false,
  initial: null,       // img2img source, CHW [-1,1]
  stepMs: 0,
  // Re-run on every change, or wait for Run. Persisted, because it is a
  // working preference and not a property of any one session.
  live: localStorage.getItem('live') !== 'off',
  stale: false,
  // Set the moment a real run is asked for. The saved run checks it rather
  // than checking whether snapshots exist, because its own fetch can resolve
  // in the window after `started` has cleared them and before the first real
  // snapshot arrives — which installed a recording on top of a live run.
  runRequested: false,
  // Words that actually describe a picture. Filled from the manifest — "a" is
  // in the vocabulary but is not one of these.
  contentWords: new Set(),
};
const panes = new Map();
let controls, strip, attention, schedule, worker, projector, wordlist, builder;
let holdoutPanel;
/** Filled in after boot; see the loadCloudPictures call below. */
const cloudPics = { data: null };
let replay = null, logview = null, splitter = null, stepDemo = null;

// --------------------------------------------------------------- boot
async function boot() {
  captureErrors();
  log.info('Inside Diffusion starting');
  document.title = APP.name;
  await buildBrand();
  buildPlate();

  buildModes();
  buildViewSwitch();
  // BEFORE buildPanes: that ends by calling syncPanes(), which lays the
  // gutters out. Constructed after, the guard there silently skipped and the
  // gutters stayed hidden until the first time a view was toggled.
  splitter = new GridSplitter($('#viewGrid'));
  buildPanes();
  wireHint();

  controls = new Controls({
    model: $('#controlsModel'),
    prompt: $('#controlsPrompt'),
    sampling: $('#controlsSampling'),
    run: $('#runControls'),
  }, onControlChange);

  // Which models exist is read from disk, never hardcoded here. Done before the
  // first paint of the picker so it never shows an empty select.
  state.index = await loadIndex('./weights/');
  log.info(`${state.index.models.length} models available`,
    state.index.models.map((m) => m.name).join(', '));
  controls.setOptions('model', state.index.models.map((m) => ({
    value: m.name,
    label: `${m.label} — ${(m.bytes / 1e6).toFixed(1)} MB`,
    title: m.blurb,
  })), state.index.default);
  // Apply the starting mode immediately. Without this, mode-specific controls
  // (Change amount, which only means anything for image-to-image) are visible
  // on first paint and only hide once a tab is clicked.
  controls.setMode(state.mode, { image: !!state.initial });
  strip = new StepStrip($('#strip'), 16);
  attention = new AttentionPanel($('#attnCanvas'));
  schedule = new SchedulePanel($('#schedCanvas'), alphaBar);
  buildInspectorToggles();
  wireTransport();
  wireDropzone();
  wireChrome();

  setStatus('Loading the projection…', 0.05);
  log.start('proj');
  projector = await loadProjection('./weights/');
  log.end('proj', `projection loaded — ${projector.cloudN} points, ${projector.k} components`);

  /* A HOLDER, not the value, because the value is not here yet and boot must
     not wait for it. The views keep this object and read `.data` each frame:
     null means "draw plain marks", and the frame after the fetch lands they
     upgrade themselves with no reload and no re-create. Deliberately not
     awaited, and a failure leaves `.data` null rather than throwing — the
     pictures are an enrichment of the views, never a requirement of them. */
  for (const v of VIEWS) {
    const mod = MODULES[v.id];
    panes.get(v.id).renderer = mod.create({ projector, imgSize: 16, cloudPics });
  }
  loadCloudPictures('./weights/').then((p) => {
    cloudPics.data = p;
    if (p) log.info(`${p.n} training pictures loaded for the floor and the map`);
  });

  startWorker(controls.get('model'));
  requestAnimationFrame(frame);
}

/**
 * (Re)start the inference worker on a given model.
 *
 * Switching model replaces the worker rather than reloading weights inside it.
 * A model is a different UNet with different channel widths, so every cached
 * buffer in there is the wrong size; tearing it down is both simpler and
 * impossible to get subtly wrong.
 */
function startWorker(name) {
  if (worker) worker.terminate();
  state.ready = false;
  const m = state.index.models.find((x) => x.name === name) || state.index.models[0];
  setStatus(`Loading the ${m.label} model…`, 0.1);
  log.start('model');
  log.info(`loading the ${m.label} model`,
    `${(m.bytes / 1e6).toFixed(1)} MB · ${m.params.toLocaleString()} numbers`);
  worker = new Worker(new URL('./worker.js', import.meta.url), { type: 'module' });
  worker.onmessage = onWorkerMessage;
  worker.onerror = (e) => fail(`worker failed: ${e.message || 'unknown error'}`);
  // ABSOLUTE base, deliberately. Inside a worker, fetch() resolves a relative
  // URL against the WORKER's location (js/), not the document's — so
  // './weights/' asked for js/weights/, got the 404 page, and failed as
  // "Unexpected token '<'". Resolve it here, where the document base is right.
  worker.postMessage({
    type: 'init',
    base: new URL(`./weights/${m.name}/`, location.href).href,
  });
}

function onWorkerMessage(e) {
  const m = e.data;
  if (m.type === 'loading') {
    setStatus('Loading the model…', 0.1 + m.progress * 0.85);
  } else if (m.type === 'manifest') {
    // Arrives BEFORE the 2.4 MB weight blob. The vocabulary is the first thing
    // a newcomer needs, so show it immediately rather than making them wait
    // for a download to find out which words exist.
    // Every word that names a part, flattened out of the slot lists. `none` is
    // in there and is a real answer — "no wings" is a choice, not a gap.
    const byslot = m.manifest.caption.words;
    state.contentWords = new Set(Object.values(byslot).flat());
    state.manifest = m.manifest;
    $('#logWords').textContent = String(state.contentWords.size);
    $('#modelInfo').textContent =
      `${m.manifest.name} · ${(m.manifest.total_bytes / 1e6).toFixed(1)} MB · `
      + `${state.contentWords.size} words`;
    $('#modelInfo').hidden = false;
    // The vocabulary comes from the manifest, never a hardcoded copy. The
    // grammar is built here rather than reaching into the worker's copy —
    // same manifest, same rules, and the main thread needs it for the builder.
    state.grammar = new Grammar(m.manifest.caption,
      new Map(m.manifest.vocab.map((w, i) => [w, i])));

    // The sentence builder is the input. Its slot was reserved when the panel
    // was built; the vocabulary needed to fill it only exists now.
    builder = new PromptBuilder(controls.slot('prompt'), state.grammar,
      (p) => setPrompt(p, builder));
    wordlist = new WordList($('#wordlist'), state.grammar,
      (p) => setPrompt(p, wordlist));
    setPrompt(controls.get('prompt'), null, true);
  } else if (m.type === 'ready') {
    state.ready = true;
    log.end('model', `${m.manifest ? m.manifest.name : 'model'} ready`);
    hideStatus();
    revealChrome();
    // OPEN ON SOMETHING WORKING. The saved run appears immediately, so the
    // first frame is a finished monster rather than an empty viewport — and
    // then, if live update is on, a real run starts and takes over. With live
    // off the recording is what you look at until you press Run.
    showDemo(m.manifest);
    if (state.live) run();
    else markStale(true);
  } else if (m.type === 'started') {
    state.snaps = [];
    state.head = 0;
    state.following = true;
    state.playing = true;
    attention.setWords(m.tokenWords);
    reportPrompt(m);
    setTransport(true);
    log.start('run');
    log.info(m.sentence || 'run started',
      `${m.total} steps · guidance ${controls.get('guidance')}`
      + (state.initial ? ` · from a picture at ${controls.get('strength')}` : ''));
    if (m.unknown && m.unknown.length) {
      log.warn(`ignored: ${m.unknown.join(', ')}`, 'not in the vocabulary');
    }
  } else if (m.type === 'trainStep') {
    // The Making of tab's live step. Arrives out of band from any run — the
    // worker answers it immediately rather than queueing behind a sampling
    // loop, so opening the tab mid-run does not stall either one.
    if (stepDemo) stepDemo.show(m);
  } else if (m.type === 'snapshot') {
    state.snaps.push(m);
    if (state.following) state.head = state.snaps.length - 1;
    $('#scrub').max = String(Math.max(1, state.snaps.length - 1));
  } else if (m.type === 'done') {
    state.playing = false;
    state.stepMs = m.medianStepMs;
    setTransport(false);
    $('#stepMs').textContent = `${Math.round(m.medianStepMs)} ms/step`;
    log.end('run', `finished — ${Math.round(m.medianStepMs)} ms per step`);
  } else if (m.type === 'error') {
    log.error(m.message);
    fail(m.message);
  }
}

/**
 * The mark and the wordmark, inlined so they inherit the palette.
 *
 * Both are generated by tools/make_brand.py from the same pixel grid the model
 * draws in — the mark IS a monster the composer built, not an illustration of
 * one, and the wordmark is set in a 5x7 bitmap face defined in that script. A
 * geometric sans here said "developer tool"; the product is pixels, so the
 * brand is pixels.
 *
 * Inlined rather than <img> because an external SVG cannot inherit
 * `currentColor`, and the whole point is that they follow the theme.
 */
async function buildBrand() {
  const grab = async (u) => (await fetch(u)).text();
  try {
    const [mark, word] = await Promise.all([
      grab('./img/mark.svg'), grab('./img/wordmark.svg'),
    ]);
    $('#brandMark').innerHTML = mark;
    $('#brandWord').innerHTML = word;
  } catch {
    // A missing brand asset must not take the app down with it.
    $('#brandWord').textContent = APP.name.toUpperCase();
  }
}

/**
 * The controls hint: fades in when the pointer nears the bottom of the
 * viewport, out again when it leaves.
 *
 * MEASURED IN JAVASCRIPT, NOT CSS, and the reason is specific rather than
 * stylistic — see the note on `.hint` in app.css. The short version: the CSS
 * build needs a hover zone that is simultaneously `pointer-events: none` (so
 * it does not swallow drags) and hit-testable (so `:hover` matches). Those
 * requirements contradict each other, the rule parses, and nothing ever
 * appears.
 *
 * `classList.toggle(name, bool)` no-ops when the answer has not changed, so
 * ordinary movement across the middle of the viewport touches no styles.
 */
const HINT_MS = 2600;      // how long it stays after the pointer arrives

function wireHint() {
  const port = $('.viewport');
  const hint = $('#hint');
  let timer = 0;
  /* ON ARRIVAL, THEN AWAY — not on proximity to an edge.
     The proximity version keyed on the BOTTOM of the viewport, because that is
     where the hint used to sit. Now that it lives in the top-left chrome
     corner, "hover near the bottom, text appears at the top" would be a
     puzzle, and a zone near the top would fight the view switch.
     Arriving in the viewport is the moment the hint is for: you are about to
     drag something. It says how, then gets out of the way on its own. */
  port.addEventListener('pointerenter', () => {
    hint.classList.add('on');
    clearTimeout(timer);
    timer = setTimeout(() => hint.classList.remove('on'), HINT_MS);
  });
  // Leaving hides it at once, and cancels the timer — otherwise a re-entry
  // inside the window inherits the old countdown and flashes.
  port.addEventListener('pointerleave', () => {
    clearTimeout(timer);
    hint.classList.remove('on');
  });
}

/** The finished picture, drawn big enough to be found. */
function drawResult(snap) {
  const cv = $('#resultCanvas');
  const g = cv.getContext('2d');
  if (!snap) return;
  const n = 256;
  const img = g.createImageData(16, 16);
  for (let i = 0; i < n; i++) {
    for (let c = 0; c < 3; c++) {
      const v = Math.max(-1, Math.min(1, snap.x0[c * n + i]));
      img.data[i * 4 + c] = Math.round((v + 1) * 127.5);
    }
    img.data[i * 4 + 3] = 255;
  }
  g.putImageData(img, 0, 0);
  const done = snap.index + 1 >= snap.total;
  // It shows the model's CURRENT GUESS at the finished picture, which is a
  // real thing at every step and not only at the end — saying so is what stops
  // a half-formed monster mid-run reading as the final answer.
  // SHORT, because this caption sits in a 92px column and its height decides
  // the height of the whole inspector row. "its guess so far · step 28 of 30"
  // wrapped to three lines while "finished" is one, so arriving at the end
  // shrank the row by 14px and dropped everything below it — see the reserved
  // height on #panel-result figcaption, which is the other half of the fix.
  $('#resultNote').textContent = snap.demo ? 'saved example'
    : done ? 'finished'
    : `its guess · ${snap.index + 1} of ${snap.total}`;
}

// --------------------------------------------------------------- modes
/**
 * Show the chrome that belongs to a mode.
 *
 * The training mode REPLACES the view grid rather than joining it. The two are
 * not comparable — the grid is one real run of the image model, this is a
 * different, much smaller model learning a 2-D distribution — and side by side
 * they would invite exactly the misreading the Landscape view already needs a
 * caveat about.
 */
function applyMode(id) {
  const training = id === 'train';
  $('#train2d').hidden = !training;
  $('#viewGrid').hidden = training;
  // The gutters float OVER the grid rather than inside it, so hiding the grid
  // leaves them drawing two lines across the replay. Coming back, `layout()`
  // decides which of them belongs on screen for the current pane count.
  if (training) for (const g of document.querySelectorAll('.gutter')) g.hidden = true;
  else if (splitter) splitter.layout(state.active.size);
  $('#viewSwitch').hidden = training;
  $('#hint').hidden = training;
  $('#inspectors').hidden = training;
  $('.timeline').hidden = training;
  $('.hud').hidden = training || !state.ready;
  $('#dock').hidden = training;

  if (training) {
    startTraining();
  } else {
    // Stop the replay when the tab is left, or it goes on ticking unseen and
    // is mid-run when you come back to it.
    if (replay) { replay.stop(); $('#trainPlay').textContent = 'Play the run'; }
    // Coming back to Make does not spend four seconds unasked either — the
    // snapshots from the last run are still there to look at.
    if (!state.snaps.length) autoRun();
  }
}

/**
 * Act 4, the withheld combinations.
 *
 * Its own loader for the same reason the replay has one: 28 kB nobody browsing
 * the Make tab ever needs. It is also the only section whose data is a RESULT
 * rather than a recording, so a failure to load must not take the rest of the
 * tab with it — the section's honesty note is in the markup and survives, and
 * the panel simply says the measurement is unavailable.
 */
function startHoldout() {
  if (holdoutPanel) return;
  holdoutPanel = new HoldoutPanel({
    rows: $('#hoRows'), scale: $('#hoScale'), verdict: $('#hoVerdict'),
  });
  holdoutPanel.load('./weights/')
    .then(() => { holdoutPanel.build(); log.info('held-out results loaded'); })
    .catch((e) => {
      $('#hoVerdict').textContent =
        'The measurement could not be loaded, so nothing is claimed here.';
      log.warn(`held-out results unavailable — ${e.message}`);
    });
}

/**
 * Load and show the real run's history for whichever model is selected.
 *
 * Lazily, and once: it is ~200 kB per model and most visits never open this
 * tab, so paying for it during the initial load would slow down the thing
 * everybody does see.
 */
async function startTraining() {
  const run = controls.get('model');
  if (!replay) {
    replay = new TrainingReplay({
      strip: $('#trainStrip'), loss: $('#trainLoss'),
      scrub: $('#trainScrub'), end: $('#trainEnd'), blurb: $('#trainBlurb'),
      phases: $('#trainPhases'), facts: $('#trainFacts'),
    });
    $('#trainScrub').oninput = (e) => {
      replay.stop();
      $('#trainPlay').textContent = 'Play the run';
      replay.setIndex(parseInt(e.target.value, 10));
    };
    $('#trainPlay').onclick = () => {
      const btn = $('#trainPlay');
      if (replay.playing) { replay.stop(); btn.textContent = 'Play the run'; return; }
      replay.play((running) => { btn.textContent = running ? 'Pause' : 'Play again'; });
    };
    // Jumping by phase button stops playback, so the button has to stop
    // claiming it is playing.
    replay.onJump = () => { $('#trainPlay').textContent = 'Play the run'; };
  }
  // The facts rail reads the parameter count and download size straight off the
  // loaded model's manifest rather than keeping a second copy.
  replay.manifest = state.manifest;

  startHoldout();
  startStepDemo();

  if (replay.loaded === run) { replay.setIndex(replay.at); return; }
  try {
    log.start('hist');
    const meta = await replay.load(run);
    replay.loaded = run;
    log.end('hist', `training history for '${run}' — ${meta.steps.length} marks`);
  } catch (err) {
    $('#trainBlurb').textContent =
      'No training history was exported for this model. Run train/export_monsters.py.';
  }
}

/**
 * Act 1: the live training step.
 *
 * Built once and kept — it holds the exported examples and its own canvases,
 * and the model it drives is the one already loaded for the Make tab. It runs
 * a step on open so the panel is never an empty frame waiting to be operated;
 * the first thing you see is a finished step you can then change.
 */
async function startStepDemo() {
  if (!state.manifest) return;
  if (!stepDemo) {
    stepDemo = new TrainingStep({
      pick: $('#stepPick'), colour: $('#stepColour'),
      noise: $('#stepNoise'), noiseOut: $('#stepNoiseOut'), run: $('#stepRun'),
      tiles: $('#stepTiles'), loss: $('#stepLoss'), gauge: $('#stepGauge'),
      sentence: $('#stepSentence'), verdict: $('#stepVerdict'),
      errGain: $('#stepErrGain'),
    }, (msg) => worker.postMessage(msg, msg.x0 ? [msg.x0] : []));
    try {
      log.start('examples');
      const d = await stepDemo.load();
      stepDemo.setPalette(state.manifest);
      stepDemo.build();
      log.end('examples', `${d.examples.length} real training pictures`);
    } catch (err) {
      stepDemo = null;
      $('#stepVerdict').textContent =
        'No training examples were exported. Run train/export_monsters.py.';
      log.warn('training examples missing', String(err && err.message));
      return;
    }
  } else {
    stepDemo.setPalette(state.manifest);
  }
  stepDemo.run();
}

// --------------------------------------------------------------- chrome
/**
 * The credit plate, built from APP rather than written into the markup.
 *
 * It lives at the bottom of the "Runs locally" panel. That is not filing: who
 * made a thing and the promise that it does not phone home are the same
 * question asked twice, and the stamp is the answer to the first.
 *
 * Links open in a new tab: this app holds unsaved state — a run in progress, a
 * dropped image, a camera angle — and navigating away throws all of it out with
 * no warning.
 *
 * A link is only rendered when its URL is set, so removing `repo` from the
 * config removes the link rather than leaving a dead one on screen. That
 * matters here because `repo` is currently a placeholder.
 */
function buildPlate() {
  const plate = $('#plate');
  plate.innerHTML = '';

  // Every outbound link opens in a new tab, and the two attributes travel
  // together — `rel` alone does nothing, `target` alone hands the opened page a
  // handle back into this one. The app holds unsaved work (a run in progress, a
  // dropped image, a camera angle) that the back button cannot recover, so a
  // same-tab navigation from here silently destroys it.
  // The aria-label is non-visual: the plate keeps its single grey and gains no
  // visual vocabulary of its own, while a screen reader still announces the jump.
  const link = (href, text) => {
    const a = document.createElement('a');
    a.href = href;
    a.textContent = text;
    a.target = '_blank';
    a.rel = 'noopener noreferrer';
    a.setAttribute('aria-label', `${text} (opens in a new tab)`);
    return a;
  };

  // The provenance line stays UNBROKEN — it is one stamp, and a wrapped stamp
  // reads as a sentence that ran out of room. The CSS pins it with `nowrap`
  // and tightens the tracking to fit; if a longer name ever pushes it out,
  // tighten further rather than letting it wrap onto two lines.
  const line = document.createElement('span');
  line.append(`${APP.name.toUpperCase()} · A `);
  line.append(APP.imprintUrl ? link(APP.imprintUrl, APP.imprint) : APP.imprint);
  line.append(' SOLUTION · ');
  // The handle points at THIS PROJECT'S repository rather than the profile:
  // from inside the app the useful destination is the source of the thing you
  // are looking at, and the repo header carries the handle anyway. One route
  // per destination — this is why there is no separate "Source" row.
  line.append(APP.repo ? link(APP.repo, APP.author) : APP.author);
  plate.appendChild(line);

  // The copyright is a SECOND LINE OF THE SAME STAMP, not a separate element:
  // same type, same grey, and NO LINK ON ANY PART OF IT. A notice is not a
  // navigation surface — a line whose every word might be clickable stops
  // reading as a statement and starts reading as a menu. The terms stay one hop
  // away through the handle above, which points at the repository the LICENSE
  // sits in.
  const rights = document.createElement('span');
  rights.append(`© ${APP.year} ${APP.imprint} · ${APP.licence.toUpperCase()}`);
  plate.appendChild(rights);
}

/**
 * The status bar: About, the log, and the popover policy.
 *
 * About is HOVER because it is passive text. Settings and the log are CLICK
 * because they hold controls and scrollable history — see js/ui/popover.js for
 * why that distinction is not stylistic.
 */
function wireChrome() {
  installDismiss();

  hoverPop($('#privacy'), $('#privBtn'), $('#privPanel'));

  logview = new LogView({
    list: $('#logList'), count: $('#logCount'), panel: $('#logPanel'),
  });
  const logPop = clickPop($('#logpop'), $('#logBtn'), $('#logPanel'), {
    onOpen: () => logview.setOpen(true),
    onClose: () => logview.setOpen(false),
  });
  $('#logCopy').onclick = async () => {
    try {
      await navigator.clipboard.writeText(log.toText());
      announce('Log copied to the clipboard.');
    } catch {
      log.warn('could not copy — the browser refused clipboard access');
    }
  };
  $('#logClear').onclick = () => { log.clear(); log.info('log cleared'); };

  clickPop($('#settings'), $('#settingsBtn'), $('#settingsPanel'));
  // Anchored right: a 320px panel hanging left from a button near the window
  // edge would sit half off-screen.
  $('#settingsPanel').classList.add('right');
  void logPop;
}

function buildModes() {
  const el = $('#modes');
  el.innerHTML = '';
  for (const m of MODES) {
    const b = document.createElement('button');
    b.className = 'tab';
    b.textContent = m.label;
    b.setAttribute('aria-selected', String(m.id === state.mode));
    b.title = m.desc;
    if (m.soon) {
      b.classList.add('soon');
      b.disabled = true;
      b.setAttribute('aria-disabled', 'true');
      const tag = document.createElement('em');
      tag.textContent = 'soon';
      b.appendChild(tag);
      el.appendChild(b);
      continue;
    }
    b.onclick = () => {
      state.mode = m.id;
      for (const c of el.children) c.setAttribute('aria-selected', String(c === b));
      controls.setMode(m.id, { image: !!state.initial });
      applyMode(m.id);
    };
    el.appendChild(b);
  }
}

/**
 * The view switcher.
 *
 * These are TOGGLES, not tabs, and the first build did not say so — four
 * buttons in a row read as "pick one", so nobody discovered that several can
 * be on at once. Now each carries a tick box, the group is labelled "Show",
 * and the count is stated. "Solo" is a different KIND of control — it is an
 * action, not a state — so it is separated by a rule and styled differently
 * rather than sitting in the row looking like a fifth view.
 */
function buildViewSwitch() {
  const el = $('#viewSwitch');
  el.innerHTML = '';

  // NO "Show" LABEL. Four tick boxes in a row over a visualisation are already
  // legible as "pick which of these to see", and the word only widened the bar
  // — which floats over the picture — to caption something obvious. The group
  // is still named for screen readers by the aria-label on the container.
  for (const v of VIEWS) {
    const b = document.createElement('button');
    b.dataset.v = v.id;
    b.title = `${v.label} — ${v.short}`;
    b.innerHTML = `<i class="tick" aria-hidden="true"></i>${v.label}`;
    b.setAttribute('aria-pressed', String(state.active.has(v.id)));
    b.onclick = () => {
      if (state.active.has(v.id)) state.active.delete(v.id);
      else state.active.add(v.id);
      // Never leave the viewport empty — a blank pane reads as a crash.
      if (!state.active.size) state.active.add(v.id);
      syncPanes();
    };
    el.appendChild(b);
  }

  // NO "Only this" BUTTON. It collapsed to whichever view came first in VIEWS,
  // not to the one you were looking at, so from any other selection it behaved
  // as "reset to the Funnel" — a button whose label described something it did
  // not do. Deselecting the others is one extra click and is unambiguous.
}

function buildPanes() {
  const grid = $('#viewGrid');
  grid.innerHTML = '';
  for (const v of VIEWS) {
    const pane = document.createElement('div');
    pane.className = 'pane';
    const cv = document.createElement('canvas');
    const lab = document.createElement('span');
    lab.className = 'plabel';
    // The `.pnote` line is filled by the view itself each frame — see note()
    // in views/draw.js. It is the view's caveat about its own honesty, and it
    // belongs directly under the name and description it qualifies.
    lab.innerHTML = `<span class="pmain"><b>${v.label}</b><em>${v.short}</em></span>`
      + `<span class="pnote"></span>`;
    pane.appendChild(cv); pane.appendChild(lab);
    // Explanation attached to the view it explains, not filed in an About box.
    attachPaneHelp(pane, v.label, v.help);
    grid.appendChild(pane);
    const cam = makeCamera(v.id);
    // The network is a computation graph, so it pans and zooms rather than
    // orbiting — rotating a diagram means nothing, but it was also the only
    // pane you could not move at all, which read as broken rather than as flat.
    attachCamera(cv, cam, () => { }, { flat: v.id === 'network' });
    panes.set(v.id, { pane, canvas: cv, cam, renderer: null });
  }
  syncPanes();
}

function syncPanes() {
  for (const [id, p] of panes) p.pane.hidden = !state.active.has(id);
  $('#viewGrid').dataset.n = String(state.active.size);
  for (const b of $('#viewSwitch').querySelectorAll('button[data-v]')) {
    b.setAttribute('aria-pressed', String(state.active.has(b.dataset.v)));
  }
  if (splitter) splitter.layout(state.active.size);
}

function buildInspectorToggles() {
  const el = $('#inspect');
  el.innerHTML = '';
  for (const p of PANELS) {
    const row = document.createElement('div');
    row.className = 'chkrow';

    const lab = document.createElement('label');
    lab.className = 'chk';
    const box = document.createElement('input');
    box.type = 'checkbox';
    box.checked = p.default;
    box.onchange = () => {
      const target = $(`#panel-${p.id}`);
      if (target) target.hidden = !box.checked;
      // Attention needs the internals captured, which costs memory per step,
      // so it is only collected when the panel is actually on.
      if (p.id === 'attention') autoRun();
    };
    lab.appendChild(box);
    const txt = document.createElement('span');
    txt.innerHTML = `${p.label}<em>${p.blurb}</em>`;
    lab.appendChild(txt);
    row.appendChild(lab);
    el.appendChild(row);

    const target = $(`#panel-${p.id}`);
    if (target) target.hidden = !p.default;
  }

  /**
   * Put an inspector explanation under the "?" that opened it.
   *
   * These are absolutely positioned against the inspector ROW, not against
   * their own panel, because a panel is only as wide as its content and the
   * card is 430px — anchored to the panel, the two on the right would hang off
   * the window. So the row is the containing block and the left edge is
   * computed: line it up with its button, then clamp it inside the row. Left at
   * the CSS default the cross-attention card opened at the far left of the row,
   * a third of the window away from the button that asked for it.
   */
  function positionInline(btn, body) {
    const row = $('#inspectors').getBoundingClientRect();
    const b = btn.getBoundingClientRect();
    const PAD = 14;
    // The width is COMPUTED, not measured: `body.offsetWidth` is 0 while the
    // panel is still hidden, and measuring after revealing it means the first
    // frame paints at the previous card's position. This mirrors the CSS —
    // `width: min(430px, calc(100% - 28px))` — so the two must change together.
    const w = Math.min(430, row.width - PAD * 2);
    const want = b.left - row.left - 12;
    body.style.left = `${Math.round(Math.max(PAD, Math.min(want, row.width - w - PAD)))}px`;
  }

  // Each 2-D panel gets the same explanation treatment as the 3-D views.
  for (const [id, h] of Object.entries(PANEL_HELP)) {
    const host = $(`#panel-${id}`);
    if (!host) continue;
    // Every panel must own an <h4>. Falling back to the host itself was the
    // first version and it threw: `host.insertBefore(body, host.nextSibling)`
    // asks a node to insert before its own SIBLING, which is not its child.
    const head = host.querySelector('h4');
    if (!head) continue;
    const btn = document.createElement('button');
    btn.className = 'phelp inline';
    btn.type = 'button';
    btn.textContent = '?';
    const panelLabel = (PANELS.find((p) => p.id === id) || {}).label || id;
    btn.setAttribute('aria-label', `What am I looking at? — ${panelLabel}`);
    const body = helpBody(panelLabel, h);
    body.classList.add('hpanel', 'inline');
    body.hidden = true;
    // Same as every other "?": hover to peek, click to pin, dismissed by
    // clicking away or by Escape. Positioning has to happen BEFORE the panel is
    // shown or the first frame flashes at the previous card's left edge.
    hoverClickPop(host, btn, body, {
      onOpen: () => { btn.classList.add('on'); positionInline(btn, body); },
      onClose: () => btn.classList.remove('on'),
    });
    head.appendChild(btn);
    host.insertBefore(body, head.nextSibling);
  }
}

/**
 * Show the saved run, if this model ships one.
 *
 * Loaded lazily and never awaited by anything that matters: it is 90 kB and a
 * nicety, so a slow or missing file costs a log line and nothing else. It only
 * paints while there is nothing better on screen — a real run that finishes
 * first wins, and so does one already in progress.
 */
async function showDemo(manifest) {
  if (!manifest || !manifest.demo) return;
  try {
    const base = new URL(`./weights/${manifest.name}/`, location.href).href;
    const snaps = await loadDemo(manifest, base);
    if (!snaps || !snaps.length) return;
    // A real run was asked for while this was loading. It wins: it is the
    // thing the user is actually waiting for, and it fills the viewport with
    // its first step in about 130 ms anyway.
    if (state.runRequested || state.snaps.length) return;
    state.snaps = snaps;
    state.head = snaps.length - 1;
    state.following = false;
    attention.setWords(manifest.demo.words || []);
    $('#scrub').max = String(snaps.length - 1);
    $('#scrub').value = String(state.head);
    log.info('showing a saved run', `${snaps.length} steps · ${manifest.demo.prompt}`);
  } catch (err) {
    log.warn('no saved run to show', String(err && err.message));
  }
}

// --------------------------------------------------------------- transport
function wireTransport() {
  $('#play').onclick = () => {
    // In the training mode there is no finished state to run again from —
    // training pauses and resumes, and Restart throws the model away.
    if (state.mode === 'train') return;   // nothing to run here
    if (state.snaps.length && state.head < state.snaps.length - 1) {
      state.playing = !state.playing;
      setTransport(state.playing);
    } else {
      run();
    }
  };
  $('#restart').onclick = () => {
    run();
  };

  /**
   * Live update on or off.
   *
   * Turning it ON runs immediately if the controls have moved since the last
   * run — otherwise the button would appear to do nothing, which is the worst
   * kind of toggle. Turning it OFF changes nothing on screen; the picture you
   * are looking at stays the one you asked for.
   */
  const liveBtn = $('#live');
  const syncLive = () => {
    liveBtn.setAttribute('aria-pressed', String(state.live));
    liveBtn.title = state.live
      ? 'Re-running automatically when you change something'
      : 'Changes wait for Run';
  };
  liveBtn.onclick = () => {
    state.live = !state.live;
    localStorage.setItem('live', state.live ? 'on' : 'off');
    syncLive();
    log.info(`live update ${state.live ? 'on' : 'off'}`,
      state.live ? 'every change re-runs' : 'changes wait for Run');
    if (state.live && state.stale) run();
  };
  syncLive();
  $('#scrub').oninput = (e) => {
    state.head = parseInt(e.target.value, 10);
    state.following = state.head >= state.snaps.length - 1;
    state.playing = false;
    setTransport(false);
  };
}

function setTransport(playing) {
  $('#play').textContent = playing ? 'Pause' : (isFinished() ? 'Run again' : 'Play');
}

const isFinished = () => !state.snaps.length || state.head >= state.snaps.length - 1;

/**
 * One prompt, two views of it — the sentence and the word list — kept in step.
 *
 * `from` is whichever of them the user just touched, and is skipped so it is
 * not reset underneath their cursor mid-interaction. Everything else adopts the
 * new value, so the two can never drift apart and disagree about what is about
 * to be generated.
 */
function setPrompt(p, from, silent) {
  controls.set('prompt', p, true);
  if (builder && builder !== from) builder.syncFrom(p);
  if (wordlist && wordlist !== from) wordlist.syncFrom(p);
  if (!silent) autoRun();
}

function onControlChange(id, v) {
  if (id === 'speed') return;              // playback only, no rerun
  // A different model is a different network, so it needs a fresh worker and a
  // fresh download. `ready` goes false until it arrives, which stops `run()`
  // firing against the model that is on its way out.
  if (id === 'model') {
    log.info(`switching to the ${v} model`);
    startWorker(v);
    // The training tab shows THIS model's run, so switching model switches the
    // history too — otherwise the tab would quietly describe the other one.
    if (state.mode === 'train') startTraining();
    return;
  }
  // `prompt` never reaches here: the sentence builder and the word list both
  // route through setPrompt(), which syncs the other view before running.
  autoRun();
}

/**
 * Re-run because something CHANGED, rather than because Run was pressed.
 *
 * Every caller that fires on an edit goes through here, so "live update" is one
 * decision in one place instead of a condition repeated at nine call sites. If
 * live is off the change is kept — it is already in `controls` — and the only
 * thing that does not happen is spending four seconds on it unasked.
 */
function autoRun() {
  if (state.live) return run();
  markStale(true);
}

/** Say plainly that what is on screen is no longer what the controls describe. */
function markStale(v) {
  state.stale = v && state.ready;
  $('#play').classList.toggle('stale', state.stale);
  $('.hud').classList.toggle('stale', state.stale);
  if (state.stale) {
    // "Changed" only means something once there is a picture it disagrees
    // with. On a first visit with live update off, nothing has changed — the
    // app simply has not run yet, and saying otherwise invents a history.
    $('#phase').textContent = state.snaps.length ? 'Changed' : 'Ready';
    $('.hud').classList.remove('done');
  }
}

function run() {
  if (!state.ready) return;
  state.runRequested = true;
  markStale(false);
  // Capturing internals costs memory on every step, so only do it when the
  // panel that consumes them is actually on. Look the checkbox up by the
  // panel it belongs to, not by DOM position — an index into children[] broke
  // the moment a row was added.
  const attnPanel = $('#panel-attention');
  const wantAttention = !!attnPanel && !attnPanel.hidden;
  worker.postMessage({
    type: 'start',
    prompt: controls.get('prompt'),
    steps: Math.round(controls.get('steps')),
    guidance: parseFloat(controls.get('guidance')),
    seed: Math.round(controls.get('seed')),
    // The presence of a picture is the whole switch. No image, no strength:
    // `1` makes the sampler take the full schedule, which IS text-to-image.
    strength: state.initial ? parseFloat(controls.get('strength')) : 1,
    initial: state.initial,
    collectInternals: wantAttention,
  });
}

// ------------------------------------------------- starting from a picture
/**
 * The image slot in the dock.
 *
 * This is the ONLY thing that decides between generating from words and
 * generating from a picture. There is no mode to pick and nothing to undo: add
 * an image and the next run starts from it, clear it and the next run starts
 * from noise, and the prompt you built survives both.
 *
 * The whole window is a drop target, not just the 52px button. Requiring a
 * small square to be hit exactly is a needless accuracy test, and a file
 * dropped anywhere else would otherwise be opened BY THE BROWSER, navigating
 * away from the app and losing the run.
 */
function wireDropzone() {
  const dz = $('#dropzone');
  const input = $('#fileInput');
  $('#openBtn').onclick = () => input.click();
  $('#clearImg').onclick = (e) => { e.stopPropagation(); clearImage(); };
  input.onchange = () => input.files[0] && loadImage(input.files[0]);

  for (const ev of ['dragenter', 'dragover']) {
    document.addEventListener(ev, (e) => {
      if (state.mode !== 'make') return;
      e.preventDefault();
      dz.classList.add('over');
    });
  }
  for (const ev of ['dragleave', 'drop']) {
    document.addEventListener(ev, (e) => {
      e.preventDefault();
      dz.classList.remove('over');
    });
  }
  document.addEventListener('drop', (e) => {
    if (state.mode !== 'make') return;
    const f = e.dataTransfer && e.dataTransfer.files[0];
    if (f) loadImage(f);
  });
}

function clearImage() {
  if (state.initial) log.info('picture removed — back to generating from words');
  state.initial = null;
  $('#preview').hidden = true;
  $('#dropzone').classList.remove('has');
  $('#clearImg').hidden = true;
  $('#dropHint').textContent = '+ Add image';
  $('#fileInput').value = '';
  controls.setMode(state.mode, { image: false });
  autoRun();
}

/**
 * Any image -> the 16 x 16 tile the model actually sees.
 *
 * TWO THINGS HERE ARE NOT OPTIONAL, and both were wrong in the first version.
 *
 * 1. FLATTEN ONTO THE MODEL'S BACKGROUND. A canvas starts transparent black, so
 *    reading RGB and discarding alpha turns every transparent pixel into pure
 *    black. The test image — a creature on "white" — is 46% transparent, and
 *    the model was handed a creature on a black field: a picture nowhere near
 *    anything it was trained on, which it duly ignored. The background colour
 *    comes from the manifest because it is a property of the trained model.
 *
 * 2. HALVE REPEATEDLY. Browsers sample rather than average when a single
 *    drawImage shrinks by a large factor, so 475px straight to 16px throws away
 *    almost all of the picture and keeps whatever happened to land under the
 *    sample points. Stepping down by halves averages every source pixel in.
 *
 * The first canvas is capped so that a phone photograph does not allocate a
 * 48-megapixel buffer to be thrown away immediately.
 */
const MAX_WORK = 1024;

function toTile(img, bg) {
  // Cover-crop to square first, so a wide photo is not squashed.
  const s = Math.min(img.width, img.height);
  const sx = (img.width - s) / 2, sy = (img.height - s) / 2;

  let size = Math.min(s, MAX_WORK);
  let cur = document.createElement('canvas');
  cur.width = cur.height = size;
  let g = cur.getContext('2d');
  g.imageSmoothingEnabled = true;
  g.imageSmoothingQuality = 'high';
  g.fillStyle = `rgb(${bg[0]},${bg[1]},${bg[2]})`;
  g.fillRect(0, 0, size, size);
  g.drawImage(img, sx, sy, s, s, 0, 0, size, size);

  while (size > 32) {
    const half = Math.max(16, Math.round(size / 2));
    const next = document.createElement('canvas');
    next.width = next.height = half;
    const ng = next.getContext('2d');
    ng.imageSmoothingEnabled = true;
    ng.imageSmoothingQuality = 'high';
    ng.drawImage(cur, 0, 0, half, half);
    cur = next;
    size = half;
  }

  const tile = document.createElement('canvas');
  tile.width = tile.height = 16;
  const tg = tile.getContext('2d');
  tg.imageSmoothingEnabled = true;
  tg.imageSmoothingQuality = 'high';
  tg.drawImage(cur, 0, 0, 16, 16);
  return tile;
}

function loadImage(file) {
  const url = URL.createObjectURL(file);
  const img = new Image();
  img.onload = () => {
    const bg = state.manifest?.background || [248, 249, 252];
    const c = toTile(img, bg);
    const d = c.getContext('2d').getImageData(0, 0, 16, 16).data;
    const out = new Float32Array(3 * 256);
    for (let i = 0; i < 256; i++) {
      for (let ch = 0; ch < 3; ch++) out[ch * 256 + i] = d[i * 4 + ch] / 127.5 - 1;
    }
    state.initial = out;
    log.good(`picture loaded — ${file.name}`,
      `${img.width}x${img.height} → 16x16`);
    const prev = $('#preview').getContext('2d');
    prev.imageSmoothingEnabled = false;
    prev.clearRect(0, 0, 16, 16);
    prev.drawImage(c, 0, 0, 16, 16);
    $('#preview').hidden = false;
    $('#dropzone').classList.add('has');
    $('#clearImg').hidden = false;
    // The thumbnail sits BESIDE the text now rather than covering it, so the
    // slot can say whose picture it is holding. Long names are ellipsised in
    // CSS; the full one stays on the title.
    $('#dropHint').textContent = file.name;
    $('#dropzone').title = `${file.name} → 16 × 16`;
    URL.revokeObjectURL(url);
    // "Change amount" only means something once there is something to change.
    controls.setMode(state.mode, { image: true });
    autoRun();
  };
  img.onerror = () => { announce('That file could not be read as an image.'); URL.revokeObjectURL(url); };
  img.src = url;
}

// --------------------------------------------------------------- loop
let last = performance.now(), acc = 0;

/**
 * One frame. Split out from the rAF driver so it can be called directly.
 *
 * That is not only for tests: the automation browser used to verify this app
 * never fires requestAnimationFrame at all (measured: 0 frames per second,
 * canvases left at the default 300x150). A render loop that cannot be driven
 * by hand is a render loop that cannot be checked there.
 */
function render(now) {
  const dt = Math.min(0.1, (now - last) / 1000);
  last = now;

  if (state.mode === 'train') return;   // static; the scrubber drives it

  if (state.playing && state.snaps.length) {
    acc += dt * 4 * parseFloat(controls.get('speed'));
    while (acc >= 1) {
      acc -= 1;
      if (state.head < state.snaps.length - 1) state.head++;
      else state.following = true;
    }
    $('#scrub').value = String(state.head);
  }

  const snap = state.snaps[Math.min(state.head, state.snaps.length - 1)];
  for (const [id, p] of panes) {
    if (p.pane.hidden || !p.renderer) continue;
    p.renderer.draw(p.canvas, p.cam, state.snaps, state.head);
  }
  if (snap) {
    drawResult(snap);
    strip.update(snap);
    if (!$('#panel-attention').hidden) attention.draw(snap);
    if (!$('#panel-schedule').hidden) schedule.draw(snap);
    const pct = Math.round(((snap.index + 1) / snap.total) * 100);
    $('#pct').textContent = `${pct}%`;
    $('#bar').style.width = `${pct}%`;
    const done = snap.index + 1 >= snap.total;
    // `stale` outranks the rest: what is on screen is a real finished run, but
    // it is no longer the run the controls describe, and "Finished" would be
    // answering a question nobody asked. Checked here rather than only in
    // markStale() because this recomputes every frame and would clobber it.
    // A recording says so. Everything else in this app is computed in front
    // of you, and letting a saved run wear the word "Finished" would claim
    // work that did not just happen.
    $('#phase').textContent = snap.demo ? 'Example'
      : state.stale ? 'Changed'
      : state.playing ? 'Denoising' : done ? 'Finished' : 'Paused';
    // (`stale` is only ever set with snapshots present here — see markStale.)
    // The pill goes quiet when there is nothing in progress — see .hud.done.
    $('.hud').classList.toggle('done', done && !state.playing && !state.stale);
    $('.hud').classList.toggle('stale', state.stale);
    $('.hud').classList.toggle('example', !!snap.demo);
    $('#tNow').textContent = `t = ${snap.t}`;
    $('#stepOf').textContent = `step ${snap.index + 1} / ${snap.total}`;
  }
}

function frame(now) {
  render(now);
  requestAnimationFrame(frame);
}

// --------------------------------------------------------------- status
function setStatus(text, p) {
  const o = $('#overlay');
  o.hidden = false;
  $('#overlayText').textContent = text;
  $('#overlayBar').style.width = `${Math.round(p * 100)}%`;
}
function hideStatus() { $('#overlay').hidden = true; }
function revealChrome() {
  for (const el of document.querySelectorAll('[data-reveal]')) el.hidden = false;
}
function fail(msg) {
  const o = $('#overlay');
  o.hidden = false;
  o.classList.add('failed');
  $('#overlayText').textContent = msg;
  $('#overlayBar').style.width = '100%';
}
function announce(msg) {
  const el = $('#announce');
  el.textContent = msg;
  clearTimeout(announce._t);
  announce._t = setTimeout(() => { el.textContent = ''; }, 4000);
}
/**
 * Say plainly what the model actually understood.
 *
 * The first version wrote a small orange line reading "ignored: cat". That is
 * accurate and it was still missed: the user typed two different prompts, got
 * the same picture both times, and reasonably concluded the app ignored the
 * prompt entirely. When NOTHING is understood the result really is the same
 * image every time, so that case needs saying loudly, not noting quietly.
 */
function reportPrompt(m) {
  const el = $('#promptNote');
  el.className = 'note';
  const unknown = m.unknown || [];
  const defaulted = m.defaulted || [];

  // With a slot caption, EVERY prompt produces a full monster — unmentioned
  // slots get filled in. So "nothing was understood" no longer shows up as a
  // missing picture; it shows up as the same monster for every prompt, which
  // is exactly the confusion this note exists to prevent. The test is whether
  // any word landed at all, not whether the caption came out complete — it
  // always comes out complete.
  const bits = [];

  if (!m.known || !m.known.length) {
    el.classList.add('bad');
    el.innerHTML = `<b>None of those words are in the model's vocabulary.</b> `
      + `Everything was filled in for you, so you will get the same monster `
      + `whatever you type. Pick from the list below.`;
    el.hidden = false;
    return;
  }

  if (unknown.length) {
    const list = unknown.map((w) => `“${w}”`).join(', ');
    bits.push(`${list} ignored — not ${unknown.length > 1 ? 'words' : 'a word'} this model knows.`);
  }
  if (defaulted.length) {
    const filled = defaulted.map((s) => `<b>${m.slots[s]}</b> for ${s}`).join(', ');
    bits.push(`Every monster needs a colour, a body and eyes, so it chose ${filled}.`);
  }
  if (!bits.length) { el.hidden = true; return; }
  el.innerHTML = bits.join(' ');
  el.hidden = false;
}

// Debug handle. `render` is exposed so the app can be driven and inspected in
// environments where requestAnimationFrame does not fire; `panes` so a test
// can reach the cameras.
window.__app = state;
window.__debug = {
  render, panes,
  get controls() { return controls; },
  get replay() { return replay; },
};

boot().catch((e) => fail(e.message || String(e)));
