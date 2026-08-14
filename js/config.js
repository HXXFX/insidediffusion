/**
 * THE SINGLE DECLARATIVE SOURCE OF TRUTH.
 *
 * Every control, view, sampler and default is declared here once. The UI is
 * generated from `CONTROLS`, the view switcher from `VIEWS`, validation from
 * the same ranges. Adding a slider should mean editing this file plus the code
 * that reads its value — and nothing else. If you find yourself editing markup
 * to add an option, something has drifted and should come back here.
 *
 * WHERE TO CHANGE THINGS
 *   look and feel ............. css/app.css
 *   what the controls are ..... this file
 *   what a view draws ......... js/views/<name>.js
 *   the model itself .......... js/nn/unet.js  (and train/model.py — both)
 *
 * The vocabulary is NOT here: it is read from weights/manifest.json, because
 * it is a property of the trained model and hardcoding it in two places is how
 * they diverge.
 */

export const APP = {
  name: 'Inside Diffusion',
  nameParts: ['Inside', 'Diffusion'],   // first word is the distinguishing one
  tagline: 'Watch a diffusion model think, one step at a time.',
  imprint: 'WINDNOISE',
  author: 'HXXFX',
  year: 2026,
  // The handle in the credit plate points at THIS PROJECT'S repository, not at
  // the GitHub profile: from inside the app the useful destination is the
  // source of the thing you are looking at, and the repo header carries the
  // handle anyway, so the profile is one hop away rather than lost.
  repo: 'https://github.com/HXXFX/insidediffusion',
  imprintUrl: 'https://windnoise.org',
  licence: 'All rights reserved',
};

/**
 * Modes are top-level tabs. Panel order follows the user's workflow, which is
 * deliberately not the order the code executes in.
 */
/**
 * TWO MODES, NOT THREE. Text-to-image and image-to-image were the same code
 * path with one argument different, split across two tabs — which forced a
 * choice of MODE before the user knew what either did, and made "actually, from
 * this picture" cost them the prompt they had just built. The image slot lives
 * in the dock now: empty and it generates from the words, filled and it starts
 * from your picture. Same run button either way.
 */
/**
 * "Making of", after three tries.
 *
 *   "Training"        read as an offer — it promises you can train something
 *                     here, and you cannot.
 *   "How it learned"  fixed the tense but not the clarity: a sentence
 *                     fragment on a tab does not say what you will find.
 *   "Making of"       is a form everybody already knows from film and games.
 *                     It says: a recording, about this thing, made by someone
 *                     else, watchable.
 *
 * Rejected: "Monster Lab" (that is where you MAKE monsters — it describes the
 * other tab), "Model BTS" (an abbreviation to learn before you can use it).
 */
export const MODES = [
  { id: 'make', label: 'Make', desc: 'Build a prompt and watch a monster resolve out of noise. Add a picture to start from that instead.' },
  { id: 'train', label: 'Making of',
    desc: 'A recording of this model learning to draw — 30,000 steps, replayed.' },
];

/**
 * Sliders and selects. `mode` limits a control to certain tabs.
 *
 * NAMING RULE, learned the hard way: a control is named for WHAT IT DOES, not
 * what it costs. The cost belongs in `tip`.
 *
 * `tip` is `{ does, cost, when }` — three separate answers, rendered as three
 * labelled rows. It used to be one string with blank lines between the same
 * three parts, which came out as a wall of prose you had to read in full to
 * find the sentence you wanted; the fields make each answer findable and stop
 * the structure depending on whoever writes the next one remembering it.
 *
 * `does` says what changes, in terms of what appears on screen. `cost` is time
 * or bytes, concretely. `when` names a value to try and what you will see.
 * Numbers in these come from the sweeps recorded beside each control.
 */
export const CONTROLS = [
  {
    /**
     * Options are filled at load time from weights/models.json, not written
     * here — which models exist is a fact about what has been trained and
     * exported, and a second copy in this file would go stale the first time
     * one is added.
     */
    id: 'model', group: 'model', kind: 'select', default: 'fast',
    label: 'Model size', options: [],
    tip: {
      does: 'Which of the two trained models draws your monster. Detailed has twice as many numbers in it, so it gets edges and small parts closer to the pictures it learned from.',
      cost: 'Fast is 2.4 MB and about 110 ms a step. Detailed is 4.8 MB and about 220 ms — twice the wait, and twice the download.',
      when: 'Stay on Fast; it runs on anything. Move to Detailed on a desktop if shapes look rough — measured, it is about a third more accurate.',
    },
  },
  {
    /**
     * `sentence` renders as an editable sentence with one dropdown per caption
     * slot, mounted by main.js once the vocabulary has loaded — NOT as a text
     * box. A text box invites prompts the model cannot answer, and the failure
     * is silent: an unknown word leaves its slot at the default and returns a
     * clean monster that ignores what was asked. See js/ui/builder.js.
     */
    id: 'prompt', group: 'prompt', kind: 'sentence', mode: ['make'],
    label: 'Prompt', default: 'a green slime with three eyes, horns, bat wings and two legs',
    tip: {
      does: 'The monster to draw. Six dropdowns — colour, body, eyes, horns, wings, legs — and 37 words between them.',
      cost: 'Free. Changing any word starts a new run.',
      when: 'They are dropdowns rather than a text box so you cannot ask for a word the model has never seen. Change the horns and watch the top rows of the attention panel change with them.',
    },
  },
  {
    id: 'steps', group: 'sampling', kind: 'range', min: 4, max: 60, step: 1, default: 30,
    label: 'Steps',
    tip: {
      does: 'How many times the model looks at the picture and takes a little of the noise out. More steps means smaller, more careful corrections.',
      cost: 'One full pass of the network each: 60 steps takes twice as long as 30.',
      when: 'Raise it if the monster looks rough or muddy. Below about 8 the picture visibly falls apart; above about 40 there is almost nothing left to gain.',
    },
  },
  {
    /**
     * DEFAULT 1.0 — measured twice, and the reasoning behind the old 1.5 was
     * wrong on both the number and the mechanism.
     *
     * Swept with train/sweep_guidance.py against ground truth (the dataset is
     * procedural, so the exact right answer for every prompt is known — this is
     * real error, not the proxy statistic the first sweep had to use):
     *
     *     guidance  1.00 -> 0.0018    every monster crisp and correct
     *     guidance  1.25 -> 0.0039
     *     guidance  1.50 -> 0.0194    colour drifting off the palette
     *     guidance  2.00 -> 0.1310
     *     guidance  3.00 -> 0.2079    heavy colour noise
     *     guidance  6.00 -> 0.3357    noise
     *
     * It was believed that guidance failed because the training data was
     * deterministic — one drawing per prompt, so no distribution to sharpen
     * toward. Each word now covers two or three drawings, and the optimum did
     * not move. The real reason is that this model's prompt adherence is
     * already essentially perfect (0.0020 is under one level in 255 per
     * channel), so guidance has nothing left to buy and only adds saturation.
     *
     * So it ships at 1.0, where guidance is OFF, and the tip says so plainly.
     * The range still reaches 8 ON PURPOSE: watching it fall apart is one of
     * the most instructive things in the app, and with the slider at 1.0 the
     * user gets to cause that rather than land on it.
     */
    id: 'guidance', group: 'sampling', kind: 'range', min: 0, max: 8, step: 0.1, default: 1.0,
    label: 'Guidance',
    warnAbove: 1.5,
    tip: {
      does: 'How hard the model is pushed toward your words and away from a generic monster. At 1 it is switched off and the model simply follows the prompt.',
      cost: 'Double, above or below 1: any value except exactly 1 runs the network twice per step.',
      when: 'Leave it at 1 — measured, this model is at its most accurate there, because it already follows the prompt almost exactly and guidance has nothing left to buy. Drag it to 3 once anyway: the colours leave the palette and the monster comes apart. That is what over-guidance looks like, and it is the clearest thing in the app.',
    },
  },
  {
    id: 'seed', group: 'sampling', kind: 'int', min: 0, max: 999999, default: 1337,
    label: 'Seed',
    tip: {
      does: 'Which square of random noise the run starts from. Every generation begins as static; this number chooses which static.',
      cost: 'Free.',
      when: 'The same seed and the same words give the same monster, exactly — every time, on any machine. Change only the seed to see how much room your prompt leaves: you get a different monster that still matches every word you picked.',
    },
  },
  {
    id: 'sampler', group: 'sampling', kind: 'select', default: 'ddim',
    label: 'Sampler',
    options: [
      { value: 'ddim', label: 'DDIM — repeatable' },
      { value: 'ddpm', label: 'DDPM — adds noise each step' },
    ],
    tip: {
      does: 'The rule for getting from one step to the next, once the model has said where the noise is.',
      cost: 'The same either way.',
      when: 'DDIM repeats exactly — same seed, same picture. DDPM stirs in fresh noise at every step, so two runs of the same prompt differ slightly. Switch to DDPM and watch the trail in the Funnel view wander instead of falling straight.',
    },
  },
  {
    /**
     * DEFAULT 0.45, NOT 0.7 — measured with train/img2img_demo.py, which runs a
     * real photograph through the whole range. Difference between the output
     * and the source image:
     *
     *     0.20 -> 0.1591    your picture, lightly repainted
     *     0.35 -> 0.2377    a monster, still clearly built from your picture
     *     0.50 -> 0.2767    a monster that borrowed its shape and colour
     *     0.65 -> 0.2831
     *     0.80 -> 0.2867
     *     1.00 -> 0.2902    your picture ignored entirely
     *
     * The curve flattens after ~0.5: at 0.65 the result is already within 2.5%
     * of what you get from pure noise, so the old 0.7 default sat in the region
     * where the feature appears not to work. Anything above about 0.6 is worth
     * offering but is a poor place to land.
     */
    id: 'strength', group: 'sampling', kind: 'range', min: 0.1, max: 1, step: 0.05,
    default: 0.45, mode: ['make'], needs: 'image',
    label: 'Change amount',
    warnAbove: 0.65,
    tip: {
      does: 'How far your picture is pushed back toward noise before the model rebuilds it. The higher it goes, the less of your picture survives.',
      cost: 'Lower is faster: it starts partway down the schedule, so it runs fewer steps.',
      when: '0.2 hands your picture back with a light repaint. 0.45 — the default — keeps its shape and colour but makes a monster of it. Past about 0.65 the result is within 2.5% of what pure noise gives, so your picture stops mattering at all.',
    },
  },
  // `shape` and `lr` belonged to the live 2-D trainer, which the Training tab
  // no longer runs — it replays the real model's own history instead, and there
  // is nothing there for a learning-rate slider to act on. The trainer itself
  // is kept in js/train2d/ and still works; it is simply not wired in.
  {
    id: 'speed', group: 'run', kind: 'range', min: 0.25, max: 8, step: 0.25, default: 1,
    label: 'Playback speed',
    tip: {
      does: 'How quickly finished steps are played back to you.',
      cost: 'None. The model has already done the work — this only changes the replay.',
      when: 'Slow it down to read the numbers as they change; speed it up once you know what you are looking at.',
    },
  },
];

/**
 * The four views. All of them ship; any subset can be on screen at once and
 * the layout follows the count. Each is a pure function of one step snapshot,
 * which is what keeps them in lockstep with each other and with the model.
 */
/**
 * The four views.
 *
 * `help` is not decoration — it is the reason the app exists. Every view is
 * abstract enough that a newcomer cannot decode it unaided, and an unexplained
 * 3-D plot teaches nothing at all. Each one answers three questions in a fixed
 * order: WHAT AM I LOOKING AT, WHAT IS EACH THING, WHAT SHOULD I NOTICE.
 *
 * Write these for someone who has never heard the word "latent". No jargon
 * without an immediate plain-English gloss; no symbol that is not in the key.
 */
export const VIEWS = [
  {
    id: 'funnel', label: 'Funnel', letter: 'B', default: true,
    short: 'Your picture falling out of noise.',
    help: {
      what: 'Your picture, shown as a single dot, falling from pure noise down to a finished image. One dot on the trail per step.',
      key: [
        ['accent', 'The clay trail', 'the route your picture has taken so far'],
        ['accent-2', 'The amber dot', 'where your picture is right now'],
        ['pic', 'The picture beside the dot', 'its guess at the finished monster, redrawn every step'],
        ['pic', 'Specks on the floor', '500 real training monsters, each painted its own colour — the three nearest to you are shown as pictures'],
        ['axis', 'Height', 'how much noise is left — top is pure static, the floor is clean'],
      ],
      watch: 'The trail starts high above the floor and finishes among the coloured specks — and the three pictures show what it landed nearest. That is the whole job of the model: steer random static until it lands among things it has seen before.',
    },
  },
  {
    id: 'landscape', label: 'Landscape', letter: 'A', default: true,
    short: 'Where believable pictures live.',
    help: {
      what: 'A map of where believable pictures live. The ground rises where pictures are likely and stays flat where they are not.',
      key: [
        ['surface', 'The hills', 'how likely a picture at that spot is'],
        ['pic', 'The picture above each hill', 'a real training monster from that peak — what kind of thing lives there'],
        ['accent-2', 'The amber dot', 'where your picture sits, with its current guess above it'],
        ['axis', 'σ (sigma)', 'how much noise is left, printed bottom right'],
      ],
      watch: 'At the start there is one wide, soft hill — almost anything is possible. As the noise drains, separate peaks appear, each with its own kind of monster on top, and your dot settles onto one. That is the picture becoming specific.',
      // Stated on the view itself as well. Sitting in a tile beside the funnel
      // this could otherwise be read as depicting your image, which it cannot.
      caveat: 'This is a flat shadow of a 768-dimensional space, so it shows roughly a quarter of the real picture. It is honest about the data, but it is not your image.',
    },
  },
  {
    /* ALL FOUR ARE ON BY DEFAULT. Two were, on the reasoning that a quarter
       pane is small and four abstract plots at once is a lot to meet. What that
       missed is that the four are not four takes on the same thing — they answer
       different questions, and the one people connect with first turned out to
       be Towers, which was off. Measured cost of the change: 55 fps to 43 with
       all four drawing, which is still smooth to orbit. */
    id: 'voxels', label: 'Towers', letter: 'C', default: true,
    short: 'One tower per pixel.',
    help: {
      what: 'One tower for each pixel of your 16 × 16 image, so 256 towers standing on a grid.',
      key: [
        ['axis', 'Tower height', 'how much noise the model still sees on that pixel'],
        ['surface', 'Tower colour', "the model's current guess for that pixel's colour"],
      ],
      watch: 'Towers over the background collapse first, while towers over the object stay tall. The model settles the easy, empty areas long before it commits to the details.',
    },
  },
  {
    id: 'network', label: 'Network', letter: 'D', default: true,
    short: 'The model your picture passes through.',
    help: {
      what: 'The neural network your picture passes through, once per step. It shrinks the image down, thinks, then builds it back up.',
      key: [
        ['box', 'Each box', 'one layer. The label is its size and how many channels it holds'],
        ['tile', 'The small grey tiles', 'the ACTUAL numbers inside that layer right now, not an illustration'],
        ['pic', 'Goes in, comes out', 'the picture handed to the network, and the one it hands back'],
        ['dash', 'Dashed lines', 'shortcuts that carry fine detail past the middle so it is not lost'],
        ['accent-2', 'The amber arrow and the spark', 'where your prompt enters; the spark is one trip through the network, once per step'],
      ],
      watch: 'The image goes 16 → 8 → 4 and back to 16. The narrow middle is where the model considers the whole picture at once; the shortcuts are what keep the edges sharp. When the spark stops running, the run has finished.',
    },
  },
];

/** Explanations for the 2-D panels, same three-question shape. */
export const PANEL_HELP = {
  strip: {
    what: 'The single most useful picture in the app: one denoising step, taken apart.',
    key: [
      ['tile', 'x t', 'the noisy image as it stands right now'],
      ['tile', 'e-hat', 'the noise the model believes is hiding in it'],
      ['tile', 'x0-hat', 'that noise subtracted — its guess at the finished picture'],
      ['tile', 'x t-1', 'a little noise added back, ready for the next step'],
    ],
    watch: 'The third tile becomes recognisable long before the first one does. That is the point: the model is not sharpening a blurry picture, it is predicting the noise and taking it away.',
  },
  /**
   * MEASURED, NOT ASSUMED — and the first version of this text overclaimed.
   *
   * It said the grid under a word "usually lights up where the object is",
   * which is the tidy story everyone expects from an attention map. Measuring
   * the real maps over a run (share of attention falling in each word's own
   * region, against the uniform baseline that region would get by chance):
   *
   *     colour -> body interior    0.60-0.71  vs 0.50   clear, and steady
   *     horns  -> top two rows     0.26 -> 0.47 vs 0.25  only at the very end
   *     legs   -> bottom two rows  peaks 0.31   vs 0.25  weak
   *     wings  -> outer columns    0.57         vs 0.50  barely there
   *
   * So one word localises strongly, one localises late, and two hardly do. The
   * honest version is more interesting than the tidy one anyway: it shows that
   * a real attention map is a tendency and not a spotlight, which is worth
   * knowing before someone reads a paper's cherry-picked figure.
   */
  attention: {
    what: 'Which parts of the image are paying attention to which word of your prompt. One grid per slot in the caption.',
    key: [
      ['grid', 'One grid per word', 'brighter squares are paying more attention to that word'],
      ['tok', 'bos / eos', 'housekeeping tokens, not real words — they mark the start and the end'],
    ],
    watch: 'Look at the colour word: its grid stays bright over the body and dim around the edges for the whole run, because the colour has to be painted everywhere the creature is. Then watch the horns word in the last few steps, where it suddenly concentrates on the top two rows — the model leaves that detail until late.',
    caveat: 'Real attention is a tendency, not a spotlight. Colour localises strongly and horns localise at the end, but wings and legs only lean toward their own regions — noticeably, but far less than a textbook diagram would suggest.',
  },
  schedule: {
    what: 'The recipe for how much noise is added at each point in the process.',
    key: [
      ['accent', 'Clay line', 'how much of the original picture survives'],
      ['accent-2', 'Amber line', 'how much noise has been mixed in'],
      ['ink', 'Dark marker', 'the step you are looking at'],
    ],
    watch: 'The run travels right to left, from all noise to clean. Most of the visible change happens in the last third, where the blue line climbs steeply.',
  },
};

/** Inspector panels, shown under or beside the viewport. Explanations live in
 *  PANEL_HELP above, keyed by the same id. */
export const PANELS = [
  { id: 'strip', label: 'Step breakdown', default: true,
    blurb: 'One denoising step taken apart. The clearest thing in the app.' },
  // The map is captured at the 8x8 cross-attention layer, not at 16x16 — the
  // first attention block sits after one downsample. Say 8x8, because the
  // grid on screen is 8x8 and claiming 16 would be visibly wrong to anyone
  // who counts the squares.
  { id: 'attention', label: 'Word attention', default: true,
    blurb: 'Which parts of the image are listening to which word.' },
  { id: 'schedule', label: 'Noise recipe', default: false,
    blurb: 'How much noise is added at each point in the run.' },
];

/**
 * Camera defaults, per view.
 *
 * SEPARATE FROM THE ALLOWED RANGE, on purpose. An earlier build folded the two
 * together by scaling one shared pitch differently in each view, which meant
 * the same drag produced three different angles and top-down was unreachable
 * in all of them — worst in the hero view, which topped out at 51°. A default
 * is where the camera STARTS. The clamp below is what the user may reach, and
 * it is the same for every view.
 */
export const CAMERA = {
  // Just under vertical: at exactly ±90° the horizontal axes project onto a
  // single line and the scene collapses to a stripe. 1.5533 rad ≈ 89°.
  pitchMin: -1.5533,
  pitchMax: 1.5533,
  defaults: {
    funnel: { yaw: 0.62, pitch: 0.42, dist: 3.5 },
    landscape: { yaw: 0.60, pitch: 0.46, dist: 3.1 },
    voxels: { yaw: 0.85, pitch: 0.44, dist: 3.0 },
    network: { yaw: 0, pitch: 0, dist: 1 },
  },
};

/**
 * Prompts offered as a starting point. The app opens on the first one, so it
 * lands on a finished-looking result rather than an empty state.
 *
 * Chosen to span the axes rather than to look nice: between them they use every
 * body, several eye sets, and each of horns, wings and legs both present and
 * absent. Someone clicking through all eight has seen what the model can do.
 */
export const SAMPLE_PROMPTS = [
  'a green slime with three eyes, horns, bat wings and two legs',
  'a purple blob with one eye and big wings',
  'a red spike with angry eyes, spikes and many legs',
  'a blue ghost with huge eyes',
  'a yellow egg with sleepy eyes, ears and feather wings',
  'a pink drop with four eyes, antenna, bug wings and long legs',
  'a white worm with cross eyes and horns',
  'a black cube with two eyes, spikes and big wings',
];

export const DEFAULTS = Object.fromEntries(CONTROLS.map((c) => [c.id, c.default]));
