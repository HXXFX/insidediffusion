/**
 * The saved run the app opens on.
 *
 * WHY IT EXISTS. The app reaches "model ready" in about 90 ms and then has
 * nothing to draw until a run finishes, which is another four seconds — and
 * with live update switched off it has nothing to draw at all until you press
 * Run. Either way the first thing a visitor saw was an empty viewport and five
 * blank tiles, which is the one state that makes a working app look broken.
 *
 * So a real run of this model, with the app's own defaults, is exported by
 * train/export_monsters.py::export_demo and shown immediately.
 *
 * IT IS A RECORDING, AND THE APP SAYS SO. Everything else on screen is computed
 * in front of you; presenting this as live would be exactly the overclaiming
 * the attention panel had to be corrected for. It is honest on its own terms —
 * the same weights, the same code path, the same defaults — and it is not
 * happening now. The moment a real run produces a snapshot it takes over.
 *
 * The snapshots it builds are the SAME SHAPE the worker posts, so every view,
 * panel and the scrubber consume them without knowing the difference. That is
 * the whole trick: no view needs a demo code path.
 */

/**
 * @param manifest  the model manifest, which carries the `demo` block
 * @param base      where the model's files live
 * @returns snapshot[] | null
 */
export async function loadDemo(manifest, base) {
  const d = manifest && manifest.demo;
  if (!d) return null;

  const buf = await (await fetch(`${base}demo.bin`)).arrayBuffer();
  const bytes = new Uint8Array(buf);
  const [steps, ch, h, w] = d.shape;
  const n = ch * h * w;
  const per = steps * n;
  if (bytes.length < per * d.order.length) {
    throw new Error(`demo.bin is ${bytes.length} bytes, expected ${per * d.order.length}`);
  }

  // One dequantisation range per KIND, matching the export. Per-step ranges
  // would make the 3-D trajectory jitter as the scale moved under it.
  const plane = {};
  d.order.forEach((k, ki) => {
    const [lo, hi] = d.ranges[k];
    const scale = (hi - lo) / 255;
    const off = ki * per;
    const out = new Float32Array(per);
    for (let i = 0; i < per; i++) out[i] = lo + bytes[off + i] * scale;
    plane[k] = out;
  });

  const snaps = [];
  for (let i = 0; i < steps; i++) {
    const cut = (k) => plane[k].subarray(i * n, (i + 1) * n);
    snaps.push({
      type: 'snapshot',
      index: i, total: steps,
      t: d.t[i], alphaBar: d.alphaBar[i],
      noiseLevel: Math.sqrt(1 - d.alphaBar[i]),
      ms: 0,
      x: cut('x'), eps: cut('eps'), x0: cut('x0'), next: cut('next'),
      // No captured activations: the Network and cross-attention panels say so
      // themselves when internals are missing, which is the right answer here —
      // this run genuinely did not record them.
      internals: null,
      // The one field a real snapshot does not have. Everything that needs to
      // tell the difference tests this rather than inferring it.
      demo: true,
    });
  }
  return snaps;
}
