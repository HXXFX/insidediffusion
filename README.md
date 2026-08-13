# Inside Diffusion

**Watch a diffusion model think, one step at a time.**

## https://insidediffusion.windnoise.org

A diffusion model turns random static into a picture by removing a little noise
at a time. Every explanation of that process is a diagram of something that
already happened. This one is not a diagram: a real model runs in your browser
and four views of its actual internal state update in step with it.

Nothing is uploaded. There is no server. The model is 2.4 MB and runs on your
own machine.

---

## What it does

Build a prompt from dropdowns — one editable sentence along the bottom of the
window — and watch a monster resolve out of noise:

> a **green** **slime** with **three** eyes, **horns**, **bat wings** and **two legs**

Every part is a dropdown, so a prompt this model cannot understand is impossible
to write. Drop a picture into the slot beside the sentence and it starts from
that instead; take it out and it goes back to generating from the words.

Four views, any of which can be on screen at once:

| view | what it shows |
|---|---|
| **Funnel** | your picture as a single dot, falling from pure noise onto the set of real images |
| **Landscape** | a map of which pictures are likely — hills are likely, flats are not |
| **Towers** | one tower per pixel, so you can watch noise drain out unevenly |
| **Network** | the model itself, with the real numbers flowing through it |

And three panels underneath:

- **Step breakdown** — one denoising step taken apart into its four pieces
- **Word attention** — which pixels are listening to which word of your prompt
- **Noise recipe** — how much noise is added at each point in the run

A picture you drop in is reduced to 16 × 16, pushed part-way back toward noise,
and rebuilt — turn the change amount down and you get your picture lightly
repainted, turn it up and you get a monster that borrowed its shape and colour
from it.

## How it was made

The second tab is in three parts.

**One training step**, run live on the model already loaded in your browser. It
takes a real training picture, hides it under a known amount of noise, and asks
the model what noise was added — then scores how close it got. Change the
picture, the colour, or how much noise, and watch the answer change. That is the
whole training loop; the only thing it cannot do is update the weights.

**Thirty thousand of them.** Every monster here was drawn by the real model at
that point in its own 30,000-step run, from the same ten prompts each time — step
50 is genuinely what it could do after fifty gradient steps. Jump between phases
and watch colour arrive first, then bodies, then the details.

**Why the curve lies.** The loss, with how much the pictures actually changed
drawn over it. They agree closely, and that is the trap: both fall 84% of the way
by step 200, while the output is still coloured smears. A curve that looks
finished by step 1,000 still has a factor of seven to go, and that last 7× is the
difference between a blob and a monster.

---

## Why it is so small

The whole design follows from one decision: **at 16 × 16 the model's entire
state is 768 numbers**, which is few enough to draw honestly. Every view shows
real values. Nothing is a stand-in.

That constraint sets everything else:

- **The model is trained from scratch**, not adapted from Stable Diffusion. A
  packaged model is a sealed box — the internals this app exists to reveal would
  have to be faked. This one is 1.2 million numbers and every layer is readable.
- **The vocabulary is closed**: 37 words in six slots. Small enough to show you
  all of it, which means a prompt it cannot understand is impossible to write.
- **The training data is procedural** — the monsters are generated from a set of
  hand-drawn parts, so there is no scraped dataset behind any of it.

Because the data is generated, the *exact* right answer for every prompt is
known. That turns "does it look good" into a number, and every claim the app
makes about the model has been measured rather than assumed.

---

## Two models

| | Fast | Detailed |
|---|---|---|
| size | 2.4 MB | 4.8 MB |
| numbers | 1.2 M | 2.4 M |
| a 30-step run | ~3.6 s | ~7.4 s |
| accuracy | 0.0027 | 0.0018 |

Fast is the default and works anywhere. Detailed is about a third more accurate
and wants a desktop. Switch under **More settings** along the bottom; nothing
else changes, and the Making of tab follows to that model's own run.

---

## Requirements

A modern browser. That is the whole list — no install, no sign-in, no GPU.

The model runs on your CPU in a background thread, so the views stay smooth
while it works.

---

## A note on what you will see

The app tries hard not to overstate what it is showing you.

The attention panel is the clearest example. The tidy story is that each word
lights up its own part of the picture. What actually happens, measured on the
running model, is that the **colour** word does that strongly and steadily, the
**horns** word does it only in the last few steps, and **wings** and **legs**
merely lean that way. Real attention is a tendency, not a spotlight — and that
is worth knowing before you meet your first cherry-picked figure in a paper.

Where a view is a simplification, it says so on the view.

---

## Licence

Proprietary. All rights reserved. See [LICENSE](LICENSE).

**Your images and your data stay yours.** Images you open are never uploaded,
nothing you type is transmitted, and pictures you generate are yours to use.
There is no account, no tracking, and no analytics.

---

<sub>© 2026 WINDNOISE</sub>
