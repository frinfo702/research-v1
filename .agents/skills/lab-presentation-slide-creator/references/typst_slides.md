# Typst Slides Reference

## When to Use

- Use Typst when asked to create slide decks from an outline or notes.
- Output a `.typ` file using the `touying` package.
- Start from `assets/lab_slide_template.typ` unless the user already has a house style.

## Minimal Typst Slide Deck

```typst
#import "@preview/touying:0.6.1": *
#import themes.simple: *

#show: simple-theme.with(aspect-ratio: "16-9")

#title-slide[
  = Talk Title
  #v(1em)
  Presenter Name \
  #datetime.today().display("[year]-[month]-[day]")
]
```

## Slide Structure

- Each slide is a `== Heading` section or a `#slide[...]` block
- Keep one message per slide
- Use short bullets and a single figure per slide when possible
- Use `= Section Title` for section divider slides
- Put speaker-only detail in `#speaker-note[...]`

## Example Skeleton

```typst
#import "@preview/touying:0.6.1": *
#import themes.simple: *

#show: simple-theme.with(aspect-ratio: "16-9")

#title-slide[
  = Paper Title
  #v(1em)
  Presenter \
  Lab Seminar, #datetime.today().display("[year]-[month]-[day]")
]

== Takeaway

#align(center + horizon)[
  _"One sentence that captures the core contribution."_
]

= Problem

== Problem

- Why this problem matters
- What fails in prior work

== Key Idea

- One sentence idea
- Diagram or equation intuition

#figure(image("fig/key_idea.png", width: 60%))

== Method

- Short algorithm steps

$ L = sum_i ell(f(x_i), y_i) + lambda R(theta) $

== Experiments

- Dataset, metric, baseline
- Main result: *+X% on benchmark Y*

#figure(image("fig/results.png", width: 70%))

== Limitations

- Weakness 1
- Weakness 2

== Takeaway

#align(center + horizon)[
  _"Restate the one-line takeaway."_

  Questions?
]
```

## Speaker Notes

Add speaker notes inside slides using `#speaker-note`:

```typst
== Prior Work Gap

- Method A fails when ...

#speaker-note[
  Explain why prior methods fail on long sequences.
]
```

Notes appear in presenter mode but not in the PDF output.
Alternatively, maintain a separate `notes.md` with per-slide scripts.

## Practical Defaults

- Prefer `16-9` aspect ratio unless the lab already uses `4-3`
- Keep captions interpretive: say what the figure proves
- Use figure placeholders like `fig/results.png` and list missing assets explicitly
- If a slide needs more than 4 bullets, split it

## Figures

- Place images in `fig/` and reference with `image("fig/filename.png", width: X%)`
- Always add a caption explaining the takeaway:
  ```typst
  #figure(
    image("fig/results.png", width: 70%),
    caption: [Main result: our method outperforms baselines on Y.]
  )
  ```

## Equations

- Inline: `$f(x)$`
- Block (centered): `$ L = ... $`
- Introduce intuition in bullet text before showing the equation

## Compilation

```bash
typst compile slides.typ          # → slides.pdf
typst watch slides.typ            # live preview
```
