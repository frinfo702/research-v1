# Quarto Slides Reference

## When to Use

- Use Quarto when asked to create slide decks from an outline or notes.
- Prefer Quarto .qmd when the user already uses Quarto or wants reproducible slides.

## Minimal Quarto Slide Deck (Revealjs)

```yaml
---
title: "Talk Title"
author: "Your Name"
format:
  revealjs:
    slide-number: true
    theme: simple
---
```

## Slide Structure

- Every slide title must use `#` (h1). `##` or deeper headings are NOT recognized as slide titles in Quarto Revealjs.
- `---` can separate slides without titles, but titled slides always use `#`.
- Keep one message per slide.
- Use short bullets and a single figure per slide when possible.

## Example Skeleton

```markdown
---
title: "Paper Title"
author: "Presenter"
format:
  revealjs:
    slide-number: true
    theme: simple
---

# One-line takeaway

---

# Problem

- Why this problem matters
- What fails in prior work

---

# Key Idea

- One sentence idea
- Diagram or equation intuition

---

# Method

- Short algorithm steps

---

# Experiments

- Dataset, metric, baseline
- Main result

---

# Limitations

- 1-2 weaknesses

---

# Takeaway

- Restate the one-line takeaway
```

## Notes and Speaker Script

- Add speaker notes under `::: notes` blocks
- Keep notes concise and aligned with the slide goal

## Figures

- Reference local images in `fig/` or `assets/` and keep filenames short.
- Always explain what the figure shows and the takeaway.
- When a slide contains one image, Quarto auto-applies the "content with caption" layout. In this layout, the image **must come last** in the slide. Text before image works; image before text breaks rendering.

```markdown
# Good Example

- This figure shows the model architecture.

![](figures/architecture.png)
```

```markdown
# Bad Example (rendering breaks)

![](figures/architecture.png)

- This figure shows the model architecture.
```
