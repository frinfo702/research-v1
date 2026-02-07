---
name: paper-lab-presentation
description: >
  Transform rough paper notes into structured lab presentation materials.
  Use when: preparing a paper talk for lab seminar, building slide outlines
  from reading notes, converting raw paper summaries into presentation scripts,
  creating Quarto slide decks (.qmd/.pptx), or drafting Q&A prep.
  Converts "literal-translation" style notes into a story-driven presentation
  that covers novelty, scope, conditions, and reproducibility.
---

# Paper Lab Presentation

Transform rough reading notes into a structured, story-driven lab presentation.
Avoid Japanese-translation-style bullet lists. Instead, produce a narrative that
covers problem awareness, core ideas, and critical examination.

## Required Output Sections

Every presentation must address these four questions explicitly:

1. What is new? — State the delta from prior work in one sentence.
2. What can it do and what can it not? — Scope, assumptions, and constraints.
3. Under what conditions does it work? — Data/compute/setting dependencies, ablation highlights.
4. What to know for reproduction? — Key equations, hyperparameters, and pitfalls.

## Workflow

1. Gather constraints.
   - Ask for time limit, audience level, required depth, format (slides/notes), and output type (outline, script, Q&A).
   - Define the explanation goal: "After this talk, the audience should be able to ___."

2. Extract the paper core.
   - Identify problem, gap, key idea, contributions, assumptions, and novelty versus prior work.
   - Write a one-sentence takeaway and three-bullet contributions.

3. Build a story arc.
   - Use internally: Problem, Gap, Idea, How, Evidence, Limits, Takeaway.
   - These labels are for planning only. Slide titles must be audience-facing and descriptive, not meta labels like "How (1)" or "1スライドでまとめ".

4. Handle equations.
   - Give intuition before math. Define variables and assumptions. Show the equation. Explain why this form arises and what it enables.

5. Use figures as evidence.
   - For each figure: what question it answers, how to read it, and the one-sentence takeaway.

6. Summarize experiments.
   - State task, dataset, metric, baseline, main result, and implication.

7. Produce output.
   - Slide outline with per-slide goal, key message, and time budget.
   - Speaking notes or narrative summary.
   - Q&A prep: 3 likely weaknesses, 2 strongest evidence points, 1 extension idea.
   - If requested, produce a Quarto .qmd slide deck (see references/quarto_slides.md).

## Writing Style Rules (Critical)

All output must be in Japanese. Follow these rules strictly:

- Write in plain, natural Japanese sentences. Do not compress vocabulary or omit particles and adverbs for brevity. Write as a human would speak.
- Do not over-use markdown formatting. Avoid excessive bold, nested lists, or decorative headers. A flat, readable structure is preferred. Use bold only when truly necessary for emphasis.
- Prefer full sentences over noun-phrase bullet fragments. "この手法はXを仮定している" is better than "仮定: X".
- Do not produce "translation-style" output that mirrors the English paper sentence by sentence. Restructure into claim, evidence, implication.
- Keep technical terms in their original English where that is the convention in the field (e.g., "attention mechanism", "ablation study"), but write the surrounding explanation in natural Japanese.

## Output Formats

- Slide outline with per-slide goal, key message, and time budget.
- Speaking notes or narrative summary.
- Q&A prep list.
- Quarto slide deck (.qmd) when asked.

## References

- Read `references/presentation_guide.md` for templates, checklists, and phrasing patterns.
- Read `references/quarto_slides.md` for Quarto slide deck conventions and structure.
