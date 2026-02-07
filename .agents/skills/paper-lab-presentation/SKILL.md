---
name: paper-lab-presentation
description: Summarize and present research papers for lab talks with clear explanation goals, story-driven structure, equation intuition, and non-translation paraphrasing. Use when preparing or reviewing a paper presentation, building slide outlines, speaking notes, Q&A prep, or creating Quarto slide decks (.qmd) for talks.
---

# Paper Lab Presentation

## Overview

Enable concise, story-driven lab presentations of papers; avoid literal translation; clarify the explanation goal and audience takeaways.

## Workflow

1. Clarify constraints and goal.
   - Ask for time limit, audience background, required depth, format (slides/notes), and desired output (outline, script, Q&A prep).
   - Define the explanation goal: who should understand what, and be able to do or argue after the talk.
2. Extract the paper's core.
   - Identify problem, gap, key idea, contributions, assumptions, and novelty vs prior work.
   - Write a 1-sentence takeaway and 3-bullet contributions.
3. Build the story arc.
   - Use Problem -> Gap -> Idea -> How -> Evidence -> Limits -> Takeaway.
   - Treat these labels as internal planning only. Do not copy them into audience-facing slide titles unless explicitly requested.
   - Keep every section tied to the explanation goal.
4. Handle equations with meaning.
   - Introduce intuition before math, define variables, state assumptions, then show the equation.
   - Explain why the form arises (derivation or design choice) and what it enables.
5. Use figures as evidence.
   - Use figures extracted from Papers.
   - Explain what question each figure answers, what to read, and the takeaway.
6. Summarize experiments.
   - Provide task/dataset/metric, baseline, main result, and what it implies.
7. Produce output.
   - Provide slide outline (with timing) and speaking notes.
   - Include likely questions and short answers.
   - If requested, produce a Quarto .qmd slide deck (see references).
   - Use audience-facing slide titles. Avoid meta or instruction-like titles such as "How (1)", "Evidence (2)", "1スライドでまとめ", or "研究室向けの見どころ" unless the user explicitly asks for them.

## Output Formats

- Slide outline with per-slide goal, key message, and time budget.
- Speaking notes or narrative summary.
- Q&A prep list.
- Quarto slide deck (.qmd) when asked.

## References

- Use `references/presentation_guide.md` for templates, checklists, and phrasing patterns.
- Use `references/quarto_slides.md` for Quarto slide deck conventions and minimal structure.
