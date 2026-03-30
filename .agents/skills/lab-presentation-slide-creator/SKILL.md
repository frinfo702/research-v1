---
name: lab-presentation-slide-creator
description: Use when the user wants to create, revise, or review slides for a research lab presentation from a paper, notes, or an existing deck, especially when the output should be a slide-by-slide outline or a Typst/touying deck. Do not use for generic paper summarization, broad presentation coaching without slide work, or non-slide documents. Produces a lab-talk slide plan, optional speaker notes, and optionally a ready-to-edit `.typ` deck.
---

# Lab Presentation Slide Creator

## Overview

Create or refine research lab slides with a clear story arc, one message per slide, and enough technical depth for the audience. Optimize for talk delivery, not literal translation of the paper.

## Use When

- The user wants a new lab-talk deck from a paper, notes, or an abstract.
- The user wants an existing slide deck reviewed or tightened.
- The user wants a Typst/touying slide deck, not just prose notes.
- The user wants slide-by-slide speaking notes tied to the deck.

## Do Not Use When

- The request is only to summarize a paper with no slide artifact.
- The user is asking for general speaking advice without creating or editing slides.
- The output is a report, memo, poster, or manuscript rather than a slide deck.

## Trigger Examples

Positive examples:

- "Turn this paper into an 8-slide lab presentation."
- "Review this seminar deck and make the storyline tighter."
- "Draft a Typst/touying deck for tomorrow's paper reading."

Negative examples:

- "Summarize this paper for me."
- "How should I present more confidently?"
- "Write a related-work section for this project."

## Inputs To Confirm

Confirm only what is needed to produce the artifact:

- Talk length or target slide count
- Audience background
- Source material: paper, notes, current deck, or figures
- Output shape: outline, slide review, or Typst deck
- Constraints: required sections, lab template, speaker name, venue

If details are missing, state brief assumptions and proceed.

## Workflow

1. Identify the artifact.
   - New deck: build a slide plan first, then draft slides.
   - Existing deck: review structure, density, and missing transitions before rewriting.
   - Outline only: stop at slide-by-slide goals and speaker cues.
2. Extract the talk spine.
   - Define the audience takeaway in one sentence.
   - Reduce the paper to problem, gap, idea, mechanism, evidence, and limits.
3. Map the spine to slides.
   - Each slide should answer one question or make one claim.
   - Prefer Problem -> Gap -> Idea -> How -> Evidence -> Limits -> Takeaway.
   - Use the time-based patterns in `references/presentation_guide.md`.
4. Write slide content.
   - Keep titles claim-oriented when possible.
   - Default to short bullets, one visual, and one explicit takeaway per slide.
   - Put details that are needed for speaking but not for reading into speaker notes.
5. Handle technical content carefully.
   - For equations: intuition first, then variables, assumptions, and why the form matters.
   - For figures: state the question, what to read, and the conclusion.
6. If the output is Typst, use `assets/lab_slide_template.typ` as the starting point and consult `references/typst_slides.md` for syntax and structure.
7. If reviewing an existing deck, return concrete slide-level fixes, not vague presentation advice.

## Output Contract

Default response structure:

- `artifact`: outline, deck review, or `.typ` draft
- `goal`: audience takeaway and assumptions
- `slide_plan`: ordered slides with title, purpose, and key content
- `speaker_notes`: only when helpful or requested
- `open_items`: missing figures, unresolved assumptions, or `none`

When producing a Typst deck, emit a ready-to-edit `.typ` file body and list any external figures it expects.

## References

- Use `references/presentation_guide.md` for story templates, timing patterns, and review checklists.
- Use `references/typst_slides.md` for Typst/touying conventions and compilation guidance.
- Use `assets/lab_slide_template.typ` when creating a new Typst slide deck.
