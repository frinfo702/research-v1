---
name: paper-lab-presentation
description: >
  Transform rough paper notes into structured lab presentation materials.
  Use when: preparing a paper talk for lab seminar, building slide outlines
  from reading notes, converting raw paper summaries into presentation scripts,
  creating Quarto slide decks (.qmd/.pptx).
  Converts "literal-translation" style notes into a story-driven presentation
  that covers novelty, scope, conditions, and reproducibility.
---

# Paper Lab Presentation

Transform rough reading notes into a structured, story-driven lab presentation.
Avoid Japanese-translation-style bullet lists. Instead, produce a narrative that
covers problem awareness, core ideas, and critical examination.

## Metadata and Slide Title Rules (Mandatory)

- Presentation title must be exactly the paper title. Do not add private subtitles, taglines, or decorative prefixes/suffixes.
- Author line must use the paper's authors, not the presenter name. If authors are many, abbreviate in a standard form (e.g., "First Author et al.").
- Every slide title must start with one of these fixed categories only: `問題設定`, `モチベーション`, `先行研究`, `新規性`, `アイデア`, `提案手法`, `方法`, `実験設定`, `実験結果`, `制限`, `アブレーション`.
- Slide title format is conditional: if the same category appears on multiple slides, use `カテゴリ：短い言い切り (対応する論文の節番号)` (e.g., `結果：主要ベースラインを上回る (4.2)`); if the category appears on only one slide, use `カテゴリ (対応する論文の節番号)` with no phrase after `：`.
- When `：短い言い切り` is used, the phrase must be short, decisive, and not a question.
- Every slide title must include, in parentheses, the corresponding source section number from the paper (e.g., `"(3)"`, `"(4.2)"`, `"(5.1)"`).
- Section references in slide titles do not need to follow paper order. Presentation story order takes priority.

## Required Output Sections

Every presentation must address these four questions explicitly:

1. What is new? — State the delta from prior work in one sentence.
2. What can it do and what can it not? — Scope, assumptions, and constraints.
3. Under what conditions does it work? — Data/compute/setting dependencies, ablation highlights.
4. What to know for reproduction? — Key equations, hyperparameters, and pitfalls.

## Workflow

1. Gather constraints.
   - Ask for time limit, audience level, required depth, format (slides/notes), and output type (outline, script).
   - Define the explanation goal: "After this talk, the audience should be able to \_\_\_."

2. Extract the paper core.
   - Identify problem, gap, key idea, contributions, assumptions, and novelty versus prior work.
   - Write a one-sentence takeaway and three-bullet contributions.

3. Build a story arc.
   - Use internally: Problem, Gap, Idea, How, Evidence, Limits, Takeaway.
   - These labels are for planning only. Final slide titles must follow the mandatory fixed-category format above.
   - Append section mapping in parentheses to each slide title per the mandatory rule above.

4. Handle equations.
   - Give intuition before math. Define variables and assumptions. Show the equation. Explain why this form arises and what it enables.

5. Use figures as evidence.
   - For each figure: what question it answers, how to read it, and the one-sentence takeaway.

6. Summarize experiments.
   - State task, dataset, metric, baseline, main result, and implication.

7. Produce output.
   - Slide outline with per-slide goal, key message, and time budget.
   - If requested, produce a Quarto .qmd slide deck (see references/quarto_slides.md).

## Writing Style Rules (Critical)

All output must be in Japanese. Follow these rules strictly:

- Write in plain, natural Japanese sentences. Do not compress vocabulary or omit particles and adverbs for brevity. Write as a human would speak.
- Keep tone polite and use decisive sentence endings with `〜する` as default instead of `〜します` / `〜です`.
- When a harder formal tone is needed, prefer sentence-final nominal style (体言止め) over repeated `〜である`.
- Do not over-use markdown formatting. Avoid excessive bold, nested lists, or decorative headers. A flat, readable structure is preferred. Use bold only when truly necessary for emphasis.
- Prefer full sentences over noun-phrase bullet fragments. "この手法はXを仮定している" is better than "仮定: X".
- Do not produce "translation-style" output that mirrors the English paper sentence by sentence. Restructure into claim, evidence, implication.
- Keep technical terms in their original English where that is the convention in the field (e.g., "attention mechanism", "ablation study"), but write the surrounding explanation in natural Japanese.
- Use bullet points as the default output structure for slide outlines, notes, and Q&A prep.

## Output Formats

- Presentation header block containing paper-exact title and paper author list (or standard abbreviation when many).
- Slide outline with per-slide goal, key message, and time budget.
- Speaking notes or narrative summary.
- Quarto slide deck (.qmd) when asked.

## References

- Read `references/presentation_guide.md` for templates, checklists, and phrasing patterns.
- Read `references/quarto_slides.md` for Quarto slide deck conventions and structure.
