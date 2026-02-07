# Paper Presentation Guide

## Quick Questions to Ask

- Time limit and expected slide count
- Audience expertise level
- Required sections (background, method, experiments, related work)
- Goal: "After this talk, the audience should be able to ___."

## The Four Required Questions

Every presentation must explicitly answer these. They form the backbone of the talk.

### 1. What is new?

State how this paper differs from prior work in one sentence. This is the novelty claim. It should be concrete enough that the audience can repeat it after the talk.

### 2. What can it do and what can it not?

Clarify the scope of applicability. What assumptions does the method require? What inputs, data types, or domains does it cover? Where does it explicitly not apply? This prevents the audience from over-generalizing the results.

### 3. Under what conditions does it work?

Identify the dependencies: dataset characteristics, compute requirements, hyperparameter sensitivity. Summarize ablation results to show which components matter and which do not.

### 4. What to know for reproduction?

List the key equations (with variable definitions), critical hyperparameters, and known pitfalls. If someone in the audience wanted to implement this, what would they need to get right?

## Story Arc Template

1. Problem setup — why this matters
2. Prior work limitations — what the gap is
3. Core idea — the one new thing
4. Mechanism — intuition first, then equations and details
5. Experimental evidence — what improved and by how much
6. Limitations and future work — honest assessment
7. Takeaway — restate the one-sentence novelty

## Equation Handling

- Provide intuition before showing math
- Define all variables and state assumptions
- Explain where the equation comes from (derivation or design choice)
- State what the equation enables or why it matters

## Figure Handling

- State what question the figure answers
- Explain axes, comparisons, and conditions
- Summarize the takeaway in one sentence

## Avoiding Literal Translation

Do not translate English sentences into Japanese one-by-one. Instead, restructure:

- Original: "We minimize X by Y"
- Restructured: "目的はXを達成することである。仮定Yのもとで、Zを最小化することでこれを実現する。"

Always convert to the pattern: claim, evidence, implication.

## Slide Outline Templates

### 10 min / 8-10 slides

1. Title and one-line takeaway
2. Problem and motivation
3. Prior work gap
4. Key idea (with diagram)
5. Method details (equation intuition)
6. Method details (architecture or algorithm)
7. Experiment setup
8. Main results
9. Limitations and ablation
10. Takeaway and Q&A

### 20 min / 12-15 slides

The 10 min version plus:

- Related work positioning
- Additional ablations or qualitative examples
- Error analysis or failure cases

## Q&A Prep

- 3 likely weaknesses the audience might point out
- 2 strongest evidence points to defend the paper
- 1 extension idea or deployment scenario

## Writing Style Reminders

These apply to all Japanese output:

- Write full, natural sentences. Do not drop particles or adverbs for compression.
- Avoid excessive markdown formatting. Flat, readable text is preferred.
- Do not mirror the English paper sentence by sentence. Restructure ideas.
- Keep standard English technical terms as-is (attention mechanism, ablation study, etc.) but write explanations in natural Japanese.
