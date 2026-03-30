# Lab Presentation Guide

## Quick Questions

- Time limit and expected slide count
- Audience expertise level
- Required sections (background, method, experiments, related work)
- Goal: "Who should understand what, and to what depth?"
- Artifact: outline, review feedback, or editable deck

## Goal Definition

- Write a single sentence of what the audience should be able to do after the talk
- Example: "Lab members can explain the assumptions and when to use this method"

## Story Template

1. Problem setup
2. Prior work limitations (gap)
3. Core idea (what is new)
4. Mechanism (intuition -> equations/details)
5. Experimental evidence (what improved and by how much)
6. Limitations and future work
7. Takeaway

## Slide Writing Rules

- One message per slide
- Prefer claim-oriented titles over section labels
- Put dense derivations into speaker notes unless the audience needs the full derivation
- Keep tables only when the exact values matter; otherwise convert to a visual or a sentence
- End evidence slides with a one-line interpretation, not just the raw result

## Equation Handling Checklist

- Provide intuition first
- Define variables and assumptions
- Explain where it comes from (derivation or design choice)
- State what it enables or why it matters

## Figure Handling Checklist

- What question does this figure answer?
- Explain axes, comparisons, and conditions
- Summarize the takeaway in one sentence

## Avoid Literal Translation

- Do not translate sentences; convert to "claim -> evidence -> implication"
- Do: state goal, assumptions, decisions, and constraints explicitly
- Example:
  - Original: "We minimize X by Y"
  - Rewrite: "Goal is X. Under assumption Y, we minimize Z to achieve it"

## Slide Outline Templates

### 5 min / 5-6 slides

1. Title + takeaway
2. Problem + gap
3. Key idea
4. Method or experiment setup
5. Main result
6. Takeaway + Q&A

### 10 min / 8-10 slides

1. Title + one-line takeaway
2. Problem & motivation
3. Prior work gap
4. Key idea (diagram)
5. Method details (equation intuition)
6. Method details (architecture/algorithm)
7. Experiment setup
8. Main results
9. Limitations / ablation
10. Takeaway + Q&A

### 20 min / 12-15 slides

- 10 min version +
- Related work positioning
- Additional ablations or qualitative examples
- Error analysis or failure cases

## Existing Deck Review Checklist

- Does each slide have a clear purpose?
- Are there slides with more than one core claim?
- Does the transition from gap -> idea -> evidence feel explicit?
- Are equations shown before their intuition is established?
- Are figures annotated enough for the audience to read them quickly?
- Is there at least one limitations slide or spoken acknowledgment?
- Does the last content slide restate the takeaway instead of ending on raw numbers?

## Q&A Prep

- 3 likely weaknesses
- 2 strongest evidence points
- 1 extension or deployment idea

## Suggested Output Structure

- Goal
- One-line takeaway
- Slide outline
- Speaking notes
- Likely questions
