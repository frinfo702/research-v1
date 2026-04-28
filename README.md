# Research Notes

A personal workspace for reading logs, seminar materials, and exploratory code snippets in machine learning and computer vision.

## What this is

This repository serves primarily as a document archive—slides, literature summaries, and investigation notes for seminars. The `src/` directory contains small, ad-hoc experiments and tutorials (e.g., GANs, diffusion models, transformers) run out of curiosity; most are incomplete and not intended as reusable implementations.

## Repository Structure

```txt
├── docs/          # Investigation notes, reading summaries, and seminar docs
├── manuscript/    # Drafts and writing materials
├── presentation/  # Slide decks and figures for seminars
├── src/           # Casual experiments and tutorial code (exploratory, not maintained)
├── assets/        # PDFs, attachments, and templates
├── pyproject.toml # Python environment definition
└── README.md
```

## Setup

Requires Python >= 3.12.

```bash
# Install dependencies
uv sync
```

Or with `pip`:

```bash
pip install -e .
```

## Dependencies

See `pyproject.toml` for the full list. The environment includes PyTorch, Transformers, Diffusers, TensorFlow, and Jupyter for quick prototyping.
