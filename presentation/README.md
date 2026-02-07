# Presentation

This directory is for presentation materials.
We plan to keep presentation materials in this structure going forward:

```text
presentation/
├── README.md
├── templates/
│   └── pptx/
│       ├── <reference-theme>.pptx        # Quarto/Pandoc reference-doc
│       └── notes.md                   # Template notes (layouts, usage)
├── shared/
│   ├── assets/
│   │   ├── logo/                      # University/lab logos, background assets
│   │   └── fonts/                     # If needed (check license requirements)
│   ├── figures/
│   │   ├── src/                       # Editable sources (drawio, figma exports, .ai, etc.)
│   │   └── export/                    # Slide-ready exports (pdf/png/svg)
│   └── style/
│       ├── brand.yml                  # Optional shared brand colors
│       └── csl/                       # Citation style if used in slides
├── decks/
│   ├── 2026-02-lab-seminar/
│   │   ├── slide.qmd                  # Quarto source
│   │   ├── slide.pptx                 # Generated output (commit policy TBD)
│   │   ├── data/                      # Per-deck data
│   │   ├── figures/                   # Per-deck figures
│   │   └── notes/                     # Speaker notes, Q&A
│   └── 2026-03-conference/
│       └── ...
├── scripts/
│   └── export_figures.sh              # Optional batch export script
└── build/                             # Output for all decks (if used)
```

## Configuration structure

- Common format settings are managed in `presentation/shared/style/_common-format.yml`.
- Each project keeps local settings in its own `_quarto.yml`.
- Quarto does not automatically inherit parent `_quarto.yml` from nested projects, so each project loads common settings via:

```yaml
metadata-files:
  - ../../shared/style/_common-format.yml
```

- `pptx` font sizing itself is controlled through `presentation/shared/style/_pptx-reference.yml`.
- Generate the reference PPTX after updating the YAML:

```bash
python3 presentation/scripts/build-pptx-reference.py
```

## Create new project

You can create new sub-directory by following command

```bash
quarto create project
```

Then add `metadata-files` in each project `_quarto.yml` to apply common settings.

## Output the result

You can get the output by following command. 
By default, output format is set to `.pptx` and `.pdf`.

```bash
quarto render <file-path.qmd>
```
