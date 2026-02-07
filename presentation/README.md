# Presentation

This directory is for presentation materials.

## Configuration structure

- Common format settings are managed in `presentation/_common-format.yml`.
- Each project keeps local settings in its own `_quarto.yml`.
- Quarto does not automatically inherit parent `_quarto.yml` from nested projects, so each project loads common settings via:

```yaml
metadata-files:
  - ../_common-format.yml
```

- `pptx` font sizing itself is controlled through `presentation/_pptx-reference.yml`.
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
