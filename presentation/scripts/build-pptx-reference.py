#!/usr/bin/env python3
import argparse
import re
import subprocess
import tempfile
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile


FONT_SCALE_RE = re.compile(r"^\s*font_scale:\s*([0-9]*\.?[0-9]+)\s*$", re.MULTILINE)
MIN_SIZE_RE = re.compile(r"^\s*minimum_size:\s*(\d+)\s*$", re.MULTILINE)
SIZE_ATTR_RE = re.compile(r'sz="(\d+)"')


def parse_config(config_path: Path) -> tuple[float, int]:
    text = config_path.read_text(encoding="utf-8")

    scale_match = FONT_SCALE_RE.search(text)
    min_match = MIN_SIZE_RE.search(text)
    if not scale_match or not min_match:
        raise ValueError(
            f"Invalid config in {config_path}. Required keys: font_scale, minimum_size."
        )

    font_scale = float(scale_match.group(1))
    minimum_size = int(min_match.group(1))

    if font_scale <= 0:
        raise ValueError("font_scale must be > 0.")
    if minimum_size <= 0:
        raise ValueError("minimum_size must be > 0.")

    return font_scale, minimum_size


def build_reference_doc(output_path: Path, font_scale: float, minimum_size: int) -> None:
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        source_pptx = tmp_path / "reference-default.pptx"
        unzip_dir = tmp_path / "unzip"
        unzip_dir.mkdir(parents=True, exist_ok=True)

        with source_pptx.open("wb") as f:
            subprocess.run(
                ["quarto", "pandoc", "--print-default-data-file", "reference.pptx"],
                check=True,
                stdout=f,
            )

        with ZipFile(source_pptx, "r") as zf:
            zf.extractall(unzip_dir)

        for xml_path in unzip_dir.rglob("*.xml"):
            if "ppt" not in xml_path.parts:
                continue
            text = xml_path.read_text(encoding="utf-8")
            updated = SIZE_ATTR_RE.sub(
                lambda m: f'sz="{max(minimum_size, int(round(int(m.group(1)) * font_scale)))}"',
                text,
            )
            if updated != text:
                xml_path.write_text(updated, encoding="utf-8")

        output_path.parent.mkdir(parents=True, exist_ok=True)
        with ZipFile(output_path, "w", compression=ZIP_DEFLATED) as zf:
            for path in unzip_dir.rglob("*"):
                if path.is_file():
                    zf.write(path, path.relative_to(unzip_dir))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a custom PPTX reference document from YAML config."
    )
    parser.add_argument(
        "--config",
        default="presentation/_pptx-reference.yml",
        help="Path to YAML config file.",
    )
    parser.add_argument(
        "--output",
        default="presentation/_templates/pptx-reference-small.pptx",
        help="Output PPTX path.",
    )
    args = parser.parse_args()

    config_path = Path(args.config).resolve()
    output_path = Path(args.output).resolve()

    font_scale, minimum_size = parse_config(config_path)
    build_reference_doc(output_path, font_scale, minimum_size)
    print(f"Generated: {output_path}")


if __name__ == "__main__":
    main()
