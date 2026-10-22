"""Export numbered signal layers from layered PSD files for inference."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

from PIL import Image

STAIN_BY_NUMBER = {
    1: "mineral", 2: "ac", 3: "calcein", 4: "trap", 5: "dapi",
    6: "ap", 7: "edu", 8: "cfo", 9: "sfo",
}
LAYER_NUMBER_PATTERN = re.compile(r"^\s*([1-9])_")


def find_psd_files(source: str | Path) -> list[Path]:
    """Return one selected PSD/PSB or all PSD/PSB files below a directory."""
    source = Path(source).expanduser()
    if source.is_file():
        if source.suffix.casefold() not in {".psd", ".psb"}:
            raise ValueError(f"Input file is not a PSD or PSB: {source}")
        return [source]
    if not source.is_dir():
        raise FileNotFoundError(f"PSD input does not exist: {source}")
    return sorted(
        path for path in source.rglob("*")
        if path.is_file() and path.suffix.casefold() in {".psd", ".psb"}
    )


def numbered_layers(container) -> dict[int, object]:
    """Find numbered layers recursively, treating a numbered group as one signal."""
    found: dict[int, object] = {}

    def visit(current) -> None:
        for layer in current:
            match = LAYER_NUMBER_PATTERN.match(layer.name or "")
            if match:
                number = int(match.group(1))
                if number in found:
                    raise ValueError(
                        f"Duplicate {number}_ layers: {found[number].name!r} and {layer.name!r}"
                    )
                found[number] = layer
            elif layer.is_group():
                visit(layer)

    visit(container)
    return found


def render_layer_on_canvas(layer, canvas_size: tuple[int, int]) -> Image.Image:
    """Composite one layer/group at its PSD coordinates on a full-size canvas."""
    rendered = layer.composite()
    if rendered is None:
        raise ValueError(f"Layer {layer.name!r} has no renderable pixels")
    rendered = rendered.convert("RGBA")
    if rendered.size == canvas_size:
        return rendered
    bbox = layer.bbox
    left = int(bbox.x1 if hasattr(bbox, "x1") else bbox[0])
    top = int(bbox.y1 if hasattr(bbox, "y1") else bbox[1])
    canvas = Image.new("RGBA", canvas_size, (0, 0, 0, 0))
    canvas.alpha_composite(rendered, dest=(left, top))
    return canvas


def format_psd_dataset(
    psd_source: str | Path,
    output_root: str | Path,
    *,
    overwrite: bool = False,
) -> Path:
    """Export signal layers from one PSD or a directory of PSD samples."""
    try:
        from psd_tools import PSDImage
    except ImportError as exc:
        raise RuntimeError(
            "PSD input requires psd-tools. Install it with: pip install psd-tools"
        ) from exc

    files = find_psd_files(psd_source)
    if not files:
        raise FileNotFoundError(f"No .psd or .psb files found under: {psd_source}")

    sample_names: set[str] = set()
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    for source in files:
        sample = source.stem.strip()
        if sample in sample_names:
            raise ValueError(f"Multiple PSD files have the same sample name: {sample!r}")
        sample_names.add(sample)
        psd = PSDImage.open(source)
        layers = numbered_layers(psd)
        missing = sorted(set(STAIN_BY_NUMBER) - set(layers))
        if missing:
            names = ", ".join(f"{n}_ ({STAIN_BY_NUMBER[n]})" for n in missing)
            raise ValueError(f"PSD {source.name!r} is missing required signal layers: {names}")

        sample_dir = output_root / sample
        sample_dir.mkdir(parents=True, exist_ok=True)
        for number, stain in STAIN_BY_NUMBER.items():
            destination = sample_dir / f"{stain}.png"
            if destination.exists() and not overwrite:
                continue
            render_layer_on_canvas(layers[number], psd.size).convert("RGB").save(destination, format="PNG")

    print(f"Formatted {len(files)} layered PSD sample(s) in {output_root}")
    return output_root


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export numbered PSD layers for inference")
    parser.add_argument("psd_source", help="One layered PSD/PSB or a directory containing them")
    parser.add_argument("output_root", help="Directory for model-ready sample folders")
    parser.add_argument("--overwrite", action="store_true", help="Re-render existing PNG files")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    format_psd_dataset(args.psd_source, args.output_root, overwrite=args.overwrite)
