"""Export numbered signal layers from layered PSD files for inference."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

from PIL import Image

STAIN_BY_NUMBER = {
    1: "mineral",
    2: "ac",
    3: "calcein",
    4: "trap",
    5: "dapi",
    6: "ap",
    7: "edu",
    8: "cfo",
    9: "sfo",
}
LAYER_NUMBER_PATTERN = re.compile(r"^\s*([1-9])_")
LAYERED_SUFFIXES = frozenset({".psd", ".psb", ".pxd"})


def find_psd_files(source: str | Path) -> list[Path]:
    """Return one layered document or all layered documents below a directory."""
    source = Path(source).expanduser()
    if source.is_dir() and source.suffix.casefold() == ".pxd":
        return [source]
    if source.is_file():
        if source.suffix.casefold() not in LAYERED_SUFFIXES:
            raise ValueError(f"Input file is not a PSD, PSB, or PXD: {source}")
        return [source]
    if not source.is_dir():
        raise FileNotFoundError(f"Layered input does not exist: {source}")
    return sorted(
        path
        for path in source.rglob("*")
        if (path.is_file() or (path.is_dir() and path.suffix.casefold() == ".pxd"))
        and path.suffix.casefold() in LAYERED_SUFFIXES
    )


def layered_sample_name(source: str | Path, dataset_source: str | Path) -> str:
    """Return a stable sample path while preserving recursive input folders."""
    source = Path(source)
    dataset_source = Path(dataset_source)
    if source == dataset_source or dataset_source.is_file():
        return source.stem.strip()
    return source.relative_to(dataset_source).with_suffix("").as_posix()


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
        raise FileNotFoundError(
            f"No .psd, .psb, or .pxd files found under: {psd_source}"
        )

    sample_names: set[str] = set()
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    for source in files:
        sample = layered_sample_name(source, psd_source)
        sample_key = sample.casefold()
        if sample_key in sample_names:
            raise ValueError(
                f"Multiple layered files resolve to the same sample path: {sample!r}"
            )
        sample_names.add(sample_key)
        try:
            psd = PSDImage.open(source)
        except Exception as exc:
            raise ValueError(
                f"Could not open layered file {source}. PXD input must contain "
                "Photoshop-compatible layered image data."
            ) from exc
        layers = numbered_layers(psd)
        missing = sorted(set(STAIN_BY_NUMBER) - set(layers))
        if missing:
            names = ", ".join(f"{n}_ ({STAIN_BY_NUMBER[n]})" for n in missing)
            raise ValueError(
                f"Layered file {source.name!r} is missing required signal layers: {names}"
            )

        sample_dir = output_root / sample
        sample_dir.mkdir(parents=True, exist_ok=True)
        for number, stain in STAIN_BY_NUMBER.items():
            destination = sample_dir / f"{stain}.png"
            if destination.exists() and not overwrite:
                continue
            render_layer_on_canvas(layers[number], psd.size).convert("RGB").save(
                destination, format="PNG"
            )

    print(f"Formatted {len(files)} layered sample(s) in {output_root}")
    return output_root


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export numbered layered documents for inference"
    )
    parser.add_argument(
        "psd_source", help="One layered PSD/PSB/PXD or a directory containing them"
    )
    parser.add_argument("output_root", help="Directory for model-ready sample folders")
    parser.add_argument(
        "--overwrite", action="store_true", help="Re-render existing PNG files"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    format_psd_dataset(args.psd_source, args.output_root, overwrite=args.overwrite)
