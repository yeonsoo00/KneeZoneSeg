"""Export aligned PSD signal layers using one shared overlapping crop window."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from format_layered_psd import (
    STAIN_BY_NUMBER,
    find_psd_files,
    numbered_layers,
    render_layer_on_canvas,
)


def layer_canvas_box(layer, canvas_size: tuple[int, int]) -> tuple[int, int, int, int]:
    """Return a PSD-space layer/group box, clamped to the canvas."""
    bbox = layer.bbox
    raw = (
        int(bbox.x1 if hasattr(bbox, "x1") else bbox[0]),
        int(bbox.y1 if hasattr(bbox, "y1") else bbox[1]),
        int(bbox.x2 if hasattr(bbox, "x2") else bbox[2]),
        int(bbox.y2 if hasattr(bbox, "y2") else bbox[3]),
    )
    width, height = canvas_size
    box = (
        max(0, min(width, raw[0])),
        max(0, min(height, raw[1])),
        max(0, min(width, raw[2])),
        max(0, min(height, raw[3])),
    )
    if box[2] <= box[0] or box[3] <= box[1]:
        raise ValueError(f"Layer {layer.name!r} has an empty PSD-space window")
    return box


def common_crop_box(
    layers: dict[int, object],
    canvas_size: tuple[int, int],
) -> tuple[int, int, int, int]:
    """Return the intersection window shared by all nine signal layers."""
    boxes = {
        number: layer_canvas_box(layers[number], canvas_size)
        for number in STAIN_BY_NUMBER
    }
    crop = (
        max(box[0] for box in boxes.values()),
        max(box[1] for box in boxes.values()),
        min(box[2] for box in boxes.values()),
        min(box[3] for box in boxes.values()),
    )
    if crop[2] <= crop[0] or crop[3] <= crop[1]:
        details = ", ".join(f"{number}_={box}" for number, box in boxes.items())
        raise ValueError(
            "The nine aligned signal layers have no common overlapping window: "
            + details
        )
    return crop


def format_psd_dataset(
    psd_source: str | Path,
    output_root: str | Path,
    *,
    overwrite: bool = False,
) -> Path:
    """Export every signal with the identical common PSD-space crop."""
    try:
        from psd_tools import PSDImage
    except ImportError as exc:
        raise RuntimeError(
            "PSD input requires psd-tools. Install it with: pip install psd-tools"
        ) from exc

    files = find_psd_files(psd_source)
    if not files:
        raise FileNotFoundError(f"No .psd or .psb files found under: {psd_source}")

    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    sample_names: set[str] = set()
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

        layer_boxes = {
            number: layer_canvas_box(layers[number], psd.size)
            for number in STAIN_BY_NUMBER
        }
        crop_box = common_crop_box(layers, psd.size)
        metadata = {
            "source_psd": str(source.resolve()),
            "source_size_bytes": source.stat().st_size,
            "source_mtime_ns": source.stat().st_mtime_ns,
            "psd_size": list(psd.size),
            "common_crop_box": list(crop_box),
            "cropped_size": [crop_box[2] - crop_box[0], crop_box[3] - crop_box[1]],
            "layer_boxes": {
                f"{number}_{STAIN_BY_NUMBER[number]}": list(layer_boxes[number])
                for number in STAIN_BY_NUMBER
            },
        }

        sample_dir = output_root / sample
        sample_dir.mkdir(parents=True, exist_ok=True)
        metadata_path = sample_dir / "crop_metadata.json"
        metadata_matches = False
        if metadata_path.is_file():
            try:
                metadata_matches = json.loads(metadata_path.read_text()) == metadata
            except (OSError, json.JSONDecodeError):
                pass

        for number, stain in STAIN_BY_NUMBER.items():
            destination = sample_dir / f"{stain}.png"
            if destination.is_file() and not overwrite and metadata_matches:
                continue
            # First place every layer in the same PSD coordinate system, then
            # apply exactly one shared crop box. No signal is cropped by itself.
            aligned = render_layer_on_canvas(layers[number], psd.size)
            aligned.crop(crop_box).convert("RGB").save(destination, format="PNG")

        metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
        width, height = metadata["cropped_size"]
        print(f"  {sample}: common PSD crop {crop_box} -> {width}x{height}")

    print(f"Formatted {len(files)} common-cropped PSD sample(s) in {output_root}")
    return output_root


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export aligned PSD signals using their common crop window"
    )
    parser.add_argument("psd_source", help="One layered PSD/PSB or a directory containing them")
    parser.add_argument("output_root", help="Directory for model-ready sample folders")
    parser.add_argument("--overwrite", action="store_true", help="Re-render existing PNG files")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    format_psd_dataset(args.psd_source, args.output_root, overwrite=args.overwrite)
