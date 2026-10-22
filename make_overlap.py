#!/usr/bin/env python3
"""Draw prediction-mask boundaries over original-size signal images.

The first digit of a mask name selects its signal image:
1=mineral, 2=AC, 3=calcein, 4=TRAP, 5=DAPI, 6=AP,
7=EdU, 8=CFO, and 9=SFO. Cross-signal masks use a 50/50 blend.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import cv2
import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent

DEFAULT_PRED_ROOT = SCRIPT_DIR / "Output"
DEFAULT_IMAGE_ROOT = PROJECT_DIR / "Input" / "Testdata" / "Image"
DEFAULT_OUTPUT_ROOT = DEFAULT_PRED_ROOT / "Overlap"

SIGNAL_CANDIDATES = {
    "mineral": ("mineral.png", "mineral_high.png"),
    "ac": ("ac_high.png", "ac.png"),
    "calcein": ("calcein_high.png", "calcein.png", "calcein2.png"),
    "trap": ("trap_high.png", "trap.png"),
    "dapi": ("dapi_high.png", "dapi.png"),
    "ap": ("ap_high.png", "ap.png", "ap_high2.png"),
    "edu": ("edu_high.png", "edu.png"),
    "cfo": ("cfo_high.png", "cfo.png"),
    "sfo": ("sfo_high.png", "sfo.png"),
}

DIGIT_TO_SIGNAL = {
    "1": "mineral",
    "2": "ac",
    "3": "calcein",
    "4": "trap",
    "5": "dapi",
    "6": "ap",
    "7": "edu",
    "8": "cfo",
    "9": "sfo",
}

# Masks predicted from two signal branches use both signals as background.
CROSS_SIGNALS = {
    "9a": ("sfo", "cfo"),
    "9b": ("sfo", "cfo"),
    "8e": ("cfo", "ap"),
    "8f": ("cfo", "ap"),
    "6g": ("ap", "trap"),
    "6h": ("ap", "trap"),
    "8h": ("ap", "trap"),
}

MASK_RE = re.compile(r"^[1-9][a-h]$", re.IGNORECASE)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Draw green mask outlines over original-size signal images."
    )
    parser.add_argument("--pred-root", type=Path, default=DEFAULT_PRED_ROOT)
    parser.add_argument("--image-root", type=Path, default=DEFAULT_IMAGE_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument(
        "--mask-threshold", type=int, default=127,
        help="Mask threshold in the 0-255 prediction PNG (default: 127).",
    )
    parser.add_argument(
        "--boundary-thickness", type=int, default=4,
        help="Green outline thickness in original-image pixels (default: 4).",
    )
    parser.add_argument(
        "--overwrite", action="store_true", help="Replace existing overlap PNGs."
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Check matching files without writing."
    )
    return parser.parse_args()


def find_signal(sample_dir: Path, signal: str) -> Path:
    for filename in SIGNAL_CANDIDATES[signal]:
        path = sample_dir / filename
        if path.is_file():
            return path
    names = ", ".join(SIGNAL_CANDIDATES[signal])
    raise FileNotFoundError(f"No {signal} signal in {sample_dir} (tried {names})")


def read_display_image(path: Path) -> np.ndarray:
    """Read any common PNG depth and return a display-ready BGR uint8 image."""
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise OSError(f"Could not read {path}")
    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    elif image.ndim == 3 and image.shape[2] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)

    if image.dtype == np.uint8:
        return image

    values = image.astype(np.float32)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return np.zeros(image.shape, dtype=np.uint8)
    low, high = np.percentile(finite, (0.5, 99.5))
    if high <= low:
        return np.zeros(image.shape, dtype=np.uint8)
    return np.clip((values - low) * (255.0 / (high - low)), 0, 255).astype(np.uint8)


def make_background(sample_dir: Path, mask_key: str) -> np.ndarray:
    signals = CROSS_SIGNALS.get(mask_key, (DIGIT_TO_SIGNAL[mask_key[0]],))
    images = [read_display_image(find_signal(sample_dir, name)) for name in signals]
    target_h, target_w = images[0].shape[:2]
    aligned = [images[0]]
    for image in images[1:]:
        if image.shape[:2] != (target_h, target_w):
            image = cv2.resize(image, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
        aligned.append(image)
    return np.mean(np.stack(aligned).astype(np.float32), axis=0)


def make_overlap(
    background: np.ndarray,
    mask_path: Path,
    threshold: int,
    boundary_thickness: int,
) -> np.ndarray:
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise OSError(f"Could not read {mask_path}")
    height, width = background.shape[:2]
    if mask.shape != (height, width):
        mask = cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST)
    mask_binary = (mask > threshold).astype(np.uint8)
    contours, _ = cv2.findContours(
        mask_binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE
    )
    overlap = np.clip(background, 0, 255).astype(np.uint8)
    cv2.drawContours(
        overlap, contours, -1, (0, 255, 0), boundary_thickness, cv2.LINE_AA
    )
    return overlap


def main() -> int:
    args = parse_args()
    if args.boundary_thickness < 1:
        raise ValueError("Boundary thickness must be at least 1")
    if not args.pred_root.is_dir():
        raise FileNotFoundError(f"Prediction root does not exist: {args.pred_root}")
    if not args.image_root.is_dir():
        raise FileNotFoundError(f"Image root does not exist: {args.image_root}")

    written = skipped = failed = 0
    sample_dirs = sorted(
        path for path in args.pred_root.iterdir()
        if path.is_dir() and path.resolve() != args.output_root.resolve()
    )
    for pred_dir in sample_dirs:
        sample_dir = args.image_root / pred_dir.name
        if not sample_dir.is_dir():
            print(f"[skip] Missing background folder: {sample_dir}")
            skipped += 1
            continue

        mask_paths = sorted(
            path for path in pred_dir.glob("*.png") if MASK_RE.fullmatch(path.stem)
        )
        for mask_path in mask_paths:
            mask_key = mask_path.stem.lower()
            destination = args.output_root / pred_dir.name / mask_path.name
            if destination.exists() and not args.overwrite:
                skipped += 1
                continue
            try:
                background = make_background(sample_dir, mask_key)
                if args.dry_run:
                    print(f"[ok] {mask_path} -> {destination} ({background.shape[1]}x{background.shape[0]})")
                    written += 1
                    continue
                overlap = make_overlap(
                    background, mask_path, args.mask_threshold,
                    args.boundary_thickness,
                )
                destination.parent.mkdir(parents=True, exist_ok=True)
                if not cv2.imwrite(str(destination), overlap):
                    raise OSError(f"Could not write {destination}")
                written += 1
            except (FileNotFoundError, OSError, ValueError) as error:
                print(f"[warning] {pred_dir.name}/{mask_path.name}: {error}")
                failed += 1

    action = "would create" if args.dry_run else "created"
    print(f"Done: {action} {written}, skipped {skipped}, failed {failed}.")
    print(f"Output root: {args.output_root}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
