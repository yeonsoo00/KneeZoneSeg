"""Reusable gap measurement and visualization for predicted growth-plate zones."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image


STAINS = ("mineral", "ac", "calcein", "trap", "dapi", "ap", "edu", "cfo", "sfo")
STAIN_TO_DIGIT = {name: str(index + 1) for index, name in enumerate(STAINS)}


@dataclass(frozen=True)
class GapResult:
    gap_mask: np.ndarray
    thicknesses: np.ndarray
    trimmed_mean: float
    standard_deviation: float
    maximum: float
    x_start: int
    x_end: int
    upper_zone: str
    lower_zone: str


def load_zone_mask(prediction_dir: str | Path, stain: str) -> np.ndarray:
    """Load and union all predicted subzone masks belonging to one stain."""
    prediction_dir = Path(prediction_dir)
    digit = STAIN_TO_DIGIT[stain]
    paths = sorted(prediction_dir.glob(f"{digit}[a-h].png"))
    if not paths:
        raise FileNotFoundError(f"No {stain} prediction masks ({digit}a-{digit}h) in {prediction_dir}")

    union = None
    for path in paths:
        mask = np.asarray(Image.open(path).convert("L")) > 127
        if union is None:
            union = mask
        elif mask.shape != union.shape:
            raise ValueError(f"Prediction mask sizes differ in {prediction_dir}")
        else:
            union |= mask
    return union


def _trimmed_mean(values: np.ndarray, trim_fraction: float) -> float:
    ordered = np.sort(values.astype(np.float64))
    trim = int(np.floor(ordered.size * trim_fraction))
    if trim and ordered.size > 2 * trim:
        ordered = ordered[trim:-trim]
    return float(ordered.mean())


def measure_vertical_gap(
    first_mask: np.ndarray,
    second_mask: np.ndarray,
    first_name: str,
    second_name: str,
    trim_fraction: float = 0.10,
) -> GapResult:
    """Measure the vertical, facing-boundary gap in mutually supported columns.

    The horizontal domain is the intersection of columns occupied by both zones.
    This makes the measurement follow the shorter horizontal extent. The zone with
    the smaller median y coordinate is treated as the upper zone.
    """
    if first_mask.shape != second_mask.shape or first_mask.ndim != 2:
        raise ValueError("The selected zone masks must be equally sized 2-D arrays")
    if not first_mask.any() or not second_mask.any():
        raise ValueError("Both selected zones must contain predicted pixels")

    first_median = float(np.median(np.nonzero(first_mask)[0]))
    second_median = float(np.median(np.nonzero(second_mask)[0]))
    if first_median <= second_median:
        upper, lower = first_mask, second_mask
        upper_name, lower_name = first_name, second_name
    else:
        upper, lower = second_mask, first_mask
        upper_name, lower_name = second_name, first_name

    common_columns = np.flatnonzero(upper.any(axis=0) & lower.any(axis=0))
    if common_columns.size == 0:
        raise ValueError("The selected zones have no shared horizontal range")

    gap_mask = np.zeros_like(upper, dtype=bool)
    thicknesses = []
    used_columns = []
    for x in common_columns:
        upper_bottom = int(np.flatnonzero(upper[:, x])[-1])
        lower_top = int(np.flatnonzero(lower[:, x])[0])
        thickness = max(0, lower_top - upper_bottom - 1)
        thicknesses.append(thickness)
        used_columns.append(int(x))
        if thickness:
            gap_mask[upper_bottom + 1:lower_top, x] = True

    values = np.asarray(thicknesses, dtype=np.float64)
    return GapResult(
        gap_mask=gap_mask,
        thicknesses=values,
        trimmed_mean=_trimmed_mean(values, trim_fraction),
        standard_deviation=float(values.std(ddof=0)),
        maximum=float(values.max()),
        x_start=min(used_columns),
        x_end=max(used_columns),
        upper_zone=upper_name,
        lower_zone=lower_name,
    )


def _load_normalized_signal(path: Path, size: tuple[int, int]) -> np.ndarray:
    image = Image.open(path).convert("RGB").resize(size, Image.Resampling.BILINEAR)
    array = np.asarray(image, dtype=np.float32)
    low, high = np.percentile(array, (1, 99))
    if high <= low:
        return np.zeros_like(array, dtype=np.float32)
    return np.clip((array - low) / (high - low), 0.0, 1.0)


def render_gap_overlay(
    signal_dir: str | Path,
    background_signals: Iterable[str],
    first_mask: np.ndarray,
    second_mask: np.ndarray,
    gap_mask: np.ndarray,
) -> Image.Image:
    """Render selected signals with two zone masks and their gap overlaid."""
    height, width = first_mask.shape
    arrays = []
    for signal in background_signals:
        path = Path(signal_dir) / f"{signal}.png"
        if path.exists():
            arrays.append(_load_normalized_signal(path, (width, height)))
    background = np.mean(arrays, axis=0) if arrays else np.zeros((height, width, 3), np.float32)
    canvas = (background * 255.0).astype(np.uint8)

    def blend(mask: np.ndarray, color: tuple[int, int, int], alpha: float) -> None:
        color_array = np.asarray(color, dtype=np.float32)
        canvas[mask] = ((1.0 - alpha) * canvas[mask] + alpha * color_array).astype(np.uint8)

    blend(first_mask, (0, 210, 255), 0.45)
    blend(second_mask, (210, 70, 255), 0.45)
    blend(gap_mask, (255, 45, 30), 0.72)
    return Image.fromarray(canvas, mode="RGB")
