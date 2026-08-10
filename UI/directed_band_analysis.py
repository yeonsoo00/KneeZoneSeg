"""Directed mask subtraction, automatic reversal, and ROI-clipped band statistics."""

from __future__ import annotations

import numpy as np
from PIL import Image

from zone_analysis import GapResult


def _trimmed_mean(values: np.ndarray, fraction: float = 0.10) -> float:
    ordered = np.sort(values.astype(np.float64))
    trim = int(np.floor(ordered.size * fraction))
    if trim and ordered.size > 2 * trim:
        ordered = ordered[trim:-trim]
    return float(ordered.mean())


def _directed_gap(upper_mask: np.ndarray, lower_mask: np.ndarray) -> np.ndarray:
    """Build the positive band from lower-top minus upper-bottom per column."""
    common_x = np.flatnonzero(upper_mask.any(axis=0) & lower_mask.any(axis=0))
    gap = np.zeros_like(upper_mask, dtype=bool)
    for x in common_x:
        upper_bottom = int(np.flatnonzero(upper_mask[:, x])[-1])
        lower_top = int(np.flatnonzero(lower_mask[:, x])[0])
        if lower_top - upper_bottom - 1 > 0:
            gap[upper_bottom + 1:lower_top, x] = True
    return gap


def measure_positive_gap_band(
    first_mask: np.ndarray,
    second_mask: np.ndarray,
    first_name: str,
    second_name: str,
    roi: tuple[int, int, int, int] | None = None,
) -> GapResult:
    """Measure a user-ordered gap, reversing order only if it is non-positive."""
    if first_mask.shape != second_mask.shape or first_mask.ndim != 2:
        raise ValueError("The selected masks must be equally sized 2-D arrays")
    if not first_mask.any() or not second_mask.any():
        raise ValueError("Both selected masks must contain predicted pixels")

    full_gap = _directed_gap(first_mask, second_mask)
    upper_name, lower_name = first_name, second_name
    if not full_gap.any():
        full_gap = _directed_gap(second_mask, first_mask)
        upper_name, lower_name = second_name, first_name
    if not full_gap.any():
        raise ValueError("Neither mask order produces a positive in-between band")

    height, width = full_gap.shape
    if roi is None:
        x_min, y_min, x_max, y_max = 0, 0, width - 1, height - 1
    else:
        x_min, y_min, x_max, y_max = roi
        x_min, x_max = sorted((max(0, min(width - 1, int(x_min))), max(0, min(width - 1, int(x_max)))))
        y_min, y_max = sorted((max(0, min(height - 1, int(y_min))), max(0, min(height - 1, int(y_max)))))

    # This is the actual cut-off: no positive-band pixel outside the rubber-band
    # rectangle survives into either the preview or the statistics.
    restricted = np.zeros_like(full_gap, dtype=bool)
    restricted[y_min:y_max + 1, x_min:x_max + 1] = full_gap[
        y_min:y_max + 1, x_min:x_max + 1
    ]
    counts = restricted.sum(axis=0).astype(np.float64)
    positive_x = np.flatnonzero(counts > 0)
    if positive_x.size == 0:
        raise ValueError("The rubber-band ROI contains no positive in-between band")
    thicknesses = counts[positive_x]

    return GapResult(
        gap_mask=restricted,
        thicknesses=thicknesses,
        trimmed_mean=_trimmed_mean(thicknesses),
        standard_deviation=float(thicknesses.std(ddof=0)),
        maximum=float(thicknesses.max()),
        x_start=int(positive_x.min()),
        x_end=int(positive_x.max()),
        upper_zone=upper_name,
        lower_zone=lower_name,
    )


def make_positive_band_preview(gap_mask: np.ndarray) -> Image.Image:
    image = np.zeros((*gap_mask.shape, 3), dtype=np.uint8)
    image[gap_mask] = (255, 255, 255)
    return Image.fromarray(image, mode="RGB")
