"""Positive gap-band statistics with optional rectangular ROI restriction."""

from __future__ import annotations

import numpy as np
from PIL import Image

from zone_analysis import GapResult, measure_vertical_gap


def _trimmed_mean(values: np.ndarray, fraction: float = 0.10) -> float:
    ordered = np.sort(values.astype(np.float64))
    trim = int(np.floor(ordered.size * fraction))
    if trim and ordered.size > 2 * trim:
        ordered = ordered[trim:-trim]
    return float(ordered.mean())


def measure_positive_gap_band(
    first_mask: np.ndarray,
    second_mask: np.ndarray,
    first_name: str,
    second_name: str,
    roi: tuple[int, int, int, int] | None = None,
) -> GapResult:
    """Measure positive gap pixels per column, optionally inside an x/y ROI.

    ``roi`` is inclusive and ordered as ``(x_min, y_min, x_max, y_max)``.
    Columns with no positive gap pixels after ROI clipping are excluded from all
    statistics. Thus the reported thicknesses describe only the displayed
    ``gap > 0`` band.
    """
    full = measure_vertical_gap(first_mask, second_mask, first_name, second_name)
    height, width = full.gap_mask.shape
    if roi is None:
        x_min, y_min, x_max, y_max = 0, 0, width - 1, height - 1
    else:
        x_min, y_min, x_max, y_max = roi
        x_min = max(0, min(width - 1, int(x_min)))
        x_max = max(0, min(width - 1, int(x_max)))
        y_min = max(0, min(height - 1, int(y_min)))
        y_max = max(0, min(height - 1, int(y_max)))
        x_min, x_max = sorted((x_min, x_max))
        y_min, y_max = sorted((y_min, y_max))

    restricted = np.zeros_like(full.gap_mask, dtype=bool)
    restricted[y_min:y_max + 1, x_min:x_max + 1] = full.gap_mask[
        y_min:y_max + 1, x_min:x_max + 1
    ]
    column_counts = restricted.sum(axis=0).astype(np.float64)
    positive_x = np.flatnonzero(column_counts > 0)
    if positive_x.size == 0:
        raise ValueError("The selected range contains no positive in-between gap band")
    thicknesses = column_counts[positive_x]

    return GapResult(
        gap_mask=restricted,
        thicknesses=thicknesses,
        trimmed_mean=_trimmed_mean(thicknesses),
        standard_deviation=float(thicknesses.std(ddof=0)),
        maximum=float(thicknesses.max()),
        x_start=int(positive_x.min()),
        x_end=int(positive_x.max()),
        upper_zone=full.upper_zone,
        lower_zone=full.lower_zone,
    )


def make_positive_band_preview(gap_mask: np.ndarray) -> Image.Image:
    """Create a high-contrast preview: positive band in white, background black."""
    image = np.zeros((*gap_mask.shape, 3), dtype=np.uint8)
    image[gap_mask] = (255, 255, 255)
    return Image.fromarray(image, mode="RGB")
