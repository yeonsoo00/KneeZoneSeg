"""Create and quantify the five biological growth-plate zones.

Zones are formed column-by-column between two predicted boundary masks. The
common horizontal support of each pair is trimmed by five percent at both
edges, matching the presentation workflow in ``test.ipynb``.
"""

from __future__ import annotations

import csv
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import numpy as np
from PIL import Image


STAINS = ("mineral", "ac", "calcein", "trap", "dapi", "ap", "edu", "cfo", "sfo")
SIDE_EXCLUSION = 0.05
DEFAULT_PIXEL_SIZE_UM = 3.45

ZONE_DEFINITIONS = OrderedDict(
    [
        ("Germinal", ("9a", "6b")),
        ("Columnar", ("6a", "6d")),
        ("Pre-hypertrophic", ("6c", "1d")),
        ("Hypertrophic", ("1c", "9d")),
        ("GP", ("9a", "9d")),
    ]
)

ZONE_FILENAMES = {
    "Germinal": "Germinal.png",
    "Columnar": "Columnar.png",
    "Pre-hypertrophic": "Pre_hypertrophic.png",
    "Hypertrophic": "Hypertrophic.png",
    "GP": "GP.png",
}


@dataclass(frozen=True)
class ZoneResult:
    """Mask, geometry, and signal statistics for one biological zone."""

    name: str
    boundary_1: str
    boundary_2: str
    mask: np.ndarray
    original_x_start: int
    original_x_end: int
    x_start: int
    x_end: int
    thickness_x: np.ndarray
    thickness_top: np.ndarray
    thickness_bottom: np.ndarray
    thicknesses_px: np.ndarray
    area_px: int
    area_um2: float
    width_px: int
    width_um: float
    mean_thickness_px: float
    mean_thickness_um: float
    median_thickness_px: float
    median_thickness_um: float
    std_thickness_px: float
    std_thickness_um: float
    min_thickness_px: float
    min_thickness_um: float
    max_thickness_px: float
    max_thickness_um: float
    mean_signal_gray: float
    mean_signal_r: float
    mean_signal_g: float
    mean_signal_b: float

    def csv_row(self) -> dict[str, object]:
        return {
            "Zone": self.name,
            "Boundary_1": self.boundary_1,
            "Boundary_2": self.boundary_2,
            "Side_exclusion_each_edge": SIDE_EXCLUSION,
            "Analyzed_x_start": self.x_start,
            "Analyzed_x_end": self.x_end,
            "Area_px": self.area_px,
            "Area_um2": self.area_um2,
            "Width_px": self.width_px,
            "Width_um": self.width_um,
            "Mean_thickness_px": self.mean_thickness_px,
            "Mean_thickness_um": self.mean_thickness_um,
            "Median_thickness_px": self.median_thickness_px,
            "Median_thickness_um": self.median_thickness_um,
            "Std_thickness_px": self.std_thickness_px,
            "Std_thickness_um": self.std_thickness_um,
            "Min_thickness_px": self.min_thickness_px,
            "Min_thickness_um": self.min_thickness_um,
            "Max_thickness_px": self.max_thickness_px,
            "Max_thickness_um": self.max_thickness_um,
            "Mean_signal_gray": self.mean_signal_gray,
            "Mean_signal_R": self.mean_signal_r,
            "Mean_signal_G": self.mean_signal_g,
            "Mean_signal_B": self.mean_signal_b,
        }


def load_binary_mask(path: str | Path) -> np.ndarray:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Required prediction mask was not found: {path}")
    return np.asarray(Image.open(path).convert("L")) > 0


def get_x_extent(mask: np.ndarray) -> tuple[int, int]:
    if mask.ndim != 2 or not mask.any():
        raise ValueError("Boundary mask must be a non-empty 2-D array")
    x = np.nonzero(mask)[1]
    return int(x.min()), int(x.max())


def get_band_centerline(mask: np.ndarray) -> np.ndarray:
    """Return median y per column, interpolating missing boundary columns."""
    width = mask.shape[1]
    centerline = np.full(width, np.nan, dtype=np.float32)
    for x in range(width):
        y = np.flatnonzero(mask[:, x])
        if y.size:
            centerline[x] = float(np.median(y))
    valid = np.isfinite(centerline)
    if valid.sum() < 2:
        raise ValueError("Boundary mask does not contain enough points")
    x_all = np.arange(width)
    return np.interp(x_all, x_all[valid], centerline[valid]).astype(np.float32)


def get_common_x_range(
    mask_1: np.ndarray,
    mask_2: np.ndarray,
    side_exclusion: float = SIDE_EXCLUSION,
) -> tuple[int, int, int, int]:
    """Return original and edge-trimmed common horizontal ranges, inclusive."""
    if not 0.0 <= side_exclusion < 0.5:
        raise ValueError("side_exclusion must be in [0, 0.5)")
    x1_min, x1_max = get_x_extent(mask_1)
    x2_min, x2_max = get_x_extent(mask_2)
    common_min = max(x1_min, x2_min)
    common_max = min(x1_max, x2_max)
    if common_min >= common_max:
        raise ValueError("The boundary masks have no common horizontal range")
    width = common_max - common_min + 1
    trim = int(round(width * side_exclusion))
    x_start, x_end = common_min + trim, common_max - trim
    if x_start >= x_end:
        raise ValueError("The common range is too small after edge exclusion")
    return common_min, common_max, x_start, x_end


def create_zone_between_masks(
    mask_1: np.ndarray,
    mask_2: np.ndarray,
    *,
    side_exclusion: float = SIDE_EXCLUSION,
    remove_boundary_masks: bool = True,
    padding: int = 0,
) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    """Fill the zone between two interpolated boundary centerlines."""
    if mask_1.shape != mask_2.shape or mask_1.ndim != 2:
        raise ValueError("Boundary masks must be equally sized 2-D arrays")
    height, _ = mask_1.shape
    line_1 = get_band_centerline(mask_1)
    line_2 = get_band_centerline(mask_2)
    ranges = get_common_x_range(mask_1, mask_2, side_exclusion)
    _, _, x_start, x_end = ranges
    zone = np.zeros(mask_1.shape, dtype=bool)
    for x in range(x_start, x_end + 1):
        y_top = max(0, min(int(round(line_1[x])), int(round(line_2[x]))) + padding)
        y_bottom = min(height - 1, max(int(round(line_1[x])), int(round(line_2[x]))) - padding)
        if y_bottom > y_top:
            zone[y_top : y_bottom + 1, x] = True
    if remove_boundary_masks:
        zone[mask_1 | mask_2] = False
    return zone, ranges


def _load_signal(signal_path: str | Path | None, shape: tuple[int, int]) -> np.ndarray | None:
    if signal_path is None:
        return None
    path = Path(signal_path)
    if not path.is_file():
        raise FileNotFoundError(f"Background signal was not found: {path}")
    height, width = shape
    image = Image.open(path).convert("RGB").resize((width, height), Image.Resampling.BILINEAR)
    return np.asarray(image, dtype=np.float32)


def _measure_zone(
    name: str,
    boundaries: tuple[str, str],
    mask: np.ndarray,
    ranges: tuple[int, int, int, int],
    signal_rgb: np.ndarray | None,
    pixel_size_um: float,
) -> ZoneResult:
    x_values, tops, bottoms, thicknesses = [], [], [], []
    for x in range(ranges[2], ranges[3] + 1):
        y = np.flatnonzero(mask[:, x])
        if y.size:
            x_values.append(x)
            tops.append(int(y.min()))
            bottoms.append(int(y.max()))
            thicknesses.append(int(y.max() - y.min() + 1))
    if not thicknesses:
        raise ValueError(f"{name} is empty after the 5% edge exclusion")

    values = np.asarray(thicknesses, dtype=np.float32)
    area_px = int(mask.sum())
    width_px = ranges[3] - ranges[2] + 1
    if signal_rgb is None:
        mean_r = mean_g = mean_b = mean_gray = float("nan")
    else:
        zone_rgb = signal_rgb[mask]
        mean_r, mean_g, mean_b = (float(v) for v in zone_rgb.mean(axis=0))
        mean_gray = float((0.299 * zone_rgb[:, 0] + 0.587 * zone_rgb[:, 1] + 0.114 * zone_rgb[:, 2]).mean())

    mean_px = float(values.mean())
    median_px = float(np.median(values))
    std_px = float(values.std(ddof=0))
    min_px = float(values.min())
    max_px = float(values.max())
    return ZoneResult(
        name=name,
        boundary_1=boundaries[0], boundary_2=boundaries[1], mask=mask,
        original_x_start=ranges[0], original_x_end=ranges[1],
        x_start=ranges[2], x_end=ranges[3],
        thickness_x=np.asarray(x_values, dtype=np.int32),
        thickness_top=np.asarray(tops, dtype=np.int32),
        thickness_bottom=np.asarray(bottoms, dtype=np.int32),
        thicknesses_px=values,
        area_px=area_px, area_um2=area_px * pixel_size_um**2,
        width_px=width_px, width_um=width_px * pixel_size_um,
        mean_thickness_px=mean_px, mean_thickness_um=mean_px * pixel_size_um,
        median_thickness_px=median_px, median_thickness_um=median_px * pixel_size_um,
        std_thickness_px=std_px, std_thickness_um=std_px * pixel_size_um,
        min_thickness_px=min_px, min_thickness_um=min_px * pixel_size_um,
        max_thickness_px=max_px, max_thickness_um=max_px * pixel_size_um,
        mean_signal_gray=mean_gray, mean_signal_r=mean_r,
        mean_signal_g=mean_g, mean_signal_b=mean_b,
    )


def analyze_zones(
    prediction_dir: str | Path,
    signal_path: str | Path | None,
    *,
    side_exclusion: float = SIDE_EXCLUSION,
    pixel_size_um: float = DEFAULT_PIXEL_SIZE_UM,
) -> OrderedDict[str, ZoneResult]:
    """Create and measure all five zones from a prediction directory."""
    if pixel_size_um <= 0:
        raise ValueError("pixel_size_um must be positive")
    prediction_dir = Path(prediction_dir)
    required = sorted({key for pair in ZONE_DEFINITIONS.values() for key in pair})
    masks = {key: load_binary_mask(prediction_dir / f"{key}.png") for key in required}
    shapes = {mask.shape for mask in masks.values()}
    if len(shapes) != 1:
        raise ValueError("All required prediction masks must have the same size")
    shape = next(iter(shapes))
    signal = _load_signal(signal_path, shape)
    results: OrderedDict[str, ZoneResult] = OrderedDict()
    for name, boundaries in ZONE_DEFINITIONS.items():
        zone, ranges = create_zone_between_masks(
            masks[boundaries[0]], masks[boundaries[1]], side_exclusion=side_exclusion
        )
        results[name] = _measure_zone(name, boundaries, zone, ranges, signal, pixel_size_um)
    return results


def export_zone_csv(results: Mapping[str, ZoneResult], destination: str | Path) -> Path:
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    rows = [result.csv_row() for result in results.values()]
    if not rows:
        raise ValueError("There are no zone results to export")
    with destination.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return destination


def save_zone_masks(results: Mapping[str, ZoneResult], output_dir: str | Path) -> list[Path]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for name, result in results.items():
        path = output_dir / ZONE_FILENAMES[name]
        Image.fromarray(result.mask.astype(np.uint8) * 255, mode="L").save(path)
        paths.append(path)
    return paths
