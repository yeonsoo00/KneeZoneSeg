"""Contour-based visualizations for five-zone growth-plate analysis."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage

from zone_metrics import ZoneResult, save_zone_masks


ZONE_COLORS = {
    "Germinal": (138, 43, 226),
    "Columnar": (230, 45, 55),
    "Pre-hypertrophic": (20, 180, 70),
    "Hypertrophic": (40, 100, 235),
    "GP": (255, 165, 0),
}

INDIVIDUAL_CONTOUR_WIDTH = 2.0 * 0.70
CONTOUR_SUPERSAMPLE = 5


def _load_background(signal_path: str | Path, shape: tuple[int, int]) -> np.ndarray:
    height, width = shape
    image = (
        Image.open(signal_path)
        .convert("RGB")
        .resize((width, height), Image.Resampling.BILINEAR)
    )
    array = np.asarray(image, dtype=np.float32)
    low = float(array.min())
    high = float(np.percentile(array, 99.8))
    if high <= low:
        high = float(array.max())
    if high <= low:
        return np.clip(array, 0, 255).astype(np.uint8)
    return (np.clip((array - low) / (high - low), 0, 1) * 255).astype(np.uint8)


def _binary_erosion_3x3(mask: np.ndarray) -> np.ndarray:
    padded = np.pad(mask.astype(bool), 1, mode="constant", constant_values=False)
    neighborhoods = [
        padded[dy : dy + mask.shape[0], dx : dx + mask.shape[1]]
        for dy in range(3)
        for dx in range(3)
    ]
    return np.logical_and.reduce(neighborhoods)


def _contour(mask: np.ndarray, thickness: int = 1) -> np.ndarray:
    boundary = mask.astype(bool) & ~_binary_erosion_3x3(mask)
    for _ in range(max(0, thickness - 1)):
        padded = np.pad(boundary, 1, mode="constant")
        boundary = np.logical_or.reduce(
            [
                padded[dy : dy + mask.shape[0], dx : dx + mask.shape[1]]
                for dy in range(3)
                for dx in range(3)
            ]
        )
    return boundary


def _blend(
    canvas: np.ndarray, mask: np.ndarray, color: tuple[int, int, int], alpha: float
) -> None:
    if mask.any():
        canvas[mask] = (
            (1.0 - alpha) * canvas[mask].astype(np.float32)
            + alpha * np.asarray(color, dtype=np.float32)
        ).astype(np.uint8)


def _draw_antialiased_contour(
    canvas: np.ndarray,
    mask: np.ndarray,
    color: tuple[int, int, int],
    width: float,
) -> None:
    """Draw a subpixel-width contour by rendering large and downsampling."""
    height, image_width = mask.shape
    if not mask.any():
        return
    scale = CONTOUR_SUPERSAMPLE
    mask_large = np.repeat(np.repeat(mask.astype(bool), scale, axis=0), scale, axis=1)
    thickness = max(1, round(width * scale))
    outside = thickness // 2
    inside = thickness - outside
    structure = np.ones((3, 3), dtype=bool)
    outer = (
        ndimage.binary_dilation(
            mask_large,
            structure=structure,
            iterations=outside,
        )
        if outside
        else mask_large
    )
    inner = (
        ndimage.binary_erosion(
            mask_large,
            structure=structure,
            iterations=inside,
            border_value=0,
        )
        if inside
        else mask_large
    )
    alpha_large = (outer & ~inner).astype(np.uint8) * 255
    alpha = (
        np.asarray(
            Image.fromarray(alpha_large).resize(
                (image_width, height), Image.Resampling.LANCZOS
            ),
            dtype=np.float32,
        )
        / 255.0
    )
    alpha = alpha[..., None]
    canvas[:] = np.clip(
        canvas.astype(np.float32) * (1.0 - alpha)
        + np.asarray(color, dtype=np.float32) * alpha,
        0,
        255,
    ).astype(np.uint8)


def render_zone_measurement(
    signal_path: str | Path,
    result: ZoneResult,
) -> Image.Image:
    """Render one zone with a compact red contour and summary annotation."""
    canvas = _load_background(signal_path, result.mask.shape)
    color = ZONE_COLORS[result.name]
    _blend(canvas, result.mask, color, 0.28)
    _draw_antialiased_contour(
        canvas,
        result.mask,
        (255, 35, 35),
        INDIVIDUAL_CONTOUR_WIDTH,
    )
    image = Image.fromarray(canvas, mode="RGB")
    draw = ImageDraw.Draw(image)
    text = (
        f"{result.name} ({result.boundary_1}-{result.boundary_2})\n"
        f"central 90%; area {result.area_px:,} px2\n"
        f"thickness mean {result.mean_thickness_px:.1f} px; "
        f"median {result.median_thickness_px:.1f} px"
    )
    box = draw.textbbox((0, 0), text)
    draw.rectangle((5, 5, box[2] + 15, box[3] + 15), fill=(255, 255, 255))
    draw.multiline_text((10, 10), text, fill=(0, 0, 0), spacing=3)
    return image


def render_all_zones(
    signal_path: str | Path,
    results: Mapping[str, ZoneResult],
) -> Image.Image:
    """Render the total five-zone visualization with colored contour outlines."""
    if not results:
        raise ValueError("There are no zone results to visualize")
    shape = next(iter(results.values())).mask.shape
    canvas = _load_background(signal_path, shape)

    # GP spans the full plate, so draw it first with a lighter fill. The four
    # biological subzones remain visible above it.
    order = [
        name
        for name in ("GP", "Germinal", "Columnar", "Pre-hypertrophic", "Hypertrophic")
        if name in results
    ]
    for name in order:
        result = results[name]
        alpha = 0.08 if name == "GP" else 0.28
        _blend(canvas, result.mask, ZONE_COLORS[name], alpha)
    for name in order:
        canvas[_contour(results[name].mask, 2)] = ZONE_COLORS[name]

    legend_width = 190
    height, width = shape
    total = Image.new("RGB", (width + legend_width, height), "white")
    total.paste(Image.fromarray(canvas, mode="RGB"), (0, 0))
    draw = ImageDraw.Draw(total)
    draw.text((width + 14, 18), "Growth-plate zones", fill=(0, 0, 0))
    y = 50
    for name in ("Germinal", "Columnar", "Pre-hypertrophic", "Hypertrophic", "GP"):
        if name not in results:
            continue
        color = ZONE_COLORS[name]
        draw.rectangle((width + 14, y, width + 34, y + 14), fill=color, outline=color)
        draw.text((width + 42, y), name, fill=(0, 0, 0))
        y += 28
    draw.text(
        (width + 14, y + 12),
        "5% removed from\neach horizontal edge",
        fill=(55, 55, 55),
        spacing=3,
    )
    return total


def save_zone_visualizations(
    signal_path: str | Path,
    results: Mapping[str, ZoneResult],
    output_dir: str | Path,
) -> list[Path]:
    """Save zone masks, individual measurements, and total contour view."""
    output_dir = Path(output_dir)
    mask_dir = output_dir / "masks"
    visualization_dir = output_dir / "visualizations"
    visualization_dir.mkdir(parents=True, exist_ok=True)
    paths = save_zone_masks(results, mask_dir)
    for name, result in results.items():
        filename = name.replace("-", "_") + "_measurement.png"
        path = visualization_dir / filename
        render_zone_measurement(signal_path, result).save(path)
        paths.append(path)
    total_path = visualization_dir / "All_zones_contour_overlay.png"
    render_all_zones(signal_path, results).save(total_path)
    paths.append(total_path)
    return paths
