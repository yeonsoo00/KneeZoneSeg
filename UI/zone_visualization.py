"""Contour-based visualizations for five-zone growth-plate analysis."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Mapping

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy import ndimage

from zone_metrics import ZoneResult, save_zone_masks


ZONE_COLORS = {
    "Germinal": (138, 43, 226),
    "Columnar": (230, 45, 55),
    "Pre-hypertrophic": (20, 180, 70),
    "Hypertrophic": (40, 100, 235),
    "GP": (255, 165, 0),
}

CONTOUR_WIDTH = 2.0 * 0.70
CONTOUR_SUPERSAMPLE = 5
MAX_VISUALIZATION_DIMENSION = 4096
APP_PREVIEW_MAX_DIMENSION = 440

# Each entry is (filename/preview label, stains to normalize and average).
# Keeping the recipes here ensures the GUI preview and saved exports use the
# same biologically relevant backgrounds.
ZONE_BACKGROUND_RECIPES = {
    "GP": (("SFO_CFO", ("sfo", "cfo")),),
    "Germinal": (
        ("SFO_CFO", ("sfo", "cfo")),
        ("AP_DAPI", ("ap", "dapi")),
    ),
    "Columnar": (("AP_DAPI", ("ap", "dapi")),),
    "Pre-hypertrophic": (
        ("AP_DAPI", ("ap", "dapi")),
        ("Mineral", ("mineral",)),
    ),
    "Hypertrophic": (("Mineral_SFO", ("mineral", "sfo")),),
}


def _signal_paths(
    signal_paths: str | Path | Iterable[str | Path],
) -> tuple[Path, ...]:
    if isinstance(signal_paths, (str, Path)):
        paths = (Path(signal_paths),)
    else:
        paths = tuple(Path(path) for path in signal_paths)
    if not paths:
        raise ValueError("At least one background signal is required")
    missing = [path for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "Required visualization background was not found: "
            + ", ".join(str(path) for path in missing)
        )
    return paths


def _visualization_shape(
    signal_paths: str | Path | Iterable[str | Path],
    max_dimension: int,
) -> tuple[int, int]:
    """Return the first background's shape, capped to the requested dimension."""
    if max_dimension < 1:
        raise ValueError("max_dimension must be positive")
    first = _signal_paths(signal_paths)[0]
    with Image.open(first) as image:
        width, height = image.size
    scale = min(1.0, max_dimension / max(width, height))
    return max(1, round(height * scale)), max(1, round(width * scale))


def _load_one_background(signal_path: str | Path, shape: tuple[int, int]) -> np.ndarray:
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


def _load_background(
    signal_paths: str | Path | Iterable[str | Path],
    shape: tuple[int, int],
) -> np.ndarray:
    """Normalize each requested signal independently and average its RGB image."""
    paths = _signal_paths(signal_paths)
    normalized = [
        _load_one_background(path, shape).astype(np.float32) for path in paths
    ]
    return np.rint(np.mean(normalized, axis=0)).astype(np.uint8)


def background_paths(
    signal_dir: str | Path,
    stains: Iterable[str],
) -> tuple[Path, ...]:
    """Resolve a stain recipe to the formatted input PNG files."""
    signal_dir = Path(signal_dir)
    return tuple(signal_dir / f"{stain.casefold()}.png" for stain in stains)


def _resize_mask(mask: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    """Resize a binary mask without creating interpolated label values."""
    height, width = shape
    if mask.shape == shape:
        return mask.astype(bool, copy=False)
    image = Image.fromarray(mask.astype(np.uint8) * 255, mode="L")
    return (
        np.asarray(
            image.resize((width, height), Image.Resampling.NEAREST), dtype=np.uint8
        )
        > 0
    )


def _resolution_scale(
    source_shape: tuple[int, int], target_shape: tuple[int, int]
) -> float:
    """Geometric pixel scale used to preserve relative line and text sizes."""
    source_h, source_w = source_shape
    target_h, target_w = target_shape
    return max(1.0, ((target_h / source_h) * (target_w / source_w)) ** 0.5)


def _default_font(size: int) -> ImageFont.ImageFont:
    try:
        return ImageFont.load_default(size=size)
    except TypeError:  # Pillow < 10 compatibility for source installations.
        return ImageFont.load_default()


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
    scale = (
        CONTOUR_SUPERSAMPLE if width < 2.0 and max(height, image_width) <= 1024 else 1
    )
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
    signal_path: str | Path | Iterable[str | Path],
    result: ZoneResult,
    *,
    max_dimension: int = MAX_VISUALIZATION_DIMENSION,
) -> Image.Image:
    """Render one zone at the background's native/high-resolution dimensions."""
    target_shape = _visualization_shape(signal_path, max_dimension)
    mask = _resize_mask(result.mask, target_shape)
    resolution_scale = _resolution_scale(result.mask.shape, target_shape)
    display_scale = (
        1.0 if max_dimension <= APP_PREVIEW_MAX_DIMENSION else resolution_scale
    )
    canvas = _load_background(signal_path, target_shape)
    color = ZONE_COLORS[result.name]
    _blend(canvas, mask, color, 0.28)
    _draw_antialiased_contour(
        canvas,
        mask,
        (255, 35, 35),
        CONTOUR_WIDTH * display_scale,
    )
    image = Image.fromarray(canvas, mode="RGB")
    draw = ImageDraw.Draw(image)
    annotation_scale = min(display_scale, 4.0)
    font = _default_font(max(10, round(10 * annotation_scale)))
    margin = max(5, round(5 * annotation_scale))
    inset = max(5, round(5 * annotation_scale))
    text = (
        f"{result.name} ({result.boundary_1}-{result.boundary_2})\n"
        f"central 90%; area {result.area_px:,} px2\n"
        f"thickness mean {result.mean_thickness_px:.1f} px; "
        f"median {result.median_thickness_px:.1f} px"
    )
    box = draw.multiline_textbbox((0, 0), text, font=font, spacing=3)
    draw.rectangle(
        (margin, margin, box[2] + margin + 2 * inset, box[3] + margin + 2 * inset),
        fill=(255, 255, 255),
    )
    draw.multiline_text(
        (margin + inset, margin + inset),
        text,
        fill=(0, 0, 0),
        font=font,
        spacing=3,
    )
    return image


def render_all_zones(
    signal_path: str | Path | Iterable[str | Path],
    results: Mapping[str, ZoneResult],
    *,
    max_dimension: int = MAX_VISUALIZATION_DIMENSION,
) -> Image.Image:
    """Render the total five-zone visualization with colored contour outlines."""
    if not results:
        raise ValueError("There are no zone results to visualize")
    source_shape = next(iter(results.values())).mask.shape
    shape = _visualization_shape(signal_path, max_dimension)
    resolution_scale = _resolution_scale(source_shape, shape)
    display_scale = (
        1.0 if max_dimension <= APP_PREVIEW_MAX_DIMENSION else resolution_scale
    )
    canvas = _load_background(signal_path, shape)

    # GP spans the full plate, so draw it first with a lighter fill. The four
    # biological subzones remain visible above it.
    order = [
        name
        for name in ("GP", "Germinal", "Columnar", "Pre-hypertrophic", "Hypertrophic")
        if name in results
    ]
    resized_masks = {name: _resize_mask(results[name].mask, shape) for name in order}
    for name in order:
        alpha = 0.08 if name == "GP" else 0.28
        _blend(canvas, resized_masks[name], ZONE_COLORS[name], alpha)
    for name in order:
        _draw_antialiased_contour(
            canvas,
            resized_masks[name],
            ZONE_COLORS[name],
            CONTOUR_WIDTH * display_scale,
        )

    legend_scale = min(display_scale, 4.0)
    legend_width = round(190 * legend_scale)
    height, width = shape
    total = Image.new("RGB", (width + legend_width, height), "white")
    total.paste(Image.fromarray(canvas, mode="RGB"), (0, 0))
    draw = ImageDraw.Draw(total)
    font = _default_font(max(10, round(10 * legend_scale)))
    x_padding = round(14 * legend_scale)
    draw.text(
        (width + x_padding, round(18 * legend_scale)),
        "Growth-plate zones",
        fill=(0, 0, 0),
        font=font,
    )
    y = round(50 * legend_scale)
    for name in ("Germinal", "Columnar", "Pre-hypertrophic", "Hypertrophic", "GP"):
        if name not in results:
            continue
        color = ZONE_COLORS[name]
        swatch_width = round(20 * legend_scale)
        swatch_height = round(14 * legend_scale)
        draw.rectangle(
            (
                width + x_padding,
                y,
                width + x_padding + swatch_width,
                y + swatch_height,
            ),
            fill=color,
            outline=color,
        )
        draw.text(
            (width + round(42 * legend_scale), y),
            name,
            fill=(0, 0, 0),
            font=font,
        )
        y += round(28 * legend_scale)
    draw.text(
        (width + x_padding, y + round(12 * legend_scale)),
        "5% removed from\neach horizontal edge",
        fill=(55, 55, 55),
        font=font,
        spacing=3,
    )
    return total


def save_zone_visualizations(
    signal_source: str | Path,
    results: Mapping[str, ZoneResult],
    output_dir: str | Path,
) -> list[Path]:
    """Save masks and the fixed stain-recipe visualizations for every zone.

    ``signal_source`` may be the formatted sample directory or one signal file
    inside it. Accepting a file retains compatibility with the notebook API.
    """
    signal_source = Path(signal_source)
    signal_dir = signal_source if signal_source.is_dir() else signal_source.parent
    output_dir = Path(output_dir)
    mask_dir = output_dir / "masks"
    visualization_dir = output_dir / "visualizations"
    visualization_dir.mkdir(parents=True, exist_ok=True)
    high_resolution_shape = _visualization_shape(
        signal_dir / "sfo.png", MAX_VISUALIZATION_DIMENSION
    )
    high_resolution_size = (
        high_resolution_shape[1],
        high_resolution_shape[0],
    )
    paths = save_zone_masks(results, mask_dir, size=high_resolution_size)
    for name, recipes in ZONE_BACKGROUND_RECIPES.items():
        if name not in results:
            continue
        zone_name = name.replace("-", "_")
        for background_name, stains in recipes:
            path = visualization_dir / (
                f"{zone_name}_{background_name}_measurement.png"
            )
            render_zone_measurement(
                background_paths(signal_dir, stains), results[name]
            ).save(path)
            paths.append(path)
    total_path = visualization_dir / "All_zones_SFO_contour_overlay.png"
    render_all_zones(signal_dir / "sfo.png", results).save(total_path)
    paths.append(total_path)
    return paths
