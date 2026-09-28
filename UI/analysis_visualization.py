"""Background-aware visualizations for zone and positive-band analysis."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image


def _normalized_signal(path: Path, size: tuple[int, int]) -> np.ndarray:
    image = Image.open(path).convert("RGB").resize(size, Image.Resampling.BILINEAR)
    array = np.asarray(image, dtype=np.float32)
    low = float(array.min())
    high = float(np.percentile(array, 99.8))
    # Sparse fluorescence images can have a zero 99th percentile. Falling back
    # to the true maximum prevents a valid background from becoming all black.
    if high <= low:
        high = float(array.max())
    if high <= low:
        return np.clip(array / 255.0, 0.0, 1.0)
    return np.clip((array - low) / (high - low), 0.0, 1.0)


def load_background(
    signal_dir: str | Path,
    selected_signals: Iterable[str],
    size: tuple[int, int],
) -> np.ndarray:
    """Load and average selected signal images, with mineral fallback."""
    signal_dir = Path(signal_dir)
    paths = [signal_dir / f"{name}.png" for name in selected_signals]
    paths = [path for path in paths if path.is_file()]
    mineral = signal_dir / "mineral.png"
    if not paths and mineral.is_file():
        paths = [mineral]
    if not paths:
        raise FileNotFoundError(
            f"No selected background signal images were found in {signal_dir}"
        )
    return np.mean([_normalized_signal(path, size) for path in paths], axis=0)


def render_background_and_masks(
    signal_dir: str | Path,
    selected_signals: Iterable[str],
    first_mask: np.ndarray,
    second_mask: np.ndarray,
    positive_band: np.ndarray,
) -> tuple[Image.Image, Image.Image]:
    """Return a bright mask overlay and a background multiplied by the band."""
    height, width = first_mask.shape
    background = load_background(signal_dir, selected_signals, (width, height))

    # Main overlay: preserve the signal background and brighten both predicted
    # zones toward white. The positive band is brightest.
    overlay = background.copy()
    zone_mask = first_mask | second_mask
    overlay[zone_mask] = 0.30 * overlay[zone_mask] + 0.70
    overlay[positive_band] = 0.12 * overlay[positive_band] + 0.88

    # Second panel: literal image multiplication. Only positive-band pixels keep
    # their background signal intensity; everything outside is black.
    multiplied = background * positive_band[..., None].astype(np.float32)
    return (
        Image.fromarray((np.clip(overlay, 0, 1) * 255).astype(np.uint8), mode="RGB"),
        Image.fromarray((np.clip(multiplied, 0, 1) * 255).astype(np.uint8), mode="RGB"),
    )
