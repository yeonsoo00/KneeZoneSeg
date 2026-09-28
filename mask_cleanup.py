"""Post-processing helpers for predicted binary masks."""

from __future__ import annotations

import cv2
import numpy as np
from scipy import ndimage


def largest_connected_component(mask: np.ndarray) -> np.ndarray:
    """Return only the largest 8-connected foreground component."""
    binary = (np.asarray(mask) > 0).astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        binary, connectivity=8
    )
    if count <= 1:
        return np.zeros_like(binary, dtype=np.uint8)
    largest_label = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return (labels == largest_label).astype(np.uint8)


def finalize_binary_mask(mask: np.ndarray) -> np.ndarray:
    """Apply final thresholded-mask postprocessing for evaluation and inference."""
    largest = largest_connected_component(mask)
    return ndimage.binary_fill_holes(largest > 0).astype(np.uint8)
