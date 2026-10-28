"""Post-processing helpers for predicted binary masks."""

from __future__ import annotations

import numpy as np
from scipy import ndimage


def largest_connected_component(mask: np.ndarray) -> np.ndarray:
    """Return only the largest 8-connected foreground component."""
    binary = (np.asarray(mask) > 0).astype(np.uint8)
    labels, count = ndimage.label(
        binary,
        structure=np.ones((3, 3), dtype=np.uint8),
    )
    if count == 0:
        return np.zeros_like(binary, dtype=np.uint8)
    sizes = np.bincount(labels.ravel())
    sizes[0] = 0
    largest_label = int(np.argmax(sizes))
    return (labels == largest_label).astype(np.uint8)


def finalize_binary_mask(mask: np.ndarray) -> np.ndarray:
    """Apply final thresholded-mask postprocessing for evaluation and inference."""
    largest = largest_connected_component(mask)
    return ndimage.binary_fill_holes(largest > 0).astype(np.uint8)
