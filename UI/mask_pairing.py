"""Selection helpers for anatomically compatible predicted-mask pairs."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image


STAINS = ("mineral", "ac", "calcein", "trap", "dapi", "ap", "edu", "cfo", "sfo")
STAIN_TO_DIGIT = {name: str(index + 1) for index, name in enumerate(STAINS)}
UPPER_SUFFIXES = set("aceg")
LOWER_SUFFIXES = set("bdfh")


def available_mask_keys(prediction_dir: str | Path, stain: str) -> list[str]:
    """Return available predicted mask keys for one stain."""
    digit = STAIN_TO_DIGIT[stain]
    return [
        path.stem.lower()
        for path in sorted(Path(prediction_dir).glob(f"{digit}[a-h].png"))
    ]


def build_mask_pairs(
    prediction_dir: str | Path,
    first_stain: str,
    second_stain: str,
) -> list[tuple[str, str, str]]:
    """Build matching or upper/upper and lower/lower mask combinations."""
    first = available_mask_keys(prediction_dir, first_stain)
    second = available_mask_keys(prediction_dir, second_stain)
    if not first or not second:
        return []

    # Four-zone stains use corresponding zones: a-a, b-b, c-c, d-d.
    if len(first) <= 4 and len(second) <= 4:
        second_by_suffix = {key[1]: key for key in second}
        return [
            (key, second_by_suffix[key[1]], "matching")
            for key in first
            if key[1] in second_by_suffix
        ]

    # Larger sets can have several upper and lower structures. Expose every
    # parity-compatible combination so the user can choose the intended pair.
    pairs: list[tuple[str, str, str]] = []
    for label, suffixes in (("upper", UPPER_SUFFIXES), ("lower", LOWER_SUFFIXES)):
        first_group = [key for key in first if key[1] in suffixes]
        second_group = [key for key in second if key[1] in suffixes]
        pairs.extend((left, right, label) for left in first_group for right in second_group)
    return pairs


def load_mask_key(prediction_dir: str | Path, mask_key: str) -> np.ndarray:
    """Load one selected predicted mask as a Boolean array."""
    path = Path(prediction_dir) / f"{mask_key.lower()}.png"
    if not path.is_file():
        raise FileNotFoundError(f"Prediction mask not found: {path}")
    return np.asarray(Image.open(path).convert("L")) > 127
