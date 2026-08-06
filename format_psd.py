"""Convert numbered stain PSD files into model-ready inference folders."""

from __future__ import annotations

import argparse
import re
from collections import defaultdict
from pathlib import Path

STAIN_BY_NUMBER = {
    1: "mineral", 2: "ac", 3: "calcein", 4: "trap", 5: "dapi",
    6: "ap", 7: "edu", 8: "cfo", 9: "sfo",
}
PSD_NAME_PATTERN = re.compile(
    r"^(?P<number>[1-9])_(?P<sample>.+?)\s*\((?P<stain>[^()]*)\)\s*$",
    re.IGNORECASE,
)


def find_psd_files(root: str | Path) -> list[Path]:
    """Find PSD/PSB files recursively and case-insensitively."""
    root = Path(root)
    if not root.is_dir():
        raise FileNotFoundError(f"PSD input directory does not exist: {root}")
    return sorted(
        path for path in root.rglob("*")
        if path.is_file() and path.suffix.casefold() in {".psd", ".psb"}
    )


def _parse_psd_name(path: Path) -> tuple[str, int]:
    match = PSD_NAME_PATTERN.fullmatch(path.stem.strip())
    if match is None:
        raise ValueError(
            f"Unrecognized PSD filename: {path.name!r}. Expected, for example, "
            "'6_CCC_K10_hK_FL1_M_s2c3 (AP).psd'."
        )
    number = int(match.group("number"))
    sample = match.group("sample").strip()
    expected = STAIN_BY_NUMBER[number]
    written = re.sub(r"[^a-z0-9]", "", match.group("stain").casefold())
    if written != expected:
        raise ValueError(
            f"Stain mismatch in {path.name!r}: prefix {number} requires "
            f"({expected.upper()}), not ({match.group('stain')})."
        )
    return sample, number


def format_psd_dataset(
    psd_root: str | Path,
    output_root: str | Path,
    *,
    overwrite: bool = False,
) -> Path:
    """Render PSD composites and group them into one folder per sample."""
    try:
        from psd_tools import PSDImage
    except ImportError as exc:
        raise RuntimeError(
            "PSD input requires psd-tools. Install it with: pip install psd-tools"
        ) from exc

    files = find_psd_files(psd_root)
    if not files:
        raise FileNotFoundError(f"No .psd or .psb files found under: {psd_root}")

    grouped = defaultdict(dict)
    for path in files:
        sample, number = _parse_psd_name(path)
        if number in grouped[sample]:
            raise ValueError(
                f"Duplicate stain {number} for sample {sample!r}: "
                f"{grouped[sample][number]} and {path}"
            )
        grouped[sample][number] = path

    for sample, numbered_files in grouped.items():
        missing = sorted(set(STAIN_BY_NUMBER) - set(numbered_files))
        if missing:
            names = ", ".join(f"{n} ({STAIN_BY_NUMBER[n]})" for n in missing)
            raise ValueError(f"Sample {sample!r} is missing required stains: {names}")

    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    for sample, numbered_files in sorted(grouped.items()):
        sample_dir = output_root / sample
        sample_dir.mkdir(parents=True, exist_ok=True)
        for number, source in sorted(numbered_files.items()):
            destination = sample_dir / f"{STAIN_BY_NUMBER[number]}.png"
            if destination.exists() and not overwrite:
                continue
            composite = PSDImage.open(source).composite()
            if composite is None:
                raise ValueError(f"PSD has no renderable composite: {source}")
            # Preserve RGB because SFO HSV preprocessing needs its colour data.
            composite.convert("RGB").save(destination, format="PNG")

    print(f"Formatted {len(grouped)} sample(s) from {len(files)} PSD file(s) in {output_root}")
    return output_root


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Format numbered stain PSDs for inference")
    parser.add_argument("psd_root", help="Directory containing numbered PSD files")
    parser.add_argument("output_root", help="Directory for model-ready sample folders")
    parser.add_argument("--overwrite", action="store_true", help="Re-render existing PNG files")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    format_psd_dataset(args.psd_root, args.output_root, overwrite=args.overwrite)
