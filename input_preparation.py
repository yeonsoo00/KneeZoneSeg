"""Normalize recursive layered documents and raster folders for inference."""

from __future__ import annotations

import json
import re
from pathlib import Path

from PIL import Image, ImageOps

from common_crop_psd import format_psd_dataset
from data import SIGNAL_FILES
from format_layered_psd import STAIN_BY_NUMBER, find_psd_files, layered_sample_name


RASTER_SUFFIXES = frozenset({".png", ".jpg", ".jpeg"})
OPTIONAL_STAINS = frozenset({"edu"})
STAIN_BY_PREFIX = {str(number): stain for number, stain in STAIN_BY_NUMBER.items()}


def _is_below(path: Path, parent: Path) -> bool:
    try:
        path.resolve().relative_to(parent.resolve())
        return True
    except ValueError:
        return False


def _stain_match(path: Path) -> tuple[str, int] | None:
    """Return the stain and preference rank for a supported raster filename."""
    if path.suffix.casefold() not in RASTER_SUFFIXES:
        return None
    stem = path.stem.casefold().strip()
    numbered = re.match(r"^([1-9])(?:[_\-\s]|$)", stem)
    if numbered:
        return STAIN_BY_PREFIX[numbered.group(1)], 0
    for stain, names in SIGNAL_FILES.items():
        for rank, filename in enumerate(names, start=1):
            if stem == Path(filename).stem.casefold():
                return stain, rank
    return None


def signal_images_in(directory: Path) -> dict[str, Path]:
    """Find the preferred file for every recognized stain in one directory."""
    candidates: dict[str, tuple[int, Path]] = {}
    try:
        children = sorted(directory.iterdir(), key=lambda path: path.name.casefold())
    except OSError:
        return {}
    for path in children:
        if not path.is_file():
            continue
        match = _stain_match(path)
        if match is None:
            continue
        stain, rank = match
        previous = candidates.get(stain)
        if previous is None or (rank, path.name.casefold()) < (
            previous[0],
            previous[1].name.casefold(),
        ):
            candidates[stain] = (rank, path)
    return {stain: value[1] for stain, value in candidates.items()}


def find_raster_sample_directories(
    source: str | Path,
    *,
    exclude_roots: tuple[Path, ...] = (),
) -> list[tuple[Path, dict[str, Path]]]:
    """Find recursive image folders containing all required stains except EdU."""
    source = Path(source).expanduser()
    if not source.is_dir():
        return []
    required = set(SIGNAL_FILES) - set(OPTIONAL_STAINS)
    directories = [source, *(path for path in source.rglob("*") if path.is_dir())]
    found: list[tuple[Path, dict[str, Path]]] = []
    for directory in directories:
        if any(_is_below(directory, excluded) for excluded in exclude_roots):
            continue
        images = signal_images_in(directory)
        if required.issubset(images):
            found.append((directory, images))
    return found


def raster_sample_name(directory: Path, dataset_source: Path) -> str:
    """Return a recursive sample path for a raster-image directory."""
    relative = directory.relative_to(dataset_source)
    return directory.name if relative == Path(".") else relative.as_posix()


def format_raster_samples(
    samples: list[tuple[Path, dict[str, Path]]],
    dataset_source: Path,
    output_root: Path,
    *,
    overwrite: bool = False,
) -> list[str]:
    """Convert JPG/PNG sample folders to the standard nine-PNG layout."""
    formatted: list[str] = []
    for source_dir, images in samples:
        sample = raster_sample_name(source_dir, dataset_source)
        destination_dir = output_root / sample
        destination_dir.mkdir(parents=True, exist_ok=True)
        source_map: dict[str, dict[str, int | str] | None] = {}
        with Image.open(next(iter(images.values()))) as opened:
            reference_size = ImageOps.exif_transpose(opened).size
        for stain in SIGNAL_FILES:
            source_image = images.get(stain)
            if source_image is None:
                source_map[stain] = None
            else:
                stat = source_image.stat()
                source_map[stain] = {
                    "path": str(source_image.resolve()),
                    "size_bytes": stat.st_size,
                    "mtime_ns": stat.st_mtime_ns,
                }
        metadata = {
            "source_directory": str(source_dir.resolve()),
            "source_images": source_map,
        }
        metadata_path = destination_dir / "input_metadata.json"
        metadata_matches = False
        if metadata_path.is_file():
            try:
                metadata_matches = (
                    json.loads(metadata_path.read_text(encoding="utf-8")) == metadata
                )
            except (OSError, json.JSONDecodeError):
                pass
        for stain in SIGNAL_FILES:
            destination = destination_dir / f"{stain}.png"
            source_image = images.get(stain)
            if destination.is_file() and not overwrite and metadata_matches:
                continue
            if source_image is None:
                image = Image.new("RGB", reference_size, "black")
            else:
                with Image.open(source_image) as opened:
                    image = ImageOps.exif_transpose(opened).convert("RGB")
            image.save(destination, format="PNG")
        metadata_path.write_text(
            json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
        )
        formatted.append(sample)
        print(f"  {sample}: formatted JPG/PNG signal folder")
    return formatted


def prepare_input_dataset(
    source: str | Path,
    output_root: str | Path,
    *,
    overwrite: bool = False,
) -> Path:
    """Prepare all recursive PSD/PSB/PXD files and JPG/PNG sample folders."""
    source = Path(source).expanduser()
    output_root = Path(output_root).expanduser()
    if not source.exists():
        raise FileNotFoundError(f"Input does not exist: {source}")
    if source.is_file() and source.suffix.casefold() not in {".psd", ".psb", ".pxd"}:
        raise ValueError("A single input file must be PSD, PSB, or PXD")

    layered_files = find_psd_files(source)
    raster_samples = find_raster_sample_directories(
        source,
        exclude_roots=(output_root,),
    )
    if not layered_files and not raster_samples:
        raise FileNotFoundError(
            "No layered PSD/PSB/PXD files or JPG/PNG sample folders were found. "
            "Each image folder must contain Mineral, AC, Calcein, TRAP, DAPI, "
            "AP, CFO, and SFO; EdU is optional."
        )

    dataset_root = source if source.is_dir() else source.parent
    layered_names = {
        layered_sample_name(path, source).casefold() for path in layered_files
    }
    raster_names = {
        raster_sample_name(path, dataset_root).casefold() for path, _ in raster_samples
    }
    collisions = sorted(layered_names & raster_names)
    if collisions:
        raise ValueError(
            "Layered documents and image folders resolve to the same sample path(s): "
            + ", ".join(collisions)
        )

    output_root.mkdir(parents=True, exist_ok=True)
    if layered_files:
        format_psd_dataset(source, output_root, overwrite=overwrite)
    if raster_samples:
        format_raster_samples(
            raster_samples,
            dataset_root,
            output_root,
            overwrite=overwrite,
        )
    print(
        f"Prepared {len(layered_files) + len(raster_samples)} sample(s) in {output_root}"
    )
    return output_root
