"""
Data loading utilities for knee growth plate zone detection project.

This module defines a PyTorch Dataset that loads multi‑stained images and
their corresponding segmentation masks from disk. Each sample consists of
up to nine input stains (Mineral, AC, Calcein, TRAP, DAPI, AP, EdU, CFO,
SFO) and multiple output masks describing the different regions (zones)
present in the growth plate. Some stains may have both a standard and a
"high" version; if available the "high" intensity image is preferred.

Directory layout (as described in the project readme):

Input images live under `image_root/sample_name/` and masks live under
`mask_root/sample_name/`.  Within each sample folder there are PNG files
for each stain and for each ground truth mask.  The dataset attempts to
load a consistent set of signals defined in the `SIGNAL_FILES` mapping
below.  If a file is missing it is replaced with a zero array of the same
size as the first loaded image in the sample.

Ground truth masks are loaded by scanning for all `.png` files in the
mask folder that match the pattern of a number followed by a letter (e.g.
"1a.png", "9d.png").  Each mask becomes one channel in the output tensor.

Note:
    The dataset does not perform any heavy on‑the‑fly augmentation.  To
    add augmentations (e.g., flips, rotations), compose this dataset with
    torchvision transforms in your training script.

Example usage::

    from torch.utils.data import DataLoader
    from knee_zone_detection.data import KneeZoneDataset

    dataset = KneeZoneDataset(
        image_root='/path/to/Image',
        mask_root='/path/to/Mask',
        samples=['sample1', 'sample2'],
        transforms=your_transforms
    )
    loader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=4)

"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple, Set

# ---------------------------------------------------------------------
# Hard-coded valid GT mask keys.  These keys correspond to the expected
# ground truth masks per stain as described in the project requirements.
# The keys are case-insensitive and will be normalised to lower case.
EXPECTED_MASK_KEYS: List[str] = [
    "1a", "1b", "1c", "1d",
    "2a", "2b", "2c", "2d", "2e", "2f",
    "3a", "3b", "3c", "3d",
    "4a", "4b", "4c", "4d",
    "5a", "5b", "5c", "5d",
    "6a", "6b", "6c", "6d", "6e", "6f", "6g", "6h",
    "7a", "7b", "7c", "7d", "7e", "7f",
    "8a", "8b", "8c", "8d", "8e", "8f",
    "9a", "9b", "9c", "9d",
]
EXPECTED_MASK_KEYS_SET: Set[str] = set(EXPECTED_MASK_KEYS)

def extract_gt_key(filename: str) -> Optional[str]:
    """
    Extract one valid GT mask key from a filename in a strict way.

    Handles random upper/lower case and common separators, but ignores
    background or other non-GT files.  Returns the lower-case key if
    matched, otherwise None.

    Examples matched:
        1a.png
        1A.png
        1a_copy.png
        1A-copy.png
        1a something.png

    Examples ignored:
        1.png
        2High.png
        ac_a.png
        3label.png
        8light.png
    """
    stem = Path(filename).stem.lower().strip()
    # normalise separators so we only accept keys as standalone tokens
    norm = re.sub(r"[\s\-]+", "_", stem)
    # match only at the beginning, followed by end or separator
    m = re.match(r"^([1-9][a-h])(?:_|$)", norm)
    if not m:
        return None
    key = m.group(1)
    if key not in EXPECTED_MASK_KEYS_SET:
        return None
    return key

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset

SIGNAL_FILES: Dict[str, Tuple[str, ...]] = {
    # Each entry lists alternative file names in order of preference.  The
    # dataset will use the first file that exists.  Additional file names
    # can be added here if your dataset uses different naming conventions.
    'mineral': ('mineral.png',),
    'ac': ('ac_high.png', 'ac.png'),
    'calcein': ('calcein_high.png', 'calcein.png', 'calcein2.png'),
    'trap': ('trap_high.png', 'trap.png'),
    'dapi': ('dapi_high.png', 'dapi.png'),
    'ap': ('ap_high.png', 'ap.png', 'ap_high2.png'),
    'edu': ('edu_high.png', 'edu.png'),
    'cfo': ('cfo_high.png', 'cfo.png'),
    'sfo': ('sfo_high.png', 'sfo.png'),
}

MASK_PATTERN = re.compile(r"^(\d+[a-zA-Z])\.png$")


class KneeZoneDataset(Dataset):
    """PyTorch Dataset for knee zone segmentation.

    Args:
        image_root (str or Path): Root directory containing input images.
        mask_root (str or Path): Root directory containing ground truth masks.
        samples (List[str]): List of sample names to include.
        transforms (Callable, optional): Optional transform to be applied on
            both image and mask. The callable should accept image (C,H,W)
            and mask (K,H,W) tensors and return the transformed versions.
    """

    def __init__(
        self,
        image_root: str | Path,
        mask_root: str | Path,
        samples: List[str],
        transforms: Optional[Callable[[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]] = None,
        *,
        use_hsv: bool = False,
        hsv_groups: Optional[List[List[int]]] = None,
        sfo_hsv: bool = False,
        sfo_hsv_channels: int = 1,
        target_size: Optional[int] = None,
        target_height: Optional[int] = None,
        target_width: Optional[int] = None,
        selected_keys: Optional[List[str]] = None,
    ) -> None:
        """Initialize the dataset.

        Args:
            image_root: Root directory containing input images.
            mask_root: Root directory containing ground truth masks.
            samples: List of sample names to include.
            transforms: Optional transform applied to image and mask.
            use_hsv: If True, compute a hue channel from the loaded signals and
                append it as an additional channel.  This may be useful when
                combining multiple stains into a color representation to help
                distinguish zones.  See `hsv_groups` for how the hue is
                computed.
            hsv_groups: Optional list of three lists, each specifying indices
                of signals to combine into the R, G, and B components for
                pseudo‑RGB before computing the hue.  Each inner list should
                contain indices referring to the order of signals defined in
                `SIGNAL_FILES`.  The default groups split the signals evenly
                into three groups.  If the number of signals is not divisible
                by three, the remaining signals are appended to the last group.
        """
        self.image_root = Path(image_root)
        self.mask_root = Path(mask_root)
        self.samples = samples
        self.transforms = transforms
        self.use_hsv = use_hsv
        self.hsv_groups = hsv_groups
        self.sfo_hsv = sfo_hsv
        self.sfo_hsv_channels = sfo_hsv_channels

        # Selected mask keys: if provided, only these keys will be loaded and returned.
        # Keys should be specified in a case-insensitive manner.  If None, all expected
        # keys defined in EXPECTED_MASK_KEYS are used.  This allows training separate
        # models for different signal families.
        self.selected_keys: Optional[List[str]] = None
        if selected_keys is not None:
            # Normalise to lower-case and remove duplicates
            sel_set = set(k.lower() for k in selected_keys if isinstance(k, str) and k)
            # Filter the expected mask keys to include only selected keys, preserving order
            self.selected_keys = [k for k in EXPECTED_MASK_KEYS if k in sel_set]

        # Target resize parameters.  If provided, these override the automatic
        # reference shape computed from the first sample.  ``target_size``
        # applies to both height and width (square).  ``target_height`` and
        # ``target_width`` can be used to specify non‑square shapes.  Any
        # specified dimension is rounded down to the nearest multiple of 16.
        self.target_size = target_size
        self.target_height = target_height
        self.target_width = target_width

        # Map each image sample name to one or more corresponding mask directories.
        # Some mask directories include additional suffixes after the first
        # five underscore-separated fields.  We match mask directories by
        # splitting the image sample name by '_' and taking the first five
        # tokens as a base identifier.  All directories in mask_root whose
        # names start with this base identifier (case-insensitive) are
        # collected for averaging across labelers.
        # Example: image sample 'CCC_K10_F1_L1_Alex' matches mask directories
        # like 'CCC_K10_F1_L1_ALEX_oldstack_PIXEL_AlexFixed'.
        self.sample_to_mask_dirs: Dict[str, List[Path]] = {}
        # List mask root directories only once
        if self.mask_root.is_dir():
            mask_dirs = [p for p in self.mask_root.iterdir() if p.is_dir()]
        else:
            mask_dirs = []
        # Precompute mapping for each sample
        for sname in self.samples:
            # derive base id: first five underscore-separated tokens
            tokens = sname.split('_')
            base_tokens = tokens[:5]
            base_id = '_'.join(base_tokens).lower()
            matched = [d for d in mask_dirs if d.name.lower().startswith(base_id)]
            if not matched:
                # Fall back to exact match of sample name (case-insensitive)
                matched = [d for d in mask_dirs if d.name.lower() == sname.lower()]
            self.sample_to_mask_dirs[sname] = matched

        # Determine a reference spatial shape to which all images and masks will
        # be resized.  This ensures that DataLoader can stack tensors across
        # samples without size mismatches.  By default we use the shape of
        # the first sample's loaded signals.  However, if target_size,
        # target_height or target_width are provided, they override the
        # automatically computed shape.  All dimensions are rounded down
        # to the nearest multiple of 16 to align with the UNet architecture.
        self.reference_shape: Optional[Tuple[int, int]] = None
        if self.samples:
            # Determine default reference shape from first sample
            self.reference_shape = None
            first_dir = self.image_root / self.samples[0]
            tmp_signals = self._load_signals(first_dir)
            ref_h, ref_w = tmp_signals.shape[1], tmp_signals.shape[2]
            # Override with user-specified target dimensions if provided
            if self.target_size is not None and self.target_size > 0:
                ref_h = ref_w = self.target_size
            else:
                if self.target_height is not None and self.target_height > 0:
                    ref_h = self.target_height
                if self.target_width is not None and self.target_width > 0:
                    ref_w = self.target_width
            # Adjust reference shape to be divisible by 16
            ref_h = (ref_h // 16) * 16
            ref_w = (ref_w // 16) * 16
            self.reference_shape = (ref_h, ref_w)

        # Determine the canonical order of mask keys to be used.  If specific
        # selected_keys were provided, only those keys are kept (in the order
        # defined by EXPECTED_MASK_KEYS).  Otherwise, use all expected mask keys.
        if self.selected_keys is None:
            self.mask_keys_order: List[str] = EXPECTED_MASK_KEYS.copy()
        else:
            self.mask_keys_order = self.selected_keys.copy()

        # ------------------------------------------------------------------
        # Pre-check for missing ground truth masks
        # For each sample, determine which of the expected mask keys are missing
        # among the available mask files.  Missing keys are recorded and
        # reported.  This helps identify incomplete annotations before
        # training starts.
        self.missing_masks: Dict[str, List[str]] = {}
        for sname, dirs in self.sample_to_mask_dirs.items():
            existing_keys: Set[str] = set()
            for d in dirs:
                if not d.is_dir():
                    continue
                for fpath in d.glob('*.png'):
                    key = extract_gt_key(fpath.name)
                    if key is not None:
                        existing_keys.add(key)
            missing = [k for k in self.mask_keys_order if k not in existing_keys]
            if missing:
                self.missing_masks[sname] = missing
        if self.missing_masks:
            msg_lines = [
                "[KneeZoneDataset] Warning: The following ground truth masks are missing:"
            ]
            for sname, missing in self.missing_masks.items():
                msg_lines.append(f"  Sample '{sname}': missing {', '.join(sorted(missing))}")
            print("\n".join(msg_lines))

    def __len__(self) -> int:
        return len(self.samples)

    @staticmethod
    def _load_image(path: Path) -> np.ndarray:
        """Load an image file as a float32 NumPy array normalised to [0,1]."""
        img = Image.open(path).convert('L')  # convert to grayscale
        arr = np.array(img, dtype=np.float32) / 255.0
        return arr

    def _load_signals(self, sample_path: Path) -> torch.Tensor:
        """Load all signal images for a single sample into a tensor of shape (C,H,W)."""
        channels: List[np.ndarray] = []
        first_shape: Optional[Tuple[int, int]] = None
        for signal, filenames in SIGNAL_FILES.items():
            found = None
            for fname in filenames:
                candidate = sample_path / fname
                if candidate.exists():
                    found = candidate
                    break
            if found is not None:
                arr = self._load_image(found)
            else:
                # If no file found, create a zero array using first known shape
                if first_shape is not None:
                    arr = np.zeros(first_shape, dtype=np.float32)
                else:
                    # If this is the first channel and nothing is loaded yet,
                    # we cannot know the spatial size; in that case create
                    # a dummy 1x1 array.  The transform later can handle it.
                    arr = np.zeros((1, 1), dtype=np.float32)
            # Record shape for subsequent missing channels
            if first_shape is None and arr.size != 0:
                first_shape = arr.shape
            channels.append(arr)
        # If there are size mismatches between channels, resize them to match
        # the first non‑empty channel.  This is a simple nearest neighbour
        # resize to keep this loader lightweight.  For more advanced
        # interpolation use torchvision.transforms within transforms argument.
        if first_shape is None:
            raise RuntimeError(f"No images found in sample {sample_path}")
        H, W = first_shape
        aligned = []
        for arr in channels:
            if arr.size == 0:
                # skip empty array
                aligned.append(np.zeros((H, W), dtype=np.float32))
                continue
            # If the array shape matches the reference shape, append directly
            if arr.shape == (H, W):
                aligned.append(arr)
                continue
            # If the array shape appears transposed (W,H), transpose it
            if arr.shape == (W, H):
                arr = arr.T
            # Resize to reference shape if necessary
            if arr.shape != (H, W):
                arr_img = Image.fromarray((arr * 255).astype(np.uint8))
                arr_resized = arr_img.resize((W, H), Image.BILINEAR)
                arr = np.array(arr_resized, dtype=np.float32) / 255.0
            aligned.append(arr)
        stacked = np.stack(aligned, axis=0)  # (C,H,W)
        # Optionally compute a hue channel from grouped signals
        if self.use_hsv:
            # Determine groupings
            num_signals = len(aligned)
            if self.hsv_groups is None:
                # Split signals evenly into three groups
                group_size = max(1, num_signals // 3)
                g1 = list(range(0, group_size))
                g2 = list(range(group_size, 2 * group_size))
                g3 = list(range(2 * group_size, num_signals))
                groups = [g1, g2, g3]
            else:
                groups = self.hsv_groups
                # Validate groups
                if len(groups) != 3:
                    raise ValueError("hsv_groups must contain exactly three sublists for R, G, B")
            # Compute R, G, B as the mean of grouped signals
            H, W = aligned[0].shape
            r = np.zeros((H, W), dtype=np.float32)
            g = np.zeros((H, W), dtype=np.float32)
            b = np.zeros((H, W), dtype=np.float32)
            def safe_mean(indices: List[int]) -> np.ndarray:
                if not indices:
                    return np.zeros((H, W), dtype=np.float32)
                channels = [aligned[idx] for idx in indices if idx < len(aligned)]
                if not channels:
                    return np.zeros((H, W), dtype=np.float32)
                return np.mean(np.stack(channels, axis=0), axis=0)
            r = safe_mean(groups[0])
            g = safe_mean(groups[1])
            b = safe_mean(groups[2])
            # Normalize R,G,B to [0,1]
            rgb = np.stack([r, g, b], axis=2)  # (H,W,3)
            # Convert to HSV and extract hue
            # Avoid division by zero
            maxc = rgb.max(axis=2)
            minc = rgb.min(axis=2)
            delta = maxc - minc
            # Hue calculation
            hue = np.zeros((H, W), dtype=np.float32)
            # Where delta != 0
            mask = delta > 1e-5
            # Indices where max is R, G, B
            r_mask = (maxc == rgb[:, :, 0]) & mask
            g_mask = (maxc == rgb[:, :, 1]) & mask
            b_mask = (maxc == rgb[:, :, 2]) & mask
            # Compute hue according to HSV definitions
            hue[r_mask] = ((rgb[:, :, 1] - rgb[:, :, 2])[r_mask] / delta[r_mask]) % 6
            hue[g_mask] = ((rgb[:, :, 2] - rgb[:, :, 0])[g_mask] / delta[g_mask]) + 2
            hue[b_mask] = ((rgb[:, :, 0] - rgb[:, :, 1])[b_mask] / delta[b_mask]) + 4
            hue = hue / 6.0  # Scale to [0,1]
            hue[~mask] = 0.0
            # Append hue channel to stacked
            stacked = np.concatenate([stacked, hue[None, :, :]], axis=0)
        # Optionally convert SFO channel into HSV and append hue/saturation
        if self.sfo_hsv:
            # Locate SFO file
            sfo_path = None
            for fname in SIGNAL_FILES['sfo']:
                candidate = sample_path / fname
                if candidate.exists():
                    sfo_path = candidate
                    break
            if sfo_path is not None:
                # Load SFO as RGB
                img_rgb = Image.open(sfo_path).convert('RGB')
                arr_rgb = np.array(img_rgb, dtype=np.float32) / 255.0  # (H,W,3)
                # Compute hue and saturation
                maxc = arr_rgb.max(axis=2)
                minc = arr_rgb.min(axis=2)
                delta = maxc - minc
                hue = np.zeros(arr_rgb.shape[:2], dtype=np.float32)
                sat = np.zeros(arr_rgb.shape[:2], dtype=np.float32)
                mask_nonzero = delta > 1e-5
                r = arr_rgb[:, :, 0]
                g = arr_rgb[:, :, 1]
                b = arr_rgb[:, :, 2]
                r_mask = (maxc == r) & mask_nonzero
                g_mask = (maxc == g) & mask_nonzero
                b_mask = (maxc == b) & mask_nonzero
                hue[r_mask] = ((g - b)[r_mask] / delta[r_mask]) % 6
                hue[g_mask] = ((b - r)[g_mask] / delta[g_mask]) + 2
                hue[b_mask] = ((r - g)[b_mask] / delta[b_mask]) + 4
                hue = hue / 6.0
                # Saturation calculation
                sat[maxc > 0] = delta[maxc > 0] / maxc[maxc > 0]
                # Resize hue and saturation to match stacked size if needed
                H_stacked, W_stacked = stacked.shape[1:]
                if hue.shape != (H_stacked, W_stacked):
                    hue_img = Image.fromarray((hue * 255).astype(np.uint8))
                    hue_resized = np.array(hue_img.resize((W_stacked, H_stacked), Image.BILINEAR), dtype=np.float32) / 255.0
                    sat_img = Image.fromarray((sat * 255).astype(np.uint8))
                    sat_resized = np.array(sat_img.resize((W_stacked, H_stacked), Image.BILINEAR), dtype=np.float32) / 255.0
                else:
                    hue_resized = hue
                    sat_resized = sat
                # Append hue and optionally saturation
                if self.sfo_hsv_channels >= 1:
                    stacked = np.concatenate([stacked, hue_resized[None, :, :]], axis=0)
                if self.sfo_hsv_channels >= 2:
                    stacked = np.concatenate([stacked, sat_resized[None, :, :]], axis=0)
        # Resize to reference shape if defined and different from current shape
        if hasattr(self, 'reference_shape') and self.reference_shape is not None:
            ref_h, ref_w = self.reference_shape
            ch, h, w = stacked.shape
            if (h, w) != (ref_h, ref_w):
                resized_channels = []
                for c in range(ch):
                    arr = stacked[c]
                    img_pil = Image.fromarray((arr * 255).astype(np.uint8))
                    img_resized = img_pil.resize((ref_w, ref_h), Image.BILINEAR)
                    arr_resized = np.array(img_resized, dtype=np.float32) / 255.0
                    resized_channels.append(arr_resized)
                stacked = np.stack(resized_channels, axis=0)
        return torch.from_numpy(stacked)

    def _load_masks(self, mask_paths: List[Path] | Path) -> torch.Tensor:
        """Load zone masks for a sample as a tensor of shape (K,H,W).

        This function supports averaging across multiple labelers.  Mask files
        are grouped by their base name (e.g., ``1a``, ``2b``) ignoring
        differences in suffixes, case, or spaces.  If multiple files
        correspond to the same mask (e.g., ``1a.png``, ``1A_1.png``), their
        binary arrays are averaged to produce a soft mask in the range
        [0,1].  If there are no mask files, a ``RuntimeError`` is raised.
        """
        # Gather all PNG mask files
        # Accept either a single mask directory or a list of directories.
        if isinstance(mask_paths, Path):
            dirs = [mask_paths]
        else:
            dirs = mask_paths
        mask_files: List[Path] = []
        for d in dirs:
            if not d.is_dir():
                continue
            mask_files.extend([f for f in d.glob('*.png') if f.is_file()])
        if not mask_files:
            raise RuntimeError(f"No mask files found in {mask_paths}")
        # Group files by their GT mask key using the strict parser.
        groups: Dict[str, List[Path]] = {}
        for fpath in mask_files:
            key = extract_gt_key(fpath.name)
            if key is None:
                continue
            groups.setdefault(key, []).append(fpath)
        if not groups:
            raise RuntimeError(f"No valid mask files found in {mask_paths}")
        # Determine canonical image size from first mask
        first_path = next(iter(next(iter(groups.values()))))
        sample_arr = np.array(Image.open(first_path).convert('L'), dtype=np.float32)
        # Use orientation of the first mask as canonical; if additional masks
        # have transposed shape (W,H), we will transpose them to match
        H, W = sample_arr.shape
        # Build mask array for all known keys in a consistent order
        masks: List[np.ndarray] = []
        for key in getattr(self, 'mask_keys_order', []):
            if key in groups:
                file_list = groups[key]
                arrs: List[np.ndarray] = []
                for fpath in file_list:
                    arr = np.array(Image.open(fpath).convert('L'), dtype=np.float32)
                    # If the array shape matches canonical (H,W), keep it
                    if arr.shape == (H, W):
                        pass
                    # If the array appears transposed (W,H), transpose it
                    elif arr.shape == (W, H):
                        arr = arr.T
                    # Resize to match canonical shape if needed
                    if arr.shape != (H, W):
                        arr_img = Image.fromarray(arr.astype(np.uint8))
                        arr_img = arr_img.resize((W, H), Image.NEAREST)
                        arr = np.array(arr_img, dtype=np.float32)
                    # Convert to binary mask (0 or 1)
                    binary = (arr > 0).astype(np.float32)
                    arrs.append(binary)
                # Average across labelers
                if len(arrs) == 1:
                    mean_mask = arrs[0]
                else:
                    mean_mask = np.stack(arrs, axis=0).mean(axis=0)
            else:
                # No mask for this key; create an all-zero mask
                mean_mask = np.zeros((H, W), dtype=np.float32)
            masks.append(mean_mask)
        mask_arr = np.stack(masks, axis=0)  # (K,H,W)
        # Resize mask to reference shape if defined and shapes differ
        if hasattr(self, 'reference_shape') and self.reference_shape is not None:
            ref_h, ref_w = self.reference_shape
            k, h, w = mask_arr.shape
            if (h, w) != (ref_h, ref_w):
                resized_masks = []
                for m in mask_arr:
                    img_pil = Image.fromarray((m * 255).astype(np.uint8))
                    img_resized = img_pil.resize((ref_w, ref_h), Image.NEAREST)
                    arr_resized = np.array(img_resized, dtype=np.float32) / 255.0
                    resized_masks.append(arr_resized)
                mask_arr = np.stack(resized_masks, axis=0)
        mask_tensor = torch.from_numpy(mask_arr)
        return mask_tensor

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        sample_name = self.samples[index]
        img_dir = self.image_root / sample_name
        # Resolve mask directories for this sample
        mask_dirs = self.sample_to_mask_dirs.get(sample_name, [])
        image = self._load_signals(img_dir)  # (C,H,W)
        mask = self._load_masks(mask_dirs)    # (K,H,W)
        # Build a validity mask indicating which channels have ground truth.
        # A value of 1.0 means the corresponding mask is present; 0.0 means missing.
        # We use missing_masks mapping computed in __init__.
        num_classes = len(self.mask_keys_order)
        valid_vec = torch.ones(num_classes, dtype=torch.float32)
        if sample_name in self.missing_masks:
            missing_keys = self.missing_masks[sample_name]
            # convert keys to lower case to match mask_keys_order
            for mk in missing_keys:
                mk_lower = mk.lower()
                try:
                    idx = self.mask_keys_order.index(mk_lower)
                except ValueError:
                    # fallback if key is not found
                    continue
                valid_vec[idx] = 0.0
        # Apply transforms if provided
        if self.transforms:
            image, mask = self.transforms(image, mask)
        return {'image': image, 'mask': mask, 'valid': valid_vec, 'name': sample_name}


def list_samples(root: str | Path) -> List[str]:
    """List all sample directories under a given root.

    This helper assumes that each subdirectory under `root` corresponds to
    one sample.  It returns the names of all subdirectories (not full
    paths).
    """
    root_path = Path(root)
    samples = [p.name for p in root_path.iterdir() if p.is_dir()]
    samples.sort()
    return samples