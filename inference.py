"""
Inference script for knee growth plate zone detection.

Given a trained model checkpoint and a directory of input samples, this
script runs the model on each sample and saves the predicted masks as
binary PNG images.  It is intended for evaluation or deployment.

Example usage::

    python -m knee_zone_detection.inference \
        --image-root /path/to/Input/Image \
        --checkpoint ./checkpoints/best_model.pth \
        --output-dir ./predictions

This script loads the same dataset class used for training but does not
require ground truth masks.  It normalises input images to [0,1] and
applies a sigmoid to the network outputs.  Each output channel is
thresholded at 0.5 to produce a binary mask.  Predicted masks are saved
with filenames corresponding to their class index (e.g., ``0.png``,
``1.png``, etc.).

Note: The names of the predicted mask files correspond to the sorted
ground truth mask names encountered during training.  If you need to
preserve original mask names, you should save a mapping from class
indices to mask names during training and load it here.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import torch
from torchvision.utils import save_image

from data import KneeZoneDataset, list_samples
from model import UNetFusion, MultiBranchUNet
import torch.nn.functional as F
from typing import List, Tuple, Dict
from data import SIGNAL_FILES
from tqdm import tqdm

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run inference with trained knee segmentation model")
    parser.add_argument('--image_root', type=str, default='/home/yec23006/projects/research/KneeGrowthPlate/ZoneSeg/Input/Testdata/Image/', help='Root directory of input images')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint (.pth)')
    parser.add_argument('--output_dir', type=str, required=True, help='Directory to save predictions')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size for inference')
    parser.add_argument('--workers', type=int, default=2, help='Number of DataLoader workers')
    parser.add_argument('--use_hsv', action='store_true', help='Append a hue channel computed from grouped signals as in training')
    parser.add_argument('--hsv_groups', type=str, default='', help='Comma-separated lists of signal indices for R,G,B (e.g., "0 1 2,3 4 5,6 7 8")')
    # Optional mask root to determine number of output classes
    parser.add_argument('--mask_root', type=str, default='', help='Root directory of ground truth masks; used to infer output channel count')
    # SFO HSV options
    parser.add_argument('--sfo_hsv', action='store_true', help='Append HSV-derived channels from the SFO RGB image')
    parser.add_argument('--sfo_hsv_channels', type=int, default=1,
                        help='Number of channels to append from the SFO HSV representation (1=hue only, 2=hue and saturation)')
    # Target image size options (must match training)
    parser.add_argument('--img_size', type=int, default=512,
                        help='Target square image size (height=width). Overrides img-height and img-width if > 0.')
    parser.add_argument('--img_height', type=int, default=0,
                        help='Target image height (overridden by img-size if set).')
    parser.add_argument('--img_width', type=int, default=0,
                        help='Target image width (overridden by img-size if set).')
    # Optionally specify a subset of mask keys to load and predict.  Provide
    # comma-separated or space-separated list of keys (e.g., "1a 1b 1c" or "1a,1b").
    # Must match the keys used during training.  If not provided, all mask
    # keys defined in the training dataset are used.
    parser.add_argument('--selected_keys', type=str, default='',
                        help='Subset of mask keys to predict (comma/space separated)')
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    samples = list_samples(args.image_root)
    # Parse hsv groups if provided
    hsv_groups = None
    if args.hsv_groups:
        groups_str = args.hsv_groups.split(',')
        hsv_groups = []
        for grp in groups_str:
            indices = [int(i) for i in grp.strip().split() if i.strip() != '']
            hsv_groups.append(indices)
    # Determine input channels and output classes by loading a single sample's images and masks
    # Use KneeZoneDataset to inspect shapes only; we ignore masks for inference
    # Determine which mask root to use for inferring the number of classes. If a
    # mask root is provided and exists, use it; otherwise fall back to the
    # image root.  This is a best effort: if no masks exist, inference
    # cannot determine the correct number of output channels and may fail.
    mask_root = args.mask_root or args.image_root
    # Parse selected mask keys if provided
    selected_keys = None
    if args.selected_keys:
        import re
        tokens = re.split(r'[\s,]+', args.selected_keys.strip())
        sel = [t.strip().lower() for t in tokens if t.strip()]
        if sel:
            selected_keys = sel

    # Create a temporary dataset to determine mask keys and channel indices
    # Create a temporary dataset object only to reuse signal loading and mask-key ordering.
    # Do NOT index temp_dataset[0], because that would try to load GT masks.
    temp_dataset = KneeZoneDataset(
        args.image_root,
        mask_root,
        [samples[0]],
        use_hsv=args.use_hsv,
        hsv_groups=hsv_groups,
        sfo_hsv=args.sfo_hsv,
        sfo_hsv_channels=args.sfo_hsv_channels,
        target_size=args.img_size if args.img_size > 0 else None,
        target_height=args.img_height if args.img_height > 0 else None,
        target_width=args.img_width if args.img_width > 0 else None,
        selected_keys=selected_keys,
    )

    # Load only signals from the first sample to determine input channel count.
    first_sample_path = Path(args.image_root) / samples[0]
    tmp_image = temp_dataset._load_signals(first_sample_path)   # [C,H,W]

    in_ch = tmp_image.shape[0]
    mask_keys = temp_dataset.mask_keys_order
    num_classes = len(mask_keys)

    # Build mask key to index mapping
    key_to_idx = {k.lower(): i for i, k in enumerate([k.lower() for k in mask_keys])}
    # Identify cross groups and single groups as in training
    cross_groups_indices: dict[str, List[int]] = {}
    cross_keys_set: set[str] = set()
    if '9a' in key_to_idx and '9b' in key_to_idx:
        cross_groups_indices['9ab'] = [key_to_idx['9a'], key_to_idx['9b']]
        cross_keys_set.update(['9a', '9b'])
    if '8e' in key_to_idx and '8f' in key_to_idx:
        cross_groups_indices['8ef'] = [key_to_idx['8e'], key_to_idx['8f']]
        cross_keys_set.update(['8e', '8f'])
    if '6g' in key_to_idx and '8h' in key_to_idx:
        cross_groups_indices['6g8h'] = [key_to_idx['6g'], key_to_idx['8h']]
        cross_keys_set.update(['6g', '8h'])
    import re
    group_to_indices: dict[str, List[int]] = {}
    for k, idx in key_to_idx.items():
        if k in cross_keys_set:
            continue
        m = re.match(r'^(\d+)', k)
        if not m:
            continue
        digit = m.group(1)
        group_to_indices.setdefault(digit, []).append(idx)
    for grp in group_to_indices:
        group_to_indices[grp].sort()

    # Determine gating channel indices
    signal_order = list(SIGNAL_FILES.keys())
    try:
        cfo_idx = signal_order.index('cfo')
    except ValueError:
        cfo_idx = None
    try:
        ap_idx = signal_order.index('ap')
    except ValueError:
        ap_idx = None
    try:
        trap_idx = signal_order.index('trap')
    except ValueError:
        trap_idx = None
    try:
        sfo_idx = signal_order.index('sfo')
    except ValueError:
        sfo_idx = None
    if temp_dataset.sfo_hsv and temp_dataset.sfo_hsv_channels >= 1:
        sfo_hue_idx = in_ch - temp_dataset.sfo_hsv_channels
    else:
        sfo_hue_idx = sfo_idx
    cross_groups_cfg: dict[str, Tuple[int, List[int]]] = {}
    if '9ab' in cross_groups_indices and cfo_idx is not None and sfo_hue_idx is not None:
        cross_groups_cfg['9ab'] = (len(cross_groups_indices['9ab']), [cfo_idx, sfo_hue_idx])
    if '8ef' in cross_groups_indices and cfo_idx is not None and ap_idx is not None:
        cross_groups_cfg['8ef'] = (len(cross_groups_indices['8ef']), [cfo_idx, ap_idx])
    if '6g8h' in cross_groups_indices and ap_idx is not None and trap_idx is not None:
        cross_groups_cfg['6g8h'] = (len(cross_groups_indices['6g8h']), [ap_idx, trap_idx])
    # Build branches and gating information for the multi-branch model.  Each
    # single-digit group becomes its own branch with the number of masks
    # corresponding to its indices.  Cross groups become branches with
    # gating.  This mirrors the configuration used during training.
    branches: Dict[str, int] = {grp: len(idxs) for grp, idxs in group_to_indices.items()}
    for grp, (out_ch, gating_channels) in cross_groups_cfg.items():
        branches[grp] = out_ch
    gating_info: Dict[str, List[int]] = {
        grp: gating_channels for grp, (out_ch, gating_channels) in cross_groups_cfg.items()
    }
    # Instantiate the multi-branch model.  We do not specify use_fusion here
    # because the fusion behaviour is encapsulated in the branch decoders and
    # heads.  ``base_ch`` is fixed at 32 for inference as in training.
    model = MultiBranchUNet(
        n_channels=in_ch,
        branches=branches,
        gating_info=gating_info,
        base_ch=32,
        bilinear=True,
    )
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    state_dict = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(state_dict)
    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
    model.to(device)
    model.eval()

    # Use only the signal loading from the dataset to prepare inputs
    with torch.no_grad():
        for sample_name in tqdm(samples):
            dataset_path = Path(args.image_root) / sample_name
            # Load signals directly
            image_tensor = temp_dataset._load_signals(dataset_path)  # (C,H,W)
            # Ensure size matches model expectations via reference shape
            image_tensor = image_tensor.unsqueeze(0).to(device)  # add batch dim
            outputs_dict = model(image_tensor)
            # Compose full logits
            N = image_tensor.shape[0]
            H = next(iter(outputs_dict.values())).shape[2]
            W = next(iter(outputs_dict.values())).shape[3]
            full_logits = torch.zeros((N, num_classes, H, W), device=device)
            # Single groups
            for grp, idxs in group_to_indices.items():
                if grp not in outputs_dict:
                    continue
                logits = outputs_dict[grp]
                if logits.shape[2:] != (H, W):
                    logits = F.interpolate(logits, size=(H, W), mode='bilinear', align_corners=False)
                for j, idx in enumerate(idxs):
                    full_logits[:, idx] = logits[:, j]
            # Cross groups
            for grp, idxs in cross_groups_indices.items():
                if grp not in outputs_dict:
                    continue
                logits = outputs_dict[grp]
                if logits.shape[2:] != (H, W):
                    logits = F.interpolate(logits, size=(H, W), mode='bilinear', align_corners=False)
                for j, idx in enumerate(idxs):
                    full_logits[:, idx] = logits[:, j]
            probs = torch.sigmoid(full_logits)[0]  # (C,H,W) for single sample
            pred_bin = (probs > 0.5).float()
            # Save masks using mask key names
            sample_out_dir = Path(args.output_dir) / sample_name
            sample_out_dir.mkdir(parents=True, exist_ok=True)
            for i, key in enumerate(mask_keys):
                mask = pred_bin[i:i+1]  # (1,H,W)
                filename = f"{key}.png"
                save_path = sample_out_dir / filename
                save_image(mask, str(save_path))


if __name__ == '__main__':
    main()