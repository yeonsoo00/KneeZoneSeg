"""
Training script for knee growth plate zone detection.

This script trains a segmentation model on the provided dataset using
PyTorch.  It supports training on multiple GPUs via ``torch.nn.DataParallel``,
logging to Weights & Biases (wandb) for monitoring metrics and images, and
checkpointing the best model by validation loss.

Example usage from the command line::

    python -m knee_zone_detection.train \
        --image-root /path/to/Input/Image \
        --mask-root /path/to/Input/Mask \
        --output-dir ./checkpoints \
        --epochs 50 \
        --batch-size 2 \
        --lr 1e-4 \
        --workers 4

"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import List, Tuple, Dict

import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

from data import KneeZoneDataset, list_samples, SIGNAL_FILES
from model import UNetFusion, MultiBranchUNet
from losses import dice_loss, hd95_loss, assd
import numpy as np  

# -----------------------------------------------------------------------------
# Helper functions for wandb overlay logging
# -----------------------------------------------------------------------------

def _normalize_vis_channel(x: torch.Tensor) -> np.ndarray:
    """Normalize a single channel tensor to [0,1] numpy."""
    x = x.detach().float().cpu().numpy()
    x_min = x.min()
    x_max = x.max()
    if x_max > x_min:
        x = (x - x_min) / (x_max - x_min + 1e-8)
    else:
        x = np.zeros_like(x, dtype=np.float32)
    return x.astype(np.float32)


def overlay_mask_np(
    image_rgb: np.ndarray,
    mask: torch.Tensor,
    color: tuple = (1.0, 0.0, 0.0),
    alpha: float = 0.4,
    thr: float = 0.5,
) -> np.ndarray:
    """
    Overlay a single mask on an RGB background image.

    Args:
        image_rgb: numpy array [H,W,3] in [0,1]
        mask: tensor [H,W]
        color: RGB color
        alpha: blend factor
        thr: threshold for overlay
    """
    overlay = image_rgb.copy()
    mask_np = mask.detach().float().cpu().numpy()
    binary = mask_np > thr
    for c in range(3):
        overlay[:, :, c] = np.where(
            binary,
            (1.0 - alpha) * overlay[:, :, c] + alpha * color[c],
            overlay[:, :, c],
        )
    return np.clip(overlay, 0.0, 1.0)


def _make_gray_rgb(ch: torch.Tensor) -> np.ndarray:
    """Convert a single channel tensor [H,W] to grayscale RGB numpy [H,W,3]."""
    x = _normalize_vis_channel(ch)
    return np.stack([x, x, x], axis=-1)


def _make_two_channel_composite(ch1: torch.Tensor, ch2: torch.Tensor) -> np.ndarray:
    """
    Make a simple RGB composite from two channels:
      R = ch1
      G = ch2
      B = average(ch1, ch2)
    """
    x1 = _normalize_vis_channel(ch1)
    x2 = _normalize_vis_channel(ch2)
    xb = 0.5 * (x1 + x2)
    rgb = np.stack([x1, x2, xb], axis=-1)
    return np.clip(rgb, 0.0, 1.0)


def get_background_for_mask_key(
    image: torch.Tensor,
    mask_key: str,
    signal_to_input_idx: dict,
) -> np.ndarray:
    """
    Return background RGB image corresponding to the given prediction/GT mask key.

    image: tensor [C,H,W]
    mask_key: e.g. '1a', '8e', '9b'
    signal_to_input_idx: maps signal names to channel indices in image tensor
    """
    k = mask_key.lower().strip()

    def has(sig: str) -> bool:
        return sig in signal_to_input_idx and signal_to_input_idx[sig] < image.shape[0]

    def ch(sig: str) -> torch.Tensor:
        return image[signal_to_input_idx[sig]]

    # ---- special fusion backgrounds first ----
    # 9a, 9b use SFO + CFO
    if k in ("9a", "9b") and has("sfo") and has("cfo"):
        return _make_two_channel_composite(ch("sfo"), ch("cfo"))

    # 8e, 8f use CFO + AP
    if k in ("8e", "8f") and has("cfo") and has("ap"):
        return _make_two_channel_composite(ch("cfo"), ch("ap"))

    # 6g / 6h / 8h use AP + TRAP
    if k in ("6g", "6h", "8h") and has("ap") and has("trap"):
        return _make_two_channel_composite(ch("ap"), ch("trap"))

    # ---- regular per-signal backgrounds ----
    if k.startswith("1") and has("mineral"):
        return _make_gray_rgb(ch("mineral"))
    if k.startswith("2") and has("ac"):
        return _make_gray_rgb(ch("ac"))
    if k.startswith("3") and has("calcein"):
        return _make_gray_rgb(ch("calcein"))
    if k.startswith("4") and has("trap"):
        return _make_gray_rgb(ch("trap"))
    if k.startswith("5") and has("dapi"):
        return _make_gray_rgb(ch("dapi"))
    if k.startswith("6") and has("ap"):
        return _make_gray_rgb(ch("ap"))
    if k.startswith("7") and has("edu"):
        return _make_gray_rgb(ch("edu"))
    if k.startswith("8") and has("cfo"):
        return _make_gray_rgb(ch("cfo"))
    if k.startswith("9") and has("sfo"):
        return _make_gray_rgb(ch("sfo"))

    # fallback: first channel
    return _make_gray_rgb(image[0])


def create_overlay_panel_for_sample(
    image: torch.Tensor,
    masks: torch.Tensor,
    mask_keys: List[str],
    signal_to_input_idx: dict,
    valid_vec: torch.Tensor | None = None,
    color: tuple = (1.0, 0.0, 0.0),
    alpha: float = 0.4,
    thr: float = 0.5,
    max_cols: int = 4,
) -> np.ndarray:
    """
    Create one big panel image containing all overlays for a single sample.

    Each tile uses the background corresponding to that mask key.
    """
    num_masks = min(len(mask_keys), masks.shape[0])
    tile_h = masks.shape[1]
    tile_w = masks.shape[2]

    # choose which channels to show
    shown_indices = []
    for j in range(num_masks):
        if valid_vec is None or valid_vec[j].item() > 0:
            shown_indices.append(j)

    if len(shown_indices) == 0:
        shown_indices = list(range(num_masks))

    n = len(shown_indices)
    ncols = min(max_cols, n)
    nrows = int(np.ceil(n / ncols))

    panel = np.ones((nrows * tile_h, ncols * tile_w, 3), dtype=np.float32)

    for idx_panel, j in enumerate(shown_indices):
        r = idx_panel // ncols
        c = idx_panel % ncols

        bg = get_background_for_mask_key(
            image=image,
            mask_key=mask_keys[j],
            signal_to_input_idx=signal_to_input_idx,
        )
        ov = overlay_mask_np(
            bg,
            masks[j],
            color=color,
            alpha=alpha,
            thr=thr,
        )

        y0 = r * tile_h
        y1 = y0 + tile_h
        x0 = c * tile_w
        x1 = x0 + tile_w
        panel[y0:y1, x0:x1] = ov

    return np.clip(panel, 0.0, 1.0)


def assemble_full_logits_from_outputs(
    outputs_dict,
    images,
    masks,
    num_classes,
    group_to_indices,
    cross_groups_indices,
):
    """Convert model outputs dict to full logits tensor [N,C,H,W]."""
    N = images.shape[0]
    H = masks.shape[2]
    W = masks.shape[3]
    full_logits = images.new_zeros((N, num_classes, H, W))

    for grp, idxs in group_to_indices.items():
        if grp not in outputs_dict:
            continue
        logits = outputs_dict[grp]
        if logits.shape[2:] != (H, W):
            logits = F.interpolate(logits, size=(H, W), mode='bilinear', align_corners=False)
        for j, idx in enumerate(idxs):
            full_logits[:, idx] = logits[:, j]

    for grp, idxs in cross_groups_indices.items():
        if grp not in outputs_dict:
            continue
        logits = outputs_dict[grp]
        if logits.shape[2:] != (H, W):
            logits = F.interpolate(logits, size=(H, W), mode='bilinear', align_corners=False)
        for j, idx in enumerate(idxs):
            full_logits[:, idx] = logits[:, j]

    return full_logits


def log_overlay_panels_to_wandb(
    model,
    dataset,
    device,
    epoch,
    split_name,
    num_classes,
    group_to_indices,
    cross_groups_indices,
    mask_keys,
    signal_to_input_idx,
):
    """
    Log one sample with all prediction/GT overlays using corresponding backgrounds.

    W&B sections:
      Pred/
      GT/
      Val/
    """
    if not WANDB_AVAILABLE or len(dataset) == 0:
        return

    was_training = model.training
    model.eval()

    sample = dataset[0]
    image = sample["image"].unsqueeze(0).to(device)
    mask = sample["mask"].unsqueeze(0).to(device)
    valid = sample["valid"].unsqueeze(0).to(device)
    sample_name = sample["name"]

    with torch.no_grad():
        outputs_dict = model(image)
        full_logits = assemble_full_logits_from_outputs(
            outputs_dict=outputs_dict,
            images=image,
            masks=mask,
            num_classes=num_classes,
            group_to_indices=group_to_indices,
            cross_groups_indices=cross_groups_indices,
        )
        probs = torch.sigmoid(full_logits)

    pred_panel = create_overlay_panel_for_sample(
        image=image[0].detach().cpu(),
        masks=probs[0].detach().cpu(),
        mask_keys=mask_keys,
        signal_to_input_idx=signal_to_input_idx,
        valid_vec=valid[0].detach().cpu(),
        color=(1.0, 0.0, 0.0),
        alpha=0.4,
        thr=0.5,
        max_cols=4,
    )

    gt_panel = create_overlay_panel_for_sample(
        image=image[0].detach().cpu(),
        masks=mask[0].detach().cpu(),
        mask_keys=mask_keys,
        signal_to_input_idx=signal_to_input_idx,
        valid_vec=valid[0].detach().cpu(),
        color=(0.0, 1.0, 0.0),
        alpha=0.4,
        thr=0.5,
        max_cols=4,
    )

    valid_keys = [
        k for k, v in zip(mask_keys, valid[0].detach().cpu().tolist()) if v > 0
    ]
    missing_keys = [
        k for k, v in zip(mask_keys, valid[0].detach().cpu().tolist()) if v == 0
    ]

    wandb.log(
        {
            f"Pred/{split_name}_overlay_panel": wandb.Image(
                pred_panel,
                caption=f"{split_name} prediction overlays | sample={sample_name} | epoch={epoch}",
            ),
            f"GT/{split_name}_overlay_panel": wandb.Image(
                gt_panel,
                caption=f"{split_name} GT overlays | sample={sample_name} | epoch={epoch}",
            ),
            f"Val/{split_name}_valid_GT_keys": ", ".join(valid_keys),
            f"Val/{split_name}_missing_GT_keys": ", ".join(missing_keys),
        },
        step=epoch,
    )

    if was_training:
        model.train()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train knee zone segmentation model")
    parser.add_argument('--image_root', type=str, default = '/home/yec23006/projects/research/KneeGrowthPlate/ZoneSeg/Input/Image', help='Root directory of input images')
    parser.add_argument('--mask_root', type=str, default = '/home/yec23006/projects/research/KneeGrowthPlate/ZoneSeg/Input/Mask', help='Root directory of ground truth masks')
    parser.add_argument('--output_dir', type=str, default = 'ckpt', help='Directory to save checkpoints')
    parser.add_argument('--epochs', type=int, default=1000, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--workers', type=int, default=4, help='Number of data loader workers')
    parser.add_argument('--val_split', type=float, default=0.1, help='Fraction of data to use for validation')
    parser.add_argument('--dice_weight', type=float, default=1.0, help='Weight for Dice loss')
    parser.add_argument('--hd_weight', type=float, default=0.0, help='Weight for HD95 loss')
    parser.add_argument('--assd_weight', type=float, default=0.0, help='Weight for ASSD loss')
    parser.add_argument('--bce_weight', type=float, default=0.0, help='Weight for BCE loss term')
    parser.add_argument('--pair_weight', type=float, default=0.1, help='Weight for pair consistency penalty')
    parser.add_argument('--project', type=str, default='knee_seg_fusionunet', help='wandb project name')

    # HSV options
    parser.add_argument('--use_hsv', action='store_true', help='Append a hue channel computed from grouped signals')
    parser.add_argument('--hsv_groups', type=str, default='', help='Comma-separated lists of signal indices for R,G,B (e.g., "0 1 2,3 4 5,6 7 8")')
    # Model size
    parser.add_argument('--base_ch', type=int, default=32, help='Base number of channels for the U-Net model (reduce to save memory)')
    # SFO HSV options
    parser.add_argument('--sfo_hsv', action='store_true', help='Append HSV-derived channels from the SFO RGB image')
    parser.add_argument('--sfo_hsv_channels', type=int, default=1,
                        help='Number of channels to append from the SFO HSV representation (1=hue only, 2=hue and saturation)')

    # Target image size options
    parser.add_argument('--img_size', type=int, default = 512,
                        help='Target square image size (height=width). Overrides img-height and img-width if > 0.')
    parser.add_argument('--img_height', type=int, default=0,
                        help='Target image height (overridden by img-size if set).')
    parser.add_argument('--img_width', type=int, default=0,
                        help='Target image width (overridden by img-size if set).')
    # Optionally specify a subset of mask keys to train on.  Provide a
    # comma-separated or space-separated list of keys (e.g., "1a 1b 1c" or "1a,1b").
    # When specified, the dataset will only load and return these mask
    # channels, allowing separate models per signal or fused group.  By
    # default (empty string) all expected mask keys are used.
    parser.add_argument('--selected-keys', type=str, default='',
                        help='Subset of mask keys to train on (comma/space separated).')
    return parser.parse_args()


def split_samples(all_samples: List[str], val_split: float) -> tuple[list[str], list[str]]:
    # Simple deterministic split by sorting and slicing
    n_total = len(all_samples)
    n_val = int(n_total * val_split)
    val_samples = all_samples[:n_val]
    train_samples = all_samples[n_val:]
    return train_samples, val_samples


def main() -> None:
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # Discover sample names
    all_samples = list_samples(args.image_root)
    train_samples, val_samples = split_samples(all_samples, args.val_split)
    print(f"Found {len(all_samples)} samples: {len(train_samples)} train, {len(val_samples)} val")

    # Create datasets and loaders
    # Parse hsv groups if provided
    hsv_groups = None
    if args.hsv_groups:
        groups_str = args.hsv_groups.split(',')
        hsv_groups = []
        for grp in groups_str:
            indices = [int(i) for i in grp.strip().split() if i.strip() != '']
            hsv_groups.append(indices)
    # Parse selected keys if provided
    selected_keys = None
    if hasattr(args, 'selected_keys') and args.selected_keys:
        # split by commas or whitespace
        import re
        tokens = re.split(r'[\s,]+', args.selected_keys.strip())
        sel = [t.strip().lower() for t in tokens if t.strip()]
        if sel:
            selected_keys = sel

    train_dataset = KneeZoneDataset(
        args.image_root,
        args.mask_root,
        train_samples,
        use_hsv=args.use_hsv,
        hsv_groups=hsv_groups,
        sfo_hsv=args.sfo_hsv,
        sfo_hsv_channels=args.sfo_hsv_channels,
        target_size=args.img_size if args.img_size > 0 else None,
        target_height=args.img_height if args.img_height > 0 else None,
        target_width=args.img_width if args.img_width > 0 else None,
        selected_keys=selected_keys,
    )
    val_dataset = KneeZoneDataset(
        args.image_root,
        args.mask_root,
        val_samples,
        use_hsv=args.use_hsv,
        hsv_groups=hsv_groups,
        sfo_hsv=args.sfo_hsv,
        sfo_hsv_channels=args.sfo_hsv_channels,
        target_size=args.img_size if args.img_size > 0 else None,
        target_height=args.img_height if args.img_height > 0 else None,
        target_width=args.img_width if args.img_width > 0 else None,
        selected_keys=selected_keys,
    )
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False,
                            num_workers=args.workers, pin_memory=True)

    # Determine the number of input channels and the ordered mask keys
    sample = train_dataset[0]
    in_ch = sample['image'].shape[0]
    mask_keys = train_dataset.mask_keys_order  # list of mask keys in consistent order
    num_classes = len(mask_keys)
    print(f"Model with {in_ch} input channels and {num_classes} output classes")

    # Build mapping from mask keys to their indices
    key_to_idx = {k.lower(): i for i, k in enumerate([k.lower() for k in mask_keys])}

    # Identify cross-group masks based on project specification
    cross_groups_indices: dict[str, List[int]] = {}
    cross_keys_set: set[str] = set()
    # SFO (9) and CFO pair: keys '9a' and '9b'
    if '9a' in key_to_idx and '9b' in key_to_idx:
        cross_groups_indices['9ab'] = [key_to_idx['9a'], key_to_idx['9b']]
        cross_keys_set.update(['9a', '9b'])
    # CFO (8) and AP pair: keys '8e' and '8f'
    if '8e' in key_to_idx and '8f' in key_to_idx:
        cross_groups_indices['8ef'] = [key_to_idx['8e'], key_to_idx['8f']]
        cross_keys_set.update(['8e', '8f'])
    # AP (6) and TRAP/CFO pair: keys '6g' and '8h'
    if '6g' in key_to_idx and '8h' in key_to_idx:
        cross_groups_indices['6g8h'] = [key_to_idx['6g'], key_to_idx['8h']]
        cross_keys_set.update(['6g', '8h'])

    # Build single-head groups by signal digit (excluding cross-group keys)
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
    # Sort indices within each group for consistency
    for grp in group_to_indices:
        group_to_indices[grp].sort()

    # Determine gating channel indices based on input signals
    # Map signal names to their positions in the stacked input tensor
    signal_order = list(SIGNAL_FILES.keys())  # base signals order
    # Base indices
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
    # Determine index for SFO HSV hue channel
    # If SFO HSV channels are appended, they are the last channels in the stacked tensor
    if train_dataset.sfo_hsv and train_dataset.sfo_hsv_channels >= 1:
        sfo_hue_idx = in_ch - train_dataset.sfo_hsv_channels  # first appended SFO HSV channel
    else:
        sfo_hue_idx = sfo_idx

    # Build cross-group configuration for the model: mapping to (num_outputs, gating_channels)
    cross_groups_cfg: dict[str, Tuple[int, List[int]]] = {}
    if '9ab' in cross_groups_indices and cfo_idx is not None and sfo_hue_idx is not None:
        cross_groups_cfg['9ab'] = (len(cross_groups_indices['9ab']), [cfo_idx, sfo_hue_idx])
    if '8ef' in cross_groups_indices and cfo_idx is not None and ap_idx is not None:
        cross_groups_cfg['8ef'] = (len(cross_groups_indices['8ef']), [cfo_idx, ap_idx])
    if '6g8h' in cross_groups_indices and ap_idx is not None and trap_idx is not None:
        cross_groups_cfg['6g8h'] = (len(cross_groups_indices['6g8h']), [ap_idx, trap_idx])

    # Build branch definitions for the multi-branch model.  Each single-digit
    # group becomes its own branch with ``len(idxs)`` output channels.  Cross
    # groups become separate branches with the number of output channels and
    # gating information derived above.  This design allows each task
    # family to have its own decoder and head while sharing the encoder.
    branches: Dict[str, int] = {grp: len(idxs) for grp, idxs in group_to_indices.items()}
    for grp, (out_ch, gating_channels) in cross_groups_cfg.items():
        branches[grp] = out_ch
    # Extract gating channel indices for cross-stain branches.  Only cross
    # branches require gating; single branches are not present in this dict.
    gating_info: Dict[str, List[int]] = {
        grp: gating_channels for grp, (out_ch, gating_channels) in cross_groups_cfg.items()
    }
    # Instantiate the multi-branch model.  This replaces the old MultiHeadUNet
    # and provides separate decoders for each branch, with optional gating for
    # cross-stain families.  The encoder width is set by ``base_ch`` and
    # bilinear upsampling is used throughout.
    model = MultiBranchUNet(
        n_channels=in_ch,
        branches=branches,
        gating_info=gating_info,
        base_ch=args.base_ch,
        bilinear=True,
    )

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # DataParallel if multiple GPUs
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs via DataParallel")
        model = torch.nn.DataParallel(model)
    model.to(device)

    # Optimiser and scheduler
    optimizer = Adam(model.parameters(), lr=args.lr)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)

    # Prepare mapping from group name to target indices (including cross groups)
    all_groups_indices = {**group_to_indices, **cross_groups_indices}

    # Prepare list of class index pairs for pair consistency loss
    pair_indices: List[Tuple[int, int]] = []
    # Define pairs by letters
    letter_pairs = [('a', 'b'), ('c', 'd'), ('e', 'f'), ('g', 'h')]
    # Map key to index
    for d in range(1, 10):
        digit = str(d)
        for first, second in letter_pairs:
            k1 = f"{digit}{first}"
            k2 = f"{digit}{second}"
            if k1 in key_to_idx and k2 in key_to_idx:
                pair_indices.append((key_to_idx[k1], key_to_idx[k2]))
    # Special cross-digit pair 6g and 8h
    if '6g' in key_to_idx and '8h' in key_to_idx:
        pair_indices.append((key_to_idx['6g'], key_to_idx['8h']))

    # Initialise wandb
    if WANDB_AVAILABLE:
        wandb.init(project=args.project, config=vars(args))
        wandb.watch(model, log='all')

    best_val_loss = float('inf')
    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        for batch in train_loader:
            images = batch['image'].to(device)
            masks = batch['mask'].to(device)
            optimizer.zero_grad()
            # Forward pass: get outputs per group
            outputs_dict = model(images)
            # Compose full logits tensor
            # Determine spatial height and width from masks
            # masks shape: (N, K, H, W)
            N = images.shape[0]
            H = masks.shape[2]
            W = masks.shape[3]
            # Allocate full logits with shape (N, num_classes, H, W)
            full_logits = images.new_zeros((N, num_classes, H, W))
            # Fill logits for single groups
            for grp, idxs in group_to_indices.items():
                if grp not in outputs_dict:
                    continue
                logits = outputs_dict[grp]
                # If spatial dims mismatch, resize
                if logits.shape[2:] != (H, W):
                    logits = F.interpolate(logits, size=(H, W), mode='bilinear', align_corners=False)
                for j, idx in enumerate(idxs):
                    full_logits[:, idx] = logits[:, j]
            # Fill logits for cross groups
            for grp, idxs in cross_groups_indices.items():
                if grp not in outputs_dict:
                    continue
                logits = outputs_dict[grp]
                if logits.shape[2:] != (H, W):
                    logits = F.interpolate(logits, size=(H, W), mode='bilinear', align_corners=False)
                for j, idx in enumerate(idxs):
                    full_logits[:, idx] = logits[:, j]
            # Validity mask per sample and class
            valid_mask = batch['valid'].to(device)  # (N, K)
            # Precompute sigmoid predictions
            probs = torch.sigmoid(full_logits)
            # Compute BCE loss only over valid channels
            # reduction='none' to compute per-element loss
            bce_tensor = F.binary_cross_entropy_with_logits(full_logits, masks, reduction='none')  # (N,K,H,W)
            # Broadcast validity mask to match (N,K,H,W)
            valid_mask_broadcast = valid_mask.unsqueeze(-1).unsqueeze(-1)  # (N,K,1,1)
            bce_tensor = bce_tensor * valid_mask_broadcast
            # Denominator: total number of valid elements (sum of valid entries * H * W)
            denom = valid_mask_broadcast.sum() * H * W
            if denom.item() > 0:
                bce_loss = bce_tensor.sum() / denom
            else:
                bce_loss = torch.tensor(0.0, device=device)
            # Compute Dice loss only over valid channels
            eps = 1e-6
            # Flatten predictions and targets
            probs_flat = probs.view(N, num_classes, -1)
            masks_flat = masks.view(N, num_classes, -1)
            intersection = (probs_flat * masks_flat).sum(dim=2)  # (N,K)
            union = probs_flat.sum(dim=2) + masks_flat.sum(dim=2)  # (N,K)
            dice_per = (2.0 * intersection + eps) / (union + eps)  # (N,K)
            # Apply validity mask
            # Multiply dice coefficients by valid_mask and sum
            dice_sum = (dice_per * valid_mask).sum()
            denom_dice = valid_mask.sum() + eps
            dice = 1.0 - (dice_sum / denom_dice)
            # Compute HD95 and ASSD only on valid channels per sample
            hd_vals = []
            assd_vals = []
            # Iterate per sample
            for i in range(N):
                valid_indices = (valid_mask[i] > 0).nonzero(as_tuple=False).flatten().tolist()
                if not valid_indices:
                    continue
                # Extract predictions and targets for valid channels
                pred_sub = probs[i:i+1, valid_indices]
                target_sub = masks[i:i+1, valid_indices]
                # Compute hd95 and assd for this sample
                # hd95_loss and assd expect probabilities; use pred_sub and target_sub
                if args.hd_weight != 0:
                    hd_sample = hd95_loss(pred_sub, target_sub)
                    hd_vals.append(hd_sample.item())
                
                if args.assd_weight != 0:
                    assd_sample = assd(pred_sub, target_sub)                    
                    assd_vals.append(assd_sample.item())

            if args.hd_weight != 0:
                if hd_vals:
                    hd = torch.tensor(float(sum(hd_vals)) / len(hd_vals), device=device) * args.hd_weight
                else:
                    hd = torch.tensor(0.0, device=device)

            if args.assd_weight != 0:
                if assd_vals:
                    a_sd = torch.tensor(float(sum(assd_vals)) / len(assd_vals), device=device) * args.assd_weight
                else:
                    a_sd = torch.tensor(0.0, device=device)

            # Compute pair consistency penalty only on pairs where both masks are valid
            pair_pen = 0.0
            if pair_indices and args.pair_weight != 0:
                pair_losses = []
                for (i1, i2) in pair_indices:
                    # Determine for each sample if this pair is valid
                    valid_pair = (valid_mask[:, i1] * valid_mask[:, i2]).view(N, 1, 1)
                    total_valid = valid_pair.sum() * H * W
                    if total_valid.item() == 0:
                        continue
                    product = (probs[:, i1] * probs[:, i2]) * valid_pair
                    penalty = product.sum() / total_valid
                    pair_losses.append(penalty)
                if pair_losses:
                    pair_pen = sum(pair_losses) / len(pair_losses)
                else:
                    pair_pen = torch.tensor(0.0, device=device)
            # Total loss
            if args.assd_weight != 0 and args.hd_weight != 0: # Both hd95 and assd
                total_loss = args.bce_weight * bce_loss + args.dice_weight * dice + hd + a_sd + args.pair_weight * pair_pen
            elif args.assd_weight == 0 and args.hd_weight == 0: # No hd95 nor assd
                total_loss = args.bce_weight * bce_loss + args.dice_weight * dice + args.pair_weight * pair_pen
            elif args.assd_weight != 0 and args.hd_weight == 0: # No hd95 loss
                total_loss = args.bce_weight * bce_loss + args.dice_weight * dice + a_sd + args.pair_weight * pair_pen
            elif args.assd_weight == 0 and args.hd_weight != 0: # No assd loss
                total_loss = args.bce_weight * bce_loss + args.dice_weight * dice + hd + args.pair_weight * pair_pen
            
            total_loss.backward()
            optimizer.step()
            running_loss += total_loss.item() * images.size(0)
        train_loss = running_loss / len(train_loader.dataset)

        # Validation
        model.eval()
        val_running_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                images = batch['image'].to(device)
                masks = batch['mask'].to(device)
                outputs_dict = model(images)
                # Compose full logits as for training
                N = images.shape[0]
                H = masks.shape[2]
                W = masks.shape[3]
                full_logits = images.new_zeros((N, num_classes, H, W))
                for grp, idxs in group_to_indices.items():
                    if grp not in outputs_dict:
                        continue
                    logits = outputs_dict[grp]
                    if logits.shape[2:] != (H, W):
                        logits = F.interpolate(logits, size=(H, W), mode='bilinear', align_corners=False)
                    for j, idx in enumerate(idxs):
                        full_logits[:, idx] = logits[:, j]
                for grp, idxs in cross_groups_indices.items():
                    if grp not in outputs_dict:
                        continue
                    logits = outputs_dict[grp]
                    if logits.shape[2:] != (H, W):
                        logits = F.interpolate(logits, size=(H, W), mode='bilinear', align_corners=False)
                    for j, idx in enumerate(idxs):
                        full_logits[:, idx] = logits[:, j]
                # Validity mask per sample
                valid_mask = batch['valid'].to(device)  # (N,K)
                probs_val = torch.sigmoid(full_logits)
                # BCE
                bce_tensor_val = F.binary_cross_entropy_with_logits(full_logits, masks, reduction='none')
                valid_mask_broad = valid_mask.unsqueeze(-1).unsqueeze(-1)
                bce_tensor_val = bce_tensor_val * valid_mask_broad
                denom_val = valid_mask_broad.sum() * H * W
                if denom_val.item() > 0:
                    bce_loss_val = bce_tensor_val.sum() / denom_val
                else:
                    bce_loss_val = torch.tensor(0.0, device=device)
                # Dice
                eps = 1e-6
                probs_flat_val = probs_val.view(N, num_classes, -1)
                masks_flat_val = masks.view(N, num_classes, -1)
                intersection_val = (probs_flat_val * masks_flat_val).sum(dim=2)
                union_val = probs_flat_val.sum(dim=2) + masks_flat_val.sum(dim=2)
                dice_per_val = (2.0 * intersection_val + eps) / (union_val + eps)
                dice_sum_val = (dice_per_val * valid_mask).sum()
                denom_dice_val = valid_mask.sum() + eps
                dice_val = 1.0 - (dice_sum_val / denom_dice_val)
                # HD95 and ASSD
                hd_vals = []
                assd_vals = []
                for i in range(N):
                    valid_indices = (valid_mask[i] > 0).nonzero(as_tuple=False).flatten().tolist()
                    if not valid_indices:
                        continue
                    pred_sub = probs_val[i:i+1, valid_indices]
                    target_sub = masks[i:i+1, valid_indices]
                    if args.hd_weight != 0:
                        hd_sample = hd95_loss(pred_sub, target_sub)
                        hd_vals.append(hd_sample.item())
                    if args.assd_weight != 0:
                        assd_sample = assd(pred_sub, target_sub)
                        assd_vals.append(assd_sample.item())
                if args.hd_weight != 0:
                    if hd_vals:
                        hd_val = torch.tensor(float(sum(hd_vals)) / len(hd_vals), device=device) * args.hd_weight
                    else:
                        hd_val = torch.tensor(0.0, device=device)
                if args.assd_weight != 0:
                    if assd_vals:
                        a_sd_val = torch.tensor(float(sum(assd_vals)) / len(assd_vals), device=device) * args.assd_weight
                    else:
                        a_sd_val = torch.tensor(0.0, device=device)
                # Pair penalty
                pair_pen_val = 0.0
                if pair_indices and args.pair_weight != 0:
                    pair_losses_val = []
                    for (i1, i2) in pair_indices:
                        valid_pair_val = (valid_mask[:, i1] * valid_mask[:, i2]).view(N, 1, 1)
                        total_valid_val = valid_pair_val.sum() * H * W
                        if total_valid_val.item() == 0:
                            continue
                        product_val = (probs_val[:, i1] * probs_val[:, i2]) * valid_pair_val
                        penalty_val = product_val.sum() / total_valid_val
                        pair_losses_val.append(penalty_val)
                    if pair_losses_val:
                        pair_pen_val = sum(pair_losses_val) / len(pair_losses_val)
                    else:
                        pair_pen_val = torch.tensor(0.0, device=device)
                if args.assd_weight != 0 and args.hd_weight != 0: # Both HD95 and ASSD
                    val_loss_batch = args.bce_weight * bce_loss_val + args.dice_weight * dice_val + hd_val + a_sd_val + args.pair_weight * pair_pen_val
                elif args.assd_weight == 0 and args.hd_weight == 0: # No HD95 nor ASSD
                    val_loss_batch = args.bce_weight * bce_loss_val + args.dice_weight * dice_val + args.pair_weight * pair_pen_val
                elif args.assd_weight == 0 and args.hd_weight != 0: # HD95 only
                    val_loss_batch = args.bce_weight * bce_loss_val + args.dice_weight * dice_val + hd_val + args.pair_weight * pair_pen_val
                elif args.assd_weight != 0 and args.hd_weight == 0: # ASSD only
                    val_loss_batch = args.bce_weight * bce_loss_val + args.dice_weight * dice_val + a_sd_val + args.pair_weight * pair_pen_val

                val_running_loss += val_loss_batch.item() * images.size(0)
        val_loss = val_running_loss / len(val_loader.dataset)
        scheduler.step(val_loss)

        print(f"Epoch {epoch}/{args.epochs}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}")
        if WANDB_AVAILABLE:
            wandb.log({'epoch': epoch, 'train_loss': train_loss, 'val_loss': val_loss})

            # Map each signal name to the corresponding input channel index
            signal_to_input_idx = {sig.lower(): i for i, sig in enumerate(SIGNAL_FILES.keys())}

            log_overlay_panels_to_wandb(
                model=model,
                dataset=train_dataset,
                device=device,
                epoch=epoch,
                split_name='train',
                num_classes=num_classes,
                group_to_indices=group_to_indices,
                cross_groups_indices=cross_groups_indices,
                mask_keys=mask_keys,
                signal_to_input_idx=signal_to_input_idx,
            )

            log_overlay_panels_to_wandb(
                model=model,
                dataset=val_dataset,
                device=device,
                epoch=epoch,
                split_name='val',
                num_classes=num_classes,
                group_to_indices=group_to_indices,
                cross_groups_indices=cross_groups_indices,
                mask_keys=mask_keys,
                signal_to_input_idx=signal_to_input_idx,
            )
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            ckpt_path = Path(args.output_dir) / f"best_model_ch64_dicepen.pth"
            # Save state dict; when using DataParallel, save module state_dict
            state_dict = model.module.state_dict() if isinstance(model, torch.nn.DataParallel) else model.state_dict()
            torch.save(state_dict, ckpt_path)
            print(f"Saved checkpoint to {ckpt_path}")

    if WANDB_AVAILABLE:
        wandb.finish()


if __name__ == '__main__':
    main()