"""
Loss functions for knee growth plate zone detection.

This module defines differentiable loss functions used to train the
segmentation model.  The primary objective is a combination of Dice loss
and the 95th percentile Hausdorff distance (HD95).  Optionally, the
Average Symmetric Surface Distance (ASSD) can be added to encourage
overall surface smoothness.

Dice loss measures overlap between predicted and ground truth masks and
encourages correct segmentation of the area.  Hausdorff distance is a
metric on sets that measures the worst deviation between two surfaces.  In
practice we compute the 95th percentile Hausdorff distance to reduce
sensitivity to outliers.

Because HD95 and ASSD are non‑differentiable metrics, they are not
backpropagated directly.  Instead, we compute them on the binarised
predictions (after a sigmoid) and incorporate them into the total loss
as an auxiliary term.  This makes training slower but helps achieve
precise boundaries.  Feel free to adjust weights of the terms in the
combined loss function.

Note:
    SciPy is required for computing Hausdorff and ASSD distances.  If
    SciPy is not available, the corresponding terms will be zero and a
    warning will be logged.
"""

from __future__ import annotations

import logging
from typing import Iterable, Tuple, List

import torch
import torch.nn.functional as F

try:
    import numpy as np
    from scipy.spatial.distance import directed_hausdorff
    from scipy.ndimage import morphology, distance_transform_edt
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    logging.warning(
        "SciPy is not available. HD95 and ASSD terms will be zero."
    )


def dice_loss(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Compute mean Dice loss over all channels.

    Args:
        pred: Tensor of shape (N,C,H,W) after sigmoid (range [0,1]).
        target: Tensor of shape (N,C,H,W) with binary ground truth.
        eps: Smoothing term to avoid division by zero.
    Returns:
        Dice loss averaged over channels and batch.
    """
    n, c, h, w = pred.shape
    pred_flat = pred.view(n, c, -1)
    target_flat = target.view(n, c, -1)
    intersection = (pred_flat * target_flat).sum(-1)
    denominator = pred_flat.sum(-1) + target_flat.sum(-1)
    dice = (2 * intersection + eps) / (denominator + eps)
    return 1 - dice.mean()


def _hd95_per_channel(pred_bin: np.ndarray, target_bin: np.ndarray) -> float:
    """
    Compute the 95th percentile Hausdorff distance for a single channel.

    Args:
        pred_bin: Binary 2D numpy array of prediction.
        target_bin: Binary 2D numpy array of ground truth.
    Returns:
        HD95 distance in pixels.  If both masks are empty returns 0.  If
        only one mask is empty returns inf.
    """
    pred_bin = pred_bin.astype(bool)
    target_bin = target_bin.astype(bool)

    # Handle empty masks
    if pred_bin.sum() == 0 and target_bin.sum() == 0:
        return 0.0
    if pred_bin.sum() == 0 or target_bin.sum() == 0:
        return float("inf")

    # Compute surface using logical XOR with erosion
    pred_eroded = morphology.binary_erosion(pred_bin)
    target_eroded = morphology.binary_erosion(target_bin)
    pred_surface = np.logical_xor(pred_bin, pred_eroded)
    target_surface = np.logical_xor(target_bin, target_eroded)

    pred_pts = np.stack(np.nonzero(pred_surface), axis=1)
    target_pts = np.stack(np.nonzero(target_surface), axis=1)

    # If both surfaces are empty, return 0
    if len(pred_pts) == 0 and len(target_pts) == 0:
        return 0.0
    # If one surface is empty, return inf
    if len(pred_pts) == 0 or len(target_pts) == 0:
        return float("inf")

    # Subsample large sets of surface points
    max_points = 1000
    if len(pred_pts) > max_points:
        idx = np.random.choice(len(pred_pts), max_points, replace=False)
        pred_pts = pred_pts[idx]
    if len(target_pts) > max_points:
        idx = np.random.choice(len(target_pts), max_points, replace=False)
        target_pts = target_pts[idx]

    # Compute distance transforms
    pred_dt = distance_transform_edt(~pred_bin)
    target_dt = distance_transform_edt(~target_bin)

    pred_to_gt = pred_dt[target_pts[:, 0], target_pts[:, 1]]
    gt_to_pred = target_dt[pred_pts[:, 0], pred_pts[:, 1]]
    all_dists = np.concatenate([pred_to_gt, gt_to_pred])

    return float(np.percentile(all_dists, 95))


def hd95_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Compute the mean HD95 across batch and channels.

    This loss is not differentiable.  It should be multiplied by a weight
    and added to other differentiable losses.  When SciPy is unavailable
    the loss returns zero.

    Args:
        pred: Tensor of shape (N,C,H,W) after sigmoid.
        target: Tensor of shape (N,C,H,W) binary ground truth.
    Returns:
        Scalar tensor representing mean HD95.
    """
    if not SCIPY_AVAILABLE:
        return pred.new_tensor(0.0)
    pred_np = pred.detach().cpu().numpy() > 0.5
    target_np = target.detach().cpu().numpy() > 0.5
    distances: List[float] = []
    for n in range(pred_np.shape[0]):
        for c in range(pred_np.shape[1]):
            d = _hd95_per_channel(pred_np[n, c], target_np[n, c])
            if d != float('inf'):
                distances.append(d)
    if not distances:
        return pred.new_tensor(0.0)
    return pred.new_tensor(sum(distances) / len(distances))


def assd(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    Compute Average Symmetric Surface Distance (ASSD) between masks.

    Like HD95, ASSD is a non‑differentiable metric computed on binary
    masks.  It measures the average distance between the surfaces of two
    binary objects.  If both masks are empty the distance is zero; if
    exactly one is empty it is infinite.  Surfaces are computed using
    logical XOR of the mask and its erosion.
    """
    if not SCIPY_AVAILABLE:
        return pred.new_tensor(0.0)
    pred_np = pred.detach().cpu().numpy() > 0.5
    target_np = target.detach().cpu().numpy() > 0.5
    distances: List[float] = []
    for n in range(pred_np.shape[0]):
        for c in range(pred_np.shape[1]):
            p = pred_np[n, c].astype(bool)
            t = target_np[n, c].astype(bool)
            if p.sum() == 0 and t.sum() == 0:
                continue
            if p.sum() == 0 or t.sum() == 0:
                distances.append(float("inf"))
                continue
            # Compute surfaces using XOR with erosion
            p_eroded = morphology.binary_erosion(p)
            t_eroded = morphology.binary_erosion(t)
            p_surface = np.logical_xor(p, p_eroded)
            t_surface = np.logical_xor(t, t_eroded)
            p_pts = np.stack(np.nonzero(p_surface), axis=1)
            t_pts = np.stack(np.nonzero(t_surface), axis=1)
            if len(p_pts) == 0 and len(t_pts) == 0:
                distances.append(0.0)
                continue
            if len(p_pts) == 0 or len(t_pts) == 0:
                distances.append(float("inf"))
                continue
            # Distance transforms
            p_dt = distance_transform_edt(~p)
            t_dt = distance_transform_edt(~t)
            p_to_t = p_dt[t_pts[:, 0], t_pts[:, 1]].mean()
            t_to_p = t_dt[p_pts[:, 0], p_pts[:, 1]].mean()
            distances.append((p_to_t + t_to_p) / 2.0)
    if not distances:
        return pred.new_tensor(0.0)
    finite_distances = [d for d in distances if np.isfinite(d)]
    if not finite_distances:
        return pred.new_tensor(0.0)
    return pred.new_tensor(sum(finite_distances) / len(finite_distances))


class CombinedLoss(torch.nn.Module):
    """Combined segmentation loss: Dice + HD95 + ASSD.

    The loss is computed as::

        total_loss = dice_weight * dice_loss
                     + hd_weight   * HD95
                     + assd_weight * ASSD

    Args:
        dice_weight: Weight for Dice loss.
        hd_weight: Weight for HD95 term.
        assd_weight: Weight for ASSD term.
    """

    def __init__(self, dice_weight: float = 1.0, hd_weight: float = 1.0, assd_weight: float = 1.0) -> None:
        super().__init__()
        self.dice_weight = dice_weight
        self.hd_weight = hd_weight
        self.assd_weight = assd_weight

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute combined loss.

        Args:
            logits: Raw network outputs of shape (N,C,H,W).
            target: Binary ground truth masks of shape (N,C,H,W).
        Returns:
            Scalar tensor representing the combined loss.
        """
        pred = torch.sigmoid(logits)
        loss = 0.0
        if self.dice_weight != 0:
            loss = loss + self.dice_weight * dice_loss(pred, target)
        if self.hd_weight != 0:
            hd = hd95_loss(pred, target)
            loss = loss + self.hd_weight * hd
        if self.assd_weight != 0:
            a = assd(pred, target)
            loss = loss + self.assd_weight * a
        return loss