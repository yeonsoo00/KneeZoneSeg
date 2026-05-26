"""
Model definitions for knee growth plate segmentation.

This module implements a lightweight yet flexible UNet‑like architecture
with optional feature fusion between the encoder and decoder.  The goal
is to capture both local detail and global context necessary to delineate
thin zones in histology images.  The model accepts a variable number of
input channels (one per stain) and outputs a tensor with one channel
per segmentation class.

The architecture consists of an encoder built from a sequence of
``ConvBlock`` modules that downsample the spatial resolution and a decoder
that upsamples while concatenating skip connections from the encoder.  A
``FusionBlock`` optionally fuses features from multiple scales before the
final prediction.  You can adjust the depth and width of the network
through the ``channels`` parameter.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    """A basic convolutional block: two conv layers with batch norm and ReLU."""
    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 3, padding: int = 1) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size, padding=padding, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size, padding=padding, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class Down(nn.Module):
    """Downsampling block: max pool followed by a conv block."""
    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.pool = nn.MaxPool2d(2)
        self.conv = ConvBlock(in_ch, out_ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(x)
        return self.conv(x)


class Up(nn.Module):
    """Upsampling block: transposed conv followed by a conv block."""
    def __init__(self, in_ch: int, out_ch: int, bilinear: bool = True) -> None:
        super().__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = ConvBlock(in_ch, out_ch)
        else:
            self.up = nn.ConvTranspose2d(in_ch // 2, in_ch // 2, kernel_size=2, stride=2)
            self.conv = ConvBlock(in_ch, out_ch)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        # x1: decoder input, x2: skip connection
        x1 = self.up(x1)
        # Pad x1 if necessary
        diff_y = x2.size()[2] - x1.size()[2]
        diff_x = x2.size()[3] - x1.size()[3]
        x1 = nn.functional.pad(x1, [diff_x // 2, diff_x - diff_x // 2,
                                    diff_y // 2, diff_y - diff_y // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class FusionBlock(nn.Module):
    """Feature fusion block for aggregating multi‑scale features.

    This block takes the outputs from each decoder stage and reduces them
    to a common number of channels before summing and convolving.  The
    intuition is to merge coarse and fine information prior to prediction.
    """
    def __init__(self, in_channels: Iterable[int], out_ch: int) -> None:
        super().__init__()
        self.reducers = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(ch, out_ch, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True)
            ) for ch in in_channels
        ])
        self.out_conv = nn.Sequential(
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, features: Tuple[torch.Tensor, ...]) -> torch.Tensor:
        # Resize all features to the spatial size of the largest one
        target_size = features[0].size()[2:]
        fused = 0
        for feat, reducer in zip(features, self.reducers):
            if feat.size()[2:] != target_size:
                feat = nn.functional.interpolate(feat, size=target_size, mode='bilinear', align_corners=True)
            fused = fused + reducer(feat)
        return self.out_conv(fused)


class UNetFusion(nn.Module):
    """UNet architecture with optional feature fusion and configurable depth."""
    def __init__(self, n_channels: int, n_classes: int, base_ch: int = 32, bilinear: bool = True, use_fusion: bool = True) -> None:
        super().__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        self.use_fusion = use_fusion

        # Encoder
        self.inc = ConvBlock(n_channels, base_ch)
        self.down1 = Down(base_ch, base_ch * 2)
        self.down2 = Down(base_ch * 2, base_ch * 4)
        self.down3 = Down(base_ch * 4, base_ch * 8)
        self.down4 = Down(base_ch * 8, base_ch * 8)

        # Decoder
        self.up1 = Up(base_ch * 16, base_ch * 4, bilinear)
        self.up2 = Up(base_ch * 8, base_ch * 2, bilinear)
        self.up3 = Up(base_ch * 4, base_ch, bilinear)
        self.up4 = Up(base_ch * 2, base_ch, bilinear)

        # Optional fusion of decoder outputs
        if use_fusion:
            self.fusion = FusionBlock(
                in_channels=[base_ch * 4, base_ch * 2, base_ch, base_ch],
                out_ch=base_ch
            )
            self.outc = nn.Conv2d(base_ch, n_classes, kernel_size=1)
        else:
            self.outc = nn.Conv2d(base_ch, n_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        x = self.up1(x5, x4)
        dec1 = x
        x = self.up2(x, x3)
        dec2 = x
        x = self.up3(x, x2)
        dec3 = x
        x = self.up4(x, x1)
        dec4 = x

        if self.use_fusion:
            fused = self.fusion((dec1, dec2, dec3, dec4))
            return self.outc(fused)
        else:
            return self.outc(dec4)


# -----------------------------------------------------------------------------
# Multi-head UNet for stain-specific and cross-stain segmentation
#
# This model extends the basic UNet with separate output heads for each
# stain group (e.g. mineral, AC, Calcein, TRAP) and for certain cross-stain
# pairs.  Each head predicts only the masks relevant to its group.  For
# cross-stain groups, a simple gating mechanism modulates the fused features
# using the relevant input channels (e.g. SFO and CFO for 9a/9b).
#
# The network shares a common backbone (encoder + decoder) to extract
# high-level features from all input channels.  After feature extraction,
# single-head outputs are computed via 1x1 convolutions.  Cross-heads first
# generate a gating mask from the specified input channels, then apply this
# mask to the fused features before the final 1x1 convolution.  This gating
# encourages the network to focus on regions where the relevant signals are
# present.

class SingleHead(nn.Module):
    """Segmentation head for a single stain group.

    Args:
        in_ch: Number of input channels (from the fused UNet features).
        out_ch: Number of output classes for this group.
    """
    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=1)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.conv(features)


class CrossHead(nn.Module):
    """Segmentation head for a cross-stain group with simple gating.

    This head produces predictions for a group of classes that depend on
    multiple input stains (e.g. SFO+CFO).  It uses a learnable gating
    mechanism to modulate the fused features based on the relevant input
    channels.

    Args:
        in_ch: Number of input channels (from the fused UNet features).
        out_ch: Number of output classes for this group.
        gating_in_ch: Number of channels used to compute the gating mask.
    """
    def __init__(self, in_ch: int, out_ch: int, gating_in_ch: int) -> None:
        super().__init__()
        # Convolution to compute a gating mask from the relevant input signals
        self.gating_conv = nn.Sequential(
            nn.Conv2d(gating_in_ch, 1, kernel_size=1),
            nn.Sigmoid()
        )
        # Final convolution to produce class logits
        self.out_conv = nn.Conv2d(in_ch, out_ch, kernel_size=1)

    def forward(self, features: torch.Tensor, gating_input: torch.Tensor) -> torch.Tensor:
        # gating_input: (N, gating_in_ch, H, W) but H/W may differ from features
        # Compute a gating mask of shape (N,1,hg,wg)
        gating_mask = self.gating_conv(gating_input)
        # Align gating mask to features spatial dimensions and orientation
        # features shape: (N, C, H_f, W_f)
        _, _, H_f, W_f = features.shape
        _, _, H_m, W_m = gating_mask.shape
        # If gating mask appears transposed (i.e., height matches feature width and vice versa), transpose it
        if H_m == W_f and W_m == H_f:
            gating_mask = gating_mask.transpose(2, 3)
        # Resize gating mask to match features if needed
        if gating_mask.shape[2:] != (H_f, W_f):
            # interpolate to match feature size
            gating_mask = nn.functional.interpolate(gating_mask, size=(H_f, W_f), mode='bilinear', align_corners=False)
        # Apply gating mask
        gated = features * gating_mask
        return self.out_conv(gated)


class MultiHeadUNet(nn.Module):
    """
    [Deprecated] UNet-based model with multiple output heads.

    This class is retained for backward compatibility but is no longer
    used in the latest training scripts.  See ``MultiBranchUNet`` for
    the preferred architecture.
    """

    def __init__(
        self,
        n_channels: int,
        group_out_channels: Dict[str, int],
        cross_groups: Dict[str, Tuple[int, List[int]]],
        base_ch: int = 32,
        use_fusion: bool = True,
        bilinear: bool = True,
    ) -> None:
        super().__init__()
        self.n_channels = n_channels
        self.group_out_channels = group_out_channels
        self.cross_groups = cross_groups
        self.use_fusion = use_fusion

        # Backbone UNet: returns fused features of shape (N, base_ch, H, W)
        self.backbone = UNetFusion(
            n_channels,
            n_classes=base_ch,
            base_ch=base_ch,
            bilinear=bilinear,
            use_fusion=use_fusion,
        )

        # Heads for single-signal groups
        self.single_heads = nn.ModuleDict()
        for group, out_ch in group_out_channels.items():
            self.single_heads[group] = SingleHead(base_ch, out_ch)

        # Heads for cross-signal groups
        self.cross_heads = nn.ModuleDict()
        for group, (out_ch, gating_channels) in cross_groups.items():
            self.cross_heads[group] = CrossHead(base_ch, out_ch, len(gating_channels))
            # Store gating channel indices for this group
            self.cross_heads[group].gating_indices = gating_channels

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        # x shape: (N, C, H, W)
        # Get fused features from backbone
        # The UNetFusion returns logits for base_ch classes; we treat those as features
        features = self.backbone(x)  # (N, base_ch, H, W)

        outputs: Dict[str, torch.Tensor] = {}
        # Single-head predictions
        for group, head in self.single_heads.items():
            outputs[group] = head(features)
        # Cross-head predictions
        for group, head in self.cross_heads.items():
            # Extract gating input channels from x using stored indices
            idxs = head.gating_indices  # type: ignore[attr-defined]
            gating_input = x[:, idxs, :, :]
            outputs[group] = head(features, gating_input)
        return outputs


# -----------------------------------------------------------------------------
# Multi-branch UNet architecture
#
# In this variant each task family (e.g. masks derived from one stain or a
# specific fusion of stains) has its own decoder/branch.  A shared encoder
# extracts low-level features from all input channels; each branch then
# performs its own upsampling and produces its outputs.  Cross-stain branches
# use a gating mechanism to modulate the decoder features using the relevant
# input channels (e.g. CFO+SFO, CFO+AP, AP+TRAP).

class SharedEncoder(nn.Module):
    """Shared encoder that produces multi-scale feature maps.

    This module mirrors the downsampling path of ``UNetFusion`` but returns
    all intermediate feature maps required for skip connections.  The
    encoder operates on the full set of input channels and is shared by
    all branches.

    Args:
        n_channels: Number of input channels (stains + optional hue/saturation).
        base_ch: Base number of channels for the first convolution.
        bilinear: If True, use bilinear upsampling in corresponding decoders.
    """

    def __init__(self, n_channels: int, base_ch: int = 32, bilinear: bool = True) -> None:
        super().__init__()
        self.inc = ConvBlock(n_channels, base_ch)
        self.down1 = Down(base_ch, base_ch * 2)
        self.down2 = Down(base_ch * 2, base_ch * 4)
        self.down3 = Down(base_ch * 4, base_ch * 8)
        self.down4 = Down(base_ch * 8, base_ch * 8)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        return x1, x2, x3, x4, x5


class BranchDecoder(nn.Module):
    """Decoder for a single branch.

    Given encoder feature maps, this module upsamples the deepest features
    back to full resolution using four successive ``Up`` blocks that mirror
    the encoder.  The final output has ``base_ch`` channels and serves as
    the input to the branch head.

    Args:
        base_ch: Base channel width used by the shared encoder.
        bilinear: Whether to use bilinear upsampling in each ``Up`` block.
    """

    def __init__(self, base_ch: int = 32, bilinear: bool = True) -> None:
        super().__init__()
        # Channel dimensions follow the UNetFusion structure:
        # x1: base_ch, x2: 2*base_ch, x3: 4*base_ch, x4: 8*base_ch, x5: 8*base_ch
        # Up blocks double the spatial size and reduce channels accordingly.
        self.up1 = Up(base_ch * 16, base_ch * 4, bilinear)
        self.up2 = Up(base_ch * 8, base_ch * 2, bilinear)
        self.up3 = Up(base_ch * 4, base_ch, bilinear)
        self.up4 = Up(base_ch * 2, base_ch, bilinear)

    def forward(self, features: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]) -> torch.Tensor:
        x1, x2, x3, x4, x5 = features
        # x5: (N, 8*base_ch, H/16, W/16)
        x = self.up1(x5, x4)  # -> (N, 4*base_ch, H/8, W/8)
        x = self.up2(x, x3)   # -> (N, 2*base_ch, H/4, W/4)
        x = self.up3(x, x2)   # -> (N, base_ch, H/2, W/2)
        x = self.up4(x, x1)   # -> (N, base_ch, H, W)
        return x


class BranchHead(nn.Module):
    """Segmentation head for a branch without gating.

    This head applies a 1×1 convolution to map the branch decoder output
    (``base_ch`` channels) to the desired number of output channels for
    that branch.

    Args:
        base_ch: Number of channels in the branch decoder output.
        out_ch: Number of output channels (i.e. number of masks) for this branch.
    """

    def __init__(self, base_ch: int, out_ch: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(base_ch, out_ch, kernel_size=1)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.conv(features)


class CrossBranchHead(nn.Module):
    """Segmentation head for a cross-stain branch with gating.

    The gating mechanism modulates the branch decoder output using a
    per-pixel mask derived from a subset of the input channels.  The mask
    suppresses activations where the relevant signals are absent.

    Args:
        base_ch: Number of channels in the branch decoder output.
        out_ch: Number of output channels (i.e. number of masks) for this branch.
        gating_in_ch: Number of input channels used to compute the gating mask.
    """

    def __init__(self, base_ch: int, out_ch: int, gating_in_ch: int) -> None:
        super().__init__()
        self.gating_conv = nn.Sequential(
            nn.Conv2d(gating_in_ch, 1, kernel_size=1),
            nn.Sigmoid()
        )
        self.conv = nn.Conv2d(base_ch, out_ch, kernel_size=1)
        # We store gating indices externally on the head instance
        self.gating_indices: List[int] = []

    def forward(self, features: torch.Tensor, gating_input: torch.Tensor) -> torch.Tensor:
        """Forward pass for a cross-stain branch.

        Args:
            features: Decoder output tensor of shape (N, base_ch, H, W).
            gating_input: Subset of the original input tensor of shape
                (N, gating_in_ch, H_in, W_in).  Typically this is a selection
                of input channels corresponding to the relevant stains.

        Returns:
            Tensor of shape (N, out_ch, H, W) representing per-mask logits.
        """
        # Compute gating mask (N,1,hg,wg)
        gating_mask = self.gating_conv(gating_input)
        # Align gating mask to feature map spatial dimensions and orientation
        H_f, W_f = features.shape[2], features.shape[3]
        _, _, H_m, W_m = gating_mask.shape
        # If the gating mask appears transposed relative to the feature map,
        # transpose it.  This handles cases where input images have swapped
        # axes (e.g. due to inconsistent file naming).
        if H_m == W_f and W_m == H_f:
            gating_mask = gating_mask.transpose(2, 3)
        # Resize gating mask to match features if necessary
        if gating_mask.shape[2:] != (H_f, W_f):
            gating_mask = nn.functional.interpolate(
                gating_mask,
                size=(H_f, W_f),
                mode='bilinear',
                align_corners=False
            )
        # Apply gating
        gated = features * gating_mask
        return self.conv(gated)


class MultiBranchUNet(nn.Module):
    """UNet-based model with separate decoder branches for each task family.

    This architecture decouples the upsampling path for different groups of
    masks.  All input channels are processed by a shared encoder to extract
    hierarchical features.  Each branch then upsamples these features
    independently and produces its own segmentation masks.  Cross-stain
    branches can leverage a gating mechanism to focus on the relevant
    combination of stains.

    Args:
        n_channels: Number of input channels (including all stains and
            optional hue/saturation channels).
        branches: Dict mapping branch names to the number of output
            masks for that branch.  Each branch corresponds to a family
            of masks (e.g. all Mineral masks, all AC masks) or a cross
            family (e.g. SFO+CFO).
        gating_info: Dict mapping cross-branch names to a list of
            gating channel indices.  The indices refer to the input
            channels used to compute the gating mask.  If a branch is
            not present in this dict, it is treated as a single branch
            without gating.
        base_ch: Base channel width for the encoder/decoder.
        bilinear: Whether to use bilinear upsampling.
    """

    def __init__(
        self,
        n_channels: int,
        branches: Dict[str, int],
        gating_info: Dict[str, List[int]],
        base_ch: int = 32,
        bilinear: bool = True,
    ) -> None:
        super().__init__()
        self.n_channels = n_channels
        self.branches = branches
        self.gating_info = gating_info

        # Shared encoder
        self.encoder = SharedEncoder(n_channels, base_ch=base_ch, bilinear=bilinear)

        # For each branch, create its own decoder and head
        self.branch_decoders = nn.ModuleDict()
        self.branch_heads = nn.ModuleDict()
        for name, out_ch in branches.items():
            # Decoder
            self.branch_decoders[name] = BranchDecoder(base_ch=base_ch, bilinear=bilinear)
            # Head: cross or single
            if name in gating_info:
                gating_in_ch = len(gating_info[name])
                head = CrossBranchHead(base_ch, out_ch, gating_in_ch)
                # store gating indices on the head for use in forward
                head.gating_indices = gating_info[name]
            else:
                head = BranchHead(base_ch, out_ch)
            self.branch_heads[name] = head

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        # Compute encoder features once for all branches
        features = self.encoder(x)  # tuple of (x1,x2,x3,x4,x5)
        outputs: Dict[str, torch.Tensor] = {}
        for name in self.branches:
            # Decode features for this branch
            dec_feats = self.branch_decoders[name](features)
            head = self.branch_heads[name]
            # Cross-stain branch if head has gating
            if isinstance(head, CrossBranchHead):
                # Extract gating input channels from x using stored indices
                idxs = head.gating_indices  # type: ignore[attr-defined]
                gating_input = x[:, idxs, :, :]
                outputs[name] = head(dec_feats, gating_input)
            else:
                outputs[name] = head(dec_feats)
        return outputs