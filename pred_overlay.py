import os
import re
from pathlib import Path

import cv2
import numpy as np


# =========================================================
# USER PATHS
# =========================================================
IMAGE_ROOT = "/home/yec23006/projects/research/KneeGrowthPlate/ZoneSeg/Input/Testdata/Image"
PRED_ROOT  = "//home/yec23006/projects/research/KneeGrowthPlate/ZoneSeg/Output/SemiSeparatedUnetFusion_32"
SAVE_ROOT  = "/home/yec23006/projects/research/KneeGrowthPlate/ZoneSeg/Output/SemiSeparatedUnetFusion_32/Overlay"

os.makedirs(SAVE_ROOT, exist_ok=True)


# =========================================================
# IMAGE FILE CANDIDATES
# =========================================================
SIGNAL_FILES = {
    "mineral":  ["mineral.png"],
    "ac":       ["ac_high.png", "ac.png"],
    "calcein":  ["calcein_high.png", "calcein.png", "calcein2.png"],
    "trap":     ["trap_high.png", "trap.png"],
    "dapi":     ["dapi_high.png", "dapi.png"],
    "ap":       ["ap_high.png", "ap.png", "ap_high2.png"],
    "edu":      ["edu_high.png", "edu.png"],
    "cfo":      ["cfo_high.png", "cfo.png"],
    "sfo":      ["sfo_high.png", "sfo.png"],
}


# =========================================================
# HELPERS
# =========================================================
def find_existing_image(sample_img_dir: Path, signal_name: str):
    for fname in SIGNAL_FILES[signal_name]:
        p = sample_img_dir / fname
        if p.exists():
            return p
    return None


def load_gray_or_rgb(path: Path):
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {path}")

    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    elif img.ndim == 3 and img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

    return img


def normalize_to_uint8(img):
    img = img.astype(np.float32)
    mn, mx = img.min(), img.max()
    if mx > mn:
        img = (img - mn) / (mx - mn)
    else:
        img = np.zeros_like(img, dtype=np.float32)
    img = (img * 255.0).clip(0, 255).astype(np.uint8)
    return img


def load_signal_as_bgr(sample_img_dir: Path, signal_name: str):
    p = find_existing_image(sample_img_dir, signal_name)
    if p is None:
        return None
    img = load_gray_or_rgb(p)
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    if img.shape[2] == 3:
        return normalize_to_uint8(img)
    return None


def resize_to_match(img, target_hw):
    h, w = target_hw
    if img.shape[:2] == (h, w):
        return img
    return cv2.resize(img, (w, h), interpolation=cv2.INTER_LINEAR)


def load_mask(mask_path: Path):
    m = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if m is None:
        raise FileNotFoundError(f"Cannot read mask: {mask_path}")
    return m


def extract_mask_key(mask_filename: str):
    stem = Path(mask_filename).stem.lower().strip()
    m = re.match(r"^([1-9][a-h])$", stem)
    if m:
        return m.group(1)
    return None


def make_two_signal_overlay_background(img1, img2, alpha1=0.5, alpha2=0.5):
    """
    Overlay two background images directly.

    Both images should already be BGR uint8 and same size.
    """
    if img1.shape[:2] != img2.shape[:2]:
        raise ValueError(f"Image shapes do not match: {img1.shape} vs {img2.shape}")

    out = cv2.addWeighted(
        img1, alpha1,
        img2, alpha2,
        0
    )
    return out


def get_background_for_mask(sample_img_dir: Path, mask_key: str, target_hw):
    """
    Return a BGR background image matching the biological logic.
    """

    if mask_key in ("9a", "9b"):
        sfo = load_signal_as_bgr(sample_img_dir, "sfo")
        cfo = load_signal_as_bgr(sample_img_dir, "cfo")
        if sfo is not None and cfo is not None:
            sfo = resize_to_match(sfo, target_hw)
            cfo = resize_to_match(cfo, target_hw)
            return make_two_signal_overlay_background(sfo, cfo, alpha1=0.5, alpha2=0.5)

    if mask_key in ("8e", "8f"):
        cfo = load_signal_as_bgr(sample_img_dir, "cfo")
        ap  = load_signal_as_bgr(sample_img_dir, "ap")
        if cfo is not None and ap is not None:
            cfo = resize_to_match(cfo, target_hw)
            ap  = resize_to_match(ap, target_hw)
            return make_two_signal_overlay_background(cfo, ap, alpha1=0.5, alpha2=0.5)

    if mask_key in ("6g", "6h", "8h"):
        ap   = load_signal_as_bgr(sample_img_dir, "ap")
        trap = load_signal_as_bgr(sample_img_dir, "trap")
        if ap is not None and trap is not None:
            ap   = resize_to_match(ap, target_hw)
            trap = resize_to_match(trap, target_hw)
            return make_two_signal_overlay_background(ap, trap, alpha1=0.5, alpha2=0.5)

    # regular backgrounds by first digit
    digit = mask_key[0]
    digit_to_signal = {
        "1": "mineral",
        "2": "ac",
        "3": "calcein",
        "4": "trap",
        "5": "dapi",
        "6": "ap",
        "7": "edu",
        "8": "cfo",
        "9": "sfo",
    }

    signal = digit_to_signal.get(digit, "mineral")
    bg = load_signal_as_bgr(sample_img_dir, signal)

    if bg is None:
        # fallback to mineral
        bg = load_signal_as_bgr(sample_img_dir, "mineral")

    if bg is None:
        raise RuntimeError(f"No usable background image found for sample: {sample_img_dir.name}")

    bg = resize_to_match(bg, target_hw)
    return bg


def overlay_mask_on_background(
    background_bgr,
    mask_gray,
    overlay_color=(0, 0, 255),   # red in BGR
    alpha_fill=0.28,
    boundary_color=(0, 255, 255),  # yellow in BGR
    boundary_thickness=1,
):
    """
    Makes both:
    - semi-transparent filled mask overlay
    - boundary overlay for thin lines
    """
    bg = background_bgr.copy()
    mask_bin = (mask_gray > 127).astype(np.uint8)

    # Filled overlay
    color_img = np.zeros_like(bg, dtype=np.uint8)
    color_img[:, :] = overlay_color
    filled = bg.copy()
    filled[mask_bin > 0] = cv2.addWeighted(
        bg[mask_bin > 0], 1.0 - alpha_fill,
        color_img[mask_bin > 0], alpha_fill,
        0
    )

    # Boundary overlay
    contours, _ = cv2.findContours(mask_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(filled, contours, -1, boundary_color, boundary_thickness)

    return filled


# =========================================================
# MAIN
# =========================================================
sample_dirs = sorted([p for p in Path(PRED_ROOT).iterdir() if p.is_dir()])

print(f"Found {len(sample_dirs)} sample folders in prediction root.")

for sample_pred_dir in sample_dirs:
    sample_name = sample_pred_dir.name
    sample_img_dir = Path(IMAGE_ROOT) / sample_name
    sample_save_dir = Path(SAVE_ROOT) / sample_name
    sample_save_dir.mkdir(parents=True, exist_ok=True)

    if not sample_img_dir.exists():
        print(f"[WARN] Missing image folder for sample: {sample_name}")
        continue

    mask_files = sorted(sample_pred_dir.glob("*.png"))
    if not mask_files:
        print(f"[WARN] No predicted masks found in: {sample_pred_dir}")
        continue

    print(f"Processing {sample_name} ...")

    for mask_path in mask_files:
        mask_key = extract_mask_key(mask_path.name)
        if mask_key is None:
            print(f"  [skip] Not a mask key file: {mask_path.name}")
            continue

        mask = load_mask(mask_path)
        h, w = mask.shape[:2]

        try:
            bg = get_background_for_mask(sample_img_dir, mask_key, (h, w))
        except Exception as e:
            print(f"  [WARN] Could not build background for {sample_name}/{mask_key}: {e}")
            continue

        overlay = overlay_mask_on_background(
            background_bgr=bg,
            mask_gray=mask,
            overlay_color=(0, 0, 255),      # red fill
            alpha_fill=0.28,
            boundary_color=(0, 255, 255),   # yellow edge
            boundary_thickness=1,
        )

        save_path = sample_save_dir / f"{mask_key}_overlay.png"
        cv2.imwrite(str(save_path), overlay)

print("Done.")