"""Per-frame color palette extraction via K-means in LAB color space.

For each frame, run K-means (k=5) on the LAB-converted pixel cloud to
find dominant color cluster centroids. Output a long-form DataFrame with
one row per (frame, cluster), stored to ``data/processed/frames.parquet``.

LAB is chosen because Euclidean distance in LAB approximates perceptual
color difference, so K-means clusters group visually-similar colors.
References:
  - cv2.cvtColor: https://docs.opencv.org/4.x/de/d25/imgproc_color_conversions.html
  - sklearn.cluster.KMeans: https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html

Stored LAB values are in **canonical scale** (L: 0-100, a/b: ~-128..127),
not OpenCV's 0-255 packed encoding. See ``_opencv_lab_to_canonical``.

CLI:
    python -m src.color
    python -m src.color --only kofola_menevice
"""

from __future__ import annotations

import argparse
import logging
import warnings
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.exceptions import ConvergenceWarning

from src import config
from src.fetch import BrandSpec, load_corpus

# Frames dominated by a single color (logo cards, full-screen text) yield
# fewer distinct pixels than k, triggering this warning. KMeans still
# returns k centroids and weights remain valid, so silence the noise.
warnings.filterwarnings("ignore", category=ConvergenceWarning)

logger = logging.getLogger(__name__)

# Max edge length after downsampling. 200 was empirically enough to preserve
# K-means centroids vs full resolution; smaller is faster, bigger is wasteful.
DOWNSAMPLE_MAX_EDGE: int = 200

# Brightness floor on the *canonical* L channel (0-100). Frames with mean L
# below this are treated as near-black and skipped. Brief uses an
# OpenCV-scaled threshold of 20; 20/255*100 ≈ 7.84.
MIN_CANONICAL_L: float = 7.84


def _downsample(bgr: np.ndarray, max_edge: int = DOWNSAMPLE_MAX_EDGE) -> np.ndarray:
    """Resize image so its longest edge is at most ``max_edge`` pixels."""
    h, w = bgr.shape[:2]
    longest = max(h, w)
    if longest <= max_edge:
        return bgr
    scale = max_edge / longest
    new_size = (round(w * scale), round(h * scale))
    # INTER_AREA is the correct choice for shrinking — averages pixel
    # neighbourhoods rather than picking single representatives.
    return cv2.resize(bgr, new_size, interpolation=cv2.INTER_AREA)


def _opencv_lab_to_canonical(lab_opencv: np.ndarray) -> np.ndarray:
    """Convert OpenCV's 8-bit LAB encoding to canonical CIE-LAB.

    OpenCV stores 8-bit LAB as: L in [0,255] (representing 0-100),
    a in [0,255] (offset by 128), b in [0,255] (offset by 128).
    Canonical: L in [0,100], a in [~-128,127], b in [~-128,127].
    """
    out = lab_opencv.astype(np.float32)
    out[..., 0] *= 100.0 / 255.0
    out[..., 1] -= 128.0
    out[..., 2] -= 128.0
    return out


def _canonical_lab_to_rgb(lab_canonical: np.ndarray) -> np.ndarray:
    """Convert canonical LAB back to 8-bit RGB.

    Inverts ``_opencv_lab_to_canonical`` then cv2.COLOR_LAB2BGR then BGR->RGB.
    Accepts an Nx3 array, returns Nx3 uint8 RGB.
    """
    lab_opencv = lab_canonical.copy()
    lab_opencv[..., 0] *= 255.0 / 100.0
    lab_opencv[..., 1] += 128.0
    lab_opencv[..., 2] += 128.0
    lab_opencv = np.clip(lab_opencv, 0, 255).astype(np.uint8)
    # cvtColor wants a 2D image (H, W, 3); reshape Nx3 -> 1xNx3
    as_image = lab_opencv.reshape(1, -1, 3)
    bgr = cv2.cvtColor(as_image, cv2.COLOR_LAB2BGR).reshape(-1, 3)
    return bgr[:, ::-1].copy()  # BGR -> RGB


def _rgb_to_hex(rgb: np.ndarray) -> list[str]:
    """Convert an Nx3 uint8 RGB array to a list of '#RRGGBB' strings."""
    return [f"#{r:02X}{g:02X}{b:02X}" for r, g, b in rgb]


def extract_palette(
    image_path: Path,
    brand_id: str,
    *,
    k: int = config.KMEANS_K,
    random_state: int = config.RANDOM_STATE,
    n_init: int = config.KMEANS_N_INIT,
) -> pd.DataFrame | None:
    """Extract a k-cluster LAB palette for one frame.

    Args:
        image_path: Path to a JPEG frame.
        brand_id: Corpus id the frame belongs to (stored as a column).
        k: Number of K-means clusters.
        random_state: Seed for reproducibility.
        n_init: KMeans n_init passes.

    Returns:
        DataFrame with one row per cluster (k rows) and columns:
            frame_id, brand_id, cluster_idx, weight,
            lab_l, lab_a, lab_b, rgb_r, rgb_g, rgb_b, hex
        Returns None if the frame is too dark (below MIN_CANONICAL_L).

    Raises:
        FileNotFoundError: If the image cannot be read.
    """
    bgr = cv2.imread(str(image_path))
    if bgr is None:
        raise FileNotFoundError(f"cv2 could not read {image_path}")

    bgr_small = _downsample(bgr)
    lab_opencv = cv2.cvtColor(bgr_small, cv2.COLOR_BGR2LAB)
    lab_canonical = _opencv_lab_to_canonical(lab_opencv)
    pixels = lab_canonical.reshape(-1, 3)

    mean_l = float(pixels[:, 0].mean())
    if mean_l < MIN_CANONICAL_L:
        logger.debug("[%s] %s: mean L %.2f < %.2f, skip", brand_id, image_path.name, mean_l, MIN_CANONICAL_L)
        return None

    kmeans = KMeans(
        n_clusters=k,
        random_state=random_state,
        n_init=n_init,
    ).fit(pixels)

    labels = kmeans.labels_
    centroids = kmeans.cluster_centers_  # shape (k, 3) in canonical LAB
    counts = np.bincount(labels, minlength=k).astype(np.float64)
    weights = counts / counts.sum()

    # Sort by weight desc so cluster_idx 0 is always the most prevalent.
    order = np.argsort(weights)[::-1]
    centroids = centroids[order]
    weights = weights[order]

    rgb = _canonical_lab_to_rgb(centroids)
    hex_codes = _rgb_to_hex(rgb)

    return pd.DataFrame(
        {
            "frame_id": image_path.stem,
            "brand_id": brand_id,
            "cluster_idx": np.arange(k, dtype=np.int32),
            "weight": weights.astype(np.float32),
            "lab_l": centroids[:, 0].astype(np.float32),
            "lab_a": centroids[:, 1].astype(np.float32),
            "lab_b": centroids[:, 2].astype(np.float32),
            "rgb_r": rgb[:, 0].astype(np.uint8),
            "rgb_g": rgb[:, 1].astype(np.uint8),
            "rgb_b": rgb[:, 2].astype(np.uint8),
            "hex": hex_codes,
        }
    )


def extract_all_palettes(
    specs: list[BrandSpec], *, only: str | None = None
) -> pd.DataFrame:
    """Run palette extraction across every frame of every (selected) brand.

    Args:
        specs: Corpus rows.
        only: If set, restrict to a single id.

    Returns:
        Concatenated long-form DataFrame across all frames.
    """
    rows: list[pd.DataFrame] = []
    for spec in specs:
        if only and spec.id != only:
            continue
        frame_paths = sorted(config.FRAMES_DIR.glob(f"{spec.id}_*.jpg"))
        if not frame_paths:
            logger.warning("[%s] no frames found in %s", spec.id, config.FRAMES_DIR)
            continue
        n_kept = 0
        for fp in frame_paths:
            df = extract_palette(fp, spec.id)
            if df is None:
                continue
            rows.append(df)
            n_kept += 1
        logger.info("[%s] palettes for %d / %d frames", spec.id, n_kept, len(frame_paths))

    if not rows:
        return pd.DataFrame(
            columns=[
                "frame_id",
                "brand_id",
                "cluster_idx",
                "weight",
                "lab_l",
                "lab_a",
                "lab_b",
                "rgb_r",
                "rgb_g",
                "rgb_b",
                "hex",
            ]
        )
    return pd.concat(rows, ignore_index=True)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract LAB K-means palettes per frame.")
    parser.add_argument(
        "--only", type=str, default=None, help="Restrict to a single corpus id."
    )
    parser.add_argument(
        "--verbose", action="store_true", help="Emit DEBUG-level logs."
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    config.setup_logging(verbose=args.verbose)
    config.ensure_dirs()
    specs = load_corpus()
    df = extract_all_palettes(specs, only=args.only)
    df.to_parquet(config.FRAMES_PARQUET, index=False)
    logger.info(
        "Wrote %d palette rows (%d frames, %d brands) -> %s",
        len(df),
        df["frame_id"].nunique() if len(df) else 0,
        df["brand_id"].nunique() if len(df) else 0,
        config.FRAMES_PARQUET,
    )


if __name__ == "__main__":
    main()
