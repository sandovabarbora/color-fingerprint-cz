"""Visual QA: render side-by-side image + extracted palette for a few frames.

This is the brief's "non-negotiable" sanity check. K-means on LAB can
produce surprising results — shadows dominate, accent colors collapse,
etc. Looking at the actual frames next to their extracted palettes is the
only way to know the upstream pipeline is honest.

Output: ``outputs/figures/sanity_{brand_id}.png`` (gitignored — local QA
only, not part of the public report).

Companion notebook: ``notebooks/01_sanity_check.ipynb`` is a one-cell
wrapper that calls ``main()``.

CLI:
    python -m src.sanity                       # 5 frames per brand
    python -m src.sanity --n 8 --seed 7
    python -m src.sanity --only kofola_menevice
"""

from __future__ import annotations

import argparse
import logging
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Rectangle

from src import config

logger = logging.getLogger(__name__)


def _load_frame(brand_id: str, frame_id: str) -> np.ndarray:
    """Load a frame as RGB uint8."""
    import cv2

    path = config.FRAMES_DIR / f"{frame_id}.jpg"
    bgr = cv2.imread(str(path))
    if bgr is None:
        raise FileNotFoundError(path)
    return bgr[:, :, ::-1]  # BGR -> RGB


def _draw_palette(ax: plt.Axes, palette_df: pd.DataFrame) -> None:
    """Draw weight-proportional horizontal swatches on ``ax``.

    Palette rows are pre-sorted by weight desc (cluster_idx=0 dominant).
    """
    palette = palette_df.sort_values("cluster_idx").reset_index(drop=True)
    x = 0.0
    for _, row in palette.iterrows():
        w = float(row["weight"])
        rgb = (
            int(row["rgb_r"]) / 255,
            int(row["rgb_g"]) / 255,
            int(row["rgb_b"]) / 255,
        )
        ax.add_patch(Rectangle((x, 0), w, 1, facecolor=rgb, edgecolor="white", linewidth=0.5))
        # Hex label if the band is wide enough to read
        if w > 0.08:
            luminance = 0.299 * rgb[0] + 0.587 * rgb[1] + 0.114 * rgb[2]
            text_color = "white" if luminance < 0.5 else "black"
            ax.text(
                x + w / 2,
                0.5,
                row["hex"],
                ha="center",
                va="center",
                fontsize=7,
                color=text_color,
                family="monospace",
            )
        x += w
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def render_brand_sanity(
    brand_id: str,
    palettes: pd.DataFrame,
    n_frames: int = 5,
    seed: int = 0,
    out_dir: Path = config.FIGURES_DIR,
) -> Path:
    """Render n_frames random (image, palette) pairs for one brand.

    Args:
        brand_id: Corpus id.
        palettes: The frames.parquet DataFrame (filtered or not).
        n_frames: How many frames to sample.
        seed: RNG seed for reproducible sampling.
        out_dir: Where to write the PNG.

    Returns:
        Path to the written PNG.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    brand_palettes = palettes[palettes["brand_id"] == brand_id]
    frame_ids = sorted(brand_palettes["frame_id"].unique())
    if not frame_ids:
        raise ValueError(f"No palettes for brand_id={brand_id}")

    rng = random.Random(seed)
    chosen = rng.sample(frame_ids, k=min(n_frames, len(frame_ids)))
    chosen.sort()  # display in temporal order

    fig, axes = plt.subplots(
        nrows=len(chosen),
        ncols=2,
        figsize=(10, 1.6 * len(chosen)),
        gridspec_kw={"width_ratios": [3, 2], "wspace": 0.05, "hspace": 0.25},
    )
    if len(chosen) == 1:
        axes = np.array([axes])

    fig.suptitle(f"{brand_id} — sanity check ({len(chosen)} random frames)", fontsize=11)

    for row_ax, frame_id in zip(axes, chosen, strict=True):
        img_ax, pal_ax = row_ax
        img = _load_frame(brand_id, frame_id)
        img_ax.imshow(img)
        img_ax.set_title(frame_id, fontsize=8, loc="left")
        img_ax.set_xticks([])
        img_ax.set_yticks([])

        palette_df = brand_palettes[brand_palettes["frame_id"] == frame_id]
        _draw_palette(pal_ax, palette_df)

    out_path = out_dir / f"sanity_{brand_id}.png"
    # tight_layout warns on Rectangle-patch axes; bbox_inches='tight' on
    # savefig handles the crop, so the warning is purely cosmetic.
    fig.subplots_adjust(top=0.95, bottom=0.02, left=0.02, right=0.98)
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return out_path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render per-brand sanity figures.")
    parser.add_argument("--only", type=str, default=None, help="Restrict to one brand id.")
    parser.add_argument("--n", type=int, default=5, help="Frames per brand.")
    parser.add_argument("--seed", type=int, default=0, help="RNG seed.")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    config.setup_logging(verbose=args.verbose)
    config.ensure_dirs()

    palettes = pd.read_parquet(config.FRAMES_PARQUET)
    brand_ids = (
        [args.only] if args.only else sorted(palettes["brand_id"].unique())
    )
    for brand_id in brand_ids:
        out = render_brand_sanity(
            brand_id, palettes, n_frames=args.n, seed=args.seed
        )
        logger.info("[%s] wrote %s", brand_id, out.name)


if __name__ == "__main__":
    main()
