"""Per-brand color-fingerprint metrics derived from frame palettes.

For each brand, aggregate three metrics across all its frames:

1. **Palette diversity** — expected pairwise LAB distance between two
   pixels sampled by cluster prevalence. Higher = broader color range.
2. **Saturation arc** — weighted mean and std of saturation across
   frames (cluster-weighted within each frame, then mean/std across
   frames). The std captures the "tension" of the spot — boring spots
   are flat, dramatic ones swing.
3. **Brand color anchor** — fraction of total cluster weight whose hue
   lies within ±15° of the brand's anchor color. Measures how much of
   the screen-time the brand color actually occupies.

Outputs ``data/processed/campaigns.parquet``.

CLI:
    python -m src.metrics
"""

from __future__ import annotations

import argparse
import colorsys
import logging

import numpy as np
import pandas as pd

from src import config
from src.fetch import BrandSpec, load_corpus

logger = logging.getLogger(__name__)


# --- Helpers ---------------------------------------------------------------


def _hex_to_rgb01(hex_color: str) -> tuple[float, float, float]:
    """'#RRGGBB' -> (r, g, b) in [0,1]."""
    s = hex_color.lstrip("#")
    return tuple(int(s[i : i + 2], 16) / 255.0 for i in (0, 2, 4))  # type: ignore[return-value]


def _rgb01_to_hue_deg(rgb: tuple[float, float, float]) -> float:
    """RGB in [0,1] -> hue in [0, 360)."""
    h, _s, _v = colorsys.rgb_to_hsv(*rgb)
    return h * 360.0


def _rgb01_to_sat(rgb: tuple[float, float, float]) -> float:
    """RGB in [0,1] -> HSV saturation in [0, 1]."""
    _h, s, _v = colorsys.rgb_to_hsv(*rgb)
    return s


def _hue_circular_distance(h1: float, h2: float) -> float:
    """Smallest angular distance on the hue circle, in degrees."""
    d = abs(h1 - h2) % 360.0
    return d if d <= 180.0 else 360.0 - d


# --- Metric: palette diversity --------------------------------------------


def palette_diversity(palette_df: pd.DataFrame) -> float:
    """Expected pairwise LAB distance between two cluster-weighted pixels.

    Args:
        palette_df: K rows with columns lab_l, lab_a, lab_b, weight.

    Returns:
        Σᵢⱼ wᵢwⱼ ‖cᵢ-cⱼ‖₂ — 0 for a single-color image, larger for
        varied palettes. Typical values 20-80 in canonical LAB.
    """
    centroids = palette_df[["lab_l", "lab_a", "lab_b"]].to_numpy(dtype=np.float64)
    weights = palette_df["weight"].to_numpy(dtype=np.float64)
    diffs = centroids[:, None, :] - centroids[None, :, :]  # (k, k, 3)
    dists = np.linalg.norm(diffs, axis=-1)  # (k, k)
    w_outer = weights[:, None] * weights[None, :]  # (k, k)
    return float((w_outer * dists).sum())


# --- Metric: saturation arc ------------------------------------------------


def frame_saturation(palette_df: pd.DataFrame) -> float:
    """Cluster-weight-weighted mean HSV saturation for one frame.

    Saturation is computed in HSV from each centroid's RGB, then
    averaged with weights = cluster weights. Result in [0, 1].
    """
    weights = palette_df["weight"].to_numpy(dtype=np.float64)
    sats = np.array(
        [
            _rgb01_to_sat((int(r) / 255, int(g) / 255, int(b) / 255))
            for r, g, b in palette_df[["rgb_r", "rgb_g", "rgb_b"]].to_numpy()
        ],
        dtype=np.float64,
    )
    return float((weights * sats).sum())


# --- Metric: brand anchor --------------------------------------------------


def brand_anchor_share(
    palettes: pd.DataFrame,
    anchor_hex: str,
    tolerance_deg: float = config.BRAND_ANCHOR_HUE_TOLERANCE_DEG,
) -> float:
    """Fraction of total cluster weight whose hue is within ``tolerance_deg``
    of the anchor color's hue.

    Args:
        palettes: All palette rows for one brand.
        anchor_hex: Brand anchor color, e.g. ``"#FFC72C"``.
        tolerance_deg: Circular hue tolerance in degrees.

    Returns:
        anchor_share in [0, 1]. Total cluster weight per brand =
        number of frames (each frame sums to 1.0), so dividing by
        n_frames gives a fraction.
    """
    anchor_hue = _rgb01_to_hue_deg(_hex_to_rgb01(anchor_hex))
    hues = np.array(
        [
            _rgb01_to_hue_deg((int(r) / 255, int(g) / 255, int(b) / 255))
            for r, g, b in palettes[["rgb_r", "rgb_g", "rgb_b"]].to_numpy()
        ],
        dtype=np.float64,
    )
    dists = np.array(
        [_hue_circular_distance(h, anchor_hue) for h in hues],
        dtype=np.float64,
    )
    mask = dists <= tolerance_deg
    total_weight = float(palettes["weight"].sum())
    if total_weight == 0:
        return 0.0
    return float(palettes["weight"].to_numpy(dtype=np.float64)[mask].sum() / total_weight)


# --- Aggregation -----------------------------------------------------------


def compute_brand_metrics(
    brand_id: str,
    spec: BrandSpec,
    palettes: pd.DataFrame,
) -> dict[str, object]:
    """Aggregate the three metrics for one brand.

    Args:
        brand_id: Corpus id.
        spec: Corresponding BrandSpec (for sector + anchor lookup).
        palettes: frames.parquet rows for this brand_id only.

    Returns:
        A flat dict ready to become a DataFrame row.
    """
    frame_ids = sorted(palettes["frame_id"].unique())
    per_frame_diversity: list[float] = []
    per_frame_saturation: list[float] = []
    for fid in frame_ids:
        sub = palettes[palettes["frame_id"] == fid]
        per_frame_diversity.append(palette_diversity(sub))
        per_frame_saturation.append(frame_saturation(sub))

    diversity = float(np.mean(per_frame_diversity)) if per_frame_diversity else 0.0
    sat_mean = float(np.mean(per_frame_saturation)) if per_frame_saturation else 0.0
    sat_std = float(np.std(per_frame_saturation)) if per_frame_saturation else 0.0
    anchor = brand_anchor_share(palettes, spec.anchor_hex)

    return {
        "brand_id": brand_id,
        "brand": spec.brand,
        "sector": spec.sector,
        "anchor_hex": spec.anchor_hex,
        "n_frames": len(frame_ids),
        "diversity": diversity,
        "sat_mean": sat_mean,
        "sat_std": sat_std,
        "brand_anchor": anchor,
    }


def compute_all(specs: list[BrandSpec], palettes: pd.DataFrame) -> pd.DataFrame:
    """Compute per-brand metrics for every spec present in palettes.

    Returns:
        DataFrame with one row per brand_id. Brands absent from
        ``palettes`` are skipped with a WARNING.
    """
    rows: list[dict[str, object]] = []
    for spec in specs:
        sub = palettes[palettes["brand_id"] == spec.id]
        if sub.empty:
            logger.warning("[%s] no palette rows — skip", spec.id)
            continue
        row = compute_brand_metrics(spec.id, spec, sub)
        rows.append(row)
        logger.info(
            "[%s] diversity=%.1f sat_mean=%.3f sat_std=%.3f anchor_share=%.3f n_frames=%d",
            spec.id,
            row["diversity"],
            row["sat_mean"],
            row["sat_std"],
            row["brand_anchor"],
            row["n_frames"],
        )
    return pd.DataFrame(rows)


# --- CLI -------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute per-brand fingerprint metrics.")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    config.setup_logging(verbose=args.verbose)
    config.ensure_dirs()
    specs = load_corpus()
    palettes = pd.read_parquet(config.FRAMES_PARQUET)
    metrics = compute_all(specs, palettes)
    metrics.to_parquet(config.CAMPAIGNS_PARQUET, index=False)
    logger.info("Wrote %d brand metric rows -> %s", len(metrics), config.CAMPAIGNS_PARQUET)


if __name__ == "__main__":
    main()
