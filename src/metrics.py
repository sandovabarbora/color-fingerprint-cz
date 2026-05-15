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

# Threshold below which a cluster's hue is too noisy to count toward the
# "dominant chromatic hue" — desaturated centroids (grays, browns at the
# low end) have unstable hue and would dilute the signal.
DOMINANT_HUE_SAT_MIN: float = 0.15

# Arc-shape thresholds (operate on saturation in [0, 1])
ARC_FLAT_RANGE: float = 0.10        # if max - min < this, classify 'flat'
ARC_QUADRATIC_CURVATURE: float = 0.4  # |a| in y = a*t² + b*t + c with t in [0,1]

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


# --- Metric: dominant chromatic hue + brand-color gap ---------------------


def dominant_chromatic_hue(palettes: pd.DataFrame) -> float | None:
    """Saturation-weighted circular mean of hue across all clusters.

    Desaturated centroids (s < ``DOMINANT_HUE_SAT_MIN``) are excluded
    because their hue is numerically unstable and conceptually meaningless
    (a near-gray pixel doesn't have a "color identity"). The remaining
    chromatic centroids contribute as unit vectors on the hue circle
    weighted by cluster_weight * saturation.

    Args:
        palettes: All palette rows for one brand.

    Returns:
        Mean hue in degrees [0, 360), or None if no centroid passes the
        saturation gate (i.e. the brand's whole spot is achromatic).
    """
    rgb = palettes[["rgb_r", "rgb_g", "rgb_b"]].to_numpy() / 255.0
    hsv = np.array([colorsys.rgb_to_hsv(r, g, b) for r, g, b in rgb])
    hues_deg = hsv[:, 0] * 360.0
    sats = hsv[:, 1]
    weights = palettes["weight"].to_numpy(dtype=np.float64) * sats
    mask = sats >= DOMINANT_HUE_SAT_MIN
    if not mask.any() or weights[mask].sum() == 0:
        return None
    angles = np.deg2rad(hues_deg[mask])
    w = weights[mask]
    x = float((w * np.cos(angles)).sum())
    y = float((w * np.sin(angles)).sum())
    mean = np.rad2deg(np.arctan2(y, x))
    return float(mean % 360.0)


def brand_color_gap_deg(palettes: pd.DataFrame, anchor_hex: str) -> float | None:
    """Circular distance between dominant chromatic hue and anchor hue.

    Returns:
        Gap in degrees [0, 180], or None if the spot has no chromatic
        identity (entirely achromatic).
    """
    dominant = dominant_chromatic_hue(palettes)
    if dominant is None:
        return None
    anchor_hue = _rgb01_to_hue_deg(_hex_to_rgb01(anchor_hex))
    return _hue_circular_distance(dominant, anchor_hue)


# --- Metric: saturation arc shape -----------------------------------------


def arc_shape(palettes: pd.DataFrame) -> str:
    """Classify the saturation trajectory shape of one brand's spot.

    Procedure:
      1. Per-frame cluster-weighted saturation → time series.
      2. If max-min < ARC_FLAT_RANGE → 'flat'.
      3. Otherwise fit a quadratic y = a·t² + b·t + c (t normalized to
         [0, 1]) and inspect curvature plus apex position:
         - |a| ≥ ARC_QUADRATIC_CURVATURE with apex in [0.2, 0.8]:
           a < 0 → 'peak' (inverted U), a > 0 → 'valley' (U).
         - Otherwise fall back to linear slope sign → 'rising' / 'falling'.

    Returns:
        One of {'flat', 'rising', 'falling', 'peak', 'valley'}.
    """
    frame_ids = sorted(palettes["frame_id"].unique())
    if len(frame_ids) < 3:
        return "flat"
    sats = np.array(
        [frame_saturation(palettes[palettes["frame_id"] == fid]) for fid in frame_ids],
        dtype=np.float64,
    )
    if float(sats.max() - sats.min()) < ARC_FLAT_RANGE:
        return "flat"
    t = np.linspace(0.0, 1.0, len(sats))
    a, b, _c = np.polyfit(t, sats, 2)
    if abs(a) >= ARC_QUADRATIC_CURVATURE:
        # apex of y = a*t² + b*t + c is at t = -b / (2a)
        apex = -b / (2 * a)
        if 0.2 <= apex <= 0.8:
            return "peak" if a < 0 else "valley"
    slope = float(np.polyfit(t, sats, 1)[0])
    return "rising" if slope > 0 else "falling"


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
    color_gap = brand_color_gap_deg(palettes, spec.anchor_hex)
    shape = arc_shape(palettes)

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
        "color_gap_deg": float(color_gap) if color_gap is not None else float("nan"),
        "arc_shape": shape,
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
            "[%s] div=%.1f sat=%.2f/%.2f anchor=%.2f gap=%s shape=%s n=%d",
            spec.id,
            row["diversity"],
            row["sat_mean"],
            row["sat_std"],
            row["brand_anchor"],
            f"{row['color_gap_deg']:.1f}°" if not np.isnan(row["color_gap_deg"]) else "n/a",
            row["arc_shape"],
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
