"""Method audit: sensitivity of the brand-color-gap thesis to parameter
choices.

Two sweeps:

1. **Anchor tolerance** (used: ±15°). Re-rank brands at tolerance =
   5, 10, 15, 20, 25 and report Spearman rank correlation against the
   used setting. Stable rank correlation across tolerances means the
   thesis is not an artifact of where we drew the brand-color line.

2. **K-means k** (used: k=5). Re-fit K-means on every frame at
   k = 3, 4, 5, 6, 7, 8 and report rank-correlation of the resulting
   brand-color-gap rankings against k=5. Slower, since each k restarts
   the color stage on every frame.

Output is printed to stdout in a copy-paste-friendly format. Numbers go
into the README's Methods section.

Usage:
    uv run python -m scripts.sensitivity --tolerance-only      # cheap
    uv run python -m scripts.sensitivity                       # both
"""

from __future__ import annotations

import argparse
import logging
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from src import color, config, metrics
from src.fetch import brand_kebab_id, load_corpus

warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)


def _brand_level_palettes() -> pd.DataFrame:
    """Load frames.parquet and remap brand_id to brand-level (matches render)."""
    palettes = pd.read_parquet(config.FRAMES_PARQUET)
    specs = load_corpus()
    spot_to_brand = {s.id: brand_kebab_id(s.brand) for s in specs}
    palettes["brand_id"] = palettes["brand_id"].map(spot_to_brand).fillna(palettes["brand_id"])
    return palettes


def _brand_level_anchor_map() -> dict[str, str]:
    """Map brand kebab id -> anchor hex (one per brand)."""
    specs = load_corpus()
    return {brand_kebab_id(s.brand): s.anchor_hex for s in specs}


def sweep_tolerance() -> None:
    """Sensitivity of anchor_share to ±° tolerance choice."""
    palettes = _brand_level_palettes()
    anchors = _brand_level_anchor_map()
    brand_ids = sorted(palettes["brand_id"].unique())
    tolerances = [5, 10, 15, 20, 25]

    table: dict[int, dict[str, float]] = {}
    for tol in tolerances:
        shares: dict[str, float] = {}
        for bid in brand_ids:
            sub = palettes[palettes["brand_id"] == bid]
            anchor = anchors.get(bid)
            if anchor is None:
                continue
            shares[bid] = metrics.brand_anchor_share(sub, anchor_hex=anchor, tolerance_deg=tol)
        table[tol] = shares

    # Spearman correlation against the used tolerance (15)
    base = pd.Series(table[15]).reindex(brand_ids)
    print()
    print("ANCHOR-TOLERANCE SENSITIVITY (Spearman ρ vs used setting ±15°)")
    print("-" * 60)
    for tol in tolerances:
        s = pd.Series(table[tol]).reindex(brand_ids)
        rho, _p = spearmanr(base.to_numpy(), s.to_numpy())
        print(f"  ±{tol:>2}°   ρ = {rho:.3f}")
    print()
    print("Per-brand anchor share at each tolerance:")
    df = pd.DataFrame(table).reindex(brand_ids)
    df.columns = [f"±{c}°" for c in df.columns]
    print(df.round(3).to_string())


def sweep_k(sample_n: int | None = None) -> None:
    """Sensitivity of color_gap_deg to K-means k.

    Re-runs the color stage on every retained frame at each k, then
    aggregates to brand level and re-ranks. Spearman correlation against
    the used k = 5 shows whether the ranking is stable.

    Args:
        sample_n: if set, sub-sample this many frames per brand for speed.
            None runs on every frame.
    """
    specs = load_corpus()
    spot_to_brand = {s.id: brand_kebab_id(s.brand) for s in specs}
    anchors = _brand_level_anchor_map()
    # Group frame files by brand
    frames_by_brand: dict[str, list[Path]] = {}
    for spec in specs:
        brand_kid = spot_to_brand[spec.id]
        files = sorted(config.FRAMES_DIR.glob(f"{spec.id}_*.jpg"))
        if sample_n is not None and len(files) > sample_n:
            rng = np.random.default_rng(config.RANDOM_STATE)
            files = list(rng.choice(files, size=sample_n, replace=False))
        frames_by_brand.setdefault(brand_kid, []).extend(files)

    ks = [3, 4, 5, 6, 7, 8]
    gaps: dict[int, dict[str, float]] = {}
    for k in ks:
        logger.info("sweeping k=%d ...", k)
        per_brand_gap: dict[str, float] = {}
        for brand_kid, files in frames_by_brand.items():
            palettes_rows: list[pd.DataFrame] = []
            for fp in files:
                df = color.extract_palette(fp, brand_id=brand_kid, k=k)
                if df is not None:
                    palettes_rows.append(df)
            if not palettes_rows:
                continue
            pal = pd.concat(palettes_rows, ignore_index=True)
            gap = metrics.brand_color_gap_deg(pal, anchor_hex=anchors[brand_kid])
            per_brand_gap[brand_kid] = gap if gap is not None else float("nan")
        gaps[k] = per_brand_gap

    brand_ids = sorted(gaps[5].keys())
    base = pd.Series(gaps[5]).reindex(brand_ids)
    print()
    print("K-MEANS K SENSITIVITY (Spearman ρ of brand-color-gap rank vs k=5)")
    print("-" * 60)
    for k in ks:
        s = pd.Series(gaps[k]).reindex(brand_ids)
        rho, _p = spearmanr(base.to_numpy(), s.to_numpy())
        print(f"  k = {k}   ρ = {rho:.3f}")
    print()
    print("Per-brand gap (°) at each k:")
    df = pd.DataFrame(gaps).reindex(brand_ids)
    print(df.round(1).to_string())


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Method sensitivity audit.")
    parser.add_argument(
        "--tolerance-only", action="store_true", help="Skip the k sweep (expensive)."
    )
    parser.add_argument(
        "--k-sample", type=int, default=20, help="Frames per brand for k sweep."
    )
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    config.setup_logging(verbose=args.verbose)
    sweep_tolerance()
    if not args.tolerance_only:
        sweep_k(sample_n=args.k_sample)


if __name__ == "__main__":
    main()
