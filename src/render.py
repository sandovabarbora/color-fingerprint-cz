"""Render the single-page HTML report from the campaigns + frames parquets.

The output is a self-contained ``outputs/index.html`` with inline CSS and
inline SVG — no external assets, no JavaScript. Target page weight is
under 80KB (brief §6 Step 8 acceptance).

Sections rendered:
  1. Header
  2. Methodology box
  3. Per-brand SVG color strips (one row per brand)
  4. Cross-brand diversity x saturation-std SVG scatter
  5. 3-5 insights (from ``insights.yaml`` if present; else placeholders)
  6. Disclaimer + license footer

CLI:
    python -m src.render
"""

from __future__ import annotations

import argparse
import base64
import io
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from jinja2 import Environment, FileSystemLoader, select_autoescape
from PIL import Image

from src import config

logger = logging.getLogger(__name__)

AUTHOR_NAME = "Barbora Šandová"
REPO_URL = "https://github.com/sandovabarbora/color-fingerprint-cz"


# --- Color strip SVG -------------------------------------------------------


@dataclass(frozen=True)
class StripDims:
    width: int = 800
    height: int = 64


DEFAULT_STRIP_DIMS = StripDims()


def build_strip_png(
    brand_id: str, palettes: pd.DataFrame, dims: StripDims = DEFAULT_STRIP_DIMS
) -> str:
    """Build the per-brand color strip as a base64-inlined PNG.

    Each frame becomes one vertical column; within the column the K
    cluster colors stack top-to-bottom with heights proportional to
    cluster weight (so the dominant color visually fills most of the
    column). PNG was chosen over SVG because for 100+ frames the
    rect-count makes inline SVG balloon past the brief's 80KB cap.

    Args:
        brand_id: Corpus id whose frames to render.
        palettes: The full frames.parquet DataFrame.
        dims: Pixel dimensions of the rendered PNG.

    Returns:
        An ``<img src="data:image/png;base64,...">`` string ready to inline.
    """
    brand = palettes[palettes["brand_id"] == brand_id]
    frame_ids = sorted(brand["frame_id"].unique())
    if not frame_ids:
        return ""

    canvas = np.zeros((dims.height, dims.width, 3), dtype=np.uint8)
    n = len(frame_ids)
    for i, fid in enumerate(frame_ids):
        x0 = round(i * dims.width / n)
        x1 = round((i + 1) * dims.width / n)
        if x1 <= x0:
            x1 = x0 + 1
        frame = brand[brand["frame_id"] == fid].sort_values("cluster_idx")
        y = 0
        last_rgb = (0, 0, 0)
        for _, row in frame.iterrows():
            h = round(float(row["weight"]) * dims.height)
            rgb = (int(row["rgb_r"]), int(row["rgb_g"]), int(row["rgb_b"]))
            canvas[y : y + h, x0:x1] = rgb
            last_rgb = rgb
            y += h
        # Round-off can leave a 1-2px gap at the bottom; backfill with
        # the smallest (last) cluster's color rather than black.
        if y < dims.height:
            canvas[y : dims.height, x0:x1] = last_rgb

    pil = Image.fromarray(canvas, mode="RGB")
    buf = io.BytesIO()
    pil.save(buf, format="PNG", optimize=True)
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    return f'<img class="strip" alt="{brand_id} color strip" src="data:image/png;base64,{b64}">'


# --- Cross-brand scatter SVG ----------------------------------------------


@dataclass(frozen=True)
class ScatterDims:
    width: int = 540
    height: int = 360
    margin_left: int = 60
    margin_right: int = 30
    margin_top: int = 30
    margin_bottom: int = 50


def _nice_ticks(lo: float, hi: float, n: int = 4) -> list[float]:
    """Pick ~n round-ish tick locations spanning [lo, hi]."""
    span = hi - lo
    if span <= 0:
        return [lo]
    raw_step = span / n
    magnitude = 10 ** int(_floor_log10(raw_step))
    for mult in (1, 2, 2.5, 5, 10):
        step = mult * magnitude
        if step >= raw_step:
            break
    start = step * (lo // step)
    ticks: list[float] = []
    t = start
    while t <= hi + 1e-9:
        if t >= lo - 1e-9:
            ticks.append(round(t, 6))
        t += step
    return ticks


def _floor_log10(x: float) -> int:
    import math

    return math.floor(math.log10(x)) if x > 0 else 0


DEFAULT_SCATTER_DIMS = ScatterDims()


def build_scatter_svg(
    campaigns: pd.DataFrame, dims: ScatterDims = DEFAULT_SCATTER_DIMS
) -> str:
    """Build the cross-brand scatter (diversity vs sat_std)."""
    df = campaigns.copy()
    if df.empty:
        return f'<svg class="scatter" viewBox="0 0 {dims.width} {dims.height}"></svg>'

    x_min, x_max = float(df["diversity"].min()), float(df["diversity"].max())
    y_min, y_max = float(df["sat_std"].min()), float(df["sat_std"].max())
    # 12% padding either side so dots don't sit on the axes
    x_pad = max((x_max - x_min) * 0.12, 1.0)
    y_pad = max((y_max - y_min) * 0.12, 0.01)
    x_lo, x_hi = x_min - x_pad, x_max + x_pad
    y_lo, y_hi = y_min - y_pad, y_max + y_pad

    plot_l = dims.margin_left
    plot_r = dims.width - dims.margin_right
    plot_t = dims.margin_top
    plot_b = dims.height - dims.margin_bottom

    def x_to_px(x: float) -> float:
        return plot_l + (x - x_lo) / (x_hi - x_lo) * (plot_r - plot_l)

    def y_to_px(y: float) -> float:
        return plot_b - (y - y_lo) / (y_hi - y_lo) * (plot_b - plot_t)

    parts: list[str] = [
        f'<svg class="scatter" viewBox="0 0 {dims.width} {dims.height}" '
        f'xmlns="http://www.w3.org/2000/svg">'
    ]

    # Axes
    parts.append(
        f'<line x1="{plot_l}" y1="{plot_b}" x2="{plot_r}" y2="{plot_b}" '
        f'stroke="#888" stroke-width="0.6"/>'
    )
    parts.append(
        f'<line x1="{plot_l}" y1="{plot_t}" x2="{plot_l}" y2="{plot_b}" '
        f'stroke="#888" stroke-width="0.6"/>'
    )

    # Ticks
    for xt in _nice_ticks(x_lo, x_hi, n=4):
        px = x_to_px(xt)
        parts.append(
            f'<line x1="{px:.1f}" y1="{plot_b}" x2="{px:.1f}" y2="{plot_b + 4}" '
            f'stroke="#888" stroke-width="0.5"/>'
        )
        parts.append(
            f'<text x="{px:.1f}" y="{plot_b + 16}" text-anchor="middle" '
            f'font-size="10" fill="#555">{xt:g}</text>'
        )
    for yt in _nice_ticks(y_lo, y_hi, n=4):
        py = y_to_px(yt)
        parts.append(
            f'<line x1="{plot_l - 4}" y1="{py:.1f}" x2="{plot_l}" y2="{py:.1f}" '
            f'stroke="#888" stroke-width="0.5"/>'
        )
        parts.append(
            f'<text x="{plot_l - 6}" y="{py + 3:.1f}" text-anchor="end" '
            f'font-size="10" fill="#555">{yt:.2f}</text>'
        )

    # Axis labels
    cx = (plot_l + plot_r) / 2
    cy = (plot_t + plot_b) / 2
    parts.append(
        f'<text x="{cx}" y="{dims.height - 12}" text-anchor="middle" '
        f'font-size="11" fill="#444">palette diversity →</text>'
    )
    parts.append(
        f'<text x="16" y="{cy}" transform="rotate(-90, 16, {cy})" '
        f'text-anchor="middle" font-size="11" fill="#444">'
        f'saturation tension (std) →</text>'
    )

    # Dots + labels
    for _, row in df.iterrows():
        px = x_to_px(float(row["diversity"]))
        py = y_to_px(float(row["sat_std"]))
        parts.append(
            f'<circle cx="{px:.1f}" cy="{py:.1f}" r="6.5" '
            f'fill="{row["anchor_hex"]}" stroke="#222" stroke-width="0.6"/>'
        )
        parts.append(
            f'<text x="{px + 10:.1f}" y="{py + 3.5:.1f}" font-size="10" '
            f'fill="#222">{row["brand"]}</text>'
        )

    parts.append("</svg>")
    return "".join(parts)


# --- Insights --------------------------------------------------------------


def load_insights(path: Path = config.PROJECT_ROOT / "config" / "insights.yaml") -> list[dict[str, str]]:
    """Load curated insights from YAML, or return placeholders if missing."""
    if not path.exists():
        return [
            {
                "obs": "Insights pending.",
                "interp": "Edit config/insights.yaml to populate this section.",
            }
        ]
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    return list(raw.get("insights", []))


# --- Rendering -------------------------------------------------------------


def build_brand_rows(
    campaigns: pd.DataFrame, palettes: pd.DataFrame
) -> list[dict[str, Any]]:
    """Build the per-brand row data passed to the template."""
    rows: list[dict[str, Any]] = []
    # Order by diversity desc — biggest stories first, dampest last
    for _, c in campaigns.sort_values("diversity", ascending=False).iterrows():
        rows.append(
            {
                "brand_id": c["brand_id"],
                "brand": c["brand"],
                "sector": c["sector"],
                "anchor_hex": c["anchor_hex"],
                "n_frames": int(c["n_frames"]),
                "strip_html": build_strip_png(c["brand_id"], palettes),
            }
        )
    return rows


def render_report(
    campaigns: pd.DataFrame,
    palettes: pd.DataFrame,
    *,
    out_path: Path = config.REPORT_HTML,
    templates_dir: Path = config.TEMPLATES_DIR,
) -> Path:
    """Render the full HTML report and write it to disk."""
    env = Environment(
        loader=FileSystemLoader(str(templates_dir)),
        autoescape=select_autoescape(["html"]),
        trim_blocks=True,
        lstrip_blocks=True,
    )
    css = env.get_template("style.css.j2").render()
    html = env.get_template("report.html.j2").render(
        inline_css=css,
        author=AUTHOR_NAME,
        repo_url=REPO_URL,
        brand_rows=build_brand_rows(campaigns, palettes),
        scatter_svg=build_scatter_svg(campaigns),
        insights=load_insights(),
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(html, encoding="utf-8")
    return out_path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render the HTML report.")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    config.setup_logging(verbose=args.verbose)
    config.ensure_dirs()
    campaigns = pd.read_parquet(config.CAMPAIGNS_PARQUET)
    palettes = pd.read_parquet(config.FRAMES_PARQUET)
    out = render_report(campaigns, palettes)
    size_kb = out.stat().st_size / 1024
    logger.info("Wrote %s (%.1f KB)", out, size_kb)


if __name__ == "__main__":
    main()
