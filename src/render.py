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
from scipy.cluster.hierarchy import leaves_list, linkage
from scipy.spatial.distance import squareform

from src import config
from src import metrics as metrics_mod
from src.fetch import brand_kebab_id, load_corpus

logger = logging.getLogger(__name__)

AUTHOR_NAME = "Barbora Šandová"
REPO_URL = "https://github.com/sandovabarbora/color-fingerprint-cz"

ARC_GLYPHS: dict[str, str] = {
    "flat": "—",
    "rising": "↗",
    "falling": "↘",
    "peak": "⌒",
    "valley": "⌣",
}


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

    # Median crosshair (quadrant split)
    x_med = float(df["diversity"].median())
    y_med = float(df["sat_std"].median())
    x_med_px = x_to_px(x_med)
    y_med_px = y_to_px(y_med)
    parts.append(
        f'<line x1="{x_med_px:.1f}" y1="{plot_t}" x2="{x_med_px:.1f}" y2="{plot_b}" '
        f'stroke="#BBB" stroke-width="0.5" stroke-dasharray="3 3"/>'
    )
    parts.append(
        f'<line x1="{plot_l}" y1="{y_med_px:.1f}" x2="{plot_r}" y2="{y_med_px:.1f}" '
        f'stroke="#BBB" stroke-width="0.5" stroke-dasharray="3 3"/>'
    )

    # Quadrant labels at the four corners of the plot area
    quad_labels = [
        (plot_l + 6, plot_t + 12, "start", "Mood-piece"),
        (plot_r - 6, plot_t + 12, "end", "Performative"),
        (plot_l + 6, plot_b - 6, "start", "Corporate-flat"),
        (plot_r - 6, plot_b - 6, "end", "Product-hero"),
    ]
    for tx, ty, anchor, label in quad_labels:
        parts.append(
            f'<text x="{tx}" y="{ty}" text-anchor="{anchor}" font-size="9.5" '
            f'fill="#888" font-style="italic">{label}</text>'
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


# --- Brand-color gap horizontal bars --------------------------------------


def build_gap_svg(campaigns: pd.DataFrame, width: int = 720, row_h: int | None = None) -> str:
    """Horizontal bar chart of brand-color gap (degrees), sorted ascending.

    Each bar is filled with the brand's anchor color so the reader can
    see what swatch is "missing" from the picture. A reference tick at
    90° marks the threshold beyond which the on-screen color is essentially
    a different color from the brand color.

    Adapts row height to the number of bars so a 13-row chart stays
    readable and a 47-row chart stays page-friendly.
    """
    df = campaigns.dropna(subset=["color_gap_deg"]).copy()
    if df.empty:
        return ""
    df = df.sort_values("color_gap_deg", ascending=True).reset_index(drop=True)
    n = len(df)
    if row_h is None:
        row_h = 22 if n <= 20 else 14 if n <= 40 else 12
    # Build per-row labels: brand for unique brands, brand + index for multi-spot brands
    brand_counts = df["brand"].value_counts()
    seen: dict[str, int] = {}
    labels: list[str] = []
    for brand in df["brand"]:
        if brand_counts[brand] == 1:
            labels.append(brand)
        else:
            seen[brand] = seen.get(brand, 0) + 1
            labels.append(f"{brand} {seen[brand]:02d}")
    label_w = 130
    plot_l = label_w
    plot_r = width - 40
    value_w = plot_r - plot_l
    height = row_h * n + 28  # +28 for axis label at top
    max_deg = 180.0

    parts: list[str] = [
        f'<svg class="gap-chart" viewBox="0 0 {width} {height}" '
        f'xmlns="http://www.w3.org/2000/svg">'
    ]

    # Axis ticks
    for deg in (0, 45, 90, 135, 180):
        x = plot_l + deg / max_deg * value_w
        parts.append(
            f'<line x1="{x:.1f}" y1="20" x2="{x:.1f}" y2="{height - 4}" '
            f'stroke="#EEE" stroke-width="0.5"/>'
        )
        parts.append(
            f'<text x="{x:.1f}" y="14" text-anchor="middle" font-size="10" '
            f'fill="#777">{deg}°</text>'
        )
    # 90° reference: visual threshold for "different color"
    x90 = plot_l + 90 / max_deg * value_w
    parts.append(
        f'<line x1="{x90:.1f}" y1="20" x2="{x90:.1f}" y2="{height - 4}" '
        f'stroke="#AAA" stroke-width="0.6" stroke-dasharray="2 3"/>'
    )

    for i, row in df.iterrows():
        y = 24 + i * row_h
        gap = float(row["color_gap_deg"])
        bar_w = gap / max_deg * value_w
        parts.append(
            f'<text x="{plot_l - 8}" y="{y + row_h * 0.65:.1f}" '
            f'text-anchor="end" font-size="10" fill="#222">{labels[i]}</text>'
        )
        parts.append(
            f'<rect x="{plot_l}" y="{y}" width="{bar_w:.1f}" height="{row_h - 6}" '
            f'fill="{row["anchor_hex"]}" stroke="#222" stroke-width="0.3"/>'
        )
        parts.append(
            f'<text x="{plot_l + bar_w + 6:.1f}" y="{y + row_h * 0.65:.1f}" '
            f'font-size="10" fill="#555">{gap:.0f}°</text>'
        )
    parts.append("</svg>")
    return "".join(parts)


# --- Sparkline (per-brand saturation trajectory) --------------------------


def build_sparkline_svg(
    brand_id: str, palettes: pd.DataFrame, width: int = 160, height: int = 32
) -> str:
    """Render the per-frame saturation trajectory as a tiny inline SVG.

    Y axis is fixed to [0, 1] so sparklines are visually comparable across
    brands. A reference dotted line at sat=0.5 anchors the eye.
    """
    sub = palettes[palettes["brand_id"] == brand_id]
    if sub.empty:
        return ""
    series = metrics_mod.saturation_trajectory(sub)
    if not series:
        return ""
    n = len(series)
    pad_x = 1
    pad_y = 2
    x_step = (width - 2 * pad_x) / max(n - 1, 1)

    def y_to_px(v: float) -> float:
        return pad_y + (1 - v) * (height - 2 * pad_y)

    pts = " ".join(
        f"{pad_x + i * x_step:.1f},{y_to_px(v):.1f}" for i, v in enumerate(series)
    )
    mid = y_to_px(0.5)
    return (
        f'<svg class="spark" viewBox="0 0 {width} {height}" '
        f'xmlns="http://www.w3.org/2000/svg" aria-label="saturation trajectory">'
        f'<line x1="0" y1="{mid:.1f}" x2="{width}" y2="{mid:.1f}" '
        f'stroke="#D8D2C5" stroke-width="0.6" stroke-dasharray="2 3"/>'
        f'<polyline points="{pts}" fill="none" stroke="#222" stroke-width="1.2" '
        f'stroke-linecap="round" stroke-linejoin="round"/>'
        f"</svg>"
    )


# --- Polar hue radial (per brand small multiple) --------------------------


def build_hue_radial_svg(
    brand_id: str,
    palettes: pd.DataFrame,
    anchor_hex: str,
    size: int = 132,
    n_bins: int = 24,
) -> str:
    """Render a per-brand polar histogram of saturation-weighted hue.

    Each angular bin is drawn as a thin rotated rectangle pointing outward
    from the centre; the rect height encodes weighted density, the rect
    fill encodes that bin's hue. Using rect+transform instead of arc-paths
    cuts per-bin SVG bytes by ~3x and keeps the file under the page-weight
    budget. The brand's anchor hue is marked as a tick on the rim.
    """
    sub = palettes[palettes["brand_id"] == brand_id]
    if sub.empty:
        return ""
    centers, weights = metrics_mod.hue_density(sub, n_bins=n_bins)
    max_w = max(weights) if weights else 1.0
    if max_w <= 0:
        max_w = 1.0
    cx = cy = size / 2
    rim_r = size / 2 - 6
    inner_r = rim_r * 0.2
    bin_arc = 360.0 / n_bins
    # Width of each rect = chord at inner_r for the bin arc. Slight overlap
    # via 1.02 multiplier makes adjacent wedges meet cleanly.
    rect_w = 2 * inner_r * np.tan(np.deg2rad(bin_arc / 2)) * 1.02

    parts: list[str] = [
        f'<svg class="radial" viewBox="0 0 {size} {size}" '
        f'xmlns="http://www.w3.org/2000/svg" aria-label="hue density">'
        f'<circle cx="{cx}" cy="{cy}" r="{rim_r:.1f}" fill="none" '
        f'stroke="#E2DDD0" stroke-width="0.6"/>'
    ]
    for hue_deg, w in zip(centers, weights, strict=True):
        if w <= 0:
            continue
        # Hue 0° = right in SVG, then counter-clockwise visually (we flip
        # via -hue_deg in the rotate). The wedge is drawn pointing up at
        # rotate(0) and then spun to its hue angle.
        bar_h = (rim_r - inner_r) * (w / max_w)
        fill = f"hsl({hue_deg:.0f} 72% 55%)"
        # rotate(-hue_deg) makes hue 0° point right; SVG's rotation centre
        # defaults to (0,0), so we translate the bar to (cx, cy) first.
        parts.append(
            f'<rect x="{-rect_w / 2:.2f}" y="{inner_r:.1f}" '
            f'width="{rect_w:.2f}" height="{bar_h:.2f}" fill="{fill}" '
            f'transform="translate({cx},{cy}) rotate({90 - hue_deg:.0f})"/>'
        )
    # Anchor tick at the rim
    anchor_rgb = _hex_to_rgb01(anchor_hex)
    anchor_hue = _rgb01_to_hue_deg(anchor_rgb)
    ax = np.deg2rad(anchor_hue)
    x_in = cx + (rim_r - 4) * np.cos(ax)
    y_in = cy - (rim_r - 4) * np.sin(ax)
    x_out = cx + (rim_r + 5) * np.cos(ax)
    y_out = cy - (rim_r + 5) * np.sin(ax)
    parts.append(
        f'<line x1="{x_in:.1f}" y1="{y_in:.1f}" x2="{x_out:.1f}" y2="{y_out:.1f}" '
        f'stroke="{anchor_hex}" stroke-width="2.4" stroke-linecap="round"/>'
    )
    parts.append("</svg>")
    return "".join(parts)


# small local helpers (avoid circular import vs metrics.py for two trivial fns)


def _hex_to_rgb01(hex_color: str) -> tuple[float, float, float]:
    s = hex_color.lstrip("#")
    return tuple(int(s[i : i + 2], 16) / 255.0 for i in (0, 2, 4))  # type: ignore[return-value]


def _rgb01_to_hue_deg(rgb: tuple[float, float, float]) -> float:
    import colorsys

    h, _s, _v = colorsys.rgb_to_hsv(*rgb)
    return h * 360.0


# --- Bootstrap CI bar chart -----------------------------------------------


def build_sector_ci_svg(
    campaigns: pd.DataFrame,
    metric: str = "sat_std",
    width: int = 720,
    row_h: int = 36,
    min_n: int = 2,
) -> str:
    """Per-sector bootstrap CI on the chosen metric.

    For each sector with >= min_n brands, compute a 95% bootstrap CI on
    the mean of ``metric`` across the brands in that sector. Render as a
    Cleveland-style dot-with-bar plot, sorted by point estimate.

    Args:
        campaigns: campaigns.parquet (one row per brand).
        metric: column name to summarize (default 'sat_std' for the
            'saturation tension' story).
        width: SVG width.
        row_h: row height per sector.
        min_n: sectors with fewer brands are excluded (CI undefined).
    """
    by_sector = campaigns.groupby("sector")[metric].apply(np.asarray)
    rows: list[tuple[str, float, float, float, int]] = []
    for sector, vals in by_sector.items():
        if len(vals) < min_n:
            continue
        m, lo, hi = metrics_mod.bootstrap_mean_ci(vals)
        rows.append((sector, m, lo, hi, len(vals)))
    if not rows:
        return ""
    rows.sort(key=lambda r: r[1])
    n = len(rows)
    label_w = 130
    plot_l = label_w
    plot_r = width - 60
    value_w = plot_r - plot_l
    height = 28 + row_h * n
    all_vals = [v for r in rows for v in (r[2], r[3])]
    lo_lim = min(0, min(all_vals) - 0.01)
    hi_lim = max(all_vals) + 0.02

    def x_to_px(v: float) -> float:
        return plot_l + (v - lo_lim) / (hi_lim - lo_lim) * value_w

    parts = [
        f'<svg class="sector-ci" viewBox="0 0 {width} {height}" '
        f'xmlns="http://www.w3.org/2000/svg">'
    ]
    # Axis ticks: 4 round numbers across the range
    n_ticks = 4
    for i in range(n_ticks + 1):
        v = lo_lim + (hi_lim - lo_lim) * i / n_ticks
        x = x_to_px(v)
        parts.append(
            f'<line x1="{x:.1f}" y1="22" x2="{x:.1f}" y2="{height - 4}" '
            f'stroke="#EEE" stroke-width="0.5"/>'
        )
        parts.append(
            f'<text x="{x:.1f}" y="16" text-anchor="middle" font-size="10" '
            f'fill="#777">{v:.2f}</text>'
        )

    for i, (sector, m, lo, hi, k) in enumerate(rows):
        y = 32 + i * row_h
        # Label
        parts.append(
            f'<text x="{plot_l - 12}" y="{y + 4:.1f}" text-anchor="end" '
            f'font-size="12" fill="#222" font-weight="500">{sector}</text>'
        )
        parts.append(
            f'<text x="{plot_l - 12}" y="{y + 17:.1f}" text-anchor="end" '
            f'font-size="10" fill="#888">n={k} brands</text>'
        )
        # CI bar
        x_lo = x_to_px(lo)
        x_hi = x_to_px(hi)
        x_m = x_to_px(m)
        parts.append(
            f'<line x1="{x_lo:.1f}" y1="{y}" x2="{x_hi:.1f}" y2="{y}" '
            f'stroke="#666" stroke-width="1.5"/>'
        )
        parts.append(
            f'<line x1="{x_lo:.1f}" y1="{y - 4}" x2="{x_lo:.1f}" y2="{y + 4}" '
            f'stroke="#666" stroke-width="1.5"/>'
        )
        parts.append(
            f'<line x1="{x_hi:.1f}" y1="{y - 4}" x2="{x_hi:.1f}" y2="{y + 4}" '
            f'stroke="#666" stroke-width="1.5"/>'
        )
        # Point estimate
        parts.append(
            f'<circle cx="{x_m:.1f}" cy="{y}" r="4.5" fill="#181818"/>'
        )
        # CI numeric tail
        parts.append(
            f'<text x="{x_hi + 8:.1f}" y="{y + 4:.1f}" font-size="10" '
            f'fill="#555">[{lo:.2f}, {hi:.2f}]</text>'
        )
    parts.append("</svg>")
    return "".join(parts)


# --- Brand-similarity dendrogram ------------------------------------------


def _dendrogram_layout(
    z: np.ndarray, n_leaves: int, leaf_order: np.ndarray
) -> tuple[list[dict[str, float]], list[dict[str, float]]]:
    """Compute (nodes, edges) for a horizontal dendrogram.

    Args:
        z: scipy linkage matrix, shape (n-1, 4).
        n_leaves: number of original observations.
        leaf_order: list of original-index positions in the leaf order
            scipy chose for plotting.

    Returns:
        nodes: list of {idx, x, y} for every cluster (leaves + internal).
        edges: list of {x1, y1, x2, y2} line segments to render.
    """
    # Leaves get y = their index in leaf_order
    leaf_y: dict[int, float] = {leaf: i for i, leaf in enumerate(leaf_order)}
    node_x: dict[int, float] = {leaf: 0.0 for leaf in leaf_order}
    node_y: dict[int, float] = {leaf: float(leaf_y[leaf]) for leaf in leaf_order}

    edges: list[dict[str, float]] = []
    for i, row in enumerate(z):
        a, b, dist, _n = int(row[0]), int(row[1]), float(row[2]), int(row[3])
        merged_idx = n_leaves + i
        y_a = node_y[a]
        y_b = node_y[b]
        x_a = node_x[a]
        x_b = node_x[b]
        # The merged node sits at x = dist, y = mean of its children
        y_m = (y_a + y_b) / 2
        node_y[merged_idx] = y_m
        node_x[merged_idx] = dist
        # Two horizontal lines from children up to merge x, and a vertical
        # bracket connecting them.
        edges.append({"x1": x_a, "y1": y_a, "x2": dist, "y2": y_a})
        edges.append({"x1": x_b, "y1": y_b, "x2": dist, "y2": y_b})
        edges.append({"x1": dist, "y1": y_a, "x2": dist, "y2": y_b})
    nodes = [
        {"idx": float(leaf), "x": 0.0, "y": float(leaf_y[leaf])} for leaf in leaf_order
    ]
    return nodes, edges


def build_dendrogram_svg(
    palettes: pd.DataFrame,
    campaigns: pd.DataFrame,
    width: int = 760,
    row_h: int = 30,
    label_w: int = 170,
) -> str:
    """Hierarchical clustering on between-brand energy distance.

    Computes the pairwise Székely energy distance between every pair of
    brands' weighted LAB palettes, runs UPGMA (average linkage), and
    renders a horizontal dendrogram. Labels are colored with each brand's
    anchor swatch dot so the cluster colors carry brand identity.
    """
    brand_ids = list(campaigns["brand_id"])
    n = len(brand_ids)
    if n < 2:
        return ""
    dmat = metrics_mod.brand_distance_matrix(palettes, brand_ids)
    condensed = squareform(dmat, checks=False)
    z = linkage(condensed, method="average")
    leaf_order = leaves_list(z)
    _nodes, edges = _dendrogram_layout(z, n, leaf_order)

    # x-axis range: 0 to max merge distance. Floor to a small positive
    # number so degenerate fixtures (all-identical brands) don't divide
    # by zero in tests.
    max_d = max(float(z[:, 2].max()), 1e-6)
    plot_l = label_w
    plot_r = width - 40
    plot_w = plot_r - plot_l
    height = 32 + row_h * n

    def x_to_px(x_val: float) -> float:
        return plot_l + (x_val / max_d) * plot_w

    def y_to_px(y_val: float) -> float:
        return 28 + y_val * row_h

    parts = [
        f'<svg class="dendrogram" viewBox="0 0 {width} {height}" '
        f'xmlns="http://www.w3.org/2000/svg">'
    ]
    # Axis at top
    parts.append(
        f'<line x1="{plot_l}" y1="22" x2="{plot_r}" y2="22" '
        f'stroke="#DDD" stroke-width="0.5"/>'
    )
    for k in range(5):
        v = max_d * k / 4
        x = x_to_px(v)
        parts.append(
            f'<line x1="{x:.1f}" y1="18" x2="{x:.1f}" y2="22" '
            f'stroke="#888" stroke-width="0.5"/>'
        )
        parts.append(
            f'<text x="{x:.1f}" y="14" text-anchor="middle" font-size="9" '
            f'fill="#777">{v:.1f}</text>'
        )
    parts.append(
        f'<text x="{(plot_l + plot_r) / 2:.0f}" y="6" text-anchor="middle" '
        f'font-size="9" fill="#999" letter-spacing="1">'
        f'ENERGY DISTANCE (LAB)</text>'
    )

    # Edges
    for e in edges:
        parts.append(
            f'<line x1="{x_to_px(e["x1"]):.1f}" y1="{y_to_px(e["y1"]):.1f}" '
            f'x2="{x_to_px(e["x2"]):.1f}" y2="{y_to_px(e["y2"]):.1f}" '
            f'stroke="#222" stroke-width="0.9"/>'
        )

    # Leaf labels
    brand_lookup = campaigns.set_index("brand_id")
    for leaf_idx in leaf_order:
        bid = brand_ids[leaf_idx]
        row = brand_lookup.loc[bid]
        y = y_to_px(float(np.where(leaf_order == leaf_idx)[0][0]))
        # Swatch dot
        parts.append(
            f'<circle cx="{plot_l - 14:.1f}" cy="{y:.1f}" r="4" '
            f'fill="{row["anchor_hex"]}" stroke="#222" stroke-width="0.3"/>'
        )
        # Label (brand name)
        parts.append(
            f'<text x="{plot_l - 24:.1f}" y="{y + 3.5:.1f}" text-anchor="end" '
            f'font-size="11" fill="#181818">{row["brand"]}</text>'
        )
        # Sector below
        parts.append(
            f'<text x="{plot_l - 24:.1f}" y="{y + 15:.1f}" text-anchor="end" '
            f'font-size="9" fill="#888" letter-spacing="0.5">'
            f'{row["sector"].upper()}</text>'
        )

    parts.append("</svg>")
    return "".join(parts)


# --- Insights --------------------------------------------------------------


def load_patterns_and_playbook(
    path: Path = config.PROJECT_ROOT / "config" / "insights.yaml",
) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    """Load patterns + playbook from YAML.

    Returns:
        ``(patterns, playbook)``. Patterns is a list of
        ``{title, evidence, takeaway}`` dicts (what's true); playbook is
        a list of ``{title, body}`` dicts (what to do about it). Empty
        lists if the file is absent.
    """
    if not path.exists():
        return ([], [])
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    patterns = list(raw.get("patterns", []))
    playbook = list(raw.get("playbook", []))
    return patterns, playbook


# --- Rendering -------------------------------------------------------------


def build_brand_rows(
    campaigns: pd.DataFrame, palettes: pd.DataFrame
) -> list[dict[str, Any]]:
    """Build the per-brand row data passed to the template."""
    rows: list[dict[str, Any]] = []
    # Order by brand-color-gap ascending: brand-led spots first
    # (the camp that compresses the gap), then world-led spots. This
    # makes the page's central thesis visible as you scroll the strips.
    for _, c in campaigns.sort_values("color_gap_deg", ascending=True).iterrows():
        rows.append(
            {
                "brand_id": c["brand_id"],
                "brand": c["brand"],
                "sector": c["sector"],
                "anchor_hex": c["anchor_hex"],
                "n_frames": int(c["n_frames"]),
                "arc_shape": c.get("arc_shape", "flat"),
                "arc_glyph": ARC_GLYPHS.get(c.get("arc_shape", "flat"), "—"),
                "strip_html": build_strip_png(c["brand_id"], palettes),
                "sparkline_svg": build_sparkline_svg(c["brand_id"], palettes),
                "radial_svg": build_hue_radial_svg(
                    c["brand_id"], palettes, c["anchor_hex"]
                ),
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
    patterns, playbook = load_patterns_and_playbook()
    # Spot-level data for the gap-chart hero (one bar per of the ~47 spots
    # so the bimodal distribution is visible). Falls back to campaigns
    # if spots.parquet is absent (single-spot-per-brand corpora).
    spots_path = config.PROCESSED_DIR / "spots.parquet"
    spots = pd.read_parquet(spots_path) if spots_path.exists() else campaigns
    n_total = len(spots)
    n_close = int((spots["color_gap_deg"] < 20).sum())
    n_far = int((spots["color_gap_deg"] > 100).sum())
    n_mid = n_total - n_close - n_far
    counts = {
        "total": n_total,
        "close": n_close,
        "far": n_far,
        "mid": n_mid,
        "n_brands": len(campaigns),
    }
    html = env.get_template("report.html.j2").render(
        inline_css=css,
        author=AUTHOR_NAME,
        repo_url=REPO_URL,
        counts=counts,
        brand_rows=build_brand_rows(campaigns, palettes),
        scatter_svg=build_scatter_svg(campaigns),
        sector_ci_svg=build_sector_ci_svg(campaigns, metric="sat_std"),
        gap_svg=build_gap_svg(spots),
        dendrogram_svg=build_dendrogram_svg(palettes, campaigns),
        patterns=patterns,
        playbook=playbook,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(html, encoding="utf-8")
    return out_path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render the HTML report.")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def _remap_palettes_to_brand_level(palettes: pd.DataFrame) -> pd.DataFrame:
    """Replace spot-level brand_id with brand-level brand_id by joining
    against the corpus. After this remap, every palette row carries the
    same brand_id that campaigns.parquet uses, so the SVG/PNG builders
    aggregate across all spots of a brand without any signature change.
    """
    try:
        specs = load_corpus()
    except Exception as exc:
        logger.warning("corpus load failed (%s); keeping spot-level brand_id", exc)
        return palettes
    spot_to_brand = {s.id: brand_kebab_id(s.brand) for s in specs}
    out = palettes.copy()
    out["brand_id"] = out["brand_id"].map(spot_to_brand).fillna(out["brand_id"])
    return out


def main() -> None:
    args = _parse_args()
    config.setup_logging(verbose=args.verbose)
    config.ensure_dirs()
    campaigns = pd.read_parquet(config.CAMPAIGNS_PARQUET)
    palettes = pd.read_parquet(config.FRAMES_PARQUET)
    palettes = _remap_palettes_to_brand_level(palettes)
    out = render_report(campaigns, palettes)
    size_kb = out.stat().st_size / 1024
    logger.info("Wrote %s (%.1f KB)", out, size_kb)


if __name__ == "__main__":
    main()
