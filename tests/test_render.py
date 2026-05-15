"""Smoke tests for src.render: SVG/PNG builders + Jinja2 template render."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src import render


def _tiny_palettes() -> pd.DataFrame:
    """Two brands, two frames each, k=5 clusters per frame."""
    rows = []
    for brand_id in ("brand_a", "brand_b"):
        for fid in (0, 1):
            for k in range(5):
                rows.append(
                    {
                        "frame_id": f"{brand_id}_{fid:04d}",
                        "brand_id": brand_id,
                        "cluster_idx": k,
                        "weight": np.float32(0.5 if k == 0 else 0.5 / 4),
                        "lab_l": np.float32(50.0),
                        "lab_a": np.float32(0.0),
                        "lab_b": np.float32(0.0),
                        "rgb_r": np.uint8(128 + 10 * k),
                        "rgb_g": np.uint8(64),
                        "rgb_b": np.uint8(255 - 10 * k),
                        "hex": f"#{128 + 10 * k:02X}40{255 - 10 * k:02X}",
                    }
                )
    return pd.DataFrame(rows)


def _tiny_campaigns() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "brand_id": "brand_a",
                "brand": "Brand A",
                "sector": "QSR",
                "anchor_hex": "#FF0000",
                "n_frames": 2,
                "diversity": 22.5,
                "sat_mean": 0.30,
                "sat_std": 0.08,
                "brand_anchor": 0.15,
                "color_gap_deg": 30.0,
                "arc_shape": "flat",
            },
            {
                "brand_id": "brand_b",
                "brand": "Brand B",
                "sector": "Retail",
                "anchor_hex": "#0000FF",
                "n_frames": 2,
                "diversity": 18.0,
                "sat_mean": 0.40,
                "sat_std": 0.12,
                "brand_anchor": 0.30,
                "color_gap_deg": 120.0,
                "arc_shape": "rising",
            },
        ]
    )


def test_build_strip_png_returns_img_tag() -> None:
    palettes = _tiny_palettes()
    out = render.build_strip_png("brand_a", palettes)
    assert out.startswith('<img class="strip"')
    assert 'src="data:image/png;base64,' in out


def test_build_strip_png_handles_missing_brand() -> None:
    palettes = _tiny_palettes()
    out = render.build_strip_png("nonexistent", palettes)
    assert out == ""


def test_build_scatter_svg_has_one_dot_per_brand() -> None:
    campaigns = _tiny_campaigns()
    svg = render.build_scatter_svg(campaigns)
    assert svg.startswith('<svg class="scatter"')
    assert svg.count("<circle") == len(campaigns)


def test_render_report_writes_html_under_80kb(tmp_path: Path) -> None:
    campaigns = _tiny_campaigns()
    palettes = _tiny_palettes()
    out_path = tmp_path / "index.html"
    render.render_report(campaigns, palettes, out_path=out_path)
    assert out_path.exists()
    html = out_path.read_text(encoding="utf-8")
    assert "<title>" in html
    assert "Brand A" in html and "Brand B" in html
    # Brief acceptance §8: page weight < 80KB. Tiny fixtures should be well under.
    assert out_path.stat().st_size < 80 * 1024
