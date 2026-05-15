"""Tests for src.metrics: property tests on diversity, anchor share, hue math."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src import metrics


def _palette(centroids_lab: list[tuple[float, float, float]], weights: list[float]) -> pd.DataFrame:
    """Build a single-frame palette DataFrame from explicit centroids."""
    arr = np.array(centroids_lab, dtype=np.float32)
    # rgb / hex aren't used by palette_diversity, but we fill them for completeness
    return pd.DataFrame(
        {
            "frame_id": "f0",
            "brand_id": "t",
            "cluster_idx": list(range(len(centroids_lab))),
            "weight": np.array(weights, dtype=np.float32),
            "lab_l": arr[:, 0],
            "lab_a": arr[:, 1],
            "lab_b": arr[:, 2],
            "rgb_r": np.array([255, 0, 0, 0, 0][: len(centroids_lab)], dtype=np.uint8),
            "rgb_g": np.array([0, 255, 0, 0, 0][: len(centroids_lab)], dtype=np.uint8),
            "rgb_b": np.array([0, 0, 255, 0, 0][: len(centroids_lab)], dtype=np.uint8),
            "hex": ["#FF0000", "#00FF00", "#0000FF", "#000000", "#FFFFFF"][: len(centroids_lab)],
        }
    )


def test_diversity_is_zero_for_single_color_palette() -> None:
    """All clusters at one centroid -> all pairwise distances are 0."""
    pal = _palette([(50, 0, 0)] * 5, [0.5, 0.2, 0.15, 0.1, 0.05])
    assert metrics.palette_diversity(pal) == 0.0


def test_diversity_increases_with_color_separation() -> None:
    near = _palette([(50, 0, 0), (52, 1, -1)], [0.5, 0.5])
    far = _palette([(50, -60, -60), (50, 60, 60)], [0.5, 0.5])
    assert metrics.palette_diversity(far) > metrics.palette_diversity(near)


def test_diversity_is_weighted_by_cluster_size() -> None:
    """Distance contribution scales with weight product."""
    centroids = [(0, 0, 0), (100, 0, 0)]
    big_first = metrics.palette_diversity(_palette(centroids, [0.9, 0.1]))
    even = metrics.palette_diversity(_palette(centroids, [0.5, 0.5]))
    # Even split puts more probability mass on cross-cluster pairs
    assert even > big_first


def test_hue_circular_distance_wraps_around() -> None:
    assert metrics._hue_circular_distance(10.0, 350.0) == 20.0
    assert metrics._hue_circular_distance(0.0, 180.0) == 180.0
    assert metrics._hue_circular_distance(45.0, 45.0) == 0.0


def test_brand_anchor_share_full_match() -> None:
    """All clusters match the anchor hue -> share is 1.0."""
    # Pure red across all clusters -> hue ≈ 0 -> matches #FF0000 anchor
    pal = _palette([(0, 0, 0)] * 5, [0.5, 0.2, 0.15, 0.1, 0.05])
    pal["rgb_r"] = 255
    pal["rgb_g"] = 0
    pal["rgb_b"] = 0
    share = metrics.brand_anchor_share(pal, anchor_hex="#FF0000")
    assert share == pytest.approx(1.0)


def test_brand_anchor_share_zero_when_far() -> None:
    """Anchor red vs all-green clusters -> share is 0."""
    pal = _palette([(0, 0, 0)] * 2, [0.5, 0.5])
    pal["rgb_r"] = 0
    pal["rgb_g"] = 255
    pal["rgb_b"] = 0
    share = metrics.brand_anchor_share(pal, anchor_hex="#FF0000")
    assert share == 0.0


def test_frame_saturation_zero_for_gray() -> None:
    pal = _palette([(50, 0, 0)] * 2, [0.5, 0.5])
    pal["rgb_r"] = 128
    pal["rgb_g"] = 128
    pal["rgb_b"] = 128
    assert metrics.frame_saturation(pal) == 0.0


def test_frame_saturation_one_for_pure_primary() -> None:
    pal = _palette([(50, 0, 0)] * 2, [0.5, 0.5])
    pal["rgb_r"] = 255
    pal["rgb_g"] = 0
    pal["rgb_b"] = 0
    assert metrics.frame_saturation(pal) == 1.0
