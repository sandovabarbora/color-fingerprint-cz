"""Tests for src.color: K-means determinism, palette structure, hex format."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from src import color, config


def _write_solid_jpg(path: Path, rgb: tuple[int, int, int]) -> Path:
    """Write a 60x60 solid-color JPG."""
    img = np.zeros((60, 60, 3), dtype=np.uint8)
    img[:, :] = rgb[::-1]  # BGR for cv2
    cv2.imwrite(str(path), img)
    return path


def _write_split_jpg(path: Path, rgb_a: tuple[int, int, int], rgb_b: tuple[int, int, int]) -> Path:
    """Write a 60x60 JPG split left/right between two colors."""
    img = np.zeros((60, 60, 3), dtype=np.uint8)
    img[:, :30] = rgb_a[::-1]
    img[:, 30:] = rgb_b[::-1]
    cv2.imwrite(str(path), img)
    return path


def test_kmeans_centroids_are_deterministic(tmp_path: Path) -> None:
    """Same seed + same input must produce centroids within FP tolerance.

    LAB centroids are compared with atol=1.0 (well below the perceptual
    JND of ~2.3 LAB units), accommodating sklearn 1.5+ KMeans's parallel
    n_init implementation which can introduce sub-unit centroid drift.
    """
    img_path = _write_split_jpg(tmp_path / "split.jpg", (200, 30, 30), (30, 30, 200))
    a = color.extract_palette(img_path, brand_id="t")
    b = color.extract_palette(img_path, brand_id="t")
    assert a is not None and b is not None
    # Weights are deterministic (integer pixel counts)
    np.testing.assert_array_equal(a["weight"].to_numpy(), b["weight"].to_numpy())
    np.testing.assert_allclose(a[["lab_l", "lab_a", "lab_b"]], b[["lab_l", "lab_a", "lab_b"]], atol=1.0)


def test_palette_has_k_clusters_and_weights_sum_to_one(tmp_path: Path) -> None:
    img_path = _write_split_jpg(tmp_path / "split.jpg", (255, 0, 0), (0, 255, 0))
    df = color.extract_palette(img_path, brand_id="t", k=config.KMEANS_K)
    assert df is not None
    assert len(df) == config.KMEANS_K
    assert df["weight"].sum() == 1.0 or abs(df["weight"].sum() - 1.0) < 1e-5


def test_clusters_sorted_by_weight_desc(tmp_path: Path) -> None:
    img_path = _write_split_jpg(tmp_path / "split.jpg", (255, 0, 0), (0, 255, 0))
    df = color.extract_palette(img_path, brand_id="t")
    assert df is not None
    weights = df["weight"].to_numpy()
    assert all(weights[i] >= weights[i + 1] for i in range(len(weights) - 1))


def test_hex_format_is_uppercase_rrggbb(tmp_path: Path) -> None:
    img_path = _write_solid_jpg(tmp_path / "solid.jpg", (128, 64, 200))
    df = color.extract_palette(img_path, brand_id="t")
    assert df is not None
    for h in df["hex"]:
        assert h.startswith("#")
        assert len(h) == 7
        assert h[1:].isalnum() and h[1:].upper() == h[1:]


def test_black_frame_is_skipped(tmp_path: Path) -> None:
    img_path = _write_solid_jpg(tmp_path / "black.jpg", (5, 5, 5))
    df = color.extract_palette(img_path, brand_id="t")
    assert df is None
