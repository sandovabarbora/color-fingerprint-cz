"""Smoke tests: project scaffold imports and constants are sane."""

from __future__ import annotations

from src import config


def test_paths_resolve_under_project_root() -> None:
    """All declared paths must live inside the project root."""
    root = config.PROJECT_ROOT
    for path in (
        config.CONFIG_DIR,
        config.DATA_DIR,
        config.RAW_DIR,
        config.FRAMES_DIR,
        config.PROCESSED_DIR,
        config.OUTPUTS_DIR,
        config.TEMPLATES_DIR,
        config.CORPUS_FILE,
        config.FRAMES_PARQUET,
        config.CAMPAIGNS_PARQUET,
        config.REPORT_HTML,
        config.PIPELINE_LOG,
    ):
        assert root in path.parents or path == root, f"{path} escapes project root"


def test_pipeline_constants_have_sane_defaults() -> None:
    """Sanity-check the numeric constants the rest of the pipeline depends on."""
    assert config.FPS == 1
    assert config.KMEANS_K == 5
    assert config.RANDOM_STATE == 42
    assert config.KMEANS_N_INIT >= 1
    assert config.MIN_DURATION_S < config.MAX_DURATION_S
    assert config.MIN_VIEW_COUNT > 0
    assert 0 < config.BRAND_ANCHOR_HUE_TOLERANCE_DEG <= 90
