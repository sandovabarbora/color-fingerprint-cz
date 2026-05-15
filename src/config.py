"""Project configuration: filesystem paths, pipeline constants, logging setup.

Importing this module has no side effects beyond defining names. Call
``setup_logging()`` once at the start of any pipeline entry point, and
``ensure_dirs()`` before writing to ``data/`` or ``outputs/``.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

# --- Filesystem layout -----------------------------------------------------

PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent
CONFIG_DIR: Path = PROJECT_ROOT / "config"
DATA_DIR: Path = PROJECT_ROOT / "data"
RAW_DIR: Path = DATA_DIR / "raw"
FRAMES_DIR: Path = DATA_DIR / "frames"
PROCESSED_DIR: Path = DATA_DIR / "processed"
OUTPUTS_DIR: Path = PROJECT_ROOT / "outputs"
TEMPLATES_DIR: Path = PROJECT_ROOT / "templates"
FIGURES_DIR: Path = OUTPUTS_DIR / "figures"

CORPUS_FILE: Path = CONFIG_DIR / "corpus.yaml"
FRAMES_PARQUET: Path = PROCESSED_DIR / "frames.parquet"
CAMPAIGNS_PARQUET: Path = PROCESSED_DIR / "campaigns.parquet"
REPORT_HTML: Path = OUTPUTS_DIR / "index.html"
PIPELINE_LOG: Path = PROJECT_ROOT / "pipeline.log"

# --- Pipeline constants ----------------------------------------------------

FPS: int = 1
KMEANS_K: int = 5
RANDOM_STATE: int = 42
KMEANS_N_INIT: int = 10

MIN_DURATION_S: float = 15.0
# 180s admits musical/storytelling formats (e.g. Mc'n'Roll, 134s) that
# behave like spots but exceed the typical 60-90s TV slot. Long-form
# content (interviews, behind-the-scenes) is still filtered.
MAX_DURATION_S: float = 180.0
MIN_VIEW_COUNT: int = 10_000

INTRO_SKIP_S: float = 0.5
OUTRO_SKIP_S: float = 0.5

# Drop frames whose mean LAB-L is below this (true-black intros/outros).
# LAB L ranges 0-100; OpenCV scales it to 0-255 (so 20 here means ~8/100 in
# the canonical scale — quite dark). See OpenCV color conversion docs:
# https://docs.opencv.org/4.x/de/d25/imgproc_color_conversions.html
MIN_FRAME_BRIGHTNESS_L: int = 20

BRAND_ANCHOR_HUE_TOLERANCE_DEG: float = 15.0


# --- Logging ---------------------------------------------------------------


def setup_logging(verbose: bool = False) -> logging.Logger:
    """Configure root logger: stdout at INFO (or DEBUG if verbose), file at DEBUG.

    Idempotent — clears existing handlers before adding new ones, so it is
    safe to call repeatedly (notebooks, tests).

    Args:
        verbose: If True, the stdout handler emits DEBUG records too.

    Returns:
        The root logger.
    """
    root = logging.getLogger()
    root.setLevel(logging.DEBUG)

    for handler in list(root.handlers):
        root.removeHandler(handler)

    fmt = logging.Formatter(
        fmt="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )

    stdout = logging.StreamHandler(sys.stdout)
    stdout.setLevel(logging.DEBUG if verbose else logging.INFO)
    stdout.setFormatter(fmt)
    root.addHandler(stdout)

    PIPELINE_LOG.parent.mkdir(parents=True, exist_ok=True)
    file_handler = logging.FileHandler(PIPELINE_LOG, mode="a", encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(fmt)
    root.addHandler(file_handler)

    return root


def ensure_dirs() -> None:
    """Create raw / frames / processed / outputs / figures directories if absent."""
    for d in (RAW_DIR, FRAMES_DIR, PROCESSED_DIR, OUTPUTS_DIR, FIGURES_DIR):
        d.mkdir(parents=True, exist_ok=True)
