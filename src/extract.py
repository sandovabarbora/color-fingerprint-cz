"""Extract frames from downloaded spots at 1 fps using ffmpeg-python.

For each video in ``data/raw/{id}.mp4``, write frames to
``data/frames/{id}_{seq:04d}.jpg``, skipping the first ``INTRO_SKIP_S``
and last ``OUTRO_SKIP_S`` seconds to avoid logo-dominated intros/outros.

Reference: ffmpeg-python — https://github.com/kkroening/ffmpeg-python

CLI:
    python -m src.extract                 # extract all available raw videos
    python -m src.extract --only kofola_menevice
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import ffmpeg  # type: ignore[import-untyped]

from src import config
from src.fetch import BrandSpec, load_corpus

logger = logging.getLogger(__name__)

# JPEG quality scale for ffmpeg's mjpeg encoder: 1 (best) to 31 (worst).
# 5 is high quality at ~10x smaller files than PNG; color centroids are
# indistinguishable from lossless at this setting.
JPEG_QSCALE: int = 5


def _probe_duration(video_path: Path) -> float:
    """Return total duration of ``video_path`` in seconds.

    Raises:
        ffmpeg.Error: If the file is unreadable or has no duration metadata.
    """
    probe = ffmpeg.probe(str(video_path))
    return float(probe["format"]["duration"])


def extract_frames(
    video_path: Path,
    brand_id: str,
    *,
    fps: int = config.FPS,
    intro_skip_s: float = config.INTRO_SKIP_S,
    outro_skip_s: float = config.OUTRO_SKIP_S,
    frames_dir: Path = config.FRAMES_DIR,
) -> list[Path]:
    """Extract frames from one video and return the list of output paths.

    Args:
        video_path: Path to the source MP4.
        brand_id: Corpus id used as filename prefix.
        fps: Output frames per second.
        intro_skip_s: Seconds to skip at the start.
        outro_skip_s: Seconds to skip at the end.
        frames_dir: Directory to write JPEGs into (created if missing).

    Returns:
        Sorted list of generated frame paths. Empty if extraction yielded
        nothing (e.g. video shorter than intro+outro slack).

    Raises:
        ffmpeg.Error: On unrecoverable ffmpeg failure (file corrupt etc.).
    """
    frames_dir.mkdir(parents=True, exist_ok=True)

    duration = _probe_duration(video_path)
    trim_window = duration - intro_skip_s - outro_skip_s
    if trim_window <= 0:
        logger.warning(
            "[%s] video too short (%.2fs) to drop %.2fs intro + %.2fs outro",
            brand_id,
            duration,
            intro_skip_s,
            outro_skip_s,
        )
        return []

    # Remove any stale frames for this brand so re-runs are deterministic
    for stale in frames_dir.glob(f"{brand_id}_*.jpg"):
        stale.unlink()

    output_pattern = str(frames_dir / f"{brand_id}_%04d.jpg")
    (
        ffmpeg.input(str(video_path), ss=intro_skip_s, t=trim_window)
        .filter("fps", fps=fps)
        .output(output_pattern, **{"qscale:v": JPEG_QSCALE, "start_number": 0})
        .overwrite_output()
        .run(quiet=True, capture_stdout=True, capture_stderr=True)
    )

    frames = sorted(frames_dir.glob(f"{brand_id}_*.jpg"))
    logger.info(
        "[%s] extracted %d frames from %.1fs window (duration=%.1fs)",
        brand_id,
        len(frames),
        trim_window,
        duration,
    )
    return frames


def extract_all(specs: list[BrandSpec], *, only: str | None = None) -> dict[str, list[Path]]:
    """Extract frames for every spec that has a downloaded video.

    Args:
        specs: Corpus rows.
        only: If set, restrict to a single id.

    Returns:
        Mapping ``{id: [frame_path, ...]}``. Missing raw files contribute
        an empty list.
    """
    results: dict[str, list[Path]] = {}
    for spec in specs:
        if only and spec.id != only:
            continue
        video_path = config.RAW_DIR / f"{spec.id}.mp4"
        if not video_path.exists():
            logger.warning("[%s] missing raw video (%s) — skip", spec.id, video_path.name)
            results[spec.id] = []
            continue
        try:
            results[spec.id] = extract_frames(video_path, spec.id)
        except ffmpeg.Error as exc:
            stderr = exc.stderr.decode("utf-8", errors="replace") if exc.stderr else ""
            logger.warning("[%s] ffmpeg error: %s", spec.id, stderr[:200])
            results[spec.id] = []
    return results


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract frames at 1 fps from raw videos.")
    parser.add_argument(
        "--only",
        type=str,
        default=None,
        help="Restrict to a single corpus id.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Emit DEBUG-level logs to stdout.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    config.setup_logging(verbose=args.verbose)
    config.ensure_dirs()
    specs = load_corpus()
    results = extract_all(specs, only=args.only)
    total = sum(len(v) for v in results.values())
    n_brands = sum(1 for v in results.values() if v)
    logger.info(
        "Extract complete: %d frames across %d brands",
        total,
        n_brands,
    )


if __name__ == "__main__":
    main()
