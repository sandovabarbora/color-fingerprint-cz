"""Fetch spot videos from YouTube via yt-dlp.

Reads ``config/corpus.yaml``, resolves each entry to a YouTube video URL
via search (preferring channel-scoped search when a ``channel_url`` is
provided), filters by duration and view count, and downloads to
``data/raw/{id}.mp4``.

Reference (yt-dlp Python API): https://github.com/yt-dlp/yt-dlp#embedding-yt-dlp

CLI:
    python -m src.fetch              # fetch everything in corpus
    python -m src.fetch --dry-run    # resolve URLs but don't download
    python -m src.fetch --only mcdonalds_syrova
"""

from __future__ import annotations

import argparse
import logging
import urllib.parse
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml
from yt_dlp import YoutubeDL  # type: ignore[import-untyped]
from yt_dlp.utils import DownloadError, ExtractorError  # type: ignore[import-untyped]

from src import config

logger = logging.getLogger(__name__)

# How many candidates to scan when channel-scoped search returns a list.
# Reach further down the list for channels where the first hit isn't the spot.
CHANNEL_SEARCH_DEPTH: int = 6


@dataclass(frozen=True)
class BrandSpec:
    """One row from corpus.yaml."""

    id: str
    brand: str
    sector: str
    channel_url: str | None
    search_query: str
    expected_year: int
    anchor_hex: str


def brand_kebab_id(brand_display: str) -> str:
    """Convert a brand display name to the kebab-id used in brand-level
    aggregates (matches the logic in metrics.compute_all)."""
    return (
        "".join(c if c.isalnum() else "_" for c in brand_display.lower())
        .strip("_")
        .replace("__", "_")
    )


def load_corpus(path: Path = config.CORPUS_FILE) -> list[BrandSpec]:
    """Parse the corpus YAML into typed BrandSpec records.

    Args:
        path: Path to corpus.yaml.

    Returns:
        Ordered list of BrandSpec entries.

    Raises:
        FileNotFoundError: If the corpus file is missing.
        KeyError: If a required field is missing from a row.
    """
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    return [BrandSpec(**row) for row in raw["brands"]]


def _build_ydl_opts(target_path: Path, dry_run: bool) -> dict[str, Any]:
    """Build the yt-dlp options dict for a single download.

    Format ``best[ext=mp4]/best`` prefers a pre-muxed MP4 (no ffmpeg merge
    step needed) and falls back to whatever 'best' single-file is available.
    """
    return {
        "outtmpl": str(target_path.with_suffix(".%(ext)s")),
        "format": "best[ext=mp4]/best",
        "quiet": True,
        "no_warnings": True,
        "noprogress": True,
        "simulate": dry_run,
        "skip_download": dry_run,
        "noplaylist": True,
        "retries": 3,
        "socket_timeout": 30,
        "ignoreerrors": False,
    }


def _resolve_search(spec: BrandSpec) -> str:
    """Build the yt-dlp source string for a brand row.

    Three modes, in priority order:

    1. **Direct URL** — if ``search_query`` starts with ``http://`` or
       ``https://``, treat it as a YouTube video URL and skip search
       entirely. This is what ``scripts/discover_spots.py`` writes when
       it has resolved a specific spot already.
    2. **Channel-scoped search** — when ``channel_url`` is given, use
       YouTube's per-channel search endpoint to avoid reaction/review
       noise from global search.
    3. **Global search** — fallback ``ytsearch1:`` over the query.
    """
    if spec.search_query.startswith(("http://", "https://")):
        return spec.search_query
    if spec.channel_url:
        base = spec.channel_url.rstrip("/")
        return f"{base}/search?query={urllib.parse.quote(spec.search_query)}"
    return f"ytsearch1:{spec.search_query}"


def fetch_channel_top_n(
    channel_url: str, max_count: int, over_fetch: int | None = None
) -> list[dict[str, Any]]:
    """Walk a channel's ``/videos`` tab and return up to ``max_count``
    info_dicts that pass duration + view filters.

    Args:
        channel_url: e.g. ``https://www.youtube.com/@mcdonaldsczech``.
        max_count: max number of qualifying entries to return.
        over_fetch: how many entries to inspect from the top of the
            channel. Defaults to ``max_count * 3`` to allow rejected
            entries (too long, too few views, non-spot content).

    Returns:
        List of yt-dlp info_dicts (each with ``id``, ``title``,
        ``duration``, ``view_count``, ``webpage_url``). Empty list on
        error or no matches.
    """
    if over_fetch is None:
        over_fetch = max_count * 3
    videos_url = channel_url.rstrip("/") + "/videos"
    opts: dict[str, Any] = {
        "quiet": True,
        "no_warnings": True,
        "skip_download": True,
        "playlist_items": f"1-{over_fetch}",
        "socket_timeout": 30,
    }
    try:
        with YoutubeDL(opts) as ydl:
            info = ydl.extract_info(videos_url, download=False)
    except (DownloadError, ExtractorError) as exc:
        logger.warning("channel %s resolve error: %s", videos_url, exc)
        return []
    entries = (info or {}).get("entries") or []
    passing: list[dict[str, Any]] = []
    for entry in entries:
        if not entry:
            continue
        passed, _reason = _passes_filters(entry)
        if passed:
            passing.append(entry)
        if len(passing) >= max_count:
            break
    return passing


def _passes_filters(info: dict[str, Any]) -> tuple[bool, str]:
    """Apply the brief's runtime filters to a yt-dlp info_dict.

    Returns:
        (passed, reason). reason is empty if passed.
    """
    duration = info.get("duration")
    view_count = info.get("view_count")
    if duration is None:
        return False, "missing duration"
    if not (config.MIN_DURATION_S <= duration <= config.MAX_DURATION_S):
        return False, f"duration {duration}s out of [15, 90]"
    if view_count is None:
        return False, "missing view_count"
    if view_count < config.MIN_VIEW_COUNT:
        return False, f"view_count {view_count} < {config.MIN_VIEW_COUNT}"
    return True, ""


def _resolve_video(spec: BrandSpec) -> dict[str, Any] | None:
    """Resolve a corpus row to a single video info_dict that passes filters.

    For channel-scoped queries we scan up to ``CHANNEL_SEARCH_DEPTH``
    candidates and return the first that passes the duration / view-count
    filters. For ``ytsearch1:`` we accept only the single returned candidate.

    Args:
        spec: Brand corpus row.

    Returns:
        The chosen info_dict, or None if nothing passed filters.
    """
    source = _resolve_search(spec)
    logger.info("[%s] resolving: %s", spec.id, source)

    opts: dict[str, Any] = {
        "quiet": True,
        "no_warnings": True,
        "skip_download": True,
        "noplaylist": True,
        "socket_timeout": 30,
    }
    if spec.channel_url:
        opts["playlist_items"] = f"1-{CHANNEL_SEARCH_DEPTH}"

    try:
        with YoutubeDL(opts) as ydl:
            info = ydl.extract_info(source, download=False)
    except (DownloadError, ExtractorError) as exc:
        logger.warning("[%s] yt-dlp resolve error: %s", spec.id, exc)
        return None

    if info is None:
        logger.warning("[%s] resolver returned None", spec.id)
        return None

    entries = info.get("entries") or [info]
    candidates = [e for e in entries if e]
    if not candidates:
        logger.warning("[%s] empty result set", spec.id)
        return None

    for candidate in candidates:
        passed, reason = _passes_filters(candidate)
        title = (candidate.get("title") or "")[:60]
        logger.debug(
            "[%s]   cand id=%s dur=%ss views=%s title=%r -> %s",
            spec.id,
            candidate.get("id"),
            candidate.get("duration"),
            candidate.get("view_count"),
            title,
            "PASS" if passed else f"reject ({reason})",
        )
        if passed:
            logger.info(
                "[%s] selected id=%s title=%r duration=%ss views=%s",
                spec.id,
                candidate.get("id"),
                title,
                candidate.get("duration"),
                candidate.get("view_count"),
            )
            return candidate

    logger.warning(
        "[%s] no candidate passed filters out of %d (channel-scoped=%s)",
        spec.id,
        len(candidates),
        bool(spec.channel_url),
    )
    return None


def fetch_spot(spec: BrandSpec, *, dry_run: bool = False) -> Path | None:
    """Resolve, filter, and download a single spot.

    Args:
        spec: Brand corpus row.
        dry_run: If True, resolve and filter but skip the actual download.

    Returns:
        Path to the downloaded file (or expected path in dry-run), or
        None if no candidate passed the filters.

    Notes:
        Does not raise on per-spot failure — logs WARNING and returns None
        so a missing video does not break the rest of the pipeline.
    """
    config.ensure_dirs()
    target = config.RAW_DIR / f"{spec.id}.mp4"

    if target.exists() and not dry_run:
        logger.info("[%s] cached, skipping (%s)", spec.id, target.name)
        return target

    chosen = _resolve_video(spec)
    if chosen is None:
        return None

    if dry_run:
        return target

    video_url = chosen.get("webpage_url") or chosen.get("url")
    if not video_url:
        logger.warning("[%s] chosen candidate has no webpage_url", spec.id)
        return None

    opts = _build_ydl_opts(target_path=config.RAW_DIR / spec.id, dry_run=False)
    try:
        with YoutubeDL(opts) as ydl:
            ydl.extract_info(video_url, download=True)
    except (DownloadError, ExtractorError) as exc:
        logger.warning("[%s] download error: %s", spec.id, exc)
        return None

    if not target.exists():
        logger.warning(
            "[%s] yt-dlp claimed success but %s is missing", spec.id, target.name
        )
        return None

    logger.info("[%s] downloaded -> %s", spec.id, target.name)
    return target


def fetch_all(
    specs: list[BrandSpec], *, dry_run: bool = False, only: str | None = None
) -> dict[str, Path | None]:
    """Fetch every spec sequentially.

    Args:
        specs: Brand corpus rows.
        dry_run: Resolve only, don't download.
        only: If provided, fetch only this id (matched against ``BrandSpec.id``).

    Returns:
        Mapping ``{id: Path | None}`` — None means skipped or failed.
    """
    results: dict[str, Path | None] = {}
    for spec in specs:
        if only and spec.id != only:
            continue
        results[spec.id] = fetch_spot(spec, dry_run=dry_run)
    return results


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fetch spot videos from YouTube.")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Resolve and filter spots but skip the actual download.",
    )
    parser.add_argument(
        "--only",
        type=str,
        default=None,
        help="Restrict to a single corpus id (e.g. mcdonalds_syrova).",
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
    specs = load_corpus()
    logger.info("Loaded %d corpus entries", len(specs))
    results = fetch_all(specs, dry_run=args.dry_run, only=args.only)
    n_ok = sum(1 for v in results.values() if v is not None)
    logger.info(
        "Fetch complete: %d/%d succeeded%s",
        n_ok,
        len(results),
        " (dry-run)" if args.dry_run else "",
    )


if __name__ == "__main__":
    main()
