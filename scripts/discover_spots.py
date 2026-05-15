"""Scrape brand YouTube channels for qualifying spots.

Reads ``config/brands.yaml`` (brand definitions with channel + anchor),
walks each channel for the last N videos that pass duration + view
filters, and writes ``config/corpus.yaml`` with one row per discovered
spot. Each row stores the direct YouTube URL in ``search_query``, so the
main pipeline re-runs deterministically without re-discovering.

Usage:
    uv run python -m scripts.discover_spots          # default: 5 per brand
    uv run python -m scripts.discover_spots --max 4
    uv run python -m scripts.discover_spots --only mcdonalds
"""

from __future__ import annotations

import argparse
import logging
import re
from pathlib import Path
from typing import Any

import yaml

from src import config
from src.fetch import fetch_channel_top_n

logger = logging.getLogger(__name__)

BRANDS_FILE = config.CONFIG_DIR / "brands.yaml"
OUT_FILE = config.CONFIG_DIR / "corpus.yaml"


def _slug(text: str, max_len: int = 28) -> str:
    """Conservative kebab-slug for filenames."""
    s = re.sub(r"[^a-zA-Z0-9]+", "_", text.lower()).strip("_")
    return s[:max_len].rstrip("_")


def discover(max_per_brand: int, only: str | None = None) -> list[dict[str, Any]]:
    """For each brand in brands.yaml, walk the channel and collect spots.

    Returns a list of corpus rows (compatible with fetch.BrandSpec).
    Rows are ordered by brand, then by channel-position within brand.
    """
    raw = yaml.safe_load(BRANDS_FILE.read_text(encoding="utf-8"))
    brands = raw["brands"]
    rows: list[dict[str, Any]] = []
    for bd in brands:
        if only and bd["id"] != only:
            continue
        if not bd.get("channel_url"):
            logger.warning("[%s] no channel_url, skipping discovery", bd["id"])
            continue
        logger.info("[%s] discovering up to %d spots", bd["id"], max_per_brand)
        infos = fetch_channel_top_n(bd["channel_url"], max_count=max_per_brand)
        if not infos:
            logger.warning("[%s] no qualifying spots found", bd["id"])
            continue
        for idx, info in enumerate(infos, start=1):
            video_id = info.get("id", "?")
            title = info.get("title", "?")
            duration = info.get("duration")
            views = info.get("view_count")
            url = info.get("webpage_url", f"https://www.youtube.com/watch?v={video_id}")
            spot_id = f"{bd['id']}_{idx:02d}_{_slug(title, 20)}"
            rows.append(
                {
                    "id": spot_id,
                    "brand": bd["brand"],
                    "sector": bd["sector"],
                    "channel_url": None,
                    "search_query": url,
                    "expected_year": info.get("release_year") or info.get("upload_date", "")[:4] or 0,
                    "anchor_hex": bd["anchor_hex"],
                }
            )
            logger.info(
                "[%s]   %s  dur=%ss views=%s title=%r",
                bd["id"],
                video_id,
                duration,
                views,
                title[:50],
            )
    return rows


def write_corpus(rows: list[dict[str, Any]], out: Path = OUT_FILE) -> None:
    """Write discovered spots to corpus.yaml in the existing schema."""
    payload = {"brands": rows}
    text = yaml.safe_dump(payload, sort_keys=False, allow_unicode=True, width=120)
    out.write_text(text, encoding="utf-8")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Discover spots per brand from channels.")
    parser.add_argument("--max", type=int, default=5, help="Max spots per brand.")
    parser.add_argument("--only", type=str, default=None, help="Restrict to one brand id.")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    config.setup_logging(verbose=args.verbose)
    rows = discover(max_per_brand=args.max, only=args.only)
    write_corpus(rows)
    logger.info(
        "Wrote %d discovered spots across %d brands -> %s",
        len(rows),
        len({r["brand"] for r in rows}),
        OUT_FILE,
    )


if __name__ == "__main__":
    main()
