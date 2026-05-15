# Color Fingerprint of Czech Advertising

A reproducible Python pipeline that ingests Czech TV advertising spots from
public YouTube channels, extracts dominant colors per frame using K-means in
CIE-LAB color space, computes brand-level fingerprint metrics, and renders
the results as a single-page HTML report.

**Live report:** [`outputs/index.html`](outputs/index.html)

---

## What this is

A bridge artifact between data engineering and the visual language of
advertising creative. Three things distinguish it from a generic analytics
notebook:

- **Real pipeline, not a one-off script.** Each stage (`fetch / extract /
  color / metrics / render`) is an independent, idempotent module with type
  hints, structured logging, and tests on the critical path.
- **Honest about its tradeoffs.** Every interpretive choice (sampling rate,
  color space, anchor definition, sample size) is documented in the
  Limitations section below.
- **Speaks the language of creative.** The output is a minimalist HTML
  report designed to be read, not just dashboarded: color strips that look
  like contact sheets, a scatter you can read in three seconds.

---

## Quickstart

Prerequisites: `ffmpeg`, Python ≥ 3.11, `uv`.

```bash
make install          # uv sync (also installs Python 3.11 if needed)
make all              # full pipeline: fetch → extract → color → metrics → render
open outputs/index.html
```

Or run each stage independently. Run `make help` for the full target list.
Cache makes re-runs cheap: `fetch` skips spots already in `data/raw/`,
`color` re-uses extracted frames, etc.

```bash
make fetch            # YouTube downloads via yt-dlp
make extract          # 1 fps frame extraction via ffmpeg-python
make color            # K-means(k=5) in LAB → frames.parquet
make metrics          # per-brand fingerprint → campaigns.parquet
make render           # Jinja2 → outputs/index.html
make test             # pytest
```

---

## Methodology

1. **Fetch.** A curated list of 10 publicly available Czech spots
   (`config/corpus.yaml`, six sectors) is resolved against YouTube via
   `yt-dlp`. Channel-scoped search is used when a channel URL is provided,
   global `ytsearch1:` otherwise. Spots are filtered to 15–180s duration
   and view-count > 10k to bias toward actual TV creative.

2. **Extract.** Each spot is sampled at 1 fps via `ffmpeg-python`, with the
   first and last 0.5s dropped (logo intros/outros otherwise dominate
   clustering). JPEG q=5 keeps file sizes small without affecting cluster
   centroids.

3. **Color.** Each frame is downsampled to ≤200px edge, converted to
   canonical CIE-LAB (where Euclidean distance approximates perceptual
   difference), and clustered with `KMeans(k=5, random_state=42)`. Near-
   black frames (mean L < 7.84) are skipped: K-means on uniform-dark
   pixel clouds collapses centroids and produces noise.

4. **Metrics.** Three per-brand aggregates:
   - **Palette diversity.** Expected pairwise LAB distance between two
     pixels sampled by cluster prevalence (`Σwᵢwⱼ‖cᵢ-cⱼ‖`).
   - **Saturation arc.** Cluster-weighted HSV saturation per frame, then
     mean and std across frames. The std captures the spot's "tension".
   - **Brand anchor.** Fraction of total cluster weight whose hue is
     within ±15° of the brand's anchor color (circular hue distance).

5. **Render.** A single static HTML page with inline CSS, no JavaScript.
   Per-brand color strips are base64-inlined PNGs (PIL); the scatter,
   hue radials, sector CI plot and dendrogram are inline SVG. Bodoni
   Moda and Manrope are loaded from Google Fonts. Page weight ~110 KB
   (the original brief targeted 80 KB; the added analyses earn the
   extra weight).

---

## Limitations

Every choice trades one fidelity for another. The honest list:

- **1 fps sampling misses flash cuts.** A 3-frame product flash at 24 fps
  is invisible to us. For typical TV cuts (≥1s shots) this is fine; for
  music-video–style editing it under-samples.
- **LAB K-means conflates lighting and pigment.** A red object under blue
  light and a magenta object under white light both contribute the same
  centroid. We measure what's *on screen*, not what's been art-directed.
- **Brand anchor colors are hand-curated.** A brand's "true" color is
  often a system, not a single hex. We picked one canonical hex per brand
  and accept that anchor-share numbers shift if the anchor moves.
- **n = 1 spot per brand (mostly).** Cross-brand contrast is real; intra-
  brand variation is unmeasured except for McDonald's, which contributes
  two spots specifically to make that variation visible.
- **K = 5 is a choice, not a discovery.** Higher k surfaces more accent
  colors but dilutes the palette story. We didn't sweep k.
- **YouTube ≠ broadcast.** Compression and upload pipelines shift color.
  Treat the absolute hex values as advisory; the *relative* per-brand
  comparison is the load-bearing part.

---

## Project structure

```
color-fingerprint-cz/
├── config/
│   ├── corpus.yaml         # curated brand × spot list
│   └── insights.yaml       # observations rendered into the report
├── src/
│   ├── config.py           # paths, constants, logging
│   ├── fetch.py            # yt-dlp wrapper
│   ├── extract.py          # ffmpeg frame extraction
│   ├── color.py            # K-means in LAB
│   ├── metrics.py          # diversity, saturation arc, brand anchor
│   ├── sanity.py           # local-only visual QA
│   └── render.py           # Jinja2 → HTML
├── templates/              # report.html.j2 + style.css.j2
├── tests/                  # pytest suite (19 tests)
├── notebooks/
│   └── 01_sanity_check.ipynb
├── data/
│   ├── raw/                # gitignored (videos)
│   ├── frames/             # gitignored (JPEGs)
│   └── processed/          # parquet, committable
└── outputs/index.html      # the deliverable
```

---

## License & disclaimer

- Code: MIT (see `LICENSE`)
- Derived data in `data/processed/`: CC-BY-4.0
- Source video material is **not** redistributed; only analytical metrics
  are committed

Analysis of publicly available video material. Brands and campaigns are
referenced for analytical purposes only. No agency attribution is implied.

---

Built by [Barbora Šandová](https://datasimply.eu).
