# Color Fingerprint of Czech Advertising

A reproducible Python pipeline that ingests publicly available Czech advertising
spots from YouTube, extracts dominant colors per frame using K-means in LAB
color space, computes brand-level fingerprint metrics, and renders the results
as a single-page HTML report.

**Status:** scaffold in place — full pipeline coming.

## Quickstart

```bash
# Install dependencies
make install

# Run the whole pipeline (fetch → extract → color → metrics → render)
make all

# Open the report
open outputs/index.html
```

## Make targets

Run `make help` to list all available targets.

## Project structure

```
bubble-color-fingerprint/
├── config/corpus.yaml         # curated brand × spot list
├── src/                       # pipeline modules
├── templates/                 # Jinja2 HTML templates
├── tests/                     # pytest suite
├── notebooks/                 # sanity-check notebooks
├── data/                      # raw + derived data (raw/ and frames/ gitignored)
└── outputs/index.html         # the deliverable
```

## License

- Code: MIT (see `LICENSE`)
- Derived data in `data/processed/`: CC-BY-4.0
- Source video material is NOT redistributed; only analytical metrics are committed

## Disclaimer

Analysis of publicly available video material. Brands and campaigns are
referenced for analytical purposes only. No agency attribution is implied.
