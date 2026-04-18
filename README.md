# EuroSAT-Land-Cover-Classification

CNN-based land use classification on EuroSAT — comparing RGB vs. 13-band Sentinel-2 multispectral input.

## Notebooks

- [`notebooks/01_data_exploration.ipynb`](notebooks/01_data_exploration.ipynb) — RGB EDA (class balance, sample grid). Run from the **repository root** so PNGs land in [`figures/`](figures/) (`rgb_class_distribution_train.png`, `rgb_sample_grid.png`).
- [`notebooks/02_data_exploration_multispectral.ipynb`](notebooks/02_data_exploration_multispectral.ipynb) — 13-band MSI EDA (class balance, composites, per-band stats, class-mean spectra, band correlation, RGB–MSI alignment). Same **root** working directory so exports go to [`figures/`](figures/).

## Dependencies

See [`requirements.txt`](requirements.txt). The course spec requires Python and reproducible work; it does not mandate a particular installer.

We use **[uv](https://docs.astral.sh/uv/)** to create the environment and install deps (fast, deterministic resolves, same `requirements.txt` as pip):

```bash
uv venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
uv pip install -r requirements.txt
```

If you prefer classic tooling, a virtualenv plus `pip install -r requirements.txt` is equivalent for this project.
