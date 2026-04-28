---
title: EuroSAT RGB Land Cover Classifier
sdk: streamlit
app_file: app/app.py
---

# EuroSAT Land Cover Classification

CNN-based land cover classification on EuroSAT, comparing RGB imagery with 13-band Sentinel-2 multispectral input.

## Streamlit RGB Demo

`app/app.py` is a Hugging Face Spaces-ready Streamlit demo for the EuroSAT-RGB ResNet-50 classifier. It shows an Esri World Imagery map centered on Bergen, Norway, lets a user draw a rectangle, fetches the corresponding RGB map tiles, and displays the predicted EuroSAT land cover class plus the top-3 class probabilities.

The RGB model was trained on EuroSAT-RGB tiles, which are about 64x64 pixels and roughly 640m on a side. Predictions on arbitrary map regions are illustrative; for best results, draw a rectangle of roughly 500m-1km on a side over land.

Classes: Annual Crop, Forest, Herbaceous Vegetation, Highway, Industrial Buildings, Pasture, Permanent Crop, Residential Buildings, River, SeaLake.

Validation accuracy on EuroSAT-RGB: **96.8%**.

Main GitHub repo: [davidlsan/EuroSAT-Land-Cover-Classification](https://github.com/davidlsan/EuroSAT-Land-Cover-Classification)

## Run Locally

Place the trained RGB checkpoint at:

```bash
weights/rgb_e15_best.pt
```

Install dependencies with uv and start the app:

```bash
uv sync
uv run streamlit run app/app.py
```

For Hugging Face Spaces, use the Streamlit SDK and include `app/app.py`, `app/model_utils.py`, `app/tile_utils.py`, `requirements.txt`, and the checkpoint at `weights/rgb_e15_best.pt`.

## Notebooks

- `[notebooks/01_data_exploration.ipynb](notebooks/01_data_exploration.ipynb)` - RGB EDA (class balance, sample grid). Run from the repository root so PNGs land in `[figures/](figures/)`.
- `[notebooks/02_data_exploration_multispectral.ipynb](notebooks/02_data_exploration_multispectral.ipynb)` - 13-band MSI EDA (class balance, composites, per-band stats, class-mean spectra, band correlation, RGB-MSI alignment).

## Training

Run from the repository root:

```bash
python3 main.py --modality rgb --epochs 15 --batch-size 32 --num-workers 2 --lr 1e-3
```

### CLI Flags

- `--modality`: values `rgb` or `msi`. RGB loads the `blanchon/EuroSAT_RGB` dataset, while MSI loads `blanchon/EuroSAT_MSI`.
- `--epochs`: number of full passes over the training split. Defaults to `15`.
- `--batch-size`: number of samples per batch. Defaults to `32`.
- `--num-workers`: DataLoader worker processes. Defaults to `2`, but `0` is safer for debugging.
- `--lr`: learning rate for Adam. Defaults to `1e-3`.
- `--seed`: seeds Python, NumPy, and PyTorch for best-effort reproducibility.

