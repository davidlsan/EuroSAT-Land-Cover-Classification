# EuroSAT Land Cover Classification

CNN-based land cover classification on EuroSAT, comparing RGB imagery with 13-band Sentinel-2 multispectral input.

## Streamlit RGB Demo

`app/streamlit-app.py` is a Hugging Face Spaces-ready Streamlit demo for the EuroSAT-RGB ResNet-50 classifier. It shows an Esri World Imagery map centered on Bergen, Norway, lets a user draw a rectangle, fetches the corresponding RGB map tiles, and displays the predicted EuroSAT land cover class plus the top-3 class probabilities.

The RGB model was trained on EuroSAT-RGB tiles, which are about 64x64 pixels and roughly 640m on a side. Predictions on arbitrary map regions are illustrative; for best results, draw a rectangle of roughly 500m-1km on a side over land.

Classes: Annual Crop, Forest, Herbaceous Vegetation, Highway, Industrial Buildings, Pasture, Permanent Crop, Residential Buildings, River, SeaLake.

Validation accuracy on EuroSAT-RGB: **96.8%**.

Demonstration video: [EuroSAT Streamlit app walkthrough](https://www.youtube.com/watch?v=mTnQ_tIOUjE).

Link to the [Hugging Face Spaces demo](https://huggingface.co/spaces/davidlsan/EuroSAT_RGB_Land_Cover_Classifier).

Main GitHub repo: [davidlsan/EuroSAT-Land-Cover-Classification](https://github.com/davidlsan/EuroSAT-Land-Cover-Classification)

## Reproducibility

The project is intended to be reproducible from a clean checkout with `uv`. Dependencies are pinned in `uv.lock` and exported to `requirements.txt` for Hugging Face Spaces.

1. Install dependencies:

```bash
uv sync
```

2. Train a model from the repository root. The datasets are downloaded from Hugging Face: `blanchon/EuroSAT_RGB` for RGB and `blanchon/EuroSAT_MSI` for multispectral input.

```bash
uv run python main.py --modality rgb --epochs 15 --batch-size 32 --num-workers 2 --lr 1e-3 --seed 42
uv run python main.py --modality msi --epochs 15 --batch-size 32 --num-workers 2 --lr 1e-3 --seed 42
```

3. Evaluate the saved checkpoint:

```bash
uv run python eval.py --modality rgb --ckpt weights/rgb_e15_best.pt --batch-size 32 --num-workers 2 --seed 42
uv run python eval.py --modality msi --ckpt weights/msi_e15_best.pt --batch-size 32 --num-workers 2 --seed 42
```

Evaluation artifacts are written to `eval_results/`, and `eval_results/latest_run.txt` points to the most recent run. The `--seed` flag seeds Python, NumPy, and PyTorch for best-effort reproducibility; exact metrics can still vary slightly across CPU, CUDA, and Apple MPS backends.

## Run The Demo Locally

Place the trained RGB checkpoint at:

```bash
weights/rgb_e15_best.pt
```

Start the app:

```bash
uv run streamlit run app/streamlit-app.py
```

For Hugging Face Spaces, use the Streamlit SDK and include `app/streamlit-app.py`, `app/model_utils.py`, `app/tile_utils.py`, `requirements.txt`, and the checkpoint at `weights/rgb_e15_best.pt`.

## Notebooks

- `[notebooks/01_data_exploration.ipynb](notebooks/01_data_exploration.ipynb)` - RGB EDA (class balance, sample grid). Run from the repository root so PNGs land in `[figures/](figures/)`.
- `[notebooks/02_data_exploration_multispectral.ipynb](notebooks/02_data_exploration_multispectral.ipynb)` - 13-band MSI EDA (class balance, composites, per-band stats, class-mean spectra, band correlation, RGB-MSI alignment).

## CLI Flags

- `--modality`: values `rgb` or `msi`. RGB loads the `blanchon/EuroSAT_RGB` dataset, while MSI loads `blanchon/EuroSAT_MSI`.
- `--epochs`: number of full passes over the training split. Defaults to `15`.
- `--batch-size`: number of samples per batch. Defaults to `32`.
- `--num-workers`: DataLoader worker processes. Defaults to `2`, but `0` is safer for debugging.
- `--lr`: learning rate for Adam. Defaults to `1e-3`.
- `--seed`: seeds Python, NumPy, and PyTorch for best-effort reproducibility.

