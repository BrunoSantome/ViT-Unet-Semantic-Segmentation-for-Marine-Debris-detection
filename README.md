# Marine Debris Semantic Segmentation (INM705 Coursework)

Pixel-level semantic segmentation of marine plastic debris on Sentinel-2 satellite imagery, using the [MARIDA](https://marine-debris.github.io/) dataset (11-band patches, 256×256, 11 aggregated classes including Marine Debris, Sargassum, Ships, Clouds, Marine Water, etc.).

Two models are implemented and compared:

- **U-Net baseline** — Implementation of the original MARIDA U-Net for reference results.
- **ViT-UNet hybrid** — a U-Net decoder built on top of a pretrained ViT encoder (`vit_base_patch16_224` from `timm`), with the patch-embedding layer adapted from 3 RGB channels to 11 Sentinel-2 bands.

## Repository layout

```
src/
  u-net-Baseline/   U-Net model, training, evaluation
  u-net-vit/        ViT-UNet model, training, evaluation, checkpoints
  utils/            shared metrics, label/colour mappings, dataset assets
  utils/u-netr.py   UNETR architecture model for reference purposes
notebooks/
  marida_runs.ipynb  File used for training models in google collab pro
  marida_runs2.ipynb File used for evaluating models in google collab pro      
  inference_notebook.ipynb  load a checkpoint, evaluate test set, render predictions on a test patch randomly selected or picked with it's id
data/MARIDA/        dataset root (patches/, splits/) - empty put data here
outputs/            generated figures, Marida original paper, 
```

## Workflow

1. **Train** — `python src/u-net-vit/train.py` (or the baseline equivalent). Logs to wandb, writes checkpoints under `checkpoints/`.
2. **Evaluate** — `python src/u-net-vit/evaluation.py --predict_masks True` runs the test set and writes per-patch georeferenced class-index `.tif`s to `data/predicted_unet/`.


3. **Visualise** — open the generated masks in QGIS (Paletted/Unique values style with the MARIDA palette) or use `notebooks/inference_notebook.ipynb` to render an RGB composite alongside the predicted mask.

## Setup

```bash
pip install -r requirements.txt
```
