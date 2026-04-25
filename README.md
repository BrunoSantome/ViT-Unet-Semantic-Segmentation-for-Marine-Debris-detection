# Marine Debris Semantic Segmentation (INM705 Coursework)

Pixel-level semantic segmentation of marine plastic debris on Sentinel-2 satellite imagery, using the [MARIDA](https://zenodo.org/records/5151941) [MARIDA github](https://marine-debris.github.io/) dataset (11-band patches, 256×256, 11 aggregated classes including Marine Debris, Sargassum, Ships, Clouds, Marine Water, etc.).

There is three models in total, in the current repository only the U-Net baseline model was cloned since the evaluation and training of the model are re-used and tuned for the ViT-UNet model. 

- **Random Forest** — Refer to the github [Github](https://marine-debris.github.io/) of MARIDA study for further details on it.
- **U-Net baseline** — Cloning of the original MARIDA U-Net for reference results.
- **ViT-UNet hybrid** — a U-Net decoder built on top of a pretrained ViT encoder (`vit_base_patch16_224` from `timm`), with the patch-embedding layer adapted from 3 RGB channels to 11 Sentinel-2 bands.

## Repository layout

```
src/
  u-net-Baseline/   U-Net model, training, evaluation
  u-net-vit/        ViT-UNet model, training, evaluation, checkpoints
  u-net-vit/wandb   Zip file with the set of runs performed in local and google collab pro
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

1. **Train** — `python src/u-net-vit/train.py --data_path /content/data/MARIDA/` (or the baseline equivalent), important to precise the correct path for the data. Logs to wandb, writes checkpoints under `checkpoints/`.  Parameters can be passed to tune everything, see train.py parser for parameters, default and description.

2. **Evaluate** — `python src/u-net-vit/evaluation.py --predict_masks True` runs the test set and writes per-patch georeferenced class-index `.tif`s to `data/predicted_unet/`. Parameters can be passed to tune everything, see evaluation.py parser for parameters, default and description.



3. **Visualise** — open the generated masks in QGIS (Paletted/Unique values style with the MARIDA palette) or use `notebooks/inference_notebook.ipynb` to render an RGB composite alongside the predicted mask.

## Setup

```bash
pip install -r requirements.txt
```

Download the MARIDA dataset at [Link](https://zenodo.org/records/5151941) and put under data/ folder

Download the Checkpoint from huggingface [Link] (https://huggingface.co/Brunaquen/MARIDA-ViT-UNet-Checkpoint/tree/main)

Put the checkpoint under src/vit-net-vit/checkpoints/best_model/ , the file is too large to upload to github.
