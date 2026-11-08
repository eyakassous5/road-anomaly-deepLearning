# Road Anomaly Detection — Streamlit Demo

This is a friendly local tool for visually checking road images with the trained
**DeepLabV3+ / ResNet34** binary segmentation model. It highlights possible
anomalies, groups them into regions, shows their approximate location and model
confidence, and creates downloadable results.

## Project files

Keep these files in the same folder:

```text
road-anomaly-app-main/
├── app.py
├── requirements.txt
├── README.md
├── deeplabv3p_resnet34.pt
└── examples/                 # optional sample PNG/JPG images
```

The checkpoint is not retrained or modified by the app.

## Requirements

- Python 3.10 recommended
- Internet connection for installing Python packages
- About 1 GB or more of free disk space for packages and caches

## Install and run (Windows)

Open Command Prompt or PowerShell in this folder:

```powershell
py -3.10 -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

Streamlit should open the app in your browser. If it does not, open
`http://localhost:8501`.

## Using the app

1. Upload one or more PNG, JPG, JPEG, or WEBP road images.
2. Use the sidebar to adjust sensitivity, overlay opacity, minimum region size,
   and highlight color. The model output is cached, so these display settings
   update without running inference again.
3. Read the plain-language verdict, inspect the numbered regions in the Result
   tab, and download the mask, full-resolution overlay, HTML report, or CSV
   summary.

The other tabs show the binary mask, a probability heatmap, and an original vs
annotated comparison. Session history keeps the latest analyzed images in the
current browser session. If an `examples/` folder exists, the app also offers a
one-click sample image.

## Model behavior

- Architecture: `smp.DeepLabV3Plus` with a ResNet34 encoder
- Input: RGB image resized to 512 × 288 using bilinear interpolation
- Normalization: ImageNet mean `[0.485, 0.456, 0.406]` and standard deviation
  `[0.229, 0.224, 0.225]`
- Output: one logit map, sigmoid, and a user-adjustable display threshold
  (default `0.5`)

The preprocessing and model weights remain unchanged from the training setup.

## Important limitation

The notebook reports a dataset of only 60 images. Treat this as an experimental
academic prototype, not a production road-safety or autonomous-driving system.
The displayed percentage is the fraction of pixels flagged in the resized
image, not a calibrated probability of danger.

## If installation or checkpoint loading fails

Send the complete terminal error. In particular, if `load_state_dict` reports
missing or unexpected keys, do not edit the checkpoint: align the
`segmentation-models-pytorch` / `timm` versions with the training environment.
