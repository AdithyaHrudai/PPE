# PPE Detection & Safety Compliance System

Real-time detection of Personal Protective Equipment on industrial/construction-site
imagery, with **per-worker compliance analysis** and **multi-object tracking** — built
with Python, PyTorch, Ultralytics YOLO11, OpenCV and Streamlit.

**Live demo:** https://detect-ppe-online.streamlit.app/

<img width="1271" height="710" alt="PPE detection — image analysis" src="https://github.com/user-attachments/assets/b3a93d01-7622-470a-9cdb-f459b893bb86" />

<img width="1281" height="717" alt="PPE detection — video analysis" src="https://github.com/user-attachments/assets/69741f55-779f-4330-bbcd-a04f2798118a" />

## What it does

- **Object detection** — a YOLO11 model fine-tuned on construction-site imagery detects
  8 classes: Boots, Ear-protection, Glass, Glove, Helmet, Mask, Person, Vest.
- **Per-worker compliance** — every PPE item is geometrically associated to the worker
  wearing it (containment analysis, with a head-region constraint for helmets), and each
  worker is judged against a configurable *Required PPE* list. Compliant workers are
  drawn green, violators red with the missing equipment named.
- **Video analytics with tracking** — ByteTrack assigns persistent worker IDs across
  frames, producing unique-worker counts, a per-second compliance timeline and a
  downloadable **violation report (CSV)**.
- **Class-aware confidence thresholds** — helmet and vest classes run at stricter
  confidence floors (suppressing hair-as-helmet and shirt-as-vest false positives),
  while the small glasses class runs at a permissive floor — all *without retraining*
  and tunable live from the sidebar. See [TUNING_GUIDE.md](TUNING_GUIDE.md).
- **High-accuracy mode** — 1280 px inference with test-time augmentation, substantially
  improving recall on small and partially visible objects (glasses, sideways helmets,
  distant workers) at ~3-4x the processing time.
- **Live capture** — analyze a webcam snapshot for on-the-spot compliance checks.

## Architecture

```
image / video frame
        │
        ▼
 YOLO11 detector (PyTorch)          model.py   — PPEDetector: inference, tracking,
        │                                        class-aware confidence filtering
        ▼
 person ↔ PPE association           compliance.py — containment-based assignment,
        │                                        per-worker compliance verdicts
        ▼
 OpenCV annotation + analytics      app.py     — Streamlit UI, video pipeline
 (colour-coded boxes, timeline,                  (H.264 writer + ffmpeg fallback),
  violation CSV)                                 metrics dashboard
```

## Model performance (validation set)

| Class | Precision | Recall | mAP@50 | mAP@50-95 |
|---|---|---|---|---|
| **All** | 0.967 | 0.876 | **0.925** | 0.730 |
| Helmet | 0.988 | 0.852 | 0.920 | 0.670 |
| Vest | 0.979 | 0.875 | 0.952 | 0.739 |
| Person | 0.936 | 0.936 | 0.946 | 0.857 |
| Boots | 0.965 | 0.842 | 0.880 | 0.656 |

YOLO11, 100 epochs @ 640 px. Reproduce with `python evaluate.py --weights best.pt`.

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

Requires Python 3.10+. The app loads `best.pt` from the project root (override with the
`PPE_MODEL_PATH` environment variable). `ffmpeg` (see `packages.txt`) is optional but
recommended for browser-playable output video.

## Training pipeline

- `train.py` — full training configuration: YOLO11 backbone, cosine LR schedule, early
  stopping, and augmentation tuned for site conditions (lighting/HSV, scale, rotation,
  perspective, mosaic, mixup, copy-paste).
- `train_colab.ipynb` — the same pipeline packaged for a free Colab/Kaggle GPU.
- `evaluate.py` — per-class mAP/precision/recall on the validation or test split.
- `data.yaml` — dataset configuration (YOLO format).

To deploy a newly trained model, replace `best.pt` — the app auto-detects the model's
class names, so no code changes are needed.

## Project structure

```
app.py           Streamlit application (UI, image & video pipelines)
model.py         PPEDetector — YOLO11 wrapper: inference, ByteTrack, per-class thresholds
compliance.py    Person↔PPE association and compliance evaluation
train.py         Training script (GPU)
evaluate.py      Validation metrics
requirements.txt Runtime dependencies (CPU-only torch for cloud deploys)
packages.txt     System packages for Streamlit Cloud (ffmpeg)
```
