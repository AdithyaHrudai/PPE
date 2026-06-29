# PPE Detection — Improvements & Handoff

## TL;DR
- Fixed several bugs that would crash or misbehave on a live Streamlit deployment.
- Added a **class-aware confidence threshold** that suppresses the *hair-detected-as-helmet*
  false positives **immediately, with no retraining** (verified on the current model).
- Provided a **GPU training notebook** (`train_colab.ipynb`) + local scripts to train a
  stronger model when you have a GPU.

> Nothing has been pushed to GitHub.

---

## 1. Production / "goes live" fixes

| Issue | Where | Fix |
|---|---|---|
| `use_column_width` deprecated/removed in modern Streamlit | `app.py` | Switched to `use_container_width`. |
| `torch.load` (torch ≥ 2.6) refuses to load `best.pt` (`weights_only=True`) | `model.py` | Robust loader with trusted-fallback (`_load_yolo`). |
| Installed `ultralytics 8.1.29` **cannot load** `best.pt` (it is a **YOLO11** model using `C3k2`) | `requirements.txt` | Pinned `ultralytics>=8.3.0`; upgraded locally to 8.4.75. |
| Output video uses `mp4v` → **won't play** in the browser (`st.video`) | `app.py` | Try H.264 (`avc1`) writer; ffmpeg re-encode fallback. |
| Video temp file opened **while still locked** (Windows) and **never deleted** | `app.py` | Write→close before reading; cleaned up in `finally`. |
| `model.py` was broken dead code (loaded a detector as a *classifier* from a missing path) | `model.py` | Replaced with a reusable `PPEDetector`. |
| No error handling / no confidence controls / raw tensor table | `app.py` | Added try/except, conf+IoU+helmet sliders, progress bar, named results table. |

## 2. The hair-as-helmet false positives

**Root cause (confirmed empirically):** the model emits many *low-confidence* `Helmet`
detections on hair/background (e.g. `Helmet 0.05` on an airport ceiling, helmet boxes on
bare heads in indoor scenes). The old app used YOLO's default threshold, so these surfaced.

**Fix (works now, no retraining):** `PPEDetector` enforces a **per-class confidence floor**.
The helmet class (`Helmet` *or* `Hardhat`, whichever the loaded model uses) defaults to
**0.55**, while other classes use the global threshold. Low-confidence helmet guesses on
hair are dropped; genuine helmets (usually high confidence) are kept. Tune it live from the
sidebar slider *"Helmet confidence"*.

**Fix (when you retrain):** training on the repo's `css-data` includes the `NO-Hardhat`
(bare-head) class as hard negatives, plus a larger backbone and lighting/angle augmentation —
all of which further reduce the confusion. See below.

## 3. Training a stronger model

You chose: **GPU notebook** + train on the repo's **`css-data` (10 classes)**.

- **`train_colab.ipynb`** — open in Google Colab or Kaggle (GPU runtime), run top to bottom,
  download `best.pt`. Handles dataset setup (Kaggle input / Roboflow API / manual upload),
  builds `data.yaml`, trains **YOLO11s** (try `yolo11m` for more accuracy), validates with
  per-class metrics (Hardhat highlighted), and shows the confusion matrix.
- **`train.py`** — same configuration for a local GPU machine: `python train.py`
  (defaults to `yolov8s` for compatibility; `--model yolo11s.pt` after `pip install -U ultralytics`).
- **`evaluate.py`** — `python evaluate.py --weights <best.pt>` for per-class metrics.
- **`data.yaml`** — local dataset config (the original `ppe_data.yaml` pointed at dead
  `/kaggle/input` paths).

### Using a newly trained model
The new model will be 10-class (`Hardhat`...). **No app code changes needed** — the app
auto-detects the helmet class name. Just replace `best.pt`, or:
```
set PPE_MODEL_PATH=runs\ppe\ppe_yolo11s\weights\best.pt
```

## 4. Notes / environment
- This dev machine is **CPU-only** (`torch 2.10.0+cpu`); full training must run on a GPU.
  The training pipeline was smoke-tested locally (1 epoch, 2% data) to confirm correctness.
- The deployed `best.pt` is an **8-class** model (`Boots, Ear-protection, Glass, Glove,
  Helmet, Mask, Person, Vest`) — different from the repo's 10-class `css-data`. If you want
  to keep the exact 8 classes, train on the original 8-class dataset instead of `css-data`.
