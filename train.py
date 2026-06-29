"""
Train an improved PPE detector.

Goals addressed here:
  1. Higher helmet (Hardhat) accuracy across varied conditions — driven by a
     larger backbone and augmentation that simulates different lighting, scale
     and camera angles.
  2. Fewer "hair detected as helmet" false positives — a higher-capacity model
     separates hair from helmets far better than YOLOv8n, and we lean on the
     NO-Hardhat (bare head) class as a hard negative. Inference-time class-aware
     thresholds in model.py finish the job.

Usage:
    python train.py                          # sensible defaults (yolo11s, 120 epochs)
    python train.py --model yolo11m.pt       # more capacity if you have the GPU
    python train.py --model yolov8s.pt       # fallback if ultralytics < 8.3
    python train.py --epochs 80 --batch 8 --imgsz 640

After training, point the app at the new weights:
    set PPE_MODEL_PATH=runs\ppe\<run-name>\weights\best.pt   (or copy to best.pt)
"""

import argparse
import os

import torch
from ultralytics import YOLO


def parse_args():
    p = argparse.ArgumentParser(description="Train PPE detector")
    p.add_argument("--model", default="yolov8s.pt",
                   help="Base weights. yolov8s/m works on all ultralytics versions. "
                        "For best results upgrade ultralytics (pip install -U ultralytics) "
                        "and pass --model yolo11s.pt or yolo11m.pt.")
    p.add_argument("--data", default="data.yaml")
    p.add_argument("--epochs", type=int, default=120)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--batch", type=int, default=-1,
                   help="-1 = auto batch size (uses ~60%% GPU memory).")
    p.add_argument("--name", default="ppe_yolo11s")
    p.add_argument("--device", default=None, help="cuda device id, or 'cpu'.")
    p.add_argument("--resume", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()

    device = args.device
    if device is None:
        device = 0 if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        print("WARNING: No CUDA GPU detected. Training on CPU will be very slow. "
              "Consider a GPU machine (Colab/Kaggle) for full training.")
        # Keep CPU runs from hanging forever with the heaviest defaults.
        if args.batch == -1:
            args.batch = 8

    model = YOLO(args.model)

    results = model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=device,
        project="runs/ppe",
        name=args.name,
        resume=args.resume,

        # --- training schedule ---
        optimizer="auto",
        cos_lr=True,            # smoother convergence at the tail
        patience=30,            # early stop if val plateaus
        close_mosaic=15,        # disable mosaic for the last epochs -> cleaner boxes
        seed=0,

        # --- loss weighting ---
        # Slightly higher cls weight sharpens Hardhat vs NO-Hardhat vs hair.
        box=7.5,
        cls=0.7,
        dfl=1.5,

        # --- augmentation for "various conditions" ---
        hsv_h=0.015,            # hue
        hsv_s=0.7,              # saturation
        hsv_v=0.5,              # brightness / lighting (indoor, outdoor, glare)
        degrees=10.0,           # camera tilt
        translate=0.1,
        scale=0.5,              # near vs far workers / small helmets
        shear=2.0,
        perspective=0.0005,     # viewpoint variation
        flipud=0.0,
        fliplr=0.5,
        mosaic=1.0,
        mixup=0.1,
        copy_paste=0.1,         # paste helmets/heads into more contexts

        plots=True,
        val=True,
    )

    print("\nTraining complete.")
    # Use the trainer's actual save_dir (ultralytics may nest under runs/detect).
    save_dir = getattr(model.trainer, "save_dir", os.path.join("runs", "ppe", args.name))
    best = os.path.join(str(save_dir), "weights", "best.pt")
    print(f"Best weights: {best}")
    print("To use them in the app, either copy that file to best.pt or set:")
    print("  set PPE_MODEL_PATH=" + best)


if __name__ == "__main__":
    main()
