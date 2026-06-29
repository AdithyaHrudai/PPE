"""
Evaluate a PPE model on the validation/test split and print per-class metrics,
with the Hardhat row highlighted (that's the class we care most about).

Usage:
    python evaluate.py                       # evaluate best.pt on the val split
    python evaluate.py --weights runs/ppe/ppe_yolo11s/weights/best.pt --split test
"""

import argparse

from ultralytics import YOLO


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--weights", default="best.pt")
    p.add_argument("--data", default="data.yaml")
    p.add_argument("--split", default="val", choices=["val", "test"])
    p.add_argument("--imgsz", type=int, default=640)
    args = p.parse_args()

    model = YOLO(args.weights)
    metrics = model.val(data=args.data, split=args.split, imgsz=args.imgsz, verbose=True)

    names = model.names
    print("\n=== Per-class mAP50 / mAP50-95 ===")
    for i, c in enumerate(metrics.box.ap_class_index):
        name = names[int(c)]
        marker = "  <-- HELMET" if name == "Hardhat" else ""
        print(f"{name:>16}: mAP50={metrics.box.ap50[i]:.3f}  "
              f"mAP50-95={metrics.box.ap[i]:.3f}{marker}")

    print(f"\nOverall  mAP50={metrics.box.map50:.3f}  mAP50-95={metrics.box.map:.3f}")


if __name__ == "__main__":
    main()
