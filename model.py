"""
Shared PPE detection module.

This replaces the previous (broken) version of this file, which tried to
``torch.load`` a YOLO checkpoint and use it as an image classifier — that code
referenced a path that does not exist when deployed and would crash on import.

``PPEDetector`` wraps the Ultralytics YOLO model and adds *class-aware
confidence thresholds*. This is the main lever we use to stop human hair (the
top of a bare head) from being reported as a "Hardhat": the Hardhat class is
held to a stricter confidence bar than the other classes, so low-confidence
helmet guesses on hair are dropped while genuine helmets (usually high
confidence) survive.
"""

from __future__ import annotations

import os
from typing import Dict, Optional

import cv2
import numpy as np
import torch
from ultralytics import YOLO


def _load_yolo(weights_path: str) -> YOLO:
    """Load YOLO weights robustly across torch/ultralytics versions.

    torch >= 2.6 changed ``torch.load`` to default ``weights_only=True``, which
    makes older Ultralytics releases (which don't allowlist their model classes)
    crash with an UnpicklingError. We trust our own checkpoint, so on that
    failure we retry with ``weights_only=False`` by temporarily patching
    ``torch.load``. Newer Ultralytics handles this itself and never hits the
    fallback."""
    try:
        return YOLO(weights_path)
    except Exception as exc:  # noqa: BLE001
        if "weights_only" not in str(exc) and "WeightsUnpickler" not in str(exc):
            raise
        original_load = torch.load

        def _trusting_load(*args, **kwargs):
            kwargs.setdefault("weights_only", False)
            return original_load(*args, **kwargs)

        torch.load = _trusting_load
        try:
            return YOLO(weights_path)
        finally:
            torch.load = original_load

# NOTE: the authoritative class names come from the loaded model
# (``self.names``). The list below is only a fallback when a checkpoint has no
# embedded names. The currently deployed best.pt is an 8-class YOLO11 model:
#   {0: Boots, 1: Ear-protection, 2: Glass, 3: Glove, 4: Helmet, 5: Mask,
#    6: Person, 7: Vest}
CLASS_NAMES = [
    "Boots", "Ear-protection", "Glass", "Glove", "Helmet", "Mask", "Person", "Vest",
]

# Names used by different PPE datasets for the head-protection class. We treat
# all of them as "the helmet class" so the strict threshold applies regardless
# of which model is loaded (the deployed 8-class "Helmet" model, or a model
# trained on the repo's css-data where it is "Hardhat").
HELMET_CLASS_NAMES = ("Helmet", "Hardhat")

# Default global confidence. Per-class overrides below take precedence.
DEFAULT_CONF = 0.35
DEFAULT_IOU = 0.5

# Strict default confidence for the helmet class — this is the main lever that
# suppresses "hair detected as helmet" false positives, which tend to be
# low-confidence. Tune from the Streamlit sidebar.
HELMET_CONF = 0.55

# Per-class minimum confidence. Keyed by name; classes not listed use the
# global threshold. "NO-Hardhat" (bare head) is only present in css-data models.
PER_CLASS_CONF: Dict[str, float] = {
    "Helmet": HELMET_CONF,
    "Hardhat": HELMET_CONF,
    "NO-Hardhat": 0.45,
}


def _resolve_model_path(model_path: Optional[str]) -> str:
    """Resolve the weights path relative to this file so it works regardless of
    the process working directory (important on Streamlit Cloud)."""
    if model_path is None:
        model_path = os.environ.get("PPE_MODEL_PATH", "best.pt")
    if os.path.isabs(model_path) and os.path.exists(model_path):
        return model_path
    here = os.path.dirname(os.path.abspath(__file__))
    candidate = os.path.join(here, model_path)
    return candidate if os.path.exists(candidate) else model_path


class PPEDetector:
    """Thin, reusable wrapper around an Ultralytics YOLO PPE model."""

    def __init__(
        self,
        model_path: Optional[str] = None,
        conf: float = DEFAULT_CONF,
        iou: float = DEFAULT_IOU,
        per_class_conf: Optional[Dict[str, float]] = None,
    ):
        resolved = _resolve_model_path(model_path)
        if not os.path.exists(resolved):
            raise FileNotFoundError(
                f"Model weights not found at '{resolved}'. Place 'best.pt' next "
                f"to the app or set the PPE_MODEL_PATH environment variable."
            )
        self.model = _load_yolo(resolved)
        # Prefer the model's own class names if present; fall back to ours.
        self.names = getattr(self.model, "names", None) or {
            i: n for i, n in enumerate(CLASS_NAMES)
        }
        self.conf = conf
        self.iou = iou
        self.per_class_conf = dict(PER_CLASS_CONF if per_class_conf is None else per_class_conf)

    # -- core inference --------------------------------------------------
    def predict(self, source, conf: Optional[float] = None, iou: Optional[float] = None):
        """Run detection and return a single Ultralytics Results object with
        per-class confidence filtering already applied.

        We run inference at the *lowest* threshold among the global and
        per-class values, then filter in post so each class keeps its own bar.
        """
        base_conf = self.conf if conf is None else conf
        base_iou = self.iou if iou is None else iou

        # Lowest threshold we might keep, so nothing we want is pruned early.
        floor_conf = min([base_conf, *self.per_class_conf.values()]) if self.per_class_conf else base_conf

        results = self.model.predict(
            source,
            conf=float(floor_conf),
            iou=float(base_iou),
            verbose=False,
        )
        result = results[0]
        self._apply_per_class_conf(result, base_conf)
        return result

    def _apply_per_class_conf(self, result, base_conf: float):
        """Drop boxes whose confidence is below their class-specific threshold."""
        boxes = result.boxes
        if boxes is None or boxes.data is None or len(boxes) == 0:
            return

        cls_ids = boxes.cls.cpu().numpy().astype(int)
        confs = boxes.conf.cpu().numpy()

        keep = np.ones(len(boxes), dtype=bool)
        for i, (c, p) in enumerate(zip(cls_ids, confs)):
            name = self.names.get(int(c), str(c)) if isinstance(self.names, dict) else CLASS_NAMES[int(c)]
            threshold = self.per_class_conf.get(name, base_conf)
            if p < threshold:
                keep[i] = False

        if not keep.all():
            # Rebuild the Boxes object with only the kept rows.
            import torch

            result.boxes = result.boxes[torch.as_tensor(keep)]

    # -- convenience helpers --------------------------------------------
    @staticmethod
    def annotate_rgb(result) -> np.ndarray:
        """Return the plotted detections as an RGB image (Streamlit-ready)."""
        bgr = result.plot()
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    @staticmethod
    def annotate_bgr(result) -> np.ndarray:
        """Return the plotted detections as a BGR image (OpenCV/VideoWriter-ready)."""
        return result.plot()

    def summary(self, result):
        """Human-readable list of detections: [(class_name, confidence), ...]."""
        boxes = result.boxes
        if boxes is None or len(boxes) == 0:
            return []
        cls_ids = boxes.cls.cpu().numpy().astype(int)
        confs = boxes.conf.cpu().numpy()
        out = []
        for c, p in zip(cls_ids, confs):
            name = self.names.get(int(c), str(c)) if isinstance(self.names, dict) else CLASS_NAMES[int(c)]
            out.append((name, float(p)))
        return out
