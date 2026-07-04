"""
Person-level PPE compliance analysis.

Turns raw YOLO detections into a per-worker safety report: each detected
person is associated with the PPE items worn on/around them (by geometric
containment), then judged against a configurable list of required equipment.

Association rule: a PPE item is assigned to the person whose bounding box
contains the largest share of the item's area, provided that share exceeds
``CONTAINMENT_THRESHOLD``. Head protection must additionally sit in the upper
half of the person's box, which stops helmets carried in hands or lying on
the ground nearby from counting as "worn".
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

PERSON_CLASS = "Person"
HEAD_CLASSES = ("Helmet", "Hardhat")

# Share of an item's area that must fall inside a person's box to count as worn.
CONTAINMENT_THRESHOLD = 0.4

# BGR palette (colour-blind-friendly green/red pair, amber for equipment)
_GREEN = (96, 174, 39)    # compliant worker
_RED = (54, 67, 244)      # non-compliant worker
_AMBER = (15, 158, 255)   # PPE item boxes
_GRAY = (160, 160, 160)   # workers when no PPE requirement is configured


@dataclass
class Worker:
    box: np.ndarray            # xyxy, pixels
    conf: float
    track_id: Optional[int]    # persistent ID when tracking, else None
    required: Dict[str, bool]  # required class -> present on this person

    @property
    def compliant(self) -> bool:
        return all(self.required.values())

    @property
    def missing(self) -> List[str]:
        return [name for name, ok in self.required.items() if not ok]

    @property
    def label(self) -> str:
        return f"Worker {self.track_id}" if self.track_id is not None else "Worker"


@dataclass
class Detection:
    name: str
    box: np.ndarray
    conf: float


def _containment(item: np.ndarray, person: np.ndarray) -> float:
    """Fraction of the item box's area that lies inside the person box."""
    ix1 = max(item[0], person[0])
    iy1 = max(item[1], person[1])
    ix2 = min(item[2], person[2])
    iy2 = min(item[3], person[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    area = max(1e-6, (item[2] - item[0]) * (item[3] - item[1]))
    return float(inter / area)


def analyze(result, names, required_ppe: Sequence[str]) -> Tuple[List[Worker], List[Detection]]:
    """Split a YOLO result into workers and PPE detections and evaluate each
    worker against ``required_ppe``. Returns (workers, non-person detections)."""
    boxes = result.boxes
    workers: List[Worker] = []
    items: List[Detection] = []
    if boxes is None or len(boxes) == 0:
        return workers, items

    xyxy = boxes.xyxy.cpu().numpy()
    confs = boxes.conf.cpu().numpy()
    cls_ids = boxes.cls.cpu().numpy().astype(int)
    ids = boxes.id.cpu().numpy().astype(int) if boxes.id is not None else None

    for i, c in enumerate(cls_ids):
        name = names.get(int(c), str(c)) if isinstance(names, dict) else str(c)
        if name == PERSON_CLASS:
            workers.append(Worker(
                box=xyxy[i],
                conf=float(confs[i]),
                track_id=int(ids[i]) if ids is not None else None,
                required={r: False for r in required_ppe},
            ))
        else:
            items.append(Detection(name=name, box=xyxy[i], conf=float(confs[i])))

    if not workers:
        return workers, items

    for item in items:
        if item.name not in required_ppe:
            continue
        scores = [_containment(item.box, w.box) for w in workers]
        best = int(np.argmax(scores))
        if scores[best] < CONTAINMENT_THRESHOLD:
            continue
        worker = workers[best]
        if item.name in HEAD_CLASSES:
            head_limit = worker.box[1] + 0.5 * (worker.box[3] - worker.box[1])
            item_cy = (item.box[1] + item.box[3]) / 2.0
            if item_cy > head_limit:
                continue
        worker.required[item.name] = True

    return workers, items


def annotate(frame_bgr: np.ndarray, workers: List[Worker], items: List[Detection]) -> np.ndarray:
    """Draw colour-coded compliance boxes on a copy of the frame (BGR in/out).

    Workers are green when fully equipped, red when PPE is missing; individual
    PPE detections are drawn in amber with their confidence."""
    img = frame_bgr.copy()
    h, w = img.shape[:2]
    scale = max(0.45, min(w, h) / 900.0)
    thick = max(1, int(round(2 * scale)))

    for det in items:
        x1, y1, x2, y2 = det.box.astype(int)
        cv2.rectangle(img, (x1, y1), (x2, y2), _AMBER, thick)
        _label(img, f"{det.name} {det.conf:.2f}", x1, y1, _AMBER, scale)

    for worker in workers:
        color = _GRAY if not worker.required else (_GREEN if worker.compliant else _RED)
        x1, y1, x2, y2 = worker.box.astype(int)
        cv2.rectangle(img, (x1, y1), (x2, y2), color, thick + 1)
        if not worker.required:
            text = worker.label
        elif worker.compliant:
            text = f"{worker.label} | PPE OK"
        else:
            text = f"{worker.label} | Missing: {', '.join(worker.missing)}"
        _label(img, text, x1, y1, color, scale)

    return img


def _label(img: np.ndarray, text: str, x: int, y: int, color, scale: float) -> None:
    """Filled label box with white text, placed above (or just inside) a box."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    fs = max(0.4, 0.5 * scale)
    ft = max(1, int(round(scale)))
    (tw, th), base = cv2.getTextSize(text, font, fs, ft)
    y_top = y - th - base - 4
    if y_top < 0:
        y_top = y + 2
    cv2.rectangle(img, (int(x), int(y_top)), (int(x) + tw + 6, int(y_top) + th + base + 4), color, -1)
    cv2.putText(img, text, (int(x) + 3, int(y_top) + th + 2), font, fs,
                (255, 255, 255), ft, cv2.LINE_AA)
