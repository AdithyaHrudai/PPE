# Threshold Tuning Guide

How and when to adjust the detection settings in the PPE Detection app
(sidebar → *Analysis quality* and *Detection thresholds*).

## The four controls

| Control | What it does | Default |
|---|---|---|
| **Analysis quality** | Inference resolution. *Standard* runs at 640 px (matches training, fastest). *High accuracy* runs at 1280 px plus test-time augmentation on images — markedly better recall on small or partially visible objects, at ~3–4× the processing time. | Standard |
| **Confidence threshold** | Global floor: detections scoring below it are discarded. Lower → more detections but more false positives. Higher → cleaner output but more misses. | 0.35 |
| **Helmet confidence** | Class-specific floor for helmets, applied on top of the global one. Kept stricter by default because low-confidence helmet guesses are usually hair or bare heads. | 0.55 |
| **Vest confidence** | Class-specific floor for safety vests. Kept stricter by default because shirts and jackets in vest-like colours produce mid-confidence false vests. | 0.50 |
| **IoU (NMS) threshold** | How much two boxes may overlap before non-maximum suppression merges them. | 0.50 |

The glasses class additionally has a built-in permissive floor (0.25) because
genuine glasses detections are small objects that rarely score above 0.4.

## Symptom → adjustment

| Symptom | What to change | Why |
|---|---|---|
| Objects clearly present but not detected | *High accuracy* mode; then lower **Confidence** to 0.20–0.30 | Small/distant objects lose too much detail at 640 px; resolution recovers more than a lower threshold does |
| Glasses (or other small items) missed | *High accuracy* mode | On sample footage, 1280 px + TTA found 4× the glasses of the 640 px baseline |
| Helmets missed at odd angles (sideways, partial, far away) | *High accuracy* mode first; then lower **Helmet confidence** to 0.40–0.45 | Off-angle helmets score in the 0.45–0.55 band that the strict default filters out |
| Hair or bare heads labelled as helmets | Raise **Helmet confidence** to 0.60–0.70 | Hair false positives are almost always low-confidence |
| Shirts/jackets labelled as safety vests | Raise **Vest confidence** to 0.55–0.65 | Genuine vests usually score ≥ 0.5; clothing look-alikes sit below that |
| Genuine vests flagged as "Missing: Vest" | Lower **Vest confidence** to 0.35–0.45 | The stricter floor may be dropping real but occluded vests |
| One object drawn with two overlapping boxes | Lower **IoU** to 0.3–0.4 | More aggressive duplicate suppression |
| Two adjacent workers merged into a single box | Raise **IoU** to 0.6–0.7 | Less aggressive suppression keeps close boxes separate |
| Random boxes on background clutter | Raise **Confidence** to 0.45–0.55 | Clutter detections are low-confidence |

## A practical workflow

1. Start at the defaults in *Standard* quality.
2. If anything visible is being missed, switch to *High accuracy* **before**
   touching any slider — resolution fixes more recall problems than thresholds do.
3. Adjust the one class-specific slider that matches your symptom (helmet or
   vest), in steps of 0.05, re-running on the same image so you can compare.
4. Only move the global confidence when the problem affects *all* classes.

## When tuning is the wrong tool

Thresholds trade off errors the model already makes; they cannot add knowledge.
Retrain instead (see `train.py` / `train_colab.ipynb`) when:

- the footage is **out of domain** — animated/CGI safety videos, thermal or IR
  cameras, extreme fisheye/CCTV angles, night scenes;
- a class fails **consistently** regardless of threshold (e.g. a helmet colour
  or vest style absent from the training data);
- you need classes the model doesn't have (e.g. harnesses, ear muffs vs plugs).
