"""
PPE Detection & Safety Compliance — Streamlit application.

Pipeline: YOLO11 object detection (PyTorch) -> class-aware confidence
filtering -> person/PPE association (compliance.py) -> colour-coded
annotation (OpenCV). Video adds ByteTrack multi-object tracking so each
worker keeps a persistent ID, enabling site-level analytics (unique worker
count, compliance timeline, violation report).
"""

import io
import os
import shutil
import subprocess
import tempfile

import cv2
import numpy as np
import pandas as pd
import streamlit as st
import torch
from PIL import Image

import compliance
from model import (
    DEFAULT_CONF,
    DEFAULT_IMGSZ,
    DEFAULT_IOU,
    HELMET_CLASS_NAMES,
    HELMET_CONF,
    HIGH_ACCURACY_IMGSZ,
    PER_CLASS_CONF,
    PPEDetector,
    VEST_CONF,
)

st.set_page_config(
    page_title="PPE Detection & Safety Compliance",
    page_icon="🦺",
    layout="wide",
)

st.markdown(
    """
    <style>
      #MainMenu, footer, [data-testid="stToolbar"] {visibility: hidden;}
      .block-container {padding-top: 2.2rem; padding-bottom: 2rem;}
      h1 {font-weight: 700; letter-spacing: -0.02em; margin-bottom: 0.2rem;}
      [data-testid="stMetric"] {
        background: rgba(128, 128, 128, 0.08);
        border: 1px solid rgba(128, 128, 128, 0.25);
        border-radius: 10px;
        padding: 12px 16px;
      }
    </style>
    """,
    unsafe_allow_html=True,
)

OUTPUT_DIR = "output"


# ----------------------------- helpers -----------------------------
@st.cache_resource(show_spinner="Loading detection model…")
def load_detector():
    return PPEDetector()


def _threshold_overrides(hardhat_conf, vest_conf):
    """Apply the chosen helmet/vest confidences to whichever class names the
    loaded model uses, keeping other per-class defaults."""
    overrides = dict(PER_CLASS_CONF)
    for name in HELMET_CLASS_NAMES:
        overrides[name] = hardhat_conf
    for name in ("Vest", "Safety Vest"):
        overrides[name] = vest_conf
    return overrides


def analyze_image(pil_image, detector, conf, iou, required_ppe, imgsz, augment):
    """Detect, evaluate compliance and annotate. Returns (annotated RGB,
    workers, detections)."""
    result = detector.predict(pil_image, conf=conf, iou=iou, imgsz=imgsz, augment=augment)
    workers, items = compliance.analyze(result, detector.names, required_ppe)
    bgr = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
    annotated_bgr = compliance.annotate(bgr, workers, items)
    return cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB), workers, items


def _open_writer(path, fps, size):
    """Open a VideoWriter, preferring a browser-playable H.264 codec.

    OpenCV's default 'mp4v' produces MPEG-4 Part 2, which most browsers (and
    therefore st.video) refuse to play. We try H.264 first and fall back."""
    for codec in ("avc1", "H264", "mp4v"):
        fourcc = cv2.VideoWriter_fourcc(*codec)
        writer = cv2.VideoWriter(path, fourcc, fps, size)
        if writer.isOpened():
            return writer, codec
        writer.release()
    return None, None


def _reencode_h264(src_path):
    """If ffmpeg is available, re-encode to H.264 so the browser can play it."""
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        return src_path
    dst_path = src_path.replace("_detected.mp4", "_detected_h264.mp4")
    try:
        subprocess.run(
            [ffmpeg, "-y", "-i", src_path, "-vcodec", "libx264",
             "-pix_fmt", "yuv420p", "-movflags", "+faststart", dst_path],
            check=True, capture_output=True,
        )
        return dst_path
    except (subprocess.CalledProcessError, OSError):
        return src_path


def process_video(video_path, detector, conf, iou, required_ppe, frame_stride, use_tracking, imgsz):
    """Run detection (+tracking) over a video, writing an annotated copy and
    collecting per-frame compliance statistics."""
    if use_tracking:
        detector.reset_tracker()

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError("Could not open the uploaded video. The format may be unsupported.")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    base_name = os.path.splitext(os.path.basename(video_path))[0]
    output_path = os.path.join(OUTPUT_DIR, f"{base_name}_detected.mp4")

    out, codec = _open_writer(output_path, fps / max(frame_stride, 1), (width, height))
    if out is None:
        cap.release()
        raise RuntimeError("Could not initialise a video writer with any available codec.")

    seen_ids = set()
    peak_workers = 0
    frames_analyzed = 0
    violation_frames = 0
    timeline_rows = []      # (time_s, workers, compliant, compliance %)
    violation_rows = []     # (frame, time_s, worker, missing)

    progress = st.progress(0.0, text="Analyzing video…")
    frame_idx = 0
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_idx % frame_stride == 0:
                if use_tracking:
                    result = detector.track(frame, conf=conf, iou=iou, imgsz=imgsz)
                else:
                    result = detector.predict(frame, conf=conf, iou=iou, imgsz=imgsz)
                workers, items = compliance.analyze(result, detector.names, required_ppe)
                out.write(compliance.annotate(frame, workers, items))

                frames_analyzed += 1
                time_s = frame_idx / fps
                peak_workers = max(peak_workers, len(workers))
                seen_ids.update(w.track_id for w in workers if w.track_id is not None)

                if workers and required_ppe:
                    compliant = sum(w.compliant for w in workers)
                    timeline_rows.append((
                        round(time_s, 2), len(workers), compliant,
                        round(100.0 * compliant / len(workers), 1),
                    ))
                    offenders = [w for w in workers if not w.compliant]
                    if offenders:
                        violation_frames += 1
                        for w in offenders:
                            violation_rows.append((
                                frame_idx, round(time_s, 2), w.label, ", ".join(w.missing),
                            ))
            frame_idx += 1
            if total_frames:
                progress.progress(min(frame_idx / total_frames, 1.0), text="Analyzing video…")
    finally:
        cap.release()
        out.release()
        progress.empty()

    if codec == "mp4v":  # not browser-friendly; try to re-encode
        output_path = _reencode_h264(output_path)

    timeline = pd.DataFrame(
        timeline_rows, columns=["Time (s)", "Workers", "Compliant", "Compliance %"]
    )
    violations = pd.DataFrame(
        violation_rows, columns=["Frame", "Time (s)", "Worker", "Missing PPE"]
    )
    return {
        "output_path": output_path,
        "unique_workers": len(seen_ids) if seen_ids else peak_workers,
        "tracked": bool(seen_ids),
        "frames_analyzed": frames_analyzed,
        "violation_frames": violation_frames,
        "avg_compliance": float(timeline["Compliance %"].mean()) if len(timeline) else None,
        "timeline": timeline,
        "violations": violations,
    }


def render_image_results(pil_image, detector, conf, iou, required_ppe, source_name, imgsz, augment):
    """Shared renderer for the image-upload and camera tabs."""
    try:
        annotated, workers, items = analyze_image(
            pil_image, detector, conf, iou, required_ppe, imgsz, augment
        )
    except Exception as e:  # noqa: BLE001 - surface any inference error to the user
        st.error(f"Detection failed: {e}")
        return

    col1, col2 = st.columns(2)
    with col1:
        st.image(pil_image, caption="Original", use_container_width=True)
    with col2:
        st.image(annotated, caption="Analyzed — green: compliant, red: PPE missing, amber: detected equipment",
                 use_container_width=True)

    compliant = sum(w.compliant for w in workers)
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Workers detected", len(workers))
    m2.metric("Compliant", compliant if required_ppe else "—")
    m3.metric("Violations", len(workers) - compliant if required_ppe else "—")
    m4.metric("PPE items detected", len(items))

    if workers and required_ppe:
        rows = []
        for i, w in enumerate(workers, start=1):
            row = {"Worker": w.label if w.track_id is not None else f"Worker {i}"}
            for r in required_ppe:
                row[r] = "✓" if w.required[r] else "✗"
            row["Status"] = "Compliant" if w.compliant else f"Missing: {', '.join(w.missing)}"
            rows.append(row)
        st.subheader("Per-worker compliance")
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    elif not workers and (items or required_ppe):
        st.info("No workers detected in this frame — showing raw PPE detections only.")

    if items or workers:
        with st.expander("All detections"):
            det_rows = [{"Class": "Person", "Confidence": round(w.conf, 3)} for w in workers]
            det_rows += [{"Class": d.name, "Confidence": round(d.conf, 3)} for d in items]
            st.dataframe(pd.DataFrame(det_rows), use_container_width=True, hide_index=True)
    else:
        st.info("No objects detected above the current thresholds. Try lowering the confidence threshold.")

    ok, buf = cv2.imencode(".jpg", cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR))
    if ok:
        st.download_button(
            "Download analyzed image",
            io.BytesIO(buf.tobytes()),
            file_name=f"{source_name}_analyzed.jpg",
            mime="image/jpeg",
        )


# ----------------------------- header -----------------------------
st.title("PPE Detection & Safety Compliance")
st.caption(
    "Detects protective equipment on site imagery and evaluates per-worker compliance. "
    "YOLO11 · PyTorch · OpenCV · ByteTrack multi-object tracking"
)

try:
    detector = load_detector()
except FileNotFoundError as e:
    st.error(str(e))
    st.stop()

# ----------------------------- sidebar -----------------------------
st.sidebar.header("Configuration")

ppe_options = sorted(
    n for n in detector.names.values()
    if n != compliance.PERSON_CLASS and not n.startswith("NO-")
    and n not in ("Safety Cone", "machinery", "vehicle")
)
default_required = [n for n in ("Helmet", "Hardhat", "Vest", "Safety Vest") if n in ppe_options]
required_ppe = st.sidebar.multiselect(
    "Required PPE per worker",
    options=ppe_options,
    default=default_required,
    help="Each detected worker is checked for these items. Workers missing any of them are flagged red.",
)

quality = st.sidebar.radio(
    "Analysis quality",
    ["Standard", "High accuracy"],
    help="High accuracy runs inference at 1280 px (plus test-time augmentation on images) — "
         "substantially better at small or partially visible objects such as glasses and "
         "sideways helmets, at roughly 3-4x the processing time.",
)
imgsz = HIGH_ACCURACY_IMGSZ if quality == "High accuracy" else DEFAULT_IMGSZ
use_tta = quality == "High accuracy"

with st.sidebar.expander("Detection thresholds"):
    conf = st.slider(
        "Confidence threshold", 0.05, 0.95, DEFAULT_CONF, 0.05,
        help="Global floor for all classes. Lower it if objects are missed; raise it if you see spurious boxes.",
    )
    iou = st.slider(
        "IoU (NMS) threshold", 0.1, 0.95, DEFAULT_IOU, 0.05,
        help="Lower it if the same object gets duplicate boxes; raise it if overlapping workers merge into one box.",
    )
    hardhat_conf = st.slider(
        "Helmet confidence", 0.05, 0.95, HELMET_CONF, 0.05,
        help="Raise to suppress hair-as-helmet false positives; lower (with High accuracy) if genuine helmets are missed.",
    )
    vest_conf = st.slider(
        "Vest confidence", 0.05, 0.95, VEST_CONF, 0.05,
        help="Raise to stop shirts/jackets being detected as safety vests; lower if genuine vests are missed.",
    )
    st.caption("See the About tab for the full threshold tuning guide.")

with st.sidebar.expander("Video options"):
    use_tracking = st.toggle(
        "Worker tracking (ByteTrack)", value=True,
        help="Assigns a persistent ID to each worker across frames, enabling unique-worker counts.",
    )
    frame_stride = st.slider("Process every Nth frame", 1, 10, 1,
                             help="Higher = faster processing, lower temporal resolution.")

st.sidebar.divider()
device = "CUDA GPU" if torch.cuda.is_available() else "CPU"
st.sidebar.caption(
    f"**Model:** YOLO11 · {len(detector.names)} classes\n\n"
    f"**Inference device:** {device}\n\n"
    f"**Classes:** {', '.join(detector.names.values())}"
)

detector.per_class_conf = _threshold_overrides(hardhat_conf, vest_conf)

# ----------------------------- tabs -----------------------------
tab_image, tab_video, tab_camera, tab_about = st.tabs(
    ["Image Analysis", "Video Analysis", "Live Capture", "About"]
)

with tab_image:
    uploaded_image = st.file_uploader("Upload a site image", type=["jpg", "jpeg", "png"])
    if uploaded_image:
        image = Image.open(uploaded_image).convert("RGB")
        render_image_results(
            image, detector, conf, iou, required_ppe,
            source_name=os.path.splitext(uploaded_image.name)[0],
            imgsz=imgsz, augment=use_tta,
        )
    else:
        st.info("Upload a JPG or PNG image of a work site to run PPE detection and compliance analysis.")

with tab_video:
    # NOTE: we deliberately do NOT pass `type=` here. Newer Streamlit versions
    # validate the uploaded file's *browser-reported* MIME type against that
    # list, and some browsers report a perfectly valid .mp4 with a type the
    # dropzone then rejects ("video/mp4 files are not allowed"). Accepting any
    # file and validating the extension ourselves makes MP4 uploads reliable.
    ALLOWED_VIDEO_EXTS = (".mp4", ".mov", ".avi", ".mkv", ".m4v", ".mpeg", ".mpg")
    uploaded_video = st.file_uploader("Upload a site video", help="MP4, MOV, AVI, MKV, M4V, MPEG/MPG")

    if uploaded_video:
        ext = os.path.splitext(uploaded_video.name)[1].lower()
        if ext not in ALLOWED_VIDEO_EXTS:
            st.error(
                f"Unsupported video type '{ext or 'unknown'}'. "
                f"Please upload one of: {', '.join(ALLOWED_VIDEO_EXTS)}."
            )
        else:
            # Invalidate stale results when a different file is uploaded.
            if st.session_state.get("video_name") != uploaded_video.name:
                st.session_state.pop("video_result", None)
                st.session_state["video_name"] = uploaded_video.name

            st.video(uploaded_video)
            if st.button("Analyze video", type="primary"):
                temp_input_path = None
                try:
                    # Write to a temp file and CLOSE it before OpenCV reads it
                    # (Windows locks open file handles).
                    with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as temp_input:
                        temp_input.write(uploaded_video.read())
                        temp_input_path = temp_input.name
                    st.session_state["video_result"] = process_video(
                        temp_input_path, detector, conf, iou, required_ppe,
                        frame_stride, use_tracking, imgsz,
                    )
                except Exception as e:  # noqa: BLE001
                    st.error(f"Video processing failed: {e}")
                finally:
                    if temp_input_path and os.path.exists(temp_input_path):
                        try:
                            os.remove(temp_input_path)
                        except OSError:
                            pass

            result = st.session_state.get("video_result")
            if result:
                m1, m2, m3, m4 = st.columns(4)
                m1.metric(
                    "Unique workers" if result["tracked"] else "Peak workers",
                    result["unique_workers"],
                )
                m2.metric(
                    "Avg compliance",
                    f"{result['avg_compliance']:.0f}%" if result["avg_compliance"] is not None else "—",
                )
                m3.metric("Violation frames", result["violation_frames"])
                m4.metric("Frames analyzed", result["frames_analyzed"])

                st.video(result["output_path"])
                with open(result["output_path"], "rb") as f:
                    st.download_button(
                        "Download analyzed video", f,
                        file_name=os.path.basename(result["output_path"]),
                    )

                if len(result["timeline"]):
                    st.subheader("Compliance timeline")
                    st.line_chart(
                        result["timeline"].set_index("Time (s)")[["Compliance %"]],
                        height=240,
                    )
                if len(result["violations"]):
                    st.subheader("Violation report")
                    st.dataframe(result["violations"], use_container_width=True, hide_index=True)
                    st.download_button(
                        "Download violation report (CSV)",
                        result["violations"].to_csv(index=False).encode(),
                        file_name="ppe_violation_report.csv",
                        mime="text/csv",
                    )
                elif required_ppe and len(result["timeline"]):
                    st.success("No PPE violations detected in this video.")
    else:
        st.info(
            "Upload a video to run frame-by-frame PPE detection with worker tracking, "
            "a compliance timeline and a downloadable violation report."
        )

with tab_camera:
    st.caption("Capture a frame from your camera for instant on-the-spot compliance checks.")
    snapshot = st.camera_input("Take a photo")
    if snapshot:
        image = Image.open(snapshot).convert("RGB")
        render_image_results(image, detector, conf, iou, required_ppe,
                             source_name="camera_capture", imgsz=imgsz, augment=use_tta)

with tab_about:
    st.markdown(
        f"""
### How it works

**1 — Detection.** A YOLO11 model (PyTorch) fine-tuned on construction-site imagery
detects {len(detector.names)} classes: {', '.join(detector.names.values())}.

**2 — Class-aware confidence filtering.** Each class can carry its own confidence
floor. The helmet class is held to a stricter bar (default {HELMET_CONF}) than other
classes, which suppresses the classic *hair-detected-as-helmet* false positive
without retraining.

**3 — Compliance analysis.** Every detected PPE item is associated to the worker
whose bounding box contains the largest share of it (geometric containment ≥
{compliance.CONTAINMENT_THRESHOLD:.0%}). Head protection must additionally sit in the
upper half of the worker's box, so a helmet carried in a hand does not count as worn.
Each worker is then judged against the configurable *Required PPE* list.

**4 — Tracking (video).** ByteTrack assigns persistent IDs across frames, enabling
unique-worker counts, a per-second compliance timeline, and a violation report
that names the worker and the missing equipment.

### Model performance (validation set)

| Class | Precision | Recall | mAP@50 | mAP@50-95 |
|---|---|---|---|---|
| **All** | 0.967 | 0.876 | **0.925** | 0.730 |
| Helmet | 0.988 | 0.852 | 0.920 | 0.670 |
| Vest | 0.979 | 0.875 | 0.952 | 0.739 |
| Person | 0.936 | 0.936 | 0.946 | 0.857 |
| Boots | 0.965 | 0.842 | 0.880 | 0.656 |

*YOLO11 trained for 100 epochs at 640 px. The full training pipeline
(augmentation for lighting/scale/viewpoint variation, cosine LR schedule,
early stopping) lives in `train.py` / `train_colab.ipynb`; `evaluate.py`
reproduces these metrics.*

### Threshold tuning guide

| Symptom | Adjustment |
|---|---|
| Objects clearly present but not detected | Switch **Analysis quality** to *High accuracy*; if still missed, lower **Confidence threshold** to 0.20–0.30 |
| Glasses / small items missed | *High accuracy* mode — small objects need the higher inference resolution far more than a lower threshold |
| Helmets missed at odd angles (sideways, partial) | *High accuracy* mode first; then lower **Helmet confidence** to 0.40–0.45 |
| Hair or bare heads detected as helmets | Raise **Helmet confidence** to 0.60–0.70 |
| Shirts / jackets detected as safety vests | Raise **Vest confidence** to 0.55–0.65 |
| Genuine vests flagged as missing | Lower **Vest confidence** to 0.35–0.45 |
| Same object gets two overlapping boxes | Lower **IoU threshold** to 0.3–0.4 |
| Two workers standing close merge into one box | Raise **IoU threshold** to 0.6–0.7 |
| Random boxes on background clutter | Raise **Confidence threshold** to 0.45–0.55 |

General principle: **confidence** trades missed detections (false negatives) against
spurious ones (false positives); **resolution** governs small-object recall; **IoU**
controls how aggressively overlapping boxes are merged. Threshold tuning cannot fix a
domain gap — animated/CGI footage, thermal imagery or unusual camera angles need
retraining on matching data (see `train.py`).

### Tech stack

Python · PyTorch · Ultralytics YOLO11 · OpenCV · ByteTrack · Streamlit · pandas
        """
    )
