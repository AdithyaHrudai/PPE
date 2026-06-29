import os
import shutil
import subprocess
import tempfile

import cv2
import pandas as pd
import streamlit as st
from PIL import Image

from model import (
    DEFAULT_CONF,
    DEFAULT_IOU,
    HELMET_CLASS_NAMES,
    HELMET_CONF,
    PER_CLASS_CONF,
    PPEDetector,
)


def _helmet_overrides(hardhat_conf):
    """Apply the chosen helmet confidence to whichever helmet class name the
    loaded model uses (Helmet / Hardhat), keeping other per-class defaults."""
    overrides = dict(PER_CLASS_CONF)
    for name in HELMET_CLASS_NAMES:
        overrides[name] = hardhat_conf
    return overrides

st.set_page_config(page_title="PPE Detection", page_icon="🦺", layout="wide")

OUTPUT_DIR = "output"


@st.cache_resource(show_spinner="Loading model…")
def load_detector():
    return PPEDetector()


def detect_image(pil_image, detector, conf, iou, hardhat_conf, original_filename="detected_image"):
    detector.per_class_conf = _helmet_overrides(hardhat_conf)
    result = detector.predict(pil_image, conf=conf, iou=iou)
    annotated_rgb = detector.annotate_rgb(result)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    output_path = os.path.join(OUTPUT_DIR, f"{original_filename}_detected.jpg")
    cv2.imwrite(output_path, cv2.cvtColor(annotated_rgb, cv2.COLOR_RGB2BGR))

    return annotated_rgb, detector.summary(result), output_path


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


def detect_video(video_path, detector, conf, iou, hardhat_conf, frame_stride=1):
    detector.per_class_conf = _helmet_overrides(hardhat_conf)

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

    progress = st.progress(0.0, text="Processing video…")
    frame_idx = 0
    written = 0
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_idx % frame_stride == 0:
                result = detector.predict(frame, conf=conf, iou=iou)
                out.write(detector.annotate_bgr(result))
                written += 1
            frame_idx += 1
            if total_frames:
                progress.progress(min(frame_idx / total_frames, 1.0), text="Processing video…")
    finally:
        cap.release()
        out.release()
        progress.empty()

    if codec == "mp4v":  # not browser-friendly; try to re-encode
        output_path = _reencode_h264(output_path)
    return output_path


# ----------------------------- UI -----------------------------
st.title("🦺 Personal Protective Equipment (PPE) Detection")

try:
    detector = load_detector()
except FileNotFoundError as e:
    st.error(str(e))
    st.stop()

st.sidebar.header("Input")
input_type = st.sidebar.radio("Select input type", ["Image", "Video"])

st.sidebar.header("Detection settings")
conf = st.sidebar.slider("Confidence threshold", 0.05, 0.95, DEFAULT_CONF, 0.05)
iou = st.sidebar.slider("IoU (NMS) threshold", 0.1, 0.95, DEFAULT_IOU, 0.05)
hardhat_conf = st.sidebar.slider(
    "Helmet confidence (raise to reduce hair-as-helmet false positives)",
    0.05, 0.95, HELMET_CONF, 0.05,
)

if input_type == "Image":
    uploaded_image = st.sidebar.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"])
    if uploaded_image:
        image = Image.open(uploaded_image).convert("RGB")
        col1, col2 = st.columns(2)
        with col1:
            st.image(image, caption="Original Image", use_container_width=True)

        try:
            filename = os.path.splitext(uploaded_image.name)[0]
            annotated_image, summary, image_output_path = detect_image(
                image, detector, conf, iou, hardhat_conf, original_filename=filename
            )
        except Exception as e:  # noqa: BLE001 - surface any inference error to the user
            st.error(f"Detection failed: {e}")
            st.stop()

        with col2:
            st.image(annotated_image, caption="Detected Image", use_container_width=True)

        st.subheader("Detected Objects")
        if summary:
            df = pd.DataFrame(summary, columns=["Class", "Confidence"])
            df["Confidence"] = df["Confidence"].round(3)
            st.dataframe(df, use_container_width=True)
        else:
            st.info("No PPE objects detected above the current thresholds.")

        with open(image_output_path, "rb") as f:
            st.download_button("Download Detected Image", f, file_name=os.path.basename(image_output_path))

elif input_type == "Video":
    # NOTE: we deliberately do NOT pass `type=` here. Newer Streamlit versions
    # validate the uploaded file's *browser-reported* MIME type against that
    # list, and some browsers report a perfectly valid .mp4 with a type the
    # dropzone then rejects ("video/mp4 files are not allowed"). Accepting any
    # file and validating the extension ourselves makes MP4 uploads reliable.
    ALLOWED_VIDEO_EXTS = (".mp4", ".mov", ".avi", ".mkv", ".m4v", ".mpeg", ".mpg")
    uploaded_video = st.sidebar.file_uploader(
        "Upload a Video", help="MP4, MOV, AVI, MKV, M4V, MPEG/MPG"
    )
    frame_stride = st.sidebar.slider("Process every Nth frame (higher = faster)", 1, 10, 1)
    if uploaded_video:
        ext = os.path.splitext(uploaded_video.name)[1].lower()
        if ext not in ALLOWED_VIDEO_EXTS:
            st.error(
                f"Unsupported video type '{ext or 'unknown'}'. "
                f"Please upload one of: {', '.join(ALLOWED_VIDEO_EXTS)}."
            )
            st.stop()
        st.video(uploaded_video)

        temp_input_path = None
        try:
            # Write to a temp file and CLOSE it before OpenCV reads it
            # (Windows locks open file handles, which broke the previous version).
            suffix = os.path.splitext(uploaded_video.name)[1] or ".mp4"
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_input:
                temp_input.write(uploaded_video.read())
                temp_input_path = temp_input.name

            with st.spinner("Running object detection on video…"):
                output_video_path = detect_video(
                    temp_input_path, detector, conf, iou, hardhat_conf, frame_stride
                )

            st.success("Done! Here's the processed video:")
            st.video(output_video_path)
            with open(output_video_path, "rb") as f:
                st.download_button(
                    "Download Processed Video", f, file_name=os.path.basename(output_video_path)
                )
        except Exception as e:  # noqa: BLE001
            st.error(f"Video processing failed: {e}")
        finally:
            if temp_input_path and os.path.exists(temp_input_path):
                try:
                    os.remove(temp_input_path)
                except OSError:
                    pass
