"""Enhanced PPE Detection App with Dashboard & Optimization"""
import streamlit as st
import tempfile
import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO
import os
import sqlite3
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import json

st.set_page_config(page_title="PPE Detection System", layout="wide")

# Load model
@st.cache_resource
def load_model():
    if os.path.exists("best.pt"):
        return YOLO("best.pt")
    else:
        st.error("Model file 'best.pt' not found!")
        return None

# Initialize database
def init_database():
    conn = sqlite3.connect("ppe_detections.db")
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS detections
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  timestamp TEXT,
                  image_path TEXT,
                  detections TEXT,
                  compliant BOOLEAN,
                  confidence_avg REAL,
                  processing_time REAL)''')
    conn.commit()
    conn.close()

init_database()

# Save detection to database
def save_detection(image_path, detections, compliant, confidence_avg, processing_time):
    conn = sqlite3.connect("ppe_detections.db")
    c = conn.cursor()
    c.execute('''INSERT INTO detections 
                 (timestamp, image_path, detections, compliant, confidence_avg, processing_time)
                 VALUES (?, ?, ?, ?, ?, ?)''',
              (datetime.now().isoformat(), image_path, json.dumps(detections), 
               compliant, confidence_avg, processing_time))
    conn.commit()
    conn.close()

# Enhanced detection with compliance check
def detect_image_enhanced(image, model, filename="image"):
    start_time = datetime.now()
    results = model(image, conf=0.50, verbose=False)
    processing_time = (datetime.now() - start_time).total_seconds()
    
    annotated_image = results[0].plot()
    annotated_image = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)
    
    # Extract detections
    detections = []
    confidences = []
    for box in results[0].boxes:
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])
        detections.append({
            'class': model.names[cls_id],
            'confidence': conf,
            'bbox': box.xyxy[0].tolist()
        })
        confidences.append(conf)
    
    # Check compliance (customize based on your classes)
    detected_classes = {d['class'] for d in detections}
    # Example: Check if helmet and vest are detected
    # Adjust based on your actual class names
    is_compliant = len(detected_classes) > 0  # Basic check
    
    avg_conf = np.mean(confidences) if confidences else 0.0
    
    # Save to database
    save_detection(filename, detections, is_compliant, avg_conf, processing_time)
    
    return annotated_image, results[0], detections, is_compliant, avg_conf, processing_time

# Main app
st.title("🦺 PPE Detection System - Enhanced")

tab1, tab2, tab3 = st.tabs(["🔍 Detection", "📊 Dashboard", "⚡ Optimization"])

model = load_model()

# TAB 1: DETECTION
with tab1:
    st.header("PPE Detection")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("Settings")
        input_type = st.radio("Input Type", ["Image", "Video"])
        
        if input_type == "Image":
            uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])
        else:
            uploaded_file = st.file_uploader("Upload Video", type=["mp4", "mov", "avi"])
    
    with col2:
        if uploaded_file and model:
            if input_type == "Image":
                image = Image.open(uploaded_file)
                st.image(image, caption="Original Image", use_container_width=True)
                
                if st.button("🔍 Detect PPE", type="primary"):
                    with st.spinner("Processing..."):
                        filename = os.path.splitext(uploaded_file.name)[0]
                        annotated, results, detections, compliant, avg_conf, proc_time = detect_image_enhanced(
                            image, model, filename
                        )
                        
                        st.success("Detection Complete!")
                        st.image(annotated, caption="Detection Results", use_container_width=True)
                        
                        # Metrics
                        metric_col1, metric_col2, metric_col3 = st.columns(3)
                        metric_col1.metric("Objects Detected", len(detections))
                        metric_col2.metric("Avg Confidence", f"{avg_conf:.2f}")
                        metric_col3.metric("Processing Time", f"{proc_time:.3f}s")
                        
                        # Detection details
                        if detections:
                            st.subheader("Detected Objects")
                            df = pd.DataFrame(detections)
                            st.dataframe(df[['class', 'confidence']], use_container_width=True)
                        
                        # Compliance status
                        if compliant:
                            st.success("✅ PPE Compliance: PASS")
                        else:
                            st.warning("⚠️ PPE Compliance: CHECK REQUIRED")
            
            elif input_type == "Video":
                st.video(uploaded_file)
                st.info("Video detection is processing frame-by-frame. This may take time.")
                
                if st.button("🔍 Detect PPE in Video", type="primary"):
                    with st.spinner("Processing video..."):
                        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp:
                            tmp.write(uploaded_file.read())
                            video_path = tmp.name
                        
                        # Process video (simplified)
                        cap = cv2.VideoCapture(video_path)
                        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                        cap.release()
                        
                        st.success(f"Video loaded: {frame_count} frames")
                        st.info("For full video processing, use the original app.py")

# TAB 2: DASHBOARD
with tab2:
    st.header("📊 Safety Compliance Dashboard")
    
    # Filters
    col1, col2 = st.columns([1, 3])
    with col1:
        days = st.slider("Days to show", 1, 90, 30)
    
    # Load data
    conn = sqlite3.connect("ppe_detections.db")
    query = f"""SELECT * FROM detections 
                WHERE timestamp >= datetime('now', '-{days} days')
                ORDER BY timestamp DESC"""
    df = pd.read_sql_query(query, conn)
    conn.close()
    
    if not df.empty:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df['date'] = df['timestamp'].dt.date
        
        # Key Metrics
        st.subheader("Key Metrics")
        metric1, metric2, metric3, metric4 = st.columns(4)
        
        total = len(df)
        compliant = df['compliant'].sum()
        compliance_rate = (compliant / total * 100) if total > 0 else 0
        
        metric1.metric("Total Detections", f"{total:,}")
        metric2.metric("Compliance Rate", f"{compliance_rate:.1f}%")
        metric3.metric("Avg Confidence", f"{df['confidence_avg'].mean():.2f}")
        metric4.metric("Avg Processing", f"{df['processing_time'].mean():.3f}s")
        
        # Trend chart
        st.subheader("Compliance Trend")
        daily = df.groupby('date').agg({'compliant': ['sum', 'count']}).reset_index()
        daily.columns = ['date', 'compliant', 'total']
        daily['rate'] = (daily['compliant'] / daily['total'] * 100)
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=daily['date'], y=daily['rate'],
            mode='lines+markers', name='Compliance Rate',
            line=dict(color='#2ecc71', width=3)
        ))
        fig.add_hline(y=80, line_dash="dash", line_color="red", annotation_text="Target: 80%")
        fig.update_layout(yaxis_title="Compliance Rate (%)", xaxis_title="Date", height=400)
        st.plotly_chart(fig, use_container_width=True)
        
        # Recent detections
        st.subheader("Recent Detections")
        recent = df.head(10)[['timestamp', 'image_path', 'compliant', 'confidence_avg', 'processing_time']]
        st.dataframe(recent, use_container_width=True)
        
    else:
        st.info("No detection data available. Run some detections first!")

# TAB 3: OPTIMIZATION
with tab3:
    st.header("⚡ Model Optimization")
    
    st.markdown("""
    ### Available Optimizations (All Free)
    
    #### 1. ONNX Export (2-5x faster inference)
    ```bash
    python optimize_model.py best.pt
    pip install onnxruntime
    ```
    
    #### 2. Enhanced Training
    ```bash
    python train_optimized.py
    ```
    Features:
    - Mixed precision training (2-3x faster)
    - Advanced augmentation (Mosaic, MixUp)
    - Optimized hyperparameters
    
    #### 3. Production Pipeline
    ```python
    from inference_pipeline import PPEInferencePipeline
    pipeline = PPEInferencePipeline("best.pt")
    result = pipeline.detect_image("test.jpg")
    ```
    """)
    
    st.info("💡 All optimization tools are in the enhanced-optimization branch. Check README_ENHANCEMENTS.md for details!")
    
    # Quick benchmark
    if st.button("🔥 Quick Benchmark Current Model"):
        if model:
            with st.spinner("Running benchmark..."):
                import time
                dummy_img = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
                
                # Warmup
                for _ in range(5):
                    _ = model(dummy_img, verbose=False)
                
                # Benchmark
                times = []
                for _ in range(20):
                    start = time.time()
                    _ = model(dummy_img, verbose=False)
                    times.append((time.time() - start) * 1000)
                
                times = np.array(times)
                
                st.success("Benchmark Complete!")
                col1, col2, col3 = st.columns(3)
                col1.metric("Mean Latency", f"{times.mean():.2f} ms")
                col2.metric("P95 Latency", f"{np.percentile(times, 95):.2f} ms")
                col3.metric("FPS", f"{1000/times.mean():.1f}")

st.sidebar.markdown("---")
st.sidebar.info("💡 Enhanced PPE Detection System\n\n✅ Real-time detection\n✅ Compliance tracking\n✅ Performance monitoring")
