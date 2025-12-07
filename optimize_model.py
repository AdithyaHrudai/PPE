"""Model Optimization: ONNX Export & Performance Benchmarking"""
import torch
from ultralytics import YOLO
import os
import time
import numpy as np

def export_to_onnx(model_path="best.pt", simplify=True):
    """Export PyTorch model to ONNX for deployment"""
    print("\n" + "="*60)
    print("Exporting to ONNX format...")
    print("="*60)
    
    model = YOLO(model_path)
    model.export(
        format='onnx',
        simplify=simplify,
        dynamic=False,
        opset=12
    )
    
    onnx_path = model_path.replace('.pt', '.onnx')
    original_size = os.path.getsize(model_path) / (1024 * 1024)
    onnx_size = os.path.getsize(onnx_path) / (1024 * 1024)
    
    print(f"\nOriginal (.pt): {original_size:.2f} MB")
    print(f"ONNX (.onnx): {onnx_size:.2f} MB")
    print(f"\nONNX model saved: {onnx_path}")
    print("\nBenefits:")
    print("  - 2-5x faster inference with ONNX Runtime")
    print("  - Cross-platform deployment")
    print("  - Easy integration with C++/Java")
    print("="*60 + "\n")
    
    return onnx_path

def benchmark_model(model_path, num_runs=100, img_size=640):
    """Benchmark inference performance"""
    print("\n" + "="*60)
    print(f"Benchmarking: {os.path.basename(model_path)}")
    print("="*60)
    
    model = YOLO(model_path)
    dummy_img = np.random.randint(0, 255, (img_size, img_size, 3), dtype=np.uint8)
    
    # Warmup
    print("\nWarming up (10 runs)...")
    for _ in range(10):
        _ = model(dummy_img, verbose=False)
    
    # Benchmark
    print(f"Running {num_runs} inferences...\n")
    times = []
    for i in range(num_runs):
        start = time.time()
        _ = model(dummy_img, verbose=False)
        times.append((time.time() - start) * 1000)
        if (i + 1) % 20 == 0:
            print(f"  Progress: {i+1}/{num_runs}")
    
    times = np.array(times)
    
    print("\nResults:")
    print(f"  Mean:     {times.mean():.2f} ms")
    print(f"  Median:   {np.median(times):.2f} ms")
    print(f"  Std Dev:  {times.std():.2f} ms")
    print(f"  Min:      {times.min():.2f} ms")
    print(f"  Max:      {times.max():.2f} ms")
    print(f"  P50:      {np.percentile(times, 50):.2f} ms")
    print(f"  P95:      {np.percentile(times, 95):.2f} ms")
    print(f"  P99:      {np.percentile(times, 99):.2f} ms")
    print(f"\n  FPS:      {1000/times.mean():.2f}")
    print("="*60 + "\n")
    
    return times

def export_tflite(model_path="best.pt"):
    """Export to TensorFlow Lite for mobile deployment"""
    print("\nExporting to TensorFlow Lite...")
    model = YOLO(model_path)
    model.export(format='tflite')
    print(f"TFLite model saved: {model_path.replace('.pt', '_saved_model')}")

def compare_formats(model_path="best.pt"):
    """Compare different export formats"""
    print("\n" + "#"*60)
    print("#" + " "*20 + "MODEL OPTIMIZATION" + " "*21 + "#")
    print("#"*60 + "\n")
    
    # Benchmark original
    print("[1/3] Benchmarking PyTorch model...")
    pt_times = benchmark_model(model_path)
    
    # Export and benchmark ONNX
    print("[2/3] Exporting to ONNX...")
    onnx_path = export_to_onnx(model_path)
    
    # Summary
    print("[3/3] Summary\n")
    print("Recommendations:")
    print("  1. Use ONNX for production deployment (install: pip install onnxruntime)")
    print("  2. Use TFLite for mobile/edge devices (install: pip install tensorflow)")
    print("  3. Use TensorRT for NVIDIA GPUs (5-10x speedup, free)")
    print("  4. Use OpenVINO for Intel CPUs (2-3x speedup, free)")
    print("\nAll mentioned tools are FREE and open-source!\n")
    print("#"*60 + "\n")

if __name__ == "__main__":
    import sys
    model_path = sys.argv[1] if len(sys.argv) > 1 else "best.pt"
    
    if not os.path.exists(model_path):
        print(f"Error: Model file '{model_path}' not found!")
        sys.exit(1)
    
    compare_formats(model_path)
