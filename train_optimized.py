"""Enhanced PPE Detection Training with Free Optimization Tools"""
import torch
from ultralytics import YOLO
import albumentations as A
from pathlib import Path

# Configuration
class Config:
    MODEL_BASE = 'yolov8n.pt'
    IMG_SIZE = 640
    BATCH_SIZE = 16
    EPOCHS = 100
    USE_AMP = True  # Mixed Precision (free)
    PATIENCE = 20
    DATA_YAML = 'data.yaml'
    SAVE_DIR = 'runs/train/optimized'

# Advanced free augmentation
def get_training_config():
    return {
        'epochs': Config.EPOCHS,
        'imgsz': Config.IMG_SIZE,
        'batch': Config.BATCH_SIZE,
        'patience': Config.PATIENCE,
        'amp': Config.USE_AMP,
        'cache': True,
        'workers': 8,
        'hsv_h': 0.015,
        'hsv_s': 0.7,
        'hsv_v': 0.4,
        'degrees': 10.0,
        'translate': 0.1,
        'scale': 0.5,
        'perspective': 0.0001,
        'flipud': 0.0,
        'fliplr': 0.5,
        'mosaic': 1.0,
        'mixup': 0.1,
        'copy_paste': 0.1,
        'lr0': 0.01,
        'lrf': 0.01,
        'momentum': 0.937,
        'weight_decay': 0.0005,
        'warmup_epochs': 3.0,
        'box': 7.5,
        'cls': 0.5,
        'dfl': 1.5,
        'val': True,
        'plots': True,
        'save': True,
        'save_period': 10,
        'project': Config.SAVE_DIR,
    }

def train_optimized():
    print("Starting optimized PPE training...")
    model = YOLO(Config.MODEL_BASE)
    results = model.train(data=Config.DATA_YAML, **get_training_config())
    metrics = model.val()
    print(f"Training complete! mAP50: {metrics.box.map50:.4f}")
    return model, metrics

if __name__ == "__main__":
    train_optimized()
