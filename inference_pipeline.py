"""Robust Inference Pipeline with Error Handling & Monitoring"""
import cv2
import numpy as np
from ultralytics import YOLO
import json
import sqlite3
from datetime import datetime
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PPEInferencePipeline:
    def __init__(self, model_path="best.pt", db_path="ppe_detections.db"):
        self.model = YOLO(model_path)
        self.db_path = db_path
        self._init_database()
        logger.info(f"Pipeline initialized with {model_path}")
    
    def _init_database(self):
        conn = sqlite3.connect(self.db_path)
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
    
    def detect_image(self, image_path, conf_threshold=0.5):
        try:
            start = datetime.now()
            results = self.model(image_path, conf=conf_threshold, verbose=False)
            processing_time = (datetime.now() - start).total_seconds()
            
            detections = []
            confidences = []
            for box in results[0].boxes:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                detections.append({
                    'class': self.model.names[cls_id],
                    'confidence': conf,
                    'bbox': box.xyxy[0].tolist()
                })
                confidences.append(conf)
            
            # Compliance check
            required_ppe = {'helmet', 'vest'}
            detected = {d['class'] for d in detections}
            is_compliant = required_ppe.issubset(detected)
            
            # Save to database
            self._save_detection(
                str(image_path),
                json.dumps(detections),
                is_compliant,
                np.mean(confidences) if confidences else 0.0,
                processing_time
            )
            
            return {
                'detections': detections,
                'compliant': is_compliant,
                'confidence_avg': np.mean(confidences) if confidences else 0.0,
                'processing_time': processing_time,
                'annotated_image': results[0].plot()
            }
        except Exception as e:
            logger.error(f"Error: {str(e)}")
            return {'error': str(e)}
    
    def _save_detection(self, path, detections, compliant, conf, time):
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute('''INSERT INTO detections 
                     (timestamp, image_path, detections, compliant, confidence_avg, processing_time)
                     VALUES (?, ?, ?, ?, ?, ?)''',
                  (datetime.now().isoformat(), path, detections, compliant, conf, time))
        conn.commit()
        conn.close()
    
    def get_stats(self, days=7):
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute('''SELECT COUNT(*), SUM(CASE WHEN compliant = 1 THEN 1 ELSE 0 END),
                            AVG(confidence_avg), AVG(processing_time)
                     FROM detections
                     WHERE timestamp >= datetime('now', '-' || ? || ' days')''', (days,))
        row = c.fetchone()
        conn.close()
        
        if row and row[0] > 0:
            return {
                'total': row[0],
                'compliant': row[1],
                'compliance_rate': (row[1] / row[0]) * 100,
                'avg_confidence': row[2],
                'avg_time': row[3]
            }
        return None
