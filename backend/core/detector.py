import cv2
import os
from pathlib import Path
import numpy as np
from ultralytics import YOLO
import logging
import tempfile
from math import atan2, degrees
from .models.image import Image

class ObjectDetector:
    def __init__(self):
        # Get the path to the model file relative to this file
        model_path = Path(__file__).parent.parent / 'yolov8n.pt'
        self.model = YOLO(str(model_path))
        self.current_image = None
        
        # Define the classes we're interested in
        self.target_classes = {
            'person': 0,
            'bicycle': 1,
            'car': 2,
            'motorcycle': 3,
            'bus': 5,
            'truck': 7,
            'traffic light': 9,
            'fire hydrant': 10,
            'stop sign': 11,
            'crosswalk': 12
        }
        
        # Create reverse mapping for class names
        self.class_names = {v: k for k, v in self.target_classes.items()}
        
        # Define paths
        self.base_dir = Path(__file__).parent.parent.parent / 'data'
        self.unprocessed_dir = self.base_dir / 'images' / 'unprocessed'
        self.annotated_dir = self.base_dir / 'images' / 'annotated'
        self.train_dir = self.base_dir / 'images' / 'train'
        
        # Create directories if they don't exist
        self.unprocessed_dir.mkdir(parents=True, exist_ok=True)
        self.annotated_dir.mkdir(parents=True, exist_ok=True)
        self.train_dir.mkdir(parents=True, exist_ok=True)
        
    def get_class_name(self, class_id: int) -> str:
        """Convert a class ID back to its string name"""
        return self.class_names.get(class_id, "unknown")
        
    def calculate_rotation_angle(self, box):
        """Calculate the rotation angle of a bounding box based on its shape"""
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        width = x2 - x1
        height = y2 - y1
        
        # If the box is significantly wider than tall, it might be rotated
        if width > height * 1.5:
            # Calculate angle based on the longer side
            angle = degrees(atan2(y2 - y1, x2 - x1))
            return angle
        return 0
        
    def process_image(self, image: Image) -> dict:
        """Process an image and return detections
        
        Args:
            image: Image object to process
        """
        # Run YOLO detection
        results = self.model(image.data)
        
        # Get YOLO detections
        detections = []
        for box in results[0].boxes:
            class_id = int(box.cls)
            class_name = self.model.names[class_id]
            
            if class_name in self.target_classes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                height, width = image.data.shape[:2]
                
                x_center = (x1 + x2) / 2 / width
                y_center = (y1 + y2) / 2 / height
                box_width = (x2 - x1) / width
                box_height = (y2 - y1) / height
                
                # Calculate rotation angle for certain classes
                rotation_angle = 0
                if class_name in ['traffic light', 'stop sign', 'fire hydrant']:
                    rotation_angle = self.calculate_rotation_angle(box)
                
                class_id = self.target_classes[class_name]
                detections.append({
                    'class_id': class_id,
                    'class_name': class_name,
                    'bbox': {
                        'x_center': x_center,
                        'y_center': y_center,
                        'width': box_width,
                        'height': box_height,
                        'rotation_angle': rotation_angle
                    },
                    'confidence': float(box.conf[0])
                })
            
        # Create annotated image
        annotated = image.copy()
        height, width = annotated.data.shape[:2]
        
        # Draw all detections
        for det in detections:
            # Convert normalized coordinates back to pixel coordinates
            x_center = det['bbox']['x_center'] * width
            y_center = det['bbox']['y_center'] * height
            box_width = det['bbox']['width'] * width
            box_height = det['bbox']['height'] * height
            
            x1 = int(x_center - box_width/2)
            y1 = int(y_center - box_height/2)
            x2 = int(x_center + box_width/2)
            y2 = int(y_center + box_height/2)
            
            # Draw rectangle
            cv2.rectangle(annotated.data, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Add label with confidence
            label = f"{det['class_name']} ({det['confidence']:.2f})"
            label_y = max(20, y1 - 10)  # Keep label at least 20px from top
            cv2.putText(annotated.data, label, (x1, label_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        
        # Save annotated image
        annotated_path = self.annotated_dir / f'annotated_{image.name}'
        annotated.save(annotated_path)
        logging.info(f"Saved annotated image to {annotated_path}")
        
        return detections
        
    def save_detections(self, detections, output_path: str):
        """Save detections in YOLO format"""
        with open(output_path, 'w') as f:
            for detection in detections:
                bbox = detection['bbox']
                f.write(f"{detection['class_id']} {bbox['x_center']} {bbox['y_center']} {bbox['width']} {bbox['height']} {bbox.get('rotation_angle', 0)}\n") 