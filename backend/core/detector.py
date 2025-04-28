import cv2
import os
from pathlib import Path
import numpy as np
from ultralytics import YOLO
import logging
import tempfile
from math import atan2, degrees

class ObjectDetector:
    def __init__(self):
        # Get the path to the model file relative to this file
        model_path = Path(__file__).parent.parent.parent / 'yolov8n.pt'
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
        
    def detect_crosswalk(self, image):
        """Detect crosswalk using image processing techniques"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=50, minLineLength=50, maxLineGap=10)
        
        if lines is not None:
            parallel_lines = []
            for i in range(len(lines)):
                for j in range(i+1, len(lines)):
                    line1 = lines[i][0]
                    line2 = lines[j][0]
                    angle1 = np.arctan2(line1[3] - line1[1], line1[2] - line1[0])
                    angle2 = np.arctan2(line2[3] - line2[1], line2[2] - line2[0])
                    if abs(angle1 - angle2) < 0.1:
                        parallel_lines.append((line1, line2))
            
            if parallel_lines:
                all_points = []
                for line1, line2 in parallel_lines:
                    all_points.extend([(line1[0], line1[1]), (line1[2], line1[3]),
                                     (line2[0], line2[1]), (line2[2], line2[3])])
                
                if all_points:
                    points = np.array(all_points)
                    x_min, y_min = points.min(axis=0)
                    x_max, y_max = points.max(axis=0)
                    
                    height, width = image.shape[:2]
                    x_center = (x_min + x_max) / 2 / width
                    y_center = (y_min + y_max) / 2 / height
                    box_width = (x_max - x_min) / width
                    box_height = (y_max - y_min) / height
                    
                    return [(self.target_classes['crosswalk'], x_center, y_center, box_width, box_height)]
        
        return []
        
    def process_image(self, image_path_or_array):
        """Process an image and return detections with class names
        
        Args:
            image_path_or_array: Either a path to an image file or a numpy array containing the image
        """
        # Handle numpy array input
        if isinstance(image_path_or_array, np.ndarray):
            self.current_image = image_path_or_array
            image_name = "annotated_image.jpg"  # Default name for numpy array input
        else:
            # Handle file path input
            if not os.path.isabs(image_path_or_array):
                image_path_or_array = str(self.unprocessed_dir / image_path_or_array)
                
            self.current_image = cv2.imread(image_path_or_array)
            if self.current_image is None:
                raise ValueError(f"Could not load image at {image_path_or_array}")
            image_name = Path(image_path_or_array).name
            
        # Run YOLO detection
        results = self.model(self.current_image)
        
        # Get YOLO detections
        detections = []
        for box in results[0].boxes:
            class_id = int(box.cls)
            class_name = self.model.names[class_id]
            
            if class_name in self.target_classes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                height, width = self.current_image.shape[:2]
                
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
        
        # Add crosswalk detections
        crosswalk_detections = self.detect_crosswalk(self.current_image)
        for det in crosswalk_detections:
            detections.append({
                'class_id': det[0],
                'class_name': self.get_class_name(det[0]),
                'bbox': {
                    'x_center': det[1],
                    'y_center': det[2],
                    'width': det[3],
                    'height': det[4],
                    'rotation_angle': 0
                },
                'confidence': 1.0
            })
            
        # Create annotated image
        annotated_image = self.current_image.copy()
        height, width = annotated_image.shape[:2]
        
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
            cv2.rectangle(annotated_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Add label with confidence
            label = f"{det['class_name']} ({det['confidence']:.2f})"
            label_y = max(20, y1 - 10)  # Keep label at least 20px from top
            cv2.putText(annotated_image, label, (x1, label_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        
        # Save annotated image in the annotated directory
        annotated_path = self.annotated_dir / ('annotated_' + image_name)
        cv2.imwrite(str(annotated_path), annotated_image)
        logging.info(f"Saved annotated image to {annotated_path}")
        
        return detections
        
    def save_detections(self, detections, output_path: str):
        """Save detections in YOLO format"""
        with open(output_path, 'w') as f:
            for detection in detections:
                bbox = detection['bbox']
                f.write(f"{detection['class_id']} {bbox['x_center']} {bbox['y_center']} {bbox['width']} {bbox['height']} {bbox.get('rotation_angle', 0)}\n") 