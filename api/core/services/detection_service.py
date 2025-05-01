from typing import Dict, List, Optional
from fastapi import HTTPException, UploadFile
from pathlib import Path
import base64
import tempfile
import os
import logging
import contextlib
import cv2
import numpy as np
from math import cos, sin, radians
import traceback

# Import backend functionality
from backend.core.detector import ObjectDetector
from backend.core.models.image import Image

# Initialize detector
detector = ObjectDetector()

# Define paths
UNPROCESSED_DIR = Path(__file__).parent.parent.parent.parent / "data" / "images" / "unprocessed"
ANNOTATED_DIR = Path(__file__).parent.parent.parent.parent / "data" / "images" / "annotated"

# Create directories if they don't exist
UNPROCESSED_DIR.mkdir(parents=True, exist_ok=True)
ANNOTATED_DIR.mkdir(parents=True, exist_ok=True)

def draw_rotated_rectangle(img, center, width, height, angle, color, thickness):
    """Draw a rotated rectangle on the image"""
    # Convert angle to radians
    angle_rad = radians(angle)
    
    # Calculate the four corners of the rectangle
    w = width * img.shape[1]
    h = height * img.shape[0]
    cx = center[0] * img.shape[1]
    cy = center[1] * img.shape[0]
    
    # Calculate the rotation matrix
    cos_a = cos(angle_rad)
    sin_a = sin(angle_rad)
    
    # Calculate the four corners
    corners = np.array([
        [cx - w/2, cy - h/2],
        [cx + w/2, cy - h/2],
        [cx + w/2, cy + h/2],
        [cx - w/2, cy + h/2]
    ])
    
    # Rotate the corners
    rot_matrix = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
    rotated_corners = np.dot(corners - [cx, cy], rot_matrix.T) + [cx, cy]
    
    # Convert to integer coordinates
    rotated_corners = rotated_corners.astype(int)
    
    # Draw the rotated rectangle
    for i in range(4):
        cv2.line(img, tuple(rotated_corners[i]), tuple(rotated_corners[(i+1)%4]), color, thickness)

def draw_polygon(img, points, color, thickness):
    """Draw a polygon on the image"""
    points = np.array([(int(x * img.shape[1]), int(y * img.shape[0])) for x, y in points], np.int32)
    points = points.reshape((-1, 1, 2))
    cv2.polylines(img, [points], True, color, thickness)

async def process_image(image: Image) -> Dict:
    """Process an image and return detections with annotated image.
    
    Args:
        image: Image object to process
        
    Returns:
        Dict containing:
        - detections: List of detected objects
        - annotated_path: Path to the annotated image
    """
    # Process image
    detections = detector.process_image(image)
    
    # Create annotated image
    annotated = image.copy()
    
    # Draw detections
    for det in detections:
        bbox = det["bbox"]
        
        # Draw either rotated rectangle or polygon based on bbox type
        if "polygon_points" in bbox and bbox["polygon_points"]:
            points = [(p["x"], p["y"]) for p in bbox["polygon_points"]]
            draw_polygon(annotated.data, points, (0, 255, 0), 2)
        else:
            center = (bbox["x_center"], bbox["y_center"])
            width = bbox["width"]
            height = bbox["height"]
            angle = bbox.get("rotation_angle", 0)
            draw_rotated_rectangle(annotated.data, center, width, height, angle, (0, 255, 0), 2)
        
        # Add label
        label = f"{det['class_name']} ({det['confidence']:.2f})"
        if "polygon_points" in bbox and bbox["polygon_points"]:
            x = int(bbox["polygon_points"][0]["x"] * image.shape[1])
            y = int(bbox["polygon_points"][0]["y"] * image.shape[0])
        else:
            x = int((bbox["x_center"] - bbox["width"]/2) * image.shape[1])
            y = int((bbox["y_center"] - bbox["height"]/2) * image.shape[0])
        
        cv2.putText(
            annotated.data,
            label,
            (x, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            2
        )
    
    # Save annotated image
    annotated_path = ANNOTATED_DIR / f'annotated_{image.name}'
    annotated.save(annotated_path)
    logging.info(f"Saved annotated image to {annotated_path}")

    # Save YOLO-format annotation file
    annotation_txt_path = annotated_path.with_suffix('.txt')
    with open(annotation_txt_path, 'w') as f:
        for det in detections:
            bbox = det['bbox']
            f.write(f"{det['class_id']} {bbox['x_center']} {bbox['y_center']} {bbox['width']} {bbox['height']}\n")
    logging.info(f"Saved annotation file to {annotation_txt_path}")
    
    return {
        "detections": detections,
        "annotated_path": str(annotated_path)
    }

async def detect_objects(image_name: str) -> Dict:
    """Run object detection on an image"""
    image_path = UNPROCESSED_DIR / image_name
    if not image_path.exists():
        raise HTTPException(status_code=404, detail="Image not found")
    return await process_image(image_path=str(image_path))

async def get_classes() -> Dict:
    """Get available object classes"""
    return {"classes": detector.target_classes}

async def annotate_image(image_name: str) -> Dict:
    """Create an annotated version of the image with detections"""
    return await process_image(image_path=image_name) 