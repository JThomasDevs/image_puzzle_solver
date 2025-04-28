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

@contextlib.contextmanager
def temp_image_file(image_bytes=None):
    """Context manager for handling temporary image files"""
    temp_path = None
    try:
        if image_bytes:
            with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as temp_file:
                temp_file.write(image_bytes)
                temp_path = temp_file.name
            yield temp_path
        else:
            yield None
    finally:
        if temp_path and os.path.exists(temp_path):
            os.unlink(temp_path)

def get_original_image_name(image_path: str) -> str:
    """Get the original image name from a path, handling both regular paths and temp files"""
    image_name = Path(image_path).name
    if image_name.startswith('tmp'):
        # If it's a temp file, try to get the original name from the unprocessed directory
        for file in UNPROCESSED_DIR.glob('*.jpg'):
            if not file.name.startswith('annotated_'):
                return file.name
    return image_name

async def process_image(
    file: Optional[UploadFile] = None,
    image_b64: Optional[str] = None,
    image_path: Optional[str] = None,
    processing_params: Optional[Dict] = None,
    image_name: Optional[str] = None
) -> Dict:
    """Process an image and return detections with annotated image.
    
    Args:
        file: The image file to process (optional)
        image_b64: Base64 encoded image data (optional)
        image_path: Path to image file in data directory (optional)
        processing_params: Optional processing parameters
        image_name: Name to use for the image if using image_b64 (required if image_b64 is provided)
        
    Returns:
        Dict containing:
        - detections: List of detected objects
        - annotated_image_b64: Base64 encoded annotated image
    """
    # Load image from either file, base64, or path
    if file:
        # Read the uploaded file
        contents = await file.read()
        image = cv2.imdecode(np.frombuffer(contents, np.uint8), cv2.IMREAD_COLOR)
        image_name = file.filename
    elif image_b64:
        # Decode base64 image
        image_bytes = base64.b64decode(image_b64)
        image = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
        if not image_name:
            raise HTTPException(status_code=400, detail="image_name must be provided when using image_b64.")
    elif image_path:
        # Load from file path, ensuring it's relative to UNPROCESSED_DIR
        if isinstance(image_path, str):
            image_path = Path(image_path)
        if not image_path.is_absolute():
            image_path = UNPROCESSED_DIR / image_path
        image = cv2.imread(str(image_path))
        if image is None:
            raise HTTPException(status_code=400, detail=f"Could not load image from path: {image_path}")
        image_name = image_path.name
    else:
        raise HTTPException(status_code=400, detail="No valid image source provided. Must provide file, image_b64 with image_name, or image_path.")

    if image is None:
        raise HTTPException(status_code=400, detail="Invalid image data")
    
    # Process image
    detections = detector.process_image(image, image_name=image_name)
    
    # Create annotated image
    annotated_image = image.copy()
    for det in detections:
        bbox = det["bbox"]
        
        # Draw either rotated rectangle or polygon based on bbox type
        if "polygon_points" in bbox and bbox["polygon_points"]:
            # Draw polygon
            points = [(p["x"], p["y"]) for p in bbox["polygon_points"]]
            draw_polygon(annotated_image, points, (0, 255, 0), 2)
        else:
            # Draw rotated rectangle
            center = (bbox["x_center"], bbox["y_center"])
            width = bbox["width"]
            height = bbox["height"]
            angle = bbox.get("rotation_angle", 0)
            draw_rotated_rectangle(annotated_image, center, width, height, angle, (0, 255, 0), 2)
        
        # Add label
        label = f"{det['class_name']} ({det['confidence']:.2f})"
        if "polygon_points" in bbox and bbox["polygon_points"]:
            # Use first point for label position
            x = int(bbox["polygon_points"][0]["x"] * image.shape[1])
            y = int(bbox["polygon_points"][0]["y"] * image.shape[0])
        else:
            x = int((bbox["x_center"] - bbox["width"]/2) * image.shape[1])
            y = int((bbox["y_center"] - bbox["height"]/2) * image.shape[0])
        
        cv2.putText(
            annotated_image,
            label,
            (x, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            2
        )
    
    # Save annotated image
    annotated_path = ANNOTATED_DIR / ('annotated_' + image_name)
    cv2.imwrite(str(annotated_path), annotated_image)
    logging.info(f"Saved annotated image to {annotated_path}")

    # Save YOLO-format annotation file in the annotated directory
    annotation_txt_path = annotated_path.with_suffix('.txt')
    with open(annotation_txt_path, 'w') as f:
        for det in detections:
            bbox = det['bbox']
            class_id = det['class_id']
            x_center = bbox['x_center']
            y_center = bbox['y_center']
            width = bbox['width']
            height = bbox['height']
            f.write(f"{class_id} {x_center} {y_center} {width} {height}\n")
    logging.info(f"Saved annotation file to {annotation_txt_path}")
    
    # Convert annotated image to base64 for response
    _, buffer = cv2.imencode('.jpg', annotated_image)
    annotated_image_b64 = base64.b64encode(buffer).decode('utf-8')
    
    return {
        "detections": detections,
        "annotated_image_b64": annotated_image_b64
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