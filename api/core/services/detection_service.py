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
    processing_params: Optional[Dict] = None
) -> Dict:
    """Process an image and return detections with annotated image.
    
    Args:
        file: The image file to process (optional)
        image_b64: Base64 encoded image data (optional)
        image_path: Path to image file in data directory (optional)
        processing_params: Optional processing parameters
        
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
        image_name = "annotated_image.jpg"  # Default name for base64 input
    else:
        # Load from file path
        image = cv2.imread(image_path)
        if image is None:
            raise HTTPException(status_code=400, detail=f"Could not load image from path: {image_path}")
        image_name = Path(image_path).name
    
    if image is None:
        raise HTTPException(status_code=400, detail="Invalid image data")
    
    # Process image
    detections = detector.process_image(image)
    
    # Create annotated image
    annotated_image = image.copy()
    for det in detections:
        bbox = det["bbox"]
        x1 = int((bbox["x_center"] - bbox["width"]/2) * image.shape[1])
        y1 = int((bbox["y_center"] - bbox["height"]/2) * image.shape[0])
        x2 = int((bbox["x_center"] + bbox["width"]/2) * image.shape[1])
        y2 = int((bbox["y_center"] + bbox["height"]/2) * image.shape[0])
        
        cv2.rectangle(annotated_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            annotated_image,
            f"{det['class_name']} ({det['confidence']:.2f})",
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            2
        )
    
    # Save annotated image
    annotated_path = ANNOTATED_DIR / ('annotated_' + image_name)
    cv2.imwrite(str(annotated_path), annotated_image)
    logging.info(f"Saved annotated image to {annotated_path}")
    
    # Convert annotated image to base64 for response
    _, buffer = cv2.imencode('.jpg', annotated_image)
    annotated_image_b64 = base64.b64encode(buffer).decode('utf-8')
    
    return {
        "detections": detections,
        "annotated_image_b64": annotated_image_b64
    }

async def detect_objects(image_name: str) -> Dict:
    """Run object detection on an image"""
    image_path = str(UNPROCESSED_DIR / image_name)
    if not Path(image_path).exists():
        raise HTTPException(status_code=404, detail="Image not found")
    return await process_image(image_path=image_path)

async def get_classes() -> Dict:
    """Get available object classes"""
    return {"classes": detector.target_classes}

async def annotate_image(image_name: str) -> Dict:
    """Create an annotated version of the image with detections"""
    return await process_image(image_path=image_name) 