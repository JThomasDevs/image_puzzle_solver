from typing import Dict, List, Optional
from fastapi import HTTPException
from pathlib import Path
import base64
import tempfile
import os
import logging
import contextlib

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

async def process_image(image_b64: Optional[str] = None, image_path: Optional[str] = None, processing_params: Optional[Dict] = None) -> Dict:
    """Process an image and return detections with annotated image.
    
    Args:
        image_b64: Base64 encoded image data (optional)
        image_path: Path to image file (optional)
        processing_params: Optional dictionary of processing parameters
        
    Returns:
        Dictionary containing processed image data, detections, and any errors
    """
    logging.info("Received process_image request")
    if not image_b64 and not image_path:
        raise HTTPException(status_code=400, detail="Missing image input. Provide either image_b64 or image_path.")
        
    try:
        if image_b64:
            # Handle base64 image
            image_bytes = base64.b64decode(image_b64)
            with temp_image_file(image_bytes) as temp_path:
                if not temp_path:
                    raise HTTPException(status_code=500, detail="Failed to create temporary file")
                logging.info(f"Saved uploaded image to {temp_path}")
                
                # Process image using detector
                detections = detector.process_image(temp_path)
                
                # Find the annotated image in the annotated directory using the original image name
                original_name = get_original_image_name(temp_path)
                annotated_path = ANNOTATED_DIR / ('annotated_' + original_name)
                if not os.path.exists(annotated_path):
                    raise HTTPException(status_code=500, detail=f"Annotated image not found at {annotated_path}")
                
                # Read the annotated image and encode as base64
                with open(annotated_path, "rb") as f:
                    img_bytes = f.read()
                    img_b64 = base64.b64encode(img_bytes).decode("utf-8")
                
                # Create markdown snippet for chat display
                markdown_snippet = f"![Annotated Image](data:image/jpeg;base64,{img_b64})"
                
                logging.info("Image processed successfully.")
                return {
                    "annotated_image_b64": img_b64,
                    "detections": detections,
                    "markdown_snippet": markdown_snippet,
                    "error": None
                }
        else:
            # Handle direct file path
            if not os.path.exists(image_path):
                raise HTTPException(status_code=404, detail=f"Image not found at {image_path}")
            logging.info(f"Using image at {image_path}")

            # Process image using detector
            detections = detector.process_image(image_path)
            
            # Find the annotated image in the annotated directory
            annotated_path = ANNOTATED_DIR / ('annotated_' + Path(image_path).name)
            if not os.path.exists(annotated_path):
                raise HTTPException(status_code=500, detail=f"Annotated image not found at {annotated_path}")
            
            # Read the annotated image and encode as base64
            with open(annotated_path, "rb") as f:
                img_bytes = f.read()
                img_b64 = base64.b64encode(img_bytes).decode("utf-8")
            
            # Create markdown snippet for chat display
            markdown_snippet = f"![Annotated Image](data:image/jpeg;base64,{img_b64})"
            
            logging.info("Image processed successfully.")
            return {
                "annotated_image_b64": img_b64,
                "detections": detections,
                "markdown_snippet": markdown_snippet,
                "error": None
            }
    except HTTPException:
        raise
    except Exception as e:
        logging.exception("Exception during image processing:")
        raise HTTPException(status_code=500, detail=f"Exception: {str(e)}")

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