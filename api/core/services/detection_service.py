from typing import Dict, List, Optional
from fastapi import HTTPException
from pathlib import Path
import base64
import tempfile
import os
import logging

# Import backend functionality
from backend.core.detector import ObjectDetector

# Initialize detector
detector = ObjectDetector()

# Define paths
UNPROCESSED_DIR = Path(__file__).parent.parent.parent.parent / "backend" / "data" / "images" / "unprocessed"
ANNOTATED_DIR = Path(__file__).parent.parent.parent.parent / "backend" / "data" / "images" / "annotated"

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
            with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as temp_file:
                temp_file.write(image_bytes)
                temp_path = temp_file.name
            logging.info(f"Saved uploaded image to {temp_path}")
        else:
            # Handle direct file path
            temp_path = str(DATASET_DIR / image_path)
            if not os.path.exists(temp_path):
                raise HTTPException(status_code=404, detail=f"Image not found at {temp_path}")
            logging.info(f"Using image at {temp_path}")

        # Process image using detector
        detections = detector.process_image(temp_path)
        
        # Find the annotated image in the annotated directory
        annotated_path = ANNOTATED_DIR / ('annotated_' + Path(temp_path).name)
        if not os.path.exists(annotated_path):
            raise HTTPException(status_code=500, detail=f"Annotated image not found at {annotated_path}")
        
        # Read the annotated image and encode as base64
        with open(annotated_path, "rb") as f:
            img_bytes = f.read()
            img_b64 = base64.b64encode(img_bytes).decode("utf-8")
        
        # Create markdown snippet for chat display
        markdown_snippet = f"![Annotated Image](data:image/jpeg;base64,{img_b64})"
        
        # Clean up temporary files only if we created one
        if image_b64:
            os.unlink(temp_path)
        
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
    return await process_image(image_path=image_name)

async def get_classes() -> Dict:
    """Get available object classes"""
    return {"classes": detector.target_classes}

async def annotate_image(image_name: str) -> Dict:
    """Create an annotated version of the image with detections"""
    return await process_image(image_path=image_name) 