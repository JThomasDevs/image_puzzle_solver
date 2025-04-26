from typing import Dict, List
from fastapi import HTTPException
from pathlib import Path

# Import backend functionality
from backend.core.detector import ObjectDetector

# Initialize detector
detector = ObjectDetector()

# Define paths
DATASET_DIR = Path(__file__).parent.parent.parent.parent / "backend" / "data" / "images" / "train"

async def detect_objects(image_name: str) -> Dict:
    """Run object detection on an image"""
    image_path = DATASET_DIR / image_name
    if not image_path.exists():
        raise HTTPException(status_code=404, detail="Image not found")
        
    try:
        result = detector.process_image(str(image_path))
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

async def get_classes() -> Dict:
    """Get available object classes"""
    return {"classes": detector.target_classes}

async def annotate_image(image_name: str) -> Dict:
    """Create an annotated version of the image with detections"""
    image_path = DATASET_DIR / image_name
    if not image_path.exists():
        raise HTTPException(status_code=404, detail="Image not found")
        
    try:
        result = detector.process_image(str(image_path), save_annotations=True)
        annotated_image = Path(result['annotated_image']).name
        return {
            "annotated_image": annotated_image,
            "detections": result["detections"]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) 