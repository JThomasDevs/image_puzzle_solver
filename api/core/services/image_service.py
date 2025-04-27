from pathlib import Path
import shutil
from typing import List, Dict
from fastapi import HTTPException

# Import backend functionality
from backend.core.detector import ObjectDetector

# Initialize detector
detector = ObjectDetector()

# Define paths
DATASET_DIR = Path(__file__).parent.parent.parent.parent / "data" / "images" / "unprocessed"
DATASET_DIR.mkdir(parents=True, exist_ok=True)

async def list_images() -> List[str]:
    """List all available images"""
    images = []
    for file in DATASET_DIR.glob("*.jpg"):
        if not file.name.startswith("annotated_"):
            images.append(file.name)
    return sorted(images)

async def get_image(image_name: str) -> Dict:
    """Get image details and annotations"""
    image_path = DATASET_DIR / image_name
    if not image_path.exists():
        raise HTTPException(status_code=404, detail="Image not found")

    # Check for annotation file
    annotation_path = image_path.with_suffix('.txt')
    detections = []
    
    if annotation_path.exists():
        with open(annotation_path) as f:
            lines = f.readlines()
            for line in lines:
                if line.strip():
                    class_id, x_center, y_center, width, height = map(float, line.strip().split())
                    class_id = int(class_id)
                    detections.append({
                        'class_id': class_id,
                        'class_name': detector.get_class_name(class_id),
                        'confidence': 1.0,
                        'bbox': {
                            'x_center': x_center,
                            'y_center': y_center,
                            'width': width,
                            'height': height
                        }
                    })
    
    return {"detections": detections}

async def upload_image(file) -> Dict:
    """Upload a new image"""
    file_path = DATASET_DIR / file.filename
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    return {"message": "Image uploaded successfully", "filename": file.filename}

async def save_annotations(image_name: str, annotations: List[Dict]) -> Dict:
    """Save annotations for an image"""
    image_path = DATASET_DIR / image_name
    if not image_path.exists():
        raise HTTPException(status_code=404, detail="Image not found")
        
    # Convert annotations to YOLO format
    detections = [
        {
            'class_id': d['class_id'],
            'x_center': d['bbox']['x_center'],
            'y_center': d['bbox']['y_center'],
            'width': d['bbox']['width'],
            'height': d['bbox']['height']
        }
        for d in annotations
    ]
    
    # Save detections
    label_path = str(image_path.with_suffix('.txt'))
    detector.save_detections(detections, label_path)
    return {"message": "Annotations saved successfully", "label_path": label_path} 