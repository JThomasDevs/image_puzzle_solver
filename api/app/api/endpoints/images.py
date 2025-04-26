from fastapi import APIRouter, HTTPException
from typing import List
import cv2
import os
from pathlib import Path

from app.core.config import settings
from app.models.image import ImageDetection, ImageUpdate, BoundingBox
from collect_training_data import TrainingDataCollector

router = APIRouter()
collector = TrainingDataCollector()

@router.get("/images", response_model=List[str])
async def list_images():
    """List all available images in the dataset"""
    try:
        return [f.name for f in settings.IMAGES_DIR.glob("*.jpg")]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/images/{image_name}", response_model=ImageDetection)
async def get_image_detections(image_name: str):
    """Get detections for a specific image"""
    try:
        image_path = settings.IMAGES_DIR / image_name
        if not image_path.exists():
            raise HTTPException(status_code=404, detail="Image not found")
            
        # Process image
        collector.process_image(str(image_path))
        
        # Read detections from label file
        label_path = image_path.with_suffix('.txt')
        detections = []
        if label_path.exists():
            with open(label_path, 'r') as f:
                for line in f:
                    class_id, x_center, y_center, width, height = map(float, line.strip().split())
                    detections.append(BoundingBox(
                        class_id=int(class_id),
                        x_center=x_center,
                        y_center=y_center,
                        width=width,
                        height=height
                    ))
        
        return ImageDetection(
            image_path=str(image_path),
            detections=detections
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.put("/images/{image_name}", response_model=ImageDetection)
async def update_image_detections(image_name: str, update: ImageUpdate):
    """Update detections for a specific image"""
    try:
        image_path = settings.IMAGES_DIR / image_name
        if not image_path.exists():
            raise HTTPException(status_code=404, detail="Image not found")
            
        # Save updated detections
        label_path = image_path.with_suffix('.txt')
        with open(label_path, 'w') as f:
            for detection in update.detections:
                f.write(f"{detection.class_id} {detection.x_center} {detection.y_center} {detection.width} {detection.height}\n")
            if update.crosswalk_detections:
                for detection in update.crosswalk_detections:
                    f.write(f"{detection.class_id} {detection.x_center} {detection.y_center} {detection.width} {detection.height}\n")
        
        return await get_image_detections(image_name)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) 