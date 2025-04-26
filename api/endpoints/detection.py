from fastapi import APIRouter, HTTPException
from typing import List, Dict

from ..core.services import detection_service

router = APIRouter()

@router.post("/detect/{image_name}")
async def detect_objects(image_name: str):
    """Run object detection on an image"""
    return await detection_service.detect_objects(image_name)

@router.get("/classes")
async def get_classes():
    """Get available object classes"""
    return await detection_service.get_classes()

@router.post("/annotate/{image_name}")
async def annotate_image(image_name: str):
    """Create an annotated version of the image with detections"""
    return await detection_service.annotate_image(image_name) 