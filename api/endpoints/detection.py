from fastapi import APIRouter, HTTPException, UploadFile, File
from typing import List, Dict, Optional
import base64

from ..core.services import detection_service

router = APIRouter()

@router.post("/process")
async def process_image(
    image_b64: Optional[str] = None,
    image_path: Optional[str] = None,
    processing_params: Optional[Dict] = None
):
    """Process an image and return detections with annotated image.
    
    Args:
        image_b64: Base64 encoded image data (optional)
        image_path: Path to image file (optional)
        processing_params: Optional dictionary of processing parameters
    """
    return await detection_service.process_image(image_b64, image_path, processing_params)

@router.post("/upload")
async def upload_and_process(file: UploadFile = File(...)):
    """Upload and process an image"""
    contents = await file.read()
    image_b64 = base64.b64encode(contents).decode('utf-8')
    return await detection_service.process_image(image_b64=image_b64)

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