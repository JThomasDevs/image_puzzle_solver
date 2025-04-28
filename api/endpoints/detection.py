from fastapi import APIRouter, HTTPException, UploadFile, File, Body
from typing import List, Dict, Optional
import base64
from pydantic import BaseModel

from ..core.services import detection_service, image_service

router = APIRouter()

class ProcessImageRequest(BaseModel):
    image_b64: Optional[str] = None
    image_path: Optional[str] = None
    processing_params: Optional[Dict] = None

@router.post("/process")
async def process_image(body: ProcessImageRequest):
    """Process an image and return detections with annotated image.
    
    Args:
        image_b64: Base64 encoded image data (optional)
        image_path: Path to image file in data directory (optional)
        processing_params: Optional dictionary of processing parameters
        
    Note: Only one of image_b64 or image_path must be provided.
    """
    image_b64 = body.image_b64
    image_path = body.image_path
    processing_params = body.processing_params

    # Validate input
    if (image_b64 is None) == (image_path is None):
        raise HTTPException(
            status_code=400,
            detail="Please provide exactly one of: image_b64 or image_path"
        )
    
    return await detection_service.process_image(None, image_b64, image_path, processing_params)

@router.post("/upload")
async def upload_and_process(file: UploadFile = File(...)):
    """Upload and process an image.
    
    This endpoint:
    1. First uploads the image using the image service
    2. Then processes it for object detection
    """
    # First upload the image
    upload_response = await image_service.upload_image(file)
    filename = upload_response["filename"]
    
    # Then process it
    return await detection_service.detect_objects(filename)

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