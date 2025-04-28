from fastapi import APIRouter, HTTPException, UploadFile, File, Body
from typing import List, Dict, Optional
import base64
from pydantic import BaseModel

from ..core.services import detection_service, image_service

router = APIRouter()

class ProcessImageRequest(BaseModel):
    image_b64: Optional[str] = None
    image_name: Optional[str] = None
    image_path: Optional[str] = None
    processing_params: Optional[Dict] = None

@router.post("/process")
async def process_image(body: ProcessImageRequest):
    """Process an image and return detections with annotated image.
    
    Args:
        image_b64: Base64 encoded image data (optional, requires image_name)
        image_name: Name to use for the image if using image_b64 (required if image_b64 is provided)
        image_path: Path to image file in data directory (optional)
        processing_params: Optional dictionary of processing parameters
        
    Note: Provide either image_b64 (with image_name) or image_path.
    """
    image_b64 = body.image_b64
    image_name = body.image_name
    image_path = body.image_path
    processing_params = body.processing_params

    # Validate input
    if image_b64:
        if not image_name:
            raise HTTPException(
                status_code=400,
                detail="When using image_b64, you must also provide image_name."
            )
    elif not image_path:
        raise HTTPException(
            status_code=400,
            detail="Please provide either image_b64 (with image_name) or image_path."
        )
    # If both are provided, prefer image_b64
    return await detection_service.process_image(None, image_b64, image_path, processing_params, image_name=image_name)

@router.post("/upload")
async def upload_and_process(file: UploadFile = File(...), processing_params: Optional[Dict] = None):
    """Upload and process an image.
    
    This endpoint:
    1. First uploads the image using the image service
    2. Then processes it for object detection
    
    Args:
        file: The image file to upload
        processing_params: Optional dictionary of processing parameters
    """
    # First upload the image
    upload_response = await image_service.upload_image(file)
    filename = upload_response["filename"]
    
    # Create request body
    request = ProcessImageRequest(
        image_path=filename,
        processing_params=processing_params
    )
    
    # Then process it
    return await process_image(request)

@router.post("/annotate/{image_name}")
async def annotate_image(image_name: str, processing_params: Optional[Dict] = None):
    """Create an annotated version of the image with detections
    
    Args:
        image_name: Name of the image to annotate
        processing_params: Optional dictionary of processing parameters
    """
    request = ProcessImageRequest(
        image_path=image_name,
        processing_params=processing_params
    )
    return await process_image(request)

@router.get("/classes")
async def get_classes():
    """Return the list of available detection classes."""
    return await detection_service.get_classes() 