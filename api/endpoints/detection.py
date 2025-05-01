from fastapi import APIRouter, HTTPException, UploadFile, File
from typing import Dict, Optional
from pydantic import BaseModel
from pathlib import Path

from ..core.services import detection_service, image_service
from backend.core.models.image import Image

router = APIRouter()


@router.post("/process/{image_path}")
async def process_image(image_path: str):
    """Process an image and return detections with annotated image.
    
    Args:
        image_path: Path to image file in data directory (required)
    """
    image = Image(UNPROCESSED_DIR / image_path)
    return await detection_service.process_image(image)

@router.post("/upload")
async def upload_and_process(file: UploadFile = File(...)):
    """Upload and process an image.
    
    This endpoint:
    1. First uploads the image using the image service
    2. Then processes it for object detection
    
    Args:
        file: The image file to upload
    """
    # First upload the image
    upload_response = await image_service.upload_image(file)
    filename = upload_response["filename"]
    
    # Then process it
    image = Image(UNPROCESSED_DIR / filename)
    return await detection_service.process_image(image)

@router.get("/classes")
async def get_classes():
    """Return the list of available detection classes."""
    return await detection_service.get_classes() 