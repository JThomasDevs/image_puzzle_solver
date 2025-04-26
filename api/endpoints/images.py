from fastapi import APIRouter, HTTPException, UploadFile, File
from pathlib import Path
import shutil
from typing import List, Dict

from ..core.services import image_service

router = APIRouter()

@router.get("/")
async def list_images():
    """Get list of available images"""
    return await image_service.list_images()

@router.get("/{image_name}")
async def get_image(image_name: str):
    """Get image details and annotations"""
    return await image_service.get_image(image_name)

@router.post("/upload")
async def upload_image(file: UploadFile = File(...)):
    """Upload a new image"""
    return await image_service.upload_image(file)

@router.post("/{image_name}/annotations")
async def save_annotations(image_name: str, annotations: List[Dict]):
    """Save annotations for an image"""
    return await image_service.save_annotations(image_name, annotations) 