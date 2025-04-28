from fastapi import APIRouter, HTTPException, UploadFile, File
from pathlib import Path
import shutil
from typing import List, Dict, Optional, Tuple
from pydantic import BaseModel, Field, field_validator
import numpy as np

from ..core.services import image_service

router = APIRouter()

class Point(BaseModel):
    """2D point coordinates"""
    x: float = Field(..., ge=0, le=1, description="X coordinate (0-1)")
    y: float = Field(..., ge=0, le=1, description="Y coordinate (0-1)")

class BoundingBox(BaseModel):
    """Bounding box that can be either axis-aligned or rotated.
    
    For axis-aligned boxes:
    - Use x_center, y_center, width, height
    - rotation_angle should be 0
    
    For rotated boxes:
    - Use polygon_points to define the vertices
    - rotation_angle is the angle in degrees (0-360)
    """
    # Axis-aligned box parameters
    x_center: Optional[float] = Field(None, ge=0, le=1, description="X coordinate of center (0-1)")
    y_center: Optional[float] = Field(None, ge=0, le=1, description="Y coordinate of center (0-1)")
    width: Optional[float] = Field(None, ge=0, le=1, description="Width of box (0-1)")
    height: Optional[float] = Field(None, ge=0, le=1, description="Height of box (0-1)")
    
    # Rotated box parameters
    rotation_angle: float = Field(0, ge=0, le=360, description="Rotation angle in degrees (0-360)")
    polygon_points: Optional[List[Point]] = Field(None, description="List of polygon vertices for non-rectangular shapes")
    
    @field_validator('polygon_points')
    @classmethod
    def validate_polygon(cls, v, info):
        if v is not None:
            if len(v) < 3:
                raise ValueError("Polygon must have at least 3 points")
            # Check if points form a convex polygon
            points = [(p.x, p.y) for p in v]
            if not cls._is_convex(points):
                raise ValueError("Polygon points must form a convex shape")
        return v
    
    @field_validator('x_center', 'y_center', 'width', 'height')
    @classmethod
    def validate_axis_aligned(cls, v, info):
        if v is not None and info.data.get('polygon_points') is not None:
            raise ValueError("Cannot specify both axis-aligned parameters and polygon points")
        return v
    
    @staticmethod
    def _is_convex(points: List[Tuple[float, float]]) -> bool:
        """Check if a polygon is convex using cross product method"""
        if len(points) < 3:
            return False
            
        def cross(o, a, b):
            return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
            
        # Get the sign of the first non-zero cross product
        sign = 0
        n = len(points)
        for i in range(n):
            o = points[i]
            a = points[(i + 1) % n]
            b = points[(i + 2) % n]
            curr = cross(o, a, b)
            if curr != 0:
                if sign == 0:
                    sign = curr
                elif curr * sign < 0:
                    return False
        return True

class Annotation(BaseModel):
    """Single object annotation"""
    class_id: int = Field(..., ge=0, description="ID of the detected class")
    bbox: BoundingBox = Field(..., description="Bounding box coordinates")
    confidence: Optional[float] = Field(None, ge=0, le=1, description="Detection confidence score (0-1)")

class ImageResponse(BaseModel):
    """Response model for image details"""
    filename: str = Field(..., description="Name of the image file")
    path: str = Field(..., description="Full path to the image file")
    detections: List[Annotation] = Field(default_factory=list, description="List of detected objects")

class UploadResponse(BaseModel):
    """Response model for image upload"""
    message: str = Field(..., description="Success message")
    filename: str = Field(..., description="Name of the uploaded file")

class SaveAnnotationsRequest(BaseModel):
    """Request model for saving annotations"""
    annotations: List[Annotation] = Field(..., description="List of annotations to save")

class SaveAnnotationsResponse(BaseModel):
    """Response model for saving annotations"""
    message: str = Field(..., description="Success message")

@router.get("/", response_model=List[str])
async def list_images() -> List[str]:
    """Get list of available images in the unprocessed directory.
    
    Returns:
        List[str]: A sorted list of image filenames (excluding annotated images)
        
    Example:
        ["image1.jpg", "image2.jpg"]
    """
    return await image_service.list_images()

@router.get("/{image_name}", response_model=ImageResponse)
async def get_image(image_name: str) -> ImageResponse:
    """Get detailed information about a specific image including its annotations.
    
    Args:
        image_name (str): The name of the image file to retrieve
        
    Returns:
        ImageResponse: Detailed information about the image
        
    Raises:
        HTTPException: 404 if the image is not found
    """
    return await image_service.get_image(image_name)

@router.post("/upload", response_model=UploadResponse)
async def upload_image(file: UploadFile = File(...)) -> UploadResponse:
    """Upload a new image to the unprocessed directory.
    
    Args:
        file (UploadFile): The image file to upload. Must be a valid image file.
        
    Returns:
        UploadResponse: Information about the uploaded file
    """
    return await image_service.upload_image(file)

@router.post("/{image_name}/annotations", response_model=SaveAnnotationsResponse)
async def save_annotations(
    image_name: str,
    request: SaveAnnotationsRequest
) -> SaveAnnotationsResponse:
    """Save or update annotations for a specific image.
    
    Args:
        image_name (str): Name of the image to save annotations for
        request (SaveAnnotationsRequest): Request containing annotations to save
        
    Returns:
        SaveAnnotationsResponse: Confirmation of saved annotations
        
    Raises:
        HTTPException: 404 if the image is not found
    """
    return await image_service.save_annotations(image_name, request.annotations) 