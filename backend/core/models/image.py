from pathlib import Path
import cv2
import numpy as np
from typing import Optional, Union

class Image:
    def __init__(self, 
                 source: Union[str, Path, np.ndarray],
                 name: Optional[str] = None):
        """Initialize an Image object
        
        Args:
            source: Can be:
                - Path to image file (str or Path)
                - numpy array containing image data
            name: Optional name for the image. Required if source is numpy array
        """
        self.name = name
        self._data: Optional[np.ndarray] = None
        self._path: Optional[Path] = None
        
        if isinstance(source, np.ndarray):
            if name is None:
                raise ValueError("name is required when source is a numpy array")
            self._data = source
        else:
            self._path = Path(source)
            self.name = name or self._path.name
            
    @property
    def data(self) -> np.ndarray:
        """Get image data as numpy array, loading from disk if necessary"""
        if self._data is None:
            if self._path is None:
                raise RuntimeError("No image data or path available")
            self._data = cv2.imread(str(self._path))
            if self._data is None:
                raise ValueError(f"Could not load image from {self._path}")
        return self._data
    
    def save(self, path: Union[str, Path]) -> None:
        """Save the image to disk"""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(path), self.data)
        
    @property
    def shape(self):
        """Get image dimensions"""
        return self.data.shape
        
    def copy(self) -> 'Image':
        """Create a copy of the image"""
        return Image(self.data.copy(), self.name)
