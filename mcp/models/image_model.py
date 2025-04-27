from typing import Any, Dict, List
import cv2
import numpy as np
from .base_model import BaseModel

class ImageModel(BaseModel):
    def __init__(self):
        super().__init__()
        self._data = {
            'original_image': None,
            'processed_image': None,
            'detections': [],
            'error': None
        }

    def load_image(self, image_path: str) -> bool:
        try:
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError("Failed to load image")
            
            self._data['original_image'] = image
            self._data['processed_image'] = image.copy()
            self._data['error'] = None
            self.notify_observers()
            return True
        except Exception as e:
            self._data['error'] = str(e)
            self.notify_observers()
            return False

    def process_image(self, processing_params: Dict[str, Any] = None) -> bool:
        try:
            if self._data['original_image'] is None:
                raise ValueError("No image loaded")

            # Apply image processing based on parameters
            processed = self._data['original_image'].copy()
            if processing_params:
                # Add your image processing logic here
                pass

            self._data['processed_image'] = processed
            self._data['error'] = None
            self.notify_observers()
            return True
        except Exception as e:
            self._data['error'] = str(e)
            self.notify_observers()
            return False

    def set_detections(self, detections: List[Dict[str, Any]]):
        self._data['detections'] = detections
        self.notify_observers()

    def clear(self):
        self._data = {
            'original_image': None,
            'processed_image': None,
            'detections': [],
            'error': None
        }
        self.notify_observers() 