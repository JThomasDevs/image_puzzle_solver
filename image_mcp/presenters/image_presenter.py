from typing import Any, Dict, List
import cv2
import base64
import numpy as np
from .base_presenter import BasePresenter

class ImagePresenter(BasePresenter):
    def model_updated(self, model):
        """Called when the image model is updated"""
        model_data = model.get_data()
        formatted_data = self.format_data_for_view(model_data)
        self.set_view_data(formatted_data)

    def format_data_for_view(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Format the model data for view consumption"""
        formatted = {
            'has_image': False,
            'image_data': None,
            'processed_image_data': None,
            'detections': data.get('detections', []),
            'error': data.get('error')
        }

        # Convert original image to base64 if it exists
        if data.get('original_image') is not None:
            formatted['has_image'] = True
            formatted['image_data'] = self._image_to_base64(data['original_image'])

        # Convert processed image to base64 if it exists
        if data.get('processed_image') is not None:
            formatted['processed_image_data'] = self._image_to_base64(data['processed_image'])

        return formatted

    def _image_to_base64(self, image: np.ndarray) -> str:
        """Convert an OpenCV image to base64 string"""
        success, buffer = cv2.imencode('.jpg', image)
        if not success:
            return None
        return base64.b64encode(buffer).decode('utf-8') 