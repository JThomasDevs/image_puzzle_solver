from typing import Any, Dict, List
from ..models.image_model import ImageModel
from ..presenters.image_presenter import ImagePresenter
from .base_controller import BaseController

class ImageController(BaseController):
    def __init__(self):
        model = ImageModel()
        presenter = ImagePresenter()
        super().__init__(model, presenter)

    def load_image(self, image_path: str) -> bool:
        """Load an image from the given path"""
        return self.model.load_image(image_path)

    def process_image(self, processing_params: Dict[str, Any] = None) -> bool:
        """Process the currently loaded image"""
        return self.model.process_image(processing_params)

    def set_detections(self, detections: List[Dict[str, Any]]):
        """Set detection results for the current image"""
        self.model.set_detections(detections)

    def clear(self):
        """Clear the current image and associated data"""
        self.model.clear()

    def get_formatted_data(self) -> Dict[str, Any]:
        """Get the formatted data for view consumption"""
        return self.get_view_data() 