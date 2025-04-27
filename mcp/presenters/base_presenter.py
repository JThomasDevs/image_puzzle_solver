from typing import Any, Dict
from abc import ABC, abstractmethod

class BasePresenter(ABC):
    def __init__(self):
        self._view_data: Dict[str, Any] = {}

    @abstractmethod
    def model_updated(self, model):
        """Called when the model is updated"""
        pass

    def get_view_data(self) -> Dict[str, Any]:
        """Get the current view data"""
        return self._view_data.copy()

    def set_view_data(self, data: Dict[str, Any]):
        """Update the view data"""
        self._view_data = data.copy()

    @abstractmethod
    def format_data_for_view(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Format the model data for view consumption"""
        pass 