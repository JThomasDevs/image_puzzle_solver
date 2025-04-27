from typing import Any, Dict
from ..models.base_model import BaseModel
from ..presenters.base_presenter import BasePresenter

class BaseController:
    def __init__(self, model: BaseModel, presenter: BasePresenter):
        self.model = model
        self.presenter = presenter
        self.model.add_observer(self.presenter)

    def update_model(self, data: Dict[str, Any]):
        """Update the model with new data"""
        self.model.set_data(data)

    def get_view_data(self) -> Dict[str, Any]:
        """Get the current view data from the presenter"""
        return self.presenter.get_view_data()

    def cleanup(self):
        """Clean up resources and remove observers"""
        self.model.remove_observer(self.presenter) 