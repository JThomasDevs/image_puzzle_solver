from typing import Any, Dict

class BaseModel:
    def __init__(self):
        self._data: Dict[str, Any] = {}
        self._observers = []

    def add_observer(self, observer):
        if observer not in self._observers:
            self._observers.append(observer)

    def remove_observer(self, observer):
        if observer in self._observers:
            self._observers.remove(observer)

    def notify_observers(self):
        for observer in self._observers:
            observer.model_updated(self)

    def get_data(self) -> Dict[str, Any]:
        return self._data.copy()

    def set_data(self, data: Dict[str, Any]):
        self._data = data.copy()
        self.notify_observers() 