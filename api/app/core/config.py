from pydantic import BaseSettings
from pathlib import Path

class Settings(BaseSettings):
    # API Settings
    API_V1_STR: str = "/api/v1"
    PROJECT_NAME: str = "Image Puzzle Solver API"
    
    # Dataset Settings
    DATASET_DIR: Path = Path("dataset")
    IMAGES_DIR: Path = DATASET_DIR / "images" / "train"
    LABELS_DIR: Path = DATASET_DIR / "labels" / "train"
    
    # Model Settings
    MODEL_PATH: str = "yolov8n.pt"
    
    class Config:
        case_sensitive = True

settings = Settings() 