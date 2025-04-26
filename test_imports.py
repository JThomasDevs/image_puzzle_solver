import sys
from pathlib import Path

# Add the current directory to the Python path
sys.path.append(str(Path(__file__).parent))

try:
    from api.app.main import app
    from api.app.core.config import settings
    from api.app.models.image import ImageDetection, BoundingBox
    from collect_training_data import TrainingDataCollector
    
    print("All imports successful!")
    print(f"Project name: {settings.PROJECT_NAME}")
    print(f"API version: {settings.API_V1_STR}")
    
except ImportError as e:
    print(f"Import error: {e}")
    print("Python path:", sys.path) 