import pytest
from fastapi.testclient import TestClient
from pathlib import Path
import shutil
import os
from api.main import app
from api.core.services.detection_service import UNPROCESSED_DIR, ANNOTATED_DIR

@pytest.fixture(scope="session")
def test_client():
    """Create a test client for the FastAPI application"""
    return TestClient(app)

@pytest.fixture(scope="session")
def test_data_dir():
    """Create a temporary test data directory"""
    test_dir = Path("tests/data")
    test_dir.mkdir(exist_ok=True)
    
    # Create test image directories
    (test_dir / "images" / "unprocessed").mkdir(parents=True, exist_ok=True)
    (test_dir / "images" / "annotated").mkdir(parents=True, exist_ok=True)
    
    yield test_dir
    
    # Cleanup after tests
    shutil.rmtree(test_dir)

@pytest.fixture(scope="function")
def sample_image(test_data_dir):
    """Create a sample image for testing"""
    # Create a simple test image
    import cv2
    import numpy as np
    
    # Create a 100x100 black image with a white rectangle
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    img[20:80, 20:80] = 255
    
    # Save the image
    img_path = test_data_dir / "images" / "unprocessed" / "test_image.jpg"
    cv2.imwrite(str(img_path), img)
    
    return img_path

@pytest.fixture(scope="function")
def mock_detector(mocker, test_data_dir):
    """Mock the ObjectDetector class"""
    from backend.core.detector import ObjectDetector
    
    # Create a mock detector
    mock_detector = mocker.Mock(spec=ObjectDetector)
    
    # Mock the process_image method to return sample detections and create annotated image
    def mock_process_image(image_path):
        # Create annotated image
        import cv2
        import numpy as np
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        img[20:80, 20:80] = 255
        annotated_path = ANNOTATED_DIR / f"annotated_{Path(image_path).name}"
        cv2.imwrite(str(annotated_path), img)
        
        return [
            {
                "class_id": 0,
                "class_name": "person",
                "x_center": 0.5,
                "y_center": 0.5,
                "width": 0.2,
                "height": 0.2,
                "confidence": 0.95
            }
        ]
    
    mock_detector.process_image.side_effect = mock_process_image
    
    # Add target_classes attribute with string keys to match API response
    mock_detector.target_classes = {"0": "person", "1": "car", "2": "bicycle"}
    
    # Mock save_detections to return a path
    mock_detector.save_detections.return_value = "test_annotations.txt"
    
    # Patch the detector in the detection service
    mocker.patch("api.core.services.detection_service.detector", mock_detector)
    
    return mock_detector 