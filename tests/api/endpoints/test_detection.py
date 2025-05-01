import pytest
import base64
from pathlib import Path
import io

def test_upload_and_process(test_client, test_data_dir, sample_image):
    """Test the upload endpoint that processes an image.
    
    This test:
    1. Takes a sample image
    2. Uploads it via the /upload endpoint
    3. Verifies the response contains detections and annotated image path
    """
    with open(sample_image, "rb") as f:
        file_content = f.read()
    test_file = ("test_image.jpg", io.BytesIO(file_content), "image/jpeg")
    
    response = test_client.post(
        "/api/v1/detection/upload",
        files={"file": test_file}
    )
    
    assert response.status_code == 200
    assert "detections" in response.json()
    assert "annotated_path" in response.json()

def test_process_existing_image(test_client, test_data_dir, sample_image):
    """Test processing an existing image via the /process endpoint.
    
    This test:
    1. First uploads an image via the images service
    2. Then processes it via the /process endpoint
    3. Verifies the response contains detections and annotated image path
    """
    # First upload via images service
    with open(sample_image, "rb") as f:
        file_content = f.read()
    test_file = ("test_image.jpg", io.BytesIO(file_content), "image/jpeg")
    upload_response = test_client.post(
        "/api/v1/images/upload",
        files={"file": test_file}
    )
    assert upload_response.status_code == 200
    
    # Now process the uploaded image using path parameter
    response = test_client.post(
        "/api/v1/detection/process/test_image.jpg"
    )
    
    assert response.status_code == 200
    assert "detections" in response.json()
    assert "annotated_path" in response.json()

def test_process_nonexistent_image(test_client):
    """Test processing a non-existent image.
    
    Should return a 400 error with appropriate message.
    """
    response = test_client.post(
        "/api/v1/detection/process/nonexistent.jpg"
    )
    
    assert response.status_code == 400
    assert "Could not load image" in response.json()["detail"]

def test_get_classes(test_client):
    """Test getting available detection classes."""
    response = test_client.get("/api/v1/detection/classes")
    
    assert response.status_code == 200
    assert isinstance(response.json()["classes"], dict) 