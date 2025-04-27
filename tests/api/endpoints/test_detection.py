import pytest
import base64
from pathlib import Path
import io

def test_process_image_with_file(test_client, test_data_dir, sample_image, mock_detector):
    """Test the process_image endpoint with file upload.
    
    This test:
    1. Takes a sample image created by the sample_image fixture
    2. Creates a test file object with the image data
    3. Sends a POST request to /api/v1/detection/upload with the file
    4. Verifies that:
       - The response is successful
       - The response includes the annotated image in base64
       - The response includes detections
       - The response includes a markdown snippet
    5. Verifies that the mock detector was called exactly once
    """
    # Create a test file
    with open(sample_image, "rb") as f:
        file_content = f.read()
    
    # Create a test file object
    test_file = ("test_image.jpg", io.BytesIO(file_content), "image/jpeg")
    
    # Send the request
    response = test_client.post(
        "/api/v1/detection/upload",
        files={"file": test_file}
    )
    
    assert response.status_code == 200
    assert "annotated_image_b64" in response.json()
    assert "detections" in response.json()
    assert "markdown_snippet" in response.json()
    
    # Verify the mock detector was called
    mock_detector.process_image.assert_called_once()

def test_process_image_with_base64(test_client, test_data_dir, sample_image, mock_detector):
    """Test the process_image endpoint with base64 image.
    
    This test:
    1. Takes a sample image created by the sample_image fixture
    2. Converts the image to base64
    3. Sends a POST request to /api/v1/detection/process with the base64 data
    4. Verifies that:
       - The response is successful
       - The response includes the annotated image in base64
       - The response includes detections
       - The response includes a markdown snippet
    5. Verifies that the mock detector was called exactly once
    """
    # Read the test image and convert to base64
    with open(sample_image, "rb") as f:
        image_bytes = f.read()
    image_b64 = base64.b64encode(image_bytes).decode("utf-8")
    
    # Send the request with query parameters
    response = test_client.post(
        "/api/v1/detection/process",
        params={
            "image_b64": image_b64,
            "processing_params": "{}"
        }
    )
    
    assert response.status_code == 200
    assert "annotated_image_b64" in response.json()
    assert "detections" in response.json()
    assert "markdown_snippet" in response.json()
    
    # Verify the mock detector was called
    mock_detector.process_image.assert_called_once()

def test_detect_objects(test_client, test_data_dir, sample_image, mock_detector):
    """Test the detect_objects endpoint.
    
    This test:
    1. First uploads a test image using the upload endpoint
    2. Then calls POST /api/v1/detection/detect/{image_name} to run detection
    3. Verifies that:
       - The response is successful
       - The response includes detections
       - The response includes the annotated image in base64
    4. Verifies that the mock detector was called exactly once
    """
    # First upload an image
    with open(sample_image, "rb") as f:
        file_content = f.read()
    test_file = ("test_image.jpg", io.BytesIO(file_content), "image/jpeg")
    test_client.post("/api/v1/images/upload", files={"file": test_file})
    
    # Then try to detect objects using the detect endpoint
    response = test_client.post("/api/v1/detection/detect/test_image.jpg")
    
    if response.status_code != 200:
        print(f"Error response: {response.json()}")
    
    assert response.status_code == 200
    assert "detections" in response.json()
    assert "annotated_image_b64" in response.json()
    
    # Verify the mock detector was called
    mock_detector.process_image.assert_called_once()

def test_get_classes(test_client, mock_detector):
    """Test the get_classes endpoint.
    
    This test:
    1. Calls GET /api/v1/detection/classes
    2. Verifies that:
       - The response is successful
       - The response includes the classes dictionary
       - The classes match those defined in the mock detector
    """
    response = test_client.get("/api/v1/detection/classes")
    
    assert response.status_code == 200
    assert "classes" in response.json()
    assert isinstance(response.json()["classes"], dict)
    
    # Verify the mock detector's target_classes were returned
    assert response.json()["classes"] == mock_detector.target_classes 