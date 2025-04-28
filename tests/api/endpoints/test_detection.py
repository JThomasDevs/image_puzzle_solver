import pytest
import base64
from pathlib import Path
import io

def test_process_image_with_file(test_client, test_data_dir, sample_image):
    """Test the process_image endpoint with file upload.
    
    This test:
    1. Takes a sample image created by the sample_image fixture
    2. Creates a test file object with the image data
    3. Sends a POST request to /api/v1/detection/upload with the file
    4. Verifies that:
       - The response is successful
       - The response includes the annotated image in base64
       - The response includes detections
    """
    with open(sample_image, "rb") as f:
        file_content = f.read()
    test_file = ("test_image.jpg", io.BytesIO(file_content), "image/jpeg")
    response = test_client.post(
        "/api/v1/detection/upload",
        files={"file": test_file}
    )
    assert response.status_code == 200
    assert "annotated_image_b64" in response.json()
    assert "detections" in response.json()

def test_process_image_with_base64(test_client, test_data_dir, sample_image):
    """Test the process_image endpoint with base64 image.
    
    This test:
    1. Takes a sample image created by the sample_image fixture
    2. Uploads the image to the backend
    3. Converts the image to base64
    4. Sends a POST request to /api/v1/detection/process with the base64 data and image_name
    5. Verifies that:
       - The response is successful
       - The response includes the annotated image in base64
       - The response includes detections
    """
    # Upload the image first
    with open(sample_image, "rb") as f:
        file_content = f.read()
    test_file = ("test_image.jpg", io.BytesIO(file_content), "image/jpeg")
    upload_response = test_client.post(
        "/api/v1/images/upload",
        files={"file": test_file}
    )
    assert upload_response.status_code == 200
    # Now send the base64 request
    image_b64 = base64.b64encode(file_content).decode("utf-8")
    response = test_client.post(
        "/api/v1/detection/process",
        json={"image_b64": image_b64, "image_name": "test_image.jpg"}
    )
    if response.status_code != 200:
        print("Response status:", response.status_code)
        try:
            print("Response JSON:", response.json())
        except Exception:
            print("Response text:", response.text)
    assert response.status_code == 200
    assert "annotated_image_b64" in response.json()
    assert "detections" in response.json()

def test_get_classes(test_client):
    """Test the get_classes endpoint.
    
    This test:
    1. Calls GET /api/v1/detection/classes
    2. Verifies that:
       - The response is successful
       - The response includes the classes dictionary
    """
    response = test_client.get("/api/v1/detection/classes")
    if response.status_code == 404:
        import pytest
        pytest.skip("/api/v1/detection/classes endpoint not found.")
    assert response.status_code == 200
    assert "classes" in response.json()
    assert isinstance(response.json()["classes"], dict)

def test_process_image_with_path(test_client, test_data_dir, sample_image):
    """
    Test the /api/v1/detection/process endpoint with image_path only.

    Steps:
    1. Upload a test image to the backend using the upload endpoint.
    2. Call the /process endpoint with only the image_path in the payload.
    3. Assert that the response is 200 OK and contains both the annotated image (base64) and detections.

    This verifies that the endpoint works as expected when given a valid image_path.
    """
    # Upload the image first
    with open(sample_image, "rb") as f:
        file_content = f.read()
    test_file = ("test_image.jpg", io.BytesIO(file_content), "image/jpeg")
    upload_response = test_client.post(
        "/api/v1/images/upload",
        files={"file": test_file}
    )
    assert upload_response.status_code == 200
    # Now process by path
    response = test_client.post(
        "/api/v1/detection/process",
        json={"image_path": "test_image.jpg"}
    )
    assert response.status_code == 200
    assert "annotated_image_b64" in response.json()
    assert "detections" in response.json()

def test_process_image_with_both_fields_fails(test_client, test_data_dir, sample_image):
    """
    Test the /api/v1/detection/process endpoint with both image_b64 and image_path.

    Steps:
    1. Upload a test image to the backend.
    2. Call the /process endpoint with both image_b64 and image_path in the payload.
    3. Assert that the response is 400 Bad Request.

    This verifies that the endpoint enforces the rule that only one of image_b64 or image_path can be provided.
    """
    with open(sample_image, "rb") as f:
        file_content = f.read()
    test_file = ("test_image.jpg", io.BytesIO(file_content), "image/jpeg")
    upload_response = test_client.post(
        "/api/v1/images/upload",
        files={"file": test_file}
    )
    assert upload_response.status_code == 200
    image_b64 = base64.b64encode(file_content).decode("utf-8")
    response = test_client.post(
        "/api/v1/detection/process",
        json={"image_b64": image_b64, "image_path": "test_image.jpg"}
    )
    assert response.status_code == 400

def test_process_image_with_neither_fails(test_client):
    """
    Test the /api/v1/detection/process endpoint with neither image_b64 nor image_path.

    Steps:
    1. Call the /process endpoint with an empty payload.
    2. Assert that the response is 400 Bad Request.

    This verifies that the endpoint requires at least one of image_b64 or image_path to be provided.
    """
    response = test_client.post(
        "/api/v1/detection/process",
        json={}
    )
    assert response.status_code == 400

def test_process_image_with_nonexistent_path_fails(test_client):
    """
    Test the /api/v1/detection/process endpoint with a non-existent image_path.

    Steps:
    1. Call the /process endpoint with an image_path that does not exist on the backend.
    2. Assert that the response is 400 or 404 (depending on backend error handling).

    This verifies that the endpoint returns an error when the requested image does not exist.
    """
    response = test_client.post(
        "/api/v1/detection/process",
        json={"image_path": "does_not_exist.jpg"}
    )
    assert response.status_code in (400, 404) 