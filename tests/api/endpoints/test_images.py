import pytest
from fastapi import UploadFile
from pathlib import Path
import io
import cv2
import numpy as np

def test_list_images(test_client, test_data_dir):
    """Test the list_images endpoint.
    
    This test:
    1. Creates an isolated test environment with its own directory structure
    2. Creates test images in the unprocessed directory
    3. Creates an annotated image in the annotated directory
    4. Temporarily patches the DATASET_DIR to use our test environment
    5. Calls the /api/v1/images/ endpoint
    6. Verifies that:
       - The response is successful
       - All test images are listed
       - Annotated images are excluded
       - The list is sorted alphabetically
    7. Cleans up by restoring the original DATASET_DIR
    """
    # Create an isolated test directory
    test_dir = test_data_dir / "test_list_images"
    test_dir.mkdir(exist_ok=True)
    unprocessed_dir = test_dir / "images" / "unprocessed"
    annotated_dir = test_dir / "images" / "annotated"
    unprocessed_dir.mkdir(parents=True, exist_ok=True)
    annotated_dir.mkdir(parents=True, exist_ok=True)
    
    # Create some test images in the unprocessed directory
    test_images = ["test1.jpg", "test2.jpg", "test3.jpg"]
    
    # Create the test images
    for img_name in test_images:
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        cv2.imwrite(str(unprocessed_dir / img_name), img)
    
    # Create an annotated image in the annotated directory
    annotated_img = np.zeros((100, 100, 3), dtype=np.uint8)
    cv2.imwrite(str(annotated_dir / "annotated_test1.jpg"), annotated_img)
    
    # Patch the DATASET_DIR to use our test directory
    import api.core.services.image_service as image_service
    original_dir = image_service.DATASET_DIR
    image_service.DATASET_DIR = unprocessed_dir
    
    try:
        # Get the list of images
        response = test_client.get("/api/v1/images/")
        
        assert response.status_code == 200
        images = response.json()
        assert isinstance(images, list)
        
        # Verify all test images are in the response
        for img_name in test_images:
            assert img_name in images
        
        # Verify annotated image is NOT in the response
        assert "annotated_test1.jpg" not in images
        
        # Verify the list is sorted
        assert images == sorted(images)
    finally:
        # Restore the original directory
        image_service.DATASET_DIR = original_dir

def test_upload_image(test_client, test_data_dir, sample_image):
    """Test the upload_image endpoint.
    
    This test:
    1. Takes a sample image created by the sample_image fixture
    2. Creates a test file object with the image data
    3. Sends a POST request to /api/v1/images/upload with the file
    4. Verifies that:
       - The response is successful
       - The filename is returned correctly
       - The file was actually saved to the unprocessed directory
    """
    # Create a test file
    with open(sample_image, "rb") as f:
        file_content = f.read()
    
    # Create a test file object
    test_file = ("test_image.jpg", io.BytesIO(file_content), "image/jpeg")
    
    # Send the request
    response = test_client.post(
        "/api/v1/images/upload",
        files={"file": test_file}
    )
    
    assert response.status_code == 200
    assert "filename" in response.json()
    assert response.json()["filename"] == "test_image.jpg"
    
    # Verify the file was saved
    saved_path = test_data_dir / "images" / "unprocessed" / "test_image.jpg"
    assert saved_path.exists()

def test_get_image(test_client, test_data_dir, sample_image):
    """Test the get_image endpoint.
    
    This test:
    1. First uploads a test image using the upload endpoint
    2. Then calls GET /api/v1/images/{image_name} to retrieve the image details
    3. Verifies that:
       - The response is successful
       - The response includes detections information
    """
    # First upload an image
    with open(sample_image, "rb") as f:
        file_content = f.read()
    test_file = ("test_image.jpg", io.BytesIO(file_content), "image/jpeg")
    test_client.post("/api/v1/images/upload", files={"file": test_file})
    
    # Then try to get it
    response = test_client.get("/api/v1/images/test_image.jpg")
    assert response.status_code == 200
    assert "detections" in response.json()

def test_save_annotations(test_client, test_data_dir, sample_image):
    """Test the save_annotations endpoint.
    
    This test:
    1. First uploads a test image
    2. Creates test annotations in YOLO format
    3. Sends a PUT request to /api/v1/images/{image_name}/annotations
    4. Verifies that:
       - The response is successful
       - The success message is returned
       - The annotation file is created in the correct location
    """
    # Create an isolated test directory
    test_dir = test_data_dir / "test_save_annotations"
    test_dir.mkdir(exist_ok=True)
    unprocessed_dir = test_dir / "images" / "unprocessed"
    unprocessed_dir.mkdir(parents=True, exist_ok=True)
    import api.core.services.image_service as image_service
    original_dir = image_service.DATASET_DIR
    image_service.DATASET_DIR = unprocessed_dir
    try:
        with open(sample_image, "rb") as f:
            file_content = f.read()
        test_file = ("test_image.jpg", io.BytesIO(file_content), "image/jpeg")
        upload_response = test_client.post("/api/v1/images/upload", files={"file": test_file})
        assert upload_response.status_code == 200
        image_path = unprocessed_dir / "test_image.jpg"
        assert image_path.exists(), f"Image not found at {image_path}"
        annotations = [
            {
                "class_id": 0,
                "bbox": {
                    "x_center": 0.5,
                    "y_center": 0.5,
                    "width": 0.2,
                    "height": 0.2,
                    "rotation_angle": 0,
                    "polygon_points": None
                }
            }
        ]
        response = test_client.put(
            "/api/v1/images/test_image.jpg/annotations",
            json={"annotations": annotations}
        )
        assert response.status_code == 200
        assert "message" in response.json()
        assert response.json()["message"] == "Annotations saved successfully"
        annotation_path = unprocessed_dir / "test_image.txt"
        print(f"Looking for annotation file at: {annotation_path}")
        print(f"Directory contents: {list(unprocessed_dir.glob('*'))}")
        assert annotation_path.exists(), f"Annotation file not found at {annotation_path}"
    finally:
        image_service.DATASET_DIR = original_dir 