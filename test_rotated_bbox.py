import cv2
import numpy as np
from pathlib import Path
import base64
from api.endpoints.images import Point, BoundingBox, Annotation
from api.core.services.detection_service import process_image

# Path to the test image
image_path = "data/images/unprocessed/traffic_cone_construction_0.jpg"

# Create a test annotation with a rotated bounding box
test_annotation = Annotation(
    class_id=0,  # Assuming 0 is the class ID for traffic cones
    confidence=0.95,
    bbox=BoundingBox(
        x_center=0.5,
        y_center=0.5,
        width=0.3,
        height=0.4,
        rotation_angle=45  # 45 degree rotation
    )
)

# Process the image
result = process_image(image_path=image_path)

# Print the result
print("Processing complete. Check the annotated image in data/images/annotated/") 