import requests
import base64
from PIL import Image
import io

# API endpoint
url = "http://localhost:8000/api/v1/detection/detect/car_on_street_10.jpg"

# Make the request
response = requests.post(url)
if response.status_code == 200:
    result = response.json()
    
    # Decode and save the annotated image
    annotated_image_b64 = result['annotated_image_b64']
    image_data = base64.b64decode(annotated_image_b64)
    image = Image.open(io.BytesIO(image_data))
    image.save('test_rotated_result.jpg')
    
    print("Processing complete. Check test_rotated_result.jpg for the results.")
    print("Detections:", result['detections'])
else:
    print(f"Error: {response.status_code}")
    print(response.text) 