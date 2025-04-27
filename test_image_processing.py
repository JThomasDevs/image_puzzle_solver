import requests
import base64
import cv2
import numpy as np
from pathlib import Path
import tempfile
import os

def load_image_as_base64(image_path):
    """Load an image file and convert it to base64 string"""
    with open(image_path, 'rb') as f:
        image_bytes = f.read()
        return base64.b64encode(image_bytes).decode('utf-8')

def save_base64_image(base64_str, output_path):
    """Save a base64-encoded image to a file"""
    image_bytes = base64.b64decode(base64_str)
    with open(output_path, 'wb') as f:
        f.write(image_bytes)

def display_image(image_path):
    """Display an image using OpenCV"""
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Could not load image at {image_path}")
        return
        
    # Create a window and display the image
    window_name = 'Annotated Image'
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.imshow(window_name, img)
    
    # Wait for a key press and then close the window
    while True:
        key = cv2.waitKey(1) & 0xFF
        if key == 27 or cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:  # ESC key or window closed
            break
            
    cv2.destroyAllWindows()
    cv2.waitKey(1)  # Additional wait to ensure window is destroyed

def process_image(image_path):
    """Process an image through the MCP server and display the annotated result"""
    try:
        # Convert image to base64
        image_b64 = load_image_as_base64(image_path)
        
        # Send request to MCP server
        response = requests.post(
            'http://localhost:8010/process_image',
            json={
                'image_b64': image_b64,
                'processing_params': {}
            }
        )
        
        if response.status_code != 200:
            print(f"Error: Server returned status code {response.status_code}")
            print(f"Response: {response.text}")
            return
            
        result = response.json()
        
        if result.get('error'):
            print(f"Error processing image: {result['error']}")
            return
            
        # Create a temporary file for the annotated image
        with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as temp_file:
            temp_path = temp_file.name
            
        # Save the annotated image
        save_base64_image(result['processed_image_data'], temp_path)
        
        # Display the annotated image
        print("\nDisplaying annotated image...")
        print("Press any key to close the window")
        display_image(temp_path)
        
        # Clean up
        os.unlink(temp_path)
        
        # Print detections if any
        if result.get('detections'):
            print("\nDetections found:")
            for det in result['detections']:
                print(f"- {det.get('class_name', 'Unknown')} (confidence: {det.get('confidence', 0):.2f})")
                
    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    # Test with a sample image
    test_image = "image.png"  # You can change this to any image path
    if not os.path.exists(test_image):
        print(f"Error: Test image not found at {test_image}")
    else:
        print(f"Processing image: {test_image}")
        process_image(test_image) 