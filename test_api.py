import requests
import base64
import cv2
import numpy as np
from pathlib import Path
import os
import random

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

def get_unannotated_image():
    """Find a random unannotated image from the unprocessed directory"""
    unprocessed_dir = Path("data/images/unprocessed")
    annotated_dir = Path("data/images/annotated")
    
    # Get all unprocessed images
    unprocessed_images = [f for f in unprocessed_dir.glob("*.jpg") if not f.name.startswith("annotated_")]
    
    # Get all annotated images
    annotated_images = {f.name.replace("annotated_", "") for f in annotated_dir.glob("annotated_*.jpg")}
    
    # Find unannotated images
    unannotated_images = [f for f in unprocessed_images if f.name not in annotated_images]
    
    if not unannotated_images:
        print("No unannotated images found!")
        return None
        
    # Choose a random unannotated image
    return random.choice(unannotated_images)

def process_image(image_path):
    """Process an image through the API server and save the annotated result"""
    try:
        # Convert image to base64
        image_b64 = load_image_as_base64(image_path)
        
        # Send request to API server
        response = requests.post(
            'http://localhost:8000/api/v1/detection/upload',
            files={'file': open(image_path, 'rb')}
        )
        
        if response.status_code != 200:
            print(f"Error: Server returned status code {response.status_code}")
            print(f"Response: {response.text}")
            return
            
        result = response.json()
        
        if result.get('error'):
            print(f"Error processing image: {result['error']}")
            return
            
        # Create output directory if it doesn't exist
        output_dir = "data/images/annotated"
        os.makedirs(output_dir, exist_ok=True)
        
        # Generate output filename
        input_filename = os.path.basename(image_path)
        output_path = os.path.join(output_dir, f"annotated_{input_filename}")
        
        # Save the annotated image
        save_base64_image(result['annotated_image_b64'], output_path)
        print(f"\nSaved annotated image to: {output_path}")
        
        # Print detections if any
        if result.get('detections'):
            print("\nDetections found:")
            for det in result['detections']:
                print(f"- {det.get('class_name', 'Unknown')} (confidence: {det.get('confidence', 0):.2f})")
                
    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    # Find a random unannotated image
    test_image = get_unannotated_image()
    if test_image is None:
        print("No unannotated images available to process.")
    else:
        print(f"Processing image: {test_image}")
        process_image(str(test_image)) 