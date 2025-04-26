from duckduckgo_search import DDGS
import os
import time
import requests
from pathlib import Path
import hashlib
from datetime import datetime
import cv2
import numpy as np

def get_image_hash(image_data):
    """Generate a hash of the image content"""
    try:
        # Convert image data to numpy array
        nparr = np.frombuffer(image_data, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            return None
        # Resize image to a common size to ensure consistent hashing
        img = cv2.resize(img, (32, 32))
        # Calculate hash
        return hashlib.md5(img.tobytes()).hexdigest()
    except Exception:
        return None

def is_duplicate_image(image_data, output_dir):
    """Check if an image with the same content already exists"""
    new_hash = get_image_hash(image_data)
    if new_hash is None:
        return False
    
    # Check all existing images
    for file in output_dir.glob('*.jpg'):
        try:
            existing_img = cv2.imread(str(file))
            if existing_img is None:
                continue
            existing_img = cv2.resize(existing_img, (32, 32))
            existing_hash = hashlib.md5(existing_img.tobytes()).hexdigest()
            if existing_hash == new_hash:
                return True
        except Exception:
            continue
    return False

def download_images(query, num_images=50, output_dir=None):
    """Download images using DuckDuckGo"""
    if output_dir is None:
        # Use the new directory structure
        output_dir = Path(__file__).parent.parent / 'backend' / 'data' / 'images' / 'unprocessed'
    
    output_dir = Path(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    downloaded = 0
    skipped = 0
    
    with DDGS() as ddgs:
        try:
            # Search for images
            images = list(ddgs.images(
                query,
                max_results=num_images
            ))
            
            # Download each image
            for i, image in enumerate(images):
                try:
                    # Get image data
                    response = requests.get(image['image'], timeout=5)
                    if response.status_code == 200:
                        image_data = response.content
                        
                        # Check for duplicates
                        if is_duplicate_image(image_data, output_dir):
                            print(f"Skipping duplicate image {i}")
                            skipped += 1
                            continue
                        
                        # Generate unique filename with timestamp
                        filename = f"{query.replace(' ', '_')}_{timestamp}_{i}.jpg"
                        filepath = output_dir / filename
                        
                        with open(filepath, 'wb') as f:
                            f.write(image_data)
                            
                        print(f"Downloaded {filename}")
                        downloaded += 1
                        time.sleep(0.5)  # Small delay between downloads
                        
                except Exception as e:
                    print(f"Error downloading image {i}: {str(e)}")
                    continue
                    
        except Exception as e:
            print(f"Error searching for {query}: {str(e)}")
            
    print(f"\nDownload summary for '{query}':")
    print(f"Successfully downloaded: {downloaded}")
    print(f"Skipped duplicates: {skipped}")

def main():
    # List of objects to download
    objects = [
        'crosswalk street',
        'traffic light intersection',
        'stop sign road',
        'bicycle on road',
        'bus on street',
        'fire hydrant sidewalk',
        'traffic cone construction',
        'motorcycle on road',
        'truck on highway',
        'car on street'
    ]
    
    # Download images for each object
    for obj in objects:
        print(f"\nDownloading images for: {obj}")
        download_images(obj, num_images=20)  # Reduced number for testing
        time.sleep(2)  # Delay between different searches

if __name__ == '__main__':
    main() 