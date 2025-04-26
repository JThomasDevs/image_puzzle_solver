from duckduckgo_search import DDGS
import os
import time
import requests
from pathlib import Path

def download_images(query, num_images=50, output_dir=None):
    """Download images using DuckDuckGo"""
    if output_dir is None:
        # Use the new directory structure
        output_dir = Path(__file__).parent.parent / 'backend' / 'data' / 'images' / 'unprocessed'
    
    os.makedirs(output_dir, exist_ok=True)
    
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
                        # Save image
                        filename = f"{query.replace(' ', '_')}_{i}.jpg"
                        filepath = os.path.join(output_dir, filename)
                        
                        with open(filepath, 'wb') as f:
                            f.write(response.content)
                            
                        print(f"Downloaded {filename}")
                        time.sleep(0.5)  # Small delay between downloads
                        
                except Exception as e:
                    print(f"Error downloading image {i}: {str(e)}")
                    continue
                    
        except Exception as e:
            print(f"Error searching for {query}: {str(e)}")

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