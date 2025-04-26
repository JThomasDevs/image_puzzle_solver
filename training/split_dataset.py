import os
import random
import shutil
from pathlib import Path

def split_dataset(train_dir='dataset/images/train', val_dir='dataset/images/val', val_split=0.2):
    """Split the dataset into training and validation sets"""
    # Create validation directory if it doesn't exist
    os.makedirs(val_dir, exist_ok=True)
    
    # Get all images
    images = list(Path(train_dir).glob('*.jpg'))
    
    # Calculate number of validation images
    num_val = int(len(images) * val_split)
    
    # Randomly select validation images
    val_images = random.sample(images, num_val)
    
    # Move validation images
    for img_path in val_images:
        dest_path = Path(val_dir) / img_path.name
        shutil.move(str(img_path), str(dest_path))
        print(f"Moved {img_path.name} to validation set")
        
    print(f"\nSplit complete:")
    print(f"Training images: {len(images) - num_val}")
    print(f"Validation images: {num_val}")

if __name__ == '__main__':
    split_dataset() 