from ultralytics import YOLO
import os
import yaml
from pathlib import Path

def create_dataset_yaml():
    """Create the dataset configuration YAML file for YOLO training"""
    data = {
        'path': 'dataset',  # dataset root dir
        'train': 'images/train',  # train images
        'val': 'images/val',  # val images
        'test': 'images/test',  # test images
        'names': {
            0: 'crosswalk',
            1: 'traffic_light',
            2: 'stop_sign',
            3: 'bicycle',
            4: 'bus',
            5: 'fire_hydrant',
            6: 'traffic_cone',
            7: 'motorcycle',
            8: 'truck',
            9: 'car'
        }
    }
    
    with open('dataset.yaml', 'w') as f:
        yaml.dump(data, f, default_flow_style=False)

def train_model():
    """Train a custom YOLO model for puzzle object detection"""
    # Create dataset configuration
    create_dataset_yaml()
    
    # Initialize a new YOLO model
    model = YOLO('yolov8n.pt')  # Start with the nano model
    
    # Train the model
    results = model.train(
        data='dataset.yaml',
        epochs=100,
        imgsz=640,
        batch=16,
        patience=50,
        device='0' if torch.cuda.is_available() else 'cpu',
        project='puzzle_model',
        name='train'
    )
    
    return results

if __name__ == '__main__':
    # Create necessary directories
    os.makedirs('dataset/images/train', exist_ok=True)
    os.makedirs('dataset/images/val', exist_ok=True)
    os.makedirs('dataset/images/test', exist_ok=True)
    os.makedirs('dataset/labels/train', exist_ok=True)
    os.makedirs('dataset/labels/val', exist_ok=True)
    os.makedirs('dataset/labels/test', exist_ok=True)
    
    # Train the model
    results = train_model()
    
    # Save the trained model
    model = YOLO('puzzle_model/train/weights/best.pt')
    model.export(format='onnx')  # Export to ONNX format for better compatibility 