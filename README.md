# Image Puzzle Solver

A tool for detecting and labeling objects in images, with special focus on crosswalk detection. This project uses YOLO for object detection and custom image processing for crosswalk detection.

## Features

- Object detection using YOLOv8
- Custom crosswalk detection using image processing techniques
- Interactive labeling interface
- Adjustable bounding boxes
- YOLO format label output

## Requirements

- Python 3.x
- OpenCV
- Ultralytics (YOLOv8)
- NumPy

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd image_puzzle_solver
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

Run the labeling script:
```bash
python collect_training_data.py
```

The script will:
1. Process images in the dataset directory
2. Detect objects using YOLO
3. Detect crosswalks using custom image processing
4. Allow interactive adjustment of detections
5. Save annotations in YOLO format

## Project Structure

- `collect_training_data.py`: Main script for image processing and labeling
- `dataset/`: Directory containing training images and labels
  - `images/train/`: Training images
  - `labels/train/`: YOLO format labels

## License

[Add your license here] 