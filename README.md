# Image Puzzle Solver

A tool for detecting and labeling objects in images, with special focus on crosswalk detection. This project uses YOLO for object detection and custom image processing for crosswalk detection.

## Project Structure

```
image_puzzle_solver/
├── api/                        # FastAPI backend API
│   ├── core/
│   │   └── services/           # Service logic (detection, image handling)
│   ├── endpoints/              # API endpoint definitions
│   ├── static/                 # Static files (if any)
│   ├── main.py                 # FastAPI app entrypoint
│   └── run.py                  # API server runner
├── backend/                    # Core detection and processing logic
│   ├── core/
│   │   └── detector.py         # Object detection implementation (YOLO, etc.)
│   └── utils/                  # Utility functions
├── data/                       # Data storage
│   └── images/
│       ├── annotated/          # Annotated images (output)
│       ├── unprocessed/        # Raw/unprocessed images (input)
│       ├── train/              # Training images and labels
│       ├── test/               # Test images
│       └── val/                # Validation images
├── frontend/                   # React frontend
│   ├── public/                 # Static frontend files
│   └── src/
│       ├── components/         # React components
│       ├── App.js, index.js    # Main frontend logic
│       └── styles.css          # Frontend styles
├── tests/                      # Test suite
│   └── api/
│       ├── endpoints/          # Endpoint tests
│       └── core/               # Core service tests
├── training/                   # Model training scripts and utilities
│   ├── image_annotation.py     # Annotation and processing for training
│   ├── train_puzzle_model.py   # Model training script
│   ├── split_dataset.py        # Dataset splitting utility
│   └── download_training_images.py # Download images for training
├── requirements.txt            # Python dependencies
├── README.md                   # Project documentation
├── .gitignore
├── pytest.ini
├── yolov8n.pt                  # YOLOv8 model weights
├── test_api.py, test_endpoint.py, test_solver.py, test_rotated_bbox.py, test_rotated_detection.py
└── venv/                       # Python virtual environment (should be in .gitignore)
```

## Features

- Object detection using YOLOv8
- Custom crosswalk detection using image processing techniques
- RESTful API for image processing
- Interactive web interface for image annotation
- YOLO format label output

## Requirements

### Backend
- Python 3.x
- OpenCV
- Ultralytics (YOLOv8)
- FastAPI
- NumPy

### Frontend
- Node.js
- React
- Material-UI

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd image_puzzle_solver
```

2. Set up the backend:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

3. Set up the frontend:
```bash
cd frontend
npm install
```

## Usage

1. Start the API server:
```bash
cd api
python run.py
```

2. Start the frontend development server:
```bash
cd frontend
npm start
```

The web interface will be available at http://localhost:3000, and the API at http://localhost:8000.

## API Endpoints

- `GET /api/v1/images` - List all available images
- `GET /api/v1/images/{image_name}` - Get detections for a specific image
- `PUT /api/v1/images/{image_name}/annotations` - Update (replace) annotations for a specific image
- `POST /api/v1/detection/upload` - Upload an image for processing
- `POST /api/v1/detection/process` - Process an image (base64 or by path)
- `POST /api/v1/detection/annotate/{image_name}` - Run detection and get the annotated image and detection data for a specific image
- `GET /api/v1/detection/classes` - Get available object classes
