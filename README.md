# Image Puzzle Solver

A tool for detecting and labeling objects in images, with special focus on crosswalk detection. This project uses YOLO for object detection and custom image processing for crosswalk detection.

## Project Structure

```
image_puzzle_solver/
├── backend/               # Core detection and processing logic
│   ├── core/             # Core functionality
│   │   └── detector.py   # Object detection implementation
│   └── utils/            # Utility functions
├── api/                  # FastAPI backend API
│   ├── app/
│   │   ├── main.py      # FastAPI application
│   │   ├── models.py    # Pydantic models
│   │   └── routes/      # API endpoints
│   └── run.py           # API server runner
└── frontend/            # React frontend
    ├── public/          # Static files
    └── src/             # React components and logic
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
- `PUT /api/v1/images/{image_name}` - Update detections for a specific image

## License

[Add your license here] 