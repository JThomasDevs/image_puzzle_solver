# Training Documentation

This directory contains scripts for training and preparing the puzzle solver model.

## Files

- `download_training_images.py`: Downloads training images using DuckDuckGo search
- `image_annotation.py`: Handles image annotation and processing

## Training Process

1. Use `download_training_images.py` to collect training images
2. Use `image_annotation.py` to annotate and process the images

## Requirements

- Python 3.x
- YOLOv8
- OpenCV
- Other dependencies listed in the main `requirements.txt`

## Notes

- The training data should be stored in the `data/images/` directory (organized into `train/`, `val/`, `test/`, and `unprocessed/` subfolders)
- The trained model will be saved in the `backend/core/` directory 