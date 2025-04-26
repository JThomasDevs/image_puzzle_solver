# Training Documentation

This directory contains scripts for training and preparing the puzzle solver model.

## Files

- `download_training_images.py`: Downloads training images using DuckDuckGo search
- `collect_training_data.py`: Processes and labels training data, including custom crosswalk detection
- `train_puzzle_model.py`: Trains the YOLO model on the collected data
- `split_dataset.py`: Utility for splitting the dataset into train/val/test sets

## Training Process

1. Use `download_training_images.py` to collect training images
2. Use `collect_training_data.py` to label the images
3. Use `split_dataset.py` to split the data into train/val/test sets
4. Use `train_puzzle_model.py` to train the model

## Requirements

- Python 3.x
- YOLO
- OpenCV
- Other dependencies listed in the main `requirements.txt`

## Notes

- The training data should be stored in the `dataset/` directory
- The trained model will be saved in the `backend/` directory 