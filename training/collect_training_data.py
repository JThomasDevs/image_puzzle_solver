import cv2
import os
import json
from pathlib import Path
import numpy as np
from ultralytics import YOLO
import shutil

class TrainingDataCollector:
    def __init__(self):
        """Initialize the training data collector with YOLO model and directory setup.
        
        The collector uses YOLOv8 for initial object detection and provides tools for:
        - Automatic object detection
        - Manual bounding box annotation
        - Crosswalk detection using computer vision
        - Interactive adjustment of detections
        
        Directory structure:
        - unprocessed/: Raw downloaded images
        - annotated/: Images with visualized bounding boxes
        - train/: Processed images with their annotations
        """
        self.model = YOLO('yolov8n.pt')  # Use base model for initial labeling
        self.current_image = None
        self.annotations = []
        
        # Define base directories
        self.base_dir = Path(__file__).parent.parent / 'backend' / 'data' / 'images'
        self.unprocessed_dir = self.base_dir / 'unprocessed'
        self.annotated_dir = self.base_dir / 'annotated'
        self.train_dir = self.base_dir / 'train'
        
        # Create directories if they don't exist
        self.annotated_dir.mkdir(parents=True, exist_ok=True)
        self.train_dir.mkdir(parents=True, exist_ok=True)
        
        # Define the classes we're interested in
        self.target_classes = {
            'person': 0,
            'bicycle': 1,
            'car': 2,
            'motorcycle': 3,
            'bus': 5,
            'truck': 7,
            'traffic light': 9,
            'fire hydrant': 10,
            'stop sign': 11,
            'crosswalk': 12  # Added crosswalk as a new class
        }
        
    def detect_crosswalk(self, image):
        """Detect crosswalk using image processing techniques"""
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Apply edge detection
        edges = cv2.Canny(gray, 50, 150)
        
        # Find lines using Hough transform
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=50, minLineLength=50, maxLineGap=10)
        
        if lines is not None:
            # Draw lines on a copy of the image
            line_image = image.copy()
            for line in lines:
                x1, y1, x2, y2 = line[0]
                cv2.line(line_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Find parallel lines that could indicate a crosswalk
            parallel_lines = []
            for i in range(len(lines)):
                for j in range(i+1, len(lines)):
                    line1 = lines[i][0]
                    line2 = lines[j][0]
                    # Check if lines are roughly parallel
                    angle1 = np.arctan2(line1[3] - line1[1], line1[2] - line1[0])
                    angle2 = np.arctan2(line2[3] - line2[1], line2[2] - line2[0])
                    if abs(angle1 - angle2) < 0.1:  # Lines are parallel
                        parallel_lines.append((line1, line2))
            
            if parallel_lines:
                # Find the bounding box of parallel lines
                all_points = []
                for line1, line2 in parallel_lines:
                    all_points.extend([(line1[0], line1[1]), (line1[2], line1[3]),
                                     (line2[0], line2[1]), (line2[2], line2[3])])
                
                if all_points:
                    points = np.array(all_points)
                    x_min, y_min = points.min(axis=0)
                    x_max, y_max = points.max(axis=0)
                    
                    # Convert to YOLO format (normalized center x, center y, width, height)
                    height, width = image.shape[:2]
                    x_center = (x_min + x_max) / 2 / width
                    y_center = (y_min + y_max) / 2 / height
                    box_width = (x_max - x_min) / width
                    box_height = (y_max - y_min) / height
                    
                    return [(self.target_classes['crosswalk'], x_center, y_center, box_width, box_height)]
        
        return []
        
    def process_image(self, image_path: str):
        """Process and annotate an image with object detections.
        
        This is the main annotation workflow that:
        1. Loads the image and runs YOLO detection
        2. Detects crosswalks using computer vision
        3. Displays current detections and options
        4. Allows interactive editing of annotations
        5. Saves the annotated image and detection labels
        
        The function provides an interactive interface for:
        - Viewing current detections
        - Adding new bounding boxes
        - Removing incorrect detections
        - Adjusting crosswalk boxes
        - Saving or skipping images
        
        Args:
            image_path (str): Path to the image file to process
            
        Returns:
            bool: True if the image was processed and saved, False if skipped
            
        Raises:
            ValueError: If the image cannot be loaded
        """
        self.current_image = cv2.imread(image_path)
        if self.current_image is None:
            raise ValueError(f"Could not load image at {image_path}")
            
        # Run initial detection
        results = self.model(self.current_image)
        
        # Detect crosswalks
        crosswalk_detections = self.detect_crosswalk(self.current_image)
        
        while True:
            # Save annotated image
            annotated_image = results[0].plot()
            
            # Draw crosswalk detections
            for class_id, x_center, y_center, width, height in crosswalk_detections:
                height_img, width_img = self.current_image.shape[:2]
                
                # Calculate box coordinates
                x1 = max(0, int((x_center - width/2) * width_img))
                y1 = max(0, int((y_center - height/2) * height_img))
                x2 = min(width_img, int((x_center + width/2) * width_img))
                y2 = min(height_img, int((y_center + height/2) * height_img))
                
                # Draw rectangle
                cv2.rectangle(annotated_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                # Calculate label position (ensure it's within image bounds)
                label_y = max(20, y1 - 10)  # Keep label at least 20px from top
                cv2.putText(annotated_image, 'crosswalk', (x1, label_y), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            
            annotated_path = str(Path(image_path).with_name('annotated_' + Path(image_path).name))
            cv2.imwrite(annotated_path, annotated_image)
            print(f"\nSaved annotated image to: {annotated_path}")
            
            # Get user input for corrections
            print("\nCurrent detections:")
            for i, box in enumerate(results[0].boxes):
                class_id = int(box.cls)
                class_name = self.model.names[class_id]
                confidence = float(box.conf)
                print(f"{i}: {class_name} (confidence: {confidence:.2f})")
                
            if crosswalk_detections:
                print("\nCrosswalk bounding box:")
                for i, (class_id, x_center, y_center, width, height) in enumerate(crosswalk_detections):
                    print(f"Box {i}: Center ({x_center:.2f}, {y_center:.2f}), Size ({width:.2f}, {height:.2f})")
            
            print("\nOptions:")
            print("1. Keep all detections")
            print("2. Add new bounding box")
            print("3. Remove detection")
            print("4. Adjust crosswalk box")
            print("5. Skip image")
            print("6. Save and move to next")
            
            choice = input("Enter your choice (1-6): ")
            
            if choice == '1':
                self._save_detections(results[0], image_path, crosswalk_detections)
                break
            elif choice == '2':
                self._add_manual_box(image_path)
            elif choice == '3':
                self._remove_detection(results[0], image_path)
            elif choice == '4':
                crosswalk_detections = self._adjust_crosswalk_box(crosswalk_detections)
            elif choice == '5':
                return
            elif choice == '6':
                self._save_detections(results[0], image_path, crosswalk_detections)
                return True
                
        return False
        
    def _save_detections(self, result, image_path, crosswalk_detections=None):
        """Save object detections in YOLO format.
        
        Converts bounding box coordinates to YOLO format:
        - Center x, center y (normalized 0-1)
        - Width, height (normalized 0-1)
        
        Creates a text file with one line per detection:
        <class_id> <x_center> <y_center> <width> <height>
        
        Args:
            result: YOLO detection result object
            image_path (str): Path to the original image
            crosswalk_detections (list, optional): List of crosswalk detections in format
                [(class_id, x_center, y_center, width, height), ...]
        """
        # Convert to YOLO format
        height, width = self.current_image.shape[:2]
        labels = []
        
        # Save YOLO detections
        for box in result.boxes:
            class_id = int(box.cls)
            class_name = self.model.names[class_id]
            
            # Only save detections for our target classes
            if class_name in self.target_classes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                
                # Convert to YOLO format (normalized center x, center y, width, height)
                x_center = (x1 + x2) / 2 / width
                y_center = (y1 + y2) / 2 / height
                box_width = (x2 - x1) / width
                box_height = (y2 - y1) / height
                
                labels.append(f"{self.target_classes[class_name]} {x_center} {y_center} {box_width} {box_height}")
        
        # Add crosswalk detections
        if crosswalk_detections:
            for class_id, x_center, y_center, box_width, box_height in crosswalk_detections:
                labels.append(f"{class_id} {x_center} {y_center} {box_width} {box_height}")
            
        # Save labels
        label_path = str(Path(image_path).with_suffix('.txt'))
        with open(label_path, 'w') as f:
            f.write('\n'.join(labels))
            
        print(f"\nSaved {len(labels)} detections to {label_path}")
        if crosswalk_detections:
            print(f"Including {len(crosswalk_detections)} crosswalk detections")
        
    def _add_manual_box(self, image_path):
        """Add a manual bounding box annotation.
        
        Provides an interactive interface to:
        1. Select object class from available classes
        2. Input normalized coordinates (0-1 range):
           - Center X, Y coordinates
           - Box width and height
        
        The coordinates should be normalized to image dimensions:
        - X coordinates: divided by image width
        - Y coordinates: divided by image height
        
        Args:
            image_path (str): Path to the image being annotated
        """
        print("\nSelect object class:")
        for name, class_id in self.target_classes.items():
            print(f"{class_id}: {name}")
        class_id = int(input("Enter class number: "))
        
        print("\nEnter bounding box coordinates (normalized 0-1):")
        x_center = float(input("Center X (0-1): "))
        y_center = float(input("Center Y (0-1): "))
        width = float(input("Width (0-1): "))
        height = float(input("Height (0-1): "))
        
        # Save new label
        label_path = str(Path(image_path).with_suffix('.txt'))
        with open(label_path, 'a') as f:
            f.write(f"\n{class_id} {x_center} {y_center} {width} {height}")
                
    def _remove_detection(self, result, image_path):
        """Remove an incorrect detection.
        
        Provides an interface to:
        1. List all current detections with class names
        2. Select a detection to remove
        3. Update and save the remaining detections
        
        Args:
            result: YOLO detection result object
            image_path (str): Path to the image being annotated
        """
        print("\nSelect detection to remove:")
        for i, box in enumerate(result.boxes):
            class_id = int(box.cls)
            class_name = self.model.names[class_id]
            print(f"{i}: {class_name}")
            
        choice = int(input("Enter detection number to remove: "))
        
        # Remove the selected detection
        boxes = result.boxes.tolist()
        del boxes[choice]
        
        # Save updated labels
        self._save_detections(result, image_path)

    def _adjust_crosswalk_box(self, crosswalk_detections):
        """Adjust the dimensions of a crosswalk bounding box.
        
        Provides interactive adjustment of:
        1. Box center position (x, y)
        2. Box width
        3. Box height
        
        All coordinates are normalized (0-1 range) relative to image dimensions.
        Ensures adjusted values stay within valid bounds.
        
        Args:
            crosswalk_detections (list): List of crosswalk detections in format
                [(class_id, x_center, y_center, width, height), ...]
                
        Returns:
            list: Updated crosswalk detections with adjusted coordinates
        """
        if not crosswalk_detections:
            print("No crosswalk detection to adjust")
            return crosswalk_detections
            
        print("\nAdjust crosswalk box:")
        print("1. Move center")
        print("2. Adjust width")
        print("3. Adjust height")
        print("4. Cancel")
        
        choice = input("Enter your choice (1-4): ")
        
        if choice == '4':
            return crosswalk_detections
            
        # Get the first crosswalk detection
        class_id, x_center, y_center, width, height = crosswalk_detections[0]
        
        if choice == '1':
            print("\nCurrent center position:", (x_center, y_center))
            x_center = float(input("Enter new X center (0-1): "))
            y_center = float(input("Enter new Y center (0-1): "))
        elif choice == '2':
            print("\nCurrent width:", width)
            width = float(input("Enter new width (0-1): "))
        elif choice == '3':
            print("\nCurrent height:", height)
            height = float(input("Enter new height (0-1): "))
            
        # Ensure values are within bounds
        x_center = max(0, min(1, x_center))
        y_center = max(0, min(1, y_center))
        width = max(0.01, min(1, width))
        height = max(0.01, min(1, height))
        
        return [(class_id, x_center, y_center, width, height)]

def main():
    collector = TrainingDataCollector()
    
    # Process images in the unprocessed directory
    for image_file in os.listdir(collector.unprocessed_dir):
        if image_file.endswith(('.jpg', '.jpeg', '.png')) and not image_file.startswith('annotated_'):
            image_path = os.path.join(collector.unprocessed_dir, image_file)
            print(f"\nProcessing {image_file}...")
            if collector.process_image(image_path):
                # Move the processed image and its annotation to train directory
                processed_img = Path(image_path)
                annotation_file = processed_img.with_suffix('.txt')
                
                if annotation_file.exists():
                    # Move image and annotation to train directory
                    shutil.move(str(processed_img), str(collector.train_dir / processed_img.name))
                    shutil.move(str(annotation_file), str(collector.train_dir / annotation_file.name))
                    
                    # Move annotated version to annotated directory
                    annotated_img = processed_img.parent / f'annotated_{processed_img.name}'
                    if annotated_img.exists():
                        shutil.move(str(annotated_img), str(collector.annotated_dir / annotated_img.name))
                break

if __name__ == '__main__':
    main() 