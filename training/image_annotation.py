import cv2
import os
import json
from pathlib import Path
import numpy as np
from ultralytics import YOLO
import shutil

class ImageAnnotator:
    """Handles the annotation of images with bounding boxes.
    
    This class provides tools for:
    - Automatic object detection using YOLO
    - Manual bounding box annotation
    - Crosswalk detection using computer vision
    - Interactive adjustment of detections
    
    The annotation process supports both automatic detection and manual refinement,
    with all coordinates normalized to the YOLO format (0-1 range).
    """
    
    def __init__(self, model_path='yolov8n.pt', annotated_dir=None):
        """Initialize the annotator with a YOLO model.
        
        Args:
            model_path (str): Path to the YOLO model weights
            annotated_dir (Path, optional): Directory to save annotated images
        """
        self.model = YOLO(model_path)
        self.current_image = None
        self.annotated_dir = Path(annotated_dir) if annotated_dir else None
        
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
            'crosswalk': 12
        }
    
    def detect_crosswalk(self, image):
        """Detect crosswalk using image processing techniques.
        
        Uses edge detection and line detection to identify potential crosswalks
        in the image based on parallel line patterns.
        
        Args:
            image (np.ndarray): Input image in BGR format
            
        Returns:
            list: Detected crosswalks in format [(class_id, x_center, y_center, width, height), ...]
        """
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
        
    def annotate_image(self, image_path: str):
        """Annotate an image with bounding boxes.
        
        Main annotation workflow:
        1. Load image and run YOLO detection
        2. Detect crosswalks
        3. Allow interactive editing of annotations
        4. Save results
        
        Args:
            image_path (str): Path to the image to annotate
            
        Returns:
            tuple: (bool, list) - Success flag and list of annotations
        """
        try:
            self.current_image = cv2.imread(image_path)
            if self.current_image is None:
                raise ValueError(f"Could not load image at {image_path}")
                
            results = self.model(self.current_image)
            crosswalk_detections = self.detect_crosswalk(self.current_image)
            
            while True:
                annotated_image = self._create_annotated_image(results[0], crosswalk_detections)
                
                # Save annotated image to the correct directory
                if self.annotated_dir:
                    annotated_path = self.annotated_dir / f'annotated_{Path(image_path).name}'
                else:
                    annotated_path = Path(image_path).with_name('annotated_' + Path(image_path).name)
                    
                cv2.imwrite(str(annotated_path), annotated_image)
                print(f"\nSaved annotated image to: {annotated_path}")
                
                # Display current detections
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
                
                # Get user choice
                print("\nOptions:")
                print("1. Keep all detections")
                print("2. Add new bounding box")
                print("3. Remove detection")
                print("4. Adjust crosswalk box")
                print("5. Skip image")
                print("6. Save and move to next")
                print("q. Quit annotation")
                
                try:
                    choice = input("Enter your choice (1-6 or q): ").lower()
                except KeyboardInterrupt:
                    print("\nAnnotation interrupted. Skipping image.")
                    return False, []
                
                if choice == 'q':
                    print("\nQuitting annotation process.")
                    return False, []
                elif choice == '1':
                    annotations = self._save_detections(results[0], image_path, crosswalk_detections)
                    return True, annotations
                elif choice == '2':
                    self._add_manual_box(image_path)
                elif choice == '3':
                    self._remove_detection(results[0], image_path)
                elif choice == '4':
                    crosswalk_detections = self._adjust_crosswalk_box(crosswalk_detections)
                elif choice == '5':
                    return False, []
                elif choice == '6':
                    annotations = self._save_detections(results[0], image_path, crosswalk_detections)
                    return True, annotations
                
        except KeyboardInterrupt:
            print("\nAnnotation interrupted. Skipping image.")
            return False, []
        except Exception as e:
            print(f"\nError processing {Path(image_path).name}: {str(e)}")
            return False, []
    
    def _create_annotated_image(self, result, crosswalk_detections):
        """Create a visualization of all detections on the image.
        
        Args:
            result: YOLO detection result
            crosswalk_detections: List of crosswalk detections
            
        Returns:
            np.ndarray: Image with drawn bounding boxes
        """
        annotated_image = result.plot()
        
        # Draw crosswalk detections
        for class_id, x_center, y_center, width, height in crosswalk_detections:
            height_img, width_img = self.current_image.shape[:2]
            x1 = max(0, int((x_center - width/2) * width_img))
            y1 = max(0, int((y_center - height/2) * height_img))
            x2 = min(width_img, int((x_center + width/2) * width_img))
            y2 = min(height_img, int((y_center + height/2) * height_img))
            cv2.rectangle(annotated_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label_y = max(20, y1 - 10)
            cv2.putText(annotated_image, 'crosswalk', (x1, label_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        
        return annotated_image
    
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
        
        return labels
        
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

class ImageProcessor:
    """Manages the processing and organization of training images.
    
    This class handles:
    - Directory structure management
    - Image processing workflow coordination
    - File organization and movement between directories
    - Integration with ImageAnnotator for annotation
    
    Note: This class does not handle the actual collection/downloading of images.
    For image collection, see download_training_images.py
    """
    
    def __init__(self, base_dir=None):
        """Initialize the image processor with directory structure.
        
        Args:
            base_dir (Path, optional): Base directory for image processing
        """
        if base_dir is None:
            base_dir = Path(__file__).parent.parent / 'backend' / 'data' / 'images'
        
        self.base_dir = Path(base_dir)
        self.unprocessed_dir = self.base_dir / 'unprocessed'
        self.annotated_dir = self.base_dir / 'annotated'
        self.train_dir = self.base_dir / 'train'
        
        # Create directories
        for dir_path in [self.unprocessed_dir, self.annotated_dir, self.train_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        # Initialize annotator with the correct annotated directory
        self.annotator = ImageAnnotator(annotated_dir=self.annotated_dir)
    
    def process_images(self):
        """Process all unprocessed images with annotations.
        
        Workflow:
        1. Find images in the unprocessed directory
        2. For each image:
           - Run it through annotation process
           - If successfully annotated, organize files into appropriate directories
           - Handle any errors during processing
        """
        for image_file in os.listdir(self.unprocessed_dir):
            if image_file.endswith(('.jpg', '.jpeg', '.png')) and not image_file.startswith('annotated_'):
                image_path = os.path.join(self.unprocessed_dir, image_file)
                print(f"\nProcessing {image_file}...")
                
                try:
                    success, annotations = self.annotator.annotate_image(image_path)
                    if success:
                        self._organize_files(image_path)
                except Exception as e:
                    print(f"Error processing {image_file}: {str(e)}")
    
    def _organize_files(self, image_path):
        """Organize processed files into appropriate directories.
        
        Moves:
        - Original image -> train directory
        - Annotation file -> train directory
        
        Note: Annotated images are now saved directly to the annotated directory
        by the ImageAnnotator
        
        Args:
            image_path (str): Path to the processed image
        """
        processed_img = Path(image_path)
        annotation_file = processed_img.with_suffix('.txt')
        
        if annotation_file.exists():
            # Move files to appropriate directories
            shutil.move(str(processed_img), str(self.train_dir / processed_img.name))
            shutil.move(str(annotation_file), str(self.train_dir / annotation_file.name))

def main():
    processor = ImageProcessor()
    processor.process_images()

if __name__ == '__main__':
    main() 