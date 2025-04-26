import torch
from ultralytics import YOLO
import json
import cv2
import numpy as np
from typing import List, Tuple, Optional

class ImagePuzzleSolver:
    def __init__(self):
        # Initialize YOLO model
        self.model = YOLO('yolov8n.pt')  # Using the nano model for faster processing
        
    def _detect_game_board(self, image: np.ndarray) -> Optional[np.ndarray]:
        """
        Detect and crop the game board area from the input image by looking for the blue instruction box
        and the grid of images below it
        
        Args:
            image: Input image as numpy array
            
        Returns:
            Cropped image containing only the game board, or None if detection fails
        """
        # Convert to HSV color space for better color detection
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # Define range for blue color (instruction box)
        lower_blue = np.array([100, 150, 50])
        upper_blue = np.array([140, 255, 255])
        
        # Create a mask for blue regions
        mask = cv2.inRange(hsv, lower_blue, upper_blue)
        
        # Find contours in the blue mask
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return None
            
        # Find the largest blue contour (instruction box)
        instruction_box = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(instruction_box)
        
        # The grid of images should be directly below the instruction box
        # Calculate the grid area (assuming standard layout)
        grid_top = y + h + 5  # Add small gap between instruction box and grid
        grid_height = int(h * 3.5)  # Increased height to capture full grid
        grid_width = w  # Grid width matches instruction box width
        
        # Calculate the grid boundaries
        min_x = x
        max_x = x + grid_width
        min_y = grid_top
        max_y = min(grid_top + grid_height, image.shape[0])
        
        # Add a small margin
        margin = 5
        min_x = max(0, min_x - margin)
        min_y = max(0, min_y - margin)
        max_x = min(image.shape[1], max_x + margin)
        max_y = min(image.shape[0], max_y + margin)
        
        # Save debug image
        debug_img = image.copy()
        cv2.rectangle(debug_img, (x, y), (x+w, y+h), (0, 255, 0), 2)  # Instruction box
        cv2.rectangle(debug_img, (min_x, min_y), (max_x, max_y), (0, 0, 255), 2)  # Grid area
        cv2.imwrite("debug_layout.png", debug_img)
        
        # Crop the image to the grid area and save it
        cropped = image[min_y:max_y, min_x:max_x]
        cv2.imwrite("cropped_board.png", cropped)
        return cropped
        
    def _split_image_into_grid(self, image: np.ndarray) -> List[np.ndarray]:
        """
        Split the input image into 9 equal sections (3x3 grid)
        
        Args:
            image: Input image as numpy array
            
        Returns:
            List of 9 image sections
        """
        height, width = image.shape[:2]
        section_height = height // 3
        section_width = width // 3
        
        sections = []
        for row in range(3):
            for col in range(3):
                y1 = row * section_height
                y2 = (row + 1) * section_height
                x1 = col * section_width
                x2 = (col + 1) * section_width
                section = image[y1:y2, x1:x2]
                sections.append(section)
        
        return sections
    
    def analyze_game_board(self, image_path: str, target_object: str) -> dict:
        """
        Analyze a game board image containing 9 pictures in a 3x3 grid
        
        Args:
            image_path: Path to the game board image
            target_object: Object to search for in the images
            
        Returns:
            dict: Results in the format:
            {
                "target": "target_object",
                "correct_positions": [1, 3, 7],  # Numbers 1-9 indicating which sections contain the target
                "section_details": [
                    {
                        "position": 1,
                        "contains_target": true,
                        "confidence": 0.95,
                        "instances": [
                            {
                                "bbox": [x1, y1, x2, y2],
                                "confidence": 0.95
                            }
                        ]
                    },
                    ...
                ]
            }
        """
        # Load the game board image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image at {image_path}")
        
        # Detect and crop the game board
        cropped_image = self._detect_game_board(image)
        if cropped_image is None:
            raise ValueError("Could not detect game board in the image")
        
        # Split into 9 sections
        sections = self._split_image_into_grid(cropped_image)
        
        results = {
            "target": target_object,
            "correct_positions": [],
            "section_details": []
        }
        
        # Analyze each section
        for i, section in enumerate(sections, 1):
            # Run inference on the section
            detections = self.model(section)
            
            section_result = {
                "position": i,
                "contains_target": False,
                "confidence": 0.0,
                "instances": []
            }
            
            # Check detections
            for detection in detections[0].boxes:
                class_id = int(detection.cls)
                class_name = self.model.names[class_id]
                
                # Check if the detected object matches our target
                if class_name.lower() == target_object.lower():
                    confidence = float(detection.conf)
                    bbox = detection.xyxy[0].tolist()
                    
                    section_result["contains_target"] = True
                    section_result["confidence"] = max(section_result["confidence"], confidence)
                    section_result["instances"].append({
                        "bbox": bbox,
                        "confidence": confidence,
                        "class": class_name
                    })
            
            # If target was found, add to correct positions
            if section_result["contains_target"]:
                results["correct_positions"].append(i)
            
            results["section_details"].append(section_result)
        
        return results

def main():
    # Example usage
    solver = ImagePuzzleSolver()
    
    # Example with a game board image
    image_path = "path/to/game_board.jpg"
    target_object = "dog"  # Example target object
    
    results = solver.analyze_game_board(image_path, target_object)
    print(json.dumps(results, indent=2))

if __name__ == "__main__":
    main() 