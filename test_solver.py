import cv2
import numpy as np
from image_puzzle_solver import ImagePuzzleSolver

def main():
    # Initialize the solver
    solver = ImagePuzzleSolver()
    
    # Load the original image
    image_path = "image.png"
    original_image = cv2.imread(image_path)
    
    # Detect and crop the game board
    cropped_image = solver._detect_game_board(original_image)
    
    if cropped_image is not None:
        # Save the cropped image
        cv2.imwrite("cropped_board.png", cropped_image)
        print("Successfully cropped the game board to cropped_board.png")
        
        # Try different common objects that might be in the images
        target_objects = ["person", "car", "dog", "cat", "bird"]
        
        for target_object in target_objects:
            print(f"\nAnalyzing for {target_object}...")
            results = solver.analyze_game_board("cropped_board.png", target_object)
            
            if results["correct_positions"]:
                print(f"Found {target_object} in positions: {results['correct_positions']}")
                
                # Visualize the results on the cropped image
                height, width = cropped_image.shape[:2]
                section_height = height // 3
                section_width = width // 3
                
                # Draw grid lines
                for i in range(1, 3):
                    cv2.line(cropped_image, (0, i * section_height), (width, i * section_height), (0, 255, 0), 2)
                    cv2.line(cropped_image, (i * section_width, 0), (i * section_width, height), (0, 255, 0), 2)
                
                # Draw numbers and highlight correct positions
                for pos in results['correct_positions']:
                    row = (pos - 1) // 3
                    col = (pos - 1) % 3
                    x = col * section_width + section_width // 2
                    y = row * section_height + section_height // 2
                    cv2.circle(cropped_image, (x, y), 30, (0, 0, 255), -1)
                    cv2.putText(cropped_image, str(pos), (x-10, y+10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                
                # Save the visualized result
                cv2.imwrite(f"solved_board_{target_object}.png", cropped_image.copy())
                print(f"Saved visualization to solved_board_{target_object}.png")
    else:
        print("Failed to detect the game board in the image")

if __name__ == "__main__":
    main() 