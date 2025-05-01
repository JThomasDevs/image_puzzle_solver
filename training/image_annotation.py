import os
import requests
from pathlib import Path
import shutil

API_PROCESS = "http://localhost:8000/api/v1/detection/process"
API_SAVE_ANNOT = "http://localhost:8000/api/v1/images/{}/annotations"

class ImageAnnotator:
    """Handles annotation of images using the backend API."""
    def __init__(self, annotated_dir=None):
        self.annotated_dir = Path(annotated_dir) if annotated_dir else None

    def annotate_image(self, image_path: str):
        """Annotate an image using the backend API.
        Args:
            image_path (str): Path to the image to annotate
        Returns:
            tuple: (bool, list) - Success flag and list of annotations
        """
        image_name = Path(image_path).name
        response = requests.post(f"{API_PROCESS}/{image_name}")
        if response.status_code == 200:
            print(f"Annotated {image_name} via API.")
            return True, response.json().get("detections", [])
        else:
            print(f"API annotation failed for {image_name}: {response.text}")
            return False, []

    def save_manual_annotations(self, image_path, annotations):
        """Save manual annotations using the backend API.
        Args:
            image_path (str): Path to the image being annotated
            annotations (list): List of annotation dicts
        """
        image_name = Path(image_path).name
        response = requests.put(
            API_SAVE_ANNOT.format(image_name),
            json=annotations
        )
        if response.status_code == 200:
            print("Annotations saved via API.")
        else:
            print(f"Failed to save annotations via API: {response.text}")

class ImageProcessor:
    """Manages the processing and organization of training images using the backend API."""
    def __init__(self, base_dir=None):
        if base_dir is None:
            base_dir = Path(__file__).parent.parent / 'backend' / 'data' / 'images'
        self.base_dir = Path(base_dir)
        self.unprocessed_dir = self.base_dir / 'unprocessed'
        self.annotated_dir = self.base_dir / 'annotated'
        self.train_dir = self.base_dir / 'train'
        for dir_path in [self.unprocessed_dir, self.annotated_dir, self.train_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
        self.annotator = ImageAnnotator(annotated_dir=self.annotated_dir)

    def process_images(self):
        """Process all unprocessed images with annotations using the backend API."""
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
        Note: Annotated images and .txt files are saved by the backend API.
        Args:
            image_path (str): Path to the processed image
        """
        processed_img = Path(image_path)
        annotation_file = processed_img.with_suffix('.txt')
        if annotation_file.exists():
            shutil.move(str(processed_img), str(self.train_dir / processed_img.name))
            shutil.move(str(annotation_file), str(self.train_dir / annotation_file.name))

def main():
    processor = ImageProcessor()
    processor.process_images()

if __name__ == '__main__':
    main() 