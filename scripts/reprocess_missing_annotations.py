import os
from pathlib import Path
import requests

API_URL = "http://localhost:8000/api/v1/detection/annotate/"
ANNOTATED_DIR = Path('data/images/annotated')
UNPROCESSED_DIR = Path('data/images/unprocessed')

reprocessed = []

for annotated_img in ANNOTATED_DIR.iterdir():
    if annotated_img.name.startswith('annotated_') and annotated_img.suffix in ['.jpg', '.png']:
        base_name = annotated_img.name.replace('annotated_', '')
        txt_file = annotated_img.with_suffix('.txt')
        if not txt_file.exists():
            orig_path = UNPROCESSED_DIR / base_name
            if orig_path.exists():
                print(f"Reprocessing {base_name}...")
                response = requests.post(f'{API_URL}{base_name}')
                print(f"Status: {response.status_code}")
                reprocessed.append(base_name)

print(f"\nReprocessed {len(reprocessed)} images:")
for name in reprocessed:
    print(f" - {name}") 