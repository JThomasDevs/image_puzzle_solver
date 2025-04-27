import requests
import json
from pathlib import Path
import base64

def test_endpoint():
    url = "http://localhost:8010/process_image"
    headers = {"Content-Type": "application/json"}
    data = {"image_path": "car_on_street_10.jpg"}
    
    response = requests.post(url, headers=headers, json=data)
    
    if response.status_code == 200:
        result = response.json()
        if "error" in result and result["error"]:
            print(f"Error: {result['error']}")
        else:
            # Save the annotated image as a file for reference
            if "annotated_image_b64" in result:
                image_data = result["annotated_image_b64"]
                output_path = Path("annotated_result.jpg")
                with open(output_path, "wb") as f:
                    f.write(base64.b64decode(image_data))
                print(f"Saved annotated image to {output_path}")
            # Print detections
            if "detections" in result:
                print("Detections:")
                for det in result["detections"]:
                    print(f"- {det['class_name']} (confidence: {det['confidence']:.2f})")
            # Print a truncated markdown snippet for chat display at the very end
            if "markdown_snippet" in result:
                b64 = result["annotated_image_b64"]
                preview = b64[:40] + "..." + b64[-20:]
                print("\n---\nPaste this in chat to display the image (truncated preview):\n")
                print(f"![Annotated Image](data:image/jpeg;base64,{preview})")
                print("\n---\n(Preview only. Full markdown is available in the JSON response if needed.)")
    else:
        print(f"Error: {response.status_code}")
        print(response.text)

if __name__ == "__main__":
    test_endpoint() 