import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import logging
from mcp.server.fastmcp import FastMCP
import base64
import tempfile
from backend.core.detector import ObjectDetector
from starlette.applications import Starlette
from starlette.routing import Mount, Route
from mcp.server.sse import SseServerTransport
from pathlib import Path
from typing import Dict, Optional
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
import json

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s - %(message)s',
)

# Create MCP instance
mcp = FastMCP("Image Puzzle Solver")

# Initialize detector
detector = ObjectDetector()

@mcp.tool()
def process_image(image_b64: str = None, image_path: str = None, processing_params: Optional[Dict] = None) -> Dict:
    """Process an image and return detections with annotated image.
    
    Args:
        image_b64: Base64 encoded image data (optional)
        image_path: Path to image file (optional)
        processing_params: Optional dictionary of processing parameters
        
    Returns:
        Dictionary containing processed image data, detections, and any errors
    """
    logging.info("Received process_image request")
    if not image_b64 and not image_path:
        logging.error("Missing image input. Provide either image_b64 or image_path.")
        return {"error": "Missing image input. Provide either image_b64 or image_path."}
        
    try:
        if image_b64:
            # Handle base64 image
            image_bytes = base64.b64decode(image_b64)
            with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as temp_file:
                temp_file.write(image_bytes)
                temp_path = temp_file.name
            logging.info(f"Saved uploaded image to {temp_path}")
        else:
            # Handle direct file path
            base_path = Path(__file__).parent / 'backend' / 'data' / 'images' / 'unprocessed'
            temp_path = str(base_path / image_path)
            logging.info(f"Using image at {temp_path}")

        # Process image using detector
        detections = detector.process_image(temp_path)
        
        # Find the annotated image in the annotated directory
        annotated_dir = Path(__file__).parent / 'backend' / 'data' / 'images' / 'annotated'
        annotated_path = annotated_dir / ('annotated_' + Path(temp_path).name)
        if not os.path.exists(annotated_path):
            logging.error(f"Annotated image not found at {annotated_path}")
            return {"error": f"Annotated image not found at {annotated_path}"}
        
        # Read the annotated image and encode as base64
        with open(annotated_path, "rb") as f:
            img_bytes = f.read()
            img_b64 = base64.b64encode(img_bytes).decode("utf-8")
        
        # Create markdown snippet for chat display
        markdown_snippet = f"![Annotated Image](data:image/jpeg;base64,{img_b64})"
        
        # Clean up temporary files only if we created one
        if image_b64:
            os.unlink(temp_path)
        
        logging.info("Image processed successfully.")
        return {
            "annotated_image_b64": img_b64,
            "detections": detections,
            "markdown_snippet": markdown_snippet,
            "error": None
        }
    except Exception as e:
        logging.exception("Exception during image processing:")
        return {"error": f"Exception: {e}"}

# Create SSE transport
sse = SseServerTransport("/messages/")

# MCP SSE handler function
async def handle_sse(request):
    async with sse.connect_sse(request.scope, request.receive, request._send) as (read_stream, write_stream):
        await mcp._mcp_server.run(read_stream, write_stream, mcp._mcp_server.create_initialization_options())

# Tool endpoint handler
async def handle_tool(request: Request):
    try:
        body = await request.json()
        image_b64 = body.get('image_b64')
        image_path = body.get('image_path')
        processing_params = body.get('processing_params', {})
        
        if not image_b64 and not image_path:
            return JSONResponse(status_code=400, content={"error": "Missing 'image_b64' or 'image_path' input"})
            
        result = process_image(image_b64, image_path, processing_params)
        return JSONResponse(content=result)
    except Exception as e:
        logging.error(f"Error handling tool request: {str(e)}")
        return JSONResponse(status_code=500, content={"error": str(e)})

# Create FastAPI app
app = FastAPI()

# Add routes
app.add_route("/sse", handle_sse)
app.add_route("/process_image", handle_tool, methods=["POST"])
app.mount("/messages", sse.handle_post_message)

if __name__ == "__main__":
    logging.info("Starting MCP server: Image Puzzle Solver")
    import uvicorn
    import signal
    import sys

    def handle_shutdown(signum, frame):
        logging.info("Received shutdown signal. Exiting immediately...")
        sys.exit(0)

    # Register signal handlers
    signal.signal(signal.SIGINT, handle_shutdown)
    signal.signal(signal.SIGTERM, handle_shutdown)

    uvicorn.run(app, host="0.0.0.0", port=8010) 