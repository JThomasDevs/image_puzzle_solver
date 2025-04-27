from modelcontextprotocol.server import MCPServer, Tool, ToolContext, ToolRequest, ToolResponse
from mcp.controllers.image_controller import ImageController
import tempfile
import base64
import os

class ProcessImageTool(Tool):
    name = "process_image"
    description = "Process an uploaded image and return processed results."

    def run(self, ctx: ToolContext, req: ToolRequest) -> ToolResponse:
        # Expecting a base64-encoded image in the request
        image_b64 = req.inputs.get("image_b64")
        processing_params = req.inputs.get("processing_params", {})
        if not image_b64:
            return ToolResponse.error("Missing 'image_b64' input.")

        # Decode and save to a temporary file
        try:
            image_bytes = base64.b64decode(image_b64)
            with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as temp_file:
                temp_file.write(image_bytes)
                temp_path = temp_file.name

            controller = ImageController()
            if not controller.load_image(temp_path):
                os.unlink(temp_path)
                return ToolResponse.error("Failed to load image.")
            if not controller.process_image(processing_params):
                os.unlink(temp_path)
                return ToolResponse.error("Failed to process image.")
            result = controller.get_formatted_data()
            os.unlink(temp_path)
            return ToolResponse.success({
                "processed_image_data": result.get("processed_image_data"),
                "detections": result.get("detections"),
                "error": result.get("error")
            })
        except Exception as e:
            return ToolResponse.error(f"Exception: {e}")

if __name__ == "__main__":
    server = MCPServer(
        tools=[ProcessImageTool()],
        # Optionally, add resources or more tools here
    )
    server.run() 