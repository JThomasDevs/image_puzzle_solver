from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pathlib import Path

from .endpoints import images, detection

app = FastAPI(
    title="Image Puzzle Solver API",
    description="API layer for the Image Puzzle Solver application",
    version="0.1.0"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Define paths
STATIC_PATH = Path(__file__).parent / "static"
STATIC_PATH.mkdir(exist_ok=True)

# Mount static files directory
app.mount("/ui", StaticFiles(directory=str(STATIC_PATH)), name="ui")

# Include API routers
app.include_router(images.router, prefix="/api/v1/images", tags=["images"])
app.include_router(detection.router, prefix="/api/v1/detection", tags=["detection"]) 