from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.endpoints import images
from app.core.config import settings

app = FastAPI(
    title=settings.PROJECT_NAME,
    description="API for the Image Puzzle Solver application",
    version="0.1.0"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Include routers
app.include_router(images.router, prefix=settings.API_V1_STR, tags=["images"])

@app.get("/")
async def root():
    return {"message": "Welcome to the Image Puzzle Solver API"} 