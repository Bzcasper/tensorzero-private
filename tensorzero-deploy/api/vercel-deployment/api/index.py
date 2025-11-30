from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel
import os
import httpx
from typing import Optional
import uuid
import json

app = FastAPI(
    title="DIY Video Generation API",
    description="Serverless video generation powered by TensorZero + Fast-Agent",
    version="1.0.0",
)

# CORS disabled for debugging
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],  # Configure for your domain in production
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )


# Request/Response Models
class VideoRequest(BaseModel):
    project_description: str
    content_mode: str = "kid"  # "kid" or "adult"
    target_duration: int = 180
    expected_steps: int = 5
    background_music_id: Optional[str] = None


class VideoResponse(BaseModel):
    job_id: str
    status: str
    message: str


# TensorZero Client
class TensorZeroClient:
    def __init__(self):
        self.base_url = os.getenv("TENSORZERO_URL", "http://localhost:3000")

    async def call_function(
        self, function_name: str, input_data: dict, variant_name: Optional[str] = None
    ):
        async with httpx.AsyncClient(timeout=60.0) as client:
            payload = {"function_name": function_name, "input": input_data, "stream": False}
            if variant_name:
                payload["variant_name"] = variant_name

            response = await client.post(f"{self.base_url}/inference", json=payload)
            response.raise_for_status()
            return response.json()


@app.get("/")
async def root():
    return {
        "service": "DIY Video Generation API",
        "status": "operational",
        "endpoints": {"health": "/health", "generate_video": "POST /api/generate_video"},
    }


@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "version": "1.0.0",
        "environment": os.getenv("VERCEL_ENV", "local"),
    }


@app.post("/api/generate_video")
async def generate_video(request: VideoRequest, background_tasks: BackgroundTasks):
    """
    Generate DIY video using TensorZero + Fast-Agent workflow

    **Note**: In serverless, this starts the process and returns immediately.
    For production, integrate with Redis/database for job tracking.
    """
    try:
        job_id = str(uuid.uuid4())

        # Validate request
        if not request.project_description:
            raise HTTPException(status_code=400, detail="project_description is required")

        if request.content_mode not in ["kid", "adult"]:
            raise HTTPException(status_code=400, detail="content_mode must be 'kid' or 'adult'")

        # Start generation in background
        background_tasks.add_task(process_video_generation, job_id=job_id, request=request)

        return VideoResponse(
            job_id=job_id,
            status="processing",
            message=f"Video generation started for: {request.project_description}",
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


async def process_video_generation(job_id: str, request: VideoRequest):
    """Background task for video generation"""
    try:
        tz_client = TensorZeroClient()

        # Step 1: Generate script with TensorZero
        script_result = await tz_client.call_function(
            function_name="script_generator",
            input_data={
                "project_description": request.project_description,
                "content_mode": request.content_mode,
                "target_duration_seconds": request.target_duration,
                "expected_steps": request.expected_steps,
            },
        )

        print(
            f"[{job_id}] Script generated: {script_result.get('content', {}).get('output', {}).get('title', 'Unknown')}"
        )

        # For serverless, store results in external DB/storage
        # This is a simplified version - production would use Redis/DynamoDB

    except Exception as e:
        print(f"[{job_id}] Error: {e}")


# Vercel automatically detects the 'app' variable
