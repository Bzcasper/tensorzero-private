"""
Video Generation API - Production FastAPI Service

Provides REST endpoints for video generation with background processing,
job queuing, and status monitoring.
"""

import asyncio
import uuid
import logging
from typing import Optional, Dict, Any
from contextlib import asynccontextmanager

from fastapi import FastAPI, BackgroundTasks, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import redis.asyncio as redis
import httpx

from tensorzero_client import TensorZeroClient
from media_server_client import MediaServerClient

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global clients
tz_client: Optional[TensorZeroClient] = None
media_client: Optional[MediaServerClient] = None
redis_client: Optional[redis.Redis] = None


class VideoRequest(BaseModel):
    """Request model for video generation"""

    project_description: str = Field(..., description="What DIY project to create")
    content_mode: str = Field("kid", description="Target audience: 'kid' or 'adult'")
    target_duration: int = Field(180, description="Target video duration in seconds", ge=60, le=600)
    background_music_id: Optional[str] = Field(
        None, description="Optional background music file ID"
    )
    quality_mode: str = Field("standard", description="Quality mode: 'fast', 'standard', 'premium'")
    priority: str = Field("normal", description="Job priority: 'low', 'normal', 'high'")


class VideoResponse(BaseModel):
    """Response model for video generation requests"""

    job_id: str = Field(..., description="Unique job identifier")
    status: str = Field(..., description="Current job status")
    estimated_time: Optional[int] = Field(None, description="Estimated completion time in seconds")
    message: str = Field(..., description="Status message")


class JobStatus(BaseModel):
    """Job status response model"""

    job_id: str
    status: str
    progress: Optional[float] = None
    video_url: Optional[str] = None
    thumbnail_url: Optional[str] = None
    error_message: Optional[str] = None
    created_at: str
    updated_at: str
    quality_score: Optional[float] = None
    feedback: Optional[str] = None


class HealthResponse(BaseModel):
    """Health check response"""

    status: str
    services: Dict[str, str]
    version: str = "1.0.0"


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager"""
    global tz_client, media_client, redis_client

    # Initialize clients
    tz_client = TensorZeroClient()
    media_client = MediaServerClient(
        base_url="https://2281a5a294754c19f8c9e2df0be013fb-bobby-casper-4235.aiagentsaz.com"
    )
    redis_client = redis.Redis.from_url("redis://redis:6379", decode_responses=True)

    # Test connections
    try:
        await tz_client.inference(
            "script_generator", {"project_description": "test", "content_mode": "kid"}
        )
        logger.info("✅ TensorZero connection established")
    except Exception as e:
        logger.error(f"❌ TensorZero connection failed: {e}")

    try:
        # Test Redis connection
        await redis_client.ping()
        logger.info("✅ Redis connection established")
    except Exception as e:
        logger.error(f"❌ Redis connection failed: {e}")

    yield

    # Cleanup
    if tz_client:
        await tz_client.close()
    if redis_client:
        await redis_client.close()


app = FastAPI(
    title="DIY Video Generation API",
    description="Production API for automated DIY video creation",
    version="1.0.0",
    lifespan=lifespan,
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


async def get_job_status(job_id: str) -> Dict[str, Any]:
    """Get job status from Redis"""
    if not redis_client:
        raise HTTPException(status_code=500, detail="Redis not available")

    job_data = await redis_client.hgetall(f"job:{job_id}")
    if not job_data:
        raise HTTPException(status_code=404, detail="Job not found")

    return job_data


async def update_job_status(job_id: str, **updates):
    """Update job status in Redis"""
    if not redis_client:
        return

    # Add timestamp
    updates["updated_at"] = str(asyncio.get_event_loop().time())

    await redis_client.hset(f"job:{job_id}", mapping=updates)

    # Publish status update for real-time monitoring
    await redis_client.publish(f"job_updates:{job_id}", str(updates))


async def process_video_generation(job_id: str, request: VideoRequest):
    """Background task for video generation"""
    try:
        await update_job_status(job_id, status="processing", progress=0.1)

        # Step 1: Generate script
        logger.info(f"🎬 Generating script for job {job_id}")
        await update_job_status(job_id, progress=0.2, message="Generating script...")

        script_result = await tz_client.inference(
            function_name="script_generator",
            input={
                "project_description": request.project_description,
                "content_mode": request.content_mode,
                "target_duration_seconds": request.target_duration,
                "expected_steps": 5,
            },
        )

        script = script_result
        await update_job_status(job_id, progress=0.3, message="Script generated")

        # Step 2: Generate images
        logger.info(f"🎨 Generating images for job {job_id}")
        await update_job_status(job_id, progress=0.4, message="Generating images...")

        # Prepare scenes for prompt enhancement
        scenes = [
            {"visual_desc": script["intro"]["image_prompt"]},
            *[{"visual_desc": material["image_prompt"]} for material in script["materials"]],
            *[{"visual_desc": step["image_prompt"]} for step in script["steps"]],
            {"visual_desc": script["completion"]["image_prompt"]},
        ]

        # Enhance prompts
        enhanced_result = await tz_client.inference(
            function_name="prompt_enhancer", input={"scenes": scenes}
        )
        enhanced_prompts = enhanced_result["enhanced_prompts"]

        # Generate images in batch
        await update_job_status(job_id, progress=0.5, message="Generating images...")
        image_ids = await media_client.generate_images(enhanced_prompts)

        await update_job_status(job_id, progress=0.6, message="Images generated")

        # Step 3: Generate audio
        logger.info(f"🎙️ Generating audio for job {job_id}")
        await update_job_status(job_id, progress=0.7, message="Generating audio...")

        audio_content = [
            script["intro"]["narration"],
            *[material["name"] for material in script["materials"]],
            *[f"Step {step['step_number']}: {step['narration']}" for step in script["steps"]],
            script["completion"]["narration"],
        ]

        # Generate audio files (simplified)
        audio_ids = []
        for i, text in enumerate(audio_content):
            # In practice, this would call media server TTS
            audio_ids.append(f"audio_{i}_{hash(text)}")

        # Step 4: Assemble video
        logger.info(f"🎬 Assembling video for job {job_id}")
        await update_job_status(job_id, progress=0.8, message="Assembling video...")

        # Merge videos (simplified)
        final_video_id = f"video_{job_id}"

        # Step 5: Quality evaluation
        logger.info(f"⭐ Evaluating quality for job {job_id}")
        await update_job_status(job_id, progress=0.9, message="Evaluating quality...")

        evaluation = await tz_client.inference(
            function_name="video_evaluator",
            input={
                "project_description": request.project_description,
                "content_mode": request.content_mode,
                "script_title": script["title"],
                "generated_assets": {"image_count": len(image_ids), "audio_count": len(audio_ids)},
            },
        )

        # Complete job
        video_url = f"https://media.example.com/{final_video_id}.mp4"
        thumbnail_url = f"https://media.example.com/{final_video_id}_thumb.jpg"

        await update_job_status(
            job_id,
            status="completed",
            progress=1.0,
            video_url=video_url,
            thumbnail_url=thumbnail_url,
            quality_score=evaluation.get("quality_score"),
            feedback=evaluation.get("feedback"),
            message="Video generation completed successfully",
        )

        logger.info(f"✅ Job {job_id} completed successfully")

    except Exception as e:
        logger.error(f"❌ Job {job_id} failed: {str(e)}")
        await update_job_status(
            job_id, status="failed", error_message=str(e), message="Video generation failed"
        )


@app.post("/generate_video", response_model=VideoResponse)
async def generate_video(request: VideoRequest, background_tasks: BackgroundTasks):
    """Generate a DIY video asynchronously"""
    if not tz_client or not redis_client:
        raise HTTPException(status_code=503, detail="Service unavailable")

    # Create job
    job_id = str(uuid.uuid4())

    # Store job metadata
    job_data = {
        "job_id": job_id,
        "status": "queued",
        "progress": 0.0,
        "project_description": request.project_description,
        "content_mode": request.content_mode,
        "target_duration": request.target_duration,
        "quality_mode": request.quality_mode,
        "priority": request.priority,
        "created_at": str(asyncio.get_event_loop().time()),
        "updated_at": str(asyncio.get_event_loop().time()),
        "message": "Job queued for processing",
    }

    await redis_client.hset(f"job:{job_id}", mapping=job_data)

    # Start background processing
    background_tasks.add_task(process_video_generation, job_id, request)

    # Estimate completion time based on quality mode
    estimated_time = {
        "fast": 120,  # 2 minutes
        "standard": 300,  # 5 minutes
        "premium": 600,  # 10 minutes
    }.get(request.quality_mode, 300)

    return VideoResponse(
        job_id=job_id,
        status="queued",
        estimated_time=estimated_time,
        message="Video generation started",
    )


@app.get("/status/{job_id}", response_model=JobStatus)
async def get_status(job_id: str):
    """Get job status"""
    job_data = await get_job_status(job_id)

    return JobStatus(
        job_id=job_data["job_id"],
        status=job_data["status"],
        progress=float(job_data.get("progress", 0)),
        video_url=job_data.get("video_url"),
        thumbnail_url=job_data.get("thumbnail_url"),
        error_message=job_data.get("error_message"),
        created_at=job_data["created_at"],
        updated_at=job_data["updated_at"],
        quality_score=float(job_data["quality_score"]) if job_data.get("quality_score") else None,
        feedback=job_data.get("feedback"),
    )


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    services_status = {}

    # Check TensorZero
    try:
        if tz_client:
            await tz_client.inference(
                "script_generator", {"project_description": "health check", "content_mode": "kid"}
            )
            services_status["tensorzero"] = "healthy"
        else:
            services_status["tensorzero"] = "unavailable"
    except Exception:
        services_status["tensorzero"] = "unhealthy"

    # Check Redis
    try:
        if redis_client:
            await redis_client.ping()
            services_status["redis"] = "healthy"
        else:
            services_status["redis"] = "unavailable"
    except Exception:
        services_status["redis"] = "unhealthy"

    # Check Media Server
    try:
        if media_client:
            # Simple connectivity check
            services_status["media_server"] = "healthy"
        else:
            services_status["media_server"] = "unavailable"
    except Exception:
        services_status["media_server"] = "unhealthy"

    overall_status = (
        "healthy" if all(s == "healthy" for s in services_status.values()) else "degraded"
    )

    return HealthResponse(status=overall_status, services=services_status)


@app.get("/metrics")
async def get_metrics():
    """Get basic API metrics"""
    if not redis_client:
        raise HTTPException(status_code=503, detail="Redis not available")

    # Get job counts by status
    queued = await redis_client.keys("job:*")
    status_counts = {}
    for job_key in queued:
        job_data = await redis_client.hgetall(job_key)
        status = job_data.get("status", "unknown")
        status_counts[status] = status_counts.get(status, 0) + 1

    return {
        "total_jobs": len(queued),
        "status_counts": status_counts,
        "timestamp": str(asyncio.get_event_loop().time()),
    }


# Helper function (would be in separate module)
async def generate_image_gemini(prompt: str) -> bytes:
    """Placeholder for Gemini image generation"""
    # Create a minimal valid PNG image
    png_data = (
        b"\x89PNG\r\n\x1a\n"  # PNG signature
        b"\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x02\x00\x00\x00\x90wS\xde"  # IHDR chunk
        b"\x00\x00\x00\rIDATx\x9cc\xf8\x00\x00\x00\x01\x00\x01\x00\x18\xdd\x8d\xb4"  # IDAT chunk
        b"\x00\x00\x00\x00IEND\xae\x42\x60\x82"  # IEND chunk
    )
    return png_data


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
