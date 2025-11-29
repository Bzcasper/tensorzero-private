#!/usr/bin/env python3
"""
Background Worker for Video Processing

Processes video generation jobs from Redis queue.
Handles long-running tasks without blocking the API.
"""

import asyncio
import logging
import os
import signal
import sys
from typing import Dict, Any

import redis.asyncio as redis
from tensorzero_client import TensorZeroClient
from media_server_client import MediaServerClient

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class VideoWorker:
    """Background worker for video processing jobs"""

    def __init__(self):
        self.redis_client = redis.Redis.from_url(
            os.getenv("REDIS_URL", "redis://redis:6379"), decode_responses=True
        )
        self.tz_client = TensorZeroClient(
            base_url=os.getenv("TENSORZERO_BASE_URL", "http://tensorzero:3000")
        )
        self.media_client = MediaServerClient(
            base_url=os.getenv(
                "MEDIA_SERVER_URL",
                "https://2281a5a294754c19f8c9e2df0be013fb-bobby-casper-4235.aiagentsaz.com",
            )
        )
        self.running = True

    async def process_job(self, job_id: str, job_data: Dict[str, Any]):
        """Process a single video generation job"""
        try:
            logger.info(f"🎬 Processing job {job_id}")

            # Update status to processing
            await self.update_job_status(job_id, status="processing", progress=0.1)

            # Extract job parameters
            project_description = job_data["project_description"]
            content_mode = job_data.get("content_mode", "kid")
            target_duration = int(job_data.get("target_duration", 180))

            # Step 1: Generate script
            logger.info(f"📝 Generating script for: {project_description}")
            await self.update_job_status(job_id, progress=0.2, message="Generating script...")

            script_result = await self.tz_client.inference(
                function_name="script_generator",
                input={
                    "project_description": project_description,
                    "content_mode": content_mode,
                    "target_duration_seconds": target_duration,
                    "expected_steps": 5,
                },
            )

            script = script_result
            await self.update_job_status(job_id, progress=0.3, message="Script generated")

            # Step 2: Generate images
            logger.info(f"🎨 Generating images for job {job_id}")
            await self.update_job_status(job_id, progress=0.4, message="Generating images...")

            # Prepare and enhance prompts
            prompts = [
                script["intro"]["image_prompt"],
                *[material["image_prompt"] for material in script["materials"]],
                *[step["image_prompt"] for step in script["steps"]],
                script["completion"]["image_prompt"],
            ]

            enhanced_prompts = []
            for i, prompt in enumerate(prompts):
                enhanced = await self.tz_client.inference(
                    function_name="prompt_enhancer",
                    input={
                        "basic_prompt": prompt,
                        "context": "DIY tutorial scene",
                        "target_style": "high quality digital illustration, vibrant colors",
                    },
                )
                enhanced_prompts.append(enhanced)

            # Generate and upload images
            image_ids = []
            for i, enhanced_prompt in enumerate(enhanced_prompts):
                await self.update_job_status(
                    job_id,
                    progress=0.4 + (i / len(enhanced_prompts)) * 0.2,
                    message=f"Generating image {i + 1}/{len(enhanced_prompts)}...",
                )

                # Generate image (placeholder)
                image_bytes = await generate_image_gemini(enhanced_prompt)

                # Upload to media server
                file_id = await self.media_client.upload(image_bytes, media_type="image")
                image_ids.append(file_id)

            await self.update_job_status(job_id, progress=0.6, message="Images generated")

            # Step 3: Generate audio
            logger.info(f"🎙️ Generating audio for job {job_id}")
            await self.update_job_status(job_id, progress=0.7, message="Generating audio...")

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
            await self.update_job_status(job_id, progress=0.8, message="Assembling video...")

            # Merge videos (simplified)
            final_video_id = f"video_{job_id}"

            # Step 5: Quality evaluation
            logger.info(f"⭐ Evaluating quality for job {job_id}")
            await self.update_job_status(job_id, progress=0.9, message="Evaluating quality...")

            evaluation = await self.tz_client.inference(
                function_name="video_evaluator",
                input={
                    "project_description": project_description,
                    "content_mode": content_mode,
                    "script_title": script["title"],
                    "generated_assets": {
                        "image_count": len(image_ids),
                        "audio_count": len(audio_ids),
                    },
                },
            )

            # Complete job
            video_url = f"https://media.example.com/{final_video_id}.mp4"
            thumbnail_url = f"https://media.example.com/{final_video_id}_thumb.jpg"

            await self.update_job_status(
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
            await self.update_job_status(
                job_id, status="failed", error_message=str(e), message="Video generation failed"
            )

    async def update_job_status(self, job_id: str, **updates):
        """Update job status in Redis"""
        # Add timestamp
        updates["updated_at"] = str(asyncio.get_event_loop().time())

        await self.redis_client.hset(f"job:{job_id}", mapping=updates)

        # Publish status update for real-time monitoring
        await self.redis_client.publish(f"job_updates:{job_id}", str(updates))

    async def run(self):
        """Main worker loop"""
        logger.info("🚀 Starting video processing worker")

        # Set up signal handlers for graceful shutdown
        def signal_handler(signum, frame):
            logger.info("🛑 Shutdown signal received")
            self.running = False

        signal.signal(signal.SIGTERM, signal_handler)
        signal.signal(signal.SIGINT, signal_handler)

        try:
            while self.running:
                try:
                    # Wait for job from queue (simplified - in practice use proper queue)
                    # For now, just process any queued jobs
                    job_keys = await self.redis_client.keys("job:*")

                    for job_key in job_keys:
                        job_data = await self.redis_client.hgetall(job_key)

                        if job_data.get("status") == "queued":
                            job_id = job_data["job_id"]
                            await self.process_job(job_id, job_data)
                            break  # Process one job at a time

                    # Wait before checking again
                    await asyncio.sleep(5)

                except Exception as e:
                    logger.error(f"Worker error: {e}")
                    await asyncio.sleep(10)  # Wait longer on error

        except Exception as e:
            logger.error(f"Worker failed: {e}")
        finally:
            await self.cleanup()

    async def cleanup(self):
        """Clean up resources"""
        logger.info("🧹 Cleaning up worker resources")
        await self.redis_client.close()
        await self.tz_client.close()


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


async def main():
    """Entry point"""
    worker = VideoWorker()
    await worker.run()


if __name__ == "__main__":
    asyncio.run(main())
