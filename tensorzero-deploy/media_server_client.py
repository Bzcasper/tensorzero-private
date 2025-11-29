"""
Media Server Client for Fast Agent Integration

This module provides a robust async client for the Media Server API endpoints.
Handles video generation, audio processing, file storage, and job polling.
"""

import os
import httpx
import asyncio
import logging
from typing import List, Dict, Any, Optional
from io import BytesIO

logger = logging.getLogger(__name__)


class MediaServerClient:
    """Async client for Media Server API operations"""

    def __init__(self, base_url: str):
        self.base_url = base_url
        self.headers = {
            "Authorization": f"Bearer {os.getenv('IMAGE_AUTH', '')}",
            "Content-Type": "application/json",
        }
        self.client = httpx.AsyncClient(timeout=300.0)  # Long timeout for generation

    async def upload(self, file_bytes: bytes, media_type: str, filename: str = "file") -> str:
        """Upload file to Media Server storage

        Args:
            file_bytes: Raw file bytes
            media_type: "image", "video", or "audio"
            filename: Optional filename

        Returns:
            File ID for use in other operations
        """
        content_type = {"image": "image/png", "video": "video/mp4", "audio": "audio/mpeg"}.get(
            media_type, "application/octet-stream"
        )

        files = {"file": (filename, BytesIO(file_bytes), content_type)}
        data = {"media_type": media_type}

        logger.info(f"📤 Uploading {media_type} file ({len(file_bytes)} bytes)")

        response = await self.client.post(
            f"{self.base_url}/api/v1/media/storage", files=files, data=data
        )
        response.raise_for_status()

        result = response.json()
        file_id = result.get("file_id") or result.get("id")

        if not file_id:
            raise ValueError(f"Upload failed: {result}")

        logger.info(f"✅ Uploaded {media_type}: {file_id}")
        return file_id

    async def generate_captioned_video(
        self,
        background_id: str,
        text: str,
        width: int = 1080,
        height: int = 1920,
        kokoro_voice: str = "af_heart",
    ) -> str:
        """Generate video with TTS and captions

        Args:
            background_id: Image file ID for background
            text: Narration text for TTS
            width: Video width (default 1080 for portrait)
            height: Video height (default 1920 for portrait)
            kokoro_voice: TTS voice to use

        Returns:
            Job ID for polling completion
        """
        payload = {
            "background_id": background_id,
            "text": text,
            "width": str(width),
            "height": str(height),
            "kokoro_voice": kokoro_voice,
            "image_effect": "ken_burns",
            "caption_config_font_size": "120",
            "caption_config_font_color": "#ffffff",
            "caption_config_stroke_color": "#000000",
        }

        logger.info(f"🎬 Generating captioned video: '{text[:50]}...'")

        response = await self.client.post(
            f"{self.base_url}/api/v1/media/video-tools/generate/tts-captioned-video",
            data=payload,  # Form data
            headers={**self.headers, "Content-Type": "application/x-www-form-urlencoded"},
        )
        response.raise_for_status()

        result = response.json()
        job_id = result.get("id") or result.get("file_id")

        if not job_id:
            raise ValueError(f"Video generation failed: {result}")

        logger.info(f"✅ Video generation job started: {job_id}")
        return job_id

    async def poll_job(self, job_id: str, max_attempts: int = 120, delay: int = 5) -> str:
        """Poll job status until completion

        Args:
            job_id: Job/file ID to poll
            max_attempts: Maximum polling attempts
            delay: Seconds between polls

        Returns:
            Job ID when completed

        Raises:
            Exception: If job fails or times out
        """
        logger.info(f"⏳ Polling job status: {job_id}")

        for attempt in range(max_attempts):
            try:
                response = await self.client.get(
                    f"{self.base_url}/api/v1/media/storage/{job_id}/status"
                )
                response.raise_for_status()

                status_data = response.json()
                status = status_data.get("status")

                if status in ("completed", "ready"):
                    logger.info(f"✅ Job {job_id} completed successfully")
                    return job_id
                elif status == "failed":
                    error_msg = status_data.get("error", "Unknown error")
                    raise Exception(f"Job {job_id} failed: {error_msg}")

                if attempt % 6 == 0:  # Log every 30 seconds
                    logger.info(
                        f"⏳ Job {job_id} status: {status} (attempt {attempt + 1}/{max_attempts})"
                    )

                await asyncio.sleep(delay)

            except httpx.HTTPStatusError as e:
                if e.response.status_code == 404:
                    # Job might not be registered yet, continue polling
                    logger.debug(f"Job {job_id} not found, continuing to poll")
                    await asyncio.sleep(delay)
                else:
                    raise

        raise Exception(f"Job {job_id} timed out after {max_attempts * delay} seconds")

    async def merge_videos(
        self, video_ids: List[str], background_music_id: Optional[str] = None
    ) -> str:
        """Merge multiple videos into one

        Args:
            video_ids: List of video file IDs to merge
            background_music_id: Optional background music file ID

        Returns:
            Job ID for the merge operation
        """
        data = {"video_ids": ",".join(video_ids)}

        if background_music_id:
            data["background_music_id"] = background_music_id
            data["background_music_volume"] = "0.3"
            logger.info(f"🎵 Adding background music: {background_music_id}")

        logger.info(f"🔗 Merging {len(video_ids)} videos")

        response = await self.client.post(
            f"{self.base_url}/api/v1/media/video-tools/merge",
            data=data,
            headers={**self.headers, "Content-Type": "application/x-www-form-urlencoded"},
        )
        response.raise_for_status()

        result = response.json()
        job_id = result.get("id") or result.get("file_id")

        if not job_id:
            raise ValueError(f"Video merge failed: {result}")

        logger.info(f"✅ Merge job started: {job_id}")
        return job_id

    async def generate_images(self, prompts: List[str]) -> List[str]:
        """Generate images from prompts

        Args:
            prompts: List of image generation prompts

        Returns:
            List of generated image file IDs
        """
        logger.info(f"🎨 Generating {len(prompts)} images")

        image_ids = []
        for i, prompt in enumerate(prompts):
            logger.info(f"🎨 Generating image {i + 1}/{len(prompts)}: '{prompt[:50]}...'")

            # Call image generation endpoint
            payload = {"prompt": prompt, "model": "flux-dev"}
            response = await self.client.post(
                f"{self.base_url}/generate_images", json=payload, headers=self.headers
            )
            response.raise_for_status()

            result = response.json()
            image_url = result.get("image_url")

            if not image_url:
                raise ValueError(f"Image generation failed: {result}")

            # Download and re-upload to get file ID
            image_response = await self.client.get(image_url)
            image_response.raise_for_status()
            image_bytes = image_response.content

            file_id = await self.upload(image_bytes, "image", f"generated_image_{i}.png")
            image_ids.append(file_id)

        logger.info(f"✅ Generated {len(image_ids)} images")
        return image_ids

    async def generate_audio(self, dialogues: List[str]) -> List[str]:
        """Generate audio from dialogue text

        Args:
            dialogues: List of dialogue texts to convert to speech

        Returns:
            List of generated audio file IDs
        """
        logger.info(f"🎙️ Generating {len(dialogues)} audio files")

        audio_ids = []
        for i, dialogue in enumerate(dialogues):
            logger.info(f"🎙️ Generating audio {i + 1}/{len(dialogues)}: '{dialogue[:50]}...'")

            # Call TTS endpoint
            payload = {
                "text": dialogue,
                "voice": "af_heart" if "kid" in dialogue.lower() else "af_bella",
                "model": "kokoro",
            }
            response = await self.client.post(
                f"{self.base_url}/generate_audio", json=payload, headers=self.headers
            )
            response.raise_for_status()

            result = response.json()
            audio_url = result.get("audio_url")

            if not audio_url:
                raise ValueError(f"Audio generation failed: {result}")

            # Download and re-upload to get file ID
            audio_response = await self.client.get(audio_url)
            audio_response.raise_for_status()
            audio_bytes = audio_response.content

            file_id = await self.upload(audio_bytes, "audio", f"generated_audio_{i}.mp3")
            audio_ids.append(file_id)

        logger.info(f"✅ Generated {len(audio_ids)} audio files")
        return audio_ids

    async def assemble_video(self, images: List[str], audios: List[str], timings: List[int]) -> str:
        """Assemble final video from images and audio

        Args:
            images: List of image file IDs
            audios: List of audio file IDs
            timings: List of durations for each segment in seconds

        Returns:
            Final video file ID
        """
        logger.info(f"🎬 Assembling video with {len(images)} images and {len(audios)} audio tracks")

        payload = {
            "image_ids": images,
            "audio_ids": audios,
            "timings": timings,
            "output_format": "mp4",
            "resolution": "1080p",
        }

        response = await self.client.post(
            f"{self.base_url}/assemble_video", json=payload, headers=self.headers
        )
        response.raise_for_status()

        result = response.json()
        video_url = result.get("video_url")

        if not video_url:
            raise ValueError(f"Video assembly failed: {result}")

        # Download and re-upload to get file ID
        video_response = await self.client.get(video_url)
        video_response.raise_for_status()
        video_bytes = video_response.content

        file_id = await self.upload(video_bytes, "video", "final_video.mp4")

        logger.info(f"✅ Video assembled: {file_id}")
        return file_id

    async def download(self, file_id: str) -> bytes:
        """Download file from Media Server

        Args:
            file_id: File ID to download

        Returns:
            Raw file bytes
        """
        logger.info(f"💾 Downloading file: {file_id}")

        response = await self.client.get(f"{self.base_url}/api/v1/media/storage/{file_id}")
        response.raise_for_status()

        logger.info(f"✅ Downloaded {file_id} ({len(response.content)} bytes)")
        return response.content

    async def close(self):
        """Close the HTTP client"""
        await self.client.aclose()
