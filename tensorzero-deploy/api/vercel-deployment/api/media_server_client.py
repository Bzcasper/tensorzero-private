import os
import httpx
from typing import List, Dict, Any, Optional


class MediaServerClient:
    """Client for media server API calls"""

    def __init__(self):
        self.base_url = os.getenv(
            "MEDIA_SERVER_URL",
            "https://2281a5a294754c19f8c9e2df0be013fb-bobby-casper-4235.aiagentsaz.com",
        )
        self.auth_header = os.getenv("IMAGE_AUTH", "Bearer 80408040")
        self.timeout = 120.0  # Longer timeout for media operations

    async def generate_images(self, prompts: List[str]) -> List[str]:
        """
        Generate images from prompts

        Args:
            prompts: List of image prompts

        Returns:
            List of image URLs/IDs
        """
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.post(
                f"{self.base_url}/generate_images",
                json={"prompts": prompts},
                headers={"Authorization": self.auth_header},
            )
            response.raise_for_status()
            return response.json().get("image_urls", [])

    async def generate_audio(self, dialogues: List[str]) -> List[str]:
        """
        Generate audio from dialogues

        Args:
            dialogues: List of dialogue texts

        Returns:
            List of audio URLs/IDs
        """
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.post(
                f"{self.base_url}/generate_audio",
                json={"dialogues": dialogues},
                headers={"Authorization": self.auth_header},
            )
            response.raise_for_status()
            return response.json().get("audio_urls", [])

    async def assemble_video(self, images: List[str], audios: List[str], timings: List[int]) -> str:
        """
        Assemble video from images, audios, and timings

        Args:
            images: List of image URLs/IDs
            audios: List of audio URLs/IDs
            timings: List of timing durations

        Returns:
            Video URL/ID
        """
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.post(
                f"{self.base_url}/assemble_video",
                json={"images": images, "audios": audios, "timings": timings},
                headers={"Authorization": self.auth_header},
            )
            response.raise_for_status()
            return response.json().get("video_url", "")
