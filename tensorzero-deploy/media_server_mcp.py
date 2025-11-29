"""
Media Server MCP Server

This MCP server wraps the Media Server API endpoints as standardized MCP tools.
Provides progress notifications and proper error handling for video/audio processing.
"""

import asyncio
import logging
import os
from typing import Any, Dict, List, Optional
from contextlib import asynccontextmanager
import httpx
from mcp import Tool
from mcp.server import Server
from mcp.types import TextContent, ImageContent, EmbeddedResource, LoggingLevel
import mcp.server.stdio
from pydantic import BaseModel, Field

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Media Server configuration
MEDIA_SERVER_URL = "https://2281a5a294754c19f8c9e2df0be013fb-bobby-casper-4235.aiagentsaz.com"
IMAGE_AUTH = None  # Will be set from environment

# Initialize HTTP client
client = httpx.AsyncClient(timeout=300.0, follow_redirects=True)

# MCP Server setup
server = Server("media-server-mcp")


class UploadFileArgs(BaseModel):
    """Arguments for file upload tool"""

    file_data: str = Field(description="Base64 encoded file data")
    media_type: str = Field(description="Type of media: 'image', 'video', or 'audio'")
    filename: str = Field(description="Optional filename", default="file")


class GenerateCaptionedVideoArgs(BaseModel):
    """Arguments for video generation with captions"""

    background_id: str = Field(description="File ID of background image")
    text: str = Field(description="Narration text for TTS and captions")
    width: int = Field(description="Video width", default=1080)
    height: int = Field(description="Video height", default=1920)
    voice: str = Field(description="TTS voice", default="af_heart")


class MergeVideosArgs(BaseModel):
    """Arguments for merging videos"""

    video_ids: str = Field(description="Comma-separated list of video file IDs")
    background_music_id: Optional[str] = Field(
        description="Optional background music file ID", default=None
    )


class PollJobArgs(BaseModel):
    """Arguments for polling job status"""

    job_id: str = Field(description="Job/file ID to check status")
    max_attempts: int = Field(description="Maximum polling attempts", default=120)
    delay: int = Field(description="Seconds between polls", default=5)


@server.list_tools()
async def list_tools() -> List[Tool]:
    """List all available Media Server tools"""
    return [
        Tool(
            name="upload_file",
            description="Upload a file to Media Server storage",
            inputSchema=UploadFileArgs.model_json_schema(),
        ),
        Tool(
            name="generate_captioned_video",
            description="Generate video with TTS and captions from background image",
            inputSchema=GenerateCaptionedVideoArgs.model_json_schema(),
        ),
        Tool(
            name="merge_videos",
            description="Merge multiple videos into one final video",
            inputSchema=MergeVideosArgs.model_json_schema(),
        ),
        Tool(
            name="poll_job_status",
            description="Poll a job/file status until completion",
            inputSchema=PollJobArgs.model_json_schema(),
        ),
        Tool(
            name="download_file",
            description="Download a file from Media Server storage",
            inputSchema={
                "type": "object",
                "properties": {"file_id": {"type": "string", "description": "File ID to download"}},
                "required": ["file_id"],
            },
        ),
        Tool(
            name="get_video_info",
            description="Get metadata information about a video file",
            inputSchema={
                "type": "object",
                "properties": {"file_id": {"type": "string", "description": "Video file ID"}},
                "required": ["file_id"],
            },
        ),
    ]


async def make_request(method: str, endpoint: str, **kwargs) -> Dict[str, Any]:
    """Make authenticated request to Media Server"""
    headers = {"Content-Type": "application/json"}
    if IMAGE_AUTH:
        headers["Authorization"] = f"Bearer {IMAGE_AUTH}"

    url = f"{MEDIA_SERVER_URL}{endpoint}"

    try:
        response = await client.request(method, url, headers=headers, **kwargs)
        response.raise_for_status()
        return response.json()
    except httpx.HTTPStatusError as e:
        logger.error(f"HTTP {e.response.status_code}: {e.response.text}")
        raise
    except Exception as e:
        logger.error(f"Request failed: {str(e)}")
        raise


async def poll_job_status(job_id: str, max_attempts: int = 120, delay: int = 5) -> str:
    """Poll job status until completion with progress reporting"""
    logger.info(f"Polling job status: {job_id}")

    for attempt in range(max_attempts):
        try:
            status_data = await make_request("GET", f"/api/v1/media/storage/{job_id}/status")

            status = status_data.get("status")
            if status in ("completed", "ready"):
                logger.info(f"Job {job_id} completed successfully")
                return job_id
            elif status == "failed":
                error_msg = status_data.get("error", "Unknown error")
                raise Exception(f"Job {job_id} failed: {error_msg}")

            # Report progress every 30 seconds
            if attempt % 6 == 0:
                logger.info(f"Job {job_id} status: {status} (attempt {attempt + 1}/{max_attempts})")

            await asyncio.sleep(delay)

        except httpx.HTTPStatusError as e:
            if e.response.status_code == 404:
                # Job might not be registered yet, continue polling
                logger.debug(f"Job {job_id} not found, continuing to poll")
                await asyncio.sleep(delay)
            else:
                raise

    raise Exception(f"Job {job_id} timed out after {max_attempts * delay} seconds")


@server.call_tool()
async def upload_file(arguments: Dict[str, Any]) -> List[TextContent]:
    """Upload file to Media Server storage"""
    args = UploadFileArgs(**arguments)

    # Decode base64 file data
    import base64

    file_bytes = base64.b64decode(args.file_data)

    # Prepare multipart form data
    files = {"file": (args.filename, file_bytes, "application/octet-stream")}
    data = {"media_type": args.media_type}

    headers = {}
    if IMAGE_AUTH:
        headers["Authorization"] = f"Bearer {IMAGE_AUTH}"

    try:
        response = await client.post(
            f"{MEDIA_SERVER_URL}/api/v1/media/storage", files=files, data=data, headers=headers
        )
        response.raise_for_status()

        result = response.json()
        file_id = result.get("file_id") or result.get("id")

        if not file_id:
            raise ValueError(f"Upload failed: {result}")

        logger.info(f"Successfully uploaded {args.media_type} file: {file_id}")

        return [TextContent(type="text", text=f"File uploaded successfully. File ID: {file_id}")]

    except Exception as e:
        error_msg = f"File upload failed: {str(e)}"
        logger.error(error_msg)
        return [TextContent(type="text", text=error_msg)]


@server.call_tool()
async def generate_captioned_video(arguments: Dict[str, Any]) -> List[TextContent]:
    """Generate video with TTS and captions"""
    args = GenerateCaptionedVideoArgs(**arguments)

    payload = {
        "background_id": args.background_id,
        "text": args.text,
        "width": str(args.width),
        "height": str(args.height),
        "kokoro_voice": args.voice,
        "image_effect": "ken_burns",
        "caption_config_font_size": "120",
        "caption_config_font_color": "#ffffff",
        "caption_config_stroke_color": "#000000",
    }

    try:
        logger.info(f"Starting video generation with text: '{args.text[:50]}...'")

        result = await make_request(
            "POST", "/api/v1/media/video-tools/generate/tts-captioned-video", json=payload
        )
        job_id = result.get("id") or result.get("file_id")

        if not job_id:
            raise ValueError(f"Video generation failed: {result}")

        logger.info(f"Starting video generation with text: '{args.text[:50]}...'")

        # Poll for completion
        final_id = await poll_job_status(job_id)

        return [
            TextContent(
                type="text", text=f"Video generated successfully. Final file ID: {final_id}"
            )
        ]

    except Exception as e:
        error_msg = f"Video generation failed: {str(e)}"
        logger.error(error_msg)
        return [TextContent(type="text", text=error_msg)]


@server.call_tool()
async def merge_videos(arguments: Dict[str, Any]) -> List[TextContent]:
    """Merge multiple videos into one"""
    args = MergeVideosArgs(**arguments)

    data = {"video_ids": args.video_ids}
    if args.background_music_id:
        data["background_music_id"] = args.background_music_id
        data["background_music_volume"] = "0.3"

    try:
        logger.info(f"Merging {len(args.video_ids.split(','))} videos")

        result = await make_request("POST", "/api/v1/media/video-tools/merge", data=data)
        job_id = result.get("id") or result.get("file_id")

        if not job_id:
            raise ValueError(f"Video merge failed: {result}")

        # Poll for completion
        final_id = await poll_job_status(job_id)

        return [
            TextContent(type="text", text=f"Videos merged successfully. Final file ID: {final_id}")
        ]

    except Exception as e:
        error_msg = f"Video merge failed: {str(e)}"
        logger.error(error_msg)
        return [TextContent(type="text", text=error_msg)]


@server.call_tool()
async def poll_job_status_tool(arguments: Dict[str, Any]) -> List[TextContent]:
    """Poll job status until completion"""
    args = PollJobArgs(**arguments)

    try:
        final_id = await poll_job_status(args.job_id, args.max_attempts, args.delay)

        return [TextContent(type="text", text=f"Job {args.job_id} completed successfully")]

    except Exception as e:
        error_msg = f"Job polling failed: {str(e)}"
        logger.error(error_msg)
        return [TextContent(type="text", text=error_msg)]


@server.call_tool()
async def download_file(arguments: Dict[str, Any]) -> List[TextContent]:
    """Download file from Media Server"""
    file_id = arguments["file_id"]

    try:
        response = await client.get(f"{MEDIA_SERVER_URL}/api/v1/media/storage/{file_id}")
        response.raise_for_status()

        # Return file info (actual download would be handled by client)
        return [
            TextContent(
                type="text",
                text=f"File {file_id} is ready for download ({len(response.content)} bytes)",
            )
        ]

    except Exception as e:
        error_msg = f"File download failed: {str(e)}"
        logger.error(error_msg)
        return [TextContent(type="text", text=error_msg)]


@server.call_tool()
async def get_video_info(arguments: Dict[str, Any]) -> List[TextContent]:
    """Get video metadata"""
    file_id = arguments["file_id"]

    try:
        result = await make_request("GET", f"/api/v1/media/video-tools/info/{file_id}")

        info_text = f"""Video Information for {file_id}:
- Duration: {result.get("duration", "Unknown")}
- Resolution: {result.get("width", "?")}x{result.get("height", "?")}
- Codec: {result.get("codec", "Unknown")}
- Size: {result.get("size_bytes", 0)} bytes
"""

        return [TextContent(type="text", text=info_text)]

    except Exception as e:
        error_msg = f"Failed to get video info: {str(e)}"
        logger.error(error_msg)
        return [TextContent(type="text", text=error_msg)]


@asynccontextmanager
async def lifespan(server):
    """Manage server lifecycle"""
    global IMAGE_AUTH
    IMAGE_AUTH = os.getenv("IMAGE_AUTH")

    yield

    # Cleanup
    await client.aclose()


async def main():
    """Run the MCP server"""
    # Import here to avoid circular imports
    import os

    async with mcp.server.stdio.stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream, server.create_initialization_options())


if __name__ == "__main__":
    asyncio.run(main())
