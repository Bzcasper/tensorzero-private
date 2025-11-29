"""
Vercel-compatible API handler for DIY Video Generation

This module provides serverless API endpoints for video generation
deployed on Vercel platform.
"""

import os
import json
from http.server import BaseHTTPRequestHandler
from tensorzero_client import TensorZeroClient
from media_server_client import MediaServerClient


class handler(BaseHTTPRequestHandler):
    """Vercel serverless function handler"""

    def __init__(self, *args, **kwargs):
        # Initialize clients
        self.tz_client = TensorZeroClient(
            base_url=os.getenv("TENSORZERO_BASE_URL", "http://localhost:3000")
        )
        self.media_client = MediaServerClient(
            base_url=os.getenv(
                "MEDIA_SERVER_URL",
                "https://2281a5a294754c19f8c9e2df0be013fb-bobby-casper-4235.aiagentsaz.com",
            )
        )
        super().__init__(*args, **kwargs)

    def do_POST(self):
        """Handle POST requests"""
        if self.path == "/api/generate_video":
            self.handle_generate_video()
        else:
            self.send_error(404, "Endpoint not found")

    def do_GET(self):
        """Handle GET requests"""
        if self.path == "/api/health":
            self.handle_health()
        else:
            self.send_error(404, "Endpoint not found")

    def handle_generate_video(self):
        """Handle video generation requests"""
        try:
            # Read request body
            content_length = int(self.headers["Content-Length"])
            post_data = self.rfile.read(content_length)
            request_data = json.loads(post_data.decode("utf-8"))

            # Extract parameters
            project_description = request_data.get("project_description", "")
            content_mode = request_data.get("content_mode", "kid")
            target_duration = request_data.get("target_duration", 180)

            # Generate script
            script_result = self.tz_client.inference(
                "script_generator",
                {
                    "project_description": project_description,
                    "content_mode": content_mode,
                    "target_duration_seconds": target_duration,
                },
            )
            script = script_result

            # Prepare scenes for enhancement
            scenes = [
                {"visual_desc": script["intro"]["image_prompt"]},
                *[{"visual_desc": material["image_prompt"]} for material in script["materials"]],
                *[{"visual_desc": step["image_prompt"]} for step in script["steps"]],
                {"visual_desc": script["completion"]["image_prompt"]},
            ]

            # Enhance prompts
            enhanced_result = self.tz_client.inference("prompt_enhancer", {"scenes": scenes})
            enhanced_prompts = enhanced_result["enhanced_prompts"]

            # Generate images
            image_ids = self.media_client.generate_images(enhanced_prompts)

            # Generate audio
            dialogues = [
                script["intro"]["narration"],
                *[material["name"] for material in script["materials"]],
                *[f"Step {step['step_number']}: {step['narration']}" for step in script["steps"]],
                script["completion"]["narration"],
            ]
            audio_ids = self.media_client.generate_audio(dialogues)

            # Calculate timings (simplified)
            num_segments = len(image_ids)
            segment_duration = target_duration // num_segments
            timings = [segment_duration] * num_segments

            # Assemble video
            video_id = self.media_client.assemble_video(image_ids, audio_ids, timings)

            # Quality evaluation
            evaluation = self.tz_client.inference(
                "quality_evaluator",
                {
                    "video_url": f"https://media.example.com/{video_id}",
                    "script": script,
                    "target_duration": target_duration,
                },
            )

            # Return response
            response = {
                "success": True,
                "video_id": video_id,
                "video_url": f"https://media.example.com/{video_id}",
                "quality_score": evaluation.get("score", 0.0),
                "feedback": evaluation.get("feedback", ""),
                "passed": evaluation.get("pass", False),
            }

            self.send_response(200)
            self.send_header("Content-type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(response).encode())

        except Exception as e:
            error_response = {"success": False, "error": str(e)}
            self.send_response(500)
            self.send_header("Content-type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(error_response).encode())

    def handle_health(self):
        """Handle health check"""
        health_response = {"status": "healthy", "service": "diy-video-api", "version": "1.0.0"}

        self.send_response(200)
        self.send_header("Content-type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps(health_response).encode())
