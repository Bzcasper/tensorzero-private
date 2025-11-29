"""
Mock TensorZero Service for Testing

A minimal FastAPI server that mimics TensorZero inference endpoints
for testing the video production pipeline without requiring real API keys.
"""

import asyncio
import json
import uuid
from typing import Dict, Any, Optional
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn

app = FastAPI(title="Mock TensorZero", version="1.0.0")


class InferenceRequest(BaseModel):
    function_name: str
    input: Dict[str, Any]
    stream: bool = False
    variant_name: Optional[str] = None
    episode_id: Optional[str] = None


@app.post("/inference")
async def inference(request: InferenceRequest) -> Dict[str, Any]:
    """Mock inference endpoint"""

    # Generate mock inference ID
    inference_id = str(uuid.uuid4())

    # Mock responses based on function name
    if request.function_name == "script_generator":
        response = generate_mock_script(request.input)
    elif request.function_name == "prompt_enhancer":
        response = enhance_mock_prompts(request.input)
    elif request.function_name == "quality_evaluator":
        response = evaluate_mock_quality(request.input)
    else:
        raise HTTPException(status_code=400, detail=f"Unknown function: {request.function_name}")

    # Add inference metadata
    response["_inference_id"] = inference_id

    return {"output": {"parsed": response}}


@app.post("/feedback")
async def feedback(data: Dict[str, Any]) -> Dict[str, Any]:
    """Mock feedback endpoint"""
    print(f"📊 Mock feedback received: {data}")
    return {"success": True}


@app.get("/health")
async def health():
    """Health check"""
    return {"status": "healthy"}


def generate_mock_script(input_data: Dict[str, Any]) -> Dict[str, Any]:
    """Generate a mock video script"""
    project = input_data.get("project_description", "unknown project")
    content_mode = input_data.get("content_mode", "kid")

    return {
        "title": f"How to Make a {project.title()}",
        "intro": {
            "narration": f"Welcome! Today we're going to learn how to make a {project}. This is perfect for {content_mode}s!",
            "image_prompt": f"A cheerful {content_mode} looking excited about making a {project}"
        },
        "materials": [
            {
                "name": "Paper",
                "image_prompt": f"A sheet of colorful paper on a clean table"
            },
            {
                "name": "Scissors",
                "image_prompt": f"Safe children's scissors on a table"
            }
        ],
        "steps": [
            {
                "step_number": 1,
                "title": "Prepare the paper",
                "narration": "First, take your paper and fold it in half.",
                "image_prompt": f"Hands folding paper in half for a {project}"
            },
            {
                "step_number": 2,
                "title": "Make the first fold",
                "narration": "Now fold the top corners down to the center.",
                "image_prompt": f"Step-by-step folding of paper for {project} creation"
            },
            {
                "step_number": 3,
                "title": "Complete the shape",
                "narration": "Fold the wings and you're done!",
                "image_prompt": f"Final step of creating a {project}"
            }
        ],
        "completion": {
            "narration": f"Great job! You now have your own {project}. Have fun playing with it!",
            "image_prompt": f"A completed {project} ready to use"
        },
        "thumbnail_prompt": f"An attractive thumbnail showing a {project} with text overlay"
    }


def enhance_mock_prompts(input_data: Dict[str, Any]) -> Dict[str, Any]:
    """Enhance image prompts"""
    scenes = input_data.get("scenes", [])

    enhanced_prompts = []
    for scene in scenes:
        basic_prompt = scene.get("visual_desc", "")
        enhanced = f"{basic_prompt}, high quality digital illustration, vibrant colors, detailed, professional, 1080p, cinematic lighting"
        enhanced_prompts.append(enhanced)

    return {"enhanced_prompts": enhanced_prompts}


def evaluate_mock_quality(input_data: Dict[str, Any]) -> Dict[str, Any]:
    """Mock quality evaluation"""
    # Always return a passing score for testing
    return {
        "score": 8.5,
        "feedback": "Excellent quality! Clear instructions, good visuals, and appropriate for the target audience.",
        "pass": True
    }


if __name__ == "__main__":
