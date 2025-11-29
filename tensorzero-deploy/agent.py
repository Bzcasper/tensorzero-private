"""
DIY Video Production Agent - Fast Agent Workflow

This module demonstrates a complete fast-agent workflow for automated DIY video production.
All prompts are stored in TensorZero templates - NO PROMPTS IN CODE.

Architecture:
- TensorZero: Handles all AI prompts and model routing
- Fast Agent: Orchestrates the workflow logic
- Media Server: Provides video/audio processing APIs
"""

"""
DIY Video Production Fast Agent Workflow

This module implements a complete fast-agent workflow for automated DIY video production.
All prompts are stored in TensorZero templates - NO PROMPTS IN CODE.

Workflow:
1. script_generator -> Generate video script
2. prompt_enhancer -> Enhance image prompts
3. image_generator -> Generate images
4. video_segment_creator -> Create video segments
5. quality_evaluator -> Evaluate final quality
"""

import asyncio
import uuid
import logging
import traceback
from typing import Dict, Any, List, Optional
from fast_agent.core.fastagent import FastAgent
from tensorzero_client import TensorZeroClient
from media_server_client import MediaServerClient
import os

# Setup comprehensive logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(), logging.FileHandler("diy_video_production.log")],
)
logger = logging.getLogger(__name__)

# Initialize Fast Agent
fast = FastAgent("DIY Video Producer")

# Initialize clients
tz_client = TensorZeroClient()
media_client = MediaServerClient(
    base_url=os.getenv(
        "MEDIA_SERVER_URL",
        "https://2281a5a294754c19f8c9e2df0be013fb-bobby-casper-4235.aiagentsaz.com",
    )
)


class MetricsReporter:
    """Handles metrics reporting to TensorZero for quality improvement"""

    def __init__(self, tz_client: TensorZeroClient):
        self.tz_client = tz_client
        self.episode_id = None

    def start_episode(self, project_description: str) -> str:
        """Start a new episode for tracking user satisfaction"""
        import uuid

        self.episode_id = str(uuid.uuid4())
        logger.info(f"📊 Started metrics episode: {self.episode_id}")
        return self.episode_id

    async def report_inference_metrics(
        self, inference_id: str, quality_score: float, generation_time: float, estimated_cost: float
    ):
        """Report metrics for a specific inference"""
        try:
            await self.tz_client.feedback(
                inference_id=inference_id, metric_name="video_quality_score", value=quality_score
            )
            await self.tz_client.feedback(
                inference_id=inference_id,
                metric_name="video_generation_time",
                value=generation_time,
            )
            await self.tz_client.feedback(
                inference_id=inference_id, metric_name="video_generation_cost", value=estimated_cost
            )
            await self.tz_client.feedback(
                inference_id=inference_id, metric_name="video_completion_rate", value=True
            )
            logger.info(
                f"📊 Reported inference metrics: quality={quality_score}, time={generation_time}s, cost=${estimated_cost}"
            )
        except Exception as e:
            logger.error(f"❌ Failed to report inference metrics: {e}")

    async def report_user_satisfaction(self, satisfied: bool):
        """Report user satisfaction for the episode"""
        if not self.episode_id:
            logger.warning("⚠️ No active episode for user satisfaction reporting")
            return

        try:
            await self.tz_client.feedback(
                inference_id=None,  # Episode-level metric
                metric_name="user_satisfaction",
                value=satisfied,
                episode_id=self.episode_id,
            )
            logger.info(f"📊 Reported user satisfaction: {satisfied}")
        except Exception as e:
            logger.error(f"❌ Failed to report user satisfaction: {e}")

    def map_score_to_rating(self, score: float) -> str:
        """Map numeric quality score to rating category"""
        if score >= 9:
            return "EXCELLENT"
        elif score >= 7:
            return "GOOD"
        elif score >= 5:
            return "FAIR"
        else:
            return "POOR"


# Initialize metrics reporter
metrics = MetricsReporter(tz_client)


@fast.agent(
    name="script_generator",
    # NO PROMPTS HERE - they're in TensorZero templates!
)
async def script_generator(
    agent, project_description: str, content_mode: str = "kid"
) -> Dict[str, Any]:
    """Generate DIY video script using TensorZero"""

    logger.info(f"🎬 Generating script for: {project_description}")

    try:
        result = await tz_client.inference(
            function_name="script_generator",
            input={
                "project_description": project_description,
                "content_mode": content_mode,
                "target_duration_seconds": 180,
                "expected_steps": 5,
            },
        )

        # Store inference_id for feedback later
        agent.context["script_inference_id"] = result.get("inference_id")

        # Return the generated script
        return result

    except Exception as e:
        logger.error(f"❌ Script generation failed: {str(e)}")
        raise


@fast.agent(
    name="quality_evaluator",
    # NO PROMPTS HERE
)
async def quality_evaluator(
    agent, video_metadata: Dict[str, Any], expected_outcome: str
) -> Dict[str, Any]:
    """Evaluate video quality using TensorZero"""

    logger.info(f"⭐ Evaluating video quality")

    try:
        result = await tz_client.inference(
            function_name="quality_evaluator",
            input={
                "video_metadata": video_metadata,
                "expected_outcome": expected_outcome,
                "quality_criteria": [
                    "clarity",
                    "engagement",
                    "educational_value",
                    "technical_quality",
                ],
            },
        )

        evaluation = result

        # Send feedback to TensorZero for learning
        if "script_inference_id" in agent.context:
            await tz_client.feedback(
                inference_id=agent.context["script_inference_id"],
                metric_name="video_quality_score",
                value=evaluation.get("score", 0.0),
            )

        logger.info(f"✅ Quality evaluation: {evaluation.get('score', 0.0)}/10")
        return evaluation

    except Exception as e:
        logger.error(f"❌ Quality evaluation failed: {str(e)}")
        raise


@fast.agent(
    name="video_quality_judge",
    # Advanced evaluation with vision analysis
)
async def video_quality_judge(agent, workflow_state: dict, previous_feedback: str = "") -> str:
    """Advanced video quality evaluation using TensorZero video_evaluator"""

    logger.info(f"⭐ Advanced video quality analysis")

    try:
        # Call TensorZero function
        eval_result = await tz_client.inference(
            function_name="quality_evaluator",
            input={
                "video_url": workflow_state.get("video_url", ""),
                "script": workflow_state.get("script", {}),
                "target_duration": workflow_state.get("target_duration", 180),
            },
        )

        # Report Metric to TensorZero immediately
        # We use a dummy inference ID or the ID from the generation step if tracked
        await tz_client.report_metric(
            metric_name="video_quality_score",
            value=eval_result["quality_score"],
            inference_id=workflow_state.get("last_gen_inference_id"),
        )

        # Return the categorical label required by evaluator_optimizer
        # We also attach the feedback to the return string for the generator to parse
        return f"{eval_result['rating_label']} | Feedback: {eval_result['feedback']} | Flags: script={eval_result.get('regenerate_script', False)}, images={eval_result.get('regenerate_images', False)}"

    except Exception as e:
        logger.error(f"❌ Advanced evaluation failed: {str(e)}")
        return "POOR | Feedback: Evaluation failed | Flags: script=True, images=True"


@fast.agent(name="image_generator", servers=["media_server"])
async def image_generator(agent, enhanced_prompt: str) -> str:
    """Generate image using Gemini API and upload via MCP"""

    logger.info(f"🎨 Generating image from prompt")

    try:
        # Direct API call (could also route through TensorZero)
        image_bytes = await generate_image_gemini(enhanced_prompt)

        # Convert to base64 for MCP tool
        import base64

        file_data = base64.b64encode(image_bytes).decode("utf-8")

        # Use MCP tool to upload
        result = await agent.call_tool(
            "upload_file",
            {"file_data": file_data, "media_type": "image", "filename": "generated_image.png"},
        )

        # Extract file ID from result
        result_text = result[0].text
        file_id = result_text.split("File ID: ")[1] if "File ID: " in result_text else result_text

        logger.info(f"✅ Image generated and uploaded: {file_id}")
        return file_id

    except Exception as e:
        logger.error(f"❌ Image generation failed: {str(e)}")
        raise


@fast.agent(name="video_segment_creator", servers=["media_server"])
async def video_segment_creator(agent, background_id: str, narration: str) -> str:
    """Create video segment with TTS and captions using Media Server"""

    logger.info(f"🎬 Creating video segment for: {narration[:30]}...")

    try:
        # Generate video using Media Server
        job_id = await media_client.generate_captioned_video(
            background_id=background_id, text=narration, width=1080, height=1920
        )

        # Poll for completion
        video_id = await media_client.poll_job(job_id)

        logger.info(f"✅ Video segment created: {video_id}")
        return video_id

    except Exception as e:
        logger.error(f"❌ Video segment creation failed: {str(e)}")
        raise


@fast.agent("video_pipeline_smart")
async def smart_producer(agent, request: str, previous_feedback: str = ""):
    """Smart producer that parses feedback to avoid full regeneration (Cost Optimization)"""
    import json

    # Parse input request (assuming it's a JSON string of state)
    try:
        state = json.loads(request)
    except:
        # Initial run
        state = {"project_description": request, "content_mode": "kid"}

    # Check feedback to decide what to run
    regen_script = True
    regen_images = True

    if previous_feedback:
        # Parse flags from the judge's output string
        if "script=False" in previous_feedback:
            regen_script = False
        if "images=False" in previous_feedback:
            regen_images = False
        logger.info(f"🔄 Refinement Loop: Script={regen_script}, Images={regen_images}")

    # --- EXECUTE WORKFLOW STEPS CONDITIONALLY ---

    if regen_script or not state.get("script"):
        # Call TensorZero Script Generator
        script_data = await tz_client.inference(
            function_name="script_generator",
            input={
                "project_description": state["project_description"],
                "content_mode": state.get("content_mode", "kid"),
                "target_duration_seconds": 180,
            },
        )
        state["script"] = script_data
        # Store inference ID for metrics linking
        state["last_gen_inference_id"] = script_data.get("_inference_id")

    if regen_images or not state.get("image_ids"):
        # Call Image Generation Logic (abstracted)
        # Only generating images for scenes in the script
        logger.info("🎨 Generating images for script scenes...")

        # Prepare all prompts for batch enhancement
        all_prompts = [
            state["script"]["intro"]["image_prompt"],
            *[material["image_prompt"] for material in state["script"]["materials"]],
            *[step["image_prompt"] for step in state["script"]["steps"]],
            state["script"]["completion"]["image_prompt"],
        ]

        # Enhance all prompts at once
        enhanced_result = await tz_client.inference(
            function_name="prompt_enhancer",
            input={"scenes": [{"visual_desc": prompt} for prompt in all_prompts]},
        )
        enhanced_prompts = enhanced_result["enhanced_prompts"]

        # Generate images
        image_ids = []
        for enhanced_prompt in enhanced_prompts:
            # Convert to base64 for MCP tool
            import base64

            image_bytes = await generate_image_gemini(enhanced_prompt)
            file_data = base64.b64encode(image_bytes).decode("utf-8")

            # Use MCP tool to upload
            result = await agent.call_tool(
                "upload_file",
                {"file_data": file_data, "media_type": "image", "filename": "generated_image.png"},
            )

            # Extract file ID from result
            result_text = result[0].text
            file_id = (
                result_text.split("File ID: ")[1] if "File ID: " in result_text else result_text
            )
            image_ids.append(file_id)

        state["image_ids"] = image_ids

        # Generate audio for all scenes
        logger.info("🎙️ Generating audio for script scenes...")
        dialogues = [
            state["script"]["intro"]["narration"],
            *[material["name"] for material in state["script"]["materials"]],
            *[
                f"Step {step['step_number']}: {step['narration']}"
                for step in state["script"]["steps"]
            ],
            state["script"]["completion"]["narration"],
        ]

        audio_ids = []
        for dialogue in dialogues:
            # Use MCP tool to generate TTS
            result = await agent.call_tool(
                "generate_captioned_video",
                {
                    "background_id": "dummy",  # We just need audio, will extract it
                    "text": dialogue,
                    "width": 1080,
                    "height": 1920,
                    "voice": "af_heart",
                },
            )
            # For simplicity, we'll use the video ID as audio ID
            # In practice, you'd extract just the audio
            result_text = result[0].text
            job_id = (
                result_text.split("job started: ")[1] if "job started: " in result_text else "dummy"
            )
            audio_ids.append(job_id)

        state["audio_ids"] = audio_ids

    # Return updated state as string for the next loop iteration
    return json.dumps(state)


@fast.chain(
    name="diy_video_pipeline",
    sequence=[
        "script_generator",
        "image_generator",  # Will be called multiple times
        "video_segment_creator",  # Will be called multiple times
        "quality_evaluator",
    ],
)
@fast.evaluator_optimizer(
    name="quality_assured_video",
    generator="video_pipeline_smart",
    evaluator="video_quality_judge",
    min_rating="GOOD",  # Requires score >= 7
    max_refinements=3,
)
@fast.evaluator_optimizer(
    name="quality_assured_video",
    generator="diy_video_pipeline",
    evaluator="video_quality_judge",
    min_rating="GOOD",  # Require at least GOOD quality (score >= 7)
    max_refinements=3,
)
async def main(project_description: str = "paper airplane"):
    """Main workflow - orchestrates everything with quality assurance"""

    # Start metrics episode
    episode_id = metrics.start_episode(project_description)
    start_time = asyncio.get_event_loop().time()

    async with fast.run() as agent:
        try:
            # Use evaluator-optimizer for quality-assured video generation
            print("🎯 Starting quality-assured video generation...")

            result = await agent.quality_assured_video.send(project_description=project_description)

            # Calculate metrics
            end_time = asyncio.get_event_loop().time()
            generation_time = end_time - start_time
            estimated_cost = 0.05  # Rough estimate based on API calls

            # Report metrics if we have inference IDs
            if hasattr(result, "script") and result.get("script"):
                # This would be populated by the individual agents
                pass  # Metrics reporting handled in individual agents

            print(f"✅ Quality-assured video completed!")
            print(f"📊 Generation time: {generation_time:.1f}s")
            print(f"💰 Estimated cost: ${estimated_cost}")

            # Ask for user feedback (in a real system, this would be collected via UI)
            user_satisfied = (
                input("Are you satisfied with the generated video? (y/n): ").lower().startswith("y")
            )
            await metrics.report_user_satisfaction(user_satisfied)

            return result

        except Exception as e:
            logger.error(f"💥 Quality-assured pipeline failed: {str(e)}")
            return {"status": "error", "message": str(e), "episode_id": episode_id}


async def run_basic_pipeline(project_description: str = "paper airplane"):
    """Fallback basic pipeline without evaluator-optimizer"""

    async with fast.run() as agent:
        episode_id = str(uuid.uuid4())
        start_time = asyncio.get_event_loop().time()

        # Step 1: Generate script
        print("📝 Generating script...")
        script = await agent.script_generator.send(
            project_description=project_description, content_mode="kid"
        )

        print(f"✅ Script: {script.get('title', 'Unknown')}")

        # Step 2: Generate images for each scene
        print("🎨 Generating images...")

        # Prepare all prompts for batch enhancement
        all_prompts = [
            script["intro"]["image_prompt"],
            *[material["image_prompt"] for material in script["materials"]],
            *[step["image_prompt"] for step in script["steps"]],
            script["completion"]["image_prompt"],
        ]

        # Enhance all prompts at once (this would need to be updated to handle batch processing)
        # For now, enhance them individually
        enhanced_prompts = []
        for prompt in all_prompts:
            enhanced = await agent.prompt_enhancer.send(
                basic_prompt=prompt, context="DIY tutorial scene"
            )
            enhanced_prompts.append(enhanced)

        # Generate images
        image_ids = []
        for enhanced_prompt in enhanced_prompts:
            image_id = await agent.image_generator.send(enhanced_prompt)
            image_ids.append(image_id)

        print(f"✅ Generated {len(image_ids)} images total")

        # Step 3: Create video segments
        print("🎬 Creating video segments...")
        video_ids = []

        # Create video segments for each scene
        scenes = [
            ("intro", script["intro"]["narration"]),
            *[
                (f"material_{i}", material["name"])
                for i, material in enumerate(script["materials"])
            ],
            *[
                (f"step_{i}", f"Step {step['step_number']}: {step['narration']}")
                for i, step in enumerate(script["steps"])
            ],
            ("completion", script["completion"]["narration"]),
        ]

        for i, (scene_type, narration) in enumerate(scenes):
            video_id = await agent.video_segment_creator.send(
                background_id=image_ids[i], narration=narration
            )
            video_ids.append(video_id)

        # Step 4: Merge videos using MCP
        print("🔗 Merging videos...")
        merge_result = await agent.call_tool("merge_videos", {"video_ids": ",".join(video_ids)})
        merge_text = merge_result[0].text
        final_video_id = (
            merge_text.split("Final file ID: ")[1]
            if "Final file ID: " in merge_text
            else merge_text
        )

        # Step 5: Evaluate quality
        print("⭐ Evaluating quality...")
        evaluation = await agent.quality_evaluator.send(
            video_metadata={
                "title": script["title"],
                "duration_seconds": 180,
                "scene_count": len(script["steps"]),
                "has_audio": True,
                "target_audience": "kid",
            },
            expected_outcome=project_description,
        )

        # Calculate and report metrics
        end_time = asyncio.get_event_loop().time()
        generation_time = end_time - start_time
        estimated_cost = 0.05  # Rough estimate

        if "script_inference_id" in agent.context:
            await metrics.report_inference_metrics(
                agent.context["script_inference_id"],
                evaluation.get("score", 0.0),
                generation_time,
                estimated_cost,
            )

        print(f"✅ Quality Score: {evaluation.get('score', 0.0)}/10")
        print(f"📹 Final Video: {final_video_id}")
        print(f"📊 Metrics: {generation_time:.1f}s, ${estimated_cost}")

        return {
            "video_id": final_video_id,
            "script": script,
            "quality_score": evaluation.get("score", 0.0),
            "passed": evaluation.get("pass", False),
            "feedback": evaluation.get("feedback", ""),
            "generation_time": generation_time,
            "estimated_cost": estimated_cost,
        }


# Helper function for image generation (would be in separate module)
async def generate_image_gemini(prompt: str) -> bytes:
    """Placeholder for Gemini image generation"""
    # This would call the actual Gemini API
    # For now, return dummy bytes representing a minimal PNG image
    logger.info(f"🤖 Would generate image with Gemini for prompt: {prompt[:100]}...")

    # Create a minimal valid 1x1 PNG image
    png_data = (
        b"\x89PNG\r\n\x1a\n"  # PNG signature
        b"\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x02\x00\x00\x00\x90wS\xde"  # IHDR chunk
        b"\x00\x00\x00\rIDATx\x9cc\xf8\x00\x00\x00\x01\x00\x01\x00\x18\xdd\x8d\xb4"  # IDAT chunk
        b"\x00\x00\x00\x00IEND\xae\x42\x60\x82"  # IEND chunk
    )
    return png_data


if __name__ == "__main__":
    asyncio.run(main())
