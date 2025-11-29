#!/usr/bin/env python3
"""
DIY Video Production Agent - Complete Runner

This script demonstrates the complete DIY video production pipeline:
1. TensorZero script generation
2. Prompt enhancement
3. Image generation via Gemini
4. Video assembly via Media Server MCP
5. Quality evaluation

Usage:
    python run_diy_video.py "your project description"

Example:
    python run_diy_video.py "how to make paper airplanes that fly far"
"""

import asyncio
import sys
import os
from pathlib import Path

# Add the project root to Python path
sys.path.insert(0, str(Path(__file__).parent))


async def main():
    """Run the complete DIY video production pipeline with quality assurance"""
    if len(sys.argv) < 2:
        print("Usage: python run_diy_video.py 'project description' [--basic]")
        print("Example: python run_diy_video.py 'how to make paper airplanes'")
        print("Use --basic to skip quality assurance loop")
        sys.exit(1)

    project_description = sys.argv[1]
    use_quality_assurance = "--basic" not in sys.argv

    try:
        # Import here to avoid import errors in environments without fast-agent
        from agent import main as run_quality_pipeline, run_basic_pipeline

        if use_quality_assurance:
            print("🎯 Using Quality-Assured Pipeline (with evaluator-optimizer)")
            result = await run_quality_pipeline(project_description)
        else:
            print("🔄 Using Basic Pipeline (single pass)")
            result = await run_basic_pipeline(project_description)

        if result.get("status") == "error":
            print("❌ PIPELINE FAILED")
            print("=" * 50)
            print(f"Error: {result.get('message', 'Unknown error')}")
            print("=" * 50)
            sys.exit(1)

        print("🎉 PIPELINE COMPLETE")
        print("=" * 50)
        print(f"Video ID: {result['video_id']}")
        print(f"Quality Score: {result['quality_score']}/10")
        if result.get("passed"):
            print("✅ Quality Check: PASSED")
        else:
            print("❌ Quality Check: FAILED - Consider regenerating")
        if result.get("feedback"):
            print(f"Feedback: {result['feedback'][:100]}...")
        if result.get("generation_time"):
            print(f"⏱️ Generation Time: {result['generation_time']:.1f}s")
        if result.get("estimated_cost"):
            print(f"💰 Estimated Cost: ${result['estimated_cost']:.3f}")
        print("=" * 50)

        # Quality improvement suggestions
        if result.get("quality_score", 0) < 8.0:
            print("\n💡 Quality Improvement Tips:")
            print("- Check script clarity and completeness")
            print("- Verify image prompts are descriptive")
            print("- Ensure audio narration is natural")
            print("- Review video transitions and pacing")
            print("- Consider regenerating with different model variants")

    except ImportError as e:
        print("❌ Missing dependencies. Please run:")
        print("   uv sync")
        print(f"Error: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Pipeline failed: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)

    project_description = sys.argv[1]

    try:
        # Import here to avoid import errors in environments without fast-agent
        from agent import main as run_pipeline

        print("🎬 Starting DIY Video Production Pipeline")
        print(f"📝 Project: {project_description}")
        print("=" * 50)

        result = await run_pipeline(project_description)

        print("\n" + "=" * 50)
        print("🎉 PIPELINE COMPLETE")
        print("=" * 50)
        print(f"Video ID: {result['video_id']}")
        print(f"Quality Score: {result['quality_score']}/10")
        print(f"Passed: {'✅' if result.get('passed', False) else '❌'}")
        if result.get("feedback"):
            print(f"Feedback: {result['feedback'][:100]}...")
        print("=" * 50)

    except ImportError as e:
        print("❌ Missing dependencies. Please run:")
        print("   uv sync")
        print(f"Error: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Pipeline failed: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
