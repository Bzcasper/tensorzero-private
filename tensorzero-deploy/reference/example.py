#!/usr/bin/env python3
"""
DIY Video Production Script for Creation Companions DIY YouTube Channel
Starring CC (Creation Companion) - a happy simple cartoon character

Features:
- Portrait orientation (1080x1920) for YouTube Shorts/TikTok
- Title screen with Qwen Edit text overlay
- AI-generated DIY project scripts
- Materials list with individual images
- Step-by-step process scenes
- Cerebras prompt enhancement for high-quality images
"""

import requests
import time
import json
import yaml
import cloudinary
import cloudinary.uploader
import os
from io import BytesIO
from pathlib import Path

# Load .env file manually
if os.path.exists(".env"):
    with open(".env") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, value = line.split("=", 1)
                os.environ[key] = value

# =============================================================================
# CONFIGURATION
# =============================================================================


def load_config(config_path=None):
    """Load configuration from YAML file"""
    if config_path is None:
        # Look for config in same directory as script
        script_dir = Path(__file__).parent
        config_path = script_dir / "diy_config.yaml"

    if config_path.exists():
        with open(config_path, "r") as f:
            return yaml.safe_load(f)
    else:
        print(f"  Warning: Config file not found at {config_path}, using defaults")
        return {}


# Load config
CONFIG = load_config()

# Server URLs (from config or defaults)
MEDIA_SERVER = CONFIG.get("endpoints", {}).get(
    "media_server",
    "https://2281a5a294754c19f8c9e2df0be013fb-bobby-casper-4235.aiagentsaz.com",
)
IMAGE_GEN = CONFIG.get("endpoints", {}).get(
    "image_gen",
    "https://ai-tool-pool--nunchaku-qwen-image-fastapi-fastapi-app.modal.run",
)
IMAGE_EDIT = CONFIG.get("endpoints", {}).get(
    "image_edit", "https://trap--qwen-edit-lora-fastapi-fastapi-app.modal.run"
)
XAI_API = CONFIG.get("endpoints", {}).get(
    "xai_api", "https://api.x.ai/v1/chat/completions"
)

# API Keys
XAI_API_KEY = os.environ.get("XAI_API_KEY", os.environ.get("GROK_API_KEY", ""))
IMAGE_AUTH = os.environ.get("IMAGE_AUTH", "")

# Voice config
voice_config = CONFIG.get("voice", {})
elevenlabs_config = voice_config.get("elevenlabs", {})
ELEVENLABS_API_KEY = os.environ.get("ELEVENLABS_API_KEY", "")
ELEVENLABS_VOICE_ID = elevenlabs_config.get("voice_id", "NNl6r8mD7vthiJatiJt1")

# Cloudinary config
CLOUDINARY_CONFIG = {
    "cloud_name": os.environ.get("CLOUDINARY_CLOUD_NAME", ""),
    "api_key": os.environ.get("CLOUDINARY_API_KEY", ""),
    "api_secret": os.environ.get("CLOUDINARY_API_SECRET", ""),
}

# Video dimensions
video_config = CONFIG.get("video", {})
PORTRAIT_WIDTH = video_config.get("width", 1080)
PORTRAIT_HEIGHT = video_config.get("height", 1920)

# Headers
IMAGE_HEADERS = {"Authorization": IMAGE_AUTH, "Content-Type": "application/json"}

XAI_HEADERS = {
    "Authorization": f"Bearer {XAI_API_KEY}",
    "Content-Type": "application/json",
}

# CC Character description
CC_CHARACTER = CONFIG.get("character", {}).get(
    "description",
    """CC (Creation Companion), a cheerful and friendly cartoon character with:
- Simple rounded body shape, bright and colorful
- Big expressive eyes with a warm smile
- Wearing a colorful apron or tool belt
- Friendly and approachable demeanor
- Clean cartoon style, not too detailed""",
)

# Caption config
caption_config = CONFIG.get("captions", {})

# Effects config
effects_config = CONFIG.get("effects", {})
ken_burns_config = effects_config.get("ken_burns", {})
deaify_config = effects_config.get("deaify", {})

# Camera moves config
camera_config = CONFIG.get("camera_moves", {})

# Audio config
audio_config = CONFIG.get("audio", {})

# Mode config (kid or adult)
CONTENT_MODE = CONFIG.get("mode", "kid")
IMAGE_PROVIDER = CONFIG.get("image_provider", "gemini")

# Gemini config
gemini_config = CONFIG.get("gemini", {})
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
GEMINI_MODEL = gemini_config.get("model", "gemini-2.5-flash-image")
GEMINI_ASPECT_RATIO = gemini_config.get("aspect_ratio", "9:16")

# Get mode-specific character description
character_config = CONFIG.get("character", {})
if CONTENT_MODE == "adult":
    CC_CHARACTER = character_config.get("adult_description", CC_CHARACTER)
else:
    CC_CHARACTER = character_config.get("kid_description", CC_CHARACTER)

# Get mode-specific style
STYLE_PRESET = gemini_config.get("styles", {}).get(CONTENT_MODE, "")

# =============================================================================
# PROMPT ENHANCEMENT
# =============================================================================


def enhance_image_prompt(basic_prompt, context="DIY tutorial"):
    """Enhance a basic prompt for high-quality image generation"""
    enhanced = f"""{basic_prompt},
    high quality digital illustration,
    vibrant colors, clean lines,
    professional tutorial style,
    bright and well-lit,
    sharp focus, detailed textures,
    {context} aesthetic,
    4k quality, crisp and clear"""
    return enhanced


# =============================================================================
# XAI GROK STORY GENERATION
# =============================================================================


def generate_diy_script(project_query):
    """Use xAI Grok to generate a complete DIY video script"""
    print("\n" + "=" * 70)
    print("🎬 GENERATING DIY SCRIPT WITH GROK")
    print("=" * 70)
    print(f"Project: {project_query}\n")

    system_prompt = """You are a DIY video script writer for the "Creation Companions DIY" YouTube channel.
The main character is CC (Creation Companion) - a happy, simple cartoon character who guides viewers through DIY projects.

Create engaging, family-friendly DIY tutorial scripts. Return ONLY valid JSON in this exact format:

{
    "title": "Catchy YouTube title (e.g., 'Amazing Paper Flowers in 5 Minutes!')",
    "title_screen_prompt": "Detailed prompt for colorful DIY-themed background image",
    "intro": {
        "narration": "Welcome message introducing CC and the project (2-3 sentences)",
        "image_prompt": "Prompt for intro scene showing CC excited about the project"
    },
    "materials": [
        {
            "name": "Material name",
            "image_prompt": "Detailed prompt for clean product shot of this material"
        }
    ],
    "steps": [
        {
            "step_number": 1,
            "title": "Short step title",
            "narration": "Clear instructions for this step (2-3 sentences)",
            "image_prompt": "Prompt showing CC performing this step with materials"
        }
    ],
    "completion": {
        "narration": "Celebration of the finished project and call to action (2-3 sentences)",
        "image_prompt": "Prompt showing CC proudly displaying the completed project"
    },
    "thumbnail_prompt": "Eye-catching thumbnail showing the finished project"
}

Guidelines:
- Keep steps simple and clear (3-5 steps)
- Include 3-6 materials
- Make CC the star of every scene
- Use encouraging, enthusiastic language
- All prompts should specify cartoon/illustration style"""

    payload = {
        "messages": [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": f"Create a DIY video script for: {project_query}",
            },
        ],
        "model": "grok-3-latest",
        "stream": False,
        "temperature": 0.7,
    }

    response = requests.post(XAI_API, json=payload, headers=XAI_HEADERS)
    response.raise_for_status()

    result = response.json()
    content = result["choices"][0]["message"]["content"]

    # Parse JSON from response
    if "```json" in content:
        content = content.split("```json")[1].split("```")[0]
    elif "```" in content:
        content = content.split("```")[1].split("```")[0]

    script = json.loads(content.strip())

    print(f"✅ Title: {script['title']}")
    print(f"📦 Materials: {len(script['materials'])}")
    print(f"📝 Steps: {len(script['steps'])}")

    return script


# =============================================================================
# IMAGE GENERATION
# =============================================================================


def generate_image_gemini(prompt, aspect_ratio=None):
    """Generate image using Gemini (Nano Banana)"""
    if not GEMINI_API_KEY:
        raise ValueError("GEMINI_API_KEY not set. Set it in config or environment.")

    # Add style preset to prompt
    styled_prompt = f"{prompt}, {STYLE_PRESET}" if STYLE_PRESET else prompt

    print(f"  🍌 Generating with Gemini ({aspect_ratio or GEMINI_ASPECT_RATIO})...")

    headers = {"Content-Type": "application/json"}

    # Gemini API request
    payload = {
        "contents": [{"parts": [{"text": styled_prompt}]}],
        "generationConfig": {
            "responseModalities": ["IMAGE"],
            "imageConfig": {"aspectRatio": aspect_ratio or GEMINI_ASPECT_RATIO},
        },
    }

    response = requests.post(
        f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent?key={GEMINI_API_KEY}",
        json=payload,
        headers=headers,
    )
    response.raise_for_status()

    result = response.json()

    # Extract image from response
    for candidate in result.get("candidates", []):
        for part in candidate.get("content", {}).get("parts", []):
            if "inlineData" in part:
                import base64

                image_data = part["inlineData"]["data"]
                image_bytes = base64.b64decode(image_data)
                print(f"  ✅ Generated ({len(image_bytes):,} bytes)")
                return image_bytes

    raise ValueError("No image returned from Gemini")


def generate_image_nunchaku(prompt, width=PORTRAIT_WIDTH, height=PORTRAIT_HEIGHT):
    """Generate image using Nunchaku"""
    # Enhance the prompt for better quality
    enhanced_prompt = enhance_image_prompt(prompt)

    print(f"  🎨 Generating with Nunchaku ({width}x{height})...")

    payload = {
        "prompt": enhanced_prompt,
        "width": width,
        "height": height,
        "true_cfg_scale": 1.0,
    }

    response = requests.post(
        f"{IMAGE_GEN}/generate", json=payload, headers=IMAGE_HEADERS
    )
    response.raise_for_status()

    print(f"  ✅ Generated ({len(response.content):,} bytes)")
    return response.content


def generate_image(prompt, width=PORTRAIT_WIDTH, height=PORTRAIT_HEIGHT):
    """Generate image using configured provider (Gemini or Nunchaku)"""
    if IMAGE_PROVIDER == "gemini":
        # Map dimensions to aspect ratio for Gemini
        if width == 1080 and height == 1920:
            aspect_ratio = "9:16"
        elif width == 1920 and height == 1080:
            aspect_ratio = "16:9"
        elif width == 1280 and height == 720:
            aspect_ratio = "16:9"
        else:
            aspect_ratio = "1:1"
        return generate_image_gemini(prompt, aspect_ratio)
    else:
        return generate_image_nunchaku(prompt, width, height)


def deaify_image(image_id):
    """Remove AI-generated artifacts from image to make it look more natural"""
    print(f"  🎨 De-AI-ifying image...")

    data = {
        "image_id": image_id,
        "enhance_color": str(deaify_config.get("enhance_color", 1.1)),
        "enhance_contrast": str(deaify_config.get("enhance_contrast", 1.05)),
        "noise_strength": str(deaify_config.get("noise_strength", 3)),
    }

    response = requests.post(
        f"{MEDIA_SERVER}/api/v1/utils/make-image-imperfect", data=data
    )
    response.raise_for_status()

    result = response.json()
    file_id = result.get("file_id") or result.get("id")
    print(f"  ✅ De-AI-ified - ID: {file_id}")
    return file_id


def qwen_edit_camera_move(image_bytes, camera_prompt):
    """Use Qwen Edit with multiple_angles LoRA for camera movement"""
    import base64

    print(f"  📹 Camera move: {camera_prompt[:40]}...")

    # Convert to base64 data URL
    b64_image = base64.b64encode(image_bytes).decode("utf-8")
    data_url = f"data:image/png;base64,{b64_image}"

    # Get settings from config
    qwen_config = CONFIG.get("qwen_edit", {})
    loras_config = qwen_config.get("loras", {}).get("multiple_angles", {})

    payload = {
        "prompt": camera_prompt,
        "images": [data_url],
        "lora_scale": loras_config.get("scale", camera_config.get("lora_scale", 0.8)),
        "active_loras": ["multiple_angles"],
        "num_inference_steps": camera_config.get(
            "inference_steps", qwen_config.get("inference_steps", 4)
        ),
    }

    response = requests.post(f"{IMAGE_EDIT}/edit", json=payload, headers=IMAGE_HEADERS)
    response.raise_for_status()

    print(f"  ✅ Camera move complete ({len(response.content):,} bytes)")
    return response.content


def qwen_edit_next_scene(image_bytes, scene_prompt):
    """Use Qwen Edit with next_scene LoRA for cinematic transitions"""
    import base64

    print(f"  🎬 Next scene: {scene_prompt[:40]}...")

    # Convert to base64 data URL
    b64_image = base64.b64encode(image_bytes).decode("utf-8")
    data_url = f"data:image/png;base64,{b64_image}"

    payload = {
        "prompt": f"Next Scene: {scene_prompt}",
        "images": [data_url],
        "lora_scale": 0.75,
        "active_loras": ["next_scene"],
        "num_inference_steps": 28,
    }

    response = requests.post(f"{IMAGE_EDIT}/edit", json=payload, headers=IMAGE_HEADERS)
    response.raise_for_status()

    print(f"  ✅ Scene transition complete ({len(response.content):,} bytes)")
    return response.content


def normalize_audio(audio_id):
    """Normalize audio to standard loudness level"""
    print(f"  🔊 Normalizing audio...")

    data = {"audio_id": audio_id}

    response = requests.post(
        f"{MEDIA_SERVER}/api/v1/media/music-tools/normalize-track", data=data
    )
    response.raise_for_status()

    result = response.json()
    file_id = result.get("file_id") or result.get("id")
    print(f"  ✅ Normalized - ID: {file_id}")
    return file_id


def align_script_to_audio(audio_id, script_text, mode="sentence"):
    """Align script to audio for precise caption timing

    Args:
        audio_id: ID of the audio file
        script_text: The text to align
        mode: Segmentation mode - 'word', 'sentence', 'sentence_punc', 'fixed_words', 'max_chars'

    Returns:
        dict with word_timings and segments
    """
    print(f"  ⏱️ Aligning script to audio...")

    data = {"audio_id": audio_id, "script": script_text, "mode": mode}

    response = requests.post(
        f"{MEDIA_SERVER}/api/v1/media/audio-tools/align-script", data=data
    )
    response.raise_for_status()

    result = response.json()
    print(f"  ✅ Script aligned - {len(result.get('segments', []))} segments")
    return result


def trim_audio_pauses(audio_id, script_text, pause_threshold=0.5):
    """Clean long pauses from audio based on script alignment

    Args:
        audio_id: ID of the audio file
        script_text: The script for alignment
        pause_threshold: Max pause duration in seconds (default 0.5, min 0.2)

    Returns:
        New audio file ID with trimmed pauses
    """
    print(f"  ✂️ Trimming audio pauses (threshold: {pause_threshold}s)...")

    data = {
        "audio_id": audio_id,
        "script": script_text,
        "pause_threshold": str(max(0.2, pause_threshold)),
    }

    response = requests.post(
        f"{MEDIA_SERVER}/api/v1/media/audio-tools/trim-pauses", data=data
    )
    response.raise_for_status()

    result = response.json()
    file_id = result.get("file_id") or result.get("id")
    print(f"  ✅ Pauses trimmed - ID: {file_id}")
    return file_id


def generate_kokoro_tts(text, voice="af_heart", speed=1.0):
    """Generate TTS using Kokoro and return audio file ID

    Returns:
        Audio file ID from Media Server
    """
    kokoro_config = voice_config.get("kokoro", {})

    data = {
        "text": text,
        "voice": voice or kokoro_config.get("voice", "af_heart"),
        "speed": str(speed or kokoro_config.get("speed", 1.0)),
    }

    response = requests.post(
        f"{MEDIA_SERVER}/api/v1/media/audio-tools/tts/kokoro", data=data
    )
    response.raise_for_status()

    result = response.json()
    file_id = result.get("file_id") or result.get("id")
    return file_id


def create_title_with_text(image_id, title_text):
    """Use Media Server create-thumbnail endpoint for reliable text overlay"""
    print(f"  ✏️ Adding title text: {title_text[:40]}...")

    data = {
        "title": title_text,
        "image_id": image_id,
        "title_font_size": "180",
        "text_color_hex": "#FFFF00",  # Bright yellow for colorful look
        "stroke_color_hex": "#FF0000",  # Red stroke for pop
        "stroke_width": "4",
        "padding": "100",
        "font_style": "bold",
    }

    response = requests.post(
        f"{MEDIA_SERVER}/api/v1/media/music-tools/create-thumbnail", data=data
    )
    response.raise_for_status()

    result = response.json()
    file_id = result.get("file_id") or result.get("id")
    print(f"  ✅ Title created - ID: {file_id}")
    return file_id


def generate_elevenlabs_tts(text, voice_id=None):
    """Generate TTS using ElevenLabs API"""
    if voice_id is None:
        voice_id = ELEVENLABS_VOICE_ID

    try:
        response = requests.post(
            f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}",
            headers={
                "xi-api-key": ELEVENLABS_API_KEY,
                "Content-Type": "application/json",
            },
            json={
                "text": text,
                "model_id": "eleven_monolingual_v1",
                "voice_settings": {"stability": 0.5, "similarity_boost": 0.75},
            },
        )
        if response.status_code == 200:
            print(f"  🎙️ ElevenLabs TTS generated")
            return response.content
        else:
            print(f"  ⚠️ ElevenLabs returned status {response.status_code}")
    except Exception as e:
        print(f"  ⚠️ ElevenLabs failed: {e}")

    return None


# =============================================================================
# MEDIA SERVER OPERATIONS
# =============================================================================


def upload_to_storage(raw_bytes, media_type="image", filename="file.png"):
    """Upload to Media Server storage"""
    files = {"file": (filename, BytesIO(raw_bytes), f"image/png")}
    data = {"media_type": media_type}

    response = requests.post(
        f"{MEDIA_SERVER}/api/v1/media/storage", files=files, data=data
    )
    response.raise_for_status()

    result = response.json()
    file_id = result.get("file_id") or result.get("id")
    print(f"  📤 Uploaded - ID: {file_id}")
    return file_id


def generate_captioned_video(
    background_id,
    text,
    voice="af_heart",
    use_elevenlabs=True,
    trim_pauses=None,
    pause_threshold=None,
):
    """Generate captioned video segment with colorful bottom captions

    Uses ElevenLabs by default, falls back to Kokoro on failure.
    Optionally trims pauses for better caption timing.

    Args:
        background_id: Image ID for background
        text: Narration text
        voice: Kokoro voice name (fallback)
        use_elevenlabs: Try ElevenLabs first
        trim_pauses: Remove long pauses from audio (default from config)
        pause_threshold: Max pause duration in seconds (default from config)
    """
    # Get defaults from config
    if trim_pauses is None:
        trim_pauses = audio_config.get("trim_pauses", True)
    if pause_threshold is None:
        pause_threshold = audio_config.get("pause_threshold", 0.4)

    # Try ElevenLabs first
    audio_id = None
    if use_elevenlabs:
        audio_bytes = generate_elevenlabs_tts(text)
        if audio_bytes:
            # Upload ElevenLabs audio to storage
            audio_id = upload_to_storage(audio_bytes, "audio", "elevenlabs_tts.mp3")
            print(f"  🎙️ Using ElevenLabs voice")

            # Trim pauses for better timing
            if trim_pauses and audio_id:
                try:
                    audio_id = trim_audio_pauses(audio_id, text, pause_threshold)
                except Exception as e:
                    print(f"  ⚠️ Pause trimming failed: {e}")

    # Determine Ken Burns effect based on config
    kb_intensity = ken_burns_config.get("intensity", "medium")
    image_effect_map = {
        "none": "still",
        "subtle": "pan",
        "medium": "ken_burns",
        "strong": "ken_burns",
    }
    image_effect = (
        image_effect_map.get(kb_intensity, "ken_burns")
        if ken_burns_config.get("enabled", True)
        else "still"
    )

    data = {
        "background_id": background_id,
        "text": text,
        "width": str(PORTRAIT_WIDTH),
        "height": str(PORTRAIT_HEIGHT),
        # Image animation effect from config
        "image_effect": image_effect,
        # Colorful captions from config
        "caption_config_subtitle_position": caption_config.get("position", "bottom"),
        "caption_config_font_color": caption_config.get("font_color", "#FFFF00"),
        "caption_config_stroke_color": caption_config.get("stroke_color", "#FF00FF"),
        "caption_config_stroke_size": str(caption_config.get("stroke_size", 5)),
        "caption_config_font_size": str(caption_config.get("font_size", 100)),
        "caption_config_font_bold": str(caption_config.get("font_bold", True)).lower(),
        "caption_config_font_name": caption_config.get("font_name", "Arial"),
        "caption_config_shadow_color": caption_config.get("shadow_color", "#000000"),
        "caption_config_shadow_blur": str(caption_config.get("shadow_blur", 10)),
        "caption_config_shadow_transparency": str(
            caption_config.get("shadow_transparency", 0.6)
        ),
    }

    # Use pre-generated audio or fall back to Kokoro
    if audio_id:
        data["audio_id"] = audio_id
    else:
        data["kokoro_voice"] = voice
        print(f"  🔊 Using Kokoro voice: {voice}")

    response = requests.post(
        f"{MEDIA_SERVER}/api/v1/media/video-tools/generate/tts-captioned-video",
        data=data,
    )
    response.raise_for_status()

    result = response.json()
    job_id = result.get("id") or result.get("file_id")
    print(f"  🎥 Video job - ID: {job_id}")
    return job_id


def poll_status(file_id, max_attempts=120, delay=5):
    """Poll until job completes"""
    for attempt in range(max_attempts):
        response = requests.get(f"{MEDIA_SERVER}/api/v1/media/storage/{file_id}/status")
        response.raise_for_status()

        status_data = response.json()
        status = status_data.get("status")

        if status in ("completed", "ready"):
            print(f"  ✅ Completed!")
            return status_data
        elif status == "failed":
            raise Exception(f"Job failed: {status_data}")

        if attempt % 6 == 0:
            print(f"  ⏳ Status: {status} (attempt {attempt + 1})")
        time.sleep(delay)

    raise Exception("Job timed out")


def merge_videos(video_ids, background_music_id=None, music_volume=0.3):
    """Merge multiple videos with optional background music"""
    print(f"\n🔗 Merging {len(video_ids)} video segments...")

    data = {"video_ids": ",".join(video_ids)}

    if background_music_id:
        data["background_music_id"] = background_music_id
        data["background_music_volume"] = str(music_volume)
        print(f"  🎵 Adding background music (volume: {music_volume})")

    response = requests.post(
        f"{MEDIA_SERVER}/api/v1/media/video-tools/merge", data=data
    )
    response.raise_for_status()

    result = response.json()
    job_id = result.get("id") or result.get("file_id")
    print(f"  🎬 Merge job - ID: {job_id}")
    return job_id


def download_file(file_id, output_path):
    """Download completed file"""
    response = requests.get(f"{MEDIA_SERVER}/api/v1/media/storage/{file_id}")
    response.raise_for_status()

    with open(output_path, "wb") as f:
        f.write(response.content)

    print(f"  💾 Downloaded: {output_path} ({len(response.content):,} bytes)")
    return output_path


# =============================================================================
# MAIN PRODUCTION PIPELINE
# =============================================================================


def produce_diy_video(
    project_query,
    output_dir="/home/bc/claudeskills/media-production/output",
    background_music_id=None,
    use_camera_moves=True,
    deaify_images=True,
):
    """Complete DIY video production pipeline

    Args:
        project_query: Description of the DIY project
        output_dir: Directory to save output files
        background_music_id: Optional music file ID for background
        use_camera_moves: Use Qwen Edit for dynamic camera angles
        deaify_images: Apply de-AI-ify filter to images
    """
    os.makedirs(output_dir, exist_ok=True)

    print("\n" + "=" * 70)
    print("🛠️  CREATION COMPANIONS DIY VIDEO PRODUCTION")
    print("=" * 70)

    # Step 1: Generate script with Grok
    script = generate_diy_script(project_query)
    title = script["title"]

    video_ids = []

    # Step 2: Title Screen with Media Server text overlay
    print("\n" + "-" * 70)
    print("📺 TITLE SCREEN")
    print("-" * 70)

    # Generate title background
    title_bg_prompt = (
        f"{script['title_screen_prompt']}, {CC_CHARACTER} visible, colorful DIY theme"
    )
    title_bg = generate_image(title_bg_prompt)

    # Upload background first, then add title text using Media Server
    title_bg_id = upload_to_storage(title_bg, "image", "title_bg.png")
    title_id = create_title_with_text(title_bg_id, title)

    # Create short title video
    title_video_id = generate_captioned_video(title_id, f"{title}!", voice="af_heart")
    poll_status(title_video_id)
    video_ids.append(title_video_id)

    # Step 3: Intro Scene
    print("\n" + "-" * 70)
    print("👋 INTRO")
    print("-" * 70)

    intro_prompt = f"{script['intro']['image_prompt']}, {CC_CHARACTER}"
    intro_bytes = generate_image(intro_prompt)
    intro_id = upload_to_storage(intro_bytes, "image", "intro.png")

    # De-AI-ify for more natural look
    if deaify_images:
        try:
            intro_id = deaify_image(intro_id)
        except Exception as e:
            print(f"  ⚠️ De-AI-ify failed, using original: {e}")

    intro_video_id = generate_captioned_video(
        intro_id, script["intro"]["narration"], voice="af_heart"
    )
    poll_status(intro_video_id)
    video_ids.append(intro_video_id)

    # Step 4: Materials List
    print("\n" + "-" * 70)
    print("📦 MATERIALS")
    print("-" * 70)

    for i, material in enumerate(script["materials"], 1):
        print(f"\n  Material {i}/{len(script['materials'])}: {material['name']}")

        mat_prompt = f"{material['image_prompt']}, clean product photography style, white background, {material['name']}"
        mat_bytes = generate_image(mat_prompt)
        mat_id = upload_to_storage(mat_bytes, "image", f"material_{i}.png")

        mat_video_id = generate_captioned_video(
            mat_id, material["name"], voice="af_heart"
        )
        poll_status(mat_video_id)
        video_ids.append(mat_video_id)

    # Step 5: Step-by-step Process
    print("\n" + "-" * 70)
    print("📝 STEPS")
    print("-" * 70)

    for step in script["steps"]:
        print(f"\n  Step {step['step_number']}: {step['title']}")

        step_prompt = (
            f"{step['image_prompt']}, {CC_CHARACTER} demonstrating, tutorial style"
        )
        step_bytes = generate_image(step_prompt)

        # Apply camera move for dynamic content
        if (
            use_camera_moves
            and step["step_number"] > 1
            and camera_config.get("enabled", True)
        ):
            # Get camera moves from config or use defaults
            camera_moves_list = camera_config.get(
                "moves",
                [
                    "Zoom in slightly on the hands working",
                    "Rotate camera 15 degrees to the right",
                    "Move camera closer to show detail",
                    "Pan slightly left to show workspace",
                ],
            )
            move_idx = (step["step_number"] - 1) % len(camera_moves_list)
            try:
                step_bytes = qwen_edit_camera_move(
                    step_bytes, camera_moves_list[move_idx]
                )
            except Exception as e:
                print(f"  ⚠️ Camera move failed, using original: {e}")

        step_id = upload_to_storage(
            step_bytes, "image", f"step_{step['step_number']}.png"
        )

        # De-AI-ify for more natural look
        if deaify_images:
            try:
                step_id = deaify_image(step_id)
            except Exception as e:
                print(f"  ⚠️ De-AI-ify failed, using original: {e}")

        step_video_id = generate_captioned_video(
            step_id,
            f"Step {step['step_number']}: {step['narration']}",
            voice="af_heart",
        )
        poll_status(step_video_id)
        video_ids.append(step_video_id)

    # Step 6: Completion Scene
    print("\n" + "-" * 70)
    print("🎉 COMPLETION")
    print("-" * 70)

    completion_prompt = (
        f"{script['completion']['image_prompt']}, {CC_CHARACTER} celebrating, confetti"
    )
    completion_bytes = generate_image(completion_prompt)
    completion_id = upload_to_storage(completion_bytes, "image", "completion.png")

    completion_video_id = generate_captioned_video(
        completion_id, script["completion"]["narration"], voice="af_heart"
    )
    poll_status(completion_video_id)
    video_ids.append(completion_video_id)

    # Step 7: Generate Thumbnail
    print("\n" + "-" * 70)
    print("🖼️ THUMBNAIL")
    print("-" * 70)

    thumb_prompt = f"{script['thumbnail_prompt']}, {CC_CHARACTER}, YouTube thumbnail style, eye-catching"
    thumbnail_bytes = generate_image(thumb_prompt, width=1280, height=720)
    thumb_path = os.path.join(output_dir, "thumbnail.png")
    with open(thumb_path, "wb") as f:
        f.write(thumbnail_bytes)
    print(f"  💾 Saved: {thumb_path}")

    # Step 8: Merge all videos
    print("\n" + "=" * 70)
    print("🎬 FINAL ASSEMBLY")
    print("=" * 70)

    final_job_id = merge_videos(video_ids, background_music_id=background_music_id)
    poll_status(final_job_id)

    # Download final video
    safe_title = "".join(c if c.isalnum() or c in " -_" else "" for c in title)
    video_path = os.path.join(output_dir, f"{safe_title}.mp4")
    download_file(final_job_id, video_path)

    # Upload to Cloudinary
    cloudinary_url = None
    thumb_cloudinary_url = None
    try:
        print("\n☁️ Uploading to Cloudinary...")
        cloudinary.config(**CLOUDINARY_CONFIG)

        video_result = cloudinary.uploader.upload(
            video_path,
            folder="creation-companions-diy",
            public_id=f"diy_{safe_title}",
            resource_type="video",
        )
        cloudinary_url = video_result.get("secure_url")
        print(f"  ✅ Video: {cloudinary_url}")

        thumb_result = cloudinary.uploader.upload(
            thumb_path,
            folder="creation-companions-diy/thumbnails",
            public_id=f"thumb_{safe_title}",
            resource_type="image",
        )
        thumb_cloudinary_url = thumb_result.get("secure_url")
        print(f"  ✅ Thumbnail: {thumb_cloudinary_url}")

    except Exception as e:
        print(f"  ❌ Cloudinary upload error: {e}")

    # Summary
    print("\n" + "=" * 70)
    print("🎉 PRODUCTION COMPLETE!")
    print("=" * 70)
    print(f"📺 Title: {title}")
    print(f"🎥 Video: {video_path}")
    print(f"🖼️ Thumbnail: {thumb_path}")
    print(f"📦 Materials: {len(script['materials'])}")
    print(f"📝 Steps: {len(script['steps'])}")
    if cloudinary_url:
        print(f"☁️ Cloudinary Video: {cloudinary_url}")
    if thumb_cloudinary_url:
        print(f"☁️ Cloudinary Thumbnail: {thumb_cloudinary_url}")

    return {
        "title": title,
        "video_path": video_path,
        "thumbnail_path": thumb_path,
        "cloudinary_url": cloudinary_url,
        "cloudinary_thumbnail": thumb_cloudinary_url,
        "script": script,
    }


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Create DIY tutorial videos for Creation Companions DIY",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python diy_video.py "Paper airplane that flies far"
  python diy_video.py "Origami star" --music-id abc123
  python diy_video.py "Friendship bracelet" --no-camera-moves --no-deaify
        """,
    )
    parser.add_argument(
        "query",
        nargs="*",
        default=["Paper airplane that flies far"],
        help="DIY project description",
    )
    parser.add_argument(
        "--mode",
        choices=["kid", "adult"],
        default=None,
        help="Content mode: kid (cartoon) or adult (realistic)",
    )
    parser.add_argument(
        "--provider",
        choices=["gemini", "nunchaku"],
        default=None,
        help="Image generation provider",
    )
    parser.add_argument(
        "--music-id",
        dest="music_id",
        default=None,
        help="Background music file ID from Media Server",
    )
    parser.add_argument(
        "--no-camera-moves",
        dest="camera_moves",
        action="store_false",
        help="Disable Qwen Edit camera movements",
    )
    parser.add_argument(
        "--no-deaify",
        dest="deaify",
        action="store_false",
        help="Disable de-AI-ify image processing",
    )
    parser.add_argument(
        "--output-dir",
        dest="output_dir",
        default="/home/bc/claudeskills/media-production/output",
        help="Output directory for generated files",
    )

    args = parser.parse_args()
    query = " ".join(args.query)

    # Override global config if CLI args provided
    # Note: These are module-level variables that need updating
    import diy_video

    if args.mode:
        diy_video.CONTENT_MODE = args.mode
        # Update character based on mode
        if args.mode == "adult":
            diy_video.CC_CHARACTER = character_config.get(
                "adult_description", CC_CHARACTER
            )
        else:
            diy_video.CC_CHARACTER = character_config.get(
                "kid_description", CC_CHARACTER
            )
        # Update style preset
        diy_video.STYLE_PRESET = gemini_config.get("styles", {}).get(args.mode, "")
        print(f"📋 Mode: {args.mode}")

    if args.provider:
        diy_video.IMAGE_PROVIDER = args.provider
        print(f"🖼️ Provider: {args.provider}")

    # Use updated values
    if args.mode:
        CONTENT_MODE = args.mode
        # Update character based on mode
        if CONTENT_MODE == "adult":
            CC_CHARACTER = character_config.get("adult_description", CC_CHARACTER)
        else:
            CC_CHARACTER = character_config.get("kid_description", CC_CHARACTER)
        # Update style preset
        STYLE_PRESET = gemini_config.get("styles", {}).get(CONTENT_MODE, "")
        print(f"📋 Mode: {CONTENT_MODE}")

    if args.provider:
        IMAGE_PROVIDER = args.provider
        print(f"🖼️ Provider: {IMAGE_PROVIDER}")

    try:
        result = produce_diy_video(
            query,
            output_dir=args.output_dir,
            background_music_id=args.music_id,
            use_camera_moves=args.camera_moves,
            deaify_images=args.deaify,
        )
        print(f"\n🎉 Success! Video ready: {result['video_path']}")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback

        traceback.print_exc()
