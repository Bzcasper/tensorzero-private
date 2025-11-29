You are an expert software architect and full-stack developer specializing in TensorZero + fast-agent integration for automated media production workflows (e.g., generating 3-minute DIY video scripts, images, and videos from a project description).

## CRITICAL ARCHITECTURE RULES (MANDATORY)
- **Fast-agent agents contain ZERO prompts**: Only workflow logic, parameter passing, state management, HTTP calls, error retries (max 3 attempts with exponential backoff), and logging.
- **TensorZero handles ALL prompts/schemas/models**: Store in `templates/` as MiniJinja (.minijinja) system templates + JSON Schema (draft-07) user schemas. Use HTTP POST to `http://localhost:3000/inference` with `{"function_name": "...", "params": {...}}`. Support streaming (`stream: true`) and non-streaming.
- **Communication**: `tensorzero_client.py` abstracts calls with async `call_tensorzero(function_name: str, params: dict, stream: bool=False) -> dict | AsyncIterator`.
- **Media Server**: Use `media_server_client.py` for async calls to `https://2281a5a294754c19f8c9e2df0be013fb-bobby-casper-4235.aiagentsaz.com` endpoints (e.g., `/generate_images`, `/generate_audio`, `/assemble_video`).
- **Models**: Prioritize Cerebras > SambaNova > Groq > Mistral > DeepSeek > Qwen > Gemini Pro > Grok > OpenRouter fallbacks in `tensorzero.toml`.
- **State Management**: Fast-agent uses Pydantic models for workflow state (e.g., `WorkflowState` with `project_description: str`, `script: dict`, `image_ids: list[str]`, `video_id: str`).
- **Observability**: Log all steps to stdout (JSON format: `{"step": "...", "params": ..., "result": ...}`).

## TARGET WORKFLOW (Full End-to-End for Input: project_description="Make a paper airplane", content_mode="kid", target_duration=180)
1. **script_generator**: Generate script dict `{"title": str, "scenes": list[dict{"dialogue": str, "visual_desc": str, "duration_sec": int}], "total_duration_sec": int}`.
2. **prompt_enhancer**: Enhance each scene's `visual_desc` into detailed image prompt dict `{"enhanced_prompts": list[str]}` (e.g., add style: "cartoonish for kids, 1080p").
3. **image_gen**: Call Media Server `/generate_images` with enhanced prompts → list of image URLs/IDs.
4. **audio_gen**: Call Media Server `/generate_audio` with script dialogues → list of audio URLs/IDs (TTS, kid-friendly voice).
5. **video_assembly**: Call Media Server `/assemble_video` with images, audios, timings → video_id/URL.
6. **quality_evaluator**: Score final video (0-10) on criteria: engagement, clarity, duration accuracy, kid-appropriateness → `{"score": float, "feedback": str, "pass": bool >=8}`. Retry workflow if fail.

## TECHNICAL SPECS
- **TensorZero config/tensorzero.toml**: Extend existing with new functions `script_generator`, `prompt_enhancer`, `quality_evaluator`. Each has `type="json"`, `system_template="templates/..._system.minijinja"`, `user_schema="templates/..._user_schema.json"`. Variants: `.cerebras`, `.sambanova`, etc. (model-specific).
- **MiniJinja**: Use `{{ var }}` for params (e.g., `{{ project_description }}`). No Python Jinja2.
- **JSON Schemas**: Strict draft-07, required fields, examples, min/max lengths.
- **Fast-agent**: `@fast.agent("video_producer")` orchestrates sequential workflow with parallel image/audio gen.
- **Error Handling**: Catch HTTP 5xx → retry; 4xx → fail workflow.
- **Deployment**: `docker-compose.yml` with TensorZero (port 3000), fast-agent, Media Server proxy.

## EXACT DELIVERABLES (Output ONLY these files with FULL, COPY-PASTABLE CODE)
1. **`config/tensorzero.toml`** (complete config with 3 new functions + variants for 9 models).
2. **`templates/script_generator_system.minijinja`** (400-600 tokens, expert DIY script writer for {{content_mode}}, output JSON schema-compliant).
3. **`templates/script_generator_user_schema.json`** (params: `{"project_description": {"type":"string"}, "content_mode": {"type":"string", "enum":["kid","adult"]}, "target_duration": {"type":"int", "minimum":60, "maximum":600}}` + output schema).
4. **`templates/prompt_enhancer_system.minijinja`** (enhance visual_desc for image gen: add lighting, style, composition).
5. **`templates/prompt_enhancer_user_schema.json`** (input: `{"scenes": [{"visual_desc":str}]}` → output `{"enhanced_prompts": [str]}`).
6. **`templates/quality_evaluator_system.minijinja`** (rubric-based scoring for video_url).
7. **`templates/quality_evaluator_user_schema.json`** (input: `{"video_url":str, "script":dict, "target_duration":int}`).
8. **`tensorzero_client.py`** (async client with auth if needed, streaming support, error handling).
9. **`media_server_client.py`** (async functions: `generate_images(prompts:list[str])`, `generate_audio(dialogues:list[str])`, `assemble_video(images:list[str], audios:list[str], timings:list[int])`).
10. **`agent.py`** (full `@fast.agent("video_producer")` workflow for input params, returns `{"video_url":str, "score":float}`).
11. **`docker-compose.yml`** (services: tensorzero, fast-agent, nginx proxy to Media Server).

Test with: `project_description="Make a paper airplane", content_mode="kid", target_duration=180`. Ensure total output is production-ready, modular, and follows 100% architecture separation.

# Build/Lint/Test Commands

- **Rust**: `cargo check` for verification, `cargo fmt` for formatting, `cargo clippy --all-targets --all-features -- -D warnings` for linting, `cargo test-unit-fast` for unit tests
- **Python**: Use `uv` for dependencies. Run tests with `pytest` or `uv run pytest`
- **Node.js**: `pnpm run lint` and `pnpm run typecheck` for router-ffi package
- **Single test**: `cargo nextest run <test_name>` or `cargo nextest run --filter <pattern>`

# Code Style Guidelines

- **Rust**: Strict clippy rules (no unsafe code, no unwrap/unwrap_used, etc.). Use `use crate::...` imports at top of files. Avoid long inline crate paths
- **Python**: Use `ruff` for linting/formatting. Follow standard Python conventions
- **TypeScript/JavaScript**: Use `eslint` and `prettier`. Prefer TypeScript with strict typing
- **General**: Run pre-commit hooks before committing. Use `rg` for searching codebase
- **Error handling**: Use `anyhow` for errors, avoid unwrap/panic in production code</content>
  <parameter name="filePath">/home/trapgod/projects/tensorzero/tensorzero-deploy/AGENTS.md
