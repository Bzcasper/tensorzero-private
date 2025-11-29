# TensorZero System Overview

## Architecture

The system is a multi-agent content generation pipeline orchestrated by a TensorZero Gateway. It uses a microservices approach with Docker Compose.

### Components

1. **TensorZero Gateway**: The central router and model manager.
    * **Port**: 3000
    * **Config**: `tensorzero-deploy/config/tensorzero.toml`
    * **Observability**: Connects to ClickHouse for metrics/logs.
2. **ClickHouse**: Data warehouse for TensorZero observability.
    * **Port**: 8123 (HTTP), 9000 (Native)
    * **Volume**: `tensorzero_clickhouse_data`

## Agents

The system defines four specialized agents (as described in `AGENTS.md`):

1. **Manager Agent**:
    * **Role**: Orchestrator.
    * **Responsibility**: Breaks down topics into subtasks, assigns work to other agents, and ensures the final output meets quality standards.
2. **Writer Agent**:
    * **Role**: Content Creator.
    * **Responsibility**: Drafts the initial content based on the Manager's outline.
3. **Editor Agent**:
    * **Role**: Quality Assurance.
    * **Responsibility**: Refines the draft, checking for clarity, tone, and grammar.
4. **Publisher Agent**:
    * **Role**: Final Output.
    * **Responsibility**: Formats the content for the target platform (e.g., blog, social media) and handles "publishing" (simulated or API).

## Model Configuration (`tensorzero.toml`)

The Gateway is configured to route requests to a diverse set of LLM providers, utilizing both free and paid tiers, with advanced routing strategies.

### Defined Models

*   **`cerebras_llama70b`**: Uses Cerebras (`llama3.3-70b`) - Instant speed.
*   **`sambanova_405b`**: Uses SambaNova (`Meta-Llama-3.1-405B-Instruct`) - Massive reasoning.
*   **`groq_llama33`**: Uses Groq (`llama-3.3-70b-versatile`) - Reliable speed.
*   **`mistral_large`**: Uses Mistral (`mistral-large-latest`) - High quality.
*   **`deepseek_v3`**: Uses DeepSeek (`deepseek-chat`) - Best value/logic.
*   **`qwen_25_together`**: Uses Together AI (`Qwen/Qwen2.5-72B-Instruct-Turbo`) - SOTA open source.
*   **`gemini_pro`**: Uses Google AI Studio (`gemini-1.5-pro`) - High context (Paid).
*   **`grok_2`**: Uses xAI (`grok-2-1212`) - Reasoning & knowledge (Paid).
*   **`liquid_lfm_40b`**: Uses OpenRouter (`liquid/lfm-40b`) - Free MoE.
*   **`gemini_2_flash_free`**: Uses OpenRouter (`google/gemini-2.0-flash-exp:free`) - Free.

### Functions

*   **`chat`**: The main interface for agent communication.
    *   **Variants**: `cerebras`, `sambanova`, `groq`, `mistral`, `deepseek`, `gemini_pro`, `grok`, `free_liquid`, `free_gemini`.
    *   **Experimentation**: Weighted random routing (15% for Cerebras/SambaNova, 10% for others).
*   **`best_of_n_chat`**: Quality via Rejection Sampling.
    *   **Mechanism**: Generates 3 drafts with Cerebras (Llama 3.3) and uses SambaNova (Llama 405B) to select the best one.
*   **`mixture_of_n_chat`**: Quality via Fusion.
    *   **Mechanism**: Generates 3 drafts (Mistral, DeepSeek, Qwen) and uses Gemini 1.5 Pro to fuse them into one answer.

## Setup & Deployment

*   **Prerequisites**: Docker, Docker Compose, API keys (CEREBRAS, SAMBANOVA, GROQ, MISTRAL, DEEPSEEK, TOGETHER, GEMINI, GROK, OPENROUTER).
*   **Start**: `docker-compose -f tensorzero-deploy/docker-compose.yml up -d`
*   **Environment**: API keys are passed via environment variables to the Gateway container.
