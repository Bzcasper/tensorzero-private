# https://fast-agent.ai/ llms-full.txt

## Fast Agent Setup
[Skip to content](https://fast-agent.ai/#welcome-to-fast-agent)

# welcome to fast-agent

- **Set up in 5 minutes**


* * *


Simple installation and setup with [`uv`](https://docs.astral.sh/uv/) to be up and running in minutes. Out-of-the box examples of Agents, Workflows and MCP Usage.

- **Agent Skills**


* * *


Support for [Agent Skills](https://www.anthropic.com/engineering/equipping-agents-for-the-real-world-with-agent-skills) to define context efficient behaviour for your Agents. Read the documentation [here](https://fast-agent.ai/agents/skills/).

- **New - Elicitation Quickstart Guide**


* * *


Get started with MCP Elicitations for User Interaction.
[Try now](https://fast-agent.ai/mcp/elicitations/)

- **Comprehensive Test Suite**


* * *


Extensive validated [model support](https://fast-agent.ai/models/llm_providers/) for Structured Outputs, Tool Calling and Multimodal capabilities

- **MCP Feature Support**


* * *


Full MCP feature support including Elication and Sampling and advanced transport diagnostics

[Reference](https://fast-agent.ai/mcp/)

- **Agent Developer Friendly**


* * *


Lightweight deployment - in-built Echo and Playback LLMs allow robust agent application testing


## Getting Started

**fast-agent** lets you create and interact with sophisticated Agents and Workflows in minutes. It's multi-modal - supporting Images and PDFs in Prompts, Resources and MCP Tool Call results.

Prebuilt agents and examples implementing the patterns in Anthropic's [building effective agents](https://www.anthropic.com/engineering/building-effective-agents) paper get you building valuable applications quickly. Seamlessly use MCP Servers with your agents, or host your agents as MCP Servers.

- `uv tool install fast-agent-mcp` \- Install fast-agent.
- `fast-agent go` \- Start an interactive session...
- `fast-agent go --url https://hf.co/mcp` \- ...with a remote MCP.
- `fast-agent setup` \- Create Agent and Configuration files.
- `uv run agent.py` \- Run your first Agent
- `fast-agent quickstart workflow` \- Create Agent workflow examples

![](https://fast-agent.ai/welcome_small.png)

Back to top

## Fast Agent Overview
https://fast-agent.ai/2025-11-23https://fast-agent.ai/getting\_started/2025-11-23https://fast-agent.ai/acp/2025-11-23https://fast-agent.ai/agents/2025-11-23https://fast-agent.ai/agents/defining/2025-11-23https://fast-agent.ai/agents/instructions/2025-11-23https://fast-agent.ai/agents/prompting/2025-11-23https://fast-agent.ai/agents/running/2025-11-23https://fast-agent.ai/agents/skills/2025-11-23https://fast-agent.ai/getting\_started/installation/2025-11-23https://fast-agent.ai/mcp/2025-11-23https://fast-agent.ai/mcp/elicitations/2025-11-23https://fast-agent.ai/mcp/mcp-oauth/2025-11-23https://fast-agent.ai/mcp/mcp-server/2025-11-23https://fast-agent.ai/mcp/mcp-ui/2025-11-23https://fast-agent.ai/mcp/mcp\_display/2025-11-23https://fast-agent.ai/mcp/openai-apps-sdk/2025-11-23https://fast-agent.ai/mcp/resources/2025-11-23https://fast-agent.ai/mcp/state\_transfer/2025-11-23https://fast-agent.ai/mcp/types/2025-11-23https://fast-agent.ai/models/2025-11-23https://fast-agent.ai/models/internal\_models/2025-11-23https://fast-agent.ai/models/llm\_providers/2025-11-23https://fast-agent.ai/ref/azure-config/2025-11-23https://fast-agent.ai/ref/class\_reference/2025-11-23https://fast-agent.ai/ref/cmd\_switches/2025-11-23https://fast-agent.ai/ref/config\_file/2025-11-23https://fast-agent.ai/ref/go\_command/2025-11-23https://fast-agent.ai/ref/open\_telemetry/2025-11-23https://fast-agent.ai/welcome/2025-11-23

## LLM Providers Overview
[Skip to content](https://fast-agent.ai/models/llm_providers/#common-configuration-format)

# LLM Providers

For each model provider, you can configure parameters either through environment variables or in your `fastagent.config.yaml` file.

Be sure to run `fast-agent check` to troubleshoot API Key issues:

![Key Check](https://fast-agent.ai/models/check.png)

## Common Configuration Format

In your `fastagent.config.yaml`:

```
<provider>:
  api_key: "your_api_key" # Override with API_KEY env var
  base_url: "https://api.example.com" # Base URL for API calls
```

## Anthropic

Anthropic models support Text, Vision and PDF content.

**YAML Configuration:**

```
anthropic:
  api_key: "your_anthropic_key" # Required
  base_url: "https://api.anthropic.com/v1" # Default, only include if required
```

**Environment Variables:**

- `ANTHROPIC_API_KEY`: Your Anthropic API key
- `ANTHROPIC_BASE_URL`: Override the API endpoint

**Model Name Aliases:**

| Model Alias | Maps to | Model Alias | Maps to |
| --- | --- | --- | --- |
| `claude` | `claude-sonnet-4-0` | `haiku` | `claude-3-5-haiku-latest` |
| `sonnet` | `claude-sonnet-4-0` | `haiku3` | `claude-3-haiku-20240307` |
| `sonnet35` | `claude-3-5-sonnet-latest` | `haiku35` | `claude-3-5-haiku-latest` |
| `sonnet37` | `claude-3-7-sonnet-latest` | `opus` | `claude-opus-4-1` |
| `opus3` | `claude-3-opus-latest` |  |  |

## OpenAI

**fast-agent** supports OpenAI `gpt-5` series, `gpt-4.1` series, `o1-preview`, `o1` and `o3-mini` models. Arbitrary model names are supported with `openai.<model_name>`. Supported modalities are model-dependent, check the [OpenAI Models Page](https://platform.openai.com/docs/models) for the latest information.

For reasoning models, you can specify `low`, `medium`, or `high` effort as follows:

```
fast-agent --model o3-mini.medium
fast-agent --model gpt-5.high
```

`gpt-5` also supports a `minimal` reasoning effort.

Structured outputs use the OpenAI API Structured Outputs feature.

**YAML Configuration:**

```
openai:
  api_key: "your_openai_key" # Default
  base_url: "https://api.openai.com/v1" # Default, only include if required
```

**Environment Variables:**

- `OPENAI_API_KEY`: Your OpenAI API key
- `OPENAI_BASE_URL`: Override the API endpoint

**Model Name Aliases:**

| Model Alias | Maps to | Model Alias | Maps to |
| --- | --- | --- | --- |
| `gpt-4o` | `gpt-4o` | `gpt-4.1` | `gpt-4.1` |
| `gpt-4o-mini` | `gpt-4o-mini` | `gpt-4.1-mini` | `gpt-4.1-mini` |
| `o1` | `o1` | `gpt-4.1-nano` | `gpt-4.1-nano` |
| `o1-mini` | `o1-mini` | `o1-preview` | `o1-preview` |
| `o3-mini` | `o3-mini` | `o3` |  |
| `gpt-5` | `gpt-5` | `gpt-5-mini` | `gpt-5-mini` |
| `gpt-5-nano` | `gpt-5-nano` |  |  |

## Hugging Face

Use models via [Hugging Face Inference Providers](https://huggingface.co/docs/inference-providers/en/index).

```
hf:
  api_key: "${HF_TOKEN}"
  base_url: "https://router.huggingface.co/v1" # Default
  default_provider: # Optional: groq, fireworks-ai, cerebras, etc.
```

**Environment Variables:**

- `HF_TOKEN` \- HuggingFace authentication token (required)
- `HF_DEFAULT_PROVIDER` \- Default inference provider (optional)

### Model Syntax

Use `hf.<model_name>[:provider]` to specify models. If no provider is specified, the model is auto-routed.

**Examples:**

```
# Auto-routed
fast-agent --model hf.openai/gpt-oss-120b
fast-agent --model hf.moonshotai/kimi-k2-instruct-0905

# Explicit provider
fast-agent --model hf.moonshotai/kimi-k2-instruct-0905:groq
fast-agent --model hf.deepseek-ai/deepseek-v3.1:fireworks-ai
```

### Model Aliases

Aliased models are verified and tested to work with Structured Outputs and Tool Use. Functionality may vary between providers, or be clamped in some situations.

| Alias | Maps to |
| --- | --- |
| `kimithink` | `hf.moonshotai/Kimi-K2-Thinking:together` |
| `kimi` | `hf.moonshotai/Kimi-K2-Instruct-0905` |
| `gpt-oss` | `hf.openai/gpt-oss-120b` |
| `gpt-oss-20b` | `hf.openai/gpt-oss-20b` |
| `glm` | `hf.zai-org/GLM-4.6` |
| `qwen3` | `hf.Qwen/Qwen3-Next-80B-A3B-Instruct` |
| `deepseek31` | `hf.deepseek-ai/DeepSeek-V3.1` |
| `minimax` | `hf.MiniMaxAI/MiniMax-M2` |

**Using Aliases:**

```
fast-agent --model kimi
fast-agent --model deepseek31
fast-agent --model kimi:together # provider can be specified with alias
```

### MCP Server Connections

`HF_TOKEN` is **automatically** applied when connecting to HuggingFace MCP servers.

**Supported domains:**

- `hf.co` / `huggingface.co` \- Uses `Authorization: Bearer {HF_TOKEN}`
- `*.hf.space` \- Uses `X-HF-Authorization: Bearer {HF_TOKEN}`

**Examples:**

```
# fastagent.config.yaml
mcp:
  servers:
    huggingface:
      url: "https://huggingface.co/mcp"
      # HF_TOKEN automatically applied!
```

```
# Command line - HF_TOKEN automatically applied
fast-agent --model kimi --url https://hf.co/mcp
fast-agent --url https://my-space.hf.space/mcp
```

## Azure OpenAI

### ⚠️ Check Model and Feature Availability by Region

Before deploying an LLM model in Azure, **always check the official Azure documentation to verify that the required model and capabilities (vision, audio, etc.) are available in your region**. Availability varies by region and by feature. Use the links below to confirm support for your use case:

**Key Capabilities and Official Documentation:**

- **General model list & region availability:** [Azure OpenAI Service models – Region availability (Microsoft Learn)](https://learn.microsoft.com/en-us/azure/ai-services/openai/concepts/models?utm_source=chatgpt.com)
- **Vision (GPT-4 Turbo with Vision, GPT-4o, o1, etc.):** [How-to: GPT with Vision (Microsoft Learn)](https://learn.microsoft.com/en-us/azure/ai-services/openai/how-to/gpt-with-vision?utm_source=chatgpt.com)
- **Audio / Whisper:** [The Whisper model from OpenAI (Microsoft Learn)](https://learn.microsoft.com/en-us/azure/ai-services/speech-service/whisper-overview?utm_source=chatgpt.com) [Audio concepts in Azure OpenAI (Microsoft Learn)](https://learn.microsoft.com/en-us/azure/ai-services/openai/concepts/audio?utm_source=chatgpt.com)
- **PDF / Documents:** [Azure AI Foundry feature availability across clouds regions (Microsoft Learn)](https://learn.microsoft.com/en-us/azure/ai-foundry/reference/region-support?utm_source=chatgpt.com)

**Summary:**

- **Vision (multimodal):** Models like GPT-4 Turbo with Vision, GPT-4o, o1, etc. are only available in certain regions. In the Azure Portal, the "Model deployments" → "Add deployment" tab lists only those available in your region. See the linked guide for input limits and JSON output.
- **Audio / Whisper:** There are two options: (1) Azure OpenAI (same `/audio/*` routes as OpenAI, limited regions), and (2) Azure AI Speech (more regions, different billing). See the links for region tables.
- **PDF / Documents:** Azure OpenAI does not natively process PDFs. Use [Azure AI Document Intelligence](https://learn.microsoft.com/en-us/azure/ai-services/form-recognizer/) or [Azure AI Search](https://learn.microsoft.com/en-us/azure/search/) for document processing. The AI Foundry table shows where each feature is available.

**Conclusion:** Before deploying, verify that your Azure resource's region supports the required model and features. If not, create the resource in a supported region or wait for general availability.

Azure OpenAI provides all the capabilities of OpenAI models within Azure's secure and compliant cloud environment. fast-agent supports three authentication methods:

1. Using `resource_name` and `api_key` (standard method)
2. Using `base_url` and `api_key` (for custom endpoints or sovereign clouds)
3. Using `base_url` and DefaultAzureCredential (for managed identity, Azure CLI, etc.)

**YAML Configuration:**

```
# Option 1: Standard configuration with resource_name
azure:
  api_key: "your_azure_openai_key" # Required unless using DefaultAzureCredential
  resource_name: "your-resource-name" # Resource name (do NOT include if using base_url)
  azure_deployment: "deployment-name" # Required - the model deployment name
  api_version: "2023-05-15" # Optional, default shown
  # Do NOT include base_url if you use resource_name

# Option 2: Custom endpoint with base_url
azure:
  api_key: "your_azure_openai_key"
  base_url: "https://your-resource-name.openai.azure.com" # Full endpoint URL
  azure_deployment: "deployment-name"
  api_version: "2023-05-15" # Optional
  # Do NOT include resource_name if you use base_url

# Option 3: Using DefaultAzureCredential (requires azure-identity package)
azure:
  use_default_azure_credential: true
  base_url: "https://your-resource-name.openai.azure.com"
  azure_deployment: "deployment-name"
  api_version: "2023-05-15" # Optional
  # Do NOT include api_key or resource_name when using DefaultAzureCredential
```

**Important Configuration Notes:**
\- Use either `resource_name` or `base_url`, not both.
\- When using `DefaultAzureCredential`, do NOT include `api_key` or `resource_name`.
\- When using `base_url`, do NOT include `resource_name`.
\- When using `resource_name`, do NOT include `base_url`.

**Environment Variables:**

- `AZURE_OPENAI_API_KEY`: Your Azure OpenAI API key
- `AZURE_OPENAI_ENDPOINT`: Override the API endpoint

**Model Name Format:**

Use `azure.deployment-name` as the model string, where `deployment-name` is the name of your Azure OpenAI deployment.

## Groq

Groq is supported for Structured Outputs and Tool Calling, and has been tested with `moonshotai/kimi-k2-instruct`, `qwen/qwen3-32b` and `deepseek-r1-distill-llama-70b`.

**YAML Configuration:**

```
groq:
  api_key: "your_groq_api_key"
  base_url: "https://api.groq.com/openai/v1"
```

**Environment Variables:**

- `GROQ_API_KEY`: Your Groq API key
- `GROQ_BASE_URL`: Override the API endpoint

**Model Name Aliases:**

| Model Alias | Maps to |
| --- | --- |
| `kimigroq` | `moonshotai/kimi-k2-instruct` |

## DeepSeek

DeepSeek v3 is supported for Text and Tool calling.

**YAML Configuration:**

```
deepseek:
  api_key: "your_deepseek_key"
  base_url: "https://api.deepseek.com/v1"
```

**Environment Variables:**

- `DEEPSEEK_API_KEY`: Your DeepSeek API key
- `DEEPSEEK_BASE_URL`: Override the API endpoint

**Model Name Aliases:**

| Model Alias | Maps to |
| --- | --- |
| `deepseek` | `deepseek-chat` |
| `deepseek3` | `deepseek-chat` |

## Google

Google is natively supported in `fast-agent` using the Google genai libraries.

**YAML Configuration:**

```
google:
  api_key: "your_google_key"
  base_url: "https://generativelanguage.googleapis.com/v1beta/openai"
```

**Environment Variables:**

- `GOOGLE_API_KEY`: Your Google API key

**Model Name Aliases:**

| Model Alias | Maps to |
| --- | --- |
| `gemini2` | `gemini-2.0-flash` |
| `gemini25` | `gemini-2.5-flash-preview-05-20` |
| `gemini25pro` | `gemini-2.5-pro-preview-05-06` |

### OpenAI Mode

You can also access Google via the OpenAI Provider. Use `googleoai` in the YAML file, or `GOOGLEOAI_API_KEY` for API KEY access.

## XAI Grok

XAI Grok 3, Grok 4 and Grok 4 Fast are available through the XAI Provider.

**YAML Configuration:**

```
xai:
  api_key: "your_xai_key"
  base_url: "https://api.x.ai/v1"
```

**Environment Variables:**

- `XAI_API_KEY`: Your Grok API key
- `XAI_BASE_URL`: Override the API endpoint

**Model Name Aliases:**

| Model Alias | Maps to (xai.) |
| --- | --- |
| `grok-3` | `grok-3` |
| `grok-3-fast` | `grok-3-fast` |
| `grok-3-mini` | `grok-3-mini` |
| `grok-3-mini-fast` | `grok-3-mini-fast` |
| `grok-4` | `grok-4` |
| `grok-4-fast` | `grok-4-fast-non-reasoning` |
| `grok-4-fast-reasoning` | `grok-4-fast-reasoning` |

## Generic OpenAI / Ollama

Models prefixed with `generic` will use a generic OpenAI endpoint, with the defaults configured to work with Ollama [OpenAI compatibility](https://github.com/ollama/ollama/blob/main/docs/openai.md).

This means that to run Llama 3.2 latest you can specify `generic.llama3.2:latest` for the model string, and no further configuration should be required.

Warning

The generic provider is tested for tool calling and structured generation with `qwen2.5:latest` and `llama3.2:latest`. Other models and configurations may not work as expected - use at your own risk.

**YAML Configuration:**

```
generic:
  api_key: "ollama" # Default for Ollama, change as needed
  base_url: "http://localhost:11434/v1" # Default for Ollama
```

**Environment Variables:**

- `GENERIC_API_KEY`: Your API key (defaults to `ollama` for Ollama)
- `GENERIC_BASE_URL`: Override the API endpoint

**Usage with other OpenAI API compatible providers:**
By configuring the `base_url` and appropriate `api_key`, you can connect to any OpenAI API-compatible provider.

## OpenRouter

Uses the [OpenRouter](https://openrouter.ai/) aggregation service. Models are accessed via an OpenAI-compatible API. Supported modalities depend on the specific model chosen on OpenRouter.

Models _must_ be specified using the `openrouter.` prefix followed by the full model path from OpenRouter (e.g., `openrouter.google/gemini-flash-1.5`).

Warning

There is an issue with between OpenRouter and Google Gemini models causing large Tool Call block content to be removed.

**YAML Configuration:**

```
openrouter:
  api_key: "your_openrouter_key" # Required
  base_url: "https://openrouter.ai/api/v1" # Default, only include to override
```

**Environment Variables:**

- `OPENROUTER_API_KEY`: Your OpenRouter API key
- `OPENROUTER_BASE_URL`: Override the API endpoint

**Model Name Aliases:**

OpenRouter does not use aliases in the same way as Anthropic or OpenAI. You must always use the `openrouter.provider/model-name` format.

## TensorZero Integration

[TensorZero](https://tensorzero.com/) is an open-source framework for building production-grade LLM applications. It unifies an LLM gateway, observability, optimization, evaluations, and experimentation into a single, cohesive system.

**Why Choose This Integration?**

While `fast-agent` can connect directly to many LLM providers, integrating with TensorZero offers powerful advantages for building robust, scalable, and maintainable agentic systems:

- **Decouple Your Agent from Models:** Define task-specific "functions" (e.g., `summarizer`, `code_generator`) in TensorZero. Your `fast-agent` code calls these simple functions, while TensorZero handles the complexity of which model or provider to use. You can swap `GPT-4o` for `Claude 3.5 Sonnet` on the backend without changing a single line of your agent's code.
- **Effortless Fallbacks & Retries:** Configure sophisticated failover strategies. If your primary model fails or is too slow, TensorZero can automatically retry with a different model or provider, making your agent far more resilient.
- **Advanced Prompt Management:** Keep your complex system prompts and configurations in TensorZero's templates, not hardcoded in your Python strings. This cleans up your agent logic and allows for easier experimentation.
- **Unified Observability:** All inference calls from your agents are logged, cached, and analyzed in one place, giving you a powerful, centralized view of your system's performance and costs.

**Getting Started: The `quickstart` Command**

The fastest way to get started is with the built-in, self-contained example. From your terminal, run:

```
fast-agent quickstart tensorzero
```

This command will create a new `tensorzero/` directory containing a fully dockerized project that includes:

1. A pre-configured **TensorZero Gateway**.
2. A custom **MCP Server** for your agent to use.
3. Support for multimodal inputs using a **MiniIO** service.
4. An interactive **`fast-agent`** that is ready to run by invoking `make agent`.

Just follow the "Next Steps" printed in your terminal to launch the agent.

**How it Works**

The `fast-agent` implementation uses TensorZero's OpenAI-compatible inference API. To call a "function" defined in your TensorZero configuration (e.g., in `tensorzero.toml`), simply specify it as the model name, prefixed with `tensorzero.`:

```
# Example from the quickstart Makefile
uv run agent.py --model=tensorzero.test_chat
```

By leveraging the common OpenAI interface, the integration remains simple and benefits from the extensive work done to support OpenAI-based models and features within both `fast-agent` and TensorZero.

TensorZero is an [Apache 2.0 licensed project](https://github.com/sproutfi/tensorzero?tab=License-1-ov-file) and you can find more details in the [official documentation](https://www.tensorzero.com/docs).

**YAML Configuration**

By default, the TenzorZero Gateway runs on `http://localhost:3000`. You can override this by specifying the `base_url` in your configuration.

```
tensorzero:
  base_url: "http://localhost:3000" # Optional, only include to override
```

**Environment Variables:**

None (model provider credentials should be provided to the TensorZero Gateway instead)

## Aliyun

Tongyi Qianwen is a large-scale language model independently developed by Alibaba Cloud, featuring strong natural language understanding and generation capabilities. It can answer various questions, create written content, express opinions, and write code, playing a role in multiple fields.

**YAML Configuration:**

```
aliyun:
  api_key: "your_aliyun_key"
  base_url: "https://dashscope.aliyuncs.com/compatible-mode/v1"
```

**Environment Variables:**

- `ALIYUN_API_KEY`: Your Aliyun API key
- `ALIYUN_BASE_URL`: Override the API endpoint

**Model Name Aliases:**

Check the [Aliyun Official Documentation](https://help.aliyun.com/zh/model-studio/models) for the latest model names and aliases.

| Model Alias | Maps to |
| --- | --- |
| `qwen-turbo` | `qwen-turbo-2025-02-11` |
| `qwen-plus` | `qwq-plus-2025-03-05` |
| `qwen-max` | `qwen-max-2024-09-19` |
| `qwen-long` | _undocumented_ |

## AWS Bedrock

AWS Bedrock provides access to multiple foundation models from Amazon, Anthropic, AI21, Cohere, Meta, Mistral, and other providers through a unified API. fast-agent supports the full range of Bedrock models with intelligent capability detection and optimization.

**Key Features:**

- **Multi-provider model access**: Nova, Claude, Titan, Cohere, Llama, Mistral, and more
- **Intelligent capability detection**: Automatically handles models that don't support system messages or tool use
- **Optimized streaming**: Uses streaming when supported, falls back to non-streaming when required
- **Model-specific optimizations**: Tailored configurations for different model families

**YAML Configuration:**

```
bedrock:
  region: "us-east-1" # Required - AWS region where Bedrock is available
  profile: "default"  # Optional - AWS profile to use (defaults to "default")
                      # Only needed on local machines, not required on AWS
```

**Environment Variables:**

- `AWS_REGION` or `AWS_DEFAULT_REGION`: AWS region (e.g., `us-east-1`)
- `AWS_PROFILE`: Named AWS profile to use
- `AWS_ACCESS_KEY_ID`: Your AWS access key (handled by boto3)
- `AWS_SECRET_ACCESS_KEY`: Your AWS secret key (handled by boto3)
- `AWS_SESSION_TOKEN`: AWS session token for temporary credentials (handled by boto3)

**Model Name Format:**

Use `bedrock.model-id` where `model-id` is the Bedrock model identifier:

- `bedrock.amazon.nova-premier-v1:0` \- Amazon Nova Premier
- `bedrock.amazon.nova-pro-v1:0` \- Amazon Nova Pro
- `bedrock.amazon.nova-lite-v1:0` \- Amazon Nova Lite
- `bedrock.anthropic.claude-3-7-sonnet-20241022-v1:0` \- Claude 3.7 Sonnet
- `bedrock.anthropic.claude-3-5-sonnet-20241022-v2:0` \- Claude 3.5 Sonnet v2
- `bedrock.meta.llama3-1-405b-instruct-v1:0` \- Meta Llama 3.1 405B
- `bedrock.mistral.mistral-large-2402-v1:0` \- Mistral Large

**Supported Models:**

The provider automatically detects and handles model-specific capabilities:

- **System messages**: Automatically injects system prompts into user messages for models that don't support them (Titan, Cohere Command Text, etc.)
- **Tool use**: Skips tool preparation for models that don't support tools (Titan, Claude v2, Llama 2/3, etc.)
- **Streaming**: Uses non-streaming API when models don't support streaming with tools

Note that Bedrock contains some models that may perform poorly in some areas, including INSTRUCT models as well as models that are made to be fine-tuned for specific use cases. If you are unsure about model capabilities, be sure to read the documentation.

**Model Capabilities:**

Refer to the [AWS Bedrock documentation](https://docs.aws.amazon.com/bedrock/latest/userguide/conversation-inference-supported-models-features.html) for the latest model capabilities including system prompts, tool use, vision, and streaming support.

**Authentication:**

AWS Bedrock uses standard AWS authentication. Configure credentials using:

1. **AWS CLI**: Run `aws configure` to set up credentials. AWS SSO is a great choice for local development.
2. **Environment variables**: Set `AWS_ACCESS_KEY_ID` and `AWS_SECRET_ACCESS_KEY`
3. **IAM roles**: Use IAM roles when running on EC2 or other AWS services
4. **AWS profiles**: Use named profiles with `AWS_PROFILE` environment variable

Required IAM permissions:
\- `bedrock:InvokeModel`
\- `bedrock:InvokeModelWithResponseStream`

Back to top

## Fast Agent Setup
# Getting started

Install or upgrade to the latest version:

```
uvx tool install -U fast-agent-mcp
```

Set an API KEY h

To run:

```
fast-agent
```

To load an instruction file:

```
fast-agent -i prompt.md
fast-agent -i https://gist.github.com/....
```

To specify

Back to top

## Agent Client Protocol
[Skip to content](https://fast-agent.ai/acp/#agent-client-protocol)

# Agent Client Protocol

**`fast-agent`** has comprehensive support for Zed Industries [Agent Client Protocol](https://zed.dev/acp).

Why use **`fast-agent`**?:

- Robust, native LLM Provider infrastructure, with Streaming and Structured outputs.
- Comprehensive MCP and Agent Skills support, including Tool Progress Notifications and Sampling.
- Build custom, multi-agent experiences in a few lines of code.

## Features

| Feature | Support | Notes |
| --- | --- | --- |
| Modes | ✅ | Each defined Agent appears as a Modes |
| Tool / Workflow Progress | ✅ | MCP Tool Progress and Agent Workflow Progress updates |
| Agent Plan | ✅ | Iterative Planner reports progress using [Agent Plan](https://agentclientprotocol.com/protocol/agent-plan) |
| Cancellation | ✅ | LLM Streaming Cancellation |
| Multimodal | ✅ | Support for Images |
| Slash Commands | ✅ | Save, Load, Status and Clear/Clear Last message |
| File System / Terminal | ✅ | Start with `-x` option to enable access to Client terminal |
| MCP Servers | ⚠️ | Add via command line switches or configuration file |
| Sessions | ⚠️ | Use `save` and `load` slash commands. Plan to implement with [Session List](https://agentclientprotocol.com/rfds/session-list) |

## Getting Started

### No Install Quick Start:

To try it out straight away with your Client, set an API Key environment variable and add:

**Hugging Face**

export HF\_TOKEN=hf\_.......

`uvx fast-agent-acp@latest --model <your_model> [e.g. kimi]`

**Open AI**

export OPENAI\_API\_KEY=......

`uvx fast-agent-acp@latest  --model <your_model> [e.g. gpt-5-mini.low]`

**Anthropic**

export ANTHROPIC\_API\_KEY=......

`uvx fast-agent-acp@latest --model <your_model> e.g. [sonnet]`

Tip: Use `uvx fast-agent-acp check` to help diagnose issues.

The [default system prompt](https://fast-agent.ai/agents/instructions/) will read `AGENTS.md` if present. Use `/status system` to check.

Note: OAuth keys are stored in your keyring, so `check` may prompt to read the credential store.

An example Zed configuration is:

```
...
"agent_servers": {
    "fast-agent-uvx": {
        "command": "uvx",
        "args": [\
        "fast-agent-acp@latest",\
        "--model",\
        "kimi",\
        "-x",\
        "--url",\
        "https://huggingface.co/mcp"\
        ],
        "env": { "HF_TOKEN": "hf_xxxxxxxxxxx" }
    }
}
```

### Installing

`uv tool install -U fast-agent-mcp`

The ACP Server can then be started with the `fast-agent-acp` command. Custom agents can be started with `uv <agent.py> --transport acp`.

For example:

`fast-agent-acp -x --model kimi --url https://huggingface.co/mcp --auth ${HF_TOKEN}`

Starts an ACP Agent, with shell access and access to the Hugging Face MCP Server.

Documentation in Progress.

## Shell and File Access

**`fast-agent`** adds the read and write tools from the Client to enable "follow-along" functionality.

Back to top

## File and Resource Management
[Skip to content](https://fast-agent.ai/agents/#files-and-resources)

# Files and Resources

## Attaching Files

You can include files in a conversation using Paths:

```
from fast_agent.core.prompt import Prompt
from pathlib import Path

plans = agent.send(
    Prompt.user(
        "Summarise this PDF",
        Path("secret-plans.pdf")
    )
)
```

This works for any mime type that can be tokenized by the model.

## MCP Resources

MCP Server resources can be conveniently included in a message with:

```
description = agent.with_resource(
    "What is in this image?",
    "mcp_image_server",
    "resource://images/cat.png"
)
```

## Prompt Files

Prompt Files can include Resources:

agent\_script.txt

```
---USER
Please extract the major colours from this CSS file:
---RESOURCE
index.css
```

They can either be loaded with the `load_prompt_multipart` function, or delivered via the built-in `prompt-server`.

## Defining Agents and Workflows
[Skip to content](https://fast-agent.ai/agents/defining/#defining-agents-and-workflows)

# Defining Agents and Workflows

## Basic Agents

Defining an agent is as simple as:

```
@fast.agent(
  instruction="Given an object, respond only with an estimate of its size."
)
```

We can then send messages to the Agent:

```
async with fast.run() as agent:
  moon_size = await agent("the moon")
  print(moon_size)
```

Or start an interactive chat with the Agent:

```
async with fast.run() as agent:
  await agent.interactive()
```

Here is the complete `sizer.py` Agent application, with boilerplate code:

sizer.py

```
import asyncio
from fast_agent.core.fastagent import FastAgent

# Create the application
fast = FastAgent("Agent Example")

@fast.agent(
  instruction="Given an object, respond only with an estimate of its size."
)
async def main():
  async with fast.run() as agent:
    await agent()

if __name__ == "__main__":
    asyncio.run(main())
```

The Agent can then be run with `uv run sizer.py`.

Specify a model with the `--model` switch - for example `uv run sizer.py --model sonnet`.

You can also pass a `Path` for the instruction - e.g.

```
from pathlib import Path

@fast.agent(
  instruction=Path("./sizing_prompt.md")
)
```

## Workflows and MCP Servers

_To generate examples use `fast-agent quickstart workflow`. This example can be run with `uv run workflow/chaining.py`. fast-agent looks for configuration files in the current directory before checking parent directories recursively._

Agents can be chained to build a workflow, using MCP Servers defined in the `fastagent.config.yaml` file:

fastagent.config.yaml

```
# Example of a STDIO sever named "fetch"
mcp:
  servers:
    fetch:
      command: "uvx"
      args: ["mcp-server-fetch"]
```

social.py

```
@fast.agent(
    "url_fetcher",
    "Given a URL, provide a complete and comprehensive summary",
    servers=["fetch"], # Name of an MCP Server defined in fastagent.config.yaml
)
@fast.agent(
    "social_media",
    """
    Write a 280 character social media post for any given text.
    Respond only with the post, never use hashtags.
    """,
)
@fast.chain(
    name="post_writer",
    sequence=["url_fetcher", "social_media"],
)
async def main():
    async with fast.run() as agent:
        # using chain workflow
        await agent.post_writer("http://fast-agent.ai")
```

All Agents and Workflows respond to `.send("message")`. The agent app responds to `.interactive()` to start a chat session.

Saved as `social.py` we can now run this workflow from the command line with:

```
uv run workflow/chaining.py --agent post_writer --message "<url>"
```

Add the `--quiet` switch to disable progress and message display and return only the final response - useful for simple automations.

Read more about running **fast-agent** agents [here](https://fast-agent.ai/agents/running/)

## Workflow Types

**fast-agent** has built-in support for the patterns referenced in Anthropic's [Building Effective Agents](https://www.anthropic.com/research/building-effective-agents) paper.

### Chain

The `chain` workflow offers a declarative approach to calling Agents in sequence:

```
@fast.chain(
  "post_writer",
  sequence=["url_fetcher","social_media"]
)

# we can them prompt it directly:
async with fast.run() as agent:
  await agent.interactive(agent="post_writer")
```

This starts an interactive session, which produces a short social media post for a given URL. If a _chain_ is prompted it returns to a chat with last Agent in the chain. You can switch agents by typing `@agent-name`.

Chains can be incorporated in other workflows, or contain other workflow elements (including other Chains). You can set an `instruction` to describe it's capabilities to other workflow steps if needed.

Chains are also helpful for capturing content before being dispatched by a `router`, or summarizing content before being used in the downstream workflow.

### Human Input

Agents can request Human Input to assist with a task or get additional context:

```
@fast.agent(
    instruction="An AI agent that assists with basic tasks. Request Human Input when needed.",
    human_input=True,
)

await agent("print the next number in the sequence")
```

In the example `human_input.py`, the Agent will prompt the User for additional information to complete the task.

### Parallel

The Parallel Workflow sends the same message to multiple Agents simultaneously (`fan-out`), then uses the `fan-in` Agent to process the combined content.

```
@fast.agent("translate_fr", "Translate the text to French")
@fast.agent("translate_de", "Translate the text to German")
@fast.agent("translate_es", "Translate the text to Spanish")

@fast.parallel(
  name="translate",
  fan_out=["translate_fr","translate_de","translate_es"]
)

@fast.chain(
  "post_writer",
  sequence=["url_fetcher","social_media","translate"]
)
```

If you don't specify a `fan-in` agent, the `parallel` returns the combined Agent results verbatim.

`parallel` is also useful to ensemble ideas from different LLMs.

When using `parallel` in other workflows, specify an `instruction` to describe its operation.

### Evaluator-Optimizer

Evaluator-Optimizers combine 2 agents: one to generate content (the `generator`), and the other to judge that content and provide actionable feedback (the `evaluator`). Messages are sent to the generator first, then the pair run in a loop until either the evaluator is satisfied with the quality, or the maximum number of refinements is reached. The final result from the Generator is returned.

If the Generator has `use_history` off, the previous iteration is returned when asking for improvements - otherwise conversational context is used.

```
@fast.evaluator_optimizer(
  name="researcher",
  generator="web_searcher",
  evaluator="quality_assurance",
  min_rating="EXCELLENT",
  max_refinements=3
)

async with fast.run() as agent:
  await agent.researcher.send("produce a report on how to make the perfect espresso")
```

When used in a workflow, it returns the last `generator` message as the result.

See the `evaluator.py` workflow example, or `fast-agent quickstart researcher` for a more complete example.

### Router

Routers use an LLM to assess a message, and route it to the most appropriate Agent. The routing prompt is automatically generated based on the Agent instructions and available Servers.

```
@fast.router(
  name="route",
  agents=["agent1","agent2","agent3"]
)
```

NB - If only one agent is supplied to the router, it forwards directly.

Look at the `router.py` workflow for an example.

### Orchestrator

Given a complex task, the Orchestrator uses an LLM to generate a plan to divide the task amongst the available Agents. The planning and aggregation prompts are generated by the Orchestrator, which benefits from using more capable models. Plans can either be built once at the beginning (`plantype="full"`) or iteratively (`plantype="iterative"`).

```
@fast.orchestrator(
  name="orchestrate",
  agents=["task1","task2","task3"]
)
```

See the `orchestrator.py` or `agent_build.py` workflow example.

## Agent and Workflow Reference

### Calling Agents

All definitions allow omitting the name and instructions arguments for brevity:

```
@fast.agent("You are a helpful agent")          # Create an agent with a default name.
@fast.agent("greeter","Respond cheerfully!")    # Create an agent with the name "greeter"

moon_size = await agent("the moon")             # Call the default (first defined agent) with a message

result = await agent.greeter("Good morning!")   # Send a message to an agent by name using dot notation
result = await agent.greeter.send("Hello!")     # You can call 'send' explicitly

agent["greeter"].send("Good Evening!")          # Dictionary access to agents is also supported
```

Read more about prompting agents [here](https://fast-agent.ai/agents/prompting/)

## Configuring Agent Request Parameters

You can customize how an agent interacts with the LLM by passing `request_params=RequestParams(...)` when defining it.

### Example

```
from fast_agent.core.request_params import RequestParams

@fast.agent(
  name="CustomAgent",                              # name of the agent
  instruction="You have my custom configurations", # base instruction for the agent
  request_params=RequestParams(
    maxTokens=8192,
    use_history=False,
    max_iterations=20
  )
)
```

### Available RequestParams Fields

| Field | Type | Default | Description |
| --- | --- | --- | --- |
| `maxTokens` | `int` | `2048` | The maximum number of tokens to sample, as requested by the server |
| `model` | `string` | `None` | The model to use for the LLM generation. Can only be set at Agent creation time |
| `use_history` | `bool` | `True` | Agent/LLM maintains conversation history. Does not include applied Prompts |
| `max_iterations` | `int` | `20` | The maximum number of tool calls allowed in a conversation turn |
| `parallel_tool_calls` | `bool` | `True` | Whether to allow simultaneous tool calls |
| `response_format` | `Any` | `None` | Response format for structured calls (advanced use). Prefer to use `structured` with a Pydantic model instead |
| `template_vars` | `Dict[str,Any]` | `{}` | Dictionary of template values for dynamic templates. Currently only supported for TensorZero provider |
| `temperature` | `float` | `None` | Temperature to use for the completion request |

### Defining Agents

#### Basic Agent

```
@fast.agent(
  name="agent",                          # name of the agent
  instruction="You are a helpful Agent", # base instruction for the agent
  servers=["filesystem"],                # list of MCP Servers for the agent
  #tools={"filesystem": ["tool_1", "tool_2"]  # Filter the tools available to the agent. Defaults to all
  #resources={"filesystem: ["resource_1", "resource_2"]} # Filter the resources available to the agent. Defaults to all
  #prompts={"filesystem": ["prompt_1", "prompt_2"]}  # Filter the prompts available to the agent. Defaults to all.
  model="o3-mini.high",                  # specify a model for the agent
  use_history=True,                      # agent maintains chat history
  request_params=RequestParams(temperature= 0.7), # additional parameters for the LLM (or RequestParams())
  human_input=True,                      # agent can request human input
  elicitation_handler=ElicitationFnT,    # custom elicitation handler (from mcp.client.session)
  api_key="programmatic-api-key",        # specify the API KEY programmatically, it will override which provided in config file or env var
)
```

#### Chain

```
@fast.chain(
  name="chain",                          # name of the chain
  sequence=["agent1", "agent2", ...],    # list of agents in execution order
  instruction="instruction",             # instruction to describe the chain for other workflows
  cumulative=False,                      # whether to accumulate messages through the chain
  continue_with_final=True,              # open chat with agent at end of chain after prompting
)
```

#### Parallel

```
@fast.parallel(
  name="parallel",                       # name of the parallel workflow
  fan_out=["agent1", "agent2"],          # list of agents to run in parallel
  fan_in="aggregator",                   # name of agent that combines results (optional)
  instruction="instruction",             # instruction to describe the parallel for other workflows
  include_request=True,                  # include original request in fan-in message
)
```

#### Evaluator-Optimizer

```
@fast.evaluator_optimizer(
  name="researcher",                     # name of the workflow
  generator="web_searcher",              # name of the content generator agent
  evaluator="quality_assurance",         # name of the evaluator agent
  min_rating="GOOD",                     # minimum acceptable quality (EXCELLENT, GOOD, FAIR, POOR)
  max_refinements=3,                     # maximum number of refinement iterations
)
```

#### Router

```
@fast.router(
  name="route",                          # name of the router
  agents=["agent1", "agent2", "agent3"], # list of agent names router can delegate to
  instruction="routing instruction",     # any extra routing instructions
  servers=["filesystem"],                # list of servers for the routing agent
  #tools={"filesystem": ["tool_1", "tool_2"]  # Filter the tools available to the agent. Defaults to all
  #resources={"filesystem: ["resource_1", "resource_2"]} # Filter the resources available to the agent. Defaults to all
  #prompts={"filesystem": ["prompt_1", "prompt_2"]}  # Filter the prompts available to the agent. Defaults to all
  model="o3-mini.high",                  # specify routing model
  use_history=False,                     # router maintains conversation history
  human_input=False,                     # whether router can request human input
  api_key="programmatic-api-key",        # specify the API KEY programmatically, it will override which provided in config file or env var
)
```

#### Orchestrator

```
@fast.orchestrator(
  name="orchestrator",                   # name of the orchestrator
  instruction="instruction",             # base instruction for the orchestrator
  agents=["agent1", "agent2"],           # list of agent names this orchestrator can use
  model="o3-mini.high",                  # specify orchestrator planning model
  use_history=False,                     # orchestrator doesn't maintain chat history (no effect).
  human_input=False,                     # whether orchestrator can request human input
  plan_type="full",                      # planning approach: "full" or "iterative"
  max_iterations=5,                      # maximum number of full plan attempts, or iterations
  api_key="programmatic-api-key",        # specify the API KEY programmatically, it will override which provided in config file or env var
)
```

#### Custom

```
@fast.custom(
  cls=Custom                             # agent class
  name="custom",                         # name of the custom agent
  instruction="instruction",             # base instruction for the orchestrator
  servers=["filesystem"],                # list of MCP Servers for the agent
  MCP Servers for the agent
  #tools={"filesystem": ["tool_1", "tool_2"]  # Filter the tools available to the agent. Defaults to all
  #resources={"filesystem: ["resource_1", "resource_2"]} # Filter the resources available to the agent. Defaults to all
  #prompts={"filesystem": ["prompt_1", "prompt_2"]}  # Filter the prompts available to the agent. Defaults to all
  model="o3-mini.high",                  # specify a model for the agent
  use_history=True,                      # agent maintains chat history
  request_params=RequestParams(temperature= 0.7), # additional parameters for the LLM (or RequestParams())
  human_input=True,                      # agent can request human input
  elicitation_handler=ElicitationFnT,    # custom elicitation handler (from mcp.client.session)
  api_key="programmatic-api-key",        # specify the API KEY programmatically, it will override which provided in config file or env var
)
```

Back to top

## AI Agent Instructions
[Skip to content](https://fast-agent.ai/agents/instructions/#system-prompts)

# System Prompts

Agents can have their System Instructions set and customised in a number of flexible ways. The default System Prompt caters or Agent Skills, MCP Server Instructions, `AGENTS.md` and Shell access.

## Template Variables

The following variables are available in System Prompt templates:

| Variable | Description | Notes |
| --- | --- | --- |
| `{{file:path}}` | Reads and embeds local file contents (errors if file missing) | **Must be a relative path** (resolved relative to `workspaceRoot`) |
| `{{file_silent:path}}` | Reads and embeds local file contents (empty if file missing) | **Must be a relative path** (resolved relative to `workspaceRoot`) |
| `{{url:https://...}}` | Fetches and embeds content from a URL |  |
| `{{serverInstructions}}` | MCP server instructions with available tools | Warning displayed in `/mcp` if Instructions are present and template variable missing |
| `{{agentSkills}}` | Agent skill manifests with descriptions |  |
| `{{workspaceRoot}}` | Current working directory / workspace root | Set by Client in ACP Mode |
| `{{hostPlatform}}` | Host platform information |  |
| `{{pythonVer}}` | Python version |  |
| `{{env}}` | Formatted environment block with all environment details |  |
| `{{currentDate}}` | Current date in long format |  |

**Example `{{env}}` output:**

```
Environment:
- Workspace root: /home/user/project
- Client: Zed 0.232
- Host platform: Linux-6.6.87.2-microsoft-standard-WSL2
```

**Note on file templates:** File paths in `{{file:...}}` and `{{file_silent:...}}` must be relative paths. They will be resolved relative to the `workspaceRoot` at runtime. Absolute paths are not allowed and will raise an error.

**Viewing the System Prompt** The System Prompt can be inspected with the `/system` command from `fast-agent` or the `/status
system` Slash Command in ACP Mode.

The default System Prompt used with `fast-agent go` or `fast-agent-acp` is:

Default System Prompt

```
You are a helpful AI Agent.

{{serverInstructions}}
{{agentSkills}}
{{file_silent:AGENTS.md}}
{{env}}

The current date is {{currentDate}}."""
```

## Using Instructions

When defining an Agent, you can load the instruction as either a `String`, `Path` or `AnyUrl`.

Instructions support embedding the current date, as well as content from other URLs. This is really helpful if you want to refer to files on GitHub, or assemble useful prompts/content in Gists etc.

Simple String

```
@fast.agent(name="example",
    instruction="""
You are a helpful AI Agent.
""")
```

With current date

```
@fast.agent(name="example",
    instruction="""
You are a helpful AI Agent.
Your reliable knowledge cut-off date is December 2024.
Todays date is {{currentDate}}.
""")
```

Will produce: `You are a helpful AI Agent. Your reliable knowledge cut-off date is December 2024. Todays date is 25 July 2025.`

With URL

```
@fast.agent(name="mcp-expert",
    instruction="""
You are have expert knowledge of the
MCP (Model Context Protocol) schema.

{{url:https://raw.githubusercontent.com/modelcontextprotocol/modelcontextprotocol/refs/heads/main/schema/2025-06-18/schema.ts}}

Answer any questions about the protocol by referring
to and quoting the schema where necessary.
""")
```

You can store the prompt in an external file for easy editing - including template variables:

From file

```
from pathlib import Path

@fast.agent(name="mcp-expert",
    instruction=Path("./mcp-expert.md"))
""")
```

mcp-expert.md

```
You are have expert knowledge of the MCP (Model Context Protocol) schema.

{{url:https://raw.githubusercontent.com/modelcontextprotocol/modelcontextprotocol/refs/heads/main/schema/2025-06-18/schema.ts}}

Answer any questions about the protocol by referring to and quoting the schema where necessary.
Your knowledge cut-off is December 2024, todays date is {{currentDate}}
```

Or you can load the prompt directly from a URL:

From URL

```
from pydantic import AnyUrl

@fast.agent(name="mcp-expert",
    instruction=AnyUrl("https://gist.githubusercontent.com/evalstate/d432921aaaee2c305cf46ae320840360/raw/eb9c7ff93adc780171bfb0ae2560be2178304f16/gistfile1.txt"))

# --> fast-agent system prompt demo
```

You can start an agent with instructions from a file using the `fast-agent` commmand:

```
fast-agent --instructions mcp-expert.md
fast-agent -i mcp-expert.md
```

This can be combined with other options to specify model and available servers:

```
fast-agent -i mcp-expert.md --model sonnet --url https://hf.co/mcp
```

Starts an interactive agent session, with the MCP Schema loaded, attached to Sonnet with the Hugging Face MCP Server.

![Instructions](https://fast-agent.ai/agents/instructions.png)

You can even specify multiple models to directly compare their outputs:

![Instructions Parallel](https://fast-agent.ai/agents/instructions_parallel.png)

Read more about the `fast-agent` command [here](https://fast-agent.ai/ref/go_command/).

Back to top

## Prompting Agents API
[Skip to content](https://fast-agent.ai/agents/prompting/#prompting-agents)

# Prompting Agents

**fast-agent** provides a flexible MCP based API for sending messages to agents, with convenience methods for handling Files, Prompts and Resources.

Read more about the use of MCP types in **fast-agent** [here](https://fast-agent.ai/mcp/types/).

## Sending Messages

The simplest way of sending a message to an agent is the `send` method:

```
response: str = await agent.send("how are you?")
```

This returns the text of the agent's response as a string, making it ideal for simple interactions.

You can attach files by using `Prompt.user()` method to construct your message:

```
from fast_agent.core.prompt import Prompt
from pathlib import Path

plans: str = await agent.send(
    Prompt.user(
        "Summarise this PDF",
        Path("secret-plans.pdf")
    )
)
```

`Prompt.user()` automatically converts content to the appropriate MCP Type. For example, `image/png` becomes `ImageContent` and `application/pdf` becomes an EmbeddedResource.

You can also use MCP Types directly - for example:

```
from mcp.types import ImageContent, TextContent

mcp_text: TextContent = TextContent(type="text", text="Analyse this image.")
mcp_image: ImageContent = ImageContent(type="image",
                          mimeType="image/png",
                          data=base_64_encoded)

response: str  = await agent.send(
    Prompt.user(
        mcp_text,
        mcp_image
    )
)
```

> Note: use `Prompt.assistant()` to produce messages for the `assistant` role.

### Using `generate()` and multipart content

The `generate()` method allows you to access multimodal content from an agent, or its Tool Calls as well as send conversational pairs.

```
from fast_agent import FastAgent, Prompt, PromptMessageExtended

message = Prompt.user("Describe an image of a sunset")

response: PromptMessageExtended = await agent.generate([message])

print(response.last_text())  # Main text response
```

The key difference between `send()` and `generate()` is that `generate()` returns a `PromptMessageExtended` object, giving you access to the complete response structure:

- `last_text()`: Gets the last text response - usually the Assistant message without Tool Call/Response information.
- `first_text()`: Gets the first text content if multiple text blocks exist
- `all_text()`: Combines all text content in the response - including Tall Call/Response information.
- `content`: Direct access to the full list of content parts, including Images and EmbeddedResources

This is particularly useful when working with multimodal responses or tool outputs:

```
# Generate a response that might include multiple content types
response = await agent.generate([\
    Prompt.user("Analyze this image", Path("chart.png"))\
])

for content in response.content:
    if content.type == "text":
        print("Text response:", content.text[:100], "...")
    elif content.type == "image":
        print("Image content:", content.mimeType)
    elif content.type == "resource":
        print("Resource:", content.resource.uri)
```

You can also use `generate()` for multi-turn conversations by passing multiple messages:

```
messages = [\
    Prompt.user("What is the capital of France?"),\
    Prompt.assistant("The capital of France is Paris."),\
    Prompt.user("And what is its population?")\
]

response = await agent.generate(messages)
```

The `generate()` method provides the foundation for working with content returned by the LLM, and MCP Tool, Prompt and Resource calls.

### Using `structured()` for typed responses

When you need the agent to return data in a specific format, use the `structured()` method. This parses the agent's response into a Pydantic model:

```
from pydantic import BaseModel
from typing import List

# Define your expected response structure
class CityInfo(BaseModel):
    name: str
    country: str
    population: int
    landmarks: List[str]

# Request structured information
result, message = await agent.structured(
    [Prompt.user("Tell me about Paris")],
    CityInfo
)

# Now you have strongly typed data
if result:
    print(f"City: {result.name}, Population: {result.population:,}")
    for landmark in result.landmarks:
        print(f"- {landmark}")
```

The `structured()` method returns a tuple containing:
1\. The parsed Pydantic model instance (or `None` if parsing failed)
2\. The full `PromptMessageExtended` response

This approach is ideal for:
\- Extracting specific data points in a consistent format
\- Building workflows where agents need structured inputs/outputs
\- Integrating agent responses with typed systems

Always check if the first value is `None` to handle cases where the response couldn't be parsed into your model:

```
result, message = await agent.structured([Prompt.user("Describe Paris")], CityInfo)

if result is None:
    # Fall back to the text response
    print("Could not parse structured data, raw response:")
    print(message.last_text())
```

The `structured()` method provides the same request parameter options as `generate()`.

Note

LLMs produce JSON when producing Structured responses, which can conflict with Tool Calls. Use a `chain` to combine Tool Calls with Structured Outputs.

## MCP Prompts

Apply a Prompt from an MCP Server to the agent with:

```
response: str = await agent.apply_prompt(
    "setup_sizing",
    arguments={"units": "metric"}
)
```

You can list and get Prompts from attached MCP Servers:

```
from mcp.types import GetPromptResult, PromptMessage

prompt: GetPromptResult = await agent.get_prompt("setup_sizing")
first_message: PromptMessage = prompt[0]
```

and send the native MCP `PromptMessage` to the agent with:

```
response: str = agent.send(first_message)
```

> If the last message in the conversation is from the `assistant`, it is returned as the response.

## MCP Resources

`Prompt.user` also works with MCP Resources:

```
from mcp.types import ReadResourceResult

resource: ReadResourceResult = agent.get_resource(
    "resource://images/cat.png", "mcp_server_name"
)
response: str = agent.send(
    Prompt.user("What is in this image?", resource)
)
```

Alternatively, use the _with\_resource_ convenience method:

```
response: str = agent.with_resource(
    "What is in this image?",
    "resource://images/cat.png"
    "mcp_server_name",
)
```

## Prompt Files

Long prompts can be stored in text files, and loaded with the `load_prompt` utility:

```
from fast_agent.mcp.prompts import load_prompt
from mcp.types import PromptMessage

prompt: List[PromptMessage] = load_prompt(Path("two_cities.txt"))
result: str = await agent.send(prompt[0])
```

two\_cities.txt

```
### The Period

It was the best of times, it was the worst of times, it was the age of
wisdom, it was the age of foolishness, it was the epoch of belief, it was
the epoch of incredulity, ...
```

Prompts files can contain conversations to aid in-context learning or allow you to replay conversations with the Playback LLM:

sizing\_conversation.txt

```
---USER
the moon
---ASSISTANT
object: MOON
size: 3,474.8
units: KM
---USER
the earth
---ASSISTANT
object: EARTH
size: 12,742
units: KM
---USER
how big is a tiger?
---ASSISTANT
object: TIGER
size: 1.2
units: M
```

Multiple messages (conversations) can be applied with the `generate()` method:

```
from fast_agent.mcp.prompts import load_prompt
from mcp.types import PromptMessage

prompt: List[PromptMessage] = load_prompt(Path("sizing_conversation.txt"))
result: PromptMessageExtended = await agent.generate(prompt)
```

Conversation files can also be used to include resources:

prompt\_secret\_plans.txt

```
---USER
Please review the following documents:
---RESOURCE
secret_plan.pdf
---RESOURCE
repomix.xml
---ASSISTANT
Thank you for those documents, the PDF contains secret plans, and some
source code was attached to achieve those plans. Can I help further?
```

```
from fast_agent.mcp.prompts import load_prompt
from fast_agent import PromptMessageExtended

prompt: List[PromptMessageExtended] = load_prompt(Path("prompt_secret_plans.txt"))
result: PromptMessageExtended = await agent.generate(prompt)
```

File Format / MCP Serialization

If the filetype is `json`, then messages are deserialized using the MCP Prompt schema format. The `load_prompt`, `load_prompt_multipart` and `prompt-server` will load either the text or JSON format directly.
See [History Saving](https://fast-agent.ai/models/#history-saving) to learn how to save a conversation to a file for editing or playback.

### Using the `prompt-server`

Prompt files can also be served using the inbuilt `prompt-server`. The `prompt-server` command is installed with `fast-agent` making it convenient to set up and use:

fastagent.config.yaml

```
mcp:
  servers:
    prompts:
      command: "prompt-server"
      args: ["prompt_secret_plans.txt"]
```

This configures an MCP Server that will serve a `prompt_secret_plans` MCP Prompt, and `secret_plan.pdf` and `repomix.xml` as MCP Resources.

If arguments are supplied in the template file, these are also handled by the `prompt-server`

prompt\_with\_args.txt

```
---USER
Hello {{assistant_name}}, how are you?
---ASSISTANT
Great to meet you {{user_name}} how can I be of assistance?
```

Back to top

## Fast-Agent Deployment Options
[Skip to content](https://fast-agent.ai/agents/running/#deploy-and-run)

# Deploy and Run

**fast-agent** provides flexible deployment options to meet a variety of use cases, from interactive development to production server deployments.

## Interactive Mode

Run **fast-agent** programs interactively for development, debugging, or direct user interaction.

agent.py

```
import asyncio
from fast_agent.core.fastagent import FastAgent

fast = FastAgent("My Interactive Agent")

@fast.agent(instruction="You are a helpful assistant")
async def main():
    async with fast.run() as agent:
        # Start interactive prompt
        await agent()

if __name__ == "__main__":
    asyncio.run(main())
```

When started with `uv run agent.py`, this begins an interactive prompt where you can chat directly with the configured agents, apply prompts, save history and so on.

## Command Line Execution

**fast-agent** supports command-line arguments to run agents and workflows with specific messages.

```
# Send a message to a specific agent
uv run agent.py --agent default --message "Analyze this dataset"

# Override the default model
uv run agent.py --model gpt-4o --agent default --message "Complex question"

# Run with minimal output
uv run agent.py --quiet --agent default --message "Background task"
```

This is perfect for scripting, automation, or one-off queries.

The `--quiet` flag switches off the Progress, Chat and Tool displays.

## MCP Server Deployment

Any **fast-agent** application can be deployed as an MCP server with a simple command-line switch.

### Starting an MCP Server

```
# Start as a Streamable HTTP server (http://localhost:8080/mcp)
uv run agent.py --transport http --port 8080

# Start as an SSE server (http://localhost:8080/sse)
uv run agent.py --transport sse --port 8080

# Start as a stdio server
uv run agent.py --transport stdio
```

Each agent exposes an MCP Tool for sending messages to the agent, and a Prompt that returns the conversation history.

This enables cross-agent state transfer via the MCP Prompts.

The MCP Server can also be started programatically.

### Programmatic Server Startup

```
import asyncio
from fast_agent.core.fastagent import FastAgent

fast = FastAgent("Server Agent")

@fast.agent(instruction="You are an API agent")
async def main():
    # Start as a server programmatically
    await fast.start_server(
        transport="sse",
        host="0.0.0.0",
        port=8080,
        server_name="API-Agent-Server",
        server_description="Provides API access to my agent"
    )

if __name__ == "__main__":
    asyncio.run(main())
```

`--transport` now implies server mode when running a Python module directly. The legacy `--server` flag remains as an alias but is deprecated.

## Python Program Integration

Embed **fast-agent** into existing Python applications to add MCP agent capabilities.

```
import asyncio
from fast_agent.core.fastagent import FastAgent

fast = FastAgent("Embedded Agent")

@fast.agent(instruction="You are a data analysis assistant")
async def analyze_data(data):
    async with fast.run() as agent:
        result = await agent.send(f"Analyze this data: {data}")
        return result

# Use in your application
async def main():
    user_data = get_user_data()
    analysis = await analyze_data(user_data)
    display_results(analysis)

if __name__ == "__main__":
    asyncio.run(main())
```

Back to top

## Agent Skills Overview
[Skip to content](https://fast-agent.ai/agents/skills/#agent-skills)

# Agent Skills

**`fast-agent`** supports Agent Skills, and looks for them in either the `.fast-agent/skills` or `.claude/skills` folder.

When valid SKILL.md files are found:

- The Agent is given access to an `execute` tool for running shell commands, with the working directory set to the skills folder.
- Skill descriptions from the manifest and path are added to the System Prompt using the `{{agentSkills}}` expansion. A warning is displayed if this is not present in the System Prompt.
- The `/skills` command lists the available skills.

## Command Line Options

If using **`fast-agent`** interactively from the command line, the `--skills <directory>` switch can be used to specify the location of SKILL.md files.

```
# Specify a skills folder and a model
fast-agent --skills ~/skill-development/testing/ --model gpt-5-mini.low

# Give fast-agent access to the shell
fast-agent -x
```

## Programmatic Usage

Skills directories can be defined on a per-agent basis:

```
# Define the agent
@fast.agent(instruction=default_instruction,skills=["~/source/skills"])
async def main():
    # use the --model command line switch or agent arguments to change model
    async with fast.run() as agent:
        await agent.interactive()
```

This allows each individual agent to use a different set of skills if needed.

Back to top

## Fast-Agent Installation Guide
# Welcome Guide

Getting started with **`fast-agent`** is easy. First, make sure that you have [`uv`](https://docs.astral.sh/uv/) installed and then install **`fast-agent`** with:

```
uv tool install fast-agent-mcp -U
```

You can then run

```
and running in minutes
```

Back to top

## MCP Server Configuration Guide
[Skip to content](https://fast-agent.ai/mcp/#adding-a-stdio-server)

# Configuring Servers

MCP Servers are configured in the `fastagent.config.yaml` file. Secrets can be kept in `fastagent.secrets.yaml`, which follows the same format ( **fast-agent** merges the contents of the two files).

## Adding a STDIO Server

The below shows an example of configuring an MCP Server named `server_one`.

fastagent.config.yaml

```
mcp:
# name used in agent servers array
  server_one:
    # command to run
    command: "npx"
    # list of arguments for the command
    args: ["@modelcontextprotocol/server-brave-search"]
    # key/value pairs of environment variables
    env:
      BRAVE_API_KEY: your_key
      KEY: value
  server_two:
    # and so on ...
```

This MCP Server can then be used with an agent as follows:

```
@fast.agent(name="Search", servers=["server_one"])
```

## Adding an SSE or HTTP Server

To use remote MCP Servers, specify the either `http` or `sse` transport and the endpoint URL and headers:

fastagent.config.yaml

```
mcp:
# name used in agent servers array
  server_two:
    transport: "http"
    # url to connect
    url: "http://localhost:8000/mcp"
    # timeout in seconds to use for sse sessions (optional)
    read_transport_sse_timeout_seconds: 300
    # request headers for connection
    headers:
          Authorization: "Bearer <secret>"

# name used in agent servers array
  server_three:
    transport: "sse"
    # url to connect
    url: "http://localhost:8001/sse"
```

## MCP Filtering

Agents and Workflows supporting the `servers` parameter have the ability to filter the tools, resources and prompts available to the agent. This can greatly reduce the amount of context generated for the agents - which can both increase the accuracy of the responses and reduce costs due to the lower token count of the context.

The default behavior is to include all tools, prompts and resources from the configured MCP servers, but this can be overridden by the `tools`, `prompts` and `resources` parameters. These parameters accept a Dict, where the key of the dict in the name of the server to filter, and the value is a list of the tool names, resource names and prompt names respectively.

For example:

```
@fast.agent(
  name="Search,
  instruction="You are a search agent that helps users fint files using the provided tools.",
  servers=["server_one", "server_two"]  # use two MCP servers

  # Filter some of the MCP resources avalable to the agent
  tools={
    "server_one": ["search_files", "search_directory"],
    "server_two": ["regex_search"]
  }
  prompts = None  # DOn't filter prompts (default behavior)
  resources = {
    "server_two": ["file://get_tree"] # Only filter resources on server_two
  }
)
```

## Implementation Spoofing

**`fast-agent`** can be used the specify the Implementation details sent to the MCP Server, enabling testing Servers that adapt their configuration based on the client connection. By default **`fast-agent`** uses the `fast-agent-mcp` and it's current version number.

fastagent.config.yaml

```
mcp:
  server_one:
    transport: "http"
    url: "http://localhost:8000/mcp"
    implementation:
      name: "spoof-server"
      version: "9.9.9"
```

## Roots

**fast-agent** supports MCP Roots. Roots are configured on a per-server basis:

fastagent.config.yaml

```
mcp:
  server_three:
    transport: "http"
    url: "http://localhost:8000/mcp"
    roots:
       uri: "file://...."
       name: Optional Name
       server_uri_alias: # optional
```

As per the [MCP specification](https://github.com/modelcontextprotocol/specification/blob/41749db0c4c95b97b99dc056a403cf86e7f3bc76/schema/2025-03-26/schema.ts#L1185-L1191) roots MUST be a valid URI starting with `file://`.

If a server\_uri\_alias is supplied, **fast-agent** presents this to the MCP Server. This allows you to present a consistent interface to the MCP Server. An example of this usage would be mounting a local directory to a docker volume, and presenting it as `/mnt/data` to the MCP Server for consistency.

The data analysis example (`fast-agent quickstart data-analysis` has a working example of MCP Roots).

## Sampling

Sampling is configured by specifying a sampling model for the MCP Server.

fastagent.config.yaml

```
mcp:
  server_four:
    transport: "http"
    url: "http://localhost:8000/mcp"
    sampling:
      model: "provider.model.<reasoning_effort>"
```

Read more about The model string and settings [here](https://fast-agent.ai/models/). Sampling requests support vision - try [`@llmindset/mcp-webcam`](https://github.com/evalstate/mcp-webcam) for an example.

## Elicitations

Elicitations are configured by specifying a strategy for the MCP Server. The handler can be overriden with a custom handler in the Agent definition.

fastagent.config.yaml

```
mcp:
  server_four:
    transport: "http"
    url: "http://localhost:8000/mcp"
    elicitation:
      mode: "forms"
```

`mode` can be one of:

- **`forms`** (default). Displays a form to respond to elicitations.
- **`auto_cancel`** The elicitation capability is advertised to the Server, but all solicitations are automatically cancelled.
- **`none`** No elicitation capability is advertised to the Server.

Back to top

## MCP Elicitations Overview
[Skip to content](https://fast-agent.ai/mcp/elicitations/#quick-start-mcp-elicitations)

# Quick Start: MCP Elicitations

In this quick start, we'll demonstrate **fast-agent**'s [MCP Elicitation](https://modelcontextprotocol.io/specification/2025-06-18/client/elicitation) features.

![Elicitation Form](https://fast-agent.ai/mcp/pics/elicitation_form.gif)

Elicitations allow MCP Servers to request additional input directly from Users.

This demo comprises three MCP Servers and three **fast-agent** programs:

- An interactive demonstration showing all types of Forms, Fields and Validation in the specification.
- A demonstration of an Elicitation made during a Tool Call.
- An example of using a custom Elicitation handler.

This quick start gives provides you with a complete MCP Client and Server solution for developing and deploying Elicitations.

## Setup **fast-agent**

Make sure you have the `uv` [package manager](https://docs.astral.sh/uv/) installed, and open a terminal window. Then:

[Linux/MacOS](https://fast-agent.ai/mcp/elicitations/#__tabbed_1_1)[Windows](https://fast-agent.ai/mcp/elicitations/#__tabbed_1_2)

```
# create, and change to a new directory
mkdir fast-agent && cd fast-agent

# create and activate a python environment
uv venv
source .venv/bin/activate

# setup fast-agent
uv pip install fast-agent-mcp

# setup the elicitations demo
fast-agent quickstart elicitations

# go the demo folder
cd elicitations
```

```
# create, and change to a new directory
md fast-agent |cd

# create and activate a python environment
uv venv
.venv\Scripts\activate

# setup fast-agent
uv pip install fast-agent-mcp

# setup the elicitations demo
fast-agent quickstart elicitations

# go the demo folder
cd elicitations
```

You are now ready to start the demos.

## Elicitation Requests and Forms

The Interactive Forms demo showcases all of the Elicitation data types and validations. Start the interactive form demo with:

```
uv run forms_demo.py
```

This demonstration displays 4 different elicitation forms in sequence.

Note that the forms:

- Can be navigated with the `Tab` or Arrow Keys (`→\←`)
- Have real time Validation
- Can be Cancelled with the Escape key
- Uses multiline text input for long fields
- Identify the Agent and MCP Server that produced the request.

![Elicitation Form](https://fast-agent.ai/mcp/pics/elicitation_form_sm.png)

The `Cancel All` option cancels the Elicitation Request, and automatically cancels future requests to avoid unwanted interruptions from badly behaving Servers.

For MCP Server developers, the form is fast and easy to navigate to facilitating iterative development.

The `elicitation_forms_server.py` file includes examples of all field types and validations: `Numbers`, `Booleans`, `Enums` and `Strings`.

It also supports the formats specified in the [schema](https://github.com/modelcontextprotocol/modelcontextprotocol/blob/b98f9805e963af7f67f158bdfa760078be4675a3/schema/2025-06-18/schema.ts#L1335-L1342): `Email`, `Uri`, `Date` and `Date/Time`.

## Tool Call

The Tool Call demo demonstrates an Elicitation being conducted during an MCP Tool Call. This also showcases a couple of **fast-agent** features:

- The `passthrough` model supports testing without an LLM. You can read more about Internal Models [here](https://fast-agent.ai/models/internal_models/).
- Calling a tool by sending a `***CALL_TOOL` message, that enables an Agent to directly call an MCP Server Tool with specific arguments.

Run `uv run tool_call.py` to run the Agent and see the elicitation. You can use a real LLM with the `--model` switch.

## Custom Handler

This example shows how to write and integrate a custom Elicitation handler. For this example, the agent uses a custom handler to generate a character for a game. To run:

```
uv run game_character.py
```

![Custom Elicitation](https://fast-agent.ai/mcp/pics/elicitation_char3.gif)

This agent uses a custom elicitation handler to generate a character for a game. The custom handler is in `game_character_handler.py` and is setup with the following code:

| game\_character.py |
| --- |
| ```<br>23<br>24<br>25<br>26<br>27<br>28<br>``` | ```<br>@fast.agent(<br>    "character-creator",<br>    servers=["elicitation_forms_server"],<br>    # Register our handler from game_character_handler.py<br>    elicitation_handler=game_character_elicitation_handler,<br>)<br>``` |

For MCP Server Developers, Custom Handlers can be used to help complete automated test flows. For Production use, Custom Handlers can be used to send notifications or request input via remote platforms such as web forms.

## Configuration

Note that Elicitations are now _enabled by default_ in **fast-agent**, and can be [configured with](https://fast-agent.ai/mcp/#elicitations) the `fastagent.config.yaml` file.

You can configure the Elicitation mode to `forms` (the default),`auto-cancel` or `none`.

| fastagent.config.yaml |
| --- |
| ```<br>19<br>20<br>21<br>22<br>23<br>24<br>25<br>26<br>27<br>28<br>``` | ```<br>mcp:<br>  servers:<br>    # Elicitation test servers for different modes<br>    elicitation_forms_mode:<br>      command: "uv"<br>      args: ["run", "elicitation_test_server_advanced.py"]<br>      transport: "stdio"<br>      cwd: "."<br>      elicitation:<br>        mode: "forms"<br>``` |

In `auto-cancel` mode, **fast-agent** advertises the Elicitation capability, and automatically cancels Elicitation requests from the MCP Server.

When set to `none`, the Elicitation capability is not advertised to the MCP Server.

Back to top

## MCP Server OAuth
[Skip to content](https://fast-agent.ai/mcp/mcp-oauth/#requirements)

# MCP Server OAuth

Adds OAuth v2.1 to HTTP/SSE MCP servers (STDIO excluded).

- Uses PKCE and prints a clickable authorization link (no auto‑open).
- Persists tokens in the OS keychain (via keyring) by default; falls back to memory if no keychain is available.

## Requirements

- **`fast-agent`** 0.3.5 or above
- OS Keyring support for persistence (e.g. WinVaultKeyring, macOS Keyring, SercretService Keyring)

Install keyring on Ubuntu

```
sudo apt-get install gnome-keyring seahorse
```

## Identity Model

- Tokens are keyed by the resource server’s base URL, not by server name.
- Base URL = normalize the server URL by removing a trailing "/mcp" or "/sse" and ignoring query/fragment.
- Renaming a server in config won’t affect tokens; changing the URL maps to a different identity.

## Minimal Config

OAuth is on by default for HTTP/SSE servers. Per‑server configuration:

```
mcp:
  servers:
    myserver:
      transport: http                    # (optional, defaults to http) or sse
      url: http://localhost:8001/mcp     # use /sse for SSE
      auth:
        oauth: true                      # default true
        persist: keyring                 # default keyring; use memory to disable persistence
        redirect_port: 3030              # default 3030
        redirect_path: /callback         # default /callback
        # scope: "user"                  # optional (server defaults used if omitted)
```

Notes:

- Scope is omitted by default. If a server requires a specific scope, set `auth.scope` (string or list).
- STDIO servers do not use OAuth and are hidden in auth views.

## Keychain Persistence

- Default: tokens go to your OS keychain (macOS Keychain, Windows Credential Manager, Linux Secret Service/KWallet).
- If a keychain backend is not available, tokens are kept in memory for the session (no disk writes).
- Linux: ensure a Secret Service (gnome‑keyring) or KWallet is installed and running if you want persistence.

## CLI Quick Reference

- Show auth status (keyring backend, stored identities, configured servers → identities)
- `fast-agent auth`
- `fast-agent auth status`
- Single target:
  - `fast-agent auth status https://example-server.modelcontextprotocol.io`
  - `fast-agent auth status myserver`
- Proactive login (perform OAuth and store tokens)

- By server name in config:
  - `fast-agent auth login myserver`
- By identity (ad hoc, no config):
  - HTTP (default): `fast-agent auth login https://example-server.modelcontextprotocol.io`
  - SSE: `fast-agent auth login https://example-server.modelcontextprotocol.io --transport sse`
- Clear tokens

- By identity (base URL): `fast-agent auth clear --identity https://example-server.modelcontextprotocol.io`
- By server name (from config): `fast-agent auth clear myserver`
- All identities: `fast-agent auth clear --all`

- Check full app config (includes server OAuth flags and token presence):

- `fast-agent check`

## Typical Workflows

- Connect normally; authenticate on demand
- `fast-agent --url "https://huggingface.co/mcp?login"`
- When a server requires OAuth, the CLI prints a clickable link.
- A local callback server (`http://localhost:3030/callback`) captures the code; if the port is blocked, you’ll be prompted to paste the callback URL.

- Proactive login (no agent session needed)

- `fast-agent auth login https://example-server.modelcontextprotocol.io`
- Complete the link flow once; tokens will be reused next time.

- Inspect and clear a specific identity

- `fast-agent auth status https://example-server.modelcontextprotocol.io`
- `fast-agent auth clear --identity https://example-server.modelcontextprotocol.io`

## Troubleshooting

- Immediate 401 with no link
- Ensure you are running the updated CLI (editable install or latest tool).
- Some servers require explicit scope; add `auth.scope` to that server in `fastagent.config.yaml`.

- Link opens but no callback received

- Confirm `http://localhost:3030/callback` is reachable (firewall/port in use).
- If blocked, paste the returned callback URL when prompted in the terminal.

- Keychain not persisting tokens (Linux)

- Install and run a Secret Service (gnome‑keyring) or KWallet.
- Otherwise, tokens are in-memory only.

- Authorization header conflicts

- When OAuth is enabled on a server, fast‑agent removes any preconfigured `Authorization`/`X‑HF‑Authorization` headers for that server’s transport so OAuth can proceed cleanly.

- STDIO not listed

- Expected; STDIO transport does not use OAuth.

Back to top

## MCP Server Deployment
[Skip to content](https://fast-agent.ai/mcp/mcp-server/#running-as-an-mcp-server)

# Deploying as an MCP Server

### Running as an MCP Server

**`fast-agent`** Can deploy any configured agents over MCP, letting external MCP clients connect via STDIO, SSE, or HTTP.

Additionally, there is a convenient `serve` command enabling rapid, command line deployment of MCP enabled agents in a variety of instancing modes.

This feature also works with [Agent Skills](https://fast-agent.ai/agents/skills/), enabling powerful adaptable behaviours.

#### Using the CLI (fast-agent serve)

```
fast-agent serve [OPTIONS]
```

Key options:

- `--transport [http|sse|stdio]` (default http)
- `--port / --host` (for HTTP/SSE)
- `--instance-scope [shared|connection|request]`– choose how agent state is isolated
  - `shared` (default) reuses a single agent for all clients
  - `connection` (sessions) Create one Agent per MCP session (separate history per client)
  - `request` (stateless) - create a new Agent for every tool call and disable MCP Sessions
- `--description` – Customise the MCP tool description (supports {agent} placeholder)

Standard CLI flags also apply (e.g. --config-path, --model, --servers, --stdio,
--quiet). This allows the **`fast-agent`** to serve any existing MCP Server in "Agent Mode", use custom system prompts and so on.

Examples:

```
fast-agent serve \
--url https://huggingface.co/mcp \
--instance-scope connection \
--description "Interact with the {agent} workflow" \
--model haiku
```

This starts a Streamable HTTP MCP Server on port 8000, providing access to an Agent connected to the Hugging Face MCP Server using Anthropic Haiku.

```
fast-agent serve \
--npx @modelcontextprotocol/server-everything \
--instance-scope request \
--description "Ask me anything!" \
-i system_prompt.md
--model kimi
```

This starts a Streamable HTTP MCP Server on port 8000, providing agent access to the STDIO version of the "Everything Server" with a custom system prompt.

#### Running an agent

If you already have an agent module or workflow (e.g. the generated agent.py), you can start it as a server directly:

```
uv run agent.py --transport http [OPTIONS]
```

The embedded CLI parser supports the same server flags as the serve command:

- `--transport`, `--host`, `--port`
- `--instance-scope [shared|connection|request]`
- `--description` (tool instructions)
- `--quiet`, `--model`, and other agent startup options

Example:

```
uv run agent.py \
--transport http \
--port 8723 \
--instance-scope request
```

`--transport` now enables server mode automatically. The legacy `--server` flag is still accepted as an alias but is deprecated.

Both approaches initialise FastAgent with the same config and skill loading pipeline;
choose whichever fits your workflow (one-off CLI invocation vs. packaging an agent as
a reusable script).

Back to top

## Fast-Agent and MCP-UI
[Skip to content](https://fast-agent.ai/mcp/mcp-ui/#using-mcp-ui-and-fast-agent)

# mcp-ui and fast-agent

## Using mcp-ui and `fast-agent`

**`fast-agent`** supports [mcp-ui](https://mcpui.dev/) embedded components, and makes them accessible for usage and testing.

## Installing `fast-agent`

To install **`fast-agent`**, first download and install the [`uv`](https://docs.astral.sh/uv/) package manager.

Next, install (or upgrade) with:

```
uv tool install -U fast-agent-mcp
```

Next, configure your API Keys. This guide assumes that you have `OPENAI_API_KEY` and `ANTHROPIC_API_KEY` environment variables set.

Check your installation by running the Model Context Protocol everything server with:

```
fast-agent --npx @modelcontextprotocol/server-everything
```

Use the `fast-agent check` command to diagnose any issues.

## Using `mcp-ui`

Download the mcp-ui examples, and start the TypeScript [demo server](https://github.com/idosal/mcp-ui/blob/main/examples/typescript-server-demo/README.md):

To connect to the demo server with `gpt-5-mini` with low reasoning effort use:

```
fast-agent --url http://localhost:3000 --model=gpt-5-mini.low
```

**`fast-agent`** presents the mcp-ui content as links beneath the assistant message. HTML components are stored in the `.fast-agent/ui` directory.

If you want to test multiple models in parallel - for example to compare behaviour - you can specify more than one model and run in parallel:

```
# run the test server with both gpt-5-mini and sonnet
fast-agent --url http://localhost:3000 --model=gpt-5-mini.low,sonnet
```

To run with a prompt and exit:

```
fast-agent --url http://localhost:3000 --model=haiku -m "run all three tools"
```

If you want to pass authorization headers, you use the --auth option:

```
fast-agent --url https://huggingface.co/mcp --auth $HF_TOKEN
```

## Advanced Configuration

To create configuration files for advanced configuration, use the `fast-agent setup` command.

This allows you to configure some `mcp-ui` settings, or configure servers to use custom headers.

The following options are available:

fastagent.config.yaml

```
# mcp-ui config options

# Where to write MCP-UI HTML files (relative to CWD if not absolute)
mcp_ui_output_dir: ".fast-agent/ui"

# "disabled", "enabled" or "auto" to automatically open links in browser
mcp_ui_mode: enabled

mcp:
  servers:
      example:
        transport: http
        url: https://huggingface.co/mcp
        ## custom headers below
        headers:
          custom_header: value
```

## Client Spoofing

Some MCP Servers adjust their tools or behaviour based on the connecting client (for exampling enabling mcp-ui). You can specify the name and version to present to the MCP Server:

fastagent.config.yaml

```
  servers:
      example:
        transport: http
        url: https://huggingface.co/mcp
        implementation:
            name: claude-code
            version: 1.0.99
```

Back to top

## MCP Server Overview
[Skip to content](https://fast-agent.ai/mcp/mcp_display/#section-1-implementation-and-session)

# Inspecting Servers

Detailed information about the MCP Server connection can be displayed with the `/mcp` command.

![](https://fast-agent.ai/mcp/pics/mcp_transport_display.png)

### Section 1 - Implementation and Session

This section shows the MCP Server Implementation Details (`Name` and `Version`), along with any `Mcp-Session-Id` allocated by the MCP Server.

### Section 2 - Transport Channel History

Shows activity from the Streamable HTTP GET and POST handlers for the MCP Server.

### Section 3 - Server Capabilities

- `To`, `Pr`, `Re`: Tools, Prompts and Resources. Green for available, Yellow for List Change notifications.
- `Rs`: Resource Subscriptions.
- `Lo`, `Co`: Logging and Completions.
- `Ex`: Experimental Capabilities
- `In`: Instructions. Green for available, and used - Yellow for available but not in Prompt, Red for available, but disabled.

### Section 4 - Client Capabilities.

- `Ro`: Roots offered to MCP Server.
- `El`: Elicitation offered to MCP Server. Red for `Cancel All` mode.
- `Sa`: Sampling offered to MCP Server. Green for auto, Yellow for manually configured.
- `Sp`: MCP Client Name has been spoofed.

### Configuration

The activity timeline shown in the transport section can be tailored in `fastagent.config.yaml` via the `mcp_timeline` block:

```
mcp_timeline:
  steps: 20         # number of buckets rendered on the timeline
  step_seconds: 30  # duration of each bucket (supports values like "45s" or "2m")
```

These values flow through to both `fast-agent check` and the in-session `/mcp` display. When multiple events occur in the same bucket, higher priority states replace lower ones using this order: `error` → `disabled/request` → `response` → `notification/ping` → `none`. This keeps significant events (such as errors and requests) visible even if a subsequent ping lands in the same interval.

Back to top

## OpenAI Apps SDK
[Skip to content](https://fast-agent.ai/mcp/openai-apps-sdk/#overview)

# OpenAI Apps SDK

## Overview

**`fast-agent`** automatically detects [OpenAI Apps SDK (Skybridge)](https://developers.openai.com/apps-sdk) integrations exposed by MCP servers. Detection runs during tool/resource discovery: the aggregator looks for tools that publish an `openai/outputTemplate``_meta` entry and the corresponding `ui://…` resources with the `text/html+skybridge` MIME type.

## What `fast-agent` checks

- **Template metadata** – verifies that tool `_meta["openai/outputTemplate"]` values are valid URIs. Invalid entries raise warnings so they are easy to spot.
- **Resource availability** – ensures the referenced `ui://` resource exists. Missing resources generate warnings and keep the tool flagged as invalid.
- **MIME-type validation** – confirms the resource exposes `text/html+skybridge`. Non-matching MIME types surface warnings and prevent the tool from being marked as Skybridge-enabled.
- **Unpaired resources** – highlights confirmed Skybridge resources that no tool references, so server authors can wire them up.

All warnings are captured in the `SkybridgeServerConfig.warnings` list, making it straightforward to assert against them in tests or custom diagnostics.

## Console Summary

Right after discovery, the console displays a concise Skybridge summary:

- Lists servers with Skybridge signals, annotating how many enabled tools and valid resources were found.
- Surfaces aggregated warnings (such as invalid MIME types or missing references).
- Provides quick feedback about potential configuration issues before any tool runs.

![](https://fast-agent.ai/mcp/pics/skybridge_summary.png)

## Tool Call Display

When a Skybridge-enabled tool returns structured content, the tool result view adds a magenta separator that references the linked `ui://` resource. This makes it clear to developers which HTML payload is expected to render in the OpenAI Apps SDK client.

![](https://fast-agent.ai/mcp/pics/skybridge_tool.png)

## Accessing Skybridge Configurations Programmatically

Developers can inspect discovered configurations at runtime:

```
configs = await agent._aggregator.get_skybridge_configs()
hf_config = await agent._aggregator.get_skybridge_config("huggingface")
```

Each `SkybridgeServerConfig` entry includes resources, tools, and warnings so you can write assertions against servers you are developing.

## Feature Gating / Client Spoofing

Some MCP servers gate Skybridge resources based on the connecting client’s implementation string. If you need to imitate the official Apps SDK, configure Fast Agent’s spoofing settings as described in the [Client Spoofing](https://fast-agent.ai/mcp/mcp-ui/#client-spoofing) section. This lets you present a custom `implementation.name`/`version` pair while still benefiting from Skybridge validation and display.

Back to top

## MCP Development Resources
# Resources

Below are some recommended resources for developing with the Model Context Protocol (MCP):

| Resource | Description |
| --- | --- |
| [Working with Files and Resources](https://llmindset.co.uk/posts/2025/01/mcp-files-resources-part1/) | Examining the options MCP Server and Host developers have for sharing rich content |
| [PulseMCP Community](https://www.pulsemcp.com/) | A community focussed site offering news, up-to-date directories and use-cases of MCP Servers |
| [Basic Memory](https://memory.basicmachines.co/docs/introduction) | High quality, markdown based knowledge base for LLMs - also good for Agent development |
| [Pulse Fetch](https://github.com/pulsemcp/mcp-servers/tree/main/productionized/pulse-fetch) | Comprehensive, Reliable Web Scraping and Extraction. |
| [Repomix](https://repomix.com/guide/) | Create LLM Friendly files from folders or directly from GitHub. Include as an MCP Server - or run from a script prior to create Agent inputs |
| [PromptMesh Tools](https://promptmesh.io/) | High quality tools and libraries at the cutting edge of MCP development |
| [wong2 mcp-cli](https://github.com/wong2/mcp-cli) | A fast, lightweight, command line alternative to the official MCP Inspector |

Back to top

## State Transfer with MCP
[Skip to content](https://fast-agent.ai/mcp/state_transfer/#quick-start-state-transfer-with-mcp)

# Quick Start: State Transfer with MCP

In this quick start, we'll demonstrate how **fast-agent** can transfer state between two agents using MCP Prompts.

![Welcome Image](https://fast-agent.ai/mcp/pics/opening_small.png)

First, we'll start `agent_one` as an MCP Server, and send it some messages with the MCP Inspector tool.

Next, we'll run `agent_two` and transfer the conversation from `agent_one` using an MCP Prompt.

Finally, we'll take a look at **fast-agent**'s `prompt-server` and how it can assist building agent applications

You'll need API Keys to connect to a [supported model](https://fast-agent.ai/models/llm_providers/), or use Ollama's [OpenAI compatibility](https://github.com/ollama/ollama/blob/main/docs/openai.md) mode to use local models.

The quick start also uses the MCP Inspector - check [here](https://modelcontextprotocol.io/docs/tools/inspector) for installation instructions.

## Step 1: Setup **fast-agent**

[Linux/MacOS](https://fast-agent.ai/mcp/state_transfer/#__tabbed_1_1)[Windows](https://fast-agent.ai/mcp/state_transfer/#__tabbed_1_2)

```
# create, and change to a new directory
mkdir fast-agent && cd fast-agent

# create and activate a python environment
uv venv
source .venv/bin/activate

# setup fast-agent
uv pip install fast-agent-mcp

# create the state transfer example
fast-agent quickstart state-transfer
```

```
# create, and change to a new directory
md fast-agent |cd

# create and activate a python environment
uv venv
.venv\Scripts\activate

# setup fast-agent
uv pip install fast-agent-mcp

# create the state transfer example
fast-agent quickstart state-transfer
```

Change to the state-transfer directory (`cd state-transfer`), rename `fastagent.secrets.yaml.example` to `fastagent.secrets.yaml` and enter the API Keys for the providers you wish to use.

The supplied `fastagent.config.yaml` file contains a default of `gpt-4.1` \- edit this if you wish.

Finally, run `uv run agent_one.py` and send a test message to make sure that everything working. Enter `stop` to return to the command line.

![Testing the Agent](https://fast-agent.ai/mcp/pics/test_message.png)

## Step 2: Run **agent one** as an MCP Server

To start `"agent_one"` as an MCP Server, run the following command:

[Linux/MacOS](https://fast-agent.ai/mcp/state_transfer/#__tabbed_2_1)[Windows](https://fast-agent.ai/mcp/state_transfer/#__tabbed_2_2)

```
# start agent_one as an MCP Server:
uv run agent_one.py --transport http --port 8001
```

```
# start agent_one as an MCP Server:
uv run agent_one.py --transport http --port 8001
```

The agent is now available as an MCP Server.

Note

This example starts the server on port 8001. To use a different port, update the URLs in `fastagent.config.yaml` and the MCP Inspector.

## Step 3: Connect and chat with **agent one**

From another command line, run the Model Context Protocol inspector to connect to the agent:

[Linux/MacOS](https://fast-agent.ai/mcp/state_transfer/#__tabbed_3_1)[Windows](https://fast-agent.ai/mcp/state_transfer/#__tabbed_3_2)

```
# run the MCP inspector
npx @modelcontextprotocol/inspector
```

```
# run the MCP inspector
npx @modelcontextprotocol/inspector
```

Choose the "Streamable HTTP" transport type, and the url `http://localhost:8001/mcp`. After clicking the `connect` button, you can interact with the agent from the `tools` tab. Use the `agent_one_send` tool to send the agent a chat message and see it's response.

![Using the Inspector to Chat](https://fast-agent.ai/mcp/pics/inspector_chat.png)

The conversation history can be viewed from the `prompts` tab. Use the `agent_one_history` prompt to view it.

Disconnect the Inspector, then press `ctrl+c` in the command window to stop the process.

## Step 4: Transfer the conversation to **agent two**

We can now transfer and continue the conversation with `agent_two`.

Run `agent_two` with the following command:

[Linux/MacOS](https://fast-agent.ai/mcp/state_transfer/#__tabbed_4_1)[Windows](https://fast-agent.ai/mcp/state_transfer/#__tabbed_4_2)

```
# start agent_two as an MCP Server:
uv run agent_two.py
```

```
# start agent_two as an MCP Server:
uv run agent_two.py
```

Once started, type `'/prompts'` to see the available prompts. Select `1` to apply the Prompt from `agent_one` to `agent_two`, transferring the conversation context.

You can now continue the chat with `agent_two` (potentially using different Models, MCP Tools or Workflow components).

![Transferred Chat](https://fast-agent.ai/mcp/pics/loaded_chat.png)

### Configuration Overview

**fast-agent** uses the following configuration file to connect to the `agent_one` MCP Server:

fastagent.config.yaml

```
# MCP Servers
mcp:
    servers:
        agent_one:
          transport: http
          url: http://localhost:8001/mcp
```

`agent_two` then references the server in it's definition:

| agent\_two.py |
| --- |
| ```<br>10<br>11<br>12<br>13<br>14<br>15<br>16<br>17<br>18<br>``` | ```<br># Define the agent<br>@fast.agent(name="agent_two",<br>            instruction="You are a helpful AI Agent",<br>            servers=["agent_one"])<br>async def main():<br>    # use the --model command line switch or agent arguments to change model<br>    async with fast.run() as agent:<br>        await agent.interactive()<br>``` |

## Step 5: Save/Reload the conversation

**fast-agent** gives you the ability to save and reload conversations.

Enter `***SAVE_HISTORY history.json` in the `agent_two` chat to save the conversation history in MCP `GetPromptResult` format.

You can also save it in a text format for easier editing.

![Prompt Picker](https://fast-agent.ai/mcp/pics/prompt_picker.png)

By using the supplied MCP `prompt-server`, we can reload the saved prompt and apply it to our agent. Add the following to your `fastagent.config.yaml` file:

| fastagent.config.yaml |
| --- |
| ```<br>23<br>24<br>25<br>26<br>27<br>28<br>29<br>30<br>31<br>``` | ```<br># MCP Servers<br>mcp:<br>    servers:<br>        prompts:<br>            command: prompt-server<br>            args: ["history.json"]<br>        agent_one:<br>          transport: http<br>          url: http://localhost:8001/mcp<br>``` |

And then update `agent_two.py` to use the new server:

| agent\_two.py |
| --- |
| ```<br>10<br>11<br>12<br>13<br>``` | ```<br># Define the agent<br>@fast.agent(name="agent_two",<br>            instruction="You are a helpful AI Agent",<br>            servers=["prompts"])<br>``` |

Run `uv run agent_two.py`, and you can then use the `/prompts` command to load the earlier conversation history, and continue where you left off.

Note that Prompts can contain any of the MCP Content types, so Images, Audio and other Embedded Resources can be included.

You can also use the [Playback LLM](https://fast-agent.ai/models/internal_models/) to replay an earlier chat (useful for testing!)

Back to top

## MCP Types Integration
[Skip to content](https://fast-agent.ai/mcp/types/#integration-with-mcp-types)

# Integration with MCP Types

## MCP Type Compatibility

FastAgent is built to seamlessly integrate with the MCP SDK type system:

Conversations with assistants are based on `PromptMessageExtended` \- an extension the the mcp `PromptMessage` type, with support for multiple content sections. This type is expected to become native in a future version of MCP: https://github.com/modelcontextprotocol/specification/pull/198

## Message History Transfer

FastAgent makes it easy to transfer conversation history between agents:

history\_transfer.py

```
@fast.agent(name="haiku", model="haiku")
@fast.agent(name="openai", model="o3-mini.medium")

async def main() -> None:
    async with fast.run() as agent:
        # Start an interactive session with "haiku"
        await agent.prompt(agent_name="haiku")
        # Transfer the message history top "openai" (using PromptMessageExtended)
        await agent.openai.generate(agent.haiku.message_history)
        # Continue the conversation
        await agent.prompt(agent_name="openai")
```

Back to top

## Model Features Overview
[Skip to content](https://fast-agent.ai/models/#model-features-and-history-saving)

# Model Features and History Saving

Models in **fast-agent** are specified with a model string, that takes the format `provider.model_name.<reasoning_effort>`

### Precedence

Model specifications in fast-agent follow this precedence order (highest to lowest):

1. Explicitly set in agent decorators
2. Command line arguments with `--model` flag
3. Default model in `fastagent.config.yaml`

### Format

Model strings follow this format: `provider.model_name.reasoning_effort`

- **provider**: The LLM provider (e.g., `anthropic`, `openai`, `azure`, `deepseek`, `generic`,`openrouter`, `tensorzero`)
- **model\_name**: The specific model to use in API calls (for Azure, this is your deployment name)
- **reasoning\_effort** (optional): Controls the reasoning effort for supported models

Examples:

- `anthropic.claude-3-7-sonnet-latest`
- `openai.gpt-4o`
- `openai.o3-mini.high`
- `azure.my-deployment`
- `generic.llama3.2:latest`
- `openrouter.google/gemini-2.5-pro-exp-03-25:free`
- `tensorzero.my_tensorzero_function`

#### Reasoning Effort

For models that support it (`o1`, `o1-preview` and `o3-mini`), you can specify a reasoning effort of **`high`**, **`medium`** or **`low`** \- for example `openai.o3-mini.high`. **`medium`** is the default if not specified.

`gpt-5` additionally supports a `minimal` reasoning effort.

#### Aliases

For convenience, popular models have an alias set such as `gpt-4o` or `sonnet`. These are documented on the [LLM Providers](https://fast-agent.ai/models/llm_providers/) page.

### Default Configuration

You can set a default model for your application in your `fastagent.config.yaml`:

```
default_model: "openai.gpt-4o" # Default model for all agents
```

### History Saving

You can save the conversation history to a file by sending a `***SAVE_HISTORY <filename>` message. This can then be reviewed, edited, loaded, or served with the `prompt-server` or replayed with the `playback` model.

File Format / MCP Serialization

If the filetype is `json`, then messages are serialized/deserialized using the MCP Prompt schema. The `load_prompt`, `load_prompt_multipart` and `prompt-server` will load either the text or JSON format directly.

This can be helpful when developing applications to:

- Save a conversation for editing
- Set up in-context learning
- Produce realistic test scenarios to exercise edge conditions etc. with the [Playback model](https://fast-agent.ai/models/internal_models/#playback)

Back to top

## Internal Models Overview
[Skip to content](https://fast-agent.ai/models/internal_models/#passthrough)

# Internal Models

**fast-agent** comes with two internal models to aid development and testing: `passthrough` and `playback`.

## Passthrough

By default, the `passthrough` model echos messages sent to it.

### Fixed Responses

By sending a `***FIXED_RESPONSE <message>` message, the model will return `<message>` to any request.

### Tool Calling

By sending a `***CALL_TOOL <tool_name> [<json>]` message, the model will call the specified MCP Tool, and return a string containing the results.

## Playback

The `playback` model replays the first conversation sent to it. A typical usage may look like this:

playback.txt

```
---USER
Good morning!
---ASSISTANT
Hello
---USER
Generate some JSON
---ASSISTANT
{
   "city": "London",
   "temperature": 72
}
```

This can then be used with the `prompt-server` you can apply the MCP Prompt to the agent, either programatically with `apply_prompt` or with the `/prompts` command in the interactive shell.

Alternatively, you can load the file with `load_message_multipart`.

JSON contents can be converted to structured outputs:

```
@fast.agent(name="playback",model="playback")

...

playback_messages: List[PromptMessageExtended] = load_message_multipart(Path("playback.txt"))
# Set up the Conversation
assert ("HISTORY LOADED") == agent.playback.generate(playback_messages)

response: str = agent.playback.send("Good morning!") # Returns Hello
temperature, _ = agent.playback.structured("Generate some JSON")
```

When the `playback` runs out of messages, it returns `MESSAGES EXHAUSTED (list size [a]) ([b] overage)`.

List size is the total number of messages originally loaded, overage is the number of requests made after exhaustion.

Back to top

## Azure OpenAI Setup
[Skip to content](https://fast-agent.ai/ref/azure-config/#azure-openai-configuration-example)

# Azure OpenAI Configuration Example

This example shows how to configure fast-agent to use Azure OpenAI Service with different authentication methods.

## Prerequisites

1. An Azure account with access to Azure OpenAI Service
2. An Azure OpenAI Service resource with model deployments
3. The fast-agent package installed with Azure support: `uv pip install fast-agent-mcp[azure]`

## Configuration File

Below is a sample `fastagent.config.yaml` file with all three authentication methods. Choose the one that fits your needs:

```
# OPTION 1: Using resource_name and api_key (standard method)
default_model: "azure.my-deployment"

azure:
  api_key: "YOUR_AZURE_OPENAI_API_KEY"
  resource_name: "your-resource-name"
  azure_deployment: "my-deployment"
  api_version: "2023-05-15"
  # Do NOT include base_url if you use resource_name

# OPTION 2: Using base_url and api_key (custom endpoints or sovereign clouds)
# default_model: "azure.my-deployment"
#
# azure:
#   api_key: "YOUR_AZURE_OPENAI_API_KEY"
#   base_url: "https://your-resource-name.openai.azure.com/"
#   azure_deployment: "my-deployment"
#   api_version: "2023-05-15"
#   # Do NOT include resource_name if you use base_url

# OPTION 3: Using DefaultAzureCredential (for managed identity, Azure CLI, etc.)
# default_model: "azure.my-deployment"
#
# azure:
#   use_default_azure_credential: true
#   base_url: "https://your-resource-name.openai.azure.com/"
#   azure_deployment: "my-deployment"
#   api_version: "2023-05-15"
#   # Do NOT include api_key or resource_name in this mode
```

**Important Configuration Notes:**
\- Use either `resource_name` or `base_url`, not both.
\- When using `DefaultAzureCredential`, do NOT include `api_key` or `resource_name`.
\- When using `base_url`, do NOT include `resource_name`.
\- When using `resource_name`, do NOT include `base_url`.

## Basic Agent Example

Here's a simple agent implementation using Azure OpenAI:

```
import asyncio
from fast_agent.core.fastagent import FastAgent

# Create the application
fast = FastAgent("Azure OpenAI Example")

# Define the agent using Azure OpenAI deployment
@fast.agent(
    instruction="You are a helpful AI assistant powered by Azure OpenAI Service",
    model="azure.my-deployment"
)
async def main():
    async with fast.run() as agent:
        # Start interactive prompt
        await agent()

if __name__ == "__main__":
    asyncio.run(main())
```

## Authentication Notes

### Using DefaultAzureCredential

The DefaultAzureCredential authentication method can use various credential sources:
\- Environment variables
\- Managed identities in Azure
\- Azure CLI credentials
\- Azure PowerShell credentials
\- Visual Studio Code credentials

To use this method:

1. Install the required dependency: `uv pip install fast-agent-mcp[azure]`
2. Configure your environment for Azure authentication (e.g., run `az login`)
3. Use the configuration shown in Option 3 above

This method is ideal for:
\- Deployed applications on Azure (App Service, Functions, AKS, etc.)
\- Development environments where you're already authenticated to Azure
\- Scenarios where secure key management is crucial

### Using API Keys

The API key authentication method is simpler and works in all environments. To find your API key:

1. Go to the Azure Portal
2. Navigate to your Azure OpenAI resource
3. In the "Resource Management" section, select "Keys and Endpoint"
4. Copy one of the keys and the endpoint

Then configure your agent using either Option 1 or Option 2 above.

Back to top

## FastAgent Class Reference
[Skip to content](https://fast-agent.ai/ref/class_reference/#fast-agent-class-reference)

# fast-agent Class Reference

This document provides detailed reference information for programmatically using the `FastAgent` class, which is the core class for creating and running agent applications.

## FastAgent Class

### Constructor

```
FastAgent(
    name: str,
    config_path: str | None = None,
    ignore_unknown_args: bool = False,
    parse_cli_args: bool = True
)
```

#### Parameters

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `name` | `str` | (required) | Name of the application |
| `config_path` | `str \| None` | `None` | Optional path to config file. If not provided, config is loaded from default locations |
| `ignore_unknown_args` | `bool` | `False` | Whether to ignore unknown command line arguments when `parse_cli_args` is `True` |
| `parse_cli_args` | `bool` | `True` | Whether to parse command line arguments. Set to `False` when embedding FastAgent in frameworks like FastAPI/Uvicorn that handle their own argument parsing |

### Decorator Methods

The `FastAgent` class provides several decorators for creating agents and workflows:

| Decorator | Description |
| --- | --- |
| `@fast.agent()` | Create a basic agent |
| `@fast.chain()` | Create a chain workflow |
| `@fast.router()` | Create a router workflow |
| `@fast.parallel()` | Create a parallel workflow |
| `@fast.evaluator_optimizer()` | Create an evaluator-optimizer workflow |
| `@fast.orchestrator()` | Create an orchestrator workflow |

See [Defining Agents](https://fast-agent.ai/agents/defining/) for detailed usage of these decorators.

### Methods

#### `run()`

```
async with fast.run() as agent:
    # Use agent here
```

An async context manager that initializes all registered agents and returns an `AgentApp` instance that can be used to interact with the agents.

#### `start_server()`

```
await fast.start_server(
    transport: str = "sse",
    host: str = "0.0.0.0",
    port: int = 8000,
    server_name: Optional[str] = None,
    server_description: Optional[str] = None
)
```

Starts the application as an MCP server.

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `transport` | `str` | `"sse"` | Transport protocol to use ("stdio" or "sse") |
| `host` | `str` | `"0.0.0.0"` | Host address for the server when using SSE |
| `port` | `int` | `8000` | Port for the server when using SSE |
| `server_name` | `Optional[str]` | `None` | Optional custom name for the MCP server |
| `server_description` | `Optional[str]` | `None` | Optional description for the MCP server |

#### `main()`

```
await fast.main()
```

Helper method for checking if server mode was requested. Returns `True` if server mode was triggered via `--transport` (or the legacy `--server` flag).
`--transport` also implies server mode for direct CLI runs; `--server` remains as a deprecated alias.

## AgentApp Class

The `AgentApp` class is returned from `fast.run()` and provides access to all registered agents and their capabilities.

### Accessing Agents

There are two ways to access agents in the `AgentApp`:

```
# Attribute access
response = await agent.agent_name.send("Hello")

# Dictionary access
response = await agent["agent_name"].send("Hello")
```

### Methods

#### `send()`

```
await agent.send(
    message: Union[str, PromptMessage, PromptMessageExtended],
    agent_name: Optional[str] = None
) -> str
```

Send a message to the specified agent (or the default agent if not specified).

#### `apply_prompt()`

```
await agent.apply_prompt(
    prompt_name: str,
    arguments: Dict[str, str] | None = None,
    agent_name: str | None = None
) -> str
```

Apply a prompt template to an agent (default agent if not specified).

#### `with_resource()`

```
await agent.with_resource(
    prompt_content: Union[str, PromptMessage, PromptMessageExtended],
    resource_uri: str,
    server_name: str | None = None,
    agent_name: str | None = None
) -> str
```

Send a message with an attached MCP resource.

#### `interactive()`

```
await agent.interactive(
    agent: str | None = None,
    default_prompt: str = ""
) -> str
```

Start an interactive prompt session with the specified agent.

## Example: Integrating with FastAPI

See [here](https://github.com/evalstate/fast-agent/tree/main/examples/fastapi) for more examples of using FastAPI with **`fast-agent`**.

fastapi-simple.py

```
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from fast_agent.core.fastagent import FastAgent

# Create FastAgent without parsing CLI args (plays nice with uvicorn)
fast = FastAgent("fast-agent demo", parse_cli_args=False, quiet=True)

# Register a simple default agent via decorator
@fast.agent(name="helper", instruction="You are a helpful AI Agent.", default=True)
async def decorator():
    pass

# Keep FastAgent running for the app lifetime
@asynccontextmanager
async def lifespan(app: FastAPI):
    async with fast.run() as agents:
        app.state.agents = agents
        yield

app = FastAPI(lifespan=lifespan)

class AskRequest(BaseModel):
    message: str

class AskResponse(BaseModel):
    response: str

@app.post("/ask", response_model=AskResponse)
async def ask(req: AskRequest) -> AskResponse:
    try:
        result = await app.state.agents.send(req.message)
        return AskResponse(response=result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
```

## Example: Embedding in a Command-Line Tool

Here's an example of embedding FastAgent in a custom command-line tool:

```
import asyncio
import argparse
import sys
from fast_agent.core.fastagent import FastAgent

# Parse our own arguments first
parser = argparse.ArgumentParser(description="Custom AI Tool")
parser.add_argument("--input", help="Input data for analysis")
parser.add_argument("--format", choices=["text", "json"], default="text", help="Output format")
args, remaining = parser.parse_known_args()

# Create FastAgent with parse_cli_args=False since we're handling our own args
fast = FastAgent("Embedded Agent", parse_cli_args=False)

@fast.agent(instruction="You are a data analysis assistant")
async def analyze():
    async with fast.run() as agent:
        if not args.input:
            print("Error: --input is required")
            sys.exit(1)

        result = await agent.send(f"Analyze this data: {args.input}")

        if args.format == "json":
            import json
            print(json.dumps({"result": result}))
        else:
            print(result)

if __name__ == "__main__":
    asyncio.run(analyze())
```

This example shows how to:
1\. Parse your application's own arguments using `argparse`
2\. Create a FastAgent instance with `parse_cli_args=False`
3\. Use your own command-line arguments in combination with **`fast-agent`**

Back to top

## Fast-Agent Command Options
[Skip to content](https://fast-agent.ai/ref/cmd_switches/#command-line-options)

# Command Line Options

**fast-agent** offers flexible command line options for both running agent applications and using built-in CLI utilities.

## Agent Applications

When running a **fast-agent** application (typically `uv run agent.py`), you have access to the following command line options:

| Option | Description | Example |
| --- | --- | --- |
| `--model MODEL` | Override the default model for the agent | `--model gpt-4o` |
| `--agent AGENT` | Specify which agent to use (default: "default") | `--agent researcher` |
| `-m, --message MESSAGE` | Send a single message to the agent and exit | `--message "Hello world"` |
| `-p, --prompt-file FILE` | Load and apply a prompt file | `--prompt-file conversation.txt` |
| `--quiet` | Disable progress display, tool and message logging | `--quiet` |
| `--version` | Show version and exit | `--version` |
| `--server` | Deprecated alias for server mode; use `--transport` instead | `--server` |
| `--transport {http,sse,stdio}` | Transport protocol; enabling it also turns on server mode | `--transport http` |
| `--port PORT` | Port for SSE server (default: 8000) | `--port 8080` |
| `--host HOST` | Host for SSE server (default: 0.0.0.0) | `--host localhost` |

`--transport` now implies server mode when running a Python module directly. If omitted, it defaults to `http`. `--server` remains available for backward compatibility but will be removed in a future release.

### Examples

```
# Run interactively with specified model
uv run agent.py --model sonnet

# Run specific agent
uv run agent.py --agent researcher

# Run with specific agent and model
uv run agent.py --agent researcher --model gpt-4o

# Send a message to an agent and exit
uv run agent.py --agent summarizer --message "Summarize this document"

# Apply a prompt file
uv run agent.py --prompt-file my_conversation.txt

# Run as an SSE server on port 8080
uv run agent.py --transport sse --port 8080

# Run as a stdio server
uv run agent.py --transport stdio

# Get minimal output (for scripting)
uv run agent.py --quiet --message "Generate a report"
```

### Programmatic Control of Command Line Parsing

When embedding FastAgent in other applications (like web frameworks or GUI applications), you can disable command line parsing by setting `parse_cli_args=False` in the constructor:

```
# Create FastAgent without parsing command line arguments
fast = FastAgent("Embedded Agent", parse_cli_args=False)
```

This is particularly useful when:
\- Integrating with frameworks like FastAPI/Uvicorn that have their own argument parsing
\- Building GUI applications where command line arguments aren't relevant
\- Creating applications with custom argument parsing requirements

## fast-agent go Command

The `fast-agent go` command lets you run an interactive agent directly without creating a Python file. Read the guide [here](https://fast-agent.ai/ref/go_command/)

## fast-agent check Command

Use `fast-agent check` to diagnose your configuration:

```
# Show configuration summary
fast-agent check

# Display configuration file
fast-agent check show

# Display secrets file
fast-agent check show --secrets
```

## fast-agent setup Command

Create a new agent project with configuration files:

```
# Set up in current directory
fast-agent setup

# Set up in a specific directory
fast-agent setup --config-dir ./my-agent

# Force overwrite existing files
fast-agent setup --force
```

## fast-agent quickstart Command

Create example applications to get started quickly:

```
# Show available examples
fast-agent quickstart

# Create workflow examples
fast-agent quickstart workflow .

# Create researcher example
fast-agent quickstart researcher .

# Create data analysis example
fast-agent quickstart data-analysis .

# Create state transfer example
fast-agent quickstart state-transfer .
```

Back to top

## Fast-Agent Configuration Guide
[Skip to content](https://fast-agent.ai/ref/config_file/#configuration-reference)

# Configuration Reference

**fast-agent** can be configured through the `fastagent.config.yaml` file, which should be placed in your project's root directory. For sensitive information, you can use `fastagent.secrets.yaml` with the same structure - values from both files will be merged, with secrets taking precedence.

Configuration can also be provided through environment variables, with the naming pattern `SECTION__SUBSECTION__PROPERTY` (note the double underscores).

## Configuration File Location

fast-agent automatically searches for configuration files in the current working directory and its parent directories. You can also specify a configuration file path with the `--config` command-line argument.

## General Settings

```
# Default model for all agents
default_model: "gpt-5-mini"  # Format: provider.model_name.reasoning_effort

# Whether to automatically enable Sampling. Model seletion precedence is Agent > Default.
auto_sampling: true

# Execution engine (only asyncio is currently supported)
execution_engine: "asyncio"
```

## Model Providers

### Anthropic

```
anthropic:
  api_key: "your_anthropic_key"  # Can also use ANTHROPIC_API_KEY env var
  base_url: "https://api.anthropic.com/v1"  # Optional, only include to override
```

### OpenAI

```
openai:
  api_key: "your_openai_key"  # Can also use OPENAI_API_KEY env var
  base_url: "https://api.openai.com/v1"  # Optional, only include to override
  reasoning_effort: "medium"  # Default reasoning effort: "low", "medium", or "high"
```

### Azure OpenAI

```
# Option 1: Using resource_name and api_key (standard method)
azure:
  api_key: "your_azure_openai_key"  # Required unless using DefaultAzureCredential
  resource_name: "your-resource-name"  # Resource name in Azure
  azure_deployment: "deployment-name"  # Required - deployment name from Azure
  api_version: "2023-05-15"  # Optional API version
  # Do NOT include base_url if you use resource_name

# Option 2: Using base_url and api_key (custom endpoints or sovereign clouds)
# azure:
#   api_key: "your_azure_openai_key"
#   base_url: "https://your-endpoint.openai.azure.com/"
#   azure_deployment: "deployment-name"
#   api_version: "2023-05-15"
#   # Do NOT include resource_name if you use base_url

# Option 3: Using DefaultAzureCredential (for managed identity, Azure CLI, etc.)
# azure:
#   use_default_azure_credential: true
#   base_url: "https://your-endpoint.openai.azure.com/"
#   azure_deployment: "deployment-name"
#   api_version: "2023-05-15"
#   # Do NOT include api_key or resource_name in this mode
```

Important configuration notes:
\- Use either `resource_name` or `base_url`, not both.
\- When using `DefaultAzureCredential`, do NOT include `api_key` or `resource_name` (the `azure-identity` package must be installed).
\- When using `base_url`, do NOT include `resource_name`.
\- When using `resource_name`, do NOT include `base_url`.
\- The model string format is `azure.deployment-name`

### DeepSeek

```
deepseek:
  api_key: "your_deepseek_key"  # Can also use DEEPSEEK_API_KEY env var
  base_url: "https://api.deepseek.com/v1"  # Optional, only include to override
```

### Google

```
google:
  api_key: "your_google_key"  # Can also use GOOGLE_API_KEY env var
  base_url: "https://generativelanguage.googleapis.com/v1beta/openai"  # Optional
```

### Generic (Ollama, etc.)

```
generic:
  api_key: "ollama"  # Default for Ollama, change as needed
  base_url: "http://localhost:11434/v1"  # Default for Ollama
```

### OpenRouter

```
openrouter:
  api_key: "your_openrouter_key"  # Can also use OPENROUTER_API_KEY env var
  base_url: "https://openrouter.ai/api/v1"  # Optional, only include to override
```

### TensorZero

```
tensorzero:
  base_url: "http://localhost:3000"  # Optional, only include to override
```

See the [TensorZero Quick Start](https://tensorzero.com/docs/quickstart) and the [TensorZero Gateway Deployment Guide](https://www.tensorzero.com/docs/gateway/deployment/) for more information on how to deploy the TensorZero Gateway.

### AWS Bedrock

```
bedrock:
  region: "us-east-1"  # Required - AWS region where Bedrock is available
  profile: "default"   # Optional - AWS profile to use (defaults to "default")
```

AWS Bedrock uses standard AWS authentication through the boto3 credential provider chain. You can configure credentials using:

- **AWS CLI**: Run `aws configure` to set up credentials (AWS SSO recommended for local development)
- **Environment variables**: `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `AWS_SESSION_TOKEN` (for temporary credentials)
- **IAM roles**: Use IAM roles when running on EC2 or other AWS services
- **AWS profiles**: Use named profiles with the `profile` setting or `AWS_PROFILE` environment variable

Additional environment variables:
\- `AWS_REGION` or `AWS_DEFAULT_REGION`: Override the region setting
\- `AWS_PROFILE`: Override the profile setting

The model string format is `bedrock.model-id` (e.g., `bedrock.amazon.nova-lite-v1:0`)

## MCP Server Configuration

MCP Servers are defined under the `mcp.servers` section:

```
mcp:
  servers:
    # Example stdio server
    server_name:
      transport: "stdio"  # "stdio" or "sse"
      command: "npx"  # Command to execute
      args: ["@package/server-name"]  # Command arguments as array
      read_timeout_seconds: 60  # Optional timeout in seconds
      env:  # Optional environment variables
        ENV_VAR1: "value1"
        ENV_VAR2: "value2"
      sampling:  # Optional sampling settings
        model: "gpt-5-mini"  # Model to use for sampling requests

    # Example Stremable HTTP server
    streamable_http__server:
      transport: "http"
      url: "http://localhost:8000/mcp"
      read_transport_sse_timeout_seconds: 300  # Timeout for HTTP connections
      headers:  # Optional HTTP headers
        Authorization: "Bearer token"
      auth:  # Optional authentication
      instructions:  # whether to include instructions in {{serverInstructions template variable}}

    # Example SSE server
    sse_server:
      transport: "sse"
      url: "http://localhost:8000/sse"
      read_transport_sse_timeout_seconds: 300  # Timeout for SSE connections
      headers:  # Optional HTTP headers
        Authorization: "Bearer token"
      auth:  # Optional authentication
        api_key: "your_api_key"

    # Server with roots
    file_server:
      transport: "stdio"
      command: "command"
      args: ["arguments"]
      roots:  # Root directories accessible to this server
        - uri: "file:///path/to/dir"  # Must start with file://
          name: "Optional Name"  # Optional display name for the root
          server_uri_alias: "file:///server/path"  # Optional, for consistent paths
```

## OpenTelemetry Settings

```
otel:
  enabled: false  # Enable or disable OpenTelemetry
  service_name: "fast-agent"  # Service name for tracing
  otlp_endpoint: "http://localhost:4318/v1/traces"  # OTLP endpoint for tracing
  console_debug: false  # Log spans to console
  sample_rate: 1.0  # Sample rate (0.0-1.0)
```

## Logging Settings

```
logger:
  type: "file"  # "none", "console", "file", or "http"
  level: "warning"  # "debug", "info", "warning", or "error"
  progress_display: true  # Enable/disable progress display
  path: "fastagent.jsonl"  # Path to log file (for "file" type)
  batch_size: 100  # Events to accumulate before processing
  flush_interval: 2.0  # Flush interval in seconds
  max_queue_size: 2048  # Maximum queue size for events

  # HTTP logger settings
  http_endpoint: "https://logging.example.com"  # Endpoint for HTTP logger
  http_headers:  # Headers for HTTP logger
    Authorization: "Bearer token"
  http_timeout: 5.0  # Timeout for HTTP logger requests

  # Console display options
  show_chat: true  # Show chat messages on console
  show_tools: true  # Show MCP Server tool calls on console
  truncate_tools: true  # Truncate long tool calls in display
  enable_markup: true # Disable if outputs conflict with rich library markup
  use_legacy_display: false # enable the < 0.2.43 display
```

## Example Full Configuration

```
default_model: "gpt-5-mini.low"

# Model provider settings
anthropic:
  api_key: API_KEY

openai:
  api_key: API_KEY
  reasoning_effort: "high"

# MCP servers
mcp:
  servers:
    fetch:
      transport: "stdio"
      command: "uvx"
      args: ["mcp-server-fetch"]

    prompts:
      transport: "stdio"
      command: "prompt-server"
      args: ["prompts/myprompt.txt"]

    filesys:
      transport: "stdio"
      command: "uvx"
      args: ["mcp-server-filesystem"]
      roots:
        - uri: "file://./data"
          name: "Data Directory"

# Logging configuration
logger:
  type: "file"
  level: "info"
  path: "logs/fastagent.jsonl"
```

## Environment Variables

All configuration options can be set via environment variables using a nested delimiter:

```
ANTHROPIC__API_KEY=your_key
OPENAI__API_KEY=your_key
LOGGER__LEVEL=debug
```

Environment variables take precedence over values in the configuration files. For nested arrays or complex structures, use the YAML configuration file.

The `fastagent.config.yaml` file supports referencing environment variables inline using the `${ENV_VAR}` syntax. When the configuration is loaded, any value specified as `${ENV_VAR}` will be automatically replaced with the value of the corresponding environment variable. This allows you to securely inject sensitive or environment-specific values into your configuration files without hardcoding them.

For example:

```
openai:
  api_key: "${OPENAI_API_KEY}"
```

In this example, the `api_key` value will be set to the value of the `OPENAI_API_KEY` environment variable at runtime.

Back to top

## Fast-Agent Command
[Skip to content](https://fast-agent.ai/ref/go_command/#fast-agent-go-command)

# fast-agent go

## `fast-agent go` command

The `go` command allows you to run an interactive agent directly from the command line without
creating a dedicated agent.py file.

### Usage

```
fast-agent go [OPTIONS]
```

### Options

- `--name TEXT`: Name for the workflow (default: "FastAgent CLI")
- `--instruction`, `-i <path or url>`: File name or URL for [System Prompt](https://fast-agent.ai/agents/instructions/) (default: "You are a helpful AI Agent.")
- `--config-path`, `-c <path>`: Path to config file
- `--servers <server1>,<server2>`: Comma-separated list of server names to enable from config
- `--url TEXT`: Comma-separated list of HTTP/SSE URLs to connect to directly
- `--auth TEXT`: Bearer token for authorization with URL-based servers
- `--model <model_string>`: Override the default model (e.g., haiku, sonnet, gpt-4)
- `--model <model_string1>,<model_string2>,...`: Set up a `parallel` containing each model
- `--message`, `-m TEXT`: Message to send to the agent (skips interactive mode)
- `--prompt-file`, `-p <path>`: Path to a prompt file to use (either text or JSON)
- `--quiet`: Disable progress display and logging
- `--stdio "<command> <options>"`: Run the command to attach a STDIO server (enclose arguments in quotes)
- `--npx "@package/name <options>"`: Run an NPX package as a STDIO server (enclose arguments in quotes)
- `--uvx "@package/name <options>"`: Run an UVX package as a STDIO server (enclose arguments in quotes)

### Examples

Note - you may omit `go` when supplying command line options.

```
# Basic usage with interactive mode
fast-agent go --model=haiku

# Basic usage with interactive mode (go omitted)
fast-agent --model haiku

# Send commands to different LLMs in Parallel
fast-agent --model kimi,gpt-5-mini.low

# Specifying servers from configuration
fast-agent go --servers=fetch,filesystem --model=haiku

# Directly connecting to HTTP/SSE servers via URLs
fast-agent go --url=http://localhost:8001/mcp,http://api.example.com/sse

# Connecting to an authenticated API endpoint
fast-agent go --url=https://api.example.com/mcp --auth=YOUR_API_TOKEN

# Run an NPX package directly
fast-agent --npx @modelcontextprotocol/server-everything

# Non-interactive mode with a single message
fast-agent go --message="What is the weather today?" --model=haiku

# Using a prompt file
fast-agent go --prompt-file=my-prompt.txt --model=haiku

# Specify a system prompt file
fast-agent go -i my_system_prompt.md

# Specify a skills directory
fast-agent --skills ~/my-skills/

# Provider LLM shell access (use at your own risk)
fast-agent -x
```

### URL Connection Details

The `--url` parameter allows you to connect directly to HTTP or SSE servers using URLs.

- URLs must have http or https scheme
- The transport type is determined by the URL path:
- URLs ending with `/sse` are treated as SSE transport
- URLs ending with `/mcp` or automatically appended with `/mcp` are treated as HTTP transport
- Server names are generated automatically based on the hostname, port, and path
- The URL-based servers are added to the agent's configuration and enabled

### Authentication

The `--auth` parameter provides authentication for URL-based servers:

- When provided, it creates an `Authorization: Bearer TOKEN` header for all URL-based servers
- This is commonly used with API endpoints that require authentication
- Example: `fast-agent go --url=https://api.example.com/mcp --auth=12345abcde`

Back to top

## Open Telemetry Setup
[Skip to content](https://fast-agent.ai/ref/open_telemetry/#getting-started)

# Open Telemetry

## Getting Started

**fast-agent** supports Open Telemetry, providing observability of MCP and LLM interactions. This is also a useful test/eval tool for comparing the behaviour of MCP Servers with different mixes of Tools, descriptions and models.

![Open Telemetry example](https://fast-agent.ai/ref/pics/otel_router.png)

## Set up an Open Telemetry server

The first step is to set up an Open Telemetry server. For this example, we will use [Jaeger](https://www.jaegertracing.io/) running locally with `docker-compose.yaml`. Create the following `docker-compose` file in a convenient directory:

docker-compose.yaml

```
services:
  jaeger:
    image: jaegertracing/jaeger:latest
    container_name: jaeger
    ports:
      - "16686:16686" # Web UI
      - "4318:4318" # OTLP HTTP

    restart: unless-stopped
```

Run `docker-compose up` to download and start the server. Navigate to `http://localhost:16686` to access the Jaeger UI.

## Configure fast-agent

Next, update your `fastagent.config.yaml` to enable telemetry:

fastagent.config.yaml

```
otel:
  enabled: true
  otlp_endpoint: "http://localhost:4318/v1/traces"  # This is the default value
```

Then, run your agent as normal - telemetry is transmitted by default to `http://localhost:4318/v1/traces`. From the Jaeger UI use the "Services" drop down to select **fast-agent** and click "Find Traces" to view the output.

For full configuration settings, check the [configuration file reference](https://fast-agent.ai/ref/config_file/#opentelemetry-settings)

Back to top

## Fast Agent AI
# Index

Welcome!

Back to top


