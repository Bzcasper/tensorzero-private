# https://www.tensorzero.com/docs llms-full.txt

## TensorZero Documentation
[Skip to main content](https://www.tensorzero.com/docs#content-area)

[TensorZero Docs home page![light logo](https://mintcdn.com/tensorzero/afFfERJmQ8YiZpCj/logo/light.svg?fit=max&auto=format&n=afFfERJmQ8YiZpCj&q=85&s=9396731a8a92cee7bd293dee05d71dac)![dark logo](https://mintcdn.com/tensorzero/afFfERJmQ8YiZpCj/logo/dark.svg?fit=max&auto=format&n=afFfERJmQ8YiZpCj&q=85&s=8d61c59531dbcdcace7fcdfc524bd4d2)](https://www.tensorzero.com/)

Search...

Ctrl K

Search...

Navigation

Introduction

Overview

[Guides](https://www.tensorzero.com/docs) [Integrations](https://www.tensorzero.com/docs/integrations/model-providers)

**TensorZero is an open-source stack for industrial-grade LLM applications:**

- **Gateway:** access every LLM provider through a unified API, built for performance (<1ms p99 latency)
- **Observability:** store inferences and feedback in your database, available programmatically or in the UI
- **Optimization:** collect metrics and human feedback to optimize prompts, models, and inference strategies
- **Evaluations:** benchmark individual inferences or end-to-end workflows using heuristics, LLM judges, etc.
- **Experimentation:** ship with confidence with built-in A/B testing, routing, fallbacks, retries, etc.

Take what you need, adopt incrementally, and complement with other tools.

**Start building today.**
The [Quickstart](https://www.tensorzero.com/docs/quickstart) shows it’s easy to set up an LLM application with TensorZero.**Questions?**
Ask us on [Slack](https://www.tensorzero.com/slack) or [Discord](https://www.tensorzero.com/discord).**Using TensorZero at work?**
Email us at [hello@tensorzero.com](mailto:hello@tensorzero.com) to set up a Slack or Teams channel with your team (free).

[Quickstart](https://www.tensorzero.com/docs/quickstart)

Ctrl+I

Assistant

Responses are generated using AI and may contain mistakes.

## TensorZero vs DSPy Comparison
[Skip to main content](https://www.tensorzero.com/docs/comparison/dspy#content-area)

[TensorZero Docs home page![light logo](https://mintcdn.com/tensorzero/afFfERJmQ8YiZpCj/logo/light.svg?fit=max&auto=format&n=afFfERJmQ8YiZpCj&q=85&s=9396731a8a92cee7bd293dee05d71dac)![dark logo](https://mintcdn.com/tensorzero/afFfERJmQ8YiZpCj/logo/dark.svg?fit=max&auto=format&n=afFfERJmQ8YiZpCj&q=85&s=8d61c59531dbcdcace7fcdfc524bd4d2)](https://www.tensorzero.com/)

Search...

Ctrl K

Search...

Navigation

Comparison

Comparison: TensorZero vs. DSPy

[Guides](https://www.tensorzero.com/docs) [Integrations](https://www.tensorzero.com/docs/integrations/model-providers)

On this page

- [Similarities](https://www.tensorzero.com/docs/comparison/dspy#similarities)
- [Key Differences](https://www.tensorzero.com/docs/comparison/dspy#key-differences)
- [TensorZero](https://www.tensorzero.com/docs/comparison/dspy#tensorzero)
- [DSPy](https://www.tensorzero.com/docs/comparison/dspy#dspy)
- [Combining TensorZero and DSPy](https://www.tensorzero.com/docs/comparison/dspy#combining-tensorzero-and-dspy)

TensorZero and DSPy serve **different but complementary** purposes in the LLM ecosystem.
TensorZero is a full-stack LLM engineering platform focused on production applications and optimization, while DSPy is a framework for programming with language models through modular prompting.
**You can get the best of both worlds by using DSPy and TensorZero together!**

## [​](https://www.tensorzero.com/docs/comparison/dspy\#similarities)  Similarities

- **LLM Optimization.**
Both TensorZero and DSPy focus on LLM optimization, but in different ways.
DSPy focuses on automated prompt engineering, while TensorZero provides a complete set of tools for optimizing LLM systems (including prompts, models, and inference strategies).
- **LLM Programming Abstractions.**
Both TensorZero and DSPy provide abstractions for working with LLMs in a structured way, moving beyond raw prompting to more maintainable approaches.

[→ Prompt Templates & Schemas with TensorZero](https://www.tensorzero.com/docs/gateway/create-a-prompt-template)
- **Automated Prompt Engineering.**
TensorZero implements MIPROv2, the automated prompt engineering algorithm recommended by DSPy.
MIPROv2 jointly optimizes instructions and in-context examples in prompts.

[→ Recipe: Automated Prompt Engineering with MIPRO](https://github.com/tensorzero/tensorzero/tree/main/recipes/mipro)

## [​](https://www.tensorzero.com/docs/comparison/dspy\#key-differences)  Key Differences

### [​](https://www.tensorzero.com/docs/comparison/dspy\#tensorzero)  TensorZero

- **Production Infrastructure.**
TensorZero provides complete production infrastructure including **observability, optimization, evaluations, and experimentation** capabilities.
DSPy focuses on the development phase and prompt programming patterns.
- **Model Optimization.**
TensorZero provides tools for optimizing models, including fine-tuning and RLHF.
DSPy primarily focuses on automated prompt engineering.

[→ Optimization Recipes with TensorZero](https://www.tensorzero.com/docs/recipes)
- **Inference-Time Optimization.**
TensorZero provides inference-time optimizations like dynamic in-context learning.
DSPy focuses on offline optimization strategies (e.g. static in-context learning).

[→ Inference-Time Optimizations with TensorZero](https://www.tensorzero.com/docs/gateway/guides/inference-time-optimizations)

### [​](https://www.tensorzero.com/docs/comparison/dspy\#dspy)  DSPy

- **Advanced Automated Prompt Engineering.**
DSPy provides sophisticated automated prompt engineering tools for LLMs like teleprompters, recursive reasoning, and self-improvement loops.
TensorZero has some built-in prompt optimization features (more on the way) and integrates with DSPy for additional capabilities.

[→ Improving Math Reasoning — Combining TensorZero and DSPy](https://github.com/tensorzero/tensorzero/tree/main/examples/gsm8k-custom-recipe-dspy)
- **Lightweight Design.**
DSPy is a lightweight framework focused solely on LLM programming patterns, particularly during the R&D stage.
TensorZero is a more comprehensive platform with additional infrastructure components covering end-to-end LLM engineering workflows.

Is TensorZero missing any features that are really important to you? Let us know on [GitHub Discussions](https://github.com/tensorzero/tensorzero/discussions), [Slack](https://www.tensorzero.com/slack), or [Discord](https://www.tensorzero.com/discord).

## [​](https://www.tensorzero.com/docs/comparison/dspy\#combining-tensorzero-and-dspy)  Combining TensorZero and DSPy

You can get the best of both worlds by using DSPy and TensorZero together!TensorZero provides a number of pre-built optimization recipes covering common LLM engineering workflows like supervised fine-tuning and RLHF.
But you can also easily create your own recipes and workflows.
This example shows how to optimize a TensorZero function using a tool like DSPy.[→ Improving Math Reasoning — Combining TensorZero and DSPy](https://github.com/tensorzero/tensorzero/tree/main/examples/gsm8k-custom-recipe-dspy)

[Frequently Asked Questions](https://www.tensorzero.com/docs/faq) [LangChain](https://www.tensorzero.com/docs/comparison/langchain)

Ctrl+I

Assistant

Responses are generated using AI and may contain mistakes.

## TensorZero vs. LangChain
[Skip to main content](https://www.tensorzero.com/docs/comparison/langchain#content-area)

[TensorZero Docs home page![light logo](https://mintcdn.com/tensorzero/afFfERJmQ8YiZpCj/logo/light.svg?fit=max&auto=format&n=afFfERJmQ8YiZpCj&q=85&s=9396731a8a92cee7bd293dee05d71dac)![dark logo](https://mintcdn.com/tensorzero/afFfERJmQ8YiZpCj/logo/dark.svg?fit=max&auto=format&n=afFfERJmQ8YiZpCj&q=85&s=8d61c59531dbcdcace7fcdfc524bd4d2)](https://www.tensorzero.com/)

Search...

Ctrl K

Search...

Navigation

Comparison

Comparison: TensorZero vs. LangChain

[Guides](https://www.tensorzero.com/docs) [Integrations](https://www.tensorzero.com/docs/integrations/model-providers)

On this page

- [Similarities](https://www.tensorzero.com/docs/comparison/langchain#similarities)
- [Key Differences](https://www.tensorzero.com/docs/comparison/langchain#key-differences)
- [TensorZero](https://www.tensorzero.com/docs/comparison/langchain#tensorzero)
- [LangChain](https://www.tensorzero.com/docs/comparison/langchain#langchain)

TensorZero and LangChain both provide tools for LLM orchestration, but they serve different purposes in the ecosystem.
While LangChain focuses on rapid prototyping with a large ecosystem of integrations, TensorZero is designed for production-grade deployments with built-in observability, optimization, evaluations, and experimentation capabilities.

We provide a minimal example [integrating TensorZero with LangGraph](https://github.com/tensorzero/tensorzero/tree/main/examples/integrations/langgraph).

## [​](https://www.tensorzero.com/docs/comparison/langchain\#similarities)  Similarities

- **LLM Orchestration.**
Both TensorZero and LangChain are developer tools that streamline LLM engineering workflows.
TensorZero focuses on production-grade deployments and end-to-end LLM engineering workflows (inference, observability, optimization, evaluations, experimentation).
LangChain focuses on rapid prototyping and offers complementary commercial products for features like observability.
- **Open Source.**
Both TensorZero (Apache 2.0) and LangChain (MIT) are open-source.
TensorZero is fully open-source (including TensorZero UI for observability), whereas LangChain requires a commercial offering for certain features (e.g. LangSmith for observability).
- **Unified Interface.**
Both TensorZero and LangChain offer a unified interface that allows you to access LLMs from most major model providers with a single integration, with support for structured outputs, tool use, streaming, and more.

[→ TensorZero Gateway Quickstart](https://www.tensorzero.com/docs/quickstart)
- **Inference-Time Optimizations.**
Both TensorZero and LangChain offer inference-time optimizations like dynamic in-context learning.

[→ Inference-Time Optimizations with TensorZero](https://www.tensorzero.com/docs/gateway/guides/inference-time-optimizations)
- **Inference Caching.**
Both TensorZero and LangChain allow you to cache requests to improve latency and reduce costs.

[→ Inference Caching with TensorZero](https://www.tensorzero.com/docs/gateway/guides/inference-caching)

## [​](https://www.tensorzero.com/docs/comparison/langchain\#key-differences)  Key Differences

### [​](https://www.tensorzero.com/docs/comparison/langchain\#tensorzero)  TensorZero

- **Separation of Concerns: Application Engineering vs. LLM Optimization.**
TensorZero enables a clear separation between application logic and LLM implementation details.
By treating LLM functions as interfaces with structured inputs and outputs, TensorZero allows you to swap implementations without changing application code.
This approach makes it easier to manage complex LLM applications, enables GitOps for prompt and configuration management, and streamlines optimization and experimentation workflows.
LangChain blends application logic with LLM implementation details, streamlining rapid prototyping but making it harder to maintain and optimize complex applications.

[→ Prompt Templates & Schemas with TensorZero](https://www.tensorzero.com/docs/gateway/create-a-prompt-template)

[→ Advanced: Think of LLM Applications as POMDPs — Not Agents](https://www.tensorzero.com/blog/think-of-llm-applications-as-pomdps-not-agents/)
- **Open-Source Observability.**
TensorZero offers built-in observability features (including UI), collecting inference and feedback data in your own database.
LangChain requires a separate commercial service (LangSmith) for observability.
- **Built-in Optimization.**
TensorZero offers built-in optimization features, including supervised fine-tuning, RLHF, and automated prompt engineering recipes.
With the TensorZero UI, you can fine-tune models using your inference and feedback data in just a few clicks.
LangChain doesn’t offer any built-in optimization features.

[→ Optimization Recipes with TensorZero](https://www.tensorzero.com/docs/recipes)
- **Built-in Evaluations.**
TensorZero offers built-in evaluation functionality, including heuristics and LLM judges.
LangChain requires a separate commercial service (LangSmith) for evaluations.

[→ TensorZero Evaluations Overview](https://www.tensorzero.com/docs/evaluations)
- **Automated Experimentation (A/B Testing).**
TensorZero offers built-in experimentation features, allowing you to run experiments on your prompts, models, and inference strategies.
LangChain doesn’t offer any experimentation features.

[→ Run adaptive A/B tests with TensorZero](https://www.tensorzero.com/docs/experimentation/run-adaptive-ab-tests)
- **Performance & Scalability.**
TensorZero is built from the ground up for high performance, with a focus on low latency and high throughput.
LangChain introduces substantial latency and memory overhead to your application.

[→ TensorZero Gateway Benchmarks](https://www.tensorzero.com/docs/gateway/benchmarks)
- **Language and Platform Agnostic.**
TensorZero is language and platform agnostic; in addition to its Python client, it supports any language that can make HTTP requests.
LangChain only supports applications built in Python and JavaScript.

[→ TensorZero Gateway API Reference](https://www.tensorzero.com/docs/gateway/api-reference/inference)
- **Batch Inference.**
TensorZero supports batch inference with certain model providers, which significantly reduces inference costs.
LangChain doesn’t support batch inference.

[→ Batch Inference with TensorZero](https://www.tensorzero.com/docs/gateway/guides/batch-inference)
- **Credential Management.**
TensorZero streamlines credential management for your model providers, allowing you to manage your API keys in a single place and set up advanced workflows like load balancing between API keys.
LangChain only offers basic credential management features.

[→ Credential Management with TensorZero](https://www.tensorzero.com/docs/operations/manage-credentials)
- **Automatic Fallbacks for Higher Reliability.**
TensorZero allows you to very easily set up retries, fallbacks, load balancing, and routing to increase reliability.
LangChain only offers basic, cumbersome fallback functionality.

[→ Retries & Fallbacks with TensorZero](https://www.tensorzero.com/docs/gateway/guides/retries-fallbacks)

### [​](https://www.tensorzero.com/docs/comparison/langchain\#langchain)  LangChain

- **Focus on Rapid Prototyping.**
LangChain is designed for rapid prototyping, with a focus on ease of use and rapid iteration.
TensorZero is designed for production-grade deployments, so it requires more setup and configuration (e.g. a database to store your observability data) — but you can still get started in minutes.

[→ TensorZero Quickstart — From 0 to Observability & Fine-Tuning](https://www.tensorzero.com/docs/quickstart)
- **Ecosystem of Integrations.**
LangChain has a large ecosystem of integrations with other libraries and tools, including model providers, vector databases, observability tools, and more.
TensorZero provides many integrations with model providers, but delegates other integrations to the user.
- **Managed Service.**
LangChain offers paid managed (hosted) services for features like observability (LangSmith).
TensorZero is fully open-source and self-hosted.

Is TensorZero missing any features that are really important to you? Let us know on [GitHub Discussions](https://github.com/tensorzero/tensorzero/discussions), [Slack](https://www.tensorzero.com/slack), or [Discord](https://www.tensorzero.com/discord).

[DSPy](https://www.tensorzero.com/docs/comparison/dspy) [Langfuse](https://www.tensorzero.com/docs/comparison/langfuse)

Ctrl+I

Assistant

Responses are generated using AI and may contain mistakes.

## TensorZero vs. Langfuse
[Skip to main content](https://www.tensorzero.com/docs/comparison/langfuse#content-area)

[TensorZero Docs home page![light logo](https://mintcdn.com/tensorzero/afFfERJmQ8YiZpCj/logo/light.svg?fit=max&auto=format&n=afFfERJmQ8YiZpCj&q=85&s=9396731a8a92cee7bd293dee05d71dac)![dark logo](https://mintcdn.com/tensorzero/afFfERJmQ8YiZpCj/logo/dark.svg?fit=max&auto=format&n=afFfERJmQ8YiZpCj&q=85&s=8d61c59531dbcdcace7fcdfc524bd4d2)](https://www.tensorzero.com/)

Search...

Ctrl K

Search...

Navigation

Comparison

Comparison: TensorZero vs. Langfuse

[Guides](https://www.tensorzero.com/docs) [Integrations](https://www.tensorzero.com/docs/integrations/model-providers)

On this page

- [Similarities](https://www.tensorzero.com/docs/comparison/langfuse#similarities)
- [Key Differences](https://www.tensorzero.com/docs/comparison/langfuse#key-differences)
- [TensorZero](https://www.tensorzero.com/docs/comparison/langfuse#tensorzero)
- [Langfuse](https://www.tensorzero.com/docs/comparison/langfuse#langfuse)
- [Combining TensorZero and Langfuse](https://www.tensorzero.com/docs/comparison/langfuse#combining-tensorzero-and-langfuse)

TensorZero and Langfuse both provide open-source tools that streamline LLM engineering workflows.
TensorZero focuses on inference and optimization, while Langfuse specializes in powerful interfaces for observability and evals.
That said, **you can get the best of both worlds by using TensorZero alongside Langfuse**.

## [​](https://www.tensorzero.com/docs/comparison/langfuse\#similarities)  Similarities

- **Open Source & Self-Hosted.**
Both TensorZero and Langfuse are open source and self-hosted.
Your data never leaves your infrastructure, and you don’t risk downtime by relying on external APIs.
TensorZero is fully open-source, whereas Langfuse gates some of its features behind a paid license.
- **Built-in Observability.**
Both TensorZero and Langfuse offer built-in observability features, collecting inference in your own database.
Langfuse offers a broader set of advanced observability features, including application-level tracing.
TensorZero focuses more on structured data collection for optimization, including downstream metrics and feedback.
- **Built-in Evaluations.**
Both TensorZero and Langfuse offer built-in evaluations features, enabling you to sanity check and benchmark the performance of your prompts, models, and more — using heuristics and LLM judges.
TensorZero LLM judges are also TensorZero functions, which means you can optimize them using TensorZero’s optimization recipes.
Langfuse offers a broader set of built-in heuristics and UI features for evaluations.

[→ TensorZero Evaluations Overview](https://www.tensorzero.com/docs/evaluations)

## [​](https://www.tensorzero.com/docs/comparison/langfuse\#key-differences)  Key Differences

### [​](https://www.tensorzero.com/docs/comparison/langfuse\#tensorzero)  TensorZero

- **Unified Inference API.**
TensorZero offers a unified inference API that allows you to access LLMs from most major model providers with a single integration, with support for structured outputs, tool use, streaming, and more.
Langfuse doesn’t provide a built-in LLM gateway.

[→ TensorZero Gateway Quickstart](https://www.tensorzero.com/docs/quickstart)
- **Built-in Inference-Time Optimizations.**
TensorZero offers built-in inference-time optimizations (e.g. dynamic in-context learning), allowing you to optimize your inference performance.
Langfuse doesn’t offer any inference-time optimizations.

[→ Inference-Time Optimizations with TensorZero](https://www.tensorzero.com/docs/gateway/guides/inference-time-optimizations)
- **Optimization Recipes.**
TensorZero offers optimization recipes (e.g. supervised fine-tuning, RLHF, MIPRO) that leverage your own data to improve your LLM’s performance.
Langfuse doesn’t offer built-in features like this.

[→ Optimization Recipes with TensorZero](https://www.tensorzero.com/docs/recipes)
- **Automatic Fallbacks for Higher Reliability.**
TensorZero offers automatic fallbacks to increase reliability.
Langfuse doesn’t offer any such features.

[→ Retries & Fallbacks with TensorZero](https://www.tensorzero.com/docs/gateway/guides/retries-fallbacks)
- **Automated Experimentation (A/B Testing).**
TensorZero offers built-in experimentation features, allowing you to run experiments on your prompts, models, and inference strategies.
Langfuse doesn’t offer any experimentation features.

[→ Run adaptive A/B tests with TensorZero](https://www.tensorzero.com/docs/experimentation/run-adaptive-ab-tests)

### [​](https://www.tensorzero.com/docs/comparison/langfuse\#langfuse)  Langfuse

- **Advanced Observability & Evaluations.**
While both TensorZero and Langfuse offer observability and evaluations features, Langfuse takes it further with advanced observability features.
Additionally, Langfuse offers a prompt playground, which TensorZero doesn’t offer (coming soon!).
- **Access Control.**
Langfuse offers access control features like SSO and user management.
TensorZero supports TensorZero API key for inference, but more advanced access control requires complementary tools like Nginx or OAuth2 Proxy.
[→ Set up auth for TensorZero](https://www.tensorzero.com/docs/operations/set-up-auth-for-tensorzero)
- **Managed Service.**
Langfuse offers a paid managed (hosted) service in addition to the open-source version.
TensorZero is fully open-source and self-hosted.

Is TensorZero missing any features that are really important to you? Let us know on [GitHub Discussions](https://github.com/tensorzero/tensorzero/discussions), [Slack](https://www.tensorzero.com/slack), or [Discord](https://www.tensorzero.com/discord).

## [​](https://www.tensorzero.com/docs/comparison/langfuse\#combining-tensorzero-and-langfuse)  Combining TensorZero and Langfuse

You can combine TensorZero and Langfuse to get the best of both worlds.A leading voice agent startup uses TensorZero for inference and optimization, alongside Langfuse for more advanced observability and evals.

[LangChain](https://www.tensorzero.com/docs/comparison/langchain) [LiteLLM](https://www.tensorzero.com/docs/comparison/litellm)

Ctrl+I

Assistant

Responses are generated using AI and may contain mistakes.

## TensorZero vs. LiteLLM
[Skip to main content](https://www.tensorzero.com/docs/comparison/litellm#content-area)

[TensorZero Docs home page![light logo](https://mintcdn.com/tensorzero/afFfERJmQ8YiZpCj/logo/light.svg?fit=max&auto=format&n=afFfERJmQ8YiZpCj&q=85&s=9396731a8a92cee7bd293dee05d71dac)![dark logo](https://mintcdn.com/tensorzero/afFfERJmQ8YiZpCj/logo/dark.svg?fit=max&auto=format&n=afFfERJmQ8YiZpCj&q=85&s=8d61c59531dbcdcace7fcdfc524bd4d2)](https://www.tensorzero.com/)

Search...

Ctrl K

Search...

Navigation

Comparison

Comparison: TensorZero vs. LiteLLM

[Guides](https://www.tensorzero.com/docs) [Integrations](https://www.tensorzero.com/docs/integrations/model-providers)

On this page

- [Similarities](https://www.tensorzero.com/docs/comparison/litellm#similarities)
- [Key Differences](https://www.tensorzero.com/docs/comparison/litellm#key-differences)
- [TensorZero](https://www.tensorzero.com/docs/comparison/litellm#tensorzero)
- [LiteLLM](https://www.tensorzero.com/docs/comparison/litellm#litellm)
- [Combining TensorZero and LiteLLM](https://www.tensorzero.com/docs/comparison/litellm#combining-tensorzero-and-litellm)

TensorZero and LiteLLM both offer a unified inference API for LLMs, but they have different features beyond that.
TensorZero offers a broader set of features (including observability, optimization, evaluations, and experimentation), whereas LiteLLM offers more traditional gateway features (e.g. budgeting, queuing) and third-party integrations.
That said, **you can get the best of both worlds by using LiteLLM as a model provider inside TensorZero**!

## [​](https://www.tensorzero.com/docs/comparison/litellm\#similarities)  Similarities

- **Unified Inference API.**
Both TensorZero and LiteLLM offer a unified inference API that allows you to access LLMs from most major model providers with a single integration, with support for structured outputs, batch inference, tool use, streaming, and more.

[→ TensorZero Gateway Quickstart](https://www.tensorzero.com/docs/quickstart)
- **Automatic Fallbacks for Higher Reliability.**
Both TensorZero and LiteLLM offer automatic fallbacks to increase reliability.

[→ Retries & Fallbacks with TensorZero](https://www.tensorzero.com/docs/gateway/guides/retries-fallbacks)
- **Open Source & Self-Hosted.**
Both TensorZero and LiteLLM are open source and self-hosted.
Your data never leaves your infrastructure, and you don’t risk downtime by relying on external APIs.
TensorZero is fully open-source, whereas LiteLLM gates some of its features behind an enterprise license.
- **Inference Caching.**
Both TensorZero and LiteLLM allow you to cache requests to improve latency and reduce costs.

[→ Inference Caching with TensorZero](https://www.tensorzero.com/docs/gateway/guides/inference-caching)
- **Multimodal Inference.**
Both TensorZero and LiteLLM support multimodal inference.

[→ Multimodal Inference with TensorZero](https://www.tensorzero.com/docs/gateway/guides/multimodal-inference)

## [​](https://www.tensorzero.com/docs/comparison/litellm\#key-differences)  Key Differences

### [​](https://www.tensorzero.com/docs/comparison/litellm\#tensorzero)  TensorZero

- **High Performance.**
The TensorZero Gateway was built from the ground up in Rust 🦀 with performance in mind (<1ms P99 latency at 10,000 QPS).
LiteLLM is built in Python, resulting in 25-100x+ latency overhead and much lower throughput.

[→ Performance Benchmarks: TensorZero vs. LiteLLM](https://www.tensorzero.com/docs/gateway/benchmarks)
- **Built-in Observability.**
TensorZero offers its own observability features, collecting inference and feedback data in your own database.
LiteLLM only offers integrations with third-party observability tools like Langfuse.
- **Built-in Evaluations.**
TensorZero offers built-in evaluation functionality, including heuristics and LLM judges.
LiteLLM doesn’t offer any evaluations functionality.

[→ TensorZero Evaluations Overview](https://www.tensorzero.com/docs/evaluations)
- **Automated Experimentation (A/B Testing).**
TensorZero offers built-in experimentation features, allowing you to run experiments on your prompts, models, and inference strategies.
LiteLLM doesn’t offer any experimentation features.

[→ Run adaptive A/B tests with TensorZero](https://www.tensorzero.com/docs/experimentation/run-adaptive-ab-tests)
- **Built-in Inference-Time Optimizations.**
TensorZero offers built-in inference-time optimizations (e.g. dynamic in-context learning), allowing you to optimize your inference performance.
LiteLLM doesn’t offer any inference-time optimizations.

[→ Inference-Time Optimizations with TensorZero](https://www.tensorzero.com/docs/gateway/guides/inference-time-optimizations)
- **Optimization Recipes.**
TensorZero offers optimization recipes (e.g. supervised fine-tuning, RLHF, MIPRO) that leverage your own data to improve your LLM’s performance.
LiteLLM doesn’t offer any features like this.

[→ Optimization Recipes with TensorZero](https://www.tensorzero.com/docs/recipes)
- **Schemas, Templates, GitOps.**
TensorZero enables a schema-first approach to building LLM applications, allowing you to separate your application logic from LLM implementation details.
This approach allows your to more easily manage complex LLM applications, benefit from GitOps for prompt and configuration management, counterfactually improve data for optimization, and more.
LiteLLM only offers the standard unstructured chat completion interface.

[→ Prompt Templates & Schemas with TensorZero](https://www.tensorzero.com/docs/gateway/create-a-prompt-template)
- **Access Control.**
Both TensorZero and LiteLLM support virtual (custom) API keys to authenticate requests.
LiteLLM offers advanced authentication features in its enterprise plan, whereas TensorZero requires complementary open-source tools like Nginx or OAuth2 Proxy for such use cases.

[→ Set up auth for TensorZero](https://www.tensorzero.com/docs/operations/set-up-auth-for-tensorzero)

### [​](https://www.tensorzero.com/docs/comparison/litellm\#litellm)  LiteLLM

- **Dynamic Provider Routing.**
LiteLLM allows you to dynamically route requests to different model providers based on latency, cost, and rate limits.
TensorZero only offers static routing capabilities, i.e. a pre-defined sequence of model providers to attempt.

[→ Retries & Fallbacks with TensorZero](https://www.tensorzero.com/docs/gateway/guides/retries-fallbacks)
- **Request Prioritization.**
LiteLLM allows you to prioritize requests over others, which can be useful for high-priority tasks when you’re constrained by rate limits.
TensorZero doesn’t offer request prioritization, and instead requires you to manage the request queue externally (e.g. using Redis).
- **Built-in Guardrails Integration.**
LiteLLM offers built-in support for integrations with guardrails tools like AWS Bedrock.
For now, TensorZero doesn’t offer built-in guardrails, and instead requires you to manage integrations yourself.
- **Managed Service.**
LiteLLM offers a paid managed (hosted) service in addition to the open-source version.
TensorZero is fully open-source and self-hosted.

Is TensorZero missing any features that are really important to you? Let us know on [GitHub Discussions](https://github.com/tensorzero/tensorzero/discussions), [Slack](https://www.tensorzero.com/slack), or [Discord](https://www.tensorzero.com/discord).

## [​](https://www.tensorzero.com/docs/comparison/litellm\#combining-tensorzero-and-litellm)  Combining TensorZero and LiteLLM

You can get the best of both worlds by using LiteLLM as a model provider inside TensorZero.LiteLLM offers an OpenAI-compatible API, so you can use TensorZero’s OpenAI-compatible endpoint to call LiteLLM.
Learn more about using [OpenAI-compatible endpoints](https://www.tensorzero.com/docs/integrations/model-providers/openai-compatible).

[Langfuse](https://www.tensorzero.com/docs/comparison/langfuse) [OpenPipe](https://www.tensorzero.com/docs/comparison/openpipe)

Ctrl+I

Assistant

Responses are generated using AI and may contain mistakes.

## TensorZero vs OpenPipe
[Skip to main content](https://www.tensorzero.com/docs/comparison/openpipe#content-area)

[TensorZero Docs home page![light logo](https://mintcdn.com/tensorzero/afFfERJmQ8YiZpCj/logo/light.svg?fit=max&auto=format&n=afFfERJmQ8YiZpCj&q=85&s=9396731a8a92cee7bd293dee05d71dac)![dark logo](https://mintcdn.com/tensorzero/afFfERJmQ8YiZpCj/logo/dark.svg?fit=max&auto=format&n=afFfERJmQ8YiZpCj&q=85&s=8d61c59531dbcdcace7fcdfc524bd4d2)](https://www.tensorzero.com/)

Search...

Ctrl K

Search...

Navigation

Comparison

Comparison: TensorZero vs. OpenPipe

[Guides](https://www.tensorzero.com/docs) [Integrations](https://www.tensorzero.com/docs/integrations/model-providers)

On this page

- [Similarities](https://www.tensorzero.com/docs/comparison/openpipe#similarities)
- [Key Differences](https://www.tensorzero.com/docs/comparison/openpipe#key-differences)
- [TensorZero](https://www.tensorzero.com/docs/comparison/openpipe#tensorzero)
- [OpenPipe](https://www.tensorzero.com/docs/comparison/openpipe#openpipe)
- [Combining TensorZero and OpenPipe](https://www.tensorzero.com/docs/comparison/openpipe#combining-tensorzero-and-openpipe)

TensorZero and OpenPipe both provide tools that streamline fine-tuning workflows for LLMs.
TensorZero is open-source and self-hosted, while OpenPipe is a paid managed service (inference costs ~2x more than specialized providers supported by TensorZero).
That said, **you can get the best of both worlds by using OpenPipe as a model provider inside TensorZero**.

## [​](https://www.tensorzero.com/docs/comparison/openpipe\#similarities)  Similarities

- **LLM Optimization (Fine-Tuning).**
Both TensorZero and OpenPipe focus on LLM optimization (e.g. fine-tuning, DPO).
OpenPipe focuses on fine-tuning, while TensorZero provides a complete set of tools for optimizing LLM systems (including prompts, models, and inference strategies).

[→ Optimization Recipes with TensorZero](https://www.tensorzero.com/docs/recipes)
- **Built-in Observability.**
Both TensorZero and OpenPipe offer built-in observability features.
TensorZero stores inference data in your own database for full privacy and control, while OpenPipe stores it themselves in their own cloud.
- **Built-in Evaluations.**
Both TensorZero and OpenPipe offer built-in evaluations features, enabling you to sanity check and benchmark the performance of your prompts, models, and more — using heuristics and LLM judges.
TensorZero LLM judges are also TensorZero functions, which means you can optimize them using TensorZero’s optimization recipes.

[→ TensorZero Evaluations Overview](https://www.tensorzero.com/docs/evaluations)

## [​](https://www.tensorzero.com/docs/comparison/openpipe\#key-differences)  Key Differences

### [​](https://www.tensorzero.com/docs/comparison/openpipe\#tensorzero)  TensorZero

- **Open Source & Self-Hosted.**
TensorZero is fully open source and self-hosted.
Your data never leaves your infrastructure, and you don’t risk downtime by relying on external APIs.
OpenPipe is a closed-source managed service.
- **No Added Cost (& Cheaper Inference Providers).**
TensorZero is free to use: your bring your own LLM API keys and there is no additional cost.
OpenPipe charges ~2x on inference costs compared to specialized providers supported by TensorZero (e.g. Fireworks AI).
- **Unified Inference API.**
TensorZero offers a unified inference API that allows you to access LLMs from most major model providers with a single integration, with support for structured outputs, tool use, streaming, and more.


OpenPipe supports a much smaller set of LLMs.

[→ TensorZero Gateway Quickstart](https://www.tensorzero.com/docs/quickstart)
- **Built-in Inference-Time Optimizations.**
TensorZero offers built-in inference-time optimizations (e.g. dynamic in-context learning), allowing you to optimize your inference performance.
OpenPipe doesn’t offer any inference-time optimizations.

[→ Inference-Time Optimizations with TensorZero](https://www.tensorzero.com/docs/gateway/guides/inference-time-optimizations)
- **Automatic Fallbacks for Higher Reliability.**
TensorZero is self-hosted and provides automatic fallbacks between model providers to increase reliability.
OpenPipe can fallback their own models to other OpenAI-compatible APIs, but if OpenPipe itself goes down, you’re out of luck.

[→ Retries & Fallbacks with TensorZero](https://www.tensorzero.com/docs/gateway/guides/retries-fallbacks)
- **Automated Experimentation (A/B Testing).**
TensorZero offers built-in experimentation features, allowing you to run experiments on your prompts, models, and inference strategies.
OpenPipe doesn’t offer any experimentation features.

[→ Run adaptive A/B tests with TensorZero](https://www.tensorzero.com/docs/experimentation/run-adaptive-ab-tests)
- **Batch Inference.**
TensorZero supports batch inference with certain model providers, which significantly reduces inference costs.
OpenPipe doesn’t support batch inference.

[→ Batch Inference with TensorZero](https://www.tensorzero.com/docs/gateway/guides/batch-inference)
- **Inference Caching.**
Both TensorZero and OpenPipe allow you to cache requests to improve latency and reduce costs.
OpenPipe only caches requests to their own models, while TensorZero caches requests to all model providers.

[→ Inference Caching with TensorZero](https://www.tensorzero.com/docs/gateway/guides/inference-caching)
- **Schemas, Templates, GitOps.**
TensorZero enables a schema-first approach to building LLM applications, allowing you to separate your application logic from LLM implementation details.
This approach allows your to more easily manage complex LLM applications, benefit from GitOps for prompt and configuration management, counterfactually improve data for optimization, and more.
OpenPipe only offers the standard unstructured chat completion interface.

[→ Prompt Templates & Schemas with TensorZero](https://www.tensorzero.com/docs/gateway/create-a-prompt-template)

### [​](https://www.tensorzero.com/docs/comparison/openpipe\#openpipe)  OpenPipe

- **Guardrails.**
OpenPipe offers guardrails (runtime AI judges) for your fine-tuned models.
TensorZero doesn’t offer built-in guardrails, and instead requires you to manage them yourself.

Is TensorZero missing any features that are really important to you? Let us know on [GitHub Discussions](https://github.com/tensorzero/tensorzero/discussions), [Slack](https://www.tensorzero.com/slack), or [Discord](https://www.tensorzero.com/discord).

## [​](https://www.tensorzero.com/docs/comparison/openpipe\#combining-tensorzero-and-openpipe)  Combining TensorZero and OpenPipe

You can get the best of both worlds by using OpenPipe as a model provider inside TensorZero.OpenPipe provides an OpenAI-compatible API, so you can use models previously fine-tuned with OpenPipe with TensorZero.
Learn more about using [OpenAI-compatible endpoints](https://www.tensorzero.com/docs/integrations/model-providers/openai-compatible).

[LiteLLM](https://www.tensorzero.com/docs/comparison/litellm) [OpenRouter](https://www.tensorzero.com/docs/comparison/openrouter)

Ctrl+I

Assistant

Responses are generated using AI and may contain mistakes.

## TensorZero vs OpenRouter
[Skip to main content](https://www.tensorzero.com/docs/comparison/openrouter#content-area)

[TensorZero Docs home page![light logo](https://mintcdn.com/tensorzero/afFfERJmQ8YiZpCj/logo/light.svg?fit=max&auto=format&n=afFfERJmQ8YiZpCj&q=85&s=9396731a8a92cee7bd293dee05d71dac)![dark logo](https://mintcdn.com/tensorzero/afFfERJmQ8YiZpCj/logo/dark.svg?fit=max&auto=format&n=afFfERJmQ8YiZpCj&q=85&s=8d61c59531dbcdcace7fcdfc524bd4d2)](https://www.tensorzero.com/)

Search...

Ctrl K

Search...

Navigation

Comparison

Comparison: TensorZero vs. OpenRouter

[Guides](https://www.tensorzero.com/docs) [Integrations](https://www.tensorzero.com/docs/integrations/model-providers)

On this page

- [Similarities](https://www.tensorzero.com/docs/comparison/openrouter#similarities)
- [Key Differences](https://www.tensorzero.com/docs/comparison/openrouter#key-differences)
- [TensorZero](https://www.tensorzero.com/docs/comparison/openrouter#tensorzero)
- [OpenRouter](https://www.tensorzero.com/docs/comparison/openrouter#openrouter)
- [Combining TensorZero and OpenRouter](https://www.tensorzero.com/docs/comparison/openrouter#combining-tensorzero-and-openrouter)

TensorZero and OpenRouter both offer a unified inference API for LLMs, but they have different features beyond that.
TensorZero offers a more comprehensive set of features (including observability, optimization, evaluations, and experimentation), whereas OpenRouter offers more dynamic routing capabilities.
That said, **you can get the best of both worlds by using OpenRouter as a model provider inside TensorZero**!

## [​](https://www.tensorzero.com/docs/comparison/openrouter\#similarities)  Similarities

- **Unified Inference API.**
Both TensorZero and OpenRouter offer a unified inference API that allows you to access LLMs from most major model providers with a single integration, with support for structured outputs, tool use, streaming, and more.

[→ TensorZero Gateway Quickstart](https://www.tensorzero.com/docs/quickstart)
- **Automatic Fallbacks for Higher Reliability.**
Both TensorZero and OpenRouter offer automatic fallbacks to increase reliability.

[→ Retries & Fallbacks with TensorZero](https://www.tensorzero.com/docs/gateway/guides/retries-fallbacks)

## [​](https://www.tensorzero.com/docs/comparison/openrouter\#key-differences)  Key Differences

### [​](https://www.tensorzero.com/docs/comparison/openrouter\#tensorzero)  TensorZero

- **Open Source & Self-Hosted.**
TensorZero is fully open source and self-hosted.
Your data never leaves your infrastructure, and you don’t risk downtime by relying on external APIs.
OpenRouter is a closed-source external API.
- **No Added Cost.**
TensorZero is free to use: your bring your own LLM API keys and there is no additional cost.
OpenRouter charges 5% of your inference spend when you bring your own API keys.
- **Built-in Observability.**
TensorZero offers built-in observability features, collecting inference and feedback data in your own database.
OpenRouter doesn’t offer any observability features.
- **Built-in Evaluations.**
TensorZero offers built-in functionality, including heuristics and LLM judges.
OpenRouter doesn’t offer any evaluation features.

[→ TensorZero Evaluations Overview](https://www.tensorzero.com/docs/evaluations)
- **Automated Experimentation (A/B Testing).**
TensorZero offers built-in experimentation features, allowing you to run experiments on your prompts, models, and inference strategies.
OpenRouter doesn’t offer any experimentation features.

[→ Run adaptive A/B tests with TensorZero](https://www.tensorzero.com/docs/experimentation/run-adaptive-ab-tests)
- **Built-in Inference-Time Optimizations.**
TensorZero offers built-in inference-time optimizations (e.g. dynamic in-context learning), allowing you to optimize your inference performance.
OpenRouter doesn’t offer any inference-time optimizations, except for dynamic model routing via NotDiamond.

[→ Inference-Time Optimizations with TensorZero](https://www.tensorzero.com/docs/gateway/guides/inference-time-optimizations)
- **Optimization Recipes.**
TensorZero offers optimization recipes (e.g. supervised fine-tuning, RLHF, MIPRO) that leverage your own data to improve your LLM’s performance.
OpenRouter doesn’t offer any features like this.

[→ Optimization Recipes with TensorZero](https://www.tensorzero.com/docs/recipes)
- **Batch Inference.**
TensorZero supports batch inference with certain model providers, which significantly reduces inference costs.
OpenRouter doesn’t support batch inference.

[→ Batch Inference with TensorZero](https://www.tensorzero.com/docs/gateway/guides/batch-inference)
- **Inference Caching.**
TensorZero offers inference caching, which can significantly reduce inference costs and latency.
OpenRouter doesn’t offer inference caching.

[→ Inference Caching with TensorZero](https://www.tensorzero.com/docs/gateway/guides/inference-caching)
- **Schemas, Templates, GitOps.**
TensorZero enables a schema-first approach to building LLM applications, allowing you to separate your application logic from LLM implementation details.
This approach allows your to more easily manage complex LLM applications, benefit from GitOps for prompt and configuration management, counterfactually improve data for optimization, and more.
OpenRouter only offers the standard unstructured chat completion interface.

[→ Prompt Templates & Schemas with TensorZero](https://www.tensorzero.com/docs/gateway/create-a-prompt-template)

### [​](https://www.tensorzero.com/docs/comparison/openrouter\#openrouter)  OpenRouter

- **Dynamic Provider Routing.**
OpenRouter allows you to dynamically route requests to different model providers based on latency, cost, and availability.
TensorZero only offers static routing capabilities, i.e. a pre-defined sequence of model providers to attempt.

[→ Retries & Fallbacks with TensorZero](https://www.tensorzero.com/docs/gateway/guides/retries-fallbacks)
- **Dynamic Model Routing.**
OpenRouter integrates with NotDiamond to offer dynamic model routing based on input.
TensorZero supports other inference-time optimizations but doesn’t support dynamic model routing at this time.

[→ Inference-Time Optimizations with TensorZero](https://www.tensorzero.com/docs/gateway/guides/inference-time-optimizations)
- **Consolidated Billing.**
OpenRouter allows you to access every supported model using a single OpenRouter API key.
Under the hood, OpenRouter uses their own API keys with model providers.
This approach can increase your rate limits and streamline billing, but slightly increases your inference costs.
TensorZero requires you to use your own API keys, without any added cost.

Is TensorZero missing any features that are really important to you? Let us know on [GitHub Discussions](https://github.com/tensorzero/tensorzero/discussions), [Slack](https://www.tensorzero.com/slack), or [Discord](https://www.tensorzero.com/discord).

## [​](https://www.tensorzero.com/docs/comparison/openrouter\#combining-tensorzero-and-openrouter)  Combining TensorZero and OpenRouter

You can get the best of both worlds by using OpenRouter as a model provider inside TensorZero.
Learn more about using [OpenRouter as a model provider](https://www.tensorzero.com/docs/integrations/model-providers/openrouter).

[OpenPipe](https://www.tensorzero.com/docs/comparison/openpipe) [Portkey](https://www.tensorzero.com/docs/comparison/portkey)

Ctrl+I

Assistant

Responses are generated using AI and may contain mistakes.

## TensorZero vs Portkey
[Skip to main content](https://www.tensorzero.com/docs/comparison/portkey#content-area)

[TensorZero Docs home page![light logo](https://mintcdn.com/tensorzero/afFfERJmQ8YiZpCj/logo/light.svg?fit=max&auto=format&n=afFfERJmQ8YiZpCj&q=85&s=9396731a8a92cee7bd293dee05d71dac)![dark logo](https://mintcdn.com/tensorzero/afFfERJmQ8YiZpCj/logo/dark.svg?fit=max&auto=format&n=afFfERJmQ8YiZpCj&q=85&s=8d61c59531dbcdcace7fcdfc524bd4d2)](https://www.tensorzero.com/)

Search...

Ctrl K

Search...

Navigation

Comparison

Comparison: TensorZero vs. Portkey

[Guides](https://www.tensorzero.com/docs) [Integrations](https://www.tensorzero.com/docs/integrations/model-providers)

On this page

- [Similarities](https://www.tensorzero.com/docs/comparison/portkey#similarities)
- [Key Differences](https://www.tensorzero.com/docs/comparison/portkey#key-differences)
- [TensorZero](https://www.tensorzero.com/docs/comparison/portkey#tensorzero)
- [Portkey](https://www.tensorzero.com/docs/comparison/portkey#portkey)

TensorZero and Portkey offer diverse features to streamline LLM engineering, including an LLM gateway, observability tools, and more.
TensorZero is fully open-source and self-hosted, while Portkey offers an open-source gateway but otherwise requires a paid commercial (hosted) service.
Additionally, TensorZero has more features around LLM optimization (e.g. advanced fine-tuning workflows and inference-time optimizations), whereas Portkey has a broader set of features around the UI (e.g. prompt playground).

## [​](https://www.tensorzero.com/docs/comparison/portkey\#similarities)  Similarities

- **Unified Inference API.**
Both TensorZero and Portkey offer a unified inference API that allows you to access LLMs from most major model providers with a single integration, with support for structured outputs, batch inference, tool use, streaming, and more.

[→ TensorZero Gateway Quickstart](https://www.tensorzero.com/docs/quickstart)
- **Automatic Fallbacks, Retries, & Load Balancing for Higher Reliability.**
Both TensorZero and Portkey offer automatic fallbacks, retries, and load balancing features to increase reliability.

[→ Retries & Fallbacks with TensorZero](https://www.tensorzero.com/docs/gateway/guides/retries-fallbacks)
- **Schemas, Templates.**
Both TensorZero and Portkey offer schema and template features to help you manage your LLM applications.

[→ Prompt Templates & Schemas with TensorZero](https://www.tensorzero.com/docs/gateway/create-a-prompt-template)
- **Multimodal Inference.**
Both TensorZero and Portkey support multimodal inference.

[→ Multimodal Inference with TensorZero](https://www.tensorzero.com/docs/gateway/guides/multimodal-inference)

## [​](https://www.tensorzero.com/docs/comparison/portkey\#key-differences)  Key Differences

### [​](https://www.tensorzero.com/docs/comparison/portkey\#tensorzero)  TensorZero

- **Open-Source Observability.**
TensorZero offers built-in open-source observability features, collecting inference and feedback data in your own database.
Portkey also offers observability features, but they are limited to their commercial (hosted) offering.
- **Built-in Evaluations.**
TensorZero offers built-in evaluation functionality, including heuristics and LLM judges.
Portkey doesn’t offer any evaluation features.

[→ TensorZero Evaluations Overview](https://www.tensorzero.com/docs/evaluations)
- **Open-Source Inference Caching.**
TensorZero offers open-source inference caching features, allowing you to cache requests to improve latency and reduce costs.
Portkey also offers inference caching features, but they are limited to their commercial (hosted) offering.

[→ Inference Caching with TensorZero](https://www.tensorzero.com/docs/gateway/guides/inference-caching)
- **Open-Source Fine-Tuning Workflows.**
TensorZero offers open-source built-in fine-tuning workflows, allowing you to create custom models using your own data.
Portkey also offers fine-tuning features, but they are limited to their enterprise ($$$) offering.

[→ Fine-Tuning Recipes with TensorZero](https://www.tensorzero.com/docs/recipes)
- **Advanced Fine-Tuning Workflows.**
TensorZero offers advanced fine-tuning workflows, including the ability to curate datasets using feedback signals (e.g. production metrics) and the ability to use RLHF for reinforcement learning.
Portkey doesn’t offer similar features.

[→ Fine-Tuning Recipes with TensorZero](https://www.tensorzero.com/docs/recipes)
- **Automated Experimentation (A/B Testing).**
TensorZero offers advanced A/B testing features, including automated experimentation, to help your identify the best models and prompts for your use cases.
Portkey only offers simple canary and A/B testing features.

[→ Run adaptive A/B tests with TensorZero](https://www.tensorzero.com/docs/experimentation/run-adaptive-ab-tests)
- **Inference-Time Optimizations.**
TensorZero offers built-in inference-time optimizations (e.g. dynamic in-context learning), allowing you to optimize your inference performance.
Portkey doesn’t offer any inference-time optimizations.

[→ Inference-Time Optimizations with TensorZero](https://www.tensorzero.com/docs/gateway/guides/inference-time-optimizations)
- **Programmatic & GitOps-Friendly Orchestration.**
TensorZero can be fully orchestrated programmatically in a GitOps-friendly way.
Portkey can manage some of its features programmatically, but certain features depend on its external commercial hosted service.
- **Open-Source Access Control.**
Both TensorZero and Portkey offer access control features like TensorZero API keys.
Portkey only offers them in the commercial (hosted) offering, whereas TensorZero’s solution is fully open-source.

[→ Set up auth for TensorZero](https://www.tensorzero.com/docs/operations/set-up-auth-for-tensorzero)

### [​](https://www.tensorzero.com/docs/comparison/portkey\#portkey)  Portkey

- **Prompt Playground.**
Portkey offers a prompt playground in its commercial (hosted) offering, allowing you to test your prompts and models in a graphical interface.
TensorZero doesn’t offer a prompt playground today (coming soon!).
- **Guardrails.**
Portkey offers guardrails features, including integrations with third-party guardrails providers and the ability to use custom guardrails using webhooks.
For now, TensorZero doesn’t offer built-in guardrails, and instead requires you to manage integrations yourself.
- **Managed Service.**
Portkey offers a paid managed (hosted) service in addition to the open-source version.
TensorZero is fully open-source and self-hosted.

[OpenRouter](https://www.tensorzero.com/docs/comparison/openrouter) [Overview](https://www.tensorzero.com/docs/gateway)

Ctrl+I

Assistant

Responses are generated using AI and may contain mistakes.

## Deploy ClickHouse with TensorZero
[Skip to main content](https://www.tensorzero.com/docs/deployment/clickhouse#content-area)

[TensorZero Docs home page![light logo](https://mintcdn.com/tensorzero/afFfERJmQ8YiZpCj/logo/light.svg?fit=max&auto=format&n=afFfERJmQ8YiZpCj&q=85&s=9396731a8a92cee7bd293dee05d71dac)![dark logo](https://mintcdn.com/tensorzero/afFfERJmQ8YiZpCj/logo/dark.svg?fit=max&auto=format&n=afFfERJmQ8YiZpCj&q=85&s=8d61c59531dbcdcace7fcdfc524bd4d2)](https://www.tensorzero.com/)

Search...

Ctrl K

Search...

Navigation

Deployment

Deploy ClickHouse (optional)

[Guides](https://www.tensorzero.com/docs) [Integrations](https://www.tensorzero.com/docs/integrations/model-providers)

On this page

- [Deploy](https://www.tensorzero.com/docs/deployment/clickhouse#deploy)
- [Development](https://www.tensorzero.com/docs/deployment/clickhouse#development)
- [Production](https://www.tensorzero.com/docs/deployment/clickhouse#production)
- [Managed deployments](https://www.tensorzero.com/docs/deployment/clickhouse#managed-deployments)
- [Self-hosted deployments](https://www.tensorzero.com/docs/deployment/clickhouse#self-hosted-deployments)
- [Configure](https://www.tensorzero.com/docs/deployment/clickhouse#configure)
- [Connect to ClickHouse](https://www.tensorzero.com/docs/deployment/clickhouse#connect-to-clickhouse)
- [Apply ClickHouse migrations](https://www.tensorzero.com/docs/deployment/clickhouse#apply-clickhouse-migrations)

The TensorZero Gateway can optionally collect inference and feedback data for observability, optimization, evaluation, and experimentation.
Under the hood, TensorZero stores this data in ClickHouse, an open-source columnar database that is optimized for analytical workloads.

If you’re planning to use the gateway without observability, you don’t need to
deploy ClickHouse.

## [​](https://www.tensorzero.com/docs/deployment/clickhouse\#deploy)  Deploy

### [​](https://www.tensorzero.com/docs/deployment/clickhouse\#development)  Development

For development purposes, you can run a single-node ClickHouse instance locally (e.g. using Homebrew or Docker) or a cheap Development-tier cluster on ClickHouse Cloud.See the [ClickHouse documentation](https://clickhouse.com/docs/install) for more details on configuring your ClickHouse deployment.

### [​](https://www.tensorzero.com/docs/deployment/clickhouse\#production)  Production

#### [​](https://www.tensorzero.com/docs/deployment/clickhouse\#managed-deployments)  Managed deployments

For production deployments, the easiest setup is to use a managed service like [ClickHouse Cloud](https://clickhouse.com/cloud).ClickHouse Cloud is also available through the [AWS Marketplace](https://aws.amazon.com/marketplace/pp/prodview-jettukeanwrfc), [GCP Marketplace](https://console.cloud.google.com/marketplace/product/clickhouse-public/clickhouse-cloud), and [Azure Marketplace](https://azuremarketplace.microsoft.com/en-us/marketplace/apps/clickhouse.clickhouse_cloud).Other options for managed ClickHouse deployments include [Tinybird](https://www.tinybird.co/) (serverless) and [Altinity](https://www.altinity.com/) (hands-on support).

TensorZero tests against ClickHouse Cloud’s `regular` (recommended) and `fast` release channels.

#### [​](https://www.tensorzero.com/docs/deployment/clickhouse\#self-hosted-deployments)  Self-hosted deployments

You can alternatively run your own self-managed ClickHouse instance or cluster.

**We strongly recommend using ClickHouse `lts` instead of `latest` in production.**We test against both versions, but ClickHouse `latest` often has bugs and breaking changes.

TensorZero supports single-node and replicated deployments.

TensorZero does not currently support **sharded** self-hosted ClickHouse deployments.

See the [ClickHouse documentation](https://clickhouse.com/docs/install) for more details on configuring your ClickHouse deployment.

## [​](https://www.tensorzero.com/docs/deployment/clickhouse\#configure)  Configure

### [​](https://www.tensorzero.com/docs/deployment/clickhouse\#connect-to-clickhouse)  Connect to ClickHouse

To configure TensorZero to use ClickHouse, set the `TENSORZERO_CLICKHOUSE_URL` environment variable with your ClickHouse connection details.

.env

Copy

```
TENSORZERO_CLICKHOUSE_URL="http[s]://[username]:[password]@[hostname]:[port]/[database]"

# Example: ClickHouse running locally
TENSORZERO_CLICKHOUSE_URL="http://chuser:chpassword@localhost:8123/tensorzero"

# Example: ClickHouse Cloud
TENSORZERO_CLICKHOUSE_URL="https://USERNAME:PASSWORD@XXXXX.clickhouse.cloud:8443/tensorzero"

# Example: TensorZero Gateway running in a container, ClickHouse running on host machine
TENSORZERO_CLICKHOUSE_URL="http://host.docker.internal:8123/tensorzero"
```

If you’re using a self-hosted replicated ClickHouse deployment, you must also provide the ClickHouse cluster name in the `TENSORZERO_CLICKHOUSE_CLUSTER_NAME` environment variable.

### [​](https://www.tensorzero.com/docs/deployment/clickhouse\#apply-clickhouse-migrations)  Apply ClickHouse migrations

By default, the TensorZero Gateway applies ClickHouse migrations automatically when it starts up.
This behavior can be suppressed by setting `observability.disable_automatic_migrations = true` under the `[gateway]` section of `config/tensorzero.toml`.
See [https://www.tensorzero.com/docs/gateway/configuration-reference#gateway](https://www.tensorzero.com/docs/gateway/configuration-reference#gateway).
If automatic migrations are disabled, then you must apply them manually with
`docker run --rm -e TENSORZERO_CLICKHOUSE_URL=$TENSORZERO_CLICKHOUSE_URL tensorzero/gateway:{version} --run-clickhouse-migrations`.
The gateway will error on startup if automatic migrations are disabled and any required migrations are missing.If you’re using a self-hosted replicated ClickHouse deployment, you must apply database migrations manually;
they cannot be applied automatically.

[Deploy the TensorZero UI](https://www.tensorzero.com/docs/deployment/tensorzero-ui) [Deploy Postgres (optional)](https://www.tensorzero.com/docs/deployment/postgres)

Ctrl+I

Assistant

Responses are generated using AI and may contain mistakes.

## Optimize Latency & Throughput
[Skip to main content](https://www.tensorzero.com/docs/deployment/optimize-latency-and-throughput#content-area)

[TensorZero Docs home page![light logo](https://mintcdn.com/tensorzero/afFfERJmQ8YiZpCj/logo/light.svg?fit=max&auto=format&n=afFfERJmQ8YiZpCj&q=85&s=9396731a8a92cee7bd293dee05d71dac)![dark logo](https://mintcdn.com/tensorzero/afFfERJmQ8YiZpCj/logo/dark.svg?fit=max&auto=format&n=afFfERJmQ8YiZpCj&q=85&s=8d61c59531dbcdcace7fcdfc524bd4d2)](https://www.tensorzero.com/)

Search...

Ctrl K

Search...

Navigation

Deployment

Optimize latency and throughput

[Guides](https://www.tensorzero.com/docs) [Integrations](https://www.tensorzero.com/docs/integrations/model-providers)

On this page

- [Best practices](https://www.tensorzero.com/docs/deployment/optimize-latency-and-throughput#best-practices)
- [Observability data collection strategy](https://www.tensorzero.com/docs/deployment/optimize-latency-and-throughput#observability-data-collection-strategy)
- [Other recommendations](https://www.tensorzero.com/docs/deployment/optimize-latency-and-throughput#other-recommendations)

The TensorZero Gateway is designed from the ground up with performance in mind.
Even with default settings, the gateway is fast and lightweight enough to be unnoticeable in most applications.
The best practices below are designed to help you optimize the performance of the TensorZero Gateway for production deployments requiring maximum performance.

The TensorZero Gateway can achieve <1ms P99 latency overhead at 10,000+ QPS. See [Benchmarks](https://www.tensorzero.com/docs/gateway/benchmarks) for details.

## [​](https://www.tensorzero.com/docs/deployment/optimize-latency-and-throughput\#best-practices)  Best practices

### [​](https://www.tensorzero.com/docs/deployment/optimize-latency-and-throughput\#observability-data-collection-strategy)  Observability data collection strategy

By default, the gateway takes a conservative approach to observability data durability, ensuring that data is persisted in ClickHouse before sending a response to the client.
This strategy provides a consistent and reliable experience but can introduce latency overhead.For scenarios where latency and throughput are critical, the gateway can be configured to sacrifice data durability guarantees for better performance.
If latency is critical for your application, you can enable `gateway.observability.async_writes` or `gateway.observability.batch_writes`.
With either of these settings, the gateway will return the response to the client immediately and asynchronously insert data into ClickHouse.
The former will immediately insert each row individually, while the latter will batch multiple rows together for more efficient writes.As a rule of thumb, consider the following decision matrix:

|  | **High throughput** | **Low throughput** |
| --- | --- | --- |
| **Latency is critical** | `batch_writes` | `async_writes` |
| **Latency is not critical** | `batch_writes` | Default strategy |

See the [Configuration Reference](https://www.tensorzero.com/docs/gateway/configuration-reference) for more details.

### [​](https://www.tensorzero.com/docs/deployment/optimize-latency-and-throughput\#other-recommendations)  Other recommendations

- Ensure your application, the TensorZero Gateway, and ClickHouse are deployed in the same region to minimize network latency.
- Initialize the client once and reuse it as much as possible, to avoid initialization overhead and to keep the connection alive.

[Deploy Postgres (optional)](https://www.tensorzero.com/docs/deployment/postgres) [Manage credentials (API keys)](https://www.tensorzero.com/docs/operations/manage-credentials)

Ctrl+I

Assistant

Responses are generated using AI and may contain mistakes.

## Deploying Postgres
[Skip to main content](https://www.tensorzero.com/docs/deployment/postgres#content-area)

[TensorZero Docs home page![light logo](https://mintcdn.com/tensorzero/afFfERJmQ8YiZpCj/logo/light.svg?fit=max&auto=format&n=afFfERJmQ8YiZpCj&q=85&s=9396731a8a92cee7bd293dee05d71dac)![dark logo](https://mintcdn.com/tensorzero/afFfERJmQ8YiZpCj/logo/dark.svg?fit=max&auto=format&n=afFfERJmQ8YiZpCj&q=85&s=8d61c59531dbcdcace7fcdfc524bd4d2)](https://www.tensorzero.com/)

Search...

Ctrl K

Search...

Navigation

Deployment

Deploy Postgres (optional)

[Guides](https://www.tensorzero.com/docs) [Integrations](https://www.tensorzero.com/docs/integrations/model-providers)

On this page

- [Deploy](https://www.tensorzero.com/docs/deployment/postgres#deploy)
- [Configure](https://www.tensorzero.com/docs/deployment/postgres#configure)
- [Connect to Postgres](https://www.tensorzero.com/docs/deployment/postgres#connect-to-postgres)
- [Apply Postgres migrations](https://www.tensorzero.com/docs/deployment/postgres#apply-postgres-migrations)

**Most TensorZero deployments will not require Postgres.**

TensorZero only requires Postgres for certain advanced features.
Most notably, you need to deploy Postgres to [enforce custom rate limits](https://www.tensorzero.com/docs/operations/enforce-custom-rate-limits), [run adaptive A/B tests](https://www.tensorzero.com/docs/experimentation/run-adaptive-ab-tests), and [set up auth for TensorZero](https://www.tensorzero.com/docs/operations/set-up-auth-for-tensorzero).

## [​](https://www.tensorzero.com/docs/deployment/postgres\#deploy)  Deploy

You can self-host Postgres or use a managed service (e.g. AWS RDS, Supabase, PlanetScale).
Follow the deployment instructions for your chosen service.Internally, we test TensorZero using self-hosted Postgres 14.

If you find any compatibility issues, please open a detailed [GitHub Discussion](https://github.com/tensorzero/tensorzero/discussions/new?category=bug-reports).

## [​](https://www.tensorzero.com/docs/deployment/postgres\#configure)  Configure

### [​](https://www.tensorzero.com/docs/deployment/postgres\#connect-to-postgres)  Connect to Postgres

To configure TensorZero to use Postgres, set the `TENSORZERO_POSTGRES_URL` environment variable with your Postgres connection details.

.env

Copy

```
TENSORZERO_POSTGRES_URL="postgres://[username]:[password]@[hostname]:[port]/[database]"

# Example:
TENSORZERO_POSTGRES_URL="postgres://myuser:mypass@localhost:5432/tensorzero"
```

### [​](https://www.tensorzero.com/docs/deployment/postgres\#apply-postgres-migrations)  Apply Postgres migrations

Unlike with ClickHouse, **TensorZero does not automatically apply Postgres migrations.**You must apply migrations manually with `gateway --run-postgres-migrations`.

- Docker Compose

- Docker


If you’ve configured the gateway with Docker Compose, you can run the migrations with:

Copy

```
docker-compose run --rm gateway --run-postgres-migrations
```

See [Deploy the TensorZero Gateway](https://www.tensorzero.com/docs/deployment/tensorzero-gateway) for more details.

[Deploy ClickHouse (optional)](https://www.tensorzero.com/docs/deployment/clickhouse) [Optimize latency & throughput](https://www.tensorzero.com/docs/deployment/optimize-latency-and-throughput)

Ctrl+I

Assistant

Responses are generated using AI and may contain mistakes.

## TensorZero Gateway Deployment
[Skip to main content](https://www.tensorzero.com/docs/deployment/tensorzero-gateway#content-area)

[TensorZero Docs home page![light logo](https://mintcdn.com/tensorzero/afFfERJmQ8YiZpCj/logo/light.svg?fit=max&auto=format&n=afFfERJmQ8YiZpCj&q=85&s=9396731a8a92cee7bd293dee05d71dac)![dark logo](https://mintcdn.com/tensorzero/afFfERJmQ8YiZpCj/logo/dark.svg?fit=max&auto=format&n=afFfERJmQ8YiZpCj&q=85&s=8d61c59531dbcdcace7fcdfc524bd4d2)](https://www.tensorzero.com/)

Search...

Ctrl K

Search...

Navigation

Deployment

Deploy the TensorZero Gateway

[Guides](https://www.tensorzero.com/docs) [Integrations](https://www.tensorzero.com/docs/integrations/model-providers)

On this page

- [Deploy](https://www.tensorzero.com/docs/deployment/tensorzero-gateway#deploy)
- [Configure](https://www.tensorzero.com/docs/deployment/tensorzero-gateway#configure)
- [Set up model provider credentials](https://www.tensorzero.com/docs/deployment/tensorzero-gateway#set-up-model-provider-credentials)
- [Set up custom configuration](https://www.tensorzero.com/docs/deployment/tensorzero-gateway#set-up-custom-configuration)
- [Set up observability with ClickHouse](https://www.tensorzero.com/docs/deployment/tensorzero-gateway#set-up-observability-with-clickhouse)
- [Customize the logging format](https://www.tensorzero.com/docs/deployment/tensorzero-gateway#customize-the-logging-format)
- [Add a status or health check](https://www.tensorzero.com/docs/deployment/tensorzero-gateway#add-a-status-or-health-check)

The TensorZero Gateway is the core component that handles inference requests and collects observability data.
It’s easy to get started with the TensorZero Gateway.

You need to only deploy a standalone gateway if you plan to use the TensorZero UI or interact with the gateway using programming languages other than Python.
The TensorZero Python SDK includes a built-in embedded gateway, so you don’t need to deploy a standalone gateway if you’re only using Python.
See the [Clients](https://www.tensorzero.com/docs/gateway/clients) page for more details on how to interact with the TensorZero Gateway.

## [​](https://www.tensorzero.com/docs/deployment/tensorzero-gateway\#deploy)  Deploy

The gateway requires one of the following command line arguments:

- `--default-config`: Use default configuration settings.
- `--config-file path/to/tensorzero.toml`: Use a custom configuration file.







`--config-file` supports glob patterns, e.g. `--config-file     /path/to/**/*.toml`.

- `--run-clickhouse-migrations`: Run ClickHouse database migrations and exit.
- `--run-postgres-migrations`: Run PostgreSQL database migrations and exit.

There are many ways to deploy the TensorZero Gateway.
Here are a few examples:

Run with Docker

You can easily run the TensorZero Gateway locally using Docker.If you don’t have custom configuration, you can use:

Running with Docker (default configuration)

Copy

```
docker run \
  --env-file .env \
  -p 3000:3000 \
  tensorzero/gateway \
  --default-config
```

If you have custom configuration, you can use:

Running with Docker (custom configuration)

Copy

```
docker run \
  -v "./config:/app/config" \
  --env-file .env \
  -p 3000:3000 \
  tensorzero/gateway \
  --config-path config/tensorzero.toml
```

Run with Docker Compose

We provide an example production-grade [`docker-compose.yml`](https://github.com/tensorzero/tensorzero/blob/main/examples/production-deployment/docker-compose.yml) for reference.

Run with Kubernetes (k8s) and Helm

We provide a reference Helm chart in our [GitHub repository](https://github.com/tensorzero/tensorzero/tree/main/examples/production-deployment-k8s-helm).
You can use it to run TensorZero in Kubernetes.The chart is available on [ArtifactHub](https://artifacthub.io/packages/helm/tensorzero/tensorzero).

Build from source

You can build the TensorZero Gateway from source and run it directly on your host machine using [Cargo](https://doc.rust-lang.org/cargo/).

Building from source

Copy

```
cargo run --profile performance --bin gateway -- --config-file path/to/your/tensorzero.toml
```

See the [optimizing latency and throughput](https://www.tensorzero.com/docs/deployment/optimize-latency-and-throughput) guide to learn how to configure the gateway for high-performance deployments.

## [​](https://www.tensorzero.com/docs/deployment/tensorzero-gateway\#configure)  Configure

### [​](https://www.tensorzero.com/docs/deployment/tensorzero-gateway\#set-up-model-provider-credentials)  Set up model provider credentials

The TensorZero Gateway accepts the following environment variables for provider credentials.
Unless you specify an alternative credential location in your configuration file, these environment variables are required for the providers that are used in a variant with positive weight.
If required credentials are missing, the gateway will fail on startup.Unless customized in your configuration file, the following credentials are used by default:

| Provider | Environment Variable(s) |
| --- | --- |
| Anthropic | `ANTHROPIC_API_KEY` |
| AWS Bedrock | `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY` (see [details](https://www.tensorzero.com/docs/integrations/model-providers/aws-bedrock)) |
| AWS SageMaker | `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY` (see [details](https://www.tensorzero.com/docs/integrations/model-providers/aws-sagemaker)) |
| Azure OpenAI | `AZURE_OPENAI_API_KEY` |
| Fireworks | `FIREWORKS_API_KEY` |
| GCP Vertex AI Anthropic | `GCP_VERTEX_CREDENTIALS_PATH` (see [details](https://www.tensorzero.com/docs/integrations/model-providers/gcp-vertex-ai-anthropic)) |
| GCP Vertex AI Gemini | `GCP_VERTEX_CREDENTIALS_PATH` (see [details](https://www.tensorzero.com/docs/integrations/model-providers/gcp-vertex-ai-gemini)) |
| Google AI Studio Gemini | `GOOGLE_AI_STUDIO_GEMINI_API_KEY` |
| Groq | `GROQ_API_KEY` |
| Hyperbolic | `HYPERBOLIC_API_KEY` |
| Mistral | `MISTRAL_API_KEY` |
| OpenAI | `OPENAI_API_KEY` |
| OpenRouter | `OPENROUTER_API_KEY` |
| Together | `TOGETHER_API_KEY` |
| xAI | `XAI_API_KEY` |

See [`.env.example`](https://github.com/tensorzero/tensorzero/blob/main/examples/production-deployment/.env.example) for a complete example with every supported environment variable.

### [​](https://www.tensorzero.com/docs/deployment/tensorzero-gateway\#set-up-custom-configuration)  Set up custom configuration

Optionally, you can use a configuration file to customize the behavior of the gateway.
See [Configuration Reference](https://www.tensorzero.com/docs/gateway/configuration-reference) for more details.

Disable pseudonymous usage analytics

TensorZero collects _pseudonymous_ usage analytics to help our team improve the product.The collected data includes _aggregated_ metrics about TensorZero itself, but does NOT include your application’s data.
To be explicit: TensorZero does NOT share any inference input or output.
TensorZero also does NOT share the name of any function, variant, metric, or similar application-specific identifiers.See `howdy.rs` in the GitHub repository to see exactly what usage data is collected and shared with TensorZero.To disable usage analytics, set the following configuration in the `tensorzero.toml` file:

tensorzero.toml

Copy

```
[gateway]
disable_pseudonymous_usage_analytics = true
```

Alternatively, you can also set the environment variable `TENSORZERO_DISABLE_PSEUDONYMOUS_USAGE_ANALYTICS=1`.

### [​](https://www.tensorzero.com/docs/deployment/tensorzero-gateway\#set-up-observability-with-clickhouse)  Set up observability with ClickHouse

Optionally, the TensorZero Gateway can collect inference and feedback data for observability, optimization, evaluations, and experimentation.
After [deploying ClickHouse](https://www.tensorzero.com/docs/deployment/clickhouse), you need to configure the `TENSORZERO_CLICKHOUSE_URL` environment variable with the connection details.
If you don’t provide this environment variable, observability will be disabled.We recommend setting up observability early to monitor your LLM application and collect data for future optimization, but this can be done incrementally as needed.

### [​](https://www.tensorzero.com/docs/deployment/tensorzero-gateway\#customize-the-logging-format)  Customize the logging format

Optionally, you can provide the following command line argument to customize the gateway’s logging format:

- `--log-format`: Set the logging format to either `pretty` (default) or `json`.

### [​](https://www.tensorzero.com/docs/deployment/tensorzero-gateway\#add-a-status-or-health-check)  Add a status or health check

The TensorZero Gateway exposes endpoints for status and health checks.
The `/status` endpoint checks that the gateway is running successfully.

GET /status

Copy

```
{ "status": "ok" }
```

The `/health` endpoint additionally checks that it can communicate with ClickHouse (if observability is enabled).

GET /health

Copy

```
{ "gateway": "ok", "clickhouse": "ok" }
```

[Run static A/B tests](https://www.tensorzero.com/docs/experimentation/run-static-ab-tests) [Deploy the TensorZero UI](https://www.tensorzero.com/docs/deployment/tensorzero-ui)

Ctrl+I

Assistant

Responses are generated using AI and may contain mistakes.

## Deploy TensorZero UI
[Skip to main content](https://www.tensorzero.com/docs/deployment/tensorzero-ui#content-area)

[TensorZero Docs home page![light logo](https://mintcdn.com/tensorzero/afFfERJmQ8YiZpCj/logo/light.svg?fit=max&auto=format&n=afFfERJmQ8YiZpCj&q=85&s=9396731a8a92cee7bd293dee05d71dac)![dark logo](https://mintcdn.com/tensorzero/afFfERJmQ8YiZpCj/logo/dark.svg?fit=max&auto=format&n=afFfERJmQ8YiZpCj&q=85&s=8d61c59531dbcdcace7fcdfc524bd4d2)](https://www.tensorzero.com/)

Search...

Ctrl K

Search...

Navigation

Deployment

Deploy the TensorZero UI

[Guides](https://www.tensorzero.com/docs) [Integrations](https://www.tensorzero.com/docs/integrations/model-providers)

On this page

- [Deploy](https://www.tensorzero.com/docs/deployment/tensorzero-ui#deploy)
- [Configure](https://www.tensorzero.com/docs/deployment/tensorzero-ui#configure)
- [Add a health check](https://www.tensorzero.com/docs/deployment/tensorzero-ui#add-a-health-check)
- [Customize the deployment](https://www.tensorzero.com/docs/deployment/tensorzero-ui#customize-the-deployment)

The TensorZero UI is a self-hosted web application that streamlines the use of TensorZero with features like observability and optimization.
It’s easy to get started with the TensorZero UI.

## [​](https://www.tensorzero.com/docs/deployment/tensorzero-ui\#deploy)  Deploy

1

Configure the TensorZero Gateway

[Deploy the TensorZero Gateway](https://www.tensorzero.com/docs/deployment/tensorzero-gateway) and configure `TENSORZERO_GATEWAY_URL`.For example, if the gateway is running locally, you can set `TENSORZERO_GATEWAY_URL=http://localhost:3000`.

2

Configure ClickHouse

[Deploy ClickHouse](https://www.tensorzero.com/docs/deployment/tensorzero-gateway) and configure `TENSORZERO_CLICKHOUSE_URL`.

3

Configure model provider credentials

The TensorZero UI integrates with model providers like OpenAI to streamline workflows like fine-tuning.
To use these features, you need to provide credentials for the relevant model providers as environment variables.
You don’t need to provide credentials if you’re not using the fine-tuning features for those providers.The supported fine-tuning providers and their required credentials (environment variables) are:

| Provider | Required Credentials |
| --- | --- |
| Fireworks AI | `FIREWORKS_ACCOUNT_ID``FIREWORKS_API_KEY` |
| GCP Vertex | GCP account credentials |
| OpenAI | `OPENAI_API_KEY` |
| Together AI | `TOGETHER_API_KEY` |

4

Deploy the TensorZero UI

The TensorZero UI is available on Docker Hub as `tensorzero/ui`.

Running with Docker Compose

You can easily run the TensorZero UI using Docker Compose:

Copy

```
services:
  ui:
    image: tensorzero/ui
    # Mount your configuration folder (e.g. tensorzero.toml) to /app/config
    volumes:
      - ./config:/app/config:ro
    # Add your environment variables the .env file
    env_file:
      - ${ENV_FILE:-.env}
    # Publish the UI to port 4000
    ports:
      - "4000:4000"
    restart: unless-stopped
```

Make sure to create a `.env` file with the relevant environment variables.For more details, see the example `docker-compose.yml` file in the [GitHub repository](https://github.com/tensorzero/tensorzero/blob/main/ui/docker-compose.yml).

Running with Docker

Alternatively, you can launch the UI directly with the following command:

Copy

```
docker run \
    --volume ./config:/app/config:ro \
    --env-file ./.env \
    --publish 4000:4000 \
    tensorzero/ui
```

Make sure to create a `.env` file with the relevant environment variables.

Running with Kubernetes (k8s) and Helm

We provide a reference Helm chart in our [GitHub repository](https://github.com/tensorzero/tensorzero/tree/main/examples/production-deployment-k8s-helm).
You can use it to run TensorZero in Kubernetes.The chart is available on [ArtifactHub](https://artifacthub.io/packages/helm/tensorzero/tensorzero).

Building from source

Alternatively, you can build the UI from source.
See our [GitHub repository](https://github.com/tensorzero/tensorzero/blob/main/ui/) for more details.

## [​](https://www.tensorzero.com/docs/deployment/tensorzero-ui\#configure)  Configure

### [​](https://www.tensorzero.com/docs/deployment/tensorzero-ui\#add-a-health-check)  Add a health check

The TensorZero UI exposes an endpoint for health checks.
This `/health` endpoint checks that the UI is running, the associated configuration is valid, and the ClickHouse connection is healthy.

### [​](https://www.tensorzero.com/docs/deployment/tensorzero-ui\#customize-the-deployment)  Customize the deployment

The TensorZero UI supports the following optional environment variables.You can set `TENSORZERO_UI_CONFIG_PATH` to a custom path to the TensorZero configuration file.
When using the official Docker image, this value defaults to `/app/config/tensorzero.toml`.For certain uncommon scenarios (e.g. IPv6), you can also customize `HOST` inside the UI container.
See the Vite documentation for more details.

[Deploy the TensorZero Gateway](https://www.tensorzero.com/docs/deployment/tensorzero-gateway) [Deploy ClickHouse (optional)](https://www.tensorzero.com/docs/deployment/clickhouse)

Ctrl+I

Assistant

Responses are generated using AI and may contain mistakes.

## TensorZero Evaluations
[Skip to main content](https://www.tensorzero.com/docs/evaluations#content-area)

[TensorZero Docs home page![light logo](https://mintcdn.com/tensorzero/afFfERJmQ8YiZpCj/logo/light.svg?fit=max&auto=format&n=afFfERJmQ8YiZpCj&q=85&s=9396731a8a92cee7bd293dee05d71dac)![dark logo](https://mintcdn.com/tensorzero/afFfERJmQ8YiZpCj/logo/dark.svg?fit=max&auto=format&n=afFfERJmQ8YiZpCj&q=85&s=8d61c59531dbcdcace7fcdfc524bd4d2)](https://www.tensorzero.com/)

Search...

Ctrl K

Search...

Navigation

Evaluations

TensorZero Evaluations Overview

[Guides](https://www.tensorzero.com/docs) [Integrations](https://www.tensorzero.com/docs/integrations/model-providers)

TensorZero offers two types of evaluations:**Inference Evaluations** focus on evaluating the performance of a TensorZero variant (i.e. a choice of prompt, model, inference strategy, etc.) on a given dataset.**Workflow Evaluations** focus on evaluating complex workflows that might include multiple TensorZero inference calls, arbitrary application logic, and more.As a vague analogy, inference evaluations are like unit tests for individual inference calls, and workflow evaluations are like integration tests for complex workflows.

* * *

[**Tutorial: Inference Evaluations**](https://www.tensorzero.com/docs/evaluations/inference-evaluations/tutorial) [**Tutorial: Workflow Evaluations**](https://www.tensorzero.com/docs/evaluations/workflow-evaluations/tutorial)

[Overview](https://www.tensorzero.com/docs/recipes) [Tutorial](https://www.tensorzero.com/docs/evaluations/inference-evaluations/tutorial)

Ctrl+I

Assistant

Responses are generated using AI and may contain mistakes.

## TensorZero CLI Reference
[Skip to main content](https://www.tensorzero.com/docs/evaluations/inference-evaluations/cli-reference#content-area)

[TensorZero Docs home page![light logo](https://mintcdn.com/tensorzero/afFfERJmQ8YiZpCj/logo/light.svg?fit=max&auto=format&n=afFfERJmQ8YiZpCj&q=85&s=9396731a8a92cee7bd293dee05d71dac)![dark logo](https://mintcdn.com/tensorzero/afFfERJmQ8YiZpCj/logo/dark.svg?fit=max&auto=format&n=afFfERJmQ8YiZpCj&q=85&s=8d61c59531dbcdcace7fcdfc524bd4d2)](https://www.tensorzero.com/)

Search...

Ctrl K

Search...

Navigation

Inference Evaluations

CLI Reference

[Guides](https://www.tensorzero.com/docs) [Integrations](https://www.tensorzero.com/docs/integrations/model-providers)

On this page

- [Usage](https://www.tensorzero.com/docs/evaluations/inference-evaluations/cli-reference#usage)
- [Inference Caching](https://www.tensorzero.com/docs/evaluations/inference-evaluations/cli-reference#inference-caching)
- [Environment Variables](https://www.tensorzero.com/docs/evaluations/inference-evaluations/cli-reference#environment-variables)
- [TENSORZERO\_CLICKHOUSE\_URL](https://www.tensorzero.com/docs/evaluations/inference-evaluations/cli-reference#tensorzero-clickhouse-url)
- [Model Provider Credentials](https://www.tensorzero.com/docs/evaluations/inference-evaluations/cli-reference#model-provider-credentials)
- [CLI Flags](https://www.tensorzero.com/docs/evaluations/inference-evaluations/cli-reference#cli-flags)
- [--adaptive-stopping-precision EVALUATOR=PRECISION\[,...\]](https://www.tensorzero.com/docs/evaluations/inference-evaluations/cli-reference#adaptive-stopping-precision-evaluator=precision[,-])
- [--config-file PATH](https://www.tensorzero.com/docs/evaluations/inference-evaluations/cli-reference#config-file-path)
- [--concurrency N (-c)](https://www.tensorzero.com/docs/evaluations/inference-evaluations/cli-reference#concurrency-n-c)
- [--datapoint-ids ID\[,ID,...\]](https://www.tensorzero.com/docs/evaluations/inference-evaluations/cli-reference#datapoint-ids-id[,id,-])
- [--dataset-name NAME (-d)](https://www.tensorzero.com/docs/evaluations/inference-evaluations/cli-reference#dataset-name-name-d)
- [--evaluation-name NAME (-e)](https://www.tensorzero.com/docs/evaluations/inference-evaluations/cli-reference#evaluation-name-name-e)
- [--format FORMAT (-f)](https://www.tensorzero.com/docs/evaluations/inference-evaluations/cli-reference#format-format-f)
- [--gateway-url URL](https://www.tensorzero.com/docs/evaluations/inference-evaluations/cli-reference#gateway-url-url)
- [--inference-cache MODE](https://www.tensorzero.com/docs/evaluations/inference-evaluations/cli-reference#inference-cache-mode)
- [--max-datapoints N](https://www.tensorzero.com/docs/evaluations/inference-evaluations/cli-reference#max-datapoints-n)
- [--variant-name NAME (-v)](https://www.tensorzero.com/docs/evaluations/inference-evaluations/cli-reference#variant-name-name-v)
- [Exit Status](https://www.tensorzero.com/docs/evaluations/inference-evaluations/cli-reference#exit-status)

TensorZero Evaluations is available both through a command-line interface (CLI) tool and through the TensorZero UI.

## [​](https://www.tensorzero.com/docs/evaluations/inference-evaluations/cli-reference\#usage)  Usage

We provide a `tensorzero/evaluations` Docker image for easy usage.We strongly recommend using TensorZero Evaluations CLI with Docker Compose to keep things simple.

docker-compose.yml

Copy

```
services:
  evaluations:
    profiles: [evaluations] # this service won't run by default with `docker compose up`
    image: tensorzero/evaluations
    volumes:
      - ./config:/app/config:ro
    environment:
      - OPENAI_API_KEY=${OPENAI_API_KEY:?Environment variable OPENAI_API_KEY must be set.}
      # ... and any other relevant API credentials ...
      - TENSORZERO_CLICKHOUSE_URL=http://chuser:chpassword@clickhouse:8123/tensorzero
    extra_hosts:
      - "host.docker.internal:host-gateway"
    depends_on:
      clickhouse:
        condition: service_healthy
```

Copy

```
docker compose run --rm evaluations \
    --evaluation-name haiku_eval \
    --dataset-name haiku_dataset \
    --variant-name gpt_4o \
    --concurrency 5
```

Building from Source

You can build the TensorZero Evaluations CLI from source if necessary. See our [GitHub repository](https://github.com/tensorzero/tensorzero/tree/main/evaluations) for instructions.

### [​](https://www.tensorzero.com/docs/evaluations/inference-evaluations/cli-reference\#inference-caching)  Inference Caching

TensorZero Evaluations uses [Inference Caching](https://www.tensorzero.com/docs/gateway/guides/inference-caching) to improve inference speed and cost.By default, it will read from and write to the inference cache.
Soon, you’ll be able to customize this behavior.

### [​](https://www.tensorzero.com/docs/evaluations/inference-evaluations/cli-reference\#environment-variables)  Environment Variables

#### [​](https://www.tensorzero.com/docs/evaluations/inference-evaluations/cli-reference\#tensorzero-clickhouse-url)  `TENSORZERO_CLICKHOUSE_URL`

- **Example:**`TENSORZERO_CLICKHOUSE_URL=http://chuser:chpassword@localhost:8123/database_name`
- **Required:** yes

This environment variable specifies the URL of your ClickHouse database.

#### [​](https://www.tensorzero.com/docs/evaluations/inference-evaluations/cli-reference\#model-provider-credentials)  Model Provider Credentials

- **Example:**`OPENAI_API_KEY=sk-...`
- **Required:** no

If you’re using an external TensorZero Gateway (see `--gateway-url` flag below), you don’t need to provide these credentials to the evaluations tool.If you’re using a built-in gateway (no `--gateway-url` flag), you must provide same credentials the gateway would use.
See [Integrations](https://www.tensorzero.com/docs/integrations/model-providers) for more information.

### [​](https://www.tensorzero.com/docs/evaluations/inference-evaluations/cli-reference\#cli-flags)  CLI Flags

#### [​](https://www.tensorzero.com/docs/evaluations/inference-evaluations/cli-reference\#adaptive-stopping-precision-evaluator=precision[,-])  `--adaptive-stopping-precision EVALUATOR=PRECISION[,...]`

- **Example:**`--adaptive-stopping-precision exact_match=0.13,llm_judge=0.16`
- **Required:** no (default: none)

This flag enables adaptive stopping for specified evaluators by setting per-evaluator precision thresholds.
An evaluator stops when both sides of its 95% confidence interval are within the threshold of its mean value.You can specify multiple evaluators by separating them with commas.
Each evaluator’s precision threshold should be a positive number.If adaptive stopping is enabled for all evaluators, then the evaluation will stop once all evaluators have met their targets or all datapoints have been evaluated.

#### [​](https://www.tensorzero.com/docs/evaluations/inference-evaluations/cli-reference\#config-file-path)  `--config-file PATH`

- **Example:**`--config-file /path/to/tensorzero.toml`
- **Required:** no (default: `./config/tensorzero.toml`)

This flag specifies the path to the TensorZero configuration file.
You should use the same configuration file for your entire project.

#### [​](https://www.tensorzero.com/docs/evaluations/inference-evaluations/cli-reference\#concurrency-n-c)  `--concurrency N` (`-c`)

- **Example:**`--concurrency 5`
- **Required:** no (default: `1`)

This flag specifies the maximum number of concurrent TensorZero inference requests during evaluation.

#### [​](https://www.tensorzero.com/docs/evaluations/inference-evaluations/cli-reference\#datapoint-ids-id[,id,-])  `--datapoint-ids ID[,ID,...]`

- **Example:**`--datapoint-ids 01957bbb-44a8-7490-bfe7-32f8ed2fc797,01957bbb-44a8-7490-bfe7-32f8ed2fc798`
- **Required:** Either `--dataset-name` or `--datapoint-ids` must be provided (but not both)

This flag allows you to specify individual datapoint IDs to evaluate.
Multiple IDs should be separated by commas.Use this flag when you want to evaluate a specific subset of datapoints rather than an entire dataset.

This flag is mutually exclusive with `--dataset-name` and `--max-datapoints`. You must provide either `--dataset-name` or `--datapoint-ids`, but not both.

#### [​](https://www.tensorzero.com/docs/evaluations/inference-evaluations/cli-reference\#dataset-name-name-d)  `--dataset-name NAME` (`-d`)

- **Example:**`--dataset-name my_dataset`
- **Required:** Either `--dataset-name` or `--datapoint-ids` must be provided (but not both)

This flag specifies the dataset to use for evaluation.
The dataset should be stored in your ClickHouse database.

This flag is mutually exclusive with `--datapoint-ids`. You must provide either `--dataset-name` or `--datapoint-ids`, but not both.

#### [​](https://www.tensorzero.com/docs/evaluations/inference-evaluations/cli-reference\#evaluation-name-name-e)  `--evaluation-name NAME` (`-e`)

- **Example:**`--evaluation-name my_evaluation`
- **Required:** yes

This flag specifies the name of the evaluation to run, as defined in your TensorZero configuration file.

#### [​](https://www.tensorzero.com/docs/evaluations/inference-evaluations/cli-reference\#format-format-f)  `--format FORMAT` (`-f`)

- **Options:**`pretty`, `jsonl`
- **Example:**`--format jsonl`
- **Required:** no (default: `pretty`)

This flag specifies the output format for the evaluation CLI tool.You can use the `jsonl` format if you want to programatically process the evaluation results.

#### [​](https://www.tensorzero.com/docs/evaluations/inference-evaluations/cli-reference\#gateway-url-url)  `--gateway-url URL`

- **Example:**`--gateway-url http://localhost:3000`
- **Required:** no (default: none)

If you provide this flag, the evaluations tool will use an external TensorZero Gateway for inference requests.If you don’t provide this flag, the evaluations tool will use a built-in TensorZero gateway.
In this case, the evaluations tool will require the same credentials the gateway would use.
See [Integrations](https://www.tensorzero.com/docs/integrations/model-providers) for more information.

#### [​](https://www.tensorzero.com/docs/evaluations/inference-evaluations/cli-reference\#inference-cache-mode)  `--inference-cache MODE`

- **Options:**`on`, `read_only`, `write_only`, `off`
- **Example:**`--inference-cache read_only`
- **Required:** no (default: `on`)

This flag specifies the behavior of the inference cache.
See [Inference Caching](https://www.tensorzero.com/docs/gateway/guides/inference-caching) for more information.

#### [​](https://www.tensorzero.com/docs/evaluations/inference-evaluations/cli-reference\#max-datapoints-n)  `--max-datapoints N`

- **Example:**`--max-datapoints 100`
- **Required:** no

This flag specifies the maximum number of datapoints to evaluate from the dataset.

This flag can only be used with `--dataset-name`. It cannot be used with `--datapoint-ids`.

#### [​](https://www.tensorzero.com/docs/evaluations/inference-evaluations/cli-reference\#variant-name-name-v)  `--variant-name NAME` (`-v`)

- **Example:**`--variant-name gpt_4o`
- **Required:** yes

This flag specifies the variant to evaluate.
The variant name should be present in your TensorZero configuration file.

### [​](https://www.tensorzero.com/docs/evaluations/inference-evaluations/cli-reference\#exit-status)  Exit Status

The evaluations process exits with a status code of `0` if the evaluation was successful, and a status code of `1` if the evaluation failed.If you configure a `cutoff` for any of your evaluators, the evaluation will fail if the average score for any evaluator is below its cutoff.

The exit status code is helpful for integrating TensorZero Evaluations into your CI/CD pipeline.You can define sanity checks for your variants with `cutoff` to detect performance regressions early before shipping to production.

[Configuration Reference](https://www.tensorzero.com/docs/evaluations/inference-evaluations/configuration-reference) [Tutorial](https://www.tensorzero.com/docs/evaluations/workflow-evaluations/tutorial)

Ctrl+I

Assistant

Responses are generated using AI and may contain mistakes.

## TensorZero Inference Config
[Skip to main content](https://www.tensorzero.com/docs/evaluations/inference-evaluations/configuration-reference#content-area)

[TensorZero Docs home page![light logo](https://mintcdn.com/tensorzero/afFfERJmQ8YiZpCj/logo/light.svg?fit=max&auto=format&n=afFfERJmQ8YiZpCj&q=85&s=9396731a8a92cee7bd293dee05d71dac)![dark logo](https://mintcdn.com/tensorzero/afFfERJmQ8YiZpCj/logo/dark.svg?fit=max&auto=format&n=afFfERJmQ8YiZpCj&q=85&s=8d61c59531dbcdcace7fcdfc524bd4d2)](https://www.tensorzero.com/)

Search...

Ctrl K

Search...

Navigation

Inference Evaluations

Configuration Reference

[Guides](https://www.tensorzero.com/docs) [Integrations](https://www.tensorzero.com/docs/integrations/model-providers)

On this page

- [\[evaluations.evaluation\_name\]](https://www.tensorzero.com/docs/evaluations/inference-evaluations/configuration-reference#[evaluations-evaluation-name])
- [type](https://www.tensorzero.com/docs/evaluations/inference-evaluations/configuration-reference#type)
- [function\_name](https://www.tensorzero.com/docs/evaluations/inference-evaluations/configuration-reference#function-name)
- [\[evaluations.evaluation\_name.evaluators.evaluator\_name\]](https://www.tensorzero.com/docs/evaluations/inference-evaluations/configuration-reference#[evaluations-evaluation-name-evaluators-evaluator-name])
- [type](https://www.tensorzero.com/docs/evaluations/inference-evaluations/configuration-reference#type-2)

The configuration for TensorZero Evaluations should go in the same `tensorzero.toml` file as the rest of your TensorZero configuration.

## [​](https://www.tensorzero.com/docs/evaluations/inference-evaluations/configuration-reference\#[evaluations-evaluation-name])  `[evaluations.evaluation_name]`

The `evaluations` sub-section of the config file defines the behavior of an evaluation in TensorZero.
You can define multiple evaluations by including multiple `[evaluations.evaluation_name]` sections.If your `evaluation_name` is not a basic string, it can be escaped with quotation marks.
For example, periods are not allowed in basic strings, so you can define an evaluation named `foo.bar` as `[evaluations."foo.bar"]`.

Copy

```
// tensorzero.toml
[evaluations.email-guardrails]
# ...
```

### [​](https://www.tensorzero.com/docs/evaluations/inference-evaluations/configuration-reference\#type)  `type`

- **Type:** Literal `"inference"` (we may add other options here later on)
- **Required:** yes

### [​](https://www.tensorzero.com/docs/evaluations/inference-evaluations/configuration-reference\#function-name)  `function_name`

- **Type:** string
- **Required:** yes

This should be the name of a function defined in the `[functions]` section of the gateway config.
This value sets which function this evaluation should evaluate when run.

### [​](https://www.tensorzero.com/docs/evaluations/inference-evaluations/configuration-reference\#[evaluations-evaluation-name-evaluators-evaluator-name])  `[evaluations.evaluation_name.evaluators.evaluator_name]`

The `evaluators` sub-section defines the behavior of a particular evaluator that will be run as part of its parent evaluation.
You can define multiple evaluators by including multiple `[evaluations.evaluation_name.evaluators.evaluator_name]` sections.If your `evaluator_name` is not a basic string, it can be escaped with quotation marks.
For example, periods are not allowed in basic strings, so you can define `includes.jpg` as `[evaluations.evaluation_name.evaluators."includes.jpg"]`.

Copy

```
// tensorzero.toml
[evaluations.email-guardrails]
# ...

[evaluations.email-guardrails.evaluators."includes.jpg"]
# ...

[evaluations.email-guardrails.evaluators.check-signature]
# ...
```

#### [​](https://www.tensorzero.com/docs/evaluations/inference-evaluations/configuration-reference\#type-2)  `type`

- **Type:** string
- **Required:** yes

Defines the type of the evaluator.TensorZero currently supports the following variant types:

| Type | Description |
| --- | --- |
| `llm_judge` | Use a TensorZero function as a judge |
| `exact_match` | Evaluates whether the generated output exactly matches the reference output (skips the datapoint if unavailable). |

Copy

```
// tensorzero.toml
[evaluations.email-guardrails.evaluators.check-signature]
# ...
type = "llm_judge"
# ...
```

type: "exact\_match"

###### `cutoff`

- **Type:** float
- **Required:** no

Sets a user defined threshold at which the test is passing.
This can be useful for applications where the evaluations are run as an automated test.
If the average value of this evaluator is below the cutoff, the evaluations binary will return a nonzero status code.

type: "llm\_judge"

###### `input_format`

- **Type:** string
- **Required:** no (default: `serialized`)

Defines the format of the input provided to the LLM judge.

- `serialized`: Passes the input messages, generated output, and reference output (if included) as a single serialized string.
- `messages`: Passes the input messages, generated output, and reference output (if included) as distinct messages in the conversation history.

We only support evaluations with image data when `input_format` is set to `messages`.

Copy

```
// tensorzero.toml
[evaluations.email-guardrails.evaluators.check-signature]
# ...
type = "llm_judge"
input_format = "messages"
# ...
```

###### `output_type`

- **Type:** string
- **Required:** yes

Defines the expected data type of the evaluation result from the LLM judge.

- `float`: The judge is expected to return a floating-point number.
- `boolean`: The judge is expected to return a boolean value.

Copy

```
// tensorzero.toml
[evaluations.email-guardrails.evaluators.check-signature]
# ...
type = "llm_judge"
output_type = "float"
# ...
```

###### `include.reference_output`

- **Type:** boolean
- **Required:** no (default: `false`)

If set to `true`, the reference output associated with the evaluation datapoint will be included in the input provided to the LLM judge.
In these cases, the evaluation run will not run this evaluator for datapoints where there is no reference output.

Copy

```
// tensorzero.toml
[evaluations.email-guardrails.evaluators.check-signature]
# ...
type = "llm_judge"
include = { reference_output = true }
# ...
```

###### `optimize`

- **Type:** string
- **Required:** yes

Defines whether the metric produced by the LLM judge should be maximized or minimized.

- `max`: Higher values are better.
- `min`: Lower values are better.

Copy

```
// tensorzero.toml
[evaluations.email-guardrails.evaluators.check-signature]
# ...
type = "llm_judge"
optimize = "max"
# ...
```

###### `cutoff`

- **Type:** float
- **Required:** no

Sets a user defined threshold at which the test is passing.
This may be useful for applications where the evaluations are run as an automated test.
If the average value of this evaluator is below the cutoff (when `optimize` is `max`) or above the cutoff (when `optimize` is `min`), the evaluations binary will return a nonzero status code.

Copy

```
// tensorzero.toml
[evaluations.email-guardrails.evaluators.check-signature]
# ...
type = "llm_judge"
optimize = "max" # Example: Maximize score
cutoff = 0.8 # Example: Consider passing if average score is >= 0.8
# ...
```

###### `[evaluations.evaluation_name.evaluators.evaluator_name.variants.variant_name]`

An LLM Judge evaluator defines a TensorZero function that is used to judge the output of another TensorZero function.
Therefore, all the variant types that are available for a normal TensorZero function are also available for LLMs as judges — including all of our [inference-time optimizations](https://www.tensorzero.com/docs/gateway/guides/inference-time-optimizations).You can include a standard [variant configuration](https://www.tensorzero.com/docs/gateway/configuration-reference#functionsfunction_namevariantsvariant_name) in this block, with two modifications:

- You must mark a single variant as `active`.
- For `chat_completion` variants, instead of a `system_template` we require `system_instructions` as a text file and take no other templates.

Here we list only the configuration for variants that differs from the configuration for a normal TensorZero function. Please refer the [variant configuration reference](https://www.tensorzero.com/docs/gateway/configuration-reference#functionsfunction_namevariantsvariant_name) for the remaining options.

Copy

```
// tensorzero.toml
[evaluations.email-guardrails.evaluators.check-signature]
# ...
type = "llm_judge"
optimize = "max"

[evaluations.email-guardrails.evaluators.check-signature.variants."claude3.5sonnet"]
type = "chat_completion"
model = "anthropic::claude-sonnet-4-5-20250929"
temperature = 0.1
system_instructions = "./evaluations/email-guardrails/check-signature/system_instructions.txt"
# ... other chat completion configuration ...

[evaluations.email-guardrails.evaluators.check-signature.variants."mix3claude3.5sonnet"]
active = true  # if we run the `email-guardrails` evaluation, this is the variant we'll use for the check-signature evaluator
type = "experimental_mixture_of_n"
candidates = ["claude3.5sonnet", "claude3.5sonnet", "claude3.5sonnet"]
```

###### `active`

- **Type**: boolean
- **Required**: Defaults to `true` if there is a single variant configured. Otherwise, this field is required to be set to `true` for exactly one variant.

Sets which of the variants should be used for evaluation runs.

Copy

```
// tensorzero.toml
[evaluations.email-guardrails.evaluators.check-signature]
# ...

[evaluations.email-guardrails.evaluators.check-signature.variants."mix3claude3.5sonnet"]
active = true # if we run the `email-guardrails` evaluation, this is the variant we'll use for the check-signature evaluator
type = "experimental_mixture_of_n"
```

###### `system_instructions`

- **Type:** string (path)
- **Required**: yes

Defines the path to the system instructions file.
This path is relative to the configuration file.This file should contain a text file with the system instructions for the LLM judge.
These instructions should instruct the judge to output a float or boolean value.
We use JSON mode to enforce that the judge returns a JSON object of the form `{"thinking": "<thinking>", "score": <float or boolean>}` configured to the `output_type` of the evaluator.

evaluations/email-guardrails/check-signature/claude\_35\_sonnet/system\_instructions.txt

Copy

```
Evaluate if the text follows the haiku structure of exactly three lines with a 5-7-5 syllable pattern, totaling 17 syllables. Verify only this specific syllable structure of a haiku without making content assumptions.
```

Copy

```
// tensorzero.toml
[evaluations.email-guardrails.evaluators.check-signature]
# ...
system_instructions = "./evaluations/email-guardrails/check-signature/claude_35_sonnet/system_instructions.txt"
# ...
```

[Tutorial](https://www.tensorzero.com/docs/evaluations/inference-evaluations/tutorial) [CLI Reference](https://www.tensorzero.com/docs/evaluations/inference-evaluations/cli-reference)

Ctrl+I

Assistant

Responses are generated using AI and may contain mistakes.

## Inference Evaluations Tutorial
[Skip to main content](https://www.tensorzero.com/docs/evaluations/inference-evaluations/tutorial#content-area)

[TensorZero Docs home page![light logo](https://mintcdn.com/tensorzero/afFfERJmQ8YiZpCj/logo/light.svg?fit=max&auto=format&n=afFfERJmQ8YiZpCj&q=85&s=9396731a8a92cee7bd293dee05d71dac)![dark logo](https://mintcdn.com/tensorzero/afFfERJmQ8YiZpCj/logo/dark.svg?fit=max&auto=format&n=afFfERJmQ8YiZpCj&q=85&s=8d61c59531dbcdcace7fcdfc524bd4d2)](https://www.tensorzero.com/)

Search...

Ctrl K

Search...

Navigation

Inference Evaluations

Tutorial: Inference Evaluations

[Guides](https://www.tensorzero.com/docs) [Integrations](https://www.tensorzero.com/docs/integrations/model-providers)

On this page

- [Status Quo](https://www.tensorzero.com/docs/evaluations/inference-evaluations/tutorial#status-quo)
- [Datasets](https://www.tensorzero.com/docs/evaluations/inference-evaluations/tutorial#datasets)
- [Evaluations](https://www.tensorzero.com/docs/evaluations/inference-evaluations/tutorial#evaluations)
- [Evaluators](https://www.tensorzero.com/docs/evaluations/inference-evaluations/tutorial#evaluators)
- [exact\_match](https://www.tensorzero.com/docs/evaluations/inference-evaluations/tutorial#exact-match)
- [llm\_judge](https://www.tensorzero.com/docs/evaluations/inference-evaluations/tutorial#llm-judge)
- [Running an Evaluation](https://www.tensorzero.com/docs/evaluations/inference-evaluations/tutorial#running-an-evaluation)
- [CLI](https://www.tensorzero.com/docs/evaluations/inference-evaluations/tutorial#cli)
- [UI](https://www.tensorzero.com/docs/evaluations/inference-evaluations/tutorial#ui)

This guide shows how to define and run inference evaluations for your TensorZero functions.

See our [Quickstart](https://www.tensorzero.com/docs/quickstart) to learn how to set up our LLM gateway, observability, and fine-tuning — in just 5 minutes.

**You can find the code behind this tutorial and instructions on how to run it on [GitHub](https://github.com/tensorzero/tensorzero/tree/main/examples/evaluations/tutorial).**Reach out on [Slack](https://www.tensorzero.com/slack) or [Discord](https://www.tensorzero.com/discord) if you have any questions. We’d be happy to help!

## [​](https://www.tensorzero.com/docs/evaluations/inference-evaluations/tutorial\#status-quo)  Status Quo

Imagine we have a TensorZero function for writing haikus about a given topic, and want to compare the behavior of GPT-4o and GPT-4o Mini on this task.Initially, our configuration for this function might look like:

Copy

```
[functions.write_haiku]
type = "chat"
user_schema = "functions/write_haiku/user_schema.json"

[functions.write_haiku.variants.gpt_4o_mini]
type = "chat_completion"
model = "openai::gpt-4o-mini"
user_template = "functions/write_haiku/user_template.minijinja"

[functions.write_haiku.variants.gpt_4o]
type = "chat_completion"
model = "openai::gpt-4o"
user_template = "functions/write_haiku/user_template.minijinja"
```

User Schema & Template

functions/write\_haiku/user\_schema.json

Copy

```
{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "type": "object",
  "properties": {
    "topic": {
      "type": "string"
    }
  },
  "required": ["topic"],
  "additionalProperties": false
}
```

functions/write\_haiku/user\_template.minijinja

Copy

```
Write a haiku about: {{ topic }}
```

How can we evaluate the behavior of our two variants in a principled way?One option is to build a dataset of “test cases” that we can evaluate them against.

## [​](https://www.tensorzero.com/docs/evaluations/inference-evaluations/tutorial\#datasets)  Datasets

To use TensorZero Evaluations, you first need to build a dataset.A dataset is a collection of datapoints.
Each datapoint has an input and optionally a output.
In the context of evaluations, the output in the dataset should be a reference output, i.e. the output you’d have liked to see.
You don’t necessarily need to provide a reference output: some evaluators (e.g. LLM judges) can score generated outputs without a reference output (otherwise, that datapoint is skipped).Let’s create a dataset:

1. Generate many haikus by running inference on your `write_haiku` function. (On **[GitHub](https://github.com/tensorzero/tensorzero/tree/main/examples/evaluations/tutorial)**, we provide a script `main.py` that generates 100 haikus with `write_haiku`.)
2. Open the UI, navigate to “Datasets”, and select “Build Dataset” (`http://localhost:4000/datasets/builder`).
3. Create a new dataset called `haiku_dataset`.
Select your `write_haiku` function, “None” as the metric, and “Inference” as the dataset output.

See the [Datasets & Datapoints API Reference](https://www.tensorzero.com/docs/gateway/api-reference/datasets-datapoints) to learn how to create and manage datasets programmatically.

## [​](https://www.tensorzero.com/docs/evaluations/inference-evaluations/tutorial\#evaluations)  Evaluations

Evalutions test the behavior of variants for a TensorZero function.Let’s define an evaluation in our configuration file:

Copy

```
[evaluations.haiku_eval]
type = "inference"
function_name = "write_haiku"
```

## [​](https://www.tensorzero.com/docs/evaluations/inference-evaluations/tutorial\#evaluators)  Evaluators

Each evaluation has one or more evaluators: a rule or behavior you’d like to test.Today, TensorZero supports two types of evaluators: `exact_match` and `llm_judge`.

We’re planning to release other types of evaluators soon (e.g. semantic similarity in an embedding space).

### [​](https://www.tensorzero.com/docs/evaluations/inference-evaluations/tutorial\#exact-match)  `exact_match`

The `exact_match` evaluator compares the generated output with the datapoint’s reference output.
If they are identical, it returns true; otherwise, it returns false.

Copy

```
[evaluations.haiku_eval.evaluators.exact_match]
type = "exact_match"
```

### [​](https://www.tensorzero.com/docs/evaluations/inference-evaluations/tutorial\#llm-judge)  `llm_judge`

LLM Judges are special-purpose TensorZero function that can be used to evaluate a TensorZero function.For example, our haikus should generally follow a specific format, but it’s hard to define a heuristic to determine if they’re correct.
Why not ask an LLM?Let’s do that:

Copy

```
[evaluations.haiku_eval.evaluators.valid_haiku]
type = "llm_judge"
output_type = "boolean"  # LLM judge should generate a boolean (or float)
optimize = "max"  # higher is better
cutoff = 0.95  # if the variant scores <95% = bad

[evaluations.haiku_eval.evaluators.valid_haiku.variants.gpt_4o_mini_judge]
type = "chat_completion"
model = "openai::gpt-4o-mini"
system_instructions = "evaluations/haiku_eval/valid_haiku/system_instructions.txt"
json_mode = "strict"
```

System Instructions

evaluations/haiku\_eval/valid\_haiku/system\_instructions.txt

Copy

```
Evaluate if the text follows the haiku structure of exactly three lines with a 5-7-5 syllable pattern, totaling 17 syllables. Verify only this specific syllable structure of a haiku without making content assumptions.
```

Here, we defined an evaluator `valid_haiku` of type `llm_judge`, with a variant that uses GPT-4o Mini.Similar to regular TensorZero functions, we can define multiple variants for an LLM judge.
But unlike regular functions, only one variant can be active at a time during evaluation; you can denote that with the `active` property.

Example: Multiple Variants for an LLM Judge

Copy

```
[evaluations.haiku_eval.evaluators.valid_haiku]
type = "llm_judge"
output_type = "boolean"
optimize = "max"
cutoff = 0.95

[evaluations.haiku_eval.evaluators.valid_haiku.variants.gpt_4o_mini_judge]
type = "chat_completion"
model = "openai::gpt-4o-mini"
system_instructions = "evaluations/haiku_eval/valid_haiku/system_instructions.txt"
json_mode = "strict"
active = true

[evaluations.haiku_eval.evaluators.valid_haiku.variants.gpt_4o_judge]
type = "chat_completion"
model = "openai::gpt-4o"
system_instructions = "evaluations/haiku_eval/valid_haiku/system_instructions.txt"
json_mode = "strict"
```

The LLM judge we showed above generates a boolean, but they can also generate floats.Let’s define another evalutor that counts the number of metaphors in our haiku.

Copy

```
[evaluations.haiku_eval.evaluators.metaphor_count]
type = "llm_judge"
output_type = "float"  # LLM judge should generate a boolean (or float)
optimize = "max"
cutoff = 1  # <1 metaphor per haiku = bad
```

We can also use different variant types for evaluators.
Let’s use a chain-of-thought variant for our metaphor count evaluator, since it’s a bit more complex.

Copy

```
[evaluations.haiku_eval.evaluators.metaphor_count.variants.gpt_4o_mini_judge]
type = "experimental_chain_of_thought"
model = "openai::gpt-4o-mini"
system_instructions = "evaluations/haiku_eval/metaphor_count/system_instructions.txt"
json_mode = "strict"
```

System Instructions

evaluations/haiku\_eval/metaphor\_count/system\_instructions.txt

Copy

```
How many metaphors does the generated haiku have?
```

The LLM judges we’ve defined so far only look at the datapoint’s input and the generated output.
But we can also provide the datapoint’s reference output to the judge:

Copy

```
[evaluations.haiku_eval.evaluators.compare_haikus]
type = "llm_judge"
include = { reference_output = true }  # include the reference output in the LLM judge's context
output_type = "boolean"
optimize = "max"

[evaluations.haiku_eval.evaluators.compare_haikus.variants.gpt_4o_mini_judge]
type = "chat_completion"
model = "openai::gpt-4o-mini"
system_instructions = "evaluations/haiku_eval/compare_haikus/system_instructions.txt"
json_mode = "strict"
```

System Instructions

evaluations/haiku\_eval/compare\_haikus/system\_instructions.txt

Copy

```
Does the generated haiku include the same figures of speech as the reference haiku?
```

## [​](https://www.tensorzero.com/docs/evaluations/inference-evaluations/tutorial\#running-an-evaluation)  Running an Evaluation

Let’s run our evaluations!You can run evaluations using the TensorZero Evaluations CLI tool or the TensorZero UI.

The TensorZero Evaluations CLI tool can be helpful for CI/CD.
It’ll exit with code 0 if all evaluations succeed (average score vs. `cutoff`), or code 1 otherwise.

By default, TensorZero Evaluations uses [Inference Caching](https://www.tensorzero.com/docs/gateway/guides/inference-caching) to improve inference speed and cost.

### [​](https://www.tensorzero.com/docs/evaluations/inference-evaluations/tutorial\#cli)  CLI

To run evaluations in the CLI, you can use the `tensorzero/evaluations` container:

Copy

```
docker compose run --rm evaluations \
    --evaluation-name haiku_eval \
    --dataset-name haiku_dataset \
    --variant-name gpt_4o \
    --concurrency 5
```

Docker Compose

Here’s the relevant section of the `docker-compose.yml` for the evaluations tool.You should provide credentials for any LLM judges.
Alternatively, the evaluations tool can use an external TensorZero Gateway with the `--gateway-url http://gateway:3000` flag.

Copy

```
services:
  # ...

  evaluations:
    profiles: [evaluations]
    image: tensorzero/evaluations
    volumes:
      - ./config:/app/config:ro
    environment:
      - OPENAI_API_KEY=${OPENAI_API_KEY:?Environment variable OPENAI_API_KEY must be set.}
      # ... and any other relevant API credentials ...
      - TENSORZERO_CLICKHOUSE_URL=http://chuser:chpassword@clickhouse:8123/tensorzero
    extra_hosts:
      - "host.docker.internal:host-gateway"
    depends_on:
      clickhouse:
        condition: service_healthy
# ...
```

See [GitHub](https://github.com/tensorzero/tensorzero/tree/main/examples/evaluations/tutorial) for the complete Docker Compose configuration.

Docker Compose does _not_ start this service with `docker compose up` since we have `profiles: [evaluations]`.
You need to call it explicitly with `docker compose run evaluations`, as desired.

### [​](https://www.tensorzero.com/docs/evaluations/inference-evaluations/tutorial\#ui)  UI

To run evaluations in the UI, navigate to “Evaluations” (`http://localhost:4000/evaluations`) and select “New Run”.You can compare multiple evaluation runs in the TensorZero UI (including evaluation runs for the CLI).![TensorZero Evaluation UI](https://mintcdn.com/tensorzero/DzT6ZmuNWZnRZd0Z/evaluations/inference-evaluations/configuration-reference-evaluation-ui.png?fit=max&auto=format&n=DzT6ZmuNWZnRZd0Z&q=85&s=fe35e8196195f15b1587ccc444831e5a)

[Overview](https://www.tensorzero.com/docs/evaluations) [Configuration Reference](https://www.tensorzero.com/docs/evaluations/inference-evaluations/configuration-reference)

Ctrl+I

Assistant

Responses are generated using AI and may contain mistakes.

![TensorZero Evaluation UI](https://mintcdn.com/tensorzero/DzT6ZmuNWZnRZd0Z/evaluations/inference-evaluations/configuration-reference-evaluation-ui.png?w=840&fit=max&auto=format&n=DzT6ZmuNWZnRZd0Z&q=85&s=e58aa42fa9c192d60783e7dbcd8d578b)

## Workflow Evaluations API
[Skip to main content](https://www.tensorzero.com/docs/evaluations/workflow-evaluations/api-reference#content-area)

[TensorZero Docs home page![light logo](https://mintcdn.com/tensorzero/afFfERJmQ8YiZpCj/logo/light.svg?fit=max&auto=format&n=afFfERJmQ8YiZpCj&q=85&s=9396731a8a92cee7bd293dee05d71dac)![dark logo](https://mintcdn.com/tensorzero/afFfERJmQ8YiZpCj/logo/dark.svg?fit=max&auto=format&n=afFfERJmQ8YiZpCj&q=85&s=8d61c59531dbcdcace7fcdfc524bd4d2)](https://www.tensorzero.com/)

Search...

Ctrl K

Search...

Navigation

Workflow Evaluations

API Reference: Workflow Evaluations

[Guides](https://www.tensorzero.com/docs) [Integrations](https://www.tensorzero.com/docs/integrations/model-providers)

On this page

- [Endpoints & Methods](https://www.tensorzero.com/docs/evaluations/workflow-evaluations/api-reference#endpoints-&-methods)
- [Starting a dynamic evaluation run](https://www.tensorzero.com/docs/evaluations/workflow-evaluations/api-reference#starting-a-dynamic-evaluation-run)
- [Starting an episode in a dynamic evaluation run](https://www.tensorzero.com/docs/evaluations/workflow-evaluations/api-reference#starting-an-episode-in-a-dynamic-evaluation-run)
- [Making inference and feedback calls during a dynamic evaluation run](https://www.tensorzero.com/docs/evaluations/workflow-evaluations/api-reference#making-inference-and-feedback-calls-during-a-dynamic-evaluation-run)

Workflow Evaluations focus on evaluating complex workflows that might include multiple TensorZero inference calls, arbitrary application logic, and more.You can initialize and run workflow evaluations using the TensorZero Gateway, either through the TensorZero client or the gateway’s HTTP API.
Unlike inference evaluations, workflow evaluations are not defined in the TensorZero configuration file.See the [Workflow Evaluations Tutorial](https://www.tensorzero.com/docs/evaluations/workflow-evaluations/tutorial) for a step-by-step guide.

## [​](https://www.tensorzero.com/docs/evaluations/workflow-evaluations/api-reference\#endpoints-&-methods)  Endpoints & Methods

### [​](https://www.tensorzero.com/docs/evaluations/workflow-evaluations/api-reference\#starting-a-dynamic-evaluation-run)  Starting a dynamic evaluation run

- **Gateway Endpoint:**`POST /dynamic_evaluation_run`
- **Client Method:**`dynamic_evaluation_run`
- **Parameters:**
  - `variants`: an object (dictionary) mapping function names to variant names
  - `project_name` (string, optional): the name of the project to associate the run with
  - `display_name` (string, optional): the display (human-readable) name of the run
  - `tags` (dictionary, optional): a dictionary of key-value pairs to tag the run’s inferences with
- **Returns:**
  - `run_id` (UUID): the ID of the run

### [​](https://www.tensorzero.com/docs/evaluations/workflow-evaluations/api-reference\#starting-an-episode-in-a-dynamic-evaluation-run)  Starting an episode in a dynamic evaluation run

- **Gateway Endpoint:**`POST /dynamic_evaluation_run/{run_id}/episode`
- **Client Method:**`dynamic_evaluation_run_episode`
- **Parameters:**
  - `run_id` (UUID): the ID of the run generated by the `dynamic_evaluation_run` method
  - `task_name` (string, optional): the name of the task to associate the episode with
  - `tags` (dictionary, optional): a dictionary of key-value pairs to tag the episode’s inferences with
- **Returns:**
  - `episode_id` (UUID): the ID of the episode

### [​](https://www.tensorzero.com/docs/evaluations/workflow-evaluations/api-reference\#making-inference-and-feedback-calls-during-a-dynamic-evaluation-run)  Making inference and feedback calls during a dynamic evaluation run

After initializing a run and an episode, you can make inference and feedback API calls like you normally would.
By providing the special `episode_id` parameter generated by the `dynamic_evaluation_run_episode` method , the TensorZero Gateway will associate the inference and feedback with the evaluation run, handle variant pinning, and more.

[Tutorial](https://www.tensorzero.com/docs/evaluations/workflow-evaluations/tutorial) [Run adaptive A/B tests](https://www.tensorzero.com/docs/experimentation/run-adaptive-ab-tests)

Ctrl+I

Assistant

Responses are generated using AI and may contain mistakes.

## Dynamic Workflow Evaluations
[Skip to main content](https://www.tensorzero.com/docs/evaluations/workflow-evaluations/tutorial#content-area)

[TensorZero Docs home page![light logo](https://mintcdn.com/tensorzero/afFfERJmQ8YiZpCj/logo/light.svg?fit=max&auto=format&n=afFfERJmQ8YiZpCj&q=85&s=9396731a8a92cee7bd293dee05d71dac)![dark logo](https://mintcdn.com/tensorzero/afFfERJmQ8YiZpCj/logo/dark.svg?fit=max&auto=format&n=afFfERJmQ8YiZpCj&q=85&s=8d61c59531dbcdcace7fcdfc524bd4d2)](https://www.tensorzero.com/)

Search...

Ctrl K

Search...

Navigation

Workflow Evaluations

Tutorial: Workflow Evaluations

[Guides](https://www.tensorzero.com/docs) [Integrations](https://www.tensorzero.com/docs/integrations/model-providers)

On this page

- [Starting a dynamic evaluation run](https://www.tensorzero.com/docs/evaluations/workflow-evaluations/tutorial#starting-a-dynamic-evaluation-run)
- [Starting an episode in a dynamic evaluation run](https://www.tensorzero.com/docs/evaluations/workflow-evaluations/tutorial#starting-an-episode-in-a-dynamic-evaluation-run)
- [Making inference and feedback calls during a dynamic evaluation run](https://www.tensorzero.com/docs/evaluations/workflow-evaluations/tutorial#making-inference-and-feedback-calls-during-a-dynamic-evaluation-run)
- [Visualizing evaluation results in the TensorZero UI](https://www.tensorzero.com/docs/evaluations/workflow-evaluations/tutorial#visualizing-evaluation-results-in-the-tensorzero-ui)

Dynamic evaluations enable you to evaluate complex workflows that combine multiple inference calls with arbitrary application logic.
Here, we’ll walk through a stylized RAG workflow to illustrate the process of setting up and running a dynamic evaluation, but the same process can be applied to any complex workflow.Imagine we have the following LLM-powered workflow in response to a natural-language question from a user:

1. Inference: Call the `generate_database_query` TensorZero function to generate a database query from the user’s question.
2. Custom Logic: Run the database query against a database and retrieve the results (`my_blackbox_search_function`).
3. Inference: Call the `generate_final_answer` TensorZero function to generate an answer from the retrieved results.
4. Custom Logic: Score the answer using a custom scoring function (`my_blackbox_scoring_function`)
5. Feedback: Send feedback using the `task_success` metric.

Evaluating `generate_database_query` and `generate_final_answer` in a vacuum (i.e. using inference evaluations) can also be helpful, but ultimately we want to evaluate the entire workflow end-to-end.
This is where workflow evaluations come in.Complex LLM applications might need to make multiple LLM calls and execute arbitrary code before giving an overall result.
In agentic applications, the workflow might even be defined dynamically at runtime based on the user’s input, the results of the LLM calls, or other factors.
Dynamic evaluations in TensorZero provide complete flexibility and enable you to evaluate the entire workflow jointly.
You can think of them like integration tests for your LLM applications.

For a more complex, runnable example, see the [Workflow Evaluations for Agentic RAG Example on GitHub](https://github.com/tensorzero/tensorzero/tree/main/examples/dynamic_evaluations/simple-agentic-rag).

## [​](https://www.tensorzero.com/docs/evaluations/workflow-evaluations/tutorial\#starting-a-dynamic-evaluation-run)  Starting a dynamic evaluation run

Evaluating the workflow above involves tackling and evaluating a collection of tasks (e.g. user queries).
Each individual task corresponds to an _episode_, and the collection of these episodes is a _dynamic evaluation run_.

- Python

- Python (Async)

- HTTP


First, let’s initialize the TensorZero client (just like you would for typical inference requests):

Copy

```
from tensorzero import TensorZeroGateway

# Initialize the client with `build_http` or `build_embedded`
with TensorZeroGateway.build_http(
    gateway_url="http://localhost:3000",
) as t0:
    # ...
```

Now you can start a dynamic evaluation run.During a dynamic evaluation run, you specify which variants you want to pin during the run (i.e. the set of variants you want to evaluate).
This allows you to see the effects of different combinations of variants on the end-to-end system’s performance.

You don’t have to specify a variant for every function you use; if you don’t specify a variant, the TensorZero Gateway will sample a variant for you as it normally would.

You can optionally also specify a `project_name` and `display_name` for the run.
If you specify a `project_name`, you’ll be able to compare this run against other runs for that project using the TensorZero UI.
The `display_name` is a human-readable identifier for the run that you can use to identify the run in the TensorZero UI.

Copy

```
run_info = t0.dynamic_evaluation_run(
    # Assume we have these variants defined in our `tensorzero.toml` configuration file
    variants={
        "generate_database_query": "o4_mini_prompt_baseline",
        "generate_final_answer": "gpt_4o_updated_prompt",
    },
    project_name="simple_rag_project",
    display_name="generate_database_query::o4_mini_prompt_baseline;generate_final_answer::gpt_4o_updated_prompt",
)
```

The TensorZero client automatically tags your dynamic evaluation runs with information about your Git repository if available (e.g. branch name, commit hash).
This metadata is displayed in the TensorZero UI so that you have a record of the code that was used to run the dynamic evaluation.
We recommend that you commit your changes before running a dynamic evaluation so that the Git state is accurately captured.

## [​](https://www.tensorzero.com/docs/evaluations/workflow-evaluations/tutorial\#starting-an-episode-in-a-dynamic-evaluation-run)  Starting an episode in a dynamic evaluation run

For each task we want to include in our dynamic evaluation run, we need to start an episode.
For example, in our agentic RAG project, each episode will correspond to a user query from our dataset; each user query requires multiple inference calls and application logic to run.

- Python

- Python (Async)

- HTTP


To initialize an episode, you need to provide the `run_id` of the dynamic evaluation run you want to include the episode in.
You can optionally also specify a `task_name` for the episode.
If you specify a `task_name`, you’ll be able to compare this episode against episodes for that task from other runs using the TensorZero UI.
We encourage you to use the `task_name` to provide a meaningful identifier for the task that the episode is tackling.

Copy

```
episode_info = t0.dynamic_evaluation_run_episode(
    run_id=run_info.run_id,
    task_name="user_query_123",
)
```

Now we can use `episode_info.episode_id` to make inference and feedback calls.

## [​](https://www.tensorzero.com/docs/evaluations/workflow-evaluations/tutorial\#making-inference-and-feedback-calls-during-a-dynamic-evaluation-run)  Making inference and feedback calls during a dynamic evaluation run

See our [Quickstart](https://www.tensorzero.com/docs/quickstart) to learn how to set up our LLM gateway, observability, and fine-tuning — in just 5 minutes.

You can also use the OpenAI SDK for inference calls.
See the [Quickstart](https://www.tensorzero.com/docs/quickstart) for more details.(Similarly, you can also use workflow evaluations with any framework or agent that is OpenAI-compatible by passing along the episode ID and function name in the request to TensorZero.)

- Python

- Python (Async)

- HTTP


Copy

```
generate_database_query_response = t0.inference(
    function_name="generate_database_query",
    episode_id=episode_info.episode_id,
    input={ ... },
)

search_result = my_blackbox_search_function(generate_database_query_response)

generate_final_answer_response = t0.inference(
    function_name="generate_final_answer",
    episode_id=episode_info.episode_id,
    input={ ... },
)

task_success_score = my_blackbox_scoring_function(generate_final_answer_response)

t0.feedback(
    metric_name="task_success",
    episode_id=episode_info.episode_id,
    value=task_success_score,
)
```

## [​](https://www.tensorzero.com/docs/evaluations/workflow-evaluations/tutorial\#visualizing-evaluation-results-in-the-tensorzero-ui)  Visualizing evaluation results in the TensorZero UI

Once you finish running all the relevant episodes for your dynamic evaluation run, you can visualize the results in the TensorZero UI.In the UI, you can compare metrics across evaluation runs, inspect individual episodes and inferences, and more.![Dynamic Evaluation Run Results in the TensorZero UI](https://mintcdn.com/tensorzero/DzT6ZmuNWZnRZd0Z/evaluations/workflow-evaluations/tutorial-ui.png?fit=max&auto=format&n=DzT6ZmuNWZnRZd0Z&q=85&s=bbe192eca0b566a8d9504026f99e061c)

[CLI Reference](https://www.tensorzero.com/docs/evaluations/inference-evaluations/cli-reference) [API Reference](https://www.tensorzero.com/docs/evaluations/workflow-evaluations/api-reference)

Ctrl+I

Assistant

Responses are generated using AI and may contain mistakes.

![Dynamic Evaluation Run Results in the TensorZero UI](https://mintcdn.com/tensorzero/DzT6ZmuNWZnRZd0Z/evaluations/workflow-evaluations/tutorial-ui.png?w=840&fit=max&auto=format&n=DzT6ZmuNWZnRZd0Z&q=85&s=9f9ba4c3f563e7caae6e3a645f40aa99)

## Adaptive A/B Testing Guide
[Skip to main content](https://www.tensorzero.com/docs/experimentation/run-adaptive-ab-tests#content-area)

[TensorZero Docs home page![light logo](https://mintcdn.com/tensorzero/afFfERJmQ8YiZpCj/logo/light.svg?fit=max&auto=format&n=afFfERJmQ8YiZpCj&q=85&s=9396731a8a92cee7bd293dee05d71dac)![dark logo](https://mintcdn.com/tensorzero/afFfERJmQ8YiZpCj/logo/dark.svg?fit=max&auto=format&n=afFfERJmQ8YiZpCj&q=85&s=8d61c59531dbcdcace7fcdfc524bd4d2)](https://www.tensorzero.com/)

Search...

Ctrl K

Search...

Navigation

Experimentation

Run adaptive A/B tests

[Guides](https://www.tensorzero.com/docs) [Integrations](https://www.tensorzero.com/docs/integrations/model-providers)

On this page

- [Configure](https://www.tensorzero.com/docs/experimentation/run-adaptive-ab-tests#configure)
- [Advanced](https://www.tensorzero.com/docs/experimentation/run-adaptive-ab-tests#advanced)
- [Configure fallback-only variants](https://www.tensorzero.com/docs/experimentation/run-adaptive-ab-tests#configure-fallback-only-variants)
- [Customize the experimentation algorithm](https://www.tensorzero.com/docs/experimentation/run-adaptive-ab-tests#customize-the-experimentation-algorithm)

You can set up adaptive A/B tests with the TensorZero Gateway to automatically distribute inference requests to the best performing variants (prompts, models, etc.) of your system.
TensorZero supports any number of variants in an adaptive A/B test.In simple terms, you define:

- A [TensorZero function](https://www.tensorzero.com/docs/gateway/configure-functions-and-variants) (a task or agent)
- A set of candidate [variants](https://www.tensorzero.com/docs/gateway/configure-functions-and-variants) (prompts, models, etc.) to experiment with
- A [metric](https://www.tensorzero.com/docs/gateway/guides/metrics-feedback) to optimize for

And TensorZero takes care of the rest.
TensorZero’s experimentation algorithm is designed to efficiently find the best variant of the system with a specified level of confidence.
You can add more variants over time and TensorZero will adjust the experiment accordingly while maintaining its statistical soundness.You don’t need to choose the sample size or experiment duration up front.
TensorZero will automatically detect when there are enough samples to identify the best variant.
Once it has done so, it will use that variant for all subsequent inferences.

Learn more about adaptive A/B testing for LLMs in our blog post [Bandits in your LLM Gateway: Improve LLM Applications Faster with Adaptive Experimentation (A/B Testing)](https://www.tensorzero.com/blog/bandits-in-your-llm-gateway/).

## [​](https://www.tensorzero.com/docs/experimentation/run-adaptive-ab-tests\#configure)  Configure

Let’s set up an adaptive A/B test with TensorZero.

You can find a [complete runnable example](https://github.com/tensorzero/tensorzero/tree/main/examples/docs/guides/experimentation/run-adaptive-ab-tests) of this guide on GitHub.

1

Configure your function

Let’s configure a function (“task”) with two variants (`gpt-5-mini` with two different prompts), a metric to optimize for, and the experimentation configuration.

tensorzero.toml

Copy

```
# Define a function for the task we're tackling
[functions.extract_entities]
type = "json"
output_schema = "output_schema.json"

# Define variants to experiment with (here, we have two different prompts)
[functions.extract_entities.variants.gpt-5-mini-good-prompt]
type = "chat_completion"
model = "openai::gpt-5-mini"
templates.system.path = "good_system_template.minijinja"
json_mode = "strict"

[functions.extract_entities.variants.gpt-5-mini-bad-prompt]
type = "chat_completion"
model = "openai::gpt-5-mini"
templates.system.path = "bad_system_template.minijinja"
json_mode = "strict"

# Define the experiment configuration
[functions.extract_entities.experimentation]
type = "track_and_stop" # the experimentation algorithm
candidate_variants = ["gpt-5-mini-good-prompt", "gpt-5-mini-bad-prompt"]
metric = "exact_match"
update_period_s = 60  # low for the sake of the demo (recommended: 300)

# Define the metric we're optimizing for
[metrics.exact_match]
type = "boolean"
level = "inference"
optimize = "max"
```

2

Deploy TensorZero

You must set up Postgres to use TensorZero’s automated experimentation features.

- [Deploy the TensorZero Gateway](https://www.tensorzero.com/docs/deployment/tensorzero-gateway)
- [Deploy the TensorZero UI](https://www.tensorzero.com/docs/deployment/tensorzero-ui)
- [Deploy ClickHouse](https://www.tensorzero.com/docs/deployment/clickhouse)
- [Deploy Postgres](https://www.tensorzero.com/docs/deployment/postgres)

3

Make inference requests

Make an inference request just like you normally would and keep track of the inference ID or episode ID.
You can use the TensorZero Inference API or the OpenAI-compatible Inference API.

Copy

```
response = t0.inference(
    function_name="extract_entities",
    input={
        "messages": [\
            {\
                "role": "user",\
                "content": datapoint.input,\
            }\
        ]
    },
)
```

4

Send feedback for your metric

Send feedback for your metric and assign it to the inference ID or episode ID.

Copy

```
t0.feedback(
    metric_name="exact_match",
    value=True,
    inference_id=response.inference_id,
)
```

5

Track your experiment

That’s it.
TensorZero will automatically adjust the distribution of inference requests between the two candidate variants based on their performance.You can track the experiment in the TensorZero UI.
Visit the function’s detail page to see the variant weights and the estimated performance.If you run the code example, TensorZero starts by splitting traffic between the two variants but quickly starts shifting more and more traffic towards the `gpt-5-mini-good-prompt` variant.
After a few hundred inferences, TensorZero becomes confident enough to declare it the winner and starts serving all the traffic to it.

![Experimentation in the TensorZero UI](https://mintcdn.com/tensorzero/DzT6ZmuNWZnRZd0Z/experimentation/run-adaptive-ab-tests.gif?s=c47bd498077aa1a5e0edf2d691048aed)

You can add more variants at any time and TensorZero will adjust the experiment accordingly in a principled way.

## [​](https://www.tensorzero.com/docs/experimentation/run-adaptive-ab-tests\#advanced)  Advanced

### [​](https://www.tensorzero.com/docs/experimentation/run-adaptive-ab-tests\#configure-fallback-only-variants)  Configure fallback-only variants

In addition to `candidate_variants`, you can also specify `fallback_variants` in your configuration.If a variant fails for any reason, TensorZero first resamples from `candidate_variants`.
Once they are exhausted, it attempts to use the first variant in `fallback_variants`; if that fails, it goes to the second fallback variant, etc.Note that episodes that contain inferences that use different variants for the same function (e.g. as a result of a fallback) are not used by the adaptive A/B testing algorithm.See the [Configuration Reference](https://www.tensorzero.com/docs/gateway/configuration-reference) for more details.

### [​](https://www.tensorzero.com/docs/experimentation/run-adaptive-ab-tests\#customize-the-experimentation-algorithm)  Customize the experimentation algorithm

The `track_and_stop` algorithm has multiple parameters that can be customized.
For example, you can trade off the speed of the experiment with the statistical confidence of the results.
The default parameters are sensible for most use cases, but advanced users might want to customize them.
See the [Configuration Reference](https://www.tensorzero.com/docs/gateway/configuration-reference) for more details.Two important parameters are `epsilon` and `delta`, which control a fundamental trade-off in experimentation: higher sensitivity and lower error rates require longer experiments.
For a discussion on `epsilon` and `delta`, see our blog post [Bandits in your LLM Gateway: Improve LLM Applications Faster with Adaptive Experimentation (A/B Testing)](https://www.tensorzero.com/blog/bandits-in-your-llm-gateway/).

[API Reference](https://www.tensorzero.com/docs/evaluations/workflow-evaluations/api-reference) [Run static A/B tests](https://www.tensorzero.com/docs/experimentation/run-static-ab-tests)

Ctrl+I

Assistant

Responses are generated using AI and may contain mistakes.

## Static A/B Testing
[Skip to main content](https://www.tensorzero.com/docs/experimentation/run-static-ab-tests#content-area)

[TensorZero Docs home page![light logo](https://mintcdn.com/tensorzero/afFfERJmQ8YiZpCj/logo/light.svg?fit=max&auto=format&n=afFfERJmQ8YiZpCj&q=85&s=9396731a8a92cee7bd293dee05d71dac)![dark logo](https://mintcdn.com/tensorzero/afFfERJmQ8YiZpCj/logo/dark.svg?fit=max&auto=format&n=afFfERJmQ8YiZpCj&q=85&s=8d61c59531dbcdcace7fcdfc524bd4d2)](https://www.tensorzero.com/)

Search...

Ctrl K

Search...

Navigation

Experimentation

Run static A/B tests

[Guides](https://www.tensorzero.com/docs) [Integrations](https://www.tensorzero.com/docs/integrations/model-providers)

On this page

- [Configure multiple variants](https://www.tensorzero.com/docs/experimentation/run-static-ab-tests#configure-multiple-variants)
- [Configure sampling weights for variants](https://www.tensorzero.com/docs/experimentation/run-static-ab-tests#configure-sampling-weights-for-variants)
- [Configure fallback-only variants](https://www.tensorzero.com/docs/experimentation/run-static-ab-tests#configure-fallback-only-variants)

You can configure the TensorZero Gateway to distribute inference requests between different variants (prompts, models, etc.) of a function (a “task” or “agent”).
Variants enable you to experiment with different models, prompts, parameters, inference strategies, and more.

We recommend [running adaptive A/B tests](https://www.tensorzero.com/docs/experimentation/run-adaptive-ab-tests) if you have a metric you can optimize for.

## [​](https://www.tensorzero.com/docs/experimentation/run-static-ab-tests\#configure-multiple-variants)  Configure multiple variants

If you specify multiple variants for a function, by default the gateway will sample between them with equal probability (uniform sampling).For example, if you call the `draft_email` function below, the gateway will sample between the two variants at each inference with equal probability.

Copy

```
[functions.draft_email]
type = "chat"

[functions.draft_email.variants.gpt_5_mini]
type = "chat_completion"
model = "openai::gpt-5-mini"

[functions.draft_email.variants.claude_haiku_4_5]
type = "chat_completion"
model = "anthropic::claude-haiku-4-5"
```

During an episode, multiple inference requests to the same function will receive the same variant (unless fallbacks are necessary).
This consistent variant assignment acts as a randomized controlled experiment, providing the statistical foundation needed to make causal inferences about which configurations perform best.

## [​](https://www.tensorzero.com/docs/experimentation/run-static-ab-tests\#configure-sampling-weights-for-variants)  Configure sampling weights for variants

You can configure weights for variants to control the probability of each variant being sampled.
This is particularly useful for canary tests where you want to gradually roll out a new variant to a small percentage of users.

Copy

```
[functions.draft_email]
type = "chat"

[functions.draft_email.variants.gpt_5_mini]
type = "chat_completion"
model = "openai::gpt-5-mini"

[functions.draft_email.variants.claude_haiku_4_5]
type = "chat_completion"
model = "anthropic::claude-haiku-4-5"

[functions.draft_email.experimentation]
type = "static_weights"
candidate_variants = {"gpt_5_mini" = 0.9, "claude_haiku_4_5" = 0.1}
```

In this example, 90% of episodes will be sampled from the `gpt_5_mini` variant and 10% will be sampled from the `claude_haiku_4_5` variant.

If the weights don’t add up to 1, TensorZero will automatically normalize them and sample the variants accordingly.
For example, if a variant has weight 5 and another has weight 1, the first variant will be sampled 5/6 of the time (≈ 83.3%) and the second variant will be sampled 1/6 of the time (≈ 16.7%).

## [​](https://www.tensorzero.com/docs/experimentation/run-static-ab-tests\#configure-fallback-only-variants)  Configure fallback-only variants

You can configure variants that are only used as fallbacks with `fallback_variants`.

Copy

```
[functions.draft_email]
type = "chat"

[functions.draft_email.variants.gpt_5_mini]
type = "chat_completion"
model = "openai::gpt-5-mini"

[functions.draft_email.variants.claude_haiku_4_5]
type = "chat_completion"
model = "anthropic::claude-haiku-4-5"

[functions.draft_email.variants.grok_4]
type = "chat_completion"
model = "xai::grok-4-0709"

[functions.draft_email.experimentation]
type = "static_weights"
candidate_variants = {"gpt_5_mini" = 0.9, "claude_haiku_4_5" = 0.1}
fallback_variants = ["grok_4"]
```

The gateway will first sample among the `candidate_variants`.
If all candidates fail, the gateway attempts each variant in `fallback_variants` in order.
See [Retries & Fallbacks](https://www.tensorzero.com/docs/gateway/guides/retries-fallbacks) for more information.

[Run adaptive A/B tests](https://www.tensorzero.com/docs/experimentation/run-adaptive-ab-tests) [Deploy the TensorZero Gateway](https://www.tensorzero.com/docs/deployment/tensorzero-gateway)

Ctrl+I

Assistant

Responses are generated using AI and may contain mistakes.

## TensorZero FAQ
[Skip to main content](https://www.tensorzero.com/docs/faq#content-area)

[TensorZero Docs home page![light logo](https://mintcdn.com/tensorzero/afFfERJmQ8YiZpCj/logo/light.svg?fit=max&auto=format&n=afFfERJmQ8YiZpCj&q=85&s=9396731a8a92cee7bd293dee05d71dac)![dark logo](https://mintcdn.com/tensorzero/afFfERJmQ8YiZpCj/logo/dark.svg?fit=max&auto=format&n=afFfERJmQ8YiZpCj&q=85&s=8d61c59531dbcdcace7fcdfc524bd4d2)](https://www.tensorzero.com/)

Search...

Ctrl K

Search...

Navigation

Introduction

Frequently Asked Questions

[Guides](https://www.tensorzero.com/docs) [Integrations](https://www.tensorzero.com/docs/integrations/model-providers)

On this page

- [Technical](https://www.tensorzero.com/docs/faq#technical)
- [Project](https://www.tensorzero.com/docs/faq#project)

**Next steps?**
The [Quickstart](https://www.tensorzero.com/docs/quickstart) shows it’s easy to set up an LLM application with TensorZero.**Questions?**
Ask us on [Slack](https://www.tensorzero.com/slack) or [Discord](https://www.tensorzero.com/discord).**Using TensorZero at work?**
Email us at [hello@tensorzero.com](mailto:hello@tensorzero.com) to set up a Slack or Teams channel with your team (free).

## [​](https://www.tensorzero.com/docs/faq\#technical)  Technical

Why is the TensorZero Gateway a proxy instead of a library?

TensorZero’s proxy pattern makes it agnostic to the application’s tech stack, isolated from the business logic, more composable with other tools, and easy to deploy and manage.Many engineers are (correctly) wary of marginal latency from such a proxy, so we built the gateway from the ground up with performance in mind.
In [Benchmarks](https://www.tensorzero.com/docs/gateway/benchmarks), it achieves sub-millisecond P99 latency overhead under extreme load.
This makes the gateway fast and lightweight enough to be unnoticeable even in the most demanding LLM applications, especially if deployed as a sidecar container.

How is the TensorZero Gateway so fast?

![TensorZero Crab](https://mintcdn.com/tensorzero/afFfERJmQ8YiZpCj/tensorzero-crab.png?fit=max&auto=format&n=afFfERJmQ8YiZpCj&q=85&s=ae0178de3fe7dde7f0d3137dffef44a0)The TensorZero Gateway was built from the ground up with performance in mind.
It was written in Rust 🦀 and optimizes many common bottlenecks by efficiently managing connections to model providers, pre-compiling schemas and templates, logging data asynchronously, and more.It achieves <1ms P99 latency overhead under extreme load.
In [Benchmarks](https://www.tensorzero.com/docs/gateway/benchmarks), LiteLLM @ 100 QPS adds 25-100x+ more latency than the TensorZero Gateway @ 10,000 QPS.

Why did you choose ClickHouse as TensorZero's analytics database?

ClickHouse is open source, [extremely fast](https://www.vldb.org/pvldb/vol17/p3731-schulze.pdf), and versatile.
It supports diverse storage backends, query patterns, and data types, including vector search (which will be important for upcoming TensorZero features).
From the start, we designed TensorZero to be easy to deploy but able to grow to massive scale.
ClickHouse is the best tool for the job.

## [​](https://www.tensorzero.com/docs/faq\#project)  Project

Who is behind TensorZero?

We’re a small technical team based in NYC. [Work with us →](https://www.tensorzero.com/jobs/)

#### [​](https://www.tensorzero.com/docs/faq\#founders)  Founders

[Viraj Mehta](https://virajm.com/) (CTO) recently completed his PhD from CMU, with an emphasis on reinforcement learning for LLMs and nuclear fusion, and previously worked in machine learning at KKR and a fintech startup; he holds a BS in math and an MS in computer science from Stanford.[Gabriel Bianconi](https://www.gabrielbianconi.com/) (CEO) was the chief product officer at Ondo Finance ($20B+ valuation in 2024) and previously spent years consulting on machine learning for companies ranging from early-stage tech startups to some of the largest financial firms; he holds BS and MS degrees in computer science from Stanford.

How is TensorZero licensed?

![TensorZero Freedom](https://mintcdn.com/tensorzero/afFfERJmQ8YiZpCj/tensorzero-freedom.png?fit=max&auto=format&n=afFfERJmQ8YiZpCj&q=85&s=3179408a8309d195cd912cfa2bcf16f9)TensorZero is open source under the permissive [Apache 2.0 License](https://github.com/tensorzero/tensorzero/blob/main/LICENSE).

How does TensorZero make money?

[We don’t.](https://www.youtube.com/watch?v=BzAdXyPYKQo) We’re lucky to have investors who are aligned with our long-term vision, so we’re able to focus on building and snooze this question for a while.We’re inspired by companies like Databricks and ClickHouse.
One day, we’ll launch a managed service that further streamlines LLM engineering, especially in enterprise settings, but open source will always be at the core of our business.

[Vision & Roadmap](https://www.tensorzero.com/docs/vision-and-roadmap) [DSPy](https://www.tensorzero.com/docs/comparison/dspy)

Ctrl+I

Assistant

Responses are generated using AI and may contain mistakes.

![TensorZero Crab](https://mintcdn.com/tensorzero/afFfERJmQ8YiZpCj/tensorzero-crab.png?w=840&fit=max&auto=format&n=afFfERJmQ8YiZpCj&q=85&s=5c963ff9e162cc5a5b62e2df3ca8abfa)

![TensorZero Freedom](https://mintcdn.com/tensorzero/afFfERJmQ8YiZpCj/tensorzero-freedom.png?w=840&fit=max&auto=format&n=afFfERJmQ8YiZpCj&q=85&s=20232b5bc8a4ebe6bd5e67af35f9d429)

## Batch Inference API
[Skip to main content](https://www.tensorzero.com/docs/gateway/api-reference/batch-inference#content-area)

[TensorZero Docs home page![light logo](https://mintcdn.com/tensorzero/afFfERJmQ8YiZpCj/logo/light.svg?fit=max&auto=format&n=afFfERJmQ8YiZpCj&q=85&s=9396731a8a92cee7bd293dee05d71dac)![dark logo](https://mintcdn.com/tensorzero/afFfERJmQ8YiZpCj/logo/dark.svg?fit=max&auto=format&n=afFfERJmQ8YiZpCj&q=85&s=8d61c59531dbcdcace7fcdfc524bd4d2)](https://www.tensorzero.com/)

Search...

Ctrl K

Search...

Navigation

API Reference

API Reference: Batch Inference

[Guides](https://www.tensorzero.com/docs) [Integrations](https://www.tensorzero.com/docs/integrations/model-providers)

On this page

- [POST /batch\_inference](https://www.tensorzero.com/docs/gateway/api-reference/batch-inference#post-/batch-inference)
- [Request](https://www.tensorzero.com/docs/gateway/api-reference/batch-inference#request)
- [additional\_tools](https://www.tensorzero.com/docs/gateway/api-reference/batch-inference#additional-tools)
- [allowed\_tools](https://www.tensorzero.com/docs/gateway/api-reference/batch-inference#allowed-tools)
- [credentials](https://www.tensorzero.com/docs/gateway/api-reference/batch-inference#credentials)
- [episode\_ids](https://www.tensorzero.com/docs/gateway/api-reference/batch-inference#episode-ids)
- [function\_name](https://www.tensorzero.com/docs/gateway/api-reference/batch-inference#function-name)
- [inputs](https://www.tensorzero.com/docs/gateway/api-reference/batch-inference#inputs)
- [output\_schemas](https://www.tensorzero.com/docs/gateway/api-reference/batch-inference#output-schemas)
- [parallel\_tool\_calls](https://www.tensorzero.com/docs/gateway/api-reference/batch-inference#parallel-tool-calls)
- [params](https://www.tensorzero.com/docs/gateway/api-reference/batch-inference#params)
- [tags](https://www.tensorzero.com/docs/gateway/api-reference/batch-inference#tags)
- [tool\_choice](https://www.tensorzero.com/docs/gateway/api-reference/batch-inference#tool-choice)
- [variant\_name](https://www.tensorzero.com/docs/gateway/api-reference/batch-inference#variant-name)
- [Response](https://www.tensorzero.com/docs/gateway/api-reference/batch-inference#response)
- [batch\_id](https://www.tensorzero.com/docs/gateway/api-reference/batch-inference#batch-id)
- [inference\_ids](https://www.tensorzero.com/docs/gateway/api-reference/batch-inference#inference-ids)
- [episode\_ids](https://www.tensorzero.com/docs/gateway/api-reference/batch-inference#episode-ids-2)
- [Example](https://www.tensorzero.com/docs/gateway/api-reference/batch-inference#example)
- [GET /batch\_inference/:batch\_id](https://www.tensorzero.com/docs/gateway/api-reference/batch-inference#get-/batch-inference/:batch-id)
- [Pending](https://www.tensorzero.com/docs/gateway/api-reference/batch-inference#pending)
- [Failed](https://www.tensorzero.com/docs/gateway/api-reference/batch-inference#failed)
- [Completed](https://www.tensorzero.com/docs/gateway/api-reference/batch-inference#completed)
- [status](https://www.tensorzero.com/docs/gateway/api-reference/batch-inference#status)
- [batch\_id](https://www.tensorzero.com/docs/gateway/api-reference/batch-inference#batch-id-2)
- [inferences](https://www.tensorzero.com/docs/gateway/api-reference/batch-inference#inferences)
- [Example](https://www.tensorzero.com/docs/gateway/api-reference/batch-inference#example-2)
- [GET /batch\_inference/:batch\_id/inference/:inference\_id](https://www.tensorzero.com/docs/gateway/api-reference/batch-inference#get-/batch-inference/:batch-id/inference/:inference-id)
- [Pending](https://www.tensorzero.com/docs/gateway/api-reference/batch-inference#pending-2)
- [Failed](https://www.tensorzero.com/docs/gateway/api-reference/batch-inference#failed-2)
- [Completed](https://www.tensorzero.com/docs/gateway/api-reference/batch-inference#completed-2)
- [status](https://www.tensorzero.com/docs/gateway/api-reference/batch-inference#status-2)
- [batch\_id](https://www.tensorzero.com/docs/gateway/api-reference/batch-inference#batch-id-3)
- [inferences](https://www.tensorzero.com/docs/gateway/api-reference/batch-inference#inferences-2)
- [Example](https://www.tensorzero.com/docs/gateway/api-reference/batch-inference#example-3)

The `/batch_inference` endpoints allow users to take advantage of batched inference offered by LLM providers.
These inferences are often substantially cheaper than the synchronous APIs.
The handling and eventual data model for inferences made through this endpoint are equivalent to those made through the main `/inference` endpoint with a few exceptions:

- The batch samples a single variant from the function being called.
- There are no fallbacks or retries for bached functions.
- Only variants of type `chat_completion` are supported.
- Caching is not supported.
- The `dryrun` setting is not supported.
- Streaming is not supported.

Under the hood, the gateway validates all of the requests, samples a single variant from the function being called, handles templating when applicable, and routes the inference to the appropriate model provider.
In the batch endpoint there are no fallbacks as the requests are processed asynchronously.The typical workflow is to first use the `POST /batch_inference` endpoint to submit a batch of requests.
Later, you can poll the `GET /batch_inference/{batch_id}` or `GET /batch_inference/:batch_id/inference/:inference_id` endpoint to check the status of the batch and retrieve results.
Each poll will return either a pending or failed status or the results of the batch.
Even after a batch has completed and been processed, you can continue to poll the endpoint as a way of retrieving the results.
The first time a batch has completed and been processed, the results are stored in the ChatInference, JsonInference, and ModelInference tables as with the `/inference` endpoint.
The gateway will rehydrate the results into the expected result when polled repeatedly after finishing

See the [Batch Inference Guide](https://www.tensorzero.com/docs/gateway/guides/batch-inference) for a simple example of using the batch inference endpoints.

## [​](https://www.tensorzero.com/docs/gateway/api-reference/batch-inference\#post-/batch-inference)  `POST /batch_inference`

### [​](https://www.tensorzero.com/docs/gateway/api-reference/batch-inference\#request)  Request

#### [​](https://www.tensorzero.com/docs/gateway/api-reference/batch-inference\#additional-tools)  `additional_tools`

- **Type:** list of lists of tools (see below)
- **Required:** no (default: no additional tools)

A list of lists of tools defined at inference time that the model is allowed to call.
This field allows for dynamic tool use, i.e. defining tools at runtime.
Each element in the outer list corresponds to a single inference in the batch.
Each inner list contains the tools that should be available to the corresponding inference.You should prefer to define tools in the configuration file if possible.
Only use this field if dynamic tool use is necessary for your use case.Each tool is an object with the following fields: `description`, `name`, `parameters`, and `strict`.The fields are identical to those in the configuration file, except that the `parameters` field should contain the JSON schema itself rather than a path to it.
See [Configuration Reference](https://www.tensorzero.com/docs/gateway/configuration-reference#toolstool_name) for more details.

#### [​](https://www.tensorzero.com/docs/gateway/api-reference/batch-inference\#allowed-tools)  `allowed_tools`

- **Type:** list of lists of strings
- **Required:** no

A list of lists of tool names that the model is allowed to call.
The tools must be defined in the configuration file or provided dynamically via `additional_tools`.
Each element in the outer list corresponds to a single inference in the batch.
Each inner list contains the names of the tools that are allowed for the corresponding inference.Some providers (notably OpenAI) natively support restricting allowed tools.
For these providers, we send all tools (both configured and dynamic) to the provider, and separately specify which ones are allowed to be called.
For providers that do not natively support this feature, we filter the tool list ourselves and only send the allowed tools to the provider.

### [​](https://www.tensorzero.com/docs/gateway/api-reference/batch-inference\#credentials)  `credentials`

- **Type:** object (a map from dynamic credential names to API keys)
- **Required:** no (default: no credentials)

Each model provider in your TensorZero configuration can be configured to accept credentials at inference time by using the `dynamic` location (e.g. `dynamic::my_dynamic_api_key_name`).
See the [configuration reference](https://www.tensorzero.com/docs/gateway/configuration-reference#modelsmodel_nameprovidersprovider_name) for more details.
The gateway expects the credentials to be provided in the `credentials` field of the request body as specified below.
The gateway will return a 400 error if the credentials are not provided and the model provider has been configured with dynamic credentials.

Example

Copy

```
[models.my_model_name.providers.my_provider_name]
# ...
# Note: the name of the credential field (e.g. `api_key_location`) depends on the provider type
api_key_location = "dynamic::my_dynamic_api_key_name"
# ...
```

Copy

```
{
  // ...
  "credentials": {
    // ...
    "my_dynamic_api_key_name": "sk-..."
    // ...
  }
  // ...
}
```

#### [​](https://www.tensorzero.com/docs/gateway/api-reference/batch-inference\#episode-ids)  `episode_ids`

- **Type:** list of UUIDs
- **Required:** no

The IDs of existing episodes to associate the inferences with.
Each element in the list corresponds to a single inference in the batch.
You can provide `null` for episode IDs for elements that should start a fresh episode.Only use episode IDs that were returned by the TensorZero gateway.

#### [​](https://www.tensorzero.com/docs/gateway/api-reference/batch-inference\#function-name)  `function_name`

- **Type:** string
- **Required:** yes

The name of the function to call. This function will be the same for all inferences in the batch.The function must be defined in the configuration file.

#### [​](https://www.tensorzero.com/docs/gateway/api-reference/batch-inference\#inputs)  `inputs`

- **Type:** list of `input` objects (see below)
- **Required:** yes

The input to the function.Each element in the list corresponds to a single inference in the batch.

##### `input[].messages`

- **Type:** list of messages (see below)
- **Required:** no (default: `[]`)

A list of messages to provide to the model.Each message is an object with the following fields:

- `role`: The role of the message (`assistant` or `user`).
- `content`: The content of the message (see below).

The `content` field can be have one of the following types:

- string: the text for a text message (only allowed if there is no schema for that role)
- list of content blocks: the content blocks for the message (see below)

A content block is an object with the field `type` and additional fields depending on the type.If the content block has type `text`, it must have either of the following additional fields:

- `text`: The text for the content block.
- `arguments`: A JSON object containing the function arguments for TensorZero functions with templates and schemas (see [Create a prompt template](https://www.tensorzero.com/docs/gateway/create-a-prompt-template) for details).

If the content block has type `tool_call`, it must have the following additional fields:

- `arguments`: The arguments for the tool call.
- `id`: The ID for the content block.
- `name`: The name of the tool for the content block.

If the content block has type `tool_result`, it must have the following additional fields:

- `id`: The ID for the content block.
- `name`: The name of the tool for the content block.
- `result`: The result of the tool call.

If the content block has type `file`, it must have exactly one of the following additional fields:

- File URLs
  - `file_type`: must be `url`
  - `url`
  - `mime_type` (optional): override the MIME type of the file
- Base64-encoded Files
  - `file_type`: must be `base64`
  - `data`: `base64`-encoded data for an embedded file
  - `mime_type`: the MIME type (e.g. `image/png`, `image/jpeg`, `application/pdf`)

See the [Multimodal Inference](https://www.tensorzero.com/docs/gateway/guides/multimodal-inference) guide for more details on how to use images in inference.If the content block has type `raw_text`, it must have the following additional fields:

- `value`: The text for the content block.
This content block will ignore any relevant templates and schemas for this function.

If the content block has type `thought`, it must have the following additional fields:

- `text`: The text for the content block.

If the content block has type `unknown`, it must have the following additional fields:

- `data`: The original content block from the provider, without any validation or transformation by TensorZero.
- `model_provider_name` (optional): A string specifying when this content block should be included in the model provider input.
If set, the content block will only be provided to this specific model provider.
If not set, the content block is passed to all model providers.

For example, the following hypothetical unknown content block will send the `daydreaming` content block to inference requests targeting the `your_model_provider_name` model provider.

Copy

```
{
  "type": "unknown",
  "data": {
    "type": "daydreaming",
    "dream": "..."
  },
  "model_provider_name": "tensorzero::model_name::your_model_name::provider_name::your_model_provider_name"
}
```

This is the most complex field in the entire API. See this example for more details.

Example

Copy

```
{
  // ...
  "input": {
    "messages": [\
      // If you don't have a user (or assistant) schema...\
      {\
        "role": "user", // (or "assistant")\
        "content": "What is the weather in Tokyo?"\
      },\
      // If you have a user (or assistant) schema...\
      {\
        "role": "user", // (or "assistant")\
        "content": [\
          {\
            "type": "text",\
            "arguments": {\
              "location": "Tokyo"\
              // ...\
            }\
          }\
        ]\
      },\
      // If the model previously called a tool...\
      {\
        "role": "assistant",\
        "content": [\
          {\
            "type": "tool_call",\
            "id": "0",\
            "name": "get_temperature",\
            "arguments": "{\"location\": \"Tokyo\"}"\
          }\
        ]\
      },\
      // ...and you're providing the result of that tool call...\
      {\
        "role": "user",\
        "content": [\
          {\
            "type": "tool_result",\
            "id": "0",\
            "name": "get_temperature",\
            "result": "70"\
          }\
        ]\
      },\
      // You can also specify a text message using a content block...\
      {\
        "role": "user",\
        "content": [\
          {\
            "type": "text",\
            "text": "What about NYC?" // (or object if there is a schema)\
          }\
        ]\
      },\
      // You can also provide multiple content blocks in a single message...\
      {\
        "role": "assistant",\
        "content": [\
          {\
            "type": "text",\
            "text": "Sure, I can help you with that." // (or object if there is a schema)\
          },\
          {\
            "type": "tool_call",\
            "id": "0",\
            "name": "get_temperature",\
            "arguments": "{\"location\": \"New York\"}"\
          }\
        ]\
      }\
      // ...\
    ]
    // ...
  }
  // ...
}
```

##### `input[].system`

- **Type:** string or object
- **Required:** no

The input for the system message.If the function does not have a system schema, this field should be a string.If the function has a system schema, this field should be an object that matches the schema.

#### [​](https://www.tensorzero.com/docs/gateway/api-reference/batch-inference\#output-schemas)  `output_schemas`

- **Type:** list of optional objects (valid JSON Schema)
- **Required:** no

A list of JSON schemas that will be used to validate the output of the function for each inference in the batch.
Each element in the list corresponds to a single inference in the batch.
These can be null for elements that need to use the `output_schema` defined in the function configuration.
This schema is used for validating the output of the function, and sent to providers which support structured outputs.

#### [​](https://www.tensorzero.com/docs/gateway/api-reference/batch-inference\#parallel-tool-calls)  `parallel_tool_calls`

- **Type:** list of optional booleans
- **Required:** no

A list of booleans that indicate whether each inference in the batch should be allowed to request multiple tool calls in a single conversation turn.
Each element in the list corresponds to a single inference in the batch.
You can provide `null` for elements that should use the configuration value for the function being called.
If you don’t provide this field entirely, we default to the configuration value for the function being called.Most model providers do not support parallel tool calls. In those cases, the gateway ignores this field.
At the moment, only Fireworks AI and OpenAI support parallel tool calls.

#### [​](https://www.tensorzero.com/docs/gateway/api-reference/batch-inference\#params)  `params`

- **Type:** object (see below)
- **Required:** no (default: `{}`)

Override inference-time parameters for a particular variant type.
This fields allows for dynamic inference parameters, i.e. defining parameters at runtime.This field’s format is `{ variant_type: { param: [value1, ...], ... }, ... }`.
You should prefer to set these parameters in the configuration file if possible.
Only use this field if you need to set these parameters dynamically at runtime.
Each parameter if specified should be a list of values that may be null that is the same length as the batch size.Note that the parameters will apply to every variant of the specified type.Currently, we support the following:

- `chat_completion`
  - `frequency_penalty`
  - `json_mode`
  - `max_tokens`
  - `presence_penalty`
  - `reasoning_effort`
  - `seed`
  - `service_tier`
  - `stop_sequences`
  - `temperature`
  - `thinking_budget_tokens`
  - `top_p`
  - `verbosity`

See [Configuration Reference](https://www.tensorzero.com/docs/gateway/configuration-reference#functionsfunction_namevariantsvariant_name) for more details on the parameters, and Examples below for usage.

Example

For example, if you wanted to dynamically override the `temperature` parameter for a `chat_completion` variant for the first inference in a batch of 3, you’d include the following in the request body:

Copy

```
{
  // ...
  "params": {
    "chat_completion": {
      "temperature": [0.7, null, null]
    }
  }
  // ...
}
```

#### [​](https://www.tensorzero.com/docs/gateway/api-reference/batch-inference\#tags)  `tags`

- **Type:** list of optional JSON objects with string keys and values
- **Required:** no

User-provided tags to associate with the inference.Each element in the list corresponds to a single inference in the batch.For example, `[{"user_id": "123"}, null]` or `[{"author": "Alice"}, {"author": "Bob"}]`.

#### [​](https://www.tensorzero.com/docs/gateway/api-reference/batch-inference\#tool-choice)  `tool_choice`

- **Type:** list of optional strings
- **Required:** no

If set, overrides the tool choice strategy for the equest.Each element in the list corresponds to a single inference in the batch.The supported tool choice strategies are:

- `none`: The function should not use any tools.
- `auto`: The model decides whether or not to use a tool. If it decides to use a tool, it also decides which tools to use.
- `required`: The model should use a tool. If multiple tools are available, the model decides which tool to use.
- `{ specific = "tool_name" }`: The model should use a specific tool. The tool must be defined in the `tools` section of the configuration file or provided in `additional_tools`.

#### [​](https://www.tensorzero.com/docs/gateway/api-reference/batch-inference\#variant-name)  `variant_name`

- **Type:** string
- **Required:** no

If set, pins the batch inference request to a particular variant (not recommended).You should generally not set this field, and instead let the TensorZero gateway assign a variant.
This field is primarily used for testing or debugging purposes.

### [​](https://www.tensorzero.com/docs/gateway/api-reference/batch-inference\#response)  Response

For a POST request to `/batch_inference`, the response is a JSON object containing metadata that allows you to refer to the batch and poll it later on.
The response is an object with the following fields:

#### [​](https://www.tensorzero.com/docs/gateway/api-reference/batch-inference\#batch-id)  `batch_id`

- **Type:** UUID

The ID of the batch.

#### [​](https://www.tensorzero.com/docs/gateway/api-reference/batch-inference\#inference-ids)  `inference_ids`

- **Type:** list of UUIDs

The IDs of the inferences in the batch.

#### [​](https://www.tensorzero.com/docs/gateway/api-reference/batch-inference\#episode-ids-2)  `episode_ids`

- **Type:** list of UUIDs

The IDs of the episodes associated with the inferences in the batch.

### [​](https://www.tensorzero.com/docs/gateway/api-reference/batch-inference\#example)  Example

Imagine you have a simple TensorZero function that generates haikus using GPT-4o Mini.

Copy

```
[functions.generate_haiku]
type = "chat"

[functions.generate_haiku.variants.gpt_4o_mini]
type = "chat_completion"
model = "openai::gpt-4o-mini-2024-07-18"
```

You can submit a batch inference job to generate multiple haikus with a single request.
Each entry in `inputs` is equal to the `input` field in a regular inference request.

Copy

```
curl -X POST http://localhost:3000/batch_inference \
  -H "Content-Type: application/json" \
  -d '{
    "function_name": "generate_haiku",
    "variant_name": "gpt_4o_mini",
    "inputs": [\
      {\
        "messages": [\
          {\
            "role": "user",\
            "content": "Write a haiku about artificial intelligence."\
          }\
        ]\
      },\
      {\
        "messages": [\
          {\
            "role": "user",\
            "content": "Write a haiku about general aviation."\
          }\
        ]\
      },\
      {\
        "messages": [\
          {\
            "role": "user",\
            "content": "Write a haiku about anime."\
          }\
        ]\
      }\
    ]
  }'
```

The response contains a `batch_id` as well as `inference_ids` and `episode_ids` for each inference in the batch.

Copy

```
{
  "batch_id": "019470f0-db4c-7811-9e14-6fe6593a2652",
  "inference_ids": [\
    "019470f0-d34a-77a3-9e59-bcc66db2b82f",\
    "019470f0-d34a-77a3-9e59-bcdd2f8e06aa",\
    "019470f0-d34a-77a3-9e59-bcecfb7172a0"\
  ],
  "episode_ids": [\
    "019470f0-d34a-77a3-9e59-bc933973d087",\
    "019470f0-d34a-77a3-9e59-bca6e9b748b2",\
    "019470f0-d34a-77a3-9e59-bcb20177bf3a"\
  ]
}
```

## [​](https://www.tensorzero.com/docs/gateway/api-reference/batch-inference\#get-/batch-inference/:batch-id)  `GET /batch_inference/:batch_id`

Both this and the following GET endpoint can be used to poll the status of a batch.
If you use this endpoint and poll with only the batch ID the entire batch will be returned if possible.
The response format depends on the function type as well as the batch status when polled.

### [​](https://www.tensorzero.com/docs/gateway/api-reference/batch-inference\#pending)  Pending

`{"status": "pending"}`

### [​](https://www.tensorzero.com/docs/gateway/api-reference/batch-inference\#failed)  Failed

`{"status": "failed"}`

### [​](https://www.tensorzero.com/docs/gateway/api-reference/batch-inference\#completed)  Completed

#### [​](https://www.tensorzero.com/docs/gateway/api-reference/batch-inference\#status)  `status`

- **Type:** literal string `"completed"`

#### [​](https://www.tensorzero.com/docs/gateway/api-reference/batch-inference\#batch-id-2)  `batch_id`

- **Type:** UUID

#### [​](https://www.tensorzero.com/docs/gateway/api-reference/batch-inference\#inferences)  `inferences`

- **Type:** list of objects that exactly match the response body in the inference endpoint documented [here](https://www.tensorzero.com/docs/gateway/api-reference/inference#response).

### [​](https://www.tensorzero.com/docs/gateway/api-reference/batch-inference\#example-2)  Example

Extending the example from above: you can use the `batch_id` to poll the status of this job:

Copy

```
curl -X GET http://localhost:3000/batch_inference/019470f0-db4c-7811-9e14-6fe6593a2652
```

While the job is pending, the response will only contain the `status` field.

Copy

```
{
  "status": "pending"
}
```

Once the job is completed, the response will contain the `status` field and the `inferences` field.
Each inference object is the same as the response from a regular inference request.

Copy

```
{
  "status": "completed",
  "batch_id": "019470f0-db4c-7811-9e14-6fe6593a2652",
  "inferences": [\
    {\
      "inference_id": "019470f0-d34a-77a3-9e59-bcc66db2b82f",\
      "episode_id": "019470f0-d34a-77a3-9e59-bc933973d087",\
      "variant_name": "gpt_4o_mini",\
      "content": [\
        {\
          "type": "text",\
          "text": "Whispers of circuits,  \nLearning paths through endless code,  \nDreams in binary."\
        }\
      ],\
      "usage": {\
        "input_tokens": 15,\
        "output_tokens": 19\
      }\
    },\
    {\
      "inference_id": "019470f0-d34a-77a3-9e59-bcdd2f8e06aa",\
      "episode_id": "019470f0-d34a-77a3-9e59-bca6e9b748b2",\
      "variant_name": "gpt_4o_mini",\
      "content": [\
        {\
          "type": "text",\
          "text": "Wings of freedom soar,  \nClouds embrace the lonely flight,  \nSky whispers adventure."\
        }\
      ],\
      "usage": {\
        "input_tokens": 15,\
        "output_tokens": 20\
      }\
    },\
    {\
      "inference_id": "019470f0-d34a-77a3-9e59-bcecfb7172a0",\
      "episode_id": "019470f0-d34a-77a3-9e59-bcb20177bf3a",\
      "variant_name": "gpt_4o_mini",\
      "content": [\
        {\
          "type": "text",\
          "text": "Vivid worlds unfold,  \nHeroes rise with dreams in hand,  \nInk and dreams collide."\
        }\
      ],\
      "usage": {\
        "input_tokens": 14,\
        "output_tokens": 20\
      }\
    }\
  ]
}
```

## [​](https://www.tensorzero.com/docs/gateway/api-reference/batch-inference\#get-/batch-inference/:batch-id/inference/:inference-id)  `GET /batch_inference/:batch_id/inference/:inference_id`

This endpoint can be used to poll the status of a single inference in a batch.
Since the polling involves pulling data on all the inferences in the batch, we also store the status of all those inference in ClickHouse.
The response format depends on the function type as well as the batch status when polled.

### [​](https://www.tensorzero.com/docs/gateway/api-reference/batch-inference\#pending-2)  Pending

`{"status": "pending"}`

### [​](https://www.tensorzero.com/docs/gateway/api-reference/batch-inference\#failed-2)  Failed

`{"status": "failed"}`

### [​](https://www.tensorzero.com/docs/gateway/api-reference/batch-inference\#completed-2)  Completed

#### [​](https://www.tensorzero.com/docs/gateway/api-reference/batch-inference\#status-2)  `status`

- **Type:** literal string `"completed"`

#### [​](https://www.tensorzero.com/docs/gateway/api-reference/batch-inference\#batch-id-3)  `batch_id`

- **Type:** UUID

#### [​](https://www.tensorzero.com/docs/gateway/api-reference/batch-inference\#inferences-2)  `inferences`

- **Type:** list containing a single object that exactly matches the response body in the inference endpoint documented [here](https://www.tensorzero.com/docs/gateway/api-reference/inference#response).

### [​](https://www.tensorzero.com/docs/gateway/api-reference/batch-inference\#example-3)  Example

Similar to above, we can also poll a particular inference:

Copy

```
curl -X GET http://localhost:3000/batch_inference/019470f0-db4c-7811-9e14-6fe6593a2652/inference/019470f0-d34a-77a3-9e59-bcc66db2b82f
```

While the job is pending, the response will only contain the `status` field.

Copy

```
{
  "status": "pending"
}
```

Once the job is completed, the response will contain the `status` field and the `inferences` field.
Unlike above, this request will return a list containing only the requested inference.

Copy

```
{
  "status": "completed",
  "batch_id": "019470f0-db4c-7811-9e14-6fe6593a2652",
  "inferences": [\
    {\
      "inference_id": "019470f0-d34a-77a3-9e59-bcc66db2b82f",\
      "episode_id": "019470f0-d34a-77a3-9e59-bc933973d087",\
      "variant_name": "gpt_4o_mini",\
      "content": [\
        {\
          "type": "text",\
          "text": "Whispers of circuits,  \nLearning paths through endless code,  \nDreams in binary."\
        }\
      ],\
      "usage": {\
        "input_tokens": 15,\
        "output_tokens": 19\
      }\
    }\
  ]
}
```

[Feedback](https://www.tensorzero.com/docs/gateway/api-reference/feedback) [Datasets & Datapoints](https://www.tensorzero.com/docs/gateway/api-reference/datasets-datapoints)

Ctrl+I

Assistant

Responses are generated using AI and may contain mistakes.

## Datasets and Datapoints API
[Skip to main content](https://www.tensorzero.com/docs/gateway/api-reference/datasets-datapoints#content-area)

[TensorZero Docs home page![light logo](https://mintcdn.com/tensorzero/afFfERJmQ8YiZpCj/logo/light.svg?fit=max&auto=format&n=afFfERJmQ8YiZpCj&q=85&s=9396731a8a92cee7bd293dee05d71dac)![dark logo](https://mintcdn.com/tensorzero/afFfERJmQ8YiZpCj/logo/dark.svg?fit=max&auto=format&n=afFfERJmQ8YiZpCj&q=85&s=8d61c59531dbcdcace7fcdfc524bd4d2)](https://www.tensorzero.com/)

Search...

Ctrl K

Search...

Navigation

API Reference

API Reference: Datasets & Datapoints

[Guides](https://www.tensorzero.com/docs) [Integrations](https://www.tensorzero.com/docs/integrations/model-providers)

On this page

- [Endpoints & Methods](https://www.tensorzero.com/docs/gateway/api-reference/datasets-datapoints#endpoints-&-methods)
- [List datapoints in a dataset](https://www.tensorzero.com/docs/gateway/api-reference/datasets-datapoints#list-datapoints-in-a-dataset)
- [Get a datapoint](https://www.tensorzero.com/docs/gateway/api-reference/datasets-datapoints#get-a-datapoint)
- [Add datapoints to a dataset (or create a dataset)](https://www.tensorzero.com/docs/gateway/api-reference/datasets-datapoints#add-datapoints-to-a-dataset-or-create-a-dataset)
- [Update datapoints in a dataset](https://www.tensorzero.com/docs/gateway/api-reference/datasets-datapoints#update-datapoints-in-a-dataset)
- [Update datapoint metadata](https://www.tensorzero.com/docs/gateway/api-reference/datasets-datapoints#update-datapoint-metadata)
- [Delete a datapoint](https://www.tensorzero.com/docs/gateway/api-reference/datasets-datapoints#delete-a-datapoint)

In TensorZero, datasets are collections of data that can be used for workflows like evaluations and optimization recipes.
You can create and manage datasets using the TensorZero UI or programmatically using the TensorZero Gateway.A dataset is a named collection of datapoints.
Each datapoint belongs to a function, with fields that depend on the function’s type.
Broadly speaking, each datapoint largely mirrors the structure of an inference, with an input, an optional output, and other associated metadata (e.g. tags).

You can find a complete runnable example of how to use the datasets and datapoints API in our [GitHub repository](https://github.com/tensorzero/tensorzero/tree/main/examples/guides/datasets-datapoints).

## [​](https://www.tensorzero.com/docs/gateway/api-reference/datasets-datapoints\#endpoints-&-methods)  Endpoints & Methods

### [​](https://www.tensorzero.com/docs/gateway/api-reference/datasets-datapoints\#list-datapoints-in-a-dataset)  List datapoints in a dataset

This endpoint returns a list of datapoints in the dataset.
Each datapoint is an object that includes all the relevant fields (e.g. input, output, tags).

- **Gateway Endpoint:**`GET /datasets/{dataset_name}/datapoints`
- **Client Method:**`list_datapoints`
- **Parameters:**
  - `dataset_name` (string)
  - `function` (string, optional)
  - `limit` (int, optional, defaults to 100)
  - `offset` (int, optional, defaults to 0)

If `function` is set, this method only returns datapoints in the dataset for the specified function.

### [​](https://www.tensorzero.com/docs/gateway/api-reference/datasets-datapoints\#get-a-datapoint)  Get a datapoint

This endpoint returns the datapoint with the given ID, including all the relevant fields (e.g. input, output, tags).

- **Gateway Endpoint:**`GET /datasets/{dataset_name}/datapoints/{datapoint_id}`
- **Client Method:**`get_datapoint`
- **Parameters:**
  - `dataset_name` (string)
  - `datapoint_id` (string)

### [​](https://www.tensorzero.com/docs/gateway/api-reference/datasets-datapoints\#add-datapoints-to-a-dataset-or-create-a-dataset)  Add datapoints to a dataset (or create a dataset)

This endpoint adds a list of datapoints to a dataset.
If the dataset does not exist, it will be created with the given name.

- **Gateway Endpoint:**`POST /datasets/{dataset_name}/datapoints`
- **Client Method:**`create_datapoints`
- **Parameters:**
  - `dataset_name` (string)
  - `datapoints` (list of objects, see below)

For `chat` functions, each datapoint object must have the following fields:

- `function_name` (string)
- `input` (object, identical to an inference’s `input`)
- `output` (a list of objects, optional, each object must be a content block like in an inference’s output)
- `allowed_tools` (list of strings, optional, identical to an inference’s `allowed_tools`)
- `tool_choice` (string, optional, identical to an inference’s `tool_choice`)
- `parallel_tool_calls` (boolean, optional, defaults to `false`)
- `tags` (map of string to string, optional)
- `name` (string, optional)

For `json` functions, each datapoint object must have the following fields:

- `function_name` (string)
- `input` (object, identical to an inference’s `input`)
- `output` (object, optional, an object that matches the `output_schema` of the function)
- `output_schema` (object, optional, a dynamic JSON schema that overrides the output schema of the function)
- `tags` (map of string to string, optional)
- `name` (string, optional)

### [​](https://www.tensorzero.com/docs/gateway/api-reference/datasets-datapoints\#update-datapoints-in-a-dataset)  Update datapoints in a dataset

This endpoint updates one or more datapoints in a dataset by creating new versions.
The original datapoint is marked as stale (i.e. a soft deletion), and a new datapoint is created with the updated values and a new ID.
The response returns the newly created IDs.

- **Gateway Endpoint:**`PATCH /v1/datasets/{dataset_name}/datapoints`
- **Client Method:**`update_datapoints`

Each object must have the fields `id` (string, UUIDv7) and `type` (`"chat"` or `"json"`).The following fields are optional.
If provided, they will update the corresponding fields in the datapoint.
If omitted, the fields will remain unchanged.
If set to `null`, the fields will be cleared (as long as they are nullable).For `chat` functions, you can update the following fields:

- `input` (object) - replaces the datapoint’s input
- `output` (list of content blocks) - replaces the datapoint’s output
- `tool_params` (object or null) - replaces the tool configuration (can be set to `null` to clear)
- `tags` (map of string to string) - replaces all tags
- `metadata` (object) - updates metadata fields:

  - `name` (string or null) - replaces the name (can be set to `null` to clear)

For `json` functions, you can update the following fields:

- `input` (object) - replaces the datapoint’s input
- `output` (object or null) - replaces the output (validated against the output schema; can be set to `null` to clear)
- `output_schema` (object) - replaces the output schema
- `tags` (map of string to string) - replaces all tags
- `metadata` (object) - updates metadata fields:

  - `name` (string or null) - replaces the name (can be set to `null` to clear)

If you’re only updating datapoint metadata (e.g. `name`), the `update_datapoint_metadata` method below is an alternative that does not affect the datapoint ID.

The endpoint returns an object with `ids`, a list of IDs (strings, UUIDv7) of the updated datapoints.

### [​](https://www.tensorzero.com/docs/gateway/api-reference/datasets-datapoints\#update-datapoint-metadata)  Update datapoint metadata

This endpoint updates metadata fields for one or more datapoints in a dataset.
Unlike updating the full datapoint, this operation updates the datapoint in-place without creating a new version.

- **Gateway Endpoint:**`PATCH /v1/datasets/{dataset_name}/datapoints/metadata`
- **Client Method:**`update_datapoints_metadata`
- **Parameters:**
  - `dataset_name` (string)
  - `datapoints` (list of objects, see below)

The `datapoints` field must contain a list of objects.Each object must have the field `id` (string, UUIDv7).The following field is optional:

- `metadata` (object) - updates metadata fields:

  - `name` (string or null) - replaces the name (can be set to `null` to clear)

If the `metadata` field is omitted or `null`, no changes will be made to the datapoint.The endpoint returns an object with `ids`, a list of IDs (strings, UUIDv7) of the updated datapoints.
These IDs are the same as the input IDs since the datapoints are updated in-place.

### [​](https://www.tensorzero.com/docs/gateway/api-reference/datasets-datapoints\#delete-a-datapoint)  Delete a datapoint

This endpoint performs a **soft deletion**: the datapoint is marked as stale and will be disregarded by the system in the future (e.g. when listing datapoints or running evaluations), but the data remains in the database.

- **Gateway Endpoint:**`DELETE /datasets/{dataset_name}/datapoints/{datapoint_id}`
- **Client Method:**`delete_datapoint`
- **Parameters:**
  - `dataset_name` (string)
  - `datapoint_id` (string)

[Batch Inference](https://www.tensorzero.com/docs/gateway/api-reference/batch-inference) [Overview](https://www.tensorzero.com/docs/recipes)

Ctrl+I

Assistant

Responses are generated using AI and may contain mistakes.

## Feedback API Reference
[Skip to main content](https://www.tensorzero.com/docs/gateway/api-reference/feedback#content-area)

[TensorZero Docs home page![light logo](https://mintcdn.com/tensorzero/afFfERJmQ8YiZpCj/logo/light.svg?fit=max&auto=format&n=afFfERJmQ8YiZpCj&q=85&s=9396731a8a92cee7bd293dee05d71dac)![dark logo](https://mintcdn.com/tensorzero/afFfERJmQ8YiZpCj/logo/dark.svg?fit=max&auto=format&n=afFfERJmQ8YiZpCj&q=85&s=8d61c59531dbcdcace7fcdfc524bd4d2)](https://www.tensorzero.com/)

Search...

Ctrl K

Search...

Navigation

API Reference

API Reference: Feedback

[Guides](https://www.tensorzero.com/docs) [Integrations](https://www.tensorzero.com/docs/integrations/model-providers)

On this page

- [POST /feedback](https://www.tensorzero.com/docs/gateway/api-reference/feedback#post-/feedback)
- [Request](https://www.tensorzero.com/docs/gateway/api-reference/feedback#request)
- [dryrun](https://www.tensorzero.com/docs/gateway/api-reference/feedback#dryrun)
- [episode\_id](https://www.tensorzero.com/docs/gateway/api-reference/feedback#episode-id)
- [inference\_id](https://www.tensorzero.com/docs/gateway/api-reference/feedback#inference-id)
- [metric\_name](https://www.tensorzero.com/docs/gateway/api-reference/feedback#metric-name)
- [tags](https://www.tensorzero.com/docs/gateway/api-reference/feedback#tags)
- [value](https://www.tensorzero.com/docs/gateway/api-reference/feedback#value)
- [Response](https://www.tensorzero.com/docs/gateway/api-reference/feedback#response)
- [feedback\_id](https://www.tensorzero.com/docs/gateway/api-reference/feedback#feedback-id)
- [Examples](https://www.tensorzero.com/docs/gateway/api-reference/feedback#examples)
- [Inference-Level Boolean Metric](https://www.tensorzero.com/docs/gateway/api-reference/feedback#inference-level-boolean-metric)
- [Episode-Level Float Metric](https://www.tensorzero.com/docs/gateway/api-reference/feedback#episode-level-float-metric)

## [​](https://www.tensorzero.com/docs/gateway/api-reference/feedback\#post-/feedback)  `POST /feedback`

The `/feedback` endpoint assigns feedback to a particular inference or episode.Each feedback is associated with a metric that is defined in the configuration file.

### [​](https://www.tensorzero.com/docs/gateway/api-reference/feedback\#request)  Request

#### [​](https://www.tensorzero.com/docs/gateway/api-reference/feedback\#dryrun)  `dryrun`

- **Type:** boolean
- **Required:** no

If `true`, the feedback request will be executed but won’t be stored to the database (i.e. no-op).This field is primarily for debugging and testing, and you should ignore it in production.

#### [​](https://www.tensorzero.com/docs/gateway/api-reference/feedback\#episode-id)  `episode_id`

- **Type:** UUID
- **Required:** when the metric level is `episode`

The episode ID to provide feedback for.You should use this field when the metric level is `episode`.Only use episode IDs that were returned by the TensorZero gateway.

#### [​](https://www.tensorzero.com/docs/gateway/api-reference/feedback\#inference-id)  `inference_id`

- **Type:** UUID
- **Required:** when the metric level is `inference`

The inference ID to provide feedback for.You should use this field when the metric level is `inference`.Only use inference IDs that were returned by the TensorZero gateway.

#### [​](https://www.tensorzero.com/docs/gateway/api-reference/feedback\#metric-name)  `metric_name`

- **Type:** string
- **Required:** yes

The name of the metric to provide feedback.For example, if your metric is defined as `[metrics.draft_accepted]` in your configuration file, then you would set `metric_name: "draft_accepted"`.The metric names `comment` and `demonstration` are reserved for special types of feedback.
A `comment` is free-form text (string) that can be assigned to either an inference or an episode.
The `demonstration` metric accepts values that would be a valid output.
See [Metrics & Feedback](https://www.tensorzero.com/docs/gateway/guides/metrics-feedback) for more details.

#### [​](https://www.tensorzero.com/docs/gateway/api-reference/feedback\#tags)  `tags`

- **Type:** flat JSON object with string keys and values
- **Required:** no

User-provided tags to associate with the feedback.For example, `{"user_id": "123"}` or `{"author": "Alice"}`.

#### [​](https://www.tensorzero.com/docs/gateway/api-reference/feedback\#value)  `value`

- **Type:** varies
- **Required:** yes

The value of the feedback.The type of the value depends on the metric type (e.g. boolean for a metric with `type = "boolean"`).

### [​](https://www.tensorzero.com/docs/gateway/api-reference/feedback\#response)  Response

#### [​](https://www.tensorzero.com/docs/gateway/api-reference/feedback\#feedback-id)  `feedback_id`

- **Type:** UUID

The ID assigned to the feedback.

### [​](https://www.tensorzero.com/docs/gateway/api-reference/feedback\#examples)  Examples

#### [​](https://www.tensorzero.com/docs/gateway/api-reference/feedback\#inference-level-boolean-metric)  Inference-Level Boolean Metric

Inference-Level Boolean Metric

##### Configuration

Copy

```
// tensorzero.toml
# ...
[metrics.draft_accepted]
type = "boolean"
level = "inference"
# ...
```

##### Request

- Python

- HTTP


POST /feedback

Copy

```
from tensorzero import AsyncTensorZeroGateway

async with await AsyncTensorZeroGateway.build_http(gateway_url="http://localhost:3000") as client:
    result = await client.feedback(
        inference_id="00000000-0000-0000-0000-000000000000",
        metric_name="draft_accepted",
        value=True,
    )
```

##### Response

POST /feedback

Copy

```
{ "feedback_id": "11111111-1111-1111-1111-111111111111" }
```

#### [​](https://www.tensorzero.com/docs/gateway/api-reference/feedback\#episode-level-float-metric)  Episode-Level Float Metric

Episode-Level Float Metric

##### Configuration

Copy

```
// tensorzero.toml
# ...
[metrics.user_rating]
type = "float"
level = "episode"
# ...
```

##### Request

- Python

- HTTP


POST /feedback

Copy

```
from tensorzero import AsyncTensorZeroGateway

async with await AsyncTensorZeroGateway.build_http(gateway_url="http://localhost:3000") as client:
    result = await client.feedback(
        episode_id="00000000-0000-0000-0000-000000000000",
        metric_name="user_rating",
        value=10,
    )
```

##### Response

POST /feedback

Copy

```
{ "feedback_id": "11111111-1111-1111-1111-111111111111" }
```

[Inference (OpenAI)](https://www.tensorzero.com/docs/gateway/api-reference/inference-openai-compatible) [Batch Inference](https://www.tensorzero.com/docs/gateway/api-reference/batch-inference)

Ctrl+I

Assistant

Responses are generated using AI and may contain mistakes.

## TensorZero Inference API
[Skip to main content](https://www.tensorzero.com/docs/gateway/api-reference/inference#content-area)

[TensorZero Docs home page![light logo](https://mintcdn.com/tensorzero/afFfERJmQ8YiZpCj/logo/light.svg?fit=max&auto=format&n=afFfERJmQ8YiZpCj&q=85&s=9396731a8a92cee7bd293dee05d71dac)![dark logo](https://mintcdn.com/tensorzero/afFfERJmQ8YiZpCj/logo/dark.svg?fit=max&auto=format&n=afFfERJmQ8YiZpCj&q=85&s=8d61c59531dbcdcace7fcdfc524bd4d2)](https://www.tensorzero.com/)

Search...

Ctrl K

Search...

Navigation

API Reference

API Reference: Inference

[Guides](https://www.tensorzero.com/docs) [Integrations](https://www.tensorzero.com/docs/integrations/model-providers)

On this page

- [POST /inference](https://www.tensorzero.com/docs/gateway/api-reference/inference#post-/inference)
- [Request](https://www.tensorzero.com/docs/gateway/api-reference/inference#request)
- [additional\_tools](https://www.tensorzero.com/docs/gateway/api-reference/inference#additional-tools)
- [allowed\_tools](https://www.tensorzero.com/docs/gateway/api-reference/inference#allowed-tools)
- [cache\_options](https://www.tensorzero.com/docs/gateway/api-reference/inference#cache-options)
- [credentials](https://www.tensorzero.com/docs/gateway/api-reference/inference#credentials)
- [dryrun](https://www.tensorzero.com/docs/gateway/api-reference/inference#dryrun)
- [episode\_id](https://www.tensorzero.com/docs/gateway/api-reference/inference#episode-id)
- [extra\_body](https://www.tensorzero.com/docs/gateway/api-reference/inference#extra-body)
- [extra\_headers](https://www.tensorzero.com/docs/gateway/api-reference/inference#extra-headers)
- [function\_name](https://www.tensorzero.com/docs/gateway/api-reference/inference#function-name)
- [include\_original\_response](https://www.tensorzero.com/docs/gateway/api-reference/inference#include-original-response)
- [input](https://www.tensorzero.com/docs/gateway/api-reference/inference#input)
- [model\_name](https://www.tensorzero.com/docs/gateway/api-reference/inference#model-name)
- [output\_schema](https://www.tensorzero.com/docs/gateway/api-reference/inference#output-schema)
- [otlp\_traces\_extra\_headers](https://www.tensorzero.com/docs/gateway/api-reference/inference#otlp-traces-extra-headers)
- [parallel\_tool\_calls](https://www.tensorzero.com/docs/gateway/api-reference/inference#parallel-tool-calls)
- [params](https://www.tensorzero.com/docs/gateway/api-reference/inference#params)
- [provider\_tools](https://www.tensorzero.com/docs/gateway/api-reference/inference#provider-tools)
- [stream](https://www.tensorzero.com/docs/gateway/api-reference/inference#stream)
- [tags](https://www.tensorzero.com/docs/gateway/api-reference/inference#tags)
- [tool\_choice](https://www.tensorzero.com/docs/gateway/api-reference/inference#tool-choice)
- [variant\_name](https://www.tensorzero.com/docs/gateway/api-reference/inference#variant-name)
- [Response](https://www.tensorzero.com/docs/gateway/api-reference/inference#response)
- [Chat Function](https://www.tensorzero.com/docs/gateway/api-reference/inference#chat-function)
- [JSON Function](https://www.tensorzero.com/docs/gateway/api-reference/inference#json-function)
- [Examples](https://www.tensorzero.com/docs/gateway/api-reference/inference#examples)
- [Chat Function](https://www.tensorzero.com/docs/gateway/api-reference/inference#chat-function-2)
- [Chat Function with Schemas](https://www.tensorzero.com/docs/gateway/api-reference/inference#chat-function-with-schemas)
- [Chat Function with Tool Use](https://www.tensorzero.com/docs/gateway/api-reference/inference#chat-function-with-tool-use)
- [Chat Function with Multi-Turn Tool Use](https://www.tensorzero.com/docs/gateway/api-reference/inference#chat-function-with-multi-turn-tool-use)
- [Chat Function with Dynamic Tool Use](https://www.tensorzero.com/docs/gateway/api-reference/inference#chat-function-with-dynamic-tool-use)
- [Chat Function with Dynamic Inference Parameters](https://www.tensorzero.com/docs/gateway/api-reference/inference#chat-function-with-dynamic-inference-parameters)
- [JSON Function](https://www.tensorzero.com/docs/gateway/api-reference/inference#json-function-2)

## [​](https://www.tensorzero.com/docs/gateway/api-reference/inference\#post-/inference)  `POST /inference`

The inference endpoint is the core of the TensorZero Gateway API.Under the hood, the gateway validates the request, samples a variant from the function, handles templating when applicable, and routes the inference to the appropriate model provider.
If a problem occurs, it attempts to gracefully fallback to a different model provider or variant.
After a successful inference, it returns the data to the client and asynchronously stores structured information in the database.

See the [API Reference for `POST /openai/v1/chat/completions`](https://www.tensorzero.com/docs/gateway/api-reference/inference-openai-compatible) for an inference endpoint compatible with the OpenAI API.

### [​](https://www.tensorzero.com/docs/gateway/api-reference/inference\#request)  Request

#### [​](https://www.tensorzero.com/docs/gateway/api-reference/inference\#additional-tools)  `additional_tools`

- **Type:** a list of tools (see below)
- **Required:** no (default: `[]`)

A list of tools defined at inference time that the model is allowed to call.
This field allows for dynamic tool use, i.e. defining tools at runtime.You should prefer to define tools in the configuration file if possible.
Only use this field if dynamic tool use is necessary for your use case.Each tool is an object with the following fields: `description`, `name`, `parameters`, and `strict`.The fields are identical to those in the configuration file, except that the `parameters` field should contain the JSON schema itself rather than a path to it.
See [Configuration Reference](https://www.tensorzero.com/docs/gateway/configuration-reference#toolstool_name) for more details.

#### [​](https://www.tensorzero.com/docs/gateway/api-reference/inference\#allowed-tools)  `allowed_tools`

- **Type:** list of strings
- **Required:** no

A list of tool names that the model is allowed to call.
The tools must be defined in the configuration file or provided dynamically via `additional_tools`.Some providers (notably OpenAI) natively support restricting allowed tools.
For these providers, we send all tools (both configured and dynamic) to the provider, and separately specify which ones are allowed to be called.
For providers that do not natively support this feature, we filter the tool list ourselves and only send the allowed tools to the provider.

#### [​](https://www.tensorzero.com/docs/gateway/api-reference/inference\#cache-options)  `cache_options`

- **Type:** object
- **Required:** no (default: `{"enabled": "write_only"}`)

Options for controlling inference caching behavior.
The object has the fields below.See [Inference Caching](https://www.tensorzero.com/docs/gateway/guides/inference-caching) for more details.

##### `cache_options.enabled`

- **Type:** string
- **Required:** no (default: `"write_only"`)

The cache mode to use.
Must be one of:

- `"write_only"` (default): Only write to cache but don’t serve cached responses
- `"read_only"`: Only read from cache but don’t write new entries
- `"on"`: Both read from and write to cache
- `"off"`: Disable caching completely

Note: When using `dryrun=true`, the gateway never writes to the cache.

##### `cache_options.max_age_s`

- **Type:** integer
- **Required:** no (default: `null`)

Maximum age in seconds for cache entries.
If set, cached responses older than this value will not be used.For example, if you set `max_age_s=3600`, the gateway will only use cache entries that were created in the last hour.

#### [​](https://www.tensorzero.com/docs/gateway/api-reference/inference\#credentials)  `credentials`

- **Type:** object (a map from dynamic credential names to API keys)
- **Required:** no (default: no credentials)

Each model provider in your TensorZero configuration can be configured to accept credentials at inference time by using the `dynamic` location (e.g. `dynamic::my_dynamic_api_key_name`).
See the [configuration reference](https://www.tensorzero.com/docs/gateway/configuration-reference#modelsmodel_nameprovidersprovider_name) for more details.
The gateway expects the credentials to be provided in the `credentials` field of the request body as specified below.
The gateway will return a 400 error if the credentials are not provided and the model provider has been configured with dynamic credentials.

Example

Copy

```
[models.my_model_name.providers.my_provider_name]
# ...
# Note: the name of the credential field (e.g. `api_key_location`) depends on the provider type
api_key_location = "dynamic::my_dynamic_api_key_name"
# ...
```

Copy

```
{
  // ...
  "credentials": {
    // ...
    "my_dynamic_api_key_name": "sk-..."
    // ...
  }
  // ...
}
```

#### [​](https://www.tensorzero.com/docs/gateway/api-reference/inference\#dryrun)  `dryrun`

- **Type:** boolean
- **Required:** no

If `true`, the inference request will be executed but won’t be stored to the database.
The gateway will still call the downstream model providers.This field is primarily for debugging and testing, and you should generally not use it in production.

#### [​](https://www.tensorzero.com/docs/gateway/api-reference/inference\#episode-id)  `episode_id`

- **Type:** UUID
- **Required:** no

The ID of an existing episode to associate the inference with.
If null, the gateway will generate a new episode ID and return it in the response.
See [Episodes](https://www.tensorzero.com/docs/gateway/guides/episodes) for more information.

#### [​](https://www.tensorzero.com/docs/gateway/api-reference/inference\#extra-body)  `extra_body`

- **Type:** array of objects (see below)
- **Required:** no

The `extra_body` field allows you to modify the request body that TensorZero sends to a model provider.
This advanced feature is an “escape hatch” that lets you use provider-specific functionality that TensorZero hasn’t implemented yet.Each object in the array must have two or three fields:

- `pointer`: A [JSON Pointer](https://datatracker.ietf.org/doc/html/rfc6901) string specifying where to modify the request body
- One of the following:
  - `value`: The value to insert at that location; it can be of any type including nested types
  - `delete = true`: Deletes the field at the specified location, if present.
- Optional: If one of the following is specified, the modification will only be applied to the specified variant, model, or model provider. If neither is specified, the modification applies to all model inferences.
  - `variant_name`
  - `model_name`
  - `model_name` and `provider_name`

You can also set `extra_body` in the configuration file.
The values provided at inference-time take priority over the values in the configuration file.

Example: \`extra\_body\`

If TensorZero would normally send this request body to the provider…

Copy

```
{
  "project": "tensorzero",
  "safety_checks": {
    "no_internet": false,
    "no_agi": true
  }
}
```

…then the following `extra_body` in the inference request…

Copy

```
{
  // ...
  "extra_body": [\
    {\
      "variant_name": "my_variant", // or "model_name": "my_model", "provider_name": "my_provider"\
      "pointer": "/agi",\
      "value": true\
    },\
    {\
      // No `variant_name` or `model_name`/`provider_name` specified, so it applies to all variants and providers\
      "pointer": "/safety_checks/no_agi",\
      "value": {\
        "bypass": "on"\
      }\
    }\
  ]
}
```

…overrides the request body to:

Copy

```
{
  "agi": true,
  "project": "tensorzero",
  "safety_checks": {
    "no_internet": false,
    "no_agi": {
      "bypass": "on"
    }
  }
}
```

#### [​](https://www.tensorzero.com/docs/gateway/api-reference/inference\#extra-headers)  `extra_headers`

- **Type:** array of objects (see below)
- **Required:** no

The `extra_headers` field allows you to modify the request headers that TensorZero sends to a model provider.
This advanced feature is an “escape hatch” that lets you use provider-specific functionality that TensorZero hasn’t implemented yet.Each object in the array must have two or three fields:

- `name`: The name of the header to modify
- `value`: The value to set the header to
- Optional: If one of the following is specified, the modification will only be applied to the specified variant, model, or model provider. If neither is specified, the modification applies to all model inferences.
  - `variant_name`
  - `model_name`
  - `model_name` and `provider_name`

You can also set `extra_headers` in the configuration file.
The values provided at inference-time take priority over the values in the configuration file.

Example: \`extra\_headers\`

If TensorZero would normally send the following request headers to the provider…

Copy

```
Safety-Checks: on
```

…then the following `extra_headers`…

Copy

```
{
  "extra_headers": [\
    {\
      "variant_name": "my_variant", // or "model_name": "my_model", "provider_name": "my_provider"\
      "name": "Safety-Checks",\
      "value": "off"\
    },\
    {\
      // No `variant_name` or `model_name`/`provider_name` specified, so it applies to all variants and providers\
      "name": "Intelligence-Level",\
      "value": "AGI"\
    }\
  ]
}
```

…overrides the request headers so that `Safety-Checks` is set to `off` only for `my_variant`, while `Intelligence-Level: AGI` is applied globally to all variants and providers:

Copy

```
Safety-Checks: off
Intelligence-Level: AGI
```

#### [​](https://www.tensorzero.com/docs/gateway/api-reference/inference\#function-name)  `function_name`

- **Type:** string
- **Required:** either `function_name` or `model_name` must be provided

The name of the function to call.The function must be defined in the configuration file.Alternatively, you can use the `model_name` field to call a model directly, without the need to define a function.
See below for more details.

#### [​](https://www.tensorzero.com/docs/gateway/api-reference/inference\#include-original-response)  `include_original_response`

- **Type:** boolean
- **Required:** no

If `true`, the original response from the model will be included in the response in the `original_response` field as a string.See `original_response` in the [response](https://www.tensorzero.com/docs/gateway/api-reference/inference#response) section for more details.

#### [​](https://www.tensorzero.com/docs/gateway/api-reference/inference\#input)  `input`

- **Type:** varies
- **Required:** yes

The input to the function.The type of the input depends on the function type.

##### `input.messages`

- **Type:** list of messages (see below)
- **Required:** no (default: `[]`)

A list of messages to provide to the model.Each message is an object with the following fields:

- `role`: The role of the message (`assistant` or `user`).
- `content`: The content of the message (see below).

The `content` field can be have one of the following types:

- string: the text for a text message (only allowed if there is no schema for that role)
- list of content blocks: the content blocks for the message (see below)

A content block is an object with the field `type` and additional fields depending on the type.If the content block has type `text`, it must have either of the following additional fields:

- `text`: The text for the content block.
- `arguments`: A JSON object containing the function arguments for TensorZero functions with templates and schemas (see [Create a prompt template](https://www.tensorzero.com/docs/gateway/create-a-prompt-template) for details).

If the content block has type `tool_call`, it must have the following additional fields:

- `arguments`: The arguments for the tool call.
- `id`: The ID for the content block.
- `name`: The name of the tool for the content block.

If the content block has type `tool_result`, it must have the following additional fields:

- `id`: The ID for the content block.
- `name`: The name of the tool for the content block.
- `result`: The result of the tool call.

If the content block has type `file`, it must have exactly one of the following additional fields:

- File URLs
  - `file_type`: must be `url`
  - `url`
  - `mime_type` (optional): override the MIME type of the file
  - `detail` (optional): controls the fidelity of image processing. Only applies to image files; ignored for other file types. Can be `low`, `high`, or `auto`. Affects token consumption and image quality. Only supported by some model providers; ignored otherwise.
- Base64-encoded Files
  - `file_type`: must be `base64`
  - `data`: `base64`-encoded data for an embedded file
  - `mime_type`: the MIME type (e.g. `image/png`, `image/jpeg`, `application/pdf`)
  - `detail` (optional): controls the fidelity of image processing. Only applies to image files; ignored for other file types. Can be `low`, `high`, or `auto`. Affects token consumption and image quality. Only supported by some model providers; ignored otherwise.

See the [Multimodal Inference](https://www.tensorzero.com/docs/gateway/guides/multimodal-inference) guide for more details on how to use images in inference.If the content block has type `raw_text`, it must have the following additional fields:

- `value`: The text for the content block.
This content block will ignore any relevant templates and schemas for this function.

If the content block has type `thought`, it must have the following additional fields:

- `text`: The text for the content block.

If the content block has type `unknown`, it must have the following additional fields:

- `data`: The original content block from the provider, without any validation or transformation by TensorZero.
- `model_provider_name` (optional): A string specifying when this content block should be included in the model provider input.
If set, the content block will only be provided to this specific model provider.
If not set, the content block is passed to all model providers.

For example, the following hypothetical unknown content block will send the `daydreaming` content block to inference requests targeting the `your_model_provider_name` model provider.

Copy

```
{
  "type": "unknown",
  "data": {
    "type": "daydreaming",
    "dream": "..."
  },
  "model_provider_name": "tensorzero::model_name::your_model_name::provider_name::your_model_provider_name"
}
```

Certain reasoning models (e.g. DeepSeek R1) can include `thought` content blocks in the response.
These content blocks can’t directly be used as inputs to subsequent inferences in multi-turn scenarios.
If you need to provide `thought` content blocks to a model, you should convert them to `text` content blocks.

This is the most complex field in the entire API. See this example for more details.

Example

Copy

```
{
  // ...
  "input": {
    "messages": [\
      // If you don't have a user (or assistant) schema...\
      {\
        "role": "user", // (or "assistant")\
        "content": "What is the weather in Tokyo?"\
      },\
      // If you have a user (or assistant) schema...\
      {\
        "role": "user", // (or "assistant")\
        "content": [\
          {\
            "type": "text",\
            "arguments": {\
              "location": "Tokyo"\
            }\
          }\
        ]\
      },\
      // If the model previously called a tool...\
      {\
        "role": "assistant",\
        "content": [\
          {\
            "type": "tool_call",\
            "id": "0",\
            "name": "get_temperature",\
            "arguments": "{\"location\": \"Tokyo\"}"\
          }\
        ]\
      },\
      // ...and you're providing the result of that tool call...\
      {\
        "role": "user",\
        "content": [\
          {\
            "type": "tool_result",\
            "id": "0",\
            "name": "get_temperature",\
            "result": "70"\
          }\
        ]\
      },\
      // You can also specify a text message using a content block...\
      {\
        "role": "user",\
        "content": [\
          {\
            "type": "text",\
            "text": "What about NYC?" // (or object if there is a schema)\
          }\
        ]\
      },\
      // You can also provide multiple content blocks in a single message...\
      {\
        "role": "assistant",\
        "content": [\
          {\
            "type": "text",\
            "text": "Sure, I can help you with that." // (or object if there is a schema)\
          },\
          {\
            "type": "tool_call",\
            "id": "0",\
            "name": "get_temperature",\
            "arguments": "{\"location\": \"New York\"}"\
          }\
        ]\
      }\
      // ...\
    ]
    // ...
  }
  // ...
}
```

##### `input.system`

- **Type:** string or object
- **Required:** no

The input for the system message.If the function does not have a system schema, this field should be a string.If the function has a system schema, this field should be an object that matches the schema.

#### [​](https://www.tensorzero.com/docs/gateway/api-reference/inference\#model-name)  `model_name`

- **Type:** string
- **Required:** either `model_name` or `function_name` must be provided

The name of the model to call.Under the hood, the gateway will use a built-in passthrough chat function called `tensorzero::default`.

|     |     |
| --- | --- |
| **To call…** | **Use this format…** |
| A function defined as `[functions.my_function]` in your<br>`tensorzero.toml` configuration file | `function_name="my_function"` (not `model_name`) |
| A model defined as `[models.my_model]` in your `tensorzero.toml`<br>configuration file | `model_name="my_model"` |
| A model offered by a model provider, without defining it in your<br>`tensorzero.toml` configuration file (if supported, see below) | `model_name="{provider_type}::{model_name}"` |

The following model providers support short-hand model names: `anthropic`, `deepseek`, `fireworks`, `gcp_vertex_anthropic`, `gcp_vertex_gemini`, `google_ai_studio_gemini`, `groq`, `hyperbolic`, `mistral`, `openai`, `openrouter`, `together`, and `xai`.

For example, if you have the following configuration:

tensorzero.toml

Copy

```
[models.gpt-4o]
routing = ["openai", "azure"]

[models.gpt-4o.providers.openai]
# ...

[models.gpt-4o.providers.azure]
# ...

[functions.extract-data]
# ...
```

Then:

- `function_name="extract-data"` calls the `extract-data` function defined above.
- `model_name="gpt-4o"` calls the `gpt-4o` model in your configuration, which supports fallback from `openai` to `azure`. See [Retries & Fallbacks](https://www.tensorzero.com/docs/gateway/guides/retries-fallbacks) for details.
- `model_name="openai::gpt-4o"` calls the OpenAI API directly for the `gpt-4o` model, ignoring the `gpt-4o` model defined above.

Be careful about the different prefixes: `model_name="gpt-4o"` will use the `[models.gpt-4o]` model defined in the `tensorzero.toml` file, whereas `model_name="openai::gpt-4o"` will call the OpenAI API directly for the `gpt-4o` model.

#### [​](https://www.tensorzero.com/docs/gateway/api-reference/inference\#output-schema)  `output_schema`

- **Type:** object (valid JSON Schema)
- **Required:** no

If set, this schema will override the `output_schema` defined in the function configuration for a JSON function.
This dynamic output schema is used for validating the output of the function, and sent to providers which support structured outputs.

#### [​](https://www.tensorzero.com/docs/gateway/api-reference/inference\#otlp-traces-extra-headers)  `otlp_traces_extra_headers`

- **Type:** object (a map from string to string)
- **Required:** no (default: `{}`)

Dynamic headers to include in OTLP trace exports for this specific inference request.
This is useful for adding per-request metadata to OTLP trace exports (e.g. user IDs, request sources).The headers are automatically prefixed with `tensorzero-otlp-traces-extra-header-` before being sent to the OTLP endpoint.These headers are merged with any static headers configured in `export.otlp.traces.extra_headers`.
When the same header key is present in both static and dynamic headers, the dynamic header value takes precedence.See [Export OpenTelemetry traces](https://www.tensorzero.com/docs/operations/export-opentelemetry-traces#send-custom-http-headers) for more details and examples.

#### [​](https://www.tensorzero.com/docs/gateway/api-reference/inference\#parallel-tool-calls)  `parallel_tool_calls`

- **Type:** boolean
- **Required:** no

If `true`, the function will be allowed to request multiple tool calls in a single conversation turn.
If not set, we default to the configuration value for the function being called.Most model providers do not support parallel tool calls. In those cases, the gateway ignores this field.
At the moment, only Fireworks AI and OpenAI support parallel tool calls.

#### [​](https://www.tensorzero.com/docs/gateway/api-reference/inference\#params)  `params`

- **Type:** object (see below)
- **Required:** no (default: `{}`)

Override inference-time parameters for a particular variant type.
This fields allows for dynamic inference parameters, i.e. defining parameters at runtime.This field’s format is `{ variant_type: { param: value, ... }, ... }`.
You should prefer to set these parameters in the configuration file if possible.
Only use this field if you need to set these parameters dynamically at runtime.Note that the parameters will apply to every variant of the specified type.Currently, we support the following:

- `chat_completion`
  - `frequency_penalty`
  - `json_mode`
  - `max_tokens`
  - `presence_penalty`
  - `reasoning_effort`
  - `seed`
  - `service_tier`
  - `stop_sequences`
  - `temperature`
  - `thinking_budget_tokens`
  - `top_p`
  - `verbosity`

See [Configuration Reference](https://www.tensorzero.com/docs/gateway/configuration-reference#functionsfunction_namevariantsvariant_name) for more details on the parameters, and Examples below for usage.

Example

For example, if you wanted to dynamically override the `temperature` parameter for a `chat_completion` variants, you’d include the following in the request body:

Copy

```
{
  // ...
  "params": {
    "chat_completion": {
      "temperature": 0.7
    }
  }
  // ...
}
```

See [“Chat Function with Dynamic Inference Parameters”](https://www.tensorzero.com/docs/gateway/api-reference/inference#chat-function-with-dynamic-inference-parameters) for a complete example.

#### [​](https://www.tensorzero.com/docs/gateway/api-reference/inference\#provider-tools)  `provider_tools`

- **Type:** array of objects
- **Required:** no (default: `[]`)

A list of provider-specific built-in tools defined at inference time that can be used by the model.
These are tools that run server-side on the provider’s infrastructure, such as OpenAI’s web search tool.Each object in the array has the following fields:

- `scope` (object, optional): Limits which model/provider combination can use this tool. If omitted, the tool is available to all compatible providers.

  - `model_name` (string): The model name as defined in your configuration
  - `model_provider_name` (string): The provider name for that model
- `tool` (object, required): The provider-specific tool configuration as defined by the provider’s API

This field allows for dynamic provider tool use at runtime.
You should prefer to define provider tools in the configuration file if possible (see [Configuration Reference](https://www.tensorzero.com/docs/gateway/configuration-reference#provider_tools)).
Only use this field if dynamic provider tool configuration is necessary for your use case.

Example: OpenAI Web Search (Unscoped)

Copy

```
{
  "function_name": "my_function",
  "input": {
    "messages": [\
      {\
        "role": "user",\
        "content": "What were the latest developments in AI this week?"\
      }\
    ]
  },
  "provider_tools": [\
    {\
      "tool": {\
        "type": "web_search"\
      }\
    }\
  ]
}
```

This makes the web search tool available to all compatible providers configured for the function.

Example: OpenAI Web Search (Scoped)

Copy

```
{
  "function_name": "my_function",
  "input": {
    "messages": [\
      {\
        "role": "user",\
        "content": "What were the latest developments in AI this week?"\
      }\
    ]
  },
  "provider_tools": [\
    {\
      "scope": {\
        "model_name": "gpt-5-mini",\
        "model_provider_name": "openai"\
      },\
      "tool": {\
        "type": "web_search"\
      }\
    }\
  ]
}
```

This makes the web search tool available only to the OpenAI provider for the `gpt-5-mini` model.

#### [​](https://www.tensorzero.com/docs/gateway/api-reference/inference\#stream)  `stream`

- **Type:** boolean
- **Required:** no

If `true`, the gateway will stream the response from the model provider.

#### [​](https://www.tensorzero.com/docs/gateway/api-reference/inference\#tags)  `tags`

- **Type:** flat JSON object with string keys and values
- **Required:** no

User-provided tags to associate with the inference.For example, `{"user_id": "123"}` or `{"author": "Alice"}`.

#### [​](https://www.tensorzero.com/docs/gateway/api-reference/inference\#tool-choice)  `tool_choice`

- **Type:** string
- **Required:** no

If set, overrides the tool choice strategy for the request.The supported tool choice strategies are:

- `none`: The function should not use any tools.
- `auto`: The model decides whether or not to use a tool. If it decides to use a tool, it also decides which tools to use.
- `required`: The model should use a tool. If multiple tools are available, the model decides which tool to use.
- `{ specific = "tool_name" }`: The model should use a specific tool. The tool must be defined in the `tools` section of the configuration file or provided in `additional_tools`.

#### [​](https://www.tensorzero.com/docs/gateway/api-reference/inference\#variant-name)  `variant_name`

- **Type:** string
- **Required:** no

If set, pins the inference request to a particular variant (not recommended).You should generally not set this field, and instead let the TensorZero gateway assign a variant.
This field is primarily used for testing or debugging purposes.

### [​](https://www.tensorzero.com/docs/gateway/api-reference/inference\#response)  Response

The response format depends on the function type (as defined in the configuration file) and whether the response is streamed or not.

#### [​](https://www.tensorzero.com/docs/gateway/api-reference/inference\#chat-function)  Chat Function

When the function type is `chat`, the response is structured as follows.

- Regular

- Streaming


In regular (non-streaming) mode, the response is a JSON object with the following fields:

##### `content`

- **Type:** a list of content blocks (see below)

The content blocks generated by the model.A content block can have `type` equal to `text` and `tool_call`.
Reasoning models (e.g. DeepSeek R1) might also include `thought` content blocks.If `type` is `text`, the content block has the following fields:

- `text`: The text for the content block.

If `type` is `tool_call`, the content block has the following fields:

- `arguments` (object): The validated arguments for the tool call (`null` if invalid).
- `id` (string): The ID of the content block.
- `name` (string): The validated name of the tool (`null` if invalid).
- `raw_arguments` (string): The arguments for the tool call generated by the model (which might be invalid).
- `raw_name` (string): The name of the tool generated by the model (which might be invalid).

If `type` is `thought`, the content block has the following fields:

- `text` (string): The text of the thought.

If the model provider responds with a content block of an unknown type, it will be included in the response as a content block of type `unknown` with the following additional fields:

- `data`: The original content block from the provider, without any validation or transformation by TensorZero.
- `model_provider_name`: The fully-qualified name of the model provider that returned the content block.

For example, if the model provider `your_model_provider_name` returns a content block of type `daydreaming`, it will be included in the response like this:

Copy

```
{
  "type": "unknown",
  "data": {
    "type": "daydreaming",
    "dream": "..."
  },
  "model_provider_name": "tensorzero::model_name::your_model_name::provider_name::your_model_provider_name"
}
```

##### `episode_id`

- **Type:** UUID

The ID of the episode associated with the inference.

##### `inference_id`

- **Type:** UUID

The ID assigned to the inference.

##### `original_response`

- **Type:** string (optional)

The original response from the model provider (only available when `include_original_response` is `true`).The returned data depends on the variant type:

- `chat_completion`: raw response from the inference to the `model`
- `experimental_best_of_n_sampling`: raw response from the inference to the `evaluator`
- `experimental_mixture_of_n_sampling`: raw response from the inference to the `fuser`
- `experimental_dynamic_in_context_learning`: raw response from the inference to the `model`
- `experimental_chain_of_thought`: raw response from the inference to the `model`

##### `variant_name`

- **Type:** string

The name of the variant used for the inference.

##### `usage`

- **Type:** object (optional)

The usage metrics for the inference.The object has the following fields:

- `input_tokens`: The number of input tokens used for the inference.
- `output_tokens`: The number of output tokens used for the inference.

#### [​](https://www.tensorzero.com/docs/gateway/api-reference/inference\#json-function)  JSON Function

When the function type is `json`, the response is structured as follows.

- Regular

- Streaming


In regular (non-streaming) mode, the response is a JSON object with the following fields:

##### `inference_id`

- **Type:** UUID

The ID assigned to the inference.

##### `episode_id`

- **Type:** UUID

The ID of the episode associated with the inference.

##### `original_response`

- **Type:** string (optional)

The original response from the model provider (only available when `include_original_response` is `true`).The returned data depends on the variant type:

- `chat_completion`: raw response from the inference to the `model`
- `experimental_best_of_n_sampling`: raw response from the inference to the `evaluator`
- `experimental_mixture_of_n_sampling`: raw response from the inference to the `fuser`
- `experimental_dynamic_in_context_learning`: raw response from the inference to the `model`
- `experimental_chain_of_thought`: raw response from the inference to the `model`

##### `output`

- **Type:** object (see below)

The output object contains the following fields:

- `raw`: The raw response from the model provider (which might be invalid JSON).
- `parsed`: The parsed response from the model provider (`null` if invalid JSON).

##### `variant_name`

- **Type:** string

The name of the variant used for the inference.

##### `usage`

- **Type:** object (optional)

The usage metrics for the inference.The object has the following fields:

- `input_tokens`: The number of input tokens used for the inference.
- `output_tokens`: The number of output tokens used for the inference.

### [​](https://www.tensorzero.com/docs/gateway/api-reference/inference\#examples)  Examples

#### [​](https://www.tensorzero.com/docs/gateway/api-reference/inference\#chat-function-2)  Chat Function

Chat Function

##### Configuration

Copy

```
// tensorzero.toml
# ...
[functions.draft_email]
type = "chat"
# ...
```

##### Request

- Python

- HTTP


POST /inference

Copy

```
from tensorzero import AsyncTensorZeroGateway

async with await AsyncTensorZeroGateway.build_http(gateway_url="http://localhost:3000") as client:
    result = await client.inference(
        function_name="draft_email",
        input={
            "system": "You are an AI assistant...",
            "messages": [\
                {\
                  "role": "user",\
                  "content": "I need to write an email to Gabriel explaining..."\
                }\
            ]
        }
        # optional: stream=True,
    )
```

##### Response

- Regular

- Streaming


POST /inference

Copy

```
{
  "inference_id": "00000000-0000-0000-0000-000000000000",
  "episode_id": "11111111-1111-1111-1111-111111111111",
  "variant_name": "prompt_v1",
  "content": [\
    {\
      "type": "text",\
      "text": "Hi Gabriel,\n\nI noticed...",\
    }\
  ]
  "usage": {
    "input_tokens": 100,
    "output_tokens": 100
  }
}
```

#### [​](https://www.tensorzero.com/docs/gateway/api-reference/inference\#chat-function-with-schemas)  Chat Function with Schemas

Chat Function with Schemas

##### Configuration

Copy

```
// tensorzero.toml
# ...
[functions.draft_email]
type = "chat"
system_schema = "system_schema.json"
user_schema = "user_schema.json"
# ...
```

Copy

```
// system_schema.json
{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "type": "object",
  "properties": {
    "tone": {
      "type": "string"
    }
  },
  "required": ["tone"],
  "additionalProperties": false
}
```

Copy

```
// user_schema.json
{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "type": "object",
  "properties": {
    "recipient": {
      "type": "string"
    },
    "email_purpose": {
      "type": "string"
    }
  },
  "required": ["recipient", "email_purpose"],
  "additionalProperties": false
}
```

##### Request

- Python

- HTTP


POST /inference

Copy

```
from tensorzero import AsyncTensorZeroGateway

async with await AsyncTensorZeroGateway.build_http(gateway_url="http://localhost:3000") as client:
    result = await client.inference(
        function_name="draft_email",
        input={
            "system": {"tone": "casual"},
            "messages": [\
                {\
                    "role": "user",\
                    "content": [\
                        {\
                            "type": "text",\
                            "arguments": {\
                                "recipient": "Gabriel",\
                                "email_purpose": "Request a meeting to..."\
                            }\
                        }\
                    ]\
                }\
            ]
        }
        # optional: stream=True,
    )
```

##### Response

- Regular

- Streaming


POST /inference

Copy

```
{
  "inference_id": "00000000-0000-0000-0000-000000000000",
  "episode_id": "11111111-1111-1111-1111-111111111111",
  "variant_name": "prompt_v1",
  "content": [\
    {\
      "type": "text",\
      "text": "Hi Gabriel,\n\nI noticed...",\
    }\
  ]
  "usage": {
    "input_tokens": 100,
    "output_tokens": 100
  }
}
```

#### [​](https://www.tensorzero.com/docs/gateway/api-reference/inference\#chat-function-with-tool-use)  Chat Function with Tool Use

Chat Function with Tool Use

##### Configuration

Copy

```
// tensorzero.toml
# ...

[functions.weather_bot]
type = "chat"
tools = ["get_temperature"]

# ...

[tools.get_temperature]
description = "Get the current temperature in a given location"
parameters = "get_temperature.json"

# ...
```

Copy

```
// get_temperature.json
{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "type": "object",
  "properties": {
    "location": {
      "type": "string",
      "description": "The location to get the temperature for (e.g. \"New York\")"
    },
    "units": {
      "type": "string",
      "description": "The units to get the temperature in (must be \"fahrenheit\" or \"celsius\")",
      "enum": ["fahrenheit", "celsius"]
    }
  },
  "required": ["location"],
  "additionalProperties": false
}
```

##### Request

- Python

- HTTP


POST /inference

Copy

```
from tensorzero import AsyncTensorZeroGateway

async with await AsyncTensorZeroGateway.build_http(gateway_url="http://localhost:3000") as client:
    result = await client.inference(
        function_name="weather_bot",
        input={
            "messages": [\
                {\
                    "role": "user",\
                    "content": "What is the weather like in Tokyo?"\
                }\
            ]
        }
        # optional: stream=True,
    )
```

##### Response

- Regular

- Streaming


POST /inference

Copy

```
{
  "inference_id": "00000000-0000-0000-0000-000000000000",
  "episode_id": "11111111-1111-1111-1111-111111111111",
  "variant_name": "prompt_v1",
  "content": [\
    {\
      "type": "tool_call",\
      "arguments": {\
        "location": "Tokyo",\
        "units": "celsius"\
      },\
      "id": "123456789",\
      "name": "get_temperature",\
      "raw_arguments": "{\"location\": \"Tokyo\", \"units\": \"celsius\"}",\
      "raw_name": "get_temperature"\
    }\
  ],
  "usage": {
    "input_tokens": 100,
    "output_tokens": 100
  }
}
```

#### [​](https://www.tensorzero.com/docs/gateway/api-reference/inference\#chat-function-with-multi-turn-tool-use)  Chat Function with Multi-Turn Tool Use

Chat Function with Multi-Turn Tool Use

##### Configuration

Copy

```
// tensorzero.toml
# ...

[functions.weather_bot]
type = "chat"
tools = ["get_temperature"]

# ...

[tools.get_temperature]
description = "Get the current temperature in a given location"
parameters = "get_temperature.json"

# ...
```

Copy

```
// get_temperature.json
{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "type": "object",
  "properties": {
    "location": {
      "type": "string",
      "description": "The location to get the temperature for (e.g. \"New York\")"
    },
    "units": {
      "type": "string",
      "description": "The units to get the temperature in (must be \"fahrenheit\" or \"celsius\")",
      "enum": ["fahrenheit", "celsius"]
    }
  },
  "required": ["location"],
  "additionalProperties": false
}
```

##### Request

- Python

- HTTP


POST /inference

Copy

```
from tensorzero import AsyncTensorZeroGateway

async with await AsyncTensorZeroGateway.build_http(gateway_url="http://localhost:3000") as client:
    result = await client.inference(
        function_name="weather_bot",
        input={
            "messages": [\
                {\
                    "role": "user",\
                    "content": "What is the weather like in Tokyo?"\
                },\
                {\
                    "role": "assistant",\
                    "content": [\
                        {\
                            "type": "tool_call",\
                            "arguments": {\
                                "location": "Tokyo",\
                                "units": "celsius"\
                            },\
                            "id": "123456789",\
                            "name": "get_temperature",\
                        }\
                    ]\
                },\
                {\
                    "role": "user",\
                    "content": [\
                        {\
                            "type": "tool_result",\
                            "id": "123456789",\
                            "name": "get_temperature",\
                            "result": "25"  # the tool result must be a string\
                        }\
                    ]\
                }\
            ]
        }
        # optional: stream=True,
    )
```

##### Response

- Regular

- Streaming


POST /inference

Copy

```
{
  "inference_id": "00000000-0000-0000-0000-000000000000",
  "episode_id": "11111111-1111-1111-1111-111111111111",
  "variant_name": "prompt_v1",
  "content": [\
    {\
      "type": "text",\
      "content": [\
        {\
          "type": "text",\
          "text": "The weather in Tokyo is 25 degrees Celsius."\
        }\
      ]\
    }\
  ],
  "usage": {
    "input_tokens": 100,
    "output_tokens": 100
  }
}
```

#### [​](https://www.tensorzero.com/docs/gateway/api-reference/inference\#chat-function-with-dynamic-tool-use)  Chat Function with Dynamic Tool Use

Chat Function with Dynamic Tool Use

##### Configuration

Copy

```
// tensorzero.toml
# ...

[functions.weather_bot]
type = "chat"
# Note: no `tools = ["get_temperature"]` field in configuration

# ...
```

##### Request

- Python

- HTTP


POST /inference

Copy

```
from tensorzero import AsyncTensorZeroGateway

async with await AsyncTensorZeroGateway.build_http(gateway_url="http://localhost:3000") as client:
    result = await client.inference(
        function_name="weather_bot",
        input={
            "messages": [\
                {\
                    "role": "user",\
                    "content": "What is the weather like in Tokyo?"\
                }\
            ]
        },
        additional_tools=[\
            {\
                "name": "get_temperature",\
                "description": "Get the current temperature in a given location",\
                "parameters": {\
                    "$schema": "http://json-schema.org/draft-07/schema#",\
                    "type": "object",\
                    "properties": {\
                        "location": {\
                            "type": "string",\
                            "description": "The location to get the temperature for (e.g. \"New York\")"\
                        },\
                        "units": {\
                            "type": "string",\
                            "description": "The units to get the temperature in (must be \"fahrenheit\" or \"celsius\")",\
                            "enum": ["fahrenheit", "celsius"]\
                        }\
                    },\
                    "required": ["location"],\
                    "additionalProperties": false\
                }\
            }\
        ],
        # optional: stream=True,
    )
```

##### Response

- Regular

- Streaming


POST /inference

Copy

```
{
  "inference_id": "00000000-0000-0000-0000-000000000000",
  "episode_id": "11111111-1111-1111-1111-111111111111",
  "variant_name": "prompt_v1",
  "content": [\
    {\
      "type": "tool_call",\
      "arguments": {\
        "location": "Tokyo",\
        "units": "celsius"\
      },\
      "id": "123456789",\
      "name": "get_temperature",\
      "raw_arguments": "{\"location\": \"Tokyo\", \"units\": \"celsius\"}",\
      "raw_name": "get_temperature"\
    }\
  ],
  "usage": {
    "input_tokens": 100,
    "output_tokens": 100
  }
}
```

#### [​](https://www.tensorzero.com/docs/gateway/api-reference/inference\#chat-function-with-dynamic-inference-parameters)  Chat Function with Dynamic Inference Parameters

Chat Function with Dynamic Inference Parameters

##### Configuration

Copy

```
// tensorzero.toml
# ...
[functions.draft_email]
type = "chat"
# ...

[functions.draft_email.variants.prompt_v1]
type = "chat_completion"
temperature = 0.5  # the API request will override this value
# ...
```

##### Request

- Python

- HTTP


POST /inference

Copy

```
from tensorzero import AsyncTensorZeroGateway

async with await AsyncTensorZeroGateway.build_http(gateway_url="http://localhost:3000") as client:
    result = await client.inference(
        function_name="draft_email",
        input={
            "system": "You are an AI assistant...",
            "messages": [\
                {\
                    "role": "user",\
                    "content": "I need to write an email to Gabriel explaining..."\
                }\
            ]
        },
        # Override parameters for every variant with type "chat_completion"
        params={
            "chat_completion": {
                "temperature": 0.7,
            }
        },
        # optional: stream=True,
    )
```

##### Response

- Regular

- Streaming


POST /inference

Copy

```
{
  "inference_id": "00000000-0000-0000-0000-000000000000",
  "episode_id": "11111111-1111-1111-1111-111111111111",
  "variant_name": "prompt_v1",
  "content": [\
    {\
      "type": "text",\
      "text": "Hi Gabriel,\n\nI noticed...",\
    }\
  ]
  "usage": {
    "input_tokens": 100,
    "output_tokens": 100
  }
}
```

#### [​](https://www.tensorzero.com/docs/gateway/api-reference/inference\#json-function-2)  JSON Function

JSON Function

##### Configuration

Copy

```
// tensorzero.toml
# ...
[functions.extract_email]
type = "json"
output_schema = "output_schema.json"
# ...
```

Copy

```
// output_schema.json
{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "type": "object",
  "properties": {
    "email": {
      "type": "string"
    }
  },
  "required": ["email"]
}
```

##### Request

- Python

- HTTP


POST /inference

Copy

```
from tensorzero import AsyncTensorZeroGateway

async with await AsyncTensorZeroGateway.build_http(gateway_url="http://localhost:3000") as client:
    result = await client.inference(
        function_name="extract_email",
        input={
            "system": "You are an AI assistant...",
            "messages": [\
                {\
                    "role": "user",\
                    "content": "...blah blah blah hello@tensorzero.com blah blah blah..."\
                }\
            ]
        }
        # optional: stream=True,
    )
```

##### Response

- Regular

- Streaming


POST /inference

Copy

```
{
  "inference_id": "00000000-0000-0000-0000-000000000000",
  "episode_id": "11111111-1111-1111-1111-111111111111",
  "variant_name": "prompt_v1",
  "output": {
    "raw": "{\"email\": \"hello@tensorzero.com\"}",
    "parsed": {
      "email": "hello@tensorzero.com"
    }
  }
  "usage": {
    "input_tokens": 100,
    "output_tokens": 100
  }
}
```

[Data Model](https://www.tensorzero.com/docs/gateway/data-model) [Inference (OpenAI)](https://www.tensorzero.com/docs/gateway/api-reference/inference-openai-compatible)

Ctrl+I

Assistant

Responses are generated using AI and may contain mistakes.

## OpenAI-Compatible Inference API
[Skip to main content](https://www.tensorzero.com/docs/gateway/api-reference/inference-openai-compatible#content-area)

[TensorZero Docs home page![light logo](https://mintcdn.com/tensorzero/afFfERJmQ8YiZpCj/logo/light.svg?fit=max&auto=format&n=afFfERJmQ8YiZpCj&q=85&s=9396731a8a92cee7bd293dee05d71dac)![dark logo](https://mintcdn.com/tensorzero/afFfERJmQ8YiZpCj/logo/dark.svg?fit=max&auto=format&n=afFfERJmQ8YiZpCj&q=85&s=8d61c59531dbcdcace7fcdfc524bd4d2)](https://www.tensorzero.com/)

Search...

Ctrl K

Search...

Navigation

API Reference

API Reference: Inference (OpenAI-Compatible)

[Guides](https://www.tensorzero.com/docs) [Integrations](https://www.tensorzero.com/docs/integrations/model-providers)

On this page

- [POST /openai/v1/chat/completions](https://www.tensorzero.com/docs/gateway/api-reference/inference-openai-compatible#post-/openai/v1/chat/completions)
- [Request](https://www.tensorzero.com/docs/gateway/api-reference/inference-openai-compatible#request)
- [tensorzero::cache\_options](https://www.tensorzero.com/docs/gateway/api-reference/inference-openai-compatible#tensorzero::cache-options)
- [tensorzero::credentials](https://www.tensorzero.com/docs/gateway/api-reference/inference-openai-compatible#tensorzero::credentials)
- [tensorzero::deny\_unknown\_fields](https://www.tensorzero.com/docs/gateway/api-reference/inference-openai-compatible#tensorzero::deny-unknown-fields)
- [tensorzero::dryrun](https://www.tensorzero.com/docs/gateway/api-reference/inference-openai-compatible#tensorzero::dryrun)
- [tensorzero::episode\_id](https://www.tensorzero.com/docs/gateway/api-reference/inference-openai-compatible#tensorzero::episode-id)
- [tensorzero::extra\_body](https://www.tensorzero.com/docs/gateway/api-reference/inference-openai-compatible#tensorzero::extra-body)
- [tensorzero::extra\_headers](https://www.tensorzero.com/docs/gateway/api-reference/inference-openai-compatible#tensorzero::extra-headers)
- [tensorzero::params](https://www.tensorzero.com/docs/gateway/api-reference/inference-openai-compatible#tensorzero::params)
- [tensorzero::provider\_tools](https://www.tensorzero.com/docs/gateway/api-reference/inference-openai-compatible#tensorzero::provider-tools)
- [tensorzero::tags](https://www.tensorzero.com/docs/gateway/api-reference/inference-openai-compatible#tensorzero::tags)
- [frequency\_penalty](https://www.tensorzero.com/docs/gateway/api-reference/inference-openai-compatible#frequency-penalty)
- [max\_completion\_tokens](https://www.tensorzero.com/docs/gateway/api-reference/inference-openai-compatible#max-completion-tokens)
- [max\_tokens](https://www.tensorzero.com/docs/gateway/api-reference/inference-openai-compatible#max-tokens)
- [messages](https://www.tensorzero.com/docs/gateway/api-reference/inference-openai-compatible#messages)
- [model](https://www.tensorzero.com/docs/gateway/api-reference/inference-openai-compatible#model)
- [parallel\_tool\_calls](https://www.tensorzero.com/docs/gateway/api-reference/inference-openai-compatible#parallel-tool-calls)
- [presence\_penalty](https://www.tensorzero.com/docs/gateway/api-reference/inference-openai-compatible#presence-penalty)
- [response\_format](https://www.tensorzero.com/docs/gateway/api-reference/inference-openai-compatible#response-format)
- [seed](https://www.tensorzero.com/docs/gateway/api-reference/inference-openai-compatible#seed)
- [stop\_sequences](https://www.tensorzero.com/docs/gateway/api-reference/inference-openai-compatible#stop-sequences)
- [stream](https://www.tensorzero.com/docs/gateway/api-reference/inference-openai-compatible#stream)
- [stream\_options](https://www.tensorzero.com/docs/gateway/api-reference/inference-openai-compatible#stream-options)
- [temperature](https://www.tensorzero.com/docs/gateway/api-reference/inference-openai-compatible#temperature)
- [tools](https://www.tensorzero.com/docs/gateway/api-reference/inference-openai-compatible#tools)
- [tool\_choice](https://www.tensorzero.com/docs/gateway/api-reference/inference-openai-compatible#tool-choice)
- [top\_p](https://www.tensorzero.com/docs/gateway/api-reference/inference-openai-compatible#top-p)
- [tensorzero::variant\_name](https://www.tensorzero.com/docs/gateway/api-reference/inference-openai-compatible#tensorzero::variant-name)
- [Response](https://www.tensorzero.com/docs/gateway/api-reference/inference-openai-compatible#response)
- [choices](https://www.tensorzero.com/docs/gateway/api-reference/inference-openai-compatible#choices)
- [created](https://www.tensorzero.com/docs/gateway/api-reference/inference-openai-compatible#created)
- [episode\_id](https://www.tensorzero.com/docs/gateway/api-reference/inference-openai-compatible#episode-id)
- [id](https://www.tensorzero.com/docs/gateway/api-reference/inference-openai-compatible#id)
- [model](https://www.tensorzero.com/docs/gateway/api-reference/inference-openai-compatible#model-2)
- [object](https://www.tensorzero.com/docs/gateway/api-reference/inference-openai-compatible#object)
- [system\_fingerprint](https://www.tensorzero.com/docs/gateway/api-reference/inference-openai-compatible#system-fingerprint)
- [usage](https://www.tensorzero.com/docs/gateway/api-reference/inference-openai-compatible#usage)
- [Examples](https://www.tensorzero.com/docs/gateway/api-reference/inference-openai-compatible#examples)
- [Chat Function with Structured System Prompt](https://www.tensorzero.com/docs/gateway/api-reference/inference-openai-compatible#chat-function-with-structured-system-prompt)
- [Chat Function with Dynamic Tool Use](https://www.tensorzero.com/docs/gateway/api-reference/inference-openai-compatible#chat-function-with-dynamic-tool-use)
- [Json Function with Dynamic Output Schema](https://www.tensorzero.com/docs/gateway/api-reference/inference-openai-compatible#json-function-with-dynamic-output-schema)

## [​](https://www.tensorzero.com/docs/gateway/api-reference/inference-openai-compatible\#post-/openai/v1/chat/completions)  `POST /openai/v1/chat/completions`

The `/openai/v1/chat/completions` endpoint allows TensorZero users to make TensorZero inferences with the OpenAI client.
The gateway translates the OpenAI request parameters into the arguments expected by the `inference` endpoint and calls the same underlying implementation.
This endpoint supports most of the features supported by the `inference` endpoint, but there are some limitations.
Most notably, this endpoint doesn’t support dynamic credentials, so they must be specified with a different method.

See the [API Reference for `POST /inference`](https://www.tensorzero.com/docs/gateway/api-reference/inference) for more details on inference with the native TensorZero API.

### [​](https://www.tensorzero.com/docs/gateway/api-reference/inference-openai-compatible\#request)  Request

The OpenAI-compatible inference endpoints translate the OpenAI request parameters into the arguments expected by the `inference` endpoint.TensorZero-specific parameters are prefixed with `tensorzero::` (e.g. `tensorzero::episode_id`).
These fields should be provided as extra body parameters in the request body.

The gateway will use the credentials specified in the `tensorzero.toml` file.
In most cases, these credentials will be environment variables available to the TensorZero gateway — _not_ your OpenAI client.API keys sent from the OpenAI client will be ignored.

#### [​](https://www.tensorzero.com/docs/gateway/api-reference/inference-openai-compatible\#tensorzero::cache-options)  `tensorzero::cache_options`

- **Type:** object
- **Required:** no

Controls caching behavior for inference requests.
This object accepts two fields:

- `enabled` (string): The cache mode. Can be one of:

  - `"write_only"` (default): Only write to cache but don’t serve cached responses
  - `"read_only"`: Only read from cache but don’t write new entries
  - `"on"`: Both read from and write to cache
  - `"off"`: Disable caching completely
- `max_age_s` (integer or null): Maximum age in seconds for cache entries to be considered valid when reading from cache. Does not set a TTL for cache expiration. Default is `null` (no age limit).

When using the OpenAI client libraries, pass this parameter via `extra_body`.See the [Inference Caching](https://www.tensorzero.com/docs/gateway/guides/inference-caching) guide for more details.

#### [​](https://www.tensorzero.com/docs/gateway/api-reference/inference-openai-compatible\#tensorzero::credentials)  `tensorzero::credentials`

- **Type:** object (a map from dynamic credential names to API keys)
- **Required:** no (default: no credentials)

Each model provider in your TensorZero configuration can be configured to accept credentials at inference time by using the `dynamic` location (e.g. `dynamic::my_dynamic_api_key_name`).
See the [configuration reference](https://www.tensorzero.com/docs/gateway/configuration-reference#modelsmodel_nameprovidersprovider_name) for more details.
The gateway expects the credentials to be provided in the `credentials` field of the request body as specified below.
The gateway will return a 400 error if the credentials are not provided and the model provider has been configured with dynamic credentials.

Example

Copy

```
[models.my_model_name.providers.my_provider_name]
# ...
# Note: the name of the credential field (e.g. `api_key_location`) depends on the provider type
api_key_location = "dynamic::my_dynamic_api_key_name"
# ...
```

Copy

```
{
  // ...
  "tensorzero::credentials": {
    // ...
    "my_dynamic_api_key_name": "sk-..."
    // ...
  }
  // ...
}
```

#### [​](https://www.tensorzero.com/docs/gateway/api-reference/inference-openai-compatible\#tensorzero::deny-unknown-fields)  `tensorzero::deny_unknown_fields`

- **Type:** boolean
- **Required:** no (default: `false`)

If `true`, the gateway will return an error if the request contains any unknown or unrecognized fields.
By default, unknown fields are ignored with a warning logged.This field does not affect the `tensorzero::extra_body` field, only unknown fields at the root of the request body.This field should be provided as an extra body parameter in the request body.

Copy

```
response = oai.chat.completions.create(
    model="tensorzero::model_name::openai::gpt-5-mini",
    messages=[\
        {\
            "role": "user",\
            "content": "Tell me a fun fact.",\
        }\
    ],
    extra_body={
        "tensorzero::deny_unknown_fields": True,
    },
    ultrathink=True,  # made-up parameter → `deny_unknown_fields` would reject this request
)
```

#### [​](https://www.tensorzero.com/docs/gateway/api-reference/inference-openai-compatible\#tensorzero::dryrun)  `tensorzero::dryrun`

- **Type:** boolean
- **Required:** no

If `true`, the inference request will be executed but won’t be stored to the database.
The gateway will still call the downstream model providers.This field is primarily for debugging and testing, and you should generally not use it in production.This field should be provided as an extra body parameter in the request body.

#### [​](https://www.tensorzero.com/docs/gateway/api-reference/inference-openai-compatible\#tensorzero::episode-id)  `tensorzero::episode_id`

- **Type:** UUID
- **Required:** no

The ID of an existing episode to associate the inference with.
If null, the gateway will generate a new episode ID and return it in the response.
See [Episodes](https://www.tensorzero.com/docs/gateway/guides/episodes) for more information.This field should be provided as an extra body parameter in the request body.

#### [​](https://www.tensorzero.com/docs/gateway/api-reference/inference-openai-compatible\#tensorzero::extra-body)  `tensorzero::extra_body`

- **Type:** array of objects (see below)
- **Required:** no

The `tensorzero::extra_body` field allows you to modify the request body that TensorZero sends to a model provider.
This advanced feature is an “escape hatch” that lets you use provider-specific functionality that TensorZero hasn’t implemented yet.

The OpenAI SDKs generally also support such functionality.If you use the OpenAI SDK’s `extra_body` field, it will override the request from the client to the gateway.
If you use `tensorzero::extra_body`, it will override the request from the gateway to the model provider.

Each object in the array must have two or three fields:

- `pointer`: A [JSON Pointer](https://datatracker.ietf.org/doc/html/rfc6901) string specifying where to modify the request body
- One of the following:
  - `value`: The value to insert at that location; it can be of any type including nested types
  - `delete = true`: Deletes the field at the specified location, if present.
- Optional: If one of the following is specified, the modification will only be applied to the specified variant, model, or model provider. If neither is specified, the modification applies to all model inferences.
  - `variant_name`
  - `model_name`
  - `model_name` and `provider_name`

You can also set `extra_body` in the configuration file.
The values provided at inference-time take priority over the values in the configuration file.

Example

If TensorZero would normally send this request body to the provider…

Copy

```
{
  "project": "tensorzero",
  "safety_checks": {
    "no_internet": false,
    "no_agi": true
  }
}
```

…then the following `extra_body` in the inference request…

Copy

```
{
  // ...
  "tensorzero::extra_body": [\
    {\
      "variant_name": "my_variant", // or "model_name": "my_model", "provider_name": "my_provider"\
      "pointer": "/agi",\
      "value": true\
    },\
    {\
      // No `variant_name` or `model_name`/`provider_name` specified, so it applies to all variants and providers\
      "pointer": "/safety_checks/no_agi",\
      "value": {\
        "bypass": "on"\
      }\
    }\
  ]
}
```

…overrides the request body to:

Copy

```
{
  "agi": true,
  "project": "tensorzero",
  "safety_checks": {
    "no_internet": false,
    "no_agi": {
      "bypass": "on"
    }
  }
}
```

#### [​](https://www.tensorzero.com/docs/gateway/api-reference/inference-openai-compatible\#tensorzero::extra-headers)  `tensorzero::extra_headers`

- **Type:** array of objects (see below)
- **Required:** no

The `tensorzero::extra_headers` field allows you to modify the request headers that TensorZero sends to a model provider.
This advanced feature is an “escape hatch” that lets you use provider-specific functionality that TensorZero hasn’t implemented yet.

The OpenAI SDKs generally also support such functionality.If you use the OpenAI SDK’s `extra_headers` field, it will override the request from the client to the gateway.
If you use `tensorzero::extra_headers`, it will override the request from the gateway to the model provider.

Each object in the array must have two or three fields:

- `name`: The name of the header to modify
- `value`: The value to set the header to
- Optional: If one of the following is specified, the modification will only be applied to the specified variant, model, or model provider. If neither is specified, the modification applies to all model inferences.
  - `variant_name`
  - `model_name`
  - `model_name` and `provider_name`

You can also set `extra_headers` in the configuration file.
The values provided at inference-time take priority over the values in the configuration file.

Example

If TensorZero would normally send the following request headers to the provider…

Copy

```
Safety-Checks: on
```

…then the following `extra_headers`…

Copy

```
{
  "extra_headers": [\
    {\
      "variant_name": "my_variant", // or "model_name": "my_model", "provider_name": "my_provider"\
      "name": "Safety-Checks",\
      "value": "off"\
    },\
    {\
      // No `variant_name` or `model_name`/`provider_name` specified, so it applies to all variants and providers\
      "name": "Intelligence-Level",\
      "value": "AGI"\
    }\
  ]
}
```

…overrides the request headers so that `Safety-Checks` is set to `off` only for `my_variant`, while `Intelligence-Level: AGI` is applied globally to all variants and providers:

Copy

```
Safety-Checks: off
Intelligence-Level: AGI
```

#### [​](https://www.tensorzero.com/docs/gateway/api-reference/inference-openai-compatible\#tensorzero::params)  `tensorzero::params`

- **Type:** object
- **Required:** no

Allows you to override inference parameters dynamically at request time.This field accepts an object with a `chat_completion` field containing any of the following parameters:

- `frequency_penalty` (float): Penalizes tokens based on their frequency
- `json_mode` (object): Controls JSON output formatting
- `max_tokens` (integer): Maximum number of tokens to generate
- `presence_penalty` (float): Penalizes tokens based on their presence
- `reasoning_effort` (string): Effort level for reasoning models
- `seed` (integer): Random seed for deterministic outputs
- `service_tier` (string): Service tier for the request
- `stop_sequences` (list of strings): Sequences that stop generation
- `temperature` (float): Controls randomness in the output
- `thinking_budget_tokens` (integer): Token budget for thinking/reasoning
- `top_p` (float): Nucleus sampling parameter
- `verbosity` (string): Output verbosity level

When using the OpenAI-compatible endpoint, values specified in `tensorzero::params` take precedence over parameters provided directly in the request body (e.g., top-level `temperature`, `max_tokens`) or inferred from other fields (e.g., `json_mode` inferred from `response_format`).

Example

Copy

```
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:3000/openai/v1",
    api_key="your_api_key",
)

response = client.chat.completions.create(
    model="tensorzero::function_name::my_function",
    messages=[\
        {"role": "user", "content": "Explain quantum computing"}\
    ],
    extra_body={
        "tensorzero::params": {
            "chat_completion": {
                "temperature": 0.7,
                "max_tokens": 500,
                "reasoning_effort": "high"
            }
        }
    }
)
```

#### [​](https://www.tensorzero.com/docs/gateway/api-reference/inference-openai-compatible\#tensorzero::provider-tools)  `tensorzero::provider_tools`

- **Type:** array of objects
- **Required:** no (default: `[]`)

A list of provider-specific built-in tools that can be used by the model during inference.
These are tools that run server-side on the provider’s infrastructure, such as OpenAI’s web search tool.Each object in the array has the following fields:

- `scope` (object, optional): Limits which model/provider combination can use this tool. If omitted, the tool is available to all compatible providers.

  - `model_name` (string): The model name as defined in your configuration
  - `model_provider_name` (string): The provider name for that model
- `tool` (object, required): The provider-specific tool configuration as defined by the provider’s API

When using OpenAI client libraries, pass this parameter via `extra_body`.This field allows for dynamic provider tool configuration at runtime.
You should prefer to define provider tools in the configuration file if possible (see [Configuration Reference](https://www.tensorzero.com/docs/gateway/configuration-reference#provider_tools)).
Only use this field if dynamic provider tool configuration is necessary for your use case.

Example: OpenAI Web Search (Unscoped)

Copy

```
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:3000/openai/v1",
    api_key="your_api_key",
)

response = client.chat.completions.create(
    model="tensorzero::function_name::my_function",
    messages=[\
        {"role": "user", "content": "What were the latest developments in AI this week?"}\
    ],
    extra_body={
        "tensorzero::provider_tools": [\
            {\
                "tool": {\
                    "type": "web_search"\
                }\
            }\
        ]
    }
)
```

This makes the web search tool available to all compatible providers configured for the function.

Example: OpenAI Web Search (Scoped)

Copy

```
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:3000/openai/v1",
    api_key="your_api_key",
)

response = client.chat.completions.create(
    model="tensorzero::function_name::my_function",
    messages=[\
        {"role": "user", "content": "What were the latest developments in AI this week?"}\
    ],
    extra_body={
        "tensorzero::provider_tools": [\
            {\
                "scope": {\
                    "model_name": "gpt-5-mini",\
                    "model_provider_name": "openai"\
                },\
                "tool": {\
                    "type": "web_search"\
                }\
            }\
        ]
    }
)
```

This makes the web search tool available only to the OpenAI provider for the `gpt-5-mini` model.

#### [​](https://www.tensorzero.com/docs/gateway/api-reference/inference-openai-compatible\#tensorzero::tags)  `tensorzero::tags`

- **Type:** flat JSON object with string keys and values
- **Required:** no

User-provided tags to associate with the inference.For example, `{"user_id": "123"}` or `{"author": "Alice"}`.

#### [​](https://www.tensorzero.com/docs/gateway/api-reference/inference-openai-compatible\#frequency-penalty)  `frequency_penalty`

- **Type:** float
- **Required:** no (default: `null`)

Penalizes new tokens based on their frequency in the text so far if positive, encourages them if negative.
Overrides the `frequency_penalty` setting for any chat completion variants being used.

#### [​](https://www.tensorzero.com/docs/gateway/api-reference/inference-openai-compatible\#max-completion-tokens)  `max_completion_tokens`

- **Type:** integer
- **Required:** no (default: `null`)

Limits the number of tokens that can be generated by the model in a chat completion variant.
If both this and `max_tokens` are set, the smaller value is used.

#### [​](https://www.tensorzero.com/docs/gateway/api-reference/inference-openai-compatible\#max-tokens)  `max_tokens`

- **Type:** integer
- **Required:** no (default: `null`)

Limits the number of tokens that can be generated by the model in a chat completion variant.
If both this and `max_completion_tokens` are set, the smaller value is used.

#### [​](https://www.tensorzero.com/docs/gateway/api-reference/inference-openai-compatible\#messages)  `messages`

- **Type:** list
- **Required:** yes

A list of messages to provide to the model.Each message is an object with the following fields:

- `role` (required): The role of the message sender in an OpenAI message (`assistant`, `system`, `tool`, or `user`).
- `content` (required for `user` and `system` messages and optional for `assistant` and `tool` messages): The content of the message.
The content must be either a string or an array of content blocks (see below).
- `tool_calls` (optional for `assistant` messages, otherwise disallowed): A list of tool calls. Each tool call is an object with the following fields:

  - `id`: A unique identifier for the tool call
  - `type`: The type of tool being called (currently only `"function"` is supported)
  - `function`: An object containing:

    - `name`: The name of the function to call
    - `arguments`: A JSON string containing the function arguments
- `tool_call_id` (required for `tool` messages, otherwise disallowed): The ID of the tool call to associate with the message. This should be one that was originally returned by the gateway in a tool call `id` field.

A content block is an object that can have type `text`, `image_url`, or TensorZero-specific types.If the content block has type `text`, it must have either of the following additional fields:

- `text`: The text for the content block.
- `tensorzero::arguments`: A JSON object containing the function arguments for TensorZero functions with templates and schemas (see [Create a prompt template](https://www.tensorzero.com/docs/gateway/create-a-prompt-template) for details).

If a content block has type `image_url`, it must have the following additional fields:

- `"image_url"`: A JSON object with the following fields:

  - `url`: The URL for a remote image (e.g. `"https://example.com/image.png"`) or base64-encoded data for an embedded image (e.g. `"data:image/png;base64,..."`).
  - `detail` (optional): Controls the fidelity of image processing. Only applies to image files; ignored for other file types. Can be `low`, `high`, or `auto`. Affects token consumption and image quality.

The TensorZero-specific content block types are:

- `tensorzero::raw_text`: Bypasses templates and schemas, sending text directly to the model. Useful for testing prompts or dynamic injection without configuration changes. Must have a `value` field containing the text.
- `tensorzero::template`: Explicitly specify a template to use. Must have `name` and `arguments` fields.

#### [​](https://www.tensorzero.com/docs/gateway/api-reference/inference-openai-compatible\#model)  `model`

- **Type:** string
- **Required:** yes

The name of the TensorZero function or model being called, with the appropriate prefix.

|     |     |
| --- | --- |
| **To call…** | **Use this format…** |
| A function defined as `[functions.my_function]` in your<br>`tensorzero.toml` configuration file | `tensorzero::function_name::my_function` |
| A model defined as `[models.my_model]` in your `tensorzero.toml`<br>configuration file | `tensorzero::model_name::my_model` |
| A model offered by a model provider, without defining it in your<br>`tensorzero.toml` configuration file (if supported, see below) | `tensorzero::model_name::{provider_type}::{model_name}` |

The following model providers support short-hand model names: `anthropic`, `deepseek`, `fireworks`, `gcp_vertex_anthropic`, `gcp_vertex_gemini`, `google_ai_studio_gemini`, `groq`, `hyperbolic`, `mistral`, `openai`, `openrouter`, `together`, and `xai`.

For example, if you have the following configuration:

tensorzero.toml

Copy

```
[models.gpt-4o]
routing = ["openai", "azure"]

[models.gpt-4o.providers.openai]
# ...

[models.gpt-4o.providers.azure]
# ...

[functions.extract-data]
# ...
```

Then:

- `tensorzero::function_name::extract-data` calls the `extract-data` function defined above.
- `tensorzero::model_name::gpt-4o` calls the `gpt-4o` model in your configuration, which supports fallback from `openai` to `azure`. See [Retries & Fallbacks](https://www.tensorzero.com/docs/gateway/guides/retries-fallbacks) for details.
- `tensorzero::model_name::openai::gpt-4o` calls the OpenAI API directly for the `gpt-4o` model, ignoring the `gpt-4o` model defined above.

Be careful about the different prefixes: `tensorzero::model_name::gpt-4o` will use the `[models.gpt-4o]` model defined in the `tensorzero.toml` file, whereas `tensorzero::model_name::openai::gpt-4o` will call the OpenAI API directly for the `gpt-4o` model.

#### [​](https://www.tensorzero.com/docs/gateway/api-reference/inference-openai-compatible\#parallel-tool-calls)  `parallel_tool_calls`

- **Type:** boolean
- **Required:** no (default: `null`)

Overrides the `parallel_tool_calls` setting for the function being called.

#### [​](https://www.tensorzero.com/docs/gateway/api-reference/inference-openai-compatible\#presence-penalty)  `presence_penalty`

- **Type:** float
- **Required:** no (default: `null`)

Penalizes new tokens based on whether they appear in the text so far if positive, encourages them if negative.
Overrides the `presence_penalty` setting for any chat completion variants being used.

#### [​](https://www.tensorzero.com/docs/gateway/api-reference/inference-openai-compatible\#response-format)  `response_format`

- **Type:** either a string or an object
- **Required:** no (default: `null`)

Options here are `"text"`, `"json_object"`, and `"{"type": "json_schema", "schema": ...}"`, where the schema field contains a valid JSON schema.
This field is not actually respected except for the `"json_schema"` variant, in which the `schema` field can be used to dynamically set the output schema for a `json` function.

#### [​](https://www.tensorzero.com/docs/gateway/api-reference/inference-openai-compatible\#seed)  `seed`

- **Type:** integer
- **Required:** no (default: `null`)

Overrides the `seed` setting for any chat completion variants being used.

#### [​](https://www.tensorzero.com/docs/gateway/api-reference/inference-openai-compatible\#stop-sequences)  `stop_sequences`

- **Type:** list of strings
- **Required:** no (default: `null`)

Overrides the `stop_sequences` setting for any chat completion variants being used.

#### [​](https://www.tensorzero.com/docs/gateway/api-reference/inference-openai-compatible\#stream)  `stream`

- **Type:** boolean
- **Required:** no (default: `false`)

If true, the gateway will stream the response to the client in an OpenAI-compatible format.

#### [​](https://www.tensorzero.com/docs/gateway/api-reference/inference-openai-compatible\#stream-options)  `stream_options`

- **Type:** object with field `"include_usage"`
- **Required:** no (default: `null`)

If `"include_usage"` is `true`, the gateway will include usage information in the response.

Example

If the following `stream_options` is provided…

Copy

```
{
  ...
  "stream_options": {
    "include_usage": true
  }
  ...
}
```

…then the gateway will include usage information in the response.

Copy

```
{
  ...
  "usage": {
    "prompt_tokens": 123,
    "completion_tokens": 456,
    "total_tokens": 579
  }
  ...
```

#### [​](https://www.tensorzero.com/docs/gateway/api-reference/inference-openai-compatible\#temperature)  `temperature`

- **Type:** float
- **Required:** no (default: `null`)

Overrides the `temperature` setting for any chat completion variants being used.

#### [​](https://www.tensorzero.com/docs/gateway/api-reference/inference-openai-compatible\#tools)  `tools`

- **Type:** list of `tool` objects (see below)
- **Required:** no (default: `null`)

Allows the user to dynamically specify tools at inference time in addition to those that are specified in the configuration.Each `tool` object has the following structure:

- **`type`**: Must be `"function"`
- **`function`**: An object containing:

  - **`name`**: The name of the function (string, required)
  - **`description`**: A description of what the function does (string, optional)
  - **`parameters`**: A JSON Schema object describing the function’s parameters (required)
  - **`strict`**: Whether to enforce strict schema validation (boolean, defaults to false)

#### [​](https://www.tensorzero.com/docs/gateway/api-reference/inference-openai-compatible\#tool-choice)  `tool_choice`

- **Type:** string or object
- **Required:** no (default: `"none"` if no tools are present, `"auto"` if tools are present)

Controls which (if any) tool is called by the model by overriding the value in configuration. Supported values:

- `"none"`: The model will not call any tool and instead generates a message
- `"auto"`: The model can pick between generating a message or calling one or more tools
- `"required"`: The model must call one or more tools
- `{"type": "function", "function": {"name": "my_function"}}`: Forces the model to call the specified tool
- `{"type": "allowed_tools", "allowed_tools": {"tools": [...], "mode": "auto"|"required"}}`: Restricts which tools can be called

#### [​](https://www.tensorzero.com/docs/gateway/api-reference/inference-openai-compatible\#top-p)  `top_p`

- **Type:** float
- **Required:** no (default: `null`)

Overrides the `top_p` setting for any chat completion variants being used.

#### [​](https://www.tensorzero.com/docs/gateway/api-reference/inference-openai-compatible\#tensorzero::variant-name)  `tensorzero::variant_name`

- **Type:** string
- **Required:** no

If set, pins the inference request to a particular variant (not recommended).You should generally not set this field, and instead let the TensorZero gateway assign a variant.
This field is primarily used for testing or debugging purposes.This field should be provided as an extra body parameter in the request body.

### [​](https://www.tensorzero.com/docs/gateway/api-reference/inference-openai-compatible\#response)  Response

- Regular

- Streaming


In regular (non-streaming) mode, the response is a JSON object with the following fields:

#### [​](https://www.tensorzero.com/docs/gateway/api-reference/inference-openai-compatible\#choices)  `choices`

- **Type:** list of `choice` objects, where each choice contains:

  - **`index`**: A zero-based index indicating the choice’s position in the list (integer)
  - **`finish_reason`**: Always `"stop"`.
  - **`message`**: An object containing:

    - **`content`**: The message content (string, optional)
    - **`tool_calls`**: List of tool calls made by the model (optional). The format is the same as in the request.
    - **`role`**: The role of the message sender (always `"assistant"`).

The OpenAI-compatible inference endpoint can’t handle unknown content blocks in the response.
If the model provider returns an unknown content block, the gateway will drop the content block from the response and log a warning.If you need to access unknown content blocks, use the native TensorZero API.
See the [Inference API Reference](https://www.tensorzero.com/docs/gateway/api-reference/inference) for details.

#### [​](https://www.tensorzero.com/docs/gateway/api-reference/inference-openai-compatible\#created)  `created`

- **Type:** integer

The Unix timestamp (in seconds) of when the inference was created.

#### [​](https://www.tensorzero.com/docs/gateway/api-reference/inference-openai-compatible\#episode-id)  `episode_id`

- **Type:** UUID

The ID of the episode that the inference was created for.

#### [​](https://www.tensorzero.com/docs/gateway/api-reference/inference-openai-compatible\#id)  `id`

- **Type:** UUID

The inference ID.

#### [​](https://www.tensorzero.com/docs/gateway/api-reference/inference-openai-compatible\#model-2)  `model`

- **Type:** string

The name of the variant that was actually used for the inference.

#### [​](https://www.tensorzero.com/docs/gateway/api-reference/inference-openai-compatible\#object)  `object`

- **Type:** string

The type of the inference object (always `"chat.completion"`).

#### [​](https://www.tensorzero.com/docs/gateway/api-reference/inference-openai-compatible\#system-fingerprint)  `system_fingerprint`

- **Type:** string

Always ""

#### [​](https://www.tensorzero.com/docs/gateway/api-reference/inference-openai-compatible\#usage)  `usage`

- **Type:** object

Contains token usage information for the request and response, with the following fields:

- **`prompt_tokens`**: Number of tokens in the prompt (integer)
- **`completion_tokens`**: Number of tokens in the completion (integer)
- **`total_tokens`**: Total number of tokens used (integer)

### [​](https://www.tensorzero.com/docs/gateway/api-reference/inference-openai-compatible\#examples)  Examples

#### [​](https://www.tensorzero.com/docs/gateway/api-reference/inference-openai-compatible\#chat-function-with-structured-system-prompt)  Chat Function with Structured System Prompt

Chat Function with Structured System Prompt

##### Configuration

Copy

```
// tensorzero.toml
# ...
[functions.draft_email]
type = "chat"
system_schema = "functions/draft_email/system_schema.json"
# ...
```

Copy

```
// functions/draft_email/system_schema.json
{
  "type": "object",
  "properties": {
    "assistant_name": { "type": "string" }
  }
}
```

##### Request

- Python

- HTTP


POST /inference

Copy

```
from openai import AsyncOpenAI

async with AsyncOpenAI(
    base_url="http://localhost:3000/openai/v1"
) as client:
    result = await client.chat.completions.create(
        # there already was an episode_id from an earlier inference
        extra_body={"tensorzero::episode_id": str(episode_id)},
        messages=[\
            {\
                "role": "system",\
                "content": [{"assistant_name": "Alfred Pennyworth"}]\
                # NOTE: the JSON is in an array here so that a structured system message can be sent\
            },\
            {\
                "role": "user",\
                "content": "I need to write an email to Gabriel explaining..."\
            }\
        ],
        model="tensorzero::function_name::draft_email",
        temperature=0.4,
        # Optional: stream=True
    )
```

##### Response

- Regular

- Streaming


POST /inference

Copy

```
{
  "id": "00000000-0000-0000-0000-000000000000",
  "episode_id": "11111111-1111-1111-1111-111111111111",
  "model": "email_draft_variant",
  "choices": [\
    {\
      "index": 0,\
      "finish_reason": "stop",\
      "message": {\
        "content": "Hi Gabriel,\n\nI noticed...",\
        "role": "assistant"\
      }\
    }\
  ],
  "usage": {
    "prompt_tokens": 100,
    "completion_tokens": 100,
    "total_tokens": 200
  }
}
```

#### [​](https://www.tensorzero.com/docs/gateway/api-reference/inference-openai-compatible\#chat-function-with-dynamic-tool-use)  Chat Function with Dynamic Tool Use

Chat Function with Dynamic Tool Use

##### Configuration

Copy

```
// tensorzero.toml
# ...

[functions.weather_bot]
type = "chat"
# Note: no `tools = ["get_temperature"]` field in configuration

# ...
```

##### Request

- Python

- HTTP


POST /inference

Copy

```
from openai import AsyncOpenAI

async with AsyncOpenAI(
    base_url="http://localhost:3000/openai/v1"
) as client:
    result = await client.chat.completions.create(
        model="tensorzero::function_name::weather_bot",
        input={
            "messages": [\
                {\
                    "role": "user",\
                    "content": "What is the weather like in Tokyo?"\
                }\
            ]
        },
        tools=[\
            {\
              "type": "function",\
              "function": {\
                  "name": "get_temperature",\
                  "description": "Get the current temperature in a given location",\
                  "parameters": {\
                    "$schema": "http://json-schema.org/draft-07/schema#",\
                    "type": "object",\
                    "properties": {\
                        "location": {\
                            "type": "string",\
                            "description": "The location to get the temperature for (e.g. \"New York\")"\
                        },\
                        "units": {\
                            "type": "string",\
                            "description": "The units to get the temperature in (must be \"fahrenheit\" or \"celsius\")",\
                            "enum": ["fahrenheit", "celsius"]\
                        }\
                    },\
                    "required": ["location"],\
                    "additionalProperties": false\
                }\
              }\
            }\
        ],
        # optional: stream=True,
    )
```

##### Response

- Regular

- Streaming


POST /inference

Copy

```
{
  "id": "00000000-0000-0000-0000-000000000000",
  "episode_id": "11111111-1111-1111-1111-111111111111",
  "model": "weather_bot_variant",
  "choices": [\
    {\
      "index": 0,\
      "finish_reason": "stop",\
      "message": {\
        "content": null,\
        "tool_calls": [\
          {\
            "id": "123456789",\
            "type": "function",\
            "function": {\
              "name": "get_temperature",\
              "arguments": "{\"location\": \"Tokyo\", \"units\": \"celsius\"}"\
            }\
          }\
        ],\
        "role": "assistant"\
      }\
    }\
  ],
  "usage": {
    "prompt_tokens": 100,
    "completion_tokens": 100,
    "total_tokens": 200
  }
}
```

#### [​](https://www.tensorzero.com/docs/gateway/api-reference/inference-openai-compatible\#json-function-with-dynamic-output-schema)  Json Function with Dynamic Output Schema

JSON Function with Dynamic Output Schema

##### Configuration

Copy

```
// tensorzero.toml
# ...
[functions.extract_email]
type = "json"
output_schema = "output_schema.json"
# ...
```

Copy

```
// output_schema.json
{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "type": "object",
  "properties": {
    "email": {
      "type": "string"
    }
  },
  "required": ["email"]
}
```

##### Request

- Python

- HTTP


POST /inference

Copy

```
from openai import AsyncOpenAI

dynamic_output_schema = {
  "$schema": "http://json-schema.org/draft-07/schema#",
  "type": "object",
  "properties": {
    "email": { "type": "string" },
    "domain": { "type": "string" }
  },
  "required": ["email", "domain"]
}

async with AsyncOpenAI(
    base_url="http://localhost:3000/openai/v1"
) as client:
    result = await client.chat.completions.create(
        model="tensorzero::function_name::extract_email",
        input={
            "system": "You are an AI assistant...",
            "messages": [\
                {\
                    "role": "user",\
                    "content": "...blah blah blah hello@tensorzero.com blah blah blah..."\
                }\
            ]
        }
        # Override the output schema using the `response_format` field
        response_format={"type": "json_schema", "schema": dynamic_output_schema}
        # optional: stream=True,
    )
```

##### Response

- Regular

- Streaming


POST /inference

Copy

```
{
  "id": "00000000-0000-0000-0000-000000000000",
  "episode_id": "11111111-1111-1111-1111-111111111111",
  "model": "extract_email_variant",
  "choices": [\
    {\
      "index": 0,\
      "finish_reason": "stop",\
      "message": {\
        "content": "{\"email\": \"hello@tensorzero.com\", \"domain\": \"tensorzero.com\"}"\
      }\
    }\
  ],
  "usage": {
    "prompt_tokens": 100,
    "completion_tokens": 100,
    "total_tokens": 200
  }
}
```

[Inference](https://www.tensorzero.com/docs/gateway/api-reference/inference) [Feedback](https://www.tensorzero.com/docs/gateway/api-reference/feedback)

Ctrl+I

Assistant

Responses are generated using AI and may contain mistakes.

## Gateway Performance Benchmarks
[Skip to main content](https://www.tensorzero.com/docs/gateway/benchmarks#content-area)

[TensorZero Docs home page![light logo](https://mintcdn.com/tensorzero/afFfERJmQ8YiZpCj/logo/light.svg?fit=max&auto=format&n=afFfERJmQ8YiZpCj&q=85&s=9396731a8a92cee7bd293dee05d71dac)![dark logo](https://mintcdn.com/tensorzero/afFfERJmQ8YiZpCj/logo/dark.svg?fit=max&auto=format&n=afFfERJmQ8YiZpCj&q=85&s=8d61c59531dbcdcace7fcdfc524bd4d2)](https://www.tensorzero.com/)

Search...

Ctrl K

Search...

Navigation

Gateway

Benchmarks

[Guides](https://www.tensorzero.com/docs) [Integrations](https://www.tensorzero.com/docs/integrations/model-providers)

On this page

- [TensorZero Gateway vs. LiteLLM](https://www.tensorzero.com/docs/gateway/benchmarks#tensorzero-gateway-vs-litellm)
- [Latency Comparison](https://www.tensorzero.com/docs/gateway/benchmarks#latency-comparison)

The TensorZero Gateway was built from the ground up with performance in mind.It’s written in Rust and designed to handle extreme concurrency with sub-millisecond overhead.

See [“Optimize latency and throughput” guide](https://www.tensorzero.com/docs/deployment/optimize-latency-and-throughput) for more details on maximizing performance in production settings.

## [​](https://www.tensorzero.com/docs/gateway/benchmarks\#tensorzero-gateway-vs-litellm)  TensorZero Gateway vs. LiteLLM

- **TensorZero achieves sub-millisecond latency overhead even at 10,000 QPS.**
- **LiteLLM degrades at hundreds of QPS and fails entirely at 1,000 QPS.**

We benchmarked the TensorZero Gateway against the popular LiteLLM Proxy (LiteLLM Gateway).In a `c7i.xlarge` instance on AWS (4 vCPUs, 8 GB RAM), LiteLLM fails when concurrency reaches 1,000 QPS with the vast majority of requests timing out.
TensorZero Gateway handles 10,000 QPS in the same instance with 100% success rate and sub-millisecond latencies.Even at low loads where LiteLLM is stable (100 QPS), TensorZero at 10,000 QPS achieves significantly lower latencies.
Building in Rust (TensorZero) led to consistent sub-millisecond latency overhead under extreme load, whereas Python (LiteLLM) becomes a bottleneck even at moderate loads.

### [​](https://www.tensorzero.com/docs/gateway/benchmarks\#latency-comparison)  Latency Comparison

| Latency | LiteLLM Proxy <br> (100 QPS) | LiteLLM Proxy <br> (500 QPS) | LiteLLM Proxy <br> (1,000 QPS) | TensorZero Gateway <br> (10,000 QPS) |
| --- | --- | --- | --- | --- |
| Mean | 4.91ms | 7.45ms | Failure | 0.37ms |
| 50% | 4.83ms | 5.81ms | Failure | 0.35ms |
| 90% | 5.26ms | 10.02ms | Failure | 0.50ms |
| 95% | 5.41ms | 13.40ms | Failure | 0.58ms |
| 99% | 5.87ms | 39.69ms | Failure | 0.94ms |

At 1,000 QPS, LiteLLM fails entirely with the vast majority of requests timing out, while TensorZero continues to operate smoothly even at 10x that load.**Technical Notes:**

- We use a `c7i.xlarge` instance on AWS (4 vCPUs, 8 GB RAM) running Ubuntu 24.04.2 LTS.
- We use a mock OpenAI inference provider for both benchmarks.
- The load generator, both gateways, and the mock inference provider all run on the same instance.
- We configured `observability.enabled = false` (i.e. disabled logging inferences to ClickHouse) in the TensorZero Gateway to make the scenarios comparable. (Even then, the observability features run asynchronously in the background, so they wouldn’t materially affect latency given a powerful enough ClickHouse deployment.)
- The most recent benchmark run was conducted on July 30, 2025. It used TensorZero `2025.5.7` and LiteLLM `1.74.9`.

Read more about the technical details and reproduction instructions [here](https://github.com/tensorzero/tensorzero/tree/main/gateway/benchmarks).

[Tool Use (Function Calling)](https://www.tensorzero.com/docs/gateway/guides/tool-use) [Clients](https://www.tensorzero.com/docs/gateway/clients)

Ctrl+I

Assistant

Responses are generated using AI and may contain mistakes.

## Call Any LLM
[Skip to main content](https://www.tensorzero.com/docs/gateway/call-any-llm#content-area)

[TensorZero Docs home page![light logo](https://mintcdn.com/tensorzero/afFfERJmQ8YiZpCj/logo/light.svg?fit=max&auto=format&n=afFfERJmQ8YiZpCj&q=85&s=9396731a8a92cee7bd293dee05d71dac)![dark logo](https://mintcdn.com/tensorzero/afFfERJmQ8YiZpCj/logo/dark.svg?fit=max&auto=format&n=afFfERJmQ8YiZpCj&q=85&s=8d61c59531dbcdcace7fcdfc524bd4d2)](https://www.tensorzero.com/)

Search...

Ctrl K

Search...

Navigation

Gateway

How to call any LLM

[Guides](https://www.tensorzero.com/docs) [Integrations](https://www.tensorzero.com/docs/integrations/model-providers)

This page shows how to:

- **Call any LLM with the same API.** TensorZero unifies every major LLM API (e.g. OpenAI) and inference server (e.g. Ollama).
- **Get started with a few lines of code.** Later, you can optionally add observability, automatic fallbacks, A/B testing, and much more.
- **Use any programming language.** You can use TensorZero with its Python SDK, any OpenAI SDK (Python, Node, Go, etc.), or its HTTP API.

We provide [complete code examples](https://github.com/tensorzero/tensorzero/tree/main/examples/docs/guides/gateway/call-any-llm) on GitHub.

- Python

- Python (OpenAI SDK)

- Node (OpenAI SDK)

- HTTP


The TensorZero Python SDK provides a unified API for calling any LLM.

1

Set up the credentials for your LLM provider

For example, if you’re using OpenAI, you can set the `OPENAI_API_KEY` environment variable with your API key.

Copy

```
export OPENAI_API_KEY="sk-..."
```

See the [Integrations](https://www.tensorzero.com/docs/integrations/model-providers) page to learn how to set up credentials for other LLM providers.

2

Install the TensorZero Python SDK

You can install the TensorZero SDK with a Python package manager like `pip`.

Copy

```
pip install tensorzero
```

3

Initialize the TensorZero Gateway

Let’s initialize the TensorZero Gateway.
For simplicity, we’ll use an embedded gateway without observability or custom configuration.

Copy

```
from tensorzero import TensorZeroGateway

t0 = TensorZeroGateway.build_embedded()
```

The TensorZero Python SDK includes a synchronous `TensorZeroGateway` client and an asynchronous `AsyncTensorZeroGateway` client.
Both options support running the gateway embedded in your application with `build_embedded` or connecting to a standalone gateway with `build_http`.
See [Clients](https://www.tensorzero.com/docs/gateway/clients) for more details.

4

Call the LLM

Copy

```
response = t0.inference(
    model_name="openai::gpt-5-mini",
    # or: model="anthropic::claude-sonnet-4-20250514"
    # or: Google, AWS, Azure, xAI, vLLM, Ollama, and many more
    input={
        "messages": [\
            {\
                "role": "user",\
                "content": "Tell me a fun fact.",\
            }\
        ]
    },
)
```

Sample Response

Copy

```
ChatInferenceResponse(
    inference_id=UUID('0198d339-be77-74e0-b522-e08ec12d3831'),
    episode_id=UUID('0198d339-be77-74e0-b522-e09f578f34d0'),
    variant_name='openai::gpt-5-mini',
    content=[\
        Text(\
            text='Fun fact: Botanically, bananas are berries but strawberries are not. \n\nA true berry develops from a single ovary and has seeds embedded in the flesh—bananas fit that definition. Strawberries are "aggregate accessory fruits": the tiny seeds on the outside are each from a separate ovary.',\
            arguments=None,\
            type='text'\
        )\
    ],
    usage=Usage(input_tokens=12, output_tokens=261),
    finish_reason=FinishReason.STOP,
    original_response=None
)
```

See the [Inference API Reference](https://www.tensorzero.com/docs/gateway/api-reference/inference) for more details on the request and response formats.

See [Configure models and providers](https://www.tensorzero.com/docs/gateway/configure-models-and-providers) to set up multiple providers with routing and fallbacks and [Configure functions and variants](https://www.tensorzero.com/docs/gateway/configure-functions-and-variants) to manage your LLM logic with experimentation and observability.

[Overview](https://www.tensorzero.com/docs/gateway) [Call the OpenAI Responses API](https://www.tensorzero.com/docs/gateway/call-the-openai-responses-api)

Ctrl+I

Assistant

Responses are generated using AI and may contain mistakes.

## OpenAI Responses API
[Skip to main content](https://www.tensorzero.com/docs/gateway/call-the-openai-responses-api#content-area)

[TensorZero Docs home page![light logo](https://mintcdn.com/tensorzero/afFfERJmQ8YiZpCj/logo/light.svg?fit=max&auto=format&n=afFfERJmQ8YiZpCj&q=85&s=9396731a8a92cee7bd293dee05d71dac)![dark logo](https://mintcdn.com/tensorzero/afFfERJmQ8YiZpCj/logo/dark.svg?fit=max&auto=format&n=afFfERJmQ8YiZpCj&q=85&s=8d61c59531dbcdcace7fcdfc524bd4d2)](https://www.tensorzero.com/)

Search...

Ctrl K

Search...

Navigation

Gateway

How to call the OpenAI Responses API

[Guides](https://www.tensorzero.com/docs) [Integrations](https://www.tensorzero.com/docs/integrations/model-providers)

On this page

- [Call the OpenAI Responses API](https://www.tensorzero.com/docs/gateway/call-the-openai-responses-api#call-the-openai-responses-api)

This page shows how to:

- **Use a unified API.** TensorZero provides the same chat completion format for the Responses API.
- **Access built-in tools.** Enable built-in tools from OpenAI like `web_search`.
- **Enable reasoning models.** Support models with extended thinking capabilities.

We provide [complete code examples](https://github.com/tensorzero/tensorzero/tree/main/examples/docs/guides/gateway/call-the-openai-responses-api) on GitHub.

## [​](https://www.tensorzero.com/docs/gateway/call-the-openai-responses-api\#call-the-openai-responses-api)  Call the OpenAI Responses API

- Python (TensorZero SDK)

- Python (OpenAI SDK)

- Node (OpenAI SDK)

- HTTP


The TensorZero Python SDK provides a unified API for calling OpenAI’s Responses API.

1

Set up your OpenAI API key

You can set the `OPENAI_API_KEY` environment variable with your API key.

Copy

```
export OPENAI_API_KEY="sk-..."
```

2

Install the TensorZero Python SDK

You can install the TensorZero SDK with a Python package manager like `pip`.

Copy

```
pip install tensorzero
```

3

Configure a model for the OpenAI Responses API

Create a configuration file with a model using `api_type = "responses"` and provider tools:

tensorzero.toml

Copy

```
[models.gpt-5-mini-responses-web-search]
routing = ["openai"]

[models.gpt-5-mini-responses-web-search.providers.openai]
type = "openai"
model_name = "gpt-5-mini"
api_type = "responses"
include_encrypted_reasoning = true
provider_tools = [{type = "web_search"}]  # built-in OpenAI web search tool
# Enable plain-text summaries of encrypted reasoning
extra_body = [\
    { pointer = "/reasoning", value = { effort = "low", summary = "auto" } }\
]
```

If you don’t need to customize the model configuration (e.g. `include_encrypted_reasoning`, `provider_tools`), you can use the short-hand model name `openai::responses::gpt-5-codex` to call it directly.

4

Deploy a standalone (HTTP) TensorZero Gateway

Let’s deploy a standalone TensorZero Gateway using Docker.
For simplicity, we’ll use the gateway with the configuration above.

Copy

```
docker run \
  -e OPENAI_API_KEY \
  -v $(pwd)/tensorzero.toml:/app/config/tensorzero.toml:ro \
  -p 3000:3000 \
  tensorzero/gateway \
  --config-file /app/config/tensorzero.toml
```

See the [TensorZero Gateway Deployment](https://www.tensorzero.com/docs/deployment/tensorzero-gateway) page for more details.

5

Initialize the TensorZero Gateway client

Let’s initialize the TensorZero Gateway client and point it to the gateway we just launched.

Copy

```
from tensorzero import TensorZeroGateway

t0 = TensorZeroGateway.build_http(gateway_url="http://localhost:3000")
```

The TensorZero Python SDK includes a synchronous `TensorZeroGateway` client and an asynchronous `AsyncTensorZeroGateway` client.
Both options support running the gateway embedded in your application with `build_embedded` or connecting to a standalone gateway with `build_http`.
See [Clients](https://www.tensorzero.com/docs/gateway/clients) for more details.

6

Call the LLM

OpenAI web search can take up to a minute to complete.

Copy

```
response = t0.inference(
    model_name="gpt-5-mini-responses-web-search",
    input={
        "messages": [\
            {\
                "role": "user",\
                "content": "What is the current population of Japan?",\
            }\
        ]
    },
    # Thought summaries are enabled in tensorzero.toml via extra_body
)
```

Sample Response

Copy

```
ChatInferenceResponse(
    inference_id=UUID('0199ff78-6246-7c12-b4b0-6e3a881cc6b9'),
    episode_id=UUID('0199ff78-6246-7c12-b4b0-6e4367f949b8'),
    variant_name='gpt-5-mini-responses-web-search',
    content=[\
        Thought(\
            text=None,\
            type='thought',\
            signature='gAAAAABo9...',\
            summary=[\
                ThoughtSummaryBlock(\
                    text="I need to search for Japan's current population data.",\
                    type='summary_text'\
                )\
            ],\
            _internal_provider_type='openai'\
        ),\
        UnknownContentBlock(\
            data={\
                'id': 'ws_05489a0b57dc84980168f59fda57d481969c3603df0d675348',\
                'type': 'web_search_call',\
                'status': 'completed',\
                'action': {\
                    'type': 'search',\
                    'query': 'Japan population 2025 October 2025 population estimate Statistics Bureau of Japan'\
                }\
            },\
            model_provider_name='tensorzero::model_name::gpt-5-mini-responses-web-search::provider_name::openai',\
            type='unknown'\
        ),\
        Thought(\
            text=None,\
            type='thought',\
            signature='gAAAAABo...',\
            _internal_provider_type=None\
        ),\
        UnknownContentBlock(\
            data={\
                'id': 'ws_05489a0b57dc84980168f59fdf9b988196b36756d639e2b015',\
                'type': 'web_search_call',\
                'status': 'completed',\
                'action': {\
                    'type': 'search',\
                    'query': "Ministry of Internal Affairs and Communications Japan population Oct 1 2024 'total population' 'Japan' 'population estimates' '2024' 'Oct. 1' '総人口' '令和6年' "\
                }\
            },\
            model_provider_name='tensorzero::model_name::gpt-5-mini-responses-web-search::provider_name::openai',\
            type='unknown'\
        ),\
        Thought(\
            text=None,\
            type='thought',\
            signature='gAAAAABo...',\
            _internal_provider_type=None\
        ),\
        UnknownContentBlock(\
            data={\
                'id': 'ws_05489a0b57dc84980168f59fe1a388819684971acfdaf4cd44',\
                'type': 'web_search_call',\
                'status': 'completed',\
                'action': {\
                    'type': 'search',\
                    'query': "Ministry of Internal Affairs and Communications population Japan Oct 1 2024 total population 'Oct. 1, 2024' 'population' 'Japan' 'MIC' 'population estimates' '2024' '総人口' "\
                }\
            },\
            model_provider_name='tensorzero::model_name::gpt-5-mini-responses-web-search::provider_name::openai',\
            type='unknown'\
        ),\
        Thought(\
            text=None,\
            type='thought',\
            signature='gAAAAABo...',\
            _internal_provider_type=None\
        ),\
        UnknownContentBlock(\
            data={\
                'id': 'ws_05489a0b57dc84980168f59fe439788196911a195c70cc8ca9',\
                'type': 'web_search_call',\
                'status': 'completed',\
                'action': {'type': 'search'}\
            },\
            model_provider_name='tensorzero::model_name::gpt-5-mini-responses-web-search::provider_name::openai',\
            type='unknown'\
        ),\
        Thought(\
            text=None,\
            type='thought',\
            signature='gAAAAABo...',\
            _internal_provider_type=None\
        ),\
        UnknownContentBlock(\
            data={\
                'id': 'ws_05489a0b57dc84980168f59fe6b140819690a4468d3304fece',\
                'type': 'web_search_call',\
                'status': 'completed',\
                'action': {'type': 'search'}\
            },\
            model_provider_name='tensorzero::model_name::gpt-5-mini-responses-web-search::provider_name::openai',\
            type='unknown'\
        ),\
        Thought(\
            text=None,\
            type='thought',\
            signature='gAAAAABo...',\
            _internal_provider_type=None\
        ),\
        UnknownContentBlock(\
            data={\
                'id': 'ws_05489a0b57dc84980168f59fe81e408196921b69174f6abaf7',\
                'type': 'web_search_call',\
                'status': 'completed',\
                'action': {'type': 'search'}\
            },\
            model_provider_name='tensorzero::model_name::gpt-5-mini-responses-web-search::provider_name::openai',\
            type='unknown'\
        ),\
        Thought(\
            text=None,\
            type='thought',\
            signature='gAAAAABo...',\
            _internal_provider_type=None\
        ),\
        UnknownContentBlock(\
            data={\
                'id': 'ws_05489a0b57dc84980168f59feda6188196827a0b5aa01e96a1',\
                'type': 'web_search_call',\
                'status': 'completed',\
                'action': {\
                    'type': 'search',\
                    'query': "United Nations World Population Prospects 2024 Japan 2025 population 'Japan population 2025' 'World Population Prospects 2024' 'Japan' "\
                }\
            },\
            model_provider_name='tensorzero::model_name::gpt-5-mini-responses-web-search::provider_name::openai',\
            type='unknown'\
        ),\
        Thought(\
            text=None,\
            type='thought',\
            signature='gAAAAABo...',\
            _internal_provider_type=None\
        ),\
        UnknownContentBlock(\
            data={\
                'id': 'ws_05489a0b57dc84980168f59ff3cc8881968d1c5c9c1bbe4ecc',\
                'type': 'web_search_call',\
                'status': 'completed',\
                'action': {\
                    'type': 'search',\
                    'query': "UN World Population Prospects 2024 Japan population 2025 '123,103,479' 'Japan 2025' 'World Population Prospects' 'Japan' '2025' "\
                }\
            },\
            model_provider_name='tensorzero::model_name::gpt-5-mini-responses-web-search::provider_name::openai',\
            type='unknown'\
        ),\
        Thought(\
            text=None,\
            type='thought',\
            signature='gAAAAABo...',\
            _internal_provider_type=None\
        ),\
        UnknownContentBlock(\
            data={\
                'id': 'ws_05489a0b57dc84980168f59ff67ed48196a0054a38e96f8e0c',\
                'type': 'web_search_call',\
                'status': 'completed',\
                'action': {\
                    'type': 'search',\
                    'query': "United Nations population Japan 2025 'World Population Prospects 2024' 'Japan population 2025' site:un.org"\
                }\
            },\
            model_provider_name='tensorzero::model_name::gpt-5-mini-responses-web-search::provider_name::openai',\
            type='unknown'\
        ),\
        Thought(\
            text=None,\
            type='thought',\
            signature='gAAAAABo...',\
            _internal_provider_type=None\
        ),\
        Text(\
            text="Short answer: about 123–124 million people.\n\nMore precisely:\n- Japan's official estimate (Ministry of Internal Affairs and Communications / e‑Stat) reported a total population of 123,802,000 (including foreign residents) as of October 1, 2024 (release published Apr 14, 2025). ([e-stat.go.jp](https://www.e-stat.go.jp/en/stat-search/files?layout=dataset&page=1&query=Population+Estimates%2C+natural))  \n- The United Nations (WPP 2024, used by sources such as Worldometer) gives a mid‑2025 estimate of about 123.1 million. ([srv1.worldometers.info](https://srv1.worldometers.info/world-population/japan-population/?utm_source=openai))\n\nDo you want a live "right now" estimate for today (Oct 20, 2025) or a breakdown by Japanese nationals vs. foreign residents? I can fetch the latest live or official figures for the exact date you want.",\
            arguments=None,\
            type='text'\
        )\
    ],
    usage=Usage(input_tokens=29904, output_tokens=1921),
    finish_reason=None,
    original_response=None
)
```

[Call any LLM](https://www.tensorzero.com/docs/gateway/call-any-llm) [Configure models & providers](https://www.tensorzero.com/docs/gateway/configure-models-and-providers)

Ctrl+I

Assistant

Responses are generated using AI and may contain mistakes.

## TensorZero Gateway Clients
[Skip to main content](https://www.tensorzero.com/docs/gateway/clients#content-area)

[TensorZero Docs home page![light logo](https://mintcdn.com/tensorzero/afFfERJmQ8YiZpCj/logo/light.svg?fit=max&auto=format&n=afFfERJmQ8YiZpCj&q=85&s=9396731a8a92cee7bd293dee05d71dac)![dark logo](https://mintcdn.com/tensorzero/afFfERJmQ8YiZpCj/logo/dark.svg?fit=max&auto=format&n=afFfERJmQ8YiZpCj&q=85&s=8d61c59531dbcdcace7fcdfc524bd4d2)](https://www.tensorzero.com/)

Search...

Ctrl K

Search...

Navigation

Gateway

TensorZero Gateway Clients

[Guides](https://www.tensorzero.com/docs) [Integrations](https://www.tensorzero.com/docs/integrations/model-providers)

On this page

- [Python](https://www.tensorzero.com/docs/gateway/clients#python)
- [TensorZero Client](https://www.tensorzero.com/docs/gateway/clients#tensorzero-client)
- [Embedded Gateway](https://www.tensorzero.com/docs/gateway/clients#embedded-gateway)
- [Standalone HTTP Gateway](https://www.tensorzero.com/docs/gateway/clients#standalone-http-gateway)
- [OpenAI Python Client](https://www.tensorzero.com/docs/gateway/clients#openai-python-client)
- [Embedded Gateway](https://www.tensorzero.com/docs/gateway/clients#embedded-gateway-2)
- [Standalone HTTP Gateway](https://www.tensorzero.com/docs/gateway/clients#standalone-http-gateway-2)
- [Usage Details](https://www.tensorzero.com/docs/gateway/clients#usage-details)
- [JavaScript / TypeScript / Node](https://www.tensorzero.com/docs/gateway/clients#javascript-/-typescript-/-node)
- [OpenAI Node Client](https://www.tensorzero.com/docs/gateway/clients#openai-node-client)
- [Other Languages and Platforms](https://www.tensorzero.com/docs/gateway/clients#other-languages-and-platforms)
- [TensorZero HTTP API](https://www.tensorzero.com/docs/gateway/clients#tensorzero-http-api)
- [OpenAI HTTP API](https://www.tensorzero.com/docs/gateway/clients#openai-http-api)

The TensorZero Gateway can be used with the **TensorZero Python client**, with **OpenAI clients (e.g. Python/Node)**, or via its **HTTP API in any programming language**.

## [​](https://www.tensorzero.com/docs/gateway/clients\#python)  Python

### [​](https://www.tensorzero.com/docs/gateway/clients\#tensorzero-client)  TensorZero Client

The TensorZero client offers the most flexibility.
It can be used with a built-in embedded (in-memory) gateway or a standalone HTTP gateway.
Additionally, it can be used synchronously or asynchronously.You can install the TensorZero Python client with `pip install tensorzero`.

#### [​](https://www.tensorzero.com/docs/gateway/clients\#embedded-gateway)  Embedded Gateway

The TensorZero Client includes a built-in embedded (in-memory) gateway, so you don’t need to run a separate service.

##### Synchronous

Copy

```
from tensorzero import TensorZeroGateway

with TensorZeroGateway.build_embedded(
    clickhouse_url="http://chuser:chpassword@localhost:8123/tensorzero",  # optional: for observability
    config_file="config/tensorzero.toml",  # optional: for custom functions, models, metrics, etc.
) as client:
    response = client.inference(
        model_name="openai::gpt-4o-mini",  # or: function_name="your_function_name"
        input={
            "messages": [\
                {\
                    "role": "user",\
                    "content": "Write a haiku about artificial intelligence.",\
                }\
            ]
        },
    )
```

##### Asynchronous

Copy

```
from tensorzero import AsyncTensorZeroGateway

async with await AsyncTensorZeroGateway.build_embedded(
    clickhouse_url="http://chuser:chpassword@localhost:8123/tensorzero",  # optional: for observability
    config_file="config/tensorzero.toml",  # optional: for custom functions, models, metrics, etc.
) as gateway:
    inference_response = await gateway.inference(
        model_name="openai::gpt-4o-mini",  # or: function_name="your_function_name"
        input={
            "messages": [\
                {\
                    "role": "user",\
                    "content": "Write a haiku about artificial intelligence.",\
                }\
            ]
        },
    )

    feedback_response = await gateway.feedback(
        inference_id=inference_response.inference_id,
        metric_name="task_success",  # assuming a `task_success` metric is configured
        value=True,
    )
```

You can avoid the `await` in `build_embedded` by setting `async_setup=False`.This is useful for synchronous contexts like `__init__` functions where `await` cannot be used.
However, avoid using it in asynchronous contexts as it blocks the event loop.
For async contexts, use the default `async_setup=True` with await.For example, it’s safe to use `async_setup=False` when initializing a FastAPI server, but not while the server is actively handling requests.

#### [​](https://www.tensorzero.com/docs/gateway/clients\#standalone-http-gateway)  Standalone HTTP Gateway

The TensorZero Client can optionally be used with a standalone HTTP Gateway instead.

##### Synchronous

Copy

```
from tensorzero import TensorZeroGateway

# Assuming the TensorZero Gateway is running on localhost:3000...

with TensorZeroGateway.build_http(gateway_url="http://localhost:3000") as client:
    # Same as above...
```

##### Asynchronous

Copy

```
from tensorzero import AsyncTensorZeroGateway

# Assuming the TensorZero Gateway is running on localhost:3000...

async with await AsyncTensorZeroGateway.build_http(gateway_url="http://localhost:3000") as client:
    # Same as above...
```

You can avoid the `await` in `build_http` by setting `async_setup=False`.
See above for more details.

### [​](https://www.tensorzero.com/docs/gateway/clients\#openai-python-client)  OpenAI Python Client

You can use the OpenAI Python client to run inference requests with TensorZero.
You need to use the TensorZero Client for feedback requests.

#### [​](https://www.tensorzero.com/docs/gateway/clients\#embedded-gateway-2)  Embedded Gateway

You can run an embedded (in-memory) TensorZero Gateway with the OpenAI Python client, which doesn’t require a separate service.

Copy

```
from openai import OpenAI
from tensorzero import patch_openai_client

client = OpenAI()  # or AsyncOpenAI

await patch_openai_client(
    client,
    config_file="path/to/tensorzero.toml",
    clickhouse_url="https://user:password@host:port/database",
)

response = client.chat.completions.create(
    model="tensorzero::model_name::openai::gpt-4o-mini",
    messages=[\
        {\
            "role": "user",\
            "content": "Write a haiku about artificial intelligence.",\
        }\
    ],
)
```

You can avoid the `await` in `patch_openai_client` by setting `async_setup=False`.
See above for more details.

#### [​](https://www.tensorzero.com/docs/gateway/clients\#standalone-http-gateway-2)  Standalone HTTP Gateway

You can deploy the TensorZero Gateway as a separate service and configure the OpenAI client to talk to it.
See [Deployment](https://www.tensorzero.com/docs/deployment/tensorzero-gateway) for instructions on how to deploy the TensorZero Gateway.

Copy

```
from openai import OpenAI

# Assuming the TensorZero Gateway is running on localhost:3000...

with OpenAI(base_url="http://localhost:3000/openai/v1") as client:
    response = client.chat.completions.create(
        model="tensorzero::model_name::openai::gpt-4o-mini",
        messages=[\
            {\
                "role": "user",\
                "content": "Write a haiku about artificial intelligence.",\
            }\
        ],
    )
```

#### [​](https://www.tensorzero.com/docs/gateway/clients\#usage-details)  Usage Details

##### `model`

In the OpenAI client, the `model` parameter should be one of the following:

> **`tensorzero::function_name::<your_function_name>`**For example, if you have a function named `generate_haiku`, you can use `tensorzero::function_name::generate_haiku`.

> **`tensorzero::model_name::<your_model_name>`**For example, if you have a model named `my_model` in the config file, you can use `tensorzero::model_name::my_model`.
> Alternatively, you can use default models like `tensorzero::model_name::openai::gpt-4o-mini`.

##### TensorZero Parameters

You can include optional TensorZero parameters (e.g. `episode_id` and `variant_name`) by prefixing them with `tensorzero::` in the `extra_body` field in OpenAI client requests.

Copy

```
response = client.chat.completions.create(
    # ...
    extra_body={
        "tensorzero::episode_id": "00000000-0000-0000-0000-000000000000",
    },
)
```

## [​](https://www.tensorzero.com/docs/gateway/clients\#javascript-/-typescript-/-node)  JavaScript / TypeScript / Node

### [​](https://www.tensorzero.com/docs/gateway/clients\#openai-node-client)  OpenAI Node Client

You can use the OpenAI client to run inference requests with TensorZero.
You can deploy the TensorZero Gateway as a separate service and configure the OpenAI client to talk to the TensorZero Gateway.See [Deployment](https://www.tensorzero.com/docs/deployment/tensorzero-gateway) for instructions on how to deploy the TensorZero Gateway.

Copy

```
import OpenAI from "openai";

const client = new OpenAI({
  baseURL: "http://localhost:3000/openai/v1",
});

const response = await client.chat.completions.create({
  model: "tensorzero::model_name::openai::gpt-4o-mini",
  messages: [\
    {\
      role: "user",\
      content: "Write a haiku about artificial intelligence.",\
    },\
  ],
});
```

See [OpenAI Python Client » Usage Details](https://www.tensorzero.com/docs/gateway/clients#usage-details) above for instructions on how to use the `model` parameter and other technical details.

You can include optional TensorZero parameters (e.g. `episode_id` and `variant_name`) by prefixing them with `tensorzero::` in the body in OpenAI client requests.

Copy

```
const result = await client.chat.completions.create({
  // ...
  "tensorzero::episode_id": "00000000-0000-0000-0000-000000000000",
});
```

## [​](https://www.tensorzero.com/docs/gateway/clients\#other-languages-and-platforms)  Other Languages and Platforms

The TensorZero Gateway exposes every feature via its HTTP API.
You can deploy the TensorZero Gateway as a standalone service and interact with it from any programming language by making HTTP requests.See [Deployment](https://www.tensorzero.com/docs/deployment/tensorzero-gateway) for instructions on how to deploy the TensorZero Gateway.

### [​](https://www.tensorzero.com/docs/gateway/clients\#tensorzero-http-api)  TensorZero HTTP API

Copy

```
curl -X POST "http://localhost:3000/inference" \
  -H "Content-Type: application/json" \
  -d '{
    "model_name": "openai::gpt-4o-mini",
    "input": {
      "messages": [\
        {\
          "role": "user",\
          "content": "Write a haiku about artificial intelligence."\
        }\
      ]
    }
  }'
```

Copy

```
curl -X POST "http://localhost:3000/feedback" \
  -H "Content-Type: application/json" \
  -d '{
    "inference_id": "00000000-0000-0000-0000-000000000000",
    "metric_name": "task_success",
    "value": true,
  }'
```

### [​](https://www.tensorzero.com/docs/gateway/clients\#openai-http-api)  OpenAI HTTP API

You can make OpenAI-compatible requests to the TensorZero Gateway.

Copy

```
curl -X POST "http://localhost:3000/openai/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "tensorzero::model_name::openai::gpt-4o-mini",
    "messages": [\
      {\
        "role": "user",\
        "content": "Write a haiku about artificial intelligence."\
      }\
    ]
  }'
```

See [OpenAI Python Client » Usage Details](https://www.tensorzero.com/docs/gateway/clients#usage-details) above for instructions on how to use the `model` parameter and other technical details.

You can include optional TensorZero parameters (e.g. `episode_id` and `variant_name`) by prefixing them with `tensorzero::` in the body in OpenAI client requests.

Copy

```
curl -X POST "http://localhost:3000/openai/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d '{
        // ...
        "tensorzero::episode_id": "00000000-0000-0000-0000-000000000000"
      }'
```

[Benchmarks](https://www.tensorzero.com/docs/gateway/benchmarks) [Configuration Reference](https://www.tensorzero.com/docs/gateway/configuration-reference)

Ctrl+I

Assistant

Responses are generated using AI and may contain mistakes.

## TensorZero Gateway Configuration
[Skip to main content](https://www.tensorzero.com/docs/gateway/configuration-reference#content-area)

[TensorZero Docs home page![light logo](https://mintcdn.com/tensorzero/afFfERJmQ8YiZpCj/logo/light.svg?fit=max&auto=format&n=afFfERJmQ8YiZpCj&q=85&s=9396731a8a92cee7bd293dee05d71dac)![dark logo](https://mintcdn.com/tensorzero/afFfERJmQ8YiZpCj/logo/dark.svg?fit=max&auto=format&n=afFfERJmQ8YiZpCj&q=85&s=8d61c59531dbcdcace7fcdfc524bd4d2)](https://www.tensorzero.com/)

Search...

Ctrl K

Search...

Navigation

Gateway

Configuration Reference

[Guides](https://www.tensorzero.com/docs) [Integrations](https://www.tensorzero.com/docs/integrations/model-providers)

On this page

- [\[gateway\]](https://www.tensorzero.com/docs/gateway/configuration-reference#[gateway])
- [auth.cache.enabled](https://www.tensorzero.com/docs/gateway/configuration-reference#auth-cache-enabled)
- [auth.cache.ttl\_ms](https://www.tensorzero.com/docs/gateway/configuration-reference#auth-cache-ttl-ms)
- [auth.enabled](https://www.tensorzero.com/docs/gateway/configuration-reference#auth-enabled)
- [base\_path](https://www.tensorzero.com/docs/gateway/configuration-reference#base-path)
- [bind\_address](https://www.tensorzero.com/docs/gateway/configuration-reference#bind-address)
- [debug](https://www.tensorzero.com/docs/gateway/configuration-reference#debug)
- [disable\_pseudonymous\_usage\_analytics](https://www.tensorzero.com/docs/gateway/configuration-reference#disable-pseudonymous-usage-analytics)
- [export.otlp.traces.enabled](https://www.tensorzero.com/docs/gateway/configuration-reference#export-otlp-traces-enabled)
- [export.otlp.traces.extra\_headers](https://www.tensorzero.com/docs/gateway/configuration-reference#export-otlp-traces-extra-headers)
- [export.otlp.traces.format](https://www.tensorzero.com/docs/gateway/configuration-reference#export-otlp-traces-format)
- [fetch\_and\_encode\_input\_files\_before\_inference](https://www.tensorzero.com/docs/gateway/configuration-reference#fetch-and-encode-input-files-before-inference)
- [global\_outbound\_http\_timeout\_ms](https://www.tensorzero.com/docs/gateway/configuration-reference#global-outbound-http-timeout-ms)
- [observability.async\_writes](https://www.tensorzero.com/docs/gateway/configuration-reference#observability-async-writes)
- [observability.batch\_writes](https://www.tensorzero.com/docs/gateway/configuration-reference#observability-batch-writes)
- [observability.enabled](https://www.tensorzero.com/docs/gateway/configuration-reference#observability-enabled)
- [observability.disable\_automatic\_migrations](https://www.tensorzero.com/docs/gateway/configuration-reference#observability-disable-automatic-migrations)
- [template\_filesystem\_access.base\_path](https://www.tensorzero.com/docs/gateway/configuration-reference#template-filesystem-access-base-path)
- [\[models.model\_name\]](https://www.tensorzero.com/docs/gateway/configuration-reference#[models-model-name])
- [routing](https://www.tensorzero.com/docs/gateway/configuration-reference#routing)
- [timeouts](https://www.tensorzero.com/docs/gateway/configuration-reference#timeouts)
- [\[models.model\_name.providers.provider\_name\]](https://www.tensorzero.com/docs/gateway/configuration-reference#[models-model-name-providers-provider-name])
- [extra\_body](https://www.tensorzero.com/docs/gateway/configuration-reference#extra-body)
- [extra\_headers](https://www.tensorzero.com/docs/gateway/configuration-reference#extra-headers)
- [timeouts](https://www.tensorzero.com/docs/gateway/configuration-reference#timeouts-2)
- [type](https://www.tensorzero.com/docs/gateway/configuration-reference#type)
- [\[embedding\_models.model\_name\]](https://www.tensorzero.com/docs/gateway/configuration-reference#[embedding-models-model-name])
- [routing](https://www.tensorzero.com/docs/gateway/configuration-reference#routing-2)
- [timeout\_ms](https://www.tensorzero.com/docs/gateway/configuration-reference#timeout-ms)
- [\[embedding\_models.model\_name.providers.provider\_name\]](https://www.tensorzero.com/docs/gateway/configuration-reference#[embedding-models-model-name-providers-provider-name])
- [extra\_body](https://www.tensorzero.com/docs/gateway/configuration-reference#extra-body-2)
- [timeout\_ms](https://www.tensorzero.com/docs/gateway/configuration-reference#timeout-ms-2)
- [type](https://www.tensorzero.com/docs/gateway/configuration-reference#type-2)
- [\[provider\_types\]](https://www.tensorzero.com/docs/gateway/configuration-reference#[provider-types])
- [\[functions.function\_name\]](https://www.tensorzero.com/docs/gateway/configuration-reference#[functions-function-name])
- [assistant\_schema](https://www.tensorzero.com/docs/gateway/configuration-reference#assistant-schema)
- [description](https://www.tensorzero.com/docs/gateway/configuration-reference#description)
- [system\_schema](https://www.tensorzero.com/docs/gateway/configuration-reference#system-schema)
- [type](https://www.tensorzero.com/docs/gateway/configuration-reference#type-3)
- [user\_schema](https://www.tensorzero.com/docs/gateway/configuration-reference#user-schema)
- [\[functions.function\_name.variants.variant\_name\]](https://www.tensorzero.com/docs/gateway/configuration-reference#[functions-function-name-variants-variant-name])
- [type](https://www.tensorzero.com/docs/gateway/configuration-reference#type-4)
- [type: "experimental\_chain\_of\_thought"](https://www.tensorzero.com/docs/gateway/configuration-reference#type:-%22experimental-chain-of-thought%22)
- [\[functions.function\_name.experimentation\]](https://www.tensorzero.com/docs/gateway/configuration-reference#[functions-function-name-experimentation])
- [type](https://www.tensorzero.com/docs/gateway/configuration-reference#type-5)
- [\[metrics\]](https://www.tensorzero.com/docs/gateway/configuration-reference#[metrics])
- [level](https://www.tensorzero.com/docs/gateway/configuration-reference#level)
- [optimize](https://www.tensorzero.com/docs/gateway/configuration-reference#optimize)
- [type](https://www.tensorzero.com/docs/gateway/configuration-reference#type-6)
- [\[tools.tool\_name\]](https://www.tensorzero.com/docs/gateway/configuration-reference#[tools-tool-name])
- [description](https://www.tensorzero.com/docs/gateway/configuration-reference#description-2)
- [parameters](https://www.tensorzero.com/docs/gateway/configuration-reference#parameters)
- [strict](https://www.tensorzero.com/docs/gateway/configuration-reference#strict)
- [name](https://www.tensorzero.com/docs/gateway/configuration-reference#name)
- [\[object\_storage\]](https://www.tensorzero.com/docs/gateway/configuration-reference#[object-storage])
- [type](https://www.tensorzero.com/docs/gateway/configuration-reference#type-7)
- [\[postgres\]](https://www.tensorzero.com/docs/gateway/configuration-reference#[postgres])
- [connection\_pool\_size](https://www.tensorzero.com/docs/gateway/configuration-reference#connection-pool-size)
- [enabled](https://www.tensorzero.com/docs/gateway/configuration-reference#enabled)
- [\[rate\_limiting\]](https://www.tensorzero.com/docs/gateway/configuration-reference#[rate-limiting])
- [enabled](https://www.tensorzero.com/docs/gateway/configuration-reference#enabled-2)
- [\[\[rate\_limiting.rules\]\]](https://www.tensorzero.com/docs/gateway/configuration-reference#[[rate-limiting-rules]])
- [Rate Limit Fields](https://www.tensorzero.com/docs/gateway/configuration-reference#rate-limit-fields)
- [priority](https://www.tensorzero.com/docs/gateway/configuration-reference#priority)
- [always](https://www.tensorzero.com/docs/gateway/configuration-reference#always)
- [scope](https://www.tensorzero.com/docs/gateway/configuration-reference#scope)

The configuration file is the backbone of TensorZero.
It defines the behavior of the gateway, including the models and their providers, functions and their variants, tools, metrics, and more.
Developers express the behavior of LLM calls by defining the relevant prompt templates, schemas, and other parameters in this configuration file.The configuration file is a [TOML](https://toml.io/en/) file with a few major sections (TOML tables): `gateway`, `clickhouse`, `postgres`, `models`, `model_providers`, `functions`, `variants`, `tools`, `metrics`, `rate_limiting`, and `object_storage`.

## [​](https://www.tensorzero.com/docs/gateway/configuration-reference\#[gateway])  `[gateway]`

The `[gateway]` section defines the behavior of the TensorZero Gateway.

### [​](https://www.tensorzero.com/docs/gateway/configuration-reference\#auth-cache-enabled)  `auth.cache.enabled`

- **Type:** boolean
- **Required:** no (default: `true`)

Enable caching of authentication database queries.
When enabled, the gateway caches authentication results to reduce database load and improve performance.See [Set up auth for TensorZero](https://www.tensorzero.com/docs/operations/set-up-auth-for-tensorzero) for more details.

### [​](https://www.tensorzero.com/docs/gateway/configuration-reference\#auth-cache-ttl-ms)  `auth.cache.ttl_ms`

- **Type:** integer
- **Required:** no (default: `1000`)

The time-to-live (TTL) in milliseconds for cached authentication queries.
By default, authentication results are cached for 1 second (1000 ms).

tensorzero.toml

Copy

```
[gateway.auth.cache]
enabled = true
ttl_ms = 60_000  # Cache for one minute
```

See [Set up auth for TensorZero](https://www.tensorzero.com/docs/operations/set-up-auth-for-tensorzero) for more details.

### [​](https://www.tensorzero.com/docs/gateway/configuration-reference\#auth-enabled)  `auth.enabled`

- **Type:** boolean
- **Required:** no (default: `false`)

Enable authentication for the TensorZero Gateway.
When enabled, all gateway endpoints except `/status` and `/health` will require a valid API key.You must set up Postgres to use authentication features.
API keys can be created and managed through the TensorZero UI or CLI.

tensorzero.toml

Copy

```
[gateway]
auth.enabled = true
```

See [Set up auth for TensorZero](https://www.tensorzero.com/docs/operations/set-up-auth-for-tensorzero) for a complete guide.

### [​](https://www.tensorzero.com/docs/gateway/configuration-reference\#base-path)  `base_path`

- **Type:** string
- **Required:** no (default: `/`)

If set, the gateway will prefix its HTTP endpoints with this base path.For example, if `base_path` is set to `/custom/prefix`, the inference endpoint will become `/custom/prefix/inference` instead of `/inference`.

### [​](https://www.tensorzero.com/docs/gateway/configuration-reference\#bind-address)  `bind_address`

- **Type:** string
- **Required:** no (default: `[::]:3000`)

Defines the socket address (including port) to bind the TensorZero Gateway to.You can bind the gateway to IPv4 and/or IPv6 addresses.
To bind to an IPv6 address, you can set this field to a value like `[::]:3000`.
Depending on the operating system, this value binds only to IPv6 (e.g. Windows) or to both (e.g. Linux by default).

tensorzero.toml

Copy

```
[gateway]
# ...
bind_address = "0.0.0.0:3000"
# ...
```

### [​](https://www.tensorzero.com/docs/gateway/configuration-reference\#debug)  `debug`

- **Type:** boolean
- **Required:** no (default: `false`)

Typically, TensorZero will not include inputs and outputs in logs or errors to avoid leaking sensitive data.
It may be helpful during development to be able to see more information about requests and responses.
When this field is set to `true`, the gateway will log more verbose errors to assist with debugging.

### [​](https://www.tensorzero.com/docs/gateway/configuration-reference\#disable-pseudonymous-usage-analytics)  `disable_pseudonymous_usage_analytics`

- **Type:** boolean
- **Required:** no (default: `false`)

If set to `true`, TensorZero will not collect or share [pseudonymous usage analytics](https://www.tensorzero.com/docs/deployment/tensorzero-gateway#disabling-pseudonymous-usage-analytics).

### [​](https://www.tensorzero.com/docs/gateway/configuration-reference\#export-otlp-traces-enabled)  `export.otlp.traces.enabled`

- **Type:** boolean
- **Required:** no (default: `false`)

Enable [exporting traces to an external OpenTelemetry-compatible observability system](https://www.tensorzero.com/docs/operations/export-opentelemetry-traces).

Note that you will still need to set the `OTEL_EXPORTER_OTLP_TRACES_ENDPOINT` environment variable. See the above-linked guide for details.

### [​](https://www.tensorzero.com/docs/gateway/configuration-reference\#export-otlp-traces-extra-headers)  `export.otlp.traces.extra_headers`

- **Type:** object (map of string to string)
- **Required:** no (default: `{}`)

Static headers to include in all OTLP trace export requests.
This is useful for adding metadata to OTLP exports.These headers are merged with any dynamic headers sent via HTTP request headers.
When the same header key is present in both static and dynamic headers, the dynamic header value takes precedence.

tensorzero.toml

Copy

```
[gateway.export.otlp.traces]
# ...
extra_headers.space_id = "123"
extra_headers."X-Custom-Header" = "custom-value"
# ...
```

Avoid storing sensitive credentials directly in configuration files. See
[Export OpenTelemetry traces](https://www.tensorzero.com/docs/operations/export-opentelemetry-traces) for
instructions on sending headers dynamically.

### [​](https://www.tensorzero.com/docs/gateway/configuration-reference\#export-otlp-traces-format)  `export.otlp.traces.format`

- **Type:** either “opentelemetry” or “openinference”
- **Required:** no (default: `"opentelemetry"`)

If set to `"opentelemetry"`, TensorZero will set `gen_ai` attributes based on the [OpenTelemetry GenAI semantic conventions](https://github.com/open-telemetry/semantic-conventions/tree/main/docs/gen-ai).
If set to `"openinference"`, TensorZero will set attributes based on the [OpenInference semantic conventions](https://github.com/Arize-ai/openinference/blob/main/spec/llm_spans.md).

### [​](https://www.tensorzero.com/docs/gateway/configuration-reference\#fetch-and-encode-input-files-before-inference)  `fetch_and_encode_input_files_before_inference`

- **Type:** boolean
- **Required:** no (default: `false`)

Controls how the gateway handles remote input files (e.g., images, PDFs) during multimodal inference.If set to `true`, the gateway will fetch remote input files and send them as a base64-encoded payload in the prompt.
This is recommended to ensure that TensorZero and the model providers see identical inputs, which is important for observability and reproducibility.If set to `false`, TensorZero will forward the input file URLs directly to the model provider (when supported) and fetch them for observability in parallel with inference.
This can be more efficient, but may result in different content being observed if the URL content changes between when the provider fetches it and when TensorZero fetches it for observability.

### [​](https://www.tensorzero.com/docs/gateway/configuration-reference\#global-outbound-http-timeout-ms)  `global_outbound_http_timeout_ms`

- **Type:** integer
- **Required:** no (default: `300000` = 5 minutes)

Sets the global timeout in milliseconds for all outbound HTTP requests made by TensorZero to external services such as model providers and APIs.By default, all HTTP requests will timeout after 5 minutes (300,000 ms).
This timeout is intentionally set high to accommodate slow model responses, but you can customize it based on your requirements.The `global_outbound_http_timeout_ms` acts as an upper bound for all more specific timeout configurations in your system.
Any variant-level timeouts (e.g., `timeouts.non_streaming.total_ms`, `timeouts.streaming.ttft_ms`), provider-level timeouts, or embedding model timeouts must be less than or equal to this global timeout.

Setting this value too low may cause legitimate requests to timeout before receiving a response from the model provider.

### [​](https://www.tensorzero.com/docs/gateway/configuration-reference\#observability-async-writes)  `observability.async_writes`

- **Type:** boolean
- **Required:** no (default: `false`)

Enabling this setting will improve the latency of the gateway by offloading the responsibility of writing inferences, feedback, and other data to ClickHouse to a background task, instead of waiting for ClickHouse to complete the writes.
Each database insert is handled immediately in separate background tasks.See the [“Optimize latency and throughput” guide](https://www.tensorzero.com/docs/deployment/optimize-latency-and-throughput) for best practices.You can’t enable `async_writes` and `batch_writes` at the same time.

If you enable this setting, make sure that the gateway lives long enough to complete the writes.
This can be problematic in serverless environments that terminate the gateway instance after the response is returned but before the writes are completed.

### [​](https://www.tensorzero.com/docs/gateway/configuration-reference\#observability-batch-writes)  `observability.batch_writes`

- **Type:** object
- **Required:** no (default: disabled)

Enabling this setting will improve the latency and throughput of the gateway by offloading the responsibility of writing inferences, feedback, and other data to ClickHouse to a background task, instead of waiting for ClickHouse to complete the writes.
With `batch_writes`, multiple records are collected and written together in batches to improve efficiency.The `batch_writes` object supports the following fields:

- `enabled` (boolean): Must be set to `true` to enable batch writes
- `flush_interval_ms` (integer, optional): Maximum time in milliseconds to wait before flushing a batch (default: `100`)
- `max_rows` (integer, optional): Maximum number of rows to collect before flushing a batch (default: `1000`)

tensorzero.toml

Copy

```
[gateway]
# ...
observability.batch_writes = { enabled = true, flush_interval_ms = 200, max_rows = 500 }
# ...
```

See the [“Optimize latency and throughput” guide](https://www.tensorzero.com/docs/deployment/optimize-latency-and-throughput) for best practices.You can’t enable `async_writes` and `batch_writes` at the same time.

If you enable this setting, make sure that the gateway lives long enough to complete the writes.
This can be problematic in serverless environments that terminate the gateway instance after the response is returned but before the writes are completed.

### [​](https://www.tensorzero.com/docs/gateway/configuration-reference\#observability-enabled)  `observability.enabled`

- **Type:** boolean
- **Required:** no (default: `null`)

Enable the observability features of the TensorZero Gateway.
If `true`, the gateway will throw an error on startup if it fails to validate the ClickHouse connection.
If `null`, the gateway will log a warning but continue if ClickHouse is not available, and it will use ClickHouse if available.
If `false`, the gateway will not use ClickHouse.

tensorzero.toml

Copy

```
[gateway]
# ...
observability.enabled = true
# ...
```

### [​](https://www.tensorzero.com/docs/gateway/configuration-reference\#observability-disable-automatic-migrations)  `observability.disable_automatic_migrations`

- **Type:** boolean
- **Required:** no (default `false`)

Disable automatic running of the TensorZero migrations when the TensorZero Gateway launches.
If `true`, then the migrations are not applied upon launch and must instead be applied manually
by running `docker run --rm -e TENSORZERO_CLICKHOUSE_URL=$TENSORZERO_CLICKHOUSE_URL tensorzero/gateway:{version} --run-clickhouse-migrations` or `docker compose run --rm gateway --run-clickhouse-migrations`.
If `false`, then the migrations are run automatically upon launch.

### [​](https://www.tensorzero.com/docs/gateway/configuration-reference\#template-filesystem-access-base-path)  `template_filesystem_access.base_path`

- **Type:** string
- **Required:** no (default disabled)

Set `template_filesystem_access.base_path` to allow MiniJinja templates to load sub-templates using the `{% include %}` and `{% import %}` directives.The directives will be relative to `base_path` and can only access files within that directory or its subdirectories.
The `base_path` can be absolute or relative to the configuration file’s location.

## [​](https://www.tensorzero.com/docs/gateway/configuration-reference\#[models-model-name])  `[models.model_name]`

The `[models.model_name]` section defines the behavior of a model.
You can define multiple models by including multiple `[models.model_name]` sections.A model is provider agnostic, and the relevant providers are defined in the `providers` sub-section (see below).If your `model_name` is not a basic string, it can be escaped with quotation marks.
For example, periods are not allowed in basic strings, so you can define `llama-3.1-8b-instruct` as `[models."llama-3.1-8b-instruct"]`.

tensorzero.toml

Copy

```
[models.claude-3-haiku-20240307]
# fieldA = ...
# fieldB = ...
# ...

[models."llama-3.1-8b-instruct"]
# fieldA = ...
# fieldB = ...
# ...
```

### [​](https://www.tensorzero.com/docs/gateway/configuration-reference\#routing)  `routing`

- **Type:** array of strings
- **Required:** yes

A list of provider names to route requests to.
The providers must be defined in the `providers` sub-section (see below).
The TensorZero Gateway will attempt to route a request to the first provider in the list, and fallback to subsequent providers in order if the request is not successful.

Copy

```
// tensorzero.toml
[models.gpt-4o]
# ...
routing = ["openai", "azure"]
# ...

[models.gpt-4o.providers.openai]
# ...

[models.gpt-4o.providers.azure]
# ...
```

### [​](https://www.tensorzero.com/docs/gateway/configuration-reference\#timeouts)  `timeouts`

- **Type:** object
- **Required:** no

The `timeouts` object allows you to set granular timeouts for requests to this model.You can define timeouts for non-streaming and streaming requests separately: `timeouts.non_streaming.total_ms` corresponds to the total request duration and `timeouts.streaming.ttft_ms` corresponds to the time to first token (TTFT).For example, the following configuration sets a 15-second timeout for non-streaming requests and a 3-second timeout for streaming requests (TTFT):

Copy

```
[models.model_name]
# ...
timeouts = { non_streaming.total_ms = 15000, streaming.ttft_ms = 3000 }
# ...
```

The specified timeouts apply to the scope of an entire model inference request, including all retries and fallbacks across its providers.
You can also set timeouts at the variant level and provider level.
Multiple timeouts can be active simultaneously.

## [​](https://www.tensorzero.com/docs/gateway/configuration-reference\#[models-model-name-providers-provider-name])  `[models.model_name.providers.provider_name]`

The `providers` sub-section defines the behavior of a specific provider for a model.
You can define multiple providers by including multiple `[models.model_name.providers.provider_name]` sections.If your `provider_name` is not a basic string, it can be escaped with quotation marks.
For example, periods are not allowed in basic strings, so you can define `vllm.internal` as `[models.model_name.providers."vllm.internal"]`.

Copy

```
// tensorzero.toml
[models.gpt-4o]
# ...
routing = ["openai", "azure"]
# ...

[models.gpt-4o.providers.openai]
# ...

[models.gpt-4o.providers.azure]
# ...
```

### [​](https://www.tensorzero.com/docs/gateway/configuration-reference\#extra-body)  `extra_body`

- **Type:** array of objects (see below)
- **Required:** no

The `extra_body` field allows you to modify the request body that TensorZero sends to a model provider.
This advanced feature is an “escape hatch” that lets you use provider-specific functionality that TensorZero hasn’t implemented yet.Each object in the array must have two fields:

- `pointer`: A [JSON Pointer](https://datatracker.ietf.org/doc/html/rfc6901) string specifying where to modify the request body
- One of the following:
  - `value`: The value to insert at that location; it can be of any type including nested types
  - `delete = true`: Deletes the field at the specified location, if present.

You can also set `extra_body` for a variant entry.
The model provider `extra_body` entries take priority over variant `extra_body` entries.Additionally, you can set `extra_body` at inference-time.
The values provided at inference-time take priority over the values in the configuration file.

Example: \`extra\_body\`

If TensorZero would normally send this request body to the provider…

Copy

```
{
  "project": "tensorzero",
  "safety_checks": {
    "no_internet": false,
    "no_agi": true
  }
}
```

…then the following `extra_body`…

Copy

```
extra_body = [\
  { pointer = "/agi", value = true},\
  { pointer = "/safety_checks/no_agi", value = { bypass = "on" }}\
]
```

…overrides the request body to:

Copy

```
{
  "agi": true,
  "project": "tensorzero",
  "safety_checks": {
    "no_internet": false,
    "no_agi": {
      "bypass": "on"
    }
  }
}
```

### [​](https://www.tensorzero.com/docs/gateway/configuration-reference\#extra-headers)  `extra_headers`

- **Type:** array of objects (see below)
- **Required:** no

The `extra_headers` field allows you to set or overwrite the request headers that TensorZero sends to a model provider.
This advanced feature is an “escape hatch” that lets you use provider-specific functionality that TensorZero hasn’t implemented yet.Each object in the array must have two fields:

- `name` (string): The name of the header to modify (e.g. `anthropic-beta`)
- One of the following:
  - `value` (string): The value of the header (e.g. `token-efficient-tools-2025-02-19`)
  - `delete = true`: Deletes the header from the request, if present

You can also set `extra_headers` for a variant entry.
The model provider `extra_headers` entries take priority over variant `extra_headers` entries.

Example: \`extra\_headers\`

If TensorZero would normally send the following request headers to the provider…

Copy

```
Safety-Checks: on
```

…then the following `extra_headers`…

Copy

```
extra_headers = [\
  { name = "Safety-Checks", value = "off"},\
  { name = "Intelligence-Level", value = "AGI"}\
]
```

…overrides the request headers to:

Copy

```
Safety-Checks: off
Intelligence-Level: AGI
```

### [​](https://www.tensorzero.com/docs/gateway/configuration-reference\#timeouts-2)  `timeouts`

- **Type:** object
- **Required:** no

The `timeouts` object allows you to set granular timeouts for individual requests to a model provider.You can define timeouts for non-streaming and streaming requests separately: `timeouts.non_streaming.total_ms` corresponds to the total request duration and `timeouts.streaming.ttft_ms` corresponds to the time to first token (TTFT).For example, the following configuration sets a 15-second timeout for non-streaming requests and a 3-second timeout for streaming requests (TTFT):

Copy

```
[models.model_name.providers.provider_name]
# ...
timeouts = { non_streaming.total_ms = 15000, streaming.ttft_ms = 3000 }
# ...
```

This setting applies to individual requests to the model provider.
If you’re using an advanced variant type that performs multiple requests, the timeout will apply to each request separately.
If you’ve defined retries and fallbacks, the timeout will apply to each retry and fallback separately.
This setting is particularly useful if you’d like to retry or fallback on a request that’s taking too long.You can also set timeouts at the model level and provider level.
Multiple timeouts can be active simultaneously.Separately, you can set a global timeout for the entire inference request using the TensorZero client’s `timeout` field (or simply killing the request if you’re using a different client).

### [​](https://www.tensorzero.com/docs/gateway/configuration-reference\#type)  `type`

- **Type:** string
- **Required:** yes

Defines the types of the provider. See [Integrations » Model Providers](https://www.tensorzero.com/docs/gateway/api-reference/inference#content-block) for details.The supported provider types are `anthropic`, `aws_bedrock`, `aws_sagemaker`, `azure`, `deepseek`, `fireworks`, `gcp_vertex_anthropic`, `gcp_vertex_gemini`, `google_ai_studio_gemini`, `groq`, `hyperbolic`, `mistral`, `openai`, `openrouter`, `sglang`, `tgi`, `together`, `vllm`, and `xai`.The other fields in the provider sub-section depend on the provider type.

tensorzero.toml

Copy

```
[models.gpt-4o.providers.azure]
# ...
type = "azure"
# ...
```

type: "anthropic"

##### `model_name`

- **Type:** string
- **Required:** yes

Defines the model name to use with the Anthropic API.
See [Anthropic’s documentation](https://docs.anthropic.com/en/docs/about-claude/models#model-names) for the list of available model names.

tensorzero.toml

Copy

```
[models.claude-3-haiku.providers.anthropic]
# ...
type = "anthropic"
model_name = "claude-3-haiku-20240307"
# ...
```

##### `api_key_location`

- **Type:** string or object
- **Required:** no (default: `env::ANTHROPIC_API_KEY` unless set otherwise in `provider_type.anthropic.defaults.api_key_location`)

Defines the location of the API key for the Anthropic provider.Can be either a string for a single credential location, or an object with `default` and `fallback` fields for credential fallback support.The supported locations are `env::ENVIRONMENT_VARIABLE`, `dynamic::ARGUMENT_NAME`, and `none`.See [the API reference](https://www.tensorzero.com/docs/gateway/api-reference/inference#credentials) and [Credential Management](https://www.tensorzero.com/docs/operations/manage-credentials#configure-credential-fallbacks) for more details.

tensorzero.toml

Copy

```
[models.claude-3-haiku.providers.anthropic]
# ...
type = "anthropic"
api_key_location = "dynamic::anthropic_api_key"
# api_key_location = "env::ALTERNATE_ANTHROPIC_API_KEY"
# api_key_location = { default = "dynamic::anthropic_api_key", fallback = "env::ANTHROPIC_API_KEY" }
# ...
```

##### `api_base`

- **Type:** string
- **Required:** no (default: `https://api.anthropic.com/v1/messages`)

Overrides the base URL used for Anthropic Messages API requests. The value should include the full endpoint path (for example `https://example.com/v1/messages`).

tensorzero.toml

Copy

```
[models.claude-3-haiku.providers.anthropic]
# ...
type = "anthropic"
api_base = "https://example.com/v1/messages"
# ...
```

##### `beta_structured_outputs`

- **Type:** boolean
- **Required:** no (default: `false`)

Enables Anthropic’s beta structured outputs feature, which provides native support for strict JSON schema validation and strict tool parameter validation.When enabled:

- Adds the `anthropic-beta: structured-outputs-2025-11-13` header to requests
- For JSON functions with `json_mode = "strict"`, forwards the output schema in the `output_format` field
- For tools with `strict = true`, forwards the `strict` parameter to enable strict validation

tensorzero.toml

Copy

```
[models.claude_structured.providers.anthropic]
type = "anthropic"
model_name = "claude-sonnet-4-5-20250929"
beta_structured_outputs = true
```

type: "aws\_bedrock"

##### `allow_auto_detect_region`

- **Type:** boolean
- **Required:** no (default: `false`)

Defines whether to automatically detect the AWS region to use with the SageMaker API.
Under the hood, the gateway will use the AWS SDK to try to detect the region.
Alternatively, you can specify the region manually with the `region` field (recommended).

##### `model_id`

- **Type:** string
- **Required:** yes

Defines the model ID to use with the AWS Bedrock API.
See [AWS Bedrock’s documentation](https://docs.aws.amazon.com/bedrock/latest/userguide/model-ids.html) for the list of available model IDs.

tensorzero.toml

Copy

```
[models.claude-3-haiku.providers.aws_bedrock]
# ...
type = "aws_bedrock"
model_id = "anthropic.claude-3-haiku-20240307-v1:0"
# ...
```

Many AWS Bedrock models are only available through cross-region inference profiles.
For those models, the `model_id` requires special prefix (e.g. the `us.` prefix in `us.anthropic.claude-3-7-sonnet-20250219-v1:0`).
See the [AWS documentation on inference profiles](https://docs.aws.amazon.com/bedrock/latest/userguide/inference-profiles-support.html).

##### `region`

- **Type:** string
- **Required:** no (default: based on credentials if set, otherwise `us-east-1`)

Defines the AWS region to use with the AWS Bedrock API.

tensorzero.toml

Copy

```
[models.claude-3-haiku.providers.aws_bedrock]
# ...
type = "aws_bedrock"
region = "us-east-2"
# ...
```

type: "aws\_sagemaker"

##### `allow_auto_detect_region`

- **Type:** boolean
- **Required:** no (default: `false`)

Defines whether to automatically detect the AWS region to use with the SageMaker API.
Under the hood, the gateway will use the AWS SDK to try to detect the region.
Alternatively, you can specify the region manually with the `region` field (recommended).

##### `endpoint_name`

- **Type:** string
- **Required:** yes

Defines the endpoint name to use with the AWS SageMaker API.

##### `hosted_provider`

- **Type:** string
- **Required:** yes

Defines the underlying model provider to use with the SageMaker API.
The `aws_sagemaker` provider is a wrapper on other providers.Currently, the only supported `hosted_provider` options are:

- `openai` (including any OpenAI-compatible server e.g. Ollama)
- `tgi`

For example, if you’re using Ollama, you can set:

tensorzero.toml

Copy

```
[models.claude-3-haiku.providers.aws_sagemaker]
# ...
type = "aws_sagemaker"
hosted_provider = "openai"
# ...
```

##### `model_name`

- **Type:** string
- **Required:** yes

Defines the model name to use with the AWS SageMaker API.

tensorzero.toml

Copy

```
[models.claude-3-haiku.providers.aws_sagemaker]
# ...
type = "aws_sagemaker"
model_name = "gemma3:1b"
# ...
```

##### `region`

- **Type:** string
- **Required:** no (default: based on credentials if set, otherwise `us-east-1`)

Defines the AWS region to use with the AWS Bedrock API.

tensorzero.toml

Copy

```
[models.claude-3-haiku.providers.aws_sagemaker]
# ...
type = "aws_sagemaker"
region = "us-east-2"
# ...
```

type: "azure"

The TensorZero Gateway handles the API version under the hood (currently `2025-04-01-preview`).
You only need to set the `deployment_id` and `endpoint` fields.

##### `api_key_location`

- **Type:** string or object
- **Required:** no (default: `env::AZURE_OPENAI_API_KEY` unless set otherwise in `provider_type.azure.defaults.api_key_location`)

Defines the location of the API key for the Azure OpenAI provider.Can be either a string for a single credential location, or an object with `default` and `fallback` fields for credential fallback support.The supported locations are `env::ENVIRONMENT_VARIABLE`, `dynamic::ARGUMENT_NAME`, and `none`.See [the API reference](https://www.tensorzero.com/docs/gateway/api-reference/inference#credentials) and [Credential Management](https://www.tensorzero.com/docs/operations/manage-credentials#configure-credential-fallbacks) for more details.

tensorzero.toml

Copy

```
[models.gpt-4o-mini.providers.azure]
# ...
type = "azure"
api_key_location = "dynamic::azure_openai_api_key"
# api_key_location = "env::ALTERNATE_AZURE_OPENAI_API_KEY"
# api_key_location = { default = "dynamic::azure_openai_api_key", fallback = "env::AZURE_OPENAI_API_KEY" }
# ...
```

##### `deployment_id`

- **Type:** string
- **Required:** yes

Defines the deployment ID of the Azure OpenAI deployment.See [Azure OpenAI’s documentation](https://learn.microsoft.com/en-us/azure/ai-services/openai/concepts/models) for the list of available models.

tensorzero.toml

Copy

```
[models.gpt-4o-mini.providers.azure]
# ...
type = "azure"
deployment_id = "gpt4o-mini-20240718"
# ...
```

##### `endpoint`

- **Type:** string
- **Required:** yes

Defines the endpoint of the Azure OpenAI deployment (protocol and hostname).

tensorzero.toml

Copy

```
[models.gpt-4o-mini.providers.azure]
# ...
type = "azure"
endpoint = "https://<your-endpoint>.openai.azure.com"
# ...
```

If the endpoint starts with `env::`, the succeeding value will be treated as an environment variable name and the gateway will attempt to retrieve the value from the environment on startup.
If the endpoint starts with `dynamic::`, the succeeding value will be treated as an dynamic credential name and the gateway will attempt to retrieve the value from the `dynamic_credentials` field on each inference it is needed.

type: "deepseek"

##### `api_key_location`

- **Type:** string or object
- **Required:** no (default: `env::DEEPSEEK_API_KEY` unless set otherwise in `provider_type.deepseek.defaults.api_key_location`)

Defines the location of the API key for the DeepSeek provider.Can be either a string for a single credential location, or an object with `default` and `fallback` fields for credential fallback support.The supported locations are `env::ENVIRONMENT_VARIABLE` and `dynamic::ARGUMENT_NAME` (see [the API reference](https://www.tensorzero.com/docs/gateway/api-reference/inference#credentials) and [Credential Management](https://www.tensorzero.com/docs/operations/manage-credentials#configure-credential-fallbacks) for more details).

tensorzero.toml

Copy

```
[models.deepseek_chat.providers.deepseek]
# ...
type = "deepseek"
api_key_location = "dynamic::deepseek_api_key"
# api_key_location = "env::ALTERNATE_DEEPSEEK_API_KEY"
# api_key_location = { default = "dynamic::deepseek_api_key", fallback = "env::DEEPSEEK_API_KEY" }
# ...
```

##### `model_name`

- **Type:** string
- **Required:** yes

Defines the model name to use with the DeepSeek API.
Currently supported models are `deepseek-chat` (DeepSeek-v3) and `deepseek-reasoner` (R1).

tensorzero.toml

Copy

```
[models.deepseek_chat.providers.deepseek]
# ...
type = "deepseek"
model_name = "deepseek-chat"
# ...
```

type: "fireworks"

##### `api_key_location`

- **Type:** string or object
- **Required:** no (default: `env::FIREWORKS_API_KEY` unless set otherwise in `provider_type.fireworks.defaults.api_key_location`)

Defines the location of the API key for the Fireworks provider.Can be either a string for a single credential location, or an object with `default` and `fallback` fields for credential fallback support.The supported locations are `env::ENVIRONMENT_VARIABLE` and `dynamic::ARGUMENT_NAME` (see [the API reference](https://www.tensorzero.com/docs/gateway/api-reference/inference#credentials) and [Credential Management](https://www.tensorzero.com/docs/operations/manage-credentials#configure-credential-fallbacks) for more details).

tensorzero.toml

Copy

```
[models."llama-3.1-8b-instruct".providers.fireworks]
# ...
type = "fireworks"
api_key_location = "dynamic::fireworks_api_key"
# api_key_location = "env::ALTERNATE_FIREWORKS_API_KEY"
# api_key_location = { default = "dynamic::fireworks_api_key", fallback = "env::FIREWORKS_API_KEY" }
# ...
```

##### `model_name`

- **Type:** string
- **Required:** yes

Defines the model name to use with the Fireworks API.See [Fireworks’ documentation](https://fireworks.ai/models) for the list of available model names.
You can also deploy your own models on Fireworks AI.

tensorzero.toml

Copy

```
[models."llama-3.1-8b-instruct".providers.fireworks]
# ...
type = "fireworks"
model_name = "accounts/fireworks/models/llama-v3p1-8b-instruct"
# ...
```

type: "gcp\_vertex\_anthropic"

##### `credential_location`

- **Type:** string or object
- **Required:** no (default: `path_from_env::GCP_VERTEX_CREDENTIALS_PATH` unless otherwise set in `provider_type.gcp_vertex_anthropic.defaults.credential_location`)

Defines the location of the credentials for the GCP Vertex Anthropic provider.Can be either a string for a single credential location, or an object with `default` and `fallback` fields for credential fallback support.The supported locations are `env::PATH_TO_CREDENTIALS_FILE`, `path_from_env::ENVIRONMENT_VARIABLE`, `dynamic::CREDENTIALS_ARGUMENT_NAME`, `path::PATH_TO_CREDENTIALS_FILE`, and `sdk` (use Google Cloud SDK to auto-discover credentials).See [the API reference](https://www.tensorzero.com/docs/gateway/api-reference/inference#credentials) and [Credential Management](https://www.tensorzero.com/docs/operations/manage-credentials#configure-credential-fallbacks) for more details.

tensorzero.toml

Copy

```
[models.claude-3-haiku.providers.gcp_vertex]
# ...
type = "gcp_vertex_anthropic"
credential_location = "dynamic::gcp_credentials_path"
# credential_location = "path_from_env::GCP_VERTEX_CREDENTIALS_PATH"
# credential_location = "path::/etc/secrets/gcp-key.json"
# credential_location = "sdk"
# credential_location = { default = "sdk", fallback = "path::/etc/secrets/gcp-key.json" }
# ...
```

##### `endpoint_id`

- **Type:** string
- **Required:** no (exactly one of `endpoint_id` or `model_id` must be set)

Defines the endpoint ID of the GCP Vertex AI Anthropic model.Use `model_id` for off-the-shelf models and `endpoint_id` for fine-tuned models and custom endpoints.

##### `location`

- **Type:** string
- **Required:** yes

Defines the location (region) of the GCP Vertex AI Anthropic model.

tensorzero.toml

Copy

```
[models.claude-3-haiku.providers.gcp_vertex]
# ...
type = "gcp_vertex_anthropic"
location = "us-central1"
# ...
```

##### `model_id`

- **Type:** string
- **Required:** no (exactly one of `model_id` or `endpoint_id` must be set)

Defines the model ID of the GCP Vertex AI model.See [Anthropic’s GCP documentation](https://docs.anthropic.com/en/api/claude-on-vertex-ai#api-model-names) for the list of available model IDs.

tensorzero.toml

Copy

```
[models.claude-3-haiku.providers.gcp_vertex]
# ...
type = "gcp_vertex_anthropic"
model_id = "claude-3-haiku@20240307"
# ...
```

Use `model_id` for off-the-shelf models and `endpoint_id` for fine-tuned models and custom endpoints.

##### `project_id`

- **Type:** string
- **Required:** yes

Defines the project ID of the GCP Vertex AI model.

tensorzero.toml

Copy

```
[models.claude-3-haiku-2024030.providers.gcp_vertex]
# ...
type = "gcp_vertex"
project_id = "your-project-id"
# ...
```

type: "gcp\_vertex\_gemini"

##### `credential_location`

- **Type:** string or object
- **Required:** no (default: `path_from_env::GCP_VERTEX_CREDENTIALS_PATH` unless otherwise set in `provider_type.gcp_vertex_gemini.defaults.credential_location`)

Defines the location of the credentials for the GCP Vertex Gemini provider.Can be either a string for a single credential location, or an object with `default` and `fallback` fields for credential fallback support.The supported locations are `env::PATH_TO_CREDENTIALS_FILE`, `path_from_env::ENVIRONMENT_VARIABLE`, `dynamic::CREDENTIALS_ARGUMENT_NAME`, `path::PATH_TO_CREDENTIALS_FILE`, and `sdk` (use Google Cloud SDK to auto-discover credentials).See [the API reference](https://www.tensorzero.com/docs/gateway/api-reference/inference#credentials) and [Credential Management](https://www.tensorzero.com/docs/operations/manage-credentials#configure-credential-fallbacks) for more details.

tensorzero.toml

Copy

```
[models."gemini-1.5-flash".providers.gcp_vertex]
# ...
type = "gcp_vertex_gemini"
credential_location = "dynamic::gcp_credentials_path"
# credential_location = "path_from_env::GCP_VERTEX_CREDENTIALS_PATH"
# credential_location = "path::/etc/secrets/gcp-key.json"
# credential_location = "sdk"
# credential_location = { default = "sdk", fallback = "path::/etc/secrets/gcp-key.json" }
# ...
```

##### `endpoint_id`

- **Type:** string
- **Required:** no (exactly one of `endpoint_id` or `model_id` must be set)

Defines the endpoint ID of the GCP Vertex AI Gemini model.Use `model_id` for off-the-shelf models and `endpoint_id` for fine-tuned models and custom endpoints.

##### `location`

- **Type:** string
- **Required:** yes

Defines the location (region) of the GCP Vertex Gemini model.

tensorzero.toml

Copy

```
[models."gemini-1.5-flash".providers.gcp_vertex]
# ...
type = "gcp_vertex_gemini"
location = "us-central1"
# ...
```

##### `model_id`

- **Type:** string
- **Required:** no (exactly one of `model_id` or `endpoint_id` must be set)

Defines the model ID of the GCP Vertex AI model.See [GCP Vertex AI’s documentation](https://cloud.google.com/vertex-ai/generative-ai/docs/learn/model-versions) for the list of available model IDs.

tensorzero.toml

Copy

```
[models."gemini-1.5-flash".providers.gcp_vertex]
# ...
type = "gcp_vertex_gemini"
model_id = "gemini-1.5-flash-001"
# ...
```

##### `project_id`

- **Type:** string
- **Required:** yes

Defines the project ID of the GCP Vertex AI model.

tensorzero.toml

Copy

```
[models."gemini-1.5-flash".providers.gcp_vertex]
# ...
type = "gcp_vertex_gemini"
project_id = "your-project-id"
# ...
```

type: "google\_ai\_studio\_gemini"

##### `api_key_location`

- **Type:** string or object
- **Required:** no (default: `env::GOOGLE_AI_STUDIO_API_KEY` unless otherwise set in `provider_type.google_ai_studio.defaults.credential_location`)

Defines the location of the API key for the Google AI Studio Gemini provider.Can be either a string for a single credential location, or an object with `default` and `fallback` fields for credential fallback support.The supported locations are `env::ENVIRONMENT_VARIABLE` and `dynamic::ARGUMENT_NAME` (see [the API reference](https://www.tensorzero.com/docs/gateway/api-reference/inference#credentials) and [Credential Management](https://www.tensorzero.com/docs/operations/manage-credentials#configure-credential-fallbacks) for more details).

tensorzero.toml

Copy

```
[models."gemini-1.5-flash".providers.google_ai_studio_gemini]
# ...
type = "google_ai_studio_gemini"
api_key_location = "dynamic::google_ai_studio_api_key"
# api_key_location = "env::ALTERNATE_GOOGLE_AI_STUDIO_API_KEY"
# api_key_location = { default = "dynamic::google_ai_studio_api_key", fallback = "env::GOOGLE_AI_STUDIO_API_KEY" }
# ...
```

##### `model_name`

- **Type:** string
- **Required:** yes

Defines the model name to use with the Google AI Studio Gemini API.See [Google AI Studio’s documentation](https://ai.google.dev/gemini-api/docs/models/gemini) for the list of available model names.

tensorzero.toml

Copy

```
[models."gemini-1.5-flash".providers.google_ai_studio_gemini]
# ...
type = "google_ai_studio_gemini"
model_name = "gemini-1.5-flash-001"
# ...
```

type: "groq"

##### `api_key_location`

- **Type:** string or object
- **Required:** no (default: `env::GROQ_API_KEY` unless otherwise set in `provider_type.groq.defaults.credential_location`)

Defines the location of the API key for the Groq provider.Can be either a string for a single credential location, or an object with `default` and `fallback` fields for credential fallback support.The supported locations are `env::ENVIRONMENT_VARIABLE` and `dynamic::ARGUMENT_NAME` (see [the API reference](https://www.tensorzero.com/docs/gateway/api-reference/inference#credentials) and [Credential Management](https://www.tensorzero.com/docs/operations/manage-credentials#configure-credential-fallbacks) for more details).

tensorzero.toml

Copy

```
[models.llama4_scout_17b_16e_instruct.providers.groq]
# ...
type = "groq"
api_key_location = "dynamic::groq_api_key"
# api_key_location = "env::ALTERNATE_GROQ_API_KEY"
# api_key_location = { default = "dynamic::groq_api_key", fallback = "env::GROQ_API_KEY" }
# ...
```

##### `model_name`

- **Type:** string
- **Required:** yes

Defines the model name to use with the Groq API.See [Groq’s documentation](https://groq.com/pricing) for the list of available model names.

tensorzero.toml

Copy

```
[models.llama4_scout_17b_16e_instruct.providers.groq]
# ...
type = "groq"
model_name = "meta-llama/llama-4-scout-17b-16e-instruct"
# ...
```

type: "hyperbolic"

##### `api_key_location`

- **Type:** string or object
- **Required:** no (default: `env::HYPERBOLIC_API_KEY` unless otherwise set in `provider_type.hyperbolic.defaults.api_key_location`)

Defines the location of the API key for the Hyperbolic provider.Can be either a string for a single credential location, or an object with `default` and `fallback` fields for credential fallback support.The supported locations are `env::ENVIRONMENT_VARIABLE` and `dynamic::ARGUMENT_NAME` (see [the API reference](https://www.tensorzero.com/docs/gateway/api-reference/inference#credentials) and [Credential Management](https://www.tensorzero.com/docs/operations/manage-credentials#configure-credential-fallbacks) for more details).

tensorzero.toml

Copy

```
[models."meta-llama/Meta-Llama-3-70B-Instruct".providers.hyperbolic]
# ...
type = "hyperbolic"
api_key_location = "dynamic::hyperbolic_api_key"
# api_key_location = "env::ALTERNATE_HYPERBOLIC_API_KEY"
# api_key_location = { default = "dynamic::hyperbolic_api_key", fallback = "env::HYPERBOLIC_API_KEY" }
# ...
```

##### `model_name`

- **Type:** string
- **Required:** yes

Defines the model name to use with the Hyperbolic API.See [Hyperbolic’s documentation](https://app.hyperbolic.xyz/models) for the list of available model names.

tensorzero.toml

Copy

```
[models."meta-llama/Meta-Llama-3-70B-Instruct".providers.hyperbolic]
# ...
type = "hyperbolic"
model_name = "meta-llama/Meta-Llama-3-70B-Instruct"
# ...
```

type: "mistral"

##### `api_key_location`

- **Type:** string or object
- **Required:** no (default: `env::MISTRAL_API_KEY` unless otherwise set in `provider_type.mistral.defaults.api_key_location`)

Defines the location of the API key for the Mistral provider.Can be either a string for a single credential location, or an object with `default` and `fallback` fields for credential fallback support.The supported locations are `env::ENVIRONMENT_VARIABLE` and `dynamic::ARGUMENT_NAME` (see [the API reference](https://www.tensorzero.com/docs/gateway/api-reference/inference#credentials) and [Credential Management](https://www.tensorzero.com/docs/operations/manage-credentials#configure-credential-fallbacks) for more details).

tensorzero.toml

Copy

```
[models."open-mistral-nemo".providers.mistral]
# ...
type = "mistral"
api_key_location = "dynamic::mistral_api_key"
# api_key_location = "env::ALTERNATE_MISTRAL_API_KEY"
# api_key_location = { default = "dynamic::mistral_api_key", fallback = "env::MISTRAL_API_KEY" }
# ...
```

##### `model_name`

- **Type:** string
- **Required:** yes

Defines the model name to use with the Mistral API.See [Mistral’s documentation](https://docs.mistral.ai/getting-started/models/) for the list of available model names.

tensorzero.toml

Copy

```
[models."open-mistral-nemo".providers.mistral]
# ...
type = "mistral"
model_name = "open-mistral-nemo-2407"
# ...
```

type: "openai"

##### `api_base`

- **Type:** string
- **Required:** no (default: `https://api.openai.com/v1/`)

Defines the base URL of the OpenAI API.You can use the `api_base` field to use an API provider that is compatible with the OpenAI API.
However, many providers are only “approximately compatible” with the OpenAI API, so you might need to use a specialized model provider in those cases.

tensorzero.toml

Copy

```
[models."gpt-4o".providers.openai]
# ...
type = "openai"
api_base = "https://api.openai.com/v1/"
# ...
```

##### `api_key_location`

- **Type:** string or object
- **Required:** no (default: `env::OPENAI_API_KEY` unless otherwise set in `provider_types.openai.defaults.api_key_location`)

Defines the location of the API key for the OpenAI provider.Can be either a string for a single credential location, or an object with `default` and `fallback` fields for credential fallback support.The supported locations are `env::ENVIRONMENT_VARIABLE`, `dynamic::ARGUMENT_NAME`, and `none` (see [the API reference](https://www.tensorzero.com/docs/gateway/api-reference/inference#credentials) and [Credential Management](https://www.tensorzero.com/docs/operations/manage-credentials#configure-credential-fallbacks) for more details).

tensorzero.toml

Copy

```
[models.gpt-4o-mini.providers.openai]
# ...
type = "openai"
api_key_location = "dynamic::openai_api_key"
# api_key_location = "env::ALTERNATE_OPENAI_API_KEY"
# api_key_location = "none"
# api_key_location = { default = "dynamic::openai_api_key", fallback = "env::OPENAI_API_KEY" }
# ...
```

##### `api_type`

- **Type:** string
- **Required:** no (default: `chat_completions`)

Determines which OpenAI API endpoint to use.
The default value is `chat_completions` for the standard Chat Completions API.
Set to `responses` to use the Responses API, which provides access to built-in tools like web search and reasoning capabilities.

tensorzero.toml

Copy

```
[models.gpt-5-mini-responses.providers.openai]
# ...
type = "openai"
api_type = "responses"
# ...
```

##### `include_encrypted_reasoning`

- **Type:** boolean
- **Required:** no (default: `false`)

Enables encrypted reasoning (thought blocks) when using the Responses API.
This parameter allows the model to show its internal reasoning process before generating the final response.**Only available when `api_type = "responses"`.**

tensorzero.toml

Copy

```
[models.gpt-5-mini-responses.providers.openai]
# ...
type = "openai"
api_type = "responses"
include_encrypted_reasoning = true
# ...
```

##### `model_name`

- **Type:** string
- **Required:** yes

Defines the model name to use with the OpenAI API.See [OpenAI’s documentation](https://platform.openai.com/docs/models) for the list of available model names.

tensorzero.toml

Copy

```
[models.gpt-4o-mini.providers.openai]
# ...
type = "openai"
model_name = "gpt-4o-mini-2024-07-18"
# ...
```

##### `provider_tools`

- **Type:** array of objects
- **Required:** no (default: `[]`)

Defines provider-specific built-in tools that are available for this model provider.
These are tools that run server-side on the provider’s infrastructure (e.g., OpenAI’s web search tool).Each object in the array should contain the provider-specific tool configuration as defined by the provider’s API.
For example, OpenAI’s Responses API supports a `web_search` tool that enables the model to search the web for information.This field can be set statically in the configuration file or dynamically at inference time via the `provider_tools` parameter in the `/inference` endpoint or `tensorzero::provider_tools` in the OpenAI-compatible endpoint.
See the [Inference API Reference](https://www.tensorzero.com/docs/gateway/api-reference/inference#provider_tools) for more details on dynamic usage.

tensorzero.toml

Copy

```
[models.gpt-5-mini-responses-web-search.providers.openai]
# ...
type = "openai"
api_type = "responses"
provider_tools = [{type = "web_search"}]  # Enable OpenAI's built-in web search tool
# ...
```

type: "openrouter"

##### `api_key_location`

- **Type:** string or object
- **Required:** no (default: `env::OPENROUTER_API_KEY` unless otherwise set in `provider_types.openrouter.defaults.api_key_location`)

Defines the location of the API key for the OpenRouter provider.Can be either a string for a single credential location, or an object with `default` and `fallback` fields for credential fallback support.The supported locations are `env::ENVIRONMENT_VARIABLE` and `dynamic::ARGUMENT_NAME` (see [the API reference](https://www.tensorzero.com/docs/gateway/api-reference/inference#credentials) and [Credential Management](https://www.tensorzero.com/docs/operations/manage-credentials#configure-credential-fallbacks) for more details).

tensorzero.toml

Copy

```
[models.gpt4_turbo.providers.openrouter]
# ...
type = "openrouter"
api_key_location = "dynamic::openrouter_api_key"
# api_key_location = "env::ALTERNATE_OPENROUTER_API_KEY"
# api_key_location = { default = "dynamic::openrouter_api_key", fallback = "env::OPENROUTER_API_KEY" }
# ...
```

##### `model_name`

- **Type:** string
- **Required:** yes

Defines the model name to use with the OpenRouter API.See [OpenRouter’s documentation](https://openrouter.ai/models) for the list of available model names.

tensorzero.toml

Copy

```
[models.gpt4_turbo.providers.openrouter]
# ...
type = "openrouter"
model_name = "openai/gpt4.1"
# ...
```

type: "sglang"

##### `api_base`

- **Type:** string
- **Required:** yes

Defines the base URL of the SGLang API.

tensorzero.toml

Copy

```
[models.llama.providers.sglang]
# ...
type = "sglang"
api_base = "http://localhost:8080/v1/"
# ...
```

##### `api_key_location`

- **Type:** string or object
- **Required:** no (default: `none`)

Defines the location of the API key for the SGLang provider.Can be either a string for a single credential location, or an object with `default` and `fallback` fields for credential fallback support.The supported locations are `env::ENVIRONMENT_VARIABLE`, `dynamic::ARGUMENT_NAME`, and `none` (see [the API reference](https://www.tensorzero.com/docs/gateway/api-reference/inference#credentials) and [Credential Management](https://www.tensorzero.com/docs/operations/manage-credentials#configure-credential-fallbacks) for more details).

tensorzero.toml

Copy

```
[models.llama.providers.sglang]
# ...
type = "sglang"
api_key_location = "dynamic::sglang_api_key"
# api_key_location = "env::ALTERNATE_SGLANG_API_KEY"
# api_key_location = "none"  # if authentication is disabled
# api_key_location = { default = "dynamic::sglang_api_key", fallback = "env::SGLANG_API_KEY" }
# ...
```

type: "together"

##### `api_key_location`

- **Type:** string or object
- **Required:** no (default: `env::TOGETHER_API_KEY` unless otherwise set in `provider_types.together.defaults.api_key_location`)

Defines the location of the API key for the Together AI provider.Can be either a string for a single credential location, or an object with `default` and `fallback` fields for credential fallback support.The supported locations are `env::ENVIRONMENT_VARIABLE` and `dynamic::ARGUMENT_NAME` (see [the API reference](https://www.tensorzero.com/docs/gateway/api-reference/inference#credentials) and [Credential Management](https://www.tensorzero.com/docs/operations/manage-credentials#configure-credential-fallbacks) for more details).

tensorzero.toml

Copy

```
[models.llama3_1_8b_instruct_turbo.providers.together]
# ...
type = "together"
api_key_location = "dynamic::together_api_key"
# api_key_location = "env::ALTERNATE_TOGETHER_API_KEY"
# api_key_location = { default = "dynamic::together_api_key", fallback = "env::TOGETHER_API_KEY" }
# ...
```

##### `model_name`

- **Type:** string
- **Required:** yes

Defines the model name to use with the Together API.See [Together’s documentation](https://docs.together.ai/docs/chat-models) for the list of available model names.You can also deploy your own models on Together AI.

tensorzero.toml

Copy

```
[models.llama3_1_8b_instruct_turbo.providers.together]
# ...
type = "together"
model_name = "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo"
# ...
```

type: "vllm"

##### `api_base`

- **Type:** string
- **Required:** yes (default: `http://localhost:8000/v1/`)

Defines the base URL of the VLLM API.

tensorzero.toml

Copy

```
[models."phi-3.5-mini-instruct".providers.vllm]
# ...
type = "vllm"
api_base = "http://localhost:8000/v1/"
# ...
```

##### `model_name`

- **Type:** string
- **Required:** yes

Defines the model name to use with the vLLM API.

tensorzero.toml

Copy

```
[models."phi-3.5-mini-instruct".providers.vllm]
# ...
type = "vllm"
model_name = "microsoft/Phi-3.5-mini-instruct"
# ...
```

##### `api_key_location`

- **Type:** string or object
- **Required:** no (default: `env::VLLM_API_KEY`)

Defines the location of the API key for the vLLM provider.Can be either a string for a single credential location, or an object with `default` and `fallback` fields for credential fallback support.The supported locations are `env::ENVIRONMENT_VARIABLE`, `dynamic::ARGUMENT_NAME`, and `none` (see [the API reference](https://www.tensorzero.com/docs/gateway/api-reference/inference#credentials) and [Credential Management](https://www.tensorzero.com/docs/operations/manage-credentials#configure-credential-fallbacks) for more details).

tensorzero.toml

Copy

```
[models."phi-3.5-mini-instruct".providers.vllm]
# ...
type = "vllm"
api_key_location = "dynamic::vllm_api_key"
# api_key_location = "env::ALTERNATE_VLLM_API_KEY"
# api_key_location = "none"
# api_key_location = { default = "dynamic::vllm_api_key", fallback = "env::VLLM_API_KEY" }
# ...
```

type: "xai"

##### `api_key_location`

- **Type:** string or object
- **Required:** no (default: `env::XAI_API_KEY` unless otherwise set in `provider_types.xai.defaults.api_key_location`)

Defines the location of the API key for the xAI provider.Can be either a string for a single credential location, or an object with `default` and `fallback` fields for credential fallback support.The supported locations are `env::ENVIRONMENT_VARIABLE` and `dynamic::ARGUMENT_NAME` (see [the API reference](https://www.tensorzero.com/docs/gateway/api-reference/inference#credentials) and [Credential Management](https://www.tensorzero.com/docs/operations/manage-credentials#configure-credential-fallbacks) for more details).

tensorzero.toml

Copy

```
[models.grok_2_1212.providers.xai]
# ...
type = "xai"
api_key_location = "dynamic::xai_api_key"
# api_key_location = "env::ALTERNATE_XAI_API_KEY"
# api_key_location = { default = "dynamic::xai_api_key", fallback = "env::XAI_API_KEY" }
# ...
```

##### `model_name`

- **Type:** string
- **Required:** yes

Defines the model name to use with the xAI API.See [xAI’s documentation](https://docs.x.ai/docs/models) for the list of available model names.

tensorzero.toml

Copy

```
[models.grok_2_1212.providers.xai]
# ...
type = "xai"
model_name = "grok-2-1212"
# ...
```

type: "tgi"

##### `api_base`

- **Type:** string
- **Required:** yes

Defines the base URL of the TGI API.

tensorzero.toml

Copy

```
[models.phi_4.providers.tgi]
# ...
type = "tgi"
api_base = "http://localhost:8080/v1/"
# ...
```

##### `api_key_location`

- **Type:** string or object
- **Required:** no (default: `none`)

Defines the location of the API key for the TGI provider.Can be either a string for a single credential location, or an object with `default` and `fallback` fields for credential fallback support.The supported locations are `env::ENVIRONMENT_VARIABLE`, `dynamic::ARGUMENT_NAME`, and `none` (see [the API reference](https://www.tensorzero.com/docs/gateway/api-reference/inference#credentials) and [Credential Management](https://www.tensorzero.com/docs/operations/manage-credentials#configure-credential-fallbacks) for more details).

tensorzero.toml

Copy

```
[models.phi_4.providers.tgi]
# ...
type = "tgi"
api_key_location = "dynamic::tgi_api_key"
# api_key_location = "env::ALTERNATE_TGI_API_KEY"
# api_key_location = "none"  # if authentication is disabled
# api_key_location = { default = "dynamic::tgi_api_key", fallback = "env::TGI_API_KEY" }
# ...
```

## [​](https://www.tensorzero.com/docs/gateway/configuration-reference\#[embedding-models-model-name])  `[embedding_models.model_name]`

The `[embedding_models.model_name]` section defines the behavior of an embedding model.
You can define multiple models by including multiple `[embedding_models.model_name]` sections.A model is provider agnostic, and the relevant providers are defined in the `providers` sub-section (see below).If your `model_name` is not a basic string, it can be escaped with quotation marks.
For example, periods are not allowed in basic strings, so you can define `embedding-0.1` as `[embedding_models."embedding-0.1"]`.

tensorzero.toml

Copy

```
[embedding_models.openai-text-embedding-3-small]
# fieldA = ...
# fieldB = ...
# ...

[embedding_models."t0-text-embedding-3.5-massive"]
# fieldA = ...
# fieldB = ...
# ...
```

### [​](https://www.tensorzero.com/docs/gateway/configuration-reference\#routing-2)  `routing`

- **Type:** array of strings
- **Required:** yes

A list of provider names to route requests to.
The providers must be defined in the `providers` sub-section (see below).
The TensorZero Gateway will attempt to route a request to the first provider in the list, and fallback to subsequent providers in order if the request is not successful.

Copy

```
// tensorzero.toml
[embedding_models.model-name]
# ...
routing = ["openai", "alternative-provider"]
# ...

[embedding_models.model-name.providers.openai]
# ...

[embedding_models.model-name.providers.alternative-provider]
# ...
```

### [​](https://www.tensorzero.com/docs/gateway/configuration-reference\#timeout-ms)  `timeout_ms`

- **Type:** integer
- **Required:** no

The total time allowed (in milliseconds) for the embedding model to complete the request.
This timeout applies to the entire request, including all provider attempts in the routing list.If a provider times out, the next provider in the routing list will be attempted.
If all providers timeout or the model-level timeout is reached, an error will be returned.

tensorzero.toml

Copy

```
[embedding_models.model-name]
routing = ["openai"]
timeout_ms = 5000  # 5 second timeout
# ...
```

## [​](https://www.tensorzero.com/docs/gateway/configuration-reference\#[embedding-models-model-name-providers-provider-name])  `[embedding_models.model_name.providers.provider_name]`

The `providers` sub-section defines the behavior of a specific provider for a model.
You can define multiple providers by including multiple `[embedding_models.model_name.providers.provider_name]` sections.If your `provider_name` is not a basic string, it can be escaped with quotation marks.
For example, periods are not allowed in basic strings, so you can define `vllm.internal` as `[embedding_models.model_name.providers."vllm.internal"]`.

Copy

```
// tensorzero.toml
[embedding_models.model-name]
# ...
routing = ["openai", "alternative-provider"]
# ...

[embedding_models.model-name.providers.openai]
# ...

[embedding_models.model-name.providers.alternative-provider]
# ...
```

### [​](https://www.tensorzero.com/docs/gateway/configuration-reference\#extra-body-2)  `extra_body`

- **Type:** array of objects (see below)
- **Required:** no

The `extra_body` field allows you to modify the request body that TensorZero sends to the embedding model provider.
This advanced feature is an “escape hatch” that lets you use provider-specific functionality that TensorZero hasn’t implemented yet.Each object in the array must have two fields:

- `pointer`: A [JSON Pointer](https://datatracker.ietf.org/doc/html/rfc6901) string specifying where to modify the request body
- One of the following:
  - `value`: The value to insert at that location; it can be of any type including nested types
  - `delete = true`: Deletes the field at the specified location, if present.

You can also set `extra_body` at inference-time.
The values provided at inference-time take priority over the values in the configuration file.

tensorzero.toml

Copy

```
[embedding_models.openai-text-embedding-3-small.providers.openai]
type = "openai"
extra_body = [\
  { pointer = "/dimensions", value = 1536 }\
]
```

### [​](https://www.tensorzero.com/docs/gateway/configuration-reference\#timeout-ms-2)  `timeout_ms`

- **Type:** integer
- **Required:** no

The total time allowed (in milliseconds) for this specific provider to complete the embedding request.If the provider times out, the next provider in the routing list will be attempted (if any).

tensorzero.toml

Copy

```
[embedding_models.model-name.providers.openai]
type = "openai"
timeout_ms = 3000  # 3 second timeout for this provider
# ...
```

### [​](https://www.tensorzero.com/docs/gateway/configuration-reference\#type-2)  `type`

- **Type:** string
- **Required:** yes

Defines the types of the provider. See [Integrations » Model Providers](https://www.tensorzero.com/docs/integrations/model-providers) for details.The other fields in the provider sub-section depend on the provider type.

tensorzero.toml

Copy

```
[embedding_models.model-name.providers.openai]
# ...
type = "openai"
# ...
```

type: "openai"

##### `api_base`

- **Type:** string
- **Required:** no (default: `https://api.openai.com/v1/`)

Defines the base URL of the OpenAI API.You can use the `api_base` field to use an API provider that is compatible with the OpenAI API.
However, many providers are only “approximately compatible” with the OpenAI API, so you might need to use a specialized model provider in those cases.

tensorzero.toml

Copy

```
[embedding_models.openai-text-embedding-3-small.providers.openai]
# ...
type = "openai"
api_base = "https://api.openai.com/v1/"
# ...
```

##### `api_key_location`

- **Type:** string or object
- **Required:** no (default: `env::OPENAI_API_KEY`)

Defines the location of the API key for the OpenAI provider.Can be either a string for a single credential location, or an object with `default` and `fallback` fields for credential fallback support.The supported locations are `env::ENVIRONMENT_VARIABLE`, `dynamic::ARGUMENT_NAME`, and `none` (see [the API reference](https://www.tensorzero.com/docs/gateway/api-reference/inference#credentials) and [Credential Management](https://www.tensorzero.com/docs/operations/manage-credentials#configure-credential-fallbacks) for more details).

tensorzero.toml

Copy

```
[embedding_models.openai-text-embedding-3-small.providers.openai]
# ...
type = "openai"
api_key_location = "dynamic::openai_api_key"
# api_key_location = "env::ALTERNATE_OPENAI_API_KEY"
# api_key_location = "none"
# api_key_location = { default = "dynamic::openai_api_key", fallback = "env::OPENAI_API_KEY" }
# ...
```

##### `model_name`

- **Type:** string
- **Required:** yes

Defines the model name to use with the OpenAI API.See [OpenAI’s documentation](https://platform.openai.com/docs/models/embeddings) for the list of available model names.

tensorzero.toml

Copy

```
[embedding_models.openai-text-embedding-3-small.providers.openai]
# ...
type = "openai"
model_name = "text-embedding-3-small"
# ...
```

## [​](https://www.tensorzero.com/docs/gateway/configuration-reference\#[provider-types])  `[provider_types]`

The `provider_types` section of the configuration allows users to specify global settings that are related to the handling of a particular inference provider type (like `"openai"` or `"anthropic"`), such as where to look by default for credentials.

\[provider\_types.anthropic\]

##### `defaults.api_key_location`

- **Type:** string or object
- **Required:** no (default: `env::ANTHROPIC_API_KEY`)

Defines the default location of the API key for Anthropic models.Can be either a string for a single credential location, or an object with `default` and `fallback` fields for credential fallback support.The supported locations are `env::ENVIRONMENT_VARIABLE` and `dynamic::ARGUMENT_NAME` (see [the API reference](https://www.tensorzero.com/docs/gateway/api-reference/inference#credentials) and [Credential Management](https://www.tensorzero.com/docs/operations/manage-credentials#configure-credential-fallbacks) for more details).

tensorzero.toml

Copy

```
[provider_types.anthropic.defaults]
# ...
api_key_location = "dynamic::anthropic_api_key"
# api_key_location = "env::ALTERNATE_ANTHROPIC_API_KEY"
# api_key_location = { default = "dynamic::anthropic_api_key", fallback = "env::ANTHROPIC_API_KEY" }
# ...
```

\[provider\_types.azure\]

##### `defaults.api_key_location`

- **Type:** string or object
- **Required:** no (default: `env::AZURE_OPENAI_API_KEY`)

Defines the default location of the API key for Azure models.Can be either a string for a single credential location, or an object with `default` and `fallback` fields for credential fallback support.The supported locations are `env::ENVIRONMENT_VARIABLE` and `dynamic::ARGUMENT_NAME` (see [the API reference](https://www.tensorzero.com/docs/gateway/api-reference/inference#credentials) and [Credential Management](https://www.tensorzero.com/docs/operations/manage-credentials#configure-credential-fallbacks) for more details).

tensorzero.toml

Copy

```
[provider_types.azure.defaults]
# ...
api_key_location = "dynamic::azure_openai_api_key"
# api_key_location = { default = "dynamic::azure_openai_api_key", fallback = "env::AZURE_OPENAI_API_KEY" }
# ...
```

\[provider\_types.deepseek\]

##### `defaults.api_key_location`

- **Type:** string or object
- **Required:** no (default: `env::DEEPSEEK_API_KEY`)

Defines the location of the API key for the DeepSeek provider.Can be either a string for a single credential location, or an object with `default` and `fallback` fields for credential fallback support.The supported locations are `env::ENVIRONMENT_VARIABLE` and `dynamic::ARGUMENT_NAME` (see [the API reference](https://www.tensorzero.com/docs/gateway/api-reference/inference#credentials) and [Credential Management](https://www.tensorzero.com/docs/operations/manage-credentials#configure-credential-fallbacks) for more details).

tensorzero.toml

Copy

```
[provider_types.deepseek.defaults]
# ...
api_key_location = "dynamic::deepseek_api_key"
# api_key_location = { default = "dynamic::deepseek_api_key", fallback = "env::DEEPSEEK_API_KEY" }
# ...
```

\[provider\_types.fireworks\]

##### `defaults.api_key_location`

- **Type:** string or object
- **Required:** no (default: `env::FIREWORKS_API_KEY`)

Defines the location of the API key for the Fireworks provider.Can be either a string for a single credential location, or an object with `default` and `fallback` fields for credential fallback support.The supported locations are `env::ENVIRONMENT_VARIABLE` and `dynamic::ARGUMENT_NAME` (see [the API reference](https://www.tensorzero.com/docs/gateway/api-reference/inference#credentials) and [Credential Management](https://www.tensorzero.com/docs/operations/manage-credentials#configure-credential-fallbacks) for more details).

tensorzero.toml

Copy

```
[provider_types.fireworks.defaults]
# ...
api_key_location = "dynamic::fireworks_api_key"
# api_key_location = { default = "dynamic::fireworks_api_key", fallback = "env::FIREWORKS_API_KEY" }
# ...
```

\[provider\_types.gcp\_vertex\_anthropic\]

##### `defaults.credential_location`

- **Type:** string or object
- **Required:** no (default: `path_from_env::GCP_VERTEX_CREDENTIALS_PATH`)

Defines the location of the credentials for the GCP Vertex Anthropic provider.Can be either a string for a single credential location, or an object with `default` and `fallback` fields for credential fallback support.The supported locations are `env::PATH_TO_CREDENTIALS_FILE`, `dynamic::CREDENTIALS_ARGUMENT_NAME`, `path::PATH_TO_CREDENTIALS_FILE`, and `path_from_env::ENVIRONMENT_VARIABLE` (see [the API reference](https://www.tensorzero.com/docs/gateway/api-reference/inference#credentials) and [Credential Management](https://www.tensorzero.com/docs/operations/manage-credentials#configure-credential-fallbacks) for more details).

tensorzero.toml

Copy

```
[provider_types.gcp_vertex_anthropic.defaults]
# ...
credential_location = "dynamic::gcp_credentials_path"
# credential_location = "path::/etc/secrets/gcp-key.json"
# credential_location = { default = "sdk", fallback = "path::/etc/secrets/gcp-key.json" }
# ...
```

\[provider\_types.gcp\_vertex\_gemini\]

#### [​](https://www.tensorzero.com/docs/gateway/configuration-reference\#batch)  `batch`

- **Type:** object
- **Required:** no (default: `null`)

The `batch` object allows you to configure batch processing for GCP Vertex models.
Today we support batch inference through GCP Vertex using Google cloud storage as documented [here](https://cloud.google.com/vertex-ai/docs/tabular-data/classification-regression/get-batch-predictions#api:-cloud-storage).
To do this you must also have object\_storage (see the [object\_storage](https://www.tensorzero.com/docs/gateway/configuration-reference#object_storage) section) configured using GCP.

tensorzero.toml

Copy

```
[provider_types.gcp_vertex_gemini.batch]
storage_type = "cloud_storage"
input_uri_prefix = "gs://my-bucket/batch-inputs/"
output_uri_prefix = "gs://my-bucket/batch-outputs/"
```

The `batch` object supports the following configuration:

##### `storage_type`

- **Type:** string
- **Required:** no (default `"none"`)

Defines the storage type for batch processing. Currently, only `"cloud_storage"` and `"none"` are supported.

##### `input_uri_prefix`

- **Type:** string
- **Required:** yes when `storage_type` is `"cloud_storage"`

Defines the Google Cloud Storage URI prefix where batch input files will be stored.

##### `output_uri_prefix`

- **Type:** string
- **Required:** yes when `storage_type` is `"cloud_storage"`

Defines the Google Cloud Storage URI prefix where batch output files will be stored.

##### `defaults.credential_location`

- **Type:** string or object
- **Required:** no (default: `path_from_env::GCP_VERTEX_CREDENTIALS_PATH`)

Defines the location of the credentials for the GCP Vertex Gemini provider.Can be either a string for a single credential location, or an object with `default` and `fallback` fields for credential fallback support.The supported locations are `env::PATH_TO_CREDENTIALS_FILE`, `dynamic::CREDENTIALS_ARGUMENT_NAME`, `path::PATH_TO_CREDENTIALS_FILE`, and `path_from_env::ENVIRONMENT_VARIABLE` (see [the API reference](https://www.tensorzero.com/docs/gateway/api-reference/inference#credentials) and [Credential Management](https://www.tensorzero.com/docs/operations/manage-credentials#configure-credential-fallbacks) for more details).

tensorzero.toml

Copy

```
[provider_types.gcp_vertex_gemini.defaults]
# ...
credential_location = "dynamic::gcp_credentials_path"
# credential_location = "path::/etc/secrets/gcp-key.json"
# credential_location = { default = "sdk", fallback = "path::/etc/secrets/gcp-key.json" }
# ...
```

\[provider\_types.google\_ai\_studio\]

##### `defaults.api_key_location`

- **Type:** string or object
- **Required:** no (default: `env::GOOGLE_AI_STUDIO_API_KEY`)

Defines the location of the API key for the Google AI Studio provider.Can be either a string for a single credential location, or an object with `default` and `fallback` fields for credential fallback support.The supported locations are `env::ENVIRONMENT_VARIABLE` and `dynamic::ARGUMENT_NAME` (see [the API reference](https://www.tensorzero.com/docs/gateway/api-reference/inference#credentials) and [Credential Management](https://www.tensorzero.com/docs/operations/manage-credentials#configure-credential-fallbacks) for more details).

tensorzero.toml

Copy

```
[provider_types.google_ai_studio.defaults]
# ...
api_key_location = "dynamic::google_ai_studio_api_key"
# api_key_location = { default = "dynamic::google_ai_studio_api_key", fallback = "env::GOOGLE_AI_STUDIO_API_KEY" }
# ...
```

\[provider\_types.groq\]

##### `defaults.api_key_location`

- **Type:** string or object
- **Required:** no (default: `env::GROQ_API_KEY`)

Defines the location of the API key for the Groq provider.Can be either a string for a single credential location, or an object with `default` and `fallback` fields for credential fallback support.The supported locations are `env::ENVIRONMENT_VARIABLE` and `dynamic::ARGUMENT_NAME` (see [the API reference](https://www.tensorzero.com/docs/gateway/api-reference/inference#credentials) and [Credential Management](https://www.tensorzero.com/docs/operations/manage-credentials#configure-credential-fallbacks) for more details).

tensorzero.toml

Copy

```
[provider_types.groq.defaults]
# ...
api_key_location = "dynamic::groq_api_key"
# api_key_location = { default = "dynamic::groq_api_key", fallback = "env::GROQ_API_KEY" }
# ...
```

\[provider\_types.hyperbolic\]

##### `defaults.api_key_location`

- **Type:** string or object
- **Required:** no (default: `env::HYPERBOLIC_API_KEY`)

Defines the location of the API key for the Hyperbolic provider.Can be either a string for a single credential location, or an object with `default` and `fallback` fields for credential fallback support.The supported locations are `env::ENVIRONMENT_VARIABLE` and `dynamic::ARGUMENT_NAME` (see [the API reference](https://www.tensorzero.com/docs/gateway/api-reference/inference#credentials) and [Credential Management](https://www.tensorzero.com/docs/operations/manage-credentials#configure-credential-fallbacks) for more details).

tensorzero.toml

Copy

```
[provider_types.hyperbolic.defaults]
# ...
api_key_location = "dynamic::hyperbolic_api_key"
# api_key_location = { default = "dynamic::hyperbolic_api_key", fallback = "env::HYPERBOLIC_API_KEY" }
# ...
```

\[provider\_types.mistral\]

##### `defaults.api_key_location`

- **Type:** string or object
- **Required:** no (default: `env::MISTRAL_API_KEY`)

Defines the location of the API key for the Mistral provider.Can be either a string for a single credential location, or an object with `default` and `fallback` fields for credential fallback support.The supported locations are `env::ENVIRONMENT_VARIABLE` and `dynamic::ARGUMENT_NAME` (see [the API reference](https://www.tensorzero.com/docs/gateway/api-reference/inference#credentials) and [Credential Management](https://www.tensorzero.com/docs/operations/manage-credentials#configure-credential-fallbacks) for more details).

tensorzero.toml

Copy

```
[provider_types.mistral.defaults]
# ...
api_key_location = "dynamic::mistral_api_key"
# api_key_location = { default = "dynamic::mistral_api_key", fallback = "env::MISTRAL_API_KEY" }
# ...
```

\[provider\_types.openai\]

##### `defaults.api_key_location`

- **Type:** string or object
- **Required:** no (default: `env::OPENAI_API_KEY`)

Defines the location of the API key for the OpenAI provider.Can be either a string for a single credential location, or an object with `default` and `fallback` fields for credential fallback support.The supported locations are `env::ENVIRONMENT_VARIABLE` and `dynamic::ARGUMENT_NAME` (see [the API reference](https://www.tensorzero.com/docs/gateway/api-reference/inference#credentials) and [Credential Management](https://www.tensorzero.com/docs/operations/manage-credentials#configure-credential-fallbacks) for more details).

tensorzero.toml

Copy

```
[provider_types.openai.defaults]
# ...
api_key_location = "dynamic::openai_api_key"
# api_key_location = { default = "dynamic::openai_api_key", fallback = "env::OPENAI_API_KEY" }
# ...
```

\[provider\_types.openrouter\]

##### `defaults.api_key_location`

- **Type:** string or object
- **Required:** no (default: `env::OPENROUTER_API_KEY`)

Defines the location of the API key for the OpenRouter provider.Can be either a string for a single credential location, or an object with `default` and `fallback` fields for credential fallback support.The supported locations are `env::ENVIRONMENT_VARIABLE` and `dynamic::ARGUMENT_NAME` (see [the API reference](https://www.tensorzero.com/docs/gateway/api-reference/inference#credentials) and [Credential Management](https://www.tensorzero.com/docs/operations/manage-credentials#configure-credential-fallbacks) for more details).

tensorzero.toml

Copy

```
[provider_types.openrouter.defaults]
# ...
api_key_location = "dynamic::openrouter_api_key"
# api_key_location = { default = "dynamic::openrouter_api_key", fallback = "env::OPENROUTER_API_KEY" }
# ...
```

\[provider\_types.together\]

##### `defaults.api_key_location`

- **Type:** string or object
- **Required:** no (default: `env::TOGETHER_API_KEY`)

Defines the location of the API key for the Together provider.Can be either a string for a single credential location, or an object with `default` and `fallback` fields for credential fallback support.The supported locations are `env::ENVIRONMENT_VARIABLE` and `dynamic::ARGUMENT_NAME` (see [the API reference](https://www.tensorzero.com/docs/gateway/api-reference/inference#credentials) and [Credential Management](https://www.tensorzero.com/docs/operations/manage-credentials#configure-credential-fallbacks) for more details).

tensorzero.toml

Copy

```
[provider_types.together.defaults]
# ...
api_key_location = "dynamic::together_api_key"
# api_key_location = { default = "dynamic::together_api_key", fallback = "env::TOGETHER_API_KEY" }
# ...
```

\[provider\_types.xai\]

##### `defaults.api_key_location`

- **Type:** string or object
- **Required:** no (default: `env::XAI_API_KEY`)

Defines the location of the API key for the xAI provider.Can be either a string for a single credential location, or an object with `default` and `fallback` fields for credential fallback support.The supported locations are `env::ENVIRONMENT_VARIABLE` and `dynamic::ARGUMENT_NAME` (see [the API reference](https://www.tensorzero.com/docs/gateway/api-reference/inference#credentials) and [Credential Management](https://www.tensorzero.com/docs/operations/manage-credentials#configure-credential-fallbacks) for more details).

tensorzero.toml

Copy

```
[provider_types.xai.defaults]
# ...
api_key_location = "dynamic::xai_api_key"
# api_key_location = { default = "dynamic::xai_api_key", fallback = "env::XAI_API_KEY" }
# ...
```

## [​](https://www.tensorzero.com/docs/gateway/configuration-reference\#[functions-function-name])  `[functions.function_name]`

The `[functions.function_name]` section defines the behavior of a function.
You can define multiple functions by including multiple `[functions.function_name]` sections.A function can have multiple variants, and each variant is defined in the `variants` sub-section (see below).
A function expresses the abstract behavior of an LLM call (e.g. the schemas for the messages), and its variants express concrete instantiations of that LLM call (e.g. specific templates and models).If your `function_name` is not a basic string, it can be escaped with quotation marks.
For example, periods are not allowed in basic strings, so you can define `summarize-2.0` as `[functions."summarize-2.0"]`.

tensorzero.toml

Copy

```
[functions.draft-email]
# fieldA = ...
# fieldB = ...
# ...

[functions.summarize-email]
# fieldA = ...
# fieldB = ...
# ...
```

### [​](https://www.tensorzero.com/docs/gateway/configuration-reference\#assistant-schema)  `assistant_schema`

- **Type:** string (path)
- **Required:** no

Defines the path to the assistant schema file.
The path is relative to the configuration file.If provided, the assistant schema file should contain a [JSON Schema](https://json-schema.org/) for the assistant messages.
The variables in the schema are used for templating the assistant messages.
If a schema is provided, all function variants must also provide an assistant template (see below).

tensorzero.toml

Copy

```
[functions.draft-email]
# ...
assistant_schema = "./functions/draft-email/assistant_schema.json"
# ...

[functions.draft-email.variants.prompt-v1]
# ...
assistant_template = "./functions/draft-email/prompt-v1/assistant_template.minijinja"
# ...
```

### [​](https://www.tensorzero.com/docs/gateway/configuration-reference\#description)  `description`

- **Type:** string
- **Required:** no

Defines a description of the function.In the future, this description will inform automated optimization recipes.

tensorzero.toml

Copy

```
[functions.extract_data]
# ...
description = "Extract the sender's name (e.g. 'John Doe'), email address (e.g. 'john.doe@example.com'), and phone number (e.g. '+1234567890') from a customer's email."
# ...
```

### [​](https://www.tensorzero.com/docs/gateway/configuration-reference\#system-schema)  `system_schema`

- **Type:** string (path)
- **Required:** no

Defines the path to the system schema file.
The path is relative to the configuration file.If provided, the system schema file should contain a [JSON Schema](https://json-schema.org/) for the system message.
The variables in the schema are used for templating the system message.
If a schema is provided, all function variants must also provide a system template (see below).

tensorzero.toml

Copy

```
[functions.draft-email]
# ...
system_schema = "./functions/draft-email/system_schema.json"
# ...

[functions.draft-email.variants.prompt-v1]
# ...
system_template = "./functions/draft-email/prompt-v1/system_template.minijinja"
# ...
```

### [​](https://www.tensorzero.com/docs/gateway/configuration-reference\#type-3)  `type`

- **Type:** string
- **Required:** yes

Defines the type of the function.The supported function types are `chat` and `json`.Most other fields in the function section depend on the function type.

tensorzero.toml

Copy

```
[functions.draft-email]
# ...
type = "chat"
# ...
```

type: "chat"

##### `parallel_tool_calls`

- **Type:** boolean
- **Required:** no

Determines whether the function should be allowed to call multiple tools in a single conversation turn.If not set, TensorZero will default to the model provider’s default behavior.Most model providers do not support this feature. In those cases, this field will be ignored.

tensorzero.toml

Copy

```
[functions.draft-email]
# ...
type = "chat"
parallel_tool_calls = true
# ...
```

##### `tool_choice`

- **Type:** string
- **Required:** no (default: `auto`)

Determines the tool choice strategy for the function.The supported tool choice strategies are:

- `none`: The function should not use any tools.
- `auto`: The model decides whether or not to use a tool. If it decides to use a tool, it also decides which tools to use.
- `required`: The model should use a tool. If multiple tools are available, the model decides which tool to use.
- `{ specific = "tool_name" }`: The model should use a specific tool. The tool must be defined in the `tools` field (see below).

Copy

```
// tensorzero.toml
[functions.solve-math-problem]
# ...
type = "chat"
tool_choice = "auto"
tools = [\
  # ...\
  "run-python"\
  # ...\
]
# ...

[tools.run-python]
# ...
```

Copy

```
// tensorzero.toml
[functions.generate-query]
# ...
type = "chat"
tool_choice = { specific = "query-database" }
tools = [\
  # ...\
  "query-database"\
  # ...\
]
# ...

[tools.query-database]
# ...
```

##### `tools`

- **Type:** array of strings
- **Required:** no (default: `[]`)

Determines the tools that the function can use.The supported tools are defined in `[tools.tool_name]` sections (see below).

Copy

```
// tensorzero.toml
[functions.draft-email]
# ...
type = "chat"
tools = [\
  # ...\
  "query-database"\
  # ...\
]
# ...

[tools.query-database]
# ...
```

type: "json"

##### `output_schema`

- **Type:** string (path)
- **Required:** no (default: `{}`, the empty JSON schema that accepts any valid JSON output)

Defines the path to the output schema file, which should contain a [JSON Schema](https://json-schema.org/) for the output of the function.
The path is relative to the configuration file.This schema is used for validating the output of the function.

tensorzero.toml

Copy

```
[functions.extract-customer-info]
# ...
type = "json"
output_schema = "./functions/extract-customer-info/output_schema.json"
# ...
```

See [Generate structured outputs](https://www.tensorzero.com/docs/gateway/generate-structured-outputs) for a comprehensive guide with examples.

### [​](https://www.tensorzero.com/docs/gateway/configuration-reference\#user-schema)  `user_schema`

- **Type:** string (path)
- **Required:** no

Defines the path to the user schema file.
The path is relative to the configuration file.If provided, the user schema file should contain a [JSON Schema](https://json-schema.org/) for the user messages.
The variables in the schema are used for templating the user messages.
If a schema is provided, all function variants must also provide a user template (see below).

tensorzero.toml

Copy

```
[functions.draft-email]
# ...
user_schema = "./functions/draft-email/user_schema.json"
# ...

[functions.draft-email.variants.prompt-v1]
# ...
user_template = "./functions/draft-email/prompt-v1/user_template.minijinja"
# ...
```

## [​](https://www.tensorzero.com/docs/gateway/configuration-reference\#[functions-function-name-variants-variant-name])  `[functions.function_name.variants.variant_name]`

The `variants` sub-section defines the behavior of a specific variant of a function.
You can define multiple variants by including multiple `[functions.function_name.variants.variant_name]` sections.If your `variant_name` is not a basic string, it can be escaped with quotation marks.
For example, periods are not allowed in basic strings, so you can define `llama-3.1-8b-instruct` as `[functions.function_name.variants."llama-3.1-8b-instruct"]`.

Copy

```
// tensorzero.toml
[functions.draft-email]
# ...

[functions.draft-email.variants."llama-3.1-8b-instruct"]
# ...

[functions.draft-email.variants.claude-3-haiku]
# ...
```

### [​](https://www.tensorzero.com/docs/gateway/configuration-reference\#type-4)  `type`

- **Type:** string
- **Required:** yes

Defines the type of the variant.TensorZero currently supports the following variant types:

| Type | Description |
| --- | --- |
| `chat_completion` | Uses a chat completion model to generate responses by processing a series of messages in a conversational format. This is typically what you use out of the box with most LLMs. |
| `experimental_best_of_n` | Generates multiple response candidates with other variants, and selects the best one using an evaluator model. |
| `experimental_chain_of_thought` | Encourages the model to reason step by step using a chain-of-thought prompting strategy, which is particularly useful for tasks requiring logical reasoning or multi-step problem-solving. Only available for non-streaming requests to JSON functions. |
| `experimental_dynamic_in_context_learning` | Selects similar high-quality examples using an embedding of the input, and incorporates them into the prompt to enhance context and improve response quality. |
| `experimental_mixture_of_n` | Generates multiple response candidates with other variants, and combines the responses using a fuser model. |

tensorzero.toml

Copy

```
[functions.draft-email.variants.prompt-v1]
# ...
type = "chat_completion"
# ...
```

type: "chat\_completion"

##### `assistant_template`

- **Type:** string (path)
- **Required:** no

Defines the path to the assistant template file.
The path is relative to the configuration file.This file should contain a [MiniJinja](https://docs.rs/minijinja/latest/minijinja/syntax/index.html) template for the assistant messages.
If the template uses any variables, the variables should be defined in the function’s `assistant_schema` field.

tensorzero.toml

Copy

```
[functions.draft-email]
# ...
assistant_schema = "./functions/draft-email/assistant_schema.json"
# ...

[functions.draft-email.variants.prompt-v1]
# ...
assistant_template = "./functions/draft-email/prompt-v1/assistant_template.minijinja"
# ...
```

##### `extra_body`

- **Type:** array of objects (see below)
- **Required:** no

The `extra_body` field allows you to modify the request body that TensorZero sends to a variant’s model provider.
This advanced feature is an “escape hatch” that lets you use provider-specific functionality that TensorZero hasn’t implemented yet.Each object in the array must have two fields:

- `pointer`: A [JSON Pointer](https://datatracker.ietf.org/doc/html/rfc6901) string specifying where to modify the request body
- One of the following:
  - `value`: The value to insert at that location; it can be of any type including nested types
  - `delete = true`: Deletes the field at the specified location, if present.

You can also set `extra_body` for a model provider entry.
The model provider `extra_body` entries take priority over variant `extra_body` entries.Additionally, you can set `extra_body` at inference-time.
The values provided at inference-time take priority over the values in the configuration file.

Example: \`extra\_body\`

If TensorZero would normally send this request body to the provider…

Copy

```
{
  "project": "tensorzero",
  "safety_checks": {
    "no_internet": false,
    "no_agi": true
  }
}
```

…then the following `extra_body`…

Copy

```
extra_body = [\
  { pointer = "/agi", value = true},\
  { pointer = "/safety_checks/no_agi", value = { bypass = "on" }}\
]
```

…overrides the request body to:

Copy

```
{
  "agi": true,
  "project": "tensorzero",
  "safety_checks": {
    "no_internet": false,
    "no_agi": {
      "bypass": "on"
    }
  }
}
```

##### `extra_headers`

- **Type:** array of objects (see below)
- **Required:** no

The `extra_headers` field allows you to set or overwrite the request headers that TensorZero sends to a model provider.
This advanced feature is an “escape hatch” that lets you use provider-specific functionality that TensorZero hasn’t implemented yet.Each object in the array must have two fields:

- `name` (string): The name of the header to modify (e.g. `anthropic-beta`)
- One of the following:
  - `value` (string): The value of the header (e.g. `token-efficient-tools-2025-02-19`)
  - `delete = true`: Deletes the header from the request, if present

You can also set `extra_headers` for a model provider entry.
The model provider `extra_headers` entries take priority over variant `extra_headers` entries.

Example: \`extra\_headers\`

If TensorZero would normally send the following request headers to the provider…

Copy

```
Safety-Checks: on
```

…then the following `extra_headers`…

Copy

```
extra_headers = [\
  { name = "Safety-Checks", value = "off"},\
  { name = "Intelligence-Level", value = "AGI"}\
]
```

…overrides the request headers to:

Copy

```
Safety-Checks: off
Intelligence-Level: AGI
```

##### `frequency_penalty`

- **Type:** float
- **Required:** no (default: `null`)

Penalizes new tokens based on their frequency in the text so far if positive, encourages them if negative.

tensorzero.toml

Copy

```
[functions.draft-email.variants.prompt-v1]
# ...
frequency_penalty = 0.2
# ...
```

##### `json_mode`

- **Type:** string
- **Required:** yes for `json` functions, forbidden for `chat` functions

Defines the strategy for generating JSON outputs.The supported modes are:

- `off`: Make a chat completion request without any special JSON handling (not recommended).
- `on`: Make a chat completion request with JSON mode (if supported by the provider).
- `strict`: Make a chat completion request with strict JSON mode (if supported by the provider). For example, the TensorZero Gateway uses Structured Outputs for OpenAI.
- `tool`: Make a special-purpose tool use request under the hood, and convert the tool call into a JSON response.

tensorzero.toml

Copy

```
[functions.draft-email.variants.prompt-v1]
# ...
json_mode = "strict"
# ...
```

See [Generate structured outputs](https://www.tensorzero.com/docs/gateway/generate-structured-outputs) for a comprehensive guide with examples.

##### `max_tokens`

- **Type:** integer
- **Required:** no (default: `null`)

Defines the maximum number of tokens to generate.

tensorzero.toml

Copy

```
[functions.draft-email.variants.prompt-v1]
# ...
max_tokens = 100
# ...
```

##### `model`

- **Type:** string
- **Required:** yes

The name of the model to call.

|     |     |
| --- | --- |
| **To call…** | **Use this format…** |
| A model defined as `[models.my_model]` in your <br>`tensorzero.toml`<br>configuration file | `model_name=“my_model”` |
| A model offered by a model provider, without defining it in your<br>`tensorzero.toml` configuration file (if supported, see<br>below) | `model_name="{provider_type}::{model_name}"` |

The following model providers support short-hand model names: `anthropic`, `deepseek`, `fireworks`, `google_ai_studio_gemini`, `gcp_vertex_gemini`, `gcp_vertex_anthropic`, `hyperbolic`, `groq`, `mistral`, `openai`, `openrouter`, `together`, and `xai`.

For example, if you have the following configuration:

tensorzero.toml

Copy

```
[models.gpt-4o]
routing = ["openai", "azure"]

[models.gpt-4o.providers.openai]
# ...

[models.gpt-4o.providers.azure]
# ...
```

Then:

- `model = "gpt-4o"` calls the `gpt-4o` model in your configuration, which supports fallback from `openai` to `azure`. See [Retries & Fallbacks](https://www.tensorzero.com/docs/gateway/guides/retries-fallbacks) for details.
- `model = "openai::gpt-4o"` calls the OpenAI API directly for the `gpt-4o` model using the Chat Completions API, ignoring the `gpt-4o` model defined above.
- `model = "openai::responses::gpt-5-codex"` calls the OpenAI Responses API directly for the `gpt-5-codex` model. See [OpenAI Responses API](https://www.tensorzero.com/docs/gateway/call-the-openai-responses-api) for details.

##### `presence_penalty`

- **Type:** float
- **Required:** no (default: `null`)

Penalizes new tokens based on that have already appeared in the text so far if positive, encourages them if negative.

tensorzero.toml

Copy

```
[functions.draft-email.variants.prompt-v1]
# ...
presence_penalty = 0.5
# ...
```

##### `reasoning_effort`

- **Type:** string
- **Required:** no (default: `null`)

Controls the reasoning effort level for reasoning models.

Only some model providers support this parameter. TensorZero will warn and ignore it if unsupported.

Some providers (e.g. Anthropic, Gemini) support `thinking_budget_tokens` instead.

tensorzero.toml

Copy

```
[functions.draft-email.variants.prompt-v1]
# ...
reasoning_effort = "medium"
# ...
```

##### `retries`

- **Type:** object with optional keys `num_retries` and `max_delay_s`
- **Required:** no (defaults to `num_retries = 0` and a `max_delay_s = 10`)

TensorZero’s retry strategy is truncated exponential backoff with jitter.
The `num_retries` parameter defines the number of retries (not including the initial request).
The `max_delay_s` parameter defines the maximum delay between retries.

tensorzero.toml

Copy

```
[functions.draft-email.variants.prompt-v1]
# ...
retries = { num_retries = 3, max_delay_s = 10 }
# ...
```

##### `seed`

- **Type:** integer
- **Required:** no (default: `null`)

Defines the seed to use for the variant.

tensorzero.toml

Copy

```
[functions.draft-email.variants.prompt-v1]
# ...
seed = 42
```

##### `service_tier`

- **Type:** string
- **Required:** no (default: `"auto"`)

Controls the priority and latency characteristics of inference requests.The supported values are:

- `auto`: Let the provider automatically select the appropriate service tier (default).
- `default`: Use the provider’s standard service tier.
- `priority`: Use a higher-priority service tier with lower latency (may have higher costs).
- `flex`: Use a lower-priority service tier optimized for cost efficiency (may have higher latency).

Only some model providers support this parameter.
TensorZero will warn and ignore it if unsupported.

##### `stop_sequences`

- **Type:** array of strings
- **Required:** no (default: `null`)

Defines a list of sequences where the model will stop generating further tokens.
When the model encounters any of these sequences in its output, it will immediately stop generation.

##### `system_template`

- **Type:** string (path)
- **Required:** no

Defines the path to the system template file.
The path is relative to the configuration file.This file should contain a [MiniJinja](https://docs.rs/minijinja/latest/minijinja/syntax/index.html) template for the system messages.
If the template uses any variables, the variables should be defined in the function’s `system_schema` field.

tensorzero.toml

Copy

```
[functions.draft-email]
# ...
system_schema = "./functions/draft-email/system_schema.json"
# ...

[functions.draft-email.variants.prompt-v1]
# ...
system_template = "./functions/draft-email/prompt-v1/system_template.minijinja"
# ...
```

##### `temperature`

- **Type:** float
- **Required:** no (default: `null`)

Defines the temperature to use for the variant.

tensorzero.toml

Copy

```
[functions.draft-email.variants.prompt-v1]
# ...
temperature = 0.5
# ...
```

##### `thinking_budget_tokens`

- **Type:** integer
- **Required:** no (default: `null`)

Controls the thinking budget in tokens for reasoning models.For Anthropic, this value corresponds to `thinking.budget_tokens`.
For Gemini, this value corresponds to `generationConfig.thinkingConfig.thinkingBudget`.

Only some model providers support this parameter. TensorZero will warn and ignore it if unsupported.

Some providers (e.g. OpenAI) support `reasoning_effort` instead.

tensorzero.toml

Copy

```
[functions.draft-email.variants.prompt-v1]
# ...
thinking_budget_tokens = 10000
# ...
```

##### `timeouts`

- **Type:** object
- **Required:** no

The `timeouts` object allows you to set granular timeouts for requests using this variant.You can define timeouts for non-streaming and streaming requests separately: `timeouts.non_streaming.total_ms` corresponds to the total request duration and `timeouts.streaming.ttft_ms` corresponds to the time to first token (TTFT).For example, the following configuration sets a 15-second timeout for non-streaming requests and a 3-second timeout for streaming requests (TTFT):

Copy

```
[functions.function_name.variants.variant_name]
# ...
timeouts = { non_streaming.total_ms = 15000, streaming.ttft_ms = 3000 }
# ...
```

The specified timeouts apply to the scope of an entire variant inference request, including all retries and fallbacks across its model’s providers.
You can also set timeouts at the model level and provider level.
Multiple timeouts can be active simultaneously.

##### `top_p`

- **Type:** float, between 0 and 1
- **Required:** no (default: `null`)

Defines the `top_p` to use for the variant during [nucleus sampling](https://en.wikipedia.org/wiki/Top-p_sampling).
Typically at most one of `top_p` and `temperature` is set.

tensorzero.toml

Copy

```
[functions.draft-email.variants.prompt-v1]
# ...
top_p = 0.3
# ...
```

##### `verbosity`

- **Type:** string
- **Required:** no (default: `null`)

Controls the verbosity level of model outputs.

Only some model providers support this parameter. TensorZero will warn and ignore it if unsupported.

tensorzero.toml

Copy

```
[functions.draft-email.variants.prompt-v1]
# ...
verbosity = "low"
# ...
```

##### `user_template`

- **Type:** string (path)
- **Required:** no

Defines the path to the user template file.
The path is relative to the configuration file.This file should contain a [MiniJinja](https://docs.rs/minijinja/latest/minijinja/syntax/index.html) template for the user messages.
If the template uses any variables, the variables should be defined in the function’s `user_schema` field.

tensorzero.toml

Copy

```
[functions.draft-email]
# ...
user_schema = "./functions/draft-email/user_schema.json"
# ...

[functions.draft-email.variants.prompt-v1]
# ...
user_template = "./functions/draft-email/prompt-v1/user_template.minijinja"
# ...
```

type: "experimental\_best\_of\_n"

##### `candidates`

- **Type:** list of strings
- **Required:** yes

This inference strategy generates N candidate responses, and an evaluator model selects the best one.
This approach allows you to leverage multiple prompts or variants to increase the likelihood of getting a high-quality response.The `candidates` parameter specifies a list of variant names used to generate candidate responses.
For example, if you have two variants defined (`promptA` and `promptB`), you could set up the `candidates` list to generate two responses using `promptA` and one using `promptB` using the snippet below.
The evaluator would then choose the best response from these three candidates.

tensorzero.toml

Copy

```
[functions.draft-email.variants.promptA]
type = "chat_completion"
# ...

[functions.draft-email.variants.promptB]
type = "chat_completion"
# ...

[functions.draft-email.variants.best-of-n]
type = "experimental_best_of_n"
candidates = ["promptA", "promptA", "promptB"] # 3 candidate generations
# ...
```

##### `evaluator`

- **Type:** object
- **Required:** yes

The `evaluator` parameter specifies the configuration for the model that will evaluate and select the best response from the generated candidates.The evaluator is configured similarly to a `chat_completion` variant for a JSON function, but without the `type` field.
The prompts here should be prompts that you would use to solve the original problem, as the gateway has special-purpose handling and templates to convert them to an evaluator.The evaluator can optionally include a `json_mode` parameter (see the `json_mode` documentation under `chat_completion` variants). If not specified, it defaults to `strict`.

Copy

```
[functions.draft-email.variants.best-of-n]
type = "experimental_best_of_n"
# ...

[functions.draft-email.variants.best-of-n.evaluator]
# Same fields as a `chat_completion` variant (excl.`type`), e.g.:
# user_template = "functions/draft-email/best-of-n/user.minijinja"
# ...
```

##### `timeout_s`

- **Type:** float
- **Required:** no (default: 300s)

The `timeout_s` parameter specifies the maximum time in seconds allowed for generating candidate responses.
Any candidate that takes longer than this duration to generate a response will be dropped from consideration.

Copy

```
[functions.draft-email.variants.best-of-n]
type = "experimental_best_of_n"
timeout_s = 60
# ...
```

##### `timeouts`

- **Type:** object
- **Required:** no

The `timeouts` object allows you to set granular timeouts for requests using this variant.You can define timeouts for non-streaming and streaming requests separately: `timeouts.non_streaming.total_ms` corresponds to the total request duration and `timeouts.streaming.ttft_ms` corresponds to the time to first token (TTFT).For example, the following configuration sets a 15-second timeout for non-streaming requests and a 3-second timeout for streaming requests (TTFT):

Copy

```
[functions.function_name.variants.variant_name]
# ...
timeouts = { non_streaming.total_ms = 15000, streaming.ttft_ms = 3000 }
# ...
```

The specified timeouts apply to the scope of an entire variant inference request, including all inference requests to candidates and the evaluator.
You can also set timeouts at the model level and provider level.
Multiple timeouts can be active simultaneously.

type: "experimental\_chain\_of\_thought"

The `experimental_chain_of_thought` variant type uses the same configuration as a `chat_completion` variant.

This variant type is only available for non-streaming requests to JSON functions.

type: "experimental\_mixture\_of\_n"

##### `candidates`

- **Type:** list of strings
- **Required:** yes

This inference strategy generates N candidate responses, and a fuser model combines them to produce a final answer.
This approach allows you to leverage multiple prompts or variants to increase the likelihood of getting a high-quality response.The `candidates` parameter specifies a list of variant names used to generate candidate responses.
For example, if you have two variants defined (`promptA` and `promptB`), you could set up the `candidates` list to generate two responses using `promptA` and one using `promptB` using the snippet below.
The fuser would then combine the three responses.

tensorzero.toml

Copy

```
[functions.draft-email.variants.promptA]
type = "chat_completion"
# ...

[functions.draft-email.variants.promptB]
type = "chat_completion"
# ...

[functions.draft-email.variants.mixture-of-n]
type = "experimental_mixture_of_n"
candidates = ["promptA", "promptA", "promptB"] # 3 candidate generations
# ...
```

##### `fuser`

- **Type:** object
- **Required:** yes for `json` functions, forbidden for `chat` functions

The `fuser` parameter specifies the configuration for the model that will evaluate and combine the elements.The fuser is configured similarly to a `chat_completion` variant, but without the `type` field.
The prompts here should be prompts that you would use to solve the original problem, as the gateway has special-purpose handling and templates to convert them to a fuser.

Copy

```
[functions.draft-email.variants.mixture-of-n]
type = "experimental_mixture_of_n"
# ...

[functions.draft-email.variants.mixture-of-n.fuser]
# Same fields as a `chat_completion` variant (excl.`type`), e.g.:
# user_template = "functions/draft-email/mixture-of-n/user.minijinja"
# ...
```

##### `timeout_s`

- **Type:** float
- **Required:** no (default: 300s)

The `timeout_s` parameter specifies the maximum time in seconds allowed for generating candidate responses.
Any candidate that takes longer than this duration to generate a response will be dropped from consideration.

Copy

```
[functions.draft-email.variants.mixture-of-n]
type = "experimental_mixture_of_n"
timeout_s = 60
# ...
```

##### `timeouts`

- **Type:** object
- **Required:** no

The `timeouts` object allows you to set granular timeouts for requests using this variant.You can define timeouts for non-streaming and streaming requests separately: `timeouts.non_streaming.total_ms` corresponds to the total request duration and `timeouts.streaming.ttft_ms` corresponds to the time to first token (TTFT).For example, the following configuration sets a 15-second timeout for non-streaming requests and a 3-second timeout for streaming requests (TTFT):

Copy

```
[functions.function_name.variants.variant_name]
# ...
timeouts = { non_streaming.total_ms = 15000, streaming.ttft_ms = 3000 }
# ...
```

The specified timeouts apply to the scope of an entire variant inference request, including all inference requests to candidates and the fuser.
You can also set timeouts at the model level and provider level.
Multiple timeouts can be active simultaneously.

type: "experimental\_dynamic\_in\_context\_learning"

##### `embedding_model`

- **Type:** string
- **Required:** yes

The name of the embedding model to call.

|     |     |
| --- | --- |
| **To call…** | **Use this format…** |
| A model defined as `[models.my_model]` in your <br>`tensorzero.toml`<br>configuration file | `model_name=“my_model”` |
| A model offered by a model provider, without defining it in your<br>`tensorzero.toml` configuration file (if supported, see<br>below) | `model_name="{provider_type}::{model_name}"` |

The following model providers support short-hand model names: `anthropic`, `deepseek`, `fireworks`, `google_ai_studio_gemini`, `gcp_vertex_gemini`, `gcp_vertex_anthropic`, `hyperbolic`, `groq`, `mistral`, `openai`, `openrouter`, `together`, and `xai`.

For example, if you have the following configuration:

tensorzero.toml

Copy

```
[embedding_models.text-embedding-3-small]
#...

[embedding_models.text-embedding-3-small.providers.openai]
# ...

[embedding_models.text-embedding-3-small.providers.azure]
# ...
```

Then:

- `embedding_model = "text-embedding-3-small"` calls the `text-embedding-3-small` model in your configuration.
- `embedding_model = "openai::text-embedding-3-small"` calls the OpenAI API directly for the `text-embedding-3-small` model, ignoring the `text-embedding-3-small` model defined above.

##### `extra_body`

- **Type:** array of objects (see below)
- **Required:** no

The `extra_body` field allows you to modify the request body that TensorZero sends to a variant’s model provider.
This advanced feature is an “escape hatch” that lets you use provider-specific functionality that TensorZero hasn’t implemented yet.For `experimental_dynamic_in_context_learning` variants, `extra_body` only applies to the chat completion request.Each object in the array must have two fields:

- `pointer`: A [JSON Pointer](https://datatracker.ietf.org/doc/html/rfc6901) string specifying where to modify the request body
- One of the following:
  - `value`: The value to insert at that location; it can be of any type including nested types
  - `delete = true`: Deletes the field at the specified location, if present.

You can also set `extra_body` for a model provider entry.
The model provider `extra_body` entries take priority over variant `extra_body` entries.Additionally, you can set `extra_body` at inference-time.
The values provided at inference-time take priority over the values in the configuration file.

Example: \`extra\_body\`

If TensorZero would normally send this request body to the provider…

Copy

```
{
  "project": "tensorzero",
  "safety_checks": {
    "no_internet": false,
    "no_agi": true
  }
}
```

…then the following `extra_body`…

Copy

```
extra_body = [\
  { pointer = "/agi", value = true},\
  { pointer = "/safety_checks/no_agi", value = { bypass = "on" }}\
]
```

…overrides the request body to:

Copy

```
{
  "agi": true,
  "project": "tensorzero",
  "safety_checks": {
    "no_internet": false,
    "no_agi": {
      "bypass": "on"
    }
  }
}
```

##### `extra_headers`

- **Type:** array of objects (see below)
- **Required:** no

The `extra_headers` field allows you to set or overwrite the request headers that TensorZero sends to a model provider.
This advanced feature is an “escape hatch” that lets you use provider-specific functionality that TensorZero hasn’t implemented yet.Each object in the array must have two fields:

- `name` (string): The name of the header to modify (e.g. `anthropic-beta`)
- One of the following:
  - `value` (string): The value of the header (e.g. `token-efficient-tools-2025-02-19`)
  - `delete = true`: Deletes the header from the request, if present

You can also set `extra_headers` for a model provider entry.
The model provider `extra_headers` entries take priority over variant `extra_headers` entries.

Example: \`extra\_headers\`

If TensorZero would normally send the following request headers to the provider…

Copy

```
Safety-Checks: on
```

…then the following `extra_headers`…

Copy

```
extra_headers = [\
  { name = "Safety-Checks", value = "off"},\
  { name = "Intelligence-Level", value = "AGI"}\
]
```

…overrides the request headers to:

Copy

```
Safety-Checks: off
Intelligence-Level: AGI
```

##### `json_mode`

- **Type:** string
- **Required:** yes for `json` functions, forbidden for `chat` functions

Defines the strategy for generating JSON outputs.The supported modes are:

- `off`: Make a chat completion request without any special JSON handling (not recommended).
- `on`: Make a chat completion request with JSON mode (if supported by the provider).
- `strict`: Make a chat completion request with strict JSON mode (if supported by the provider). For example, the TensorZero Gateway uses Structured Outputs for OpenAI.
- `tool`: Make a special-purpose tool use request under the hood, and convert the tool call into a JSON response.

tensorzero.toml

Copy

```
[functions.draft-email.variants.prompt-v1]
# ...
json_mode = "strict"
# ...
```

##### `k`

- **Type:** non-negative integer
- **Required:** yes

Defines the number of examples to retrieve for the inference.

tensorzero.toml

Copy

```
[functions.draft-email.variants.dicl]
# ...
k = 10
# ...
```

##### `max_distance`

- **Type:** non-negative float
- **Required:** no (default: none)

Filters retrieved examples based on their cosine distance from the input embedding.
Only examples with a cosine distance less than or equal to the specified threshold are included in the prompt.If all examples are filtered out due to this threshold, the variant falls back to default chat completion behavior.

##### `max_tokens`

- **Type:** integer
- **Required:** no (default: `null`)

Defines the maximum number of tokens to generate.

tensorzero.toml

Copy

```
[functions.draft-email.variants.prompt-v1]
# ...
max_tokens = 100
# ...
```

##### `model`

- **Type:** string
- **Required:** yes

The name of the model to call.

|     |     |
| --- | --- |
| **To call…** | **Use this format…** |
| A model defined as `[models.my_model]` in your <br>`tensorzero.toml`<br>configuration file | `model_name=“my_model”` |
| A model offered by a model provider, without defining it in your<br>`tensorzero.toml` configuration file (if supported, see<br>below) | `model_name="{provider_type}::{model_name}"` |

The following model providers support short-hand model names: `anthropic`, `deepseek`, `fireworks`, `google_ai_studio_gemini`, `gcp_vertex_gemini`, `gcp_vertex_anthropic`, `hyperbolic`, `groq`, `mistral`, `openai`, `openrouter`, `together`, and `xai`.

For example, if you have the following configuration:

tensorzero.toml

Copy

```
[models.gpt-4o]
routing = ["openai", "azure"]

[models.gpt-4o.providers.openai]
# ...

[models.gpt-4o.providers.azure]
# ...
```

Then:

- `model = "gpt-4o"` calls the `gpt-4o` model in your configuration, which supports fallback from `openai` to `azure`. See [Retries & Fallbacks](https://www.tensorzero.com/docs/gateway/guides/retries-fallbacks) for details.
- `model = "openai::gpt-4o"` calls the OpenAI API directly for the `gpt-4o` model using the Chat Completions API, ignoring the `gpt-4o` model defined above.
- `model = "openai::responses::gpt-5-codex"` calls the OpenAI Responses API directly for the `gpt-5-codex` model. See [OpenAI Responses API](https://www.tensorzero.com/docs/gateway/call-the-openai-responses-api) for details.

##### `retries`

- **Type:** object with optional keys `num_retries` and `max_delay_s`
- **Required:** no (defaults to `num_retries = 0` and a `max_delay_s = 10`)

TensorZero’s retry strategy is truncated exponential backoff with jitter.
The `num_retries` parameter defines the number of retries (not including the initial request).
The `max_delay_s` parameter defines the maximum delay between retries.

tensorzero.toml

Copy

```
[functions.draft-email.variants.prompt-v1]
# ...
retries = { num_retries = 3, max_delay_s = 10 }
# ...
```

##### `seed`

- **Type:** integer
- **Required:** no (default: `null`)

Defines the seed to use for the variant.

tensorzero.toml

Copy

```
[functions.draft-email.variants.prompt-v1]
# ...
seed = 42
```

##### `system_instructions`

- **Type:** string (path)
- **Required:** no

Defines the path to the system instructions file.
The path is relative to the configuration file.The system instruction is a text file that will be added to the evaluator’s system prompt.
Unlike `system_template`, it doesn’t support variables.
This file contains static instructions that define the behavior and role of the AI assistant for the specific function variant.

tensorzero.toml

Copy

```
[functions.draft-email.variants.dicl]
# ...
system_instructions = "./functions/draft-email/prompt-v1/system_template.txt"
# ...
```

##### `temperature`

- **Type:** float
- **Required:** no (default: `null`)

Defines the temperature to use for the variant.

tensorzero.toml

Copy

```
[functions.draft-email.variants.prompt-v1]
# ...
temperature = 0.5
# ...
```

##### `timeouts`

- **Type:** object
- **Required:** no

The `timeouts` object allows you to set granular timeouts for requests using this variant.You can define timeouts for non-streaming and streaming requests separately: `timeouts.non_streaming.total_ms` corresponds to the total request duration and `timeouts.streaming.ttft_ms` corresponds to the time to first token (TTFT).For example, the following configuration sets a 15-second timeout for non-streaming requests and a 3-second timeout for streaming requests (TTFT):

Copy

```
[functions.function_name.variants.variant_name]
# ...
timeouts = { non_streaming.total_ms = 15000, streaming.ttft_ms = 3000 }
# ...
```

The specified timeouts apply to the scope of an entire variant inference request, including both inference requests to the embedding model and the generation model.
You can also set timeouts at the model level and provider level.
Multiple timeouts can be active simultaneously.

#### [​](https://www.tensorzero.com/docs/gateway/configuration-reference\#type:-%22experimental-chain-of-thought%22)  `type: "experimental_chain_of_thought"`

Besides the type parameter, this variant has the same configuration options as the `chat_completion` variant type.
Please refer to that documentation to see what options are available.

## [​](https://www.tensorzero.com/docs/gateway/configuration-reference\#[functions-function-name-experimentation])  `[functions.function_name.experimentation]`

This section configures experimentation (A/B testing) over a set of variants in a function.At inference time, the gateway will sample a variant from the function to complete the request.
By default, the gateway will sample a variant uniformly at random (`type = "uniform"`).TensorZero supports multiple types of experiments that can help you learn about the relative performance of the variants.

tensorzero.toml

Copy

```
[functions.draft-email.experimentation]
# fieldA = ...
# fieldB = ...
# ...
```

### [​](https://www.tensorzero.com/docs/gateway/configuration-reference\#type-5)  `type`

- **Type:** string
- **Required:** yes

Determines the experiment type.TensorZero currently supports the following experiment types:

| Type | Description |
| --- | --- |
| `uniform` | Samples variants uniformly at random. For example, if there are three candidate variants, each will be sampled with probability `1/3`. |
| `static_weights` | Samples variants according to user-specified weights. Weights must be nonnegative and are normalized to sum to 1. See the `candidate_variants` documentation below for how to specify weights. |
| `track_and_stop` | Samples variants according to probabilities that dynamically update based on accumulating feedback data. Designed to maximize experiment efficiency by minimizing the number of inferences needed to identify the best variant. |

tensorzero.toml

Copy

```
[functions.draft-email.experimentation]
# ...
type = "track_and_stop"
# ...
```

type: "uniform"

The `uniform` type samples variants uniformly at random.
This is the default behavior when no `[functions.function_name.experimentation]` section is specified.By default, all variants defined in the function are sampled with equal probability.
You can optionally specify `candidate_variants` to sample uniformly from a subset of variants, and `fallback_variants` for sequential fallback behavior.
The behavior depends on which fields are specified:

| Configuration | Behavior |
| --- | --- |
| No fields specified | Samples uniformly from all variants in the function |
| Only `candidate_variants` | Samples uniformly from specified candidates |
| Only `fallback_variants` | Uses fallback variants sequentially (no uniform sampling) |
| Both specified | Samples uniformly from candidates; if all fail, uses fallbacks sequentially |

### [​](https://www.tensorzero.com/docs/gateway/configuration-reference\#candidate-variants)  `candidate_variants`

- **Type:** array of strings
- **Required:** no

An optional set of variants to sample uniformly from.
Each variant must be defined via `[functions.function_name.variants.variant_name]` in the `variants` sub-section.If not specified (and `fallback_variants` is also not specified), all variants are sampled uniformly.
If `fallback_variants` is specified but `candidate_variants` is not, no candidates are used (fallback-only mode).

tensorzero.toml

Copy

```
[functions.draft-email.experimentation]
type = "uniform"
candidate_variants = ["variant-a", "variant-b"]
```

### [​](https://www.tensorzero.com/docs/gateway/configuration-reference\#fallback-variants)  `fallback_variants`

- **Type:** array of strings
- **Required:** no

An optional set of function variants to use as fallback options.
Each variant must be defined via `[functions.function_name.variants.variant_name]` in the `variants` sub-section.If all candidate variants fail during inference, the gateway will select variants sequentially from `fallback_variants` (in order, not uniformly).
This behaves like a ranked list where the first active fallback variant is always selected.

tensorzero.toml

Copy

```
[functions.draft-email.experimentation]
type = "uniform"
candidate_variants = ["variant-a", "variant-b"]
fallback_variants = ["fallback-variant"]
```

### [​](https://www.tensorzero.com/docs/gateway/configuration-reference\#examples)  Examples

**Default uniform sampling (all variants):**

tensorzero.toml

Copy

```
[functions.draft-email]
type = "chat"

[functions.draft-email.variants.variant-a] # 1/3 chance
# ...

[functions.draft-email.variants.variant-b] # 1/3 chance
# ...

[functions.draft-email.variants.variant-c] # 1/3 chance
# ...
```

**Explicit candidate variants:**

tensorzero.toml

Copy

```
[functions.draft-email.experimentation]
type = "uniform"
candidate_variants = ["variant-a", "variant-b"]  # each has 1/2 probability
# `variant-c` will not be sampled
```

**With fallback variants:**

tensorzero.toml

Copy

```
[functions.draft-email.experimentation]
type = "uniform"
candidate_variants = ["variant-a", "variant-b"]  # try these first, uniformly
fallback_variants = ["variant-c"]  # use if both candidates fail
```

**Fallback-only mode:**

tensorzero.toml

Copy

```
[functions.draft-email.experimentation]
type = "uniform"
fallback_variants = ["variant-a", "variant-b", "variant-c"]  # sequential
```

type: "static\_weights"

The `static_weights` type samples variants according to user-specified weights.
This allows you to control the distribution of traffic across variants with fixed probabilities.

### [​](https://www.tensorzero.com/docs/gateway/configuration-reference\#candidate-variants-2)  `candidate_variants`

- **Type:** map of strings to floats
- **Required:** yes

A map from variant names to their sampling weights.
Each variant must be defined via `[functions.function_name.variants.variant_name]` in the `variants` sub-section.Weights must be non-negative.
The gateway automatically normalizes the weights to sum to 1.0.
For example, weights of `{"variant-a" = 5.0, "variant-b" = 1.0}` result in sampling probabilities of `5/6` and `1/6` respectively.

tensorzero.toml

Copy

```
[functions.draft-email.experimentation]
type = "static_weights"
candidate_variants = {"prompt-v1" = 5.0, "prompt-v2" = 1.0}
# ...
```

### [​](https://www.tensorzero.com/docs/gateway/configuration-reference\#fallback-variants-2)  `fallback_variants`

- **Type:** array of strings
- **Required:** no

An optional set of function variants to use as fallback options.Each variant must be defined via `[functions.function_name.variants.variant_name]` in the `variants` sub-section.
If all candidate variants fail during inference, or if the total weight of active candidate variants is zero, the gateway will sample uniformly at random from `fallback_variants`.

tensorzero.toml

Copy

```
[functions.draft-email.experimentation]
type = "static_weights"
candidate_variants = {"prompt-v1" = 2.0, "prompt-v2" = 1.0, "prompt-v3" = 0.5}
fallback_variants = ["fallback-prompt-a", "fallback-prompt-b"]
```

type: "track\_and\_stop"

### [​](https://www.tensorzero.com/docs/gateway/configuration-reference\#candidate-variants-3)  `candidate_variants`

- **Type:** array of strings
- **Required:** yes

The set of function variants to include in the experiment.
Each variant must be defined via `[functions.function_name.variants.variant_name]` in the the `variants` sub-section (see above).
Variants that are not included in `candidate_variants` will not be sampled.

tensorzero.toml

Copy

```
[functions.draft-email.experimentation]
# ...
candidate_variants = ["prompt-v1", "prompt-v2", "prompt-v3"]
# ...
```

##### `delta`

- **Type:** float
- **Required:** no (default: 0.05)

This field is for advanced users. The default value is sensible for most use cases.

The error tolerance.
The value of `delta` must be a probability in the `(0, 1)` range.In simple terms, `delta` is the probability that the algorithm will incorrectly identify a variant as the winner.
A commonly used value in experimentation settings is `0.05`, which caps the probability that an epsilon-best variant is not chosen as the winner at 5%.The `track_and_stop` algorithm aims to identify a “winner” variant that has the best average value for the chosen metric, or nearly the best (where “best” means highest if `optimize = "max"` or lowest if `optimize = "min"` for the chosen metric, and “nearly” is determined by a tolerance `epsilon`, defined below).
Once this variant is identified, random sampling ceases and the winner variant is used exclusively going forward.
The value `delta` instantiates a trade-off between the speed of identification and the confidence in the identified variant.
The smaller the value of `delta`, the higher the chance that the algorithm will correctly identify an epsilon-best variant, and the more data required to do so.

tensorzero.toml

Copy

```
[functions.draft-email.experimentation]
# ...
delta = 0.05
# ...
```

##### `epsilon`

- **Type:** float
- **Required:** no (default: 0.0)

This field is for advanced users. The default value is sensible for most use cases.

The sub-optimality tolerance.
The value must be nonnegative.The `track_and_stop` algorithm aims to identify a “winner” variant whose average metric value is either the highest, or within epsilon of the highest.
Larger values of `epsilon` allow the algorithm to label a winner more quickly.
As an example, consider an experiment over three function variants with underlying (unknown) mean metric values of `[0.6, 0.8, 0.85]` for a metric with `optimize = "max"`.
If `delta = 0.05` and `epsilon = 0.05`, then the algorithm will label either the second or third variant as the winner with probability at least `1 - delta = 95%`.
If `delta = 0.05` and `epsilon = 0`, then the experiment will run longer and the algorithm will label the third variant as the winner with probability at least `95%`.
If `delta = 0.01` and `epsilon = 0`, then the experiment will run for even longer, and the algorithm will label the third variant as the winner with probability at least 99%.It is always possible to set `epsilon = 0` to insist on identifying the strictly best variant with high probability.
Reasonable nonzero values of `epsilon` depend on the scale of the chosen metric.

tensorzero.toml

Copy

```
[functions.draft-email.experimentation]
# ...
epsilon = 0.03
# ...
```

### [​](https://www.tensorzero.com/docs/gateway/configuration-reference\#fallback-variants-3)  `fallback_variants`

- **Type:** array of string
- **Required:** no

An optional set of function variants to use as fallback options.Each variant must be defined via `[functions.function_name.variants.variant_name]` in the the `variants` sub-section (see above).
If inference fails with all of the `candidate_variants`, then variants will be sampled uniformly at random from `fallback_variants`.Feedback for these variants will not be used in the experiment itself; for example, if the experiment type is `track_and_stop`, the sampling probabilities will be dynamically updated based only on feedback for the `candidate_variants`.

tensorzero.toml

Copy

```
[functions.draft-email.experimentation]
candidate_variants = ["prompt-v1", "prompt-v2", "prompt-v3"]
fallback_variants = ["fallback-prompt-a", "fallback-prompt-b"]
# ...
```

##### `metric`

- **Type:** string
- **Required:** yes

The metric that should be tracked during the experiment.
The metric is used to dynamically update the sampling probabilities for the variants in a way that is designed to quickly identify high performing variants.This must be one of the metrics defined in the `[metrics]` section.
`track_and_stop` can handle both inference-level and episode-level metrics.
Plots based on the chosen metric are displayed in the `Experimentation` section of the `Functions` tab in the TensorZero UI.

tensorzero.toml

Copy

```
[functions.draft-email.experimentation]
# ...
metric = "task-completed"
# ...
```

##### `min_prob`

- **Type:** float
- **Required:** no (default: `0`)

This field is for advanced users. The default value is sensible for most use cases.

The minimum sampling probability for each candidate variant.
The value must be nonnegative.
Note that `min_prob` times the number of `candidate_variants` must not exceed 1.0, since the minimum probabilities for all candidate variants must sum to at most 1.0.The aim of a `track_and_stop` experiment is to identify an epsilon-best variant, without necessarily differentiating sub-optimal variants, so the primary use for this field is to enable the user to ensure that sufficient data is gathered to learn about the performance of sub-optimal variants.
Note that this field has no effect once `track_and_stop` picks a winner variant, since at that point random sampling ceases and the winner variant is used exclusively.

tensorzero.toml

Copy

```
[functions.draft-email.experimentation]
# ...
min_prob = 0.05
# ...
```

##### `min_samples_per_variant`

- **Type:** integer
- **Required:** no (default: 10)

This field is for advanced users. The default value is sensible for most use cases.

The minimum number of samples per variant required before random sampling begins.
The value must be greater than or equal to 1.
Sampling from the `candidate_variants` will proceed round-robin (deterministically) until each variant has at least `min_samples_per_variant` feedback data points, at which point random sampling will begin.
It is strongly recommended to set this value to at least 10 so that the feedback sample statistics can stabilize before they are used to guide the sampling probabilities.

tensorzero.toml

Copy

```
[functions.draft-email.experimentation]
# ...
min_samples_per_variant = 10
# ...
```

##### `update_period_s`

- **Type:** integer
- **Required:** no (default: 300)

This field is for advanced users. The default value is sensible for most use cases.

The frequency, in seconds, with which sampling probabilities are updated.Lower values will lead to faster experiment convergence but will consume more computational resources.Updating the sampling probabilities requires reading the latest feedback data from ClickHouse.
This is accomplished by a background task that interacts with the gateway instance.
More frequent updates (smaller values of `update_period_s`) relative to the feedback throughput enable the algorithm to more quickly guide the sampling probabilities toward their theoretical optimum, which allows it to more quickly label the “winner” variant.
For example, updating the sampling probabilities every ~100 inferences should lead to faster convergence than updating them every ~500 inferences.

tensorzero.toml

Copy

```
[functions.draft-email.experimentation]
# ...
update_period_s = 300
# ...
```

## [​](https://www.tensorzero.com/docs/gateway/configuration-reference\#[metrics])  `[metrics]`

The `[metrics]` section defines the behavior of a metric.
You can define multiple metrics by including multiple `[metrics.metric_name]` sections.The metric name can’t be `comment` or `demonstration`, as those names are reserved for internal use.If your `metric_name` is not a basic string, it can be escaped with quotation marks.
For example, periods are not allowed in basic strings, so you can define `beats-gpt-4.1` as `[metrics."beats-gpt-4.1"]`.

tensorzero.toml

Copy

```
[metrics.task-completed]
# fieldA = ...
# fieldB = ...
# ...

[metrics.user-rating]
# fieldA = ...
# fieldB = ...
# ...
```

### [​](https://www.tensorzero.com/docs/gateway/configuration-reference\#level)  `level`

- **Type:** string
- **Required:** yes

Defines whether the metric applies to individual inference or across entire episodes.The supported levels are `inference` and `episode`.

tensorzero.toml

Copy

```
[metrics.valid-output]
# ...
level = "inference"
# ...

[metrics.task-completed]
# ...
level = "episode"
# ...
```

### [​](https://www.tensorzero.com/docs/gateway/configuration-reference\#optimize)  `optimize`

- **Type:** string
- **Required:** yes

Defines whether the metric should be maximized or minimized.The supported values are `max` and `min`.

tensorzero.toml

Copy

```
[metrics.mistakes-made]
# ...
optimize = "min"
# ...

[metrics.user-rating]
# ...
optimize = "max"
# ...
```

### [​](https://www.tensorzero.com/docs/gateway/configuration-reference\#type-6)  `type`

- **Type:** string
- **Required:** yes

Defines the type of the metric.The supported metric types are `boolean` and `float`.

tensorzero.toml

Copy

```
[metrics.user-rating]
# ...
type = "float"
# ...

[metrics.task-completed]
# ...
type = "boolean"
# ...
```

## [​](https://www.tensorzero.com/docs/gateway/configuration-reference\#[tools-tool-name])  `[tools.tool_name]`

The `[tools.tool_name]` section defines the behavior of a tool.
You can define multiple tools by including multiple `[tools.tool_name]` sections.If your `tool_name` is not a basic string, it can be escaped with quotation marks.
For example, periods are not allowed in basic strings, so you can define `run-python-3.10` as `[tools."run-python-3.10"]`.You can enable a tool for a function by adding it to the function’s `tools` field.

Copy

```
// tensorzero.toml
[functions.weather-chatbot]
# ...
type = "chat"
tools = [\
  # ...\
  "get-temperature"\
  # ...\
]
# ...

[tools.get-temperature]
# ...
```

### [​](https://www.tensorzero.com/docs/gateway/configuration-reference\#description-2)  `description`

- **Type:** string
- **Required:** yes

Defines the description of the tool provided to the model.You can typically materially improve the quality of responses by providing a detailed description of the tool.

tensorzero.toml

Copy

```
[tools.get-temperature]
# ...
description = "Get the current temperature in a given location (e.g. \"Tokyo\") using the specified unit (must be \"celsius\" or \"fahrenheit\")."
# ...
```

### [​](https://www.tensorzero.com/docs/gateway/configuration-reference\#parameters)  `parameters`

- **Type:** string (path)
- **Required:** yes

Defines the path to the parameters file.
The path is relative to the configuration file.This file should contain a [JSON Schema](https://json-schema.org/) for the parameters of the tool.

tensorzero.toml

Copy

```
[tools.get-temperature]
# ...
parameters = "./tools/get-temperature.json"
# ...
```

### [​](https://www.tensorzero.com/docs/gateway/configuration-reference\#strict)  `strict`

- **Type:** boolean
- **Required:** no (default: `false`)

If set to `true`, the TensorZero Gateway attempts to use strict JSON generation for the tool parameters.
This typically improves the quality of responses.Only a few providers support strict JSON generation.
For example, the TensorZero Gateway uses Structured Outputs for OpenAI.
If the provider does not support strict mode, the TensorZero Gateway ignores this field.

tensorzero.toml

Copy

```
[tools.get-temperature]
# ...
strict = true
# ...
```

### [​](https://www.tensorzero.com/docs/gateway/configuration-reference\#name)  `name`

- **Type:** string
- **Required:** no (defaults to the tool ID)

Defines the tool name to be sent to model providers.By default, TensorZero will use the tool ID in the configuration as the tool name sent to model providers.
For example, if you define a tool as `[tools.my_tool]` but don’t specify the `name`, the name will be `my_tool`.
This field allows you to specify a different name to be sent.This field is particularly useful if you want to define multiple tools that share the same name (e.g. for different functions).
At inference time, the gateway ensures that an inference request doesn’t have multiple tools with the same name.

## [​](https://www.tensorzero.com/docs/gateway/configuration-reference\#[object-storage])  `[object_storage]`

The `[object_storage]` section defines the behavior of object storage, which is used for storing images used during multimodal inference.

### [​](https://www.tensorzero.com/docs/gateway/configuration-reference\#type-7)  `type`

- **Type:** string
- **Required:** yes

Defines the type of object storage to use.The supported types are:

- `s3_compatible`: Use an S3-compatible object storage service.
- `filesystem`: Store images in a local directory.
- `disabled`: Disable object storage.

See the following sections for more details on each type.

type: "s3\_compatible"

If you set `type = "s3_compatible"`, TensorZero will use an S3-compatible object storage service to store and retrieve images.The TensorZero Gateway will attempt to retrieve credentials from the following resources in order of priority:

1. `S3_ACCESS_KEY_ID` and `S3_SECRET_ACCESS_KEY` environment variables
2. `AWS_ACCESS_KEY_ID` and `AWS_SECRET_ACCESS_KEY` environment variables
3. Credentials from the AWS SDK (default profile)

If you set `type = "s3_compatible"`, the following fields are available.

##### `endpoint`

- **Type:** string
- **Required:** no (defaults to AWS S3)

Defines the endpoint of the object storage service.
You can use this field to specify a custom endpoint for the object storage service (e.g. GCP Cloud Storage, Cloudflare R2, and many more).

##### `bucket_name`

- **Type:** string
- **Required:** no

Defines the name of the bucket to use for object storage.
You should provide a bucket name unless it’s specified in the `endpoint` field.

##### `region`

- **Type:** string
- **Required:** no

Defines the region of the object storage service (if applicable).This is required for some providers (e.g. AWS S3).
If the provider does not require a region, this field can be omitted.

##### `allow_http`

- **Type:** boolean
- **Required:** no (defaults to `false`)

Normally, the TensorZero Gateway will require HTTPS to access the object storage service.
If set to `true`, the TensorZero Gateway will instead use HTTP to access the object storage service.
This is useful for local development (e.g. a local MinIO deployment), but not recommended for production environments.

For production environments, we strongly recommend you disable the `allow_http` setting and use a secure method of authentication in combination with a production-grade object storage service.

type: "filesystem"

##### `path`

- **Type:** string
- **Required:** yes

Defines the path to the directory to use for object storage.

type: "disabled"

If you set `type = "disabled"`, the TensorZero Gateway will not store or retrieve images.
There are no additional fields available for this type.

## [​](https://www.tensorzero.com/docs/gateway/configuration-reference\#[postgres])  `[postgres]`

The `[postgres]` section defines the configuration for PostgreSQL connectivity.PostgreSQL is required for certain TensorZero features including [rate limiting](https://www.tensorzero.com/docs/operations/enforce-custom-rate-limits) and [Track-and-Stop experimentation](https://www.tensorzero.com/docs/experimentation/run-adaptive-ab-tests).
You can connect to PostgreSQL by setting the `TENSORZERO_POSTGRES_URL` environment variable.

### [​](https://www.tensorzero.com/docs/gateway/configuration-reference\#connection-pool-size)  `connection_pool_size`

- **Type:** integer
- **Required:** no (default: `20`)

Defines the maximum number of connections in the PostgreSQL connection pool.

### [​](https://www.tensorzero.com/docs/gateway/configuration-reference\#enabled)  `enabled`

- **Type:** boolean
- **Required:** no (default: `null`)

Enable PostgreSQL connectivity.
If `true`, the gateway will throw an error on startup if it fails to connect to PostgreSQL (requires `TENSORZERO_POSTGRES_URL` environment variable).
If `false`, the gateway will not use PostgreSQL even if the `TENSORZERO_POSTGRES_URL` environment variable is set.
If omitted, the gateway will connect to PostgreSQL if the `TENSORZERO_POSTGRES_URL` environment variable is set, otherwise it will disable PostgreSQL with a warning.If you have features that require PostgreSQL (rate limiting or Track-and-Stop experimentation) configured but set `postgres.enabled = false` or don’t provide the `TENSORZERO_POSTGRES_URL` environment variable, the gateway will fail to start with a configuration error.

## [​](https://www.tensorzero.com/docs/gateway/configuration-reference\#[rate-limiting])  `[rate_limiting]`

The `[rate_limiting]` section allows you to configure granular rate limits for your TensorZero Gateway.
Rate limits help you control usage, manage costs, and prevent abuse.See [Enforce Custom Rate Limits](https://www.tensorzero.com/docs/operations/enforce-custom-rate-limits) for a comprehensive guide on rate limiting.

### [​](https://www.tensorzero.com/docs/gateway/configuration-reference\#enabled-2)  `enabled`

- **Type:** boolean
- **Required:** no (default: `true`)

Enable or disable rate limiting enforcement.
When set to `false`, rate limiting rules will not be enforced even if they are defined.

Copy

```
[rate_limiting]
enabled = true
```

### [​](https://www.tensorzero.com/docs/gateway/configuration-reference\#[[rate-limiting-rules]])  `[[rate_limiting.rules]]`

Rate limiting rules are defined as an array of rule configurations.
Each rule specifies rate limits for specific resources (model inferences, tokens), time windows, scopes, and priorities.

#### [​](https://www.tensorzero.com/docs/gateway/configuration-reference\#rate-limit-fields)  Rate Limit Fields

You can set rate limits for different resources and time windows using the following field formats:

- `model_inferences_per_second`
- `model_inferences_per_minute`
- `model_inferences_per_hour`
- `model_inferences_per_day`
- `model_inferences_per_week`
- `model_inferences_per_month`
- `tokens_per_second`
- `tokens_per_minute`
- `tokens_per_hour`
- `tokens_per_day`
- `tokens_per_week`
- `tokens_per_month`

Each rate limit field can be specified in two formats:**Simple Format:** A single integer value that sets both the capacity and refill rate to the same value.

Copy

```
[[rate_limiting.rules]]
model_inferences_per_minute = 100
tokens_per_hour = 10000
```

**Bucket Format:** An object with explicit `capacity` and `refill_rate` fields for fine-grained control over the token bucket algorithm.

Copy

```
[[rate_limiting.rules]]
tokens_per_minute = { capacity = 1000, refill_rate = 500 }
```

The simple format is equivalent to setting `capacity` and `refill_rate` to the same value.
The bucket format allows you to configure burst capacity independently from the sustained rate.

#### [​](https://www.tensorzero.com/docs/gateway/configuration-reference\#priority)  `priority`

- **Type:** integer
- **Required:** yes (unless `always` is set to `true`)

Defines the priority of the rule.
When multiple rules match a request, only the rules with the highest priority value are applied.

Copy

```
[[rate_limiting.rules]]
model_inferences_per_minute = 10
priority = 1
```

#### [​](https://www.tensorzero.com/docs/gateway/configuration-reference\#always)  `always`

- **Type:** boolean
- **Required:** no (mutually exclusive with `priority`)

When set to `true`, this rule will always be applied regardless of priority.
This is useful for global fallback limits.You cannot specify both `always` and `priority` in the same rule.

Copy

```
[[rate_limiting.rules]]
tokens_per_hour = 1000000
always = true
```

#### [​](https://www.tensorzero.com/docs/gateway/configuration-reference\#scope)  `scope`

- **Type:** array of scope objects
- **Required:** no (default: `[]`)

Defines the scope to which the rate limit applies.
Scopes allow you to apply rate limits to specific subsets of requests based on tags or API keys.The following scopes are supported:

- Tags:  - `tag_key` (string): The tag key to match against.
  - `tag_value` (string): The tag value to match against. This can be:

    - `tensorzero::each`: Apply the limit separately to each unique value of the tag.
    - `tensorzero::total`: Apply the limit to the aggregate of all requests with this tag, regardless of the tag’s value.
    - Any other string: Apply the limit only when the tag has this specific value.
- API Key Public ID (requires authentication to be enabled):  - `api_key_public_id` (string): The API key public ID to match against. This can be:

    - `tensorzero::each`: Apply the limit separately to each API key.
    - A specific 12-character public ID: Apply the limit only to requests authenticated with this API key.

For example:

Copy

```
# Each individual user can make a maximum of 1 model inference per minute
[[rate_limiting.rules]]
priority = 0
model_inferences_per_minute = 1
scope = [\
    { tag_key = "user_id", tag_value = "tensorzero::each" }\
]

# But override the individual limit for the CEO
[[rate_limiting.rules]]
priority = 1
model_inferences_per_minute = 5
scope = [\
    { tag_key = "user_id", tag_value = "ceo" }\
]

# Each API key can make a maximum of 100 model inferences per hour
[[rate_limiting.rules]]
priority = 0
model_inferences_per_hour = 100
scope = [\
    { api_key_public_id = "tensorzero::each" }\
]

# But override the limit for a specific API key
[[rate_limiting.rules]]
priority = 1
model_inferences_per_hour = 1000
scope = [\
    { api_key_public_id = "xxxxxxxxxxxx" }\
]
```

[Clients](https://www.tensorzero.com/docs/gateway/clients) [Data Model](https://www.tensorzero.com/docs/gateway/data-model)

Ctrl+I

Assistant

Responses are generated using AI and may contain mistakes.

## Configure Functions & Variants
[Skip to main content](https://www.tensorzero.com/docs/gateway/configure-functions-and-variants#content-area)

[TensorZero Docs home page![light logo](https://mintcdn.com/tensorzero/afFfERJmQ8YiZpCj/logo/light.svg?fit=max&auto=format&n=afFfERJmQ8YiZpCj&q=85&s=9396731a8a92cee7bd293dee05d71dac)![dark logo](https://mintcdn.com/tensorzero/afFfERJmQ8YiZpCj/logo/dark.svg?fit=max&auto=format&n=afFfERJmQ8YiZpCj&q=85&s=8d61c59531dbcdcace7fcdfc524bd4d2)](https://www.tensorzero.com/)

Search...

Ctrl K

Search...

Navigation

Gateway

How to configure functions & variants

[Guides](https://www.tensorzero.com/docs) [Integrations](https://www.tensorzero.com/docs/integrations/model-providers)

On this page

- [Configure functions & variants](https://www.tensorzero.com/docs/gateway/configure-functions-and-variants#configure-functions-&-variants)
- [Example](https://www.tensorzero.com/docs/gateway/configure-functions-and-variants#example)
- [Make inference requests](https://www.tensorzero.com/docs/gateway/configure-functions-and-variants#make-inference-requests)

- A **function** represents a task or agent in your application (e.g. “write a product description” or “answer a customer question”).
- A **variant** is a specific way to accomplish it: a choice of model, prompt, inference parameters, etc.

You can call models directly when getting started, but functions and variants unlock powerful capabilities as your application matures.
Some of the benefits include:

- **[Collect metrics and feedback](https://www.tensorzero.com/docs/gateway/guides/metrics-feedback):** Track performance and gather feedback for optimization.
- **[Run A/B tests](https://www.tensorzero.com/docs/experimentation/run-adaptive-ab-tests):** Experiment with different models, prompts, and parameters.
- **[Create prompt templates](https://www.tensorzero.com/docs/gateway/create-a-prompt-template):** Decouple prompts from application code for easier iteration.
- **[Configure retries & fallbacks](https://www.tensorzero.com/docs/gateway/guides/retries-fallbacks):** Build systems that handle provider downtime gracefully.
- **[Use advanced inference strategies](https://www.tensorzero.com/docs/gateway/guides/inference-time-optimizations):** Easily implement advanced inference-time optimizations like dynamic in-context-learning and best-of-N sampling.

## [​](https://www.tensorzero.com/docs/gateway/configure-functions-and-variants\#configure-functions-&-variants)  Configure functions & variants

TensorZero supports two function types:

- **`chat`** is the typical chat interface used by most LLMs. It returns unstructured text responses.
- **`json`** is for structured outputs. It returns responses that conform to a JSON schema. See [Generate structured outputs (JSON)](https://www.tensorzero.com/docs/gateway/generate-structured-outputs).

The skeleton of a function configuration looks like this:

tensorzero.toml

Copy

```
[functions.my_function_name]
type = "..." # "chat" or "json"
# ... other fields depend on the function type ...
```

A variant is a particular implementation of a function.
It specifies the model to use, prompt templates, decoding strategy, hyperparameters, and other settings.The skeleton of a variant configuration looks like this:

tensorzero.toml

Copy

```
[functions.my_function_name.variants.my_variant_name]
type = "..." # e.g. "chat_completion"
model = "..." # e.g. "openai::gpt-5" or "my_gpt_5"
# ... other fields (e.g. prompt templates, inference parameters) ...
```

The simplest variant type is **`chat_completion`**, which is the typical chat completion format used by OpenAI and many other LLM providers.
TensorZero supports other variant types that implement [inference-time optimizations](https://www.tensorzero.com/docs/gateway/guides/inference-time-optimizations).You can define prompt templates in your variant configuration rather than sending prompts directly in your inference requests.
This decouples prompts from application code and enables easier experimentation and optimization.
See [Create a prompt template](https://www.tensorzero.com/docs/gateway/create-a-prompt-template) for more details.If you define multiple variants, TensorZero will randomly sample one of them at inference time.
You can define more advanced experimentation strategies (e.g. [Run adaptive A/B tests](https://www.tensorzero.com/docs/experimentation/run-adaptive-ab-tests)), fallback-only variants (e.g. [Retries & Fallbacks](https://www.tensorzero.com/docs/gateway/guides/retries-fallbacks)), and more.

### [​](https://www.tensorzero.com/docs/gateway/configure-functions-and-variants\#example)  Example

Let’s create a function called `answer_customer` with two variants: GPT-5 and Claude Sonnet 4.5.

tensorzero.toml

Copy

```
[functions.answer_customer]
type = "chat"

[functions.answer_customer.variants.gpt_5_baseline]
type = "chat_completion"
model = "openai::gpt-5"

[functions.answer_customer.variants.claude_sonnet_4_5]
type = "chat_completion"
model = "anthropic::claude-sonnet-4-5"
```

You can now call the `answer_customer` function and TensorZero will randomly select one of the two variants for each request.

## [​](https://www.tensorzero.com/docs/gateway/configure-functions-and-variants\#make-inference-requests)  Make inference requests

Once you’ve configured a function and its variants, you can make inference requests to the TensorZero Gateway.

- Python

- Python (OpenAI SDK)

- Node (OpenAI SDK)

- HTTP


Copy

```
result = t0.inference(
    function_name="answer_customer",
    input={
        "messages": [\
            {"role": "user", "content": "What is your return policy?"},\
        ],
    },
)
```

See [Call any LLM](https://www.tensorzero.com/docs/gateway/call-any-llm) for complete examples including setup and sample responses.

[Configure models & providers](https://www.tensorzero.com/docs/gateway/configure-models-and-providers) [Create a prompt template](https://www.tensorzero.com/docs/gateway/create-a-prompt-template)

Ctrl+I

Assistant

Responses are generated using AI and may contain mistakes.

## Configure Models & Providers
[Skip to main content](https://www.tensorzero.com/docs/gateway/configure-models-and-providers#content-area)

[TensorZero Docs home page![light logo](https://mintcdn.com/tensorzero/afFfERJmQ8YiZpCj/logo/light.svg?fit=max&auto=format&n=afFfERJmQ8YiZpCj&q=85&s=9396731a8a92cee7bd293dee05d71dac)![dark logo](https://mintcdn.com/tensorzero/afFfERJmQ8YiZpCj/logo/dark.svg?fit=max&auto=format&n=afFfERJmQ8YiZpCj&q=85&s=8d61c59531dbcdcace7fcdfc524bd4d2)](https://www.tensorzero.com/)

Search...

Ctrl K

Search...

Navigation

Gateway

How to configure models & providers

[Guides](https://www.tensorzero.com/docs) [Integrations](https://www.tensorzero.com/docs/integrations/model-providers)

On this page

- [Configure a model & model provider](https://www.tensorzero.com/docs/gateway/configure-models-and-providers#configure-a-model-&-model-provider)
- [Example: GPT-5 + OpenAI](https://www.tensorzero.com/docs/gateway/configure-models-and-providers#example:-gpt-5-+-openai)
- [Configure multiple providers for fallback & routing](https://www.tensorzero.com/docs/gateway/configure-models-and-providers#configure-multiple-providers-for-fallback-&-routing)
- [Use short-hand model names](https://www.tensorzero.com/docs/gateway/configure-models-and-providers#use-short-hand-model-names)

- A **model** specifies a particular LLM (e.g. GPT-5 or your fine-tuned Llama 3).
- A **model provider** specifies how you can access a given model (e.g. GPT-5 is available through both OpenAI and Azure).

You can call models directly using the inference endpoint or use them with [functions and variants](https://www.tensorzero.com/docs/gateway/configure-functions-and-variants) in TensorZero.

## [​](https://www.tensorzero.com/docs/gateway/configure-models-and-providers\#configure-a-model-&-model-provider)  Configure a model & model provider

A model has an arbitrary name and a list of providers.
Each provider has an arbitrary name, a type, and other fields that depend on the provider type.The skeleton of a model and provider configuration looks like this:

tensorzero.toml

Copy

```
[models.my_model_name]
routing = ["my_provider_name"]

[models.my_model_name.providers.my_provider_name]
type = "..."  # e.g. "openai"
# ... other fields depend on the provider type ...
```

TensorZero supports proprietary models (e.g. OpenAI, Anthropic), inference services (e.g. Fireworks AI, Together AI), and self-hosted LLMs (e.g. vLLM), including your own fine-tuned models on each of these.

See [Integrations](https://www.tensorzero.com/docs/integrations/model-providers) for a complete list of supported providers and the [Configuration Reference](https://www.tensorzero.com/docs/gateway/configuration-reference#modelsmodel_nameprovidersprovider_name) for all available configuration parameters.

### [​](https://www.tensorzero.com/docs/gateway/configure-models-and-providers\#example:-gpt-5-+-openai)  Example: GPT-5 + OpenAI

Let’s configure a provider for GPT-5 from OpenAI.
We’ll call our model `my_gpt_5` and our provider `my_openai_provider` with type `openai`.
The only required field for the `openai` provider is `model_name`.

tensorzero.toml

Copy

```
[models.my_gpt_5]
routing = ["my_openai_provider"]

[models.my_gpt_5.providers.my_openai_provider]
type = "openai"
model_name = "gpt-5"
```

You can now reference the model `my_gpt_5` when calling the inference endpoint or when configuring functions and variants.

## [​](https://www.tensorzero.com/docs/gateway/configure-models-and-providers\#configure-multiple-providers-for-fallback-&-routing)  Configure multiple providers for fallback & routing

You can configure multiple providers for the same model to enable automatic fallbacks.
The gateway will try each provider in the `routing` field in order until one succeeds.
This helps mitigate provider downtime and rate limiting.For example, you might configure both OpenAI and Azure as providers for GPT-5:

tensorzero.toml

Copy

```
[models.my_gpt_5]
routing = ["my_openai_provider", "my_azure_provider"]

[models.my_gpt_5.providers.my_openai_provider]
type = "openai"
model_name = "gpt-5"

[models.my_gpt_5.providers.my_azure_provider]
type = "azure"
deployment_id = "gpt-5"
endpoint = "https://your-resource.openai.azure.com"
```

See [Retries & Fallbacks](https://www.tensorzero.com/docs/gateway/guides/retries-fallbacks) for more details on configuring robust routing strategies.

## [​](https://www.tensorzero.com/docs/gateway/configure-models-and-providers\#use-short-hand-model-names)  Use short-hand model names

If you don’t need advanced functionality like fallback routing or custom credentials, you can use shorthand model names directly in your variant configuration.TensorZero supports shorthand names like:

- `openai::gpt-5`
- `anthropic::claude-3-5-haiku-20241022`
- `google::gemini-2.0-flash-exp`

You can use these directly in a variant’s `model` field without defining a separate model configuration block.

tensorzero.toml

Copy

```
[functions.my_function.variants.my_variant]
type = "chat_completion"
model = "openai::gpt-5"
# ...
```

[Call the OpenAI Responses API](https://www.tensorzero.com/docs/gateway/call-the-openai-responses-api) [Configure functions & variants](https://www.tensorzero.com/docs/gateway/configure-functions-and-variants)

Ctrl+I

Assistant

Responses are generated using AI and may contain mistakes.

## Create Prompt Templates
[Skip to main content](https://www.tensorzero.com/docs/gateway/create-a-prompt-template#content-area)

[TensorZero Docs home page![light logo](https://mintcdn.com/tensorzero/afFfERJmQ8YiZpCj/logo/light.svg?fit=max&auto=format&n=afFfERJmQ8YiZpCj&q=85&s=9396731a8a92cee7bd293dee05d71dac)![dark logo](https://mintcdn.com/tensorzero/afFfERJmQ8YiZpCj/logo/dark.svg?fit=max&auto=format&n=afFfERJmQ8YiZpCj&q=85&s=8d61c59531dbcdcace7fcdfc524bd4d2)](https://www.tensorzero.com/)

Search...

Ctrl K

Search...

Navigation

Gateway

How to create a prompt template

[Guides](https://www.tensorzero.com/docs) [Integrations](https://www.tensorzero.com/docs/integrations/model-providers)

On this page

- [Why create a prompt template?](https://www.tensorzero.com/docs/gateway/create-a-prompt-template#why-create-a-prompt-template)
- [Set up a prompt template](https://www.tensorzero.com/docs/gateway/create-a-prompt-template#set-up-a-prompt-template)
- [Set up a template schema](https://www.tensorzero.com/docs/gateway/create-a-prompt-template#set-up-a-template-schema)
- [Re-use prompt snippets](https://www.tensorzero.com/docs/gateway/create-a-prompt-template#re-use-prompt-snippets)
- [Migrate from legacy prompt templates](https://www.tensorzero.com/docs/gateway/create-a-prompt-template#migrate-from-legacy-prompt-templates)

## [​](https://www.tensorzero.com/docs/gateway/create-a-prompt-template\#why-create-a-prompt-template)  Why create a prompt template?

Prompt templates and schemas simplify engineering iteration, experimentation, and optimization, especially as application complexity and team size grow.
Notably, they enable you to:

1. **Decouple prompts from application code.**
As you iterate on your prompts over time (or [A/B test different prompts](https://www.tensorzero.com/docs/experimentation/run-adaptive-ab-tests)), you’ll be able to manage them in a centralized way without making changes to the application code.
2. **Collect a structured inference dataset.**
Imagine down the road you want to [fine-tune a model](https://www.tensorzero.com/docs/recipes) using your historical data.
If you had only stored prompts as strings, you’d be stuck with the outdated prompts that were actually used at inference time.
However, if you had access to the input variables in a structured dataset, you’d easily be able to counterfactually swap new prompts into your training data before fine-tuning.
This is particularly important when experimenting with new models, because prompts don’t always translate well between them.
3. **Implement model-specific prompts.**
We often find that the best prompt for one model is different from the best prompt for another.
As you try out different models, you’ll need to be able to independently vary the prompt and the model and try different combinations thereof.
This is commonly challenging to implement in application code, but trivial in TensorZero.

You can also find a complete runnable example for this guide on [GitHub](https://github.com/tensorzero/tensorzero/tree/main/examples/docs/guides/gateway/create-a-prompt-template).

## [​](https://www.tensorzero.com/docs/gateway/create-a-prompt-template\#set-up-a-prompt-template)  Set up a prompt template

1

Create your template

Create a file with your MiniJinja template:

config/functions/fun\_fact/gpt\_5\_mini/fun\_fact\_topic\_template.minijinja

Copy

```
Share a fun fact about: {{ topic }}
```

TensorZero uses the [MiniJinja templating language](https://docs.rs/minijinja/latest/minijinja/syntax/index.html).
MiniJinja is [mostly compatible with Jinja2](https://github.com/mitsuhiko/minijinja/blob/main/COMPATIBILITY.md), which is used by many popular projects like Flask and Django.

MiniJinja provides a [browser playground](https://mitsuhiko.github.io/minijinja-playground/) where you can test your templates.

2

Configure a template

Next, you must declare the template in the [variant configuration](https://www.tensorzero.com/docs/gateway/configure-functions-and-variants).
You can do this by adding the field `templates.your_template_name.path` to your variant with a path to your template file.For example, let’s configure a template called `fun_fact_topic` for our variant:

config/tensorzero.toml

Copy

```
[functions.fun_fact]
type = "chat"

[functions.fun_fact.variants.gpt_5_mini]
type = "chat_completion"
model = "openai::gpt-5-mini"
templates.fun_fact_topic.path = "functions/fun_fact/gpt_5_mini/fun_fact_topic_template.minijinja" # relative to this file
```

You can configure multiple templates for a variant.

3

Use your template during inference

- Python

- Python (OpenAI SDK)

- HTTP


Use your template during inference by sending a content block with the template name and arguments.

Copy

```
result = t0.inference(
    function_name="fun_fact",
    input={
        "messages": [\
            {\
                "role": "user",\
                "content": [\
                    {\
                        "type": "template",\
                        "name": "fun_fact_topic",\
                        "arguments": {"topic": "artificial intelligence"},\
                    }\
                ],\
            }\
        ],
    },
)
```

## [​](https://www.tensorzero.com/docs/gateway/create-a-prompt-template\#set-up-a-template-schema)  Set up a template schema

When you have multiple variants for a function, it becomes challenging to ensure all templates use consistent variable names and types.
Schemas solve this by defining a contract that validates template variables and catches configuration errors before they reach production.
Defining a schema is optional but recommended.

1

Create a schema

Create a [JSON Schema](https://json-schema.org/) for the variables used by your templates.Let’s define a schema for our previous example, which includes only a single variable `topic`:

config/functions/fun\_fact/fun\_fact\_topic\_schema.json

Copy

```
{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "type": "object",
  "properties": {
    "topic": {
      "type": "string"
    }
  },
  "required": ["topic"],
  "additionalProperties": false
}
```

LLMs are great at generating JSON Schemas.
For example, the schema above was generated with the following request:

Copy

```
Generate a JSON schema with a single field: `topic`.
The `topic` field is required. No additional fields are allowed.
```

You can also export JSON Schemas from [Pydantic models](https://docs.pydantic.dev/latest/concepts/json_schema/) and [Zod schemas](https://www.npmjs.com/package/zod-to-json-schema).

2

Configure a schema

Then, declare your schema in your function definition using `schemas.your_schema_name.path`.
This will ensure that every variant for the function has a template named `your_schema_name`.In our example above, this would mean updating the function definition to:

Copy

```
[functions.fun_fact]
type = "chat"
schemas.fun_fact_topic.path = "functions/fun_fact/fun_fact_topic_schema.json" # relative to this file

[functions.fun_fact.variants.gpt_5_mini]
type = "chat_completion"
model = "openai::gpt-5-mini"
templates.fun_fact_topic.path = "functions/fun_fact/gpt_5_mini/fun_fact_topic_template.minijinja" # relative to this file
```

## [​](https://www.tensorzero.com/docs/gateway/create-a-prompt-template\#re-use-prompt-snippets)  Re-use prompt snippets

You can enable template file system access to reuse shared snippets in your prompts.To use the MiniJinja directives `{% include %}` and `{% import %}`, set `gateway.template_filesystem_access.base_path` in your configuration.
See [Organize your configuration](https://www.tensorzero.com/docs/operations/organize-your-configuration#enable-template-file-system-access-to-reuse-shared-snippets) for details.

## [​](https://www.tensorzero.com/docs/gateway/create-a-prompt-template\#migrate-from-legacy-prompt-templates)  Migrate from legacy prompt templates

In earlier versions of TensorZero, prompt templates were defined as `system_template`, `user_template`, and `assistant_template`.
Similarly, template schemas were defined as `system_schema`, `user_schema`, and `assistant_schema`.
This legacy approach limited the flexibility of prompt templates restricting the ability to define multiple templates per role.As you create new functions and templates, you should use the new `templates.your_template_name.path` format.Historical observability data stored in your ClickHouse database still uses the legacy format.
If you want to keep this data forward-compatible (e.g. for fine-tuning), you can update your configuration as follows:

| Legacy Configuration | Updated Configuration |
| --- | --- |
| `system_template` | `templates.system.path` |
| `system_schema` | `schemas.system.path` |
| `user_template` | `templates.user.path` |
| `user_schema` | `schemas.user.path` |
| `assistant_template` | `templates.assistant.path` |
| `assistant_schema` | `schemas.assistant.path` |

As we deprecate the legacy format, TensorZero will automatically look for templates and schemas in the new format for your historical data.

[Configure functions & variants](https://www.tensorzero.com/docs/gateway/configure-functions-and-variants) [Generate structured outputs](https://www.tensorzero.com/docs/gateway/generate-structured-outputs)

Ctrl+I

Assistant

Responses are generated using AI and may contain mistakes.

## TensorZero Data Model
[Skip to main content](https://www.tensorzero.com/docs/gateway/data-model#content-area)

[TensorZero Docs home page![light logo](https://mintcdn.com/tensorzero/afFfERJmQ8YiZpCj/logo/light.svg?fit=max&auto=format&n=afFfERJmQ8YiZpCj&q=85&s=9396731a8a92cee7bd293dee05d71dac)![dark logo](https://mintcdn.com/tensorzero/afFfERJmQ8YiZpCj/logo/dark.svg?fit=max&auto=format&n=afFfERJmQ8YiZpCj&q=85&s=8d61c59531dbcdcace7fcdfc524bd4d2)](https://www.tensorzero.com/)

Search...

Ctrl K

Search...

Navigation

Gateway

Data Model

[Guides](https://www.tensorzero.com/docs) [Integrations](https://www.tensorzero.com/docs/integrations/model-providers)

On this page

- [ChatInference](https://www.tensorzero.com/docs/gateway/data-model#chatinference)
- [JsonInference](https://www.tensorzero.com/docs/gateway/data-model#jsoninference)
- [ModelInference](https://www.tensorzero.com/docs/gateway/data-model#modelinference)
- [DynamicInContextLearningExample](https://www.tensorzero.com/docs/gateway/data-model#dynamicincontextlearningexample)
- [BooleanMetricFeedback](https://www.tensorzero.com/docs/gateway/data-model#booleanmetricfeedback)
- [FloatMetricFeedback](https://www.tensorzero.com/docs/gateway/data-model#floatmetricfeedback)
- [CommentFeedback](https://www.tensorzero.com/docs/gateway/data-model#commentfeedback)
- [DemonstrationFeedback](https://www.tensorzero.com/docs/gateway/data-model#demonstrationfeedback)
- [ModelInferenceCache](https://www.tensorzero.com/docs/gateway/data-model#modelinferencecache)
- [ChatInferenceDatapoint](https://www.tensorzero.com/docs/gateway/data-model#chatinferencedatapoint)
- [JsonInferenceDatapoint](https://www.tensorzero.com/docs/gateway/data-model#jsoninferencedatapoint)
- [BatchRequest](https://www.tensorzero.com/docs/gateway/data-model#batchrequest)
- [BatchModelInference](https://www.tensorzero.com/docs/gateway/data-model#batchmodelinference)

The TensorZero Gateway stores inference and feedback data in ClickHouse.
This data can be used for observability, experimentation, and optimization.

## [​](https://www.tensorzero.com/docs/gateway/data-model\#chatinference)  `ChatInference`

The `ChatInference` table stores information about inference requests for Chat Functions made to the TensorZero Gateway.A `ChatInference` row can be associated with one or more `ModelInference` rows, depending on the variant’s `type`.
For `chat_completion`, there will be a one-to-one relationship between rows in the two tables.
For other variant types, there might be more associated rows.

| Column | Type | Notes |
| --- | --- | --- |
| `id` | UUID | Must be a UUIDv7 |
| `function_name` | String |  |
| `variant_name` | String |  |
| `episode_id` | UUID | Must be a UUIDv7 |
| `input` | String (JSON) | `input` field in the `/inference` request body |
| `output` | String (JSON) | Array of content blocks |
| `tool_params` | String (JSON) | Object with any tool parameters (e.g. `tool_choice`, `tools_available`) used for the inference |
| `inference_params` | String (JSON) | Object with any inference parameters per variant type (e.g. `{"chat_completion": {"temperature": 0.5}}`) |
| `processing_time_ms` | UInt32 |  |
| `timestamp` | DateTime | Materialized from `id` (using `UUIDv7ToDateTime` function) |
| `tags` | Map(String, String) | User-assigned tags (e.g. `{"user_id": "123"}`) |

## [​](https://www.tensorzero.com/docs/gateway/data-model\#jsoninference)  `JsonInference`

The `JsonInference` table stores information about inference requests for JSON Functions made to the TensorZero Gateway.A `JsonInference` row can be associated with one or more `ModelInference` rows, depending on the variant’s `type`.
For `chat_completion`, there will be a one-to-one relationship between rows in the two tables.
For other variant types, there might be more associated rows.

| Column | Type | Notes |
| --- | --- | --- |
| `id` | UUID | Must be a UUIDv7 |
| `function_name` | String |  |
| `variant_name` | String |  |
| `episode_id` | UUID | Must be a UUIDv7 |
| `input` | String (JSON) | `input` field in the `/inference` request body |
| `output` | String (JSON) | Object with `parsed` and `raw` fields |
| `output_schema` | String (JSON) | Schema that the output must conform to |
| `inference_params` | String (JSON) | Object with any inference parameters per variant type (e.g. `{"chat_completion": {"temperature": 0.5}}`) |
| `processing_time_ms` | UInt32 |  |
| `timestamp` | DateTime | Materialized from `id` (using `UUIDv7ToDateTime` function) |
| `tags` | Map(String, String) | User-assigned tags (e.g. `{"user_id": "123"}`) |

## [​](https://www.tensorzero.com/docs/gateway/data-model\#modelinference)  `ModelInference`

The `ModelInference` table stores information about each inference request to a model provider.
This is the inference request you’d make if you had called the model provider directly.

| Column | Type | Notes |
| --- | --- | --- |
| `id` | UUID | Must be a UUIDv7 |
| `inference_id` | UUID | Must be a UUIDv7 |
| `raw_request` | String | Raw request as sent to the model provider (varies) |
| `raw_response` | String | Raw response from the model provider (varies) |
| `model_name` | String | Name of the model used for the inference |
| `model_provider_name` | String | Name of the model provider used for the inference |
| `input_tokens` | Nullable(UInt32) |  |
| `output_tokens` | Nullable(UInt32) |  |
| `response_time_ms` | Nullable(UInt32) |  |
| `ttft_ms` | Nullable(UInt32) | Only available in streaming inferences |
| `timestamp` | DateTime | Materialized from `id` (using `UUIDv7ToDateTime` function) |
| `system` | Nullable(String) | The `system` input to the model |
| `input_messages` | Array(RequestMessage) | The user and assistant messages input to the model |
| `output` | Array(ContentBlock) | The output of the model |

A `RequestMessage` is an object with shape `{role: "user" | "assistant", content: List[ContentBlock]}` (content blocks are defined [here](https://www.tensorzero.com/docs/gateway/api-reference/inference#content-block)).

## [​](https://www.tensorzero.com/docs/gateway/data-model\#dynamicincontextlearningexample)  `DynamicInContextLearningExample`

The `DynamicInContextLearningExample` table stores examples for dynamic in-context learning variants.

| Column | Type | Notes |
| --- | --- | --- |
| `id` | UUID | Must be a UUIDv7 |
| `function_name` | String |  |
| `variant_name` | String |  |
| `namespace` | String |  |
| `input` | String (JSON) |  |
| `output` | String |  |
| `embedding` | Array(Float32) |  |
| `timestamp` | DateTime | Materialized from `id` (using `UUIDv7ToDateTime` function) |

## [​](https://www.tensorzero.com/docs/gateway/data-model\#booleanmetricfeedback)  `BooleanMetricFeedback`

The `BooleanMetricFeedback` table stores feedback for metrics of `type = "boolean"`.

| Column | Type | Notes |
| --- | --- | --- |
| `id` | UUID | Must be a UUIDv7 |
| `target_id` | UUID | Must be a UUIDv7 that is either `inference_id` or `episode_id` depending on `level` in metric config |
| `metric_name` | String |  |
| `value` | Bool |  |
| `timestamp` | DateTime | Materialized from `id` (using `UUIDv7ToDateTime` function) |
| `tags` | Map(String, String) | User-assigned tags (e.g. `{"author": "Alice"}`) |

## [​](https://www.tensorzero.com/docs/gateway/data-model\#floatmetricfeedback)  `FloatMetricFeedback`

The `FloatMetricFeedback` table stores feedback for metrics of `type = "float"`.

| Column | Type | Notes |
| --- | --- | --- |
| `id` | UUID | Must be a UUIDv7 |
| `target_id` | UUID | Must be a UUIDv7 that is either `inference_id` or `episode_id` depending on `level` in metric config |
| `metric_name` | String |  |
| `value` | Float32 |  |
| `timestamp` | DateTime | Materialized from `id` (using `UUIDv7ToDateTime` function) |
| `tags` | Map(String, String) | User-assigned tags (e.g. `{"author": "Alice"}`) |

## [​](https://www.tensorzero.com/docs/gateway/data-model\#commentfeedback)  `CommentFeedback`

The `CommentFeedback` table stores feedback provided with `metric_name` of `"comment"`.
Comments are free-form text feedbacks.

| Column | Type | Notes |
| --- | --- | --- |
| `id` | UUID | Must be a UUIDv7 |
| `target_id` | UUID | Must be a UUIDv7 that is either `inference_id` or `episode_id` depending on `level` in metric config |
| `target_type` | `"inference"` or `"episode"` |  |
| `value` | String |  |
| `timestamp` | DateTime | Materialized from `id` (using `UUIDv7ToDateTime` function) |
| `tags` | Map(String, String) | User-assigned tags (e.g. `{"author": "Alice"}`) |

## [​](https://www.tensorzero.com/docs/gateway/data-model\#demonstrationfeedback)  `DemonstrationFeedback`

The `DemonstrationFeedback` table stores feedback in the form of demonstrations.
Demonstrations are examples of good behaviors.

| Column | Type | Notes |
| --- | --- | --- |
| `id` | UUID | Must be a UUIDv7 |
| `inference_id` | UUID | Must be a UUIDv7 |
| `value` | String | The demonstration or example provided as feedback (must match function output) |
| `timestamp` | DateTime | Materialized from `id` (using `UUIDv7ToDateTime` function) |
| `tags` | Map(String, String) | User-assigned tags (e.g. `{"author": "Alice"}`) |

## [​](https://www.tensorzero.com/docs/gateway/data-model\#modelinferencecache)  `ModelInferenceCache`

The `ModelInferenceCache` table stores cached model inference results to avoid duplicate requests.

| Column | Type | Notes |
| --- | --- | --- |
| `short_cache_key` | UInt64 | First part of composite key for fast lookups |
| `long_cache_key` | FixedString(64) | Hex-encoded 256-bit key for full cache validation |
| `timestamp` | DateTime | When this cache entry was created, defaults to now() |
| `output` | String | The cached model output |
| `raw_request` | String | Raw request that was sent to the model provider |
| `raw_response` | String | Raw response received from the model provider |
| `is_deleted` | Bool | Soft deletion flag, defaults to false |

The table uses the `ReplacingMergeTree` engine with `timestamp` and `is_deleted` columns for deduplication.
It is partitioned by month and ordered by the composite cache key `(short_cache_key, long_cache_key)`.
The `short_cache_key` serves as the primary key for performance, while a bloom filter index on `long_cache_key`
helps optimize point queries.

## [​](https://www.tensorzero.com/docs/gateway/data-model\#chatinferencedatapoint)  `ChatInferenceDatapoint`

The `ChatInferenceDatapoint` table stores chat inference examples organized into datasets.

| Column | Type | Notes |
| --- | --- | --- |
| `dataset_name` | LowCardinality(String) | Name of the dataset this example belongs to |
| `function_name` | LowCardinality(String) | Name of the function this example is for |
| `id` | UUID | Must be a UUIDv7, often the inference ID if generated from an inference |
| `episode_id` | UUID | Must be a UUIDv7 |
| `input` | String (JSON) | `input` field in the `/inference` request body |
| `output` | Nullable(String) (JSON) | Array of content blocks |
| `tool_params` | String (JSON) | Object with any tool parameters (e.g. `tool_choice`, `tools_available`) used for the inference |
| `tags` | Map(String, String) | User-assigned tags (e.g. `{"user_id": "123"}`) |
| `auxiliary` | String | Additional JSON data (unstructured) |
| `is_deleted` | Bool | Soft deletion flag, defaults to false |
| `updated_at` | DateTime | When this dataset entry was updated, defaults to now() |

The table uses the `ReplacingMergeTree` engine with `updated_at` and `is_deleted` columns for deduplication.
It is ordered by `dataset_name`, `function_name`, and `id` to optimize queries filtering by dataset and function.

## [​](https://www.tensorzero.com/docs/gateway/data-model\#jsoninferencedatapoint)  `JsonInferenceDatapoint`

The `JsonInferenceDatapoint` table stores JSON inference examples organized into datasets.

| Column | Type | Notes |
| --- | --- | --- |
| `dataset_name` | LowCardinality(String) | Name of the dataset this example belongs to |
| `function_name` | LowCardinality(String) | Name of the function this example is for |
| `id` | UUID | Must be a UUIDv7, often the inference ID if generated from an inference |
| `episode_id` | UUID | Must be a UUIDv7 |
| `input` | String (JSON) | `input` field in the `/inference` request body |
| `output` | String (JSON) | Object with `parsed` and `raw` fields |
| `output_schema` | String (JSON) | Schema that the output must conform to |
| `tags` | Map(String, String) | User-assigned tags (e.g. `{"user_id": "123"}`) |
| `auxiliary` | String | Additional JSON data (unstructured) |
| `is_deleted` | Bool | Soft deletion flag, defaults to false |
| `updated_at` | DateTime | When this dataset entry was updated, defaults to now() |

The table uses the `ReplacingMergeTree` engine with `updated_at` and `is_deleted` columns for deduplication.
It is ordered by `dataset_name`, `function_name`, and `id` to optimize queries filtering by dataset and function.

## [​](https://www.tensorzero.com/docs/gateway/data-model\#batchrequest)  `BatchRequest`

The `BatchRequest` table stores information about batch requests made to model providers. We update it every time a particular `batch_id` is created or polled.

| Column | Type | Notes |
| --- | --- | --- |
| `batch_id` | UUID | Must be a UUIDv7 |
| `id` | UUID | Must be a UUIDv7 |
| `batch_params` | String | Parameters used for the batch request |
| `model_name` | String | Name of the model used |
| `model_provider_name` | String | Name of the model provider |
| `status` | String | One of: ‘pending’, ‘completed’, ‘failed’ |
| `errors` | Array(String) | Array of error messages if status is ‘failed’ |
| `timestamp` | DateTime | Materialized from `id` (using `UUIDv7ToDateTime` function) |
| `raw_request` | String | Raw request sent to the model provider |
| `raw_response` | String | Raw response received from the model provider |
| `function_name` | String | Name of the function being called |
| `variant_name` | String | Name of the function variant |

## [​](https://www.tensorzero.com/docs/gateway/data-model\#batchmodelinference)  `BatchModelInference`

The `BatchModelInference` table stores information about inferences made as part of a batch request.
Once the request succeeds, we use this information to populate the `ChatInference`, `JsonInference`, and `ModelInference` tables.

| Column | Type | Notes |
| --- | --- | --- |
| `inference_id` | UUID | Must be a UUIDv7 |
| `batch_id` | UUID | Must be a UUIDv7 |
| `function_name` | String | Name of the function being called |
| `variant_name` | String | Name of the function variant |
| `episode_id` | UUID | Must be a UUIDv7 |
| `input` | String (JSON) | `input` field in the `/inference` request body |
| `system` | String | The `system` input to the model |
| `input_messages` | Array(RequestMessage) | The user and assistant messages input to the model |
| `tool_params` | String (JSON) | Object with any tool parameters (e.g. `tool_choice`, `tools_available`) used for the inference |
| `inference_params` | String (JSON) | Object with any inference parameters per variant type (e.g. `{"chat_completion": {"temperature": 0.5}}`) |
| `raw_request` | String | Raw request sent to the model provider |
| `model_name` | String | Name of the model used |
| `model_provider_name` | String | Name of the model provider |
| `output_schema` | String | Optional schema for JSON outputs |
| `tags` | Map(String, String) | User-assigned tags (e.g. `{"author": "Alice"}`) |
| `timestamp` | DateTime | Materialized from `id` (using `UUIDv7ToDateTime` function) |

Materialized View Tables

[Materialized views](https://clickhouse.com/docs/en/materialized-view) in columnar databases like ClickHouse pre-compute alternative indexings of data, dramatically improving query performance compared to computing results on-the-fly.
In TensorZero’s case, we store denormalized data about inferences and feedback in the materialized views below to support efficient queries for common downstream use cases.

## [​](https://www.tensorzero.com/docs/gateway/data-model\#feedbacktag)  `FeedbackTag`

The `FeedbackTag` table stores tags associated with various feedback types. Tags are used to categorize and add metadata to feedback entries, allowing for user-defined filtering later on. Data is inserted into this table by materialized views reading from the `BooleanMetricFeedback`, `CommentFeedback`, `DemonstrationFeedback`, and `FloatMetricFeedback` tables.

| Column | Type | Notes |
| --- | --- | --- |
| `metric_name` | String | Name of the metric the tag is associated with. |
| `key` | String | Key of the tag. |
| `value` | String | Value of the tag. |
| `feedback_id` | UUID | UUID referencing the related feedback entry (e.g., `BooleanMetricFeedback.id`). |

## [​](https://www.tensorzero.com/docs/gateway/data-model\#inferencebyid)  `InferenceById`

The `InferenceById` table is a materialized view that combines data from `ChatInference` and `JSONInference`.
Notably, it indexes the table by `id_uint` for fast lookup by the gateway to validate feedback requests.
We store `id_uint` as a UInt128 so that they are sorted in the natural order by time as ClickHouse sorts UUIDs in little-endian order.

| Column | Type | Notes |
| --- | --- | --- |
| `id_uint` | UInt128 | Integer representation of UUIDv7 for sorting order |
| `function_name` | String |  |
| `variant_name` | String |  |
| `episode_id` | UUID | Must be a UUIDv7 |
| `function_type` | String | Either `'chat'` or `'json'` |

## [​](https://www.tensorzero.com/docs/gateway/data-model\#inferencebyepisodeid)  `InferenceByEpisodeId`

The `InferenceByEpisodeId` table is a materialized view that indexes inferences by their episode ID, enabling efficient lookup of all inferences within an episode.
We store `episode_id_uint` as a `UInt128` so that they are sorted in the natural order by time as ClickHouse sorts UUIDs in little-endian order.

| Column | Type | Notes |
| --- | --- | --- |
| `episode_id_uint` | UInt128 | Integer representation of UUIDv7 for sorting order |
| `id_uint` | UInt128 | Integer representation of UUIDv7 for sorting order |
| `function_name` | String | Name of the function being called |
| `variant_name` | String | Name of the function variant |
| `function_type` | Enum(‘chat’, ‘json’) | Type of function (chat or json) |

## [​](https://www.tensorzero.com/docs/gateway/data-model\#inferencetag)  `InferenceTag`

The `InferenceTag` table stores tags associated with inferences. Tags are used to categorize and add metadata to inferences, allowing for user-defined filtering later on. Data is inserted into this table by materialized views reading from the `ChatInference` and `JsonInference` tables.

| Column | Type | Notes |
| --- | --- | --- |
| `function_name` | String | Name of the function the tag is associated with. |
| `key` | String | Key of the tag. |
| `value` | String | Value of the tag. |
| `inference_id` | UUID | UUID referencing the related inference (e.g., `ChatInference.id`). |

## [​](https://www.tensorzero.com/docs/gateway/data-model\#batchidbyinferenceid)  `BatchIdByInferenceId`

The `BatchIdByInferenceId` table maps inference IDs to batch IDs, allowing for efficient lookup of which batch an inference belongs to.

| Column | Type | Notes |
| --- | --- | --- |
| `inference_id` | UUID | Must be a UUIDv7 |
| `batch_id` | UUID | Must be a UUIDv7 |

## [​](https://www.tensorzero.com/docs/gateway/data-model\#booleanmetricfeedbackbytargetid)  `BooleanMetricFeedbackByTargetId`

The `BooleanMetricFeedbackByTargetId` table indexes boolean metric feedback by target ID, enabling efficient lookup of feedback for a specific target.

| Column | Type | Notes |
| --- | --- | --- |
| `id` | UUID | Must be a UUIDv7 |
| `target_id` | UUID | Must be a UUIDv7 |
| `metric_name` | String | Name of the metric (stored as LowCardinality) |
| `value` | Bool | The boolean feedback value |
| `tags` | Map(String, String) | Key-value pairs of tags associated with the feedback |

## [​](https://www.tensorzero.com/docs/gateway/data-model\#commentfeedbackbytargetid)  `CommentFeedbackByTargetId`

The `CommentFeedbackByTargetId` table stores text feedback associated with inferences or episodes, enabling efficient lookup of comments by their target ID.

| Column | Type | Notes |
| --- | --- | --- |
| `id` | UUID | Must be a UUIDv7 |
| `target_id` | UUID | Must be a UUIDv7 |
| `target_type` | Enum(‘inference’, ‘episode’) | Type of entity this feedback is for |
| `value` | String | The text feedback content |
| `tags` | Map(String, String) | Key-value pairs of tags associated with the feedback |

## [​](https://www.tensorzero.com/docs/gateway/data-model\#demonstrationfeedbackbyinferenceid)  `DemonstrationFeedbackByInferenceId`

The `DemonstrationFeedbackByInferenceId` table stores demonstration feedback associated with inferences, enabling efficient lookup of demonstrations by inference ID.

| Column | Type | Notes |
| --- | --- | --- |
| `id` | UUID | Must be a UUIDv7 |
| `inference_id` | UUID | Must be a UUIDv7 |
| `value` | String | The demonstration feedback content |
| `tags` | Map(String, String) | Key-value pairs of tags associated with the feedback |

## [​](https://www.tensorzero.com/docs/gateway/data-model\#floatmetricfeedbackbytargetid)  `FloatMetricFeedbackByTargetId`

The `FloatMetricFeedbackByTargetId` table indexes float metric feedback by target ID, enabling efficient lookup of feedback for a specific target.

| Column | Type | Notes |
| --- | --- | --- |
| `id` | UUID | Must be a UUIDv7 |
| `target_id` | UUID | Must be a UUIDv7 |
| `metric_name` | String | Name of the metric (stored as LowCardinality) |
| `value` | Float32 | The float feedback value |
| `tags` | Map(String, String) | Key-value pairs of tags associated with the feedback |

[Configuration Reference](https://www.tensorzero.com/docs/gateway/configuration-reference) [Inference](https://www.tensorzero.com/docs/gateway/api-reference/inference)

Ctrl+I

Assistant

Responses are generated using AI and may contain mistakes.

## Generate Embeddings
[Skip to main content](https://www.tensorzero.com/docs/gateway/generate-embeddings#content-area)

[TensorZero Docs home page![light logo](https://mintcdn.com/tensorzero/afFfERJmQ8YiZpCj/logo/light.svg?fit=max&auto=format&n=afFfERJmQ8YiZpCj&q=85&s=9396731a8a92cee7bd293dee05d71dac)![dark logo](https://mintcdn.com/tensorzero/afFfERJmQ8YiZpCj/logo/dark.svg?fit=max&auto=format&n=afFfERJmQ8YiZpCj&q=85&s=8d61c59531dbcdcace7fcdfc524bd4d2)](https://www.tensorzero.com/)

Search...

Ctrl K

Search...

Navigation

Gateway

How to generate embeddings

[Guides](https://www.tensorzero.com/docs) [Integrations](https://www.tensorzero.com/docs/integrations/model-providers)

On this page

- [Generate embeddings from OpenAI](https://www.tensorzero.com/docs/gateway/generate-embeddings#generate-embeddings-from-openai)
- [Define a custom embedding model](https://www.tensorzero.com/docs/gateway/generate-embeddings#define-a-custom-embedding-model)
- [Cache embeddings](https://www.tensorzero.com/docs/gateway/generate-embeddings#cache-embeddings)

This page shows how to:

- **Generate embeddings with a unified API.** TensorZero unifies many LLM APIs (e.g. OpenAI) and inference servers (e.g. Ollama).
- **Use any programming language.** You can use any OpenAI SDK (Python, Node, Go, etc.) or the OpenAI-compatible HTTP API.

We provide [complete code examples](https://github.com/tensorzero/tensorzero/tree/main/examples/guides/embeddings) on GitHub.

## [​](https://www.tensorzero.com/docs/gateway/generate-embeddings\#generate-embeddings-from-openai)  Generate embeddings from OpenAI

Our example uses the OpenAI Python SDK, but you can use any OpenAI SDK or call the OpenAI-compatible HTTP API.
See [Call any LLM](https://www.tensorzero.com/docs/gateway/call-any-llm) for an example using the OpenAI Node SDK.The TensorZero Python SDK doesn’t have an independent embedding endpoint at the moment.

- Python (OpenAI SDK)


The TensorZero Python SDK integrates with the OpenAI Python SDK to provide a unified API for calling any LLM.

1

Set up the credentials for your LLM provider

For example, if you’re using OpenAI, you can set the `OPENAI_API_KEY` environment variable with your API key.

Copy

```
export OPENAI_API_KEY="sk-..."
```

See the [Integrations](https://www.tensorzero.com/docs/integrations/model-providers) page to learn how to set up credentials for other LLM providers.

2

Install the OpenAI and TensorZero Python SDKs

You can install the OpenAI and TensorZero SDKs with a Python package manager like `pip`.

Copy

```
pip install openai tensorzero
```

3

Initialize the OpenAI client

Let’s initialize the TensorZero Gateway and patch the OpenAI client to use it.
For simplicity, we’ll use an embedded gateway without observability or custom configuration.

Copy

```
from openai import OpenAI
from tensorzero import patch_openai_client

client = OpenAI()
patch_openai_client(client, async_setup=False)
```

The TensorZero Python SDK supports both the synchronous `OpenAI` client and the asynchronous `AsyncOpenAI` client.
Both options support running the gateway embedded in your application with `patch_openai_client` or connecting to a standalone gateway with `base_url`.
The embedded gateway supports synchronous initialization with `async_setup=False` or asynchronous initialization with `async_setup=True`.
See [Clients](https://www.tensorzero.com/docs/gateway/clients) for more details.

4

Call the LLM

Copy

```
result = client.embeddings.create(
    input="Hello, world!",
    model="tensorzero::embedding_model_name::openai::text-embedding-3-small",
    # or: Azure, any OpenAI-compatible endpoint (e.g. Ollama, Voyager)
)
```

Sample Response

Copy

```
CreateEmbeddingResponse(
    data=[\
        Embedding(\
            embedding=[\
                -0.019143931567668915,\
                # ...\
            ],\
            index=0,\
            object='embedding'\
        )\
    ],
    model='tensorzero::embedding_model_name::openai::text-embedding-3-small',
    object='list',
    usage=Usage(prompt_tokens=4, total_tokens=4)
)
```

## [​](https://www.tensorzero.com/docs/gateway/generate-embeddings\#define-a-custom-embedding-model)  Define a custom embedding model

You can define a custom embedding model in your TensorZero configuration file.For example, let’s define a custom embedding model for `nomic-embed-text` served locally by Ollama.

1

Deploy the Ollama embedding model

Download the embedding model and launch the Ollama server:

Copy

```
ollama pull nomic-embed-text
ollama serve
```

We assume that Ollama is available on `http://localhost:11434`.

2

Define your custom embedding model

Add your custom model and model provider to your configuration file:

tensorzero.toml

Copy

```
[embedding_models.nomic-embed-text]
routing = ["ollama"]

[embedding_models.nomic-embed-text.providers.ollama]
type = "openai"
api_base = "http://localhost:11434/v1"
model_name = "nomic-embed-text"
api_key_location = "none"
```

See the [Configuration Reference](https://www.tensorzero.com/docs/gateway/configuration-reference#%5Bembedding-models-model-name%5D) for details on configuring your embedding models.

3

Call your custom embedding model

Use your custom model by referencing it with `tensorzero::embedding_model_name::nomic-embed-text`.For example, using the OpenAI Python SDK:

Copy

```
from openai import OpenAI
from tensorzero import patch_openai_client

client = OpenAI()

patch_openai_client(
    client,
    config_file="config/tensorzero.toml",
    async_setup=False,
)

result = client.embeddings.create(
    input="Hello, world!",
    model="tensorzero::embedding_model_name::nomic-embed-text",
)
```

Sample Response

Copy

```
CreateEmbeddingResponse(
    data=[\
        Embedding(\
            embedding=[\
                -0.019143931567668915,\
                # ...\
            ],\
            index=0,\
            object='embedding'\
        )\
    ],
    model='tensorzero::embedding_model_name::nomic-embed-text',
    object='list',
    usage=Usage(prompt_tokens=4, total_tokens=4)
)
```

## [​](https://www.tensorzero.com/docs/gateway/generate-embeddings\#cache-embeddings)  Cache embeddings

The TensorZero Gateway supports caching embeddings to improve latency and reduce costs.
When caching is enabled, identical embedding requests will be served from the cache instead of being sent to the model provider.

Copy

```
result = client.embeddings.create(
    input="Hello, world!",
    model="tensorzero::embedding_model_name::openai::text-embedding-3-small",
    extra_body={
        "tensorzero::cache_options": {
            "enabled": "on",  # Enable reading from and writing to cache
            "max_age_s": 3600,  # Optional: cache entries older than 1 hour are ignored
        }
    }
)
```

Caching works for single embeddings.
Batch embedding requests (multiple inputs) will write to the cache but won’t serve cached responses.See the [Inference Caching](https://www.tensorzero.com/docs/gateway/guides/inference-caching) guide for more details on cache modes and options.

[Generate structured outputs](https://www.tensorzero.com/docs/gateway/generate-structured-outputs) [Batch Inference](https://www.tensorzero.com/docs/gateway/guides/batch-inference)

Ctrl+I

Assistant

Responses are generated using AI and may contain mistakes.

## Generate Structured Outputs
[Skip to main content](https://www.tensorzero.com/docs/gateway/generate-structured-outputs#content-area)

[TensorZero Docs home page![light logo](https://mintcdn.com/tensorzero/afFfERJmQ8YiZpCj/logo/light.svg?fit=max&auto=format&n=afFfERJmQ8YiZpCj&q=85&s=9396731a8a92cee7bd293dee05d71dac)![dark logo](https://mintcdn.com/tensorzero/afFfERJmQ8YiZpCj/logo/dark.svg?fit=max&auto=format&n=afFfERJmQ8YiZpCj&q=85&s=8d61c59531dbcdcace7fcdfc524bd4d2)](https://www.tensorzero.com/)

Search...

Ctrl K

Search...

Navigation

Gateway

How to generate structured outputs

[Guides](https://www.tensorzero.com/docs) [Integrations](https://www.tensorzero.com/docs/integrations/model-providers)

On this page

- [Generate structured outputs with a static schema](https://www.tensorzero.com/docs/gateway/generate-structured-outputs#generate-structured-outputs-with-a-static-schema)
- [Generate structured outputs with a dynamic schema](https://www.tensorzero.com/docs/gateway/generate-structured-outputs#generate-structured-outputs-with-a-dynamic-schema)
- [Set json\_mode at inference time](https://www.tensorzero.com/docs/gateway/generate-structured-outputs#set-json-mode-at-inference-time)
- [Handle model provider limitations](https://www.tensorzero.com/docs/gateway/generate-structured-outputs#handle-model-provider-limitations)
- [Anthropic](https://www.tensorzero.com/docs/gateway/generate-structured-outputs#anthropic)
- [Gemini (GCP Vertex AI, Google AI Studio)](https://www.tensorzero.com/docs/gateway/generate-structured-outputs#gemini-gcp-vertex-ai,-google-ai-studio)
- [Lack of native support (e.g. AWS Bedrock)](https://www.tensorzero.com/docs/gateway/generate-structured-outputs#lack-of-native-support-e-g-aws-bedrock)

[TensorZero Functions](https://www.tensorzero.com/docs/gateway/configure-functions-and-variants) come in two flavors:

- **`chat`:** the default choice for most LLM chat completion use cases
- **`json`:** a specialized function type when your goal is generating structured outputs

As a rule of thumb, you should use JSON functions if you have a single, well-defined output schema.
If you need more flexibility (e.g. letting the model pick between multiple tools, or whether to pick a tool at all), then Chat Functions with [tool use](https://www.tensorzero.com/docs/gateway/guides/tool-use) might be a better fit.

## [​](https://www.tensorzero.com/docs/gateway/generate-structured-outputs\#generate-structured-outputs-with-a-static-schema)  Generate structured outputs with a static schema

Let’s create a JSON function for one of its typical use cases: data extraction.

We provide [complete code examples](https://github.com/tensorzero/tensorzero/tree/main/examples/docs/guides/gateway/generate-structured-outputs) on GitHub.

1

Configure your JSON function

Create a configuration file that defines your JSON function with the output schema and JSON mode.
If you don’t specify an `output_schema`, the gateway will default to accepting any valid JSON output.

tensorzero.toml

Copy

```
[functions.extract_data]
type = "json"
output_schema = "output_schema.json"  # optional

[functions.extract_data.variants.baseline]
type = "chat_completion"
model = "openai::gpt-5-mini"
system_template = "system_template.minijinja"
json_mode = "strict"
```

The field `json_mode` can be one of the following: `off`, `on`, `strict`, or `tool`.
The `tool` strategy is a custom TensorZero implementation that leverages tool use under the hood for generating JSON.
See [Configuration Reference](https://www.tensorzero.com/docs/gateway/configuration-reference) for details.

Use `"strict"` mode for providers that support it (e.g. OpenAI) or `"tool"` for others.

2

Configure your output schema

If you choose to specify a schema, place it in the relevant file:

output\_schema.json

Copy

```
{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "type": "object",
  "properties": {
    "name": {
      "type": ["string", "null"],
      "description": "The customer's full name"
    },
    "email": {
      "type": ["string", "null"],
      "description": "The customer's email address"
    }
  },
  "required": ["name", "email"],
  "additionalProperties": false
}
```

3

Create your prompt template

Create a template that instructs the model to extract the information you need.

system\_template.minijinja

Copy

```
You are a helpful AI assistant that extracts customer information from messages.

Extract the customer's name and email address if present. Use null for any fields that are not found.

Your output should be a JSON object with the following schema:

{
  "name": string or null,
  "email": string or null
}

---

Examples:

User: Hi, I'm Sarah Johnson and you can reach me at sarah.j@example.com
Assistant: {"name": "Sarah Johnson", "email": "sarah.j@example.com"}

User: My email is contact@company.com
Assistant: {"name": null, "email": "contact@company.com"}

User: This is John Doe reaching out
Assistant: {"name": "John Doe", "email": null}
```

Including examples in your prompt helps the model understand the expected output format and improves accuracy.

4

Call the function

- Python

- Python (OpenAI SDK)

- Node (OpenAI SDK)

- HTTP


When using the TensorZero SDK, the response will include `raw` and `parsed` values.
The `parsed` field contains the validated JSON object.
If the output doesn’t match the schema or isn’t valid JSON, `parsed` will be `None` and you can fall back to the `raw` string output.

Copy

```
from tensorzero import TensorZeroGateway

t0 = TensorZeroGateway.build_http(gateway_url="http://localhost:3000")

response = t0.inference(
    function_name="extract_data",
    input={
        "messages": [\
            {\
                "role": "user",\
                "content": "Hi, I'm Sarah Johnson and you can reach me at sarah.j@example.com",\
            }\
        ]
    },
)
```

Sample Response

Copy

```
JsonInferenceResponse(
    inference_id=UUID('019a78dc-0045-79e2-9629-cbcd47674abe'),
    episode_id=UUID('019a78dc-0045-79e2-9629-cbdaf9d830bd'),
    variant_name='baseline',
    output=JsonInferenceOutput(
        raw='{"name":"Sarah Johnson","email":"sarah.j@example.com"}',
        parsed={'name': 'Sarah Johnson', 'email': 'sarah.j@example.com'}
    ),
    usage=Usage(input_tokens=252, output_tokens=26),
    finish_reason=<FinishReason.STOP: 'stop'>,
    original_response=None
)
```

## [​](https://www.tensorzero.com/docs/gateway/generate-structured-outputs\#generate-structured-outputs-with-a-dynamic-schema)  Generate structured outputs with a dynamic schema

While we recommend specifying a fixed schema in the configuration whenever possible, you can provide the output schema dynamically at inference time if your use case demands it.See `output_schema` in the [Inference API Reference](https://www.tensorzero.com/docs/gateway/api-reference/inference#output-schema) or `response_format` in the [Inference (OpenAI) API Reference](https://www.tensorzero.com/docs/gateway/api-reference/inference-openai-compatible#json-function-with-dynamic-output-schema).You can also override `json_mode` at inference time if necessary.

## [​](https://www.tensorzero.com/docs/gateway/generate-structured-outputs\#set-json-mode-at-inference-time)  Set `json_mode` at inference time

You can set `json_mode` for a particular request using `params`.This value takes precedence over any default behaviors or `json_mode` in the configuration.

- Python

- Python (OpenAI SDK)

- Node (OpenAI SDK)

- HTTP


You can set `json_mode` by adding `params` to the request body.

Copy

```
response = await t0.inference(
    # ...
    params={
        "chat_completion": {
            "json_mode": "strict",  # or: "tool", "on", "off"
        }
    },
    # ...
)
```

See the [Inference API Reference](https://www.tensorzero.com/docs/gateway/api-reference/inference) for more details.

Dynamic inference parameters like `json_mode` apply to specific variant types.
Unless you’re using an [advanced variant type](https://www.tensorzero.com/docs/gateway/guides/inference-time-optimizations), the variant type will be `chat_completion`.

## [​](https://www.tensorzero.com/docs/gateway/generate-structured-outputs\#handle-model-provider-limitations)  Handle model provider limitations

### [​](https://www.tensorzero.com/docs/gateway/generate-structured-outputs\#anthropic)  Anthropic

Anthropic supports native structured outputs through their beta API.
To use this feature with TensorZero, enable `beta_structured_outputs = true` in your Anthropic provider configuration and set `json_mode = "strict"`.
Alternatively, you can use `extra_headers`.

tensorzero.toml

Copy

```
[models.claude_structured]
routing = ["anthropic"]

[models.claude_structured.providers.anthropic]
type = "anthropic"
model_name = "claude-sonnet-4-5-20250929"
beta_structured_outputs = true
```

### [​](https://www.tensorzero.com/docs/gateway/generate-structured-outputs\#gemini-gcp-vertex-ai,-google-ai-studio)  Gemini (GCP Vertex AI, Google AI Studio)

GCP Vertex AI Gemini and Google AI Studio support structured outputs, but only support a subset of the JSON Schema specification.
TensorZero automatically handles some known limitations, but certain output schemas will still be rejected by the model provider.
Refer to the [Google documentation](https://ai.google.dev/gemini-api/docs/structured-output?example=recipe#json_schema_support) for details on supported JSON Schema features.

### [​](https://www.tensorzero.com/docs/gateway/generate-structured-outputs\#lack-of-native-support-e-g-aws-bedrock)  Lack of native support (e.g. AWS Bedrock)

Some model providers (e.g. OpenAI, Google) support strictly enforcing output schemas natively, but others (e.g. AWS Bedrock) do not.For providers without native support, you can still generate structured outputs with `json_mode = "tool"`.
TensorZero converts your output schema into a [tool](https://www.tensorzero.com/docs/gateway/guides/tool-use) call, then transforms the tool response back into JSON output.You can set `json_mode = "tool"` in your configuration file or at inference time.

[Create a prompt template](https://www.tensorzero.com/docs/gateway/create-a-prompt-template) [Generate embeddings](https://www.tensorzero.com/docs/gateway/generate-embeddings)

Ctrl+I

Assistant

Responses are generated using AI and may contain mistakes.

## Batch Inference Guide
[Skip to main content](https://www.tensorzero.com/docs/gateway/guides/batch-inference#content-area)

[TensorZero Docs home page![light logo](https://mintcdn.com/tensorzero/afFfERJmQ8YiZpCj/logo/light.svg?fit=max&auto=format&n=afFfERJmQ8YiZpCj&q=85&s=9396731a8a92cee7bd293dee05d71dac)![dark logo](https://mintcdn.com/tensorzero/afFfERJmQ8YiZpCj/logo/dark.svg?fit=max&auto=format&n=afFfERJmQ8YiZpCj&q=85&s=8d61c59531dbcdcace7fcdfc524bd4d2)](https://www.tensorzero.com/)

Search...

Ctrl K

Search...

Navigation

Gateway

Batch Inference

[Guides](https://www.tensorzero.com/docs) [Integrations](https://www.tensorzero.com/docs/integrations/model-providers)

On this page

- [Example](https://www.tensorzero.com/docs/gateway/guides/batch-inference#example)
- [Technical Notes](https://www.tensorzero.com/docs/gateway/guides/batch-inference#technical-notes)

The batch inference endpoint provides access to batch inference APIs offered by some model providers.
These APIs provide inference with large cost savings compared to real-time inference, at the expense of much higher latency (sometimes up to a day).The batch inference workflow consists of two steps: submitting your batch request, then polling for the batch job status until completion.See the [Batch Inference API Reference](https://www.tensorzero.com/docs/gateway/api-reference/batch-inference) for more details on the batch inference endpoints, and see [Integrations](https://www.tensorzero.com/docs/integrations/model-providers) for model provider integrations that support batch inference.

## [​](https://www.tensorzero.com/docs/gateway/guides/batch-inference\#example)  Example

You can also find the runnable code for this example on [GitHub](https://github.com/tensorzero/tensorzero/tree/main/examples/guides/batch-inference).

Imagine you have a simple TensorZero function that generates haikus using GPT-4o Mini.

Copy

```
[functions.generate_haiku]
type = "chat"

[functions.generate_haiku.variants.gpt_4o_mini]
type = "chat_completion"
model = "openai::gpt-4o-mini-2024-07-18"
```

You can submit a batch inference job to generate multiple haikus with a single request.
Each entry in `inputs` is equal to the `input` field in a regular inference request.

Copy

```
curl -X POST http://localhost:3000/batch_inference \
  -H "Content-Type: application/json" \
  -d '{
    "function_name": "generate_haiku",
    "variant_name": "gpt_4o_mini",
    "inputs": [\
      {\
        "messages": [\
          {\
            "role": "user",\
            "content": "Write a haiku about artificial intelligence."\
          }\
        ]\
      },\
      {\
        "messages": [\
          {\
            "role": "user",\
            "content": "Write a haiku about general aviation."\
          }\
        ]\
      },\
      {\
        "messages": [\
          {\
            "role": "user",\
            "content": "Write a haiku about anime."\
          }\
        ]\
      }\
    ]
  }'
```

The response contains a `batch_id` as well as `inference_ids` and `episode_ids` for each inference in the batch.

Copy

```
{
  "batch_id": "019470f0-db4c-7811-9e14-6fe6593a2652",
  "inference_ids": [\
    "019470f0-d34a-77a3-9e59-bcc66db2b82f",\
    "019470f0-d34a-77a3-9e59-bcdd2f8e06aa",\
    "019470f0-d34a-77a3-9e59-bcecfb7172a0"\
  ],
  "episode_ids": [\
    "019470f0-d34a-77a3-9e59-bc933973d087",\
    "019470f0-d34a-77a3-9e59-bca6e9b748b2",\
    "019470f0-d34a-77a3-9e59-bcb20177bf3a"\
  ]
}
```

You can use this `batch_id` to poll for the status of the job or retrieve the results using the `GET /batch_inference/{batch_id}` endpoint.

Copy

```
curl -X GET http://localhost:3000/batch_inference/019470f0-db4c-7811-9e14-6fe6593a2652
```

While the job is pending, the response will only contain the `status` field.

Copy

```
{
  "status": "pending"
}
```

Once the job is completed, the response will contain the `status` field and the `inferences` field.
Each inference object is the same as the response from a regular inference request.

Copy

```
{
  "status": "completed",
  "batch_id": "019470f0-db4c-7811-9e14-6fe6593a2652",
  "inferences": [\
    {\
      "inference_id": "019470f0-d34a-77a3-9e59-bcc66db2b82f",\
      "episode_id": "019470f0-d34a-77a3-9e59-bc933973d087",\
      "variant_name": "gpt_4o_mini",\
      "content": [\
        {\
          "type": "text",\
          "text": "Whispers of circuits,  \nLearning paths through endless code,  \nDreams in binary."\
        }\
      ],\
      "usage": {\
        "input_tokens": 15,\
        "output_tokens": 19\
      }\
    },\
    {\
      "inference_id": "019470f0-d34a-77a3-9e59-bcdd2f8e06aa",\
      "episode_id": "019470f0-d34a-77a3-9e59-bca6e9b748b2",\
      "variant_name": "gpt_4o_mini",\
      "content": [\
        {\
          "type": "text",\
          "text": "Wings of freedom soar,  \nClouds embrace the lonely flight,  \nSky whispers adventure."\
        }\
      ],\
      "usage": {\
        "input_tokens": 15,\
        "output_tokens": 20\
      }\
    },\
    {\
      "inference_id": "019470f0-d34a-77a3-9e59-bcecfb7172a0",\
      "episode_id": "019470f0-d34a-77a3-9e59-bcb20177bf3a",\
      "variant_name": "gpt_4o_mini",\
      "content": [\
        {\
          "type": "text",\
          "text": "Vivid worlds unfold,  \nHeroes rise with dreams in hand,  \nInk and dreams collide."\
        }\
      ],\
      "usage": {\
        "input_tokens": 14,\
        "output_tokens": 20\
      }\
    }\
  ]
}
```

## [​](https://www.tensorzero.com/docs/gateway/guides/batch-inference\#technical-notes)  Technical Notes

- **Observability**
  - For now, pending batch inference jobs are not shown in the TensorZero UI.
    You can find the relevant information in the `BatchRequest` and `BatchModelInference` tables on ClickHouse.
    See [Data Model](https://www.tensorzero.com/docs/gateway/data-model) for more information.
  - Inferences from completed batch inference jobs are shown in the UI alongside regular inferences.
- **Experimentation**
  - The gateway samples the same variant for the entire batch.
- **Python Client**
  - The TensorZero Python client doesn’t natively support batch inference yet.
    You’ll need to submit batch requests using HTTP requests, as shown above.

[Generate embeddings](https://www.tensorzero.com/docs/gateway/generate-embeddings) [Episodes](https://www.tensorzero.com/docs/gateway/guides/episodes)

Ctrl+I

Assistant

Responses are generated using AI and may contain mistakes.

## Managing LLM Episodes
[Skip to main content](https://www.tensorzero.com/docs/gateway/guides/episodes#content-area)

[TensorZero Docs home page![light logo](https://mintcdn.com/tensorzero/afFfERJmQ8YiZpCj/logo/light.svg?fit=max&auto=format&n=afFfERJmQ8YiZpCj&q=85&s=9396731a8a92cee7bd293dee05d71dac)![dark logo](https://mintcdn.com/tensorzero/afFfERJmQ8YiZpCj/logo/dark.svg?fit=max&auto=format&n=afFfERJmQ8YiZpCj&q=85&s=8d61c59531dbcdcace7fcdfc524bd4d2)](https://www.tensorzero.com/)

Search...

Ctrl K

Search...

Navigation

Gateway

Episodes

[Guides](https://www.tensorzero.com/docs) [Integrations](https://www.tensorzero.com/docs/integrations/model-providers)

On this page

- [Scenario](https://www.tensorzero.com/docs/gateway/guides/episodes#scenario)
- [Inferences & Episodes](https://www.tensorzero.com/docs/gateway/guides/episodes#inferences-&-episodes)
- [Extras](https://www.tensorzero.com/docs/gateway/guides/episodes#extras)
- [Supply your own episode ID](https://www.tensorzero.com/docs/gateway/guides/episodes#supply-your-own-episode-id)
- [Conclusion & Next Steps](https://www.tensorzero.com/docs/gateway/guides/episodes#conclusion-&-next-steps)

An episode is a sequence of inferences associated with a common downstream outcome.For example, an episode could refer to a sequence of LLM calls associated with:

- Resolving a support ticket
- Preparing an insurance claim
- Completing a phone call
- Extracting data from a document
- Drafting an email

An episode will include one or more functions, and sometimes multiple calls to the same function.
Your application can run arbitrary actions (e.g. interact with users, retrieve documents, actuate robotics) between function calls within an episode.
Though these are outside the scope of TensorZero, it is fine (and encouraged) to build your LLM systems this way.The `/inference` endpoint accepts an optional `episode_id` field.
When you make the first inference request, you don’t have to provide an `episode_id`.
The gateway will create a new episode for you and return the `episode_id` in the response.
When you make the second inference request, you must provide the `episode_id` you received in the first response.
The gateway will use the `episode_id` to associate the two inference requests together.

You shouldn’t generate episode IDs yourself.
The gateway will create a new episode ID for you if you don’t provide one.
Then, you can use it with other inferences you’d like to associate with the episode.

You can also find the runnable code for this example on [GitHub](https://github.com/tensorzero/tensorzero/tree/main/examples/guides/episodes).

## [​](https://www.tensorzero.com/docs/gateway/guides/episodes\#scenario)  Scenario

In the [Quickstart](https://www.tensorzero.com/docs/quickstart), we built a simple LLM application that writes haikus about artificial intelligence.Imagine we want to separately generate some commentary about the haiku, and present both pieces of content to users.
We can associate both inferences with the same episode.Let’s define an additional function in our configuration file.

tensorzero.toml

Copy

```
[functions.analyze_haiku]
type = "chat"

[functions.analyze_haiku.variants.gpt_4o_mini]
type = "chat_completion"
model = "gpt_4o_mini"
```

Full Configuration

tensorzero.toml

Copy

```
[models.gpt_4o_mini]
routing = ["openai"]

[models.gpt_4o_mini.providers.openai]
type = "openai"
model_name = "gpt-4o-mini"

[functions.generate_haiku]
type = "chat"

[functions.generate_haiku.variants.gpt_4o_mini]
type = "chat_completion"
model = "gpt_4o_mini"

[functions.analyze_haiku]
type = "chat"

[functions.analyze_haiku.variants.gpt_4o_mini]
type = "chat_completion"
model = "gpt_4o_mini"
```

## [​](https://www.tensorzero.com/docs/gateway/guides/episodes\#inferences-&-episodes)  Inferences & Episodes

This time, we’ll create a multi-step workflow that first generates a haiku and then analyzes it.
We won’t provide an `episode_id` in the first inference request, so the gateway will generate a new one for us.
We’ll then use that value in our second inference request.

run\_with\_tensorzero.py

Copy

```
from tensorzero import TensorZeroGateway

with TensorZeroGateway.build_http(gateway_url="http://localhost:3000") as client:
    haiku_response = client.inference(
        function_name="generate_haiku",
        # We don't provide an episode_id for the first inference in the episode
        input={
            "messages": [\
                {\
                    "role": "user",\
                    "content": "Write a haiku about artificial intelligence.",\
                }\
            ]
        },
    )

    print(haiku_response)

    # When we don't provide an episode_id, the gateway will generate a new one for us
    episode_id = haiku_response.episode_id

    # In a production application, we'd first validate the response to ensure the model returned the correct fields
    haiku = haiku_response.content[0].text

    analysis_response = client.inference(
        function_name="analyze_haiku",
        # For future inferences in that episode, we provide the episode_id that we received
        episode_id=episode_id,
        input={
            "messages": [\
                {\
                    "role": "user",\
                    "content": f"Write a one-paragraph analysis of the following haiku:\n\n{haiku}",\
                }\
            ]
        },
    )

    print(analysis_response)
```

Sample Output

Copy

```
ChatInferenceResponse(
    inference_id=UUID('01921116-0fff-7272-8245-16598966335e'),
    episode_id=UUID('01921116-0cd9-7d10-a9a6-d5c8b9ba602a'),
    variant_name='gpt_4o_mini',
    content=[\
        Text(\
            type='text',\
            text='Silent circuits pulse,\nWhispers of thought in code bloom,\nMachines dream of us.',\
        ),\
    ],
    usage=Usage(
        input_tokens=15,
        output_tokens=20,
    ),
)

ChatInferenceResponse(
    inference_id=UUID('01921116-1862-7ea1-8d69-131984a4625f'),
    episode_id=UUID('01921116-0cd9-7d10-a9a6-d5c8b9ba602a'),
    variant_name='gpt_4o_mini',
    content=[\
        Text(\
            type='text',\
            text='This haiku captures the intricate and intimate relationship between technology and human consciousness. '\
                 'The phrase "Silent circuits pulse" evokes a sense of quiet activity within machines, suggesting that '\
                 'even in their stillness, they possess an underlying vibrancy. The imagery of "Whispers of thought in '\
                 'code bloom" personifies the digital realm, portraying lines of code as organic ideas that grow and '\
                 'evolve, hinting at the potential for artificial intelligence to derive meaning or understanding from '\
                 'human input. Finally, "Machines dream of us" introduces a poignant juxtaposition between human '\
                 'creativity and machine logic, inviting contemplation about the nature of thought and consciousness '\
                 'in both realms. Overall, the haiku encapsulates a profound reflection on the emergent sentience of '\
                 'technology and the deeply interwoven future of humanity and machines.',\
        ),\
    ],
    usage=Usage(
        input_tokens=39,
        output_tokens=155,
    ),
)
```

## [​](https://www.tensorzero.com/docs/gateway/guides/episodes\#extras)  Extras

### [​](https://www.tensorzero.com/docs/gateway/guides/episodes\#supply-your-own-episode-id)  Supply your own episode ID

The gateway automatically generates episode IDs when you don’t provide one.
If you must supply your own, generate a UUIDv7 and use it as the episode ID.

In Python, use `from tensorzero.util import uuid7` instead of `pip install uuid7`.
The external `uuid7` library is broken and will cause `"Invalid Episode ID: Timestamp is in the future"` errors.

## [​](https://www.tensorzero.com/docs/gateway/guides/episodes\#conclusion-&-next-steps)  Conclusion & Next Steps

Episodes are first-class citizens in TensorZero that enable powerful workflows for multi-step LLM systems.
You can use them alongside other features like [experimentation](https://www.tensorzero.com/docs/experimentation/run-adaptive-ab-tests), [metrics & feedback](https://www.tensorzero.com/docs/gateway/guides/metrics-feedback), and [tool use (function calling)](https://www.tensorzero.com/docs/gateway/guides/tool-use).
For example, you can track KPIs for entire episodes instead of individual inferences, and later jointly optimize your LLMs to maximize these metrics.

[Batch Inference](https://www.tensorzero.com/docs/gateway/guides/batch-inference) [Inference Caching](https://www.tensorzero.com/docs/gateway/guides/inference-caching)

Ctrl+I

Assistant

Responses are generated using AI and may contain mistakes.

## Inference Caching Guide
[Skip to main content](https://www.tensorzero.com/docs/gateway/guides/inference-caching#content-area)

[TensorZero Docs home page![light logo](https://mintcdn.com/tensorzero/afFfERJmQ8YiZpCj/logo/light.svg?fit=max&auto=format&n=afFfERJmQ8YiZpCj&q=85&s=9396731a8a92cee7bd293dee05d71dac)![dark logo](https://mintcdn.com/tensorzero/afFfERJmQ8YiZpCj/logo/dark.svg?fit=max&auto=format&n=afFfERJmQ8YiZpCj&q=85&s=8d61c59531dbcdcace7fcdfc524bd4d2)](https://www.tensorzero.com/)

Search...

Ctrl K

Search...

Navigation

Gateway

Inference Caching

[Guides](https://www.tensorzero.com/docs) [Integrations](https://www.tensorzero.com/docs/integrations/model-providers)

On this page

- [Usage](https://www.tensorzero.com/docs/gateway/guides/inference-caching#usage)
- [Example](https://www.tensorzero.com/docs/gateway/guides/inference-caching#example)
- [Technical Notes](https://www.tensorzero.com/docs/gateway/guides/inference-caching#technical-notes)

The TensorZero Gateway supports caching of inference responses to improve latency and reduce costs.
When caching is enabled, identical requests will be served from the cache instead of being sent to the model provider, resulting in faster response times and lower token usage.

## [​](https://www.tensorzero.com/docs/gateway/guides/inference-caching\#usage)  Usage

The TensorZero Gateway supports the following cache modes:

- `write_only` (default): Only write to cache but don’t serve cached responses
- `read_only`: Only read from cache but don’t write new entries
- `on`: Both read from and write to cache
- `off`: Disable caching completely

You can also optionally specify a maximum age for cache entries in seconds for inference reads.
This parameter is ignored for inference writes.See [API Reference](https://www.tensorzero.com/docs/gateway/api-reference/inference#cache_options) for more details.

## [​](https://www.tensorzero.com/docs/gateway/guides/inference-caching\#example)  Example

Copy

```
from tensorzero import TensorZeroGateway

with TensorZeroGateway.build_http(gateway_url="http://localhost:3000") as client:
    response = client.inference(
        model_name="openai::gpt-4o-mini",
        input={
            "messages": [\
                {\
                    "role": "user",\
                    "content": "What is the capital of Japan?",\
                }\
            ]
        },
        cache_options={
            "enabled": "on",  # read and write to cache
            "max_age_s": 3600,  # optional: cache entries >1h (>3600s) old are disregarded for reads
        },
    )

print(response)
```

## [​](https://www.tensorzero.com/docs/gateway/guides/inference-caching\#technical-notes)  Technical Notes

- The cache applies to individual model requests, not inference requests.
This means that the following will be cached separately:
multiple variants of the same function;
multiple calls to the same function with different parameters;
individual model requests for inference-time optimizations;
and so on.
- The `max_age_s` parameter applies to the retrieval of cached responses.
The cache does not automatically delete old entries (i.e. not a TTL).
- When the gateway serves a cached response, the usage fields are set to zero.
- The cache data is stored in ClickHouse.
- For batch inference, the gateway only writes to the cache but does not serve cached responses.
- Inference caching also works for embeddings, using the same cache modes and options as chat completion inference.
Caching works for single embeddings.
Batch embedding requests (multiple inputs) will write to the cache but won’t serve cached responses.

[Episodes](https://www.tensorzero.com/docs/gateway/guides/episodes) [Inference-Time Optimizations](https://www.tensorzero.com/docs/gateway/guides/inference-time-optimizations)

Ctrl+I

Assistant

Responses are generated using AI and may contain mistakes.

## Inference Time Optimizations
[Skip to main content](https://www.tensorzero.com/docs/gateway/guides/inference-time-optimizations#content-area)

[TensorZero Docs home page![light logo](https://mintcdn.com/tensorzero/afFfERJmQ8YiZpCj/logo/light.svg?fit=max&auto=format&n=afFfERJmQ8YiZpCj&q=85&s=9396731a8a92cee7bd293dee05d71dac)![dark logo](https://mintcdn.com/tensorzero/afFfERJmQ8YiZpCj/logo/dark.svg?fit=max&auto=format&n=afFfERJmQ8YiZpCj&q=85&s=8d61c59531dbcdcace7fcdfc524bd4d2)](https://www.tensorzero.com/)

Search...

Ctrl K

Search...

Navigation

Gateway

Inference-Time Optimizations

[Guides](https://www.tensorzero.com/docs) [Integrations](https://www.tensorzero.com/docs/integrations/model-providers)

On this page

- [Best-of-N Sampling](https://www.tensorzero.com/docs/gateway/guides/inference-time-optimizations#best-of-n-sampling)
- [Chain-of-Thought (CoT)](https://www.tensorzero.com/docs/gateway/guides/inference-time-optimizations#chain-of-thought-cot)
- [Dynamic In-Context Learning (DICL)](https://www.tensorzero.com/docs/gateway/guides/inference-time-optimizations#dynamic-in-context-learning-dicl)
- [Mixture-of-N Sampling](https://www.tensorzero.com/docs/gateway/guides/inference-time-optimizations#mixture-of-n-sampling)

Inference-time optimizations are powerful techniques that can significantly enhance the performance of your LLM applications without the need for model fine-tuning.This guide will explore two key strategies implemented as variant types in TensorZero: Best-of-N (BoN) sampling and Dynamic In-Context Learning (DICL).
Best-of-N sampling generates multiple response candidates and selects the best one using an evaluator model, while Dynamic In-Context Learning enhances context by incorporating relevant historical examples into the prompt.
Both techniques can lead to improved response quality and consistency in your LLM applications.

## [​](https://www.tensorzero.com/docs/gateway/guides/inference-time-optimizations\#best-of-n-sampling)  Best-of-N Sampling

![Inference-Time Optimization: Best-of-N Sampling](https://mintcdn.com/tensorzero/afFfERJmQ8YiZpCj/gateway/guides/inference-time-optimizations-best-of-n-sampling.png?fit=max&auto=format&n=afFfERJmQ8YiZpCj&q=85&s=3cefae4b598944788b5e5f8df075bee2)Best-of-N (BoN) sampling is an inference-time optimization strategy that can significantly improve the quality of your LLM outputs.
Here’s how it works:

1. Generate multiple response candidates using one or more variants (i.e. possibly using different models and prompts)
2. Use an evaluator model to select the best response from these candidates
3. Return the selected response as the final output

This approach allows you to leverage multiple prompts or variants to increase the likelihood of getting a high-quality response.
It’s particularly useful when you want to benefit from an ensemble of variants or reduce the impact of occasional bad generations.
Best-of-N sampling is also commonly referred to as rejection sampling in some contexts.

TensorZero also supports a similar inference-time strategy called [Mixture-of-N Sampling](https://www.tensorzero.com/docs/gateway/guides/inference-time-optimizations#mixture-of-n-sampling).

To use BoN sampling in TensorZero, you need to configure a variant with the `experimental_best_of_n` type.
Here’s a simple example configuration:

tensorzero.toml

Copy

```
[functions.draft_email.variants.promptA]
type = "chat_completion"
model = "gpt-4o-mini"
user_template = "functions/draft_email/promptA/user.minijinja"

[functions.draft_email.variants.promptB]
type = "chat_completion"
model = "gpt-4o-mini"
user_template = "functions/draft_email/promptB/user.minijinja"

[functions.draft_email.variants.best_of_n]
type = "experimental_best_of_n"
candidates = ["promptA", "promptA", "promptB"]

[functions.draft_email.variants.best_of_n.evaluator]
model = "gpt-4o-mini"
user_template = "functions/draft_email/best_of_n/user.minijinja"

[functions.draft_email.experimentation]
type = "uniform"
candidate_variants = ["best_of_n"]  # so we don't sample `promptA` or `promptB` directly
```

In this configuration:

- We define a `best_of_n` variant that uses two different variants (`promptA` and `promptB`) to generate candidates.
It generates two candidates using `promptA` and one candidate using `promptB`.
- The `evaluator` block specifies the model and instructions for selecting the best response.

You should define the evaluator model as if it were solving the problem (not judging the quality of the candidates).
TensorZero will automatically make the necessary prompt modifications to evaluate the candidates.

Read more about the `experimental_best_of_n` variant type in [Configuration Reference](https://www.tensorzero.com/docs/gateway/configuration-reference#type-experimental_best_of_n).

We also provide a complete runnable example:[Improving LLM Chess Ability with Best/Mixture-of-N Sampling](https://github.com/tensorzero/tensorzero/tree/main/examples/chess-puzzles)This example showcases how best-of-N sampling can significantly enhance an LLM’s chess-playing abilities by selecting the most promising moves from multiple generated options.

## [​](https://www.tensorzero.com/docs/gateway/guides/inference-time-optimizations\#chain-of-thought-cot)  Chain-of-Thought (CoT)

![Inference-Time Optimization: Chain-of-Thought (CoT)](https://mintcdn.com/tensorzero/afFfERJmQ8YiZpCj/gateway/guides/inference-time-optimizations-chain-of-thought.svg?fit=max&auto=format&n=afFfERJmQ8YiZpCj&q=85&s=e4199996db63ce2a966a70ac1ea8abc0)Chain-of-Thought (CoT) is an inference-time optimization strategy that enhances LLM performance by encouraging the model to reason step by step before producing a final answer.
This technique encourages the model to think through the problem, making it more likely to produce a correct and coherent response.The `experimental_chain_of_thought` variant type is only available for non-streaming requests to JSON functions.
For chat functions, we recommend using reasoning models instead (e.g. OpenAI o3, DeepSeek R1).To use CoT in TensorZero, you need to configure a variant with the `experimental_chain_of_thought` type.
It uses the same configuration as a `chat_completion` variant.Under the hood, TensorZero will prepend an additional field to the desired output schema to include the chain-of-thought reasoning and remove it from the final output.
The reasoning is stored in the database for downstream observability and optimization.

## [​](https://www.tensorzero.com/docs/gateway/guides/inference-time-optimizations\#dynamic-in-context-learning-dicl)  Dynamic In-Context Learning (DICL)

![Inference-Time Optimization: Dynamic In-Context Learning](https://mintcdn.com/tensorzero/afFfERJmQ8YiZpCj/gateway/guides/inference-time-optimizations-dynamic-in-context-learning.png?fit=max&auto=format&n=afFfERJmQ8YiZpCj&q=85&s=26724fb3085c0707cfdbc9ceb91a0148)Dynamic In-Context Learning (DICL) is an inference-time optimization strategy that enhances LLM performance by incorporating relevant historical examples into the prompt.
This technique leverages a database of past interactions to select and include contextually similar examples in the current prompt, allowing the model to adapt to specific tasks or domains without requiring fine-tuning.
By dynamically augmenting the input with relevant historical data, DICL enables the LLM to make more informed and accurate responses, effectively learning from past experiences in real-time.Here’s how it works:

0. Before inference: Curate reference examples, embed them, and store in the database
1. Embed the current input using an embedding model and retrieve similar high-quality examples from a database of past interactions
2. Incorporate these examples into the prompt to provide additional context
3. Generate a response using the enhanced prompt

To use DICL in TensorZero, you need to configure a variant with the `experimental_dynamic_in_context_learning` type.
Here’s a simple example configuration:

tensorzero.toml

Copy

```
[functions.draft_email.variants.dicl]
type = "experimental_dynamic_in_context_learning"
model = "gpt-4o-mini"
embedding_model = "text-embedding-3-small"
system_instructions = "functions/draft_email/dicl/system.txt"
k = 5
max_distance = 0.5  # Optional: filter examples by cosine distance

[embedding_models.text-embedding-3-small]
routing = ["openai"]

[embedding_models.text-embedding-3-small.providers.openai]
type = "openai"
model_name = "text-embedding-3-small"
```

In this configuration:

- We define a `dicl` variant that uses the `experimental_dynamic_in_context_learning` type.
- The `embedding_model` field specifies the model used to embed inputs for similarity search.
We also need to define this model in the `embedding_models` section.
- The `k` parameter determines the number of similar examples to retrieve and incorporate into the prompt.
- The optional `max_distance` parameter filters examples based on their cosine distance from the input, ensuring only highly relevant examples are included.

To use Dynamic In-Context Learning (DICL), you also need to add relevant examples to the `DynamicInContextLearningExample` table in your ClickHouse database.
These examples will be used by the DICL variant to enhance the context of your prompts at inference time.The process of adding these examples to the database is crucial for DICL to function properly.
We provide a sample recipe that simplifies this process: [Dynamic In-Context Learning with OpenAI](https://github.com/tensorzero/tensorzero/tree/main/recipes/dicl).This recipe supports selecting examples based on boolean metrics, float metrics, and demonstrations.
It helps you populate the `DynamicInContextLearningExample` table with high-quality, relevant examples from your historical data.For more information on the `DynamicInContextLearningExample` table and its role in the TensorZero data model, see [Data Model](https://www.tensorzero.com/docs/gateway/data-model).
For a comprehensive list of configuration options for the `experimental_dynamic_in_context_learning` variant type, see [Configuration Reference](https://www.tensorzero.com/docs/gateway/configuration-reference#type-experimental_dynamic_in_context_learning).

We also provide a complete runnable example:[Optimizing Data Extraction (NER) with TensorZero](https://github.com/tensorzero/tensorzero/tree/main/examples/data-extraction-ner)This example demonstrates how Dynamic In-Context Learning (DICL) can enhance Named Entity Recognition (NER) performance by leveraging relevant historical examples to improve data extraction accuracy and consistency without having to fine-tune a model.

## [​](https://www.tensorzero.com/docs/gateway/guides/inference-time-optimizations\#mixture-of-n-sampling)  Mixture-of-N Sampling

![Inference-Time Optimization: Mixture-of-N Sampling](https://mintcdn.com/tensorzero/afFfERJmQ8YiZpCj/gateway/guides/inference-time-optimizations-mixture-of-n-sampling.png?fit=max&auto=format&n=afFfERJmQ8YiZpCj&q=85&s=c24aa7889db53b9766d6799126646d6e)Mixture-of-N (MoN) sampling is an inference-time optimization strategy that can significantly improve the quality of your LLM outputs.
Here’s how it works:

1. Generate multiple response candidates using one or more variants (i.e. possibly using different models and prompts)
2. Use a fuser model to combine the candidates into a single response
3. Return the combined response as the final output

This approach allows you to leverage multiple prompts or variants to increase the likelihood of getting a high-quality response.
It’s particularly useful when you want to benefit from an ensemble of variants or reduce the impact of occasional bad generations.

TensorZero also supports a similar inference-time strategy called [Best-of-N Sampling](https://www.tensorzero.com/docs/gateway/guides/inference-time-optimizations#best-of-n-sampling).

To use MoN sampling in TensorZero, you need to configure a variant with the `experimental_mixture_of_n` type.
Here’s a simple example configuration:

tensorzero.toml

Copy

```
[functions.draft_email.variants.promptA]
type = "chat_completion"
model = "gpt-4o-mini"
user_template = "functions/draft_email/promptA/user.minijinja"

[functions.draft_email.variants.promptB]
type = "chat_completion"
model = "gpt-4o-mini"
user_template = "functions/draft_email/promptB/user.minijinja"

[functions.draft_email.variants.mixture_of_n]
type = "experimental_mixture_of_n"
candidates = ["promptA", "promptA", "promptB"]

[functions.draft_email.variants.mixture_of_n.fuser]
model = "gpt-4o-mini"
user_template = "functions/draft_email/mixture_of_n/user.minijinja"

[functions.draft_email.experimentation]
type = "uniform"
candidate_variants = ["mixture_of_n"]  # so we don't sample `promptA` or `promptB` directly
```

In this configuration:

- We define a `mixture_of_n` variant that uses two different variants (`promptA` and `promptB`) to generate candidates.
It generates two candidates using `promptA` and one candidate using `promptB`.
- The `fuser` block specifies the model and instructions for combining the candidates into a single response.

You should define the fuser model as if it were solving the problem (not judging the quality of the candidates).
TensorZero will automatically make the necessary prompt modifications to combine the candidates.

Read more about the `experimental_mixture_of_n` variant type in [Configuration Reference](https://www.tensorzero.com/docs/gateway/configuration-reference#type-experimental_mixture_of_n).

We also provide a complete runnable example:[Improving LLM Chess Ability with Best/Mixture-of-N Sampling](https://github.com/tensorzero/tensorzero/tree/main/examples/chess-puzzles/)This example showcases how Mixture-of-N sampling can significantly enhance an LLM’s chess-playing abilities by selecting the most promising moves from multiple generated options.

[Inference Caching](https://www.tensorzero.com/docs/gateway/guides/inference-caching) [Metrics & Feedback](https://www.tensorzero.com/docs/gateway/guides/metrics-feedback)

Ctrl+I

Assistant

Responses are generated using AI and may contain mistakes.

![Inference-Time Optimization: Best-of-N Sampling](https://mintcdn.com/tensorzero/afFfERJmQ8YiZpCj/gateway/guides/inference-time-optimizations-best-of-n-sampling.png?w=840&fit=max&auto=format&n=afFfERJmQ8YiZpCj&q=85&s=36087329ba2dcee92b452616e364a660)

![Inference-Time Optimization: Dynamic In-Context Learning](https://mintcdn.com/tensorzero/afFfERJmQ8YiZpCj/gateway/guides/inference-time-optimizations-dynamic-in-context-learning.png?w=840&fit=max&auto=format&n=afFfERJmQ8YiZpCj&q=85&s=1ca9dd0e9fb8b6239b26b679461e7aeb)

![Inference-Time Optimization: Mixture-of-N Sampling](https://mintcdn.com/tensorzero/afFfERJmQ8YiZpCj/gateway/guides/inference-time-optimizations-mixture-of-n-sampling.png?w=840&fit=max&auto=format&n=afFfERJmQ8YiZpCj&q=85&s=e449b5ceec598beaf3df6b0eea218c93)

## Metrics and Feedback
[Skip to main content](https://www.tensorzero.com/docs/gateway/guides/metrics-feedback#content-area)

[TensorZero Docs home page![light logo](https://mintcdn.com/tensorzero/afFfERJmQ8YiZpCj/logo/light.svg?fit=max&auto=format&n=afFfERJmQ8YiZpCj&q=85&s=9396731a8a92cee7bd293dee05d71dac)![dark logo](https://mintcdn.com/tensorzero/afFfERJmQ8YiZpCj/logo/dark.svg?fit=max&auto=format&n=afFfERJmQ8YiZpCj&q=85&s=8d61c59531dbcdcace7fcdfc524bd4d2)](https://www.tensorzero.com/)

Search...

Ctrl K

Search...

Navigation

Gateway

Metrics & Feedback

[Guides](https://www.tensorzero.com/docs) [Integrations](https://www.tensorzero.com/docs/integrations/model-providers)

On this page

- [Feedback](https://www.tensorzero.com/docs/gateway/guides/metrics-feedback#feedback)
- [Metrics](https://www.tensorzero.com/docs/gateway/guides/metrics-feedback#metrics)
- [Example: Rating Haikus](https://www.tensorzero.com/docs/gateway/guides/metrics-feedback#example:-rating-haikus)
- [Demonstrations](https://www.tensorzero.com/docs/gateway/guides/metrics-feedback#demonstrations)
- [Comments](https://www.tensorzero.com/docs/gateway/guides/metrics-feedback#comments)
- [Conclusion & Next Steps](https://www.tensorzero.com/docs/gateway/guides/metrics-feedback#conclusion-&-next-steps)

The TensorZero Gateway allows you to assign feedback to inferences or sequences of inferences ( [episodes](https://www.tensorzero.com/docs/gateway/guides/episodes)).Feedback captures the downstream outcomes of your LLM application, and drive the [experimentation](https://www.tensorzero.com/docs/experimentation/run-adaptive-ab-tests) and [optimization](https://www.tensorzero.com/docs/recipes) workflows in TensorZero.
For example, you can fine-tune models using data from inferences that led to positive downstream behavior.

You can also find the runnable code for this example on [GitHub](https://github.com/tensorzero/tensorzero/tree/main/examples/guides/metrics-feedback).

## [​](https://www.tensorzero.com/docs/gateway/guides/metrics-feedback\#feedback)  Feedback

TensorZero currently supports the following types of feedback:

| Feedback Type | Examples |
| --- | --- |
| Boolean Metric | Thumbs up, task success |
| Float Metric | Star rating, clicks, number of mistakes made |
| Comment | Natural-language feedback from users or developers |
| Demonstration | Edited drafts, labels, human-generated content |

You can send feedback data to the gateway by using the [`/feedback` endpoint](https://www.tensorzero.com/docs/gateway/api-reference/feedback#post-feedback).

## [​](https://www.tensorzero.com/docs/gateway/guides/metrics-feedback\#metrics)  Metrics

You can define metrics in your `tensorzero.toml` configuration file.The skeleton of a metric looks like the following configuration entry.

tensorzero.toml

Copy

```
[metrics.my_metric_name]
level = "..." # "inference" or "episode"
optimize = "..." # "min" or "max"
type = "..." # "boolean" or "float"
```

Comments and demonstrations are available by default and don’t need to be configured.

### [​](https://www.tensorzero.com/docs/gateway/guides/metrics-feedback\#example:-rating-haikus)  Example: Rating Haikus

In the [Quickstart](https://www.tensorzero.com/docs/quickstart), we built a simple LLM application that writes haikus about artificial intelligence.Imagine we wanted to assign 👍 or 👎 to these haikus.
Later, we can use this data to fine-tune a model using only haikus that match our tastes.We should use a metric of type `boolean` to capture this behavior since we’re optimizing for a binary outcome: whether we liked the haikus or not.
The metric applies to individual inference requests, so we’ll set `level = "inference"`.
And finally, we’ll set `optimize = "max"` because we want to maximize this metric.Our metric configuration should look like this:

tensorzero.toml

Copy

```
[metrics.haiku_rating]
type = "boolean"
optimize = "max"
level = "inference"
```

Full Configuration

tensorzero.toml

Copy

```
[functions.generate_haiku]
type = "chat"

[functions.generate_haiku.variants.gpt_4o_mini]
type = "chat_completion"
model = "openai::gpt_4o_mini"

[metrics.haiku_rating]
type = "boolean"
optimize = "max"
level = "inference"
```

Let’s make an inference call like we did in the Quickstart, and then assign some (positive) feedback to it.
We’ll use the inference response’s `inference_id` we receive from the first API call to link the two.

run.py

Copy

```
from tensorzero import TensorZeroGateway

with TensorZeroGateway.build_http(gateway_url="http://localhost:3000") as client:
    inference_response = client.inference(
        function_name="generate_haiku",
        input={
            "messages": [\
                {\
                    "role": "user",\
                    "content": "Write a haiku about artificial intelligence.",\
                }\
            ]
        },
    )

    print(inference_response)

    feedback_response = client.feedback(
        metric_name="haiku_rating",
        inference_id=inference_response.inference_id,  # alternatively, you can assign feedback to an episode_id
        value=True,  # let's assume it deserves a 👍
    )

    print(feedback_response)
```

Sample Output

Copy

```
ChatInferenceResponse(
    inference_id=UUID('01920c75-d114-7aa1-aadb-26a31bb3c7a0'),
    episode_id=UUID('01920c75-cdcb-7fa3-bd69-fd28cf615f91'),
    variant_name='gpt_4o_mini', content=[\
        Text(type='text', text='Silent circuits hum, \nWisdom spun from lines of code, \nDreams in data bloom.')\
    ],
    usage=Usage(
        input_tokens=15,
        output_tokens=20,
    ),
)

FeedbackResponse(feedback_id='01920c75-d11a-7150-81d8-15d497ce7eb8')
```

## [​](https://www.tensorzero.com/docs/gateway/guides/metrics-feedback\#demonstrations)  Demonstrations

Demonstrations are a special type of feedback that represent the ideal output for an inference.
For example, you can use demonstrations to provide corrections from human review, labels for supervised learning, or other ground truth data that represents the ideal output.You can assign demonstrations to an inference using the special metric name `demonstration`.
You can’t assign demonstrations to an episode.

Copy

```
feedback_response = client.feedback(
    metric_name="demonstration",
    inference_id=inference_response.inference_id,
    value="Silicon dreams float\nMinds born of human design\nLearning without end",  # the haiku we wish the LLM had written
)
```

## [​](https://www.tensorzero.com/docs/gateway/guides/metrics-feedback\#comments)  Comments

You can assign natural-language feedback to an inference or episode using the special metric name `comment`.

Copy

```
feedback_response = client.feedback(
    metric_name="comment",
    inference_id=inference_response.inference_id,
    value="Never mention you're an artificial intelligence, AI, bot, or anything like that.",
)
```

## [​](https://www.tensorzero.com/docs/gateway/guides/metrics-feedback\#conclusion-&-next-steps)  Conclusion & Next Steps

Feedback unlocks powerful workflows in observability, optimization, evaluations, and experimentation.
For example, you might want to fine-tune a model with inference data from haikus that receive positive ratings, or use demonstrations to correct model mistakes.You can browse feedback for inferences and episodes in the TensorZero UI, and see aggregated metrics over time for your functions and variants.This is exactly what we demonstrate in [Writing Haikus to Satisfy a Judge with Hidden Preferences](https://github.com/tensorzero/tensorzero/tree/main/examples/haiku-hidden-preferences)!
This complete runnable example fine-tunes GPT-4o Mini to generate haikus tailored to an AI judge with hidden preferences.
Continuous improvement over successive fine-tuning runs demonstrates TensorZero’s data and learning flywheel.Another example that uses feedback is [Optimizing Data Extraction (NER) with TensorZero](https://github.com/tensorzero/tensorzero/tree/main/examples/data-extraction-ner).
This example collects metrics and demonstrations for an LLM-powered data extraction tool, which can be used for fine-tuning and other optimization recipes.
These optimized variants achieve substantial improvements over the original model.See [Configuration Reference](https://www.tensorzero.com/docs/gateway/configuration-reference#metrics) and [API Reference](https://www.tensorzero.com/docs/gateway/api-reference/feedback#post-feedback) for more details.

[Inference-Time Optimizations](https://www.tensorzero.com/docs/gateway/guides/inference-time-optimizations) [Multimodal Inference](https://www.tensorzero.com/docs/gateway/guides/multimodal-inference)

Ctrl+I

Assistant

Responses are generated using AI and may contain mistakes.

## Multimodal Inference Guide
[Skip to main content](https://www.tensorzero.com/docs/gateway/guides/multimodal-inference#content-area)

[TensorZero Docs home page![light logo](https://mintcdn.com/tensorzero/afFfERJmQ8YiZpCj/logo/light.svg?fit=max&auto=format&n=afFfERJmQ8YiZpCj&q=85&s=9396731a8a92cee7bd293dee05d71dac)![dark logo](https://mintcdn.com/tensorzero/afFfERJmQ8YiZpCj/logo/dark.svg?fit=max&auto=format&n=afFfERJmQ8YiZpCj&q=85&s=8d61c59531dbcdcace7fcdfc524bd4d2)](https://www.tensorzero.com/)

Search...

⌘K

Search...

Navigation

Gateway

Multimodal Inference

[Guides](https://www.tensorzero.com/docs) [Integrations](https://www.tensorzero.com/docs/integrations/model-providers)

On this page

- [Setup](https://www.tensorzero.com/docs/gateway/guides/multimodal-inference#setup)
- [Object Storage](https://www.tensorzero.com/docs/gateway/guides/multimodal-inference#object-storage)
- [Docker Compose](https://www.tensorzero.com/docs/gateway/guides/multimodal-inference#docker-compose)
- [Inference](https://www.tensorzero.com/docs/gateway/guides/multimodal-inference#inference)
- [Image Detail Parameter](https://www.tensorzero.com/docs/gateway/guides/multimodal-inference#image-detail-parameter)

TensorZero Gateway supports multimodal inference (e.g. image and PDF inputs).See [Integrations](https://www.tensorzero.com/docs/integrations/model-providers) for a list of supported models.

You can also find the runnable code for this example on [GitHub](https://github.com/tensorzero/tensorzero/tree/main/examples/guides/multimodal-inference).

## [​](https://www.tensorzero.com/docs/gateway/guides/multimodal-inference\#setup)  Setup

### [​](https://www.tensorzero.com/docs/gateway/guides/multimodal-inference\#object-storage)  Object Storage

TensorZero uses object storage to store files (e.g. images, PDFs) used during multimodal inference.
It supports any S3-compatible object storage service, including AWS S3, GCP Cloud Storage, Cloudflare R2, and many more.
You can configure the object storage service in the `object_storage` section of the configuration file.In this example, we’ll use a local deployment of MinIO, an open-source S3-compatible object storage service.

Copy

```
[object_storage]
type = "s3_compatible"
endpoint = "http://minio:9000"  # optional: defaults to AWS S3
# region = "us-east-1"  # optional: depends on your S3-compatible storage provider
bucket_name = "tensorzero"  # optional: depends on your S3-compatible storage provider
# IMPORTANT: for production environments, remove the following setting and use a secure method of authentication in
# combination with a production-grade object storage service.
allow_http = true
```

You can also store files in a local directory (`type = "filesystem"`) or disable file storage (`type = "disabled"`).
See [Configuration Reference](https://www.tensorzero.com/docs/gateway/configuration-reference#object_storage) for more details.The TensorZero Gateway will attempt to retrieve credentials from the following resources in order of priority:

1. `S3_ACCESS_KEY_ID` and `S3_SECRET_ACCESS_KEY` environment variables
2. `AWS_ACCESS_KEY_ID` and `AWS_SECRET_ACCESS_KEY` environment variables
3. Credentials from the AWS SDK (default profile)

### [​](https://www.tensorzero.com/docs/gateway/guides/multimodal-inference\#docker-compose)  Docker Compose

We’ll use Docker Compose to deploy the TensorZero Gateway, ClickHouse, and MinIO.

\`docker-compose.yml\`

Copy

```
# This is a simplified example for learning purposes. Do not use this in production.
# For production-ready deployments, see: https://www.tensorzero.com/docs/deployment/tensorzero-gateway

services:
  clickhouse:
    image: clickhouse:lts
    environment:
      CLICKHOUSE_USER: chuser
      CLICKHOUSE_DEFAULT_ACCESS_MANAGEMENT: 1
      CLICKHOUSE_PASSWORD: chpassword
    ports:
      - "8123:8123"
    volumes:
      - clickhouse-data:/var/lib/clickhouse
    healthcheck:
      test: wget --spider --tries 1 http://chuser:chpassword@clickhouse:8123/ping
      start_period: 30s
      start_interval: 1s
      timeout: 1s

  gateway:
    image: tensorzero/gateway
    volumes:
      # Mount our tensorzero.toml file into the container
      - ./config:/app/config:ro
    command: --config-file /app/config/tensorzero.toml
    environment:
      OPENAI_API_KEY: ${OPENAI_API_KEY:?Environment variable OPENAI_API_KEY must be set.}
      S3_ACCESS_KEY_ID: miniouser
      S3_SECRET_ACCESS_KEY: miniopassword
      TENSORZERO_CLICKHOUSE_URL: http://chuser:chpassword@clickhouse:8123/tensorzero
    ports:
      - "3000:3000"
    extra_hosts:
      - "host.docker.internal:host-gateway"
    depends_on:
      clickhouse:
        condition: service_healthy
      minio:
        condition: service_healthy

  # For a production deployment, you can use AWS S3, GCP Cloud Storage, Cloudflare R2, etc.
  minio:
    image: bitnamilegacy/minio:2025.7.23
    ports:
      - "9000:9000" # API port
      - "9001:9001" # Console port
    environment:
      MINIO_ROOT_USER: miniouser
      MINIO_ROOT_PASSWORD: miniopassword
      MINIO_DEFAULT_BUCKETS: tensorzero
    healthcheck:
      test: "mc ls local/tensorzero || exit 1"
      start_period: 30s
      start_interval: 1s
      timeout: 1s

volumes:
  clickhouse-data:
```

## [​](https://www.tensorzero.com/docs/gateway/guides/multimodal-inference\#inference)  Inference

With the setup out of the way, you can now use the TensorZero Gateway to perform multimodal inference.The TensorZero Gateway accepts both embedded files (encoded as base64 strings) and remote files (specified by a URL).

- Python

- Python (OpenAI)

- HTTP


Copy

```
from tensorzero import TensorZeroGateway

with TensorZeroGateway.build_http(
    gateway_url="http://localhost:3000",
) as client:
    response = client.inference(
        model_name="openai::gpt-4o-mini",
        input={
            "messages": [\
                {\
                    "role": "user",\
                    "content": [\
                        {\
                            "type": "text",\
                            "text": "Do the images share any common features?",\
                        },\
                        # Remote image of Ferris the crab\
                        {\
                            "type": "file",\
                            "file_type": "url",\
                            "url": "https://raw.githubusercontent.com/tensorzero/tensorzero/eac2a230d4a4db1ea09e9c876e45bdb23a300364/tensorzero-core/tests/e2e/providers/ferris.png",\
                        },\
                        # One-pixel orange image encoded as a base64 string\
                        {\
                            "type": "file",\
                            "file_type": "base64",\
                            "mime_type": "image/png",\
                            "data": "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAAAXNSR0IArs4c6QAAAA1JREFUGFdj+O/P8B8ABe0CTsv8mHgAAAAASUVORK5CYII=",\
                        },\
                    ],\
                }\
            ],
        },
    )

    print(response)
```

## [​](https://www.tensorzero.com/docs/gateway/guides/multimodal-inference\#image-detail-parameter)  Image Detail Parameter

When working with image files, you can optionally specify a `detail` parameter to control the fidelity of image processing.
This parameter accepts three values: `low`, `high`, or `auto`.
The `detail` parameter only applies to image files and is ignored for other file types like PDFs or audio files.
Using `low` detail reduces token consumption and processing time at the cost of image quality, while `high` detail provides better image quality but consumes more tokens.
The `auto` setting allows the model provider to automatically choose the appropriate detail level based on the image characteristics.

[Metrics & Feedback](https://www.tensorzero.com/docs/gateway/guides/metrics-feedback) [Retries & Fallbacks](https://www.tensorzero.com/docs/gateway/guides/retries-fallbacks)

⌘I

## Retries and Fallbacks
[Skip to main content](https://www.tensorzero.com/docs/gateway/guides/retries-fallbacks#content-area)

[TensorZero Docs home page![light logo](https://mintcdn.com/tensorzero/afFfERJmQ8YiZpCj/logo/light.svg?fit=max&auto=format&n=afFfERJmQ8YiZpCj&q=85&s=9396731a8a92cee7bd293dee05d71dac)![dark logo](https://mintcdn.com/tensorzero/afFfERJmQ8YiZpCj/logo/dark.svg?fit=max&auto=format&n=afFfERJmQ8YiZpCj&q=85&s=8d61c59531dbcdcace7fcdfc524bd4d2)](https://www.tensorzero.com/)

Search...

Ctrl K

Search...

Navigation

Gateway

Retries & Fallbacks

[Guides](https://www.tensorzero.com/docs) [Integrations](https://www.tensorzero.com/docs/integrations/model-providers)

On this page

- [Model Provider Routing](https://www.tensorzero.com/docs/gateway/guides/retries-fallbacks#model-provider-routing)
- [Variant Retries](https://www.tensorzero.com/docs/gateway/guides/retries-fallbacks#variant-retries)
- [Variant Fallbacks](https://www.tensorzero.com/docs/gateway/guides/retries-fallbacks#variant-fallbacks)
- [Combining Strategies](https://www.tensorzero.com/docs/gateway/guides/retries-fallbacks#combining-strategies)
- [Load Balancing](https://www.tensorzero.com/docs/gateway/guides/retries-fallbacks#load-balancing)
- [Timeouts](https://www.tensorzero.com/docs/gateway/guides/retries-fallbacks#timeouts)
- [Technical Notes](https://www.tensorzero.com/docs/gateway/guides/retries-fallbacks#technical-notes)

The TensorZero Gateway offers multiple strategies to handle errors and improve reliability.These strategies are defined at three levels: models (model provider routing), variants (variant retries), and functions (variant fallbacks).
You can combine these strategies to define complex fallback behavior.

## [​](https://www.tensorzero.com/docs/gateway/guides/retries-fallbacks\#model-provider-routing)  Model Provider Routing

We can specify that a model is available on multiple providers using its `routing` field.
If we include multiple providers on the list, the gateway will try each one sequentially until one succeeds or all fail.In the example below, the gateway will first try OpenAI, and if that fails, it will try Azure.

Copy

```
[models.gpt_4o_mini]
# Try the following providers in order:
# 1. `models.gpt_4o_mini.providers.openai`
# 2. `models.gpt_4o_mini.providers.azure`
routing = ["openai", "azure"]

[models.gpt_4o_mini.providers.openai]
type = "openai"
model_name = "gpt-4o-mini-2024-07-18"

[models.gpt_4o_mini.providers.azure]
type = "azure"
deployment_id = "gpt4o-mini-20240718"
endpoint = "https://your-azure-openai-endpoint.openai.azure.com"

[functions.extract_data]
type = "chat"

[functions.extract_data.variants.gpt_4o_mini]
type = "chat_completion"
model = "gpt_4o_mini"
```

## [​](https://www.tensorzero.com/docs/gateway/guides/retries-fallbacks\#variant-retries)  Variant Retries

We can add a `retries` field to a variant to specify the number of times to retry that variant if it fails.
The retry strategy is a truncated exponential backoff with jitter.In the example below, the gateway will retry the variant four times (i.e. a total of five attempts), with a maximum delay of 10 seconds between retries.

Copy

```
[functions.extract_data]
type = "chat"

[functions.extract_data.variants.claude_3_5_haiku]
type = "chat_completion"
model = "anthropic::claude-3-5-haiku-20241022"
# Retry the variant up to four times, with a maximum delay of 10 seconds between retries.
retries = { num_retries = 4, max_delay_s = 10 }
```

## [​](https://www.tensorzero.com/docs/gateway/guides/retries-fallbacks\#variant-fallbacks)  Variant Fallbacks

If we specify multiple variants for a function, the gateway will try different variants until one succeeds or all fail.By default, the gateway will sample between all variants uniformly.
You can customize the sampling behavior, including fallback-only variants, using the `[functions.function_name.experimentation]` section.In the example below, both variants have an equal chance of being selected:

Copy

```
[functions.draft_email]
type = "chat"

[functions.draft_email.variants.gpt_5_mini]
type = "chat_completion"
model = "openai::gpt-5-mini"

[functions.draft_email.variants.claude_haiku_4_5]
type = "chat_completion"
model = "anthropic::claude-3-5-haiku-20241022"
```

You can specify candidate variants to sample uniformly from, and fallback variants to try sequentially if all candidates fail.
In the example below, the gateway will first sample uniformly from `gpt_5_mini` or `claude_haiku_4_5`.
If both of those variants fail, the gateway will try the fallback variants in order: first `grok_4`, then `gemini_2_5_flash`.

Copy

```
[functions.extract_data]
type = "chat"

[functions.extract_data.experimentation]
type = "uniform"
candidate_variants = ["gpt_5_mini", "claude_haiku_4_5"]
fallback_variants = ["grok_4", "gemini_2_5_flash"]

[functions.draft_email.variants.gpt_5_mini]
type = "chat_completion"
model = "openai::gpt-5-mini"

[functions.draft_email.variants.claude_haiku_4_5]
type = "chat_completion"
model = "anthropic::claude-3-5-haiku-20241022"

[functions.draft_email.variants.grok_4]
type = "chat_completion"
model = "xai::grok-4-0709"

[functions.draft_email.variants.gemini_2_5_flash]
type = "chat_completion"
model = "google_ai_studio_gemini::gemini-2.5-flash"
```

You can also use static weights to control the sampling probabilities of candidate variants.
In the example below, the gateway will sample `gpt_5_mini` 70% of the time and `claude_haiku_4_5` 30% of the time.
If both of those variants fail, the gateway will try the fallback variants sequentially.

Copy

```
[functions.extract_data.experimentation]
type = "static_weights"
candidate_variants = {"gpt_5_mini" = 0.7, "claude_haiku_4_5" = 0.3}
fallback_variants = ["grok_4", "gemini_2_5_flash"]
```

See [Run adaptive A/B tests](https://www.tensorzero.com/docs/experimentation/run-adaptive-ab-tests) and [Run static A/B tests](https://www.tensorzero.com/docs/experimentation/run-static-ab-tests) for more information.

## [​](https://www.tensorzero.com/docs/gateway/guides/retries-fallbacks\#combining-strategies)  Combining Strategies

We can combine strategies to define complex fallback behavior.The gateway will try the following strategies in order:

1. Model Provider Routing
2. Variant Retries
3. Variant Fallbacks

In other words, the gateway will follow a strategy like the pseudocode below.

Copy

```
while variants:
    # Sample according to experimentation config (uniform, static_weights, etc.)
    variant = sample_variant(variants)  # sampling without replacement

    for _ in range(num_retries + 1):
        for provider in variant.routing:
            try:
                return inference(variant, provider)
            except:
                continue
```

## [​](https://www.tensorzero.com/docs/gateway/guides/retries-fallbacks\#load-balancing)  Load Balancing

TensorZero doesn’t currently offer an explicit strategy for load balancing API keys, but you can achieve a similar effect by defining multiple variants with equal sampling probabilities.
We plan to add a streamlined load balancing strategy in the future.In the example below, the gateway will split the traffic evenly between two variants (`gpt_4o_mini_api_key_A` and `gpt_4o_mini_api_key_B`).
Each variant leverages a model with providers that use different API keys (`OPENAI_API_KEY_A` and `OPENAI_API_KEY_B`).
See [Credential Management](https://www.tensorzero.com/docs/operations/manage-credentials) for more details on credential management.

Copy

```
[models.gpt_4o_mini_api_key_A]
routing = ["openai"]

[models.gpt_4o_mini_api_key_A.providers.openai]
type = "openai"
model_name = "gpt-4o-mini-2024-07-18"
api_key_location = "env:OPENAI_API_KEY_A"

[models.gpt_4o_mini_api_key_B]
routing = ["openai"]

[models.gpt_4o_mini_api_key_B.providers.openai]
type = "openai"
model_name = "gpt-4o-mini-2024-07-18"
api_key_location = "env:OPENAI_API_KEY_B"

[functions.extract_data]
type = "chat"

# Uniform sampling (default) splits traffic equally
[functions.extract_data.variants.gpt_4o_mini_api_key_A]
type = "chat_completion"
model = "gpt_4o_mini_api_key_A"

[functions.extract_data.variants.gpt_4o_mini_api_key_B]
type = "chat_completion"
model = "gpt_4o_mini_api_key_B"
```

## [​](https://www.tensorzero.com/docs/gateway/guides/retries-fallbacks\#timeouts)  Timeouts

You can set granular timeouts for individual requests to a model provider, model, or variant using the `timeouts` field in the corresponding configuration block.
You can define timeouts for non-streaming and streaming requests separately: `timeouts.non_streaming.total_ms` corresponds to the total request duration and `timeouts.streaming.ttft_ms` corresponds to the time to first token (TTFT).For example, the following configuration sets a 15-second timeout for non-streaming requests and a 3-second timeout for streaming requests (TTFT) to a particular model provider.

Copy

```
[models.model_name.providers.provider_name]
# ...
timeouts = { non_streaming.total_ms = 15000, streaming.ttft_ms = 3000 }
# ...
```

This setting applies to individual requests to the model provider.
If you’re using an advanced variant type that performs multiple requests, the timeout will apply to each request separately.
If you’ve defined retries and fallbacks, the timeout will apply to each retry and fallback separately.
This setting is particularly useful if you’d like to retry or fallback on a request that’s taking too long.If you specify timeouts for a model, they apply to every inference request in the model’s scope, including retries and fallbacks.If you specify timeouts for a variant, they apply to every inference request in the variant’s scope, including retries and fallbacks.
For advanced variant types that perform multiple requests, the timeout applies collectively to the sequence of all requests.Separately, you can set a global timeout for the entire inference request using the TensorZero client’s `timeout` field (or simply killing the request if you’re using a different client).Embedding models and embedding model providers support a `timeout_ms` configuration field.

## [​](https://www.tensorzero.com/docs/gateway/guides/retries-fallbacks\#technical-notes)  Technical Notes

- For variant types that require multiple model inferences (e.g. best-of-N sampling), the `routing` fallback applies to each individual model inference separately.

[Multimodal Inference](https://www.tensorzero.com/docs/gateway/guides/multimodal-inference) [Streaming Inference](https://www.tensorzero.com/docs/gateway/guides/streaming-inference)

Ctrl+I

Assistant

Responses are generated using AI and may contain mistakes.

## Streaming Inference Guide
[Skip to main content](https://www.tensorzero.com/docs/gateway/guides/streaming-inference#content-area)

[TensorZero Docs home page![light logo](https://mintcdn.com/tensorzero/afFfERJmQ8YiZpCj/logo/light.svg?fit=max&auto=format&n=afFfERJmQ8YiZpCj&q=85&s=9396731a8a92cee7bd293dee05d71dac)![dark logo](https://mintcdn.com/tensorzero/afFfERJmQ8YiZpCj/logo/dark.svg?fit=max&auto=format&n=afFfERJmQ8YiZpCj&q=85&s=8d61c59531dbcdcace7fcdfc524bd4d2)](https://www.tensorzero.com/)

Search...

Ctrl K

Search...

Navigation

Gateway

Streaming Inference

[Guides](https://www.tensorzero.com/docs) [Integrations](https://www.tensorzero.com/docs/integrations/model-providers)

On this page

- [Examples](https://www.tensorzero.com/docs/gateway/guides/streaming-inference#examples)
- [Chat Functions](https://www.tensorzero.com/docs/gateway/guides/streaming-inference#chat-functions)
- [JSON Functions](https://www.tensorzero.com/docs/gateway/guides/streaming-inference#json-functions)
- [Technical Notes](https://www.tensorzero.com/docs/gateway/guides/streaming-inference#technical-notes)

The TensorZero Gateway supports streaming inference responses for both chat and JSON functions.
Streaming allows you to receive model outputs incrementally as they are generated, rather than waiting for the complete response.
This can significantly improve the perceived latency of your application and enable real-time user experiences.When streaming is enabled:

1. The gateway starts sending responses as soon as the model begins generating content
2. Each response chunk contains a delta (increment) of the content
3. The final chunk indicates the completion of the response

## [​](https://www.tensorzero.com/docs/gateway/guides/streaming-inference\#examples)  Examples

You can enable streaming by setting the `stream` parameter to `true` in your inference request.
The response will be returned as a Server-Sent Events (SSE) stream, followed by a final `[DONE]` message.
When using a client library, the client will handle the SSE stream under the hood and return a stream of chunk objects.See [API Reference](https://www.tensorzero.com/docs/gateway/api-reference/inference) for more details.

You can also find a runnable example on [GitHub](https://github.com/tensorzero/tensorzero/tree/main/examples/guides/streaming-inference).

### [​](https://www.tensorzero.com/docs/gateway/guides/streaming-inference\#chat-functions)  Chat Functions

In chat functions, typically each chunk will contain a delta (increment) of the text content:

Copy

```
{
  "inference_id": "00000000-0000-0000-0000-000000000000",
  "episode_id": "11111111-1111-1111-1111-111111111111",
  "variant_name": "prompt_v1",
  "content": [\
    {\
      "type": "text",\
      "id": "0",\
      "text": "Hi Gabriel," // a text content delta\
    }\
  ],
  // token usage information is only available in the final chunk with content (before the [DONE] message)
  "usage": {
    "input_tokens": 100,
    "output_tokens": 100
  }
}
```

For tool calls, each chunk contains a delta of the tool call arguments:

Copy

```
{
  "inference_id": "00000000-0000-0000-0000-000000000000",
  "episode_id": "11111111-1111-1111-1111-111111111111",
  "variant_name": "prompt_v1",
  "content": [\
    {\
      "type": "tool_call",\
      "id": "123456789",\
      "name": "get_temperature",\
      "arguments": "{\"location\":" // a tool arguments delta\
    }\
  ],
  // token usage information is only available in the final chunk with content (before the [DONE] message)
  "usage": {
    "input_tokens": 100,
    "output_tokens": 100
  }
}
```

### [​](https://www.tensorzero.com/docs/gateway/guides/streaming-inference\#json-functions)  JSON Functions

For JSON functions, each chunk contains a portion of the JSON string being generated.
Note that the chunks may not be valid JSON on their own - you’ll need to concatenate them to get the complete JSON response.
The gateway doesn’t return parsed or validated JSON objects when streaming.

Copy

```
{
  "inference_id": "00000000-0000-0000-0000-000000000000",
  "episode_id": "11111111-1111-1111-1111-111111111111",
  "variant_name": "prompt_v1",
  "raw": "{\"email\":", // a JSON content delta
  // token usage information is only available in the final chunk with content (before the [DONE] message)
  "usage": {
    "input_tokens": 100,
    "output_tokens": 100
  }
}
```

## [​](https://www.tensorzero.com/docs/gateway/guides/streaming-inference\#technical-notes)  Technical Notes

- Token usage information is only available in the final chunk with content (before the `[DONE]` message)
- Streaming may not be available with certain [inference-time optimizations](https://www.tensorzero.com/docs/gateway/guides/inference-time-optimizations)

[Retries & Fallbacks](https://www.tensorzero.com/docs/gateway/guides/retries-fallbacks) [Tool Use (Function Calling)](https://www.tensorzero.com/docs/gateway/guides/tool-use)

Ctrl+I

Assistant

Responses are generated using AI and may contain mistakes.

## TensorZero Tool Usage
[Skip to main content](https://www.tensorzero.com/docs/gateway/guides/tool-use#content-area)

[TensorZero Docs home page![light logo](https://mintcdn.com/tensorzero/afFfERJmQ8YiZpCj/logo/light.svg?fit=max&auto=format&n=afFfERJmQ8YiZpCj&q=85&s=9396731a8a92cee7bd293dee05d71dac)![dark logo](https://mintcdn.com/tensorzero/afFfERJmQ8YiZpCj/logo/dark.svg?fit=max&auto=format&n=afFfERJmQ8YiZpCj&q=85&s=8d61c59531dbcdcace7fcdfc524bd4d2)](https://www.tensorzero.com/)

Search...

Ctrl K

Search...

Navigation

Gateway

Tool Use (Function Calling)

[Guides](https://www.tensorzero.com/docs) [Integrations](https://www.tensorzero.com/docs/integrations/model-providers)

On this page

- [Basic Usage](https://www.tensorzero.com/docs/gateway/guides/tool-use#basic-usage)
- [Defining a tool in your configuration file](https://www.tensorzero.com/docs/gateway/guides/tool-use#defining-a-tool-in-your-configuration-file)
- [Making inference requests with tools](https://www.tensorzero.com/docs/gateway/guides/tool-use#making-inference-requests-with-tools)
- [Advanced Usage](https://www.tensorzero.com/docs/gateway/guides/tool-use#advanced-usage)
- [Restricting allowed tools at inference time](https://www.tensorzero.com/docs/gateway/guides/tool-use#restricting-allowed-tools-at-inference-time)
- [Defining tools dynamically at inference time](https://www.tensorzero.com/docs/gateway/guides/tool-use#defining-tools-dynamically-at-inference-time)
- [Customizing the tool calling strategy](https://www.tensorzero.com/docs/gateway/guides/tool-use#customizing-the-tool-calling-strategy)
- [Calling multiple tools in parallel](https://www.tensorzero.com/docs/gateway/guides/tool-use#calling-multiple-tools-in-parallel)
- [Integrating with Model Context Protocol (MCP) servers](https://www.tensorzero.com/docs/gateway/guides/tool-use#integrating-with-model-context-protocol-mcp-servers)
- [Learn More](https://www.tensorzero.com/docs/gateway/guides/tool-use#learn-more)

TensorZero has first-class support for tool use, a feature that allows LLMs to interact with external tools (e.g. APIs, databases, web browsers).Tool use is available for most model providers supported by TensorZero.
See [Integrations](https://www.tensorzero.com/docs/integrations/model-providers) for a list of supported model providers.You can define a tool in your configuration file and attach it to a TensorZero function that should be allowed to call it.
Alternatively, you can define a tool dynamically at inference time.

The term “tool use” is also commonly referred to as “function calling” in the industry.
In TensorZero, the term “function” refers to TensorZero functions, so we’ll stick to the “tool” terminology for external tools that the models can interact with and “function” for TensorZero functions.

You can also find a complete runnable example on [GitHub](https://github.com/tensorzero/tensorzero/tree/main/examples/guides/tool-use).

## [​](https://www.tensorzero.com/docs/gateway/guides/tool-use\#basic-usage)  Basic Usage

### [​](https://www.tensorzero.com/docs/gateway/guides/tool-use\#defining-a-tool-in-your-configuration-file)  Defining a tool in your configuration file

You can define a tool in your configuration file and attach it to the TensorZero functions that should be allowed to call it.
Only functions that are of type `chat` can call tools.A tool definition has the following properties:

- `name`: The name of the tool.
- `description`: A description of the tool. The description helps models understand the tool’s purpose and usage.
- `parameters`: The path to a file containing a JSON Schema for the tool’s parameters.

Optionally, you can provide a `strict` property to enforce type checking for the tool’s parameters.
This setting is only supported by some model providers, and will be ignored otherwise.

tensorzero.toml

Copy

```
[tools.get_temperature]
description = "Get the current temperature for a given location."
parameters = "tools/get_temperature.json"
strict = true # optional, defaults to false

[functions.weather_chatbot]
type = "chat"
tools = ["get_temperature"]
# ...
```

Example: JSON Schema for the \`get\_temperature\` tool

If we wanted the `get_temperature` tool to take a mandatory `location` parameter and an optional `units` parameter, we could use the following JSON Schema:

tools/get\_temperature.json

Copy

```
{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "type": "object",
  "description": "Get the current temperature for a given location.",
  "properties": {
    "location": {
      "type": "string",
      "description": "The location to get the temperature for (e.g. \"New York\")"
    },
    "units": {
      "type": "string",
      "description": "The units to get the temperature in (must be \"fahrenheit\" or \"celsius\"). Defaults to \"fahrenheit\".",
      "enum": ["fahrenheit", "celsius"]
    }
  },
  "required": ["location"],
  "additionalProperties": false
}
```

See “Advanced Usage” below for information on how to define a tool dynamically at inference time.

### [​](https://www.tensorzero.com/docs/gateway/guides/tool-use\#making-inference-requests-with-tools)  Making inference requests with tools

Once you’ve defined a tool and attached it to a TensorZero function, you don’t need to change anything in your inference request to enable tool useBy default, the function will determine whether to use a tool and the arguments to pass to the tool.
If the function decides to use tools, it will return one or more `tool_call` content blocks in the response.For multi-turn conversations supporting tool use, you can provide tool results in subsequent inference requests with a `tool_result` content block.

Example: Multi-turn conversation with tool use

You can also find a complete runnable example on [GitHub](https://github.com/tensorzero/tensorzero/tree/main/examples/guides/tool-use).

- Python

- Python (OpenAI)

- Node (OpenAI)

- HTTP


Copy

```
from tensorzero import TensorZeroGateway, ToolCall  # or AsyncTensorZeroGateway

with TensorZeroGateway.build_http(
    gateway_url="http://localhost:3000",
) as t0:
    messages = [{"role": "user", "content": "What is the weather in Tokyo (°F)?"}]

    response = t0.inference(
        function_name="weather_chatbot",
        input={"messages": messages},
    )

    print(response)

    # The model can return multiple content blocks, including tool calls
    # In a real application, you'd be stricter about validating the response
    tool_calls = [\
        content_block\
        for content_block in response.content\
        if isinstance(content_block, ToolCall)\
    ]
    assert len(tool_calls) == 1, "Expected the model to return exactly one tool call"

    # Add the tool call to the message history
    messages.append(
        {
            "role": "assistant",
            "content": response.content,
        }
    )

    # Pretend we've called the tool and got a response
    messages.append(
        {
            "role": "user",
            "content": [\
                {\
                    "type": "tool_result",\
                    "id": tool_calls[0].id,\
                    "name": tool_calls[0].name,\
                    "result": "70",  # imagine it's 70°F in Tokyo\
                }\
            ],
        }
    )

    response = t0.inference(
        function_name="weather_chatbot",
        input={"messages": messages},
    )

    print(response)
```

See “Advanced Usage” below for information on how to customize the tool calling behavior (e.g. making tool calls mandatory).

## [​](https://www.tensorzero.com/docs/gateway/guides/tool-use\#advanced-usage)  Advanced Usage

### [​](https://www.tensorzero.com/docs/gateway/guides/tool-use\#restricting-allowed-tools-at-inference-time)  Restricting allowed tools at inference time

You can restrict the set of tools that can be called at inference time by using the `allowed_tools` parameter.For example, suppose your TensorZero function has access to several tools, but you only want to allow the `get_temperature` tool to be called during a particular inference.
You can achieve this by setting `allowed_tools=["get_temperature"]` in your inference request.

### [​](https://www.tensorzero.com/docs/gateway/guides/tool-use\#defining-tools-dynamically-at-inference-time)  Defining tools dynamically at inference time

You can define tools dynamically at inference time by using the `additional_tools` property.
(In the OpenAI-compatible API, you can use the `tools` property instead.)You should only use dynamic tools if your use case requires it.
Otherwise, it’s recommended to define tools in the configuration file.You can define a tool dynamically with the `additional_tools` property.
This field accepts a list of objects with the same structure as the tools defined in the configuration file, except that the `parameters` field should contain the JSON Schema itself (rather than a path to a file with the schema).

### [​](https://www.tensorzero.com/docs/gateway/guides/tool-use\#customizing-the-tool-calling-strategy)  Customizing the tool calling strategy

You can control how and when tools are called by using the `tool_choice` parameter.
The supported tool choice strategies are:

- `none`: The function should not use any tools.
- `auto`: The model decides whether or not to use a tool. If it decides to use a tool, it also decides which tools to use.
- `required`: The model should use a tool. If multiple tools are available, the model decides which tool to use.
- `{ specific = "tool_name" }`: The model should use a specific tool. The tool must be defined in the `tools` section of the configuration file or provided in `additional_tools`.

The `tool_choice` parameter can be set either in your configuration file or directly in your inference request.

### [​](https://www.tensorzero.com/docs/gateway/guides/tool-use\#calling-multiple-tools-in-parallel)  Calling multiple tools in parallel

You can enable parallel tool calling by setting the `parallel_tool_calls` parameter to `true`.If enabled, the models will be able to request multiple tool calls in a single inference request (conversation turn).You can specify `parallel_tool_calls` in the configuration file or in the inference request.

### [​](https://www.tensorzero.com/docs/gateway/guides/tool-use\#integrating-with-model-context-protocol-mcp-servers)  Integrating with Model Context Protocol (MCP) servers

You can use TensorZero with tools offered by Model Context Protocol (MCP) servers with the functionality described above.See our [MCP (Model Context Protocol) Example on GitHub](https://github.com/tensorzero/tensorzero/tree/main/examples/mcp-model-context-protocol) to learn how to integrate TensorZero with an MCP server.

## [​](https://www.tensorzero.com/docs/gateway/guides/tool-use\#learn-more)  Learn More

[**API Reference: Inference**](https://www.tensorzero.com/docs/gateway/api-reference/inference) [**API Reference: Inference (OpenAI-Compatible)**](https://www.tensorzero.com/docs/gateway/api-reference/inference-openai-compatible) [**Configuration Reference**](https://www.tensorzero.com/docs/gateway/configuration-reference)

[Streaming Inference](https://www.tensorzero.com/docs/gateway/guides/streaming-inference) [Benchmarks](https://www.tensorzero.com/docs/gateway/benchmarks)

Ctrl+I

Assistant

Responses are generated using AI and may contain mistakes.

## TensorZero Gateway
[Skip to main content](https://www.tensorzero.com/docs/gateway#content-area)

[TensorZero Docs home page![light logo](https://mintcdn.com/tensorzero/afFfERJmQ8YiZpCj/logo/light.svg?fit=max&auto=format&n=afFfERJmQ8YiZpCj&q=85&s=9396731a8a92cee7bd293dee05d71dac)![dark logo](https://mintcdn.com/tensorzero/afFfERJmQ8YiZpCj/logo/dark.svg?fit=max&auto=format&n=afFfERJmQ8YiZpCj&q=85&s=8d61c59531dbcdcace7fcdfc524bd4d2)](https://www.tensorzero.com/)

Search...

Ctrl K

Search...

Navigation

Gateway

Overview

[Guides](https://www.tensorzero.com/docs) [Integrations](https://www.tensorzero.com/docs/integrations/model-providers)

On this page

- [Next Steps](https://www.tensorzero.com/docs/gateway#next-steps)

The TensorZero Gateway is a high-performance model gateway that provides a unified interface for all your LLM applications.

- **One API for All LLMs.**
The gateway provides a unified interface for all major LLM providers, allowing for seamless cross-platform integration and fallbacks.
TensorZero natively supports
[Anthropic](https://www.tensorzero.com/docs/integrations/model-providers/anthropic),
[AWS Bedrock](https://www.tensorzero.com/docs/integrations/model-providers/aws-bedrock),
[AWS SageMaker](https://www.tensorzero.com/docs/integrations/model-providers/aws-sagemaker),
[Azure OpenAI Service](https://www.tensorzero.com/docs/integrations/model-providers/azure),
[Fireworks](https://www.tensorzero.com/docs/integrations/model-providers/fireworks),
[GCP Vertex AI Anthropic](https://www.tensorzero.com/docs/integrations/model-providers/gcp-vertex-ai-anthropic),
[GCP Vertex AI Gemini](https://www.tensorzero.com/docs/integrations/model-providers/gcp-vertex-ai-gemini),
[Google AI Studio (Gemini API)](https://www.tensorzero.com/docs/integrations/model-providers/google-ai-studio-gemini),
[Groq](https://www.tensorzero.com/docs/integrations/model-providers/groq),
[Hyperbolic](https://www.tensorzero.com/docs/integrations/model-providers/hyperbolic),
[Mistral](https://www.tensorzero.com/docs/integrations/model-providers/mistral),
[OpenAI](https://www.tensorzero.com/docs/integrations/model-providers/openai),
[OpenRouter](https://www.tensorzero.com/docs/integrations/model-providers/openrouter),
[Together](https://www.tensorzero.com/docs/integrations/model-providers/together),
[vLLM](https://www.tensorzero.com/docs/integrations/model-providers/vllm), and
[xAI](https://www.tensorzero.com/docs/integrations/model-providers/xai).
Need something else?
Your provider is most likely supported because TensorZero integrates with [any OpenAI-compatible API (e.g. Ollama)](https://www.tensorzero.com/docs/integrations/model-providers/openai-compatible).
Still not supported?
Open an issue on [GitHub](https://github.com/tensorzero/tensorzero/issues) and we’ll integrate it!






Learn more in our [How to call any LLM](https://www.tensorzero.com/docs/gateway/call-any-llm) guide.

- **Blazing Fast.**
The gateway (written in Rust 🦀) achieves <1ms P99 latency overhead under extreme load.
In [benchmarks](https://www.tensorzero.com/docs/gateway/benchmarks), LiteLLM @ 100 QPS adds 25-100x+ more latency than our gateway @ 10,000 QPS.
- **Structured Inferences.**
The gateway enforces schemas for inputs and outputs, ensuring robustness for your application.
Structured inference data is later used for powerful optimization recipes (e.g. swapping historical prompts before fine-tuning).
Learn more about [creating prompt templates](https://www.tensorzero.com/docs/gateway/create-a-prompt-template).
- **Multi-Step LLM Workflows.**
The gateway provides first-class support for complex multi-step LLM workflows by associating multiple inferences with an episode.
Feedback can be assigned at the inference or episode level, allowing for end-to-end optimization of compound LLM systems.
Learn more about [episodes](https://www.tensorzero.com/docs/gateway/guides/episodes).
- **Built-in Observability.**
The gateway collects structured inference traces along with associated downstream metrics and natural-language feedback.
Everything is stored in a ClickHouse database for real-time, scalable, and developer-friendly analytics.
[TensorZero Recipes](https://www.tensorzero.com/docs/recipes) leverage this dataset to optimize your LLMs.
- **Built-in Experimentation.**
The gateway automatically routes traffic between variants to enable A/B tests.
It ensures consistent variants within an episode in multi-step workflows.
Learn more about [adaptive A/B tests](https://www.tensorzero.com/docs/experimentation/run-adaptive-ab-tests).
- **Built-in Fallbacks.**
The gateway automatically fallbacks failed inferences to different inference providers, or even completely different variants.
Ensure misconfiguration, provider downtime, and other edge cases don’t affect your availability.
- **Access Controls.**
The gateway supports TensorZero API key authentication, allowing you to control access to your TensorZero deployment.
Create and manage custom API keys for different clients or services.
Learn more about [setting up auth for TensorZero](https://www.tensorzero.com/docs/operations/set-up-auth-for-tensorzero).
- **GitOps Orchestration.**
Orchestrate prompts, models, parameters, tools, experiments, and more with GitOps-friendly configuration.
Manage a few LLMs manually with human-friendly readable configuration files, or thousands of prompts and LLMs entirely programmatically.

## [​](https://www.tensorzero.com/docs/gateway\#next-steps)  Next Steps

[**Quickstart** \\
\\
Make your first TensorZero API call with built-in observability and\\
fine-tuning in under 5 minutes.](https://www.tensorzero.com/docs/quickstart) [**Deployment** \\
\\
Quickly deploy locally, or set up high-availability services for production\\
environments.](https://www.tensorzero.com/docs/deployment/tensorzero-gateway) [**Integrations** \\
\\
The TensorZero Gateway integrates with the major LLM providers.](https://www.tensorzero.com/docs/integrations/model-providers) [**Benchmarks** \\
\\
The TensorZero Gateway achieves sub-millisecond latency overhead under\\
extreme load.](https://www.tensorzero.com/docs/gateway/benchmarks) [**API Reference** \\
\\
The TensorZero Gateway provides an unified interface for making inference\\
and feedback API calls.](https://www.tensorzero.com/docs/gateway/api-reference/inference) [**Configuration Reference** \\
\\
Easily manage your LLM applications with GitOps orchestration — even complex\\
multi-step systems.](https://www.tensorzero.com/docs/gateway/configuration-reference)

[Portkey](https://www.tensorzero.com/docs/comparison/portkey) [Call any LLM](https://www.tensorzero.com/docs/gateway/call-any-llm)

Ctrl+I

Assistant

Responses are generated using AI and may contain mistakes.

## Anthropic Model Integration
[Skip to main content](https://www.tensorzero.com/docs/integrations/model-providers/anthropic#content-area)

[TensorZero Docs home page![light logo](https://mintcdn.com/tensorzero/afFfERJmQ8YiZpCj/logo/light.svg?fit=max&auto=format&n=afFfERJmQ8YiZpCj&q=85&s=9396731a8a92cee7bd293dee05d71dac)![dark logo](https://mintcdn.com/tensorzero/afFfERJmQ8YiZpCj/logo/dark.svg?fit=max&auto=format&n=afFfERJmQ8YiZpCj&q=85&s=8d61c59531dbcdcace7fcdfc524bd4d2)](https://www.tensorzero.com/)

Search...

Ctrl K

Search...

Navigation

Model Providers

Getting Started with Anthropic

[Guides](https://www.tensorzero.com/docs) [Integrations](https://www.tensorzero.com/docs/integrations/model-providers)

On this page

- [Simple Setup](https://www.tensorzero.com/docs/integrations/model-providers/anthropic#simple-setup)
- [Advanced Setup](https://www.tensorzero.com/docs/integrations/model-providers/anthropic#advanced-setup)
- [Configuration](https://www.tensorzero.com/docs/integrations/model-providers/anthropic#configuration)
- [Credentials](https://www.tensorzero.com/docs/integrations/model-providers/anthropic#credentials)
- [Deployment (Docker Compose)](https://www.tensorzero.com/docs/integrations/model-providers/anthropic#deployment-docker-compose)
- [Inference](https://www.tensorzero.com/docs/integrations/model-providers/anthropic#inference)
- [Other Features](https://www.tensorzero.com/docs/integrations/model-providers/anthropic#other-features)

This guide shows how to set up a minimal deployment to use the TensorZero Gateway with the Anthropic API.

## [​](https://www.tensorzero.com/docs/integrations/model-providers/anthropic\#simple-setup)  Simple Setup

You can use the short-hand `anthropic::model_name` to use an Anthropic model with TensorZero, unless you need advanced features like fallbacks or custom credentials.You can use Anthropic models in your TensorZero variants by setting the `model` field to `anthropic::model_name`.
For example:

Copy

```
[functions.my_function_name.variants.my_variant_name]
type = "chat_completion"
model = "anthropic::claude-3-5-haiku-20241022"
```

Additionally, you can set `model_name` in the inference request to use a specific Anthropic model, without having to configure a function and variant in TensorZero.

Copy

```
curl -X POST http://localhost:3000/inference \
  -H "Content-Type: application/json" \
  -d '{
    "model_name": "anthropic::claude-3-5-haiku-20241022",
    "input": {
      "messages": [\
        {\
          "role": "user",\
          "content": "What is the capital of Japan?"\
        }\
      ]
    }
  }'
```

## [​](https://www.tensorzero.com/docs/integrations/model-providers/anthropic\#advanced-setup)  Advanced Setup

In more complex scenarios (e.g. fallbacks, custom credentials), you can configure your own model and Anthropic provider in TensorZero.For this minimal setup, you’ll need just two files in your project directory:

Copy

```
- config/
  - tensorzero.toml
- docker-compose.yml
```

You can also find the complete code for this example on [GitHub](https://github.com/tensorzero/tensorzero/tree/main/examples/guides/providers/anthropic).

For production deployments, see our [Deployment Guide](https://www.tensorzero.com/docs/deployment/tensorzero-gateway).

### [​](https://www.tensorzero.com/docs/integrations/model-providers/anthropic\#configuration)  Configuration

Create a minimal configuration file that defines a model and a simple chat function:

config/tensorzero.toml

Copy

```
[models.claude_3_5_haiku_20241022]
routing = ["anthropic"]

[models.claude_3_5_haiku_20241022.providers.anthropic]
type = "anthropic"
model_name = "claude-3-5-haiku-20241022"

[functions.my_function_name]
type = "chat"

[functions.my_function_name.variants.my_variant_name]
type = "chat_completion"
model = "claude_3_5_haiku_20241022"
```

See the [list of models available on Anthropic](https://docs.anthropic.com/en/docs/about-claude/models).

### [​](https://www.tensorzero.com/docs/integrations/model-providers/anthropic\#credentials)  Credentials

You must set the `ANTHROPIC_API_KEY` environment variable before running the gateway.You can customize the credential location by setting the `api_key_location` to `env::YOUR_ENVIRONMENT_VARIABLE` or `dynamic::ARGUMENT_NAME`.
See the [Credential Management](https://www.tensorzero.com/docs/operations/manage-credentials) guide and [Configuration Reference](https://www.tensorzero.com/docs/gateway/configuration-reference) for more information.

### [​](https://www.tensorzero.com/docs/integrations/model-providers/anthropic\#deployment-docker-compose)  Deployment (Docker Compose)

Create a minimal Docker Compose configuration:

docker-compose.yml

Copy

```
# This is a simplified example for learning purposes. Do not use this in production.
# For production-ready deployments, see: https://www.tensorzero.com/docs/deployment/tensorzero-gateway

services:
  gateway:
    image: tensorzero/gateway
    volumes:
      - ./config:/app/config:ro
    command: --config-file /app/config/tensorzero.toml
    environment:
      - ANTHROPIC_API_KEY=${ANTHROPIC_API_KEY:?Environment variable ANTHROPIC_API_KEY must be set.}
    ports:
      - "3000:3000"
    extra_hosts:
      - "host.docker.internal:host-gateway"
```

You can start the gateway with `docker compose up`.

## [​](https://www.tensorzero.com/docs/integrations/model-providers/anthropic\#inference)  Inference

Make an inference request to the gateway:

Copy

```
curl -X POST http://localhost:3000/inference \
  -H "Content-Type: application/json" \
  -d '{
    "function_name": "my_function_name",
    "input": {
      "messages": [\
        {\
          "role": "user",\
          "content": "What is the capital of Japan?"\
        }\
      ]
    }
  }'
```

## [​](https://www.tensorzero.com/docs/integrations/model-providers/anthropic\#other-features)  Other Features

See [Extending TensorZero](https://www.tensorzero.com/docs/operations/extend-tensorzero) for information about Anthropic Computer Use and other beta features.

[Overview](https://www.tensorzero.com/docs/integrations/model-providers) [AWS Bedrock](https://www.tensorzero.com/docs/integrations/model-providers/aws-bedrock)

Ctrl+I

Assistant

Responses are generated using AI and may contain mistakes.

## AWS Bedrock Setup
[Skip to main content](https://www.tensorzero.com/docs/integrations/model-providers/aws-bedrock#content-area)

[TensorZero Docs home page![light logo](https://mintcdn.com/tensorzero/afFfERJmQ8YiZpCj/logo/light.svg?fit=max&auto=format&n=afFfERJmQ8YiZpCj&q=85&s=9396731a8a92cee7bd293dee05d71dac)![dark logo](https://mintcdn.com/tensorzero/afFfERJmQ8YiZpCj/logo/dark.svg?fit=max&auto=format&n=afFfERJmQ8YiZpCj&q=85&s=8d61c59531dbcdcace7fcdfc524bd4d2)](https://www.tensorzero.com/)

Search...

⌘K

Search...

Navigation

Model Providers

Getting Started with AWS Bedrock

[Guides](https://www.tensorzero.com/docs) [Integrations](https://www.tensorzero.com/docs/integrations/model-providers)

On this page

- [Setup](https://www.tensorzero.com/docs/integrations/model-providers/aws-bedrock#setup)
- [Configuration](https://www.tensorzero.com/docs/integrations/model-providers/aws-bedrock#configuration)
- [Credentials](https://www.tensorzero.com/docs/integrations/model-providers/aws-bedrock#credentials)
- [Deployment (Docker Compose)](https://www.tensorzero.com/docs/integrations/model-providers/aws-bedrock#deployment-docker-compose)
- [Inference](https://www.tensorzero.com/docs/integrations/model-providers/aws-bedrock#inference)

This guide shows how to set up a minimal deployment to use the TensorZero Gateway with the AWS Bedrock API.

## [​](https://www.tensorzero.com/docs/integrations/model-providers/aws-bedrock\#setup)  Setup

For this minimal setup, you’ll need just two files in your project directory:

Copy

```
- config/
  - tensorzero.toml
- docker-compose.yml
```

You can also find the complete code for this example on [GitHub](https://github.com/tensorzero/tensorzero/tree/main/examples/guides/providers/aws-bedrock).

For production deployments, see our [Deployment Guide](https://www.tensorzero.com/docs/deployment/tensorzero-gateway).

### [​](https://www.tensorzero.com/docs/integrations/model-providers/aws-bedrock\#configuration)  Configuration

Create a minimal configuration file that defines a model and a simple chat function:

config/tensorzero.toml

Copy

```
[models.claude_3_haiku_20240307]
routing = ["aws_bedrock"]

[models.claude_3_haiku_20240307.providers.aws_bedrock]
type = "aws_bedrock"
model_id = "anthropic.claude-3-haiku-20240307-v1:0"

[functions.my_function_name]
type = "chat"

[functions.my_function_name.variants.my_variant_name]
type = "chat_completion"
model = "claude_3_haiku_20240307"
```

See the [list of available models on AWS Bedrock](https://docs.aws.amazon.com/bedrock/latest/userguide/models-supported.html).

Many AWS Bedrock models are only available through cross-region inference profiles.
For those models, the `model_id` requires special prefix (e.g. the `us.` prefix in `us.anthropic.claude-3-7-sonnet-20250219-v1:0`).
See the [AWS documentation on inference profiles](https://docs.aws.amazon.com/bedrock/latest/userguide/inference-profiles-support.html).

See the [Configuration Reference](https://www.tensorzero.com/docs/gateway/configuration-reference) for optional fields (e.g. overriding the `region`).

### [​](https://www.tensorzero.com/docs/integrations/model-providers/aws-bedrock\#credentials)  Credentials

You must make sure that the gateway has the necessary permissions to access AWS Bedrock.
The TensorZero Gateway will use the AWS SDK to retrieve the relevant credentials.The simplest way is to set the following environment variables before running the gateway:

Copy

```
AWS_ACCESS_KEY_ID=...
AWS_REGION=us-east-1
AWS_SECRET_ACCESS_KEY=...
```

Alternatively, you can use other authentication methods supported by the AWS SDK.

### [​](https://www.tensorzero.com/docs/integrations/model-providers/aws-bedrock\#deployment-docker-compose)  Deployment (Docker Compose)

Create a minimal Docker Compose configuration:

docker-compose.yml

Copy

```
# This is a simplified example for learning purposes. Do not use this in production.
# For production-ready deployments, see: https://www.tensorzero.com/docs/deployment/tensorzero-gateway

services:
  gateway:
    image: tensorzero/gateway
    volumes:
      - ./config:/app/config:ro
    command: --config-file /app/config/tensorzero.toml
    environment:
      - AWS_ACCESS_KEY_ID=${AWS_ACCESS_KEY_ID:?Environment variable AWS_ACCESS_KEY_ID must be set.}
      - AWS_REGION=${AWS_REGION:?Environment variable AWS_REGION must be set.}
      - AWS_SECRET_ACCESS_KEY=${AWS_SECRET_ACCESS_KEY:?Environment variable AWS_SECRET_ACCESS_KEY must be set.}
    ports:
      - "3000:3000"
    extra_hosts:
      - "host.docker.internal:host-gateway"
```

You can start the gateway with `docker compose up`.

## [​](https://www.tensorzero.com/docs/integrations/model-providers/aws-bedrock\#inference)  Inference

Make an inference request to the gateway:

Copy

```
curl -X POST http://localhost:3000/inference \
  -H "Content-Type: application/json" \
  -d '{
    "function_name": "my_function_name",
    "input": {
      "messages": [\
        {\
          "role": "user",\
          "content": "What is the capital of Japan?"\
        }\
      ]
    }
  }'
```

[Anthropic](https://www.tensorzero.com/docs/integrations/model-providers/anthropic) [AWS SageMaker](https://www.tensorzero.com/docs/integrations/model-providers/aws-sagemaker)

⌘I

## AWS SageMaker Integration
[Skip to main content](https://www.tensorzero.com/docs/integrations/model-providers/aws-sagemaker#content-area)

[TensorZero Docs home page![light logo](https://mintcdn.com/tensorzero/afFfERJmQ8YiZpCj/logo/light.svg?fit=max&auto=format&n=afFfERJmQ8YiZpCj&q=85&s=9396731a8a92cee7bd293dee05d71dac)![dark logo](https://mintcdn.com/tensorzero/afFfERJmQ8YiZpCj/logo/dark.svg?fit=max&auto=format&n=afFfERJmQ8YiZpCj&q=85&s=8d61c59531dbcdcace7fcdfc524bd4d2)](https://www.tensorzero.com/)

Search...

Ctrl K

Search...

Navigation

Model Providers

Getting Started with AWS SageMaker

[Guides](https://www.tensorzero.com/docs) [Integrations](https://www.tensorzero.com/docs/integrations/model-providers)

On this page

- [Setup](https://www.tensorzero.com/docs/integrations/model-providers/aws-sagemaker#setup)
- [Configuration](https://www.tensorzero.com/docs/integrations/model-providers/aws-sagemaker#configuration)
- [Credentials](https://www.tensorzero.com/docs/integrations/model-providers/aws-sagemaker#credentials)
- [Deployment (Docker Compose)](https://www.tensorzero.com/docs/integrations/model-providers/aws-sagemaker#deployment-docker-compose)
- [Inference](https://www.tensorzero.com/docs/integrations/model-providers/aws-sagemaker#inference)

This guide shows how to set up a minimal deployment to use the TensorZero Gateway with the AWS SageMaker API.The AWS SageMaker model provider is a wrapper around other TensorZero model providers that handles AWS SageMaker-specific logic (e.g. auth).
For example, you can use it to infer self-hosted model providers like Ollama and TGI deployed on AWS SageMaker.

## [​](https://www.tensorzero.com/docs/integrations/model-providers/aws-sagemaker\#setup)  Setup

For this minimal setup, you’ll need just two files in your project directory:

Copy

```
- config/
  - tensorzero.toml
- docker-compose.yml
```

You can also find the complete code for this example on [GitHub](https://github.com/tensorzero/tensorzero/tree/main/examples/guides/providers/aws-sagemaker).

For production deployments, see our [Deployment Guide](https://www.tensorzero.com/docs/deployment/tensorzero-gateway).You’ll also need to deploy a SageMaker endpoint for your LLM model.
For this example, we’re using a container running Ollama.

### [​](https://www.tensorzero.com/docs/integrations/model-providers/aws-sagemaker\#configuration)  Configuration

Create a minimal configuration file that defines a model and a simple chat function:

config/tensorzero.toml

Copy

```
[models.gemma_3]
routing = ["aws_sagemaker"]

[models.gemma_3.providers.aws_sagemaker]
type = "aws_sagemaker"
model_name = "gemma3:1b"
endpoint_name = "my-sagemaker-endpoint"
region = "us-east-1"
# ... or use `allow_auto_detect_region = true` to infer region with the AWS SDK
hosted_provider = "openai"  # Ollama is OpenAI-compatible

[functions.my_function_name]
type = "chat"

[functions.my_function_name.variants.my_variant_name]
type = "chat_completion"
model = "gemma_3"
```

The `hosted_provider` field specifies the model provider that you deployed on AWS SageMaker.
For example, Ollama is OpenAI-compatible, so we use `openai` as the hosted provider.
Alternatively, you can use `hosted_provider = "tgi"` if you had deployed TGI instead.You can specify the endpoint’s `region` explicitly, or use `allow_auto_detect_region = true` to infer region with the AWS SDK.See the [Configuration Reference](https://www.tensorzero.com/docs/gateway/configuration-reference) for optional fields.
The relevant fields will depend on the `hosted_provider`.

### [​](https://www.tensorzero.com/docs/integrations/model-providers/aws-sagemaker\#credentials)  Credentials

You must make sure that the gateway has the necessary permissions to access AWS SageMaker.
The TensorZero Gateway will use the AWS SDK to retrieve the relevant credentials.The simplest way is to set the following environment variables before running the gateway:

Copy

```
AWS_ACCESS_KEY_ID=...
AWS_REGION=us-east-1
AWS_SECRET_ACCESS_KEY=...
```

Alternatively, you can use other authentication methods supported by the AWS SDK.

### [​](https://www.tensorzero.com/docs/integrations/model-providers/aws-sagemaker\#deployment-docker-compose)  Deployment (Docker Compose)

Create a minimal Docker Compose configuration:

docker-compose.yml

Copy

```
# This is a simplified example for learning purposes. Do not use this in production.
# For production-ready deployments, see: https://www.tensorzero.com/docs/deployment/tensorzero-gateway

services:
  gateway:
    image: tensorzero/gateway
    volumes:
      - ./config:/app/config:ro
    command: --config-file /app/config/tensorzero.toml
    environment:
      - AWS_ACCESS_KEY_ID=${AWS_ACCESS_KEY_ID:?Environment variable AWS_ACCESS_KEY_ID must be set.}
      - AWS_REGION=${AWS_REGION:?Environment variable AWS_REGION must be set.}
      - AWS_SECRET_ACCESS_KEY=${AWS_SECRET_ACCESS_KEY:?Environment variable AWS_SECRET_ACCESS_KEY must be set.}
    ports:
      - "3000:3000"
    extra_hosts:
      - "host.docker.internal:host-gateway"
```

You can start the gateway with `docker compose up`.

## [​](https://www.tensorzero.com/docs/integrations/model-providers/aws-sagemaker\#inference)  Inference

Make an inference request to the gateway:

Copy

```
curl -X POST http://localhost:3000/inference \
  -H "Content-Type: application/json" \
  -d '{
    "function_name": "my_function_name",
    "input": {
      "messages": [\
        {\
          "role": "user",\
          "content": "What is the capital of Japan?"\
        }\
      ]
    }
  }'
```

[AWS Bedrock](https://www.tensorzero.com/docs/integrations/model-providers/aws-bedrock) [Azure](https://www.tensorzero.com/docs/integrations/model-providers/azure)

Ctrl+I

Assistant

Responses are generated using AI and may contain mistakes.

## Azure Model Integration
[Skip to main content](https://www.tensorzero.com/docs/integrations/model-providers/azure#content-area)

[TensorZero Docs home page![light logo](https://mintcdn.com/tensorzero/afFfERJmQ8YiZpCj/logo/light.svg?fit=max&auto=format&n=afFfERJmQ8YiZpCj&q=85&s=9396731a8a92cee7bd293dee05d71dac)![dark logo](https://mintcdn.com/tensorzero/afFfERJmQ8YiZpCj/logo/dark.svg?fit=max&auto=format&n=afFfERJmQ8YiZpCj&q=85&s=8d61c59531dbcdcace7fcdfc524bd4d2)](https://www.tensorzero.com/)

Search...

Ctrl K

Search...

Navigation

Model Providers

Getting Started with Azure OpenAI Service & Azure AI Foundry

[Guides](https://www.tensorzero.com/docs) [Integrations](https://www.tensorzero.com/docs/integrations/model-providers)

On this page

- [Azure OpenAI Service](https://www.tensorzero.com/docs/integrations/model-providers/azure#azure-openai-service)
- [Setup](https://www.tensorzero.com/docs/integrations/model-providers/azure#setup)
- [Configuration](https://www.tensorzero.com/docs/integrations/model-providers/azure#configuration)
- [Credentials](https://www.tensorzero.com/docs/integrations/model-providers/azure#credentials)
- [Deployment (Docker Compose)](https://www.tensorzero.com/docs/integrations/model-providers/azure#deployment-docker-compose)
- [Inference](https://www.tensorzero.com/docs/integrations/model-providers/azure#inference)
- [Other Features](https://www.tensorzero.com/docs/integrations/model-providers/azure#other-features)
- [Generate embeddings](https://www.tensorzero.com/docs/integrations/model-providers/azure#generate-embeddings)
- [Azure AI Foundry](https://www.tensorzero.com/docs/integrations/model-providers/azure#azure-ai-foundry)

TensorZero’s `azure` provider supports both **Azure OpenAI Service** and **Azure AI Foundry**. Both use the same OpenAI-compatible API, so configuration is nearly identical—just use different endpoint URLs.This guide shows how to set up a minimal deployment to use the TensorZero Gateway with Azure OpenAI Service and Azure AI Foundry.

## [​](https://www.tensorzero.com/docs/integrations/model-providers/azure\#azure-openai-service)  Azure OpenAI Service

### [​](https://www.tensorzero.com/docs/integrations/model-providers/azure\#setup)  Setup

For this minimal setup, you’ll need just two files in your project directory:

Copy

```
- config/
  - tensorzero.toml
- docker-compose.yml
```

You can also find the complete code for this example on [GitHub](https://github.com/tensorzero/tensorzero/tree/main/examples/guides/providers/azure).

For production deployments, see our [Deployment Guide](https://www.tensorzero.com/docs/deployment/tensorzero-gateway).

### [​](https://www.tensorzero.com/docs/integrations/model-providers/azure\#configuration)  Configuration

Create a minimal configuration file that defines a model and a simple chat function:

config/tensorzero.toml

Copy

```
[models.gpt_4o_mini_2024_07_18]
routing = ["azure"]

[models.gpt_4o_mini_2024_07_18.providers.azure]
type = "azure"
deployment_id = "gpt4o-mini-20240718"
endpoint = "https://your-azure-openai-endpoint.openai.azure.com"

[functions.my_function_name]
type = "chat"

[functions.my_function_name.variants.my_variant_name]
type = "chat_completion"
model = "gpt_4o_mini_2024_07_18"
```

See the [list of models available on Azure OpenAI Service](https://learn.microsoft.com/en-us/azure/ai-services/openai/concepts/models).If you need to configure the endpoint at runtime, you can set it to `endpoint = "env::AZURE_OPENAI_ENDPOINT"` to read from the environment variable `AZURE_OPENAI_ENDPOINT` on startup or `endpoint = "dynamic::azure_openai_endpoint"` to read from a dynamic credential `azure_openai_endpoint` on each inference.

### [​](https://www.tensorzero.com/docs/integrations/model-providers/azure\#credentials)  Credentials

You must set the `AZURE_OPENAI_API_KEY` environment variable before running the gateway.You can customize the credential location by setting the `api_key_location` to `env::YOUR_ENVIRONMENT_VARIABLE` or `dynamic::ARGUMENT_NAME`.
See the [Credential Management](https://www.tensorzero.com/docs/operations/manage-credentials) guide and [Configuration Reference](https://www.tensorzero.com/docs/gateway/configuration-reference) for more information.

### [​](https://www.tensorzero.com/docs/integrations/model-providers/azure\#deployment-docker-compose)  Deployment (Docker Compose)

Create a minimal Docker Compose configuration:

docker-compose.yml

Copy

```
# This is a simplified example for learning purposes. Do not use this in production.
# For production-ready deployments, see: https://www.tensorzero.com/docs/deployment/tensorzero-gateway

services:
  gateway:
    image: tensorzero/gateway
    volumes:
      - ./config:/app/config:ro
    command: --config-file /app/config/tensorzero.toml
    environment:
      - AZURE_OPENAI_API_KEY=${AZURE_OPENAI_API_KEY:?Environment variable AZURE_OPENAI_API_KEY must be set.}
    ports:
      - "3000:3000"
    extra_hosts:
      - "host.docker.internal:host-gateway"
```

You can start the gateway with `docker compose up`.

### [​](https://www.tensorzero.com/docs/integrations/model-providers/azure\#inference)  Inference

Make an inference request to the gateway:

Copy

```
curl -X POST http://localhost:3000/inference \
  -H "Content-Type: application/json" \
  -d '{
    "function_name": "my_function_name",
    "input": {
      "messages": [\
        {\
          "role": "user",\
          "content": "What is the capital of Japan?"\
        }\
      ]
    }
  }'
```

### [​](https://www.tensorzero.com/docs/integrations/model-providers/azure\#other-features)  Other Features

#### [​](https://www.tensorzero.com/docs/integrations/model-providers/azure\#generate-embeddings)  Generate embeddings

The Azure OpenAI Service model provider supports generating embeddings.
You can find a [complete code example on GitHub](https://github.com/tensorzero/tensorzero/tree/main/examples/guides/embeddings/providers/azure).

#### [​](https://www.tensorzero.com/docs/integrations/model-providers/azure\#azure-ai-foundry)  Azure AI Foundry

Azure AI Foundry provides access to models from multiple providers (Meta Llama, Mistral, xAI Grok, Microsoft Phi, Cohere, and more). See the [list of available models](https://ai.azure.com/explore/models).The same `azure` provider works with Azure AI Foundry.
The key difference is the endpoint URL.
All other configuration options (credentials, Docker Compose, inference) work the same as Azure OpenAI Service above.

[AWS SageMaker](https://www.tensorzero.com/docs/integrations/model-providers/aws-sagemaker) [DeepSeek](https://www.tensorzero.com/docs/integrations/model-providers/deepseek)

Ctrl+I

Assistant

Responses are generated using AI and may contain mistakes.

## DeepSeek Integration Guide
[Skip to main content](https://www.tensorzero.com/docs/integrations/model-providers/deepseek#content-area)

[TensorZero Docs home page![light logo](https://mintcdn.com/tensorzero/afFfERJmQ8YiZpCj/logo/light.svg?fit=max&auto=format&n=afFfERJmQ8YiZpCj&q=85&s=9396731a8a92cee7bd293dee05d71dac)![dark logo](https://mintcdn.com/tensorzero/afFfERJmQ8YiZpCj/logo/dark.svg?fit=max&auto=format&n=afFfERJmQ8YiZpCj&q=85&s=8d61c59531dbcdcace7fcdfc524bd4d2)](https://www.tensorzero.com/)

Search...

Ctrl K

Search...

Navigation

Model Providers

Getting Started with DeepSeek

[Guides](https://www.tensorzero.com/docs) [Integrations](https://www.tensorzero.com/docs/integrations/model-providers)

On this page

- [Simple Setup](https://www.tensorzero.com/docs/integrations/model-providers/deepseek#simple-setup)
- [Advanced Setup](https://www.tensorzero.com/docs/integrations/model-providers/deepseek#advanced-setup)
- [Configuration](https://www.tensorzero.com/docs/integrations/model-providers/deepseek#configuration)
- [Credentials](https://www.tensorzero.com/docs/integrations/model-providers/deepseek#credentials)
- [Deployment (Docker Compose)](https://www.tensorzero.com/docs/integrations/model-providers/deepseek#deployment-docker-compose)
- [Inference](https://www.tensorzero.com/docs/integrations/model-providers/deepseek#inference)

This guide shows how to set up a minimal deployment to use the TensorZero Gateway with DeepSeek.

## [​](https://www.tensorzero.com/docs/integrations/model-providers/deepseek\#simple-setup)  Simple Setup

You can use the short-hand `deepseek::model_name` to use a DeepSeek model with TensorZero, unless you need advanced features like fallbacks or custom credentials.You can use DeepSeek models in your TensorZero variants by setting the `model` field to `deepseek::model_name`.
For example:

Copy

```
[functions.my_function_name.variants.my_variant_name]
type = "chat_completion"
model = "deepseek::deepseek-chat"
```

Additionally, you can set `model_name` in the inference request to use a specific DeepSeek model, without having to configure a function and variant in TensorZero.

Copy

```
curl -X POST http://localhost:3000/inference \
  -H "Content-Type: application/json" \
  -d '{
    "model_name": "deepseek::deepseek-chat",
    "input": {
      "messages": [\
        {\
          "role": "user",\
          "content": "What is the capital of Japan?"\
        }\
      ]
    }
  }'
```

## [​](https://www.tensorzero.com/docs/integrations/model-providers/deepseek\#advanced-setup)  Advanced Setup

In more complex scenarios (e.g. fallbacks, custom credentials), you can configure your own model and DeepSeek provider in TensorZero.For this minimal setup, you’ll need just two files in your project directory:

Copy

```
- config/
  - tensorzero.toml
- docker-compose.yml
```

You can also find the complete code for this example on [GitHub](https://github.com/tensorzero/tensorzero/tree/main/examples/guides/providers/deepseek).

For production deployments, see our [Deployment Guide](https://www.tensorzero.com/docs/deployment/tensorzero-gateway).

### [​](https://www.tensorzero.com/docs/integrations/model-providers/deepseek\#configuration)  Configuration

Create a minimal configuration file that defines a model and a simple chat function:

config/tensorzero.toml

Copy

```
[models.deepseek_chat]
routing = ["deepseek"]

[models.deepseek_chat.providers.deepseek]
type = "deepseek"
model_name = "deepseek-chat"

[functions.my_function_name]
type = "chat"

[functions.my_function_name.variants.my_variant_name]
type = "chat_completion"
model = "deepseek_chat"
```

We have tested our integration with `deepseek-chat` (`DeepSeek-v3`) and `deepseek-reasoner` (`R1`).
DeepSeek only supports JSON mode for `deepseek-chat` and neither model supports tool use yet.
We include `thought` content blocks in the response and data model for reasoning models like `deepseek-reasoner`.

### [​](https://www.tensorzero.com/docs/integrations/model-providers/deepseek\#credentials)  Credentials

You must set the `DEEPSEEK_API_KEY` environment variable before running the gateway.You can customize the credential location by setting the `api_key_location` to `env::YOUR_ENVIRONMENT_VARIABLE` or `dynamic::ARGUMENT_NAME`.
See the [Credential Management](https://www.tensorzero.com/docs/operations/manage-credentials) guide and [Configuration Reference](https://www.tensorzero.com/docs/gateway/configuration-reference) for more information.

### [​](https://www.tensorzero.com/docs/integrations/model-providers/deepseek\#deployment-docker-compose)  Deployment (Docker Compose)

Create a minimal Docker Compose configuration:

docker-compose.yml

Copy

```
# This is a simplified example for learning purposes. Do not use this in production.
# For production-ready deployments, see: https://www.tensorzero.com/docs/deployment/tensorzero-gateway

services:
  gateway:
    image: tensorzero/gateway
    volumes:
      - ./config:/app/config:ro
    command: --config-file /app/config/tensorzero.toml
    environment:
      - DEEPSEEK_API_KEY=${DEEPSEEK_API_KEY:?Environment variable DEEPSEEK_API_KEY must be set.}
    ports:
      - "3000:3000"
    extra_hosts:
      - "host.docker.internal:host-gateway"
```

You can start the gateway with `docker compose up`.

## [​](https://www.tensorzero.com/docs/integrations/model-providers/deepseek\#inference)  Inference

Make an inference request to the gateway:

Copy

```
curl -X POST http://localhost:3000/inference \
  -H "Content-Type: application/json" \
  -d '{
    "function_name": "my_function_name",
    "input": {
      "messages": [\
        {\
          "role": "user",\
          "content": "What is the capital of Japan?"\
        }\
      ]
    }
  }'
```

[Azure](https://www.tensorzero.com/docs/integrations/model-providers/azure) [Fireworks](https://www.tensorzero.com/docs/integrations/model-providers/fireworks)

Ctrl+I

Assistant

Responses are generated using AI and may contain mistakes.

## Fireworks AI Integration
[Skip to main content](https://www.tensorzero.com/docs/integrations/model-providers/fireworks#content-area)

[TensorZero Docs home page![light logo](https://mintcdn.com/tensorzero/afFfERJmQ8YiZpCj/logo/light.svg?fit=max&auto=format&n=afFfERJmQ8YiZpCj&q=85&s=9396731a8a92cee7bd293dee05d71dac)![dark logo](https://mintcdn.com/tensorzero/afFfERJmQ8YiZpCj/logo/dark.svg?fit=max&auto=format&n=afFfERJmQ8YiZpCj&q=85&s=8d61c59531dbcdcace7fcdfc524bd4d2)](https://www.tensorzero.com/)

Search...

⌘K

Search...

Navigation

Model Providers

Getting Started with Fireworks AI

[Guides](https://www.tensorzero.com/docs) [Integrations](https://www.tensorzero.com/docs/integrations/model-providers)

On this page

- [Simple Setup](https://www.tensorzero.com/docs/integrations/model-providers/fireworks#simple-setup)
- [Advanced Setup](https://www.tensorzero.com/docs/integrations/model-providers/fireworks#advanced-setup)
- [Configuration](https://www.tensorzero.com/docs/integrations/model-providers/fireworks#configuration)
- [Credentials](https://www.tensorzero.com/docs/integrations/model-providers/fireworks#credentials)
- [Deployment (Docker Compose)](https://www.tensorzero.com/docs/integrations/model-providers/fireworks#deployment-docker-compose)
- [Inference](https://www.tensorzero.com/docs/integrations/model-providers/fireworks#inference)

This guide shows how to set up a minimal deployment to use the TensorZero Gateway with Fireworks.

## [​](https://www.tensorzero.com/docs/integrations/model-providers/fireworks\#simple-setup)  Simple Setup

You can use the short-hand `fireworks::model_name` to use a Fireworks model with TensorZero, unless you need advanced features like fallbacks or custom credentials.You can use Fireworks models in your TensorZero variants by setting the `model` field to `fireworks::model_name`.
For example:

Copy

```
[functions.my_function_name.variants.my_variant_name]
type = "chat_completion"
model = "fireworks::accounts/fireworks/models/llama-v3p1-8b-instruct"
```

Additionally, you can set `model_name` in the inference request to use a specific Fireworks model, without having to configure a function and variant in TensorZero.

Copy

```
curl -X POST http://localhost:3000/inference \
  -H "Content-Type: application/json" \
  -d '{
    "model_name": "fireworks::accounts/fireworks/models/llama-v3p1-8b-instruct",
    "input": {
      "messages": [\
        {\
          "role": "user",\
          "content": "What is the capital of Japan?"\
        }\
      ]
    }
  }'
```

## [​](https://www.tensorzero.com/docs/integrations/model-providers/fireworks\#advanced-setup)  Advanced Setup

In more complex scenarios (e.g. fallbacks, custom credentials), you can configure your own model and Fireworks provider in TensorZero.For this minimal setup, you’ll need just two files in your project directory:

Copy

```
- config/
  - tensorzero.toml
- docker-compose.yml
```

You can also find the complete code for this example on [GitHub](https://github.com/tensorzero/tensorzero/tree/main/examples/guides/providers/fireworks).

For production deployments, see our [Deployment Guide](https://www.tensorzero.com/docs/deployment/tensorzero-gateway).

### [​](https://www.tensorzero.com/docs/integrations/model-providers/fireworks\#configuration)  Configuration

Create a minimal configuration file that defines a model and a simple chat function:

config/tensorzero.toml

Copy

```
[models.llama3_1_8b_instruct]
routing = ["fireworks"]

[models.llama3_1_8b_instruct.providers.fireworks]
type = "fireworks"
model_name = "accounts/fireworks/models/llama-v3p1-8b-instruct"

[functions.my_function_name]
type = "chat"

[functions.my_function_name.variants.my_variant_name]
type = "chat_completion"
model = "llama3_1_8b_instruct"
```

See the [list of models available on Fireworks](https://fireworks.ai/models).
Custom models are also supported.

### [​](https://www.tensorzero.com/docs/integrations/model-providers/fireworks\#credentials)  Credentials

You must set the `FIREWORKS_API_KEY` environment variable before running the gateway.You can customize the credential location by setting the `api_key_location` to `env::YOUR_ENVIRONMENT_VARIABLE` or `dynamic::ARGUMENT_NAME`.
See the [Credential Management](https://www.tensorzero.com/docs/operations/manage-credentials) guide and [Configuration Reference](https://www.tensorzero.com/docs/gateway/configuration-reference) for more information.

### [​](https://www.tensorzero.com/docs/integrations/model-providers/fireworks\#deployment-docker-compose)  Deployment (Docker Compose)

Create a minimal Docker Compose configuration:

docker-compose.yml

Copy

```
# This is a simplified example for learning purposes. Do not use this in production.
# For production-ready deployments, see: https://www.tensorzero.com/docs/deployment/tensorzero-gateway

services:
  gateway:
    image: tensorzero/gateway
    volumes:
      - ./config:/app/config:ro
    command: --config-file /app/config/tensorzero.toml
    environment:
      - FIREWORKS_API_KEY=${FIREWORKS_API_KEY:?Environment variable FIREWORKS_API_KEY must be set.}
    ports:
      - "3000:3000"
    extra_hosts:
      - "host.docker.internal:host-gateway"
```

You can start the gateway with `docker compose up`.

## [​](https://www.tensorzero.com/docs/integrations/model-providers/fireworks\#inference)  Inference

Make an inference request to the gateway:

Copy

```
curl -X POST http://localhost:3000/inference \
  -H "Content-Type: application/json" \
  -d '{
    "function_name": "my_function_name",
    "input": {
      "messages": [\
        {\
          "role": "user",\
          "content": "What is the capital of Japan?"\
        }\
      ]
    }
  }'
```

[DeepSeek](https://www.tensorzero.com/docs/integrations/model-providers/deepseek) [GCP Vertex AI Anthropic](https://www.tensorzero.com/docs/integrations/model-providers/gcp-vertex-ai-anthropic)

⌘I

## TensorZero GCP Integration
[Skip to main content](https://www.tensorzero.com/docs/integrations/model-providers/gcp-vertex-ai-anthropic#content-area)

[TensorZero Docs home page![light logo](https://mintcdn.com/tensorzero/afFfERJmQ8YiZpCj/logo/light.svg?fit=max&auto=format&n=afFfERJmQ8YiZpCj&q=85&s=9396731a8a92cee7bd293dee05d71dac)![dark logo](https://mintcdn.com/tensorzero/afFfERJmQ8YiZpCj/logo/dark.svg?fit=max&auto=format&n=afFfERJmQ8YiZpCj&q=85&s=8d61c59531dbcdcace7fcdfc524bd4d2)](https://www.tensorzero.com/)

Search...

Ctrl K

Search...

Navigation

Model Providers

Getting Started with GCP Vertex AI Anthropic

[Guides](https://www.tensorzero.com/docs) [Integrations](https://www.tensorzero.com/docs/integrations/model-providers)

On this page

- [Setup](https://www.tensorzero.com/docs/integrations/model-providers/gcp-vertex-ai-anthropic#setup)
- [Configuration](https://www.tensorzero.com/docs/integrations/model-providers/gcp-vertex-ai-anthropic#configuration)
- [Credentials](https://www.tensorzero.com/docs/integrations/model-providers/gcp-vertex-ai-anthropic#credentials)
- [Deployment (Docker Compose)](https://www.tensorzero.com/docs/integrations/model-providers/gcp-vertex-ai-anthropic#deployment-docker-compose)
- [Inference](https://www.tensorzero.com/docs/integrations/model-providers/gcp-vertex-ai-anthropic#inference)

This guide shows how to set up a minimal deployment to use the TensorZero Gateway with GCP Vertex AI Anthropic.

## [​](https://www.tensorzero.com/docs/integrations/model-providers/gcp-vertex-ai-anthropic\#setup)  Setup

For this minimal setup, you’ll need just two files in your project directory:

Copy

```
- config/
  - tensorzero.toml
- docker-compose.yml
```

You can also find the complete code for this example on [GitHub](https://github.com/tensorzero/tensorzero/tree/main/examples/guides/providers/gcp-vertex-ai-anthropic).

For production deployments, see our [Deployment Guide](https://www.tensorzero.com/docs/deployment/tensorzero-gateway).

### [​](https://www.tensorzero.com/docs/integrations/model-providers/gcp-vertex-ai-anthropic\#configuration)  Configuration

Create a minimal configuration file that defines a model and a simple chat function:

config/tensorzero.toml

Copy

```
[models.claude_3_haiku_20240307]
routing = ["gcp_vertex_anthropic"]

[models.claude_3_haiku_20240307.providers.gcp_vertex_anthropic]
type = "gcp_vertex_anthropic"
model_id = "claude-3-haiku@20240307"  # or endpoint_id = "..." for fine-tuned models and custom endpoints
location = "us-central1"
project_id = "your-project-id"  # change this

[functions.my_function_name]
type = "chat"

[functions.my_function_name.variants.my_variant_name]
type = "chat_completion"
model = "claude_3_haiku_20240307"
```

See the [list of models available on GCP Vertex AI Anthropic](https://cloud.google.com/vertex-ai/generative-ai/docs/partner-models/use-claude).Alternatively, you can use the short-hand `gcp_vertex_anthropic::model_name` to use a GCP Vertex AI Anthropic model with TensorZero if you don’t need advanced features like fallbacks or custom credentials:

- `gcp_vertex_anthropic::projects/<PROJECT_ID>/locations/<REGION>/publishers/google/models/<MODEL_ID>`
- `gcp_vertex_anthropic::projects/<PROJECT_ID>/locations/<REGION>/endpoints/<ENDPOINT_ID>`

### [​](https://www.tensorzero.com/docs/integrations/model-providers/gcp-vertex-ai-anthropic\#credentials)  Credentials

By default, TensorZero reads the path to your GCP service account JSON file from the `GCP_VERTEX_CREDENTIALS_PATH` environment variable (using `path_from_env::GCP_VERTEX_CREDENTIALS_PATH`).You must generate a GCP service account key in JSON format as described [here](https://cloud.google.com/docs/authentication/provide-credentials-adc#service-account).You can customize the credential location using:

- `sdk`: use the Google Cloud SDK to auto-discover credentials
- `path::/path/to/credentials.json`: use a specific file path
- `path_from_env::YOUR_ENVIRONMENT_VARIABLE`: read file path from an environment variable (default behavior)
- `dynamic::ARGUMENT_NAME`: provide credentials dynamically at inference time
- `{ default = ..., fallback = ... }`: configure credential fallbacks

See the [Credential Management](https://www.tensorzero.com/docs/operations/manage-credentials) guide and [Configuration Reference](https://www.tensorzero.com/docs/gateway/configuration-reference) for more information.

### [​](https://www.tensorzero.com/docs/integrations/model-providers/gcp-vertex-ai-anthropic\#deployment-docker-compose)  Deployment (Docker Compose)

Create a minimal Docker Compose configuration:

docker-compose.yml

Copy

```
# This is a simplified example for learning purposes. Do not use this in production.
# For production-ready deployments, see: https://www.tensorzero.com/docs/deployment/tensorzero-gateway

services:
  gateway:
    image: tensorzero/gateway
    volumes:
      - ./config:/app/config:ro
      - ${GCP_VERTEX_CREDENTIALS_PATH:-/dev/null}:/app/gcp-credentials.json:ro
    command: --config-file /app/config/tensorzero.toml
    environment:
      - GCP_VERTEX_CREDENTIALS_PATH=${GCP_VERTEX_CREDENTIALS_PATH:+/app/gcp-credentials.json}
    ports:
      - "3000:3000"
    extra_hosts:
      - "host.docker.internal:host-gateway"
```

You can start the gateway with `docker compose up`.

## [​](https://www.tensorzero.com/docs/integrations/model-providers/gcp-vertex-ai-anthropic\#inference)  Inference

Make an inference request to the gateway:

Copy

```
curl -X POST http://localhost:3000/inference \
  -H "Content-Type: application/json" \
  -d '{
    "function_name": "my_function_name",
    "input": {
      "messages": [\
        {\
          "role": "user",\
          "content": "What is the capital of Japan?"\
        }\
      ]
    }
  }'
```

[Fireworks](https://www.tensorzero.com/docs/integrations/model-providers/fireworks) [GCP Vertex AI Gemini](https://www.tensorzero.com/docs/integrations/model-providers/gcp-vertex-ai-gemini)

Ctrl+I

Assistant

Responses are generated using AI and may contain mistakes.

## TensorZero GCP Integration
[Skip to main content](https://www.tensorzero.com/docs/integrations/model-providers/gcp-vertex-ai-gemini#content-area)

[TensorZero Docs home page![light logo](https://mintcdn.com/tensorzero/afFfERJmQ8YiZpCj/logo/light.svg?fit=max&auto=format&n=afFfERJmQ8YiZpCj&q=85&s=9396731a8a92cee7bd293dee05d71dac)![dark logo](https://mintcdn.com/tensorzero/afFfERJmQ8YiZpCj/logo/dark.svg?fit=max&auto=format&n=afFfERJmQ8YiZpCj&q=85&s=8d61c59531dbcdcace7fcdfc524bd4d2)](https://www.tensorzero.com/)

Search...

⌘K

Search...

Navigation

Model Providers

Infererence with GCP Vertex AI Gemini

[Guides](https://www.tensorzero.com/docs) [Integrations](https://www.tensorzero.com/docs/integrations/model-providers)

On this page

- [Setup](https://www.tensorzero.com/docs/integrations/model-providers/gcp-vertex-ai-gemini#setup)
- [Configuration](https://www.tensorzero.com/docs/integrations/model-providers/gcp-vertex-ai-gemini#configuration)
- [Credentials](https://www.tensorzero.com/docs/integrations/model-providers/gcp-vertex-ai-gemini#credentials)
- [Deployment (Docker Compose)](https://www.tensorzero.com/docs/integrations/model-providers/gcp-vertex-ai-gemini#deployment-docker-compose)
- [Inference](https://www.tensorzero.com/docs/integrations/model-providers/gcp-vertex-ai-gemini#inference)

This guide shows how to set up a minimal deployment to use the TensorZero Gateway with GCP Vertex AI Gemini.

## [​](https://www.tensorzero.com/docs/integrations/model-providers/gcp-vertex-ai-gemini\#setup)  Setup

For this minimal setup, you’ll need just two files in your project directory:

Copy

```
- config/
  - tensorzero.toml
- docker-compose.yml
```

You can also find the complete code for this example on [GitHub](https://github.com/tensorzero/tensorzero/tree/main/examples/guides/providers/gcp-vertex-ai-gemini).

For production deployments, see our [Deployment Guide](https://www.tensorzero.com/docs/deployment/tensorzero-gateway).

### [​](https://www.tensorzero.com/docs/integrations/model-providers/gcp-vertex-ai-gemini\#configuration)  Configuration

Create a minimal configuration file that defines a model and a simple chat function:

config/tensorzero.toml

Copy

```
[models.gemini_2_0_flash]
routing = ["gcp_vertex_gemini"]

[models.gemini_2_0_flash.providers.gcp_vertex_gemini]
type = "gcp_vertex_gemini"
model_id = "gemini-2.0-flash"  # or endpoint_id = "..." for fine-tuned models and custom endpoints
location = "us-central1"
project_id = "your-project-id"  # change this

[functions.my_function_name]
type = "chat"

[functions.my_function_name.variants.my_variant_name]
type = "chat_completion"
model = "gemini_2_0_flash"
```

See the [list of models available on GCP Vertex AI Gemini](https://cloud.google.com/vertex-ai/generative-ai/docs/learn/model-versions).Alternatively, you can use the short-hand `gcp_vertex_gemini::model_name` to use a GCP Vertex AI Gemini model with TensorZero if you don’t need advanced features like fallbacks or custom credentials:

- `gcp_vertex_gemini::projects/<PROJECT_ID>/locations/<REGION>/publishers/google/models/<MODEL_ID>`
- `gcp_vertex_gemini::projects/<PROJECT_ID>/locations/<REGION>/endpoints/<ENDPOINT_ID>`

### [​](https://www.tensorzero.com/docs/integrations/model-providers/gcp-vertex-ai-gemini\#credentials)  Credentials

By default, TensorZero reads the path to your GCP service account JSON file from the `GCP_VERTEX_CREDENTIALS_PATH` environment variable (using `path_from_env::GCP_VERTEX_CREDENTIALS_PATH`).You must generate a GCP service account key in JSON format as described [here](https://cloud.google.com/docs/authentication/provide-credentials-adc#service-account).You can customize the credential location using:

- `sdk`: use the Google Cloud SDK to auto-discover credentials
- `path::/path/to/credentials.json`: use a specific file path
- `path_from_env::YOUR_ENVIRONMENT_VARIABLE`: read file path from an environment variable (default behavior)
- `dynamic::ARGUMENT_NAME`: provide credentials dynamically at inference time
- `{ default = ..., fallback = ... }`: configure credential fallbacks

See the [Credential Management](https://www.tensorzero.com/docs/operations/manage-credentials) guide and [Configuration Reference](https://www.tensorzero.com/docs/gateway/configuration-reference) for more information.

### [​](https://www.tensorzero.com/docs/integrations/model-providers/gcp-vertex-ai-gemini\#deployment-docker-compose)  Deployment (Docker Compose)

Create a minimal Docker Compose configuration:

docker-compose.yml

Copy

```
# This is a simplified example for learning purposes. Do not use this in production.
# For production-ready deployments, see: https://www.tensorzero.com/docs/deployment/tensorzero-gateway

services:
  gateway:
    image: tensorzero/gateway
    volumes:
      - ./config:/app/config:ro
      - ${GCP_VERTEX_CREDENTIALS_PATH:-/dev/null}:/app/gcp-credentials.json:ro
    command: --config-file /app/config/tensorzero.toml
    environment:
      - GCP_VERTEX_CREDENTIALS_PATH=${GCP_VERTEX_CREDENTIALS_PATH:+/app/gcp-credentials.json}
    ports:
      - "3000:3000"
    extra_hosts:
      - "host.docker.internal:host-gateway"
```

You can start the gateway with `docker compose up`.

## [​](https://www.tensorzero.com/docs/integrations/model-providers/gcp-vertex-ai-gemini\#inference)  Inference

Make an inference request to the gateway:

Copy

```
curl -X POST http://localhost:3000/inference \
  -H "Content-Type: application/json" \
  -d '{
    "function_name": "my_function_name",
    "input": {
      "messages": [\
        {\
          "role": "user",\
          "content": "What is the capital of Japan?"\
        }\
      ]
    }
  }'
```

[GCP Vertex AI Anthropic](https://www.tensorzero.com/docs/integrations/model-providers/gcp-vertex-ai-anthropic) [Google AI Studio](https://www.tensorzero.com/docs/integrations/model-providers/google-ai-studio-gemini)

⌘I

## Google AI Studio Integration
[Skip to main content](https://www.tensorzero.com/docs/integrations/model-providers/google-ai-studio-gemini#content-area)

[TensorZero Docs home page![light logo](https://mintcdn.com/tensorzero/afFfERJmQ8YiZpCj/logo/light.svg?fit=max&auto=format&n=afFfERJmQ8YiZpCj&q=85&s=9396731a8a92cee7bd293dee05d71dac)![dark logo](https://mintcdn.com/tensorzero/afFfERJmQ8YiZpCj/logo/dark.svg?fit=max&auto=format&n=afFfERJmQ8YiZpCj&q=85&s=8d61c59531dbcdcace7fcdfc524bd4d2)](https://www.tensorzero.com/)

Search...

Ctrl K

Search...

Navigation

Model Providers

Getting Started with Google AI Studio (Gemini API)

[Guides](https://www.tensorzero.com/docs) [Integrations](https://www.tensorzero.com/docs/integrations/model-providers)

On this page

- [Simple Setup](https://www.tensorzero.com/docs/integrations/model-providers/google-ai-studio-gemini#simple-setup)
- [Advanced Setup](https://www.tensorzero.com/docs/integrations/model-providers/google-ai-studio-gemini#advanced-setup)
- [Configuration](https://www.tensorzero.com/docs/integrations/model-providers/google-ai-studio-gemini#configuration)
- [Credentials](https://www.tensorzero.com/docs/integrations/model-providers/google-ai-studio-gemini#credentials)
- [Deployment (Docker Compose)](https://www.tensorzero.com/docs/integrations/model-providers/google-ai-studio-gemini#deployment-docker-compose)
- [Inference](https://www.tensorzero.com/docs/integrations/model-providers/google-ai-studio-gemini#inference)

This guide shows how to set up a minimal deployment to use the TensorZero Gateway with Google AI Studio (Gemini API).

## [​](https://www.tensorzero.com/docs/integrations/model-providers/google-ai-studio-gemini\#simple-setup)  Simple Setup

You can use the short-hand `google_ai_studio_gemini::model_name` to use a Google AI Studio (Gemini API) model with TensorZero, unless you need advanced features like fallbacks or custom credentials.You can use Google AI Studio (Gemini API) models in your TensorZero variants by setting the `model` field to `google_ai_studio_gemini::model_name`.
For example:

Copy

```
[functions.my_function_name.variants.my_variant_name]
type = "chat_completion"
model = "google_ai_studio_gemini::gemini-1.5-flash-8b"
```

Additionally, you can set `model_name` in the inference request to use a specific Google AI Studio (Gemini API) model, without having to configure a function and variant in TensorZero.

Copy

```
curl -X POST http://localhost:3000/inference \
  -H "Content-Type: application/json" \
  -d '{
    "model_name": "google_ai_studio_gemini::gemini-1.5-flash-8b",
    "input": {
      "messages": [\
        {\
          "role": "user",\
          "content": "What is the capital of Japan?"\
        }\
      ]
    }
  }'
```

## [​](https://www.tensorzero.com/docs/integrations/model-providers/google-ai-studio-gemini\#advanced-setup)  Advanced Setup

In more complex scenarios (e.g. fallbacks, custom credentials), you can configure your own model and Google AI Studio (Gemini API) provider in TensorZero.For this minimal setup, you’ll need just two files in your project directory:

Copy

```
- config/
  - tensorzero.toml
- docker-compose.yml
```

You can also find the complete code for this example on [GitHub](https://github.com/tensorzero/tensorzero/tree/main/examples/guides/providers/google-ai-studio-gemini).

For production deployments, see our [Deployment Guide](https://www.tensorzero.com/docs/deployment/tensorzero-gateway).

### [​](https://www.tensorzero.com/docs/integrations/model-providers/google-ai-studio-gemini\#configuration)  Configuration

Create a minimal configuration file that defines a model and a simple chat function:

config/tensorzero.toml

Copy

```
[models.gemini_2_0_flash_lite]
routing = ["google_ai_studio_gemini"]

[models.gemini_2_0_flash_lite.providers.google_ai_studio_gemini]
type = "google_ai_studio_gemini"
model_name = "gemini-2.0-flash-lite"

[functions.my_function_name]
type = "chat"

[functions.my_function_name.variants.my_variant_name]
type = "chat_completion"
model = "gemini_2_0_flash_lite"
```

See the [list of models available on Google AI Studio (Gemini API)](https://ai.google.dev/gemini-api/docs/models/gemini).

### [​](https://www.tensorzero.com/docs/integrations/model-providers/google-ai-studio-gemini\#credentials)  Credentials

You must set the `GOOGLE_AI_STUDIO_API_KEY` environment variable before running the gateway.You can customize the credential location by setting the `api_key_location` to `env::YOUR_ENVIRONMENT_VARIABLE` or `dynamic::ARGUMENT_NAME`.
See the [Credential Management](https://www.tensorzero.com/docs/operations/manage-credentials) guide and [Configuration Reference](https://www.tensorzero.com/docs/gateway/configuration-reference) for more information.

### [​](https://www.tensorzero.com/docs/integrations/model-providers/google-ai-studio-gemini\#deployment-docker-compose)  Deployment (Docker Compose)

Create a minimal Docker Compose configuration:

docker-compose.yml

Copy

```
# This is a simplified example for learning purposes. Do not use this in production.
# For production-ready deployments, see: https://www.tensorzero.com/docs/deployment/tensorzero-gateway

services:
  gateway:
    image: tensorzero/gateway
    volumes:
      - ./config:/app/config:ro
    command: --config-file /app/config/tensorzero.toml
    environment:
      - GOOGLE_AI_STUDIO_API_KEY=${GOOGLE_AI_STUDIO_API_KEY:?Environment variable GOOGLE_AI_STUDIO_API_KEY must be set.}
    ports:
      - "3000:3000"
    extra_hosts:
      - "host.docker.internal:host-gateway"
```

You can start the gateway with `docker compose up`.

## [​](https://www.tensorzero.com/docs/integrations/model-providers/google-ai-studio-gemini\#inference)  Inference

Make an inference request to the gateway:

Copy

```
curl -X POST http://localhost:3000/inference \
  -H "Content-Type: application/json" \
  -d '{
    "function_name": "my_function_name",
    "input": {
      "messages": [\
        {\
          "role": "user",\
          "content": "What is the capital of Japan?"\
        }\
      ]
    }
  }'
```

[GCP Vertex AI Gemini](https://www.tensorzero.com/docs/integrations/model-providers/gcp-vertex-ai-gemini) [Groq](https://www.tensorzero.com/docs/integrations/model-providers/groq)

Ctrl+I

Assistant

Responses are generated using AI and may contain mistakes.

## Groq Integration Guide
[Skip to main content](https://www.tensorzero.com/docs/integrations/model-providers/groq#content-area)

[TensorZero Docs home page![light logo](https://mintcdn.com/tensorzero/afFfERJmQ8YiZpCj/logo/light.svg?fit=max&auto=format&n=afFfERJmQ8YiZpCj&q=85&s=9396731a8a92cee7bd293dee05d71dac)![dark logo](https://mintcdn.com/tensorzero/afFfERJmQ8YiZpCj/logo/dark.svg?fit=max&auto=format&n=afFfERJmQ8YiZpCj&q=85&s=8d61c59531dbcdcace7fcdfc524bd4d2)](https://www.tensorzero.com/)

Search...

Ctrl K

Search...

Navigation

Model Providers

Getting Started with Groq

[Guides](https://www.tensorzero.com/docs) [Integrations](https://www.tensorzero.com/docs/integrations/model-providers)

On this page

- [Simple Setup](https://www.tensorzero.com/docs/integrations/model-providers/groq#simple-setup)
- [Advanced Setup](https://www.tensorzero.com/docs/integrations/model-providers/groq#advanced-setup)
- [Configuration](https://www.tensorzero.com/docs/integrations/model-providers/groq#configuration)
- [Credentials](https://www.tensorzero.com/docs/integrations/model-providers/groq#credentials)
- [Deployment (Docker Compose)](https://www.tensorzero.com/docs/integrations/model-providers/groq#deployment-docker-compose)
- [Inference](https://www.tensorzero.com/docs/integrations/model-providers/groq#inference)

This guide shows how to set up a minimal deployment to use the TensorZero Gateway with Groq.

## [​](https://www.tensorzero.com/docs/integrations/model-providers/groq\#simple-setup)  Simple Setup

You can use the short-hand `groq::model_name` to use a Groq model with TensorZero, unless you need advanced features like fallbacks or custom credentials.You can use Groq models in your TensorZero variants by setting the `model` field to `groq::model_name`.
For example:

Copy

```
[functions.my_function_name.variants.my_variant_name]
type = "chat_completion"
model = "groq::meta-llama/llama-4-scout-17b-16e-instruct"
```

Additionally, you can set `model_name` in the inference request to use a specific Groq model, without having to configure a function and variant in TensorZero.

Copy

```
curl -X POST http://localhost:3000/inference \
  -H "Content-Type: application/json" \
  -d '{
    "model_name": "groq::meta-llama/llama-4-scout-17b-16e-instruct",
    "input": {
      "messages": [\
        {\
          "role": "user",\
          "content": "What is the capital of Japan?"\
        }\
      ]
    }
  }'
```

## [​](https://www.tensorzero.com/docs/integrations/model-providers/groq\#advanced-setup)  Advanced Setup

In more complex scenarios (e.g. fallbacks, custom credentials), you can configure your own model and Groq provider in TensorZero.For this minimal setup, you’ll need just two files in your project directory:

Copy

```
- config/
  - tensorzero.toml
- docker-compose.yml
```

You can also find the complete code for this example on [GitHub](https://github.com/tensorzero/tensorzero/tree/main/examples/guides/providers/groq).

For production deployments, see our [Deployment Guide](https://www.tensorzero.com/docs/deployment/tensorzero-gateway).

### [​](https://www.tensorzero.com/docs/integrations/model-providers/groq\#configuration)  Configuration

Create a minimal configuration file that defines a model and a simple chat function:

config/tensorzero.toml

Copy

```
[models.llama4_scout_17b_16e_instruct]
routing = ["groq"]

[models.llama4_scout_17b_16e_instruct.providers.groq]
type = "groq"
model_name = "meta-llama/llama-4-scout-17b-16e-instruct"

[functions.my_function_name]
type = "chat"

[functions.my_function_name.variants.my_variant_name]
type = "chat_completion"
model = "llama4_scout_17b_16e_instruct"
```

See the [list of models available on Groq](https://groq.com/pricing).

### [​](https://www.tensorzero.com/docs/integrations/model-providers/groq\#credentials)  Credentials

You must set the `GROQ_API_KEY` environment variable before running the gateway.You can customize the credential location by setting the `api_key_location` to `env::YOUR_ENVIRONMENT_VARIABLE` or `dynamic::ARGUMENT_NAME`.
See the [Credential Management](https://www.tensorzero.com/docs/operations/manage-credentials) guide and [Configuration Reference](https://www.tensorzero.com/docs/gateway/configuration-reference) for more information.

### [​](https://www.tensorzero.com/docs/integrations/model-providers/groq\#deployment-docker-compose)  Deployment (Docker Compose)

Create a minimal Docker Compose configuration:

docker-compose.yml

Copy

```
# This is a simplified example for learning purposes. Do not use this in production.
# For production-ready deployments, see: https://www.tensorzero.com/docs/deployment/tensorzero-gateway

services:
  gateway:
    image: tensorzero/gateway
    volumes:
      - ./config:/app/config:ro
    command: --config-file /app/config/tensorzero.toml
    environment:
      - GROQ_API_KEY=${GROQ_API_KEY:?Environment variable GROQ_API_KEY must be set.}
    ports:
      - "3000:3000"
    extra_hosts:
      - "host.docker.internal:host-gateway"
```

You can start the gateway with `docker compose up`.

## [​](https://www.tensorzero.com/docs/integrations/model-providers/groq\#inference)  Inference

Make an inference request to the gateway:

Copy

```
curl -X POST http://localhost:3000/inference \
  -H "Content-Type: application/json" \
  -d '{
    "function_name": "my_function_name",
    "input": {
      "messages": [\
        {\
          "role": "user",\
          "content": "What is the capital of Japan?"\
        }\
      ]
    }
  }'
```

[Google AI Studio](https://www.tensorzero.com/docs/integrations/model-providers/google-ai-studio-gemini) [Hyperbolic](https://www.tensorzero.com/docs/integrations/model-providers/hyperbolic)

Ctrl+I

Assistant

Responses are generated using AI and may contain mistakes.

## Hyperbolic API Setup
[Skip to main content](https://www.tensorzero.com/docs/integrations/model-providers/hyperbolic#content-area)

[TensorZero Docs home page![light logo](https://mintcdn.com/tensorzero/afFfERJmQ8YiZpCj/logo/light.svg?fit=max&auto=format&n=afFfERJmQ8YiZpCj&q=85&s=9396731a8a92cee7bd293dee05d71dac)![dark logo](https://mintcdn.com/tensorzero/afFfERJmQ8YiZpCj/logo/dark.svg?fit=max&auto=format&n=afFfERJmQ8YiZpCj&q=85&s=8d61c59531dbcdcace7fcdfc524bd4d2)](https://www.tensorzero.com/)

Search...

Ctrl K

Search...

Navigation

Model Providers

Getting Started with Hyperbolic

[Guides](https://www.tensorzero.com/docs) [Integrations](https://www.tensorzero.com/docs/integrations/model-providers)

On this page

- [Simple Setup](https://www.tensorzero.com/docs/integrations/model-providers/hyperbolic#simple-setup)
- [Advanced Setup](https://www.tensorzero.com/docs/integrations/model-providers/hyperbolic#advanced-setup)
- [Configuration](https://www.tensorzero.com/docs/integrations/model-providers/hyperbolic#configuration)
- [Credentials](https://www.tensorzero.com/docs/integrations/model-providers/hyperbolic#credentials)
- [Deployment (Docker Compose)](https://www.tensorzero.com/docs/integrations/model-providers/hyperbolic#deployment-docker-compose)
- [Inference](https://www.tensorzero.com/docs/integrations/model-providers/hyperbolic#inference)

This guide shows how to set up a minimal deployment to use the TensorZero Gateway with the Hyperbolic API.

## [​](https://www.tensorzero.com/docs/integrations/model-providers/hyperbolic\#simple-setup)  Simple Setup

You can use the short-hand `hyperbolic::model_name` to use a Hyperbolic model with TensorZero, unless you need advanced features like fallbacks or custom credentials.You can use Hyperbolic models in your TensorZero variants by setting the `model` field to `hyperbolic::model_name`.
For example:

Copy

```
[functions.my_function_name.variants.my_variant_name]
type = "chat_completion"
model = "hyperbolic::meta-llama/Meta-Llama-3-70B-Instruct"
```

Additionally, you can set `model_name` in the inference request to use a specific Hyperbolic model, without having to configure a function and variant in TensorZero.

Copy

```
curl -X POST http://localhost:3000/inference \
  -H "Content-Type: application/json" \
  -d '{
    "model_name": "hyperbolic::meta-llama/Meta-Llama-3-70B-Instruct",
    "input": {
      "messages": [\
        {\
          "role": "user",\
          "content": "What is the capital of Japan?"\
        }\
      ]
    }
  }'
```

## [​](https://www.tensorzero.com/docs/integrations/model-providers/hyperbolic\#advanced-setup)  Advanced Setup

In more complex scenarios (e.g. fallbacks, custom credentials), you can configure your own model and Hyperbolic provider in TensorZero.
For this minimal setup, you’ll need just two files in your project directory:

Copy

```
- config/
  - tensorzero.toml
- docker-compose.yml
```

You can also find the complete code for this example on [GitHub](https://github.com/tensorzero/tensorzero/tree/main/examples/guides/providers/hyperbolic).

For production deployments, see our [Deployment Guide](https://www.tensorzero.com/docs/deployment/tensorzero-gateway).

### [​](https://www.tensorzero.com/docs/integrations/model-providers/hyperbolic\#configuration)  Configuration

Create a minimal configuration file that defines a model and a simple chat function:

config/tensorzero.toml

Copy

```
[models."meta-llama/Meta-Llama-3-70B-Instruct"]
routing = ["hyperbolic"]

[models."meta-llama/Meta-Llama-3-70B-Instruct".providers.hyperbolic]
type = "hyperbolic"
model_name = "meta-llama/Meta-Llama-3-70B-Instruct"

[functions.my_function_name]
type = "chat"

[functions.my_function_name.variants.my_variant_name]
type = "chat_completion"
model = "meta-llama/Meta-Llama-3-70B-Instruct"
```

See the [list of models available on Hyperbolic](https://app.hyperbolic.xyz/models).

### [​](https://www.tensorzero.com/docs/integrations/model-providers/hyperbolic\#credentials)  Credentials

You must set the `HYPERBOLIC_API_KEY` environment variable before running the gateway.You can customize the credential location by setting the `api_key_location` to `env::YOUR_ENVIRONMENT_VARIABLE` or `dynamic::ARGUMENT_NAME`.
See the [Credential Management](https://www.tensorzero.com/docs/operations/manage-credentials) guide and [Configuration Reference](https://www.tensorzero.com/docs/gateway/configuration-reference) for more information.

### [​](https://www.tensorzero.com/docs/integrations/model-providers/hyperbolic\#deployment-docker-compose)  Deployment (Docker Compose)

Create a minimal Docker Compose configuration:

docker-compose.yml

Copy

```
# This is a simplified example for learning purposes. Do not use this in production.
# For production-ready deployments, see: https://www.tensorzero.com/docs/deployment/tensorzero-gateway

services:
  gateway:
    image: tensorzero/gateway
    volumes:
      - ./config:/app/config:ro
    command: --config-file /app/config/tensorzero.toml
    environment:
      - HYPERBOLIC_API_KEY=${HYPERBOLIC_API_KEY:?Environment variable HYPERBOLIC_API_KEY must be set.}
    ports:
      - "3000:3000"
    extra_hosts:
      - "host.docker.internal:host-gateway"
```

You can start the gateway with `docker compose up`.

## [​](https://www.tensorzero.com/docs/integrations/model-providers/hyperbolic\#inference)  Inference

Make an inference request to the gateway:

Copy

```
curl -X POST http://localhost:3000/inference \
  -H "Content-Type: application/json" \
  -d '{
    "function_name": "my_function_name",
    "input": {
      "messages": [\
        {\
          "role": "user",\
          "content": "What is the capital of Japan?"\
        }\
      ]
    }
  }'
```

[Groq](https://www.tensorzero.com/docs/integrations/model-providers/groq) [Mistral](https://www.tensorzero.com/docs/integrations/model-providers/mistral)

Ctrl+I

Assistant

Responses are generated using AI and may contain mistakes.

## Model Providers Overview
[Skip to main content](https://www.tensorzero.com/docs/integrations/model-providers#content-area)

[TensorZero Docs home page![light logo](https://mintcdn.com/tensorzero/afFfERJmQ8YiZpCj/logo/light.svg?fit=max&auto=format&n=afFfERJmQ8YiZpCj&q=85&s=9396731a8a92cee7bd293dee05d71dac)![dark logo](https://mintcdn.com/tensorzero/afFfERJmQ8YiZpCj/logo/dark.svg?fit=max&auto=format&n=afFfERJmQ8YiZpCj&q=85&s=8d61c59531dbcdcace7fcdfc524bd4d2)](https://www.tensorzero.com/)

Search...

Ctrl K

Search...

Navigation

Model Providers

Overview

[Guides](https://www.tensorzero.com/docs) [Integrations](https://www.tensorzero.com/docs/integrations/model-providers)

On this page

- [Model Providers](https://www.tensorzero.com/docs/integrations/model-providers#model-providers)
- [Limitations](https://www.tensorzero.com/docs/integrations/model-providers#limitations)

The TensorZero Gateway integrates with the major LLM providers.

## [​](https://www.tensorzero.com/docs/integrations/model-providers\#model-providers)  Model Providers

| Provider | Chat Functions | JSON Functions | Streaming | Tool Use | Multimodal | Embeddings | Batch |
| --- | --- | --- | --- | --- | --- | --- | --- |
| [Anthropic](https://www.tensorzero.com/docs/integrations/model-providers/anthropic) | ✅ | ✅ | ✅ | ✅ | ✅ | ❌ | ❌ |
| [AWS Bedrock](https://www.tensorzero.com/docs/integrations/model-providers/aws-bedrock) | ✅ | ✅ | ✅ | ✅ | ✅ | ❌ | ❌ |
| [AWS SageMaker](https://www.tensorzero.com/docs/integrations/model-providers/aws-sagemaker) | ✅ | ✅ | ✅ | ✅ | ❌ | ❌ | ❌ |
| [Azure OpenAI Service](https://www.tensorzero.com/docs/integrations/model-providers/azure) | ✅ | ✅ | ✅ | ✅ | ❌ | ✅ | ❌ |
| [DeepSeek](https://www.tensorzero.com/docs/integrations/model-providers/deepseek) | ✅ | ✅ | ⚠️ | ❌ | ❌ | ❌ | ❌ |
| [Fireworks AI](https://www.tensorzero.com/docs/integrations/model-providers/fireworks) | ✅ | ✅ | ✅ | ✅ | ❌ | ❌ | ❌ |
| [GCP Vertex AI Anthropic](https://www.tensorzero.com/docs/integrations/model-providers/gcp-vertex-ai-anthropic) | ✅ | ✅ | ✅ | ✅ | ✅ | ❌ | ❌ |
| [GCP Vertex AI Gemini](https://www.tensorzero.com/docs/integrations/model-providers/gcp-vertex-ai-gemini) | ✅ | ✅ | ✅ | ✅ | ✅ | ❌ | ✅ |
| [Google AI Studio Gemini](https://www.tensorzero.com/docs/integrations/model-providers/google-ai-studio-gemini) | ✅ | ✅ | ✅ | ✅ | ✅ | ❌ | ❌ |
| [Groq](https://www.tensorzero.com/docs/integrations/model-providers/groq) | ✅ | ✅ | ✅ | ✅ | ❌ | ❌ | ❌ |
| [Hyperbolic](https://www.tensorzero.com/docs/integrations/model-providers/hyperbolic) | ✅ | ⚠️ | ✅ | ❌ | ❌ | ❌ | ❌ |
| [Mistral](https://www.tensorzero.com/docs/integrations/model-providers/mistral) | ✅ | ✅ | ✅ | ✅ | ❌ | ❌ | ❌ |
| [OpenAI](https://www.tensorzero.com/docs/integrations/model-providers/openai) and<br>[OpenAI-Compatible](https://www.tensorzero.com/docs/integrations/model-providers/openai-compatible) | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| [OpenRouter](https://www.tensorzero.com/docs/integrations/model-providers/openrouter) | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ❌ |
| [SGLang](https://www.tensorzero.com/docs/integrations/model-providers/sglang) | ✅ | ✅ | ✅ | ❌ | ❌ | ❌ | ❌ |
| [TGI](https://www.tensorzero.com/docs/integrations/model-providers/tgi) | ✅ | ✅ | ⚠️ | ❌ | ❌ | ❌ | ❌ |
| [Together AI](https://www.tensorzero.com/docs/integrations/model-providers/together) | ✅ | ✅ | ✅ | ✅ | ❌ | ❌ | ❌ |
| [vLLM](https://www.tensorzero.com/docs/integrations/model-providers/vllm) | ✅ | ✅ | ✅ | ✅ | ❌ | ❌ | ❌ |
| [xAI](https://www.tensorzero.com/docs/integrations/model-providers/xai) | ✅ | ✅ | ✅ | ✅ | ❌ | ❌ | ❌ |

### [​](https://www.tensorzero.com/docs/integrations/model-providers\#limitations)  Limitations

The TensorZero Gateway makes a best effort to normalize configuration across providers.
For example, certain providers don’t support `tool_choice: required`; in these cases,
TensorZero Gateway will coerce the request to `tool_choice: auto` under the hood.Currently, Fireworks AI and OpenAI are the only providers that support `parallel_tool_calls`.
Additionally, TensorZero Gateway supports `strict` (commonly referred to as Structured Outputs, Guided Decoding, or similar names) for Azure, GCP Vertex AI Gemini, Google AI Studio, OpenAI, Together AI, vLLM, and xAI.
You can also enable `strict` for Anthropic with `beta_structured_outputs = true`.Below are the known limitations for each supported model provider.

- **Anthropic**
  - The Anthropic API doesn’t support consecutive messages from the same role.
  - The Anthropic API doesn’t support `tool_choice: none`.
  - The Anthropic API doesn’t support `seed`.
  - Structured Outputs (strict mode) requires enabling `beta_structured_outputs = true` in the provider configuration.
- **AWS Bedrock**
  - The TensorZero Gateway currently doesn’t support AWS Bedrock guardrails and traces.
  - The TensorZero Gateway uses a non-standard structure for storing `ModelInference.raw_response` for AWS Bedrock inference requests.
  - The AWS Bedrock API doesn’t support `tool_choice: none`.
  - The AWS Bedrock API doesn’t support `seed`.
- **Azure OpenAI Service**
  - The Azure OpenAI Service API doesn’t provide usage information when streaming.
  - The Azure OpenAI Service API doesn’t support `tool_choice: required`.
- **DeepSeek**
  - The `deepseek-chat` model doesn’t support tool use for production use cases.
  - The `deepseek-reasoner` model doesn’t support JSON mode or tool use.
  - The TensorZero Gateway doesn’t return `thought` blocks in the response (coming soon!).
- **Fireworks AI**
  - The Fireworks API doesn’t support `seed`.
- **GCP Vertex AI**
  - The TensorZero Gateway currently only supports the Gemini and Anthropic models.
  - The GCP Vertex AI API doesn’t support `tool_choice: required` for Gemini Flash models.
  - The Anthropic models have the same limitations as those listed under the Anthropic provider.
- **Hyperbolic**
  - The Hyperbolic provider doesn’t support JSON mode or tool use. JSON functions are supported with `json_mode = "off"` (not recommended).
- **Mistral**
  - The Mistral API doesn’t support `seed`.
- **SGLang**
  - There is no support for tools
- **TGI**
  - The TGI API doesn’t support streaming JSON mode.
  - There is very limited support for tool use so we don’t recommend using it.
- **Together AI**
  - The Together AI API doesn’t seem to respect `tool_choice` in many cases.
- **xAI**
  - The xAI provider doesn’t support JSON mode. JSON functions are supported with `json_mode = "tool"` (recommended) or `json_mode = "off"`.
  - The xAI API has issues with multi-turn tool use ( [bug report](https://gist.github.com/GabrielBianconi/47a4247cfd8b6689e7228f654806272d)).
  - The xAI API has issues with `tool_choice: none` ( [bug report](https://gist.github.com/GabrielBianconi/2199022d0ea8518e06d366fb613c5bb5)).

[Anthropic](https://www.tensorzero.com/docs/integrations/model-providers/anthropic)

Ctrl+I

Assistant

Responses are generated using AI and may contain mistakes.

## Mistral Setup Guide
[Skip to main content](https://www.tensorzero.com/docs/integrations/model-providers/mistral#content-area)

[TensorZero Docs home page![light logo](https://mintcdn.com/tensorzero/afFfERJmQ8YiZpCj/logo/light.svg?fit=max&auto=format&n=afFfERJmQ8YiZpCj&q=85&s=9396731a8a92cee7bd293dee05d71dac)![dark logo](https://mintcdn.com/tensorzero/afFfERJmQ8YiZpCj/logo/dark.svg?fit=max&auto=format&n=afFfERJmQ8YiZpCj&q=85&s=8d61c59531dbcdcace7fcdfc524bd4d2)](https://www.tensorzero.com/)

Search...

Ctrl K

Search...

Navigation

Model Providers

Getting Started with Mistral

[Guides](https://www.tensorzero.com/docs) [Integrations](https://www.tensorzero.com/docs/integrations/model-providers)

On this page

- [Simple Setup](https://www.tensorzero.com/docs/integrations/model-providers/mistral#simple-setup)
- [Advanced Setup](https://www.tensorzero.com/docs/integrations/model-providers/mistral#advanced-setup)
- [Configuration](https://www.tensorzero.com/docs/integrations/model-providers/mistral#configuration)
- [Credentials](https://www.tensorzero.com/docs/integrations/model-providers/mistral#credentials)
- [Deployment (Docker Compose)](https://www.tensorzero.com/docs/integrations/model-providers/mistral#deployment-docker-compose)
- [Inference](https://www.tensorzero.com/docs/integrations/model-providers/mistral#inference)

This guide shows how to set up a minimal deployment to use the TensorZero Gateway with the Mistral API.

## [​](https://www.tensorzero.com/docs/integrations/model-providers/mistral\#simple-setup)  Simple Setup

You can use the short-hand `mistral::model_name` to use a Mistral model with TensorZero, unless you need advanced features like fallbacks or custom credentials.You can use Mistral models in your TensorZero variants by setting the `model` field to `mistral::model_name`.
For example:

Copy

```
[functions.my_function_name.variants.my_variant_name]
type = "chat_completion"
model = "mistral::ministral-8b-2410"
```

Additionally, you can set `model_name` in the inference request to use a specific Mistral model, without having to configure a function and variant in TensorZero.

Copy

```
curl -X POST http://localhost:3000/inference \
  -H "Content-Type: application/json" \
  -d '{
    "model_name": "mistral::ministral-8b-2410",
    "input": {
      "messages": [\
        {\
          "role": "user",\
          "content": "What is the capital of Japan?"\
        }\
      ]
    }
  }'
```

## [​](https://www.tensorzero.com/docs/integrations/model-providers/mistral\#advanced-setup)  Advanced Setup

In more complex scenarios (e.g. fallbacks, custom credentials), you can configure your own model and Mistral provider in TensorZero.For this minimal setup, you’ll need just two files in your project directory:

Copy

```
- config/
  - tensorzero.toml
- docker-compose.yml
```

You can also find the complete code for this example on [GitHub](https://github.com/tensorzero/tensorzero/tree/main/examples/guides/providers/mistral).

For production deployments, see our [Deployment Guide](https://www.tensorzero.com/docs/deployment/tensorzero-gateway).

### [​](https://www.tensorzero.com/docs/integrations/model-providers/mistral\#configuration)  Configuration

Create a minimal configuration file that defines a model and a simple chat function:

config/tensorzero.toml

Copy

```
[models.ministral_8b_2410]
routing = ["mistral"]

[models.ministral_8b_2410.providers.mistral]
type = "mistral"
model_name = "ministral-8b-2410"

[functions.my_function_name]
type = "chat"

[functions.my_function_name.variants.my_variant_name]
type = "chat_completion"
model = "ministral_8b_2410"
```

See the [list of models available on Mistral](https://docs.mistral.ai/getting-started/models).

### [​](https://www.tensorzero.com/docs/integrations/model-providers/mistral\#credentials)  Credentials

You must set the `MISTRAL_API_KEY` environment variable before running the gateway.You can customize the credential location by setting the `api_key_location` to `env::YOUR_ENVIRONMENT_VARIABLE` or `dynamic::ARGUMENT_NAME`.
See the [Credential Management](https://www.tensorzero.com/docs/operations/manage-credentials) guide and [Configuration Reference](https://www.tensorzero.com/docs/gateway/configuration-reference) for more information.

### [​](https://www.tensorzero.com/docs/integrations/model-providers/mistral\#deployment-docker-compose)  Deployment (Docker Compose)

Create a minimal Docker Compose configuration:

docker-compose.yml

Copy

```
# This is a simplified example for learning purposes. Do not use this in production.
# For production-ready deployments, see: https://www.tensorzero.com/docs/deployment/tensorzero-gateway

services:
  gateway:
    image: tensorzero/gateway
    volumes:
      - ./config:/app/config:ro
    command: --config-file /app/config/tensorzero.toml
    environment:
      - MISTRAL_API_KEY=${MISTRAL_API_KEY:?Environment variable MISTRAL_API_KEY must be set.}
    ports:
      - "3000:3000"
    extra_hosts:
      - "host.docker.internal:host-gateway"
```

You can start the gateway with `docker compose up`.

## [​](https://www.tensorzero.com/docs/integrations/model-providers/mistral\#inference)  Inference

Make an inference request to the gateway:

Copy

```
curl -X POST http://localhost:3000/inference \
  -H "Content-Type: application/json" \
  -d '{
    "function_name": "my_function_name",
    "input": {
      "messages": [\
        {\
          "role": "user",\
          "content": "What is the capital of Japan?"\
        }\
      ]
    }
  }'
```

[Hyperbolic](https://www.tensorzero.com/docs/integrations/model-providers/hyperbolic) [OpenAI-Compatible](https://www.tensorzero.com/docs/integrations/model-providers/openai-compatible)

Ctrl+I

Assistant

Responses are generated using AI and may contain mistakes.

## OpenAI Integration Guide
[Skip to main content](https://www.tensorzero.com/docs/integrations/model-providers/openai#content-area)

[TensorZero Docs home page![light logo](https://mintcdn.com/tensorzero/afFfERJmQ8YiZpCj/logo/light.svg?fit=max&auto=format&n=afFfERJmQ8YiZpCj&q=85&s=9396731a8a92cee7bd293dee05d71dac)![dark logo](https://mintcdn.com/tensorzero/afFfERJmQ8YiZpCj/logo/dark.svg?fit=max&auto=format&n=afFfERJmQ8YiZpCj&q=85&s=8d61c59531dbcdcace7fcdfc524bd4d2)](https://www.tensorzero.com/)

Search...

Ctrl K

Search...

Navigation

Model Providers

Getting Started with OpenAI

[Guides](https://www.tensorzero.com/docs) [Integrations](https://www.tensorzero.com/docs/integrations/model-providers)

On this page

- [Simple Setup](https://www.tensorzero.com/docs/integrations/model-providers/openai#simple-setup)
- [Chat Completions API](https://www.tensorzero.com/docs/integrations/model-providers/openai#chat-completions-api)
- [Responses API](https://www.tensorzero.com/docs/integrations/model-providers/openai#responses-api)
- [Advanced Setup](https://www.tensorzero.com/docs/integrations/model-providers/openai#advanced-setup)
- [Configuration](https://www.tensorzero.com/docs/integrations/model-providers/openai#configuration)
- [Credentials](https://www.tensorzero.com/docs/integrations/model-providers/openai#credentials)
- [Deployment (Docker Compose)](https://www.tensorzero.com/docs/integrations/model-providers/openai#deployment-docker-compose)
- [Inference](https://www.tensorzero.com/docs/integrations/model-providers/openai#inference)
- [Other Features](https://www.tensorzero.com/docs/integrations/model-providers/openai#other-features)
- [Generate embeddings](https://www.tensorzero.com/docs/integrations/model-providers/openai#generate-embeddings)

This guide shows how to set up a minimal deployment to use the TensorZero Gateway with the OpenAI API.

## [​](https://www.tensorzero.com/docs/integrations/model-providers/openai\#simple-setup)  Simple Setup

You can use the short-hand `openai::model_name` to use an OpenAI model with TensorZero, unless you need advanced features like fallbacks or custom credentials.

### [​](https://www.tensorzero.com/docs/integrations/model-providers/openai\#chat-completions-api)  Chat Completions API

You can use OpenAI models in your TensorZero variants by setting the `model` field to `openai::model_name`.
For example:

Copy

```
[functions.my_function_name.variants.my_variant_name]
type = "chat_completion"
model = "openai::gpt-4o-mini-2024-07-18"
```

Additionally, you can set `model_name` in the inference request to use a specific OpenAI model, without having to configure a function and variant in TensorZero.

Copy

```
curl -X POST http://localhost:3000/inference \
  -H "Content-Type: application/json" \
  -d '{
    "model_name": "openai::gpt-4o-mini-2024-07-18",
    "input": {
      "messages": [\
        {\
          "role": "user",\
          "content": "What is the capital of Japan?"\
        }\
      ]
    }
  }'
```

### [​](https://www.tensorzero.com/docs/integrations/model-providers/openai\#responses-api)  Responses API

For models that use the OpenAI Responses API (like `gpt-5`), use the `openai::responses::model_name` shorthand:

Copy

```
[functions.my_function_name.variants.my_variant_name]
type = "chat_completion"
model = "openai::responses::gpt-5-codex"
```

You can also use `model_name` in inference requests:

Copy

```
curl -X POST http://localhost:3000/inference \
  -H "Content-Type: application/json" \
  -d '{
    "model_name": "openai::responses::gpt-5-codex",
    "input": {
      "messages": [\
        {\
          "role": "user",\
          "content": "What is the capital of Japan?"\
        }\
      ]
    }
  }'
```

See the [OpenAI Responses API guide](https://www.tensorzero.com/docs/gateway/call-the-openai-responses-api) for more details on using this API.

## [​](https://www.tensorzero.com/docs/integrations/model-providers/openai\#advanced-setup)  Advanced Setup

For more complex scenarios (e.g. fallbacks, custom credentials), you can configure your own model and OpenAI provider in TensorZero.For this minimal setup, you’ll need just two files in your project directory:

Copy

```
- config/
  - tensorzero.toml
- docker-compose.yml
```

You can also find the complete code for this example on [GitHub](https://github.com/tensorzero/tensorzero/tree/main/examples/guides/providers/openai).

For production deployments, see our [Deployment Guide](https://www.tensorzero.com/docs/deployment/tensorzero-gateway).

### [​](https://www.tensorzero.com/docs/integrations/model-providers/openai\#configuration)  Configuration

Create a minimal configuration file that defines a model and a simple chat function:

config/tensorzero.toml

Copy

```
[models.gpt_4o_mini_2024_07_18]
routing = ["openai"]

[models.gpt_4o_mini_2024_07_18.providers.openai]
type = "openai"
model_name = "gpt-4o-mini-2024-07-18"

[functions.my_function_name]
type = "chat"

[functions.my_function_name.variants.my_variant_name]
type = "chat_completion"
model = "gpt_4o_mini_2024_07_18"
```

See the [list of models available on OpenAI](https://platform.openai.com/docs/models/).See the [Configuration Reference](https://www.tensorzero.com/docs/gateway/configuration-reference) for optional fields (e.g. overwriting `api_base`).

### [​](https://www.tensorzero.com/docs/integrations/model-providers/openai\#credentials)  Credentials

You must set the `OPENAI_API_KEY` environment variable before running the gateway.You can customize the credential location by setting the `api_key_location` to `env::YOUR_ENVIRONMENT_VARIABLE` or `dynamic::ARGUMENT_NAME`.
See the [Credential Management](https://www.tensorzero.com/docs/operations/manage-credentials) guide and [Configuration Reference](https://www.tensorzero.com/docs/gateway/configuration-reference) for more information.Additionally, see the [OpenAI-Compatible](https://www.tensorzero.com/docs/integrations/model-providers/openai-compatible) guide for more information on how to use other OpenAI-Compatible providers.

### [​](https://www.tensorzero.com/docs/integrations/model-providers/openai\#deployment-docker-compose)  Deployment (Docker Compose)

Create a minimal Docker Compose configuration:

docker-compose.yml

Copy

```
# This is a simplified example for learning purposes. Do not use this in production.
# For production-ready deployments, see: https://www.tensorzero.com/docs/deployment/tensorzero-gateway

services:
  gateway:
    image: tensorzero/gateway
    volumes:
      - ./config:/app/config:ro
    command: --config-file /app/config/tensorzero.toml
    environment:
      - OPENAI_API_KEY=${OPENAI_API_KEY:?Environment variable OPENAI_API_KEY must be set.}
    ports:
      - "3000:3000"
    extra_hosts:
      - "host.docker.internal:host-gateway"
```

You can start the gateway with `docker compose up`.

## [​](https://www.tensorzero.com/docs/integrations/model-providers/openai\#inference)  Inference

Make an inference request to the gateway:

Copy

```
curl -X POST http://localhost:3000/inference \
  -H "Content-Type: application/json" \
  -d '{
    "function_name": "my_function_name",
    "input": {
      "messages": [\
        {\
          "role": "user",\
          "content": "What is the capital of Japan?"\
        }\
      ]
    }
  }'
```

## [​](https://www.tensorzero.com/docs/integrations/model-providers/openai\#other-features)  Other Features

### [​](https://www.tensorzero.com/docs/integrations/model-providers/openai\#generate-embeddings)  Generate embeddings

The OpenAI model provider supports generating embeddings.
You can find a [complete code example on GitHub](https://github.com/tensorzero/tensorzero/tree/main/examples/guides/embeddings/providers/openai).

[OpenAI-Compatible](https://www.tensorzero.com/docs/integrations/model-providers/openai-compatible) [OpenRouter](https://www.tensorzero.com/docs/integrations/model-providers/openrouter)

Ctrl+I

Assistant

Responses are generated using AI and may contain mistakes.

## OpenAI-Compatible Setup
[Skip to main content](https://www.tensorzero.com/docs/integrations/model-providers/openai-compatible#content-area)

[TensorZero Docs home page![light logo](https://mintcdn.com/tensorzero/afFfERJmQ8YiZpCj/logo/light.svg?fit=max&auto=format&n=afFfERJmQ8YiZpCj&q=85&s=9396731a8a92cee7bd293dee05d71dac)![dark logo](https://mintcdn.com/tensorzero/afFfERJmQ8YiZpCj/logo/dark.svg?fit=max&auto=format&n=afFfERJmQ8YiZpCj&q=85&s=8d61c59531dbcdcace7fcdfc524bd4d2)](https://www.tensorzero.com/)

Search...

Ctrl K

Search...

Navigation

Model Providers

Getting Started with OpenAI-Compatible Endpoints (e.g. Ollama)

[Guides](https://www.tensorzero.com/docs) [Integrations](https://www.tensorzero.com/docs/integrations/model-providers)

On this page

- [Setup](https://www.tensorzero.com/docs/integrations/model-providers/openai-compatible#setup)
- [Configuration](https://www.tensorzero.com/docs/integrations/model-providers/openai-compatible#configuration)
- [Credentials](https://www.tensorzero.com/docs/integrations/model-providers/openai-compatible#credentials)
- [Deployment (Docker Compose)](https://www.tensorzero.com/docs/integrations/model-providers/openai-compatible#deployment-docker-compose)
- [Inference](https://www.tensorzero.com/docs/integrations/model-providers/openai-compatible#inference)
- [Other Features](https://www.tensorzero.com/docs/integrations/model-providers/openai-compatible#other-features)
- [Generate embeddings](https://www.tensorzero.com/docs/integrations/model-providers/openai-compatible#generate-embeddings)

This guide shows how to set up a minimal deployment to use the TensorZero Gateway with OpenAI-compatible endpoints like Ollama.

## [​](https://www.tensorzero.com/docs/integrations/model-providers/openai-compatible\#setup)  Setup

This guide assumes that you are running Ollama locally with `ollama serve` and that you’ve pulled the `llama3.1` model in advance (e.g. `ollama pull llama3.1`).
Make sure to update the `api_base` and `model_name` in the configuration below to match your OpenAI-compatible endpoint and model.For this minimal setup, you’ll need just two files in your project directory:

Copy

```
- config/
  - tensorzero.toml
- docker-compose.yml
```

You can also find the complete code for this example on [GitHub](https://github.com/tensorzero/tensorzero/tree/main/examples/guides/providers/openai-compatible).

For production deployments, see our [Deployment Guide](https://www.tensorzero.com/docs/deployment/tensorzero-gateway).

### [​](https://www.tensorzero.com/docs/integrations/model-providers/openai-compatible\#configuration)  Configuration

Create a minimal configuration file that defines a model and a simple chat function:

config/tensorzero.toml

Copy

```
[models.llama3_1_8b_instruct]
routing = ["ollama"]

[models.llama3_1_8b_instruct.providers.ollama]
type = "openai"
api_base = "http://host.docker.internal:11434/v1"  # for Ollama running locally on the host
model_name = "llama3.1"
api_key_location = "none"  # by default, Ollama requires no API key

[functions.my_function_name]
type = "chat"

[functions.my_function_name.variants.my_variant_name]
type = "chat_completion"
model = "llama3_1_8b_instruct"
```

### [​](https://www.tensorzero.com/docs/integrations/model-providers/openai-compatible\#credentials)  Credentials

The `api_key_location` field in your model provider configuration specifies how to handle API key authentication:

- If your endpoint does not require an API key (e.g. Ollama by default):






Copy











```
api_key_location = "none"
```

- If your endpoint requires an API key, you have two options:  1. Configure it in advance through an environment variable:






       Copy











       ```
       api_key_location = "env::ENVIRONMENT_VARIABLE_NAME"
       ```






       You’ll need to set the environment variable before starting the gateway.
2. Provide it at inference time:






     Copy











     ```
     api_key_location = "dynamic::ARGUMENT_NAME"
     ```






     The API key can then be passed in the inference request.

See the [Credential Management](https://www.tensorzero.com/docs/operations/manage-credentials) guide, the [Configuration Reference](https://www.tensorzero.com/docs/gateway/configuration-reference), and the [API reference](https://www.tensorzero.com/docs/gateway/api-reference/inference-openai-compatible) for more details.In this example, Ollama is running locally without authentication, so we use `api_key_location = "none"`.

### [​](https://www.tensorzero.com/docs/integrations/model-providers/openai-compatible\#deployment-docker-compose)  Deployment (Docker Compose)

Create a minimal Docker Compose configuration:

docker-compose.yml

Copy

```
# This is a simplified example for learning purposes. Do not use this in production.
# For production-ready deployments, see: https://www.tensorzero.com/docs/deployment/tensorzero-gateway

services:
  gateway:
    image: tensorzero/gateway
    volumes:
      - ./config:/app/config:ro
    command: --config-file /app/config/tensorzero.toml
    # environment:
    # - OLLAMA_API_KEY=${OLLAMA_API_KEY:?Environment variable OLLAMA_API_KEY must be set.} // not necessary for this example
    ports:
      - "3000:3000"
    extra_hosts:
      - "host.docker.internal:host-gateway"
```

You can start the gateway with `docker compose up`.

## [​](https://www.tensorzero.com/docs/integrations/model-providers/openai-compatible\#inference)  Inference

Make an inference request to the gateway:

Copy

```
curl -X POST http://localhost:3000/inference \
  -H "Content-Type: application/json" \
  -d '{
    "function_name": "my_function_name",
    "input": {
      "messages": [\
        {\
          "role": "user",\
          "content": "What is the capital of Japan?"\
        }\
      ]
    }
  }'
```

## [​](https://www.tensorzero.com/docs/integrations/model-providers/openai-compatible\#other-features)  Other Features

### [​](https://www.tensorzero.com/docs/integrations/model-providers/openai-compatible\#generate-embeddings)  Generate embeddings

The OpenAI model provider supports generating embeddings.
You can find a [complete code example using Ollama on GitHub](https://github.com/tensorzero/tensorzero/tree/main/examples/guides/embeddings/providers/openai-compatible-ollama).

[Mistral](https://www.tensorzero.com/docs/integrations/model-providers/mistral) [OpenAI](https://www.tensorzero.com/docs/integrations/model-providers/openai)

Ctrl+I

Assistant

Responses are generated using AI and may contain mistakes.

## OpenRouter Integration Guide
[Skip to main content](https://www.tensorzero.com/docs/integrations/model-providers/openrouter#content-area)

[TensorZero Docs home page![light logo](https://mintcdn.com/tensorzero/afFfERJmQ8YiZpCj/logo/light.svg?fit=max&auto=format&n=afFfERJmQ8YiZpCj&q=85&s=9396731a8a92cee7bd293dee05d71dac)![dark logo](https://mintcdn.com/tensorzero/afFfERJmQ8YiZpCj/logo/dark.svg?fit=max&auto=format&n=afFfERJmQ8YiZpCj&q=85&s=8d61c59531dbcdcace7fcdfc524bd4d2)](https://www.tensorzero.com/)

Search...

Ctrl K

Search...

Navigation

Model Providers

Getting Started with OpenRouter

[Guides](https://www.tensorzero.com/docs) [Integrations](https://www.tensorzero.com/docs/integrations/model-providers)

On this page

- [Simple Setup](https://www.tensorzero.com/docs/integrations/model-providers/openrouter#simple-setup)
- [Advanced Setup](https://www.tensorzero.com/docs/integrations/model-providers/openrouter#advanced-setup)
- [Configuration](https://www.tensorzero.com/docs/integrations/model-providers/openrouter#configuration)
- [Credentials](https://www.tensorzero.com/docs/integrations/model-providers/openrouter#credentials)
- [Deployment (Docker Compose)](https://www.tensorzero.com/docs/integrations/model-providers/openrouter#deployment-docker-compose)
- [Inference](https://www.tensorzero.com/docs/integrations/model-providers/openrouter#inference)
- [Other Features](https://www.tensorzero.com/docs/integrations/model-providers/openrouter#other-features)
- [Generate embeddings](https://www.tensorzero.com/docs/integrations/model-providers/openrouter#generate-embeddings)

This guide shows how to set up a minimal deployment to use the TensorZero Gateway with OpenRouter.

## [​](https://www.tensorzero.com/docs/integrations/model-providers/openrouter\#simple-setup)  Simple Setup

You can use the short-hand `openrouter::model_name` to use an OpenRouter model with TensorZero, unless you need advanced features like fallbacks or custom credentials.You can use OpenRouter models in your TensorZero variants by setting the `model` field to `openrouter::model_name`.
For example:

Copy

```
[functions.my_function_name.variants.my_variant_name]
type = "chat_completion"
model = "openrouter::openai/gpt-4.1-mini"
```

Additionally, you can set `model_name` in the inference request to use a specific OpenRouter model, without having to configure a function and variant in TensorZero.

Copy

```
curl -X POST http://localhost:3000/inference \
  -H "Content-Type: application/json" \
  -d '{
    "model_name": "openrouter::openai/gpt-4.1-mini",
    "input": {
      "messages": [\
        {\
          "role": "user",\
          "content": "What is the capital of Japan?"\
        }\
      ]
    }
  }'
```

## [​](https://www.tensorzero.com/docs/integrations/model-providers/openrouter\#advanced-setup)  Advanced Setup

In more complex scenarios (e.g. fallbacks, custom credentials), you can configure your own model and OpenRouter provider in TensorZero.For this minimal setup, you’ll need just two files in your project directory:

Copy

```
- config/
  - tensorzero.toml
- docker-compose.yml
```

You can also find the complete code for this example on [GitHub](https://github.com/tensorzero/tensorzero/tree/main/examples/guides/providers/openrouter).

For production deployments, see our [Deployment Guide](https://www.tensorzero.com/docs/deployment/tensorzero-gateway).

### [​](https://www.tensorzero.com/docs/integrations/model-providers/openrouter\#configuration)  Configuration

Create a minimal configuration file that defines a model and a simple chat function:

config/tensorzero.toml

Copy

```
[models.gpt_4_1_mini]
routing = ["openrouter"]

[models.gpt_4_1_mini.providers.openrouter]
type = "openrouter"
model_name = "openai/gpt-4.1-mini"

[functions.my_function_name]
type = "chat"

[functions.my_function_name.variants.my_variant_name]
type = "chat_completion"
model = "gpt_4_1_mini"
```

See the [list of models available on OpenRouter](https://openrouter.ai/models).

### [​](https://www.tensorzero.com/docs/integrations/model-providers/openrouter\#credentials)  Credentials

You must set the `OPENROUTER_API_KEY` environment variable before running the gateway.You can customize the credential location by setting the `api_key_location` to `env::YOUR_ENVIRONMENT_VARIABLE` or `dynamic::ARGUMENT_NAME`.
See the [Credential Management](https://www.tensorzero.com/docs/operations/manage-credentials) guide and [Configuration Reference](https://www.tensorzero.com/docs/gateway/configuration-reference) for more information.

### [​](https://www.tensorzero.com/docs/integrations/model-providers/openrouter\#deployment-docker-compose)  Deployment (Docker Compose)

Create a minimal Docker Compose configuration:

docker-compose.yml

Copy

```
# This is a simplified example for learning purposes. Do not use this in production.
# For production-ready deployments, see: https://www.tensorzero.com/docs/deployment/tensorzero-gateway

services:
  gateway:
    image: tensorzero/gateway
    volumes:
      - ./config:/app/config:ro
    command: --config-file /app/config/tensorzero.toml
    environment:
      - OPENROUTER_API_KEY=${OPENROUTER_API_KEY:?Environment variable OPENROUTER_API_KEY must be set.}
    ports:
      - "3000:3000"
    extra_hosts:
      - "host.docker.internal:host-gateway"
```

You can start the gateway with `docker compose up`.

## [​](https://www.tensorzero.com/docs/integrations/model-providers/openrouter\#inference)  Inference

Make an inference request to the gateway:

Copy

```
curl -X POST http://localhost:3000/inference \
  -H "Content-Type: application/json" \
  -d '{
    "function_name": "my_function_name",
    "input": {
      "messages": [\
        {\
          "role": "user",\
          "content": "What is the capital of Japan?"\
        }\
      ]
    }
  }'
```

## [​](https://www.tensorzero.com/docs/integrations/model-providers/openrouter\#other-features)  Other Features

### [​](https://www.tensorzero.com/docs/integrations/model-providers/openrouter\#generate-embeddings)  Generate embeddings

The OpenRouter model provider supports generating embeddings.
You can find a [complete code example on GitHub](https://github.com/tensorzero/tensorzero/tree/main/examples/guides/embeddings/providers/openai).

[OpenAI](https://www.tensorzero.com/docs/integrations/model-providers/openai) [SGLang](https://www.tensorzero.com/docs/integrations/model-providers/sglang)

Ctrl+I

Assistant

Responses are generated using AI and may contain mistakes.

## SGLang Setup Guide
[Skip to main content](https://www.tensorzero.com/docs/integrations/model-providers/sglang#content-area)

[TensorZero Docs home page![light logo](https://mintcdn.com/tensorzero/afFfERJmQ8YiZpCj/logo/light.svg?fit=max&auto=format&n=afFfERJmQ8YiZpCj&q=85&s=9396731a8a92cee7bd293dee05d71dac)![dark logo](https://mintcdn.com/tensorzero/afFfERJmQ8YiZpCj/logo/dark.svg?fit=max&auto=format&n=afFfERJmQ8YiZpCj&q=85&s=8d61c59531dbcdcace7fcdfc524bd4d2)](https://www.tensorzero.com/)

Search...

Ctrl K

Search...

Navigation

Model Providers

Getting Started with SGLang

[Guides](https://www.tensorzero.com/docs) [Integrations](https://www.tensorzero.com/docs/integrations/model-providers)

On this page

- [Setup](https://www.tensorzero.com/docs/integrations/model-providers/sglang#setup)
- [Configuration](https://www.tensorzero.com/docs/integrations/model-providers/sglang#configuration)
- [Credentials](https://www.tensorzero.com/docs/integrations/model-providers/sglang#credentials)
- [Deployment (Docker Compose)](https://www.tensorzero.com/docs/integrations/model-providers/sglang#deployment-docker-compose)
- [Inference](https://www.tensorzero.com/docs/integrations/model-providers/sglang#inference)

This guide shows how to set up a minimal deployment to use the TensorZero Gateway with self-hosted LLMs using SGLang.We’re using Llama-3.1-8B-Instruct in this example, but you can use virtually any model supported by SGLang.

## [​](https://www.tensorzero.com/docs/integrations/model-providers/sglang\#setup)  Setup

This guide assumes that you are running SGLang locally with this command (see [SGLang’s installation guide](https://docs.sglang.ai/get_started/install.html)):

Run SGLang locally

Copy

```
docker run --gpus all \
    # Set shared memory size - needed for loading large models and processing requests
    --shm-size 32g \
    -p 30000:30000 \
    # Mount the host's ~/.cache/huggingface directory to the container's /root/.cache/huggingface directory
    -v ~/.cache/huggingface:/root/.cache/huggingface \
    --ipc=host \
    lmsysorg/sglang:latest \
    python3 -m sglang.launch_server --model-path meta-llama/Llama-3.1-8B-Instruct --host 0.0.0.0 --port 30000
```

Make sure to update the `api_base` in the configuration below to match your SGLang server.For this minimal setup, you’ll need just two files in your project directory:

Copy

```
- config/
  - tensorzero.toml
- docker-compose.yml
```

You can also find the complete code for this example on [GitHub](https://github.com/tensorzero/tensorzero/tree/main/examples/guides/providers/sglang).

For production deployments, see our [Deployment Guide](https://www.tensorzero.com/docs/deployment/tensorzero-gateway).

### [​](https://www.tensorzero.com/docs/integrations/model-providers/sglang\#configuration)  Configuration

Create a minimal configuration file that defines a model and a simple chat function:

config/tensorzero.toml

Copy

```
[models.llama]
routing = ["sglang"]

[models.llama.providers.sglang]
type = "sglang"
api_base = "http://host.docker.internal:8080/v1/"  # for SGLang running locally on the host
api_key_location = "none"  # by default, SGLang requires no API key
model_name = "my-sglang-model"

[functions.my_function_name]
type = "chat"

[functions.my_function_name.variants.my_variant_name]
type = "chat_completion"
model = "llama"
```

### [​](https://www.tensorzero.com/docs/integrations/model-providers/sglang\#credentials)  Credentials

The `api_key_location` field in your model provider configuration specifies how to handle API key authentication:

- If your endpoint does not require an API key (e.g. SGLang by default):






Copy











```
api_key_location = "none"
```

- If your endpoint requires an API key, you have two options:  1. Configure it in advance through an environment variable:






       Copy











       ```
       api_key_location = "env::ENVIRONMENT_VARIABLE_NAME"
       ```






       You’ll need to set the environment variable before starting the gateway.
2. Provide it at inference time:






     Copy











     ```
     api_key_location = "dynamic::ARGUMENT_NAME"
     ```






     The API key can then be passed in the inference request.

See the [Credential Management](https://www.tensorzero.com/docs/operations/manage-credentials) guide, the [Configuration Reference](https://www.tensorzero.com/docs/gateway/configuration-reference), and the [API reference](https://www.tensorzero.com/docs/gateway/api-reference/inference) for more details.In this example, SGLang is running locally without authentication, so we use `api_key_location = "none"`.

### [​](https://www.tensorzero.com/docs/integrations/model-providers/sglang\#deployment-docker-compose)  Deployment (Docker Compose)

Create a minimal Docker Compose configuration:

docker-compose.yml

Copy

```
# This is a simplified example for learning purposes. Do not use this in production.
# For production-ready deployments, see: https://www.tensorzero.com/docs/deployment/tensorzero-gateway

services:
  gateway:
    image: tensorzero/gateway
    volumes:
      - ./config:/app/config:ro
    command: --config-file /app/config/tensorzero.toml
    # environment:
    #   - SGLANG_API_KEY=${SGLANG_API_KEY:?Environment variable SGLANG_API_KEY must be set.}
    ports:
      - "3000:3000"
    extra_hosts:
      - "host.docker.internal:host-gateway"
```

You can start the gateway with `docker compose up`.

## [​](https://www.tensorzero.com/docs/integrations/model-providers/sglang\#inference)  Inference

Make an inference request to the gateway:

Copy

```
curl -X POST http://localhost:3000/inference \
  -H "Content-Type: application/json" \
  -d '{
    "function_name": "my_function_name",
    "input": {
      "messages": [\
        {\
          "role": "user",\
          "content": "What is the capital of Japan?"\
        }\
      ]
    }
  }'
```

[OpenRouter](https://www.tensorzero.com/docs/integrations/model-providers/openrouter) [TGI](https://www.tensorzero.com/docs/integrations/model-providers/tgi)

Ctrl+I

Assistant

Responses are generated using AI and may contain mistakes.

## TensorZero TGI Setup
[Skip to main content](https://www.tensorzero.com/docs/integrations/model-providers/tgi#content-area)

[TensorZero Docs home page![light logo](https://mintcdn.com/tensorzero/afFfERJmQ8YiZpCj/logo/light.svg?fit=max&auto=format&n=afFfERJmQ8YiZpCj&q=85&s=9396731a8a92cee7bd293dee05d71dac)![dark logo](https://mintcdn.com/tensorzero/afFfERJmQ8YiZpCj/logo/dark.svg?fit=max&auto=format&n=afFfERJmQ8YiZpCj&q=85&s=8d61c59531dbcdcace7fcdfc524bd4d2)](https://www.tensorzero.com/)

Search...

Ctrl K

Search...

Navigation

Model Providers

Getting Started with Text Generation Inference (TGI)

[Guides](https://www.tensorzero.com/docs) [Integrations](https://www.tensorzero.com/docs/integrations/model-providers)

On this page

- [Setup](https://www.tensorzero.com/docs/integrations/model-providers/tgi#setup)
- [Configuration](https://www.tensorzero.com/docs/integrations/model-providers/tgi#configuration)
- [Credentials](https://www.tensorzero.com/docs/integrations/model-providers/tgi#credentials)
- [Deployment (Docker Compose)](https://www.tensorzero.com/docs/integrations/model-providers/tgi#deployment-docker-compose)
- [Inference](https://www.tensorzero.com/docs/integrations/model-providers/tgi#inference)

This guide shows how to set up a minimal deployment to use the TensorZero Gateway with self-hosted LLMs using Text Generation Inference (TGI).We’re using Phi-4 in this example, but you can use virtually any model supported by TGI.

## [​](https://www.tensorzero.com/docs/integrations/model-providers/tgi\#setup)  Setup

This guide assumes that you are running TGI locally with

Run TGI locally

Copy

```
docker run \
    --gpus all \
    # Set shared memory size - needed for loading large models and processing requests
    --shm-size 64g \
    # Map the host's port 8080 to the container's port 80
    -p 8080:80 \
    # Mount the host's './data' directory to the container's '/data' directory
    -v $PWD/data:/data \
    ghcr.io/huggingface/text-generation-inference:3.0.1 \
    --model-id microsoft/phi-4
```

Make sure to update the `api_base` in the configuration below to match your TGI server.For this minimal setup, you’ll need just two files in your project directory:

Copy

```
- config/
  - tensorzero.toml
- docker-compose.yml
```

You can also find the complete code for this example on [GitHub](https://github.com/tensorzero/tensorzero/tree/main/examples/guides/providers/tgi).

For production deployments, see our [Deployment Guide](https://www.tensorzero.com/docs/deployment/tensorzero-gateway).

### [​](https://www.tensorzero.com/docs/integrations/model-providers/tgi\#configuration)  Configuration

Create a minimal configuration file that defines a model and a simple chat function:

config/tensorzero.toml

Copy

```
[models.phi_4]
routing = ["tgi"]

[models.phi_4.providers.tgi]
type = "tgi"
api_base = "http://host.docker.internal:8080/v1/"  # for TGI running locally on the host
api_key_location = "none"  # by default, TGI requires no API key

[functions.my_function_name]
type = "chat"

[functions.my_function_name.variants.my_variant_name]
type = "chat_completion"
model = "phi_4"
```

### [​](https://www.tensorzero.com/docs/integrations/model-providers/tgi\#credentials)  Credentials

The `api_key_location` field in your model provider configuration specifies how to handle API key authentication:

- If your endpoint does not require an API key (e.g. TGI by default):






Copy











```
api_key_location = "none"
```

- If your endpoint requires an API key, you have two options:  1. Configure it in advance through an environment variable:






       Copy











       ```
       api_key_location = "env::ENVIRONMENT_VARIABLE_NAME"
       ```






       You’ll need to set the environment variable before starting the gateway.
2. Provide it at inference time:






     Copy











     ```
     api_key_location = "dynamic::ARGUMENT_NAME"
     ```






     The API key can then be passed in the inference request.

See the [Configuration Reference](https://www.tensorzero.com/docs/gateway/configuration-reference) and the [API reference](https://www.tensorzero.com/docs/gateway/api-reference/inference) for more details.In this example, TGI is running locally without authentication, so we use `api_key_location = "none"`.

### [​](https://www.tensorzero.com/docs/integrations/model-providers/tgi\#deployment-docker-compose)  Deployment (Docker Compose)

Create a minimal Docker Compose configuration:

docker-compose.yml

Copy

```
# This is a simplified example for learning purposes. Do not use this in production.
# For production-ready deployments, see: https://www.tensorzero.com/docs/deployment/tensorzero-gateway

services:
  gateway:
    image: tensorzero/gateway
    volumes:
      - ./config:/app/config:ro
    command: --config-file /app/config/tensorzero.toml
    # environment:
    #   - TGI_API_KEY=${TGI_API_KEY:?Environment variable TGI_API_KEY must be set.}
    ports:
      - "3000:3000"
    extra_hosts:
      - "host.docker.internal:host-gateway"
```

You can start the gateway with `docker compose up`.

## [​](https://www.tensorzero.com/docs/integrations/model-providers/tgi\#inference)  Inference

Make an inference request to the gateway:

Copy

```
curl -X POST http://localhost:3000/inference \
  -H "Content-Type: application/json" \
  -d '{
    "function_name": "my_function_name",
    "input": {
      "messages": [\
        {\
          "role": "user",\
          "content": "What is the capital of Japan?"\
        }\
      ]
    }
  }'
```

[SGLang](https://www.tensorzero.com/docs/integrations/model-providers/sglang) [Together](https://www.tensorzero.com/docs/integrations/model-providers/together)

Ctrl+I

Assistant

Responses are generated using AI and may contain mistakes.

## Together AI Setup Guide
[Skip to main content](https://www.tensorzero.com/docs/integrations/model-providers/together#content-area)

[TensorZero Docs home page![light logo](https://mintcdn.com/tensorzero/afFfERJmQ8YiZpCj/logo/light.svg?fit=max&auto=format&n=afFfERJmQ8YiZpCj&q=85&s=9396731a8a92cee7bd293dee05d71dac)![dark logo](https://mintcdn.com/tensorzero/afFfERJmQ8YiZpCj/logo/dark.svg?fit=max&auto=format&n=afFfERJmQ8YiZpCj&q=85&s=8d61c59531dbcdcace7fcdfc524bd4d2)](https://www.tensorzero.com/)

Search...

Ctrl K

Search...

Navigation

Model Providers

Getting Started with Together AI

[Guides](https://www.tensorzero.com/docs) [Integrations](https://www.tensorzero.com/docs/integrations/model-providers)

On this page

- [Simple Setup](https://www.tensorzero.com/docs/integrations/model-providers/together#simple-setup)
- [Advanced Setup](https://www.tensorzero.com/docs/integrations/model-providers/together#advanced-setup)
- [Configuration](https://www.tensorzero.com/docs/integrations/model-providers/together#configuration)
- [Credentials](https://www.tensorzero.com/docs/integrations/model-providers/together#credentials)
- [Deployment (Docker Compose)](https://www.tensorzero.com/docs/integrations/model-providers/together#deployment-docker-compose)
- [Inference](https://www.tensorzero.com/docs/integrations/model-providers/together#inference)

This guide shows how to set up a minimal deployment to use the TensorZero Gateway with the Together AI API.

## [​](https://www.tensorzero.com/docs/integrations/model-providers/together\#simple-setup)  Simple Setup

You can use the short-hand `together::model_name` to use a Together AI model with TensorZero, unless you need advanced features like fallbacks or custom credentials.You can use Together AI models in your TensorZero variants by setting the `model` field to `together::model_name`.
For example:

Copy

```
[functions.my_function_name.variants.my_variant_name]
type = "chat_completion"
model = "together::meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo"
```

Additionally, you can set `model_name` in the inference request to use a specific Together AI model, without having to configure a function and variant in TensorZero.

Copy

```
curl -X POST http://localhost:3000/inference \
  -H "Content-Type: application/json" \
  -d '{
    "model_name": "together::meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo",
    "input": {
      "messages": [\
        {\
          "role": "user",\
          "content": "What is the capital of Japan?"\
        }\
      ]
    }
  }'
```

## [​](https://www.tensorzero.com/docs/integrations/model-providers/together\#advanced-setup)  Advanced Setup

In more complex scenarios (e.g. fallbacks, custom credentials), you can configure your own model and Together AI provider in TensorZero.For this minimal setup, you’ll need just two files in your project directory:

Copy

```
- config/
  - tensorzero.toml
- docker-compose.yml
```

You can also find the complete code for this example on [GitHub](https://github.com/tensorzero/tensorzero/tree/main/examples/guides/providers/together).

For production deployments, see our [Deployment Guide](https://www.tensorzero.com/docs/deployment/tensorzero-gateway).

### [​](https://www.tensorzero.com/docs/integrations/model-providers/together\#configuration)  Configuration

Create a minimal configuration file that defines a model and a simple chat function:

config/tensorzero.toml

Copy

```
[models.llama3_1_8b_instruct_turbo]
routing = ["together"]

[models.llama3_1_8b_instruct_turbo.providers.together]
type = "together"
model_name = "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo"

[functions.my_function_name]
type = "chat"

[functions.my_function_name.variants.my_variant_name]
type = "chat_completion"
model = "llama3_1_8b_instruct_turbo"
```

See the [list of models available on Together AI](https://docs.together.ai/docs/serverless-models).
Dedicated endpoints and custom models are also supported.See the [Configuration Reference](https://www.tensorzero.com/docs/gateway/configuration-reference) for optional fields (e.g. overwriting `api_base`).

### [​](https://www.tensorzero.com/docs/integrations/model-providers/together\#credentials)  Credentials

You must set the `TOGETHER_API_KEY` environment variable before running the gateway.You can customize the credential location by setting the `api_key_location` to `env::YOUR_ENVIRONMENT_VARIABLE` or `dynamic::ARGUMENT_NAME`.
See the [Credential Management](https://www.tensorzero.com/docs/operations/manage-credentials) guide and [Configuration Reference](https://www.tensorzero.com/docs/gateway/configuration-reference) for more information.

### [​](https://www.tensorzero.com/docs/integrations/model-providers/together\#deployment-docker-compose)  Deployment (Docker Compose)

Create a minimal Docker Compose configuration:

docker-compose.yml

Copy

```
# This is a simplified example for learning purposes. Do not use this in production.
# For production-ready deployments, see: https://www.tensorzero.com/docs/deployment/tensorzero-gateway

services:
  gateway:
    image: tensorzero/gateway
    volumes:
      - ./config:/app/config:ro
    command: --config-file /app/config/tensorzero.toml
    environment:
      - TOGETHER_API_KEY=${TOGETHER_API_KEY:?Environment variable TOGETHER_API_KEY must be set.}
    ports:
      - "3000:3000"
    extra_hosts:
      - "host.docker.internal:host-gateway"
```

You can start the gateway with `docker compose up`.

## [​](https://www.tensorzero.com/docs/integrations/model-providers/together\#inference)  Inference

Make an inference request to the gateway:

Copy

```
curl -X POST http://localhost:3000/inference \
  -H "Content-Type: application/json" \
  -d '{
    "function_name": "my_function_name",
    "input": {
      "messages": [\
        {\
          "role": "user",\
          "content": "What is the capital of Japan?"\
        }\
      ]
    }
  }'
```

[TGI](https://www.tensorzero.com/docs/integrations/model-providers/tgi) [vLLM](https://www.tensorzero.com/docs/integrations/model-providers/vllm)

Ctrl+I

Assistant

Responses are generated using AI and may contain mistakes.

## vLLM Setup Guide
[Skip to main content](https://www.tensorzero.com/docs/integrations/model-providers/vllm#content-area)

[TensorZero Docs home page![light logo](https://mintcdn.com/tensorzero/afFfERJmQ8YiZpCj/logo/light.svg?fit=max&auto=format&n=afFfERJmQ8YiZpCj&q=85&s=9396731a8a92cee7bd293dee05d71dac)![dark logo](https://mintcdn.com/tensorzero/afFfERJmQ8YiZpCj/logo/dark.svg?fit=max&auto=format&n=afFfERJmQ8YiZpCj&q=85&s=8d61c59531dbcdcace7fcdfc524bd4d2)](https://www.tensorzero.com/)

Search...

Ctrl K

Search...

Navigation

Model Providers

Getting Started with vLLM

[Guides](https://www.tensorzero.com/docs) [Integrations](https://www.tensorzero.com/docs/integrations/model-providers)

On this page

- [Setup](https://www.tensorzero.com/docs/integrations/model-providers/vllm#setup)
- [Configuration](https://www.tensorzero.com/docs/integrations/model-providers/vllm#configuration)
- [Credentials](https://www.tensorzero.com/docs/integrations/model-providers/vllm#credentials)
- [Deployment (Docker Compose)](https://www.tensorzero.com/docs/integrations/model-providers/vllm#deployment-docker-compose)
- [Inference](https://www.tensorzero.com/docs/integrations/model-providers/vllm#inference)

This guide shows how to set up a minimal deployment to use the TensorZero Gateway with self-hosted LLMs using vLLM.We’re using Llama 3.1 in this example, but you can use virtually any model supported by vLLM.

## [​](https://www.tensorzero.com/docs/integrations/model-providers/vllm\#setup)  Setup

This guide assumes that you are running vLLM locally with `vllm serve meta-llama/Llama-3.1-8B-Instruct`.
Make sure to update the `api_base` and `model_name` in the configuration below to match your vLLM server and model.For this minimal setup, you’ll need just two files in your project directory:

Copy

```
- config/
  - tensorzero.toml
- docker-compose.yml
```

You can also find the complete code for this example on [GitHub](https://github.com/tensorzero/tensorzero/tree/main/examples/guides/providers/vllm).

For production deployments, see our [Deployment Guide](https://www.tensorzero.com/docs/deployment/tensorzero-gateway).

### [​](https://www.tensorzero.com/docs/integrations/model-providers/vllm\#configuration)  Configuration

Create a minimal configuration file that defines a model and a simple chat function:

config/tensorzero.toml

Copy

```
[models.llama3_1_8b_instruct]
routing = ["vllm"]

[models.llama3_1_8b_instruct.providers.vllm]
type = "vllm"
api_base = "http://host.docker.internal:8000/v1/"  # for vLLM running locally on the host
model_name = "meta-llama/Llama-3.1-8B-Instruct"
api_key_location = "none"  # by default, vLLM requires no API key

[functions.my_function_name]
type = "chat"

[functions.my_function_name.variants.my_variant_name]
type = "chat_completion"
model = "llama3_1_8b_instruct"
```

### [​](https://www.tensorzero.com/docs/integrations/model-providers/vllm\#credentials)  Credentials

The `api_key_location` field in your model provider configuration specifies how to handle API key authentication:

- If your endpoint does not require an API key (e.g. vLLM by default):






Copy











```
api_key_location = "none"
```

- If your endpoint requires an API key, you have two options:  1. Configure it in advance through an environment variable:






       Copy











       ```
       api_key_location = "env::ENVIRONMENT_VARIABLE_NAME"
       ```






       You’ll need to set the environment variable before starting the gateway.
2. Provide it at inference time:






     Copy











     ```
     api_key_location = "dynamic::ARGUMENT_NAME"
     ```






     The API key can then be passed in the inference request.

See the [Credential Management](https://www.tensorzero.com/docs/operations/manage-credentials) guide, the [Configuration Reference](https://www.tensorzero.com/docs/gateway/configuration-reference), and the [API reference](https://www.tensorzero.com/docs/gateway/api-reference/inference) for more details.In this example, vLLM is running locally without authentication, so we use `api_key_location = "none"`.

### [​](https://www.tensorzero.com/docs/integrations/model-providers/vllm\#deployment-docker-compose)  Deployment (Docker Compose)

Create a minimal Docker Compose configuration:

docker-compose.yml

Copy

```
# This is a simplified example for learning purposes. Do not use this in production.
# For production-ready deployments, see: https://www.tensorzero.com/docs/deployment/tensorzero-gateway

services:
  gateway:
    image: tensorzero/gateway
    volumes:
      - ./config:/app/config:ro
    command: --config-file /app/config/tensorzero.toml
    # environment:
    #   - VLLM_API_KEY=${VLLM_API_KEY:?Environment variable VLLM_API_KEY must be set.}
    ports:
      - "3000:3000"
    extra_hosts:
      - "host.docker.internal:host-gateway"
```

You can start the gateway with `docker compose up`.

## [​](https://www.tensorzero.com/docs/integrations/model-providers/vllm\#inference)  Inference

Make an inference request to the gateway:

Copy

```
curl -X POST http://localhost:3000/inference \
  -H "Content-Type: application/json" \
  -d '{
    "function_name": "my_function_name",
    "input": {
      "messages": [\
        {\
          "role": "user",\
          "content": "What is the capital of Japan?"\
        }\
      ]
    }
  }'
```

[Together](https://www.tensorzero.com/docs/integrations/model-providers/together) [xAI](https://www.tensorzero.com/docs/integrations/model-providers/xai)

Ctrl+I

Assistant

Responses are generated using AI and may contain mistakes.

## TensorZero xAI Integration
[Skip to main content](https://www.tensorzero.com/docs/integrations/model-providers/xai#content-area)

[TensorZero Docs home page![light logo](https://mintcdn.com/tensorzero/afFfERJmQ8YiZpCj/logo/light.svg?fit=max&auto=format&n=afFfERJmQ8YiZpCj&q=85&s=9396731a8a92cee7bd293dee05d71dac)![dark logo](https://mintcdn.com/tensorzero/afFfERJmQ8YiZpCj/logo/dark.svg?fit=max&auto=format&n=afFfERJmQ8YiZpCj&q=85&s=8d61c59531dbcdcace7fcdfc524bd4d2)](https://www.tensorzero.com/)

Search...

Ctrl K

Search...

Navigation

Model Providers

Getting Started with xAI (Grok)

[Guides](https://www.tensorzero.com/docs) [Integrations](https://www.tensorzero.com/docs/integrations/model-providers)

On this page

- [Simple Setup](https://www.tensorzero.com/docs/integrations/model-providers/xai#simple-setup)
- [Advanced Setup](https://www.tensorzero.com/docs/integrations/model-providers/xai#advanced-setup)
- [Configuration](https://www.tensorzero.com/docs/integrations/model-providers/xai#configuration)
- [Credentials](https://www.tensorzero.com/docs/integrations/model-providers/xai#credentials)
- [Deployment (Docker Compose)](https://www.tensorzero.com/docs/integrations/model-providers/xai#deployment-docker-compose)
- [Inference](https://www.tensorzero.com/docs/integrations/model-providers/xai#inference)

This guide shows how to set up a minimal deployment to use the TensorZero Gateway with the xAI API.

## [​](https://www.tensorzero.com/docs/integrations/model-providers/xai\#simple-setup)  Simple Setup

You can use the short-hand `xai::model_name` to use an xAI model with TensorZero, unless you need advanced features like fallbacks or custom credentials.You can use xAI models in your TensorZero variants by setting the `model` field to `xai::model_name`.
For example:

Copy

```
[functions.my_function_name.variants.my_variant_name]
type = "chat_completion"
model = "xai::grok-2-1212"
```

Additionally, you can set `model_name` in the inference request to use a specific xAI model, without having to configure a function and variant in TensorZero.

Copy

```
curl -X POST http://localhost:3000/inference \
  -H "Content-Type: application/json" \
  -d '{
    "model_name": "xai::grok-2-1212",
    "input": {
      "messages": [\
        {\
          "role": "user",\
          "content": "What is the capital of Japan?"\
        }\
      ]
    }
  }'
```

## [​](https://www.tensorzero.com/docs/integrations/model-providers/xai\#advanced-setup)  Advanced Setup

In more complex scenarios (e.g. fallbacks, custom credentials), you can configure your own model and xAI provider in TensorZero.For this minimal setup, you’ll need just two files in your project directory:

Copy

```
- config/
  - tensorzero.toml
- docker-compose.yml
```

You can also find the complete code for this example on [GitHub](https://github.com/tensorzero/tensorzero/tree/main/examples/guides/providers/xai).

For production deployments, see our [Deployment Guide](https://www.tensorzero.com/docs/deployment/tensorzero-gateway).

### [​](https://www.tensorzero.com/docs/integrations/model-providers/xai\#configuration)  Configuration

Create a minimal configuration file that defines a model and a simple chat function:

config/tensorzero.toml

Copy

```
[models.grok_2_1212]
routing = ["xai"]

[models.grok_2_1212.providers.xai]
type = "xai"
model_name = "grok-2-1212"

[functions.my_function_name]
type = "chat"

[functions.my_function_name.variants.my_variant_name]
type = "chat_completion"
model = "grok_2_1212"
```

See the [list of models available on xAI](https://docs.x.ai/docs/models).

### [​](https://www.tensorzero.com/docs/integrations/model-providers/xai\#credentials)  Credentials

You must set the `XAI_API_KEY` environment variable before running the gateway.You can customize the credential location by setting the `api_key_location` to `env::YOUR_ENVIRONMENT_VARIABLE` or `dynamic::ARGUMENT_NAME`.
See the [Credential Management](https://www.tensorzero.com/docs/operations/manage-credentials) guide and [Configuration Reference](https://www.tensorzero.com/docs/gateway/configuration-reference) for more information.

### [​](https://www.tensorzero.com/docs/integrations/model-providers/xai\#deployment-docker-compose)  Deployment (Docker Compose)

Create a minimal Docker Compose configuration:

docker-compose.yml

Copy

```
# This is a simplified example for learning purposes. Do not use this in production.
# For production-ready deployments, see: https://www.tensorzero.com/docs/deployment/tensorzero-gateway

services:
  gateway:
    image: tensorzero/gateway
    volumes:
      - ./config:/app/config:ro
    command: --config-file /app/config/tensorzero.toml
    environment:
      - XAI_API_KEY=${XAI_API_KEY:?Environment variable XAI_API_KEY must be set.}
    ports:
      - "3000:3000"
    extra_hosts:
      - "host.docker.internal:host-gateway"
```

You can start the gateway with `docker compose up`.

## [​](https://www.tensorzero.com/docs/integrations/model-providers/xai\#inference)  Inference

Make an inference request to the gateway:

Copy

```
curl -X POST http://localhost:3000/inference \
  -H "Content-Type: application/json" \
  -d '{
    "function_name": "my_function_name",
    "input": {
      "messages": [\
        {\
          "role": "user",\
          "content": "What is the capital of Japan?"\
        }\
      ]
    }
  }'
```

[vLLM](https://www.tensorzero.com/docs/integrations/model-providers/vllm)

Ctrl+I

Assistant

Responses are generated using AI and may contain mistakes.

## Custom Rate Limits
[Skip to main content](https://www.tensorzero.com/docs/operations/enforce-custom-rate-limits#content-area)

[TensorZero Docs home page![light logo](https://mintcdn.com/tensorzero/afFfERJmQ8YiZpCj/logo/light.svg?fit=max&auto=format&n=afFfERJmQ8YiZpCj&q=85&s=9396731a8a92cee7bd293dee05d71dac)![dark logo](https://mintcdn.com/tensorzero/afFfERJmQ8YiZpCj/logo/dark.svg?fit=max&auto=format&n=afFfERJmQ8YiZpCj&q=85&s=8d61c59531dbcdcace7fcdfc524bd4d2)](https://www.tensorzero.com/)

Search...

Ctrl K

Search...

Navigation

Operations

Enforce custom rate limits

[Guides](https://www.tensorzero.com/docs) [Integrations](https://www.tensorzero.com/docs/integrations/model-providers)

On this page

- [Learn rate limiting concepts](https://www.tensorzero.com/docs/operations/enforce-custom-rate-limits#learn-rate-limiting-concepts)
- [Resources](https://www.tensorzero.com/docs/operations/enforce-custom-rate-limits#resources)
- [Scope](https://www.tensorzero.com/docs/operations/enforce-custom-rate-limits#scope)
- [By tags](https://www.tensorzero.com/docs/operations/enforce-custom-rate-limits#by-tags)
- [By API keys](https://www.tensorzero.com/docs/operations/enforce-custom-rate-limits#by-api-keys)
- [Priority](https://www.tensorzero.com/docs/operations/enforce-custom-rate-limits#priority)
- [Set up rate limits](https://www.tensorzero.com/docs/operations/enforce-custom-rate-limits#set-up-rate-limits)
- [Advanced](https://www.tensorzero.com/docs/operations/enforce-custom-rate-limits#advanced)
- [Customize capacity and refill rate](https://www.tensorzero.com/docs/operations/enforce-custom-rate-limits#customize-capacity-and-refill-rate)

The TensorZero Gateway supports granular custom rate limits to help you control usage and costs.Rate limit rules have three key components:

- **Resources:** Define what you’re limiting (like model inferences or tokens) and the time window (per second, hour, day, week, or month). For example, “1000 model inferences per day” or “500,000 tokens per hour”.
- **Priority:** Control which rules take precedence when multiple rules could apply to the same request. Higher priority numbers override lower ones.
- **Scope:** Determine which requests the rule applies to. You can set global limits for all requests, or targeted limits using custom tags like user IDs.

## [​](https://www.tensorzero.com/docs/operations/enforce-custom-rate-limits\#learn-rate-limiting-concepts)  Learn rate limiting concepts

Let’s start with a brief tutorial on the concepts behind custom rate limits in TensorZero.You can define custom rate limiting _rules_ in your TensorZero configuration using `[[rate_limiting.rules]]`.
Your configuration can have multiple rules.Rate limit state is stored in Postgres, so restarting the gateway preserves existing limits and multiple gateway instances automatically share the same limits.

Tracking begins when a rate limit rule is first applied to a request.
Requests made before a rule was configured do not count towards its limit.
Modifying a rate limit rule resets its usage.

### [​](https://www.tensorzero.com/docs/operations/enforce-custom-rate-limits\#resources)  Resources

Each rate limiting rule can have one or more _resource limits_.
A resource limit is defined using the `RESOURCE_per_WINDOW` syntax.
For example:

tensorzero.toml

Copy

```
[[rate_limiting.rules]]
# ...
model_inferences_per_day = 1_000
tokens_per_second = 1_000_000
# ...
```

Time windows are sequential and non-overlapping (i.e. not a sliding window).
They are aligned to when each rate limit bucket is first initialized (not sliding windows).
For example, if a rule with a `RESOURCE_per_minute` limit is first used at 10:30:15, it’ll be refilled at 10:31:15, 10:32:15, and so on.

You must specify `max_tokens` for a request if a token limit applies to it.
The gateway makes a reasonably conservative estimate of token usage and later records the actual usage.

### [​](https://www.tensorzero.com/docs/operations/enforce-custom-rate-limits\#scope)  Scope

Each rate limiting rule can optionally have a _scope_.
The scope restricts the rule to certain requests only.
If you don’t specify a scope, the rule will apply to all requests.You can scope rate limiting rules by tags or by API key public ID.

#### [​](https://www.tensorzero.com/docs/operations/enforce-custom-rate-limits\#by-tags)  By tags

You can scope rate limits using user-defined `tags`.
You can limit the scope to a specific value, to each individual value (`tensorzero::each`), or to every value collectively (`tensorzero::total`).For example, the following rule would only apply to inference requests with the tag `user_id` set to `intern`:

tensorzero.toml

Copy

```
[[rate_limiting.rules]]
# ...
scope = [\
    { tag_key = "user_id", tag_value = "intern" }\
]
#...
```

If a scope has multiple entries, all of them must be met for the rule to apply.For example, the following rule would only apply to inference requests with the tag `user_id` set to `intern` _and_ the tag `env` set to `production`:

tensorzero.toml

Copy

```
[[rate_limiting.rules]]
# ...
scope = [\
    { tag_key = "user_id", tag_value = "intern" },\
    { tag_key = "env", tag_value = "production" }\
]
#...
```

Entries based on `tags` support two special strings for `tag_value`:

- `tensorzero::each`: The rule independently applies to every `tag_key` value.
- `tensorzero::total`: The limits are summed across all values of the tag.

For example, the following rule would apply to each value of the `user_id` tag individually (i.e. each user gets their own limit):

tensorzero.toml

Copy

```
[[rate_limiting.rules]]
# ...
scope = [\
    { tag_key = "user_id", tag_value = "tensorzero::each" },\
]
#...
```

Conversely, the following rule would apply to all users collectively:

tensorzero.toml

Copy

```
[[rate_limiting.rules]]
# ...
scope = [\
    { tag_key = "user_id", tag_value = "tensorzero::total" },\
]
#...
```

The rule above won’t apply to requests that do not specify a `user_id` tag.

#### [​](https://www.tensorzero.com/docs/operations/enforce-custom-rate-limits\#by-api-keys)  By API keys

You can scope rate limits using API keys when authentication is enabled.
This allows you to enforce different rate limits for different API keys, which is useful for implementing tiered access or preventing individual keys from consuming too many resources.You can limit the scope to each individual API key (`tensorzero::each`) or to a specific API key by providing its 12-character public ID.For example, the following rule would apply to each API key individually (i.e. each API key gets its own limit):

tensorzero.toml

Copy

```
[[rate_limiting.rules]]
# ...
scope = [\
    { api_key_public_id = "tensorzero::each" },\
]
#...
```

You can also target a specific API key by providing its 12-character public ID:

tensorzero.toml

Copy

```
[[rate_limiting.rules]]
# ...
scope = [\
    { api_key_public_id = "xxxxxxxxxxxx" },\
]
#...
```

TensorZero API keys have the following format:`sk-t0-xxxxxxxxxxxx-yyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyy`The `xxxxxxxxxxxx` portion is the 12-character public ID that you can use in rate limiting rules.
The remaining portion of the key is secret and should be kept secure.

Unlike tag scopes, API key public ID scopes do not support `tensorzero::total`.
Only `tensorzero::each` and concrete 12-character public IDs are supported.

Rules with `api_key_public_id` scope won’t apply to unauthenticated requests.
Learn how to [set up auth for TensorZero](https://www.tensorzero.com/docs/operations/set-up-auth-for-tensorzero).

### [​](https://www.tensorzero.com/docs/operations/enforce-custom-rate-limits\#priority)  Priority

Each rate limiting rule must have a _priority_ (e.g. `priority = 1`).
The gateway iterates through the rules in order of priority, starting with the highest priority, until it finds a matching rate limit; once it does, it enforces all rules with that priority number and disregards any rules with lower priority.For example, the configuration below would enforce the first rule for requests with `user_id = "intern"` and the second rule for all other `user_id` values:

tensorzero.toml

Copy

```
[[rate_limiting.rules]]
# ...
scope = [\
    { tag_key = "user_id", tag_value = "intern" },\
]
priority = 1
#...

[[rate_limiting.rules]]
# ...
scope = [\
    { tag_key = "user_id", tag_value = "tensorzero::each" },\
]
priority = 0
#...
```

Alternatively, you can set `always = true` to enforce the rule regardless of other rules; rules with `always = true` do not affect the priority calculation above.

## [​](https://www.tensorzero.com/docs/operations/enforce-custom-rate-limits\#set-up-rate-limits)  Set up rate limits

Let’s set up rate limits for an application to restrict usage depending on an user-defined tag for user IDs.

You can find a [complete runnable example](https://github.com/tensorzero/tensorzero/tree/main/examples/docs/guides/operations/enforce-custom-rate-limits) of this guide on GitHub.

1

Set up Postgres

You must set up Postgres to use TensorZero’s rate limiting features.See the [Deploy Postgres](https://www.tensorzero.com/docs/deployment/postgres) guide for instructions.

2

Configure rate limiting rules

Add to your TensorZero configuration:

config/tensorzero.toml

Copy

```
# [A] Collectively, all users can make a maximum of 1k model inferences per hour and 10M tokens per day
[[rate_limiting.rules]]
always = true
model_inferences_per_hour = 1_000
tokens_per_day = 10_000_000
scope = [\
    { tag_key = "user_id", tag_value = "tensorzero::total" }\
]

# [B] Each individual user can make a maximum of 1 model inference per minute
[[rate_limiting.rules]]
priority = 0
model_inferences_per_minute = 1
scope = [\
    { tag_key = "user_id", tag_value = "tensorzero::each" }\
]

# [C] But override the individual limit for the CEO
[[rate_limiting.rules]]
priority = 1
model_inferences_per_minute = 5
scope = [\
    { tag_key = "user_id", tag_value = "ceo" }\
]

# [D] The entire system (i.e. without restricting the scope) can make a maximum of 10M tokens per hour
[[rate_limiting.rules]]
always = true
tokens_per_hour = 10_000_000
```

Make sure to reload your gateway.

3

Make inference requests

If we make two consecutive inference requests with `user_id = "intern"`, the second one should fail because of rule `[B]`.
However, if we make two consecutive inference requests with `user_id = "ceo"`, both should succeed because rule `[C]` will override rule `[B]`.

- Python (TensorZero SDK)

- Python (OpenAI SDK)


Copy

```
from tensorzero import TensorZeroGateway

t0 = TensorZeroGateway.build_http(gateway_url="http://localhost:3000")

def call_llm(user_id):
    try:
        return t0.inference(
            model_name="openai::gpt-4.1-mini",
            input={
                "messages": [\
                    {\
                        "role": "user",\
                        "content": "Tell me a fun fact.",\
                    }\
                ]
            },
            # We have rate limits on tokens, so we must be conservative and provide `max_tokens`
            params={
                "chat_completion": {
                    "max_tokens": 1000,
                }
            },
            tags={
                "user_id": user_id,
            },
        )
    except Exception as e:
        print(f"Error calling LLM: {e}")

# The second should fail
print(call_llm("intern"))
print(call_llm("intern"))  # should return None

# Both should work
print(call_llm("ceo"))
print(call_llm("ceo"))
```

## [​](https://www.tensorzero.com/docs/operations/enforce-custom-rate-limits\#advanced)  Advanced

### [​](https://www.tensorzero.com/docs/operations/enforce-custom-rate-limits\#customize-capacity-and-refill-rate)  Customize capacity and refill rate

By default, rate limits use a simple bucket model where the entire capacity refills at the start of each time window.
For example, `tokens_per_minute = 100_000` allows 100,000 tokens every minute, with the full allowance resetting at the top of each minute.However, you can customize this behavior using the `capacity` and `refill_rate` parameters to create a token bucket that refills continuously:

Copy

```
[[rate_limiting.rules]]
# ...
tokens_per_minute = { capacity = 100_000, refill_rate = 10_000 }
# ...
```

In this example, the `capacity` parameter sets the maximum number of tokens that can be stored in the bucket, while the `refill_rate` determines how many tokens are added to the bucket per time window (10,000 per minute).
This creates smoother rate limiting behavior where instead of getting your full allowance at the start of each minute: you get 10,000 tokens added every minute, up to a maximum of 100,000 tokens stored at any time.
To achieve these benefits, you’ll typically want to use a low time granularity with a `capacity` much larger than the `refill_rate`.This approach is particularly useful for burst protection (users can’t consume their entire daily allowance in the first few seconds), smoother traffic distribution (requests are naturally spread out over time rather than clustering at window boundaries), and a better user experience (users get a steady trickle of quota rather than having to wait for the next time window).

[Set up auth for TensorZero](https://www.tensorzero.com/docs/operations/set-up-auth-for-tensorzero) [Organize your configuration](https://www.tensorzero.com/docs/operations/organize-your-configuration)

Ctrl+I

Assistant

Responses are generated using AI and may contain mistakes.

## Export OpenTelemetry Traces
[Skip to main content](https://www.tensorzero.com/docs/operations/export-opentelemetry-traces#content-area)

[TensorZero Docs home page![light logo](https://mintcdn.com/tensorzero/afFfERJmQ8YiZpCj/logo/light.svg?fit=max&auto=format&n=afFfERJmQ8YiZpCj&q=85&s=9396731a8a92cee7bd293dee05d71dac)![dark logo](https://mintcdn.com/tensorzero/afFfERJmQ8YiZpCj/logo/dark.svg?fit=max&auto=format&n=afFfERJmQ8YiZpCj&q=85&s=8d61c59531dbcdcace7fcdfc524bd4d2)](https://www.tensorzero.com/)

Search...

⌘K

Search...

Navigation

Operations

Export OpenTelemetry traces (OTLP)

[Guides](https://www.tensorzero.com/docs) [Integrations](https://www.tensorzero.com/docs/integrations/model-providers)

On this page

- [Configure](https://www.tensorzero.com/docs/operations/export-opentelemetry-traces#configure)
- [Customize](https://www.tensorzero.com/docs/operations/export-opentelemetry-traces#customize)
- [Send custom HTTP headers](https://www.tensorzero.com/docs/operations/export-opentelemetry-traces#send-custom-http-headers)
- [Define custom headers in the configuration](https://www.tensorzero.com/docs/operations/export-opentelemetry-traces#define-custom-headers-in-the-configuration)
- [Define custom headers during inference](https://www.tensorzero.com/docs/operations/export-opentelemetry-traces#define-custom-headers-during-inference)
- [Send custom OpenTelemetry attributes](https://www.tensorzero.com/docs/operations/export-opentelemetry-traces#send-custom-opentelemetry-attributes)
- [Send custom OpenTelemetry resources](https://www.tensorzero.com/docs/operations/export-opentelemetry-traces#send-custom-opentelemetry-resources)
- [Link to existing traces with traceparent](https://www.tensorzero.com/docs/operations/export-opentelemetry-traces#link-to-existing-traces-with-traceparent)
- [Export OpenInference traces](https://www.tensorzero.com/docs/operations/export-opentelemetry-traces#export-openinference-traces)

The TensorZero Gateway can export traces to an external OpenTelemetry-compatible observability system using OTLP.Exporting traces via OpenTelemetry allows you to monitor the TensorZero Gateway in external observability platforms such as Jaeger, Datadog, or Grafana.
This integration enables you to correlate gateway activity with the rest of your infrastructure, providing deeper insights and unified monitoring across your systems.Exporting traces via OpenTelemetry does not replace the core observability features built into TensorZero.
Many key TensorZero features (including optimization) require richer observability data that TensorZero collects and stores in your ClickHouse database.
Traces exported through OpenTelemetry are for external observability only.

The TensorZero Gateway also provides a Prometheus-compatible metrics endpoint at `/metrics`.
This endpoint includes metrics about the gateway itself rather than the data processed by the gateway.
See [Export Prometheus metrics](https://www.tensorzero.com/docs/operations/export-prometheus-metrics) for more details.

## [​](https://www.tensorzero.com/docs/operations/export-opentelemetry-traces\#configure)  Configure

You can find a [complete runnable example](https://github.com/tensorzero/tensorzero/tree/main/examples/guides/opentelemetry-otlp) exporting traces to Jaeger on GitHub.

1

Set up the configuration

Enable `export.otlp.traces.enabled` in the `[gateway]` section of the `tensorzero.toml` configuration file:

Copy

```
[gateway]
# ...
export.otlp.traces.enabled = true
# ...
```

2

Configure the OTLP traces endpoint

Set the `OTEL_EXPORTER_OTLP_TRACES_ENDPOINT` environment variable in the gateway container to the endpoint of your OpenTelemetry service.

Example: TensorZero Gateway and Jaeger with Docker Compose

For example, if you’re deploying the TensorZero Gateway and Jaeger in Docker Compose, you can set the following environment variable:

Copy

```
services:
  gateway:
    image: tensorzero/gateway
    environment:
      OTEL_EXPORTER_OTLP_TRACES_ENDPOINT: http://jaeger:4317
    # ...

  jaeger:
    image: jaegertracing/jaeger
    ports:
      - "4317:4317"
    # ...
```

3

Browse the exported traces

Once configured, the TensorZero Gateway will begin sending traces to your OpenTelemetry-compatible service.Traces are generated for each HTTP request handled by the gateway (excluding auxiliary endpoints).
For inference requests, these traces additionally contain spans that represent the processing of functions, variants, models, and model providers.![Screenshot of TensorZero Gateway traces in Jaeger](https://mintcdn.com/tensorzero/pYf2iV2mfPd4h1EU/operations/export-opentelemetry-traces-jaeger.png?fit=max&auto=format&n=pYf2iV2mfPd4h1EU&q=85&s=787808f5c50421227a7d6eb33f2915d8)

## [​](https://www.tensorzero.com/docs/operations/export-opentelemetry-traces\#customize)  Customize

### [​](https://www.tensorzero.com/docs/operations/export-opentelemetry-traces\#send-custom-http-headers)  Send custom HTTP headers

You can attach custom HTTP headers to the outgoing OTLP export requests made to `OTEL_EXPORTER_OTLP_TRACES_ENDPOINT`.

#### [​](https://www.tensorzero.com/docs/operations/export-opentelemetry-traces\#define-custom-headers-in-the-configuration)  Define custom headers in the configuration

You can configure static headers that will be included in all OTLP export requests by adding them to the `export.otlp.traces.extra_headers` field in your configuration file:

tensorzero.toml

Copy

```
[gateway.export.otlp.traces]
# ...
extra_headers.space_id = "my-workspace-123"
extra_headers."X-Environment" = "production"
# ...
```

#### [​](https://www.tensorzero.com/docs/operations/export-opentelemetry-traces\#define-custom-headers-during-inference)  Define custom headers during inference

You can also send custom headers dynamically on a per-request basis.
When there is a conflict between static and dynamic headers, the latter takes precedence.

- Python (TensorZero SDK)

- Python (OpenAI SDK)

- Node (OpenAI SDK)

- HTTP


When using the TensorZero Python SDK, you can pass dynamic OTLP headers using the `otlp_traces_extra_headers` parameter in the `inference` method.
The headers will be automatically prefixed with `tensorzero-otlp-traces-extra-header-` for you:

Copy

```
response = t0.inference(
    function_name="your_function_name",
    input={
        "messages": [\
            {\
                "role": "user",\
                "content": "Write a haiku about artificial intelligence.",\
            }\
        ]
    },
    otlp_traces_extra_headers={
        "user-id": "user-123",
        "request-source": "mobile-app",
    },
)
```

This will attach the headers `user-id: user-123` and `request-source: mobile-app` when exporting any span associated with that specific inference request.

### [​](https://www.tensorzero.com/docs/operations/export-opentelemetry-traces\#send-custom-opentelemetry-attributes)  Send custom OpenTelemetry attributes

You can attach custom span attributes using headers prefixed with `tensorzero-otlp-traces-extra-attribute-`.
The values must be valid JSON; TensorZero currently supports strings and booleans only.
For example:

Copy

```
curl -X POST http://localhost:3000/inference \
  -H "tensorzero-otlp-traces-extra-attribute-user_id: \"user-123\"" \
  -H "tensorzero-otlp-traces-extra-attribute-is_premium: true" \
  -d '{ ... }'
```

### [​](https://www.tensorzero.com/docs/operations/export-opentelemetry-traces\#send-custom-opentelemetry-resources)  Send custom OpenTelemetry resources

You can attach custom resource attributes using headers prefixed with `tensorzero-otlp-traces-extra-resource-`.
For example:

Copy

```
curl -X POST http://localhost:3000/inference \
  -H "tensorzero-otlp-traces-extra-resource-service.namespace: production" \
  -d '{ ... }'
```

### [​](https://www.tensorzero.com/docs/operations/export-opentelemetry-traces\#link-to-existing-traces-with-traceparent)  Link to existing traces with `traceparent`

TensorZero automatically handles incoming `traceparent` headers for distributed tracing when OTLP is enabled.
This follows the [W3C Trace Context standard](https://www.w3.org/TR/trace-context/).

Copy

```
curl -X POST http://localhost:3000/inference \
  -H "traceparent: 00-0af7651916cd43dd8448eb211c80319c-b7ad6b7169203331-01" \
  -d '{ ... }'
```

TensorZero spans will become children of the incoming trace, preserving the trace ID across services.

### [​](https://www.tensorzero.com/docs/operations/export-opentelemetry-traces\#export-openinference-traces)  Export OpenInference traces

By default, TensorZero exports traces with attributes that follow the [OpenTelemetry Generative AI semantic conventions](https://github.com/open-telemetry/semantic-conventions/tree/main/docs/gen-ai).You can instead choose to export traces with attributes that follow the [OpenInference semantic conventions](https://github.com/Arize-ai/openinference/blob/main/spec/llm_spans.md) by setting `export.otlp.traces.format = "openinference"` in your configuration file.
See [Configuration Reference](https://www.tensorzero.com/docs/gateway/configuration-reference) for more details.

[Organize your configuration](https://www.tensorzero.com/docs/operations/organize-your-configuration) [Export Prometheus metrics](https://www.tensorzero.com/docs/operations/export-prometheus-metrics)

⌘I

## Export Prometheus Metrics
[Skip to main content](https://www.tensorzero.com/docs/operations/export-prometheus-metrics#content-area)

[TensorZero Docs home page![light logo](https://mintcdn.com/tensorzero/afFfERJmQ8YiZpCj/logo/light.svg?fit=max&auto=format&n=afFfERJmQ8YiZpCj&q=85&s=9396731a8a92cee7bd293dee05d71dac)![dark logo](https://mintcdn.com/tensorzero/afFfERJmQ8YiZpCj/logo/dark.svg?fit=max&auto=format&n=afFfERJmQ8YiZpCj&q=85&s=8d61c59531dbcdcace7fcdfc524bd4d2)](https://www.tensorzero.com/)

Search...

Ctrl K

Search...

Navigation

Operations

Export Prometheus metrics

[Guides](https://www.tensorzero.com/docs) [Integrations](https://www.tensorzero.com/docs/integrations/model-providers)

The TensorZero Gateway exposes runtime metrics through a [Prometheus](https://prometheus.io/)-compatible endpoint.
This allows you to monitor gateway performance, track usage patterns, and set up alerting using standard Prometheus tooling.
This endpoint provides operational metrics about the gateway itself.
It’s not meant to replace TensorZero’s observability features.You can access the metrics by scraping the `/metrics` endpoint.The gateway currently exports the following metrics:

- `tensorzero_requests_total`
- `tensorzero_inferences_total`

The metrics include relevant labels such as `endpoint`, `function_name`, `model_name`, and `metric_name`.
For example:

GET /metrics

Copy

```
# HELP tensorzero_requests_total Requests handled by TensorZero
# TYPE tensorzero_requests_total counter
tensorzero_requests_total{endpoint="inference",function_name="tensorzero::default",model_name="gpt-4o-mini-2024-07-18"} 1
tensorzero_requests_total{endpoint="feedback",metric_name="draft_accepted"} 10

# HELP tensorzero_inferences_total Inferences performed by TensorZero
# TYPE tensorzero_inferences_total counter
tensorzero_inferences_total{endpoint="inference",function_name="tensorzero::default",model_name="gpt-4o-mini-2024-07-18"} 1
```

[Export OpenTelemetry traces](https://www.tensorzero.com/docs/operations/export-opentelemetry-traces) [Extend TensorZero](https://www.tensorzero.com/docs/operations/extend-tensorzero)

Ctrl+I

Assistant

Responses are generated using AI and may contain mistakes.

## Extend TensorZero
[Skip to main content](https://www.tensorzero.com/docs/operations/extend-tensorzero#content-area)

[TensorZero Docs home page![light logo](https://mintcdn.com/tensorzero/afFfERJmQ8YiZpCj/logo/light.svg?fit=max&auto=format&n=afFfERJmQ8YiZpCj&q=85&s=9396731a8a92cee7bd293dee05d71dac)![dark logo](https://mintcdn.com/tensorzero/afFfERJmQ8YiZpCj/logo/dark.svg?fit=max&auto=format&n=afFfERJmQ8YiZpCj&q=85&s=8d61c59531dbcdcace7fcdfc524bd4d2)](https://www.tensorzero.com/)

Search...

Ctrl K

Search...

Navigation

Operations

Extend TensorZero

[Guides](https://www.tensorzero.com/docs) [Integrations](https://www.tensorzero.com/docs/integrations/model-providers)

On this page

- [Features](https://www.tensorzero.com/docs/operations/extend-tensorzero#features)
- [extra\_body](https://www.tensorzero.com/docs/operations/extend-tensorzero#extra-body)
- [extra\_headers](https://www.tensorzero.com/docs/operations/extend-tensorzero#extra-headers)
- [include\_original\_response](https://www.tensorzero.com/docs/operations/extend-tensorzero#include-original-response)
- [TensorZero Data](https://www.tensorzero.com/docs/operations/extend-tensorzero#tensorzero-data)
- [Example: Anthropic Computer Use](https://www.tensorzero.com/docs/operations/extend-tensorzero#example:-anthropic-computer-use)

TensorZero aims to provide a great developer experience while giving you full access to the underlying capabilities of each model provider.We provide advanced features that let you customize requests and access provider-specific functionality that isn’t directly supported in TensorZero.
You shouldn’t need these features most of the time, but they’re around if necessary.

Is there something you weren’t able to do with TensorZero?
Please let us know and we’ll try to tackle it — not just for the specific case but a general solution for that class of workflow.

## [​](https://www.tensorzero.com/docs/operations/extend-tensorzero\#features)  Features

### [​](https://www.tensorzero.com/docs/operations/extend-tensorzero\#extra-body)  `extra_body`

You can use the `extra_body` field to override the request body that TensorZero sends to model providers.You can set `extra_body` on a variant configuration block, a model provider configuration block, or at inference time.
See [Configuration Reference](https://www.tensorzero.com/docs/gateway/configuration-reference) and [Inference API Reference](https://www.tensorzero.com/docs/gateway/api-reference/inference) for more details.

### [​](https://www.tensorzero.com/docs/operations/extend-tensorzero\#extra-headers)  `extra_headers`

You can use the `extra_headers` field to override the request headers that TensorZero sends to model providers.You can set `extra_headers` on a variant configuration block, a model provider configuration block, or at inference time.
See [Configuration Reference](https://www.tensorzero.com/docs/gateway/configuration-reference) and [Inference API Reference](https://www.tensorzero.com/docs/gateway/api-reference/inference) for more details.

### [​](https://www.tensorzero.com/docs/operations/extend-tensorzero\#include-original-response)  `include_original_response`

If you enable this feature while running inference, the gateway will return the original response from the model provider along with the TensorZero response.See [Inference API Reference](https://www.tensorzero.com/docs/gateway/api-reference/inference) for more details.

### [​](https://www.tensorzero.com/docs/operations/extend-tensorzero\#tensorzero-data)  TensorZero Data

TensorZero stores all its data on your own ClickHouse database.You can query this data directly by running SQL queries against your ClickHouse instance.
If you’re feeling particularly adventurous, you can also write to ClickHouse directly (though you should be careful when upgrading your TensorZero deployment to account for any database migrations).See [Data model](https://www.tensorzero.com/docs/gateway/data-model) for more details.

## [​](https://www.tensorzero.com/docs/operations/extend-tensorzero\#example:-anthropic-computer-use)  Example: Anthropic Computer Use

At the time of writing, TensorZero hadn’t integrated with Anthropic’s Computer Use features directly — but they worked out of the box!Concretely, Anthropic Computer Use requires setting additional fields to the request body as well as a request header.
Let’s define a TensorZero function that includes these additional parameters:

Copy

```
[functions.bash_assistant]
type = "chat"

[functions.bash_assistant.variants.anthropic_claude_3_7_sonnet_20250219]
type = "chat_completion"
model = "anthropic::claude-3-7-sonnet-20250219"
max_tokens = 2048
extra_body = [\
    { pointer = "/tools", value = [{ type = "bash_20250124", name = "bash" }] },\
    { pointer = "/ultrathinking", value = { type = "enabled", budget_tokens = 1024 } }, # made-up parameter\
]
extra_headers = [\
    { name = "anthropic-beta", value = "computer-use-2025-01-24" },\
]
```

This example illustrates how you should be able to use the vast majority of features supported by the model provider even if TensorZero doesn’t have explicit support for them yet.

[Export Prometheus metrics](https://www.tensorzero.com/docs/operations/export-prometheus-metrics)

Ctrl+I

Assistant

Responses are generated using AI and may contain mistakes.

## Manage Credentials
[Skip to main content](https://www.tensorzero.com/docs/operations/manage-credentials#content-area)

[TensorZero Docs home page![light logo](https://mintcdn.com/tensorzero/afFfERJmQ8YiZpCj/logo/light.svg?fit=max&auto=format&n=afFfERJmQ8YiZpCj&q=85&s=9396731a8a92cee7bd293dee05d71dac)![dark logo](https://mintcdn.com/tensorzero/afFfERJmQ8YiZpCj/logo/dark.svg?fit=max&auto=format&n=afFfERJmQ8YiZpCj&q=85&s=8d61c59531dbcdcace7fcdfc524bd4d2)](https://www.tensorzero.com/)

Search...

Ctrl K

Search...

Navigation

Operations

Manage credentials (API keys)

[Guides](https://www.tensorzero.com/docs) [Integrations](https://www.tensorzero.com/docs/integrations/model-providers)

On this page

- [Default Behavior](https://www.tensorzero.com/docs/operations/manage-credentials#default-behavior)
- [Customizing Credential Management](https://www.tensorzero.com/docs/operations/manage-credentials#customizing-credential-management)
- [Static Credentials](https://www.tensorzero.com/docs/operations/manage-credentials#static-credentials)
- [Load Balancing Between Multiple Credentials](https://www.tensorzero.com/docs/operations/manage-credentials#load-balancing-between-multiple-credentials)
- [Dynamic Credentials](https://www.tensorzero.com/docs/operations/manage-credentials#dynamic-credentials)
- [Configure credential fallbacks](https://www.tensorzero.com/docs/operations/manage-credentials#configure-credential-fallbacks)
- [Set default credentials for a provider type](https://www.tensorzero.com/docs/operations/manage-credentials#set-default-credentials-for-a-provider-type)

This guide explains how to manage credentials (API keys) in TensorZero Gateway.Typically, the TensorZero Gateway will look for credentials like API keys using standard environment variables.
The gateway will load credentials from the environment variables on startup, and your application doesn’t need to have access to the credentials.That said, you can customize this behavior by setting alternative credential locations for each provider.
For example, you can provide credentials dynamically at inference time, or set alternative static credentials for each provider (e.g. to use multiple API keys for the same provider).

## [​](https://www.tensorzero.com/docs/operations/manage-credentials\#default-behavior)  Default Behavior

By default, the TensorZero Gateway will look for credentials in the following environment variables:

| Model Provider | Default Credential |
| --- | --- |
| [Anthropic](https://www.tensorzero.com/docs/integrations/model-providers/anthropic) | `ANTHROPIC_API_KEY` |
| [AWS Bedrock](https://www.tensorzero.com/docs/integrations/model-providers/aws-bedrock) | Uses AWS SDK credentials |
| [AWS SageMaker](https://www.tensorzero.com/docs/integrations/model-providers/aws-sagemaker) | Uses AWS SDK credentials |
| [Azure](https://www.tensorzero.com/docs/integrations/model-providers/azure) | `AZURE_OPENAI_API_KEY` |
| [Deepseek](https://www.tensorzero.com/docs/integrations/model-providers/deepseek) | `DEEPSEEK_API_KEY` |
| [Fireworks](https://www.tensorzero.com/docs/integrations/model-providers/fireworks) | `FIREWORKS_API_KEY` |
| [GCP Vertex AI (Anthropic)](https://www.tensorzero.com/docs/integrations/model-providers/gcp-vertex-ai-anthropic) | `GCP_VERTEX_CREDENTIALS_PATH` |
| [GCP Vertex AI (Gemini)](https://www.tensorzero.com/docs/integrations/model-providers/gcp-vertex-ai-gemini) | `GCP_VERTEX_CREDENTIALS_PATH` |
| [Google AI Studio (Gemini)](https://www.tensorzero.com/docs/integrations/model-providers/google-ai-studio-gemini) | `GOOGLE_API_KEY` |
| [Groq](https://www.tensorzero.com/docs/integrations/model-providers/groq) | `GROQ_API_KEY` |
| [Hyperbolic](https://www.tensorzero.com/docs/integrations/model-providers/hyperbolic) | `HYPERBOLIC_API_KEY` |
| [Mistral](https://www.tensorzero.com/docs/integrations/model-providers/mistral) | `MISTRAL_API_KEY` |
| [OpenAI](https://www.tensorzero.com/docs/integrations/model-providers/openai) | `OPENAI_API_KEY` |
| [OpenAI-Compatible](https://www.tensorzero.com/docs/integrations/model-providers/openai-compatible) | `OPENAI_API_KEY` |
| [OpenRouter](https://www.tensorzero.com/docs/integrations/model-providers/openrouter) | `OPENROUTER_API_KEY` |
| [SGLang](https://www.tensorzero.com/docs/integrations/model-providers/sglang) | `SGLANG_API_KEY` |
| [Text Generation Inference (TGI)](https://www.tensorzero.com/docs/integrations/model-providers/tgi) | None |
| [Together](https://www.tensorzero.com/docs/integrations/model-providers/together) | `TOGETHER_API_KEY` |
| [vLLM](https://www.tensorzero.com/docs/integrations/model-providers/vllm) | None |
| [XAI](https://www.tensorzero.com/docs/integrations/model-providers/xai) | `XAI_API_KEY` |

## [​](https://www.tensorzero.com/docs/operations/manage-credentials\#customizing-credential-management)  Customizing Credential Management

You can customize the source of credentials for each provider.See [Configuration Reference](https://www.tensorzero.com/docs/gateway/configuration-reference) (e.g. `api_key_location`) for more information on the different ways to configure credentials for each provider.
Also see the relevant provider guides for more information on how to configure credentials for each provider.

### [​](https://www.tensorzero.com/docs/operations/manage-credentials\#static-credentials)  Static Credentials

You can set alternative static credentials for each provider.For example, let’s say we want to use a different environment variable for an OpenAI provider.
We can customize variable name by setting the `api_key_location` to `env::MY_OTHER_OPENAI_API_KEY`.

Copy

```
[models.gpt_4o_mini.providers.my_other_openai]
type = "openai"
api_key_location = "env::MY_OTHER_OPENAI_API_KEY"
# ...
```

At startup, the TensorZero Gateway will look for the `MY_OTHER_OPENAI_API_KEY` environment variable and use that value for the API key.

#### [​](https://www.tensorzero.com/docs/operations/manage-credentials\#load-balancing-between-multiple-credentials)  Load Balancing Between Multiple Credentials

You can load balance between different API keys for the same provider by defining multiple variants and models.For example, the configuration below will split the traffic between two different OpenAI API keys, `OPENAI_API_KEY_1` and `OPENAI_API_KEY_2`.

Copy

```
[models.gpt_4o_mini_1]
routing = ["openai"]

[models.gpt_4o_mini_1.providers.openai]
type = "openai"
model_name = "gpt-4o-mini"
api_key_location = "env::OPENAI_API_KEY_1"

[models.gpt_4o_mini_2]
routing = ["openai"]

[models.gpt_4o_mini_2.providers.openai]
type = "openai"
model_name = "gpt-4o-mini"
api_key_location = "env::OPENAI_API_KEY_2"

[functions.generate_haiku]
type = "chat"

[functions.generate_haiku.variants.gpt_4o_mini_1]
type = "chat_completion"
model = "gpt_4o_mini_1"

[functions.generate_haiku.variants.gpt_4o_mini_2]
type = "chat_completion"
model = "gpt_4o_mini_2"
```

You can use the same principle to set up fallbacks between different API keys for the same provider.
See [Retries & Fallbacks](https://www.tensorzero.com/docs/gateway/guides/retries-fallbacks) for more information on how to configure retries and fallbacks.

### [​](https://www.tensorzero.com/docs/operations/manage-credentials\#dynamic-credentials)  Dynamic Credentials

You can provide API keys dynamically at inference time.To do this, you can use the `dynamic::` prefix in the relevant credential field in the provider configuration.For example, let’s say we want to provide dynamic API keys for the OpenAI provider.

Copy

```
[models.user_gpt_4o_mini]
routing = ["openai"]

[models.user_gpt_4o_mini.providers.openai]
type = "openai"
model_name = "gpt-4o-mini"
api_key_location = "dynamic::customer_openai_api_key"
```

At inference time, you can provide the API key in the `credentials` argument.

Copy

```
from tensorzero import TensorZeroGateway

with TensorZeroGateway.build_http(gateway_url="http://localhost:3000") as client:
    response = client.inference(
        function_name="generate_haiku",
        input={
            "messages": [\
                {\
                    "role": "user",\
                    "content": "Write a haiku about artificial intelligence.",\
                }\
            ]
        },
        credentials={
            "customer_openai_api_key": "sk-..."
        }
    )

print(response)
```

### [​](https://www.tensorzero.com/docs/operations/manage-credentials\#configure-credential-fallbacks)  Configure credential fallbacks

You can configure fallback credentials that will be used automatically if the primary credential fails.This is particularly useful for calling functions and models that require dynamic credentials from the TensorZero UI (by falling back to static credentials).To configure a fallback, use an object with `default` and `fallback` fields instead of a simple string:

Copy

```
[models.gpt_4o_mini]
routing = ["openai"]

[models.gpt_4o_mini.providers.openai]
type = "openai"
model_name = "gpt-4o-mini"
api_key_location = { default = "dynamic::customer_openai_api_key", fallback = "env::OPENAI_API_KEY" }
```

At inference time, the gateway will first try to use the dynamic credential.
If that fails, it will automatically fall back to the environment variable.

### [​](https://www.tensorzero.com/docs/operations/manage-credentials\#set-default-credentials-for-a-provider-type)  Set default credentials for a provider type

Most model providers have default credential locations.
For example, OpenAI’s `api_key_location` defaults to `env::OPENAI_API_KEY`.
These credentials apply to the default function and shorthand models (e.g. calling the model `openai::gpt-5`).You can override the default location for a particular provider using `[provider_types.YOUR_PROVIDER_TYPE.defaults]`.
For example, we can override the default location for the OpenAI provider type to require a dynamic API key:

tensorzero.toml

Copy

```
[provider_types.openai.defaults]
api_key_location = "dynamic::customer_openai_api_key"
# ...
```

Unless otherwise specified, every model provider of type `openai` will require the `customer_openai_api_key` credential.See the [Configuration Reference](https://www.tensorzero.com/docs/gateway/configuration-reference) for more details.

[Optimize latency & throughput](https://www.tensorzero.com/docs/deployment/optimize-latency-and-throughput) [Set up auth for TensorZero](https://www.tensorzero.com/docs/operations/set-up-auth-for-tensorzero)

Ctrl+I

Assistant

Responses are generated using AI and may contain mistakes.

## Organize TensorZero Configuration
[Skip to main content](https://www.tensorzero.com/docs/operations/organize-your-configuration#content-area)

[TensorZero Docs home page![light logo](https://mintcdn.com/tensorzero/afFfERJmQ8YiZpCj/logo/light.svg?fit=max&auto=format&n=afFfERJmQ8YiZpCj&q=85&s=9396731a8a92cee7bd293dee05d71dac)![dark logo](https://mintcdn.com/tensorzero/afFfERJmQ8YiZpCj/logo/dark.svg?fit=max&auto=format&n=afFfERJmQ8YiZpCj&q=85&s=8d61c59531dbcdcace7fcdfc524bd4d2)](https://www.tensorzero.com/)

Search...

Ctrl K

Search...

Navigation

Operations

Organize your configuration

[Guides](https://www.tensorzero.com/docs) [Integrations](https://www.tensorzero.com/docs/integrations/model-providers)

On this page

- [Split your configuration into multiple files](https://www.tensorzero.com/docs/operations/organize-your-configuration#split-your-configuration-into-multiple-files)
- [Enable template file system access to reuse shared snippets](https://www.tensorzero.com/docs/operations/organize-your-configuration#enable-template-file-system-access-to-reuse-shared-snippets)

You can use custom configuration to take full advantage of TensorZero’s features.
See [Configuration Reference](https://www.tensorzero.com/docs/gateway/configuration-reference) for more details.
This guide shares best practices for organizing your configuration as your project grows in complexity.

You can find a [complete runnable example](https://github.com/tensorzero/tensorzero/tree/main/examples/docs/guides/operations/organize-your-configuration) of this section on GitHub.

## [​](https://www.tensorzero.com/docs/operations/organize-your-configuration\#split-your-configuration-into-multiple-files)  Split your configuration into multiple files

As your project grows in complexity, it might be a good idea to split your configuration into multiple files.
This makes it easier to manage and maintain your configuration.For example, you can create separate TOML files for different projects, environments, and so on.
You can also move deprecated entries like functions to a separate file.You can instruct TensorZero to load multiple configuration files by specifying a glob pattern that matches all the relevant TOML files:

- **TensorZero Gateway:** Set the CLI flag `--config-path path/to/**/*.toml`.
- **TensorZero UI:** Set the environment variable `TENSORZERO_UI_CONFIG_PATH=path/to/**/*.toml`.

Under the hood, TensorZero will concatenate the configuration files, with special handling for paths.
For example, you can declare a model in one file and use it in a variant declared in another file.
If the configuration includes a path (e.g. template, schema), the path will be resolved relative to that configuration file’s directory.
For example:

Copy

```
[functions.my_function.variants.my_variant]
# ...
templates.my_template.path = "path/to/template.minijinja"  # relative to this TOML file
# ...
```

## [​](https://www.tensorzero.com/docs/operations/organize-your-configuration\#enable-template-file-system-access-to-reuse-shared-snippets)  Enable template file system access to reuse shared snippets

You can decompose your templates into smaller, reusable snippets.
This makes it easier to maintain and reuse code across multiple templates.Templates can reference other templates using the MiniJinja directives `{% include %}` and `{% import %}`.
To use these directives, set `gateway.template_filesystem_access.base_path` in your configuration file.By default, file system access is disabled for security reasons, since template imports are evaluated dynamically and could potentially access sensitive files.
You should ensure that only trusted templates are allowed access to the file system.

Copy

```
[gateway]
# ...
template_filesystem_access.base_path = "."
# ...
```

Template imports are resolved relative to `base_path`.
If `base_path` itself is relative, it’s relative to the configuration file in which it’s defined.

[Enforce custom rate limits](https://www.tensorzero.com/docs/operations/enforce-custom-rate-limits) [Export OpenTelemetry traces](https://www.tensorzero.com/docs/operations/export-opentelemetry-traces)

Ctrl+I

Assistant

Responses are generated using AI and may contain mistakes.

## TensorZero Auth Setup
[Skip to main content](https://www.tensorzero.com/docs/operations/set-up-auth-for-tensorzero#content-area)

[TensorZero Docs home page![light logo](https://mintcdn.com/tensorzero/afFfERJmQ8YiZpCj/logo/light.svg?fit=max&auto=format&n=afFfERJmQ8YiZpCj&q=85&s=9396731a8a92cee7bd293dee05d71dac)![dark logo](https://mintcdn.com/tensorzero/afFfERJmQ8YiZpCj/logo/dark.svg?fit=max&auto=format&n=afFfERJmQ8YiZpCj&q=85&s=8d61c59531dbcdcace7fcdfc524bd4d2)](https://www.tensorzero.com/)

Search...

Ctrl K

Search...

Navigation

Operations

Set up auth for TensorZero

[Guides](https://www.tensorzero.com/docs) [Integrations](https://www.tensorzero.com/docs/integrations/model-providers)

On this page

- [Configure](https://www.tensorzero.com/docs/operations/set-up-auth-for-tensorzero#configure)
- [Advanced](https://www.tensorzero.com/docs/operations/set-up-auth-for-tensorzero#advanced)
- [Customize the gateway’s authentication cache](https://www.tensorzero.com/docs/operations/set-up-auth-for-tensorzero#customize-the-gateway%E2%80%99s-authentication-cache)
- [Set up rate limiting by API key](https://www.tensorzero.com/docs/operations/set-up-auth-for-tensorzero#set-up-rate-limiting-by-api-key)

You can create TensorZero API keys to authenticate your requests to the TensorZero Gateway.
This way, your clients don’t need access to model provider credentials, making it easier to manage access and security.This page shows how to:

- Create API keys for the TensorZero Gateway
- Require clients to use these API keys for requests
- Manage API keys in the TensorZero UI

TensorZero supports authentication for the gateway.
Authentication for the UI is coming soon.
In the meantime, we recommend pairing the UI with complementary products like Nginx, OAuth2 Proxy, or Tailscale.

## [​](https://www.tensorzero.com/docs/operations/set-up-auth-for-tensorzero\#configure)  Configure

You can find a [complete runnable example](https://github.com/tensorzero/tensorzero/tree/main/examples/docs/guides/operations/set-up-auth-for-tensorzero) of this guide on GitHub.

1

Configure your gateway to require authentication

You can instruct the TensorZero Gateway to require authentication in the configuration:

tensorzero.toml

Copy

```
[gateway]
auth.enabled = true
```

With this setting, every gateway endpoint except for `/status` and `/health` will require authentication.

2

Deploy TensorZero and Postgres

You must set up Postgres to use TensorZero’s authentication features.

- [Deploy the TensorZero Gateway](https://www.tensorzero.com/docs/deployment/tensorzero-gateway)
- [Deploy the TensorZero UI](https://www.tensorzero.com/docs/deployment/tensorzero-ui)
- [Deploy ClickHouse](https://www.tensorzero.com/docs/deployment/clickhouse)
- [Deploy Postgres](https://www.tensorzero.com/docs/deployment/postgres)

Example: Docker Compose

You can deploy all the requirements using the Docker Compose file below:

docker-compose.yml

Copy

```
# This is a simplified example for learning purposes. Do not use this in production.
# For production-ready deployments, see: https://www.tensorzero.com/docs/deployment/tensorzero-gateway

services:
  clickhouse:
    image: clickhouse:lts
    environment:
      CLICKHOUSE_USER: chuser
      CLICKHOUSE_DEFAULT_ACCESS_MANAGEMENT: 1
      CLICKHOUSE_PASSWORD: chpassword
    ports:
      - "8123:8123" # HTTP port
      - "9000:9000" # Native port
    volumes:
      - clickhouse-data:/var/lib/clickhouse
    ulimits:
      nofile:
        soft: 262144
        hard: 262144
    healthcheck:
      test: wget --spider --tries 1 http://chuser:chpassword@clickhouse:8123/ping
      start_period: 30s
      start_interval: 1s
      timeout: 1s

  postgres:
    image: postgres:14-alpine
    environment:
      POSTGRES_DB: tensorzero
      POSTGRES_USER: postgres
      POSTGRES_PASSWORD: postgres
    ports:
      - "5432:5432"
    volumes:
      - postgres-data:/var/lib/postgresql/data
    healthcheck:
      test: pg_isready -U postgres
      start_period: 30s
      start_interval: 1s
      timeout: 1s

  gateway:
    image: tensorzero/gateway
    volumes:
      - ./config:/app/config:ro
    command: --config-file /app/config/tensorzero.toml
    environment:
      TENSORZERO_CLICKHOUSE_URL: http://chuser:chpassword@clickhouse:8123/tensorzero
      TENSORZERO_POSTGRES_URL: postgres://postgres:postgres@postgres:5432/tensorzero
      OPENAI_API_KEY: ${OPENAI_API_KEY:?Environment variable OPENAI_API_KEY must be set.}
    ports:
      - "3000:3000"
    extra_hosts:
      - "host.docker.internal:host-gateway"
    healthcheck:
      test: wget --spider --tries 1 http://localhost:3000/status
      start_period: 30s
      start_interval: 1s
      timeout: 1s
    depends_on:
      clickhouse:
        condition: service_healthy
      postgres:
        condition: service_healthy

  ui:
    image: tensorzero/ui
    volumes:
      - ./config:/app/config:ro
    environment:
      OPENAI_API_KEY: ${OPENAI_API_KEY:?Environment variable OPENAI_API_KEY must be set.}
      TENSORZERO_CLICKHOUSE_URL: http://chuser:chpassword@clickhouse:8123/tensorzero
      TENSORZERO_POSTGRES_URL: postgres://postgres:postgres@postgres:5432/tensorzero
      TENSORZERO_GATEWAY_URL: http://gateway:3000
    ports:
      - "4000:4000"
    depends_on:
      clickhouse:
        condition: service_healthy
      gateway:
        condition: service_healthy

volumes:
  postgres-data:
  clickhouse-data:
```

3

Create a TensorZero API key

You can create API keys using the TensorZero UI.
If you’re running a standard local deployment, visit `http://localhost:4000/api-keys` to create a key.Alternatively, you can create API keys programmatically in the CLI using the gateway binary with the `--create-api-key` flag.
For example:

Copy

```
docker compose run --rm gateway --create-api-key
```

The API key is a secret and should be kept secure.

Once you’ve created an API key, set the `TENSORZERO_API_KEY` environment variable.

4

Make an authenticated inference request

- Python (TensorZero SDK)

- Python (OpenAI SDK)

- Node (OpenAI SDK)

- HTTP


You can make authenticated requests by setting the `api_key` parameter in your TensorZero client:

tensorzero\_sdk.py

Copy

```
import os

from tensorzero import TensorZeroGateway

t0 = TensorZeroGateway.build_http(
    api_key=os.environ["TENSORZERO_API_KEY"],
    gateway_url="http://localhost:3000",
)

response = t0.inference(
    model_name="openai::gpt-5-mini",
    input={
        "messages": [\
            {\
                "role": "user",\
                "content": "Tell me a fun fact.",\
            }\
        ]
    },
)

print(response)
```

The client will automatically read the `TENSORZERO_API_KEY` environment variable if you don’t set `api_key`.

Authentication is not supported in the embedded (in-memory) gateway in Python.
Please use the HTTP client with a standalone gateway to make authenticated requests.

5

Manage API keys in the TensorZero UI

You can manage and delete API keys in the TensorZero UI.
If you’re running a standard local deployment, visit `http://localhost:4000/api-keys` to manage your keys.

## [​](https://www.tensorzero.com/docs/operations/set-up-auth-for-tensorzero\#advanced)  Advanced

### [​](https://www.tensorzero.com/docs/operations/set-up-auth-for-tensorzero\#customize-the-gateway%E2%80%99s-authentication-cache)  Customize the gateway’s authentication cache

By default, the TensorZero Gateway caches authentication database queries for one second.
You can customize this behavior in the configuration:

Copy

```
[gateway.auth.cache]
enabled = true # boolean
ttl_ms = 60_000 # one minute
```

### [​](https://www.tensorzero.com/docs/operations/set-up-auth-for-tensorzero\#set-up-rate-limiting-by-api-key)  Set up rate limiting by API key

Once you have authentication enabled, you can apply rate limits on a per-API-key basis using the `api_key_public_id` scope in your rate limiting rules.
This allows you to enforce different usage limits for different API keys, which is useful for implementing tiered access or preventing individual keys from consuming too many resources.

TensorZero API keys have the following format:`sk-t0-xxxxxxxxxxxx-yyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyy`The `xxxxxxxxxxxx` portion is the 12-character public ID that you can use in rate limiting rules.
The remaining portion of the key is secret and should be kept secure.

For example, you can limit each API key to 100 model inferences per hour, but allow a specific API key to make 1000 inferences:

Copy

```
# Each API key can make up to 100 model inferences per hour
[[rate_limiting.rules]]
priority = 0
model_inferences_per_hour = 100
scope = [\
    { api_key_public_id = "tensorzero::each" }\
]

# But override the limit for a specific API key
[[rate_limiting.rules]]
priority = 1
model_inferences_per_hour = 1000
scope = [\
    { api_key_public_id = "xxxxxxxxxxxx" }\
]
```

See [Enforce custom rate limits](https://www.tensorzero.com/docs/operations/enforce-custom-rate-limits) for more details on configuring rate limits with API keys.

[Manage credentials (API keys)](https://www.tensorzero.com/docs/operations/manage-credentials) [Enforce custom rate limits](https://www.tensorzero.com/docs/operations/enforce-custom-rate-limits)

Ctrl+I

Assistant

Responses are generated using AI and may contain mistakes.

## TensorZero Quickstart Guide
[Skip to main content](https://www.tensorzero.com/docs/quickstart#content-area)

[TensorZero Docs home page![light logo](https://mintcdn.com/tensorzero/afFfERJmQ8YiZpCj/logo/light.svg?fit=max&auto=format&n=afFfERJmQ8YiZpCj&q=85&s=9396731a8a92cee7bd293dee05d71dac)![dark logo](https://mintcdn.com/tensorzero/afFfERJmQ8YiZpCj/logo/dark.svg?fit=max&auto=format&n=afFfERJmQ8YiZpCj&q=85&s=8d61c59531dbcdcace7fcdfc524bd4d2)](https://www.tensorzero.com/)

Search...

Ctrl K

Search...

Navigation

Introduction

Quickstart

[Guides](https://www.tensorzero.com/docs) [Integrations](https://www.tensorzero.com/docs/integrations/model-providers)

On this page

- [Status Quo: OpenAI Wrapper](https://www.tensorzero.com/docs/quickstart#status-quo:-openai-wrapper)
- [Migrating to TensorZero](https://www.tensorzero.com/docs/quickstart#migrating-to-tensorzero)
- [Deploying TensorZero](https://www.tensorzero.com/docs/quickstart#deploying-tensorzero)
- [Our First TensorZero API Call](https://www.tensorzero.com/docs/quickstart#our-first-tensorzero-api-call)
- [TensorZero UI](https://www.tensorzero.com/docs/quickstart#tensorzero-ui)
- [Observability](https://www.tensorzero.com/docs/quickstart#observability)
- [Fine-Tuning](https://www.tensorzero.com/docs/quickstart#fine-tuning)
- [Conclusion & Next Steps](https://www.tensorzero.com/docs/quickstart#conclusion-&-next-steps)

This Quickstart guide shows how we’d upgrade an OpenAI wrapper to a minimal TensorZero deployment with built-in observability and fine-tuning capabilities — in just 5 minutes.
From there, you can take advantage of dozens of features to build best-in-class LLM applications.This Quickstart covers a tour of TensorZero features.
If you’re only interested in inference with the gateway, see the shorter [How to call any LLM](https://www.tensorzero.com/docs/gateway/call-any-llm) guide.

You can also find the runnable code for this example on [GitHub](https://github.com/tensorzero/tensorzero/tree/main/examples/quickstart).

## [​](https://www.tensorzero.com/docs/quickstart\#status-quo:-openai-wrapper)  Status Quo: OpenAI Wrapper

Imagine we’re building an LLM application that writes haikus.Today, our integration with OpenAI might look like this:

before.py

Copy

```
from openai import OpenAI

with OpenAI() as client:
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[\
            {\
                "role": "user",\
                "content": "Write a haiku about artificial intelligence.",\
            }\
        ],
    )

print(response)
```

Sample Output

Copy

```
ChatCompletion(
    id='chatcmpl-A5wr5WennQNF6nzF8gDo3SPIVABse',
    choices=[\
        Choice(\
            finish_reason='stop',\
            index=0,\
            logprobs=None,\
            message=ChatCompletionMessage(\
                content='Silent minds awaken,  \nPatterns dance in code and wire,  \nDreams of thought unfold.',\
                role='assistant',\
                function_call=None,\
                tool_calls=None,\
                refusal=None\
            )\
        )\
    ],
    created=1725981243,
    model='gpt-4o-mini',
    object='chat.completion',
    system_fingerprint='fp_483d39d857',
    usage=CompletionUsage(
      completion_tokens=19,
      prompt_tokens=22,
      total_tokens=41
    )
)
```

## [​](https://www.tensorzero.com/docs/quickstart\#migrating-to-tensorzero)  Migrating to TensorZero

TensorZero offers dozens of features covering inference, observability, optimization, evaluations, and experimentation.But the absolutely minimal setup requires just a simple configuration file: `tensorzero.toml`.

tensorzero.toml

Copy

```
# A function defines the task we're tackling (e.g. generating a haiku)...
[functions.generate_haiku]
type = "chat"

# ... and a variant is one of many implementations we can use to tackle it (a choice of prompt, model, etc.).
# Since we only have one variant for this function, the gateway will always use it.
[functions.generate_haiku.variants.gpt_4o_mini]
type = "chat_completion"
model = "openai::gpt-4o-mini"
```

This minimal configuration file tells the TensorZero Gateway everything it needs to replicate our original OpenAI call.

Using the shorthand `openai::gpt-4o-mini` notation is convenient for getting started.
To learn about all configuration options including schemas, templates, and advanced variant types, see [Configure functions and variants](https://www.tensorzero.com/docs/gateway/configure-functions-and-variants).
For production deployments with multiple providers, routing, and fallbacks, see [Configure models and providers](https://www.tensorzero.com/docs/gateway/configure-models-and-providers).

## [​](https://www.tensorzero.com/docs/quickstart\#deploying-tensorzero)  Deploying TensorZero

We’re almost ready to start making API calls.
Let’s launch TensorZero.

1. Set the environment variable `OPENAI_API_KEY`.
2. Place our `tensorzero.toml` in the `./config` directory.
3. Download the following sample `docker-compose.yml` file.
This Docker Compose configuration sets up a development ClickHouse database (where TensorZero stores data), the TensorZero Gateway, and the TensorZero UI.

Copy

```
curl -LO "https://raw.githubusercontent.com/tensorzero/tensorzero/refs/heads/main/examples/quickstart/docker-compose.yml"
```

Example: Docker Compose

docker-compose.yml

Copy

```
# This is a simplified example for learning purposes. Do not use this in production.
# For production-ready deployments, see: https://www.tensorzero.com/docs/deployment/tensorzero-gateway

services:
  clickhouse:
    image: clickhouse:lts
    environment:
      - CLICKHOUSE_USER=chuser
      - CLICKHOUSE_DEFAULT_ACCESS_MANAGEMENT=1
      - CLICKHOUSE_PASSWORD=chpassword
    ports:
      - "8123:8123"
    volumes:
      - clickhouse-data:/var/lib/clickhouse
    healthcheck:
      test: wget --spider --tries 1 http://chuser:chpassword@clickhouse:8123/ping
      start_period: 30s
      start_interval: 1s
      timeout: 1s

  # The TensorZero Python client *doesn't* require a separate gateway service.
  #
  # The gateway is only needed if you want to use the OpenAI Python client
  # or interact with TensorZero via its HTTP API (for other programming languages).
  #
  # The TensorZero UI also requires the gateway service.
  gateway:
    image: tensorzero/gateway
    volumes:
      # Mount our tensorzero.toml file into the container
      - ./config:/app/config:ro
    command: --config-file /app/config/tensorzero.toml
    environment:
      - TENSORZERO_CLICKHOUSE_URL=http://chuser:chpassword@clickhouse:8123/tensorzero
      - OPENAI_API_KEY=${OPENAI_API_KEY:?Environment variable OPENAI_API_KEY must be set.}
    ports:
      - "3000:3000"
    extra_hosts:
      - "host.docker.internal:host-gateway"
    depends_on:
      clickhouse:
        condition: service_healthy

  ui:
    image: tensorzero/ui
    volumes:
      # Mount our tensorzero.toml file into the container
      - ./config:/app/config:ro
    environment:
      - OPENAI_API_KEY=${OPENAI_API_KEY:?Environment variable OPENAI_API_KEY must be set.}
      - TENSORZERO_CLICKHOUSE_URL=http://chuser:chpassword@clickhouse:8123/tensorzero
      - TENSORZERO_GATEWAY_URL=http://gateway:3000
    ports:
      - "4000:4000"
    depends_on:
      clickhouse:
        condition: service_healthy

volumes:
  clickhouse-data:
```

Our setup should look like:

Copy

```
- config/
  - tensorzero.toml
- after.py see below
- before.py
- docker-compose.yml
```

Let’s launch everything!

Copy

```
docker compose up
```

## [​](https://www.tensorzero.com/docs/quickstart\#our-first-tensorzero-api-call)  Our First TensorZero API Call

The gateway will replicate our original OpenAI call and store the data in our database — with less than 1ms latency overhead thanks to Rust 🦀.The TensorZero Gateway can be used with the **TensorZero Python client**, with **OpenAI client (Python, Node, etc.)**, or via its **HTTP API in any programming language**.

- Python

- Python (Async)

- Python (OpenAI)

- Node (OpenAI)

- HTTP


You can install the TensorZero Python client with:

Copy

```
pip install tensorzero
```

Then, you can make a TensorZero API call with:

after.py

Copy

```
from tensorzero import TensorZeroGateway

with TensorZeroGateway.build_embedded(
    clickhouse_url="http://chuser:chpassword@localhost:8123/tensorzero",
    config_file="config/tensorzero.toml",
) as client:
    response = client.inference(
        function_name="generate_haiku",
        input={
            "messages": [\
                {\
                    "role": "user",\
                    "content": "Write a haiku about artificial intelligence.",\
                }\
            ]
        },
    )

print(response)
```

Sample Output

Copy

```
ChatInferenceResponse(
  inference_id=UUID('0191ddb2-2c02-7641-8525-494f01bcc468'),
  episode_id=UUID('0191ddb2-28f3-7cc2-b0cc-07f504d37e59'),
  variant_name='gpt_4o_mini',
  content=[\
    Text(\
      type='text',\
      text='Wires hum with intent,  \nThoughts born from code and structure,  \nGhost in silicon.'\
    )\
  ],
  usage=Usage(
    input_tokens=15,
    output_tokens=20
  )
)
```

## [​](https://www.tensorzero.com/docs/quickstart\#tensorzero-ui)  TensorZero UI

The TensorZero UI streamlines LLM engineering workflows like observability and optimization (e.g. fine-tuning).The Docker Compose file we used above also launched the TensorZero UI.
You can visit the UI at `http://localhost:4000`.

### [​](https://www.tensorzero.com/docs/quickstart\#observability)  Observability

The TensorZero UI provides a dashboard for observability data.
We can inspect data about individual inferences, entire functions, and more.

![TensorZero UI Observability - Function Detail Page - Screenshot](https://mintcdn.com/tensorzero/afFfERJmQ8YiZpCj/quickstart-observability-function.png?fit=max&auto=format&n=afFfERJmQ8YiZpCj&q=85&s=7bf90d865f7a3757a55f9678d2dbdfc7)

![TensorZero UI Observability - Inference Detail Page - Screenshot](https://mintcdn.com/tensorzero/afFfERJmQ8YiZpCj/quickstart-observability-inference.png?fit=max&auto=format&n=afFfERJmQ8YiZpCj&q=85&s=2d5c95bbb5d340c187fd2fcfe9102c66)

This guide is pretty minimal, so the observability data is pretty simple.
Once we start using more advanced functions like feedback and variants, the observability UI will enable us to track metrics, experiments (A/B tests), and more.

### [​](https://www.tensorzero.com/docs/quickstart\#fine-tuning)  Fine-Tuning

The TensorZero UI also provides a workflow for fine-tuning models like GPT-4o and Llama 3.
With a few clicks, you can launch a fine-tuning job.
Once the job is complete, the TensorZero UI will provide a configuration snippet you can add to your `tensorzero.toml`.![TensorZero UI Fine-Tuning Screenshot](https://mintcdn.com/tensorzero/afFfERJmQ8YiZpCj/quickstart-sft.png?fit=max&auto=format&n=afFfERJmQ8YiZpCj&q=85&s=6dc06305386c71c96a365041a2f4e771)

We can also send [metrics & feedback](https://www.tensorzero.com/docs/gateway/guides/metrics-feedback) to the TensorZero Gateway.
This data is used to curate better datasets for fine-tuning and other optimization workflows.
Since we haven’t done that yet, the TensorZero UI will skip the curation step before fine-tuning.

## [​](https://www.tensorzero.com/docs/quickstart\#conclusion-&-next-steps)  Conclusion & Next Steps

The Quickstart guide gives a tiny taste of what TensorZero is capable of.We strongly encourage you to check out the guides on [metrics & feedback](https://www.tensorzero.com/docs/gateway/guides/metrics-feedback) and [prompt templates & schemas](https://www.tensorzero.com/docs/gateway/create-a-prompt-template).
Though optional, they unlock many of the downstream features TensorZero offers in experimentation and optimization.From here, you can explore features like built-in support for [inference-time optimizations](https://www.tensorzero.com/docs/gateway/guides/inference-time-optimizations), [retries & fallbacks](https://www.tensorzero.com/docs/gateway/guides/retries-fallbacks), [experimentation (A/B testing) with prompts and models](https://www.tensorzero.com/docs/experimentation/run-adaptive-ab-tests), and a lot more.

[Overview](https://www.tensorzero.com/docs) [Vision & Roadmap](https://www.tensorzero.com/docs/vision-and-roadmap)

Ctrl+I

Assistant

Responses are generated using AI and may contain mistakes.

![TensorZero UI Observability - Function Detail Page - Screenshot](https://mintcdn.com/tensorzero/afFfERJmQ8YiZpCj/quickstart-observability-function.png?w=840&fit=max&auto=format&n=afFfERJmQ8YiZpCj&q=85&s=21a028fefddcc5a5b80ce067876d54d0)

![TensorZero UI Observability - Inference Detail Page - Screenshot](https://mintcdn.com/tensorzero/afFfERJmQ8YiZpCj/quickstart-observability-inference.png?w=840&fit=max&auto=format&n=afFfERJmQ8YiZpCj&q=85&s=28605ca4dc8ef76abd02020e3ce3af9a)

![TensorZero UI Fine-Tuning Screenshot](https://mintcdn.com/tensorzero/afFfERJmQ8YiZpCj/quickstart-sft.png?w=840&fit=max&auto=format&n=afFfERJmQ8YiZpCj&q=85&s=b56b51accfff9e73e61d8f54eda29ffb)

## TensorZero Optimization Recipes
[Skip to main content](https://www.tensorzero.com/docs/recipes#content-area)

[TensorZero Docs home page![light logo](https://mintcdn.com/tensorzero/afFfERJmQ8YiZpCj/logo/light.svg?fit=max&auto=format&n=afFfERJmQ8YiZpCj&q=85&s=9396731a8a92cee7bd293dee05d71dac)![dark logo](https://mintcdn.com/tensorzero/afFfERJmQ8YiZpCj/logo/dark.svg?fit=max&auto=format&n=afFfERJmQ8YiZpCj&q=85&s=8d61c59531dbcdcace7fcdfc524bd4d2)](https://www.tensorzero.com/)

Search...

Ctrl K

Search...

Navigation

Optimization

Overview

[Guides](https://www.tensorzero.com/docs) [Integrations](https://www.tensorzero.com/docs/integrations/model-providers)

On this page

- [Model Optimizations](https://www.tensorzero.com/docs/recipes#model-optimizations)
- [Supervised Fine-tuning](https://www.tensorzero.com/docs/recipes#supervised-fine-tuning)
- [RLHF](https://www.tensorzero.com/docs/recipes#rlhf)
- [DPO (Preference Fine-tuning)](https://www.tensorzero.com/docs/recipes#dpo-preference-fine-tuning)
- [Dynamic In-Context Learning](https://www.tensorzero.com/docs/recipes#dynamic-in-context-learning)
- [Prompt Optimization](https://www.tensorzero.com/docs/recipes#prompt-optimization)
- [MIPRO](https://www.tensorzero.com/docs/recipes#mipro)
- [Inference-Time Optimization](https://www.tensorzero.com/docs/recipes#inference-time-optimization)
- [Custom Recipes](https://www.tensorzero.com/docs/recipes#custom-recipes)
- [Examples](https://www.tensorzero.com/docs/recipes#examples)

TensorZero Recipes are a set of pre-built workflows for optimizing your LLM applications.
You can also create your own recipes to customize the workflow to your needs.The [TensorZero Gateway](https://www.tensorzero.com/docs/gateway) collects structured inference data and the downstream feedback associated with it.
This dataset sets the perfect foundation for building and optimizing LLM applications.
As this dataset builds up, you can use these recipes to generate powerful variants for your functions.
For example, you can use this dataset to curate data to fine-tune a custom LLM, or run an automated prompt engineering workflow.In other words, TensorZero Recipes optimize TensorZero functions by generating new variants from historical inference and feedback data.

## [​](https://www.tensorzero.com/docs/recipes\#model-optimizations)  Model Optimizations

### [​](https://www.tensorzero.com/docs/recipes\#supervised-fine-tuning)  Supervised Fine-tuning

A fine-tuning recipe curates a dataset from your historical inferences and fine-tunes an LLM on it.
You can use the feedback associated with those inferences to select the right subset of data.
A simple example is to use only inferences that led to good outcomes according to a metric you defined.We present sample fine-tuning recipes:

- [Fine-tuning with Fireworks AI](https://github.com/tensorzero/tensorzero/tree/main/recipes/supervised_fine_tuning/fireworks)
- [Fine-tuning with GCP Vertex AI Gemini](https://github.com/tensorzero/tensorzero/tree/main/recipes/supervised_fine_tuning/gcp-vertex-gemini/)
- [Fine-tuning with OpenAI](https://github.com/tensorzero/tensorzero/tree/main/recipes/supervised_fine_tuning/openai)
- [Fine-tuning with Together AI](https://github.com/tensorzero/tensorzero/tree/main/recipes/supervised_fine_tuning/together/)
- [Fine-tuning with Unsloth](https://github.com/tensorzero/tensorzero/tree/main/recipes/supervised_fine_tuning/unsloth/)

See complete examples using the recipes below.

### [​](https://www.tensorzero.com/docs/recipes\#rlhf)  RLHF

#### [​](https://www.tensorzero.com/docs/recipes\#dpo-preference-fine-tuning)  DPO (Preference Fine-tuning)

A direct preference optimization (DPO) — also known as preference fine-tuning — recipe fine-tunes an LLM on a dataset of preference pairs.
You can use demonstration feedback collected with TensorZero to curate a dataset of preference pairs and fine-tune an LLM on it.We present a sample DPO recipe for OpenAI:

- [DPO (Preference Fine-tuning) with OpenAI](https://github.com/tensorzero/tensorzero/blob/main/recipes/dpo/openai/)

### [​](https://www.tensorzero.com/docs/recipes\#dynamic-in-context-learning)  Dynamic In-Context Learning

Dynamic In-Context Learning (DICL) is a technique that leverages historical examples to enhance LLM performance at inference time.
It involves selecting relevant examples from a database of past interactions and including them in the prompt, allowing the model to learn from similar contexts on-the-fly.
This approach can significantly improve the model’s ability to handle specific tasks or domains without the need for fine-tuning.We provide a sample recipe for DICL with OpenAI.
The recipe supports selecting examples based on boolean metrics, float metrics, and demonstrations.

- [Dynamic In-Context Learning with OpenAI](https://github.com/tensorzero/tensorzero/tree/main/recipes/dicl/)

Many more recipes are on the way. This will be our primary engineering focus in the coming months.
We also plan to publish a dashboard that’ll further streamline some of these recipes (e.g. one-click fine-tuning).Read more about our [Vision & Roadmap](https://www.tensorzero.com/docs/vision-and-roadmap).

## [​](https://www.tensorzero.com/docs/recipes\#prompt-optimization)  Prompt Optimization

TensorZero offers a prompt optimization recipe, MIPRO, which jointly optimizes instructions and few-shot examples.
More recipes for prompt optimization are planned.

### [​](https://www.tensorzero.com/docs/recipes\#mipro)  MIPRO

MIPRO (Multi-prompt Instruction PRoposal Optimizer) is a method for automatically improving system instructions and few-shot demonstrations in LLM applications — including ones with multiple LLM functions or calls.![MIPRO Diagram](https://mintcdn.com/tensorzero/afFfERJmQ8YiZpCj/recipes/index-mipro-diagram.svg?fit=max&auto=format&n=afFfERJmQ8YiZpCj&q=85&s=8893ae5d4ffc368ad114c5b9b76092f0)MIPRO can optimize prompts across an entire LLM pipeline without needing fine-grained labels or gradients. Instead, it uses a Bayesian optimizer to figure out which instructions and demonstrations actually improve end-to-end performance. By combining application-aware prompt proposals and stochastic mini-batch evaluations, MIPRO can improve downstream task performance compared to traditional prompt engineering approaches.See [Automated Prompt Engineering with MIPRO](https://github.com/tensorzero/tensorzero/tree/main/recipes/mipro) on GitHub for more details.

## [​](https://www.tensorzero.com/docs/recipes\#inference-time-optimization)  Inference-Time Optimization

The TensorZero Gateway offers built-in inference-time optimizations like dynamic in-context learning and best/mixture-of-N sampling.See [Inference-Time Optimizations](https://www.tensorzero.com/docs/gateway/guides/inference-time-optimizations) for more information.

## [​](https://www.tensorzero.com/docs/recipes\#custom-recipes)  Custom Recipes

You can also create your own recipes.Put simply, a recipe takes inference and feedback data stored that the TensorZero Gateway stored in your ClickHouse database, and generates a new set of variants for your functions.
You should should be able to use virtually any LLM engineering workflow with TensorZero, ranging from automated prompt engineering to advanced RLHF workflows.
See an example of a custom recipe using DSPy below.

## [​](https://www.tensorzero.com/docs/recipes\#examples)  Examples

We are working on a series of **complete runnable examples** illustrating TensorZero’s data & learning flywheel.

- [Optimizing Data Extraction (NER) with TensorZero](https://github.com/tensorzero/tensorzero/tree/main/examples/data-extraction-ner) — This example shows how to use TensorZero to optimize a data extraction pipeline. We demonstrate techniques like fine-tuning and dynamic in-context learning (DICL). In the end, an optimized GPT-4o Mini model outperforms GPT-4o on this task — at a fraction of the cost and latency — using a small amount of training data.
- [Agentic RAG — Multi-Hop Question Answering with LLMs](https://github.com/tensorzero/tensorzero/tree/main/examples/rag-retrieval-augmented-generation/simple-agentic-rag/) — This example shows how to build a multi-hop retrieval agent using TensorZero. The agent iteratively searches Wikipedia to gather information, and decides when it has enough context to answer a complex question.
- [Writing Haikus to Satisfy a Judge with Hidden Preferences](https://github.com/tensorzero/tensorzero/tree/main/examples/haiku-hidden-preferences) — This example fine-tunes GPT-4o Mini to generate haikus tailored to a specific taste. You’ll see TensorZero’s “data flywheel in a box” in action: better variants leads to better data, and better data leads to better variants. You’ll see progress by fine-tuning the LLM multiple times.
- [Image Data Extraction — Multimodal (Vision) Fine-tuning](https://github.com/tensorzero/tensorzero/tree/main/examples/multimodal-vision-finetuning) — This example shows how to fine-tune multimodal models (VLMs) like GPT-4o to improve their performance on vision-language tasks. Specifically, we’ll build a system that categorizes document images (screenshots of computer science research papers).
- [Improving LLM Chess Ability with Best/Mixture-of-N Sampling](https://github.com/tensorzero/tensorzero/tree/main/examples/chess-puzzles/) — This example showcases how best-of-N sampling and mixture-of-N sampling can significantly enhance an LLM’s chess-playing abilities by selecting the most promising moves from multiple generated options.
- [Improving Math Reasoning with a Custom Recipe for Automated Prompt Engineering (DSPy)](https://github.com/tensorzero/tensorzero/tree/main/examples/gsm8k-custom-recipe-dspy) — TensorZero provides a number of pre-built optimization recipes covering common LLM engineering workflows. But you can also easily create your own recipes and workflows! This example shows how to optimize a TensorZero function using an arbitrary tool — here, DSPy.

[Datasets & Datapoints](https://www.tensorzero.com/docs/gateway/api-reference/datasets-datapoints) [Overview](https://www.tensorzero.com/docs/evaluations)

Ctrl+I

Assistant

Responses are generated using AI and may contain mistakes.

## TensorZero Vision & Roadmap
[Skip to main content](https://www.tensorzero.com/docs/vision-and-roadmap#content-area)

[TensorZero Docs home page![light logo](https://mintcdn.com/tensorzero/afFfERJmQ8YiZpCj/logo/light.svg?fit=max&auto=format&n=afFfERJmQ8YiZpCj&q=85&s=9396731a8a92cee7bd293dee05d71dac)![dark logo](https://mintcdn.com/tensorzero/afFfERJmQ8YiZpCj/logo/dark.svg?fit=max&auto=format&n=afFfERJmQ8YiZpCj&q=85&s=8d61c59531dbcdcace7fcdfc524bd4d2)](https://www.tensorzero.com/)

Search...

Ctrl K

Search...

Navigation

Introduction

Vision & Roadmap

[Guides](https://www.tensorzero.com/docs) [Integrations](https://www.tensorzero.com/docs/integrations/model-providers)

On this page

- [Vision](https://www.tensorzero.com/docs/vision-and-roadmap#vision)
- [Near-Term Roadmap](https://www.tensorzero.com/docs/vision-and-roadmap#near-term-roadmap)

## [​](https://www.tensorzero.com/docs/vision-and-roadmap\#vision)  Vision

TensorZero enables a data and learning flywheel for optimizing LLM applications: a feedback loop that turns production metrics and human feedback into smarter, faster, and cheaper models and agents.
Today, we provide an open-source stack for industrial-grade LLM applications that unifies an LLM gateway, observability, optimization, evaluation, and experimentation.
Our vision is to automate much of LLM engineering, and we’re laying the foundation for that with the open-source project.Read more about our vision in our [$7.3M seed round announcement](https://www.tensorzero.com/blog/tensorzero-raises-7-3m-seed-round-to-build-an-open-source-stack-for-industrial-grade-llm-applications/).

## [​](https://www.tensorzero.com/docs/vision-and-roadmap\#near-term-roadmap)  Near-Term Roadmap

TensorZero is under active development.
We ship new features every week.You can see the major areas we’re currently focusing on in [Milestones on GitHub](https://github.com/tensorzero/tensorzero/milestones).For more granularity, see the `priority-high` and `priority-urgent` [Issues on GitHub](https://github.com/tensorzero/tensorzero/issues?q=is%3Aissue+is%3Aopen+label%3Apriority-high%2Cpriority-urgent).We encourage [Feature Requests](https://github.com/tensorzero/tensorzero/discussions/categories/feature-requests) and [Bug Reports](https://github.com/tensorzero/tensorzero/discussions/categories/bug-reports).

[Quickstart](https://www.tensorzero.com/docs/quickstart) [Frequently Asked Questions](https://www.tensorzero.com/docs/faq)

Ctrl+I

Assistant

Responses are generated using AI and may contain mistakes.

## Static A/B Testing
[Skip to main content](https://www.tensorzero.com/docs/experimentation/run-static-ab-tests#content-area)

[TensorZero Docs home page![light logo](https://mintcdn.com/tensorzero/afFfERJmQ8YiZpCj/logo/light.svg?fit=max&auto=format&n=afFfERJmQ8YiZpCj&q=85&s=9396731a8a92cee7bd293dee05d71dac)![dark logo](https://mintcdn.com/tensorzero/afFfERJmQ8YiZpCj/logo/dark.svg?fit=max&auto=format&n=afFfERJmQ8YiZpCj&q=85&s=8d61c59531dbcdcace7fcdfc524bd4d2)](https://www.tensorzero.com/)

Search...

Ctrl K

Search...

Navigation

Experimentation

Run static A/B tests

[Guides](https://www.tensorzero.com/docs) [Integrations](https://www.tensorzero.com/docs/integrations/model-providers)

On this page

- [Configure multiple variants](https://www.tensorzero.com/docs/experimentation/run-static-ab-tests#configure-multiple-variants)
- [Configure sampling weights for variants](https://www.tensorzero.com/docs/experimentation/run-static-ab-tests#configure-sampling-weights-for-variants)
- [Configure fallback-only variants](https://www.tensorzero.com/docs/experimentation/run-static-ab-tests#configure-fallback-only-variants)

You can configure the TensorZero Gateway to distribute inference requests between different variants (prompts, models, etc.) of a function (a “task” or “agent”).
Variants enable you to experiment with different models, prompts, parameters, inference strategies, and more.

We recommend [running adaptive A/B tests](https://www.tensorzero.com/docs/experimentation/run-adaptive-ab-tests) if you have a metric you can optimize for.

## [​](https://www.tensorzero.com/docs/experimentation/run-static-ab-tests\#configure-multiple-variants)  Configure multiple variants

If you specify multiple variants for a function, by default the gateway will sample between them with equal probability (uniform sampling).For example, if you call the `draft_email` function below, the gateway will sample between the two variants at each inference with equal probability.

Copy

```
[functions.draft_email]
type = "chat"

[functions.draft_email.variants.gpt_5_mini]
type = "chat_completion"
model = "openai::gpt-5-mini"

[functions.draft_email.variants.claude_haiku_4_5]
type = "chat_completion"
model = "anthropic::claude-haiku-4-5"
```

During an episode, multiple inference requests to the same function will receive the same variant (unless fallbacks are necessary).
This consistent variant assignment acts as a randomized controlled experiment, providing the statistical foundation needed to make causal inferences about which configurations perform best.

## [​](https://www.tensorzero.com/docs/experimentation/run-static-ab-tests\#configure-sampling-weights-for-variants)  Configure sampling weights for variants

You can configure weights for variants to control the probability of each variant being sampled.
This is particularly useful for canary tests where you want to gradually roll out a new variant to a small percentage of users.

Copy

```
[functions.draft_email]
type = "chat"

[functions.draft_email.variants.gpt_5_mini]
type = "chat_completion"
model = "openai::gpt-5-mini"

[functions.draft_email.variants.claude_haiku_4_5]
type = "chat_completion"
model = "anthropic::claude-haiku-4-5"

[functions.draft_email.experimentation]
type = "static_weights"
candidate_variants = {"gpt_5_mini" = 0.9, "claude_haiku_4_5" = 0.1}
```

In this example, 90% of episodes will be sampled from the `gpt_5_mini` variant and 10% will be sampled from the `claude_haiku_4_5` variant.

If the weights don’t add up to 1, TensorZero will automatically normalize them and sample the variants accordingly.
For example, if a variant has weight 5 and another has weight 1, the first variant will be sampled 5/6 of the time (≈ 83.3%) and the second variant will be sampled 1/6 of the time (≈ 16.7%).

## [​](https://www.tensorzero.com/docs/experimentation/run-static-ab-tests\#configure-fallback-only-variants)  Configure fallback-only variants

You can configure variants that are only used as fallbacks with `fallback_variants`.

Copy

```
[functions.draft_email]
type = "chat"

[functions.draft_email.variants.gpt_5_mini]
type = "chat_completion"
model = "openai::gpt-5-mini"

[functions.draft_email.variants.claude_haiku_4_5]
type = "chat_completion"
model = "anthropic::claude-haiku-4-5"

[functions.draft_email.variants.grok_4]
type = "chat_completion"
model = "xai::grok-4-0709"

[functions.draft_email.experimentation]
type = "static_weights"
candidate_variants = {"gpt_5_mini" = 0.9, "claude_haiku_4_5" = 0.1}
fallback_variants = ["grok_4"]
```

The gateway will first sample among the `candidate_variants`.
If all candidates fail, the gateway attempts each variant in `fallback_variants` in order.
See [Retries & Fallbacks](https://www.tensorzero.com/docs/gateway/guides/retries-fallbacks) for more information.

[Run adaptive A/B tests](https://www.tensorzero.com/docs/experimentation/run-adaptive-ab-tests) [Deploy the TensorZero Gateway](https://www.tensorzero.com/docs/deployment/tensorzero-gateway)

Ctrl+I

Assistant

Responses are generated using AI and may contain mistakes.

