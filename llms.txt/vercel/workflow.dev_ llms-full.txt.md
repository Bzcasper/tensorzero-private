# https://useworkflow.dev/ llms-full.txt

## Durable Async JavaScript
Workflow DevKit is in beta

# Make any TypeScript Function Durable

use workflowbrings durability, reliability, and observability to async JavaScript. Build apps and AI Agents that can suspend, resume, and maintain state with ease.

[Get Started](https://useworkflow.dev/docs/getting-started)

```
npm i workflow
```

## Reliability-as-code

Move from hand-rolled queues and custom retries to durable, resumable code with simple directives.

With Workflow DevKitWith WDKWithout Workflow DevKitWithout WDK

```
export async function welcome(userId: string) {
  "use workflow";
  const user = await getUser(userId);
  const { subject, body } = await generateEmail({
    name: user.name, plan: user.plan
  });
  const { status } = await sendEmail({
    to: user.email,
    subject,
    body,
  });
  return { status, subject, body };
}
```

Running the getUser step...

## Effortless setup

With a simple declarative API to define and use your workflows.

### Creating a workflow

```
import { sleep } from "workflow";
import {
  createUser,
  sendWelcomeEmail,
  sendOneWeekCheckInEmail
} from "./steps"

export async function userSignup(email) {
  "use workflow";

  // Create the user and send the welcome email
  const user = await createUser(email);
  await sendWelcomeEmail(email);

  // Pause for 7 days
  // without consuming any resources
  await sleep("7 days");
  await sendOneWeekCheckInEmail(email);

  return { userId: user.id, status: "done" };
}
```

### Defining steps

```
import { Resend } from 'resend';
import { FatalError } from 'workflow';

export async function sendWelcomeEmail(email) {
  "use step"

  const resend = new Resend('YOUR_API_KEY');

  const resp = await resend.emails.send({
    from: 'Acme <onboarding@resend.dev>',
    to: [email],
    subject: 'Welcome!',
    html: `Thanks for joining Acme.`,
  });

  if (resp.error) {
    throw new FatalError(resp.error.message);
  }
};

// Other steps...
```

## Observability. Inspect every run end‑to‑end. Pause, replay, and time‑travel through steps with traces, logs, and metrics automatically.

workflow()100ms

process()

parse()

transform()

enrich()

validate()

## Universally compatible. Works with the frameworks you already use with more coming soon.

[Next.jsNext.js](https://useworkflow.dev/docs/getting-started/next) [ViteVite](https://useworkflow.dev/docs/getting-started/vite) [HonoHono](https://useworkflow.dev/docs/getting-started/hono) [NitroNitro](https://useworkflow.dev/docs/getting-started/nitro) [SvelteKitSvelteKit](https://useworkflow.dev/docs/getting-started/sveltekit) [NuxtNuxt](https://useworkflow.dev/docs/getting-started/nuxt) [Express](https://useworkflow.dev/docs/getting-started/express)

Coming soonClick on a framework to request support for itClick a framework to request support

NestJSNestJS

### Reliability, minus the plumbing

Start with plain async code. No queues to wire, no schedulers to tune, no YAML. Best‑in‑class DX that compiles reliability into your app with zero config.

### See every step, instantly

Inspect every run end‑to‑end. Pause, replay, and time‑travel through steps with traces, logs, and metrics automatically captured — no extra services or setup.

### A versatile paradigm

Workflows can power a wide array of apps, from streaming realtime agents, to CI/CD pipelines, or multi day email subscriptions workflows.

## Run anywhere, no lock‑in

The same code runs locally on your laptop, in Docker, on Vercel or any other cloud. Open source and portable by design.

DigitalOcean

AWS

```
export async function welcome(userId: string) {
  "use workflow";

  const user = await getUser(userId);
  const { subject, body } = await generateEmail({
    name: user.name, plan: user.plan
  });

  const { status } = await sendEmail({
    to: user.email,
    subject,
    body,
  });

  return { status, subject, body };
}
```

Docker

Vercel

## Build anything withAI Agents

Build reliable, long-running processes with automatic retries, state persistence, and observability built in.

```
export async function aiAgentWorkflow(query: string) {
  "use workflow";

  // Step 1: Generate initial response
  const response = await generateResponse(query);

  // Step 2: Research and validate
  const facts = await researchFacts(response);

  // Step 3: Refine with fact-checking
  const refined = await refineWithFacts(response, facts);

  return { response: refined, sources: facts };
}
```

## Get started quickly

See Workflow DevKit in action with one of our templates.

[**Story Generator Slack Bot** \\
Slackbot that generates children's stories from collaborative input.\\
![Story Generator Slack Bot](https://useworkflow.dev/_next/image?url=%2F_next%2Fstatic%2Fmedia%2Fstorytime.ca77a370.png&w=1920&q=75&dpl=dpl_3rWCNfNCfRctqxrpWpgFxVoEstdN)](https://vercel.com/guides/stateful-slack-bots-with-vercel-workflow) [**Flight Booking App** \\
Use Workflow to make AI agents more reliable and production-ready.\\
![Flight Booking App](https://useworkflow.dev/_next/image?url=%2F_next%2Fstatic%2Fmedia%2Fflight.5cbd4e8b.png&w=1920&q=75&dpl=dpl_3rWCNfNCfRctqxrpWpgFxVoEstdN)](https://github.com/vercel/workflow-examples/tree/main/flight-booking-app) [**Natural Language Image Search** \\
A free, open-source template for building natural language image search.\\
![Natural Language Image Search](https://useworkflow.dev/_next/image?url=%2F_next%2Fstatic%2Fmedia%2Fvectr.5b2641cc.png&w=1920&q=75&dpl=dpl_3rWCNfNCfRctqxrpWpgFxVoEstdN)](https://www.vectr.store/)

## Create your first workflow today.

[Get started](https://useworkflow.dev/docs/getting-started)

## Chat

What is Workflow?How does retrying work?What control flow patterns are there?How do directives work?

Tip: You can open and close chat with `⌘I`

0 / 1000

## RetryableError in Workflows
[API Reference](https://useworkflow.dev/docs/api-reference) [workflow](https://useworkflow.dev/docs/api-reference/workflow)

# RetryableError

When a `RetryableError` is thrown in a step, it indicates that the workflow should retry a step. Additionally, it contains a parameter `retryAfter` indicating when the step should be retried after.

You should use this when you want to retry a step or retry after a certain duration.

```
import { RetryableError } from "workflow"

async function retryableWorkflow() {
    "use workflow"
    await retryStep();
}

async function retryStep() {
    "use step"
    throw new RetryableError("Retryable!")
}
```

The difference between `Error` and `RetryableError` may not be entirely obvious, since when both are thrown, they both retry. The difference is that `RetryableError` has an additional configurable `retryAfter` parameter.

## [API Signature](https://useworkflow.dev/docs/api-reference/workflow/retryable-error\#api-signature)

### [Parameters](https://useworkflow.dev/docs/api-reference/workflow/retryable-error\#parameters)

| Name | Type | Description |
| --- | --- | --- |
| `options` | RetryableErrorOptions |  |
| `message` | string |  |

#### [RetryableErrorOptions](https://useworkflow.dev/docs/api-reference/workflow/retryable-error\#retryableerroroptions)

| Name | Type | Description |
| --- | --- | --- |
| `retryAfter` | number \| StringValue \| Date | The number of milliseconds to wait before retrying the step.<br>Can also be a duration string (e.g., "5s", "2m") or a Date object.<br>If not provided, the step will be retried after 1 second (1000 milliseconds). |

## [Examples](https://useworkflow.dev/docs/api-reference/workflow/retryable-error\#examples)

### [Retrying after a duration](https://useworkflow.dev/docs/api-reference/workflow/retryable-error\#retrying-after-a-duration)

`RetryableError` can be configured with a `retryAfter` parameter to specify when the step should be retried after.

```
import { RetryableError } from "workflow"

async function retryableWorkflow() {
    "use workflow"
    await retryStep();
}

async function retryStep() {
    "use step"
    throw new RetryableError("Retryable!", {
        retryAfter: "5m" // - supports "5m", "30s", "1h", etc.
    })
}
```

You can also specify the retry delay in milliseconds:

```
import { RetryableError } from "workflow"

async function retryableWorkflow() {
    "use workflow"
    await retryStep();
}

async function retryStep() {
    "use step"
    throw new RetryableError("Retryable!", {
        retryAfter: 5000 // - 5000 milliseconds = 5 seconds
    })
}
```

Or retry at a specific date and time:

```
import { RetryableError } from "workflow"

async function retryableWorkflow() {
    "use workflow"
    await retryStep();
}

async function retryStep() {
    "use step"
    throw new RetryableError("Retryable!", {
        retryAfter: new Date(Date.now() + 60000) // - retry after 1 minute
    })
}
```

[FatalError\\
\\
Previous Page](https://useworkflow.dev/docs/api-reference/workflow/fatal-error) [workflow/api\\
\\
Next Page](https://useworkflow.dev/docs/api-reference/workflow-api)

On this page

[API Signature](https://useworkflow.dev/docs/api-reference/workflow/retryable-error#api-signature) [Parameters](https://useworkflow.dev/docs/api-reference/workflow/retryable-error#parameters) [RetryableErrorOptions](https://useworkflow.dev/docs/api-reference/workflow/retryable-error#retryableerroroptions) [Examples](https://useworkflow.dev/docs/api-reference/workflow/retryable-error#examples) [Retrying after a duration](https://useworkflow.dev/docs/api-reference/workflow/retryable-error#retrying-after-a-duration)

[GitHubEdit this page on GitHub](https://github.com/vercel/workflow/edit/main/content/docs/api-reference/workflow/retryable-error.mdx) Scroll to topGive feedbackCopy pageAsk AI about this pageOpen in chat

## Chat

What is Workflow?How does retrying work?What control flow patterns are there?How do directives work?

Tip: You can open and close chat with `⌘I`

0 / 1000

## Node.js Module Errors
[Errors](https://useworkflow.dev/docs/errors)

# node-js-module-in-workflow

This error occurs when you try to import or use Node.js core modules (like `fs`, `http`, `crypto`, `path`, etc.) directly inside a workflow function.

## [Error Message](https://useworkflow.dev/docs/errors/node-js-module-in-workflow\#error-message)

```
Cannot use Node.js module "fs" in workflow functions. Move this module to a step function.
```

## [Why This Happens](https://useworkflow.dev/docs/errors/node-js-module-in-workflow\#why-this-happens)

Workflow functions run in a sandboxed environment without full Node.js runtime access. This restriction is important for maintaining **determinism** \- the ability to replay workflows exactly and resume from where they left off after suspensions or failures.

Node.js modules have side effects and non-deterministic behavior that could break workflow replay guarantees.

## [Quick Fix](https://useworkflow.dev/docs/errors/node-js-module-in-workflow\#quick-fix)

Move any code using Node.js modules to a step function. Step functions have full Node.js runtime access.

For example, when trying to read a file in a workflow function, you should move the code to a step function.

**Before:**

```
import * as fs from 'fs';

export async function processFileWorkflow(filePath: string) {
  "use workflow";

  // This will cause an error - Node.js module in workflow context
  const content = fs.readFileSync(filePath, 'utf-8');
  return content;
}
```

**After:**

```
import * as fs from 'fs';

export async function processFileWorkflow(filePath: string) {
  "use workflow";

  // Call step function that has Node.js access
  const content = await read(filePath);
  return content;
}

async function read(filePath: string) {
  "use step";

  // Node.js modules are allowed in step functions
  return fs.readFileSync(filePath, 'utf-8');
}
```

## [Common Node.js Modules](https://useworkflow.dev/docs/errors/node-js-module-in-workflow\#common-nodejs-modules)

These common Node.js core modules cannot be used in workflow functions:

- File system: `fs`, `path`
- Network: `http`, `https`, `net`, `dns`, `fetch`
- Process: `child_process`, `cluster`
- Crypto: `crypto` (use Web Crypto API instead)
- Operating system: `os`
- Streams: `stream` (use Web Streams API instead)

You can use Web Platform APIs in workflow functions (like `Headers`, `crypto.randomUUID()`, `Response`, etc.), since these are available in the sandboxed environment.

[fetch-in-workflow\\
\\
Previous Page](https://useworkflow.dev/docs/errors/fetch-in-workflow) [serialization-failed\\
\\
Next Page](https://useworkflow.dev/docs/errors/serialization-failed)

On this page

[Error Message](https://useworkflow.dev/docs/errors/node-js-module-in-workflow#error-message) [Why This Happens](https://useworkflow.dev/docs/errors/node-js-module-in-workflow#why-this-happens) [Quick Fix](https://useworkflow.dev/docs/errors/node-js-module-in-workflow#quick-fix) [Common Node.js Modules](https://useworkflow.dev/docs/errors/node-js-module-in-workflow#common-nodejs-modules)

[GitHubEdit this page on GitHub](https://github.com/vercel/workflow/edit/main/content/docs/errors/node-js-module-in-workflow.mdx) Scroll to topGive feedbackCopy pageAsk AI about this pageOpen in chat

## Chat

What is Workflow?How does retrying work?What control flow patterns are there?How do directives work?

Tip: You can open and close chat with `⌘I`

0 / 1000

## Workflow Chat Transport
[API Reference](https://useworkflow.dev/docs/api-reference) [@workflow/ai](https://useworkflow.dev/docs/api-reference/workflow-ai)

# WorkflowChatTransport

The `@workflow/ai` package is currently in active development and should be considered experimental.

A chat transport implementation for the AI SDK that provides reliable message streaming with automatic reconnection to interrupted streams. This transport is a drop-in replacement for the default AI SDK transport, enabling seamless recovery from network issues, page refreshes, or Vercel Function timeouts.

`WorkflowChatTransport` implements the [`ChatTransport`](https://ai-sdk.dev/docs/ai-sdk-ui/transport) interface from the AI SDK and is designed to work with workflow-based chat applications. It requires endpoints that return the `x-workflow-run-id` header to enable stream resumption.

```
import { useChat } from '@ai-sdk/react';
import { WorkflowChatTransport } from '@workflow/ai';

export default function Chat() {
  const { messages, sendMessage } = useChat({
    transport: new WorkflowChatTransport(),
  });

  return (
    <div>
      {messages.map((m) => (
        <div key={m.id}>{m.content}</div>
      ))}
    </div>
  );
}
```

## [API Signature](https://useworkflow.dev/docs/api-reference/workflow-ai/workflow-chat-transport\#api-signature)

### [Class](https://useworkflow.dev/docs/api-reference/workflow-ai/workflow-chat-transport\#class)

| Name | Type | Description |
| --- | --- | --- |
| `api` | any |  |
| `fetch` | any |  |
| `onChatSendMessage` | any |  |
| `onChatEnd` | any |  |
| `maxConsecutiveErrors` | any |  |
| `prepareSendMessagesRequest` | any |  |
| `prepareReconnectToStreamRequest` | any |  |
| `sendMessages` | (options: SendMessagesOptions<UI\_MESSAGE> & ChatRequestOptions) => Promise<ReadableStream<UIMessageChunk>> | Sends messages to the chat endpoint and returns a stream of response chunks.<br>This method handles the entire chat lifecycle including:<br>\- Sending messages to the /api/chat endpoint<br>\- Streaming response chunks<br>\- Automatic reconnection if the stream is interrupted |
| `sendMessagesIterator` | any |  |
| `reconnectToStream` | (options: ReconnectToStreamOptions & ChatRequestOptions) => Promise<ReadableStream<UIMessageChunk> \| null> | Reconnects to an existing chat stream that was previously interrupted.<br>This method is useful for resuming a chat session after network issues,<br>page refreshes, or Vercel Function timeouts. |
| `reconnectToStreamIterator` | any |  |
| `onFinish` | any |  |

### [WorkflowChatTransportOptions](https://useworkflow.dev/docs/api-reference/workflow-ai/workflow-chat-transport\#workflowchattransportoptions)

| Name | Type | Description |
| --- | --- | --- |
| `api` | string | API endpoint for chat requests<br>Defaults to /api/chat if not provided |
| `fetch` | { (input: RequestInfo \| URL, init?: RequestInit \| undefined): Promise<Response>; (input: string \| Request \| URL, init?: RequestInit \| undefined): Promise<...>; } | Custom fetch implementation to use for HTTP requests.<br>Defaults to the global fetch function if not provided. |
| `onChatSendMessage` | OnChatSendMessage<UI\_MESSAGE> | Callback invoked after successfully sending messages to the chat endpoint.<br>Useful for tracking chat history and inspecting response headers. |
| `onChatEnd` | OnChatEnd | Callback invoked when a chat stream ends (receives a "finish" chunk).<br>Useful for cleanup operations or state updates. |
| `maxConsecutiveErrors` | number | Maximum number of consecutive errors allowed during reconnection attempts.<br>Defaults to 3 if not provided. |
| `prepareSendMessagesRequest` | PrepareSendMessagesRequest<UI\_MESSAGE> | Function to prepare the request for sending messages.<br>Allows customizing the API endpoint, headers, credentials, and body. |
| `prepareReconnectToStreamRequest` | PrepareReconnectToStreamRequest | Function to prepare the request for reconnecting to a stream.<br>Allows customizing the API endpoint, headers, and credentials. |

## [Key Features](https://useworkflow.dev/docs/api-reference/workflow-ai/workflow-chat-transport\#key-features)

- **Automatic Reconnection**: Automatically recovers from interrupted streams with configurable retry limits
- **Workflow Integration**: Seamlessly works with workflow-based endpoints that provide the `x-workflow-run-id` header
- **Customizable Requests**: Allows intercepting and modifying requests via `prepareSendMessagesRequest` and `prepareReconnectToStreamRequest`
- **Stream Callbacks**: Provides hooks for tracking chat lifecycle via `onChatSendMessage` and `onChatEnd`
- **Custom Fetch**: Supports custom fetch implementations for advanced use cases

## [Good to Know](https://useworkflow.dev/docs/api-reference/workflow-ai/workflow-chat-transport\#good-to-know)

- The transport expects chat endpoints to return the `x-workflow-run-id` header in the response to enable stream resumption
- By default, the transport posts to `/api/chat` and reconnects via `/api/chat/{runId}/stream`
- The `onChatSendMessage` callback receives the full response object, allowing you to extract and store the workflow run ID for session resumption
- Stream interruptions are automatically detected when a "finish" chunk is not received in the initial response
- The `maxConsecutiveErrors` option controls how many reconnection attempts are made before giving up (default: 3)

## [Examples](https://useworkflow.dev/docs/api-reference/workflow-ai/workflow-chat-transport\#examples)

### [Basic Chat Setup](https://useworkflow.dev/docs/api-reference/workflow-ai/workflow-chat-transport\#basic-chat-setup)

```
'use client';

import { useChat } from '@ai-sdk/react';
import { WorkflowChatTransport } from '@workflow/ai';
import { useState } from 'react';

export default function BasicChat() {
  const [input, setInput] = useState('');
  const { messages, sendMessage } = useChat({
    transport: new WorkflowChatTransport(),
  });

  return (
    <div>
      <div className="space-y-4">
        {messages.map((m) => (
          <div key={m.id}>
            <strong>{m.role}:</strong> {m.content}
          </div>
        ))}
      </div>

      <form
        onSubmit={(e) => {
          e.preventDefault();
          sendMessage({ text: input });
          setInput('');
        }}
      >
        <input
          value={input}
          placeholder="Say something..."
          onChange={(e) => setInput(e.currentTarget.value)}
        />
      </form>
    </div>
  );
}
```

### [With Session Persistence and Resumption](https://useworkflow.dev/docs/api-reference/workflow-ai/workflow-chat-transport\#with-session-persistence-and-resumption)

```
'use client';

import { useChat } from '@ai-sdk/react';
import { WorkflowChatTransport } from '@workflow/ai';
import { useMemo, useState } from 'react';

export default function ChatWithResumption() {
  const [input, setInput] = useState('');
  const activeWorkflowRunId = useMemo(() => {
    if (typeof window === 'undefined') return;
    return localStorage.getItem('active-workflow-run-id') ?? undefined;
  }, []);

  const { messages, sendMessage } = useChat({
    resume: !!activeWorkflowRunId,
    transport: new WorkflowChatTransport({
      onChatSendMessage: (response, options) => {
        // Save chat history to localStorage
        localStorage.setItem(
          'chat-history',
          JSON.stringify(options.messages)
        );

        // Extract and store the workflow run ID for session resumption
        const workflowRunId = response.headers.get('x-workflow-run-id');
        if (workflowRunId) {
          localStorage.setItem('active-workflow-run-id', workflowRunId);
        }
      },
      onChatEnd: ({ chatId, chunkIndex }) => {
        console.log(`Chat ${chatId} completed with ${chunkIndex} chunks`);
        // Clear the active run ID when chat completes
        localStorage.removeItem('active-workflow-run-id');
      },
    }),
  });

  return (
    <div>
      <div className="space-y-4">
        {messages.map((m) => (
          <div key={m.id}>
            <strong>{m.role}:</strong> {m.content}
          </div>
        ))}
      </div>

      <form
        onSubmit={(e) => {
          e.preventDefault();
          sendMessage({ text: input });
          setInput('');
        }}
      >
        <input
          value={input}
          placeholder="Say something..."
          onChange={(e) => setInput(e.currentTarget.value)}
        />
      </form>
    </div>
  );
}
```

### [With Custom Request Configuration](https://useworkflow.dev/docs/api-reference/workflow-ai/workflow-chat-transport\#with-custom-request-configuration)

```
'use client';

import { useChat } from '@ai-sdk/react';
import { WorkflowChatTransport } from '@workflow/ai';
import { useState } from 'react';

export default function ChatWithCustomConfig() {
  const [input, setInput] = useState('');
  const { messages, sendMessage } = useChat({
    transport: new WorkflowChatTransport({
      prepareSendMessagesRequest: async (config) => {
        return {
          ...config,
          api: '/api/chat',
          headers: {
            ...config.headers,
            'Authorization': `Bearer ${process.env.NEXT_PUBLIC_API_TOKEN}`,
            'X-Custom-Header': 'custom-value',
          },
          credentials: 'include',
        };
      },
      prepareReconnectToStreamRequest: async (config) => {
        return {
          ...config,
          headers: {
            ...config.headers,
            'Authorization': `Bearer ${process.env.NEXT_PUBLIC_API_TOKEN}`,
          },
          credentials: 'include',
        };
      },
      maxConsecutiveErrors: 5,
    }),
  });

  return (
    <div>
      <div className="space-y-4">
        {messages.map((m) => (
          <div key={m.id}>
            <strong>{m.role}:</strong> {m.content}
          </div>
        ))}
      </div>

      <form
        onSubmit={(e) => {
          e.preventDefault();
          sendMessage({ text: input });
          setInput('');
        }}
      >
        <input
          value={input}
          placeholder="Say something..."
          onChange={(e) => setInput(e.currentTarget.value)}
        />
      </form>
    </div>
  );
}
```

## [See Also](https://useworkflow.dev/docs/api-reference/workflow-ai/workflow-chat-transport\#see-also)

- [DurableAgent](https://useworkflow.dev/docs/api-reference/workflow-ai/durable-agent) \- Building durable AI agents within workflows
- [AI SDK `useChat` Documentation](https://ai-sdk.dev/docs/reference/ai-sdk-ui/use-chat) \- Using `useChat` with custom transports
- [Workflows and Steps](https://useworkflow.dev/docs/foundations/workflows-and-steps) \- Understanding workflow fundamentals
- ["flight-booking-app" Example](https://github.com/vercel/workflow-examples/tree/main/flight-booking-app) \- An example application which uses `WorkflowChatTransport`

[DurableAgent\\
\\
Previous Page](https://useworkflow.dev/docs/api-reference/workflow-ai/durable-agent)

On this page

[API Signature](https://useworkflow.dev/docs/api-reference/workflow-ai/workflow-chat-transport#api-signature) [Class](https://useworkflow.dev/docs/api-reference/workflow-ai/workflow-chat-transport#class) [WorkflowChatTransportOptions](https://useworkflow.dev/docs/api-reference/workflow-ai/workflow-chat-transport#workflowchattransportoptions) [Key Features](https://useworkflow.dev/docs/api-reference/workflow-ai/workflow-chat-transport#key-features) [Good to Know](https://useworkflow.dev/docs/api-reference/workflow-ai/workflow-chat-transport#good-to-know) [Examples](https://useworkflow.dev/docs/api-reference/workflow-ai/workflow-chat-transport#examples) [Basic Chat Setup](https://useworkflow.dev/docs/api-reference/workflow-ai/workflow-chat-transport#basic-chat-setup) [With Session Persistence and Resumption](https://useworkflow.dev/docs/api-reference/workflow-ai/workflow-chat-transport#with-session-persistence-and-resumption) [With Custom Request Configuration](https://useworkflow.dev/docs/api-reference/workflow-ai/workflow-chat-transport#with-custom-request-configuration) [See Also](https://useworkflow.dev/docs/api-reference/workflow-ai/workflow-chat-transport#see-also)

[GitHubEdit this page on GitHub](https://github.com/vercel/workflow/edit/main/content/docs/api-reference/workflow-ai/workflow-chat-transport.mdx) Scroll to topGive feedbackCopy pageAsk AI about this pageOpen in chat

## Chat

What is Workflow?How does retrying work?What control flow patterns are there?How do directives work?

Tip: You can open and close chat with `⌘I`

0 / 1000

## Fetch API in Workflows
[API Reference](https://useworkflow.dev/docs/api-reference) [workflow](https://useworkflow.dev/docs/api-reference/workflow)

# fetch

Makes HTTP requests from within a workflow. This is a special step function that wraps the standard `fetch` API, automatically handling serialization and providing retry semantics.

This is useful when you need to call external APIs or services from within your workflow.

`fetch` is a _special_ type of step function provided and should be called directly inside workflow functions.

```
import { fetch } from "workflow"

async function apiWorkflow() {
    "use workflow"

    // Fetch data from an API
    const response = await fetch("https://api.example.com/data")
    return await response.json()
}
```

## [API Signature](https://useworkflow.dev/docs/api-reference/workflow/fetch\#api-signature)

### [Parameters](https://useworkflow.dev/docs/api-reference/workflow/fetch\#parameters)

Accepts the same arguments as web [`fetch`](https://developer.mozilla.org/en-US/docs/Web/API/Window/fetch)

| Name | Type | Description |
| --- | --- | --- |
| `args` | \[input: string \| URL \| Request, init?: RequestInit \| undefined\] |  |

### [Returns](https://useworkflow.dev/docs/api-reference/workflow/fetch\#returns)

Returns the same response as web [`fetch`](https://developer.mozilla.org/en-US/docs/Web/API/Window/fetch)

`Promise<Response>`

## [Examples](https://useworkflow.dev/docs/api-reference/workflow/fetch\#examples)

### [Basic Usage](https://useworkflow.dev/docs/api-reference/workflow/fetch\#basic-usage)

Here's a simple example of how you can use `fetch` inside your workflow.

```
import { fetch } from "workflow"

async function apiWorkflow() {
    "use workflow"

    // Fetch data from an API
    const response = await fetch("https://api.example.com/data")
    const data = await response.json()

    // Make a POST request
    const postResponse = await fetch("https://api.example.com/create", {
        method: "POST",
        headers: {
            "Content-Type": "application/json"
        },
        body: JSON.stringify({ name: "test" })
    })

    return data
}
```

We call `fetch()` with a URL and optional request options, just like the standard fetch API. The workflow runtime automatically handles the response serialization.

This API is provided as a convenience to easily use `fetch` in workflow, but often, you might want to extend and implement your own fetch for more powerful error handing and retry logic.

### [Customizing Fetch Behavior](https://useworkflow.dev/docs/api-reference/workflow/fetch\#customizing-fetch-behavior)

Here's an example of a custom fetch wrapper that provides more sophisticated error handling with custom retry logic:

```
import { FatalError, RetryableError } from "workflow"

export async function customFetch(
    url: string,
    init?: RequestInit
) {
    "use step"

    const response = await fetch(url, init)

    // Handle client errors (4xx) - don't retry
    if (response.status >= 400 && response.status < 500) {
        if (response.status === 429) {
            // Rate limited - retry with backoff from Retry-After header
            const retryAfter = response.headers.get("Retry-After")

            if (retryAfter) {
                // The Retry-After header is either a number (seconds) or an RFC 7231 date string
                const retryAfterValue = /^\d+$/.test(retryAfter)
                    ? parseInt(retryAfter) * 1000  // Convert seconds to milliseconds
                    : new Date(retryAfter);        // Parse RFC 7231 date format

                // Use `RetryableError` to customize the retry
                throw new RetryableError(
                    `Rate limited by ${url}`,
                    { retryAfter: retryAfterValue }
                )
            }
        }

        // Other client errors are fatal (400, 401, 403, 404, etc.)
        throw new FatalError(
            `Client error ${response.status}: ${response.statusText}`
        )
    }

    // Handle server errors (5xx) - will retry automatically
    if (!response.ok) {
        throw new Error(
            `Server error ${response.status}: ${response.statusText}`
        )
    }

    return response
}
```

This example demonstrates:

- Setting custom `maxRetries` to 5 attempts.
- Throwing [`FatalError`](https://useworkflow.dev/docs/api-reference/workflow/fatal-error) for client errors (400-499) to prevent retries.
- Handling 429 rate limiting by reading the `Retry-After` header and using [`RetryableError`](https://useworkflow.dev/docs/api-reference/workflow/retryable-error).
- Allowing automatic retries for server errors (5xx).

[defineHook\\
\\
Previous Page](https://useworkflow.dev/docs/api-reference/workflow/define-hook) [getStepMetadata\\
\\
Next Page](https://useworkflow.dev/docs/api-reference/workflow/get-step-metadata)

On this page

[API Signature](https://useworkflow.dev/docs/api-reference/workflow/fetch#api-signature) [Parameters](https://useworkflow.dev/docs/api-reference/workflow/fetch#parameters) [Returns](https://useworkflow.dev/docs/api-reference/workflow/fetch#returns) [Examples](https://useworkflow.dev/docs/api-reference/workflow/fetch#examples) [Basic Usage](https://useworkflow.dev/docs/api-reference/workflow/fetch#basic-usage) [Customizing Fetch Behavior](https://useworkflow.dev/docs/api-reference/workflow/fetch#customizing-fetch-behavior)

[GitHubEdit this page on GitHub](https://github.com/vercel/workflow/edit/main/content/docs/api-reference/workflow/fetch.mdx) Scroll to topGive feedbackCopy pageAsk AI about this pageOpen in chat

## Chat

What is Workflow?How does retrying work?What control flow patterns are there?How do directives work?

Tip: You can open and close chat with `⌘I`

0 / 1000

## Streaming Workflows
[Foundations](https://useworkflow.dev/docs/foundations)

# Streaming

Workflows can stream data in real-time to clients without waiting for the entire workflow to complete. This enables progress updates, AI-generated content, log messages, and other incremental data to be delivered as workflows execute.

## [Getting Started with `getWritable()`](https://useworkflow.dev/docs/foundations/streaming\#getting-started-with-getwritable)

Every workflow run has a default writable stream that steps can write to using [`getWritable()`](https://useworkflow.dev/docs/api-reference/workflow/get-writable). Data written to this stream becomes immediately available to clients consuming the workflow's output.

workflows/simple-streaming.ts

```

```

### [Consuming the Stream](https://useworkflow.dev/docs/foundations/streaming\#consuming-the-stream)

Use the `Run` object's `readable` property to consume the stream from your API route:

app/api/stream/route.ts

```

```

When a client makes a request to this endpoint, they'll receive each message as it's written, without waiting for the workflow to complete.

### [Resuming Streams from a Specific Point](https://useworkflow.dev/docs/foundations/streaming\#resuming-streams-from-a-specific-point)

Use `run.getReadable({ startIndex })` to resume a stream from a specific position. This is useful for reconnecting after timeouts or network interruptions:

app/api/resume-stream/\[runId\]/route.ts

```

```

This allows clients to reconnect and continue receiving data from where they left off, rather than restarting from the beginning.

## [Streams as Data Types](https://useworkflow.dev/docs/foundations/streaming\#streams-as-data-types)

[`ReadableStream`](https://developer.mozilla.org/en-US/docs/Web/API/ReadableStream) and [`WritableStream`](https://developer.mozilla.org/en-US/docs/Web/API/WritableStream) are standard Web Streams API types that Workflow DevKit makes serializable. These are not custom types - they follow the web standard - but Workflow DevKit adds the ability to pass them between functions while maintaining their streaming capabilities.

Unlike regular values that are fully serialized to the event log, streams maintain their streaming capabilities when passed between functions.

**Key properties:**

- Stream references can be passed between workflow and step functions
- Stream data flows directly without being stored in the event log
- Streams preserve their state across workflow suspension points

**How Streams Persist Across Workflow Suspensions**

Streams in Workflow DevKit are backed by persistent, resumable storage provided by the "world" implementation. This is what enables streams to maintain their state even when workflows suspend and resume:

- **Vercel deployments**: Streams are backed by a performant Redis-based stream
- **Local development**: Stream chunks are stored in the filesystem

### [Passing Streams as Arguments](https://useworkflow.dev/docs/foundations/streaming\#passing-streams-as-arguments)

Since streams are serializable data types, you don't need to use the special [`getWritable()`](https://useworkflow.dev/docs/api-reference/workflow/get-writable). You can even wire your own streams through workflows, passing them as arguments from outside into steps.

Here's an example of passing a request body stream through a workflow to a step that processes it:

app/api/upload/route.ts

```

```

workflows/streaming.ts

```

```

## [Important Limitation](https://useworkflow.dev/docs/foundations/streaming\#important-limitation)

**Streams Cannot Be Used Directly in Workflow Context**

You cannot read from or write to streams directly within a workflow function. All stream operations must happen in step functions.

Workflow functions must be deterministic to support replay. Since streams bypass the event log for performance, reading stream data in a workflow would break determinism - each replay could see different data. By requiring all stream operations to happen in steps, the framework ensures consistent behavior.

For more on determinism and replay, see [Workflows and Steps](https://useworkflow.dev/docs/foundations/workflows-and-steps).

workflows/bad-example.ts

```

```

workflows/good-example.ts

```

```

## [Namespaced Streams](https://useworkflow.dev/docs/foundations/streaming\#namespaced-streams)

Use `getWritable({ namespace: 'name' })` to create multiple independent streams for different types of data. This is useful when you want to separate logs, metrics, data outputs, or other distinct channels.

workflows/multi-stream.ts

```

```

### [Consuming Namespaced Streams](https://useworkflow.dev/docs/foundations/streaming\#consuming-namespaced-streams)

Use `run.getReadable({ namespace: 'name' })` to access specific streams:

app/api/multi-stream/route.ts

```

```

## [Common Patterns](https://useworkflow.dev/docs/foundations/streaming\#common-patterns)

### [Progress Updates for Long-Running Tasks](https://useworkflow.dev/docs/foundations/streaming\#progress-updates-for-long-running-tasks)

Send incremental progress updates to keep users informed during lengthy workflows:

workflows/batch-processing.ts

```

```

### [Streaming AI Responses with `DurableAgent`](https://useworkflow.dev/docs/foundations/streaming\#streaming-ai-responses-with-durableagent)

Stream AI-generated content using [`DurableAgent`](https://useworkflow.dev/docs/api-reference/workflow-ai/durable-agent) from `@workflow/ai`. Tools can also emit progress updates to the same stream using [data chunks](https://ai-sdk.dev/docs/ai-sdk-ui/streaming-data#streaming-custom-data) with the [`UIMessageChunk`](https://ai-sdk.dev/docs/ai-sdk-ui/stream-protocol) type from the AI SDK:

workflows/ai-assistant.ts

```

```

app/api/ai-assistant/route.ts

```

```

For a complete implementation, see the [flight booking example](https://github.com/vercel/workflow-examples/tree/main/flight-booking-app) which demonstrates streaming AI responses with tool progress updates.

### [Streaming Between Steps](https://useworkflow.dev/docs/foundations/streaming\#streaming-between-steps)

One step produces a stream and another step consumes it:

workflows/stream-pipeline.ts

```

```

### [Processing Large Files Without Memory Overhead](https://useworkflow.dev/docs/foundations/streaming\#processing-large-files-without-memory-overhead)

Process large files by streaming chunks through transformation steps:

workflows/file-processing.ts

```

```

## [Best Practices](https://useworkflow.dev/docs/foundations/streaming\#best-practices)

**Release locks properly:**

```
const writer = writable.getWriter();
try {
  await writer.write(data);
} finally {
  writer.releaseLock(); // Always release
}
```

Stream locks acquired in a step only apply within that step, not across other steps. This enables multiple writers to write to the same stream concurrently.

If a lock is not released, the step process cannot terminate. Even though the step returns and the workflow continues, the underlying process will remain active until it times out.

**Close streams when done:**

```
async function finalizeStream() {
  "use step";

  await getWritable().close(); // Signal completion
}
```

Streams are automatically closed when the workflow run completes, but explicitly closing them signals completion to consumers earlier.

**Use typed streams for type safety:**

```
const writable = getWritable<MyDataType>();
const writer = writable.getWriter();
await writer.write({ /* typed data */ });
```

## [Stream Failures](https://useworkflow.dev/docs/foundations/streaming\#stream-failures)

When a step returns a stream, the step is considered successful once it returns, even if the stream later encounters an error. The workflow won't automatically retry the step. The consumer of the stream must handle errors gracefully. For more on retry behavior, see [Errors and Retries](https://useworkflow.dev/docs/foundations/errors-and-retries).

workflows/stream-error-handling.ts

```

```

Stream errors don't trigger automatic retries for the producer step. Design your stream consumers to handle errors appropriately. Since the stream is already in an errored state, retrying the consumer won't help - use `FatalError` to fail the workflow immediately.

## [Related Documentation](https://useworkflow.dev/docs/foundations/streaming\#related-documentation)

- [`getWritable()` API Reference](https://useworkflow.dev/docs/api-reference/workflow/get-writable) \- Get the workflow's writable stream
- [`sleep()` API Reference](https://useworkflow.dev/docs/api-reference/workflow/sleep) \- Pause workflow execution for a duration
- [`start()` API Reference](https://useworkflow.dev/docs/api-reference/workflow-api/start) \- Start workflows and access the `Run` object
- [`getRun()` API Reference](https://useworkflow.dev/docs/api-reference/workflow-api/get-run) \- Retrieve runs and their streams later
- [DurableAgent](https://useworkflow.dev/docs/api-reference/workflow-ai/durable-agent) \- AI agents with built-in streaming support
- [Errors and Retries](https://useworkflow.dev/docs/foundations/errors-and-retries) \- Understanding error handling and retry behavior
- [Serialization](https://useworkflow.dev/docs/foundations/serialization) \- Understanding what data types can be passed in workflows
- [Workflows and Steps](https://useworkflow.dev/docs/foundations/workflows-and-steps) \- Core concepts of workflow execution

[Hooks & Webhooks\\
\\
Previous Page](https://useworkflow.dev/docs/foundations/hooks) [Serialization\\
\\
Next Page](https://useworkflow.dev/docs/foundations/serialization)

On this page

[Getting Started with `getWritable()`](https://useworkflow.dev/docs/foundations/streaming#getting-started-with-getwritable) [Consuming the Stream](https://useworkflow.dev/docs/foundations/streaming#consuming-the-stream) [Resuming Streams from a Specific Point](https://useworkflow.dev/docs/foundations/streaming#resuming-streams-from-a-specific-point) [Streams as Data Types](https://useworkflow.dev/docs/foundations/streaming#streams-as-data-types) [Passing Streams as Arguments](https://useworkflow.dev/docs/foundations/streaming#passing-streams-as-arguments) [Important Limitation](https://useworkflow.dev/docs/foundations/streaming#important-limitation) [Namespaced Streams](https://useworkflow.dev/docs/foundations/streaming#namespaced-streams) [Consuming Namespaced Streams](https://useworkflow.dev/docs/foundations/streaming#consuming-namespaced-streams) [Common Patterns](https://useworkflow.dev/docs/foundations/streaming#common-patterns) [Progress Updates for Long-Running Tasks](https://useworkflow.dev/docs/foundations/streaming#progress-updates-for-long-running-tasks) [Streaming AI Responses with `DurableAgent`](https://useworkflow.dev/docs/foundations/streaming#streaming-ai-responses-with-durableagent) [Streaming Between Steps](https://useworkflow.dev/docs/foundations/streaming#streaming-between-steps) [Processing Large Files Without Memory Overhead](https://useworkflow.dev/docs/foundations/streaming#processing-large-files-without-memory-overhead) [Best Practices](https://useworkflow.dev/docs/foundations/streaming#best-practices) [Stream Failures](https://useworkflow.dev/docs/foundations/streaming#stream-failures) [Related Documentation](https://useworkflow.dev/docs/foundations/streaming#related-documentation)

[GitHubEdit this page on GitHub](https://github.com/vercel/workflow/edit/main/content/docs/foundations/streaming.mdx) Scroll to topGive feedbackCopy pageAsk AI about this pageOpen in chat

## Next.js Workflow Integration
[API Reference](https://useworkflow.dev/docs/api-reference)

# workflow/next

Next.js integration for Workflow DevKit that automatically configures bundling and runtime support.

## [Functions](https://useworkflow.dev/docs/api-reference/workflow-next\#functions)

[**withWorkflow** \\
\\
Configures webpack/turbopack loaders to transform workflow code (`"use step"`/`"use workflow"` directives)](https://useworkflow.dev/docs/api-reference/workflow-next/with-workflow)

[start\\
\\
Previous Page](https://useworkflow.dev/docs/api-reference/workflow-api/start) [withWorkflow\\
\\
Next Page](https://useworkflow.dev/docs/api-reference/workflow-next/with-workflow)

On this page

[Functions](https://useworkflow.dev/docs/api-reference/workflow-next#functions)

[GitHubEdit this page on GitHub](https://github.com/vercel/workflow/edit/main/content/docs/api-reference/workflow-next/index.mdx) Scroll to topGive feedbackCopy pageAsk AI about this pageOpen in chat

## Chat

What is Workflow?How does retrying work?What control flow patterns are there?How do directives work?

Tip: You can open and close chat with `⌘I`

0 / 1000

## Workflow Errors Guide
# Errors

Fix common mistakes when creating and executing workflows in the **Workflow DevKit**.

[**fetch-in-workflow** \\
\\
Learn how to use fetch in workflow functions.](https://useworkflow.dev/docs/errors/fetch-in-workflow) [**node-js-module-in-workflow** \\
\\
Learn how to use Node.js modules in workflows.](https://useworkflow.dev/docs/errors/node-js-module-in-workflow) [**serialization-failed** \\
\\
Learn how to handle serialization failures in workflows.](https://useworkflow.dev/docs/errors/serialization-failed) [**start-invalid-workflow-function** \\
\\
Learn how to start an invalid workflow function.](https://useworkflow.dev/docs/errors/start-invalid-workflow-function) [**webhook-invalid-respond-with-value** \\
\\
Learn how to use the correct `respondWith` values for webhooks.](https://useworkflow.dev/docs/errors/webhook-invalid-respond-with-value) [**webhook-response-not-sent** \\
\\
Learn how to send responses when using manual webhook response mode.](https://useworkflow.dev/docs/errors/webhook-response-not-sent)

## [Learn More](https://useworkflow.dev/docs/errors\#learn-more)

- [API Reference](https://useworkflow.dev/docs/api-reference) \- Complete API documentation
- [Foundations](https://useworkflow.dev/docs/foundations) \- Architecture and core concepts
- [Examples](https://github.com/vercel/workflow) \- Sample implementations
- [GitHub Issues](https://github.com/vercel/workflow/issues) \- Report bugs and request features

[Postgres World\\
\\
Previous Page](https://useworkflow.dev/docs/deploying/world/postgres-world) [fetch-in-workflow\\
\\
Next Page](https://useworkflow.dev/docs/errors/fetch-in-workflow)

On this page

[Learn More](https://useworkflow.dev/docs/errors#learn-more)

[GitHubEdit this page on GitHub](https://github.com/vercel/workflow/edit/main/content/docs/errors/index.mdx) Scroll to topGive feedbackCopy pageAsk AI about this pageOpen in chat

## SvelteKit Workflows
[Getting Started](https://useworkflow.dev/docs/getting-started)

# SvelteKit

This guide will walk through setting up your first workflow in a SvelteKit app. Along the way, you'll learn more about the concepts that are fundamental to using the development kit in your own projects.

## [Create Your SvelteKit Project](https://useworkflow.dev/docs/getting-started/sveltekit\#create-your-sveltekit-project)

Start by creating a new SvelteKit project. This command will create a new directory named `my-workflow-app` with a minimal setup and setup a SvelteKit project inside it.

```
npx sv create my-workflow-app --template=minimal --types=ts --no-add-ons
```

Enter the newly made directory:

```
cd my-workflow-app
```

### [Install `workflow`](https://useworkflow.dev/docs/getting-started/sveltekit\#install-workflow)

npm

pnpm

yarn

bun

```
npm i workflow
```

### [Configure Vite](https://useworkflow.dev/docs/getting-started/sveltekit\#configure-vite)

Add `workflowPlugin()` to your Vite config. This enables usage of the `"use workflow"` and `"use step"` directives.

vite.config.ts

```

```

### \#\#\# [Setup IntelliSense for TypeScript (Optional)](https://useworkflow.dev/docs/getting-started/sveltekit\\#setup-intellisense-for-typescript-optional)

## [Create Your First Workflow](https://useworkflow.dev/docs/getting-started/sveltekit\#create-your-first-workflow)

Create a new file for our first workflow:

workflows/user-signup.ts

```

```

We'll fill in those functions next, but let's take a look at this code:

- We define a **workflow** function with the directive `"use workflow"`. Think of the workflow function as the _orchestrator_ of individual **steps**.
- The Workflow DevKit's `sleep` function allows us to suspend execution of the workflow without using up any resources. A sleep can be a few seconds, hours, days, or even months long.

## [Create Your Workflow Steps](https://useworkflow.dev/docs/getting-started/sveltekit\#create-your-workflow-steps)

Let's now define those missing functions.

workflows/user-signup.ts

```

```

Taking a look at this code:

- Business logic lives inside **steps**. When a step is invoked inside a **workflow**, it gets enqueued to run on a separate request while the workflow is suspended, just like `sleep`.
- If a step throws an error, like in `sendWelcomeEmail`, the step will automatically be retried until it succeeds (or hits the step's max retry count).
- Steps can throw a `FatalError` if an error is intentional and should not be retried.

We'll dive deeper into workflows, steps, and other ways to suspend or handle events in [Foundations](https://useworkflow.dev/docs/foundations).

## [Create Your Route Handler](https://useworkflow.dev/docs/getting-started/sveltekit\#create-your-route-handler)

To invoke your new workflow, we'll have to add your workflow to a `POST` API route handler, `src/routes/api/signup/+server.ts` with the following code:

src/routes/api/signup/+server.ts

```

```

This route handler creates a `POST` request endpoint at `/api/signup` that will trigger your workflow.

Workflows can be triggered from API routes or any server-side code.

## [Run in development](https://useworkflow.dev/docs/getting-started/sveltekit\#run-in-development)

To start your development server, run the following command in your terminal in the SvelteKit root directory:

```
npm run dev
```

Once your development server is running, you can trigger your workflow by running this command in the terminal:

```
curl -X POST --json '{"email":"hello@example.com"}' http://localhost:5173/api/signup
```

Check the SvelteKit development server logs to see your workflow execute as well as the steps that are being processed.

Additionally, you can use the [Workflow DevKit CLI or Web UI](https://useworkflow.dev/docs/observability) to inspect your workflow runs and steps in detail.

```
npx workflow inspect runs
# or add '--web' for an interactive Web based UI
```

![Workflow DevKit Web UI](https://useworkflow.dev/o11y-ui.png)

## [Deploying to production](https://useworkflow.dev/docs/getting-started/sveltekit\#deploying-to-production)

Workflow DevKit apps currently work best when deployed to [Vercel](https://vercel.com/home) and needs no special configuration.

Check the [Deploying](https://useworkflow.dev/docs/deploying) section to learn how your workflows can be deployed elsewhere.

## [Next Steps](https://useworkflow.dev/docs/getting-started/sveltekit\#next-steps)

- Learn more about the [Foundations](https://useworkflow.dev/docs/foundations).
- Check [Errors](https://useworkflow.dev/docs/errors) if you encounter issues.
- Explore the [API Reference](https://useworkflow.dev/docs/api-reference).

[Nuxt\\
\\
This guide will walk through setting up your first workflow in a Nuxt app. Along the way, you'll learn more about the concepts that are fundamental to using the development kit in your own projects.](https://useworkflow.dev/docs/getting-started/nuxt) [Foundations\\
\\
Next Page](https://useworkflow.dev/docs/foundations)

On this page

[Create Your SvelteKit Project](https://useworkflow.dev/docs/getting-started/sveltekit#create-your-sveltekit-project) [Install `workflow`](https://useworkflow.dev/docs/getting-started/sveltekit#install-workflow) [Configure Vite](https://useworkflow.dev/docs/getting-started/sveltekit#configure-vite) [Setup IntelliSense for TypeScript (Optional)](https://useworkflow.dev/docs/getting-started/sveltekit#setup-intellisense-for-typescript-optional) [Create Your First Workflow](https://useworkflow.dev/docs/getting-started/sveltekit#create-your-first-workflow) [Create Your Workflow Steps](https://useworkflow.dev/docs/getting-started/sveltekit#create-your-workflow-steps) [Create Your Route Handler](https://useworkflow.dev/docs/getting-started/sveltekit#create-your-route-handler) [Run in development](https://useworkflow.dev/docs/getting-started/sveltekit#run-in-development) [Deploying to production](https://useworkflow.dev/docs/getting-started/sveltekit#deploying-to-production) [Next Steps](https://useworkflow.dev/docs/getting-started/sveltekit#next-steps)

[GitHubEdit this page on GitHub](https://github.com/vercel/workflow/edit/main/content/docs/getting-started/sveltekit.mdx) Scroll to topGive feedbackCopy pageAsk AI about this pageOpen in chat

## Understanding Directives
How it works

# Understanding Directives

This guide explores how JavaScript directives enable the Workflow DevKit's execution model. For getting started with workflows, see the [getting started](https://useworkflow.dev/docs/getting-started) guides for your framework.

The Workflow Development Kit uses JavaScript directives (`"use workflow"` and `"use step"`) as the foundation for its durable execution model. Directives provide the compile-time semantic boundary necessary for workflows to suspend, resume, and maintain deterministic behavior across replays.

This page explores how directives enable this execution model and the design principles that led us here.

To understand how directives work, let's first understand what workflows and steps are in the Workflow DevKit.

## [Workflows and Steps Primer](https://useworkflow.dev/docs/how-it-works/understanding-directives\#workflows-and-steps-primer)

The Workflow DevKit has two types of functions:

**Step functions** are side-effecting operations with full Node.js runtime access. Think of them like named RPC calls - they run once, their result is persisted, and they can be [retried on failure](https://useworkflow.dev/docs/foundations/errors-and-retries):

```
async function fetchUserData(userId: string) {
  "use step";

  // Full Node.js access: database calls, API requests, file I/O
  const user = await db.query('SELECT * FROM users WHERE id = ?', [userId]);
  return user;
}
```

**Workflow functions** are deterministic orchestrators that coordinate steps. They must be pure functions - during replay, the same step results always produce the same output. This is necessary because workflows resume by replaying their code from the beginning using cached step results; non-deterministic logic would break resumption. They run in a sandboxed environment without direct Node.js access:

```
export async function onboardUser(userId: string) {
  "use workflow";

  const user = await fetchUserData(userId); // Calls step

  // Non-deterministic code would break replay behavior
  if (Math.random() > 0.5) {
    await sendWelcomeEmail(user);
  }

  return `Onboarded ${user.name}!`;
}
```

**The key insight:** Workflows resume from suspension by replaying their code using cached step results from the event log. When a step like `await fetchUserData(userId)` is called:

- **If already executed:** Returns the cached result immediately from the event log
- **If not yet executed:** Suspends the workflow, enqueues the step for background execution, and resumes later with the result

This replay mechanism requires deterministic code. If `Math.random()` weren't seeded, the first execution might return `0.7` (sending the email) but replay might return `0.3` (skipping it), thus breaking resumption. The Workflow DevKit sandbox provides seeded `Math.random()` and `Date` to ensure consistent behavior across replays.

For a deeper dive into workflows and steps, see [Workflows and Steps](https://useworkflow.dev/docs/foundations/workflows-and-steps).

## [The Core Challenge](https://useworkflow.dev/docs/how-it-works/understanding-directives\#the-core-challenge)

This execution model enables powerful durability features - workflows can suspend for days, survive restarts, and resume from any point. However, it also requires a semantic boundary in the code that tells **the compiler, runtime, and developer** that execution semantics have changed.

The challenge: how do we mark this boundary in a way that:

1. Enables compile-time transformations and validation
2. Prevents accidental use of non-deterministic APIs
3. Allows static analysis of workflow structure
4. Feels natural to JavaScript developers

Let's look at where directives have been used before, and the alternatives we considered:

## [Prior art on directives](https://useworkflow.dev/docs/how-it-works/understanding-directives\#prior-art-on-directives)

JavaScript directives have precedent for changing execution semantics within a defined scope:

- `"use strict"` (introduced in ECMAScript 5 in 2009, TC39-standardized) changes language rules to make the runtime faster, safer, and more predictable.
- `"use client"` and `"use server"` (introduced by [React Server Components](https://react.dev/reference/rsc/server-components)) define an explicit boundary of "where" code gets executed - client-side browser JavaScript vs server-side Node.js.
- `"use workflow"` (introduced by the Workflow DevKit) defines both "where" code runs (in a deterministic sandbox environment) and "how" it runs (deterministic, resumable, sandboxed execution semantics).

Directives provide a build-time contract.

When the Workflow DevKit sees `"use workflow"`, it:

- Bundles the workflow and its dependencies into code that can be run in a sandbox
- Restricts access to Node.js APIs in that sandbox
- Enables future functionality and optimizations only possible with a build tool
  - For instance, the bundled workflow code can be statically analyzed to generate UML diagrams/visualizations of the workflow

In addition to being important to the compiler, `"use workflow"` explicitly signals to the developer that you are entering a different execution mode.

The `"use workflow"` directive is also used by the Language Server Plugin shipped with Workflow DevKit to provide IntelliSense to your IDE. Check the [getting started instructions](https://useworkflow.dev/docs/getting-started) for your framework for details on setting up the Language Server Plugin.

But we didn't get here immediately. This took some discovery to arrive at:

## [Alternatives We Explored](https://useworkflow.dev/docs/how-it-works/understanding-directives\#alternatives-we-explored)

Before settling on directives, we prototyped several other approaches. Each had significant limitations that made them unsuitable for production use.

### [Runtime-Only "Suspense" API](https://useworkflow.dev/docs/how-it-works/understanding-directives\#runtime-only-suspense-api)

Our first proof of concept used a wrapper-based API without a build step:

```
export const myWorkflow = workflow(() => {
  const message = run(async () => step());
  return `${message}!`;
});
```

This implementation used "throwing promises" (similar to early React Suspense) to suspend execution. When a step needed to run, we'd throw a promise, catch it at the workflow boundary, execute the step, and replay the workflow with the result.

**The problems:**

**1\. Every side effect needed wrapping**

Any operation that could produce non-deterministic results had to be wrapped in `run()`:

```
export const myWorkflow = workflow(async () => {
  // These would be non-deterministic without wrapping
  const now = await run(() => Date.now());
  const random = await run(() => Math.random());
  const user = await run(() => fetchUser());

  return { now, random, user };
});
```

This was verbose and easy to forget. Moreover, if a developer forgot to wrap something innocent like using `Date.now()`, it led to unstable runtime behavior.

For example:

```
export const myWorkflow = workflow(async () => {
  // Nothing stops you from doing this:
  const now = Date.now(); // Non-deterministic, untracked!
  const user = await run(() => fetchUser());

  // This workflow would produce different results on replay
  return { now, user };
});
```

**2\. Closures and mutation became unpredictable**

Variables captured in closures would behave unexpectedly when steps mutated them:

```
export const myWorkflow = workflow(async () => {
  let counter = 0;

  await run(() => {
    counter++; // This mutation happens during step execution
    return saveToDatabase(counter);
  });

  console.log(counter); // What is counter here?
  // During execution: 1 (mutation preserved)
  // During replay: 0 (mutation lost)
  // Inconsistent behavior!
});
```

The workflow function would replay multiple times, but mutations inside `run()` callbacks wouldn't persist across replays. This made reasoning about state nearly impossible.

**3\. Error handling broke down**

Since we used thrown promises for control flow, `try/catch` blocks became unreliable:

```
export const myWorkflow = workflow(async () => {
  try {
    const result = await run(() => step());
    return result;
  } catch (error) {
    // This could catch:
    // 1. A real error from the step
    // 2. The thrown promise used for suspension
    // 3. An error during replay
    // Hard to distinguish without special handling
    console.error(error);
  }
});
```

### [Generator-Based API](https://useworkflow.dev/docs/how-it-works/understanding-directives\#generator-based-api)

We explored using generators for explicit suspension points, inspired by libraries like Effect.ts:

```
export const myWorkflow = workflow(function*() {
  const message = yield* run(() => step());
  return `${message}!`;
});
```

We're big fans of [Effect.ts](https://effect.website/) and the power of generator-based APIs for effect management. However, for workflow orchestration specifically, we found the syntax too heavy for developers unfamiliar with generators.

**The problems:**

**1\. Syntax felt more like a DSL than JavaScript**

Generators require a custom mental model that differs significantly from familiar async/await patterns. The `yield*` syntax and generator delegation were unfamiliar to many developers:

```
// Standard async/await (familiar)
const result = await fetchData();

// Generator-based (unfamiliar)
const result = yield* run(() => fetchData());
```

Complex workflows became particularly verbose and difficult to read:

```
export const myWorkflow = workflow(function*() {
  const user = yield* run(() => fetchUser());

  // Can't use Promise.all directly - need sequential calls or custom helpers
  const orders = yield* run(() => fetchOrders(user.id));
  const payments = yield* run(() => fetchPayments(user.id));

  // Or create a custom generator-aware parallel helper:
  const [orders2, payments2] = yield* all([\
    run(() => fetchOrders(user.id)),\
    run(() => fetchPayments(user.id))\
  ]);

  return { user, orders, payments };
});
```

**2\. Still no compile-time sandboxing**

Like the runtime-only approach, generators couldn't prevent non-deterministic code:

```
export const myWorkflow = workflow(function*() {
  const now = Date.now(); // Still possible, still problematic
  const user = yield* run(() => fetchUser());
  return { now, user };
});
```

The generator syntax addressed suspension but didn't solve the fundamental sandboxing problem.

### [File System-Based Conventions](https://useworkflow.dev/docs/how-it-works/understanding-directives\#file-system-based-conventions)

We explored using file system conventions to identify workflows and steps, similar to how modern frameworks handle routing (Next.js, Hono, Nitro, SvelteKit):

workflows

onboarding.ts

checkout.ts

steps

send-email.ts

charge-payment.ts

With this approach, any function in the `workflows/` directory would be transformed as a workflow, and any function in `steps/` would be a step. No directives needed, just file locations.

**Why this could work:**

- Clear separation of concerns
- Enables compiler transformations based on file path
- Familiar pattern for developers used to file-based routing, for example Next.js

**Why we moved away:**

**1\. Too opinionated for diverse ecosystems**

Different frameworks and developers have strong opinions about project structure. Forcing a specific directory layout often caused conflicts across various conventions, especially in existing codebases.

**2\. No support for publishable, reusable functions**

We want developers to be able to publish libraries to npm that include step and workflow directives. Ideally, logic that is isomorphic so it could be used with and without Workflow DevKit. File system conventions made this impossible.

**3\. Migration and code reuse became difficult**

Migrating existing code required moving files and restructuring projects rather than adding a single line.

The directive approach solved all these issues: it works in any project structure, supports code reuse and migration, enables npm packages, and allows functions to adapt to their execution context.

### [Decorators](https://useworkflow.dev/docs/how-it-works/understanding-directives\#decorators)

We considered decorators, but they presented significant challenges both technical and ergonomic.

**Decorators are non-yet-standard and class-focused**

Decorators are not yet a standard syntax ( [TC39 proposal](https://github.com/tc39/proposal-decorators)) and they currently only work with classes. A class decorator approach could look like this:

```
import {workflow, step} from "workflow";

class MyWorkflow {
  @workflow()
  static async processOrder(orderId: string) {
    const order = await this.fetchOrder(orderId);
    const payment = await this.processPayment(order);
    return { orderId, payment };
  }

  @step()
  static async fetchOrder(orderId: string) {
    // ...
  }
}
```

This approach requires:

- Writing class boilerplate with static methods
- Storing/mutating class properties was not obvious (similar closure/mutation issues as the runtime-only approach)
- Class-based syntax that doesn't feel "JavaScript native" to developers used to functional patterns

As the JavaScript ecosystem has moved toward function-forward programming (exemplified by React's shift from class components to functions and hooks), requiring developers to use classes felt like a step backward and also didn't match our own personal taste as authors of the DevKit.

**The core problem: Presents workflows as regular runtime code**

While decorators can be handled at compile-time with build tool support, they present workflow functions as if they were regular, composable JavaScript code, when they're actually compile-time declarations that need special handling.

See the [Macro Wrapper](https://useworkflow.dev/docs/how-it-works/understanding-directives#macro-wrapper-approach) section below for a deeper dive into why this approach breaks down with concrete examples.

### [Macro Wrapper Approach](https://useworkflow.dev/docs/how-it-works/understanding-directives\#macro-wrapper-approach)

We also explored compile-time macro approaches - using a compiler to transform wrapper functions or decorators into directive-based code:

```
// Function wrapper approach
import { useWorkflow } from "workflow"

export const processOrder = useWorkflow(async (orderId: string) => {
  const order = await fetchOrder(orderId);
  return { orderId };
});

// Decorator approach (would work similarly)
class MyWorkflow {
  @workflow()
  static async processOrder(orderId: string) {
    const order = await fetchOrder(orderId);
    return { orderId };
  }

  // ...
}
```

The compiler could transform both to be equivalent to WDK's directive approach:

```
export const processOrder = async (orderId: string) => {
  "use workflow";
  const order = await fetchOrder(orderId);
  return { orderId };
};
```

The benefit is that macros could enforce types and provide "Go To Definition" or other LSP features out of the box.

However, **the core problem remains: Workflows aren't runtime values**

The fundamental issue is that both wrappers and decorators make workflows appear to be **first-class, runtime values** when they're actually **compile-time declarations**. This mismatch between syntax and semantics creates numerous failure modes.

**Concrete examples of how this breaks:**

```
// Someone writes a "helpful" utility
function withRetry(fn: Function) {
  return useWorkflow(async (...args) => { // Works with useWorkflow
    try {
      return await fn(...args);
    } catch (error) {
      return await fn(...args); // Retry once
    }
  });
}

// Note: the same utility would be written similarly for a decorator based syntax

// Usage looks innocent in both cases
export const processOrder = withRetry(async (orderId: string) => {
  // Is this deterministic? Can it call steps?
  // Nothing in this function indicates the developer is in the
  // deterministic sandboxed workflow
  // Also where is the retry happening? inside or outside the workflow?
  const order = await fetchOrder(orderId);
  return order;
});
```

The developer writing `processOrder` has no visible signal that they're in a deterministic, sandboxed environment. It's also ambiguous whether the retry logic executes inside the workflow or outside, and the actual behavior likely doesn't match developer intuition.

**Why the compiler can't catch this:**

To detect that `processOrder` is actually a workflow, the compiler would need whole-program analysis to track that:

1. `withRetry` returns the result of `useWorkflow`
2. Therefore `processOrder = withRetry(...)` is a workflow
3. The function passed to `withRetry` will execute in a sandboxed context

This level of cross-function analysis is impractical for build tools - it would require analyzing every function call chain in your entire codebase and all dependencies. The compiler can only reliably detect direct `useWorkflow` calls, not calls hidden behind abstractions.

## [How Directives Solve These Problems](https://useworkflow.dev/docs/how-it-works/understanding-directives\#how-directives-solve-these-problems)

Directives address all the issues we encountered with previous approaches:

**1\. Compile-time semantic boundary**

The `"use workflow"` directive tells the compiler to treat this code differently:

```
export async function processOrder(orderId: string) {
  "use workflow"; // Compiler knows: transform this for sandbox execution

  const order = await fetchOrder(orderId); // Compiler knows: this is a step call
  return { orderId, order };
}
```

**2\. Build-time validation**

The compiler can enforce restrictions before deployment:

```
export async function badWorkflow() {
  "use workflow";

  const crypto = require('crypto'); // Build error: Node.js module in workflow
  return crypto.randomBytes(16);
}
```

In fact, Workflow DevKit will throw an error that links to this error page: [Node.js module in workflow](https://useworkflow.dev/docs/errors/node-js-module-in-workflow)

**3\. No closure ambiguity**

Steps are transformed into function calls that communicate with the runtime:

```
export async function processOrder(orderId: string) {
  "use workflow";

  let counter = 0;

  // This essentially becomes: await enqueueStep("updateCounter", [counter])
  // The step receives counter as a parameter, not a closure
  await updateCounter(counter);

  console.log(counter); // Always 0, consistently
}
```

Callbacks, however, run inside the workflow sandbox and work as expected:

```
export async function processOrders(orderIds: string[]) {
  "use workflow";

  let successCount = 0;

  // Callbacks run in the workflow context, not skipped on replay
  await Promise.all(orderIds.map(async (orderId) => {
    const order = await fetchOrder(orderId); // Step call
    if (order.status === 'completed') {
      successCount++; // Mutation works correctly
    }
  }));

  console.log(successCount); // Consistent across replays
  return { total: orderIds.length, successful: successCount };
}
```

The callback runs in the workflow sandbox, so closure reads and mutations behave consistently across replays.

**4\. Natural syntax**

Looks and feels like regular JavaScript:

```
export async function processOrder(orderId: string) {
  "use workflow";

  // Standard async/await patterns work naturally
  const [order, user] = await Promise.all([\
    fetchOrder(orderId),\
    fetchUser(userId)\
  ]);

  return { order, user };
}
```

**5\. Consistent syntax for steps**

The `"use step"` directive maintains consistency. While steps run in the full Node.js runtime and _could_ work without a directive, they need some way to signal to the workflow runtime that they're steps.

We could have used a function wrapper just for steps:

```
// Mixed approach (inconsistent)
export async function processOrder(orderId: string) {
  "use workflow"; // Directive for workflow

  const order = await step(async () => fetchOrder(orderId));
  return order;
}

const fetchOrder = useStep(() => { // Wrapper for step?
  // ...
})
```

Mixing syntaxes felt inconsistent.

An alternative approach we considered was to treat _all_ async function calls as steps by default:

```
export async function processOrder(orderId: string) {
  "use workflow";

  // Every async call becomes a step automatically?
  const [order, user] = await Promise.all([\
    fetchOrder(orderId), // Step\
    fetchUser(userId)    // Step\
  ]);

  return { order, user };
}
```

This breaks down because many valid async operations inside workflows aren't steps:

```
export async function processOrder(orderId: string) {
  "use workflow";

  // These are valid async calls that SHOULD NOT be steps:
  const results = await Promise.all([...]); // Language primitive
  const winner = await Promise.race([...]); // Language primitive

  // Helper function that formats data
  const formatted = await formatOrderData(order); // Pure JavaScript helper
}
```

By requiring explicit `"use step"` directives, developers have fine-grained control over what becomes a durable, retryable step versus what runs inline in the workflow sandbox.

To understand how directives are transformed at compile time, see [How the Code Transform Works](https://useworkflow.dev/docs/how-it-works/code-transform).

## [What Directives Enable](https://useworkflow.dev/docs/how-it-works/understanding-directives\#what-directives-enable)

Because `"use workflow"` defines a compile-time semantic boundary, we can provide:

### Build-Time Validation

The compiler catches invalid patterns before deployment: detects disallowed imports, prevents direct side effects, and validates workflow structure.

### Static Analysis

Analyze workflow code without executing it: generate UML or DAG diagrams automatically, provide observability and visualization, and optimize execution paths.

### Durable Execution

Workflows can safely suspend and resume: persist execution state between steps, resume from checkpoints after failures or deploys, and scale to zero without losing progress.

### Future Optimizations

The semantic boundary enables planned improvements: smaller serialized state for faster checkpoints, smarter scheduling based on workflow structure, and more efficient suspension and resumption.

## [Directives as a JavaScript Pattern](https://useworkflow.dev/docs/how-it-works/understanding-directives\#directives-as-a-javascript-pattern)

Directives in JavaScript have always been contracts between the developer and the execution environment. `"use strict"` made this pattern familiar - it's a string literal that changes how code is interpreted.

While JavaScript doesn't yet have first-class support for custom directives (like Rust's `#[attribute]` or C++'s `#pragma`), string literal directives are the most pragmatic tool available today.

As TC39 members, we at Vercel are actively working with the standards body and broader ecosystem to explore formal specifications for pragma-like syntax or macro annotations that can express execution semantics.

## [Closing Thoughts](https://useworkflow.dev/docs/how-it-works/understanding-directives\#closing-thoughts)

Directives aren't about syntax preference, they're about expressing semantic boundaries. `"use workflow"` tells the compiler, developer, and runtime that this code is deterministic, resumable, and sandboxed.

This clarity enables the Workflow Development Kit to provide durable execution with familiar JavaScript patterns, while maintaining the compile-time guarantees necessary for reliable workflow orchestration.

[Idempotency\\
\\
Previous Page](https://useworkflow.dev/docs/foundations/idempotency) [How the Directives Work\\
\\
Next Page](https://useworkflow.dev/docs/how-it-works/code-transform)

On this page

[Workflows and Steps Primer](https://useworkflow.dev/docs/how-it-works/understanding-directives#workflows-and-steps-primer) [The Core Challenge](https://useworkflow.dev/docs/how-it-works/understanding-directives#the-core-challenge) [Prior art on directives](https://useworkflow.dev/docs/how-it-works/understanding-directives#prior-art-on-directives) [Alternatives We Explored](https://useworkflow.dev/docs/how-it-works/understanding-directives#alternatives-we-explored) [Runtime-Only "Suspense" API](https://useworkflow.dev/docs/how-it-works/understanding-directives#runtime-only-suspense-api) [Generator-Based API](https://useworkflow.dev/docs/how-it-works/understanding-directives#generator-based-api) [File System-Based Conventions](https://useworkflow.dev/docs/how-it-works/understanding-directives#file-system-based-conventions) [Decorators](https://useworkflow.dev/docs/how-it-works/understanding-directives#decorators) [Macro Wrapper Approach](https://useworkflow.dev/docs/how-it-works/understanding-directives#macro-wrapper-approach) [How Directives Solve These Problems](https://useworkflow.dev/docs/how-it-works/understanding-directives#how-directives-solve-these-problems) [What Directives Enable](https://useworkflow.dev/docs/how-it-works/understanding-directives#what-directives-enable) [Directives as a JavaScript Pattern](https://useworkflow.dev/docs/how-it-works/understanding-directives#directives-as-a-javascript-pattern) [Closing Thoughts](https://useworkflow.dev/docs/how-it-works/understanding-directives#closing-thoughts)

[GitHubEdit this page on GitHub](https://github.com/vercel/workflow/edit/main/content/docs/how-it-works/understanding-directives.mdx) Scroll to topGive feedbackCopy pageAsk AI about this pageOpen in chat

## Deploying Workflows
# Deploying

This section is currently experimental and subject to change. Try it out and share your feedback on [GitHub](https://github.com/vercel/workflow/discussions).

Workflows can run on any infrastructure through **Worlds**. A World is an adapter responsible for handling workflow storage, queuing, authentication, and streaming through a given backend.

## [What are Worlds?](https://useworkflow.dev/docs/deploying\#what-are-worlds)

A **World** connects workflows to the infrastructure that powers them. Think of it as the "environment" where your workflows live and execute. The World interface abstracts away the differences between local development and production deployments, allowing the same workflow code to run seamlessly across different environments.

## [Default Behavior](https://useworkflow.dev/docs/deploying\#default-behavior)

Worlds are automatically configured depending on the scenario:

- **Local development** \- Automatically uses the Local World
- **Vercel deployments** \- Automatically uses the Vercel World

When using other worlds, you can explicitly set the configuration through environment variables. Reference the documentation for the appropriate world for configuration details.

## [Built-in Worlds](https://useworkflow.dev/docs/deploying\#built-in-worlds)

Workflow DevKit ships with two world implementations:

[**Local World** \\
\\
Filesystem-based backend for local development, storing data in `.workflow-data/` directory.](https://useworkflow.dev/docs/deploying/world/local-world) [**Vercel World** \\
\\
Production-ready backend for Vercel deployments, integrated with Vercel's infrastructure.](https://useworkflow.dev/docs/deploying/world/vercel-world)

## [Building a World](https://useworkflow.dev/docs/deploying\#building-a-world)

On top of the default Worlds provided by Workflow DevKit, you can also build new world implementations for custom infrastructure:

- Database backends (PostgreSQL, MySQL, MongoDB, etc.)
- Cloud providers (AWS, GCP, Azure, etc.)
- Custom queue systems
- Third-party platforms

To build a custom world, use a community-implemented `World`, or implement the `World` interface yourself. The following interfaces are required:

- **Storage** \- Persisting workflow runs, steps, hooks, and metadata
- **Queue** \- Enqueuing and processing workflow steps asynchronously
- **AuthProvider** \- Handling authentication for API access
- **Streamer** \- Managing readable and writable streams

See the [World API Reference](https://useworkflow.dev/docs/deploying/world) for implementation details.

### [Using a third-party World](https://useworkflow.dev/docs/deploying\#using-a-third-party-world)

For custom backends and third-party world implementations, refer to the specific world's documentation for configuration details. Each world may have its own set of required environment variables and configuration options.

## [Observability](https://useworkflow.dev/docs/deploying\#observability)

The [Observability tools](https://useworkflow.dev/docs/observability) (CLI and Web UI) can connect to any world backend to inspect workflow data. By default, they connect to your local environment, but they can also be configured to inspect remote environments:

```
# Inspect local workflows
npx workflow inspect runs

# Inspect remote workflows (custom worlds)
npx workflow inspect runs --backend <your-world-name>
```

Learn more about [Observability](https://useworkflow.dev/docs/observability) tools.

## [Learn More](https://useworkflow.dev/docs/deploying\#learn-more)

- [Local World](https://useworkflow.dev/docs/deploying/world/local-world) \- Local development backend
- [Vercel World](https://useworkflow.dev/docs/deploying/world/vercel-world) \- Production backend for Vercel
- [World API Reference](https://useworkflow.dev/docs/deploying/world) \- Building custom worlds
- [Observability](https://useworkflow.dev/docs/observability) \- Inspecting workflow data

[Observability\\
\\
Previous Page](https://useworkflow.dev/docs/observability) [Worlds\\
\\
Next Page](https://useworkflow.dev/docs/deploying/world)

On this page

[What are Worlds?](https://useworkflow.dev/docs/deploying#what-are-worlds) [Default Behavior](https://useworkflow.dev/docs/deploying#default-behavior) [Built-in Worlds](https://useworkflow.dev/docs/deploying#built-in-worlds) [Building a World](https://useworkflow.dev/docs/deploying#building-a-world) [Using a third-party World](https://useworkflow.dev/docs/deploying#using-a-third-party-world) [Observability](https://useworkflow.dev/docs/deploying#observability) [Learn More](https://useworkflow.dev/docs/deploying#learn-more)

[GitHubEdit this page on GitHub](https://github.com/vercel/workflow/edit/main/content/docs/deploying/index.mdx) Scroll to topGive feedbackCopy pageAsk AI about this pageOpen in chat

## Start Workflow Run
[API Reference](https://useworkflow.dev/docs/api-reference) [workflow/api](https://useworkflow.dev/docs/api-reference/workflow-api)

# start

Start/enqueue a new workflow run.

```
import { start } from 'workflow/api';
import { myWorkflow } from './workflows/my-workflow';

const run = await start(myWorkflow);
```

## [API Signature](https://useworkflow.dev/docs/api-reference/workflow-api/start\#api-signature)

### [Parameters](https://useworkflow.dev/docs/api-reference/workflow-api/start\#parameters)

This function has multiple signatures.

#### Signature 1

| Name | Type | Description |
| --- | --- | --- |
| `workflow` | WorkflowFunction<TArgs, TResult> \| WorkflowMetadata | The imported workflow function to start. |
| `args` | TArgs | The arguments to pass to the workflow (optional). |
| `options` | StartOptions | The options for the workflow run (optional). |

#### Signature 2

| Name | Type | Description |
| --- | --- | --- |
| `workflow` | WorkflowMetadata \| WorkflowFunction<\[\], TResult> |  |
| `options` | StartOptions |  |

#### [StartOptions](https://useworkflow.dev/docs/api-reference/workflow-api/start\#startoptions)

| Name | Type | Description |
| --- | --- | --- |
| `deploymentId` | string | The deployment ID to use for the workflow run.<br>\*\*Deprecated\*\*: This property should not be set in user code under normal circumstances.<br>It is automatically inferred from environment variables when deploying to Vercel.<br>Only set this if you are doing something advanced and know what you are doing. |

### [Returns](https://useworkflow.dev/docs/api-reference/workflow-api/start\#returns)

Returns a `Run` object:

| Name | Type | Description |
| --- | --- | --- |
| `runId` | string | The ID of the workflow run. |
| `cancel` | () =\> Promise<void> | Cancels the workflow run. |
| `status` | Promise<"pending" \| "running" \| "completed" \| "failed" \| "paused" \| "cancelled"> | The status of the workflow run. |
| `returnValue` | Promise<TResult> | The return value of the workflow run.<br>Polls the workflow return value until it is completed. |
| `workflowName` | Promise<string> | The name of the workflow. |
| `createdAt` | Promise<Date> | The timestamp when the workflow run was created. |
| `startedAt` | Promise<Date \| undefined> | The timestamp when the workflow run started execution.<br>Returns undefined if the workflow has not started yet. |
| `completedAt` | Promise<Date \| undefined> | The timestamp when the workflow run completed.<br>Returns undefined if the workflow has not completed yet. |
| `readable` | ReadableStream<any> | The readable stream of the workflow run. |
| `getReadable` | <R = any>(options?: WorkflowReadableStreamOptions \| undefined) => ReadableStream<R> | Retrieves the workflow run's default readable stream, which reads chunks<br>written to the corresponding writable stream getWritable . |

Learn more about [`WorkflowReadableStreamOptions`](https://useworkflow.dev/docs/api-reference/workflow-api/get-run#workflowreadablestreamoptions).

## [Good to Know](https://useworkflow.dev/docs/api-reference/workflow-api/start\#good-to-know)

- The `start()` function is used in runtime/non-workflow contexts to programmatically trigger workflow executions.
- This is different from calling workflow functions directly, which is the typical pattern in Next.js applications.
- The function returns immediately after enqueuing the workflow - it doesn't wait for the workflow to complete.
- All arguments must be [serializable](https://useworkflow.dev/docs/foundations/serialization).

## [Examples](https://useworkflow.dev/docs/api-reference/workflow-api/start\#examples)

### [With Arguments](https://useworkflow.dev/docs/api-reference/workflow-api/start\#with-arguments)

```
import { start } from 'workflow/api';
import { userSignupWorkflow } from './workflows/user-signup';

const run = await start(userSignupWorkflow, ['user@example.com']);
```

### [With `StartOptions`](https://useworkflow.dev/docs/api-reference/workflow-api/start\#with-startoptions)

```
import { start } from 'workflow/api';
import { myWorkflow } from './workflows/my-workflow';

const run = await start(myWorkflow, ['arg1', 'arg2'], {
  deploymentId: 'custom-deployment-id'
});
```

[resumeWebhook\\
\\
Previous Page](https://useworkflow.dev/docs/api-reference/workflow-api/resume-webhook) [workflow/next\\
\\
Next Page](https://useworkflow.dev/docs/api-reference/workflow-next)

On this page

[API Signature](https://useworkflow.dev/docs/api-reference/workflow-api/start#api-signature) [Parameters](https://useworkflow.dev/docs/api-reference/workflow-api/start#parameters) [StartOptions](https://useworkflow.dev/docs/api-reference/workflow-api/start#startoptions) [Returns](https://useworkflow.dev/docs/api-reference/workflow-api/start#returns) [Good to Know](https://useworkflow.dev/docs/api-reference/workflow-api/start#good-to-know) [Examples](https://useworkflow.dev/docs/api-reference/workflow-api/start#examples) [With Arguments](https://useworkflow.dev/docs/api-reference/workflow-api/start#with-arguments) [With `StartOptions`](https://useworkflow.dev/docs/api-reference/workflow-api/start#with-startoptions)

[GitHubEdit this page on GitHub](https://github.com/vercel/workflow/edit/main/content/docs/api-reference/workflow-api/start.mdx) Scroll to topGive feedbackCopy pageAsk AI about this pageOpen in chat

## Workflow Metadata API
[API Reference](https://useworkflow.dev/docs/api-reference) [workflow](https://useworkflow.dev/docs/api-reference/workflow)

# getWorkflowMetadata

Returns additional metadata available in the current workflow function.

You may want to use this function when you need to:

- Log workflow run IDs
- Access timing information of a workflow

If you need to access step context, take a look at [`getStepMetadata`](https://useworkflow.dev/docs/api-reference/workflow/get-step-metadata).

```
import { getWorkflowMetadata } from "workflow"

async function testWorkflow() {
    "use workflow"

    const ctx = getWorkflowMetadata()
    console.log(ctx.workflowRunId)
}
```

## [API Signature](https://useworkflow.dev/docs/api-reference/workflow/get-workflow-metadata\#api-signature)

### [Parameters](https://useworkflow.dev/docs/api-reference/workflow/get-workflow-metadata\#parameters)

This function does not accept any parameters.

### [Returns](https://useworkflow.dev/docs/api-reference/workflow/get-workflow-metadata\#returns)

| Name | Type | Description |
| --- | --- | --- |
| `workflowRunId` | string | Unique identifier for the workflow run. |
| `workflowStartedAt` | Date | Timestamp when the workflow run started. |
| `url` | string | The URL where the workflow can be triggered. |

[getStepMetadata\\
\\
Previous Page](https://useworkflow.dev/docs/api-reference/workflow/get-step-metadata) [getWritable\\
\\
Retrieves the current workflow run's default writable stream.](https://useworkflow.dev/docs/api-reference/workflow/get-writable)

On this page

[API Signature](https://useworkflow.dev/docs/api-reference/workflow/get-workflow-metadata#api-signature) [Parameters](https://useworkflow.dev/docs/api-reference/workflow/get-workflow-metadata#parameters) [Returns](https://useworkflow.dev/docs/api-reference/workflow/get-workflow-metadata#returns)

[GitHubEdit this page on GitHub](https://github.com/vercel/workflow/edit/main/content/docs/api-reference/workflow/get-workflow-metadata.mdx) Scroll to topGive feedbackCopy pageAsk AI about this pageOpen in chat

## Get Writable Stream
[API Reference](https://useworkflow.dev/docs/api-reference) [workflow](https://useworkflow.dev/docs/api-reference/workflow)

# getWritable

Retrieves the current workflow run's default writable stream.

The writable stream can be obtained in workflow functions and passed to steps, or called directly within step functions to write data that can be read outside the workflow by using the `readable` property of the [`Run` object](https://useworkflow.dev/docs/api-reference/workflow-api/get-run).

Use this function in your workflows and steps to produce streaming output that can be consumed by clients in real-time.

This function can only be called inside a workflow or step function (functions
with `"use workflow"` or `"use step"` directive)

**Important:** While you can call `getWritable()` inside a workflow function
to obtain the stream, you **cannot interact with the stream directly** in the
workflow context (e.g., calling `getWriter()`, `write()`, or `close()`). The
stream must be passed to step functions as arguments, or steps can call
`getWritable()` directly themselves.

```
import { getWritable } from "workflow";

export async function myWorkflow() {
  "use workflow";

  // Get the writable stream
  const writable = getWritable();

  // Pass it to a step function to interact with it
  await writeToStream(writable);
}

async function writeToStream(writable: WritableStream) {
  "use step";

  const writer = writable.getWriter();
  await writer.write(new TextEncoder().encode("Hello from workflow!"));
  writer.releaseLock();
  await writable.close();
}
```

## [API Signature](https://useworkflow.dev/docs/api-reference/workflow/get-writable\#api-signature)

### [Parameters](https://useworkflow.dev/docs/api-reference/workflow/get-writable\#parameters)

| Name | Type | Description |
| --- | --- | --- |
| `options` | WorkflowWritableStreamOptions | Optional configuration for the writable stream |

### [Returns](https://useworkflow.dev/docs/api-reference/workflow/get-writable\#returns)

`WritableStream<W>`

Returns a `WritableStream<W>` where `W` is the type of data you plan to write to the stream.

## [Good to Know](https://useworkflow.dev/docs/api-reference/workflow/get-writable\#good-to-know)

- **Workflow functions can only obtain the stream** \- Call `getWritable()` in a workflow to get the stream reference, but you cannot call methods like `getWriter()`, `write()`, or `close()` directly in the workflow context.
- **Step functions can interact with streams** \- Steps can receive the stream as an argument or call `getWritable()` directly, and they can freely interact with it (write, close, etc.).
- When called from a workflow, the stream must be passed as an argument to steps for interaction.
- When called from a step, it retrieves the same workflow-scoped stream directly.
- Always release the writer lock after writing to prevent resource leaks.
- The stream can write binary data (using `TextEncoder`) or structured objects.
- Remember to close the stream when finished to signal completion.

## [Examples](https://useworkflow.dev/docs/api-reference/workflow/get-writable\#examples)

### [Basic Text Streaming](https://useworkflow.dev/docs/api-reference/workflow/get-writable\#basic-text-streaming)

Here's a simple example streaming text data:

```
import { sleep, getWritable } from "workflow";

export async function outputStreamWorkflow() {
  "use workflow";

  const writable = getWritable();

  await sleep("1s");
  await stepWithOutputStream(writable);
  await sleep("1s");
  await stepCloseOutputStream(writable);

  return "done";
}

async function stepWithOutputStream(writable: WritableStream) {
  "use step";

  const writer = writable.getWriter();
  // Write binary data using TextEncoder
  await writer.write(new TextEncoder().encode("Hello, world!"));
  writer.releaseLock();
}

async function stepCloseOutputStream(writable: WritableStream) {
  "use step";

  // Close the stream to signal completion
  await writable.close();
}
```

### [Calling `getWritable()` Inside Steps](https://useworkflow.dev/docs/api-reference/workflow/get-writable\#calling-getwritable-inside-steps)

You can also call `getWritable()` directly inside step functions without passing it as a parameter:

```
import { sleep, getWritable } from "workflow";

export async function outputStreamFromStepWorkflow() {
  "use workflow";

  // No need to create or pass the stream - steps can get it themselves
  await sleep("1s");
  await stepWithOutputStreamInside();
  await sleep("1s");
  await stepCloseOutputStreamInside();

  return "done";
}

async function stepWithOutputStreamInside() {
  "use step";

  // Call getWritable() directly inside the step
  const writable = getWritable();
  const writer = writable.getWriter();

  await writer.write(new TextEncoder().encode("Hello from step!"));
  writer.releaseLock();
}

async function stepCloseOutputStreamInside() {
  "use step";

  // Call getWritable() to get the same stream
  const writable = getWritable();
  await writable.close();
}
```

### [Using Namespaced Streams in Steps](https://useworkflow.dev/docs/api-reference/workflow/get-writable\#using-namespaced-streams-in-steps)

You can also use namespaced streams when calling `getWritable()` from steps:

```
import { getWritable } from "workflow";

export async function multiStreamWorkflow() {
  "use workflow";

  // Steps will access both streams by namespace
  await writeToDefaultStream();
  await writeToNamedStream();
  await closeStreams();

  return "done";
}

async function writeToDefaultStream() {
  "use step";

  const writable = getWritable(); // Default stream
  const writer = writable.getWriter();
  await writer.write({ message: "Default stream data" });
  writer.releaseLock();
}

async function writeToNamedStream() {
  "use step";

  const writable = getWritable({ namespace: "logs" });
  const writer = writable.getWriter();
  await writer.write({ log: "Named stream data" });
  writer.releaseLock();
}

async function closeStreams() {
  "use step";

  await getWritable().close(); // Close default stream
  await getWritable({ namespace: "logs" }).close(); // Close named stream
}
```

### [Advanced Chat Streaming](https://useworkflow.dev/docs/api-reference/workflow/get-writable\#advanced-chat-streaming)

Here's a more complex example showing how you might stream AI chat responses:

```
import { getWritable } from "workflow";
import { generateId, streamText, type UIMessageChunk } from "ai";

export async function chat(messages: UIMessage[]) {
  "use workflow";

  // Get typed writable stream for UI message chunks
  const writable = getWritable<UIMessageChunk>();

  // Start the stream
  await startStream(writable);

  let currentMessages = [...messages];

  // Process messages in steps
  for (let i = 0; i < MAX_STEPS; i++) {
    const result = await streamTextStep(currentMessages, writable);
    currentMessages.push(result.messages);

    if (result.finishReason !== "tool-calls") {
      break;
    }
  }

  // End the stream
  await endStream(writable);
}

async function startStream(writable: WritableStream<UIMessageChunk>) {
  "use step";

  const writer = writable.getWriter();

  // Send start message
  writer.write({
    type: "start",
    messageMetadata: {
      createdAt: Date.now(),
      messageId: generateId(),
    },
  });

  writer.releaseLock();
}

async function streamTextStep(
  messages: UIMessage[],
  writable: WritableStream<UIMessageChunk>
) {
  "use step";

  const writer = writable.getWriter();

  // Call streamText from the AI SDK
  const result = streamText({
    model: "gpt-4",
    messages,
    /* other options */
  });

  // Pipe the AI stream into the writable stream
  const reader = result
    .toUIMessageStream({ sendStart: false, sendFinish: false })
    .getReader();

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    await writer.write(value);
  }

  reader.releaseLock();

  // Close the stream
  writer.close();
  writer.releaseLock();
}

async function endStream(writable: WritableStream<UIMessageChunk>) {
  "use step";

  // Close the stream to signal completion
  await writable.close();
}
```

[getWorkflowMetadata\\
\\
Previous Page](https://useworkflow.dev/docs/api-reference/workflow/get-workflow-metadata) [sleep\\
\\
Next Page](https://useworkflow.dev/docs/api-reference/workflow/sleep)

On this page

[API Signature](https://useworkflow.dev/docs/api-reference/workflow/get-writable#api-signature) [Parameters](https://useworkflow.dev/docs/api-reference/workflow/get-writable#parameters) [Returns](https://useworkflow.dev/docs/api-reference/workflow/get-writable#returns) [Good to Know](https://useworkflow.dev/docs/api-reference/workflow/get-writable#good-to-know) [Examples](https://useworkflow.dev/docs/api-reference/workflow/get-writable#examples) [Basic Text Streaming](https://useworkflow.dev/docs/api-reference/workflow/get-writable#basic-text-streaming) [Calling `getWritable()` Inside Steps](https://useworkflow.dev/docs/api-reference/workflow/get-writable#calling-getwritable-inside-steps) [Using Namespaced Streams in Steps](https://useworkflow.dev/docs/api-reference/workflow/get-writable#using-namespaced-streams-in-steps) [Advanced Chat Streaming](https://useworkflow.dev/docs/api-reference/workflow/get-writable#advanced-chat-streaming)

[GitHubEdit this page on GitHub](https://github.com/vercel/workflow/edit/main/content/docs/api-reference/workflow/get-writable.mdx) Scroll to topGive feedbackCopy pageAsk AI about this pageOpen in chat

## Serialization Error Guide
[Errors](https://useworkflow.dev/docs/errors)

# serialization-failed

This error occurs when you try to pass non-serializable data between execution boundaries in your workflow. All data passed between workflow functions, step functions, and the workflow runtime must be serializable to persist in the event log.

## [Error Message](https://useworkflow.dev/docs/errors/serialization-failed\#error-message)

```
Failed to serialize workflow arguments. Ensure you're passing serializable types
(plain objects, arrays, primitives, Date, RegExp, Map, Set).
```

This error can appear when:

- Serializing workflow arguments when calling `start()`
- Serializing workflow return values
- Serializing step arguments
- Serializing step return values

## [Why This Happens](https://useworkflow.dev/docs/errors/serialization-failed\#why-this-happens)

Workflows persist their state using an event log. Every value that crosses execution boundaries must be:

1. **Serialized** to be stored in the event log
2. **Deserialized** when the workflow resumes

Functions, class instances, symbols, and other non-serializable types cannot be properly reconstructed after serialization, which would break workflow replay.

## [Common Causes](https://useworkflow.dev/docs/errors/serialization-failed\#common-causes)

### [Passing Functions](https://useworkflow.dev/docs/errors/serialization-failed\#passing-functions)

```
// Error - functions cannot be serialized
export async function processWorkflow() {
  "use workflow";

  const callback = () => console.log('done');
  await processStep(callback); // Error!
}
```

**Solution:** Pass data instead, then define the function logic in the step.

```
// Fixed - pass configuration data instead
export async function processWorkflow() {
  "use workflow";

  await processStep({ shouldLog: true });
}

async function processStep(config: { shouldLog: boolean }) {
  "use step";

  if (config.shouldLog) {
    console.log('done');
  }
}
```

### [Class Instances](https://useworkflow.dev/docs/errors/serialization-failed\#class-instances)

```
class User {
  constructor(public name: string) {}
  greet() { return `Hello ${this.name}`; }
}

// Error - class instances lose methods after serialization
export async function greetWorkflow() {
  "use workflow";

  await greetStep(new User('Alice')); // Error!
}
```

**Solution:** Pass plain objects and reconstruct the class in the step.

```
class User {
  constructor(public name: string) {}
  greet() { return `Hello ${this.name}`; }
}

// Fixed - pass plain object, reconstruct in step
export async function greetWorkflow() {
  "use workflow";

  await greetStep({ name: 'Alice' });
}

async function greetStep(userData: { name: string }) {
  "use step";

  const user = new User(userData.name);
  console.log(user.greet());
}
```

## [Supported Serializable Types](https://useworkflow.dev/docs/errors/serialization-failed\#supported-serializable-types)

Workflow DevKit supports these types across execution boundaries:

### [Standard JSON Types](https://useworkflow.dev/docs/errors/serialization-failed\#standard-json-types)

- `string`, `number`, `boolean`, `null`
- Arrays of serializable values
- Plain objects with serializable values

To learn more about supported types, see the [Serialization](https://useworkflow.dev/docs/foundations/serialization) section.

## [Debugging Serialization Issues](https://useworkflow.dev/docs/errors/serialization-failed\#debugging-serialization-issues)

To identify what's causing serialization to fail:

1. **Check the error stack trace** \- it often shows which property failed
2. **Simplify your data** \- temporarily pass smaller objects to isolate the issue
3. **Ensure you are using supported data types** \- see the [Serialization](https://useworkflow.dev/docs/foundations/serialization) section for more details

[node-js-module-in-workflow\\
\\
Previous Page](https://useworkflow.dev/docs/errors/node-js-module-in-workflow) [start-invalid-workflow-function\\
\\
Next Page](https://useworkflow.dev/docs/errors/start-invalid-workflow-function)

On this page

[Error Message](https://useworkflow.dev/docs/errors/serialization-failed#error-message) [Why This Happens](https://useworkflow.dev/docs/errors/serialization-failed#why-this-happens) [Common Causes](https://useworkflow.dev/docs/errors/serialization-failed#common-causes) [Passing Functions](https://useworkflow.dev/docs/errors/serialization-failed#passing-functions) [Class Instances](https://useworkflow.dev/docs/errors/serialization-failed#class-instances) [Supported Serializable Types](https://useworkflow.dev/docs/errors/serialization-failed#supported-serializable-types) [Standard JSON Types](https://useworkflow.dev/docs/errors/serialization-failed#standard-json-types) [Debugging Serialization Issues](https://useworkflow.dev/docs/errors/serialization-failed#debugging-serialization-issues)

[GitHubEdit this page on GitHub](https://github.com/vercel/workflow/edit/main/content/docs/errors/serialization-failed.mdx) Scroll to topGive feedbackCopy pageAsk AI about this pageOpen in chat

## Error Handling Strategies
[Foundations](https://useworkflow.dev/docs/foundations)

# Errors & Retrying

By default, errors thrown inside steps are retried. Additionally, Workflow DevKit provides two new types of errors you can use to customize retries.

## [Default Retrying](https://useworkflow.dev/docs/foundations/errors-and-retries\#default-retrying)

By default, steps retry up to 3 times on arbitrary errors. You can customize the number of retries by adding a `maxRetries` property to the step function.

```
async function callApi(endpoint: string) {
  "use step";

  const response = await fetch(endpoint);

  if (response.status >= 500) {
    // Any uncaught error gets retried
    throw new Error("Uncaught exceptions get retried!");
  }

  return response.json();
}

callApi.maxRetries = 5; // Set a custom number of retries
```

Steps get enqueued immediately after a failure. Read on to see how this can be customized.

When a retried step performs external side effects (payments, emails, API
writes), ensure those calls are **idempotent** to avoid duplicate
side effects. See [Idempotency](https://useworkflow.dev/docs/foundations/idempotency) for
more information.

## [Intentional Errors](https://useworkflow.dev/docs/foundations/errors-and-retries\#intentional-errors)

When your step needs to intentionally throw an error and skip retrying, simply throw a [`FatalError`](https://useworkflow.dev/docs/api-reference/workflow/fatal-error).

```
import { FatalError } from "workflow";

async function callApi(endpoint: string) {
  "use step";

  const response = await fetch(endpoint);

  if (response.status >= 500) {
    // Any uncaught error gets retried
    throw new Error("Uncaught exceptions get retried!");
  }

  if (response.status === 404) {
    throw new FatalError("Resource not found. Skipping retries.");
  }

  return response.json();
}
```

## [Customize Retry Behavior](https://useworkflow.dev/docs/foundations/errors-and-retries\#customize-retry-behavior)

When you need to customize the delay on a retry, use [`RetryableError`](https://useworkflow.dev/docs/api-reference/workflow/retryable-error) and set the `retryAfter` property.

```
import { FatalError, RetryableError } from "workflow";

async function callApi(endpoint: string) {
  "use step";

  const response = await fetch(endpoint);

  if (response.status >= 500) {
    throw new Error("Uncaught exceptions get retried!");
  }

  if (response.status === 404) {
    throw new FatalError("Resource not found. Skipping retries.");
  }

  if (response.status === 429) {
    throw new RetryableError("Rate limited. Retrying...", {
      retryAfter: "1m", // Duration string
    });
  }

  return response.json();
}
```

## [Advanced Example](https://useworkflow.dev/docs/foundations/errors-and-retries\#advanced-example)

This final example combines everything we've learned, along with [`getStepMetadata`](https://useworkflow.dev/docs/api-reference/workflow/get-step-metadata).

```
import { FatalError, RetryableError, getStepMetadata } from "workflow";

async function callApi(endpoint: string) {
  "use step";

  const metadata = getStepMetadata();

  const response = await fetch(endpoint);

  if (response.status >= 500) {
    // Exponential backoffs
    throw new RetryableError("Backing off...", {
      retryAfter: (metadata.attempt ** 2) * 1000,
    });
  }

  if (response.status === 404) {
    throw new FatalError("Resource not found. Skipping retries.");
  }

  if (response.status === 429) {
    throw new RetryableError("Rate limited. Retrying...", {
      retryAfter: new Date(Date.now() + 60000),  // Date instance
    });
  }

  return response.json();
}
callApi.maxRetries = 5;
```

## [Rolling Back Failed Steps](https://useworkflow.dev/docs/foundations/errors-and-retries\#rolling-back-failed-steps)

When a workflow fails partway through, it can leave the system in an inconsistent state.
A common pattern to address this is "rollbacks": for each successful step, record a corresponding rollback action that can undo it.
If a later step fails, run the rollbacks in reverse order to roll back.

Key guidelines:

- Make rollbacks steps as well, so they are durable and benefit from retries.
- Ensure rollbacks are [idempotent](https://useworkflow.dev/docs/foundations/idempotency); they may run more than once.
- Only enqueue a compensation after its forward step succeeds.

```
// Forward steps
async function reserveInventory(orderId: string) {
  "use step";
  // ... call inventory service to reserve ...
}

async function chargePayment(orderId: string) {
  "use step";
  // ... charge the customer ...
}

// Rollback steps
async function releaseInventory(orderId: string) {
  "use step";
  // ... undo inventory reservation ...
}

async function refundPayment(orderId: string) {
  "use step";
  // ... refund the charge ...
}

export async function placeOrderSaga(orderId: string) {
  "use workflow";

  const rollbacks: Array<() => Promise<void>> = [];

  try {
    await reserveInventory(orderId);
    rollbacks.push(() => releaseInventory(orderId));

    await chargePayment(orderId);
    rollbacks.push(() => refundPayment(orderId));

    // ... more steps & rollbacks ...
  } catch (e) {
    for (const rollback of rollbacks.reverse()) {
      await rollback();
    }
    // Rethrow so the workflow records the failure after rollbacks
    throw e;
  }
}
```

[Control Flow Patterns\\
\\
Previous Page](https://useworkflow.dev/docs/foundations/control-flow-patterns) [Hooks & Webhooks\\
\\
Next Page](https://useworkflow.dev/docs/foundations/hooks)

On this page

[Default Retrying](https://useworkflow.dev/docs/foundations/errors-and-retries#default-retrying) [Intentional Errors](https://useworkflow.dev/docs/foundations/errors-and-retries#intentional-errors) [Customize Retry Behavior](https://useworkflow.dev/docs/foundations/errors-and-retries#customize-retry-behavior) [Advanced Example](https://useworkflow.dev/docs/foundations/errors-and-retries#advanced-example) [Rolling Back Failed Steps](https://useworkflow.dev/docs/foundations/errors-and-retries#rolling-back-failed-steps)

[GitHubEdit this page on GitHub](https://github.com/vercel/workflow/edit/main/content/docs/foundations/errors-and-retries.mdx) Scroll to topGive feedbackCopy pageAsk AI about this pageOpen in chat

## Workflow API Reference
[API Reference](https://useworkflow.dev/docs/api-reference)

# workflow/api

API reference for runtime functions from the `workflow/api` package.

## [Functions](https://useworkflow.dev/docs/api-reference/workflow-api\#functions)

Workflow DevKit provides runtime functions that are used outside of workflow and step functions. These are accessed from the runtime entrypoint (e.g. where `start(workflowFn)` is called):

[**start()** \\
\\
Start/enqueue a new workflow run.](https://useworkflow.dev/docs/api-reference/workflow-api/start) [**resumeHook()** \\
\\
Resume a workflow by sending a payload to a hook.](https://useworkflow.dev/docs/api-reference/workflow-api/resume-hook) [**resumeWebhook()** \\
\\
Resume a workflow by sending a `Request` to a webhook.](https://useworkflow.dev/docs/api-reference/workflow-api/resume-webhook) [**getRun()** \\
\\
Get workflow run status and metadata without waiting for completion.](https://useworkflow.dev/docs/api-reference/workflow-api/get-run)

[RetryableError\\
\\
Previous Page](https://useworkflow.dev/docs/api-reference/workflow/retryable-error) [getRun\\
\\
Next Page](https://useworkflow.dev/docs/api-reference/workflow-api/get-run)

On this page

[Functions](https://useworkflow.dev/docs/api-reference/workflow-api#functions)

[GitHubEdit this page on GitHub](https://github.com/vercel/workflow/edit/main/content/docs/api-reference/workflow-api/index.mdx) Scroll to topGive feedbackCopy pageAsk AI about this pageOpen in chat

## Webhook Invalid Response Error
[Errors](https://useworkflow.dev/docs/errors)

# webhook-invalid-respond-with-value

This error occurs when you provide an invalid value for the `respondWith` option when creating a webhook. The `respondWith` option must be either `"manual"` or a `Response` object.

## [Error Message](https://useworkflow.dev/docs/errors/webhook-invalid-respond-with-value\#error-message)

```
Invalid `respondWith` value: [value]
```

## [Why This Happens](https://useworkflow.dev/docs/errors/webhook-invalid-respond-with-value\#why-this-happens)

When creating a webhook with `createWebhook()`, you can specify how the webhook should respond to incoming HTTP requests using the `respondWith` option. This option only accepts specific values:

1. `"manual"` \- Allows you to manually send a response from within the workflow
2. A `Response` object - A pre-defined response to send immediately
3. `undefined` (default) - Returns a `202 Accepted` response

## [Common Causes](https://useworkflow.dev/docs/errors/webhook-invalid-respond-with-value\#common-causes)

### [Using an Invalid String Value](https://useworkflow.dev/docs/errors/webhook-invalid-respond-with-value\#using-an-invalid-string-value)

```
// Error - invalid string value
export async function webhookWorkflow() {
  "use workflow";

  const webhook = await createWebhook({
    respondWith: "automatic", // Error!
  });
}
```

**Solution:** Use `"manual"` or provide a `Response` object.

```
// Fixed - use "manual"
export async function webhookWorkflow() {
  "use workflow";

  const webhook = await createWebhook({
    respondWith: "manual",
  });

  const request = await webhook;

  // Send custom response
  await request.respondWith(new Response("OK", { status: 200 }));
}
```

### [Using a Non-Response Object](https://useworkflow.dev/docs/errors/webhook-invalid-respond-with-value\#using-a-non-response-object)

```
// Error - plain object instead of Response
export async function webhookWorkflow() {
  "use workflow";

  const webhook = await createWebhook({
    respondWith: { status: 200, body: "OK" }, // Error!
  });
}
```

**Solution:** Create a proper `Response` object.

```
// Fixed - use Response constructor
export async function webhookWorkflow() {
  "use workflow";

  const webhook = await createWebhook({
    respondWith: new Response("OK", { status: 200 }),
  });
}
```

## [Valid Usage Examples](https://useworkflow.dev/docs/errors/webhook-invalid-respond-with-value\#valid-usage-examples)

### [Default Behavior (202 Response)](https://useworkflow.dev/docs/errors/webhook-invalid-respond-with-value\#default-behavior-202-response)

```
// Returns 202 Accepted automatically
const webhook = await createWebhook();
const request = await webhook;
// No need to send a response
```

### [Manual Response](https://useworkflow.dev/docs/errors/webhook-invalid-respond-with-value\#manual-response)

```
// Manual response control
const webhook = await createWebhook({
  respondWith: "manual",
});

const request = await webhook;

// Process the request...
const data = await request.json();

// Send custom response
await request.respondWith(
  new Response(JSON.stringify({ success: true }), {
    status: 200,
    headers: { "Content-Type": "application/json" },
  })
);
```

### [Pre-defined Response](https://useworkflow.dev/docs/errors/webhook-invalid-respond-with-value\#pre-defined-response)

```
// Immediate response
const webhook = await createWebhook({
  respondWith: new Response("Request received", { status: 200 }),
});

const request = await webhook;
// Response already sent
```

## [Learn More](https://useworkflow.dev/docs/errors/webhook-invalid-respond-with-value\#learn-more)

- [createWebhook() API Reference](https://useworkflow.dev/docs/api-reference/workflow/create-webhook)
- [resumeWebhook() API Reference](https://useworkflow.dev/docs/api-reference/workflow-api/resume-webhook)
- [Webhooks Guide](https://useworkflow.dev/docs/foundations/hooks)

[start-invalid-workflow-function\\
\\
Previous Page](https://useworkflow.dev/docs/errors/start-invalid-workflow-function) [webhook-response-not-sent\\
\\
Next Page](https://useworkflow.dev/docs/errors/webhook-response-not-sent)

On this page

[Error Message](https://useworkflow.dev/docs/errors/webhook-invalid-respond-with-value#error-message) [Why This Happens](https://useworkflow.dev/docs/errors/webhook-invalid-respond-with-value#why-this-happens) [Common Causes](https://useworkflow.dev/docs/errors/webhook-invalid-respond-with-value#common-causes) [Using an Invalid String Value](https://useworkflow.dev/docs/errors/webhook-invalid-respond-with-value#using-an-invalid-string-value) [Using a Non-Response Object](https://useworkflow.dev/docs/errors/webhook-invalid-respond-with-value#using-a-non-response-object) [Valid Usage Examples](https://useworkflow.dev/docs/errors/webhook-invalid-respond-with-value#valid-usage-examples) [Default Behavior (202 Response)](https://useworkflow.dev/docs/errors/webhook-invalid-respond-with-value#default-behavior-202-response) [Manual Response](https://useworkflow.dev/docs/errors/webhook-invalid-respond-with-value#manual-response) [Pre-defined Response](https://useworkflow.dev/docs/errors/webhook-invalid-respond-with-value#pre-defined-response) [Learn More](https://useworkflow.dev/docs/errors/webhook-invalid-respond-with-value#learn-more)

[GitHubEdit this page on GitHub](https://github.com/vercel/workflow/edit/main/content/docs/errors/webhook-invalid-respond-with-value.mdx) Scroll to topGive feedbackCopy pageAsk AI about this pageOpen in chat

## Control Flow Patterns
[Foundations](https://useworkflow.dev/docs/foundations)

# Control Flow Patterns

Common distributed control flow patterns are simple to implement in workflows and require learning no new syntax. You can just use familiar async/await patterns.

## [Sequential Execution](https://useworkflow.dev/docs/foundations/control-flow-patterns\#sequential-execution)

The simplest way to orchestrate steps is to execute them one after another, where each step can be dependent on the previous step.

```
export async function dataPipelineWorkflow(data: any) {
  "use workflow";

  const validated = await validateData(data);
  const processed = await processData(validated);
  const stored = await storeData(processed);

  return stored;
}
```

## [Parallel Execution](https://useworkflow.dev/docs/foundations/control-flow-patterns\#parallel-execution)

When you need to execute multiple steps in parallel, you can use `Promise.all` to run them all at the same time.

```
export async function fetchUserData(userId: string) {
  "use workflow";

  const [user, orders, preferences] = await Promise.all([\
    fetchUser(userId),\
    fetchOrders(userId),\
    fetchPreferences(userId)\
  ]);

  return { user, orders, preferences };
}
```

This not only applies to steps—since [`sleep()`](https://useworkflow.dev/docs/api-reference/workflow/sleep) and [`webhook`](https://useworkflow.dev/docs/api-reference/workflow/create-webhook) are also just promises, we can await those in parallel too.
We can also use `Promise.race` instead of `Promise.all` to stop executing promises after the first one completes.

```

import { sleep, createWebhook } from "workflow";

export async function runExternalTask(userId: string) {
  "use workflow";

  const webhook = createWebhook();
  await executeExternalTask(webhook.url); // Send the webhook somewhere

  // Wait for the external webhook to be hit, with a timeout of 1 day,
  // whichever comes first
  await Promise.race([\
    webhook,\
    sleep('1 day'),\
  ]);

  console.log("Done")
}
```

## [A full example](https://useworkflow.dev/docs/foundations/control-flow-patterns\#a-full-example)

Here's a simplified example taken from the [birthday card generator demo](https://github.com/vercel/workflow-examples/tree/main/birthday-card-generator), to illustrate how more complex orchestration can be modelled in promises.

```
import { createWebhook, sleep } from "workflow"

async function birthdayWorkflow(
    prompt: string,
    email: string,
    friends: string[],
    birthday: Date
) {
    "use workflow";

    // Generate a birthday card with sequential steps
    const text = await makeCardText(prompt)
    const image = await makeCardImage(text)

    // Create webhooks for each friend who's invited to the birthday party
    const webhooks = friends.map(_ => createWebhook())

    // Send out all the RSVP invites in parallel steps
    await Promise.all(
        friends.map(
            (friend, i) => sendRSVPEmail(friend, webhooks[i])
        )
    )

    // Collect RSVPs as they are made without blocking the workflow
    let rsvps = []
    webhooks.map(
        webhook => webhook
            .then(req => req.json())
            .then(( { rsvp } ) => rsvps.push(rsvp))
    )

    // Wait until the birthday
    await sleep(birthday)

    // Send birthday card with as many rsvps were collected
    await sendBirthdayCard(text, image, rsvps, email)

    return { text, image, status: "Sent" }
}
```

[Starting Workflows\\
\\
Previous Page](https://useworkflow.dev/docs/foundations/starting-workflows) [Errors & Retrying\\
\\
Next Page](https://useworkflow.dev/docs/foundations/errors-and-retries)

On this page

[Sequential Execution](https://useworkflow.dev/docs/foundations/control-flow-patterns#sequential-execution) [Parallel Execution](https://useworkflow.dev/docs/foundations/control-flow-patterns#parallel-execution) [A full example](https://useworkflow.dev/docs/foundations/control-flow-patterns#a-full-example)

[GitHubEdit this page on GitHub](https://github.com/vercel/workflow/edit/main/content/docs/foundations/control-flow-patterns.mdx) Scroll to topGive feedbackCopy pageAsk AI about this pageOpen in chat

## Get Step Metadata
[API Reference](https://useworkflow.dev/docs/api-reference) [workflow](https://useworkflow.dev/docs/api-reference/workflow)

# getStepMetadata

Returns metadata available in the current step function.

You may want to use this function when you need to:

- Track retry attempts in error handling
- Access timing information of a step and execution metadata
- Generate idempotency keys for external APIs

This function can only be called inside a step function.

```
import { getStepMetadata } from "workflow";

async function testWorkflow() {
  "use workflow";
  await logStepId();
}

async function logStepId() {
  "use step";
  const ctx = getStepMetadata();
  console.log(ctx.stepId); // Grab the current step ID
}
```

### [Example: Use `stepId` as an idempotency key](https://useworkflow.dev/docs/api-reference/workflow/get-step-metadata\#example-use-stepid-as-an-idempotency-key)

```
import { getStepMetadata } from "workflow";

async function chargeUser(userId: string, amount: number) {
  "use step";
  const { stepId } = getStepMetadata();

  await stripe.charges.create(
    {
      amount,
      currency: "usd",
      customer: userId,
    },
    {
      idempotencyKey: `charge:${stepId}`,
    }
  );
}
```

Learn more about patterns and caveats in the [Idempotency](https://useworkflow.dev/docs/foundations/idempotency) guide.

## [API Signature](https://useworkflow.dev/docs/api-reference/workflow/get-step-metadata\#api-signature)

### [Parameters](https://useworkflow.dev/docs/api-reference/workflow/get-step-metadata\#parameters)

This function does not accept any parameters.

### [Returns](https://useworkflow.dev/docs/api-reference/workflow/get-step-metadata\#returns)

| Name | Type | Description |
| --- | --- | --- |
| `stepId` | string | Unique identifier for the currently executing step.<br>Useful to use as part of an idempotency key for critical<br>operations that must only be executed once (such as charging a customer). |
| `stepStartedAt` | Date | Timestamp when the current step started. |
| `attempt` | number | The number of times the current step has been executed. This will increase with each retry. |

[fetch\\
\\
Previous Page](https://useworkflow.dev/docs/api-reference/workflow/fetch) [getWorkflowMetadata\\
\\
Next Page](https://useworkflow.dev/docs/api-reference/workflow/get-workflow-metadata)

On this page

[Example: Use `stepId` as an idempotency key](https://useworkflow.dev/docs/api-reference/workflow/get-step-metadata#example-use-stepid-as-an-idempotency-key) [API Signature](https://useworkflow.dev/docs/api-reference/workflow/get-step-metadata#api-signature) [Parameters](https://useworkflow.dev/docs/api-reference/workflow/get-step-metadata#parameters) [Returns](https://useworkflow.dev/docs/api-reference/workflow/get-step-metadata#returns)

[GitHubEdit this page on GitHub](https://github.com/vercel/workflow/edit/main/content/docs/api-reference/workflow/get-step-metadata.mdx) Scroll to topGive feedbackCopy pageAsk AI about this pageOpen in chat

## Workflow Serialization Guide
[Foundations](https://useworkflow.dev/docs/foundations)

# Serialization

All function arguments and return values passed between workflow and step functions must be serializable. Workflow DevKit uses a custom serialization system built on top of [devalue](https://github.com/sveltejs/devalue). This system supports standard JSON types, as well as a few additional popular Web API types.

The serialization system ensures that all data persists correctly across workflow suspensions and resumptions, enabling durable execution.

## [Supported Serializable Types](https://useworkflow.dev/docs/foundations/serialization\#supported-serializable-types)

The following types can be serialized and passed through workflow functions:

**Standard JSON Types:**

- `string`
- `number`
- `boolean`
- `null`
- Arrays of serializable values
- Objects with string keys and serializable values

**Extended Types:**

- `undefined`
- `bigint`
- `ArrayBuffer`
- `BigInt64Array`, `BigUint64Array`
- `Date`
- `Float32Array`, `Float64Array`
- `Int8Array`, `Int16Array`, `Int32Array`
- `Map<Serializable, Serializable>`
- `RegExp`
- `Set<Serializable>`
- `URL`
- `URLSearchParams`
- `Uint8Array`, `Uint8ClampedArray`, `Uint16Array`, `Uint32Array`

**Notable:**

These types have special handling and are explained in detail in the sections below.

- `Headers`
- `Request`
- `Response`
- `ReadableStream<Serializable>`
- `WritableStream<Serializable>`

## [Streaming](https://useworkflow.dev/docs/foundations/serialization\#streaming)

`ReadableStream` and `WritableStream` are supported as serializable types with special handling. These streams can be passed between workflow and step functions while maintaining their streaming capabilities.

For complete information about using streams in workflows, including patterns for AI streaming, file processing, and progress updates, see the [Streaming Guide](https://useworkflow.dev/docs/foundations/streaming).

## [Request & Response](https://useworkflow.dev/docs/foundations/serialization\#request--response)

The Web API [`Request`](https://developer.mozilla.org/en-US/docs/Web/API/Request) and [`Response`](https://developer.mozilla.org/en-US/docs/Web/API/Response) APIs are supported by the serialization system,
and can be passed around between workflow and step functions similarly to other data types.

As a convenience, these two APIs are treated slightly differently when used
within a workflow function: calling the `text()` / `json()` / `arrayBuffer()` instance
methods is automatically treated as a step function invocation. This allows you to consume
the body directly in the workflow context while maintaining proper serialization and caching.

For example, consider how receiving a webhook request provides the entire `Request`
instance into the workflow context. You may consume the body of that request directly
in the workflow, which will be cached as a step result for future resumptions of the workflow:

workflows/webhook.ts

```

```

### [Using `fetch` in Workflows](https://useworkflow.dev/docs/foundations/serialization\#using-fetch-in-workflows)

Because `Request` and `Response` are serializable, Workflow DevKit provides a `fetch` function that can be used directly in workflow functions:

workflows/api-call.ts

```

```

The implementation is straightforward - `fetch` from workflow is a step function that wraps the standard `fetch`:

Implementation

```

```

This allows you to make HTTP requests directly in workflow functions while maintaining deterministic replay behavior through automatic caching.

## [Pass-by-Value Semantics](https://useworkflow.dev/docs/foundations/serialization\#pass-by-value-semantics)

**Parameters are passed by value, not by reference.** Steps receive deserialized copies of data. Mutations inside a step won't affect the original in the workflow.

**Incorrect:**

workflows/incorrect-mutation.ts

```

```

**Correct - return the modified data:**

workflows/correct-mutation.ts

```

```

[Streaming\\
\\
Previous Page](https://useworkflow.dev/docs/foundations/streaming) [Idempotency\\
\\
Next Page](https://useworkflow.dev/docs/foundations/idempotency)

On this page

[Supported Serializable Types](https://useworkflow.dev/docs/foundations/serialization#supported-serializable-types) [Streaming](https://useworkflow.dev/docs/foundations/serialization#streaming) [Request & Response](https://useworkflow.dev/docs/foundations/serialization#request--response) [Using `fetch` in Workflows](https://useworkflow.dev/docs/foundations/serialization#using-fetch-in-workflows) [Pass-by-Value Semantics](https://useworkflow.dev/docs/foundations/serialization#pass-by-value-semantics)

[GitHubEdit this page on GitHub](https://github.com/vercel/workflow/edit/main/content/docs/foundations/serialization.mdx) Scroll to topGive feedbackCopy pageAsk AI about this pageOpen in chat

## Hooks and Webhooks
[Foundations](https://useworkflow.dev/docs/foundations)

# Hooks & Webhooks

Hooks provide a powerful mechanism for pausing workflow execution and resuming it later with external data. They enable workflows to wait for external events, user interactions (also known as "human in the loop"), or HTTP requests. This guide will teach you the core concepts, starting with the low-level Hook primitive and building up to the higher-level Webhook abstraction.

## [Understanding Hooks](https://useworkflow.dev/docs/foundations/hooks\#understanding-hooks)

At their core, **Hooks** are a low-level primitive that allows you to pause a workflow and resume it later with arbitrary [serializable data](https://useworkflow.dev/docs/foundations/serialization). Think of them as suspension points in your workflow where you're waiting for external input.

When you create a hook, it generates a unique token that external systems can use to send data back to your workflow. This makes hooks perfect for scenarios like:

- Waiting for approval from a user or admin
- Receiving data from an external system or service
- Implementing event-driven workflows that react to multiple events over time

### [Creating Your First Hook](https://useworkflow.dev/docs/foundations/hooks\#creating-your-first-hook)

Let's start with a simple example. Here's a workflow that creates a hook and waits for external data:

```
import { createHook } from "workflow";

export async function approvalWorkflow() {
  "use workflow";

  // Create a hook that expects an approval payload
  const hook = createHook<{ approved: boolean; comment: string }>();

  console.log("Waiting for approval...");
  console.log("Send approval to token:", hook.token);

  // Workflow pauses here until data is sent
  const result = await hook;

  if (result.approved) {
    console.log("Approved with comment:", result.comment);
    // Continue with approved workflow...
  } else {
    console.log("Rejected:", result.comment);
    // Handle rejection...
  }
}
```

The workflow will pause at `await hook` until external code sends data to resume it.

See the full API reference for [`createHook()`](https://useworkflow.dev/docs/api-reference/workflow/create-hook) for all available options.

### [Resuming a Hook](https://useworkflow.dev/docs/foundations/hooks\#resuming-a-hook)

To send data to a waiting workflow, use [`resumeHook()`](https://useworkflow.dev/docs/api-reference/workflow-api/resume-hook) from an API route, server action, or any other external context:

```
import { resumeHook } from "workflow/api";

// In an API route or external handler
export async function POST(request: Request) {
  const { token, approved, comment } = await request.json();

  try {
    // Resume the workflow with the approval data
    const result = await resumeHook(token, { approved, comment });
    return Response.json({ success: true, runId: result.runId });
  } catch (error) {
    return Response.json({ error: "Invalid token" }, { status: 404 });
  }
}
```

The key points:

- Hooks allow you to pass **any [serializable data](https://useworkflow.dev/docs/foundations/serialization)** as the payload
- You need the hook's `token` to resume it
- The workflow will resume execution right where it left off

### [Custom Tokens for Deterministic Hooks](https://useworkflow.dev/docs/foundations/hooks\#custom-tokens-for-deterministic-hooks)

By default, hooks generate a random token. However, you often want to use a **custom token** that external systems can reconstruct. This is especially useful for long-running workflows where the same workflow instance should handle multiple events.

For example, imagine a Slack bot where each channel should have its own workflow instance:

```
import { createHook } from "workflow";

export async function slackChannelBot(channelId: string) {
  "use workflow";

  // Use channel ID in the token so Slack webhooks can find this workflow
  const hook = createHook<SlackMessage>({
    token: `slack_messages:${channelId}`
  });

  for await (const message of hook) {
    console.log(`${message.user}: ${message.text}`);

    if (message.text === "/stop") {
      break;
    }

    await processMessage(message);
  }
}

async function processMessage(message: SlackMessage) {
  "use step";
  // Process the Slack message
}
```

Now your Slack webhook handler can deterministically resume the correct workflow:

```
import { resumeHook } from "workflow/api";

export async function POST(request: Request) {
  const slackEvent = await request.json();
  const channelId = slackEvent.channel;

  try {
    // Reconstruct the token using the channel ID
    await resumeHook(`slack_messages:${channelId}`, slackEvent);

    return new Response("OK");
  } catch (error) {
    return new Response("Hook not found", { status: 404 });
  }
}
```

### [Receiving Multiple Events](https://useworkflow.dev/docs/foundations/hooks\#receiving-multiple-events)

Hooks are _reusable_ \- they implement `AsyncIterable`, which means you can use `for await...of` to receive multiple events over time:

```
import { createHook } from "workflow";

export async function dataCollectionWorkflow() {
  "use workflow";

  const hook = createHook<{ value: number; done?: boolean }>();

  const values: number[] = [];

  // Keep receiving data until we get a "done" signal
  for await (const payload of hook) {
    values.push(payload.value);

    if (payload.done) {
      break;
    }
  }

  console.log("Collected values:", values);
  return values;
}
```

Each time you call `resumeHook()` with the same token, the loop receives another value.

## [Understanding Webhooks](https://useworkflow.dev/docs/foundations/hooks\#understanding-webhooks)

While hooks are powerful, they require you to manually handle HTTP requests and route them to workflows. **Webhooks** solve this by providing a higher-level abstraction built on top of hooks that:

1. Automatically serializes the entire HTTP [`Request`](https://developer.mozilla.org/en-US/docs/Web/API/Request) object
2. Provides an automatically addressable `url` property pointing to the generated webhook endpoint
3. Handles sending HTTP [`Response`](https://developer.mozilla.org/en-US/docs/Web/API/Response) objects back to the caller

When using Workflow DevKit, webhooks are automatically wired up at `/.well-known/workflow/v1/webhook/:token` without any additional setup.

See the full API reference for [`createWebhook()`](https://useworkflow.dev/docs/api-reference/workflow/create-webhook) for all available options.

### [Creating Your First Webhook](https://useworkflow.dev/docs/foundations/hooks\#creating-your-first-webhook)

Here's a simple webhook that receives HTTP requests:

```
import { createWebhook } from "workflow";

export async function webhookWorkflow() {
  "use workflow";

  const webhook = createWebhook();

  // The webhook is automatically available at this URL
  console.log("Send HTTP requests to:", webhook.url);
  // Example: https://your-app.com/.well-known/workflow/v1/webhook/lJHkuMdQ2FxSFTbUMU84k

  // Workflow pauses until an HTTP request is received
  const request = await webhook;

  console.log("Received request:", request.method, request.url);

  // Access the request body
  const data = await request.json();
  console.log("Data:", data);
}
```

The webhook will automatically respond with a `202 Accepted` status by default. External systems can simply make an HTTP request to the `webhook.url` to resume your workflow.

### [Sending Custom Responses](https://useworkflow.dev/docs/foundations/hooks\#sending-custom-responses)

Webhooks provide two ways to send custom HTTP responses: **static responses** and **dynamic responses**.

#### [Static Responses](https://useworkflow.dev/docs/foundations/hooks\#static-responses)

Use the `respondWith` option to provide a static response that will be sent automatically for every request:

```
import { createWebhook } from "workflow";

export async function webhookWithStaticResponse() {
  "use workflow";

  const webhook = createWebhook({
    respondWith: Response.json({
      success: true, message: "Webhook received"
    }),
  });

  const request = await webhook;

  // The response was already sent automatically
  // Continue processing the request asynchronously
  const data = await request.json();
  await processData(data);
}

async function processData(data: any) {
  "use step";
  // Long-running processing here
}
```

#### [Dynamic Responses (Manual Mode)](https://useworkflow.dev/docs/foundations/hooks\#dynamic-responses-manual-mode)

For dynamic responses based on the request content, set `respondWith: "manual"` and call the `respondWith()` method on the request:

```
import { createWebhook, type RequestWithResponse } from "workflow";

async function sendCustomResponse(request: RequestWithResponse, message: string) {
  "use step";

  // Call respondWith() to send the response
  await request.respondWith(
    new Response(
      JSON.stringify({ message }),
      {
        status: 200,
        headers: { "Content-Type": "application/json" }
      }
    )
  );
}

export async function webhookWithDynamicResponse() {
  "use workflow";

  // Set respondWith to "manual" to handle responses yourself
  const webhook = createWebhook({ respondWith: "manual" });

  const request = await webhook;
  const data = await request.json();

  // Decide what response to send based on the data
  if (data.type === "urgent") {
    await sendCustomResponse(request, "Processing urgently");
  } else {
    await sendCustomResponse(request, "Processing normally");
  }

  // Continue workflow...
}
```

When using `respondWith: "manual"`, the `respondWith()` method **must** be called from within a step function due to serialization requirements. This requirement may be removed in the future.

### [Handling Multiple Webhook Requests](https://useworkflow.dev/docs/foundations/hooks\#handling-multiple-webhook-requests)

Like hooks, webhooks support iteration:

```
import { createWebhook, type RequestWithResponse } from "workflow";

async function respondToSlack(request: RequestWithResponse, text: string) {
  "use step";

  await request.respondWith(
    new Response(
      JSON.stringify({ response_type: "in_channel", text }),
      { headers: { "Content-Type": "application/json" } }
    )
  );
}

export async function slackCommandWorkflow(channelId: string) {
  "use workflow";

  const webhook = createWebhook({
    token: `slack_command:${channelId}`,
    respondWith: "manual"
  });

  console.log("Configure Slack command webhook:", webhook.url);

  for await (const request of webhook) {
    const formData = await request.formData();
    const command = formData.get("command");
    const text = formData.get("text");

    if (command === "/status") {
      await respondToSlack(request, "Checking status...");
      const status = await checkSystemStatus();
      await postToSlack(channelId, `Status: ${status}`);
    }

    if (text === "stop") {
      await respondToSlack(request, "Stopping workflow...");
      break;
    }
  }
}

async function checkSystemStatus() {
  "use step";
  return "All systems operational";
}

async function postToSlack(channelId: string, message: string) {
  "use step";
  // Post message to Slack
}
```

## [Hooks vs. Webhooks: When to Use Each](https://useworkflow.dev/docs/foundations/hooks\#hooks-vs-webhooks-when-to-use-each)

| Feature | Hooks | Webhooks |
| --- | --- | --- |
| **Data Format** | Arbitrary serializable data | HTTP `Request` objects |
| **URL** | No automatic URL | Automatic `webhook.url` property |
| **Response Handling** | N/A | Can send HTTP `Response` (static or dynamic) |
| **Use Case** | Custom integrations, type-safe payloads | HTTP webhooks, standard REST APIs |
| **Resuming** | [`resumeHook()`](https://useworkflow.dev/docs/api-reference/workflow-api/resume-hook) | Automatic via HTTP, or [`resumeWebhook()`](https://useworkflow.dev/docs/api-reference/workflow-api/resume-webhook) |

**Use Hooks when:**

- You need full control over the payload structure
- You're integrating with custom event sources
- You want strong TypeScript typing with [`defineHook()`](https://useworkflow.dev/docs/api-reference/workflow/define-hook)

**Use Webhooks when:**

- You're receiving HTTP requests from external services
- You need to send HTTP responses back to the caller
- You want automatic URL routing without writing API handlers

## [Advanced Patterns](https://useworkflow.dev/docs/foundations/hooks\#advanced-patterns)

### [Type-Safe Hooks with `defineHook()`](https://useworkflow.dev/docs/foundations/hooks\#type-safe-hooks-with-definehook)

The [`defineHook()`](https://useworkflow.dev/docs/api-reference/workflow/define-hook) helper provides type safety and runtime validation between creating and resuming hooks using [Standard Schema v1](https://standardschema.dev/). Use any compliant validator like Zod or Valibot:

```
import { defineHook } from "workflow";
import { z } from "zod";

// Define the hook with schema for type safety and runtime validation
const approvalHook = defineHook({
  schema: z.object({
    requestId: z.string(),
    approved: z.boolean(),
    approvedBy: z.string(),
    comment: z.string().transform((value) => value.trim()),
  }),
});

// In your workflow
export async function documentApprovalWorkflow(documentId: string) {
  "use workflow";

  const hook = approvalHook.create({
    token: `approval:${documentId}`
  });

  // Payload is type-safe and validated
  const approval = await hook;

  console.log(`Document ${approval.requestId} ${approval.approved ? "approved" : "rejected"}`);
  console.log(`By: ${approval.approvedBy}, Comment: ${approval.comment}`);
}

// In your API route - both type-safe and runtime-validated!
export async function POST(request: Request) {
  const { documentId, ...approvalData } = await request.json();

  try {
    // The schema validates the payload before resuming the workflow
    await approvalHook.resume(`approval:${documentId}`, approvalData);
    return new Response("OK");
  } catch (error) {
    return Response.json({ error: "Invalid token or validation failed" }, { status: 400 });
  }
}
```

This pattern is especially valuable in larger applications where the workflow and API code are in separate files, providing both compile-time type safety and runtime validation.

## [Best Practices](https://useworkflow.dev/docs/foundations/hooks\#best-practices)

### [Token Design](https://useworkflow.dev/docs/foundations/hooks\#token-design)

When using custom tokens:

- **Make them deterministic**: Base them on data the external system can reconstruct (like channel IDs, user IDs, etc.)
- **Use namespacing**: Prefix tokens to avoid conflicts (e.g., `slack:${channelId}`, `github:${repoId}`)
- **Include routing information**: Ensure the token contains enough information to identify the correct workflow instance

### [Response Handling in Webhooks](https://useworkflow.dev/docs/foundations/hooks\#response-handling-in-webhooks)

- Use **static responses** (`respondWith: Response`) for simple acknowledgments
- Use **manual mode** (`respondWith: "manual"`) when responses depend on request processing
- Remember that `respondWith()` must be called from within a step function

### [Iterating Over Events](https://useworkflow.dev/docs/foundations/hooks\#iterating-over-events)

Both hooks and webhooks support iteration, making them perfect for long-running event loops:

```
const hook = createHook<Event>();

for await (const event of hook) {
  await processEvent(event);

  if (shouldStop(event)) {
    break;
  }
}
```

This pattern allows a single workflow instance to handle multiple events over time, maintaining state between events.

## [Related Documentation](https://useworkflow.dev/docs/foundations/hooks\#related-documentation)

- [Serialization](https://useworkflow.dev/docs/foundations/serialization) \- Understanding what data can be passed through hooks
- [`createHook()` API Reference](https://useworkflow.dev/docs/api-reference/workflow/create-hook)
- [`createWebhook()` API Reference](https://useworkflow.dev/docs/api-reference/workflow/create-webhook)
- [`defineHook()` API Reference](https://useworkflow.dev/docs/api-reference/workflow/define-hook)
- [`resumeHook()` API Reference](https://useworkflow.dev/docs/api-reference/workflow-api/resume-hook)
- [`resumeWebhook()` API Reference](https://useworkflow.dev/docs/api-reference/workflow-api/resume-webhook)

[Errors & Retrying\\
\\
Previous Page](https://useworkflow.dev/docs/foundations/errors-and-retries) [Streaming\\
\\
Next Page](https://useworkflow.dev/docs/foundations/streaming)

On this page

[Understanding Hooks](https://useworkflow.dev/docs/foundations/hooks#understanding-hooks) [Creating Your First Hook](https://useworkflow.dev/docs/foundations/hooks#creating-your-first-hook) [Resuming a Hook](https://useworkflow.dev/docs/foundations/hooks#resuming-a-hook) [Custom Tokens for Deterministic Hooks](https://useworkflow.dev/docs/foundations/hooks#custom-tokens-for-deterministic-hooks) [Receiving Multiple Events](https://useworkflow.dev/docs/foundations/hooks#receiving-multiple-events) [Understanding Webhooks](https://useworkflow.dev/docs/foundations/hooks#understanding-webhooks) [Creating Your First Webhook](https://useworkflow.dev/docs/foundations/hooks#creating-your-first-webhook) [Sending Custom Responses](https://useworkflow.dev/docs/foundations/hooks#sending-custom-responses) [Static Responses](https://useworkflow.dev/docs/foundations/hooks#static-responses) [Dynamic Responses (Manual Mode)](https://useworkflow.dev/docs/foundations/hooks#dynamic-responses-manual-mode) [Handling Multiple Webhook Requests](https://useworkflow.dev/docs/foundations/hooks#handling-multiple-webhook-requests) [Hooks vs. Webhooks: When to Use Each](https://useworkflow.dev/docs/foundations/hooks#hooks-vs-webhooks-when-to-use-each) [Advanced Patterns](https://useworkflow.dev/docs/foundations/hooks#advanced-patterns) [Type-Safe Hooks with `defineHook()`](https://useworkflow.dev/docs/foundations/hooks#type-safe-hooks-with-definehook) [Best Practices](https://useworkflow.dev/docs/foundations/hooks#best-practices) [Token Design](https://useworkflow.dev/docs/foundations/hooks#token-design) [Response Handling in Webhooks](https://useworkflow.dev/docs/foundations/hooks#response-handling-in-webhooks) [Iterating Over Events](https://useworkflow.dev/docs/foundations/hooks#iterating-over-events) [Related Documentation](https://useworkflow.dev/docs/foundations/hooks#related-documentation)

[GitHubEdit this page on GitHub](https://github.com/vercel/workflow/edit/main/content/docs/foundations/hooks.mdx) Scroll to topGive feedbackCopy pageAsk AI about this pageOpen in chat

## Nitro Workflow Setup
[Getting Started](https://useworkflow.dev/docs/getting-started)

# Nitro

This guide will walk through setting up your first workflow in a Nitro v3 project. Along the way, you'll learn more about the concepts that are fundamental to using the development kit in your own projects.

## [Create Your Nitro Project](https://useworkflow.dev/docs/getting-started/nitro\#create-your-nitro-project)

Start by creating a new [Nitro v3](https://v3.nitro.build/) project. This command will create a new directory named `nitro-app` and setup a Nitro project inside it.

```
npx create-nitro-app
```

Enter the newly made directory:

```
cd nitro-app
```

### [Install `workflow`](https://useworkflow.dev/docs/getting-started/nitro\#install-workflow)

npm

pnpm

yarn

bun

```
npm i workflow
```

### [Configure Nitro](https://useworkflow.dev/docs/getting-started/nitro\#configure-nitro)

Add `workflow/nitro` module to your `nitro.config.ts` This enables usage of the `"use workflow"` and `"use step"` directives.

nitro.config.ts

```

```

### Setup IntelliSense for TypeScript (Optional)

## [Create Your First Workflow](https://useworkflow.dev/docs/getting-started/nitro\#create-your-first-workflow)

Create a new file for our first workflow:

workflows/user-signup.ts

```

```

We'll fill in those functions next, but let's take a look at this code:

- We define a **workflow** function with the directive `"use workflow"`. Think of the workflow function as the _orchestrator_ of individual **steps**.
- The Workflow DevKit's `sleep` function allows us to suspend execution of the workflow without using up any resources. A sleep can be a few seconds, hours, days, or even months long.

## [Create Your Workflow Steps](https://useworkflow.dev/docs/getting-started/nitro\#create-your-workflow-steps)

Let's now define those missing functions.

workflows/user-signup.ts

```

```

Taking a look at this code:

- Business logic lives inside **steps**. When a step is invoked inside a **workflow**, it gets enqueued to run on a separate request while the workflow is suspended, just like `sleep`.
- If a step throws an error, like in `sendWelcomeEmail`, the step will automatically be retried until it succeeds (or hits the step's max retry count).
- Steps can throw a `FatalError` if an error is intentional and should not be retried.

We'll dive deeper into workflows, steps, and other ways to suspend or handle
events in [Foundations](https://useworkflow.dev/docs/foundations).

## [Create Your Route Handler](https://useworkflow.dev/docs/getting-started/nitro\#create-your-route-handler)

To invoke your new workflow, we'll create a new API route handler at `server/api/signup.post.ts` with the following code:

server/api/signup.post.ts

```

```

This Route Handler creates a `POST` request endpoint at `/api/signup` that will trigger your workflow.

Workflows can be triggered from API routes or any server-side
code.

## [Run in development](https://useworkflow.dev/docs/getting-started/nitro\#run-in-development)

To start your development server, run the following command in your terminal in the Nitro root directory:

```
npm run dev
```

Once your development server is running, you can trigger your workflow by running this command in the terminal:

```
curl -X POST --json '{"email":"hello@example.com"}' http://localhost:3000/api/signup
```

Check the Nitro development server logs to see your workflow execute as well as the steps that are being processed.

Additionally, you can use the [Workflow DevKit CLI or Web UI](https://useworkflow.dev/docs/observability) to inspect your workflow runs and steps in detail.

```
npx workflow inspect runs # add '--web' for an interactive Web based UI
```

![Workflow DevKit Web UI](https://useworkflow.dev/o11y-ui.png)

## [Deploying to production](https://useworkflow.dev/docs/getting-started/nitro\#deploying-to-production)

Workflow DevKit apps currently work best when deployed to [Vercel](https://vercel.com/home) and needs no special configuration.

Check the [Deploying](https://useworkflow.dev/docs/deploying) section to learn how your workflows can be deployed elsewhere.

## [Next Steps](https://useworkflow.dev/docs/getting-started/nitro\#next-steps)

- Learn more about the [Foundations](https://useworkflow.dev/docs/foundations).
- Check [Errors](https://useworkflow.dev/docs/errors) if you encounter issues.
- Explore the [API Reference](https://useworkflow.dev/docs/api-reference).

[Hono\\
\\
This guide will walk through setting up your first workflow in a Hono app. Along the way, you'll learn more about the concepts that are fundamental to using the development kit in your own projects.](https://useworkflow.dev/docs/getting-started/hono) [Nuxt\\
\\
This guide will walk through setting up your first workflow in a Nuxt app. Along the way, you'll learn more about the concepts that are fundamental to using the development kit in your own projects.](https://useworkflow.dev/docs/getting-started/nuxt)

On this page

[Create Your Nitro Project](https://useworkflow.dev/docs/getting-started/nitro#create-your-nitro-project) [Install `workflow`](https://useworkflow.dev/docs/getting-started/nitro#install-workflow) [Configure Nitro](https://useworkflow.dev/docs/getting-started/nitro#configure-nitro) [Create Your First Workflow](https://useworkflow.dev/docs/getting-started/nitro#create-your-first-workflow) [Create Your Workflow Steps](https://useworkflow.dev/docs/getting-started/nitro#create-your-workflow-steps) [Create Your Route Handler](https://useworkflow.dev/docs/getting-started/nitro#create-your-route-handler) [Run in development](https://useworkflow.dev/docs/getting-started/nitro#run-in-development) [Deploying to production](https://useworkflow.dev/docs/getting-started/nitro#deploying-to-production) [Next Steps](https://useworkflow.dev/docs/getting-started/nitro#next-steps)

[GitHubEdit this page on GitHub](https://github.com/vercel/workflow/edit/main/content/docs/getting-started/nitro.mdx) Scroll to topGive feedbackCopy pageAsk AI about this pageOpen in chat

## Chat

What is Workflow?How does retrying work?What control flow patterns are there?How do directives work?

Tip: You can open and close chat with `⌘I`

0 / 1000

## Invalid Workflow Function Error
[Errors](https://useworkflow.dev/docs/errors)

# start-invalid-workflow-function

This error occurs when you try to call `start()` with a function that is not a valid workflow function or when the Workflow DevKit is not configured correctly.

## [Error Message](https://useworkflow.dev/docs/errors/start-invalid-workflow-function\#error-message)

```
'start' received an invalid workflow function. Ensure the Workflow DevKit
is configured correctly and the function includes a 'use workflow' directive.
```

## [Why This Happens](https://useworkflow.dev/docs/errors/start-invalid-workflow-function\#why-this-happens)

The `start()` function expects a workflow function that has been properly processed by Workflow DevKit's build system. During the build process, workflow functions are transformed and marked with special metadata that `start()` uses to identify and execute them.

This error typically happens when:

- The function is missing the `"use workflow"` directive
- The workflow isn't being built/transformed correctly
- The function isn't exported from the workflow file
- The wrong function is being imported

## [Common Causes](https://useworkflow.dev/docs/errors/start-invalid-workflow-function\#common-causes)

### [Missing `"use workflow"` Directive](https://useworkflow.dev/docs/errors/start-invalid-workflow-function\#missing-use-workflow-directive)

workflows/order.ts

```

```

**Solution:** Add the `"use workflow"` directive.

workflows/order.ts

```

```

### [Incorrect Import](https://useworkflow.dev/docs/errors/start-invalid-workflow-function\#incorrect-import)

app/api/route.ts

```

```

**Solution:** Import the correct workflow function.

app/api/route.ts

```

```

### [Next.js Configuration Missing](https://useworkflow.dev/docs/errors/start-invalid-workflow-function\#nextjs-configuration-missing)

next.config.ts

```

```

**Solution:** Wrap with `withWorkflow()`.

next.config.ts

```

```

[serialization-failed\\
\\
Previous Page](https://useworkflow.dev/docs/errors/serialization-failed) [webhook-invalid-respond-with-value\\
\\
Next Page](https://useworkflow.dev/docs/errors/webhook-invalid-respond-with-value)

On this page

[Error Message](https://useworkflow.dev/docs/errors/start-invalid-workflow-function#error-message) [Why This Happens](https://useworkflow.dev/docs/errors/start-invalid-workflow-function#why-this-happens) [Common Causes](https://useworkflow.dev/docs/errors/start-invalid-workflow-function#common-causes) [Missing `"use workflow"` Directive](https://useworkflow.dev/docs/errors/start-invalid-workflow-function#missing-use-workflow-directive) [Incorrect Import](https://useworkflow.dev/docs/errors/start-invalid-workflow-function#incorrect-import) [Next.js Configuration Missing](https://useworkflow.dev/docs/errors/start-invalid-workflow-function#nextjs-configuration-missing)

[GitHubEdit this page on GitHub](https://github.com/vercel/workflow/edit/main/content/docs/errors/start-invalid-workflow-function.mdx) Scroll to topGive feedbackCopy pageAsk AI about this pageOpen in chat

## Workflow World Interface
[Deploying](https://useworkflow.dev/docs/deploying)

# Worlds

This page is a work in progress.

The workflow `World` is an interface that abstracts how workflows and steps communicate with the outside world, letting you build custom adapters and reuse them across different environments, infrastructure, and hosting providers.

## [Choosing a World implementation](https://useworkflow.dev/docs/deploying/world\#choosing-a-world-implementation)

- By default, Workflow uses the [Local world](https://useworkflow.dev/docs/deploying/world/local-world) for easy local development.
- When deployed on Vercel, Workflow switches to the [Vercel world](https://useworkflow.dev/docs/deploying/world/vercel-world).

If you want to use a different World implementation, set the `WORKFLOW_TARGET_WORLD` environment variable to your desired World package's NPM name:

```
export WORKFLOW_TARGET_WORLD=@my-namespace/my-npm-package
export MY_WORLD_CONFIG=... # implementation-specific configuration
```

## [Built-in Worlds](https://useworkflow.dev/docs/deploying/world\#built-in-worlds)

Workflow DevKit provides two built-in world implementations:

- [Local World](https://useworkflow.dev/docs/deploying/world/local-world) \- Filesystem-based for local development
- [Vercel World](https://useworkflow.dev/docs/deploying/world/vercel-world) \- Production-ready for Vercel deployments

## [Community Worlds](https://useworkflow.dev/docs/deploying/world\#community-worlds)

- [Postgres World](https://useworkflow.dev/docs/deploying/world/postgres-world) \- Reference implementation for a multi-host PostgreSQL backend world.
- [Jazz World](https://github.com/garden-co/workflow-world-jazz) \- A full World implementation built on top of [Jazz](https://jazz.tools/)

[Deploying\\
\\
Previous Page](https://useworkflow.dev/docs/deploying) [Local World\\
\\
Next Page](https://useworkflow.dev/docs/deploying/world/local-world)

On this page

[Choosing a World implementation](https://useworkflow.dev/docs/deploying/world#choosing-a-world-implementation) [Built-in Worlds](https://useworkflow.dev/docs/deploying/world#built-in-worlds) [Community Worlds](https://useworkflow.dev/docs/deploying/world#community-worlds)

[GitHubEdit this page on GitHub](https://github.com/vercel/workflow/edit/main/content/docs/deploying/world/index.mdx) Scroll to topGive feedbackCopy pageAsk AI about this pageOpen in chat

## Chat

What is Workflow?How does retrying work?What control flow patterns are there?How do directives work?

Tip: You can open and close chat with `⌘I`

0 / 1000

## Fatal Error Handling
[API Reference](https://useworkflow.dev/docs/api-reference) [workflow](https://useworkflow.dev/docs/api-reference/workflow)

# FatalError

When a `FatalError` is thrown in a step, it indicates that the workflow should not retry a step, marking it as failure.

You should use this when you don't want a specific step to retry.

```
import { FatalError } from "workflow"

async function fallibleWorkflow() {
    "use workflow"
    await fallibleStep();
}

async function fallibleStep() {
    "use step"
    throw new FatalError("Fallible!")
}
```

## [API Signature](https://useworkflow.dev/docs/api-reference/workflow/fatal-error\#api-signature)

### [Parameters](https://useworkflow.dev/docs/api-reference/workflow/fatal-error\#parameters)

| Name | Type | Description |
| --- | --- | --- |
| `message` | string | The error message. |

[sleep\\
\\
Previous Page](https://useworkflow.dev/docs/api-reference/workflow/sleep) [RetryableError\\
\\
Next Page](https://useworkflow.dev/docs/api-reference/workflow/retryable-error)

On this page

[API Signature](https://useworkflow.dev/docs/api-reference/workflow/fatal-error#api-signature) [Parameters](https://useworkflow.dev/docs/api-reference/workflow/fatal-error#parameters)

[GitHubEdit this page on GitHub](https://github.com/vercel/workflow/edit/main/content/docs/api-reference/workflow/fatal-error.mdx) Scroll to topGive feedbackCopy pageAsk AI about this pageOpen in chat

## Getting Started Guide
# Getting Started

Start by choosing your framework. Each guide will walk you through the steps to install the dependencies and start running your first workflow.

[Next.jsNext.js](https://useworkflow.dev/docs/getting-started/next) [ViteVite](https://useworkflow.dev/docs/getting-started/vite) [ExpressExpress](https://useworkflow.dev/docs/getting-started/express) [HonoHono](https://useworkflow.dev/docs/getting-started/hono) [NitroNitro](https://useworkflow.dev/docs/getting-started/nitro) [NuxtNuxt](https://useworkflow.dev/docs/getting-started/nuxt) [SvelteKitSvelteKit](https://useworkflow.dev/docs/getting-started/sveltekit)

AstroComing soon

TanStack StartComing soon

[Next.js\\
\\
This guide will walk through setting up your first workflow in a Next.js app. Along the way, you'll learn more about the concepts that are fundamental to using the development kit in your own projects.](https://useworkflow.dev/docs/getting-started/next)

## Workflows and Steps
[Foundations](https://useworkflow.dev/docs/foundations)

# Workflows and Steps

Workflows (a.k.a. _durable functions_) are a programming model for building long-running, stateful application logic that can maintain its execution state across restarts, failures, or user events. Unlike traditional serverless functions that lose all state when they terminate, workflows persist their progress and can resume exactly where they left off.

Moreover, workflows let you easily model complex multi-step processes in simple, elegant code. To do this, we introduce two fundamental entities:

1. **Workflow Functions**: Functions that orchestrate/organize steps
2. **Step Functions**: Functions that carry out the actual work

## [Workflow Functions](https://useworkflow.dev/docs/foundations/workflows-and-steps\#workflow-functions)

_Directive: `"use workflow"`_

Workflow functions define the entrypoint of a workflow and organize how step functions are called. This type of function does not have access to the Node.js runtime, and usable `npm` packages are limited.

Although this may seem limiting initially, this feature is important in order to suspend and accurately resume execution of workflows.

It helps to think of the workflow function less like a full JavaScript runtime and more like "stitching together" various steps using conditionals, loops, try/catch handlers, `Promise.all`, and other language primitives.

```
export async function processOrderWorkflow(orderId: string) {
  "use workflow";

  // Orchestrate multiple steps
  const order = await fetchOrder(orderId);
  const payment = await chargePayment(order);

  return { orderId, status: 'completed' };
}
```

**Key Characteristics:**

- Runs in a sandboxed environment without full Node.js access
- All step results are persisted to the event log
- Must be **deterministic** to allow resuming after failures

Determinism in the workflow is required to resume the workflow from a suspension. Essentially, the workflow code gets re-run multiple times during its lifecycle, each time using an event log to resume the workflow to the correct spot.

The sandboxed environment that workflows run in already ensures determinism. For instance, `Math.random` and `Date` constructors are fixed in workflow runs, so you are safe to use them, and the framework ensures that the values don't change across replays.

## [Step Functions](https://useworkflow.dev/docs/foundations/workflows-and-steps\#step-functions)

_Directive: `"use step"`_

Step functions perform the actual work in a workflow and have full runtime access.

```
async function chargePayment(order: Order) {
  "use step";

  // Full Node.js access - use any npm package
  const stripe = new Stripe(process.env.STRIPE_KEY);

  const charge = await stripe.charges.create({
    amount: order.total,
    currency: 'usd',
    source: order.paymentToken
  });

  return { chargeId: charge.id };
}
```

**Key Characteristics:**

- Full Node.js runtime and npm package access
- Automatic retry on errors
- Results persisted for replay

By default, steps have a maximum of 3 retry attempts before they fail and propagate the error to the workflow. Learn more about errors and retrying in the [Errors & Retrying](https://useworkflow.dev/docs/foundations/errors-and-retries) page.

**Important:** Due to serialization, parameters are passed by **value, not by reference**. If you pass an object or array to a step and mutate it, those changes will **not** be visible in the workflow context. Always return modified data from your step functions instead. See [Pass-by-Value Semantics](https://useworkflow.dev/docs/foundations/serialization#pass-by-value-semantics) for details and examples.

Step functions are primarily meant to be used inside a workflow.

Calling a step from outside a workflow or from another step will essentially run the step in the same process like a normal function (in other words, the `use step` directive is a no-op). This means you can reuse step functions in other parts of your codebase without needing to duplicate business logic.

```
async function updateUser(userId: string) {
  "use step";
  await db.insert(...);
}

// Used inside a workflow
export async function userOnboardingWorkflow(userId: string) {
  "use workflow";
  await updateUser(userId);
  // ... more steps
}

// Used directly outside a workflow
export async function POST() {
  await updateUser("123");
  // ... more logic
}
```

Keep in mind that calling a step function outside of a workflow function will not have retry semantics, nor will it be observable. Additionally, certain workflow-specific functions like [`getStepMetadata()`](https://useworkflow.dev/docs/api-reference/workflow/get-step-metadata) will throw an error when used inside a step that's called outside a workflow.

### [Suspension and Resumption](https://useworkflow.dev/docs/foundations/workflows-and-steps\#suspension-and-resumption)

Workflow functions have the ability to automatically suspend while they wait on asynchronous work. While suspended, the workflow's state is stored via the event log and no compute resources are used until the workflow resumes execution.

There are multiple ways a workflow can suspend:

- Waiting on a step function: the workflow yields while the step runs in the step runtime.
- Using `sleep()` to pause for some fixed duration.
- Awaiting on a promise returned by [`createWebhook()`](https://useworkflow.dev/docs/api-reference/workflow/create-webhook), which resumes the workflow when an external system passes data into the workflow.

```
import { sleep, createWebhook } from 'workflow';

export async function documentReviewProcess(userId: string) {
  "use workflow";

  await sleep("1 month"); // Sleep will suspend without consuming any resources

  // Create a webhook for external workflow resumption
  const webhook = createWebhook();

  // Send the webhook url to some external service or in an email, etc.
  await sendHumanApprovalEmail("Click this link to accept the review", webhook.url)

  const data = await webhook; // The workflow suspends till the URL is resumed

  console.log("Document reviewed!")
}
```

## [Writing Workflows](https://useworkflow.dev/docs/foundations/workflows-and-steps\#writing-workflows)

### [Basic Structure](https://useworkflow.dev/docs/foundations/workflows-and-steps\#basic-structure)

The simplest workflow consists of a workflow function and one or more step functions.

```
// Workflow function (orchestrates the steps)
export async function greetingWorkflow(name: string) {
  "use workflow";

  const message = await greet(name);
  return { message };
}

// Step function (does the actual work)
async function greet(name: string) {
  "use step";

  // Access Node.js APIs
  const message = `Hello ${name} at ${new Date().toISOString()}`;
  console.log(message);
  return message;
}
```

### [Project structure](https://useworkflow.dev/docs/foundations/workflows-and-steps\#project-structure)

While you can organize workflow and step functions however you like, we find that larger projects benefit from some structure:

workflows

userOnboarding

index.ts

steps.ts

aiVideoGeneration

index.ts

steps

transcribeUpload.ts

generateVideo.ts

notifyUser.ts

shared

validateInput.ts

logActivity.ts

You can choose to organize your steps into a single `steps.ts` file or separate files within a `steps` folder. The `shared` folder is a good place to put common steps that are used by multiple workflows.

Splitting up steps and workflows will also help avoid most bundler related bugs with the Workflow DevKit.

[Foundations\\
\\
Previous Page](https://useworkflow.dev/docs/foundations) [Starting Workflows\\
\\
Next Page](https://useworkflow.dev/docs/foundations/starting-workflows)

On this page

[Workflow Functions](https://useworkflow.dev/docs/foundations/workflows-and-steps#workflow-functions) [Step Functions](https://useworkflow.dev/docs/foundations/workflows-and-steps#step-functions) [Suspension and Resumption](https://useworkflow.dev/docs/foundations/workflows-and-steps#suspension-and-resumption) [Writing Workflows](https://useworkflow.dev/docs/foundations/workflows-and-steps#writing-workflows) [Basic Structure](https://useworkflow.dev/docs/foundations/workflows-and-steps#basic-structure) [Project structure](https://useworkflow.dev/docs/foundations/workflows-and-steps#project-structure)

[GitHubEdit this page on GitHub](https://github.com/vercel/workflow/edit/main/content/docs/foundations/workflows-and-steps.mdx) Scroll to topGive feedbackCopy pageAsk AI about this pageOpen in chat

## Next.js Workflows Guide
[Getting Started](https://useworkflow.dev/docs/getting-started)

# Next.js

This guide will walk through setting up your first workflow in a Next.js app. Along the way, you'll learn more about the concepts that are fundamental to using the development kit in your own projects.

## [Create Your Next.js Project](https://useworkflow.dev/docs/getting-started/next\#create-your-nextjs-project)

Start by creating a new Next.js project. This command will create a new directory named `my-workflow-app` and set up a Next.js project inside it.

```
npm create next-app@latest my-workflow-app
```

Enter the newly created directory:

```
cd my-workflow-app
```

### [Install `workflow`](https://useworkflow.dev/docs/getting-started/next\#install-workflow)

npm

pnpm

yarn

bun

```
npm i workflow
```

### [Configure Next.js](https://useworkflow.dev/docs/getting-started/next\#configure-nextjs)

Wrap your `next.config.ts` with `withWorkflow()`. This enables usage of the `"use workflow"` and `"use step"` directives.

next.config.ts

```

```

### \#\#\# [Setup IntelliSense for TypeScript (Optional)](https://useworkflow.dev/docs/getting-started/next\\#setup-intellisense-for-typescript-optional)

### \#\#\# [Configure Proxy Handler (if applicable)](https://useworkflow.dev/docs/getting-started/next\\#configure-proxy-handler-if-applicable)

## [Create Your First Workflow](https://useworkflow.dev/docs/getting-started/next\#create-your-first-workflow)

Create a new file for our first workflow:

workflows/user-signup.ts

```

```

We'll fill in those functions next, but let's take a look at this code:

- We define a **workflow** function with the directive `"use workflow"`. Think of the workflow function as the _orchestrator_ of individual **steps**.
- The Workflow DevKit's `sleep` function allows us to suspend execution of the workflow without using up any resources. A sleep can be a few seconds, hours, days, or even months long.

## [Create Your Workflow Steps](https://useworkflow.dev/docs/getting-started/next\#create-your-workflow-steps)

Let's now define those missing functions.

workflows/user-signup.ts

```

```

Taking a look at this code:

- Business logic lives inside **steps**. When a step is invoked inside a **workflow**, it gets enqueued to run on a separate request while the workflow is suspended, just like `sleep`.
- If a step throws an error, like in `sendWelcomeEmail`, the step will automatically be retried until it succeeds (or hits the step's max retry count).
- Steps can throw a `FatalError` if an error is intentional and should not be retried.

We'll dive deeper into workflows, steps, and other ways to suspend or handle events in [Foundations](https://useworkflow.dev/docs/foundations).

## [Create Your Route Handler](https://useworkflow.dev/docs/getting-started/next\#create-your-route-handler)

To invoke your new workflow, we'll need to add your workflow to a `POST` API Route Handler, `app/api/signup/route.ts`, with the following code:

app/api/signup/route.ts

```

```

This Route Handler creates a `POST` request endpoint at `/api/signup` that will trigger your workflow.

Workflows can be triggered from API routes, Server Actions, or any server-side code.

## [Run in development](https://useworkflow.dev/docs/getting-started/next\#run-in-development)

To start your development server, run the following command in your terminal in the Next.js root directory:

```
npm run dev
```

Once your development server is running, you can trigger your workflow by running this command in the terminal:

```
curl -X POST --json '{"email":"hello@example.com"}' http://localhost:3000/api/signup
```

Check the Next.js development server logs to see your workflow execute, as well as the steps that are being processed.

Additionally, you can use the [Workflow DevKit CLI or Web UI](https://useworkflow.dev/docs/observability) to inspect your workflow runs and steps in detail.

```
npx workflow inspect runs
# or add '--web' for an interactive Web based UI
```

![Workflow DevKit Web UI](https://useworkflow.dev/o11y-ui.png)

## [Deploying to production](https://useworkflow.dev/docs/getting-started/next\#deploying-to-production)

Workflow DevKit apps currently work best when deployed to [Vercel](https://vercel.com/home) and need no special configuration.

Check the [Deploying](https://useworkflow.dev/docs/deploying) section to learn how your workflows can be deployed elsewhere.

## [Next Steps](https://useworkflow.dev/docs/getting-started/next\#next-steps)

- Learn more about the [Foundations](https://useworkflow.dev/docs/foundations).
- Check [Errors](https://useworkflow.dev/docs/errors) if you encounter issues.
- Explore the [API Reference](https://useworkflow.dev/docs/api-reference).

[Getting Started\\
\\
Start by choosing your framework. Each guide will walk you through the steps to install the dependencies and start running your first workflow.](https://useworkflow.dev/docs/getting-started) [Vite\\
\\
Next Page](https://useworkflow.dev/docs/getting-started/vite)

On this page

[Create Your Next.js Project](https://useworkflow.dev/docs/getting-started/next#create-your-nextjs-project) [Install `workflow`](https://useworkflow.dev/docs/getting-started/next#install-workflow) [Configure Next.js](https://useworkflow.dev/docs/getting-started/next#configure-nextjs) [Setup IntelliSense for TypeScript (Optional)](https://useworkflow.dev/docs/getting-started/next#setup-intellisense-for-typescript-optional) [Configure Proxy Handler (if applicable)](https://useworkflow.dev/docs/getting-started/next#configure-proxy-handler-if-applicable) [Create Your First Workflow](https://useworkflow.dev/docs/getting-started/next#create-your-first-workflow) [Create Your Workflow Steps](https://useworkflow.dev/docs/getting-started/next#create-your-workflow-steps) [Create Your Route Handler](https://useworkflow.dev/docs/getting-started/next#create-your-route-handler) [Run in development](https://useworkflow.dev/docs/getting-started/next#run-in-development) [Deploying to production](https://useworkflow.dev/docs/getting-started/next#deploying-to-production) [Next Steps](https://useworkflow.dev/docs/getting-started/next#next-steps)

[GitHubEdit this page on GitHub](https://github.com/vercel/workflow/edit/main/content/docs/getting-started/next.mdx) Scroll to topGive feedbackCopy pageAsk AI about this pageOpen in chat

## Chat

What is Workflow?How does retrying work?What control flow patterns are there?How do directives work?

Tip: You can open and close chat with `⌘I`

0 / 1000

## Resume Webhook API
[API Reference](https://useworkflow.dev/docs/api-reference) [workflow/api](https://useworkflow.dev/docs/api-reference/workflow-api)

# resumeWebhook

Resumes a workflow run by sending an HTTP `Request` to a webhook identified by its token.

This function creates a `hook_received` event and re-triggers the workflow to continue execution. It's designed to be called from API routes or server actions that receive external HTTP requests.

`resumeWebhook` is a runtime function that must be called from outside a workflow function.

```
import { resumeWebhook } from "workflow/api";

export async function POST(request: Request) {
  const url = new URL(request.url);
  const token = url.searchParams.get('token');

  if (!token) {
    return new Response('Missing token', { status: 400 });
  }

  try {
    const response = await resumeWebhook(token, request);
    return response;
  } catch (error) {
    return new Response('Webhook not found', { status: 404 });
  }
}
```

## [API Signature](https://useworkflow.dev/docs/api-reference/workflow-api/resume-webhook\#api-signature)

### [Parameters](https://useworkflow.dev/docs/api-reference/workflow-api/resume-webhook\#parameters)

| Name | Type | Description |
| --- | --- | --- |
| `token` | string | The unique token identifying the hook |
| `request` | Request | The request to send to the hook |

### [Returns](https://useworkflow.dev/docs/api-reference/workflow-api/resume-webhook\#returns)

Returns a `Promise<Response>` that resolves to:

- `Response`: The HTTP response from the workflow's `respondWith()` call

Throws an error if the webhook token is not found or invalid.

## [Examples](https://useworkflow.dev/docs/api-reference/workflow-api/resume-webhook\#examples)

### [Basic API Route](https://useworkflow.dev/docs/api-reference/workflow-api/resume-webhook\#basic-api-route)

Forward incoming HTTP requests to a webhook by token:

```
import { resumeWebhook } from "workflow/api";

export async function POST(request: Request) {
  const url = new URL(request.url);
  const token = url.searchParams.get('token');

  if (!token) {
    return new Response('Token required', { status: 400 });
  }

  try {
    const response = await resumeWebhook(token, request);
    return response; // Returns the workflow's custom response
  } catch (error) {
    return new Response('Webhook not found', { status: 404 });
  }
}
```

### [GitHub Webhook Handler](https://useworkflow.dev/docs/api-reference/workflow-api/resume-webhook\#github-webhook-handler)

Handle GitHub webhook events and forward them to workflows:

```
import { resumeWebhook } from "workflow/api";
import { verifyGitHubSignature } from "@/lib/github";

export async function POST(request: Request) {
  // Extract repository name from URL
  const url = new URL(request.url);
  const repo = url.pathname.split('/').pop();

  // Verify GitHub signature
  const signature = request.headers.get('x-hub-signature-256');
  const isValid = await verifyGitHubSignature(request, signature);

  if (!isValid) {
    return new Response('Invalid signature', { status: 401 });
  }

  // Construct deterministic token
  const token = `github_webhook:${repo}`;

  try {
    const response = await resumeWebhook(token, request);
    return response;
  } catch (error) {
    return new Response('Workflow not found', { status: 404 });
  }
}
```

### [Slack Slash Command Handler](https://useworkflow.dev/docs/api-reference/workflow-api/resume-webhook\#slack-slash-command-handler)

Process Slack slash commands and route them to workflow webhooks:

```
import { resumeWebhook } from "workflow/api";

export async function POST(request: Request) {
  const formData = await request.formData();
  const channelId = formData.get('channel_id') as string;
  const command = formData.get('command') as string;

  // Verify Slack request signature
  const slackSignature = request.headers.get('x-slack-signature');
  if (!slackSignature) {
    return new Response('Unauthorized', { status: 401 });
  }

  // Construct token from channel ID
  const token = `slack_command:${channelId}`;

  try {
    const response = await resumeWebhook(token, request);
    return response;
  } catch (error) {
    // If no workflow is listening, return a default response
    return new Response(
      JSON.stringify({
        response_type: 'ephemeral',
        text: 'No active workflow for this channel'
      }),
      {
        headers: { 'Content-Type': 'application/json' }
      }
    );
  }
}
```

### [Multi-Tenant Webhook Router](https://useworkflow.dev/docs/api-reference/workflow-api/resume-webhook\#multi-tenant-webhook-router)

Route webhooks to different workflows based on tenant/organization:

```
import { resumeWebhook } from "workflow/api";

export async function POST(request: Request) {
  const url = new URL(request.url);

  // Extract tenant and webhook ID from path
  // e.g., /api/webhooks/tenant-123/webhook-abc
  const [, , , tenantId, webhookId] = url.pathname.split('/');

  if (!tenantId || !webhookId) {
    return new Response('Invalid webhook URL', { status: 400 });
  }

  // Verify API key for tenant
  const apiKey = request.headers.get('authorization');
  const isAuthorized = await verifyTenantApiKey(tenantId, apiKey);

  if (!isAuthorized) {
    return new Response('Unauthorized', { status: 401 });
  }

  // Construct namespaced token
  const token = `tenant:${tenantId}:webhook:${webhookId}`;

  try {
    const response = await resumeWebhook(token, request);
    return response;
  } catch (error) {
    return new Response('Webhook not found or expired', { status: 404 });
  }
}

async function verifyTenantApiKey(tenantId: string, apiKey: string | null) {
  // Verify API key logic
  return apiKey === process.env[`TENANT_${tenantId}_API_KEY`];
}
```

### [Server Action (Next.js)](https://useworkflow.dev/docs/api-reference/workflow-api/resume-webhook\#server-action-nextjs)

Use `resumeWebhook` in a Next.js server action:

```
'use server';

import { resumeWebhook } from "workflow/api";

export async function triggerWebhook(
  token: string,
  payload: Record<string, any>
) {
  // Create a Request object from the payload
  const request = new Request('http://localhost/webhook', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify(payload),
  });

  try {
    const response = await resumeWebhook(token, request);

    // Parse and return the response
    const contentType = response.headers.get('content-type');
    if (contentType?.includes('application/json')) {
      return await response.json();
    }

    return await response.text();
  } catch (error) {
    throw new Error('Webhook not found');
  }
}
```

## [Related Functions](https://useworkflow.dev/docs/api-reference/workflow-api/resume-webhook\#related-functions)

- [`createWebhook()`](https://useworkflow.dev/docs/api-reference/workflow/create-webhook) \- Create a webhook in a workflow
- [`resumeHook()`](https://useworkflow.dev/docs/api-reference/workflow-api/resume-hook) \- Resume a hook with arbitrary payload
- [`defineHook()`](https://useworkflow.dev/docs/api-reference/workflow/define-hook) \- Type-safe hook helper

[resumeHook\\
\\
Previous Page](https://useworkflow.dev/docs/api-reference/workflow-api/resume-hook) [start\\
\\
Next Page](https://useworkflow.dev/docs/api-reference/workflow-api/start)

On this page

[API Signature](https://useworkflow.dev/docs/api-reference/workflow-api/resume-webhook#api-signature) [Parameters](https://useworkflow.dev/docs/api-reference/workflow-api/resume-webhook#parameters) [Returns](https://useworkflow.dev/docs/api-reference/workflow-api/resume-webhook#returns) [Examples](https://useworkflow.dev/docs/api-reference/workflow-api/resume-webhook#examples) [Basic API Route](https://useworkflow.dev/docs/api-reference/workflow-api/resume-webhook#basic-api-route) [GitHub Webhook Handler](https://useworkflow.dev/docs/api-reference/workflow-api/resume-webhook#github-webhook-handler) [Slack Slash Command Handler](https://useworkflow.dev/docs/api-reference/workflow-api/resume-webhook#slack-slash-command-handler) [Multi-Tenant Webhook Router](https://useworkflow.dev/docs/api-reference/workflow-api/resume-webhook#multi-tenant-webhook-router) [Server Action (Next.js)](https://useworkflow.dev/docs/api-reference/workflow-api/resume-webhook#server-action-nextjs) [Related Functions](https://useworkflow.dev/docs/api-reference/workflow-api/resume-webhook#related-functions)

[GitHubEdit this page on GitHub](https://github.com/vercel/workflow/edit/main/content/docs/api-reference/workflow-api/resume-webhook.mdx) Scroll to topGive feedbackCopy pageAsk AI about this pageOpen in chat

## Get Workflow Run Status
[API Reference](https://useworkflow.dev/docs/api-reference) [workflow/api](https://useworkflow.dev/docs/api-reference/workflow-api)

# getRun

Retrieves the workflow run metadata and status information for a given run ID. This function provides immediate access to workflow run details without waiting for completion, making it ideal for status checking and monitoring.

Use this function when you need to check workflow status, get timing information, or access workflow metadata without blocking on workflow completion.

```
import { getRun } from 'workflow/api';

const run = getRun('my-run-id');
```

## [API Signature](https://useworkflow.dev/docs/api-reference/workflow-api/get-run\#api-signature)

### [Parameters](https://useworkflow.dev/docs/api-reference/workflow-api/get-run\#parameters)

| Name | Type | Description |
| --- | --- | --- |
| `runId` | string | The workflow run ID obtained from start. |

### [Returns](https://useworkflow.dev/docs/api-reference/workflow-api/get-run\#returns)

Returns a `Run` object:

| Name | Type | Description |
| --- | --- | --- |
| `runId` | string | The ID of the workflow run. |
| `cancel` | () =\> Promise<void> | Cancels the workflow run. |
| `status` | Promise<"pending" \| "running" \| "completed" \| "failed" \| "paused" \| "cancelled"> | The status of the workflow run. |
| `returnValue` | Promise<TResult> | The return value of the workflow run.<br>Polls the workflow return value until it is completed. |
| `workflowName` | Promise<string> | The name of the workflow. |
| `createdAt` | Promise<Date> | The timestamp when the workflow run was created. |
| `startedAt` | Promise<Date \| undefined> | The timestamp when the workflow run started execution.<br>Returns undefined if the workflow has not started yet. |
| `completedAt` | Promise<Date \| undefined> | The timestamp when the workflow run completed.<br>Returns undefined if the workflow has not completed yet. |
| `readable` | ReadableStream<any> | The readable stream of the workflow run. |
| `getReadable` | <R = any>(options?: WorkflowReadableStreamOptions \| undefined) => ReadableStream<R> | Retrieves the workflow run's default readable stream, which reads chunks<br>written to the corresponding writable stream getWritable . |

#### [WorkflowReadableStreamOptions](https://useworkflow.dev/docs/api-reference/workflow-api/get-run\#workflowreadablestreamoptions)

| Name | Type | Description |
| --- | --- | --- |
| `namespace` | string | An optional namespace to distinguish between multiple streams associated<br>with the same workflow run. |
| `startIndex` | number | The index number of the starting chunk to begin reading the stream from. |
| `ops` | Promise<any>\[\] | Any asynchronous operations that need to be performed before the execution<br>environment is paused / terminated<br>(i.e. using [`waitUntil()`](https://developer.mozilla.org/docs/Web/API/ExtendableEvent/waitUntil) or similar). |
| `global` | Record<string, any> | The global object to use for hydrating types from the global scope.<br>Defaults to [`globalThis`](https://developer.mozilla.org/docs/Web/JavaScript/Reference/Global_Objects/globalThis). |

## [Examples](https://useworkflow.dev/docs/api-reference/workflow-api/get-run\#examples)

### [Basic Status Check](https://useworkflow.dev/docs/api-reference/workflow-api/get-run\#basic-status-check)

Check the current status of a workflow run:

```
import { getRun } from 'workflow/api';

export async function GET(req: Request) {
  const url = new URL(req.url);
  const runId = url.searchParams.get('runId');

  if (!runId) {
    return Response.json({ error: 'No runId provided' }, { status: 400 });
  }

  try {
    const run = getRun(runId);
    const status = await run.status;

    return Response.json({ status });
  } catch (error) {
    return Response.json(
      { error: 'Workflow run not found' },
      { status: 404 }
    );
  }
}
```

## [Related Functions](https://useworkflow.dev/docs/api-reference/workflow-api/get-run\#related-functions)

- [`start()`](https://useworkflow.dev/docs/api-reference/workflow-api/start) \- Start a new workflow and get its run ID.

[workflow/api\\
\\
Previous Page](https://useworkflow.dev/docs/api-reference/workflow-api) [resumeHook\\
\\
Next Page](https://useworkflow.dev/docs/api-reference/workflow-api/resume-hook)

On this page

[API Signature](https://useworkflow.dev/docs/api-reference/workflow-api/get-run#api-signature) [Parameters](https://useworkflow.dev/docs/api-reference/workflow-api/get-run#parameters) [Returns](https://useworkflow.dev/docs/api-reference/workflow-api/get-run#returns) [WorkflowReadableStreamOptions](https://useworkflow.dev/docs/api-reference/workflow-api/get-run#workflowreadablestreamoptions) [Examples](https://useworkflow.dev/docs/api-reference/workflow-api/get-run#examples) [Basic Status Check](https://useworkflow.dev/docs/api-reference/workflow-api/get-run#basic-status-check) [Related Functions](https://useworkflow.dev/docs/api-reference/workflow-api/get-run#related-functions)

[GitHubEdit this page on GitHub](https://github.com/vercel/workflow/edit/main/content/docs/api-reference/workflow-api/get-run.mdx) Scroll to topGive feedbackCopy pageAsk AI about this pageOpen in chat

## Postgres World
[Deploying](https://useworkflow.dev/docs/deploying) [World](https://useworkflow.dev/docs/deploying/world)

# Postgres World

The PostgreSQL world is a reference implementation of a [World](https://useworkflow.dev/docs/deploying/world) that's fully backed by PostgreSQL, including job processing (using [pg-boss](https://github.com/timgit/pg-boss)) and streaming (using PostgreSQL's NOTIFY and LISTEN).

This world is designed for long-running processes, so it can receive and dispatch events from a PostgreSQL database, and isn't meant to be deployed on serverless platforms like Vercel due to that nature.

## [Installation](https://useworkflow.dev/docs/deploying/world/postgres-world\#installation)

Install the `@workflow/world-postgres` package:

npm

pnpm

yarn

bun

```
npm install @workflow/world-postgres
```

## [Add Environment Variables](https://useworkflow.dev/docs/deploying/world/postgres-world\#add-environment-variables)

Add the following environment variables to your `.env` file required for workflows:

```
WORKFLOW_TARGET_WORLD="@workflow/world-postgres"
WORKFLOW_POSTGRES_URL="postgres://postgres:password@db.yourdb.co:5432/postgres"
WORKFLOW_POSTGRES_JOB_PREFIX="workflow_"
WORKFLOW_POSTGRES_WORKER_CONCURRENCY=10
```

The world configuration is automatically read from environment variables:

- `WORKFLOW_TARGET_WORLD` \- Required, specifies which world implementation to use
- `WORKFLOW_POSTGRES_URL` \- PostgreSQL connection string (defaults to `postgres://world:world@localhost:5432/world`)
- `WORKFLOW_POSTGRES_JOB_PREFIX` \- Prefix for queue job names
- `WORKFLOW_POSTGRES_WORKER_CONCURRENCY` \- Number of concurrent workers (defaults to `10` if not specified)

## [Set Up the Database Schema](https://useworkflow.dev/docs/deploying/world/postgres-world\#set-up-the-database-schema)

Run the setup script to create the required database tables:

```
pnpm exec workflow-postgres-setup
```

This will create the following tables in your PostgreSQL database:

- `workflow_runs` \- Stores workflow execution state
- `workflow_events` \- Stores workflow events
- `workflow_steps` \- Stores workflow step state
- `workflow_hooks` \- Stores workflow hooks
- `workflow_stream_chunks` \- Stores streaming data

You should see output like:

```
🔧 Setting up database schema...
📍 Connection: postgres://postgres:****@db.yourcloudprovider.co:5432/postgres
✅ Database schema created successfully!
```

## [Initialize the PostgreSQL World](https://useworkflow.dev/docs/deploying/world/postgres-world\#initialize-the-postgresql-world)

Starting the PostgreSQL World will vary between different frameworks. The main idea is starting the PostgreSQL World on server startup.

Here are some examples of what it might look like:

### [Next.js](https://useworkflow.dev/docs/deploying/world/postgres-world\#nextjs)

Create an `instrumentation.ts` file in your project root to initialize and start the world:

instrumentation.ts

```

```

Learn more about [Next.js Instrumentation](https://nextjs.org/docs/app/guides/instrumentation).

### [SvelteKit](https://useworkflow.dev/docs/deploying/world/postgres-world\#sveltekit)

Create a file `src/hooks.server.ts`:

src/hooks.server.ts

```

```

Learn more about [SvelteKit Hooks](https://svelte.dev/docs/kit/hooks).

### [Nitro-based Apps](https://useworkflow.dev/docs/deploying/world/postgres-world\#nitro-based-apps)

Create a plugin to start the PostgreSQL World. This will be invoked when the Nitro server starts.

plugins/start-pg-world.ts

```

```

Add the plugin to your `nitro.config.ts` file. This enables the plugin:

nitro.config.ts

```

```

Learn more about [Nitro Plugins](https://v3.nitro.build/docs/plugins)

## [How it works](https://useworkflow.dev/docs/deploying/world/postgres-world\#how-it-works)

The Postgres World uses PostgreSQL as a durable backend for workflow execution:

- **Job Queue**: Uses [pg-boss](https://github.com/timgit/pg-boss) for reliable job processing
- **Event Streaming**: Leverages PostgreSQL's NOTIFY/LISTEN for real-time event distribution
- **State Persistence**: All workflow state is stored in PostgreSQL tables
- **Worker Management**: Supports configurable concurrent workers for job processing

This setup ensures that your workflows can survive application restarts and failures, with all state reliably persisted to your PostgreSQL database.

[Vercel World\\
\\
Previous Page](https://useworkflow.dev/docs/deploying/world/vercel-world) [Errors\\
\\
Next Page](https://useworkflow.dev/docs/errors)

On this page

[Installation](https://useworkflow.dev/docs/deploying/world/postgres-world#installation) [Add Environment Variables](https://useworkflow.dev/docs/deploying/world/postgres-world#add-environment-variables) [Set Up the Database Schema](https://useworkflow.dev/docs/deploying/world/postgres-world#set-up-the-database-schema) [Initialize the PostgreSQL World](https://useworkflow.dev/docs/deploying/world/postgres-world#initialize-the-postgresql-world) [Next.js](https://useworkflow.dev/docs/deploying/world/postgres-world#nextjs) [SvelteKit](https://useworkflow.dev/docs/deploying/world/postgres-world#sveltekit) [Nitro-based Apps](https://useworkflow.dev/docs/deploying/world/postgres-world#nitro-based-apps) [How it works](https://useworkflow.dev/docs/deploying/world/postgres-world#how-it-works)

[GitHubEdit this page on GitHub](https://github.com/vercel/workflow/edit/main/content/docs/deploying/world/postgres-world.mdx) Scroll to topGive feedbackCopy pageAsk AI about this pageOpen in chat

## Chat

What is Workflow?How does retrying work?What control flow patterns are there?How do directives work?

Tip: You can open and close chat with `⌘I`

0 / 1000

## Code Transformation Guide
How it works

# How the Directives Work

This is an advanced guide that dives into internals of the Workflow DevKit directive and is not required reading to use workflows. To simply use the Workflow DevKit, check out the [getting started](https://useworkflow.dev/docs/getting-started) guides for your framework.

Workflows use special directives to mark code for transformation by the Workflow DevKit compiler. This page explains how `"use workflow"` and `"use step"` directives work, what transformations are applied, and why they're necessary for durable execution.

## [Directives Overview](https://useworkflow.dev/docs/how-it-works/code-transform\#directives-overview)

Workflows use two directives to mark functions for special handling:

```
export async function handleUserSignup(email: string) {
  "use workflow";

  const user = await createUser(email);
  await sendWelcomeEmail(user);

  return { userId: user.id };
}

async function createUser(email: string) {
  "use step";

  return { id: crypto.randomUUID(), email };
}
```

**Key directives:**

- `"use workflow"`: Marks a function as a durable workflow entry point
- `"use step"`: Marks a function as an atomic, retryable step

These directives trigger the `@workflow/swc-plugin` compiler to transform your code in different ways depending on the execution context.

## [The Three Transformation Modes](https://useworkflow.dev/docs/how-it-works/code-transform\#the-three-transformation-modes)

The compiler operates in three distinct modes, transforming the same source code differently for each execution context:

### [Comparison Table](https://useworkflow.dev/docs/how-it-works/code-transform\#comparison-table)

| Mode | Used In | Purpose | Output API Route | Required? |
| --- | --- | --- | --- | --- |
| Step | Build time | Bundles step handlers | `.well-known/workflow/v1/step` | Yes |
| Workflow | Build time | Bundles workflow orchestrators | `.well-known/workflow/v1/flow` | Yes |
| Client | Build/Runtime | Provides workflow IDs and types to `start` | Your application code | Optional\* |

\\* Client mode is **recommended** for better developer experience—it provides automatic ID generation and type safety. Without it, you must manually construct workflow IDs or use the build manifest.

## [Detailed Transformation Examples](https://useworkflow.dev/docs/how-it-works/code-transform\#detailed-transformation-examples)

Step ModeWorkflow ModeClient Mode

**Step Mode** creates the step execution bundle served at `/.well-known/workflow/v1/step`.

**Input:**

```
export async function createUser(email: string) {
  "use step";
  return { id: crypto.randomUUID(), email };
}
```

**Output:**

```
import { registerStepFunction } from "workflow/internal/private";

export async function createUser(email: string) {
  return { id: crypto.randomUUID(), email };
}

registerStepFunction("step//workflows/user.js//createUser", createUser);
```

**What happens:**

- The `"use step"` directive is removed
- The function body is kept completely intact (no transformation)
- The function is registered with the runtime using `registerStepFunction()`
- Step functions run with full Node.js/Deno/Bun access

**Why no transformation?** Step functions execute in your main runtime with full access to Node.js APIs, file system, databases, etc. They don't need any special handling—they just run normally.

**ID Format:** Step IDs follow the pattern `step//{filepath}//{functionName}`, where the filepath is relative to your project root.

**Workflow Mode** creates the workflow execution bundle served at `/.well-known/workflow/v1/flow`.

**Input:**

```
export async function createUser(email: string) {
  "use step";
  return { id: crypto.randomUUID(), email };
}

export async function handleUserSignup(email: string) {
  "use workflow";
  const user = await createUser(email);
  return { userId: user.id };
}
```

**Output:**

```
export async function createUser(email: string) {
  return globalThis[Symbol.for("WORKFLOW_USE_STEP")]("step//workflows/user.js//createUser")(email);
}

export async function handleUserSignup(email: string) {
  const user = await createUser(email);
  return { userId: user.id };
}
handleUserSignup.workflowId = "workflow//workflows/user.js//handleUserSignup";
```

**What happens:**

- Step function bodies are **replaced** with calls to `globalThis[Symbol.for("WORKFLOW_USE_STEP")]`
- Workflow function bodies remain **intact**—they execute deterministically during replay
- The workflow function gets a `workflowId` property for runtime identification
- The `"use workflow"` directive is removed

**Why this transformation?** When a workflow executes, it needs to replay past steps from the event log rather than re-executing them. The `WORKFLOW_USE_STEP` symbol is a special runtime hook that:

1. Checks if the step has already been executed (in the event log)
2. If yes: Returns the cached result
3. If no: Triggers a suspension and enqueues the step for background execution

**ID Format:** Workflow IDs follow the pattern `workflow//{filepath}//{functionName}`. The `workflowId` property is attached to the function to allow [`start()`](https://useworkflow.dev/docs/api-reference/workflow-api/start) to work at runtime.

**Client Mode** transforms workflow functions in your application code to prevent direct execution.

**Input:**

```
export async function handleUserSignup(email: string) {
  "use workflow";
  const user = await createUser(email);
  return { userId: user.id };
}
```

**Output:**

```
export async function handleUserSignup(email: string) {
  throw new Error("You attempted to execute ...");
}
handleUserSignup.workflowId = "workflow//workflows/user.js//handleUserSignup";
```

**What happens:**

- Workflow function bodies are **replaced** with an error throw
- The `workflowId` property is added (same as workflow mode)
- Step functions are not transformed in client mode

**Why this transformation?** Workflow functions cannot be called directly—they must be started using [`start()`](https://useworkflow.dev/docs/api-reference/workflow-api/start). The error prevents accidental direct execution while the `workflowId` property allows the `start()` function to identify which workflow to launch.

The IDs are generated exactly like in workflow mode to ensure they can be directly referenced at runtime.

**Client mode is optional:** While recommended for better developer experience (automatic IDs and type safety), you can skip client mode and instead:

- Manually construct workflow IDs using the pattern `workflow//{filepath}//{functionName}`
- Use the workflow manifest file generated during build to lookup IDs
- Pass IDs directly to `start()` as strings

All framework integrations include client mode as a loader by default.

## [Generated Files](https://useworkflow.dev/docs/how-it-works/code-transform\#generated-files)

When you build your application, the Workflow DevKit generates three handler files in `.well-known/workflow/v1/`:

### [`flow.js`](https://useworkflow.dev/docs/how-it-works/code-transform\#flowjs)

Contains all workflow functions transformed in **workflow mode**. This file is imported by your framework to handle workflow execution requests at `POST /.well-known/workflow/v1/flow`.

**How it's structured:**

All workflow code is bundled together and embedded as a string inside `flow.js`. When a workflow needs to execute, this bundled code is run inside a **Node.js VM** (virtual machine) to ensure:

- **Determinism**: The same inputs always produce the same outputs
- **Side-effect prevention**: Direct access to Node.js APIs, file system, network, etc. is blocked
- **Sandboxed execution**: Workflow orchestration logic is isolated from the main runtime

**Build-time validation:**

The workflow mode transformation validates your code during the build:

- Catches invalid Node.js API usage (like `fs`, `http`, `child_process`)
- Prevents imports of modules that would break determinism

Most invalid patterns cause **build-time errors**, catching issues before deployment.

**What it does:**

- Exports a `POST` handler that accepts Web standard `Request` objects
- Executes bundled workflow code inside a Node.js VM for each request
- Handles workflow execution, replay, and resumption
- Returns execution results to the orchestration layer

**Why a VM?** Workflow functions must be deterministic to support replay. The VM sandbox prevents accidental use of non-deterministic APIs or side effects. All side effects should be performed in [step functions](https://useworkflow.dev/docs/foundations/workflows-and-steps#step-functions) instead.

### [`step.js`](https://useworkflow.dev/docs/how-it-works/code-transform\#stepjs)

Contains all step functions transformed in **step mode**. This file is imported by your framework to handle step execution requests at `POST /.well-known/workflow/v1/step`.

**What it does:**

- Exports a `POST` handler that accepts Web standard `Request` objects
- Executes individual steps with full runtime access
- Returns step results to the orchestration layer

### [`webhook.js`](https://useworkflow.dev/docs/how-it-works/code-transform\#webhookjs)

Contains webhook handling logic for delivering external data to running workflows via [`createWebhook()`](https://useworkflow.dev/docs/api-reference/workflow/create-webhook).

**What it does:**

- Exports a `POST` handler that accepts webhook payloads
- Validates tokens and routes data to the correct workflow run
- Resumes workflow execution after webhook delivery

**Note:** The webhook file structure varies by framework. Next.js generates `webhook/[token]/route.js` to leverage App Router's dynamic routing, while other frameworks generate a single `webhook.js` or `webhook.mjs` handler.

## [Why Three Modes?](https://useworkflow.dev/docs/how-it-works/code-transform\#why-three-modes)

The multi-mode transformation enables the Workflow DevKit's durable execution model:

1. **Step Mode** (required) - Bundles executable step functions that can access the full runtime
2. **Workflow Mode** (required) - Creates orchestration logic that can replay from event logs
3. **Client Mode** (optional) - Prevents direct execution and enables type-safe workflow references

This separation allows:

- **Deterministic replay**: Workflows can be safely replayed from event logs without re-executing side effects
- **Sandboxed orchestration**: Workflow logic runs in a controlled VM without direct runtime access
- **Stateless execution**: Your compute can scale to zero and resume from any point in the workflow
- **Type safety**: TypeScript works seamlessly with workflow references (when using client mode)

## [Determinism and Replay](https://useworkflow.dev/docs/how-it-works/code-transform\#determinism-and-replay)

A key aspect of the transformation is maintaining **deterministic replay** for workflow functions.

**Workflow functions must be deterministic:**

- Same inputs always produce the same outputs
- No direct side effects (no API calls, no database writes, no file I/O)
- Can use seeded random/time APIs provided by the VM (`Math.random()`, `Date.now()`, etc.)

Because workflow functions are deterministic and have no side effects, they can be safely re-run multiple times to calculate what the next step should be. This is why workflow function bodies remain intact in workflow mode—they're pure orchestration logic.

**Step functions can be non-deterministic:**

- Can make API calls, database queries, etc.
- Have full access to Node.js runtime and APIs
- Results are cached in the event log after first execution

Learn more about [Workflows and Steps](https://useworkflow.dev/docs/foundations/workflows-and-steps).

## [ID Generation](https://useworkflow.dev/docs/how-it-works/code-transform\#id-generation)

The compiler generates stable IDs for workflows and steps based on file paths and function names:

**Pattern:**`{type}//{filepath}//{functionName}`

**Examples:**

- `workflow//workflows/user-signup.js//handleUserSignup`
- `step//workflows/user-signup.js//createUser`
- `step//workflows/payments/checkout.ts//processPayment`

**Key properties:**

- **Stable**: IDs don't change unless you rename files or functions
- **Unique**: Each workflow/step has a unique identifier
- **Portable**: Works across different runtimes and deployments

Although IDs can change when files are moved or functions are renamed, Workflow DevKit function assume atomic versioning in the World. This means changing IDs won't break old workflows from running, but will prevent run from being upgraded and will cause your workflow/step names to change in the observability across deployments.

## [Framework Integration](https://useworkflow.dev/docs/how-it-works/code-transform\#framework-integration)

These transformations are framework-agnostic—they output standard JavaScript that works anywhere.

**For users**: Your framework handles all transformations automatically. See the [Getting Started](https://useworkflow.dev/docs/getting-started) guide for your framework.

**For framework authors**: Learn how to integrate these transformations into your framework in [Building Framework Integrations](https://useworkflow.dev/docs/how-it-works/framework-integrations).

## [Debugging Transformed Code](https://useworkflow.dev/docs/how-it-works/code-transform\#debugging-transformed-code)

If you need to debug transformation issues, you can inspect the generated files:

1. **Look in `.well-known/workflow/v1/`**: Check the generated `flow.js`, `step.js`,`webhook.js`, and other emitted debug files.
2. **Check build logs**: Most frameworks log transformation activity during builds
3. **Verify directives**: Ensure `"use workflow"` and `"use step"` are the first statements in functions
4. **Check file locations**: Transformations only apply to files in configured source directories

[Understanding Directives\\
\\
Previous Page](https://useworkflow.dev/docs/how-it-works/understanding-directives) [Framework Integrations\\
\\
Next Page](https://useworkflow.dev/docs/how-it-works/framework-integrations)

On this page

[Directives Overview](https://useworkflow.dev/docs/how-it-works/code-transform#directives-overview) [The Three Transformation Modes](https://useworkflow.dev/docs/how-it-works/code-transform#the-three-transformation-modes) [Comparison Table](https://useworkflow.dev/docs/how-it-works/code-transform#comparison-table) [Detailed Transformation Examples](https://useworkflow.dev/docs/how-it-works/code-transform#detailed-transformation-examples) [Generated Files](https://useworkflow.dev/docs/how-it-works/code-transform#generated-files) [`flow.js`](https://useworkflow.dev/docs/how-it-works/code-transform#flowjs) [`step.js`](https://useworkflow.dev/docs/how-it-works/code-transform#stepjs) [`webhook.js`](https://useworkflow.dev/docs/how-it-works/code-transform#webhookjs) [Why Three Modes?](https://useworkflow.dev/docs/how-it-works/code-transform#why-three-modes) [Determinism and Replay](https://useworkflow.dev/docs/how-it-works/code-transform#determinism-and-replay) [ID Generation](https://useworkflow.dev/docs/how-it-works/code-transform#id-generation) [Framework Integration](https://useworkflow.dev/docs/how-it-works/code-transform#framework-integration) [Debugging Transformed Code](https://useworkflow.dev/docs/how-it-works/code-transform#debugging-transformed-code)

[GitHubEdit this page on GitHub](https://github.com/vercel/workflow/edit/main/content/docs/how-it-works/code-transform.mdx) Scroll to topGive feedbackCopy pageAsk AI about this pageOpen in chat

## Vite Workflow Setup
[Getting Started](https://useworkflow.dev/docs/getting-started)

# Vite

This guide will walk through setting up your first workflow in a Vite app. Along the way, you'll learn more about the concepts that are fundamental to using the development kit in your own projects.

* * *

## [Create Your Vite Project](https://useworkflow.dev/docs/getting-started/vite\#create-your-vite-project)

Start by creating a new Vite project. This command will create a new directory named `my-workflow-app` with a minimal setup and setup a Vite project inside it.

```
npm create vite@latest my-workflow-app -- --template react-ts
```

Enter the newly made directory:

```
cd my-workflow-app
```

### [Install `workflow` and `nitro`](https://useworkflow.dev/docs/getting-started/vite\#install-workflow-and-nitro)

npm

pnpm

yarn

bun

```
npm i workflow nitro
```

While Vite provides the build tooling and development server, Nitro adds the server framework needed for API routes and deployment. Together they enable building full-stack applications with workflow support. Learn more about Nitro [here](https://v3.nitro.build/).

### [Configure Vite](https://useworkflow.dev/docs/getting-started/vite\#configure-vite)

Add `workflow()` to your Vite config. This enables usage of the `"use workflow"` and `"use step"` directives.

vite.config.ts

```

```

### \#\#\# [Setup IntelliSense for TypeScript (Optional)](https://useworkflow.dev/docs/getting-started/vite\\#setup-intellisense-for-typescript-optional)

## [Create Your First Workflow](https://useworkflow.dev/docs/getting-started/vite\#create-your-first-workflow)

Create a new file for our first workflow:

workflows/user-signup.ts

```

```

We'll fill in those functions next, but let's take a look at this code:

- We define a **workflow** function with the directive `"use workflow"`. Think of the workflow function as the _orchestrator_ of individual **steps**.
- The Workflow DevKit's `sleep` function allows us to suspend execution of the workflow without using up any resources. A sleep can be a few seconds, hours, days, or even months long.

## [Create Your Workflow Steps](https://useworkflow.dev/docs/getting-started/vite\#create-your-workflow-steps)

Let's now define those missing functions.

workflows/user-signup.ts

```

```

Taking a look at this code:

- Business logic lives inside **steps**. When a step is invoked inside a **workflow**, it gets enqueued to run on a separate request while the workflow is suspended, just like `sleep`.
- If a step throws an error, like in `sendWelcomeEmail`, the step will automatically be retried until it succeeds (or hits the step's max retry count).
- Steps can throw a `FatalError` if an error is intentional and should not be retried.

We'll dive deeper into workflows, steps, and other ways to suspend or handle events in [Foundations](https://useworkflow.dev/docs/foundations).

## [Create Your Route Handler](https://useworkflow.dev/docs/getting-started/vite\#create-your-route-handler)

To invoke your new workflow, we'll have to add your workflow to a `POST` API route handler, `api/signup.post.ts` with the following code:

api/signup.post.ts

```

```

This route handler creates a `POST` request endpoint at `/api/signup` that will trigger your workflow.

Workflows can be triggered from API routes or any server-side code.

## [Run in development](https://useworkflow.dev/docs/getting-started/vite\#run-in-development)

To start your development server, run the following command in your terminal in the Vite root directory:

```
npm run dev
```

Once your development server is running, you can trigger your workflow by running this command in the terminal:

```
curl -X POST --json '{"email":"hello@example.com"}' http://localhost:5173/api/signup
```

Check the Vite development server logs to see your workflow execute as well as the steps that are being processed.

Additionally, you can use the [Workflow DevKit CLI or Web UI](https://useworkflow.dev/docs/observability) to inspect your workflow runs and steps in detail.

```
npx workflow inspect runs
# or add '--web' for an interactive Web based UI
```

![Workflow DevKit Web UI](https://useworkflow.dev/o11y-ui.png)

* * *

## [Deploying to production](https://useworkflow.dev/docs/getting-started/vite\#deploying-to-production)

Workflow DevKit apps currently work best when deployed to [Vercel](https://vercel.com/home) and needs no special configuration.

Check the [Deploying](https://useworkflow.dev/docs/deploying) section to learn how your workflows can be deployed elsewhere.

## [Next Steps](https://useworkflow.dev/docs/getting-started/vite\#next-steps)

- Learn more about the [Foundations](https://useworkflow.dev/docs/foundations).
- Check [Errors](https://useworkflow.dev/docs/errors) if you encounter issues.
- Explore the [API Reference](https://useworkflow.dev/docs/api-reference).

[Next.js\\
\\
This guide will walk through setting up your first workflow in a Next.js app. Along the way, you'll learn more about the concepts that are fundamental to using the development kit in your own projects.](https://useworkflow.dev/docs/getting-started/next) [Express\\
\\
Next Page](https://useworkflow.dev/docs/getting-started/express)

On this page

[Create Your Vite Project](https://useworkflow.dev/docs/getting-started/vite#create-your-vite-project) [Install `workflow` and `nitro`](https://useworkflow.dev/docs/getting-started/vite#install-workflow-and-nitro) [Configure Vite](https://useworkflow.dev/docs/getting-started/vite#configure-vite) [Setup IntelliSense for TypeScript (Optional)](https://useworkflow.dev/docs/getting-started/vite#setup-intellisense-for-typescript-optional) [Create Your First Workflow](https://useworkflow.dev/docs/getting-started/vite#create-your-first-workflow) [Create Your Workflow Steps](https://useworkflow.dev/docs/getting-started/vite#create-your-workflow-steps) [Create Your Route Handler](https://useworkflow.dev/docs/getting-started/vite#create-your-route-handler) [Run in development](https://useworkflow.dev/docs/getting-started/vite#run-in-development) [Deploying to production](https://useworkflow.dev/docs/getting-started/vite#deploying-to-production) [Next Steps](https://useworkflow.dev/docs/getting-started/vite#next-steps)

[GitHubEdit this page on GitHub](https://github.com/vercel/workflow/edit/main/content/docs/getting-started/vite.mdx) Scroll to topGive feedbackCopy pageAsk AI about this pageOpen in chat

## Local World Workflow
[Deploying](https://useworkflow.dev/docs/deploying) [World](https://useworkflow.dev/docs/deploying/world)

# Local World

The **Local World** (`@workflow/world-local`) is a filesystem-based workflow backend designed for local development and testing. It stores workflow data as JSON files on disk and provides in-memory queuing.

The local world is perfect for local development because it:

- Requires no external services or configuration
- Stores data as readable JSON files for easy debugging
- Provides instant feedback during development
- Works seamlessly with Next.js development server

## [How It Works](https://useworkflow.dev/docs/deploying/world/local-world\#how-it-works)

### [Storage](https://useworkflow.dev/docs/deploying/world/local-world\#storage)

The local world stores all workflow data as JSON files in a configurable directory:

```
.workflow-data/
├── runs/
│   └── <run-id>.json
├── steps/
│   └── <run-id>/
│       └── <step-id>.json
├── hooks/
│   └── <hook-id>.json
└── streams/
    └── <run-id>/
        └── <stream-id>.json
```

Each file contains the full state of a run, step, hook, or stream, making it easy to inspect workflow data directly.

### [Queuing](https://useworkflow.dev/docs/deploying/world/local-world\#queuing)

The local world uses an in-memory queue with HTTP transport:

1. When a step is enqueued, it's added to an in-memory queue
2. The queue processes steps by sending HTTP requests to your development server
3. Steps are executed at the `.well-known/workflow/v1/step` endpoint

The queue automatically detects your development server's port and adjusts the queue URL accordingly.

### [Authentication](https://useworkflow.dev/docs/deploying/world/local-world\#authentication)

The local world provides a simple authentication implementation since no authentication is required or enforced in local development.

```
getAuthHeaders(): Promise<Record<string, string>> {
  return Promise.resolve({});
}
```

## [Configuration](https://useworkflow.dev/docs/deploying/world/local-world\#configuration)

### [Data Directory](https://useworkflow.dev/docs/deploying/world/local-world\#data-directory)

By default, workflow data is stored in `.workflow-data/` in your project root. This can be customized through environment variables or programmatically.

**Environment variable:**

```
export WORKFLOW_EMBEDDED_DATA_DIR=./custom-workflow-data
```

**Programmatically:**

```
import { createEmbeddedWorld } from '@workflow/world-local';

const world = createEmbeddedWorld({ dataDir: './custom-workflow-data' });
```

### [Port](https://useworkflow.dev/docs/deploying/world/local-world\#port)

By default, the embedded world **automatically detects** which port your application is listening on using process introspection. This works seamlessly with frameworks like SvelteKit, Vite, and others that use non-standard ports.

**Auto-detection example** (recommended):

```
import { createEmbeddedWorld } from '@workflow/world-local';

// Port is automatically detected - no configuration needed!
const world = createEmbeddedWorld();
```

If auto-detection fails, the world will fall back to the `PORT` environment variable, then to port `3000`.

**Manual port override** (when needed):

You can override the auto-detected port by specifying it explicitly:

```
import { createEmbeddedWorld } from '@workflow/world-local';

const world = createEmbeddedWorld({ port: 3000 });
```

### [Base URL](https://useworkflow.dev/docs/deploying/world/local-world\#base-url)

For advanced use cases like HTTPS or custom hostnames, you can override the entire base URL. When set, this takes precedence over all port detection and configuration.

**Use cases:**

- HTTPS dev servers (e.g., `next dev --experimental-https`)
- Custom hostnames (e.g., `local.example.com`)
- Non-localhost development

**Environment variable:**

```
export WORKFLOW_EMBEDDED_BASE_URL=https://local.example.com:3000
```

**Programmatically:**

```
import { createEmbeddedWorld } from '@workflow/world-local';

// HTTPS
const world = createEmbeddedWorld({
  baseUrl: 'https://localhost:3000'
});

// Custom hostname
const world = createEmbeddedWorld({
  baseUrl: 'https://local.example.com:3000'
});
```

## [Usage](https://useworkflow.dev/docs/deploying/world/local-world\#usage)

### [Automatic (Recommended)](https://useworkflow.dev/docs/deploying/world/local-world\#automatic-recommended)

The local world is used automatically during local development:

```
# Start your Next.js dev server
npm run dev

# Workflows automatically use local world
```

### [Manual](https://useworkflow.dev/docs/deploying/world/local-world\#manual)

You can explicitly set the local world through environment variables:

```
export WORKFLOW_TARGET_WORLD=embedded

npm run dev
```

## [Development Workflow](https://useworkflow.dev/docs/deploying/world/local-world\#development-workflow)

A typical development workflow with local world:

1. **Start your dev server:**



```
npm run dev
```

2. **Trigger a workflow:**



```
curl -X POST --json '{"email":"test@example.com"}' http://localhost:3000/api/signup
```

3. **Inspect the results:**
   - Use the [CLI or Web UI](https://useworkflow.dev/docs/observability)
   - Check JSON files in `.workflow-data/`
   - View development server logs

## [Inspecting Data](https://useworkflow.dev/docs/deploying/world/local-world\#inspecting-data)

### [Using Observability Tools](https://useworkflow.dev/docs/deploying/world/local-world\#using-observability-tools)

The local world integrates with the Workflow DevKit's observability tools:

```
# View runs with CLI
npx workflow inspect runs

# View runs with Web UI
npx workflow inspect runs --web
```

Learn more in the [Observability](https://useworkflow.dev/docs/observability) section.

## [Limitations](https://useworkflow.dev/docs/deploying/world/local-world\#limitations)

The local world is designed for development, not production:

- **Not scalable** \- Uses in-memory queuing
- **Not persistent** \- Data is stored in local files
- **Single instance** \- Cannot handle distributed deployments
- **No authentication** \- Suitable only for local development

For production deployments, use the [Vercel World](https://useworkflow.dev/docs/deploying/world/vercel-world).

## [API Reference](https://useworkflow.dev/docs/deploying/world/local-world\#api-reference)

### [`createEmbeddedWorld`](https://useworkflow.dev/docs/deploying/world/local-world\#createembeddedworld)

Creates a local world instance:

```
function createEmbeddedWorld(
  args?: Partial<{
    dataDir: string;
    port: number;
    baseUrl: string;
  }>
): World
```

**Parameters:**

- `args` \- Optional configuration object:
  - `dataDir` \- Directory for storing workflow data (default: `.workflow-data/` or `WORKFLOW_EMBEDDED_DATA_DIR` env var)
  - `port` \- Port override for queue transport (default: auto-detected → `PORT` env var → `3000`)
  - `baseUrl` \- Full base URL override for queue transport (default: `http://localhost:{port}` or `WORKFLOW_EMBEDDED_BASE_URL` env var)

**Returns:**

- `World` \- A world instance implementing the World interface

**Examples:**

```
import { createEmbeddedWorld } from '@workflow/world-local';

// Use all defaults (recommended - auto-detects port)
const world = createEmbeddedWorld();

// Custom data directory
const world = createEmbeddedWorld({ dataDir: './my-data' });

// Override port
const world = createEmbeddedWorld({ port: 3000 });

// HTTPS with custom hostname
const world = createEmbeddedWorld({
  baseUrl: 'https://local.example.com:3000'
});

// Multiple options
const world = createEmbeddedWorld({
  dataDir: './my-data',
  baseUrl: 'https://localhost:3000'
});
```

## [Learn More](https://useworkflow.dev/docs/deploying/world/local-world\#learn-more)

- [World Interface](https://useworkflow.dev/docs/deploying/world) \- Understanding the World interface
- [Vercel World](https://useworkflow.dev/docs/deploying/world/vercel-world) \- For production deployments
- [Observability](https://useworkflow.dev/docs/observability) \- Monitoring and debugging tools

[Worlds\\
\\
Previous Page](https://useworkflow.dev/docs/deploying/world) [Vercel World\\
\\
Next Page](https://useworkflow.dev/docs/deploying/world/vercel-world)

On this page

[How It Works](https://useworkflow.dev/docs/deploying/world/local-world#how-it-works) [Storage](https://useworkflow.dev/docs/deploying/world/local-world#storage) [Queuing](https://useworkflow.dev/docs/deploying/world/local-world#queuing) [Authentication](https://useworkflow.dev/docs/deploying/world/local-world#authentication) [Configuration](https://useworkflow.dev/docs/deploying/world/local-world#configuration) [Data Directory](https://useworkflow.dev/docs/deploying/world/local-world#data-directory) [Port](https://useworkflow.dev/docs/deploying/world/local-world#port) [Base URL](https://useworkflow.dev/docs/deploying/world/local-world#base-url) [Usage](https://useworkflow.dev/docs/deploying/world/local-world#usage) [Automatic (Recommended)](https://useworkflow.dev/docs/deploying/world/local-world#automatic-recommended) [Manual](https://useworkflow.dev/docs/deploying/world/local-world#manual) [Development Workflow](https://useworkflow.dev/docs/deploying/world/local-world#development-workflow) [Inspecting Data](https://useworkflow.dev/docs/deploying/world/local-world#inspecting-data) [Using Observability Tools](https://useworkflow.dev/docs/deploying/world/local-world#using-observability-tools) [Limitations](https://useworkflow.dev/docs/deploying/world/local-world#limitations) [API Reference](https://useworkflow.dev/docs/deploying/world/local-world#api-reference) [`createEmbeddedWorld`](https://useworkflow.dev/docs/deploying/world/local-world#createembeddedworld) [Learn More](https://useworkflow.dev/docs/deploying/world/local-world#learn-more)

[GitHubEdit this page on GitHub](https://github.com/vercel/workflow/edit/main/content/docs/deploying/world/local-world.mdx) Scroll to topGive feedbackCopy pageAsk AI about this pageOpen in chat

## Understanding Idempotency
[Foundations](https://useworkflow.dev/docs/foundations)

# Idempotency

Idempotency is a property of an operation that ensures it can be safely retried without producing duplicate side effects.

In distributed systems (calling external APIs), it is not always possible to ensure an operation has only been performed once just by seeing if it succeeds.
Consider a payment API that charges the user $10, but due to network failures, the confirmation response is lost. When the step retries (because the previous attempt was considered a failure), it will charge the user again.

To prevent this, many external APIs support idempotency keys. An idempotency key is a unique identifier for an operation that can be used to deduplicate requests.

## [The core pattern: use the step ID as your idempotency key](https://useworkflow.dev/docs/foundations/idempotency\#the-core-pattern-use-the-step-id-as-your-idempotency-key)

Every step invocation has a stable `stepId` that stays the same across retries.
Use it as the idempotency key when calling third-party APIs.

```
import { getStepMetadata } from "workflow";

async function chargeUser(userId: string, amount: number) {
  "use step";

  const { stepId } = getStepMetadata();

  // Example: Stripe-style idempotency key
  // This guarantees only one charge is created even if the step retries
  await stripe.charges.create(
    {
      amount,
      currency: "usd",
      customer: userId,
    },
    {
      idempotencyKey: stepId,
    }
  );
}
```

Why this works:

- **Stable across retries**: `stepId` does not change between attempts.
- **Globally unique per step**: Fulfills the uniqueness requirement for an idempotency key.

## [Best practices](https://useworkflow.dev/docs/foundations/idempotency\#best-practices)

- **Always provide idempotency keys to external side effects that are not idempotent** inside steps (payments, emails, SMS, queues).
- **Prefer `stepId` as your key**; it is stable across retries and unique per step.
- **Keep keys deterministic**; avoid including timestamps or attempt counters.
- **Handle 409/conflict responses** gracefully; treat them as success if the prior attempt completed.

## [Related docs](https://useworkflow.dev/docs/foundations/idempotency\#related-docs)

- Learn about retries in [Errors & Retrying](https://useworkflow.dev/docs/foundations/errors-and-retries)
- API reference: [`getStepMetadata`](https://useworkflow.dev/docs/api-reference/workflow/get-step-metadata)

[Serialization\\
\\
Previous Page](https://useworkflow.dev/docs/foundations/serialization) [Understanding Directives\\
\\
Next Page](https://useworkflow.dev/docs/how-it-works/understanding-directives)

On this page

[The core pattern: use the step ID as your idempotency key](https://useworkflow.dev/docs/foundations/idempotency#the-core-pattern-use-the-step-id-as-your-idempotency-key) [Best practices](https://useworkflow.dev/docs/foundations/idempotency#best-practices) [Related docs](https://useworkflow.dev/docs/foundations/idempotency#related-docs)

[GitHubEdit this page on GitHub](https://github.com/vercel/workflow/edit/main/content/docs/foundations/idempotency.mdx) Scroll to topGive feedbackCopy pageAsk AI about this pageOpen in chat

## Framework Integrations Guide
How it works

# Framework Integrations

**For users:** If you just want to use Workflow DevKit with an existing framework, check out the [Getting Started](https://useworkflow.dev/docs/getting-started) guide instead. This page is for framework authors who want to integrate Workflow DevKit with their framework or runtime.

This guide walks you through building a framework integration for Workflow DevKit using Bun as a concrete example. The same principles apply to any JavaScript runtime (Node.js, Deno, Cloudflare Workers, etc.).

**Prerequisites:** Before building a framework integration, we recommend reading [How the Directives Work](https://useworkflow.dev/docs/how-it-works/code-transform) to understand the transformation system that powers Workflow DevKit.

## [What You'll Build](https://useworkflow.dev/docs/how-it-works/framework-integrations\#what-youll-build)

A framework integration has two main components:

1. **Build-time**: Generate workflow handler files (`flow.js`, `step.js`, `webhook.js`)
2. **Runtime**: Expose these handlers as HTTP endpoints in your application server

The purple boxes are what you implement—everything else is provided by Workflow DevKit.

## [Example: Bun Integration](https://useworkflow.dev/docs/how-it-works/framework-integrations\#example-bun-integration)

Let's build a complete integration for Bun. Bun is unique because it serves as both a runtime (needs code transformations) and a framework (provides `Bun.serve()` for HTTP routing).

A working example can be [found here](https://github.com/vercel/workflow-examples/tree/main/custom-adapter). For a production-ready reference, see the [Next.js integration](https://github.com/vercel/workflow/tree/main/packages/next).

### [Step 1: Generate Handler Files](https://useworkflow.dev/docs/how-it-works/framework-integrations\#step-1-generate-handler-files)

Use the `workflow` CLI to generate the handler bundles. The CLI scans your `workflows/` directory and creates `flow.js`, `step.js`, and `webhook.js`.

package.json

```

```

**For production integrations:** Instead of using the CLI, extend the `BaseBuilder` class directly in your framework plugin. This gives you control over file watching, custom output paths, and framework-specific hooks. See the [Next.js plugin](https://github.com/vercel/workflow/tree/main/packages/next) for an example.

**What gets generated:**

- `/.well-known/workflow/v1/flow.js` \- Handles workflow execution (workflow mode transform)
- `/.well-known/workflow/v1/step.js` \- Handles step execution (step mode transform)
- `/.well-known/workflow/v1/webhook.js` \- Handles webhook delivery

Each file exports a `POST` function that accepts Web standard `Request` objects.

### [Step 2: Add Client Mode Transform (Optional)](https://useworkflow.dev/docs/how-it-works/framework-integrations\#step-2-add-client-mode-transform-optional)

Client mode transforms your application code to provide better DX. Add a Bun plugin to apply this transformation at runtime:

workflow-plugin.ts

```

```

Activate the plugin in `bunfig.toml`:

bunfig.toml

```

```

**What this does:**

- Attaches workflow IDs to functions for use with `start()`
- Provides TypeScript type safety
- Prevents accidental direct execution of workflows

**Why optional?** Without client mode, you can still use workflows by manually constructing IDs or referencing the build manifest.

### [Step 3: Expose HTTP Endpoints](https://useworkflow.dev/docs/how-it-works/framework-integrations\#step-3-expose-http-endpoints)

Wire up the generated handlers to HTTP endpoints using `Bun.serve()`:

server.ts

```

```

**That's it!** Your Bun integration is complete.

## [Understanding the Endpoints](https://useworkflow.dev/docs/how-it-works/framework-integrations\#understanding-the-endpoints)

Your integration must expose three HTTP endpoints. The generated handlers manage all protocol details—you just route requests.

### [Workflow Endpoint](https://useworkflow.dev/docs/how-it-works/framework-integrations\#workflow-endpoint)

**Route:**`POST /.well-known/workflow/v1/flow`

Executes workflow orchestration logic. The workflow function is "rendered" multiple times during execution—each time it progresses until it encounters the next step.

**Called when:**

- Starting a new workflow
- Resuming after a step completes
- Resuming after a webhook or hook triggers
- Recovering from failures

### [Step Endpoint](https://useworkflow.dev/docs/how-it-works/framework-integrations\#step-endpoint)

**Route:**`POST /.well-known/workflow/v1/step`

Executes individual atomic operations within workflows. Each step runs exactly once per execution (unless retried due to failure). Steps have full runtime access (Node.js APIs, file system, databases, etc.).

### [Webhook Endpoint](https://useworkflow.dev/docs/how-it-works/framework-integrations\#webhook-endpoint)

**Route:**`POST /.well-known/workflow/v1/webhook/:token`

Delivers webhook data to running workflows via [`createWebhook()`](https://useworkflow.dev/docs/api-reference/workflow/create-webhook). The `:token` parameter identifies which workflow run should receive the data.

The webhook file structure varies by framework. Next.js generates `webhook/[token]/route.js` to leverage App Router's dynamic routing, while other frameworks generate a single `webhook.js` handler.

## [Adapting to Other Frameworks](https://useworkflow.dev/docs/how-it-works/framework-integrations\#adapting-to-other-frameworks)

The Bun example demonstrates the core pattern. To adapt for your framework:

### [Build-Time](https://useworkflow.dev/docs/how-it-works/framework-integrations\#build-time)

**Option 1: Use the CLI** (simplest)

```
workflow build
```

This will default to scanning the `./workflows` top-level directory for workflow files, and will output bundled files directly into your working directory.

**Option 2: Extend `BaseBuilder`** (recommended)

```
import { BaseBuilder } from '@workflow/cli/dist/lib/builders/base-builder';

class MyFrameworkBuilder extends BaseBuilder {
  constructor(options) {
    super({
      dirs: ['workflows'],
      workingDir: options.rootDir,
      watch: options.dev,
    });
  }

  override async build(): Promise<void> {
    const inputFiles = await this.getInputFiles();

    await this.createWorkflowsBundle({
      outfile: '/path/to/.well-known/workflow/v1/flow.js',
      format: 'esm',
      inputFiles,
    });

    await this.createStepsBundle({
      outfile: '/path/to/.well-known/workflow/v1/step.js',
      format: 'esm',
      inputFiles,
    });

    await this.createWebhookBundle({
      outfile: '/path/to/.well-known/workflow/v1/webhook.js',
    });
  }
}
```

If your framework supports virtual server routes and dev mode watching, make sure to adapt accordingly. Please open a PR to the Workflow DevKit if the base builder class is missing necessary functionality.

Hook into your framework's build:

pseudocode.ts

```

```

### [Runtime (Client Mode)](https://useworkflow.dev/docs/how-it-works/framework-integrations\#runtime-client-mode)

Add a loader/plugin for your bundler:

**Rollup/Vite:**

```
export function workflowPlugin() {
  return {
    name: 'workflow-client-transform',
    async transform(code, id) {
      if (!code.match(/(use step|use workflow)/)) return null;

      const result = await transform(code, {
        filename: id,
        jsc: {
          experimental: {
            plugins: [[require.resolve("@workflow/swc-plugin"), { mode: "client" }]],
          },
        },
      });

      return { code: result.code, map: result.map };
    },
  };
}
```

**Webpack:**

```
module.exports = {
  module: {
    rules: [\
      {\
        test: /\.(ts|tsx|js|jsx)$/,\
        use: 'workflow-client-loader', // Similar implementation\
      },\
    ],
  },
};
```

### [HTTP Server](https://useworkflow.dev/docs/how-it-works/framework-integrations\#http-server)

Route the three endpoints to the generated handlers. The exact implementation depends on your framework's routing API.

In the bun example above, we left routing to the user. Essentially, the user has to serve routes like this:

server.ts

```

```

Production framework integrations should handle this routing in the plugin instead of leaving it to the user, and this depends on each framework's unique implementaiton.
Check the Workflow DevKit source code for examples of production framework implementations.
In the future, the Workflow DevKit will emit more routes under the `.well-known/workflow` namespace.

## [Security](https://useworkflow.dev/docs/how-it-works/framework-integrations\#security)

**How are these HTTP endpoints secured?**

Security is handled by the **world abstraction** you're using:

**Vercel (`@workflow/world-vercel`):**

- Vercel Queue will support private invoke, making routes inaccessible from the public internet
- Handlers receive only a message ID that must be retrieved from Vercel's backend
- Impossible to craft custom payloads without valid queue-issued message IDs

**Custom implementations:**

- Implement authentication via framework middleware
- Use API keys, JWT validation, or other auth schemes
- Network-level security (VPCs, private networks, firewall rules)
- Rate limiting and request validation

Learn more about [world abstractions](https://useworkflow.dev/docs/deploying/world).

## [Testing Your Integration](https://useworkflow.dev/docs/how-it-works/framework-integrations\#testing-your-integration)

### [1\. Test Build Output](https://useworkflow.dev/docs/how-it-works/framework-integrations\#1-test-build-output)

Create a test workflow:

workflows/test.ts

```

```

Run your build and verify:

- `.well-known/workflow/v1/flow.js` exists
- `.well-known/workflow/v1/step.js` exists
- `.well-known/workflow/v1/webhook.js` exists

### [2\. Test HTTP Endpoints](https://useworkflow.dev/docs/how-it-works/framework-integrations\#2-test-http-endpoints)

Start your server and verify routes respond:

```
curl -X POST http://localhost:3000/.well-known/workflow/v1/flow
curl -X POST http://localhost:3000/.well-known/workflow/v1/step
curl -X POST http://localhost:3000/.well-known/workflow/v1/webhook/test
```

(Should respond but not trigger meaningful code without authentication/proper workflow run)

### [3\. Run a Workflow End-to-End](https://useworkflow.dev/docs/how-it-works/framework-integrations\#3-run-a-workflow-end-to-end)

```
import { start } from "workflow/api";
import { handleUserSignup } from "./workflows/test";

const run = await start(handleUserSignup, ["test@example.com"]);
console.log("Workflow started:", run.runId);
```

[How the Directives Work\\
\\
Previous Page](https://useworkflow.dev/docs/how-it-works/code-transform) [Observability\\
\\
Next Page](https://useworkflow.dev/docs/observability)

On this page

[What You'll Build](https://useworkflow.dev/docs/how-it-works/framework-integrations#what-youll-build) [Example: Bun Integration](https://useworkflow.dev/docs/how-it-works/framework-integrations#example-bun-integration) [Step 1: Generate Handler Files](https://useworkflow.dev/docs/how-it-works/framework-integrations#step-1-generate-handler-files) [Step 2: Add Client Mode Transform (Optional)](https://useworkflow.dev/docs/how-it-works/framework-integrations#step-2-add-client-mode-transform-optional) [Step 3: Expose HTTP Endpoints](https://useworkflow.dev/docs/how-it-works/framework-integrations#step-3-expose-http-endpoints) [Understanding the Endpoints](https://useworkflow.dev/docs/how-it-works/framework-integrations#understanding-the-endpoints) [Workflow Endpoint](https://useworkflow.dev/docs/how-it-works/framework-integrations#workflow-endpoint) [Step Endpoint](https://useworkflow.dev/docs/how-it-works/framework-integrations#step-endpoint) [Webhook Endpoint](https://useworkflow.dev/docs/how-it-works/framework-integrations#webhook-endpoint) [Adapting to Other Frameworks](https://useworkflow.dev/docs/how-it-works/framework-integrations#adapting-to-other-frameworks) [Build-Time](https://useworkflow.dev/docs/how-it-works/framework-integrations#build-time) [Runtime (Client Mode)](https://useworkflow.dev/docs/how-it-works/framework-integrations#runtime-client-mode) [HTTP Server](https://useworkflow.dev/docs/how-it-works/framework-integrations#http-server) [Security](https://useworkflow.dev/docs/how-it-works/framework-integrations#security) [Testing Your Integration](https://useworkflow.dev/docs/how-it-works/framework-integrations#testing-your-integration) [1\. Test Build Output](https://useworkflow.dev/docs/how-it-works/framework-integrations#1-test-build-output) [2\. Test HTTP Endpoints](https://useworkflow.dev/docs/how-it-works/framework-integrations#2-test-http-endpoints) [3\. Run a Workflow End-to-End](https://useworkflow.dev/docs/how-it-works/framework-integrations#3-run-a-workflow-end-to-end)

[GitHubEdit this page on GitHub](https://github.com/vercel/workflow/edit/main/content/docs/how-it-works/framework-integrations.mdx) Scroll to topGive feedbackCopy pageAsk AI about this pageOpen in chat

## Durable AI Agents
[API Reference](https://useworkflow.dev/docs/api-reference) [@workflow/ai](https://useworkflow.dev/docs/api-reference/workflow-ai)

# DurableAgent

The `@workflow/ai` package is currently in active development and should be considered experimental.

The `DurableAgent` class enables you to create AI-powered agents that can maintain state across workflow steps, call tools, and gracefully handle interruptions and resumptions.

Tool calls can be implemented as workflow steps for automatic retries, or as regular workflow-level logic utilizing core library features such as [`sleep()`](https://useworkflow.dev/docs/api-reference/workflow/sleep) and [Hooks](https://useworkflow.dev/docs/foundations/hooks).

```
import { DurableAgent } from '@workflow/ai/agent';
import { getWritable } from 'workflow';
import { z } from 'zod';
import type { UIMessageChunk } from 'ai';

async function getWeather({ city }: { city: string }) {
  "use step";

  return `Weather in ${city} is sunny`;
}

async function myAgent() {
  "use workflow";

  const agent = new DurableAgent({
    model: 'anthropic/claude-haiku-4.5',
    system: 'You are a helpful weather assistant.',
    tools: {
      getWeather: {
        description: 'Get weather for a city',
        inputSchema: z.object({ city: z.string() }),
        execute: getWeather,
      },
    },
  });

  // The agent will stream its output to the workflow
  // run's default output stream
  const writable = getWritable<UIMessageChunk>();

  await agent.stream({
    messages: [{ role: 'user', content: 'How is the weather in San Francisco?' }],
    writable,
  });
}
```

## [API Signature](https://useworkflow.dev/docs/api-reference/workflow-ai/durable-agent\#api-signature)

### [Class](https://useworkflow.dev/docs/api-reference/workflow-ai/durable-agent\#class)

| Name | Type | Description |
| --- | --- | --- |
| `model` | any |  |
| `tools` | any |  |
| `system` | any |  |
| `generate` | () =\> void |  |
| `stream` | <TTools extends ToolSet = ToolSet>(options: DurableAgentStreamOptions<TTools>) => Promise<{ messages: ModelMessage\[\]; }> |  |

### [DurableAgentOptions](https://useworkflow.dev/docs/api-reference/workflow-ai/durable-agent\#durableagentoptions)

| Name | Type | Description |
| --- | --- | --- |
| `model` | string \| (() => Promise<LanguageModelV2>) | The model provider to use for the agent.<br>This should be a string compatible with the Vercel AI Gateway (e.g., 'anthropic/claude-opus'),<br>or a step function that returns a `LanguageModelV2` instance. |
| `tools` | ToolSet | A set of tools available to the agent.<br>Tools can be implemented as workflow steps for automatic retries and persistence,<br>or as regular workflow-level logic using core library features like sleep() and Hooks. |
| `system` | string | Optional system prompt to guide the agent's behavior. |

### [DurableAgentStreamOptions](https://useworkflow.dev/docs/api-reference/workflow-ai/durable-agent\#durableagentstreamoptions)

| Name | Type | Description |
| --- | --- | --- |
| `messages` | ModelMessage\[\] | The conversation messages to process. Should follow the AI SDK's ModelMessage format. |
| `system` | string | Optional system prompt override. If provided, overrides the system prompt from the constructor. |
| `writable` | WritableStream<UIMessageChunk> | The stream to which the agent writes message chunks. For example, use `getWritable<UIMessageChunk>()` to write to the workflow's default output stream. |
| `preventClose` | boolean | If true, prevents the writable stream from being closed after streaming completes.<br>Defaults to false (stream will be closed). |
| `sendStart` | boolean | If true, sends a 'start' chunk at the beginning of the stream.<br>Defaults to true. |
| `sendFinish` | boolean | If true, sends a 'finish' chunk at the end of the stream.<br>Defaults to true. |
| `stopWhen` | StopCondition<NoInfer<ToolSet>> \| StopCondition<NoInfer<ToolSet>>\[\] | Condition for stopping the generation when there are tool results in the last step.<br>When the condition is an array, any of the conditions can be met to stop the generation. |
| `onStepFinish` | StreamTextOnStepFinishCallback<any> | Callback function to be called after each step completes. |
| `prepareStep` | PrepareStepCallback<TTools> | Callback function called before each step in the agent loop.<br>Use this to modify settings, manage context, or inject messages dynamically. |

## [Key Features](https://useworkflow.dev/docs/api-reference/workflow-ai/durable-agent\#key-features)

- **Durable Execution**: Agents can be interrupted and resumed without losing state
- **Flexible Tool Implementation**: Tools can be implemented as workflow steps for automatic retries, or as regular workflow-level logic
- **Stream Processing**: Handles streaming responses and tool calls in a structured way
- **Workflow Native**: Fully integrated with Workflow DevKit for production-grade reliability

## [Good to Know](https://useworkflow.dev/docs/api-reference/workflow-ai/durable-agent\#good-to-know)

- Tools can be implemented as workflow steps (using `"use step"` for automatic retries), or as regular workflow-level logic
- Tools can use core library features like `sleep()` and Hooks within their `execute` functions
- The agent processes tool calls iteratively until completion
- The `stream()` method returns `{ messages }` containing the full conversation history, including initial messages, assistant responses, and tool results

## [Examples](https://useworkflow.dev/docs/api-reference/workflow-ai/durable-agent\#examples)

### [Basic Agent with Tools](https://useworkflow.dev/docs/api-reference/workflow-ai/durable-agent\#basic-agent-with-tools)

```
import { DurableAgent } from '@workflow/ai/agent';
import { getWritable } from 'workflow';
import { z } from 'zod';
import type { UIMessageChunk } from 'ai';

async function getWeather({ location }: { location: string }) {
  "use step";
  // Fetch weather data
  const response = await fetch(`https://api.weather.com?location=${location}`);
  return response.json();
}

async function weatherAgentWorkflow(userQuery: string) {
  'use workflow';

  const agent = new DurableAgent({
    model: 'anthropic/claude-haiku-4.5',
    tools: {
      getWeather: {
        description: 'Get current weather for a location',
        inputSchema: z.object({ location: z.string() }),
        execute: getWeather,
      },
    },
    system: 'You are a helpful weather assistant. Always provide accurate weather information.',
  });

  await agent.stream({
    messages: [\
      {\
        role: 'user',\
        content: userQuery,\
      },\
    ],
    writable: getWritable<UIMessageChunk>(),
  });
}
```

### [Multiple Tools](https://useworkflow.dev/docs/api-reference/workflow-ai/durable-agent\#multiple-tools)

```
import { DurableAgent } from '@workflow/ai/agent';
import { getWritable } from 'workflow';
import { z } from 'zod';
import type { UIMessageChunk } from 'ai';

async function getWeather({ location }: { location: string }) {
  "use step";
  return `Weather in ${location}: Sunny, 72°F`;
}

async function searchEvents({ location, category }: { location: string; category: string }) {
  "use step";
  return `Found 5 ${category} events in ${location}`;
}

async function multiToolAgentWorkflow(userQuery: string) {
  'use workflow';

  const agent = new DurableAgent({
    model: 'anthropic/claude-haiku-4.5',
    tools: {
      getWeather: {
        description: 'Get weather for a location',
        inputSchema: z.object({ location: z.string() }),
        execute: getWeather,
      },
      searchEvents: {
        description: 'Search for upcoming events in a location',
        inputSchema: z.object({ location: z.string(), category: z.string() }),
        execute: searchEvents,
      },
    },
  });

  await agent.stream({
    messages: [\
      {\
        role: 'user',\
        content: userQuery,\
      },\
    ],
    writable: getWritable<UIMessageChunk>(),
  });
}
```

### [Multi-turn Conversation](https://useworkflow.dev/docs/api-reference/workflow-ai/durable-agent\#multi-turn-conversation)

```
import { DurableAgent } from '@workflow/ai/agent';
import { z } from 'zod';

async function searchProducts({ query }: { query: string }) {
  "use step";
  // Search product database
  return `Found 3 products matching "${query}"`;
}

async function multiTurnAgentWorkflow() {
  'use workflow';

  const agent = new DurableAgent({
    model: 'anthropic/claude-haiku-4.5',
    tools: {
      searchProducts: {
        description: 'Search for products',
        inputSchema: z.object({ query: z.string() }),
        execute: searchProducts,
      },
    },
  });

  const writable = getWritable<UIMessageChunk>();

  // First user message
  //   - Result is streamed to the provided `writable` stream
  //   - Message history is returned in `messages` for LLM context
  let { messages } = await agent.stream({
    messages: [\
      { role: 'user', content: 'Find me some laptops' }\
    ],
    writable,
  });

  // Continue the conversation with the accumulated message history
  const result = await agent.stream({
    messages: [\
      ...messages,\
      { role: 'user', content: 'Which one has the best battery life?' }\
    ],
    writable,
  });

  // result.messages now contains the complete conversation history
  return result.messages;
}
```

### [Tools with Workflow Library Features](https://useworkflow.dev/docs/api-reference/workflow-ai/durable-agent\#tools-with-workflow-library-features)

```
import { DurableAgent } from '@workflow/ai/agent';
import { sleep, defineHook, getWritable } from 'workflow';
import { z } from 'zod';
import type { UIMessageChunk } from 'ai';

// Define a reusable hook type
const approvalHook = defineHook<{ approved: boolean; reason: string }>();

async function scheduleTask({ delaySeconds }: { delaySeconds: number }) {
  // Note: No "use step" for this tool call,
  // since `sleep()` is a workflow level function
  await sleep(`${delaySeconds}s`);
  return `Slept for ${delaySeconds} seconds`;
}

async function requestApproval({ message }: { message: string }) {
  // Note: No "use step" for this tool call either,
  // since hooks are awaited at the workflow level

  // Utilize a Hook for Human-in-the-loop approval
  const hook = approvalHook.create({
    metadata: { message }
  });

  console.log(`Approval needed - token: ${hook.token}`);

  // Wait for the approval payload
  const approval = await hook;

  if (approval.approved) {
    return `Request approved: ${approval.reason}`;
  } else {
    throw new Error(`Request denied: ${approval.reason}`);
  }
}

async function agentWithLibraryFeaturesWorkflow(userRequest: string) {
  'use workflow';

  const agent = new DurableAgent({
    model: 'anthropic/claude-haiku-4.5',
    tools: {
      scheduleTask: {
        description: 'Pause the workflow for the specified number of seconds',
        inputSchema: z.object({
          delaySeconds: z.number(),
        }),
        execute: scheduleTask,
      },
      requestApproval: {
        description: 'Request approval for an action',
        inputSchema: z.object({ message: z.string() }),
        execute: requestApproval,
      },
    },
  });

  await agent.stream({
    messages: [{ role: 'user', content: userRequest }],
    writable: getWritable<UIMessageChunk>(),
  });
}
```

## [See Also](https://useworkflow.dev/docs/api-reference/workflow-ai/durable-agent\#see-also)

- [WorkflowChatTransport](https://useworkflow.dev/docs/api-reference/workflow-ai/workflow-chat-transport) \- Transport layer for AI SDK streams
- [Workflows and Steps](https://useworkflow.dev/docs/foundations/workflows-and-steps) \- Understanding workflow fundamentals
- [AI SDK Documentation](https://ai-sdk.dev/docs) \- AI SDK documentation reference

[@workflow/ai\\
\\
Previous Page](https://useworkflow.dev/docs/api-reference/workflow-ai) [WorkflowChatTransport\\
\\
Next Page](https://useworkflow.dev/docs/api-reference/workflow-ai/workflow-chat-transport)

On this page

[API Signature](https://useworkflow.dev/docs/api-reference/workflow-ai/durable-agent#api-signature) [Class](https://useworkflow.dev/docs/api-reference/workflow-ai/durable-agent#class) [DurableAgentOptions](https://useworkflow.dev/docs/api-reference/workflow-ai/durable-agent#durableagentoptions) [DurableAgentStreamOptions](https://useworkflow.dev/docs/api-reference/workflow-ai/durable-agent#durableagentstreamoptions) [Key Features](https://useworkflow.dev/docs/api-reference/workflow-ai/durable-agent#key-features) [Good to Know](https://useworkflow.dev/docs/api-reference/workflow-ai/durable-agent#good-to-know) [Examples](https://useworkflow.dev/docs/api-reference/workflow-ai/durable-agent#examples) [Basic Agent with Tools](https://useworkflow.dev/docs/api-reference/workflow-ai/durable-agent#basic-agent-with-tools) [Multiple Tools](https://useworkflow.dev/docs/api-reference/workflow-ai/durable-agent#multiple-tools) [Multi-turn Conversation](https://useworkflow.dev/docs/api-reference/workflow-ai/durable-agent#multi-turn-conversation) [Tools with Workflow Library Features](https://useworkflow.dev/docs/api-reference/workflow-ai/durable-agent#tools-with-workflow-library-features) [See Also](https://useworkflow.dev/docs/api-reference/workflow-ai/durable-agent#see-also)

[GitHubEdit this page on GitHub](https://github.com/vercel/workflow/edit/main/content/docs/api-reference/workflow-ai/durable-agent.mdx) Scroll to topGive feedbackCopy pageAsk AI about this pageOpen in chat

## Create Webhook
[API Reference](https://useworkflow.dev/docs/api-reference) [workflow](https://useworkflow.dev/docs/api-reference/workflow)

# createWebhook

Creates a webhook that can be used to suspend and resume a workflow run upon receiving an HTTP request.

Webhooks provide a way for external systems to send HTTP requests directly to your workflow. Unlike hooks which accept arbitrary payloads, webhooks work with standard HTTP `Request` objects and can return HTTP `Response` objects.

```
import { createWebhook } from "workflow"

export async function webhookWorkflow() {
  "use workflow";
  const webhook = createWebhook();
  console.log('Webhook URL:', webhook.url);

  const request = await webhook; // Suspends until HTTP request received
  console.log('Received request:', request.method, request.url);
}
```

## [API Signature](https://useworkflow.dev/docs/api-reference/workflow/create-webhook\#api-signature)

### [Parameters](https://useworkflow.dev/docs/api-reference/workflow/create-webhook\#parameters)

This function has multiple signatures.

#### Signature 1

| Name | Type | Description |
| --- | --- | --- |
| `options` | WebhookOptions & { respondWith: "manual"; } |  |

#### Signature 2

| Name | Type | Description |
| --- | --- | --- |
| `options` | WebhookOptions |  |

### [Returns](https://useworkflow.dev/docs/api-reference/workflow/create-webhook\#returns)

This function has multiple signatures.

#### Signature 1

`Webhook<RequestWithResponse>`

#### Signature 2

`Webhook<Request>`

The returned `Webhook` object has:

- `url`: The HTTP endpoint URL that external systems can call
- `token`: The unique token identifying this webhook
- Implements `AsyncIterable<RequestWithResponse>` for handling multiple requests

The `RequestWithResponse` type extends the standard `Request` interface with a `respondWith(response: Response)` method for sending custom responses back to the caller.

## [Examples](https://useworkflow.dev/docs/api-reference/workflow/create-webhook\#examples)

### [Basic Usage](https://useworkflow.dev/docs/api-reference/workflow/create-webhook\#basic-usage)

Create a webhook that receives HTTP requests and logs the request details:

```
import { createWebhook } from "workflow"

export async function basicWebhookWorkflow() {
  "use workflow";

  const webhook = createWebhook();
  console.log('Send requests to:', webhook.url);

  const request = await webhook;

  console.log('Method:', request.method);
  console.log('Headers:', Object.fromEntries(request.headers));

  const body = await request.text();
  console.log('Body:', body);
}
```

### [Responding to Webhook Requests](https://useworkflow.dev/docs/api-reference/workflow/create-webhook\#responding-to-webhook-requests)

Use the `respondWith()` method to send custom HTTP responses. Note that `respondWith()` must be called from within a step function:

```
import { createWebhook, type RequestWithResponse } from "workflow"

async function sendResponse(request: RequestWithResponse) {
  "use step";
  await request.respondWith(
    new Response(JSON.stringify({ success: true, message: 'Received!' }), {
      status: 200,
      headers: { 'Content-Type': 'application/json' }
    })
  );
}

export async function respondingWebhookWorkflow() {
  "use workflow";

  const webhook = createWebhook();
  console.log('Webhook URL:', webhook.url);

  const request = await webhook;

  // Send a custom response back to the caller
  await sendResponse(request);

  // Continue workflow processing
  const data = await request.json();
  await processData(data);
}

async function processData(data: any) {
  "use step";
  // Process the webhook data
  console.log('Processing:', data);
}
```

### [Customizing Tokens](https://useworkflow.dev/docs/api-reference/workflow/create-webhook\#customizing-tokens)

Tokens are used to identify a specific webhook. You can customize the token to be more specific to a use case.

```
import { type RequestWithResponse } from "workflow"

async function sendAck(request: RequestWithResponse) {
  "use step";
  await request.respondWith(
    new Response(JSON.stringify({ received: true }), {
      headers: { 'Content-Type': 'application/json' }
    })
  );
}

export async function githubWebhookWorkflow(repoName: string) {
  "use workflow";

  // Use a deterministic token based on the repository
  const webhook = createWebhook({
    token: `github_webhook:${repoName}`,
  });

  console.log('Configure GitHub webhook:', webhook.url);

  const request = await webhook;
  const event = await request.json();

  await sendAck(request);

  await deployCommit(event);
}

async function deployCommit(event: any) {
  "use step";
  // Deploy logic here
}
```

### [Waiting for Multiple Requests](https://useworkflow.dev/docs/api-reference/workflow/create-webhook\#waiting-for-multiple-requests)

You can also wait for multiple requests by using the `for await...of` syntax.

```
import { createWebhook, type RequestWithResponse } from "workflow"

async function sendSlackResponse(request: RequestWithResponse, message: string) {
  "use step";
  await request.respondWith(
    new Response(
      JSON.stringify({
        response_type: 'in_channel',
        text: message
      }),
      { headers: { 'Content-Type': 'application/json' } }
    )
  );
}

async function sendStopResponse(request: RequestWithResponse) {
  "use step";
  await request.respondWith(
    new Response('Stopping workflow...')
  );
}

export async function slackCommandWorkflow(channelId: string) {
  "use workflow";

  const webhook = createWebhook({
    token: `slack_command:${channelId}`,
  });

  for await (const request of webhook) {
    const formData = await request.formData();
    const command = formData.get('command');
    const text = formData.get('text');

    if (command === '/status') {
      // Respond immediately to Slack
      await sendSlackResponse(request, 'Checking status...');

      // Process the command
      const status = await checkSystemStatus();
      await postToSlack(channelId, `Status: ${status}`);
    }

    if (text === 'stop') {
      await sendStopResponse(request);
      break;
    }
  }
}

async function checkSystemStatus() {
  "use step";
  return "All systems operational";
}

async function postToSlack(channelId: string, message: string) {
  "use step";
  // Post message to Slack
}
```

## [Related Functions](https://useworkflow.dev/docs/api-reference/workflow/create-webhook\#related-functions)

- [`createHook()`](https://useworkflow.dev/docs/api-reference/workflow/create-hook) \- Lower-level hook primitive for arbitrary payloads
- [`defineHook()`](https://useworkflow.dev/docs/api-reference/workflow/define-hook) \- Type-safe hook helper
- [`resumeWebhook()`](https://useworkflow.dev/docs/api-reference/workflow-api/resume-webhook) \- Resume a webhook from an API route

[createHook\\
\\
Previous Page](https://useworkflow.dev/docs/api-reference/workflow/create-hook) [defineHook\\
\\
Next Page](https://useworkflow.dev/docs/api-reference/workflow/define-hook)

On this page

[API Signature](https://useworkflow.dev/docs/api-reference/workflow/create-webhook#api-signature) [Parameters](https://useworkflow.dev/docs/api-reference/workflow/create-webhook#parameters) [Returns](https://useworkflow.dev/docs/api-reference/workflow/create-webhook#returns) [Examples](https://useworkflow.dev/docs/api-reference/workflow/create-webhook#examples) [Basic Usage](https://useworkflow.dev/docs/api-reference/workflow/create-webhook#basic-usage) [Responding to Webhook Requests](https://useworkflow.dev/docs/api-reference/workflow/create-webhook#responding-to-webhook-requests) [Customizing Tokens](https://useworkflow.dev/docs/api-reference/workflow/create-webhook#customizing-tokens) [Waiting for Multiple Requests](https://useworkflow.dev/docs/api-reference/workflow/create-webhook#waiting-for-multiple-requests) [Related Functions](https://useworkflow.dev/docs/api-reference/workflow/create-webhook#related-functions)

[GitHubEdit this page on GitHub](https://github.com/vercel/workflow/edit/main/content/docs/api-reference/workflow/create-webhook.mdx) Scroll to topGive feedbackCopy pageAsk AI about this pageOpen in chat

## Workflow Code Configuration
[API Reference](https://useworkflow.dev/docs/api-reference) [workflow/next](https://useworkflow.dev/docs/api-reference/workflow-next)

# withWorkflow

Configures webpack/turbopack loaders to transform workflow code (`"use step"`/`"use workflow"` directives)

## [Usage](https://useworkflow.dev/docs/api-reference/workflow-next/with-workflow\#usage)

To enable `"use step"` and `"use workflow"` directives while developing locally or deploying to production, wrap your `nextConfig` with `withWorkflow`.

next.config.ts

```

```

If you are exporting a function in your `next.config` you will need to ensure you call the function returned from `withWorkflow`.

next.config.ts

```

```

[workflow/next\\
\\
Previous Page](https://useworkflow.dev/docs/api-reference/workflow-next) [@workflow/ai\\
\\
Next Page](https://useworkflow.dev/docs/api-reference/workflow-ai)

On this page

[Usage](https://useworkflow.dev/docs/api-reference/workflow-next/with-workflow#usage)

[GitHubEdit this page on GitHub](https://github.com/vercel/workflow/edit/main/content/docs/api-reference/workflow-next/with-workflow.mdx) Scroll to topGive feedbackCopy pageAsk AI about this pageOpen in chat

## Nuxt Workflow Guide
[Getting Started](https://useworkflow.dev/docs/getting-started)

# Nuxt

This guide will walk through setting up your first workflow in a Nuxt app. Along the way, you'll learn more about the concepts that are fundamental to using the development kit in your own projects.

## [Create Your Nuxt Project](https://useworkflow.dev/docs/getting-started/nuxt\#create-your-nuxt-project)

Start by creating a new Nuxt project. This command will create a new directory named `nuxt-app` and setup a Nuxt project inside it.

```
npm create nuxt@latest nuxt-app
```

Enter the newly made directory:

```
cd nuxt-app
```

### [Install `workflow`](https://useworkflow.dev/docs/getting-started/nuxt\#install-workflow)

npm

pnpm

yarn

bun

```
npm i workflow
```

### [Configure Nuxt](https://useworkflow.dev/docs/getting-started/nuxt\#configure-nuxt)

Add `workflow` to your `nuxt.config.ts`. This automatically configures the Nitro integration and enables usage of the `"use workflow"` and `"use step"` directives.

nuxt.config.ts

```

```

This will also automatically enable the TypeScript plugin, which provides helpful IntelliSense hints in your IDE for workflow and step functions.

### Disable TypeScript Plugin (Optional)

## [Create Your First Workflow](https://useworkflow.dev/docs/getting-started/nuxt\#create-your-first-workflow)

Create a new file for our first workflow:

server/workflows/user-signup.ts

```

```

We'll fill in those functions next, but let's take a look at this code:

- We define a **workflow** function with the directive `"use workflow"`. Think of the workflow function as the _orchestrator_ of individual **steps**.
- The Workflow DevKit's `sleep` function allows us to suspend execution of the workflow without using up any resources. A sleep can be a few seconds, hours, days, or even months long.

## [Create Your Workflow Steps](https://useworkflow.dev/docs/getting-started/nuxt\#create-your-workflow-steps)

Let's now define those missing functions.

server/workflows/user-signup.ts

```

```

Taking a look at this code:

- Business logic lives inside **steps**. When a step is invoked inside a **workflow**, it gets enqueued to run on a separate request while the workflow is suspended, just like `sleep`.
- If a step throws an error, like in `sendWelcomeEmail`, the step will automatically be retried until it succeeds (or hits the step's max retry count).
- Steps can throw a `FatalError` if an error is intentional and should not be retried.

We'll dive deeper into workflows, steps, and other ways to suspend or handle
events in [Foundations](https://useworkflow.dev/docs/foundations).

## [Create Your API Route](https://useworkflow.dev/docs/getting-started/nuxt\#create-your-api-route)

To invoke your new workflow, we'll create a new API route handler at `server/api/signup.post.ts` with the following code:

server/api/signup.post.ts

```

```

This API route creates a `POST` request endpoint at `/api/signup` that will trigger your workflow.

Workflows can be triggered from API routes or any server-side
code.

## [Run in development](https://useworkflow.dev/docs/getting-started/nuxt\#run-in-development)

To start your development server, run the following command in your terminal in the Nuxt root directory:

```
npm run dev
```

Once your development server is running, you can trigger your workflow by running this command in the terminal:

```
curl -X POST --json '{"email":"hello@example.com"}' http://localhost:3000/api/signup
```

Check the Nuxt development server logs to see your workflow execute as well as the steps that are being processed.

Additionally, you can use the [Workflow DevKit CLI or Web UI](https://useworkflow.dev/docs/observability) to inspect your workflow runs and steps in detail.

```
npx workflow inspect runs # add '--web' for an interactive Web based UI
```

![Workflow DevKit Web UI](https://useworkflow.dev/o11y-ui.png)

## [Deploying to production](https://useworkflow.dev/docs/getting-started/nuxt\#deploying-to-production)

Workflow DevKit apps currently work best when deployed to [Vercel](https://vercel.com/home) and needs no special configuration.

Check the [Deploying](https://useworkflow.dev/docs/deploying) section to learn how your workflows can be deployed elsewhere.

## [Next Steps](https://useworkflow.dev/docs/getting-started/nuxt\#next-steps)

- Learn more about the [Foundations](https://useworkflow.dev/docs/foundations).
- Check [Errors](https://useworkflow.dev/docs/errors) if you encounter issues.
- Explore the [API Reference](https://useworkflow.dev/docs/api-reference).

[Nitro\\
\\
This guide will walk through setting up your first workflow in a Nitro v3 project. Along the way, you'll learn more about the concepts that are fundamental to using the development kit in your own projects.](https://useworkflow.dev/docs/getting-started/nitro) [SvelteKit\\
\\
This guide will walk through setting up your first workflow in a SvelteKit app. Along the way, you'll learn more about the concepts that are fundamental to using the development kit in your own projects.](https://useworkflow.dev/docs/getting-started/sveltekit)

On this page

[Create Your Nuxt Project](https://useworkflow.dev/docs/getting-started/nuxt#create-your-nuxt-project) [Install `workflow`](https://useworkflow.dev/docs/getting-started/nuxt#install-workflow) [Configure Nuxt](https://useworkflow.dev/docs/getting-started/nuxt#configure-nuxt) [Create Your First Workflow](https://useworkflow.dev/docs/getting-started/nuxt#create-your-first-workflow) [Create Your Workflow Steps](https://useworkflow.dev/docs/getting-started/nuxt#create-your-workflow-steps) [Create Your API Route](https://useworkflow.dev/docs/getting-started/nuxt#create-your-api-route) [Run in development](https://useworkflow.dev/docs/getting-started/nuxt#run-in-development) [Deploying to production](https://useworkflow.dev/docs/getting-started/nuxt#deploying-to-production) [Next Steps](https://useworkflow.dev/docs/getting-started/nuxt#next-steps)

[GitHubEdit this page on GitHub](https://github.com/vercel/workflow/edit/main/content/docs/getting-started/nuxt.mdx) Scroll to topGive feedbackCopy pageAsk AI about this pageOpen in chat

## Chat

What is Workflow?How does retrying work?What control flow patterns are there?How do directives work?

Tip: You can open and close chat with `⌘I`

0 / 1000

## Type-Safe Hook Definition
[API Reference](https://useworkflow.dev/docs/api-reference) [workflow](https://useworkflow.dev/docs/api-reference/workflow)

# defineHook

Creates a type-safe hook helper that ensures the payload type is consistent between hook creation and resumption.

This is a lightweight wrapper around [`createHook()`](https://useworkflow.dev/docs/api-reference/workflow/create-hook) and [`resumeHook()`](https://useworkflow.dev/docs/api-reference/workflow-api/resume-hook) to avoid type mismatches. It also supports optional runtime validation and transformation of payloads using any [Standard Schema v1](https://standardschema.dev/) compliant validator like Zod or Valibot.

We recommend using `defineHook()` over `createHook()` in production codebases for better type safety and optional runtime validation.

```
import { defineHook } from "workflow";

const nameHook = defineHook<{
  name: string;
}>();

export async function nameWorkflow() {
  "use workflow";

  const hook = nameHook.create();
  const result = await hook; // Fully typed as { name: string }
  console.log('Name:', result.name);
}
```

## [API Signature](https://useworkflow.dev/docs/api-reference/workflow/define-hook\#api-signature)

### [Parameters](https://useworkflow.dev/docs/api-reference/workflow/define-hook\#parameters)

| Name | Type | Description |
| --- | --- | --- |
| `__0` | { schema?: StandardSchemaV1<TInput, TOutput>; } | Schema used to validate and transform the input payload before resuming |

### [Returns](https://useworkflow.dev/docs/api-reference/workflow/define-hook\#returns)

| Name | Type | Description |
| --- | --- | --- |
| `create` | (options?: any) => Hook<T> | Creates a new hook with the defined payload type. |
| `resume` | (token: string, payload: T) => Promise<any> | Resumes a hook by sending a payload with the defined type. |

## [Examples](https://useworkflow.dev/docs/api-reference/workflow/define-hook\#examples)

### [Basic Type-Safe Hook Definition](https://useworkflow.dev/docs/api-reference/workflow/define-hook\#basic-type-safe-hook-definition)

By defining the hook once with a specific payload type, you can reuse it in multiple workflows and API routes with automatic type safety.

```
import { defineHook } from "workflow";

// Define once with a specific payload type
const approvalHook = defineHook<{
  approved: boolean;
  comment: string;
}>();

// In your workflow
export async function workflowWithApproval() {
  "use workflow";

  const hook = approvalHook.create();
  const result = await hook; // Fully typed as { approved: boolean; comment: string }

  console.log('Approved:', result.approved);
  console.log('Comment:', result.comment);
}
```

### [Resuming with Type Safety](https://useworkflow.dev/docs/api-reference/workflow/define-hook\#resuming-with-type-safety)

Hooks can be resumed using the same defined hook and a token. By using the same hook, you can ensure that the payload matches the defined type when resuming a hook.

```
// Use the same defined hook to resume
export async function POST(request: Request) {
  const { token, approved, comment } = await request.json();

  // Type-safe resumption - TypeScript ensures the payload matches
  const result = await approvalHook.resume(token, {
    approved,
    comment,
  });

  if (!result) {
    return Response.json({ error: 'Hook not found' }, { status: 404 });
  }

  return Response.json({ success: true, runId: result.runId });
}
```

### [Validate and Transform with Schema](https://useworkflow.dev/docs/api-reference/workflow/define-hook\#validate-and-transform-with-schema)

You can provide runtime validation and transformation of hook payloads using the `schema` option. This option accepts any validator that conforms to the [Standard Schema v1](https://standardschema.dev/) specification.

Standard Schema is a standardized specification for schema validation libraries. Most popular validation libraries support it, including Zod, Valibot, ArkType, and Effect Schema. You can also write custom validators.

#### [Using Zod with defineHook](https://useworkflow.dev/docs/api-reference/workflow/define-hook\#using-zod-with-definehook)

Here's an example using [Zod](https://zod.dev/) to validate and transform hook payloads:

```
import { defineHook } from "workflow";
import { z } from "zod";

export const approvalHook = defineHook({
  schema: z.object({
    approved: z.boolean(),
    comment: z.string().min(1).transform((value) => value.trim()),
  }),
});

export async function approvalWorkflow(approvalId: string) {
  "use workflow";

  const hook = approvalHook.create({
    token: `approval:${approvalId}`,
  });

  // Payload is automatically typed based on the schema
  const { approved, comment } = await hook;
  console.log('Approved:', approved);
  console.log('Comment (trimmed):', comment);
}
```

When resuming the hook from an API route, the schema validates and transforms the incoming payload before the workflow resumes:

```
export async function POST(request: Request) {
  // Incoming payload: { token: "...", approved: true, comment: "   Ready!   " }
  const { token, approved, comment } = await request.json();

  // The schema validates and transforms the payload:
  // - Checks that `approved` is a boolean
  // - Checks that `comment` is a non-empty string
  // - Trims whitespace from the comment
  // If validation fails, an error is thrown and the hook is not resumed
  await approvalHook.resume(token, {
    approved,
    comment, // Automatically trimmed to "Ready!"
  });

  return Response.json({ success: true });
}
```

#### [Using Other Standard Schema Libraries](https://useworkflow.dev/docs/api-reference/workflow/define-hook\#using-other-standard-schema-libraries)

The same pattern works with any Standard Schema v1 compliant library. Here's an example with [Valibot](https://valibot.dev/):

```
import { defineHook } from "workflow";
import * as v from "valibot";

export const approvalHook = defineHook({
  schema: v.object({
    approved: v.boolean(),
    comment: v.pipe(v.string(), v.minLength(1), v.trim()),
  }),
});
```

### [Customizing Tokens](https://useworkflow.dev/docs/api-reference/workflow/define-hook\#customizing-tokens)

Tokens are used to identify a specific hook and for resuming a hook. You can customize the token to be more specific to a use case.

```
const slackHook = defineHook<{ text: string; userId: string }>();

export async function slackBotWorkflow(channelId: string) {
  "use workflow";

  const hook = slackHook.create({
    token: `slack:${channelId}`,
  });

  const message = await hook;
  console.log(`Message from ${message.userId}: ${message.text}`);
}
```

## [Related Functions](https://useworkflow.dev/docs/api-reference/workflow/define-hook\#related-functions)

- [`createHook()`](https://useworkflow.dev/docs/api-reference/workflow/create-hook) \- Create a hook in a workflow.
- [`resumeHook()`](https://useworkflow.dev/docs/api-reference/workflow-api/resume-hook) \- Resume a hook with a payload.

[createWebhook\\
\\
Previous Page](https://useworkflow.dev/docs/api-reference/workflow/create-webhook) [fetch\\
\\
Next Page](https://useworkflow.dev/docs/api-reference/workflow/fetch)

On this page

[API Signature](https://useworkflow.dev/docs/api-reference/workflow/define-hook#api-signature) [Parameters](https://useworkflow.dev/docs/api-reference/workflow/define-hook#parameters) [Returns](https://useworkflow.dev/docs/api-reference/workflow/define-hook#returns) [Examples](https://useworkflow.dev/docs/api-reference/workflow/define-hook#examples) [Basic Type-Safe Hook Definition](https://useworkflow.dev/docs/api-reference/workflow/define-hook#basic-type-safe-hook-definition) [Resuming with Type Safety](https://useworkflow.dev/docs/api-reference/workflow/define-hook#resuming-with-type-safety) [Validate and Transform with Schema](https://useworkflow.dev/docs/api-reference/workflow/define-hook#validate-and-transform-with-schema) [Using Zod with defineHook](https://useworkflow.dev/docs/api-reference/workflow/define-hook#using-zod-with-definehook) [Using Other Standard Schema Libraries](https://useworkflow.dev/docs/api-reference/workflow/define-hook#using-other-standard-schema-libraries) [Customizing Tokens](https://useworkflow.dev/docs/api-reference/workflow/define-hook#customizing-tokens) [Related Functions](https://useworkflow.dev/docs/api-reference/workflow/define-hook#related-functions)

[GitHubEdit this page on GitHub](https://github.com/vercel/workflow/edit/main/content/docs/api-reference/workflow/define-hook.mdx) Scroll to topGive feedbackCopy pageAsk AI about this pageOpen in chat

## Webhook Response Error
[Errors](https://useworkflow.dev/docs/errors)

# webhook-response-not-sent

This error occurs when a webhook is configured with `respondWith: "manual"` but the workflow does not send a response using `request.respondWith()` before the webhook execution completes.

## [Error Message](https://useworkflow.dev/docs/errors/webhook-response-not-sent\#error-message)

```
Workflow run did not send a response
```

## [Why This Happens](https://useworkflow.dev/docs/errors/webhook-response-not-sent\#why-this-happens)

When you create a webhook with `respondWith: "manual"`, you are responsible for calling `request.respondWith()` to send the HTTP response back to the caller. If the workflow execution completes without sending a response, this error will be thrown.

The webhook infrastructure waits for a response to be sent, and if none is provided, it cannot complete the HTTP request properly.

## [Common Causes](https://useworkflow.dev/docs/errors/webhook-response-not-sent\#common-causes)

### [Forgetting to Call `request.respondWith()`](https://useworkflow.dev/docs/errors/webhook-response-not-sent\#forgetting-to-call-requestrespondwith)

```
// Error - no response sent
export async function webhookWorkflow() {
  "use workflow";

  const webhook = await createWebhook({
    respondWith: "manual",
  });

  const request = await webhook;
  const data = await request.json();

  // Process data...
  console.log(data);

  // Error: workflow ends without calling request.respondWith()
}
```

**Solution:** Always call `request.respondWith()` when using manual response mode.

```
// Fixed - response sent
export async function webhookWorkflow() {
  "use workflow";

  const webhook = await createWebhook({
    respondWith: "manual",
  });

  const request = await webhook;
  const data = await request.json();

  // Process data...
  console.log(data);

  // Send response before workflow ends
  await request.respondWith(new Response("Processed", { status: 200 }));
}
```

### [Conditional Response Logic](https://useworkflow.dev/docs/errors/webhook-response-not-sent\#conditional-response-logic)

```
// Error - response only sent in some branches
export async function webhookWorkflow() {
  "use workflow";

  const webhook = await createWebhook({
    respondWith: "manual",
  });

  const request = await webhook;
  const data = await request.json();

  if (data.isValid) {
    await request.respondWith(new Response("OK", { status: 200 }));
  }
  // Error: no response when data.isValid is false
}
```

**Solution:** Ensure all code paths send a response.

```
// Fixed - response sent in all branches
export async function webhookWorkflow() {
  "use workflow";

  const webhook = await createWebhook({
    respondWith: "manual",
  });

  const request = await webhook;
  const data = await request.json();

  if (data.isValid) {
    await request.respondWith(new Response("OK", { status: 200 }));
  } else {
    await request.respondWith(new Response("Invalid data", { status: 400 }));
  }
}
```

### [Exception Before Response](https://useworkflow.dev/docs/errors/webhook-response-not-sent\#exception-before-response)

```
// Error - exception thrown before response
export async function webhookWorkflow() {
  "use workflow";

  const webhook = await createWebhook({
    respondWith: "manual",
  });

  const request = await webhook;

  // Error occurs here
  throw new Error("Something went wrong");

  // Never reached
  await request.respondWith(new Response("OK", { status: 200 }));
}
```

**Solution:** Use try-catch to handle errors and send appropriate responses.

```
// Fixed - error handling with response
export async function webhookWorkflow() {
  "use workflow";

  const webhook = await createWebhook({
    respondWith: "manual",
  });

  const request = await webhook;

  try {
    // Process request...
    const result = await processData(request);
    await request.respondWith(new Response("OK", { status: 200 }));
  } catch (error) {
    // Send error response
    await request.respondWith(
      new Response("Internal error", { status: 500 })
    );
  }
}
```

## [Alternative: Use Default Response Mode](https://useworkflow.dev/docs/errors/webhook-response-not-sent\#alternative-use-default-response-mode)

If you don't need custom response control, consider using the default response mode which automatically returns a `202 Accepted` response:

```
// Automatic 202 response - no manual response needed
export async function webhookWorkflow() {
  "use workflow";

  const webhook = await createWebhook();
  const request = await webhook;

  // Process request asynchronously
  await processData(request);

  // No need to call request.respondWith()
}
```

## [Learn More](https://useworkflow.dev/docs/errors/webhook-response-not-sent\#learn-more)

- [createWebhook() API Reference](https://useworkflow.dev/docs/api-reference/workflow/create-webhook)
- [resumeWebhook() API Reference](https://useworkflow.dev/docs/api-reference/workflow-api/resume-webhook)
- [Webhooks Guide](https://useworkflow.dev/docs/foundations/hooks)

[webhook-invalid-respond-with-value\\
\\
Previous Page](https://useworkflow.dev/docs/errors/webhook-invalid-respond-with-value) [API Reference\\
\\
Next Page](https://useworkflow.dev/docs/api-reference)

On this page

[Error Message](https://useworkflow.dev/docs/errors/webhook-response-not-sent#error-message) [Why This Happens](https://useworkflow.dev/docs/errors/webhook-response-not-sent#why-this-happens) [Common Causes](https://useworkflow.dev/docs/errors/webhook-response-not-sent#common-causes) [Forgetting to Call `request.respondWith()`](https://useworkflow.dev/docs/errors/webhook-response-not-sent#forgetting-to-call-requestrespondwith) [Conditional Response Logic](https://useworkflow.dev/docs/errors/webhook-response-not-sent#conditional-response-logic) [Exception Before Response](https://useworkflow.dev/docs/errors/webhook-response-not-sent#exception-before-response) [Alternative: Use Default Response Mode](https://useworkflow.dev/docs/errors/webhook-response-not-sent#alternative-use-default-response-mode) [Learn More](https://useworkflow.dev/docs/errors/webhook-response-not-sent#learn-more)

[GitHubEdit this page on GitHub](https://github.com/vercel/workflow/edit/main/content/docs/errors/webhook-response-not-sent.mdx) Scroll to topGive feedbackCopy pageAsk AI about this pageOpen in chat

## Fetch Error Handling
[Errors](https://useworkflow.dev/docs/errors)

# fetch-in-workflow

This error occurs when you try to use `fetch()` directly in a workflow function, or when a library (like the AI SDK) tries to call `fetch()` under the hood.

## [Error Message](https://useworkflow.dev/docs/errors/fetch-in-workflow\#error-message)

```
Global "fetch" is unavailable in workflow functions. Use the "fetch" step function from "workflow" to make HTTP requests.
```

## [Why This Happens](https://useworkflow.dev/docs/errors/fetch-in-workflow\#why-this-happens)

Workflow functions run in a sandboxed environment without direct access to `fetch()`.

Many libraries make HTTP requests under the hood. For example, the AI SDK's `generateText()` function calls `fetch()` to make HTTP requests to AI providers. When these libraries run inside a workflow function, they fail because the global `fetch` is not available.

## [Quick Fix](https://useworkflow.dev/docs/errors/fetch-in-workflow\#quick-fix)

Import the `fetch` step function from the `workflow` package and assign it to `globalThis.fetch` inside your workflow function. This version of `fetch` is a step function that wraps the standard `fetch` API, automatically handling serialization and providing retry capabilities. This will also make `fetch()` available to all functions and libraries in the current workflow function.

**Before:**

workflows/ai.ts

```

```

**After:**

workflows/ai.ts

```

```

## [Common Scenarios](https://useworkflow.dev/docs/errors/fetch-in-workflow\#common-scenarios)

### [AI SDK Integration](https://useworkflow.dev/docs/errors/fetch-in-workflow\#ai-sdk-integration)

This is the most common scenario - using AI SDK functions that make HTTP requests:

```
import { generateText, streamText } from 'ai';
import { openai } from '@ai-sdk/openai';
import { fetch } from 'workflow';

export async function aiWorkflow(userMessage: string) {
  "use workflow";

  globalThis.fetch = fetch;

  // generateText makes HTTP requests to OpenAI
  const response = await generateText({
    model: openai('gpt-4'),
    prompt: userMessage,
  });

  return response.text;
}
```

### [Direct API Calls](https://useworkflow.dev/docs/errors/fetch-in-workflow\#direct-api-calls)

You can also use the fetch step function directly for your own HTTP requests:

```
import { fetch } from 'workflow';

export async function dataWorkflow() {
  "use workflow";

  // Use fetch directly for HTTP requests
  const response = await fetch('https://api.example.com/data');
  const data = await response.json();

  return data;
}
```

For more details on the `fetch` step function, see the [fetch API reference](https://useworkflow.dev/docs/api-reference/workflow/fetch).

[Errors\\
\\
Previous Page](https://useworkflow.dev/docs/errors) [node-js-module-in-workflow\\
\\
Next Page](https://useworkflow.dev/docs/errors/node-js-module-in-workflow)

On this page

[Error Message](https://useworkflow.dev/docs/errors/fetch-in-workflow#error-message) [Why This Happens](https://useworkflow.dev/docs/errors/fetch-in-workflow#why-this-happens) [Quick Fix](https://useworkflow.dev/docs/errors/fetch-in-workflow#quick-fix) [Common Scenarios](https://useworkflow.dev/docs/errors/fetch-in-workflow#common-scenarios) [AI SDK Integration](https://useworkflow.dev/docs/errors/fetch-in-workflow#ai-sdk-integration) [Direct API Calls](https://useworkflow.dev/docs/errors/fetch-in-workflow#direct-api-calls)

[GitHubEdit this page on GitHub](https://github.com/vercel/workflow/edit/main/content/docs/errors/fetch-in-workflow.mdx) Scroll to topGive feedbackCopy pageAsk AI about this pageOpen in chat

## Express Workflow Setup
[Getting Started](https://useworkflow.dev/docs/getting-started)

# Express

This guide will walk through setting up your first workflow in a Express app. Along the way, you'll learn more about the concepts that are fundamental to using the development kit in your own projects.

* * *

## [Create Your Express Project](https://useworkflow.dev/docs/getting-started/express\#create-your-express-project)

Start by creating a new Express project.

```
mkdir my-workflow-app
```

Enter the newly made directory:

```
cd my-workflow-app
```

Initialize the project:

```
npm init --y
```

### [Install `workflow`, `express` and `nitro`](https://useworkflow.dev/docs/getting-started/express\#install-workflow-express-and-nitro)

npm

pnpm

yarn

bun

```
npm i workflow express nitro rollup
```

By default, Express doesn't include a build system. Nitro adds one which enables compiling workflows, runs, and deploys for development and production. Learn more about Nitro [here](https://v3.nitro.build/).

If using TypeScript, you need to install the `@types/express` package.

```
npm i -D @types/express
```

### [Configure Nitro](https://useworkflow.dev/docs/getting-started/express\#configure-nitro)

Create a new file `nitro.config.ts` for your Nitro configuration with module `workflow/nitro`. This enables usage of the `"use workflow"` and `"use step"` directives.

nitro.config.ts

```

```

### Setup IntelliSense for TypeScript (Optional)

### [Update `package.json`](https://useworkflow.dev/docs/getting-started/express\#update-packagejson)

To use the Nitro builder, update your `package.json` to include the following scripts:

package.json

```

```

## [Create Your First Workflow](https://useworkflow.dev/docs/getting-started/express\#create-your-first-workflow)

Create a new file for our first workflow:

workflows/user-signup.ts

```

```

We'll fill in those functions next, but let's take a look at this code:

- We define a **workflow** function with the directive `"use workflow"`. Think of the workflow function as the _orchestrator_ of individual **steps**.
- The Workflow DevKit's `sleep` function allows us to suspend execution of the workflow without using up any resources. A sleep can be a few seconds, hours, days, or even months long.

## [Create Your Workflow Steps](https://useworkflow.dev/docs/getting-started/express\#create-your-workflow-steps)

Let's now define those missing functions.

workflows/user-signup.ts

```

```

Taking a look at this code:

- Business logic lives inside **steps**. When a step is invoked inside a **workflow**, it gets enqueued to run on a separate request while the workflow is suspended, just like `sleep`.
- If a step throws an error, like in `sendWelcomeEmail`, the step will automatically be retried until it succeeds (or hits the step's max retry count).
- Steps can throw a `FatalError` if an error is intentional and should not be retried.

We'll dive deeper into workflows, steps, and other ways to suspend or handle
events in [Foundations](https://useworkflow.dev/docs/foundations).

## [Create Your Route Handler](https://useworkflow.dev/docs/getting-started/express\#create-your-route-handler)

To invoke your new workflow, we'll create both the Express app and a new API route handler at `src/index.ts` with the following code:

src/index.ts

```

```

This route handler creates a `POST` request endpoint at `/api/signup` that will trigger your workflow.

## [Run in development](https://useworkflow.dev/docs/getting-started/express\#run-in-development)

To start your development server, run the following command in your terminal in the Express root directory:

```
npm run dev
```

Once your development server is running, you can trigger your workflow by running this command in the terminal:

```
curl -X POST --json '{"email":"hello@example.com"}' http://localhost:3000/api/signup
```

Check the Express development server logs to see your workflow execute as well as the steps that are being processed.

Additionally, you can use the [Workflow DevKit CLI or Web UI](https://useworkflow.dev/docs/observability) to inspect your workflow runs and steps in detail.

```
npx workflow inspect runs # add '--web' for an interactive Web based UI
```

![Workflow DevKit Web UI](https://useworkflow.dev/o11y-ui.png)

* * *

## [Deploying to production](https://useworkflow.dev/docs/getting-started/express\#deploying-to-production)

Workflow DevKit apps currently work best when deployed to [Vercel](https://vercel.com/home) and needs no special configuration.

Check the [Deploying](https://useworkflow.dev/docs/deploying) section to learn how your workflows can be deployed elsewhere.

## [Next Steps](https://useworkflow.dev/docs/getting-started/express\#next-steps)

- Learn more about the [Foundations](https://useworkflow.dev/docs/foundations).
- Check [Errors](https://useworkflow.dev/docs/errors) if you encounter issues.
- Explore the [API Reference](https://useworkflow.dev/docs/api-reference).

[Vite\\
\\
Previous Page](https://useworkflow.dev/docs/getting-started/vite) [Hono\\
\\
This guide will walk through setting up your first workflow in a Hono app. Along the way, you'll learn more about the concepts that are fundamental to using the development kit in your own projects.](https://useworkflow.dev/docs/getting-started/hono)

On this page

[Create Your Express Project](https://useworkflow.dev/docs/getting-started/express#create-your-express-project) [Install `workflow`, `express` and `nitro`](https://useworkflow.dev/docs/getting-started/express#install-workflow-express-and-nitro) [Configure Nitro](https://useworkflow.dev/docs/getting-started/express#configure-nitro) [Update `package.json`](https://useworkflow.dev/docs/getting-started/express#update-packagejson) [Create Your First Workflow](https://useworkflow.dev/docs/getting-started/express#create-your-first-workflow) [Create Your Workflow Steps](https://useworkflow.dev/docs/getting-started/express#create-your-workflow-steps) [Create Your Route Handler](https://useworkflow.dev/docs/getting-started/express#create-your-route-handler) [Run in development](https://useworkflow.dev/docs/getting-started/express#run-in-development) [Deploying to production](https://useworkflow.dev/docs/getting-started/express#deploying-to-production) [Next Steps](https://useworkflow.dev/docs/getting-started/express#next-steps)

[GitHubEdit this page on GitHub](https://github.com/vercel/workflow/edit/main/content/docs/getting-started/express.mdx) Scroll to topGive feedbackCopy pageAsk AI about this pageOpen in chat

## Chat

What is Workflow?How does retrying work?What control flow patterns are there?How do directives work?

Tip: You can open and close chat with `⌘I`

0 / 1000

## AI Workflow Integration
[API Reference](https://useworkflow.dev/docs/api-reference)

# @workflow/ai

The `@workflow/ai` package is currently in active development and should be considered experimental.

Helpers for integrating AI SDK for building AI-powered workflows.

## [Classes](https://useworkflow.dev/docs/api-reference/workflow-ai\#classes)

[**DurableAgent** \\
\\
A class for building durable AI agents that maintain state across workflow steps and handle tool execution with automatic retries.](https://useworkflow.dev/docs/api-reference/workflow-ai/durable-agent) [**WorkflowChatTransport** \\
\\
A drop-in transport for the AI SDK for automatic reconnection in interrupted streams.](https://useworkflow.dev/docs/api-reference/workflow-ai/workflow-chat-transport)

[withWorkflow\\
\\
Previous Page](https://useworkflow.dev/docs/api-reference/workflow-next/with-workflow) [DurableAgent\\
\\
Next Page](https://useworkflow.dev/docs/api-reference/workflow-ai/durable-agent)

On this page

[Classes](https://useworkflow.dev/docs/api-reference/workflow-ai#classes)

[GitHubEdit this page on GitHub](https://github.com/vercel/workflow/edit/main/content/docs/api-reference/workflow-ai/index.mdx) Scroll to topGive feedbackCopy pageAsk AI about this pageOpen in chat

## Resume Workflow Hook
[API Reference](https://useworkflow.dev/docs/api-reference) [workflow/api](https://useworkflow.dev/docs/api-reference/workflow-api)

# resumeHook

Resumes a workflow run by sending a payload to a hook identified by its token.

It creates a `hook_received` event and re-triggers the workflow to continue execution.

`resumeHook` is a runtime function that must be called from outside a workflow function.

```
import { resumeHook } from "workflow/api";

export async function POST(request: Request) {
  const { token, data } = await request.json();

  try {
    const result = await resumeHook(token, data);
    return Response.json({
      runId: result.runId
    });
  } catch (error) {
    return new Response('Hook not found', { status: 404 });
  }
}
```

## [API Signature](https://useworkflow.dev/docs/api-reference/workflow-api/resume-hook\#api-signature)

### [Parameters](https://useworkflow.dev/docs/api-reference/workflow-api/resume-hook\#parameters)

| Name | Type | Description |
| --- | --- | --- |
| `token` | string | The unique token identifying the hook |
| `payload` | NonNullable<T> | The data payload to send to the hook |

### [Returns](https://useworkflow.dev/docs/api-reference/workflow-api/resume-hook\#returns)

Returns a `Promise<Hook>` that resolves to:

| Name | Type | Description |
| --- | --- | --- |
| `runId` | string |  |
| `hookId` | string |  |
| `token` | string |  |
| `ownerId` | string |  |
| `projectId` | string |  |
| `environment` | string |  |
| `createdAt` | Date |  |
| `metadata` | unknown |  |

## [Examples](https://useworkflow.dev/docs/api-reference/workflow-api/resume-hook\#examples)

### [Basic API Route](https://useworkflow.dev/docs/api-reference/workflow-api/resume-hook\#basic-api-route)

Using `resumeHook` in a basic API route to resume a hook:

```
import { resumeHook } from "workflow/api";

export async function POST(request: Request) {
  const { token, data } = await request.json();

  try {
    const result = await resumeHook(token, data);

    return Response.json({
      success: true,
      runId: result.runId
    });
  } catch (error) {
    return new Response('Hook not found', { status: 404 });
  }
}
```

### [With Type Safety](https://useworkflow.dev/docs/api-reference/workflow-api/resume-hook\#with-type-safety)

Defining a payload type and using `resumeHook` to resume a hook with type safety:

```
import { resumeHook } from "workflow/api";

type ApprovalPayload = {
  approved: boolean;
  comment: string;
};

export async function POST(request: Request) {
  const { token, approved, comment } = await request.json();

  try {
    const result = await resumeHook<ApprovalPayload>(token, {
      approved,
      comment,
    });

    return Response.json({ runId: result.runId });
  } catch (error) {
    return Response.json({ error: 'Invalid token' }, { status: 404 });
  }
}
```

### [Server Action (Next.js)](https://useworkflow.dev/docs/api-reference/workflow-api/resume-hook\#server-action-nextjs)

Using `resumeHook` in Next.js server actions to resume a hook:

```
'use server';

import { resumeHook } from "workflow/api";

export async function approveRequest(token: string, approved: boolean) {
  try {
    const result = await resumeHook(token, { approved });
    return result.runId;
  } catch (error) {
    throw new Error('Invalid approval token');
  }
}
```

### [Webhook Handler](https://useworkflow.dev/docs/api-reference/workflow-api/resume-hook\#webhook-handler)

Using `resumeHook` in a generic webhook handler to resume a hook:

```
import { resumeHook } from "workflow/api";

// Generic webhook handler that forwards data to a hook
export async function POST(request: Request) {
  const url = new URL(request.url);
  const token = url.searchParams.get('token');

  if (!token) {
    return Response.json({ error: 'Missing token' }, { status: 400 });
  }

  try {
    const body = await request.json();
    const result = await resumeHook(token, body);

    return Response.json({ success: true, runId: result.runId });
  } catch (error) {
    return Response.json({ error: 'Hook not found' }, { status: 404 });
  }
}
```

## [Related Functions](https://useworkflow.dev/docs/api-reference/workflow-api/resume-hook\#related-functions)

- [`createHook()`](https://useworkflow.dev/docs/api-reference/workflow/create-hook) \- Create a hook in a workflow.
- [`defineHook()`](https://useworkflow.dev/docs/api-reference/workflow/define-hook) \- Type-safe hook helper.

[getRun\\
\\
Previous Page](https://useworkflow.dev/docs/api-reference/workflow-api/get-run) [resumeWebhook\\
\\
Next Page](https://useworkflow.dev/docs/api-reference/workflow-api/resume-webhook)

On this page

[API Signature](https://useworkflow.dev/docs/api-reference/workflow-api/resume-hook#api-signature) [Parameters](https://useworkflow.dev/docs/api-reference/workflow-api/resume-hook#parameters) [Returns](https://useworkflow.dev/docs/api-reference/workflow-api/resume-hook#returns) [Examples](https://useworkflow.dev/docs/api-reference/workflow-api/resume-hook#examples) [Basic API Route](https://useworkflow.dev/docs/api-reference/workflow-api/resume-hook#basic-api-route) [With Type Safety](https://useworkflow.dev/docs/api-reference/workflow-api/resume-hook#with-type-safety) [Server Action (Next.js)](https://useworkflow.dev/docs/api-reference/workflow-api/resume-hook#server-action-nextjs) [Webhook Handler](https://useworkflow.dev/docs/api-reference/workflow-api/resume-hook#webhook-handler) [Related Functions](https://useworkflow.dev/docs/api-reference/workflow-api/resume-hook#related-functions)

[GitHubEdit this page on GitHub](https://github.com/vercel/workflow/edit/main/content/docs/api-reference/workflow-api/resume-hook.mdx) Scroll to topGive feedbackCopy pageAsk AI about this pageOpen in chat

## Workflow Observability Tools
# Observability

Workflow DevKit provides powerful tools to inspect, monitor, and debug your workflows through the CLI and Web UI. These tools allow you to inspect workflow runs, steps, webhooks, events, and stream output.

## [Quick Start](https://useworkflow.dev/docs/observability\#quick-start)

```
npx workflow
```

The CLI comes pre-installed with the Workflow DevKit and registers the `workflow` command. If the `workflow` package is not already installed, `npx workflow` will install it globally, or use the local installed version if available.

Get started inspecting your local workflows:

```
# See all available commands
npx workflow inspect --help

# List recent workflow runs
npx workflow inspect runs
```

## [Web UI](https://useworkflow.dev/docs/observability\#web-ui)

Workflow DevKit ships with a local web UI for inspecting your workflows. The CLI
will locally serve the Web UI when using the `--web` flag.

```
# Launch Web UI for visual exploration
npx workflow inspect runs --web
```

![Workflow DevKit Web UI](https://useworkflow.dev/o11y-ui.png)

## [Backends](https://useworkflow.dev/docs/observability\#backends)

The Workflow DevKit CLI can inspect data from any [World](https://useworkflow.dev/docs/deploying#what-are-worlds). By default, it inspects data in your local development environment. For example, if you are using NextJS to develop workflows locally, the
CLI will find the data in your `.next/workflow-data/` directory.

If you're deploying workflows to a production environment, but want to inspect the data by using the CLI, you can specify the world you are using by setting the `--backend` flag to your world's name or package name, e.g. `vercel`.

Backends might require additional configuration. If you're missing environment variables, the World package should provide instructions on how to configure it.

### [Vercel Backend](https://useworkflow.dev/docs/observability\#vercel-backend)

To inspect workflows running on Vercel, ensure you're logged in to the Vercel CLI and have linked your project. See [Vercel CLI authentication and project linking docs](https://vercel.com/docs/cli/project-linking) for more information. Then, simply specify the backend as `vercel`.

```
# Inspect workflows running on Vercel
npx workflow inspect runs --backend vercel
```

[Framework Integrations\\
\\
Previous Page](https://useworkflow.dev/docs/how-it-works/framework-integrations) [Deploying\\
\\
Next Page](https://useworkflow.dev/docs/deploying)

On this page

[Quick Start](https://useworkflow.dev/docs/observability#quick-start) [Web UI](https://useworkflow.dev/docs/observability#web-ui) [Backends](https://useworkflow.dev/docs/observability#backends) [Vercel Backend](https://useworkflow.dev/docs/observability#vercel-backend)

[GitHubEdit this page on GitHub](https://github.com/vercel/workflow/edit/main/content/docs/observability/index.mdx) Scroll to topGive feedbackCopy pageAsk AI about this pageOpen in chat

## Workflow API Documentation
https://useworkflow.dev/2025-11-25T17:34:29.795Zhttps://useworkflow.dev/docs/api-reference2025-11-25T17:34:29.795Zhttps://useworkflow.dev/docs/api-reference/workflow2025-11-25T17:34:29.795Zhttps://useworkflow.dev/docs/api-reference/workflow-ai2025-11-25T17:34:29.795Zhttps://useworkflow.dev/docs/api-reference/workflow-ai/durable-agent2025-11-27T04:06:53.558Zhttps://useworkflow.dev/docs/api-reference/workflow-ai/workflow-chat-transport2025-11-25T17:34:29.795Zhttps://useworkflow.dev/docs/api-reference/workflow-api2025-11-25T17:34:29.795Zhttps://useworkflow.dev/docs/api-reference/workflow-api/get-run2025-11-25T17:34:29.795Zhttps://useworkflow.dev/docs/api-reference/workflow-api/resume-hook2025-11-25T17:34:29.795Zhttps://useworkflow.dev/docs/api-reference/workflow-api/resume-webhook2025-11-25T17:34:29.795Zhttps://useworkflow.dev/docs/api-reference/workflow-api/start2025-11-25T17:34:29.795Zhttps://useworkflow.dev/docs/api-reference/workflow-next2025-11-25T17:34:29.795Zhttps://useworkflow.dev/docs/api-reference/workflow-next/with-workflow2025-11-25T17:34:29.795Zhttps://useworkflow.dev/docs/api-reference/workflow/create-hook2025-11-25T17:34:29.795Zhttps://useworkflow.dev/docs/api-reference/workflow/create-webhook2025-11-25T17:34:29.795Zhttps://useworkflow.dev/docs/api-reference/workflow/define-hook2025-11-25T17:34:29.795Zhttps://useworkflow.dev/docs/api-reference/workflow/fatal-error2025-11-25T17:34:29.795Zhttps://useworkflow.dev/docs/api-reference/workflow/fetch2025-11-25T17:34:29.795Zhttps://useworkflow.dev/docs/api-reference/workflow/get-step-metadata2025-11-25T17:34:29.795Zhttps://useworkflow.dev/docs/api-reference/workflow/get-workflow-metadata2025-11-25T17:34:29.795Zhttps://useworkflow.dev/docs/api-reference/workflow/get-writable2025-11-25T17:34:29.795Zhttps://useworkflow.dev/docs/api-reference/workflow/retryable-error2025-11-25T17:34:29.795Zhttps://useworkflow.dev/docs/api-reference/workflow/sleep2025-11-25T17:34:29.795Zhttps://useworkflow.dev/docs/deploying2025-11-25T17:34:29.795Zhttps://useworkflow.dev/docs/deploying/world2025-11-25T17:34:29.795Zhttps://useworkflow.dev/docs/deploying/world/local-world2025-11-25T17:34:29.795Zhttps://useworkflow.dev/docs/deploying/world/postgres-world2025-11-26T19:04:40.580Zhttps://useworkflow.dev/docs/deploying/world/vercel-world2025-11-25T17:34:29.795Zhttps://useworkflow.dev/docs/errors2025-11-25T17:34:29.795Zhttps://useworkflow.dev/docs/errors/fetch-in-workflow2025-11-25T17:34:29.795Zhttps://useworkflow.dev/docs/errors/node-js-module-in-workflow2025-11-25T17:34:29.795Zhttps://useworkflow.dev/docs/errors/serialization-failed2025-11-25T17:34:29.795Zhttps://useworkflow.dev/docs/errors/start-invalid-workflow-function2025-11-25T17:34:29.795Zhttps://useworkflow.dev/docs/errors/webhook-invalid-respond-with-value2025-11-25T17:34:29.795Zhttps://useworkflow.dev/docs/errors/webhook-response-not-sent2025-11-25T17:34:29.795Zhttps://useworkflow.dev/docs/foundations2025-11-25T17:34:29.795Zhttps://useworkflow.dev/docs/foundations/control-flow-patterns2025-11-25T17:34:29.795Zhttps://useworkflow.dev/docs/foundations/errors-and-retries2025-11-25T17:34:29.795Zhttps://useworkflow.dev/docs/foundations/hooks2025-11-25T17:34:29.795Zhttps://useworkflow.dev/docs/foundations/idempotency2025-11-25T17:34:29.795Zhttps://useworkflow.dev/docs/foundations/serialization2025-11-25T17:34:29.795Zhttps://useworkflow.dev/docs/foundations/starting-workflows2025-11-25T17:34:29.795Zhttps://useworkflow.dev/docs/foundations/streaming2025-11-25T17:34:29.795Zhttps://useworkflow.dev/docs/foundations/workflows-and-steps2025-11-25T17:34:29.795Zhttps://useworkflow.dev/docs/getting-started2025-11-25T17:34:29.795Zhttps://useworkflow.dev/docs/getting-started/express2025-11-25T17:34:29.795Zhttps://useworkflow.dev/docs/getting-started/hono2025-11-25T17:34:29.795Zhttps://useworkflow.dev/docs/getting-started/next2025-11-25T17:34:29.795Zhttps://useworkflow.dev/docs/getting-started/nitro2025-11-25T17:34:29.795Zhttps://useworkflow.dev/docs/getting-started/nuxt2025-11-25T17:34:29.795Zhttps://useworkflow.dev/docs/getting-started/sveltekit2025-11-25T17:34:29.795Zhttps://useworkflow.dev/docs/getting-started/vite2025-11-25T17:34:29.795Zhttps://useworkflow.dev/docs/how-it-works/code-transform2025-11-25T17:34:29.795Zhttps://useworkflow.dev/docs/how-it-works/framework-integrations2025-11-25T17:34:29.795Zhttps://useworkflow.dev/docs/how-it-works/understanding-directives2025-11-25T17:34:29.795Zhttps://useworkflow.dev/docs/observability2025-11-25T17:34:29.795Z

<urlsetxmlns="http://www.sitemaps.org/schemas/sitemap/0.9">

<url>

<loc>https://useworkflow.dev/</loc>

<lastmod>2025-11-25T17:34:29.795Z</lastmod>

...

</url>

<url>

<loc>https://useworkflow.dev/docs/api-reference</loc>

<lastmod>2025-11-25T17:34:29.795Z</lastmod>

...

</url>

<url>

<loc>https://useworkflow.dev/docs/api-reference/workflow</loc>

<lastmod>2025-11-25T17:34:29.795Z</lastmod>

...

</url>

<url>

<loc>https://useworkflow.dev/docs/api-reference/workflow-ai</loc>

<lastmod>2025-11-25T17:34:29.795Z</lastmod>

...

</url>

<url>

<loc>https://useworkflow.dev/docs/api-reference/workflow-ai/durable-agent</loc>

<lastmod>2025-11-27T04:06:53.558Z</lastmod>

...

</url>

<url>

<loc>https://useworkflow.dev/docs/api-reference/workflow-ai/workflow-chat-transport</loc>

<lastmod>2025-11-25T17:34:29.795Z</lastmod>

...

</url>

<url>

<loc>https://useworkflow.dev/docs/api-reference/workflow-api</loc>

<lastmod>2025-11-25T17:34:29.795Z</lastmod>

...

</url>

<url>

<loc>https://useworkflow.dev/docs/api-reference/workflow-api/get-run</loc>

<lastmod>2025-11-25T17:34:29.795Z</lastmod>

...

</url>

<url>

<loc>https://useworkflow.dev/docs/api-reference/workflow-api/resume-hook</loc>

<lastmod>2025-11-25T17:34:29.795Z</lastmod>

...

</url>

<url>

<loc>https://useworkflow.dev/docs/api-reference/workflow-api/resume-webhook</loc>

<lastmod>2025-11-25T17:34:29.795Z</lastmod>

...

</url>

<url>

<loc>https://useworkflow.dev/docs/api-reference/workflow-api/start</loc>

<lastmod>2025-11-25T17:34:29.795Z</lastmod>

...

</url>

<url>

<loc>https://useworkflow.dev/docs/api-reference/workflow-next</loc>

<lastmod>2025-11-25T17:34:29.795Z</lastmod>

...

</url>

<url>

<loc>https://useworkflow.dev/docs/api-reference/workflow-next/with-workflow</loc>

<lastmod>2025-11-25T17:34:29.795Z</lastmod>

...

</url>

<url>

<loc>https://useworkflow.dev/docs/api-reference/workflow/create-hook</loc>

<lastmod>2025-11-25T17:34:29.795Z</lastmod>

...

</url>

<url>

<loc>https://useworkflow.dev/docs/api-reference/workflow/create-webhook</loc>

<lastmod>2025-11-25T17:34:29.795Z</lastmod>

...

</url>

<url>

<loc>https://useworkflow.dev/docs/api-reference/workflow/define-hook</loc>

<lastmod>2025-11-25T17:34:29.795Z</lastmod>

...

</url>

<url>

<loc>https://useworkflow.dev/docs/api-reference/workflow/fatal-error</loc>

<lastmod>2025-11-25T17:34:29.795Z</lastmod>

...

</url>

<url>

<loc>https://useworkflow.dev/docs/api-reference/workflow/fetch</loc>

<lastmod>2025-11-25T17:34:29.795Z</lastmod>

...

</url>

<url>

<loc>https://useworkflow.dev/docs/api-reference/workflow/get-step-metadata</loc>

<lastmod>2025-11-25T17:34:29.795Z</lastmod>

...

</url>

<url>

<loc>https://useworkflow.dev/docs/api-reference/workflow/get-workflow-metadata</loc>

<lastmod>2025-11-25T17:34:29.795Z</lastmod>

...

</url>

<url>

<loc>https://useworkflow.dev/docs/api-reference/workflow/get-writable</loc>

<lastmod>2025-11-25T17:34:29.795Z</lastmod>

...

</url>

<url>

<loc>https://useworkflow.dev/docs/api-reference/workflow/retryable-error</loc>

<lastmod>2025-11-25T17:34:29.795Z</lastmod>

...

</url>

<url>

<loc>https://useworkflow.dev/docs/api-reference/workflow/sleep</loc>

<lastmod>2025-11-25T17:34:29.795Z</lastmod>

...

</url>

<url>

<loc>https://useworkflow.dev/docs/deploying</loc>

<lastmod>2025-11-25T17:34:29.795Z</lastmod>

...

</url>

<url>

<loc>https://useworkflow.dev/docs/deploying/world</loc>

<lastmod>2025-11-25T17:34:29.795Z</lastmod>

...

</url>

<url>

<loc>https://useworkflow.dev/docs/deploying/world/local-world</loc>

<lastmod>2025-11-25T17:34:29.795Z</lastmod>

...

</url>

<url>

<loc>https://useworkflow.dev/docs/deploying/world/postgres-world</loc>

<lastmod>2025-11-26T19:04:40.580Z</lastmod>

...

</url>

<url>

<loc>https://useworkflow.dev/docs/deploying/world/vercel-world</loc>

<lastmod>2025-11-25T17:34:29.795Z</lastmod>

...

</url>

<url>

<loc>https://useworkflow.dev/docs/errors</loc>

<lastmod>2025-11-25T17:34:29.795Z</lastmod>

...

</url>

<url>

<loc>https://useworkflow.dev/docs/errors/fetch-in-workflow</loc>

<lastmod>2025-11-25T17:34:29.795Z</lastmod>

...

</url>

<url>

<loc>https://useworkflow.dev/docs/errors/node-js-module-in-workflow</loc>

<lastmod>2025-11-25T17:34:29.795Z</lastmod>

...

</url>

<url>

<loc>https://useworkflow.dev/docs/errors/serialization-failed</loc>

<lastmod>2025-11-25T17:34:29.795Z</lastmod>

...

</url>

<url>

<loc>https://useworkflow.dev/docs/errors/start-invalid-workflow-function</loc>

<lastmod>2025-11-25T17:34:29.795Z</lastmod>

...

</url>

<url>

<loc>https://useworkflow.dev/docs/errors/webhook-invalid-respond-with-value</loc>

<lastmod>2025-11-25T17:34:29.795Z</lastmod>

...

</url>

<url>

<loc>https://useworkflow.dev/docs/errors/webhook-response-not-sent</loc>

<lastmod>2025-11-25T17:34:29.795Z</lastmod>

...

</url>

<url>

<loc>https://useworkflow.dev/docs/foundations</loc>

<lastmod>2025-11-25T17:34:29.795Z</lastmod>

...

</url>

<url>

<loc>https://useworkflow.dev/docs/foundations/control-flow-patterns</loc>

<lastmod>2025-11-25T17:34:29.795Z</lastmod>

...

</url>

<url>

<loc>https://useworkflow.dev/docs/foundations/errors-and-retries</loc>

<lastmod>2025-11-25T17:34:29.795Z</lastmod>

...

</url>

<url>

<loc>https://useworkflow.dev/docs/foundations/hooks</loc>

<lastmod>2025-11-25T17:34:29.795Z</lastmod>

...

</url>

<url>

<loc>https://useworkflow.dev/docs/foundations/idempotency</loc>

<lastmod>2025-11-25T17:34:29.795Z</lastmod>

...

</url>

<url>

<loc>https://useworkflow.dev/docs/foundations/serialization</loc>

<lastmod>2025-11-25T17:34:29.795Z</lastmod>

...

</url>

<url>

<loc>https://useworkflow.dev/docs/foundations/starting-workflows</loc>

<lastmod>2025-11-25T17:34:29.795Z</lastmod>

...

</url>

<url>

<loc>https://useworkflow.dev/docs/foundations/streaming</loc>

<lastmod>2025-11-25T17:34:29.795Z</lastmod>

...

</url>

<url>

<loc>https://useworkflow.dev/docs/foundations/workflows-and-steps</loc>

<lastmod>2025-11-25T17:34:29.795Z</lastmod>

...

</url>

<url>

<loc>https://useworkflow.dev/docs/getting-started</loc>

<lastmod>2025-11-25T17:34:29.795Z</lastmod>

...

</url>

<url>

<loc>https://useworkflow.dev/docs/getting-started/express</loc>

<lastmod>2025-11-25T17:34:29.795Z</lastmod>

...

</url>

<url>

<loc>https://useworkflow.dev/docs/getting-started/hono</loc>

<lastmod>2025-11-25T17:34:29.795Z</lastmod>

...

</url>

<url>

<loc>https://useworkflow.dev/docs/getting-started/next</loc>

<lastmod>2025-11-25T17:34:29.795Z</lastmod>

...

</url>

<url>

<loc>https://useworkflow.dev/docs/getting-started/nitro</loc>

<lastmod>2025-11-25T17:34:29.795Z</lastmod>

...

</url>

<url>

<loc>https://useworkflow.dev/docs/getting-started/nuxt</loc>

<lastmod>2025-11-25T17:34:29.795Z</lastmod>

...

</url>

<url>

<loc>https://useworkflow.dev/docs/getting-started/sveltekit</loc>

<lastmod>2025-11-25T17:34:29.795Z</lastmod>

...

</url>

<url>

<loc>https://useworkflow.dev/docs/getting-started/vite</loc>

<lastmod>2025-11-25T17:34:29.795Z</lastmod>

...

</url>

<url>

<loc>https://useworkflow.dev/docs/how-it-works/code-transform</loc>

<lastmod>2025-11-25T17:34:29.795Z</lastmod>

...

</url>

<url>

<loc>https://useworkflow.dev/docs/how-it-works/framework-integrations</loc>

<lastmod>2025-11-25T17:34:29.795Z</lastmod>

...

</url>

<url>

<loc>https://useworkflow.dev/docs/how-it-works/understanding-directives</loc>

<lastmod>2025-11-25T17:34:29.795Z</lastmod>

...

</url>

<url>

<loc>https://useworkflow.dev/docs/observability</loc>

<lastmod>2025-11-25T17:34:29.795Z</lastmod>

...

</url>

...

</urlset>

## Starting Workflows
[Foundations](https://useworkflow.dev/docs/foundations)

# Starting Workflows

Once you've defined your workflow functions, you need to trigger them to begin execution. This is done using the `start()` function from `workflow/api`, which enqueues a new workflow run and returns a `Run` object that you can use to track its progress.

## [The `start()` Function](https://useworkflow.dev/docs/foundations/starting-workflows\#the-start-function)

The [`start()`](https://useworkflow.dev/docs/api-reference/workflow-api/start) function is used to programmatically trigger workflow executions from runtime contexts like API routes, Server Actions, or any server-side code.

```
import { start } from 'workflow/api';
import { handleUserSignup } from './workflows/user-signup';

export async function POST(request: Request) {
  const { email } = await request.json();

  // Start the workflow
  const run = await start(handleUserSignup, [email]);

  return Response.json({
    message: 'Workflow started',
    runId: run.runId
  });
}
```

**Key Points:**

- `start()` returns immediately after enqueuing the workflow - it doesn't wait for completion
- The first argument is your workflow function
- The second argument is an array of arguments to pass to the workflow (optional if the workflow takes no arguments)
- All arguments must be [serializable](https://useworkflow.dev/docs/foundations/serialization)

**Learn more**: [`start()` API Reference](https://useworkflow.dev/docs/api-reference/workflow-api/start)

## [The `Run` Object](https://useworkflow.dev/docs/foundations/starting-workflows\#the-run-object)

When you call `start()`, it returns a [`Run`](https://useworkflow.dev/docs/api-reference/workflow-api/start#returns) object that provides access to the workflow's status and results.

```
import { start } from 'workflow/api';
import { processOrder } from './workflows/process-order';

const run = await start(processOrder, [orderId]);

// The run object has properties you can await
console.log('Run ID:', run.runId);

// Check the workflow status
const status = await run.status; // 'running' | 'completed' | 'failed'

// Get the workflow's return value (blocks until completion)
const result = await run.returnValue;
```

**Key Properties:**

- `runId` \- Unique identifier for this workflow run
- `status` \- Current status of the workflow (async)
- `returnValue` \- The value returned by the workflow function (async, blocks until completion)
- `readable` \- ReadableStream for streaming updates from the workflow

Most `Run` properties are async getters that return promises. You need to `await` them to get their values. For a complete list of properties and methods, see the API reference below.

**Learn more**: [`Run` API Reference](https://useworkflow.dev/docs/api-reference/workflow-api/start#returns)

## [Common Patterns](https://useworkflow.dev/docs/foundations/starting-workflows\#common-patterns)

### [Fire and Forget](https://useworkflow.dev/docs/foundations/starting-workflows\#fire-and-forget)

The most common pattern is to start a workflow and immediately return, letting it execute in the background:

```
import { start } from 'workflow/api';
import { sendNotifications } from './workflows/notifications';

export async function POST(request: Request) {
  // Start workflow and don't wait for it
  const run = await start(sendNotifications, [userId]);

  // Return immediately
  return Response.json({
    message: 'Notifications queued',
    runId: run.runId
  });
}
```

### [Wait for Completion](https://useworkflow.dev/docs/foundations/starting-workflows\#wait-for-completion)

If you need to wait for the workflow to complete before responding:

```
import { start } from 'workflow/api';
import { generateReport } from './workflows/reports';

export async function POST(request: Request) {
  const run = await start(generateReport, [reportId]);

  // Wait for the workflow to complete
  const report = await run.returnValue;

  return Response.json({ report });
}
```

Be cautious when waiting for `returnValue` \- if your workflow takes a long time, your API route may timeout.

### [Stream Updates to Client](https://useworkflow.dev/docs/foundations/starting-workflows\#stream-updates-to-client)

Stream real-time updates from your workflow as it executes, without waiting for completion:

```
import { start } from 'workflow/api';
import { generateAIContent } from './workflows/ai-generation';

export async function POST(request: Request) {
  const { prompt } = await request.json();

  // Start the workflow
  const run = await start(generateAIContent, [prompt]);

  // Get the readable stream (can also use run.readable as shorthand)
  const stream = run.getReadable();

  // Return the stream immediately
  return new Response(stream, {
    headers: {
      'Content-Type': 'application/octet-stream',
    },
  });
}
```

Your workflow can write to the stream using [`getWritable()`](https://useworkflow.dev/docs/api-reference/workflow/get-writable):

```
import { getWritable } from 'workflow';

export async function generateAIContent(prompt: string) {
  'use workflow';

  const writable = getWritable();

  await streamContentToClient(writable, prompt);

  return { status: 'complete' };
}

async function streamContentToClient(
  writable: WritableStream,
  prompt: string
) {
  'use step';

  const writer = writable.getWriter();

  // Stream updates as they become available
  for (let i = 0; i < 10; i++) {
    const chunk = new TextEncoder().encode(`Update ${i}\n`);
    await writer.write(chunk);
  }

  writer.releaseLock();
}
```

Streams are particularly useful for AI workflows where you want to show progress to users in real-time, or for long-running processes that produce intermediate results.

**Learn more**: [Streaming in Workflows](https://useworkflow.dev/docs/foundations/serialization#streaming)

### [Check Status Later](https://useworkflow.dev/docs/foundations/starting-workflows\#check-status-later)

You can retrieve a workflow run later using its `runId` with [`getRun()`](https://useworkflow.dev/docs/api-reference/workflow-api/get-run):

```
import { getRun } from 'workflow/api';

export async function GET(request: Request) {
  const url = new URL(request.url);
  const runId = url.searchParams.get('runId');

  // Retrieve the existing run
  const run = getRun(runId);

  // Check its status
  const status = await run.status;

  if (status === 'completed') {
    const result = await run.returnValue;
    return Response.json({ result });
  }

  return Response.json({ status });
}
```

## [Next Steps](https://useworkflow.dev/docs/foundations/starting-workflows\#next-steps)

Now that you understand how to start workflows and track their execution:

- Learn about [Control Flow Patterns](https://useworkflow.dev/docs/foundations/control-flow-patterns) for organizing complex workflows
- Explore [Errors & Retrying](https://useworkflow.dev/docs/foundations/errors-and-retries) to handle failures gracefully
- Check the [`start()` API Reference](https://useworkflow.dev/docs/api-reference/workflow-api/start) for complete details

[Workflows and Steps\\
\\
Previous Page](https://useworkflow.dev/docs/foundations/workflows-and-steps) [Control Flow Patterns\\
\\
Next Page](https://useworkflow.dev/docs/foundations/control-flow-patterns)

On this page

[The `start()` Function](https://useworkflow.dev/docs/foundations/starting-workflows#the-start-function) [The `Run` Object](https://useworkflow.dev/docs/foundations/starting-workflows#the-run-object) [Common Patterns](https://useworkflow.dev/docs/foundations/starting-workflows#common-patterns) [Fire and Forget](https://useworkflow.dev/docs/foundations/starting-workflows#fire-and-forget) [Wait for Completion](https://useworkflow.dev/docs/foundations/starting-workflows#wait-for-completion) [Stream Updates to Client](https://useworkflow.dev/docs/foundations/starting-workflows#stream-updates-to-client) [Check Status Later](https://useworkflow.dev/docs/foundations/starting-workflows#check-status-later) [Next Steps](https://useworkflow.dev/docs/foundations/starting-workflows#next-steps)

[GitHubEdit this page on GitHub](https://github.com/vercel/workflow/edit/main/content/docs/foundations/starting-workflows.mdx) Scroll to topGive feedbackCopy pageAsk AI about this pageOpen in chat

## Workflow Sleep Function
[API Reference](https://useworkflow.dev/docs/api-reference) [workflow](https://useworkflow.dev/docs/api-reference/workflow)

# sleep

Suspends a workflow for a specified duration or until an end date without consuming any resources. Once the duration or end date passes, the workflow will resume execution.

This is useful when you want to resume a workflow after some duration or date.

`sleep` is a _special_ type of step function and should be called directly inside workflow functions.

```
import { sleep } from "workflow"

async function testWorkflow() {
    "use workflow"
    await sleep("10s")
}
```

## [API Signature](https://useworkflow.dev/docs/api-reference/workflow/sleep\#api-signature)

### [Parameters](https://useworkflow.dev/docs/api-reference/workflow/sleep\#parameters)

This function has multiple signatures.

#### Signature 1

| Name | Type | Description |
| --- | --- | --- |
| `duration` | StringValue | The duration to sleep for, this is a string in the format<br>of `"1000ms"`, `"1s"`, `"1m"`, `"1h"`, or `"1d"`. |

#### Signature 2

| Name | Type | Description |
| --- | --- | --- |
| `date` | Date | The date to sleep until, this must be a future date. |

#### Signature 3

| Name | Type | Description |
| --- | --- | --- |
| `durationMs` | number | The duration to sleep for in milliseconds. |

## [Examples](https://useworkflow.dev/docs/api-reference/workflow/sleep\#examples)

### [Sleeping With a Duration](https://useworkflow.dev/docs/api-reference/workflow/sleep\#sleeping-with-a-duration)

You can specify a duration for `sleep` to suspend the workflow for a fixed amount of time.

```
import { sleep } from "workflow"

async function testWorkflow() {
    "use workflow"
    await sleep("1d")
}
```

### [Sleeping Until an End Date](https://useworkflow.dev/docs/api-reference/workflow/sleep\#sleeping-until-an-end-date)

You can specify a future `Date` object for `sleep` to suspend the workflow until a specific date.

```
import { sleep } from "workflow"

async function testWorkflow() {
    "use workflow"
    await sleep(new Date(Date.now() + 10_000))
}
```

[getWritable\\
\\
Retrieves the current workflow run's default writable stream.](https://useworkflow.dev/docs/api-reference/workflow/get-writable) [FatalError\\
\\
Next Page](https://useworkflow.dev/docs/api-reference/workflow/fatal-error)

On this page

[API Signature](https://useworkflow.dev/docs/api-reference/workflow/sleep#api-signature) [Parameters](https://useworkflow.dev/docs/api-reference/workflow/sleep#parameters) [Examples](https://useworkflow.dev/docs/api-reference/workflow/sleep#examples) [Sleeping With a Duration](https://useworkflow.dev/docs/api-reference/workflow/sleep#sleeping-with-a-duration) [Sleeping Until an End Date](https://useworkflow.dev/docs/api-reference/workflow/sleep#sleeping-until-an-end-date)

[GitHubEdit this page on GitHub](https://github.com/vercel/workflow/edit/main/content/docs/api-reference/workflow/sleep.mdx) Scroll to topGive feedbackCopy pageAsk AI about this pageOpen in chat

## Vercel World Deployment
[Deploying](https://useworkflow.dev/docs/deploying) [World](https://useworkflow.dev/docs/deploying/world)

# Vercel World

The **Vercel World** (`@workflow/world-vercel`) is a production-ready workflow backend integrated with Vercel's infrastructure. It provides scalable storage, robust queuing, and secure authentication for workflow deployments on the Vercel platform.

The Vercel world is designed for production deployments and provides:

- **Scalable storage** \- Cloud-based workflow data persistence
- **Distributed queuing** \- Reliable asynchronous step execution
- **Secure authentication** \- API token-based access control
- **Multi-environment support** \- Separate production, preview, and development environments
- **Team support** \- Integration with Vercel teams and projects

## [How It Works](https://useworkflow.dev/docs/deploying/world/vercel-world\#how-it-works)

### [Storage](https://useworkflow.dev/docs/deploying/world/vercel-world\#storage)

The Vercel world stores workflow data using Vercel's Workflow Storage API:

- Runs, steps, hooks, and streams are stored in Vercel's cloud infrastructure
- Data is automatically replicated and backed up
- Storage scales automatically with your application
- Data is encrypted at rest and in transit

### [Queuing](https://useworkflow.dev/docs/deploying/world/vercel-world\#queuing)

The Vercel world uses Vercel's distributed queue system:

1. When a step is enqueued, it's added to Vercel's queue service
2. The queue distributes steps across available serverless functions
3. Steps are executed with automatic retries and error handling
4. Failed steps are retried according to retry policies

### [Authentication](https://useworkflow.dev/docs/deploying/world/vercel-world\#authentication)

The Vercel world uses token-based authentication:

- API requests include authentication headers
- Tokens are scoped to specific projects and teams
- Tokens can be environment-specific (production, preview, development)
- Authentication is handled automatically in Vercel deployments

## [Deploying to Vercel](https://useworkflow.dev/docs/deploying/world/vercel-world\#deploying-to-vercel)

### [Automatic Configuration](https://useworkflow.dev/docs/deploying/world/vercel-world\#automatic-configuration)

When you deploy to Vercel, the world is configured automatically:

```
npx vercel deploy
```

No additional configuration is needed, Vercel automatically:

- Selects the Vercel World backend
- Sets up authentication
- Configures project and team IDs
- Provides storage and queuing infrastructure

### [Deployment Workflow](https://useworkflow.dev/docs/deploying/world/vercel-world\#deployment-workflow)

Deploy your application to Vercel using the Vercel CLI:

```
npx vercel deploy --prod
```

Workflows run automatically:

- Use Vercel World in production
- Data stored in Vercel's infrastructure
- Steps queued and executed on Vercel's serverless functions

## [Multi-Environment Support](https://useworkflow.dev/docs/deploying/world/vercel-world\#multi-environment-support)

The Vercel world supports multiple environments:

- **Production** \- Your live production deployment
- **Preview** \- Preview deployments for pull requests
- **Development** \- Development environment testing

Each environment has isolated workflow data, ensuring that development and testing don't interfere with production workflows.

## [Remote Access Configuration](https://useworkflow.dev/docs/deploying/world/vercel-world\#remote-access-configuration)

To inspect production workflows from your local machine using [observability tools](https://useworkflow.dev/docs/observability), you need to configure remote access.

### [Getting Authentication Tokens](https://useworkflow.dev/docs/deploying/world/vercel-world\#getting-authentication-tokens)

1. Go to [Vercel Dashboard → Settings → Tokens](https://vercel.com/account/tokens)
2. Create a new token with appropriate scopes
3. Save the token securely

### [Environment Variables](https://useworkflow.dev/docs/deploying/world/vercel-world\#environment-variables)

Configure remote access via environment variables:

```
# Set the target world
export WORKFLOW_TARGET_WORLD=vercel

# Authentication token
export WORKFLOW_VERCEL_AUTH_TOKEN=<your-token>

# Environment (production, preview, development)
export WORKFLOW_VERCEL_ENV=production

# Project ID
export WORKFLOW_VERCEL_PROJECT=<project-id>

# Team ID (if using Vercel teams)
export WORKFLOW_VERCEL_TEAM=<team-id>
```

### [CLI Flags](https://useworkflow.dev/docs/deploying/world/vercel-world\#cli-flags)

You can also pass configuration via CLI flags when using observability tools:

```
npx workflow inspect runs \
  --backend=vercel \
  --env=production \
  --project=my-project \
  --team=my-team \
  --authToken=<your-token>
```

Learn more about remote inspection in the [Observability](https://useworkflow.dev/docs/observability) section.

## [Team Support](https://useworkflow.dev/docs/deploying/world/vercel-world\#team-support)

If your project belongs to a Vercel team, the Vercel world automatically integrates with team permissions:

- Respects team access controls
- Requires team-scoped authentication tokens
- Isolates workflow data per team

## [Scalability](https://useworkflow.dev/docs/deploying/world/vercel-world\#scalability)

The Vercel world is designed for production scale:

- **Automatic scaling** \- Handles any number of concurrent workflows
- **Distributed execution** \- Steps run across multiple serverless functions
- **Global distribution** \- Works with Vercel's global edge network
- **High availability** \- Built-in redundancy and failover

## [Security](https://useworkflow.dev/docs/deploying/world/vercel-world\#security)

The Vercel world implements security best practices:

- **Token-based authentication** \- Secure API access
- **Environment isolation** \- Production, preview, and development data are separate
- **Encryption** \- Data encrypted at rest and in transit
- **Team permissions** \- Respects Vercel team access controls

## [API Reference](https://useworkflow.dev/docs/deploying/world/vercel-world\#api-reference)

### [createVercelWorld](https://useworkflow.dev/docs/deploying/world/vercel-world\#createvercelworld)

Creates a Vercel world instance:

```
function createVercelWorld(
  config?: APIConfig
): World
```

**Parameters:**

- `config.token` \- Authentication token
- `config.headers` \- Custom headers including:
  - `x-vercel-environment` \- Environment name
  - `x-vercel-project-id` \- Project ID
  - `x-vercel-team-id` \- Team ID
- `config.baseUrl` \- API base URL (default: `https://api.vercel.com/v1/workflow`)

**Returns:**

- `World` \- A world instance implementing the World interface

**Example:**

```
import { createVercelWorld } from '@workflow/world-vercel';

const world = createVercelWorld({
  token: process.env.WORKFLOW_VERCEL_AUTH_TOKEN,
  headers: {
    'x-vercel-environment': 'production',
    'x-vercel-project-id': 'my-project',
    'x-vercel-team-id': 'my-team',
  },
});
```

## [Troubleshooting](https://useworkflow.dev/docs/deploying/world/vercel-world\#troubleshooting)

### [Authentication Errors](https://useworkflow.dev/docs/deploying/world/vercel-world\#authentication-errors)

If you see authentication errors:

1. Verify your token is valid: `vercel whoami --token <your-token>`
2. Check token has necessary scopes
3. Ensure project and team IDs are correct

### [Environment Not Found](https://useworkflow.dev/docs/deploying/world/vercel-world\#environment-not-found)

If the environment is not found:

1. Verify the environment name (`production`, `preview`, `development`)
2. Check the project has been deployed to that environment
3. Ensure your token has access to the project

### [Deployment Issues](https://useworkflow.dev/docs/deploying/world/vercel-world\#deployment-issues)

If workflows don't work after deployment:

1. Verify `withWorkflow()` is wrapping your Next.js config
2. Check build logs for errors
3. Ensure workflow files are in the correct directory
4. Test locally first with [Local World](https://useworkflow.dev/docs/deploying/world/local-world)

## [Learn More](https://useworkflow.dev/docs/deploying/world/vercel-world\#learn-more)

- [World Interface](https://useworkflow.dev/docs/deploying/world) \- Understanding the World interface
- [Local World](https://useworkflow.dev/docs/deploying/world/local-world) \- For local development
- [Observability](https://useworkflow.dev/docs/observability) \- Monitoring and debugging tools
- [Vercel Deployment Documentation](https://vercel.com/docs/deployments/overview)
- [Workflow Documentation on Vercel](https://vercel.com/docs/workflow)

[Local World\\
\\
Previous Page](https://useworkflow.dev/docs/deploying/world/local-world) [Postgres World\\
\\
Next Page](https://useworkflow.dev/docs/deploying/world/postgres-world)

On this page

[How It Works](https://useworkflow.dev/docs/deploying/world/vercel-world#how-it-works) [Storage](https://useworkflow.dev/docs/deploying/world/vercel-world#storage) [Queuing](https://useworkflow.dev/docs/deploying/world/vercel-world#queuing) [Authentication](https://useworkflow.dev/docs/deploying/world/vercel-world#authentication) [Deploying to Vercel](https://useworkflow.dev/docs/deploying/world/vercel-world#deploying-to-vercel) [Automatic Configuration](https://useworkflow.dev/docs/deploying/world/vercel-world#automatic-configuration) [Deployment Workflow](https://useworkflow.dev/docs/deploying/world/vercel-world#deployment-workflow) [Multi-Environment Support](https://useworkflow.dev/docs/deploying/world/vercel-world#multi-environment-support) [Remote Access Configuration](https://useworkflow.dev/docs/deploying/world/vercel-world#remote-access-configuration) [Getting Authentication Tokens](https://useworkflow.dev/docs/deploying/world/vercel-world#getting-authentication-tokens) [Environment Variables](https://useworkflow.dev/docs/deploying/world/vercel-world#environment-variables) [CLI Flags](https://useworkflow.dev/docs/deploying/world/vercel-world#cli-flags) [Team Support](https://useworkflow.dev/docs/deploying/world/vercel-world#team-support) [Scalability](https://useworkflow.dev/docs/deploying/world/vercel-world#scalability) [Security](https://useworkflow.dev/docs/deploying/world/vercel-world#security) [API Reference](https://useworkflow.dev/docs/deploying/world/vercel-world#api-reference) [createVercelWorld](https://useworkflow.dev/docs/deploying/world/vercel-world#createvercelworld) [Troubleshooting](https://useworkflow.dev/docs/deploying/world/vercel-world#troubleshooting) [Authentication Errors](https://useworkflow.dev/docs/deploying/world/vercel-world#authentication-errors) [Environment Not Found](https://useworkflow.dev/docs/deploying/world/vercel-world#environment-not-found) [Deployment Issues](https://useworkflow.dev/docs/deploying/world/vercel-world#deployment-issues) [Learn More](https://useworkflow.dev/docs/deploying/world/vercel-world#learn-more)

[GitHubEdit this page on GitHub](https://github.com/vercel/workflow/edit/main/content/docs/deploying/world/vercel-world.mdx) Scroll to topGive feedbackCopy pageAsk AI about this pageOpen in chat

## Workflow API Reference
[API Reference](https://useworkflow.dev/docs/api-reference)

# workflow

Core workflow primitives including steps, context management, streaming, webhooks, and error handling.

## [Installation](https://useworkflow.dev/docs/api-reference/workflow\#installation)

npm

pnpm

yarn

bun

```
npm i workflow
```

## [Functions](https://useworkflow.dev/docs/api-reference/workflow\#functions)

Workflow DevKit contains the following functions you can use inside your workflow functions:

[**getWorkflowMetadata()** \\
\\
A function that returns context about the current workflow execution.](https://useworkflow.dev/docs/api-reference/workflow/get-workflow-metadata) [**getStepMetadata()** \\
\\
A function that returns context about the current step execution.](https://useworkflow.dev/docs/api-reference/workflow/get-step-metadata) [**sleep()** \\
\\
Sleeping workflows for a specified duration. Deterministic and replay-safe.](https://useworkflow.dev/docs/api-reference/workflow/sleep) [**fetch()** \\
\\
Make HTTP requests from within a workflow with automatic retry semantics.](https://useworkflow.dev/docs/api-reference/workflow/fetch) [**createHook()** \\
\\
Create a low-level hook to receive arbitrary payloads from external systems.](https://useworkflow.dev/docs/api-reference/workflow/create-hook) [**defineHook()** \\
\\
Type-safe hook helper for consistent payload types.](https://useworkflow.dev/docs/api-reference/workflow/define-hook) [**createWebhook()** \\
\\
Create a webhook that suspends the workflow until an HTTP request is received.](https://useworkflow.dev/docs/api-reference/workflow/create-webhook) [**getWritable()** \\
\\
Access the current workflow run's default stream.](https://useworkflow.dev/docs/api-reference/workflow/get-writable)

## [Error Classes](https://useworkflow.dev/docs/api-reference/workflow\#error-classes)

Workflow DevKit includes error classes that can be thrown in a workflow or step to change the error exit strategy of a workflow.

[**FatalError()** \\
\\
When thrown, marks a step as failed and the step is not retried.](https://useworkflow.dev/docs/api-reference/workflow/fatal-error) [**RetryableError()** \\
\\
When thrown, marks a step as retryable with an optional parameter.](https://useworkflow.dev/docs/api-reference/workflow/retryable-error)

[API Reference\\
\\
Previous Page](https://useworkflow.dev/docs/api-reference) [createHook\\
\\
Next Page](https://useworkflow.dev/docs/api-reference/workflow/create-hook)

On this page

[Installation](https://useworkflow.dev/docs/api-reference/workflow#installation) [Functions](https://useworkflow.dev/docs/api-reference/workflow#functions) [Error Classes](https://useworkflow.dev/docs/api-reference/workflow#error-classes)

[GitHubEdit this page on GitHub](https://github.com/vercel/workflow/edit/main/content/docs/api-reference/workflow/index.mdx) Scroll to topGive feedbackCopy pageAsk AI about this pageOpen in chat

## Chat

What is Workflow?How does retrying work?What control flow patterns are there?How do directives work?

Tip: You can open and close chat with `⌘I`

0 / 1000

## Workflow DevKit API
# API Reference

All the functions and primitives that come with Workflow DevKit by package.

[**workflow** \\
\\
Core workflow primitives including steps, context management, streaming, webhooks, and error handling.](https://useworkflow.dev/docs/api-reference/workflow) [**workflow/api** \\
\\
API reference for runtime functions from the `workflow/api` package.](https://useworkflow.dev/docs/api-reference/workflow-api) [**workflow/next** \\
\\
Next.js integration for Workflow DevKit that automatically configures bundling and runtime support.](https://useworkflow.dev/docs/api-reference/workflow-next) [**@workflow/ai** \\
\\
Helpers for integrating AI SDK for building AI-powered workflows.](https://useworkflow.dev/docs/api-reference/workflow-ai)

[webhook-response-not-sent\\
\\
Previous Page](https://useworkflow.dev/docs/errors/webhook-response-not-sent) [workflow\\
\\
Next Page](https://useworkflow.dev/docs/api-reference/workflow)

## Create Workflow Hook
[API Reference](https://useworkflow.dev/docs/api-reference) [workflow](https://useworkflow.dev/docs/api-reference/workflow)

# createHook

Creates a low-level hook primitive that can be used to resume a workflow run with arbitrary payloads.

Hooks allow external systems to send data to a paused workflow without the HTTP-specific constraints of webhooks. They're identified by a token and can receive any serializable payload.

```
import { createHook } from "workflow"

export async function hookWorkflow() {
  "use workflow";
  const hook = createHook();
  const result = await hook; // Suspends the workflow until the hook is resumed
}
```

## [API Signature](https://useworkflow.dev/docs/api-reference/workflow/create-hook\#api-signature)

### [Parameters](https://useworkflow.dev/docs/api-reference/workflow/create-hook\#parameters)

| Name | Type | Description |
| --- | --- | --- |
| `options` | HookOptions | Configuration options for the hook. |

#### [HookOptions](https://useworkflow.dev/docs/api-reference/workflow/create-hook\#hookoptions)

| Name | Type | Description |
| --- | --- | --- |
| `token` | string | Unique token that is used to associate with the hook.<br>When specifying an explicit token, the token should be constructed<br>with information that the dispatching side can reliably reconstruct<br>the token with the information it has available.<br>If not provided, a randomly generated token will be assigned. |
| `metadata` | Serializable | Additional user-defined data to include with the hook payload. |

### [Returns](https://useworkflow.dev/docs/api-reference/workflow/create-hook\#returns)

`Hook<T>`

#### [Hook](https://useworkflow.dev/docs/api-reference/workflow/create-hook\#hook)

| Name | Type | Description |
| --- | --- | --- |
| `token` | string | The token used to identify this hook. |

The returned `Hook` object also implements `AsyncIterable<T>`, which allows you to iterate over incoming payloads using `for await...of` syntax.

## [Examples](https://useworkflow.dev/docs/api-reference/workflow/create-hook\#examples)

### [Basic Usage](https://useworkflow.dev/docs/api-reference/workflow/create-hook\#basic-usage)

When creating a hook, you can specify a payload type to be used for automatic type safety.

```
import { createHook } from "workflow"

export async function approvalWorkflow() {
  "use workflow";

  const hook = createHook<{ approved: boolean; comment: string }>();
  console.log('Send approval to token:', hook.token);

  const result = await hook;

  if (result.approved) {
    console.log('Approved with comment:', result.comment);
  }
}
```

### [Customizing Tokens](https://useworkflow.dev/docs/api-reference/workflow/create-hook\#customizing-tokens)

Tokens are used to identify a specific hook. You can customize the token to be more specific to a use case.

```
import { createHook } from "workflow";

export async function slackBotWorkflow(channelId: string) {
  "use workflow";

  // Token constructed from channel ID
  const hook = createHook<SlackMessage>({
    token: `slack_webhook:${channelId}`,
  });

  for await (const message of hook) {
    if (message.text === '/stop') {
      break;
    }
    await processMessage(message);
  }
}
```

### [Waiting for Multiple Payloads](https://useworkflow.dev/docs/api-reference/workflow/create-hook\#waiting-for-multiple-payloads)

You can also wait for multiple payloads by using the `for await...of` syntax.

```
import { createHook } from "workflow"

export async function collectHookWorkflow() {
  'use workflow';

  const hook = createHook<{ message: string; done?: boolean }>();

  const payloads = [];
  for await (const payload of hook) {
    payloads.push(payload);

    if (payload.done) break;
  }

  return payloads;
}
```

## [Related Functions](https://useworkflow.dev/docs/api-reference/workflow/create-hook\#related-functions)

- [`defineHook()`](https://useworkflow.dev/docs/api-reference/workflow/define-hook) \- Type-safe hook helper
- [`resumeHook()`](https://useworkflow.dev/docs/api-reference/workflow-api/resume-hook) \- Resume a hook with a payload
- [`createWebhook()`](https://useworkflow.dev/docs/api-reference/workflow/create-webhook) \- Higher-level HTTP webhook abstraction

[workflow\\
\\
Previous Page](https://useworkflow.dev/docs/api-reference/workflow) [createWebhook\\
\\
Next Page](https://useworkflow.dev/docs/api-reference/workflow/create-webhook)

On this page

[API Signature](https://useworkflow.dev/docs/api-reference/workflow/create-hook#api-signature) [Parameters](https://useworkflow.dev/docs/api-reference/workflow/create-hook#parameters) [HookOptions](https://useworkflow.dev/docs/api-reference/workflow/create-hook#hookoptions) [Returns](https://useworkflow.dev/docs/api-reference/workflow/create-hook#returns) [Hook](https://useworkflow.dev/docs/api-reference/workflow/create-hook#hook) [Examples](https://useworkflow.dev/docs/api-reference/workflow/create-hook#examples) [Basic Usage](https://useworkflow.dev/docs/api-reference/workflow/create-hook#basic-usage) [Customizing Tokens](https://useworkflow.dev/docs/api-reference/workflow/create-hook#customizing-tokens) [Waiting for Multiple Payloads](https://useworkflow.dev/docs/api-reference/workflow/create-hook#waiting-for-multiple-payloads) [Related Functions](https://useworkflow.dev/docs/api-reference/workflow/create-hook#related-functions)

[GitHubEdit this page on GitHub](https://github.com/vercel/workflow/edit/main/content/docs/api-reference/workflow/create-hook.mdx) Scroll to topGive feedbackCopy pageAsk AI about this pageOpen in chat

## Workflow Programming Foundations
# Foundations

Workflow programming can be a slight shift from how you traditionally write real-world applications. Learning the foundations now will go a long way toward helping you use workflows effectively.

[**Workflows and Steps** \\
\\
Learn about the building blocks of durability](https://useworkflow.dev/docs/foundations/workflows-and-steps) [**Starting Workflows** \\
\\
Trigger workflows and track their execution using the `start()` function.](https://useworkflow.dev/docs/foundations/starting-workflows) [**Control Flow Patterns** \\
\\
Common control flow patterns useful in workflows.](https://useworkflow.dev/docs/foundations/control-flow-patterns) [**Errors & Retrying** \\
\\
Types of errors and how retrying work in workflows.](https://useworkflow.dev/docs/foundations/errors-and-retries) [**Webhooks (and hooks)** \\
\\
Respond to external events in your workflow using hooks and webhooks.](https://useworkflow.dev/docs/foundations/hooks)

[SvelteKit\\
\\
This guide will walk through setting up your first workflow in a SvelteKit app. Along the way, you'll learn more about the concepts that are fundamental to using the development kit in your own projects.](https://useworkflow.dev/docs/getting-started/sveltekit) [Workflows and Steps\\
\\
Next Page](https://useworkflow.dev/docs/foundations/workflows-and-steps)

## Hono Workflow Setup
[Getting Started](https://useworkflow.dev/docs/getting-started)

# Hono

This guide will walk through setting up your first workflow in a Hono app. Along the way, you'll learn more about the concepts that are fundamental to using the development kit in your own projects.

## [Create Your Hono Project](https://useworkflow.dev/docs/getting-started/hono\#create-your-hono-project)

Start by creating a new Hono project. This command will create a new directory named `my-workflow-app` and set up a Hono project inside it.

```
npm create hono@latest my-workflow-app -- --template=nodejs
```

Enter the newly created directory:

```
cd my-workflow-app
```

### [Install `workflow`, `nitro`, and `rollup`](https://useworkflow.dev/docs/getting-started/hono\#install-workflow-nitro-and-rollup)

npm

pnpm

yarn

bun

```
npm i workflow nitro rollup
```

By default, Hono doesn't include a build system. Nitro adds one which enables compiling workflows, runs, and deploys for development and production. Learn more about Nitro [here](https://v3.nitro.build/).

### [Configure Nitro](https://useworkflow.dev/docs/getting-started/hono\#configure-nitro)

Create a new file `nitro.config.ts` for your Nitro configuration with module `workflow/nitro`. This enables usage of the `"use workflow"` and `"use step"` directives.

nitro.config.ts

```

```

### Setup IntelliSense for TypeScript (Optional)

### [Update `package.json`](https://useworkflow.dev/docs/getting-started/hono\#update-packagejson)

To use the Nitro builder, update your `package.json` to include the following scripts:

package.json

```

```

## [Create Your First Workflow](https://useworkflow.dev/docs/getting-started/hono\#create-your-first-workflow)

Create a new file for our first workflow:

workflows/user-signup.ts

```

```

We'll fill in those functions next, but let's take a look at this code:

- We define a **workflow** function with the directive `"use workflow"`. Think of the workflow function as the _orchestrator_ of individual **steps**.
- The Workflow DevKit's `sleep` function allows us to suspend execution of the workflow without using up any resources. A sleep can be a few seconds, hours, days, or even months long.

## [Create Your Workflow Steps](https://useworkflow.dev/docs/getting-started/hono\#create-your-workflow-steps)

Let's now define those missing functions.

workflows/user-signup.ts

```

```

Taking a look at this code:

- Business logic lives inside **steps**. When a step is invoked inside a **workflow**, it gets enqueued to run on a separate request while the workflow is suspended, just like `sleep`.
- If a step throws an error, like in `sendWelcomeEmail`, the step will automatically be retried until it succeeds (or hits the step's max retry count).
- Steps can throw a `FatalError` if an error is intentional and should not be retried.

We'll dive deeper into workflows, steps, and other ways to suspend or handle
events in [Foundations](https://useworkflow.dev/docs/foundations).

## [Create Your Route Handler](https://useworkflow.dev/docs/getting-started/hono\#create-your-route-handler)

To invoke your new workflow, we'll create a new API route handler at `src/index.ts` with the following code:

src/index.ts

```

```

This route handler creates a `POST` request endpoint at `/api/signup` that will trigger your workflow.

## [Run in development](https://useworkflow.dev/docs/getting-started/hono\#run-in-development)

To start your development server, run the following command in your terminal in the Hono root directory:

```
npm run dev
```

Once your development server is running, you can trigger your workflow by running this command in the terminal:

```
curl -X POST --json '{"email":"hello@example.com"}' http://localhost:3000/api/signup
```

Check the Hono development server logs to see your workflow execute as well as the steps that are being processed.

Additionally, you can use the [Workflow DevKit CLI or Web UI](https://useworkflow.dev/docs/observability) to inspect your workflow runs and steps in detail.

```
npx workflow inspect runs # add '--web' for an interactive Web based UI
```

![Workflow DevKit Web UI](https://useworkflow.dev/o11y-ui.png)

## [Deploying to production](https://useworkflow.dev/docs/getting-started/hono\#deploying-to-production)

Workflow DevKit apps currently work best when deployed to [Vercel](https://vercel.com/home) and needs no special configuration.

Check the [Deploying](https://useworkflow.dev/docs/deploying) section to learn how your workflows can be deployed elsewhere.

## [Next Steps](https://useworkflow.dev/docs/getting-started/hono\#next-steps)

- Learn more about the [Foundations](https://useworkflow.dev/docs/foundations).
- Check [Errors](https://useworkflow.dev/docs/errors) if you encounter issues.
- Explore the [API Reference](https://useworkflow.dev/docs/api-reference).

[Express\\
\\
Previous Page](https://useworkflow.dev/docs/getting-started/express) [Nitro\\
\\
This guide will walk through setting up your first workflow in a Nitro v3 project. Along the way, you'll learn more about the concepts that are fundamental to using the development kit in your own projects.](https://useworkflow.dev/docs/getting-started/nitro)

On this page

[Create Your Hono Project](https://useworkflow.dev/docs/getting-started/hono#create-your-hono-project) [Install `workflow`, `nitro`, and `rollup`](https://useworkflow.dev/docs/getting-started/hono#install-workflow-nitro-and-rollup) [Configure Nitro](https://useworkflow.dev/docs/getting-started/hono#configure-nitro) [Update `package.json`](https://useworkflow.dev/docs/getting-started/hono#update-packagejson) [Create Your First Workflow](https://useworkflow.dev/docs/getting-started/hono#create-your-first-workflow) [Create Your Workflow Steps](https://useworkflow.dev/docs/getting-started/hono#create-your-workflow-steps) [Create Your Route Handler](https://useworkflow.dev/docs/getting-started/hono#create-your-route-handler) [Run in development](https://useworkflow.dev/docs/getting-started/hono#run-in-development) [Deploying to production](https://useworkflow.dev/docs/getting-started/hono#deploying-to-production) [Next Steps](https://useworkflow.dev/docs/getting-started/hono#next-steps)

[GitHubEdit this page on GitHub](https://github.com/vercel/workflow/edit/main/content/docs/getting-started/hono.mdx) Scroll to topGive feedbackCopy pageAsk AI about this pageOpen in chat

## Chat

What is Workflow?How does retrying work?What control flow patterns are there?How do directives work?

Tip: You can open and close chat with `⌘I`

0 / 1000

