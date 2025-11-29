# https://google.github.io/adk-docs/#go llms-full.txt

## Agent Development Kit
[Skip to content](https://google.github.io/adk-docs/#learn-more)

[Edit this page](https://github.com/google/adk-docs/edit/main/docs/index.md "Edit this page")[View source of this page](https://github.com/google/adk-docs/raw/main/docs/index.md "View source of this page")

![Agent Development Kit Logo](https://google.github.io/adk-docs/assets/agent-development-kit.png)

# Agent Development Kit

Agent Development Kit (ADK) is a flexible and modular framework for **developing**
**and deploying AI agents**. While optimized for Gemini and the Google ecosystem,
ADK is **model-agnostic**, **deployment-agnostic**, and is built for
**compatibility with other frameworks**. ADK was designed to make agent
development feel more like software development, to make it easier for
developers to create, deploy, and orchestrate agentic architectures that range
from simple tasks to complex workflows.

ALERT: ADK Python v1.19.0 requires Python 3.10 or higher

ADK Python release v1.19.0 requires Python 3.10 or higher. This change
is breaking for anyone attempting to use the v1.19.0 release with Python
3.9. For more release details, check out the
[release notes](https://github.com/google/adk-python/releases/tag/v1.19.0).

News: ADK Go v0.2.0 released!

ADK Go release v0.2.0 is live with a variety of improvements, including new
features, bug fixes, documentation updates, and significant code refactoring.
For release details, check out the
[release notes](https://github.com/google/adk-go/releases/tag/v0.2.0).

Get started:

[Python](https://google.github.io/adk-docs/#python)[Go](https://google.github.io/adk-docs/#go)[Java](https://google.github.io/adk-docs/#java)

`pip install google-adk`

`go get google.golang.org/adk`

pom.xml

```
<dependency>
    <groupId>com.google.adk</groupId>
    <artifactId>google-adk</artifactId>
    <version>0.3.0</version>
</dependency>
```

build.gradle

```
dependencies {
    implementation 'com.google.adk:google-adk:0.3.0'
}
```

[Start with Python](https://google.github.io/adk-docs/get-started/python/) [Start with Go](https://google.github.io/adk-docs/get-started/go/) [Start with Java](https://google.github.io/adk-docs/get-started/java/)

* * *

## Learn more [¶](https://google.github.io/adk-docs/\#learn-more "Permanent link")

[Watch "Introducing Agent Development Kit"!](https://www.youtube.com/watch?v=zgrOwow_uTQ)

- **Flexible Orchestration**


* * *


Define workflows using workflow agents (`Sequential`, `Parallel`, `Loop`)
for predictable pipelines, or leverage LLM-driven dynamic routing
(`LlmAgent` transfer) for adaptive behavior.

[**Learn about agents**](https://google.github.io/adk-docs/agents/)

- **Multi-Agent Architecture**


* * *


Build modular and scalable applications by composing multiple specialized
agents in a hierarchy. Enable complex coordination and delegation.

[**Explore multi-agent systems**](https://google.github.io/adk-docs/agents/multi-agents/)

- **Rich Tool Ecosystem**


* * *


Equip agents with diverse capabilities: use pre-built tools (Search, Code
Exec), create custom functions, integrate 3rd-party libraries, or even use
other agents as tools.

[**Browse tools**](https://google.github.io/adk-docs/tools/)

- **Deployment Ready**


* * *


Containerize and deploy your agents anywhere – run locally, scale with
Vertex AI Agent Engine, or integrate into custom infrastructure using Cloud
Run or Docker.

[**Deploy agents**](https://google.github.io/adk-docs/deploy/)

- **Built-in Evaluation**


* * *


Systematically assess agent performance by evaluating both the final
response quality and the step-by-step execution trajectory against
predefined test cases.

[**Evaluate agents**](https://google.github.io/adk-docs/evaluate/)

- **Building Safe and Secure Agents**


* * *


Learn how to building powerful and trustworthy agents by implementing
security and safety patterns and best practices into your agent's design.

[**Safety and Security**](https://google.github.io/adk-docs/safety/)


Back to top

## A2A Remote Agent Guide
[Skip to content](https://google.github.io/adk-docs/a2a/quickstart-consuming-go/#quickstart-consuming-a-remote-agent-via-a2a)

[Edit this page](https://github.com/google/adk-docs/edit/main/docs/a2a/quickstart-consuming-go.md "Edit this page") [View source of this page](https://github.com/google/adk-docs/raw/main/docs/a2a/quickstart-consuming-go.md "View source of this page")

# Quickstart: Consuming a remote agent via A2A [¶](https://google.github.io/adk-docs/a2a/quickstart-consuming-go/\#quickstart-consuming-a-remote-agent-via-a2a "Permanent link")

Supported in ADKGoExperimental

This quickstart covers the most common starting point for any developer: **"There is a remote agent, how do I let my ADK agent use it via A2A?"**. This is crucial for building complex multi-agent systems where different agents need to collaborate and interact.

## Overview [¶](https://google.github.io/adk-docs/a2a/quickstart-consuming-go/\#overview "Permanent link")

This sample demonstrates the **Agent-to-Agent (A2A)** architecture in the Agent Development Kit (ADK), showcasing how multiple agents can work together to handle complex tasks. The sample implements an agent that can roll dice and check if numbers are prime.

```
┌─────────────────┐    ┌──────────────────┐    ┌────────────────────┐
│   Root Agent    │───▶│   Roll Agent     │    │   Remote Prime     │
│  (Local)        │    │   (Local)        │    │   Agent            │
│                 │    │                  │    │  (localhost:8001)  │
│                 │───▶│                  │◀───│                    │
└─────────────────┘    └──────────────────┘    └────────────────────┘
```

The A2A Basic sample consists of:

- **Root Agent** (`root_agent`): The main orchestrator that delegates tasks to specialized sub-agents
- **Roll Agent** (`roll_agent`): A local sub-agent that handles dice rolling operations
- **Prime Agent** (`prime_agent`): A remote A2A agent that checks if numbers are prime, this agent is running on a separate A2A server

## Exposing Your Agent with the ADK Server [¶](https://google.github.io/adk-docs/a2a/quickstart-consuming-go/\#exposing-your-agent-with-the-adk-server "Permanent link")

In the `a2a_basic` example, you will first need to expose the `check_prime_agent` via an A2A server, so that the local root agent can use it.

### 1\. Getting the Sample Code [¶](https://google.github.io/adk-docs/a2a/quickstart-consuming-go/\#getting-the-sample-code "Permanent link")

First, make sure you have Go installed and your environment is set up.

You can clone and navigate to the [**`a2a_basic`** sample](https://github.com/google/adk-docs/tree/main/examples/go/a2a_basic) here:

```
cd examples/go/a2a_basic
```

As you'll see, the folder structure is as follows:

```
a2a_basic/
├── remote_a2a/
│   └── check_prime_agent/
│       └── main.go
├── go.mod
├── go.sum
└── main.go # local root agent
```

#### Main Agent (`a2a_basic/main.go`) [¶](https://google.github.io/adk-docs/a2a/quickstart-consuming-go/\#main-agent-a2a_basicmaingo "Permanent link")

- **`rollDieTool`**: Function tool for rolling dice
- **`newRollAgent`**: Local agent specialized in dice rolling
- **`newPrimeAgent`**: Remote A2A agent configuration
- **`newRootAgent`**: Main orchestrator with delegation logic

#### Remote Prime Agent (`a2a_basic/remote_a2a/check_prime_agent/main.go`) [¶](https://google.github.io/adk-docs/a2a/quickstart-consuming-go/\#remote-prime-agent-a2a_basicremote_a2acheck_prime_agentmaingo "Permanent link")

- **`checkPrimeTool`**: Prime number checking algorithm
- **`main`**: Implementation of the prime checking service and A2A server.

### 2\. Start the Remote Prime Agent server [¶](https://google.github.io/adk-docs/a2a/quickstart-consuming-go/\#start-the-remote-prime-agent-server "Permanent link")

To show how your ADK agent can consume a remote agent via A2A, you'll first need to start a remote agent server, which will host the prime agent (under `check_prime_agent`).

```
# Start the remote a2a server that serves the check_prime_agent on port 8001
go run remote_a2a/check_prime_agent/main.go
```

Once executed, you should see something like:

```
2025/11/06 11:00:19 Starting A2A prime checker server on port 8001
2025/11/06 11:00:19 Starting the web server: &{port:8001}
2025/11/06 11:00:19
2025/11/06 11:00:19 Web servers starts on http://localhost:8001
2025/11/06 11:00:19        a2a:  you can access A2A using jsonrpc protocol: http://localhost:8001
```

### 3\. Look out for the required agent card of the remote agent [¶](https://google.github.io/adk-docs/a2a/quickstart-consuming-go/\#look-out-for-the-required-agent-card-of-the-remote-agent "Permanent link")

A2A Protocol requires that each agent must have an agent card that describes what it does.

In the Go ADK, the agent card is generated dynamically when you expose an agent using the A2A launcher. You can visit `http://localhost:8001/.well-known/agent-card.json` to see the generated card.

### 4\. Run the Main (Consuming) Agent [¶](https://google.github.io/adk-docs/a2a/quickstart-consuming-go/\#run-the-main-consuming-agent "Permanent link")

```
# In a separate terminal, run the main agent
go run main.go
```

#### How it works [¶](https://google.github.io/adk-docs/a2a/quickstart-consuming-go/\#how-it-works "Permanent link")

The main agent uses `remoteagent.New` to consume the remote agent (`prime_agent` in our example). As you can see below, it requires the `Name`, `Description`, and the `AgentCardSource` URL.

a2a\_basic/main.go

```
func newPrimeAgent() (agent.Agent, error) {
    remoteAgent, err := remoteagent.NewA2A(remoteagent.A2AConfig{
        Name:            "prime_agent",
        Description:     "Agent that handles checking if numbers are prime.",
        AgentCardSource: "http://localhost:8001",
    })
    if err != nil {
        return nil, fmt.Errorf("failed to create remote prime agent: %w", err)
    }
    return remoteAgent, nil
}
```

Then, you can simply use the remote agent in your root agent. In this case, `primeAgent` is used as one of the sub-agents in the `root_agent` below:

a2a\_basic/main.go

```
func newRootAgent(ctx context.Context, rollAgent, primeAgent agent.Agent) (agent.Agent, error) {
    model, err := gemini.NewModel(ctx, "gemini-2.0-flash", &genai.ClientConfig{})
    if err != nil {
        return nil, err
    }
    return llmagent.New(llmagent.Config{
        Name:  "root_agent",
        Model: model,
        Instruction: `
      You are a helpful assistant that can roll dice and check if numbers are prime.
      You delegate rolling dice tasks to the roll_agent and prime checking tasks to the prime_agent.
      Follow these steps:
      1. If the user asks to roll a die, delegate to the roll_agent.
      2. If the user asks to check primes, delegate to the prime_agent.
      3. If the user asks to roll a die and then check if the result is prime, call roll_agent first, then pass the result to prime_agent.
      Always clarify the results before proceeding.
    `,
        SubAgents: []agent.Agent{rollAgent, primeAgent},
        Tools:     []tool.Tool{},
    })
}
```

## Example Interactions [¶](https://google.github.io/adk-docs/a2a/quickstart-consuming-go/\#example-interactions "Permanent link")

Once both your main and remote agents are running, you can interact with the root agent to see how it calls the remote agent via A2A:

**Simple Dice Rolling:**
This interaction uses a local agent, the Roll Agent:

```
User: Roll a 6-sided die
Bot calls tool: transfer_to_agent with args: map[agent_name:roll_agent]
Bot calls tool: roll_die with args: map[sides:6]
Bot: I rolled a 6-sided die and the result is 6.
```

**Prime Number Checking:**

This interaction uses a remote agent via A2A, the Prime Agent:

```
User: Is 7 a prime number?
Bot calls tool: transfer_to_agent with args: map[agent_name:prime_agent]
Bot calls tool: prime_checking with args: map[nums:[7]]
Bot: Yes, 7 is a prime number.
```

**Combined Operations:**

This interaction uses both the local Roll Agent and the remote Prime Agent:

```
User: roll a die and check if it's a prime
Bot: Okay, I will first roll a die and then check if the result is a prime number.

Bot calls tool: transfer_to_agent with args: map[agent_name:roll_agent]
Bot calls tool: roll_die with args: map[sides:6]
Bot calls tool: transfer_to_agent with args: map[agent_name:prime_agent]
Bot calls tool: prime_checking with args: map[nums:[3]]
Bot: 3 is a prime number.
```

## Next Steps [¶](https://google.github.io/adk-docs/a2a/quickstart-consuming-go/\#next-steps "Permanent link")

Now that you have created an agent that's using a remote agent via an A2A server, the next step is to learn how to expose your own agent.

- [**A2A Quickstart (Exposing)**](https://google.github.io/adk-docs/a2a/quickstart-exposing-go/): Learn how to expose your existing agent so that other agents can use it via the A2A Protocol.

Back to top

## LLM Flow API Overview
* * *

package com.google.adk.flows.llmflows

- Related Packages





Package



Description



[com.google.adk.flows](https://google.github.io/adk-docs/api-reference/java/com/google/adk/flows/package-summary.html)







[com.google.adk.flows.llmflows.audio](https://google.github.io/adk-docs/api-reference/java/com/google/adk/flows/llmflows/audio/package-summary.html)

- All Classes and InterfacesInterfacesClasses







Class



Description



[AgentTransfer](https://google.github.io/adk-docs/api-reference/java/com/google/adk/flows/llmflows/AgentTransfer.html "class in com.google.adk.flows.llmflows")





[`RequestProcessor`](https://google.github.io/adk-docs/api-reference/java/com/google/adk/flows/llmflows/RequestProcessor.html "interface in com.google.adk.flows.llmflows") that handles agent transfer for LLM flow.





[AutoFlow](https://google.github.io/adk-docs/api-reference/java/com/google/adk/flows/llmflows/AutoFlow.html "class in com.google.adk.flows.llmflows")





LLM flow with automatic agent transfer support.





[BaseLlmFlow](https://google.github.io/adk-docs/api-reference/java/com/google/adk/flows/llmflows/BaseLlmFlow.html "class in com.google.adk.flows.llmflows")





A basic flow that calls the LLM in a loop until a final response is generated.





[Basic](https://google.github.io/adk-docs/api-reference/java/com/google/adk/flows/llmflows/Basic.html "class in com.google.adk.flows.llmflows")





[`RequestProcessor`](https://google.github.io/adk-docs/api-reference/java/com/google/adk/flows/llmflows/RequestProcessor.html "interface in com.google.adk.flows.llmflows") that handles basic information to build the LLM request.





[Contents](https://google.github.io/adk-docs/api-reference/java/com/google/adk/flows/llmflows/Contents.html "class in com.google.adk.flows.llmflows")





[`RequestProcessor`](https://google.github.io/adk-docs/api-reference/java/com/google/adk/flows/llmflows/RequestProcessor.html "interface in com.google.adk.flows.llmflows") that populates content in request for LLM flows.





[Examples](https://google.github.io/adk-docs/api-reference/java/com/google/adk/flows/llmflows/Examples.html "class in com.google.adk.flows.llmflows")





[`RequestProcessor`](https://google.github.io/adk-docs/api-reference/java/com/google/adk/flows/llmflows/RequestProcessor.html "interface in com.google.adk.flows.llmflows") that populates examples in LLM request.





[Functions](https://google.github.io/adk-docs/api-reference/java/com/google/adk/flows/llmflows/Functions.html "class in com.google.adk.flows.llmflows")





Utility class for handling function calls.





[Identity](https://google.github.io/adk-docs/api-reference/java/com/google/adk/flows/llmflows/Identity.html "class in com.google.adk.flows.llmflows")





[`RequestProcessor`](https://google.github.io/adk-docs/api-reference/java/com/google/adk/flows/llmflows/RequestProcessor.html "interface in com.google.adk.flows.llmflows") that gives the agent identity from the framework





[Instructions](https://google.github.io/adk-docs/api-reference/java/com/google/adk/flows/llmflows/Instructions.html "class in com.google.adk.flows.llmflows")





[`RequestProcessor`](https://google.github.io/adk-docs/api-reference/java/com/google/adk/flows/llmflows/RequestProcessor.html "interface in com.google.adk.flows.llmflows") that handles instructions and global instructions for LLM flows.





[RequestProcessor](https://google.github.io/adk-docs/api-reference/java/com/google/adk/flows/llmflows/RequestProcessor.html "interface in com.google.adk.flows.llmflows")







[RequestProcessor.RequestProcessingResult](https://google.github.io/adk-docs/api-reference/java/com/google/adk/flows/llmflows/RequestProcessor.RequestProcessingResult.html "class in com.google.adk.flows.llmflows")







[ResponseProcessor](https://google.github.io/adk-docs/api-reference/java/com/google/adk/flows/llmflows/ResponseProcessor.html "interface in com.google.adk.flows.llmflows")







[ResponseProcessor.ResponseProcessingResult](https://google.github.io/adk-docs/api-reference/java/com/google/adk/flows/llmflows/ResponseProcessor.ResponseProcessingResult.html "class in com.google.adk.flows.llmflows")







[SingleFlow](https://google.github.io/adk-docs/api-reference/java/com/google/adk/flows/llmflows/SingleFlow.html "class in com.google.adk.flows.llmflows")





Basic LLM flow with fixed request processors and no response post-processing.

## After Agent Callback
Enclosing class:`Callbacks`Functional Interface:This is a functional interface and can therefore be used as the assignment target for a lambda expression or method reference.

* * *

[@FunctionalInterface](https://docs.oracle.com/en/java/javase/17/docs/api/java.base/java/lang/FunctionalInterface.html "class or interface in java.lang")public static interface Callbacks.AfterAgentCallback

- ## Method Summary [![Link icon](https://google.github.io/adk-docs/api-reference/java/resource-files/link.svg)](https://google.github.io/adk-docs/api-reference/java/com/google/adk/agents/Callbacks.AfterAgentCallback.html\#method-summary)





All MethodsInstance MethodsAbstract Methods







Modifier and Type



Method



Description



`io.reactivex.rxjava3.core.Maybe<com.google.genai.types.Content>`



`call(CallbackContext callbackContext)`





Async callback after agent runs.


- ## Method Details [![Link icon](https://google.github.io/adk-docs/api-reference/java/resource-files/link.svg)](https://google.github.io/adk-docs/api-reference/java/com/google/adk/agents/Callbacks.AfterAgentCallback.html\#method-detail)



- ### call [![Link icon](https://google.github.io/adk-docs/api-reference/java/resource-files/link.svg)](https://google.github.io/adk-docs/api-reference/java/com/google/adk/agents/Callbacks.AfterAgentCallback.html\#call(com.google.adk.agents.CallbackContext))





io.reactivex.rxjava3.core.Maybe<com.google.genai.types.Content>call( [CallbackContext](https://google.github.io/adk-docs/api-reference/java/com/google/adk/agents/CallbackContext.html "class in com.google.adk.agents") callbackContext)



Async callback after agent runs.

Parameters:`callbackContext` \- Callback context.Returns:modified content, or empty to keep original.

## Instruction Static
[java.lang.Object](https://docs.oracle.com/en/java/javase/17/docs/api/java.base/java/lang/Object.html "class or interface in java.lang")

[java.lang.Record](https://docs.oracle.com/en/java/javase/17/docs/api/java.base/java/lang/Record.html "class or interface in java.lang")

com.google.adk.agents.Instruction.Static

All Implemented Interfaces:`Instruction`Enclosing interface:`Instruction`

* * *

public static record Instruction.Static( [String](https://docs.oracle.com/en/java/javase/17/docs/api/java.base/java/lang/String.html "class or interface in java.lang") instruction)
extends [Record](https://docs.oracle.com/en/java/javase/17/docs/api/java.base/java/lang/Record.html "class or interface in java.lang")
implements [Instruction](https://google.github.io/adk-docs/api-reference/java/com/google/adk/agents/Instruction.html "interface in com.google.adk.agents")

Plain instruction directly provided to the agent.

- ## Nested Class Summary [![Link icon](https://google.github.io/adk-docs/api-reference/java/resource-files/link.svg)](https://google.github.io/adk-docs/api-reference/java/com/google/adk/agents/Instruction.Static.html\#nested-class-summary)





### Nested classes/interfaces inherited from interface com.google.adk.agents. [Instruction](https://google.github.io/adk-docs/api-reference/java/com/google/adk/agents/Instruction.html "interface in com.google.adk.agents") [![Link icon](https://google.github.io/adk-docs/api-reference/java/resource-files/link.svg)](https://google.github.io/adk-docs/api-reference/java/com/google/adk/agents/Instruction.Static.html\#nested-classes-inherited-from-class-com.google.adk.agents.Instruction)

`Instruction.Provider, Instruction.Static`

- ## Constructor Summary [![Link icon](https://google.github.io/adk-docs/api-reference/java/resource-files/link.svg)](https://google.github.io/adk-docs/api-reference/java/com/google/adk/agents/Instruction.Static.html\#constructor-summary)



Constructors





Constructor



Description



`Static(String instruction)`





Creates an instance of a `Static` record class.

- ## Method Summary [![Link icon](https://google.github.io/adk-docs/api-reference/java/resource-files/link.svg)](https://google.github.io/adk-docs/api-reference/java/com/google/adk/agents/Instruction.Static.html\#method-summary)





All MethodsInstance MethodsConcrete Methods







Modifier and Type



Method



Description



`final boolean`



`equals(Object o)`





Indicates whether some other object is "equal to" this one.





`final int`



`hashCode()`





Returns a hash code value for this object.





`String`



`instruction()`





Returns the value of the `instruction` record component.





`final String`



`toString()`





Returns a string representation of this record class.













### Methods inherited from class java.lang. [Object](https://docs.oracle.com/en/java/javase/17/docs/api/java.base/java/lang/Object.html "class or interface in java.lang") [![Link icon](https://google.github.io/adk-docs/api-reference/java/resource-files/link.svg)](https://google.github.io/adk-docs/api-reference/java/com/google/adk/agents/Instruction.Static.html\#methods-inherited-from-class-java.lang.Object)

`clone, finalize, getClass, notify, notifyAll, wait, wait, wait`


- ## Constructor Details [![Link icon](https://google.github.io/adk-docs/api-reference/java/resource-files/link.svg)](https://google.github.io/adk-docs/api-reference/java/com/google/adk/agents/Instruction.Static.html\#constructor-detail)



- ### Static [![Link icon](https://google.github.io/adk-docs/api-reference/java/resource-files/link.svg)](https://google.github.io/adk-docs/api-reference/java/com/google/adk/agents/Instruction.Static.html\#%3Cinit%3E(java.lang.String))





publicStatic( [String](https://docs.oracle.com/en/java/javase/17/docs/api/java.base/java/lang/String.html "class or interface in java.lang") instruction)



Creates an instance of a `Static` record class.

Parameters:`instruction` \- the value for the `instruction` record component


- ## Method Details [![Link icon](https://google.github.io/adk-docs/api-reference/java/resource-files/link.svg)](https://google.github.io/adk-docs/api-reference/java/com/google/adk/agents/Instruction.Static.html\#method-detail)



- ### toString [![Link icon](https://google.github.io/adk-docs/api-reference/java/resource-files/link.svg)](https://google.github.io/adk-docs/api-reference/java/com/google/adk/agents/Instruction.Static.html\#toString())





public final[String](https://docs.oracle.com/en/java/javase/17/docs/api/java.base/java/lang/String.html "class or interface in java.lang")toString()



Returns a string representation of this record class. The representation contains the name of the class, followed by the name and value of each of the record components.

Specified by:`toString` in class `Record`Returns:a string representation of this object

- ### hashCode [![Link icon](https://google.github.io/adk-docs/api-reference/java/resource-files/link.svg)](https://google.github.io/adk-docs/api-reference/java/com/google/adk/agents/Instruction.Static.html\#hashCode())





public finalinthashCode()



Returns a hash code value for this object. The value is derived from the hash code of each of the record components.

Specified by:`hashCode` in class `Record`Returns:a hash code value for this object

- ### equals [![Link icon](https://google.github.io/adk-docs/api-reference/java/resource-files/link.svg)](https://google.github.io/adk-docs/api-reference/java/com/google/adk/agents/Instruction.Static.html\#equals(java.lang.Object))





public finalbooleanequals( [Object](https://docs.oracle.com/en/java/javase/17/docs/api/java.base/java/lang/Object.html "class or interface in java.lang") o)



Indicates whether some other object is "equal to" this one. The objects are equal if the other object is of the same class and if all the record components are equal. All components in this record class are compared with [`Objects::equals(Object,Object)`](https://docs.oracle.com/en/java/javase/17/docs/api/java.base/java/util/Objects.html#equals(java.lang.Object,java.lang.Object) "class or interface in java.util").

Specified by:`equals` in class `Record`Parameters:`o` \- the object with which to compareReturns:`true` if this object is the same as the `o` argument; `false` otherwise.

- ### instruction [![Link icon](https://google.github.io/adk-docs/api-reference/java/resource-files/link.svg)](https://google.github.io/adk-docs/api-reference/java/com/google/adk/agents/Instruction.Static.html\#instruction())





public[String](https://docs.oracle.com/en/java/javase/17/docs/api/java.base/java/lang/String.html "class or interface in java.lang")instruction()



Returns the value of the `instruction` record component.

Returns:the value of the `instruction` record component

## MCP Initialization Exception
[java.lang.Object](https://docs.oracle.com/en/java/javase/17/docs/api/java.base/java/lang/Object.html "class or interface in java.lang")

[java.lang.Throwable](https://docs.oracle.com/en/java/javase/17/docs/api/java.base/java/lang/Throwable.html "class or interface in java.lang")

[java.lang.Exception](https://docs.oracle.com/en/java/javase/17/docs/api/java.base/java/lang/Exception.html "class or interface in java.lang")

[java.lang.RuntimeException](https://docs.oracle.com/en/java/javase/17/docs/api/java.base/java/lang/RuntimeException.html "class or interface in java.lang")

[com.google.adk.tools.mcp.McpToolset.McpToolsetException](https://google.github.io/adk-docs/api-reference/java/com/google/adk/tools/mcp/McpToolset.McpToolsetException.html "class in com.google.adk.tools.mcp")

com.google.adk.tools.mcp.McpToolset.McpInitializationException

All Implemented Interfaces:`Serializable`Enclosing class:`McpToolset`

* * *

public static class McpToolset.McpInitializationExceptionextends [McpToolset.McpToolsetException](https://google.github.io/adk-docs/api-reference/java/com/google/adk/tools/mcp/McpToolset.McpToolsetException.html "class in com.google.adk.tools.mcp")

Exception thrown when there's an error during MCP session initialization.

See Also:

- [Serialized Form](https://google.github.io/adk-docs/api-reference/java/serialized-form.html#com.google.adk.tools.mcp.McpToolset.McpInitializationException)

- ## Constructor Summary [![Link icon](https://google.github.io/adk-docs/api-reference/java/resource-files/link.svg)](https://google.github.io/adk-docs/api-reference/java/com/google/adk/tools/mcp/McpToolset.McpInitializationException.html\#constructor-summary)



Constructors





Constructor



Description



`McpInitializationException(String message,
Throwable cause)`

- ## Method Summary [![Link icon](https://google.github.io/adk-docs/api-reference/java/resource-files/link.svg)](https://google.github.io/adk-docs/api-reference/java/com/google/adk/tools/mcp/McpToolset.McpInitializationException.html\#method-summary)





### Methods inherited from class java.lang. [Throwable](https://docs.oracle.com/en/java/javase/17/docs/api/java.base/java/lang/Throwable.html "class or interface in java.lang") [![Link icon](https://google.github.io/adk-docs/api-reference/java/resource-files/link.svg)](https://google.github.io/adk-docs/api-reference/java/com/google/adk/tools/mcp/McpToolset.McpInitializationException.html\#methods-inherited-from-class-java.lang.Throwable)

`addSuppressed, fillInStackTrace, getCause, getLocalizedMessage, getMessage, getStackTrace, getSuppressed, initCause, printStackTrace, printStackTrace, printStackTrace, setStackTrace, toString`





### Methods inherited from class java.lang. [Object](https://docs.oracle.com/en/java/javase/17/docs/api/java.base/java/lang/Object.html "class or interface in java.lang") [![Link icon](https://google.github.io/adk-docs/api-reference/java/resource-files/link.svg)](https://google.github.io/adk-docs/api-reference/java/com/google/adk/tools/mcp/McpToolset.McpInitializationException.html\#methods-inherited-from-class-java.lang.Object)

`clone, equals, finalize, getClass, hashCode, notify, notifyAll, wait, wait, wait`


- ## Constructor Details [![Link icon](https://google.github.io/adk-docs/api-reference/java/resource-files/link.svg)](https://google.github.io/adk-docs/api-reference/java/com/google/adk/tools/mcp/McpToolset.McpInitializationException.html\#constructor-detail)



- ### McpInitializationException [![Link icon](https://google.github.io/adk-docs/api-reference/java/resource-files/link.svg)](https://google.github.io/adk-docs/api-reference/java/com/google/adk/tools/mcp/McpToolset.McpInitializationException.html\#%3Cinit%3E(java.lang.String,java.lang.Throwable))





publicMcpInitializationException( [String](https://docs.oracle.com/en/java/javase/17/docs/api/java.base/java/lang/String.html "class or interface in java.lang") message,
[Throwable](https://docs.oracle.com/en/java/javase/17/docs/api/java.base/java/lang/Throwable.html "class or interface in java.lang") cause)

## Java LLM Functions
No usage of com.google.adk.flows.llmflows.Functions

## ADK Utils Package
* * *

package com.google.adk.utils

- Related Packages





Package



Description



[com.google.adk](https://google.github.io/adk-docs/api-reference/java/com/google/adk/package-summary.html)

- Classes





Class



Description



[CollectionUtils](https://google.github.io/adk-docs/api-reference/java/com/google/adk/utils/CollectionUtils.html "class in com.google.adk.utils")





Frequently used code snippets for collections.





[InstructionUtils](https://google.github.io/adk-docs/api-reference/java/com/google/adk/utils/InstructionUtils.html "class in com.google.adk.utils")





Utility methods for handling instruction templates.





[Pairs](https://google.github.io/adk-docs/api-reference/java/com/google/adk/utils/Pairs.html "class in com.google.adk.utils")





Utility class for creating ConcurrentHashMaps.

## RunConfig Class
[java.lang.Object](https://docs.oracle.com/en/java/javase/17/docs/api/java.base/java/lang/Object.html "class or interface in java.lang")

com.google.adk.agents.RunConfig

* * *

public abstract class RunConfigextends [Object](https://docs.oracle.com/en/java/javase/17/docs/api/java.base/java/lang/Object.html "class or interface in java.lang")

Configuration to modify an agent's LLM's underlying behavior.

- ## Nested Class Summary [![Link icon](https://google.github.io/adk-docs/api-reference/java/resource-files/link.svg)](https://google.github.io/adk-docs/api-reference/java/com/google/adk/agents/RunConfig.html\#nested-class-summary)



Nested Classes





Modifier and Type



Class



Description



`static class`



`RunConfig.Builder`





Builder for [`RunConfig`](https://google.github.io/adk-docs/api-reference/java/com/google/adk/agents/RunConfig.html "class in com.google.adk.agents").





`static enum`



`RunConfig.StreamingMode`





Streaming mode for the runner.

- ## Constructor Summary [![Link icon](https://google.github.io/adk-docs/api-reference/java/resource-files/link.svg)](https://google.github.io/adk-docs/api-reference/java/com/google/adk/agents/RunConfig.html\#constructor-summary)



Constructors





Constructor



Description



`RunConfig()`

- ## Method Summary [![Link icon](https://google.github.io/adk-docs/api-reference/java/resource-files/link.svg)](https://google.github.io/adk-docs/api-reference/java/com/google/adk/agents/RunConfig.html\#method-summary)





All MethodsStatic MethodsInstance MethodsAbstract MethodsConcrete Methods







Modifier and Type



Method



Description



`static RunConfig.Builder`



`builder()`







`static RunConfig.Builder`



`builder(RunConfig runConfig)`







`abstract int`



`maxLlmCalls()`







`abstract @Nullable com.google.genai.types.AudioTranscriptionConfig`



`outputAudioTranscription()`







`abstract com.google.common.collect.ImmutableList<com.google.genai.types.Modality>`



`responseModalities()`







`abstract boolean`



`saveInputBlobsAsArtifacts()`







`abstract @Nullable com.google.genai.types.SpeechConfig`



`speechConfig()`







`abstract RunConfig.StreamingMode`



`streamingMode()`















### Methods inherited from class java.lang. [Object](https://docs.oracle.com/en/java/javase/17/docs/api/java.base/java/lang/Object.html "class or interface in java.lang") [![Link icon](https://google.github.io/adk-docs/api-reference/java/resource-files/link.svg)](https://google.github.io/adk-docs/api-reference/java/com/google/adk/agents/RunConfig.html\#methods-inherited-from-class-java.lang.Object)

`clone, equals, finalize, getClass, hashCode, notify, notifyAll, toString, wait, wait, wait`


- ## Constructor Details [![Link icon](https://google.github.io/adk-docs/api-reference/java/resource-files/link.svg)](https://google.github.io/adk-docs/api-reference/java/com/google/adk/agents/RunConfig.html\#constructor-detail)



- ### RunConfig [![Link icon](https://google.github.io/adk-docs/api-reference/java/resource-files/link.svg)](https://google.github.io/adk-docs/api-reference/java/com/google/adk/agents/RunConfig.html\#%3Cinit%3E())





publicRunConfig()


- ## Method Details [![Link icon](https://google.github.io/adk-docs/api-reference/java/resource-files/link.svg)](https://google.github.io/adk-docs/api-reference/java/com/google/adk/agents/RunConfig.html\#method-detail)



- ### speechConfig [![Link icon](https://google.github.io/adk-docs/api-reference/java/resource-files/link.svg)](https://google.github.io/adk-docs/api-reference/java/com/google/adk/agents/RunConfig.html\#speechConfig())





public abstract@Nullable com.google.genai.types.SpeechConfigspeechConfig()

- ### responseModalities [![Link icon](https://google.github.io/adk-docs/api-reference/java/resource-files/link.svg)](https://google.github.io/adk-docs/api-reference/java/com/google/adk/agents/RunConfig.html\#responseModalities())





public abstractcom.google.common.collect.ImmutableList<com.google.genai.types.Modality>responseModalities()

- ### saveInputBlobsAsArtifacts [![Link icon](https://google.github.io/adk-docs/api-reference/java/resource-files/link.svg)](https://google.github.io/adk-docs/api-reference/java/com/google/adk/agents/RunConfig.html\#saveInputBlobsAsArtifacts())





public abstractbooleansaveInputBlobsAsArtifacts()

- ### streamingMode [![Link icon](https://google.github.io/adk-docs/api-reference/java/resource-files/link.svg)](https://google.github.io/adk-docs/api-reference/java/com/google/adk/agents/RunConfig.html\#streamingMode())





public abstract[RunConfig.StreamingMode](https://google.github.io/adk-docs/api-reference/java/com/google/adk/agents/RunConfig.StreamingMode.html "enum class in com.google.adk.agents")streamingMode()

- ### outputAudioTranscription [![Link icon](https://google.github.io/adk-docs/api-reference/java/resource-files/link.svg)](https://google.github.io/adk-docs/api-reference/java/com/google/adk/agents/RunConfig.html\#outputAudioTranscription())





public abstract@Nullable com.google.genai.types.AudioTranscriptionConfigoutputAudioTranscription()

- ### maxLlmCalls [![Link icon](https://google.github.io/adk-docs/api-reference/java/resource-files/link.svg)](https://google.github.io/adk-docs/api-reference/java/com/google/adk/agents/RunConfig.html\#maxLlmCalls())





public abstractintmaxLlmCalls()

- ### builder [![Link icon](https://google.github.io/adk-docs/api-reference/java/resource-files/link.svg)](https://google.github.io/adk-docs/api-reference/java/com/google/adk/agents/RunConfig.html\#builder())





public static[RunConfig.Builder](https://google.github.io/adk-docs/api-reference/java/com/google/adk/agents/RunConfig.Builder.html "class in com.google.adk.agents")builder()

- ### builder [![Link icon](https://google.github.io/adk-docs/api-reference/java/resource-files/link.svg)](https://google.github.io/adk-docs/api-reference/java/com/google/adk/agents/RunConfig.html\#builder(com.google.adk.agents.RunConfig))





public static[RunConfig.Builder](https://google.github.io/adk-docs/api-reference/java/com/google/adk/agents/RunConfig.Builder.html "class in com.google.adk.agents")builder( [RunConfig](https://google.github.io/adk-docs/api-reference/java/com/google/adk/agents/RunConfig.html "class in com.google.adk.agents") runConfig)

## Python ADK Quickstart
[Skip to content](https://google.github.io/adk-docs/get-started/python/#python-quickstart-for-adk)

[Edit this page](https://github.com/google/adk-docs/edit/main/docs/get-started/python.md "Edit this page") [View source of this page](https://github.com/google/adk-docs/raw/main/docs/get-started/python.md "View source of this page")

# Python Quickstart for ADK [¶](https://google.github.io/adk-docs/get-started/python/\#python-quickstart-for-adk "Permanent link")

This guide shows you how to get up and running with Agent Development Kit
(ADK) for Python. Before you start, make sure you have the following installed:

- Python 3.10 or later
- `pip` for installing packages

## Installation [¶](https://google.github.io/adk-docs/get-started/python/\#installation "Permanent link")

Install ADK by running the following command:

```
pip install google-adk
```

Recommended: create and activate a Python virtual environment

Create a Python virtual environment:

```
python -m venv .venv
```

Activate the Python virtual environment:

[Windows CMD](https://google.github.io/adk-docs/get-started/python/#windows-cmd)[Windows Powershell](https://google.github.io/adk-docs/get-started/python/#windows-powershell)[MacOS / Linux](https://google.github.io/adk-docs/get-started/python/#macos--linux)

```
.venv\Scripts\activate.bat
```

```
.venv\Scripts\Activate.ps1
```

```
source .venv/bin/activate
```

## Create an agent project [¶](https://google.github.io/adk-docs/get-started/python/\#create-an-agent-project "Permanent link")

Run the `adk create` command to start a new agent project.

```
adk create my_agent
```

### Explore the agent project [¶](https://google.github.io/adk-docs/get-started/python/\#explore-the-agent-project "Permanent link")

The created agent project has the following structure, with the `agent.py`
file containing the main control code for the agent.

```
my_agent/
    agent.py      # main agent code
    .env          # API keys or project IDs
    __init__.py
```

## Update your agent project [¶](https://google.github.io/adk-docs/get-started/python/\#update-your-agent-project "Permanent link")

The `agent.py` file contains a `root_agent` definition which is the only
required element of an ADK agent. You can also define tools for the agent to
use. Update the generated `agent.py` code to include a `get_current_time` tool
for use by the agent, as shown in the following code:

```
from google.adk.agents.llm_agent import Agent

# Mock tool implementation
def get_current_time(city: str) -> dict:
    """Returns the current time in a specified city."""
    return {"status": "success", "city": city, "time": "10:30 AM"}

root_agent = Agent(
    model='gemini-3-pro-preview',
    name='root_agent',
    description="Tells the current time in a specified city.",
    instruction="You are a helpful assistant that tells the current time in cities. Use the 'get_current_time' tool for this purpose.",
    tools=[get_current_time],
)
```

### Set your API key [¶](https://google.github.io/adk-docs/get-started/python/\#set-your-api-key "Permanent link")

This project uses the Gemini API, which requires an API key. If you
don't already have Gemini API key, create a key in Google AI Studio on the
[API Keys](https://aistudio.google.com/app/apikey) page.

In a terminal window, write your API key into an `.env` file as an environment variable:

Update: my\_agent/.env

```
echo 'GOOGLE_API_KEY="YOUR_API_KEY"' > .env
```

Using other AI models with ADK

ADK supports the use of many generative AI models. For more
information on configuring other models in ADK agents, see
[Models & Authentication](https://google.github.io/adk-docs/agents/models).

## Run your agent [¶](https://google.github.io/adk-docs/get-started/python/\#run-your-agent "Permanent link")

You can run your ADK agent with an interactive command-line interface using the
`adk run` command or the ADK web user interface provided by the ADK using the
`adk web` command. Both these options allow you to test and interact with your
agent.

### Run with command-line interface [¶](https://google.github.io/adk-docs/get-started/python/\#run-with-command-line-interface "Permanent link")

Run your agent using the `adk run` command-line tool.

```
adk run my_agent
```

![adk-run.png](https://google.github.io/adk-docs/assets/adk-run.png)

### Run with web interface [¶](https://google.github.io/adk-docs/get-started/python/\#run-with-web-interface "Permanent link")

The ADK framework provides web interface you can use to test and interact with
your agent. You can start the web interface using the following command:

```
adk web --port 8000
```

Note

Run this command from the **parent directory** that contains your
`my_agent/` folder. For example, if your agent is inside `agents/my_agent/`,
run `adk web` from the `agents/` directory.

This command starts a web server with a chat interface for your agent. You can
access the web interface at (http://localhost:8000). Select the agent at the
upper left corner and type a request.

![adk-web-dev-ui-chat.png](https://google.github.io/adk-docs/assets/adk-web-dev-ui-chat.png)

## Next: Build your agent [¶](https://google.github.io/adk-docs/get-started/python/\#next-build-your-agent "Permanent link")

Now that you have ADK installed and your first agent running, try building
your own agent with our build guides:

- [Build your agent](https://google.github.io/adk-docs/tutorials/)

Back to top

## Agent Controller Overview
No usage of com.google.adk.web.AdkWebServer.AgentController

## ADK Agent Compiler
[java.lang.Object](https://docs.oracle.com/en/java/javase/17/docs/api/java.base/java/lang/Object.html "class or interface in java.lang")

com.google.adk.web.AgentCompilerLoader

* * *

@Service
public class AgentCompilerLoaderextends [Object](https://docs.oracle.com/en/java/javase/17/docs/api/java.base/java/lang/Object.html "class or interface in java.lang")

Dynamically compiles and loads ADK [`BaseAgent`](https://google.github.io/adk-docs/api-reference/java/com/google/adk/agents/BaseAgent.html "class in com.google.adk.agents") implementations from source files. It
orchestrates the discovery of the ADK core JAR, compilation of agent sources using the Eclipse
JDT (ECJ) compiler, and loading of compiled agents into isolated classloaders. Agents are
identified by a public static field named `ROOT_AGENT`. Supports agent organization in
subdirectories or as individual `.java` files.

- ## Constructor Summary [![Link icon](https://google.github.io/adk-docs/api-reference/java/resource-files/link.svg)](https://google.github.io/adk-docs/api-reference/java/com/google/adk/web/AgentCompilerLoader.html\#constructor-summary)



Constructors





Constructor



Description



`AgentCompilerLoader(AgentLoadingProperties properties)`





Initializes the loader with agent configuration and proactively attempts to locate the ADK core
JAR.

- ## Method Summary [![Link icon](https://google.github.io/adk-docs/api-reference/java/resource-files/link.svg)](https://google.github.io/adk-docs/api-reference/java/com/google/adk/web/AgentCompilerLoader.html\#method-summary)





All MethodsInstance MethodsConcrete Methods







Modifier and Type



Method



Description



`Map<String, BaseAgent>`



`loadAgents()`





Discovers, compiles, and loads agents from the configured source directory.













### Methods inherited from class java.lang. [Object](https://docs.oracle.com/en/java/javase/17/docs/api/java.base/java/lang/Object.html "class or interface in java.lang") [![Link icon](https://google.github.io/adk-docs/api-reference/java/resource-files/link.svg)](https://google.github.io/adk-docs/api-reference/java/com/google/adk/web/AgentCompilerLoader.html\#methods-inherited-from-class-java.lang.Object)

`clone, equals, finalize, getClass, hashCode, notify, notifyAll, toString, wait, wait, wait`


- ## Constructor Details [![Link icon](https://google.github.io/adk-docs/api-reference/java/resource-files/link.svg)](https://google.github.io/adk-docs/api-reference/java/com/google/adk/web/AgentCompilerLoader.html\#constructor-detail)



- ### AgentCompilerLoader [![Link icon](https://google.github.io/adk-docs/api-reference/java/resource-files/link.svg)](https://google.github.io/adk-docs/api-reference/java/com/google/adk/web/AgentCompilerLoader.html\#%3Cinit%3E(com.google.adk.web.config.AgentLoadingProperties))





publicAgentCompilerLoader( [AgentLoadingProperties](https://google.github.io/adk-docs/api-reference/java/com/google/adk/web/config/AgentLoadingProperties.html "class in com.google.adk.web.config") properties)



Initializes the loader with agent configuration and proactively attempts to locate the ADK core
JAR. This JAR, containing [`BaseAgent`](https://google.github.io/adk-docs/api-reference/java/com/google/adk/agents/BaseAgent.html "class in com.google.adk.agents") and other core ADK types, is crucial for agent
compilation. The location strategy (see `locateAndPrepareAdkCoreJar()`) includes
handling directly available JARs and extracting nested JARs (e.g., in Spring Boot fat JARs) to
ensure it's available for the compilation classpath.

Parameters:`properties` \- Configuration detailing agent source locations and compilation settings.


- ## Method Details [![Link icon](https://google.github.io/adk-docs/api-reference/java/resource-files/link.svg)](https://google.github.io/adk-docs/api-reference/java/com/google/adk/web/AgentCompilerLoader.html\#method-detail)



- ### loadAgents [![Link icon](https://google.github.io/adk-docs/api-reference/java/resource-files/link.svg)](https://google.github.io/adk-docs/api-reference/java/com/google/adk/web/AgentCompilerLoader.html\#loadAgents())





public[Map](https://docs.oracle.com/en/java/javase/17/docs/api/java.base/java/util/Map.html "class or interface in java.util") < [String](https://docs.oracle.com/en/java/javase/17/docs/api/java.base/java/lang/String.html "class or interface in java.lang"), [BaseAgent](https://google.github.io/adk-docs/api-reference/java/com/google/adk/agents/BaseAgent.html "class in com.google.adk.agents") >loadAgents()
throws [IOException](https://docs.oracle.com/en/java/javase/17/docs/api/java.base/java/io/IOException.html "class or interface in java.io")



Discovers, compiles, and loads agents from the configured source directory.



The process for each potential "agent unit" (a subdirectory or a root `.java` file):





1. Collects `.java` source files.

2. Compiles these sources using ECJ (see `compileSourcesWithECJ(List, Path)`) into a
    temporary, unit-specific output directory. This directory is cleaned up on JVM exit.

3. Creates a dedicated [`URLClassLoader`](https://docs.oracle.com/en/java/javase/17/docs/api/java.base/java/net/URLClassLoader.html "class or interface in java.net") for the compiled unit, isolating its classes.

4. Scans compiled classes for a public static field `ROOT_AGENT` assignable to [`BaseAgent`](https://google.github.io/adk-docs/api-reference/java/com/google/adk/agents/BaseAgent.html "class in com.google.adk.agents"). This field serves as the designated entry point for an agent.

5. Instantiates and stores the [`BaseAgent`](https://google.github.io/adk-docs/api-reference/java/com/google/adk/agents/BaseAgent.html "class in com.google.adk.agents") if found, keyed by its name.


This approach allows for dynamic addition of agents without pre-compilation and supports
independent classpaths per agent unit if needed (though current implementation uses a shared
parent classloader).

Returns:A map of successfully loaded agent names to their [`BaseAgent`](https://google.github.io/adk-docs/api-reference/java/com/google/adk/agents/BaseAgent.html "class in com.google.adk.agents") instances. Returns
an empty map if the source directory isn't configured or no agents are found.Throws:`IOException` \- If an I/O error occurs (e.g., creating temp directories, reading sources).

## EventActions.Builder Overview
Packages that use [EventActions.Builder](https://google.github.io/adk-docs/api-reference/java/com/google/adk/events/EventActions.Builder.html "class in com.google.adk.events")

Package

Description

[com.google.adk.events](https://google.github.io/adk-docs/api-reference/java/com/google/adk/events/class-use/EventActions.Builder.html#com.google.adk.events)

- ## Uses of [EventActions.Builder](https://google.github.io/adk-docs/api-reference/java/com/google/adk/events/EventActions.Builder.html "class in com.google.adk.events") in [com.google.adk.events](https://google.github.io/adk-docs/api-reference/java/com/google/adk/events/package-summary.html) [![Link icon](https://google.github.io/adk-docs/api-reference/java/resource-files/link.svg)](https://google.github.io/adk-docs/api-reference/java/com/google/adk/events/class-use/EventActions.Builder.html\#com.google.adk.events)



Methods in [com.google.adk.events](https://google.github.io/adk-docs/api-reference/java/com/google/adk/events/package-summary.html) that return [EventActions.Builder](https://google.github.io/adk-docs/api-reference/java/com/google/adk/events/EventActions.Builder.html "class in com.google.adk.events")





Modifier and Type



Method



Description



`EventActions.Builder`



EventActions.Builder.`artifactDelta(ConcurrentMap<String, com.google.genai.types.Part> value)`







`static EventActions.Builder`



EventActions.`builder()`







`EventActions.Builder`



EventActions.Builder.`endInvocation(boolean endInvocation)`







`EventActions.Builder`



EventActions.Builder.`escalate(boolean escalate)`







`EventActions.Builder`



EventActions.Builder.`merge(EventActions other)`







`EventActions.Builder`



EventActions.Builder.`requestedAuthConfigs(ConcurrentMap<String, ConcurrentMap<String,Object>> value)`







`EventActions.Builder`



EventActions.Builder.`skipSummarization(boolean skipSummarization)`







`EventActions.Builder`



EventActions.Builder.`stateDelta(ConcurrentMap<String,Object> value)`







`EventActions.Builder`



EventActions.`toBuilder()`







`EventActions.Builder`



EventActions.Builder.`transferToAgent(String agentId)`

## AdkWebServer AgentRunRequest
Packages that use [AdkWebServer.AgentRunRequest](https://google.github.io/adk-docs/api-reference/java/com/google/adk/web/AdkWebServer.AgentRunRequest.html "class in com.google.adk.web")

Package

Description

[com.google.adk.web](https://google.github.io/adk-docs/api-reference/java/com/google/adk/web/class-use/AdkWebServer.AgentRunRequest.html#com.google.adk.web)

- ## Uses of [AdkWebServer.AgentRunRequest](https://google.github.io/adk-docs/api-reference/java/com/google/adk/web/AdkWebServer.AgentRunRequest.html "class in com.google.adk.web") in [com.google.adk.web](https://google.github.io/adk-docs/api-reference/java/com/google/adk/web/package-summary.html) [![Link icon](https://google.github.io/adk-docs/api-reference/java/resource-files/link.svg)](https://google.github.io/adk-docs/api-reference/java/com/google/adk/web/class-use/AdkWebServer.AgentRunRequest.html\#com.google.adk.web)



Methods in [com.google.adk.web](https://google.github.io/adk-docs/api-reference/java/com/google/adk/web/package-summary.html) with parameters of type [AdkWebServer.AgentRunRequest](https://google.github.io/adk-docs/api-reference/java/com/google/adk/web/AdkWebServer.AgentRunRequest.html "class in com.google.adk.web")





Modifier and Type



Method



Description



`List<Event>`



AdkWebServer.AgentController.`agentRun(AdkWebServer.AgentRunRequest request)`





Executes a non-streaming agent run for a given session and message.





`org.springframework.web.servlet.mvc.method.annotation.SseEmitter`



AdkWebServer.AgentController.`agentRunSse(AdkWebServer.AgentRunRequest request)`





Executes an agent run and streams the resulting events using Server-Sent Events (SSE).

## BaseExampleProvider Usage
Packages that use [BaseExampleProvider](https://google.github.io/adk-docs/api-reference/java/com/google/adk/examples/BaseExampleProvider.html "interface in com.google.adk.examples")

Package

Description

[com.google.adk.agents](https://google.github.io/adk-docs/api-reference/java/com/google/adk/examples/class-use/BaseExampleProvider.html#com.google.adk.agents)

[com.google.adk.examples](https://google.github.io/adk-docs/api-reference/java/com/google/adk/examples/class-use/BaseExampleProvider.html#com.google.adk.examples)

- ## Uses of [BaseExampleProvider](https://google.github.io/adk-docs/api-reference/java/com/google/adk/examples/BaseExampleProvider.html "interface in com.google.adk.examples") in [com.google.adk.agents](https://google.github.io/adk-docs/api-reference/java/com/google/adk/agents/package-summary.html) [![Link icon](https://google.github.io/adk-docs/api-reference/java/resource-files/link.svg)](https://google.github.io/adk-docs/api-reference/java/com/google/adk/examples/class-use/BaseExampleProvider.html\#com.google.adk.agents)



Methods in [com.google.adk.agents](https://google.github.io/adk-docs/api-reference/java/com/google/adk/agents/package-summary.html) that return types with arguments of type [BaseExampleProvider](https://google.github.io/adk-docs/api-reference/java/com/google/adk/examples/BaseExampleProvider.html "interface in com.google.adk.examples")





Modifier and Type



Method



Description



`Optional<BaseExampleProvider>`



LlmAgent.`exampleProvider()`









Methods in [com.google.adk.agents](https://google.github.io/adk-docs/api-reference/java/com/google/adk/agents/package-summary.html) with parameters of type [BaseExampleProvider](https://google.github.io/adk-docs/api-reference/java/com/google/adk/examples/BaseExampleProvider.html "interface in com.google.adk.examples")





Modifier and Type



Method



Description



`LlmAgent.Builder`



LlmAgent.Builder.`exampleProvider(BaseExampleProvider exampleProvider)`

- ## Uses of [BaseExampleProvider](https://google.github.io/adk-docs/api-reference/java/com/google/adk/examples/BaseExampleProvider.html "interface in com.google.adk.examples") in [com.google.adk.examples](https://google.github.io/adk-docs/api-reference/java/com/google/adk/examples/package-summary.html) [![Link icon](https://google.github.io/adk-docs/api-reference/java/resource-files/link.svg)](https://google.github.io/adk-docs/api-reference/java/com/google/adk/examples/class-use/BaseExampleProvider.html\#com.google.adk.examples)



Methods in [com.google.adk.examples](https://google.github.io/adk-docs/api-reference/java/com/google/adk/examples/package-summary.html) with parameters of type [BaseExampleProvider](https://google.github.io/adk-docs/api-reference/java/com/google/adk/examples/BaseExampleProvider.html "interface in com.google.adk.examples")





Modifier and Type



Method



Description



`static String`



ExampleUtils.`buildExampleSi(BaseExampleProvider exampleProvider,
String query)`





Builds a formatted few-shot example string for the given query.

## Annotations Schema
Enclosing class:`Annotations`

* * *

[@Target](https://docs.oracle.com/en/java/javase/17/docs/api/java.base/java/lang/annotation/Target.html "class or interface in java.lang.annotation")({ [METHOD](https://docs.oracle.com/en/java/javase/17/docs/api/java.base/java/lang/annotation/ElementType.html#METHOD "class or interface in java.lang.annotation"), [PARAMETER](https://docs.oracle.com/en/java/javase/17/docs/api/java.base/java/lang/annotation/ElementType.html#PARAMETER "class or interface in java.lang.annotation")})
[@Retention](https://docs.oracle.com/en/java/javase/17/docs/api/java.base/java/lang/annotation/Retention.html "class or interface in java.lang.annotation")( [RUNTIME](https://docs.oracle.com/en/java/javase/17/docs/api/java.base/java/lang/annotation/RetentionPolicy.html#RUNTIME "class or interface in java.lang.annotation"))
public static @interface Annotations.Schema

The annotation for binding the 'Schema' input.

- ## Optional Element Summary [![Link icon](https://google.github.io/adk-docs/api-reference/java/resource-files/link.svg)](https://google.github.io/adk-docs/api-reference/java/com/google/adk/tools/Annotations.Schema.html\#annotation-interface-optional-element-summary)



Optional Elements





Modifier and Type



Optional Element



Description



`String`



`description`







`String`



`name`


- ## Element Details



- ### name [![Link icon](https://google.github.io/adk-docs/api-reference/java/resource-files/link.svg)](https://google.github.io/adk-docs/api-reference/java/com/google/adk/tools/Annotations.Schema.html\#name())





[String](https://docs.oracle.com/en/java/javase/17/docs/api/java.base/java/lang/String.html "class or interface in java.lang")name

Default:`""`

- ### description [![Link icon](https://google.github.io/adk-docs/api-reference/java/resource-files/link.svg)](https://google.github.io/adk-docs/api-reference/java/com/google/adk/tools/Annotations.Schema.html\#description())





[String](https://docs.oracle.com/en/java/javase/17/docs/api/java.base/java/lang/String.html "class or interface in java.lang")description

Default:`""`

## Runner Class Usage
Packages that use [Runner](https://google.github.io/adk-docs/api-reference/java/com/google/adk/runner/Runner.html "class in com.google.adk.runner")

Package

Description

[com.google.adk.runner](https://google.github.io/adk-docs/api-reference/java/com/google/adk/runner/class-use/Runner.html#com.google.adk.runner)

[com.google.adk.web](https://google.github.io/adk-docs/api-reference/java/com/google/adk/runner/class-use/Runner.html#com.google.adk.web)

- ## Uses of [Runner](https://google.github.io/adk-docs/api-reference/java/com/google/adk/runner/Runner.html "class in com.google.adk.runner") in [com.google.adk.runner](https://google.github.io/adk-docs/api-reference/java/com/google/adk/runner/package-summary.html) [![Link icon](https://google.github.io/adk-docs/api-reference/java/resource-files/link.svg)](https://google.github.io/adk-docs/api-reference/java/com/google/adk/runner/class-use/Runner.html\#com.google.adk.runner)



Subclasses of [Runner](https://google.github.io/adk-docs/api-reference/java/com/google/adk/runner/Runner.html "class in com.google.adk.runner") in [com.google.adk.runner](https://google.github.io/adk-docs/api-reference/java/com/google/adk/runner/package-summary.html)





Modifier and Type



Class



Description



`class`



`InMemoryRunner`





The class for the in-memory GenAi runner, using in-memory artifact and session services.

- ## Uses of [Runner](https://google.github.io/adk-docs/api-reference/java/com/google/adk/runner/Runner.html "class in com.google.adk.runner") in [com.google.adk.web](https://google.github.io/adk-docs/api-reference/java/com/google/adk/web/package-summary.html) [![Link icon](https://google.github.io/adk-docs/api-reference/java/resource-files/link.svg)](https://google.github.io/adk-docs/api-reference/java/com/google/adk/runner/class-use/Runner.html\#com.google.adk.web)



Methods in [com.google.adk.web](https://google.github.io/adk-docs/api-reference/java/com/google/adk/web/package-summary.html) that return [Runner](https://google.github.io/adk-docs/api-reference/java/com/google/adk/runner/Runner.html "class in com.google.adk.runner")





Modifier and Type



Method



Description



`Runner`



AdkWebServer.RunnerService.`getRunner(String appName)`





Gets the Runner instance for a given application name.

## LoopAgent.Builder Overview
Packages that use [LoopAgent.Builder](https://google.github.io/adk-docs/api-reference/java/com/google/adk/agents/LoopAgent.Builder.html "class in com.google.adk.agents")

Package

Description

[com.google.adk.agents](https://google.github.io/adk-docs/api-reference/java/com/google/adk/agents/class-use/LoopAgent.Builder.html#com.google.adk.agents)

- ## Uses of [LoopAgent.Builder](https://google.github.io/adk-docs/api-reference/java/com/google/adk/agents/LoopAgent.Builder.html "class in com.google.adk.agents") in [com.google.adk.agents](https://google.github.io/adk-docs/api-reference/java/com/google/adk/agents/package-summary.html) [![Link icon](https://google.github.io/adk-docs/api-reference/java/resource-files/link.svg)](https://google.github.io/adk-docs/api-reference/java/com/google/adk/agents/class-use/LoopAgent.Builder.html\#com.google.adk.agents)



Methods in [com.google.adk.agents](https://google.github.io/adk-docs/api-reference/java/com/google/adk/agents/package-summary.html) that return [LoopAgent.Builder](https://google.github.io/adk-docs/api-reference/java/com/google/adk/agents/LoopAgent.Builder.html "class in com.google.adk.agents")





Modifier and Type



Method



Description



`LoopAgent.Builder`



LoopAgent.Builder.`afterAgentCallback(Callbacks.AfterAgentCallback afterAgentCallback)`







`LoopAgent.Builder`



LoopAgent.Builder.`afterAgentCallback(List<com.google.adk.agents.Callbacks.AfterAgentCallbackBase> afterAgentCallback)`







`LoopAgent.Builder`



LoopAgent.Builder.`beforeAgentCallback(Callbacks.BeforeAgentCallback beforeAgentCallback)`







`LoopAgent.Builder`



LoopAgent.Builder.`beforeAgentCallback(List<com.google.adk.agents.Callbacks.BeforeAgentCallbackBase> beforeAgentCallback)`







`static LoopAgent.Builder`



LoopAgent.`builder()`







`LoopAgent.Builder`



LoopAgent.Builder.`description(String description)`







`LoopAgent.Builder`



LoopAgent.Builder.`maxIterations(int maxIterations)`







`LoopAgent.Builder`



LoopAgent.Builder.`maxIterations(Optional<Integer> maxIterations)`







`LoopAgent.Builder`



LoopAgent.Builder.`name(String name)`







`LoopAgent.Builder`



LoopAgent.Builder.`subAgents(BaseAgent... subAgents)`







`LoopAgent.Builder`



LoopAgent.Builder.`subAgents(List<? extends BaseAgent> subAgents)`

## LLM Instructions Processor
[java.lang.Object](https://docs.oracle.com/en/java/javase/17/docs/api/java.base/java/lang/Object.html "class or interface in java.lang")

com.google.adk.flows.llmflows.Instructions

All Implemented Interfaces:`RequestProcessor`

* * *

public final class Instructionsextends [Object](https://docs.oracle.com/en/java/javase/17/docs/api/java.base/java/lang/Object.html "class or interface in java.lang")
implements [RequestProcessor](https://google.github.io/adk-docs/api-reference/java/com/google/adk/flows/llmflows/RequestProcessor.html "interface in com.google.adk.flows.llmflows")

[`RequestProcessor`](https://google.github.io/adk-docs/api-reference/java/com/google/adk/flows/llmflows/RequestProcessor.html "interface in com.google.adk.flows.llmflows") that handles instructions and global instructions for LLM flows.

- ## Nested Class Summary [![Link icon](https://google.github.io/adk-docs/api-reference/java/resource-files/link.svg)](https://google.github.io/adk-docs/api-reference/java/com/google/adk/flows/llmflows/Instructions.html\#nested-class-summary)





### Nested classes/interfaces inherited from interface com.google.adk.flows.llmflows. [RequestProcessor](https://google.github.io/adk-docs/api-reference/java/com/google/adk/flows/llmflows/RequestProcessor.html "interface in com.google.adk.flows.llmflows") [![Link icon](https://google.github.io/adk-docs/api-reference/java/resource-files/link.svg)](https://google.github.io/adk-docs/api-reference/java/com/google/adk/flows/llmflows/Instructions.html\#nested-classes-inherited-from-class-com.google.adk.flows.llmflows.RequestProcessor)

`RequestProcessor.RequestProcessingResult`

- ## Constructor Summary [![Link icon](https://google.github.io/adk-docs/api-reference/java/resource-files/link.svg)](https://google.github.io/adk-docs/api-reference/java/com/google/adk/flows/llmflows/Instructions.html\#constructor-summary)



Constructors





Constructor



Description



`Instructions()`

- ## Method Summary [![Link icon](https://google.github.io/adk-docs/api-reference/java/resource-files/link.svg)](https://google.github.io/adk-docs/api-reference/java/com/google/adk/flows/llmflows/Instructions.html\#method-summary)





All MethodsInstance MethodsConcrete Methods







Modifier and Type



Method



Description



`io.reactivex.rxjava3.core.Single<RequestProcessor.RequestProcessingResult>`



`processRequest(InvocationContext context,
LlmRequest request)`





Process the LLM request as part of the pre-processing stage.













### Methods inherited from class java.lang. [Object](https://docs.oracle.com/en/java/javase/17/docs/api/java.base/java/lang/Object.html "class or interface in java.lang") [![Link icon](https://google.github.io/adk-docs/api-reference/java/resource-files/link.svg)](https://google.github.io/adk-docs/api-reference/java/com/google/adk/flows/llmflows/Instructions.html\#methods-inherited-from-class-java.lang.Object)

`clone, equals, finalize, getClass, hashCode, notify, notifyAll, toString, wait, wait, wait`


- ## Constructor Details [![Link icon](https://google.github.io/adk-docs/api-reference/java/resource-files/link.svg)](https://google.github.io/adk-docs/api-reference/java/com/google/adk/flows/llmflows/Instructions.html\#constructor-detail)



- ### Instructions [![Link icon](https://google.github.io/adk-docs/api-reference/java/resource-files/link.svg)](https://google.github.io/adk-docs/api-reference/java/com/google/adk/flows/llmflows/Instructions.html\#%3Cinit%3E())





publicInstructions()


- ## Method Details [![Link icon](https://google.github.io/adk-docs/api-reference/java/resource-files/link.svg)](https://google.github.io/adk-docs/api-reference/java/com/google/adk/flows/llmflows/Instructions.html\#method-detail)



- ### processRequest [![Link icon](https://google.github.io/adk-docs/api-reference/java/resource-files/link.svg)](https://google.github.io/adk-docs/api-reference/java/com/google/adk/flows/llmflows/Instructions.html\#processRequest(com.google.adk.agents.InvocationContext,com.google.adk.models.LlmRequest))





publicio.reactivex.rxjava3.core.Single< [RequestProcessor.RequestProcessingResult](https://google.github.io/adk-docs/api-reference/java/com/google/adk/flows/llmflows/RequestProcessor.RequestProcessingResult.html "class in com.google.adk.flows.llmflows") >processRequest( [InvocationContext](https://google.github.io/adk-docs/api-reference/java/com/google/adk/agents/InvocationContext.html "class in com.google.adk.agents") context,
[LlmRequest](https://google.github.io/adk-docs/api-reference/java/com/google/adk/models/LlmRequest.html "class in com.google.adk.models") request)



Description copied from interface: `RequestProcessor`



Process the LLM request as part of the pre-processing stage.

Specified by:`processRequest` in interface `RequestProcessor`Parameters:`context` \- the invocation context.`request` \- the LLM request to process.Returns:a list of events generated during processing (if any).

## ADK CLI Documentation
```
.. adk cli documentation master file, created by
   sphinx-quickstart on Mon Aug 11 11:48:40 2025.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

adk cli documentation
=====================

Add your content using ``reStructuredText`` syntax. See the
`reStructuredText <https://www.sphinx-doc.org/en/master/usage/restructuredtext/index.html>`_
documentation for details.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   cli
```

## Agent Configuration
# AgentConfig

The config for the YAML schema to create an agent.

## Any of

- [LlmAgentConfig](https://google.github.io/adk-docs/api-reference/agentconfig/#tab-pane_anyOf_i0)
- [LoopAgentConfig](https://google.github.io/adk-docs/api-reference/agentconfig/#tab-pane_anyOf_i1)
- [ParallelAgentConfig](https://google.github.io/adk-docs/api-reference/agentconfig/#tab-pane_anyOf_i2)
- [SequentialAgentConfig](https://google.github.io/adk-docs/api-reference/agentconfig/#tab-pane_anyOf_i3)
- [BaseAgentConfig](https://google.github.io/adk-docs/api-reference/agentconfig/#tab-pane_anyOf_i4)

#### LlmAgentConfig

Type: object

The config for the YAML schema of a LlmAgent.
No Additional Properties

## agent\_class

#### Agent Class

Type: enum (of string)Default: "LlmAgent"

The value is used to uniquely identify the LlmAgent class. If it is empty, it is by default an LlmAgent.

#### Must be one of:

- "LlmAgent"
- ""

## nameRequired

#### Name

Type: string

Required. The name of the agent.

## description

#### Description

Type: stringDefault: ""

Optional. The description of the agent.

## sub\_agents

#### Sub Agents

Default: null

Optional. The sub-agents of the agent.

## Any of

- [Option 1](https://google.github.io/adk-docs/api-reference/agentconfig/#tab-pane_anyOf_i0_sub_agents_anyOf_i0)
- [Option 2](https://google.github.io/adk-docs/api-reference/agentconfig/#tab-pane_anyOf_i0_sub_agents_anyOf_i1)

Type: array

No Additional Items

#### Each item of this array must be:

#### AgentRefConfig

Type: object

The config for the reference to another agent.
No Additional Properties

## config\_path

#### Config Path

Default: null

## Any of

- [Option 1](https://google.github.io/adk-docs/api-reference/agentconfig/#tab-pane_anyOf_i0_sub_agents_anyOf_i0_items_config_path_anyOf_i0)
- [Option 2](https://google.github.io/adk-docs/api-reference/agentconfig/#tab-pane_anyOf_i0_sub_agents_anyOf_i0_items_config_path_anyOf_i1)

Type: string

Type: null

## code

#### Code

Default: null

## Any of

- [Option 1](https://google.github.io/adk-docs/api-reference/agentconfig/#tab-pane_anyOf_i0_sub_agents_anyOf_i0_items_code_anyOf_i0)
- [Option 2](https://google.github.io/adk-docs/api-reference/agentconfig/#tab-pane_anyOf_i0_sub_agents_anyOf_i0_items_code_anyOf_i1)

Type: string

Type: null

Type: null

## before\_agent\_callbacks

#### Before Agent Callbacks

Default: null

Optional. The before _agent_ callbacks of the agent.

Example:

```
before_agent_callbacks:
  - name: my_library.security_callbacks.before_agent_callback


```

## Any of

- [Option 1](https://google.github.io/adk-docs/api-reference/agentconfig/#tab-pane_anyOf_i0_before_agent_callbacks_anyOf_i0)
- [Option 2](https://google.github.io/adk-docs/api-reference/agentconfig/#tab-pane_anyOf_i0_before_agent_callbacks_anyOf_i1)

Type: array

No Additional Items

#### Each item of this array must be:

#### CodeConfig

Type: object

Code reference config for a variable, a function, or a class.

This config is used for configuring callbacks and tools.
No Additional Properties

## nameRequired

#### Name

Type: string

## args

#### Args

Default: null

## Any of

- [Option 1](https://google.github.io/adk-docs/api-reference/agentconfig/#tab-pane_anyOf_i0_before_agent_callbacks_anyOf_i0_items_args_anyOf_i0)
- [Option 2](https://google.github.io/adk-docs/api-reference/agentconfig/#tab-pane_anyOf_i0_before_agent_callbacks_anyOf_i0_items_args_anyOf_i1)

Type: array

No Additional Items

#### Each item of this array must be:

#### ArgumentConfig

Type: object

An argument passed to a function or a class's constructor.
No Additional Properties

## name

#### Name

Default: null

## Any of

- [Option 1](https://google.github.io/adk-docs/api-reference/agentconfig/#tab-pane_anyOf_i0_before_agent_callbacks_anyOf_i0_items_args_anyOf_i0_items_name_anyOf_i0)
- [Option 2](https://google.github.io/adk-docs/api-reference/agentconfig/#tab-pane_anyOf_i0_before_agent_callbacks_anyOf_i0_items_args_anyOf_i0_items_name_anyOf_i1)

Type: string

Type: null

## valueRequired

#### Value

Type: object

Type: null

Type: null

## after\_agent\_callbacks

#### After Agent Callbacks

Default: null

Optional. The after _agent_ callbacks of the agent.

## Any of

- [Option 1](https://google.github.io/adk-docs/api-reference/agentconfig/#tab-pane_anyOf_i0_after_agent_callbacks_anyOf_i0)
- [Option 2](https://google.github.io/adk-docs/api-reference/agentconfig/#tab-pane_anyOf_i0_after_agent_callbacks_anyOf_i1)

Type: array

No Additional Items

#### Each item of this array must be:

#### CodeConfig

Type: object

Code reference config for a variable, a function, or a class.

This config is used for configuring callbacks and tools.
[Same definition as CodeConfig](https://google.github.io/adk-docs/api-reference/agentconfig/#anyOf_i0_before_agent_callbacks_anyOf_i0_items)

Type: null

## model

#### Model

Default: null

Optional. LlmAgent.model. If not set, the model will be inherited from the ancestor.

## Any of

- [Option 1](https://google.github.io/adk-docs/api-reference/agentconfig/#tab-pane_anyOf_i0_model_anyOf_i0)
- [Option 2](https://google.github.io/adk-docs/api-reference/agentconfig/#tab-pane_anyOf_i0_model_anyOf_i1)

Type: string

Type: null

## instructionRequired

#### Instruction

Type: string

Required. LlmAgent.instruction.

## disallow\_transfer\_to\_parent

#### Disallow Transfer To Parent

Default: null

Optional. LlmAgent.disallow _transfer_ to\_parent.

## Any of

- [Option 1](https://google.github.io/adk-docs/api-reference/agentconfig/#tab-pane_anyOf_i0_disallow_transfer_to_parent_anyOf_i0)
- [Option 2](https://google.github.io/adk-docs/api-reference/agentconfig/#tab-pane_anyOf_i0_disallow_transfer_to_parent_anyOf_i1)

Type: boolean

Type: null

## disallow\_transfer\_to\_peers

#### Disallow Transfer To Peers

Default: null

Optional. LlmAgent.disallow _transfer_ to\_peers.

## Any of

- [Option 1](https://google.github.io/adk-docs/api-reference/agentconfig/#tab-pane_anyOf_i0_disallow_transfer_to_peers_anyOf_i0)
- [Option 2](https://google.github.io/adk-docs/api-reference/agentconfig/#tab-pane_anyOf_i0_disallow_transfer_to_peers_anyOf_i1)

Type: boolean

Type: null

## input\_schema

Default: null

Optional. LlmAgent.input\_schema.

## Any of

- [CodeConfig](https://google.github.io/adk-docs/api-reference/agentconfig/#tab-pane_anyOf_i0_input_schema_anyOf_i0)
- [Option 2](https://google.github.io/adk-docs/api-reference/agentconfig/#tab-pane_anyOf_i0_input_schema_anyOf_i1)

#### CodeConfig

Type: object

Code reference config for a variable, a function, or a class.

This config is used for configuring callbacks and tools.
[Same definition as CodeConfig](https://google.github.io/adk-docs/api-reference/agentconfig/#anyOf_i0_before_agent_callbacks_anyOf_i0_items)

Type: null

## output\_schema

Default: null

Optional. LlmAgent.output\_schema.

## Any of

- [CodeConfig](https://google.github.io/adk-docs/api-reference/agentconfig/#tab-pane_anyOf_i0_output_schema_anyOf_i0)
- [Option 2](https://google.github.io/adk-docs/api-reference/agentconfig/#tab-pane_anyOf_i0_output_schema_anyOf_i1)

#### CodeConfig

Type: object

Code reference config for a variable, a function, or a class.

This config is used for configuring callbacks and tools.
[Same definition as CodeConfig](https://google.github.io/adk-docs/api-reference/agentconfig/#anyOf_i0_before_agent_callbacks_anyOf_i0_items)

Type: null

## output\_key

#### Output Key

Default: null

Optional. LlmAgent.output\_key.

## Any of

- [Option 1](https://google.github.io/adk-docs/api-reference/agentconfig/#tab-pane_anyOf_i0_output_key_anyOf_i0)
- [Option 2](https://google.github.io/adk-docs/api-reference/agentconfig/#tab-pane_anyOf_i0_output_key_anyOf_i1)

Type: string

Type: null

## include\_contents

#### Include Contents

Type: enum (of string)Default: "default"

Optional. LlmAgent.include\_contents.

#### Must be one of:

- "default"
- "none"

## tools

#### Tools

Default: null

Optional. LlmAgent.tools.

Examples:

For ADK built-in tools in `google.adk.tools` package, they can be referenced

directly with the name:

```
tools:
  - name: google_search
  - name: load_memory


```

For user-defined tools, they can be referenced with fully qualified name:

```
tools:
  - name: my_library.my_tools.my_tool


```

For tools that needs to be created via functions:

```
tools:
  - name: my_library.my_tools.create_tool
    args:
      - name: param1
        value: value1
      - name: param2
        value: value2


```

For more advanced tools, instead of specifying arguments in config, it's

recommended to define them in Python files and reference them. E.g.,

```
# tools.py
my_mcp_toolset = MCPToolset(
    connection_params=StdioServerParameters(
        command="npx",
        args=["-y", "@notionhq/notion-mcp-server"],
        env={"OPENAPI_MCP_HEADERS": NOTION_HEADERS},
    )
)


```

Then, reference the toolset in config:

```
tools:
  - name: tools.my_mcp_toolset


```

## Any of

- [Option 1](https://google.github.io/adk-docs/api-reference/agentconfig/#tab-pane_anyOf_i0_tools_anyOf_i0)
- [Option 2](https://google.github.io/adk-docs/api-reference/agentconfig/#tab-pane_anyOf_i0_tools_anyOf_i1)

Type: array

No Additional Items

#### Each item of this array must be:

#### ToolConfig

Type: object

The configuration for a tool.

The config supports these types of tools:

1\. ADK built-in tools

2\. User-defined tool instances

3\. User-defined tool classes

4\. User-defined functions that generate tool instances

5\. User-defined function tools

For examples:

1. For ADK built-in tool instances or classes in `google.adk.tools` package,


they can be referenced directly with the `name` and optionally with

`args`.


```
tools:
  - name: google_search
  - name: AgentTool
    args:
      agent: ./another_agent.yaml
      skip_summarization: true
```

2. For user-defined tool instances, the `name` is the fully qualified path


to the tool instance.


```
tools:
  - name: my_package.my_module.my_tool
```

3. For user-defined tool classes (custom tools), the `name` is the fully


qualified path to the tool class and `args` is the arguments for the tool.


```
tools:
  - name: my_package.my_module.my_tool_class
    args:
      my_tool_arg1: value1
      my_tool_arg2: value2
```

4. For user-defined functions that generate tool instances, the `name` is


the fully qualified path to the function and `args` is passed to the


function as arguments.


```
tools:
  - name: my_package.my_module.my_tool_function
    args:
      my_function_arg1: value1
      my_function_arg2: value2
```

The function must have the following signature:

```
def my_function(args: ToolArgsConfig) -> BaseTool:
...
```

5. For user-defined function tools, the `name` is the fully qualified path


to the function.


```
tools:
  - name: my_package.my_module.my_function_tool
```

If the above use cases don't suffice, users can define a custom tool config

by extending BaseToolConfig and override from\_config() in the custom tool.

No Additional Properties

## nameRequired

#### Name

Type: string

The name of the tool.

For ADK built-in tools, `name` is the name of the tool, e.g. `google_search`

or `AgentTool`.

For user-defined tools, the name is the fully qualified path to the tool, e.g.

`my_package.my_module.my_tool`.

## args

Default: null

The args for the tool.

## Any of

- [ToolArgsConfig](https://google.github.io/adk-docs/api-reference/agentconfig/#tab-pane_anyOf_i0_tools_anyOf_i0_items_args_anyOf_i0)
- [Option 2](https://google.github.io/adk-docs/api-reference/agentconfig/#tab-pane_anyOf_i0_tools_anyOf_i0_items_args_anyOf_i1)

#### ToolArgsConfig

Type: object

Config to host free key-value pairs for the args in ToolConfig.

## _Additional Properties_

Additional Properties of any type are allowed.

Type: object

Type: null

Type: null

## before\_model\_callbacks

#### Before Model Callbacks

Default: null

Optional. LlmAgent.before _model_ callbacks.

Example:

```
before_model_callbacks:
  - name: my_library.callbacks.before_model_callback


```

## Any of

- [Option 1](https://google.github.io/adk-docs/api-reference/agentconfig/#tab-pane_anyOf_i0_before_model_callbacks_anyOf_i0)
- [Option 2](https://google.github.io/adk-docs/api-reference/agentconfig/#tab-pane_anyOf_i0_before_model_callbacks_anyOf_i1)

Type: array

No Additional Items

#### Each item of this array must be:

#### CodeConfig

Type: object

Code reference config for a variable, a function, or a class.

This config is used for configuring callbacks and tools.
[Same definition as CodeConfig](https://google.github.io/adk-docs/api-reference/agentconfig/#anyOf_i0_before_agent_callbacks_anyOf_i0_items)

Type: null

## after\_model\_callbacks

#### After Model Callbacks

Default: null

Optional. LlmAgent.after _model_ callbacks.

## Any of

- [Option 1](https://google.github.io/adk-docs/api-reference/agentconfig/#tab-pane_anyOf_i0_after_model_callbacks_anyOf_i0)
- [Option 2](https://google.github.io/adk-docs/api-reference/agentconfig/#tab-pane_anyOf_i0_after_model_callbacks_anyOf_i1)

Type: array

No Additional Items

#### Each item of this array must be:

#### CodeConfig

Type: object

Code reference config for a variable, a function, or a class.

This config is used for configuring callbacks and tools.
[Same definition as CodeConfig](https://google.github.io/adk-docs/api-reference/agentconfig/#anyOf_i0_before_agent_callbacks_anyOf_i0_items)

Type: null

## before\_tool\_callbacks

#### Before Tool Callbacks

Default: null

Optional. LlmAgent.before _tool_ callbacks.

## Any of

- [Option 1](https://google.github.io/adk-docs/api-reference/agentconfig/#tab-pane_anyOf_i0_before_tool_callbacks_anyOf_i0)
- [Option 2](https://google.github.io/adk-docs/api-reference/agentconfig/#tab-pane_anyOf_i0_before_tool_callbacks_anyOf_i1)

Type: array

No Additional Items

#### Each item of this array must be:

#### CodeConfig

Type: object

Code reference config for a variable, a function, or a class.

This config is used for configuring callbacks and tools.
[Same definition as CodeConfig](https://google.github.io/adk-docs/api-reference/agentconfig/#anyOf_i0_before_agent_callbacks_anyOf_i0_items)

Type: null

## after\_tool\_callbacks

#### After Tool Callbacks

Default: null

Optional. LlmAgent.after _tool_ callbacks.

## Any of

- [Option 1](https://google.github.io/adk-docs/api-reference/agentconfig/#tab-pane_anyOf_i0_after_tool_callbacks_anyOf_i0)
- [Option 2](https://google.github.io/adk-docs/api-reference/agentconfig/#tab-pane_anyOf_i0_after_tool_callbacks_anyOf_i1)

Type: array

No Additional Items

#### Each item of this array must be:

#### CodeConfig

Type: object

Code reference config for a variable, a function, or a class.

This config is used for configuring callbacks and tools.
[Same definition as CodeConfig](https://google.github.io/adk-docs/api-reference/agentconfig/#anyOf_i0_before_agent_callbacks_anyOf_i0_items)

Type: null

## generate\_content\_config

Default: null

Optional. LlmAgent.generate _content_ config.

## Any of

- [GenerateContentConfig](https://google.github.io/adk-docs/api-reference/agentconfig/#tab-pane_anyOf_i0_generate_content_config_anyOf_i0)
- [Option 2](https://google.github.io/adk-docs/api-reference/agentconfig/#tab-pane_anyOf_i0_generate_content_config_anyOf_i1)

#### GenerateContentConfig

Type: object

Optional model configuration parameters.

For more information, see `Content generation parameters
<https://cloud.google.com/vertex-ai/generative-ai/docs/multimodal/content-generation-parameters>`\_.
No Additional Properties

## httpOptions

Default: null

Used to override HTTP request options.

## Any of

- [HttpOptions](https://google.github.io/adk-docs/api-reference/agentconfig/#tab-pane_anyOf_i0_generate_content_config_anyOf_i0_httpOptions_anyOf_i0)
- [Option 2](https://google.github.io/adk-docs/api-reference/agentconfig/#tab-pane_anyOf_i0_generate_content_config_anyOf_i0_httpOptions_anyOf_i1)

#### HttpOptions

Type: object

HTTP options to be used in each of the requests.
No Additional Properties

## baseUrl

#### Baseurl

Default: null

The base URL for the AI platform service endpoint.

## Any of

- [Option 1](https://google.github.io/adk-docs/api-reference/agentconfig/#tab-pane_anyOf_i0_generate_content_config_anyOf_i0_httpOptions_anyOf_i0_baseUrl_anyOf_i0)
- [Option 2](https://google.github.io/adk-docs/api-reference/agentconfig/#tab-pane_anyOf_i0_generate_content_config_anyOf_i0_httpOptions_anyOf_i0_baseUrl_anyOf_i1)

Type: string

Type: null

## apiVersion

#### Apiversion

Default: null

Specifies the version of the API to use.

## Any of

- [Option 1](https://google.github.io/adk-docs/api-reference/agentconfig/#tab-pane_anyOf_i0_generate_content_config_anyOf_i0_httpOptions_anyOf_i0_apiVersion_anyOf_i0)
- [Option 2](https://google.github.io/adk-docs/api-reference/agentconfig/#tab-pane_anyOf_i0_generate_content_config_anyOf_i0_httpOptions_anyOf_i0_apiVersion_anyOf_i1)

Type: string

Type: null

## headers

#### Headers

Default: null

Additional HTTP headers to be sent with the request.

## Any of

- [Option 1](https://google.github.io/adk-docs/api-reference/agentconfig/#tab-pane_anyOf_i0_generate_content_config_anyOf_i0_httpOptions_anyOf_i0_headers_anyOf_i0)
- [Option 2](https://google.github.io/adk-docs/api-reference/agentconfig/#tab-pane_anyOf_i0_generate_content_config_anyOf_i0_httpOptions_anyOf_i0_headers_anyOf_i1)

Type: object

## _Additional Properties_

Each additional property must conform to the following schema

Type: string

Type: null

## timeout

#### Timeout

Default: null

Timeout for the request in milliseconds.

## Any of

- [Option 1](https://google.github.io/adk-docs/api-reference/agentconfig/#tab-pane_anyOf_i0_generate_content_config_anyOf_i0_httpOptions_anyOf_i0_timeout_anyOf_i0)
- [Option 2](https://google.github.io/adk-docs/api-reference/agentconfig/#tab-pane_anyOf_i0_generate_content_config_anyOf_i0_httpOptions_anyOf_i0_timeout_anyOf_i1)

Type: integer

Type: null

## clientArgs

#### Clientargs

Default: null

Args passed to the HTTP client.

## Any of

- [Option 1](https://google.github.io/adk-docs/api-reference/agentconfig/#tab-pane_anyOf_i0_generate_content_config_anyOf_i0_httpOptions_anyOf_i0_clientArgs_anyOf_i0)
- [Option 2](https://google.github.io/adk-docs/api-reference/agentconfig/#tab-pane_anyOf_i0_generate_content_config_anyOf_i0_httpOptions_anyOf_i0_clientArgs_anyOf_i1)

Type: object

## _Additional Properties_

Additional Properties of any type are allowed.

Type: object

Type: null

## asyncClientArgs

#### Asyncclientargs

Default: null

Args passed to the async HTTP client.

## Any of

- [Option 1](https://google.github.io/adk-docs/api-reference/agentconfig/#tab-pane_anyOf_i0_generate_content_config_anyOf_i0_httpOptions_anyOf_i0_asyncClientArgs_anyOf_i0)
- [Option 2](https://google.github.io/adk-docs/api-reference/agentconfig/#tab-pane_anyOf_i0_generate_content_config_anyOf_i0_httpOptions_anyOf_i0_asyncClientArgs_anyOf_i1)

Type: object

## _Additional Properties_

Additional Properties of any type are allowed.

Type: object

Type: null

## extraBody

#### Extrabody

Default: null

Extra parameters to add to the request body.

The structure must match the backend API's request structure.

\- VertexAI backend API docs: https://cloud.google.com/vertex-ai/docs/reference/rest

\- GeminiAPI backend API docs: https://ai.google.dev/api/rest

## Any of

- [Option 1](https://google.github.io/adk-docs/api-reference/agentconfig/#tab-pane_anyOf_i0_generate_content_config_anyOf_i0_httpOptions_anyOf_i0_extraBody_anyOf_i0)
- [Option 2](https://google.github.io/adk-docs/api-reference/agentconfig/#tab-pane_anyOf_i0_generate_content_config_anyOf_i0_httpOptions_anyOf_i0_extraBody_anyOf_i1)

Type: object

## _Additional Properties_

Additional Properties of any type are allowed.

Type: object

Type: null

## retryOptions

Default: null

HTTP retry options for the request.

## Any of

- [HttpRetryOptions](https://google.github.io/adk-docs/api-reference/agentconfig/#tab-pane_anyOf_i0_generate_content_config_anyOf_i0_httpOptions_anyOf_i0_retryOptions_anyOf_i0)
- [Option 2](https://google.github.io/adk-docs/api-reference/agentconfig/#tab-pane_anyOf_i0_generate_content_config_anyOf_i0_httpOptions_anyOf_i0_retryOptions_anyOf_i1)

#### HttpRetryOptions

Type: object

HTTP retry options to be used in each of the requests.
No Additional Properties

## attempts

#### Attempts

Default: null

Maximum number of attempts, including the original request.

If 0 or 1, it means no retries.

## Any of

- [Option 1](https://google.github.io/adk-docs/api-reference/agentconfig/#tab-pane_anyOf_i0_generate_content_config_anyOf_i0_httpOptions_anyOf_i0_retryOptions_anyOf_i0_attempts_anyOf_i0)
- [Option 2](https://google.github.io/adk-docs/api-reference/agentconfig/#tab-pane_anyOf_i0_generate_content_config_anyOf_i0_httpOptions_anyOf_i0_retryOptions_anyOf_i0_attempts_anyOf_i1)

Type: integer

Type: null

## initialDelay

#### Initialdelay

Default: null

Initial delay before the first retry, in fractions of a second.

## Any of

- [Option 1](https://google.github.io/adk-docs/api-reference/agentconfig/#tab-pane_anyOf_i0_generate_content_config_anyOf_i0_httpOptions_anyOf_i0_retryOptions_anyOf_i0_initialDelay_anyOf_i0)
- [Option 2](https://google.github.io/adk-docs/api-reference/agentconfig/#tab-pane_anyOf_i0_generate_content_config_anyOf_i0_httpOptions_anyOf_i0_retryOptions_anyOf_i0_initialDelay_anyOf_i1)

Type: number

Type: null

## maxDelay

#### Maxdelay

Default: null

Maximum delay between retries, in fractions of a second.

## Any of

- [Option 1](https://google.github.io/adk-docs/api-reference/agentconfig/#tab-pane_anyOf_i0_generate_content_config_anyOf_i0_httpOptions_anyOf_i0_retryOptions_anyOf_i0_maxDelay_anyOf_i0)
- [Option 2](https://google.github.io/adk-docs/api-reference/agentconfig/#tab-pane_anyOf_i0_generate_content_config_anyOf_i0_httpOptions_anyOf_i0_retryOptions_anyOf_i0_maxDelay_anyOf_i1)

Type: number

Type: null

## expBase

#### Expbase

Default: null

Multiplier by which the delay increases after each attempt.

## Any of

- [Option 1](https://google.github.io/adk-docs/api-reference/agentconfig/#tab-pane_anyOf_i0_generate_content_config_anyOf_i0_httpOptions_anyOf_i0_retryOptions_anyOf_i0_expBase_anyOf_i0)
- [Option 2](https://google.github.io/adk-docs/api-reference/agentconfig/#tab-pane_anyOf_i0_generate_content_config_anyOf_i0_httpOptions_anyOf_i0_retryOptions_anyOf_i0_expBase_anyOf_i1)

Type: number

Type: null

## jitter

#### Jitter

Default: null

Randomness factor for the delay.

## Any of

- [Option 1](https://google.github.io/adk-docs/api-reference/agentconfig/#tab-pane_anyOf_i0_generate_content_config_anyOf_i0_httpOptions_anyOf_i0_retryOptions_anyOf_i0_jitter_anyOf_i0)
- [Option 2](https://google.github.io/adk-docs/api-reference/agentconfig/#tab-pane_anyOf_i0_generate_content_config_anyOf_i0_httpOptions_anyOf_i0_retryOptions_anyOf_i0_jitter_anyOf_i1)

Type: number

Type: null

## httpStatusCodes

#### Httpstatuscodes

Default: null

List of HTTP status codes that should trigger a retry.

If not specified, a default set of retryable codes may be used.

## Any of

- [Option 1](https://google.github.io/adk-docs/api-reference/agentconfig/#tab-pane_anyOf_i0_generate_content_config_anyOf_i0_httpOptions_anyOf_i0_retryOptions_anyOf_i0_httpStatusCodes_anyOf_i0)
- [Option 2](https://google.github.io/adk-docs/api-reference/agentconfig/#tab-pane_anyOf_i0_generate_content_config_anyOf_i0_httpOptions_anyOf_i0_retryOptions_anyOf_i0_httpStatusCodes_anyOf_i1)

Type: array of integer

No Additional Items

#### Each item of this array must be:

Type: integer

Type: null

Type: null

Type: null

## systemInstruction

#### Systeminstruction

Default: null

Instructions for the model to steer it toward better performance.

For example, "Answer as concisely as possible" or "Don't use technical

terms in your response".

## Any of

- [Content](https://google.github.io/adk-docs/api-reference/agentconfig/#tab-pane_anyOf_i0_generate_content_config_anyOf_i0_systemInstruction_anyOf_i0)
- [Option 2](https://google.github.io/adk-docs/api-reference/agentconfig/#tab-pane_anyOf_i0_generate_content_config_anyOf_i0_systemInstruction_anyOf_i1)
- [File](https://google.github.io/adk-docs/api-reference/agentconfig/#tab-pane_anyOf_i0_generate_content_config_anyOf_i0_systemInstruction_anyOf_i2)
- [Part](https://google.github.io/adk-docs/api-reference/agentconfig/#tab-pane_anyOf_i0_generate_content_config_anyOf_i0_systemInstruction_anyOf_i3)
- [Option 5](https://google.github.io/adk-docs/api-reference/agentconfig/#tab-pane_anyOf_i0_generate_content_config_anyOf_i0_systemInstruction_anyOf_i4)
- [Option 6](https://google.github.io/adk-docs/api-reference/agentconfig/#tab-pane_anyOf_i0_generate_content_config_anyOf_i0_systemInstruction_anyOf_i5)

#### Content

Type: object

Contains the multi-part content of a message.
No Additional Properties

## parts

#### Parts

Default: null

List of parts that constitute a single message. Each part may have

a different IANA MIME type.

## Any of

- [Option 1](https://google.github.io/adk-docs/api-reference/agentconfig/#tab-pane_anyOf_i0_generate_content_config_anyOf_i0_systemInstruction_anyOf_i0_parts_anyOf_i0)
- [Option 2](https://google.github.io/adk-docs/api-reference/agentconfig/#tab-pane_anyOf_i0_generate_content_config_anyOf_i0_systemInstruction_anyOf_i0_parts_anyOf_i1)

Type: array

No Additional Items

#### Each item of this array must be:

#### Part

Type: object

A datatype containing media content.

Exactly one field within a Part should be set, representing the specific type

of content being conveyed. Using multiple fields within the same `Part`

instance is considered invalid.

No Additional Properties

## videoMetadata

Default: null

Metadata for a given video.

## Any of

- [VideoMetadata](https://google.github.io/adk-docs/api-reference/agentconfig/#tab-pane_anyOf_i0_generate_content_config_anyOf_i0_systemInstruction_anyOf_i0_parts_anyOf_i0_items_videoMetadata_anyOf_i0)
- [Option 2](https://google.github.io/adk-docs/api-reference/agentconfig/#tab-pane_anyOf_i0_generate_content_config_anyOf_i0_systemInstruction_anyOf_i0_parts_anyOf_i0_items_videoMetadata_anyOf_i1)

#### VideoMetadata

Type: object

Describes how the video in the Part should be used by the model.
No Additional Properties

## fps

#### Fps

Default: null

The frame rate of the video sent to the model. If not specified, the

default value will be 1.0. The fps range is (0.0, 24.0\].

## Any of

- [Option 1](https://google.github.io/adk-docs/api-reference/agentconfig/#tab-pane_anyOf_i0_generate_content_config_anyOf_i0_systemInstruction_anyOf_i0_parts_anyOf_i0_items_videoMetadata_anyOf_i0_fps_anyOf_i0)
- [Option 2](https://google.github.io/adk-docs/api-reference/agentconfig/#tab-pane_anyOf_i0_generate_content_config_anyOf_i0_systemInstruction_anyOf_i0_parts_anyOf_i0_items_videoMetadata_anyOf_i0_fps_anyOf_i1)

Type: number

Type: null

## endOffset

#### Endoffset

Default: null

Optional. The end offset of the video.

## Any of

- [Option 1](https://google.github.io/adk-docs/api-reference/agentconfig/#tab-pane_anyOf_i0_generate_content_config_anyOf_i0_systemInstruction_anyOf_i0_parts_anyOf_i0_items_videoMetadata_anyOf_i0_endOffset_anyOf_i0)
- [Option 2](https://google.github.io/adk-docs/api-reference/agentconfig/#tab-pane_anyOf_i0_generate_content_config_anyOf_i0_systemInstruction_anyOf_i0_parts_anyOf_i0_items_videoMetadata_anyOf_i0_endOffset_anyOf_i1)

Type: string

Type: null

## startOffset

#### Startoffset

Default: null

Optional. The start offset of the video.

## Any of

- [Option 1](https://google.github.io/adk-docs/api-reference/agentconfig/#tab-pane_anyOf_i0_generate_content_config_anyOf_i0_systemInstruction_anyOf_i0_parts_anyOf_i0_items_videoMetadata_anyOf_i0_startOffset_anyOf_i0)
- [Option 2](https://google.github.io/adk-docs/api-reference/agentconfig/#tab-pane_anyOf_i0_generate_content_config_anyOf_i0_systemInstruction_anyOf_i0_parts_anyOf_i0_items_videoMetadata_anyOf_i0_startOffset_anyOf_i1)

Type: string

Type: null

Type: null

## thought

#### Thought

Default: null

Indicates if the part is thought from the model.

## Any of

- [Option 1](https://google.github.io/adk-docs/api-reference/agentconfig/#tab-pane_anyOf_i0_generate_content_config_anyOf_i0_systemInstruction_anyOf_i0_parts_anyOf_i0_items_thought_anyOf_i0)
- [Option 2](https://google.github.io/adk-docs/api-reference/agentconfig/#tab-pane_anyOf_i0_generate_content_config_anyOf_i0_systemInstruction_anyOf_i0_parts_anyOf_i0_items_thought_anyOf_i1)

Type: boolean

Type: null

## inlineData

Default: null

Optional. Inlined bytes data.

## Any of

- [Blob](https://google.github.io/adk-docs/api-reference/agentconfig/#tab-pane_anyOf_i0_generate_content_config_anyOf_i0_systemInstruction_anyOf_i0_parts_anyOf_i0_items_inlineData_anyOf_i0)
- [Option 2](https://google.github.io/adk-docs/api-reference/agentconfig/#tab-pane_anyOf_i0_generate_content_config_anyOf_i0_systemInstruction_anyOf_i0_parts_anyOf_i0_items_inlineData_anyOf_i1)

#### Blob

Type: object

Content blob.
No Additional Properties

## displayName

#### Displayname

Default: null

Optional. Display name of the blob. Used to provide a label or filename to distinguish blobs. This field is not currently used in the Gemini GenerateContent calls.

## Any of

- [Option 1](https://google.github.io/adk-docs/api-reference/agentconfig/#tab-pane_anyOf_i0_generate_content_config_anyOf_i0_systemInstruction_anyOf_i0_parts_anyOf_i0_items_inlineData_anyOf_i0_displayName_anyOf_i0)
- [Option 2](https://google.github.io/adk-docs/api-reference/agentconfig/#tab-pane_anyOf_i0_generate_content_config_anyOf_i0_systemInstruction_anyOf_i0_parts_anyOf_i0_items_inlineData_anyOf_i0_displayName_anyOf_i1)

Type: string

Type: null

## data

#### Data

Default: null

Required. Raw bytes.

## Any of

- [Option 1](https://google.github.io/adk-docs/api-reference/agentconfig/#tab-pane_anyOf_i0_generate_content_config_anyOf_i0_systemInstruction_anyOf_i0_parts_anyOf_i0_items_inlineData_anyOf_i0_data_anyOf_i0)
- [Option 2](https://google.github.io/adk-docs/api-reference/agentconfig/#tab-pane_anyOf_i0_generate_content_config_anyOf_i0_systemInstruction_anyOf_i0_parts_anyOf_i0_items_inlineData_anyOf_i0_data_anyOf_i1)

Type: stringFormat: base64url

Type: null

## mimeType

#### Mimetype

Default: null

Required. The IANA standard MIME type of the source data.

## Any of

- [Option 1](https://google.github.io/adk-docs/api-reference/agentconfig/#tab-pane_anyOf_i0_generate_content_config_anyOf_i0_systemInstruction_anyOf_i0_parts_anyOf_i0_items_inlineData_anyOf_i0_mimeType_anyOf_i0)
- [Option 2](https://google.github.io/adk-docs/api-reference/agentconfig/#tab-pane_anyOf_i0_generate_content_config_anyOf_i0_systemInstruction_anyOf_i0_parts_anyOf_i0_items_inlineData_anyOf_i0_mimeType_anyOf_i1)

Type: string

Type: null

Type: null

## fileData

Default: null

Optional. URI based data.

## Any of

- [FileData](https://google.github.io/adk-docs/api-reference/agentconfig/#tab-pane_anyOf_i0_generate_content_config_anyOf_i0_systemInstruction_anyOf_i0_parts_anyOf_i0_items_fileData_anyOf_i0)
- [Option 2](https://google.github.io/adk-docs/api-reference/agentconfig/#tab-pane_anyOf_i0_generate_content_config_anyOf_i0_systemInstruction_anyOf_i0_parts_anyOf_i0_items_fileData_anyOf_i1)

#### FileData

Type: object

URI based data.
No Additional Properties

## displayName

#### Displayname

Default: null

Optional. Display name of the file data. Used to provide a label or filename to distinguish file datas. It is not currently used in the Gemini GenerateContent calls.

## Any of

- [Option 1](https://google.github.io/adk-docs/api-reference/agentconfig/#tab-pane_anyOf_i0_generate_content_config_anyOf_i0_systemInstruction_anyOf_i0_parts_anyOf_i0_items_fileData_anyOf_i0_displayName_anyOf_i0)
- [Option 2](https://google.github.io/adk-docs/api-reference/agentconfig/#tab-pane_anyOf_i0_generate_content_config_anyOf_i0_systemInstruction_anyOf_i0_parts_anyOf_i0_items_fileData_anyOf_i0_displayName_anyOf_i1)

Type: string

Type: null

## fileUri

#### Fileuri

Default: null

Required. URI.

## Any of

- [Option 1](https://google.github.io/adk-docs/api-reference/agentconfig/#tab-pane_anyOf_i0_generate_content_config_anyOf_i0_systemInstruction_anyOf_i0_parts_anyOf_i0_items_fileData_anyOf_i0_fileUri_anyOf_i0)
- [Option 2](https://google.github.io/adk-docs/api-reference/agentconfig/#tab-pane_anyOf_i0_generate_content_config_anyOf_i0_systemInstruction_anyOf_i0_parts_anyOf_i0_items_fileData_anyOf_i0_fileUri_anyOf_i1)

Type: string

Type: null

## mimeType

#### Mimetype

Default: null

Required. The IANA standard MIME type of the source data.

## Any of

- [Option 1](https://google.github.io/adk-docs/api-reference/agentconfig/#tab-pane_anyOf_i0_generate_content_config_anyOf_i0_systemInstruction_anyOf_i0_parts_anyOf_i0_items_fileData_anyOf_i0_mimeType_anyOf_i0)
- [Option 2](https://google.github.io/adk-docs/api-reference/agentconfig/#tab-pane_anyOf_i0_generate_content_config_anyOf_i0_systemInstruction_anyOf_i0_parts_anyOf_i0_items_fileData_anyOf_i0_mimeType_anyOf_i1)

Type: string

Type: null

Type: null

## thoughtSignature

#### Thoughtsignature

Default: null

An opaque signature for the thought so it can be reused in subsequent requests.

## Any of

- [Option 1](https://google.github.io/adk-docs/api-reference/agentconfig/#tab-pane_anyOf_i0_generate_content_config_anyOf_i0_systemInstruction_anyOf_i0_parts_anyOf_i0_items_thoughtSignature_anyOf_i0)
- [Option 2](https://google.github.io/adk-docs/api-reference/agentconfig/#tab-pane_anyOf_i0_generate_content_config_anyOf_i0_systemInstruction_anyOf_i0_parts_anyOf_i0_items_thoughtSignature_anyOf_i1)

Type: stringFormat: base64url

Type: null

## codeExecutionResult

Default: null

Optional. Result of executing the \[ExecutableCode\].

## Any of

- [CodeExecutionResult](https://google.github.io/adk-docs/api-reference/agentconfig/#tab-pane_anyOf_i0_generate_content_config_anyOf_i0_systemInstruction_anyOf_i0_parts_anyOf_i0_items_codeExecutionResult_anyOf_i0)
- [Option 2](https://google.github.io/adk-docs/api-reference/agentconfig/#tab-pane_anyOf_i0_generate_content_config_anyOf_i0_systemInstruction_anyOf_i0_parts_anyOf_i0_items_codeExecutionResult_anyOf_i1)

#### CodeExecutionResult

Type: object

Result of executing the \[ExecutableCode\].

Only generated when using the \[CodeExecution\] tool, and always follows a

`part` containing the \[ExecutableCode\].
No Additional Properties

## outcome

Default: null

Required. Outcome of the code execution.

## Any of

- [Outcome](https://google.github.io/adk-docs/api-reference/agentconfig/#tab-pane_anyOf_i0_generate_content_config_anyOf_i0_systemInstruction_anyOf_i0_parts_anyOf_i0_items_codeExecutionResult_anyOf_i0_outcome_anyOf_i0)
- [Option 2](https://google.github.io/adk-docs/api-reference/agentconfig/#tab-pane_anyOf_i0_generate_content_config_anyOf_i0_systemInstruction_anyOf_i0_parts_anyOf_i0_items_codeExecutionResult_anyOf_i0_outcome_anyOf_i1)

#### Outcome

Type: enum (of string)

Required. Outcome of the code execution.

#### Must be one of:

- "OUTCOME\_UNSPECIFIED"
- "OUTCOME\_OK"
- "OUTCOME\_FAILED"
- "OUTCOME\_DEADLINE\_EXCEEDED"

Type: null

## output

#### Output

Default: null

Optional. Contains stdout when code execution is successful, stderr or other description otherwise.

## Any of

- [Option 1](https://google.github.io/adk-docs/api-reference/agentconfig/#tab-pane_anyOf_i0_generate_content_config_anyOf_i0_systemInstruction_anyOf_i0_parts_anyOf_i0_items_codeExecutionResult_anyOf_i0_output_anyOf_i0)
- [Option 2](https://google.github.io/adk-docs/api-reference/agentconfig/#tab-pane_anyOf_i0_generate_content_config_anyOf_i0_systemInstruction_anyOf_i0_parts_anyOf_i0_items_codeExecutionResult_anyOf_i0_output_anyOf_i1)

Type: string

Type: null

Type: null

## executableCode

Default: null

Optional. Code generated by the model that is meant to be executed.

## Any of

- [ExecutableCode](https://google.github.io/adk-docs/api-reference/agentconfig/#tab-pane_anyOf_i0_generate_content_config_anyOf_i0_systemInstruction_anyOf_i0_parts_anyOf_i0_items_executableCode_anyOf_i0)
- [Option 2](https://google.github.io/adk-docs/api-reference/agentconfig/#tab-pane_anyOf_i0_generate_content_config_anyOf_i0_systemInstruction_anyOf_i0_parts_anyOf_i0_items_executableCode_anyOf_i1)

#### ExecutableCode

Type: object

Code generated by the model that is meant to be executed, and the result returned to the model.

Generated when using the \[CodeExecution\] tool, in which the code will be

automatically executed, and a corresponding \[CodeExecutionResult\] will also be

generated.

No Additional Properties

## code

#### Code

Default: null

Required. The code to be executed.

## Any of

- [Option 1](https://google.github.io/adk-docs/api-reference/agentconfig/#tab-pane_anyOf_i0_generate_content_config_anyOf_i0_systemInstruction_anyOf_i0_parts_anyOf_i0_items_executableCode_anyOf_i0_code_anyOf_i0)
- [Option 2](https://google.github.io/adk-docs/api-reference/agentconfig/#tab-pane_anyOf_i0_generate_content_config_anyOf_i0_systemInstruction_anyOf_i0_parts_anyOf_i0_items_executableCode_anyOf_i0_code_anyOf_i1)

Type: string

Type: null

## language

Default: null

Required. Programming language of the `code`.

## Any of

- [Language](https://google.github.io/adk-docs/api-reference/agentconfig/#tab-pane_anyOf_i0_generate_content_config_anyOf_i0_systemInstruction_anyOf_i0_parts_anyOf_i0_items_executableCode_anyOf_i0_language_anyOf_i0)
- [Option 2](https://google.github.io/adk-docs/api-reference/agentconfig/#tab-pane_anyOf_i0_generate_content_config_anyOf_i0_systemInstruction_anyOf_i0_parts_anyOf_i0_items_executableCode_anyOf_i0_language_anyOf_i1)

#### Language

Type: enum (of string)

Required. Programming language of the `code`.

#### Must be one of:

- "LANGUAGE\_UNSPECIFIED"
- "PYTHON"

Type: null

Type: null

## functionCall

Default: null

Optional. A predicted \[FunctionCall\] returned from the model that contains a string representing the \[FunctionDeclaration.name\] with the parameters and their values.

## Any of

- [FunctionCall](https://google.github.io/adk-docs/api-reference/agentconfig/#tab-pane_anyOf_i0_generate_content_config_anyOf_i0_systemInstruction_anyOf_i0_parts_anyOf_i0_items_functionCall_anyOf_i0)
- [Option 2](https://google.github.io/adk-docs/api-reference/agentconfig/#tab-pane_anyOf_i0_generate_content_config_anyOf_i0_systemInstruction_anyOf_i0_parts_anyOf_i0_items_functionCall_anyOf_i1)

#### FunctionCall

Type: object

A function call.
No Additional Properties

## id

#### Id

Default: null

The unique id of the function call. If populated, the client to execute the

`function_call` and return the response with the matching `id`.

## Any of

- [Option 1](https://google.github.io/adk-docs/api-reference/agentconfig/#tab-pane_anyOf_i0_generate_content_config_anyOf_i0_systemInstruction_anyOf_i0_parts_anyOf_i0_items_functionCall_anyOf_i0_id_anyOf_i0)
- [Option 2](https://google.github.io/adk-docs/api-reference/agentconfig/#tab-pane_anyOf_i0_generate_content_config_anyOf_i0_systemInstruction_anyOf_i0_parts_anyOf_i0_items_functionCall_anyOf_i0_id_anyOf_i1)

Type: string

Type: null

## args

#### Args

Default: null

Optional. The function parameters and values in JSON object format. See \[FunctionDeclaration.parameters\] for parameter details.

## Any of

- [Option 1](https://google.github.io/adk-docs/api-reference/agentconfig/#tab-pane_anyOf_i0_generate_content_config_anyOf_i0_systemInstruction_anyOf_i0_parts_anyOf_i0_items_functionCall_anyOf_i0_args_anyOf_i0)
- [Option 2](https://google.github.io/adk-docs/api-reference/agentconfig/#tab-pane_anyOf_i0_generate_content_config_anyOf_i0_systemInstruction_anyOf_i0_parts_anyOf_i0_items_functionCall_anyOf_i0_args_anyOf_i1)

Type: object

## _Additional Properties_

Additional Properties of any type are allowed.

Type: object

Type: null

## name

#### Name

Default: null

Required. The name of the function to call. Matches \[FunctionDeclaration.name\].

## Any of

- [Option 1](https://google.github.io/adk-docs/api-reference/agentconfig/#tab-pane_anyOf_i0_generate_content_config_anyOf_i0_systemInstruction_anyOf_i0_parts_anyOf_i0_items_functionCall_anyOf_i0_name_anyOf_i0)
- [Option 2](https://google.github.io/adk-docs/api-reference/agentconfig/#tab-pane_anyOf_i0_generate_content_config_anyOf_i0_systemInstruction_anyOf_i0_parts_anyOf_i0_items_functionCall_anyOf_i0_name_anyOf_i1)

Type: string

Type: null

Type: null

## functionResponse

Default: null

Optional. The result output of a \[FunctionCall\] that contains a string representing the \[FunctionDeclaration.name\] and a structured JSON object containing any output from the function call. It is used as context to the model.

## Any of

- [FunctionResponse](https://google.github.io/adk-docs/api-reference/agentconfig/#tab-pane_anyOf_i0_generate_content_config_anyOf_i0_systemInstruction_anyOf_i0_parts_anyOf_i0_items_functionResponse_anyOf_i0)
- [Option 2](https://google.github.io/adk-docs/api-reference/agentconfig/#tab-pane_anyOf_i0_generate_content_config_anyOf_i0_systemInstruction_anyOf_i0_parts_anyOf_i0_items_functionResponse_anyOf_i1)

#### FunctionResponse

Type: object

A function response.
No Additional Properties

## willContinue

#### Willcontinue

Default: null

Signals that function call continues, and more responses will be returned, turning the function call into a generator. Is only applicable to NON _BLOCKING function calls (see FunctionDeclaration.behavior for details), ignored otherwise. If false, the default, future responses will not be considered. Is only applicable to NON_ BLOCKING function calls, is ignored otherwise. If set to false, future responses will not be considered. It is allowed to return empty `response` with `will_continue=False` to signal that the function call is finished.

## Any of

- [Option 1](https://google.github.io/adk-docs/api-reference/agentconfig/#tab-pane_anyOf_i0_generate_content_config_anyOf_i0_systemInstruction_anyOf_i0_parts_anyOf_i0_items_functionResponse_anyOf_i0_willContinue_anyOf_i0)
- [Option 2](https://google.github.io/adk-docs/api-reference/agentconfig/#tab-pane_anyOf_i0_generate_content_config_anyOf_i0_systemInstruction_anyOf_i0_parts_anyOf_i0_items_functionResponse_anyOf_i0_willContinue_anyOf_i1)

Type: boolean

Type: null

## scheduling

Default: null

Specifies how the response should be scheduled in the conversation. Only applicable to NON _BLOCKING function calls, is ignored otherwise. Defaults to WHEN_ IDLE.

## Any of

- [FunctionResponseScheduling](https://google.github.io/adk-docs/api-reference/agentconfig/#tab-pane_anyOf_i0_generate_content_config_anyOf_i0_systemInstruction_anyOf_i0_parts_anyOf_i0_items_functionResponse_anyOf_i0_scheduling_anyOf_i0)
- [Option 2](https://google.github.io/adk-docs/api-reference/agentconfig/#tab-pane_anyOf_i0_generate_content_config_anyOf_i0_systemInstruction_anyOf_i0_parts_anyOf_i0_items_functionResponse_anyOf_i0_scheduling_anyOf_i1)

#### FunctionResponseScheduling

Type: enum (of string)

Specifies how the response should be scheduled in the conversation.

#### Must be one of:

- "SCHEDULING\_UNSPECIFIED"
- "SILENT"
- "WHEN\_IDLE"
- "INTERRUPT"

Type: null

## id

#### Id

Default: null

Optional. The id of the function call this response is for. Populated by the client to match the corresponding function call `id`.

## Any of

- [Option 1](https://google.github.io/adk-docs/api-reference/agentconfig/#tab-pane_anyOf_i0_generate_content_config_anyOf_i0_systemInstruction_anyOf_i0_parts_anyOf_i0_items_functionResponse_anyOf_i0_id_anyOf_i0)
- [Option 2](https://google.github.io/adk-docs/api-reference/agentconfig/#tab-pane_anyOf_i0_generate_content_config_anyOf_i0_systemInstruction_anyOf_i0_parts_anyOf_i0_items_functionResponse_anyOf_i0_id_anyOf_i1)

Type: string

Type: null

## name

#### Name

Default: null

Required. The name of the function to call. Matches \[FunctionDeclaration.name\] and \[FunctionCall.name\].

## Any of

- [Option 1](https://google.github.io/adk-docs/api-reference/agentconfig/#tab-pane_anyOf_i0_generate_content_config_anyOf_i0_systemInstruction_anyOf_i0_parts_anyOf_i0_items_functionResponse_anyOf_i0_name_anyOf_i0)
- [Option 2](https://google.github.io/adk-docs/api-reference/agentconfig/#tab-pane_anyOf_i0_generate_content_config_anyOf_i0_systemInstruction_anyOf_i0_parts_anyOf_i0_items_functionResponse_anyOf_i0_name_anyOf_i1)

Type: string

Type: null

## response

#### Response

Default: null

Required. The function response in JSON object format. Use "output" key to specify function output and "error" key to specify error details (if any). If "output" and "error" keys are not specified, then whole "response" is treated as function output.

## Any of

- [Option 1](https://google.github.io/adk-docs/api-reference/agentconfig/#tab-pane_anyOf_i0_generate_content_config_anyOf_i0_systemInstruction_anyOf_i0_parts_anyOf_i0_items_functionResponse_anyOf_i0_response_anyOf_i0)
- [Option 2](https://google.github.io/adk-docs/api-reference/agentconfig/#tab-pane_anyOf_i0_generate_content_config_anyOf_i0_systemInstruction_anyOf_i0_parts_anyOf_i0_items_functionResponse_anyOf_i0_response_anyOf_i1)

Type: object

## _Additional Properties_

Additional Properties of any type are allowed.

Type: object

Type: null

Type: null

## text

#### Text

Default: null

Optional. Text part (can be code).

## Any of

- [Option 1](https://google.github.io/adk-docs/api-reference/agentconfig/#tab-pane_anyOf_i0_generate_content_config_anyOf_i0_systemInstruction_anyOf_i0_parts_anyOf_i0_items_text_anyOf_i0)
- [Option 2](https://google.github.io/adk-docs/api-reference/agentconfig/#tab-pane_anyOf_i0_generate_content_config_anyOf_i0_systemInstruction_anyOf_i0_parts_anyOf_i0_items_text_anyOf_i1)

Type: string

Type: null

Type: null

## role

#### Role

Default: null

Optional. The producer of the content. Must be either 'user' or

'model'. Useful to set for multi-turn conversations, otherwise can be

empty. If role is not specified, SDK will determine the role.

## Any of

- [Option 1](https://google.github.io/adk-docs/api-reference/agentconfig/#tab-pane_anyOf_i0_generate_content_config_anyOf_i0_systemInstruction_anyOf_i0_role_anyOf_i0)
- [Option 2](https://google.github.io/adk-docs/api-reference/agentconfig/#tab-pane_anyOf_i0_generate_content_config_anyOf_i0_systemInstruction_anyOf_i0_role_anyOf_i1)

Type: string

Type: null

Type: array

No Additional Items

#### Each item of this array must be:

## Any of

- [File](https://google.github.io/adk-docs/api-reference/agentconfig/#tab-pane_anyOf_i0_generate_content_config_anyOf_i0_systemInstruction_anyOf_i1_items_anyOf_i0)
- [Part](https://google.github.io/adk-docs/api-reference/agentconfig/#tab-pane_anyOf_i0_generate_content_config_anyOf_i0_systemInstruction_anyOf_i1_items_anyOf_i1)
- [Option 3](https://google.github.io/adk-docs/api-reference/agentconfig/#tab-pane_anyOf_i0_generate_content_config_anyOf_i0_systemInstruction_anyOf_i1_items_anyOf_i2)

#### File

Type: object

A file uploaded to the API.
No Additional Properties

## name

#### Name

Default: null

The `File` resource name. The ID (name excluding the "files/" prefix) can contain up to 40 characters that are lowercase alphanumeric or dashes (-). The ID cannot start or end with a dash. If the name is empty on create, a unique name will be generated. Example: `files/123-456`

## Any of

- [Option 1](https://google.github.io/adk-docs/api-reference/agentconfig/#tab-pane_anyOf_i0_generate_content_config_anyOf_i0_systemInstruction_anyOf_i1_items_anyOf_i0_name_anyOf_i0)
- [Option 2](https://google.github.io/adk-docs/api-reference/agentconfig/#tab-pane_anyOf_i0_generate_content_config_anyOf_i0_systemInstruction_anyOf_i1_items_anyOf_i0_name_anyOf_i1)

Type: string

Type: null

## displayName

#### Displayname

Default: null

Optional. The human-readable display name for the `File`. The display name must be no more than 512 characters in length, including spaces. Example: 'Welcome Image'

## Any of

- [Option 1](https://google.github.io/adk-docs/api-reference/agentconfig/#tab-pane_anyOf_i0_generate_content_config_anyOf_i0_systemInstruction_anyOf_i1_items_anyOf_i0_displayName_anyOf_i0)
- [Option 2](https://google.github.io/adk-docs/api-reference/agentconfig/#tab-pane_anyOf_i0_generate_content_config_anyOf_i0_systemInstruction_anyOf_i1_items_anyOf_i0_displayName_anyOf_i1)

Type: string

Type: null

## mimeType

#### Mimetype

Default: null

Output only. MIME type of the file.

## Any of

- [Option 1](https://google.github.io/adk-docs/api-reference/agentconfig/#tab-pane_anyOf_i0_generate_content_config_anyOf_i0_systemInstruction_anyOf_i1_items_anyOf_i0_mimeType_anyOf_i0)
- [Option 2](https://google.github.io/adk-docs/api-reference/agentconfig/#tab-pane_anyOf_i0_generate_content_config_anyOf_i0_systemInstruction_anyOf_i1_items_anyOf_i0_mimeType_anyOf_i1)

Type: string

Type: null

## sizeBytes

#### Sizebytes

Default: null

Output only. Size of the file in bytes.

## Any of

- [Option 1](https://google.github.io/adk-docs/api-reference/agentconfig/#tab-pane_anyOf_i0_generate_content_config_anyOf_i0_systemInstruction_anyOf_i1_items_anyOf_i0_sizeBytes_anyOf_i0)
- [Option 2](https://google.github.io/adk-docs/api-reference/agentconfig/#tab-pane_anyOf_i0_generate_content_config_anyOf_i0_systemInstruction_anyOf_i1_items_anyOf_i0_sizeBytes_anyOf_i1)

Type: integer

Type: null

## createTime

#### Createtime

Default: null

Output only. The timestamp of when the `File` was created.

## Any of

- [Option 1](https://google.github.io/adk-docs/api-reference/agentconfig/#tab-pane_anyOf_i0_generate_content_config_anyOf_i0_systemInstruction_anyOf_i1_items_anyOf_i0_createTime_anyOf_i0)
- [Option 2](https://google.github.io/adk-docs/api-reference/agentconfig/#tab-pane_anyOf_i0_generate_content_config_anyOf_i0_systemInstruction_anyOf_i1_items_anyOf_i0_createTime_anyOf_i1)

Type: stringFormat: date-time

Type: null

## expirationTime

#### Expirationtime

Default: null

Output only. The timestamp of when the `File` will be deleted. Only set if the `File` is scheduled to expire.

## Any of

- [Option 1](https://google.github.io/adk-docs/api-reference/agentconfig/#tab-pane_anyOf_i0_generate_content_config_anyOf_i0_systemInstruction_anyOf_i1_items_anyOf_i0_expirationTime_anyOf_i0)
- [Option 2](https://google.github.io/adk-docs/api-reference/agentconfig/#tab-pane_anyOf_i0_generate_content_config_anyOf_i0_systemInstruction_anyOf_i1_items_anyOf_i0_expirationTime_anyOf_i1)

Type: stringFormat: date-time

Type: null

## updateTime

#### Updatetime

Default: null

Output only. The timestamp of when the `File` was last updated.

## Any of

- [Option 1](https://google.github.io/adk-docs/api-reference/agentconfig/#tab-pane_anyOf_i0_generate_content_config_anyOf_i0_systemInstruction_anyOf_i1_items_anyOf_i0_updateTime_anyOf_i0)
- [Option 2](https://google.github.io/adk-docs/api-reference/agentconfig/#tab-pane_anyOf_i0_generate_content_config_anyOf_i0_systemInstruction_anyOf_i1_items_anyOf_i0_updateTime_anyOf_i1)

Type: stringFormat: date-time

Type: null

## sha256Hash

#### Sha256Hash

Default: null

Output only. SHA-256 hash of the uploaded bytes. The hash value is encoded in base64 format.

## Any of

- [Option 1](https://google.github.io/adk-docs/api-reference/agentconfig/#tab-pane_anyOf_i0_generate_content_config_anyOf_i0_systemInstruction_anyOf_i1_items_anyOf_i0_sha256Hash_anyOf_i0)
- [Option 2](https://google.github.io/adk-docs/api-reference/agentconfig/#tab-pane_anyOf_i0_generate_content_config_anyOf_i0_systemInstruction_anyOf_i1_items_anyOf_i0_sha256Hash_anyOf_i1)

Type: string

Type: null

## uri

#### Uri

Default: null

Output only. The URI of the `File`.

## Any of

- [Option 1](https://google.github.io/adk-docs/api-reference/agentconfig/#tab-pane_anyOf_i0_generate_content_config_anyOf_i0_systemInstruction_anyOf_i1_items_anyOf_i0_uri_anyOf_i0)
- [Option 2](https://google.github.io/adk-docs/api-reference/agentconfig/#tab-pane_anyOf_i0_generate_content_config_anyOf_i0_systemInstruction_anyOf_i1_items_anyOf_i0_uri_anyOf_i1)

Type: string

Type: null

## downloadUri

#### Downloaduri

Default: null

Output only. The URI of the `File`, only set for downloadable (generated) files.

## Any of

- [Option 1](https://google.github.io/adk-docs/api-reference/agentconfig/#tab-pane_anyOf_i0_generate_content_config_anyOf_i0_systemInstruction_anyOf_i1_items_anyOf_i0_downloadUri_anyOf_i0)
- [Option 2](https://google.github.io/adk-docs/api-reference/agentconfig/#tab-pane_anyOf_i0_generate_content_config_anyOf_i0_systemInstruction_anyOf_i1_items_anyOf_i0_downloadUri_anyOf_i1)

Type: string

Type: null

## state

Default: null

Output only. Processing state of the File.

## Any of

- [FileState](https://google.github.io/adk-docs/api-reference/agentconfig/#tab-pane_anyOf_i0_generate_content_config_anyOf_i0_systemInstruction_anyOf_i1_items_anyOf_i0_state_anyOf_i0)
- [Option 2](https://google.github.io/adk-docs/api-reference/agentconfig/#tab-pane_anyOf_i0_generate_content_config_anyOf_i0_systemInstruction_anyOf_i1_items_anyOf_i0_state_anyOf_i1)

#### FileState

Type: enum (of string)

State for the lifecycle of a File.

#### Must be one of:

- "STATE\_UNSPECIFIED"
- "PROCESSING"
- "ACTIVE"
- "FAILED"

Type: null

## source

Default: null

Output only. The source of the `File`.

## Any of

- [FileSource](https://google.github.io/adk-docs/api-reference/agentconfig/#tab-pane_anyOf_i0_generate_content_config_anyOf_i0_systemInstruction_anyOf_i1_items_anyOf_i0_source_anyOf_i0)
- [Option 2](https://google.github.io/adk-docs/api-reference/agentconfig/#tab-pane_anyOf_i0_generate_content_config_anyOf_i0_systemInstruction_anyOf_i1_items_anyOf_i0_source_anyOf_i1)

#### FileSource

Type: enum (of string)

Source of the File.

#### Must be one of:

- "SOURCE\_UNSPECIFIED"
- "UPLOADED"
- "GENERATED"

Type: null

## videoMetadata

#### Videometadata

Default: null

Output only. Metadata for a video.

## Any of

- [Option 1](https://google.github.io/adk-docs/api-reference/agentconfig/#tab-pane_anyOf_i0_generate_content_config_anyOf_i0_systemInstruction_anyOf_i1_items_anyOf_i0_videoMetadata_anyOf_i0)
- [Option 2](https://google.github.io/adk-docs/api-reference/agentconfig/#tab-pane_anyOf_i0_generate_content_config_anyOf_i0_systemInstruction_anyOf_i1_items_anyOf_i0_videoMetadata_anyOf_i1)

Type: object

## _Additional Properties_

Additional Properties of any type are allowed.

Type: object

Type: null

## error

Default: null

Output only. Error status if File processing failed.

## Any of

- [FileStatus](https://google.github.io/adk-docs/api-reference/agentconfig/#tab-pane_anyOf_i0_generate_content_config_anyOf_i0_systemInstruction_anyOf_i1_items_anyOf_i0_error_anyOf_i0)
- [Option 2](https://google.github.io/adk-docs/api-reference/agentconfig/#tab-pane_anyOf_i0_generate_content_config_anyOf_i0_systemInstruction_anyOf_i1_items_anyOf_i0_error_anyOf_i1)

#### FileStatus

Type: object

Status of a File that uses a common error model.
No Additional Properties

## details

#### Details

Default: null

A list of messages that carry the error details. There is a common set of message types for APIs to use.

## Any of

- [Option 1](https://google.github.io/adk-docs/api-reference/agentconfig/#tab-pane_anyOf_i0_generate_content_config_anyOf_i0_systemInstruction_anyOf_i1_items_anyOf_i0_error_anyOf_i0_details_anyOf_i0)
- [Option 2](https://google.github.io/adk-docs/api-reference/agentconfig/#tab-pane_anyOf_i0_generate_content_config_anyOf_i0_systemInstruction_anyOf_i1_items_anyOf_i0_error_anyOf_i0_details_anyOf_i1)

Type: array of object

No Additional Items

#### Each item of this array must be:

Type: object

## _Additional Properties_

Additional Properties of any type are allowed.

Type: object

Type: null

## message

#### Message

Default: null

A list of messages that carry the error details. There is a common set of message types for APIs to use.

## Any of

- [Option 1](https://google.github.io/adk-docs/api-reference/agentconfig/#tab-pane_anyOf_i0_generate_content_config_anyOf_i0_systemInstruction_anyOf_i1_items_anyOf_i0_error_anyOf_i0_message_anyOf_i0)
- [Option 2](https://google.github.io/adk-docs/api-reference/agentconfig/#tab-pane_anyOf_i0_generate_content_config_anyOf_i0_systemInstruction_anyOf_i1_items_anyOf_i0_error_anyOf_i0_message_anyOf_i1)

Type: string

Type: null

## code

#### Code

Default: null

The status code. 0 for OK, 1 for CANCELLED

## Any of

- [Option 1](https://google.github.io/adk-docs/api-reference/agentconfig/#tab-pane_anyOf_i0_generate_content_config_anyOf_i0_systemInstruction_anyOf_i1_items_anyOf_i0_error_anyOf_i0_code_anyOf_i0)
- [Option 2](https://google.github.io/adk-docs/api-reference/agentconfig/#tab-pane_anyOf_i0_generate_content_config_anyOf_i0_systemInstruction_anyOf_i1_items_anyOf_i0_error_anyOf_i0_code_anyOf_i1)

Type: integer

Type: null

Type: null

#### Part

Type: object

A datatype containing media content.

Exactly one field within a Part should be set, representing the specific type

of content being conveyed. Using multiple fields within the same `Part`

instance is considered invalid.

[Same definition as Part](https://google.github.io/adk-docs/api-reference/agentconfig/#anyOf_i0_generate_content_config_anyOf_i0_systemInstruction_anyOf_i0_parts_anyOf_i0_items)

Type: string

#### File

Type: object

A file uploaded to the API.
[Same definition as File](https://google.github.io/adk-docs/api-reference/agentconfig/#anyOf_i0_generate_content_config_anyOf_i0_systemInstruction_anyOf_i1_items_anyOf_i0)

#### Part

Type: object

A datatype containing media content.

Exactly one field within a Part should be set, representing the specific type

of content being conveyed. Using multiple fields within the same `Part`

instance is considered invalid.

[Same definition as Part](https://google.github.io/adk-docs/api-reference/agentconfig/#anyOf_i0_generate_content_config_anyOf_i0_systemInstruction_anyOf_i0_parts_anyOf_i0_items)

Type: string

Type: null

## temperature

#### Temperature

Default: null

Value that controls the degree of randomness in token selection.

Lower temperatures are good for prompts that require a less open-ended or

creative response, while higher temperatures can lead to more diverse or

creative results.

## Any of

- [Option 1](https://google.github.io/adk-docs/api-reference/agentconfig/#tab-pane_anyOf_i0_generate_content_config_anyOf_i0_temperature_anyOf_i0)
- [Option 2](https://google.github.io/adk-docs/api-reference/agentconfig/#tab-pane_anyOf_i0_generate_content_config_anyOf_i0_temperature_anyOf_i1)

Type: number

Type: null

## topP

#### Topp

Default: null

Tokens are selected from the most to least probable until the sum

of their probabilities equals this value. Use a lower value for less

random responses and a higher value for more random responses.

## Any of

- [Option 1](https://google.github.io/adk-docs/api-reference/agentconfig/#tab-pane_anyOf_i0_generate_content_config_anyOf_i0_topP_anyOf_i0)
- [Option 2](https://google.github.io/adk-docs/api-reference/agentconfig/#tab-pane_anyOf_i0_generate_content_config_anyOf_i0_topP_anyOf_i1)

Type: number

Type: null

## topK

#### Topk

Default: null

For each token selection step, the `top_k` tokens with the

highest probabilities are sampled. Then tokens are further filtered based

on `top_p` with the final token selected using temperature sampling. Use

a lower number for less random responses and a higher number for more

random responses.

## Any of

- [Option 1](https://google.github.io/adk-docs/api-reference/agentconfig/#tab-pane_anyOf_i0_generate_content_config_anyOf_i0_topK_anyOf_i0)
- [Option 2](https://google.github.io/adk-docs/api-reference/agentconfig/#tab-pane_anyOf_i0_generate_content_config_anyOf_i0_topK_anyOf_i1)

Type: number

Type: null

## candidateCount

#### Candidatecount

Default: null

Number of response variations to return.

## Any of

- [Option 1](https://google.github.io/adk-docs/api-reference/agentconfig/#tab-pane_anyOf_i0_generate_content_config_anyOf_i0_candidateCount_anyOf_i0)
- [Option 2](https://google.github.io/adk-docs/api-reference/agentconfig/#tab-pane_anyOf_i0_generate_content_config_anyOf_i0_candidateCount_anyOf_i1)

Type: integer

Type: null

## maxOutputTokens

#### Maxoutputtokens

Default: null

Maximum number of tokens that can be generated in the response.

## Any of

- [Option 1](https://google.github.io/adk-docs/api-reference/agentconfig/#tab-pane_anyOf_i0_generate_content_config_anyOf_i0_maxOutputTokens_anyOf_i0)
- [Option 2](https://google.github.io/adk-docs/api-reference/agentconfig/#tab-pane_anyOf_i0_generate_content_config_anyOf_i0_maxOutputTokens_anyOf_i1)

Type: integer

Type: null

## stopSequences

#### Stopsequences

Default: null

List of strings that tells the model to stop generating text if one

of the strings is encountered in the response.

## Any of

- [Option 1](https://google.github.io/adk-docs/api-reference/agentconfig/#tab-pane_anyOf_i0_generate_content_config_anyOf_i0_stopSequences_anyOf_i0)
- [Option 2](https://google.github.io/adk-docs/api-reference/agentconfig/#tab-pane_anyOf_i0_generate_content_config_anyOf_i0_stopSequences_anyOf_i1)

Type: array of string

No Additional Items

#### Each item of this array must be:

Type: string

Type: null

## responseLogprobs

#### Responselogprobs

Default: null

Whether to return the log probabilities of the tokens that were

chosen by the model at each step.

## Any of

- [Option 1](https://google.github.io/adk-docs/api-reference/agentconfig/#tab-pane_anyOf_i0_generate_content_config_anyOf_i0_responseLogprobs_anyOf_i0)
- [Option 2](https://google.github.io/adk-docs/api-reference/agentconfig/#tab-pane_anyOf_i0_generate_content_config_anyOf_i0_responseLogprobs_anyOf_i1)

Type: boolean

Type: null

## logprobs

#### Logprobs

Default: null

Number of top candidate tokens to return the log probabilities for

at each generation step.

## Any of

- [Option 1](https://google.github.io/adk-docs/api-reference/agentconfig/#tab-pane_anyOf_i0_generate_content_config_anyOf_i0_logprobs_anyOf_i0)
- [Option 2](https://google.github.io/adk-docs/api-reference/agentconfig/#tab-pane_anyOf_i0_generate_content_config_anyOf_i0_logprobs_anyOf_i1)

Type: integer

Type: null

## presencePenalty

#### Presencepenalty

Default: null

Positive values penalize tokens that already appear in the

generated text, increasing the probability of generating more diverse

content.

## Any of

- [Option 1](https://google.github.io/adk-docs/api-reference/agentconfig/#tab-pane_anyOf_i0_generate_content_config_anyOf_i0_presencePenalty_anyOf_i0)
- [Option 2](https://google.github.io/adk-docs/api-reference/agentconfig/#tab-pane_anyOf_i0_generate_content_config_anyOf_i0_presencePenalty_anyOf_i1)

Type: number

Type: null

## frequencyPenalty

#### Frequencypenalty

Default: null

Positive values penalize tokens that repeatedly appear in the

generated text, increasing the probability of generating more diverse

content.

## Any of

- [Option 1](https://google.github.io/adk-docs/api-reference/agentconfig/#tab-pane_anyOf_i0_generate_content_config_anyOf_i0_frequencyPenalty_anyOf_i0)
- [Option 2](https://google.github.io/adk-docs/api-reference/agentconfig/#tab-pane_anyOf_i0_generate_content_config_anyOf_i0_frequencyPenalty_anyOf_i1)

Type: number

Type: null

## seed

#### Seed

Default: null

When `seed` is fixed to a specific number, the model makes a best

effort to provide the same response for repeated requests. By default, a

random number is used.

## Any of

- [Option 1](https://google.github.io/adk-docs/api-reference/agentconfig/#tab-pane_anyOf_i0_generate_content_config_anyOf_i0_seed_anyOf_i0)
- [Option 2](https://google.github.io/adk-docs/api-reference/agentconfig/#tab-pane_anyOf_i0_generate_content_config_anyOf_i0_seed_anyOf_i1)

Type: integer

Type: null

## responseMimeType

#### Responsemimetype

Default: null

Output response mimetype of the generated candidate text.

Supported mimetype:

\- `text/plain`: (default) Text output.

\- `application/json`: JSON response in the candidates.

The model needs to be prompted to output the appropriate response type,

otherwise the behavior is undefined.

This is a preview feature.

## Any of

- [Option 1](https://google.github.io/adk-docs/api-reference/agentconfig/#tab-pane_anyOf_i0_generate_content_config_anyOf_i0_responseMimeType_anyOf_i0)
- [Option 2](https://google.github.io/adk-docs/api-reference/agentconfig/#tab-pane_anyOf_i0_generate_content_config_anyOf_i0_responseMimeType_anyOf_i1)

Type: string

Type: null

## responseSchema

#### Responseschema

Default: null

The `Schema` object allows the definition of input and output data types.

These types can be objects, but also primitives and arrays.

Represents a select subset of an [OpenAPI 3.0 schema\\
\\
object](https://spec.openapis.org/oas/v3.0.3#schema).

If set, a compatible response _mime_ type must also be set.

Compatible mimetypes: `application/json`: Schema for JSON response.

## Any of

- [Option 1](https://google.github.io/adk-docs/api-reference/agentconfig/#tab-pane_anyOf_i0_generate_content_config_anyOf_i0_responseSchema_anyOf_i0)
- [Schema](https://google.github.io/adk-docs/api-reference/agentconfig/#tab-pane_anyOf_i0_generate_content_config_anyOf_i0_responseSchema_anyOf_i1)
- [Option 3](https://google.github.io/adk-docs/api-reference/agentconfig/#tab-pane_anyOf_i0_generate_content_config_anyOf_i0_responseSchema_anyOf_i2)

Type: object

## _Additional Properties_

Additional Properties of any type are allowed.

Type: object

#### Schema

Type: object

Schema is used to define the format of input/output data.

Represents a select subset of an [OpenAPI 3.0 schema\\
\\
object](https://spec.openapis.org/oas/v3.0.3#schema-object). More fields may

be added in the future as needed.

No Additional Properties

## additionalProperties

#### Additionalproperties

Default: null

Optional. Can either be a boolean or an object; controls the presence of additional properties.

## Any of

- [Option 1](https://google.github.io/adk-docs/api-reference/agentconfig/#tab-pane_anyOf_i0_generate_content_config_anyOf_i0_responseSchema_anyOf_i1_additionalProperties_anyOf_i0)
- [Option 2](https://google.github.io/adk-docs/api-reference/agentconfig/#tab-pane_anyOf_i0_generate_content_config_anyOf_i0_responseSchema_anyOf_i1_additionalProperties_anyOf_i1)

Type: object

Type: null

## defs

#### Defs

Default: null

Optional. A map of definitions for use by `ref` Only allowed at the root of the schema.

## Any of

- [Option 1](https://google.github.io/adk-docs/api-reference/agentconfig/#tab-pane_anyOf_i0_generate_content_config_anyOf_i0_responseSchema_anyOf_i1_defs_anyOf_i0)
- [Option 2](https://google.github.io/adk-docs/api-reference/agentconfig/#tab-pane_anyOf_i0_generate_content_config_anyOf_i0_responseSchema_anyOf_i1_defs_anyOf_i1)

Type: object

## _Additional Properties_

Each additional property must conform to the following schema

#### Schema

Type: object

Schema is used to define the format of input/output data.

Represents a select subset of an [OpenAPI 3.0 schema\\
\\
object](https://spec.openapis.org/oas/v3.0.3#schema-object). More fields may

be added in the future as needed.

[Same definition as Schema](https://google.github.io/adk-docs/api-reference/agentconfig/#anyOf_i0_generate_content_config_anyOf_i0_responseSchema_anyOf_i1)

Type: null

## ref

#### Ref

Default: null

Optional. Allows indirect references between schema nodes. The value should be a valid reference to a child of the root `defs`. For example, the following schema defines a reference to a schema node named "Pet": type: object properties: pet: ref: #/defs/Pet defs: Pet: type: object properties: name: type: string The value of the "pet" property is a reference to the schema node named "Pet". See details in https://json-schema.org/understanding-json-schema/structuring

## Any of

- [Option 1](https://google.github.io/adk-docs/api-reference/agentconfig/#tab-pane_anyOf_i0_generate_content_config_anyOf_i0_responseSchema_anyOf_i1_ref_anyOf_i0)
- [Option 2](https://google.github.io/adk-docs/api-reference/agentconfig/#tab-pane_anyOf_i0_generate_content_config_anyOf_i0_responseSchema_anyOf_i1_ref_anyOf_i1)

Type: string

Type: null

## anyOf

#### Anyof

Default: null

Optional. The value should be validated against any (one or more) of the subschemas in the list.

## Any of

- [Option 1](https://google.github.io/adk-docs/api-reference/agentconfig/#tab-pane_anyOf_i0_generate_content_config_anyOf_i0_responseSchema_anyOf_i1_anyOf_anyOf_i0)
- [Option 2](https://google.github.io/adk-docs/api-reference/agentconfig/#tab-pane_anyOf_i0_generate_content_config_anyOf_i0_responseSchema_anyOf_i1_anyOf_anyOf_i1)

Type: array

No Additional Items

#### Each item of this array must be:

#### Schema

Type: object

Schema is used to define the format of input/output data.

Represents a select subset of an [OpenAPI 3.0 schema\\
\\
object](https://spec.openapis.org/oas/v3.0.3#schema-object). More fields may

be added in the future as needed.

[Same definition as Schema](https://google.github.io/adk-docs/api-reference/agentconfig/#anyOf_i0_generate_content_config_anyOf_i0_responseSchema_anyOf_i1)

Type: null

## default

#### Default

Default: null

Optional. Default value of the data.

## Any of

- [Option 1](https://google.github.io/adk-docs/api-reference/agentconfig/#tab-pane_anyOf_i0_generate_content_config_anyOf_i0_responseSchema_anyOf_i1_default_anyOf_i0)
- [Option 2](https://google.github.io/adk-docs/api-reference/agentconfig/#tab-pane_anyOf_i0_generate_content_config_anyOf_i0_responseSchema_anyOf_i1_default_anyOf_i1)

Type: object

Type: null

## description

#### Description

Default: null

Optional. The description of the data.

## Any of

- [Option 1](https://google.github.io/adk-docs/api-reference/agentconfig/#tab-pane_anyOf_i0_generate_content_config_anyOf_i0_responseSchema_anyOf_i1_description_anyOf_i0)
- [Option 2](https://google.github.io/adk-docs/api-reference/agentconfig/#tab-pane_anyOf_i0_generate_content_config_anyOf_i0_responseSchema_anyOf_i1_description_anyOf_i1)

Type: string

Type: null

## enum

#### Enum

Default: null

Optional. Possible values of the element of primitive type with enum format. Examples: 1. We can define direction as : {type:STRING, format:enum, enum:\["EAST", NORTH", "SOUTH", "WEST"\]} 2. We can define apartment number as : {type:INTEGER, format:enum, enum:\["101", "201", "301"\]}

## Any of

- [Option 1](https://google.github.io/adk-docs/api-reference/agentconfig/#tab-pane_anyOf_i0_generate_content_config_anyOf_i0_responseSchema_anyOf_i1_enum_anyOf_i0)
- [Option 2](https://google.github.io/adk-docs/api-reference/agentconfig/#tab-pane_anyOf_i0_generate_content_config_anyOf_i0_responseSchema_anyOf_i1_enum_anyOf_i1)

Type: array of string

No Additional Items

#### Each item of this array must be:

Type: string

Type: null

## example

#### Example

Default: null

Optional. Example of the object. Will only populated when the object is the root.

## Any of

- [Option 1](https://google.github.io/adk-docs/api-reference/agentconfig/#tab-pane_anyOf_i0_generate_content_config_anyOf_i0_responseSchema_anyOf_i1_example_anyOf_i0)
- [Option 2](https://google.github.io/adk-docs/api-reference/agentconfig/#tab-pane_anyOf_i0_generate_content_config_anyOf_i0_responseSchema_anyOf_i1_example_anyOf_i1)

Type: object

Type: null

## format

#### Format

Default: null

Optional. The format of the data. Supported formats: for NUMBER type: "float", "double" for INTEGER type: "int32", "int64" for STRING type: "email", "byte", etc

## Any of

- [Option 1](https://google.github.io/adk-docs/api-reference/agentconfig/#tab-pane_anyOf_i0_generate_content_config_anyOf_i0_responseSchema_anyOf_i1_format_anyOf_i0)
- [Option 2](https://google.github.io/adk-docs/api-reference/agentconfig/#tab-pane_anyOf_i0_generate_content_config_anyOf_i0_responseSchema_anyOf_i1_format_anyOf_i1)

Type: string

Type: null

## items

Default: null

Optional. SCHEMA FIELDS FOR TYPE ARRAY Schema of the elements of Type.ARRAY.

## Any of

- [Schema](https://google.github.io/adk-docs/api-reference/agentconfig/#tab-pane_anyOf_i0_generate_content_config_anyOf_i0_responseSchema_anyOf_i1_items_anyOf_i0)
- [Option 2](https://google.github.io/adk-docs/api-reference/agentconfig/#tab-pane_anyOf_i0_generate_content_config_anyOf_i0_responseSchema_anyOf_i1_items_anyOf_i1)

#### Schema

Type: object

Schema is used to define the format of input/output data.

Represents a select subset of an [OpenAPI 3.0 schema\\
\\
object](https://spec.openapis.org/oas/v3.0.3#schema-object). More fields may

be added in the future as needed.

[Same definition as Schema](https://google.github.io/adk-docs/api-reference/agentconfig/#anyOf_i0_generate_content_config_anyOf_i0_responseSchema_anyOf_i1)

Type: null

## maxItems

#### Maxitems

Default: null

Optional. Maximum number of the elements for Type.ARRAY.

## Any of

- [Option 1](https://google.github.io/adk-docs/api-reference/agentconfig/#tab-pane_anyOf_i0_generate_content_config_anyOf_i0_responseSchema_anyOf_i1_maxItems_anyOf_i0)
- [Option 2](https://google.github.io/adk-docs/api-reference/agentconfig/#tab-pane_anyOf_i0_generate_content_config_anyOf_i0_responseSchema_anyOf_i1_maxItems_anyOf_i1)

Type: integer

Type: null

## maxLength

#### Maxlength

Default: null

Optional. Maximum length of the Type.STRING

## Any of

- [Option 1](https://google.github.io/adk-docs/api-reference/agentconfig/#tab-pane_anyOf_i0_generate_content_config_anyOf_i0_responseSchema_anyOf_i1_maxLength_anyOf_i0)
- [Option 2](https://google.github.io/adk-docs/api-reference/agentconfig/#tab-pane_anyOf_i0_generate_content_config_anyOf_i0_responseSchema_anyOf_i1_maxLength_anyOf_i1)

Type: integer

Type: null

## maxProperties

#### Maxproperties

Default: null

Optional. Maximum number of the properties for Type.OBJECT.

## Any of

- [Option 1](https://google.github.io/adk-docs/api-reference/agentconfig/#tab-pane_anyOf_i0_generate_content_config_anyOf_i0_responseSchema_anyOf_i1_maxProperties_anyOf_i0)
- [Option 2](https://google.github.io/adk-docs/api-reference/agentconfig/#tab-pane_anyOf_i0_generate_content_config_anyOf_i0_responseSchema_anyOf_i1_maxProperties_anyOf_i1)

Type: integer

Type: null

## maximum

#### Maximum

Default: null

Optional. Maximum value of the Type.INTEGER and Type.NUMBER

## Any of

- [Option 1](https://google.github.io/adk-docs/api-reference/agentconfig/#tab-pane_anyOf_i0_generate_content_config_anyOf_i0_responseSchema_anyOf_i1_maximum_anyOf_i0)
- [Option 2](https://google.github.io/adk-docs/api-reference/agentconfig/#tab-pane_anyOf_i0_generate_content_config_anyOf_i0_responseSchema_anyOf_i1_maximum_anyOf_i1)

Type: number

Type: null

## minItems

#### Minitems

Default: null

Optional. Minimum number of the elements for Type.ARRAY.

## Any of

- [Option 1](https://google.github.io/adk-docs/api-reference/agentconfig/#tab-pane_anyOf_i0_generate_content_config_anyOf_i0_responseSchema_anyOf_i1_minItems_anyOf_i0)
- [Option 2](https://google.github.io/adk-docs/api-reference/agentconfig/#tab-pane_anyOf_i0_generate_content_config_anyOf_i0_responseSchema_anyOf_i1_minItems_anyOf_i1)

Type: integer

Type: null

## minLength

#### Minlength

Default: null

Optional. SCHEMA FIELDS FOR TYPE STRING Minimum length of the Type.STRING

## Any of

- [Option 1](https://google.github.io/adk-docs/api-reference/agentconfig/#tab-pane_anyOf_i0_generate_content_config_anyOf_i0_responseSchema_anyOf_i1_minLength_anyOf_i0)
- [Option 2](https://google.github.io/adk-docs/api-reference/agentconfig/#tab-pane_anyOf_i0_generate_content_config_anyOf_i0_responseSchema_anyOf_i1_minLength_anyOf_i1)

Type: integer

Type: null

## minProperties

#### Minproperties

Default: null

Optional. Minimum number of the properties for Type.OBJECT.

## Any of

- [Option 1](https://google.github.io/adk-docs/api-reference/agentconfig/#tab-pane_anyOf_i0_generate_content_config_anyOf_i0_responseSchema_anyOf_i1_minProperties_anyOf_i0)
- [Option 2](https://google.github.io/adk-docs/api-reference/agentconfig/#tab-pane_anyOf_i0_generate_content_config_anyOf_i0_responseSchema_anyOf_i1_minProperties_anyOf_i1)

Type: integer

Type: null

## minimum

#### Minimum

Default: null

Optional. SCHEMA FIELDS FOR TYPE INTEGER and NUMBER Minimum value of the Type.INTEGER and Type.NUMBER

## Any of

- [Option 1](https://google.github.io/adk-docs/api-reference/agentconfig/#tab-pane_anyOf_i0_generate_content_config_anyOf_i0_responseSchema_anyOf_i1_minimum_anyOf_i0)
- [Option 2](https://google.github.io/adk-docs/api-reference/agentconfig/#tab-pane_anyOf_i0_generate_content_config_anyOf_i0_responseSchema_anyOf_i1_minimum_anyOf_i1)

Type: number

Type: null

## nullable

#### Nullable

Default: null

Optional. Indicates if the value may be null.

## Any of

- [Option 1](https://google.github.io/adk-docs/api-reference/agentconfig/#tab-pane_anyOf_i0_generate_content_config_anyOf_i0_responseSchema_anyOf_i1_nullable_anyOf_i0)
- [Option 2](https://google.github.io/adk-docs/api-reference/agentconfig/#tab-pane_anyOf_i0_generate_content_config_anyOf_i0_responseSchema_anyOf_i1_nullable_anyOf_i1)

Type: boolean

Type: null

## pattern

#### Pattern

Default: null

Optional. Pattern of the Type.STRING to restrict a string to a regular expression.

## Any of

- [Option 1](https://google.github.io/adk-docs/api-reference/agentconfig/#tab-pane_anyOf_i0_generate_content_config_anyOf_i0_responseSchema_anyOf_i1_pattern_anyOf_i0)
- [Option 2](https://google.github.io/adk-docs/api-reference/agentconfig/#tab-pane_anyOf_i0_generate_content_config_anyOf_i0_responseSchema_anyOf_i1_pattern_anyOf_i1)

Type: string

Type: null

## properties

#### Properties

Default: null

Optional. SCHEMA FIELDS FOR TYPE OBJECT Properties of Type.OBJECT.

## Any of

- [Option 1](https://google.github.io/adk-docs/api-reference/agentconfig/#tab-pane_anyOf_i0_generate_content_config_anyOf_i0_responseSchema_anyOf_i1_properties_anyOf_i0)
- [Option 2](https://google.github.io/adk-docs/api-reference/agentconfig/#tab-pane_anyOf_i0_generate_content_config_anyOf_i0_responseSchema_anyOf_i1_properties_anyOf_i1)

Type: object

## _Additional Properties_

Each additional property must conform to the following schema

#### Schema

Type: object

Schema is used to define the format of input/output data.

Represents a select subset of an [OpenAPI 3.0 schema\\
\\
object](https://spec.openapis.org/oas/v3.0.3#schema-object). More fields may

be added in the future as needed.

[Same definition as Schema](https://google.github.io/adk-docs/api-reference/agentconfig/#anyOf_i0_generate_content_config_anyOf_i0_responseSchema_anyOf_i1)

Type: null

## propertyOrdering

#### Propertyordering

Default: null

Optional. The order of the properties. Not a standard field in open api spec. Only used to support the order of the properties.

## Any of

- [Option 1](https://google.github.io/adk-docs/api-reference/agentconfig/#tab-pane_anyOf_i0_generate_content_config_anyOf_i0_responseSchema_anyOf_i1_propertyOrdering_anyOf_i0)
- [Option 2](https://google.github.io/adk-docs/api-reference/agentconfig/#tab-pane_anyOf_i0_generate_content_config_anyOf_i0_responseSchema_anyOf_i1_propertyOrdering_anyOf_i1)

Type: array of string

No Additional Items

#### Each item of this array must be:

Type: string

Type: null

## required

#### Required

Default: null

Optional. Required properties of Type.OBJECT.

## Any of

- [Option 1](https://google.github.io/adk-docs/api-reference/agentconfig/#tab-pane_anyOf_i0_generate_content_config_anyOf_i0_responseSchema_anyOf_i1_required_anyOf_i0)
- [Option 2](https://google.github.io/adk-docs/api-reference/agentconfig/#tab-pane_anyOf_i0_generate_content_config_anyOf_i0_responseSchema_anyOf_i1_required_anyOf_i1)

Type: array of string

No Additional Items

#### Each item of this array must be:

Type: string

Type: null

## title

#### Title

Default: null

Optional. The title of the Schema.

## Any of

- [Option 1](https://google.github.io/adk-docs/api-reference/agentconfig/#tab-pane_anyOf_i0_generate_content_config_anyOf_i0_responseSchema_anyOf_i1_title_anyOf_i0)
- [Option 2](https://google.github.io/adk-docs/api-reference/agentconfig/#tab-pane_anyOf_i0_generate_content_config_anyOf_i0_responseSchema_anyOf_i1_title_anyOf_i1)

Type: string

Type: null

## type

Default: null

Optional. The type of the data.

## Any of

- [Type](https://google.github.io/adk-docs/api-reference/agentconfig/#tab-pane_anyOf_i0_generate_content_config_anyOf_i0_responseSchema_anyOf_i1_type_anyOf_i0)
- [Option 2](https://google.github.io/adk-docs/api-reference/agentconfig/#tab-pane_anyOf_i0_generate_content_config_anyOf_i0_responseSchema_anyOf_i1_type_anyOf_i1)

#### Type

Type: enum (of string)

Optional. The type of the data.

#### Must be one of:

- "TYPE\_UNSPECIFIED"
- "STRING"
- "NUMBER"
- "INTEGER"
- "BOOLEAN"
- "ARRAY"
- "OBJECT"
- "NULL"

Type: null

Type: null

## responseJsonSchema

#### Responsejsonschema

Default: null

Optional. Output schema of the generated response.

This is an alternative to `response_schema` that accepts [JSON\\
\\
Schema](https://json-schema.org/). If set, `response_schema` must be

omitted, but `response_mime_type` is required. While the full JSON Schema

may be sent, not all features are supported. Specifically, only the

following properties are supported: - `$id` \- `$defs` \- `$ref` \- `$anchor`

\- `type` \- `format` \- `title` \- `description` \- `enum` (for strings and

numbers) - `items` \- `prefixItems` \- `minItems` \- `maxItems` \- `minimum` -

`maximum` \- `anyOf` \- `oneOf` (interpreted the same as `anyOf`) -

`properties` \- `additionalProperties` \- `required` The non-standard

`propertyOrdering` property may also be set. Cyclic references are

unrolled to a limited degree and, as such, may only be used within

non-required properties. (Nullable properties are not sufficient.) If

`$ref` is set on a sub-schema, no other properties, except for than those

starting as a `$`, may be set.

## Any of

- [Option 1](https://google.github.io/adk-docs/api-reference/agentconfig/#tab-pane_anyOf_i0_generate_content_config_anyOf_i0_responseJsonSchema_anyOf_i0)
- [Option 2](https://google.github.io/adk-docs/api-reference/agentconfig/#tab-pane_anyOf_i0_generate_content_config_anyOf_i0_responseJsonSchema_anyOf_i1)

Type: object

Type: null

## routingConfig

Default: null

Configuration for model router requests.

## Any of

- [GenerationConfigRoutingConfig](https://google.github.io/adk-docs/api-reference/agentconfig/#tab-pane_anyOf_i0_generate_content_config_anyOf_i0_routingConfig_anyOf_i0)
- [Option 2](https://google.github.io/adk-docs/api-reference/agentconfig/#tab-pane_anyOf_i0_generate_content_config_anyOf_i0_routingConfig_anyOf_i1)

#### GenerationConfigRoutingConfig

Type: object

The configuration for routing the request to a specific model.
No Additional Properties

## autoMode

Default: null

Automated routing.

## Any of

- [GenerationConfigRoutingConfigAutoRoutingMode](https://google.github.io/adk-docs/api-reference/agentconfig/#tab-pane_anyOf_i0_generate_content_config_anyOf_i0_routingConfig_anyOf_i0_autoMode_anyOf_i0)
- [Option 2](https://google.github.io/adk-docs/api-reference/agentconfig/#tab-pane_anyOf_i0_generate_content_config_anyOf_i0_routingConfig_anyOf_i0_autoMode_anyOf_i1)

#### GenerationConfigRoutingConfigAutoRoutingMode

Type: object

When automated routing is specified, the routing will be determined by the pretrained routing model and customer provided model routing preference.
No Additional Properties

## modelRoutingPreference

#### Modelroutingpreference

Default: null

The model routing preference.

## Any of

- [Option 1](https://google.github.io/adk-docs/api-reference/agentconfig/#tab-pane_anyOf_i0_generate_content_config_anyOf_i0_routingConfig_anyOf_i0_autoMode_anyOf_i0_modelRoutingPreference_anyOf_i0)
- [Option 2](https://google.github.io/adk-docs/api-reference/agentconfig/#tab-pane_anyOf_i0_generate_content_config_anyOf_i0_routingConfig_anyOf_i0_autoMode_anyOf_i0_modelRoutingPreference_anyOf_i1)

Type: enum (of string)

#### Must be one of:

- "UNKNOWN"
- "PRIORITIZE\_QUALITY"
- "BALANCED"
- "PRIORITIZE\_COST"

Type: null

Type: null

## manualMode

Default: null

Manual routing.

## Any of

- [GenerationConfigRoutingConfigManualRoutingMode](https://google.github.io/adk-docs/api-reference/agentconfig/#tab-pane_anyOf_i0_generate_content_config_anyOf_i0_routingConfig_anyOf_i0_manualMode_anyOf_i0)
- [Option 2](https://google.github.io/adk-docs/api-reference/agentconfig/#tab-pane_anyOf_i0_generate_content_config_anyOf_i0_routingConfig_anyOf_i0_manualMode_anyOf_i1)

#### GenerationConfigRoutingConfigManualRoutingMode

Type: object

When manual routing is set, the specified model will be used directly.
No Additional Properties

## modelName

#### Modelname

Default: null

The model name to use. Only the public LLM models are accepted. See [Supported models](https://cloud.google.com/vertex-ai/generative-ai/docs/model-reference/inference#supported-models).

## Any of

- [Option 1](https://google.github.io/adk-docs/api-reference/agentconfig/#tab-pane_anyOf_i0_generate_content_config_anyOf_i0_routingConfig_anyOf_i0_manualMode_anyOf_i0_modelName_anyOf_i0)
- [Option 2](https://google.github.io/adk-docs/api-reference/agentconfig/#tab-pane_anyOf_i0_generate_content_config_anyOf_i0_routingConfig_anyOf_i0_manualMode_anyOf_i0_modelName_anyOf_i1)

Type: string

Type: null

Type: null

Type: null

## modelSelectionConfig

Default: null

Configuration for model selection.

## Any of

- [ModelSelectionConfig](https://google.github.io/adk-docs/api-reference/agentconfig/#tab-pane_anyOf_i0_generate_content_config_anyOf_i0_modelSelectionConfig_anyOf_i0)
- [Option 2](https://google.github.io/adk-docs/api-reference/agentconfig/#tab-pane_anyOf_i0_generate_content_config_anyOf_i0_modelSelectionConfig_anyOf_i1)

#### ModelSelectionConfig

Type: object

Config for model selection.
No Additional Properties

## featureSelectionPreference

Default: null

Options for feature selection preference.

## Any of

- [FeatureSelectionPreference](https://google.github.io/adk-docs/api-reference/agentconfig/#tab-pane_anyOf_i0_generate_content_config_anyOf_i0_modelSelectionConfig_anyOf_i0_featureSelectionPreference_anyOf_i0)
- [Option 2](https://google.github.io/adk-docs/api-reference/agentconfig/#tab-pane_anyOf_i0_generate_content_config_anyOf_i0_modelSelectionConfig_anyOf_i0_featureSelectionPreference_anyOf_i1)

#### FeatureSelectionPreference

Type: enum (of string)

Options for feature selection preference.

#### Must be one of:

- "FEATURE\_SELECTION\_PREFERENCE\_UNSPECIFIED"
- "PRIORITIZE\_QUALITY"
- "BALANCED"
- "PRIORITIZE\_COST"

Type: null

Type: null

## safetySettings

#### Safetysettings

Default: null

Safety settings in the request to block unsafe content in the

response.

## Any of

- [Option 1](https://google.github.io/adk-docs/api-reference/agentconfig/#tab-pane_anyOf_i0_generate_content_config_anyOf_i0_safetySettings_anyOf_i0)
- [Option 2](https://google.github.io/adk-docs/api-reference/agentconfig/#tab-pane_anyOf_i0_generate_content_config_anyOf_i0_safetySettings_anyOf_i1)

Type: array

No Additional Items

#### Each item of this array must be:

#### SafetySetting

Type: object

Safety settings.
No Additional Properties

## method

Default: null

Determines if the harm block method uses probability or probability

and severity scores.

## Any of

- [HarmBlockMethod](https://google.github.io/adk-docs/api-reference/agentconfig/#tab-pane_anyOf_i0_generate_content_config_anyOf_i0_safetySettings_anyOf_i0_items_method_anyOf_i0)
- [Option 2](https://google.github.io/adk-docs/api-reference/agentconfig/#tab-pane_anyOf_i0_generate_content_config_anyOf_i0_safetySettings_anyOf_i0_items_method_anyOf_i1)

#### HarmBlockMethod

Type: enum (of string)

Optional.

Specify if the threshold is used for probability or severity score. If not

specified, the threshold is used for probability score.

#### Must be one of:

- "HARM\_BLOCK\_METHOD\_UNSPECIFIED"
- "SEVERITY"
- "PROBABILITY"

Type: null

## category

Default: null

Required. Harm category.

## Any of

- [HarmCategory](https://google.github.io/adk-docs/api-reference/agentconfig/#tab-pane_anyOf_i0_generate_content_config_anyOf_i0_safetySettings_anyOf_i0_items_category_anyOf_i0)
- [Option 2](https://google.github.io/adk-docs/api-reference/agentconfig/#tab-pane_anyOf_i0_generate_content_config_anyOf_i0_safetySettings_anyOf_i0_items_category_anyOf_i1)

#### HarmCategory

Type: enum (of string)

Required. Harm category.

#### Must be one of:

- "HARM\_CATEGORY\_UNSPECIFIED"
- "HARM\_CATEGORY\_HATE\_SPEECH"
- "HARM\_CATEGORY\_DANGEROUS\_CONTENT"
- "HARM\_CATEGORY\_HARASSMENT"
- "HARM\_CATEGORY\_SEXUALLY\_EXPLICIT"
- "HARM\_CATEGORY\_CIVIC\_INTEGRITY"
- "HARM\_CATEGORY\_IMAGE\_HATE"
- "HARM\_CATEGORY\_IMAGE\_DANGEROUS\_CONTENT"
- "HARM\_CATEGORY\_IMAGE\_HARASSMENT"
- "HARM\_CATEGORY\_IMAGE\_SEXUALLY\_EXPLICIT"

Type: null

## threshold

Default: null

Required. The harm block threshold.

## Any of

- [HarmBlockThreshold](https://google.github.io/adk-docs/api-reference/agentconfig/#tab-pane_anyOf_i0_generate_content_config_anyOf_i0_safetySettings_anyOf_i0_items_threshold_anyOf_i0)
- [Option 2](https://google.github.io/adk-docs/api-reference/agentconfig/#tab-pane_anyOf_i0_generate_content_config_anyOf_i0_safetySettings_anyOf_i0_items_threshold_anyOf_i1)

#### HarmBlockThreshold

Type: enum (of string)

Required. The harm block threshold.

#### Must be one of:

- "HARM\_BLOCK\_THRESHOLD\_UNSPECIFIED"
- "BLOCK\_LOW\_AND\_ABOVE"
- "BLOCK\_MEDIUM\_AND\_ABOVE"
- "BLOCK\_ONLY\_HIGH"
- "BLOCK\_NONE"
- "OFF"

Type: null

Type: null

## tools

#### Tools

Default: null

Code that enables the system to interact with external systems to

perform an action outside of the knowledge and scope of the model.

## Any of

- [Option 1](https://google.github.io/adk-docs/api-reference/agentconfig/#tab-pane_anyOf_i0_generate_content_config_anyOf_i0_tools_anyOf_i0)
- [Option 2](https://google.github.io/adk-docs/api-reference/agentconfig/#tab-pane_anyOf_i0_generate_content_config_anyOf_i0_tools_anyOf_i1)

Type: array

No Additional Items

#### Each item of this array must be:

## Any of

- [Tool](https://google.github.io/adk-docs/api-reference/agentconfig/#tab-pane_anyOf_i0_generate_content_config_anyOf_i0_tools_anyOf_i0_items_anyOf_i0)
- [Tool](https://google.github.io/adk-docs/api-reference/agentconfig/#tab-pane_anyOf_i0_generate_content_config_anyOf_i0_tools_anyOf_i0_items_anyOf_i1)

#### Tool

Type: object

Tool details of a tool that the model may use to generate a response.
No Additional Properties

## functionDeclarations

#### Functiondeclarations

Default: null

List of function declarations that the tool supports.

## Any of

- [Option 1](https://google.github.io/adk-docs/api-reference/agentconfig/#tab-pane_anyOf_i0_generate_content_config_anyOf_i0_tools_anyOf_i0_items_anyOf_i0_functionDeclarations_anyOf_i0)
- [Option 2](https://google.github.io/adk-docs/api-reference/agentconfig/#tab-pane_anyOf_i0_generate_content_config_anyOf_i0_tools_anyOf_i0_items_anyOf_i0_functionDeclarations_anyOf_i1)

Type: array

No Additional Items

#### Each item of this array must be:

#### FunctionDeclaration

Type: object

Defines a function that the model can generate JSON inputs for.

The inputs are based on `OpenAPI 3.0 specifications
<https://spec.openapis.org/oas/v3.0.3>`\_.
No Additional Properties

## behavior

Default: null

Defines the function behavior.

## Any of

- [Behavior](https://google.github.io/adk-docs/api-reference/agentconfig/#tab-pane_anyOf_i0_generate_content_config_anyOf_i0_tools_anyOf_i0_items_anyOf_i0_functionDeclarations_anyOf_i0_items_behavior_anyOf_i0)
- [Option 2](https://google.github.io/adk-docs/api-reference/agentconfig/#tab-pane_anyOf_i0_generate_content_config_anyOf_i0_tools_anyOf_i0_items_anyOf_i0_functionDeclarations_anyOf_i0_items_behavior_anyOf_i1)

#### Behavior

Type: enum (of string)

Defines the function behavior. Defaults to `BLOCKING`.

#### Must be one of:

- "UNSPECIFIED"
- "BLOCKING"
- "NON\_BLOCKING"

Type: null

## description

#### Description

Default: null

Optional. Description and purpose of the function. Model uses it to decide how and whether to call the function.

## Any of

- [Option 1](https://google.github.io/adk-docs/api-reference/agentconfig/#tab-pane_anyOf_i0_generate_content_config_anyOf_i0_tools_anyOf_i0_items_anyOf_i0_functionDeclarations_anyOf_i0_items_description_anyOf_i0)
- [Option 2](https://google.github.io/adk-docs/api-reference/agentconfig/#tab-pane_anyOf_i0_generate_content_config_anyOf_i0_tools_anyOf_i0_items_anyOf_i0_functionDeclarations_anyOf_i0_items_description_anyOf_i1)

Type: string

Type: null

## name

#### Name

Default: null

Required. The name of the function to call. Must start with a letter or an underscore. Must be a-z, A-Z, 0-9, or contain underscores, dots and dashes, with a maximum length of 64.

## Any of

- [Option 1](https://google.github.io/adk-docs/api-reference/agentconfig/#tab-pane_anyOf_i0_generate_content_config_anyOf_i0_tools_anyOf_i0_items_anyOf_i0_functionDeclarations_anyOf_i0_items_name_anyOf_i0)
- [Option 2](https://google.github.io/adk-docs/api-reference/agentconfig/#tab-pane_anyOf_i0_generate_content_config_anyOf_i0_tools_anyOf_i0_items_anyOf_i0_functionDeclarations_anyOf_i0_items_name_anyOf_i1)

Type: string

Type: null

## parameters

Default: null

Optional. Describes the parameters to this function in JSON Schema Object format. Reflects the Open API 3.03 Parameter Object. string Key: the name of the parameter. Parameter names are case sensitive. Schema Value: the Schema defining the type used for the parameter. For function with no parameters, this can be left unset. Parameter names must start with a letter or an underscore and must only contain chars a-z, A-Z, 0-9, or underscores with a maximum length of 64. Example with 1 required and 1 optional parameter: type: OBJECT properties: param1: type: STRING param2: type: INTEGER required: - param1

## Any of

- [Schema](https://google.github.io/adk-docs/api-reference/agentconfig/#tab-pane_anyOf_i0_generate_content_config_anyOf_i0_tools_anyOf_i0_items_anyOf_i0_functionDeclarations_anyOf_i0_items_parameters_anyOf_i0)
- [Option 2](https://google.github.io/adk-docs/api-reference/agentconfig/#tab-pane_anyOf_i0_generate_content_config_anyOf_i0_tools_anyOf_i0_items_anyOf_i0_functionDeclarations_anyOf_i0_items_parameters_anyOf_i1)

#### Schema

Type: object

Schema is used to define the format of input/output data.

Represents a select subset of an [OpenAPI 3.0 schema\\
\\
object](https://spec.openapis.org/oas/v3.0.3#schema-object). More fields may

be added in the future as needed.

[Same definition as Schema](https://google.github.io/adk-docs/api-reference/agentconfig/#anyOf_i0_generate_content_config_anyOf_i0_responseSchema_anyOf_i1)

Type: null

## parametersJsonSchema

#### Parametersjsonschema

Default: null

Optional. Describes the parameters to the function in JSON Schema format. The schema must describe an object where the properties are the parameters to the function. For example: `{ "type": "object", "properties": { "name": { "type": "string" }, "age": { "type": "integer" } }, "additionalProperties": false, "required": ["name", "age"], "propertyOrdering": ["name", "age"] }` This field is mutually exclusive with `parameters`.

## Any of

- [Option 1](https://google.github.io/adk-docs/api-reference/agentconfig/#tab-pane_anyOf_i0_generate_content_config_anyOf_i0_tools_anyOf_i0_items_anyOf_i0_functionDeclarations_anyOf_i0_items_parametersJsonSchema_anyOf_i0)
- [Option 2](https://google.github.io/adk-docs/api-reference/agentconfig/#tab-pane_anyOf_i0_generate_content_config_anyOf_i0_tools_anyOf_i0_items_anyOf_i0_functionDeclarations_anyOf_i0_items_parametersJsonSchema_anyOf_i1)

Type: object

Type: null

## response

Default: null

Optional. Describes the output from this function in JSON Schema format. Reflects the Open API 3.03 Response Object. The Schema defines the type used for the response value of the function.

## Any of

- [Schema](https://google.github.io/adk-docs/api-reference/agentconfig/#tab-pane_anyOf_i0_generate_content_config_anyOf_i0_tools_anyOf_i0_items_anyOf_i0_functionDeclarations_anyOf_i0_items_response_anyOf_i0)
- [Option 2](https://google.github.io/adk-docs/api-reference/agentconfig/#tab-pane_anyOf_i0_generate_content_config_anyOf_i0_tools_anyOf_i0_items_anyOf_i0_functionDeclarations_anyOf_i0_items_response_anyOf_i1)

#### Schema

Type: object

Schema is used to define the format of input/output data.

Represents a select subset of an [OpenAPI 3.0 schema\\
\\
object](https://spec.openapis.org/oas/v3.0.3#schema-object). More fields may

be added in the future as needed.

[Same definition as Schema](https://google.github.io/adk-docs/api-reference/agentconfig/#anyOf_i0_generate_content_config_anyOf_i0_responseSchema_anyOf_i1)

Type: null

## responseJsonSchema

#### Responsejsonschema

Default: null

Optional. Describes the output from this function in JSON Schema format. The value specified by the schema is the response value of the function. This field is mutually exclusive with `response`.

## Any of

- [Option 1](https://google.github.io/adk-docs/api-reference/agentconfig/#tab-pane_anyOf_i0_generate_content_config_anyOf_i0_tools_anyOf_i0_items_anyOf_i0_functionDeclarations_anyOf_i0_items_responseJsonSchema_anyOf_i0)
- [Option 2](https://google.github.io/adk-docs/api-reference/agentconfig/#tab-pane_anyOf_i0_generate_content_config_anyOf_i0_tools_anyOf_i0_items_anyOf_i0_functionDeclarations_anyOf_i0_items_responseJsonSchema_anyOf_i1)

Type: object

Type: null

Type: null

## retrieval

Default: null

Optional. Retrieval tool type. System will always execute the provided retrieval tool(s) to get external knowledge to answer the prompt. Retrieval results are presented to the model for generation.

## Any of

- [Retrieval](https://google.github.io/adk-docs/api-reference/agentconfig/#tab-pane_anyOf_i0_generate_content_config_anyOf_i0_tools_anyOf_i0_items_anyOf_i0_retrieval_anyOf_i0)
- [Option 2](https://google.github.io/adk-docs/api-reference/agentconfig/#tab-pane_anyOf_i0_generate_content_config_anyOf_i0_tools_anyOf_i0_items_anyOf_i0_retrieval_anyOf_i1)

#### Retrieval

Type: object

Defines a retrieval tool that model can call to access external knowledge.
No Additional Properties

## disableAttribution

#### Disableattribution

Default: null

Optional. Deprecated. This option is no longer supported.

## Any of

- [Option 1](https://google.github.io/adk-docs/api-reference/agentconfig/#tab-pane_anyOf_i0_generate_content_config_anyOf_i0_tools_anyOf_i0_items_anyOf_i0_retrieval_anyOf_i0_disableAttribution_anyOf_i0)
- [Option 2](https://google.github.io/adk-docs/api-reference/agentconfig/#tab-pane_anyOf_i0_generate_content_config_anyOf_i0_tools_anyOf_i0_items_anyOf_i0_retrieval_anyOf_i0_disableAttribution_anyOf_i1)

Type: boolean

Type: null

## externalApi

Default: null

Use data source powered by external API for grounding.

## Any of

- [ExternalApi](https://google.github.io/adk-docs/api-reference/agentconfig/#tab-pane_anyOf_i0_generate_content_config_anyOf_i0_tools_anyOf_i0_items_anyOf_i0_retrieval_anyOf_i0_externalApi_anyOf_i0)
- [Option 2](https://google.github.io/adk-docs/api-reference/agentconfig/#tab-pane_anyOf_i0_generate_content_config_anyOf_i0_tools_anyOf_i0_items_anyOf_i0_retrieval_anyOf_i0_externalApi_anyOf_i1)

#### ExternalApi

Type: object

Retrieve from data source powered by external API for grounding.

The external API is not owned by Google, but need to follow the pre-defined

API spec.
No Additional Properties

## apiAuth

Default: null

The authentication config to access the API. Deprecated. Please use auth\_config instead.

## Any of

- [ApiAuth](https://google.github.io/adk-docs/api-reference/agentconfig/#tab-pane_anyOf_i0_generate_content_config_anyOf_i0_tools_anyOf_i0_items_anyOf_i0_retrieval_anyOf_i0_externalApi_anyOf_i0_apiAuth_anyOf_i0)
- [Option 2](https://google.github.io/adk-docs/api-reference/agentconfig/#tab-pane_anyOf_i0_generate_content_config_anyOf_i0_tools_anyOf_i0_items_anyOf_i0_retrieval_anyOf_i0_externalApi_anyOf_i0_apiAuth_anyOf_i1)

#### ApiAuth

Type: object

The generic reusable api auth config.

Deprecated. Please use AuthConfig (google/cloud/aiplatform/master/auth.proto)

instead.
No Additional Properties

## apiKeyConfig

Default: null

The API secret.

## Any of

- [ApiAuthApiKeyConfig](https://google.github.io/adk-docs/api-reference/agentconfig/#tab-pane_anyOf_i0_generate_content_config_anyOf_i0_tools_anyOf_i0_items_anyOf_i0_retrieval_anyOf_i0_externalApi_anyOf_i0_apiAuth_anyOf_i0_apiKeyConfig_anyOf_i0)
- [Option 2](https://google.github.io/adk-docs/api-reference/agentconfig/#tab-pane_anyOf_i0_generate_content_config_anyOf_i0_tools_anyOf_i0_items_anyOf_i0_retrieval_anyOf_i0_externalApi_anyOf_i0_apiAuth_anyOf_i0_apiKeyConfig_anyOf_i1)

#### ApiAuthApiKeyConfig

Type: object

The API secret.
No Additional Properties

## apiKeySecretVersion

#### Apikeysecretversion

Default: null

Required. The SecretManager secret version resource name storing API key. e.g. projects/{project}/secrets/{secret}/versions/{version}

## Any of

- [Option 1](https://google.github.io/adk-docs/api-reference/agentconfig/#tab-pane_anyOf_i0_generate_content_config_anyOf_i0_tools_anyOf_i0_items_anyOf_i0_retrieval_anyOf_i0_externalApi_anyOf_i0_apiAuth_anyOf_i0_apiKeyConfig_anyOf_i0_apiKeySecretVersion_anyOf_i0)
- [Option 2](https://google.github.io/adk-docs/api-reference/agentconfig/#tab-pane_anyOf_i0_generate_content_config_anyOf_i0_tools_anyOf_i0_items_anyOf_i0_retrieval_anyOf_i0_externalApi_anyOf_i0_apiAuth_anyOf_i0_apiKeyConfig_anyOf_i0_apiKeySecretVersion_anyOf_i1)

Type: string

Type: null

## apiKeyString

#### Apikeystring

Default: null

The API key string. Either this or `api_key_secret_version` must be set.

## Any of

- [Option 1](https://google.github.io/adk-docs/api-reference/agentconfig/#tab-pane_anyOf_i0_generate_content_config_anyOf_i0_tools_anyOf_i0_items_anyOf_i0_retrieval_anyOf_i0_externalApi_anyOf_i0_apiAuth_anyOf_i0_apiKeyConfig_anyOf_i0_apiKeyString_anyOf_i0)
- [Option 2](https://google.github.io/adk-docs/api-reference/agentconfig/#tab-pane_anyOf_i0_generate_content_config_anyOf_i0_tools_anyOf_i0_items_anyOf_i0_retrieval_anyOf_i0_externalApi_anyOf_i0_apiAuth_anyOf_i0_apiKeyConfig_anyOf_i0_apiKeyString_anyOf_i1)

Type: string

Type: null

Type: null

Type: null

## apiSpec

Default: null

The API spec that the external API implements.

## Any of

- [ApiSpec](https://google.github.io/adk-docs/api-reference/agentconfig/#tab-pane_anyOf_i0_generate_content_config_anyOf_i0_tools_anyOf_i0_items_anyOf_i0_retrieval_anyOf_i0_externalApi_anyOf_i0_apiSpec_anyOf_i0)
- [Option 2](https://google.github.io/adk-docs/api-reference/agentconfig/#tab-pane_anyOf_i0_generate_content_config_anyOf_i0_tools_anyOf_i0_items_anyOf_i0_retrieval_anyOf_i0_externalApi_anyOf_i0_apiSpec_anyOf_i1)

#### ApiSpec

Type: enum (of string)

The API spec that the external API implements.

#### Must be one of:

- "API\_SPEC\_UNSPECIFIED"
- "SIMPLE\_SEARCH"
- "ELASTIC\_SEARCH"

Type: null

## authConfig

Default: null

The authentication config to access the API.

## Any of

- [AuthConfig](https://google.github.io/adk-docs/api-reference/agentconfig/#tab-pane_anyOf_i0_generate_content_config_anyOf_i0_tools_anyOf_i0_items_anyOf_i0_retrieval_anyOf_i0_externalApi_anyOf_i0_authConfig_anyOf_i0)
- [Option 2](https://google.github.io/adk-docs/api-reference/agentconfig/#tab-pane_anyOf_i0_generate_content_config_anyOf_i0_tools_anyOf_i0_items_anyOf_i0_retrieval_anyOf_i0_externalApi_anyOf_i0_authConfig_anyOf_i1)

#### AuthConfig

Type: object

Auth configuration to run the extension.
No Additional Properties

## apiKeyConfig

Default: null

Config for API key auth.

## Any of

- [ApiKeyConfig](https://google.github.io/adk-docs/api-reference/agentconfig/#tab-pane_anyOf_i0_generate_content_config_anyOf_i0_tools_anyOf_i0_items_anyOf_i0_retrieval_anyOf_i0_externalApi_anyOf_i0_authConfig_anyOf_i0_apiKeyConfig_anyOf_i0)
- [Option 2](https://google.github.io/adk-docs/api-reference/agentconfig/#tab-pane_anyOf_i0_generate_content_config_anyOf_i0_tools_anyOf_i0_items_anyOf_i0_retrieval_anyOf_i0_externalApi_anyOf_i0_authConfig_anyOf_i0_apiKeyConfig_anyOf_i1)

#### ApiKeyConfig

Type: object

Config for authentication with API key.
No Additional Properties

## apiKeyString

#### Apikeystring

Default: null

The API key to be used in the request directly.

## Any of

- [Option 1](https://google.github.io/adk-docs/api-reference/agentconfig/#tab-pane_anyOf_i0_generate_content_config_anyOf_i0_tools_anyOf_i0_items_anyOf_i0_retrieval_anyOf_i0_externalApi_anyOf_i0_authConfig_anyOf_i0_apiKeyConfig_anyOf_i0_apiKeyString_anyOf_i0)
- [Option 2](https://google.github.io/adk-docs/api-reference/agentconfig/#tab-pane_anyOf_i0_generate_content_config_anyOf_i0_tools_anyOf_i0_items_anyOf_i0_retrieval_anyOf_i0_externalApi_anyOf_i0_authConfig_anyOf_i0_apiKeyConfig_anyOf_i0_apiKeyString_anyOf_i1)

Type: string

Type: null

Type: null

## authType

Default: null

Type of auth scheme.

## Any of

- [AuthType](https://google.github.io/adk-docs/api-reference/agentconfig/#tab-pane_anyOf_i0_generate_content_config_anyOf_i0_tools_anyOf_i0_items_anyOf_i0_retrieval_anyOf_i0_externalApi_anyOf_i0_authConfig_anyOf_i0_authType_anyOf_i0)
- [Option 2](https://google.github.io/adk-docs/api-reference/agentconfig/#tab-pane_anyOf_i0_generate_content_config_anyOf_i0_tools_anyOf_i0_items_anyOf_i0_retrieval_anyOf_i0_externalApi_anyOf_i0_authConfig_anyOf_i0_authType_anyOf_i1)

#### AuthType

Type: enum (of string)

Type of auth scheme.

#### Must be one of:

- "AUTH\_TYPE\_UNSPECIFIED"
- "NO\_AUTH"
- "API\_KEY\_AUTH"
- "HTTP\_BASIC\_AUTH"
- "GOOGLE\_SERVICE\_ACCOUNT\_AUTH"
- "OAUTH"
- "OIDC\_AUTH"

Type: null

## googleServiceAccountConfig

Default: null

Config for Google Service Account auth.

## Any of

- [AuthConfigGoogleServiceAccountConfig](https://google.github.io/adk-docs/api-reference/agentconfig/#tab-pane_anyOf_i0_generate_content_config_anyOf_i0_tools_anyOf_i0_items_anyOf_i0_retrieval_anyOf_i0_externalApi_anyOf_i0_authConfig_anyOf_i0_googleServiceAccountConfig_anyOf_i0)
- [Option 2](https://google.github.io/adk-docs/api-reference/agentconfig/#tab-pane_anyOf_i0_generate_content_config_anyOf_i0_tools_anyOf_i0_items_anyOf_i0_retrieval_anyOf_i0_externalApi_anyOf_i0_authConfig_anyOf_i0_googleServiceAccountConfig_anyOf_i1)

#### AuthConfigGoogleServiceAccountConfig

Type: object

Config for Google Service Account Authentication.
No Additional Properties

## serviceAccount

#### Serviceaccount

Default: null

Optional. The service account that the extension execution service runs as. - If the service account is specified, the `iam.serviceAccounts.getAccessToken` permission should be granted to Vertex AI Extension Service Agent (https://cloud.google.com/vertex-ai/docs/general/access-control#service-agents) on the specified service account. - If not specified, the Vertex AI Extension Service Agent will be used to execute the Extension.

## Any of

- [Option 1](https://google.github.io/adk-docs/api-reference/agentconfig/#tab-pane_anyOf_i0_generate_content_config_anyOf_i0_tools_anyOf_i0_items_anyOf_i0_retrieval_anyOf_i0_externalApi_anyOf_i0_authConfig_anyOf_i0_googleServiceAccountConfig_anyOf_i0_serviceAccount_anyOf_i0)
- [Option 2](https://google.github.io/adk-docs/api-reference/agentconfig/#tab-pane_anyOf_i0_generate_content_config_anyOf_i0_tools_anyOf_i0_items_anyOf_i0_retrieval_anyOf_i0_externalApi_anyOf_i0_authConfig_anyOf_i0_googleServiceAccountConfig_anyOf_i0_serviceAccount_anyOf_i1)

Type: string

Type: null

Type: null

## httpBasicAuthConfig

Default: null

Config for HTTP Basic auth.

## Any of

- [AuthConfigHttpBasicAuthConfig](https://google.github.io/adk-docs/api-reference/agentconfig/#tab-pane_anyOf_i0_generate_content_config_anyOf_i0_tools_anyOf_i0_items_anyOf_i0_retrieval_anyOf_i0_externalApi_anyOf_i0_authConfig_anyOf_i0_httpBasicAuthConfig_anyOf_i0)
- [Option 2](https://google.github.io/adk-docs/api-reference/agentconfig/#tab-pane_anyOf_i0_generate_content_config_anyOf_i0_tools_anyOf_i0_items_anyOf_i0_retrieval_anyOf_i0_externalApi_anyOf_i0_authConfig_anyOf_i0_httpBasicAuthConfig_anyOf_i1)

#### AuthConfigHttpBasicAuthConfig

Type: object

Config for HTTP Basic Authentication.
No Additional Properties

## credentialSecret

#### Credentialsecret

Default: null

Required. The name of the SecretManager secret version resource storing the base64 encoded credentials. Format: `projects/{project}/secrets/{secrete}/versions/{version}` \- If specified, the `secretmanager.versions.access` permission should be granted to Vertex AI Extension Service Agent (https://cloud.google.com/vertex-ai/docs/general/access-control#service-agents) on the specified resource.

## Any of

- [Option 1](https://google.github.io/adk-docs/api-reference/agentconfig/#tab-pane_anyOf_i0_generate_content_config_anyOf_i0_tools_anyOf_i0_items_anyOf_i0_retrieval_anyOf_i0_externalApi_anyOf_i0_authConfig_anyOf_i0_httpBasicAuthConfig_anyOf_i0_credentialSecret_anyOf_i0)
- [Option 2](https://google.github.io/adk-docs/api-reference/agentconfig/#tab-pane_anyOf_i0_generate_content_config_anyOf_i0_tools_anyOf_i0_items_anyOf_i0_retrieval_anyOf_i0_externalApi_anyOf_i0_authConfig_anyOf_i0_httpBasicAuthConfig_anyOf_i0_credentialSecret_anyOf_i1)

Type: string

Type: null

Type: null

## oauthConfig

Default: null

Config for user oauth.

## Any of

- [AuthConfigOauthConfig](https://google.github.io/adk-docs/api-reference/agentconfig/#tab-pane_anyOf_i0_generate_content_config_anyOf_i0_tools_anyOf_i0_items_anyOf_i0_retrieval_anyOf_i0_externalApi_anyOf_i0_authConfig_anyOf_i0_oauthConfig_anyOf_i0)
- [Option 2](https://google.github.io/adk-docs/api-reference/agentconfig/#tab-pane_anyOf_i0_generate_content_config_anyOf_i0_tools_anyOf_i0_items_anyOf_i0_retrieval_anyOf_i0_externalApi_anyOf_i0_authConfig_anyOf_i0_oauthConfig_anyOf_i1)

#### AuthConfigOauthConfig

Type: object

Config for user oauth.
No Additional Properties

## accessToken

#### Accesstoken

Default: null

Access token for extension endpoint. Only used to propagate token from \[\[ExecuteExtensionRequest.runtime _auth_ config\]\] at request time.

## Any of

- [Option 1](https://google.github.io/adk-docs/api-reference/agentconfig/#tab-pane_anyOf_i0_generate_content_config_anyOf_i0_tools_anyOf_i0_items_anyOf_i0_retrieval_anyOf_i0_externalApi_anyOf_i0_authConfig_anyOf_i0_oauthConfig_anyOf_i0_accessToken_anyOf_i0)
- [Option 2](https://google.github.io/adk-docs/api-reference/agentconfig/#tab-pane_anyOf_i0_generate_content_config_anyOf_i0_tools_anyOf_i0_items_anyOf_i0_retrieval_anyOf_i0_externalApi_anyOf_i0_authConfig_anyOf_i0_oauthConfig_anyOf_i0_accessToken_anyOf_i1)

Type: string

Type: null

## serviceAccount

#### Serviceaccount

Default: null

The service account used to generate access tokens for executing the Extension. - If the service account is specified, the `iam.serviceAccounts.getAccessToken` permission should be granted to Vertex AI Extension Service Agent (https://cloud.google.com/vertex-ai/docs/general/access-control#service-agents) on the provided service account.

## Any of

- [Option 1](https://google.github.io/adk-docs/api-reference/agentconfig/#tab-pane_anyOf_i0_generate_content_config_anyOf_i0_tools_anyOf_i0_items_anyOf_i0_retrieval_anyOf_i0_externalApi_anyOf_i0_authConfig_anyOf_i0_oauthConfig_anyOf_i0_serviceAccount_anyOf_i0)
- [Option 2](https://google.github.io/adk-docs/api-reference/agentconfig/#tab-pane_anyOf_i0_generate_content_config_anyOf_i0_tools_anyOf_i0_items_anyOf_i0_retrieval_anyOf_i0_externalApi_anyOf_i0_authConfig_anyOf_i0_oauthConfig_anyOf_i0_serviceAccount_anyOf_i1)

Type: string

Type: null

Type: null

## oidcConfig

Default: null

Config for user OIDC auth.

## Any of

- [AuthConfigOidcConfig](https://google.github.io/adk-docs/api-reference/agentconfig/#tab-pane_anyOf_i0_generate_content_config_anyOf_i0_tools_anyOf_i0_items_anyOf_i0_retrieval_anyOf_i0_externalApi_anyOf_i0_authConfig_anyOf_i0_oidcConfig_anyOf_i0)
- [Option 2](https://google.github.io/adk-docs/api-reference/agentconfig/#tab-pane_anyOf_i0_generate_content_config_anyOf_i0_tools_anyOf_i0_items_anyOf_i0_retrieval_anyOf_i0_externalApi_anyOf_i0_authConfig_anyOf_i0_oidcConfig_anyOf_i1)

#### AuthConfigOidcConfig

Type: object

Config for user OIDC auth.
No Additional Properties

## idToken

#### Idtoken

Default: null

OpenID Connect formatted ID token for extension endpoint. Only used to propagate token from \[\[ExecuteExtensionRequest.runtime _auth_ config\]\] at request time.

## Any of

- [Option 1](https://google.github.io/adk-docs/api-reference/agentconfig/#tab-pane_anyOf_i0_generate_content_config_anyOf_i0_tools_anyOf_i0_items_anyOf_i0_retrieval_anyOf_i0_externalApi_anyOf_i0_authConfig_anyOf_i0_oidcConfig_anyOf_i0_idToken_anyOf_i0)
- [Option 2](https://google.github.io/adk-docs/api-reference/agentconfig/#tab-pane_anyOf_i0_generate_content_config_anyOf_i0_tools_anyOf_i0_items_anyOf_i0_retrieval_anyOf_i0_externalApi_anyOf_i0_authConfig_anyOf_i0_oidcConfig_anyOf_i0_idToken_anyOf_i1)

Type: string

Type: null

## serviceAccount

#### Serviceaccount

Default: null

The service account used to generate an OpenID Connect (OIDC)-compatible JWT token signed by the Google OIDC Provider (accounts.google.com) for extension endpoint (https://cloud.google.com/iam/docs/create-short-lived-credentials-direct#sa-credentials-oidc). - The audience for the token will be set to the URL in the server url defined in the OpenApi spec. - If the service account is provided, the service account should grant `iam.serviceAccounts.getOpenIdToken` permission to Vertex AI Extension Service Agent (https://cloud.google.com/vertex-ai/docs/general/access-control#service-agents).

## Any of

- [Option 1](https://google.github.io/adk-docs/api-reference/agentconfig/#tab-pane_anyOf_i0_generate_content_config_anyOf_i0_tools_anyOf_i0_items_anyOf_i0_retrieval_anyOf_i0_externalApi_anyOf_i0_authConfig_anyOf_i0_oidcConfig_anyOf_i0_serviceAccount_anyOf_i0)
- [Option 2](https://google.github.io/adk-docs/api-reference/agentconfig/#tab-pane_anyOf_i0_generate_content_config_anyOf_i0_tools_anyOf_i0_items_anyOf_i0_retrieval_anyOf_i0_externalApi_anyOf_i0_authConfig_anyOf_i0_oidcConfig_anyOf_i0_serviceAccount_anyOf_i1)

Type: string

Type: null

Type: null

Type: null

## elasticSearchParams

Default: null

Parameters for the elastic search API.

## Any of

- [ExternalApiElasticSearchParams](https://google.github.io/adk-docs/api-reference/agentconfig/#tab-pane_anyOf_i0_generate_content_config_anyOf_i0_tools_anyOf_i0_items_anyOf_i0_retrieval_anyOf_i0_externalApi_anyOf_i0_elasticSearchParams_anyOf_i0)
- [Option 2](https://google.github.io/adk-docs/api-reference/agentconfig/#tab-pane_anyOf_i0_generate_content_config_anyOf_i0_tools_anyOf_i0_items_anyOf_i0_retrieval_anyOf_i0_externalApi_anyOf_i0_elasticSearchParams_anyOf_i1)

#### ExternalApiElasticSearchParams

Type: object

The search parameters to use for the ELASTIC\_SEARCH spec.
No Additional Properties

## index

#### Index

Default: null

The ElasticSearch index to use.

## Any of

- [Option 1](https://google.github.io/adk-docs/api-reference/agentconfig/#tab-pane_anyOf_i0_generate_content_config_anyOf_i0_tools_anyOf_i0_items_anyOf_i0_retrieval_anyOf_i0_externalApi_anyOf_i0_elasticSearchParams_anyOf_i0_index_anyOf_i0)
- [Option 2](https://google.github.io/adk-docs/api-reference/agentconfig/#tab-pane_anyOf_i0_generate_content_config_anyOf_i0_tools_anyOf_i0_items_anyOf_i0_retrieval_anyOf_i0_externalApi_anyOf_i0_elasticSearchParams_anyOf_i0_index_anyOf_i1)

Type: string

Type: null

## numHits

#### Numhits

Default: null

Optional. Number of hits (chunks) to request. When specified, it is passed to Elasticsearch as the `num_hits` param.

## Any of

- [Option 1](https://google.github.io/adk-docs/api-reference/agentconfig/#tab-pane_anyOf_i0_generate_content_config_anyOf_i0_tools_anyOf_i0_items_anyOf_i0_retrieval_anyOf_i0_externalApi_anyOf_i0_elasticSearchParams_anyOf_i0_numHits_anyOf_i0)
- [Option 2](https://google.github.io/adk-docs/api-reference/agentconfig/#tab-pane_anyOf_i0_generate_content_config_anyOf_i0_tools_anyOf_i0_items_anyOf_i0_retrieval_anyOf_i0_externalApi_anyOf_i0_elasticSearchParams_anyOf_i0_numHits_anyOf_i1)

Type: integer

Type: null

## searchTemplate

#### Searchtemplate

Default: null

The ElasticSearch search template to use.

## Any of

- [Option 1](https://google.github.io/adk-docs/api-reference/agentconfig/#tab-pane_anyOf_i0_generate_content_config_anyOf_i0_tools_anyOf_i0_items_anyOf_i0_retrieval_anyOf_i0_externalApi_anyOf_i0_elasticSearchParams_anyOf_i0_searchTemplate_anyOf_i0)
- [Option 2](https://google.github.io/adk-docs/api-reference/agentconfig/#tab-pane_anyOf_i0_generate_content_config_anyOf_i0_tools_anyOf_i0_items_anyOf_i0_retrieval_anyOf_i0_externalApi_anyOf_i0_elasticSearchParams_anyOf_i0_searchTemplate_anyOf_i1)

Type: string

Type: null

Type: null

## endpoint

#### Endpoint

Default: null

The endpoint of the external API. The system will call the API at this endpoint to retrieve the data for grounding. Example: https://acme.com:443/search

## Any of

- [Option 1](https://google.github.io/adk-docs/api-reference/agentconfig/#tab-pane_anyOf_i0_generate_content_config_anyOf_i0_tools_anyOf_i0_items_anyOf_i0_retrieval_anyOf_i0_externalApi_anyOf_i0_endpoint_anyOf_i0)
- [Option 2](https://google.github.io/adk-docs/api-reference/agentconfig/#tab-pane_anyOf_i0_generate_content_config_anyOf_i0_tools_anyOf_i0_items_anyOf_i0_retrieval_anyOf_i0_externalApi_anyOf_i0_endpoint_anyOf_i1)

Type: string

Type: null

## simpleSearchParams

Default: null

Parameters for the simple search API.

## Any of

- [ExternalApiSimpleSearchParams](https://google.github.io/adk-docs/api-reference/agentconfig/#tab-pane_anyOf_i0_generate_content_config_anyOf_i0_tools_anyOf_i0_items_anyOf_i0_retrieval_anyOf_i0_externalApi_anyOf_i0_simpleSearchParams_anyOf_i0)
- [Option 2](https://google.github.io/adk-docs/api-reference/agentconfig/#tab-pane_anyOf_i0_generate_content_config_anyOf_i0_tools_anyOf_i0_items_anyOf_i0_retrieval_anyOf_i0_externalApi_anyOf_i0_simpleSearchParams_anyOf_i1)

#### ExternalApiSimpleSearchParams

Type: object

The search parameters to use for SIMPLE\_SEARCH spec.

Type: null

Type: null

## vertexAiSearch

Default: null

Set to use data source powered by Vertex AI Search.

## Any of

- [VertexAISearch](https://google.github.io/adk-docs/api-reference/agentconfig/#tab-pane_anyOf_i0_generate_content_config_anyOf_i0_tools_anyOf_i0_items_anyOf_i0_retrieval_anyOf_i0_vertexAiSearch_anyOf_i0)
- [Option 2](https://google.github.io/adk-docs/api-reference/agentconfig/#tab-pane_anyOf_i0_generate_content_config_anyOf_i0_tools_anyOf_i0_items_anyOf_i0_retrieval_anyOf_i0_vertexAiSearch_anyOf_i1)

#### VertexAISearch

Type: object

Retrieve from Vertex AI Search datastore or engine for grounding.

datastore and engine are mutually exclusive. See

https://cloud.google.com/products/agent-builder
No Additional Properties

## dataStoreSpecs

#### Datastorespecs

Default: null

Specifications that define the specific DataStores to be searched, along with configurations for those data stores. This is only considered for Engines with multiple data stores. It should only be set if engine is used.

## Any of

- [Option 1](https://google.github.io/adk-docs/api-reference/agentconfig/#tab-pane_anyOf_i0_generate_content_config_anyOf_i0_tools_anyOf_i0_items_anyOf_i0_retrieval_anyOf_i0_vertexAiSearch_anyOf_i0_dataStoreSpecs_anyOf_i0)
- [Option 2](https://google.github.io/adk-docs/api-reference/agentconfig/#tab-pane_anyOf_i0_generate_content_config_anyOf_i0_tools_anyOf_i0_items_anyOf_i0_retrieval_anyOf_i0_vertexAiSearch_anyOf_i0_dataStoreSpecs_anyOf_i1)

Type: array

No Additional Items

#### Each item of this array must be:

#### VertexAISearchDataStoreSpec

Type: object

Define data stores within engine to filter on in a search call and configurations for those data stores.

For more information, see

https://cloud.google.com/generative-ai-app-builder/docs/reference/rpc/google.cloud.discoveryengine.v1#datastorespec
No Additional Properties

## dataStore

#### Datastore

Default: null

Full resource name of DataStore, such as Format: `projects/{project}/locations/{location}/collections/{collection}/dataStores/{dataStore}`

## Any of

- [Option 1](https://google.github.io/adk-docs/api-reference/agentconfig/#tab-pane_anyOf_i0_generate_content_config_anyOf_i0_tools_anyOf_i0_items_anyOf_i0_retrieval_anyOf_i0_vertexAiSearch_anyOf_i0_dataStoreSpecs_anyOf_i0_items_dataStore_anyOf_i0)
- [Option 2](https://google.github.io/adk-docs/api-reference/agentconfig/#tab-pane_anyOf_i0_generate_content_config_anyOf_i0_tools_anyOf_i0_items_anyOf_i0_retrieval_anyOf_i0_vertexAiSearch_anyOf_i0_dataStoreSpecs_anyOf_i0_items_dataStore_anyOf_i1)

Type: string

Type: null

## filter

#### Filter

Default: null

Optional. Filter specification to filter documents in the data store specified by data\_store field. For more information on filtering, see [Filtering](https://cloud.google.com/generative-ai-app-builder/docs/filter-search-metadata)

## Any of

- [Option 1](https://google.github.io/adk-docs/api-reference/agentconfig/#tab-pane_anyOf_i0_generate_content_config_anyOf_i0_tools_anyOf_i0_items_anyOf_i0_retrieval_anyOf_i0_vertexAiSearch_anyOf_i0_dataStoreSpecs_anyOf_i0_items_filter_anyOf_i0)
- [Option 2](https://google.github.io/adk-docs/api-reference/agentconfig/#tab-pane_anyOf_i0_generate_content_config_anyOf_i0_tools_anyOf_i0_items_anyOf_i0_retrieval_anyOf_i0_vertexAiSearch_anyOf_i0_dataStoreSpecs_anyOf_i0_items_filter_anyOf_i1)

Type: string

Type: null

Type: null

## datastore

#### Datastore

Default: null

Optional. Fully-qualified Vertex AI Search data store resource ID. Format: `projects/{project}/locations/{location}/collections/{collection}/dataStores/{dataStore}`

## Any of

- [Option 1](https://google.github.io/adk-docs/api-reference/agentconfig/#tab-pane_anyOf_i0_generate_content_config_anyOf_i0_tools_anyOf_i0_items_anyOf_i0_retrieval_anyOf_i0_vertexAiSearch_anyOf_i0_datastore_anyOf_i0)
- [Option 2](https://google.github.io/adk-docs/api-reference/agentconfig/#tab-pane_anyOf_i0_generate_content_config_anyOf_i0_tools_anyOf_i0_items_anyOf_i0_retrieval_anyOf_i0_vertexAiSearch_anyOf_i0_datastore_anyOf_i1)

Type: string

Type: null

## engine

#### Engine

Default: null

Optional. Fully-qualified Vertex AI Search engine resource ID. Format: `projects/{project}/locations/{location}/collections/{collection}/engines/{engine}`

## Any of

- [Option 1](https://google.github.io/adk-docs/api-reference/agentconfig/#tab-pane_anyOf_i0_generate_content_config_anyOf_i0_tools_anyOf_i0_items_anyOf_i0_retrieval_anyOf_i0_vertexAiSearch_anyOf_i0_engine_anyOf_i0)
- [Option 2](https://google.github.io/adk-docs/api-reference/agentconfig/#tab-pane_anyOf_i0_generate_content_config_anyOf_i0_tools_anyOf_i0_items_anyOf_i0_retrieval_anyOf_i0_vertexAiSearch_anyOf_i0_engine_anyOf_i1)

Type: string

Type: null

## filter

#### Filter

Default: null

Optional. Filter strings to be passed to the search API.

## Any of

- [Option 1](https://google.github.io/adk-docs/api-reference/agentconfig/#tab-pane_anyOf_i0_generate_content_config_anyOf_i0_tools_anyOf_i0_items_anyOf_i0_retrieval_anyOf_i0_vertexAiSearch_anyOf_i0_filter_anyOf_i0)
- [Option 2](https://google.github.io/adk-docs/api-reference/agentconfig/#tab-pane_anyOf_i0_generate_content_config_anyOf_i0_tools_anyOf_i0_items_anyOf_i0_retrieval_anyOf_i0_vertexAiSearch_anyOf_i0_filter_anyOf_i1)

Type: string

Type: null

## maxResults

#### Maxresults

Default: null

Optional. Number of search results to return per query. The default value is 10. The maximumm allowed value is 10.

## Any of

- [Option 1](https://google.github.io/adk-docs/api-reference/agentconfig/#tab-pane_anyOf_i0_generate_content_config_anyOf_i0_tools_anyOf_i0_items_anyOf_i0_retrieval_anyOf_i0_vertexAiSearch_anyOf_i0_maxResults_anyOf_i0)
- [Option 2](https://google.github.io/adk-docs/api-reference/agentconfig/#tab-pane_anyOf_i0_generate_content_config_anyOf_i0_tools_anyOf_i0_items_anyOf_i0_retrieval_anyOf_i0_vertexAiSearch_anyOf_i0_maxResults_anyOf_i1)

Type: integer

Type: null

Type: null

## vertexRagStore

Default: null

Set to use data source powered by Vertex RAG store. User data is uploaded via the VertexRagDataService.

## Any of

- [VertexRagStore](https://google.github.io/adk-docs/api-reference/agentconfig/#tab-pane_anyOf_i0_generate_content_config_anyOf_i0_tools_anyOf_i0_items_anyOf_i0_retrieval_anyOf_i0_vertexRagStore_anyOf_i0)
- [Option 2](https://google.github.io/adk-docs/api-reference/agentconfig/#tab-pane_anyOf_i0_generate_content_config_anyOf_i0_tools_anyOf_i0_items_anyOf_i0_retrieval_anyOf_i0_vertexRagStore_anyOf_i1)

#### VertexRagStore

Type: object

Retrieve from Vertex RAG Store for grounding.
No Additional Properties

## ragCorpora

#### Ragcorpora

Default: null

Optional. Deprecated. Please use rag\_resources instead.

## Any of

- [Option 1](https://google.github.io/adk-docs/api-reference/agentconfig/#tab-pane_anyOf_i0_generate_content_config_anyOf_i0_tools_anyOf_i0_items_anyOf_i0_retrieval_anyOf_i0_vertexRagStore_anyOf_i0_ragCorpora_anyOf_i0)
- [Option 2](https://google.github.io/adk-docs/api-reference/agentconfig/#tab-pane_anyOf_i0_generate_content_config_anyOf_i0_tools_anyOf_i0_items_anyOf_i0_retrieval_anyOf_i0_vertexRagStore_anyOf_i0_ragCorpora_anyOf_i1)

Type: array of string

No Additional Items

#### Each item of this array must be:

Type: string

Type: null

## ragResources

#### Ragresources

Default: null

Optional. The representation of the rag source. It can be used to specify corpus only or ragfiles. Currently only support one corpus or multiple files from one corpus. In the future we may open up multiple corpora support.

## Any of

- [Option 1](https://google.github.io/adk-docs/api-reference/agentconfig/#tab-pane_anyOf_i0_generate_content_config_anyOf_i0_tools_anyOf_i0_items_anyOf_i0_retrieval_anyOf_i0_vertexRagStore_anyOf_i0_ragResources_anyOf_i0)
- [Option 2](https://google.github.io/adk-docs/api-reference/agentconfig/#tab-pane_anyOf_i0_generate_content_config_anyOf_i0_tools_anyOf_i0_items_anyOf_i0_retrieval_anyOf_i0_vertexRagStore_anyOf_i0_ragResources_anyOf_i1)

Type: array

No Additional Items

#### Each item of this array must be:

#### VertexRagStoreRagResource

Type: object

The definition of the Rag resource.
No Additional Properties

## ragCorpus

#### Ragcorpus

Default: null

Optional. RagCorpora resource name. Format: `projects/{project}/locations/{location}/ragCorpora/{rag_corpus}`

## Any of

- [Option 1](https://google.github.io/adk-docs/api-reference/agentconfig/#tab-pane_anyOf_i0_generate_content_config_anyOf_i0_tools_anyOf_i0_items_anyOf_i0_retrieval_anyOf_i0_vertexRagStore_anyOf_i0_ragResources_anyOf_i0_items_ragCorpus_anyOf_i0)
- [Option 2](https://google.github.io/adk-docs/api-reference/agentconfig/#tab-pane_anyOf_i0_generate_content_config_anyOf_i0_tools_anyOf_i0_items_anyOf_i0_retrieval_anyOf_i0_vertexRagStore_anyOf_i0_ragResources_anyOf_i0_items_ragCorpus_anyOf_i1)

Type: string

Type: null

## ragFileIds

#### Ragfileids

Default: null

Optional. rag _file_ id. The files should be in the same rag _corpus set in rag_ corpus field.

## Any of

- [Option 1](https://google.github.io/adk-docs/api-reference/agentconfig/#tab-pane_anyOf_i0_generate_content_config_anyOf_i0_tools_anyOf_i0_items_anyOf_i0_retrieval_anyOf_i0_vertexRagStore_anyOf_i0_ragResources_anyOf_i0_items_ragFileIds_anyOf_i0)
- [Option 2](https://google.github.io/adk-docs/api-reference/agentconfig/#tab-pane_anyOf_i0_generate_content_config_anyOf_i0_tools_anyOf_i0_items_anyOf_i0_retrieval_anyOf_i0_vertexRagStore_anyOf_i0_ragResources_anyOf_i0_items_ragFileIds_anyOf_i1)

Type: array of string

No Additional Items

#### Each item of this array must be:

Type: string

Type: null

Type: null

## ragRetrievalConfig

Default: null

Optional. The retrieval config for the Rag query.

## Any of

- [RagRetrievalConfig](https://google.github.io/adk-docs/api-reference/agentconfig/#tab-pane_anyOf_i0_generate_content_config_anyOf_i0_tools_anyOf_i0_items_anyOf_i0_retrieval_anyOf_i0_vertexRagStore_anyOf_i0_ragRetrievalConfig_anyOf_i0)
- [Option 2](https://google.github.io/adk-docs/api-reference/agentconfig/#tab-pane_anyOf_i0_generate_content_config_anyOf_i0_tools_anyOf_i0_items_anyOf_i0_retrieval_anyOf_i0_vertexRagStore_anyOf_i0_ragRetrievalConfig_anyOf_i1)

#### RagRetrievalConfig

Type: object

Specifies the context retrieval config.
No Additional Properties

## filter

Default: null

Optional. Config for filters.

## Any of

- [RagRetrievalConfigFilter](https://google.github.io/adk-docs/api-reference/agentconfig/#tab-pane_anyOf_i0_generate_content_config_anyOf_i0_tools_anyOf_i0_items_anyOf_i0_retrieval_anyOf_i0_vertexRagStore_anyOf_i0_ragRetrievalConfig_anyOf_i0_filter_anyOf_i0)
- [Option 2](https://google.github.io/adk-docs/api-reference/agentconfig/#tab-pane_anyOf_i0_generate_content_config_anyOf_i0_tools_anyOf_i0_items_anyOf_i0_retrieval_anyOf_i0_vertexRagStore_anyOf_i0_ragRetrievalConfig_anyOf_i0_filter_anyOf_i1)

#### RagRetrievalConfigFilter

Type: object

Config for filters.
No Additional Properties

## metadataFilter

#### Metadatafilter

Default: null

Optional. String for metadata filtering.

## Any of

- [Option 1](https://google.github.io/adk-docs/api-reference/agentconfig/#tab-pane_anyOf_i0_generate_content_config_anyOf_i0_tools_anyOf_i0_items_anyOf_i0_retrieval_anyOf_i0_vertexRagStore_anyOf_i0_ragRetrievalConfig_anyOf_i0_filter_anyOf_i0_metadataFilter_anyOf_i0)
- [Option 2](https://google.github.io/adk-docs/api-reference/agentconfig/#tab-pane_anyOf_i0_generate_content_config_anyOf_i0_tools_anyOf_i0_items_anyOf_i0_retrieval_anyOf_i0_vertexRagStore_anyOf_i0_ragRetrievalConfig_anyOf_i0_filter_anyOf_i0_metadataFilter_anyOf_i1)

Type: string

Type: null

## vectorDistanceThreshold

#### Vectordistancethreshold

Default: null

Optional. Only returns contexts with vector distance smaller than the threshold.

## Any of

- [Option 1](https://google.github.io/adk-docs/api-reference/agentconfig/#tab-pane_anyOf_i0_generate_content_config_anyOf_i0_tools_anyOf_i0_items_anyOf_i0_retrieval_anyOf_i0_vertexRagStore_anyOf_i0_ragRetrievalConfig_anyOf_i0_filter_anyOf_i0_vectorDistanceThreshold_anyOf_i0)
- [Option 2](https://google.github.io/adk-docs/api-reference/agentconfig/#tab-pane_anyOf_i0_generate_content_config_anyOf_i0_tools_anyOf_i0_items_anyOf_i0_retrieval_anyOf_i0_vertexRagStore_anyOf_i0_ragRetrievalConfig_anyOf_i0_filter_anyOf_i0_vectorDistanceThreshold_anyOf_i1)

Type: number

Type: null

## vectorSimilarityThreshold

#### Vectorsimilaritythreshold

Default: null

Optional. Only returns contexts with vector similarity larger than the threshold.

## Any of

- [Option 1](https://google.github.io/adk-docs/api-reference/agentconfig/#tab-pane_anyOf_i0_generate_content_config_anyOf_i0_tools_anyOf_i0_items_anyOf_i0_retrieval_anyOf_i0_vertexRagStore_anyOf_i0_ragRetrievalConfig_anyOf_i0_filter_anyOf_i0_vectorSimilarityThreshold_anyOf_i0)
- [Option 2](https://google.github.io/adk-docs/api-reference/agentconfig/#tab-pane_anyOf_i0_generate_content_config_anyOf_i0_tools_anyOf_i0_items_anyOf_i0_retrieval_anyOf_i0_vertexRagStore_anyOf_i0_ragRetrievalConfig_anyOf_i0_filter_anyOf_i0_vectorSimilarityThreshold_anyOf_i1)

Type: number

Type: null

Type: null

## hybridSearch

Default: null

Optional. Config for Hybrid Search.

## Any of

- [RagRetrievalConfigHybridSearch](https://google.github.io/adk-docs/api-reference/agentconfig/#tab-pane_anyOf_i0_generate_content_config_anyOf_i0_tools_anyOf_i0_items_anyOf_i0_retrieval_anyOf_i0_vertexRagStore_anyOf_i0_ragRetrievalConfig_anyOf_i0_hybridSearch_anyOf_i0)
- [Option 2](https://google.github.io/adk-docs/api-reference/agentconfig/#tab-pane_anyOf_i0_generate_content_config_anyOf_i0_tools_anyOf_i0_items_anyOf_i0_retrieval_anyOf_i0_vertexRagStore_anyOf_i0_ragRetrievalConfig_anyOf_i0_hybridSearch_anyOf_i1)

#### RagRetrievalConfigHybridSearch

Type: object

Config for Hybrid Search.
No Additional Properties

## alpha

#### Alpha

Default: null

Optional. Alpha value controls the weight between dense and sparse vector search results. The range is \[0, 1\], while 0 means sparse vector search only and 1 means dense vector search only. The default value is 0.5 which balances sparse and dense vector search equally.

## Any of

- [Option 1](https://google.github.io/adk-docs/api-reference/agentconfig/#tab-pane_anyOf_i0_generate_content_config_anyOf_i0_tools_anyOf_i0_items_anyOf_i0_retrieval_anyOf_i0_vertexRagStore_anyOf_i0_ragRetrievalConfig_anyOf_i0_hybridSearch_anyOf_i0_alpha_anyOf_i0)
- [Option 2](https://google.github.io/adk-docs/api-reference/agentconfig/#tab-pane_anyOf_i0_generate_content_config_anyOf_i0_tools_anyOf_i0_items_anyOf_i0_retrieval_anyOf_i0_vertexRagStore_anyOf_i0_ragRetrievalConfig_anyOf_i0_hybridSearch_anyOf_i0_alpha_anyOf_i1)

Type: number

Type: null

Type: null

## ranking

Default: null

Optional. Config for ranking and reranking.

## Any of

- [RagRetrievalConfigRanking](https://google.github.io/adk-docs/api-reference/agentconfig/#tab-pane_anyOf_i0_generate_content_config_anyOf_i0_tools_anyOf_i0_items_anyOf_i0_retrieval_anyOf_i0_vertexRagStore_anyOf_i0_ragRetrievalConfig_anyOf_i0_ranking_anyOf_i0)
- [Option 2](https://google.github.io/adk-docs/api-reference/agentconfig/#tab-pane_anyOf_i0_generate_content_config_anyOf_i0_tools_anyOf_i0_items_anyOf_i0_retrieval_anyOf_i0_vertexRagStore_anyOf_i0_ragRetrievalConfig_anyOf_i0_ranking_anyOf_i1)

#### RagRetrievalConfigRanking

Type: object

Config for ranking and reranking.
No Additional Properties

## llmRanker

Default: null

Optional. Config for LlmRanker.

## Any of

- [RagRetrievalConfigRankingLlmRanker](https://google.github.io/adk-docs/api-reference/agentconfig/#tab-pane_anyOf_i0_generate_content_config_anyOf_i0_tools_anyOf_i0_items_anyOf_i0_retrieval_anyOf_i0_vertexRagStore_anyOf_i0_ragRetrievalConfig_anyOf_i0_ranking_anyOf_i0_llmRanker_anyOf_i0)
- [Option 2](https://google.github.io/adk-docs/api-reference/agentconfig/#tab-pane_anyOf_i0_generate_content_config_anyOf_i0_tools_anyOf_i0_items_anyOf_i0_retrieval_anyOf_i0_vertexRagStore_anyOf_i0_ragRetrievalConfig_anyOf_i0_ranking_anyOf_i0_llmRanker_anyOf_i1)

#### RagRetrievalConfigRankingLlmRanker

Type: object

Config for LlmRanker.
No Additional Properties

## modelName

#### Modelname

Default: null

Optional. The model name used for ranking. See [Supported models](https://cloud.google.com/vertex-ai/generative-ai/docs/model-reference/inference#supported-models).

## Any of

- [Option 1](https://google.github.io/adk-docs/api-reference/agentconfig/#tab-pane_anyOf_i0_generate_content_config_anyOf_i0_tools_anyOf_i0_items_anyOf_i0_retrieval_anyOf_i0_vertexRagStore_anyOf_i0_ragRetrievalConfig_anyOf_i0_ranking_anyOf_i0_llmRanker_anyOf_i0_modelName_anyOf_i0)
- [Option 2](https://google.github.io/adk-docs/api-reference/agentconfig/#tab-pane_anyOf_i0_generate_content_config_anyOf_i0_tools_anyOf_i0_items_anyOf_i0_retrieval_anyOf_i0_vertexRagStore_anyOf_i0_ragRetrievalConfig_anyOf_i0_ranking_anyOf_i0_llmRanker_anyOf_i0_modelName_anyOf_i1)

Type: string

Type: null

Type: null

## rankService

Default: null

Optional. Config for Rank Service.

## Any of

- [RagRetrievalConfigRankingRankService](https://google.github.io/adk-docs/api-reference/agentconfig/#tab-pane_anyOf_i0_generate_content_config_anyOf_i0_tools_anyOf_i0_items_anyOf_i0_retrieval_anyOf_i0_vertexRagStore_anyOf_i0_ragRetrievalConfig_anyOf_i0_ranking_anyOf_i0_rankService_anyOf_i0)
- [Option 2](https://google.github.io/adk-docs/api-reference/agentconfig/#tab-pane_anyOf_i0_generate_content_config_anyOf_i0_tools_anyOf_i0_items_anyOf_i0_retrieval_anyOf_i0_vertexRagStore_anyOf_i0_ragRetrievalConfig_anyOf_i0_ranking_anyOf_i0_rankService_anyOf_i1)

#### RagRetrievalConfigRankingRankService

Type: object

Config for Rank Service.
No Additional Properties

## modelName

#### Modelname

Default: null

Optional. The model name of the rank service. Format: `semantic-ranker-512@latest`

## Any of

- [Option 1](https://google.github.io/adk-docs/api-reference/agentconfig/#tab-pane_anyOf_i0_generate_content_config_anyOf_i0_tools_anyOf_i0_items_anyOf_i0_retrieval_anyOf_i0_vertexRagStore_anyOf_i0_ragRetrievalConfig_anyOf_i0_ranking_anyOf_i0_rankService_anyOf_i0_modelName_anyOf_i0)
- [Option 2](https://google.github.io/adk-docs/api-reference/agentconfig/#tab-pane_anyOf_i0_generate_content_config_anyOf_i0_tools_anyOf_i0_items_anyOf_i0_retrieval_anyOf_i0_vertexRagStore_anyOf_i0_ragRetrievalConfig_anyOf_i0_ranking_anyOf_i0_rankService_anyOf_i0_modelName_anyOf_i1)

Type: string

Type: null

Type: null

Type: null

## topK

#### Topk

Default: null

Optional. The number of contexts to retrieve.

## Any of

- [Option 1](https://google.github.io/adk-docs/api-reference/agentconfig/#tab-pane_anyOf_i0_generate_content_config_anyOf_i0_tools_anyOf_i0_items_anyOf_i0_retrieval_anyOf_i0_vertexRagStore_anyOf_i0_ragRetrievalConfig_anyOf_i0_topK_anyOf_i0)
- [Option 2](https://google.github.io/adk-docs/api-reference/agentconfig/#tab-pane_anyOf_i0_generate_content_config_anyOf_i0_tools_anyOf_i0_items_anyOf_i0_retrieval_anyOf_i0_vertexRagStore_anyOf_i0_ragRetrievalConfig_anyOf_i0_topK_anyOf_i1)

Type: integer

Type: null

Type: null

## similarityTopK

#### Similaritytopk

Default: null

Optional. Number of top k results to return from the selected corpora.

## Any of

- [Option 1](https://google.github.io/adk-docs/api-reference/agentconfig/#tab-pane_anyOf_i0_generate_content_config_anyOf_i0_tools_anyOf_i0_items_anyOf_i0_retrieval_anyOf_i0_vertexRagStore_anyOf_i0_similarityTopK_anyOf_i0)
- [Option 2](https://google.github.io/adk-docs/api-reference/agentconfig/#tab-pane_anyOf_i0_generate_content_config_anyOf_i0_tools_anyOf_i0_items_anyOf_i0_retrieval_anyOf_i0_vertexRagStore_anyOf_i0_similarityTopK_anyOf_i1)

Type: integer

Type: null

## storeContext

#### Storecontext

Default: null

Optional. Currently only supported for Gemini Multimodal Live API. In Gemini Multimodal Live API, if `store_context` bool is specified, Gemini will leverage it to automatically memorize the interactions between the client and Gemini, and retrieve context when needed to augment the response generation for users' ongoing and future interactions.

## Any of

- [Option 1](https://google.github.io/adk-docs/api-reference/agentconfig/#tab-pane_anyOf_i0_generate_content_config_anyOf_i0_tools_anyOf_i0_items_anyOf_i0_retrieval_anyOf_i0_vertexRagStore_anyOf_i0_storeContext_anyOf_i0)
- [Option 2](https://google.github.io/adk-docs/api-reference/agentconfig/#tab-pane_anyOf_i0_generate_content_config_anyOf_i0_tools_anyOf_i0_items_anyOf_i0_retrieval_anyOf_i0_vertexRagStore_anyOf_i0_storeContext_anyOf_i1)

Type: boolean

Type: null

## vectorDistanceThreshold

#### Vectordistancethreshold

Default: null

Optional. Only return results with vector distance smaller than the threshold.

## Any of

- [Option 1](https://google.github.io/adk-docs/api-reference/agentconfig/#tab-pane_anyOf_i0_generate_content_config_anyOf_i0_tools_anyOf_i0_items_anyOf_i0_retrieval_anyOf_i0_vertexRagStore_anyOf_i0_vectorDistanceThreshold_anyOf_i0)
- [Option 2](https://google.github.io/adk-docs/api-reference/agentconfig/#tab-pane_anyOf_i0_generate_content_config_anyOf_i0_tools_anyOf_i0_items_anyOf_i0_retrieval_anyOf_i0_vertexRagStore_anyOf_i0_vectorDistanceThreshold_anyOf_i1)

Type: number

Type: null

Type: null

Type: null

## googleSearch

Default: null

Optional. Google Search tool type. Specialized retrieval tool

that is powered by Google Search.

## Any of

- [GoogleSearch](https://google.github.io/adk-docs/api-reference/agentconfig/#tab-pane_anyOf_i0_generate_content_config_anyOf_i0_tools_anyOf_i0_items_anyOf_i0_googleSearch_anyOf_i0)
- [Option 2](https://google.github.io/adk-docs/api-reference/agentconfig/#tab-pane_anyOf_i0_generate_content_config_anyOf_i0_tools_anyOf_i0_items_anyOf_i0_googleSearch_anyOf_i1)

#### GoogleSearch

Type: object

Tool to support Google Search in Model. Powered by Google.
No Additional Properties

## timeRangeFilter

Default: null

Optional. Filter search results to a specific time range.

If customers set a start time, they must set an end time (and vice versa).

## Any of

- [Interval](https://google.github.io/adk-docs/api-reference/agentconfig/#tab-pane_anyOf_i0_generate_content_config_anyOf_i0_tools_anyOf_i0_items_anyOf_i0_googleSearch_anyOf_i0_timeRangeFilter_anyOf_i0)
- [Option 2](https://google.github.io/adk-docs/api-reference/agentconfig/#tab-pane_anyOf_i0_generate_content_config_anyOf_i0_tools_anyOf_i0_items_anyOf_i0_googleSearch_anyOf_i0_timeRangeFilter_anyOf_i1)

#### Interval

Type: object

Represents a time interval, encoded as a start time (inclusive) and an end time (exclusive).

The start time must be less than or equal to the end time.

When the start equals the end time, the interval is an empty interval.

(matches no time)

When both start and end are unspecified, the interval matches any time.

No Additional Properties

## startTime

#### Starttime

Default: null

The start time of the interval.

## Any of

- [Option 1](https://google.github.io/adk-docs/api-reference/agentconfig/#tab-pane_anyOf_i0_generate_content_config_anyOf_i0_tools_anyOf_i0_items_anyOf_i0_googleSearch_anyOf_i0_timeRangeFilter_anyOf_i0_startTime_anyOf_i0)
- [Option 2](https://google.github.io/adk-docs/api-reference/agentconfig/#tab-pane_anyOf_i0_generate_content_config_anyOf_i0_tools_anyOf_i0_items_anyOf_i0_googleSearch_anyOf_i0_timeRangeFilter_anyOf_i0_startTime_anyOf_i1)

Type: stringFormat: date-time

Type: null

## endTime

#### Endtime

Default: null

The end time of the interval.

## Any of

- [Option 1](https://google.github.io/adk-docs/api-reference/agentconfig/#tab-pane_anyOf_i0_generate_content_config_anyOf_i0_tools_anyOf_i0_items_anyOf_i0_googleSearch_anyOf_i0_timeRangeFilter_anyOf_i0_endTime_anyOf_i0)
- [Option 2](https://google.github.io/adk-docs/api-reference/agentconfig/#tab-pane_anyOf_i0_generate_content_config_anyOf_i0_tools_anyOf_i0_items_anyOf_i0_googleSearch_anyOf_i0_timeRangeFilter_anyOf_i0_endTime_anyOf_i1)

Type: stringFormat: date-time

Type: null

Type: null

Type: null

## googleSearchRetrieval

Default: null

Optional. GoogleSearchRetrieval tool type. Specialized retrieval tool that is powered by Google search.

## Any of

- [GoogleSearchRetrieval](https://google.github.io/adk-docs/api-reference/agentconfig/#tab-pane_anyOf_i0_generate_content_config_anyOf_i0_tools_anyOf_i0_items_anyOf_i0_googleSearchRetrieval_anyOf_i0)
- [Option 2](https://google.github.io/adk-docs/api-reference/agentconfig/#tab-pane_anyOf_i0_generate_content_config_anyOf_i0_tools_anyOf_i0_items_anyOf_i0_googleSearchRetrieval_anyOf_i1)

#### GoogleSearchRetrieval

Type: object

Tool to retrieve public web data for grounding, powered by Google.
No Additional Properties

## dynamicRetrievalConfig

Default: null

Specifies the dynamic retrieval configuration for the given source.

## Any of

- [DynamicRetrievalConfig](https://google.github.io/adk-docs/api-reference/agentconfig/#tab-pane_anyOf_i0_generate_content_config_anyOf_i0_tools_anyOf_i0_items_anyOf_i0_googleSearchRetrieval_anyOf_i0_dynamicRetrievalConfig_anyOf_i0)
- [Option 2](https://google.github.io/adk-docs/api-reference/agentconfig/#tab-pane_anyOf_i0_generate_content_config_anyOf_i0_tools_anyOf_i0_items_anyOf_i0_googleSearchRetrieval_anyOf_i0_dynamicRetrievalConfig_anyOf_i1)

#### DynamicRetrievalConfig

Type: object

Describes the options to customize dynamic retrieval.
No Additional Properties

## mode

Default: null

The mode of the predictor to be used in dynamic retrieval.

## Any of

- [DynamicRetrievalConfigMode](https://google.github.io/adk-docs/api-reference/agentconfig/#tab-pane_anyOf_i0_generate_content_config_anyOf_i0_tools_anyOf_i0_items_anyOf_i0_googleSearchRetrieval_anyOf_i0_dynamicRetrievalConfig_anyOf_i0_mode_anyOf_i0)
- [Option 2](https://google.github.io/adk-docs/api-reference/agentconfig/#tab-pane_anyOf_i0_generate_content_config_anyOf_i0_tools_anyOf_i0_items_anyOf_i0_googleSearchRetrieval_anyOf_i0_dynamicRetrievalConfig_anyOf_i0_mode_anyOf_i1)

#### DynamicRetrievalConfigMode

Type: enum (of string)

Config for the dynamic retrieval config mode.

#### Must be one of:

- "MODE\_UNSPECIFIED"
- "MODE\_DYNAMIC"

Type: null

## dynamicThreshold

#### Dynamicthreshold

Default: null

Optional. The threshold to be used in dynamic retrieval. If not set, a system default value is used.

## Any of

- [Option 1](https://google.github.io/adk-docs/api-reference/agentconfig/#tab-pane_anyOf_i0_generate_content_config_anyOf_i0_tools_anyOf_i0_items_anyOf_i0_googleSearchRetrieval_anyOf_i0_dynamicRetrievalConfig_anyOf_i0_dynamicThreshold_anyOf_i0)
- [Option 2](https://google.github.io/adk-docs/api-reference/agentconfig/#tab-pane_anyOf_i0_generate_content_config_anyOf_i0_tools_anyOf_i0_items_anyOf_i0_googleSearchRetrieval_anyOf_i0_dynamicRetrievalConfig_anyOf_i0_dynamicThreshold_anyOf_i1)

Type: number

Type: null

Type: null

Type: null

## enterpriseWebSearch

Default: null

Optional. Enterprise web search tool type. Specialized retrieval

tool that is powered by Vertex AI Search and Sec4 compliance.

## Any of

- [EnterpriseWebSearch](https://google.github.io/adk-docs/api-reference/agentconfig/#tab-pane_anyOf_i0_generate_content_config_anyOf_i0_tools_anyOf_i0_items_anyOf_i0_enterpriseWebSearch_anyOf_i0)
- [Option 2](https://google.github.io/adk-docs/api-reference/agentconfig/#tab-pane_anyOf_i0_generate_content_config_anyOf_i0_tools_anyOf_i0_items_anyOf_i0_enterpriseWebSearch_anyOf_i1)

#### EnterpriseWebSearch

Type: object

Tool to search public web data, powered by Vertex AI Search and Sec4 compliance.

Type: null

## googleMaps

Default: null

Optional. Google Maps tool type. Specialized retrieval tool

that is powered by Google Maps.

## Any of

- [GoogleMaps](https://google.github.io/adk-docs/api-reference/agentconfig/#tab-pane_anyOf_i0_generate_content_config_anyOf_i0_tools_anyOf_i0_items_anyOf_i0_googleMaps_anyOf_i0)
- [Option 2](https://google.github.io/adk-docs/api-reference/agentconfig/#tab-pane_anyOf_i0_generate_content_config_anyOf_i0_tools_anyOf_i0_items_anyOf_i0_googleMaps_anyOf_i1)

#### GoogleMaps

Type: object

Tool to support Google Maps in Model.
No Additional Properties

## authConfig

Default: null

Optional. Auth config for the Google Maps tool.

## Any of

- [AuthConfig](https://google.github.io/adk-docs/api-reference/agentconfig/#tab-pane_anyOf_i0_generate_content_config_anyOf_i0_tools_anyOf_i0_items_anyOf_i0_googleMaps_anyOf_i0_authConfig_anyOf_i0)
- [Option 2](https://google.github.io/adk-docs/api-reference/agentconfig/#tab-pane_anyOf_i0_generate_content_config_anyOf_i0_tools_anyOf_i0_items_anyOf_i0_googleMaps_anyOf_i0_authConfig_anyOf_i1)

#### AuthConfig

Type: object

Auth configuration to run the extension.
[Same definition as AuthConfig](https://google.github.io/adk-docs/api-reference/agentconfig/#anyOf_i0_generate_content_config_anyOf_i0_tools_anyOf_i0_items_anyOf_i0_retrieval_anyOf_i0_externalApi_anyOf_i0_authConfig_anyOf_i0)

Type: null

Type: null

## urlContext

Default: null

Optional. Tool to support URL context retrieval.

## Any of

- [UrlContext](https://google.github.io/adk-docs/api-reference/agentconfig/#tab-pane_anyOf_i0_generate_content_config_anyOf_i0_tools_anyOf_i0_items_anyOf_i0_urlContext_anyOf_i0)
- [Option 2](https://google.github.io/adk-docs/api-reference/agentconfig/#tab-pane_anyOf_i0_generate_content_config_anyOf_i0_tools_anyOf_i0_items_anyOf_i0_urlContext_anyOf_i1)

#### UrlContext

Type: object

Tool to support URL context retrieval.

Type: null

## codeExecution

Default: null

Optional. CodeExecution tool type. Enables the model to execute code as part of generation.

## Any of

- [ToolCodeExecution](https://google.github.io/adk-docs/api-reference/agentconfig/#tab-pane_anyOf_i0_generate_content_config_anyOf_i0_tools_anyOf_i0_items_anyOf_i0_codeExecution_anyOf_i0)
- [Option 2](https://google.github.io/adk-docs/api-reference/agentconfig/#tab-pane_anyOf_i0_generate_content_config_anyOf_i0_tools_anyOf_i0_items_anyOf_i0_codeExecution_anyOf_i1)

#### ToolCodeExecution

Type: object

Tool that executes code generated by the model, and automatically returns the result to the model.

See also \[ExecutableCode\]and \[CodeExecutionResult\] which are input and output

to this tool.

Type: null

## computerUse

Default: null

Optional. Tool to support the model interacting directly with the computer. If enabled, it automatically populates computer-use specific Function Declarations.

## Any of

- [ToolComputerUse](https://google.github.io/adk-docs/api-reference/agentconfig/#tab-pane_anyOf_i0_generate_content_config_anyOf_i0_tools_anyOf_i0_items_anyOf_i0_computerUse_anyOf_i0)
- [Option 2](https://google.github.io/adk-docs/api-reference/agentconfig/#tab-pane_anyOf_i0_generate_content_config_anyOf_i0_tools_anyOf_i0_items_anyOf_i0_computerUse_anyOf_i1)

#### ToolComputerUse

Type: object

Tool to support computer use.
No Additional Properties

## environment

Default: null

Required. The environment being operated.

## Any of

- [Environment](https://google.github.io/adk-docs/api-reference/agentconfig/#tab-pane_anyOf_i0_generate_content_config_anyOf_i0_tools_anyOf_i0_items_anyOf_i0_computerUse_anyOf_i0_environment_anyOf_i0)
- [Option 2](https://google.github.io/adk-docs/api-reference/agentconfig/#tab-pane_anyOf_i0_generate_content_config_anyOf_i0_tools_anyOf_i0_items_anyOf_i0_computerUse_anyOf_i0_environment_anyOf_i1)

#### Environment

Type: enum (of string)

Required. The environment being operated.

#### Must be one of:

- "ENVIRONMENT\_UNSPECIFIED"
- "ENVIRONMENT\_BROWSER"

Type: null

Type: null

#### Tool

Type: object

Definition for a tool the client can call.

## nameRequired

#### Name

Type: string

## title

#### Title

Default: null

## Any of

- [Option 1](https://google.github.io/adk-docs/api-reference/agentconfig/#tab-pane_anyOf_i0_generate_content_config_anyOf_i0_tools_anyOf_i0_items_anyOf_i1_title_anyOf_i0)
- [Option 2](https://google.github.io/adk-docs/api-reference/agentconfig/#tab-pane_anyOf_i0_generate_content_config_anyOf_i0_tools_anyOf_i0_items_anyOf_i1_title_anyOf_i1)

Type: string

Type: null

## description

#### Description

Default: null

## Any of

- [Option 1](https://google.github.io/adk-docs/api-reference/agentconfig/#tab-pane_anyOf_i0_generate_content_config_anyOf_i0_tools_anyOf_i0_items_anyOf_i1_description_anyOf_i0)
- [Option 2](https://google.github.io/adk-docs/api-reference/agentconfig/#tab-pane_anyOf_i0_generate_content_config_anyOf_i0_tools_anyOf_i0_items_anyOf_i1_description_anyOf_i1)

Type: string

Type: null

## inputSchemaRequired

#### Inputschema

Type: object

## _Additional Properties_

Additional Properties of any type are allowed.

Type: object

## outputSchema

#### Outputschema

Default: null

## Any of

- [Option 1](https://google.github.io/adk-docs/api-reference/agentconfig/#tab-pane_anyOf_i0_generate_content_config_anyOf_i0_tools_anyOf_i0_items_anyOf_i1_outputSchema_anyOf_i0)
- [Option 2](https://google.github.io/adk-docs/api-reference/agentconfig/#tab-pane_anyOf_i0_generate_content_config_anyOf_i0_tools_anyOf_i0_items_anyOf_i1_outputSchema_anyOf_i1)

Type: object

## _Additional Properties_

Additional Properties of any type are allowed.

Type: object

Type: null

## annotations

Default: null

## Any of

- [ToolAnnotations](https://google.github.io/adk-docs/api-reference/agentconfig/#tab-pane_anyOf_i0_generate_content_config_anyOf_i0_tools_anyOf_i0_items_anyOf_i1_annotations_anyOf_i0)
- [Option 2](https://google.github.io/adk-docs/api-reference/agentconfig/#tab-pane_anyOf_i0_generate_content_config_anyOf_i0_tools_anyOf_i0_items_anyOf_i1_annotations_anyOf_i1)

#### ToolAnnotations

Type: object

Additional properties describing a Tool to clients.

NOTE: all properties in ToolAnnotations are **hints**.

They are not guaranteed to provide a faithful description of

tool behavior (including descriptive properties like `title`).

Clients should never make tool use decisions based on ToolAnnotations

received from untrusted servers.

## title

#### Title

Default: null

## Any of

- [Option 1](https://google.github.io/adk-docs/api-reference/agentconfig/#tab-pane_anyOf_i0_generate_content_config_anyOf_i0_tools_anyOf_i0_items_anyOf_i1_annotations_anyOf_i0_title_anyOf_i0)
- [Option 2](https://google.github.io/adk-docs/api-reference/agentconfig/#tab-pane_anyOf_i0_generate_content_config_anyOf_i0_tools_anyOf_i0_items_anyOf_i1_annotations_anyOf_i0_title_anyOf_i1)

Type: string

Type: null

## readOnlyHint

#### Readonlyhint

Default: null

## Any of

- [Option 1](https://google.github.io/adk-docs/api-reference/agentconfig/#tab-pane_anyOf_i0_generate_content_config_anyOf_i0_tools_anyOf_i0_items_anyOf_i1_annotations_anyOf_i0_readOnlyHint_anyOf_i0)
- [Option 2](https://google.github.io/adk-docs/api-reference/agentconfig/#tab-pane_anyOf_i0_generate_content_config_anyOf_i0_tools_anyOf_i0_items_anyOf_i1_annotations_anyOf_i0_readOnlyHint_anyOf_i1)

Type: boolean

Type: null

## destructiveHint

#### Destructivehint

Default: null

## Any of

- [Option 1](https://google.github.io/adk-docs/api-reference/agentconfig/#tab-pane_anyOf_i0_generate_content_config_anyOf_i0_tools_anyOf_i0_items_anyOf_i1_annotations_anyOf_i0_destructiveHint_anyOf_i0)
- [Option 2](https://google.github.io/adk-docs/api-reference/agentconfig/#tab-pane_anyOf_i0_generate_content_config_anyOf_i0_tools_anyOf_i0_items_anyOf_i1_annotations_anyOf_i0_destructiveHint_anyOf_i1)

Type: boolean

Type: null

## idempotentHint

#### Idempotenthint

Default: null

## Any of

- [Option 1](https://google.github.io/adk-docs/api-reference/agentconfig/#tab-pane_anyOf_i0_generate_content_config_anyOf_i0_tools_anyOf_i0_items_anyOf_i1_annotations_anyOf_i0_idempotentHint_anyOf_i0)
- [Option 2](https://google.github.io/adk-docs/api-reference/agentconfig/#tab-pane_anyOf_i0_generate_content_config_anyOf_i0_tools_anyOf_i0_items_anyOf_i1_annotations_anyOf_i0_idempotentHint_anyOf_i1)

Type: boolean

Type: null

## openWorldHint

#### Openworldhint

Default: null

## Any of

- [Option 1](https://google.github.io/adk-docs/api-reference/agentconfig/#tab-pane_anyOf_i0_generate_content_config_anyOf_i0_tools_anyOf_i0_items_anyOf_i1_annotations_anyOf_i0_openWorldHint_anyOf_i0)
- [Option 2](https://google.github.io/adk-docs/api-reference/agentconfig/#tab-pane_anyOf_i0_generate_content_config_anyOf_i0_tools_anyOf_i0_items_anyOf_i1_annotations_anyOf_i0_openWorldHint_anyOf_i1)

Type: boolean

Type: null

## _Additional Properties_

Additional Properties of any type are allowed.

Type: object

Type: null

## \_meta

#### Meta

Default: null

## Any of

- [Option 1](https://google.github.io/adk-docs/api-reference/agentconfig/#tab-pane_anyOf_i0_generate_content_config_anyOf_i0_tools_anyOf_i0_items_anyOf_i1__meta_anyOf_i0)
- [Option 2](https://google.github.io/adk-docs/api-reference/agentconfig/#tab-pane_anyOf_i0_generate_content_config_anyOf_i0_tools_anyOf_i0_items_anyOf_i1__meta_anyOf_i1)

Type: object

## _Additional Properties_

Additional Properties of any type are allowed.

Type: object

Type: null

## _Additional Properties_

Additional Properties of any type are allowed.

Type: object

Type: null

## toolConfig

Default: null

Associates model output to a specific function call.

## Any of

- [ToolConfig](https://google.github.io/adk-docs/api-reference/agentconfig/#tab-pane_anyOf_i0_generate_content_config_anyOf_i0_toolConfig_anyOf_i0)
- [Option 2](https://google.github.io/adk-docs/api-reference/agentconfig/#tab-pane_anyOf_i0_generate_content_config_anyOf_i0_toolConfig_anyOf_i1)

#### ToolConfig

Type: object

Tool config.

This config is shared for all tools provided in the request.
No Additional Properties

## functionCallingConfig

Default: null

Optional. Function calling config.

## Any of

- [FunctionCallingConfig](https://google.github.io/adk-docs/api-reference/agentconfig/#tab-pane_anyOf_i0_generate_content_config_anyOf_i0_toolConfig_anyOf_i0_functionCallingConfig_anyOf_i0)
- [Option 2](https://google.github.io/adk-docs/api-reference/agentconfig/#tab-pane_anyOf_i0_generate_content_config_anyOf_i0_toolConfig_anyOf_i0_functionCallingConfig_anyOf_i1)

#### FunctionCallingConfig

Type: object

Function calling config.
No Additional Properties

## mode

Default: null

Optional. Function calling mode.

## Any of

- [FunctionCallingConfigMode](https://google.github.io/adk-docs/api-reference/agentconfig/#tab-pane_anyOf_i0_generate_content_config_anyOf_i0_toolConfig_anyOf_i0_functionCallingConfig_anyOf_i0_mode_anyOf_i0)
- [Option 2](https://google.github.io/adk-docs/api-reference/agentconfig/#tab-pane_anyOf_i0_generate_content_config_anyOf_i0_toolConfig_anyOf_i0_functionCallingConfig_anyOf_i0_mode_anyOf_i1)

#### FunctionCallingConfigMode

Type: enum (of string)

Config for the function calling config mode.

#### Must be one of:

- "MODE\_UNSPECIFIED"
- "AUTO"
- "ANY"
- "NONE"

Type: null

## allowedFunctionNames

#### Allowedfunctionnames

Default: null

Optional. Function names to call. Only set when the Mode is ANY. Function names should match \[FunctionDeclaration.name\]. With mode set to ANY, model will predict a function call from the set of function names provided.

## Any of

- [Option 1](https://google.github.io/adk-docs/api-reference/agentconfig/#tab-pane_anyOf_i0_generate_content_config_anyOf_i0_toolConfig_anyOf_i0_functionCallingConfig_anyOf_i0_allowedFunctionNames_anyOf_i0)
- [Option 2](https://google.github.io/adk-docs/api-reference/agentconfig/#tab-pane_anyOf_i0_generate_content_config_anyOf_i0_toolConfig_anyOf_i0_functionCallingConfig_anyOf_i0_allowedFunctionNames_anyOf_i1)

Type: array of string

No Additional Items

#### Each item of this array must be:

Type: string

Type: null

Type: null

## retrievalConfig

Default: null

Optional. Retrieval config.

## Any of

- [RetrievalConfig](https://google.github.io/adk-docs/api-reference/agentconfig/#tab-pane_anyOf_i0_generate_content_config_anyOf_i0_toolConfig_anyOf_i0_retrievalConfig_anyOf_i0)
- [Option 2](https://google.github.io/adk-docs/api-reference/agentconfig/#tab-pane_anyOf_i0_generate_content_config_anyOf_i0_toolConfig_anyOf_i0_retrievalConfig_anyOf_i1)

#### RetrievalConfig

Type: object

Retrieval config.
No Additional Properties

## latLng

Default: null

Optional. The location of the user.

## Any of

- [LatLng](https://google.github.io/adk-docs/api-reference/agentconfig/#tab-pane_anyOf_i0_generate_content_config_anyOf_i0_toolConfig_anyOf_i0_retrievalConfig_anyOf_i0_latLng_anyOf_i0)
- [Option 2](https://google.github.io/adk-docs/api-reference/agentconfig/#tab-pane_anyOf_i0_generate_content_config_anyOf_i0_toolConfig_anyOf_i0_retrievalConfig_anyOf_i0_latLng_anyOf_i1)

#### LatLng

Type: object

An object that represents a latitude/longitude pair.

This is expressed as a pair of doubles to represent degrees latitude and

degrees longitude. Unless specified otherwise, this object must conform to the

<a href="https://en.wikipedia.org/wiki/World\_Geodetic\_System#1984\_version">

WGS84 standard</a>. Values must be within normalized ranges.

No Additional Properties

## latitude

#### Latitude

Default: null

The latitude in degrees. It must be in the range \[-90.0, +90.0\].

## Any of

- [Option 1](https://google.github.io/adk-docs/api-reference/agentconfig/#tab-pane_anyOf_i0_generate_content_config_anyOf_i0_toolConfig_anyOf_i0_retrievalConfig_anyOf_i0_latLng_anyOf_i0_latitude_anyOf_i0)
- [Option 2](https://google.github.io/adk-docs/api-reference/agentconfig/#tab-pane_anyOf_i0_generate_content_config_anyOf_i0_toolConfig_anyOf_i0_retrievalConfig_anyOf_i0_latLng_anyOf_i0_latitude_anyOf_i1)

Type: number

Type: null

## longitude

#### Longitude

Default: null

The longitude in degrees. It must be in the range \[-180.0, +180.0\]

## Any of

- [Option 1](https://google.github.io/adk-docs/api-reference/agentconfig/#tab-pane_anyOf_i0_generate_content_config_anyOf_i0_toolConfig_anyOf_i0_retrievalConfig_anyOf_i0_latLng_anyOf_i0_longitude_anyOf_i0)
- [Option 2](https://google.github.io/adk-docs/api-reference/agentconfig/#tab-pane_anyOf_i0_generate_content_config_anyOf_i0_toolConfig_anyOf_i0_retrievalConfig_anyOf_i0_latLng_anyOf_i0_longitude_anyOf_i1)

Type: number

Type: null

Type: null

## languageCode

#### Languagecode

Default: null

The language code of the user.

## Any of

- [Option 1](https://google.github.io/adk-docs/api-reference/agentconfig/#tab-pane_anyOf_i0_generate_content_config_anyOf_i0_toolConfig_anyOf_i0_retrievalConfig_anyOf_i0_languageCode_anyOf_i0)
- [Option 2](https://google.github.io/adk-docs/api-reference/agentconfig/#tab-pane_anyOf_i0_generate_content_config_anyOf_i0_toolConfig_anyOf_i0_retrievalConfig_anyOf_i0_languageCode_anyOf_i1)

Type: string

Type: null

Type: null

Type: null

## labels

#### Labels

Default: null

Labels with user-defined metadata to break down billed charges.

## Any of

- [Option 1](https://google.github.io/adk-docs/api-reference/agentconfig/#tab-pane_anyOf_i0_generate_content_config_anyOf_i0_labels_anyOf_i0)
- [Option 2](https://google.github.io/adk-docs/api-reference/agentconfig/#tab-pane_anyOf_i0_generate_content_config_anyOf_i0_labels_anyOf_i1)

Type: object

## _Additional Properties_

Each additional property must conform to the following schema

Type: string

Type: null

## cachedContent

#### Cachedcontent

Default: null

Resource name of a context cache that can be used in subsequent

requests.

## Any of

- [Option 1](https://google.github.io/adk-docs/api-reference/agentconfig/#tab-pane_anyOf_i0_generate_content_config_anyOf_i0_cachedContent_anyOf_i0)
- [Option 2](https://google.github.io/adk-docs/api-reference/agentconfig/#tab-pane_anyOf_i0_generate_content_config_anyOf_i0_cachedContent_anyOf_i1)

Type: string

Type: null

## responseModalities

#### Responsemodalities

Default: null

The requested modalities of the response. Represents the set of

modalities that the model can return.

## Any of

- [Option 1](https://google.github.io/adk-docs/api-reference/agentconfig/#tab-pane_anyOf_i0_generate_content_config_anyOf_i0_responseModalities_anyOf_i0)
- [Option 2](https://google.github.io/adk-docs/api-reference/agentconfig/#tab-pane_anyOf_i0_generate_content_config_anyOf_i0_responseModalities_anyOf_i1)

Type: array of string

No Additional Items

#### Each item of this array must be:

Type: string

Type: null

## mediaResolution

Default: null

If specified, the media resolution specified will be used.

## Any of

- [MediaResolution](https://google.github.io/adk-docs/api-reference/agentconfig/#tab-pane_anyOf_i0_generate_content_config_anyOf_i0_mediaResolution_anyOf_i0)
- [Option 2](https://google.github.io/adk-docs/api-reference/agentconfig/#tab-pane_anyOf_i0_generate_content_config_anyOf_i0_mediaResolution_anyOf_i1)

#### MediaResolution

Type: enum (of string)

The media resolution to use.

#### Must be one of:

- "MEDIA\_RESOLUTION\_UNSPECIFIED"
- "MEDIA\_RESOLUTION\_LOW"
- "MEDIA\_RESOLUTION\_MEDIUM"
- "MEDIA\_RESOLUTION\_HIGH"

Type: null

## speechConfig

#### Speechconfig

Default: null

The speech generation configuration.

## Any of

- [SpeechConfig](https://google.github.io/adk-docs/api-reference/agentconfig/#tab-pane_anyOf_i0_generate_content_config_anyOf_i0_speechConfig_anyOf_i0)
- [Option 2](https://google.github.io/adk-docs/api-reference/agentconfig/#tab-pane_anyOf_i0_generate_content_config_anyOf_i0_speechConfig_anyOf_i1)
- [Option 3](https://google.github.io/adk-docs/api-reference/agentconfig/#tab-pane_anyOf_i0_generate_content_config_anyOf_i0_speechConfig_anyOf_i2)

#### SpeechConfig

Type: object

The speech generation configuration.
No Additional Properties

## voiceConfig

Default: null

The configuration for the speaker to use.

## Any of

- [VoiceConfig](https://google.github.io/adk-docs/api-reference/agentconfig/#tab-pane_anyOf_i0_generate_content_config_anyOf_i0_speechConfig_anyOf_i0_voiceConfig_anyOf_i0)
- [Option 2](https://google.github.io/adk-docs/api-reference/agentconfig/#tab-pane_anyOf_i0_generate_content_config_anyOf_i0_speechConfig_anyOf_i0_voiceConfig_anyOf_i1)

#### VoiceConfig

Type: object

The configuration for the voice to use.
No Additional Properties

## prebuiltVoiceConfig

Default: null

The configuration for the speaker to use.

## Any of

- [PrebuiltVoiceConfig](https://google.github.io/adk-docs/api-reference/agentconfig/#tab-pane_anyOf_i0_generate_content_config_anyOf_i0_speechConfig_anyOf_i0_voiceConfig_anyOf_i0_prebuiltVoiceConfig_anyOf_i0)
- [Option 2](https://google.github.io/adk-docs/api-reference/agentconfig/#tab-pane_anyOf_i0_generate_content_config_anyOf_i0_speechConfig_anyOf_i0_voiceConfig_anyOf_i0_prebuiltVoiceConfig_anyOf_i1)

#### PrebuiltVoiceConfig

Type: object

The configuration for the prebuilt speaker to use.
No Additional Properties

## voiceName

#### Voicename

Default: null

The name of the prebuilt voice to use.

## Any of

- [Option 1](https://google.github.io/adk-docs/api-reference/agentconfig/#tab-pane_anyOf_i0_generate_content_config_anyOf_i0_speechConfig_anyOf_i0_voiceConfig_anyOf_i0_prebuiltVoiceConfig_anyOf_i0_voiceName_anyOf_i0)
- [Option 2](https://google.github.io/adk-docs/api-reference/agentconfig/#tab-pane_anyOf_i0_generate_content_config_anyOf_i0_speechConfig_anyOf_i0_voiceConfig_anyOf_i0_prebuiltVoiceConfig_anyOf_i0_voiceName_anyOf_i1)

Type: string

Type: null

Type: null

Type: null

## multiSpeakerVoiceConfig

Default: null

The configuration for the multi-speaker setup.

It is mutually exclusive with the voice\_config field.

## Any of

- [MultiSpeakerVoiceConfig](https://google.github.io/adk-docs/api-reference/agentconfig/#tab-pane_anyOf_i0_generate_content_config_anyOf_i0_speechConfig_anyOf_i0_multiSpeakerVoiceConfig_anyOf_i0)
- [Option 2](https://google.github.io/adk-docs/api-reference/agentconfig/#tab-pane_anyOf_i0_generate_content_config_anyOf_i0_speechConfig_anyOf_i0_multiSpeakerVoiceConfig_anyOf_i1)

#### MultiSpeakerVoiceConfig

Type: object

The configuration for the multi-speaker setup.
No Additional Properties

## speakerVoiceConfigs

#### Speakervoiceconfigs

Default: null

The configuration for the speaker to use.

## Any of

- [Option 1](https://google.github.io/adk-docs/api-reference/agentconfig/#tab-pane_anyOf_i0_generate_content_config_anyOf_i0_speechConfig_anyOf_i0_multiSpeakerVoiceConfig_anyOf_i0_speakerVoiceConfigs_anyOf_i0)
- [Option 2](https://google.github.io/adk-docs/api-reference/agentconfig/#tab-pane_anyOf_i0_generate_content_config_anyOf_i0_speechConfig_anyOf_i0_multiSpeakerVoiceConfig_anyOf_i0_speakerVoiceConfigs_anyOf_i1)

Type: array

No Additional Items

#### Each item of this array must be:

#### SpeakerVoiceConfig

Type: object

The configuration for the speaker to use.
No Additional Properties

## speaker

#### Speaker

Default: null

The name of the speaker to use. Should be the same as in the

prompt.

## Any of

- [Option 1](https://google.github.io/adk-docs/api-reference/agentconfig/#tab-pane_anyOf_i0_generate_content_config_anyOf_i0_speechConfig_anyOf_i0_multiSpeakerVoiceConfig_anyOf_i0_speakerVoiceConfigs_anyOf_i0_items_speaker_anyOf_i0)
- [Option 2](https://google.github.io/adk-docs/api-reference/agentconfig/#tab-pane_anyOf_i0_generate_content_config_anyOf_i0_speechConfig_anyOf_i0_multiSpeakerVoiceConfig_anyOf_i0_speakerVoiceConfigs_anyOf_i0_items_speaker_anyOf_i1)

Type: string

Type: null

## voiceConfig

Default: null

The configuration for the voice to use.

## Any of

- [VoiceConfig](https://google.github.io/adk-docs/api-reference/agentconfig/#tab-pane_anyOf_i0_generate_content_config_anyOf_i0_speechConfig_anyOf_i0_multiSpeakerVoiceConfig_anyOf_i0_speakerVoiceConfigs_anyOf_i0_items_voiceConfig_anyOf_i0)
- [Option 2](https://google.github.io/adk-docs/api-reference/agentconfig/#tab-pane_anyOf_i0_generate_content_config_anyOf_i0_speechConfig_anyOf_i0_multiSpeakerVoiceConfig_anyOf_i0_speakerVoiceConfigs_anyOf_i0_items_voiceConfig_anyOf_i1)

#### VoiceConfig

Type: object

The configuration for the voice to use.
[Same definition as VoiceConfig](https://google.github.io/adk-docs/api-reference/agentconfig/#anyOf_i0_generate_content_config_anyOf_i0_speechConfig_anyOf_i0_voiceConfig_anyOf_i0)

Type: null

Type: null

Type: null

## languageCode

#### Languagecode

Default: null

Language code (ISO 639. e.g. en-US) for the speech synthesization.

Only available for Live API.

## Any of

- [Option 1](https://google.github.io/adk-docs/api-reference/agentconfig/#tab-pane_anyOf_i0_generate_content_config_anyOf_i0_speechConfig_anyOf_i0_languageCode_anyOf_i0)
- [Option 2](https://google.github.io/adk-docs/api-reference/agentconfig/#tab-pane_anyOf_i0_generate_content_config_anyOf_i0_speechConfig_anyOf_i0_languageCode_anyOf_i1)

Type: string

Type: null

Type: string

Type: null

## audioTimestamp

#### Audiotimestamp

Default: null

If enabled, audio timestamp will be included in the request to the

model.

## Any of

- [Option 1](https://google.github.io/adk-docs/api-reference/agentconfig/#tab-pane_anyOf_i0_generate_content_config_anyOf_i0_audioTimestamp_anyOf_i0)
- [Option 2](https://google.github.io/adk-docs/api-reference/agentconfig/#tab-pane_anyOf_i0_generate_content_config_anyOf_i0_audioTimestamp_anyOf_i1)

Type: boolean

Type: null

## automaticFunctionCalling

Default: null

The configuration for automatic function calling.

## Any of

- [AutomaticFunctionCallingConfig](https://google.github.io/adk-docs/api-reference/agentconfig/#tab-pane_anyOf_i0_generate_content_config_anyOf_i0_automaticFunctionCalling_anyOf_i0)
- [Option 2](https://google.github.io/adk-docs/api-reference/agentconfig/#tab-pane_anyOf_i0_generate_content_config_anyOf_i0_automaticFunctionCalling_anyOf_i1)

#### AutomaticFunctionCallingConfig

Type: object

The configuration for automatic function calling.
No Additional Properties

## disable

#### Disable

Default: null

Whether to disable automatic function calling.

If not set or set to False, will enable automatic function calling.

If set to True, will disable automatic function calling.

## Any of

- [Option 1](https://google.github.io/adk-docs/api-reference/agentconfig/#tab-pane_anyOf_i0_generate_content_config_anyOf_i0_automaticFunctionCalling_anyOf_i0_disable_anyOf_i0)
- [Option 2](https://google.github.io/adk-docs/api-reference/agentconfig/#tab-pane_anyOf_i0_generate_content_config_anyOf_i0_automaticFunctionCalling_anyOf_i0_disable_anyOf_i1)

Type: boolean

Type: null

## maximumRemoteCalls

#### Maximumremotecalls

Default: 10

If automatic function calling is enabled,

maximum number of remote calls for automatic function calling.

This number should be a positive integer.

If not set, SDK will set maximum number of remote calls to 10.

## Any of

- [Option 1](https://google.github.io/adk-docs/api-reference/agentconfig/#tab-pane_anyOf_i0_generate_content_config_anyOf_i0_automaticFunctionCalling_anyOf_i0_maximumRemoteCalls_anyOf_i0)
- [Option 2](https://google.github.io/adk-docs/api-reference/agentconfig/#tab-pane_anyOf_i0_generate_content_config_anyOf_i0_automaticFunctionCalling_anyOf_i0_maximumRemoteCalls_anyOf_i1)

Type: integer

Type: null

## ignoreCallHistory

#### Ignorecallhistory

Default: null

If automatic function calling is enabled,

whether to ignore call history to the response.

If not set, SDK will set ignore _call_ history to false,

and will append the call history to

GenerateContentResponse.automatic _function_ calling\_history.

## Any of

- [Option 1](https://google.github.io/adk-docs/api-reference/agentconfig/#tab-pane_anyOf_i0_generate_content_config_anyOf_i0_automaticFunctionCalling_anyOf_i0_ignoreCallHistory_anyOf_i0)
- [Option 2](https://google.github.io/adk-docs/api-reference/agentconfig/#tab-pane_anyOf_i0_generate_content_config_anyOf_i0_automaticFunctionCalling_anyOf_i0_ignoreCallHistory_anyOf_i1)

Type: boolean

Type: null

Type: null

## thinkingConfig

Default: null

The thinking features configuration.

## Any of

- [ThinkingConfig](https://google.github.io/adk-docs/api-reference/agentconfig/#tab-pane_anyOf_i0_generate_content_config_anyOf_i0_thinkingConfig_anyOf_i0)
- [Option 2](https://google.github.io/adk-docs/api-reference/agentconfig/#tab-pane_anyOf_i0_generate_content_config_anyOf_i0_thinkingConfig_anyOf_i1)

#### ThinkingConfig

Type: object

The thinking features configuration.
No Additional Properties

## includeThoughts

#### Includethoughts

Default: null

Indicates whether to include thoughts in the response. If true, thoughts are returned only if the model supports thought and thoughts are available.

## Any of

- [Option 1](https://google.github.io/adk-docs/api-reference/agentconfig/#tab-pane_anyOf_i0_generate_content_config_anyOf_i0_thinkingConfig_anyOf_i0_includeThoughts_anyOf_i0)
- [Option 2](https://google.github.io/adk-docs/api-reference/agentconfig/#tab-pane_anyOf_i0_generate_content_config_anyOf_i0_thinkingConfig_anyOf_i0_includeThoughts_anyOf_i1)

Type: boolean

Type: null

## thinkingBudget

#### Thinkingbudget

Default: null

Indicates the thinking budget in tokens. 0 is DISABLED. -1 is AUTOMATIC. The default values and allowed ranges are model dependent.

## Any of

- [Option 1](https://google.github.io/adk-docs/api-reference/agentconfig/#tab-pane_anyOf_i0_generate_content_config_anyOf_i0_thinkingConfig_anyOf_i0_thinkingBudget_anyOf_i0)
- [Option 2](https://google.github.io/adk-docs/api-reference/agentconfig/#tab-pane_anyOf_i0_generate_content_config_anyOf_i0_thinkingConfig_anyOf_i0_thinkingBudget_anyOf_i1)

Type: integer

Type: null

Type: null

Type: null

#### LoopAgentConfig

Type: object

The config for the YAML schema of a LoopAgent.
No Additional Properties

## agent\_class

#### Agent Class

Type: constDefault: "LoopAgent"

The value is used to uniquely identify the LoopAgent class.
Specific value: `"LoopAgent"`

## nameRequired

#### Name

Type: string

Required. The name of the agent.

## description

#### Description

Type: stringDefault: ""

Optional. The description of the agent.

## sub\_agents

#### Sub Agents

Default: null

Optional. The sub-agents of the agent.

## Any of

- [Option 1](https://google.github.io/adk-docs/api-reference/agentconfig/#tab-pane_anyOf_i1_sub_agents_anyOf_i0)
- [Option 2](https://google.github.io/adk-docs/api-reference/agentconfig/#tab-pane_anyOf_i1_sub_agents_anyOf_i1)

Type: array

No Additional Items

#### Each item of this array must be:

#### AgentRefConfig

Type: object

The config for the reference to another agent.
[Same definition as AgentRefConfig](https://google.github.io/adk-docs/api-reference/agentconfig/#anyOf_i0_sub_agents_anyOf_i0_items)

Type: null

## before\_agent\_callbacks

#### Before Agent Callbacks

Default: null

Optional. The before _agent_ callbacks of the agent.

Example:

```
before_agent_callbacks:
  - name: my_library.security_callbacks.before_agent_callback


```

## Any of

- [Option 1](https://google.github.io/adk-docs/api-reference/agentconfig/#tab-pane_anyOf_i1_before_agent_callbacks_anyOf_i0)
- [Option 2](https://google.github.io/adk-docs/api-reference/agentconfig/#tab-pane_anyOf_i1_before_agent_callbacks_anyOf_i1)

Type: array

No Additional Items

#### Each item of this array must be:

#### CodeConfig

Type: object

Code reference config for a variable, a function, or a class.

This config is used for configuring callbacks and tools.
[Same definition as CodeConfig](https://google.github.io/adk-docs/api-reference/agentconfig/#anyOf_i0_before_agent_callbacks_anyOf_i0_items)

Type: null

## after\_agent\_callbacks

#### After Agent Callbacks

Default: null

Optional. The after _agent_ callbacks of the agent.

## Any of

- [Option 1](https://google.github.io/adk-docs/api-reference/agentconfig/#tab-pane_anyOf_i1_after_agent_callbacks_anyOf_i0)
- [Option 2](https://google.github.io/adk-docs/api-reference/agentconfig/#tab-pane_anyOf_i1_after_agent_callbacks_anyOf_i1)

Type: array

No Additional Items

#### Each item of this array must be:

#### CodeConfig

Type: object

Code reference config for a variable, a function, or a class.

This config is used for configuring callbacks and tools.
[Same definition as CodeConfig](https://google.github.io/adk-docs/api-reference/agentconfig/#anyOf_i0_before_agent_callbacks_anyOf_i0_items)

Type: null

## max\_iterations

#### Max Iterations

Default: null

Optional. LoopAgent.max\_iterations.

## Any of

- [Option 1](https://google.github.io/adk-docs/api-reference/agentconfig/#tab-pane_anyOf_i1_max_iterations_anyOf_i0)
- [Option 2](https://google.github.io/adk-docs/api-reference/agentconfig/#tab-pane_anyOf_i1_max_iterations_anyOf_i1)

Type: integer

Type: null

#### ParallelAgentConfig

Type: object

The config for the YAML schema of a ParallelAgent.
No Additional Properties

## agent\_class

#### Agent Class

Type: constDefault: "ParallelAgent"

The value is used to uniquely identify the ParallelAgent class.
Specific value: `"ParallelAgent"`

## nameRequired

#### Name

Type: string

Required. The name of the agent.

## description

#### Description

Type: stringDefault: ""

Optional. The description of the agent.

## sub\_agents

#### Sub Agents

Default: null

Optional. The sub-agents of the agent.

## Any of

- [Option 1](https://google.github.io/adk-docs/api-reference/agentconfig/#tab-pane_anyOf_i2_sub_agents_anyOf_i0)
- [Option 2](https://google.github.io/adk-docs/api-reference/agentconfig/#tab-pane_anyOf_i2_sub_agents_anyOf_i1)

Type: array

No Additional Items

#### Each item of this array must be:

#### AgentRefConfig

Type: object

The config for the reference to another agent.
[Same definition as AgentRefConfig](https://google.github.io/adk-docs/api-reference/agentconfig/#anyOf_i0_sub_agents_anyOf_i0_items)

Type: null

## before\_agent\_callbacks

#### Before Agent Callbacks

Default: null

Optional. The before _agent_ callbacks of the agent.

Example:

```
before_agent_callbacks:
  - name: my_library.security_callbacks.before_agent_callback


```

## Any of

- [Option 1](https://google.github.io/adk-docs/api-reference/agentconfig/#tab-pane_anyOf_i2_before_agent_callbacks_anyOf_i0)
- [Option 2](https://google.github.io/adk-docs/api-reference/agentconfig/#tab-pane_anyOf_i2_before_agent_callbacks_anyOf_i1)

Type: array

No Additional Items

#### Each item of this array must be:

#### CodeConfig

Type: object

Code reference config for a variable, a function, or a class.

This config is used for configuring callbacks and tools.
[Same definition as CodeConfig](https://google.github.io/adk-docs/api-reference/agentconfig/#anyOf_i0_before_agent_callbacks_anyOf_i0_items)

Type: null

## after\_agent\_callbacks

#### After Agent Callbacks

Default: null

Optional. The after _agent_ callbacks of the agent.

## Any of

- [Option 1](https://google.github.io/adk-docs/api-reference/agentconfig/#tab-pane_anyOf_i2_after_agent_callbacks_anyOf_i0)
- [Option 2](https://google.github.io/adk-docs/api-reference/agentconfig/#tab-pane_anyOf_i2_after_agent_callbacks_anyOf_i1)

Type: array

No Additional Items

#### Each item of this array must be:

#### CodeConfig

Type: object

Code reference config for a variable, a function, or a class.

This config is used for configuring callbacks and tools.
[Same definition as CodeConfig](https://google.github.io/adk-docs/api-reference/agentconfig/#anyOf_i0_before_agent_callbacks_anyOf_i0_items)

Type: null

#### SequentialAgentConfig

Type: object

The config for the YAML schema of a SequentialAgent.
No Additional Properties

## agent\_class

#### Agent Class

Type: constDefault: "SequentialAgent"

The value is used to uniquely identify the SequentialAgent class.
Specific value: `"SequentialAgent"`

## nameRequired

#### Name

Type: string

Required. The name of the agent.

## description

#### Description

Type: stringDefault: ""

Optional. The description of the agent.

## sub\_agents

#### Sub Agents

Default: null

Optional. The sub-agents of the agent.

## Any of

- [Option 1](https://google.github.io/adk-docs/api-reference/agentconfig/#tab-pane_anyOf_i3_sub_agents_anyOf_i0)
- [Option 2](https://google.github.io/adk-docs/api-reference/agentconfig/#tab-pane_anyOf_i3_sub_agents_anyOf_i1)

Type: array

No Additional Items

#### Each item of this array must be:

#### AgentRefConfig

Type: object

The config for the reference to another agent.
[Same definition as AgentRefConfig](https://google.github.io/adk-docs/api-reference/agentconfig/#anyOf_i0_sub_agents_anyOf_i0_items)

Type: null

## before\_agent\_callbacks

#### Before Agent Callbacks

Default: null

Optional. The before _agent_ callbacks of the agent.

Example:

```
before_agent_callbacks:
  - name: my_library.security_callbacks.before_agent_callback


```

## Any of

- [Option 1](https://google.github.io/adk-docs/api-reference/agentconfig/#tab-pane_anyOf_i3_before_agent_callbacks_anyOf_i0)
- [Option 2](https://google.github.io/adk-docs/api-reference/agentconfig/#tab-pane_anyOf_i3_before_agent_callbacks_anyOf_i1)

Type: array

No Additional Items

#### Each item of this array must be:

#### CodeConfig

Type: object

Code reference config for a variable, a function, or a class.

This config is used for configuring callbacks and tools.
[Same definition as CodeConfig](https://google.github.io/adk-docs/api-reference/agentconfig/#anyOf_i0_before_agent_callbacks_anyOf_i0_items)

Type: null

## after\_agent\_callbacks

#### After Agent Callbacks

Default: null

Optional. The after _agent_ callbacks of the agent.

## Any of

- [Option 1](https://google.github.io/adk-docs/api-reference/agentconfig/#tab-pane_anyOf_i3_after_agent_callbacks_anyOf_i0)
- [Option 2](https://google.github.io/adk-docs/api-reference/agentconfig/#tab-pane_anyOf_i3_after_agent_callbacks_anyOf_i1)

Type: array

No Additional Items

#### Each item of this array must be:

#### CodeConfig

Type: object

Code reference config for a variable, a function, or a class.

This config is used for configuring callbacks and tools.
[Same definition as CodeConfig](https://google.github.io/adk-docs/api-reference/agentconfig/#anyOf_i0_before_agent_callbacks_anyOf_i0_items)

Type: null

#### BaseAgentConfig

Type: object

The config for the YAML schema of a BaseAgent.

Do not use this class directly. It's the base class for all agent configs.

## agent\_class

#### Agent Class

Default: "BaseAgent"

Required. The class of the agent. The value is used to differentiate among different agent classes.

## Any of

- [Option 1](https://google.github.io/adk-docs/api-reference/agentconfig/#tab-pane_anyOf_i4_agent_class_anyOf_i0)
- [Option 2](https://google.github.io/adk-docs/api-reference/agentconfig/#tab-pane_anyOf_i4_agent_class_anyOf_i1)

Type: const

Specific value: `"BaseAgent"`

Type: string

## nameRequired

#### Name

Type: string

Required. The name of the agent.

## description

#### Description

Type: stringDefault: ""

Optional. The description of the agent.

## sub\_agents

#### Sub Agents

Default: null

Optional. The sub-agents of the agent.

## Any of

- [Option 1](https://google.github.io/adk-docs/api-reference/agentconfig/#tab-pane_anyOf_i4_sub_agents_anyOf_i0)
- [Option 2](https://google.github.io/adk-docs/api-reference/agentconfig/#tab-pane_anyOf_i4_sub_agents_anyOf_i1)

Type: array

No Additional Items

#### Each item of this array must be:

#### AgentRefConfig

Type: object

The config for the reference to another agent.
[Same definition as AgentRefConfig](https://google.github.io/adk-docs/api-reference/agentconfig/#anyOf_i0_sub_agents_anyOf_i0_items)

Type: null

## before\_agent\_callbacks

#### Before Agent Callbacks

Default: null

Optional. The before _agent_ callbacks of the agent.

Example:

```
before_agent_callbacks:
  - name: my_library.security_callbacks.before_agent_callback


```

## Any of

- [Option 1](https://google.github.io/adk-docs/api-reference/agentconfig/#tab-pane_anyOf_i4_before_agent_callbacks_anyOf_i0)
- [Option 2](https://google.github.io/adk-docs/api-reference/agentconfig/#tab-pane_anyOf_i4_before_agent_callbacks_anyOf_i1)

Type: array

No Additional Items

#### Each item of this array must be:

#### CodeConfig

Type: object

Code reference config for a variable, a function, or a class.

This config is used for configuring callbacks and tools.
[Same definition as CodeConfig](https://google.github.io/adk-docs/api-reference/agentconfig/#anyOf_i0_before_agent_callbacks_anyOf_i0_items)

Type: null

## after\_agent\_callbacks

#### After Agent Callbacks

Default: null

Optional. The after _agent_ callbacks of the agent.

## Any of

- [Option 1](https://google.github.io/adk-docs/api-reference/agentconfig/#tab-pane_anyOf_i4_after_agent_callbacks_anyOf_i0)
- [Option 2](https://google.github.io/adk-docs/api-reference/agentconfig/#tab-pane_anyOf_i4_after_agent_callbacks_anyOf_i1)

Type: array

No Additional Items

#### Each item of this array must be:

#### CodeConfig

Type: object

Code reference config for a variable, a function, or a class.

This config is used for configuring callbacks and tools.
[Same definition as CodeConfig](https://google.github.io/adk-docs/api-reference/agentconfig/#anyOf_i0_before_agent_callbacks_anyOf_i0_items)

Type: null

## _Additional Properties_

Additional Properties of any type are allowed.

Type: object

## Long Running Function Tool
[java.lang.Object](https://docs.oracle.com/en/java/javase/17/docs/api/java.base/java/lang/Object.html "class or interface in java.lang")

[com.google.adk.tools.BaseTool](https://google.github.io/adk-docs/api-reference/java/com/google/adk/tools/BaseTool.html "class in com.google.adk.tools")

[com.google.adk.tools.FunctionTool](https://google.github.io/adk-docs/api-reference/java/com/google/adk/tools/FunctionTool.html "class in com.google.adk.tools")

com.google.adk.tools.LongRunningFunctionTool

* * *

public class LongRunningFunctionToolextends [FunctionTool](https://google.github.io/adk-docs/api-reference/java/com/google/adk/tools/FunctionTool.html "class in com.google.adk.tools")

A function tool that returns the result asynchronously.

- ## Method Summary [![Link icon](https://google.github.io/adk-docs/api-reference/java/resource-files/link.svg)](https://google.github.io/adk-docs/api-reference/java/com/google/adk/tools/LongRunningFunctionTool.html\#method-summary)





All MethodsStatic MethodsConcrete Methods







Modifier and Type



Method



Description



`static LongRunningFunctionTool`



`create(Class<?> cls,
String methodName)`







`static LongRunningFunctionTool`



`create(Object instance,
String methodName)`







`static LongRunningFunctionTool`



`create(Method func)`















### Methods inherited from class com.google.adk.tools. [FunctionTool](https://google.github.io/adk-docs/api-reference/java/com/google/adk/tools/FunctionTool.html "class in com.google.adk.tools") [![Link icon](https://google.github.io/adk-docs/api-reference/java/resource-files/link.svg)](https://google.github.io/adk-docs/api-reference/java/com/google/adk/tools/LongRunningFunctionTool.html\#methods-inherited-from-class-com.google.adk.tools.FunctionTool)

`create, declaration, runAsync`





### Methods inherited from class com.google.adk.tools. [BaseTool](https://google.github.io/adk-docs/api-reference/java/com/google/adk/tools/BaseTool.html "class in com.google.adk.tools") [![Link icon](https://google.github.io/adk-docs/api-reference/java/resource-files/link.svg)](https://google.github.io/adk-docs/api-reference/java/com/google/adk/tools/LongRunningFunctionTool.html\#methods-inherited-from-class-com.google.adk.tools.BaseTool)

`description, longRunning, name, processLlmRequest`





### Methods inherited from class java.lang. [Object](https://docs.oracle.com/en/java/javase/17/docs/api/java.base/java/lang/Object.html "class or interface in java.lang") [![Link icon](https://google.github.io/adk-docs/api-reference/java/resource-files/link.svg)](https://google.github.io/adk-docs/api-reference/java/com/google/adk/tools/LongRunningFunctionTool.html\#methods-inherited-from-class-java.lang.Object)

`clone, equals, finalize, getClass, hashCode, notify, notifyAll, toString, wait, wait, wait`


- ## Method Details [![Link icon](https://google.github.io/adk-docs/api-reference/java/resource-files/link.svg)](https://google.github.io/adk-docs/api-reference/java/com/google/adk/tools/LongRunningFunctionTool.html\#method-detail)



- ### create [![Link icon](https://google.github.io/adk-docs/api-reference/java/resource-files/link.svg)](https://google.github.io/adk-docs/api-reference/java/com/google/adk/tools/LongRunningFunctionTool.html\#create(java.lang.reflect.Method))





public static[LongRunningFunctionTool](https://google.github.io/adk-docs/api-reference/java/com/google/adk/tools/LongRunningFunctionTool.html "class in com.google.adk.tools")create( [Method](https://docs.oracle.com/en/java/javase/17/docs/api/java.base/java/lang/reflect/Method.html "class or interface in java.lang.reflect") func)

- ### create [![Link icon](https://google.github.io/adk-docs/api-reference/java/resource-files/link.svg)](https://google.github.io/adk-docs/api-reference/java/com/google/adk/tools/LongRunningFunctionTool.html\#create(java.lang.Class,java.lang.String))





public static[LongRunningFunctionTool](https://google.github.io/adk-docs/api-reference/java/com/google/adk/tools/LongRunningFunctionTool.html "class in com.google.adk.tools")create( [Class](https://docs.oracle.com/en/java/javase/17/docs/api/java.base/java/lang/Class.html "class or interface in java.lang") <?> cls,
[String](https://docs.oracle.com/en/java/javase/17/docs/api/java.base/java/lang/String.html "class or interface in java.lang") methodName)

- ### create [![Link icon](https://google.github.io/adk-docs/api-reference/java/resource-files/link.svg)](https://google.github.io/adk-docs/api-reference/java/com/google/adk/tools/LongRunningFunctionTool.html\#create(java.lang.Object,java.lang.String))





public static[LongRunningFunctionTool](https://google.github.io/adk-docs/api-reference/java/com/google/adk/tools/LongRunningFunctionTool.html "class in com.google.adk.tools")create( [Object](https://docs.oracle.com/en/java/javase/17/docs/api/java.base/java/lang/Object.html "class or interface in java.lang") instance,
[String](https://docs.oracle.com/en/java/javase/17/docs/api/java.base/java/lang/String.html "class or interface in java.lang") methodName)

## Workflow Agents Overview
[Skip to content](https://google.github.io/adk-docs/agents/workflow-agents/#workflow-agents)

[Edit this page](https://github.com/google/adk-docs/edit/main/docs/agents/workflow-agents/index.md "Edit this page") [View source of this page](https://github.com/google/adk-docs/raw/main/docs/agents/workflow-agents/index.md "View source of this page")

# Workflow Agents [¶](https://google.github.io/adk-docs/agents/workflow-agents/\#workflow-agents "Permanent link")

Supported in ADKPythonGoJava

This section introduces " _workflow agents_" \- **specialized agents that control the execution flow of its sub-agents**.

Workflow agents are specialized components in ADK designed purely for **orchestrating the execution flow of sub-agents**. Their primary role is to manage _how_ and _when_ other agents run, defining the control flow of a process.

Unlike [LLM Agents](https://google.github.io/adk-docs/agents/llm-agents/), which use Large Language Models for dynamic reasoning and decision-making, Workflow Agents operate based on **predefined logic**. They determine the execution sequence according to their type (e.g., sequential, parallel, loop) without consulting an LLM for the orchestration itself. This results in **deterministic and predictable execution patterns**.

ADK provides three core workflow agent types, each implementing a distinct execution pattern:

- **Sequential Agents**


* * *


Executes sub-agents one after another, in **sequence**.

[Learn more](https://google.github.io/adk-docs/agents/workflow-agents/sequential-agents/)

- **Loop Agents**


* * *


**Repeatedly** executes its sub-agents until a specific termination condition is met.

[Learn more](https://google.github.io/adk-docs/agents/workflow-agents/loop-agents/)

- **Parallel Agents**


* * *


Executes multiple sub-agents in **parallel**.

[Learn more](https://google.github.io/adk-docs/agents/workflow-agents/parallel-agents/)


## Why Use Workflow Agents? [¶](https://google.github.io/adk-docs/agents/workflow-agents/\#why-use-workflow-agents "Permanent link")

Workflow agents are essential when you need explicit control over how a series of tasks or agents are executed. They provide:

- **Predictability:** The flow of execution is guaranteed based on the agent type and configuration.
- **Reliability:** Ensures tasks run in the required order or pattern consistently.
- **Structure:** Allows you to build complex processes by composing agents within clear control structures.

While the workflow agent manages the control flow deterministically, the sub-agents it orchestrates can themselves be any type of agent, including intelligent LLM Agent instances. This allows you to combine structured process control with flexible, LLM-powered task execution.

Back to top

## Telemetry Data Reporting
[java.lang.Object](https://docs.oracle.com/en/java/javase/17/docs/api/java.base/java/lang/Object.html "class or interface in java.lang")

com.google.adk.Telemetry

* * *

public class Telemetryextends [Object](https://docs.oracle.com/en/java/javase/17/docs/api/java.base/java/lang/Object.html "class or interface in java.lang")

Utility class for capturing and reporting telemetry data within the ADK. This class provides
methods to trace various aspects of the agent's execution, including tool calls, tool responses,
LLM interactions, and data handling. It leverages OpenTelemetry for tracing and logging for
detailed information. These traces can then be exported through the ADK Dev Server UI.

- ## Method Summary [![Link icon](https://google.github.io/adk-docs/api-reference/java/resource-files/link.svg)](https://google.github.io/adk-docs/api-reference/java/com/google/adk/Telemetry.html\#method-summary)





All MethodsStatic MethodsConcrete Methods







Modifier and Type



Method



Description



`static io.opentelemetry.api.trace.Tracer`



`getTracer()`





Gets the tracer.





`static void`



`traceCallLlm(InvocationContext invocationContext,
String eventId,
LlmRequest llmRequest,
LlmResponse llmResponse)`





Traces a call to the LLM.





`static void`



`traceSendData(InvocationContext invocationContext,
String eventId,
List<com.google.genai.types.Content> data)`





Traces the sending of data (history or new content) to the agent/model.





`static void`



`traceToolCall(Map<String,Object> args)`





Traces tool call arguments.





`static void`



`traceToolResponse(InvocationContext invocationContext,
String eventId,
Event functionResponseEvent)`





Traces tool response event.













### Methods inherited from class java.lang. [Object](https://docs.oracle.com/en/java/javase/17/docs/api/java.base/java/lang/Object.html "class or interface in java.lang") [![Link icon](https://google.github.io/adk-docs/api-reference/java/resource-files/link.svg)](https://google.github.io/adk-docs/api-reference/java/com/google/adk/Telemetry.html\#methods-inherited-from-class-java.lang.Object)

`clone, equals, finalize, getClass, hashCode, notify, notifyAll, toString, wait, wait, wait`


- ## Method Details [![Link icon](https://google.github.io/adk-docs/api-reference/java/resource-files/link.svg)](https://google.github.io/adk-docs/api-reference/java/com/google/adk/Telemetry.html\#method-detail)



- ### traceToolCall [![Link icon](https://google.github.io/adk-docs/api-reference/java/resource-files/link.svg)](https://google.github.io/adk-docs/api-reference/java/com/google/adk/Telemetry.html\#traceToolCall(java.util.Map))





public staticvoidtraceToolCall( [Map](https://docs.oracle.com/en/java/javase/17/docs/api/java.base/java/util/Map.html "class or interface in java.util") < [String](https://docs.oracle.com/en/java/javase/17/docs/api/java.base/java/lang/String.html "class or interface in java.lang"), [Object](https://docs.oracle.com/en/java/javase/17/docs/api/java.base/java/lang/Object.html "class or interface in java.lang") > args)



Traces tool call arguments.

Parameters:`args` \- The arguments to the tool call.

- ### traceToolResponse [![Link icon](https://google.github.io/adk-docs/api-reference/java/resource-files/link.svg)](https://google.github.io/adk-docs/api-reference/java/com/google/adk/Telemetry.html\#traceToolResponse(com.google.adk.agents.InvocationContext,java.lang.String,com.google.adk.events.Event))





public staticvoidtraceToolResponse( [InvocationContext](https://google.github.io/adk-docs/api-reference/java/com/google/adk/agents/InvocationContext.html "class in com.google.adk.agents") invocationContext,
[String](https://docs.oracle.com/en/java/javase/17/docs/api/java.base/java/lang/String.html "class or interface in java.lang") eventId,
[Event](https://google.github.io/adk-docs/api-reference/java/com/google/adk/events/Event.html "class in com.google.adk.events") functionResponseEvent)



Traces tool response event.

Parameters:`invocationContext` \- The invocation context for the current agent run.`eventId` \- The ID of the event.`functionResponseEvent` \- The function response event.

- ### traceCallLlm [![Link icon](https://google.github.io/adk-docs/api-reference/java/resource-files/link.svg)](https://google.github.io/adk-docs/api-reference/java/com/google/adk/Telemetry.html\#traceCallLlm(com.google.adk.agents.InvocationContext,java.lang.String,com.google.adk.models.LlmRequest,com.google.adk.models.LlmResponse))





public staticvoidtraceCallLlm( [InvocationContext](https://google.github.io/adk-docs/api-reference/java/com/google/adk/agents/InvocationContext.html "class in com.google.adk.agents") invocationContext,
[String](https://docs.oracle.com/en/java/javase/17/docs/api/java.base/java/lang/String.html "class or interface in java.lang") eventId,
[LlmRequest](https://google.github.io/adk-docs/api-reference/java/com/google/adk/models/LlmRequest.html "class in com.google.adk.models") llmRequest,
[LlmResponse](https://google.github.io/adk-docs/api-reference/java/com/google/adk/models/LlmResponse.html "class in com.google.adk.models") llmResponse)



Traces a call to the LLM.

Parameters:`invocationContext` \- The invocation context.`eventId` \- The ID of the event associated with this LLM call/response.`llmRequest` \- The LLM request object.`llmResponse` \- The LLM response object.

- ### traceSendData [![Link icon](https://google.github.io/adk-docs/api-reference/java/resource-files/link.svg)](https://google.github.io/adk-docs/api-reference/java/com/google/adk/Telemetry.html\#traceSendData(com.google.adk.agents.InvocationContext,java.lang.String,java.util.List))





public staticvoidtraceSendData( [InvocationContext](https://google.github.io/adk-docs/api-reference/java/com/google/adk/agents/InvocationContext.html "class in com.google.adk.agents") invocationContext,
[String](https://docs.oracle.com/en/java/javase/17/docs/api/java.base/java/lang/String.html "class or interface in java.lang") eventId,
[List](https://docs.oracle.com/en/java/javase/17/docs/api/java.base/java/util/List.html "class or interface in java.util") <com.google.genai.types.Content> data)



Traces the sending of data (history or new content) to the agent/model.

Parameters:`invocationContext` \- The invocation context.`eventId` \- The ID of the event, if applicable.`data` \- A list of content objects being sent.

- ### getTracer [![Link icon](https://google.github.io/adk-docs/api-reference/java/resource-files/link.svg)](https://google.github.io/adk-docs/api-reference/java/com/google/adk/Telemetry.html\#getTracer())





public staticio.opentelemetry.api.trace.TracergetTracer()



Gets the tracer.

Returns:The tracer.

## Collection Utilities
[java.lang.Object](https://docs.oracle.com/en/java/javase/17/docs/api/java.base/java/lang/Object.html "class or interface in java.lang")

com.google.adk.utils.CollectionUtils

* * *

public final class CollectionUtilsextends [Object](https://docs.oracle.com/en/java/javase/17/docs/api/java.base/java/lang/Object.html "class or interface in java.lang")

Frequently used code snippets for collections.

- ## Method Summary [![Link icon](https://google.github.io/adk-docs/api-reference/java/resource-files/link.svg)](https://google.github.io/adk-docs/api-reference/java/com/google/adk/utils/CollectionUtils.html\#method-summary)





All MethodsStatic MethodsConcrete Methods







Modifier and Type



Method



Description



`static <T> boolean`



`isNullOrEmpty(Iterable<T> iterable)`





Checks if the given iterable is null or empty.













### Methods inherited from class java.lang. [Object](https://docs.oracle.com/en/java/javase/17/docs/api/java.base/java/lang/Object.html "class or interface in java.lang") [![Link icon](https://google.github.io/adk-docs/api-reference/java/resource-files/link.svg)](https://google.github.io/adk-docs/api-reference/java/com/google/adk/utils/CollectionUtils.html\#methods-inherited-from-class-java.lang.Object)

`clone, equals, finalize, getClass, hashCode, notify, notifyAll, toString, wait, wait, wait`


- ## Method Details [![Link icon](https://google.github.io/adk-docs/api-reference/java/resource-files/link.svg)](https://google.github.io/adk-docs/api-reference/java/com/google/adk/utils/CollectionUtils.html\#method-detail)



- ### isNullOrEmpty [![Link icon](https://google.github.io/adk-docs/api-reference/java/resource-files/link.svg)](https://google.github.io/adk-docs/api-reference/java/com/google/adk/utils/CollectionUtils.html\#isNullOrEmpty(java.lang.Iterable))





public static<T>booleanisNullOrEmpty( [Iterable](https://docs.oracle.com/en/java/javase/17/docs/api/java.base/java/lang/Iterable.html "class or interface in java.lang") <T> iterable)



Checks if the given iterable is null or empty.

Parameters:`iterable` \- the iterable to checkReturns:true if the iterable is null or empty, false otherwise

## Example Class Usage
Packages that use [Example](https://google.github.io/adk-docs/api-reference/java/com/google/adk/examples/Example.html "class in com.google.adk.examples")

Package

Description

[com.google.adk.agents](https://google.github.io/adk-docs/api-reference/java/com/google/adk/examples/class-use/Example.html#com.google.adk.agents)

[com.google.adk.examples](https://google.github.io/adk-docs/api-reference/java/com/google/adk/examples/class-use/Example.html#com.google.adk.examples)

- ## Uses of [Example](https://google.github.io/adk-docs/api-reference/java/com/google/adk/examples/Example.html "class in com.google.adk.examples") in [com.google.adk.agents](https://google.github.io/adk-docs/api-reference/java/com/google/adk/agents/package-summary.html) [![Link icon](https://google.github.io/adk-docs/api-reference/java/resource-files/link.svg)](https://google.github.io/adk-docs/api-reference/java/com/google/adk/examples/class-use/Example.html\#com.google.adk.agents)



Methods in [com.google.adk.agents](https://google.github.io/adk-docs/api-reference/java/com/google/adk/agents/package-summary.html) with parameters of type [Example](https://google.github.io/adk-docs/api-reference/java/com/google/adk/examples/Example.html "class in com.google.adk.examples")





Modifier and Type



Method



Description



`LlmAgent.Builder`



LlmAgent.Builder.`exampleProvider(Example... examples)`









Method parameters in [com.google.adk.agents](https://google.github.io/adk-docs/api-reference/java/com/google/adk/agents/package-summary.html) with type arguments of type [Example](https://google.github.io/adk-docs/api-reference/java/com/google/adk/examples/Example.html "class in com.google.adk.examples")





Modifier and Type



Method



Description



`LlmAgent.Builder`



LlmAgent.Builder.`exampleProvider(List<Example> examples)`

- ## Uses of [Example](https://google.github.io/adk-docs/api-reference/java/com/google/adk/examples/Example.html "class in com.google.adk.examples") in [com.google.adk.examples](https://google.github.io/adk-docs/api-reference/java/com/google/adk/examples/package-summary.html) [![Link icon](https://google.github.io/adk-docs/api-reference/java/resource-files/link.svg)](https://google.github.io/adk-docs/api-reference/java/com/google/adk/examples/class-use/Example.html\#com.google.adk.examples)



Methods in [com.google.adk.examples](https://google.github.io/adk-docs/api-reference/java/com/google/adk/examples/package-summary.html) that return [Example](https://google.github.io/adk-docs/api-reference/java/com/google/adk/examples/Example.html "class in com.google.adk.examples")





Modifier and Type



Method



Description



`abstract Example`



Example.Builder.`build()`









Methods in [com.google.adk.examples](https://google.github.io/adk-docs/api-reference/java/com/google/adk/examples/package-summary.html) that return types with arguments of type [Example](https://google.github.io/adk-docs/api-reference/java/com/google/adk/examples/Example.html "class in com.google.adk.examples")





Modifier and Type



Method



Description



`List<Example>`



BaseExampleProvider.`getExamples(String query)`

## Notion Integration Guide
[Skip to content](https://google.github.io/adk-docs/tools/third-party/notion/#notion)

[Edit this page](https://github.com/google/adk-docs/edit/main/docs/tools/third-party/notion.md "Edit this page") [View source of this page](https://github.com/google/adk-docs/raw/main/docs/tools/third-party/notion.md "View source of this page")

# Notion [¶](https://google.github.io/adk-docs/tools/third-party/notion/\#notion "Permanent link")

The [Notion MCP Server](https://github.com/makenotion/notion-mcp-server)
connects your ADK agent to Notion, allowing it to search, create, and manage
pages, databases, and more within a workspace. This gives your agent the ability
to query, create, and organize content in your Notion workspace using natural
language.

## Use cases [¶](https://google.github.io/adk-docs/tools/third-party/notion/\#use-cases "Permanent link")

- **Search your workspace**: Find project pages, meeting notes, or documents
based on content.

- **Create new content**: Generate new pages for meeting notes, project plans,
or tasks.

- **Manage tasks and databases**: Update the status of a task, add items to a
database, or change properties.

- **Organize your workspace**: Move pages, duplicate templates, or add comments
to documents.


## Prerequisites [¶](https://google.github.io/adk-docs/tools/third-party/notion/\#prerequisites "Permanent link")

- Obtain a Notion integration token by going to
[Notion Integrations](https://www.notion.so/profile/integrations) in your
profile. Refer to the
[authorization documentation](https://developers.notion.com/docs/authorization)
for more details.
- Ensure relevant pages and databases can be accessed by your integration. Visit
the Access tab in your
[Notion Integration](https://www.notion.so/profile/integrations) settings,
then grant access by selecting the pages you'd like to use.

## Use with agent [¶](https://google.github.io/adk-docs/tools/third-party/notion/\#use-with-agent "Permanent link")

[Local MCP Server](https://google.github.io/adk-docs/tools/third-party/notion/#local-mcp-server)

```
from google.adk.agents import Agent
from google.adk.tools.mcp_tool import McpToolset
from google.adk.tools.mcp_tool.mcp_session_manager import StdioConnectionParams
from mcp import StdioServerParameters

NOTION_TOKEN = "YOUR_NOTION_TOKEN"

root_agent = Agent(
    model="gemini-2.5-pro",
    name="notion_agent",
    instruction="Help users get information from Notion",
    tools=[\
        McpToolset(\
            connection_params=StdioConnectionParams(\
                server_params = StdioServerParameters(\
                    command="npx",\
                    args=[\
                        "-y",\
                        "@notionhq/notion-mcp-server",\
                    ],\
                    env={\
                        "NOTION_TOKEN": NOTION_TOKEN,\
                    }\
                ),\
                timeout=30,\
            ),\
        )\
    ],
)
```

## Available tools [¶](https://google.github.io/adk-docs/tools/third-party/notion/\#available-tools "Permanent link")

| Tool | Description |
| --- | --- |
| `notion-search` | Search across your Notion workspace and connected tools like Slack, Google Drive, and Jira. Falls back to basic workspace search if AI features aren’t available. |
| `notion-fetch` | Retrieves content from a Notion page or database by its URL |
| `notion-create-pages` | Creates one or more Notion pages with specified properties and content. |
| `notion-update-page` | Update a Notion page's properties or content. |
| `notion-move-pages` | Move one or more Notion pages or databases to a new parent. |
| `notion-duplicate-page` | Duplicate a Notion page within your workspace. This action is completed async. |
| `notion-create-database` | Creates a new Notion database, initial data source, and initial view with the specified properties. |
| `notion-update-database` | Update a Notion data source's properties, name, description, or other attributes. |
| `notion-create-comment` | Add a comment to a page |
| `notion-get-comments` | Lists all comments on a specific page, including threaded discussions. |
| `notion-get-teams` | Retrieves a list of teams (teamspaces) in the current workspace. |
| `notion-get-users` | Lists all users in the workspace with their details. |
| `notion-get-user` | Retrieve your user information by ID |
| `notion-get-self` | Retrieves information about your own bot user and the Notion workspace you’re connected to. |

## Additional resources [¶](https://google.github.io/adk-docs/tools/third-party/notion/\#additional-resources "Permanent link")

- [Notion MCP Server Documentation](https://developers.notion.com/docs/mcp)
- [Notion MCP Server Repository](https://github.com/makenotion/notion-mcp-server)

Back to top

## Vertex AI Session Service
[java.lang.Object](https://docs.oracle.com/en/java/javase/17/docs/api/java.base/java/lang/Object.html "class or interface in java.lang")

com.google.adk.sessions.VertexAiSessionService

All Implemented Interfaces:`BaseSessionService`

* * *

public final class VertexAiSessionServiceextends [Object](https://docs.oracle.com/en/java/javase/17/docs/api/java.base/java/lang/Object.html "class or interface in java.lang")
implements [BaseSessionService](https://google.github.io/adk-docs/api-reference/java/com/google/adk/sessions/BaseSessionService.html "interface in com.google.adk.sessions")

TODO: Use the genai HttpApiClient and ApiResponse methods once they are public.

- ## Constructor Summary [![Link icon](https://google.github.io/adk-docs/api-reference/java/resource-files/link.svg)](https://google.github.io/adk-docs/api-reference/java/com/google/adk/sessions/VertexAiSessionService.html\#constructor-summary)



Constructors





Constructor



Description



`VertexAiSessionService()`





Creates a session service with default configuration.





`VertexAiSessionService(String project,
String location,
HttpApiClient apiClient)`





Creates a new instance of the Vertex AI Session Service with a custom ApiClient for testing.





`VertexAiSessionService(String project,
String location,
Optional<com.google.auth.oauth2.GoogleCredentials> credentials,
Optional<com.google.genai.types.HttpOptions> httpOptions)`





Creates a session service with specified project, location, credentials, and HTTP options.

- ## Method Summary [![Link icon](https://google.github.io/adk-docs/api-reference/java/resource-files/link.svg)](https://google.github.io/adk-docs/api-reference/java/com/google/adk/sessions/VertexAiSessionService.html\#method-summary)





All MethodsInstance MethodsConcrete Methods







Modifier and Type



Method



Description



`io.reactivex.rxjava3.core.Single<Event>`



`appendEvent(Session session,
Event event)`





Appends an event to an in-memory session object and updates the session's state based on the
event's state delta, if applicable.





`io.reactivex.rxjava3.core.Single<Session>`



`createSession(String appName,
String userId,
ConcurrentMap<String,Object> state,
String sessionId)`





Creates a new session with the specified parameters.





`io.reactivex.rxjava3.core.Completable`



`deleteSession(String appName,
String userId,
String sessionId)`





Deletes a specific session.





`io.reactivex.rxjava3.core.Maybe<Session>`



`getSession(String appName,
String userId,
String sessionId,
Optional<GetSessionConfig> config)`





Retrieves a specific session, optionally filtering the events included.





`io.reactivex.rxjava3.core.Single<ListEventsResponse>`



`listEvents(String appName,
String userId,
String sessionId)`





Lists the events within a specific session.





`io.reactivex.rxjava3.core.Single<ListSessionsResponse>`



`listSessions(String appName,
String userId)`





Lists sessions associated with a specific application and user.













### Methods inherited from class java.lang. [Object](https://docs.oracle.com/en/java/javase/17/docs/api/java.base/java/lang/Object.html "class or interface in java.lang") [![Link icon](https://google.github.io/adk-docs/api-reference/java/resource-files/link.svg)](https://google.github.io/adk-docs/api-reference/java/com/google/adk/sessions/VertexAiSessionService.html\#methods-inherited-from-class-java.lang.Object)

`clone, equals, finalize, getClass, hashCode, notify, notifyAll, toString, wait, wait, wait`





### Methods inherited from interface com.google.adk.sessions. [BaseSessionService](https://google.github.io/adk-docs/api-reference/java/com/google/adk/sessions/BaseSessionService.html "interface in com.google.adk.sessions") [![Link icon](https://google.github.io/adk-docs/api-reference/java/resource-files/link.svg)](https://google.github.io/adk-docs/api-reference/java/com/google/adk/sessions/VertexAiSessionService.html\#methods-inherited-from-class-com.google.adk.sessions.BaseSessionService)

`closeSession, createSession`


- ## Constructor Details [![Link icon](https://google.github.io/adk-docs/api-reference/java/resource-files/link.svg)](https://google.github.io/adk-docs/api-reference/java/com/google/adk/sessions/VertexAiSessionService.html\#constructor-detail)



- ### VertexAiSessionService [![Link icon](https://google.github.io/adk-docs/api-reference/java/resource-files/link.svg)](https://google.github.io/adk-docs/api-reference/java/com/google/adk/sessions/VertexAiSessionService.html\#%3Cinit%3E(java.lang.String,java.lang.String,com.google.adk.sessions.HttpApiClient))





publicVertexAiSessionService( [String](https://docs.oracle.com/en/java/javase/17/docs/api/java.base/java/lang/String.html "class or interface in java.lang") project,
[String](https://docs.oracle.com/en/java/javase/17/docs/api/java.base/java/lang/String.html "class or interface in java.lang") location,
[HttpApiClient](https://google.github.io/adk-docs/api-reference/java/com/google/adk/sessions/HttpApiClient.html "class in com.google.adk.sessions") apiClient)



Creates a new instance of the Vertex AI Session Service with a custom ApiClient for testing.

- ### VertexAiSessionService [![Link icon](https://google.github.io/adk-docs/api-reference/java/resource-files/link.svg)](https://google.github.io/adk-docs/api-reference/java/com/google/adk/sessions/VertexAiSessionService.html\#%3Cinit%3E())





publicVertexAiSessionService()



Creates a session service with default configuration.

- ### VertexAiSessionService [![Link icon](https://google.github.io/adk-docs/api-reference/java/resource-files/link.svg)](https://google.github.io/adk-docs/api-reference/java/com/google/adk/sessions/VertexAiSessionService.html\#%3Cinit%3E(java.lang.String,java.lang.String,java.util.Optional,java.util.Optional))





publicVertexAiSessionService( [String](https://docs.oracle.com/en/java/javase/17/docs/api/java.base/java/lang/String.html "class or interface in java.lang") project,
[String](https://docs.oracle.com/en/java/javase/17/docs/api/java.base/java/lang/String.html "class or interface in java.lang") location,
[Optional](https://docs.oracle.com/en/java/javase/17/docs/api/java.base/java/util/Optional.html "class or interface in java.util") <com.google.auth.oauth2.GoogleCredentials> credentials,
[Optional](https://docs.oracle.com/en/java/javase/17/docs/api/java.base/java/util/Optional.html "class or interface in java.util") <com.google.genai.types.HttpOptions> httpOptions)



Creates a session service with specified project, location, credentials, and HTTP options.


- ## Method Details [![Link icon](https://google.github.io/adk-docs/api-reference/java/resource-files/link.svg)](https://google.github.io/adk-docs/api-reference/java/com/google/adk/sessions/VertexAiSessionService.html\#method-detail)



- ### createSession [![Link icon](https://google.github.io/adk-docs/api-reference/java/resource-files/link.svg)](https://google.github.io/adk-docs/api-reference/java/com/google/adk/sessions/VertexAiSessionService.html\#createSession(java.lang.String,java.lang.String,java.util.concurrent.ConcurrentMap,java.lang.String))





publicio.reactivex.rxjava3.core.Single< [Session](https://google.github.io/adk-docs/api-reference/java/com/google/adk/sessions/Session.html "class in com.google.adk.sessions") >createSession( [String](https://docs.oracle.com/en/java/javase/17/docs/api/java.base/java/lang/String.html "class or interface in java.lang") appName,
[String](https://docs.oracle.com/en/java/javase/17/docs/api/java.base/java/lang/String.html "class or interface in java.lang") userId,
@Nullable
[ConcurrentMap](https://docs.oracle.com/en/java/javase/17/docs/api/java.base/java/util/concurrent/ConcurrentMap.html "class or interface in java.util.concurrent") < [String](https://docs.oracle.com/en/java/javase/17/docs/api/java.base/java/lang/String.html "class or interface in java.lang"), [Object](https://docs.oracle.com/en/java/javase/17/docs/api/java.base/java/lang/Object.html "class or interface in java.lang") > state,
@Nullable
[String](https://docs.oracle.com/en/java/javase/17/docs/api/java.base/java/lang/String.html "class or interface in java.lang") sessionId)



Description copied from interface: `BaseSessionService`



Creates a new session with the specified parameters.

Specified by:`createSession` in interface `BaseSessionService`Parameters:`appName` \- The name of the application associated with the session.`userId` \- The identifier for the user associated with the session.`state` \- An optional map representing the initial state of the session. Can be null or
empty.`sessionId` \- An optional client-provided identifier for the session. If empty or null, the
service should generate a unique ID.Returns:The newly created [`Session`](https://google.github.io/adk-docs/api-reference/java/com/google/adk/sessions/Session.html "class in com.google.adk.sessions") instance.

- ### listSessions [![Link icon](https://google.github.io/adk-docs/api-reference/java/resource-files/link.svg)](https://google.github.io/adk-docs/api-reference/java/com/google/adk/sessions/VertexAiSessionService.html\#listSessions(java.lang.String,java.lang.String))





publicio.reactivex.rxjava3.core.Single< [ListSessionsResponse](https://google.github.io/adk-docs/api-reference/java/com/google/adk/sessions/ListSessionsResponse.html "class in com.google.adk.sessions") >listSessions( [String](https://docs.oracle.com/en/java/javase/17/docs/api/java.base/java/lang/String.html "class or interface in java.lang") appName,
[String](https://docs.oracle.com/en/java/javase/17/docs/api/java.base/java/lang/String.html "class or interface in java.lang") userId)



Description copied from interface: `BaseSessionService`



Lists sessions associated with a specific application and user.



The [`Session`](https://google.github.io/adk-docs/api-reference/java/com/google/adk/sessions/Session.html "class in com.google.adk.sessions") objects in the response typically contain only metadata (like ID,
creation time) and not the full event list or state to optimize performance.



Specified by:`listSessions` in interface `BaseSessionService`Parameters:`appName` \- The name of the application.`userId` \- The identifier of the user whose sessions are to be listed.Returns:A [`ListSessionsResponse`](https://google.github.io/adk-docs/api-reference/java/com/google/adk/sessions/ListSessionsResponse.html "class in com.google.adk.sessions") containing a list of matching sessions.

- ### listEvents [![Link icon](https://google.github.io/adk-docs/api-reference/java/resource-files/link.svg)](https://google.github.io/adk-docs/api-reference/java/com/google/adk/sessions/VertexAiSessionService.html\#listEvents(java.lang.String,java.lang.String,java.lang.String))





publicio.reactivex.rxjava3.core.Single< [ListEventsResponse](https://google.github.io/adk-docs/api-reference/java/com/google/adk/sessions/ListEventsResponse.html "class in com.google.adk.sessions") >listEvents( [String](https://docs.oracle.com/en/java/javase/17/docs/api/java.base/java/lang/String.html "class or interface in java.lang") appName,
[String](https://docs.oracle.com/en/java/javase/17/docs/api/java.base/java/lang/String.html "class or interface in java.lang") userId,
[String](https://docs.oracle.com/en/java/javase/17/docs/api/java.base/java/lang/String.html "class or interface in java.lang") sessionId)



Description copied from interface: `BaseSessionService`



Lists the events within a specific session. Supports pagination via the response object.

Specified by:`listEvents` in interface `BaseSessionService`Parameters:`appName` \- The name of the application.`userId` \- The identifier of the user.`sessionId` \- The unique identifier of the session whose events are to be listed.Returns:A [`ListEventsResponse`](https://google.github.io/adk-docs/api-reference/java/com/google/adk/sessions/ListEventsResponse.html "class in com.google.adk.sessions") containing a list of events and an optional token for
retrieving the next page.

- ### getSession [![Link icon](https://google.github.io/adk-docs/api-reference/java/resource-files/link.svg)](https://google.github.io/adk-docs/api-reference/java/com/google/adk/sessions/VertexAiSessionService.html\#getSession(java.lang.String,java.lang.String,java.lang.String,java.util.Optional))





publicio.reactivex.rxjava3.core.Maybe< [Session](https://google.github.io/adk-docs/api-reference/java/com/google/adk/sessions/Session.html "class in com.google.adk.sessions") >getSession( [String](https://docs.oracle.com/en/java/javase/17/docs/api/java.base/java/lang/String.html "class or interface in java.lang") appName,
[String](https://docs.oracle.com/en/java/javase/17/docs/api/java.base/java/lang/String.html "class or interface in java.lang") userId,
[String](https://docs.oracle.com/en/java/javase/17/docs/api/java.base/java/lang/String.html "class or interface in java.lang") sessionId,
[Optional](https://docs.oracle.com/en/java/javase/17/docs/api/java.base/java/util/Optional.html "class or interface in java.util") < [GetSessionConfig](https://google.github.io/adk-docs/api-reference/java/com/google/adk/sessions/GetSessionConfig.html "class in com.google.adk.sessions") > config)



Description copied from interface: `BaseSessionService`



Retrieves a specific session, optionally filtering the events included.

Specified by:`getSession` in interface `BaseSessionService`Parameters:`appName` \- The name of the application.`userId` \- The identifier of the user.`sessionId` \- The unique identifier of the session to retrieve.`config` \- Optional configuration to filter the events returned within the session (e.g.,
limit number of recent events, filter by timestamp). If empty, default retrieval behavior
is used (potentially all events or a service-defined limit).Returns:An [`Optional`](https://docs.oracle.com/en/java/javase/17/docs/api/java.base/java/util/Optional.html "class or interface in java.util") containing the [`Session`](https://google.github.io/adk-docs/api-reference/java/com/google/adk/sessions/Session.html "class in com.google.adk.sessions") if found, otherwise [`Optional.empty()`](https://docs.oracle.com/en/java/javase/17/docs/api/java.base/java/util/Optional.html#empty() "class or interface in java.util").

- ### deleteSession [![Link icon](https://google.github.io/adk-docs/api-reference/java/resource-files/link.svg)](https://google.github.io/adk-docs/api-reference/java/com/google/adk/sessions/VertexAiSessionService.html\#deleteSession(java.lang.String,java.lang.String,java.lang.String))





publicio.reactivex.rxjava3.core.CompletabledeleteSession( [String](https://docs.oracle.com/en/java/javase/17/docs/api/java.base/java/lang/String.html "class or interface in java.lang") appName,
[String](https://docs.oracle.com/en/java/javase/17/docs/api/java.base/java/lang/String.html "class or interface in java.lang") userId,
[String](https://docs.oracle.com/en/java/javase/17/docs/api/java.base/java/lang/String.html "class or interface in java.lang") sessionId)



Description copied from interface: `BaseSessionService`



Deletes a specific session.

Specified by:`deleteSession` in interface `BaseSessionService`Parameters:`appName` \- The name of the application.`userId` \- The identifier of the user.`sessionId` \- The unique identifier of the session to delete.

- ### appendEvent [![Link icon](https://google.github.io/adk-docs/api-reference/java/resource-files/link.svg)](https://google.github.io/adk-docs/api-reference/java/com/google/adk/sessions/VertexAiSessionService.html\#appendEvent(com.google.adk.sessions.Session,com.google.adk.events.Event))





publicio.reactivex.rxjava3.core.Single< [Event](https://google.github.io/adk-docs/api-reference/java/com/google/adk/events/Event.html "class in com.google.adk.events") >appendEvent( [Session](https://google.github.io/adk-docs/api-reference/java/com/google/adk/sessions/Session.html "class in com.google.adk.sessions") session,
[Event](https://google.github.io/adk-docs/api-reference/java/com/google/adk/events/Event.html "class in com.google.adk.events") event)



Description copied from interface: `BaseSessionService`



Appends an event to an in-memory session object and updates the session's state based on the
event's state delta, if applicable.



This method primarily modifies the passed `session` object in memory. Persisting these
changes typically requires a separate call to an update/save method provided by the specific
service implementation, or might happen implicitly depending on the implementation's design.





If the event is marked as partial (e.g., `event.isPartial() == true`), it is returned
directly without modifying the session state or event list. State delta keys starting with
[`State.TEMP_PREFIX`](https://google.github.io/adk-docs/api-reference/java/com/google/adk/sessions/State.html#TEMP_PREFIX) are ignored during state updates.



Specified by:`appendEvent` in interface `BaseSessionService`Parameters:`session` \- The [`Session`](https://google.github.io/adk-docs/api-reference/java/com/google/adk/sessions/Session.html "class in com.google.adk.sessions") object to which the event should be appended (will be
mutated).`event` \- The [`Event`](https://google.github.io/adk-docs/api-reference/java/com/google/adk/events/Event.html "class in com.google.adk.events") to append.Returns:The appended [`Event`](https://google.github.io/adk-docs/api-reference/java/com/google/adk/events/Event.html "class in com.google.adk.events") instance (or the original event if it was partial).

## Sending Messages with LiveRequestQueue
[Skip to content](https://google.github.io/adk-docs/streaming/dev-guide/part2/#part-2-sending-messages-with-liverequestqueue)

[Edit this page](https://github.com/google/adk-docs/edit/main/docs/streaming/dev-guide/part2.md "Edit this page") [View source of this page](https://github.com/google/adk-docs/raw/main/docs/streaming/dev-guide/part2.md "View source of this page")

# Part 2: Sending messages with LiveRequestQueue [¶](https://google.github.io/adk-docs/streaming/dev-guide/part2/\#part-2-sending-messages-with-liverequestqueue "Permanent link")

In Part 1, you learned the four-phase lifecycle of ADK Bidi-streaming applications. This part focuses on the upstream flow—how your application sends messages to the agent using `LiveRequestQueue`.

Unlike traditional APIs where different message types require different endpoints or channels, ADK provides a single unified interface through `LiveRequestQueue` and its `LiveRequest` message model. This part covers:

- **Message types**: Sending text via `send_content()`, streaming audio/image/video via `send_realtime()`, controlling conversation turns with activity signals, and gracefully terminating sessions with control signals
- **Concurrency patterns**: Understanding async queue management and event-loop thread safety
- **Best practices**: Creating queues in async context, ensuring proper resource cleanup, and understanding message ordering guarantees
- **Troubleshooting**: Diagnosing common issues like messages not being processed and queue lifecycle problems

Understanding `LiveRequestQueue` is essential for building responsive streaming applications that handle multimodal inputs seamlessly within async event loops.

## LiveRequestQueue and LiveRequest [¶](https://google.github.io/adk-docs/streaming/dev-guide/part2/\#liverequestqueue-and-liverequest "Permanent link")

The `LiveRequestQueue` is your primary interface for sending messages to the Agent in streaming conversations. Rather than managing separate channels for text, audio, and control signals, ADK provides a unified `LiveRequest` container that handles all message types through a single, elegant API:

Source reference: [live\_request\_queue.py](https://github.com/google/adk-python/blob/0b1784e0/src/google/adk/agents/live_request_queue.py)

```
class LiveRequest(BaseModel):
    content: Optional[Content] = None           # Text-based content and structured data
    blob: Optional[Blob] = None                 # Audio/video data and binary streams
    activity_start: Optional[ActivityStart] = None  # Signal start of user activity
    activity_end: Optional[ActivityEnd] = None      # Signal end of user activity
    close: bool = False                         # Graceful connection termination signal
```

This streamlined design handles every streaming scenario you'll encounter. The `content` and `blob` fields handle different data types, the `activity_start` and `activity_end` fields enable activity signaling, and the `close` flag provides graceful termination semantics.

The `content` and `blob` fields are mutually exclusive—only one can be set per LiveRequest. While ADK does not enforce this client-side and will attempt to send both if set, the Live API backend will reject this with a validation error. ADK's convenience methods `send_content()` and `send_realtime()` automatically ensure this constraint is met by setting only one field, so **using these methods (rather than manually creating `LiveRequest` objects) is the recommended approach**.

The following diagram illustrates how different message types flow from your application through `LiveRequestQueue` methods, into `LiveRequest` containers, and finally to the Live API:

Gemini Live API

LiveRequest Container

LiveRequestQueue Methods

Application

User Text Input

Audio Stream

Activity Signals

Close Signal

send\_content

Content

send\_realtime

Blob

send\_activity\_start

ActivityStart

send\_activity\_end

ActivityEnd

close

close=True

content: Content

blob: Blob

activity\_start/end

close: bool

WebSocket Connection

## Sending Different Message Types [¶](https://google.github.io/adk-docs/streaming/dev-guide/part2/\#sending-different-message-types "Permanent link")

`LiveRequestQueue` provides convenient methods for sending different message types to the agent. This section demonstrates practical patterns for text messages, audio/video streaming, activity signals for manual turn control, and session termination.

### send\_content(): Sends Text With Turn-by-Turn [¶](https://google.github.io/adk-docs/streaming/dev-guide/part2/\#send_content-sends-text-with-turn-by-turn "Permanent link")

The `send_content()` method sends text messages in turn-by-turn mode, where each message represents a discrete conversation turn. This signals a complete turn to the model, triggering immediate response generation.

Demo implementation: [main.py:157-158](https://github.com/google/adk-samples/blob/main/python/agents/bidi-demo/app/main.py#L157-L158)

```
content = types.Content(parts=[types.Part(text=json_message["text"])])
live_request_queue.send_content(content)
```

**Using Content and Part with ADK Bidi-streaming:**

- **`Content`** (`google.genai.types.Content`): A container that represents a single message or turn in the conversation. It holds an array of `Part` objects that together compose the complete message.

- **`Part`** (`google.genai.types.Part`): An individual piece of content within a message. For ADK Bidi-streaming with Live API, you'll use:

- `text`: Text content (including code) that you send to the model

In practice, most messages use a single text Part for ADK Bidi-streaming. The multi-part structure is designed for scenarios like:
\- Mixing text with function responses (automatically handled by ADK)
\- Combining text explanations with structured data
\- Future extensibility for new content types

For Live API, multimodal inputs (audio/video) use different mechanisms (see `send_realtime()` below), not multi-part Content.

Content and Part usage in ADK Bidi-streaming

While the Gemini API `Part` type supports many fields (`inline_data`, `file_data`, `function_call`, `function_response`, etc.), most are either handled automatically by ADK or use different mechanisms in Live API:

- **Function calls**: ADK automatically handles the function calling loop - receiving function calls from the model, executing your registered functions, and sending responses back. You don't manually construct these.
- **Images/Video**: Do NOT use `send_content()` with `inline_data`. Instead, use `send_realtime(Blob(mime_type="image/jpeg", data=...))` for continuous streaming. See [Part 5: How to Use Image and Video](https://google.github.io/adk-docs/streaming/dev-guide/part5/#how-to-use-image-and-video).

### send\_realtime(): Sends Audio, Image and Video in Real-Time [¶](https://google.github.io/adk-docs/streaming/dev-guide/part2/\#send_realtime-sends-audio-image-and-video-in-real-time "Permanent link")

The `send_realtime()` method sends binary data streams—primarily audio, image and video—flow through the `Blob` type, which handles transmission in realtime mode. Unlike text content that gets processed in turn-by-turn mode, blobs are designed for continuous streaming scenarios where data arrives in chunks. You provide raw bytes, and Pydantic automatically handles base64 encoding during JSON serialization for safe network transmission (configured in `LiveRequest.model_config`). The MIME type helps the model understand the content format.

Demo implementation: [main.py:141-145](https://github.com/google/adk-samples/blob/main/python/agents/bidi-demo/app/main.py#L141-L145)

```
audio_blob = types.Blob(
    mime_type="audio/pcm;rate=16000",
    data=audio_data
)
live_request_queue.send_realtime(audio_blob)
```

Learn More

For complete details on audio, image and video specifications, formats, and best practices, see [Part 5: How to Use Audio, Image and Video](https://google.github.io/adk-docs/streaming/dev-guide/part5/).

### Activity Signals [¶](https://google.github.io/adk-docs/streaming/dev-guide/part2/\#activity-signals "Permanent link")

Activity signals (`ActivityStart`/`ActivityEnd`) can **ONLY** be sent when automatic (server-side) Voice Activity Detection is **explicitly disabled** in your `RunConfig`. Use them when your application requires manual voice activity control, such as:

- **Push-to-talk interfaces**: User explicitly controls when they're speaking (e.g., holding a button)
- **Noisy environments**: Background noise makes automatic VAD unreliable, so you use client-side VAD or manual control
- **Client-side VAD**: You implement your own VAD algorithm on the client to reduce network overhead by only sending audio when speech is detected
- **Custom interaction patterns**: Non-speech scenarios like gesture-triggered interactions or timed audio segments

**What activity signals tell the model:**

- `ActivityStart`: "The user is now speaking - start accumulating audio for processing"
- `ActivityEnd`: "The user has finished speaking - process the accumulated audio and generate a response"

Without these signals (when VAD is disabled), the model doesn't know when to start/stop listening for speech, so you must explicitly mark turn boundaries.

**Sending Activity Signals:**

```
from google.genai import types

# Manual activity signal pattern (e.g., push-to-talk)
live_request_queue.send_activity_start()  # Signal: user started speaking

# Stream audio chunks while user holds the talk button
while user_is_holding_button:
    audio_blob = types.Blob(mime_type="audio/pcm;rate=16000", data=audio_chunk)
    live_request_queue.send_realtime(audio_blob)

live_request_queue.send_activity_end()  # Signal: user stopped speaking
```

**Default behavior (automatic VAD):** If you don't send activity signals, Live API's built-in VAD automatically detects speech boundaries in the audio stream you send via `send_realtime()`. This is the recommended approach for most applications.

Learn More

For detailed comparison of automatic VAD vs manual activity signals, including when to disable VAD and best practices, see [Part 5: Voice Activity Detection](https://google.github.io/adk-docs/streaming/dev-guide/part5/#voice-activity-detection-vad).

### Control Signals [¶](https://google.github.io/adk-docs/streaming/dev-guide/part2/\#control-signals "Permanent link")

The `close` signal provides graceful termination semantics for streaming sessions. It signals the system to cleanly close the model connection and end the Bidi-stream. In ADK Bidi-streaming, your application is responsible for sending the `close` signal explicitly:

**Manual closure in BIDI mode:** When using `StreamingMode.BIDI` (Bidi-streaming), your application should manually call `close()` when the session terminates or when errors occur. This practice minimizes session resource usage.

**Automatic closure in SSE mode:** When using the legacy `StreamingMode.SSE` (not Bidi-streaming), ADK automatically calls `close()` on the queue when it receives a `turn_complete=True` event from the model (see `base_llm_flow.py:754`).

See [Part 4: Understanding RunConfig](https://google.github.io/adk-docs/streaming/dev-guide/part4/#streamingmode-bidi-or-sse) for detailed comparison and when to use each mode.

Demo implementation: [main.py:195-213](https://github.com/google/adk-samples/blob/main/python/agents/bidi-demo/app/main.py#L195-L213)

```
try:
    logger.debug("Starting asyncio.gather for upstream and downstream tasks")
    await asyncio.gather(
        upstream_task(),
        downstream_task()
    )
    logger.debug("asyncio.gather completed normally")
except WebSocketDisconnect:
    logger.debug("Client disconnected normally")
except Exception as e:
    logger.error(f"Unexpected error in streaming tasks: {e}", exc_info=True)
finally:
    # Always close the queue, even if exceptions occurred
    logger.debug("Closing live_request_queue")
    live_request_queue.close()
```

**What happens if you don't call close()?**

Although ADK cleans up local resources automatically, failing to call `close()` in BIDI mode prevents sending a graceful termination signal to the Live API, which will then receive an abrupt disconnection after certain timeout period. This can lead to "zombie" Live API sessions that remain open on the cloud service, even though your application has finished with them. These stranded sessions may significantly decrease the number of concurrent sessions your application can handle, as they continue to count against your quota limits until they eventually timeout.

Learn More

For comprehensive error handling patterns during streaming, including when to use `break` vs `continue` and handling different error types, see [Part 3: Error Events](https://google.github.io/adk-docs/streaming/dev-guide/part3/#error-events).

## Concurrency and Thread Safety [¶](https://google.github.io/adk-docs/streaming/dev-guide/part2/\#concurrency-and-thread-safety "Permanent link")

Understanding how `LiveRequestQueue` handles concurrency is essential for building reliable streaming applications. The queue is built on `asyncio.Queue`, which means it's safe for concurrent access **within the same event loop thread** (the common case), but requires special handling when called from **different threads** (the advanced case). This section explains the design choices behind `LiveRequestQueue`'s API, when you can safely use it without extra precautions, and when you need thread-safety mechanisms like `loop.call_soon_threadsafe()`.

### Async Queue Management [¶](https://google.github.io/adk-docs/streaming/dev-guide/part2/\#async-queue-management "Permanent link")

`LiveRequestQueue` uses synchronous methods (`send_content()`, `send_realtime()`) instead of async methods, even though the underlying queue is consumed asynchronously. This design choice uses `asyncio.Queue.put_nowait()` \- a non-blocking operation that doesn't require `await`.

**Why synchronous send methods?** Convenience and simplicity. You can call them from anywhere in your async code without `await`:

Demo implementation: [main.py:129-158](https://github.com/google/adk-samples/blob/main/python/agents/bidi-demo/app/main.py#L129-L158)

```
async def upstream_task() -> None:
    """Receives messages from WebSocket and sends to LiveRequestQueue."""
    while True:
        message = await websocket.receive()

        if "bytes" in message:
            audio_data = message["bytes"]
            audio_blob = types.Blob(
                mime_type="audio/pcm;rate=16000",
                data=audio_data
            )
            live_request_queue.send_realtime(audio_blob)

        elif "text" in message:
            text_data = message["text"]
            json_message = json.loads(text_data)

            if json_message.get("type") == "text":
                content = types.Content(parts=[types.Part(text=json_message["text"])])
                live_request_queue.send_content(content)
```

This pattern mixes async I/O operations with sync CPU operations naturally. The send methods return immediately without blocking, allowing your application to stay responsive.

#### Best Practice: Create Queue in Async Context [¶](https://google.github.io/adk-docs/streaming/dev-guide/part2/\#best-practice-create-queue-in-async-context "Permanent link")

Always create `LiveRequestQueue` within an async context (async function or coroutine) to ensure it uses the correct event loop:

```
# ✅ Recommended - Create in async context
async def main():
    queue = LiveRequestQueue()  # Uses existing event loop from async context
    # This is the preferred pattern - ensures queue uses the correct event loop
    # that will run your streaming operations

# ❌ Not recommended - Creates event loop automatically
queue = LiveRequestQueue()  # Works but ADK auto-creates new loop
# This works due to ADK's safety mechanism, but may cause issues with
# loop coordination in complex applications or multi-threaded scenarios
```

**Why this matters:**`LiveRequestQueue` requires an event loop to exist when instantiated. ADK includes a safety mechanism that auto-creates a loop if none exists, but relying on this can cause unexpected behavior in multi-threaded scenarios or with custom event loop configurations.

## Message Ordering Guarantees [¶](https://google.github.io/adk-docs/streaming/dev-guide/part2/\#message-ordering-guarantees "Permanent link")

`LiveRequestQueue` provides predictable message delivery behavior:

| Guarantee | Description | Impact |
| --- | --- | --- |
| **FIFO ordering** | Messages processed in send order (guaranteed by underlying `asyncio.Queue`) | Maintains conversation context and interaction consistency |
| **No coalescing** | Each message delivered independently | No automatic batching—each send operation creates one request |
| **Unbounded by default** | Queue accepts unlimited messages without blocking | **Benefit**: Simplifies client code (no blocking on send)<br>**Risk**: Memory growth if sending faster than processing<br>**Mitigation**: Monitor queue depth in production |

> **Production Tip**: For high-throughput audio/video streaming, monitor `live_request_queue._queue.qsize()` to detect backpressure. If the queue depth grows continuously, slow down your send rate or implement batching. Note: `_queue` is an internal attribute and may change in future releases; use with caution.

## Summary [¶](https://google.github.io/adk-docs/streaming/dev-guide/part2/\#summary "Permanent link")

In this part, you learned how `LiveRequestQueue` provides a unified interface for sending messages to ADK streaming agents within an async event loop. We covered the `LiveRequest` message model and explored how to send different message types: text content via `send_content()`, audio/video blobs via `send_realtime()`, activity signals for manual turn control, and control signals for graceful termination via `close()`. You also learned best practices for async queue management, creating queues in async context, resource cleanup, and message ordering. You now understand how to use `LiveRequestQueue` as the upstream communication channel in your Bidi-streaming applications, enabling users to send messages concurrently while receiving agent responses. Next, you'll learn how to handle the downstream flow—processing the events that agents generate in response to these messages.

* * *

← [Previous: Part 1 - Introduction to ADK Bidi-streaming](https://google.github.io/adk-docs/streaming/dev-guide/part1/) \| [Next: Part 3 - Event Handling with run\_live()](https://google.github.io/adk-docs/streaming/dev-guide/part3/) →

Back to top

## GKE Deployment Guide
[Skip to content](https://google.github.io/adk-docs/deploy/gke/#deploy-to-google-kubernetes-engine-gke)

[Edit this page](https://github.com/google/adk-docs/edit/main/docs/deploy/gke.md "Edit this page") [View source of this page](https://github.com/google/adk-docs/raw/main/docs/deploy/gke.md "View source of this page")

# Deploy to Google Kubernetes Engine (GKE) [¶](https://google.github.io/adk-docs/deploy/gke/\#deploy-to-google-kubernetes-engine-gke "Permanent link")

Supported in ADKPython

[GKE](https://cloud.google.com/gke) is the Google Cloud managed Kubernetes service. It allows you to deploy and manage containerized applications using Kubernetes.

To deploy your agent you will need to have a Kubernetes cluster running on GKE. You can create a cluster using the Google Cloud Console or the `gcloud` command line tool.

In this example we will deploy a simple agent to GKE. The agent will be a FastAPI application that uses `Gemini 2.0 Flash` as the LLM. We can use Vertex AI or AI Studio as the LLM provider using the Environment variable `GOOGLE_GENAI_USE_VERTEXAI`.

## Environment variables [¶](https://google.github.io/adk-docs/deploy/gke/\#environment-variables "Permanent link")

Set your environment variables as described in the [Setup and Installation](https://google.github.io/adk-docs/get-started/installation/) guide. You also need to install the `kubectl` command line tool. You can find instructions to do so in the [Google Kubernetes Engine Documentation](https://cloud.google.com/kubernetes-engine/docs/how-to/cluster-access-for-kubectl).

```
export GOOGLE_CLOUD_PROJECT=your-project-id # Your GCP project ID
export GOOGLE_CLOUD_LOCATION=us-central1 # Or your preferred location
export GOOGLE_GENAI_USE_VERTEXAI=true # Set to true if using Vertex AI
export GOOGLE_CLOUD_PROJECT_NUMBER=$(gcloud projects describe --format json $GOOGLE_CLOUD_PROJECT | jq -r ".projectNumber")
```

If you don't have `jq` installed, you can use the following command to get the project number:

```
gcloud projects describe $GOOGLE_CLOUD_PROJECT
```

And copy the project number from the output.

```
export GOOGLE_CLOUD_PROJECT_NUMBER=YOUR_PROJECT_NUMBER
```

## Enable APIs and Permissions [¶](https://google.github.io/adk-docs/deploy/gke/\#enable-apis-and-permissions "Permanent link")

Ensure you have authenticated with Google Cloud (`gcloud auth login` and `gcloud config set project <your-project-id>`).

Enable the necessary APIs for your project. You can do this using the `gcloud` command line tool.

```
gcloud services enable \
    container.googleapis.com \
    artifactregistry.googleapis.com \
    cloudbuild.googleapis.com \
    aiplatform.googleapis.com
```

Grant necessary roles to the default compute engine service account required by the `gcloud builds submit` command.

```
ROLES_TO_ASSIGN=(
    "roles/artifactregistry.writer"
    "roles/storage.objectViewer"
    "roles/logging.viewer"
    "roles/logging.logWriter"
)

for ROLE in "${ROLES_TO_ASSIGN[@]}"; do
    gcloud projects add-iam-policy-binding "${GOOGLE_CLOUD_PROJECT}" \
        --member="serviceAccount:${GOOGLE_CLOUD_PROJECT_NUMBER}-compute@developer.gserviceaccount.com" \
        --role="${ROLE}"
done
```

## Deployment payload [¶](https://google.github.io/adk-docs/deploy/gke/\#payload "Permanent link")

When you deploy your ADK agent workflow to the Google Cloud GKE,
the following content is uploaded to the service:

- Your ADK agent code
- Any dependencies declared in your ADK agent code
- ADK API server code version used by your agent

The default deployment _does not_ include the ADK web user interface libraries,
unless you specify it as deployment setting, such as the `--with_ui` option for
`adk deploy gke` command.

## Deployment options [¶](https://google.github.io/adk-docs/deploy/gke/\#deployment-options "Permanent link")

You can deploy your agent to GKE either **manually using Kubernetes manifests** or **automatically using the `adk deploy gke` command**. Choose the approach that best suits your workflow.

## Option 1: Manual Deployment using gcloud and kubectl [¶](https://google.github.io/adk-docs/deploy/gke/\#option-1-manual-deployment-using-gcloud-and-kubectl "Permanent link")

### Create a GKE cluster [¶](https://google.github.io/adk-docs/deploy/gke/\#create-a-gke-cluster "Permanent link")

You can create a GKE cluster using the `gcloud` command line tool. This example creates an Autopilot cluster named `adk-cluster` in the `us-central1` region.

> If creating a GKE Standard cluster, make sure [Workload Identity](https://cloud.google.com/kubernetes-engine/docs/how-to/workload-identity) is enabled. Workload Identity is enabled by default in an AutoPilot cluster.

```
gcloud container clusters create-auto adk-cluster \
    --location=$GOOGLE_CLOUD_LOCATION \
    --project=$GOOGLE_CLOUD_PROJECT
```

After creating the cluster, you need to connect to it using `kubectl`. This command configures `kubectl` to use the credentials for your new cluster.

```
gcloud container clusters get-credentials adk-cluster \
    --location=$GOOGLE_CLOUD_LOCATION \
    --project=$GOOGLE_CLOUD_PROJECT
```

### Create Your Agent [¶](https://google.github.io/adk-docs/deploy/gke/\#create-your-agent "Permanent link")

We will reference the `capital_agent` example defined on the [LLM agents](https://google.github.io/adk-docs/agents/llm-agents/) page.

To proceed, organize your project files as follows:

```
your-project-directory/
├── capital_agent/
│   ├── __init__.py
│   └── agent.py       # Your agent code (see "Capital Agent example" below)
├── main.py            # FastAPI application entry point
├── requirements.txt   # Python dependencies
└── Dockerfile         # Container build instructions
```

### Code files [¶](https://google.github.io/adk-docs/deploy/gke/\#code-files "Permanent link")

Create the following files (`main.py`, `requirements.txt`, `Dockerfile`, `capital_agent/agent.py`, `capital_agent/__init__.py`) in the root of `your-project-directory/`.

1. This is the Capital Agent example inside the `capital_agent` directory

capital\_agent/agent.py

```
from google.adk.agents import LlmAgent

# Define a tool function
def get_capital_city(country: str) -> str:
     """Retrieves the capital city for a given country."""
     # Replace with actual logic (e.g., API call, database lookup)
     capitals = {"france": "Paris", "japan": "Tokyo", "canada": "Ottawa"}
     return capitals.get(country.lower(), f"Sorry, I don't know the capital of {country}.")

# Add the tool to the agent
capital_agent = LlmAgent(
       model="gemini-2.0-flash",
       name="capital_agent", #name of your agent
       description="Answers user questions about the capital city of a given country.",
       instruction="""You are an agent that provides the capital city of a country... (previous instruction text)""",
       tools=[get_capital_city] # Provide the function directly
)

# ADK will discover the root_agent instance
root_agent = capital_agent
```



Mark your directory as a python package

capital\_agent/\_\_init\_\_.py

```
from . import agent
```

2. This file sets up the FastAPI application using `get_fast_api_app()` from ADK:

main.py

```
import os

import uvicorn
from fastapi import FastAPI
from google.adk.cli.fast_api import get_fast_api_app

# Get the directory where main.py is located
AGENT_DIR = os.path.dirname(os.path.abspath(__file__))
# Example session service URI (e.g., SQLite)
SESSION_SERVICE_URI = "sqlite:///./sessions.db"
# Example allowed origins for CORS
ALLOWED_ORIGINS = ["http://localhost", "http://localhost:8080", "*"]
# Set web=True if you intend to serve a web interface, False otherwise
SERVE_WEB_INTERFACE = True

# Call the function to get the FastAPI app instance
# Ensure the agent directory name ('capital_agent') matches your agent folder
app: FastAPI = get_fast_api_app(
       agents_dir=AGENT_DIR,
       session_service_uri=SESSION_SERVICE_URI,
       allow_origins=ALLOWED_ORIGINS,
       web=SERVE_WEB_INTERFACE,
)

# You can add more FastAPI routes or configurations below if needed
# Example:
# @app.get("/hello")
# async def read_root():
#     return {"Hello": "World"}

if __name__ == "__main__":
       # Use the PORT environment variable provided by Cloud Run, defaulting to 8080
       uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))
```



_Note: We specify `agent_dir` to the directory `main.py` is in and use `os.environ.get("PORT", 8080)` for Cloud Run compatibility._

3. List the necessary Python packages:

requirements.txt

```
google-adk
# Add any other dependencies your agent needs
```

4. Define the container image:

Dockerfile

```
FROM python:3.13-slim
WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

RUN adduser --disabled-password --gecos "" myuser && \
       chown -R myuser:myuser /app

COPY . .

USER myuser

ENV PATH="/home/myuser/.local/bin:$PATH"

CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port $PORT"]
```


### Build the container image [¶](https://google.github.io/adk-docs/deploy/gke/\#build-the-container-image "Permanent link")

You need to create a Google Artifact Registry repository to store your container images. You can do this using the `gcloud` command line tool.

```
gcloud artifacts repositories create adk-repo \
    --repository-format=docker \
    --location=$GOOGLE_CLOUD_LOCATION \
    --description="ADK repository"
```

Build the container image using the `gcloud` command line tool. This example builds the image and tags it as `adk-repo/adk-agent:latest`.

```
gcloud builds submit \
    --tag $GOOGLE_CLOUD_LOCATION-docker.pkg.dev/$GOOGLE_CLOUD_PROJECT/adk-repo/adk-agent:latest \
    --project=$GOOGLE_CLOUD_PROJECT \
    .
```

Verify the image is built and pushed to the Artifact Registry:

```
gcloud artifacts docker images list \
  $GOOGLE_CLOUD_LOCATION-docker.pkg.dev/$GOOGLE_CLOUD_PROJECT/adk-repo \
  --project=$GOOGLE_CLOUD_PROJECT
```

### Configure Kubernetes Service Account for Vertex AI [¶](https://google.github.io/adk-docs/deploy/gke/\#configure-kubernetes-service-account-for-vertex-ai "Permanent link")

If your agent uses Vertex AI, you need to create a Kubernetes service account with the necessary permissions. This example creates a service account named `adk-agent-sa` and binds it to the `Vertex AI User` role.

> If you are using AI Studio and accessing the model with an API key you can skip this step.

```
kubectl create serviceaccount adk-agent-sa
```

```
gcloud projects add-iam-policy-binding projects/${GOOGLE_CLOUD_PROJECT} \
    --role=roles/aiplatform.user \
    --member=principal://iam.googleapis.com/projects/${GOOGLE_CLOUD_PROJECT_NUMBER}/locations/global/workloadIdentityPools/${GOOGLE_CLOUD_PROJECT}.svc.id.goog/subject/ns/default/sa/adk-agent-sa \
    --condition=None
```

### Create the Kubernetes manifest files [¶](https://google.github.io/adk-docs/deploy/gke/\#create-the-kubernetes-manifest-files "Permanent link")

Create a Kubernetes deployment manifest file named `deployment.yaml` in your project directory. This file defines how to deploy your application on GKE.

deployment.yaml

```
cat <<  EOF > deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: adk-agent
spec:
  replicas: 1
  selector:
    matchLabels:
      app: adk-agent
  template:
    metadata:
      labels:
        app: adk-agent
    spec:
      serviceAccount: adk-agent-sa
      containers:
      - name: adk-agent
        imagePullPolicy: Always
        image: $GOOGLE_CLOUD_LOCATION-docker.pkg.dev/$GOOGLE_CLOUD_PROJECT/adk-repo/adk-agent:latest
        resources:
          limits:
            memory: "128Mi"
            cpu: "500m"
            ephemeral-storage: "128Mi"
          requests:
            memory: "128Mi"
            cpu: "500m"
            ephemeral-storage: "128Mi"
        ports:
        - containerPort: 8080
        env:
          - name: PORT
            value: "8080"
          - name: GOOGLE_CLOUD_PROJECT
            value: $GOOGLE_CLOUD_PROJECT
          - name: GOOGLE_CLOUD_LOCATION
            value: $GOOGLE_CLOUD_LOCATION
          - name: GOOGLE_GENAI_USE_VERTEXAI
            value: "$GOOGLE_GENAI_USE_VERTEXAI"
          # If using AI Studio, set GOOGLE_GENAI_USE_VERTEXAI to false and set the following:
          # - name: GOOGLE_API_KEY
          #   value: $GOOGLE_API_KEY
          # Add any other necessary environment variables your agent might need
---
apiVersion: v1
kind: Service
metadata:
  name: adk-agent
spec:
  type: LoadBalancer
  ports:
    - port: 80
      targetPort: 8080
  selector:
    app: adk-agent
EOF
```

### Deploy the Application [¶](https://google.github.io/adk-docs/deploy/gke/\#deploy-the-application "Permanent link")

Deploy the application using the `kubectl` command line tool. This command applies the deployment and service manifest files to your GKE cluster.

```
kubectl apply -f deployment.yaml
```

After a few moments, you can check the status of your deployment using:

```
kubectl get pods -l=app=adk-agent
```

This command lists the pods associated with your deployment. You should see a pod with a status of `Running`.

Once the pod is running, you can check the status of the service using:

```
kubectl get service adk-agent
```

If the output shows a `External IP`, it means your service is accessible from the internet. It may take a few minutes for the external IP to be assigned.

You can get the external IP address of your service using:

```
kubectl get svc adk-agent -o=jsonpath='{.status.loadBalancer.ingress[0].ip}'
```

## Option 2: Automated Deployment using `adk deploy gke` [¶](https://google.github.io/adk-docs/deploy/gke/\#option-2-automated-deployment-using-adk-deploy-gke "Permanent link")

ADK provides a CLI command to streamline GKE deployment. This avoids the need to manually build images, write Kubernetes manifests, or push to Artifact Registry.

#### Prerequisites [¶](https://google.github.io/adk-docs/deploy/gke/\#prerequisites "Permanent link")

Before you begin, ensure you have the following set up:

1. **A running GKE cluster:** You need an active Kubernetes cluster on Google Cloud.

2. **Required CLIs:**
   - **`gcloud` CLI:** The Google Cloud CLI must be installed, authenticated, and configured to use your target project. Run `gcloud auth login` and `gcloud config set project [YOUR_PROJECT_ID]`.
   - **kubectl:** The Kubernetes CLI must be installed to deploy the application to your cluster.
3. **Enabled Google Cloud APIs:** Make sure the following APIs are enabled in your Google Cloud project:
   - Kubernetes Engine API (`container.googleapis.com`)
   - Cloud Build API (`cloudbuild.googleapis.com`)
   - Container Registry API (`containerregistry.googleapis.com`)
4. **Required IAM Permissions:** The user or Compute Engine default service account running the command needs, at a minimum, the following roles:

5. **Kubernetes Engine Developer** (`roles/container.developer`): To interact with the GKE cluster.

6. **Storage Object Viewer** (`roles/storage.objectViewer`): To allow Cloud Build to download the source code from the Cloud Storage bucket where gcloud builds submit uploads it.

7. **Artifact Registry Create on Push Writer** (`roles/artifactregistry.createOnPushWriter`): To allow Cloud Build to push the built container image to Artifact Registry. This role also permits the on-the-fly creation of the special gcr.io repository within Artifact Registry if needed on the first push.

8. **Logs Writer** (`roles/logging.logWriter`): To allow Cloud Build to write build logs to Cloud Logging.


### The `deploy gke` Command [¶](https://google.github.io/adk-docs/deploy/gke/\#the-deploy-gke-command "Permanent link")

The command takes the path to your agent and parameters specifying the target GKE cluster.

#### Syntax [¶](https://google.github.io/adk-docs/deploy/gke/\#syntax "Permanent link")

```
adk deploy gke [OPTIONS] AGENT_PATH
```

### Arguments & Options [¶](https://google.github.io/adk-docs/deploy/gke/\#arguments-options "Permanent link")

| Argument | Description | Required |
| --- | --- | --- |
| AGENT\_PATH | The local file path to your agent's root directory. | Yes |
| --project | The Google Cloud Project ID where your GKE cluster is located. | Yes |
| --cluster\_name | The name of your GKE cluster. | Yes |
| --region | The Google Cloud region of your cluster (e.g., us-central1). | Yes |
| --with\_ui | Deploys both the agent's back-end API and a companion front-end user interface. | No |
| --log\_level | Sets the logging level for the deployment process. Options: debug, info, warning, error. | No |

### How It Works [¶](https://google.github.io/adk-docs/deploy/gke/\#how-it-works "Permanent link")

When you run the `adk deploy gke` command, the ADK performs the following steps automatically:

- Containerization: It builds a Docker container image from your agent's source code.

- Image Push: It tags the container image and pushes it to your project's Artifact Registry.

- Manifest Generation: It dynamically generates the necessary Kubernetes manifest files (a `Deployment` and a `Service`).

- Cluster Deployment: It applies these manifests to your specified GKE cluster, which triggers the following:


The `Deployment` instructs GKE to pull the container image from Artifact Registry and run it in one or more Pods.

The `Service` creates a stable network endpoint for your agent. By default, this is a LoadBalancer service, which provisions a public IP address to expose your agent to the internet.

### Example Usage [¶](https://google.github.io/adk-docs/deploy/gke/\#example-usage "Permanent link")

Here is a practical example of deploying an agent located at `~/agents/multi_tool_agent/` to a GKE cluster named test.

```
adk deploy gke \
    --project myproject \
    --cluster_name test \
    --region us-central1 \
    --with_ui \
    --log_level info \
    ~/agents/multi_tool_agent/
```

### Verifying Your Deployment [¶](https://google.github.io/adk-docs/deploy/gke/\#verifying-your-deployment "Permanent link")

If you used `adk deploy gke`, verify the deployment using `kubectl`:

1. Check the Pods: Ensure your agent's pods are in the Running state.

```
kubectl get pods
```

You should see output like `adk-default-service-name-xxxx-xxxx ... 1/1 Running` in the default namespace.

1. Find the External IP: Get the public IP address for your agent's service.

```
kubectl get service
NAME                       TYPE           CLUSTER-IP      EXTERNAL-IP     PORT(S)        AGE
adk-default-service-name   LoadBalancer   34.118.228.70   34.63.153.253   80:32581/TCP   5d20h
```

We can navigate to the external IP and interact with the agent via UI
![alt text](https://google.github.io/adk-docs/assets/agent-gke-deployment.png)

## Testing your agent [¶](https://google.github.io/adk-docs/deploy/gke/\#testing-your-agent "Permanent link")

Once your agent is deployed to GKE, you can interact with it via the deployed UI (if enabled) or directly with its API endpoints using tools like `curl`. You'll need the service URL provided after deployment.

[UI Testing](https://google.github.io/adk-docs/deploy/gke/#ui-testing_1)[API Testing (curl)](https://google.github.io/adk-docs/deploy/gke/#api-testing-curl_1)

### UI Testing [¶](https://google.github.io/adk-docs/deploy/gke/\#ui-testing "Permanent link")

If you deployed your agent with the UI enabled:

You can test your agent by simply navigating to the kubernetes service URL in your web browser.

The ADK dev UI allows you to interact with your agent, manage sessions, and view execution details directly in the browser.

To verify your agent is working as intended, you can:

1. Select your agent from the dropdown menu.
2. Type a message and verify that you receive an expected response from your agent.

If you experience any unexpected behavior, check the pod logs for your agent using:

```
kubectl logs -l app=adk-agent
```

### API Testing (curl) [¶](https://google.github.io/adk-docs/deploy/gke/\#api-testing-curl "Permanent link")

You can interact with the agent's API endpoints using tools like `curl`. This is useful for programmatic interaction or if you deployed without the UI.

#### Set the application URL [¶](https://google.github.io/adk-docs/deploy/gke/\#set-the-application-url "Permanent link")

Replace the example URL with the actual URL of your deployed Cloud Run service.

```
export APP_URL=$(kubectl get service adk-agent -o jsonpath='{.status.loadBalancer.ingress[0].ip}')
```

#### List available apps [¶](https://google.github.io/adk-docs/deploy/gke/\#list-available-apps "Permanent link")

Verify the deployed application name.

```
curl -X GET $APP_URL/list-apps
```

_(Adjust the `app_name` in the following commands based on this output if needed. The default is often the agent directory name, e.g., `capital_agent`)_.

#### Create or Update a Session [¶](https://google.github.io/adk-docs/deploy/gke/\#create-or-update-a-session "Permanent link")

Initialize or update the state for a specific user and session. Replace `capital_agent` with your actual app name if different. The values `user_123` and `session_abc` are example identifiers; you can replace them with your desired user and session IDs.

```
curl -X POST \
    $APP_URL/apps/capital_agent/users/user_123/sessions/session_abc \
    -H "Content-Type: application/json" \
    -d '{"preferred_language": "English", "visit_count": 5}'
```

#### Run the Agent [¶](https://google.github.io/adk-docs/deploy/gke/\#run-the-agent "Permanent link")

Send a prompt to your agent. Replace `capital_agent` with your app name and adjust the user/session IDs and prompt as needed.

```
curl -X POST $APP_URL/run_sse \
    -H "Content-Type: application/json" \
    -d '{
    "app_name": "capital_agent",
    "user_id": "user_123",
    "session_id": "session_abc",
    "new_message": {
        "role": "user",
        "parts": [{\
        "text": "What is the capital of Canada?"\
        }]
    },
    "streaming": false
    }'
```

- Set `"streaming": true` if you want to receive Server-Sent Events (SSE).
- The response will contain the agent's execution events, including the final answer.

## Troubleshooting [¶](https://google.github.io/adk-docs/deploy/gke/\#troubleshooting "Permanent link")

These are some common issues you might encounter when deploying your agent to GKE:

### 403 Permission Denied for `Gemini 2.0 Flash` [¶](https://google.github.io/adk-docs/deploy/gke/\#403-permission-denied-for-gemini-20-flash "Permanent link")

This usually means that the Kubernetes service account does not have the necessary permission to access the Vertex AI API. Ensure that you have created the service account and bound it to the `Vertex AI User` role as described in the [Configure Kubernetes Service Account for Vertex AI](https://google.github.io/adk-docs/deploy/gke/#configure-kubernetes-service-account-for-vertex-ai) section. If you are using AI Studio, ensure that you have set the `GOOGLE_API_KEY` environment variable in the deployment manifest and it is valid.

### 404 or Not Found response [¶](https://google.github.io/adk-docs/deploy/gke/\#404-or-not-found-response "Permanent link")

This usually means there is an error in your request. Check the application logs to diagnose the problem.

```
export POD_NAME=$(kubectl get pod -l app=adk-agent -o jsonpath='{.items[0].metadata.name}')
kubectl logs $POD_NAME
```

### Attempt to write a readonly database [¶](https://google.github.io/adk-docs/deploy/gke/\#attempt-to-write-a-readonly-database "Permanent link")

You might see there is no session id created in the UI and the agent does not respond to any messages. This is usually caused by the SQLite database being read-only. This can happen if you run the agent locally and then create the container image which copies the SQLite database into the container. The database is then read-only in the container.

```
sqlalchemy.exc.OperationalError: (sqlite3.OperationalError) attempt to write a readonly database
[SQL: UPDATE app_states SET state=?, update_time=CURRENT_TIMESTAMP WHERE app_states.app_name = ?]
```

To fix this issue, you can either:

Delete the SQLite database file from your local machine before building the container image. This will create a new SQLite database when the container is started.

```
rm -f sessions.db
```

or (recommended) you can add a `.dockerignore` file to your project directory to exclude the SQLite database from being copied into the container image.

.dockerignore

```
sessions.db
```

Build the container image abd deploy the application again.

### Insufficent Permission to Stream Logs `ERROR: (gcloud.builds.submit)` [¶](https://google.github.io/adk-docs/deploy/gke/\#insufficent-permission-to-stream-logs-error-gcloudbuildssubmit "Permanent link")

This error can occur when you don't have sufficient permissions to stream build logs, or your VPC-SC security policy restricts access to the default logs bucket.

To check the progress of the build, follow the link provided in the error message or navigate to the Cloud Build page in the Google Cloud Console.

You can also verify the image was built and pushed to the Artifact Registry using the command under the [Build the container image](https://google.github.io/adk-docs/deploy/gke/#build-the-container-image) section.

### Gemini-2.0-Flash Not Supported in Live Api [¶](https://google.github.io/adk-docs/deploy/gke/\#gemini-20-flash-not-supported-in-live-api "Permanent link")

When using the ADK Dev UI for your deployed agent, text-based chat works, but voice (e.g., clicking the microphone button) fail. You might see a `websockets.exceptions.ConnectionClosedError` in the pod logs indicating that your model is "not supported in the live api".

This error occurs because the agent is configured with a model (like `gemini-2.0-flash` in the example) that does not support the Gemini Live API. The Live API is required for real-time, bidirectional streaming of audio and video.

## Cleanup [¶](https://google.github.io/adk-docs/deploy/gke/\#cleanup "Permanent link")

To delete the GKE cluster and all associated resources, run:

```
gcloud container clusters delete adk-cluster \
    --location=$GOOGLE_CLOUD_LOCATION \
    --project=$GOOGLE_CLOUD_PROJECT
```

To delete the Artifact Registry repository, run:

```
gcloud artifacts repositories delete adk-repo \
    --location=$GOOGLE_CLOUD_LOCATION \
    --project=$GOOGLE_CLOUD_PROJECT
```

You can also delete the project if you no longer need it. This will delete all resources associated with the project, including the GKE cluster, Artifact Registry repository, and any other resources you created.

```
gcloud projects delete $GOOGLE_CLOUD_PROJECT
```

Back to top

## Reflect and Retry Plugin
[Skip to content](https://google.github.io/adk-docs/plugins/reflect-and-retry/#reflect-and-retry-tool-plugin)

[Edit this page](https://github.com/google/adk-docs/edit/main/docs/plugins/reflect-and-retry.md "Edit this page") [View source of this page](https://github.com/google/adk-docs/raw/main/docs/plugins/reflect-and-retry.md "View source of this page")

# Reflect and Retry Tool Plugin [¶](https://google.github.io/adk-docs/plugins/reflect-and-retry/\#reflect-and-retry-tool-plugin "Permanent link")

Supported in ADKPython v1.16.0

The Reflect and Retry Tool plugin can help your agent recover from error
responses from ADK [Tools](https://google.github.io/adk-docs/tools-custom/) and automatically retry the
tool request. This plugin intercepts tool failures, provides structured guidance
to the AI model for reflection and correction, and retries the operation up to a
configurable limit. This plugin can help you build more resilience into your
agent workflows, including the following capabilities:

- **Concurrency safe**: Uses locking to safely handle parallel tool executions.
- **Configurable scope**: Tracks failures per-invocation (default) or globally.
- **Granular tracking**: Failure counts are tracked per-tool.
- **Custom error extraction**: Supports detecting errors in normal tool responses.

## Add Reflect and Retry Plugin [¶](https://google.github.io/adk-docs/plugins/reflect-and-retry/\#add-reflect-and-retry-plugin "Permanent link")

Add this plugin to your ADK workflow by adding it to the plugins setting of your
ADK project's App object, as shown below:

```
from google.adk.apps.app import App
from google.adk.plugins import ReflectAndRetryToolPlugin

app = App(
    name="my_app",
    root_agent=root_agent,
    plugins=[\
        ReflectAndRetryToolPlugin(max_retries=3),\
    ],
)
```

With this configuration, if any tool called by an agent returns an error, the
request is updated and tried again, up to a maximum of 3 attempts, per tool.

## Configuration settings [¶](https://google.github.io/adk-docs/plugins/reflect-and-retry/\#configuration-settings "Permanent link")

The Reflect and Retry Plugin has the following configuration options:

- **`max_retries`**: (optional) Total number of additional attempts the system
makes to receive a non-error response. Default value is 3.
- **`throw_exception_if_retry_exceeded`**: (optional) If set to `False`, the
system does not raise an error if the final retry attempt fails. Default
value is `True`.
- **`tracking_scope`**: (optional)
  - **`TrackingScope.INVOCATION`**: Track tool failures across a single
     invocation and user. This value is the default.
  - **`TrackingScope.GLOBAL`**: Track tool failures across all invocations
     and all users.

### Advanced configuration [¶](https://google.github.io/adk-docs/plugins/reflect-and-retry/\#advanced-configuration "Permanent link")

You can further modify the behavior of this plugin by extending the
`ReflectAndRetryToolPlugin` class. The following code sample
demonstrates a simple extension of the behavior by selecting
responses with an error status:

```
class CustomRetryPlugin(ReflectAndRetryToolPlugin):
  async def extract_error_from_result(self, *, tool, tool_args,tool_context,
  result):
    # Detect error based on response content
    if result.get('status') == 'error':
        return result
    return None  # No error detected

# add this modified plugin to your App object:
error_handling_plugin = CustomRetryPlugin(max_retries=5)
```

## Next steps [¶](https://google.github.io/adk-docs/plugins/reflect-and-retry/\#next-steps "Permanent link")

For complete code samples using the Reflect and Retry plugin, see the following:

- [Basic](https://github.com/google/adk-python/tree/main/contributing/samples/plugin_reflect_tool_retry/basic)
code sample
- [Hallucinating function name](https://github.com/google/adk-python/tree/main/contributing/samples/plugin_reflect_tool_retry/hallucinating_func_name)
code sample

Back to top

## ADK Memory Package Usage
Packages that use [com.google.adk.memory](https://google.github.io/adk-docs/api-reference/java/com/google/adk/memory/package-summary.html)

Package

Description

[com.google.adk.memory](https://google.github.io/adk-docs/api-reference/java/com/google/adk/memory/package-use.html#com.google.adk.memory)

- Classes in [com.google.adk.memory](https://google.github.io/adk-docs/api-reference/java/com/google/adk/memory/package-summary.html) used by [com.google.adk.memory](https://google.github.io/adk-docs/api-reference/java/com/google/adk/memory/package-summary.html)





Class



Description



[BaseMemoryService](https://google.github.io/adk-docs/api-reference/java/com/google/adk/memory/class-use/BaseMemoryService.html#com.google.adk.memory)





Base contract for memory services.





[MemoryEntry](https://google.github.io/adk-docs/api-reference/java/com/google/adk/memory/class-use/MemoryEntry.html#com.google.adk.memory)





Represents one memory entry.





[MemoryEntry.Builder](https://google.github.io/adk-docs/api-reference/java/com/google/adk/memory/class-use/MemoryEntry.Builder.html#com.google.adk.memory)





Builder for [`MemoryEntry`](https://google.github.io/adk-docs/api-reference/java/com/google/adk/memory/MemoryEntry.html "class in com.google.adk.memory").





[SearchMemoryResponse](https://google.github.io/adk-docs/api-reference/java/com/google/adk/memory/class-use/SearchMemoryResponse.html#com.google.adk.memory)





Represents the response from a memory search.





[SearchMemoryResponse.Builder](https://google.github.io/adk-docs/api-reference/java/com/google/adk/memory/class-use/SearchMemoryResponse.Builder.html#com.google.adk.memory)





Builder for [`SearchMemoryResponse`](https://google.github.io/adk-docs/api-reference/java/com/google/adk/memory/SearchMemoryResponse.html "class in com.google.adk.memory").

## Java Conversion Utilities
No usage of com.google.adk.tools.mcp.ConversionUtils

## GenAI Network Package
* * *

package com.google.adk.network

- Related Packages





Package



Description



[com.google.adk](https://google.github.io/adk-docs/api-reference/java/com/google/adk/package-summary.html)

- Classes





Class



Description



[ApiResponse](https://google.github.io/adk-docs/api-reference/java/com/google/adk/network/ApiResponse.html "class in com.google.adk.network")





The API response contains a response to a call to the GenAI APIs.





[HttpApiResponse](https://google.github.io/adk-docs/api-reference/java/com/google/adk/network/HttpApiResponse.html "class in com.google.adk.network")





Wraps a real HTTP response to expose the methods needed by the GenAI SDK.

## AgentLoadingProperties Overview
Packages that use [AgentLoadingProperties](https://google.github.io/adk-docs/api-reference/java/com/google/adk/web/config/AgentLoadingProperties.html "class in com.google.adk.web.config")

Package

Description

[com.google.adk.web](https://google.github.io/adk-docs/api-reference/java/com/google/adk/web/config/class-use/AgentLoadingProperties.html#com.google.adk.web)

- ## Uses of [AgentLoadingProperties](https://google.github.io/adk-docs/api-reference/java/com/google/adk/web/config/AgentLoadingProperties.html "class in com.google.adk.web.config") in [com.google.adk.web](https://google.github.io/adk-docs/api-reference/java/com/google/adk/web/package-summary.html) [![Link icon](https://google.github.io/adk-docs/api-reference/java/resource-files/link.svg)](https://google.github.io/adk-docs/api-reference/java/com/google/adk/web/config/class-use/AgentLoadingProperties.html\#com.google.adk.web)



Methods in [com.google.adk.web](https://google.github.io/adk-docs/api-reference/java/com/google/adk/web/package-summary.html) with parameters of type [AgentLoadingProperties](https://google.github.io/adk-docs/api-reference/java/com/google/adk/web/config/AgentLoadingProperties.html "class in com.google.adk.web.config")





Modifier and Type



Method



Description



`Map<String, BaseAgent>`



AdkWebServer.`loadedAgentRegistry(AgentCompilerLoader loader,
AgentLoadingProperties props)`









Constructors in [com.google.adk.web](https://google.github.io/adk-docs/api-reference/java/com/google/adk/web/package-summary.html) with parameters of type [AgentLoadingProperties](https://google.github.io/adk-docs/api-reference/java/com/google/adk/web/config/AgentLoadingProperties.html "class in com.google.adk.web.config")





Modifier



Constructor



Description



``



`AgentCompilerLoader(AgentLoadingProperties properties)`





Initializes the loader with agent configuration and proactively attempts to locate the ADK core
JAR.

## Add Session to Eval Set
[java.lang.Object](https://docs.oracle.com/en/java/javase/17/docs/api/java.base/java/lang/Object.html "class or interface in java.lang")

com.google.adk.web.AdkWebServer.AddSessionToEvalSetRequest

Enclosing class:`AdkWebServer`

* * *

public static class AdkWebServer.AddSessionToEvalSetRequestextends [Object](https://docs.oracle.com/en/java/javase/17/docs/api/java.base/java/lang/Object.html "class or interface in java.lang")

DTO for POST /apps/{appName}/eval\_sets/{evalSetId}/add-session requests. Contains information
to associate a session with an evaluation set.

- ## Field Summary [![Link icon](https://google.github.io/adk-docs/api-reference/java/resource-files/link.svg)](https://google.github.io/adk-docs/api-reference/java/com/google/adk/web/AdkWebServer.AddSessionToEvalSetRequest.html\#field-summary)



Fields





Modifier and Type



Field



Description



`String`



`evalId`







`String`



`sessionId`







`String`



`userId`

- ## Constructor Summary [![Link icon](https://google.github.io/adk-docs/api-reference/java/resource-files/link.svg)](https://google.github.io/adk-docs/api-reference/java/com/google/adk/web/AdkWebServer.AddSessionToEvalSetRequest.html\#constructor-summary)



Constructors





Constructor



Description



`AddSessionToEvalSetRequest()`

- ## Method Summary [![Link icon](https://google.github.io/adk-docs/api-reference/java/resource-files/link.svg)](https://google.github.io/adk-docs/api-reference/java/com/google/adk/web/AdkWebServer.AddSessionToEvalSetRequest.html\#method-summary)





All MethodsInstance MethodsConcrete Methods







Modifier and Type



Method



Description



`String`



`getEvalId()`







`String`



`getSessionId()`







`String`



`getUserId()`















### Methods inherited from class java.lang. [Object](https://docs.oracle.com/en/java/javase/17/docs/api/java.base/java/lang/Object.html "class or interface in java.lang") [![Link icon](https://google.github.io/adk-docs/api-reference/java/resource-files/link.svg)](https://google.github.io/adk-docs/api-reference/java/com/google/adk/web/AdkWebServer.AddSessionToEvalSetRequest.html\#methods-inherited-from-class-java.lang.Object)

`clone, equals, finalize, getClass, hashCode, notify, notifyAll, toString, wait, wait, wait`


- ## Field Details [![Link icon](https://google.github.io/adk-docs/api-reference/java/resource-files/link.svg)](https://google.github.io/adk-docs/api-reference/java/com/google/adk/web/AdkWebServer.AddSessionToEvalSetRequest.html\#field-detail)



- ### evalId [![Link icon](https://google.github.io/adk-docs/api-reference/java/resource-files/link.svg)](https://google.github.io/adk-docs/api-reference/java/com/google/adk/web/AdkWebServer.AddSessionToEvalSetRequest.html\#evalId)





public[String](https://docs.oracle.com/en/java/javase/17/docs/api/java.base/java/lang/String.html "class or interface in java.lang")evalId

- ### sessionId [![Link icon](https://google.github.io/adk-docs/api-reference/java/resource-files/link.svg)](https://google.github.io/adk-docs/api-reference/java/com/google/adk/web/AdkWebServer.AddSessionToEvalSetRequest.html\#sessionId)





public[String](https://docs.oracle.com/en/java/javase/17/docs/api/java.base/java/lang/String.html "class or interface in java.lang")sessionId

- ### userId [![Link icon](https://google.github.io/adk-docs/api-reference/java/resource-files/link.svg)](https://google.github.io/adk-docs/api-reference/java/com/google/adk/web/AdkWebServer.AddSessionToEvalSetRequest.html\#userId)





public[String](https://docs.oracle.com/en/java/javase/17/docs/api/java.base/java/lang/String.html "class or interface in java.lang")userId


- ## Constructor Details [![Link icon](https://google.github.io/adk-docs/api-reference/java/resource-files/link.svg)](https://google.github.io/adk-docs/api-reference/java/com/google/adk/web/AdkWebServer.AddSessionToEvalSetRequest.html\#constructor-detail)



- ### AddSessionToEvalSetRequest [![Link icon](https://google.github.io/adk-docs/api-reference/java/resource-files/link.svg)](https://google.github.io/adk-docs/api-reference/java/com/google/adk/web/AdkWebServer.AddSessionToEvalSetRequest.html\#%3Cinit%3E())





publicAddSessionToEvalSetRequest()


- ## Method Details [![Link icon](https://google.github.io/adk-docs/api-reference/java/resource-files/link.svg)](https://google.github.io/adk-docs/api-reference/java/com/google/adk/web/AdkWebServer.AddSessionToEvalSetRequest.html\#method-detail)



- ### getEvalId [![Link icon](https://google.github.io/adk-docs/api-reference/java/resource-files/link.svg)](https://google.github.io/adk-docs/api-reference/java/com/google/adk/web/AdkWebServer.AddSessionToEvalSetRequest.html\#getEvalId())





public[String](https://docs.oracle.com/en/java/javase/17/docs/api/java.base/java/lang/String.html "class or interface in java.lang")getEvalId()

- ### getSessionId [![Link icon](https://google.github.io/adk-docs/api-reference/java/resource-files/link.svg)](https://google.github.io/adk-docs/api-reference/java/com/google/adk/web/AdkWebServer.AddSessionToEvalSetRequest.html\#getSessionId())





public[String](https://docs.oracle.com/en/java/javase/17/docs/api/java.base/java/lang/String.html "class or interface in java.lang")getSessionId()

- ### getUserId [![Link icon](https://google.github.io/adk-docs/api-reference/java/resource-files/link.svg)](https://google.github.io/adk-docs/api-reference/java/com/google/adk/web/AdkWebServer.AddSessionToEvalSetRequest.html\#getUserId())





public[String](https://docs.oracle.com/en/java/javase/17/docs/api/java.base/java/lang/String.html "class or interface in java.lang")getUserId()

## ADK Tools Overview
[Skip to content](https://google.github.io/adk-docs/tools/#tools-for-agents)

[Edit this page](https://github.com/google/adk-docs/edit/main/docs/tools/index.md "Edit this page") [View source of this page](https://github.com/google/adk-docs/raw/main/docs/tools/index.md "View source of this page")

# Tools for Agents [¶](https://google.github.io/adk-docs/tools/\#tools-for-agents "Permanent link")

Check out the following pre-built tools that you can use with ADK agents:

### Gemini tools [¶](https://google.github.io/adk-docs/tools/\#gemini-tools "Permanent link")

[![Google Search](https://google.github.io/adk-docs/assets/tools-google-search.png)\\
\\
**Google Search** \\
\\
Perform web searches using Google Search with Gemini](https://google.github.io/adk-docs/tools/built-in-tools/#google-search) [![Gemini](https://google.github.io/adk-docs/assets/tools-gemini.png)\\
\\
**Code Execution** \\
\\
Execute code using Gemini models](https://google.github.io/adk-docs/tools/built-in-tools/#code-execution)

### Google Cloud tools [¶](https://google.github.io/adk-docs/tools/\#google-cloud-tools "Permanent link")

[![Apigee](https://google.github.io/adk-docs/assets/tools-apigee.png)\\
\\
**Apigee API Hub** \\
\\
Turn any documented API from Apigee API hub into a tool](https://google.github.io/adk-docs/tools/google-cloud-tools/#apigee-api-hub-tools) [![Apigee Integration](https://google.github.io/adk-docs/assets/tools-apigee-integration.png)\\
\\
**Application Integration** \\
\\
Link your agents to enterprise apps using Integration Connectors](https://google.github.io/adk-docs/tools/google-cloud-tools/#application-integration-tools) [![BigQuery](https://google.github.io/adk-docs/assets/tools-bigquery.png)\\
\\
**BigQuery Agent Analytics** \\
\\
Analyze and debug agent behavior at scale.](https://google.github.io/adk-docs/tools/google-cloud/bigquery-agent-analytics/) [![BigQuery](https://google.github.io/adk-docs/assets/tools-bigquery.png)\\
\\
**BigQuery Tools** \\
\\
Connect with BigQuery to retrieve data and perform analysis](https://google.github.io/adk-docs/tools/built-in-tools/#bigquery) [![Bigtable](https://google.github.io/adk-docs/assets/tools-bigtable.png)\\
\\
**Bigtable Tools** \\
\\
Interact with Bigtable to retrieve data and and execute SQL](https://google.github.io/adk-docs/tools/built-in-tools/#bigtable) [![Google Kubernetes Engine](https://google.github.io/adk-docs/assets/tools-gke.png)\\
\\
**GKE Code Executor** \\
\\
Run AI-generated code in a secure and scalable GKE environment](https://google.github.io/adk-docs/tools/built-in-tools/#gke-code-executor) [![Spanner](https://google.github.io/adk-docs/assets/tools-spanner.png)\\
\\
**Spanner Tools** \\
\\
Interact with Spanner to retrieve data, search, and execute SQL](https://google.github.io/adk-docs/tools/built-in-tools/#spanner) [![MCP Toolbox for Databases](https://google.github.io/adk-docs/assets/tools-mcp-toolbox-for-databases.png)\\
\\
**MCP Toolbox for Databases** \\
\\
Connect over 30 different data sources to your agents](https://google.github.io/adk-docs/tools/google-cloud/mcp-toolbox-for-databases/) [![Vertex AI](https://google.github.io/adk-docs/assets/tools-vertex-ai.png)\\
\\
**Vertex AI RAG Engine** \\
\\
Perform private data retrieval using Vertex AI RAG Engine](https://google.github.io/adk-docs/tools/built-in-tools/#vertex-ai-rag-engine) [![Vertex AI](https://google.github.io/adk-docs/assets/tools-vertex-ai.png)\\
\\
**Vertex AI Search** \\
\\
Search across your private, configured data stores in Vertex AI Search](https://google.github.io/adk-docs/tools/built-in-tools/#vertex-ai-search)

### Third-party tools [¶](https://google.github.io/adk-docs/tools/\#third-party-tools "Permanent link")

[![AgentQL](https://google.github.io/adk-docs/assets/tools-agentql.png)\\
\\
**AgentQL** \\
\\
Extract resilient, structured web data using natural language](https://google.github.io/adk-docs/tools/third-party/agentql/) [![Bright Data](https://google.github.io/adk-docs/assets/tools-bright-data.png)\\
\\
**Bright Data** \\
\\
One MCP for the web - connect your AI to real web data](https://google.github.io/adk-docs/tools/third-party/bright-data/) [![Browserbase](https://google.github.io/adk-docs/assets/tools-browserbase.png)\\
\\
**Browserbase** \\
\\
Powers web browsing capabilities for AI agents](https://google.github.io/adk-docs/tools/third-party/browserbase/) [![Exa](https://google.github.io/adk-docs/assets/tools-exa.png)\\
\\
**Exa** \\
\\
Search and extract structured content from websites and live data](https://google.github.io/adk-docs/tools/third-party/exa/) [![Firecrawl](https://google.github.io/adk-docs/assets/tools-firecrawl.png)\\
\\
**Firecrawl** \\
\\
Empower your AI apps with clean data from any website](https://google.github.io/adk-docs/tools/third-party/firecrawl/) [![GitHub](https://google.github.io/adk-docs/assets/tools-github.png)\\
\\
**GitHub** \\
\\
Analyze code, manage issues and PRs, and automate workflows](https://google.github.io/adk-docs/tools/third-party/github/) [![GitLab](https://google.github.io/adk-docs/assets/tools-gitlab.png)\\
\\
**GitLab** \\
\\
Perform semantic code search, inspect pipelines, manage merge requests](https://google.github.io/adk-docs/tools/third-party/gitlab/) [![Hugging Face](https://google.github.io/adk-docs/assets/tools-hugging-face.png)\\
\\
**Hugging Face** \\
\\
Access models, datasets, research papers, and AI tools](https://google.github.io/adk-docs/tools/third-party/hugging-face/) [![Notion](https://google.github.io/adk-docs/assets/tools-notion.png)\\
\\
**Notion** \\
\\
Search workspaces, create pages, and manage tasks and databases](https://google.github.io/adk-docs/tools/third-party/notion/) [![ScrapeGraphAI](https://google.github.io/adk-docs/assets/tools-scrapegraphai.png)\\
\\
**ScrapeGraphAI** \\
\\
AI-powered web scraping, crawling, and data extraction](https://google.github.io/adk-docs/tools/third-party/scrapegraphai/) [![Tavily](https://google.github.io/adk-docs/assets/tools-tavily.png)\\
\\
**Tavily** \\
\\
Provides real-time web search, extraction, and crawling tools](https://google.github.io/adk-docs/tools/third-party/tavily/)

## Build your tools [¶](https://google.github.io/adk-docs/tools/\#build-your-tools "Permanent link")

If the above tools don't meet your needs, you can build tools for your ADK
workflows using the following guides:

- **[Function Tools](https://google.github.io/adk-docs/tools-custom/function-tools/)**: Build custom tools for
your specific ADK agent needs.
- **[MCP Tools](https://google.github.io/adk-docs/tools/mcp-tools/)**: Connect MCP servers as tools
for your ADK agents.
- **[OpenAPI Integration](https://google.github.io/adk-docs/tools-custom/openapi-tools/)**:
Generate callable tools directly from an OpenAPI Specification.

Back to top

## LlmRequest Builder Class
[java.lang.Object](https://docs.oracle.com/en/java/javase/17/docs/api/java.base/java/lang/Object.html "class or interface in java.lang")

com.google.adk.models.LlmRequest.Builder

Enclosing class:`LlmRequest`

* * *

public abstract static class LlmRequest.Builderextends [Object](https://docs.oracle.com/en/java/javase/17/docs/api/java.base/java/lang/Object.html "class or interface in java.lang")

Builder for constructing [`LlmRequest`](https://google.github.io/adk-docs/api-reference/java/com/google/adk/models/LlmRequest.html "class in com.google.adk.models") instances.

- ## Constructor Summary [![Link icon](https://google.github.io/adk-docs/api-reference/java/resource-files/link.svg)](https://google.github.io/adk-docs/api-reference/java/com/google/adk/models/LlmRequest.Builder.html\#constructor-summary)



Constructors





Constructor



Description



`Builder()`

- ## Method Summary [![Link icon](https://google.github.io/adk-docs/api-reference/java/resource-files/link.svg)](https://google.github.io/adk-docs/api-reference/java/com/google/adk/models/LlmRequest.Builder.html\#method-summary)





All MethodsInstance MethodsAbstract MethodsConcrete Methods







Modifier and Type



Method



Description



`final LlmRequest.Builder`



`appendInstructions(List<String> instructions)`







`final LlmRequest.Builder`



`appendTools(List<BaseTool> tools)`







`abstract LlmRequest`



`build()`







`abstract Optional<com.google.genai.types.GenerateContentConfig>`



`config()`







`abstract LlmRequest.Builder`



`config(com.google.genai.types.GenerateContentConfig config)`







`abstract LlmRequest.Builder`



`contents(List<com.google.genai.types.Content> contents)`







`abstract LlmRequest.Builder`



`liveConnectConfig(com.google.genai.types.LiveConnectConfig liveConnectConfig)`







`abstract LlmRequest.Builder`



`model(String model)`







`final LlmRequest.Builder`



`outputSchema(com.google.genai.types.Schema schema)`





Sets the output schema for the LLM response.













### Methods inherited from class java.lang. [Object](https://docs.oracle.com/en/java/javase/17/docs/api/java.base/java/lang/Object.html "class or interface in java.lang") [![Link icon](https://google.github.io/adk-docs/api-reference/java/resource-files/link.svg)](https://google.github.io/adk-docs/api-reference/java/com/google/adk/models/LlmRequest.Builder.html\#methods-inherited-from-class-java.lang.Object)

`clone, equals, finalize, getClass, hashCode, notify, notifyAll, toString, wait, wait, wait`


- ## Constructor Details [![Link icon](https://google.github.io/adk-docs/api-reference/java/resource-files/link.svg)](https://google.github.io/adk-docs/api-reference/java/com/google/adk/models/LlmRequest.Builder.html\#constructor-detail)



- ### Builder [![Link icon](https://google.github.io/adk-docs/api-reference/java/resource-files/link.svg)](https://google.github.io/adk-docs/api-reference/java/com/google/adk/models/LlmRequest.Builder.html\#%3Cinit%3E())





publicBuilder()


- ## Method Details [![Link icon](https://google.github.io/adk-docs/api-reference/java/resource-files/link.svg)](https://google.github.io/adk-docs/api-reference/java/com/google/adk/models/LlmRequest.Builder.html\#method-detail)



- ### model [![Link icon](https://google.github.io/adk-docs/api-reference/java/resource-files/link.svg)](https://google.github.io/adk-docs/api-reference/java/com/google/adk/models/LlmRequest.Builder.html\#model(java.lang.String))





@CanIgnoreReturnValue
public abstract[LlmRequest.Builder](https://google.github.io/adk-docs/api-reference/java/com/google/adk/models/LlmRequest.Builder.html "class in com.google.adk.models")model( [String](https://docs.oracle.com/en/java/javase/17/docs/api/java.base/java/lang/String.html "class or interface in java.lang") model)

- ### contents [![Link icon](https://google.github.io/adk-docs/api-reference/java/resource-files/link.svg)](https://google.github.io/adk-docs/api-reference/java/com/google/adk/models/LlmRequest.Builder.html\#contents(java.util.List))





@CanIgnoreReturnValue
public abstract[LlmRequest.Builder](https://google.github.io/adk-docs/api-reference/java/com/google/adk/models/LlmRequest.Builder.html "class in com.google.adk.models")contents( [List](https://docs.oracle.com/en/java/javase/17/docs/api/java.base/java/util/List.html "class or interface in java.util") <com.google.genai.types.Content> contents)

- ### config [![Link icon](https://google.github.io/adk-docs/api-reference/java/resource-files/link.svg)](https://google.github.io/adk-docs/api-reference/java/com/google/adk/models/LlmRequest.Builder.html\#config(com.google.genai.types.GenerateContentConfig))





@CanIgnoreReturnValue
public abstract[LlmRequest.Builder](https://google.github.io/adk-docs/api-reference/java/com/google/adk/models/LlmRequest.Builder.html "class in com.google.adk.models")config(com.google.genai.types.GenerateContentConfig config)

- ### config [![Link icon](https://google.github.io/adk-docs/api-reference/java/resource-files/link.svg)](https://google.github.io/adk-docs/api-reference/java/com/google/adk/models/LlmRequest.Builder.html\#config())





public abstract[Optional](https://docs.oracle.com/en/java/javase/17/docs/api/java.base/java/util/Optional.html "class or interface in java.util") <com.google.genai.types.GenerateContentConfig>config()

- ### liveConnectConfig [![Link icon](https://google.github.io/adk-docs/api-reference/java/resource-files/link.svg)](https://google.github.io/adk-docs/api-reference/java/com/google/adk/models/LlmRequest.Builder.html\#liveConnectConfig(com.google.genai.types.LiveConnectConfig))





@CanIgnoreReturnValue
public abstract[LlmRequest.Builder](https://google.github.io/adk-docs/api-reference/java/com/google/adk/models/LlmRequest.Builder.html "class in com.google.adk.models")liveConnectConfig(com.google.genai.types.LiveConnectConfig liveConnectConfig)

- ### appendInstructions [![Link icon](https://google.github.io/adk-docs/api-reference/java/resource-files/link.svg)](https://google.github.io/adk-docs/api-reference/java/com/google/adk/models/LlmRequest.Builder.html\#appendInstructions(java.util.List))





@CanIgnoreReturnValue
public final[LlmRequest.Builder](https://google.github.io/adk-docs/api-reference/java/com/google/adk/models/LlmRequest.Builder.html "class in com.google.adk.models")appendInstructions( [List](https://docs.oracle.com/en/java/javase/17/docs/api/java.base/java/util/List.html "class or interface in java.util") < [String](https://docs.oracle.com/en/java/javase/17/docs/api/java.base/java/lang/String.html "class or interface in java.lang") > instructions)

- ### appendTools [![Link icon](https://google.github.io/adk-docs/api-reference/java/resource-files/link.svg)](https://google.github.io/adk-docs/api-reference/java/com/google/adk/models/LlmRequest.Builder.html\#appendTools(java.util.List))





@CanIgnoreReturnValue
public final[LlmRequest.Builder](https://google.github.io/adk-docs/api-reference/java/com/google/adk/models/LlmRequest.Builder.html "class in com.google.adk.models")appendTools( [List](https://docs.oracle.com/en/java/javase/17/docs/api/java.base/java/util/List.html "class or interface in java.util") < [BaseTool](https://google.github.io/adk-docs/api-reference/java/com/google/adk/tools/BaseTool.html "class in com.google.adk.tools") > tools)

- ### outputSchema [![Link icon](https://google.github.io/adk-docs/api-reference/java/resource-files/link.svg)](https://google.github.io/adk-docs/api-reference/java/com/google/adk/models/LlmRequest.Builder.html\#outputSchema(com.google.genai.types.Schema))





@CanIgnoreReturnValue
public final[LlmRequest.Builder](https://google.github.io/adk-docs/api-reference/java/com/google/adk/models/LlmRequest.Builder.html "class in com.google.adk.models")outputSchema(com.google.genai.types.Schema schema)



Sets the output schema for the LLM response. If set, The output content will always be a JSON
string that conforms to the schema.

- ### build [![Link icon](https://google.github.io/adk-docs/api-reference/java/resource-files/link.svg)](https://google.github.io/adk-docs/api-reference/java/com/google/adk/models/LlmRequest.Builder.html\#build())





public abstract[LlmRequest](https://google.github.io/adk-docs/api-reference/java/com/google/adk/models/LlmRequest.html "class in com.google.adk.models")build()

## LlmRegistry.LlmFactory Usage
Packages that use [LlmRegistry.LlmFactory](https://google.github.io/adk-docs/api-reference/java/com/google/adk/models/LlmRegistry.LlmFactory.html "interface in com.google.adk.models")

Package

Description

[com.google.adk.models](https://google.github.io/adk-docs/api-reference/java/com/google/adk/models/class-use/LlmRegistry.LlmFactory.html#com.google.adk.models)

- ## Uses of [LlmRegistry.LlmFactory](https://google.github.io/adk-docs/api-reference/java/com/google/adk/models/LlmRegistry.LlmFactory.html "interface in com.google.adk.models") in [com.google.adk.models](https://google.github.io/adk-docs/api-reference/java/com/google/adk/models/package-summary.html) [![Link icon](https://google.github.io/adk-docs/api-reference/java/resource-files/link.svg)](https://google.github.io/adk-docs/api-reference/java/com/google/adk/models/class-use/LlmRegistry.LlmFactory.html\#com.google.adk.models)



Methods in [com.google.adk.models](https://google.github.io/adk-docs/api-reference/java/com/google/adk/models/package-summary.html) with parameters of type [LlmRegistry.LlmFactory](https://google.github.io/adk-docs/api-reference/java/com/google/adk/models/LlmRegistry.LlmFactory.html "interface in com.google.adk.models")





Modifier and Type



Method



Description



`static void`



LlmRegistry.`registerLlm(String modelNamePattern,
LlmRegistry.LlmFactory factory)`





Registers a factory for model names matching the given regex pattern.

## FunctionTool Overview
[java.lang.Object](https://docs.oracle.com/en/java/javase/17/docs/api/java.base/java/lang/Object.html "class or interface in java.lang")

[com.google.adk.tools.BaseTool](https://google.github.io/adk-docs/api-reference/java/com/google/adk/tools/BaseTool.html "class in com.google.adk.tools")

com.google.adk.tools.FunctionTool

Direct Known Subclasses:`LongRunningFunctionTool`

* * *

public class FunctionToolextends [BaseTool](https://google.github.io/adk-docs/api-reference/java/com/google/adk/tools/BaseTool.html "class in com.google.adk.tools")

FunctionTool implements a customized function calling tool.

- ## Constructor Summary [![Link icon](https://google.github.io/adk-docs/api-reference/java/resource-files/link.svg)](https://google.github.io/adk-docs/api-reference/java/com/google/adk/tools/FunctionTool.html\#constructor-summary)



Constructors





Modifier



Constructor



Description



`protected`



`FunctionTool(Object instance,
Method func,
boolean isLongRunning)`

- ## Method Summary [![Link icon](https://google.github.io/adk-docs/api-reference/java/resource-files/link.svg)](https://google.github.io/adk-docs/api-reference/java/com/google/adk/tools/FunctionTool.html\#method-summary)





All MethodsStatic MethodsInstance MethodsConcrete Methods







Modifier and Type



Method



Description



`static FunctionTool`



`create(Class<?> cls,
String methodName)`







`static FunctionTool`



`create(Object instance,
Method func)`







`static FunctionTool`



`create(Object instance,
String methodName)`







`static FunctionTool`



`create(Method func)`







`Optional<com.google.genai.types.FunctionDeclaration>`



`declaration()`





Gets the `FunctionDeclaration` representation of this tool.





`io.reactivex.rxjava3.core.Single<Map<String,Object>>`



`runAsync(Map<String,Object> args,
ToolContext toolContext)`





Calls a tool.













### Methods inherited from class com.google.adk.tools. [BaseTool](https://google.github.io/adk-docs/api-reference/java/com/google/adk/tools/BaseTool.html "class in com.google.adk.tools") [![Link icon](https://google.github.io/adk-docs/api-reference/java/resource-files/link.svg)](https://google.github.io/adk-docs/api-reference/java/com/google/adk/tools/FunctionTool.html\#methods-inherited-from-class-com.google.adk.tools.BaseTool)

`description, longRunning, name, processLlmRequest`





### Methods inherited from class java.lang. [Object](https://docs.oracle.com/en/java/javase/17/docs/api/java.base/java/lang/Object.html "class or interface in java.lang") [![Link icon](https://google.github.io/adk-docs/api-reference/java/resource-files/link.svg)](https://google.github.io/adk-docs/api-reference/java/com/google/adk/tools/FunctionTool.html\#methods-inherited-from-class-java.lang.Object)

`clone, equals, finalize, getClass, hashCode, notify, notifyAll, toString, wait, wait, wait`


- ## Constructor Details [![Link icon](https://google.github.io/adk-docs/api-reference/java/resource-files/link.svg)](https://google.github.io/adk-docs/api-reference/java/com/google/adk/tools/FunctionTool.html\#constructor-detail)



- ### FunctionTool [![Link icon](https://google.github.io/adk-docs/api-reference/java/resource-files/link.svg)](https://google.github.io/adk-docs/api-reference/java/com/google/adk/tools/FunctionTool.html\#%3Cinit%3E(java.lang.Object,java.lang.reflect.Method,boolean))





protectedFunctionTool(@Nullable
[Object](https://docs.oracle.com/en/java/javase/17/docs/api/java.base/java/lang/Object.html "class or interface in java.lang") instance,
[Method](https://docs.oracle.com/en/java/javase/17/docs/api/java.base/java/lang/reflect/Method.html "class or interface in java.lang.reflect") func,
boolean isLongRunning)


- ## Method Details [![Link icon](https://google.github.io/adk-docs/api-reference/java/resource-files/link.svg)](https://google.github.io/adk-docs/api-reference/java/com/google/adk/tools/FunctionTool.html\#method-detail)



- ### create [![Link icon](https://google.github.io/adk-docs/api-reference/java/resource-files/link.svg)](https://google.github.io/adk-docs/api-reference/java/com/google/adk/tools/FunctionTool.html\#create(java.lang.Object,java.lang.reflect.Method))





public static[FunctionTool](https://google.github.io/adk-docs/api-reference/java/com/google/adk/tools/FunctionTool.html "class in com.google.adk.tools")create( [Object](https://docs.oracle.com/en/java/javase/17/docs/api/java.base/java/lang/Object.html "class or interface in java.lang") instance,
[Method](https://docs.oracle.com/en/java/javase/17/docs/api/java.base/java/lang/reflect/Method.html "class or interface in java.lang.reflect") func)

- ### create [![Link icon](https://google.github.io/adk-docs/api-reference/java/resource-files/link.svg)](https://google.github.io/adk-docs/api-reference/java/com/google/adk/tools/FunctionTool.html\#create(java.lang.reflect.Method))





public static[FunctionTool](https://google.github.io/adk-docs/api-reference/java/com/google/adk/tools/FunctionTool.html "class in com.google.adk.tools")create( [Method](https://docs.oracle.com/en/java/javase/17/docs/api/java.base/java/lang/reflect/Method.html "class or interface in java.lang.reflect") func)

- ### create [![Link icon](https://google.github.io/adk-docs/api-reference/java/resource-files/link.svg)](https://google.github.io/adk-docs/api-reference/java/com/google/adk/tools/FunctionTool.html\#create(java.lang.Class,java.lang.String))





public static[FunctionTool](https://google.github.io/adk-docs/api-reference/java/com/google/adk/tools/FunctionTool.html "class in com.google.adk.tools")create( [Class](https://docs.oracle.com/en/java/javase/17/docs/api/java.base/java/lang/Class.html "class or interface in java.lang") <?> cls,
[String](https://docs.oracle.com/en/java/javase/17/docs/api/java.base/java/lang/String.html "class or interface in java.lang") methodName)

- ### create [![Link icon](https://google.github.io/adk-docs/api-reference/java/resource-files/link.svg)](https://google.github.io/adk-docs/api-reference/java/com/google/adk/tools/FunctionTool.html\#create(java.lang.Object,java.lang.String))





public static[FunctionTool](https://google.github.io/adk-docs/api-reference/java/com/google/adk/tools/FunctionTool.html "class in com.google.adk.tools")create( [Object](https://docs.oracle.com/en/java/javase/17/docs/api/java.base/java/lang/Object.html "class or interface in java.lang") instance,
[String](https://docs.oracle.com/en/java/javase/17/docs/api/java.base/java/lang/String.html "class or interface in java.lang") methodName)

- ### declaration [![Link icon](https://google.github.io/adk-docs/api-reference/java/resource-files/link.svg)](https://google.github.io/adk-docs/api-reference/java/com/google/adk/tools/FunctionTool.html\#declaration())





public[Optional](https://docs.oracle.com/en/java/javase/17/docs/api/java.base/java/util/Optional.html "class or interface in java.util") <com.google.genai.types.FunctionDeclaration>declaration()



Description copied from class: `BaseTool`



Gets the `FunctionDeclaration` representation of this tool.

Overrides:`declaration` in class `BaseTool`

- ### runAsync [![Link icon](https://google.github.io/adk-docs/api-reference/java/resource-files/link.svg)](https://google.github.io/adk-docs/api-reference/java/com/google/adk/tools/FunctionTool.html\#runAsync(java.util.Map,com.google.adk.tools.ToolContext))





publicio.reactivex.rxjava3.core.Single< [Map](https://docs.oracle.com/en/java/javase/17/docs/api/java.base/java/util/Map.html "class or interface in java.util") < [String](https://docs.oracle.com/en/java/javase/17/docs/api/java.base/java/lang/String.html "class or interface in java.lang"), [Object](https://docs.oracle.com/en/java/javase/17/docs/api/java.base/java/lang/Object.html "class or interface in java.lang") >>runAsync( [Map](https://docs.oracle.com/en/java/javase/17/docs/api/java.base/java/util/Map.html "class or interface in java.util") < [String](https://docs.oracle.com/en/java/javase/17/docs/api/java.base/java/lang/String.html "class or interface in java.lang"), [Object](https://docs.oracle.com/en/java/javase/17/docs/api/java.base/java/lang/Object.html "class or interface in java.lang") > args,
[ToolContext](https://google.github.io/adk-docs/api-reference/java/com/google/adk/tools/ToolContext.html "class in com.google.adk.tools") toolContext)



Description copied from class: `BaseTool`



Calls a tool.

Overrides:`runAsync` in class `BaseTool`

## LLM Agent Overview
[Skip to content](https://google.github.io/adk-docs/agents/llm-agents/#llm-agent)

[Edit this page](https://github.com/google/adk-docs/edit/main/docs/agents/llm-agents.md "Edit this page") [View source of this page](https://github.com/google/adk-docs/raw/main/docs/agents/llm-agents.md "View source of this page")

# LLM Agent [¶](https://google.github.io/adk-docs/agents/llm-agents/\#llm-agent "Permanent link")

Supported in ADKPython v0.1.0Go v0.1.0Java v0.1.0

The `LlmAgent` (often aliased simply as `Agent`) is a core component in ADK,
acting as the "thinking" part of your application. It leverages the power of a
Large Language Model (LLM) for reasoning, understanding natural language, making
decisions, generating responses, and interacting with tools.

Unlike deterministic [Workflow Agents](https://google.github.io/adk-docs/agents/workflow-agents/) that follow
predefined execution paths, `LlmAgent` behavior is non-deterministic. It uses
the LLM to interpret instructions and context, deciding dynamically how to
proceed, which tools to use (if any), or whether to transfer control to another
agent.

Building an effective `LlmAgent` involves defining its identity, clearly guiding
its behavior through instructions, and equipping it with the necessary tools and
capabilities.

## Defining the Agent's Identity and Purpose [¶](https://google.github.io/adk-docs/agents/llm-agents/\#defining-the-agents-identity-and-purpose "Permanent link")

First, you need to establish what the agent _is_ and what it's _for_.

- **`name` (Required):** Every agent needs a unique string identifier. This
`name` is crucial for internal operations, especially in multi-agent systems
where agents need to refer to or delegate tasks to each other. Choose a
descriptive name that reflects the agent's function (e.g.,
`customer_support_router`, `billing_inquiry_agent`). Avoid reserved names like
`user`.

- **`description` (Optional, Recommended for Multi-Agent):** Provide a concise
summary of the agent's capabilities. This description is primarily used by
_other_ LLM agents to determine if they should route a task to this agent.
Make it specific enough to differentiate it from peers (e.g., "Handles
inquiries about current billing statements," not just "Billing agent").

- **`model` (Required):** Specify the underlying LLM that will power this
agent's reasoning. This is a string identifier like `"gemini-2.0-flash"`. The
choice of model impacts the agent's capabilities, cost, and performance. See
the [Models](https://google.github.io/adk-docs/agents/models/) page for available options and considerations.


[Python](https://google.github.io/adk-docs/agents/llm-agents/#python)[Go](https://google.github.io/adk-docs/agents/llm-agents/#go)[Java](https://google.github.io/adk-docs/agents/llm-agents/#java)

```
# Example: Defining the basic identity
capital_agent = LlmAgent(
    model="gemini-2.0-flash",
    name="capital_agent",
    description="Answers user questions about the capital city of a given country."
    # instruction and tools will be added next
)
```

```
// Example: Defining the basic identity
agent, err := llmagent.New(llmagent.Config{
    Name:        "capital_agent",
    Model:       model,
    Description: "Answers user questions about the capital city of a given country.",
    // instruction and tools will be added next
})
```

```
// Example: Defining the basic identity
LlmAgent capitalAgent =
    LlmAgent.builder()
        .model("gemini-2.0-flash")
        .name("capital_agent")
        .description("Answers user questions about the capital city of a given country.")
        // instruction and tools will be added next
        .build();
```

## Guiding the Agent: Instructions (`instruction`) [¶](https://google.github.io/adk-docs/agents/llm-agents/\#guiding-the-agent-instructions-instruction "Permanent link")

The `instruction` parameter is arguably the most critical for shaping an
`LlmAgent`'s behavior. It's a string (or a function returning a string) that
tells the agent:

- Its core task or goal.
- Its personality or persona (e.g., "You are a helpful assistant," "You are a witty pirate").
- Constraints on its behavior (e.g., "Only answer questions about X," "Never reveal Y").
- How and when to use its `tools`. You should explain the purpose of each tool and the circumstances under which it should be called, supplementing any descriptions within the tool itself.
- The desired format for its output (e.g., "Respond in JSON," "Provide a bulleted list").

**Tips for Effective Instructions:**

- **Be Clear and Specific:** Avoid ambiguity. Clearly state the desired actions and outcomes.
- **Use Markdown:** Improve readability for complex instructions using headings, lists, etc.
- **Provide Examples (Few-Shot):** For complex tasks or specific output formats, include examples directly in the instruction.
- **Guide Tool Use:** Don't just list tools; explain _when_ and _why_ the agent should use them.

**State:**

- The instruction is a string template, you can use the `{var}` syntax to insert dynamic values into the instruction.
- `{var}` is used to insert the value of the state variable named var.
- `{artifact.var}` is used to insert the text content of the artifact named var.
- If the state variable or artifact does not exist, the agent will raise an error. If you want to ignore the error, you can append a `?` to the variable name as in `{var?}`.

[Python](https://google.github.io/adk-docs/agents/llm-agents/#python_1)[Go](https://google.github.io/adk-docs/agents/llm-agents/#go_1)[Java](https://google.github.io/adk-docs/agents/llm-agents/#java_1)

```
# Example: Adding instructions
capital_agent = LlmAgent(
    model="gemini-2.0-flash",
    name="capital_agent",
    description="Answers user questions about the capital city of a given country.",
    instruction="""You are an agent that provides the capital city of a country.
When a user asks for the capital of a country:
1. Identify the country name from the user's query.
2. Use the `get_capital_city` tool to find the capital.
3. Respond clearly to the user, stating the capital city.
Example Query: "What's the capital of {country}?"
Example Response: "The capital of France is Paris."
""",
    # tools will be added next
)
```

```
    // Example: Adding instructions
    agent, err := llmagent.New(llmagent.Config{
        Name:        "capital_agent",
        Model:       model,
        Description: "Answers user questions about the capital city of a given country.",
        Instruction: `You are an agent that provides the capital city of a country.
When a user asks for the capital of a country:
1. Identify the country name from the user's query.
2. Use the 'get_capital_city' tool to find the capital.
3. Respond clearly to the user, stating the capital city.
Example Query: "What's the capital of {country}?"
Example Response: "The capital of France is Paris."`,
        // tools will be added next
    })
```

```
// Example: Adding instructions
LlmAgent capitalAgent =
    LlmAgent.builder()
        .model("gemini-2.0-flash")
        .name("capital_agent")
        .description("Answers user questions about the capital city of a given country.")
        .instruction(
            """
            You are an agent that provides the capital city of a country.
            When a user asks for the capital of a country:
            1. Identify the country name from the user's query.
            2. Use the `get_capital_city` tool to find the capital.
            3. Respond clearly to the user, stating the capital city.
            Example Query: "What's the capital of {country}?"
            Example Response: "The capital of France is Paris."
            """)
        // tools will be added next
        .build();
```

_(Note: For instructions that apply to_ all _agents in a system, consider using_
_`global_instruction` on the root agent, detailed further in the_
_[Multi-Agents](https://google.github.io/adk-docs/agents/multi-agents/) section.)_

## Equipping the Agent: Tools (`tools`) [¶](https://google.github.io/adk-docs/agents/llm-agents/\#equipping-the-agent-tools-tools "Permanent link")

Tools give your `LlmAgent` capabilities beyond the LLM's built-in knowledge or
reasoning. They allow the agent to interact with the outside world, perform
calculations, fetch real-time data, or execute specific actions.

- **`tools` (Optional):** Provide a list of tools the agent can use. Each item in the list can be:
  - A native function or method (wrapped as a `FunctionTool`). Python ADK automatically wraps the native function into a `FuntionTool` whereas, you must explicitly wrap your Java methods using `FunctionTool.create(...)`
  - An instance of a class inheriting from `BaseTool`.
  - An instance of another agent (`AgentTool`, enabling agent-to-agent delegation - see [Multi-Agents](https://google.github.io/adk-docs/agents/multi-agents/)).

The LLM uses the function/tool names, descriptions (from docstrings or the
`description` field), and parameter schemas to decide which tool to call based
on the conversation and its instructions.

[Python](https://google.github.io/adk-docs/agents/llm-agents/#python_2)[Go](https://google.github.io/adk-docs/agents/llm-agents/#go_2)[Java](https://google.github.io/adk-docs/agents/llm-agents/#java_2)

```
# Define a tool function
def get_capital_city(country: str) -> str:
  """Retrieves the capital city for a given country."""
  # Replace with actual logic (e.g., API call, database lookup)
  capitals = {"france": "Paris", "japan": "Tokyo", "canada": "Ottawa"}
  return capitals.get(country.lower(), f"Sorry, I don't know the capital of {country}.")

# Add the tool to the agent
capital_agent = LlmAgent(
    model="gemini-2.0-flash",
    name="capital_agent",
    description="Answers user questions about the capital city of a given country.",
    instruction="""You are an agent that provides the capital city of a country... (previous instruction text)""",
    tools=[get_capital_city] # Provide the function directly
)
```

```
// Define a tool function
type getCapitalCityArgs struct {
    Country string `json:"country" jsonschema:"The country to get the capital of."`
}
getCapitalCity := func(ctx tool.Context, args getCapitalCityArgs) (map[string]any, error) {
    // Replace with actual logic (e.g., API call, database lookup)
    capitals := map[string]string{"france": "Paris", "japan": "Tokyo", "canada": "Ottawa"}
    capital, ok := capitals[strings.ToLower(args.Country)]
    if !ok {
        return nil, fmt.Errorf("Sorry, I don't know the capital of %s.", args.Country)
    }
    return map[string]any{"result": capital}, nil
}

// Add the tool to the agent
capitalTool, err := functiontool.New(
    functiontool.Config{
        Name:        "get_capital_city",
        Description: "Retrieves the capital city for a given country.",
    },
    getCapitalCity,
)
if err != nil {
    log.Fatal(err)
}
agent, err := llmagent.New(llmagent.Config{
    Name:        "capital_agent",
    Model:       model,
    Description: "Answers user questions about the capital city of a given country.",
    Instruction: "You are an agent that provides the capital city of a country... (previous instruction text)",
    Tools:       []tool.Tool{capitalTool},
})
```

```
// Define a tool function
// Retrieves the capital city of a given country.
public static Map<String, Object> getCapitalCity(
        @Schema(name = "country", description = "The country to get capital for")
        String country) {
  // Replace with actual logic (e.g., API call, database lookup)
  Map<String, String> countryCapitals = new HashMap<>();
  countryCapitals.put("canada", "Ottawa");
  countryCapitals.put("france", "Paris");
  countryCapitals.put("japan", "Tokyo");

  String result =
          countryCapitals.getOrDefault(
                  country.toLowerCase(), "Sorry, I couldn't find the capital for " + country + ".");
  return Map.of("result", result); // Tools must return a Map
}

// Add the tool to the agent
FunctionTool capitalTool = FunctionTool.create(experiment.getClass(), "getCapitalCity");
LlmAgent capitalAgent =
    LlmAgent.builder()
        .model("gemini-2.0-flash")
        .name("capital_agent")
        .description("Answers user questions about the capital city of a given country.")
        .instruction("You are an agent that provides the capital city of a country... (previous instruction text)")
        .tools(capitalTool) // Provide the function wrapped as a FunctionTool
        .build();
```

Learn more about Tools in the [Tools](https://google.github.io/adk-docs/tools/) section.

## Advanced Configuration & Control [¶](https://google.github.io/adk-docs/agents/llm-agents/\#advanced-configuration-control "Permanent link")

Beyond the core parameters, `LlmAgent` offers several options for finer control:

### Configuring LLM Generation (`generate_content_config`) [¶](https://google.github.io/adk-docs/agents/llm-agents/\#fine-tuning-llm-generation-generate_content_config "Permanent link")

You can adjust how the underlying LLM generates responses using `generate_content_config`.

- **`generate_content_config` (Optional):** Pass an instance of [`google.genai.types.GenerateContentConfig`](https://googleapis.github.io/python-genai/genai.html#genai.types.GenerateContentConfig) to control parameters like `temperature` (randomness), `max_output_tokens` (response length), `top_p`, `top_k`, and safety settings.

[Python](https://google.github.io/adk-docs/agents/llm-agents/#python_3)[Go](https://google.github.io/adk-docs/agents/llm-agents/#go_3)[Java](https://google.github.io/adk-docs/agents/llm-agents/#java_3)

```
from google.genai import types

agent = LlmAgent(
    # ... other params
    generate_content_config=types.GenerateContentConfig(
        temperature=0.2, # More deterministic output
        max_output_tokens=250,
        safety_settings=[\
            types.SafetySetting(\
                category=types.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,\
                threshold=types.HarmBlockThreshold.BLOCK_LOW_AND_ABOVE,\
            )\
        ]
    )
)
```

```
import "google.golang.org/genai"

temperature := float32(0.2)
agent, err := llmagent.New(llmagent.Config{
    Name:  "gen_config_agent",
    Model: model,
    GenerateContentConfig: &genai.GenerateContentConfig{
        Temperature:     &temperature,
        MaxOutputTokens: 250,
    },
})
```

```
import com.google.genai.types.GenerateContentConfig;

LlmAgent agent =
    LlmAgent.builder()
        // ... other params
        .generateContentConfig(GenerateContentConfig.builder()
            .temperature(0.2F) // More deterministic output
            .maxOutputTokens(250)
            .build())
        .build();
```

### Structuring Data (`input_schema`, `output_schema`, `output_key`) [¶](https://google.github.io/adk-docs/agents/llm-agents/\#structuring-data-input_schema-output_schema-output_key "Permanent link")

For scenarios requiring structured data exchange with an `LLM Agent`, the ADK provides mechanisms to define expected input and desired output formats using schema definitions.

- **`input_schema` (Optional):** Define a schema representing the expected input structure. If set, the user message content passed to this agent _must_ be a JSON string conforming to this schema. Your instructions should guide the user or preceding agent accordingly.

- **`output_schema` (Optional):** Define a schema representing the desired output structure. If set, the agent's final response _must_ be a JSON string conforming to this schema.

- **`output_key` (Optional):** Provide a string key. If set, the text content of the agent's _final_ response will be automatically saved to the session's state dictionary under this key. This is useful for passing results between agents or steps in a workflow.
  - In Python, this might look like: `session.state[output_key] = agent_response_text`
  - In Java: `session.state().put(outputKey, agentResponseText)`
  - In Golang, within a callback handler: `ctx.State().Set(output_key, agentResponseText)`

[Python](https://google.github.io/adk-docs/agents/llm-agents/#python_4)[Go](https://google.github.io/adk-docs/agents/llm-agents/#go_4)[Java](https://google.github.io/adk-docs/agents/llm-agents/#java_4)

The input and output schema is typically a `Pydantic` BaseModel.

```
from pydantic import BaseModel, Field

class CapitalOutput(BaseModel):
    capital: str = Field(description="The capital of the country.")

structured_capital_agent = LlmAgent(
    # ... name, model, description
    instruction="""You are a Capital Information Agent. Given a country, respond ONLY with a JSON object containing the capital. Format: {"capital": "capital_name"}""",
    output_schema=CapitalOutput, # Enforce JSON output
    output_key="found_capital"  # Store result in state['found_capital']
    # Cannot use tools=[get_capital_city] effectively here
)
```

The input and output schema is a `google.genai.types.Schema` object.

```
capitalOutput := &genai.Schema{
    Type:        genai.TypeObject,
    Description: "Schema for capital city information.",
    Properties: map[string]*genai.Schema{
        "capital": {
            Type:        genai.TypeString,
            Description: "The capital city of the country.",
        },
    },
}

agent, err := llmagent.New(llmagent.Config{
    Name:         "structured_capital_agent",
    Model:        model,
    Description:  "Provides capital information in a structured format.",
    Instruction:  `You are a Capital Information Agent. Given a country, respond ONLY with a JSON object containing the capital. Format: {"capital": "capital_name"}`,
    OutputSchema: capitalOutput,
    OutputKey:    "found_capital",
    // Cannot use the capitalTool tool effectively here
})
```

The input and output schema is a `google.genai.types.Schema` object.

```
private static final Schema CAPITAL_OUTPUT =
    Schema.builder()
        .type("OBJECT")
        .description("Schema for capital city information.")
        .properties(
            Map.of(
                "capital",
                Schema.builder()
                    .type("STRING")
                    .description("The capital city of the country.")
                    .build()))
        .build();

LlmAgent structuredCapitalAgent =
    LlmAgent.builder()
        // ... name, model, description
        .instruction(
                "You are a Capital Information Agent. Given a country, respond ONLY with a JSON object containing the capital. Format: {\"capital\": \"capital_name\"}")
        .outputSchema(capitalOutput) // Enforce JSON output
        .outputKey("found_capital") // Store result in state.get("found_capital")
        // Cannot use tools(getCapitalCity) effectively here
        .build();
```

### Managing Context (`include_contents`) [¶](https://google.github.io/adk-docs/agents/llm-agents/\#managing-context-include_contents "Permanent link")

Control whether the agent receives the prior conversation history.

- **`include_contents` (Optional, Default: `'default'`):** Determines if the `contents` (history) are sent to the LLM.
  - `'default'`: The agent receives the relevant conversation history.
  - `'none'`: The agent receives no prior `contents`. It operates based solely on its current instruction and any input provided in the _current_ turn (useful for stateless tasks or enforcing specific contexts).

[Python](https://google.github.io/adk-docs/agents/llm-agents/#python_5)[Go](https://google.github.io/adk-docs/agents/llm-agents/#go_5)[Java](https://google.github.io/adk-docs/agents/llm-agents/#java_5)

```
stateless_agent = LlmAgent(
    # ... other params
    include_contents='none'
)
```

```
import "google.golang.org/adk/agent/llmagent"

agent, err := llmagent.New(llmagent.Config{
    Name:            "stateless_agent",
    Model:           model,
    IncludeContents: llmagent.IncludeContentsNone,
})
```

```
import com.google.adk.agents.LlmAgent.IncludeContents;

LlmAgent statelessAgent =
    LlmAgent.builder()
        // ... other params
        .includeContents(IncludeContents.NONE)
        .build();
```

### Planner [¶](https://google.github.io/adk-docs/agents/llm-agents/\#planner "Permanent link")

Supported in ADKPython v0.1.0

**`planner` (Optional):** Assign a `BasePlanner` instance to enable multi-step reasoning and planning before execution. There are two main planners:

- **`BuiltInPlanner`:** Leverages the model's built-in planning capabilities (e.g., Gemini's thinking feature). See [Gemini Thinking](https://ai.google.dev/gemini-api/docs/thinking) for details and examples.

Here, the `thinking_budget` parameter guides the model on the number of thinking tokens to use when generating a response. The `include_thoughts` parameter controls whether the model should include its raw thoughts and internal reasoning process in the response.



```
from google.adk import Agent
from google.adk.planners import BuiltInPlanner
from google.genai import types

my_agent = Agent(
      model="gemini-2.5-flash",
      planner=BuiltInPlanner(
          thinking_config=types.ThinkingConfig(
              include_thoughts=True,
              thinking_budget=1024,
          )
      ),
      # ... your tools here
)
```

- **`PlanReActPlanner`:** This planner instructs the model to follow a specific structure in its output: first create a plan, then execute actions (like calling tools), and provide reasoning for its steps. _It's particularly useful for models that don't have a built-in "thinking" feature_.



```
from google.adk import Agent
from google.adk.planners import PlanReActPlanner

my_agent = Agent(
      model="gemini-2.0-flash",
      planner=PlanReActPlanner(),
      # ... your tools here
)
```



The agent's response will follow a structured format:



```
[user]: ai news
[google_search_agent]: /*PLANNING*/
1. Perform a Google search for "latest AI news" to get current updates and headlines related to artificial intelligence.
2. Synthesize the information from the search results to provide a summary of recent AI news.

/*ACTION*/
/*REASONING*/
The search results provide a comprehensive overview of recent AI news, covering various aspects like company developments, research breakthroughs, and applications. I have enough information to answer the user's request.

/*FINAL_ANSWER*/
Here's a summary of recent AI news:
....
```

### Code Execution [¶](https://google.github.io/adk-docs/agents/llm-agents/\#code-execution "Permanent link")

Supported in ADKPython v0.1.0

- **`code_executor` (Optional):** Provide a `BaseCodeExecutor` instance to allow the agent to execute code blocks found in the LLM's response. ( [See Tools/Built-in tools](https://google.github.io/adk-docs/tools/built-in-tools/)).

Example for using built-in-planner:

```
from dotenv import load_dotenv

import asyncio
import os

from google.genai import types
from google.adk.agents.llm_agent import LlmAgent
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.adk.artifacts.in_memory_artifact_service import InMemoryArtifactService # Optional
from google.adk.planners import BasePlanner, BuiltInPlanner, PlanReActPlanner
from google.adk.models import LlmRequest

from google.genai.types import ThinkingConfig
from google.genai.types import GenerateContentConfig

import datetime
from zoneinfo import ZoneInfo

APP_NAME = "weather_app"
USER_ID = "1234"
SESSION_ID = "session1234"

def get_weather(city: str) -> dict:
    """Retrieves the current weather report for a specified city.

    Args:
        city (str): The name of the city for which to retrieve the weather report.

    Returns:
        dict: status and result or error msg.
    """
    if city.lower() == "new york":
        return {
            "status": "success",
            "report": (
                "The weather in New York is sunny with a temperature of 25 degrees"
                " Celsius (77 degrees Fahrenheit)."
            ),
        }
    else:
        return {
            "status": "error",
            "error_message": f"Weather information for '{city}' is not available.",
        }

def get_current_time(city: str) -> dict:
    """Returns the current time in a specified city.

    Args:
        city (str): The name of the city for which to retrieve the current time.

    Returns:
        dict: status and result or error msg.
    """

    if city.lower() == "new york":
        tz_identifier = "America/New_York"
    else:
        return {
            "status": "error",
            "error_message": (
                f"Sorry, I don't have timezone information for {city}."
            ),
        }

    tz = ZoneInfo(tz_identifier)
    now = datetime.datetime.now(tz)
    report = (
        f'The current time in {city} is {now.strftime("%Y-%m-%d %H:%M:%S %Z%z")}'
    )
    return {"status": "success", "report": report}

# Step 1: Create a ThinkingConfig
thinking_config = ThinkingConfig(
    include_thoughts=True,   # Ask the model to include its thoughts in the response
    thinking_budget=256      # Limit the 'thinking' to 256 tokens (adjust as needed)
)
print("ThinkingConfig:", thinking_config)

# Step 2: Instantiate BuiltInPlanner
planner = BuiltInPlanner(
    thinking_config=thinking_config
)
print("BuiltInPlanner created.")

# Step 3: Wrap the planner in an LlmAgent
agent = LlmAgent(
    model="gemini-2.5-pro-preview-03-25",  # Set your model name
    name="weather_and_time_agent",
    instruction="You are an agent that returns time and weather",
    planner=planner,
    tools=[get_weather, get_current_time]
)

# Session and Runner
session_service = InMemorySessionService()
session = session_service.create_session(app_name=APP_NAME, user_id=USER_ID, session_id=SESSION_ID)
runner = Runner(agent=agent, app_name=APP_NAME, session_service=session_service)

# Agent Interaction
def call_agent(query):
    content = types.Content(role='user', parts=[types.Part(text=query)])
    events = runner.run(user_id=USER_ID, session_id=SESSION_ID, new_message=content)

    for event in events:
        print(f"\nDEBUG EVENT: {event}\n")
        if event.is_final_response() and event.content:
            final_answer = event.content.parts[0].text.strip()
            print("\n🟢 FINAL ANSWER\n", final_answer, "\n")

call_agent("If it's raining in New York right now, what is the current temperature?")
```

## Putting It Together: Example [¶](https://google.github.io/adk-docs/agents/llm-agents/\#putting-it-together-example "Permanent link")

Code

Here's the complete basic `capital_agent`:

[Python](https://google.github.io/adk-docs/agents/llm-agents/#python_6)[Go](https://google.github.io/adk-docs/agents/llm-agents/#go_6)[Java](https://google.github.io/adk-docs/agents/llm-agents/#java_6)

```
# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# --- Full example code demonstrating LlmAgent with Tools vs. Output Schema ---
import json # Needed for pretty printing dicts
import asyncio

from google.adk.agents import LlmAgent
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types
from pydantic import BaseModel, Field

# --- 1. Define Constants ---
APP_NAME = "agent_comparison_app"
USER_ID = "test_user_456"
SESSION_ID_TOOL_AGENT = "session_tool_agent_xyz"
SESSION_ID_SCHEMA_AGENT = "session_schema_agent_xyz"
MODEL_NAME = "gemini-2.0-flash"

# --- 2. Define Schemas ---

# Input schema used by both agents
class CountryInput(BaseModel):
    country: str = Field(description="The country to get information about.")

# Output schema ONLY for the second agent
class CapitalInfoOutput(BaseModel):
    capital: str = Field(description="The capital city of the country.")
    # Note: Population is illustrative; the LLM will infer or estimate this
    # as it cannot use tools when output_schema is set.
    population_estimate: str = Field(description="An estimated population of the capital city.")

# --- 3. Define the Tool (Only for the first agent) ---
def get_capital_city(country: str) -> str:
    """Retrieves the capital city of a given country."""
    print(f"\n-- Tool Call: get_capital_city(country='{country}') --")
    country_capitals = {
        "united states": "Washington, D.C.",
        "canada": "Ottawa",
        "france": "Paris",
        "japan": "Tokyo",
    }
    result = country_capitals.get(country.lower(), f"Sorry, I couldn't find the capital for {country}.")
    print(f"-- Tool Result: '{result}' --")
    return result

# --- 4. Configure Agents ---

# Agent 1: Uses a tool and output_key
capital_agent_with_tool = LlmAgent(
    model=MODEL_NAME,
    name="capital_agent_tool",
    description="Retrieves the capital city using a specific tool.",
    instruction="""You are a helpful agent that provides the capital city of a country using a tool.
The user will provide the country name in a JSON format like {"country": "country_name"}.
1. Extract the country name.
2. Use the `get_capital_city` tool to find the capital.
3. Respond clearly to the user, stating the capital city found by the tool.
""",
    tools=[get_capital_city],
    input_schema=CountryInput,
    output_key="capital_tool_result", # Store final text response
)

# Agent 2: Uses output_schema (NO tools possible)
structured_info_agent_schema = LlmAgent(
    model=MODEL_NAME,
    name="structured_info_agent_schema",
    description="Provides capital and estimated population in a specific JSON format.",
    instruction=f"""You are an agent that provides country information.
The user will provide the country name in a JSON format like {{"country": "country_name"}}.
Respond ONLY with a JSON object matching this exact schema:
{json.dumps(CapitalInfoOutput.model_json_schema(), indent=2)}
Use your knowledge to determine the capital and estimate the population. Do not use any tools.
""",
    # *** NO tools parameter here - using output_schema prevents tool use ***
    input_schema=CountryInput,
    output_schema=CapitalInfoOutput, # Enforce JSON output structure
    output_key="structured_info_result", # Store final JSON response
)

# --- 5. Set up Session Management and Runners ---
session_service = InMemorySessionService()

# Create a runner for EACH agent
capital_runner = Runner(
    agent=capital_agent_with_tool,
    app_name=APP_NAME,
    session_service=session_service
)
structured_runner = Runner(
    agent=structured_info_agent_schema,
    app_name=APP_NAME,
    session_service=session_service
)

# --- 6. Define Agent Interaction Logic ---
async def call_agent_and_print(
    runner_instance: Runner,
    agent_instance: LlmAgent,
    session_id: str,
    query_json: str
):
    """Sends a query to the specified agent/runner and prints results."""
    print(f"\n>>> Calling Agent: '{agent_instance.name}' | Query: {query_json}")

    user_content = types.Content(role='user', parts=[types.Part(text=query_json)])

    final_response_content = "No final response received."
    async for event in runner_instance.run_async(user_id=USER_ID, session_id=session_id, new_message=user_content):
        # print(f"Event: {event.type}, Author: {event.author}") # Uncomment for detailed logging
        if event.is_final_response() and event.content and event.content.parts:
            # For output_schema, the content is the JSON string itself
            final_response_content = event.content.parts[0].text

    print(f"<<< Agent '{agent_instance.name}' Response: {final_response_content}")

    current_session = await session_service.get_session(app_name=APP_NAME,
                                                  user_id=USER_ID,
                                                  session_id=session_id)
    stored_output = current_session.state.get(agent_instance.output_key)

    # Pretty print if the stored output looks like JSON (likely from output_schema)
    print(f"--- Session State ['{agent_instance.output_key}']: ", end="")
    try:
        # Attempt to parse and pretty print if it's JSON
        parsed_output = json.loads(stored_output)
        print(json.dumps(parsed_output, indent=2))
    except (json.JSONDecodeError, TypeError):
         # Otherwise, print as string
        print(stored_output)
    print("-" * 30)

# --- 7. Run Interactions ---
async def main():
    # Create separate sessions for clarity, though not strictly necessary if context is managed
    print("--- Creating Sessions ---")
    await session_service.create_session(app_name=APP_NAME, user_id=USER_ID, session_id=SESSION_ID_TOOL_AGENT)
    await session_service.create_session(app_name=APP_NAME, user_id=USER_ID, session_id=SESSION_ID_SCHEMA_AGENT)

    print("--- Testing Agent with Tool ---")
    await call_agent_and_print(capital_runner, capital_agent_with_tool, SESSION_ID_TOOL_AGENT, '{"country": "France"}')
    await call_agent_and_print(capital_runner, capital_agent_with_tool, SESSION_ID_TOOL_AGENT, '{"country": "Canada"}')

    print("\n\n--- Testing Agent with Output Schema (No Tool Use) ---")
    await call_agent_and_print(structured_runner, structured_info_agent_schema, SESSION_ID_SCHEMA_AGENT, '{"country": "France"}')
    await call_agent_and_print(structured_runner, structured_info_agent_schema, SESSION_ID_SCHEMA_AGENT, '{"country": "Japan"}')

# --- Run the Agent ---
# Note: In Colab, you can directly use 'await' at the top level.
# If running this code as a standalone Python script, you'll need to use asyncio.run() or manage the event loop.
if __name__ == "__main__":
    asyncio.run(main())
```

```
package main

import (
    "context"
    "encoding/json"
    "errors"
    "fmt"
    "log"
    "strings"

    "google.golang.org/adk/agent"
    "google.golang.org/adk/agent/llmagent"
    "google.golang.org/adk/model/gemini"
    "google.golang.org/adk/runner"
    "google.golang.org/adk/session"
    "google.golang.org/adk/tool"
    "google.golang.org/adk/tool/functiontool"

    "google.golang.org/genai"
)

// --- Main Runnable Example ---

const (
    modelName = "gemini-2.0-flash"
    appName   = "agent_comparison_app"
    userID    = "test_user_456"
)

type getCapitalCityArgs struct {
    Country string `json:"country" jsonschema:"The country to get the capital of."`
}

// getCapitalCity retrieves the capital city of a given country.
func getCapitalCity(ctx tool.Context, args getCapitalCityArgs) (map[string]any, error) {
    fmt.Printf("\n-- Tool Call: getCapitalCity(country='%s') --\n", args.Country)
    capitals := map[string]string{
        "united states": "Washington, D.C.",
        "canada":        "Ottawa",
        "france":        "Paris",
        "japan":         "Tokyo",
    }
    capital, ok := capitals[strings.ToLower(args.Country)]
    if !ok {
        result := fmt.Sprintf("Sorry, I couldn't find the capital for %s.", args.Country)
        fmt.Printf("-- Tool Result: '%s' --\n", result)
        return nil, errors.New(result)
    }
    fmt.Printf("-- Tool Result: '%s' --\n", capital)
    return map[string]any{"result": capital}, nil
}

// callAgent is a helper function to execute an agent with a given prompt and handle its output.
func callAgent(ctx context.Context, a agent.Agent, outputKey string, prompt string) {
    fmt.Printf("\n>>> Calling Agent: '%s' | Query: %s\n", a.Name(), prompt)
    // Create an in-memory session service to manage agent state.
    sessionService := session.InMemoryService()

    // Create a new session for the agent interaction.
    sessionCreateResponse, err := sessionService.Create(ctx, &session.CreateRequest{
        AppName: appName,
        UserID:  userID,
    })
    if err != nil {
        log.Fatalf("Failed to create the session service: %v", err)
    }

    session := sessionCreateResponse.Session

    // Configure the runner with the application name, agent, and session service.
    config := runner.Config{
        AppName:        appName,
        Agent:          a,
        SessionService: sessionService,
    }

    // Create a new runner instance.
    r, err := runner.New(config)
    if err != nil {
        log.Fatalf("Failed to create the runner: %v", err)
    }

    // Prepare the user's message to send to the agent.
    sessionID := session.ID()
    userMsg := &genai.Content{
        Parts: []*genai.Part{
            genai.NewPartFromText(prompt),
        },
        Role: string(genai.RoleUser),
    }

    // Run the agent and process the streaming events.
    for event, err := range r.Run(ctx, userID, sessionID, userMsg, agent.RunConfig{
        StreamingMode: agent.StreamingModeSSE,
    }) {
        if err != nil {
            fmt.Printf("\nAGENT_ERROR: %v\n", err)
        } else if event.Partial {
            // Print partial responses as they are received.
            for _, p := range event.Content.Parts {
                fmt.Print(p.Text)
            }
        }
    }

    // After the run, check if there's an expected output key in the session state.
    if outputKey != "" {
        storedOutput, error := session.State().Get(outputKey)
        if error == nil {
            // Pretty-print the stored output if it's a JSON string.
            fmt.Printf("\n--- Session State ['%s']: ", outputKey)
            storedString, isString := storedOutput.(string)
            if isString {
                var prettyJSON map[string]interface{}
                if err := json.Unmarshal([]byte(storedString), &prettyJSON); err == nil {
                    indentedJSON, err := json.MarshalIndent(prettyJSON, "", "  ")
                    if err == nil {
                        fmt.Println(string(indentedJSON))
                    } else {
                        fmt.Println(storedString)
                    }
                } else {
                    fmt.Println(storedString)
                }
            } else {
                fmt.Println(storedOutput)
            }
            fmt.Println(strings.Repeat("-", 30))
        }
    }
}

func main() {
    ctx := context.Background()

    model, err := gemini.NewModel(ctx, modelName, &genai.ClientConfig{})
    if err != nil {
        log.Fatalf("Failed to create model: %v", err)
    }

    capitalTool, err := functiontool.New(
        functiontool.Config{
            Name:        "get_capital_city",
            Description: "Retrieves the capital city for a given country.",
        },
        getCapitalCity,
    )
    if err != nil {
        log.Fatalf("Failed to create function tool: %v", err)
    }

    countryInputSchema := &genai.Schema{
        Type:        genai.TypeObject,
        Description: "Input for specifying a country.",
        Properties: map[string]*genai.Schema{
            "country": {
                Type:        genai.TypeString,
                Description: "The country to get information about.",
            },
        },
        Required: []string{"country"},
    }

    capitalAgentWithTool, err := llmagent.New(llmagent.Config{
        Name:        "capital_agent_tool",
        Model:       model,
        Description: "Retrieves the capital city using a specific tool.",
        Instruction: `You are a helpful agent that provides the capital city of a country using a tool.
The user will provide the country name in a JSON format like {"country": "country_name"}.
1. Extract the country name.
2. Use the 'get_capital_city' tool to find the capital.
3. Respond clearly to the user, stating the capital city found by the tool.`,
        Tools:       []tool.Tool{capitalTool},
        InputSchema: countryInputSchema,
        OutputKey:   "capital_tool_result",
    })
    if err != nil {
        log.Fatalf("Failed to create capital agent with tool: %v", err)
    }

    capitalInfoOutputSchema := &genai.Schema{
        Type:        genai.TypeObject,
        Description: "Schema for capital city information.",
        Properties: map[string]*genai.Schema{
            "capital": {
                Type:        genai.TypeString,
                Description: "The capital city of the country.",
            },
            "population_estimate": {
                Type:        genai.TypeString,
                Description: "An estimated population of the capital city.",
            },
        },
        Required: []string{"capital", "population_estimate"},
    }
    schemaJSON, _ := json.Marshal(capitalInfoOutputSchema)
    structuredInfoAgentSchema, err := llmagent.New(llmagent.Config{
        Name:        "structured_info_agent_schema",
        Model:       model,
        Description: "Provides capital and estimated population in a specific JSON format.",
        Instruction: fmt.Sprintf(`You are an agent that provides country information.
The user will provide the country name in a JSON format like {"country": "country_name"}.
Respond ONLY with a JSON object matching this exact schema:
%s
Use your knowledge to determine the capital and estimate the population. Do not use any tools.`, string(schemaJSON)),
        InputSchema:  countryInputSchema,
        OutputSchema: capitalInfoOutputSchema,
        OutputKey:    "structured_info_result",
    })
    if err != nil {
        log.Fatalf("Failed to create structured info agent: %v", err)
    }

    fmt.Println("--- Testing Agent with Tool ---")
    callAgent(ctx, capitalAgentWithTool, "capital_tool_result", `{"country": "France"}`)
    callAgent(ctx, capitalAgentWithTool, "capital_tool_result", `{"country": "Canada"}`)

    fmt.Println("\n\n--- Testing Agent with Output Schema (No Tool Use) ---")
    callAgent(ctx, structuredInfoAgentSchema, "structured_info_result", `{"country": "France"}`)
    callAgent(ctx, structuredInfoAgentSchema, "structured_info_result", `{"country": "Japan"}`)
}
```

```
// --- Full example code demonstrating LlmAgent with Tools vs. Output Schema ---

import com.google.adk.agents.LlmAgent;
import com.google.adk.events.Event;
import com.google.adk.runner.Runner;
import com.google.adk.sessions.InMemorySessionService;
import com.google.adk.sessions.Session;
import com.google.adk.tools.Annotations;
import com.google.adk.tools.FunctionTool;
import com.google.genai.types.Content;
import com.google.genai.types.Part;
import com.google.genai.types.Schema;
import io.reactivex.rxjava3.core.Flowable;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Optional;

public class LlmAgentExample {

// --- 1. Define Constants ---
private static final String MODEL_NAME = "gemini-2.0-flash";
private static final String APP_NAME = "capital_agent_tool";
private static final String USER_ID = "test_user_456";
private static final String SESSION_ID_TOOL_AGENT = "session_tool_agent_xyz";
private static final String SESSION_ID_SCHEMA_AGENT = "session_schema_agent_xyz";

// --- 2. Define Schemas ---

// Input schema used by both agents
private static final Schema COUNTRY_INPUT_SCHEMA =
      Schema.builder()
          .type("OBJECT")
          .description("Input for specifying a country.")
          .properties(
              Map.of(
                  "country",
                  Schema.builder()
                      .type("STRING")
                      .description("The country to get information about.")
                      .build()))
          .required(List.of("country"))
          .build();

// Output schema ONLY for the second agent
private static final Schema CAPITAL_INFO_OUTPUT_SCHEMA =
      Schema.builder()
          .type("OBJECT")
          .description("Schema for capital city information.")
          .properties(
              Map.of(
                  "capital",
                  Schema.builder()
                      .type("STRING")
                      .description("The capital city of the country.")
                      .build(),
                  "population_estimate",
                  Schema.builder()
                      .type("STRING")
                      .description("An estimated population of the capital city.")
                      .build()))
          .required(List.of("capital", "population_estimate"))
          .build();

// --- 3. Define the Tool (Only for the first agent) ---
// Retrieves the capital city of a given country.
public static Map<String, Object> getCapitalCity(
      @Annotations.Schema(name = "country", description = "The country to get capital for")
      String country) {
    System.out.printf("%n-- Tool Call: getCapitalCity(country='%s') --%n", country);
    Map<String, String> countryCapitals = new HashMap<>();
    countryCapitals.put("united states", "Washington, D.C.");
    countryCapitals.put("canada", "Ottawa");
    countryCapitals.put("france", "Paris");
    countryCapitals.put("japan", "Tokyo");

    String result =
        countryCapitals.getOrDefault(
            country.toLowerCase(), "Sorry, I couldn't find the capital for " + country + ".");
    System.out.printf("-- Tool Result: '%s' --%n", result);
    return Map.of("result", result); // Tools must return a Map
}

public static void main(String[] args){
    LlmAgentExample agentExample = new LlmAgentExample();
    FunctionTool capitalTool = FunctionTool.create(agentExample.getClass(), "getCapitalCity");

    // --- 4. Configure Agents ---

    // Agent 1: Uses a tool and output_key
    LlmAgent capitalAgentWithTool =
        LlmAgent.builder()
            .model(MODEL_NAME)
            .name("capital_agent_tool")
            .description("Retrieves the capital city using a specific tool.")
            .instruction(
              """
              You are a helpful agent that provides the capital city of a country using a tool.
              1. Extract the country name.
              2. Use the `get_capital_city` tool to find the capital.
              3. Respond clearly to the user, stating the capital city found by the tool.
              """)
            .tools(capitalTool)
            .inputSchema(COUNTRY_INPUT_SCHEMA)
            .outputKey("capital_tool_result") // Store final text response
            .build();

    // Agent 2: Uses an output schema
    LlmAgent structuredInfoAgentSchema =
        LlmAgent.builder()
            .model(MODEL_NAME)
            .name("structured_info_agent_schema")
            .description("Provides capital and estimated population in a specific JSON format.")
            .instruction(
                String.format("""
                You are an agent that provides country information.
                Respond ONLY with a JSON object matching this exact schema: %s
                Use your knowledge to determine the capital and estimate the population. Do not use any tools.
                """, CAPITAL_INFO_OUTPUT_SCHEMA.toJson()))
            // *** NO tools parameter here - using output_schema prevents tool use ***
            .inputSchema(COUNTRY_INPUT_SCHEMA)
            .outputSchema(CAPITAL_INFO_OUTPUT_SCHEMA) // Enforce JSON output structure
            .outputKey("structured_info_result") // Store final JSON response
            .build();

    // --- 5. Set up Session Management and Runners ---
    InMemorySessionService sessionService = new InMemorySessionService();

    sessionService.createSession(APP_NAME, USER_ID, null, SESSION_ID_TOOL_AGENT).blockingGet();
    sessionService.createSession(APP_NAME, USER_ID, null, SESSION_ID_SCHEMA_AGENT).blockingGet();

    Runner capitalRunner = new Runner(capitalAgentWithTool, APP_NAME, null, sessionService);
    Runner structuredRunner = new Runner(structuredInfoAgentSchema, APP_NAME, null, sessionService);

    // --- 6. Run Interactions ---
    System.out.println("--- Testing Agent with Tool ---");
    agentExample.callAgentAndPrint(
        capitalRunner, capitalAgentWithTool, SESSION_ID_TOOL_AGENT, "{\"country\": \"France\"}");
    agentExample.callAgentAndPrint(
        capitalRunner, capitalAgentWithTool, SESSION_ID_TOOL_AGENT, "{\"country\": \"Canada\"}");

    System.out.println("\n\n--- Testing Agent with Output Schema (No Tool Use) ---");
    agentExample.callAgentAndPrint(
        structuredRunner,
        structuredInfoAgentSchema,
        SESSION_ID_SCHEMA_AGENT,
        "{\"country\": \"France\"}");
    agentExample.callAgentAndPrint(
        structuredRunner,
        structuredInfoAgentSchema,
        SESSION_ID_SCHEMA_AGENT,
        "{\"country\": \"Japan\"}");
}

// --- 7. Define Agent Interaction Logic ---
public void callAgentAndPrint(Runner runner, LlmAgent agent, String sessionId, String queryJson) {
    System.out.printf(
        "%n>>> Calling Agent: '%s' | Session: '%s' | Query: %s%n",
        agent.name(), sessionId, queryJson);

    Content userContent = Content.fromParts(Part.fromText(queryJson));
    final String[] finalResponseContent = {"No final response received."};
    Flowable<Event> eventStream = runner.runAsync(USER_ID, sessionId, userContent);

    // Stream event response
    eventStream.blockingForEach(event -> {
          if (event.finalResponse() && event.content().isPresent()) {
            event
                .content()
                .get()
                .parts()
                .flatMap(parts -> parts.isEmpty() ? Optional.empty() : Optional.of(parts.get(0)))
                .flatMap(Part::text)
                .ifPresent(text -> finalResponseContent[0] = text);
          }
        });

    System.out.printf("<<< Agent '%s' Response: %s%n", agent.name(), finalResponseContent[0]);

    // Retrieve the session again to get the updated state
    Session updatedSession =
        runner
            .sessionService()
            .getSession(APP_NAME, USER_ID, sessionId, Optional.empty())
            .blockingGet();

    if (updatedSession != null && agent.outputKey().isPresent()) {
      // Print to verify if the stored output looks like JSON (likely from output_schema)
      System.out.printf("--- Session State ['%s']: ", agent.outputKey().get());
      }
}
}
```

_(This example demonstrates the core concepts. More complex agents might incorporate schemas, context control, planning, etc.)_

## Related Concepts (Deferred Topics) [¶](https://google.github.io/adk-docs/agents/llm-agents/\#related-concepts-deferred-topics "Permanent link")

While this page covers the core configuration of `LlmAgent`, several related concepts provide more advanced control and are detailed elsewhere:

- **Callbacks:** Intercepting execution points (before/after model calls, before/after tool calls) using `before_model_callback`, `after_model_callback`, etc. See [Callbacks](https://google.github.io/adk-docs/callbacks/types-of-callbacks/).
- **Multi-Agent Control:** Advanced strategies for agent interaction, including planning (`planner`), controlling agent transfer (`disallow_transfer_to_parent`, `disallow_transfer_to_peers`), and system-wide instructions (`global_instruction`). See [Multi-Agents](https://google.github.io/adk-docs/agents/multi-agents/).

Back to top

## ADK App Class Guide
[Skip to content](https://google.github.io/adk-docs/apps/#apps-workflow-management-class)

[Edit this page](https://github.com/google/adk-docs/edit/main/docs/apps/index.md "Edit this page") [View source of this page](https://github.com/google/adk-docs/raw/main/docs/apps/index.md "View source of this page")

# Apps: workflow management class [¶](https://google.github.io/adk-docs/apps/\#apps-workflow-management-class "Permanent link")

Supported in ADKPython v1.14.0

The **_App_** class is a top-level container for an entire Agent Development Kit
(ADK) agent workflow. It is designed to manage the lifecycle, configuration, and
state for a collection of agents grouped by a **_root agent_**. The **App** class
separates the concerns of an agent workflow's overall operational infrastructure
from individual agents' task-oriented reasoning.

Defining an **_App_** object in your ADK workflow is optional and changes how you
organize your agent code and run your agents. From a practical perspective, you
use the **_App_** class to configure the following features for your agent workflow:

- [**Context caching**](https://google.github.io/adk-docs/context/caching/)
- [**Context compression**](https://google.github.io/adk-docs/context/compaction/)
- [**Agent resume**](https://google.github.io/adk-docs/runtime/resume/)
- [**Plugins**](https://google.github.io/adk-docs/plugins/)

This guide explains how to use the App class for configuring and managing your
ADK agent workflows.

## Purpose of App Class [¶](https://google.github.io/adk-docs/apps/\#purpose-of-app-class "Permanent link")

The **_App_** class addresses several architectural issues that arise when
building complex agentic systems:

- **Centralized configuration:** Provides a single, centralized location for
managing shared resources like API keys and database clients, avoiding the
need to pass configuration down through every agent.
- **Lifecycle management:** The **_App_** class includes **_on startup_** and
**_on shutdown_** hooks, which allow for reliable management of persistent
resources such as database connection pools or in-memory caches that need to
exist across multiple invocations.
- **State scope:** It defines an explicit boundary for application-level
state with an `app:*` prefix making the scope and lifetime of this state
clear to developers.
- **Unit of deployment:** The **_App_** concept establishes a formal _deployable_
_unit_, simplifying versioning, testing, and serving of agentic applications.

## Define an App object [¶](https://google.github.io/adk-docs/apps/\#define-an-app-object "Permanent link")

The **_App_** class is used as the primary container of your agent workflow and
contains the root agent of the project. The **_root agent_** is the container
for the primary controller agent and any additonal sub-agents.

### Define app with root agent [¶](https://google.github.io/adk-docs/apps/\#define-app-with-root-agent "Permanent link")

Create a **_root agent_** for your workflow by creating a subclass from the
**_Agent_** base class. Then define an **_App_** object and configure it with
the **_root agent_** object and optional features, as shown in the following
sample code:

agent.py

```
from google.adk.agents.llm_agent import Agent
from google.adk.apps import App

root_agent = Agent(
    model='gemini-2.5-flash',
    name='greeter_agent',
    description='An agent that provides a friendly greeting.',
    instruction='Reply with Hello, World!',
)

app = App(
    name="agents",
    root_agent=root_agent,
    # Optionally include App-level features:
    # plugins, context_cache_config, resumability_config
)
```

Recommended: Use `app` variable name

In your agent project code, set your **_App_** object to the variable name
`app` so it is compatible with the ADK command line interface runner tools.

### Run your App agent [¶](https://google.github.io/adk-docs/apps/\#run-your-app-agent "Permanent link")

You can use the **_Runner_** class to run your agent workflow using the
`app` parameter, as shown in the following code sample:

main.py

```
import asyncio
from dotenv import load_dotenv
from google.adk.runners import InMemoryRunner
from agent import app # import code from agent.py

load_dotenv() # load API keys and settings
# Set a Runner using the imported application object
runner = InMemoryRunner(app=app)

async def main():
    try:  # run_debug() requires ADK Python 1.18 or higher:
        response = await runner.run_debug("Hello there!")

    except Exception as e:
        print(f"An error occurred during agent execution: {e}")

if __name__ == "__main__":
    asyncio.run(main())
```

Version requirement for `Runner.run_debug()`

The `Runner.run_debug()` command requires ADK Python v1.18.0 or higher.
You can also use `Runner.run()`, which requires more setup code. For
more details, see the

Run your App agent with the `main.py` code using the following command:

```
python main.py
```

## Next steps [¶](https://google.github.io/adk-docs/apps/\#next-steps "Permanent link")

For a more complete sample code implementation, see the
[Hello World App](https://github.com/google/adk-python/tree/main/contributing/samples/hello_world_app)
code example.

Back to top

## ADK Installation Guide
[Skip to content](https://google.github.io/adk-docs/get-started/installation/#installing-adk)

[Edit this page](https://github.com/google/adk-docs/edit/main/docs/get-started/installation.md "Edit this page") [View source of this page](https://github.com/google/adk-docs/raw/main/docs/get-started/installation.md "View source of this page")

# Installing ADK [¶](https://google.github.io/adk-docs/get-started/installation/\#installing-adk "Permanent link")

[Python](https://google.github.io/adk-docs/get-started/installation/#python)[Go](https://google.github.io/adk-docs/get-started/installation/#go)[Java](https://google.github.io/adk-docs/get-started/installation/#java)

## Create & activate virtual environment [¶](https://google.github.io/adk-docs/get-started/installation/\#create-activate-virtual-environment "Permanent link")

We recommend creating a virtual Python environment using
[venv](https://docs.python.org/3/library/venv.html):

```
python -m venv .venv
```

Now, you can activate the virtual environment using the appropriate command for
your operating system and environment:

```
# Mac / Linux
source .venv/bin/activate

# Windows CMD:
.venv\Scripts\activate.bat

# Windows PowerShell:
.venv\Scripts\Activate.ps1
```

### Install ADK [¶](https://google.github.io/adk-docs/get-started/installation/\#install-adk "Permanent link")

```
pip install google-adk
```

(Optional) Verify your installation:

```
pip show google-adk
```

## Create a new Go module [¶](https://google.github.io/adk-docs/get-started/installation/\#create-a-new-go-module "Permanent link")

If you are starting a new project, you can create a new Go module:

```
go mod init example.com/my-agent
```

## Install ADK [¶](https://google.github.io/adk-docs/get-started/installation/\#install-adk_1 "Permanent link")

To add the ADK to your project, run the following command:

```
go get google.golang.org/adk
```

This will add the ADK as a dependency to your `go.mod` file.

(Optional) Verify your installation by checking your `go.mod` file for the `google.golang.org/adk` entry.

You can either use maven or gradle to add the `google-adk` and `google-adk-dev` package.

`google-adk` is the core Java ADK library. Java ADK also comes with a pluggable example SpringBoot server to run your agents seamlessly. This optional
package is present as part of `google-adk-dev`.

If you are using maven, add the following to your `pom.xml`:

pom.xml

```
<?xml version="1.0" encoding="UTF-8"?>
<project xmlns="http://maven.apache.org/POM/4.0.0"
        xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
        xsi:schemaLocation="http://maven.apache.org/POM/4.0.0 http://maven.apache.org/xsd/maven-4.0.0.xsd">
    <modelVersion>4.0.0</modelVersion>

    <groupId>com.example.agent</groupId>
    <artifactId>adk-agents</artifactId>
    <version>1.0-SNAPSHOT</version>

    <!-- Specify the version of Java you'll be using -->
    <properties>
        <maven.compiler.source>17</maven.compiler.source>
        <maven.compiler.target>17</maven.compiler.target>
        <project.build.sourceEncoding>UTF-8</project.build.sourceEncoding>
    </properties>

    <dependencies>
        <!-- The ADK core dependency -->
        <dependency>
            <groupId>com.google.adk</groupId>
            <artifactId>google-adk</artifactId>
            <version>0.3.0</version>
        </dependency>
        <!-- The ADK dev web UI to debug your agent -->
        <dependency>
            <groupId>com.google.adk</groupId>
            <artifactId>google-adk-dev</artifactId>
            <version>0.3.0</version>
        </dependency>
    </dependencies>

</project>
```

Here's a [complete pom.xml](https://github.com/google/adk-docs/tree/main/examples/java/cloud-run/pom.xml) file for reference.

If you are using gradle, add the dependency to your build.gradle:

build.gradle

```
dependencies {
    implementation 'com.google.adk:google-adk:0.2.0'
    implementation 'com.google.adk:google-adk-dev:0.2.0'
}
```

You should also configure Gradle to pass `-parameters` to `javac`. (Alternatively, use `@Schema(name = "...")`).

## Next steps [¶](https://google.github.io/adk-docs/get-started/installation/\#next-steps "Permanent link")

- Try creating your first agent with the [**Quickstart**](https://google.github.io/adk-docs/get-started/quickstart/)

Back to top

## Connection Details Overview
[java.lang.Object](https://docs.oracle.com/en/java/javase/17/docs/api/java.base/java/lang/Object.html "class or interface in java.lang")

com.google.adk.tools.applicationintegrationtoolset.ConnectionsClient.ConnectionDetails

Enclosing class:`ConnectionsClient`

* * *

public static class ConnectionsClient.ConnectionDetailsextends [Object](https://docs.oracle.com/en/java/javase/17/docs/api/java.base/java/lang/Object.html "class or interface in java.lang")

Represents details of a connection.

- ## Field Summary [![Link icon](https://google.github.io/adk-docs/api-reference/java/resource-files/link.svg)](https://google.github.io/adk-docs/api-reference/java/com/google/adk/tools/applicationintegrationtoolset/ConnectionsClient.ConnectionDetails.html\#field-summary)



Fields





Modifier and Type



Field



Description



`String`



`host`







`String`



`name`







`String`



`serviceName`

- ## Constructor Summary [![Link icon](https://google.github.io/adk-docs/api-reference/java/resource-files/link.svg)](https://google.github.io/adk-docs/api-reference/java/com/google/adk/tools/applicationintegrationtoolset/ConnectionsClient.ConnectionDetails.html\#constructor-summary)



Constructors





Constructor



Description



`ConnectionDetails()`

- ## Method Summary [![Link icon](https://google.github.io/adk-docs/api-reference/java/resource-files/link.svg)](https://google.github.io/adk-docs/api-reference/java/com/google/adk/tools/applicationintegrationtoolset/ConnectionsClient.ConnectionDetails.html\#method-summary)





### Methods inherited from class java.lang. [Object](https://docs.oracle.com/en/java/javase/17/docs/api/java.base/java/lang/Object.html "class or interface in java.lang") [![Link icon](https://google.github.io/adk-docs/api-reference/java/resource-files/link.svg)](https://google.github.io/adk-docs/api-reference/java/com/google/adk/tools/applicationintegrationtoolset/ConnectionsClient.ConnectionDetails.html\#methods-inherited-from-class-java.lang.Object)

`clone, equals, finalize, getClass, hashCode, notify, notifyAll, toString, wait, wait, wait`


- ## Field Details [![Link icon](https://google.github.io/adk-docs/api-reference/java/resource-files/link.svg)](https://google.github.io/adk-docs/api-reference/java/com/google/adk/tools/applicationintegrationtoolset/ConnectionsClient.ConnectionDetails.html\#field-detail)



- ### name [![Link icon](https://google.github.io/adk-docs/api-reference/java/resource-files/link.svg)](https://google.github.io/adk-docs/api-reference/java/com/google/adk/tools/applicationintegrationtoolset/ConnectionsClient.ConnectionDetails.html\#name)





public[String](https://docs.oracle.com/en/java/javase/17/docs/api/java.base/java/lang/String.html "class or interface in java.lang")name

- ### serviceName [![Link icon](https://google.github.io/adk-docs/api-reference/java/resource-files/link.svg)](https://google.github.io/adk-docs/api-reference/java/com/google/adk/tools/applicationintegrationtoolset/ConnectionsClient.ConnectionDetails.html\#serviceName)





public[String](https://docs.oracle.com/en/java/javase/17/docs/api/java.base/java/lang/String.html "class or interface in java.lang")serviceName

- ### host [![Link icon](https://google.github.io/adk-docs/api-reference/java/resource-files/link.svg)](https://google.github.io/adk-docs/api-reference/java/com/google/adk/tools/applicationintegrationtoolset/ConnectionsClient.ConnectionDetails.html\#host)





public[String](https://docs.oracle.com/en/java/javase/17/docs/api/java.base/java/lang/String.html "class or interface in java.lang")host


- ## Constructor Details [![Link icon](https://google.github.io/adk-docs/api-reference/java/resource-files/link.svg)](https://google.github.io/adk-docs/api-reference/java/com/google/adk/tools/applicationintegrationtoolset/ConnectionsClient.ConnectionDetails.html\#constructor-detail)



- ### ConnectionDetails [![Link icon](https://google.github.io/adk-docs/api-reference/java/resource-files/link.svg)](https://google.github.io/adk-docs/api-reference/java/com/google/adk/tools/applicationintegrationtoolset/ConnectionsClient.ConnectionDetails.html\#%3Cinit%3E())





publicConnectionDetails()

## Event Stream API
[java.lang.Object](https://docs.oracle.com/en/java/javase/17/docs/api/java.base/java/lang/Object.html "class or interface in java.lang")

com.google.adk.events.EventStream

All Implemented Interfaces:`Iterable<Event>`

* * *

public class EventStreamextends [Object](https://docs.oracle.com/en/java/javase/17/docs/api/java.base/java/lang/Object.html "class or interface in java.lang")
implements [Iterable](https://docs.oracle.com/en/java/javase/17/docs/api/java.base/java/lang/Iterable.html "class or interface in java.lang") < [Event](https://google.github.io/adk-docs/api-reference/java/com/google/adk/events/Event.html "class in com.google.adk.events") >

Iterable stream of [`Event`](https://google.github.io/adk-docs/api-reference/java/com/google/adk/events/Event.html "class in com.google.adk.events") objects.

- ## Constructor Summary [![Link icon](https://google.github.io/adk-docs/api-reference/java/resource-files/link.svg)](https://google.github.io/adk-docs/api-reference/java/com/google/adk/events/EventStream.html\#constructor-summary)



Constructors





Constructor



Description



`EventStream(Supplier<Event> eventSupplier)`





Constructs a new event stream.

- ## Method Summary [![Link icon](https://google.github.io/adk-docs/api-reference/java/resource-files/link.svg)](https://google.github.io/adk-docs/api-reference/java/com/google/adk/events/EventStream.html\#method-summary)





All MethodsInstance MethodsConcrete Methods







Modifier and Type



Method



Description



`Iterator<Event>`



`iterator()`





Returns an iterator that fetches events lazily.













### Methods inherited from class java.lang. [Object](https://docs.oracle.com/en/java/javase/17/docs/api/java.base/java/lang/Object.html "class or interface in java.lang") [![Link icon](https://google.github.io/adk-docs/api-reference/java/resource-files/link.svg)](https://google.github.io/adk-docs/api-reference/java/com/google/adk/events/EventStream.html\#methods-inherited-from-class-java.lang.Object)

`clone, equals, finalize, getClass, hashCode, notify, notifyAll, toString, wait, wait, wait`





### Methods inherited from interface java.lang. [Iterable](https://docs.oracle.com/en/java/javase/17/docs/api/java.base/java/lang/Iterable.html "class or interface in java.lang") [![Link icon](https://google.github.io/adk-docs/api-reference/java/resource-files/link.svg)](https://google.github.io/adk-docs/api-reference/java/com/google/adk/events/EventStream.html\#methods-inherited-from-class-java.lang.Iterable)

`forEach, spliterator`


- ## Constructor Details [![Link icon](https://google.github.io/adk-docs/api-reference/java/resource-files/link.svg)](https://google.github.io/adk-docs/api-reference/java/com/google/adk/events/EventStream.html\#constructor-detail)



- ### EventStream [![Link icon](https://google.github.io/adk-docs/api-reference/java/resource-files/link.svg)](https://google.github.io/adk-docs/api-reference/java/com/google/adk/events/EventStream.html\#%3Cinit%3E(java.util.function.Supplier))





publicEventStream( [Supplier](https://docs.oracle.com/en/java/javase/17/docs/api/java.base/java/util/function/Supplier.html "class or interface in java.util.function") < [Event](https://google.github.io/adk-docs/api-reference/java/com/google/adk/events/Event.html "class in com.google.adk.events") > eventSupplier)



Constructs a new event stream.


- ## Method Details [![Link icon](https://google.github.io/adk-docs/api-reference/java/resource-files/link.svg)](https://google.github.io/adk-docs/api-reference/java/com/google/adk/events/EventStream.html\#method-detail)



- ### iterator [![Link icon](https://google.github.io/adk-docs/api-reference/java/resource-files/link.svg)](https://google.github.io/adk-docs/api-reference/java/com/google/adk/events/EventStream.html\#iterator())





public[Iterator](https://docs.oracle.com/en/java/javase/17/docs/api/java.base/java/util/Iterator.html "class or interface in java.util") < [Event](https://google.github.io/adk-docs/api-reference/java/com/google/adk/events/Event.html "class in com.google.adk.events") >iterator()



Returns an iterator that fetches events lazily.

Specified by:`iterator` in interface `Iterable<Event>`

## Callbacks AfterToolCallback
Packages that use [Callbacks.AfterToolCallback](https://google.github.io/adk-docs/api-reference/java/com/google/adk/agents/Callbacks.AfterToolCallback.html "interface in com.google.adk.agents")

Package

Description

[com.google.adk.agents](https://google.github.io/adk-docs/api-reference/java/com/google/adk/agents/class-use/Callbacks.AfterToolCallback.html#com.google.adk.agents)

- ## Uses of [Callbacks.AfterToolCallback](https://google.github.io/adk-docs/api-reference/java/com/google/adk/agents/Callbacks.AfterToolCallback.html "interface in com.google.adk.agents") in [com.google.adk.agents](https://google.github.io/adk-docs/api-reference/java/com/google/adk/agents/package-summary.html) [![Link icon](https://google.github.io/adk-docs/api-reference/java/resource-files/link.svg)](https://google.github.io/adk-docs/api-reference/java/com/google/adk/agents/class-use/Callbacks.AfterToolCallback.html\#com.google.adk.agents)



Methods in [com.google.adk.agents](https://google.github.io/adk-docs/api-reference/java/com/google/adk/agents/package-summary.html) that return types with arguments of type [Callbacks.AfterToolCallback](https://google.github.io/adk-docs/api-reference/java/com/google/adk/agents/Callbacks.AfterToolCallback.html "interface in com.google.adk.agents")





Modifier and Type



Method



Description



`Optional<List<Callbacks.AfterToolCallback>>`



LlmAgent.`afterToolCallback()`









Methods in [com.google.adk.agents](https://google.github.io/adk-docs/api-reference/java/com/google/adk/agents/package-summary.html) with parameters of type [Callbacks.AfterToolCallback](https://google.github.io/adk-docs/api-reference/java/com/google/adk/agents/Callbacks.AfterToolCallback.html "interface in com.google.adk.agents")





Modifier and Type



Method



Description



`LlmAgent.Builder`



LlmAgent.Builder.`afterToolCallback(Callbacks.AfterToolCallback afterToolCallback)`

## Sequential Agents Overview
[Skip to content](https://google.github.io/adk-docs/agents/workflow-agents/sequential-agents/#sequential-agents)

[Edit this page](https://github.com/google/adk-docs/edit/main/docs/agents/workflow-agents/sequential-agents.md "Edit this page") [View source of this page](https://github.com/google/adk-docs/raw/main/docs/agents/workflow-agents/sequential-agents.md "View source of this page")

# Sequential agents [¶](https://google.github.io/adk-docs/agents/workflow-agents/sequential-agents/\#sequential-agents "Permanent link")

Supported in ADKPython v0.1.0Go v0.1.0Java v0.2.0

The `SequentialAgent` is a [workflow agent](https://google.github.io/adk-docs/agents/workflow-agents/) that executes its sub-agents in the order they are specified in the list.
Use the `SequentialAgent` when you want the execution to occur in a fixed, strict order.

### Example [¶](https://google.github.io/adk-docs/agents/workflow-agents/sequential-agents/\#example "Permanent link")

- You want to build an agent that can summarize any webpage, using two tools: `Get Page Contents` and `Summarize Page`. Because the agent must always call `Get Page Contents` before calling `Summarize Page` (you can't summarize from nothing!), you should build your agent using a `SequentialAgent`.

As with other [workflow agents](https://google.github.io/adk-docs/agents/workflow-agents/), the `SequentialAgent` is not powered by an LLM, and is thus deterministic in how it executes. That being said, workflow agents are concerned only with their execution (i.e. in sequence), and not their internal logic; the tools or sub-agents of a workflow agent may or may not utilize LLMs.

### How it works [¶](https://google.github.io/adk-docs/agents/workflow-agents/sequential-agents/\#how-it-works "Permanent link")

When the `SequentialAgent`'s `Run Async` method is called, it performs the following actions:

1. **Iteration:** It iterates through the sub agents list in the order they were provided.
2. **Sub-Agent Execution:** For each sub-agent in the list, it calls the sub-agent's `Run Async` method.

![Sequential Agent](https://google.github.io/adk-docs/assets/sequential-agent.png)

### Full Example: Code Development Pipeline [¶](https://google.github.io/adk-docs/agents/workflow-agents/sequential-agents/\#full-example-code-development-pipeline "Permanent link")

Consider a simplified code development pipeline:

- **Code Writer Agent:** An LLM Agent that generates initial code based on a specification.
- **Code Reviewer Agent:** An LLM Agent that reviews the generated code for errors, style issues, and adherence to best practices. It receives the output of the Code Writer Agent.
- **Code Refactorer Agent:** An LLM Agent that takes the reviewed code (and the reviewer's comments) and refactors it to improve quality and address issues.

A `SequentialAgent` is perfect for this:

```
SequentialAgent(sub_agents=[CodeWriterAgent, CodeReviewerAgent, CodeRefactorerAgent])
```

This ensures the code is written, _then_ reviewed, and _finally_ refactored, in a strict, dependable order. **The output from each sub-agent is passed to the next by storing them in state via [Output Key](https://google.github.io/adk-docs/agents/llm-agents/#structuring-data-input_schema-output_schema-output_key)**.

Shared Invocation Context

The `SequentialAgent` passes the same `InvocationContext` to each of its sub-agents. This means they all share the same session state, including the temporary (`temp:`) namespace, making it easy to pass data between steps within a single turn.

Code

[Python](https://google.github.io/adk-docs/agents/workflow-agents/sequential-agents/#python)[Go](https://google.github.io/adk-docs/agents/workflow-agents/sequential-agents/#go)[Java](https://google.github.io/adk-docs/agents/workflow-agents/sequential-agents/#java)

````
# Part of agent.py --> Follow https://google.github.io/adk-docs/get-started/quickstart/ to learn the setup

# --- 1. Define Sub-Agents for Each Pipeline Stage ---

# Code Writer Agent
# Takes the initial specification (from user query) and writes code.
code_writer_agent = LlmAgent(
    name="CodeWriterAgent",
    model=GEMINI_MODEL,
    # Change 3: Improved instruction
    instruction="""You are a Python Code Generator.
Based *only* on the user's request, write Python code that fulfills the requirement.
Output *only* the complete Python code block, enclosed in triple backticks (```python ... ```).
Do not add any other text before or after the code block.
""",
    description="Writes initial Python code based on a specification.",
    output_key="generated_code" # Stores output in state['generated_code']
)

# Code Reviewer Agent
# Takes the code generated by the previous agent (read from state) and provides feedback.
code_reviewer_agent = LlmAgent(
    name="CodeReviewerAgent",
    model=GEMINI_MODEL,
    # Change 3: Improved instruction, correctly using state key injection
    instruction="""You are an expert Python Code Reviewer.
    Your task is to provide constructive feedback on the provided code.

    **Code to Review:**
    ```python
    {generated_code}
    ```

**Review Criteria:**
1.  **Correctness:** Does the code work as intended? Are there logic errors?
2.  **Readability:** Is the code clear and easy to understand? Follows PEP 8 style guidelines?
3.  **Efficiency:** Is the code reasonably efficient? Any obvious performance bottlenecks?
4.  **Edge Cases:** Does the code handle potential edge cases or invalid inputs gracefully?
5.  **Best Practices:** Does the code follow common Python best practices?

**Output:**
Provide your feedback as a concise, bulleted list. Focus on the most important points for improvement.
If the code is excellent and requires no changes, simply state: "No major issues found."
Output *only* the review comments or the "No major issues" statement.
""",
    description="Reviews code and provides feedback.",
    output_key="review_comments", # Stores output in state['review_comments']
)

# Code Refactorer Agent
# Takes the original code and the review comments (read from state) and refactors the code.
code_refactorer_agent = LlmAgent(
    name="CodeRefactorerAgent",
    model=GEMINI_MODEL,
    # Change 3: Improved instruction, correctly using state key injection
    instruction="""You are a Python Code Refactoring AI.
Your goal is to improve the given Python code based on the provided review comments.

  **Original Code:**
  ```python
  {generated_code}
  ```

  **Review Comments:**
  {review_comments}

**Task:**
Carefully apply the suggestions from the review comments to refactor the original code.
If the review comments state "No major issues found," return the original code unchanged.
Ensure the final code is complete, functional, and includes necessary imports and docstrings.

**Output:**
Output *only* the final, refactored Python code block, enclosed in triple backticks (```python ... ```).
Do not add any other text before or after the code block.
""",
    description="Refactors code based on review comments.",
    output_key="refactored_code", # Stores output in state['refactored_code']
)

# --- 2. Create the SequentialAgent ---
# This agent orchestrates the pipeline by running the sub_agents in order.
code_pipeline_agent = SequentialAgent(
    name="CodePipelineAgent",
    sub_agents=[code_writer_agent, code_reviewer_agent, code_refactorer_agent],
    description="Executes a sequence of code writing, reviewing, and refactoring.",
    # The agents will run in the order provided: Writer -> Reviewer -> Refactorer
)

# For ADK tools compatibility, the root agent must be named `root_agent`
root_agent = code_pipeline_agent
````

```
    model, err := gemini.NewModel(ctx, modelName, &genai.ClientConfig{})
    if err != nil {
        return fmt.Errorf("failed to create model: %v", err)
    }

    codeWriterAgent, err := llmagent.New(llmagent.Config{
        Name:        "CodeWriterAgent",
        Model:       model,
        Description: "Writes initial Go code based on a specification.",
        Instruction: `You are a Go Code Generator.
Based *only* on the user's request, write Go code that fulfills the requirement.
Output *only* the complete Go code block, enclosed in triple backticks ('''go ... ''').
Do not add any other text before or after the code block.`,
        OutputKey: "generated_code",
    })
    if err != nil {
        return fmt.Errorf("failed to create code writer agent: %v", err)
    }

    codeReviewerAgent, err := llmagent.New(llmagent.Config{
        Name:        "CodeReviewerAgent",
        Model:       model,
        Description: "Reviews code and provides feedback.",
        Instruction: `You are an expert Go Code Reviewer.
Your task is to provide constructive feedback on the provided code.

**Code to Review:**
'''go
{generated_code}
'''

**Review Criteria:**
1.  **Correctness:** Does the code work as intended? Are there logic errors?
2.  **Readability:** Is the code clear and easy to understand? Follows Go style guidelines?
3.  **Idiomatic Go:** Does the code use Go's features in a natural and standard way?
4.  **Edge Cases:** Does the code handle potential edge cases or invalid inputs gracefully?
5.  **Best Practices:** Does the code follow common Go best practices?

**Output:**
Provide your feedback as a concise, bulleted list. Focus on the most important points for improvement.
If the code is excellent and requires no changes, simply state: "No major issues found."
Output *only* the review comments or the "No major issues" statement.`,
        OutputKey: "review_comments",
    })
    if err != nil {
        return fmt.Errorf("failed to create code reviewer agent: %v", err)
    }

    codeRefactorerAgent, err := llmagent.New(llmagent.Config{
        Name:        "CodeRefactorerAgent",
        Model:       model,
        Description: "Refactors code based on review comments.",
        Instruction: `You are a Go Code Refactoring AI.
Your goal is to improve the given Go code based on the provided review comments.

**Original Code:**
'''go
{generated_code}
'''

**Review Comments:**
{review_comments}

**Task:**
Carefully apply the suggestions from the review comments to refactor the original code.
If the review comments state "No major issues found," return the original code unchanged.
Ensure the final code is complete, functional, and includes necessary imports.

**Output:**
Output *only* the final, refactored Go code block, enclosed in triple backticks ('''go ... ''').
Do not add any other text before or after the code block.`,
        OutputKey: "refactored_code",
    })
    if err != nil {
        return fmt.Errorf("failed to create code refactorer agent: %v", err)
    }

    codePipelineAgent, err := sequentialagent.New(sequentialagent.Config{
        AgentConfig: agent.Config{
            Name:        appName,
            Description: "Executes a sequence of code writing, reviewing, and refactoring.",
            SubAgents: []agent.Agent{
                codeWriterAgent,
                codeReviewerAgent,
                codeRefactorerAgent,
            },
        },
    })
    if err != nil {
        return fmt.Errorf("failed to create sequential agent: %v", err)
    }
```

````
import com.google.adk.agents.LlmAgent;
import com.google.adk.agents.SequentialAgent;
import com.google.adk.events.Event;
import com.google.adk.runner.InMemoryRunner;
import com.google.adk.sessions.Session;
import com.google.genai.types.Content;
import com.google.genai.types.Part;
import io.reactivex.rxjava3.core.Flowable;

public class SequentialAgentExample {

  private static final String APP_NAME = "CodePipelineAgent";
  private static final String USER_ID = "test_user_456";
  private static final String MODEL_NAME = "gemini-2.0-flash";

  public static void main(String[] args) {
    SequentialAgentExample sequentialAgentExample = new SequentialAgentExample();
    sequentialAgentExample.runAgent(
        "Write a Java function to calculate the factorial of a number.");
  }

  public void runAgent(String prompt) {

    LlmAgent codeWriterAgent =
        LlmAgent.builder()
            .model(MODEL_NAME)
            .name("CodeWriterAgent")
            .description("Writes initial Java code based on a specification.")
            .instruction(
                """
                You are a Java Code Generator.
                Based *only* on the user's request, write Java code that fulfills the requirement.
                Output *only* the complete Java code block, enclosed in triple backticks (```java ... ```).
                Do not add any other text before or after the code block.
                """)
            .outputKey("generated_code")
            .build();

    LlmAgent codeReviewerAgent =
        LlmAgent.builder()
            .model(MODEL_NAME)
            .name("CodeReviewerAgent")
            .description("Reviews code and provides feedback.")
            .instruction(
                """
                    You are an expert Java Code Reviewer.
                    Your task is to provide constructive feedback on the provided code.

                    **Code to Review:**
                    ```java
                    {generated_code}
                    ```

                    **Review Criteria:**
                    1.  **Correctness:** Does the code work as intended? Are there logic errors?
                    2.  **Readability:** Is the code clear and easy to understand? Follows Java style guidelines?
                    3.  **Efficiency:** Is the code reasonably efficient? Any obvious performance bottlenecks?
                    4.  **Edge Cases:** Does the code handle potential edge cases or invalid inputs gracefully?
                    5.  **Best Practices:** Does the code follow common Java best practices?

                    **Output:**
                    Provide your feedback as a concise, bulleted list. Focus on the most important points for improvement.
                    If the code is excellent and requires no changes, simply state: "No major issues found."
                    Output *only* the review comments or the "No major issues" statement.
                """)
            .outputKey("review_comments")
            .build();

    LlmAgent codeRefactorerAgent =
        LlmAgent.builder()
            .model(MODEL_NAME)
            .name("CodeRefactorerAgent")
            .description("Refactors code based on review comments.")
            .instruction(
                """
                You are a Java Code Refactoring AI.
                Your goal is to improve the given Java code based on the provided review comments.

                  **Original Code:**
                  ```java
                  {generated_code}
                  ```

                  **Review Comments:**
                  {review_comments}

                **Task:**
                Carefully apply the suggestions from the review comments to refactor the original code.
                If the review comments state "No major issues found," return the original code unchanged.
                Ensure the final code is complete, functional, and includes necessary imports and docstrings.

                **Output:**
                Output *only* the final, refactored Java code block, enclosed in triple backticks (```java ... ```).
                Do not add any other text before or after the code block.
                """)
            .outputKey("refactored_code")
            .build();

    SequentialAgent codePipelineAgent =
        SequentialAgent.builder()
            .name(APP_NAME)
            .description("Executes a sequence of code writing, reviewing, and refactoring.")
            // The agents will run in the order provided: Writer -> Reviewer -> Refactorer
            .subAgents(codeWriterAgent, codeReviewerAgent, codeRefactorerAgent)
            .build();

    // Create an InMemoryRunner
    InMemoryRunner runner = new InMemoryRunner(codePipelineAgent, APP_NAME);
    // InMemoryRunner automatically creates a session service. Create a session using the service
    Session session = runner.sessionService().createSession(APP_NAME, USER_ID).blockingGet();
    Content userMessage = Content.fromParts(Part.fromText(prompt));

    // Run the agent
    Flowable<Event> eventStream = runner.runAsync(USER_ID, session.id(), userMessage);

    // Stream event response
    eventStream.blockingForEach(
        event -> {
          if (event.finalResponse()) {
            System.out.println(event.stringifyContent());
          }
        });
  }
}
````

Back to top

## GenAI Agents Runner
[java.lang.Object](https://docs.oracle.com/en/java/javase/17/docs/api/java.base/java/lang/Object.html "class or interface in java.lang")

com.google.adk.runner.Runner

Direct Known Subclasses:`InMemoryRunner`

* * *

public class Runnerextends [Object](https://docs.oracle.com/en/java/javase/17/docs/api/java.base/java/lang/Object.html "class or interface in java.lang")

The main class for the GenAI Agents runner.

- ## Constructor Summary [![Link icon](https://google.github.io/adk-docs/api-reference/java/resource-files/link.svg)](https://google.github.io/adk-docs/api-reference/java/com/google/adk/runner/Runner.html\#constructor-summary)



Constructors





Constructor



Description



`Runner(BaseAgent agent,
String appName,
BaseArtifactService artifactService,
BaseSessionService sessionService)`





Creates a new `Runner`.

- ## Method Summary [![Link icon](https://google.github.io/adk-docs/api-reference/java/resource-files/link.svg)](https://google.github.io/adk-docs/api-reference/java/com/google/adk/runner/Runner.html\#method-summary)





All MethodsInstance MethodsConcrete Methods







Modifier and Type



Method



Description



`BaseAgent`



`agent()`







`String`



`appName()`







`BaseArtifactService`



`artifactService()`







`io.reactivex.rxjava3.core.Flowable<Event>`



`runAsync(Session session,
com.google.genai.types.Content newMessage,
RunConfig runConfig)`





Runs the agent in the standard mode using a provided Session object.





`io.reactivex.rxjava3.core.Flowable<Event>`



`runAsync(String userId,
String sessionId,
com.google.genai.types.Content newMessage)`





Asynchronously runs the agent for a given user and session, processing a new message and using
a default [`RunConfig`](https://google.github.io/adk-docs/api-reference/java/com/google/adk/agents/RunConfig.html "class in com.google.adk.agents").





`io.reactivex.rxjava3.core.Flowable<Event>`



`runAsync(String userId,
String sessionId,
com.google.genai.types.Content newMessage,
RunConfig runConfig)`





Runs the agent in the standard mode.





`io.reactivex.rxjava3.core.Flowable<Event>`



`runLive(Session session,
LiveRequestQueue liveRequestQueue,
RunConfig runConfig)`





Runs the agent in live mode, appending generated events to the session.





`io.reactivex.rxjava3.core.Flowable<Event>`



`runLive(String userId,
String sessionId,
LiveRequestQueue liveRequestQueue,
RunConfig runConfig)`





Retrieves the session and runs the agent in live mode.





`io.reactivex.rxjava3.core.Flowable<Event>`



`runWithSessionId(String sessionId,
com.google.genai.types.Content newMessage,
RunConfig runConfig)`





Runs the agent asynchronously with a default user ID.





`BaseSessionService`



`sessionService()`















### Methods inherited from class java.lang. [Object](https://docs.oracle.com/en/java/javase/17/docs/api/java.base/java/lang/Object.html "class or interface in java.lang") [![Link icon](https://google.github.io/adk-docs/api-reference/java/resource-files/link.svg)](https://google.github.io/adk-docs/api-reference/java/com/google/adk/runner/Runner.html\#methods-inherited-from-class-java.lang.Object)

`clone, equals, finalize, getClass, hashCode, notify, notifyAll, toString, wait, wait, wait`


- ## Constructor Details [![Link icon](https://google.github.io/adk-docs/api-reference/java/resource-files/link.svg)](https://google.github.io/adk-docs/api-reference/java/com/google/adk/runner/Runner.html\#constructor-detail)



- ### Runner [![Link icon](https://google.github.io/adk-docs/api-reference/java/resource-files/link.svg)](https://google.github.io/adk-docs/api-reference/java/com/google/adk/runner/Runner.html\#%3Cinit%3E(com.google.adk.agents.BaseAgent,java.lang.String,com.google.adk.artifacts.BaseArtifactService,com.google.adk.sessions.BaseSessionService))





publicRunner( [BaseAgent](https://google.github.io/adk-docs/api-reference/java/com/google/adk/agents/BaseAgent.html "class in com.google.adk.agents") agent,
[String](https://docs.oracle.com/en/java/javase/17/docs/api/java.base/java/lang/String.html "class or interface in java.lang") appName,
[BaseArtifactService](https://google.github.io/adk-docs/api-reference/java/com/google/adk/artifacts/BaseArtifactService.html "interface in com.google.adk.artifacts") artifactService,
[BaseSessionService](https://google.github.io/adk-docs/api-reference/java/com/google/adk/sessions/BaseSessionService.html "interface in com.google.adk.sessions") sessionService)



Creates a new `Runner`.


- ## Method Details [![Link icon](https://google.github.io/adk-docs/api-reference/java/resource-files/link.svg)](https://google.github.io/adk-docs/api-reference/java/com/google/adk/runner/Runner.html\#method-detail)



- ### agent [![Link icon](https://google.github.io/adk-docs/api-reference/java/resource-files/link.svg)](https://google.github.io/adk-docs/api-reference/java/com/google/adk/runner/Runner.html\#agent())





public[BaseAgent](https://google.github.io/adk-docs/api-reference/java/com/google/adk/agents/BaseAgent.html "class in com.google.adk.agents")agent()

- ### appName [![Link icon](https://google.github.io/adk-docs/api-reference/java/resource-files/link.svg)](https://google.github.io/adk-docs/api-reference/java/com/google/adk/runner/Runner.html\#appName())





public[String](https://docs.oracle.com/en/java/javase/17/docs/api/java.base/java/lang/String.html "class or interface in java.lang")appName()

- ### artifactService [![Link icon](https://google.github.io/adk-docs/api-reference/java/resource-files/link.svg)](https://google.github.io/adk-docs/api-reference/java/com/google/adk/runner/Runner.html\#artifactService())





public[BaseArtifactService](https://google.github.io/adk-docs/api-reference/java/com/google/adk/artifacts/BaseArtifactService.html "interface in com.google.adk.artifacts")artifactService()

- ### sessionService [![Link icon](https://google.github.io/adk-docs/api-reference/java/resource-files/link.svg)](https://google.github.io/adk-docs/api-reference/java/com/google/adk/runner/Runner.html\#sessionService())





public[BaseSessionService](https://google.github.io/adk-docs/api-reference/java/com/google/adk/sessions/BaseSessionService.html "interface in com.google.adk.sessions")sessionService()

- ### runAsync [![Link icon](https://google.github.io/adk-docs/api-reference/java/resource-files/link.svg)](https://google.github.io/adk-docs/api-reference/java/com/google/adk/runner/Runner.html\#runAsync(java.lang.String,java.lang.String,com.google.genai.types.Content,com.google.adk.agents.RunConfig))





publicio.reactivex.rxjava3.core.Flowable< [Event](https://google.github.io/adk-docs/api-reference/java/com/google/adk/events/Event.html "class in com.google.adk.events") >runAsync( [String](https://docs.oracle.com/en/java/javase/17/docs/api/java.base/java/lang/String.html "class or interface in java.lang") userId,
[String](https://docs.oracle.com/en/java/javase/17/docs/api/java.base/java/lang/String.html "class or interface in java.lang") sessionId,
com.google.genai.types.Content newMessage,
[RunConfig](https://google.github.io/adk-docs/api-reference/java/com/google/adk/agents/RunConfig.html "class in com.google.adk.agents") runConfig)



Runs the agent in the standard mode.

Parameters:`userId` \- The ID of the user for the session.`sessionId` \- The ID of the session to run the agent in.`newMessage` \- The new message from the user to process.`runConfig` \- Configuration for the agent run.Returns:A Flowable stream of [`Event`](https://google.github.io/adk-docs/api-reference/java/com/google/adk/events/Event.html "class in com.google.adk.events") objects generated by the agent during execution.

- ### runAsync [![Link icon](https://google.github.io/adk-docs/api-reference/java/resource-files/link.svg)](https://google.github.io/adk-docs/api-reference/java/com/google/adk/runner/Runner.html\#runAsync(java.lang.String,java.lang.String,com.google.genai.types.Content))





publicio.reactivex.rxjava3.core.Flowable< [Event](https://google.github.io/adk-docs/api-reference/java/com/google/adk/events/Event.html "class in com.google.adk.events") >runAsync( [String](https://docs.oracle.com/en/java/javase/17/docs/api/java.base/java/lang/String.html "class or interface in java.lang") userId,
[String](https://docs.oracle.com/en/java/javase/17/docs/api/java.base/java/lang/String.html "class or interface in java.lang") sessionId,
com.google.genai.types.Content newMessage)



Asynchronously runs the agent for a given user and session, processing a new message and using
a default [`RunConfig`](https://google.github.io/adk-docs/api-reference/java/com/google/adk/agents/RunConfig.html "class in com.google.adk.agents").



This method initiates an agent execution within the specified session, appending the
provided new message to the session's history. It utilizes a default `RunConfig` to
control execution parameters. The method returns a stream of [`Event`](https://google.github.io/adk-docs/api-reference/java/com/google/adk/events/Event.html "class in com.google.adk.events") objects representing
the agent's activity during the run.



Parameters:`userId` \- The ID of the user initiating the session.`sessionId` \- The ID of the session in which the agent will run.`newMessage` \- The new `Content` message to be processed by the agent.Returns:A `Flowable` emitting [`Event`](https://google.github.io/adk-docs/api-reference/java/com/google/adk/events/Event.html "class in com.google.adk.events") objects generated by the agent.

- ### runAsync [![Link icon](https://google.github.io/adk-docs/api-reference/java/resource-files/link.svg)](https://google.github.io/adk-docs/api-reference/java/com/google/adk/runner/Runner.html\#runAsync(com.google.adk.sessions.Session,com.google.genai.types.Content,com.google.adk.agents.RunConfig))





publicio.reactivex.rxjava3.core.Flowable< [Event](https://google.github.io/adk-docs/api-reference/java/com/google/adk/events/Event.html "class in com.google.adk.events") >runAsync( [Session](https://google.github.io/adk-docs/api-reference/java/com/google/adk/sessions/Session.html "class in com.google.adk.sessions") session,
com.google.genai.types.Content newMessage,
[RunConfig](https://google.github.io/adk-docs/api-reference/java/com/google/adk/agents/RunConfig.html "class in com.google.adk.agents") runConfig)



Runs the agent in the standard mode using a provided Session object.

Parameters:`session` \- The session to run the agent in.`newMessage` \- The new message from the user to process.`runConfig` \- Configuration for the agent run.Returns:A Flowable stream of [`Event`](https://google.github.io/adk-docs/api-reference/java/com/google/adk/events/Event.html "class in com.google.adk.events") objects generated by the agent during execution.

- ### runLive [![Link icon](https://google.github.io/adk-docs/api-reference/java/resource-files/link.svg)](https://google.github.io/adk-docs/api-reference/java/com/google/adk/runner/Runner.html\#runLive(com.google.adk.sessions.Session,com.google.adk.agents.LiveRequestQueue,com.google.adk.agents.RunConfig))





publicio.reactivex.rxjava3.core.Flowable< [Event](https://google.github.io/adk-docs/api-reference/java/com/google/adk/events/Event.html "class in com.google.adk.events") >runLive( [Session](https://google.github.io/adk-docs/api-reference/java/com/google/adk/sessions/Session.html "class in com.google.adk.sessions") session,
[LiveRequestQueue](https://google.github.io/adk-docs/api-reference/java/com/google/adk/agents/LiveRequestQueue.html "class in com.google.adk.agents") liveRequestQueue,
[RunConfig](https://google.github.io/adk-docs/api-reference/java/com/google/adk/agents/RunConfig.html "class in com.google.adk.agents") runConfig)



Runs the agent in live mode, appending generated events to the session.

Returns:stream of events from the agent.

- ### runLive [![Link icon](https://google.github.io/adk-docs/api-reference/java/resource-files/link.svg)](https://google.github.io/adk-docs/api-reference/java/com/google/adk/runner/Runner.html\#runLive(java.lang.String,java.lang.String,com.google.adk.agents.LiveRequestQueue,com.google.adk.agents.RunConfig))





publicio.reactivex.rxjava3.core.Flowable< [Event](https://google.github.io/adk-docs/api-reference/java/com/google/adk/events/Event.html "class in com.google.adk.events") >runLive( [String](https://docs.oracle.com/en/java/javase/17/docs/api/java.base/java/lang/String.html "class or interface in java.lang") userId,
[String](https://docs.oracle.com/en/java/javase/17/docs/api/java.base/java/lang/String.html "class or interface in java.lang") sessionId,
[LiveRequestQueue](https://google.github.io/adk-docs/api-reference/java/com/google/adk/agents/LiveRequestQueue.html "class in com.google.adk.agents") liveRequestQueue,
[RunConfig](https://google.github.io/adk-docs/api-reference/java/com/google/adk/agents/RunConfig.html "class in com.google.adk.agents") runConfig)



Retrieves the session and runs the agent in live mode.

Returns:stream of events from the agent.Throws:`IllegalArgumentException` \- if the session is not found.

- ### runWithSessionId [![Link icon](https://google.github.io/adk-docs/api-reference/java/resource-files/link.svg)](https://google.github.io/adk-docs/api-reference/java/com/google/adk/runner/Runner.html\#runWithSessionId(java.lang.String,com.google.genai.types.Content,com.google.adk.agents.RunConfig))





publicio.reactivex.rxjava3.core.Flowable< [Event](https://google.github.io/adk-docs/api-reference/java/com/google/adk/events/Event.html "class in com.google.adk.events") >runWithSessionId( [String](https://docs.oracle.com/en/java/javase/17/docs/api/java.base/java/lang/String.html "class or interface in java.lang") sessionId,
com.google.genai.types.Content newMessage,
[RunConfig](https://google.github.io/adk-docs/api-reference/java/com/google/adk/agents/RunConfig.html "class in com.google.adk.agents") runConfig)



Runs the agent asynchronously with a default user ID.

Returns:stream of generated events.

## ADK Exception Hierarchies
Package Hierarchies:

- [All Packages](https://google.github.io/adk-docs/api-reference/java/overview-tree.html)

## Class Hierarchy

- java.lang. [Object](https://docs.oracle.com/en/java/javase/17/docs/api/java.base/java/lang/Object.html "class or interface in java.lang")
  - java.lang. [Throwable](https://docs.oracle.com/en/java/javase/17/docs/api/java.base/java/lang/Throwable.html "class or interface in java.lang") (implements java.io. [Serializable](https://docs.oracle.com/en/java/javase/17/docs/api/java.base/java/io/Serializable.html "class or interface in java.io"))

    - java.lang. [Exception](https://docs.oracle.com/en/java/javase/17/docs/api/java.base/java/lang/Exception.html "class or interface in java.lang")
      - com.google.adk.exceptions. [LlmCallsLimitExceededException](https://google.github.io/adk-docs/api-reference/java/com/google/adk/exceptions/LlmCallsLimitExceededException.html "class in com.google.adk.exceptions")

## Instruction Provider
[java.lang.Object](https://docs.oracle.com/en/java/javase/17/docs/api/java.base/java/lang/Object.html "class or interface in java.lang")

[java.lang.Record](https://docs.oracle.com/en/java/javase/17/docs/api/java.base/java/lang/Record.html "class or interface in java.lang")

com.google.adk.agents.Instruction.Provider

All Implemented Interfaces:`Instruction`Enclosing interface:`Instruction`

* * *

public static record Instruction.Provider( [Function](https://docs.oracle.com/en/java/javase/17/docs/api/java.base/java/util/function/Function.html "class or interface in java.util.function") < [ReadonlyContext](https://google.github.io/adk-docs/api-reference/java/com/google/adk/agents/ReadonlyContext.html "class in com.google.adk.agents"), io.reactivex.rxjava3.core.Single< [String](https://docs.oracle.com/en/java/javase/17/docs/api/java.base/java/lang/String.html "class or interface in java.lang") >> getInstruction)
extends [Record](https://docs.oracle.com/en/java/javase/17/docs/api/java.base/java/lang/Record.html "class or interface in java.lang")
implements [Instruction](https://google.github.io/adk-docs/api-reference/java/com/google/adk/agents/Instruction.html "interface in com.google.adk.agents")

Returns an instruction dynamically constructed from the given context.

- ## Nested Class Summary [![Link icon](https://google.github.io/adk-docs/api-reference/java/resource-files/link.svg)](https://google.github.io/adk-docs/api-reference/java/com/google/adk/agents/Instruction.Provider.html\#nested-class-summary)





### Nested classes/interfaces inherited from interface com.google.adk.agents. [Instruction](https://google.github.io/adk-docs/api-reference/java/com/google/adk/agents/Instruction.html "interface in com.google.adk.agents") [![Link icon](https://google.github.io/adk-docs/api-reference/java/resource-files/link.svg)](https://google.github.io/adk-docs/api-reference/java/com/google/adk/agents/Instruction.Provider.html\#nested-classes-inherited-from-class-com.google.adk.agents.Instruction)

`Instruction.Provider, Instruction.Static`

- ## Constructor Summary [![Link icon](https://google.github.io/adk-docs/api-reference/java/resource-files/link.svg)](https://google.github.io/adk-docs/api-reference/java/com/google/adk/agents/Instruction.Provider.html\#constructor-summary)



Constructors





Constructor



Description



`Provider(Function<ReadonlyContext, io.reactivex.rxjava3.core.Single<String>> getInstruction)`





Creates an instance of a `Provider` record class.

- ## Method Summary [![Link icon](https://google.github.io/adk-docs/api-reference/java/resource-files/link.svg)](https://google.github.io/adk-docs/api-reference/java/com/google/adk/agents/Instruction.Provider.html\#method-summary)





All MethodsInstance MethodsConcrete Methods







Modifier and Type



Method



Description



`final boolean`



`equals(Object o)`





Indicates whether some other object is "equal to" this one.





`Function<ReadonlyContext, io.reactivex.rxjava3.core.Single<String>>`



`getInstruction()`





Returns the value of the `getInstruction` record component.





`final int`



`hashCode()`





Returns a hash code value for this object.





`final String`



`toString()`





Returns a string representation of this record class.













### Methods inherited from class java.lang. [Object](https://docs.oracle.com/en/java/javase/17/docs/api/java.base/java/lang/Object.html "class or interface in java.lang") [![Link icon](https://google.github.io/adk-docs/api-reference/java/resource-files/link.svg)](https://google.github.io/adk-docs/api-reference/java/com/google/adk/agents/Instruction.Provider.html\#methods-inherited-from-class-java.lang.Object)

`clone, finalize, getClass, notify, notifyAll, wait, wait, wait`


- ## Constructor Details [![Link icon](https://google.github.io/adk-docs/api-reference/java/resource-files/link.svg)](https://google.github.io/adk-docs/api-reference/java/com/google/adk/agents/Instruction.Provider.html\#constructor-detail)



- ### Provider [![Link icon](https://google.github.io/adk-docs/api-reference/java/resource-files/link.svg)](https://google.github.io/adk-docs/api-reference/java/com/google/adk/agents/Instruction.Provider.html\#%3Cinit%3E(java.util.function.Function))





publicProvider( [Function](https://docs.oracle.com/en/java/javase/17/docs/api/java.base/java/util/function/Function.html "class or interface in java.util.function") < [ReadonlyContext](https://google.github.io/adk-docs/api-reference/java/com/google/adk/agents/ReadonlyContext.html "class in com.google.adk.agents"), io.reactivex.rxjava3.core.Single< [String](https://docs.oracle.com/en/java/javase/17/docs/api/java.base/java/lang/String.html "class or interface in java.lang") >> getInstruction)



Creates an instance of a `Provider` record class.

Parameters:`getInstruction` \- the value for the `getInstruction` record component


- ## Method Details [![Link icon](https://google.github.io/adk-docs/api-reference/java/resource-files/link.svg)](https://google.github.io/adk-docs/api-reference/java/com/google/adk/agents/Instruction.Provider.html\#method-detail)



- ### toString [![Link icon](https://google.github.io/adk-docs/api-reference/java/resource-files/link.svg)](https://google.github.io/adk-docs/api-reference/java/com/google/adk/agents/Instruction.Provider.html\#toString())





public final[String](https://docs.oracle.com/en/java/javase/17/docs/api/java.base/java/lang/String.html "class or interface in java.lang")toString()



Returns a string representation of this record class. The representation contains the name of the class, followed by the name and value of each of the record components.

Specified by:`toString` in class `Record`Returns:a string representation of this object

- ### hashCode [![Link icon](https://google.github.io/adk-docs/api-reference/java/resource-files/link.svg)](https://google.github.io/adk-docs/api-reference/java/com/google/adk/agents/Instruction.Provider.html\#hashCode())





public finalinthashCode()



Returns a hash code value for this object. The value is derived from the hash code of each of the record components.

Specified by:`hashCode` in class `Record`Returns:a hash code value for this object

- ### equals [![Link icon](https://google.github.io/adk-docs/api-reference/java/resource-files/link.svg)](https://google.github.io/adk-docs/api-reference/java/com/google/adk/agents/Instruction.Provider.html\#equals(java.lang.Object))





public finalbooleanequals( [Object](https://docs.oracle.com/en/java/javase/17/docs/api/java.base/java/lang/Object.html "class or interface in java.lang") o)



Indicates whether some other object is "equal to" this one. The objects are equal if the other object is of the same class and if all the record components are equal. All components in this record class are compared with [`Objects::equals(Object,Object)`](https://docs.oracle.com/en/java/javase/17/docs/api/java.base/java/util/Objects.html#equals(java.lang.Object,java.lang.Object) "class or interface in java.util").

Specified by:`equals` in class `Record`Parameters:`o` \- the object with which to compareReturns:`true` if this object is the same as the `o` argument; `false` otherwise.

- ### getInstruction [![Link icon](https://google.github.io/adk-docs/api-reference/java/resource-files/link.svg)](https://google.github.io/adk-docs/api-reference/java/com/google/adk/agents/Instruction.Provider.html\#getInstruction())





public[Function](https://docs.oracle.com/en/java/javase/17/docs/api/java.base/java/util/function/Function.html "class or interface in java.util.function") < [ReadonlyContext](https://google.github.io/adk-docs/api-reference/java/com/google/adk/agents/ReadonlyContext.html "class in com.google.adk.agents"), io.reactivex.rxjava3.core.Single< [String](https://docs.oracle.com/en/java/javase/17/docs/api/java.base/java/lang/String.html "class or interface in java.lang") >>getInstruction()



Returns the value of the `getInstruction` record component.

Returns:the value of the `getInstruction` record component

## Agent Execution Flows
* * *

package com.google.adk.flows

- Related Packages





Package



Description



[com.google.adk](https://google.github.io/adk-docs/api-reference/java/com/google/adk/package-summary.html)







[com.google.adk.flows.llmflows](https://google.github.io/adk-docs/api-reference/java/com/google/adk/flows/llmflows/package-summary.html)

- Interfaces





Class



Description



[BaseFlow](https://google.github.io/adk-docs/api-reference/java/com/google/adk/flows/BaseFlow.html "interface in com.google.adk.flows")





Interface for the execution flows to run a group of agents.

## Packages Using com.google.adk.artifacts
Packages that use [com.google.adk.artifacts](https://google.github.io/adk-docs/api-reference/java/com/google/adk/artifacts/package-summary.html)

Package

Description

[com.google.adk.agents](https://google.github.io/adk-docs/api-reference/java/com/google/adk/artifacts/package-use.html#com.google.adk.agents)

[com.google.adk.artifacts](https://google.github.io/adk-docs/api-reference/java/com/google/adk/artifacts/package-use.html#com.google.adk.artifacts)

[com.google.adk.runner](https://google.github.io/adk-docs/api-reference/java/com/google/adk/artifacts/package-use.html#com.google.adk.runner)

[com.google.adk.web](https://google.github.io/adk-docs/api-reference/java/com/google/adk/artifacts/package-use.html#com.google.adk.web)

- Classes in [com.google.adk.artifacts](https://google.github.io/adk-docs/api-reference/java/com/google/adk/artifacts/package-summary.html) used by [com.google.adk.agents](https://google.github.io/adk-docs/api-reference/java/com/google/adk/agents/package-summary.html)





Class



Description



[BaseArtifactService](https://google.github.io/adk-docs/api-reference/java/com/google/adk/artifacts/class-use/BaseArtifactService.html#com.google.adk.agents)





Base interface for artifact services.

- Classes in [com.google.adk.artifacts](https://google.github.io/adk-docs/api-reference/java/com/google/adk/artifacts/package-summary.html) used by [com.google.adk.artifacts](https://google.github.io/adk-docs/api-reference/java/com/google/adk/artifacts/package-summary.html)





Class



Description



[BaseArtifactService](https://google.github.io/adk-docs/api-reference/java/com/google/adk/artifacts/class-use/BaseArtifactService.html#com.google.adk.artifacts)





Base interface for artifact services.





[ListArtifactsResponse](https://google.github.io/adk-docs/api-reference/java/com/google/adk/artifacts/class-use/ListArtifactsResponse.html#com.google.adk.artifacts)





Response for listing artifacts.





[ListArtifactsResponse.Builder](https://google.github.io/adk-docs/api-reference/java/com/google/adk/artifacts/class-use/ListArtifactsResponse.Builder.html#com.google.adk.artifacts)





Builder for [`ListArtifactsResponse`](https://google.github.io/adk-docs/api-reference/java/com/google/adk/artifacts/ListArtifactsResponse.html "class in com.google.adk.artifacts").





[ListArtifactVersionsResponse](https://google.github.io/adk-docs/api-reference/java/com/google/adk/artifacts/class-use/ListArtifactVersionsResponse.html#com.google.adk.artifacts)





Response for listing artifact versions.





[ListArtifactVersionsResponse.Builder](https://google.github.io/adk-docs/api-reference/java/com/google/adk/artifacts/class-use/ListArtifactVersionsResponse.Builder.html#com.google.adk.artifacts)





Builder for [`ListArtifactVersionsResponse`](https://google.github.io/adk-docs/api-reference/java/com/google/adk/artifacts/ListArtifactVersionsResponse.html "class in com.google.adk.artifacts").

- Classes in [com.google.adk.artifacts](https://google.github.io/adk-docs/api-reference/java/com/google/adk/artifacts/package-summary.html) used by [com.google.adk.runner](https://google.github.io/adk-docs/api-reference/java/com/google/adk/runner/package-summary.html)





Class



Description



[BaseArtifactService](https://google.github.io/adk-docs/api-reference/java/com/google/adk/artifacts/class-use/BaseArtifactService.html#com.google.adk.runner)





Base interface for artifact services.

- Classes in [com.google.adk.artifacts](https://google.github.io/adk-docs/api-reference/java/com/google/adk/artifacts/package-summary.html) used by [com.google.adk.web](https://google.github.io/adk-docs/api-reference/java/com/google/adk/web/package-summary.html)





Class



Description



[BaseArtifactService](https://google.github.io/adk-docs/api-reference/java/com/google/adk/artifacts/class-use/BaseArtifactService.html#com.google.adk.web)





Base interface for artifact services.

## Gemini.Builder Usage
Packages that use [Gemini.Builder](https://google.github.io/adk-docs/api-reference/java/com/google/adk/models/Gemini.Builder.html "class in com.google.adk.models")

Package

Description

[com.google.adk.models](https://google.github.io/adk-docs/api-reference/java/com/google/adk/models/class-use/Gemini.Builder.html#com.google.adk.models)

- ## Uses of [Gemini.Builder](https://google.github.io/adk-docs/api-reference/java/com/google/adk/models/Gemini.Builder.html "class in com.google.adk.models") in [com.google.adk.models](https://google.github.io/adk-docs/api-reference/java/com/google/adk/models/package-summary.html) [![Link icon](https://google.github.io/adk-docs/api-reference/java/resource-files/link.svg)](https://google.github.io/adk-docs/api-reference/java/com/google/adk/models/class-use/Gemini.Builder.html\#com.google.adk.models)



Methods in [com.google.adk.models](https://google.github.io/adk-docs/api-reference/java/com/google/adk/models/package-summary.html) that return [Gemini.Builder](https://google.github.io/adk-docs/api-reference/java/com/google/adk/models/Gemini.Builder.html "class in com.google.adk.models")





Modifier and Type



Method



Description



`Gemini.Builder`



Gemini.Builder.`apiClient(com.google.genai.Client apiClient)`





Sets the explicit `Client` instance for making API calls.





`Gemini.Builder`



Gemini.Builder.`apiKey(String apiKey)`





Sets the Google Gemini API key.





`static Gemini.Builder`



Gemini.`builder()`





Returns a new Builder instance for constructing Gemini objects.





`Gemini.Builder`



Gemini.Builder.`modelName(String modelName)`





Sets the name of the Gemini model to use.





`Gemini.Builder`



Gemini.Builder.`vertexCredentials(VertexCredentials vertexCredentials)`





Sets the Vertex AI credentials.

## DefaultMcpTransportBuilder Overview
No usage of com.google.adk.tools.mcp.DefaultMcpTransportBuilder

## LlmAgent.Builder Overview
Packages that use [LlmAgent.Builder](https://google.github.io/adk-docs/api-reference/java/com/google/adk/agents/LlmAgent.Builder.html "class in com.google.adk.agents")

Package

Description

[com.google.adk.agents](https://google.github.io/adk-docs/api-reference/java/com/google/adk/agents/class-use/LlmAgent.Builder.html#com.google.adk.agents)

- ## Uses of [LlmAgent.Builder](https://google.github.io/adk-docs/api-reference/java/com/google/adk/agents/LlmAgent.Builder.html "class in com.google.adk.agents") in [com.google.adk.agents](https://google.github.io/adk-docs/api-reference/java/com/google/adk/agents/package-summary.html) [![Link icon](https://google.github.io/adk-docs/api-reference/java/resource-files/link.svg)](https://google.github.io/adk-docs/api-reference/java/com/google/adk/agents/class-use/LlmAgent.Builder.html\#com.google.adk.agents)



Methods in [com.google.adk.agents](https://google.github.io/adk-docs/api-reference/java/com/google/adk/agents/package-summary.html) that return [LlmAgent.Builder](https://google.github.io/adk-docs/api-reference/java/com/google/adk/agents/LlmAgent.Builder.html "class in com.google.adk.agents")





Modifier and Type



Method



Description



`LlmAgent.Builder`



LlmAgent.Builder.`afterAgentCallback(Callbacks.AfterAgentCallback afterAgentCallback)`







`LlmAgent.Builder`



LlmAgent.Builder.`afterAgentCallback(List<com.google.adk.agents.Callbacks.AfterAgentCallbackBase> afterAgentCallback)`







`LlmAgent.Builder`



LlmAgent.Builder.`afterAgentCallbackSync(Callbacks.AfterAgentCallbackSync afterAgentCallbackSync)`







`LlmAgent.Builder`



LlmAgent.Builder.`afterModelCallback(Callbacks.AfterModelCallback afterModelCallback)`







`LlmAgent.Builder`



LlmAgent.Builder.`afterModelCallback(List<com.google.adk.agents.Callbacks.AfterModelCallbackBase> afterModelCallback)`







`LlmAgent.Builder`



LlmAgent.Builder.`afterModelCallbackSync(Callbacks.AfterModelCallbackSync afterModelCallbackSync)`







`LlmAgent.Builder`



LlmAgent.Builder.`afterToolCallback(Callbacks.AfterToolCallback afterToolCallback)`







`LlmAgent.Builder`



LlmAgent.Builder.`afterToolCallback(List<com.google.adk.agents.Callbacks.AfterToolCallbackBase> afterToolCallbacks)`







`LlmAgent.Builder`



LlmAgent.Builder.`afterToolCallbackSync(Callbacks.AfterToolCallbackSync afterToolCallbackSync)`







`LlmAgent.Builder`



LlmAgent.Builder.`beforeAgentCallback(Callbacks.BeforeAgentCallback beforeAgentCallback)`







`LlmAgent.Builder`



LlmAgent.Builder.`beforeAgentCallback(List<com.google.adk.agents.Callbacks.BeforeAgentCallbackBase> beforeAgentCallback)`







`LlmAgent.Builder`



LlmAgent.Builder.`beforeAgentCallbackSync(Callbacks.BeforeAgentCallbackSync beforeAgentCallbackSync)`







`LlmAgent.Builder`



LlmAgent.Builder.`beforeModelCallback(Callbacks.BeforeModelCallback beforeModelCallback)`







`LlmAgent.Builder`



LlmAgent.Builder.`beforeModelCallback(List<com.google.adk.agents.Callbacks.BeforeModelCallbackBase> beforeModelCallback)`







`LlmAgent.Builder`



LlmAgent.Builder.`beforeModelCallbackSync(Callbacks.BeforeModelCallbackSync beforeModelCallbackSync)`







`LlmAgent.Builder`



LlmAgent.Builder.`beforeToolCallback(Callbacks.BeforeToolCallback beforeToolCallback)`







`LlmAgent.Builder`



LlmAgent.Builder.`beforeToolCallback(List<com.google.adk.agents.Callbacks.BeforeToolCallbackBase> beforeToolCallbacks)`







`LlmAgent.Builder`



LlmAgent.Builder.`beforeToolCallbackSync(Callbacks.BeforeToolCallbackSync beforeToolCallbackSync)`







`static LlmAgent.Builder`



LlmAgent.`builder()`





Returns a [`LlmAgent.Builder`](https://google.github.io/adk-docs/api-reference/java/com/google/adk/agents/LlmAgent.Builder.html "class in com.google.adk.agents") for [`LlmAgent`](https://google.github.io/adk-docs/api-reference/java/com/google/adk/agents/LlmAgent.html "class in com.google.adk.agents").





`LlmAgent.Builder`



LlmAgent.Builder.`description(String description)`







`LlmAgent.Builder`



LlmAgent.Builder.`disallowTransferToParent(boolean disallowTransferToParent)`







`LlmAgent.Builder`



LlmAgent.Builder.`disallowTransferToPeers(boolean disallowTransferToPeers)`







`LlmAgent.Builder`



LlmAgent.Builder.`exampleProvider(BaseExampleProvider exampleProvider)`







`LlmAgent.Builder`



LlmAgent.Builder.`exampleProvider(Example... examples)`







`LlmAgent.Builder`



LlmAgent.Builder.`exampleProvider(List<Example> examples)`







`LlmAgent.Builder`



LlmAgent.Builder.`executor(Executor executor)`







`LlmAgent.Builder`



LlmAgent.Builder.`generateContentConfig(com.google.genai.types.GenerateContentConfig generateContentConfig)`







`LlmAgent.Builder`



LlmAgent.Builder.`globalInstruction(Instruction globalInstruction)`







`LlmAgent.Builder`



LlmAgent.Builder.`globalInstruction(String globalInstruction)`







`LlmAgent.Builder`



LlmAgent.Builder.`includeContents(LlmAgent.IncludeContents includeContents)`







`LlmAgent.Builder`



LlmAgent.Builder.`inputSchema(com.google.genai.types.Schema inputSchema)`







`LlmAgent.Builder`



LlmAgent.Builder.`instruction(Instruction instruction)`







`LlmAgent.Builder`



LlmAgent.Builder.`instruction(String instruction)`







`LlmAgent.Builder`



LlmAgent.Builder.`maxSteps(int maxSteps)`







`LlmAgent.Builder`



LlmAgent.Builder.`model(BaseLlm model)`







`LlmAgent.Builder`



LlmAgent.Builder.`model(String model)`







`LlmAgent.Builder`



LlmAgent.Builder.`name(String name)`







`LlmAgent.Builder`



LlmAgent.Builder.`outputKey(String outputKey)`







`LlmAgent.Builder`



LlmAgent.Builder.`outputSchema(com.google.genai.types.Schema outputSchema)`







`LlmAgent.Builder`



LlmAgent.Builder.`planning(boolean planning)`







`LlmAgent.Builder`



LlmAgent.Builder.`subAgents(BaseAgent... subAgents)`







`LlmAgent.Builder`



LlmAgent.Builder.`subAgents(List<? extends BaseAgent> subAgents)`







`LlmAgent.Builder`



LlmAgent.Builder.`tools(Object... tools)`







`LlmAgent.Builder`



LlmAgent.Builder.`tools(List<?> tools)`









Constructors in [com.google.adk.agents](https://google.github.io/adk-docs/api-reference/java/com/google/adk/agents/package-summary.html) with parameters of type [LlmAgent.Builder](https://google.github.io/adk-docs/api-reference/java/com/google/adk/agents/LlmAgent.Builder.html "class in com.google.adk.agents")





Modifier



Constructor



Description



`protected`



`LlmAgent(LlmAgent.Builder builder)`

## Application Integration Toolset
[java.lang.Object](https://docs.oracle.com/en/java/javase/17/docs/api/java.base/java/lang/Object.html "class or interface in java.lang")

com.google.adk.tools.applicationintegrationtoolset.ApplicationIntegrationToolset

All Implemented Interfaces:`BaseToolset`, `AutoCloseable`

* * *

public class ApplicationIntegrationToolsetextends [Object](https://docs.oracle.com/en/java/javase/17/docs/api/java.base/java/lang/Object.html "class or interface in java.lang")
implements [BaseToolset](https://google.github.io/adk-docs/api-reference/java/com/google/adk/tools/BaseToolset.html "interface in com.google.adk.tools")

Application Integration Toolset

- ## Field Summary [![Link icon](https://google.github.io/adk-docs/api-reference/java/resource-files/link.svg)](https://google.github.io/adk-docs/api-reference/java/com/google/adk/tools/applicationintegrationtoolset/ApplicationIntegrationToolset.html\#field-summary)



Fields





Modifier and Type



Field



Description



`static final com.fasterxml.jackson.databind.ObjectMapper`



`OBJECT_MAPPER`

- ## Constructor Summary [![Link icon](https://google.github.io/adk-docs/api-reference/java/resource-files/link.svg)](https://google.github.io/adk-docs/api-reference/java/com/google/adk/tools/applicationintegrationtoolset/ApplicationIntegrationToolset.html\#constructor-summary)



Constructors





Constructor



Description



`ApplicationIntegrationToolset(String project,
String location,
String integration,
List<String> triggers,
String connection,
Map<String, List<String>> entityOperations,
List<String> actions,
String serviceAccountJson,
String toolNamePrefix,
String toolInstructions)`





ApplicationIntegrationToolset generates tools from a given Application Integration resource.

- ## Method Summary [![Link icon](https://google.github.io/adk-docs/api-reference/java/resource-files/link.svg)](https://google.github.io/adk-docs/api-reference/java/com/google/adk/tools/applicationintegrationtoolset/ApplicationIntegrationToolset.html\#method-summary)





All MethodsInstance MethodsConcrete Methods







Modifier and Type



Method



Description



`void`



`close()`





Performs cleanup and releases resources held by the toolset.





`io.reactivex.rxjava3.core.Flowable<BaseTool>`



`getTools(@Nullable ReadonlyContext readonlyContext)`





Return all tools in the toolset based on the provided context.













### Methods inherited from class java.lang. [Object](https://docs.oracle.com/en/java/javase/17/docs/api/java.base/java/lang/Object.html "class or interface in java.lang") [![Link icon](https://google.github.io/adk-docs/api-reference/java/resource-files/link.svg)](https://google.github.io/adk-docs/api-reference/java/com/google/adk/tools/applicationintegrationtoolset/ApplicationIntegrationToolset.html\#methods-inherited-from-class-java.lang.Object)

`clone, equals, finalize, getClass, hashCode, notify, notifyAll, toString, wait, wait, wait`





### Methods inherited from interface com.google.adk.tools. [BaseToolset](https://google.github.io/adk-docs/api-reference/java/com/google/adk/tools/BaseToolset.html "interface in com.google.adk.tools") [![Link icon](https://google.github.io/adk-docs/api-reference/java/resource-files/link.svg)](https://google.github.io/adk-docs/api-reference/java/com/google/adk/tools/applicationintegrationtoolset/ApplicationIntegrationToolset.html\#methods-inherited-from-class-com.google.adk.tools.BaseToolset)

`isToolSelected`


- ## Field Details [![Link icon](https://google.github.io/adk-docs/api-reference/java/resource-files/link.svg)](https://google.github.io/adk-docs/api-reference/java/com/google/adk/tools/applicationintegrationtoolset/ApplicationIntegrationToolset.html\#field-detail)



- ### OBJECT\_MAPPER [![Link icon](https://google.github.io/adk-docs/api-reference/java/resource-files/link.svg)](https://google.github.io/adk-docs/api-reference/java/com/google/adk/tools/applicationintegrationtoolset/ApplicationIntegrationToolset.html\#OBJECT_MAPPER)





public static finalcom.fasterxml.jackson.databind.ObjectMapperOBJECT\_MAPPER


- ## Constructor Details [![Link icon](https://google.github.io/adk-docs/api-reference/java/resource-files/link.svg)](https://google.github.io/adk-docs/api-reference/java/com/google/adk/tools/applicationintegrationtoolset/ApplicationIntegrationToolset.html\#constructor-detail)



- ### ApplicationIntegrationToolset [![Link icon](https://google.github.io/adk-docs/api-reference/java/resource-files/link.svg)](https://google.github.io/adk-docs/api-reference/java/com/google/adk/tools/applicationintegrationtoolset/ApplicationIntegrationToolset.html\#%3Cinit%3E(java.lang.String,java.lang.String,java.lang.String,java.util.List,java.lang.String,java.util.Map,java.util.List,java.lang.String,java.lang.String,java.lang.String))





publicApplicationIntegrationToolset( [String](https://docs.oracle.com/en/java/javase/17/docs/api/java.base/java/lang/String.html "class or interface in java.lang") project,
[String](https://docs.oracle.com/en/java/javase/17/docs/api/java.base/java/lang/String.html "class or interface in java.lang") location,
[String](https://docs.oracle.com/en/java/javase/17/docs/api/java.base/java/lang/String.html "class or interface in java.lang") integration,
[List](https://docs.oracle.com/en/java/javase/17/docs/api/java.base/java/util/List.html "class or interface in java.util") < [String](https://docs.oracle.com/en/java/javase/17/docs/api/java.base/java/lang/String.html "class or interface in java.lang") > triggers,
[String](https://docs.oracle.com/en/java/javase/17/docs/api/java.base/java/lang/String.html "class or interface in java.lang") connection,
[Map](https://docs.oracle.com/en/java/javase/17/docs/api/java.base/java/util/Map.html "class or interface in java.util") < [String](https://docs.oracle.com/en/java/javase/17/docs/api/java.base/java/lang/String.html "class or interface in java.lang"), [List](https://docs.oracle.com/en/java/javase/17/docs/api/java.base/java/util/List.html "class or interface in java.util") < [String](https://docs.oracle.com/en/java/javase/17/docs/api/java.base/java/lang/String.html "class or interface in java.lang") >> entityOperations,
[List](https://docs.oracle.com/en/java/javase/17/docs/api/java.base/java/util/List.html "class or interface in java.util") < [String](https://docs.oracle.com/en/java/javase/17/docs/api/java.base/java/lang/String.html "class or interface in java.lang") > actions,
[String](https://docs.oracle.com/en/java/javase/17/docs/api/java.base/java/lang/String.html "class or interface in java.lang") serviceAccountJson,
[String](https://docs.oracle.com/en/java/javase/17/docs/api/java.base/java/lang/String.html "class or interface in java.lang") toolNamePrefix,
[String](https://docs.oracle.com/en/java/javase/17/docs/api/java.base/java/lang/String.html "class or interface in java.lang") toolInstructions)



ApplicationIntegrationToolset generates tools from a given Application Integration resource.



Example Usage:





integrationTool = new ApplicationIntegrationToolset( project="test-project",
location="us-central1", integration="test-integration",
triggers=ImmutableList.of("api\_trigger/test\_trigger", "api\_trigger/test\_trigger\_2",
serviceAccountJson="{....}"),connection=null,enitityOperations=null,actions=null,toolNamePrefix="test-integration-tool",toolInstructions="This
tool is used to get response from test-integration.");





connectionTool = new ApplicationIntegrationToolset( project="test-project",
location="us-central1", integration=null, triggers=null, connection="test-connection",
entityOperations=ImmutableMap.of("Entity1", ImmutableList.of("LIST", "GET", "UPDATE")),
"Entity2", ImmutableList.of()), actions=ImmutableList.of("ExecuteCustomQuery"),
serviceAccountJson="{....}", toolNamePrefix="test-tool", toolInstructions="This tool is used to
list, get and update issues in Jira.");



Parameters:`project` \- The GCP project ID.`location` \- The GCP location of integration.`integration` \- The integration name.`triggers` \- (Optional) The list of trigger ids in the integration.`connection` \- (Optional) The connection name.`entityOperations` \- (Optional) The entity operations.`actions` \- (Optional) The actions.`serviceAccountJson` \- (Optional) The service account configuration as a dictionary. Required
if not using default service credential. Used for fetching the Application Integration or
Integration Connector resource.`toolNamePrefix` \- (Optional) The tool name prefix.`toolInstructions` \- (Optional) The tool instructions.


- ## Method Details [![Link icon](https://google.github.io/adk-docs/api-reference/java/resource-files/link.svg)](https://google.github.io/adk-docs/api-reference/java/com/google/adk/tools/applicationintegrationtoolset/ApplicationIntegrationToolset.html\#method-detail)



- ### getTools [![Link icon](https://google.github.io/adk-docs/api-reference/java/resource-files/link.svg)](https://google.github.io/adk-docs/api-reference/java/com/google/adk/tools/applicationintegrationtoolset/ApplicationIntegrationToolset.html\#getTools(com.google.adk.agents.ReadonlyContext))





publicio.reactivex.rxjava3.core.Flowable< [BaseTool](https://google.github.io/adk-docs/api-reference/java/com/google/adk/tools/BaseTool.html "class in com.google.adk.tools") >getTools(@Nullable [ReadonlyContext](https://google.github.io/adk-docs/api-reference/java/com/google/adk/agents/ReadonlyContext.html "class in com.google.adk.agents") readonlyContext)



Description copied from interface: `BaseToolset`



Return all tools in the toolset based on the provided context.

Specified by:`getTools` in interface `BaseToolset`Parameters:`readonlyContext` \- Context used to filter tools available to the agent.Returns:A Single emitting a list of tools available under the specified context.

- ### close [![Link icon](https://google.github.io/adk-docs/api-reference/java/resource-files/link.svg)](https://google.github.io/adk-docs/api-reference/java/com/google/adk/tools/applicationintegrationtoolset/ApplicationIntegrationToolset.html\#close())





publicvoidclose()
throws [Exception](https://docs.oracle.com/en/java/javase/17/docs/api/java.base/java/lang/Exception.html "class or interface in java.lang")



Description copied from interface: `BaseToolset`



Performs cleanup and releases resources held by the toolset.



NOTE: This method is invoked, for example, at the end of an agent server's lifecycle or when
the toolset is no longer needed. Implementations should ensure that any open connections,
files, or other managed resources are properly released to prevent leaks.



Specified by:`close` in interface `AutoCloseable`Specified by:`close` in interface `BaseToolset`Throws:`Exception`

## SearchMemoryResponse Builder
[java.lang.Object](https://docs.oracle.com/en/java/javase/17/docs/api/java.base/java/lang/Object.html "class or interface in java.lang")

com.google.adk.memory.SearchMemoryResponse.Builder

Enclosing class:`SearchMemoryResponse`

* * *

public abstract static class SearchMemoryResponse.Builderextends [Object](https://docs.oracle.com/en/java/javase/17/docs/api/java.base/java/lang/Object.html "class or interface in java.lang")

Builder for [`SearchMemoryResponse`](https://google.github.io/adk-docs/api-reference/java/com/google/adk/memory/SearchMemoryResponse.html "class in com.google.adk.memory").

- ## Constructor Summary [![Link icon](https://google.github.io/adk-docs/api-reference/java/resource-files/link.svg)](https://google.github.io/adk-docs/api-reference/java/com/google/adk/memory/SearchMemoryResponse.Builder.html\#constructor-summary)



Constructors





Constructor



Description



`Builder()`

- ## Method Summary [![Link icon](https://google.github.io/adk-docs/api-reference/java/resource-files/link.svg)](https://google.github.io/adk-docs/api-reference/java/com/google/adk/memory/SearchMemoryResponse.Builder.html\#method-summary)





All MethodsInstance MethodsAbstract MethodsConcrete Methods







Modifier and Type



Method



Description



`abstract SearchMemoryResponse`



`build()`





Builds the immutable [`SearchMemoryResponse`](https://google.github.io/adk-docs/api-reference/java/com/google/adk/memory/SearchMemoryResponse.html "class in com.google.adk.memory") object.





`SearchMemoryResponse.Builder`



`setMemories(List<MemoryEntry> memories)`





Sets the list of memory entries using a list.













### Methods inherited from class java.lang. [Object](https://docs.oracle.com/en/java/javase/17/docs/api/java.base/java/lang/Object.html "class or interface in java.lang") [![Link icon](https://google.github.io/adk-docs/api-reference/java/resource-files/link.svg)](https://google.github.io/adk-docs/api-reference/java/com/google/adk/memory/SearchMemoryResponse.Builder.html\#methods-inherited-from-class-java.lang.Object)

`clone, equals, finalize, getClass, hashCode, notify, notifyAll, toString, wait, wait, wait`


- ## Constructor Details [![Link icon](https://google.github.io/adk-docs/api-reference/java/resource-files/link.svg)](https://google.github.io/adk-docs/api-reference/java/com/google/adk/memory/SearchMemoryResponse.Builder.html\#constructor-detail)



- ### Builder [![Link icon](https://google.github.io/adk-docs/api-reference/java/resource-files/link.svg)](https://google.github.io/adk-docs/api-reference/java/com/google/adk/memory/SearchMemoryResponse.Builder.html\#%3Cinit%3E())





publicBuilder()


- ## Method Details [![Link icon](https://google.github.io/adk-docs/api-reference/java/resource-files/link.svg)](https://google.github.io/adk-docs/api-reference/java/com/google/adk/memory/SearchMemoryResponse.Builder.html\#method-detail)



- ### setMemories [![Link icon](https://google.github.io/adk-docs/api-reference/java/resource-files/link.svg)](https://google.github.io/adk-docs/api-reference/java/com/google/adk/memory/SearchMemoryResponse.Builder.html\#setMemories(java.util.List))





public[SearchMemoryResponse.Builder](https://google.github.io/adk-docs/api-reference/java/com/google/adk/memory/SearchMemoryResponse.Builder.html "class in com.google.adk.memory")setMemories( [List](https://docs.oracle.com/en/java/javase/17/docs/api/java.base/java/util/List.html "class or interface in java.util") < [MemoryEntry](https://google.github.io/adk-docs/api-reference/java/com/google/adk/memory/MemoryEntry.html "class in com.google.adk.memory") > memories)



Sets the list of memory entries using a list.

- ### build [![Link icon](https://google.github.io/adk-docs/api-reference/java/resource-files/link.svg)](https://google.github.io/adk-docs/api-reference/java/com/google/adk/memory/SearchMemoryResponse.Builder.html\#build())





public abstract[SearchMemoryResponse](https://google.github.io/adk-docs/api-reference/java/com/google/adk/memory/SearchMemoryResponse.html "class in com.google.adk.memory")build()



Builds the immutable [`SearchMemoryResponse`](https://google.github.io/adk-docs/api-reference/java/com/google/adk/memory/SearchMemoryResponse.html "class in com.google.adk.memory") object.

## Request Processing Result
[java.lang.Object](https://docs.oracle.com/en/java/javase/17/docs/api/java.base/java/lang/Object.html "class or interface in java.lang")

com.google.adk.flows.llmflows.RequestProcessor.RequestProcessingResult

Enclosing interface:`RequestProcessor`

* * *

public abstract static class RequestProcessor.RequestProcessingResultextends [Object](https://docs.oracle.com/en/java/javase/17/docs/api/java.base/java/lang/Object.html "class or interface in java.lang")

- ## Constructor Summary [![Link icon](https://google.github.io/adk-docs/api-reference/java/resource-files/link.svg)](https://google.github.io/adk-docs/api-reference/java/com/google/adk/flows/llmflows/RequestProcessor.RequestProcessingResult.html\#constructor-summary)



Constructors





Constructor



Description



`RequestProcessingResult()`

- ## Method Summary [![Link icon](https://google.github.io/adk-docs/api-reference/java/resource-files/link.svg)](https://google.github.io/adk-docs/api-reference/java/com/google/adk/flows/llmflows/RequestProcessor.RequestProcessingResult.html\#method-summary)





All MethodsStatic MethodsInstance MethodsAbstract MethodsConcrete Methods







Modifier and Type



Method



Description



`static RequestProcessor.RequestProcessingResult`



`create(LlmRequest updatedRequest,
Iterable<Event> events)`







`abstract Iterable<Event>`



`events()`







`abstract LlmRequest`



`updatedRequest()`















### Methods inherited from class java.lang. [Object](https://docs.oracle.com/en/java/javase/17/docs/api/java.base/java/lang/Object.html "class or interface in java.lang") [![Link icon](https://google.github.io/adk-docs/api-reference/java/resource-files/link.svg)](https://google.github.io/adk-docs/api-reference/java/com/google/adk/flows/llmflows/RequestProcessor.RequestProcessingResult.html\#methods-inherited-from-class-java.lang.Object)

`clone, equals, finalize, getClass, hashCode, notify, notifyAll, toString, wait, wait, wait`


- ## Constructor Details [![Link icon](https://google.github.io/adk-docs/api-reference/java/resource-files/link.svg)](https://google.github.io/adk-docs/api-reference/java/com/google/adk/flows/llmflows/RequestProcessor.RequestProcessingResult.html\#constructor-detail)



- ### RequestProcessingResult [![Link icon](https://google.github.io/adk-docs/api-reference/java/resource-files/link.svg)](https://google.github.io/adk-docs/api-reference/java/com/google/adk/flows/llmflows/RequestProcessor.RequestProcessingResult.html\#%3Cinit%3E())





publicRequestProcessingResult()


- ## Method Details [![Link icon](https://google.github.io/adk-docs/api-reference/java/resource-files/link.svg)](https://google.github.io/adk-docs/api-reference/java/com/google/adk/flows/llmflows/RequestProcessor.RequestProcessingResult.html\#method-detail)



- ### updatedRequest [![Link icon](https://google.github.io/adk-docs/api-reference/java/resource-files/link.svg)](https://google.github.io/adk-docs/api-reference/java/com/google/adk/flows/llmflows/RequestProcessor.RequestProcessingResult.html\#updatedRequest())





public abstract[LlmRequest](https://google.github.io/adk-docs/api-reference/java/com/google/adk/models/LlmRequest.html "class in com.google.adk.models")updatedRequest()

- ### events [![Link icon](https://google.github.io/adk-docs/api-reference/java/resource-files/link.svg)](https://google.github.io/adk-docs/api-reference/java/com/google/adk/flows/llmflows/RequestProcessor.RequestProcessingResult.html\#events())





public abstract[Iterable](https://docs.oracle.com/en/java/javase/17/docs/api/java.base/java/lang/Iterable.html "class or interface in java.lang") < [Event](https://google.github.io/adk-docs/api-reference/java/com/google/adk/events/Event.html "class in com.google.adk.events") >events()

- ### create [![Link icon](https://google.github.io/adk-docs/api-reference/java/resource-files/link.svg)](https://google.github.io/adk-docs/api-reference/java/com/google/adk/flows/llmflows/RequestProcessor.RequestProcessingResult.html\#create(com.google.adk.models.LlmRequest,java.lang.Iterable))





public static[RequestProcessor.RequestProcessingResult](https://google.github.io/adk-docs/api-reference/java/com/google/adk/flows/llmflows/RequestProcessor.RequestProcessingResult.html "class in com.google.adk.flows.llmflows")create( [LlmRequest](https://google.github.io/adk-docs/api-reference/java/com/google/adk/models/LlmRequest.html "class in com.google.adk.models") updatedRequest,
[Iterable](https://docs.oracle.com/en/java/javase/17/docs/api/java.base/java/lang/Iterable.html "class or interface in java.lang") < [Event](https://google.github.io/adk-docs/api-reference/java/com/google/adk/events/Event.html "class in com.google.adk.events") > events)

## GCS Artifact Service
[java.lang.Object](https://docs.oracle.com/en/java/javase/17/docs/api/java.base/java/lang/Object.html "class or interface in java.lang")

com.google.adk.artifacts.GcsArtifactService

All Implemented Interfaces:`BaseArtifactService`

* * *

public final class GcsArtifactServiceextends [Object](https://docs.oracle.com/en/java/javase/17/docs/api/java.base/java/lang/Object.html "class or interface in java.lang")
implements [BaseArtifactService](https://google.github.io/adk-docs/api-reference/java/com/google/adk/artifacts/BaseArtifactService.html "interface in com.google.adk.artifacts")

An artifact service implementation using Google Cloud Storage (GCS).

- ## Constructor Summary [![Link icon](https://google.github.io/adk-docs/api-reference/java/resource-files/link.svg)](https://google.github.io/adk-docs/api-reference/java/com/google/adk/artifacts/GcsArtifactService.html\#constructor-summary)



Constructors





Constructor



Description



`GcsArtifactService(String bucketName,
com.google.cloud.storage.Storage storageClient)`





Initializes the GcsArtifactService.

- ## Method Summary [![Link icon](https://google.github.io/adk-docs/api-reference/java/resource-files/link.svg)](https://google.github.io/adk-docs/api-reference/java/com/google/adk/artifacts/GcsArtifactService.html\#method-summary)





All MethodsInstance MethodsConcrete Methods







Modifier and Type



Method



Description



`io.reactivex.rxjava3.core.Completable`



`deleteArtifact(String appName,
String userId,
String sessionId,
String filename)`





Deletes all versions of the specified artifact from GCS.





`io.reactivex.rxjava3.core.Single<ListArtifactsResponse>`



`listArtifactKeys(String appName,
String userId,
String sessionId)`





Lists artifact filenames for a user and session.





`io.reactivex.rxjava3.core.Single<com.google.common.collect.ImmutableList<Integer>>`



`listVersions(String appName,
String userId,
String sessionId,
String filename)`





Lists all available versions for a given artifact.





`io.reactivex.rxjava3.core.Maybe<com.google.genai.types.Part>`



`loadArtifact(String appName,
String userId,
String sessionId,
String filename,
Optional<Integer> version)`





Loads an artifact from GCS.





`io.reactivex.rxjava3.core.Single<Integer>`



`saveArtifact(String appName,
String userId,
String sessionId,
String filename,
com.google.genai.types.Part artifact)`





Saves an artifact to GCS and assigns a new version.













### Methods inherited from class java.lang. [Object](https://docs.oracle.com/en/java/javase/17/docs/api/java.base/java/lang/Object.html "class or interface in java.lang") [![Link icon](https://google.github.io/adk-docs/api-reference/java/resource-files/link.svg)](https://google.github.io/adk-docs/api-reference/java/com/google/adk/artifacts/GcsArtifactService.html\#methods-inherited-from-class-java.lang.Object)

`clone, equals, finalize, getClass, hashCode, notify, notifyAll, toString, wait, wait, wait`


- ## Constructor Details [![Link icon](https://google.github.io/adk-docs/api-reference/java/resource-files/link.svg)](https://google.github.io/adk-docs/api-reference/java/com/google/adk/artifacts/GcsArtifactService.html\#constructor-detail)



- ### GcsArtifactService [![Link icon](https://google.github.io/adk-docs/api-reference/java/resource-files/link.svg)](https://google.github.io/adk-docs/api-reference/java/com/google/adk/artifacts/GcsArtifactService.html\#%3Cinit%3E(java.lang.String,com.google.cloud.storage.Storage))





publicGcsArtifactService( [String](https://docs.oracle.com/en/java/javase/17/docs/api/java.base/java/lang/String.html "class or interface in java.lang") bucketName,
com.google.cloud.storage.Storage storageClient)



Initializes the GcsArtifactService.

Parameters:`bucketName` \- The name of the GCS bucket to use.`storageClient` \- The GCS storage client instance.


- ## Method Details [![Link icon](https://google.github.io/adk-docs/api-reference/java/resource-files/link.svg)](https://google.github.io/adk-docs/api-reference/java/com/google/adk/artifacts/GcsArtifactService.html\#method-detail)



- ### saveArtifact [![Link icon](https://google.github.io/adk-docs/api-reference/java/resource-files/link.svg)](https://google.github.io/adk-docs/api-reference/java/com/google/adk/artifacts/GcsArtifactService.html\#saveArtifact(java.lang.String,java.lang.String,java.lang.String,java.lang.String,com.google.genai.types.Part))





publicio.reactivex.rxjava3.core.Single< [Integer](https://docs.oracle.com/en/java/javase/17/docs/api/java.base/java/lang/Integer.html "class or interface in java.lang") >saveArtifact( [String](https://docs.oracle.com/en/java/javase/17/docs/api/java.base/java/lang/String.html "class or interface in java.lang") appName,
[String](https://docs.oracle.com/en/java/javase/17/docs/api/java.base/java/lang/String.html "class or interface in java.lang") userId,
[String](https://docs.oracle.com/en/java/javase/17/docs/api/java.base/java/lang/String.html "class or interface in java.lang") sessionId,
[String](https://docs.oracle.com/en/java/javase/17/docs/api/java.base/java/lang/String.html "class or interface in java.lang") filename,
com.google.genai.types.Part artifact)



Saves an artifact to GCS and assigns a new version.

Specified by:`saveArtifact` in interface `BaseArtifactService`Parameters:`appName` \- Application name.`userId` \- User ID.`sessionId` \- Session ID.`filename` \- Artifact filename.`artifact` \- Artifact content to save.Returns:Single with assigned version number.

- ### loadArtifact [![Link icon](https://google.github.io/adk-docs/api-reference/java/resource-files/link.svg)](https://google.github.io/adk-docs/api-reference/java/com/google/adk/artifacts/GcsArtifactService.html\#loadArtifact(java.lang.String,java.lang.String,java.lang.String,java.lang.String,java.util.Optional))





publicio.reactivex.rxjava3.core.Maybe<com.google.genai.types.Part>loadArtifact( [String](https://docs.oracle.com/en/java/javase/17/docs/api/java.base/java/lang/String.html "class or interface in java.lang") appName,
[String](https://docs.oracle.com/en/java/javase/17/docs/api/java.base/java/lang/String.html "class or interface in java.lang") userId,
[String](https://docs.oracle.com/en/java/javase/17/docs/api/java.base/java/lang/String.html "class or interface in java.lang") sessionId,
[String](https://docs.oracle.com/en/java/javase/17/docs/api/java.base/java/lang/String.html "class or interface in java.lang") filename,
[Optional](https://docs.oracle.com/en/java/javase/17/docs/api/java.base/java/util/Optional.html "class or interface in java.util") < [Integer](https://docs.oracle.com/en/java/javase/17/docs/api/java.base/java/lang/Integer.html "class or interface in java.lang") > version)



Loads an artifact from GCS.

Specified by:`loadArtifact` in interface `BaseArtifactService`Parameters:`appName` \- Application name.`userId` \- User ID.`sessionId` \- Session ID.`filename` \- Artifact filename.`version` \- Optional version to load. Loads latest if empty.Returns:Maybe with loaded artifact, or empty if not found.

- ### listArtifactKeys [![Link icon](https://google.github.io/adk-docs/api-reference/java/resource-files/link.svg)](https://google.github.io/adk-docs/api-reference/java/com/google/adk/artifacts/GcsArtifactService.html\#listArtifactKeys(java.lang.String,java.lang.String,java.lang.String))





publicio.reactivex.rxjava3.core.Single< [ListArtifactsResponse](https://google.github.io/adk-docs/api-reference/java/com/google/adk/artifacts/ListArtifactsResponse.html "class in com.google.adk.artifacts") >listArtifactKeys( [String](https://docs.oracle.com/en/java/javase/17/docs/api/java.base/java/lang/String.html "class or interface in java.lang") appName,
[String](https://docs.oracle.com/en/java/javase/17/docs/api/java.base/java/lang/String.html "class or interface in java.lang") userId,
[String](https://docs.oracle.com/en/java/javase/17/docs/api/java.base/java/lang/String.html "class or interface in java.lang") sessionId)



Lists artifact filenames for a user and session.

Specified by:`listArtifactKeys` in interface `BaseArtifactService`Parameters:`appName` \- Application name.`userId` \- User ID.`sessionId` \- Session ID.Returns:Single with sorted list of artifact filenames.

- ### deleteArtifact [![Link icon](https://google.github.io/adk-docs/api-reference/java/resource-files/link.svg)](https://google.github.io/adk-docs/api-reference/java/com/google/adk/artifacts/GcsArtifactService.html\#deleteArtifact(java.lang.String,java.lang.String,java.lang.String,java.lang.String))





publicio.reactivex.rxjava3.core.CompletabledeleteArtifact( [String](https://docs.oracle.com/en/java/javase/17/docs/api/java.base/java/lang/String.html "class or interface in java.lang") appName,
[String](https://docs.oracle.com/en/java/javase/17/docs/api/java.base/java/lang/String.html "class or interface in java.lang") userId,
[String](https://docs.oracle.com/en/java/javase/17/docs/api/java.base/java/lang/String.html "class or interface in java.lang") sessionId,
[String](https://docs.oracle.com/en/java/javase/17/docs/api/java.base/java/lang/String.html "class or interface in java.lang") filename)



Deletes all versions of the specified artifact from GCS.

Specified by:`deleteArtifact` in interface `BaseArtifactService`Parameters:`appName` \- Application name.`userId` \- User ID.`sessionId` \- Session ID.`filename` \- Artifact filename.Returns:Completable indicating operation completion.

- ### listVersions [![Link icon](https://google.github.io/adk-docs/api-reference/java/resource-files/link.svg)](https://google.github.io/adk-docs/api-reference/java/com/google/adk/artifacts/GcsArtifactService.html\#listVersions(java.lang.String,java.lang.String,java.lang.String,java.lang.String))





publicio.reactivex.rxjava3.core.Single<com.google.common.collect.ImmutableList< [Integer](https://docs.oracle.com/en/java/javase/17/docs/api/java.base/java/lang/Integer.html "class or interface in java.lang") >>listVersions( [String](https://docs.oracle.com/en/java/javase/17/docs/api/java.base/java/lang/String.html "class or interface in java.lang") appName,
[String](https://docs.oracle.com/en/java/javase/17/docs/api/java.base/java/lang/String.html "class or interface in java.lang") userId,
[String](https://docs.oracle.com/en/java/javase/17/docs/api/java.base/java/lang/String.html "class or interface in java.lang") sessionId,
[String](https://docs.oracle.com/en/java/javase/17/docs/api/java.base/java/lang/String.html "class or interface in java.lang") filename)



Lists all available versions for a given artifact.

Specified by:`listVersions` in interface `BaseArtifactService`Parameters:`appName` \- Application name.`userId` \- User ID.`sessionId` \- Session ID.`filename` \- Artifact filename.Returns:Single with sorted list of version numbers.

## Load Artifacts Tool
[java.lang.Object](https://docs.oracle.com/en/java/javase/17/docs/api/java.base/java/lang/Object.html "class or interface in java.lang")

[com.google.adk.tools.BaseTool](https://google.github.io/adk-docs/api-reference/java/com/google/adk/tools/BaseTool.html "class in com.google.adk.tools")

com.google.adk.tools.LoadArtifactsTool

* * *

public final class LoadArtifactsToolextends [BaseTool](https://google.github.io/adk-docs/api-reference/java/com/google/adk/tools/BaseTool.html "class in com.google.adk.tools")

A tool that loads artifacts and adds them to the session.



This tool informs the model about available artifacts and provides their content when
requested by the model through a function call.

- ## Constructor Summary [![Link icon](https://google.github.io/adk-docs/api-reference/java/resource-files/link.svg)](https://google.github.io/adk-docs/api-reference/java/com/google/adk/tools/LoadArtifactsTool.html\#constructor-summary)



Constructors





Constructor



Description



`LoadArtifactsTool()`

- ## Method Summary [![Link icon](https://google.github.io/adk-docs/api-reference/java/resource-files/link.svg)](https://google.github.io/adk-docs/api-reference/java/com/google/adk/tools/LoadArtifactsTool.html\#method-summary)





All MethodsInstance MethodsConcrete Methods







Modifier and Type



Method



Description



`io.reactivex.rxjava3.core.Completable`



`appendArtifactsToLlmRequest(LlmRequest.Builder llmRequestBuilder,
ToolContext toolContext)`







`Optional<com.google.genai.types.FunctionDeclaration>`



`declaration()`





Gets the `FunctionDeclaration` representation of this tool.





`io.reactivex.rxjava3.core.Completable`



`processLlmRequest(LlmRequest.Builder llmRequestBuilder,
ToolContext toolContext)`





Processes the outgoing [`LlmRequest.Builder`](https://google.github.io/adk-docs/api-reference/java/com/google/adk/models/LlmRequest.Builder.html "class in com.google.adk.models").





`io.reactivex.rxjava3.core.Single<Map<String,Object>>`



`runAsync(Map<String,Object> args,
ToolContext toolContext)`





Calls a tool.













### Methods inherited from class com.google.adk.tools. [BaseTool](https://google.github.io/adk-docs/api-reference/java/com/google/adk/tools/BaseTool.html "class in com.google.adk.tools") [![Link icon](https://google.github.io/adk-docs/api-reference/java/resource-files/link.svg)](https://google.github.io/adk-docs/api-reference/java/com/google/adk/tools/LoadArtifactsTool.html\#methods-inherited-from-class-com.google.adk.tools.BaseTool)

`description, longRunning, name`





### Methods inherited from class java.lang. [Object](https://docs.oracle.com/en/java/javase/17/docs/api/java.base/java/lang/Object.html "class or interface in java.lang") [![Link icon](https://google.github.io/adk-docs/api-reference/java/resource-files/link.svg)](https://google.github.io/adk-docs/api-reference/java/com/google/adk/tools/LoadArtifactsTool.html\#methods-inherited-from-class-java.lang.Object)

`clone, equals, finalize, getClass, hashCode, notify, notifyAll, toString, wait, wait, wait`


- ## Constructor Details [![Link icon](https://google.github.io/adk-docs/api-reference/java/resource-files/link.svg)](https://google.github.io/adk-docs/api-reference/java/com/google/adk/tools/LoadArtifactsTool.html\#constructor-detail)



- ### LoadArtifactsTool [![Link icon](https://google.github.io/adk-docs/api-reference/java/resource-files/link.svg)](https://google.github.io/adk-docs/api-reference/java/com/google/adk/tools/LoadArtifactsTool.html\#%3Cinit%3E())





publicLoadArtifactsTool()


- ## Method Details [![Link icon](https://google.github.io/adk-docs/api-reference/java/resource-files/link.svg)](https://google.github.io/adk-docs/api-reference/java/com/google/adk/tools/LoadArtifactsTool.html\#method-detail)



- ### declaration [![Link icon](https://google.github.io/adk-docs/api-reference/java/resource-files/link.svg)](https://google.github.io/adk-docs/api-reference/java/com/google/adk/tools/LoadArtifactsTool.html\#declaration())





public[Optional](https://docs.oracle.com/en/java/javase/17/docs/api/java.base/java/util/Optional.html "class or interface in java.util") <com.google.genai.types.FunctionDeclaration>declaration()



Description copied from class: `BaseTool`



Gets the `FunctionDeclaration` representation of this tool.

Overrides:`declaration` in class `BaseTool`

- ### runAsync [![Link icon](https://google.github.io/adk-docs/api-reference/java/resource-files/link.svg)](https://google.github.io/adk-docs/api-reference/java/com/google/adk/tools/LoadArtifactsTool.html\#runAsync(java.util.Map,com.google.adk.tools.ToolContext))





publicio.reactivex.rxjava3.core.Single< [Map](https://docs.oracle.com/en/java/javase/17/docs/api/java.base/java/util/Map.html "class or interface in java.util") < [String](https://docs.oracle.com/en/java/javase/17/docs/api/java.base/java/lang/String.html "class or interface in java.lang"), [Object](https://docs.oracle.com/en/java/javase/17/docs/api/java.base/java/lang/Object.html "class or interface in java.lang") >>runAsync( [Map](https://docs.oracle.com/en/java/javase/17/docs/api/java.base/java/util/Map.html "class or interface in java.util") < [String](https://docs.oracle.com/en/java/javase/17/docs/api/java.base/java/lang/String.html "class or interface in java.lang"), [Object](https://docs.oracle.com/en/java/javase/17/docs/api/java.base/java/lang/Object.html "class or interface in java.lang") > args,
[ToolContext](https://google.github.io/adk-docs/api-reference/java/com/google/adk/tools/ToolContext.html "class in com.google.adk.tools") toolContext)



Description copied from class: `BaseTool`



Calls a tool.

Overrides:`runAsync` in class `BaseTool`

- ### processLlmRequest [![Link icon](https://google.github.io/adk-docs/api-reference/java/resource-files/link.svg)](https://google.github.io/adk-docs/api-reference/java/com/google/adk/tools/LoadArtifactsTool.html\#processLlmRequest(com.google.adk.models.LlmRequest.Builder,com.google.adk.tools.ToolContext))





publicio.reactivex.rxjava3.core.CompletableprocessLlmRequest( [LlmRequest.Builder](https://google.github.io/adk-docs/api-reference/java/com/google/adk/models/LlmRequest.Builder.html "class in com.google.adk.models") llmRequestBuilder,
[ToolContext](https://google.github.io/adk-docs/api-reference/java/com/google/adk/tools/ToolContext.html "class in com.google.adk.tools") toolContext)



Description copied from class: `BaseTool`



Processes the outgoing [`LlmRequest.Builder`](https://google.github.io/adk-docs/api-reference/java/com/google/adk/models/LlmRequest.Builder.html "class in com.google.adk.models").



This implementation adds the current tool's [`BaseTool.declaration()`](https://google.github.io/adk-docs/api-reference/java/com/google/adk/tools/BaseTool.html#declaration()) to the `GenerateContentConfig` within the builder. If a tool with function declarations already exists,
the current tool's declaration is merged into it. Otherwise, a new tool definition with the
current tool's declaration is created. The current tool itself is also added to the builder's
internal list of tools. Override this method for processing the outgoing request.



Overrides:`processLlmRequest` in class `BaseTool`

- ### appendArtifactsToLlmRequest [![Link icon](https://google.github.io/adk-docs/api-reference/java/resource-files/link.svg)](https://google.github.io/adk-docs/api-reference/java/com/google/adk/tools/LoadArtifactsTool.html\#appendArtifactsToLlmRequest(com.google.adk.models.LlmRequest.Builder,com.google.adk.tools.ToolContext))





publicio.reactivex.rxjava3.core.CompletableappendArtifactsToLlmRequest( [LlmRequest.Builder](https://google.github.io/adk-docs/api-reference/java/com/google/adk/models/LlmRequest.Builder.html "class in com.google.adk.models") llmRequestBuilder,
[ToolContext](https://google.github.io/adk-docs/api-reference/java/com/google/adk/tools/ToolContext.html "class in com.google.adk.tools") toolContext)

## Before Agent Callback Sync
Enclosing class:`Callbacks`Functional Interface:This is a functional interface and can therefore be used as the assignment target for a lambda expression or method reference.

* * *

[@FunctionalInterface](https://docs.oracle.com/en/java/javase/17/docs/api/java.base/java/lang/FunctionalInterface.html "class or interface in java.lang")public static interface Callbacks.BeforeAgentCallbackSync

Helper interface to allow for sync beforeAgentCallback. The function is wrapped into an async
one before being processed further.

- ## Method Summary [![Link icon](https://google.github.io/adk-docs/api-reference/java/resource-files/link.svg)](https://google.github.io/adk-docs/api-reference/java/com/google/adk/agents/Callbacks.BeforeAgentCallbackSync.html\#method-summary)





All MethodsInstance MethodsAbstract Methods







Modifier and Type



Method



Description



`Optional<com.google.genai.types.Content>`



`call(CallbackContext callbackContext)`


- ## Method Details [![Link icon](https://google.github.io/adk-docs/api-reference/java/resource-files/link.svg)](https://google.github.io/adk-docs/api-reference/java/com/google/adk/agents/Callbacks.BeforeAgentCallbackSync.html\#method-detail)



- ### call [![Link icon](https://google.github.io/adk-docs/api-reference/java/resource-files/link.svg)](https://google.github.io/adk-docs/api-reference/java/com/google/adk/agents/Callbacks.BeforeAgentCallbackSync.html\#call(com.google.adk.agents.CallbackContext))





[Optional](https://docs.oracle.com/en/java/javase/17/docs/api/java.base/java/util/Optional.html "class or interface in java.util") <com.google.genai.types.Content>call( [CallbackContext](https://google.github.io/adk-docs/api-reference/java/com/google/adk/agents/CallbackContext.html "class in com.google.adk.agents") callbackContext)

## McpToolset Exception
[java.lang.Object](https://docs.oracle.com/en/java/javase/17/docs/api/java.base/java/lang/Object.html "class or interface in java.lang")

[java.lang.Throwable](https://docs.oracle.com/en/java/javase/17/docs/api/java.base/java/lang/Throwable.html "class or interface in java.lang")

[java.lang.Exception](https://docs.oracle.com/en/java/javase/17/docs/api/java.base/java/lang/Exception.html "class or interface in java.lang")

[java.lang.RuntimeException](https://docs.oracle.com/en/java/javase/17/docs/api/java.base/java/lang/RuntimeException.html "class or interface in java.lang")

com.google.adk.tools.mcp.McpToolset.McpToolsetException

All Implemented Interfaces:`Serializable`Direct Known Subclasses:`McpToolset.McpInitializationException`, `McpToolset.McpToolLoadingException`Enclosing class:`McpToolset`

* * *

public static class McpToolset.McpToolsetExceptionextends [RuntimeException](https://docs.oracle.com/en/java/javase/17/docs/api/java.base/java/lang/RuntimeException.html "class or interface in java.lang")

Base exception for all errors originating from `McpToolset`.

See Also:

- [Serialized Form](https://google.github.io/adk-docs/api-reference/java/serialized-form.html#com.google.adk.tools.mcp.McpToolset.McpToolsetException)

- ## Constructor Summary [![Link icon](https://google.github.io/adk-docs/api-reference/java/resource-files/link.svg)](https://google.github.io/adk-docs/api-reference/java/com/google/adk/tools/mcp/McpToolset.McpToolsetException.html\#constructor-summary)



Constructors





Constructor



Description



`McpToolsetException(String message,
Throwable cause)`

- ## Method Summary [![Link icon](https://google.github.io/adk-docs/api-reference/java/resource-files/link.svg)](https://google.github.io/adk-docs/api-reference/java/com/google/adk/tools/mcp/McpToolset.McpToolsetException.html\#method-summary)





### Methods inherited from class java.lang. [Throwable](https://docs.oracle.com/en/java/javase/17/docs/api/java.base/java/lang/Throwable.html "class or interface in java.lang") [![Link icon](https://google.github.io/adk-docs/api-reference/java/resource-files/link.svg)](https://google.github.io/adk-docs/api-reference/java/com/google/adk/tools/mcp/McpToolset.McpToolsetException.html\#methods-inherited-from-class-java.lang.Throwable)

`addSuppressed, fillInStackTrace, getCause, getLocalizedMessage, getMessage, getStackTrace, getSuppressed, initCause, printStackTrace, printStackTrace, printStackTrace, setStackTrace, toString`





### Methods inherited from class java.lang. [Object](https://docs.oracle.com/en/java/javase/17/docs/api/java.base/java/lang/Object.html "class or interface in java.lang") [![Link icon](https://google.github.io/adk-docs/api-reference/java/resource-files/link.svg)](https://google.github.io/adk-docs/api-reference/java/com/google/adk/tools/mcp/McpToolset.McpToolsetException.html\#methods-inherited-from-class-java.lang.Object)

`clone, equals, finalize, getClass, hashCode, notify, notifyAll, wait, wait, wait`


- ## Constructor Details [![Link icon](https://google.github.io/adk-docs/api-reference/java/resource-files/link.svg)](https://google.github.io/adk-docs/api-reference/java/com/google/adk/tools/mcp/McpToolset.McpToolsetException.html\#constructor-detail)



- ### McpToolsetException [![Link icon](https://google.github.io/adk-docs/api-reference/java/resource-files/link.svg)](https://google.github.io/adk-docs/api-reference/java/com/google/adk/tools/mcp/McpToolset.McpToolsetException.html\#%3Cinit%3E(java.lang.String,java.lang.Throwable))





publicMcpToolsetException( [String](https://docs.oracle.com/en/java/javase/17/docs/api/java.base/java/lang/String.html "class or interface in java.lang") message,
[Throwable](https://docs.oracle.com/en/java/javase/17/docs/api/java.base/java/lang/Throwable.html "class or interface in java.lang") cause)

## MCP Conversion Utilities
[java.lang.Object](https://docs.oracle.com/en/java/javase/17/docs/api/java.base/java/lang/Object.html "class or interface in java.lang")

com.google.adk.tools.mcp.ConversionUtils

* * *

public final class ConversionUtilsextends [Object](https://docs.oracle.com/en/java/javase/17/docs/api/java.base/java/lang/Object.html "class or interface in java.lang")

Utility class for converting between different representations of MCP tools.

- ## Method Summary [![Link icon](https://google.github.io/adk-docs/api-reference/java/resource-files/link.svg)](https://google.github.io/adk-docs/api-reference/java/com/google/adk/tools/mcp/ConversionUtils.html\#method-summary)





All MethodsInstance MethodsConcrete Methods







Modifier and Type



Method



Description



`io.modelcontextprotocol.spec.McpSchema.Tool`



`adkToMcpToolType(BaseTool tool)`















### Methods inherited from class java.lang. [Object](https://docs.oracle.com/en/java/javase/17/docs/api/java.base/java/lang/Object.html "class or interface in java.lang") [![Link icon](https://google.github.io/adk-docs/api-reference/java/resource-files/link.svg)](https://google.github.io/adk-docs/api-reference/java/com/google/adk/tools/mcp/ConversionUtils.html\#methods-inherited-from-class-java.lang.Object)

`clone, equals, finalize, getClass, hashCode, notify, notifyAll, toString, wait, wait, wait`


- ## Method Details [![Link icon](https://google.github.io/adk-docs/api-reference/java/resource-files/link.svg)](https://google.github.io/adk-docs/api-reference/java/com/google/adk/tools/mcp/ConversionUtils.html\#method-detail)



- ### adkToMcpToolType [![Link icon](https://google.github.io/adk-docs/api-reference/java/resource-files/link.svg)](https://google.github.io/adk-docs/api-reference/java/com/google/adk/tools/mcp/ConversionUtils.html\#adkToMcpToolType(com.google.adk.tools.BaseTool))





publicio.modelcontextprotocol.spec.McpSchema.TooladkToMcpToolType( [BaseTool](https://google.github.io/adk-docs/api-reference/java/com/google/adk/tools/BaseTool.html "class in com.google.adk.tools") tool)

## Entity Schema Operations
[java.lang.Object](https://docs.oracle.com/en/java/javase/17/docs/api/java.base/java/lang/Object.html "class or interface in java.lang")

com.google.adk.tools.applicationintegrationtoolset.ConnectionsClient.EntitySchemaAndOperations

Enclosing class:`ConnectionsClient`

* * *

public static class ConnectionsClient.EntitySchemaAndOperationsextends [Object](https://docs.oracle.com/en/java/javase/17/docs/api/java.base/java/lang/Object.html "class or interface in java.lang")

Represents the schema and available operations for an entity.

- ## Field Summary [![Link icon](https://google.github.io/adk-docs/api-reference/java/resource-files/link.svg)](https://google.github.io/adk-docs/api-reference/java/com/google/adk/tools/applicationintegrationtoolset/ConnectionsClient.EntitySchemaAndOperations.html\#field-summary)



Fields





Modifier and Type



Field



Description



`List<String>`



`operations`







`Map<String,Object>`



`schema`

- ## Constructor Summary [![Link icon](https://google.github.io/adk-docs/api-reference/java/resource-files/link.svg)](https://google.github.io/adk-docs/api-reference/java/com/google/adk/tools/applicationintegrationtoolset/ConnectionsClient.EntitySchemaAndOperations.html\#constructor-summary)



Constructors





Constructor



Description



`EntitySchemaAndOperations()`

- ## Method Summary [![Link icon](https://google.github.io/adk-docs/api-reference/java/resource-files/link.svg)](https://google.github.io/adk-docs/api-reference/java/com/google/adk/tools/applicationintegrationtoolset/ConnectionsClient.EntitySchemaAndOperations.html\#method-summary)





### Methods inherited from class java.lang. [Object](https://docs.oracle.com/en/java/javase/17/docs/api/java.base/java/lang/Object.html "class or interface in java.lang") [![Link icon](https://google.github.io/adk-docs/api-reference/java/resource-files/link.svg)](https://google.github.io/adk-docs/api-reference/java/com/google/adk/tools/applicationintegrationtoolset/ConnectionsClient.EntitySchemaAndOperations.html\#methods-inherited-from-class-java.lang.Object)

`clone, equals, finalize, getClass, hashCode, notify, notifyAll, toString, wait, wait, wait`


- ## Field Details [![Link icon](https://google.github.io/adk-docs/api-reference/java/resource-files/link.svg)](https://google.github.io/adk-docs/api-reference/java/com/google/adk/tools/applicationintegrationtoolset/ConnectionsClient.EntitySchemaAndOperations.html\#field-detail)



- ### schema [![Link icon](https://google.github.io/adk-docs/api-reference/java/resource-files/link.svg)](https://google.github.io/adk-docs/api-reference/java/com/google/adk/tools/applicationintegrationtoolset/ConnectionsClient.EntitySchemaAndOperations.html\#schema)





public[Map](https://docs.oracle.com/en/java/javase/17/docs/api/java.base/java/util/Map.html "class or interface in java.util") < [String](https://docs.oracle.com/en/java/javase/17/docs/api/java.base/java/lang/String.html "class or interface in java.lang"), [Object](https://docs.oracle.com/en/java/javase/17/docs/api/java.base/java/lang/Object.html "class or interface in java.lang") >schema

- ### operations [![Link icon](https://google.github.io/adk-docs/api-reference/java/resource-files/link.svg)](https://google.github.io/adk-docs/api-reference/java/com/google/adk/tools/applicationintegrationtoolset/ConnectionsClient.EntitySchemaAndOperations.html\#operations)





public[List](https://docs.oracle.com/en/java/javase/17/docs/api/java.base/java/util/List.html "class or interface in java.util") < [String](https://docs.oracle.com/en/java/javase/17/docs/api/java.base/java/lang/String.html "class or interface in java.lang") >operations


- ## Constructor Details [![Link icon](https://google.github.io/adk-docs/api-reference/java/resource-files/link.svg)](https://google.github.io/adk-docs/api-reference/java/com/google/adk/tools/applicationintegrationtoolset/ConnectionsClient.EntitySchemaAndOperations.html\#constructor-detail)



- ### EntitySchemaAndOperations [![Link icon](https://google.github.io/adk-docs/api-reference/java/resource-files/link.svg)](https://google.github.io/adk-docs/api-reference/java/com/google/adk/tools/applicationintegrationtoolset/ConnectionsClient.EntitySchemaAndOperations.html\#%3Cinit%3E())





publicEntitySchemaAndOperations()

## Vertex AI RAG Retrieval
[java.lang.Object](https://docs.oracle.com/en/java/javase/17/docs/api/java.base/java/lang/Object.html "class or interface in java.lang")

[com.google.adk.tools.BaseTool](https://google.github.io/adk-docs/api-reference/java/com/google/adk/tools/BaseTool.html "class in com.google.adk.tools")

[com.google.adk.tools.retrieval.BaseRetrievalTool](https://google.github.io/adk-docs/api-reference/java/com/google/adk/tools/retrieval/BaseRetrievalTool.html "class in com.google.adk.tools.retrieval")

com.google.adk.tools.retrieval.VertexAiRagRetrieval

* * *

public class VertexAiRagRetrievalextends [BaseRetrievalTool](https://google.github.io/adk-docs/api-reference/java/com/google/adk/tools/retrieval/BaseRetrievalTool.html "class in com.google.adk.tools.retrieval")

A retrieval tool that fetches context from Vertex AI RAG.



This tool allows to retrieve relevant information based on a query using Vertex AI RAG
service. It supports configuration of rag resources and a vector distance threshold.

- ## Constructor Summary [![Link icon](https://google.github.io/adk-docs/api-reference/java/resource-files/link.svg)](https://google.github.io/adk-docs/api-reference/java/com/google/adk/tools/retrieval/VertexAiRagRetrieval.html\#constructor-summary)



Constructors





Constructor



Description



`VertexAiRagRetrieval(String name,
String description,
com.google.cloud.aiplatform.v1.VertexRagServiceClient vertexRagServiceClient,
String parent,
List<com.google.cloud.aiplatform.v1.RetrieveContextsRequest.VertexRagStore.RagResource> ragResources,
Double vectorDistanceThreshold)`

- ## Method Summary [![Link icon](https://google.github.io/adk-docs/api-reference/java/resource-files/link.svg)](https://google.github.io/adk-docs/api-reference/java/com/google/adk/tools/retrieval/VertexAiRagRetrieval.html\#method-summary)





All MethodsInstance MethodsConcrete Methods







Modifier and Type



Method



Description



`io.reactivex.rxjava3.core.Completable`



`processLlmRequest(LlmRequest.Builder llmRequestBuilder,
ToolContext toolContext)`





Processes the outgoing [`LlmRequest.Builder`](https://google.github.io/adk-docs/api-reference/java/com/google/adk/models/LlmRequest.Builder.html "class in com.google.adk.models").





`io.reactivex.rxjava3.core.Single<Map<String,Object>>`



`runAsync(Map<String,Object> args,
ToolContext toolContext)`





Calls a tool.













### Methods inherited from class com.google.adk.tools.retrieval. [BaseRetrievalTool](https://google.github.io/adk-docs/api-reference/java/com/google/adk/tools/retrieval/BaseRetrievalTool.html "class in com.google.adk.tools.retrieval") [![Link icon](https://google.github.io/adk-docs/api-reference/java/resource-files/link.svg)](https://google.github.io/adk-docs/api-reference/java/com/google/adk/tools/retrieval/VertexAiRagRetrieval.html\#methods-inherited-from-class-com.google.adk.tools.retrieval.BaseRetrievalTool)

`declaration`





### Methods inherited from class com.google.adk.tools. [BaseTool](https://google.github.io/adk-docs/api-reference/java/com/google/adk/tools/BaseTool.html "class in com.google.adk.tools") [![Link icon](https://google.github.io/adk-docs/api-reference/java/resource-files/link.svg)](https://google.github.io/adk-docs/api-reference/java/com/google/adk/tools/retrieval/VertexAiRagRetrieval.html\#methods-inherited-from-class-com.google.adk.tools.BaseTool)

`description, longRunning, name`





### Methods inherited from class java.lang. [Object](https://docs.oracle.com/en/java/javase/17/docs/api/java.base/java/lang/Object.html "class or interface in java.lang") [![Link icon](https://google.github.io/adk-docs/api-reference/java/resource-files/link.svg)](https://google.github.io/adk-docs/api-reference/java/com/google/adk/tools/retrieval/VertexAiRagRetrieval.html\#methods-inherited-from-class-java.lang.Object)

`clone, equals, finalize, getClass, hashCode, notify, notifyAll, toString, wait, wait, wait`


- ## Constructor Details [![Link icon](https://google.github.io/adk-docs/api-reference/java/resource-files/link.svg)](https://google.github.io/adk-docs/api-reference/java/com/google/adk/tools/retrieval/VertexAiRagRetrieval.html\#constructor-detail)



- ### VertexAiRagRetrieval [![Link icon](https://google.github.io/adk-docs/api-reference/java/resource-files/link.svg)](https://google.github.io/adk-docs/api-reference/java/com/google/adk/tools/retrieval/VertexAiRagRetrieval.html\#%3Cinit%3E(java.lang.String,java.lang.String,com.google.cloud.aiplatform.v1.VertexRagServiceClient,java.lang.String,java.util.List,java.lang.Double))





publicVertexAiRagRetrieval(@Nonnull
[String](https://docs.oracle.com/en/java/javase/17/docs/api/java.base/java/lang/String.html "class or interface in java.lang") name,
@Nonnull
[String](https://docs.oracle.com/en/java/javase/17/docs/api/java.base/java/lang/String.html "class or interface in java.lang") description,
@Nonnull
com.google.cloud.aiplatform.v1.VertexRagServiceClient vertexRagServiceClient,
@Nonnull
[String](https://docs.oracle.com/en/java/javase/17/docs/api/java.base/java/lang/String.html "class or interface in java.lang") parent,
@Nullable
[List](https://docs.oracle.com/en/java/javase/17/docs/api/java.base/java/util/List.html "class or interface in java.util") <com.google.cloud.aiplatform.v1.RetrieveContextsRequest.VertexRagStore.RagResource> ragResources,
@Nullable
[Double](https://docs.oracle.com/en/java/javase/17/docs/api/java.base/java/lang/Double.html "class or interface in java.lang") vectorDistanceThreshold)


- ## Method Details [![Link icon](https://google.github.io/adk-docs/api-reference/java/resource-files/link.svg)](https://google.github.io/adk-docs/api-reference/java/com/google/adk/tools/retrieval/VertexAiRagRetrieval.html\#method-detail)



- ### processLlmRequest [![Link icon](https://google.github.io/adk-docs/api-reference/java/resource-files/link.svg)](https://google.github.io/adk-docs/api-reference/java/com/google/adk/tools/retrieval/VertexAiRagRetrieval.html\#processLlmRequest(com.google.adk.models.LlmRequest.Builder,com.google.adk.tools.ToolContext))





@CanIgnoreReturnValue
publicio.reactivex.rxjava3.core.CompletableprocessLlmRequest( [LlmRequest.Builder](https://google.github.io/adk-docs/api-reference/java/com/google/adk/models/LlmRequest.Builder.html "class in com.google.adk.models") llmRequestBuilder,
[ToolContext](https://google.github.io/adk-docs/api-reference/java/com/google/adk/tools/ToolContext.html "class in com.google.adk.tools") toolContext)



Description copied from class: `BaseTool`



Processes the outgoing [`LlmRequest.Builder`](https://google.github.io/adk-docs/api-reference/java/com/google/adk/models/LlmRequest.Builder.html "class in com.google.adk.models").



This implementation adds the current tool's [`BaseTool.declaration()`](https://google.github.io/adk-docs/api-reference/java/com/google/adk/tools/BaseTool.html#declaration()) to the `GenerateContentConfig` within the builder. If a tool with function declarations already exists,
the current tool's declaration is merged into it. Otherwise, a new tool definition with the
current tool's declaration is created. The current tool itself is also added to the builder's
internal list of tools. Override this method for processing the outgoing request.



Overrides:`processLlmRequest` in class `BaseTool`

- ### runAsync [![Link icon](https://google.github.io/adk-docs/api-reference/java/resource-files/link.svg)](https://google.github.io/adk-docs/api-reference/java/com/google/adk/tools/retrieval/VertexAiRagRetrieval.html\#runAsync(java.util.Map,com.google.adk.tools.ToolContext))





publicio.reactivex.rxjava3.core.Single< [Map](https://docs.oracle.com/en/java/javase/17/docs/api/java.base/java/util/Map.html "class or interface in java.util") < [String](https://docs.oracle.com/en/java/javase/17/docs/api/java.base/java/lang/String.html "class or interface in java.lang"), [Object](https://docs.oracle.com/en/java/javase/17/docs/api/java.base/java/lang/Object.html "class or interface in java.lang") >>runAsync( [Map](https://docs.oracle.com/en/java/javase/17/docs/api/java.base/java/util/Map.html "class or interface in java.util") < [String](https://docs.oracle.com/en/java/javase/17/docs/api/java.base/java/lang/String.html "class or interface in java.lang"), [Object](https://docs.oracle.com/en/java/javase/17/docs/api/java.base/java/lang/Object.html "class or interface in java.lang") > args,
[ToolContext](https://google.github.io/adk-docs/api-reference/java/com/google/adk/tools/ToolContext.html "class in com.google.adk.tools") toolContext)



Description copied from class: `BaseTool`



Calls a tool.

Overrides:`runAsync` in class `BaseTool`

## ADK REST API Reference
[Skip to content](https://google.github.io/adk-docs/api-reference/rest/#rest-api-reference)

[Edit this page](https://github.com/google/adk-docs/edit/main/docs/api-reference/rest/index.md "Edit this page") [View source of this page](https://github.com/google/adk-docs/raw/main/docs/api-reference/rest/index.md "View source of this page")

# REST API Reference [¶](https://google.github.io/adk-docs/api-reference/rest/\#rest-api-reference "Permanent link")

This page provides a reference for the REST API provided by the ADK web server.
For details on using the ADK REST API in practice, see
[Use the API Server](https://google.github.io/adk-docs/runtime/api-server/).

Tip

You can view an updated API reference on a running ADK web server by browsing
the `/docs` location, for example at: `http://localhost:8000/docs`

## Endpoints [¶](https://google.github.io/adk-docs/api-reference/rest/\#endpoints "Permanent link")

### `/run` [¶](https://google.github.io/adk-docs/api-reference/rest/\#run "Permanent link")

This endpoint executes an agent run. It takes a JSON payload with the details of the run and returns a list of events generated during the run.

**Request Body**

The request body should be a JSON object with the following fields:

- `app_name` (string, required): The name of the agent to run.
- `user_id` (string, required): The ID of the user.
- `session_id` (string, required): The ID of the session.
- `new_message` (Content, required): The new message to send to the agent. See the [Content](https://google.github.io/adk-docs/api-reference/rest/#content-object) section for more details.
- `streaming` (boolean, optional): Whether to use streaming. Defaults to `false`.
- `state_delta` (object, optional): A delta of the state to apply before the run.

**Response Body**

The response body is a JSON array of [Event](https://google.github.io/adk-docs/api-reference/rest/#event-object) objects.

### `/run_sse` [¶](https://google.github.io/adk-docs/api-reference/rest/\#run_sse "Permanent link")

This endpoint executes an agent run using Server-Sent Events (SSE) for streaming responses. It takes the same JSON payload as the `/run` endpoint.

**Request Body**

The request body is the same as for the `/run` endpoint.

**Response Body**

The response is a stream of Server-Sent Events. Each event is a JSON object representing an [Event](https://google.github.io/adk-docs/api-reference/rest/#event-object).

## Objects [¶](https://google.github.io/adk-docs/api-reference/rest/\#objects "Permanent link")

### `Content` object [¶](https://google.github.io/adk-docs/api-reference/rest/\#content-object "Permanent link")

The `Content` object represents the content of a message. It has the following structure:

```
{
  "parts": [\
    {\
      "text": "..."\
    }\
  ],
  "role": "..."
}
```

- `parts`: A list of parts. Each part can be either text or a function call.
- `role`: The role of the author of the message (e.g., "user", "model").

### `Event` object [¶](https://google.github.io/adk-docs/api-reference/rest/\#event-object "Permanent link")

The `Event` object represents an event that occurred during an agent run. It has a complex structure with many optional fields. The most important fields are:

- `id`: The ID of the event.
- `timestamp`: The timestamp of the event.
- `author`: The author of the event.
- `content`: The content of the event.

Back to top

## LiveRequest.Builder Overview
Packages that use [LiveRequest.Builder](https://google.github.io/adk-docs/api-reference/java/com/google/adk/agents/LiveRequest.Builder.html "class in com.google.adk.agents")

Package

Description

[com.google.adk.agents](https://google.github.io/adk-docs/api-reference/java/com/google/adk/agents/class-use/LiveRequest.Builder.html#com.google.adk.agents)

- ## Uses of [LiveRequest.Builder](https://google.github.io/adk-docs/api-reference/java/com/google/adk/agents/LiveRequest.Builder.html "class in com.google.adk.agents") in [com.google.adk.agents](https://google.github.io/adk-docs/api-reference/java/com/google/adk/agents/package-summary.html) [![Link icon](https://google.github.io/adk-docs/api-reference/java/resource-files/link.svg)](https://google.github.io/adk-docs/api-reference/java/com/google/adk/agents/class-use/LiveRequest.Builder.html\#com.google.adk.agents)



Methods in [com.google.adk.agents](https://google.github.io/adk-docs/api-reference/java/com/google/adk/agents/package-summary.html) that return [LiveRequest.Builder](https://google.github.io/adk-docs/api-reference/java/com/google/adk/agents/LiveRequest.Builder.html "class in com.google.adk.agents")





Modifier and Type



Method



Description



`abstract LiveRequest.Builder`



LiveRequest.Builder.`blob(com.google.genai.types.Blob blob)`







`abstract LiveRequest.Builder`



LiveRequest.Builder.`blob(Optional<com.google.genai.types.Blob> blob)`







`static LiveRequest.Builder`



LiveRequest.`builder()`







`abstract LiveRequest.Builder`



LiveRequest.Builder.`close(Boolean close)`







`abstract LiveRequest.Builder`



LiveRequest.Builder.`close(Optional<Boolean> close)`







`abstract LiveRequest.Builder`



LiveRequest.Builder.`content(com.google.genai.types.Content content)`







`abstract LiveRequest.Builder`



LiveRequest.Builder.`content(Optional<com.google.genai.types.Content> content)`







`abstract LiveRequest.Builder`



LiveRequest.`toBuilder()`

## Instruction Usage Overview
Packages that use [Instruction](https://google.github.io/adk-docs/api-reference/java/com/google/adk/agents/Instruction.html "interface in com.google.adk.agents")

Package

Description

[com.google.adk.agents](https://google.github.io/adk-docs/api-reference/java/com/google/adk/agents/class-use/Instruction.html#com.google.adk.agents)

- ## Uses of [Instruction](https://google.github.io/adk-docs/api-reference/java/com/google/adk/agents/Instruction.html "interface in com.google.adk.agents") in [com.google.adk.agents](https://google.github.io/adk-docs/api-reference/java/com/google/adk/agents/package-summary.html) [![Link icon](https://google.github.io/adk-docs/api-reference/java/resource-files/link.svg)](https://google.github.io/adk-docs/api-reference/java/com/google/adk/agents/class-use/Instruction.html\#com.google.adk.agents)



Classes in [com.google.adk.agents](https://google.github.io/adk-docs/api-reference/java/com/google/adk/agents/package-summary.html) that implement [Instruction](https://google.github.io/adk-docs/api-reference/java/com/google/adk/agents/Instruction.html "interface in com.google.adk.agents")





Modifier and Type



Class



Description



`static final record`



`Instruction.Provider`





Returns an instruction dynamically constructed from the given context.





`static final record`



`Instruction.Static`





Plain instruction directly provided to the agent.







Methods in [com.google.adk.agents](https://google.github.io/adk-docs/api-reference/java/com/google/adk/agents/package-summary.html) that return [Instruction](https://google.github.io/adk-docs/api-reference/java/com/google/adk/agents/Instruction.html "interface in com.google.adk.agents")





Modifier and Type



Method



Description



`Instruction`



LlmAgent.`globalInstruction()`







`Instruction`



LlmAgent.`instruction()`









Methods in [com.google.adk.agents](https://google.github.io/adk-docs/api-reference/java/com/google/adk/agents/package-summary.html) with parameters of type [Instruction](https://google.github.io/adk-docs/api-reference/java/com/google/adk/agents/Instruction.html "interface in com.google.adk.agents")





Modifier and Type



Method



Description



`LlmAgent.Builder`



LlmAgent.Builder.`globalInstruction(Instruction globalInstruction)`







`LlmAgent.Builder`



LlmAgent.Builder.`instruction(Instruction instruction)`

## JsonBaseModel Overview
[java.lang.Object](https://docs.oracle.com/en/java/javase/17/docs/api/java.base/java/lang/Object.html "class or interface in java.lang")

com.google.adk.JsonBaseModel

Direct Known Subclasses:`AdkWebServer.RunEvalResult`, `Event`, `LiveRequest`, `LlmRequest`, `LlmResponse`, `Session`

* * *

public abstract class JsonBaseModelextends [Object](https://docs.oracle.com/en/java/javase/17/docs/api/java.base/java/lang/Object.html "class or interface in java.lang")

The base class for the types that needs JSON serialization/deserialization capability.

- ## Constructor Summary [![Link icon](https://google.github.io/adk-docs/api-reference/java/resource-files/link.svg)](https://google.github.io/adk-docs/api-reference/java/com/google/adk/JsonBaseModel.html\#constructor-summary)



Constructors





Constructor



Description



`JsonBaseModel()`

- ## Method Summary [![Link icon](https://google.github.io/adk-docs/api-reference/java/resource-files/link.svg)](https://google.github.io/adk-docs/api-reference/java/com/google/adk/JsonBaseModel.html\#method-summary)





All MethodsStatic MethodsInstance MethodsConcrete Methods







Modifier and Type



Method



Description



`static <T extends JsonBaseModel>
T`



`fromJsonNode(com.fasterxml.jackson.databind.JsonNode jsonNode,
Class<T> clazz)`





Deserializes a JsonNode to an object of the given type.





`static <T extends JsonBaseModel>
T`



`fromJsonString(String jsonString,
Class<T> clazz)`





Deserializes a Json string to an object of the given type.





`static com.fasterxml.jackson.databind.ObjectMapper`



`getMapper()`







`String`



`toJson()`







`protected static com.fasterxml.jackson.databind.JsonNode`



`toJsonNode(Object object)`





Serializes an object to a JsonNode.





`protected static String`



`toJsonString(Object object)`





Serializes an object to a Json string.













### Methods inherited from class java.lang. [Object](https://docs.oracle.com/en/java/javase/17/docs/api/java.base/java/lang/Object.html "class or interface in java.lang") [![Link icon](https://google.github.io/adk-docs/api-reference/java/resource-files/link.svg)](https://google.github.io/adk-docs/api-reference/java/com/google/adk/JsonBaseModel.html\#methods-inherited-from-class-java.lang.Object)

`clone, equals, finalize, getClass, hashCode, notify, notifyAll, toString, wait, wait, wait`


- ## Constructor Details [![Link icon](https://google.github.io/adk-docs/api-reference/java/resource-files/link.svg)](https://google.github.io/adk-docs/api-reference/java/com/google/adk/JsonBaseModel.html\#constructor-detail)



- ### JsonBaseModel [![Link icon](https://google.github.io/adk-docs/api-reference/java/resource-files/link.svg)](https://google.github.io/adk-docs/api-reference/java/com/google/adk/JsonBaseModel.html\#%3Cinit%3E())





publicJsonBaseModel()


- ## Method Details [![Link icon](https://google.github.io/adk-docs/api-reference/java/resource-files/link.svg)](https://google.github.io/adk-docs/api-reference/java/com/google/adk/JsonBaseModel.html\#method-detail)



- ### toJsonString [![Link icon](https://google.github.io/adk-docs/api-reference/java/resource-files/link.svg)](https://google.github.io/adk-docs/api-reference/java/com/google/adk/JsonBaseModel.html\#toJsonString(java.lang.Object))





protected static[String](https://docs.oracle.com/en/java/javase/17/docs/api/java.base/java/lang/String.html "class or interface in java.lang")toJsonString( [Object](https://docs.oracle.com/en/java/javase/17/docs/api/java.base/java/lang/Object.html "class or interface in java.lang") object)



Serializes an object to a Json string.

- ### getMapper [![Link icon](https://google.github.io/adk-docs/api-reference/java/resource-files/link.svg)](https://google.github.io/adk-docs/api-reference/java/com/google/adk/JsonBaseModel.html\#getMapper())





public staticcom.fasterxml.jackson.databind.ObjectMappergetMapper()

- ### toJson [![Link icon](https://google.github.io/adk-docs/api-reference/java/resource-files/link.svg)](https://google.github.io/adk-docs/api-reference/java/com/google/adk/JsonBaseModel.html\#toJson())





public[String](https://docs.oracle.com/en/java/javase/17/docs/api/java.base/java/lang/String.html "class or interface in java.lang")toJson()

- ### toJsonNode [![Link icon](https://google.github.io/adk-docs/api-reference/java/resource-files/link.svg)](https://google.github.io/adk-docs/api-reference/java/com/google/adk/JsonBaseModel.html\#toJsonNode(java.lang.Object))





protected staticcom.fasterxml.jackson.databind.JsonNodetoJsonNode( [Object](https://docs.oracle.com/en/java/javase/17/docs/api/java.base/java/lang/Object.html "class or interface in java.lang") object)



Serializes an object to a JsonNode.

- ### fromJsonString [![Link icon](https://google.github.io/adk-docs/api-reference/java/resource-files/link.svg)](https://google.github.io/adk-docs/api-reference/java/com/google/adk/JsonBaseModel.html\#fromJsonString(java.lang.String,java.lang.Class))





public static<T extends [JsonBaseModel](https://google.github.io/adk-docs/api-reference/java/com/google/adk/JsonBaseModel.html "class in com.google.adk") >TfromJsonString( [String](https://docs.oracle.com/en/java/javase/17/docs/api/java.base/java/lang/String.html "class or interface in java.lang") jsonString,
[Class](https://docs.oracle.com/en/java/javase/17/docs/api/java.base/java/lang/Class.html "class or interface in java.lang") <T> clazz)



Deserializes a Json string to an object of the given type.

- ### fromJsonNode [![Link icon](https://google.github.io/adk-docs/api-reference/java/resource-files/link.svg)](https://google.github.io/adk-docs/api-reference/java/com/google/adk/JsonBaseModel.html\#fromJsonNode(com.fasterxml.jackson.databind.JsonNode,java.lang.Class))





public static<T extends [JsonBaseModel](https://google.github.io/adk-docs/api-reference/java/com/google/adk/JsonBaseModel.html "class in com.google.adk") >TfromJsonNode(com.fasterxml.jackson.databind.JsonNode jsonNode,
[Class](https://docs.oracle.com/en/java/javase/17/docs/api/java.base/java/lang/Class.html "class or interface in java.lang") <T> clazz)



Deserializes a JsonNode to an object of the given type.

## RunConfig.StreamingMode Overview
Packages that use [RunConfig.StreamingMode](https://google.github.io/adk-docs/api-reference/java/com/google/adk/agents/RunConfig.StreamingMode.html "enum class in com.google.adk.agents")

Package

Description

[com.google.adk.agents](https://google.github.io/adk-docs/api-reference/java/com/google/adk/agents/class-use/RunConfig.StreamingMode.html#com.google.adk.agents)

- ## Uses of [RunConfig.StreamingMode](https://google.github.io/adk-docs/api-reference/java/com/google/adk/agents/RunConfig.StreamingMode.html "enum class in com.google.adk.agents") in [com.google.adk.agents](https://google.github.io/adk-docs/api-reference/java/com/google/adk/agents/package-summary.html) [![Link icon](https://google.github.io/adk-docs/api-reference/java/resource-files/link.svg)](https://google.github.io/adk-docs/api-reference/java/com/google/adk/agents/class-use/RunConfig.StreamingMode.html\#com.google.adk.agents)



Subclasses with type arguments of type [RunConfig.StreamingMode](https://google.github.io/adk-docs/api-reference/java/com/google/adk/agents/RunConfig.StreamingMode.html "enum class in com.google.adk.agents") in [com.google.adk.agents](https://google.github.io/adk-docs/api-reference/java/com/google/adk/agents/package-summary.html)





Modifier and Type



Class



Description



`static enum`



`RunConfig.StreamingMode`





Streaming mode for the runner.







Methods in [com.google.adk.agents](https://google.github.io/adk-docs/api-reference/java/com/google/adk/agents/package-summary.html) that return [RunConfig.StreamingMode](https://google.github.io/adk-docs/api-reference/java/com/google/adk/agents/RunConfig.StreamingMode.html "enum class in com.google.adk.agents")





Modifier and Type



Method



Description



`abstract RunConfig.StreamingMode`



RunConfig.`streamingMode()`







`static RunConfig.StreamingMode`



RunConfig.StreamingMode.`valueOf(String name)`





Returns the enum constant of this class with the specified name.





`static RunConfig.StreamingMode[]`



RunConfig.StreamingMode.`values()`





Returns an array containing the constants of this enum class, in
the order they are declared.







Methods in [com.google.adk.agents](https://google.github.io/adk-docs/api-reference/java/com/google/adk/agents/package-summary.html) with parameters of type [RunConfig.StreamingMode](https://google.github.io/adk-docs/api-reference/java/com/google/adk/agents/RunConfig.StreamingMode.html "enum class in com.google.adk.agents")





Modifier and Type



Method



Description



`abstract RunConfig.Builder`



RunConfig.Builder.`setStreamingMode(RunConfig.StreamingMode streamingMode)`

## AdkWebServer LiveWebSocketHandler
Packages that use [AdkWebServer.LiveWebSocketHandler](https://google.github.io/adk-docs/api-reference/java/com/google/adk/web/AdkWebServer.LiveWebSocketHandler.html "class in com.google.adk.web")

Package

Description

[com.google.adk.web](https://google.github.io/adk-docs/api-reference/java/com/google/adk/web/class-use/AdkWebServer.LiveWebSocketHandler.html#com.google.adk.web)

- ## Uses of [AdkWebServer.LiveWebSocketHandler](https://google.github.io/adk-docs/api-reference/java/com/google/adk/web/AdkWebServer.LiveWebSocketHandler.html "class in com.google.adk.web") in [com.google.adk.web](https://google.github.io/adk-docs/api-reference/java/com/google/adk/web/package-summary.html) [![Link icon](https://google.github.io/adk-docs/api-reference/java/resource-files/link.svg)](https://google.github.io/adk-docs/api-reference/java/com/google/adk/web/class-use/AdkWebServer.LiveWebSocketHandler.html\#com.google.adk.web)



Constructors in [com.google.adk.web](https://google.github.io/adk-docs/api-reference/java/com/google/adk/web/package-summary.html) with parameters of type [AdkWebServer.LiveWebSocketHandler](https://google.github.io/adk-docs/api-reference/java/com/google/adk/web/AdkWebServer.LiveWebSocketHandler.html "class in com.google.adk.web")





Modifier



Constructor



Description



``



`WebSocketConfig(AdkWebServer.LiveWebSocketHandler liveWebSocketHandler)`

## LLM Response Processor
* * *

public interface ResponseProcessor

- ## Nested Class Summary [![Link icon](https://google.github.io/adk-docs/api-reference/java/resource-files/link.svg)](https://google.github.io/adk-docs/api-reference/java/com/google/adk/flows/llmflows/ResponseProcessor.html\#nested-class-summary)



Nested Classes





Modifier and Type



Interface



Description



`static class`



`ResponseProcessor.ResponseProcessingResult`

- ## Method Summary [![Link icon](https://google.github.io/adk-docs/api-reference/java/resource-files/link.svg)](https://google.github.io/adk-docs/api-reference/java/com/google/adk/flows/llmflows/ResponseProcessor.html\#method-summary)





All MethodsInstance MethodsAbstract Methods







Modifier and Type



Method



Description



`io.reactivex.rxjava3.core.Single<ResponseProcessor.ResponseProcessingResult>`



`processResponse(InvocationContext context,
LlmResponse response)`





Process the LLM response as part of the post-processing stage.


- ## Method Details [![Link icon](https://google.github.io/adk-docs/api-reference/java/resource-files/link.svg)](https://google.github.io/adk-docs/api-reference/java/com/google/adk/flows/llmflows/ResponseProcessor.html\#method-detail)



- ### processResponse [![Link icon](https://google.github.io/adk-docs/api-reference/java/resource-files/link.svg)](https://google.github.io/adk-docs/api-reference/java/com/google/adk/flows/llmflows/ResponseProcessor.html\#processResponse(com.google.adk.agents.InvocationContext,com.google.adk.models.LlmResponse))





io.reactivex.rxjava3.core.Single< [ResponseProcessor.ResponseProcessingResult](https://google.github.io/adk-docs/api-reference/java/com/google/adk/flows/llmflows/ResponseProcessor.ResponseProcessingResult.html "class in com.google.adk.flows.llmflows") >processResponse( [InvocationContext](https://google.github.io/adk-docs/api-reference/java/com/google/adk/agents/InvocationContext.html "class in com.google.adk.agents") context,
[LlmResponse](https://google.github.io/adk-docs/api-reference/java/com/google/adk/models/LlmResponse.html "class in com.google.adk.models") response)



Process the LLM response as part of the post-processing stage.

Parameters:`context` \- the invocation context.`response` \- the LLM response to process.Returns:a list of events generated during processing (if any).

## Schema Validation Utility
[java.lang.Object](https://docs.oracle.com/en/java/javase/17/docs/api/java.base/java/lang/Object.html "class or interface in java.lang")

com.google.adk.SchemaUtils

* * *

public final class SchemaUtilsextends [Object](https://docs.oracle.com/en/java/javase/17/docs/api/java.base/java/lang/Object.html "class or interface in java.lang")

Utility class for validating schemas.

- ## Method Summary [![Link icon](https://google.github.io/adk-docs/api-reference/java/resource-files/link.svg)](https://google.github.io/adk-docs/api-reference/java/com/google/adk/SchemaUtils.html\#method-summary)





All MethodsStatic MethodsConcrete Methods







Modifier and Type



Method



Description



`static void`



`validateMapOnSchema(Map<String,Object> args,
com.google.genai.types.Schema schema,
Boolean isInput)`





Validates a map against a schema.





`static Map<String,Object>`



`validateOutputSchema(String output,
com.google.genai.types.Schema schema)`





Validates an output string against a schema.













### Methods inherited from class java.lang. [Object](https://docs.oracle.com/en/java/javase/17/docs/api/java.base/java/lang/Object.html "class or interface in java.lang") [![Link icon](https://google.github.io/adk-docs/api-reference/java/resource-files/link.svg)](https://google.github.io/adk-docs/api-reference/java/com/google/adk/SchemaUtils.html\#methods-inherited-from-class-java.lang.Object)

`clone, equals, finalize, getClass, hashCode, notify, notifyAll, toString, wait, wait, wait`


- ## Method Details [![Link icon](https://google.github.io/adk-docs/api-reference/java/resource-files/link.svg)](https://google.github.io/adk-docs/api-reference/java/com/google/adk/SchemaUtils.html\#method-detail)



- ### validateMapOnSchema [![Link icon](https://google.github.io/adk-docs/api-reference/java/resource-files/link.svg)](https://google.github.io/adk-docs/api-reference/java/com/google/adk/SchemaUtils.html\#validateMapOnSchema(java.util.Map,com.google.genai.types.Schema,java.lang.Boolean))





public staticvoidvalidateMapOnSchema( [Map](https://docs.oracle.com/en/java/javase/17/docs/api/java.base/java/util/Map.html "class or interface in java.util") < [String](https://docs.oracle.com/en/java/javase/17/docs/api/java.base/java/lang/String.html "class or interface in java.lang"), [Object](https://docs.oracle.com/en/java/javase/17/docs/api/java.base/java/lang/Object.html "class or interface in java.lang") > args,
com.google.genai.types.Schema schema,
[Boolean](https://docs.oracle.com/en/java/javase/17/docs/api/java.base/java/lang/Boolean.html "class or interface in java.lang") isInput)



Validates a map against a schema.

Parameters:`args` \- The map to validate.`schema` \- The schema to validate against.`isInput` \- Whether the map is an input or output.Throws:`IllegalArgumentException` \- If the map does not match the schema.

- ### validateOutputSchema [![Link icon](https://google.github.io/adk-docs/api-reference/java/resource-files/link.svg)](https://google.github.io/adk-docs/api-reference/java/com/google/adk/SchemaUtils.html\#validateOutputSchema(java.lang.String,com.google.genai.types.Schema))





public static[Map](https://docs.oracle.com/en/java/javase/17/docs/api/java.base/java/util/Map.html "class or interface in java.util") < [String](https://docs.oracle.com/en/java/javase/17/docs/api/java.base/java/lang/String.html "class or interface in java.lang"), [Object](https://docs.oracle.com/en/java/javase/17/docs/api/java.base/java/lang/Object.html "class or interface in java.lang") >validateOutputSchema( [String](https://docs.oracle.com/en/java/javase/17/docs/api/java.base/java/lang/String.html "class or interface in java.lang") output,
com.google.genai.types.Schema schema)
throws com.fasterxml.jackson.core.JsonProcessingException



Validates an output string against a schema.

Parameters:`output` \- The output string to validate.`schema` \- The schema to validate against.Returns:The output map.Throws:`IllegalArgumentException` \- If the output string does not match the schema.`com.fasterxml.jackson.core.JsonProcessingException` \- If the output string cannot be parsed.

## Claude AI Model
[java.lang.Object](https://docs.oracle.com/en/java/javase/17/docs/api/java.base/java/lang/Object.html "class or interface in java.lang")

[com.google.adk.models.BaseLlm](https://google.github.io/adk-docs/api-reference/java/com/google/adk/models/BaseLlm.html "class in com.google.adk.models")

com.google.adk.models.Claude

* * *

public class Claudeextends [BaseLlm](https://google.github.io/adk-docs/api-reference/java/com/google/adk/models/BaseLlm.html "class in com.google.adk.models")

Represents the Claude Generative AI model by Anthropic.



This class provides methods for interacting with Claude models. Streaming and live connections
are not currently supported for Claude.

- ## Constructor Summary [![Link icon](https://google.github.io/adk-docs/api-reference/java/resource-files/link.svg)](https://google.github.io/adk-docs/api-reference/java/com/google/adk/models/Claude.html\#constructor-summary)



Constructors





Constructor



Description



`Claude(String modelName,
com.anthropic.client.AnthropicClient anthropicClient)`





Constructs a new Claude instance.





`Claude(String modelName,
com.anthropic.client.AnthropicClient anthropicClient,
int maxTokens)`

- ## Method Summary [![Link icon](https://google.github.io/adk-docs/api-reference/java/resource-files/link.svg)](https://google.github.io/adk-docs/api-reference/java/com/google/adk/models/Claude.html\#method-summary)





All MethodsInstance MethodsConcrete Methods







Modifier and Type



Method



Description



`BaseLlmConnection`



`connect(LlmRequest llmRequest)`





Creates a live connection to the LLM.





`io.reactivex.rxjava3.core.Flowable<LlmResponse>`



`generateContent(LlmRequest llmRequest,
boolean stream)`





Generates one content from the given LLM request and tools.













### Methods inherited from class com.google.adk.models. [BaseLlm](https://google.github.io/adk-docs/api-reference/java/com/google/adk/models/BaseLlm.html "class in com.google.adk.models") [![Link icon](https://google.github.io/adk-docs/api-reference/java/resource-files/link.svg)](https://google.github.io/adk-docs/api-reference/java/com/google/adk/models/Claude.html\#methods-inherited-from-class-com.google.adk.models.BaseLlm)

`model`





### Methods inherited from class java.lang. [Object](https://docs.oracle.com/en/java/javase/17/docs/api/java.base/java/lang/Object.html "class or interface in java.lang") [![Link icon](https://google.github.io/adk-docs/api-reference/java/resource-files/link.svg)](https://google.github.io/adk-docs/api-reference/java/com/google/adk/models/Claude.html\#methods-inherited-from-class-java.lang.Object)

`clone, equals, finalize, getClass, hashCode, notify, notifyAll, toString, wait, wait, wait`


- ## Constructor Details [![Link icon](https://google.github.io/adk-docs/api-reference/java/resource-files/link.svg)](https://google.github.io/adk-docs/api-reference/java/com/google/adk/models/Claude.html\#constructor-detail)



- ### Claude [![Link icon](https://google.github.io/adk-docs/api-reference/java/resource-files/link.svg)](https://google.github.io/adk-docs/api-reference/java/com/google/adk/models/Claude.html\#%3Cinit%3E(java.lang.String,com.anthropic.client.AnthropicClient))





publicClaude( [String](https://docs.oracle.com/en/java/javase/17/docs/api/java.base/java/lang/String.html "class or interface in java.lang") modelName,
com.anthropic.client.AnthropicClient anthropicClient)



Constructs a new Claude instance.

Parameters:`modelName` \- The name of the Claude model to use (e.g., "claude-3-opus-20240229").`anthropicClient` \- The Anthropic API client instance.

- ### Claude [![Link icon](https://google.github.io/adk-docs/api-reference/java/resource-files/link.svg)](https://google.github.io/adk-docs/api-reference/java/com/google/adk/models/Claude.html\#%3Cinit%3E(java.lang.String,com.anthropic.client.AnthropicClient,int))





publicClaude( [String](https://docs.oracle.com/en/java/javase/17/docs/api/java.base/java/lang/String.html "class or interface in java.lang") modelName,
com.anthropic.client.AnthropicClient anthropicClient,
int maxTokens)


- ## Method Details [![Link icon](https://google.github.io/adk-docs/api-reference/java/resource-files/link.svg)](https://google.github.io/adk-docs/api-reference/java/com/google/adk/models/Claude.html\#method-detail)



- ### generateContent [![Link icon](https://google.github.io/adk-docs/api-reference/java/resource-files/link.svg)](https://google.github.io/adk-docs/api-reference/java/com/google/adk/models/Claude.html\#generateContent(com.google.adk.models.LlmRequest,boolean))





publicio.reactivex.rxjava3.core.Flowable< [LlmResponse](https://google.github.io/adk-docs/api-reference/java/com/google/adk/models/LlmResponse.html "class in com.google.adk.models") >generateContent( [LlmRequest](https://google.github.io/adk-docs/api-reference/java/com/google/adk/models/LlmRequest.html "class in com.google.adk.models") llmRequest,
boolean stream)



Description copied from class: `BaseLlm`



Generates one content from the given LLM request and tools.

Specified by:`generateContent` in class `BaseLlm`Parameters:`llmRequest` \- The LLM request containing the input prompt and parameters.`stream` \- A boolean flag indicating whether to stream the response.Returns:A Flowable of LlmResponses. For non-streaming calls, it will only yield one
LlmResponse. For streaming calls, it may yield more than one LlmResponse, but all yielded
LlmResponses should be treated as one content by merging their parts.

- ### connect [![Link icon](https://google.github.io/adk-docs/api-reference/java/resource-files/link.svg)](https://google.github.io/adk-docs/api-reference/java/com/google/adk/models/Claude.html\#connect(com.google.adk.models.LlmRequest))





public[BaseLlmConnection](https://google.github.io/adk-docs/api-reference/java/com/google/adk/models/BaseLlmConnection.html "interface in com.google.adk.models")connect( [LlmRequest](https://google.github.io/adk-docs/api-reference/java/com/google/adk/models/LlmRequest.html "class in com.google.adk.models") llmRequest)



Description copied from class: `BaseLlm`



Creates a live connection to the LLM.

Specified by:`connect` in class `BaseLlm`

## Audio Processing API
* * *

package com.google.adk.flows.llmflows.audio

- Related Packages





Package



Description



[com.google.adk.flows.llmflows](https://google.github.io/adk-docs/api-reference/java/com/google/adk/flows/llmflows/package-summary.html)

- All Classes and InterfacesInterfacesClasses







Class



Description



[SpeechClientInterface](https://google.github.io/adk-docs/api-reference/java/com/google/adk/flows/llmflows/audio/SpeechClientInterface.html "interface in com.google.adk.flows.llmflows.audio")





Interface for a speech-to-text client.





[VertexSpeechClient](https://google.github.io/adk-docs/api-reference/java/com/google/adk/flows/llmflows/audio/VertexSpeechClient.html "class in com.google.adk.flows.llmflows.audio")





Implementation of SpeechClientInterface using Vertex AI SpeechClient.

## Live WebSocket Handler
[java.lang.Object](https://docs.oracle.com/en/java/javase/17/docs/api/java.base/java/lang/Object.html "class or interface in java.lang")

org.springframework.web.socket.handler.AbstractWebSocketHandler

org.springframework.web.socket.handler.TextWebSocketHandler

com.google.adk.web.AdkWebServer.LiveWebSocketHandler

All Implemented Interfaces:`org.springframework.web.socket.WebSocketHandler`Enclosing class:`AdkWebServer`

* * *

@Component
public static class AdkWebServer.LiveWebSocketHandlerextends org.springframework.web.socket.handler.TextWebSocketHandler

WebSocket Handler for the /run\_live endpoint.



Manages bidirectional communication for live agent interactions. Assumes the
com.google.adk.runner.Runner class has a method: `public Flowable<Event> runLive(Session
session, Flowable<LiveRequest> liveRequests, List<String> modalities)`

- ## Constructor Summary [![Link icon](https://google.github.io/adk-docs/api-reference/java/resource-files/link.svg)](https://google.github.io/adk-docs/api-reference/java/com/google/adk/web/AdkWebServer.LiveWebSocketHandler.html\#constructor-summary)



Constructors





Constructor



Description



`LiveWebSocketHandler(com.fasterxml.jackson.databind.ObjectMapper objectMapper,
BaseSessionService sessionService,
AdkWebServer.RunnerService runnerService)`

- ## Method Summary [![Link icon](https://google.github.io/adk-docs/api-reference/java/resource-files/link.svg)](https://google.github.io/adk-docs/api-reference/java/com/google/adk/web/AdkWebServer.LiveWebSocketHandler.html\#method-summary)





All MethodsInstance MethodsConcrete Methods







Modifier and Type



Method



Description



`void`



`afterConnectionClosed(org.springframework.web.socket.WebSocketSession wsSession,
org.springframework.web.socket.CloseStatus status)`







`void`



`afterConnectionEstablished(org.springframework.web.socket.WebSocketSession wsSession)`







`protected void`



`handleTextMessage(org.springframework.web.socket.WebSocketSession wsSession,
org.springframework.web.socket.TextMessage message)`







`void`



`handleTransportError(org.springframework.web.socket.WebSocketSession wsSession,
Throwable exception)`















### Methods inherited from class org.springframework.web.socket.handler.TextWebSocketHandler [![Link icon](https://google.github.io/adk-docs/api-reference/java/resource-files/link.svg)](https://google.github.io/adk-docs/api-reference/java/com/google/adk/web/AdkWebServer.LiveWebSocketHandler.html\#methods-inherited-from-class-org.springframework.web.socket.handler.TextWebSocketHandler)

`handleBinaryMessage`





### Methods inherited from class org.springframework.web.socket.handler.AbstractWebSocketHandler [![Link icon](https://google.github.io/adk-docs/api-reference/java/resource-files/link.svg)](https://google.github.io/adk-docs/api-reference/java/com/google/adk/web/AdkWebServer.LiveWebSocketHandler.html\#methods-inherited-from-class-org.springframework.web.socket.handler.AbstractWebSocketHandler)

`handleMessage, handlePongMessage, supportsPartialMessages`





### Methods inherited from class java.lang. [Object](https://docs.oracle.com/en/java/javase/17/docs/api/java.base/java/lang/Object.html "class or interface in java.lang") [![Link icon](https://google.github.io/adk-docs/api-reference/java/resource-files/link.svg)](https://google.github.io/adk-docs/api-reference/java/com/google/adk/web/AdkWebServer.LiveWebSocketHandler.html\#methods-inherited-from-class-java.lang.Object)

`clone, equals, finalize, getClass, hashCode, notify, notifyAll, toString, wait, wait, wait`


- ## Constructor Details [![Link icon](https://google.github.io/adk-docs/api-reference/java/resource-files/link.svg)](https://google.github.io/adk-docs/api-reference/java/com/google/adk/web/AdkWebServer.LiveWebSocketHandler.html\#constructor-detail)



- ### LiveWebSocketHandler [![Link icon](https://google.github.io/adk-docs/api-reference/java/resource-files/link.svg)](https://google.github.io/adk-docs/api-reference/java/com/google/adk/web/AdkWebServer.LiveWebSocketHandler.html\#%3Cinit%3E(com.fasterxml.jackson.databind.ObjectMapper,com.google.adk.sessions.BaseSessionService,com.google.adk.web.AdkWebServer.RunnerService))





@Autowired
publicLiveWebSocketHandler(com.fasterxml.jackson.databind.ObjectMapper objectMapper,
[BaseSessionService](https://google.github.io/adk-docs/api-reference/java/com/google/adk/sessions/BaseSessionService.html "interface in com.google.adk.sessions") sessionService,
[AdkWebServer.RunnerService](https://google.github.io/adk-docs/api-reference/java/com/google/adk/web/AdkWebServer.RunnerService.html "class in com.google.adk.web") runnerService)


- ## Method Details [![Link icon](https://google.github.io/adk-docs/api-reference/java/resource-files/link.svg)](https://google.github.io/adk-docs/api-reference/java/com/google/adk/web/AdkWebServer.LiveWebSocketHandler.html\#method-detail)



- ### afterConnectionEstablished [![Link icon](https://google.github.io/adk-docs/api-reference/java/resource-files/link.svg)](https://google.github.io/adk-docs/api-reference/java/com/google/adk/web/AdkWebServer.LiveWebSocketHandler.html\#afterConnectionEstablished(org.springframework.web.socket.WebSocketSession))





publicvoidafterConnectionEstablished(org.springframework.web.socket.WebSocketSession wsSession)
throws [Exception](https://docs.oracle.com/en/java/javase/17/docs/api/java.base/java/lang/Exception.html "class or interface in java.lang")

Specified by:`afterConnectionEstablished` in interface `org.springframework.web.socket.WebSocketHandler`Overrides:`afterConnectionEstablished` in class `org.springframework.web.socket.handler.AbstractWebSocketHandler`Throws:`Exception`

- ### handleTextMessage [![Link icon](https://google.github.io/adk-docs/api-reference/java/resource-files/link.svg)](https://google.github.io/adk-docs/api-reference/java/com/google/adk/web/AdkWebServer.LiveWebSocketHandler.html\#handleTextMessage(org.springframework.web.socket.WebSocketSession,org.springframework.web.socket.TextMessage))





protectedvoidhandleTextMessage(org.springframework.web.socket.WebSocketSession wsSession,
org.springframework.web.socket.TextMessage message)
throws [Exception](https://docs.oracle.com/en/java/javase/17/docs/api/java.base/java/lang/Exception.html "class or interface in java.lang")

Overrides:`handleTextMessage` in class `org.springframework.web.socket.handler.AbstractWebSocketHandler`Throws:`Exception`

- ### handleTransportError [![Link icon](https://google.github.io/adk-docs/api-reference/java/resource-files/link.svg)](https://google.github.io/adk-docs/api-reference/java/com/google/adk/web/AdkWebServer.LiveWebSocketHandler.html\#handleTransportError(org.springframework.web.socket.WebSocketSession,java.lang.Throwable))





publicvoidhandleTransportError(org.springframework.web.socket.WebSocketSession wsSession,
[Throwable](https://docs.oracle.com/en/java/javase/17/docs/api/java.base/java/lang/Throwable.html "class or interface in java.lang") exception)
throws [Exception](https://docs.oracle.com/en/java/javase/17/docs/api/java.base/java/lang/Exception.html "class or interface in java.lang")

Specified by:`handleTransportError` in interface `org.springframework.web.socket.WebSocketHandler`Overrides:`handleTransportError` in class `org.springframework.web.socket.handler.AbstractWebSocketHandler`Throws:`Exception`

- ### afterConnectionClosed [![Link icon](https://google.github.io/adk-docs/api-reference/java/resource-files/link.svg)](https://google.github.io/adk-docs/api-reference/java/com/google/adk/web/AdkWebServer.LiveWebSocketHandler.html\#afterConnectionClosed(org.springframework.web.socket.WebSocketSession,org.springframework.web.socket.CloseStatus))





publicvoidafterConnectionClosed(org.springframework.web.socket.WebSocketSession wsSession,
org.springframework.web.socket.CloseStatus status)
throws [Exception](https://docs.oracle.com/en/java/javase/17/docs/api/java.base/java/lang/Exception.html "class or interface in java.lang")

Specified by:`afterConnectionClosed` in interface `org.springframework.web.socket.WebSocketHandler`Overrides:`afterConnectionClosed` in class `org.springframework.web.socket.handler.AbstractWebSocketHandler`Throws:`Exception`

## Base LLM Connection
All Known Implementing Classes:`GeminiLlmConnection`

* * *

public interface BaseLlmConnection

The base class for a live model connection.

- ## Method Summary [![Link icon](https://google.github.io/adk-docs/api-reference/java/resource-files/link.svg)](https://google.github.io/adk-docs/api-reference/java/com/google/adk/models/BaseLlmConnection.html\#method-summary)





All MethodsInstance MethodsAbstract Methods







Modifier and Type



Method



Description



`void`



`close()`





Closes the connection.





`void`



`close(Throwable throwable)`





Closes the connection with an error.





`io.reactivex.rxjava3.core.Flowable<LlmResponse>`



`receive()`





Receives the model responses.





`io.reactivex.rxjava3.core.Completable`



`sendContent(com.google.genai.types.Content content)`





Sends a user content to the model.





`io.reactivex.rxjava3.core.Completable`



`sendHistory(List<com.google.genai.types.Content> history)`





Sends the conversation history to the model.





`io.reactivex.rxjava3.core.Completable`



`sendRealtime(com.google.genai.types.Blob blob)`





Sends a chunk of audio or a frame of video to the model in realtime.


- ## Method Details [![Link icon](https://google.github.io/adk-docs/api-reference/java/resource-files/link.svg)](https://google.github.io/adk-docs/api-reference/java/com/google/adk/models/BaseLlmConnection.html\#method-detail)



- ### sendHistory [![Link icon](https://google.github.io/adk-docs/api-reference/java/resource-files/link.svg)](https://google.github.io/adk-docs/api-reference/java/com/google/adk/models/BaseLlmConnection.html\#sendHistory(java.util.List))





io.reactivex.rxjava3.core.CompletablesendHistory( [List](https://docs.oracle.com/en/java/javase/17/docs/api/java.base/java/util/List.html "class or interface in java.util") <com.google.genai.types.Content> history)



Sends the conversation history to the model.



You call this method right after setting up the model connection. The model will respond if
the last content is from user, otherwise it will wait for new user input before responding.

- ### sendContent [![Link icon](https://google.github.io/adk-docs/api-reference/java/resource-files/link.svg)](https://google.github.io/adk-docs/api-reference/java/com/google/adk/models/BaseLlmConnection.html\#sendContent(com.google.genai.types.Content))





io.reactivex.rxjava3.core.CompletablesendContent(com.google.genai.types.Content content)



Sends a user content to the model.



The model will respond immediately upon receiving the content. If you send function
responses, all parts in the content should be function responses.

- ### sendRealtime [![Link icon](https://google.github.io/adk-docs/api-reference/java/resource-files/link.svg)](https://google.github.io/adk-docs/api-reference/java/com/google/adk/models/BaseLlmConnection.html\#sendRealtime(com.google.genai.types.Blob))





io.reactivex.rxjava3.core.CompletablesendRealtime(com.google.genai.types.Blob blob)



Sends a chunk of audio or a frame of video to the model in realtime.



The model may not respond immediately upon receiving the blob. It will do voice activity
detection and decide when to respond.

- ### receive [![Link icon](https://google.github.io/adk-docs/api-reference/java/resource-files/link.svg)](https://google.github.io/adk-docs/api-reference/java/com/google/adk/models/BaseLlmConnection.html\#receive())





io.reactivex.rxjava3.core.Flowable< [LlmResponse](https://google.github.io/adk-docs/api-reference/java/com/google/adk/models/LlmResponse.html "class in com.google.adk.models") >receive()



Receives the model responses.

- ### close [![Link icon](https://google.github.io/adk-docs/api-reference/java/resource-files/link.svg)](https://google.github.io/adk-docs/api-reference/java/com/google/adk/models/BaseLlmConnection.html\#close())





voidclose()



Closes the connection.

- ### close [![Link icon](https://google.github.io/adk-docs/api-reference/java/resource-files/link.svg)](https://google.github.io/adk-docs/api-reference/java/com/google/adk/models/BaseLlmConnection.html\#close(java.lang.Throwable))





voidclose( [Throwable](https://docs.oracle.com/en/java/javase/17/docs/api/java.base/java/lang/Throwable.html "class or interface in java.lang") throwable)



Closes the connection with an error.

## ADK Web CORS Properties
[java.lang.Object](https://docs.oracle.com/en/java/javase/17/docs/api/java.base/java/lang/Object.html "class or interface in java.lang")

[java.lang.Record](https://docs.oracle.com/en/java/javase/17/docs/api/java.base/java/lang/Record.html "class or interface in java.lang")

com.google.adk.web.config.AdkWebCorsProperties

* * *

@ConfigurationProperties(prefix="adk.web.cors")
public record AdkWebCorsProperties( [String](https://docs.oracle.com/en/java/javase/17/docs/api/java.base/java/lang/String.html "class or interface in java.lang") mapping, [List](https://docs.oracle.com/en/java/javase/17/docs/api/java.base/java/util/List.html "class or interface in java.util") < [String](https://docs.oracle.com/en/java/javase/17/docs/api/java.base/java/lang/String.html "class or interface in java.lang") > origins, [List](https://docs.oracle.com/en/java/javase/17/docs/api/java.base/java/util/List.html "class or interface in java.util") < [String](https://docs.oracle.com/en/java/javase/17/docs/api/java.base/java/lang/String.html "class or interface in java.lang") > methods, [List](https://docs.oracle.com/en/java/javase/17/docs/api/java.base/java/util/List.html "class or interface in java.util") < [String](https://docs.oracle.com/en/java/javase/17/docs/api/java.base/java/lang/String.html "class or interface in java.lang") > headers, boolean allowCredentials, long maxAge)
extends [Record](https://docs.oracle.com/en/java/javase/17/docs/api/java.base/java/lang/Record.html "class or interface in java.lang")

Properties for configuring CORS in ADK Web. This class is used to load CORS settings from
application properties.

- ## Constructor Summary [![Link icon](https://google.github.io/adk-docs/api-reference/java/resource-files/link.svg)](https://google.github.io/adk-docs/api-reference/java/com/google/adk/web/config/AdkWebCorsProperties.html\#constructor-summary)



Constructors





Constructor



Description



`AdkWebCorsProperties(String mapping,
List<String> origins,
List<String> methods,
List<String> headers,
boolean allowCredentials,
long maxAge)`





Creates an instance of a `AdkWebCorsProperties` record class.

- ## Method Summary [![Link icon](https://google.github.io/adk-docs/api-reference/java/resource-files/link.svg)](https://google.github.io/adk-docs/api-reference/java/com/google/adk/web/config/AdkWebCorsProperties.html\#method-summary)





All MethodsInstance MethodsConcrete Methods







Modifier and Type



Method



Description



`boolean`



`allowCredentials()`





Returns the value of the `allowCredentials` record component.





`final boolean`



`equals(Object o)`





Indicates whether some other object is "equal to" this one.





`final int`



`hashCode()`





Returns a hash code value for this object.





`List<String>`



`headers()`





Returns the value of the `headers` record component.





`String`



`mapping()`





Returns the value of the `mapping` record component.





`long`



`maxAge()`





Returns the value of the `maxAge` record component.





`List<String>`



`methods()`





Returns the value of the `methods` record component.





`List<String>`



`origins()`





Returns the value of the `origins` record component.





`final String`



`toString()`





Returns a string representation of this record class.













### Methods inherited from class java.lang. [Object](https://docs.oracle.com/en/java/javase/17/docs/api/java.base/java/lang/Object.html "class or interface in java.lang") [![Link icon](https://google.github.io/adk-docs/api-reference/java/resource-files/link.svg)](https://google.github.io/adk-docs/api-reference/java/com/google/adk/web/config/AdkWebCorsProperties.html\#methods-inherited-from-class-java.lang.Object)

`clone, finalize, getClass, notify, notifyAll, wait, wait, wait`


- ## Constructor Details [![Link icon](https://google.github.io/adk-docs/api-reference/java/resource-files/link.svg)](https://google.github.io/adk-docs/api-reference/java/com/google/adk/web/config/AdkWebCorsProperties.html\#constructor-detail)



- ### AdkWebCorsProperties [![Link icon](https://google.github.io/adk-docs/api-reference/java/resource-files/link.svg)](https://google.github.io/adk-docs/api-reference/java/com/google/adk/web/config/AdkWebCorsProperties.html\#%3Cinit%3E(java.lang.String,java.util.List,java.util.List,java.util.List,boolean,long))





publicAdkWebCorsProperties( [String](https://docs.oracle.com/en/java/javase/17/docs/api/java.base/java/lang/String.html "class or interface in java.lang") mapping,
[List](https://docs.oracle.com/en/java/javase/17/docs/api/java.base/java/util/List.html "class or interface in java.util") < [String](https://docs.oracle.com/en/java/javase/17/docs/api/java.base/java/lang/String.html "class or interface in java.lang") > origins,
[List](https://docs.oracle.com/en/java/javase/17/docs/api/java.base/java/util/List.html "class or interface in java.util") < [String](https://docs.oracle.com/en/java/javase/17/docs/api/java.base/java/lang/String.html "class or interface in java.lang") > methods,
[List](https://docs.oracle.com/en/java/javase/17/docs/api/java.base/java/util/List.html "class or interface in java.util") < [String](https://docs.oracle.com/en/java/javase/17/docs/api/java.base/java/lang/String.html "class or interface in java.lang") > headers,
boolean allowCredentials,
long maxAge)



Creates an instance of a `AdkWebCorsProperties` record class.

Parameters:`mapping` \- the value for the `mapping` record component`origins` \- the value for the `origins` record component`methods` \- the value for the `methods` record component`headers` \- the value for the `headers` record component`allowCredentials` \- the value for the `allowCredentials` record component`maxAge` \- the value for the `maxAge` record component


- ## Method Details [![Link icon](https://google.github.io/adk-docs/api-reference/java/resource-files/link.svg)](https://google.github.io/adk-docs/api-reference/java/com/google/adk/web/config/AdkWebCorsProperties.html\#method-detail)



- ### toString [![Link icon](https://google.github.io/adk-docs/api-reference/java/resource-files/link.svg)](https://google.github.io/adk-docs/api-reference/java/com/google/adk/web/config/AdkWebCorsProperties.html\#toString())





public final[String](https://docs.oracle.com/en/java/javase/17/docs/api/java.base/java/lang/String.html "class or interface in java.lang")toString()



Returns a string representation of this record class. The representation contains the name of the class, followed by the name and value of each of the record components.

Specified by:`toString` in class `Record`Returns:a string representation of this object

- ### hashCode [![Link icon](https://google.github.io/adk-docs/api-reference/java/resource-files/link.svg)](https://google.github.io/adk-docs/api-reference/java/com/google/adk/web/config/AdkWebCorsProperties.html\#hashCode())





public finalinthashCode()



Returns a hash code value for this object. The value is derived from the hash code of each of the record components.

Specified by:`hashCode` in class `Record`Returns:a hash code value for this object

- ### equals [![Link icon](https://google.github.io/adk-docs/api-reference/java/resource-files/link.svg)](https://google.github.io/adk-docs/api-reference/java/com/google/adk/web/config/AdkWebCorsProperties.html\#equals(java.lang.Object))





public finalbooleanequals( [Object](https://docs.oracle.com/en/java/javase/17/docs/api/java.base/java/lang/Object.html "class or interface in java.lang") o)



Indicates whether some other object is "equal to" this one. The objects are equal if the other object is of the same class and if all the record components are equal. Reference components are compared with [`Objects::equals(Object,Object)`](https://docs.oracle.com/en/java/javase/17/docs/api/java.base/java/util/Objects.html#equals(java.lang.Object,java.lang.Object) "class or interface in java.util"); primitive components are compared with the `compare` method from their corresponding wrapper classes.

Specified by:`equals` in class `Record`Parameters:`o` \- the object with which to compareReturns:`true` if this object is the same as the `o` argument; `false` otherwise.

- ### mapping [![Link icon](https://google.github.io/adk-docs/api-reference/java/resource-files/link.svg)](https://google.github.io/adk-docs/api-reference/java/com/google/adk/web/config/AdkWebCorsProperties.html\#mapping())





public[String](https://docs.oracle.com/en/java/javase/17/docs/api/java.base/java/lang/String.html "class or interface in java.lang")mapping()



Returns the value of the `mapping` record component.

Returns:the value of the `mapping` record component

- ### origins [![Link icon](https://google.github.io/adk-docs/api-reference/java/resource-files/link.svg)](https://google.github.io/adk-docs/api-reference/java/com/google/adk/web/config/AdkWebCorsProperties.html\#origins())





public[List](https://docs.oracle.com/en/java/javase/17/docs/api/java.base/java/util/List.html "class or interface in java.util") < [String](https://docs.oracle.com/en/java/javase/17/docs/api/java.base/java/lang/String.html "class or interface in java.lang") >origins()



Returns the value of the `origins` record component.

Returns:the value of the `origins` record component

- ### methods [![Link icon](https://google.github.io/adk-docs/api-reference/java/resource-files/link.svg)](https://google.github.io/adk-docs/api-reference/java/com/google/adk/web/config/AdkWebCorsProperties.html\#methods())





public[List](https://docs.oracle.com/en/java/javase/17/docs/api/java.base/java/util/List.html "class or interface in java.util") < [String](https://docs.oracle.com/en/java/javase/17/docs/api/java.base/java/lang/String.html "class or interface in java.lang") >methods()



Returns the value of the `methods` record component.

Returns:the value of the `methods` record component

- ### headers [![Link icon](https://google.github.io/adk-docs/api-reference/java/resource-files/link.svg)](https://google.github.io/adk-docs/api-reference/java/com/google/adk/web/config/AdkWebCorsProperties.html\#headers())





public[List](https://docs.oracle.com/en/java/javase/17/docs/api/java.base/java/util/List.html "class or interface in java.util") < [String](https://docs.oracle.com/en/java/javase/17/docs/api/java.base/java/lang/String.html "class or interface in java.lang") >headers()



Returns the value of the `headers` record component.

Returns:the value of the `headers` record component

- ### allowCredentials [![Link icon](https://google.github.io/adk-docs/api-reference/java/resource-files/link.svg)](https://google.github.io/adk-docs/api-reference/java/com/google/adk/web/config/AdkWebCorsProperties.html\#allowCredentials())





publicbooleanallowCredentials()



Returns the value of the `allowCredentials` record component.

Returns:the value of the `allowCredentials` record component

- ### maxAge [![Link icon](https://google.github.io/adk-docs/api-reference/java/resource-files/link.svg)](https://google.github.io/adk-docs/api-reference/java/com/google/adk/web/config/AdkWebCorsProperties.html\#maxAge())





publiclongmaxAge()



Returns the value of the `maxAge` record component.

Returns:the value of the `maxAge` record component

## AdkWebServer Session Request
Packages that use [AdkWebServer.AddSessionToEvalSetRequest](https://google.github.io/adk-docs/api-reference/java/com/google/adk/web/AdkWebServer.AddSessionToEvalSetRequest.html "class in com.google.adk.web")

Package

Description

[com.google.adk.web](https://google.github.io/adk-docs/api-reference/java/com/google/adk/web/class-use/AdkWebServer.AddSessionToEvalSetRequest.html#com.google.adk.web)

- ## Uses of [AdkWebServer.AddSessionToEvalSetRequest](https://google.github.io/adk-docs/api-reference/java/com/google/adk/web/AdkWebServer.AddSessionToEvalSetRequest.html "class in com.google.adk.web") in [com.google.adk.web](https://google.github.io/adk-docs/api-reference/java/com/google/adk/web/package-summary.html) [![Link icon](https://google.github.io/adk-docs/api-reference/java/resource-files/link.svg)](https://google.github.io/adk-docs/api-reference/java/com/google/adk/web/class-use/AdkWebServer.AddSessionToEvalSetRequest.html\#com.google.adk.web)



Methods in [com.google.adk.web](https://google.github.io/adk-docs/api-reference/java/com/google/adk/web/package-summary.html) with parameters of type [AdkWebServer.AddSessionToEvalSetRequest](https://google.github.io/adk-docs/api-reference/java/com/google/adk/web/AdkWebServer.AddSessionToEvalSetRequest.html "class in com.google.adk.web")





Modifier and Type



Method



Description



`org.springframework.http.ResponseEntity<Object>`



AdkWebServer.AgentController.`addSessionToEvalSet(String appName,
String evalSetId,
AdkWebServer.AddSessionToEvalSetRequest req)`





Placeholder for adding a session to an evaluation set.

## ADK Package Hierarchies
Package Hierarchies:

- [All Packages](https://google.github.io/adk-docs/api-reference/java/overview-tree.html)

## Class Hierarchy

- java.lang. [Object](https://docs.oracle.com/en/java/javase/17/docs/api/java.base/java/lang/Object.html "class or interface in java.lang")
  - com.google.adk. [JsonBaseModel](https://google.github.io/adk-docs/api-reference/java/com/google/adk/JsonBaseModel.html "class in com.google.adk")
  - com.google.adk. [SchemaUtils](https://google.github.io/adk-docs/api-reference/java/com/google/adk/SchemaUtils.html "class in com.google.adk")
  - com.google.adk. [Telemetry](https://google.github.io/adk-docs/api-reference/java/com/google/adk/Telemetry.html "class in com.google.adk")
  - com.google.adk. [Version](https://google.github.io/adk-docs/api-reference/java/com/google/adk/Version.html "class in com.google.adk")

## ParallelAgent Builder Class
[java.lang.Object](https://docs.oracle.com/en/java/javase/17/docs/api/java.base/java/lang/Object.html "class or interface in java.lang")

com.google.adk.agents.ParallelAgent.Builder

Enclosing class:`ParallelAgent`

* * *

public static class ParallelAgent.Builderextends [Object](https://docs.oracle.com/en/java/javase/17/docs/api/java.base/java/lang/Object.html "class or interface in java.lang")

Builder for [`ParallelAgent`](https://google.github.io/adk-docs/api-reference/java/com/google/adk/agents/ParallelAgent.html "class in com.google.adk.agents").

- ## Constructor Summary [![Link icon](https://google.github.io/adk-docs/api-reference/java/resource-files/link.svg)](https://google.github.io/adk-docs/api-reference/java/com/google/adk/agents/ParallelAgent.Builder.html\#constructor-summary)



Constructors





Constructor



Description



`Builder()`

- ## Method Summary [![Link icon](https://google.github.io/adk-docs/api-reference/java/resource-files/link.svg)](https://google.github.io/adk-docs/api-reference/java/com/google/adk/agents/ParallelAgent.Builder.html\#method-summary)





All MethodsInstance MethodsConcrete Methods







Modifier and Type



Method



Description



`ParallelAgent.Builder`



`afterAgentCallback(Callbacks.AfterAgentCallback afterAgentCallback)`







`ParallelAgent.Builder`



`afterAgentCallback(List<com.google.adk.agents.Callbacks.AfterAgentCallbackBase> afterAgentCallback)`







`ParallelAgent.Builder`



`beforeAgentCallback(Callbacks.BeforeAgentCallback beforeAgentCallback)`







`ParallelAgent.Builder`



`beforeAgentCallback(List<com.google.adk.agents.Callbacks.BeforeAgentCallbackBase> beforeAgentCallback)`







`ParallelAgent`



`build()`







`ParallelAgent.Builder`



`description(String description)`







`ParallelAgent.Builder`



`name(String name)`







`ParallelAgent.Builder`



`subAgents(BaseAgent... subAgents)`







`ParallelAgent.Builder`



`subAgents(List<? extends BaseAgent> subAgents)`















### Methods inherited from class java.lang. [Object](https://docs.oracle.com/en/java/javase/17/docs/api/java.base/java/lang/Object.html "class or interface in java.lang") [![Link icon](https://google.github.io/adk-docs/api-reference/java/resource-files/link.svg)](https://google.github.io/adk-docs/api-reference/java/com/google/adk/agents/ParallelAgent.Builder.html\#methods-inherited-from-class-java.lang.Object)

`clone, equals, finalize, getClass, hashCode, notify, notifyAll, toString, wait, wait, wait`


- ## Constructor Details [![Link icon](https://google.github.io/adk-docs/api-reference/java/resource-files/link.svg)](https://google.github.io/adk-docs/api-reference/java/com/google/adk/agents/ParallelAgent.Builder.html\#constructor-detail)



- ### Builder [![Link icon](https://google.github.io/adk-docs/api-reference/java/resource-files/link.svg)](https://google.github.io/adk-docs/api-reference/java/com/google/adk/agents/ParallelAgent.Builder.html\#%3Cinit%3E())





publicBuilder()


- ## Method Details [![Link icon](https://google.github.io/adk-docs/api-reference/java/resource-files/link.svg)](https://google.github.io/adk-docs/api-reference/java/com/google/adk/agents/ParallelAgent.Builder.html\#method-detail)



- ### name [![Link icon](https://google.github.io/adk-docs/api-reference/java/resource-files/link.svg)](https://google.github.io/adk-docs/api-reference/java/com/google/adk/agents/ParallelAgent.Builder.html\#name(java.lang.String))





@CanIgnoreReturnValue
public[ParallelAgent.Builder](https://google.github.io/adk-docs/api-reference/java/com/google/adk/agents/ParallelAgent.Builder.html "class in com.google.adk.agents")name( [String](https://docs.oracle.com/en/java/javase/17/docs/api/java.base/java/lang/String.html "class or interface in java.lang") name)

- ### description [![Link icon](https://google.github.io/adk-docs/api-reference/java/resource-files/link.svg)](https://google.github.io/adk-docs/api-reference/java/com/google/adk/agents/ParallelAgent.Builder.html\#description(java.lang.String))





@CanIgnoreReturnValue
public[ParallelAgent.Builder](https://google.github.io/adk-docs/api-reference/java/com/google/adk/agents/ParallelAgent.Builder.html "class in com.google.adk.agents")description( [String](https://docs.oracle.com/en/java/javase/17/docs/api/java.base/java/lang/String.html "class or interface in java.lang") description)

- ### subAgents [![Link icon](https://google.github.io/adk-docs/api-reference/java/resource-files/link.svg)](https://google.github.io/adk-docs/api-reference/java/com/google/adk/agents/ParallelAgent.Builder.html\#subAgents(java.util.List))





@CanIgnoreReturnValue
public[ParallelAgent.Builder](https://google.github.io/adk-docs/api-reference/java/com/google/adk/agents/ParallelAgent.Builder.html "class in com.google.adk.agents")subAgents( [List](https://docs.oracle.com/en/java/javase/17/docs/api/java.base/java/util/List.html "class or interface in java.util") <? extends [BaseAgent](https://google.github.io/adk-docs/api-reference/java/com/google/adk/agents/BaseAgent.html "class in com.google.adk.agents") > subAgents)

- ### subAgents [![Link icon](https://google.github.io/adk-docs/api-reference/java/resource-files/link.svg)](https://google.github.io/adk-docs/api-reference/java/com/google/adk/agents/ParallelAgent.Builder.html\#subAgents(com.google.adk.agents.BaseAgent...))





@CanIgnoreReturnValue
public[ParallelAgent.Builder](https://google.github.io/adk-docs/api-reference/java/com/google/adk/agents/ParallelAgent.Builder.html "class in com.google.adk.agents")subAgents( [BaseAgent](https://google.github.io/adk-docs/api-reference/java/com/google/adk/agents/BaseAgent.html "class in com.google.adk.agents")... subAgents)

- ### beforeAgentCallback [![Link icon](https://google.github.io/adk-docs/api-reference/java/resource-files/link.svg)](https://google.github.io/adk-docs/api-reference/java/com/google/adk/agents/ParallelAgent.Builder.html\#beforeAgentCallback(com.google.adk.agents.Callbacks.BeforeAgentCallback))





@CanIgnoreReturnValue
public[ParallelAgent.Builder](https://google.github.io/adk-docs/api-reference/java/com/google/adk/agents/ParallelAgent.Builder.html "class in com.google.adk.agents")beforeAgentCallback( [Callbacks.BeforeAgentCallback](https://google.github.io/adk-docs/api-reference/java/com/google/adk/agents/Callbacks.BeforeAgentCallback.html "interface in com.google.adk.agents") beforeAgentCallback)

- ### beforeAgentCallback [![Link icon](https://google.github.io/adk-docs/api-reference/java/resource-files/link.svg)](https://google.github.io/adk-docs/api-reference/java/com/google/adk/agents/ParallelAgent.Builder.html\#beforeAgentCallback(java.util.List))





@CanIgnoreReturnValue
public[ParallelAgent.Builder](https://google.github.io/adk-docs/api-reference/java/com/google/adk/agents/ParallelAgent.Builder.html "class in com.google.adk.agents")beforeAgentCallback( [List](https://docs.oracle.com/en/java/javase/17/docs/api/java.base/java/util/List.html "class or interface in java.util") <com.google.adk.agents.Callbacks.BeforeAgentCallbackBase> beforeAgentCallback)

- ### afterAgentCallback [![Link icon](https://google.github.io/adk-docs/api-reference/java/resource-files/link.svg)](https://google.github.io/adk-docs/api-reference/java/com/google/adk/agents/ParallelAgent.Builder.html\#afterAgentCallback(com.google.adk.agents.Callbacks.AfterAgentCallback))





@CanIgnoreReturnValue
public[ParallelAgent.Builder](https://google.github.io/adk-docs/api-reference/java/com/google/adk/agents/ParallelAgent.Builder.html "class in com.google.adk.agents")afterAgentCallback( [Callbacks.AfterAgentCallback](https://google.github.io/adk-docs/api-reference/java/com/google/adk/agents/Callbacks.AfterAgentCallback.html "interface in com.google.adk.agents") afterAgentCallback)

- ### afterAgentCallback [![Link icon](https://google.github.io/adk-docs/api-reference/java/resource-files/link.svg)](https://google.github.io/adk-docs/api-reference/java/com/google/adk/agents/ParallelAgent.Builder.html\#afterAgentCallback(java.util.List))





@CanIgnoreReturnValue
public[ParallelAgent.Builder](https://google.github.io/adk-docs/api-reference/java/com/google/adk/agents/ParallelAgent.Builder.html "class in com.google.adk.agents")afterAgentCallback( [List](https://docs.oracle.com/en/java/javase/17/docs/api/java.base/java/util/List.html "class or interface in java.util") <com.google.adk.agents.Callbacks.AfterAgentCallbackBase> afterAgentCallback)

- ### build [![Link icon](https://google.github.io/adk-docs/api-reference/java/resource-files/link.svg)](https://google.github.io/adk-docs/api-reference/java/com/google/adk/agents/ParallelAgent.Builder.html\#build())





public[ParallelAgent](https://google.github.io/adk-docs/api-reference/java/com/google/adk/agents/ParallelAgent.html "class in com.google.adk.agents")build()

## LiveRequest Class Overview
[java.lang.Object](https://docs.oracle.com/en/java/javase/17/docs/api/java.base/java/lang/Object.html "class or interface in java.lang")

[com.google.adk.JsonBaseModel](https://google.github.io/adk-docs/api-reference/java/com/google/adk/JsonBaseModel.html "class in com.google.adk")

com.google.adk.agents.LiveRequest

* * *

public abstract class LiveRequestextends [JsonBaseModel](https://google.github.io/adk-docs/api-reference/java/com/google/adk/JsonBaseModel.html "class in com.google.adk")

Represents a request to be sent to a live connection to the LLM model.

- ## Nested Class Summary [![Link icon](https://google.github.io/adk-docs/api-reference/java/resource-files/link.svg)](https://google.github.io/adk-docs/api-reference/java/com/google/adk/agents/LiveRequest.html\#nested-class-summary)



Nested Classes





Modifier and Type



Class



Description



`static class`



`LiveRequest.Builder`





Builder for constructing [`LiveRequest`](https://google.github.io/adk-docs/api-reference/java/com/google/adk/agents/LiveRequest.html "class in com.google.adk.agents") instances.

- ## Method Summary [![Link icon](https://google.github.io/adk-docs/api-reference/java/resource-files/link.svg)](https://google.github.io/adk-docs/api-reference/java/com/google/adk/agents/LiveRequest.html\#method-summary)





All MethodsStatic MethodsInstance MethodsAbstract MethodsConcrete Methods







Modifier and Type



Method



Description



`abstract Optional<com.google.genai.types.Blob>`



`blob()`





Returns the blob of the request.





`static LiveRequest.Builder`



`builder()`







`abstract Optional<Boolean>`



`close()`





Returns whether the connection should be closed.





`abstract Optional<com.google.genai.types.Content>`



`content()`





Returns the content of the request.





`static LiveRequest`



`fromJsonString(String json)`





Deserializes a Json string to a [`LiveRequest`](https://google.github.io/adk-docs/api-reference/java/com/google/adk/agents/LiveRequest.html "class in com.google.adk.agents") object.





`boolean`



`shouldClose()`





Extracts boolean value from the close field or returns false if unset.





`abstract LiveRequest.Builder`



`toBuilder()`















### Methods inherited from class com.google.adk. [JsonBaseModel](https://google.github.io/adk-docs/api-reference/java/com/google/adk/JsonBaseModel.html "class in com.google.adk") [![Link icon](https://google.github.io/adk-docs/api-reference/java/resource-files/link.svg)](https://google.github.io/adk-docs/api-reference/java/com/google/adk/agents/LiveRequest.html\#methods-inherited-from-class-com.google.adk.JsonBaseModel)

`fromJsonNode, fromJsonString, getMapper, toJson, toJsonNode, toJsonString`





### Methods inherited from class java.lang. [Object](https://docs.oracle.com/en/java/javase/17/docs/api/java.base/java/lang/Object.html "class or interface in java.lang") [![Link icon](https://google.github.io/adk-docs/api-reference/java/resource-files/link.svg)](https://google.github.io/adk-docs/api-reference/java/com/google/adk/agents/LiveRequest.html\#methods-inherited-from-class-java.lang.Object)

`clone, equals, finalize, getClass, hashCode, notify, notifyAll, toString, wait, wait, wait`


- ## Method Details [![Link icon](https://google.github.io/adk-docs/api-reference/java/resource-files/link.svg)](https://google.github.io/adk-docs/api-reference/java/com/google/adk/agents/LiveRequest.html\#method-detail)



- ### content [![Link icon](https://google.github.io/adk-docs/api-reference/java/resource-files/link.svg)](https://google.github.io/adk-docs/api-reference/java/com/google/adk/agents/LiveRequest.html\#content())





public abstract[Optional](https://docs.oracle.com/en/java/javase/17/docs/api/java.base/java/util/Optional.html "class or interface in java.util") <com.google.genai.types.Content>content()



Returns the content of the request.



If set, send the content to the model in turn-by-turn mode.



Returns:An optional `Content` object containing the content of the request.

- ### blob [![Link icon](https://google.github.io/adk-docs/api-reference/java/resource-files/link.svg)](https://google.github.io/adk-docs/api-reference/java/com/google/adk/agents/LiveRequest.html\#blob())





public abstract[Optional](https://docs.oracle.com/en/java/javase/17/docs/api/java.base/java/util/Optional.html "class or interface in java.util") <com.google.genai.types.Blob>blob()



Returns the blob of the request.



If set, send the blob to the model in realtime mode.



Returns:An optional `Blob` object containing the blob of the request.

- ### close [![Link icon](https://google.github.io/adk-docs/api-reference/java/resource-files/link.svg)](https://google.github.io/adk-docs/api-reference/java/com/google/adk/agents/LiveRequest.html\#close())





public abstract[Optional](https://docs.oracle.com/en/java/javase/17/docs/api/java.base/java/util/Optional.html "class or interface in java.util") < [Boolean](https://docs.oracle.com/en/java/javase/17/docs/api/java.base/java/lang/Boolean.html "class or interface in java.lang") >close()



Returns whether the connection should be closed.



If set to true, the connection will be closed after the request is sent.



Returns:A boolean indicating whether the connection should be closed.

- ### shouldClose [![Link icon](https://google.github.io/adk-docs/api-reference/java/resource-files/link.svg)](https://google.github.io/adk-docs/api-reference/java/com/google/adk/agents/LiveRequest.html\#shouldClose())





publicbooleanshouldClose()



Extracts boolean value from the close field or returns false if unset.

- ### builder [![Link icon](https://google.github.io/adk-docs/api-reference/java/resource-files/link.svg)](https://google.github.io/adk-docs/api-reference/java/com/google/adk/agents/LiveRequest.html\#builder())





public static[LiveRequest.Builder](https://google.github.io/adk-docs/api-reference/java/com/google/adk/agents/LiveRequest.Builder.html "class in com.google.adk.agents")builder()

- ### toBuilder [![Link icon](https://google.github.io/adk-docs/api-reference/java/resource-files/link.svg)](https://google.github.io/adk-docs/api-reference/java/com/google/adk/agents/LiveRequest.html\#toBuilder())





public abstract[LiveRequest.Builder](https://google.github.io/adk-docs/api-reference/java/com/google/adk/agents/LiveRequest.Builder.html "class in com.google.adk.agents")toBuilder()

- ### fromJsonString [![Link icon](https://google.github.io/adk-docs/api-reference/java/resource-files/link.svg)](https://google.github.io/adk-docs/api-reference/java/com/google/adk/agents/LiveRequest.html\#fromJsonString(java.lang.String))





public static[LiveRequest](https://google.github.io/adk-docs/api-reference/java/com/google/adk/agents/LiveRequest.html "class in com.google.adk.agents")fromJsonString( [String](https://docs.oracle.com/en/java/javase/17/docs/api/java.base/java/lang/String.html "class or interface in java.lang") json)



Deserializes a Json string to a [`LiveRequest`](https://google.github.io/adk-docs/api-reference/java/com/google/adk/agents/LiveRequest.html "class in com.google.adk.agents") object.

## ADK Events Package
* * *

package com.google.adk.events

- Related Packages





Package



Description



[com.google.adk](https://google.github.io/adk-docs/api-reference/java/com/google/adk/package-summary.html)

- Classes





Class



Description



[Event](https://google.github.io/adk-docs/api-reference/java/com/google/adk/events/Event.html "class in com.google.adk.events")





Represents an event in a session.





[Event.Builder](https://google.github.io/adk-docs/api-reference/java/com/google/adk/events/Event.Builder.html "class in com.google.adk.events")





Builder for [`Event`](https://google.github.io/adk-docs/api-reference/java/com/google/adk/events/Event.html "class in com.google.adk.events").





[EventActions](https://google.github.io/adk-docs/api-reference/java/com/google/adk/events/EventActions.html "class in com.google.adk.events")





Represents the actions attached to an event.





[EventActions.Builder](https://google.github.io/adk-docs/api-reference/java/com/google/adk/events/EventActions.Builder.html "class in com.google.adk.events")





Builder for [`EventActions`](https://google.github.io/adk-docs/api-reference/java/com/google/adk/events/EventActions.html "class in com.google.adk.events").





[EventStream](https://google.github.io/adk-docs/api-reference/java/com/google/adk/events/EventStream.html "class in com.google.adk.events")





Iterable stream of [`Event`](https://google.github.io/adk-docs/api-reference/java/com/google/adk/events/Event.html "class in com.google.adk.events") objects.

## Android Utils Overview
Package Hierarchies:

- [All Packages](https://google.github.io/adk-docs/api-reference/java/overview-tree.html)

## Class Hierarchy

- java.lang. [Object](https://docs.oracle.com/en/java/javase/17/docs/api/java.base/java/lang/Object.html "class or interface in java.lang")
  - com.google.adk.utils. [CollectionUtils](https://google.github.io/adk-docs/api-reference/java/com/google/adk/utils/CollectionUtils.html "class in com.google.adk.utils")
  - com.google.adk.utils. [InstructionUtils](https://google.github.io/adk-docs/api-reference/java/com/google/adk/utils/InstructionUtils.html "class in com.google.adk.utils")
  - com.google.adk.utils. [Pairs](https://google.github.io/adk-docs/api-reference/java/com/google/adk/utils/Pairs.html "class in com.google.adk.utils")

## Callback Design Patterns
[Skip to content](https://google.github.io/adk-docs/callbacks/design-patterns-and-best-practices/#design-patterns-and-best-practices-for-callbacks)

[Edit this page](https://github.com/google/adk-docs/edit/main/docs/callbacks/design-patterns-and-best-practices.md "Edit this page") [View source of this page](https://github.com/google/adk-docs/raw/main/docs/callbacks/design-patterns-and-best-practices.md "View source of this page")

# Design Patterns and Best Practices for Callbacks [¶](https://google.github.io/adk-docs/callbacks/design-patterns-and-best-practices/\#design-patterns-and-best-practices-for-callbacks "Permanent link")

Callbacks offer powerful hooks into the agent lifecycle. Here are common design patterns illustrating how to leverage them effectively in ADK, followed by best practices for implementation.

## Design Patterns [¶](https://google.github.io/adk-docs/callbacks/design-patterns-and-best-practices/\#design-patterns "Permanent link")

These patterns demonstrate typical ways to enhance or control agent behavior using callbacks:

### 1\. Guardrails & Policy Enforcement [¶](https://google.github.io/adk-docs/callbacks/design-patterns-and-best-practices/\#guardrails-policy-enforcement "Permanent link")

**Pattern Overview:**
Intercept requests before they reach the LLM or tools to enforce rules.

**Implementation:**
\- Use `before_model_callback` to inspect the `LlmRequest` prompt
\- Use `before_tool_callback` to inspect tool arguments
\- If a policy violation is detected (e.g., forbidden topics, profanity):
\- Return a predefined response (`LlmResponse` or `dict`/`Map`) to block the operation
\- Optionally update `context.state` to log the violation

**Example Use Case:**
A `before_model_callback` checks `llm_request.contents` for sensitive keywords and returns a standard "Cannot process this request" `LlmResponse` if found, preventing the LLM call.

### 2\. Dynamic State Management [¶](https://google.github.io/adk-docs/callbacks/design-patterns-and-best-practices/\#dynamic-state-management "Permanent link")

**Pattern Overview:**
Read from and write to session state within callbacks to make agent behavior context-aware and pass data between steps.

**Implementation:**
\- Access `callback_context.state` or `tool_context.state`
\- Modifications (`state['key'] = value`) are automatically tracked in the subsequent `Event.actions.state_delta`
\- Changes are persisted by the `SessionService`

**Example Use Case:**
An `after_tool_callback` saves a `transaction_id` from the tool's result to `tool_context.state['last_transaction_id']`. A later `before_agent_callback` might read `state['user_tier']` to customize the agent's greeting.

### 3\. Logging and Monitoring [¶](https://google.github.io/adk-docs/callbacks/design-patterns-and-best-practices/\#logging-and-monitoring "Permanent link")

**Pattern Overview:**
Add detailed logging at specific lifecycle points for observability and debugging.

**Implementation:**
\- Implement callbacks (e.g., `before_agent_callback`, `after_tool_callback`, `after_model_callback`)
\- Print or send structured logs containing:
\- Agent name
\- Tool name
\- Invocation ID
\- Relevant data from the context or arguments

**Example Use Case:**
Log messages like `INFO: [Invocation: e-123] Before Tool: search_api - Args: {'query': 'ADK'}`.

### 4\. Caching [¶](https://google.github.io/adk-docs/callbacks/design-patterns-and-best-practices/\#caching "Permanent link")

**Pattern Overview:**
Avoid redundant LLM calls or tool executions by caching results.

**Implementation Steps:**
1\. **Before Operation:** In `before_model_callback` or `before_tool_callback`:
\- Generate a cache key based on the request/arguments
\- Check `context.state` (or an external cache) for this key
\- If found, return the cached `LlmResponse` or result directly

1. **After Operation:** If cache miss occurred:
2. Use the corresponding `after_` callback to store the new result in the cache using the key

**Example Use Case:**`before_tool_callback` for `get_stock_price(symbol)` checks `state[f"cache:stock:{symbol}"]`. If present, returns the cached price; otherwise, allows the API call and `after_tool_callback` saves the result to the state key.

### 5\. Request/Response Modification [¶](https://google.github.io/adk-docs/callbacks/design-patterns-and-best-practices/\#request-response-modification "Permanent link")

**Pattern Overview:**
Alter data just before it's sent to the LLM/tool or just after it's received.

**Implementation Options:**
\- **`before_model_callback`:** Modify `llm_request` (e.g., add system instructions based on `state`)
\- **`after_model_callback`:** Modify the returned `LlmResponse` (e.g., format text, filter content)
\- **`before_tool_callback`:** Modify the tool `args` dictionary (or Map in Java)
\- **`after_tool_callback`:** Modify the `tool_response` dictionary (or Map in Java)

**Example Use Case:**`before_model_callback` appends "User language preference: Spanish" to `llm_request.config.system_instruction` if `context.state['lang'] == 'es'`.

### 6\. Conditional Skipping of Steps [¶](https://google.github.io/adk-docs/callbacks/design-patterns-and-best-practices/\#conditional-skipping-of-steps "Permanent link")

**Pattern Overview:**
Prevent standard operations (agent run, LLM call, tool execution) based on certain conditions.

**Implementation:**
\- Return a value from a `before_` callback to skip the normal execution:
\- `Content` from `before_agent_callback`
\- `LlmResponse` from `before_model_callback`
\- `dict` from `before_tool_callback`
\- The framework interprets this returned value as the result for that step

**Example Use Case:**`before_tool_callback` checks `tool_context.state['api_quota_exceeded']`. If `True`, it returns `{'error': 'API quota exceeded'}`, preventing the actual tool function from running.

### 7\. Tool-Specific Actions (Authentication & Summarization Control) [¶](https://google.github.io/adk-docs/callbacks/design-patterns-and-best-practices/\#tool-specific-actions-authentication-summarization-control "Permanent link")

**Pattern Overview:**
Handle actions specific to the tool lifecycle, primarily authentication and controlling LLM summarization of tool results.

**Implementation:**
Use `ToolContext` within tool callbacks (`before_tool_callback`, `after_tool_callback`):

- **Authentication:** Call `tool_context.request_credential(auth_config)` in `before_tool_callback` if credentials are required but not found (e.g., via `tool_context.get_auth_response` or state check). This initiates the auth flow.
- **Summarization:** Set `tool_context.actions.skip_summarization = True` if the raw dictionary output of the tool should be passed back to the LLM or potentially displayed directly, bypassing the default LLM summarization step.

**Example Use Case:**
A `before_tool_callback` for a secure API checks for an auth token in state; if missing, it calls `request_credential`. An `after_tool_callback` for a tool returning structured JSON might set `skip_summarization = True`.

### 8\. Artifact Handling [¶](https://google.github.io/adk-docs/callbacks/design-patterns-and-best-practices/\#artifact-handling "Permanent link")

**Pattern Overview:**
Save or load session-related files or large data blobs during the agent lifecycle.

**Implementation:**
\- **Saving:** Use `callback_context.save_artifact` / `await tool_context.save_artifact` to store data:
\- Generated reports
\- Logs
\- Intermediate data
\- **Loading:** Use `load_artifact` to retrieve previously stored artifacts
\- **Tracking:** Changes are tracked via `Event.actions.artifact_delta`

**Example Use Case:**
An `after_tool_callback` for a "generate\_report" tool saves the output file using `await tool_context.save_artifact("report.pdf", report_part)`. A `before_agent_callback` might load a configuration artifact using `callback_context.load_artifact("agent_config.json")`.

## Best Practices for Callbacks [¶](https://google.github.io/adk-docs/callbacks/design-patterns-and-best-practices/\#best-practices-for-callbacks "Permanent link")

### Design Principles [¶](https://google.github.io/adk-docs/callbacks/design-patterns-and-best-practices/\#design-principles "Permanent link")

**Keep Focused:**
Design each callback for a single, well-defined purpose (e.g., just logging, just validation). Avoid monolithic callbacks.

**Mind Performance:**
Callbacks execute synchronously within the agent's processing loop. Avoid long-running or blocking operations (network calls, heavy computation). Offload if necessary, but be aware this adds complexity.

### Error Handling [¶](https://google.github.io/adk-docs/callbacks/design-patterns-and-best-practices/\#error-handling "Permanent link")

**Handle Errors Gracefully:**
\- Use `try...except/catch` blocks within your callback functions
\- Log errors appropriately
\- Decide if the agent invocation should halt or attempt recovery
\- Don't let callback errors crash the entire process

### State Management [¶](https://google.github.io/adk-docs/callbacks/design-patterns-and-best-practices/\#state-management "Permanent link")

**Manage State Carefully:**
\- Be deliberate about reading from and writing to `context.state`
\- Changes are immediately visible within the _current_ invocation and persisted at the end of the event processing
\- Use specific state keys rather than modifying broad structures to avoid unintended side effects
\- Consider using state prefixes (`State.APP_PREFIX`, `State.USER_PREFIX`, `State.TEMP_PREFIX`) for clarity, especially with persistent `SessionService` implementations

### Reliability [¶](https://google.github.io/adk-docs/callbacks/design-patterns-and-best-practices/\#reliability "Permanent link")

**Consider Idempotency:**
If a callback performs actions with external side effects (e.g., incrementing an external counter), design it to be idempotent (safe to run multiple times with the same input) if possible, to handle potential retries in the framework or your application.

### Testing & Documentation [¶](https://google.github.io/adk-docs/callbacks/design-patterns-and-best-practices/\#testing-documentation "Permanent link")

**Test Thoroughly:**
\- Unit test your callback functions using mock context objects
\- Perform integration tests to ensure callbacks function correctly within the full agent flow

**Ensure Clarity:**
\- Use descriptive names for your callback functions
\- Add clear docstrings explaining their purpose, when they run, and any side effects (especially state modifications)

**Use Correct Context Type:**
Always use the specific context type provided (`CallbackContext` for agent/model, `ToolContext` for tools) to ensure access to the appropriate methods and properties.

By applying these patterns and best practices, you can effectively use callbacks to create more robust, observable, and customized agent behaviors in ADK.

Back to top

## AgentTool Usage Overview
Packages that use [AgentTool](https://google.github.io/adk-docs/api-reference/java/com/google/adk/tools/AgentTool.html "class in com.google.adk.tools")

Package

Description

[com.google.adk.tools](https://google.github.io/adk-docs/api-reference/java/com/google/adk/tools/class-use/AgentTool.html#com.google.adk.tools)

- ## Uses of [AgentTool](https://google.github.io/adk-docs/api-reference/java/com/google/adk/tools/AgentTool.html "class in com.google.adk.tools") in [com.google.adk.tools](https://google.github.io/adk-docs/api-reference/java/com/google/adk/tools/package-summary.html) [![Link icon](https://google.github.io/adk-docs/api-reference/java/resource-files/link.svg)](https://google.github.io/adk-docs/api-reference/java/com/google/adk/tools/class-use/AgentTool.html\#com.google.adk.tools)



Methods in [com.google.adk.tools](https://google.github.io/adk-docs/api-reference/java/com/google/adk/tools/package-summary.html) that return [AgentTool](https://google.github.io/adk-docs/api-reference/java/com/google/adk/tools/AgentTool.html "class in com.google.adk.tools")





Modifier and Type



Method



Description



`static AgentTool`



AgentTool.`create(BaseAgent agent)`







`static AgentTool`



AgentTool.`create(BaseAgent agent,
boolean skipSummarization)`

## ListEventsResponse.Builder Overview
Packages that use [ListEventsResponse.Builder](https://google.github.io/adk-docs/api-reference/java/com/google/adk/sessions/ListEventsResponse.Builder.html "class in com.google.adk.sessions")

Package

Description

[com.google.adk.sessions](https://google.github.io/adk-docs/api-reference/java/com/google/adk/sessions/class-use/ListEventsResponse.Builder.html#com.google.adk.sessions)

- ## Uses of [ListEventsResponse.Builder](https://google.github.io/adk-docs/api-reference/java/com/google/adk/sessions/ListEventsResponse.Builder.html "class in com.google.adk.sessions") in [com.google.adk.sessions](https://google.github.io/adk-docs/api-reference/java/com/google/adk/sessions/package-summary.html) [![Link icon](https://google.github.io/adk-docs/api-reference/java/resource-files/link.svg)](https://google.github.io/adk-docs/api-reference/java/com/google/adk/sessions/class-use/ListEventsResponse.Builder.html\#com.google.adk.sessions)



Methods in [com.google.adk.sessions](https://google.github.io/adk-docs/api-reference/java/com/google/adk/sessions/package-summary.html) that return [ListEventsResponse.Builder](https://google.github.io/adk-docs/api-reference/java/com/google/adk/sessions/ListEventsResponse.Builder.html "class in com.google.adk.sessions")





Modifier and Type



Method



Description



`static ListEventsResponse.Builder`



ListEventsResponse.`builder()`







`abstract ListEventsResponse.Builder`



ListEventsResponse.Builder.`events(List<Event> events)`







`abstract ListEventsResponse.Builder`



ListEventsResponse.Builder.`nextPageToken(String nextPageToken)`

## Model Class Overview
[java.lang.Object](https://docs.oracle.com/en/java/javase/17/docs/api/java.base/java/lang/Object.html "class or interface in java.lang")

com.google.adk.models.Model

* * *

public abstract class Modelextends [Object](https://docs.oracle.com/en/java/javase/17/docs/api/java.base/java/lang/Object.html "class or interface in java.lang")

Represents a model by name or instance.

- ## Nested Class Summary [![Link icon](https://google.github.io/adk-docs/api-reference/java/resource-files/link.svg)](https://google.github.io/adk-docs/api-reference/java/com/google/adk/models/Model.html\#nested-class-summary)



Nested Classes





Modifier and Type



Class



Description



`static class`



`Model.Builder`





Builder for [`Model`](https://google.github.io/adk-docs/api-reference/java/com/google/adk/models/Model.html "class in com.google.adk.models").

- ## Constructor Summary [![Link icon](https://google.github.io/adk-docs/api-reference/java/resource-files/link.svg)](https://google.github.io/adk-docs/api-reference/java/com/google/adk/models/Model.html\#constructor-summary)



Constructors





Constructor



Description



`Model()`

- ## Method Summary [![Link icon](https://google.github.io/adk-docs/api-reference/java/resource-files/link.svg)](https://google.github.io/adk-docs/api-reference/java/com/google/adk/models/Model.html\#method-summary)





All MethodsStatic MethodsInstance MethodsAbstract MethodsConcrete Methods







Modifier and Type



Method



Description



`static Model.Builder`



`builder()`







`abstract Optional<BaseLlm>`



`model()`







`abstract Optional<String>`



`modelName()`







`abstract Model.Builder`



`toBuilder()`















### Methods inherited from class java.lang. [Object](https://docs.oracle.com/en/java/javase/17/docs/api/java.base/java/lang/Object.html "class or interface in java.lang") [![Link icon](https://google.github.io/adk-docs/api-reference/java/resource-files/link.svg)](https://google.github.io/adk-docs/api-reference/java/com/google/adk/models/Model.html\#methods-inherited-from-class-java.lang.Object)

`clone, equals, finalize, getClass, hashCode, notify, notifyAll, toString, wait, wait, wait`


- ## Constructor Details [![Link icon](https://google.github.io/adk-docs/api-reference/java/resource-files/link.svg)](https://google.github.io/adk-docs/api-reference/java/com/google/adk/models/Model.html\#constructor-detail)



- ### Model [![Link icon](https://google.github.io/adk-docs/api-reference/java/resource-files/link.svg)](https://google.github.io/adk-docs/api-reference/java/com/google/adk/models/Model.html\#%3Cinit%3E())





publicModel()


- ## Method Details [![Link icon](https://google.github.io/adk-docs/api-reference/java/resource-files/link.svg)](https://google.github.io/adk-docs/api-reference/java/com/google/adk/models/Model.html\#method-detail)



- ### modelName [![Link icon](https://google.github.io/adk-docs/api-reference/java/resource-files/link.svg)](https://google.github.io/adk-docs/api-reference/java/com/google/adk/models/Model.html\#modelName())





public abstract[Optional](https://docs.oracle.com/en/java/javase/17/docs/api/java.base/java/util/Optional.html "class or interface in java.util") < [String](https://docs.oracle.com/en/java/javase/17/docs/api/java.base/java/lang/String.html "class or interface in java.lang") >modelName()

- ### model [![Link icon](https://google.github.io/adk-docs/api-reference/java/resource-files/link.svg)](https://google.github.io/adk-docs/api-reference/java/com/google/adk/models/Model.html\#model())





public abstract[Optional](https://docs.oracle.com/en/java/javase/17/docs/api/java.base/java/util/Optional.html "class or interface in java.util") < [BaseLlm](https://google.github.io/adk-docs/api-reference/java/com/google/adk/models/BaseLlm.html "class in com.google.adk.models") >model()

- ### builder [![Link icon](https://google.github.io/adk-docs/api-reference/java/resource-files/link.svg)](https://google.github.io/adk-docs/api-reference/java/com/google/adk/models/Model.html\#builder())





public static[Model.Builder](https://google.github.io/adk-docs/api-reference/java/com/google/adk/models/Model.Builder.html "class in com.google.adk.models")builder()

- ### toBuilder [![Link icon](https://google.github.io/adk-docs/api-reference/java/resource-files/link.svg)](https://google.github.io/adk-docs/api-reference/java/com/google/adk/models/Model.html\#toBuilder())





public abstract[Model.Builder](https://google.github.io/adk-docs/api-reference/java/com/google/adk/models/Model.Builder.html "class in com.google.adk.models")toBuilder()

## Google ADK Python API
ContentsMenuExpandLight modeDark modeAuto light/dark, in light modeAuto light/dark, in dark mode[Skip to content](https://google.github.io/adk-docs/api-reference/python/#furo-main-content)

[Back to top](https://google.github.io/adk-docs/api-reference/python/#)

[View this page](https://google.github.io/adk-docs/api-reference/python/_sources/index.rst.txt "View this page")

# google [¶](https://google.github.io/adk-docs/api-reference/python/\#google "Link to this heading")

- [Submodules](https://google.github.io/adk-docs/api-reference/python/google-adk.html)
- [google.adk.a2a module](https://google.github.io/adk-docs/api-reference/python/google-adk.html#module-google.adk.a2a)
- [google.adk.agents module](https://google.github.io/adk-docs/api-reference/python/google-adk.html#module-google.adk.agents)
  - [`Agent`](https://google.github.io/adk-docs/api-reference/python/google-adk.html#google.adk.agents.Agent)
  - [`BaseAgent`](https://google.github.io/adk-docs/api-reference/python/google-adk.html#google.adk.agents.BaseAgent)
    - [`BaseAgent.after_agent_callback`](https://google.github.io/adk-docs/api-reference/python/google-adk.html#google.adk.agents.BaseAgent.after_agent_callback)
    - [`BaseAgent.before_agent_callback`](https://google.github.io/adk-docs/api-reference/python/google-adk.html#google.adk.agents.BaseAgent.before_agent_callback)
    - [`BaseAgent.description`](https://google.github.io/adk-docs/api-reference/python/google-adk.html#google.adk.agents.BaseAgent.description)
    - [`BaseAgent.name`](https://google.github.io/adk-docs/api-reference/python/google-adk.html#google.adk.agents.BaseAgent.name)
    - [`BaseAgent.parent_agent`](https://google.github.io/adk-docs/api-reference/python/google-adk.html#google.adk.agents.BaseAgent.parent_agent)
    - [`BaseAgent.sub_agents`](https://google.github.io/adk-docs/api-reference/python/google-adk.html#google.adk.agents.BaseAgent.sub_agents)
    - [`BaseAgent.config_type`](https://google.github.io/adk-docs/api-reference/python/google-adk.html#google.adk.agents.BaseAgent.config_type)
    - [`BaseAgent.from_config()`](https://google.github.io/adk-docs/api-reference/python/google-adk.html#google.adk.agents.BaseAgent.from_config)
    - [`BaseAgent.validate_name`](https://google.github.io/adk-docs/api-reference/python/google-adk.html#google.adk.agents.BaseAgent.validate_name)
    - [`BaseAgent.clone()`](https://google.github.io/adk-docs/api-reference/python/google-adk.html#google.adk.agents.BaseAgent.clone)
    - [`BaseAgent.find_agent()`](https://google.github.io/adk-docs/api-reference/python/google-adk.html#google.adk.agents.BaseAgent.find_agent)
    - [`BaseAgent.find_sub_agent()`](https://google.github.io/adk-docs/api-reference/python/google-adk.html#google.adk.agents.BaseAgent.find_sub_agent)
    - [`BaseAgent.model_post_init()`](https://google.github.io/adk-docs/api-reference/python/google-adk.html#google.adk.agents.BaseAgent.model_post_init)
    - [`BaseAgent.run_async()`](https://google.github.io/adk-docs/api-reference/python/google-adk.html#google.adk.agents.BaseAgent.run_async)
    - [`BaseAgent.run_live()`](https://google.github.io/adk-docs/api-reference/python/google-adk.html#google.adk.agents.BaseAgent.run_live)
    - [`BaseAgent.canonical_after_agent_callbacks`](https://google.github.io/adk-docs/api-reference/python/google-adk.html#google.adk.agents.BaseAgent.canonical_after_agent_callbacks)
    - [`BaseAgent.canonical_before_agent_callbacks`](https://google.github.io/adk-docs/api-reference/python/google-adk.html#google.adk.agents.BaseAgent.canonical_before_agent_callbacks)
    - [`BaseAgent.root_agent`](https://google.github.io/adk-docs/api-reference/python/google-adk.html#google.adk.agents.BaseAgent.root_agent)
  - [`InvocationContext`](https://google.github.io/adk-docs/api-reference/python/google-adk.html#google.adk.agents.InvocationContext)
    - [`InvocationContext.active_streaming_tools`](https://google.github.io/adk-docs/api-reference/python/google-adk.html#google.adk.agents.InvocationContext.active_streaming_tools)
    - [`InvocationContext.agent`](https://google.github.io/adk-docs/api-reference/python/google-adk.html#google.adk.agents.InvocationContext.agent)
    - [`InvocationContext.agent_states`](https://google.github.io/adk-docs/api-reference/python/google-adk.html#google.adk.agents.InvocationContext.agent_states)
    - [`InvocationContext.artifact_service`](https://google.github.io/adk-docs/api-reference/python/google-adk.html#google.adk.agents.InvocationContext.artifact_service)
    - [`InvocationContext.branch`](https://google.github.io/adk-docs/api-reference/python/google-adk.html#google.adk.agents.InvocationContext.branch)
    - [`InvocationContext.canonical_tools_cache`](https://google.github.io/adk-docs/api-reference/python/google-adk.html#google.adk.agents.InvocationContext.canonical_tools_cache)
    - [`InvocationContext.context_cache_config`](https://google.github.io/adk-docs/api-reference/python/google-adk.html#google.adk.agents.InvocationContext.context_cache_config)
    - [`InvocationContext.credential_service`](https://google.github.io/adk-docs/api-reference/python/google-adk.html#google.adk.agents.InvocationContext.credential_service)
    - [`InvocationContext.end_invocation`](https://google.github.io/adk-docs/api-reference/python/google-adk.html#google.adk.agents.InvocationContext.end_invocation)
    - [`InvocationContext.end_of_agents`](https://google.github.io/adk-docs/api-reference/python/google-adk.html#google.adk.agents.InvocationContext.end_of_agents)
    - [`InvocationContext.input_realtime_cache`](https://google.github.io/adk-docs/api-reference/python/google-adk.html#google.adk.agents.InvocationContext.input_realtime_cache)
    - [`InvocationContext.invocation_id`](https://google.github.io/adk-docs/api-reference/python/google-adk.html#google.adk.agents.InvocationContext.invocation_id)
    - [`InvocationContext.live_request_queue`](https://google.github.io/adk-docs/api-reference/python/google-adk.html#google.adk.agents.InvocationContext.live_request_queue)
    - [`InvocationContext.live_session_resumption_handle`](https://google.github.io/adk-docs/api-reference/python/google-adk.html#google.adk.agents.InvocationContext.live_session_resumption_handle)
    - [`InvocationContext.memory_service`](https://google.github.io/adk-docs/api-reference/python/google-adk.html#google.adk.agents.InvocationContext.memory_service)
    - [`InvocationContext.output_realtime_cache`](https://google.github.io/adk-docs/api-reference/python/google-adk.html#google.adk.agents.InvocationContext.output_realtime_cache)
    - [`InvocationContext.plugin_manager`](https://google.github.io/adk-docs/api-reference/python/google-adk.html#google.adk.agents.InvocationContext.plugin_manager)
    - [`InvocationContext.resumability_config`](https://google.github.io/adk-docs/api-reference/python/google-adk.html#google.adk.agents.InvocationContext.resumability_config)
    - [`InvocationContext.run_config`](https://google.github.io/adk-docs/api-reference/python/google-adk.html#google.adk.agents.InvocationContext.run_config)
    - [`InvocationContext.session`](https://google.github.io/adk-docs/api-reference/python/google-adk.html#google.adk.agents.InvocationContext.session)
    - [`InvocationContext.session_service`](https://google.github.io/adk-docs/api-reference/python/google-adk.html#google.adk.agents.InvocationContext.session_service)
    - [`InvocationContext.transcription_cache`](https://google.github.io/adk-docs/api-reference/python/google-adk.html#google.adk.agents.InvocationContext.transcription_cache)
    - [`InvocationContext.user_content`](https://google.github.io/adk-docs/api-reference/python/google-adk.html#google.adk.agents.InvocationContext.user_content)
    - [`InvocationContext.increment_llm_call_count()`](https://google.github.io/adk-docs/api-reference/python/google-adk.html#google.adk.agents.InvocationContext.increment_llm_call_count)
    - [`InvocationContext.model_post_init()`](https://google.github.io/adk-docs/api-reference/python/google-adk.html#google.adk.agents.InvocationContext.model_post_init)
    - [`InvocationContext.populate_invocation_agent_states()`](https://google.github.io/adk-docs/api-reference/python/google-adk.html#google.adk.agents.InvocationContext.populate_invocation_agent_states)
    - [`InvocationContext.reset_sub_agent_states()`](https://google.github.io/adk-docs/api-reference/python/google-adk.html#google.adk.agents.InvocationContext.reset_sub_agent_states)
    - [`InvocationContext.set_agent_state()`](https://google.github.io/adk-docs/api-reference/python/google-adk.html#google.adk.agents.InvocationContext.set_agent_state)
    - [`InvocationContext.should_pause_invocation()`](https://google.github.io/adk-docs/api-reference/python/google-adk.html#google.adk.agents.InvocationContext.should_pause_invocation)
    - [`InvocationContext.app_name`](https://google.github.io/adk-docs/api-reference/python/google-adk.html#google.adk.agents.InvocationContext.app_name)
    - [`InvocationContext.is_resumable`](https://google.github.io/adk-docs/api-reference/python/google-adk.html#google.adk.agents.InvocationContext.is_resumable)
    - [`InvocationContext.user_id`](https://google.github.io/adk-docs/api-reference/python/google-adk.html#google.adk.agents.InvocationContext.user_id)
  - [`LiveRequest`](https://google.github.io/adk-docs/api-reference/python/google-adk.html#google.adk.agents.LiveRequest)
    - [`LiveRequest.activity_end`](https://google.github.io/adk-docs/api-reference/python/google-adk.html#google.adk.agents.LiveRequest.activity_end)
    - [`LiveRequest.activity_start`](https://google.github.io/adk-docs/api-reference/python/google-adk.html#google.adk.agents.LiveRequest.activity_start)
    - [`LiveRequest.blob`](https://google.github.io/adk-docs/api-reference/python/google-adk.html#google.adk.agents.LiveRequest.blob)
    - [`LiveRequest.close`](https://google.github.io/adk-docs/api-reference/python/google-adk.html#google.adk.agents.LiveRequest.close)
    - [`LiveRequest.content`](https://google.github.io/adk-docs/api-reference/python/google-adk.html#google.adk.agents.LiveRequest.content)
  - [`LiveRequestQueue`](https://google.github.io/adk-docs/api-reference/python/google-adk.html#google.adk.agents.LiveRequestQueue)
    - [`LiveRequestQueue.close()`](https://google.github.io/adk-docs/api-reference/python/google-adk.html#google.adk.agents.LiveRequestQueue.close)
    - [`LiveRequestQueue.get()`](https://google.github.io/adk-docs/api-reference/python/google-adk.html#google.adk.agents.LiveRequestQueue.get)
    - [`LiveRequestQueue.send()`](https://google.github.io/adk-docs/api-reference/python/google-adk.html#google.adk.agents.LiveRequestQueue.send)
    - [`LiveRequestQueue.send_activity_end()`](https://google.github.io/adk-docs/api-reference/python/google-adk.html#google.adk.agents.LiveRequestQueue.send_activity_end)
    - [`LiveRequestQueue.send_activity_start()`](https://google.github.io/adk-docs/api-reference/python/google-adk.html#google.adk.agents.LiveRequestQueue.send_activity_start)
    - [`LiveRequestQueue.send_content()`](https://google.github.io/adk-docs/api-reference/python/google-adk.html#google.adk.agents.LiveRequestQueue.send_content)
    - [`LiveRequestQueue.send_realtime()`](https://google.github.io/adk-docs/api-reference/python/google-adk.html#google.adk.agents.LiveRequestQueue.send_realtime)
  - [`LlmAgent`](https://google.github.io/adk-docs/api-reference/python/google-adk.html#google.adk.agents.LlmAgent)
    - [`LlmAgent.after_model_callback`](https://google.github.io/adk-docs/api-reference/python/google-adk.html#google.adk.agents.LlmAgent.after_model_callback)
    - [`LlmAgent.after_tool_callback`](https://google.github.io/adk-docs/api-reference/python/google-adk.html#google.adk.agents.LlmAgent.after_tool_callback)
    - [`LlmAgent.before_model_callback`](https://google.github.io/adk-docs/api-reference/python/google-adk.html#google.adk.agents.LlmAgent.before_model_callback)
    - [`LlmAgent.before_tool_callback`](https://google.github.io/adk-docs/api-reference/python/google-adk.html#google.adk.agents.LlmAgent.before_tool_callback)
    - [`LlmAgent.code_executor`](https://google.github.io/adk-docs/api-reference/python/google-adk.html#google.adk.agents.LlmAgent.code_executor)
    - [`LlmAgent.disallow_transfer_to_parent`](https://google.github.io/adk-docs/api-reference/python/google-adk.html#google.adk.agents.LlmAgent.disallow_transfer_to_parent)
    - [`LlmAgent.disallow_transfer_to_peers`](https://google.github.io/adk-docs/api-reference/python/google-adk.html#google.adk.agents.LlmAgent.disallow_transfer_to_peers)
    - [`LlmAgent.generate_content_config`](https://google.github.io/adk-docs/api-reference/python/google-adk.html#google.adk.agents.LlmAgent.generate_content_config)
    - [`LlmAgent.global_instruction`](https://google.github.io/adk-docs/api-reference/python/google-adk.html#google.adk.agents.LlmAgent.global_instruction)
    - [`LlmAgent.include_contents`](https://google.github.io/adk-docs/api-reference/python/google-adk.html#google.adk.agents.LlmAgent.include_contents)
    - [`LlmAgent.input_schema`](https://google.github.io/adk-docs/api-reference/python/google-adk.html#google.adk.agents.LlmAgent.input_schema)
    - [`LlmAgent.instruction`](https://google.github.io/adk-docs/api-reference/python/google-adk.html#google.adk.agents.LlmAgent.instruction)
    - [`LlmAgent.model`](https://google.github.io/adk-docs/api-reference/python/google-adk.html#google.adk.agents.LlmAgent.model)
    - [`LlmAgent.on_model_error_callback`](https://google.github.io/adk-docs/api-reference/python/google-adk.html#google.adk.agents.LlmAgent.on_model_error_callback)
    - [`LlmAgent.on_tool_error_callback`](https://google.github.io/adk-docs/api-reference/python/google-adk.html#google.adk.agents.LlmAgent.on_tool_error_callback)
    - [`LlmAgent.output_key`](https://google.github.io/adk-docs/api-reference/python/google-adk.html#google.adk.agents.LlmAgent.output_key)
    - [`LlmAgent.output_schema`](https://google.github.io/adk-docs/api-reference/python/google-adk.html#google.adk.agents.LlmAgent.output_schema)
    - [`LlmAgent.planner`](https://google.github.io/adk-docs/api-reference/python/google-adk.html#google.adk.agents.LlmAgent.planner)
    - [`LlmAgent.static_instruction`](https://google.github.io/adk-docs/api-reference/python/google-adk.html#google.adk.agents.LlmAgent.static_instruction)
    - [`LlmAgent.tools`](https://google.github.io/adk-docs/api-reference/python/google-adk.html#google.adk.agents.LlmAgent.tools)
    - [`LlmAgent.config_type`](https://google.github.io/adk-docs/api-reference/python/google-adk.html#google.adk.agents.LlmAgent.config_type)
    - [`LlmAgent.validate_generate_content_config`](https://google.github.io/adk-docs/api-reference/python/google-adk.html#google.adk.agents.LlmAgent.validate_generate_content_config)
    - [`LlmAgent.canonical_global_instruction()`](https://google.github.io/adk-docs/api-reference/python/google-adk.html#google.adk.agents.LlmAgent.canonical_global_instruction)
    - [`LlmAgent.canonical_instruction()`](https://google.github.io/adk-docs/api-reference/python/google-adk.html#google.adk.agents.LlmAgent.canonical_instruction)
    - [`LlmAgent.canonical_tools()`](https://google.github.io/adk-docs/api-reference/python/google-adk.html#google.adk.agents.LlmAgent.canonical_tools)
    - [`LlmAgent.canonical_after_model_callbacks`](https://google.github.io/adk-docs/api-reference/python/google-adk.html#google.adk.agents.LlmAgent.canonical_after_model_callbacks)
    - [`LlmAgent.canonical_after_tool_callbacks`](https://google.github.io/adk-docs/api-reference/python/google-adk.html#google.adk.agents.LlmAgent.canonical_after_tool_callbacks)
    - [`LlmAgent.canonical_before_model_callbacks`](https://google.github.io/adk-docs/api-reference/python/google-adk.html#google.adk.agents.LlmAgent.canonical_before_model_callbacks)
    - [`LlmAgent.canonical_before_tool_callbacks`](https://google.github.io/adk-docs/api-reference/python/google-adk.html#google.adk.agents.LlmAgent.canonical_before_tool_callbacks)
    - [`LlmAgent.canonical_model`](https://google.github.io/adk-docs/api-reference/python/google-adk.html#google.adk.agents.LlmAgent.canonical_model)
    - [`LlmAgent.canonical_on_model_error_callbacks`](https://google.github.io/adk-docs/api-reference/python/google-adk.html#google.adk.agents.LlmAgent.canonical_on_model_error_callbacks)
    - [`LlmAgent.canonical_on_tool_error_callbacks`](https://google.github.io/adk-docs/api-reference/python/google-adk.html#google.adk.agents.LlmAgent.canonical_on_tool_error_callbacks)
  - [`LoopAgent`](https://google.github.io/adk-docs/api-reference/python/google-adk.html#google.adk.agents.LoopAgent)
    - [`LoopAgent.max_iterations`](https://google.github.io/adk-docs/api-reference/python/google-adk.html#google.adk.agents.LoopAgent.max_iterations)
    - [`LoopAgent.config_type`](https://google.github.io/adk-docs/api-reference/python/google-adk.html#google.adk.agents.LoopAgent.config_type)
  - [`McpInstructionProvider`](https://google.github.io/adk-docs/api-reference/python/google-adk.html#google.adk.agents.McpInstructionProvider)
  - [`ParallelAgent`](https://google.github.io/adk-docs/api-reference/python/google-adk.html#google.adk.agents.ParallelAgent)
    - [`ParallelAgent.config_type`](https://google.github.io/adk-docs/api-reference/python/google-adk.html#google.adk.agents.ParallelAgent.config_type)
  - [`RunConfig`](https://google.github.io/adk-docs/api-reference/python/google-adk.html#google.adk.agents.RunConfig)
    - [`RunConfig.context_window_compression`](https://google.github.io/adk-docs/api-reference/python/google-adk.html#google.adk.agents.RunConfig.context_window_compression)
    - [`RunConfig.custom_metadata`](https://google.github.io/adk-docs/api-reference/python/google-adk.html#google.adk.agents.RunConfig.custom_metadata)
    - [`RunConfig.enable_affective_dialog`](https://google.github.io/adk-docs/api-reference/python/google-adk.html#google.adk.agents.RunConfig.enable_affective_dialog)
    - [`RunConfig.input_audio_transcription`](https://google.github.io/adk-docs/api-reference/python/google-adk.html#google.adk.agents.RunConfig.input_audio_transcription)
    - [`RunConfig.max_llm_calls`](https://google.github.io/adk-docs/api-reference/python/google-adk.html#google.adk.agents.RunConfig.max_llm_calls)
    - [`RunConfig.output_audio_transcription`](https://google.github.io/adk-docs/api-reference/python/google-adk.html#google.adk.agents.RunConfig.output_audio_transcription)
    - [`RunConfig.proactivity`](https://google.github.io/adk-docs/api-reference/python/google-adk.html#google.adk.agents.RunConfig.proactivity)
    - [`RunConfig.realtime_input_config`](https://google.github.io/adk-docs/api-reference/python/google-adk.html#google.adk.agents.RunConfig.realtime_input_config)
    - [`RunConfig.response_modalities`](https://google.github.io/adk-docs/api-reference/python/google-adk.html#google.adk.agents.RunConfig.response_modalities)
    - [`RunConfig.save_live_blob`](https://google.github.io/adk-docs/api-reference/python/google-adk.html#google.adk.agents.RunConfig.save_live_blob)
    - [`RunConfig.session_resumption`](https://google.github.io/adk-docs/api-reference/python/google-adk.html#google.adk.agents.RunConfig.session_resumption)
    - [`RunConfig.speech_config`](https://google.github.io/adk-docs/api-reference/python/google-adk.html#google.adk.agents.RunConfig.speech_config)
    - [`RunConfig.streaming_mode`](https://google.github.io/adk-docs/api-reference/python/google-adk.html#google.adk.agents.RunConfig.streaming_mode)
    - [`RunConfig.support_cfc`](https://google.github.io/adk-docs/api-reference/python/google-adk.html#google.adk.agents.RunConfig.support_cfc)
    - [`RunConfig.check_for_deprecated_save_live_audio`](https://google.github.io/adk-docs/api-reference/python/google-adk.html#google.adk.agents.RunConfig.check_for_deprecated_save_live_audio)
    - [`RunConfig.validate_max_llm_calls`](https://google.github.io/adk-docs/api-reference/python/google-adk.html#google.adk.agents.RunConfig.validate_max_llm_calls)
    - [`RunConfig.save_input_blobs_as_artifacts`](https://google.github.io/adk-docs/api-reference/python/google-adk.html#google.adk.agents.RunConfig.save_input_blobs_as_artifacts)
      - [`RunConfig.msg`](https://google.github.io/adk-docs/api-reference/python/google-adk.html#google.adk.agents.RunConfig.msg)
      - [`RunConfig.wrapped_property`](https://google.github.io/adk-docs/api-reference/python/google-adk.html#google.adk.agents.RunConfig.wrapped_property)
      - [`RunConfig.field_name`](https://google.github.io/adk-docs/api-reference/python/google-adk.html#google.adk.agents.RunConfig.field_name)
    - [`RunConfig.save_live_audio`](https://google.github.io/adk-docs/api-reference/python/google-adk.html#google.adk.agents.RunConfig.save_live_audio)
      - [`RunConfig.msg`](https://google.github.io/adk-docs/api-reference/python/google-adk.html#id0)
      - [`RunConfig.wrapped_property`](https://google.github.io/adk-docs/api-reference/python/google-adk.html#id14)
      - [`RunConfig.field_name`](https://google.github.io/adk-docs/api-reference/python/google-adk.html#id15)
  - [`SequentialAgent`](https://google.github.io/adk-docs/api-reference/python/google-adk.html#google.adk.agents.SequentialAgent)
    - [`SequentialAgent.config_type`](https://google.github.io/adk-docs/api-reference/python/google-adk.html#google.adk.agents.SequentialAgent.config_type)
- [google.adk.artifacts module](https://google.github.io/adk-docs/api-reference/python/google-adk.html#module-google.adk.artifacts)
  - [`BaseArtifactService`](https://google.github.io/adk-docs/api-reference/python/google-adk.html#google.adk.artifacts.BaseArtifactService)
    - [`BaseArtifactService.delete_artifact()`](https://google.github.io/adk-docs/api-reference/python/google-adk.html#google.adk.artifacts.BaseArtifactService.delete_artifact)
    - [`BaseArtifactService.get_artifact_version()`](https://google.github.io/adk-docs/api-reference/python/google-adk.html#google.adk.artifacts.BaseArtifactService.get_artifact_version)
    - [`BaseArtifactService.list_artifact_keys()`](https://google.github.io/adk-docs/api-reference/python/google-adk.html#google.adk.artifacts.BaseArtifactService.list_artifact_keys)
    - [`BaseArtifactService.list_artifact_versions()`](https://google.github.io/adk-docs/api-reference/python/google-adk.html#google.adk.artifacts.BaseArtifactService.list_artifact_versions)
    - [`BaseArtifactService.list_versions()`](https://google.github.io/adk-docs/api-reference/python/google-adk.html#google.adk.artifacts.BaseArtifactService.list_versions)
    - [`BaseArtifactService.load_artifact()`](https://google.github.io/adk-docs/api-reference/python/google-adk.html#google.adk.artifacts.BaseArtifactService.load_artifact)
    - [`BaseArtifactService.save_artifact()`](https://google.github.io/adk-docs/api-reference/python/google-adk.html#google.adk.artifacts.BaseArtifactService.save_artifact)
  - [`FileArtifactService`](https://google.github.io/adk-docs/api-reference/python/google-adk.html#google.adk.artifacts.FileArtifactService)
    - [`FileArtifactService.delete_artifact()`](https://google.github.io/adk-docs/api-reference/python/google-adk.html#google.adk.artifacts.FileArtifactService.delete_artifact)
    - [`FileArtifactService.get_artifact_version()`](https://google.github.io/adk-docs/api-reference/python/google-adk.html#google.adk.artifacts.FileArtifactService.get_artifact_version)
    - [`FileArtifactService.list_artifact_keys()`](https://google.github.io/adk-docs/api-reference/python/google-adk.html#google.adk.artifacts.FileArtifactService.list_artifact_keys)
    - [`FileArtifactService.list_artifact_versions()`](https://google.github.io/adk-docs/api-reference/python/google-adk.html#google.adk.artifacts.FileArtifactService.list_artifact_versions)
    - [`FileArtifactService.list_versions()`](https://google.github.io/adk-docs/api-reference/python/google-adk.html#google.adk.artifacts.FileArtifactService.list_versions)
    - [`FileArtifactService.load_artifact()`](https://google.github.io/adk-docs/api-reference/python/google-adk.html#google.adk.artifacts.FileArtifactService.load_artifact)
    - [`FileArtifactService.save_artifact()`](https://google.github.io/adk-docs/api-reference/python/google-adk.html#google.adk.artifacts.FileArtifactService.save_artifact)
  - [`GcsArtifactService`](https://google.github.io/adk-docs/api-reference/python/google-adk.html#google.adk.artifacts.GcsArtifactService)
    - [`GcsArtifactService.delete_artifact()`](https://google.github.io/adk-docs/api-reference/python/google-adk.html#google.adk.artifacts.GcsArtifactService.delete_artifact)
    - [`GcsArtifactService.get_artifact_version()`](https://google.github.io/adk-docs/api-reference/python/google-adk.html#google.adk.artifacts.GcsArtifactService.get_artifact_version)
    - [`GcsArtifactService.list_artifact_keys()`](https://google.github.io/adk-docs/api-reference/python/google-adk.html#google.adk.artifacts.GcsArtifactService.list_artifact_keys)
    - [`GcsArtifactService.list_artifact_versions()`](https://google.github.io/adk-docs/api-reference/python/google-adk.html#google.adk.artifacts.GcsArtifactService.list_artifact_versions)
    - [`GcsArtifactService.list_versions()`](https://google.github.io/adk-docs/api-reference/python/google-adk.html#google.adk.artifacts.GcsArtifactService.list_versions)
    - [`GcsArtifactService.load_artifact()`](https://google.github.io/adk-docs/api-reference/python/google-adk.html#google.adk.artifacts.GcsArtifactService.load_artifact)
    - [`GcsArtifactService.save_artifact()`](https://google.github.io/adk-docs/api-reference/python/google-adk.html#google.adk.artifacts.GcsArtifactService.save_artifact)
  - [`InMemoryArtifactService`](https://google.github.io/adk-docs/api-reference/python/google-adk.html#google.adk.artifacts.InMemoryArtifactService)
    - [`InMemoryArtifactService.artifacts`](https://google.github.io/adk-docs/api-reference/python/google-adk.html#google.adk.artifacts.InMemoryArtifactService.artifacts)
    - [`InMemoryArtifactService.delete_artifact()`](https://google.github.io/adk-docs/api-reference/python/google-adk.html#google.adk.artifacts.InMemoryArtifactService.delete_artifact)
    - [`InMemoryArtifactService.get_artifact_version()`](https://google.github.io/adk-docs/api-reference/python/google-adk.html#google.adk.artifacts.InMemoryArtifactService.get_artifact_version)
    - [`InMemoryArtifactService.list_artifact_keys()`](https://google.github.io/adk-docs/api-reference/python/google-adk.html#google.adk.artifacts.InMemoryArtifactService.list_artifact_keys)
    - [`InMemoryArtifactService.list_artifact_versions()`](https://google.github.io/adk-docs/api-reference/python/google-adk.html#google.adk.artifacts.InMemoryArtifactService.list_artifact_versions)
    - [`InMemoryArtifactService.list_versions()`](https://google.github.io/adk-docs/api-reference/python/google-adk.html#google.adk.artifacts.InMemoryArtifactService.list_versions)
    - [`InMemoryArtifactService.load_artifact()`](https://google.github.io/adk-docs/api-reference/python/google-adk.html#google.adk.artifacts.InMemoryArtifactService.load_artifact)
    - [`InMemoryArtifactService.save_artifact()`](https://google.github.io/adk-docs/api-reference/python/google-adk.html#google.adk.artifacts.InMemoryArtifactService.save_artifact)
- [google.adk.apps package](https://google.github.io/adk-docs/api-reference/python/google-adk.html#module-google.adk.apps)
  - [`App`](https://google.github.io/adk-docs/api-reference/python/google-adk.html#google.adk.apps.App)
    - [`App.context_cache_config`](https://google.github.io/adk-docs/api-reference/python/google-adk.html#google.adk.apps.App.context_cache_config)
    - [`App.events_compaction_config`](https://google.github.io/adk-docs/api-reference/python/google-adk.html#google.adk.apps.App.events_compaction_config)
    - [`App.name`](https://google.github.io/adk-docs/api-reference/python/google-adk.html#google.adk.apps.App.name)
    - [`App.plugins`](https://google.github.io/adk-docs/api-reference/python/google-adk.html#google.adk.apps.App.plugins)
    - [`App.resumability_config`](https://google.github.io/adk-docs/api-reference/python/google-adk.html#google.adk.apps.App.resumability_config)
    - [`App.root_agent`](https://google.github.io/adk-docs/api-reference/python/google-adk.html#google.adk.apps.App.root_agent)
  - [`ResumabilityConfig`](https://google.github.io/adk-docs/api-reference/python/google-adk.html#google.adk.apps.ResumabilityConfig)
    - [`ResumabilityConfig.is_resumable`](https://google.github.io/adk-docs/api-reference/python/google-adk.html#google.adk.apps.ResumabilityConfig.is_resumable)
- [google.adk.auth module](https://google.github.io/adk-docs/api-reference/python/google-adk.html#module-google.adk.auth)
- [google.adk.cli module](https://google.github.io/adk-docs/api-reference/python/google-adk.html#module-google.adk.cli)
- [google.adk.code\_executors module](https://google.github.io/adk-docs/api-reference/python/google-adk.html#module-google.adk.code_executors)
  - [`BaseCodeExecutor`](https://google.github.io/adk-docs/api-reference/python/google-adk.html#google.adk.code_executors.BaseCodeExecutor)
    - [`BaseCodeExecutor.optimize_data_file`](https://google.github.io/adk-docs/api-reference/python/google-adk.html#google.adk.code_executors.BaseCodeExecutor.optimize_data_file)
    - [`BaseCodeExecutor.stateful`](https://google.github.io/adk-docs/api-reference/python/google-adk.html#google.adk.code_executors.BaseCodeExecutor.stateful)
    - [`BaseCodeExecutor.error_retry_attempts`](https://google.github.io/adk-docs/api-reference/python/google-adk.html#google.adk.code_executors.BaseCodeExecutor.error_retry_attempts)
    - [`BaseCodeExecutor.code_block_delimiters`](https://google.github.io/adk-docs/api-reference/python/google-adk.html#google.adk.code_executors.BaseCodeExecutor.code_block_delimiters)
    - [`BaseCodeExecutor.execution_result_delimiters`](https://google.github.io/adk-docs/api-reference/python/google-adk.html#google.adk.code_executors.BaseCodeExecutor.execution_result_delimiters)
    - [`BaseCodeExecutor.code_block_delimiters`](https://google.github.io/adk-docs/api-reference/python/google-adk.html#id16)
    - [`BaseCodeExecutor.error_retry_attempts`](https://google.github.io/adk-docs/api-reference/python/google-adk.html#id17)
    - [`BaseCodeExecutor.execution_result_delimiters`](https://google.github.io/adk-docs/api-reference/python/google-adk.html#id18)
    - [`BaseCodeExecutor.optimize_data_file`](https://google.github.io/adk-docs/api-reference/python/google-adk.html#id19)
    - [`BaseCodeExecutor.stateful`](https://google.github.io/adk-docs/api-reference/python/google-adk.html#id20)
    - [`BaseCodeExecutor.execute_code()`](https://google.github.io/adk-docs/api-reference/python/google-adk.html#google.adk.code_executors.BaseCodeExecutor.execute_code)
  - [`BuiltInCodeExecutor`](https://google.github.io/adk-docs/api-reference/python/google-adk.html#google.adk.code_executors.BuiltInCodeExecutor)
    - [`BuiltInCodeExecutor.execute_code()`](https://google.github.io/adk-docs/api-reference/python/google-adk.html#google.adk.code_executors.BuiltInCodeExecutor.execute_code)
    - [`BuiltInCodeExecutor.process_llm_request()`](https://google.github.io/adk-docs/api-reference/python/google-adk.html#google.adk.code_executors.BuiltInCodeExecutor.process_llm_request)
  - [`CodeExecutorContext`](https://google.github.io/adk-docs/api-reference/python/google-adk.html#google.adk.code_executors.CodeExecutorContext)
    - [`CodeExecutorContext.add_input_files()`](https://google.github.io/adk-docs/api-reference/python/google-adk.html#google.adk.code_executors.CodeExecutorContext.add_input_files)
    - [`CodeExecutorContext.add_processed_file_names()`](https://google.github.io/adk-docs/api-reference/python/google-adk.html#google.adk.code_executors.CodeExecutorContext.add_processed_file_names)
    - [`CodeExecutorContext.clear_input_files()`](https://google.github.io/adk-docs/api-reference/python/google-adk.html#google.adk.code_executors.CodeExecutorContext.clear_input_files)
    - [`CodeExecutorContext.get_error_count()`](https://google.github.io/adk-docs/api-reference/python/google-adk.html#google.adk.code_executors.CodeExecutorContext.get_error_count)
    - [`CodeExecutorContext.get_execution_id()`](https://google.github.io/adk-docs/api-reference/python/google-adk.html#google.adk.code_executors.CodeExecutorContext.get_execution_id)
    - [`CodeExecutorContext.get_input_files()`](https://google.github.io/adk-docs/api-reference/python/google-adk.html#google.adk.code_executors.CodeExecutorContext.get_input_files)
    - [`CodeExecutorContext.get_processed_file_names()`](https://google.github.io/adk-docs/api-reference/python/google-adk.html#google.adk.code_executors.CodeExecutorContext.get_processed_file_names)
    - [`CodeExecutorContext.get_state_delta()`](https://google.github.io/adk-docs/api-reference/python/google-adk.html#google.adk.code_executors.CodeExecutorContext.get_state_delta)
    - [`CodeExecutorContext.increment_error_count()`](https://google.github.io/adk-docs/api-reference/python/google-adk.html#google.adk.code_executors.CodeExecutorContext.increment_error_count)
    - [`CodeExecutorContext.reset_error_count()`](https://google.github.io/adk-docs/api-reference/python/google-adk.html#google.adk.code_executors.CodeExecutorContext.reset_error_count)
    - [`CodeExecutorContext.set_execution_id()`](https://google.github.io/adk-docs/api-reference/python/google-adk.html#google.adk.code_executors.CodeExecutorContext.set_execution_id)
    - [`CodeExecutorContext.update_code_execution_result()`](https://google.github.io/adk-docs/api-reference/python/google-adk.html#google.adk.code_executors.CodeExecutorContext.update_code_execution_result)
  - [`UnsafeLocalCodeExecutor`](https://google.github.io/adk-docs/api-reference/python/google-adk.html#google.adk.code_executors.UnsafeLocalCodeExecutor)
    - [`UnsafeLocalCodeExecutor.optimize_data_file`](https://google.github.io/adk-docs/api-reference/python/google-adk.html#google.adk.code_executors.UnsafeLocalCodeExecutor.optimize_data_file)
    - [`UnsafeLocalCodeExecutor.stateful`](https://google.github.io/adk-docs/api-reference/python/google-adk.html#google.adk.code_executors.UnsafeLocalCodeExecutor.stateful)
    - [`UnsafeLocalCodeExecutor.execute_code()`](https://google.github.io/adk-docs/api-reference/python/google-adk.html#google.adk.code_executors.UnsafeLocalCodeExecutor.execute_code)
- [google.adk.errors module](https://google.github.io/adk-docs/api-reference/python/google-adk.html#module-google.adk.errors)
- [google.adk.evaluation module](https://google.github.io/adk-docs/api-reference/python/google-adk.html#module-google.adk.evaluation)
  - [`AgentEvaluator`](https://google.github.io/adk-docs/api-reference/python/google-adk.html#google.adk.evaluation.AgentEvaluator)
    - [`AgentEvaluator.evaluate()`](https://google.github.io/adk-docs/api-reference/python/google-adk.html#google.adk.evaluation.AgentEvaluator.evaluate)
    - [`AgentEvaluator.evaluate_eval_set()`](https://google.github.io/adk-docs/api-reference/python/google-adk.html#google.adk.evaluation.AgentEvaluator.evaluate_eval_set)
    - [`AgentEvaluator.find_config_for_test_file()`](https://google.github.io/adk-docs/api-reference/python/google-adk.html#google.adk.evaluation.AgentEvaluator.find_config_for_test_file)
    - [`AgentEvaluator.migrate_eval_data_to_new_schema()`](https://google.github.io/adk-docs/api-reference/python/google-adk.html#google.adk.evaluation.AgentEvaluator.migrate_eval_data_to_new_schema)
- [google.adk.events module](https://google.github.io/adk-docs/api-reference/python/google-adk.html#module-google.adk.events)
  - [`Event`](https://google.github.io/adk-docs/api-reference/python/google-adk.html#google.adk.events.Event)
    - [`Event.actions`](https://google.github.io/adk-docs/api-reference/python/google-adk.html#google.adk.events.Event.actions)
    - [`Event.author`](https://google.github.io/adk-docs/api-reference/python/google-adk.html#google.adk.events.Event.author)
    - [`Event.branch`](https://google.github.io/adk-docs/api-reference/python/google-adk.html#google.adk.events.Event.branch)
    - [`Event.id`](https://google.github.io/adk-docs/api-reference/python/google-adk.html#google.adk.events.Event.id)
    - [`Event.invocation_id`](https://google.github.io/adk-docs/api-reference/python/google-adk.html#google.adk.events.Event.invocation_id)
    - [`Event.long_running_tool_ids`](https://google.github.io/adk-docs/api-reference/python/google-adk.html#google.adk.events.Event.long_running_tool_ids)
    - [`Event.timestamp`](https://google.github.io/adk-docs/api-reference/python/google-adk.html#google.adk.events.Event.timestamp)
    - [`Event.new_id()`](https://google.github.io/adk-docs/api-reference/python/google-adk.html#google.adk.events.Event.new_id)
    - [`Event.get_function_calls()`](https://google.github.io/adk-docs/api-reference/python/google-adk.html#google.adk.events.Event.get_function_calls)
    - [`Event.get_function_responses()`](https://google.github.io/adk-docs/api-reference/python/google-adk.html#google.adk.events.Event.get_function_responses)
    - [`Event.has_trailing_code_execution_result()`](https://google.github.io/adk-docs/api-reference/python/google-adk.html#google.adk.events.Event.has_trailing_code_execution_result)
    - [`Event.is_final_response()`](https://google.github.io/adk-docs/api-reference/python/google-adk.html#google.adk.events.Event.is_final_response)
    - [`Event.model_post_init()`](https://google.github.io/adk-docs/api-reference/python/google-adk.html#google.adk.events.Event.model_post_init)
  - [`EventActions`](https://google.github.io/adk-docs/api-reference/python/google-adk.html#google.adk.events.EventActions)
    - [`EventActions.agent_state`](https://google.github.io/adk-docs/api-reference/python/google-adk.html#google.adk.events.EventActions.agent_state)
    - [`EventActions.artifact_delta`](https://google.github.io/adk-docs/api-reference/python/google-adk.html#google.adk.events.EventActions.artifact_delta)
    - [`EventActions.compaction`](https://google.github.io/adk-docs/api-reference/python/google-adk.html#google.adk.events.EventActions.compaction)
    - [`EventActions.end_of_agent`](https://google.github.io/adk-docs/api-reference/python/google-adk.html#google.adk.events.EventActions.end_of_agent)
    - [`EventActions.escalate`](https://google.github.io/adk-docs/api-reference/python/google-adk.html#google.adk.events.EventActions.escalate)
    - [`EventActions.requested_auth_configs`](https://google.github.io/adk-docs/api-reference/python/google-adk.html#google.adk.events.EventActions.requested_auth_configs)
    - [`EventActions.requested_tool_confirmations`](https://google.github.io/adk-docs/api-reference/python/google-adk.html#google.adk.events.EventActions.requested_tool_confirmations)
    - [`EventActions.rewind_before_invocation_id`](https://google.github.io/adk-docs/api-reference/python/google-adk.html#google.adk.events.EventActions.rewind_before_invocation_id)
    - [`EventActions.skip_summarization`](https://google.github.io/adk-docs/api-reference/python/google-adk.html#google.adk.events.EventActions.skip_summarization)
    - [`EventActions.state_delta`](https://google.github.io/adk-docs/api-reference/python/google-adk.html#google.adk.events.EventActions.state_delta)
    - [`EventActions.transfer_to_agent`](https://google.github.io/adk-docs/api-reference/python/google-adk.html#google.adk.events.EventActions.transfer_to_agent)
- [google.adk.examples module](https://google.github.io/adk-docs/api-reference/python/google-adk.html#module-google.adk.examples)
  - [`BaseExampleProvider`](https://google.github.io/adk-docs/api-reference/python/google-adk.html#google.adk.examples.BaseExampleProvider)
    - [`BaseExampleProvider.get_examples()`](https://google.github.io/adk-docs/api-reference/python/google-adk.html#google.adk.examples.BaseExampleProvider.get_examples)
  - [`Example`](https://google.github.io/adk-docs/api-reference/python/google-adk.html#google.adk.examples.Example)
    - [`Example.input`](https://google.github.io/adk-docs/api-reference/python/google-adk.html#google.adk.examples.Example.input)
    - [`Example.output`](https://google.github.io/adk-docs/api-reference/python/google-adk.html#google.adk.examples.Example.output)
    - [`Example.input`](https://google.github.io/adk-docs/api-reference/python/google-adk.html#id21)
    - [`Example.output`](https://google.github.io/adk-docs/api-reference/python/google-adk.html#id22)
  - [`VertexAiExampleStore`](https://google.github.io/adk-docs/api-reference/python/google-adk.html#google.adk.examples.VertexAiExampleStore)
    - [`VertexAiExampleStore.get_examples()`](https://google.github.io/adk-docs/api-reference/python/google-adk.html#google.adk.examples.VertexAiExampleStore.get_examples)
- [google.adk.flows module](https://google.github.io/adk-docs/api-reference/python/google-adk.html#module-google.adk.flows)
- [google.adk.memory module](https://google.github.io/adk-docs/api-reference/python/google-adk.html#module-google.adk.memory)
  - [`BaseMemoryService`](https://google.github.io/adk-docs/api-reference/python/google-adk.html#google.adk.memory.BaseMemoryService)
    - [`BaseMemoryService.add_session_to_memory()`](https://google.github.io/adk-docs/api-reference/python/google-adk.html#google.adk.memory.BaseMemoryService.add_session_to_memory)
    - [`BaseMemoryService.search_memory()`](https://google.github.io/adk-docs/api-reference/python/google-adk.html#google.adk.memory.BaseMemoryService.search_memory)
  - [`InMemoryMemoryService`](https://google.github.io/adk-docs/api-reference/python/google-adk.html#google.adk.memory.InMemoryMemoryService)
    - [`InMemoryMemoryService.add_session_to_memory()`](https://google.github.io/adk-docs/api-reference/python/google-adk.html#google.adk.memory.InMemoryMemoryService.add_session_to_memory)
    - [`InMemoryMemoryService.search_memory()`](https://google.github.io/adk-docs/api-reference/python/google-adk.html#google.adk.memory.InMemoryMemoryService.search_memory)
  - [`VertexAiMemoryBankService`](https://google.github.io/adk-docs/api-reference/python/google-adk.html#google.adk.memory.VertexAiMemoryBankService)
    - [`VertexAiMemoryBankService.add_session_to_memory()`](https://google.github.io/adk-docs/api-reference/python/google-adk.html#google.adk.memory.VertexAiMemoryBankService.add_session_to_memory)
    - [`VertexAiMemoryBankService.search_memory()`](https://google.github.io/adk-docs/api-reference/python/google-adk.html#google.adk.memory.VertexAiMemoryBankService.search_memory)
  - [`VertexAiRagMemoryService`](https://google.github.io/adk-docs/api-reference/python/google-adk.html#google.adk.memory.VertexAiRagMemoryService)
    - [`VertexAiRagMemoryService.add_session_to_memory()`](https://google.github.io/adk-docs/api-reference/python/google-adk.html#google.adk.memory.VertexAiRagMemoryService.add_session_to_memory)
    - [`VertexAiRagMemoryService.search_memory()`](https://google.github.io/adk-docs/api-reference/python/google-adk.html#google.adk.memory.VertexAiRagMemoryService.search_memory)
- [google.adk.models module](https://google.github.io/adk-docs/api-reference/python/google-adk.html#module-google.adk.models)
  - [`BaseLlm`](https://google.github.io/adk-docs/api-reference/python/google-adk.html#google.adk.models.BaseLlm)
    - [`BaseLlm.model`](https://google.github.io/adk-docs/api-reference/python/google-adk.html#google.adk.models.BaseLlm.model)
    - [`BaseLlm.supported_models()`](https://google.github.io/adk-docs/api-reference/python/google-adk.html#google.adk.models.BaseLlm.supported_models)
    - [`BaseLlm.connect()`](https://google.github.io/adk-docs/api-reference/python/google-adk.html#google.adk.models.BaseLlm.connect)
    - [`BaseLlm.generate_content_async()`](https://google.github.io/adk-docs/api-reference/python/google-adk.html#google.adk.models.BaseLlm.generate_content_async)
  - [`Gemini`](https://google.github.io/adk-docs/api-reference/python/google-adk.html#google.adk.models.Gemini)
    - [`Gemini.model`](https://google.github.io/adk-docs/api-reference/python/google-adk.html#google.adk.models.Gemini.model)
    - [`Gemini.model`](https://google.github.io/adk-docs/api-reference/python/google-adk.html#id23)
    - [`Gemini.retry_options`](https://google.github.io/adk-docs/api-reference/python/google-adk.html#google.adk.models.Gemini.retry_options)
    - [`Gemini.speech_config`](https://google.github.io/adk-docs/api-reference/python/google-adk.html#google.adk.models.Gemini.speech_config)
    - [`Gemini.supported_models()`](https://google.github.io/adk-docs/api-reference/python/google-adk.html#google.adk.models.Gemini.supported_models)
    - [`Gemini.connect()`](https://google.github.io/adk-docs/api-reference/python/google-adk.html#google.adk.models.Gemini.connect)
    - [`Gemini.generate_content_async()`](https://google.github.io/adk-docs/api-reference/python/google-adk.html#google.adk.models.Gemini.generate_content_async)
    - [`Gemini.api_client`](https://google.github.io/adk-docs/api-reference/python/google-adk.html#google.adk.models.Gemini.api_client)
  - [`Gemma`](https://google.github.io/adk-docs/api-reference/python/google-adk.html#google.adk.models.Gemma)
    - [`Gemma.model`](https://google.github.io/adk-docs/api-reference/python/google-adk.html#google.adk.models.Gemma.model)
    - [`Gemma.supported_models()`](https://google.github.io/adk-docs/api-reference/python/google-adk.html#google.adk.models.Gemma.supported_models)
    - [`Gemma.generate_content_async()`](https://google.github.io/adk-docs/api-reference/python/google-adk.html#google.adk.models.Gemma.generate_content_async)
  - [`LLMRegistry`](https://google.github.io/adk-docs/api-reference/python/google-adk.html#google.adk.models.LLMRegistry)
    - [`LLMRegistry.new_llm()`](https://google.github.io/adk-docs/api-reference/python/google-adk.html#google.adk.models.LLMRegistry.new_llm)
    - [`LLMRegistry.register()`](https://google.github.io/adk-docs/api-reference/python/google-adk.html#google.adk.models.LLMRegistry.register)
    - [`LLMRegistry.resolve()`](https://google.github.io/adk-docs/api-reference/python/google-adk.html#google.adk.models.LLMRegistry.resolve)
- [google.adk.planners module](https://google.github.io/adk-docs/api-reference/python/google-adk.html#module-google.adk.planners)
  - [`BasePlanner`](https://google.github.io/adk-docs/api-reference/python/google-adk.html#google.adk.planners.BasePlanner)
    - [`BasePlanner.build_planning_instruction()`](https://google.github.io/adk-docs/api-reference/python/google-adk.html#google.adk.planners.BasePlanner.build_planning_instruction)
    - [`BasePlanner.process_planning_response()`](https://google.github.io/adk-docs/api-reference/python/google-adk.html#google.adk.planners.BasePlanner.process_planning_response)
  - [`BuiltInPlanner`](https://google.github.io/adk-docs/api-reference/python/google-adk.html#google.adk.planners.BuiltInPlanner)
    - [`BuiltInPlanner.thinking_config`](https://google.github.io/adk-docs/api-reference/python/google-adk.html#google.adk.planners.BuiltInPlanner.thinking_config)
    - [`BuiltInPlanner.apply_thinking_config()`](https://google.github.io/adk-docs/api-reference/python/google-adk.html#google.adk.planners.BuiltInPlanner.apply_thinking_config)
    - [`BuiltInPlanner.build_planning_instruction()`](https://google.github.io/adk-docs/api-reference/python/google-adk.html#google.adk.planners.BuiltInPlanner.build_planning_instruction)
    - [`BuiltInPlanner.process_planning_response()`](https://google.github.io/adk-docs/api-reference/python/google-adk.html#google.adk.planners.BuiltInPlanner.process_planning_response)
    - [`BuiltInPlanner.thinking_config`](https://google.github.io/adk-docs/api-reference/python/google-adk.html#id29)
  - [`PlanReActPlanner`](https://google.github.io/adk-docs/api-reference/python/google-adk.html#google.adk.planners.PlanReActPlanner)
    - [`PlanReActPlanner.build_planning_instruction()`](https://google.github.io/adk-docs/api-reference/python/google-adk.html#google.adk.planners.PlanReActPlanner.build_planning_instruction)
    - [`PlanReActPlanner.process_planning_response()`](https://google.github.io/adk-docs/api-reference/python/google-adk.html#google.adk.planners.PlanReActPlanner.process_planning_response)
- [google.adk.platform module](https://google.github.io/adk-docs/api-reference/python/google-adk.html#module-google.adk.platform)
- [google.adk.plugins module](https://google.github.io/adk-docs/api-reference/python/google-adk.html#module-google.adk.plugins)
  - [`BasePlugin`](https://google.github.io/adk-docs/api-reference/python/google-adk.html#google.adk.plugins.BasePlugin)
    - [`BasePlugin.after_agent_callback()`](https://google.github.io/adk-docs/api-reference/python/google-adk.html#google.adk.plugins.BasePlugin.after_agent_callback)
    - [`BasePlugin.after_model_callback()`](https://google.github.io/adk-docs/api-reference/python/google-adk.html#google.adk.plugins.BasePlugin.after_model_callback)
    - [`BasePlugin.after_run_callback()`](https://google.github.io/adk-docs/api-reference/python/google-adk.html#google.adk.plugins.BasePlugin.after_run_callback)
    - [`BasePlugin.after_tool_callback()`](https://google.github.io/adk-docs/api-reference/python/google-adk.html#google.adk.plugins.BasePlugin.after_tool_callback)
    - [`BasePlugin.before_agent_callback()`](https://google.github.io/adk-docs/api-reference/python/google-adk.html#google.adk.plugins.BasePlugin.before_agent_callback)
    - [`BasePlugin.before_model_callback()`](https://google.github.io/adk-docs/api-reference/python/google-adk.html#google.adk.plugins.BasePlugin.before_model_callback)
    - [`BasePlugin.before_run_callback()`](https://google.github.io/adk-docs/api-reference/python/google-adk.html#google.adk.plugins.BasePlugin.before_run_callback)
    - [`BasePlugin.before_tool_callback()`](https://google.github.io/adk-docs/api-reference/python/google-adk.html#google.adk.plugins.BasePlugin.before_tool_callback)
    - [`BasePlugin.close()`](https://google.github.io/adk-docs/api-reference/python/google-adk.html#google.adk.plugins.BasePlugin.close)
    - [`BasePlugin.on_event_callback()`](https://google.github.io/adk-docs/api-reference/python/google-adk.html#google.adk.plugins.BasePlugin.on_event_callback)
    - [`BasePlugin.on_model_error_callback()`](https://google.github.io/adk-docs/api-reference/python/google-adk.html#google.adk.plugins.BasePlugin.on_model_error_callback)
    - [`BasePlugin.on_tool_error_callback()`](https://google.github.io/adk-docs/api-reference/python/google-adk.html#google.adk.plugins.BasePlugin.on_tool_error_callback)
    - [`BasePlugin.on_user_message_callback()`](https://google.github.io/adk-docs/api-reference/python/google-adk.html#google.adk.plugins.BasePlugin.on_user_message_callback)
  - [`LoggingPlugin`](https://google.github.io/adk-docs/api-reference/python/google-adk.html#google.adk.plugins.LoggingPlugin)
    - [`LoggingPlugin.after_agent_callback()`](https://google.github.io/adk-docs/api-reference/python/google-adk.html#google.adk.plugins.LoggingPlugin.after_agent_callback)
    - [`LoggingPlugin.after_model_callback()`](https://google.github.io/adk-docs/api-reference/python/google-adk.html#google.adk.plugins.LoggingPlugin.after_model_callback)
    - [`LoggingPlugin.after_run_callback()`](https://google.github.io/adk-docs/api-reference/python/google-adk.html#google.adk.plugins.LoggingPlugin.after_run_callback)
    - [`LoggingPlugin.after_tool_callback()`](https://google.github.io/adk-docs/api-reference/python/google-adk.html#google.adk.plugins.LoggingPlugin.after_tool_callback)
    - [`LoggingPlugin.before_agent_callback()`](https://google.github.io/adk-docs/api-reference/python/google-adk.html#google.adk.plugins.LoggingPlugin.before_agent_callback)
    - [`LoggingPlugin.before_model_callback()`](https://google.github.io/adk-docs/api-reference/python/google-adk.html#google.adk.plugins.LoggingPlugin.before_model_callback)
    - [`LoggingPlugin.before_run_callback()`](https://google.github.io/adk-docs/api-reference/python/google-adk.html#google.adk.plugins.LoggingPlugin.before_run_callback)
    - [`LoggingPlugin.before_tool_callback()`](https://google.github.io/adk-docs/api-reference/python/google-adk.html#google.adk.plugins.LoggingPlugin.before_tool_callback)
    - [`LoggingPlugin.on_event_callback()`](https://google.github.io/adk-docs/api-reference/python/google-adk.html#google.adk.plugins.LoggingPlugin.on_event_callback)
    - [`LoggingPlugin.on_model_error_callback()`](https://google.github.io/adk-docs/api-reference/python/google-adk.html#google.adk.plugins.LoggingPlugin.on_model_error_callback)
    - [`LoggingPlugin.on_tool_error_callback()`](https://google.github.io/adk-docs/api-reference/python/google-adk.html#google.adk.plugins.LoggingPlugin.on_tool_error_callback)
    - [`LoggingPlugin.on_user_message_callback()`](https://google.github.io/adk-docs/api-reference/python/google-adk.html#google.adk.plugins.LoggingPlugin.on_user_message_callback)
  - [`PluginManager`](https://google.github.io/adk-docs/api-reference/python/google-adk.html#google.adk.plugins.PluginManager)
    - [`PluginManager.close()`](https://google.github.io/adk-docs/api-reference/python/google-adk.html#google.adk.plugins.PluginManager.close)
    - [`PluginManager.get_plugin()`](https://google.github.io/adk-docs/api-reference/python/google-adk.html#google.adk.plugins.PluginManager.get_plugin)
    - [`PluginManager.register_plugin()`](https://google.github.io/adk-docs/api-reference/python/google-adk.html#google.adk.plugins.PluginManager.register_plugin)
    - [`PluginManager.run_after_agent_callback()`](https://google.github.io/adk-docs/api-reference/python/google-adk.html#google.adk.plugins.PluginManager.run_after_agent_callback)
    - [`PluginManager.run_after_model_callback()`](https://google.github.io/adk-docs/api-reference/python/google-adk.html#google.adk.plugins.PluginManager.run_after_model_callback)
    - [`PluginManager.run_after_run_callback()`](https://google.github.io/adk-docs/api-reference/python/google-adk.html#google.adk.plugins.PluginManager.run_after_run_callback)
    - [`PluginManager.run_after_tool_callback()`](https://google.github.io/adk-docs/api-reference/python/google-adk.html#google.adk.plugins.PluginManager.run_after_tool_callback)
    - [`PluginManager.run_before_agent_callback()`](https://google.github.io/adk-docs/api-reference/python/google-adk.html#google.adk.plugins.PluginManager.run_before_agent_callback)
    - [`PluginManager.run_before_model_callback()`](https://google.github.io/adk-docs/api-reference/python/google-adk.html#google.adk.plugins.PluginManager.run_before_model_callback)
    - [`PluginManager.run_before_run_callback()`](https://google.github.io/adk-docs/api-reference/python/google-adk.html#google.adk.plugins.PluginManager.run_before_run_callback)
    - [`PluginManager.run_before_tool_callback()`](https://google.github.io/adk-docs/api-reference/python/google-adk.html#google.adk.plugins.PluginManager.run_before_tool_callback)
    - [`PluginManager.run_on_event_callback()`](https://google.github.io/adk-docs/api-reference/python/google-adk.html#google.adk.plugins.PluginManager.run_on_event_callback)
    - [`PluginManager.run_on_model_error_callback()`](https://google.github.io/adk-docs/api-reference/python/google-adk.html#google.adk.plugins.PluginManager.run_on_model_error_callback)
    - [`PluginManager.run_on_tool_error_callback()`](https://google.github.io/adk-docs/api-reference/python/google-adk.html#google.adk.plugins.PluginManager.run_on_tool_error_callback)
    - [`PluginManager.run_on_user_message_callback()`](https://google.github.io/adk-docs/api-reference/python/google-adk.html#google.adk.plugins.PluginManager.run_on_user_message_callback)
  - [`ReflectAndRetryToolPlugin`](https://google.github.io/adk-docs/api-reference/python/google-adk.html#google.adk.plugins.ReflectAndRetryToolPlugin)
    - [`ReflectAndRetryToolPlugin.after_tool_callback()`](https://google.github.io/adk-docs/api-reference/python/google-adk.html#google.adk.plugins.ReflectAndRetryToolPlugin.after_tool_callback)
    - [`ReflectAndRetryToolPlugin.extract_error_from_result()`](https://google.github.io/adk-docs/api-reference/python/google-adk.html#google.adk.plugins.ReflectAndRetryToolPlugin.extract_error_from_result)
    - [`ReflectAndRetryToolPlugin.on_tool_error_callback()`](https://google.github.io/adk-docs/api-reference/python/google-adk.html#google.adk.plugins.ReflectAndRetryToolPlugin.on_tool_error_callback)
- [google.adk.runners module](https://google.github.io/adk-docs/api-reference/python/google-adk.html#module-google.adk.runners)
  - [`InMemoryRunner`](https://google.github.io/adk-docs/api-reference/python/google-adk.html#google.adk.runners.InMemoryRunner)
    - [`InMemoryRunner.agent`](https://google.github.io/adk-docs/api-reference/python/google-adk.html#google.adk.runners.InMemoryRunner.agent)
    - [`InMemoryRunner.app_name`](https://google.github.io/adk-docs/api-reference/python/google-adk.html#google.adk.runners.InMemoryRunner.app_name)
  - [`Runner`](https://google.github.io/adk-docs/api-reference/python/google-adk.html#google.adk.runners.Runner)
    - [`Runner.app_name`](https://google.github.io/adk-docs/api-reference/python/google-adk.html#google.adk.runners.Runner.app_name)
    - [`Runner.agent`](https://google.github.io/adk-docs/api-reference/python/google-adk.html#google.adk.runners.Runner.agent)
    - [`Runner.artifact_service`](https://google.github.io/adk-docs/api-reference/python/google-adk.html#google.adk.runners.Runner.artifact_service)
    - [`Runner.plugin_manager`](https://google.github.io/adk-docs/api-reference/python/google-adk.html#google.adk.runners.Runner.plugin_manager)
    - [`Runner.session_service`](https://google.github.io/adk-docs/api-reference/python/google-adk.html#google.adk.runners.Runner.session_service)
    - [`Runner.memory_service`](https://google.github.io/adk-docs/api-reference/python/google-adk.html#google.adk.runners.Runner.memory_service)
    - [`Runner.credential_service`](https://google.github.io/adk-docs/api-reference/python/google-adk.html#google.adk.runners.Runner.credential_service)
    - [`Runner.context_cache_config`](https://google.github.io/adk-docs/api-reference/python/google-adk.html#google.adk.runners.Runner.context_cache_config)
    - [`Runner.resumability_config`](https://google.github.io/adk-docs/api-reference/python/google-adk.html#google.adk.runners.Runner.resumability_config)
    - [`Runner.agent`](https://google.github.io/adk-docs/api-reference/python/google-adk.html#id40)
    - [`Runner.app_name`](https://google.github.io/adk-docs/api-reference/python/google-adk.html#id41)
    - [`Runner.artifact_service`](https://google.github.io/adk-docs/api-reference/python/google-adk.html#id42)
    - [`Runner.close()`](https://google.github.io/adk-docs/api-reference/python/google-adk.html#google.adk.runners.Runner.close)
    - [`Runner.context_cache_config`](https://google.github.io/adk-docs/api-reference/python/google-adk.html#id43)
    - [`Runner.credential_service`](https://google.github.io/adk-docs/api-reference/python/google-adk.html#id44)
    - [`Runner.memory_service`](https://google.github.io/adk-docs/api-reference/python/google-adk.html#id45)
    - [`Runner.plugin_manager`](https://google.github.io/adk-docs/api-reference/python/google-adk.html#id46)
    - [`Runner.resumability_config`](https://google.github.io/adk-docs/api-reference/python/google-adk.html#id47)
    - [`Runner.rewind_async()`](https://google.github.io/adk-docs/api-reference/python/google-adk.html#google.adk.runners.Runner.rewind_async)
    - [`Runner.run()`](https://google.github.io/adk-docs/api-reference/python/google-adk.html#google.adk.runners.Runner.run)
    - [`Runner.run_async()`](https://google.github.io/adk-docs/api-reference/python/google-adk.html#google.adk.runners.Runner.run_async)
    - [`Runner.run_debug()`](https://google.github.io/adk-docs/api-reference/python/google-adk.html#google.adk.runners.Runner.run_debug)
    - [`Runner.run_live()`](https://google.github.io/adk-docs/api-reference/python/google-adk.html#google.adk.runners.Runner.run_live)
    - [`Runner.session_service`](https://google.github.io/adk-docs/api-reference/python/google-adk.html#id48)
- [google.adk.sessions module](https://google.github.io/adk-docs/api-reference/python/google-adk.html#module-google.adk.sessions)
  - [`BaseSessionService`](https://google.github.io/adk-docs/api-reference/python/google-adk.html#google.adk.sessions.BaseSessionService)
    - [`BaseSessionService.append_event()`](https://google.github.io/adk-docs/api-reference/python/google-adk.html#google.adk.sessions.BaseSessionService.append_event)
    - [`BaseSessionService.create_session()`](https://google.github.io/adk-docs/api-reference/python/google-adk.html#google.adk.sessions.BaseSessionService.create_session)
    - [`BaseSessionService.delete_session()`](https://google.github.io/adk-docs/api-reference/python/google-adk.html#google.adk.sessions.BaseSessionService.delete_session)
    - [`BaseSessionService.get_session()`](https://google.github.io/adk-docs/api-reference/python/google-adk.html#google.adk.sessions.BaseSessionService.get_session)
    - [`BaseSessionService.list_sessions()`](https://google.github.io/adk-docs/api-reference/python/google-adk.html#google.adk.sessions.BaseSessionService.list_sessions)
  - [`InMemorySessionService`](https://google.github.io/adk-docs/api-reference/python/google-adk.html#google.adk.sessions.InMemorySessionService)
    - [`InMemorySessionService.append_event()`](https://google.github.io/adk-docs/api-reference/python/google-adk.html#google.adk.sessions.InMemorySessionService.append_event)
    - [`InMemorySessionService.create_session()`](https://google.github.io/adk-docs/api-reference/python/google-adk.html#google.adk.sessions.InMemorySessionService.create_session)
    - [`InMemorySessionService.create_session_sync()`](https://google.github.io/adk-docs/api-reference/python/google-adk.html#google.adk.sessions.InMemorySessionService.create_session_sync)
    - [`InMemorySessionService.delete_session()`](https://google.github.io/adk-docs/api-reference/python/google-adk.html#google.adk.sessions.InMemorySessionService.delete_session)
    - [`InMemorySessionService.delete_session_sync()`](https://google.github.io/adk-docs/api-reference/python/google-adk.html#google.adk.sessions.InMemorySessionService.delete_session_sync)
    - [`InMemorySessionService.get_session()`](https://google.github.io/adk-docs/api-reference/python/google-adk.html#google.adk.sessions.InMemorySessionService.get_session)
    - [`InMemorySessionService.get_session_sync()`](https://google.github.io/adk-docs/api-reference/python/google-adk.html#google.adk.sessions.InMemorySessionService.get_session_sync)
    - [`InMemorySessionService.list_sessions()`](https://google.github.io/adk-docs/api-reference/python/google-adk.html#google.adk.sessions.InMemorySessionService.list_sessions)
    - [`InMemorySessionService.list_sessions_sync()`](https://google.github.io/adk-docs/api-reference/python/google-adk.html#google.adk.sessions.InMemorySessionService.list_sessions_sync)
  - [`Session`](https://google.github.io/adk-docs/api-reference/python/google-adk.html#google.adk.sessions.Session)
    - [`Session.app_name`](https://google.github.io/adk-docs/api-reference/python/google-adk.html#google.adk.sessions.Session.app_name)
    - [`Session.events`](https://google.github.io/adk-docs/api-reference/python/google-adk.html#google.adk.sessions.Session.events)
    - [`Session.id`](https://google.github.io/adk-docs/api-reference/python/google-adk.html#google.adk.sessions.Session.id)
    - [`Session.last_update_time`](https://google.github.io/adk-docs/api-reference/python/google-adk.html#google.adk.sessions.Session.last_update_time)
    - [`Session.state`](https://google.github.io/adk-docs/api-reference/python/google-adk.html#google.adk.sessions.Session.state)
    - [`Session.user_id`](https://google.github.io/adk-docs/api-reference/python/google-adk.html#google.adk.sessions.Session.user_id)
  - [`State`](https://google.github.io/adk-docs/api-reference/python/google-adk.html#google.adk.sessions.State)
    - [`State.APP_PREFIX`](https://google.github.io/adk-docs/api-reference/python/google-adk.html#google.adk.sessions.State.APP_PREFIX)
    - [`State.TEMP_PREFIX`](https://google.github.io/adk-docs/api-reference/python/google-adk.html#google.adk.sessions.State.TEMP_PREFIX)
    - [`State.USER_PREFIX`](https://google.github.io/adk-docs/api-reference/python/google-adk.html#google.adk.sessions.State.USER_PREFIX)
    - [`State.get()`](https://google.github.io/adk-docs/api-reference/python/google-adk.html#google.adk.sessions.State.get)
    - [`State.has_delta()`](https://google.github.io/adk-docs/api-reference/python/google-adk.html#google.adk.sessions.State.has_delta)
    - [`State.setdefault()`](https://google.github.io/adk-docs/api-reference/python/google-adk.html#google.adk.sessions.State.setdefault)
    - [`State.to_dict()`](https://google.github.io/adk-docs/api-reference/python/google-adk.html#google.adk.sessions.State.to_dict)
    - [`State.update()`](https://google.github.io/adk-docs/api-reference/python/google-adk.html#google.adk.sessions.State.update)
  - [`VertexAiSessionService`](https://google.github.io/adk-docs/api-reference/python/google-adk.html#google.adk.sessions.VertexAiSessionService)
    - [`VertexAiSessionService.append_event()`](https://google.github.io/adk-docs/api-reference/python/google-adk.html#google.adk.sessions.VertexAiSessionService.append_event)
    - [`VertexAiSessionService.create_session()`](https://google.github.io/adk-docs/api-reference/python/google-adk.html#google.adk.sessions.VertexAiSessionService.create_session)
    - [`VertexAiSessionService.delete_session()`](https://google.github.io/adk-docs/api-reference/python/google-adk.html#google.adk.sessions.VertexAiSessionService.delete_session)
    - [`VertexAiSessionService.get_session()`](https://google.github.io/adk-docs/api-reference/python/google-adk.html#google.adk.sessions.VertexAiSessionService.get_session)
    - [`VertexAiSessionService.list_sessions()`](https://google.github.io/adk-docs/api-reference/python/google-adk.html#google.adk.sessions.VertexAiSessionService.list_sessions)
- [google.adk.telemetry module](https://google.github.io/adk-docs/api-reference/python/google-adk.html#module-google.adk.telemetry)
  - [`trace_call_llm()`](https://google.github.io/adk-docs/api-reference/python/google-adk.html#google.adk.telemetry.trace_call_llm)
  - [`trace_merged_tool_calls()`](https://google.github.io/adk-docs/api-reference/python/google-adk.html#google.adk.telemetry.trace_merged_tool_calls)
  - [`trace_send_data()`](https://google.github.io/adk-docs/api-reference/python/google-adk.html#google.adk.telemetry.trace_send_data)
  - [`trace_tool_call()`](https://google.github.io/adk-docs/api-reference/python/google-adk.html#google.adk.telemetry.trace_tool_call)
- [google.adk.tools package](https://google.github.io/adk-docs/api-reference/python/google-adk.html#module-google.adk.tools)
  - [`APIHubToolset`](https://google.github.io/adk-docs/api-reference/python/google-adk.html#google.adk.tools.APIHubToolset)
    - [`APIHubToolset.close()`](https://google.github.io/adk-docs/api-reference/python/google-adk.html#google.adk.tools.APIHubToolset.close)
    - [`APIHubToolset.get_tools()`](https://google.github.io/adk-docs/api-reference/python/google-adk.html#google.adk.tools.APIHubToolset.get_tools)
  - [`AgentTool`](https://google.github.io/adk-docs/api-reference/python/google-adk.html#google.adk.tools.AgentTool)
    - [`AgentTool.agent`](https://google.github.io/adk-docs/api-reference/python/google-adk.html#google.adk.tools.AgentTool.agent)
    - [`AgentTool.skip_summarization`](https://google.github.io/adk-docs/api-reference/python/google-adk.html#google.adk.tools.AgentTool.skip_summarization)
    - [`AgentTool.from_config()`](https://google.github.io/adk-docs/api-reference/python/google-adk.html#google.adk.tools.AgentTool.from_config)
    - [`AgentTool.populate_name()`](https://google.github.io/adk-docs/api-reference/python/google-adk.html#google.adk.tools.AgentTool.populate_name)
    - [`AgentTool.run_async()`](https://google.github.io/adk-docs/api-reference/python/google-adk.html#google.adk.tools.AgentTool.run_async)
  - [`AuthToolArguments`](https://google.github.io/adk-docs/api-reference/python/google-adk.html#google.adk.tools.AuthToolArguments)
    - [`AuthToolArguments.auth_config`](https://google.github.io/adk-docs/api-reference/python/google-adk.html#google.adk.tools.AuthToolArguments.auth_config)
    - [`AuthToolArguments.function_call_id`](https://google.github.io/adk-docs/api-reference/python/google-adk.html#google.adk.tools.AuthToolArguments.function_call_id)
  - [`BaseTool`](https://google.github.io/adk-docs/api-reference/python/google-adk.html#google.adk.tools.BaseTool)
    - [`BaseTool.custom_metadata`](https://google.github.io/adk-docs/api-reference/python/google-adk.html#google.adk.tools.BaseTool.custom_metadata)
    - [`BaseTool.description`](https://google.github.io/adk-docs/api-reference/python/google-adk.html#google.adk.tools.BaseTool.description)
    - [`BaseTool.from_config()`](https://google.github.io/adk-docs/api-reference/python/google-adk.html#google.adk.tools.BaseTool.from_config)
    - [`BaseTool.is_long_running`](https://google.github.io/adk-docs/api-reference/python/google-adk.html#google.adk.tools.BaseTool.is_long_running)
    - [`BaseTool.name`](https://google.github.io/adk-docs/api-reference/python/google-adk.html#google.adk.tools.BaseTool.name)
    - [`BaseTool.process_llm_request()`](https://google.github.io/adk-docs/api-reference/python/google-adk.html#google.adk.tools.BaseTool.process_llm_request)
    - [`BaseTool.run_async()`](https://google.github.io/adk-docs/api-reference/python/google-adk.html#google.adk.tools.BaseTool.run_async)
  - [`DiscoveryEngineSearchTool`](https://google.github.io/adk-docs/api-reference/python/google-adk.html#google.adk.tools.DiscoveryEngineSearchTool)
    - [`DiscoveryEngineSearchTool.discovery_engine_search()`](https://google.github.io/adk-docs/api-reference/python/google-adk.html#google.adk.tools.DiscoveryEngineSearchTool.discovery_engine_search)
  - [`ExampleTool`](https://google.github.io/adk-docs/api-reference/python/google-adk.html#google.adk.tools.ExampleTool)
    - [`ExampleTool.examples`](https://google.github.io/adk-docs/api-reference/python/google-adk.html#google.adk.tools.ExampleTool.examples)
    - [`ExampleTool.from_config()`](https://google.github.io/adk-docs/api-reference/python/google-adk.html#google.adk.tools.ExampleTool.from_config)
    - [`ExampleTool.process_llm_request()`](https://google.github.io/adk-docs/api-reference/python/google-adk.html#google.adk.tools.ExampleTool.process_llm_request)
  - [`FunctionTool`](https://google.github.io/adk-docs/api-reference/python/google-adk.html#google.adk.tools.FunctionTool)
    - [`FunctionTool.func`](https://google.github.io/adk-docs/api-reference/python/google-adk.html#google.adk.tools.FunctionTool.func)
    - [`FunctionTool.run_async()`](https://google.github.io/adk-docs/api-reference/python/google-adk.html#google.adk.tools.FunctionTool.run_async)
  - [`LongRunningFunctionTool`](https://google.github.io/adk-docs/api-reference/python/google-adk.html#google.adk.tools.LongRunningFunctionTool)
    - [`LongRunningFunctionTool.is_long_running`](https://google.github.io/adk-docs/api-reference/python/google-adk.html#google.adk.tools.LongRunningFunctionTool.is_long_running)
  - [`MCPToolset`](https://google.github.io/adk-docs/api-reference/python/google-adk.html#google.adk.tools.MCPToolset)
  - [`McpToolset`](https://google.github.io/adk-docs/api-reference/python/google-adk.html#google.adk.tools.McpToolset)
    - [`McpToolset.close()`](https://google.github.io/adk-docs/api-reference/python/google-adk.html#google.adk.tools.McpToolset.close)
    - [`McpToolset.from_config()`](https://google.github.io/adk-docs/api-reference/python/google-adk.html#google.adk.tools.McpToolset.from_config)
    - [`McpToolset.get_tools()`](https://google.github.io/adk-docs/api-reference/python/google-adk.html#google.adk.tools.McpToolset.get_tools)
  - [`ToolContext`](https://google.github.io/adk-docs/api-reference/python/google-adk.html#google.adk.tools.ToolContext)
    - [`ToolContext.invocation_context`](https://google.github.io/adk-docs/api-reference/python/google-adk.html#google.adk.tools.ToolContext.invocation_context)
    - [`ToolContext.function_call_id`](https://google.github.io/adk-docs/api-reference/python/google-adk.html#google.adk.tools.ToolContext.function_call_id)
    - [`ToolContext.event_actions`](https://google.github.io/adk-docs/api-reference/python/google-adk.html#google.adk.tools.ToolContext.event_actions)
    - [`ToolContext.tool_confirmation`](https://google.github.io/adk-docs/api-reference/python/google-adk.html#google.adk.tools.ToolContext.tool_confirmation)
    - [`ToolContext.actions`](https://google.github.io/adk-docs/api-reference/python/google-adk.html#google.adk.tools.ToolContext.actions)
    - [`ToolContext.get_auth_response()`](https://google.github.io/adk-docs/api-reference/python/google-adk.html#google.adk.tools.ToolContext.get_auth_response)
    - [`ToolContext.request_confirmation()`](https://google.github.io/adk-docs/api-reference/python/google-adk.html#google.adk.tools.ToolContext.request_confirmation)
    - [`ToolContext.request_credential()`](https://google.github.io/adk-docs/api-reference/python/google-adk.html#google.adk.tools.ToolContext.request_credential)
    - [`ToolContext.search_memory()`](https://google.github.io/adk-docs/api-reference/python/google-adk.html#google.adk.tools.ToolContext.search_memory)
  - [`VertexAiSearchTool`](https://google.github.io/adk-docs/api-reference/python/google-adk.html#google.adk.tools.VertexAiSearchTool)
    - [`VertexAiSearchTool.data_store_id`](https://google.github.io/adk-docs/api-reference/python/google-adk.html#google.adk.tools.VertexAiSearchTool.data_store_id)
    - [`VertexAiSearchTool.search_engine_id`](https://google.github.io/adk-docs/api-reference/python/google-adk.html#google.adk.tools.VertexAiSearchTool.search_engine_id)
    - [`VertexAiSearchTool.process_llm_request()`](https://google.github.io/adk-docs/api-reference/python/google-adk.html#google.adk.tools.VertexAiSearchTool.process_llm_request)
  - [`exit_loop()`](https://google.github.io/adk-docs/api-reference/python/google-adk.html#google.adk.tools.exit_loop)
  - [`transfer_to_agent()`](https://google.github.io/adk-docs/api-reference/python/google-adk.html#google.adk.tools.transfer_to_agent)
- [google.adk.tools.agent\_tool module](https://google.github.io/adk-docs/api-reference/python/google-adk.html#module-google.adk.tools.agent_tool)
  - [`AgentTool`](https://google.github.io/adk-docs/api-reference/python/google-adk.html#google.adk.tools.agent_tool.AgentTool)
    - [`AgentTool.agent`](https://google.github.io/adk-docs/api-reference/python/google-adk.html#google.adk.tools.agent_tool.AgentTool.agent)
    - [`AgentTool.skip_summarization`](https://google.github.io/adk-docs/api-reference/python/google-adk.html#google.adk.tools.agent_tool.AgentTool.skip_summarization)
    - [`AgentTool.from_config()`](https://google.github.io/adk-docs/api-reference/python/google-adk.html#google.adk.tools.agent_tool.AgentTool.from_config)
    - [`AgentTool.populate_name()`](https://google.github.io/adk-docs/api-reference/python/google-adk.html#google.adk.tools.agent_tool.AgentTool.populate_name)
    - [`AgentTool.run_async()`](https://google.github.io/adk-docs/api-reference/python/google-adk.html#google.adk.tools.agent_tool.AgentTool.run_async)
  - [`AgentToolConfig`](https://google.github.io/adk-docs/api-reference/python/google-adk.html#google.adk.tools.agent_tool.AgentToolConfig)
    - [`AgentToolConfig.agent`](https://google.github.io/adk-docs/api-reference/python/google-adk.html#google.adk.tools.agent_tool.AgentToolConfig.agent)
    - [`AgentToolConfig.skip_summarization`](https://google.github.io/adk-docs/api-reference/python/google-adk.html#google.adk.tools.agent_tool.AgentToolConfig.skip_summarization)
- [google.adk.tools.apihub\_tool module](https://google.github.io/adk-docs/api-reference/python/google-adk.html#module-google.adk.tools.apihub_tool)
  - [`APIHubToolset`](https://google.github.io/adk-docs/api-reference/python/google-adk.html#google.adk.tools.apihub_tool.APIHubToolset)
    - [`APIHubToolset.close()`](https://google.github.io/adk-docs/api-reference/python/google-adk.html#google.adk.tools.apihub_tool.APIHubToolset.close)
    - [`APIHubToolset.get_tools()`](https://google.github.io/adk-docs/api-reference/python/google-adk.html#google.adk.tools.apihub_tool.APIHubToolset.get_tools)
- [google.adk.tools.application\_integration\_tool module](https://google.github.io/adk-docs/api-reference/python/google-adk.html#module-google.adk.tools.application_integration_tool)
  - [`ApplicationIntegrationToolset`](https://google.github.io/adk-docs/api-reference/python/google-adk.html#google.adk.tools.application_integration_tool.ApplicationIntegrationToolset)
    - [`ApplicationIntegrationToolset.close()`](https://google.github.io/adk-docs/api-reference/python/google-adk.html#google.adk.tools.application_integration_tool.ApplicationIntegrationToolset.close)
    - [`ApplicationIntegrationToolset.get_tools()`](https://google.github.io/adk-docs/api-reference/python/google-adk.html#google.adk.tools.application_integration_tool.ApplicationIntegrationToolset.get_tools)
  - [`IntegrationConnectorTool`](https://google.github.io/adk-docs/api-reference/python/google-adk.html#google.adk.tools.application_integration_tool.IntegrationConnectorTool)
    - [`IntegrationConnectorTool.EXCLUDE_FIELDS`](https://google.github.io/adk-docs/api-reference/python/google-adk.html#google.adk.tools.application_integration_tool.IntegrationConnectorTool.EXCLUDE_FIELDS)
    - [`IntegrationConnectorTool.OPTIONAL_FIELDS`](https://google.github.io/adk-docs/api-reference/python/google-adk.html#google.adk.tools.application_integration_tool.IntegrationConnectorTool.OPTIONAL_FIELDS)
    - [`IntegrationConnectorTool.run_async()`](https://google.github.io/adk-docs/api-reference/python/google-adk.html#google.adk.tools.application_integration_tool.IntegrationConnectorTool.run_async)
- [google.adk.tools.authenticated\_function\_tool module](https://google.github.io/adk-docs/api-reference/python/google-adk.html#module-google.adk.tools.authenticated_function_tool)
  - [`AuthenticatedFunctionTool`](https://google.github.io/adk-docs/api-reference/python/google-adk.html#google.adk.tools.authenticated_function_tool.AuthenticatedFunctionTool)
    - [`AuthenticatedFunctionTool.run_async()`](https://google.github.io/adk-docs/api-reference/python/google-adk.html#google.adk.tools.authenticated_function_tool.AuthenticatedFunctionTool.run_async)
- [google.adk.tools.base\_authenticated\_tool module](https://google.github.io/adk-docs/api-reference/python/google-adk.html#module-google.adk.tools.base_authenticated_tool)
  - [`BaseAuthenticatedTool`](https://google.github.io/adk-docs/api-reference/python/google-adk.html#google.adk.tools.base_authenticated_tool.BaseAuthenticatedTool)
    - [`BaseAuthenticatedTool.run_async()`](https://google.github.io/adk-docs/api-reference/python/google-adk.html#google.adk.tools.base_authenticated_tool.BaseAuthenticatedTool.run_async)
- [google.adk.tools.base\_tool module](https://google.github.io/adk-docs/api-reference/python/google-adk.html#module-google.adk.tools.base_tool)
  - [`BaseTool`](https://google.github.io/adk-docs/api-reference/python/google-adk.html#google.adk.tools.base_tool.BaseTool)
    - [`BaseTool.custom_metadata`](https://google.github.io/adk-docs/api-reference/python/google-adk.html#google.adk.tools.base_tool.BaseTool.custom_metadata)
    - [`BaseTool.description`](https://google.github.io/adk-docs/api-reference/python/google-adk.html#google.adk.tools.base_tool.BaseTool.description)
    - [`BaseTool.from_config()`](https://google.github.io/adk-docs/api-reference/python/google-adk.html#google.adk.tools.base_tool.BaseTool.from_config)
    - [`BaseTool.is_long_running`](https://google.github.io/adk-docs/api-reference/python/google-adk.html#google.adk.tools.base_tool.BaseTool.is_long_running)
    - [`BaseTool.name`](https://google.github.io/adk-docs/api-reference/python/google-adk.html#google.adk.tools.base_tool.BaseTool.name)
    - [`BaseTool.process_llm_request()`](https://google.github.io/adk-docs/api-reference/python/google-adk.html#google.adk.tools.base_tool.BaseTool.process_llm_request)
    - [`BaseTool.run_async()`](https://google.github.io/adk-docs/api-reference/python/google-adk.html#google.adk.tools.base_tool.BaseTool.run_async)
- [google.adk.tools.base\_toolset module](https://google.github.io/adk-docs/api-reference/python/google-adk.html#module-google.adk.tools.base_toolset)
  - [`BaseToolset`](https://google.github.io/adk-docs/api-reference/python/google-adk.html#google.adk.tools.base_toolset.BaseToolset)
    - [`BaseToolset.close()`](https://google.github.io/adk-docs/api-reference/python/google-adk.html#google.adk.tools.base_toolset.BaseToolset.close)
    - [`BaseToolset.from_config()`](https://google.github.io/adk-docs/api-reference/python/google-adk.html#google.adk.tools.base_toolset.BaseToolset.from_config)
    - [`BaseToolset.get_tools()`](https://google.github.io/adk-docs/api-reference/python/google-adk.html#google.adk.tools.base_toolset.BaseToolset.get_tools)
    - [`BaseToolset.get_tools_with_prefix()`](https://google.github.io/adk-docs/api-reference/python/google-adk.html#google.adk.tools.base_toolset.BaseToolset.get_tools_with_prefix)
    - [`BaseToolset.process_llm_request()`](https://google.github.io/adk-docs/api-reference/python/google-adk.html#google.adk.tools.base_toolset.BaseToolset.process_llm_request)
  - [`ToolPredicate`](https://google.github.io/adk-docs/api-reference/python/google-adk.html#google.adk.tools.base_toolset.ToolPredicate)
- [google.adk.tools.bigquery module](https://google.github.io/adk-docs/api-reference/python/google-adk.html#module-google.adk.tools.bigquery)
  - [`BigQueryCredentialsConfig`](https://google.github.io/adk-docs/api-reference/python/google-adk.html#google.adk.tools.bigquery.BigQueryCredentialsConfig)
    - [`BigQueryCredentialsConfig.model_post_init()`](https://google.github.io/adk-docs/api-reference/python/google-adk.html#google.adk.tools.bigquery.BigQueryCredentialsConfig.model_post_init)
  - [`BigQueryToolset`](https://google.github.io/adk-docs/api-reference/python/google-adk.html#google.adk.tools.bigquery.BigQueryToolset)
    - [`BigQueryToolset.close()`](https://google.github.io/adk-docs/api-reference/python/google-adk.html#google.adk.tools.bigquery.BigQueryToolset.close)
    - [`BigQueryToolset.get_tools()`](https://google.github.io/adk-docs/api-reference/python/google-adk.html#google.adk.tools.bigquery.BigQueryToolset.get_tools)
- [google.adk.tools.crewai\_tool module](https://google.github.io/adk-docs/api-reference/python/google-adk.html#module-google.adk.tools.crewai_tool)
  - [`CrewaiTool`](https://google.github.io/adk-docs/api-reference/python/google-adk.html#google.adk.tools.crewai_tool.CrewaiTool)
    - [`CrewaiTool.from_config()`](https://google.github.io/adk-docs/api-reference/python/google-adk.html#google.adk.tools.crewai_tool.CrewaiTool.from_config)
    - [`CrewaiTool.run_async()`](https://google.github.io/adk-docs/api-reference/python/google-adk.html#google.adk.tools.crewai_tool.CrewaiTool.run_async)
    - [`CrewaiTool.tool`](https://google.github.io/adk-docs/api-reference/python/google-adk.html#google.adk.tools.crewai_tool.CrewaiTool.tool)
  - [`CrewaiToolConfig`](https://google.github.io/adk-docs/api-reference/python/google-adk.html#google.adk.tools.crewai_tool.CrewaiToolConfig)
    - [`CrewaiToolConfig.description`](https://google.github.io/adk-docs/api-reference/python/google-adk.html#google.adk.tools.crewai_tool.CrewaiToolConfig.description)
    - [`CrewaiToolConfig.name`](https://google.github.io/adk-docs/api-reference/python/google-adk.html#google.adk.tools.crewai_tool.CrewaiToolConfig.name)
    - [`CrewaiToolConfig.tool`](https://google.github.io/adk-docs/api-reference/python/google-adk.html#google.adk.tools.crewai_tool.CrewaiToolConfig.tool)
- [google.adk.tools.enterprise\_search\_tool module](https://google.github.io/adk-docs/api-reference/python/google-adk.html#module-google.adk.tools.enterprise_search_tool)
  - [`EnterpriseWebSearchTool`](https://google.github.io/adk-docs/api-reference/python/google-adk.html#google.adk.tools.enterprise_search_tool.EnterpriseWebSearchTool)
    - [`EnterpriseWebSearchTool.process_llm_request()`](https://google.github.io/adk-docs/api-reference/python/google-adk.html#google.adk.tools.enterprise_search_tool.EnterpriseWebSearchTool.process_llm_request)
- [google.adk.tools.example\_tool module](https://google.github.io/adk-docs/api-reference/python/google-adk.html#module-google.adk.tools.example_tool)
  - [`ExampleTool`](https://google.github.io/adk-docs/api-reference/python/google-adk.html#google.adk.tools.example_tool.ExampleTool)
    - [`ExampleTool.examples`](https://google.github.io/adk-docs/api-reference/python/google-adk.html#google.adk.tools.example_tool.ExampleTool.examples)
    - [`ExampleTool.from_config()`](https://google.github.io/adk-docs/api-reference/python/google-adk.html#google.adk.tools.example_tool.ExampleTool.from_config)
    - [`ExampleTool.process_llm_request()`](https://google.github.io/adk-docs/api-reference/python/google-adk.html#google.adk.tools.example_tool.ExampleTool.process_llm_request)
  - [`ExampleToolConfig`](https://google.github.io/adk-docs/api-reference/python/google-adk.html#google.adk.tools.example_tool.ExampleToolConfig)
    - [`ExampleToolConfig.examples`](https://google.github.io/adk-docs/api-reference/python/google-adk.html#google.adk.tools.example_tool.ExampleToolConfig.examples)
- [google.adk.tools.exit\_loop\_tool module](https://google.github.io/adk-docs/api-reference/python/google-adk.html#module-google.adk.tools.exit_loop_tool)
  - [`exit_loop()`](https://google.github.io/adk-docs/api-reference/python/google-adk.html#google.adk.tools.exit_loop_tool.exit_loop)
- [google.adk.tools.function\_tool module](https://google.github.io/adk-docs/api-reference/python/google-adk.html#module-google.adk.tools.function_tool)
  - [`FunctionTool`](https://google.github.io/adk-docs/api-reference/python/google-adk.html#google.adk.tools.function_tool.FunctionTool)
    - [`FunctionTool.func`](https://google.github.io/adk-docs/api-reference/python/google-adk.html#google.adk.tools.function_tool.FunctionTool.func)
    - [`FunctionTool.run_async()`](https://google.github.io/adk-docs/api-reference/python/google-adk.html#google.adk.tools.function_tool.FunctionTool.run_async)
- [google.adk.tools.get\_user\_choice\_tool module](https://google.github.io/adk-docs/api-reference/python/google-adk.html#module-google.adk.tools.get_user_choice_tool)
  - [`get_user_choice()`](https://google.github.io/adk-docs/api-reference/python/google-adk.html#google.adk.tools.get_user_choice_tool.get_user_choice)
- [google.adk.tools.google\_api\_tool module](https://google.github.io/adk-docs/api-reference/python/google-adk.html#module-google.adk.tools.google_api_tool)
  - [`BigQueryToolset`](https://google.github.io/adk-docs/api-reference/python/google-adk.html#google.adk.tools.google_api_tool.BigQueryToolset)
  - [`CalendarToolset`](https://google.github.io/adk-docs/api-reference/python/google-adk.html#google.adk.tools.google_api_tool.CalendarToolset)
  - [`DocsToolset`](https://google.github.io/adk-docs/api-reference/python/google-adk.html#google.adk.tools.google_api_tool.DocsToolset)
  - [`GmailToolset`](https://google.github.io/adk-docs/api-reference/python/google-adk.html#google.adk.tools.google_api_tool.GmailToolset)
  - [`GoogleApiTool`](https://google.github.io/adk-docs/api-reference/python/google-adk.html#google.adk.tools.google_api_tool.GoogleApiTool)
    - [`GoogleApiTool.configure_auth()`](https://google.github.io/adk-docs/api-reference/python/google-adk.html#google.adk.tools.google_api_tool.GoogleApiTool.configure_auth)
    - [`GoogleApiTool.configure_sa_auth()`](https://google.github.io/adk-docs/api-reference/python/google-adk.html#google.adk.tools.google_api_tool.GoogleApiTool.configure_sa_auth)
    - [`GoogleApiTool.run_async()`](https://google.github.io/adk-docs/api-reference/python/google-adk.html#google.adk.tools.google_api_tool.GoogleApiTool.run_async)
  - [`GoogleApiToolset`](https://google.github.io/adk-docs/api-reference/python/google-adk.html#google.adk.tools.google_api_tool.GoogleApiToolset)
    - [`GoogleApiToolset.close()`](https://google.github.io/adk-docs/api-reference/python/google-adk.html#google.adk.tools.google_api_tool.GoogleApiToolset.close)
    - [`GoogleApiToolset.configure_auth()`](https://google.github.io/adk-docs/api-reference/python/google-adk.html#google.adk.tools.google_api_tool.GoogleApiToolset.configure_auth)
    - [`GoogleApiToolset.configure_sa_auth()`](https://google.github.io/adk-docs/api-reference/python/google-adk.html#google.adk.tools.google_api_tool.GoogleApiToolset.configure_sa_auth)
    - [`GoogleApiToolset.get_tools()`](https://google.github.io/adk-docs/api-reference/python/google-adk.html#google.adk.tools.google_api_tool.GoogleApiToolset.get_tools)
    - [`GoogleApiToolset.set_tool_filter()`](https://google.github.io/adk-docs/api-reference/python/google-adk.html#google.adk.tools.google_api_tool.GoogleApiToolset.set_tool_filter)
  - [`SheetsToolset`](https://google.github.io/adk-docs/api-reference/python/google-adk.html#google.adk.tools.google_api_tool.SheetsToolset)
  - [`SlidesToolset`](https://google.github.io/adk-docs/api-reference/python/google-adk.html#google.adk.tools.google_api_tool.SlidesToolset)
  - [`YoutubeToolset`](https://google.github.io/adk-docs/api-reference/python/google-adk.html#google.adk.tools.google_api_tool.YoutubeToolset)
- [google.adk.tools.google\_maps\_grounding\_tool module](https://google.github.io/adk-docs/api-reference/python/google-adk.html#google-adk-tools-google-maps-grounding-tool-module)
  - [`GoogleMapsGroundingTool`](https://google.github.io/adk-docs/api-reference/python/google-adk.html#google.adk.tools.google_maps_grounding_tool.GoogleMapsGroundingTool)
    - [`GoogleMapsGroundingTool.process_llm_request()`](https://google.github.io/adk-docs/api-reference/python/google-adk.html#google.adk.tools.google_maps_grounding_tool.GoogleMapsGroundingTool.process_llm_request)
- [google.adk.tools.google\_search\_tool module](https://google.github.io/adk-docs/api-reference/python/google-adk.html#module-google.adk.tools.google_search_tool)
  - [`GoogleSearchTool`](https://google.github.io/adk-docs/api-reference/python/google-adk.html#google.adk.tools.google_search_tool.GoogleSearchTool)
    - [`GoogleSearchTool.process_llm_request()`](https://google.github.io/adk-docs/api-reference/python/google-adk.html#google.adk.tools.google_search_tool.GoogleSearchTool.process_llm_request)
- [google.adk.tools.langchain\_tool module](https://google.github.io/adk-docs/api-reference/python/google-adk.html#module-google.adk.tools.langchain_tool)
  - [`LangchainTool`](https://google.github.io/adk-docs/api-reference/python/google-adk.html#google.adk.tools.langchain_tool.LangchainTool)
    - [`LangchainTool.from_config()`](https://google.github.io/adk-docs/api-reference/python/google-adk.html#google.adk.tools.langchain_tool.LangchainTool.from_config)
  - [`LangchainToolConfig`](https://google.github.io/adk-docs/api-reference/python/google-adk.html#google.adk.tools.langchain_tool.LangchainToolConfig)
    - [`LangchainToolConfig.description`](https://google.github.io/adk-docs/api-reference/python/google-adk.html#google.adk.tools.langchain_tool.LangchainToolConfig.description)
    - [`LangchainToolConfig.name`](https://google.github.io/adk-docs/api-reference/python/google-adk.html#google.adk.tools.langchain_tool.LangchainToolConfig.name)
    - [`LangchainToolConfig.tool`](https://google.github.io/adk-docs/api-reference/python/google-adk.html#google.adk.tools.langchain_tool.LangchainToolConfig.tool)
- [google.adk.tools.load\_artifacts\_tool module](https://google.github.io/adk-docs/api-reference/python/google-adk.html#module-google.adk.tools.load_artifacts_tool)
  - [`LoadArtifactsTool`](https://google.github.io/adk-docs/api-reference/python/google-adk.html#google.adk.tools.load_artifacts_tool.LoadArtifactsTool)
    - [`LoadArtifactsTool.process_llm_request()`](https://google.github.io/adk-docs/api-reference/python/google-adk.html#google.adk.tools.load_artifacts_tool.LoadArtifactsTool.process_llm_request)
    - [`LoadArtifactsTool.run_async()`](https://google.github.io/adk-docs/api-reference/python/google-adk.html#google.adk.tools.load_artifacts_tool.LoadArtifactsTool.run_async)
- [google.adk.tools.load\_memory\_tool module](https://google.github.io/adk-docs/api-reference/python/google-adk.html#module-google.adk.tools.load_memory_tool)
  - [`LoadMemoryResponse`](https://google.github.io/adk-docs/api-reference/python/google-adk.html#google.adk.tools.load_memory_tool.LoadMemoryResponse)
    - [`LoadMemoryResponse.memories`](https://google.github.io/adk-docs/api-reference/python/google-adk.html#google.adk.tools.load_memory_tool.LoadMemoryResponse.memories)
  - [`LoadMemoryTool`](https://google.github.io/adk-docs/api-reference/python/google-adk.html#google.adk.tools.load_memory_tool.LoadMemoryTool)
    - [`LoadMemoryTool.process_llm_request()`](https://google.github.io/adk-docs/api-reference/python/google-adk.html#google.adk.tools.load_memory_tool.LoadMemoryTool.process_llm_request)
  - [`load_memory()`](https://google.github.io/adk-docs/api-reference/python/google-adk.html#google.adk.tools.load_memory_tool.load_memory)
- [google.adk.tools.load\_web\_page module](https://google.github.io/adk-docs/api-reference/python/google-adk.html#module-google.adk.tools.load_web_page)
  - [`load_web_page()`](https://google.github.io/adk-docs/api-reference/python/google-adk.html#google.adk.tools.load_web_page.load_web_page)
- [google.adk.tools.long\_running\_tool module](https://google.github.io/adk-docs/api-reference/python/google-adk.html#module-google.adk.tools.long_running_tool)
  - [`LongRunningFunctionTool`](https://google.github.io/adk-docs/api-reference/python/google-adk.html#google.adk.tools.long_running_tool.LongRunningFunctionTool)
    - [`LongRunningFunctionTool.is_long_running`](https://google.github.io/adk-docs/api-reference/python/google-adk.html#google.adk.tools.long_running_tool.LongRunningFunctionTool.is_long_running)
- [google.adk.tools.mcp\_tool module](https://google.github.io/adk-docs/api-reference/python/google-adk.html#module-google.adk.tools.mcp_tool)
  - [`MCPTool`](https://google.github.io/adk-docs/api-reference/python/google-adk.html#google.adk.tools.mcp_tool.MCPTool)
  - [`MCPToolset`](https://google.github.io/adk-docs/api-reference/python/google-adk.html#google.adk.tools.mcp_tool.MCPToolset)
  - [`McpTool`](https://google.github.io/adk-docs/api-reference/python/google-adk.html#google.adk.tools.mcp_tool.McpTool)
    - [`McpTool.raw_mcp_tool`](https://google.github.io/adk-docs/api-reference/python/google-adk.html#google.adk.tools.mcp_tool.McpTool.raw_mcp_tool)
    - [`McpTool.run_async()`](https://google.github.io/adk-docs/api-reference/python/google-adk.html#google.adk.tools.mcp_tool.McpTool.run_async)
  - [`McpToolset`](https://google.github.io/adk-docs/api-reference/python/google-adk.html#google.adk.tools.mcp_tool.McpToolset)
    - [`McpToolset.close()`](https://google.github.io/adk-docs/api-reference/python/google-adk.html#google.adk.tools.mcp_tool.McpToolset.close)
    - [`McpToolset.from_config()`](https://google.github.io/adk-docs/api-reference/python/google-adk.html#google.adk.tools.mcp_tool.McpToolset.from_config)
    - [`McpToolset.get_tools()`](https://google.github.io/adk-docs/api-reference/python/google-adk.html#google.adk.tools.mcp_tool.McpToolset.get_tools)
  - [`SseConnectionParams`](https://google.github.io/adk-docs/api-reference/python/google-adk.html#google.adk.tools.mcp_tool.SseConnectionParams)
    - [`SseConnectionParams.url`](https://google.github.io/adk-docs/api-reference/python/google-adk.html#google.adk.tools.mcp_tool.SseConnectionParams.url)
    - [`SseConnectionParams.headers`](https://google.github.io/adk-docs/api-reference/python/google-adk.html#google.adk.tools.mcp_tool.SseConnectionParams.headers)
    - [`SseConnectionParams.timeout`](https://google.github.io/adk-docs/api-reference/python/google-adk.html#google.adk.tools.mcp_tool.SseConnectionParams.timeout)
    - [`SseConnectionParams.sse_read_timeout`](https://google.github.io/adk-docs/api-reference/python/google-adk.html#google.adk.tools.mcp_tool.SseConnectionParams.sse_read_timeout)
    - [`SseConnectionParams.headers`](https://google.github.io/adk-docs/api-reference/python/google-adk.html#id51)
    - [`SseConnectionParams.sse_read_timeout`](https://google.github.io/adk-docs/api-reference/python/google-adk.html#id52)
    - [`SseConnectionParams.timeout`](https://google.github.io/adk-docs/api-reference/python/google-adk.html#id53)
    - [`SseConnectionParams.url`](https://google.github.io/adk-docs/api-reference/python/google-adk.html#id54)
  - [`StdioConnectionParams`](https://google.github.io/adk-docs/api-reference/python/google-adk.html#google.adk.tools.mcp_tool.StdioConnectionParams)
    - [`StdioConnectionParams.server_params`](https://google.github.io/adk-docs/api-reference/python/google-adk.html#google.adk.tools.mcp_tool.StdioConnectionParams.server_params)
    - [`StdioConnectionParams.timeout`](https://google.github.io/adk-docs/api-reference/python/google-adk.html#google.adk.tools.mcp_tool.StdioConnectionParams.timeout)
    - [`StdioConnectionParams.server_params`](https://google.github.io/adk-docs/api-reference/python/google-adk.html#id55)
    - [`StdioConnectionParams.timeout`](https://google.github.io/adk-docs/api-reference/python/google-adk.html#id56)
  - [`StreamableHTTPConnectionParams`](https://google.github.io/adk-docs/api-reference/python/google-adk.html#google.adk.tools.mcp_tool.StreamableHTTPConnectionParams)
    - [`StreamableHTTPConnectionParams.url`](https://google.github.io/adk-docs/api-reference/python/google-adk.html#google.adk.tools.mcp_tool.StreamableHTTPConnectionParams.url)
    - [`StreamableHTTPConnectionParams.headers`](https://google.github.io/adk-docs/api-reference/python/google-adk.html#google.adk.tools.mcp_tool.StreamableHTTPConnectionParams.headers)
    - [`StreamableHTTPConnectionParams.timeout`](https://google.github.io/adk-docs/api-reference/python/google-adk.html#google.adk.tools.mcp_tool.StreamableHTTPConnectionParams.timeout)
    - [`StreamableHTTPConnectionParams.sse_read_timeout`](https://google.github.io/adk-docs/api-reference/python/google-adk.html#google.adk.tools.mcp_tool.StreamableHTTPConnectionParams.sse_read_timeout)
    - [`StreamableHTTPConnectionParams.terminate_on_close`](https://google.github.io/adk-docs/api-reference/python/google-adk.html#google.adk.tools.mcp_tool.StreamableHTTPConnectionParams.terminate_on_close)
    - [`StreamableHTTPConnectionParams.headers`](https://google.github.io/adk-docs/api-reference/python/google-adk.html#id57)
    - [`StreamableHTTPConnectionParams.sse_read_timeout`](https://google.github.io/adk-docs/api-reference/python/google-adk.html#id58)
    - [`StreamableHTTPConnectionParams.terminate_on_close`](https://google.github.io/adk-docs/api-reference/python/google-adk.html#id59)
    - [`StreamableHTTPConnectionParams.timeout`](https://google.github.io/adk-docs/api-reference/python/google-adk.html#id60)
    - [`StreamableHTTPConnectionParams.url`](https://google.github.io/adk-docs/api-reference/python/google-adk.html#id61)
  - [`adk_to_mcp_tool_type()`](https://google.github.io/adk-docs/api-reference/python/google-adk.html#google.adk.tools.mcp_tool.adk_to_mcp_tool_type)
  - [`gemini_to_json_schema()`](https://google.github.io/adk-docs/api-reference/python/google-adk.html#google.adk.tools.mcp_tool.gemini_to_json_schema)
- [google.adk.tools.openapi\_tool module](https://google.github.io/adk-docs/api-reference/python/google-adk.html#module-google.adk.tools.openapi_tool)
  - [`OpenAPIToolset`](https://google.github.io/adk-docs/api-reference/python/google-adk.html#google.adk.tools.openapi_tool.OpenAPIToolset)
    - [`OpenAPIToolset.close()`](https://google.github.io/adk-docs/api-reference/python/google-adk.html#google.adk.tools.openapi_tool.OpenAPIToolset.close)
    - [`OpenAPIToolset.get_tool()`](https://google.github.io/adk-docs/api-reference/python/google-adk.html#google.adk.tools.openapi_tool.OpenAPIToolset.get_tool)
    - [`OpenAPIToolset.get_tools()`](https://google.github.io/adk-docs/api-reference/python/google-adk.html#google.adk.tools.openapi_tool.OpenAPIToolset.get_tools)
  - [`RestApiTool`](https://google.github.io/adk-docs/api-reference/python/google-adk.html#google.adk.tools.openapi_tool.RestApiTool)
    - [`RestApiTool.call()`](https://google.github.io/adk-docs/api-reference/python/google-adk.html#google.adk.tools.openapi_tool.RestApiTool.call)
    - [`RestApiTool.configure_auth_credential()`](https://google.github.io/adk-docs/api-reference/python/google-adk.html#google.adk.tools.openapi_tool.RestApiTool.configure_auth_credential)
    - [`RestApiTool.configure_auth_scheme()`](https://google.github.io/adk-docs/api-reference/python/google-adk.html#google.adk.tools.openapi_tool.RestApiTool.configure_auth_scheme)
    - [`RestApiTool.from_parsed_operation()`](https://google.github.io/adk-docs/api-reference/python/google-adk.html#google.adk.tools.openapi_tool.RestApiTool.from_parsed_operation)
    - [`RestApiTool.from_parsed_operation_str()`](https://google.github.io/adk-docs/api-reference/python/google-adk.html#google.adk.tools.openapi_tool.RestApiTool.from_parsed_operation_str)
    - [`RestApiTool.run_async()`](https://google.github.io/adk-docs/api-reference/python/google-adk.html#google.adk.tools.openapi_tool.RestApiTool.run_async)
    - [`RestApiTool.set_default_headers()`](https://google.github.io/adk-docs/api-reference/python/google-adk.html#google.adk.tools.openapi_tool.RestApiTool.set_default_headers)
- [google.adk.tools.preload\_memory\_tool module](https://google.github.io/adk-docs/api-reference/python/google-adk.html#module-google.adk.tools.preload_memory_tool)
  - [`PreloadMemoryTool`](https://google.github.io/adk-docs/api-reference/python/google-adk.html#google.adk.tools.preload_memory_tool.PreloadMemoryTool)
    - [`PreloadMemoryTool.process_llm_request()`](https://google.github.io/adk-docs/api-reference/python/google-adk.html#google.adk.tools.preload_memory_tool.PreloadMemoryTool.process_llm_request)
- [google.adk.tools.retrieval module](https://google.github.io/adk-docs/api-reference/python/google-adk.html#module-google.adk.tools.retrieval)
  - [`BaseRetrievalTool`](https://google.github.io/adk-docs/api-reference/python/google-adk.html#google.adk.tools.retrieval.BaseRetrievalTool)
- [google.adk.tools.tool\_context module](https://google.github.io/adk-docs/api-reference/python/google-adk.html#module-google.adk.tools.tool_context)
  - [`ToolContext`](https://google.github.io/adk-docs/api-reference/python/google-adk.html#google.adk.tools.tool_context.ToolContext)
    - [`ToolContext.invocation_context`](https://google.github.io/adk-docs/api-reference/python/google-adk.html#google.adk.tools.tool_context.ToolContext.invocation_context)
    - [`ToolContext.function_call_id`](https://google.github.io/adk-docs/api-reference/python/google-adk.html#google.adk.tools.tool_context.ToolContext.function_call_id)
    - [`ToolContext.event_actions`](https://google.github.io/adk-docs/api-reference/python/google-adk.html#google.adk.tools.tool_context.ToolContext.event_actions)
    - [`ToolContext.tool_confirmation`](https://google.github.io/adk-docs/api-reference/python/google-adk.html#google.adk.tools.tool_context.ToolContext.tool_confirmation)
    - [`ToolContext.actions`](https://google.github.io/adk-docs/api-reference/python/google-adk.html#google.adk.tools.tool_context.ToolContext.actions)
    - [`ToolContext.get_auth_response()`](https://google.github.io/adk-docs/api-reference/python/google-adk.html#google.adk.tools.tool_context.ToolContext.get_auth_response)
    - [`ToolContext.request_confirmation()`](https://google.github.io/adk-docs/api-reference/python/google-adk.html#google.adk.tools.tool_context.ToolContext.request_confirmation)
    - [`ToolContext.request_credential()`](https://google.github.io/adk-docs/api-reference/python/google-adk.html#google.adk.tools.tool_context.ToolContext.request_credential)
    - [`ToolContext.search_memory()`](https://google.github.io/adk-docs/api-reference/python/google-adk.html#google.adk.tools.tool_context.ToolContext.search_memory)
- [google.adk.tools.toolbox\_toolset module](https://google.github.io/adk-docs/api-reference/python/google-adk.html#module-google.adk.tools.toolbox_toolset)
  - [`ToolboxToolset`](https://google.github.io/adk-docs/api-reference/python/google-adk.html#google.adk.tools.toolbox_toolset.ToolboxToolset)
    - [`ToolboxToolset.close()`](https://google.github.io/adk-docs/api-reference/python/google-adk.html#google.adk.tools.toolbox_toolset.ToolboxToolset.close)
    - [`ToolboxToolset.get_tools()`](https://google.github.io/adk-docs/api-reference/python/google-adk.html#google.adk.tools.toolbox_toolset.ToolboxToolset.get_tools)
- [google.adk.tools.transfer\_to\_agent\_tool module](https://google.github.io/adk-docs/api-reference/python/google-adk.html#module-google.adk.tools.transfer_to_agent_tool)
  - [`transfer_to_agent()`](https://google.github.io/adk-docs/api-reference/python/google-adk.html#google.adk.tools.transfer_to_agent_tool.transfer_to_agent)
- [google.adk.tools.url\_context\_tool module](https://google.github.io/adk-docs/api-reference/python/google-adk.html#module-google.adk.tools.url_context_tool)
  - [`UrlContextTool`](https://google.github.io/adk-docs/api-reference/python/google-adk.html#google.adk.tools.url_context_tool.UrlContextTool)
    - [`UrlContextTool.process_llm_request()`](https://google.github.io/adk-docs/api-reference/python/google-adk.html#google.adk.tools.url_context_tool.UrlContextTool.process_llm_request)
- [google.adk.tools.vertex\_ai\_search\_tool module](https://google.github.io/adk-docs/api-reference/python/google-adk.html#module-google.adk.tools.vertex_ai_search_tool)
  - [`VertexAiSearchTool`](https://google.github.io/adk-docs/api-reference/python/google-adk.html#google.adk.tools.vertex_ai_search_tool.VertexAiSearchTool)
    - [`VertexAiSearchTool.data_store_id`](https://google.github.io/adk-docs/api-reference/python/google-adk.html#google.adk.tools.vertex_ai_search_tool.VertexAiSearchTool.data_store_id)
    - [`VertexAiSearchTool.search_engine_id`](https://google.github.io/adk-docs/api-reference/python/google-adk.html#google.adk.tools.vertex_ai_search_tool.VertexAiSearchTool.search_engine_id)
    - [`VertexAiSearchTool.process_llm_request()`](https://google.github.io/adk-docs/api-reference/python/google-adk.html#google.adk.tools.vertex_ai_search_tool.VertexAiSearchTool.process_llm_request)
- [google.adk.utils module](https://google.github.io/adk-docs/api-reference/python/google-adk.html#module-google.adk.utils)
- [google.adk.version module](https://google.github.io/adk-docs/api-reference/python/google-adk.html#module-google.adk.version)

## Tool Predicate Interface
Functional Interface:This is a functional interface and can therefore be used as the assignment target for a lambda expression or method reference.

* * *

[@FunctionalInterface](https://docs.oracle.com/en/java/javase/17/docs/api/java.base/java/lang/FunctionalInterface.html "class or interface in java.lang")public interface ToolPredicate

Functional interface to decide whether a tool should be exposed to the LLM based on the current
context.

- ## Method Summary [![Link icon](https://google.github.io/adk-docs/api-reference/java/resource-files/link.svg)](https://google.github.io/adk-docs/api-reference/java/com/google/adk/tools/ToolPredicate.html\#method-summary)





All MethodsInstance MethodsAbstract Methods







Modifier and Type



Method



Description



`boolean`



`test(BaseTool tool,
Optional<ReadonlyContext> readonlyContext)`





Decides if the given tool is selected.


- ## Method Details [![Link icon](https://google.github.io/adk-docs/api-reference/java/resource-files/link.svg)](https://google.github.io/adk-docs/api-reference/java/com/google/adk/tools/ToolPredicate.html\#method-detail)



- ### test [![Link icon](https://google.github.io/adk-docs/api-reference/java/resource-files/link.svg)](https://google.github.io/adk-docs/api-reference/java/com/google/adk/tools/ToolPredicate.html\#test(com.google.adk.tools.BaseTool,java.util.Optional))





booleantest( [BaseTool](https://google.github.io/adk-docs/api-reference/java/com/google/adk/tools/BaseTool.html "class in com.google.adk.tools") tool,
[Optional](https://docs.oracle.com/en/java/javase/17/docs/api/java.base/java/util/Optional.html "class or interface in java.util") < [ReadonlyContext](https://google.github.io/adk-docs/api-reference/java/com/google/adk/agents/ReadonlyContext.html "class in com.google.adk.agents") > readonlyContext)



Decides if the given tool is selected.

Parameters:`tool` \- The tool to check.`readonlyContext` \- The current context.Returns:true if the tool should be selected, false otherwise.

## LlmAgent Builder Class
[java.lang.Object](https://docs.oracle.com/en/java/javase/17/docs/api/java.base/java/lang/Object.html "class or interface in java.lang")

com.google.adk.agents.LlmAgent.Builder

Enclosing class:`LlmAgent`

* * *

public static class LlmAgent.Builderextends [Object](https://docs.oracle.com/en/java/javase/17/docs/api/java.base/java/lang/Object.html "class or interface in java.lang")

Builder for [`LlmAgent`](https://google.github.io/adk-docs/api-reference/java/com/google/adk/agents/LlmAgent.html "class in com.google.adk.agents").

- ## Constructor Summary [![Link icon](https://google.github.io/adk-docs/api-reference/java/resource-files/link.svg)](https://google.github.io/adk-docs/api-reference/java/com/google/adk/agents/LlmAgent.Builder.html\#constructor-summary)



Constructors





Constructor



Description



`Builder()`

- ## Method Summary [![Link icon](https://google.github.io/adk-docs/api-reference/java/resource-files/link.svg)](https://google.github.io/adk-docs/api-reference/java/com/google/adk/agents/LlmAgent.Builder.html\#method-summary)





All MethodsInstance MethodsConcrete Methods







Modifier and Type



Method



Description



`LlmAgent.Builder`



`afterAgentCallback(Callbacks.AfterAgentCallback afterAgentCallback)`







`LlmAgent.Builder`



`afterAgentCallback(List<com.google.adk.agents.Callbacks.AfterAgentCallbackBase> afterAgentCallback)`







`LlmAgent.Builder`



`afterAgentCallbackSync(Callbacks.AfterAgentCallbackSync afterAgentCallbackSync)`







`LlmAgent.Builder`



`afterModelCallback(Callbacks.AfterModelCallback afterModelCallback)`







`LlmAgent.Builder`



`afterModelCallback(List<com.google.adk.agents.Callbacks.AfterModelCallbackBase> afterModelCallback)`







`LlmAgent.Builder`



`afterModelCallbackSync(Callbacks.AfterModelCallbackSync afterModelCallbackSync)`







`LlmAgent.Builder`



`afterToolCallback(Callbacks.AfterToolCallback afterToolCallback)`







`LlmAgent.Builder`



`afterToolCallback(List<com.google.adk.agents.Callbacks.AfterToolCallbackBase> afterToolCallbacks)`







`LlmAgent.Builder`



`afterToolCallbackSync(Callbacks.AfterToolCallbackSync afterToolCallbackSync)`







`LlmAgent.Builder`



`beforeAgentCallback(Callbacks.BeforeAgentCallback beforeAgentCallback)`







`LlmAgent.Builder`



`beforeAgentCallback(List<com.google.adk.agents.Callbacks.BeforeAgentCallbackBase> beforeAgentCallback)`







`LlmAgent.Builder`



`beforeAgentCallbackSync(Callbacks.BeforeAgentCallbackSync beforeAgentCallbackSync)`







`LlmAgent.Builder`



`beforeModelCallback(Callbacks.BeforeModelCallback beforeModelCallback)`







`LlmAgent.Builder`



`beforeModelCallback(List<com.google.adk.agents.Callbacks.BeforeModelCallbackBase> beforeModelCallback)`







`LlmAgent.Builder`



`beforeModelCallbackSync(Callbacks.BeforeModelCallbackSync beforeModelCallbackSync)`







`LlmAgent.Builder`



`beforeToolCallback(Callbacks.BeforeToolCallback beforeToolCallback)`







`LlmAgent.Builder`



`beforeToolCallback(List<com.google.adk.agents.Callbacks.BeforeToolCallbackBase> beforeToolCallbacks)`







`LlmAgent.Builder`



`beforeToolCallbackSync(Callbacks.BeforeToolCallbackSync beforeToolCallbackSync)`







`LlmAgent`



`build()`







`LlmAgent.Builder`



`description(String description)`







`LlmAgent.Builder`



`disallowTransferToParent(boolean disallowTransferToParent)`







`LlmAgent.Builder`



`disallowTransferToPeers(boolean disallowTransferToPeers)`







`LlmAgent.Builder`



`exampleProvider(BaseExampleProvider exampleProvider)`







`LlmAgent.Builder`



`exampleProvider(Example... examples)`







`LlmAgent.Builder`



`exampleProvider(List<Example> examples)`







`LlmAgent.Builder`



`executor(Executor executor)`







`LlmAgent.Builder`



`generateContentConfig(com.google.genai.types.GenerateContentConfig generateContentConfig)`







`LlmAgent.Builder`



`globalInstruction(Instruction globalInstruction)`







`LlmAgent.Builder`



`globalInstruction(String globalInstruction)`







`LlmAgent.Builder`



`includeContents(LlmAgent.IncludeContents includeContents)`







`LlmAgent.Builder`



`inputSchema(com.google.genai.types.Schema inputSchema)`







`LlmAgent.Builder`



`instruction(Instruction instruction)`







`LlmAgent.Builder`



`instruction(String instruction)`







`LlmAgent.Builder`



`maxSteps(int maxSteps)`







`LlmAgent.Builder`



`model(BaseLlm model)`







`LlmAgent.Builder`



`model(String model)`







`LlmAgent.Builder`



`name(String name)`







`LlmAgent.Builder`



`outputKey(String outputKey)`







`LlmAgent.Builder`



`outputSchema(com.google.genai.types.Schema outputSchema)`







`LlmAgent.Builder`



`planning(boolean planning)`







`LlmAgent.Builder`



`subAgents(BaseAgent... subAgents)`







`LlmAgent.Builder`



`subAgents(List<? extends BaseAgent> subAgents)`







`LlmAgent.Builder`



`tools(Object... tools)`







`LlmAgent.Builder`



`tools(List<?> tools)`







`protected void`



`validate()`















### Methods inherited from class java.lang. [Object](https://docs.oracle.com/en/java/javase/17/docs/api/java.base/java/lang/Object.html "class or interface in java.lang") [![Link icon](https://google.github.io/adk-docs/api-reference/java/resource-files/link.svg)](https://google.github.io/adk-docs/api-reference/java/com/google/adk/agents/LlmAgent.Builder.html\#methods-inherited-from-class-java.lang.Object)

`clone, equals, finalize, getClass, hashCode, notify, notifyAll, toString, wait, wait, wait`


- ## Constructor Details [![Link icon](https://google.github.io/adk-docs/api-reference/java/resource-files/link.svg)](https://google.github.io/adk-docs/api-reference/java/com/google/adk/agents/LlmAgent.Builder.html\#constructor-detail)



- ### Builder [![Link icon](https://google.github.io/adk-docs/api-reference/java/resource-files/link.svg)](https://google.github.io/adk-docs/api-reference/java/com/google/adk/agents/LlmAgent.Builder.html\#%3Cinit%3E())





publicBuilder()


- ## Method Details [![Link icon](https://google.github.io/adk-docs/api-reference/java/resource-files/link.svg)](https://google.github.io/adk-docs/api-reference/java/com/google/adk/agents/LlmAgent.Builder.html\#method-detail)



- ### name [![Link icon](https://google.github.io/adk-docs/api-reference/java/resource-files/link.svg)](https://google.github.io/adk-docs/api-reference/java/com/google/adk/agents/LlmAgent.Builder.html\#name(java.lang.String))





@CanIgnoreReturnValue
public[LlmAgent.Builder](https://google.github.io/adk-docs/api-reference/java/com/google/adk/agents/LlmAgent.Builder.html "class in com.google.adk.agents")name( [String](https://docs.oracle.com/en/java/javase/17/docs/api/java.base/java/lang/String.html "class or interface in java.lang") name)

- ### description [![Link icon](https://google.github.io/adk-docs/api-reference/java/resource-files/link.svg)](https://google.github.io/adk-docs/api-reference/java/com/google/adk/agents/LlmAgent.Builder.html\#description(java.lang.String))





@CanIgnoreReturnValue
public[LlmAgent.Builder](https://google.github.io/adk-docs/api-reference/java/com/google/adk/agents/LlmAgent.Builder.html "class in com.google.adk.agents")description( [String](https://docs.oracle.com/en/java/javase/17/docs/api/java.base/java/lang/String.html "class or interface in java.lang") description)

- ### model [![Link icon](https://google.github.io/adk-docs/api-reference/java/resource-files/link.svg)](https://google.github.io/adk-docs/api-reference/java/com/google/adk/agents/LlmAgent.Builder.html\#model(java.lang.String))





@CanIgnoreReturnValue
public[LlmAgent.Builder](https://google.github.io/adk-docs/api-reference/java/com/google/adk/agents/LlmAgent.Builder.html "class in com.google.adk.agents")model( [String](https://docs.oracle.com/en/java/javase/17/docs/api/java.base/java/lang/String.html "class or interface in java.lang") model)

- ### model [![Link icon](https://google.github.io/adk-docs/api-reference/java/resource-files/link.svg)](https://google.github.io/adk-docs/api-reference/java/com/google/adk/agents/LlmAgent.Builder.html\#model(com.google.adk.models.BaseLlm))





@CanIgnoreReturnValue
public[LlmAgent.Builder](https://google.github.io/adk-docs/api-reference/java/com/google/adk/agents/LlmAgent.Builder.html "class in com.google.adk.agents")model( [BaseLlm](https://google.github.io/adk-docs/api-reference/java/com/google/adk/models/BaseLlm.html "class in com.google.adk.models") model)

- ### instruction [![Link icon](https://google.github.io/adk-docs/api-reference/java/resource-files/link.svg)](https://google.github.io/adk-docs/api-reference/java/com/google/adk/agents/LlmAgent.Builder.html\#instruction(com.google.adk.agents.Instruction))





@CanIgnoreReturnValue
public[LlmAgent.Builder](https://google.github.io/adk-docs/api-reference/java/com/google/adk/agents/LlmAgent.Builder.html "class in com.google.adk.agents")instruction( [Instruction](https://google.github.io/adk-docs/api-reference/java/com/google/adk/agents/Instruction.html "interface in com.google.adk.agents") instruction)

- ### instruction [![Link icon](https://google.github.io/adk-docs/api-reference/java/resource-files/link.svg)](https://google.github.io/adk-docs/api-reference/java/com/google/adk/agents/LlmAgent.Builder.html\#instruction(java.lang.String))





@CanIgnoreReturnValue
public[LlmAgent.Builder](https://google.github.io/adk-docs/api-reference/java/com/google/adk/agents/LlmAgent.Builder.html "class in com.google.adk.agents")instruction( [String](https://docs.oracle.com/en/java/javase/17/docs/api/java.base/java/lang/String.html "class or interface in java.lang") instruction)

- ### globalInstruction [![Link icon](https://google.github.io/adk-docs/api-reference/java/resource-files/link.svg)](https://google.github.io/adk-docs/api-reference/java/com/google/adk/agents/LlmAgent.Builder.html\#globalInstruction(com.google.adk.agents.Instruction))





@CanIgnoreReturnValue
public[LlmAgent.Builder](https://google.github.io/adk-docs/api-reference/java/com/google/adk/agents/LlmAgent.Builder.html "class in com.google.adk.agents")globalInstruction( [Instruction](https://google.github.io/adk-docs/api-reference/java/com/google/adk/agents/Instruction.html "interface in com.google.adk.agents") globalInstruction)

- ### globalInstruction [![Link icon](https://google.github.io/adk-docs/api-reference/java/resource-files/link.svg)](https://google.github.io/adk-docs/api-reference/java/com/google/adk/agents/LlmAgent.Builder.html\#globalInstruction(java.lang.String))





@CanIgnoreReturnValue
public[LlmAgent.Builder](https://google.github.io/adk-docs/api-reference/java/com/google/adk/agents/LlmAgent.Builder.html "class in com.google.adk.agents")globalInstruction( [String](https://docs.oracle.com/en/java/javase/17/docs/api/java.base/java/lang/String.html "class or interface in java.lang") globalInstruction)

- ### subAgents [![Link icon](https://google.github.io/adk-docs/api-reference/java/resource-files/link.svg)](https://google.github.io/adk-docs/api-reference/java/com/google/adk/agents/LlmAgent.Builder.html\#subAgents(java.util.List))





@CanIgnoreReturnValue
public[LlmAgent.Builder](https://google.github.io/adk-docs/api-reference/java/com/google/adk/agents/LlmAgent.Builder.html "class in com.google.adk.agents")subAgents( [List](https://docs.oracle.com/en/java/javase/17/docs/api/java.base/java/util/List.html "class or interface in java.util") <? extends [BaseAgent](https://google.github.io/adk-docs/api-reference/java/com/google/adk/agents/BaseAgent.html "class in com.google.adk.agents") > subAgents)

- ### subAgents [![Link icon](https://google.github.io/adk-docs/api-reference/java/resource-files/link.svg)](https://google.github.io/adk-docs/api-reference/java/com/google/adk/agents/LlmAgent.Builder.html\#subAgents(com.google.adk.agents.BaseAgent...))





@CanIgnoreReturnValue
public[LlmAgent.Builder](https://google.github.io/adk-docs/api-reference/java/com/google/adk/agents/LlmAgent.Builder.html "class in com.google.adk.agents")subAgents( [BaseAgent](https://google.github.io/adk-docs/api-reference/java/com/google/adk/agents/BaseAgent.html "class in com.google.adk.agents")... subAgents)

- ### tools [![Link icon](https://google.github.io/adk-docs/api-reference/java/resource-files/link.svg)](https://google.github.io/adk-docs/api-reference/java/com/google/adk/agents/LlmAgent.Builder.html\#tools(java.util.List))





@CanIgnoreReturnValue
public[LlmAgent.Builder](https://google.github.io/adk-docs/api-reference/java/com/google/adk/agents/LlmAgent.Builder.html "class in com.google.adk.agents")tools( [List](https://docs.oracle.com/en/java/javase/17/docs/api/java.base/java/util/List.html "class or interface in java.util") <?> tools)

- ### tools [![Link icon](https://google.github.io/adk-docs/api-reference/java/resource-files/link.svg)](https://google.github.io/adk-docs/api-reference/java/com/google/adk/agents/LlmAgent.Builder.html\#tools(java.lang.Object...))





@CanIgnoreReturnValue
public[LlmAgent.Builder](https://google.github.io/adk-docs/api-reference/java/com/google/adk/agents/LlmAgent.Builder.html "class in com.google.adk.agents")tools( [Object](https://docs.oracle.com/en/java/javase/17/docs/api/java.base/java/lang/Object.html "class or interface in java.lang")... tools)

- ### generateContentConfig [![Link icon](https://google.github.io/adk-docs/api-reference/java/resource-files/link.svg)](https://google.github.io/adk-docs/api-reference/java/com/google/adk/agents/LlmAgent.Builder.html\#generateContentConfig(com.google.genai.types.GenerateContentConfig))





@CanIgnoreReturnValue
public[LlmAgent.Builder](https://google.github.io/adk-docs/api-reference/java/com/google/adk/agents/LlmAgent.Builder.html "class in com.google.adk.agents")generateContentConfig(com.google.genai.types.GenerateContentConfig generateContentConfig)

- ### exampleProvider [![Link icon](https://google.github.io/adk-docs/api-reference/java/resource-files/link.svg)](https://google.github.io/adk-docs/api-reference/java/com/google/adk/agents/LlmAgent.Builder.html\#exampleProvider(com.google.adk.examples.BaseExampleProvider))





@CanIgnoreReturnValue
public[LlmAgent.Builder](https://google.github.io/adk-docs/api-reference/java/com/google/adk/agents/LlmAgent.Builder.html "class in com.google.adk.agents")exampleProvider( [BaseExampleProvider](https://google.github.io/adk-docs/api-reference/java/com/google/adk/examples/BaseExampleProvider.html "interface in com.google.adk.examples") exampleProvider)

- ### exampleProvider [![Link icon](https://google.github.io/adk-docs/api-reference/java/resource-files/link.svg)](https://google.github.io/adk-docs/api-reference/java/com/google/adk/agents/LlmAgent.Builder.html\#exampleProvider(java.util.List))





@CanIgnoreReturnValue
public[LlmAgent.Builder](https://google.github.io/adk-docs/api-reference/java/com/google/adk/agents/LlmAgent.Builder.html "class in com.google.adk.agents")exampleProvider( [List](https://docs.oracle.com/en/java/javase/17/docs/api/java.base/java/util/List.html "class or interface in java.util") < [Example](https://google.github.io/adk-docs/api-reference/java/com/google/adk/examples/Example.html "class in com.google.adk.examples") > examples)

- ### exampleProvider [![Link icon](https://google.github.io/adk-docs/api-reference/java/resource-files/link.svg)](https://google.github.io/adk-docs/api-reference/java/com/google/adk/agents/LlmAgent.Builder.html\#exampleProvider(com.google.adk.examples.Example...))





@CanIgnoreReturnValue
public[LlmAgent.Builder](https://google.github.io/adk-docs/api-reference/java/com/google/adk/agents/LlmAgent.Builder.html "class in com.google.adk.agents")exampleProvider( [Example](https://google.github.io/adk-docs/api-reference/java/com/google/adk/examples/Example.html "class in com.google.adk.examples")... examples)

- ### includeContents [![Link icon](https://google.github.io/adk-docs/api-reference/java/resource-files/link.svg)](https://google.github.io/adk-docs/api-reference/java/com/google/adk/agents/LlmAgent.Builder.html\#includeContents(com.google.adk.agents.LlmAgent.IncludeContents))





@CanIgnoreReturnValue
public[LlmAgent.Builder](https://google.github.io/adk-docs/api-reference/java/com/google/adk/agents/LlmAgent.Builder.html "class in com.google.adk.agents")includeContents( [LlmAgent.IncludeContents](https://google.github.io/adk-docs/api-reference/java/com/google/adk/agents/LlmAgent.IncludeContents.html "enum class in com.google.adk.agents") includeContents)

- ### planning [![Link icon](https://google.github.io/adk-docs/api-reference/java/resource-files/link.svg)](https://google.github.io/adk-docs/api-reference/java/com/google/adk/agents/LlmAgent.Builder.html\#planning(boolean))





@CanIgnoreReturnValue
public[LlmAgent.Builder](https://google.github.io/adk-docs/api-reference/java/com/google/adk/agents/LlmAgent.Builder.html "class in com.google.adk.agents")planning(boolean planning)

- ### maxSteps [![Link icon](https://google.github.io/adk-docs/api-reference/java/resource-files/link.svg)](https://google.github.io/adk-docs/api-reference/java/com/google/adk/agents/LlmAgent.Builder.html\#maxSteps(int))





@CanIgnoreReturnValue
public[LlmAgent.Builder](https://google.github.io/adk-docs/api-reference/java/com/google/adk/agents/LlmAgent.Builder.html "class in com.google.adk.agents")maxSteps(int maxSteps)

- ### disallowTransferToParent [![Link icon](https://google.github.io/adk-docs/api-reference/java/resource-files/link.svg)](https://google.github.io/adk-docs/api-reference/java/com/google/adk/agents/LlmAgent.Builder.html\#disallowTransferToParent(boolean))





@CanIgnoreReturnValue
public[LlmAgent.Builder](https://google.github.io/adk-docs/api-reference/java/com/google/adk/agents/LlmAgent.Builder.html "class in com.google.adk.agents")disallowTransferToParent(boolean disallowTransferToParent)

- ### disallowTransferToPeers [![Link icon](https://google.github.io/adk-docs/api-reference/java/resource-files/link.svg)](https://google.github.io/adk-docs/api-reference/java/com/google/adk/agents/LlmAgent.Builder.html\#disallowTransferToPeers(boolean))





@CanIgnoreReturnValue
public[LlmAgent.Builder](https://google.github.io/adk-docs/api-reference/java/com/google/adk/agents/LlmAgent.Builder.html "class in com.google.adk.agents")disallowTransferToPeers(boolean disallowTransferToPeers)

- ### beforeModelCallback [![Link icon](https://google.github.io/adk-docs/api-reference/java/resource-files/link.svg)](https://google.github.io/adk-docs/api-reference/java/com/google/adk/agents/LlmAgent.Builder.html\#beforeModelCallback(com.google.adk.agents.Callbacks.BeforeModelCallback))





@CanIgnoreReturnValue
public[LlmAgent.Builder](https://google.github.io/adk-docs/api-reference/java/com/google/adk/agents/LlmAgent.Builder.html "class in com.google.adk.agents")beforeModelCallback( [Callbacks.BeforeModelCallback](https://google.github.io/adk-docs/api-reference/java/com/google/adk/agents/Callbacks.BeforeModelCallback.html "interface in com.google.adk.agents") beforeModelCallback)

- ### beforeModelCallback [![Link icon](https://google.github.io/adk-docs/api-reference/java/resource-files/link.svg)](https://google.github.io/adk-docs/api-reference/java/com/google/adk/agents/LlmAgent.Builder.html\#beforeModelCallback(java.util.List))





@CanIgnoreReturnValue
public[LlmAgent.Builder](https://google.github.io/adk-docs/api-reference/java/com/google/adk/agents/LlmAgent.Builder.html "class in com.google.adk.agents")beforeModelCallback( [List](https://docs.oracle.com/en/java/javase/17/docs/api/java.base/java/util/List.html "class or interface in java.util") <com.google.adk.agents.Callbacks.BeforeModelCallbackBase> beforeModelCallback)

- ### beforeModelCallbackSync [![Link icon](https://google.github.io/adk-docs/api-reference/java/resource-files/link.svg)](https://google.github.io/adk-docs/api-reference/java/com/google/adk/agents/LlmAgent.Builder.html\#beforeModelCallbackSync(com.google.adk.agents.Callbacks.BeforeModelCallbackSync))





@CanIgnoreReturnValue
public[LlmAgent.Builder](https://google.github.io/adk-docs/api-reference/java/com/google/adk/agents/LlmAgent.Builder.html "class in com.google.adk.agents")beforeModelCallbackSync( [Callbacks.BeforeModelCallbackSync](https://google.github.io/adk-docs/api-reference/java/com/google/adk/agents/Callbacks.BeforeModelCallbackSync.html "interface in com.google.adk.agents") beforeModelCallbackSync)

- ### afterModelCallback [![Link icon](https://google.github.io/adk-docs/api-reference/java/resource-files/link.svg)](https://google.github.io/adk-docs/api-reference/java/com/google/adk/agents/LlmAgent.Builder.html\#afterModelCallback(com.google.adk.agents.Callbacks.AfterModelCallback))





@CanIgnoreReturnValue
public[LlmAgent.Builder](https://google.github.io/adk-docs/api-reference/java/com/google/adk/agents/LlmAgent.Builder.html "class in com.google.adk.agents")afterModelCallback( [Callbacks.AfterModelCallback](https://google.github.io/adk-docs/api-reference/java/com/google/adk/agents/Callbacks.AfterModelCallback.html "interface in com.google.adk.agents") afterModelCallback)

- ### afterModelCallback [![Link icon](https://google.github.io/adk-docs/api-reference/java/resource-files/link.svg)](https://google.github.io/adk-docs/api-reference/java/com/google/adk/agents/LlmAgent.Builder.html\#afterModelCallback(java.util.List))





@CanIgnoreReturnValue
public[LlmAgent.Builder](https://google.github.io/adk-docs/api-reference/java/com/google/adk/agents/LlmAgent.Builder.html "class in com.google.adk.agents")afterModelCallback( [List](https://docs.oracle.com/en/java/javase/17/docs/api/java.base/java/util/List.html "class or interface in java.util") <com.google.adk.agents.Callbacks.AfterModelCallbackBase> afterModelCallback)

- ### afterModelCallbackSync [![Link icon](https://google.github.io/adk-docs/api-reference/java/resource-files/link.svg)](https://google.github.io/adk-docs/api-reference/java/com/google/adk/agents/LlmAgent.Builder.html\#afterModelCallbackSync(com.google.adk.agents.Callbacks.AfterModelCallbackSync))





@CanIgnoreReturnValue
public[LlmAgent.Builder](https://google.github.io/adk-docs/api-reference/java/com/google/adk/agents/LlmAgent.Builder.html "class in com.google.adk.agents")afterModelCallbackSync( [Callbacks.AfterModelCallbackSync](https://google.github.io/adk-docs/api-reference/java/com/google/adk/agents/Callbacks.AfterModelCallbackSync.html "interface in com.google.adk.agents") afterModelCallbackSync)

- ### beforeAgentCallback [![Link icon](https://google.github.io/adk-docs/api-reference/java/resource-files/link.svg)](https://google.github.io/adk-docs/api-reference/java/com/google/adk/agents/LlmAgent.Builder.html\#beforeAgentCallback(com.google.adk.agents.Callbacks.BeforeAgentCallback))





@CanIgnoreReturnValue
public[LlmAgent.Builder](https://google.github.io/adk-docs/api-reference/java/com/google/adk/agents/LlmAgent.Builder.html "class in com.google.adk.agents")beforeAgentCallback( [Callbacks.BeforeAgentCallback](https://google.github.io/adk-docs/api-reference/java/com/google/adk/agents/Callbacks.BeforeAgentCallback.html "interface in com.google.adk.agents") beforeAgentCallback)

- ### beforeAgentCallback [![Link icon](https://google.github.io/adk-docs/api-reference/java/resource-files/link.svg)](https://google.github.io/adk-docs/api-reference/java/com/google/adk/agents/LlmAgent.Builder.html\#beforeAgentCallback(java.util.List))





@CanIgnoreReturnValue
public[LlmAgent.Builder](https://google.github.io/adk-docs/api-reference/java/com/google/adk/agents/LlmAgent.Builder.html "class in com.google.adk.agents")beforeAgentCallback( [List](https://docs.oracle.com/en/java/javase/17/docs/api/java.base/java/util/List.html "class or interface in java.util") <com.google.adk.agents.Callbacks.BeforeAgentCallbackBase> beforeAgentCallback)

- ### beforeAgentCallbackSync [![Link icon](https://google.github.io/adk-docs/api-reference/java/resource-files/link.svg)](https://google.github.io/adk-docs/api-reference/java/com/google/adk/agents/LlmAgent.Builder.html\#beforeAgentCallbackSync(com.google.adk.agents.Callbacks.BeforeAgentCallbackSync))





@CanIgnoreReturnValue
public[LlmAgent.Builder](https://google.github.io/adk-docs/api-reference/java/com/google/adk/agents/LlmAgent.Builder.html "class in com.google.adk.agents")beforeAgentCallbackSync( [Callbacks.BeforeAgentCallbackSync](https://google.github.io/adk-docs/api-reference/java/com/google/adk/agents/Callbacks.BeforeAgentCallbackSync.html "interface in com.google.adk.agents") beforeAgentCallbackSync)

- ### afterAgentCallback [![Link icon](https://google.github.io/adk-docs/api-reference/java/resource-files/link.svg)](https://google.github.io/adk-docs/api-reference/java/com/google/adk/agents/LlmAgent.Builder.html\#afterAgentCallback(com.google.adk.agents.Callbacks.AfterAgentCallback))





@CanIgnoreReturnValue
public[LlmAgent.Builder](https://google.github.io/adk-docs/api-reference/java/com/google/adk/agents/LlmAgent.Builder.html "class in com.google.adk.agents")afterAgentCallback( [Callbacks.AfterAgentCallback](https://google.github.io/adk-docs/api-reference/java/com/google/adk/agents/Callbacks.AfterAgentCallback.html "interface in com.google.adk.agents") afterAgentCallback)

- ### afterAgentCallback [![Link icon](https://google.github.io/adk-docs/api-reference/java/resource-files/link.svg)](https://google.github.io/adk-docs/api-reference/java/com/google/adk/agents/LlmAgent.Builder.html\#afterAgentCallback(java.util.List))





@CanIgnoreReturnValue
public[LlmAgent.Builder](https://google.github.io/adk-docs/api-reference/java/com/google/adk/agents/LlmAgent.Builder.html "class in com.google.adk.agents")afterAgentCallback( [List](https://docs.oracle.com/en/java/javase/17/docs/api/java.base/java/util/List.html "class or interface in java.util") <com.google.adk.agents.Callbacks.AfterAgentCallbackBase> afterAgentCallback)

- ### afterAgentCallbackSync [![Link icon](https://google.github.io/adk-docs/api-reference/java/resource-files/link.svg)](https://google.github.io/adk-docs/api-reference/java/com/google/adk/agents/LlmAgent.Builder.html\#afterAgentCallbackSync(com.google.adk.agents.Callbacks.AfterAgentCallbackSync))





@CanIgnoreReturnValue
public[LlmAgent.Builder](https://google.github.io/adk-docs/api-reference/java/com/google/adk/agents/LlmAgent.Builder.html "class in com.google.adk.agents")afterAgentCallbackSync( [Callbacks.AfterAgentCallbackSync](https://google.github.io/adk-docs/api-reference/java/com/google/adk/agents/Callbacks.AfterAgentCallbackSync.html "interface in com.google.adk.agents") afterAgentCallbackSync)

- ### beforeToolCallback [![Link icon](https://google.github.io/adk-docs/api-reference/java/resource-files/link.svg)](https://google.github.io/adk-docs/api-reference/java/com/google/adk/agents/LlmAgent.Builder.html\#beforeToolCallback(com.google.adk.agents.Callbacks.BeforeToolCallback))





@CanIgnoreReturnValue
public[LlmAgent.Builder](https://google.github.io/adk-docs/api-reference/java/com/google/adk/agents/LlmAgent.Builder.html "class in com.google.adk.agents")beforeToolCallback( [Callbacks.BeforeToolCallback](https://google.github.io/adk-docs/api-reference/java/com/google/adk/agents/Callbacks.BeforeToolCallback.html "interface in com.google.adk.agents") beforeToolCallback)

- ### beforeToolCallback [![Link icon](https://google.github.io/adk-docs/api-reference/java/resource-files/link.svg)](https://google.github.io/adk-docs/api-reference/java/com/google/adk/agents/LlmAgent.Builder.html\#beforeToolCallback(java.util.List))





@CanIgnoreReturnValue
public[LlmAgent.Builder](https://google.github.io/adk-docs/api-reference/java/com/google/adk/agents/LlmAgent.Builder.html "class in com.google.adk.agents")beforeToolCallback(@Nullable
[List](https://docs.oracle.com/en/java/javase/17/docs/api/java.base/java/util/List.html "class or interface in java.util") <com.google.adk.agents.Callbacks.BeforeToolCallbackBase> beforeToolCallbacks)

- ### beforeToolCallbackSync [![Link icon](https://google.github.io/adk-docs/api-reference/java/resource-files/link.svg)](https://google.github.io/adk-docs/api-reference/java/com/google/adk/agents/LlmAgent.Builder.html\#beforeToolCallbackSync(com.google.adk.agents.Callbacks.BeforeToolCallbackSync))





@CanIgnoreReturnValue
public[LlmAgent.Builder](https://google.github.io/adk-docs/api-reference/java/com/google/adk/agents/LlmAgent.Builder.html "class in com.google.adk.agents")beforeToolCallbackSync( [Callbacks.BeforeToolCallbackSync](https://google.github.io/adk-docs/api-reference/java/com/google/adk/agents/Callbacks.BeforeToolCallbackSync.html "interface in com.google.adk.agents") beforeToolCallbackSync)

- ### afterToolCallback [![Link icon](https://google.github.io/adk-docs/api-reference/java/resource-files/link.svg)](https://google.github.io/adk-docs/api-reference/java/com/google/adk/agents/LlmAgent.Builder.html\#afterToolCallback(com.google.adk.agents.Callbacks.AfterToolCallback))





@CanIgnoreReturnValue
public[LlmAgent.Builder](https://google.github.io/adk-docs/api-reference/java/com/google/adk/agents/LlmAgent.Builder.html "class in com.google.adk.agents")afterToolCallback( [Callbacks.AfterToolCallback](https://google.github.io/adk-docs/api-reference/java/com/google/adk/agents/Callbacks.AfterToolCallback.html "interface in com.google.adk.agents") afterToolCallback)

- ### afterToolCallback [![Link icon](https://google.github.io/adk-docs/api-reference/java/resource-files/link.svg)](https://google.github.io/adk-docs/api-reference/java/com/google/adk/agents/LlmAgent.Builder.html\#afterToolCallback(java.util.List))





@CanIgnoreReturnValue
public[LlmAgent.Builder](https://google.github.io/adk-docs/api-reference/java/com/google/adk/agents/LlmAgent.Builder.html "class in com.google.adk.agents")afterToolCallback(@Nullable
[List](https://docs.oracle.com/en/java/javase/17/docs/api/java.base/java/util/List.html "class or interface in java.util") <com.google.adk.agents.Callbacks.AfterToolCallbackBase> afterToolCallbacks)

- ### afterToolCallbackSync [![Link icon](https://google.github.io/adk-docs/api-reference/java/resource-files/link.svg)](https://google.github.io/adk-docs/api-reference/java/com/google/adk/agents/LlmAgent.Builder.html\#afterToolCallbackSync(com.google.adk.agents.Callbacks.AfterToolCallbackSync))





@CanIgnoreReturnValue
public[LlmAgent.Builder](https://google.github.io/adk-docs/api-reference/java/com/google/adk/agents/LlmAgent.Builder.html "class in com.google.adk.agents")afterToolCallbackSync( [Callbacks.AfterToolCallbackSync](https://google.github.io/adk-docs/api-reference/java/com/google/adk/agents/Callbacks.AfterToolCallbackSync.html "interface in com.google.adk.agents") afterToolCallbackSync)

- ### inputSchema [![Link icon](https://google.github.io/adk-docs/api-reference/java/resource-files/link.svg)](https://google.github.io/adk-docs/api-reference/java/com/google/adk/agents/LlmAgent.Builder.html\#inputSchema(com.google.genai.types.Schema))





@CanIgnoreReturnValue
public[LlmAgent.Builder](https://google.github.io/adk-docs/api-reference/java/com/google/adk/agents/LlmAgent.Builder.html "class in com.google.adk.agents")inputSchema(com.google.genai.types.Schema inputSchema)

- ### outputSchema [![Link icon](https://google.github.io/adk-docs/api-reference/java/resource-files/link.svg)](https://google.github.io/adk-docs/api-reference/java/com/google/adk/agents/LlmAgent.Builder.html\#outputSchema(com.google.genai.types.Schema))





@CanIgnoreReturnValue
public[LlmAgent.Builder](https://google.github.io/adk-docs/api-reference/java/com/google/adk/agents/LlmAgent.Builder.html "class in com.google.adk.agents")outputSchema(com.google.genai.types.Schema outputSchema)

- ### executor [![Link icon](https://google.github.io/adk-docs/api-reference/java/resource-files/link.svg)](https://google.github.io/adk-docs/api-reference/java/com/google/adk/agents/LlmAgent.Builder.html\#executor(java.util.concurrent.Executor))





@CanIgnoreReturnValue
public[LlmAgent.Builder](https://google.github.io/adk-docs/api-reference/java/com/google/adk/agents/LlmAgent.Builder.html "class in com.google.adk.agents")executor( [Executor](https://docs.oracle.com/en/java/javase/17/docs/api/java.base/java/util/concurrent/Executor.html "class or interface in java.util.concurrent") executor)

- ### outputKey [![Link icon](https://google.github.io/adk-docs/api-reference/java/resource-files/link.svg)](https://google.github.io/adk-docs/api-reference/java/com/google/adk/agents/LlmAgent.Builder.html\#outputKey(java.lang.String))





@CanIgnoreReturnValue
public[LlmAgent.Builder](https://google.github.io/adk-docs/api-reference/java/com/google/adk/agents/LlmAgent.Builder.html "class in com.google.adk.agents")outputKey( [String](https://docs.oracle.com/en/java/javase/17/docs/api/java.base/java/lang/String.html "class or interface in java.lang") outputKey)

- ### validate [![Link icon](https://google.github.io/adk-docs/api-reference/java/resource-files/link.svg)](https://google.github.io/adk-docs/api-reference/java/com/google/adk/agents/LlmAgent.Builder.html\#validate())





protectedvoidvalidate()

- ### build [![Link icon](https://google.github.io/adk-docs/api-reference/java/resource-files/link.svg)](https://google.github.io/adk-docs/api-reference/java/com/google/adk/agents/LlmAgent.Builder.html\#build())





public[LlmAgent](https://google.github.io/adk-docs/api-reference/java/com/google/adk/agents/LlmAgent.html "class in com.google.adk.agents")build()

## Clean Up Deployments
[Skip to content](https://google.github.io/adk-docs/deploy/agent-engine/#deploy-to-vertex-ai-agent-engine)

[Edit this page](https://github.com/google/adk-docs/edit/main/docs/deploy/agent-engine.md "Edit this page") [View source of this page](https://github.com/google/adk-docs/raw/main/docs/deploy/agent-engine.md "View source of this page")

# Deploy to Vertex AI Agent Engine [¶](https://google.github.io/adk-docs/deploy/agent-engine/\#deploy-to-vertex-ai-agent-engine "Permanent link")

Supported in ADKPython

[Agent Engine](https://cloud.google.com/vertex-ai/generative-ai/docs/agent-engine/overview)
is a fully managed Google Cloud service enabling developers to deploy, manage,
and scale AI agents in production. Agent Engine handles the infrastructure to
scale agents in production so you can focus on creating intelligent and
impactful applications. This guide provides an accelerated deployment
instruction set for when you want to deploy an ADK project quickly, and a
standard, step-by-step set of instructions for when you want to carefully manage
deploying an agent to Agent Engine.

Preview: Vertex AI in express mode

If you don't have a Google Cloud project, you can try Agent Engine without cost using
[Vertex AI in Express mode](https://cloud.google.com/vertex-ai/generative-ai/docs/start/express-mode/overview).
For details on using this feature, see the [Standard deployment](https://google.github.io/adk-docs/deploy/agent-engine/#standard-deployment) section.

## Accelerated deployment [¶](https://google.github.io/adk-docs/deploy/agent-engine/\#accelerated-deployment "Permanent link")

This section describes how to perform a deployment using the
[Agent Starter Pack](https://github.com/GoogleCloudPlatform/agent-starter-pack)
(ASP) and the ADK command line interface (CLI) tool. This approach uses the ASP
tool to apply a project template to your existing project, add deployment
artifacts, and prepare your agent project for deployment. These instructions
show you how to use ASP to provision a Google Cloud project with services needed
for deploying your ADK project, as follows:

- [Prerequisites](https://google.github.io/adk-docs/deploy/agent-engine/#prerequisites-ad): Setup Google Cloud
account, a project, and install required software.
- [Prepare your ADK project](https://google.github.io/adk-docs/deploy/agent-engine/#prepare-ad): Modify your
existing ADK project files to get ready for deployment.
- [Connect to your Google Cloud project](https://google.github.io/adk-docs/deploy/agent-engine/#connect-ad):
Connect your development environment to Google Cloud and your Google Cloud
project.
- [Deploy your ADK project](https://google.github.io/adk-docs/deploy/agent-engine/#deploy-ad): Provision
required services in your Google Cloud project and upload your ADK project code.

For information on testing a deployed agent, see [Test deployed agent](https://google.github.io/adk-docs/deploy/agent-engine/#test-deployment).
For more information on using Agent Starter Pack and its command line tools,
see the
[CLI reference](https://googlecloudplatform.github.io/agent-starter-pack/cli/enhance.html)
and
[Development guide](https://googlecloudplatform.github.io/agent-starter-pack/guide/development-guide.html).

### Prerequisites [¶](https://google.github.io/adk-docs/deploy/agent-engine/\#prerequisites-ad "Permanent link")

You need the following resources configured to use this deployment path:

- **Google Cloud account**, with administrator access to:
- **Google Cloud Project**: An empty Google Cloud project with
[billing enabled](https://cloud.google.com/billing/docs/how-to/modify-project).
For information on creating projects, see
[Creating and managing projects](https://cloud.google.com/resource-manager/docs/creating-managing-projects).
- **Python Environment**: A Python version between 3.9 and 3.13.
- **UV Tool:** Manage Python development environment and running ASP
tools. For installation details, see
[Install UV](https://docs.astral.sh/uv/getting-started/installation/).
- **Google Cloud CLI tool**: The gcloud command line interface. For
installation details, see
[Google Cloud Command Line Interface](https://cloud.google.com/sdk/docs/install).
- **Make tool**: Build automation tool. This tool is part of most
Unix-based systems, for installation details, see the
[Make tool](https://www.gnu.org/software/make/) documentation.

### Prepare your ADK project [¶](https://google.github.io/adk-docs/deploy/agent-engine/\#prepare-ad "Permanent link")

When you deploy an ADK project to Agent Engine, you need some additional files
to support the deployment operation. The following ASP command backs up your
project and then adds files to your project for deployment purposes.

These instructions assume you have an existing ADK project that you are modifying
for deployment. If you do not have an ADK project, or want to use a test
project, complete the Python
[Quickstart](https://google.github.io/adk-docs/get-started/quickstart/) guide,
which creates a
[multi\_tool\_agent](https://github.com/google/adk-docs/tree/main/examples/python/snippets/get-started/multi_tool_agent)
project. The following instructions use the `multi_tool_agent` project as an
example.

To prepare your ADK project for deployment to Agent Engine:

1. In a terminal window of your development environment, navigate to the
    **parent directory** that contains your agent folder. For example, if your
    project structure is:



```
your-project-directory/
├── multi_tool_agent/
│   ├── __init__.py
│   ├── agent.py
│   └── .env
```



Navigate to `your-project-directory/`

2. Run the ASP `enhance` command to add the needed files required for
    deployment into your project.



```
uvx agent-starter-pack enhance --adk -d agent_engine
```

3. Follow the instructions from the ASP tool. In general, you can accept
    the default answers to all questions. However for the **GCP region**,
    option, make sure you select one of the
    [supported regions for Agent Engine](https://cloud.google.com/vertex-ai/generative-ai/docs/agent-engine/overview#supported-regions).


When you successfully complete this process, the tool shows the following message:

```
> Success! Your agent project is ready.
```

Note

The ASP tool may show a reminder to connect to Google Cloud while
running, but that connection is _not required_ at this stage.

For more information about the changes ASP makes to your ADK project, see
[Changes to your ADK project](https://google.github.io/adk-docs/deploy/agent-engine/#adk-asp-changes).

### Connect to your Google Cloud project [¶](https://google.github.io/adk-docs/deploy/agent-engine/\#connect-ad "Permanent link")

Before you deploy your ADK project, you must connect to Google Cloud and your
project. After logging into your Google Cloud account, you should verify that
your deployment target project is visible from your account and that it is
configured as your current project.

To connect to Google Cloud and list your project:

1. In a terminal window of your development environment, login to your
    Google Cloud account:



```
gcloud auth application-default login
```

2. Set your target project using the Google Cloud Project ID:



```
gcloud config set project your-project-id-xxxxx
```

3. Verify your Google Cloud target project is set:



```
gcloud config get-value project
```


Once you have successfully connected to Google Cloud and set your Cloud Project
ID, you are ready to deploy your ADK project files to Agent Engine.

### Deploy your ADK project [¶](https://google.github.io/adk-docs/deploy/agent-engine/\#deploy-ad "Permanent link")

When using the ASP tool, you deploy in stages. In the first stage, you run a
`make` command that provisions the services needed to run your ADK workflow on
Agent Engine. In the second stage, your project code is uploaded to the Agent
Engine service and the agent project is executed.

Important

_Make sure your Google Cloud target deployment project is set as your_ **current**
**project** _before performing these steps_. The `make backend` command uses
your currently set Google Cloud project when it performs a deployment. For
information on setting and checking your current project, see
[Connect to your Google Cloud project](https://google.github.io/adk-docs/deploy/agent-engine/#connect-ad).

To deploy your ADK project to Agent Engine in your Google Cloud project:

1. In a terminal window, ensure you are in the parent directory (e.g.,
    `your-project-directory/`) that contains your agent folder.

2. Deploy the code from the updated local project into the Google Cloud
development environment, by running the following ASP make command:



```
make backend
```


Once this process completes successfully, you should be able to interact with
the agent running on Google Cloud Agent Engine. For details on testing the
deployed agent, see the next section.

Once this process completes successfully, you should be able to interact with
the agent running on Google Cloud Agent Engine. For details on testing the
deployed agent, see
[Test deployed agent](https://google.github.io/adk-docs/deploy/agent-engine/#test-deployment).

### Changes to your ADK project [¶](https://google.github.io/adk-docs/deploy/agent-engine/\#adk-asp-changes "Permanent link")

The ASP tools add more files to your project for deployment. The procedure
below backs up your existing project files before modifying them. This guide
uses the
[multi\_tool\_agent](https://github.com/google/adk-docs/tree/main/examples/python/snippets/get-started/multi_tool_agent)
project as a reference example. The original project has the following file
structure to start with:

```
multi_tool_agent/
├─ __init__.py
├─ agent.py
└─ .env
```

After running the ASP enhance command to add Agent Engine deployment
information, the new structure is as follows:

```
multi-tool-agent/
├─ app/                 # Core application code
│   ├─ agent.py         # Main agent logic
│   ├─ agent_engine_app.py # Agent Engine application logic
│   └─ utils/           # Utility functions and helpers
├─ .cloudbuild/         # CI/CD pipeline configurations for Google Cloud Build
├─ deployment/          # Infrastructure and deployment scripts
├─ notebooks/           # Jupyter notebooks for prototyping and evaluation
├─ tests/               # Unit, integration, and load tests
├─ Makefile             # Makefile for common commands
├─ GEMINI.md            # AI-assisted development guide
└─ pyproject.toml       # Project dependencies and configuration
```

See the README.md file in your updated ADK project folder for more information.
For more information on using Agent Starter Pack, see the
[Development guide](https://googlecloudplatform.github.io/agent-starter-pack/guide/development-guide.html).

## Standard deployment [¶](https://google.github.io/adk-docs/deploy/agent-engine/\#standard-deployment "Permanent link")

This section describes how to perform a deployment to Agent Engine step-by-step.
These instructions are more appropriate if you want to carefully manage your
deployment settings, or are modifying an existing deployment with Agent Engine.

### Prerequisites [¶](https://google.github.io/adk-docs/deploy/agent-engine/\#prerequisites "Permanent link")

These instructions assume you have already defined an ADK project and GCP project. If you do not
have an ADK project, see the instructions for creating a test project in
[Define your agent](https://google.github.io/adk-docs/deploy/agent-engine/#define-your-agent).

Preview: Vertex AI in express mode

If you do not have an exising GCP project, you can try Agent Engine without cost using
[Vertex AI in Express mode](https://cloud.google.com/vertex-ai/generative-ai/docs/start/express-mode/overview).

[Google Cloud Project](https://google.github.io/adk-docs/deploy/agent-engine/#google-cloud-project)[Vertex AI express mode](https://google.github.io/adk-docs/deploy/agent-engine/#vertex-ai-express-mode)

Before starting deployment procedure, ensure you have the following:

1. **Google Cloud Project**: A Google Cloud project with the [Vertex AI API enabled](https://console.cloud.google.com/flows/enableapi?apiid=aiplatform.googleapis.com).

2. **Authenticated gcloud CLI**: You need to be authenticated with Google Cloud. Run the following command in your terminal:




```
gcloud auth application-default login
```

3. **Google Cloud Storage (GCS) Bucket**: Agent Engine requires a GCS bucket to stage your agent's code and dependencies for deployment. If you don't have a bucket, create one by following the instructions [here](https://cloud.google.com/storage/docs/creating-buckets).

4. **Python Environment**: A Python version between 3.9 and 3.13.

5. **Install Vertex AI SDK**

Agent Engine is part of the Vertex AI SDK for Python. For more information, you can review the [Agent Engine quickstart documentation](https://cloud.google.com/vertex-ai/generative-ai/docs/agent-engine/quickstart).



```
pip install google-cloud-aiplatform[adk,agent_engines]>=1.111
```


Before starting deployment procedure, ensure you have the following:

1. **API Key from Express Mode project**: Sign up for an Express Mode project with your gmail account following the [Express mode sign up](https://cloud.google.com/vertex-ai/generative-ai/docs/start/express-mode/overview#eligibility). Get an API key from that project to use with Agent Engine!

2. **Python Environment**: A Python version between 3.9 and 3.13.

3. **Install Vertex AI SDK**

Agent Engine is part of the Vertex AI SDK for Python. For more information, you can review the [Agent Engine quickstart documentation](https://cloud.google.com/vertex-ai/generative-ai/docs/agent-engine/quickstart).



```
pip install google-cloud-aiplatform[adk,agent_engines]>=1.111
```


### Define your agent [¶](https://google.github.io/adk-docs/deploy/agent-engine/\#define-your-agent "Permanent link")

These instructions assume you have an existing ADK project that you are modifying
for deployment. If you do not have an ADK project, or want to use a test
project, complete the Python
[Quickstart](https://google.github.io/adk-docs/get-started/quickstart/) guide,
which creates a
[multi\_tool\_agent](https://github.com/google/adk-docs/tree/main/examples/python/snippets/get-started/multi_tool_agent)
project. The following instructions use the `multi_tool_agent` project as an
example.

### Initialize Vertex AI [¶](https://google.github.io/adk-docs/deploy/agent-engine/\#initialize-vertex-ai "Permanent link")

Next, initialize the Vertex AI SDK. This tells the SDK which Google Cloud project and region to use, and where to stage files for deployment.

For IDE Users

You can place this initialization code in a separate `deploy.py` script along with the deployment logic for the following steps: 3 through 6.

[Google Cloud Project](https://google.github.io/adk-docs/deploy/agent-engine/#google-cloud-project_1)[Vertex AI express mode](https://google.github.io/adk-docs/deploy/agent-engine/#vertex-ai-express-mode_1)

deploy.py

```
import vertexai
from agent import root_agent # modify this if your agent is not in agent.py

# TODO: Fill in these values for your project
PROJECT_ID = "your-gcp-project-id"
LOCATION = "us-central1"  # For other options, see https://cloud.google.com/vertex-ai/generative-ai/docs/agent-engine/overview#supported-regions
STAGING_BUCKET = "gs://your-gcs-bucket-name"

# Initialize the Vertex AI SDK
vertexai.init(
    project=PROJECT_ID,
    location=LOCATION,
    staging_bucket=STAGING_BUCKET,
)
```

deploy.py

```
import vertexai
from agent import root_agent # modify this if your agent is not in agent.py

# TODO: Fill in these values for your api key
API_KEY = "your-express-mode-api-key"

# Initialize the Vertex AI SDK
vertexai.init(
    api_key=API_KEY,
)
```

### Prepare the agent for deployment [¶](https://google.github.io/adk-docs/deploy/agent-engine/\#prepare-the-agent-for-deployment "Permanent link")

To make your agent compatible with Agent Engine, you need to wrap it in an `AdkApp` object.

deploy.py

```
from vertexai import agent_engines

# Wrap the agent in an AdkApp object
app = agent_engines.AdkApp(
    agent=root_agent,
    enable_tracing=True,
)
```

Info

When an AdkApp is deployed to Agent Engine, it automatically uses `VertexAiSessionService` for persistent, managed session state. This provides multi-turn conversational memory without any additional configuration. For local testing, the application defaults to a temporary, in-memory session service.

### Test agent locally (optional) [¶](https://google.github.io/adk-docs/deploy/agent-engine/\#test-agent-locally-optional "Permanent link")

Before deploying, you can test your agent's behavior locally.

The `async_stream_query` method returns a stream of events that represent the agent's execution trace.

deploy.py

```
# Create a local session to maintain conversation history
session = await app.async_create_session(user_id="u_123")
print(session)
```

Expected output for `create_session` (local):

```
Session(id='c6a33dae-26ef-410c-9135-b434a528291f', app_name='default-app-name', user_id='u_123', state={}, events=[], last_update_time=1743440392.8689594)
```

Send a query to the agent. Copy-paste the following code to your "deploy.py" python script or a notebook.

deploy.py

```
events = []
async for event in app.async_stream_query(
    user_id="u_123",
    session_id=session.id,
    message="whats the weather in new york",
):
    events.append(event)

# The full event stream shows the agent's thought process
print("--- Full Event Stream ---")
for event in events:
    print(event)

# For quick tests, you can extract just the final text response
final_text_responses = [\
    e for e in events\
    if e.get("content", {}).get("parts", [{}])[0].get("text")\
    and not e.get("content", {}).get("parts", [{}])[0].get("function_call")\
]
if final_text_responses:
    print("\n--- Final Response ---")
    print(final_text_responses[0]["content"]["parts"][0]["text"])
```

#### Understanding the output [¶](https://google.github.io/adk-docs/deploy/agent-engine/\#understanding-the-output "Permanent link")

When you run the code above, you will see a few types of events:

- **Tool Call Event**: The model asks to call a tool (e.g., `get_weather`).
- **Tool Response Event**: The system provides the result of the tool call back to the model.
- **Model Response Event**: The final text response from the agent after it has processed the tool results.

Expected output for `async_stream_query` (local):

```
{'parts': [{'function_call': {'id': 'af-a33fedb0-29e6-4d0c-9eb3-00c402969395', 'args': {'city': 'new york'}, 'name': 'get_weather'}}], 'role': 'model'}
{'parts': [{'function_response': {'id': 'af-a33fedb0-29e6-4d0c-9eb3-00c402969395', 'name': 'get_weather', 'response': {'status': 'success', 'report': 'The weather in New York is sunny with a temperature of 25 degrees Celsius (41 degrees Fahrenheit).'}}}], 'role': 'user'}
{'parts': [{'text': 'The weather in New York is sunny with a temperature of 25 degrees Celsius (41 degrees Fahrenheit).'}], 'role': 'model'}
```

### Deploy to agent engine [¶](https://google.github.io/adk-docs/deploy/agent-engine/\#deploy-to-agent-engine "Permanent link")

Once you are satisfied with your agent's local behavior, you can deploy it. You can do this using the Python SDK or the `adk` command-line tool.

This process packages your code, builds it into a container, and deploys it to the managed Agent Engine service. This process can take several minutes.

[ADK CLI](https://google.github.io/adk-docs/deploy/agent-engine/#adk-cli)[Python](https://google.github.io/adk-docs/deploy/agent-engine/#python)[Vertex AI express mode](https://google.github.io/adk-docs/deploy/agent-engine/#vertex-ai-express-mode_2)

You can deploy from your terminal using the `adk deploy` command line tool.
The following example deploy command uses the `multi_tool_agent` sample
code as the project to be deployed:

```
adk deploy agent_engine \
    --project=my-cloud-project-xxxxx \
    --region=us-central1 \
    --staging_bucket=gs://my-cloud-project-staging-bucket-name \
    --display_name="My Agent Name" \
    /multi_tool_agent
```

Find the names of your available storage buckets in the
[Cloud Storage Bucket](https://pantheon.corp.google.com/storage/browser)
section of your deployment project in the Google Cloud Console.
For more details on using the `adk deploy` command, see the
[ADK CLI reference](https://google.github.io/adk-docs/api-reference/cli/cli.html#adk-deploy).

Tip

Make sure your main ADK agent definition (`root_agent`) is
discoverable when deploying your ADK project.

This code block initiates the deployment from a Python script or notebook.

deploy.py

```
from vertexai import agent_engines

remote_app = agent_engines.create(
    agent_engine=app,
    requirements=[\
        "google-cloud-aiplatform[adk,agent_engines]"\
    ]
)

print(f"Deployment finished!")
print(f"Resource Name: {remote_app.resource_name}")
# Resource Name: "projects/{PROJECT_NUMBER}/locations/{LOCATION}/reasoningEngines/{RESOURCE_ID}"
#       Note: The PROJECT_NUMBER is different than the PROJECT_ID.
```

Vertex AI express mode supports both ADK CLI deployment and Python deployment.

The following example deploy command uses the `multi_tool_agent` sample
code as the project to be deployed with express mode:

```
adk deploy agent_engine \
    --display_name="My Agent Name" \
    --api_key=your-api-key-here
    /multi_tool_agent
```

Tip

Make sure your main ADK agent definition (`root_agent`) is
discoverable when deploying your ADK project.

This code block initiates the deployment from a Python script or notebook.

deploy.py

```
from vertexai import agent_engines

remote_app = agent_engines.create(
    agent_engine=app,
    requirements=[\
        "google-cloud-aiplatform[adk,agent_engines]"\
    ]
)

print(f"Deployment finished!")
print(f"Resource Name: {remote_app.resource_name}")
# Resource Name: "projects/{PROJECT_NUMBER}/locations/{LOCATION}/reasoningEngines/{RESOURCE_ID}"
#       Note: The PROJECT_NUMBER is different than the PROJECT_ID.
```

#### Monitoring and verification [¶](https://google.github.io/adk-docs/deploy/agent-engine/\#monitoring-and-verification "Permanent link")

- You can monitor the deployment status in the [Agent Engine UI](https://console.cloud.google.com/vertex-ai/agents/agent-engines) in the Google Cloud Console.
- The `remote_app.resource_name` is the unique identifier for your deployed agent. You will need it to interact with the agent. You can also get this from the response returned by the ADK CLI command.
- For additional details, you can visit the Agent Engine documentation [deploying an agent](https://cloud.google.com/vertex-ai/generative-ai/docs/agent-engine/deploy) and [managing deployed agents](https://cloud.google.com/vertex-ai/generative-ai/docs/agent-engine/manage/overview).

## Test deployed agent [¶](https://google.github.io/adk-docs/deploy/agent-engine/\#test-deployment "Permanent link")

Once you have completed the deployment of your agent to Agent Engine, you can
view your deployed agent through the Google Cloud Console, and interact
with the agent using REST calls or the Vertex AI SDK for Python.

To view your deployed agent in the Cloud Console:

- Navigate to the Agent Engine page in the Google Cloud Console:
[https://console.cloud.google.com/vertex-ai/agents/agent-engines](https://console.cloud.google.com/vertex-ai/agents/agent-engines)

This page lists all deployed agents in your currently selected Google Cloud
project. If you do not see your agent listed, make sure you have your
target project selected in Google Cloud Console. For more information on
selecting an exising Google Cloud project, see
[Creating and managing projects](https://cloud.google.com/resource-manager/docs/creating-managing-projects#identifying_projects).

### Find Google Cloud project information [¶](https://google.github.io/adk-docs/deploy/agent-engine/\#find-google-cloud-project-information "Permanent link")

You need the address and resource identification for your project (`PROJECT_ID`,
`LOCATION`, `RESOURCE_ID`) to be able to test your deployment. You can use Cloud
Console or the `gcloud` command line tool to find this information.

Vertex AI express mode API key

If you are using Vertex AI express mode, you can skip this step and use your API key.

To find your project information with Google Cloud Console:

1. In the Google Cloud Console, navigate to the Agent Engine page:
    [https://console.cloud.google.com/vertex-ai/agents/agent-engines](https://console.cloud.google.com/vertex-ai/agents/agent-engines)

2. At the top of the page, select **API URLs**, and then copy the **Query**
**URL** string for your deployed agent, which should be in this format:



```
https://$(LOCATION_ID)-aiplatform.googleapis.com/v1/projects/$(PROJECT_ID)/locations/$(LOCATION_ID)/reasoningEngines/$(RESOURCE_ID):query
```


To find your project information with `gloud`:

1. In your development environment, make sure you are authenticated to
    Google Cloud and run the following command to list your project:



```
gcloud projects list
```

2. Take the Project ID used for deployment and run this command to get
    the additional details:



```
gcloud asset search-all-resources \
       --scope=projects/$(PROJECT_ID) \
       --asset-types='aiplatform.googleapis.com/ReasoningEngine' \
       --format="table(name,assetType,location,reasoning_engine_id)"
```


### Test using REST calls [¶](https://google.github.io/adk-docs/deploy/agent-engine/\#test-using-rest-calls "Permanent link")

A simple way to interact with your deployed agent in Agent Engine is to use REST
calls with the `curl` tool. This section describes the how to check your
connection to the agent and also to test processing of a request by the deployed
agent.

#### Check connection to agent [¶](https://google.github.io/adk-docs/deploy/agent-engine/\#check-connection-to-agent "Permanent link")

You can check your connection to the running agent using the **Query URL**
available in the Agent Engine section of the Cloud Console. This check does not
execute the deployed agent, but returns information about the agent.

To send a REST call get a response from deployed agent:

- In a terminal window of your development environment, build a request
and execute it:



[Google Cloud Project](https://google.github.io/adk-docs/deploy/agent-engine/#google-cloud-project_2)[Vertex AI express mode](https://google.github.io/adk-docs/deploy/agent-engine/#vertex-ai-express-mode_3)









```
curl -X GET \
      -H "Authorization: Bearer $(gcloud auth print-access-token)" \
      "https://$(LOCATION)-aiplatform.googleapis.com/v1/projects/$(PROJECT_ID)/locations/$(LOCATION)/reasoningEngines"
```











```
curl -X GET \
      -H "x-goog-api-key:YOUR-EXPRESS-MODE-API-KEY" \
      "https://aiplatform.googleapis.com/v1/reasoningEngines"
```


If your deployment was successful, this request responds with a list of valid
requests and expected data formats.

Access for agent connections

This connection test requires the calling user has a valid access token for the
deployed agent. When testing from other environments, make sure the calling user
has access to connect to the agent in your Google Cloud project.

#### Send an agent request [¶](https://google.github.io/adk-docs/deploy/agent-engine/\#send-an-agent-request "Permanent link")

When getting responses from your agent project, you must first create a
session, receive a Session ID, and then send your requests using that Session
ID. This process is described in the following instructions.

To test interaction with the deployed agent via REST:

1. In a terminal window of your development environment, create a session
    by building a request using this template:



[Google Cloud Project](https://google.github.io/adk-docs/deploy/agent-engine/#google-cloud-project_3)[Vertex AI express mode](https://google.github.io/adk-docs/deploy/agent-engine/#vertex-ai-express-mode_4)









```
curl \
       -H "Authorization: Bearer $(gcloud auth print-access-token)" \
       -H "Content-Type: application/json" \
       https://$(LOCATION)-aiplatform.googleapis.com/v1/projects/$(PROJECT_ID)/locations/$(LOCATION)/reasoningEngines/$(RESOURCE_ID):query \
       -d '{"class_method": "async_create_session", "input": {"user_id": "u_123"},}'
```











```
curl \
       -H "x-goog-api-key:YOUR-EXPRESS-MODE-API-KEY" \
       -H "Content-Type: application/json" \
       https://aiplatform.googleapis.com/v1/reasoningEngines/$(RESOURCE_ID):query \
       -d '{"class_method": "async_create_session", "input": {"user_id": "u_123"},}'
```

2. In the response to the previous command, extract the created **Session ID**
    from the **id** field:



```
{
       "output": {
           "userId": "u_123",
           "lastUpdateTime": 1757690426.337745,
           "state": {},
           "id": "4857885913439920384", # Session ID
           "appName": "9888888855577777776",
           "events": []
       }
}
```

3. In a terminal window of your development environment, send a message to
    your agent by building a request using this template and the Session ID
    created in the previous step:



[Google Cloud Project](https://google.github.io/adk-docs/deploy/agent-engine/#google-cloud-project_4)[Vertex AI express mode](https://google.github.io/adk-docs/deploy/agent-engine/#vertex-ai-express-mode_5)









```
curl \
   -H "Authorization: Bearer $(gcloud auth print-access-token)" \
   -H "Content-Type: application/json" \
https://$(LOCATION)-aiplatform.googleapis.com/v1/projects/$(PROJECT_ID)/locations/$(LOCATION)/reasoningEngines/$(RESOURCE_ID):streamQuery?alt=sse -d '{
"class_method": "async_stream_query",
"input": {
       "user_id": "u_123",
       "session_id": "4857885913439920384",
       "message": "Hey whats the weather in new york today?",
}
}'
```











```
curl \
   -H "x-goog-api-key:YOUR-EXPRESS-MODE-API-KEY" \
   -H "Content-Type: application/json" \
https://aiplatform.googleapis.com/v1/reasoningEngines/$(RESOURCE_ID):streamQuery?alt=sse -d '{
"class_method": "async_stream_query",
"input": {
       "user_id": "u_123",
       "session_id": "4857885913439920384",
       "message": "Hey whats the weather in new york today?",
}
}'
```


This request should generate a response from your deployed agent code in JSON
format. For more information about interacting with a deployed ADK agent in
Agent Engine using REST calls, see
[Manage deployed agents](https://cloud.google.com/vertex-ai/generative-ai/docs/agent-engine/manage/overview#console)
and
[Use a Agent Development Kit agent](https://cloud.google.com/vertex-ai/generative-ai/docs/agent-engine/use/adk)
in the Agent Engine documentation.

### Test using Python [¶](https://google.github.io/adk-docs/deploy/agent-engine/\#test-using-python "Permanent link")

You can use Python code for more sophisticated and repeatable testing of your
agent deployed in Agent Engine. These instructions describe how to create
a session with the deployed agent, and then send a request to the agent for
processing.

#### Create a remote session [¶](https://google.github.io/adk-docs/deploy/agent-engine/\#create-a-remote-session "Permanent link")

Use the `remote_app` object to create a connection to deployed, remote agent:

```
# If you are in a new script or used the ADK CLI to deploy, you can connect like this:
# remote_app = agent_engines.get("your-agent-resource-name")
remote_session = await remote_app.async_create_session(user_id="u_456")
print(remote_session)
```

Expected output for `create_session` (remote):

```
{'events': [],
'user_id': 'u_456',
'state': {},
'id': '7543472750996750336',
'app_name': '7917477678498709504',
'last_update_time': 1743683353.030133}
```

The `id` value is the session ID, and `app_name` is the resource ID of the
deployed agent on Agent Engine.

#### Send queries to your remote agent [¶](https://google.github.io/adk-docs/deploy/agent-engine/\#send-queries-to-your-remote-agent "Permanent link")

```
async for event in remote_app.async_stream_query(
    user_id="u_456",
    session_id=remote_session["id"],
    message="whats the weather in new york",
):
    print(event)
```

Expected output for `async_stream_query` (remote):

```
{'parts': [{'function_call': {'id': 'af-f1906423-a531-4ecf-a1ef-723b05e85321', 'args': {'city': 'new york'}, 'name': 'get_weather'}}], 'role': 'model'}
{'parts': [{'function_response': {'id': 'af-f1906423-a531-4ecf-a1ef-723b05e85321', 'name': 'get_weather', 'response': {'status': 'success', 'report': 'The weather in New York is sunny with a temperature of 25 degrees Celsius (41 degrees Fahrenheit).'}}}], 'role': 'user'}
{'parts': [{'text': 'The weather in New York is sunny with a temperature of 25 degrees Celsius (41 degrees Fahrenheit).'}], 'role': 'model'}
```

For more information about interacting with a deployed ADK agent in
Agent Engine, see
[Manage deployed agents](https://cloud.google.com/vertex-ai/generative-ai/docs/agent-engine/manage/overview)
and
[Use a Agent Development Kit agent](https://cloud.google.com/vertex-ai/generative-ai/docs/agent-engine/use/adk)
in the Agent Engine documentation.

#### Sending Multimodal Queries [¶](https://google.github.io/adk-docs/deploy/agent-engine/\#sending-multimodal-queries "Permanent link")

To send multimodal queries (e.g., including images) to your agent, you can construct the `message` parameter of `async_stream_query` with a list of `types.Part` objects. Each part can be text or an image.

To include an image, you can use `types.Part.from_uri`, providing a Google Cloud Storage (GCS) URI for the image.

```
from google.genai import types

image_part = types.Part.from_uri(
    file_uri="gs://cloud-samples-data/generative-ai/image/scones.jpg",
    mime_type="image/jpeg",
)
text_part = types.Part.from_text(
    text="What is in this image?",
)

async for event in remote_app.async_stream_query(
    user_id="u_456",
    session_id=remote_session["id"],
    message=[text_part, image_part],
):
    print(event)
```

Note

While the underlying communication with the model may involve Base64
encoding for images, the recommended and supported method for sending image
data to an agent deployed on Agent Engine is by providing a GCS URI.

## Deployment payload [¶](https://google.github.io/adk-docs/deploy/agent-engine/\#payload "Permanent link")

When you deploy your ADK agent project to Agent Engine,
the following content is uploaded to the service:

- Your ADK agent code
- Any dependencies declared in your ADK agent code

The deployment _does not_ include the ADK API server or the ADK web user
interface libraries. The Agent Engine service provides the libraries for ADK API
server functionality.

## Clean up deployments [¶](https://google.github.io/adk-docs/deploy/agent-engine/\#clean-up-deployments "Permanent link")

If you have performed deployments as tests, it is a good practice to clean up
your cloud resources after you have finished. You can delete the deployed Agent
Engine instance to avoid any unexpected charges on your Google Cloud account.

```
remote_app.delete(force=True)
```

The `force=True` parameter also deletes any child resources that were generated
from the deployed agent, such as sessions. You can also delete your deployed
agent via the
[Agent Engine UI](https://console.cloud.google.com/vertex-ai/agents/agent-engines)
on Google Cloud.

Back to top

## Packages Using com.google.adk.sessions
Packages that use [com.google.adk.sessions](https://google.github.io/adk-docs/api-reference/java/com/google/adk/sessions/package-summary.html)

Package

Description

[com.google.adk.agents](https://google.github.io/adk-docs/api-reference/java/com/google/adk/sessions/package-use.html#com.google.adk.agents)

[com.google.adk.memory](https://google.github.io/adk-docs/api-reference/java/com/google/adk/sessions/package-use.html#com.google.adk.memory)

[com.google.adk.runner](https://google.github.io/adk-docs/api-reference/java/com/google/adk/sessions/package-use.html#com.google.adk.runner)

[com.google.adk.sessions](https://google.github.io/adk-docs/api-reference/java/com/google/adk/sessions/package-use.html#com.google.adk.sessions)

[com.google.adk.web](https://google.github.io/adk-docs/api-reference/java/com/google/adk/sessions/package-use.html#com.google.adk.web)

- Classes in [com.google.adk.sessions](https://google.github.io/adk-docs/api-reference/java/com/google/adk/sessions/package-summary.html) used by [com.google.adk.agents](https://google.github.io/adk-docs/api-reference/java/com/google/adk/agents/package-summary.html)





Class



Description



[BaseSessionService](https://google.github.io/adk-docs/api-reference/java/com/google/adk/sessions/class-use/BaseSessionService.html#com.google.adk.agents)





Defines the contract for managing [`Session`](https://google.github.io/adk-docs/api-reference/java/com/google/adk/sessions/Session.html "class in com.google.adk.sessions") s and their associated [`Event`](https://google.github.io/adk-docs/api-reference/java/com/google/adk/events/Event.html "class in com.google.adk.events") s.





[Session](https://google.github.io/adk-docs/api-reference/java/com/google/adk/sessions/class-use/Session.html#com.google.adk.agents)





A [`Session`](https://google.github.io/adk-docs/api-reference/java/com/google/adk/sessions/Session.html "class in com.google.adk.sessions") object that encapsulates the [`State`](https://google.github.io/adk-docs/api-reference/java/com/google/adk/sessions/State.html "class in com.google.adk.sessions") and [`Event`](https://google.github.io/adk-docs/api-reference/java/com/google/adk/events/Event.html "class in com.google.adk.events") s of a session.





[State](https://google.github.io/adk-docs/api-reference/java/com/google/adk/sessions/class-use/State.html#com.google.adk.agents)





A [`State`](https://google.github.io/adk-docs/api-reference/java/com/google/adk/sessions/State.html "class in com.google.adk.sessions") object that also keeps track of the changes to the state.

- Classes in [com.google.adk.sessions](https://google.github.io/adk-docs/api-reference/java/com/google/adk/sessions/package-summary.html) used by [com.google.adk.memory](https://google.github.io/adk-docs/api-reference/java/com/google/adk/memory/package-summary.html)





Class



Description



[Session](https://google.github.io/adk-docs/api-reference/java/com/google/adk/sessions/class-use/Session.html#com.google.adk.memory)





A [`Session`](https://google.github.io/adk-docs/api-reference/java/com/google/adk/sessions/Session.html "class in com.google.adk.sessions") object that encapsulates the [`State`](https://google.github.io/adk-docs/api-reference/java/com/google/adk/sessions/State.html "class in com.google.adk.sessions") and [`Event`](https://google.github.io/adk-docs/api-reference/java/com/google/adk/events/Event.html "class in com.google.adk.events") s of a session.

- Classes in [com.google.adk.sessions](https://google.github.io/adk-docs/api-reference/java/com/google/adk/sessions/package-summary.html) used by [com.google.adk.runner](https://google.github.io/adk-docs/api-reference/java/com/google/adk/runner/package-summary.html)





Class



Description



[BaseSessionService](https://google.github.io/adk-docs/api-reference/java/com/google/adk/sessions/class-use/BaseSessionService.html#com.google.adk.runner)





Defines the contract for managing [`Session`](https://google.github.io/adk-docs/api-reference/java/com/google/adk/sessions/Session.html "class in com.google.adk.sessions") s and their associated [`Event`](https://google.github.io/adk-docs/api-reference/java/com/google/adk/events/Event.html "class in com.google.adk.events") s.





[Session](https://google.github.io/adk-docs/api-reference/java/com/google/adk/sessions/class-use/Session.html#com.google.adk.runner)





A [`Session`](https://google.github.io/adk-docs/api-reference/java/com/google/adk/sessions/Session.html "class in com.google.adk.sessions") object that encapsulates the [`State`](https://google.github.io/adk-docs/api-reference/java/com/google/adk/sessions/State.html "class in com.google.adk.sessions") and [`Event`](https://google.github.io/adk-docs/api-reference/java/com/google/adk/events/Event.html "class in com.google.adk.events") s of a session.

- Classes in [com.google.adk.sessions](https://google.github.io/adk-docs/api-reference/java/com/google/adk/sessions/package-summary.html) used by [com.google.adk.sessions](https://google.github.io/adk-docs/api-reference/java/com/google/adk/sessions/package-summary.html)





Class



Description



[ApiResponse](https://google.github.io/adk-docs/api-reference/java/com/google/adk/sessions/class-use/ApiResponse.html#com.google.adk.sessions)





The API response contains a response to a call to the GenAI APIs.





[BaseSessionService](https://google.github.io/adk-docs/api-reference/java/com/google/adk/sessions/class-use/BaseSessionService.html#com.google.adk.sessions)





Defines the contract for managing [`Session`](https://google.github.io/adk-docs/api-reference/java/com/google/adk/sessions/Session.html "class in com.google.adk.sessions") s and their associated [`Event`](https://google.github.io/adk-docs/api-reference/java/com/google/adk/events/Event.html "class in com.google.adk.events") s.





[GetSessionConfig](https://google.github.io/adk-docs/api-reference/java/com/google/adk/sessions/class-use/GetSessionConfig.html#com.google.adk.sessions)





Configuration for getting a session.





[GetSessionConfig.Builder](https://google.github.io/adk-docs/api-reference/java/com/google/adk/sessions/class-use/GetSessionConfig.Builder.html#com.google.adk.sessions)





Builder for [`GetSessionConfig`](https://google.github.io/adk-docs/api-reference/java/com/google/adk/sessions/GetSessionConfig.html "class in com.google.adk.sessions").





[HttpApiClient](https://google.github.io/adk-docs/api-reference/java/com/google/adk/sessions/class-use/HttpApiClient.html#com.google.adk.sessions)





Base client for the HTTP APIs.





[ListEventsResponse](https://google.github.io/adk-docs/api-reference/java/com/google/adk/sessions/class-use/ListEventsResponse.html#com.google.adk.sessions)





Response for listing events.





[ListEventsResponse.Builder](https://google.github.io/adk-docs/api-reference/java/com/google/adk/sessions/class-use/ListEventsResponse.Builder.html#com.google.adk.sessions)





Builder for [`ListEventsResponse`](https://google.github.io/adk-docs/api-reference/java/com/google/adk/sessions/ListEventsResponse.html "class in com.google.adk.sessions").





[ListSessionsResponse](https://google.github.io/adk-docs/api-reference/java/com/google/adk/sessions/class-use/ListSessionsResponse.html#com.google.adk.sessions)





Response for listing sessions.





[ListSessionsResponse.Builder](https://google.github.io/adk-docs/api-reference/java/com/google/adk/sessions/class-use/ListSessionsResponse.Builder.html#com.google.adk.sessions)





Builder for [`ListSessionsResponse`](https://google.github.io/adk-docs/api-reference/java/com/google/adk/sessions/ListSessionsResponse.html "class in com.google.adk.sessions").





[Session](https://google.github.io/adk-docs/api-reference/java/com/google/adk/sessions/class-use/Session.html#com.google.adk.sessions)





A [`Session`](https://google.github.io/adk-docs/api-reference/java/com/google/adk/sessions/Session.html "class in com.google.adk.sessions") object that encapsulates the [`State`](https://google.github.io/adk-docs/api-reference/java/com/google/adk/sessions/State.html "class in com.google.adk.sessions") and [`Event`](https://google.github.io/adk-docs/api-reference/java/com/google/adk/events/Event.html "class in com.google.adk.events") s of a session.





[Session.Builder](https://google.github.io/adk-docs/api-reference/java/com/google/adk/sessions/class-use/Session.Builder.html#com.google.adk.sessions)





Builder for [`Session`](https://google.github.io/adk-docs/api-reference/java/com/google/adk/sessions/Session.html "class in com.google.adk.sessions").





[SessionException](https://google.github.io/adk-docs/api-reference/java/com/google/adk/sessions/class-use/SessionException.html#com.google.adk.sessions)





Represents a general error that occurred during session management operations.





[State](https://google.github.io/adk-docs/api-reference/java/com/google/adk/sessions/class-use/State.html#com.google.adk.sessions)





A [`State`](https://google.github.io/adk-docs/api-reference/java/com/google/adk/sessions/State.html "class in com.google.adk.sessions") object that also keeps track of the changes to the state.

- Classes in [com.google.adk.sessions](https://google.github.io/adk-docs/api-reference/java/com/google/adk/sessions/package-summary.html) used by [com.google.adk.web](https://google.github.io/adk-docs/api-reference/java/com/google/adk/web/package-summary.html)





Class



Description



[BaseSessionService](https://google.github.io/adk-docs/api-reference/java/com/google/adk/sessions/class-use/BaseSessionService.html#com.google.adk.web)





Defines the contract for managing [`Session`](https://google.github.io/adk-docs/api-reference/java/com/google/adk/sessions/Session.html "class in com.google.adk.sessions") s and their associated [`Event`](https://google.github.io/adk-docs/api-reference/java/com/google/adk/events/Event.html "class in com.google.adk.events") s.





[Session](https://google.github.io/adk-docs/api-reference/java/com/google/adk/sessions/class-use/Session.html#com.google.adk.web)





A [`Session`](https://google.github.io/adk-docs/api-reference/java/com/google/adk/sessions/Session.html "class in com.google.adk.sessions") object that encapsulates the [`State`](https://google.github.io/adk-docs/api-reference/java/com/google/adk/sessions/State.html "class in com.google.adk.sessions") and [`Event`](https://google.github.io/adk-docs/api-reference/java/com/google/adk/events/Event.html "class in com.google.adk.events") s of a session.

## ADK Plugins Overview
[Skip to content](https://google.github.io/adk-docs/plugins/#plugins)

[Edit this page](https://github.com/google/adk-docs/edit/main/docs/plugins/index.md "Edit this page") [View source of this page](https://github.com/google/adk-docs/raw/main/docs/plugins/index.md "View source of this page")

# Plugins [¶](https://google.github.io/adk-docs/plugins/\#plugins "Permanent link")

Supported in ADKPython v1.7.0

A Plugin in Agent Development Kit (ADK) is a custom code module that can be
executed at various stages of an agent workflow lifecycle using callback hooks.
You use Plugins for functionality that is applicable across your agent workflow.
Some typical applications of Plugins are as follows:

- **Logging and tracing**: Create detailed logs of agent, tool, and
generative AI model activity for debugging and performance analysis.
- **Policy enforcement**: Implement security guardrails, such as a
function that checks if users are authorized to use a specific tool and
prevent its execution if they do not have permission.
- **Monitoring and metrics**: Collect and export metrics on token usage,
execution times, and invocation counts to monitoring systems such as
Prometheus or
[Google Cloud Observability](https://cloud.google.com/stackdriver/docs)
(formerly Stackdriver).
- **Response caching**: Check if a request has been made before, so you
can return a cached response, skipping expensive or time consuming AI model
or tool calls.
- **Request or response modification**: Dynamically add information to AI
model prompts or standardize tool output responses.

Tip

When implementing security guardrails and policies, use ADK Plugins for
better modularity and flexibility than Callbacks. For more details, see
[Callbacks and Plugins for Security Guardrails](https://google.github.io/adk-docs/safety/#callbacks-and-plugins-for-security-guardrails).

Caution

Plugins are not supported by the
[ADK web interface](https://google.github.io/adk-docs/evaluate/#1-adk-web-run-evaluations-via-the-web-ui).
If your ADK workflow uses Plugins, you must run your workflow without the
web interface.

## How do Plugins work? [¶](https://google.github.io/adk-docs/plugins/\#how-do-plugins-work "Permanent link")

An ADK Plugin extends the `BasePlugin` class and contains one or more
`callback` methods, indicating where in the agent lifecycle the Plugin should be
executed. You integrate Plugins into an agent by registering them in your
agent's `Runner` class. For more information on how and where you can trigger
Plugins in your agent application, see
[Plugin callback hooks](https://google.github.io/adk-docs/plugins/#plugin-callback-hooks).

Plugin functionality builds on
[Callbacks](https://google.github.io/adk-docs/callbacks/), which is a key design
element of the ADK's extensible architecture. While a typical Agent Callback is
configured on a _single agent, a single tool_ for a _specific task_, a Plugin is
registered _once_ on the `Runner` and its callbacks apply _globally_ to every
agent, tool, and LLM call managed by that runner. Plugins let you package
related callback functions together to be used across a workflow. This makes
Plugins an ideal solution for implementing features that cut across your entire
agent application.

## Prebuilt Plugins [¶](https://google.github.io/adk-docs/plugins/\#prebuilt-plugins "Permanent link")

ADK includes several plugins that you can add to your agent workflows
immediately:

- [**Reflect and Retry Tools**](https://google.github.io/adk-docs/plugins/reflect-and-retry/):
Tracks tool failures and intelligently retries tool requests.
- [**BigQuery Analytics**](https://google.github.io/adk-docs/tools/google-cloud/bigquery-agent-analytics/):
Enables agent logging and analysis with BigQuery.
- [**Context Filter**](https://github.com/google/adk-python/blob/main/src/google/adk/plugins/context_filter_plugin.py):
Filters the generative AI context to reduce its size.
- [**Global Instruction**](https://github.com/google/adk-python/blob/main/src/google/adk/plugins/global_instruction_plugin.py):
Plugin that provides global instructions functionality at the App level.
- [**Save Files as Artifacts**](https://github.com/google/adk-python/blob/main/src/google/adk/plugins/save_files_as_artifacts_plugin.py):
Saves files included in user messages as Artifacts.
- [**Logging**](https://github.com/google/adk-python/blame/main/src/google/adk/plugins/logging_plugin.py):
Log important information at each agent workflow callback point.

## Define and register Plugins [¶](https://google.github.io/adk-docs/plugins/\#define-and-register-plugins "Permanent link")

This section explains how to define Plugin classes and register them as part of
your agent workflow. For a complete code example, see
[Plugin Basic](https://github.com/google/adk-python/tree/main/contributing/samples/plugin_basic)
in the repository.

### Create Plugin class [¶](https://google.github.io/adk-docs/plugins/\#create-plugin-class "Permanent link")

Start by extending the `BasePlugin` class and add one or more `callback`
methods, as shown in the following code example:

count\_plugin.py

```
from google.adk.agents.base_agent import BaseAgent
from google.adk.agents.callback_context import CallbackContext
from google.adk.models.llm_request import LlmRequest
from google.adk.plugins.base_plugin import BasePlugin

class CountInvocationPlugin(BasePlugin):
  """A custom plugin that counts agent and tool invocations."""

  def __init__(self) -> None:
    """Initialize the plugin with counters."""
    super().__init__(name="count_invocation")
    self.agent_count: int = 0
    self.tool_count: int = 0
    self.llm_request_count: int = 0

  async def before_agent_callback(
      self, *, agent: BaseAgent, callback_context: CallbackContext
  ) -> None:
    """Count agent runs."""
    self.agent_count += 1
    print(f"[Plugin] Agent run count: {self.agent_count}")

  async def before_model_callback(
      self, *, callback_context: CallbackContext, llm_request: LlmRequest
  ) -> None:
    """Count LLM requests."""
    self.llm_request_count += 1
    print(f"[Plugin] LLM request count: {self.llm_request_count}")
```

This example code implements callbacks for `before_agent_callback` and
`before_model_callback` to count execution of these tasks during the lifecycle
of the agent.

### Register Plugin class [¶](https://google.github.io/adk-docs/plugins/\#register-plugin-class "Permanent link")

Integrate your Plugin class by registering it during your agent initialization
as part of your `Runner` class, using the `plugins` parameter. You can specify
multiple Plugins with this parameter. The following code example shows how to
register the `CountInvocationPlugin` plugin defined in the previous section with
a simple ADK agent.

```
from google.adk.runners import InMemoryRunner
from google.adk import Agent
from google.adk.tools.tool_context import ToolContext
from google.genai import types
import asyncio

# Import the plugin.
from .count_plugin import CountInvocationPlugin

async def hello_world(tool_context: ToolContext, query: str):
  print(f'Hello world: query is [{query}]')

root_agent = Agent(
    model='gemini-2.0-flash',
    name='hello_world',
    description='Prints hello world with user query.',
    instruction="""Use hello_world tool to print hello world and user query.
    """,
    tools=[hello_world],
)

async def main():
  """Main entry point for the agent."""
  prompt = 'hello world'
  runner = InMemoryRunner(
      agent=root_agent,
      app_name='test_app_with_plugin',

      # Add your plugin here. You can add multiple plugins.
      plugins=[CountInvocationPlugin()],
  )

  # The rest is the same as starting a regular ADK runner.
  session = await runner.session_service.create_session(
      user_id='user',
      app_name='test_app_with_plugin',
  )

  async for event in runner.run_async(
      user_id='user',
      session_id=session.id,
      new_message=types.Content(
        role='user', parts=[types.Part.from_text(text=prompt)]
      )
  ):
    print(f'** Got event from {event.author}')

if __name__ == "__main__":
  asyncio.run(main())
```

### Run the agent with the Plugin [¶](https://google.github.io/adk-docs/plugins/\#run-the-agent-with-the-plugin "Permanent link")

Run the plugin as you typically would. The following shows how to run the
command line:

```
python3 -m path.to.main
```

Plugins are not supported by the
[ADK web interface](https://google.github.io/adk-docs/evaluate/#1-adk-web-run-evaluations-via-the-web-ui).
If your ADK workflow uses Plugins, you must run your workflow without the web
interface.

The output of this previously described agent should look similar to the
following:

```
[Plugin] Agent run count: 1
[Plugin] LLM request count: 1
** Got event from hello_world
Hello world: query is [hello world]
** Got event from hello_world
[Plugin] LLM request count: 2
** Got event from hello_world
```

For more information on running ADK agents, see the
[Quickstart](https://google.github.io/adk-docs/get-started/quickstart/#run-your-agent)
guide.

## Build workflows with Plugins [¶](https://google.github.io/adk-docs/plugins/\#build-workflows-with-plugins "Permanent link")

Plugin callback hooks are a mechanism for implementing logic that intercepts,
modifies, and even controls the agent's execution lifecycle. Each hook is a
specific method in your Plugin class that you can implement to run code at a key
moment. You have a choice between two modes of operation based on your hook's
return value:

- **To Observe:** Implement a hook with no return value (`None`). This
approach is for tasks such as logging or collecting metrics, as it allows
the agent's workflow to proceed to the next step without interruption. For
example, you could use `after_tool_callback` in a Plugin to log every
tool's result for debugging.
- **To Intervene:** Implement a hook and return a value. This approach
short-circuits the workflow. The `Runner` halts processing, skips any
subsequent plugins and the original intended action, like a Model call, and
use a Plugin callback's return value as the result. A common use case is
implementing `before_model_callback` to return a cached `LlmResponse`,
preventing a redundant and costly API call.
- **To Amend:** Implement a hook and modify the Context object. This
approach allows you to modify the context data for the module to be
executed without otherwise interrupting the execution of that module. For
example, adding additional, standardized prompt text for Model object execution.

**Caution:** Plugin callback functions have precedence over callbacks
implemented at the object level. This behavior means that Any Plugin callbacks
code is executed _before_ any Agent, Model, or Tool objects callbacks are
executed. Furthermore, if a Plugin-level agent callback returns any value, and
not an empty (`None`) response, the Agent, Model, or Tool-level callback is _not_
_executed_ (skipped).

The Plugin design establishes a hierarchy of code execution and separates
global concerns from local agent logic. A Plugin is the stateful _module_ you
build, such as `PerformanceMonitoringPlugin`, while the callback hooks are the
specific _functions_ within that module that get executed. This architecture
differs fundamentally from standard Agent Callbacks in these critical ways:

- **Scope:** Plugin hooks are _global_. You register a Plugin once on the
`Runner`, and its hooks apply universally to every Agent, Model, and Tool
it manages. In contrast, Agent Callbacks are _local_, configured
individually on a specific agent instance.
- **Execution Order:** Plugins have _precedence_. For any given event, the
Plugin hooks always run before any corresponding Agent Callback. This
system behavior makes Plugins the correct architectural choice for
implementing cross-cutting features like security policies, universal
caching, and consistent logging across your entire application.

### Agent Callbacks and Plugins [¶](https://google.github.io/adk-docs/plugins/\#agent-callbacks-and-plugins "Permanent link")

As mentioned in the previous section, there are some functional similarities
between Plugins and Agent Callbacks. The following table compares the
differences between Plugins and Agent Callbacks in more detail.

|  | **Plugins** | **Agent Callbacks** |
| --- | --- | --- |
| **Scope** | **Global**: Apply to all agents/tools/LLMs in the<br>`Runner`. | **Local**: Apply only to the specific agent instance<br>they are configured on. |
| **Primary Use Case** | **Horizontal Features**: Logging, policy, monitoring,<br>global caching. | **Specific Agent Logic**: Modifying the behavior or<br>state of a single agent. |
| **Configuration** | Configure once on the `Runner`. | Configure individually on each `BaseAgent` instance. |
| **Execution Order** | Plugin callbacks run **before** Agent Callbacks. | Agent callbacks run **after** Plugin callbacks. |

## Plugin callback hooks [¶](https://google.github.io/adk-docs/plugins/\#plugin-callback-hooks "Permanent link")

You define when a Plugin is called with the callback functions to define in
your Plugin class. Callbacks are available when a user message is received,
before and after an `Runner`, `Agent`, `Model`, or `Tool` is called, for
`Events`, and when a `Model`, or `Tool` error occurs. These callbacks include,
and take precedence over, the any callbacks defined within your Agent, Model,
and Tool classes.

The following diagram illustrates callback points where you can attach and run
Plugin functionality during your agents workflow:

![ADK Plugin callback hooks](https://google.github.io/adk-docs/assets/workflow-plugin-hooks.svg)**Figure 1.** Diagram of ADK agent workflow with Plugin callback hook
locations.

The following sections describe the available callback hooks for Plugins in
more detail.

- [User Message callbacks](https://google.github.io/adk-docs/plugins/#user-message-callbacks)
- [Runner start callbacks](https://google.github.io/adk-docs/plugins/#runner-start-callbacks)
- [Agent execution callbacks](https://google.github.io/adk-docs/plugins/#agent-execution-callbacks)
- [Model callbacks](https://google.github.io/adk-docs/plugins/#model-callbacks)
- [Tool callbacks](https://google.github.io/adk-docs/plugins/#tool-callbacks)
- [Runner end callbacks](https://google.github.io/adk-docs/plugins/#runner-end-callbacks)

### User Message callbacks [¶](https://google.github.io/adk-docs/plugins/\#user-message-callbacks "Permanent link")

_A User Message c_ allback (`on_user_message_callback`) happens when a user
sends a message. The `on_user_message_callback` is the very first hook to run,
giving you a chance to inspect or modify the initial input.\

- **When It Runs:** This callback happens immediately after
`runner.run()`, before any other processing.
- **Purpose:** The first opportunity to inspect or modify the user's raw
input.
- **Flow Control:** Returns a `types.Content` object to **replace** the
user's original message.

The following code example shows the basic syntax of this callback:

```
async def on_user_message_callback(
    self,
    *,
    invocation_context: InvocationContext,
    user_message: types.Content,
) -> Optional[types.Content]:
```

### Runner start callbacks [¶](https://google.github.io/adk-docs/plugins/\#runner-start-callbacks "Permanent link")

A _Runner start_ callback (`before_run_callback`) happens when the `Runner`
object takes the potentially modified user message and prepares for execution.
The `before_run_callback` fires here, allowing for global setup before any agent
logic begins.

- **When It Runs:** Immediately after `runner.run()` is called, before
any other processing.
- **Purpose:** The first opportunity to inspect or modify the user's raw
input.
- **Flow Control:** Return a `types.Content` object to **replace** the
user's original message.

The following code example shows the basic syntax of this callback:

```
async def before_run_callback(
    self, *, invocation_context: InvocationContext
) -> Optional[types.Content]:
```

### Agent execution callbacks [¶](https://google.github.io/adk-docs/plugins/\#agent-execution-callbacks "Permanent link")

_Agent execution_ callbacks (`before_agent`, `after_agent`) happen when a
`Runner` object invokes an agent. The `before_agent_callback` runs immediately
before the agent's main work begins. The main work encompasses the agent's
entire process for handling the request, which could involve calling models or
tools. After the agent has finished all its steps and prepared a result, the
`after_agent_callback` runs.

**Caution:** Plugins that implement these callbacks are executed _before_ the
Agent-level callbacks are executed. Furthermore, if a Plugin-level agent
callback returns anything other than a `None` or null response, the Agent-level
callback is _not executed_ (skipped).

For more information about Agent callbacks defined as part of an Agent object,
see
[Types of Callbacks](https://google.github.io/adk-docs/callbacks/types-of-callbacks/#agent-lifecycle-callbacks).

### Model callbacks [¶](https://google.github.io/adk-docs/plugins/\#model-callbacks "Permanent link")

Model callbacks **(`before_model`, `after_model`, `on_model_error`)** happen
before and after a Model object executes. The Plugins feature also supports a
callback in the event of an error, as detailed below:

- If an agent needs to call an AI model, `before_model_callback` runs first.
- If the model call is successful, `after_model_callback` runs next.
- If the model call fails with an exception, the `on_model_error_callback`
is triggered instead, allowing for graceful recovery.

**Caution:** Plugins that implement the **`before_model`** and `**after_model` _callback methods are executed before_ the Model-level callbacks are executed.
Furthermore, if a Plugin-level model callback returns anything other than a
`None` or null response, the Model-level callback is _not executed_ (skipped).

#### Model on error callback details [¶](https://google.github.io/adk-docs/plugins/\#model-on-error-callback-details "Permanent link")

The on error callback for Model objects is only supported by the Plugins
feature works as follows:

- **When It Runs:** When an exception is raised during the model call.
- **Common Use Cases:** Graceful error handling, logging the specific
error, or returning a fallback response, such as "The AI service is
currently unavailable."
- **Flow Control:**
  - Returns an `LlmResponse` object to **suppress the exception**
     and provide a fallback result.
  - Returns `None` to allow the original exception to be raised.

**Note**: If the execution of the Model object returns a `LlmResponse`, the
system resumes the execution flow, and `after_model_callback` will be triggered
normally.\*\*\*\*

The following code example shows the basic syntax of this callback:

```
async def on_model_error_callback(
    self,
    *,
    callback_context: CallbackContext,
    llm_request: LlmRequest,
    error: Exception,
) -> Optional[LlmResponse]:
```

### Tool callbacks [¶](https://google.github.io/adk-docs/plugins/\#tool-callbacks "Permanent link")

Tool callbacks **(`before_tool`, `after_tool`, `on_tool_error`)** for Plugins
happen before or after the execution of a tool, or when an error occurs. The
Plugins feature also supports a callback in the event of an error, as detailed
below:\

- When an agent executes a Tool, `before_tool_callback` runs first.
- If the tool executes successfully, `after_tool_callback` runs next.
- If the tool raises an exception, the `on_tool_error_callback` is
triggered instead, giving you a chance to handle the failure. If
`on_tool_error_callback` returns a dict, `after_tool_callback` will be
triggered normally.

**Caution:** Plugins that implement these callbacks are executed _before_ the
Tool-level callbacks are executed. Furthermore, if a Plugin-level tool callback
returns anything other than a `None` or null response, the Tool-level callback
is _not executed_ (skipped).

#### Tool on error callback details [¶](https://google.github.io/adk-docs/plugins/\#tool-on-error-callback-details "Permanent link")

The on error callback for Tool objects is only supported by the Plugins feature
works as follows:

- **When It Runs:** When an exception is raised during the execution of a
tool's `run` method.
- **Purpose:** Catching specific tool exceptions (like `APIError`),
logging the failure, and providing a user-friendly error message back to
the LLM.
- **Flow Control:** Return a `dict` to **suppress the exception**, provide
a fallback result. Return `None` to allow the original exception to be raised.

**Note**: By returning a `dict`, this resumes the execution flow, and
`after_tool_callback` will be triggered normally.

The following code example shows the basic syntax of this callback:

```
async def on_tool_error_callback(
    self,
    *,
    tool: BaseTool,
    tool_args: dict[str, Any],
    tool_context: ToolContext,
    error: Exception,
) -> Optional[dict]:
```

### Event callbacks [¶](https://google.github.io/adk-docs/plugins/\#event-callbacks "Permanent link")

An _Event callback_ (`on_event_callback`) happens when an agent produces
outputs such as a text response or a tool call result, it yields them as `Event`
objects. The `on_event_callback` fires for each event, allowing you to modify it
before it's streamed to the client.

- **When It Runs:** After an agent yields an `Event` but before it's sent
to the user. An agent's run may produce multiple events.
- **Purpose:** Useful for modifying or enriching events (e.g., adding
metadata) or for triggering side effects based on specific events.
- **Flow Control:** Return an `Event` object to **replace** the original
event.

The following code example shows the basic syntax of this callback:

```
async def on_event_callback(
    self, *, invocation_context: InvocationContext, event: Event
) -> Optional[Event]:
```

### Runner end callbacks [¶](https://google.github.io/adk-docs/plugins/\#runner-end-callbacks "Permanent link")

The _Runner end_ callback **(`after_run_callback`)** happens when the agent has
finished its entire process and all events have been handled, the `Runner`
completes its run. The `after_run_callback` is the final hook, perfect for
cleanup and final reporting.

- **When It Runs:** After the `Runner` fully completes the execution of a
request.
- **Purpose:** Ideal for global cleanup tasks, such as closing connections
or finalizing logs and metrics data.
- **Flow Control:** This callback is for teardown only and cannot alter
the final result.

The following code example shows the basic syntax of this callback:

```
async def after_run_callback(
    self, *, invocation_context: InvocationContext
) -> Optional[None]:
```

## Next steps [¶](https://google.github.io/adk-docs/plugins/\#next-steps "Permanent link")

Check out these resources for developing and applying Plugins to your ADK
projects:

- For more ADK Plugin code examples, see the
[ADK Python repository](https://github.com/google/adk-python/tree/main/src/google/adk/plugins).
- For information on applying Plugins for security purposes, see
[Callbacks and Plugins for Security Guardrails](https://google.github.io/adk-docs/safety/#callbacks-and-plugins-for-security-guardrails).

Back to top

## ADK CLI Documentation
# adk cli documentation [¶](https://google.github.io/adk-docs/api-reference/cli/\#adk-cli-documentation "Link to this heading")

Add your content using `reStructuredText` syntax. See the
[reStructuredText](https://www.sphinx-doc.org/en/master/usage/restructuredtext/index.html)
documentation for details.

Contents:

- [CLI Reference](https://google.github.io/adk-docs/api-reference/cli/cli.html)
  - [adk](https://google.github.io/adk-docs/api-reference/cli/cli.html#adk)

# [adk cli](https://google.github.io/adk-docs/api-reference/cli/\#)

### Navigation

Contents:

- [CLI Reference](https://google.github.io/adk-docs/api-reference/cli/cli.html)

### Related Topics

- [Documentation overview](https://google.github.io/adk-docs/api-reference/cli/#)
  - Next: [CLI Reference](https://google.github.io/adk-docs/api-reference/cli/cli.html "next chapter")

## ADK Agent Config
[Skip to content](https://google.github.io/adk-docs/agents/config/#build-agents-with-agent-config)

[Edit this page](https://github.com/google/adk-docs/edit/main/docs/agents/config.md "Edit this page") [View source of this page](https://github.com/google/adk-docs/raw/main/docs/agents/config.md "View source of this page")

# Build agents with Agent Config [¶](https://google.github.io/adk-docs/agents/config/\#build-agents-with-agent-config "Permanent link")

Supported in ADKPython v1.11.0Experimental

The ADK Agent Config feature lets you build an ADK workflow without writing
code. An Agent Config uses a YAML format text file with a brief description of
the agent, allowing just about anyone to assemble and run an ADK agent. The
following is a simple example of an basic Agent Config definition:

```
name: assistant_agent
model: gemini-2.5-flash
description: A helper agent that can answer users' questions.
instruction: You are an agent to help answer users' various questions.
```

You can use Agent Config files to build more complex agents which can
incorporate Functions, Tools, Sub-Agents, and more. This page describes how to
build and run ADK workflows with the Agent Config feature. For detailed
information on the syntax and settings supported by the Agent Config format,
see the
[Agent Config syntax reference](https://google.github.io/adk-docs/api-reference/agentconfig/).

Experimental

The Agent Config feature is experimental and has some
[known limitations](https://google.github.io/adk-docs/agents/config/#known-limitations). We welcome your
[feedback](https://github.com/google/adk-python/issues/new?template=feature_request.md&labels=agent%20config)!

## Get started [¶](https://google.github.io/adk-docs/agents/config/\#get-started "Permanent link")

This section describes how to set up and start building agents with the ADK and
the Agent Config feature, including installation setup, building an agent, and
running your agent.

### Setup [¶](https://google.github.io/adk-docs/agents/config/\#setup "Permanent link")

You need to install the Google Agent Development Kit libraries, and provide an
access key for a generative AI model such as Gemini API. This section provides
details on what you must install and configure before you can run agents with
the Agent Config files.

Note

The Agent Config feature currently only supports Gemini models. For more
information about additional; functional restrictions, see
[Known limitations](https://google.github.io/adk-docs/agents/config/#known-limitations).

To setup ADK for use with Agent Config:

1. Install the ADK Python libraries by following the
    [Installation](https://google.github.io/adk-docs/get-started/installation/#python)
    instructions. _Python is currently required._ For more information, see the
    [Known limitations](https://google.github.io/adk-docs/agents/config/?tab=t.0#heading=h.xefmlyt7zh0i).
2. Verify that ADK is installed by running the following command in your
    terminal:



```
adk --version
```



This command should show the ADK version you have installed.


Tip

If the `adk` command fails to run and the version is not listed in step 2, make
sure your Python environment is active. Execute `source .venv/bin/activate` in
your terminal on Mac and Linux. For other platform commands, see the
[Installation](https://google.github.io/adk-docs/get-started/installation/#python)
page.

### Build an agent [¶](https://google.github.io/adk-docs/agents/config/\#build-an-agent "Permanent link")

You build an agent with Agent Config using the `adk create` command to create
the project files for an agent, and then editing the `root_agent.yaml` file it
generates for you.

To create an ADK project for use with Agent Config:

1. In your terminal window, run the following command to create a
    config-based agent:



```
adk create --type=config my_agent
```



This command generates a `my_agent/` folder, containing a
`root_agent.yaml` file and an `.env` file.

2. In the `my_agent/.env` file, set environment variables for your agent to
    access generative AI models and other services:
3. For Gemini model access through Google API, add a line to the
       file with your API key:



      ```
      GOOGLE_GENAI_USE_VERTEXAI=0
      GOOGLE_API_KEY=<your-Google-Gemini-API-key>
      ```



      You can get an API key from the Google AI Studio
      [API Keys](https://aistudio.google.com/app/apikey) page.

4. For Gemini model access through Google Cloud, add these lines to the file:



      ```
      GOOGLE_GENAI_USE_VERTEXAI=1
      GOOGLE_CLOUD_PROJECT=<your_gcp_project>
      GOOGLE_CLOUD_LOCATION=us-central1
      ```



      For information on creating a Cloud Project, see the Google Cloud docs
      for
      [Creating and managing projects](https://cloud.google.com/resource-manager/docs/creating-managing-projects).
5. Using text editor, edit the Agent Config file
    `my_agent/root_agent.yaml`, as shown below:


```
# yaml-language-server: $schema=https://raw.githubusercontent.com/google/adk-python/refs/heads/main/src/google/adk/agents/config_schemas/AgentConfig.json
name: assistant_agent
model: gemini-2.5-flash
description: A helper agent that can answer users' questions.
instruction: You are an agent to help answer users' various questions.
```

You can discover more configuration options for your `root_agent.yaml` agent
configuration file by referring to the ADK
[samples repository](https://github.com/search?q=repo%3Agoogle%2Fadk-python+path%3A%2F%5Econtributing%5C%2Fsamples%5C%2F%2F+.yaml&type=code)
or the
[Agent Config syntax](https://google.github.io/adk-docs/api-reference/agentconfig/)
reference.

### Run the agent [¶](https://google.github.io/adk-docs/agents/config/\#run-the-agent "Permanent link")

Once you have completed editing your Agent Config, you can run your agent using
the web interface, command line terminal execution, or API server mode.

To run your Agent Config-defined agent:

1. In your terminal, navigate to the `my_agent/` directory containing the
    `root_agent.yaml` file.
2. Type one of the following commands to run your agent:
   - `adk web` \- Run web UI interface for your agent.
   - `adk run` \- Run your agent in the terminal without a user
      interface.
   - `adk api_server` \- Run your agent as a service that can be
      used by other applications.

For more information on the ways to run your agent, see the _Run Your Agent_
topic in the
[Quickstart](https://google.github.io/adk-docs/get-started/quickstart/#run-your-agent).
For more information about the ADK command line options, see the
[ADK CLI reference](https://google.github.io/adk-docs/api-reference/cli/).

## Example configs [¶](https://google.github.io/adk-docs/agents/config/\#example-configs "Permanent link")

This section shows examples of Agent Config files to get you started building
agents. For additional and more complete examples, see the ADK
[samples repository](https://github.com/search?q=repo%3Agoogle%2Fadk-python+path%3A%2F%5Econtributing%5C%2Fsamples%5C%2F%2F+root_agent.yaml&type=code).

### Built-in tool example [¶](https://google.github.io/adk-docs/agents/config/\#built-in-tool-example "Permanent link")

The following example uses a built-in ADK tool function for using google search
to provide functionality to the agent. This agent automatically uses the search
tool to reply to user requests.

```
# yaml-language-server: $schema=https://raw.githubusercontent.com/google/adk-python/refs/heads/main/src/google/adk/agents/config_schemas/AgentConfig.json
name: search_agent
model: gemini-2.0-flash
description: 'an agent whose job it is to perform Google search queries and answer questions about the results.'
instruction: You are an agent whose job is to perform Google search queries and answer questions about the results.
tools:
  - name: google_search
```

For more details, see the full code for this sample in the
[ADK sample repository](https://github.com/google/adk-python/blob/main/contributing/samples/tool_builtin_config/root_agent.yaml).

### Custom tool example [¶](https://google.github.io/adk-docs/agents/config/\#custom-tool-example "Permanent link")

The following example uses a custom tool built with Python code and listed in
the `tools:` section of the config file. The agent uses this tool to check if a
list of numbers provided by the user are prime numbers.

```
# yaml-language-server: $schema=https://raw.githubusercontent.com/google/adk-python/refs/heads/main/src/google/adk/agents/config_schemas/AgentConfig.json
agent_class: LlmAgent
model: gemini-2.5-flash
name: prime_agent
description: Handles checking if numbers are prime.
instruction: |
  You are responsible for checking whether numbers are prime.
  When asked to check primes, you must call the check_prime tool with a list of integers.
  Never attempt to determine prime numbers manually.
  Return the prime number results to the root agent.
tools:
  - name: ma_llm.check_prime
```

For more details, see the full code for this sample in the
[ADK sample repository](https://github.com/google/adk-python/blob/main/contributing/samples/multi_agent_llm_config/prime_agent.yaml).

### Sub-agents example [¶](https://google.github.io/adk-docs/agents/config/\#sub-agents-example "Permanent link")

The following example shows an agent defined with two sub-agents in the
`sub_agents:` section, and an example tool in the `tools:` section of the config
file. This agent determines what the user wants, and delegates to one of the
sub-agents to resolve the request. The sub-agents are defined using Agent Config
YAML files.

```
# yaml-language-server: $schema=https://raw.githubusercontent.com/google/adk-python/refs/heads/main/src/google/adk/agents/config_schemas/AgentConfig.json
agent_class: LlmAgent
model: gemini-2.5-flash
name: root_agent
description: Learning assistant that provides tutoring in code and math.
instruction: |
  You are a learning assistant that helps students with coding and math questions.

  You delegate coding questions to the code_tutor_agent and math questions to the math_tutor_agent.

  Follow these steps:
  1. If the user asks about programming or coding, delegate to the code_tutor_agent.
  2. If the user asks about math concepts or problems, delegate to the math_tutor_agent.
  3. Always provide clear explanations and encourage learning.
sub_agents:
  - config_path: code_tutor_agent.yaml
  - config_path: math_tutor_agent.yaml
```

For more details, see the full code for this sample in the
[ADK sample repository](https://github.com/google/adk-python/blob/main/contributing/samples/multi_agent_basic_config/root_agent.yaml).

## Deploy agent configs [¶](https://google.github.io/adk-docs/agents/config/\#deploy-agent-configs "Permanent link")

You can deploy Agent Config agents with
[Cloud Run](https://google.github.io/adk-docs/deploy/cloud-run/) and
[Agent Engine](https://google.github.io/adk-docs/deploy/agent-engine/),
using the same procedure as code-based agents. For more information on how
to prepare and deploy Agent Config-based agents, see the
[Cloud Run](https://google.github.io/adk-docs/deploy/cloud-run/) and
[Agent Engine](https://google.github.io/adk-docs/deploy/agent-engine/)
deployment guides.

## Known limitations [¶](https://google.github.io/adk-docs/agents/config/\#known-limitations "Permanent link")

The Agent Config feature is experimental and includes the following
limitations:

- **Model support:** Only Gemini models are currently supported.
Integration with third-party models is in progress.
- **Programming language:** The Agent Config feature currently supports
only Python code for tools and other functionality requiring programming code.
- **ADK Tool support:** The following ADK tools are supported by the Agent
Config feature, but _not all tools are fully supported_:
  - `google_search`
  - `load_artifacts`
  - `url_context`
  - `exit_loop`
  - `preload_memory`
  - `get_user_choice`
  - `enterprise_web_search`
  - `load_web_page`: Requires a fully-qualified path to access web
     pages.
- **Agent Type Support:** The `LangGraphAgent` and `A2aAgent` types are
not yet supported.
  - `AgentTool`
  - `LongRunningFunctionTool`
  - `VertexAiSearchTool`
  - `McpToolset`
  - `ExampleTool`

## Next steps [¶](https://google.github.io/adk-docs/agents/config/\#next-steps "Permanent link")

For ideas on how and what to build with ADK Agent Configs, see the yaml-based
agent definitions in the ADK
[adk-samples](https://github.com/search?q=repo:google/adk-python+path:/%5Econtributing%5C/samples%5C//+root_agent.yaml&type=code)
repository. For detailed information on the syntax and settings supported by
the Agent Config format, see the
[Agent Config syntax reference](https://google.github.io/adk-docs/api-reference/agentconfig/).

Back to top

## Java Audio Package Overview
Package Hierarchies:

- [All Packages](https://google.github.io/adk-docs/api-reference/java/overview-tree.html)

## Class Hierarchy

- java.lang. [Object](https://docs.oracle.com/en/java/javase/17/docs/api/java.base/java/lang/Object.html "class or interface in java.lang")
  - com.google.adk.flows.llmflows.audio. [VertexSpeechClient](https://google.github.io/adk-docs/api-reference/java/com/google/adk/flows/llmflows/audio/VertexSpeechClient.html "class in com.google.adk.flows.llmflows.audio") (implements com.google.adk.flows.llmflows.audio. [SpeechClientInterface](https://google.github.io/adk-docs/api-reference/java/com/google/adk/flows/llmflows/audio/SpeechClientInterface.html "interface in com.google.adk.flows.llmflows.audio"))

## Interface Hierarchy

- java.lang. [AutoCloseable](https://docs.oracle.com/en/java/javase/17/docs/api/java.base/java/lang/AutoCloseable.html "class or interface in java.lang")
  - com.google.adk.flows.llmflows.audio. [SpeechClientInterface](https://google.github.io/adk-docs/api-reference/java/com/google/adk/flows/llmflows/audio/SpeechClientInterface.html "interface in com.google.adk.flows.llmflows.audio")

## ADK Runtime Overview
[Skip to content](https://google.github.io/adk-docs/runtime/#runtime)

[Edit this page](https://github.com/google/adk-docs/edit/main/docs/runtime/index.md "Edit this page") [View source of this page](https://github.com/google/adk-docs/raw/main/docs/runtime/index.md "View source of this page")

# Runtime [¶](https://google.github.io/adk-docs/runtime/\#runtime "Permanent link")

Supported in ADKPython v0.1.0Go v0.1.0Java v0.1.0

The ADK Runtime is the underlying engine that powers your agent application during user interactions. It's the system that takes your defined agents, tools, and callbacks and orchestrates their execution in response to user input, managing the flow of information, state changes, and interactions with external services like LLMs or storage.

Think of the Runtime as the **"engine"** of your agentic application. You define the parts (agents, tools), and the Runtime handles how they connect and run together to fulfill a user's request.

## Core Idea: The Event Loop [¶](https://google.github.io/adk-docs/runtime/\#core-idea-the-event-loop "Permanent link")

At its heart, the ADK Runtime operates on an **Event Loop**. This loop facilitates a back-and-forth communication between the `Runner` component and your defined "Execution Logic" (which includes your Agents, the LLM calls they make, Callbacks, and Tools).

![intro_components.png](https://google.github.io/adk-docs/assets/event-loop.png)

In simple terms:

1. The `Runner` receives a user query and asks the main `Agent` to start processing.
2. The `Agent` (and its associated logic) runs until it has something to report (like a response, a request to use a tool, or a state change) – it then **yields** or **emits** an `Event`.
3. The `Runner` receives this `Event`, processes any associated actions (like saving state changes via `Services`), and forwards the event onwards (e.g., to the user interface).
4. Only _after_ the `Runner` has processed the event does the `Agent`'s logic **resume** from where it paused, now potentially seeing the effects of the changes committed by the Runner.
5. This cycle repeats until the agent has no more events to yield for the current user query.

This event-driven loop is the fundamental pattern governing how ADK executes your agent code.

## The Heartbeat: The Event Loop - Inner workings [¶](https://google.github.io/adk-docs/runtime/\#the-heartbeat-the-event-loop-inner-workings "Permanent link")

The Event Loop is the core operational pattern defining the interaction between the `Runner` and your custom code (Agents, Tools, Callbacks, collectively referred to as "Execution Logic" or "Logic Components" in the design document). It establishes a clear division of responsibilities:

Note

The specific method names and parameter names may vary slightly by SDK language (e.g., `agent_to_run.run_async(...)` in Python, `agent.Run(...)` in Go, `agent_to_run.runAsync(...)` in Java ). Refer to the language-specific API documentation for details.

### Runner's Role (Orchestrator) [¶](https://google.github.io/adk-docs/runtime/\#runners-role-orchestrator "Permanent link")

The `Runner` acts as the central coordinator for a single user invocation. Its responsibilities in the loop are:

1. **Initiation:** Receives the end user's query (`new_message`) and typically appends it to the session history via the `SessionService`.
2. **Kick-off:** Starts the event generation process by calling the main agent's execution method (e.g., `agent_to_run.run_async(...)`).
3. **Receive & Process:** Waits for the agent logic to `yield` or `emit` an `Event`. Upon receiving an event, the Runner **promptly processes** it. This involves:
   - Using configured `Services` (`SessionService`, `ArtifactService`, `MemoryService`) to commit changes indicated in `event.actions` (like `state_delta`, `artifact_delta`).
   - Performing other internal bookkeeping.
4. **Yield Upstream:** Forwards the processed event onwards (e.g., to the calling application or UI for rendering).
5. **Iterate:** Signals the agent logic that processing is complete for the yielded event, allowing it to resume and generate the _next_ event.

_Conceptual Runner Loop:_

[Python](https://google.github.io/adk-docs/runtime/#python)[Go](https://google.github.io/adk-docs/runtime/#go)[Java](https://google.github.io/adk-docs/runtime/#java)

```
# Simplified view of Runner's main loop logic
def run(new_query, ...) -> Generator[Event]:
    # 1. Append new_query to session event history (via SessionService)
    session_service.append_event(session, Event(author='user', content=new_query))

    # 2. Kick off event loop by calling the agent
    agent_event_generator = agent_to_run.run_async(context)

    async for event in agent_event_generator:
        # 3. Process the generated event and commit changes
        session_service.append_event(session, event) # Commits state/artifact deltas etc.
        # memory_service.update_memory(...) # If applicable
        # artifact_service might have already been called via context during agent run

        # 4. Yield event for upstream processing (e.g., UI rendering)
        yield event
        # Runner implicitly signals agent generator can continue after yielding
```

```
// Simplified conceptual view of the Runner's main loop logic in Go
func (r *Runner) RunConceptual(ctx context.Context, session *session.Session, newQuery *genai.Content) iter.Seq2[*Event, error] {
    return func(yield func(*Event, error) bool) {
        // 1. Append new_query to session event history (via SessionService)
        // ...
        userEvent := session.NewEvent(ctx.InvocationID()) // Simplified for conceptual view
        userEvent.Author = "user"
        userEvent.LLMResponse = model.LLMResponse{Content: newQuery}

        if _, err := r.sessionService.Append(ctx, &session.AppendRequest{Event: userEvent}); err != nil {
            yield(nil, err)
            return
        }

        // 2. Kick off event stream by calling the agent
        // Assuming agent.Run also returns iter.Seq2[*Event, error]
        agentEventsAndErrs := r.agent.Run(ctx, &agent.RunRequest{Session: session, Input: newQuery})

        for event, err := range agentEventsAndErrs {
            if err != nil {
                if !yield(event, err) { // Yield event even if there's an error, then stop
                    return
                }
                return // Agent finished with an error
            }

            // 3. Process the generated event and commit changes
            // Only commit non-partial event to a session service (as seen in actual code)
            if !event.LLMResponse.Partial {
                if _, err := r.sessionService.Append(ctx, &session.AppendRequest{Event: event}); err != nil {
                    yield(nil, err)
                    return
                }
            }
            // memory_service.update_memory(...) // If applicable
            // artifact_service might have already been called via context during agent run

            // 4. Yield event for upstream processing
            if !yield(event, nil) {
                return // Upstream consumer stopped
            }
        }
        // Agent finished successfully
    }
}
```

```
// Simplified conceptual view of the Runner's main loop logic in Java.
public Flowable<Event> runConceptual(
    Session session,
    InvocationContext invocationContext,
    Content newQuery
    ) {

    // 1. Append new_query to session event history (via SessionService)
    // ...
    sessionService.appendEvent(session, userEvent).blockingGet();

    // 2. Kick off event stream by calling the agent
    Flowable<Event> agentEventStream = agentToRun.runAsync(invocationContext);

    // 3. Process each generated event, commit changes, and "yield" or "emit"
    return agentEventStream.map(event -> {
        // This mutates the session object (adds event, applies stateDelta).
        // The return value of appendEvent (a Single<Event>) is conceptually
        // just the event itself after processing.
        sessionService.appendEvent(session, event).blockingGet(); // Simplified blocking call

        // memory_service.update_memory(...) // If applicable - conceptual
        // artifact_service might have already been called via context during agent run

        // 4. "Yield" event for upstream processing
        //    In RxJava, returning the event in map effectively yields it to the next operator or subscriber.
        return event;
    });
}
```

### Execution Logic's Role (Agent, Tool, Callback) [¶](https://google.github.io/adk-docs/runtime/\#execution-logics-role-agent-tool-callback "Permanent link")

Your code within agents, tools, and callbacks is responsible for the actual computation and decision-making. Its interaction with the loop involves:

1. **Execute:** Runs its logic based on the current `InvocationContext`, including the session state _as it was when execution resumed_.
2. **Yield:** When the logic needs to communicate (send a message, call a tool, report a state change), it constructs an `Event` containing the relevant content and actions, and then `yield`s this event back to the `Runner`.
3. **Pause:** Crucially, execution of the agent logic **pauses immediately** after the `yield` statement (or `return` in RxJava). It waits for the `Runner` to complete step 3 (processing and committing).
4. **Resume:** _Only after_ the `Runner` has processed the yielded event does the agent logic resume execution from the statement immediately following the `yield`.
5. **See Updated State:** Upon resumption, the agent logic can now reliably access the session state (`ctx.session.state`) reflecting the changes that were committed by the `Runner` from the _previously yielded_ event.

_Conceptual Execution Logic:_

[Python](https://google.github.io/adk-docs/runtime/#python_1)[Go](https://google.github.io/adk-docs/runtime/#go_1)[Java](https://google.github.io/adk-docs/runtime/#java_1)

```
# Simplified view of logic inside Agent.run_async, callbacks, or tools

# ... previous code runs based on current state ...

# 1. Determine a change or output is needed, construct the event
# Example: Updating state
update_data = {'field_1': 'value_2'}
event_with_state_change = Event(
    author=self.name,
    actions=EventActions(state_delta=update_data),
    content=types.Content(parts=[types.Part(text="State updated.")])
    # ... other event fields ...
)

# 2. Yield the event to the Runner for processing & commit
yield event_with_state_change
# <<<<<<<<<<<< EXECUTION PAUSES HERE >>>>>>>>>>>>

# <<<<<<<<<<<< RUNNER PROCESSES & COMMITS THE EVENT >>>>>>>>>>>>

# 3. Resume execution ONLY after Runner is done processing the above event.
# Now, the state committed by the Runner is reliably reflected.
# Subsequent code can safely assume the change from the yielded event happened.
val = ctx.session.state['field_1']
# here `val` is guaranteed to be "value_2" (assuming Runner committed successfully)
print(f"Resumed execution. Value of field_1 is now: {val}")

# ... subsequent code continues ...
# Maybe yield another event later...
```

```
// Simplified view of logic inside Agent.Run, callbacks, or tools

// ... previous code runs based on current state ...

// 1. Determine a change or output is needed, construct the event
// Example: Updating state
updateData := map[string]interface{}{"field_1": "value_2"}
eventWithStateChange := &Event{
    Author: self.Name(),
    Actions: &EventActions{StateDelta: updateData},
    Content: genai.NewContentFromText("State updated.", "model"),
    // ... other event fields ...
}

// 2. Yield the event to the Runner for processing & commit
// In Go, this is done by sending the event to a channel.
eventsChan <- eventWithStateChange
// <<<<<<<<<<<< EXECUTION PAUSES HERE (conceptually) >>>>>>>>>>>>
// The Runner on the other side of the channel will receive and process the event.
// The agent's goroutine might continue, but the logical flow waits for the next input or step.

// <<<<<<<<<<<< RUNNER PROCESSES & COMMITS THE EVENT >>>>>>>>>>>>

// 3. Resume execution ONLY after Runner is done processing the above event.
// In a real Go implementation, this would likely be handled by the agent receiving
// a new RunRequest or context indicating the next step. The updated state
// would be part of the session object in that new request.
// For this conceptual example, we'll just check the state.
val := ctx.State.Get("field_1")
// here `val` is guaranteed to be "value_2" because the Runner would have
// updated the session state before calling the agent again.
fmt.Printf("Resumed execution. Value of field_1 is now: %v\n", val)

// ... subsequent code continues ...
// Maybe send another event to the channel later...
```

```
// Simplified view of logic inside Agent.runAsync, callbacks, or tools
// ... previous code runs based on current state ...

// 1. Determine a change or output is needed, construct the event
// Example: Updating state
ConcurrentMap<String, Object> updateData = new ConcurrentHashMap<>();
updateData.put("field_1", "value_2");

EventActions actions = EventActions.builder().stateDelta(updateData).build();
Content eventContent = Content.builder().parts(Part.fromText("State updated.")).build();

Event eventWithStateChange = Event.builder()
    .author(self.name())
    .actions(actions)
    .content(Optional.of(eventContent))
    // ... other event fields ...
    .build();

// 2. "Yield" the event. In RxJava, this means emitting it into the stream.
//    The Runner (or upstream consumer) will subscribe to this Flowable.
//    When the Runner receives this event, it will process it (e.g., call sessionService.appendEvent).
//    The 'appendEvent' in Java ADK mutates the 'Session' object held within 'ctx' (InvocationContext).

// <<<<<<<<<<<< CONCEPTUAL PAUSE POINT >>>>>>>>>>>>
// In RxJava, the emission of 'eventWithStateChange' happens, and then the stream
// might continue with a 'flatMap' or 'concatMap' operator that represents
// the logic *after* the Runner has processed this event.

// To model the "resume execution ONLY after Runner is done processing":
// The Runner's `appendEvent` is usually an async operation itself (returns Single<Event>).
// The agent's flow needs to be structured such that subsequent logic
// that depends on the committed state runs *after* that `appendEvent` completes.

// This is how the Runner typically orchestrates it:
// Runner:
//   agent.runAsync(ctx)
//     .concatMapEager(eventFromAgent ->
//         sessionService.appendEvent(ctx.session(), eventFromAgent) // This updates ctx.session().state()
//             .toFlowable() // Emits the event after it's processed
//     )
//     .subscribe(processedEvent -> { /* UI renders processedEvent */ });

// So, within the agent's own logic, if it needs to do something *after* an event it yielded
// has been processed and its state changes are reflected in ctx.session().state(),
// that subsequent logic would typically be in another step of its reactive chain.

// For this conceptual example, we'll emit the event, and then simulate the "resume"
// as a subsequent operation in the Flowable chain.

return Flowable.just(eventWithStateChange) // Step 2: Yield the event
    .concatMap(yieldedEvent -> {
        // <<<<<<<<<<<< RUNNER CONCEPTUALLY PROCESSES & COMMITS THE EVENT >>>>>>>>>>>>
        // At this point, in a real runner, ctx.session().appendEvent(yieldedEvent) would have been called
        // by the Runner, and ctx.session().state() would be updated.
        // Since we are *inside* the agent's conceptual logic trying to model this,
        // we assume the Runner's action has implicitly updated our 'ctx.session()'.

        // 3. Resume execution.
        // Now, the state committed by the Runner (via sessionService.appendEvent)
        // is reliably reflected in ctx.session().state().
        Object val = ctx.session().state().get("field_1");
        // here `val` is guaranteed to be "value_2" because the `sessionService.appendEvent`
        // called by the Runner would have updated the session state within the `ctx` object.

        System.out.println("Resumed execution. Value of field_1 is now: " + val);

        // ... subsequent code continues ...
        // If this subsequent code needs to yield another event, it would do so here.
```

This cooperative yield/pause/resume cycle between the `Runner` and your Execution Logic, mediated by `Event` objects, forms the core of the ADK Runtime.

## Key components of the Runtime [¶](https://google.github.io/adk-docs/runtime/\#key-components-of-the-runtime "Permanent link")

Several components work together within the ADK Runtime to execute an agent invocation. Understanding their roles clarifies how the event loop functions:

1. ### `Runner` [¶](https://google.github.io/adk-docs/runtime/\#runner "Permanent link")

   - **Role:** The main entry point and orchestrator for a single user query (`run_async`).
   - **Function:** Manages the overall Event Loop, receives events yielded by the Execution Logic, coordinates with Services to process and commit event actions (state/artifact changes), and forwards processed events upstream (e.g., to the UI). It essentially drives the conversation turn by turn based on yielded events. (Defined in `google.adk.runners.runner`).
2. ### Execution Logic Components [¶](https://google.github.io/adk-docs/runtime/\#execution-logic-components "Permanent link")

   - **Role:** The parts containing your custom code and the core agent capabilities.
   - **Components:**
   - `Agent` (`BaseAgent`, `LlmAgent`, etc.): Your primary logic units that process information and decide on actions. They implement the `_run_async_impl` method which yields events.
   - `Tools` (`BaseTool`, `FunctionTool`, `AgentTool`, etc.): External functions or capabilities used by agents (often `LlmAgent`) to interact with the outside world or perform specific tasks. They execute and return results, which are then wrapped in events.
   - `Callbacks` (Functions): User-defined functions attached to agents (e.g., `before_agent_callback`, `after_model_callback`) that hook into specific points in the execution flow, potentially modifying behavior or state, whose effects are captured in events.
   - **Function:** Perform the actual thinking, calculation, or external interaction. They communicate their results or needs by **yielding `Event` objects** and pausing until the Runner processes them.
3. ### `Event` [¶](https://google.github.io/adk-docs/runtime/\#event "Permanent link")

   - **Role:** The message passed back and forth between the `Runner` and the Execution Logic.
   - **Function:** Represents an atomic occurrence (user input, agent text, tool call/result, state change request, control signal). It carries both the content of the occurrence and the intended side effects (`actions` like `state_delta`).
4. ### `Services` [¶](https://google.github.io/adk-docs/runtime/\#services "Permanent link")

   - **Role:** Backend components responsible for managing persistent or shared resources. Used primarily by the `Runner` during event processing.
   - **Components:**
   - `SessionService` (`BaseSessionService`, `InMemorySessionService`, etc.): Manages `Session` objects, including saving/loading them, applying `state_delta` to the session state, and appending events to the `event history`.
   - `ArtifactService` (`BaseArtifactService`, `InMemoryArtifactService`, `GcsArtifactService`, etc.): Manages the storage and retrieval of binary artifact data. Although `save_artifact` is called via context during execution logic, the `artifact_delta` in the event confirms the action for the Runner/SessionService.
   - `MemoryService` (`BaseMemoryService`, etc.): (Optional) Manages long-term semantic memory across sessions for a user.
   - **Function:** Provide the persistence layer. The `Runner` interacts with them to ensure changes signaled by `event.actions` are reliably stored _before_ the Execution Logic resumes.
5. ### `Session` [¶](https://google.github.io/adk-docs/runtime/\#session "Permanent link")

   - **Role:** A data container holding the state and history for _one specific conversation_ between a user and the application.
   - **Function:** Stores the current `state` dictionary, the list of all past `events` (`event history`), and references to associated artifacts. It's the primary record of the interaction, managed by the `SessionService`.
6. ### `Invocation` [¶](https://google.github.io/adk-docs/runtime/\#invocation "Permanent link")

   - **Role:** A conceptual term representing everything that happens in response to a _single_ user query, from the moment the `Runner` receives it until the agent logic finishes yielding events for that query.
   - **Function:** An invocation might involve multiple agent runs (if using agent transfer or `AgentTool`), multiple LLM calls, tool executions, and callback executions, all tied together by a single `invocation_id` within the `InvocationContext`. State variables prefixed with `temp:` are strictly scoped to a single invocation and discarded afterwards.

These players interact continuously through the Event Loop to process a user's request.

## How It Works: A Simplified Invocation [¶](https://google.github.io/adk-docs/runtime/\#how-it-works-a-simplified-invocation "Permanent link")

Let's trace a simplified flow for a typical user query that involves an LLM agent calling a tool:

![intro_components.png](https://google.github.io/adk-docs/assets/invocation-flow.png)

### Step-by-Step Breakdown [¶](https://google.github.io/adk-docs/runtime/\#step-by-step-breakdown "Permanent link")

1. **User Input:** The User sends a query (e.g., "What's the capital of France?").
2. **Runner Starts:**`Runner.run_async` begins. It interacts with the `SessionService` to load the relevant `Session` and adds the user query as the first `Event` to the session history. An `InvocationContext` (`ctx`) is prepared.
3. **Agent Execution:** The `Runner` calls `agent.run_async(ctx)` on the designated root agent (e.g., an `LlmAgent`).
4. **LLM Call (Example):** The `Agent_Llm` determines it needs information, perhaps by calling a tool. It prepares a request for the `LLM`. Let's assume the LLM decides to call `MyTool`.
5. **Yield FunctionCall Event:** The `Agent_Llm` receives the `FunctionCall` response from the LLM, wraps it in an `Event(author='Agent_Llm', content=Content(parts=[Part(function_call=...)]))`, and `yields` or `emits` this event.
6. **Agent Pauses:** The `Agent_Llm`'s execution pauses immediately after the `yield`.
7. **Runner Processes:** The `Runner` receives the FunctionCall event. It passes it to the `SessionService` to record it in the history. The `Runner` then yields the event upstream to the `User` (or application).
8. **Agent Resumes:** The `Runner` signals that the event is processed, and `Agent_Llm` resumes execution.
9. **Tool Execution:** The `Agent_Llm`'s internal flow now proceeds to execute the requested `MyTool`. It calls `tool.run_async(...)`.
10. **Tool Returns Result:**`MyTool` executes and returns its result (e.g., `{'result': 'Paris'}`).
11. **Yield FunctionResponse Event:** The agent (`Agent_Llm`) wraps the tool result into an `Event` containing a `FunctionResponse` part (e.g., `Event(author='Agent_Llm', content=Content(role='user', parts=[Part(function_response=...)]))`). This event might also contain `actions` if the tool modified state (`state_delta`) or saved artifacts (`artifact_delta`). The agent `yield`s this event.
12. **Agent Pauses:**`Agent_Llm` pauses again.
13. **Runner Processes:**`Runner` receives the FunctionResponse event. It passes it to `SessionService` which applies any `state_delta`/`artifact_delta` and adds the event to history. `Runner` yields the event upstream.
14. **Agent Resumes:**`Agent_Llm` resumes, now knowing the tool result and any state changes are committed.
15. **Final LLM Call (Example):**`Agent_Llm` sends the tool result back to the `LLM` to generate a natural language response.
16. **Yield Final Text Event:**`Agent_Llm` receives the final text from the `LLM`, wraps it in an `Event(author='Agent_Llm', content=Content(parts=[Part(text=...)]))`, and `yield`s it.
17. **Agent Pauses:**`Agent_Llm` pauses.
18. **Runner Processes:**`Runner` receives the final text event, passes it to `SessionService` for history, and yields it upstream to the `User`. This is likely marked as the `is_final_response()`.
19. **Agent Resumes & Finishes:**`Agent_Llm` resumes. Having completed its task for this invocation, its `run_async` generator finishes.
20. **Runner Completes:** The `Runner` sees the agent's generator is exhausted and finishes its loop for this invocation.

This yield/pause/process/resume cycle ensures that state changes are consistently applied and that the execution logic always operates on the most recently committed state after yielding an event.

## Important Runtime Behaviors [¶](https://google.github.io/adk-docs/runtime/\#important-runtime-behaviors "Permanent link")

Understanding a few key aspects of how the ADK Runtime handles state, streaming, and asynchronous operations is crucial for building predictable and efficient agents.

### State Updates & Commitment Timing [¶](https://google.github.io/adk-docs/runtime/\#state-updates-commitment-timing "Permanent link")

- **The Rule:** When your code (in an agent, tool, or callback) modifies the session state (e.g., `context.state['my_key'] = 'new_value'`), this change is initially recorded locally within the current `InvocationContext`. The change is only **guaranteed to be persisted** (saved by the `SessionService`) _after_ the `Event` carrying the corresponding `state_delta` in its `actions` has been `yield`-ed by your code and subsequently processed by the `Runner`.

- **Implication:** Code that runs _after_ resuming from a `yield` can reliably assume that the state changes signaled in the _yielded event_ have been committed.


[Python](https://google.github.io/adk-docs/runtime/#python_2)[Go](https://google.github.io/adk-docs/runtime/#go_2)[Java](https://google.github.io/adk-docs/runtime/#java_2)

```
# Inside agent logic (conceptual)

# 1. Modify state
ctx.session.state['status'] = 'processing'
event1 = Event(..., actions=EventActions(state_delta={'status': 'processing'}))

# 2. Yield event with the delta
yield event1
# --- PAUSE --- Runner processes event1, SessionService commits 'status' = 'processing' ---

# 3. Resume execution
# Now it's safe to rely on the committed state
current_status = ctx.session.state['status'] # Guaranteed to be 'processing'
print(f"Status after resuming: {current_status}")
```

```
  // Inside agent logic (conceptual)

func (a *Agent) RunConceptual(ctx agent.InvocationContext) iter.Seq2[*session.Event, error] {
  // The entire logic is wrapped in a function that will be returned as an iterator.
  return func(yield func(*session.Event, error) bool) {
      // ... previous code runs based on current state from the input `ctx` ...
      // e.g., val := ctx.State().Get("field_1") might return "value_1" here.

      // 1. Determine a change or output is needed, construct the event
      updateData := map[string]interface{}{"field_1": "value_2"}
      eventWithStateChange := session.NewEvent(ctx.InvocationID())
      eventWithStateChange.Author = a.Name()
      eventWithStateChange.Actions = &session.EventActions{StateDelta: updateData}
      // ... other event fields ...

      // 2. Yield the event to the Runner for processing & commit.
      // The agent's execution continues immediately after this call.
      if !yield(eventWithStateChange, nil) {
          // If yield returns false, it means the consumer (the Runner)
          // has stopped listening, so we should stop producing events.
          return
      }

      // <<<<<<<<<<<< RUNNER PROCESSES & COMMITS THE EVENT >>>>>>>>>>>>
      // This happens outside the agent, after the agent's iterator has
      // produced the event.

      // 3. The agent CANNOT immediately see the state change it just yielded.
      // The state is immutable within a single `Run` invocation.
      val := ctx.State().Get("field_1")
      // `val` here is STILL "value_1" (or whatever it was at the start).
      // The updated state ("value_2") will only be available in the `ctx`
      // of the *next* `Run` invocation in a subsequent turn.

      // ... subsequent code continues, potentially yielding more events ...
      finalEvent := session.NewEvent(ctx.InvocationID())
      finalEvent.Author = a.Name()
      // ...
      yield(finalEvent, nil)
  }
}
```

```
// Inside agent logic (conceptual)
// ... previous code runs based on current state ...

// 1. Prepare state modification and construct the event
ConcurrentHashMap<String, Object> stateChanges = new ConcurrentHashMap<>();
stateChanges.put("status", "processing");

EventActions actions = EventActions.builder().stateDelta(stateChanges).build();
Content content = Content.builder().parts(Part.fromText("Status update: processing")).build();

Event event1 = Event.builder()
    .actions(actions)
    // ...
    .build();

// 2. Yield event with the delta
return Flowable.just(event1)
    .map(
        emittedEvent -> {
            // --- CONCEPTUAL PAUSE & RUNNER PROCESSING ---
            // 3. Resume execution (conceptually)
            // Now it's safe to rely on the committed state.
            String currentStatus = (String) ctx.session().state().get("status");
            System.out.println("Status after resuming (inside agent logic): " + currentStatus); // Guaranteed to be 'processing'

            // The event itself (event1) is passed on.
            // If subsequent logic within this agent step produced *another* event,
            // you'd use concatMap to emit that new event.
            return emittedEvent;
        });

// ... subsequent agent logic might involve further reactive operators
// or emitting more events based on the now-updated `ctx.session().state()`.
```

### "Dirty Reads" of Session State [¶](https://google.github.io/adk-docs/runtime/\#dirty-reads-of-session-state "Permanent link")

- **Definition:** While commitment happens _after_ the yield, code running _later within the same invocation_, but _before_ the state-changing event is actually yielded and processed, **can often see the local, uncommitted changes**. This is sometimes called a "dirty read".
- **Example:**

[Python](https://google.github.io/adk-docs/runtime/#python_3)[Go](https://google.github.io/adk-docs/runtime/#go_3)[Java](https://google.github.io/adk-docs/runtime/#java_3)

```
# Code in before_agent_callback
callback_context.state['field_1'] = 'value_1'
# State is locally set to 'value_1', but not yet committed by Runner

# ... agent runs ...

# Code in a tool called later *within the same invocation*
# Readable (dirty read), but 'value_1' isn't guaranteed persistent yet.
val = tool_context.state['field_1'] # 'val' will likely be 'value_1' here
print(f"Dirty read value in tool: {val}")

# Assume the event carrying the state_delta={'field_1': 'value_1'}
# is yielded *after* this tool runs and is processed by the Runner.
```

```
// Code in before_agent_callback
// The callback would modify the context's session state directly.
// This change is local to the current invocation context.
ctx.State.Set("field_1", "value_1")
// State is locally set to 'value_1', but not yet committed by Runner

// ... agent runs ...

// Code in a tool called later *within the same invocation*
// Readable (dirty read), but 'value_1' isn't guaranteed persistent yet.
val := ctx.State.Get("field_1") // 'val' will likely be 'value_1' here
fmt.Printf("Dirty read value in tool: %v\n", val)

// Assume the event carrying the state_delta={'field_1': 'value_1'}
// is yielded *after* this tool runs and is processed by the Runner.
```

```
// Modify state - Code in BeforeAgentCallback
// AND stages this change in callbackContext.eventActions().stateDelta().
callbackContext.state().put("field_1", "value_1");

// --- agent runs ... ---

// --- Code in a tool called later *within the same invocation* ---
// Readable (dirty read), but 'value_1' isn't guaranteed persistent yet.
Object val = toolContext.state().get("field_1"); // 'val' will likely be 'value_1' here
System.out.println("Dirty read value in tool: " + val);
// Assume the event carrying the state_delta={'field_1': 'value_1'}
// is yielded *after* this tool runs and is processed by the Runner.
```

- **Implications:**
- **Benefit:** Allows different parts of your logic within a single complex step (e.g., multiple callbacks or tool calls before the next LLM turn) to coordinate using state without waiting for a full yield/commit cycle.
- **Caveat:** Relying heavily on dirty reads for critical logic can be risky. If the invocation fails _before_ the event carrying the `state_delta` is yielded and processed by the `Runner`, the uncommitted state change will be lost. For critical state transitions, ensure they are associated with an event that gets successfully processed.

### Streaming vs. Non-Streaming Output (`partial=True`) [¶](https://google.github.io/adk-docs/runtime/\#streaming-vs-non-streaming-output-partialtrue "Permanent link")

This primarily relates to how responses from the LLM are handled, especially when using streaming generation APIs.

- **Streaming:** The LLM generates its response token-by-token or in small chunks.
- The framework (often within `BaseLlmFlow`) yields multiple `Event` objects for a single conceptual response. Most of these events will have `partial=True`.
- The `Runner`, upon receiving an event with `partial=True`, typically **forwards it immediately** upstream (for UI display) but **skips processing its `actions`** (like `state_delta`).
- Eventually, the framework yields a final event for that response, marked as non-partial (`partial=False` or implicitly via `turn_complete=True`).
- The `Runner` **fully processes only this final event**, committing any associated `state_delta` or `artifact_delta`.
- **Non-Streaming:** The LLM generates the entire response at once. The framework yields a single event marked as non-partial, which the `Runner` processes fully.
- **Why it Matters:** Ensures that state changes are applied atomically and only once based on the _complete_ response from the LLM, while still allowing the UI to display text progressively as it's generated.

## Async is Primary (`run_async`) [¶](https://google.github.io/adk-docs/runtime/\#async-is-primary-run_async "Permanent link")

- **Core Design:** The ADK Runtime is fundamentally built on asynchronous libraries (like Python's `asyncio` and Java's `RxJava`) to handle concurrent operations (like waiting for LLM responses or tool executions) efficiently without blocking.
- **Main Entry Point:**`Runner.run_async` is the primary method for executing agent invocations. All core runnable components (Agents, specific flows) use `asynchronous` methods internally.
- **Synchronous Convenience (`run`):** A synchronous `Runner.run` method exists mainly for convenience (e.g., in simple scripts or testing environments). However, internally, `Runner.run` typically just calls `Runner.run_async` and manages the async event loop execution for you.
- **Developer Experience:** We recommend designing your applications (e.g., web servers using ADK) to be asynchronous for best performance. In Python, this means using `asyncio`; in Java, leverage `RxJava`'s reactive programming model.
- **Sync Callbacks/Tools:** The ADK framework supports both asynchronous and synchronous functions for tools and callbacks.
  - **Blocking I/O:** For long-running synchronous I/O operations, the framework attempts to prevent stalls. Python ADK may use asyncio.to\_thread, while Java ADK often relies on appropriate RxJava schedulers or wrappers for blocking calls.
  - **CPU-Bound Work:** Purely CPU-intensive synchronous tasks will still block their execution thread in both environments.

Understanding these behaviors helps you write more robust ADK applications and debug issues related to state consistency, streaming updates, and asynchronous execution.

Back to top

## MCP Async Tool
[java.lang.Object](https://docs.oracle.com/en/java/javase/17/docs/api/java.base/java/lang/Object.html "class or interface in java.lang")

[com.google.adk.tools.BaseTool](https://google.github.io/adk-docs/api-reference/java/com/google/adk/tools/BaseTool.html "class in com.google.adk.tools")

com.google.adk.tools.mcp.McpAsyncTool

* * *

public final class McpAsyncToolextends [BaseTool](https://google.github.io/adk-docs/api-reference/java/com/google/adk/tools/BaseTool.html "class in com.google.adk.tools")

Initializes a MCP tool.



This wraps a MCP Tool interface and an active MCP Session. It invokes the MCP Tool through
executing the tool from remote MCP Session.

- ## Constructor Summary [![Link icon](https://google.github.io/adk-docs/api-reference/java/resource-files/link.svg)](https://google.github.io/adk-docs/api-reference/java/com/google/adk/tools/mcp/McpAsyncTool.html\#constructor-summary)



Constructors





Constructor



Description



`McpAsyncTool(io.modelcontextprotocol.spec.McpSchema.Tool mcpTool,
io.modelcontextprotocol.client.McpAsyncClient mcpSession,
McpSessionManager mcpSessionManager)`





Creates a new McpAsyncTool with the default ObjectMapper.





`McpAsyncTool(io.modelcontextprotocol.spec.McpSchema.Tool mcpTool,
io.modelcontextprotocol.client.McpAsyncClient mcpSession,
McpSessionManager mcpSessionManager,
com.fasterxml.jackson.databind.ObjectMapper objectMapper)`





Creates a new McpAsyncTool

- ## Method Summary [![Link icon](https://google.github.io/adk-docs/api-reference/java/resource-files/link.svg)](https://google.github.io/adk-docs/api-reference/java/com/google/adk/tools/mcp/McpAsyncTool.html\#method-summary)





All MethodsInstance MethodsConcrete Methods







Modifier and Type



Method



Description



`Optional<com.google.genai.types.FunctionDeclaration>`



`declaration()`





Gets the `FunctionDeclaration` representation of this tool.





`io.modelcontextprotocol.client.McpAsyncClient`



`getMcpSession()`







`io.reactivex.rxjava3.core.Single<Map<String,Object>>`



`runAsync(Map<String,Object> args,
ToolContext toolContext)`





Calls a tool.





`com.google.genai.types.Schema`



`toGeminiSchema(io.modelcontextprotocol.spec.McpSchema.JsonSchema openApiSchema)`















### Methods inherited from class com.google.adk.tools. [BaseTool](https://google.github.io/adk-docs/api-reference/java/com/google/adk/tools/BaseTool.html "class in com.google.adk.tools") [![Link icon](https://google.github.io/adk-docs/api-reference/java/resource-files/link.svg)](https://google.github.io/adk-docs/api-reference/java/com/google/adk/tools/mcp/McpAsyncTool.html\#methods-inherited-from-class-com.google.adk.tools.BaseTool)

`description, longRunning, name, processLlmRequest`





### Methods inherited from class java.lang. [Object](https://docs.oracle.com/en/java/javase/17/docs/api/java.base/java/lang/Object.html "class or interface in java.lang") [![Link icon](https://google.github.io/adk-docs/api-reference/java/resource-files/link.svg)](https://google.github.io/adk-docs/api-reference/java/com/google/adk/tools/mcp/McpAsyncTool.html\#methods-inherited-from-class-java.lang.Object)

`clone, equals, finalize, getClass, hashCode, notify, notifyAll, toString, wait, wait, wait`


- ## Constructor Details [![Link icon](https://google.github.io/adk-docs/api-reference/java/resource-files/link.svg)](https://google.github.io/adk-docs/api-reference/java/com/google/adk/tools/mcp/McpAsyncTool.html\#constructor-detail)



- ### McpAsyncTool [![Link icon](https://google.github.io/adk-docs/api-reference/java/resource-files/link.svg)](https://google.github.io/adk-docs/api-reference/java/com/google/adk/tools/mcp/McpAsyncTool.html\#%3Cinit%3E(io.modelcontextprotocol.spec.McpSchema.Tool,io.modelcontextprotocol.client.McpAsyncClient,com.google.adk.tools.mcp.McpSessionManager))





publicMcpAsyncTool(io.modelcontextprotocol.spec.McpSchema.Tool mcpTool,
io.modelcontextprotocol.client.McpAsyncClient mcpSession,
[McpSessionManager](https://google.github.io/adk-docs/api-reference/java/com/google/adk/tools/mcp/McpSessionManager.html "class in com.google.adk.tools.mcp") mcpSessionManager)



Creates a new McpAsyncTool with the default ObjectMapper.

Parameters:`mcpTool` \- The MCP tool to wrap.`mcpSession` \- The MCP session to use to call the tool.`mcpSessionManager` \- The MCP session manager to use to create new sessions.Throws:`IllegalArgumentException` \- If mcpTool or mcpSession are null.

- ### McpAsyncTool [![Link icon](https://google.github.io/adk-docs/api-reference/java/resource-files/link.svg)](https://google.github.io/adk-docs/api-reference/java/com/google/adk/tools/mcp/McpAsyncTool.html\#%3Cinit%3E(io.modelcontextprotocol.spec.McpSchema.Tool,io.modelcontextprotocol.client.McpAsyncClient,com.google.adk.tools.mcp.McpSessionManager,com.fasterxml.jackson.databind.ObjectMapper))





publicMcpAsyncTool(io.modelcontextprotocol.spec.McpSchema.Tool mcpTool,
io.modelcontextprotocol.client.McpAsyncClient mcpSession,
[McpSessionManager](https://google.github.io/adk-docs/api-reference/java/com/google/adk/tools/mcp/McpSessionManager.html "class in com.google.adk.tools.mcp") mcpSessionManager,
com.fasterxml.jackson.databind.ObjectMapper objectMapper)



Creates a new McpAsyncTool

Parameters:`mcpTool` \- The MCP tool to wrap.`mcpSession` \- The MCP session to use to call the tool.`mcpSessionManager` \- The MCP session manager to use to create new sessions.`objectMapper` \- The ObjectMapper to use to convert JSON schemas.Throws:`IllegalArgumentException` \- If mcpTool or mcpSession are null.


- ## Method Details [![Link icon](https://google.github.io/adk-docs/api-reference/java/resource-files/link.svg)](https://google.github.io/adk-docs/api-reference/java/com/google/adk/tools/mcp/McpAsyncTool.html\#method-detail)



- ### getMcpSession [![Link icon](https://google.github.io/adk-docs/api-reference/java/resource-files/link.svg)](https://google.github.io/adk-docs/api-reference/java/com/google/adk/tools/mcp/McpAsyncTool.html\#getMcpSession())





publicio.modelcontextprotocol.client.McpAsyncClientgetMcpSession()

- ### toGeminiSchema [![Link icon](https://google.github.io/adk-docs/api-reference/java/resource-files/link.svg)](https://google.github.io/adk-docs/api-reference/java/com/google/adk/tools/mcp/McpAsyncTool.html\#toGeminiSchema(io.modelcontextprotocol.spec.McpSchema.JsonSchema))





publiccom.google.genai.types.SchematoGeminiSchema(io.modelcontextprotocol.spec.McpSchema.JsonSchema openApiSchema)

- ### declaration [![Link icon](https://google.github.io/adk-docs/api-reference/java/resource-files/link.svg)](https://google.github.io/adk-docs/api-reference/java/com/google/adk/tools/mcp/McpAsyncTool.html\#declaration())





public[Optional](https://docs.oracle.com/en/java/javase/17/docs/api/java.base/java/util/Optional.html "class or interface in java.util") <com.google.genai.types.FunctionDeclaration>declaration()



Description copied from class: `BaseTool`



Gets the `FunctionDeclaration` representation of this tool.

Overrides:`declaration` in class `BaseTool`

- ### runAsync [![Link icon](https://google.github.io/adk-docs/api-reference/java/resource-files/link.svg)](https://google.github.io/adk-docs/api-reference/java/com/google/adk/tools/mcp/McpAsyncTool.html\#runAsync(java.util.Map,com.google.adk.tools.ToolContext))





publicio.reactivex.rxjava3.core.Single< [Map](https://docs.oracle.com/en/java/javase/17/docs/api/java.base/java/util/Map.html "class or interface in java.util") < [String](https://docs.oracle.com/en/java/javase/17/docs/api/java.base/java/lang/String.html "class or interface in java.lang"), [Object](https://docs.oracle.com/en/java/javase/17/docs/api/java.base/java/lang/Object.html "class or interface in java.lang") >>runAsync( [Map](https://docs.oracle.com/en/java/javase/17/docs/api/java.base/java/util/Map.html "class or interface in java.util") < [String](https://docs.oracle.com/en/java/javase/17/docs/api/java.base/java/lang/String.html "class or interface in java.lang"), [Object](https://docs.oracle.com/en/java/javase/17/docs/api/java.base/java/lang/Object.html "class or interface in java.lang") > args,
[ToolContext](https://google.github.io/adk-docs/api-reference/java/com/google/adk/tools/ToolContext.html "class in com.google.adk.tools") toolContext)



Description copied from class: `BaseTool`



Calls a tool.

Overrides:`runAsync` in class `BaseTool`

## LlmAgent IncludeContents
[java.lang.Object](https://docs.oracle.com/en/java/javase/17/docs/api/java.base/java/lang/Object.html "class or interface in java.lang")

[java.lang.Enum](https://docs.oracle.com/en/java/javase/17/docs/api/java.base/java/lang/Enum.html "class or interface in java.lang") < [LlmAgent.IncludeContents](https://google.github.io/adk-docs/api-reference/java/com/google/adk/agents/LlmAgent.IncludeContents.html "enum class in com.google.adk.agents") >

com.google.adk.agents.LlmAgent.IncludeContents

All Implemented Interfaces:`Serializable`, `Comparable<LlmAgent.IncludeContents>`, `Constable`Enclosing class:`LlmAgent`

* * *

public static enum LlmAgent.IncludeContentsextends [Enum](https://docs.oracle.com/en/java/javase/17/docs/api/java.base/java/lang/Enum.html "class or interface in java.lang") < [LlmAgent.IncludeContents](https://google.github.io/adk-docs/api-reference/java/com/google/adk/agents/LlmAgent.IncludeContents.html "enum class in com.google.adk.agents") >

Enum to define if contents of previous events should be included in requests to the underlying
LLM.

- ## Nested Class Summary [![Link icon](https://google.github.io/adk-docs/api-reference/java/resource-files/link.svg)](https://google.github.io/adk-docs/api-reference/java/com/google/adk/agents/LlmAgent.IncludeContents.html\#nested-class-summary)





### Nested classes/interfaces inherited from class java.lang. [Enum](https://docs.oracle.com/en/java/javase/17/docs/api/java.base/java/lang/Enum.html "class or interface in java.lang") [![Link icon](https://google.github.io/adk-docs/api-reference/java/resource-files/link.svg)](https://google.github.io/adk-docs/api-reference/java/com/google/adk/agents/LlmAgent.IncludeContents.html\#nested-classes-inherited-from-class-java.lang.Enum)

`Enum.EnumDesc<E extends Enum<E>>`

- ## Enum Constant Summary [![Link icon](https://google.github.io/adk-docs/api-reference/java/resource-files/link.svg)](https://google.github.io/adk-docs/api-reference/java/com/google/adk/agents/LlmAgent.IncludeContents.html\#enum-constant-summary)



Enum Constants





Enum Constant



Description



`DEFAULT`







`NONE`

- ## Method Summary [![Link icon](https://google.github.io/adk-docs/api-reference/java/resource-files/link.svg)](https://google.github.io/adk-docs/api-reference/java/com/google/adk/agents/LlmAgent.IncludeContents.html\#method-summary)





All MethodsStatic MethodsConcrete Methods







Modifier and Type



Method



Description



`static LlmAgent.IncludeContents`



`valueOf(String name)`





Returns the enum constant of this class with the specified name.





`static LlmAgent.IncludeContents[]`



`values()`





Returns an array containing the constants of this enum class, in
the order they are declared.













### Methods inherited from class java.lang. [Enum](https://docs.oracle.com/en/java/javase/17/docs/api/java.base/java/lang/Enum.html "class or interface in java.lang") [![Link icon](https://google.github.io/adk-docs/api-reference/java/resource-files/link.svg)](https://google.github.io/adk-docs/api-reference/java/com/google/adk/agents/LlmAgent.IncludeContents.html\#methods-inherited-from-class-java.lang.Enum)

`clone, compareTo, describeConstable, equals, finalize, getDeclaringClass, hashCode, name, ordinal, toString, valueOf`





### Methods inherited from class java.lang. [Object](https://docs.oracle.com/en/java/javase/17/docs/api/java.base/java/lang/Object.html "class or interface in java.lang") [![Link icon](https://google.github.io/adk-docs/api-reference/java/resource-files/link.svg)](https://google.github.io/adk-docs/api-reference/java/com/google/adk/agents/LlmAgent.IncludeContents.html\#methods-inherited-from-class-java.lang.Object)

`getClass, notify, notifyAll, wait, wait, wait`


- ## Enum Constant Details [![Link icon](https://google.github.io/adk-docs/api-reference/java/resource-files/link.svg)](https://google.github.io/adk-docs/api-reference/java/com/google/adk/agents/LlmAgent.IncludeContents.html\#enum-constant-detail)



- ### DEFAULT [![Link icon](https://google.github.io/adk-docs/api-reference/java/resource-files/link.svg)](https://google.github.io/adk-docs/api-reference/java/com/google/adk/agents/LlmAgent.IncludeContents.html\#DEFAULT)





public static final[LlmAgent.IncludeContents](https://google.github.io/adk-docs/api-reference/java/com/google/adk/agents/LlmAgent.IncludeContents.html "enum class in com.google.adk.agents")DEFAULT

- ### NONE [![Link icon](https://google.github.io/adk-docs/api-reference/java/resource-files/link.svg)](https://google.github.io/adk-docs/api-reference/java/com/google/adk/agents/LlmAgent.IncludeContents.html\#NONE)





public static final[LlmAgent.IncludeContents](https://google.github.io/adk-docs/api-reference/java/com/google/adk/agents/LlmAgent.IncludeContents.html "enum class in com.google.adk.agents")NONE


- ## Method Details [![Link icon](https://google.github.io/adk-docs/api-reference/java/resource-files/link.svg)](https://google.github.io/adk-docs/api-reference/java/com/google/adk/agents/LlmAgent.IncludeContents.html\#method-detail)



- ### values [![Link icon](https://google.github.io/adk-docs/api-reference/java/resource-files/link.svg)](https://google.github.io/adk-docs/api-reference/java/com/google/adk/agents/LlmAgent.IncludeContents.html\#values())





public static[LlmAgent.IncludeContents](https://google.github.io/adk-docs/api-reference/java/com/google/adk/agents/LlmAgent.IncludeContents.html "enum class in com.google.adk.agents")\[\]values()



Returns an array containing the constants of this enum class, in
the order they are declared.

Returns:an array containing the constants of this enum class, in the order they are declared

- ### valueOf [![Link icon](https://google.github.io/adk-docs/api-reference/java/resource-files/link.svg)](https://google.github.io/adk-docs/api-reference/java/com/google/adk/agents/LlmAgent.IncludeContents.html\#valueOf(java.lang.String))





public static[LlmAgent.IncludeContents](https://google.github.io/adk-docs/api-reference/java/com/google/adk/agents/LlmAgent.IncludeContents.html "enum class in com.google.adk.agents")valueOf( [String](https://docs.oracle.com/en/java/javase/17/docs/api/java.base/java/lang/String.html "class or interface in java.lang") name)



Returns the enum constant of this class with the specified name.
The string must match _exactly_ an identifier used to declare an
enum constant in this class. (Extraneous whitespace characters are
not permitted.)

Parameters:`name` \- the name of the enum constant to be returned.Returns:the enum constant with the specified nameThrows:`IllegalArgumentException` \- if this enum class has no constant with the specified name`NullPointerException` \- if the argument is null

