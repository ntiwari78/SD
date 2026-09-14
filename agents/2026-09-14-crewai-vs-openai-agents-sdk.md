# CrewAI vs. the OpenAI Agents SDK

**Question asked:** What's the similarity between CrewAI and the OpenAI SDK? Compare and contrast both frameworks.

**Wiki articles drawn on:** [crewai](../wiki/crewai.md), [openai-agents-sdk](../wiki/openai-agents-sdk.md), [agent-frameworks](../wiki/agent-frameworks.md), [workflow-vs-agent](../wiki/workflow-vs-agent.md), [agents-are-systems](../wiki/agents-are-systems.md), [context-engineering](../wiki/context-engineering.md), [agent-memory](../wiki/agent-memory.md), [tool-design](../wiki/tool-design.md).

**Evidence caveat, stated up front:** no source in `raw/` compares these two frameworks directly. The CrewAI material (`cre1AIDay1.txt`, `crew1AIRestOfTheDay.txt`) never mentions the Agents SDK; the Agents SDK material (`README (1).md`) never mentions CrewAI. Both are from the same course, so the framing is consistent, but **the comparison below is synthesis, not a sourced head-to-head.** There is no benchmark, no overhead measurement, and no task suite either framework has been run against alongside the other.

---

## The short answer

They are the **same machine wearing different amounts of clothing**.

Underneath, both reduce to the same three things every framework abstracts — orchestration, tool calling, structured outputs ([agent-frameworks](../wiki/agent-frameworks.md)) — and both are emphatic that nothing magical is happening. The Agents SDK material puts it as *"framework abstractions often reduce to familiar primitives such as prompts, tool calls, and messages."* The CrewAI material puts it as *"once you know one, you pretty much know them all."*

The difference is **how much of the prompt you write yourself**. The Agents SDK hands you `instructions` and gets out of the way. CrewAI asks for role, goal, and backstory, then composes the system prompt for you. Everything else — YAML separation, the task abstraction, the process switch, unified memory — follows from that one decision.

---

## What is genuinely the same

| Dimension | Both frameworks |
|---|---|
| **Core abstraction** | agent = model + instructions + tools |
| **Tool definition** | a decorator over a Python function; schema derived from name, **type hints**, and **docstring** (`@function_tool` vs. `@tool("name")`) |
| **Structured outputs** | Pydantic `BaseModel` subclasses |
| **Orchestration modes** | both support code-driven *and* LLM-driven control flow |
| **Tracing** | built in, and in both cases the recommended way to find out what actually happened |
| **Model portability** | both model-agnostic in principle (CrewAI via LiteLLM; the SDK by design) |
| **Memory across runs** | **off by default in both** — it must be explicitly opted into |
| **What they don't give you** | neither creates intelligence by adding agents |

Two of these deserve emphasis.

**Tool definition is effectively identical.** Both derive the model-facing schema from the same three signals. This matters more than it looks: it means the guidance in [tool-design](../wiki/tool-design.md) transfers between them unchanged, and it means the docstring is the contract in both. *"Tool descriptions are part of the prompt/context provided to the model. Poor descriptions can therefore produce poor tool usage."*

**Memory is opt-in in both, for the same reason.** The model is stateless; something must resupply context ([agent-memory](../wiki/agent-memory.md)). Reusing the same `Agent` object across separate `Runner.run()` calls does **not** carry history. CrewAI's `memory=True` and the SDK's `Session` are two answers to one problem, and neither is the default.

---

## Where they differ

| Dimension | OpenAI Agents SDK | CrewAI |
|---|---|---|
| **Philosophy** | lightweight, few abstractions | opinionated, "batteries included" |
| **System prompt** | you write `instructions` directly | you write **role / goal / backstory**; the framework composes the prompt |
| **Where prompts live** | in code | in **YAML**, separated from code |
| **Unit of work** | implicit — the run | explicit **`Task`** (description + expected_output, assigned to an agent) |
| **Project setup** | none; you write the code | `crewai create crew` scaffolds a project tree |
| **Orchestration switch** | choose the mechanism in code | one line: `Process.sequential` ↔ `Process.hierarchical` |
| **Delegation mechanism** | **handoffs** (`transfer_to_<agent>`) and **agents-as-tools** | `manager_agent` + `allow_delegation=True` |
| **Context between steps** | you pass it explicitly | **implicit** — all prior task outputs unless `context` is specified |
| **Memory** | `Session` (e.g. `SQLiteSession`) or manual `to_input_list()` | `memory=True` (unified memory) |
| **Guardrails** | first-class: input / output / tool, with tripwires | *not covered in the available sources* |
| **Sandbox** | `SandboxAgent` with `Manifest` and capabilities | *not covered in the available sources* |
| **Tool library** | `@function_tool` + hosted tools (web search, file search, code interpreter, hosted MCP) | `@tool` + a large built-in library (`SerperDevTool`, scrapers, file readers, image generation) |
| **Lock-in seam** | hosted tools and hosted MCP | structural: YAML↔code binding, prompt shape, process model |

> **Read the two "not covered in the available sources" rows carefully.** They mean the CrewAI transcripts don't discuss guardrails or sandboxing — *not* that CrewAI lacks them. This is a gap in the source set, and it is recorded as such in [research-gaps](../wiki/research-gaps.md).

---

## The four differences that actually change your engineering

### 1. Who writes the system prompt

This is the root difference; everything else is downstream.

- **Agents SDK:** you write `instructions`. What you wrote is what the model sees.
- **CrewAI:** you write three fields and the framework composes them. Traces show it rendering the YAML `goal` as *"your personal goal is…"* and injecting scaffolding like *"you must return the actual complete content of the final answer, not a summary."*

The claimed benefit is that you inherit the CrewAI team's prompt-engineering work. The documented cost is directly stated: *"you have a little bit less transparency into the system prompts that are being put together and sent to the LLM."*

**The consequence for practice:** with CrewAI, **the trace is the only documentation of your real prompt.** You cannot read your prompt off the page — you have to run the thing and look. That changes what "reviewing a change" means.

### 2. The implicit context default

The single most consequential behavioural difference, and the one most likely to bite.

> *"If you don't specify a context, [CrewAI] will include all of the outputs from all prior tasks in the context. If you do specify a context, then crew is more limiting, and it will only include the outputs from the tasks that you mention."*

So **being explicit narrows rather than widens** — the opposite of most people's intuition. The Agents SDK has no equivalent: context between steps is whatever you pass.

This is a framework making a context-budget decision on your behalf, invisibly ([context-engineering](../wiki/context-engineering.md)). In a long crew, the default quietly grows the prompt on every task.

### 3. The orchestration switch

CrewAI collapses the central architectural choice of [workflow-vs-agent](../wiki/workflow-vs-agent.md) into one line:

| Process | Behaviour | Equivalent to |
|---|---|---|
| `sequential` | tasks run in definition order, respecting dependencies | orchestration **by code** |
| `hierarchical` | a manager LLM decides the order and delegates | orchestration **by LLM** |

That is a genuine strength — it makes the most important decision in agent design cheap to flip and easy to A/B. The Agents SDK requires you to restructure code to make the same move (successive `Runner.run()` calls vs. handoffs or `as_tool()`).

The flip side: a one-line switch into LLM orchestration is also a one-line switch into unpredictability. The production default in both source sets is the same — *"use LLM autonomy when autonomy provides real value, not simply because the framework makes it possible."*

### 4. Delegation semantics

The Agents SDK draws a distinction CrewAI does not surface:

> **Tools = "Help me do this." Handoff = "You take over from here."**

With **agents-as-tools**, the manager keeps control and gets a result back. With **handoffs**, the specialist becomes the conversation owner. CrewAI's hierarchical mode is closer to the first — a manager delegating and collecting — with no equivalent of a true ownership transfer in the available material.

If your problem is *routing* (hand the customer to billing and let billing finish), the SDK models it directly. If your problem is *decomposition* (do these five things and give me the results), both work.

---

## Choosing between them

Based on the sources, not on benchmarks — there are none.

**Reach for the OpenAI Agents SDK when:**
- you need to see and control the exact prompt
- guardrails, tripwires, or a sandbox are requirements
- the handoff-vs-agents-as-tools distinction maps onto your problem
- you want minimal abstraction between your code and the model

**Reach for CrewAI when:**
- you want a project scaffold and a large tool library on day one
- separating prompts from code into YAML is worth something to your team (reviewable by non-engineers, editable without touching code)
- your problem decomposes naturally into agents-with-assigned-tasks
- you want to A/B code-vs-LLM orchestration cheaply

**The deciding question is not features, it's transparency tolerance.** CrewAI trades visibility for speed. If you can debug from traces and you accept that the framework authors your prompt, the trade is good. If you need to reason about the prompt statically — compliance, high-stakes output, or a team that reviews prompts as artifacts — the SDK's directness is worth the extra code.

One structural note from [agent-frameworks](../wiki/agent-frameworks.md): the thing that eventually forces teams off a framework is *"fine-grained control over context management."* CrewAI's implicit context default is exactly the kind of thing that triggers that migration.

---

## What the sources cannot tell you

Recorded as gaps rather than papered over:

1. **No head-to-head.** No benchmark, no shared task suite, no cost or latency comparison. Anyone claiming one is faster or better is not citing this source set.
2. **CrewAI's central opinion is undemonstrated.** No evaluation shows role/goal/backstory outperforms a plain system prompt — despite it being the framework's defining design choice.
3. **No sequential-vs-hierarchical comparison** on the same crew and task suite, which is surprising given how cheap the switch is.
4. **Unified memory has no reported retrieval accuracy, storage policy, or cost profile.**
5. **No agents-as-tools vs. handoffs measurement.** The distinction is conceptual; the guidance is judgment.
6. **Neither framework's overhead is measured** relative to a hand-written loop — the same gap flagged for every framework in the wiki.
7. **Source asymmetry.** CrewAI has two transcripts focused on building projects; the SDK has a structured deep-dive covering guardrails, sandboxes, and MCP. Absence of a topic in one set is not evidence of absence in the framework.

---

## The framing worth keeping

Both source sets independently arrive at the same deflation, and it is the most useful thing here:

> *"Don't over-focus on framework selection. Different frameworks often provide different implementations of broadly similar underlying capabilities."*

The capability that transfers between them — good tool descriptions, deliberate context, evaluation against a real outcome, reading traces instead of guessing — is worth more than the choice itself. See [agents-are-systems](../wiki/agents-are-systems.md) and [when-to-build-an-agent](../wiki/when-to-build-an-agent.md).
