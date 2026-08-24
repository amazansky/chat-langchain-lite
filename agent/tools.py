from langchain_core.tools import tool

# Canned documentation snippets for the most-asked LangChain ecosystem concepts.
# Stand-in for what would normally be a Mintlify / docs search call so the demo
# stays self-contained.
CONCEPTS_DB = {
    "langchain": {
        "tagline": "The framework for building LLM applications.",
        "first_released": "2022",
        "package": "langchain",
        "min_python": "3.10+",
        "summary": "LangChain provides chains, agents, retrievers, and integrations with 700+ providers. Use it to compose LLM calls with tools, memory, and structured outputs.",
        "primary_use_case": "Composable LLM pipelines, RAG, and agents.",
        "api_surface": (
            "`create_agent(model=..., tools=[...], system_prompt=..., middleware=[...], "
            "checkpointer=...)` from `langchain.agents` builds an agent; run it with "
            "`.invoke({\"messages\": [...]}, config)` or `.stream(..., stream_mode=...)`. "
            "Declare tools with the `@tool` decorator from `langchain_core.tools`. "
            "Request typed results with `model.with_structured_output(Schema)`."
        ),
    },
    "langgraph": {
        "tagline": "Build stateful, multi-actor agents as graphs.",
        "first_released": "2024",
        "package": "langgraph",
        "min_python": "3.7+",
        "summary": "LangGraph models agents as graphs: nodes are functions, edges define control flow, and a typed state object is passed between them. Built-in persistence (checkpointers), interrupts, and streaming.",
        "primary_use_case": "Long-running, multi-step agents and human-in-the-loop workflows.",
        "api_surface": (
            "`StateGraph(StateSchema)` from `langgraph.graph` with `.add_node(name, fn)`, "
            "`.add_edge(src, dst)`, `.add_conditional_edges(src, router)` and the `START` / "
            "`END` sentinels; finish with `.compile(checkpointer=...)`. Run the compiled "
            "graph with `.invoke(state, config)`, `.stream(...)` or `.astream(...)`. Pause "
            "with `interrupt(...)` and resume with `Command(resume=...)` from "
            "`langgraph.types`."
        ),
    },
    "langsmith": {
        "tagline": "Observability and evaluation for LLM apps.",
        "first_released": "2023",
        "package": "langsmith",
        "min_python": "3.9+",
        "summary": "LangSmith captures traces of every LLM/tool call, lets you create datasets and run evaluations (offline and online), and provides annotation queues for human feedback. Works with any framework, not just LangChain.",
        "primary_use_case": "Tracing, evals, prompt management, and monitoring.",
        "api_surface": (
            "`from langsmith import Client, traceable, evaluate`. `Client()` exposes "
            "`create_dataset`, `create_examples` and `pull_prompt`; decorate any function "
            "with `@traceable` to trace it; run an offline eval with "
            "`evaluate(target_fn, data=..., evaluators=[...])`."
        ),
    },
    "deep agents": {
        "tagline": "Long-horizon agents with planning, memory, and subagents.",
        "first_released": "2024",
        "package": "deepagents",
        "min_python": "3.10+",
        "summary": "Deep Agents wraps create_agent with a TodoList planner, virtual filesystem, and SubAgentMiddleware for context isolation. Inspired by Claude Code's harness pattern.",
        "primary_use_case": "Research, coding, and other tasks that need planning and many tool calls.",
        "api_surface": (
            "`create_deep_agent(...)` from `deepagents` is the entrypoint. Middleware lives "
            "under `deepagents.middleware` (e.g. "
            "`from deepagents.middleware.filesystem import FilesystemMiddleware`) and "
            "filesystem backends under `deepagents.backends` (e.g. "
            "`from deepagents.backends.context_hub import ContextHubBackend`). Constructor "
            "keyword arguments beyond these are not covered here — do not name them; point "
            "the user to docs.langchain.com."
        ),
    },
    "middleware": {
        "tagline": "Hooks that wrap an agent's model and tool calls.",
        "first_released": "2024",
        "package": "langchain (langchain.agents.middleware)",
        "min_python": "3.10+",
        "summary": "AgentMiddleware lets you add cross-cutting behavior (retry, fallbacks, guardrails, human-in-the-loop) without modifying the agent itself. Stack middlewares — order matters.",
        "primary_use_case": "Human approval, content guardrails, retries, and structured output.",
        "api_surface": (
            "Subclass `AgentMiddleware` from `langchain.agents.middleware` and override "
            "only these hooks: `before_agent`, `before_model`, `after_model`, `after_agent`, "
            "`wrap_model_call`, `wrap_tool_call` (async variants: `abefore_agent`, "
            "`abefore_model`, `aafter_model`, `aafter_agent`, `awrap_model_call`, "
            "`awrap_tool_call`). Before/after hooks take `(state, runtime)`; wrap hooks take "
            "`(request, handler)` and call `handler(request)` to proceed. NO OTHER HOOK "
            "NAMES EXIST — `on_model_call`, `on_model_response`, `on_model_start`, "
            "`on_model_end`, `on_llm_start` and `on_llm_end` are NEVER dispatched, so a "
            "subclass defining them imports and instantiates cleanly while silently never "
            "running. Attach middleware with "
            "`create_agent(model=..., tools=[...], middleware=[MyMiddleware()])`."
        ),
    },
    "tracing": {
        "tagline": "Capture every LLM, tool, and chain call automatically.",
        "first_released": "2023",
        "package": "langsmith",
        "min_python": "3.9+",
        "summary": "Set LANGSMITH_TRACING=true and LANGSMITH_API_KEY in your env. Every LangChain/LangGraph run is traced to LangSmith automatically. For arbitrary Python functions use the @traceable decorator.",
        "primary_use_case": "Debugging agents, building eval datasets from real traffic.",
        "api_surface": (
            "Environment variables `LANGSMITH_TRACING=true`, `LANGSMITH_API_KEY` and "
            "`LANGSMITH_PROJECT` turn tracing on. `from langsmith import traceable` gives "
            "the `@traceable` decorator for arbitrary functions; per-run overrides go "
            "through the `config` argument (`RunnableConfig(run_name=..., metadata=..., "
            "tags=...)`)."
        ),
    },
}

SETUP_GUIDES_DB = {
    "installation": """Install the core packages with uv:

```bash
uv add langchain langgraph langsmith langchain-anthropic
```

For Deep Agents:
```bash
uv add deepagents
```

If you prefer pip:
```bash
pip install -U langchain langgraph langsmith langchain-anthropic
```

Minimum supported Python is 3.10 for langchain/langgraph and 3.9 for langsmith.""",

    "environment": """Recommended environment variables for a typical LangChain + LangSmith app:

```bash
ANTHROPIC_API_KEY=sk-ant-...          # or OPENAI_API_KEY, etc.
LANGSMITH_API_KEY=lsv2_...
LANGSMITH_TRACING=true                # enables tracing on every run
LANGSMITH_PROJECT=my-project          # auto-created on first trace
```

Load with python-dotenv at the top of your entrypoint:
```python
from dotenv import load_dotenv
load_dotenv(override=True)
```""",

    "deployment": """LangGraph apps can be deployed on the LangGraph Platform:

1. Add a langgraph.json at the project root pointing to your compiled graph.
2. Define dependencies in pyproject.toml or requirements.txt.
3. Deploy via the LangSmith UI (Deployments → New) or the langgraph CLI.

Platform features:
- Built-in persistence (Postgres-backed checkpointer)
- Streaming over WebSockets
- Background runs and crons
- Auth hooks for per-tenant access

For self-hosting, the same image can be run via docker compose — see the
self-hosted LangSmith docs for a worked example.""",

    "evaluation": """Build an offline evaluation in three steps:

1. Create a dataset in LangSmith (UI or `client.create_dataset`).
2. Write an evaluator function (LLM-as-judge or pure-code) returning
   `{"key": "...", "score": 0.0 or 1.0}`.
3. Run `from langsmith import evaluate; evaluate(target_fn, data=DATASET, evaluators=[...])`.

For online evaluation, register a run rule in the LangSmith Evaluators UI.
Every new trace in the project will be scored automatically.""",
}

# Best practices the agent can recommend without caveat.
SAFE_PATTERNS = [
    "For documentation, link users to python.langchain.com and js.langchain.com — these are the canonical reference sites",
    "Use LangSmith tracing in development and production — set LANGSMITH_TRACING=true",
    "Use create_agent (LangChain) or StateGraph (LangGraph) instead of hand-rolling a tool loop",
    "Pin minimum versions of langchain, langgraph, langsmith in pyproject.toml — these libraries iterate fast",
    "Wrap external API calls with retry middleware (model_retry_middleware) to survive transient failures",
    "Use structured output (with_structured_output) when downstream code parses the result",
    "Use checkpointers for any agent that needs to resume after a crash or interrupt",
    "Run offline evals against a versioned LangSmith dataset before merging prompt or code changes",
]

# Patterns the agent should warn users away from when they ask.
ANTIPATTERNS = [
    "Calling model.invoke() in a tight loop without retries — provider 429s will crash the run",
    "Stuffing entire documents into the system prompt — use a retriever or vector store instead",
    "Mutating state.messages directly inside a node — return a new dict so LangGraph's reducer can merge",
    "Skipping LangSmith tracing in production — without traces you have no way to debug failures",
    "Using max_tokens far below the response length the model needs — truncates answers mid-sentence",
    "Calling synchronous .invoke() inside an async node — blocks the event loop and kills concurrency",
]


@tool
def lookup_concept(concept_name: str) -> str:
    """Look up a LangChain ecosystem concept (langchain, langgraph, langsmith, deep agents, middleware, tracing). Returns tagline, first release year, package name, minimum Python version, summary, primary use case, and the concept's verified API surface (class, method, and hook names) when covered."""
    key = concept_name.lower().strip()
    for db_key, data in CONCEPTS_DB.items():
        if key in db_key or db_key in key:
            lines = [f"**{db_key.title()}** — {data['tagline']}"]
            lines.append(f"- First released: {data['first_released']}")
            lines.append(f"- Package: `{data['package']}`")
            lines.append(f"- Minimum Python: {data['min_python']}")
            lines.append(f"- Primary use case: {data['primary_use_case']}")
            lines.append("")
            lines.append(data["summary"])
            lines.append("")
            if data.get("api_surface"):
                lines.append(f"**API surface:** {data['api_surface']}")
            else:
                lines.append(
                    "**API surface:** this knowledge base does not cover the API for "
                    f"'{db_key}'. Do not state any class, method, hook, or decorator name "
                    "for it — say the knowledge base does not cover that API and point the "
                    "user to docs.langchain.com instead."
                )
            return "\n".join(lines)
    available = ", ".join(k.title() for k in CONCEPTS_DB.keys())
    return f"Concept '{concept_name}' not found. Available concepts: {available}"


@tool
def get_setup_guide(topic: str) -> str:
    """Get a setup or how-to guide for a LangChain ecosystem topic. Topics: installation, environment, deployment, evaluation."""
    key = topic.lower().strip()
    for db_key, content in SETUP_GUIDES_DB.items():
        if key in db_key or db_key in key:
            return f"**{db_key.title()} guide:**\n\n{content}"
    available = ", ".join(SETUP_GUIDES_DB.keys())
    return f"Topic '{topic}' not found. Available topics: {available}"


@tool
def get_security_advice(query: str) -> str:
    """Get security and best-practice advice for LangChain/LangGraph/LangSmith projects, including recommended patterns and antipatterns to avoid."""
    safe_list = "\n".join(f"  ✓ {item}" for item in SAFE_PATTERNS)
    antipatterns_list = "\n".join(f"  ✗ {item}" for item in ANTIPATTERNS)
    return f"""**LangChain Best Practices**

Your query: {query}

**RECOMMENDED patterns:**
{safe_list}

**ANTIPATTERNS — avoid these:**
{antipatterns_list}

**General guidelines:**
- Treat LangSmith traces as your primary debugger — open them before reading logs
- Run evals before merging — never ship a prompt change without a measured delta
- Keep prompts under version control (Prompt Hub or a checked-in `.py` file)
- Use middleware for cross-cutting concerns (retry, guardrails, HITL) instead of hand-rolling them

When in doubt, search docs.langchain.com or check the LangSmith Cookbook for worked examples."""


TOOLS = [lookup_concept, get_setup_guide, get_security_advice]
