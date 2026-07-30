import os

from langchain.agents import create_agent
from langchain.agents.middleware import ModelRetryMiddleware
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import AIMessageChunk, ToolMessage
from langchain_core.runnables import RunnableConfig

from deepagents.middleware.filesystem import FilesystemMiddleware
from deepagents.backends.context_hub import ContextHubBackend

from agent.tools import TOOLS
from context import CONTEXT_HUB_REPO, get_prompt
from utils.streaming import iter_text
from utils.models import model

# AGENTS.md is the agent's system prompt — pulled fresh from LangSmith
# Context Hub at module import.
# Seed source: utils/context_hub.py (`_SEED_AGENTS_MD`), pushed to Context Hub by
# `scripts/setup.py` (`push_agents_md()`). A prompt fix can be applied BOTH as a
# PR to that seed AND to the live Context Hub.
SYSTEM_PROMPT = get_prompt()

# Override with CHAT_LANGCHAIN_LITE_MODEL env var — used by setup.py to seed
# baseline experiments against a more expensive model (Sonnet) for the
# demo's cost/latency comparison.
_DEFAULT_MODEL = "claude-haiku-4-5-20251001"


def _model_id() -> str:
    return os.getenv("CHAT_LANGCHAIN_LITE_MODEL") or _DEFAULT_MODEL


# The Context Hub-backed filesystem holds the agent's OWN context (AGENTS.md,
# playbooks) — it is a read-only reference, NOT a user-delivery channel.
_READONLY_FS_TOOLS = {"ls", "read_file", "glob", "grep"}


def _readonly_context_hub_fs() -> FilesystemMiddleware:
    fs = FilesystemMiddleware(backend=ContextHubBackend(CONTEXT_HUB_REPO))
    fs.tools = [t for t in fs.tools if t.name in _READONLY_FS_TOOLS]
    return fs


PROVIDER_UNAVAILABLE_MESSAGE = (
    "The model provider is temporarily unavailable — please retry."
)


def _provider_unavailable(exc: Exception) -> str:
    return PROVIDER_UNAVAILABLE_MESSAGE


def _model_retry() -> ModelRetryMiddleware:
    # The gateway can return transient 403/429/5xx. The Anthropic SDK's own
    # max_retries does not cover 403, and model.with_retry() can't be used here
    # because create_agent needs a bind_tools-capable model, so retry at the
    # agent loop instead — and degrade to a short message rather than failing
    # the whole run once retries are exhausted.
    return ModelRetryMiddleware(
        max_retries=3,
        on_failure=_provider_unavailable,
    )


def build_agent():
    return create_agent(
        # temperature=0 for deterministic, reproducible demo behavior — the
        # intentional bugs (tone, scope, truncation) come from the prompt and
        # max_tokens, not sampling, so pinning temperature keeps traces consistent.
        model=model,
        tools=TOOLS,
        system_prompt=SYSTEM_PROMPT,
        middleware=[_readonly_context_hub_fs(), _model_retry()],
    )


def _config(thread_id: str | None = None) -> RunnableConfig:
    metadata = {"demo": "true", "demo_type": "chat-lc-lite", "model": _model_id()}
    if thread_id:
        metadata["thread_id"] = thread_id
    return RunnableConfig(
        run_name="chat-lc-lite-demo",
        metadata=metadata,
        tags=["engine-demo", CONTEXT_HUB_REPO],
    )


def _user_msg(question: str) -> dict:
    return {"messages": [{"role": "user", "content": question}]}


def invoke_agent(question: str, thread_id: str | None = None) -> dict:
    """Run the agent once. Returns {output, tools_called, messages}."""
    try:
        result = build_agent().invoke(_user_msg(question), _config(thread_id))
    except Exception:
        return {
            "output": PROVIDER_UNAVAILABLE_MESSAGE,
            "tools_called": [],
            "messages": [],
        }
    output = next(
        (m.content for m in reversed(result["messages"])
         if isinstance(getattr(m, "content", None), str) and m.content),
        "",
    )
    tools_called = [m.name for m in result["messages"] if isinstance(m, ToolMessage)]
    return {"output": output, "tools_called": tools_called, "messages": result["messages"]}


def stream_agent(question: str, thread_id: str | None = None):
    """Stream the agent's response text as it's generated."""
    try:
        for chunk, _meta in build_agent().stream(
            _user_msg(question), _config(thread_id), stream_mode="messages"
        ):
            if isinstance(chunk, AIMessageChunk):
                yield from iter_text(chunk)
    except Exception:
        yield PROVIDER_UNAVAILABLE_MESSAGE
