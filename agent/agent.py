import logging

from langchain.agents import create_agent
from langchain.agents.middleware import ModelFallbackMiddleware, ModelRetryMiddleware
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import AIMessageChunk, ToolMessage
from langchain_core.runnables import RunnableConfig

from deepagents.middleware.filesystem import FilesystemMiddleware
from deepagents.backends.context_hub import ContextHubBackend

from agent.tools import TOOLS
from context import CONTEXT_HUB_REPO, get_prompt
from utils.streaming import iter_text
from utils.models import FALLBACK_MODEL, MODEL_CONFIG, RETRYABLE_ERRORS, model

_log = logging.getLogger(__name__)

# AGENTS.md is the agent's system prompt — pulled fresh from LangSmith
# Context Hub at module import.
# Seed source: utils/context_hub.py (`_SEED_AGENTS_MD`), pushed to Context Hub by
# `scripts/setup.py` (`push_agents_md()`). A prompt fix can be applied BOTH as a
# PR to that seed AND to the live Context Hub.
SYSTEM_PROMPT = get_prompt()


# utils/models.py builds the client from MODEL_CONFIG, so that is the only
# truthful source for the `model` value stamped into run metadata.
def _model_id() -> str:
    return MODEL_CONFIG["model"]


# Shown to the user when no model path answered. An empty response reads as a
# broken app; this at least says what happened and that retrying may work.
_FAILURE_MESSAGE = (
    "Sorry — I couldn't reach the model just now, so I don't have an answer for you. "
    "Please try again in a moment."
)


# The Context Hub-backed filesystem holds the agent's OWN context (AGENTS.md,
# playbooks) — it is a read-only reference, NOT a user-delivery channel.
_READONLY_FS_TOOLS = {"ls", "read_file", "glob", "grep"}


def _readonly_context_hub_fs() -> FilesystemMiddleware:
    fs = FilesystemMiddleware(backend=ContextHubBackend(CONTEXT_HUB_REPO))
    fs.tools = [t for t in fs.tools if t.name in _READONLY_FS_TOOLS]
    return fs


def _resilience_middleware() -> list:
    """Bounded retry on transient faults, then a differently-credentialed fallback."""
    retry = ModelRetryMiddleware(
        max_retries=2,
        retry_on=RETRYABLE_ERRORS,
        initial_delay=0.5,
        # Re-raise once exhausted so invoke_agent/stream_agent can report the
        # failure to the user rather than the agent continuing on an error turn.
        on_failure="error",
    )
    if FALLBACK_MODEL is None:
        return [retry]
    # Fallback goes first so it wraps the retry: each model gets its own bounded
    # retry before the next one is tried.
    return [ModelFallbackMiddleware(FALLBACK_MODEL), retry]


def build_agent():
    return create_agent(
        # temperature=0 for deterministic, reproducible demo behavior — the
        # intentional bugs (tone, scope, truncation) come from the prompt and
        # max_tokens, not sampling, so pinning temperature keeps traces consistent.
        model=model,
        tools=TOOLS,
        system_prompt=SYSTEM_PROMPT,
        middleware=[*_resilience_middleware(), _readonly_context_hub_fs()],
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
    except Exception as exc:
        # Provider SDK error hierarchies share no common base across the primary
        # and fallback providers, so the guard is broad; the traceback is kept.
        _log.exception("Agent invoke failed: %s: %s", type(exc).__name__, exc)
        return {"output": _FAILURE_MESSAGE, "tools_called": [], "messages": []}
    output = next(
        (m.content for m in reversed(result["messages"])
         if isinstance(getattr(m, "content", None), str) and m.content),
        "",
    )
    tools_called = [m.name for m in result["messages"] if isinstance(m, ToolMessage)]
    return {"output": output, "tools_called": tools_called, "messages": result["messages"]}


def stream_agent(question: str, thread_id: str | None = None):
    """Stream the agent's response text as it's generated."""
    emitted = False
    try:
        for chunk, _meta in build_agent().stream(
            _user_msg(question), _config(thread_id), stream_mode="messages"
        ):
            if isinstance(chunk, AIMessageChunk):
                for text in iter_text(chunk):
                    emitted = True
                    yield text
    except Exception as exc:
        _log.exception("Agent stream failed: %s: %s", type(exc).__name__, exc)
        yield f"\n\n{_FAILURE_MESSAGE}" if emitted else _FAILURE_MESSAGE
