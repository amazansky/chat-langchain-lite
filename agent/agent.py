import logging

from langchain.agents import create_agent
from langchain.agents.middleware import ModelFallbackMiddleware, ModelRetryMiddleware
from langchain_core.messages import AIMessage, AIMessageChunk, ToolMessage
from langchain_core.runnables import RunnableConfig

from deepagents.middleware.filesystem import FilesystemMiddleware
from deepagents.backends.context_hub import ContextHubBackend

from agent.tools import TOOLS
from context import CONTEXT_HUB_REPO, get_prompt
from utils.streaming import iter_text, text_of
from utils.models import (
    PROVIDER_ERRORS,
    build_fallback_model,
    build_model,
    is_transient_provider_error,
    resolve_model_id,
    validate_model_config,
)

logger = logging.getLogger(__name__)

# What the user sees when no model path answered. Anything is better than the
# zero-character response a provider fault used to produce.
MODEL_FAILURE_MESSAGE = (
    "Sorry — I couldn't reach the model just now, so I have no answer for that. "
    "Please try again in a moment."
)

# AGENTS.md is the agent's system prompt — pulled fresh from LangSmith
# Context Hub at module import.
# Seed source: utils/context_hub.py (`_SEED_AGENTS_MD`), pushed to Context Hub by
# `scripts/setup.py` (`push_agents_md()`). A prompt fix can be applied BOTH as a
# PR to that seed AND to the live Context Hub.
SYSTEM_PROMPT = get_prompt()

# Override with CHAT_LANGCHAIN_LITE_MODEL env var — used by setup.py to seed
# baseline experiments against a more expensive model (Sonnet) for the
# demo's cost/latency comparison. Resolution lives in utils/models so the
# model actually built and the model recorded in trace metadata can't drift.
_model_id = resolve_model_id

# A misconfigured model id, provider or credential is a startup problem: find it
# once here rather than once per user request.
validate_model_config()


# The Context Hub-backed filesystem holds the agent's OWN context (AGENTS.md,
# playbooks) — it is a read-only reference, NOT a user-delivery channel.
_READONLY_FS_TOOLS = {"ls", "read_file", "glob", "grep"}


def _readonly_context_hub_fs() -> FilesystemMiddleware:
    fs = FilesystemMiddleware(backend=ContextHubBackend(CONTEXT_HUB_REPO))
    fs.tools = [t for t in fs.tools if t.name in _READONLY_FS_TOOLS]
    return fs


def _resilience_middleware() -> list:
    retry = ModelRetryMiddleware(
        max_retries=2,
        # Only transient faults; a deterministic 400/403/404 (the shape that
        # blanks answers here) must not burn attempts before the fallback runs.
        retry_on=is_transient_provider_error,
        # Re-raise so invoke_agent/stream_agent log the real exception and say
        # so, instead of the agent continuing on a synthesized error turn.
        on_failure="error",
        initial_delay=0.5,
    )
    fallback = build_fallback_model()
    # First in the list is outermost: the fallback wraps the retry, so each
    # credential path gets its own bounded set of attempts.
    return [ModelFallbackMiddleware(fallback), retry] if fallback else [retry]


def build_agent():
    return create_agent(
        # Built per call so a CHAT_LANGCHAIN_LITE_MODEL set after import (as
        # setup.py's baseline loop does) actually takes effect.
        model=build_model(),
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
    except PROVIDER_ERRORS as exc:
        logger.exception("[agent] model call failed: %s: %s", type(exc).__name__, exc)
        return {"output": MODEL_FAILURE_MESSAGE, "tools_called": [], "messages": []}
    # Must be an AIMessage: falling back to any str-content message would
    # silently return the user's own question as the "answer".
    output = next(
        (text for m in reversed(result["messages"])
         if isinstance(m, AIMessage) and (text := text_of(m))),
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
    except PROVIDER_ERRORS as exc:
        logger.exception("[agent] model call failed: %s: %s", type(exc).__name__, exc)
        yield MODEL_FAILURE_MESSAGE
