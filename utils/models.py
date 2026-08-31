"""Centralized model initialization"""

import logging
import os
from pathlib import Path
from dotenv import load_dotenv
load_dotenv(dotenv_path=Path(__file__).resolve().parent.parent / ".env", override=True)

from anthropic import APIConnectionError as AnthropicAPIConnectionError
from anthropic import APIError as AnthropicAPIError
from openai import APIConnectionError as OpenAIAPIConnectionError
from openai import APIError as OpenAIAPIError

from langchain.chat_models import init_chat_model

from utils.provider_env import scrub_ambient_provider_env

logger = logging.getLogger(__name__)

# Must run before the client below is built — the Anthropic SDK reads these
# vars in __init__ and they can silently override `api_key=`.
scrub_ambient_provider_env()

# --- Default: OpenAI, direct ---
# model = init_chat_model("openai:gpt-4.1-mini")

# --- OpenAI via the LangSmith LLM Gateway ---
# Routes every model call through the LangSmith Gateway so that workspace
# policies (PII / secrets / allow-lists / cost caps) are enforced.
# MODEL_CONFIG is the single source the frontend's Gateway pane reads.
DEFAULT_MODEL_ID = "bedrock/anthropic.claude-sonnet-5"

MODEL_CONFIG = {
    "model": DEFAULT_MODEL_ID,
    "provider": "anthropic",
    "base_url": "https://gateway.smith.langchain.com",
}


def resolve_model_id() -> str:
    """The model this process should use.

    `CHAT_LANGCHAIN_LITE_MODEL` overrides the default — that's how
    `scripts/setup.py` seeds one baseline experiment per model, and how
    `scripts/generate_traces.py` picks a cheaper model for bulk traces.
    Read at call time, not import time, so callers can set it and then build.
    """
    return os.getenv("CHAT_LANGCHAIN_LITE_MODEL") or DEFAULT_MODEL_ID


def build_model(model_id: str | None = None):
    """Construct a gateway-backed chat model.

    A factory rather than a singleton because the model id can change between
    calls within one process (see `resolve_model_id`).
    """
    return init_chat_model(
        model=model_id or resolve_model_id(),
        model_provider=MODEL_CONFIG["provider"],
        base_url=MODEL_CONFIG["base_url"],
        api_key=os.environ["LANGSMITH_API_KEY_GATEWAY"],
        max_tokens=300,
        # Newer Sonnets switch extended thinking on by themselves as soon as
        # tools are bound, and thinking tokens are spent from max_tokens. With
        # the intentional max_tokens=300 bug that budget is often gone before
        # any text block is emitted, so the agent returns *nothing* — blank in
        # the UI, empty `output` in evals — instead of the truncated answer
        # Bug 4 is supposed to demonstrate. The two can't be reconciled: the
        # minimum thinking budget is 1024 tokens, well above our 300.
        thinking={"type": "disabled"},
    )


# Default instance for callers that just need "the" model (e.g. run_evals'
# online-evaluator config). Agent builds go through build_model() instead.
model = build_model()


# --- Fallback: a second, independently-credentialed model ---
# Deliberately NOT another route through the gateway: the faults that blank out
# answers here (missing Bedrock workspace secret, denied gateway:invoke, bad
# model id) are properties of that one credential path, so a fallback sharing it
# would fail alongside the primary.
FALLBACK_MODEL_ENV = "CHAT_LANGCHAIN_LITE_FALLBACK_MODEL"
DEFAULT_FALLBACK_MODEL_ID = "openai:gpt-4.1-mini"
FALLBACK_API_KEY_ENV = "OPENAI_API_KEY"

SKIP_STARTUP_PROBE_ENV = "CHAT_LANGCHAIN_LITE_SKIP_MODEL_PROBE"

# Provider-side failures worth turning into a user-facing message rather than an
# empty answer. Narrow on purpose: a bug in our own code should still surface.
PROVIDER_ERRORS: tuple[type[Exception], ...] = (AnthropicAPIError, OpenAIAPIError)

_TRANSIENT_STATUS_CODES = frozenset({408, 409, 425, 429, 500, 502, 503, 504})


def is_transient_provider_error(exc: Exception) -> bool:
    """True only for provider faults another attempt could plausibly clear."""
    if isinstance(exc, (AnthropicAPIConnectionError, OpenAIAPIConnectionError)):
        return True
    status = getattr(exc, "status_code", None)
    return isinstance(status, int) and status in _TRANSIENT_STATUS_CODES


def build_fallback_model():
    """Construct the fallback chat model, or None when it isn't credentialed."""
    api_key = os.getenv(FALLBACK_API_KEY_ENV)
    if not api_key:
        return None
    return init_chat_model(
        os.getenv(FALLBACK_MODEL_ENV) or DEFAULT_FALLBACK_MODEL_ID,
        api_key=api_key,
        # Same budget as the primary so a fallback answer isn't shaped
        # differently from the answers evals score.
        max_tokens=300,
    )


def validate_model_config(model_id: str | None = None) -> None:
    """Probe the configured route once so a broken model/provider fails at startup."""
    if os.getenv(SKIP_STARTUP_PROBE_ENV):
        return
    resolved = model_id or resolve_model_id()
    try:
        build_model(resolved).invoke("ping", max_tokens=1)
    except Exception as exc:
        detail = (
            f"model={resolved!r} provider={MODEL_CONFIG['provider']!r} "
            f"base_url={MODEL_CONFIG['base_url']!r} failed its startup probe: "
            f"{type(exc).__name__}: {exc}"
        )
        if build_fallback_model() is None:
            raise RuntimeError(
                f"[models] {detail} — fix the gateway configuration, or set "
                f"{FALLBACK_API_KEY_ENV} so requests can be served from the "
                f"fallback model."
            ) from exc
        logger.error("[models] %s — serving from the fallback model instead.", detail)

# --- Anthropic ---
# model = init_chat_model("anthropic:claude-sonnet-4-5")

# --- Azure OpenAI ---
# from langchain_openai import AzureChatOpenAI
# model = AzureChatOpenAI(azure_deployment="gpt-4.1-mini", streaming=True)

# --- AWS Bedrock ---
# from langchain_aws import ChatBedrockConverse
# model = ChatBedrockConverse(
#     provider="anthropic",
#     model_id="anthropic.claude-sonnet-4-20250514-v1:0",
# )