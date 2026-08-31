"""Centralized model initialization"""

import logging
import os
from dotenv import load_dotenv
load_dotenv(dotenv_path="../.env", override=True)

from anthropic import (
    APIConnectionError,
    APITimeoutError,
    InternalServerError,
    RateLimitError,
)
from langchain.chat_models import init_chat_model
from langchain_core.language_models import BaseChatModel

_log = logging.getLogger(__name__)

MAX_TOKENS = 300

# --- Default: OpenAI, direct ---
# model = init_chat_model("openai:gpt-4.1-mini")

# --- OpenAI via the LangSmith LLM Gateway ---
# Routes every model call through the LangSmith Gateway so that workspace
# policies (PII / secrets / allow-lists / cost caps) are enforced.
# MODEL_CONFIG is the single source the frontend's Gateway pane reads.
MODEL_CONFIG = {
    "model": "bedrock/anthropic.claude-sonnet-5",
    "provider": "anthropic",
    "base_url": "https://gateway.smith.langchain.com",
}
model = init_chat_model(
    model=MODEL_CONFIG["model"],
    model_provider=MODEL_CONFIG["provider"],
    base_url=MODEL_CONFIG["base_url"],
    api_key=os.environ["LANGSMITH_API_KEY_GATEWAY"],
    max_tokens=MAX_TOKENS,
)

# --- Fallback: second, independently-credentialed path ---
# A gateway-side Bedrock credential, permission, or model-id fault takes out
# MODEL_CONFIG for every request at once. This model is reached with its own
# provider key, so it does not share that failure mode. Skipped when its
# credential is absent — no fallback is better than a second dead path.
FALLBACK_MODEL_CONFIG = {
    "model": os.getenv("CHAT_LANGCHAIN_LITE_FALLBACK_MODEL", "gpt-4.1-mini"),
    "provider": os.getenv("CHAT_LANGCHAIN_LITE_FALLBACK_PROVIDER", "openai"),
    "api_key_env": os.getenv("CHAT_LANGCHAIN_LITE_FALLBACK_API_KEY_ENV", "OPENAI_API_KEY"),
}

# Transient faults worth another attempt. Deterministic ones (400 bad request,
# 403 permission, 404 model id) fail identically on every attempt, so they skip
# the retry and go straight to the fallback model.
RETRYABLE_ERRORS = (APIConnectionError, APITimeoutError, InternalServerError, RateLimitError)


def _label(config: dict) -> str:
    endpoint = config.get("base_url") or "provider default"
    return f'{config["provider"]}:{config["model"]} via {endpoint}'


def _build_fallback_model() -> BaseChatModel | None:
    """Build the secondary model, or None when its credential or package is missing."""
    api_key = os.getenv(FALLBACK_MODEL_CONFIG["api_key_env"])
    if not api_key:
        _log.warning(
            "No fallback model configured: %s is unset, so a fault on %s leaves "
            "no second path for the request.",
            FALLBACK_MODEL_CONFIG["api_key_env"],
            _label(MODEL_CONFIG),
        )
        return None
    try:
        return init_chat_model(
            model=FALLBACK_MODEL_CONFIG["model"],
            model_provider=FALLBACK_MODEL_CONFIG["provider"],
            api_key=api_key,
            max_tokens=MAX_TOKENS,
        )
    except Exception as exc:
        _log.warning(
            "Fallback model %s unavailable: %s: %s",
            _label(FALLBACK_MODEL_CONFIG),
            type(exc).__name__,
            exc,
        )
        return None


FALLBACK_MODEL = _build_fallback_model()


def _probe(client: BaseChatModel, label: str) -> Exception | None:
    try:
        client.invoke([{"role": "user", "content": "ping"}], max_tokens=1)
    except Exception as exc:
        _log.warning("Startup probe failed for %s: %s: %s", label, type(exc).__name__, exc)
        return exc
    return None


def validate_model_config() -> None:
    """Probe the configured models at startup so misconfiguration fails here, not per request."""
    primary_error = _probe(model, _label(MODEL_CONFIG))
    if primary_error is None:
        return
    if FALLBACK_MODEL is not None and _probe(FALLBACK_MODEL, _label(FALLBACK_MODEL_CONFIG)) is None:
        _log.warning(
            "Primary model %s did not resolve; serving from fallback %s until it is fixed.",
            _label(MODEL_CONFIG),
            _label(FALLBACK_MODEL_CONFIG),
        )
        return
    raise RuntimeError(
        f"No usable model. Primary {_label(MODEL_CONFIG)} did not resolve "
        f"({type(primary_error).__name__}: {primary_error}) and no fallback model "
        f"answered. Check the model id and provider in MODEL_CONFIG, the gateway "
        f"workspace credentials for that provider, and "
        f"{FALLBACK_MODEL_CONFIG['api_key_env']}."
    ) from primary_error


# Set CHAT_LANGCHAIN_LITE_VALIDATE_MODEL=false to import without touching the
# network (offline dev, unit tests).
if os.getenv("CHAT_LANGCHAIN_LITE_VALIDATE_MODEL", "true").strip().lower() not in {
    "0",
    "false",
    "no",
}:
    validate_model_config()

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
