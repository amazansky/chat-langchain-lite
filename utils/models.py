"""Centralized model initialization"""

import os
from pathlib import Path
from dotenv import load_dotenv
load_dotenv(dotenv_path=Path(__file__).resolve().parent.parent / ".env", override=True)

from langchain.chat_models import init_chat_model

from utils.provider_env import scrub_ambient_provider_env

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
        # This route is thinking-enabled and thinking tokens are drawn from
        # this same output-token budget — `thinking={"type": "disabled"}` below
        # is not honored by the gateway, so a reasoning block still lands here.
        # The cap must therefore exceed the visible answer length by a
        # comfortable margin; response length itself is controlled by the
        # "under 200 words" guidance in AGENTS.md, not by max_tokens.
        max_tokens=2000,
        thinking={"type": "disabled"},
    )


# Default instance for callers that just need "the" model (e.g. run_evals'
# online-evaluator config). Agent builds go through build_model() instead.
model = build_model()

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