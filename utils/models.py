"""Centralized model initialization"""

import os
from dotenv import load_dotenv
load_dotenv(dotenv_path="../.env", override=True)

from langchain.chat_models import init_chat_model

# --- Default: OpenAI, direct ---
# model = init_chat_model("openai:gpt-4.1-mini")

# Answers truncate mid-sentence below roughly a thousand output tokens, so a
# too-small deployment-side override is clamped rather than honored.
MIN_MAX_TOKENS = 1024


def _resolve_max_tokens() -> int:
    """Read the configured output-token ceiling, never dropping below MIN_MAX_TOKENS."""
    configured = int(os.getenv("CHAT_LANGCHAIN_LITE_MAX_TOKENS", "4096"))
    return max(configured, MIN_MAX_TOKENS)


# --- OpenAI via the LangSmith LLM Gateway ---
# Routes every model call through the LangSmith Gateway so that workspace
# policies (PII / secrets / allow-lists / cost caps) are enforced.
# MODEL_CONFIG is the single source the frontend's Gateway pane reads.
MODEL_CONFIG = {
    "model": "claude-sonnet-4-6",
    "provider": "anthropic",
    "base_url": "https://gateway.smith.langchain.com/anthropic",
}
model = init_chat_model(
    model=MODEL_CONFIG["model"],
    model_provider=MODEL_CONFIG["provider"],
    base_url=MODEL_CONFIG["base_url"],
    api_key=os.environ["LANGSMITH_API_KEY_GATEWAY"],
    max_tokens=_resolve_max_tokens(),
    temperature=0,
)

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