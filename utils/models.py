"""Centralized model initialization"""

import os
from dotenv import load_dotenv
load_dotenv(dotenv_path="../.env", override=True)

from langchain.chat_models import init_chat_model

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

# The gateway model id carries a routing prefix (`bedrock/`) that LangSmith
# cannot match against its price table, so LLM spans landed with token counts
# but null costs. Client-level `metadata` takes precedence over the ls_* params
# a chat model derives from its own fields, so it is what gives these spans a
# priceable identity — without changing the id actually sent to the gateway.
LS_MODEL_NAME = MODEL_CONFIG["model"].rsplit("/", 1)[-1]

model = init_chat_model(
    model=MODEL_CONFIG["model"],
    model_provider=MODEL_CONFIG["provider"],
    base_url=MODEL_CONFIG["base_url"],
    api_key=os.environ["LANGSMITH_API_KEY_GATEWAY"],
    max_tokens=300,
    metadata={
        "ls_provider": MODEL_CONFIG["provider"],
        "ls_model_name": LS_MODEL_NAME,
    },
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