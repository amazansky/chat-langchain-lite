"""Regression test: a transient gateway 403 must not fail the whole request.

Simulates the LangSmith LLM Gateway returning `403 Forbidden` on the first
model call and asserts the agent retries and still returns a non-empty answer.

Usage:
    python -m scripts.test_provider_retry
"""

import os
import sys
from typing import Any

# utils.models reads this at import time; the fake model below never calls out.
os.environ.setdefault("LANGSMITH_API_KEY_GATEWAY", "test-not-used")

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult

import agent.agent as agent_module
from agent.agent import PROVIDER_UNAVAILABLE_MESSAGE, invoke_agent

ANSWER = "LangGraph models agents as graphs."


class Gateway403ThenOk(BaseChatModel):
    """Fake gateway model that raises 403 on the first call, then succeeds."""

    calls: int = 0
    failures: int = 1

    @property
    def _llm_type(self) -> str:
        return "gateway-403-then-ok"

    def bind_tools(self, tools: Any, **kwargs: Any) -> "Gateway403ThenOk":
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kwargs) -> ChatResult:
        self.calls += 1
        if self.calls <= self.failures:
            raise RuntimeError("Error code: 403 - Forbidden (gateway)")
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content=ANSWER))])


def _run(failures: int) -> tuple[dict, Gateway403ThenOk]:
    fake = Gateway403ThenOk(failures=failures)
    original = agent_module.model
    agent_module.model = fake
    try:
        return invoke_agent("What is LangGraph?"), fake
    finally:
        agent_module.model = original


def main() -> int:
    failed = []

    result, fake = _run(failures=1)
    if fake.calls < 2:
        failed.append(f"expected a retry after 403, got {fake.calls} model call(s)")
    if result["output"] != ANSWER:
        failed.append(f"expected the retried answer, got {result['output']!r}")

    # Retries exhausted: degrade to a user-facing message, never an empty answer.
    result, _ = _run(failures=99)
    if result["output"] != PROVIDER_UNAVAILABLE_MESSAGE:
        failed.append(f"expected the unavailable message, got {result['output']!r}")

    for message in failed:
        print(f"FAIL: {message}")
    if failed:
        return 1
    print("PASS: transient gateway 403 is retried and never yields an empty answer")
    return 0


if __name__ == "__main__":
    sys.exit(main())
