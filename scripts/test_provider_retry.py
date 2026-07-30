"""Regression test: a transient gateway 403 must not fail the whole request.

Simulates the LangSmith LLM Gateway returning `403 Forbidden` on the first
model call and asserts the agent retries and still returns a non-empty answer.

Usage:
    python -m scripts.test_provider_retry
"""

import os
import sys

os.environ.setdefault("LANGSMITH_API_KEY_GATEWAY", "test-key")
os.environ.setdefault("LANGSMITH_TRACING", "false")

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult

import agent.agent as agent_module
from agent.agent import PROVIDER_UNAVAILABLE_MESSAGE, invoke_agent

ANSWER = "Use create_agent from langchain.agents."


class Gateway403ThenOk(BaseChatModel):
    """Fake gateway model that raises 403 for the first `failures` calls."""

    failures: int = 1
    calls: int = 0

    @property
    def _llm_type(self) -> str:
        return "gateway-403-then-ok"

    def bind_tools(self, tools, **kwargs):
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kwargs) -> ChatResult:
        self.calls += 1
        if self.calls <= self.failures:
            raise RuntimeError("Error code: 403 - Forbidden (gateway)")
        return ChatResult(
            generations=[ChatGeneration(message=AIMessage(content=ANSWER))]
        )


def _run(failures: int) -> tuple[dict, Gateway403ThenOk]:
    fake = Gateway403ThenOk(failures=failures)
    original = agent_module.model
    agent_module.model = fake
    try:
        return invoke_agent("How do I build an agent?"), fake
    finally:
        agent_module.model = original


def main() -> int:
    failed: list[str] = []

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
        print(f"❌ {message}")
    if failed:
        return 1
    print("✅ transient gateway 403 is retried and never returns an empty answer")
    return 0


if __name__ == "__main__":
    sys.exit(main())
