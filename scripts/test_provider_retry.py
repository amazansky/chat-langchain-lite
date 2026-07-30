"""Regression test: a transient gateway 403 must not fail the whole request.

Simulates the LangSmith LLM Gateway returning `403 Forbidden` on the first
model call and asserts the agent retries and still returns a non-empty answer.

Usage:
    python -m scripts.test_provider_retry
"""

import os
import sys

# utils.models reads this at import time; the fake model below never calls out.
os.environ.setdefault("LANGSMITH_API_KEY_GATEWAY", "test-not-a-real-key")
os.environ.setdefault("LANGSMITH_TRACING", "false")

from langchain.agents import create_agent
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult

from agent.agent import PROVIDER_UNAVAILABLE_MESSAGE, _model_retry

ANSWER = "LangGraph persists state with a checkpointer."


class Gateway403ThenOk(BaseChatModel):
    """Fake gateway model that raises 403 for the first `failures` calls."""

    failures: int = 1
    calls: int = 0

    @property
    def _llm_type(self) -> str:
        return "gateway-403-then-ok"

    def _generate(self, messages, stop=None, run_manager=None, **kwargs) -> ChatResult:
        self.calls += 1
        if self.calls <= self.failures:
            raise RuntimeError("Error code: 403 - Forbidden (gateway)")
        return ChatResult(
            generations=[ChatGeneration(message=AIMessage(content=ANSWER))]
        )


def _run(failures: int) -> tuple[dict, Gateway403ThenOk]:
    fake = Gateway403ThenOk(failures=failures)
    # Retry delays are irrelevant to the assertions — keep the test fast.
    retry = _model_retry()
    retry.initial_delay = 0.0
    retry.backoff_factor = 0.0
    agent = create_agent(model=fake, tools=[], middleware=[retry])
    result = agent.invoke({"messages": [{"role": "user", "content": "hi"}]})
    output = next(
        (m.content for m in reversed(result["messages"])
         if isinstance(getattr(m, "content", None), str) and m.content),
        "",
    )
    return {"output": output}, fake


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
