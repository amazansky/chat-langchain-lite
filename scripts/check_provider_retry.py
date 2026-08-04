"""Regression check: a transient gateway 403 must never surface as an empty answer.

Runs fully offline against a fake model — no gateway calls, no API keys needed.

Usage:
    python -m scripts.check_provider_retry
"""

import os
import sys

# utils.models reads the gateway key at import time.
os.environ.setdefault("LANGSMITH_API_KEY_GATEWAY", "check-provider-retry")

from langchain.agents.middleware import model_retry
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult

from agent import agent as agent_module

ANSWER = "Use a checkpointer to persist LangGraph state."


class Gateway403ThenOk(BaseChatModel):
    """Fake gateway model that raises 403 for its first `failures` calls."""

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
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content=ANSWER))])


def _run(failures: int) -> tuple[dict, Gateway403ThenOk]:
    fake = Gateway403ThenOk(failures=failures)
    agent_module.model = fake
    model_retry.time.sleep = lambda _seconds: None
    return agent_module.invoke_agent("how do I persist state?"), fake


def main() -> int:
    result, fake = _run(failures=1)
    assert fake.calls >= 2, f"expected a retry after the 403, got {fake.calls} call(s)"
    assert result["output"] == ANSWER, f"expected the answer, got {result['output']!r}"

    result, _fake = _run(failures=99)
    assert result["output"] == agent_module.PROVIDER_UNAVAILABLE_MESSAGE, (
        f"exhausted retries should degrade to a user-facing message, got {result['output']!r}"
    )

    print("✅ transient 403 is retried, exhausted retries degrade to a message")
    return 0


if __name__ == "__main__":
    sys.exit(main())
