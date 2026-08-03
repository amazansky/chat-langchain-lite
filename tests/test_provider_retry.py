"""A transient gateway 403 must be retried, never returned as an empty answer."""

import os

# utils.models reads the gateway key at import time.
os.environ.setdefault("LANGSMITH_API_KEY_GATEWAY", "test-key")

from langchain.agents.middleware import model_retry
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult

from agent import agent as agent_module
from agent.agent import PROVIDER_UNAVAILABLE_MESSAGE, invoke_agent

ANSWER = "Use a checkpointer to persist LangGraph state."


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


def _run(monkeypatch, failures: int):
    fake = Gateway403ThenOk(failures=failures)
    monkeypatch.setattr(agent_module, "model", fake)
    monkeypatch.setattr(model_retry.time, "sleep", lambda _seconds: None)
    return invoke_agent("how do I persist state?"), fake


def test_transient_403_is_retried_and_still_answers(monkeypatch):
    result, fake = _run(monkeypatch, failures=1)
    assert fake.calls >= 2
    assert result["output"] == ANSWER


def test_exhausted_retries_degrade_to_a_user_facing_message(monkeypatch):
    result, _fake = _run(monkeypatch, failures=99)
    assert result["output"] == PROVIDER_UNAVAILABLE_MESSAGE
