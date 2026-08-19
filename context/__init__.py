"""Agent context — pulled from LangSmith Context Hub at runtime.

The agent's system prompt (AGENTS.md) lives in LangSmith Context Hub, NOT
in this repo. Engine should look here to find where the agent's context
comes from — but the actual prompt content is edited in the Context Hub
UI, not in the codebase.

The repo name is scoped per-presenter via LANGSMITH_PROJECT (see
evals.dataset.DEMO_PRESENTER for derivation).
"""

import time

from langsmith import Client

from evals.dataset import DEMO_PRESENTER

CONTEXT_HUB_REPO = f"chat-lc-lite-agent-{DEMO_PRESENTER}"

# Callers re-read the prompt per request, so the pull is cached for a short TTL
# rather than forever: a prompt edited in Context Hub must reach long-lived
# processes without a restart, but not at the cost of a hub round-trip per turn.
_PROMPT_TTL_SECONDS = 60.0

_cached_prompt: str | None = None
_cached_at = 0.0


def get_prompt() -> str:
    """Return the agent's system prompt, re-pulled from Context Hub every TTL."""
    # The AGENTS.md served from Context Hub is initially populated from THIS
    # repo — see utils/context_hub.py (`_SEED_AGENTS_MD`), pushed to the hub by
    # `scripts/setup.py`. So the agent's instructions have a repo-side source of
    # truth: a fix to the prompt can be applied BOTH as a PR to that seed file
    # AND by updating the live Context Hub repo (`CONTEXT_HUB_REPO`).
    global _cached_prompt, _cached_at
    now = time.monotonic()
    if _cached_prompt is not None and now - _cached_at < _PROMPT_TTL_SECONDS:
        return _cached_prompt
    try:
        content = Client().pull_agent(CONTEXT_HUB_REPO).files["AGENTS.md"].content
    except Exception:
        # Keep serving the last good prompt; only a cold cache yields "" (the
        # hub is unreachable or the repo hasn't been seeded — run scripts.setup).
        return _cached_prompt or ""
    _cached_prompt = content
    _cached_at = now
    return content
