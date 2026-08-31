"""Guard against provider credentials leaking in from the ambient shell.

Everything in this repo talks to Anthropic models through the LangSmith
Gateway, passing the gateway key explicitly (`api_key=...`). That explicit
argument is NOT the last word: the Anthropic SDK also reads a handful of
`ANTHROPIC_*` environment variables in `Anthropic.__init__` and merges them
*after* the constructor's own auth headers.

`ANTHROPIC_CUSTOM_HEADERS` is the dangerous one. It's parsed into a header
dict (`anthropic/_client.py`), and `_api_key_auth` emits its key under the
exact same name — `X-Api-Key` — so a shell-level

    ANTHROPIC_CUSTOM_HEADERS="X-Api-Key: <some other key>"

silently *replaces* the gateway key we passed in, for every client built in
this process. The request still authenticates (the leaked key is valid), just
against the wrong workspace — which surfaces much later as a confusing
"bedrock credentials not found: secret not found in workspace secrets" 400
rather than an auth error.

This bites anyone running the demo from a machine with these vars exported
globally.

Call `scrub_ambient_provider_env()` before constructing any Anthropic client.
The SDK snapshots these vars at construction time, so scrubbing afterwards has
no effect on an existing client.
"""

import os

# Vars the Anthropic SDK reads in __init__ that can override, or redirect, the
# credentials we pass explicitly. Deliberately narrow: only what collides.
_LEAKY_VARS = (
    "ANTHROPIC_CUSTOM_HEADERS",  # can overwrite the X-Api-Key auth header
    "ANTHROPIC_API_KEY",
    "ANTHROPIC_AUTH_TOKEN",
    "ANTHROPIC_BASE_URL",  # would send gateway-scoped keys somewhere else
)


def scrub_ambient_provider_env() -> list[str]:
    """Remove inherited ANTHROPIC_* vars so explicit `api_key=` wins.

    Returns the names that were actually removed, so callers can surface it —
    a silent scrub would be its own debugging trap.
    """
    removed = [name for name in _LEAKY_VARS if os.environ.pop(name, None) is not None]
    if removed:
        print(
            "[provider_env] Ignoring inherited "
            + ", ".join(removed)
            + " — this project authenticates to the LangSmith Gateway with "
            "LANGSMITH_API_KEY instead."
        )
    return removed
