"""llm-fallback-chain - tiny provider failover for LLM calls.

Wrap an ordered list of (name, callable) provider pairs. Each call tries
them in order. If a provider raises, the chain decides whether to fall
back or re-raise, then moves on. You get back a `ChainResult` with the
return value, the winning provider name, and a trace of every failed
attempt.

    from llm_fallback_chain import FallbackChain, AllProvidersFailedError

    chain = FallbackChain([
        ("anthropic", lambda prompt: anthropic_sdk.complete(prompt)),
        ("openai",    lambda prompt: openai_sdk.complete(prompt)),
        ("gemini",    lambda prompt: gemini_sdk.complete(prompt)),
    ])

    result = chain.call("hello")
    print(result.value)      # whatever the winning provider returned
    print(result.provider)   # "anthropic" / "openai" / "gemini"
    print(result.attempts)   # [Attempt(name="anthropic", exception=..., duration_ms=...)]

Pluggable predicate so you can whitelist only certain exceptions (for
example, only fall back on rate-limit / 5xx, not on validation errors):

    chain = FallbackChain(
        providers,
        should_fall_back=lambda exc: isinstance(exc, (RateLimited, TimeoutError)),
    )

Async via `call_async`. Pair with `llm-retry-py` for per-provider retry
and `llm-circuit-breaker-py` for per-provider breaker if you want a
deeper stack.
"""

from llm_fallback_chain.chain import (
    AllProvidersFailedError,
    Attempt,
    ChainResult,
    FallbackChain,
)

__version__ = "0.1.0"

__all__ = [
    "AllProvidersFailedError",
    "Attempt",
    "ChainResult",
    "FallbackChain",
    "__version__",
]
