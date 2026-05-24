# llm-fallback-chain

[![PyPI](https://img.shields.io/pypi/v/llm-fallback-chain.svg)](https://pypi.org/project/llm-fallback-chain/)
[![Python](https://img.shields.io/pypi/pyversions/llm-fallback-chain.svg)](https://pypi.org/project/llm-fallback-chain/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

**Tiny provider failover chain for LLM calls.**

You give it an ordered list of `(name, callable)` provider pairs. It tries
them one by one. The first one that returns wins. You get back the value,
the name of the provider that worked, and a trace of every attempt that
failed before the success. If every provider failed, you get a single
`AllProvidersFailedError` carrying the full trace.

Zero runtime deps. Works sync or async. Bring your own provider SDKs.

## Install

```bash
pip install llm-fallback-chain
```

## Basic use

```python
from llm_fallback_chain import FallbackChain, AllProvidersFailedError

chain = FallbackChain([
    ("anthropic", lambda prompt: anthropic_sdk.complete(prompt)),
    ("openai",    lambda prompt: openai_sdk.complete(prompt)),
    ("gemini",    lambda prompt: gemini_sdk.complete(prompt)),
])

try:
    result = chain.call("hello")
    print(result.value)      # whatever the winning provider returned
    print(result.provider)   # "anthropic" / "openai" / "gemini"
    print(result.attempts)   # failed Attempts before the success
except AllProvidersFailedError as e:
    for a in e.attempts:
        print(a.name, a.exception, f"{a.duration_ms:.1f}ms")
```

## Async

`call_async` awaits coroutine providers. Sync providers run inline.

```python
async def anthropic(prompt): return await anthropic_async.complete(prompt)
async def openai(prompt):    return await openai_async.complete(prompt)

chain = FallbackChain([("anthropic", anthropic), ("openai", openai)])
result = await chain.call_async("hello")
```

You can mix sync and async providers in the same chain.

## Custom skip predicate

By default any exception triggers a fallback. Often that is too greedy.
You only want to fall back on transient errors (rate limit, 5xx, timeout)
and re-raise on real user errors (bad request, validation).

```python
class RateLimited(Exception): pass
class Validation(Exception): pass

chain = FallbackChain(
    providers,
    should_fall_back=lambda exc: isinstance(exc, (RateLimited, TimeoutError)),
)

# A Validation error from the first provider re-raises straight away.
# The chain does not try the next provider.
```

## Telemetry callback

Hook into every fallback for logs or metrics. It fires once per fallback,
after the failing provider and before the next one runs.

```python
def on_fallback(failed_name, exc, next_name):
    log.warning("fallback %s -> %s: %s", failed_name, next_name, exc)

chain = FallbackChain(providers, on_fallback=on_fallback)
```

It does NOT fire when the first provider works, and it does NOT fire
after the last provider (there is no next provider to fall back to).

## Related libs

Compose with these to build out the stack:

- [`llm-retry-py`](https://github.com/MukundaKatta/llm-retry-py) - per-provider retry with backoff. Wrap each provider callable in a retry, then put them in the chain.
- [`llm-circuit-breaker-py`](https://github.com/MukundaKatta/llm-circuit-breaker-py) - per-provider circuit breaker. Wrap each provider callable in a breaker so a sick provider trips fast instead of slowing every call.

A typical stack is `chain( breaker( retry( provider ) ) )`.

## What it does NOT do

- No HTTP. No SDK lock-in. You write the provider callables.
- No built-in retry. Use a per-provider retry wrapper for that.
- No global state. Each `FallbackChain` is self-contained and reusable.
- No load balancing. Strict ordered fallback only.

## License

MIT
