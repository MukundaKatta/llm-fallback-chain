"""Core FallbackChain implementation."""

from __future__ import annotations

import asyncio
import inspect
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class Attempt:
    """One provider attempt within a chain call.

    Attributes:
        name: provider name as passed to FallbackChain
        exception: the exception the provider raised (None on success)
        duration_ms: wall time the provider took, in milliseconds
    """

    name: str
    exception: BaseException | None
    duration_ms: float


@dataclass(frozen=True)
class ChainResult:
    """Outcome of a successful FallbackChain call.

    Attributes:
        value: whatever the winning provider returned
        provider: name of the provider that succeeded
        attempts: list of failed attempts that came before the success.
                  Empty when the first provider worked.
    """

    value: Any
    provider: str
    attempts: list[Attempt] = field(default_factory=list)


class AllProvidersFailedError(Exception):
    """Raised when every provider in the chain failed.

    Attributes:
        attempts: one Attempt per provider tried, in order
    """

    def __init__(self, attempts: list[Attempt]) -> None:
        self.attempts = attempts
        names = ", ".join(a.name for a in attempts)
        super().__init__(f"all providers failed: {names}")


# default predicate: any exception is a reason to fall back
def _default_should_fall_back(exc: BaseException) -> bool:
    return True


ProviderSync = Callable[..., Any]
ProviderAsync = Callable[..., Awaitable[Any]]
Provider = ProviderSync | ProviderAsync

OnFallback = Callable[[str, BaseException, str], None]
ShouldFallBack = Callable[[BaseException], bool]


class FallbackChain:
    """Ordered list of LLM providers to try in sequence.

    Each provider is a (name, callable) pair. `call(*args, **kw)` tries
    them in order until one returns without raising. Returns a `ChainResult`
    describing what won and what failed along the way. If every provider
    raises, `AllProvidersFailedError` is raised.

    The decision to fall back vs re-raise is up to `should_fall_back`.
    Default falls back on any exception. Pass a custom predicate to
    whitelist (e.g. only fall back on rate-limit / 5xx, not on a user
    validation error).

    `on_fallback(failed_name, exception, next_name)` fires after a provider
    fails and before the next one is tried. Use it for logging or metrics.
    """

    def __init__(
        self,
        providers: list[tuple[str, Provider]],
        *,
        should_fall_back: ShouldFallBack | None = None,
        on_fallback: OnFallback | None = None,
    ) -> None:
        if not providers:
            raise ValueError("providers must be a non-empty list")
        # shallow copy so caller mutations don't change the chain
        self._providers: list[tuple[str, Provider]] = list(providers)
        self._should_fall_back: ShouldFallBack = (
            should_fall_back if should_fall_back is not None else _default_should_fall_back
        )
        self._on_fallback: OnFallback | None = on_fallback

    @property
    def names(self) -> list[str]:
        """Provider names in order."""
        return [n for n, _ in self._providers]

    # ---- sync ----

    def call(self, *args: Any, **kwargs: Any) -> ChainResult:
        """Try each provider in order until one returns.

        If a provider is async (returns a coroutine), it is run via
        `asyncio.run` so this method stays synchronous. Prefer `call_async`
        if you are already inside an event loop.
        """
        failures: list[Attempt] = []
        last = len(self._providers) - 1
        for i, (name, fn) in enumerate(self._providers):
            start = time.perf_counter()
            try:
                result = fn(*args, **kwargs)
                if inspect.isawaitable(result):
                    # support async providers from a sync caller
                    result = asyncio.run(_await(result))
                return ChainResult(value=result, provider=name, attempts=failures)
            except Exception as exc:  # noqa: BLE001 - we re-raise non-fallback below
                elapsed = (time.perf_counter() - start) * 1000.0
                attempt = Attempt(name=name, exception=exc, duration_ms=elapsed)
                if not self._should_fall_back(exc):
                    # predicate said don't fall back. raise as-is.
                    raise
                failures.append(attempt)
                if i < last and self._on_fallback is not None:
                    next_name = self._providers[i + 1][0]
                    self._on_fallback(name, exc, next_name)
        raise AllProvidersFailedError(failures)

    # ---- async ----

    async def call_async(self, *args: Any, **kwargs: Any) -> ChainResult:
        """Async variant of `call`. Awaits coroutine providers; sync providers
        run inline (no thread offload)."""
        failures: list[Attempt] = []
        last = len(self._providers) - 1
        for i, (name, fn) in enumerate(self._providers):
            start = time.perf_counter()
            try:
                result = fn(*args, **kwargs)
                if inspect.isawaitable(result):
                    result = await result
                return ChainResult(value=result, provider=name, attempts=failures)
            except Exception as exc:  # noqa: BLE001 - we re-raise non-fallback below
                elapsed = (time.perf_counter() - start) * 1000.0
                attempt = Attempt(name=name, exception=exc, duration_ms=elapsed)
                if not self._should_fall_back(exc):
                    raise
                failures.append(attempt)
                if i < last and self._on_fallback is not None:
                    next_name = self._providers[i + 1][0]
                    self._on_fallback(name, exc, next_name)
        raise AllProvidersFailedError(failures)


async def _await(awaitable: Awaitable[Any]) -> Any:
    """Tiny shim so asyncio.run can drive an arbitrary awaitable."""
    return await awaitable
