import asyncio

import pytest

from llm_fallback_chain import (
    AllProvidersFailedError,
    Attempt,
    ChainResult,
    FallbackChain,
)


class RateLimited(Exception):
    pass


class Validation(Exception):
    pass


# ---------- construction ----------


def test_empty_providers_rejected():
    with pytest.raises(ValueError):
        FallbackChain([])


def test_names_property_preserves_order():
    chain = FallbackChain(
        [
            ("anthropic", lambda: "a"),
            ("openai", lambda: "b"),
            ("gemini", lambda: "c"),
        ]
    )
    assert chain.names == ["anthropic", "openai", "gemini"]


def test_caller_mutating_list_does_not_change_chain():
    providers: list[tuple[str, object]] = [
        ("anthropic", lambda: "a"),
        ("openai", lambda: "b"),
    ]
    chain = FallbackChain(providers)
    providers.append(("gemini", lambda: "c"))
    assert chain.names == ["anthropic", "openai"]


# ---------- sync: success paths ----------


def test_first_provider_succeeds():
    chain = FallbackChain(
        [
            ("anthropic", lambda prompt: f"a:{prompt}"),
            ("openai", lambda prompt: f"o:{prompt}"),
        ]
    )
    result = chain.call("hi")
    assert isinstance(result, ChainResult)
    assert result.value == "a:hi"
    assert result.provider == "anthropic"
    assert result.attempts == []


def test_second_provider_succeeds_after_first_fails():
    def boom(prompt):
        raise RateLimited("anthropic out of quota")

    chain = FallbackChain(
        [
            ("anthropic", boom),
            ("openai", lambda prompt: f"o:{prompt}"),
        ]
    )
    result = chain.call("hi")
    assert result.value == "o:hi"
    assert result.provider == "openai"
    assert len(result.attempts) == 1
    assert result.attempts[0].name == "anthropic"
    assert isinstance(result.attempts[0].exception, RateLimited)
    assert result.attempts[0].duration_ms >= 0.0


def test_third_provider_succeeds_after_two_failures():
    chain = FallbackChain(
        [
            ("a", lambda: (_ for _ in ()).throw(RuntimeError("a"))),
            ("b", lambda: (_ for _ in ()).throw(RuntimeError("b"))),
            ("c", lambda: "ok"),
        ]
    )
    result = chain.call()
    assert result.value == "ok"
    assert result.provider == "c"
    assert [a.name for a in result.attempts] == ["a", "b"]


def test_args_and_kwargs_are_forwarded():
    seen: dict = {}

    def provider(a, b, *, mode):
        seen["a"] = a
        seen["b"] = b
        seen["mode"] = mode
        return a + b

    chain = FallbackChain([("p", provider)])
    result = chain.call(1, 2, mode="x")
    assert result.value == 3
    assert seen == {"a": 1, "b": 2, "mode": "x"}


# ---------- sync: failure paths ----------


def test_all_providers_fail_raises_aggregate():
    def boom_a():
        raise RateLimited("a down")

    def boom_b():
        raise RateLimited("b down")

    chain = FallbackChain([("a", boom_a), ("b", boom_b)])
    with pytest.raises(AllProvidersFailedError) as exc:
        chain.call()
    err = exc.value
    assert len(err.attempts) == 2
    assert err.attempts[0].name == "a"
    assert err.attempts[1].name == "b"
    assert all(isinstance(a, Attempt) for a in err.attempts)
    assert all(isinstance(a.exception, RateLimited) for a in err.attempts)


def test_custom_predicate_skips_non_retryable():
    """A Validation error should re-raise immediately, no fallback."""
    called = {"openai": 0}

    def anthropic():
        raise Validation("bad input")

    def openai():
        called["openai"] += 1
        return "should not happen"

    chain = FallbackChain(
        [("anthropic", anthropic), ("openai", openai)],
        should_fall_back=lambda exc: isinstance(exc, RateLimited),
    )
    with pytest.raises(Validation):
        chain.call()
    assert called["openai"] == 0


def test_custom_predicate_allows_fallback_on_whitelisted():
    chain = FallbackChain(
        [
            ("a", lambda: (_ for _ in ()).throw(RateLimited("retry me"))),
            ("b", lambda: "ok"),
        ],
        should_fall_back=lambda exc: isinstance(exc, RateLimited),
    )
    result = chain.call()
    assert result.provider == "b"


# ---------- callback ----------


def test_on_fallback_callback_fires_for_each_fallback():
    calls: list[tuple[str, str, type]] = []

    def cb(failed_name, exc, next_name):
        calls.append((failed_name, next_name, type(exc)))

    chain = FallbackChain(
        [
            ("a", lambda: (_ for _ in ()).throw(RateLimited("a"))),
            ("b", lambda: (_ for _ in ()).throw(RateLimited("b"))),
            ("c", lambda: "ok"),
        ],
        on_fallback=cb,
    )
    result = chain.call()
    assert result.provider == "c"
    assert calls == [
        ("a", "b", RateLimited),
        ("b", "c", RateLimited),
    ]


def test_on_fallback_not_called_when_first_succeeds():
    calls = []
    chain = FallbackChain(
        [("a", lambda: "ok"), ("b", lambda: "x")],
        on_fallback=lambda n, e, nxt: calls.append(n),
    )
    chain.call()
    assert calls == []


def test_on_fallback_not_called_after_last_provider():
    """When the LAST provider fails, on_fallback should not fire for it
    because there is no next provider to fall back to."""
    calls = []
    chain = FallbackChain(
        [
            ("a", lambda: (_ for _ in ()).throw(RuntimeError("a"))),
            ("b", lambda: (_ for _ in ()).throw(RuntimeError("b"))),
        ],
        on_fallback=lambda failed, exc, nxt: calls.append((failed, nxt)),
    )
    with pytest.raises(AllProvidersFailedError):
        chain.call()
    # only one callback: a -> b. b has no successor.
    assert calls == [("a", "b")]


# ---------- async ----------


async def test_async_first_succeeds():
    async def a():
        return "a"

    chain = FallbackChain([("anthropic", a)])
    result = await chain.call_async()
    assert result.value == "a"
    assert result.provider == "anthropic"


async def test_async_second_succeeds_after_first_fails():
    async def a():
        raise RateLimited("nope")

    async def b():
        return "b"

    chain = FallbackChain([("a", a), ("b", b)])
    result = await chain.call_async()
    assert result.value == "b"
    assert result.provider == "b"
    assert len(result.attempts) == 1


async def test_async_all_fail_raises():
    async def a():
        raise RateLimited("a")

    async def b():
        raise RateLimited("b")

    chain = FallbackChain([("a", a), ("b", b)])
    with pytest.raises(AllProvidersFailedError):
        await chain.call_async()


async def test_async_callback_fires():
    calls = []

    async def a():
        raise RateLimited("a")

    async def b():
        return "b"

    chain = FallbackChain(
        [("a", a), ("b", b)],
        on_fallback=lambda failed, exc, nxt: calls.append((failed, nxt)),
    )
    await chain.call_async()
    assert calls == [("a", "b")]


async def test_async_custom_predicate_skips_non_retryable():
    async def a():
        raise Validation("bad")

    async def b():
        return "b"

    chain = FallbackChain(
        [("a", a), ("b", b)],
        should_fall_back=lambda exc: isinstance(exc, RateLimited),
    )
    with pytest.raises(Validation):
        await chain.call_async()


# ---------- mixed sync/async ----------


async def test_mixed_sync_first_async_fallback():
    def a():
        raise RateLimited("a")

    async def b():
        return "b"

    chain = FallbackChain([("a", a), ("b", b)])
    result = await chain.call_async()
    assert result.value == "b"


async def test_mixed_async_first_sync_fallback():
    async def a():
        raise RateLimited("a")

    def b():
        return "b"

    chain = FallbackChain([("a", a), ("b", b)])
    result = await chain.call_async()
    assert result.value == "b"


def test_sync_call_supports_async_provider_via_asyncio_run():
    async def a():
        await asyncio.sleep(0)
        return "ok"

    chain = FallbackChain([("a", a)])
    result = chain.call()
    assert result.value == "ok"
    assert result.provider == "a"


# ---------- introspection / wiring ----------


def test_attempt_records_duration():
    import time as _time

    def slow():
        _time.sleep(0.01)
        raise RateLimited("slow then fail")

    def fast():
        return "ok"

    chain = FallbackChain([("slow", slow), ("fast", fast)])
    result = chain.call()
    assert result.attempts[0].duration_ms >= 5.0  # generous floor for CI


def test_all_providers_failed_error_message_lists_names():
    chain = FallbackChain(
        [
            ("anthropic", lambda: (_ for _ in ()).throw(RuntimeError("x"))),
            ("openai", lambda: (_ for _ in ()).throw(RuntimeError("y"))),
        ]
    )
    with pytest.raises(AllProvidersFailedError) as exc:
        chain.call()
    msg = str(exc.value)
    assert "anthropic" in msg
    assert "openai" in msg


def test_chain_is_reusable():
    """Calling the same chain twice independently should work; trace from
    a previous call should not leak into the next."""
    counter = {"n": 0}

    def a():
        counter["n"] += 1
        if counter["n"] == 1:
            raise RateLimited("first call only")
        return "a"

    def b():
        return "b"

    chain = FallbackChain([("a", a), ("b", b)])
    r1 = chain.call()
    assert r1.provider == "b"
    assert len(r1.attempts) == 1
    r2 = chain.call()
    assert r2.provider == "a"
    assert r2.attempts == []
