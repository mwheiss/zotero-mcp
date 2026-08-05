"""Tests for the bounded Zotero API lock (with_zotero_api_lock).

Regression coverage for the wedge where one stuck op holding the process-global
RLock made every other tool block until FastMCP's ~60s client timeout, surfacing
as an opaque "-32001 Request timed out" on reads too. The lock now acquires with
a bounded wait and raises ZoteroApiBusyError for waiters instead of hanging.
"""

import multiprocessing
import os
import threading
import time

import pytest
from conftest import DummyContext

from zotero_mcp import client as _client
from zotero_mcp.client import (
    ZoteroApiBusyError,
    _SerializedCallProxy,
    _zotero_api_lock,
    with_zotero_api_lock,
)


def _hold_process_lock(lock_path, ready, release):
    os.environ["ZOTERO_MCP_API_LOCK_PATH"] = lock_path
    os.environ["ZOTERO_MCP_LOCK_TIMEOUT"] = "5"

    @with_zotero_api_lock
    def hold():
        ready.set()
        release.wait(timeout=5)

    hold()


def _try_process_lock(lock_path, result):
    os.environ["ZOTERO_MCP_API_LOCK_PATH"] = lock_path
    os.environ["ZOTERO_MCP_LOCK_TIMEOUT"] = "0.3"

    @with_zotero_api_lock
    def attempt():
        return "ran"

    try:
        result.put(attempt())
    except ZoteroApiBusyError:
        result.put("busy")


def test_uncontended_call_runs(monkeypatch):
    """With no contention the wrapped function runs and returns normally."""
    monkeypatch.setenv("ZOTERO_MCP_LOCK_TIMEOUT", "5")

    @with_zotero_api_lock
    def f(x):
        return x + 1

    assert f(41) == 42


def test_reentrant_same_thread(monkeypatch):
    """Nested decorated calls on one thread acquire the reentrant lock instantly."""
    monkeypatch.setenv("ZOTERO_MCP_LOCK_TIMEOUT", "5")

    @with_zotero_api_lock
    def inner():
        return "inner-ok"

    @with_zotero_api_lock
    def outer():
        # Simulates add_by_url -> add_by_doi: a decorated call within a decorated call.
        return inner()

    assert outer() == "inner-ok"


def test_reentrant_call_holds_one_file_lock(monkeypatch, tmp_path):
    monkeypatch.setenv("ZOTERO_MCP_API_LOCK_PATH", str(tmp_path / "api.lock"))
    acquired = []
    released = []
    real_acquire = _client.acquire_file_lock
    real_release = _client.release_file_lock

    def acquire(*args, **kwargs):
        acquired.append(1)
        return real_acquire(*args, **kwargs)

    def release(lock_file):
        released.append(1)
        return real_release(lock_file)

    monkeypatch.setattr(_client, "acquire_file_lock", acquire)
    monkeypatch.setattr(_client, "release_file_lock", release)

    @with_zotero_api_lock
    def inner():
        return "inner"

    @with_zotero_api_lock
    def outer():
        return inner()

    assert outer() == "inner"
    assert len(acquired) == 1
    assert len(released) == 1


def test_api_lock_serializes_separate_processes(tmp_path):
    context = multiprocessing.get_context("spawn")
    ready = context.Event()
    release = context.Event()
    result = context.Queue()
    lock_path = str(tmp_path / "api.lock")
    holder = context.Process(
        target=_hold_process_lock,
        args=(lock_path, ready, release),
    )
    waiter = context.Process(
        target=_try_process_lock,
        args=(lock_path, result),
    )

    holder.start()
    try:
        assert ready.wait(timeout=5), "holder process never acquired the API lock"
        waiter.start()
        waiter.join(timeout=5)
        assert not waiter.is_alive()
        assert result.get(timeout=1) == "busy"
    finally:
        release.set()
        holder.join(timeout=5)
        if waiter.is_alive():
            waiter.terminate()
            waiter.join(timeout=2)
        if holder.is_alive():
            holder.terminate()
            holder.join(timeout=2)


def test_waiter_fails_fast_when_lock_held(monkeypatch):
    """A second thread gives up with ZoteroApiBusyError instead of hanging."""
    monkeypatch.setenv("ZOTERO_MCP_LOCK_TIMEOUT", "0.3")

    holder_has_lock = threading.Event()
    release_holder = threading.Event()

    def holder():
        with _zotero_api_lock:
            holder_has_lock.set()
            # Hold past the waiter's bounded acquire window.
            release_holder.wait(timeout=5)

    t = threading.Thread(target=holder)
    t.start()
    assert holder_has_lock.wait(timeout=2), "holder never acquired the lock"

    @with_zotero_api_lock
    def blocked():
        return "should-not-run"

    start = time.monotonic()
    with pytest.raises(ZoteroApiBusyError):
        blocked()
    elapsed = time.monotonic() - start

    # Failed fast around the 0.3s bound, nowhere near a 60s client timeout.
    assert elapsed < 3, f"waiter blocked too long: {elapsed:.1f}s"

    release_holder.set()
    t.join(timeout=5)


def test_lock_released_after_busy_error(monkeypatch):
    """A timed-out acquire must NOT leave the lock held by the waiter."""
    monkeypatch.setenv("ZOTERO_MCP_LOCK_TIMEOUT", "0.3")

    holder_has_lock = threading.Event()
    release_holder = threading.Event()

    def holder():
        with _zotero_api_lock:
            holder_has_lock.set()
            release_holder.wait(timeout=5)

    t = threading.Thread(target=holder)
    t.start()
    assert holder_has_lock.wait(timeout=2)

    @with_zotero_api_lock
    def blocked():
        return "x"

    with pytest.raises(ZoteroApiBusyError):
        blocked()

    # Let the holder release; the lock must now be fully free.
    release_holder.set()
    t.join(timeout=5)

    @with_zotero_api_lock
    def after():
        return "after-ok"

    assert after() == "after-ok"


def test_zero_timeout_opt_out_blocks_until_free(monkeypatch):
    """ZOTERO_MCP_LOCK_TIMEOUT<=0 restores unbounded behaviour (waits, no raise)."""
    monkeypatch.setenv("ZOTERO_MCP_LOCK_TIMEOUT", "0")

    holder_has_lock = threading.Event()
    release_holder = threading.Event()

    def holder():
        with _zotero_api_lock:
            holder_has_lock.set()
            release_holder.wait(timeout=5)

    t = threading.Thread(target=holder)
    t.start()
    assert holder_has_lock.wait(timeout=2)

    result = {}

    @with_zotero_api_lock
    def waiter():
        result["ran"] = True
        return "ran"

    w = threading.Thread(target=lambda: result.setdefault("rv", waiter()))
    w.start()

    # Briefly: waiter should still be blocked (no ZoteroApiBusyError raised).
    time.sleep(0.5)
    assert "ran" not in result, "opt-out should block, not fail fast"

    release_holder.set()
    w.join(timeout=5)
    t.join(timeout=5)
    assert result.get("ran") is True


def test_client_proxy_serializes_direct_calls(monkeypatch):
    """Client methods are serialized even when a tool omits the decorator."""
    monkeypatch.setenv("ZOTERO_MCP_LOCK_TIMEOUT", "5")
    state_lock = threading.Lock()

    class FakeClient:
        def __init__(self):
            self.active = 0
            self.max_active = 0

        def item(self, key):
            with state_lock:
                self.active += 1
                self.max_active = max(self.max_active, self.active)
            time.sleep(0.05)
            with state_lock:
                self.active -= 1
            return key

    raw = FakeClient()
    client = _SerializedCallProxy(raw)
    threads = [threading.Thread(target=client.item, args=(str(i),)) for i in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=2)

    assert raw.max_active == 1


def test_long_semantic_update_does_not_hold_zotero_api_lock(monkeypatch):
    from zotero_mcp import semantic_search
    from zotero_mcp.tools.search import update_search_database

    monkeypatch.setenv("ZOTERO_MCP_LOCK_TIMEOUT", "0.1")
    started = threading.Event()
    release = threading.Event()

    class Search:
        def update_database(self, **kwargs):
            started.set()
            release.wait(timeout=2)
            return {"total_items": 0, "processed_items": 0}

    monkeypatch.setattr(
        semantic_search,
        "create_semantic_search",
        lambda *args, **kwargs: Search(),
    )
    worker = threading.Thread(
        target=lambda: update_search_database(ctx=DummyContext())
    )
    worker.start()
    assert started.wait(timeout=1)

    class Client:
        def item(self, key):
            return key

    try:
        assert _SerializedCallProxy(Client()).item("ABCD1234") == "ABCD1234"
    finally:
        release.set()
        worker.join(timeout=2)

    assert not worker.is_alive()


def test_library_overrides_are_session_scoped(monkeypatch):
    monkeypatch.setenv("ZOTERO_LIBRARY_ID", "default-id")
    monkeypatch.setenv("ZOTERO_LIBRARY_TYPE", "user")

    _client.set_active_library("group-a", "group", session_id="session-a")
    _client.set_active_library("feed-b", "feed", session_id="session-b")
    try:
        assert _client.get_active_library(session_id="session-a") == {
            "library_id": "group-a",
            "library_type": "group",
        }
        assert _client.get_active_library(session_id="session-b") == {
            "library_id": "feed-b",
            "library_type": "feed",
        }
        assert _client.get_active_library(session_id="unrelated") == {}
    finally:
        _client.clear_active_library(session_id="session-a")
        _client.clear_active_library(session_id="session-b")
