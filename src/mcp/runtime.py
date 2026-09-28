# MCP Async Runtime
# Provides a single shared asyncio event loop running on a background thread.
#
# MCP SDK sessions (ClientSession, stdio/sse/http transports) are bound to the
# event loop they were created in. Gradio UI callbacks are synchronous, so they
# must run all MCP coroutines on ONE persistent loop. Creating a new loop per
# call (as a naive asyncio.run would) tears down the session's transports.

import asyncio
import atexit
import threading
from concurrent.futures import Future
from typing import Any, Coroutine, Optional

from ai_companion_core import logger


class _MCPRuntime:
    """Owns a dedicated asyncio loop on a daemon thread."""

    def __init__(self):
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._thread: Optional[threading.Thread] = None
        self._ready = threading.Event()
        self._lock = threading.Lock()

    def _ensure_started(self):
        if self._loop is not None and self._loop.is_running():
            return
        with self._lock:
            if self._loop is not None and self._loop.is_running():
                return

            self._ready.clear()

            def _run():
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                self._loop = loop
                self._ready.set()
                loop.run_forever()

            self._thread = threading.Thread(target=_run, name="mcp-async-runtime", daemon=True)
            self._thread.start()

        # Wait for the loop to be ready
        self._ready.wait(timeout=10)
        if self._loop is None:
            raise RuntimeError("MCP async runtime failed to start")

    @property
    def loop(self) -> asyncio.AbstractEventLoop:
        self._ensure_started()
        assert self._loop is not None
        return self._loop

    def run(self, coro: Coroutine, timeout: Optional[float] = None) -> Any:
        """
        Run a coroutine on the shared loop and block until it completes.

        Safe to call from any (non-runtime) thread, including Gradio callbacks
        and the main thread.
        """
        loop = self.loop

        # If we're already inside the runtime loop, run inline via a task so we
        # don't deadlock waiting on ourselves.
        try:
            running = asyncio.get_running_loop()
        except RuntimeError:
            running = None

        if running is loop:
            return loop.create_task(coro)

        future: Future = asyncio.run_coroutine_threadsafe(coro, loop)
        return future.result(timeout=timeout)

    def submit(self, coro: Coroutine) -> Future:
        """Schedule a coroutine on the shared loop without blocking."""
        return asyncio.run_coroutine_threadsafe(coro, self.loop)

    def shutdown(self):
        loop = self._loop
        if loop is None:
            return
        try:
            loop.call_soon_threadsafe(loop.stop)
        except Exception:
            pass
        self._loop = None


_runtime: Optional[_MCPRuntime] = None
_runtime_lock = threading.Lock()


def get_mcp_runtime() -> _MCPRuntime:
    """Get the process-wide MCP async runtime."""
    global _runtime
    if _runtime is None:
        with _runtime_lock:
            if _runtime is None:
                _runtime = _MCPRuntime()
    return _runtime


def run_mcp_coro(coro: Coroutine, timeout: Optional[float] = None) -> Any:
    """Convenience wrapper to run an MCP coroutine on the shared loop."""
    return get_mcp_runtime().run(coro, timeout=timeout)


@atexit.register
def _shutdown_runtime():
    if _runtime is not None:
        _runtime.shutdown()
