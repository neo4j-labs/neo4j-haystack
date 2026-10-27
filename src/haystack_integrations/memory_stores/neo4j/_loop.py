"""One private event loop in a background thread, owned by one memory store."""

import asyncio
import threading
from collections.abc import Coroutine
from typing import Any, TypeVar

T = TypeVar("T")


class _LoopRunner:
    """
    Runs coroutines on a private event loop that lives in a daemon thread.

    ``neo4j-agent-memory`` is async-only, and its async Neo4j driver is bound to the loop it connected on.
    Running every library call on this one loop lets the components work from sync code, from
    ``Pipeline.run_async``, and inside an already running loop such as Jupyter's.
    """

    def __init__(self) -> None:
        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()

    def _ensure(self) -> asyncio.AbstractEventLoop:
        with self._lock:
            if self._loop is None or self._loop.is_closed():
                self._loop = asyncio.new_event_loop()
                self._thread = threading.Thread(target=self._loop.run_forever, name="neo4j-agent-memory", daemon=True)
                self._thread.start()
            return self._loop

    def run(self, coro: Coroutine[Any, Any, T], timeout: float | None = None) -> T:
        """Run ``coro`` on the private loop from synchronous code, including inside a running loop."""
        if self._thread is not None and threading.current_thread() is self._thread:
            coro.close()
            msg = "A synchronous memory call from the memory loop's own thread would deadlock."
            raise RuntimeError(msg)
        return asyncio.run_coroutine_threadsafe(coro, self._ensure()).result(timeout)

    async def arun(self, coro: Coroutine[Any, Any, T]) -> T:
        """Await ``coro`` on the private loop from any other event loop, without blocking it."""
        return await asyncio.wrap_future(asyncio.run_coroutine_threadsafe(coro, self._ensure()))

    def stop(self) -> None:
        """Stop the loop and its thread. The next ``run``/``arun`` starts a new one."""
        with self._lock:
            loop, thread = self._loop, self._thread
            self._loop, self._thread = None, None
        if loop is not None:
            loop.call_soon_threadsafe(loop.stop)
            if thread is not None and thread is not threading.current_thread():
                thread.join(timeout=5)
            if not loop.is_running():
                loop.close()
