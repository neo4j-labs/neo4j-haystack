"""An embedder wrapper that embeds each distinct text once."""

from __future__ import annotations

import asyncio
from collections import OrderedDict
from typing import Any


class _SharedEmbedder:
    """
    Wraps a ``neo4j-agent-memory`` embedder so identical texts are embedded once and the result is shared.

    The library embeds the query separately in each search (preferences, entities, facts, traces), and the
    writer embeds the query again for the message and the trace task. Sharing turns four embedding API calls per
    read into one, including for the concurrent searches, which await the same in-flight request.

    Runs on the store's single event loop, so no locking is needed. Failed embeddings are not cached.
    """

    def __init__(self, inner: Any, max_items: int = 256) -> None:
        self._inner = inner
        self._max_items = max_items
        self._cache: OrderedDict[str, asyncio.Future[list[float]]] = OrderedDict()

    # Real properties, not only __getattr__: the library's isinstance(..., Embedder) protocol check looks
    # attributes up statically and would not see delegated ones.
    @property
    def dimensions(self) -> int:
        return self._inner.dimensions

    @property
    def model(self) -> Any:
        return getattr(self._inner, "model", None)

    def __getattr__(self, name: str) -> Any:
        # anything else the library may read from its embedder
        return getattr(self._inner, name)

    def _remember(self, text: str, future: asyncio.Future[list[float]]) -> None:
        self._cache[text] = future
        self._cache.move_to_end(text)
        while len(self._cache) > self._max_items:
            self._cache.popitem(last=False)

    def _forget(self, text: str, future: asyncio.Future[list[float]]) -> None:
        if self._cache.get(text) is future:
            del self._cache[text]

    async def embed(self, text: str) -> list[float]:
        future = self._cache.get(text)
        if future is None:
            future = asyncio.ensure_future(self._inner.embed(text))
            self._remember(text, future)
        else:
            self._cache.move_to_end(text)
        try:
            return list(await asyncio.shield(future))
        except Exception:
            self._forget(text, future)
            raise

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        missing = list(dict.fromkeys(t for t in texts if t not in self._cache))
        if missing:
            loop = asyncio.get_running_loop()
            futures = {t: loop.create_future() for t in missing}
            for text, future in futures.items():
                self._remember(text, future)
            try:
                vectors = await self._inner.embed_batch(missing)
            except Exception as error:
                for text, future in futures.items():
                    self._forget(text, future)
                    future.set_exception(error)
                    future.exception()  # mark retrieved: concurrent waiters re-raise it themselves
                raise
            for text, vector in zip(missing, vectors, strict=True):
                futures[text].set_result(vector)
        return [await self.embed(t) for t in texts]
