"""LRU + TTL cache for reachability rings. Keyed by (rounded origin, minutes)."""

from __future__ import annotations

import time
from collections import OrderedDict
from typing import Callable, Optional, Tuple

from geogenie.core.coords import Origin
from geogenie.core.types import ReachRing


CacheKey = Tuple[float, float, float]  # lon4, lat4, minutes


class RingCache:
    def __init__(self, maxsize: int = 256, ttl_s: float = 3600.0) -> None:
        self.maxsize = maxsize
        self.ttl_s = ttl_s
        self._store: OrderedDict[CacheKey, Tuple[float, ReachRing]] = OrderedDict()
        self.hits = 0
        self.misses = 0

    @staticmethod
    def make_key(origin: Origin, minutes: float) -> CacheKey:
        return (round(origin.lon, 4), round(origin.lat, 4), float(minutes))

    def get(self, origin: Origin, minutes: float) -> Optional[ReachRing]:
        key = self.make_key(origin, minutes)
        item = self._store.get(key)
        if item is None:
            self.misses += 1
            return None
        ts, ring = item
        if time.monotonic() - ts > self.ttl_s:
            del self._store[key]
            self.misses += 1
            return None
        self._store.move_to_end(key)
        self.hits += 1
        return ring

    def put(self, origin: Origin, minutes: float, ring: ReachRing) -> None:
        key = self.make_key(origin, minutes)
        self._store[key] = (time.monotonic(), ring)
        self._store.move_to_end(key)
        while len(self._store) > self.maxsize:
            self._store.popitem(last=False)

    def get_or_build(
        self,
        origin: Origin,
        minutes: float,
        builder: Callable[[Origin, float], ReachRing],
    ) -> ReachRing:
        cached = self.get(origin, minutes)
        if cached is not None:
            return cached
        ring = builder(origin, minutes)
        self.put(origin, minutes, ring)
        return ring

    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total else 0.0
