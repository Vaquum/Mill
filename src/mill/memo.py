from __future__ import annotations

import threading
from collections import OrderedDict
from collections.abc import Callable, Iterator, MutableMapping
from contextlib import contextmanager


class ByteLRU(MutableMapping):
    def __init__(self, budget: int, weight: Callable, max_entries: int = 4096) -> None:
        self.budget = budget
        self.weight = weight
        self.max_entries = max_entries
        self.bytes = 0
        self._items = OrderedDict()
        self._lock = threading.RLock()

    def __getitem__(self, key):
        with self._lock:
            value, size = self._items[key]
            self._items.move_to_end(key)
            return value

    def __setitem__(self, key, value):
        size = max(1, self.weight(value))
        with self._lock:
            old = self._items.pop(key, None)
            if old is not None:
                self.bytes -= old[1]
            if size > self.budget or self.max_entries < 1:
                return
            self._items[key] = (value, size)
            self.bytes += size
            while self.bytes > self.budget or len(self._items) > self.max_entries:
                _, (_, removed) = self._items.popitem(last=False)
                self.bytes -= removed

    def __delitem__(self, key):
        with self._lock:
            _, size = self._items.pop(key)
            self.bytes -= size

    def __iter__(self):
        with self._lock:
            return iter(tuple(self._items))

    def __len__(self):
        with self._lock:
            return len(self._items)

    def pop(self, key, default=None):
        with self._lock:
            item = self._items.pop(key, None)
            if item is None:
                return default
            self.bytes -= item[1]
            return item[0]

    def clear(self):
        with self._lock:
            self._items.clear()
            self.bytes = 0


class Flights:
    def __init__(self) -> None:
        self._items = {}
        self._lock = threading.Lock()

    @contextmanager
    def hold(self, key) -> Iterator[None]:
        with self._lock:
            entry = self._items.setdefault(key, [threading.RLock(), 0])
            entry[1] += 1
        try:
            with entry[0]:
                yield
        finally:
            with self._lock:
                entry[1] -= 1
                if entry[1] == 0:
                    del self._items[key]
