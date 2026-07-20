import numpy as np
from collections import deque
from typing import override, final
from .spare_base import SpareBase


class SpareMT(SpareBase):
    """Thread-safe version of SpareBase using deque as object pools.

    This implementation allows multiple threads to borrow and recycle spare
    buffers concurrently.

    Allocations only happen on borrow that cannot be satisfied from the pool.
    Thus, if a particular buffer type isn't use (e.g. M-by-N), none of that
    type will be allocated upfront.
    """

    # NOTE on the condition that we only use append, pop, appendleft, popleft
    #      on collections.queue, we do not need to use an extra lock, since
    #      the internal lock (critical section in CPython) suffice.
    # self._lock = Lock()

    def __init__(self, max_partials_per_note: int, max_nsamps: int, dtype) -> None:
        super().__init__(max_partials_per_note, max_nsamps, dtype)
        self._pool_m = deque[np.ndarray]()
        self._pool_n = deque[np.ndarray]()
        self._pool_mn = deque[np.ndarray]()

    @override
    @final
    def borrow_spare_m(self) -> np.ndarray:
        try:
            spare_m = self._pool_m.pop()
            return spare_m
        except IndexError:
            return np.empty((self._max_m,), dtype=self._dtype)

    @override
    @final
    def borrow_spare_n(self) -> np.ndarray:
        try:
            spare_n = self._pool_n.pop()
            return spare_n
        except IndexError:
            return np.empty((self._max_n,), dtype=self._dtype)

    @override
    @final
    def borrow_spare_mn(self) -> np.ndarray:
        try:
            spare_mn = self._pool_mn.pop()
            return spare_mn
        except IndexError:
            return np.empty((self._max_m, self._max_n), dtype=self._dtype)

    @override
    @final
    def recycle_spare_m(self, buf: np.ndarray) -> None:
        assert buf.shape == (self._max_m,)
        assert buf.dtype == self._dtype
        self._pool_m.append(buf)

    @override
    @final
    def recycle_spare_n(self, buf: np.ndarray) -> None:
        assert buf.shape == (self._max_n,)
        assert buf.dtype == self._dtype
        self._pool_n.append(buf)

    @override
    @final
    def recycle_spare_mn(self, buf: np.ndarray) -> None:
        assert buf.shape == (self._max_m, self._max_n)
        assert buf.dtype == self._dtype
        self._pool_mn.append(buf)
