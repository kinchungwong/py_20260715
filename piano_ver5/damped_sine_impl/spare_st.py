import numpy as np
from typing import override, final
from .spare_base import SpareBase


class SpareST(SpareBase):
    """Single-threaded version of SpareBase using internal spare buffers.

    This implementation assumes that the functions that use buffers are
    non-reentrant, and that only one borrowed buffer of each type
    (M, N, M-by-N) is needed at any given time.
    """

    def __init__(self, max_partials_per_note: int, max_nsamps: int, dtype) -> None:
        super().__init__(max_partials_per_note, max_nsamps, dtype)
        self._spare_m = None
        self._spare_n = None
        self._spare_mn = None
        
    @override
    @final
    def borrow_spare_m(self) -> np.ndarray:
        spare_m, self._spare_m = self._spare_m, None
        if spare_m is not None:
            return spare_m
        return np.empty((self._max_m,), dtype=self._dtype)
        
    @override
    @final
    def borrow_spare_n(self) -> np.ndarray:
        spare_n, self._spare_n = self._spare_n, None
        if spare_n is not None:
            return spare_n
        return np.empty((self._max_n,), dtype=self._dtype)

    @override
    @final
    def borrow_spare_mn(self) -> np.ndarray:
        spare_mn, self._spare_mn = self._spare_mn, None
        if spare_mn is not None:
            return spare_mn
        return np.empty((self._max_m, self._max_n), dtype=self._dtype)

    @override
    @final
    def recycle_spare_m(self, buf: np.ndarray) -> None:
        assert buf.shape == (self._max_m,)
        assert buf.dtype == self._dtype
        self._spare_m = buf

    @override
    @final
    def recycle_spare_n(self, buf: np.ndarray) -> None:
        assert buf.shape == (self._max_n,)
        assert buf.dtype == self._dtype
        self._spare_n = buf

    @override
    @final
    def recycle_spare_mn(self, buf: np.ndarray) -> None:
        assert buf.shape == (self._max_m, self._max_n)
        assert buf.dtype == self._dtype
        self._spare_mn = buf
