from abc import ABC, abstractmethod
import numpy as np


class SpareBase(ABC):
    def __init__(self, max_partials_per_note: int, max_nsamps: int, dtype) -> None:
        self._max_m = max_partials_per_note
        self._max_n = max_nsamps
        self._dtype = dtype

    @property
    def max_partials_per_note(self) -> int:
        return self._max_m
    
    @property
    def max_nsamps(self) -> int:
        return self._max_n
    
    @property
    def dtype(self):
        return self._dtype

    @abstractmethod
    def borrow_spare_m(self) -> np.ndarray:
        return np.empty((self._max_m,), dtype=self._dtype)

    @abstractmethod
    def borrow_spare_n(self) -> np.ndarray:
        return np.empty((self._max_n,), dtype=self._dtype)

    @abstractmethod
    def borrow_spare_mn(self) -> np.ndarray:
        return np.empty((self._max_m, self._max_n), dtype=self._dtype)

    @abstractmethod
    def recycle_spare_m(self, buf: np.ndarray) -> None:
        del buf
        pass

    @abstractmethod
    def recycle_spare_n(self, buf: np.ndarray) -> None:
        del buf
        pass

    @abstractmethod
    def recycle_spare_mn(self, buf: np.ndarray) -> None:
        del buf
        pass
