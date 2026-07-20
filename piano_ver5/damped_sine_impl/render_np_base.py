from abc import ABC, abstractmethod
from typing import Final, Generic, TypeVar
import numpy as np

from .data_model import PartialOne, PartialM, StateOne, StateM
from .spare_base import SpareBase


_StateT = TypeVar("_StateT", StateOne, StateM)
_PartialT = TypeVar("_PartialT", PartialOne, PartialM)


class RenderNpBase(Generic[_StateT, _PartialT], ABC):
    _partial: Final[_PartialT]
    _iota: Final[np.ndarray]
    _spares: Final[SpareBase]


    def __init__(self, nsamps: int, partial: _PartialT, dtype, spares: SpareBase) -> None:
        self._partial = partial
        self._iota = np.arange(nsamps, dtype=dtype)
        self._spares = spares 

    @abstractmethod
    def render(self, state: _StateT, out: np.ndarray) -> None:
        raise NotImplementedError
