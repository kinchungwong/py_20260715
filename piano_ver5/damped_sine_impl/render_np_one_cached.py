from typing import final, override, Final
import numpy as np

from .data_model import PartialOne, StateOne
from .spare_base import SpareBase
from .render_np_base import RenderNpBase


class RenderNpOneCached(RenderNpBase[StateOne, PartialOne]):
    _expsin: Final[np.ndarray] # shape: (N,)
    _expcos: Final[np.ndarray] # shape: (N,)

    def __init__(self, nsamps: int, partial: PartialOne, dtype, spares: SpareBase) -> None:
        super().__init__(nsamps, partial, dtype, spares)
        iota = self._iota
        self._expsin = np.exp(partial.log_decay * iota) * np.sin(partial.omega * iota)
        self._expcos = np.exp(partial.log_decay * iota) * np.cos(partial.omega * iota)

    @override
    @final
    def render(self, state: StateOne, out: np.ndarray) -> None:
        buf = self._spares.borrow_spare_n()
        try:
            self._render_2(state, out, buf)
        finally:
            self._spares.recycle_spare_n(buf)

    @final
    def _render_2(self, state: StateOne, out: np.ndarray, buf: np.ndarray) -> None:
        coef_expsin = state.next_amp * np.cos(state.next_phi)
        coef_expcos = state.next_amp * np.sin(state.next_phi)
        np.multiply(self._expsin, coef_expsin, out=out)
        np.multiply(self._expcos, coef_expcos, out=buf)
        np.add(out, buf, out=out)
