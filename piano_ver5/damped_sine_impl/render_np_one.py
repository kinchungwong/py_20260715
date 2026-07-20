from typing import final, override
import numpy as np

from .data_model import PartialOne, StateOne
from .spare_base import SpareBase
from .render_np_base import RenderNpBase


class RenderNpOne(RenderNpBase[StateOne, PartialOne]):

    def __init__(self, nsamps: int, partial: PartialOne, dtype, spares: SpareBase) -> None:
        super().__init__(nsamps, partial, dtype, spares)

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
        partial = self._partial
        iota = self._iota
        # buf := omega * iota
        np.multiply(iota, partial.omega, out=buf)
        # buf := omega * iota + next_phi
        np.add(buf, state.next_phi, out=buf)
        # out := sin(omega * iota + next_phi)
        np.sin(buf, out=out)
        # buf := log_decay * iota
        np.multiply(iota, partial.log_decay, out=buf)
        # buf := exp(log_decay * iota)
        np.exp(buf, out=buf)
        # buf := next_amp * exp(log_decay * iota)
        np.multiply(buf, state.next_amp, out=buf)
        # out := next_amp * exp(log_decay * iota) * sin(omega * iota + next_phi)
        np.multiply(out, buf, out=out)
