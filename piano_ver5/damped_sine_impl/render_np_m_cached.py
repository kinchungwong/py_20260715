from typing import final, override, Final
import numpy as np

from .data_model import PartialM, StateM
from .spare_base import SpareBase
from .render_np_base import RenderNpBase


class RenderNpMCached(RenderNpBase[StateM, PartialM]):
    num_partials: Final[int] # value: M
    _stacked: Final[np.ndarray] # shape: (2*M, N)

    def __init__(self, nsamps: int, partials: PartialM, dtype, spares: SpareBase) -> None:
        super().__init__(nsamps, partials, dtype, spares)
        self.num_partials = partials.omega.shape[0]
        iota = self._iota
        #
        # exp_decay := exp(log_decay * iota) # shape: (M, N)
        #
        exp_decay = np.multiply(partials.log_decay[:, np.newaxis], iota[np.newaxis, :])
        np.exp(exp_decay, out=exp_decay)
        #
        # omega_iota := omega * iota # shape: (M, N)
        #
        omega_iota = np.multiply(partials.omega[:, np.newaxis], iota[np.newaxis, :])
        expsin = exp_decay * np.sin(omega_iota)
        expcos = exp_decay * np.cos(omega_iota)
        #
        # shape: (2*M, N)
        #
        self._stacked = np.concatenate((expsin, expcos), axis=0)

    @override
    @final
    def render(self, state: StateM, out: np.ndarray) -> None:
        # mixvec.shape: (2*M,)
        mixvec = np.concatenate(
            (
                state.next_amp * np.cos(state.next_phi),
                state.next_amp * np.sin(state.next_phi),
            ),
            axis=0,
        )

        # TODO Because of this fix (we need (2M,) instead of (M,)),
        # we can't use SpareBase at all, so we might as well remove it.

        # TODO verify then remove debug prints
        # print("buf_m.shape:", mixvec.shape)
        # print("self._stacked.shape:", self._stacked.shape)

        # self._stacked.shape: (2*M, N)
        np.matmul(mixvec, self._stacked, out=out)

    # TODO remove buggy code once the new code is fully verified.

    # @override
    # @final
    # def render(self, state: StateM, out: np.ndarray) -> None:
    #     buf_m = self._spares.borrow_spare_m()
    #     try:
    #         self._render_2(state, out, buf_m)
    #     finally:
    #         self._spares.recycle_spare_m(buf_m)

    # @final
    # def _render_2(self, state: StateM, out: np.ndarray, buf_m: np.ndarray) -> None:
    #     nump = self.num_partials
    #     buf_m[:nump] = state.next_amp * np.cos(state.next_phi)
    #     buf_m[nump:] = state.next_amp * np.sin(state.next_phi)
    #     print("buf_m.shape:", buf_m.shape)
    #     print("self._stacked.shape:", self._stacked.shape)
    #     np.matmul(buf_m, self._stacked, out=out)
