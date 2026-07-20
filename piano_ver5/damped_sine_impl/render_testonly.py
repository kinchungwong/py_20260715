"""Reference implementations of rendering functions for testing purposes.
"""

import math
import numpy as np
from .data_model import PartialOne, StateOne


def render_py_1_1(idx: int, partial: PartialOne, state: StateOne) -> float:
    """Renders a single sample at the given index, based on the parameters
    and states of one partial.

    This method is provided solely for reference and testing purposes.
    """
    exp_decay = math.exp(partial.log_decay * idx)
    sinphi = math.sin(partial.omega * idx + state.next_phi)
    return state.next_amp * exp_decay * sinphi


def render_py_1_n(nsamps: int, partial: PartialOne, state: StateOne, out: np.ndarray) -> None:
    """Renders the specified number of samples, based on the parameters
    and states of one partial, without using NumPy.

    This method is provided solely for reference and testing purposes.

    For analytical precision, internal calculations are performed in
    Python floats (64-bit). It is recommended that the output buffer
    uses `dtype` `np.float64`.
    """
    for n in range(nsamps):
        exp_decay = math.exp(partial.log_decay * n)
        sinphi = math.sin(partial.omega * n + state.next_phi)
        out[n] = state.next_amp * exp_decay * sinphi


def render_np_1_n(nsamps: int, partial: PartialOne, state: StateOne, out: np.ndarray) -> None:
    """Renders the specified number of samples, based on the parameters
    and states of one partial, vectorized with NumPy.

    This method is provided solely for reference and testing purposes.

    This implementation does not use any buffer pre-computations or
    pools.

    For analytical precision, internal calculations are performed in
    `np.float64`.
    """
    iota = np.arange(nsamps, dtype=np.float64)
    exp_decay = np.exp(partial.log_decay * iota)
    sinphi = np.sin(partial.omega * iota + state.next_phi)
    out[:] = state.next_amp * exp_decay * sinphi
