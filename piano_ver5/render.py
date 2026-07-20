import math
import numpy as np

class Partial:
    omega: float
    log_decay: float

class PartialNp:
    omega: np.ndarray
    log_decay: np.ndarray

class PartialState:
    next_amp: float
    next_phi: float

class PartialStateNp:
    next_amp: np.ndarray
    next_phi: np.ndarray

def render_1_1(idx: int, partial: Partial, state: PartialState) -> float:
    """Renders a single sample of the partial at the given index, based on the partial state."""
    exp_decay = math.exp(partial.log_decay * idx)
    sinphi = math.sin(partial.omega * idx + state.next_phi)
    return state.next_amp * exp_decay * sinphi

def render_1_n_py(nsamps: int, partial: Partial, state: PartialState, out: np.ndarray) -> None:
    for n in range(nsamps):
        exp_decay = math.exp(partial.log_decay * n)
        sinphi = math.sin(partial.omega * n + state.next_phi)
        out[n] = state.next_amp * exp_decay * sinphi

def render_1_n_np(nsamps: int, partial: Partial, state: PartialState, out: np.ndarray) -> None:
    iota = np.arange(nsamps, dtype=np.float64)
    exp_decay = np.exp(partial.log_decay * iota)
    sinphi = np.sin(partial.omega * iota + state.next_phi)
    out[:] = state.next_amp * exp_decay * sinphi

class Render_1_N_NpBuf:
    def __init__(self, nsamps: int, dtype) -> None:
        self._iota = np.arange(nsamps, dtype=dtype)
        self._buf = np.empty((nsamps,), dtype=dtype)

    def render(self, partial: Partial, state: PartialState, out: np.ndarray) -> None:
        buf = self._buf
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

class Render_1_N_NpCache:
    def __init__(self, nsamps: int, partial: Partial, dtype) -> None:
        self._partial = partial
        self._buf = np.empty((nsamps,), dtype=dtype)
        iota = np.arange(nsamps, dtype=dtype)
        self._expsin = np.exp(partial.log_decay * iota) * np.sin(partial.omega * iota)
        self._expcos = np.exp(partial.log_decay * iota) * np.cos(partial.omega * iota)

    def render(self, state: PartialState, out: np.ndarray) -> None:
        coef_expsin = state.next_amp * np.cos(state.next_phi)
        coef_expcos = state.next_amp * np.sin(state.next_phi)
        buf = self._buf
        np.multiply(self._expsin, coef_expsin, out=out)
        np.multiply(self._expcos, coef_expcos, out=buf)
        np.add(out, buf, out=out)

class Render_M_N_NpCache:
    def __init__(self, nsamps: int, partials: PartialNp, dtype) -> None:
        self._partials = partials
        self.num_partials = nump = partials.omega.shape[0]
        self._n_buf = np.empty((nump,), dtype=dtype)
        iota = np.arange(nsamps, dtype=dtype)
        # exp_decay := exp(log_decay * iota) # shape: (M, N)
        exp_decay = np.multiply(partials.log_decay[:, np.newaxis], iota[np.newaxis, :])
        np.exp(exp_decay, out=exp_decay)
        # omega_iota := omega * iota # shape: (M, N)
        omega_iota = np.multiply(partials.omega[:, np.newaxis], iota[np.newaxis, :])
        expsin = exp_decay * np.sin(omega_iota)
        expcos = exp_decay * np.cos(omega_iota)
        # shape: (2*M, N)
        self._stacked = np.concatenate((expsin, expcos), axis=0)

    def render(self, state: PartialStateNp, out: np.ndarray) -> None:
        nump = self.num_partials
        self._n_buf[:nump] = state.next_amp * np.cos(state.next_phi)
        self._n_buf[nump:] = state.next_amp * np.sin(state.next_phi)
        np.matmul(self._n_buf, self._stacked, out=out)
