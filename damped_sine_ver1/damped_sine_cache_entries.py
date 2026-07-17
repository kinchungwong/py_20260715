from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True, eq=False)
class ComboEntry:
    """Cached combo of sine and cosine with exponential decay.
    """
    freq_index: int
    tau_index: int
    freq: float
    omega: float
    tau: float
    log_decay: float
    sinexp: np.ndarray
    cosexp: np.ndarray


@dataclass(frozen=True, eq=False)
class FreqEntry:
    """Cached sine and cosine for a given frequency.
    """
    freq_index: int
    freq: float
    omega: float
    sinpart: np.ndarray
    cospart: np.ndarray


@dataclass(frozen=True, eq=False)
class DecayEntry:
    """Cached exponential decay for a given decay factor.
    """
    tau_index: int
    tau: float
    log_decay: float
    exppart: np.ndarray
