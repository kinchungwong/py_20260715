from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True, eq=False)
class PartialOne:
    omega: float
    log_decay: float


@dataclass(frozen=True, eq=False)
class PartialM:
    omega: np.ndarray     # shape: (M,)
    log_decay: np.ndarray # shape: (M,)


@dataclass(frozen=False, eq=False)
class StateOne:
    next_amp: float
    next_phi: float


@dataclass(frozen=False, eq=False)
class StateM:
    next_amp: np.ndarray # shape: (M,)
    next_phi: np.ndarray # shape: (M,)
