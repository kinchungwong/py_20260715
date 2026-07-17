from dataclasses import dataclass
import math


@dataclass(frozen=True)
class DampedSineArgsCt:
    """Real-time damped sine arguments.
    """
    samp_rate: float
    phi0: float
    freq: float
    amp0: float
    tau: float
    def __post_init__(self):
        if not (self.samp_rate > 0.0):
            raise ValueError(f"samp_rate must be positive, got {self.samp_rate}")
        if not (0.0 < self.freq < 0.5 * self.samp_rate):
            raise ValueError(f"freq must be positive and below Nyquist, got {self.freq}")
        if not (self.tau > 0.0):
            raise ValueError(f"tau must be positive or infinite, got {self.tau}")


@dataclass(frozen=True)
class DampedSineArgsDt:
    """Discrete-time damped sine arguments.
    """
    phi0: float
    omega: float
    amp0: float
    log_decay: float
    def __post_init__(self):
        if not (0.0 < self.omega < math.pi):
            raise ValueError(f"omega must be between 0.0 and pi (exclusive), got {self.omega}")
        if not (self.log_decay <= 0.0):
            raise ValueError(f"log_decay must be non-positive, got {self.log_decay}")
