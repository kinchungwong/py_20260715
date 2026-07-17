from typing import Final
from damped_sine_args import DampedSineArgsCt
from damped_sine_args_cvt import to_dt_args
from damped_sine_cache_keys import DampedSineCacheKeys

class DampedSineState:
    """Generative state of a damped sine wave, used by DampedSineCache.

    - Organized into three sections:
        - DampedSineArgs
            - Immutable
            - Mutable
        - DampedSineCacheKeys, to be injected by DampedSineCache.

    Args, DampedSineArgs, Immutable:
        - samp_rate: Sample rate in Hz.
        - freq: Frequency in Hz.
        - tau: Time constant of the decay in seconds.
        - omega: Discrete time angular frequency in radians per sample.
        - log_decay: Logarithm of the discrete time decay factor per sample.

    Args, DampedSineArgs, Mutable:
        - phi0: Initial or next phase in radians.
        - amp0: Initial or next amplitude.

    Args, DampedSineCacheKeys, to be injected by DampedSineCache.
        - cache_keys
            - non_decaying: True if tau is infinite, False otherwise.
            - freq_entry: FreqEntry if a close match is found, otherwise None.
            - decay_entry: DecayEntry if a close match is found, otherwise None.
            - combo_entry: ComboEntry if a close match is found, otherwise None.

    The continuous-time representation is given by:
        out(t) = sin(2 * pi * freq * t + phi0) * amp0 * exp(-t / tau)
    The discrete-time representation is given by:
        out[n] = sin(phi0 + omega * n) * amp0 * exp(log_decay * n)
    where:
        omega = 2 * pi * freq / samp_rate
        log_decay = -1 / (tau * samp_rate)

    Valid ranges and constraints for the parameters are:
        samp_rate: (samp_rate > 0.0)
        freq: (0.0 < freq < samp_rate / 2.0)
        omega: (0.0 < omega < pi)
        tau: (0.0 < tau <= inf)
        decay: (0.0 < decay <= 1.0)
    
    Parameter constraints are normally enforced by DampedSineArgsCt
    and DampedSineArgsDt, therefore not enforced here.
    """

    samp_rate: Final[float]
    freq: Final[float]
    tau: Final[float]
    omega: Final[float]
    log_decay: Final[float]
    phi0: float
    amp0: float
    cache_keys: DampedSineCacheKeys | None

    def __init__(self, args: DampedSineArgsCt) -> None:
        """Initialize the damped sine state from continuous-time arguments."""
        self.samp_rate = args.samp_rate
        self.freq = args.freq
        self.tau = args.tau
        args_dt = to_dt_args(args)
        self.omega = args_dt.omega
        self.log_decay = args_dt.log_decay
        self.phi0 = args.phi0
        self.amp0 = args.amp0
        self.cache_keys = None
