import math
from typing import Final
from damped_sine_cache_entries import ComboEntry, FreqEntry, DecayEntry


class DampedSineCacheKeys:
    non_decaying: Final[bool]
    freq_entry: FreqEntry | None
    decay_entry: DecayEntry | None
    combo_entry: ComboEntry | None

    def __init__(self, tau: float) -> None:
        self.non_decaying = math.isinf(tau) and tau > 0.0
        self.freq_entry = None
        self.decay_entry = None
        self.combo_entry = None
