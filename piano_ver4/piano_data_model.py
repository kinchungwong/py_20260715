"""Piano data model.
"""

from abc import ABC, abstractmethod
import math
from dataclasses import dataclass
from typing import Any, Callable, Final, Mapping
from enum import Enum


__all__ = [
    "PianoKey",
    "PianoNote",
    "PianoCfg",
    "Partial",
    "PianoModelBase",
    "NotePartialsBase",
    "DampedSine",
    "Trend",
    "LinearRamp",
    "PartialState",
    "NoteState",
]


@dataclass(frozen=True, eq=True)
class PianoKey:
    note_id: int


@dataclass(frozen=True, eq=True)
class PianoNote:
    note_id: int
    velocity: int


class PianoCfg:
    """MIDI Piano Model Args.

    Attributes:
        - sample_rate: Sample rate in Hz, needed for Nyquist.
        - max_partials: Max number of partials a note can have.
        - inharmonicity: Non-linear deviations from harmonic series.
        - amp_pcoef: Coefficient for amplitude decay of partials (see equation)
        - tau_fund: Time constant for the fundamental frequency (in seconds)
        - tau_pcoef: Coefficient for time constant decay of partials (see equation)
        - hammer_pos: Normalized position of the string strike that causes missing partials
        - attack_msecs: Attack time in milliseconds for linear ramp.
        - release_msecs: Release time in milliseconds for linear ramp.
        - note_id_min: Minimum MIDI note ID for an 88-key piano (inclusive)
        - note_id_max: Maximum MIDI note ID for an 88-key piano (inclusive)

    Equations:
        - Partial number: k = 1, 2, 3, ...
        - Inharmonicity: B
        - Partial frequency: f_k = f_1 * k * sqrt(1 + B * k^2)
        - Amplitude rolloff (amp_pcoef): p
        - Partial amplitude: A_k = 1 / (k ^ p)
        - Tau decay (tau_pcoef): q
        - Partial time constant: tau_k = tau_1 / (1 + q * (k - 1))
        - Missing partials: sin(pi * k * strike_pos) ~= 0
            - Alternatively: let value = k * strike_pos, 
                the partial vanishes if abs(value - round(value)) < epsilon.
    """
    sample_rate: float
    audible_hz: float
    max_partials: int
    inharmonicity: float
    amp_pcoef: float
    tau_fund: float
    tau_pcoef: float
    hammer_pos: float
    attack_msecs: float
    release_msecs: float
    note_id_min: int
    note_id_max: int
    audible_amp: float

    def __init__(
            self,
            *,
            sample_rate: float = 44100.0,
            audible_hz: float = 18000.0,
            max_partials: int = 16,
            inharmonicity: float = 4.0e-4,
            amp_pcoef: float = 1.2,
            tau_fund: float = 1.6,
            tau_pcoef: float = 0.25,
            hammer_pos: float = 1.0 / 7.0,
            attack_msecs: float = 6.0,
            release_msecs: float = 20.0,
            note_id_min: int = 21,
            note_id_max: int = 108,
            audible_amp: float = (1.0 / 32768.0),
            ) -> None:
        self.sample_rate = float(sample_rate)
        self.audible_hz = float(audible_hz)
        self.max_partials = int(max_partials)
        self.inharmonicity = float(inharmonicity)
        self.amp_pcoef = float(amp_pcoef)
        self.tau_fund = float(tau_fund)
        self.tau_pcoef = float(tau_pcoef)
        self.hammer_pos = float(hammer_pos)
        self.attack_msecs = float(attack_msecs)
        self.release_msecs = float(release_msecs)
        self.note_id_min = int(note_id_min)
        self.note_id_max = int(note_id_max)
        self.audible_amp = float(audible_amp)
        
    def clone(self) -> "PianoCfg":
        return PianoCfg(
            sample_rate=self.sample_rate,
            audible_hz=self.audible_hz,
            max_partials=self.max_partials,
            inharmonicity=self.inharmonicity,
            amp_pcoef=self.amp_pcoef,
            tau_fund=self.tau_fund,
            tau_pcoef=self.tau_pcoef,
            hammer_pos=self.hammer_pos,
            attack_msecs=self.attack_msecs,
            release_msecs=self.release_msecs,
            note_id_min=self.note_id_min,
            note_id_max=self.note_id_max,
            audible_amp=self.audible_amp
        )


@dataclass(frozen=True)
class Partial:
    note_id: int
    partial_k: int
    p_freq: float
    p_tau: float
    p_amp: float

    def __eq__(self, other: "Partial | Any") -> bool:
        if not isinstance(other, Partial):
            return False
        return self.note_id == other.note_id and self.partial_k == other.partial_k
    
    def __hash__(self) -> int:
        return hash((self.note_id, self.partial_k))



class PianoModelBase(ABC):
    @property
    def cfg(self) -> PianoCfg:
        raise NotImplementedError()
    
    @property
    def keys(self) -> Mapping[int, PianoKey]:
        raise NotImplementedError()

    @abstractmethod
    def _note_to_fundamental(self, note_id: int) -> float:
        raise NotImplementedError()

    @abstractmethod
    def _partial_freq(self, fund_freq: float, k: int) -> float:
        raise NotImplementedError()
    
    @abstractmethod
    def _partial_amp(self, k: int) -> float:
        raise NotImplementedError()
    
    @abstractmethod
    def _partial_tau(self, k: int) -> float:
        raise NotImplementedError()

    @abstractmethod
    def _valid_partial(self, k: int, p_freq: float, p_amp: float) -> bool:
        raise NotImplementedError()


class NotePartialsBase(ABC):
    @property
    def cfg(self) -> PianoCfg:
        raise NotImplementedError()

    @property
    def key(self) -> PianoKey:
        raise NotImplementedError()
    
    @property
    def valid_partials(self) -> tuple[Partial, ...]:
        raise NotImplementedError()


class DampedSine:
    samp_rate: Final[float]
    freq: Final[float]
    tau: Final[float]
    omega: Final[float]
    log_decay: Final[float]
    has_decay: Final[bool]

    def __init__(
            self,
            samp_rate: float,
            freq: float,
            tau: float
            ) -> None:
        if not (samp_rate > 0.0):
            raise ValueError(f"samp_rate must be positive, got {samp_rate}")
        pi = math.pi
        twopi = 2.0 * pi
        nyquist = 0.5 * samp_rate
        if freq is not None and not (0 < freq < nyquist):
            raise ValueError(f"freq must be positive and below Nyquist, got {freq}")
        if tau is not None and not (tau > 0.0):
            raise ValueError(f"tau must be positive or infinite, got {tau}")
        self.samp_rate = float(samp_rate)
        self.freq = float(freq)
        self.tau = float(tau)
        self.omega = twopi * self.freq / self.samp_rate
        self.log_decay = -1.0 / (self.tau * self.samp_rate)
        self.has_decay = self.log_decay < 0.0


class Trend(Enum):
    """Enumeration of possible trends.
    Values:
        - SILENT: The note is silent; no rendering occurs.
        - RISE: The note is rising in level (attack phase).
        - LEVEL: The note is at a constant level (sustain phase).
        - FALL: The note is falling in level (release phase).
        - RETRIG_YIELD: The note is yielding to a retriggered note, 
            falling to silence at the negative of ramp_rise.
            (This accomplishes cross-fade as the new note rises.)
    """
    SILENT = "Silent"
    RISE = "Rise"
    LEVEL = "Level"
    FALL = "Fall"
    RETRIG_YIELD = "RetrigYield"


class LinearRamp:
    samp_rate: Final[float]
    attack_msecs: Final[float]
    release_msecs: Final[float]
    ramp_rise: Final[float]
    ramp_fall: Final[float]

    def __init__(
            self,
            samp_rate: float,
            attack_msecs: float,
            release_msecs: float
            ) -> None:
        if not (samp_rate > 0.0):
            raise ValueError(f"samp_rate must be positive, got {samp_rate}")
        if not (attack_msecs > 0.0):
            raise ValueError(f"attack_msecs must be positive, got {attack_msecs}")
        if not (release_msecs > 0.0):
            raise ValueError(f"release_msecs must be positive, got {release_msecs}")
        self.samp_rate = float(samp_rate)
        self.attack_msecs = float(attack_msecs)
        self.release_msecs = float(release_msecs)
        self.ramp_rise = 1.0 / (self.attack_msecs * 0.001 * self.samp_rate)
        self.ramp_fall = -1.0 / (self.release_msecs * 0.001 * self.samp_rate)


class PartialState:
    """The live rendering state of a single partial.

    Specifically, this class is responsible for tracking the phase and amplitude
    of the DampedSine of a single partial over time.
    """
    cfg_piano: Final[PianoCfg]
    cfg_ds: Final[DampedSine]
    next_phi: float
    next_amp: float

    def __init__(
            self,
            cfg_piano: PianoCfg,
            cfg_ds: DampedSine,
            ) -> None:
        self.cfg_piano = cfg_piano
        self.cfg_ds = cfg_ds
        self.next_phi = 0.0
        self.next_amp = 1.0

    def advance(self, nsamps: int) -> None:
        """Advance the partial state by `nsamps` samples."""
        cfg_ds = self.cfg_ds
        omega = cfg_ds.omega
        log_decay = cfg_ds.log_decay
        self.next_phi = (self.next_phi + omega * nsamps) % (2.0 * math.pi)
        self.next_amp *= math.exp(log_decay * nsamps)


class NoteState:
    """The live rendering state of a single note.

    Specifically, this class is responsible for tracking the linear ramp state
    of a single note over time, and the DampedSine state of each of its partials
    over time.
    """

    """
    Note on re-trigger:
    A realistic way to implement re-triggering is to have the current note
    yield to the new note. Trend.RETRIG_YIELD is designated for this purpose.
    To implement, the current note will be allowed to fall to silence quickly,
    while a new instance of NoteState is created for the new note, which will
    rise to its new max level determined by the new velocity. The new note will
    have newly initialized PartialStates, which improves realisticity by
    ensuring the partials will get new randomized phases. Such implementation
    resolves continuity issues in ways that other approaches cannot.

    In actual implementation, each piano key will be assigned two NoteState
    instances. When a re-trigger happens, the originally active one goes into
    RETRIG_YIELD, while the newly active one goes into RISE. The original
    NoteState will eventually go into SILENT, and remain so until the next
    re-trigger.
    """

    cfg_piano: Final[PianoCfg]
    cfg_lr: Final[LinearRamp]
    trend: Trend
    level: float
    level_min: float
    level_max: float
    partial_states: list[tuple[Partial, PartialState]]

    class _ClsPrivate:
        _rand_uniform_fn: Callable[[], float]
        def __init__(self) -> None:
            import random
            self._rand_uniform_fn = random.random
        def randomized_phase(self) -> float:
            return self._rand_uniform_fn() * 2.0 * math.pi

    _CLS_PRIVATE = _ClsPrivate()

    @classmethod
    def set_random_uniform_callback(cls, rand_uniform_fn: Callable[[], float]) -> None:
        """Set the random uniform callback for generating randomized phases.

        The callback should return a float in the range [0.0, 1.0).
        """
        cls._CLS_PRIVATE._rand_uniform_fn = rand_uniform_fn

    def __init__(
            self,
            cfg_piano: PianoCfg,
            cfg_lr: LinearRamp,
            note_partials: NotePartialsBase,
            ) -> None:
        self.cfg_piano = cfg_piano
        self.cfg_lr = cfg_lr
        self.trend = Trend.SILENT
        self.level = 0.0
        self.level_min = 0.0
        self.level_max = 0.0 # To be determined by attack velocity.
        # Each partial owns its own DampedSine, derived from its own freq/tau.
        samp_rate = cfg_piano.sample_rate
        self.partial_states = [
            (
                partial,
                PartialState(cfg_piano, DampedSine(samp_rate, partial.p_freq, partial.p_tau)),
            )
            for partial in note_partials.valid_partials
        ]
    
    def attack(self, note: PianoNote) -> None:
        """Start the attack phase of the note.

        The level will rise from its current level to the max level determined
        by the note's velocity, with a ramp rate determined by the attack time
        in the configuration.
        """
        self.trend = Trend.RISE
        self.level_min = 0.0
        self.level_max = (max(0, min(127, note.velocity)) / 127) ** 2.0
        for _, state in self.partial_states:
            #
            # NOTE DampedSine initialization upon attack:
            #      Phase is randomized; amplitude is set to 1.0.
            #
            # NOTE Note on "amplitudes".
            # Each related concept has its place:
            #
            # - The normalized relative amplitudes between partials,
            #   `Partial.p_amp`, are baked into the cached waveforms.
            #   However, if the partials are rendered in real-time,
            #   `Partial.p_amp` should be used.
            #
            # - The exponential decay amplitude, `PartialState.next_amp`,
            #   is managed by PartialState (with Damped Sine), because
            #   each partial has its own decay rate, regardless of ramp.
            #
            # - The linear ramp amplitude, `NoteState.level`, exists
            #   to take care of RISE and FALL specifically, and the 
            #   SILENT state exists as a quick check to skip rendering.
            #
            state.next_phi = self._CLS_PRIVATE.randomized_phase()
            state.next_amp = 1.0

    def sustain(self) -> None:
        """Start the sustain phase of the note.

        The level is held constant at the current level.

        Note: Sustain simply means it is neither rising nor falling, as a result
        of having reached the target level. Exponential decay continues.
        """
        self.trend = Trend.LEVEL
        self.level_min = self.level
        self.level_max = self.level

    def release(self) -> None:
        """Start the release phase of the note.

        The level will fall to zero over the release time, with a negative
        ramp rate determined by the release time in the configuration.
        """
        self.trend = Trend.FALL
        self.level_min = 0.0
        self.level_max = self.level

    def silent(self) -> None:
        """Puts the note into silence immediately, without regard to continuity.

        The level is set to zero and all partials are silenced.
        """
        self.trend = Trend.SILENT
        self.level = 0.0
        self.level_min = 0.0
        self.level_max = 0.0
        for _, state in self.partial_states:
            state.next_phi = 0.0
            state.next_amp = 0.0

    def retrigger_yield(self) -> None:
        raise NotImplementedError("retrigger_yield() is not implemented yet")

    def get_ramp(self) -> float:
        """Returns the current ramp value based on the trend.

        The ramp value is positive for RISE, negative for FALL, and zero for LEVEL.
        """
        if self.trend == Trend.RISE:
            return self.cfg_lr.ramp_rise
        elif self.trend == Trend.FALL:
            return self.cfg_lr.ramp_fall
        elif self.trend == Trend.RETRIG_YIELD:
            # originally positive, make negative.
            return -self.cfg_lr.ramp_rise
        else:
            return 0.0
        
    @property
    def ramp(self) -> float:
        return self.get_ramp()

    def advance(self, nsamps: int) -> None:
        """Advance the internal state by `nsamps` samples.
        """
        #
        # NOTE Upon entering, all parameters reflect the start of the rendering request
        #      (before the rendering of the nsamps block).
        #
        if self.trend == Trend.SILENT:
            # The block was rendered as silence; no change in state.
            return
        #
        # As long as the note is not silent, the partial states are updated.
        # The sinusoids continue their phase, and the exponential decay continue.
        #
        sum_sq_amp = 0.0
        for _, state in self.partial_states:
            state.advance(nsamps)
            sum_sq_amp += state.next_amp ** 2
        #
        # We update the level if trend is RISE, FALL, or RETRIG_YIELD.
        #
        if self.trend != Trend.LEVEL:
            cfg_lr = self.cfg_lr
            if self.trend == Trend.RISE:
                ramp = cfg_lr.ramp_rise
            elif self.trend == Trend.FALL:
                # negative as configured
                ramp = cfg_lr.ramp_fall
            elif self.trend == Trend.RETRIG_YIELD:
                # originally positive, make negative.
                ramp = -cfg_lr.ramp_rise
            else:
                raise RuntimeError(f"Internal error: Unimplemented trend: {self.trend}")
            level = (self.level + ramp * nsamps)
            level = max(self.level_min, min(self.level_max, level))
            self.level = level
        #
        # We check if the note has decayed to silence, taking both level and exponential
        # decay into account.
        #
        if self.level * math.sqrt(sum_sq_amp) < self.cfg_piano.audible_amp:
            self.silent()
        #
        return
