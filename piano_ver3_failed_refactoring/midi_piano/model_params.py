from dataclasses import dataclass


@dataclass(frozen=False, eq=False)
class PianoModelParams:
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
    sample_rate: float = 44100.0
    audible_hz: float = 18000.0
    max_partials: int = 16
    inharmonicity: float = 4.0e-4
    amp_pcoef: float = 1.2
    tau_fund: float = 1.6
    tau_pcoef: float = 0.25
    hammer_pos: float = 1.0 / 7.0
    attack_msecs: float = 6.0
    release_msecs: float = 20.0
    note_id_min: int = 21
    note_id_max: int = 108

    def clone(self) -> "PianoModelParams":
        return PianoModelParams(
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
        )
