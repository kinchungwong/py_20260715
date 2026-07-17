import math
from typing import Final, Any
from dataclasses import dataclass
import numpy as np
from time import perf_counter_ns

# @dataclass(frozen=True)
# class MidiPianoNote:
#     note_id: int
#     velocity: int


@dataclass(frozen=False, eq=False)
class MidiPianoModelArgs:
    """MIDI Piano Model Args.

    Attributes:
        - sample_rate: Sample rate in Hz, needed for Nyquist.
        - max_partials: Max number of partials a note can have.
        - inharmonicity: Non-linear deviations from harmonic series.
        - amp_pcoef: Coefficient for amplitude decay of partials (see equation)
        - tau_fund: Time constant for the fundamental frequency (in seconds)
        - tau_pcoef: Coefficient for time constant decay of partials (see equation)
        - hammer_pos: Normalized position of the string strike that causes missing partials
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
    note_id_min: int = 21
    note_id_max: int = 108


@dataclass(frozen=True)
class MidiPianoPartial:
    note_id: int
    partial_id: int
    partial_freq: float
    partial_tau: float
    partial_amp: float

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, MidiPianoPartial):
            return False
        return self.note_id == other.note_id and self.partial_id == other.partial_id
    
    def __hash__(self) -> int:
        return hash((self.note_id, self.partial_id))


class MidiPianoModel:
    model_args: Final[MidiPianoModelArgs]
    partials_by_note: Final[list[list[MidiPianoPartial]]]
    all_partials: Final[list[MidiPianoPartial]]

    def __init__(self, model_args: MidiPianoModelArgs):
        # TODO clone model_args.
        self.model_args = model_args
        self.partials_by_note = [
            self.note_to_partials(note_id)
            for note_id in range(model_args.note_id_min, model_args.note_id_max + 1)
        ]
        self.all_partials = [
            partial
            for note_partials in self.partials_by_note
            for partial in note_partials
        ]

    def note_to_partials(self, note_id: int) -> list[MidiPianoPartial]:
        model_args = self.model_args
        sample_rate = model_args.sample_rate
        audible_hz = model_args.audible_hz
        nyquist = sample_rate * 0.5
        max_freq = min(audible_hz, nyquist)
        max_partials = model_args.max_partials
        inharmonicity = model_args.inharmonicity
        amp_pcoef = model_args.amp_pcoef
        tau_fund = model_args.tau_fund
        tau_pcoef = model_args.tau_pcoef
        hammer_pos = model_args.hammer_pos
        freq = self._note_to_fundamental(note_id)
        tups: list[tuple[int, float, float, float]] = []
        for kp0 in range(max_partials):
            # In keeping with physics, partials start at 1.
            # TODO implement missing partials due to string strike positions.
            k = kp0 + 1
            vanish_check = k * hammer_pos
            if abs(vanish_check - round(vanish_check)) < 1e-6:
                continue
            p_freq = freq * k * math.sqrt(1.0 + inharmonicity * k * k)
            if p_freq >= max_freq:
                break
            p_tau = tau_fund / (1.0 + tau_pcoef * (k - 1))
            p_amp = 1.0 / (k ** amp_pcoef)
            if p_amp < 1e-3:
                break
            tups.append((k, p_freq, p_tau, p_amp))

        sum_amp = sum(tup[3] for tup in tups)
        scale_amp = 1.0 / sum_amp if sum_amp > 0.0 else 1.0
        results = [
            MidiPianoPartial(
                note_id=note_id,
                partial_id=k,
                partial_freq=p_freq,
                partial_tau=p_tau,
                partial_amp=(p_amp * scale_amp),
            )
            for k, p_freq, p_tau, p_amp in tups
        ]
        return results

    def _note_to_fundamental(self, note_id: int) -> float:
        """Convert a MIDI note to its fundamental frequency in Hz."""
        return 440.0 * (2.0 ** ((note_id - 69) / 12.0))

    # def cents_to_hz(self, root_hz: float, cents: float) -> float:
    #     return root_hz * 2.0 ** (cents / 1200.0)

    def build_cache(self, partials: list[MidiPianoPartial], max_nsamps: int, dtype) -> np.ndarray:
        sample_rate = self.model_args.sample_rate
        count = len(partials)
        result = np.empty((2*count, max_nsamps), dtype=dtype)
        iota = np.arange(max_nsamps)
        for idx, partial in enumerate(partials):
            freq = partial.partial_freq
            tau = partial.partial_tau
            amp = partial.partial_amp
            omega = 2.0 * math.pi * freq / sample_rate
            log_decay = -1.0 / (tau * sample_rate)
            sinexp = amp * np.sin(omega * iota) * np.exp(log_decay * iota)
            cosexp = amp * np.cos(omega * iota) * np.exp(log_decay * iota)
            result[2*idx, :] = sinexp
            result[2*idx + 1, :] = cosexp
        return result

if __name__ == "__main__":
    model_args = MidiPianoModelArgs()
    model = MidiPianoModel(model_args)
    if True:
        print(len(model.all_partials))
    if False:
        for note_id in range(model_args.note_id_min, model_args.note_id_max + 1):
            partials = model.partials_by_note[note_id - model_args.note_id_min]
            print(f"Note {note_id}: {len(partials)} partials")
            freq_min = partials[0].partial_freq
            freq_max = partials[-1].partial_freq
            print(f"  Frequency range: {freq_min:.2f} Hz - {freq_max:.2f} Hz")

    partials_to_build = model.all_partials

    max_nsamps = 1024
    print(f"Building cache for {len(partials_to_build)} partials, max_nsamps={max_nsamps}")

    t0 = perf_counter_ns()
    np_cache = model.build_cache(partials_to_build, max_nsamps=max_nsamps, dtype=np.float32)
    t1 = perf_counter_ns()

    cache_shape = np_cache.shape
    print("Cache shape:", cache_shape)
    print(f"Cache build time: {((t1 - t0) * 1e-6):.2f} ms")

    mixvec = np.random.uniform(low=0.0, high=1.0, size=(1, cache_shape[0])).astype(np.float32)
    print("Mixvec shape:", mixvec.shape)

    output = (mixvec @ np_cache).squeeze()
    print("Output shape:", output.shape)

    for trial_id in range(10):
        t2 = perf_counter_ns()
        # output = (np_cache.T @ mixvec.T).squeeze()
        output = (mixvec @ np_cache).squeeze()
        t3 = perf_counter_ns()
        print(f"  Output multiplication time [trial={trial_id}]: {((t3 - t2) * 1e-6):.2f} ms")
