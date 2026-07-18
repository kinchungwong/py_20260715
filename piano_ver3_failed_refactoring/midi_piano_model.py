import math
from typing import Final, Any
from dataclasses import dataclass
import numpy as np
from time import perf_counter_ns

from midi_piano.model_params import MidiPianoModelArgs
from midi_piano.piano_note import MidiPianoNote
from midi_piano.piano_partial import MidiPianoPartial


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
            result[2*idx,     :] = sinexp
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

    # num_partials = 224
    num_partials = 600
    partials_to_build = model.all_partials[:num_partials]

    max_nsamps = 1024
    print(f"Building cache for {len(partials_to_build)} partials, max_nsamps={max_nsamps}")

    t0 = perf_counter_ns()
    np_cache = model.build_cache(partials_to_build, max_nsamps=max_nsamps, dtype=np.float32)
    t1 = perf_counter_ns()

    cache_shape = np_cache.shape
    print("Cache shape:", cache_shape)
    print(f"Cache build time: {((t1 - t0) * 1e-6):.2f} ms")

    mixvec = np.random.uniform(low=0.0, high=1.0, size=(2*num_partials,)).astype(np.float32)
    print("Mixvec shape:", mixvec.shape)

    nsamps = 383
    np_cache_sliced = np_cache[:, :nsamps]
    print("Cache (nsamps sliced) shape:", np_cache_sliced.shape)

    output = (mixvec @ np_cache_sliced).squeeze()
    print("Output shape:", output.shape)

    for trial_id in range(10):
        t2 = perf_counter_ns()
        # output = (np_cache.T @ mixvec.T).squeeze()
        # output[:] = (mixvec @ np_cache).squeeze()
        output[:] = (mixvec @ np_cache_sliced).squeeze()
        t3 = perf_counter_ns()
        print(f"  Output multiplication time [trial={trial_id}]: {((t3 - t2) * 1e-6):.2f} ms")

