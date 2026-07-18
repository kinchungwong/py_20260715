import math
from typing import Final
from model_params import PianoModelParams
from piano_note import PianoNote
from piano_partial import PianoPartial


class PianoModel:
    model_args: Final[PianoModelParams]
    notes: Final[list[PianoNote]]
    notes_by_id: Final[dict[int, PianoNote]]
    grouped_partials: Final[list[list[PianoPartial]]]
    partials_by_note: Final[dict[int, list[PianoPartial]]]
    all_partials: Final[list[PianoPartial]]
    partials_ids_by_note_id: Final[dict[int, list[int]]]
    lookup_by_note_partial_id: Final[dict[tuple[int, int], PianoPartial]]

    def __init__(self, model_args: PianoModelParams):
        self.model_args = model_args.clone()
        note_id_min, note_id_max = model_args.note_id_min, model_args.note_id_max
        self.notes = [
            PianoNote(note_id=note_id, velocity=0)
            for note_id in range(note_id_min, note_id_max + 1)
        ]
        self.notes_by_id = {
            note.note_id: note
            for note in self.notes
        }
        self.grouped_partials = [
            self.note_to_partials(note.note_id)
            for note in self.notes
        ]
        self.partials_by_note = {
            note_partials[0].note_id: note_partials
            for note_partials in self.grouped_partials
        }
        self.all_partials = [
            partial
            for note_partials in self.grouped_partials
            for partial in note_partials
        ]
        self.partials_ids_by_note_id = {
            note_id: [partial.partial_id for partial in note_partials]
            for note_id, note_partials in self.partials_by_note.items()
        }
        self.lookup_by_note_partial_id = {
            (partial.note_id, partial.partial_id): partial
            for partial in self.all_partials
        }

    def note_to_partials(self, note_id: int) -> list[PianoPartial]:
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
            #
            # In keeping with physics, partials start at 1.
            #
            k = kp0 + 1
            #
            # Hammer strike position dependent vanishing partials.
            #
            vanish_check = k * hammer_pos
            if abs(vanish_check - round(vanish_check)) < 1e-6:
                continue
            #
            # Inharmonicity
            #
            p_freq = freq * k * math.sqrt(1.0 + inharmonicity * k * k)
            if p_freq >= max_freq:
                break
            #
            # Time constant decay of partials
            #
            p_tau = tau_fund / (1.0 + tau_pcoef * (k - 1))
            #
            # Amplitude rolloff of partials
            #
            p_amp = 1.0 / (k ** amp_pcoef)
            if p_amp < 1e-3:
                break
            #
            # Validated partial
            #
            tups.append((k, p_freq, p_tau, p_amp))

        #
        # Normalize relative amplitudes by sum.
        #
        sum_amp = sum(tup[3] for tup in tups)
        scale_amp = 1.0 / sum_amp if sum_amp > 0.0 else 1.0

        results = [
            PianoPartial(
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
