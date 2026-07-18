import math
from typing import Final, Mapping, override
from piano_data_model import PianoModelBase, NotePartialsBase, PianoCfg, PianoKey, Partial


class PianoModel(PianoModelBase):
    def __init__(self, cfg: PianoCfg):
        self._cfg = cfg
        note_id_min, note_id_max = cfg.note_id_min, cfg.note_id_max
        self._keys = {
            note_id: PianoKey(note_id=note_id)
            for note_id in range(note_id_min, note_id_max + 1)
        }

    @property
    @override
    def cfg(self) -> PianoCfg:
        return self._cfg
    
    @property
    @override
    def keys(self) -> Mapping[int, PianoKey]:
        return self._keys

    @override
    def _note_to_fundamental(self, note_id: int) -> float:
        """Convert a MIDI note to its fundamental frequency in Hz."""
        return 440.0 * (2.0 ** ((note_id - 69) / 12.0))

    @override
    def _partial_freq(self, fund_freq: float, k: int) -> float:
        inharmonicity = self._cfg.inharmonicity
        return fund_freq * k * math.sqrt(1.0 + inharmonicity * k * k)
    
    @override
    def _partial_amp(self, k: int) -> float:
        """Compute the amplitude factor for a given partial index k,
        considering hammer position and amplitude rolloff.
        """
        # NOTE vanishing partials are handled not using divisibility checks,
        # but by deriving a mode amplitude factor from the hammer strike position.
        # This is meant to better support hammer positions that are close to
        # but deliberately not exact rationals of the string length.
        #
        # By taking absolute and raising to a high power (16), most partials
        # will get close to 1.0, and very few will get closer to zero.
        #
        cfg = self._cfg
        mode_amp = abs(math.sin(math.pi * k * cfg.hammer_pos)) ** 16
        return mode_amp / (k ** cfg.amp_pcoef)
    
    @override
    def _partial_tau(self, k: int) -> float:
        """Compute the time constant for a given partial index k,
        considering the fundamental time constant and decay rolloff.
        """
        cfg = self._cfg
        return cfg.tau_fund / (1.0 + cfg.tau_pcoef * (k - 1))

    @override
    def _valid_partial(self, k: int, p_freq: float, p_amp: float) -> bool:
        """Check if a partial is valid based on frequency and amplitude thresholds.
        """
        del k # Pyright unused
        cfg = self._cfg
        if p_freq >= min(cfg.audible_hz, cfg.sample_rate * 0.5):
            return False
        if p_amp < cfg.audible_amp:
            return False
        return True

class NotePartials(NotePartialsBase):
    _model: Final[PianoModel]
    _cfg: Final[PianoCfg]
    _key: Final[PianoKey]
    _fund_freq: Final[float]
    _partials: Final[tuple[Partial, ...]]

    def __init__(self, model: PianoModel, key: PianoKey):
        self._model = model
        self._cfg = model.cfg
        self._key = key
        self._fund_freq = fund_freq = self._model._note_to_fundamental(key.note_id)
        tups: list[tuple[int, float, float, float]] = []
        for kp0 in range(self._cfg.max_partials):
            k = kp0 + 1
            p_freq = self._model._partial_freq(fund_freq, k)
            p_amp = self._model._partial_amp(k)
            if not self._model._valid_partial(k, p_freq, p_amp):
                continue
            p_tau = self._model._partial_tau(k)
            tups.append((k, p_freq, p_amp, p_tau))
        #
        # Normalize relative amplitudes by sum of squares.
        #
        sum_sq_amp = sum((tup[2] ** 2) for tup in tups)
        scale_amp = 1.0 / math.sqrt(sum_sq_amp) if sum_sq_amp > 0.0 else 1.0
        partials = [
            Partial(note_id=key.note_id, partial_k=k, p_freq=p_freq, p_tau=p_tau, p_amp=(p_amp * scale_amp))
            for k, p_freq, p_amp, p_tau in tups
        ]
        self._partials = tuple(partials)

    @property
    @override
    def cfg(self) -> PianoCfg:
        return self._cfg

    @property
    @override
    def key(self) -> PianoKey:
        return self._key
    
    @property
    @override
    def valid_partials(self) -> tuple[Partial, ...]:
        return self._partials

    @property
    def fund_freq(self) -> float:
        return self._fund_freq
