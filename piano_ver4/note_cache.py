import math
from typing import Any, Callable, Final, Mapping
# from time import perf_counter_ns

import numpy as np

from piano_data_model import Partial, PartialState, PianoCfg, PianoKey, PianoNote, LinearRamp, NoteState, Trend
from piano_model import PianoModel, NotePartials

class PartialCache:
    _partial: Final[Partial]

    def __init__(self, cfg: PianoCfg, partial: Partial, nsamps: int):
        self._partial = partial
        sr = cfg.sample_rate
        iota = np.arange(nsamps, dtype=np.float64)
        omega = 2.0 * math.pi * partial.p_freq / sr
        log_decay = -1.0 / (partial.p_tau * sr)
        # Partial relative amplitude (partial.p_amp) baked into the exponential decay.
        # partial_exp := p_amp * exp(-t/tau)
        partial_exp = partial.p_amp * np.exp(log_decay * iota)
        # expcos := p_amp * exp(-t/tau) * cos(omega*t)
        self.expcos = partial_exp * np.cos(omega * iota) 
        # expsin := p_amp * exp(-t/tau) * sin(omega*t)
        self.expsin = partial_exp * np.sin(omega * iota)

    def render_to(self, partial_state: PartialState, out: np.ndarray) -> None:
        """Renders the partial to the output buffer (overwritten),
        based on the partial_state.

        This function does not apply the linear ramp (attack/release) envelope.
        """
        # NOTE 
        # This function uses the following mathematical identity:
        #   `sin(a + b) == sin(a) * cos(b) + cos(a) * sin(b)`
        # Let:
        #   `a = omega * t`
        #   `b = next_phi`
        # We have:
        #   `sin(omega * t + next_phi) == sin(omega * t) * cos(next_phi) + cos(omega * t) * sin(next_phi)`
        # Multiplying both sides by the exponential decay (`exp(-t / tau)`) and the amplitude (`next_amp`) gives:
        #   `next_amp * exp(-t / tau) * sin(omega * t + next_phi) == next_amp * exp(-t / tau) * sin(omega * t) * cos(next_phi) + next_amp * exp(-t / tau) * cos(omega * t) * sin(next_phi)`
        # We take the scalar terms out (anything without the time variable `t`), and we have:
        #   `expcos(t) := exp(-t / tau) * cos(omega * t)`
        #   `expsin(t) := exp(-t / tau) * sin(omega * t)`
        #   `coef_expcos := next_amp * sin(next_phi)`
        #   `coef_expsin := next_amp * cos(next_phi)`
        # Final output:
        #   `output := coef_expcos * expcos(t) + coef_expsin * expsin(t)`
        #
        next_amp = partial_state.next_amp
        next_phi = partial_state.next_phi
        coef_expcos = next_amp * math.sin(next_phi)
        coef_expsin = next_amp * math.cos(next_phi)
        out[:] = (coef_expcos * self.expcos) + (coef_expsin * self.expsin)


class NoteCache:
    _cfg: Final[PianoCfg]
    _model: Final[PianoModel]
    _key: Final[PianoKey]
    _note_partials: Final[NotePartials]
    _partials_cache: Final[list[PartialCache]]

    def __init__(self, model: PianoModel, key: PianoKey, nsamps: int):
        self._model = model
        self._cfg = model.cfg
        self._key = model.keys[key.note_id]
        self._note_partials = NotePartials(model, self._key)
        self._partials_cache: list[PartialCache] = [
            PartialCache(self._note_partials.cfg, partial, nsamps)
            for partial in self._note_partials.valid_partials
        ]

    def render_to(self, note_state: NoteState, out: np.ndarray) -> None:
        if note_state.trend == Trend.SILENT:
            out.fill(0.0)
            return
        nsamps = out.shape[0]
        buf = np.empty_like(out)
        out.fill(0.0)
        for (_, partial_state), partial_cache in zip(note_state.partial_states, self._partials_cache):
            partial_cache.render_to(partial_state, buf)
            out += buf
        if note_state.trend == Trend.LEVEL:
            out *= note_state.level
        else:
            ramp = note_state.get_ramp()
            level = note_state.level
            level_min = note_state.level_min
            level_max = note_state.level_max
            iota = np.arange(nsamps, dtype=np.float64)
            out *= np.clip(level + ramp * iota, level_min, level_max)
        note_state.advance(nsamps)
