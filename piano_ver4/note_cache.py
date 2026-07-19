import math
from typing import Any, Callable, Final, Mapping
# from time import perf_counter_ns

import numpy as np

from piano_data_model import Partial, PartialState, PianoCfg, PianoKey, PianoNote, LinearRamp, NoteState, Trend
from piano_model import PianoModel, NotePartials


class RowBuffer:
    _dtype: Final[Any]
    _bufs: Final[list[np.ndarray]]
    _lookup: Final[dict[int, int]]
    _unused: Final[set[int]]
    _used: Final[set[int]]

    def __init__(self, entries: int, nsamps: int, dtype):
        self._dtype = dtype
        self._bufs = [np.empty((nsamps,), dtype=dtype) for _ in range(entries)]
        self._lookup = {id(buf): idx for idx, buf in enumerate(self._bufs)}
        self._unused = set(range(entries))
        self._used = set()

    def borrow(self) -> np.ndarray:
        if not self._unused:
            raise RuntimeError("No unused buffers available")
        idx = self._unused.pop()
        self._used.add(idx)
        return self._bufs[idx]
    
    def recycle(self, buf: np.ndarray) -> None:
        idx = self._lookup.get(id(buf))
        if idx is None or idx not in self._used or self._bufs[idx] is not buf:
            raise RuntimeError("Cannot recycle buffer that was not borrowed")
        self._used.remove(idx)
        self._unused.add(idx)


class PartialCache:
    _partial: Final[Partial]
    _row_cache: RowBuffer | None

    def __init__(self, cfg: PianoCfg, partial: Partial, nsamps: int, row_cache: RowBuffer | None = None):
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
        self._row_cache = row_cache

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
        if self._row_cache is not None:
            buf = self._row_cache.borrow()
            try:
                out[:] = self.expcos
                np.multiply(out, coef_expcos, out=out)
                buf[:] = self.expsin
                np.multiply(buf, coef_expsin, out=buf)
                np.add(out, buf, out=out)
            finally:
                self._row_cache.recycle(buf)
        else:
            out[:] = (coef_expcos * self.expcos) + (coef_expsin * self.expsin)


class NoteCache:
    _cfg: Final[PianoCfg]
    _model: Final[PianoModel]
    _key: Final[PianoKey]
    _note_partials: Final[NotePartials]
    _partials_cache: Final[list[PartialCache]]
    _row_cache: RowBuffer | None

    def __init__(self, model: PianoModel, key: PianoKey, nsamps: int, row_cache: RowBuffer | None = None):
        self._model = model
        self._cfg = model.cfg
        self._key = model.keys[key.note_id]
        self._note_partials = NotePartials(model, self._key)
        self._partials_cache: list[PartialCache] = [
            PartialCache(self._note_partials.cfg, partial, nsamps, row_cache=row_cache)
            for partial in self._note_partials.valid_partials
        ]
        self._row_cache = row_cache

    def render_to(self, note_state: NoteState, out: np.ndarray) -> None:
        if note_state.trend == Trend.SILENT:
            out.fill(0.0)
            return
        nsamps = out.shape[0]
        assert self._row_cache is not None
        buf = self._row_cache.borrow()
        try:
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
                buf[:] = np.arange(nsamps, dtype=buf.dtype)
                np.multiply(buf, ramp, out=buf)
                np.add(buf, level, out=buf)
                np.clip(buf, level_min, level_max, out=buf)
                out *= buf
        finally:
            self._row_cache.recycle(buf)
            note_state.advance(nsamps)
