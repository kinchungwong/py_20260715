"""Three render strategies behind one interface.

All three simulate live synthesis: each owns per-voice evolving state, renders
the FINAL mix of V voices into a caller-provided `(N,)` buffer, and advances
state one block. `ver5 has no advance/envelope machinery, so the advance is
implemented here (driver-side), allocation-free via `out=`.

  * NaiveUncached      -- baseline the cache is meant to beat: recompute
                          exp/sin every block, every partial. Allocation-free
                          per block (its loss is NumPy-dispatch count).
  * Ver5PerNoteCached  -- THE user path: one impl.RenderNpMCached per voice,
                          render + accumulate. Allocates a (2M,) concat + trig
                          temps per voice per block -> memory churn.
  * BatchedCollapse    -- the research ceiling: all voices' (2M,N) tables stacked
                          into one (2*sumM, N) matrix; one np.matmul per block.
                          Allocation-free in steady state.

Strategies 2 and 3 reuse the kernel's exact (expsin|expcos)/(amp*cos|amp*sin)
pairing, so all three are bit-comparable (the correctness gate proves it).
"""
from __future__ import annotations

import math
from typing import Literal

import numpy as np

import damped_sine_impl as impl
from partial_gen import VoiceSpec

TWO_PI = 2.0 * math.pi
Mode = Literal["level", "decay"]


def _copy_state(v: VoiceSpec, dtype) -> impl.StateM:
    """Fresh, mutable attack-time state for one voice (independent copy so a
    strategy can be re-run from block 0)."""
    return impl.StateM(
        next_amp=np.array(v.state0.next_amp, dtype=dtype, copy=True),
        next_phi=np.array(v.state0.next_phi, dtype=dtype, copy=True),
    )


class _Base:
    name: str = "base"

    def __init__(self, voices: list[VoiceSpec], nsamps: int, dtype, mode: Mode) -> None:
        self._voices = voices
        self._nsamps = nsamps
        self._dtype = np.dtype(dtype)
        self._mode = mode
        self._states = [_copy_state(v, dtype) for v in voices]
        # Advance factors, cast to the working dtype once.
        self._incr = [np.asarray(v.phase_incr, dtype=dtype) for v in voices]
        self._decay = [np.asarray(v.decay_per_block, dtype=dtype) for v in voices]

    def _advance(self) -> None:
        decaying = self._mode == "decay"
        for st, incr, dec in zip(self._states, self._incr, self._decay):
            np.add(st.next_phi, incr, out=st.next_phi)
            np.mod(st.next_phi, TWO_PI, out=st.next_phi)
            if decaying:
                np.multiply(st.next_amp, dec, out=st.next_amp)

    # Subclasses implement _render(mix); render_block = render + advance.
    def _render(self, mix: np.ndarray) -> None:
        raise NotImplementedError

    def render_block(self, mix: np.ndarray) -> None:
        self._render(mix)
        self._advance()

    def analytic_bytes_per_block(self) -> int:
        return 0


class NaiveUncached(_Base):
    """Recompute amp*exp(log_decay*iota)*sin(phi+omega*iota) per block per
    partial and sum. Allocation-free (preallocated scratch)."""
    name = "naive_uncached"

    def __init__(self, voices, nsamps, dtype, mode) -> None:
        super().__init__(voices, nsamps, dtype, mode)
        self._iota = np.arange(nsamps, dtype=dtype)
        self._scr1 = np.empty(nsamps, dtype=dtype)
        self._scr2 = np.empty(nsamps, dtype=dtype)

    def _render(self, mix: np.ndarray) -> None:
        mix.fill(0.0)
        iota, scr1, scr2 = self._iota, self._scr1, self._scr2
        for v, st in zip(self._voices, self._states):
            omega = v.partials.omega
            logd = v.partials.log_decay
            amp = st.next_amp
            phi = st.next_phi
            for k in range(v.M):
                # scr1 = sin(omega_k * iota + phi_k)
                np.multiply(iota, omega[k], out=scr1)
                np.add(scr1, phi[k], out=scr1)
                np.sin(scr1, out=scr1)
                # scr2 = exp(log_decay_k * iota)
                np.multiply(iota, logd[k], out=scr2)
                np.exp(scr2, out=scr2)
                # mix += amp_k * scr1 * scr2
                np.multiply(scr1, scr2, out=scr1)
                np.multiply(scr1, amp[k], out=scr1)
                np.add(mix, scr1, out=mix)


class Ver5PerNoteCached(_Base):
    """One impl.RenderNpMCached per voice; render each into shared scratch and
    accumulate. This is the production path under test."""
    name = "ver5_per_note_cached"

    def __init__(self, voices, nsamps, dtype, mode) -> None:
        super().__init__(voices, nsamps, dtype, mode)
        kdt = self._dtype  # iota dtype; if partials are f64 the (2M,N) cache
                           # upcasts to f64 regardless -- that IS the gotcha.
        # SpareST is inert for RenderNpMCached (render uses concatenate, never
        # the pool) but the constructor requires a SpareBase.
        self._renderers = [
            impl.RenderNpMCached(
                nsamps, v.partials, kdt,
                impl.SpareST(1, nsamps, kdt),
            )
            for v in voices
        ]
        self._scratch = np.empty(nsamps, dtype=dtype)

    def _render(self, mix: np.ndarray) -> None:
        mix.fill(0.0)
        scratch = self._scratch
        for r, st in zip(self._renderers, self._states):
            r.render(st, scratch)
            np.add(mix, scratch, out=mix)

    def analytic_bytes_per_block(self) -> int:
        # Per voice/block RenderNpMCached.render allocates: cos temp, sin temp,
        # two amp*trig results, and the (2M,) concat = ~6*M elements.
        itemsize = self._states[0].next_phi.dtype.itemsize if self._states else 0
        return sum(6 * v.M for v in self._voices) * itemsize


class BatchedGemv(_Base):
    """Diagnostic: collapse only the MATMUL into one GEMV, but still build the
    coefficient vector with a per-voice Python loop. Isolates how much of
    per-note's cost is the matmul count vs the coefficient build."""
    name = "batched_1gemv"

    def __init__(self, voices, nsamps, dtype, mode) -> None:
        super().__init__(voices, nsamps, dtype, mode)
        kdt = self._dtype
        # Reuse the kernel's own _stacked so the batched path is bit-identical
        # to the per-note path (same expsin|expcos rows).
        tables = []
        for v in voices:
            r = impl.RenderNpMCached(nsamps, v.partials, kdt, impl.SpareST(1, nsamps, kdt))
            tables.append(r._stacked)
        self._big_stacked = np.ascontiguousarray(np.concatenate(tables, axis=0))
        mv = np.empty(self._big_stacked.shape[0], dtype=self._big_stacked.dtype)
        self._big_mixvec = mv
        # Precompute each voice's mixvec slice VIEWS once, so _render allocates
        # nothing (a fresh mv[a:b] each block would be a new ndarray object ->
        # gc pressure that would pollute the churn comparison).
        self._cos_views: list[np.ndarray] = []
        self._sin_views: list[np.ndarray] = []
        base = 0
        for v in voices:
            self._cos_views.append(mv[base:base + v.M])          # pairs with expsin
            self._sin_views.append(mv[base + v.M:base + 2 * v.M])  # pairs with expcos
            base += 2 * v.M

    def _render(self, mix: np.ndarray) -> None:
        for st, cv, sv in zip(self._states, self._cos_views, self._sin_views):
            np.cos(st.next_phi, out=cv); cv *= st.next_amp
            np.sin(st.next_phi, out=sv); sv *= st.next_amp
        # ONE BLAS call; mixdown fused into the GEMV.
        np.matmul(self._big_mixvec, self._big_stacked, out=mix)


class BatchedSoa(_Base):
    """The research ceiling: state as concatenated (sumM,) vectors, so BOTH the
    coefficient build (one cos + one sin over all partials) AND the advance are
    O(1) NumPy calls regardless of voice count, feeding a single GEMV. Fully
    allocation-free."""
    name = "batched_soa"

    def __init__(self, voices, nsamps, dtype, mode) -> None:
        super().__init__(voices, nsamps, dtype, mode)
        kdt = self._dtype
        # Structure-of-arrays state across ALL partials of ALL voices.
        self._phi = np.concatenate([np.asarray(v.state0.next_phi, kdt) for v in voices])
        self._amp = np.concatenate([np.asarray(v.state0.next_amp, kdt) for v in voices])
        self._incr = np.concatenate([np.asarray(v.phase_incr, kdt) for v in voices])
        self._decay = np.concatenate([np.asarray(v.decay_per_block, kdt) for v in voices])
        rows = self._phi.shape[0]  # total partials
        # big_stacked = [all expsin rows (rows,N); all expcos rows (rows,N)].
        expsin, expcos = [], []
        for v in voices:
            r = impl.RenderNpMCached(nsamps, v.partials, kdt, impl.SpareST(1, nsamps, kdt))
            expsin.append(r._stacked[:v.M])
            expcos.append(r._stacked[v.M:])
        self._big_stacked = np.ascontiguousarray(
            np.concatenate((np.vstack(expsin), np.vstack(expcos)), axis=0)
        )  # (2*rows, N)
        mv = np.empty(2 * rows, dtype=self._big_stacked.dtype)
        self._big_mixvec = mv
        self._cos_half = mv[:rows]   # pairs with expsin block
        self._sin_half = mv[rows:]   # pairs with expcos block

    def _advance(self) -> None:
        np.add(self._phi, self._incr, out=self._phi)
        np.mod(self._phi, TWO_PI, out=self._phi)
        if self._mode == "decay":
            np.multiply(self._amp, self._decay, out=self._amp)

    def _render(self, mix: np.ndarray) -> None:
        np.cos(self._phi, out=self._cos_half); self._cos_half *= self._amp
        np.sin(self._phi, out=self._sin_half); self._sin_half *= self._amp
        np.matmul(self._big_mixvec, self._big_stacked, out=mix)


STRATEGIES: dict[str, type[_Base]] = {
    NaiveUncached.name: NaiveUncached,
    Ver5PerNoteCached.name: Ver5PerNoteCached,
    BatchedGemv.name: BatchedGemv,
    BatchedSoa.name: BatchedSoa,
}
