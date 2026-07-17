import math
from typing import Final, Iterable
from bisect import bisect_left
import numpy as np
from damped_sine_args import DampedSineArgsDt
from damped_sine_args_cvt import next_dt_args
from damped_sine import DampedSine
from damped_sine_cache_entries import ComboEntry, FreqEntry, DecayEntry
from damped_sine_cache_keys import DampedSineCacheKeys
from damped_sine_state import DampedSineState

class DampedSineCache:
    """Feeble attempt to make DampedSine faster with caching.

    List of decomposition techniques.
    
    Sine, phi0 and amp0 are transformed together as:

        `amp0 * sin(phi0 + omega * iota)`
        `==>`
        `(amp0 * sin(phi0)) * cos(omega * iota) + (amp0 * cos(phi0)) * sin(omega * iota)`

    Of those, cos(omega * iota) and sin(omega * iota) can be cached for a chosen set
    of omega (freq), and a sufficiently large choice of nsamps (length).

    Then, the cached sine and cosine components are fused with the exponential decay:

        `sin(omega * iota) * exp(log(decay) * iota)`
        `cos(omega * iota) * exp(log(decay) * iota)`
    
    Again, these are cached for a chosen set of tuples (omega, decay).

    Automatically falls back to DampedSine.render_dt() if the cache cannot be used for
    any reason.

    Implementation details.

    While the runtime cache lookup parameters (omega, decay) are floating point values,
    we use application-aware approach to discretize and compare them to the chosen set
    of cached values. In order to do so, we require the initializer to know the actual
    sample rate, and the set of frequencies and time constants that will be cached.
    """

    max_nsamps: Final[int]
    samp_rate: Final[float]
    freq_sorted: Final[list[float]]
    tau_sorted: Final[list[float]]
    dtype: Final[np.dtype]
    _freq_cache: Final[dict[int, FreqEntry]]
    _decay_cache: Final[dict[int, DecayEntry]]
    _combo_cache: Final[dict[tuple[int, int], ComboEntry]]

    # New implementation:
    # Exact float match only.

    FREQ_TOL = 1e-6
    TAU_TOL = 1e-6

    def __init__(
            self, 
            max_nsamps: int, 
            samp_rate: float, 
            freq_set: Iterable[float], 
            tau_set: Iterable[float], 
            freq_tau_combos: Iterable[tuple[float, float]], 
            dtype,
        ) -> None:
        self.max_nsamps = max_nsamps
        self.samp_rate = samp_rate
        self.freq_sorted = self._sanitize_freqs(freq_set)
        self.tau_sorted = self._sanitize_taus(tau_set)
        self.dtype = dtype
        self._freq_cache = self._init_freq_cache(self.freq_sorted)
        self._decay_cache = self._init_decay_cache(self.tau_sorted)
        self._combo_cache = self._init_combo_cache(freq_tau_combos)

    def _init_freq_cache(self, freq_sorted: list[float]) -> dict[int, FreqEntry]:
        freq_cache = dict[int, FreqEntry]()
        iota = np.arange(self.max_nsamps)
        for freq_index, freq in enumerate(freq_sorted):
            omega = 2.0 * math.pi * freq / self.samp_rate
            sinpart = np.sin(omega * iota).astype(self.dtype)
            cospart = np.cos(omega * iota).astype(self.dtype)
            freq_cache[freq_index] = FreqEntry(
                freq_index=freq_index,
                freq=freq,
                omega=omega,
                sinpart=sinpart,
                cospart=cospart,
            )
        return freq_cache

    def _init_decay_cache(self, tau_sorted: list[float]) -> dict[int, DecayEntry]:
        """Initialize delay cache for a given set of tau values.
        Args:
            tau_sorted: list[float]
                Sorted list of tau values to cache.
                Must be sorted, validated, and deduplicated.
        """
        decay_cache = dict[int, DecayEntry]()
        iota = np.arange(self.max_nsamps)
        for tau_index, tau in enumerate(tau_sorted):
            log_decay = -1.0 / (tau * self.samp_rate)
            exppart = np.exp(iota * log_decay).astype(self.dtype)
            decay_cache[tau_index] = DecayEntry(
                tau_index=tau_index,
                tau=tau,
                log_decay=log_decay,
                exppart=exppart,
            )
        return decay_cache

    def _init_combo_cache(self, freq_tau_combos: Iterable[tuple[float, float]]) -> dict[tuple[int, int], ComboEntry]:
        combo_cache = dict[tuple[int, int], ComboEntry]()
        for freq, tau in freq_tau_combos:
            if math.isinf(tau):
                continue
            freq_index = self._nearest_freq_index(freq)
            if not self._within_tol(freq, self.freq_sorted[freq_index], self.FREQ_TOL):
                continue
            tau_index = self._nearest_tau_index(tau)
            if not self._within_tol(tau, self.tau_sorted[tau_index], self.TAU_TOL):
                continue
            if (freq_index, tau_index) in combo_cache:
                continue
            freq_entry = self._freq_cache[freq_index]
            decay_entry = self._decay_cache[tau_index]
            sinexp = (freq_entry.sinpart * decay_entry.exppart).astype(self.dtype)
            cosexp = (freq_entry.cospart * decay_entry.exppart).astype(self.dtype)
            combo_cache[(freq_index, tau_index)] = ComboEntry(
                freq_index=freq_index,
                tau_index=tau_index,
                freq=freq,
                omega=freq_entry.omega,
                tau=tau,
                log_decay=decay_entry.log_decay,
                sinexp=sinexp,
                cosexp=cosexp,
            )
        return combo_cache

    def stateful_render(self, state: DampedSineState, out: np.ndarray) -> None:
        assert out.ndim == 1
        nsamps = out.shape[0]
        if nsamps > self.max_nsamps:
            self._render_fallback(state, out)
        else:
            cache_keys = state.cache_keys
            if cache_keys is None:
                cache_keys = self._populate_search_keys(state)
            if cache_keys.combo_entry is not None:
                self._render_combo(state, out)
            elif cache_keys.freq_entry is not None or cache_keys.decay_entry is not None:
                self._render_partial(state, out)
            else:
                self._render_fallback(state, out)

    def _render_combo(self, state: DampedSineState, out: np.ndarray) -> None:
        assert out.ndim == 1
        nsamps = out.shape[0]
        assert nsamps <= self.max_nsamps
        cache_keys = state.cache_keys
        assert isinstance(cache_keys, DampedSineCacheKeys)
        combo = cache_keys.combo_entry
        assert isinstance(combo, ComboEntry)
        assert self._within_tol(combo.freq, state.freq, self.FREQ_TOL)
        assert self._within_tol(combo.tau, state.tau, self.TAU_TOL)
        self._write_composed_sine(
            amp0=state.amp0,
            phi0=state.phi0,
            sinpart=combo.sinexp,
            cospart=combo.cosexp,
            out=out,
        )
        self._update_state(state, nsamps)

    def _render_partial(self, state: DampedSineState, out: np.ndarray) -> None:
        assert out.ndim == 1
        nsamps = out.shape[0]
        assert nsamps <= self.max_nsamps
        cache_keys = state.cache_keys
        assert isinstance(cache_keys, DampedSineCacheKeys)
        freq_entry, decay_entry = cache_keys.freq_entry, cache_keys.decay_entry
        iota = None

        # NOTE: For the sine part, the output array is always overwritten.
        #
        if freq_entry is not None:
            assert isinstance(freq_entry, FreqEntry)
            assert self._within_tol(freq_entry.freq, state.freq, self.FREQ_TOL)
            self._write_composed_sine(
                amp0=state.amp0,
                phi0=state.phi0,
                sinpart=freq_entry.sinpart,
                cospart=freq_entry.cospart,
                out=out,
            )
        else:
            if iota is None:
                iota = np.arange(out.shape[0])
            out[:] = state.amp0 * np.sin(state.phi0 + state.omega * iota)

        # NOTE: For the decay part, the output array is always multiplied with.
        #
        if cache_keys.non_decaying:
            # Non-decaying case, no multiplicative decay to apply.
            pass
        else:
            if decay_entry is not None:
                assert isinstance(decay_entry, DecayEntry)
                assert self._within_tol(decay_entry.tau, state.tau, self.TAU_TOL)
                out[:] *= decay_entry.exppart[:nsamps]
            else:
                if iota is None:
                    iota = np.arange(out.shape[0])
                out[:] *= np.exp(iota * state.log_decay)

        self._update_state(state, nsamps)

    def _render_fallback(self, state: DampedSineState, out: np.ndarray) -> None:
        """Fallback rendering using DampedSine.render_dt() for non-cached values.

        Note: fallback is not subject to self.max_nsamps.
        """
        args_dt = DampedSineArgsDt(
            phi0=state.phi0,
            omega=state.omega,
            amp0=state.amp0,
            log_decay=state.log_decay,
        )
        next_args_dt = DampedSine.render_dt(args_dt, out)
        state.phi0 = next_args_dt.phi0
        state.amp0 = next_args_dt.amp0

    def _update_state(self, state: DampedSineState, nsamps: int) -> None:
        """Update the state after rendering `nsamps` samples."""
        # TODO streamline this code later.
        args_dt = DampedSineArgsDt(
            phi0=state.phi0,
            omega=state.omega,
            amp0=state.amp0,
            log_decay=state.log_decay,
        )
        next_args_dt = next_dt_args(args_dt, nsamps)
        state.phi0 = next_args_dt.phi0
        state.amp0 = next_args_dt.amp0

    @classmethod
    def _write_composed_sine(cls, amp0: float, phi0: float, sinpart: np.ndarray, cospart: np.ndarray, out: np.ndarray) -> None:
        """Compose a sine wave with initial values and cached components.

        Args:
            amp0: float
                Initial amplitude.
            phi0: float
                Initial phase in radians.
            sinpart: np.ndarray, shape (nsamps,)
                Cached sine component for the given frequency.
            cospart: np.ndarray, shape (nsamps,)
                Cached cosine component for the given frequency.
            out: np.ndarray, shape (nsamps,)
                Output array to write the composed sine wave.

        Note on args:
            nsamps is inferred from the minimum length of `out`, `sinpart`, and `cospart`.
            It is allowed that sinpart, cospart be longer than out.
            If in case sinpart, cospart are shorter than out, the uncomputed portions
            of out will be zeroed.
            Caller should guard against these cases, without relying on this function to
            do so.

        Implementation detail.

            We exploit the identity:
                `sin(a + b) === sin(a) * cos(b) + cos(a) * sin(b)`
            Make sure to use sine and cosine exactly as the given identity.
            
            We set `a := phi0` and `b := omega * iota`, and then multiply by `amp0` to get:
                `amp0 * sin(phi0 + omega * iota)`
                `===`
                `(amp0 * cos(phi0)) * sin(omega * iota) + (amp0 * sin(phi0)) * cos(omega * iota)`
        
        Once again, make sure to use sine and cosine exactly as the given identity,
        and make sure AI coding assistants do not change a correct implementation
        into an incorrect one.

        This function can be used with both decayed and non-decayed sine and cosine components,
        as long as the two components are generated with the same `freq` and `log_decay` value.
        """

        if sinpart.ndim != 1 or cospart.ndim != 1 or out.ndim != 1:
            raise ValueError(f"sinpart, cospart, and out must be 1D arrays, got shapes {sinpart.shape}, {cospart.shape}, {out.shape}")
        outlen = out.shape[0]
        nsamps = min(outlen, sinpart.shape[0], cospart.shape[0])

        # Note to AI assistants: re-read the docstring carefully.
        out[:nsamps] = amp0 * math.cos(phi0) * sinpart[:nsamps] + amp0 * math.sin(phi0) * cospart[:nsamps]

        if outlen > nsamps:
            out[nsamps:] = 0.0

    @classmethod
    def _nearest_index(cls, value: float, sorted_values: list[float]) -> int:
        """Find the index of the nearest value in a sorted list."""
        # NOTE infinites can complicate the comparisons inside search logic, must screen out.
        if math.isinf(value):
            raise ValueError("unsanitized search value, caller fault.")
        idx = bisect_left(sorted_values, value)
        if idx == 0:
            return 0
        if idx == len(sorted_values):
            return len(sorted_values) - 1
        before = sorted_values[idx - 1]
        after = sorted_values[idx]
        if abs(after - value) < abs(value - before):
            return idx
        else:
            return idx - 1

    def _nearest_freq_index(self, freq: float) -> int:
        return self._nearest_index(freq, self.freq_sorted)
    
    def _nearest_tau_index(self, tau: float) -> int:
        return self._nearest_index(tau, self.tau_sorted)

    def _populate_search_keys(self, state: DampedSineState) -> DampedSineCacheKeys:
        """Populate the search keys for the given state."""

        if isinstance(state.cache_keys, DampedSineCacheKeys):
            return state.cache_keys

        cache_keys = DampedSineCacheKeys(state.tau)

        freq_idx = self._nearest_freq_index(state.freq)
        freq_is_close = self._within_tol(self.freq_sorted[freq_idx], state.freq, self.FREQ_TOL)
        if freq_is_close:
            cache_keys.freq_entry = self._freq_cache[freq_idx]

        tau_idx = -1
        tau_is_close = False
        if cache_keys.non_decaying:
            # Non-decaying case, no multiplicative decay to apply.
            pass
        else:
            tau_idx = self._nearest_tau_index(state.tau)
            tau_is_close = self._within_tol(self.tau_sorted[tau_idx], state.tau, self.TAU_TOL)
            if tau_is_close:
                cache_keys.decay_entry = self._decay_cache[tau_idx]

        if freq_is_close and tau_is_close:
            if (freq_idx, tau_idx) in self._combo_cache:
                cache_keys.combo_entry = self._combo_cache[(freq_idx, tau_idx)]

        state.cache_keys = cache_keys
        return cache_keys

    def _sort_dedupe_inplace(self, values: list[float], tol: float) -> None:
        """Sort and deduplicate the list of values in-place, within a given tolerance.
        """
        if not values:
            return
        values.sort()
        new_idx = 0
        for old_idx, value in enumerate(values):
            if old_idx == 0:
                new_idx += 1
            else:
                if not self._within_tol(value, values[new_idx - 1], tol):
                    values[new_idx] = value
                    new_idx += 1
        if new_idx < len(values):
            del values[new_idx:]

    def _sanitize_freqs(self, freq_set: Iterable[float]) -> list[float]:
        """Sanitize the frequency set by filtering, deduplicating, and sorting."""
        nyquist = 0.5 * self.samp_rate
        sanitized = [f for f in freq_set if (0.0 < f < nyquist)]
        self._sort_dedupe_inplace(sanitized, self.FREQ_TOL)
        return sanitized
    
    def _sanitize_taus(self, tau_set: Iterable[float]) -> list[float]:
        """Sanitize the tau set by filtering, deduplicating, and sorting.

        Note: while tau can be infinite (non-decaying), we ignore it from
        the tau cache, since the non-decaying case does not require
        multiplicative decay processing.
        """
        sanitized = [t for t in tau_set if not math.isinf(t) and (t > 0.0)]
        self._sort_dedupe_inplace(sanitized, self.TAU_TOL)
        return sanitized

    def _within_tol(self, a: float, b: float, tol: float) -> bool:
        if math.isinf(tol):
            raise ValueError("tolerance must be finite")
        if math.isinf(a) or math.isinf(b):
            return False
        return abs(a - b) <= tol
