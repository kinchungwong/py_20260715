import math
from typing import Any, Final
from collections import defaultdict
from dataclasses import dataclass
import itertools
import numpy as np
from damped_sine import DampedSine, DampedSineRendererBase


@dataclass(frozen=True, eq=False)
class SinCosEntry:
    sin_part: np.ndarray
    cos_part: np.ndarray


@dataclass(frozen=True, eq=False)
class ExpEntry:
    exp_part: np.ndarray


class DampedSineComposedArgs:
    has_decay: Final[bool]
    sincos_entry: SinCosEntry | None
    exp_entry: ExpEntry | None

    def __init__(self, info: DampedSine) -> None:
        self.has_decay = info.has_decay
        self.sincos_entry = None
        self.exp_entry = None

    def is_useful(self) -> bool:
        return self.sincos_entry is not None or (self.has_decay and self.exp_entry is not None)


class DampedSineComposed(DampedSineRendererBase):
    """Composed damped sine renderer.

    This renderer caches the sinusoidal and exponential components of the damped sine wave for reuse.
    """

    max_nsamps: Final[int]
    _sealed: bool
    _sincos_cache: defaultdict[int, list[tuple[float, SinCosEntry]]]
    _exp_cache: defaultdict[int, list[tuple[float, ExpEntry]]]
    _iota: np.ndarray

    _LOG_DECAY_SCALE = 65536.0
    _LOG_DECAY_RTOL = 1e-8
    _LOG_DECAY_ATOL = 1e-8
    _OMEGA_SCALE = 65536.0
    _OMEGA_RTOL = 0.0
    _OMEGA_ATOL = 1e-8

    @classmethod
    def _cls_validate_tols(cls, max_nsamps: int) -> None:
        """Ensures that chosen parameters do not cause render boundaries
        to exceed 1e-4 in absolute error.

        Refer to _try_find_cache_entry() for detailed tolerance usage.
        """
        assert max(cls._OMEGA_RTOL * math.pi, cls._OMEGA_ATOL) <= 1.0 / cls._OMEGA_SCALE

        # TODO Figure out how to assert the maximum amplitude error in
        #      the exponential decay part, given the log_decay tolerances.
        #
        assert cls._LOG_DECAY_ATOL * max_nsamps <= 1e-4

        # At maximum frequency (omega = pi), the maximum phase error 
        # does not exceed: (in radians)
        assert cls._OMEGA_RTOL * math.pi * max_nsamps <= 1e-4
        assert cls._OMEGA_ATOL * max_nsamps <= 1e-4

    def __init__(self, max_nsamps: int) -> None:
        if max_nsamps <= 0:
            raise ValueError("max_nsamps must be positive")
        self._cls_validate_tols(max_nsamps)
        self.max_nsamps = max_nsamps
        self._sealed = False
        self._sincos_cache = defaultdict(list)
        self._exp_cache = defaultdict(list)
        self._iota = np.arange(self.max_nsamps)

    def is_sealed(self) -> bool:
        return self._sealed
    
    def set_sealed(self) -> None:
        self._sealed = True

    def _populate_sincos(self, info: DampedSine) -> SinCosEntry:
        result = self._try_find_sincos(info)
        if result is not None:
            return result
        omega = info.omega
        scaled_omega = int(round(omega * self._OMEGA_SCALE))
        target_list = self._sincos_cache[scaled_omega]
        iota = self._iota
        result = SinCosEntry(
            sin_part=np.sin(omega * iota),
            cos_part=np.cos(omega * iota),
        )
        target_list.append((omega, result))
        return result

    def _populate_exp(self, info: DampedSine) -> ExpEntry:
        result = self._try_find_exp(info)
        if result is not None:
            return result
        log_decay = info.log_decay
        scaled_log_decay = int(round(log_decay * self._LOG_DECAY_SCALE))
        target_list = self._exp_cache[scaled_log_decay]
        iota = self._iota
        result = ExpEntry(exp_part=np.exp(log_decay * iota))
        target_list.append((log_decay, result))
        return result

    def _try_find_sincos(self, info: DampedSine) -> SinCosEntry | None:
        float_key = info.omega
        key_scaler = self._OMEGA_SCALE
        cache_dict = self._sincos_cache
        rtol = self._OMEGA_RTOL
        atol = self._OMEGA_ATOL
        return self._try_find_cache_entry(float_key, key_scaler, cache_dict, rtol, atol)
    
    def _try_find_exp(self, info: DampedSine) -> ExpEntry | None:
        float_key = info.log_decay
        key_scaler = self._LOG_DECAY_SCALE
        cache_dict = self._exp_cache
        rtol = self._LOG_DECAY_RTOL
        atol = self._LOG_DECAY_ATOL
        return self._try_find_cache_entry(float_key, key_scaler, cache_dict, rtol, atol)
    
    def _try_find_cache_entry[EntryType](
            self,
            float_key: float,
            key_scaler: float,
            cache_dict: defaultdict[int, list[tuple[float, EntryType]]],
            rtol: float,
            atol: float
        ) -> EntryType | None:
        scaled_key = int(round(float_key * key_scaler))
        target_lists = itertools.chain(
            cache_dict.get(scaled_key - 1, []),
            cache_dict.get(scaled_key, []),
            cache_dict.get(scaled_key + 1, []),
        )
        best_tuple = min(
            target_lists,
            key=lambda tup: abs(tup[0] - float_key),
            default=None,
        )
        if best_tuple is not None:
            best_float_key, best_entry = best_tuple
            from_rtol = max(abs(float_key), abs(best_float_key)) * rtol
            from_atol = atol
            if abs(best_float_key - float_key) <= max(from_rtol, from_atol):
                return best_entry
        return None

    def try_attach(self, state: DampedSine) -> bool:
        """Attaches the renderer to a DampedSine state object.

        If the composed cache renderer is not sealed, attaching will immediately
        populate the cache with the frequency and decay components for reuse,
        and return True for success.

        If the composed cache renderer is sealed, it will only search for
        existing cache entries. If at least one of frequency or decay is found,
        it will return True. Otherwise, it will return False.
        """
        assert isinstance(state, DampedSine)
        if state.has_renderer() and state.require_renderer() is self:
            # Already attached (self), no change.
            return True
        args = DampedSineComposedArgs(info=state)
        if not self._sealed:
            # Always succeeds
            args.sincos_entry = self._populate_sincos(info=state)
            if state.has_decay:
                # Always succeeds, if has_decay
                args.exp_entry = self._populate_exp(info=state)
            # Always succeeds, args is always non-empty
            state.attach_renderer(self, args)
            return True
        else:
            # Sealed.
            freq_entry = self._try_find_sincos(info=state)
            if freq_entry:
                # Only succeeds if freq was cached before.
                args.sincos_entry = freq_entry
            if state.has_decay:
                tau_entry = self._try_find_exp(info=state)
                if tau_entry:
                    # Only succeeds if has_decay and decay was cached before.
                    args.exp_entry = tau_entry
            if args.is_useful():
                # Only detach and attach if at least one of freq or decay succeeded.
                # Otherwise, leave unchanged.
                state.attach_renderer(self, args)
                return True
        return False

    def is_auto_advance(self) -> bool:
        """Always returns True.
         
        NOTE DampedSineComposed can only function if auto_advance is True.
        """
        return True

    def set_auto_advance(self, auto_advance: bool) -> None:
        """Not supported. DampedSineComposed requires auto_advance=True, and cannot be changed.

        Raises RuntimeError if called with auto_advance=False.
        """
        if auto_advance:
            return # silently accept.
        raise RuntimeError("DampedSineComposed requires auto_advance=True, and cannot be changed")

    def render_to(self, state: "DampedSine", out: np.ndarray) -> None:
        """Renders a damped sine wave to the provided output array.
        """
        assert isinstance(state, DampedSine)
        assert isinstance(out, np.ndarray)
        assert out.ndim == 1
        renderer = state.require_renderer()
        if renderer is not self:
            raise RuntimeError("DampedSine state has a different renderer attached")
        args = state.get_renderer_args()
        if not isinstance(args, DampedSineComposedArgs):
            raise RuntimeError("DampedSine state has invalid renderer args")
        none_type = type(None)
        sincos_entry = args.sincos_entry
        if not isinstance(sincos_entry, (SinCosEntry, none_type)):
            raise RuntimeError(f"DampedSineComposedArgs.SinCosEntry corrupted: {type(sincos_entry).__name__}")
        exp_entry = args.exp_entry if state.has_decay else None
        if not isinstance(exp_entry, (ExpEntry, none_type)):
            raise RuntimeError(f"DampedSineComposedArgs.ExpEntry corrupted: {type(exp_entry).__name__}")

        # NOTE state.advance() is delegated to _render_with_cache(), and
        #      possibly further delegated.
        #
        self._render_with_cache(state, out, sincos_entry, exp_entry)

    def _render_with_cache(self, state: DampedSine, out: np.ndarray, sincos_entry: SinCosEntry | None, exp_entry: ExpEntry | None) -> None:
        """Renders a damped sine wave to the provided output array, using the provided cache entries.

        Note:
            - state.advance() is called with the actual number of samples rendered.
        """
        outlen = out.shape[0]
        outstart = 0
        while outstart < outlen:
            # NOTE _render_with_cache_up_to() automatically calls state.advance()
            #      with the actual number of samples rendered.
            rendered = self._render_with_cache_up_to(
                state,
                out[outstart:],
                sincos_entry,
                exp_entry,
            )
            outstart += rendered

    def _render_with_cache_up_to(
            self, 
            state: DampedSine, 
            out: np.ndarray, 
            sincos_entry: SinCosEntry | None, 
            exp_entry: ExpEntry | None,
            ) -> int:
        """Renders a damped sine wave to the provided output array, using the provided
        cache entries.

        NOTE:
            - The output array must be sliced to the proper starting index.
            - Meanwhile, this function handles slicing on sincos_entry and exp_entry.
            - state.advance() is automatically called with the actual number of
                samples rendered.
        """
        outlen = out.shape[0]
        nsamps = min(outlen, self.max_nsamps)
        assert nsamps >= 1 # Guaranteed: out not empty; self.max_nsamps positive.
        outslice = out[:nsamps]

        # NOTE: For the sine part, the output array is always overwritten.
        #
        if sincos_entry is not None:
            self._write_r_sin_cos(
                next_amp=state.next_amp,
                next_phi=state.next_phi,
                sin_part=sincos_entry.sin_part[:nsamps],
                cos_part=sincos_entry.cos_part[:nsamps],
                out=outslice,
            )
        else:
            outslice[:] = state.next_amp * np.sin(state.next_phi + state.omega * self._iota[:nsamps])

        # NOTE: For the decay part, the output array is always multiplied with,
        #       or left unchanged if there is no decay.
        #
        if state.has_decay:
            if exp_entry is not None:
                outslice *= exp_entry.exp_part[:nsamps]
            else:
                outslice *= np.exp(self._iota[:nsamps] * state.log_decay)

        state.advance(nsamps)
        return nsamps

    def render(self, state: "DampedSine", nsamps: int, dtype: Any) -> np.ndarray:
        """Renders a damped sine wave and returns the output array.
        """
        out = np.empty(nsamps, dtype=dtype)
        self.render_to(state, out)
        return out

    def _write_r_sin_cos(self, next_amp: float, next_phi: float, sin_part: Any, cos_part: Any, out: np.ndarray) -> None:
        """Writes the composed sine and cosine parts to the output array.

        Mathematical detail:
            We use the following identity.
                `r * sin(a + b) === r * sin(a) * cos(b) + r * cos(a) * sin(b)`
            With:
                r = next_amp
                a = next_phi
                b = omega * iota

        NOTE To humans and AI assistants:
             DO NOT modify this description or rearrange this code, otherwise a risk exists that
             AI assistants may replace a correct implementation with an incorrect one.
        """
        r_sin_a = next_amp * math.sin(next_phi)
        cos_b = cos_part # same as `np.cos(omega * iota)`
        r_cos_a = next_amp * math.cos(next_phi)
        sin_b = sin_part # same as `np.sin(omega * iota)`
        out[:] = r_sin_a * cos_b + r_cos_a * sin_b
