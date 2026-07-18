import math
from typing import Final
import numpy as np
from midi_piano_model import MidiPianoModelArgs, MidiPianoPartial


class MidiPianoPreRender:
    model_args: Final[MidiPianoModelArgs]
    partials: Final[list[MidiPianoPartial]]
    num_partials: Final[int]
    num_tones: Final[int]
    max_nsamps: Final[int]
    dtype: Final[np.typing.DTypeLike]
    cache_shape: Final[tuple[int, int]]

    def __init__(
            self, 
            model_args: MidiPianoModelArgs, 
            partials: list[MidiPianoPartial], 
            max_nsamps: int, 
            dtype: np.typing.DTypeLike = np.float64,
            ) -> None:
        # TODO clone model_args.
        if not isinstance(model_args, MidiPianoModelArgs):
            raise TypeError(f"model_args must be a MidiPianoModelArgs, got {type(model_args).__name__}")
        partials = list(partials)
        if not all(isinstance(p, MidiPianoPartial) for p in partials):
            raise TypeError(f"partials must be a list of MidiPianoPartial, got {type(partials).__name__}")
        self.model_args = model_args
        self.partials = partials
        self.num_partials = len(partials)
        self.num_tones = self.num_partials * 2
        self.max_nsamps = max_nsamps
        self.dtype = dtype
        self.cache_shape = (self.num_tones, self.max_nsamps)

    def build_cache_ver_1(self) -> np.ndarray:
        model_args = self.model_args
        sample_rate = model_args.sample_rate
        partials = self.partials
        num_partials = self.num_partials
        num_tones = self.num_tones
        assert num_tones == num_partials * 2
        max_nsamps = self.max_nsamps
        dtype = self.dtype
        result = np.empty((num_tones, max_nsamps), dtype=dtype)
        iota = np.arange(max_nsamps)
        for idx, partial in enumerate(partials):
            freq = partial.partial_freq
            tau = partial.partial_tau
            amp = partial.partial_amp
            omega = 2.0 * math.pi * freq / sample_rate
            log_decay = -1.0 / (tau * sample_rate)
            ### sinexp ###
            result[2*idx,     :] = amp * np.sin(omega * iota) * np.exp(log_decay * iota)
            ### cosexp ###
            result[2*idx + 1, :] = amp * np.cos(omega * iota) * np.exp(log_decay * iota)
        return result

    def build_cache_ver_2(self) -> np.ndarray:
        model_args = self.model_args
        sample_rate = model_args.sample_rate
        partials = self.partials
        num_partials = self.num_partials
        num_tones = self.num_tones
        assert num_tones == num_partials * 2
        max_nsamps = self.max_nsamps
        dtype = self.dtype
        iota = np.arange(max_nsamps)
        result = np.empty((num_tones, max_nsamps), dtype=dtype)
        buf1 = np.empty((max_nsamps,), dtype=dtype)
        buf2 = np.empty((max_nsamps,), dtype=dtype)
        buf3 = np.empty((max_nsamps,), dtype=dtype)
        np_multiply = np.multiply
        np_exp = np.exp
        np_sin = np.sin
        np_cos = np.cos
        buf1_fill = buf1.fill
        buf2_fill = buf2.fill
        for idx, partial in enumerate(partials):
            freq = partial.partial_freq
            tau = partial.partial_tau
            amp = partial.partial_amp
            omega = 2.0 * math.pi * freq / sample_rate
            log_decay = -1.0 / (tau * sample_rate)
            ### Reference code
            #
            # sinexp = amp * np.sin(omega * iota) * np.exp(log_decay * iota)
            # cosexp = amp * np.cos(omega * iota) * np.exp(log_decay * iota)
            # result[2*idx,     :] = sinexp
            # result[2*idx + 1, :] = cosexp
            #
            ### Modified code
            #
            ### buf1 = amp * exp(log_decay * iota) ###
            buf1_fill(log_decay)
            np_multiply(buf1, iota, out=buf1)
            np_exp(buf1, out=buf1)
            np_multiply(buf1, amp, out=buf1)
            ### buf2 = omega * iota ###
            buf2_fill(omega)
            np_multiply(buf2, iota, out=buf2)
            ### buf3 = sin(buf2) * buf1 ###
            np_sin(buf2, out=buf3)
            np_multiply(buf3, buf1, out=buf3)
            ### sinexp = buf3 ###
            result[2*idx,     :] = buf3
            ### reuse buf3
            ### buf3 = cos(buf2) * buf1 ###
            np_cos(buf2, out=buf3)
            np_multiply(buf3, buf1, out=buf3)
            ### cosexp = buf3 ###
            result[2*idx + 1, :] = buf3
        return result
    
    def build_cache_ver_3(self) -> np.ndarray:
        model_args = self.model_args
        sample_rate = model_args.sample_rate
        partials = self.partials
        num_partials = self.num_partials
        num_tones = self.num_tones
        assert num_tones == num_partials * 2
        max_nsamps = self.max_nsamps
        dtype = self.dtype
        result = np.empty((num_tones, max_nsamps), dtype=dtype)
        iota = np.arange(max_nsamps)
        amps = np.array([p.partial_amp for p in partials], dtype=dtype)
        omegas = np.array([2.0 * math.pi * p.partial_freq / sample_rate for p in partials], dtype=dtype)
        log_decays = np.array([-1.0 / (p.partial_tau * sample_rate) for p in partials], dtype=dtype)
        ###
        outer_exp = np.empty((num_partials, max_nsamps), dtype=dtype)
        np.outer(log_decays, iota, out=outer_exp)
        np.exp(outer_exp, out=outer_exp)
        np.multiply(outer_exp, amps[:, np.newaxis], out=outer_exp)
        ###
        outer_phi = np.empty((num_partials, max_nsamps), dtype=dtype)
        np.outer(omegas, iota, out=outer_phi)
        ###
        outer_sincosexp = np.empty((2, num_partials, max_nsamps), dtype=dtype)
        np.sin(outer_phi, out=outer_sincosexp[0, :, :])
        np.cos(outer_phi, out=outer_sincosexp[1, :, :])
        np.multiply(outer_sincosexp, outer_exp[np.newaxis, :, :], out=outer_sincosexp)
        result = outer_sincosexp.reshape((num_tones, max_nsamps))
        return result

if __name__ == "__main__":
    from midi_piano_model import MidiPianoModel
    from time import perf_counter_ns

    model_args = MidiPianoModelArgs()
    model = MidiPianoModel(model_args)

    max_num_partials = len(model.all_partials)
    print("Max number of partials:", max_num_partials)

    num_partials = max_num_partials
    partials_to_build = model.all_partials[:num_partials]

    max_nsamps = 1024
    print(f"Building cache for {len(partials_to_build)} partials, max_nsamps={max_nsamps}")
    pre_render = MidiPianoPreRender(model_args, partials_to_build, max_nsamps=max_nsamps, dtype=np.float32)

    # pre_render_build_cache = pre_render.build_cache_ver_1
    # pre_render_build_cache = pre_render.build_cache_ver_2
    pre_render_build_cache = pre_render.build_cache_ver_3

    t0 = perf_counter_ns()
    np_cache = pre_render_build_cache()
    t1 = perf_counter_ns()
    cache_shape = np_cache.shape
    print("Cache shape:", cache_shape)
    print(f"  Cache build time [warmup]: {((t1 - t0) * 1e-6):.2f} ms")

    for trial_id in range(10):
        t0 = perf_counter_ns()
        np_cache = pre_render_build_cache()
        t1 = perf_counter_ns()
        print(f"  Cache build time [trial={trial_id}]: {((t1 - t0) * 1e-6):.2f} ms")

    # contiguous check
    print("Is a row of cache contiguous?", np_cache[1,:].flags["C_CONTIGUOUS"])
    print("Is a column of cache contiguous?", np_cache[:,1].flags["C_CONTIGUOUS"])
