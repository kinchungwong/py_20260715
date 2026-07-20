import math
import pytest
from time import perf_counter_ns
import cProfile
import os
import numpy as np

_cur_dir_name = os.path.dirname(__file__)
print(_cur_dir_name)
os.path.abspath(_cur_dir_name)
os.path.abspath(_cur_dir_name + "/" + "damped_sine_impl")
import damped_sine_impl as impl

test_reference_funcs_f64_cases = [
    (1024.0, 1024, 128.0, 0.5, 1.0, 0.0),
    (1024.0, 1024, 256.0, 0.5, 1.0, 0.0),
    (1024.0, 1024, 512.0, 0.5, 1.0, 0.0),
    (44100.0, 1024, 11025.0, 1000.0, 1.0, 0.0 * math.pi),
    (44100.0, 1024, 11025.0, 1000.0, 1.0, 0.1 * math.pi),
    (44100.0, 1024, 11025.0, 1000.0, 1.0, 0.5 * math.pi),
    (44100.0, 1024, 11025.0, 1000.0, 1.0, 1.0 * math.pi),
    (44100.0, 1024, 11025.0, 1000.0, 1.0, 1.5 * math.pi),
    (44100.0, 1024, 11025.0, 1000.0, 1.0, 1.875 * math.pi),
    (44100.0, 1024, 22050.0, 0.005, 1.0, 0.0 * math.pi),
    (44100.0, 1024, 22050.0, 0.005, 1.0, 0.1 * math.pi),
    (44100.0, 1024, 22050.0, 0.005, 1.0, 0.5 * math.pi),
    (44100.0, 1024, 22050.0, 0.005, 1.0, 1.0 * math.pi),
    (44100.0, 1024, 22050.0, 0.005, 1.0, 1.5 * math.pi),
    (44100.0, 1024, 22050.0, 0.005, 1.0, 2.0 * math.pi),
    (44100.0, 65536, 22050.0, 1000.0, 1.0, 47.0),
    (44100.0, 65536, 22050.0, 1000.0, 0.9, 47.0),
    (44100.0, 65536, 22050.0, 1000.0, 0.1, 47.0),
    (44100.0, 65536, 22050.0, 0.005, 1.0, 47.0),
    (44100.0, 65536, 22050.0, 0.005, 0.9, 47.0),
    (44100.0, 65536, 22050.0, 0.005, 0.1, 47.0),
]


@pytest.mark.parametrize(
    "samp_rate, nsamps, freq, tau, next_amp, next_phi",
    test_reference_funcs_f64_cases,
)
def test_reference_funcs_f64(
        samp_rate: float,
        nsamps: int,
        freq: float,
        tau: float,
        next_amp: float,
        next_phi: float,
) -> None:

    omega = 2.0 * math.pi * freq / samp_rate
    log_decay = -1.0 / (tau * samp_rate)

    next_phi = next_phi % (2.0 * math.pi)

    partial = impl.PartialOne(omega=omega, log_decay=log_decay)
    state = impl.StateOne(next_amp=next_amp, next_phi=next_phi)

    fn_scalar = impl.render_py_1_1
    fn_py = impl.render_py_1_n
    fn_np = impl.render_np_1_n

    out_py = np.empty(nsamps, dtype=np.float64)
    out_np = np.empty(nsamps, dtype=np.float64)
    fn_py(nsamps, partial, state, out_py)
    fn_np(nsamps, partial, state, out_np)

    list_py = out_py.tolist()
    list_np = out_np.tolist()

    min_py = math.inf
    max_py = -math.inf
    min_np = math.inf
    max_np = -math.inf
    maxerror_py = 0.0
    maxerror_np = 0.0

    for n in range(nsamps):
        val_scalar = fn_scalar(n, partial, state)
        val_py = list_py[n]
        val_np = list_np[n]

        min_py = min(min_py, val_py)
        max_py = max(max_py, val_py)
        min_np = min(min_np, val_np)
        max_np = max(max_np, val_np)

        maxerror_py = max(maxerror_py, abs(val_scalar - val_py))
        maxerror_np = max(maxerror_np, abs(val_scalar - val_np))

    print("\n\n\n\n")
    print(f"samp_rate={samp_rate:.0f}, nsamps={nsamps}, freq={freq:.3f}, tau={tau:.3f}, omega={omega:.6f}, log_decay={log_decay:.9f}, next_amp={next_amp:.6f}, next_phi={next_phi:.3f}")
    print(f"  min_py = {min_py}")
    print(f"  max_py = {max_py}")
    print(f"  min_np = {min_np}")
    print(f"  max_np = {max_np}")
    print(f"  maxerror_py = {maxerror_py}")
    print(f"  maxerror_np = {maxerror_np}")

    #
    # Lossless 16-bit PCM has a tolerance of (2.0/65536),
    # assuming output range is [-1.0, 1.0], and internal 
    # calculations are performed in 64-bit floats.
    #
    # NOTE These claims are subject to review and correction.
    #
    out_tol_pcm16_fp64 = 2.0 / 65536
    assert maxerror_py <= out_tol_pcm16_fp64
    assert maxerror_np <= out_tol_pcm16_fp64


def test_render_np_m1_cached_fp32():
    """Test for RenderNpMCached with M=1, using 32-bit floats for rendering.
    """

    samp_rate = 44100.0
    nsamps = 1024
    freq = 11025.0
    tau = nsamps / samp_rate # Decays exp(-1), or (1/e) within the rendered length
    next_amp = 1.0
    next_phi = 0.0

    omega = 2.0 * math.pi * freq / samp_rate
    log_decay = -1.0 / (tau * samp_rate)

    partials = impl.PartialM(omega=np.array([omega]), log_decay=np.array([log_decay]))
    state = impl.StateM(next_amp=np.array([next_amp]), next_phi=np.array([next_phi]))

    # Rendering is performed in 32-bit floats, as much as possible,
    # to simulate the behavior of a real-time audio engine.
    spares = impl.SpareST(max_partials_per_note=1, max_nsamps=nsamps, dtype=np.float32)
    render_m_cached = impl.RenderNpMCached(nsamps, partials, np.float32, spares)
    out_m_cached = np.empty(nsamps, dtype=np.float32)
    t0 = perf_counter_ns()
    render_m_cached.render(state, out_m_cached)
    t1 = perf_counter_ns()

    # Reference output is rendered in 64-bit floats.
    out_expected = np.empty(nsamps, dtype=np.float64)
    impl.render_np_1_n(nsamps, impl.PartialOne(omega=omega, log_decay=log_decay), impl.StateOne(next_amp=next_amp, next_phi=next_phi), out_expected)
   
    vec_delta = out_m_cached.astype(np.float64) - out_expected
    max_delta = np.max(np.abs(vec_delta))

    print("\n\n\n\n")
    print("test_render_np_m1_cached_fp32:")
    print(f"  max(abs(out_m_cached - out_expected)) = {max_delta}")

    assert max_delta <= 2.0 / 65536

    print("  render time (ns):", t1 - t0)

if __name__ == "__main__":
    exit(pytest.main([__file__, "-v", "-s"]))
