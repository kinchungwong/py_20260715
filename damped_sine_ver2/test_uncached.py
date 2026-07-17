import pytest
import math
import numpy as np
from functools import cache

from damped_sine import DampedSine
from uncached import DampedSineUncached


def _count_positive(arr: np.ndarray) -> int:
    """Checks if the array is mostly positive, allowing for some small
    numerical errors.
    """
    # eps = np.finfo(float).eps
    # positive_count = int(np.count_nonzero(arr > -eps))
    # return positive_count >= 0.99 * len(arr)
    values = arr.tolist()
    pos_count = sum(1 for v in values if v > 0)
    return pos_count

def _count_negative(arr: np.ndarray) -> int:
    return _count_positive(-arr)

def _count_abs_not_exceeding(arr: np.ndarray, threshold: float) -> int:
    """Counts the number of elements in the array whose absolute value does not exceed the threshold.
    """
    values = arr.tolist()
    count = sum(1 for v in values if abs(v) <= threshold)
    return count


def test_damped_sine_rendering_smoke():
    # Simulated duration of one second, with 1000 samples per second,
    # frequency of 1 Hz (simulation consist of exactly one cycle),
    # and a decay time constant of 100 seconds, which is a very slow
    # decay.
    #
    samp_rate = 1000
    nsamps = samp_rate
    freq = 1
    tau = 100.0
    ds = DampedSine(samp_rate=samp_rate, freq=freq, tau=tau)
    assert math.isclose(ds.samp_rate, samp_rate)
    assert math.isclose(ds.freq, freq)
    assert math.isclose(ds.tau, tau)
    assert math.isclose(ds.omega, 2 * math.pi * freq / samp_rate)
    assert math.isclose(ds.log_decay, -1 / (tau * samp_rate))
    uncached = DampedSineUncached()
    assert uncached.try_attach(ds)
    assert ds.has_renderer()
    out = ds.render(nsamps)
    assert len(out) == nsamps
    #
    # Quadrant sign and first derivative sign test.
    #
    test_quadrants = [
        (out[0:250], 1, 1),
        (out[250:500], 1, -1),
        (out[500:750], -1, -1),
        (out[750:1000], -1, 1),
    ]
    for test_slice, expected_sign, expected_derivative_sign in test_quadrants:
        if (expected_sign > 0):
            assert _count_positive(test_slice) >= 0.99 * len(test_slice)
        elif (expected_sign < 0):
            assert _count_negative(test_slice) >= 0.99 * len(test_slice)

        diffs = np.diff(test_slice)

        if (expected_derivative_sign > 0):
            assert _count_positive(diffs) >= 0.99 * len(diffs)
        elif (expected_derivative_sign < 0):
            assert _count_negative(diffs) >= 0.99 * len(diffs)


def test_damped_sine_rendering_tau():
    # Simulated duration of one second, with 1000 samples per second,
    #
    # A very low frequency of 0.001 Hz (nearly DC),
    # next_phi of pi/2 (90 degrees, i.e. starting at the peak of the sine wave),
    # next_amp of 1.0 (starting at full amplitude),
    #
    # and a fast decay time constant of 0.1 seconds,
    # so that each passing of 0.1 seconds (100 samples),
    # it is decayed by another factor of 1/e (~0.3679).
    #
    samp_rate = 1000
    nsamps = samp_rate
    freq = 0.001
    tau = 0.1
    start_phi = math.pi * 0.5
    start_amp = 1.0
    ds = DampedSine(samp_rate=samp_rate, freq=freq, tau=tau, next_phi=start_phi, next_amp=start_amp)
    assert math.isclose(ds.samp_rate, samp_rate)
    assert math.isclose(ds.freq, freq)
    assert math.isclose(ds.tau, tau)
    assert math.isclose(ds.omega, 2 * math.pi * freq / samp_rate)
    assert math.isclose(ds.log_decay, -1 / (tau * samp_rate))
    uncached = DampedSineUncached()
    assert uncached.try_attach(ds)
    assert ds.has_renderer()
    out = ds.render(nsamps)
    assert len(out) == nsamps
    #
    # Composite amplitude test of first sample.
    # This verifies our start_phi and start_amp are set correctly.
    #
    assert math.isclose(float(out[0]), 1.0, rel_tol=1e-4, abs_tol=1e-4)
    #
    # Decay test. The amplitude should decay by a factor of 1/e 
    # every 0.1 seconds (100 samples).
    #
    # Each slice is annotated with the upper envelope and the
    # lower envelope. Because start_amp is 1.0, we omit from here.
    #
    # Because we use a very low frequency, and we started off
    # with phase of pi/2 (90 degrees), the very slow descending
    # of the q1 (second quadrant) of the sine wave would be barely
    # noticeable, and it will not affect our envelope-based test
    # (via _count_abs_not_exceeding()). The resulting deviation
    # would have contributed to at most 2 samples out of 100 samples
    # (via rounding errors at the start and end of the slice),
    # therefore we simply ignore one sample from each end of the
    # slice.
    #
    test_slices = [
        (out[1:99], math.exp(0), math.exp(-1)),
        (out[101:199], math.exp(-1), math.exp(-2)),
        (out[201:299], math.exp(-2), math.exp(-3)),
        # omitted
        (out[801:899], math.exp(-8), math.exp(-9)),
        (out[901:999], math.exp(-9), math.exp(-10)),
    ]
    for test_slice, upper_env, lower_env in test_slices:
        # Ignoring the first and last samples from each slice
        # (one-tenth of the run),
        #
        # All samples (except the first and last) are decaying below upper_env:
        #
        assert _count_abs_not_exceeding(test_slice, upper_env) == len(test_slice)
        #
        # But none had yet decayed to below lower_env:
        #
        assert _count_abs_not_exceeding(test_slice, lower_env) == 0


def test_uncached_long():
    # Smoke test for uncached renderer with long duration.
    samp_rate = 1000
    nsamps = 100000
    freq = 50.0
    tau = 50.0 # appropriate for (nsamps / samp_rate)
    ds = DampedSine(samp_rate=samp_rate, freq=freq, tau=tau)
    uncached = DampedSineUncached()
    assert uncached.try_attach(ds)
    assert ds.has_renderer()
    out = ds.render(nsamps)
    assert len(out) == nsamps
    assert math.isclose(ds.next_amp, 0.1353352832366127, rel_tol=1e-6, abs_tol=1e-6)
    # Because freq is 50 Hz, and we have 100000 samples at 1000 Hz, we have 
    # 100000 / 1000 * 50 = 5000.0 cycles exactly.
    # Given we used the default next_phi = 0.0,
    # we expect the next_phi to be close to 0.0 (mod 2*pi) after 5000 cycles,
    # subject to floating point drift. We allow a small tolerance of 1e-6 radians.
    assert abs(ds.next_phi) < 1e-6

if __name__ == "__main__":
    pytest.main([__file__])
