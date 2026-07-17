import pytest
import math
from functools import cache
from damped_sine import DampedSine


@cache
def get_smoke_samp_rates() -> list[int]:
    return [
        1, 2, 5, 10, 20, 50, 100, 200, 500, 
        1000, 2000, 5000, 10000, 20000, 50000,
        100000, 200000, 500000, 1000000
    ]

@cache
def get_smoke_freqs() -> list[float]:
    return [
        0.0001,
        0.001,
        0.01,
        0.1,
        1.0,
        10.0,
        100.0,
        1000.0,
        10000.0,
        100000.0,
    ]

@cache
def get_smoke_taus() -> list[float]:
    return [
        0.0001,
        0.001,
        0.01,
        0.1,
        1.0,
        10.0,
        100.0,
        1000.0,
        10000.0,
    ]

def test_ctor_good_smoke():
    # Smoke test for constructor with valid parameters.
    samp_rates = get_smoke_samp_rates()
    freqs = get_smoke_freqs()
    taus = get_smoke_taus()
    for samp_rate_hz in samp_rates:
        nyquist_hz = 0.5 * samp_rate_hz
        for freq_hz in freqs:
            if freq_hz >= nyquist_hz:
                # We test above nyquist (expected fails) in test_ctor_bad_nyquist().
                continue
            for tau in taus:
                ds = DampedSine(samp_rate=samp_rate_hz, freq=freq_hz, tau=tau)
                assert math.isclose(ds.samp_rate, samp_rate_hz)
                assert math.isclose(ds.freq, freq_hz)
                assert math.isclose(ds.tau, tau)
                assert math.isclose(ds.omega, 2 * math.pi * freq_hz / samp_rate_hz)
                assert math.isclose(ds.log_decay, -1 / (tau * samp_rate_hz))

def test_ctor_good_args_roundtrip():
    # Verify that the constructor correctly computes omega and log_decay, and that
    # initializing with those omega and log_decay get the original values back.
    samp_rates = get_smoke_samp_rates()
    freqs = get_smoke_freqs()
    taus = get_smoke_taus()
    for samp_rate_hz in samp_rates:
        nyquist_hz = 0.5 * samp_rate_hz
        for freq_hz in freqs:
            if freq_hz >= nyquist_hz:
                # We test above nyquist (expected fails) in test_ctor_bad_nyquist().
                continue
            for tau in taus:
                ds_with_freq_tau = DampedSine(samp_rate=samp_rate_hz, freq=freq_hz, tau=tau)
                assert math.isclose(ds_with_freq_tau.samp_rate, samp_rate_hz)
                assert math.isclose(ds_with_freq_tau.freq, freq_hz)
                assert math.isclose(ds_with_freq_tau.tau, tau)
                assert math.isclose(ds_with_freq_tau.omega, 2 * math.pi * freq_hz / samp_rate_hz)
                assert math.isclose(ds_with_freq_tau.log_decay, -1 / (tau * samp_rate_hz))
                ds_with_omega_ld = DampedSine(samp_rate=samp_rate_hz, omega=ds_with_freq_tau.omega, log_decay=ds_with_freq_tau.log_decay)
                assert math.isclose(ds_with_omega_ld.samp_rate, samp_rate_hz)
                assert math.isclose(ds_with_omega_ld.freq, freq_hz)
                assert math.isclose(ds_with_omega_ld.tau, tau)

def test_ctor_bad_nyquist():
    samp_rates = get_smoke_samp_rates()
    freqs = get_smoke_freqs()
    taus = get_smoke_taus()
    for samp_rate_hz in samp_rates:
        nyquist_hz = 0.5 * samp_rate_hz
        for freq_hz in freqs:
            if freq_hz < nyquist_hz:
                continue
            for tau in taus:
                with pytest.raises(ValueError):
                    DampedSine(samp_rate=samp_rate_hz, freq=freq_hz, tau=tau)

def test_ctor_bad_nyquist_freq_exact():
    # Test the case where freq is exactly at Nyquist, which is invalid.
    samp_rates = get_smoke_samp_rates()
    tau = 1.0
    for samp_rate_hz in samp_rates:
        nyquist_hz = 0.5 * samp_rate_hz
        with pytest.raises(ValueError):
            DampedSine(samp_rate=samp_rate_hz, freq=nyquist_hz, tau=tau)

def test_ctor_bad_nyquist_omega_exact():
    # Test the case where omega is exactly at Nyquist (math.pi), which is invalid.
    samp_rates = get_smoke_samp_rates()
    tau = 1.0
    nyquist_omega = math.pi
    for samp_rate_hz in samp_rates:
        with pytest.raises(ValueError):
            DampedSine(samp_rate=samp_rate_hz, omega=nyquist_omega, tau=tau)

def test_ctor_accepts_either_freq_or_omega():
    samp_rate = 1000
    freq = 10
    omega = 2 * math.pi * freq / samp_rate
    tau = 1.0
    log_decay = -1 / (tau * samp_rate)
    ds_with_freq_tau = DampedSine(samp_rate=samp_rate, freq=freq, tau=tau)
    ds_with_omega_tau = DampedSine(samp_rate=samp_rate, omega=omega, tau=tau)
    ds_with_freq_ld = DampedSine(samp_rate=samp_rate, freq=freq, log_decay=log_decay)
    ds_with_omega_ld = DampedSine(samp_rate=samp_rate, omega=omega, log_decay=log_decay)
    expect_equiv_list = [
        ds_with_freq_tau, 
        ds_with_omega_tau, 
        ds_with_freq_ld, 
        ds_with_omega_ld,
    ]
    for ds in expect_equiv_list:
        assert math.isclose(ds.samp_rate, samp_rate)
        assert math.isclose(ds.freq, freq)
        assert math.isclose(ds.tau, tau)
        assert math.isclose(ds.omega, omega)
        assert math.isclose(ds.log_decay, log_decay)

def test_ctor_accepts_infinite_tau():
    ds = DampedSine(samp_rate=1000, freq=1, tau=math.inf)
    assert math.isclose(ds.samp_rate, 1000)
    assert math.isclose(ds.freq, 1)
    assert math.isinf(ds.tau)
    assert math.isclose(ds.omega, 2 * math.pi * 1 / 1000)
    assert math.isclose(ds.log_decay, 0.0)

def test_ctor_accepts_zero_logdecay():
    ds = DampedSine(samp_rate=1000, freq=1, log_decay=0.0)
    assert math.isclose(ds.samp_rate, 1000)
    assert math.isclose(ds.freq, 1)
    assert math.isinf(ds.tau)
    assert math.isclose(ds.omega, 2 * math.pi * 1 / 1000)
    assert math.isclose(ds.log_decay, 0.0)

def test_ctor_neg_samp_rate():
    with pytest.raises(ValueError):
        DampedSine(samp_rate=-1, freq=1, tau=1)

def test_ctor_zero_samp_rate():
    with pytest.raises(ValueError):
        DampedSine(samp_rate=0, freq=1, tau=1)

def test_ctor_neg_tau():
    with pytest.raises(ValueError):
        DampedSine(samp_rate=1, freq=1, tau=-1)

def test_ctor_zero_tau():
    with pytest.raises(ValueError):
        DampedSine(samp_rate=1, freq=1, tau=0)

def test_ctor_neg_freq():
    with pytest.raises(ValueError):
        DampedSine(samp_rate=1, freq=-1, tau=1)

def test_ctor_zero_freq():
    with pytest.raises(ValueError):
        DampedSine(samp_rate=1, freq=0, tau=1)

def test_advance_long():
    # Smoke test for advance() with large value of nsamps.
    # Note that this does not involve rendering; it just
    # involves scalar calculations.
    samp_rate = 1000
    nsamps = 100000
    freq = 50.0
    tau = 50.0 # appropriate for (nsamps / samp_rate)
    ds = DampedSine(samp_rate=samp_rate, freq=freq, tau=tau)
    ds.advance(nsamps)
    assert math.isclose(ds.next_amp, 0.1353352832366127, rel_tol=1e-6, abs_tol=1e-6)
    # Because freq is 50 Hz, and we have 100000 samples at 1000 Hz, we have 
    # 100000 / 1000 * 50 = 5000.0 cycles exactly.
    # Given we used the default next_phi = 0.0,
    # we expect the next_phi to be close to 0.0 (mod 2*pi) after 5000 cycles,
    # subject to floating point drift. We allow a small tolerance of 1e-6 radians.
    assert abs(ds.next_phi) < 1e-6


if __name__ == "__main__":
    pytest.main([__file__])
