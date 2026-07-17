import pytest
import math
import numpy as np
from functools import cache
from time import monotonic_ns
import cProfile
import pstats

from damped_sine import DampedSine
from uncached import DampedSineUncached
from composed import DampedSineComposed


def _settings():
    class Settings:
        samp_rate = 44100
        tone_count = 100
        duration = 100.0 # seconds
        total_nsamps = int(round(samp_rate * duration))
        nsamps = 512 # per call
        freq_min = 20.0
        freq_max = 20000.0
    s = Settings()
    assert s.tone_count >= 2, "tone_count must be at least 2 to define a frequency range"
    return s

def _init_tones() -> list[DampedSine]:
    """Initializes a list of DampedSine objects with different frequencies.
    """
    settings = _settings()
    samp_rate = settings.samp_rate
    tone_count = settings.tone_count
    freq_min = settings.freq_min
    freq_max = settings.freq_max
    tones: list[DampedSine] = []
    for i in range(tone_count):
        freq = freq_min + (i / (tone_count - 1)) * (freq_max - freq_min)
        tau = float(settings.duration)
        ds = DampedSine(samp_rate=samp_rate, freq=freq, tau=tau)
        tones.append(ds)
    return tones

@pytest.mark.perf
def test_perf_composed():
    """Performance test for composed damped sine renderer.
    """
    settings = _settings()
    samp_rate = settings.samp_rate
    total_nsamps = settings.total_nsamps
    nsamps = settings.nsamps

    tones = _init_tones()

    composed = DampedSineComposed(max_nsamps=nsamps)

    t0 = monotonic_ns()

    for tone in tones:
        assert composed.try_attach(tone)

    t1 = monotonic_ns()

    out = np.empty(nsamps, dtype=np.float64)

    t2 = monotonic_ns()

    rendered_nsamps = 0
    while rendered_nsamps < total_nsamps:
        nsamps_this = min(nsamps, total_nsamps - rendered_nsamps)
        for tone in tones:
            tone.render_to(out=out[:nsamps_this])
        rendered_nsamps += nsamps_this

    t3 = monotonic_ns()

    elapsed_attach_ms = (t1 - t0) / 1e6
    elapsed_render_ms = (t3 - t2) / 1e6
    print(f"Sample rate: {samp_rate} Hz")
    print(f"Duration: {settings.duration} seconds")
    print(f"Total samples rendered: {rendered_nsamps}")
    print(f"Total tones rendered: {len(tones)}")
    print(f"Elapsed time for attaching tones: {elapsed_attach_ms:.3f} ms")
    print(f"Elapsed time for rendering: {elapsed_render_ms:.3f} ms")

if __name__ == "__main__":
    if False:
        exit(pytest.main([__file__]))
    elif True:
        perf_outfile = "test_perf_composed.pycprof"
        sortkeys = (pstats.SortKey.CUMULATIVE, pstats.SortKey.TIME)
        cProfile.run("test_perf_composed()", filename=perf_outfile)
        p = pstats.Stats(perf_outfile)
        p.strip_dirs()
        p.sort_stats(*sortkeys)
        p.print_stats(100)
    else:
        test_perf_composed()
