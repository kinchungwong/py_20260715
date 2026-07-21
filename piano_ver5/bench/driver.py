"""Measurement engine: correctness gate, adaptive timing (with a jitter
distribution), and memory-churn measurement.

Timing philosophy (mirrors the deep-research reports): perf_counter_ns, one
reading per block so we get a real distribution; a warm-up window discarded;
min-of-repeat-means as the central number; p99/max reported because real-time
audio is worst-case bound. gc is left ENABLED throughout so jitter reflects the
GC pauses a live callback would actually suffer.
"""
from __future__ import annotations

import gc
import math
import tracemalloc
from dataclasses import dataclass, field
from time import perf_counter_ns

import numpy as np

import audio_params as ap
from partial_gen import VoiceSpec, recast_voice
from strategies import _Base, STRATEGIES


# ---------------------------------------------------------------------------
# Correctness: independent f64 analytic oracle for block 0
# ---------------------------------------------------------------------------
def reference_block0(voices: list[VoiceSpec], nsamps: int) -> np.ndarray:
    """Direct time-domain damped-sine sum in float64 -- trusts no strategy.
    Mirrors ver4's _reference_mix."""
    n = np.arange(nsamps, dtype=np.float64)
    acc = np.zeros(nsamps, dtype=np.float64)
    for v in voices:
        omega = np.asarray(v.partials.omega, dtype=np.float64)
        logd = np.asarray(v.partials.log_decay, dtype=np.float64)
        amp = np.asarray(v.state0.next_amp, dtype=np.float64)
        phi = np.asarray(v.state0.next_phi, dtype=np.float64)
        for k in range(v.M):
            acc += amp[k] * np.exp(logd[k] * n) * np.sin(phi[k] + omega[k] * n)
    return acc


def correctness_gate(master_voices: list[VoiceSpec], nsamps: int) -> dict[str, float]:
    """Assert every strategy's block-0 output matches the f64 oracle at 16-bit
    tolerance. Proves RenderNpMCached is correct for this M (beyond the M=1
    test) and that naive/batched replicate the pairing exactly."""
    ref = reference_block0(master_voices, nsamps)
    peak = float(np.max(np.abs(ref)))
    # Tolerance scales with signal magnitude: an 88-voice sum has peak ~10-30, so
    # honest float32 accumulation error scales up too. A sin/cos-swap or pairing
    # bug produces error of order `peak` (~100% wrong), far above this bound.
    tol = ap.CORRECTNESS_TOL * max(1.0, peak)
    errors: dict[str, float] = {}
    for name, cls in STRATEGIES.items():
        voices = [recast_voice(v, np.float32) for v in master_voices]
        strat = cls(voices, nsamps, np.float32, "level")
        mix = np.zeros(nsamps, dtype=np.float32)
        strat._render(mix)  # block 0, no advance
        err = float(np.max(np.abs(mix.astype(np.float64) - ref)))
        errors[name] = err
        if err > tol:
            raise AssertionError(
                f"correctness FAIL: {name} at M={master_voices[0].M} "
                f"V={len(master_voices)} N={nsamps}: max|err|={err:.3e} "
                f"> tol {tol:.3e} (peak |ref|={peak:.3f})"
            )
    return errors


# ---------------------------------------------------------------------------
# Timing
# ---------------------------------------------------------------------------
@dataclass
class CellResult:
    strategy: str
    dtype_label: str      # "f32" | "f64" | "f32-trap"
    mode: str             # "level" | "decay"
    N: int
    M: int
    V: int
    us_min: float = 0.0
    us_p50: float = 0.0
    us_p99: float = 0.0
    us_max: float = 0.0
    us_mean: float = 0.0
    us_std: float = 0.0
    us_per_voice: float = 0.0
    pct_p50: float = 0.0
    pct_p99: float = 0.0
    max_voices_fit: float = 0.0
    timed_blocks: int = 0
    repeats: int = 0
    # filled in during post-processing
    speedup_vs_naive: float = field(default=float("nan"))
    speedup_vs_pernote: float = field(default=float("nan"))


def _time_blocks(strat: _Base, mix: np.ndarray, warmup: int, timed: int) -> np.ndarray:
    for _ in range(warmup):
        strat.render_block(mix)
    samples = np.empty(timed, dtype=np.int64)
    for i in range(timed):
        t0 = perf_counter_ns()
        strat.render_block(mix)
        samples[i] = perf_counter_ns() - t0
    return samples


def _clamp(x: float, lo: int, hi: int) -> int:
    return int(max(lo, min(hi, round(x))))


def run_cell(
    strategy_cls,
    master_voices: list[VoiceSpec],
    nsamps: int,
    voice_dtype,
    out_dtype,
    mode: str,
    dtype_label: str,
) -> CellResult:
    """Pilot -> adaptive block count -> repeats -> summarize one grid cell."""
    V = len(master_voices)
    M = master_voices[0].M
    voices = [recast_voice(v, voice_dtype) for v in master_voices]

    def make():
        return strategy_cls(voices, nsamps, voice_dtype, mode)

    # Pilot: estimate per-block cost to size the timed window, warmup & repeats.
    strat = make()
    mix = np.zeros(nsamps, dtype=out_dtype)
    pilot = _time_blocks(strat, mix, warmup=30, timed=10)
    est_us = max(float(np.median(pilot)) * 1e-3, 0.5)
    # ~0.8 s of timing per repeat, bounded; cheaper cells get more blocks and
    # more repeats, the expensive naive cells get fewer so the sweep stays quick.
    timed = _clamp(800_000.0 / est_us, 50, 5000)
    warmup = 128 if est_us < 500 else (32 if est_us < 5000 else 12)
    repeats = 4 if est_us < 800 else (3 if est_us < 20000 else 2)

    best_mean = math.inf
    best_samples: np.ndarray | None = None
    for _ in range(repeats):
        strat = make()  # fresh objects each repeat -> capture warmth variance
        mix = np.zeros(nsamps, dtype=out_dtype)
        s = _time_blocks(strat, mix, warmup=warmup, timed=timed)
        m = float(s.mean())
        if m < best_mean:
            best_mean = m
            best_samples = s

    assert best_samples is not None
    us = best_samples.astype(np.float64) * 1e-3
    deadline = ap.deadline_us(nsamps)
    p50 = float(np.percentile(us, 50))
    p99 = float(np.percentile(us, 99))
    per_voice = p50 / V
    res = CellResult(
        strategy=strategy_cls.name, dtype_label=dtype_label, mode=mode,
        N=nsamps, M=M, V=V,
        us_min=float(us.min()), us_p50=p50, us_p99=p99, us_max=float(us.max()),
        us_mean=float(us.mean()), us_std=float(us.std()),
        us_per_voice=per_voice,
        pct_p50=p50 / deadline * 100.0,
        pct_p99=p99 / deadline * 100.0,
        max_voices_fit=math.floor(deadline / (p99 / V)) if p99 > 0 else 0.0,
        timed_blocks=timed, repeats=repeats,
    )
    return res


# ---------------------------------------------------------------------------
# Memory churn
# ---------------------------------------------------------------------------
@dataclass
class MemResult:
    strategy: str
    N: int
    M: int
    V: int
    analytic_bytes_block: int
    transient_bytes_block: int   # tracemalloc peak of one block (python-object
                                 # overhead; numpy DATA buffers are not traced)
    gc_gen0: int
    gc_gen1: int
    gc_gen2: int
    blocks: int


def measure_memory(
    strategy_cls,
    master_voices: list[VoiceSpec],
    nsamps: int,
    voice_dtype,
    out_dtype,
    blocks: int = 10_000,
) -> MemResult:
    """Churn over a long run. Two signals: (1) analytic bytes/block from known
    array shapes; (2) gc collection counts over `blocks` blocks = the real-time
    GC-pressure KPI. tracemalloc's single-block peak is reported too but only
    sees Python-object headers, not numpy data buffers (which use malloc)."""
    voices = [recast_voice(v, voice_dtype) for v in master_voices]
    strat = strategy_cls(voices, nsamps, voice_dtype, "level")
    mix = np.zeros(nsamps, dtype=out_dtype)
    for _ in range(256):
        strat.render_block(mix)

    tracemalloc.start()
    strat.render_block(mix)
    tracemalloc.reset_peak()
    b0 = tracemalloc.get_traced_memory()[0]
    strat.render_block(mix)
    _, peak1 = tracemalloc.get_traced_memory()
    transient = max(0, peak1 - b0)
    tracemalloc.stop()

    gc.collect()
    gc0 = [s["collections"] for s in gc.get_stats()]
    for _ in range(blocks):
        strat.render_block(mix)
    gc.collect()
    gc1 = [s["collections"] for s in gc.get_stats()]
    d = [a - b for b, a in zip(gc0, gc1)]

    return MemResult(
        strategy=strategy_cls.name, N=nsamps, M=voices[0].M, V=len(voices),
        analytic_bytes_block=strat.analytic_bytes_per_block(),
        transient_bytes_block=transient,
        gc_gen0=d[0], gc_gen1=d[1], gc_gen2=d[2], blocks=blocks,
    )
