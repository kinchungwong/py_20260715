"""Physically-plausible partial/state generation for the benchmark.

Every voice is a sum of inharmonic damped-sine partials built with the ver3 /
older-project recipe (roll-off amplitudes that keep a strong fundamental).
Physical quantities are computed in float64, then cast to the caller's `dtype`
at the very end -- so `dtype` alone decides whether the render KERNEL sees
float32 or float64 arrays (this is exactly the lever behind the upcast gotcha).
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

import damped_sine_impl as impl
import audio_params as ap


@dataclass(frozen=True)
class VoiceSpec:
    """One voice: its immutable partial definition, its attack-time state, and
    the per-block advance factors (so the driver can evolve state without
    recomputing anything)."""
    midi: int
    M: int
    partials: impl.PartialM      # omega (M,), log_decay (M,)   -- kernel dtype
    state0: impl.StateM          # next_amp (M,), next_phi (M,) -- kernel dtype
    decay_per_block: np.ndarray  # (M,) = exp(log_decay * N)     -- kernel dtype
    phase_incr: np.ndarray       # (M,) = (omega * N) mod 2pi    -- kernel dtype


def midi_to_f1(midi: int) -> float:
    """Fundamental frequency (Hz) of a piano key in 12-TET, A4=440."""
    return 440.0 * 2.0 ** ((midi - 69) / 12.0)


def _partial_freqs(f1: float, k: np.ndarray) -> np.ndarray:
    """Inharmonic partial frequencies f_k = f1 * k * sqrt(1 + B*k^2)."""
    return f1 * k * np.sqrt(1.0 + ap.INHARM_B * k * k)


def realistic_partial_count(midi: int, cap: int = 40) -> int:
    """How many partials this key actually supports: those below both the
    audible ceiling and Nyquist, capped at `cap`. Low keys -> ~cap; high keys
    -> a handful. Used for the honest "mixed 88-key" estimate."""
    k = np.arange(1, cap + 1, dtype=np.float64)
    f_k = _partial_freqs(midi_to_f1(midi), k)
    omega = 2.0 * math.pi * f_k / ap.FS
    keep = (f_k < ap.AUDIBLE_HZ) & (omega < ap.NYQUIST)
    return int(np.count_nonzero(keep))


def build_partials(
    midi: int,
    M: int,
    nsamps: int,
    *,
    dtype,
    rng: np.random.Generator,
    deterministic_phase: bool = False,
) -> VoiceSpec:
    """Build a VoiceSpec of exactly M partials for `midi`.

    Raises ValueError if any partial would exceed Nyquist -- the caller is
    expected to pick a pitch (via `pitch_plan`) whose window keeps all M
    partials valid.
    """
    k = np.arange(1, M + 1, dtype=np.float64)
    f1 = midi_to_f1(midi)
    f_k = _partial_freqs(f1, k)
    omega = 2.0 * math.pi * f_k / ap.FS

    if float(omega.max()) >= ap.NYQUIST:
        bad = int(np.argmax(omega >= ap.NYQUIST)) + 1
        raise ValueError(
            f"midi={midi} M={M}: partial {bad} has omega={omega.max():.4f} "
            f">= pi (f={f_k.max():.1f} Hz > Nyquist {ap.FS/2:.0f}). "
            f"Pick a lower pitch for this M."
        )

    tau = ap.TAU_FUND / (1.0 + ap.TAU_PCOEF * (k - 1.0))
    log_decay = -1.0 / (tau * ap.FS)

    amp = 1.0 / np.power(k, ap.AMP_PCOEF)
    amp /= amp.sum()  # normalize total attack amplitude to 1.0

    if deterministic_phase:
        phi = np.zeros(M, dtype=np.float64)
    else:
        phi = rng.uniform(0.0, 2.0 * math.pi, size=M)

    # Per-block advance factors, precomputed in f64 then cast.
    decay_per_block = np.exp(log_decay * nsamps)
    phase_incr = (omega * nsamps) % (2.0 * math.pi)

    return VoiceSpec(
        midi=midi,
        M=M,
        partials=impl.PartialM(
            omega=omega.astype(dtype),
            log_decay=log_decay.astype(dtype),
        ),
        state0=impl.StateM(
            next_amp=amp.astype(dtype),
            next_phi=phi.astype(dtype),
        ),
        decay_per_block=decay_per_block.astype(dtype),
        phase_incr=phase_incr.astype(dtype),
    )


def recast_voice(v: VoiceSpec, dtype) -> VoiceSpec:
    """Same physical voice, arrays cast to `dtype`. Lets every dtype variant and
    the f64 reference share identical params (same phase draws)."""
    return VoiceSpec(
        midi=v.midi,
        M=v.M,
        partials=impl.PartialM(
            omega=v.partials.omega.astype(dtype),
            log_decay=v.partials.log_decay.astype(dtype),
        ),
        state0=impl.StateM(
            next_amp=v.state0.next_amp.astype(dtype),
            next_phi=v.state0.next_phi.astype(dtype),
        ),
        decay_per_block=v.decay_per_block.astype(dtype),
        phase_incr=v.phase_incr.astype(dtype),
    )


def pitch_plan(M: int, V: int) -> list[int]:
    """Spread V voices across the Nyquist-safe MIDI window for M partials so the
    per-note tables are genuinely distinct (not V identical copies)."""
    lo, hi = ap.PITCH_WINDOW[M]
    span = hi - lo
    if V == 1:
        return [(lo + hi) // 2]
    # Evenly spaced, distinct where possible; wraps if V exceeds the window.
    return [lo + round(i * span / (V - 1)) % (span + 1) for i in range(V)]


def realistic_keyboard() -> list[tuple[int, int]]:
    """The honest 'all 88 keys pressed' profile: (midi, partial_count) for every
    key A0..C8, partial count tapering from ~40 (bass) to a few (treble)."""
    return [(m, realistic_partial_count(m)) for m in range(ap.MIDI_LO, ap.MIDI_HI + 1)]
