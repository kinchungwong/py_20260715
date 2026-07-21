"""Canonical audio parameters, the benchmark grid, and physical-model constants.

All numbers here are sourced from the repo and the two landmark deep-research
reports (see claude/deep_research/). Nothing in this module imports the render
kernel, so it is safe to import before numpy thread-pinning is settled.
"""
from __future__ import annotations

import math

# ---------------------------------------------------------------------------
# Audio timing
# ---------------------------------------------------------------------------
FS: int = 44100  # sample rate, Hz -- canonical repo-wide

# Audio-callback block sizes (frames). Deadline for a block of N frames is the
# time the callback has to produce it: N / FS seconds.
N_PRIMARY: int = 383   # canonical piano callback   -> 8685.0 us deadline
N_LOWLAT: int = 95     # low-latency callback        -> 2154.2 us deadline
N_TEST: int = 1024     # test/convenience block      -> 23219.9 us deadline
N_SET: tuple[int, ...] = (N_PRIMARY, N_LOWLAT, N_TEST)


def deadline_us(nsamps: int) -> float:
    """Real-time budget for one block of `nsamps` frames, in microseconds."""
    return nsamps / FS * 1e6


# ---------------------------------------------------------------------------
# The benchmark grid (partials-per-note x simultaneous voices)
# ---------------------------------------------------------------------------
GRID_M: tuple[int, ...] = (5, 16, 40)      # partials per note
GRID_V: tuple[int, ...] = (1, 5, 20, 88)   # keys pressed at once

# ---------------------------------------------------------------------------
# Physical model (ver3 / older-project roll-off recipe -- keeps a strong
# fundamental, unlike ver4's sin^16 hammer weighting which suppresses k=1).
# ---------------------------------------------------------------------------
INHARM_B: float = 4.0e-4    # stiff-string inharmonicity coefficient
TAU_FUND: float = 1.6       # fundamental decay time constant, seconds
TAU_PCOEF: float = 0.25     # per-partial decay speed-up
AMP_PCOEF: float = 1.2      # amp_k = 1 / k**AMP_PCOEF (before normalization)
AUDIBLE_HZ: float = 18000.0     # partials above this are inaudible -> dropped
AUDIBLE_AMP: float = 1.0 / 32768.0  # 16-bit LSB; a production Trend.SILENT cull
                                    # would demote a voice near here, long before
                                    # any partial amplitude reaches denormal range.
NYQUIST: float = math.pi    # omega (rad/sample) must stay strictly below this

MIDI_LO: int = 21   # A0
MIDI_HI: int = 108  # C8

# Nyquist-safe MIDI window per partial count. The highest voice's top partial
# must satisfy omega_M < pi:  f1 * M*sqrt(1+B*M^2) < FS/2.
#   M=5  -> f1 < ~4388 Hz (midi < ~108)
#   M=16 -> f1 < ~1312 Hz (midi < ~88)
#   M=40 -> f1 < ~430  Hz (midi < ~69)
# Windows are chosen comfortably inside those limits and wide enough that V
# voices spread across them get genuinely distinct per-note tables.
PITCH_WINDOW: dict[int, tuple[int, int]] = {
    5: (48, 96),
    16: (36, 84),
    40: (21, 60),
}

# ---------------------------------------------------------------------------
# Cross-validation / context anchors (do NOT recompute -- compare against).
# ---------------------------------------------------------------------------
# ver4's validated per-note result (claude/deep_research/2026-07-18-...:199).
VER4_ANCHOR: dict[str, float] = {
    "N": 383, "M": 16, "V": 88,
    "us_block": 1204.78, "us_note": 13.69, "pct": 13.87,
}
# The GEMV-batching report's full-collapse ceiling (2026-07-17-...): ~1.3% of
# the N=383 deadline for the whole 88-key / 16-partial keyboard.
BATCHED_CEILING_PCT: float = 1.3

# 16-bit PCM tolerance used by the existing ver5 tests.
CORRECTNESS_TOL: float = 2.0 / 65536.0

# Reproducible phase draws.
RNG_SEED: int = 20260720
