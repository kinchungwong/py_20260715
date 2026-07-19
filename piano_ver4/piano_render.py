"""Per-note synthesis renderer for piano_ver4.

Each active note is rendered by one small BLAS GEMV over its own partials:

    note_buf = ab @ M_note

where M_note holds the static per-partial (cosexp, sinexp) rows (p_amp baked in),
and ab holds the per-block coefficients A_p = a.sin(phi), B_p = a.cos(phi) built from
each partial's live decay-amplitude `a` (PartialState.next_amp) and phase `phi`
(PartialState.next_phi). This is the identity

    p_amp . a . exp(log_decay.n) . sin(phi + omega.n)
        = A_p . cosexp_p[n] + B_p . sinexp_p[n]

Per-note separation makes the Linear Ramp (attack/release) a trivial per-note multiply.

Run pinned to one BLAS thread for deterministic timing:
    OPENBLAS_NUM_THREADS=1 .venv/bin/python piano_ver4/piano_render.py
(Per-note GEMVs are far below OpenBLAS's threading threshold, so threads are a
non-issue here regardless; the env var only removes pool-spin jitter.)
"""

import math
from time import perf_counter_ns

import numpy as np

from piano_data_model import PianoCfg, PianoNote, LinearRamp, NoteState, Trend
from piano_model import PianoModel, NotePartials


def build_cache(model: PianoModel, nsamps: int, dtype=np.float32):
    """Build the static per-partial (cosexp, sinexp) cache for every key.

    Returns (M_all, row_map, note_partials_map):
      - M_all: (total_rows, nsamps) C-contiguous float array. Rows are interleaved
        per partial as [cosexp_0, sinexp_0, cosexp_1, sinexp_1, ...], p_amp baked in.
      - row_map: note_id -> (row_start, n_partials).
      - note_partials_map: note_id -> NotePartials (reused to build NoteStates).
    """
    sr = model.cfg.sample_rate
    iota = np.arange(nsamps, dtype=np.float64)
    rows: list[np.ndarray] = []
    row_map: dict[int, tuple[int, int]] = {}
    note_partials_map: dict[int, NotePartials] = {}
    cursor = 0
    for note_id, key in model.keys.items():
        npart = NotePartials(model, key)
        note_partials_map[note_id] = npart
        partials = npart.valid_partials
        row_map[note_id] = (cursor, len(partials))
        for partial in partials:
            omega = 2.0 * math.pi * partial.p_freq / sr
            log_decay = -1.0 / (partial.p_tau * sr)
            env = partial.p_amp * np.exp(log_decay * iota)
            rows.append(env * np.cos(omega * iota))  # cosexp -> pairs with A
            rows.append(env * np.sin(omega * iota))  # sinexp -> pairs with B
        cursor += 2 * len(partials)
    M_all = np.asarray(rows, dtype=dtype)  # (total_rows, nsamps), C-contiguous
    return M_all, row_map, note_partials_map


class PianoSynth:
    """Owns the cache and one NoteState per key; renders one block at a time."""

    def __init__(self, cfg: PianoCfg, nsamps: int, dtype=np.float32) -> None:
        self.cfg = cfg
        self.nsamps = nsamps
        self.dtype = dtype
        self.model = PianoModel(cfg)
        self.cfg_lr = LinearRamp(cfg.sample_rate, cfg.attack_msecs, cfg.release_msecs)
        self.M_all, self.row_map, npmap = build_cache(self.model, nsamps, dtype)
        self.iota = np.arange(nsamps, dtype=dtype)
        self.note_states: dict[int, NoteState] = {
            note_id: NoteState(cfg, self.cfg_lr, npmap[note_id])
            for note_id in self.model.keys
        }
        self.mix = np.zeros(nsamps, dtype=dtype)
        self._ab = np.empty(2 * cfg.max_partials, dtype=dtype)  # scratch coeff buffer

    def note_on(self, note: PianoNote) -> None:
        self.note_states[note.note_id].attack(note)

    def note_off(self, note_id: int) -> None:
        self.note_states[note_id].release()

    def render_block(self) -> np.ndarray:
        mix = self.mix
        mix[:] = 0.0
        iota = self.iota
        M_all = self.M_all
        lr = self.cfg_lr
        nsamps = self.nsamps
        for note_id, ns in self.note_states.items():
            if ns.trend is Trend.SILENT:
                continue  # active-only: idle keys cost one branch, no math
            pstates = ns.partial_states
            n_part = len(pstates)
            ab = self._ab[:2 * n_part]
            # Per-block coefficients from live phase/decay-amplitude.
            amps = np.fromiter((st.next_amp for _, st in pstates), dtype=np.float64, count=n_part)
            phis = np.fromiter((st.next_phi for _, st in pstates), dtype=np.float64, count=n_part)
            ab[0::2] = amps * np.sin(phis)  # A -> cosexp rows
            ab[1::2] = amps * np.cos(phis)  # B -> sinexp rows
            start, _ = self.row_map[note_id]
            note_buf = ab @ M_all[start:start + 2 * n_part]  # (nsamps,) GEMV
            # Linear ramp envelope (per note).
            if ns.trend is Trend.LEVEL:
                note_buf *= ns.level
            else:
                ramp = lr.ramp_rise if ns.trend is Trend.RISE else lr.ramp_fall
                env = np.clip(ns.level + ramp * iota, ns.level_min, ns.level_max)
                note_buf *= env
            mix += note_buf
            ns.advance(nsamps)  # advance AFTER reading block-start state
        return mix


# ----------------------------------------------------------------------------
# Correctness + timing harness
# ----------------------------------------------------------------------------

def _reference_mix(partials, phis, amps, nsamps: int, sr: float) -> np.ndarray:
    """Independent time-domain ground truth: direct damped-sine sum (no cache)."""
    n = np.arange(nsamps, dtype=np.float64)
    acc = np.zeros(nsamps, dtype=np.float64)
    for partial, phi, a in zip(partials, phis, amps):
        omega = 2.0 * math.pi * partial.p_freq / sr
        log_decay = -1.0 / (partial.p_tau * sr)
        acc += partial.p_amp * a * np.exp(log_decay * n) * np.sin(phi + omega * n)
    return acc


def _test_waveform_identity(cfg: PianoCfg, nsamps: int, n_blocks: int = 6) -> None:
    """Render with env==1 and compare the GEMV path to the analytic reference
    across consecutive blocks (this also validates cross-block continuity)."""
    synth = PianoSynth(cfg, nsamps)
    note_id = 60  # middle C
    synth.note_on(PianoNote(note_id, 100))
    ns = synth.note_states[note_id]
    # Force constant unit envelope so we isolate the waveform/GEMV identity.
    ns.trend = Trend.LEVEL
    ns.level = 1.0
    ns.level_min = 1.0
    ns.level_max = 1.0

    partials = [p for p, _ in ns.partial_states]
    # Mirror the live state independently for the reference.
    sr = cfg.sample_rate
    phis = [st.next_phi for _, st in ns.partial_states]
    amps = [st.next_amp for _, st in ns.partial_states]  # all 1.0 at attack
    omegas = [2.0 * math.pi * p.p_freq / sr for p in partials]
    decays = [math.exp(-nsamps / (p.p_tau * sr)) for p in partials]

    worst = 0.0
    for b in range(n_blocks):
        out = synth.render_block().astype(np.float64).copy()
        ref = _reference_mix(partials, phis, amps, nsamps, sr)
        err = float(np.max(np.abs(out - ref)))
        worst = max(worst, err)
        print(f"  block {b}: max|err| = {err:.3e}   peak |ref| = {np.max(np.abs(ref)):.4f}")
        # Advance the reference in lockstep with NoteState.advance().
        phis = [(phi + w * nsamps) % (2.0 * math.pi) for phi, w in zip(phis, omegas)]
        amps = [a * d for a, d in zip(amps, decays)]
    assert worst < 1e-5, f"waveform identity failed: worst max|err| = {worst:.3e}"
    print(f"  PASS waveform identity across {n_blocks} blocks (worst = {worst:.3e})\n")


def _test_envelope(cfg: PianoCfg, nsamps: int) -> None:
    """Sanity: a fresh attack ramps the level up from 0; release ramps it down."""
    synth = PianoSynth(cfg, nsamps)
    nid = 69
    synth.note_on(PianoNote(nid, 127))
    ns = synth.note_states[nid]
    assert ns.trend is Trend.RISE and ns.level == 0.0
    synth.render_block()
    lvl_after_attack = ns.level
    assert lvl_after_attack > 0.0, "level did not rise after attack"
    # Reach the ceiling, then release.
    for _ in range(20):
        synth.render_block()
    top = ns.level
    synth.note_off(nid)
    synth.render_block()
    assert ns.level < top, "level did not fall after release"
    print(f"  PASS envelope: rise->{lvl_after_attack:.4f} ceiling->{top:.4f} "
          f"fall->{ns.level:.4f}\n")


def _bench(cfg: PianoCfg, nsamps: int, n_active: int, trials: int = 100) -> None:
    synth = PianoSynth(cfg, nsamps)
    ids = list(synth.model.keys)[:n_active]
    for nid in ids:
        synth.note_on(PianoNote(nid, 100))
        ns = synth.note_states[nid]  # hold at constant level so none decay to SILENT
        ns.trend = Trend.LEVEL
        ns.level = 0.5
        ns.level_min = 0.5
        ns.level_max = 0.5
    synth.render_block()  # warm up
    t0 = perf_counter_ns()
    for _ in range(trials):
        synth.render_block()
    t1 = perf_counter_ns()
    per_block_us = (t1 - t0) * 1e-3 / trials
    per_note_us = per_block_us / max(1, n_active)
    deadline_us = nsamps / cfg.sample_rate * 1e6
    pct = per_block_us / deadline_us * 100.0
    print(f"  {n_active:3d} notes: {per_block_us:8.2f} us/block  "
          f"({per_note_us:6.3f} us/note)  = {pct:5.2f}% of {deadline_us:.0f} us deadline")


if __name__ == "__main__":
    cfg = PianoCfg()
    nsamps = 383  # matches the piano project's callback block

    print(f"nsamps = {nsamps}  (deadline = {nsamps / cfg.sample_rate * 1e6:.0f} us)\n")

    print("Correctness: waveform identity vs analytic reference")
    _test_waveform_identity(cfg, nsamps)

    print("Correctness: linear-ramp envelope")
    _test_envelope(cfg, nsamps)

    print("Timing: render_block per active-note count")
    for n_active in (1, 10, 20, 88):
        _bench(cfg, nsamps, n_active)
