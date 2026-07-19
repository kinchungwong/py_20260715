# Per-Note Synthesis — the piano_ver4 renderer

**Date:** 2026-07-18
**Scope:** `piano_ver4/` — porting waveform generation onto the clean data model, using
one small BLAS GEMV per active note.
**Companion report:** `claude/deep_research/2026-07-17-additive-synthesis-gemv-batching.md`
(the batched-GEMV analysis this builds on). Read that first for the dispatch-overhead and
BLAS-threading background; this report assumes it.

---

## 1. Purpose

The prior report established that a damped-sine mixdown is a matrix product and that NumPy
per-call **dispatch** (~0.75 µs/call), not the sin/exp arithmetic, is the real cost. It
measured the *full-collapse* extreme — every partial of every key summed to one mono output
in a single GEMV — at ~1.3 % of the audio deadline.

This session takes the opposite, deliberately-chosen granularity: **one note at a time, all
its ~16 partials in a single GEMV.** The motivation is twofold:

1. **Per-note envelopes.** A per-note attack/release (Linear Ramp) requires keeping notes
   separate through the mixdown. Full-collapse forfeits that; per-note synthesis makes the
   envelope a trivial per-note multiply.
2. **No wasted work on idle keys.** Full-collapse reads the whole cache every block even for
   silent keys. Per-note skips silent keys entirely, and amortizes dispatch across a note's
   partials — the sweet spot between the two overheads.

The headline outcome: **full 88-key polyphony renders in ~14 % of the audio deadline**, with
per-note phase/decay continuity and per-note attack/release, correct to float32 precision.
For comparison, the earlier `py_20260707` design hit ~26 % of deadline at only **16** voices.

---

## 2. Environment

- CPU: AMD Ryzen 9 7945HX (32 logical CPUs)
- Python 3.13.14, NumPy 2.5.1 (bundled `libscipy_openblas64_`)
- venv: `py_20260715/.venv`
- Audio block: `nsamps = 383` samples @ 44100 Hz → **8685 µs deadline** per block

---

## 3. The render identity

For a partial `p` with baked-in relative amplitude `p_amp`, per-sample angular frequency
`omega_p`, per-sample log-decay `log_decay_p`, and *live* state — accumulated phase `phi`
(`PartialState.next_phi`) and accumulated cross-block decay amplitude `a`
(`PartialState.next_amp`) — the block signal is:

```
signal_p[n] = p_amp · a · exp(log_decay_p·n) · sin(phi + omega_p·n)
            = A_p · cosexp_p[n]  +  B_p · sinexp_p[n]

  cosexp_p[n] = p_amp · cos(omega_p·n) · exp(log_decay_p·n)     # STATIC, cached
  sinexp_p[n] = p_amp · sin(omega_p·n) · exp(log_decay_p·n)     # STATIC, cached
  A_p = a·sin(phi)     (per-block scalar, coefficient on the cosexp row)
  B_p = a·cos(phi)     (per-block scalar, coefficient on the sinexp row)
```

The two `*exp` rows are **static** (they depend only on `omega_p`, `log_decay_p`, and the
baked `p_amp` — none of which change after construction), so they are computed once at
startup. Everything that changes per block lives in the two scalars `A_p`, `B_p`, derived
from the live `(a, phi)`. A note's whole block is therefore one GEMV:

```
note_buf(nsamps,) = ab(2P,) @ M_note(2P, nsamps)
```

with `M_note`'s rows interleaved `[cosexp_0, sinexp_0, cosexp_1, sinexp_1, …]` and
`ab = [A_0, B_0, A_1, B_1, …]`. State is advanced (phase mod 2π, amp ×= block-decay) *after*
the render reads the block-start state, giving click-free continuity across blocks.

---

## 4. Implementation

Three files in `piano_ver4/`:

- **`piano_data_model.py`** — data model + live state (`PianoCfg`, `Partial`, `DampedSine`,
  `LinearRamp`, `Trend`, `PartialState`, `NoteState`). Written prior to this session.
- **`piano_model.py`** — concrete equations (`PianoModel`, `NotePartials`). Prior + this
  session's fixes.
- **`piano_render.py`** — **new this session**: the cache builder, the `PianoSynth` engine,
  and a correctness + timing harness.

### 4.1 Cache

`build_cache(model, nsamps, dtype=np.float32)` builds, for every key, that key's valid
partials' `(cosexp, sinexp)` rows and packs them into **one contiguous** `M_all` of shape
`(total_rows, nsamps)`, plus `row_map: note_id → (row_start, n_partials)`. A note's block is
the contiguous row-slice `M_all[start : start + 2P]`.

Crucially, the cache is built at **exactly `nsamps`** (known at init), not at some
`max_nsamps`. This sidesteps the column-slice trap from the prior report — slicing a
max-width cache along the sample axis (`M[:, :nsamps]`) is non-contiguous and forces BLAS to
copy (3–4× penalty). By sizing to `nsamps`, every per-note GEMV reads a contiguous block.

### 4.2 The render loop (`PianoSynth.render_block`)

```python
mix[:] = 0.0
for note_id, ns in self.note_states.items():
    if ns.trend is Trend.SILENT:
        continue                                        # active-only: idle keys cost one branch
    pstates = ns.partial_states
    n_part = len(pstates)
    ab = self._ab[:2 * n_part]
    amps = np.fromiter((st.next_amp for _, st in pstates), dtype=np.float64, count=n_part)
    phis = np.fromiter((st.next_phi for _, st in pstates), dtype=np.float64, count=n_part)
    ab[0::2] = amps * np.sin(phis)                      # A → cosexp rows
    ab[1::2] = amps * np.cos(phis)                      # B → sinexp rows
    start, _ = self.row_map[note_id]
    note_buf = ab @ M_all[start:start + 2 * n_part]     # (nsamps,) GEMV
    if ns.trend is Trend.LEVEL:
        note_buf *= ns.level                            # constant sustain gain
    else:
        ramp = lr.ramp_rise if ns.trend is Trend.RISE else lr.ramp_fall
        env = np.clip(ns.level + ramp * iota, ns.level_min, ns.level_max)
        note_buf *= env
    mix += note_buf
    ns.advance(nsamps)                                  # advance AFTER reading block-start state
```

`note_on` → `NoteState.attack`, `note_off` → `NoteState.release`. One `NoteState` per key is
pre-built at init; all start `SILENT`. "Active" simply means `trend is not SILENT`, so the
loop iterates all 88 keys and skips silent ones with a single branch.

### 4.3 Corrections applied to the ver4 data model

Three defects were found and fixed while wiring the renderer:

1. **`piano_model.py` — L2 normalization used the wrong tuple field.** The per-key partial
   tuples are `(k, p_freq, p_amp, p_tau)`, but the sum-of-squares normalization read
   `tup[3]` (`p_tau`) instead of `tup[2]` (`p_amp`), so relative partial amplitudes were
   scaled by a factor derived from decay constants. Fixed `tup[3] → tup[2]`.

2. **`piano_data_model.py` `NoteState` — every partial shared one `DampedSine`.**
   `PartialState.advance()` reads `omega`/`log_decay` from its `DampedSine`; sharing a single
   note-level instance meant every partial advanced with the *fundamental's* frequency and
   decay. Fix: removed `cfg_ds` from `NoteState` entirely (field + `__init__` param) and had
   `NoteState` construct a per-partial `DampedSine(sample_rate, partial.p_freq, partial.p_tau)`.

3. **`piano_data_model.py` — an import-blocking annotation bug.** `Partial.__eq__` was typed
   `other: "Partial" | Any`, which evaluates `str | Any` at class-definition time and raises
   `TypeError`. It had never fired because nothing imported the module at runtime before. Fix:
   move the whole union inside the forward-ref string, `other: "Partial | Any"`.

After these, and the author's addition of `@override` decorators and a `del k` for the unused
parameter in `_valid_partial`, **Pyright reports 0 errors / 0 warnings / 0 informations** over
the directory.

---

## 5. Reproduction

Run pinned to one BLAS thread (see §7 for why):

```bash
cd py_20260715
OPENBLAS_NUM_THREADS=1 .venv/bin/python piano_ver4/piano_render.py
```

The harness runs two correctness checks and a timing sweep.

---

## 6. Results

### 6.1 Correctness

**Waveform identity.** Middle C is attacked, its envelope forced to a constant 1.0 (to isolate
the waveform path), and six consecutive blocks are rendered and compared to an *independent*
time-domain reference — a direct `Σ_p p_amp·a·exp(log_decay·n)·sin(phi + omega·n)` computed
without touching the cache, with `(phi, a)` advanced by hand in lockstep. Because the
reference continues the phase/decay across blocks, agreement on blocks 1–5 is also the
cross-block continuity proof (no seam click).

```
block 0: max|err| = 2.539e-07   peak |ref| = 1.5769
block 1: max|err| = 3.612e-07   peak |ref| = 1.6931
block 2: max|err| = 2.847e-07   peak |ref| = 1.4921
block 3: max|err| = 3.389e-07   peak |ref| = 1.5354
block 4: max|err| = 4.563e-07   peak |ref| = 1.3013
block 5: max|err| = 3.447e-07   peak |ref| = 1.3525
→ worst max|err| = 4.6e-7  (float32-level; threshold 1e-5)
```

**Linear-ramp envelope.** A velocity-127 attack rises 0 → 1.0, holds at the ceiling, then on
`note_off` the release falls (0.5658 one block later, matching `1.0 + ramp_fall·383`).

### 6.2 Timing (`nsamps = 383`, one BLAS thread)

| active notes | µs / block | µs / note | % of 8685 µs deadline |
|---:|---:|---:|---:|
| 1  |   20.08 | 20.08 |  0.23 % |
| 10 |  144.09 | 14.41 |  1.66 % |
| 20 |  286.30 | 14.32 |  3.30 % |
| **88** | **1204.78** | **13.69** | **13.87 %** |

**Full 88-key polyphony fits in ~14 % of the deadline.** Per-note cost asymptotes to
**~13.7 µs/note**, which is almost entirely Python/NumPy dispatch on the coefficient build
(`np.fromiter` gather + sin/cos) and the `advance()` loop — the GEMV over a ~28×383 block is
negligible. The 1-note figure is higher only because the fixed 88-key dict-scan and `mix[:]=0`
are amortized across more notes as polyphony grows. This is the "one note at a time" tax,
accepted deliberately.

The lever to go below ~14 µs/note, if ever needed, is to store each note's partial
`next_amp`/`next_phi` as small NumPy arrays instead of per-`PartialState` Python floats,
removing the `np.fromiter` gather. It is a data-model change and was left out of scope.

---

## 7. BLAS threading — a sharper restatement

The prior report found BLAS threading a **no-op** at the audio block size (N=383). A
measurement this session sharpens that: on the `piano_ver3` full-collapse GEMV
(`(1, 2160) @ (2160, 1024)`, an 8.85 MB cache), the default 32-thread OpenBLAS took **~3.0 ms**;
`OPENBLAS_NUM_THREADS=1` dropped it to **~0.16 ms — an ~18× speedup.** At that size the matrix
crosses OpenBLAS's parallelization threshold and the 32-thread pool spins up and synchronizes
for what is a memory-bound GEMV; the launch/sync overhead *is* the cost. So the rule is:

> Threading is a no-op at N≈383 and **actively harmful** by N≈1024. Pin to one thread.

For the **per-note** kernel this is moot — a `(2P, 383)` block (~28×383) is far below any
threading threshold, so per-note GEMVs never thread regardless. Pinning is retained only to
remove pool-spin jitter and keep timing deterministic. The 0.16 ms full-collapse figure is
now memory-bandwidth-bound (8.85 MB / 0.16 ms ≈ 55 GB/s, realistic single-core bandwidth),
which is also why float32 (half the bytes of float64) is ~2× faster there and is kept.

---

## 8. Known gap: re-trigger, and why realism needs a crossfade

Re-trigger (striking a key already sounding) is **out of the current scope** but was analyzed,
because the interaction with the Linear Ramp is subtle and shapes the next data-model step.

`attack()` already resets each partial to `next_amp = 1.0` and a fresh random phase. That
reset is **click-free only when the envelope `level` is 0 at that instant** — true for a fresh
attack from `SILENT` (the reset is multiplied by zero), but false on a re-trigger from an
audible state, where `level > 0` and the note steps discontinuously in both amplitude
(`a_old → 1.0`) and phase. The Linear Ramp does not mask it, because it ramps `level`, and
`level` does not restart from zero on re-trigger.

Two ways to make it click-free:

- **Deferred reset at the envelope zero** — ramp `level` down to 0, re-init phase/amp at that
  zero point, ramp back up. Click-free, but the note momentarily goes **silent** (a gap).
  Acceptable for slow re-articulation, wrong for fast repeated notes.
- **Crossfade** — keep the old voice (old decayed amp, old phases) fading out while a **new**
  voice (amp 1.0, fresh random phases) fades in. The sum stays continuous *and* there is no
  gap; the new strike overlaps the still-ringing old one. This is the realistic behavior, and
  the only one that re-initializes both amplitude *and* phase while sound is continuously
  present. **This confirms the intuition that realism requires a crossfade, i.e. two
  concurrent voices per key.**

The important architectural note: a crossfade is cheap in this scheme, because **both voices
of a re-triggered key share the identical cache rows** (same `note_id` → same partials → same
`M_note`). The second voice needs no extra cache — only a second coefficient vector and a
second envelope, i.e. one more GEMV against the same block, and only during the short overlap
window. With 88 keys already inside ~14 % of the deadline, the occasional doubled voice is
effectively free. The real cost is **structural**: `NoteState` must become a small *voice pool*
per key (2 covers a single crossfade; a slightly larger pool covers a third strike arriving
mid-crossfade) — a gentle, much cheaper reprise of the voice-allocation problem the batched
approach had otherwise retired.

Two details for that future work, both "interactions with the Linear Ramp":

1. **Use an equal-*power* crossfade, not the linear (equal-gain) ramp.** The two voices have
   independent random phases → they are decorrelated → they sum in power, not amplitude. A
   linear crossfade of decorrelated signals dips ~3 dB mid-fade; power-complementary curves
   (`old_env² + new_env² ≈ const`) keep loudness flat.
2. **Per-voice reaping** — a voice ends when its fade reaches 0 *or* its exponential decay
   drops it below `audible_amp` (the same `silent()` test, applied per voice).

---

## 9. Conclusions

1. **Per-note synthesis is the right granularity.** One GEMV per active note amortizes
   dispatch across the partials, does zero work for idle keys, and makes per-note envelopes
   trivial. Measured **~13.7 µs/note**, **88 keys in ~14 % of the deadline** — correct to
   float32 (worst `max|err| = 4.6e-7`).
2. **The 16-voice ceiling is gone.** Every key can sound, unconditionally, with headroom to
   spare. Allocation/stealing/`max_voices` are unnecessary for the base polyphony.
3. **The static cache holds.** Built once at exactly `nsamps`; no LRU, no per-note-on table
   work, no column slicing.
4. **Pin BLAS to one thread.** No-op at per-note sizes, but an ~18× hazard at the larger
   full-collapse sizes; pinning is free insurance and keeps timing deterministic.
5. **Re-trigger realism will require a per-key voice pool (crossfade).** It is architecturally
   modest here because voices of a key share the same cache, but it is a real data-model step,
   deferred by choice.

---

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
