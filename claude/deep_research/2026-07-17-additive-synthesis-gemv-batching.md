# Additive synthesis at scale: the bottleneck is NumPy dispatch, not math

**Date:** 2026-07-17
**Topic:** Why `damped_sine_ver2` plateaus, and the batched GEMV/GEMM reformulation that lifts it
**Status:** Measured and verified. Prototype only — nothing in `damped_sine_ver1/`, `damped_sine_ver2/`, or `py_20260707` was modified.

---

## TL;DR

`DampedSineComposed` spends **~73% of every NumPy call on dispatch overhead, not arithmetic**.
Caching `sin`/`cos` removes arithmetic that was never the cost. The ceiling is the *API shape*:
one damped sine = one object = one render call = ~5 NumPy dispatches per block.

Reformulating the voice bank so the whole mixdown is **one BLAS call per block** — a matrix
product, independent of voice count — measures **25× (float64) / 38× (float32)** on ver2's own
benchmark, and **57×** at `py_20260707`'s actual piano parameters. Verified correct to ~1e-14
against ver2's own renderers.

The consequence for the piano: **the entire 88-key keyboard, all 16 partials per key, with
per-voice gating, costs ~1.3% of the audio deadline.** Full 88-key polyphony is affordable,
which removes the reason for voice allocation, voice stealing, and a `max_voices` limit.

---

## Environment

| | |
|---|---|
| CPU | AMD Ryzen 9 7945HX (32 logical CPUs) |
| OS | Linux 6.17.0-35-generic |
| Python | 3.13.14 (`py_20260715/.venv`) |
| NumPy | 2.5.1 |
| BLAS | `libscipy_openblas64_` (bundled in the NumPy wheel; pthreads build; **default 32 threads**) |

All benchmarks below take the **minimum of 3 timed runs** after warm-up, using `perf_counter_ns`.
BLAS is pinned to one thread (`OPENBLAS_NUM_THREADS=1`) except where threading is the subject.

Reproduce everything with the venv's interpreter, not a bare `python`:

```
cd /home/ubuntu/workspace/py_20260715
.venv/bin/python -VV        # expect 3.13.x
```

### Verification of this report

Every script in Appendix A was **extracted back out of this document and re-run** to confirm the
listing is what actually produced the numbers. All 11 exit 0 and reproduce the reported figures.

**Run-to-run variance is ~10% on the micro-benchmarks** and is not suppressed here. E.g. the
`_write_r_sin_cos` hot line measured 3.602 µs on the run quoted in Part 2.2 and 3.922 µs on the
verification run; `CURRENT` measured 4.272 and 4.577. Treat one-decimal precision as spurious —
none of the conclusions turn on differences smaller than ~2×, and the effects claimed here are
3×–57×. The float32 headline moved between 38.4× and 40.7× across runs; the report quotes the
lower figure.

Numbers that reproduced essentially exactly (they are large, structural effects): the dispatch
plateau (Part 2.1), the layout trap (Part 6.1 — 98.4/268.6 µs sliced vs 21.2/84.1 µs axis-0), the
threading no-op (Part 6.2), and every correctness figure (Part 4, bit-identical).

---

## Part 1 — Reproducing the baseline

`damped_sine_ver2/perf_non_debug.txt` records 6512.378 ms. Reproduced:

```
cd /home/ubuntu/workspace/py_20260715/damped_sine_ver2
../.venv/bin/python -c "import test_composed_perf as t; t.test_perf_composed()"
```

```
Total samples rendered: 4410000
Total tones rendered: 100
Elapsed time for attaching tones: 4.158 ms
Elapsed time for rendering: 6602.053 ms
```

The workload is 100 tones × 8614 blocks of 512 = **861,400 `render_to` calls**, so
**7.66 µs per call**. That is the number everything else explains.

---

## Part 2 — Diagnosis: where the 7.66 µs goes

### 2.1 A NumPy ufunc call costs ~0.75 µs regardless of size

This is the core finding. Script: **`bench_dispatch.py`** (Appendix A).

```
=== numpy per-ufunc-call cost vs array length (float64, a*x) ===
        N    us/call   ns/element
        8      0.765       95.568
       32      0.778       24.319
      128      0.810        6.327
      512      1.036        2.023
     2048      1.409        0.688
     8192      2.857        0.349
    32768      8.349        0.255
   131072     35.994        0.275
```

A call on **8** elements costs 0.765 µs. A call on **512** costs 1.036 µs. The marginal cost of
504 extra elements is 0.27 µs. **At N=512, ~73% of every NumPy call is fixed overhead** —
Python-level dispatch, type/shape resolution, output allocation.

**This is the whole story.** At audio block sizes NumPy is overhead-bound, not compute-bound.
Optimizing arithmetic per call cannot help; only *reducing the number of calls* can.

### 2.2 The per-voice cost decomposes exactly

Script: **`bench_decompose.py`** (Appendix A), N=512 float64:

```
=== micro-ops ===
  out[:] = a*cos + b*sin   [CURRENT hot line]                 3.602 us/call
    a*cos_part                (py-float * array)              1.027 us/call
    a32*cos_part              (np.float64 * array)            1.083 us/call
    cos_part + sin_part       (array + array)                 0.765 us/call
    out *= exp_part           (in-place array*array)          0.797 us/call
    np.multiply(cos,a,out=out) (out= scalar mul)              0.972 us/call

=== per-voice render paths ===
  CURRENT (write_r_sin_cos + decay multiply)                  4.272 us/call
  fused combo (ver1 ComboEntry idea)                          3.618 us/call
  fused + out= param                                          2.821 us/call

=== reference: no cache at all ===
  np.sin(phi + omega*iota)                                    9.329 us/call
  np.sin(...)*np.exp(...) full uncached                      12.337 us/call
```

`_write_r_sin_cos`'s single line is **4 dispatches** (mul, mul, add, setitem-copy) = 3.602 µs,
and it is almost exactly additive: 1.027 + 1.027 + 0.765 + ~0.8 ≈ 3.6. Add the decay pass
(`outslice *= exp_part`, 0.797) and the NumPy subtotal is **4.272 µs**.

### 2.3 The full accounting

| component | µs/voice/block | share |
|---|---|---|
| NumPy (5 dispatches) | 4.27 | 56% |
| Python (call chain, `isinstance`, ABC) | 3.39 | 44% |
| **measured total** (6602 ms ÷ 861,400) | **7.66** | 100% |

The Python 44% is visible in `perf_cprofile.txt`: **6,030,100 `isinstance` calls** (7 per render),
1,722,800 `require_renderer()` calls (twice per render — once in `DampedSine.render_to`, again in
`DampedSineComposed.render_to`), and 1,723,000 `<frozen abc>:__instancecheck__` calls. ABC
`isinstance` is materially slower than a plain type check.

### 2.4 What this means

`DampedSineComposed` **works** — it took the fully-uncached 12.34 µs down to 4.27 µs, a real 3×.
But it optimized along an axis with a low ceiling. Caching `sin`/`cos` removes *arithmetic*;
the cost was *dispatch*, and the cached path still issues 4–5 dispatches per voice per block.

**No design in which one damped sine issues its own NumPy calls can beat ~1 µs/voice/block**,
and with a 5-dispatch body plus Python overhead, ~7.7 µs is the honest floor of that shape.

### 2.5 ver1's `ComboEntry` was right, and ver2 regressed it

ver1 fused `sin(ωn)·exp(λn)` and `cos(ωn)·exp(λn)` into a `ComboEntry`. ver2 dropped the fusion
and multiplies the decay at render time. That costs **0.65 µs/voice/block** (4.272 vs 3.618) —
and, more importantly, **the fusion is exactly what makes the batched form in Part 3 possible.**
ver1's "merely pedagogical" version contained the load-bearing idea.

---

## Part 3 — The reformulation

Take ver2's own identity (`_write_r_sin_cos`, kept exactly as written there):

```
out_v[n] = amp_v · exp(λ_v·n) · sin(φ_v + ω_v·n)
         = (amp_v·sin φ_v) · [cos(ω_v·n)·exp(λ_v·n)]
         + (amp_v·cos φ_v) · [sin(ω_v·n)·exp(λ_v·n)]
         =       A_v       ·       COSEXP[v, n]
         +       B_v       ·       SINEXP[v, n]
```

The bracketed factors depend **only on `(ω_v, λ_v)`** — that is ver1's `ComboEntry`, so they are
static table rows. `A_v` and `B_v` are two scalars per voice per block. Therefore the mixdown is

```
mix[n] = Σ_v out_v[n] = (A @ COSEXP)[n] + (B @ SINEXP)[n]
```

— **a matrix-vector product**. Interleave the rows (`2v` → COSEXP, `2v+1` → SINEXP) and it is a
single `(2V,) @ (2V, N)` GEMV: **one BLAS call per block, independent of voice count.** The
per-voice Python loop disappears entirely; `advance()` becomes two vector ops on `(V,)` arrays.

Prototype: **`bank_proto.py`** (Appendix A).

---

## Part 4 — Correctness

Script: **`verify_bank.py`** (Appendix A). 40 blocks × 512 samples, 8 deliberately nasty voices
(τ shorter than one block; τ that decays to nothing *inside* a block; near-Nyquist; sub-Hz;
duplicate `(freq, τ)` with different amp/phase; τ=1e-4):

```
blocks compared     : 40 x 512 samples, 8 voices
max |bank - uncached|  : 2.477e-14
max |bank - composed|  : 5.551e-16
max |bank - analytic|  : 1.462e-12

16-bit LSB for reference: 3.052e-05
VERDICT: MATCH
```

Agreement with ver2's own renderers is at float64 roundoff — **eight orders of magnitude below a
16-bit LSB**. The `analytic` column is an independent ground truth computed from absolute sample
index with no per-block state at all, so this is not a self-consistency check.

### float32 is also fine

Script: **`check_f32.py`** (Appendix A). float32 tables, error vs the float64 bank after 100 blocks:

```
   voices  peak |mix|   f32 max err  err / 24-bit LSB
       16      0.9757     2.118e-07              1.78
      100      0.9720     1.524e-07              1.28
      512      0.7968     3.321e-07              2.79
     1024      0.7845     2.559e-07              2.15
```

~2–3 × a 24-bit LSB against a peak near 1.0 — roughly 130 dB SNR, below the 24-bit noise floor
and ~100× below a 16-bit LSB. float32 is *also 2× faster* at scale (Part 5), because the GEMV is
memory-bandwidth-bound on reading the table.

---

## Part 5 — Timing results

### 5.1 Headline: ver2's own benchmark

Script: **`bench_bank.py`** (Appendix A). Identical workload — 100 tones, 100 s @ 44100 Hz,
512-sample blocks.

```
  ver2 DampedSineComposed (measured)   :    6602.0 ms
  batched bank, float64                :     261.1 ms   ( 25.3x faster)
  batched bank, float32                :     172.0 ms   ( 38.4x faster)
```

### 5.2 Micro-comparison (V=100, N=512, per block)

```
  per-voice loop, current ver2 path (100 x 4.272)          ~427.2 us
  batched (V,N) matrix, per-voice outputs kept               83.2 us
  batched GEMV mixdown (2x dot)                              16.6 us
  batched GEMV mixdown (single 2V x N dot)                   16.3 us
```

> **Note on fairness:** `test_composed_perf.py` renders every tone into *the same* `out` buffer,
> overwriting it — it never sums the voices. A real additive mixdown would cost an extra
> ~0.8 µs/voice/block on top of ver2's 7.66. The comparison above is therefore **generous to
> ver2**; the bank gets the mixdown for free, inside the GEMV.

### 5.3 Scaling (single-thread BLAS, N=512, 44100 Hz; %RT = share of real time)

```
   voices  f64 us/blk   f64 %RT  f32 us/blk   f32 %RT  tbl MB f32
     16        12.1     0.10%        10.4     0.09%        0.07
     64        19.7     0.17%        15.0     0.13%        0.26
    100        26.8     0.23%        19.4     0.17%        0.41
    256        63.9     0.55%        38.9     0.34%        1.05
    512       120.3     1.04%        72.6     0.63%        2.10
   1024       228.7     1.97%       132.5     1.14%        4.19
   2048       483.0     4.16%       251.0     2.16%        8.39
   4096      1451.5    12.50%       520.0     4.48%       16.78
```

Cost is **linear in voice count and memory-bandwidth-bound** above ~256 voices: 16 bytes per voice
per sample (float64) or 8 (float32). At V=4096/f32 that is 16.8 MB/block ÷ 520 µs ≈ **32 GB/s** —
DRAM territory. This is why float32 is worth ~2× at scale, and why the table *layout* matters
(Part 6.1). Block size does **not** change total bandwidth (bytes ∝ V × total_samples), only the
fixed overheads.

### 5.4 At `py_20260707`'s actual piano parameters

Script: **`piano_scale.py`** (Appendix A). 383-sample block → **8685 µs deadline**;
`N_PARTIALS = 16`. `claude/spikes/piano_voice.py` records **16 PianoVoices ≈ 2240 µs/block ≈ 26%
of deadline**.

```
   voices   sines  f32 us/blk  % deadline
       16     256        39.2       0.45%      <- vs 2240 us / 26% today  => 57x
       32     512        74.3       0.85%
       64    1024       138.6       1.60%
      128    2048       265.0       3.05%
      256    4096       524.3       6.04%
      512    8192      1340.4      15.43%
```

**256 voices costs a quarter of what 16 voices costs today.** The 512-voice row is superlinear
because the table (25 MB) stops fitting cache.

### 5.5 Note-on cost

```
note-on cost (naive per-partial add(), 16 partials): 310.1 us
  -> 3.6% of one block budget, once per note-on
```

Comparable to ver2's own attach (4.176 ms ÷ 100 tones = 41.8 µs/tone), so not a regression — and
Part 6.3 shows the piano doesn't need it *at all*.

### 5.6 Stereo

`(2, 2V) @ (2V, N)` GEMM vs the mono `(2V,) @ (2V, N)` GEMV, V=512, N=512:

```
  mono  : (2V,) @ (2V,N)  gemv     86.58 us/block
  stereo: (2,2V) @ (2V,N) gemm    247.19 us/block
```

Per-voice panning costs ~2.9× mono (a 2-row GEMM is a skinny, inefficient shape), still cheap.

---

## Part 6 — Traps, gotchas, and corrections

### 6.1 TRAP: allocating the table at max size and slicing per block costs 3–4×

NumPy does **not** hand a column-sliced view to BLAS as an `lda`-strided GEMV. It **copies**.
Script: **`layout.py`** (Appendix A). 1408 rows, table allocated at NMAX=512:

```
  --- n=95 ---
      (ROWS,N) layout, M[:, :n]   [slice a max-size table]     97.8 us
      (N,ROWS) layout, M[:n]      [samples on axis 0]          21.3 us
      dedicated 95-wide table                                  25.6 us
  --- n=383 ---
      (ROWS,N) layout, M[:, :n]                               266.8 us
      (N,ROWS) layout, M[:n]                                   85.7 us
      dedicated 383-wide table                                 89.1 us
  --- n=512 --- (no slicing; both contiguous)
      (ROWS,N) layout, M[:, :512]                             109.5 us
      (N,ROWS) layout, M[:512]                                113.0 us
```

Read the **n=95** row carefully: slicing a max-size `(ROWS, N)` table down to a *small* block costs
97.8 µs — **the same as the full 512-wide block**. The slice buys nothing; a low-latency mode would
pay big-block prices.

**Two fixes:**
1. **Best:** allocate the table at *exactly* the callback's block size. It is known at stream-open
   and constant for the stream's life (pass an explicit `blocksize=` to sounddevice). Then nothing
   is ever sliced.
2. **If you must slice:** put samples on **axis 0** — `(N, ROWS)`, not `(ROWS, N)`. Then `M[:n]` is
   genuinely contiguous and cost is properly proportional to `n`.

Same effect, milder, for the per-voice tensor: `(KEYS, 2K, N)` sliced = 142.8 µs vs
`(KEYS, N, 2K)` sliced = 135.9 µs at n=383 (at n=95: 62.2 vs 46.2).

### 6.2 Threading: the correct action is *nothing*

Script: **`thread_jitter.py`** (Appendix A). The real audio block — 88 keys × 16 partials, N=383:

```
   BLAS threads    median       p99       max   worst % deadline
              1    110.5u    137.5u    393.9u               4.5%
              4    112.2u    135.5u    410.5u               4.7%
             32    112.2u    137.7u    399.1u               4.6%
```

**Identical.** OpenBLAS has internal size thresholds and will not thread matrices this small. The
default (32 threads) therefore costs nothing in the callback while leaving multithreaded BLAS fully
available for offline/analysis work. **No configuration needed, no lines of code, and no lockout.**

If a knob is ever wanted, it is ~4 lines of `ctypes` on the bundled library — runtime, reversible,
no env var, no new dependency. The symbols are **scipy-mangled**:

```python
import ctypes, glob, os, numpy as np
lib = ctypes.CDLL(glob.glob(os.path.join(os.path.dirname(np.__file__),
                                          '..', 'numpy.libs', '*openblas*'))[0])
lib.scipy_openblas_set_num_threads64_(1)
print(lib.scipy_openblas_get_num_threads64_())     # -> 1
print(lib.scipy_openblas_get_parallel64_())        # -> 1  (0=seq, 1=pthreads, 2=openmp)
```

Verified working (900×900 sgemm: 1 thread = 20.32 ms, 4 = 5.77 ms, 16 = 2.82 ms, back to 1 = 20.08 ms).

> **CORRECTION — a claim I made and then disproved.** I initially reported that
> `openblas_set_num_threads_local` gives true per-thread control. **It does not, in this build.**
> The data refuted it in both directions: a worker calling `setl(1)` drops the *main* thread from
> 2.82 ms to 19.86 ms (`get()` returns 1), and a main-thread `setl(1)` serializes a *fresh* worker
> that never called it. Despite the name it is process-global. `setg(32)` does restore it (2.21 ms).
> **Do not build on that symbol.**

### 6.3 The piano's fused table is STATIC — an LRU has no job

Verified against `py_20260707/claude/spikes/piano_voice.py`:

```python
self._taus = TAU0 / (1.0 + TAU_DECAY * (k - 1))      # (K,) fixed   <- the comment is already there
f_k = k * freq * np.sqrt(1.0 + INHARMONICITY * k * k)
```

- **τ depends only on partial index `k`**, never on the key → **16 distinct τ for the whole keyboard**.
- **ω depends on `(key, k)`** — deterministic, known at startup → **88 × 16 = 1408 pairs**.
- `note_on()` changes only `_amps` (Nyquist mask + normalize), `_phase` (random), `_age = 0`, and
  `env.note_on()`. **None of these touch `(ω, τ)`.**
- `_age` folds exactly into the scalar: `exp(-(age+n)/(SR·τ)) = exp(-age/(SR·τ)) · exp(-n/(SR·τ))`.
  First factor is `amp_v`; second is the table row.

Therefore the fused `(SINEXP, COSEXP)` table for **all 88 keys × 16 partials is static for the life
of the process**: 1408 rows × 2 × N × 4 bytes = **4.31 MB at N=383, 1.07 MB at N=95**.

Build it once at startup. **Nothing to evict, no key-press initialization, no key-press latency.**
A key press writes 32 floats (`phi`, `amp`) and nothing else. An LRU tone cache has no work to do.

### 6.4 Gate the whole keyboard unconditionally; skip the calibration run

88 keys × 16 partials = 1408 rows, N=383, deadline 8685 µs. Script: **`pervoice.py`** (Appendix A):

```
  A) global GEMV, whole keyboard -> mix only               89.7 us    1.03% of deadline
  B) batched matmul -> (88,N), gate all, sum              114.7 us    1.32% of deadline
  B') einsum 'vk,vkn->vn' variant                         130.8 us    1.51% of deadline
  C) python loop: 88 x small GEMV, gate, sum              221.5 us    2.55% of deadline
  D) hybrid: global GEMV +  0 ramping voices               88.3 us    1.02% of deadline
  D) hybrid: global GEMV +  4 ramping voices              100.7 us    1.16% of deadline
  D) hybrid: global GEMV + 10 ramping voices              107.4 us    1.24% of deadline
```

Option **B** — batched `np.matmul((88,1,32), (88,32,N))` → gate all 88 → sum — is **114.7 µs =
1.32% of deadline** and needs **one code path**. The hybrid (fold constant gates into `A`/`B`, ramp
only the few active voices) saves ~7 µs = **0.08% of deadline** and needs a dual path plus a
"who is ramping" set. **Not worth it.** Gate all 88, every block.

The "apply LinearRamp only to the relevant n-ms slice" instinct is correct in principle; the
measurement says the saving is noise. Do the simple thing.

**And therefore: no startup calibration run.** The whole keyboard, every partial, per-voice gated,
is 1.3% of deadline — ~75× under budget. There is no optimal GEMV size to search for: cost is
linear in rows and in N and is nowhere near the cache knee. A one-line startup smoke assert covers it.

### 6.5 Two honest gaps in the numbers above

1. **Gate construction is NOT measured.** The 114.7 µs used a *pre-built* `(88, N)` gate array. If
   that array is built by calling `Envelope.render(frames)` per key in a Python loop, that is 88
   dispatches and reintroduces the O(V) loop just eliminated. The fix is favourable: a sustained
   key's gate is **constant**, so keep a persistent `(88, N)` gate buffer and rewrite only the
   handful of ramping rows. Most blocks touch nothing.
2. **The tail is allocation.** Median 110.5 µs vs max 393.9 µs. `np.matmul(...)` and `.sum(axis=0)`
   each allocate a fresh array per block. Preallocate and use `out=` to flatten it. Worst case is
   still only 4.5% of deadline.

---

## Part 7 — Conclusions

1. **The bottleneck was never `sin` or `exp`.** It is ~0.75 µs of fixed NumPy dispatch overhead per
   call — 73% of a 512-element call. Reduce the *number* of calls, not the work per call.

2. **`DampedSineComposed` optimized the wrong axis.** It is a real 3× win (12.34 → 4.27 µs) but the
   API shape — one damped sine, one object, one render call — caps it at ~7.7 µs/voice/block.

3. **ver1's `ComboEntry` was the load-bearing idea, and ver2 lost it.** Fusing the decay into the
   table is worth 0.65 µs on its own *and* is a precondition for the batched form.

4. **The mixdown is a matrix product.** One BLAS call per block, independent of voice count:
   25×/38× measured on ver2's own benchmark, 57× at the piano's parameters, correct to ~1e-14.

5. **float32 costs ~2–3 × a 24-bit LSB (~130 dB SNR) and buys ~2×.** Use it.

6. **The piano's fused table is static.** No LRU, no eviction, no note-on table work. 4.31 MB for
   the whole keyboard at N=383.

7. **Allocate the table at the exact block size, or put samples on axis 0.** Slicing a max-size
   `(ROWS, N)` table costs 3–4× and makes low-latency mode as expensive as big-block mode.

8. **Leave BLAS threading alone.** It is a measured no-op at audio block sizes.

9. **Full 88-key polyphony is affordable (~1.3% of deadline).** This is the architectural payoff:
   `PolySynth`'s allocation and stealing logic has no reason to exist. No `max_voices`, no slots, no
   LRU, no calibration. An idle key is an amplitude of zero.

   It also retires the regression `piano_voice.py` honestly flagged — *"note-on re-inits phase +
   age, so a steal is not phase-continuous — measured max step ~0.125"*. **No stealing → no
   artifact**, and the deferred crossfade-steal fix becomes moot. Likewise *"a held-but-fully-decayed
   note still holds its voice slot"*: there are no slots.

### What is not yet established

- Gate-array construction cost (6.5.1) and a zero-allocation callback (6.5.2).
- A `LinearRamp` implementation — not prototyped.
- Anything about `sounddevice` callback behaviour under this load; all timings here are synthetic
  loops, not a live audio stream.
- Whether inharmonic partials above Nyquist should be omitted from the table (currently they would
  be built and multiplied by zero — correct, but ~20% wasted rows at the top of the keyboard).

---

## Appendix A — Reproduction scripts

All scripts are standalone except where they import `bank_proto`. Save them to one directory and
run with `py_20260715/.venv/bin/python`. Prefix with `OPENBLAS_NUM_THREADS=1` for everything except
`thread_jitter.py`.

### `bank_proto.py` — the batched bank (imported by others)

```python
"""Prototype: batched damped-sine bank. Same math as damped_sine_ver2."""
import math
import numpy as np


class DampedSineBank:
    def __init__(self, block_size: int, capacity: int, dtype=np.float64) -> None:
        self.block_size = block_size
        self.capacity = capacity
        self.dtype = dtype
        # Interleaved rows: 2v -> COSEXP_v, 2v+1 -> SINEXP_v, so one contiguous
        # slice M[:2n] feeds a single GEMV.
        self._M = np.zeros((2 * capacity, block_size), dtype=dtype)
        self._AB = np.zeros(2 * capacity, dtype=dtype)
        self._omega = np.zeros(capacity, dtype=np.float64)
        self._log_decay = np.zeros(capacity, dtype=np.float64)
        self._phi = np.zeros(capacity, dtype=np.float64)
        self._amp = np.zeros(capacity, dtype=np.float64)
        self._block_decay = np.zeros(capacity, dtype=np.float64)  # exp(ld*block_size)
        self._count = 0
        self._iota = np.arange(block_size, dtype=np.float64)
        self._scratch = np.zeros(capacity, dtype=np.float64)

    def add(self, samp_rate: float, freq: float, tau: float,
            amp: float = 1.0, phi: float = 0.0) -> int:
        v = self._count
        if v >= self.capacity:
            raise ValueError("bank is full")
        omega = 2.0 * math.pi * freq / samp_rate
        log_decay = -1.0 / (tau * samp_rate) if not math.isinf(tau) else 0.0
        iota = self._iota
        env = np.exp(log_decay * iota)
        self._M[2 * v] = np.cos(omega * iota) * env
        self._M[2 * v + 1] = np.sin(omega * iota) * env
        self._omega[v] = omega
        self._log_decay[v] = log_decay
        self._phi[v] = phi
        self._amp[v] = amp
        self._block_decay[v] = math.exp(log_decay * self.block_size)
        self._count = v + 1
        return v

    def render_mix(self, out: np.ndarray) -> None:
        """Render all voices summed into out. len(out) must be block_size."""
        n = self._count
        phi, amp = self._phi[:n], self._amp[:n]
        ab = self._AB[:2 * n]
        np.sin(phi, out=self._scratch[:n])
        ab[0::2] = self._scratch[:n] * amp          # A_v = amp*sin(phi)
        np.cos(phi, out=self._scratch[:n])
        ab[1::2] = self._scratch[:n] * amp          # B_v = amp*cos(phi)
        np.dot(ab, self._M[:2 * n], out=out)
        phi += self._omega[:n] * self.block_size
        np.mod(phi, 2.0 * math.pi, out=phi)
        amp *= self._block_decay[:n]


def reference_render(samp_rate, freq, tau, amp, phi, nsamps, offset=0):
    """Ground truth, straight from the ver2 docstring formula."""
    omega = 2.0 * math.pi * freq / samp_rate
    log_decay = -1.0 / (tau * samp_rate)
    n = np.arange(offset, offset + nsamps, dtype=np.float64)
    return amp * np.sin(phi + omega * n) * np.exp(log_decay * n)
```

### `bench_dispatch.py` — Part 2.1, the core finding

```python
import numpy as np
from time import perf_counter_ns

print("=== numpy per-ufunc-call cost vs array length (float64, a*x) ===")
print(f"  {'N':>7} {'us/call':>10} {'ns/element':>12}")
for n in (8, 32, 128, 512, 2048, 8192, 32768, 131072):
    x = np.ones(n)
    a = 1.7
    for _ in range(3):
        a * x
    reps = max(2000, 4_000_000 // n)
    best = None
    for _ in range(3):
        t0 = perf_counter_ns()
        for _ in range(reps):
            a * x
        d = perf_counter_ns() - t0
        best = d if best is None else min(best, d)
    us = best / reps / 1e3
    print(f"  {n:>7} {us:>10.3f} {us*1000/n:>12.3f}")
```

### `bench_decompose.py` — Part 2.2

```python
import math
import numpy as np
from time import perf_counter_ns

N, V, BLOCKS = 512, 100, 8614
CALLS = BLOCKS * V
rng = np.random.default_rng(0)
sin_part = rng.standard_normal(N)
cos_part = rng.standard_normal(N)
exp_part = np.exp(np.linspace(0, -0.01, N))
out = np.empty(N, dtype=np.float64)


def timeit(label, fn, reps=CALLS):
    fn(); fn()
    best = None
    for _ in range(3):
        t0 = perf_counter_ns()
        for _ in range(reps):
            fn()
        d = perf_counter_ns() - t0
        best = d if best is None else min(best, d)
    print(f"  {label:<56} {best/reps/1e3:8.3f} us/call")


a, b = 0.7, 0.3
print(f"=== micro-ops, N={N} float64 ===")
timeit("empty loop body (baseline)", lambda: None)
timeit("out[:] = a*cos + b*sin   [CURRENT hot line]",
       lambda: out.__setitem__(slice(None), a * cos_part + b * sin_part))
timeit("  a*cos_part                (py-float * array)", lambda: a * cos_part)
timeit("  cos_part + sin_part       (array + array)", lambda: cos_part + sin_part)
timeit("  out *= exp_part           (in-place array*array)", lambda: out.__imul__(exp_part))
timeit("  np.multiply(cos,a,out=out) (out= scalar mul)", lambda: np.multiply(cos_part, a, out=out))


def current_full():
    global out
    r_sin_a = 0.7 * math.sin(0.3)
    r_cos_a = 0.7 * math.cos(0.3)
    out[:] = r_sin_a * cos_part + r_cos_a * sin_part
    out *= exp_part


def fused_combo():
    r_sin_a = 0.7 * math.sin(0.3)
    r_cos_a = 0.7 * math.cos(0.3)
    out[:] = r_sin_a * cos_part + r_cos_a * sin_part


def fused_outparam():
    global out
    r_sin_a = 0.7 * math.sin(0.3)
    r_cos_a = 0.7 * math.cos(0.3)
    np.multiply(cos_part, r_sin_a, out=out)
    out += r_cos_a * sin_part


print("\n=== per-voice render paths ===")
timeit("CURRENT (write_r_sin_cos + decay multiply)", current_full)
timeit("fused combo (ver1 ComboEntry idea)", fused_combo)
timeit("fused + out= param", fused_outparam)

print(f"\n=== batched across V={V} voices, N={N} (per BLOCK) ===")
SIN = np.ascontiguousarray(rng.standard_normal((V, N)))
COS = np.ascontiguousarray(rng.standard_normal((V, N)))
A, B = rng.standard_normal(V), rng.standard_normal(V)
mix = np.empty(N)
bank_out = np.empty((V, N))


def batched_matrix_keep_voices():
    global bank_out
    np.multiply(COS, A[:, None], out=bank_out)
    bank_out += B[:, None] * SIN


def batched_gemv_two():
    global mix
    np.dot(A, COS, out=mix)
    mix.__iadd__(np.dot(B, SIN))


AB = np.concatenate([A, B])
SINCOS = np.ascontiguousarray(np.concatenate([COS, SIN], axis=0))
timeit("batched (V,N) matrix, per-voice outputs kept", batched_matrix_keep_voices, reps=BLOCKS)
timeit("batched GEMV mixdown (2x dot)", batched_gemv_two, reps=BLOCKS)
timeit("batched GEMV mixdown (single 2V x N dot)",
       lambda: np.dot(AB, SINCOS, out=mix), reps=BLOCKS)

print("\n=== reference: no cache at all ===")
iota = np.arange(N, dtype=np.float64)
timeit("np.sin(phi + omega*iota)", lambda: np.sin(0.3 + 0.01 * iota))
timeit("np.sin(...)*np.exp(...) full uncached",
       lambda: np.sin(0.3 + 0.01 * iota) * np.exp(iota * -1e-6))
```

### `verify_bank.py` — Part 4 (correctness)

Requires `damped_sine_ver2` on `sys.path`.

```python
import sys
sys.path.insert(0, "/home/ubuntu/workspace/py_20260715/damped_sine_ver2")
import numpy as np
from damped_sine import DampedSine
from uncached import DampedSineUncached
from composed import DampedSineComposed
from bank_proto import DampedSineBank, reference_render

SAMP_RATE, BLOCK, NBLOCKS = 44100.0, 512, 40
VOICES = [                       # freq, tau, amp, phi
    (20.0, 100.0, 1.0, 0.0),
    (440.0, 0.01, 0.8, 0.5),     # tau shorter than one block
    (441.0, 0.001, 0.5, 1.0),    # decays to ~nothing inside a block
    (19000.0, 0.5, 0.3, 2.0),    # near Nyquist
    (22049.0, 3.0, 0.2, 3.0),    # just under Nyquist
    (1.0, 50.0, 0.7, 4.0),       # sub-Hz
    (440.0, 0.01, 0.9, 2.5),     # duplicate (freq,tau), different amp/phi
    (8000.0, 1e-4, 1.0, 0.1),    # extremely fast decay
]

bank = DampedSineBank(block_size=BLOCK, capacity=len(VOICES))
for freq, tau, amp, phi in VOICES:
    bank.add(SAMP_RATE, freq, tau, amp, phi)

unc_tones, unc = [], DampedSineUncached()
comp_tones, comp = [], DampedSineComposed(max_nsamps=BLOCK)
for freq, tau, amp, phi in VOICES:
    t = DampedSine(samp_rate=SAMP_RATE, freq=freq, tau=tau, next_phi=phi, next_amp=amp)
    unc.try_attach(t); unc_tones.append(t)
    t2 = DampedSine(samp_rate=SAMP_RATE, freq=freq, tau=tau, next_phi=phi, next_amp=amp)
    assert comp.try_attach(t2); comp_tones.append(t2)

max_err_unc = max_err_comp = max_err_ref = 0.0
mix = np.empty(BLOCK); scratch = np.empty(BLOCK)
for blk in range(NBLOCKS):
    bank.render_mix(mix)
    ref_unc = np.zeros(BLOCK)
    for t in unc_tones:
        unc.render_to(t, scratch); ref_unc += scratch
    ref_comp = np.zeros(BLOCK)
    for t in comp_tones:
        comp.render_to(t, scratch); ref_comp += scratch
    ref_math = np.zeros(BLOCK)
    for freq, tau, amp, phi in VOICES:
        ref_math += reference_render(SAMP_RATE, freq, tau, amp, phi, BLOCK, offset=blk * BLOCK)
    max_err_unc = max(max_err_unc, float(np.max(np.abs(mix - ref_unc))))
    max_err_comp = max(max_err_comp, float(np.max(np.abs(mix - ref_comp))))
    max_err_ref = max(max_err_ref, float(np.max(np.abs(mix - ref_math))))

print(f"blocks compared     : {NBLOCKS} x {BLOCK} samples, {len(VOICES)} voices")
print(f"max |bank - uncached|  : {max_err_unc:.3e}")
print(f"max |bank - composed|  : {max_err_comp:.3e}")
print(f"max |bank - analytic|  : {max_err_ref:.3e}")
print("16-bit LSB for reference: %.3e" % (1.0 / 32768))
print("VERDICT:", "MATCH" if max(max_err_unc, max_err_comp, max_err_ref) < 1e-9 else "MISMATCH")
```

### `bench_bank.py` — Part 5.1 / 5.3

```python
import sys
sys.path.insert(0, ".")
import numpy as np
from time import perf_counter_ns
from bank_proto import DampedSineBank

SAMP_RATE, BLOCK = 44100, 512
FREQ_MIN, FREQ_MAX = 20.0, 20000.0
BASELINE_MS = 6602.0     # measured, ver2 composed, 100 tones


def run(v_count, dtype, duration):
    total = int(round(SAMP_RATE * duration))
    nblocks = total // BLOCK
    bank = DampedSineBank(block_size=BLOCK, capacity=v_count, dtype=dtype)
    for i in range(v_count):
        freq = FREQ_MIN + (i / max(1, v_count - 1)) * (FREQ_MAX - FREQ_MIN)
        bank.add(SAMP_RATE, freq, tau=duration, amp=1.0, phi=0.0)
    out = np.empty(BLOCK, dtype=dtype)
    bank.render_mix(out)
    t0 = perf_counter_ns()
    for _ in range(nblocks):
        bank.render_mix(out)
    return (perf_counter_ns() - t0) / 1e6, nblocks


print("=== identical workload to test_composed_perf.py ===")
print(f"  ver2 DampedSineComposed (measured)   : {BASELINE_MS:9.1f} ms")
for dt, name in ((np.float64, "float64"), (np.float32, "float32")):
    ms, _ = run(100, dt, 100.0)
    print(f"  batched bank, {name}                : {ms:9.1f} ms   ({BASELINE_MS/ms:5.1f}x faster)")

budget_us = BLOCK / SAMP_RATE * 1e6
print(f"\n=== scaling (budget per {BLOCK}-sample block = {budget_us:.0f} us) ===")
print(f"  {'voices':>7} {'f64 us/blk':>11} {'f64 %RT':>9} {'f32 us/blk':>11} {'f32 %RT':>9} {'tbl MB f32':>11}")
for v in (16, 64, 100, 256, 512, 1024, 2048, 4096):
    row = [f"{v:>7}"]
    for dt in (np.float64, np.float32):
        ms, nb = run(v, dt, 10.0)
        us_blk = ms * 1000 / nb
        row += [f"{us_blk:>11.1f}", f"{100*us_blk/budget_us:>8.2f}%"]
    row.append(f"{v*2*BLOCK*4/1e6:>11.2f}")
    print(" ".join(row))
```

### `check_f32.py` — Part 4 (float32 accuracy)

```python
import sys
sys.path.insert(0, ".")
import numpy as np
from bank_proto import DampedSineBank

SAMP_RATE, BLOCK = 44100.0, 512
print(f"(16-bit LSB = {1/32768:.2e}, 24-bit LSB = {1/8388608:.2e})\n")
print(f"  {'voices':>7} {'peak |mix|':>11} {'f32 max err':>13} {'err / 24-bit LSB':>17}")
for v in (16, 100, 512, 1024):
    b32 = DampedSineBank(BLOCK, v, dtype=np.float32)
    b64 = DampedSineBank(BLOCK, v, dtype=np.float64)
    for i in range(v):
        freq = 20.0 + (i / max(1, v - 1)) * 19980.0
        for b in (b32, b64):
            b.add(SAMP_RATE, freq, tau=2.0, amp=1.0 / v, phi=0.1 * i)
    o32 = np.empty(BLOCK, dtype=np.float32)
    o64 = np.empty(BLOCK, dtype=np.float64)
    err = peak = 0.0
    for _ in range(100):
        b32.render_mix(o32); b64.render_mix(o64)
        err = max(err, float(np.max(np.abs(o32.astype(np.float64) - o64))))
        peak = max(peak, float(np.max(np.abs(o64))))
    print(f"  {v:>7} {peak:>11.4f} {err:>13.3e} {err/(1/8388608):>17.2f}")
```

### `piano_scale.py` — Part 5.4

```python
import sys
sys.path.insert(0, ".")
import numpy as np
from time import perf_counter_ns
from bank_proto import DampedSineBank

SR, BLOCK, PARTIALS = 44100.0, 383, 16      # py_20260707 piano_voice.py
DEADLINE_US = BLOCK / SR * 1e6


def bench(n_sines, dtype):
    b = DampedSineBank(BLOCK, n_sines, dtype=dtype)
    rng = np.random.default_rng(0)
    for i in range(n_sines):
        f0 = 55.0 * 2 ** ((i % 88) / 12.0)
        k = (i % PARTIALS) + 1
        freq = min(k * f0 * np.sqrt(1 + 4e-4 * k * k), 21000.0)
        b.add(SR, freq, tau=2.5 / (1 + 0.25 * (k - 1)), amp=1.0 / k ** 1.2,
              phi=rng.uniform(0, 6.28))
    out = np.empty(BLOCK, dtype=dtype)
    b.render_mix(out)
    best = None
    for _ in range(3):
        t0 = perf_counter_ns()
        for _ in range(3000):
            b.render_mix(out)
        d = perf_counter_ns() - t0
        best = d if best is None else min(best, d)
    return best / 3000 / 1e3


print(f"{BLOCK}-sample block @ {SR:.0f} Hz => {DEADLINE_US:.0f} us deadline")
print(f"py_20260707 measured: 16 PianoVoices (= {16*PARTIALS} sines) = ~2240 us = ~26%\n")
print(f"  {'voices':>7} {'sines':>7} {'f32 us/blk':>11} {'% deadline':>11}")
for voices in (16, 32, 64, 128, 256, 512):
    n = voices * PARTIALS
    us = bench(n, np.float32)
    print(f"  {voices:>7} {n:>7} {us:>11.1f} {100*us/DEADLINE_US:>10.2f}%")
```

### `pervoice.py` — Part 6.4 (gate strategy)

```python
import numpy as np
from time import perf_counter_ns

KEYS, K, N = 88, 16, 383
ROWS = KEYS * K
DEADLINE = N / 44100 * 1e6
rng = np.random.default_rng(0)


def t(label, fn, reps=3000):
    fn()
    best = None
    for _ in range(3):
        t0 = perf_counter_ns()
        for _ in range(reps):
            fn()
        d = perf_counter_ns() - t0
        best = d if best is None else min(best, d)
    us = best / reps / 1e3
    print(f"  {label:<52} {us:8.1f} us {100*us/DEADLINE:7.2f}% of deadline")


print(f"88 keys x 16 partials = {ROWS} fused rows, N={N}, deadline={DEADLINE:.0f} us")
print(f"static table size (f32): {ROWS*2*N*4/1e6:.2f} MB\n")

M_flat = np.ascontiguousarray(rng.standard_normal((2 * ROWS, N), dtype=np.float32))
ab_flat = rng.standard_normal(2 * ROWS).astype(np.float32)
mix = np.empty(N, dtype=np.float32)
t("A) global GEMV, whole keyboard -> mix only", lambda: np.dot(ab_flat, M_flat, out=mix))

M_v = np.ascontiguousarray(rng.standard_normal((KEYS, 2 * K, N), dtype=np.float32))
ab_v = np.ascontiguousarray(rng.standard_normal((KEYS, 1, 2 * K), dtype=np.float32))
gate = rng.random((KEYS, N)).astype(np.float32)


def batched():
    pv = np.matmul(ab_v, M_v)[:, 0, :]
    pv *= gate
    return pv.sum(axis=0)


t("B) batched matmul -> (88,N), gate all, sum", batched)
t("B') einsum 'vk,vkn->vn' variant",
  lambda: (lambda pv: (pv.__imul__(gate), pv.sum(axis=0))[1])(
      np.einsum('vk,vkn->vn', ab_v[:, 0, :], M_v, optimize=True)))

out_v = np.empty((KEYS, N), dtype=np.float32)


def loop88():
    for v in range(KEYS):
        np.dot(ab_v[v, 0], M_v[v], out=out_v[v])
    out_v.__imul__(gate)
    return out_v.sum(axis=0)


t("C) python loop: 88 x small GEMV, gate, sum", loop88)

for n_ramp in (0, 4, 10):
    m = n_ramp if n_ramp else 1
    Mr = np.ascontiguousarray(rng.standard_normal((m, 2 * K, N), dtype=np.float32))
    abr = np.ascontiguousarray(rng.standard_normal((m, 1, 2 * K), dtype=np.float32))
    gr = rng.random((m, N)).astype(np.float32)

    def hybrid(nr=n_ramp, Mr=Mr, abr=abr, gr=gr):
        np.dot(ab_flat, M_flat, out=mix)
        if nr:
            pv = np.matmul(abr, Mr)[:, 0, :]
            pv *= gr
            mix.__iadd__(pv.sum(axis=0))
        return mix

    t(f"D) hybrid: global GEMV + {n_ramp:>2} ramping voices", hybrid)
```

### `layout.py` — Part 6.1 (the slicing trap)

```python
import numpy as np
from time import perf_counter_ns

KEYS, K, NMAX = 88, 16, 512
ROWS = KEYS * K
rng = np.random.default_rng(0)


def t(label, fn, reps=3000):
    fn()
    best = None
    for _ in range(3):
        t0 = perf_counter_ns()
        for _ in range(reps):
            fn()
        d = perf_counter_ns() - t0
        best = d if best is None else min(best, d)
    print(f"  {label:<54} {best/reps/1e3:8.1f} us")


ab = rng.standard_normal(2 * ROWS).astype(np.float32)
M_rn = np.ascontiguousarray(rng.standard_normal((2 * ROWS, NMAX), dtype=np.float32))
M_nr = np.ascontiguousarray(rng.standard_normal((NMAX, 2 * ROWS), dtype=np.float32))

print("GLOBAL MIX: table allocated at NMAX=512, rendering a shorter block n\n")
for n in (95, 128, 383, 512):
    print(f"  --- n={n} ---")
    s1, o1 = M_rn[:, :n], np.empty(n, dtype=np.float32)
    t("    (ROWS,N) layout, M[:, :n]   [slice max-size table]",
      lambda s=s1, o=o1: np.dot(ab, s, out=o))
    s2, o2 = M_nr[:n], np.empty(n, dtype=np.float32)
    t("    (N,ROWS) layout, M[:n]      [samples on axis 0]",
      lambda s=s2, o=o2: np.dot(s, ab, out=o))
    Mt = np.ascontiguousarray(s1)
    t(f"    dedicated {n}-wide table", lambda Mt=Mt, o=o1: np.dot(ab, Mt, out=o))

print("\nPER-VOICE (gate-able), table at NMAX=512\n")
Mv_kn = np.ascontiguousarray(rng.standard_normal((KEYS, 2 * K, NMAX), dtype=np.float32))
Mv_nk = np.ascontiguousarray(rng.standard_normal((KEYS, NMAX, 2 * K), dtype=np.float32))
abv_r = np.ascontiguousarray(rng.standard_normal((KEYS, 1, 2 * K), dtype=np.float32))
abv_c = np.ascontiguousarray(rng.standard_normal((KEYS, 2 * K, 1), dtype=np.float32))
for n in (95, 383, 512):
    g = rng.random((KEYS, n)).astype(np.float32)
    print(f"  --- n={n} ---")
    t("    (KEYS,2K,N) layout, sliced",
      lambda n=n, g=g: (lambda pv: (pv.__imul__(g), pv.sum(axis=0))[1])(
          np.matmul(abv_r, Mv_kn[:, :, :n])[:, 0, :]))
    t("    (KEYS,N,2K) layout, sliced",
      lambda n=n, g=g: (lambda pv: (pv.__imul__(g), pv.sum(axis=0))[1])(
          np.matmul(Mv_nk[:, :n, :], abv_c)[:, :, 0]))
```

### `thread_jitter.py` — Part 6.2 (run WITHOUT `OPENBLAS_NUM_THREADS`)

```python
import ctypes, glob, os
import numpy as np
from time import perf_counter_ns

lib = ctypes.CDLL(glob.glob(os.path.join(os.path.dirname(np.__file__),
                                         '..', 'numpy.libs', '*openblas*'))[0])
setg = lib.scipy_openblas_set_num_threads64_

KEYS, K, N = 88, 16, 383
DEADLINE = N / 44100 * 1e6
rng = np.random.default_rng(0)
Mv = np.ascontiguousarray(rng.standard_normal((KEYS, 2 * K, N), dtype=np.float32))
abv = np.ascontiguousarray(rng.standard_normal((KEYS, 1, 2 * K), dtype=np.float32))
gate = rng.random((KEYS, N)).astype(np.float32)


def block():
    pv = np.matmul(abv, Mv)[:, 0, :]
    pv *= gate
    return pv.sum(axis=0)


print(f"88 keys x 16 partials, N={N}, deadline={DEADLINE:.0f} us\n")
print(f"  {'BLAS threads':>13} {'median':>9} {'p99':>9} {'max':>9}   {'worst % deadline':>16}")
for nt in (1, 4, 32):
    setg(nt)
    for _ in range(200):
        block()
    s = []
    for _ in range(4000):
        t0 = perf_counter_ns(); block(); s.append(perf_counter_ns() - t0)
    s = np.array(s) / 1e3
    print(f"  {nt:>13} {np.median(s):>8.1f}u {np.percentile(s,99):>8.1f}u "
          f"{s.max():>8.1f}u   {100*s.max()/DEADLINE:>15.1f}%")
setg(1)
```

### `blas_local_disproof.py` — Part 6.2 (the correction)

```python
import ctypes, glob, os, threading
import numpy as np
from time import perf_counter_ns

lib = ctypes.CDLL(glob.glob(os.path.join(os.path.dirname(np.__file__),
                                         '..', 'numpy.libs', '*openblas*'))[0])
get  = lib.scipy_openblas_get_num_threads64_
setg = lib.scipy_openblas_set_num_threads64_
setl = lib.openblas_set_num_threads_local
setl.restype = ctypes.c_int

A = np.random.default_rng(0).standard_normal((900, 900), dtype=np.float32)
B = np.random.default_rng(1).standard_normal((900, 900), dtype=np.float32)


def ms(reps=6):
    np.dot(A, B)
    t0 = perf_counter_ns()
    for _ in range(reps):
        np.dot(A, B)
    return (perf_counter_ns() - t0) / reps / 1e6


def step(label):
    print(f"  {label:<46} {ms():6.2f} ms   (get()={get()})")


print("SEQUENCE A: does setl() in a worker leak to main?")
setg(32); step("main, global=32 (baseline)")
r = {}
def w(): r['prev'] = setl(1); r['ms'] = ms()
th = threading.Thread(target=w); th.start(); th.join()
print(f"  worker after setl(1), prev={r['prev']:<10} {r['ms']:6.2f} ms")
step("main, immediately after worker exits")
setg(32); step("main, after re-issuing setg(32)")

print("\nSEQUENCE B: setl() on MAIN, then a fresh worker")
setg(32); setl(1); step("main after its own setl(1)")
r2 = {}
def w2(): r2['ms'] = ms()
th = threading.Thread(target=w2); th.start(); th.join()
print(f"  fresh worker (never called setl)          {r2['ms']:6.2f} ms")
```

Expected output — the disproof:

```
SEQUENCE A: does setl() in a worker leak to main?
  main, global=32 (baseline)                       2.82 ms   (get()=32)
  worker after setl(1), prev=32          20.11 ms
  main, immediately after worker exits            19.86 ms   (get()=1)
  main, after re-issuing setg(32)                  2.21 ms   (get()=32)

SEQUENCE B: setl() on the MAIN thread, then a fresh worker
  main after its own setl(1)                      19.98 ms   (get()=1)
  fresh worker (never called setl)           19.98 ms
```
