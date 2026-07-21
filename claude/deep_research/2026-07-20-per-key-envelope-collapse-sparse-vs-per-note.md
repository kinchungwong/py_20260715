# Per-key envelope collapse: block-diagonal / sparse `(2·ΣM,N)→(88,N)` vs per-note synthesis

**Date:** 2026-07-20
**Type:** Reconstructed decision record (previous conversation context was lost; rebuilt by
searching the raw session logs + on-disk artifacts).
**Sources reconstructed from:**
- `human/outlines/20260717_0754am_pipeline.md` — the pipeline outline + Claude's appended review
  notes (the substantive block-matrix analysis; **the durable on-disk record**).
- Raw session `f3xxxxxx-xxxx-xxxx-xxxx-xxxxxxxxx91d` (project dir
  `-home-ubuntu-workspace-py-20260715-piano-ver4`), user message ~line 446 (the design decision).
- `claude/deep_research/2026-07-17-additive-synthesis-gemv-batching.md` Part 6.4 (`pervoice.py`) —
  the measured collapse options.

---

## The core tension

The single-GEMV **full collapse** `ab(2·ΣM,) @ M(2·ΣM,N) → mix(N,)` (~1.0% of the 8685 µs
deadline for 88 keys × 16 partials) is the fastest possible mixdown **but produces one mono
output** — it forfeits per-key envelopes. The moment you want a **per-key LinearRamp**
(`clip(m·x + c, 0, 1)`, with slope/level per key), the 88 keys must stay *separate through the
mixdown*: you need the intermediate collapse `(2·ΣM, N) → (88, N)`, gate each of the 88 rows with
its own ramp, then sum. Realizing that collapse cheaply is the whole problem.

## The `(88, 2·ΣM)` collapse matrix is block-diagonal / ~99% sparse

Write the collapse as `C(88, 2·ΣM) @ M(2·ΣM, N) = (88, N)`. Row *k* (key *k*) has nonzeros **only**
in the ~2·16 = 32 columns of key *k*'s partials; out of ~2·ΣM ≈ 2816 columns, ~99% are zero. `C`
is not a static 0/1 membership matrix — it holds the **per-block A/B coefficients**
(`A_p = amp·decay^age·cos(phi+age·ω)`, and the sin twin) at fixed block-diagonal positions.

Three ways to realize it, with their cost profiles (`pipeline.md` review notes, lines 124–140):

| approach | cost profile |
|---|---|
| **Dense block-diagonal GEMM** `C @ M` | hits BLAS speed, but **~99% of flops land on off-diagonal zeros** — wasted work |
| **Sparse (SciPy CSR / block-sparse)** | NumPy has **no** sparse/block-sparse type; SciPy CSR could store just the 32-per-row nonzeros — **rejected** (below) |
| **Segment-sum** `np.add.reduceat` over per-key partial ranges | **no wasted flops**, but runs at NumPy elementwise (memory-bound) speed, not BLAS |

Measured (`pervoice.py`, N=383, uniform 16 partials/key, f32, 1 BLAS thread):

| option | µs/block | % deadline |
|---|---|---|
| A) global GEMV → mix only (full collapse, no per-key env) | 88.7 | 1.02% |
| **B) batched matmul → (88,N), gate all, sum** | **113.4** | **1.31%** |
| B') einsum `'vk,vkn->vn'` | 130.1 | 1.50% |
| C) python loop: 88 × small GEMV, gate, sum | 223.7 | 2.58% |
| D) hybrid: global GEMV + 0 / 4 / 10 ramping voices | 88.8 / 101.6 / 108.5 | 1.02–1.25% |

Note: option B's clean 3D batch `(88,1,2K)@(88,2K,N)` assumes **uniform** K per key (rectangular).
For **ragged** per-key partial counts (the physical reality — bass ~40 partials, treble ~4) you
cannot use a clean 3D batch; you fall back to the block-diagonal `(88, 2·ΣM)` (sparse) **or**
`np.add.reduceat` over the ragged per-key ranges.

## Why SciPy CSR was rejected

- **New heavy dependency + wrong-shape kernels.** A `scipy.sparse` CSR `@` still crosses
  Python→C once per block, and its SpMM kernels aren't tuned for these tiny, tall-skinny,
  block-*regular* shapes — you'd likely lose the dense-BLAS speed you were paying for.
- **The sparsity is regular, not arbitrary.** CSR targets irregular sparsity; here the nonzeros
  are contiguous 32-wide blocks, which `np.add.reduceat` (segment-sum) exploits directly in NumPy
  with no dependency and no wasted flops. CSR is the wrong tool even when you do want "sparse."
- **Per-note synthesis dissolves the matrix entirely** (the chosen path). Synthesize **one note at
  a time** (its ~16 partials in one small GEMV `ab(2P,) @ M_note(2P,N)`), then multiply that note's
  `(N,)` output by its scalar ramp before adding to the mix. There is no `(88, 2·ΣM)` matrix to be
  sparse about — the block-diagonal structure *becomes the outer Python loop over notes*, each note
  is already separate so the envelope is trivial, dispatch is amortized across each note's partials,
  and silent keys are simply skipped. Tradeoff accepted: gives up the full-collapse ceiling, wins
  back per-note envelopes + skip-silent + easy re-trigger/crossfade.

## "Most keys aren't played" — two sparsities, both handled

1. **Sparse in *sounding* voices** (few keys down): sum only active-note partials.
   - **Dynamic *size*, static *allocation*** — never reallocate the mix buffer in the callback (that
     reintroduces the allocation latency tail, median 110 µs vs max 394 µs). Preallocate at 88 keys;
     operate on a variable prefix `[:32·A]` of *A* active keys.
   - **Contiguous whole-key-block gather** — keep partials grouped by key in `M` (key *k* owns rows
     `[32k:32k+32]`) so selecting active keys pulls whole blocks (~123 KB at A=10), cheap, and lands
     data contiguously (also sidesteps the column-slice `M[:, :nsamps]` layout trap → put samples on
     axis 0).
2. **Sparse in *ramping* voices** (the elegant one): most sounding voices sit in steady sustain
   (ramp clipped at 1.0); only a handful are mid attack/release. **Hybrid (option D):** run all
   steady voices through the cheap full-collapse global GEMV into the mix, give *separate* rows only
   to the few voices currently ramping, gate those, add them → 88.8 → 108.5 µs for 0 → 10 ramping.
   You pay for per-key envelopes only on the handful of keys that need one at that instant.

## Connection to the 2026-07-20 ver5 benchmark

Same axis, now quantified (`piano_ver5/bench/`, see
`2026-07-20_render_np_m_cached_benchmark.md`):
- `batched_soa` **is** the full collapse (one cos/sin + one GEMV → single `(N,)` mix) — structurally
  cannot apply per-voice envelopes. Fastest (6.4× over per-note at M=16/V=88).
- `ver5_per_note_cached` is the per-note choice — separate `(N,)` per voice, envelope-ready — at
  ~8% of deadline for 88 keys.
- So the benchmark measures the **price of keeping voices separable for the LinearRamp**: ≈6×. The
  `(88,N)` gating collapse (option B, 1.3%) is the un-built middle path that keeps per-key envelopes
  while still collapsing partials.

## Open path (not yet built)

Prototype two more benchmark strategies to measure the real per-key-envelope collapse on **ragged**
partial counts, against the per-note path:
1. **`reduceat` segment-sum** collapse `(2·ΣM,N) → (88,N)` (NumPy-native, no SciPy, no wasted flops).
2. **Hybrid** — global GEMV for steady voices + separate rows for the few ramping ones.

These would tell us whether the middle path beats per-note enough to justify its bookkeeping — the
"earn the complexity with a measurement" guardrail from the pipeline review.

---

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
