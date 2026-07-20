# Design notes: functional honesty, reentrancy, and testing strategy

**Date:** 2026-07-19 18:43 PDT
**Scope:** `piano_ver4` (frozen reference) → `piano_ver5` (functionally-transparent rework)
**Author:** Claude Code (Opus 4.8, 1M context), as a discussion partner — no code changes were made in this session.

This is a record of a design discussion, not an implementation report. It captures the
reasoning behind the direction `piano_ver5` is taking and lists concrete next steps. It
covers three threads: (1) how to test a random-phase synth without dragging in heavy
machinery, (2) what "functionally honest" should mean for this codebase, and (3) a review
of the `piano_ver5/render.py` conceptual sketch — the vocabulary it gets right, the one it
gets subtly wrong, and the design fork it exposes.

---

## 1. Testing a random-phase synth

### The two goals are different problems

There are two testing goals that look like one but aren't:

1. **Equivalence** — does the scalar OO reference (`note_cache.py`) produce the same samples
   as the vectorized path (`piano_render.py`, future `NoteCacheVec`)?
2. **Absolute correctness** — does *any* implementation put the right energy at the right
   frequencies with the right decay?

Only #2 has anything to do with FFTs. And #1 is where nearly all the real bug risk lives —
the sin/cos-swap class of error (the `tup[3]→tup[2]` normalization bug earlier, a coefficient
pairing swap, `p_amp` baked twice, envelope misapplied, a phase/decay off-by-one across block
boundaries). Every one of those is caught by a **same-phase, sample-level diff** at a `1e-6`
tolerance — no spectral machinery.

### Phase determinism: make it *state*, not *behavior*

The random phase hook currently lives inside the data model. That is an *effect* embedded in a
data object, which is the thing that makes the model non-deterministic and awkward to test.
Three ways out, in increasing cleanliness:

- **Seed the RNG** — plumb a seeded `Random` through config. This is the unwieldy path: seed
  threading, reset-between-tests, and the killer — two implementations only match if they
  *draw in the same order*, which is fragile coupling. Avoid.
- **Phases are already inspectable state — copy them (zero refactor, works today).** `next_phi`
  is public mutable state. For an equivalence test you don't control the RNG at all: attack the
  note in impl A, copy each `next_phi` into impl B's states, render both, diff. `piano_render.py`'s
  `_test_waveform_identity` already does exactly this against the analytic reference.
- **Push randomness to the edge (clean long-term move).** Make `attack(note, phases)` *receive*
  phases rather than draw them. The model becomes fully deterministic ("functional core"); the
  engine is the one place that calls the RNG ("imperative shell"). Production draws random; tests
  pass fixed or shared arrays; `hypothesis` can *generate* phase arrays and feed both impls.

Edge-injection pays off twice: the `RETRIG_YIELD`/crossfade work *requires fresh random phases on
every retrigger*. If randomness lives inside `NoteState`, every retrigger re-invokes the hook and
the untestable-determinism problem returns on the hardest-to-reason path. If phases arrive at the
boundary, the retrigger path just receives an array like everything else.

### You probably don't need the FFT port

The *expensive* part of FFT-based analysis in `py_20260707` is **discovering unknown peaks** —
windowing, full transform, peak-picking, bin-to-partial attribution. But in this project **the
model hands you the exact partial frequencies.** You never have to discover anything.

So for phase-independent spectral checks, replace "FFT the block and find the peaks" with
"project the block onto the handful of `omega_k` you already know" — a tiny sparse DFT, a few dot
products against `cos(omega_k·n)`/`sin(omega_k·n)`. The magnitude at each known bin is your
phase-independent amplitude check (and the two components recover `next_phi` if wanted). ~5 lines,
not a framework.

Below even that: a **Parseval/energy scalar**. The sum of squares of a block is approximately
phase-independent for well-separated partials (cross-terms wash out over the block) and equals
roughly `Σ amp_k²·(decay integral)/2`. One number, one assertion — coarse, but it catches missing
partials and amplitude-scaling errors instantly with zero spectral code. Caveat: the wash-out is
not exact for close/low partials over a short block, so it is a gross-error tripwire, not a tight
bound. It has a second use: **equal-power crossfade validation is inherently a "total energy stays
constant regardless of phase relationship" assertion** — the same energy scalar, not an FFT.

### The tiered testing ladder (cheapest first)

- **Tier 0** — model-parameter tests: frequencies, `p_amp`, `tau` are deterministic; test them
  directly, no rendering.
- **Tier 1** — same-phase cross-impl diff (copy `next_phi`): catches the whole
  sin/cos/baking/envelope bug class.
- **Tier 2** — analytic reference (`_reference_mix`) with known phases: absolute grounding.
- **Tier 3 (optional)** — known-frequency single-bin projection + energy scalar: phase-independent
  spot-check that trusts no implementation.

The full FFT framework sits *above* Tier 3 and, for correctness, need never be built. Reserve it
for research/visualization. The real oracle is **the second implementation + the closed-form
reference** — differential testing — which is what makes "small and human-understandable" a
winning constraint here rather than a limiting one.

---

## 2. What "functionally honest" should mean here

### Reframe the target: "no ambient effects," not "immutable"

The trap with "go functional" in a NumPy audio callback is hearing *immutable* and then allocating
a new state object per block per note (88 notes × per-partial arrays every 8.7 ms) — the GC cost
eats the win the cache was built for. Purity and the preallocated-buffer performance model look
like enemies.

They aren't, if the goal is **referential transparency of transitions plus eviction of ambient
effects**, rather than immutability of containers. A function is "honest" here if:

- it reads only its explicit arguments (no RNG, no clock, no module global, no hidden `self` state
  it didn't declare it uses), and
- it writes only its return value **or an explicitly named `out=` buffer**.

That "pure except for a declared `out=`" contract is the standard pragmatic escape hatch for
numerical code. Make it an explicit house rule so the *exact* mutation boundary is known.
Crucially, **methods are still fine** — a method that returns a new value, or writes its named out
and reads only its own data, is honest. What is being evicted is the RNG-in-`attack`, not object
orientation. So the human-readable OO of `note_cache.py` can be functionally honest; the goals stop
conflicting once the enemy is correctly named as *hidden effects*.

### The highest-leverage split: definition vs. cursor

`NoteState` today conflates two things with different lifetimes and mutability:

- **Note definition / cache** — which key, its partials, the `expcos`/`expsin` rows. Immutable,
  shareable, thread-safe, pure. Computed once.
- **Playback cursor** — `next_phi`, `next_amp`, `level`, `trend`. Tiny, mutable, evolves per block.

Splitting these pays off three ways: (1) the definition becomes genuinely immutable at no perf
cost; (2) the cursor becomes small enough that even value semantics on it is cheap, so immutability
becomes a per-struct choice rather than all-or-nothing; (3) the crossfade design falls out — **two
voices = two cursors over one shared immutable definition** — expressing structurally what was
previously the happy accident that "voices share cache rows."

### Where the current behaviors want to live

- **Randomness (attack phase)** → out of the model, arriving as *data on the event* (note-on and
  retrigger carry phases or a seed; the shell draws them).
- **Frequency/amp/tau derivation** → already pure in `PianoModel`; it is the model the rest should
  imitate.
- **Trend state machine** (`attack`/`release`/`advance` branching) → a pure transition
  `(trend, level, event) → (trend, level)`, ideally small enough to read as a table.
- **Auto-silence check** (`level·√Σamp² < audible_amp`) → a pure predicate, not a side effect
  buried in `advance`.
- **Envelope** → a pure function of `(trend, level, ramp, iota)`.
- **Render kernel** → pure-with-`out=`.

### The convergence worth exploiting

The SoA plan for `NoteCacheVec` — holding `next_phi`/`next_amp` as vectors — is *simultaneously*
the faster representation and the more functionally honest one. The per-block advance stops being
"loop over PartialState objects mutating each" and becomes one vectorized transition (vectors in,
vectors out, a couple of NumPy ops, no per-partial Python objects, no hidden state). Purity and
performance point the same direction here — the honest version is the fast path, not a tax.

### The payoff

Once randomness is event data and transitions are pure, the whole synth becomes a pure function of
**(event stream, including phase draws) → audio**. That gives record/replay: snapshot an
event+phase sequence and regression-test whole passages sample-for-sample — including the *stateful*
retrigger/crossfade path, normally the hardest thing to pin down. The differential-testing oracle
plugs straight into this.

### Sequencing caution

Do it in two passes, not one. **First** make every transition honest (no ambient effects, declared
`out=`) while leaving object shapes roughly as they are — that alone removes the data-dependency
gotchas, because the gotchas *are* the hidden effects. **Then**, separately and per-struct, decide
where immutability/value-semantics actually earns its keep (cursor: probably yes; big cache arrays:
never). Keeping the two decisions apart stops the refactor from becoming a rewrite, and avoids
dissolving the domain model into anemic bags-of-data + free functions (which would cost the
readability the README explicitly values).

---

## 3. Review of the `piano_ver5/render.py` sketch

The sketch is a deliberate purity/performance ladder of the single-partial and multi-partial render
kernels: `render_1_1` (scalar, pure) → `render_1_n_py`/`render_1_n_np` (pure, allocating) →
`Render_1_N_NpBuf` (mutable scratch member) → `Render_1_N_NpCache` / `Render_M_N_NpCache`
(immutable cache + mutable scratch). Naming convention: `A_B` = partial-count (`1`/`M`) ×
sample-count (`1`/`N`).

### The math checks out across all four variants

The identity `amp·exp(log_decay·n)·sin(omega·n + phi)` is decomposed consistently everywhere:
`amp·cos(phi)` weights the `expsin` row and `amp·sin(phi)` weights the `expcos` row. The `M_N`
stacking (`expsin` rows on top, `expcos` below) pairs correctly with the two coefficient halves.
This agrees with `note_cache.py` and `piano_render.py`, so the whole family is a valid set of
mutual differential-test oracles.

### The one concept that is subtly off: "cache ⇒ reentrant"

The caveat "a cache is immutable and thus functions using it are reentrant" is correct for a **free
function** that reads the cache and writes only `out`. But `Render_1_N_NpCache` and
`Render_M_N_NpCache` are **not reentrant as written**, because each bundles a *mutable scratch
member* (`self._buf`, `self._n_buf`) alongside the immutable cache. **Reentrancy is a whole-function
property**: it concerns the union of everything the call touches, and one mutable field poisons it
for the entire method. Two concurrent calls on the same instance race on that scratch, immutable
cache notwithstanding.

Precise statement: the *cache array* is shareable and reentrancy-safe; the *class that owns both a
cache and a scratch buffer* is not. The sketch re-coupled the two things the split was meant to
separate — it moved the mutable part from "the whole object" to "one member," but the reentrancy
consequence is the same.

### Vocabulary: name four lifetimes, not two

The words "cache" and "buffer" are currently carrying four distinct lifetimes:

- **cache / precomputed table** — immutable, shared across voices and calls, reentrancy-safe
  (`_expsin`, `_expcos`, `_stacked`).
- **state / cursor** — the small per-voice evolving thing (`next_amp`, `next_phi`).
- **scratch / workspace** — mutable, transient, belongs to *one render pass*; the sole reentrancy
  hazard (`_buf`, `_n_buf`).
- **out** — caller-owned destination.

The reentrancy rule then reads cleanly: a render is reentrant iff it *reads* {cache, state} and
*writes* {out}, with scratch either local or borrowed-per-call — never a member of the cache
object. If scratch is instance-held, the honest class name is "single-pass render context," not
"cache."

### The design fork the sketch exposes: do you need reentrancy at all?

Reentrancy is a *requirement* only under threaded rendering, nested/recursive renders, or a voice
rendering while another is mid-render. **A single-threaded callback that renders voices one at a
time into an accumulator has none of those** — each note's render fully completes before the next,
so a *single shared scratch buffer reused across all voices* is correct and fastest. In that world
the borrow/recycle pool solves a concurrency problem that does not exist, and its `id()`-dict + set
churn is pure overhead — which is why the pool needs its own benchmark before adoption; the likely
answer is "not worth it unless you go multi-threaded."

So the honest fork, which belongs in the source as an explicit decision:

- reentrancy as a **performance/concurrency need** — probably *absent* in a mono-threaded callback; or
- reentrancy as an **FP-reasoning ideal** — no shared mutable state so the render is referentially
  transparent and trivially testable, even with nothing concurrent forcing it.

Different justifications, different designs. This choice decides whether the pool exists.

### Cost of purity scales with scratch *size* — decide per-buffer

`Render_M_N_NpCache._n_buf` is `(2M,)` — tiny. Making it a local (restoring reentrancy for free)
costs one small allocation per call, essentially nothing. The `(2M, N)` cache is the big thing and
is never reallocated. So "pure vs pooled" is not a global decision: the coefficient vector wants to
be local/pure (cheap, and it buys reentrancy), while a big row/accumulator scratch is where pooling
might earn its keep *if* threaded. The sketch treats both as instance members uniformly; splitting
that decision by size is the pragmatic move.

### What the sketch gets right

- `render_1_1` as the scalar, allocation-free, genuinely-pure **definitional oracle** — everything
  else is an optimization that must equal it sampled at each `n`. Correct anchor for both
  "human-understandable" and differential testing.
- The file laid out as an explicit purity/performance **ladder**, which is exactly the "preserve
  non-production variants as explanatory artifacts" goal from the README. Getting the *names* on
  that ladder right (cache / state / scratch / out; "reentrant" reserved for versions that earn it)
  is genuine design work, and it is mostly one rename away from correct.

---

## 4. Suggested next steps

Ordered roughly by leverage and dependency. Not a commitment — a menu.

### Refactoring (do first; unblocks the rest)

1. **Split definition from cursor.** Extract an immutable note-definition/cache type from the
   mutable playback cursor. This is the single highest-leverage move; the crossfade design and the
   reentrancy story both fall out of it.
2. **Pass 1 of the honesty refactor — evict ambient effects only.** Move the phase RNG out of
   `attack`; have events carry phases (or a seed). Turn the trend state machine, the auto-silence
   predicate, and the envelope into pure functions. Leave object shapes roughly intact. Adopt and
   document the **"pure except declared `out=`"** house rule.
3. **Rename to the four-lifetime vocabulary** in the ver5 sketch (cache / state / scratch / out),
   and either move instance-held scratch out of the cache classes or rename those classes to
   "render context" to stop implying reentrancy they don't have.

### Exploration / benchmarking (answers open questions)

4. **Decide the threading model explicitly**, and record it in source. If single-threaded (likely),
   retire the borrow pool in favor of one shared scratch; if multi-threaded is a real goal, keep
   per-pass scratch and benchmark the pool.
5. **Benchmark the borrow/recycle pool** vs. a single shared scratch vs. small local allocation —
   specifically the `id()`-dict + set overhead per render at 88-voice scale. Settle whether the
   pool ever pays.
6. **Benchmark the SoA vectorized advance** (`next_phi`/`next_amp` as vectors, one vectorized
   transition) against the current per-`PartialState` Python loop. Expected to be both faster and
   more honest — confirm.
7. **Micro-benchmark `Render_M_N_NpCache` (`matmul`) vs. `Render_1_N_NpCache` summed over
   partials.** This is the per-note GEMV-vs-loop question at the ver5 abstraction; cross-check
   against `piano_ver4`'s validated 88-notes-at-~14%-of-deadline result.

### Improvement (testing rigor without heavyweight machinery)

8. **Build the differential-test harness**: same-phase cross-impl diff (Tier 1) driven by copying
   `next_phi`, plus the analytic `_reference_mix` (Tier 2). This is the real oracle.
9. **Add the known-frequency single-bin projection + Parseval energy scalar (Tier 3)** as a small,
   trust-nobody spectral spot-check — ~5 lines, no FFT framework.
10. **Property-based tests with `hypothesis`**: generate phase arrays, feed both implementations,
    assert equivalence. Naturally reuses the same phase-injection mechanism.

### Integration (later, once the core is honest)

11. **Model the retrigger/crossfade as two cursors over one shared definition**, with
    `RETRIG_YIELD` driving the yielding voice to silence at `-ramp_rise` while the new voice rises.
    Validate equal-power with the energy scalar from step 9.
12. **Establish the record/replay regression fixture**: capture an (event stream + phase draws)
    sequence and snapshot the rendered audio for sample-exact regression testing — including the
    stateful retrigger path.
13. **Live-audio integration (sounddevice)** — deferred until the pure core and its test harness are
    settled; correctness and timing harnesses come first.

---

*Prepared as a design discussion. No source files were modified in this session.*

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
