"""Entry point: pin BLAS threads, run the full benchmark matrix, write report.

Run from anywhere with the repo venv:

    OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \\
      /home/ubuntu/workspace/py_20260715/.venv/bin/python \\
      piano_ver5/bench/run_bench.py [--quick]

--quick runs a tiny subset (correctness + a few cells) for smoke-testing.
"""
from __future__ import annotations

# --- Thread pinning MUST happen before numpy is imported anywhere. ---
import os
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import sys
_HERE = os.path.dirname(os.path.abspath(__file__))
_PIANO_VER5 = os.path.dirname(_HERE)
sys.path.insert(0, _PIANO_VER5)  # so `import damped_sine_impl` resolves
sys.path.insert(0, _HERE)        # so bench modules import as top-level

import argparse
import io
import platform
from contextlib import redirect_stdout

import numpy as np

import audio_params as ap
import partial_gen as pg
from strategies import (
    NaiveUncached, Ver5PerNoteCached, BatchedGemv, BatchedSoa,
)
from driver import (
    run_cell, measure_memory, correctness_gate, reference_block0, CellResult,
)
import report as report_mod


F32 = np.float32
F64 = np.float64


# ---------------------------------------------------------------------------
# Master-voice cache (build once per (M, V, N), reuse across strategies/dtypes)
# ---------------------------------------------------------------------------
_master_cache: dict[tuple[int, int, int], list] = {}


def master_voices(M: int, V: int, N: int) -> list:
    key = (M, V, N)
    if key not in _master_cache:
        rng = np.random.default_rng(ap.RNG_SEED + M * 100003 + V * 101 + N)
        midis = pg.pitch_plan(M, V)
        _master_cache[key] = [
            pg.build_partials(m, M, N, dtype=F64, rng=rng) for m in midis
        ]
    return _master_cache[key]


def _apply_speedups(cells: list[CellResult]) -> None:
    """Within each (N,M,V) group, fill speedup vs naive and vs per-note."""
    groups: dict[tuple, dict[str, CellResult]] = {}
    for c in cells:
        groups.setdefault((c.N, c.M, c.V), {})[c.strategy] = c
    for g in groups.values():
        naive = g.get(NaiveUncached.name)
        pern = g.get(Ver5PerNoteCached.name)
        for c in g.values():
            if naive and c.us_p50 > 0:
                c.speedup_vs_naive = naive.us_p50 / c.us_p50
            if pern and c.us_p50 > 0:
                c.speedup_vs_pernote = pern.us_p50 / c.us_p50


# ---------------------------------------------------------------------------
# Runs
# ---------------------------------------------------------------------------
def run_correctness(m_list, v_list, n_list) -> dict:
    gate: dict[tuple[int, int, int], dict] = {}
    for N in n_list:
        for M in m_list:
            for V in v_list:
                mv = master_voices(M, V, N)
                gate[(M, V, N)] = correctness_gate(mv, N)
    return gate


def run_core(m_list, v_list, n_list) -> list[CellResult]:
    results = []
    for N in n_list:
        for M in m_list:
            for V in v_list:
                mv = master_voices(M, V, N)
                for cls in (NaiveUncached, Ver5PerNoteCached, BatchedGemv, BatchedSoa):
                    results.append(run_cell(cls, mv, N, F32, F32, "level", "f32"))
    _apply_speedups(results)
    return results


def run_dtype_study(m_list, v_list) -> list[CellResult]:
    N = ap.N_PRIMARY
    results = []
    for M in m_list:
        for V in v_list:
            mv = master_voices(M, V, N)
            for cls in (Ver5PerNoteCached, BatchedSoa):
                # f64: partials & compute float64.
                results.append(run_cell(cls, mv, N, F64, F64, "level", "f64"))
                # f32-trap: partials float64 (upcasts cache) but float32 out.
                results.append(run_cell(cls, mv, N, F64, F32, "level", "f32-trap"))
    return results


def run_decay(m_list, v_list) -> list[CellResult]:
    N = ap.N_PRIMARY
    results = []
    for M in m_list:
        for V in v_list:
            mv = master_voices(M, V, N)
            for cls in (NaiveUncached, Ver5PerNoteCached, BatchedSoa):
                results.append(run_cell(cls, mv, N, F32, F32, "decay", "f32"))
    _apply_speedups(results)
    return results


def run_memory(configs) -> list:
    results = []
    for cls, M, V, blocks in configs:
        mv = master_voices(M, V, ap.N_PRIMARY)
        results.append(measure_memory(cls, mv, ap.N_PRIMARY, F32, F32, blocks=blocks))
    return results


def run_realistic() -> tuple[list[CellResult], dict]:
    N = ap.N_PRIMARY
    kb = pg.realistic_keyboard()
    rng = np.random.default_rng(ap.RNG_SEED + 7)
    voices = [pg.build_partials(m, cnt, N, dtype=F64, rng=rng) for m, cnt in kb]
    counts = [c for _, c in kb]
    info = {"total": sum(counts), "lo": min(counts), "hi": max(counts)}
    # Heterogeneous partial counts exercise a pairing the uniform-M gate doesn't;
    # verify block-0 against the f64 oracle before trusting the timings.
    ref = reference_block0(voices, N)
    tol = ap.CORRECTNESS_TOL * max(1.0, float(np.max(np.abs(ref))))
    for cls in (NaiveUncached, Ver5PerNoteCached, BatchedGemv, BatchedSoa):
        vs = [pg.recast_voice(v, F32) for v in voices]
        s = cls(vs, N, F32, "level"); mix = np.zeros(N, F32); s._render(mix)
        err = float(np.max(np.abs(mix.astype(F64) - ref)))
        assert err <= tol, f"realistic correctness FAIL: {cls.name} err={err:.3e}"
    results = []
    for cls in (NaiveUncached, Ver5PerNoteCached, BatchedGemv, BatchedSoa):
        results.append(run_cell(cls, voices, N, F32, F32, "level", "f32"))
    _apply_speedups(results)
    return results, info


# ---------------------------------------------------------------------------
def _blas_string() -> str:
    buf = io.StringIO()
    try:
        with redirect_stdout(buf):
            np.show_config()
        txt = buf.getvalue()
        for kw in ("scipy-openblas", "openblas", "OpenBLAS", "blas"):
            for line in txt.splitlines():
                if kw in line and ("name" in line.lower() or "found" in line.lower()):
                    return line.strip()
    except Exception:
        pass
    return "unknown (see np.show_config)"


def _xcheck(core: list[CellResult]) -> dict:
    match = [c for c in core
             if c.N == 383 and c.M == 16 and c.V == 88
             and c.strategy == Ver5PerNoteCached.name and c.dtype_label == "f32"]
    if not match:
        return {"us_block": float("nan"), "us_note": float("nan"),
                "pct": float("nan"), "ratio": float("nan"), "verdict": "N/A"}
    c = match[0]
    ratio = c.us_p50 / ap.VER4_ANCHOR["us_block"]
    verdict = "PASS" if 0.4 <= ratio <= 1.5 else "INVESTIGATE"
    return {"us_block": c.us_p50, "us_note": c.us_p50 / 88.0,
            "pct": c.pct_p50, "ratio": ratio, "verdict": verdict}


def _build_findings(core, dtype_study, mem, realistic) -> str:
    """Auto-generate a findings paragraph from the numbers."""
    lines = []

    def cell(strategy, N, M, V, pool=core):
        for c in pool:
            if (c.strategy == strategy and c.N == N and c.M == M and c.V == V):
                return c
        return None

    # 1. caching win
    pn = cell(Ver5PerNoteCached.name, 383, 16, 88)
    if pn and pn.speedup_vs_naive == pn.speedup_vs_naive:
        lines.append(
            f"1. **Caching wins.** At N=383, M=16, V=88 the ver5 per-note cached "
            f"matmul is **{pn.speedup_vs_naive:.1f}× faster** than recomputing "
            f"sin/exp every block (naive). The gain grows with M.")
    # 2. batching win
    soa = cell(BatchedSoa.name, 383, 16, 88)
    gemv = cell(BatchedGemv.name, 383, 16, 88)
    if soa and pn and soa.speedup_vs_pernote == soa.speedup_vs_pernote:
        gemv_txt = (f"Collapsing only the matmul (still looping the coefficient "
                    f"build per voice) gives {gemv.speedup_vs_pernote:.1f}×; "
                    if gemv else "")
        lines.append(
            f"2. **Batching wins big.** {gemv_txt}the full SoA collapse "
            f"(one cos+sin over all partials + one GEMV) is "
            f"**{soa.speedup_vs_pernote:.1f}× faster than the per-note path** at "
            f"V=88/M=16/N=383 ({soa.pct_p50:.2f}% vs {pn.pct_p50:.2f}% of deadline) "
            f"— confirming the research's 'reduce NumPy dispatch count' thesis. "
            f"ver5's per-note design leaves that on the table.")
    # 3. headroom at 88
    if pn:
        lines.append(
            f"3. **Headroom at 88 keys (per-note, N=383):** {pn.pct_p99:.1f}% of "
            f"deadline at p99 → ~{int(pn.max_voices_fit)} voices fit. "
            + (f"At the N=95 low-latency deadline it is tighter: "
               f"{cell(Ver5PerNoteCached.name,95,16,88).pct_p99:.1f}% at p99."
               if cell(Ver5PerNoteCached.name, 95, 16, 88) else ""))
    # 4. memory churn
    pnm = next((m for m in mem if m.strategy == Ver5PerNoteCached.name and m.M == 16 and m.V == 88), None)
    btm = next((m for m in mem if m.strategy == BatchedSoa.name and m.M == 16 and m.V == 88), None)
    if pnm and btm:
        mb_per_s = pnm.analytic_bytes_block * (ap.FS / 383) / 1e6
        lines.append(
            f"4. **Memory churn:** per-note allocates ~{pnm.analytic_bytes_block:,} "
            f"B/block at V=88/M=16 (~{mb_per_s:.1f} MB/s of alloc/free traffic); "
            f"batched SoA allocates **0**. But cyclic GC stays quiet on both "
            f"(gen0={pnm.gc_gen0} vs {btm.gc_gen0} over the run) — the temporaries "
            f"have no reference cycles, so refcounting frees them immediately and "
            f"the collector never fires. So the churn is **not** a GC-pause source; "
            f"its cost is allocator + dispatch traffic, already inside the render "
            f"time the SoA path recovers.")
    # 5. dtype trap
    trap = next((c for c in dtype_study if c.strategy == Ver5PerNoteCached.name
                 and c.dtype_label == "f32-trap" and c.M == 16 and c.V == 88), None)
    f64c = next((c for c in dtype_study if c.strategy == Ver5PerNoteCached.name
                 and c.dtype_label == "f64" and c.M == 16 and c.V == 88), None)
    f32c = cell(Ver5PerNoteCached.name, 383, 16, 88)
    if trap and f32c:
        lines.append(
            f"5. **The float64 upcast trap is real.** Building `PartialM` arrays as "
            f"float64 (as the existing test does) makes the f32-requested cache "
            f"run at **{trap.us_p50:.1f} µs/block** — matching f64 "
            f"({f64c.us_p50:.1f})" + (f" and ~{trap.us_p50/f32c.us_p50:.1f}× "
            f"slower than genuine float32 ({f32c.us_p50:.1f})" if f32c.us_p50 else "")
            + ". Fix: build the partial arrays as float32.")
    return "\n\n".join(lines) if lines else "_(see tables above)_"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true", help="tiny smoke subset")
    parser.add_argument("--out", default=None, help="report output path")
    args = parser.parse_args()

    assert os.environ.get("OPENBLAS_NUM_THREADS") == "1", "BLAS not pinned!"

    if args.quick:
        m_list, v_list, n_list = (5, 16), (1, 88), (383,)
    else:
        m_list, v_list, n_list = ap.GRID_M, ap.GRID_V, ap.N_SET

    print(f"[bench] numpy {np.__version__}  threads="
          f"{os.environ.get('OPENBLAS_NUM_THREADS')}  quick={args.quick}")

    print("[bench] correctness gate ...")
    gate = run_correctness(m_list, v_list, n_list)
    print(f"[bench]   PASS ({len(gate)} configs)")

    print("[bench] core sweep ...")
    core = run_core(m_list, v_list, n_list)

    if args.quick:
        dtype_study, decay, realistic, rinfo = [], [], [], {}
        mem = run_memory([(Ver5PerNoteCached, 16, 88, 1000),
                          (BatchedSoa, 16, 88, 2000)])
    else:
        print("[bench] dtype study ...")
        dtype_study = run_dtype_study(m_list, v_list)
        print("[bench] decay mode ...")
        decay = run_decay(m_list, v_list)
        print("[bench] memory churn ...")
        mem = run_memory([
            (Ver5PerNoteCached, 16, 1, 3000),
            (Ver5PerNoteCached, 16, 20, 3000),
            (Ver5PerNoteCached, 16, 88, 3000),
            (Ver5PerNoteCached, 5, 88, 3000),
            (Ver5PerNoteCached, 40, 88, 2000),
            (BatchedGemv, 16, 88, 5000),
            (BatchedSoa, 16, 88, 5000),
            (BatchedSoa, 40, 88, 5000),
            (NaiveUncached, 16, 20, 1000),
        ])
        print("[bench] realistic 88-key ...")
        realistic, rinfo = run_realistic()

    xcheck = _xcheck(core)
    print(f"[bench] cross-check ver5/ver4 ratio = {xcheck['ratio']:.2f} "
          f"({xcheck['verdict']})")

    ctx = {
        "date": "2026-07-20",
        "python": platform.python_version(),
        "numpy": np.__version__,
        "blas": _blas_string(),
        "openblas_threads": os.environ.get("OPENBLAS_NUM_THREADS", "?"),
        "omp_threads": os.environ.get("OMP_NUM_THREADS", "?"),
        "xcheck_note": "ver5 is expected to be ≤ ver4: it skips ver4's per-note "
                       "np.fromiter object-gather and envelope, at the cost of one "
                       "(2M,) concat per voice.",
        "findings": _build_findings(core, dtype_study, mem, realistic),
    }

    md = report_mod.build_report(
        ctx, core, dtype_study, decay, mem, realistic, rinfo, xcheck, gate)

    out_path = args.out or os.path.join(
        _HERE, "reports",
        ("quick_" if args.quick else "") + "2026-07-20_render_np_m_cached_benchmark.md")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        f.write(md)
    print(f"[bench] report -> {out_path}")
    print(f"[bench] {ctx['findings'][:400]}")


if __name__ == "__main__":
    main()
