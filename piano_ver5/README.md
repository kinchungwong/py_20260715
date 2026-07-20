# `piano_ver5/README.md`

## Commit logs

---

### Commit (time 20260720_0050): piano_ver5: refactored and near production, with issues.

Issues:

RenderNpMCached actually need a (2M,) vector, not (M), thus SpareBase upfront design was wrong.

Since spare pool is an optimization technique, decisions are to be made based on their runtime benchmarks. Note that benchmarks don't just measure perf_counter_ns(), it must also measure long term memory churn.

---
