# piano_ver4

## Authorship attributions

### piano_data_model.py, piano_model.py

Hand written, with details of piano synthesis model lifted from py_20260707.

This implementation contain some additional tweaks.

Code reviewed performed by Claude Code, with small bug fixes applied.

Type-checked with Pyright.

### piano_render.py

Fully written by Claude Code, Opus 4.8 (high, 1M context).

Write-up (also by Claude Code): 
[[claude/deep_research/2026-07-18-per-note-synthesis-piano-ver4.md]]

## Next steps

The piano_render code written by Claude validated the approach.
To bring it into production, further improvements and refactoring
will be needed.

We intend to keep `piano_ver4` frozen, self-contained, and runnable
in its current form, to allow us to reproduce the findings and
benchmark results for reference in the future.

Any new work will go into `piano_ver5`, seeded with some validated
designs lifted from the current `piano_ver4`.

### Planned re-architecting

- PianoCache, NoteCache, and PartialCache, which hosts the logic
  and pre-rendered NumPy matrices.
    - NoteCache and PartialCache will have partial functionality
      overlap; the intention is that the entire block of partials
      will be pre-rendered as a single matrix for NoteCache.
    - In order to do so, NoteCache will contain its own matrix
      initialization code.
- NoteCacheState, which contains NoteState along with state vectors
  that are used exclusively with the pre-rendered approach.
    - The idea is that NoteCacheState should update the state vectors
      as NumPy vectors, using as few NumPy calls as practical,
      instead of using scalar Python code or allowing control flow to
      weave in and out between Python and NumPy.
- Code that are currently free functions will likely be moved into
  Python classes, primarily to improve code organization.

### Separation of production code, unit tests, and research code

Performance tuning with NumPy code generally requires benchmarking
lots of variations without changing the underlying math. As such,
there is need to preserve implementations that enable benchmarking
of non-production code paths, mainly as a explanatory tool that a
certain choice was not made (in production) because it can be shown
to run slower.

Meanwhile, both production code and unit tests belong to the
production branch, and will have to be brought up to the production
code standards.

## note_cache.py

### Authorship attribution

`note_cache.py` is hand-written, as a vitrified record of how humans
normally write code, in an object-oriented way, even if the code
isn't meant to be performant. The goal is to illustrate the flow of
data in the piano model (now that the caching technique is used),
and to provide an analytical reference so that it can be numerically
compared to other implementations in unit tests.

(Planned) There will also be an `render_uncached.py` (not yet started),
with further simplification.

(Planned) `note_cache_vec.py` will be the hand-written and redesigned
version of the `piano_render.py` contributed by Claude Code, with some
further optimizations. Refer to the source level docstring for details.
