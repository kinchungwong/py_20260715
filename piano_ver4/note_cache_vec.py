"""NumPy vectorized versions of PartialCache and NoteCache.

To minimize the overhead in both Python and NumPy, the classes in this module
are vectorized to the fullest extent possible.

For each note, the components (expcos, expsin) are precomputed and cached
in one big matrix, with all expcos as rows toward the top, and all expsin
as rows toward the bottom.

The parameters for note partials, including the static ones (p_amp, omega,
log_delay) as well as the dynamic ones (next_amp, next_phi) are also stored
as vectors, so that the entire note can be rendered in one go, without any
Python loops. This also eliminates any Python code needed to copy these
values into a NumPy vector.

The only vector that will still be initialized with Python is the linear ramp
slope selection. The Trend state machine will also be coded in Python.

Any temporary row buffers will either be an instance member, or borrowed from
a RowBuffer instance. Note that borrow-recycle also introduces some overhead,
therefore must be benchmarked.
"""

import math
from typing import Any, Callable, Final, Mapping
# from time import perf_counter_ns

import numpy as np

from piano_data_model import Partial, PartialState, PianoCfg, PianoKey, PianoNote, LinearRamp, NoteState, Trend
from piano_model import PianoModel, NotePartials
from piano_ver4.note_cache import RowBuffer


class NoteCacheVec:
    def __init__(self, model: PianoModel, key: PianoKey, nsamps: int, row_cache: RowBuffer | None = None):
        raise NotImplementedError("NoteCacheVec is not yet implemented")
    def render_to(self, note_state: NoteState, out: np.ndarray) -> None:
        raise NotImplementedError("NoteCacheVec is not yet implemented")
