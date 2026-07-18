import math
import numpy as np
from typing import Final
from model_params import PianoModelParams
from piano_note import PianoNote
from piano_partial import PianoPartial

class RenderState:
    """Tracks the render state of a piano note, including all partials,
    damped sine, and linear ramp.

    Constants:
        - model_args: PianoModelParams
        - note: PianoNote, the note_id and velocity.
        - partials: list[PianoPartial], pre-determined by the piano model and note.

    Note that due to vanishing partials, the partial_id field do not line up with
    array index. For clarity, we use the term "valid partials" when indexing into
    any arrays related to the partials.

    Abstract model. 

    The (live) render state consist of:
        - Phase and decay amplitude for each partial, for the next sample to be rendered.
        - Linear ramp state and level, for the next sample to be rendered.

    The phase and decay amplitude of the damped sine model is updated by the
    update_damped_sine() method, which is called by the render() method.

    Do not confuse decay amplitude with the linear ramp level. While both determine
    output amplitude, the decay amplitude is a property of the damped sine model,
    while level is a property of the linear ramp model.

    During live rendering, the linear ramp state (Attack, Sustain, Release) is either
    carried out from the previous render, or change in response to a new note event.
    The linear ramp state determines the linear slope of the envelope.

    Meanwhile, the linear ramp carries over the level from its previous render.
    """
    model_args: Final[PianoModelParams]
    note: Final[PianoNote]
    partials: Final[list[PianoPartial]]
    num_valid_partials: int

    def __init__(self, model_args: PianoModelParams, note: PianoNote, partials: list[PianoPartial]):
        self.model_args = model_args.clone()
        self.note = note
        self.partials = partials
        self.num_valid_partials = num_valid_partials = len(partials)
        self.p_amps = [1.0 for _ in range(num_valid_partials)]
        self.p_phis = [0.0 for _ in range(num_valid_partials)]

    def render_to(self, out: np.ndarray) -> int:
        if out.ndim != 1:
            raise ValueError(f"out must be a 1D array, got shape {out.shape}")
        nsamps = out.shape[0]

        return nsamps

    def update_damped_sine(self, nsamps: int) -> None:
        if nsamps < 0:
            raise ValueError(f"nsamps must be non-negative, got {nsamps}")
        # for idx, partial in enumerate(self.partials):
        #     freq = partial.partial_freq
        #     tau = partial.partial_tau
        #     amp = partial.partial_amp
        #     omega = 2.0 * math.pi * freq / self.model_args.sample_rate
        #     log_decay = -1.0 / (tau * self.model_args.sample_rate)
        #     phi = self.p_phis[idx]
        #     phi += omega * nsamps
        #     phi = phi % (2.0 * math.pi)
        #     self.p_phis[idx] = phi


    def update_linear_ramp(self, nsamps: int) -> None:
        pass
