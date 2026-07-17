# import math
from typing import Any
import numpy as np
from damped_sine import DampedSine, DampedSineRendererBase

class DampedSineUncached(DampedSineRendererBase):
    """Uncached damped sine renderer.
    """

    def __init__(self) -> None:
        self._auto_advance = True

    def try_attach(self, state: DampedSine) -> bool:
        """Attaches the renderer to a DampedSine state object.

        Since this renderer is compatible to all DampedSine states, it always
        succeeds and returns True.
        """
        assert isinstance(state, DampedSine)
        state.attach_renderer(self, None)
        return True

    def is_auto_advance(self) -> bool:
        """Checks whether each call to render() or render_to() automatically
        advances the internal state of the DampedSine object.
        """
        return self._auto_advance
    
    def set_auto_advance(self, auto_advance: bool) -> None:
        """Sets whether each call to render() or render_to() automatically
        advances the internal state of the DampedSine object.
        """
        self._auto_advance = auto_advance

    def render_to(self, state: "DampedSine", out: np.ndarray) -> None:
        """Renders a damped sine wave to the provided output array.
        """
        assert isinstance(state, DampedSine)
        assert isinstance(out, np.ndarray)
        shape = out.shape
        assert len(shape) == 1
        nsamps = shape[0]
        phi0, omega, amp0, log_decay = state.next_phi, state.omega, state.next_amp, state.log_decay
        iota = np.arange(nsamps)
        out[:] = (
            amp0
            * np.sin(phi0 + omega * iota)
            * np.exp(iota * log_decay)
        )
        if self._auto_advance:
            state.advance(nsamps)

    def render(self, state: "DampedSine", nsamps: int, dtype: Any) -> np.ndarray:
        """Renders a damped sine wave and returns the output array.
        """
        out = np.empty(nsamps, dtype=dtype)
        self.render_to(state, out)
        return out
