from abc import ABC, abstractmethod
import math
from typing import Any, Final
import numpy as np


class DampedSineRendererBase(ABC):
    @abstractmethod
    def try_attach(self, state: "DampedSine") -> bool:
        raise NotImplementedError("try_attach() must be implemented by subclasses")

    @abstractmethod
    def is_auto_advance(self) -> bool:
        raise NotImplementedError("is_auto_advance() must be implemented by subclasses")
    
    @abstractmethod
    def set_auto_advance(self, auto_advance: bool) -> None:
        raise NotImplementedError("set_auto_advance() must be implemented by subclasses")

    @abstractmethod
    def render_to(self, state: "DampedSine", out: np.ndarray) -> None:
        raise NotImplementedError("render_to() must be implemented by subclasses")

    @abstractmethod
    def render(self, state: "DampedSine", nsamps: int, dtype: Any) -> np.ndarray:
        raise NotImplementedError("render() must be implemented by subclasses")


class DampedSine:
    class _Private:
        _renderer: DampedSineRendererBase | None
        _renderer_args: Any | None

        def __init__(self) -> None:
            self._renderer = None
            self._renderer_args = None

    samp_rate: Final[float]
    freq: Final[float]
    tau: Final[float]
    omega: Final[float]
    log_decay: Final[float]
    has_decay: Final[bool]
    next_phi: float
    next_amp: float
    _private: Final[_Private]

    def __init__(
            self,
            samp_rate: float,
            *,
            freq: float | None = None,
            tau: float | None = None,
            omega: float | None = None,
            log_decay: float | None = None,
            next_phi: float = 0.0,
            next_amp: float = 1.0,
    ) -> None:
        if not (samp_rate > 0.0):
            raise ValueError(f"samp_rate must be positive, got {samp_rate}")
        pi = math.pi
        twopi = 2.0 * pi
        nyquist = 0.5 * samp_rate
        if freq is not None and not (0 < freq < nyquist):
            raise ValueError(f"freq must be positive and below Nyquist, got {freq}")
        if tau is not None and not (tau > 0.0):
            raise ValueError(f"tau must be positive or infinite, got {tau}")
        if omega is not None and not (0.0 < omega < pi):
            raise ValueError(f"omega must be between 0.0 and pi (exclusive), got {omega}")
        if log_decay is not None and not (log_decay <= 0.0):
            raise ValueError(f"log_decay must be non-positive, got {log_decay}")

        # Reconcile freq, omega

        if freq is not None and omega is not None:
            freq_expect = omega * samp_rate / twopi
            if not math.isclose(freq, freq_expect, rel_tol=1e-6):
                raise ValueError(f"freq and omega are inconsistent: freq={freq}, omega={omega}, samp_rate={samp_rate}")
        elif freq is not None:
            omega = twopi * freq / samp_rate
        elif omega is not None:
            freq = omega * samp_rate / twopi
        else:
            raise ValueError("Must specify either freq or omega")

        # Reconcile tau, log_decay

        if tau is not None and log_decay is not None:
            log_decay_expect = -1.0 / (tau * samp_rate)
            if not math.isclose(log_decay, log_decay_expect, rel_tol=1e-6):
                raise ValueError(f"tau and log_decay are inconsistent: tau={tau}, log_decay={log_decay}, samp_rate={samp_rate}")
        elif tau is not None:
            log_decay = -1.0 / (tau * samp_rate)
        elif log_decay is not None:
            if log_decay < 0:
                tau = -1.0 / (log_decay * samp_rate)
            else:
                tau = math.inf
        else:
            raise ValueError("Must specify either tau or log_decay")
        
        self.samp_rate = float(samp_rate)
        self.freq = float(freq)
        self.tau = float(tau)
        self.omega = float(omega)
        self.log_decay = float(log_decay)
        self.has_decay = self.log_decay < 0.0
        self.next_phi = float(next_phi)
        self.next_amp = float(next_amp)
        self._private = self._Private()

    def advance(self, nsamps: int) -> None:
        """Advance the internal state by `nsamps` samples.
        """
        self.next_phi = (self.next_phi + self.omega * nsamps) % (2.0 * math.pi)
        self.next_amp *= math.exp(self.log_decay * nsamps)

    def attach_renderer(self, renderer: DampedSineRendererBase, renderer_args: Any | None = None) -> None:
        """Attach a renderer to this DampedSine object.

        Args:
            renderer: DampedSineRendererBase
                The renderer to attach. Must implement the DampedSineRendererBase interface.
        """
        if not isinstance(renderer, DampedSineRendererBase):
            raise TypeError(f"renderer must implement DampedSineRendererBase, got {type(renderer).__name__}")
        self._private._renderer = renderer
        self._private._renderer_args = renderer_args

    def detach_renderer(self) -> None:
        """Detach the renderer from this DampedSine object.
        """
        self._private._renderer = None
        self._private._renderer_args = None

    def has_renderer(self) -> bool:
        renderer = self._private._renderer
        return isinstance(renderer, DampedSineRendererBase)

    def has_renderer_args(self) -> bool:
        return self._private._renderer_args is not None

    def require_renderer(self) -> DampedSineRendererBase:
        renderer = self._private._renderer
        if not isinstance(renderer, DampedSineRendererBase):
            raise TypeError(f"Expects renderer not None and implements DampedSineRendererBase, got {type(renderer).__name__}")
        return renderer

    def render_to(self, out: np.ndarray) -> None:
        """Render a decaying sine wave to the provided output array.

        Args:
            out: np.ndarray
                The output array to render into. Must be 1D.
        """
        self.require_renderer().render_to(self, out)

    def render(self, nsamps: int, dtype: Any = np.float64) -> np.ndarray:
        """Render a decaying sine wave and return the output array.

        Args:
            nsamps: int
                Number of samples to render.
            dtype: np.dtype
                Data type of the output array to allocate.
        Returns:
            np.ndarray
                The rendered output array.
        """
        return self.require_renderer().render(self, nsamps, dtype)

    def get_renderer_args(self) -> Any | None:
        """Get the renderer arguments associated with this DampedSine object.

        Returns:
            Any | None
                The renderer arguments, or None if no renderer is attached.
        """
        return self._private._renderer_args
