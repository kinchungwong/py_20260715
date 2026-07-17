import math
from typing import overload
import numpy as np

from damped_sine_args import DampedSineArgsCt, DampedSineArgsDt
from damped_sine_args_cvt import to_dt_args, next_dt_args, next_ct_args

_DampedSineArgs = DampedSineArgsDt | DampedSineArgsCt


class DampedSine:
    """Class for rendering damped sine waves in both continuous and discrete time.

    A damped sine wave is a sine wave whose amplitude decays exponentially over time.

    The continuous-time representation is given by:
        out(t) = sin(2 * pi * freq * t + phi0) * amp0 * exp(-t / tau)
    The discrete-time representation is given by:
        out[n] = sin(phi0 + omega * n) * amp0 * exp(log_decay * n)
    where:
        omega = 2 * pi * freq / samp_rate
        log_decay = -1 / (tau * samp_rate)

    Refer to the DampedSineArgsCt and DampedSineArgsDt dataclasses for details on
    their parameters, such as valid ranges and constraints.
    """

    @classmethod
    def render_dt(cls, args: DampedSineArgsDt, out: np.ndarray) -> DampedSineArgsDt:
        """Render a decaying sine wave in discrete time.
        """
        nsamps = out.shape[0]
        assert out.shape == (nsamps,)
        phi0, omega, amp0, log_decay = args.phi0, args.omega, args.amp0, args.log_decay
        iota = np.arange(nsamps)
        out[:] = (
            amp0
            * np.sin(phi0 + omega * iota)
            * np.exp(iota * log_decay)
        )
        return next_dt_args(args, nsamps)

    @classmethod
    def render_ct(cls, args: DampedSineArgsCt, out: np.ndarray) -> DampedSineArgsCt:
        """Render a decaying sine wave in continuous time.
        """
        cls.render_dt(to_dt_args(args), out)
        return next_ct_args(args, nsamps=out.shape[0])

    @overload
    @classmethod
    def render_ez(cls, args: DampedSineArgsDt, nsamps: int) -> tuple[np.ndarray, DampedSineArgsDt]: ...

    @overload
    @classmethod
    def render_ez(cls, args: DampedSineArgsCt, nsamps: int) -> tuple[np.ndarray, DampedSineArgsCt]: ...

    @classmethod
    def render_ez(cls, args: _DampedSineArgs, nsamps: int) -> tuple[np.ndarray, _DampedSineArgs]:
        """Convenience render function for decaying sine wave.

        This function allocates the output array and returns it along with the next args.
        This function accepts either continuous-time or discrete-time args, and returns
        the corresponding next args of the same kind.
        For simplicity, the output array type is hard-coded as np.float64.
        """
        out = np.empty(nsamps, dtype=np.float64)
        if isinstance(args, DampedSineArgsCt):
            cls.render_ct(args, out)
            return out, next_ct_args(args, nsamps)
        elif isinstance(args, DampedSineArgsDt):
            cls.render_dt(args, out)
            return out, next_dt_args(args, nsamps)
        raise TypeError(f"args must be DampedSineArgsCt or DampedSineArgsDt, got {type(args).__name__}")


if __name__ == "__main__":
    # Example usage
    freq = 440.0  # Frequency in Hz
    samp_rate = 44100.0  # Sample rate in Hz
    tau = 0.01  # Time constant in seconds
    omega = 2.0 * math.pi * freq / samp_rate
    phi0 = 0.0
    amp0 = 1.0
    log_decay = -1.0 / (tau * samp_rate)
    args_ct = DampedSineArgsCt(samp_rate=samp_rate, phi0=phi0, freq=freq, amp0=amp0, tau=tau)
    print(args_ct)
    args_dt = DampedSineArgsDt(phi0=phi0, omega=omega, amp0=amp0, log_decay=log_decay)
    print(args_dt)
    renderer = DampedSine()
    nsamps = round(2 * samp_rate / freq)  # approximately 2 cycles of the sine wave
    out_ct, next_args_ct = renderer.render_ez(args_ct, nsamps=nsamps)
    out_dt, next_args_dt = renderer.render_ez(args_dt, nsamps=nsamps)
    print(out_ct[:40])
    print(out_ct[::8])
    print(next_args_ct)
    print(out_dt[:40])
    print(out_dt[::8])
    print(next_args_dt)
