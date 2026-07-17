import math

from damped_sine_args import DampedSineArgsCt, DampedSineArgsDt


def to_dt_args(args: DampedSineArgsCt) -> DampedSineArgsDt:
    """Convert continuous-time args to discrete-time args."""
    samp_rate, phi0, freq, amp0, tau = args.samp_rate, args.phi0, args.freq, args.amp0, args.tau
    omega = 2.0 * math.pi * freq / samp_rate
    log_decay = -1.0 / (tau * samp_rate)
    return DampedSineArgsDt(phi0=phi0, omega=omega, amp0=amp0, log_decay=log_decay)

def to_ct_args(args: DampedSineArgsDt, samp_rate: float) -> DampedSineArgsCt:
    """Convert discrete-time args to continuous-time args."""
    phi0, omega, amp0, log_decay = args.phi0, args.omega, args.amp0, args.log_decay
    freq = omega * samp_rate / (2.0 * math.pi)
    tau = 1.0 / (-log_decay * samp_rate) if log_decay < 0.0 else math.inf
    return DampedSineArgsCt(samp_rate=samp_rate, phi0=phi0, freq=freq, amp0=amp0, tau=tau)

def next_dt_args(args: DampedSineArgsDt, nsamps: int) -> DampedSineArgsDt:
    """Compute the next discrete-time args after rendering `nsamps` samples.
    """
    phi0, omega, amp0, log_decay = args.phi0, args.omega, args.amp0, args.log_decay
    next_phi0 = (phi0 + omega * nsamps) % (2.0 * math.pi)
    next_amp0 = amp0 * math.exp(log_decay * nsamps)
    return DampedSineArgsDt(phi0=next_phi0, omega=omega, amp0=next_amp0, log_decay=log_decay)

def next_ct_args(args: DampedSineArgsCt, nsamps: int) -> DampedSineArgsCt:
    """Compute the next continuous-time args after rendering `nsamps` samples.
    """
    args_dt = to_dt_args(args)
    next_args_dt = next_dt_args(args_dt, nsamps)
    next_args_ct = to_ct_args(next_args_dt, samp_rate=args.samp_rate)
    # Reuse from ct args to eliminate unnecessary floating point drift
    return DampedSineArgsCt(
        samp_rate=args.samp_rate,
        phi0=next_args_ct.phi0,
        freq=args.freq,
        amp0=next_args_ct.amp0,
        tau=args.tau,
    )
