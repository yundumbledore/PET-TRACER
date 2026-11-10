"""Conditional iMF and sCM for standardized kinetic parameter vectors.

The observation vector y (TAC and AIF) is held fixed along each noise path.
The objectives below are the no-guidance, no-teacher variants of iMF and
continuous-time sCM. Only training uses forward-mode differentiation.
"""
import math

import numpy as np
import torch
from torch.func import jvp


def _column(t):
    return t[:, None, None]


def _squared_error(error, weights):
    squared = error.square()
    if weights is not None:
        squared = squared * weights
    return squared.flatten(1).mean(1)


def imf_loss(model, x, y, loss_weights=None, *, noise=None, t=None, r=None, adaptive=True):
    """iMF Algorithm 1 (arXiv:2512.02012): regress to the pairwise velocity.

    The network returns average velocity via model(state, y, t, r). Its
    diagonal prediction u(state,t,t) estimates the marginal velocity used in
    the JVP, rather than reusing the sample-dependent regression target.
    """
    noise = torch.randn_like(x) if noise is None else noise
    if t is None and r is None:
        pairs = (torch.randn((2, len(x)), dtype=x.dtype, device=x.device) - .4).sigmoid()
        t, r = pairs.max(dim=0).values, pairs.min(dim=0).values
        # The paper mixes interval examples with 50% diagonal FM examples.
        r = torch.where(torch.rand_like(t) < .5, t, r)
    elif t is None or r is None:
        raise ValueError('Supply both iMF times t and r, or neither')
    z = (1 - _column(t)) * x + _column(t) * noise

    def field(state, left_time, right_time):
        return model(state, y, right_time, left_time)

    with torch.no_grad():
        velocity = model(z, y, t, t)
    average, derivative = jvp(field, (z, r, t), (velocity, torch.zeros_like(r), torch.ones_like(t)))
    prediction = average + _column(t - r) * derivative.detach()
    error = prediction - (noise - x)
    mse = _squared_error(error, loss_weights)
    if adaptive:
        # MeanFlow's detached loss-adaptive weight, with a robust p=1/2.
        weight = (error.detach().flatten(1).square().sum(1) + 1e-3).rsqrt()
        mse = weight * mse
    return mse.mean()


def scm_loss(model, x, y, loss_weights=None, *, warmup=1.0, noise=None, t=None, adaptive=True):
    """sCM Algorithm 1, sCT branch (arXiv:2410.11081), without a teacher.

    The network returns raw TrigFlow backbone F via model(state, y, t);
    scm_log_weight(t) supplies the learned time-dependent log weight.
    Nonadaptive evaluation reports the squared unnormalized consistency defect
    instead of a bounded normalized residual or the learned weighted loss.
    """
    sigma_data = model.sigma_data
    if t is None:
        log_sigma = 1.6 * torch.randn(len(x), dtype=x.dtype, device=x.device) - 1.
        t = torch.atan(log_sigma.exp() / sigma_data)
    noise = torch.randn_like(x) if noise is None else noise
    z = sigma_data * noise
    sin_t, cos_t = _column(t.sin()), _column(t.cos())
    x_t = cos_t * x + sin_t * z
    # Analytic direction for the paired TrigFlow path (sCT, not sCD).
    dx_dt = cos_t * z - sin_t * x

    def field(state, time):
        return model(state / sigma_data, y, time)

    raw, derivative = jvp(field, (x_t, t), (dx_dt, torch.ones_like(t)))
    with torch.no_grad():
        # cos(t) times the endpoint derivative, rearranged for stability.
        g = (cos_t.square() * (dx_dt - sigma_data * raw)
             - warmup * cos_t * sin_t * (x_t + sigma_data * derivative))
        if not adaptive:
            return _squared_error(g, loss_weights).mean()
        norm = torch.linalg.vector_norm(g.flatten(1), dim=1)
        normalized = g / _column(norm + .1)
        target = raw + normalized
    log_weight = model.scm_log_weight(t).reshape(len(x))
    return (log_weight.exp() * _squared_error(raw - target, loss_weights) - log_weight).mean()


def sampling_schedule(method, steps, *, sigma_data=1.0):
    """Times of network evaluations, in descending order (zero is the endpoint)."""
    if steps < 1:
        raise ValueError('Sampling steps must be positive')
    if method in ('cm', 'imf'):
        return np.linspace(1., 0., num=steps, endpoint=False)
    if method != 'scm':
        raise ValueError(f'Unknown posterior method: {method}')
    if not math.isfinite(sigma_data) or sigma_data <= 0:
        raise ValueError('sCM sigma_data must be finite and positive')
    # At pi/2 the TrigFlow state is exactly Gaussian; unlike the paper's
    # finite-sigma benchmark we need not match the starting time of a teacher.
    if steps == 2:
        return np.array([math.pi / 2, 1.1])
    return np.linspace(math.pi / 2, 0., num=steps, endpoint=False)


def imf_sample(model, y, noise, times):
    """Compose average-velocity jumps; noise() returns a standard normal draw."""
    state = noise()
    for index, t in enumerate(times):
        r = times[index + 1] if index + 1 < len(times) else 0.
        current = state.new_full((len(state),), float(t))
        previous = state.new_full((len(state),), float(r))
        state = state - float(t - r) * model(state, y, current, previous)
    return state


def scm_sample(model, y, noise, times, *, sigma_data=None):
    """Predict the endpoint, then optionally re-noise at intermediate angles."""
    sigma_data = model.sigma_data if sigma_data is None else sigma_data
    state = sigma_data * noise()
    for index, t in enumerate(times):
        sin_t, cos_t = math.sin(t), math.cos(t)
        if index:
            state = cos_t * state + sigma_data * sin_t * noise()
        angle = state.new_full((len(state),), float(t))
        state = cos_t * state - sigma_data * sin_t * model(state / sigma_data, y, angle)
    return state
