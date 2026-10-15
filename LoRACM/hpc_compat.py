"""Portable implementation of the original HPC Euler and noise conventions.

This module intentionally preserves the historical numerical scheme, including
the first Euler step at t[0] and searchsorted (rightward) observations. It does
not replace the general RK4 simulator for new protocols.
"""
import numpy as np


def euler_observations(model, parameters, aifs, times, step=.01):
    times = np.asarray(times, dtype=float)
    grid = np.arange(times[0], times[-1] + 1e-8, step)
    if grid[-1] < times[-1]:
        grid = np.r_[grid, times[-1]]
    delta = np.r_[grid[0], np.diff(grid)]
    indices = np.searchsorted(grid, times)
    dense = np.asarray([np.interp(grid, times, curve) for curve in aifs])
    state = np.zeros((len(parameters), model.STATE_SIZE), dtype=float)
    result = np.empty((len(parameters), len(times)))
    observation = 0
    for i, (time, dt) in enumerate(zip(grid, delta)):
        value = dense[:, i]
        state = state + dt * model.rhs(time, state, parameters, value)
        while observation < len(times) and indices[observation] == i:
            result[:, observation] = model.observe(state, parameters, value)
            observation += 1
    if observation != len(times) or not np.isfinite(result).all():
        raise ValueError('Non-finite or incomplete HPC Euler simulation')
    return result


def noise_standard_deviation(clean, amplitude, times, half_life, durations=None):
    """Original heteroscedastic equation; durations=None selects HPC delta_t."""
    times = np.asarray(times, dtype=float)
    delta = np.r_[times[0], np.diff(times)] if durations is None else np.asarray(durations)
    decay = np.log(2.) / half_life
    return amplitude * np.sqrt(np.maximum(clean, 1e-8) * np.exp(-decay * times) / delta) * np.exp(decay * times)
