"""Frame-wise PET measurement noise; separate from CM's latent diffusion noise.

New protocols explicitly select moment-matched log-normal noise and acquisition
durations. Unmarked saved protocols retain their historical Gaussian behavior.
"""
import json
import numpy as np


def noise_settings(cfg):
    """Resolve a protocol without migrating or relabelling historical settings."""
    model = cfg.get('noise_model', 'gaussian')
    if model not in ('lognormal', 'gaussian'):
        raise ValueError('noise_model must be lognormal or gaussian')
    legacy_timing = 'sample_spacing' if cfg.get('simulation_mode', 'rk4') == 'hpc_euler' else 'frame_duration'
    timing = cfg.get('noise_timing', 'frame_duration' if model == 'lognormal' else legacy_timing)
    if timing not in ('frame_duration', 'sample_spacing'):
        raise ValueError('noise_timing must be frame_duration or sample_spacing')
    if model == 'lognormal' and timing != 'frame_duration':
        raise ValueError('Log-normal noise requires actual frame durations (noise_timing=frame_duration)')
    floor = float(cfg.get('noise_epsilon', 1e-8))
    if not np.isfinite(floor) or floor <= 0:
        raise ValueError('noise_epsilon must be finite and positive')
    return dict(noise_model=model, noise_timing=timing, noise_epsilon=floor)


def noise_metadata(cfg):
    return {**noise_settings(cfg), 'half_life_min': float(cfg['half_life_min']),
            'noise_prior': dict(cfg['noise_prior']), 'independent_frames': True,
            'mean_preserving': True, 'aif_noise': False}


def read_noise_metadata(data):
    """Read optional NPZ provenance; absence means unknown, not log-normal."""
    if 'measurement_noise_json' not in data:
        return None
    try:
        recorded = json.loads(str(data['measurement_noise_json']))
        if not isinstance(recorded, dict) or not all(k in recorded for k in ('noise_model', 'noise_timing', 'noise_epsilon', 'half_life_min', 'noise_prior')):
            raise ValueError('Missing noise metadata fields')
        noise_settings(recorded)
        half = float(recorded['half_life_min'])
        prior = recorded['noise_prior']
        bounds = np.asarray([prior['min'], prior['max']], dtype=float)
        if not np.isfinite(half) or half <= 0 or prior['type'] != 'uniform' or not np.isfinite(bounds).all() or not 0 <= bounds[0] < bounds[1]:
            raise ValueError('Invalid noise half-life or prior')
    except (ValueError, TypeError, KeyError) as error:
        raise ValueError('Invalid measurement_noise_json: ' + str(error)) from error
    return recorded


def apply_measurement_noise(clean, amplitude, times, half_life, standard_normals, *,
                            frame_durations=None, model='lognormal',
                            timing='frame_duration', signal_floor=1e-8):
    """Corrupt [N,T] TACs using supplied N(0,1) draws, leaving RNG policy to callers.

    Log-normal: mean=max(clean,signal_floor), variance=amplitude**2 * mean *
    exp(log(2)*times/half_life) / frame_duration. Compute eta**2 in log space
    to avoid squaring a potentially huge coefficient of variation. Samples are
    never clipped to signal_floor. Only floating-point underflow is floored to
    the smallest positive float64; overflow raises rather than capping tails.
    """
    clean = np.asarray(clean, dtype=np.float64)
    z = np.asarray(standard_normals, dtype=np.float64)
    t = np.asarray(times, dtype=np.float64)
    amplitude = np.asarray(amplitude, dtype=np.float64)
    settings = noise_settings(dict(noise_model=model, noise_timing=timing, noise_epsilon=signal_floor))
    if clean.ndim != 2 or not len(clean) or t.shape != (clean.shape[1],) or len(t) < 2 or z.shape != clean.shape:
        raise ValueError('Noise requires clean and standard_normals [N,T], and times [T]')
    if not np.isfinite(clean).all() or np.any(clean < -1e-8) or not np.isfinite(z).all():
        raise ValueError('Noise requires finite nonnegative clean TACs and finite standard normal draws')
    if not np.isfinite(t).all() or t[0] <= 0 or np.any(np.diff(t) <= 0):
        raise ValueError('Noise times must be finite, positive and increasing')
    if amplitude.shape not in ((), (len(clean), 1)) or not np.isfinite(amplitude).all() or np.any(amplitude < 0):
        raise ValueError('Noise amplitude must be finite nonnegative scalar or [N,1]')
    half_life = float(half_life)
    if not np.isfinite(half_life) or half_life <= 0:
        raise ValueError('Noise half-life must be finite and positive')
    delta = np.r_[t[0], np.diff(t)] if timing == 'sample_spacing' else np.asarray(frame_durations, dtype=np.float64)
    if delta.shape != t.shape or not np.isfinite(delta).all() or np.any(delta <= 0):
        raise ValueError('Noise requires one finite positive actual frame duration per frame')
    mean = np.maximum(clean, settings['noise_epsilon'])
    with np.errstate(over='ignore', invalid='ignore'):
        decay_time = (np.log(2.) / half_life) * t
        if model == 'gaussian':
            # Retain the operation order and unfloored centre of the HPC code.
            sigma = amplitude * np.sqrt(mean * np.exp(-decay_time) / delta) * np.exp(decay_time)
            noisy = clean + sigma * z
        else:
            with np.errstate(divide='ignore'):
                log_cv_squared = 2 * np.log(amplitude) + decay_time - np.log(delta) - np.log(mean)
            eta_squared = np.logaddexp(0., log_cv_squared)
            log_noisy = np.log(mean) + np.sqrt(eta_squared) * z - .5 * eta_squared
            if not np.isfinite(log_noisy).all() or np.any(log_noisy > np.log(np.finfo(np.float64).max)):
                raise ValueError('Log-normal noise exceeds float64 range; check activity units, noise level and timing')
            smallest = np.nextafter(np.float64(0), np.float64(1))
            noisy = np.exp(np.maximum(log_noisy, np.log(smallest)))
            # Exactly recover the noiseless signal (apart from the specified floor).
            noisy = np.where(amplitude == 0, mean, noisy)
    if not np.isfinite(noisy).all():
        raise ValueError('Measurement noise became non-finite; check units, noise level and timing')
    return noisy
