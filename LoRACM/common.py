"""Array contracts shared by generation, training, inference and plotting."""
import json
import hashlib
from pathlib import Path
import numpy as np

PARAMETERS = ['K1', 'k2', 'k3', 'k4', 'Vb']

def next_prediction_output(filename):
    """Choose an unused prediction/provenance pair without deleting history."""
    filename=Path(filename)
    if filename.suffix.lower()!='.npz': raise ValueError('Prediction output must end in .npz')
    candidate=filename
    index=2
    while candidate.exists() or candidate.with_suffix('.json').exists():
        candidate=filename.with_name(filename.stem+f'_{index:03d}'+filename.suffix)
        index+=1
    return candidate

def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024), b''): digest.update(block)
    return digest.hexdigest()

def load_config(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))

def validate_dataset(data, x_dim=5, y_dim=70):
    required = ['t_meas'] + [f'{a}_{s}' for a in ('x','y') for s in ('train','val','test')]
    if any(k not in data for k in required):
        raise ValueError('Missing dataset arrays: ' + ', '.join(k for k in required if k not in data))
    for split in ('train', 'val', 'test'):
        x, y = data[f'x_{split}'], data[f'y_{split}']
        if x.ndim != 2 or x.shape[1] != x_dim + 1:
            raise ValueError(f'x_{split} must be [N, {x_dim + 1}], including noise scale')
        if y.shape != (len(x), y_dim) or len(x) == 0:
            raise ValueError(f'y_{split} must be nonempty [N, {y_dim}]')
        if not np.isfinite(x).all() or not np.isfinite(y).all():
            raise ValueError(f'Non-finite values in {split}')
        if np.any(x[:, -1] < 0):
            raise ValueError(f'Negative noise scale in {split}')
    t = data['t_meas']
    if t.ndim != 1 or len(t) < 2 or len(t)*2 != y_dim or not np.isfinite(t).all() or t[0] <= 0 or not np.all(np.diff(t)>0):
        raise ValueError('Invalid positive, increasing time grid')
    for split in ('train', 'val', 'test'):
        if np.any(data['y_'+split][:, len(t):] < 0):
            raise ValueError(f'Input functions must be nonnegative in {split}')
    if 'frame_durations_min' in data:
        duration = data['frame_durations_min']
        if duration.shape != t.shape or not np.isfinite(duration).all() or np.any(duration <= 0):
            raise ValueError('frame_durations_min must be finite positive [T]')

def summarize_samples(samples, parameter_names=None, calculate_ki=True):
    """Summarize joint samples before discarding them; never derive Ki from medians."""
    samples = np.asarray(samples)
    if samples.ndim != 3 or samples.shape[-1] < 1 or samples.shape[1] < 2:
        raise ValueError('Expected [curves, at least 2 samples, parameters]')
    if not np.isfinite(samples).all():
        raise ValueError('Posterior samples contain non-finite values')
    result = {f'params_{name}': fun(samples, axis=1) for name, fun in
              [('mean',np.mean),('median',np.median),('std',np.std)]}
    low, high = np.quantile(samples, [.025, .975], axis=1)
    result.update(params_ci95_low=low, params_ci95_high=high, params_ci95_width=high-low)
    if calculate_ki:
        names = PARAMETERS if parameter_names is None else list(parameter_names)
        if names != PARAMETERS or samples.shape[-1] != 5:
            raise ValueError('Ki is only defined here for the ordered 2TC parameter set')
        denom = samples[..., 1] + samples[..., 2]
        valid = np.abs(denom) > 1e-12
        ki = np.full_like(denom, np.nan)
        np.divide(samples[..., 0]*samples[..., 2], denom, out=ki, where=valid)
        result.update({f'Ki_{name}':fun(ki,axis=1) for name,fun in
                       [('mean',np.nanmean),('median',np.nanmedian),('std',np.nanstd)]})
        result['Ki_valid_fraction'] = valid.mean(axis=1)
        ki_low, ki_high = np.nanquantile(ki, [.025, .975], axis=1)
        result.update(Ki_ci95_low=ki_low, Ki_ci95_high=ki_high, Ki_ci95_width=ki_high-ki_low)
    result['params_negative_fraction'] = (samples < 0).mean(axis=1)
    return result

def reconstruct_map(values, mask, voxel_indices):
    """Indices are C-order flat indices in the exact released two-dimensional mask."""
    mask = np.asarray(mask)
    indices = np.asarray(voxel_indices)
    if mask.ndim != 2 or not np.isin(mask, [0,1]).all():
        raise ValueError('Expected a binary 2D mask')
    if indices.ndim != 1 or not np.issubdtype(indices.dtype, np.integer):
        raise ValueError('voxel_indices must be a one-dimensional integer array')
    if len(indices) != len(values) or len(np.unique(indices)) != len(indices):
        raise ValueError('Voxel indices must be unique and match the number of estimates')
    if np.any(indices < 0) or np.any(indices >= mask.size):
        raise ValueError('Voxel index outside the mask')
    if not np.all(mask.ravel()[indices] == 1):
        raise ValueError('Voxel index outside the foreground')
    result = np.full(mask.size, np.nan)
    result[indices] = values
    return result.reshape(mask.shape)
