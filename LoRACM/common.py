"""Array contracts shared by generation, training, inference and plotting."""
import json
from pathlib import Path
import numpy as np

PARAMETERS = ['K1', 'k2', 'k3', 'k4', 'Vb']

def load_config(path):
    return json.loads(Path(path).read_text())

def validate_dataset(data, x_dim=5, y_dim=70):
    for split in ('train', 'val', 'test'):
        x, y = data[f'x_{split}'], data[f'y_{split}']
        if x.ndim != 2 or x.shape[1] != x_dim + 1:
            raise ValueError(f'x_{split} must be [N, {x_dim + 1}], including noise scale')
        if y.shape != (len(x), y_dim) or len(x) == 0:
            raise ValueError(f'y_{split} must be nonempty [N, {y_dim}]')
        if not np.isfinite(x).all() or not np.isfinite(y).all():
            raise ValueError(f'Non-finite values in {split}')
    t = data['t_meas']
    if t.ndim != 1 or len(t)*2 != y_dim or t[0] <= 0 or not np.all(np.diff(t)>0):
        raise ValueError('Invalid positive, increasing time grid')

def summarize_samples(samples):
    """Summarize joint samples before discarding them; never derive Ki from medians."""
    samples = np.asarray(samples)
    if samples.ndim != 3 or samples.shape[-1] != 5 or samples.shape[1] < 2:
        raise ValueError('Expected [voxels, at least 2 samples, 5 parameters]')
    if not np.isfinite(samples).all():
        raise ValueError('Posterior samples contain non-finite values')
    denom = samples[..., 1] + samples[..., 2]
    valid = np.abs(denom) > 1e-12
    ki = np.full_like(denom, np.nan)
    np.divide(samples[..., 0] * samples[..., 2], denom, out=ki, where=valid)
    result = {f'params_{name}': fun(samples, axis=1) for name, fun in
              [('mean',np.mean),('median',np.median),('std',np.std)]}
    result.update({f'Ki_{name}':fun(ki,axis=1) for name,fun in
                   [('mean',np.nanmean),('median',np.nanmedian),('std',np.nanstd)]})
    result['Ki_valid_fraction'] = valid.mean(axis=1)
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
