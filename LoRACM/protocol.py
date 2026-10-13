"""User protocol contracts and local kinetic-model plugins (single input)."""
import importlib.util
import json
from pathlib import Path
import numpy as np

FENG_NAMES = ['beta1', 'beta2', 'beta3', 'kappa1', 'kappa2', 'kappa3', 't0']


def load_protocol(path):
    """Resolve bundled model source beside a saved protocol, also after relocation."""
    path = Path(path).resolve()
    cfg = json.loads(path.read_text(encoding='utf-8'))
    if cfg.get('model') == 'custom':
        model_file = Path(cfg['model_file']).expanduser()
        if not model_file.is_absolute() and (path.parent/model_file).is_file():
            cfg['model_file'] = str((path.parent/model_file).resolve())
    return cfg


def validate_times(times, name='time points', allow_zero=False):
    t = np.asarray(times, dtype=float)
    if t.ndim != 1 or len(t) < 2 or not np.isfinite(t).all():
        raise ValueError(f'{name} must have at least two finite values')
    if (t[0] < 0 if allow_zero else t[0] <= 0) or np.any(np.diff(t) <= 0):
        raise ValueError(f'{name} must be increasing and {"nonnegative" if allow_zero else "positive"}')
    return t


def validate_names(names):
    if not isinstance(names, (list, tuple)) or not names or any(not isinstance(n, str) or not n.strip() for n in names):
        raise ValueError('parameter_names must be a nonempty list of names')
    if len(set(names)) != len(names):
        raise ValueError('Parameter names must be unique')
    return list(names)


def validate_protocol(cfg):
    names = validate_names(cfg['parameter_names'])
    t = validate_times(cfg['t_meas'])
    duration = np.asarray(cfg['frame_durations_min'], dtype=float)
    if duration.shape != t.shape or not np.isfinite(duration).all() or np.any(duration <= 0):
        raise ValueError('Supply one finite positive frame duration per measurement')
    starts, ends = t-duration/2, t+duration/2
    if np.any(starts < -1e-8) or np.any(starts[1:] < ends[:-1]-1e-8):
        raise ValueError('Frames must start at/after zero and must not overlap; t_meas are frame midpoints')
    if cfg.get('observation', 'midpoint') not in ('midpoint', 'frame_average'):
        raise ValueError('observation must be midpoint or frame_average')
    half = float(cfg['half_life_min'])
    step = float(cfg.get('integration_step_min', 0.01))
    if not np.isfinite([half, step]).all() or half <= 0 or step <= 0:
        raise ValueError('Half-life and integration step must be finite and positive')
    if list(cfg['kinetic_priors']) != names:
        raise ValueError('kinetic_priors must follow parameter_names in exactly the same order')
    for prior in list(cfg['kinetic_priors'].values()) + [cfg['noise_prior']]:
        if prior.get('type') != 'uniform' or not np.isfinite([prior['min'], prior['max']]).all() or prior['min'] >= prior['max']:
            raise ValueError('New-protocol priors require uniform finite min < max')
    if cfg['noise_prior']['min'] < 0:
        raise ValueError('Noise amplitude cannot be negative')
    for key in ('n_train', 'n_val', 'n_test', 'chunk_size'):
        if isinstance(cfg[key], bool) or not isinstance(cfg[key], int) or cfg[key] < 1:
            raise ValueError(f'{key} must be a positive integer')
    if not isinstance(cfg['seed'], int) or cfg['seed'] < 0:
        raise ValueError('seed must be a nonnegative integer')
    if cfg.get('model', '2TC') == '2TC':
        if names != TwoTC.PARAMETER_NAMES:
            raise ValueError('2TC requires ordered K1,k2,k3,k4,Vb parameters')
        if any(p['min'] < 0 for p in cfg['kinetic_priors'].values()) or cfg['kinetic_priors']['Vb']['max'] > 1:
            raise ValueError('2TC rates must be nonnegative and Vb must lie in [0,1]')
    return t, duration


class TwoTC:
    PARAMETER_NAMES = ['K1', 'k2', 'k3', 'k4', 'Vb']
    STATE_SIZE = 2

    @staticmethod
    def rhs(t, state, parameters, input_value):
        K1, k2, k3, k4, _ = parameters.T
        cf, cm = state.T
        return np.column_stack((K1*input_value-(k2+k3)*cf+k4*cm, k3*cf-k4*cm))

    @staticmethod
    def observe(state, parameters, input_value):
        vb = parameters[:, 4]
        return vb*input_value+(1-vb)*state.sum(axis=1)


def load_model(cfg):
    if cfg.get('model', '2TC') == '2TC':
        model = TwoTC
    elif cfg['model'] == 'custom':
        path = Path(cfg['model_file']).resolve()
        spec = importlib.util.spec_from_file_location('pet_user_kinetics', path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        model = module
    else:
        raise ValueError('Choose 2TC or custom')
    if list(model.PARAMETER_NAMES) != cfg['parameter_names']:
        raise ValueError('Model PARAMETER_NAMES must match the configured parameter order')
    if not isinstance(model.STATE_SIZE, int) or model.STATE_SIZE < 1:
        raise ValueError('STATE_SIZE must be a positive integer')
    return model


def feng_curve(t, parameters):
    """Feng curve with nonnegative coefficients, per-row parameters."""
    p = np.asarray(parameters)
    u = np.maximum(np.asarray(t)[None, :] - p[:, 6:7], 0)
    b1,b2,b3,k1,k2,k3 = [p[:, i:i+1] for i in range(6)]
    return (b1*u-b2-b3)*np.exp(-k1*u)+b2*np.exp(-k2*u)+b3*np.exp(-k3*u)


def read_prediction_input(path, parameter_dim, expected_times=None, expected_durations=None):
    """Read portable NPZ or legacy pandas HDF5, with an identical contract."""
    path = Path(path)
    truth = None
    if path.suffix.lower() == '.npz':
        with np.load(path, allow_pickle=False) as data:
            t = np.asarray(data['t_meas'], dtype=float)
            tac, aif = np.asarray(data['tac'], dtype=float), np.asarray(data['aif'], dtype=float)
            duration = np.asarray(data['frame_durations_min'], dtype=float)
            columns = np.asarray(data['voxel_columns']).astype(str) if 'voxel_columns' in data else np.arange(len(tac)).astype(str)
            if 'truth' in data:
                truth = np.asarray(data['truth'], dtype=float)
    else:
        import pandas as pd
        frame = pd.read_hdf(path)
        if frame.shape[1] < 4:
            raise ValueError('HDF5 needs duration, midpoint, AIF, and at least one TAC column')
        duration, t, aif = [frame.iloc[:, i].to_numpy(dtype=float) for i in range(3)]
        tac = frame.iloc[:, 3:].to_numpy(dtype=float).T
        columns = np.asarray(frame.columns[3:]).astype(str)
    t = validate_times(t)
    if tac.ndim != 2 or tac.shape[0] < 1 or tac.shape[1] != len(t) or not np.isfinite(tac).all():
        raise ValueError('tac must be finite [N, T] with at least one curve')
    if aif.shape == (len(t),):
        aif = np.broadcast_to(aif, tac.shape)
    if aif.shape != tac.shape or not np.isfinite(aif).all() or np.any(aif < 0):
        raise ValueError('aif must be finite nonnegative [T] or [N,T]')
    if duration.shape != t.shape or not np.isfinite(duration).all() or np.any(duration <= 0):
        raise ValueError('frame_durations_min must be finite positive [T]')
    if columns.shape != (len(tac),) or len(np.unique(columns)) != len(columns):
        raise ValueError('voxel_columns must be unique [N] identifiers')
    if truth is not None and (truth.shape != (len(tac), parameter_dim) or not np.isfinite(truth).all()):
        raise ValueError('truth must be finite [N, number of parameters]')
    for actual, expected, label in [(t, expected_times, 'time sampling schedule'), (duration, expected_durations, 'frame durations')]:
        if expected is not None and (actual.shape != np.asarray(expected).shape or not np.allclose(actual, expected, atol=1e-6, rtol=0)):
            raise ValueError(f'Prediction {label} differs from training; adapt to the new protocol first')
    return t, duration, np.concatenate([tac, aif], axis=1), columns, truth


def validate_file(path):
    from common import validate_dataset
    with np.load(path, allow_pickle=False) as data:
        if 'x_train' not in data or data['x_train'].ndim != 2 or data['x_train'].shape[1] < 2:
            raise ValueError('x_train must be [N,P+1], parameters then noise amplitude')
        dim = data['x_train'].shape[1]-1
        names = validate_names(data['parameter_names'].tolist()) if 'parameter_names' in data else None
        if names is None and dim != 5:
            raise ValueError('Custom parameter sets require parameter_names in the NPZ')
        if names is None:
            names = TwoTC.PARAMETER_NAMES
        if len(names) != dim:
            raise ValueError('parameter_names length does not match x arrays')
        validate_dataset(data, dim, 2*len(data['t_meas']))
        if 'observation' in data and str(data['observation']) not in ('midpoint','frame_average'):
            raise ValueError('observation must be midpoint or frame_average')
        if np.any(np.std(data['x_train'][:, :dim], axis=0) <= 0) or np.std(data['y_train']) <= 0:
            raise ValueError('Training parameter and conditioning scales must be nonzero')
        return {'parameter_names': names, 'x_dim': dim, 'y_dim': 2*len(data['t_meas']),
                't_meas': data['t_meas'].tolist(),
                'frame_durations_min': data['frame_durations_min'].tolist() if 'frame_durations_min' in data else None,
                'observation':str(data['observation']) if 'observation' in data else 'midpoint',
                'counts': {s: len(data['x_'+s]) for s in ('train','val','test')}}


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Validate a fine-tuning NPZ without training')
    parser.add_argument('dataset', nargs='?')
    parser.add_argument('--config', help='Validate a simulation protocol and its model plugin')
    args = parser.parse_args()
    if args.config:
        cfg = load_protocol(args.config)
        validate_protocol(cfg); load_model(cfg)
        print(json.dumps({'protocol_valid': True, 'parameter_names': cfg['parameter_names'],
                          'frames':len(cfg['t_meas'])}, indent=2))
    elif args.dataset:
        print(json.dumps(validate_file(args.dataset), indent=2))
    else:
        parser.error('Supply a dataset path or --config protocol.json')
