"""Generate fine-tuning pairs for explicit frame protocols and model plugins."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
from protocol import validate_protocol, validate_times, load_model, feng_curve, FENG_NAMES, load_protocol
from common import validate_dataset
from runtime import Progress, check_cancelled


def load_inputs(cfg, end):
    frame = pd.read_csv(cfg['aif_csv'])
    t = validate_times(frame.iloc[:, 0].to_numpy(), 'AIF times', allow_zero=True)
    aifs = frame.iloc[:, 1:].to_numpy(dtype=float).T
    if aifs.shape[0] < 1 or not np.isfinite(aifs).all() or np.any(aifs < 0):
        raise ValueError('AIF CSV requires at least one finite nonnegative activity column')
    if t[-1] < end-1e-8:
        raise ValueError('Measured AIF must cover the full last frame; extend it explicitly or use midpoint observations')
    if t[0] > 0:
        if cfg.get('aif_before_first') != 'zero_at_origin':
            raise ValueError('AIF must include time zero or explicitly select aif_before_first=zero_at_origin')
        t = np.r_[0., t]
        aifs = np.column_stack([np.zeros(len(aifs)), aifs])
    return t, aifs


def integrate(model, parameters, inputs, grid):
    """Vectorised fixed-grid RK4; inputs are evaluated at grid points and midpoints."""
    state = np.zeros((len(parameters), model.STATE_SIZE))
    output = np.empty((len(parameters), len(grid)))
    output[:, 0] = model.observe(state, parameters, inputs[:, 0])
    for i, dt in enumerate(np.diff(grid)):
        cp, cn = inputs[:, i], inputs[:, i+1]
        cm = (cp+cn)/2
        k1 = np.asarray(model.rhs(grid[i], state, parameters, cp))
        k2 = np.asarray(model.rhs(grid[i]+dt/2, state+dt*k1/2, parameters, cm))
        k3 = np.asarray(model.rhs(grid[i]+dt/2, state+dt*k2/2, parameters, cm))
        k4 = np.asarray(model.rhs(grid[i+1], state+dt*k3, parameters, cn))
        if any(k.shape != state.shape for k in (k1,k2,k3,k4)):
            raise ValueError('Model rhs must return [N, STATE_SIZE]')
        state = state+dt*(k1+2*k2+2*k3+k4)/6
        obs = np.asarray(model.observe(state, parameters, cn))
        if obs.shape != (len(parameters),):
            raise ValueError('Model observe must return [N]')
        output[:, i+1] = obs
    if not np.isfinite(output).all():
        raise ValueError('Non-finite simulation; check model, priors and integration step')
    return output


def observe_frames(values, grid, t, durations, mode):
    if mode == 'midpoint':
        return values[:, np.searchsorted(grid, t)]
    areas = np.cumsum((values[:, 1:]+values[:, :-1])/2*np.diff(grid), axis=1)
    areas = np.column_stack([np.zeros(len(values)), areas])
    starts = np.searchsorted(grid, np.maximum(0, t-durations/2))
    ends = np.searchsorted(grid, t+durations/2)
    return (areas[:, ends]-areas[:, starts])/durations


def main(cfg):
    t, duration = validate_protocol(cfg)
    model = load_model(cfg)
    path = Path(cfg['output'])
    if path.suffix != '.npz':
        raise ValueError('Dataset output must end in .npz')
    if path.exists() or path.with_suffix('.json').exists():
        raise FileExistsError('Dataset or provenance file exists; choose a new output')
    mode = cfg.get('observation', 'midpoint')
    end = t[-1]+duration[-1]/2 if mode == 'frame_average' else t[-1]
    grid = np.unique(np.r_[np.arange(0, end, cfg.get('integration_step_min', .01)),
                           np.maximum(0, t-duration/2), t, t+duration/2 if mode == 'frame_average' else t, end])
    grid = grid[grid <= end]
    source = cfg.get('aif_mode', 'measured')
    if source == 'measured':
        aif_t, pool = load_inputs(cfg, end)
        grid = np.unique(np.r_[grid, aif_t[aif_t <= end]])
    elif source == 'feng_cohort':
        cohort = pd.read_csv(cfg['feng_parameters_csv'])
        pool = cohort[FENG_NAMES].to_numpy(dtype=float)
        if len(pool) < 2 or not np.isfinite(pool).all() or np.any(pool < 0):
            raise ValueError('Feng cohort requires at least two finite nonnegative parameter rows')
        jitter = float(cfg.get('feng_log_jitter', 0.05))
        if not np.isfinite(jitter) or not 0 <= jitter <= .5:
            raise ValueError('feng_log_jitter must be between 0 and 0.5')
    else:
        raise ValueError('aif_mode must be measured or feng_cohort')
    rng = np.random.default_rng(cfg['seed'])
    data = {'t_meas': t, 'frame_durations_min': duration,
            'parameter_names': np.asarray(cfg['parameter_names']),
            'observation': np.asarray(mode)}
    total_pairs = sum(cfg['n_'+split] for split in ('train','val','test'))
    progress = Progress('Dataset simulation',total_pairs,'pairs')
    completed = 0
    for split in ('train','val','test'):
        xs, ys, indices = [], [], []
        total = cfg['n_'+split]
        for start in range(0, total, cfg['chunk_size']):
            check_cancelled()
            n = min(cfg['chunk_size'], total-start)
            scale = cfg.get('test_prior_scale', 1.) if split == 'test' else 1.
            ranges = [((p['min']+p['max'])/2-(p['max']-p['min'])*scale/2,
                       (p['min']+p['max'])/2+(p['max']-p['min'])*scale/2) for p in cfg['kinetic_priors'].values()]
            parameters = np.column_stack([rng.uniform(low,high,n) for low,high in ranges])
            selected = rng.integers(0, len(pool), n)
            if source == 'measured':
                inputs = np.asarray([np.interp(grid, aif_t, curve) for curve in pool[selected]])
            else:
                fp = pool[selected]*np.exp(rng.normal(0,jitter,(n,7)))
                inputs = feng_curve(grid, fp)
                # The unconstrained Feng equation can yield negative activity.
                if np.any(inputs < -1e-8):
                    raise ValueError('Sampled Feng curves became negative; inspect fitted curves or reduce jitter')
                inputs = np.maximum(inputs,0)
            clean = observe_frames(integrate(model,parameters,inputs,grid),grid,t,duration,mode)
            if np.any(clean < -1e-8):
                raise ValueError('Clean simulated TAC is negative; check model and priors')
            measured_input = observe_frames(inputs,grid,t,duration,mode)
            noise = rng.uniform(cfg['noise_prior']['min'],cfg['noise_prior']['max'],(n,1))
            decay = np.log(2)/cfg['half_life_min']
            sigma = noise*np.sqrt(np.maximum(clean,1e-8)*np.exp(-decay*t)/duration)*np.exp(decay*t)
            noisy = clean+sigma*rng.normal(size=clean.shape)
            xs.append(np.column_stack([parameters,noise]))
            ys.append(np.column_stack([noisy,measured_input]))
            indices.append(selected)
            completed += n
            progress.update(completed, split=split, split_completed=start+n, split_total=total)
        data['x_'+split],data['y_'+split] = np.concatenate(xs),np.concatenate(ys)
        data['aif_index_'+split] = np.concatenate(indices)
    validate_dataset(data,len(cfg['parameter_names']),2*len(t))
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('xb') as stream:
        np.savez_compressed(stream,**data)
    source_path = cfg['aif_csv'] if source == 'measured' else cfg['feng_parameters_csv']
    metadata = {'config':cfg,'parameter_order':cfg['parameter_names']+['noise_scale'],
                'conditioning_order':'TAC then AIF', 'solver':'Fixed-step RK4, zero initial state, linear input interpolation',
                'source_sha256':hashlib.sha256(Path(source_path).read_bytes()).hexdigest(),
                'numpy_version':np.__version__, 'noise_timing':'Explicit frame durations; decay constant ln(2)/half_life_min',
                'split_policy':'Independent parameter/noise draws, shared measured/fitted AIF pool; not subject-held-out'}
    if cfg.get('model') == 'custom':
        metadata['model_sha256'] = hashlib.sha256(Path(cfg['model_file']).read_bytes()).hexdigest()
    path.with_suffix('.json').write_text(json.dumps(metadata,indent=2)+'\n', encoding='utf-8')
    progress.update(total_pairs,state='complete',output=str(path))
    print(f'Saved {path}',flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', required=True)
    main(load_protocol(p.parse_args().config))
