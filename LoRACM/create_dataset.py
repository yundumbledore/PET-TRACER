"""Generate Scenario 1 pairs from measured AIFs, with bounded simulation memory."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
import jax
import jax.numpy as jnp
from jax import random
from simulation import generate_dataset, TwoTissueCompartment, get_central_priors
from common import load_config, validate_dataset, PARAMETERS


def main(cfg):
    frame = pd.read_csv(cfg['aif_csv'])
    t = frame.iloc[:,0].to_numpy(dtype=float)
    aifs = frame.iloc[:,1:].to_numpy(dtype=float).T
    if len(t)<2 or t[0]<=0 or not np.all(np.diff(t)>0):
        raise ValueError('First CSV column must contain increasing positive times in minutes')
    if not np.isfinite(t).all() or not np.isfinite(aifs).all() or aifs.shape[0]<1 or np.any(aifs<0):
        raise ValueError('AIF values must be finite, nonnegative and nonempty')
    if list(cfg['kinetic_priors']) != PARAMETERS:
        raise ValueError(f'Prior order must be {PARAMETERS}')
    if cfg['chunk_size']<1 or any(cfg[f'n_{s}']<1 for s in ('train','val','test')):
        raise ValueError('Chunk and split sizes must be positive')
    rng=np.random.default_rng(cfg['seed'])
    split_keys=random.split(random.PRNGKey(cfg['seed']),3)
    data={'t_meas':t}
    for split, key in zip(('train','val','test'),split_keys):
        count=cfg[f'n_{split}']
        indices=rng.integers(0,len(aifs),size=count)
        priors=cfg['kinetic_priors']
        if split=='test':
            priors=get_central_priors(priors,scale=cfg['test_prior_scale'])
        xs=[];ys=[]
        for chunk,start in enumerate(range(0,count,cfg['chunk_size'])):
            selected=indices[start:start+cfg['chunk_size']]
            y,x=generate_dataset(len(selected),TwoTissueCompartment,priors,{},cfg['noise_prior'],
                jnp.asarray(t),random.fold_in(key,chunk),cfg['half_life_min'],
                external_input_curves=aifs[selected],return_clean=False)
            xs.append(x);ys.append(y)
        data[f'x_{split}']=np.concatenate(xs)
        data[f'y_{split}']=np.concatenate(ys)
        data[f'aif_index_{split}']=indices
        print(f'{split}: {count} pairs')
    validate_dataset(data,5,2*len(t))
    path=Path(cfg['output']);path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists():
        raise FileExistsError(f'Refusing to overwrite {path}; move it or change output in the config')
    np.savez_compressed(path,**data)
    metadata={'config':cfg,'aif_columns':list(frame.columns[1:]),'parameter_order':PARAMETERS+['noise_scale'],
        'conditioning_order':'TAC then AIF','aif_sha256':hashlib.sha256(Path(cfg['aif_csv']).read_bytes()).hexdigest(),
        'numpy_version':np.__version__,'jax_version':jax.__version__,
        'randomness':'Independent keys per split and chunk; chunk size is part of the reproducibility configuration.',
        'time_convention':'Supplied midpoint sampling; noise delta_t=[t0,diff(t)], not measured frame durations.',
        'solver':'Original forward Euler, nominal 0.01 min step; searchsorted sampling.'}
    path.with_suffix('.json').write_text(json.dumps(metadata,indent=2)+'\n')
    print(f'Saved {path}')

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',default='configs/dataset_scenario1.json')
    main(load_config(p.parse_args().config))
