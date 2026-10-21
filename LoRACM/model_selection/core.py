"""Validated data, joint-sample decoding and model-conditional forward checks."""
from pathlib import Path
import json
import numpy as np
import pandas as pd

ASSETS = Path(__file__).resolve().parent/'assets'
MODELS = ('0TC', '1TC', '1TC-Dual', '2TC', '2TC-Dual', '3TC', '3TC-Dual')
COLORS = ('gray', 'red', 'blue', 'orange', 'green', 'purple', 'cyan')
PARAMETERS = ('K1','k2','k3','k4','k5','k6','Vb','ka','fA','model','delay_min')
TITLES = (r'$K_1$',r'$k_2$',r'$k_3$',r'$k_4$',r'$k_5$',r'$k_6$',r'$V_b$',r'$k_a$',r'$f_A$','Model',r'$t_d$ (min)')
ACTIVE = {0:(6,),1:(0,1,6),2:(0,1,6,7,8),3:(0,1,2,3,6),
          4:(0,1,2,3,6,7,8),5:(0,1,2,3,4,5,6),6:(0,1,2,3,4,5,6,7,8)}


def load_demo():
    """Preserve masked, C-order voxel identity and the original 35-frame schedule."""
    frame = pd.read_hdf(ASSETS/'P001_slice116.h5')
    mask = np.load(ASSETS/'P001_slice116_mask.npy',allow_pickle=False).astype(bool)
    times = frame.iloc[:,1].to_numpy(dtype=np.float64)
    aif = frame.iloc[:,2].to_numpy(dtype=np.float32)
    tac = frame.iloc[:,3:].to_numpy(dtype=np.float32).T.copy()
    if mask.ndim!=2 or tac.shape!=(int(mask.sum()),35):
        raise ValueError('P001 TAC columns do not match the slice mask and 35-frame protocol')
    if len(times)!=35 or np.any(np.diff(times)<=0) or times[0]<0:
        raise ValueError('The supplied acquisition times must increase in minutes')
    if not all(np.isfinite(a).all() for a in (times,aif,tac)):
        raise ValueError('Non-finite TAC, AIF or time value')
    ids = np.full(mask.shape,-1,dtype=np.int32)
    ids[mask] = np.arange(len(tac),dtype=np.int32)
    return dict(tac=tac,aif=aif,times=times,mask=mask,ids=ids,
                coordinates=np.argwhere(mask),spacing=np.array([3.3,1.645]))


def load_config():
    cfg=json.loads((ASSETS/'config.json').read_text(encoding='utf-8'))
    if cfg['model_settings']['x_dim']!=11 or cfg['model_settings']['y_dim']!=70:
        raise ValueError('This release requires the supplied V8TD 11-by-70 model')
    for key,length in [('x_mean',11),('x_std',11),('y_mean',70),('y_std',70)]:
        a=np.asarray(cfg['scaling_params'][key],dtype=float)
        if a.shape!=(length,) or not np.isfinite(a).all() or (key.endswith('std') and np.any(a<=0)):
            raise ValueError('Invalid checkpoint normalisation: '+key)
    return cfg


def decode(samples):
    """Round the model slot, mask unused kinetics, and retain each draw's delay."""
    samples=np.asarray(samples)
    if samples.shape[-1]!=11 or not np.isfinite(samples).all():
        raise ValueError('Expected finite joint samples with eleven parameters')
    indices=np.clip(np.rint(samples[...,9]),0,6).astype(np.int64)
    out=np.zeros_like(samples)
    out[...,9]=indices
    out[...,10]=samples[...,10]
    for m,active in ACTIVE.items():
        keep=indices==m
        for p in active:out[...,p]=np.where(keep,samples[...,p],out[...,p])
    return out


def probabilities(samples):
    votes=np.clip(np.rint(samples[...,9]),0,6).astype(np.int64)
    return np.stack([(votes==m).mean(axis=-1) for m in range(7)],axis=-1)


def conditional_samples(samples,model):
    return samples[np.clip(np.rint(samples[:,9]),0,6).astype(int)==int(model)]


def forward_median(samples,model,times,aif):
    """Use only draws assigned to the selected model, including median delay."""
    from .kinetics import simulate_kinetic_model
    draws=conditional_samples(np.asarray(samples),model)
    if not len(draws):raise ValueError('No posterior draws selected this model')
    active=ACTIVE[int(model)]
    med=np.median(draws,axis=0)
    if not np.isfinite(med[list(active)+[10]]).all():raise ValueError('Non-finite median')
    if np.any(med[list(active)]<0) or med[6]>1 or (model in (2,4,6) and med[8]>1):
        raise ValueError('Selected-model medians include a nonphysical kinetic value; samples are shown without clipping')
    # Same dense grid, first update and searchsorted convention as the notebook.
    times=np.asarray(times,dtype=float);aif=np.asarray(aif,dtype=float)
    step=max(.01,(times[-1]-times[0])/6000)
    dense=np.arange(times[0],times[-1]+1e-8,step)
    if dense[-1]<times[-1]:dense=np.r_[dense,times[-1]]
    dt=np.r_[dense[0],np.diff(dense)]
    delayed=np.interp(dense-med[10],times,aif,left=0.)
    with np.errstate(over='ignore',invalid='ignore'):
        fitted=simulate_kinetic_model(int(model),med[list(active)],delayed,dt,np.searchsorted(dense,times))
    if not np.isfinite(fitted).all():raise ValueError('The forward simulation is numerically unstable for these medians')
    return fitted,med,len(draws)


def forward_posterior(samples,model,times,aif):
    """Propagate every selected-model joint draw; summarize finite latent curves.

    Each draw keeps its own parameter vector and delay. The pointwise 95% band
    describes curve uncertainty, without adding observation noise. Nonphysical
    finite draws are retained and counted; non-finite curves are reported.
    """
    from .kinetics import simulate_kinetic_model
    model=int(model);active=list(ACTIVE[model])
    draws=conditional_samples(np.asarray(samples),model)
    if not len(draws):raise ValueError('No posterior draws selected this model')
    times=np.asarray(times,dtype=float);aif=np.asarray(aif,dtype=float)
    step=max(.01,(times[-1]-times[0])/6000)
    dense=np.arange(times[0],times[-1]+1e-8,step)
    if dense[-1]<times[-1]:dense=np.r_[dense,times[-1]]
    dt=np.r_[dense[0],np.diff(dense)];indices=np.searchsorted(dense,times)
    curves=np.full((len(draws),len(times)),np.nan)
    finite=np.isfinite(draws[:,active+[10]]).all(axis=1)
    positions=np.flatnonzero(finite)
    # Vectorize across draws while bounding dense-grid memory for large runs.
    for start in range(0,len(positions),256):
        selected=positions[start:start+256];batch=draws[selected]
        with np.errstate(over='ignore',invalid='ignore',divide='ignore'):
            delayed=np.interp(dense[:,None]-batch[:,10],times,aif,left=0.)
            curves[selected]=simulate_kinetic_model(model,batch[:,active].T,delayed,dt,indices).T
    usable=np.isfinite(curves).all(axis=1)
    if not usable.any():
        raise ValueError(f'None of the {len(draws)} selected-model draws produced a finite forward curve; posterior samples were not clipped')
    curves=curves[usable]
    nonphysical=np.any(draws[:,active]<0,axis=1)|(draws[:,6]>1)
    if model in (2,4,6):nonphysical|=draws[:,8]>1
    lower,upper=np.quantile(curves,[.025,.975],axis=0)
    return dict(curves=curves,mean=np.mean(curves,axis=0),lower=lower,upper=upper,
                count=len(curves),total=len(draws),nonphysical=int(np.count_nonzero(nonphysical&usable)))


def load_result(folder):
    folder=Path(folder)
    metadata=json.loads((folder/'metadata.json').read_text(encoding='utf-8'))
    manifest=json.loads((ASSETS/'manifest.json').read_text(encoding='utf-8'))
    if metadata.get('subject')!='P001' or metadata.get('slice_index')!=116:
        raise ValueError('This release displays P001 coronal slice 116 only')
    for key,name in [('input_sha256','P001_slice116.h5'),('mask_sha256','P001_slice116_mask.npy'),
                     ('checkpoint_sha256','epoch1169.pth'),('config_sha256','config.json')]:
        if metadata.get(key)!=manifest['files'][name]['sha256']:
            raise ValueError('Result does not match the bundled data/model: '+key)
    with np.load(folder/'selection.npz',allow_pickle=False) as f:
        result={k:f[k] for k in f.files}
    samples=np.load(folder/'posteriors.npy',mmap_mode='r',allow_pickle=False)
    count=len(result['voxel_indices'])
    if samples.ndim!=3 or samples.shape[0]!=count or samples.shape[2]!=11:
        raise ValueError('Posterior file does not match the selection summary')
    p=result['probabilities']
    if p.shape!=(count,7) or not np.isfinite(p).all() or not np.allclose(p.sum(1),1):
        raise ValueError('Invalid model probabilities')
    result['posteriors']=samples
    return result
