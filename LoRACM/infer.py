"""Predict voxel posterior summaries on CPU, one GPU, or multiple GPUs on one node."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch import nn
from tqdm import tqdm
from model_factory import build_model
from common import summarize_samples, PARAMETERS
from protocol import read_prediction_input, validate_names

@torch.inference_mode()
def sample(model, y, samples, x_dim, times):
    b,d=y.shape
    repeated=y.repeat_interleave(samples,dim=0)
    x=torch.randn(b*samples,1,x_dim,device=y.device)
    for i,t in enumerate(times):
        if i:
            x=x+t*torch.randn_like(x)
        x=model(x,repeated,torch.full((b*samples,),float(t),device=y.device))
    return x.reshape(b,samples,x_dim)

def main(args):
    if min(args.batch_size,args.sample_chunk,args.steps)<1 or args.samples<2:
        raise ValueError('Batch, sample chunk and steps must be positive; samples >= 2')
    run=Path(args.run)
    cfg=json.loads((run/'model_params.json').read_text(encoding='utf-8'))
    scaling=json.loads((run/'scaling_params.json').read_text(encoding='utf-8'))
    names = validate_names(cfg.get('parameter_names', PARAMETERS))
    if len(names) != cfg['x_dim']:
        raise ValueError('Invalid parameter_names in run configuration')
    observed_times, durations, y, columns, truth = read_prediction_input(
        args.input, cfg['x_dim'],
        None if args.allow_time_mismatch else scaling['t_meas'],
        None if args.allow_time_mismatch else scaling.get('frame_durations_min'))
    expected=np.asarray(scaling['t_meas'])
    if y.shape[1] != cfg['y_dim']:
        raise ValueError('Prediction frame count differs from model conditioning width')
    mismatch=not np.allclose(observed_times,expected,atol=1e-6,rtol=0)
    if args.max_voxels is not None:
        if args.max_voxels<1: raise ValueError('max-voxels must be positive')
        y=y[:args.max_voxels];columns=columns[:args.max_voxels]
        if truth is not None: truth=truth[:args.max_voxels]
    output=Path(args.output)
    if output.exists() or output.with_suffix('.json').exists():
        raise FileExistsError(f'Refusing to overwrite {output}')
    xm=np.asarray(scaling['x_mean']);xs=np.asarray(scaling['x_std'])
    ym=np.asarray(scaling['y_mean']);ys=np.asarray(scaling['y_std'])
    if xm.shape!=(cfg['x_dim'],) or xs.shape!=(cfg['x_dim'],) or not np.isfinite(xm).all() or not np.isfinite(xs).all() or not np.isfinite(ym).all() or not np.isfinite(ys).all() or np.any(xs<=0) or np.any(ys<=0):
        raise ValueError('Invalid normalization statistics')
    y=(y-ym)/ys
    device=torch.device(('cuda:0' if torch.cuda.is_available() else 'cpu') if args.device=='auto' else args.device)
    if args.multi_gpu and (device.type!='cuda' or torch.cuda.device_count()<2):
        raise ValueError('--multi-gpu needs at least two visible CUDA devices')
    if args.multi_gpu and device.index not in (None,0):
        raise ValueError('Use cuda:0 as the primary DataParallel device')
    torch.manual_seed(args.seed);np.random.seed(args.seed)
    model=build_model(cfg,device,training=False)
    ckpt=run/args.checkpoint
    state=torch.load(ckpt,map_location=device,weights_only=True)
    if state.get('training_mode',cfg.get('training_mode','lora'))!=cfg.get('training_mode','lora'):
        raise ValueError('Checkpoint training mode differs from model_params.json')
    model.load_state_dict(state['model_state_dict'],strict=True)
    model.eval()
    if args.multi_gpu: model=nn.DataParallel(model)
    times=np.linspace(1.,0.,num=args.steps,endpoint=False)
    if getattr(args, 'save_samples', False) and len(y)*args.samples*cfg['x_dim'] > 2_000_000:
        raise ValueError('Saving posterior draws is limited to 2 million values; use fewer curves/samples')
    outputs={}
    saved_draws=[]
    for start in tqdm(range(0,len(y),args.batch_size)):
        obs=torch.as_tensor(y[start:start+args.batch_size],dtype=torch.float32,device=device)
        draws=[]
        for offset in range(0,args.samples,args.sample_chunk):
            z=sample(model,obs,min(args.sample_chunk,args.samples-offset),cfg['x_dim'],times)
            draws.append(z.cpu().numpy()*xs+xm)
        if getattr(args, 'save_samples', False):
            saved_draws.append(np.concatenate(draws,axis=1))
        # Retain only one voxel batch of samples in CPU memory for exact medians.
        for key,value in summarize_samples(np.concatenate(draws,axis=1), names, calculate_ki=(names == PARAMETERS and cfg.get('kinetic_model', '2TC') == '2TC')).items():
            outputs.setdefault(key,[]).append(value)
    output.parent.mkdir(parents=True,exist_ok=True)
    if output.exists(): raise FileExistsError(f'Refusing to overwrite {output}')
    metadata={'config':cfg,'checkpoint':str(ckpt),'input':str(args.input),'samples':args.samples,
        'batch_size':args.batch_size,'sample_chunk':args.sample_chunk,'seed':args.seed,
        'sampling_times':times.tolist(),'time_mismatch_allowed':bool(args.allow_time_mismatch), 'time_mismatch_detected':bool(mismatch), 'observed_frame_durations_min':durations.tolist(),'torch_version':str(torch.__version__),
        'device':str(device),'multi_gpu':args.multi_gpu,
        'Ki_policy':('K1*k3/(k2+k3) for each joint draw; no clipping; |denominator|<=1e-12 excluded' if names == PARAMETERS and cfg.get('kinetic_model', '2TC') == '2TC' else 'Not calculated for this kinetic model'),
        'parameter_order':names,'std_ddof':0}
    if truth is not None:
        means = np.concatenate(outputs['params_mean'])
        metadata['test_rmse'] = np.sqrt(np.mean((means-truth)**2, axis=0)).tolist()
        metadata['test_bias'] = np.mean(means-truth, axis=0).tolist()
    extra = {'posterior_samples':np.concatenate(saved_draws)} if saved_draws else {}
    np.savez_compressed(output, **extra,**{k:np.concatenate(v) for k,v in outputs.items()},
        voxel_columns=columns,parameter_names=np.asarray(names),t_meas=expected,
        observed_t_meas=observed_times,metadata_json=json.dumps(metadata))
    output.with_suffix('.json').write_text(json.dumps(metadata,indent=2)+'\n', encoding='utf-8')
    print(f'Saved {output}')

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',default='runs/scenario1')
    p.add_argument('--checkpoint',default='best.pth')
    p.add_argument('--input',default='data/real/Sub001_slice75_adjusted.h5')
    p.add_argument('--output',default='runs/scenario1/predictions.npz')
    p.add_argument('--samples',type=int,default=1000)
    p.add_argument('--batch-size',type=int,default=16)
    p.add_argument('--sample-chunk',type=int,default=64)
    p.add_argument('--save-samples',action='store_true',help='Save joint draws for small curve sets (up to 2 million values)')
    p.add_argument('--steps',type=int,default=3)
    p.add_argument('--seed',type=int,default=0)
    p.add_argument('--device',default='auto',choices=['auto','cpu','cuda:0','mps'])
    p.add_argument('--multi-gpu',action='store_true')
    p.add_argument('--max-voxels',type=int)
    p.add_argument('--allow-time-mismatch',action='store_true')
    main(p.parse_args())
