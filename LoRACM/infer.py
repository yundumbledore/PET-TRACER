"""Predict voxel posterior summaries on CPU, one GPU, or multiple GPUs on one node."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch import nn
from tqdm import tqdm
from model_lora import build_lora_unet
from common import summarize_samples, PARAMETERS

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
    cfg=json.loads((run/'model_params.json').read_text())
    scaling=json.loads((run/'scaling_params.json').read_text())
    if cfg['x_dim']!=5:
        raise ValueError('This released inference demo supports Scenario 1 (five parameters)')
    frame=pd.read_hdf(args.input)
    if frame.shape[1]<4 or frame.shape[0]*2!=cfg['y_dim']:
        raise ValueError('Expected rows=frames; columns=duration, time, AIF, then voxel TACs')
    observed_times=frame.iloc[:,1].to_numpy(dtype=float)
    expected=np.asarray(scaling['t_meas'])
    mismatch=not np.allclose(observed_times,expected,atol=1e-6,rtol=0)

    aif=frame.iloc[:,2].to_numpy(dtype=float)
    tac=frame.iloc[:,3:].to_numpy(dtype=float).T
    columns=np.asarray([str(c) for c in frame.columns[3:]])
    if args.max_voxels is not None:
        if args.max_voxels<1: raise ValueError('max-voxels must be positive')
        tac=tac[:args.max_voxels];columns=columns[:args.max_voxels]
    y=np.concatenate([tac,np.broadcast_to(aif,tac.shape)],axis=1)
    if not np.isfinite(y).all(): raise ValueError('Non-finite TAC or AIF values')
    xm=np.asarray(scaling['x_mean']);xs=np.asarray(scaling['x_std'])
    ym=np.asarray(scaling['y_mean']);ys=np.asarray(scaling['y_std'])
    if xm.shape!=(5,) or xs.shape!=(5,) or np.any(xs<=0) or np.any(ys<=0):
        raise ValueError('Invalid normalization statistics')
    y=(y-ym)/ys
    device=torch.device(('cuda:0' if torch.cuda.is_available() else 'cpu') if args.device=='auto' else args.device)
    if args.multi_gpu and (device.type!='cuda' or torch.cuda.device_count()<2):
        raise ValueError('--multi-gpu needs at least two visible CUDA devices')
    if args.multi_gpu and device.index not in (None,0):
        raise ValueError('Use cuda:0 as the primary DataParallel device')
    torch.manual_seed(args.seed);np.random.seed(args.seed)
    model=build_lora_unet(x_dim=cfg['x_dim'],y_dim=cfg['y_dim'],embed_dim=cfg['embed_dim'],
        channels=cfg['channels'],embedy=cfg['embedy'],sigma_data=cfg['sigma_data'],r=cfg['lora_r'],
        alpha=cfg['lora_alpha'],dropout=cfg['lora_dropout'],exclude_names=('decodex',),
        pretrained_ckpt=None,train_norms=False,device=device)
    ckpt=run/args.checkpoint
    state=torch.load(ckpt,map_location=device,weights_only=True)
    model.load_state_dict(state['model_state_dict'],strict=True)
    model.eval()
    if args.multi_gpu: model=nn.DataParallel(model)
    times=np.linspace(1.,0.,num=args.steps,endpoint=False)
    outputs={}
    for start in tqdm(range(0,len(y),args.batch_size)):
        obs=torch.as_tensor(y[start:start+args.batch_size],dtype=torch.float32,device=device)
        draws=[]
        for offset in range(0,args.samples,args.sample_chunk):
            z=sample(model,obs,min(args.sample_chunk,args.samples-offset),5,times)
            draws.append(z.cpu().numpy()*xs+xm)
        # Retain only one voxel batch of samples in CPU memory for exact medians.
        for key,value in summarize_samples(np.concatenate(draws,axis=1)).items():
            outputs.setdefault(key,[]).append(value)
    output=Path(args.output);output.parent.mkdir(parents=True,exist_ok=True)
    if output.exists(): raise FileExistsError(f'Refusing to overwrite {output}')
    metadata={'config':cfg,'checkpoint':str(ckpt),'input':str(args.input),'samples':args.samples,
        'batch_size':args.batch_size,'sample_chunk':args.sample_chunk,'seed':args.seed,
        'sampling_times':times.tolist(),'time_mismatch_allowed':bool(mismatch),'torch_version':str(torch.__version__),
        'device':str(device),'multi_gpu':args.multi_gpu,
        'Ki_policy':'K1*k3/(k2+k3) for each joint draw; no clipping; |denominator|<=1e-12 excluded',
        'parameter_order':PARAMETERS,'std_ddof':0}
    np.savez_compressed(output,**{k:np.concatenate(v) for k,v in outputs.items()},
        voxel_columns=columns,parameter_names=np.asarray(PARAMETERS),t_meas=expected,
        observed_t_meas=observed_times,metadata_json=json.dumps(metadata))
    output.with_suffix('.json').write_text(json.dumps(metadata,indent=2)+'\n')
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
    p.add_argument('--steps',type=int,default=3)
    p.add_argument('--seed',type=int,default=0)
    p.add_argument('--device',default='auto',choices=['auto','cpu','cuda:0','mps'])
    p.add_argument('--multi-gpu',action='store_true')
    p.add_argument('--max-voxels',type=int)
    p.add_argument('--allow-time-mismatch',action='store_true')
    main(p.parse_args())
