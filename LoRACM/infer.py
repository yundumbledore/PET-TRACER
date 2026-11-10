"""Predict voxel posterior summaries on CPU, one GPU, or multiple GPUs on one node."""
from protocol import prediction_times
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch import nn
import copy
from model_factory import build_model
from common import summarize_samples, PARAMETERS, file_sha256, next_prediction_output
from protocol import read_prediction_input, validate_names
from runtime import select_device, Progress, check_cancelled
from lora import merge_for_inference
from posterior_registry import posterior_method, DEFAULT_STEPS
from posterior_methods import imf_sample, scm_sample, sampling_schedule

# Bound the *expanded* neural-network batch, not just the number of voxels.
# MPS has documented large-batch linear-kernel correctness limits; this also
# prevents a user's voxel_batch × sample_chunk choice from exhausting memory.
MPS_MAX_EXPANDED_BATCH = 4096


def inference_chunks(batch, chunk, device):
    if device.type == 'mps':
        batch = min(batch, MPS_MAX_EXPANDED_BATCH)
        chunk = min(chunk, max(1, MPS_MAX_EXPANDED_BATCH//batch))
    return batch, chunk


@torch.inference_mode()
def check_mps_forward(model, conditioning, x_dim, expanded_batch=MPS_MAX_EXPANDED_BATCH, *, method='cm', max_time=1.):
    """Compare identical tensors/weights on CPU and MPS before publishing results."""
    reference = copy.deepcopy(model).to('cpu')
    generator = torch.Generator().manual_seed(1729)
    batch=max(len(conditioning),expanded_batch)
    conditioning=conditioning.repeat((batch+len(conditioning)-1)//len(conditioning),1)[:batch].contiguous()
    x = torch.randn(batch,1,x_dim,generator=generator)
    t = torch.linspace(.25*max_time,max_time,batch)
    indices=torch.linspace(0,batch-1,min(16,batch)).long()
    r = t * torch.linspace(0.,.9,batch) if method == 'imf' else None
    expected_args = (x[indices],conditioning[indices].cpu(),t[indices])
    actual_args = (x.to('mps'),conditioning.to('mps'),t.to('mps'))
    if r is not None:
        expected_args += (r[indices],)
        actual_args += (r.to('mps'),)
    expected = reference(*expected_args)
    actual = model(*actual_args).cpu()[indices]
    difference = float((expected-actual).abs().max())
    if not torch.isfinite(actual).all() or not torch.allclose(actual,expected,atol=2e-3,rtol=2e-3):
        raise ValueError(f'MPS/CPU numerical check failed (max difference {difference:.6g}). Select CPU and retain this diagnostic for your PyTorch/macOS version.')
    return {'cpu_comparison_max_absolute_error':difference,'checked_rows':len(indices),'expanded_batch':batch,'atol':.002,'rtol':.002}

@torch.inference_mode()
def sample(model, y, samples, x_dim, times, *, method='cm', sigma_data=1.):
    b,d=y.shape
    repeated=y.repeat_interleave(samples,dim=0)
    # CPU-generated noise on MPS avoids backend-specific random-view issues.
    noise_device='cpu' if y.device.type=='mps' else y.device
    def noise():
        return torch.randn(b*samples,1,x_dim,device=noise_device,dtype=y.dtype).to(y.device)
    if method == 'imf':
        return imf_sample(model,repeated,noise,times).reshape(b,samples,x_dim)
    if method == 'scm':
        return scm_sample(model,repeated,noise,times,sigma_data=sigma_data).reshape(b,samples,x_dim)
    if method != 'cm':
        raise ValueError(f'Unknown posterior sampler: {method}')
    x=noise()
    for i,t in enumerate(times):
        if i:
            x=x+t*torch.randn(x.shape,device=noise_device,dtype=x.dtype).to(y.device)
        x=model(x,repeated,torch.full((b*samples,),float(t),device=y.device))
    return x.reshape(b,samples,x_dim)

def main(args):
    if min(args.batch_size,args.sample_chunk)<1 or args.samples<2:
        raise ValueError('Batch, sample chunk and steps must be positive; samples >= 2')
    run=Path(args.run)
    cfg=json.loads((run/'model_params.json').read_text(encoding='utf-8'))
    method=posterior_method(cfg)
    steps=DEFAULT_STEPS[method] if args.steps is None else args.steps
    if isinstance(steps,bool) or not isinstance(steps,int) or steps<1:
        raise ValueError('Sampling steps must be a positive integer')
    sigma_data=float(cfg.get('sigma_data',1.))
    times=sampling_schedule(method,steps,sigma_data=sigma_data)
    scaling=json.loads((run/'scaling_params.json').read_text(encoding='utf-8'))
    names = validate_names(cfg.get('parameter_names', PARAMETERS))
    if len(names) != cfg['x_dim']:
        raise ValueError('Invalid parameter_names in run configuration')
    observed_times, durations, y, columns, truth = read_prediction_input(
        args.input, cfg['x_dim'],
        None if args.allow_time_mismatch else prediction_times(scaling),
        None if args.allow_time_mismatch else scaling.get('frame_durations_min'))
    training_times=np.asarray(scaling['t_meas'])
    expected=observed_times
    time_label_difference=not np.allclose(observed_times,training_times,atol=1e-6,rtol=0)
    if y.shape[1] != cfg['y_dim']:
        raise ValueError('Prediction frame count differs from model conditioning width')
    contract=prediction_times(scaling)
    choices=list(contract.values()) if isinstance(contract,dict) else [contract]
    mismatch=not any(np.shape(choice)==observed_times.shape and np.allclose(observed_times,choice,atol=1e-6,rtol=0) for choice in choices)
    if args.max_voxels is not None:
        if args.max_voxels<1: raise ValueError('max-voxels must be positive')
        y=y[:args.max_voxels];columns=columns[:args.max_voxels]
        if truth is not None: truth=truth[:args.max_voxels]
    output=next_prediction_output(args.output) if getattr(args,'new_output',False) else Path(args.output)
    if output.exists() or output.with_suffix('.json').exists():
        raise FileExistsError(f'Refusing to overwrite {output}')
    xm=np.asarray(scaling['x_mean']);xs=np.asarray(scaling['x_std'])
    ym=np.asarray(scaling['y_mean']);ys=np.asarray(scaling['y_std'])
    if xm.shape!=(cfg['x_dim'],) or xs.shape!=(cfg['x_dim'],) or not np.isfinite(xm).all() or not np.isfinite(xs).all() or not np.isfinite(ym).all() or not np.isfinite(ys).all() or np.any(xs<=0) or np.any(ys<=0):
        raise ValueError('Invalid normalization statistics')
    y=(y-ym)/ys
    device=select_device(args.device)
    if args.multi_gpu and (device.type!='cuda' or torch.cuda.device_count()<2):
        raise ValueError('--multi-gpu needs at least two visible CUDA devices')
    if args.multi_gpu and device.index not in (None,0):
        raise ValueError('Use cuda:0 as the primary DataParallel device')
    torch.manual_seed(args.seed);np.random.seed(args.seed)
    model=build_model(cfg,torch.device('cpu'),training=False)
    ckpt=run/args.checkpoint
    state=torch.load(ckpt,map_location='cpu',weights_only=True)
    if state.get('training_mode',cfg.get('training_mode','lora'))!=cfg.get('training_mode','lora'):
        raise ValueError('Checkpoint training mode differs from model_params.json')
    if state.get('posterior_method','cm') != method:
        raise ValueError('Checkpoint posterior method differs from model_params.json')
    model.load_state_dict(state['model_state_dict'],strict=True)
    model.eval()
    merge_for_inference(model)
    model.to(device)
    mps_check=None
    batch_size,sample_chunk=inference_chunks(args.batch_size,args.sample_chunk,device)
    print(f'Device: {device}; sampler: {method}; effective curve batch {batch_size}, draw chunk {sample_chunk}; steps {steps}',flush=True)
    if device.type=='mps':
        indices=np.linspace(0,len(y)-1,min(8,len(y)),dtype=int)
        mps_check=check_mps_forward(model,torch.as_tensor(y[indices],dtype=torch.float32),cfg['x_dim'],batch_size*sample_chunk,
                                    method=method,max_time=float(times[0]))
    if args.multi_gpu: model=nn.DataParallel(model)
    if getattr(args, 'save_samples', False) and len(y)*args.samples*cfg['x_dim'] > 2_000_000:
        raise ValueError('Saving posterior draws is limited to 2 million values; use fewer curves/samples')
    ki_policy=cfg.get('ki_policy','raw')
    outputs={}
    saved_draws=[]
    progress=Progress('Posterior prediction',len(y)*args.samples,'draws')
    for start in range(0,len(y),batch_size):
        check_cancelled()
        obs=torch.as_tensor(y[start:start+batch_size],dtype=torch.float32,device=device)
        draws=[]
        for offset in range(0,args.samples,sample_chunk):
            check_cancelled()
            count=min(sample_chunk,args.samples-offset)
            z=sample(model,obs,count,cfg['x_dim'],times,method=method,sigma_data=sigma_data)
            draws.append(z.cpu().numpy()*xs+xm)
            if not np.isfinite(draws[-1]).all(): raise ValueError('Non-finite posterior draws; check checkpoint and device')
            progress.update(start*args.samples+len(obs)*(offset+count),device=str(device),curves_completed=start)
        if getattr(args, 'save_samples', False):
            saved_draws.append(np.concatenate(draws,axis=1))
        # Retain only one voxel batch of samples in CPU memory for exact medians.
        for key,value in summarize_samples(np.concatenate(draws,axis=1), names, calculate_ki=(names == PARAMETERS and cfg.get('kinetic_model', '2TC') == '2TC'),ki_policy=ki_policy).items():
            outputs.setdefault(key,[]).append(value)
    output.parent.mkdir(parents=True,exist_ok=True)
    if output.exists(): raise FileExistsError(f'Refusing to overwrite {output}')
    metadata={'config':cfg,'checkpoint':str(ckpt),'input':str(args.input),'samples':args.samples,
        'checkpoint_sha256':file_sha256(ckpt),'input_sha256':file_sha256(args.input),
        'credible_interval':'Equal-tailed 95% interval: 2.5th and 97.5th percentiles; width = upper - lower. Ki is transformed per joint draw.',
        'batch_size':args.batch_size,'sample_chunk':args.sample_chunk,'seed':args.seed,
        'effective_batch_size':batch_size,'effective_sample_chunk':sample_chunk,
        'expanded_batch_limit':MPS_MAX_EXPANDED_BATCH if device.type=='mps' else None,
        'mps_numerical_check':mps_check,'lora_weights_merged':cfg.get('training_mode','lora')=='lora',
        'random_noise_device':'cpu' if device.type=='mps' else str(device),
        'posterior_method':method,'sampling_steps':steps,
        'sampling_times':times.tolist(),'sampling_time_units':'radians' if method=='scm' else 'normalized',
        'time_mismatch_allowed':bool(args.allow_time_mismatch), 'time_mismatch_detected':bool(mismatch), 'observed_frame_durations_min':durations.tolist(),'torch_version':str(torch.__version__),
        'device':str(device),'multi_gpu':args.multi_gpu,
        'ki_policy':ki_policy,'training_t_meas':training_times.tolist(),'recorded_time_label_difference':bool(time_label_difference),
        'Ki_policy':(('K1,k2,k3 floored to 1e-4 per joint draw before Ki' if ki_policy=='hpc_floor_1e-4' else 'K1*k3/(k2+k3) for each joint draw; no clipping; |denominator|<=1e-12 excluded') if names == PARAMETERS and cfg.get('kinetic_model', '2TC') == '2TC' else 'Not calculated for this kinetic model'),
        'parameter_order':names,'std_ddof':0}
    if truth is not None:
        means = np.concatenate(outputs['params_mean'])
        metadata['test_rmse'] = np.sqrt(np.mean((means-truth)**2, axis=0)).tolist()
        metadata['test_bias'] = np.mean(means-truth, axis=0).tolist()
    extra = {'posterior_samples':np.concatenate(saved_draws)} if saved_draws else {}
    with output.open('xb') as stream:
        np.savez_compressed(stream, **extra,**{k:np.concatenate(v) for k,v in outputs.items()},
            voxel_columns=columns,parameter_names=np.asarray(names),t_meas=expected,
            observed_t_meas=observed_times,metadata_json=json.dumps(metadata))
    output.with_suffix('.json').write_text(json.dumps(metadata,indent=2)+'\n', encoding='utf-8')
    progress.update(len(y)*args.samples,state='complete',device=str(device),output=str(output))
    print(f'Saved {output}')

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',default='runs/scenario1')
    p.add_argument('--checkpoint',default='best.pth')
    p.add_argument('--input',default='data/real/Sub001_slice75_adjusted.h5')
    p.add_argument('--output',default='runs/scenario1/predictions.npz')
    p.add_argument('--new-output',action='store_true',help='Preserve earlier predictions by choosing an unused numbered output filename')
    p.add_argument('--samples',type=int,default=1000)
    p.add_argument('--batch-size',type=int,default=16)
    p.add_argument('--sample-chunk',type=int,default=64)
    p.add_argument('--save-samples',action='store_true',help='Save joint draws for small curve sets (up to 2 million values)')
    p.add_argument('--steps',type=int,help='Network evaluations per draw (default: CM 3, iMF 1, sCM 2)')
    p.add_argument('--seed',type=int,default=0)
    p.add_argument('--device',default='auto',choices=['auto','cpu','cuda:0','mps'])
    p.add_argument('--multi-gpu',action='store_true')
    p.add_argument('--max-voxels',type=int)
    p.add_argument('--allow-time-mismatch',action='store_true')
    main(p.parse_args())
