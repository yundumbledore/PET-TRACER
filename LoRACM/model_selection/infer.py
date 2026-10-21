"""Run the supplied V8TD P001 demo. No training workflow is included."""
import argparse
from pathlib import Path
import sys
import json
import shutil
import tempfile
import time
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from model_selection.core import ASSETS,MODELS,PARAMETERS,load_demo,load_config,decode,probabilities
from runtime import select_device,Progress,check_cancelled,write_json_atomic
from common import file_sha256


def load_network(device):
    import torch
    from model_selection.network import OneDUnet_v1
    cfg=load_config();settings=cfg['model_settings']
    network=OneDUnet_v1(x_dim=settings['x_dim'],y_dim=settings['y_dim'],
                       embed_dim=settings['embed_dim'],channels=settings['channels'],embedy=settings['embedy'])
    checkpoint=torch.load(ASSETS/'epoch1169.pth',map_location='cpu',weights_only=True)
    weights={k.removeprefix('_orig_mod.'):v for k,v in checkpoint['ema_state_dict'].items()}
    network.load_state_dict(weights,strict=True)
    return network.to(device).eval(),cfg


def sample(network,y,sigmas,generator,device):
    """Original alternating denoising/noise-injection rule, without torch.compile."""
    import torch
    x=torch.randn((len(y),11),generator=generator,dtype=torch.float32).to(device)*float(sigmas[0])
    for i,sigma in enumerate(sigmas):
        if i:x=x+float(sigma)*torch.randn(x.shape,generator=generator,dtype=x.dtype).to(device)
        t=torch.full((len(y),),float(sigma),dtype=torch.float32,device=device)
        x=network(x,y,t)
    return x


def inference_chunks(batch,chunk,device):
    if device.type=='mps':
        batch=min(batch,4096);chunk=min(chunk,max(1,4096//batch))
    return batch,chunk


def check_mps(network,y,batch):
    """Check this flat-vector network against CPU at the actual expanded batch."""
    import copy
    import torch
    with torch.inference_mode():
        cpu=copy.deepcopy(network).to('cpu')
        y=y.cpu().repeat((batch+len(y)-1)//len(y),1)[:batch].contiguous()
        x=torch.randn((batch,11),generator=torch.Generator().manual_seed(1729))
        t=torch.linspace(.25,1.,batch)
        ids=torch.linspace(0,batch-1,min(16,batch)).long()
        expected=cpu(x[ids],y[ids],t[ids])
        actual=network(x.to('mps'),y.to('mps'),t.to('mps')).cpu()[ids]
        error=float((expected-actual).abs().max())
        if not torch.isfinite(actual).all() or not torch.allclose(actual,expected,atol=.002,rtol=.002):
            raise ValueError(f'MPS/CPU check failed (max difference {error:.6g}); select CPU for inference')
    return dict(max_absolute_error=error,expanded_batch=batch,checked_rows=len(ids))


def run(args):
    import torch
    if min(args.samples,args.batch_size,args.sample_chunk,args.steps)<1 or args.samples<2:
        raise ValueError('Use at least two posterior draws and positive batch, chunk and step sizes')
    if args.seed<0 or (args.limit is not None and args.limit<1):raise ValueError('Invalid seed or voxel limit')
    output=Path(args.output).expanduser().resolve()
    if output.exists():raise ValueError('Output folder already exists; choose a new run name')
    started=time.monotonic();device=select_device(args.device)
    if device.type=='cpu':torch.set_num_threads(min(4,torch.get_num_threads()))
    output.parent.mkdir(parents=True,exist_ok=True)
    stage=Path(tempfile.mkdtemp(prefix='.'+output.name+'-',dir=output.parent))
    try:
        network,cfg=load_network(device);data=load_demo()
        count=min(len(data['tac']),args.limit) if args.limit is not None else len(data['tac'])
        required=count*args.samples*11*4
        if shutil.disk_usage(stage).free<required*(2.1 if args.export_npz else 1.05):
            raise ValueError('Not enough free disk space for this posterior array; reduce draws or choose another project folder')
        batch_size,sample_chunk=inference_chunks(args.batch_size,args.sample_chunk,device)
        voxels=np.arange(count,dtype=np.int32)
        config=cfg['scaling_params']
        means={k:torch.as_tensor(config[k],device=device,dtype=torch.float32) for k in config}
        sigmas=np.linspace(cfg['training_settings']['sigma_max'],0.,args.steps,endpoint=False).tolist()
        generator=torch.Generator(device='cpu').manual_seed(args.seed)
        draws=np.lib.format.open_memmap(stage/'posteriors.npy',mode='w+',dtype=np.float32,shape=(count,args.samples,11))
        probs=np.empty((count,7),dtype=np.float64)
        progress=Progress('Bayesian model selection',count,unit='voxels')
        print(f'P001 slice 116: {count:,} voxels, {args.samples} draws, {args.steps} steps, {device}',flush=True)
        mps_check=None
        with torch.inference_mode():
            for lo in range(0,count,batch_size):
                check_cancelled();hi=min(count,lo+batch_size)
                y=np.hstack((data['tac'][lo:hi],np.broadcast_to(data['aif'],(hi-lo,35))))
                y=(torch.as_tensor(y,device=device)-means['y_mean'])/means['y_std']
                if device.type=='mps' and lo==0:mps_check=check_mps(network,y,batch_size*min(sample_chunk,args.samples))
                for first in range(0,args.samples,sample_chunk):
                    check_cancelled();size=min(sample_chunk,args.samples-first)
                    repeated=y[:,None,:].expand(-1,size,-1).reshape(-1,70)
                    raw=sample(network,repeated,sigmas,generator,device)*means['x_std']+means['x_mean']
                    decoded=decode(raw.reshape(hi-lo,size,11).cpu().numpy())
                    draws[lo:hi,first:first+size]=decoded
                probs[lo:hi]=probabilities(draws[lo:hi])
                progress.update(hi,device=str(device))
        draws.flush()
        winners=probs.argmax(1);confidence=probs.max(1)
        np.savez_compressed(stage/'selection.npz',probabilities=probs,map_models=winners,
            confidences=confidence,voxel_indices=voxels,mask=data['mask'],spacing=data['spacing'])
        import pandas as pd
        pd.DataFrame({'Sample_ID':voxels,'MAP_Predicted_Model':winners,'Prediction_Confidence':confidence}).to_csv(stage/'Model_Selection.csv',index=False)
        if args.export_npz:
            check_cancelled();np.savez_compressed(stage/'Posteriors.npz',posteriors=draws)
        metadata=dict(subject='P001',slice_index=116,tracer='FDG',checkpoint_epoch=1169,
            checkpoint_sha256=file_sha256(ASSETS/'epoch1169.pth'),config_sha256=file_sha256(ASSETS/'config.json'),
            input_sha256=file_sha256(ASSETS/'P001_slice116.h5'),mask_sha256=file_sha256(ASSETS/'P001_slice116_mask.npy'),
            models=MODELS,parameter_names=PARAMETERS,draws=args.samples,steps=args.steps,sigmas=sigmas,
            batch_size=batch_size,sample_chunk=sample_chunk,requested_batch_size=args.batch_size,
            requested_sample_chunk=args.sample_chunk,mps_numerical_check=mps_check,seed=args.seed,device=str(device),
            voxel_count=count,full_slice=count==len(data['tac']),elapsed_seconds=time.monotonic()-started,
            sampling='Original V8TD alternating CM denoising and noise injection',
            delay_unit='minutes',torch_version=str(torch.__version__))
        write_json_atomic(stage/'metadata.json',metadata)
        del draws  # Release Windows' memory-map file handle before publishing.
        stage.rename(output)
        progress.update(count,state='completed',device=str(device))
        print('Saved '+str(output),flush=True)
        return output
    except BaseException:
        if 'draws' in locals():del draws
        shutil.rmtree(stage,ignore_errors=True)
        raise


def parser():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',required=True)
    p.add_argument('--samples',type=int,default=200)
    p.add_argument('--steps',type=int,default=3,help='Reverse CM sampling evaluations')
    p.add_argument('--batch-size',type=int,default=16,help='Voxel curves per batch')
    p.add_argument('--sample-chunk',type=int,default=64,help='Posterior draws per curve processed together')
    p.add_argument('--device',default='auto')
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--export-npz',action='store_true',help='Also write the original notebook-compatible Posteriors.npz')
    p.add_argument('--limit',type=int,help='Installation check: process only the first N voxels')
    return p


if __name__=='__main__':
    try:run(parser().parse_args())
    except (OSError,ValueError,RuntimeError) as error:raise SystemExit(str(error)) from error
