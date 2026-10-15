"""Extract one voxel's TAC/AIF by identity and generate joint posterior samples."""
from protocol import prediction_times
import argparse
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from protocol import read_prediction_input
from infer import main as infer


def inspect(run,input_file,voxel_id,output,samples=1000,checkpoint='best.pth',device='auto',seed=0,steps=3,sample_chunk=64):
    run=Path(run);output=Path(output)
    cfg=json.loads((run/'model_params.json').read_text(encoding='utf-8'))
    scale=json.loads((run/'scaling_params.json').read_text(encoding='utf-8'))
    t,d,y,ids,_=read_prediction_input(input_file,cfg['x_dim'],prediction_times(scale),scale.get('frame_durations_min'))
    matches=np.flatnonzero(ids==str(voxel_id))
    if len(matches)!=1: raise ValueError('Selected voxel ID must appear exactly once in the input data')
    output.parent.mkdir(parents=True,exist_ok=True)
    selected=output.with_suffix('.input.npz')
    if selected.exists() or output.exists(): raise FileExistsError('Voxel output exists; choose another output')
    i=int(matches[0]);np.savez_compressed(selected,t_meas=t,frame_durations_min=d,tac=y[i:i+1,:len(t)],
                                        aif=y[i,len(t):],voxel_columns=ids[i:i+1])
    infer(SimpleNamespace(run=str(run),input=str(selected),output=str(output),checkpoint=checkpoint,
                          samples=samples,batch_size=1,sample_chunk=sample_chunk,steps=steps,seed=seed,device=device,
                          multi_gpu=False,max_voxels=None,allow_time_mismatch=False,save_samples=True))
    return output


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',required=True);p.add_argument('--input',required=True);p.add_argument('--voxel-id',required=True)
    p.add_argument('--output',required=True);p.add_argument('--samples',type=int,default=1000)
    p.add_argument('--checkpoint',default='best.pth');p.add_argument('--device',default='auto',choices=['auto','cpu','cuda:0','mps'])
    p.add_argument('--steps',type=int,default=3);p.add_argument('--sample-chunk',type=int,default=64);p.add_argument('--seed',type=int,default=0)
    a=p.parse_args();inspect(a.run,a.input,a.voxel_id,a.output,a.samples,a.checkpoint,a.device,a.seed,a.steps,a.sample_chunk)
