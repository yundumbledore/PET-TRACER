"""Portable end-to-end installation check, including single-TAC NLS and plots."""
import argparse
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pandas as pd
import torch
from simulate_protocol import main as simulate
from protocol import validate_file
from train import main as train
from infer import main as infer
from analyse_tac import fit
from plot_posteriors import posterior_figure
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parent


def main(output, custom=False, from_scratch=False):
    output=Path(output).resolve()
    output.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(2)
    aif=output/'aif.csv'
    pd.DataFrame({'time_min':[0,.2,.5,1,2,3,4,5],
                  'input_1':[0,100,80,50,25,15,10,8],
                  'input_2':[0,80,75,60,30,18,12,10]}).to_csv(aif,index=False)
    cfg=json.loads((ROOT/'configs/dataset_custom.json').read_text(encoding='utf-8'))
    cfg.update(aif_csv=str(aif),output=str(output/'dataset.npz'),n_train=32,n_val=16,n_test=8,
               integration_step_min=.05,noise_prior={'type':'uniform','min':.001,'max':.002},chunk_size=16)
    if custom:
        cfg.update(model='custom',model_file=str(ROOT/'examples/custom_1tc.py'),
                   parameter_names=['K1','k2','Vb'],kinetic_priors={k:cfg['kinetic_priors'][k] for k in ['K1','k2','Vb']})
    protocol=output/'protocol.json';protocol.write_text(json.dumps(cfg,indent=2)+'\n', encoding='utf-8')
    simulate(cfg);report=validate_file(output/'dataset.npz')
    training=json.loads((ROOT/'configs'/('train_cm.json' if from_scratch else 'train_custom.json')).read_text(encoding='utf-8'))
    training.update(dataset_dir=str(output/'dataset.npz'),
                    x_dim=report['x_dim'],y_dim=report['y_dim'],parameter_names=report['parameter_names'],
                    kinetic_model=cfg['model'],simulation_protocol=str(protocol),epochs=1,batch_size=8,
                    device='cpu',save_dir=str(output/'runs'),exp_name='smoke',loss_weights=[1.]*report['x_dim'])
    if not from_scratch:
        training['pretrained']=str(ROOT/'checkpoints/foundation.pth')
    training_file='train_cm.json' if from_scratch else 'train.json'
    (output/training_file).write_text(json.dumps(training,indent=2)+'\n', encoding='utf-8')
    if from_scratch:
        from train_cm import main as train_cm
        train_cm(training)
    else:
        train(training)
    run=output/'runs'/'smoke'
    with np.load(output/'dataset.npz',allow_pickle=False) as data:
        t=data['t_meas'];d=data['frame_durations_min'];y=data['y_test'][:1];n=len(t)
        np.savez_compressed(output/'testing.npz',tac=y[:,:n],aif=y[:,n:],t_meas=t,
                            frame_durations_min=d,truth=data['x_test'][:1,:report['x_dim']])
    infer(SimpleNamespace(run=str(run),input=str(output/'testing.npz'),output=str(output/'posterior.npz'),
                          checkpoint='best.pth',samples=32,batch_size=1,sample_chunk=16,steps=3,seed=0,
                          device='cpu',multi_gpu=False,max_voxels=None,allow_time_mismatch=False,save_samples=True))
    nls=fit(t,d,y[0,:n],y[0,n:],cfg,max_nfev=150)
    (output/'nls.json').write_text(json.dumps(nls,indent=2)+'\n', encoding='utf-8')
    with np.load(output/'posterior.npz',allow_pickle=False) as prediction:
        if prediction['posterior_samples'].shape != (1,32,report['x_dim']): raise AssertionError('Posterior draw contract failed')
        if custom and 'Ki_mean' in prediction: raise AssertionError('Custom kinetic model must not produce 2TC Ki')
        fig=posterior_figure(prediction['posterior_samples'][0],report['parameter_names'],nls)
    fig.savefig(output/'posteriors.png',dpi=140);plt.close(fig)
    if not nls['success']: raise AssertionError('Smoke NLS did not converge')
    print(f'Portable workflow passed: {output}',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',required=True);p.add_argument('--custom-model',action='store_true')
    p.add_argument('--from-scratch',action='store_true')
    a=p.parse_args();main(a.output,a.custom_model,a.from_scratch)
