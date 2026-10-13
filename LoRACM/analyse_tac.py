"""Bounded nonlinear least-squares comparator for a single TAC."""
import argparse
import json
from pathlib import Path
import numpy as np
from scipy.optimize import least_squares
from protocol import validate_protocol, load_model, read_prediction_input, load_protocol
from simulate_protocol import integrate, observe_frames


def predict_curve(parameters, t, durations, aif, cfg, model=None):
    """Match simulation observation mode; midpoint AIF values are interpolated.

    Zero is inserted at the origin; after the last AIF sample its last value is
    held to cover the final frame. For frame-averaged inputs this is an explicit
    approximation to a continuous input, not an exact recovery of that input.
    """
    model = load_model(cfg) if model is None else model
    mode = cfg.get('observation','midpoint')
    end = t[-1]+durations[-1]/2 if mode=='frame_average' else t[-1]
    grid = np.unique(np.r_[np.arange(0,end,cfg.get('integration_step_min',.01)),
                           np.maximum(0,t-durations/2),t,t+durations/2 if mode=='frame_average' else t,end])
    grid = grid[grid <= end]
    inputs = np.interp(grid,np.r_[0,t],np.r_[0,aif])[None,:]
    curve = integrate(model,np.asarray(parameters)[None,:],inputs,grid)
    return observe_frames(curve,grid,t,durations,mode)[0]


def fit(t, durations, tac, aif, cfg, starts=3, max_nfev=500):
    validate_protocol(cfg)
    model = load_model(cfg)
    lower = np.asarray([p['min'] for p in cfg['kinetic_priors'].values()])
    upper = np.asarray([p['max'] for p in cfg['kinetic_priors'].values()])
    scale = max(float(np.max(np.abs(tac))),1e-8)
    rng = np.random.default_rng(0)
    candidates = []
    for i in range(starts):
        guess = (lower+upper)/2 if i==0 else rng.uniform(lower,upper)
        result = least_squares(lambda p:(predict_curve(p,t,durations,aif,cfg,model)-tac)/scale,
                               guess,bounds=(lower,upper),max_nfev=max_nfev,x_scale='jac')
        candidates.append(result)
    successful = [r for r in candidates if r.success]
    result = min(successful or candidates,key=lambda r:np.sum(r.fun**2))
    fitted = predict_curve(result.x,t,durations,aif,cfg,model)
    return {'parameter_names':cfg['parameter_names'],'parameters':result.x.tolist(),
            'fitted_tac':fitted.tolist(),'rmse':float(np.sqrt(np.mean((fitted-tac)**2))),
            'success':bool(result.success),'message':str(result.message),'nfev':int(result.nfev),
            'active_bounds':result.active_mask.tolist(),'bounds':{'lower':lower.tolist(),'upper':upper.tolist()},
            'method':'Bounded unweighted nonlinear least squares; multistart; no posterior uncertainty',
            'input_interpolation':'Linear between supplied midpoints; zero at origin; final AIF held to final frame end'}


def main(args):
    cfg=load_protocol(args.protocol) if args.protocol else None
    output=Path(args.output)
    if cfg is not None:
        validate_protocol(cfg)
        if output.exists(): raise FileExistsError('NLS result exists; choose a new output')
    if args.run:
        from types import SimpleNamespace
        from infer import main as infer
        run=Path(args.run)
        model_cfg=json.loads((run/'model_params.json').read_text(encoding='utf-8'))
        scaling=json.loads((run/'scaling_params.json').read_text(encoding='utf-8'))
        t,duration,y,_,_=read_prediction_input(args.input,model_cfg['x_dim'],scaling['t_meas'],scaling.get('frame_durations_min'))
        if len(y) != 1:
            raise ValueError('Single-TAC analysis requires exactly one curve')
        if cfg is not None:
            if cfg['parameter_names'] != model_cfg.get('parameter_names',['K1','k2','k3','k4','Vb']) or cfg['model'] != model_cfg.get('kinetic_model','2TC'):
                raise ValueError('NLS and CM kinetic model/parameter names must match')
            if cfg.get('observation','midpoint') != scaling.get('observation','midpoint'):
                raise ValueError('NLS observation mode differs from the adapted model')
            read_prediction_input(args.input,len(cfg['parameter_names']),cfg['t_meas'],cfg['frame_durations_min'])
        infer(SimpleNamespace(run=args.run, input=args.input, output=args.posterior_output,
              checkpoint=args.checkpoint,samples=args.samples,batch_size=1,sample_chunk=64,
              steps=3,seed=0,device=args.device,multi_gpu=False,max_voxels=None,
              allow_time_mismatch=False,save_samples=True))
    if cfg is None:
        print('Posterior prediction completed. Supply --protocol for the NLS comparator.',flush=True)
        return
    t,duration,y,_,_=read_prediction_input(args.input,len(cfg['parameter_names']),cfg['t_meas'],cfg['frame_durations_min'])
    if len(y)!=1: raise ValueError('Single-TAC NLS accepts exactly one TAC')
    result=fit(t,duration,y[0,:len(t)],y[0,len(t):],cfg)
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(result,indent=2)+'\n', encoding='utf-8')
    print(f'Saved {output}; converged={result["success"]}, RMSE={result["rmse"]:.6g}',flush=True)
    if not result['success']: raise RuntimeError('NLS did not converge; review diagnostics and bounds')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--protocol',help='Simulation protocol with model, bounds and frame conventions for NLS')
    p.add_argument('--input',required=True);p.add_argument('--output',default='nls.json')
    p.add_argument('--run',help='Adapted model folder; when provided, also save posterior draws')
    p.add_argument('--posterior-output',default='posterior.npz')
    p.add_argument('--checkpoint',default='best.pth')
    p.add_argument('--samples',type=int,default=1000)
    p.add_argument('--device',default='auto',choices=['auto','cpu','cuda:0','mps'])
    a=p.parse_args()
    if not a.run and not a.protocol: p.error('Supply --run, --protocol, or both')
    main(a)
