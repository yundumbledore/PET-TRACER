"""Evaluate LoRACM and a scratch CM on exactly the same held-out input file."""
import argparse
import hashlib
import json
from pathlib import Path
import time
from types import SimpleNamespace
import numpy as np
import pandas as pd
from infer import main as infer


def sha256(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''): digest.update(block)
    return digest.hexdigest()


def main(args):
    configs={label:json.loads((Path(run)/'model_params.json').read_text(encoding='utf-8')) for label,run in [('LoRACM',args.lora_run),('CM',args.cm_run)]}
    if configs['LoRACM'].get('training_mode','lora')!='lora' or configs['CM'].get('training_mode')!='cm':
        raise ValueError('Select a LoRACM run and a CM run trained with train_cm.py')
    names=configs['LoRACM'].get('parameter_names',['K1','k2','k3','k4','Vb'])
    if names!=configs['CM']['parameter_names']:
        raise ValueError('Model parameter names/order differ; cannot compare estimates')
    scales={label:json.loads((Path(run)/'scaling_params.json').read_text(encoding='utf-8')) for label,run in [('LoRACM',args.lora_run),('CM',args.cm_run)]}
    if scales['LoRACM'].get('observation','midpoint')!=scales['CM'].get('observation','midpoint'):
        raise ValueError('Model observation conventions differ')
    if configs['LoRACM'].get('kinetic_model','2TC')!=configs['CM'].get('kinetic_model','2TC'):
        raise ValueError('Model kinetic definitions differ')
    output=Path(args.output)
    output.mkdir(parents=True,exist_ok=False)
    input_digest=sha256(args.input)
    rows=[];details={};predictions={}
    for label,run,checkpoint in [('LoRACM',args.lora_run,args.lora_checkpoint),('CM',args.cm_run,args.cm_checkpoint)]:
        start=time.perf_counter()
        prediction=output/('loracm.npz' if label=='LoRACM' else 'cm.npz')
        infer(SimpleNamespace(run=run,input=args.input,output=str(prediction),checkpoint=checkpoint,
                              samples=args.samples,batch_size=args.batch_size,sample_chunk=args.sample_chunk,
                              steps=args.steps,seed=args.seed,device=args.device,multi_gpu=False,
                              max_voxels=None,allow_time_mismatch=False,save_samples=args.save_samples))
        metadata=json.loads(prediction.with_suffix('.json').read_text(encoding='utf-8'))
        details[label]={'inference_seconds':time.perf_counter()-start,'run':str(run),'checkpoint':checkpoint}
        summary_path=Path(run)/'training_summary.json'
        if summary_path.exists(): details[label]['training']=json.loads(summary_path.read_text(encoding='utf-8'))
        with np.load(prediction,allow_pickle=False) as data:
            predictions[label]=data['voxel_columns'].copy()
            for i,name in enumerate(names):
                row={'model':label,'parameter':name,'mean_posterior_std':float(data['params_std'][:,i].mean()),
                     'negative_draw_fraction':float(data['params_negative_fraction'][:,i].mean())}
                if 'test_rmse' in metadata:
                    row.update(rmse=metadata['test_rmse'][i],bias=metadata['test_bias'][i])
                rows.append(row)
    if not np.array_equal(predictions['LoRACM'],predictions['CM']) or sha256(args.input)!=input_digest:
        raise ValueError('Input curve identities changed during comparison; results are not paired')
    common_keys=['embed_dim','channels','embedy','epochs','lr','batch_size','noise_levels','ema_decay','sigma_data','loss_weights','seed']
    differences={key:{label:configs[label].get(key) for label in configs} for key in common_keys if configs['LoRACM'].get(key)!=configs['CM'].get(key)}
    hashes=[details[label].get('training',{}).get('dataset_sha256') for label in details]
    report={'input':str(args.input),'input_sha256':input_digest,'curves':len(predictions['CM']),
            'parameter_names':names,'samples_per_curve':args.samples,'inference_seed':args.seed,
            'models':details,'training_configuration_differences':differences,
            'same_training_dataset':hashes[0]==hashes[1] if all(hashes) else None,
            'metrics_with_truth':'RMSE and bias of posterior means; uncertainty summaries alone do not measure accuracy',
            'comparison_scope':'Target training data and held-out inputs; foundation pretraining is an additional resource for LoRACM'}
    pd.DataFrame(rows).to_csv(output/'metrics.csv',index=False)
    (output/'comparison.json').write_text(json.dumps(report,indent=2)+'\n', encoding='utf-8')
    print(f'Comparison saved: {output}',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--lora-run',required=True);p.add_argument('--cm-run',required=True)
    p.add_argument('--input',required=True);p.add_argument('--output',required=True)
    p.add_argument('--lora-checkpoint',default='best.pth');p.add_argument('--cm-checkpoint',default='best.pth')
    p.add_argument('--samples',type=int,default=1000);p.add_argument('--batch-size',type=int,default=16)
    p.add_argument('--sample-chunk',type=int,default=64);p.add_argument('--steps',type=int,default=3)
    p.add_argument('--seed',type=int,default=0);p.add_argument('--device',default='auto',choices=['auto','cpu','cuda:0','mps'])
    p.add_argument('--save-samples',action='store_true')
    main(p.parse_args())
