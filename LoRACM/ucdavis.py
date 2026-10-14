"""Shared UC Davis defaults for the numbered GUI and terminal walkthrough."""
import argparse
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
PRESETS = ['UC Davis Scenario 1', 'My own project']


def scratch_protocol(protocol,workspace,small=False):
    cfg=dict(protocol)
    cfg.update(n_train=32 if small else 1000000,n_val=16 if small else 100000,n_test=8 if small else 1000,
               output=str(Path(workspace).resolve()/'cm_dataset.npz'),seed=int(protocol['seed'])+1)
    return cfg


def defaults(workspace, preset=PRESETS[0], small=False):
    workspace = Path(workspace).resolve()
    uc = preset == PRESETS[0]
    protocol = json.loads((ROOT/'configs'/('protocol_ucdavis.json' if uc else 'dataset_custom.json')).read_text(encoding='utf-8'))
    protocol['aif_csv'] = str(ROOT/protocol['aif_csv'])
    protocol['output'] = str(workspace/'dataset.npz')
    protocol.update(workflow_preset=preset, walkthrough_size='small' if small and uc else 'full')
    training = json.loads((ROOT/'configs'/('train_scenario1.json' if uc else 'train_custom.json')).read_text(encoding='utf-8'))
    training.update(x_dim='auto',y_dim='auto',parameter_names='auto',loss_weights='auto',
                    training_mode='lora',kinetic_model='2TC',dataset_dir=protocol['output'],
                    pretrained=str(ROOT/'checkpoints/foundation.pth'),save_dir=str(workspace/'runs'),
                    simulation_protocol=str(workspace/'protocol.json'),exp_name='scenario1' if uc else 'adaptation',
                    workflow_preset=preset,walkthrough_size=protocol['walkthrough_size'],checkpoint_start_epoch=1)
    if small and uc:
        protocol.update(n_train=32,n_val=16,n_test=8)
        training.update(epochs=2,batch_size=8,checkpoint_start_epoch=1,device='cpu',exp_name='scenario1_small')
    return protocol, training


def testing_from_dataset(dataset, output):
    output = Path(output)
    if output.exists(): raise FileExistsError('Testing file exists; choose another output')
    with np.load(dataset,allow_pickle=False) as data:
        t=data['t_meas'];y=data['y_test'];count=len(t)
        if 'frame_durations_min' not in data:
            raise ValueError('Use the new walkthrough dataset with explicit frame durations')
        arrays=dict(tac=y[:,:count],aif=y[:,count:],t_meas=t,
                    frame_durations_min=data['frame_durations_min'],truth=data['x_test'][:,:-1])
    output.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(output,**arrays)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workspace',default='workspaces/ucdavis_walkthrough')
    parser.add_argument('--size',choices=['full','small'],default='full')
    parser.add_argument('--with-cm',action='store_true',help='Also prepare a separate, larger scratch-CM dataset protocol and training configuration')
    args=parser.parse_args();workspace=Path(args.workspace).resolve()
    protocol,training=defaults(workspace,small=args.size=='small')
    targets=[workspace/'protocol.json',workspace/'train.json']
    configs=[protocol,training]
    if args.with_cm:
        cm_protocol=scratch_protocol(protocol,workspace,small=args.size=='small')
        cm_training={key:value for key,value in training.items() if key not in ('pretrained','lora_r','lora_alpha','lora_dropout','train_norms')}
        cm_training.update(training_mode='cm',exp_name='cm_scratch',dataset_dir=cm_protocol['output'],simulation_protocol=str(workspace/'cm_protocol.json'))
        targets.extend([workspace/'cm_protocol.json',workspace/'train_cm.json']);configs.extend([cm_protocol,cm_training])
    if any(p.exists() for p in targets): raise FileExistsError('Walkthrough configs exist; choose another workspace')
    workspace.mkdir(parents=True,exist_ok=True)
    for path,cfg in zip(targets,configs):
        path.write_text(json.dumps(cfg,indent=2)+'\n',encoding='utf-8')
    print(f'UC Davis settings prepared: {workspace}. Continue with simulate_protocol.py and train.py.')


if __name__=='__main__': main()
