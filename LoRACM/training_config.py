"""Shared defaults and preflight validation for LoRACM and scratch CM training."""
from pathlib import Path
import math
import re
from common import PARAMETERS
from protocol import validate_names

DEFAULTS = {
    'x_dim':'auto', 'y_dim':'auto', 'parameter_names':'auto', 'loss_weights':'auto',
    'embed_dim':1024, 'channels':[32,64,128,256], 'embedy':'Linear', 'sigma_data':1.,
    'epochs':5000, 'batch_size':512, 'lr':.0001, 'weight_decay':.01,
    'lr_scheduler_patience':25, 'lr_factor':.75, 'min_lr':1e-8,
    'noise_levels':15, 'ema_decay':.99, 'seed':0, 'checkpoint_start_epoch':1, 'checkpoint_interval':25,
    'device':'auto', 'save_dir':'runs', 'drop_last':True,
    'lora_r':8, 'lora_alpha':16, 'lora_dropout':.05, 'train_norms':True,
}


def _integer(cfg, key, minimum=1):
    value=cfg[key]
    if isinstance(value,bool) or not isinstance(value,int) or value < minimum:
        raise ValueError(f'{key} must be an integer >= {minimum}')


def _finite(cfg, key, positive=True):
    value=cfg[key]
    if isinstance(value,bool) or not isinstance(value,(int,float)) or not math.isfinite(value) or (value <= 0 if positive else value < 0):
        raise ValueError(f'{key} must be finite and {"positive" if positive else "nonnegative"}')


def resolve_training_config(config, report, mode='lora'):
    if mode not in ('lora','cm'):
        raise ValueError('training_mode must be lora or cm')
    if config.get('training_mode',mode) != mode:
        raise ValueError('Use train.py for lora or train_cm.py for cm; configuration mode differs')
    cfg=dict(DEFAULTS,**config)
    cfg['training_mode']=mode
    cfg.setdefault('exp_name','adaptation' if mode=='lora' else 'cm_scratch')
    cfg.setdefault('topic','Custom PET protocol')
    for key in ('x_dim','y_dim'):
        if cfg[key] in (None,'auto'):
            cfg[key]=report[key]
        elif isinstance(cfg[key],bool) or not isinstance(cfg[key],int) or cfg[key] != report[key]:
            raise ValueError(f'{key} must match dataset ({report[key]}) or be auto')
    if cfg['parameter_names'] in (None,'auto'):
        cfg['parameter_names']=report['parameter_names']
    if validate_names(cfg['parameter_names']) != report['parameter_names']:
        raise ValueError('Training parameter names/order must match the dataset')
    if cfg['loss_weights'] in (None,'auto'):
        cfg['loss_weights']=[1.]*cfg['x_dim']
    weights=cfg['loss_weights']
    if not isinstance(weights,list) or len(weights)!=cfg['x_dim'] or any(isinstance(w,bool) or not isinstance(w,(int,float)) or not math.isfinite(w) or w<=0 for w in weights):
        raise ValueError('loss_weights requires one finite positive value per parameter, or auto')
    cfg.setdefault('kinetic_model','2TC' if cfg['parameter_names']==PARAMETERS else 'custom')
    if cfg['kinetic_model'] not in ('2TC','custom') or (cfg['kinetic_model']=='2TC' and cfg['parameter_names']!=PARAMETERS):
        raise ValueError('Choose kinetic_model=custom for another parameter set')
    for key in ('epochs','batch_size','embed_dim','checkpoint_start_epoch','checkpoint_interval'):
        _integer(cfg,key)
    if cfg['checkpoint_start_epoch'] != 1:
        cfg['legacy_checkpoint_start_epoch'] = cfg['checkpoint_start_epoch']
        cfg['checkpoint_start_epoch'] = 1
    for key in ('seed','lr_scheduler_patience'):
        _integer(cfg,key,0)
    _integer(cfg,'noise_levels',2)
    if cfg['embed_dim'] % 2:
        raise ValueError('embed_dim must be even for Gaussian Fourier features')
    channels=cfg['channels']
    if not isinstance(channels,list) or len(channels)!=4 or any(isinstance(c,bool) or not isinstance(c,int) or c<1 for c in channels):
        raise ValueError('channels must contain four positive integers')
    if channels[0]%4 or any(c%32 for c in channels[1:]):
        raise ValueError('channels[0] must be divisible by 4; remaining channels by 32 (GroupNorm)')
    if cfg['embedy'] not in ('Linear','MLP','Conv1D','GRU','LSTM'):
        raise ValueError('Supported conditioning embeddings: Linear, MLP, Conv1D, GRU, LSTM')
    for key in ('lr','min_lr','sigma_data'):
        _finite(cfg,key)
    _finite(cfg,'weight_decay',False)
    for key in ('ema_decay','lr_factor'):
        _finite(cfg,key,False)
    if not 0<=cfg['ema_decay']<1 or not 0<cfg['lr_factor']<1 or cfg['min_lr']>cfg['lr']:
        raise ValueError('Require EMA decay in [0,1), scheduler factor in (0,1), and min_lr <= lr')
    if cfg['device'] not in ('auto','cpu','cuda:0','mps'):
        raise ValueError('Choose device auto, cpu, cuda:0 or mps')
    if not isinstance(cfg['drop_last'],bool) or not isinstance(cfg['train_norms'],bool):
        raise ValueError('drop_last and train_norms must be true or false')
    if not isinstance(cfg['exp_name'],str) or not re.fullmatch(r'[A-Za-z0-9_-]+',cfg['exp_name']):
        raise ValueError('exp_name must use letters, numbers, underscores or hyphens')
    if cfg['batch_size']>report['counts']['train']:
        raise ValueError('batch_size exceeds training set; reduce it before training')
    if mode=='lora':
        _integer(cfg,'lora_r')
        _finite(cfg,'lora_alpha')
        _finite(cfg,'lora_dropout',False)
        if cfg['lora_dropout']>=1:
            raise ValueError('lora_dropout must be in [0,1)')
        if not cfg.get('pretrained') or not Path(cfg['pretrained']).is_file():
            raise ValueError('LoRACM requires an existing pretrained foundation checkpoint')
    else:
        if cfg.get('pretrained'):
            raise ValueError('Scratch CM training never loads a foundation; remove pretrained from the config')
        for key in ('pretrained','lora_r','lora_alpha','lora_dropout','train_norms','reset_parameter_heads'):
            cfg.pop(key,None)
    return cfg
