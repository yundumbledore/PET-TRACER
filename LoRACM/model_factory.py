"""Build the correct architecture for adapted or scratch-trained inference runs."""
from model import OneDUnet
from model_lora import build_lora_unet
from common import PARAMETERS


def build_model(cfg, device, training=False):
    mode=cfg.get('training_mode','lora')  # Existing release run folders remain readable.
    if mode=='cm':
        model=OneDUnet(cfg['x_dim'],cfg['y_dim'],cfg['embed_dim'],cfg['channels'],cfg['embedy'],cfg['sigma_data']).to(device)
        model.transfer_report={'training_mode':'cm','initialization':'Random; no pretrained checkpoint or LoRA adapters',
                               'foundation_loaded':False,'new_or_resized_parameters':[]}
        return model
    if mode!='lora':
        raise ValueError('Unknown training_mode in model configuration')
    return build_lora_unet(
        x_dim=cfg['x_dim'],y_dim=cfg['y_dim'],embed_dim=cfg['embed_dim'],channels=cfg['channels'],
        embedy=cfg['embedy'],sigma_data=cfg['sigma_data'],r=cfg['lora_r'],alpha=cfg['lora_alpha'],
        dropout=cfg['lora_dropout'],exclude_names=('decodex',),
        pretrained_ckpt=cfg['pretrained'] if training else None,
        train_norms=cfg.get('train_norms',True) if training else False,device=device,
        reset_parameter_heads=training and (cfg.get('parameter_names',PARAMETERS)!=PARAMETERS or cfg.get('kinetic_model','2TC')!='2TC' or cfg.get('reset_parameter_heads',False)))
