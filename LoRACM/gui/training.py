"""Shared, concise training controls for adaptation and scratch baselines."""
import json
from pathlib import Path
import numpy as np
import streamlit as st
from protocol import validate_file, load_protocol, validate_protocol
from training_config import resolve_training_config


@st.cache_data(show_spinner=False)
def dataset_report(filename, modified, size):
    return validate_file(filename)


def render(root, workspace, launch, upload_or_path, mode='lora'):
    is_lora=mode=='lora'
    prefix=mode+'_'+workspace.name
    title='Fine-tune with LoRACM' if is_lora else 'Train a CM from scratch'
    st.title(title)
    st.caption('Adapt the foundation model with LoRA and selected trainable layers.' if is_lora else 'Initialize a new consistency model randomly and train all learnable weights.')
    settings_path=workspace/('train.json' if is_lora else 'train_cm.json')
    previous=json.loads(settings_path.read_text(encoding='utf-8')) if settings_path.exists() else {}
    selection=workspace/'dataset_selection.json'
    selected=json.loads(selection.read_text(encoding='utf-8'))['path'] if selection.exists() else previous.get('dataset_dir',str(workspace/'dataset.npz'))
    source=st.text_input('Labelled dataset (.npz)',selected,key=prefix+'_data_'+workspace.name)
    data_path=Path(source).expanduser()
    if not data_path.is_absolute(): data_path=root/data_path
    report=None
    try:
        stat=data_path.stat();report=dataset_report(str(data_path),stat.st_mtime_ns,stat.st_size)
    except Exception as e:
        st.info('Choose a prepared dataset to check its dimensions and enable training.')
        with st.expander('Dataset details'): st.caption(str(e))
    if report:
        a,b,c=st.columns(3)
        a.metric('Training pairs',f"{report['counts']['train']:,}")
        b.metric('Predicted parameters · x_dim',report['x_dim'])
        c.metric('Input values · y_dim',report['y_dim'])
        st.caption('Parameter order: '+', '.join(report['parameter_names'])+'. Dimensions are derived from your data.')
    st.subheader('Training settings')
    a,b,c=st.columns(3)
    epochs=a.number_input('Epochs',min_value=1,value=int(previous.get('epochs',5000)),key=prefix+'_epochs',help='One epoch passes through the training loader. Default: 5,000.')
    batch_default=min(int(previous.get('batch_size',512)),report['counts']['train']) if report else int(previous.get('batch_size',512))
    if prefix+'_batch' not in st.session_state:
        st.session_state[prefix+'_batch']=batch_default
    if report and st.session_state[prefix+'_batch'] is not None and st.session_state[prefix+'_batch']>report['counts']['train']:
        st.session_state[prefix+'_batch']=report['counts']['train']
    batch=b.number_input('Batch size',min_value=1,value=None,key=prefix+'_batch',help='Reduce for limited memory. The initial value is capped by the dataset size.')
    lr=c.number_input('Learning rate',min_value=1e-8,value=float(previous.get('lr',.0001)),step=.00001,format='%.8f',key=prefix+'_lr',help='AdamW starting rate. Default: 0.0001.')
    a,b=st.columns(2)
    devices=['auto','cpu','cuda:0','mps']
    device=a.selectbox('Compute device',devices,index=devices.index(previous.get('device','auto')),key=prefix+'_device')
    name=b.text_input('Run name',previous.get('exp_name','adaptation' if is_lora else 'cm_scratch'),key=prefix+'_name',help='Use a new name for each experiment.')
    if is_lora:
        st.subheader('LoRA settings')
        a,b,c=st.columns(3)
        rank=a.number_input('Rank',min_value=1,value=int(previous.get('lora_r',8)),key=prefix+'_rank',help='Default 8. Controls adapter capacity.')
        alpha=b.number_input('Alpha',min_value=.0001,value=float(previous.get('lora_alpha',16)),step=1.,key=prefix+'_alpha',help='Default 16. Adapter scaling is alpha/rank.')
        dropout=c.number_input('Dropout',min_value=0.,max_value=.99,value=float(previous.get('lora_dropout',.05)),key=prefix+'_dropout')
        with st.expander('Foundation checkpoint',expanded=False):
            checkpoint=upload_or_path('Pretrained foundation (.pth)',previous.get('pretrained','checkpoints/foundation.pth'),['pth'],'training_foundation')
    with st.expander('Advanced settings',expanded=False):
        st.caption('Keep these defaults for a first run. For LoRACM, retaining the foundation architecture preserves more compatible weights.')
        st.markdown('**Architecture**')
        a,b,c=st.columns(3)
        embed=a.number_input('Embedding dimension · embed_dim',min_value=2,value=int(previous.get('embed_dim',1024)),step=128,key=prefix+'_embed',help='Must be even. The bundled foundation uses 1024.')
        embeddings=['Linear','MLP','Conv1D','GRU','LSTM']
        embedy=b.selectbox('Conditioning embedding · embedy',embeddings,index=embeddings.index(previous.get('embedy','Linear')),key=prefix+'_embedy')
        sigma=c.number_input('Boundary scale · sigma_data',min_value=.000001,value=float(previous.get('sigma_data',1.)),key=prefix+'_sigma')
        channels=st.text_input('U-Net channels',', '.join(map(str,previous.get('channels',[32,64,128,256]))),key=prefix+'_channels',help='Four widths. First divisible by 4; remaining widths divisible by 32.')
        st.caption('x_dim and y_dim are defined by the labelled dataset. Change the data schema to change these dimensions; conflicting manual CLI dimensions are rejected.')
        st.markdown('**Optimizer and consistency training**')
        a,b,c=st.columns(3)
        weight_decay=a.number_input('Weight decay',min_value=0.,value=float(previous.get('weight_decay',.01)),key=prefix+'_decay')
        noise_levels=b.number_input('Consistency noise intervals',min_value=2,value=int(previous.get('noise_levels',15)),key=prefix+'_noise')
        ema=c.number_input('EMA decay',min_value=0.,max_value=.99999,value=float(previous.get('ema_decay',.99)),step=.001,format='%.5f',key=prefix+'_ema')
        a,b,c=st.columns(3)
        patience=a.number_input('Scheduler patience (epochs)',min_value=0,value=int(previous.get('lr_scheduler_patience',25)),key=prefix+'_patience')
        factor=b.number_input('Learning-rate reduction factor',min_value=.0001,max_value=.9999,value=float(previous.get('lr_factor',.75)),key=prefix+'_factor')
        min_lr=c.number_input('Minimum learning rate',min_value=1e-12,value=float(previous.get('min_lr',1e-8)),step=1e-8,format='%.12f',key=prefix+'_minlr')
        a,b=st.columns(2)
        seed=a.number_input('Training random seed',min_value=0,value=int(previous.get('seed',0)),key=prefix+'_seed')
        start_epoch=b.number_input('First eligible best-checkpoint epoch',min_value=1,value=int(previous.get('checkpoint_start_epoch',1)),key=prefix+'_eligible')
        weights=previous.get('loss_weights','auto')
        weights_text=st.text_input('Parameter loss weights',weights if isinstance(weights,str) else ', '.join(map(str,weights)),key=prefix+'_weights',help='auto = equal weights. Otherwise one positive value per parameter, in order.')
        drop_last=st.checkbox('Drop an incomplete final training batch',value=previous.get('drop_last',True),key=prefix+'_drop_last')
        train_norms=st.checkbox('Train normalization layers',value=previous.get('train_norms',True),key=prefix+'_norms') if is_lora else True
        st.markdown('**Kinetic model and provenance**')
        protocol_default=workspace/'protocol.json'
        inferred='2TC' if not report or report['parameter_names']==['K1','k2','k3','k4','Vb'] else 'custom'
        if protocol_default.exists(): inferred=json.loads(protocol_default.read_text(encoding='utf-8')).get('model',inferred)
        schema='_'.join(report['parameter_names']) if report else 'pending'
        model_type=st.selectbox('Dataset kinetic model',['2TC','custom'],index=0 if previous.get('kinetic_model',inferred)=='2TC' else 1,key=prefix+'_kinetics_'+schema)
        protocol_source=st.text_input('Forward-model protocol JSON (optional)',previous.get('simulation_protocol',str(protocol_default) if protocol_default.exists() else ''),key=prefix+'_protocol',help='Keeps the same forward model available for the NLS comparison.')

    def configuration():
        if report is None: raise ValueError('Select a valid dataset first')
        cfg=json.loads((root/'configs'/('train_custom.json' if is_lora else 'train_cm.json')).read_text(encoding='utf-8'))
        cfg.pop('simulation_protocol',None)
        cfg.update(dataset_dir=str(data_path),x_dim='auto',y_dim='auto',parameter_names='auto',training_mode=mode,
                   kinetic_model=model_type,epochs=epochs,batch_size=batch,lr=lr,device=device,exp_name=name,
                   embed_dim=embed,embedy=embedy,channels=[int(x.strip()) for x in channels.split(',')],sigma_data=sigma,
                   weight_decay=weight_decay,noise_levels=noise_levels,ema_decay=ema,lr_scheduler_patience=patience,
                   lr_factor=factor,min_lr=min_lr,seed=seed,checkpoint_start_epoch=start_epoch,drop_last=drop_last,
                   loss_weights='auto' if weights_text.strip()=='auto' else [float(x.strip()) for x in weights_text.split(',')],
                   save_dir=str(workspace/'runs'))
        if is_lora: cfg.update(pretrained=str(checkpoint),lora_r=rank,lora_alpha=alpha,lora_dropout=dropout,train_norms=train_norms)
        if protocol_source.strip():
            p=Path(protocol_source).expanduser()
            if not p.is_absolute(): p=root/p
            protocol=load_protocol(p);validate_protocol(protocol)
            if protocol['parameter_names']!=report['parameter_names'] or protocol['model']!=model_type:
                raise ValueError('Forward protocol model/parameter order differs from the dataset')
            for field in ('t_meas','frame_durations_min'):
                actual=report[field]
                if actual is not None and (len(protocol[field])!=len(actual) or not np.allclose(protocol[field],actual,atol=1e-6,rtol=0)):
                    raise ValueError('Forward protocol '+field+' differs from the dataset')
            if protocol.get('observation','midpoint')!=report['observation']:
                raise ValueError('Forward protocol observation mode differs from the dataset')
            cfg['simulation_protocol']=str(p)
        return resolve_training_config(cfg,report,mode)

    a,b=st.columns([1,2])
    if a.button('Review configuration',disabled=report is None,key=prefix+'_review'):
        try:
            cfg=configuration();st.success('Configuration validated.');st.json(cfg)
            st.download_button('Download configuration',json.dumps(cfg,indent=2),file_name=settings_path.name)
        except Exception as e: st.error(str(e))
    if b.button('Start LoRACM fine-tuning' if is_lora else 'Start CM training',type='primary',disabled=report is None,key=prefix+'_start'):
        try:
            if st.session_state.jobs.running(): raise ValueError('Wait for or stop the active job before starting another')
            cfg=configuration();run=workspace/'runs'/cfg['exp_name']
            if run.exists(): raise ValueError('Run already exists; choose another run name')
            settings_path.parent.mkdir(parents=True,exist_ok=True)
            settings_path.write_text(json.dumps(cfg,indent=2)+'\n', encoding='utf-8')
            launch(title,['train.py' if is_lora else 'train_cm.py','--config',settings_path],run=run)
        except Exception as e: st.error(str(e))
    st.caption('Training and validation curves appear below. Both are consistency losses; assess parameter accuracy and calibration with held-out data.')
    return workspace/'runs'/name
