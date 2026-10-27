"""Independent labelled-data preparation for LoRACM and scratch CM."""
import json
from pathlib import Path
import streamlit as st
from protocol import validate_protocol, load_model, load_measured_inputs, validate_file
from runtime import write_json_atomic
from gui import state as project_state
from ucdavis import scratch_protocol
from measurement_noise import noise_settings


def render(root,workspace,launch,upload_or_path,mode='lora',small=False):
    scratch=mode=='cm'
    protocol_path=workspace/('cm_protocol.json' if scratch else 'protocol.json')
    base_protocol=protocol_path if protocol_path.exists() else workspace/'protocol.json'
    dataset_path=workspace/('cm_dataset.npz' if scratch else 'dataset.npz')
    selection=workspace/('cm_dataset_selection.json' if scratch else 'dataset_selection.json')
    st.title('2 · Prepare your CM training dataset' if scratch else '2 · Prepare your fine-tuning dataset')
    if scratch:
        st.caption('Use step 1’s kinetic model, AIFs and timing to simulate training pairs, or upload your own labelled dataset. This module trains a CM from random initialisation.')
        st.info('Starting size: 1,000,000 training pairs, 100,000 validation pairs and 1,000 test pairs. Adjust to your study and memory. The quick installation check uses 32 / 16 / 8. For a comparison at equal data size, you may explicitly import the same NPZ into both routes.')
    route=st.radio('Where will training pairs come from?',['Simulate my protocol','Use my own NPZ dataset'])
    if route=='Simulate my protocol':
        if not base_protocol.exists():
            st.info('Save your model and protocol in step 1 first.')
        else:
            cfg=json.loads(base_protocol.read_text(encoding='utf-8'))
            if scratch and not protocol_path.exists():
                cfg=scratch_protocol(cfg,workspace,small)
            cfg['output']=str(dataset_path)
            a,b,c=st.columns(3)
            n_train=a.number_input('Training pairs',min_value=2,value=int(cfg['n_train']),step=1000,key='n_train_'+workspace.name)
            n_val=b.number_input('Validation pairs',min_value=1,value=int(cfg['n_val']),step=1000,key='n_val_'+workspace.name)
            n_test=c.number_input('Test pairs',min_value=1,value=int(cfg['n_test']),step=100,key='n_test_'+workspace.name)
            saved_noise=noise_settings(cfg)
            noise_model=st.selectbox('Measurement noise',['Log-normal','Gaussian (legacy)'],
                                     index=0 if saved_noise['noise_model']=='lognormal' else 1,
                                     help='Log-normal noise preserves the clean TAC mean and scales variability using actual frame durations. Gaussian is available for older experiments.')
            a,b,c=st.columns(3)
            low=a.number_input('Noise amplitude minimum',min_value=0.,value=float(cfg['noise_prior']['min']))
            high=b.number_input('Noise amplitude maximum',min_value=0.,value=float(cfg['noise_prior']['max']))
            chunk=c.number_input('Simulation chunk size',min_value=1,value=int(cfg['chunk_size']))
            seed=st.number_input('Simulation random seed',min_value=0,value=int(cfg['seed']))
            st.caption('Choose training, validation and test sizes, then set the noise amplitude for your application.')
            if st.button('Generate CM dataset' if scratch else 'Generate dataset',type='primary'):
                try:
                    if st.session_state.jobs.running(): raise ValueError('Wait for or stop the active job first')
                    cfg.update(n_train=n_train,n_val=n_val,n_test=n_test,noise_prior={'type':'uniform','min':low,'max':high},chunk_size=chunk,seed=seed)
                    cfg.update(noise_model='lognormal' if noise_model=='Log-normal' else 'gaussian',
                               noise_timing='frame_duration' if noise_model=='Log-normal' else saved_noise['noise_timing'],
                               noise_epsilon=saved_noise['noise_epsilon'])
                    validate_protocol(cfg);load_model(cfg)
                    if cfg.get('aif_mode','measured')=='measured': load_measured_inputs(cfg)
                    if dataset_path.exists() or dataset_path.with_suffix('.json').exists():
                        raise ValueError('Dataset already exists. Import another file, or move the previous dataset and its provenance aside before regenerating.')
                    write_json_atomic(protocol_path,cfg)
                    launch('CM dataset simulation' if scratch else 'Dataset simulation',['simulate_protocol.py','--config',protocol_path])
                except Exception as error: st.error(str(error))
    else:
        previous=json.loads(selection.read_text(encoding='utf-8'))['path'] if selection.exists() else str(dataset_path)
        dataset_path=upload_or_path('Fine-tuning NPZ file' if not scratch else 'CM training NPZ file',previous,['npz'],'cm_dataset' if scratch else 'dataset')
        st.markdown('Required: `x_train/x_val/x_test` = parameters then noise amplitude; `y_train/y_val/y_test` = TAC then AIF; `t_meas`. Custom models also require `parameter_names`; explicit frame durations are recommended.')
        st.caption('Training needs labelled parameter–curve pairs. Unlabelled clinical TACs are prediction inputs.')
    if st.button('Validate and select CM dataset' if scratch else 'Validate and select dataset'):
        try:
            report=validate_file(dataset_path)
            write_json_atomic(selection,{'path':str(dataset_path),'report':report})
            training_page='CM from scratch' if scratch else '3 · Fine-tune'
            project_state.remember(training_page+':text_input:'+mode+'_'+workspace.name+'_data_'+workspace.name,str(dataset_path))
            st.success('Dataset accepted. Dimensions, parameter order, finite values and training scales checked.')
            st.json(report)
        except Exception as error: st.error(f'Dataset rejected: {error}')
    return dataset_path
