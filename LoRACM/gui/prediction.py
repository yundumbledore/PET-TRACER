"""Separate model routes, reusable test files and preserved prediction attempts."""
import json
import hashlib
from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st
from gui.state import checkpoint_picker, model_run_picker
from gui import state as project_state
from protocol import read_prediction_input
from ucdavis import testing_from_dataset, PRESETS
from common import next_prediction_output as available_output, file_sha256
from prepare_scan import prepare as prepare_scan


def remember_prediction(workspace,run,output,input_path,mode,geometry=None,geometry_metadata=None):
    project_state.remember('prediction_result:'+mode,str(output))
    project_state.remember('prediction_input:'+mode,str(input_path))
    history=project_state.get('prediction_history:'+mode,[])
    entry={'output':str(output),'input':str(input_path),'run':str(run)}
    if geometry is not None:
        entry.update(geometry=str(geometry),geometry_metadata=str(geometry_metadata))
        for key,value in [('imaging_geometry',geometry),('imaging_metadata',geometry_metadata)]:
            project_state.remember('file:'+key,str(value))
            project_state.remember('6 · Parametric imaging:text_input:'+key+'_'+workspace.name+'_path',str(value))
    project_state.remember('prediction_history:'+mode,[*[item for item in history if item['output']!=str(output)],entry])
    st.session_state['prediction_'+str(workspace)]=str(output)
    st.session_state['prediction_input_'+str(workspace)]=str(input_path)
    st.session_state.active_run=str(run)
    project_state.remember('6 · Parametric imaging:selectbox:Model for parametric imaging',str(run))
    key='imaging_predictions_'+str(run)
    project_state.remember('file:'+key,str(output))
    project_state.remember('6 · Parametric imaging:text_input:'+key+'_'+workspace.name+'_path',str(output))
    project_state.remember('file:imaging_input',str(input_path))
    project_state.remember('6 · Parametric imaging:text_input:imaging_input_'+workspace.name+'_path',str(input_path))


def render(root,workspace,launch,upload_or_path,default_run,preset,small,mode='lora'):
    label='LoRACM' if mode=='lora' else 'CM from scratch'
    st.title('4 · Predict with LoRACM' if mode=='lora' else 'CM · Predict with your scratch model')
    st.caption('Reuse a trained '+label+' model with another patient or slice. Testing times and frame durations must match its training protocol; select a new model or adapt/train for a new protocol.')
    run=model_run_picker(workspace,default_run,mode=mode)
    run=Path(st.text_input('Model run folder',str(run),key='predict_run_'+str(run))).expanduser()
    if not run.is_absolute(): run=root/run
    routes=['UC Davis patient slice','Held-out simulated pairs','My own patient scan','My own file']
    route=st.selectbox('Prediction input',routes,index=0 if preset==PRESETS[0] else 2)
    testing=workspace/('testing.npz' if mode=='lora' else 'cm_testing.npz')
    input_default=str(root/'data/real/Sub001_slice75_adjusted.h5') if route=='UC Davis patient slice' else str(testing)
    if route=='Held-out simulated pairs':
        if st.button('Prepare held-out testing file'):
            try:
                selection=workspace/('dataset_selection.json' if mode=='lora' else 'cm_dataset_selection.json')
                source=json.loads(selection.read_text(encoding='utf-8'))['path'] if selection.exists() else workspace/('dataset.npz' if mode=='lora' else 'cm_dataset.npz')
                testing_from_dataset(source,testing);st.success('Testing file prepared from this model route’s held-out pairs.')
            except Exception as error: st.error(str(error))
        st.caption('These pairs can measure accuracy with known labels. They do not have anatomical geometry for step 6.')
    elif route=='UC Davis patient slice':
        st.caption('Use all 24,237 foreground TACs for the supplied coronal map in step 6.')
    else:
        st.caption('Load another patient, slice or project file. For its parametric images, also supply its own mask, voxel mapping and display metadata in step 6.')
    input_path=upload_or_path('Testing dataset',input_default,['npz','h5','hdf5'],('testing_' if mode=='lora' else 'cm_testing_')+route)
    st.markdown('NPZ: `tac [N,T]`, `aif [T]` or `[N,T]`, `t_meas [T]`, `frame_durations_min [T]`; optional `voxel_columns [N]` and `truth [N,P]`. Legacy pandas HDF5 is also accepted.')
    mask_path=None;mapping_csv=None;geometry=None;geometry_metadata=None
    if route=='My own patient scan':
        st.subheader('Corresponding coronal mask')
        mask_path=upload_or_path('Patient mask (.npy or mask NPZ)',str(workspace/'patient_mask.npy'),['npy','npz'],'patient_mask')
        st.caption('Supply a binary 2D mask of this coronal TAC slice, or choose the corresponding slice from a 3D mask. One input TAC is required for every foreground pixel. NIfTI masks can be exported to NumPy first.')
        volume=st.checkbox('The mask is a 3D volume',value=False)
        axis=1;index=None
        if volume:
            a,b=st.columns(2)
            axis=a.selectbox('Mask slice array axis',[0,1,2],index=1)
            index=b.number_input('Mask slice index',min_value=0,value=0)
        order=st.selectbox('TAC-to-mask mapping',['Foreground pixels in NumPy C order','Explicit mapping CSV'])
        if order=='Explicit mapping CSV': mapping_csv=upload_or_path('Voxel mapping CSV (voxel_columns,voxel_indices)',str(workspace/'voxel_mapping.csv'),['csv'],'patient_mapping')
        a,b=st.columns(2)
        origin=a.selectbox('Image origin',['lower','upper'])
        aspect=b.number_input('Pixel aspect ratio',min_value=.000001,value=1.)
        signature=hashlib.sha256(json.dumps([str(input_path),str(mask_path),str(mapping_csv),axis,index,origin,aspect,
                           [(str(p),p.stat().st_mtime_ns,p.stat().st_size) for p in [input_path,mask_path]+([mapping_csv] if mapping_csv else []) if p.exists()]]).encode()).hexdigest()[:12]
        verified=st.checkbox('I verified the TAC voxel order, mask orientation and pixel aspect ratio',value=False,key='patient_verified_'+signature)
    a,b,c=st.columns(3)
    samples=a.number_input('Posterior samples per curve',min_value=2,value=8 if small else 1000,key='prediction_samples_'+workspace.name)
    batch=b.number_input('Curve batch size',min_value=1,value=64 if small else 16,key='prediction_batch_'+workspace.name)
    chunk=c.number_input('Sample chunk',min_value=1,value=8 if small else 64,key='prediction_chunk_'+workspace.name)
    a,b,c=st.columns(3)
    steps=a.number_input('Reverse sampling steps',min_value=1,value=3,help='Consistency sampling evaluations. More steps do not necessarily improve a consistency model.')
    seed=b.number_input('Inference random seed',min_value=0,value=0)
    device=c.selectbox('Prediction device',['auto','cpu','cuda:0','mps'])
    st.caption('Auto selects CUDA, then Apple MPS, then CPU. Expanded MPS batches are capped and checked against CPU.')
    checkpoint=checkpoint_picker(run)
    output=Path(st.text_input('Prediction output',str(run/'predictions.npz'),key='predict_output_'+str(run))).expanduser()
    if not output.is_absolute(): output=root/output
    st.caption('Each prediction is preserved. If this filename exists, the next result uses _002, _003, and so on.')
    def validate():
        nonlocal geometry,geometry_metadata
        cfg=json.loads((run/'model_params.json').read_text(encoding='utf-8'))
        if cfg.get('training_mode','lora')!=mode: raise ValueError('Select a '+label+' run on this page. Use the other model’s prediction page for its runs.')
        scale=json.loads((run/'scaling_params.json').read_text(encoding='utf-8'))
        payload=read_prediction_input(input_path,cfg['x_dim'],scale['t_meas'],scale.get('frame_durations_min'))
        if mask_path is not None:
            if not verified: raise ValueError('Supply the corresponding mask and confirm the displayed voxel-mapping and display conventions')
            identity=hashlib.sha256(json.dumps([file_sha256(input_path),file_sha256(mask_path),file_sha256(mapping_csv) if mapping_csv else None,
                                               axis,index,origin,aspect]).encode()).hexdigest()[:20]
            folder=workspace/'patient_scans'/identity
            geometry,geometry_metadata=folder/'geometry.npz',folder/'geometry.json'
            if not geometry.exists() or not geometry_metadata.exists():
                attempt=2
                base=folder
                while folder.exists():
                    folder=base.with_name(base.name+f'_{attempt:03d}');attempt+=1
                geometry,geometry_metadata=prepare_scan(input_path,mask_path,folder,cfg['x_dim'],scale['t_meas'],scale.get('frame_durations_min'),
                                                        verified=verified,mapping_csv=mapping_csv,origin=origin,aspect=aspect,axis=axis,index=index)
        elif route=='UC Davis patient slice':
            geometry=root/'data/real/Sub001_slice75_geometry.npz';geometry_metadata=root/'data/real/Sub001_slice75_geometry.json'
        return payload
    if st.button('Validate input'):
        try:
            payload=validate();st.success(f'Accepted {len(payload[2])} curves on the training schedule.')
        except Exception as error: st.error(f'Input rejected: {error}')
    if st.button('Run prediction',type='primary'):
        try:
            if st.session_state.jobs.running(): raise ValueError('Wait for or stop the active job first')
            if not (run/checkpoint).exists(): raise ValueError('Checkpoint missing; finish training or select another checkpoint')
            validate();destination=available_output(output)
            if launch(label+' posterior prediction',['infer.py','--run',run,'--checkpoint',checkpoint,'--input',input_path,'--output',destination,
                    '--samples',samples,'--batch-size',batch,'--sample-chunk',chunk,'--steps',steps,'--seed',seed,'--device',device]):
                remember_prediction(workspace,run,destination,input_path,mode,geometry,geometry_metadata)
                st.info('This prediction will be saved to '+str(destination))
        except Exception as error: st.error(str(error))
    history=project_state.get('prediction_history:'+mode,[])
    candidates=list(dict.fromkeys([entry['output'] for entry in history if entry['run']==str(run) and Path(entry['output']).exists()]+[str(p) for p in sorted(run.glob('predictions*.npz'))]))
    if candidates:
        with st.expander('Open an earlier prediction'):
            chosen=st.selectbox('Saved prediction',candidates,format_func=lambda value:Path(value).name)
            if st.button('Show saved prediction'):
                with np.load(chosen,allow_pickle=False) as data:
                    metadata=json.loads(str(data['metadata_json'])) if 'metadata_json' in data else {}
                previous=next((entry for entry in history if entry['output']==chosen),{})
                remember_prediction(workspace,run,Path(chosen),metadata.get('input',str(input_path)),mode,
                                    previous.get('geometry'),previous.get('geometry_metadata'))
    result=Path(project_state.get('prediction_result:'+mode,str(output)))
    if result.parent!=output.parent: return run
    if result.exists():
        with np.load(result,allow_pickle=False) as data:
            st.subheader('Posterior summaries · '+result.name)
            st.dataframe(pd.DataFrame(data['params_mean'][:100],columns=data['parameter_names'].tolist(),index=data['voxel_columns'][:100]))
            if 'Ki_mean' in data: st.caption('2TC Ki statistics use joint draws. Inspect negative-draw fractions and tails.')
        st.download_button('Download predictions',result.read_bytes(),file_name=result.name)
        if result.with_suffix('.json').exists():
            metadata=json.loads(result.with_suffix('.json').read_text(encoding='utf-8'))
            st.caption('Input: '+str(metadata.get('input','unknown')))
            st.json(metadata)
    elif project_state.get('prediction_result:'+mode):
        st.info('The new prediction has not completed. Its result will appear here; earlier files remain available above.')
    return run
