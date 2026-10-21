"""Single-curve import, posterior viewing and nonlinear least-squares comparison."""
from protocol import prediction_times
import json
from pathlib import Path
from datetime import datetime
import hashlib
import io
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from protocol import read_prediction_input, validate_protocol, load_model, load_protocol
from plot_posteriors import posterior_figure
from gui.state import checkpoint_picker, model_run_picker
from gui import state as project_state


@st.cache_data(show_spinner=False)
def prediction_payload(filename,modified,size,dim,times,durations):
    return read_prediction_input(filename,dim,times,durations)


def render(root, workspace, launch, default_run=None, default_input=None,enable_cm=False,mode=None):
    st.title('5 · One TAC, full posterior, and an NLS comparison')
    st.write('Load a tissue time–activity curve and its input function. The trained consistency model returns joint posterior samples; nonlinear least squares supplies a bounded point estimate from the same kinetic model.')
    default_run=default_run or workspace/'runs'/'adaptation'
    if st.session_state.get('active_run') and workspace in Path(st.session_state.active_run).parents:
        default_run=Path(st.session_state.active_run)
    default_run=model_run_picker(workspace,default_run,mode=mode)
    run = Path(st.text_input('Model run folder',str(default_run),key='single_run_'+str(default_run))).expanduser()
    if not run.is_absolute(): run=root/run
    source = st.text_input('Single-TAC CSV or prediction NPZ/HDF5',project_state.get('single_file',str(default_input or workspace/'single_tac.csv')),key='single_source_'+workspace.name)
    upload=st.file_uploader('Choose your curve file',type=['csv','npz','h5','hdf5'])
    st.caption('CSV columns: time_min, duration_min, tac, aif. Activity units and correction convention must match training. For an NPZ/HDF5 with multiple TACs, select one curve below.')
    st.download_button('Download CSV format example','time_min,duration_min,tac,aif\n0.5,1,120,400\n1.5,1,180,250\n2.5,1,200,150\n',file_name='single_tac_format.csv')
    input_path=Path(source).expanduser()
    if not input_path.is_absolute(): input_path=root/input_path
    if upload is not None:
        digest=hashlib.sha256(upload.getbuffer()).hexdigest()[:12]
        input_path=workspace/'inputs'/(digest+'-'+Path(upload.name).name)
        input_path.parent.mkdir(parents=True,exist_ok=True)
        if not input_path.exists(): input_path.write_bytes(upload.getbuffer())
        project_state.remember('single_file',str(input_path))
        project_state.remember('5 · Single TAC & NLS:text_input:single_source_'+workspace.name,str(input_path))
    cfg=None; payload=None; selected=None
    try:
        cfg=json.loads((run/'model_params.json').read_text(encoding='utf-8'))
        cfg.setdefault('parameter_names',['K1','k2','k3','k4','Vb'])
        scale=json.loads((run/'scaling_params.json').read_text(encoding='utf-8'))
        if 'smoke' in cfg.get('exp_name','').lower():
            st.warning('Installation-check model: these posteriors demonstrate the interface and are not scientifically validated estimates.')
        if input_path.exists():
            if input_path.suffix.lower()=='.csv':
                frame=pd.read_csv(input_path)
                needed=['time_min','duration_min','tac','aif']
                if any(k not in frame for k in needed): raise ValueError('CSV needs time_min, duration_min, tac, aif columns')
                buffer=io.BytesIO()
                np.savez_compressed(buffer,t_meas=frame.time_min.to_numpy(),frame_durations_min=frame.duration_min.to_numpy(),
                                    tac=frame.tac.to_numpy()[None,:],aif=frame.aif.to_numpy())
                digest=hashlib.sha256(buffer.getvalue()).hexdigest()[:12]
                converted=workspace/'inputs'/(digest+'-single.npz')
                converted.parent.mkdir(parents=True,exist_ok=True)
                if not converted.exists(): converted.write_bytes(buffer.getvalue())
                input_path=converted
            stat=input_path.stat()
            payload=prediction_payload(str(input_path),stat.st_mtime_ns,stat.st_size,cfg['x_dim'],prediction_times(scale),scale.get('frame_durations_min'))
            t,duration,y,columns,_=payload
            selected=st.selectbox('TAC identifier',columns.tolist())
            index=columns.tolist().index(selected)
            st.line_chart(pd.DataFrame({'TAC':y[index,:len(t)],'Input function':y[index,len(t):]},index=t),x_label='Time (minutes)')
    except Exception as e: st.info(f'Choose a compatible run and input: {e}')
    a,b,c=st.columns(3)
    samples=a.number_input('Posterior draws',min_value=2,max_value=100000,value=1000)
    with b: checkpoint=checkpoint_picker(run,'Single-TAC checkpoint')
    device=c.selectbox('Single-TAC device',['auto','cpu','cuda:0','mps'])
    a,b,c=st.columns(3)
    steps=a.number_input('Reverse sampling steps',min_value=1,value=3)
    chunk=b.number_input('Sample chunk',min_value=1,value=64)
    seed=c.number_input('Inference random seed',min_value=0,value=0)
    comparison_run=None;comparison_checkpoint='best.pth'
    if enable_cm:
        compare=st.checkbox('Overlay CM from scratch posterior',value=False,disabled=bool(cfg and cfg.get('training_mode','lora')!='lora'))
        if compare:
            comparison_run=Path(st.text_input('Scratch CM run',str(workspace/'runs'/'cm_scratch'))).expanduser()
            if not comparison_run.is_absolute(): comparison_run=root/comparison_run
            comparison_checkpoint=checkpoint_picker(comparison_run,'CM comparison checkpoint')
    protocol_default=run/'protocol.json' if (run/'protocol.json').exists() else workspace/'protocol.json'
    use_nls=st.checkbox('Compare with nonlinear least squares',value=protocol_default.exists())
    use_forward=st.checkbox('Overlay forward TAC from posterior median',value=protocol_default.exists())
    protocol_source=st.text_input('Kinetic model and fitting bounds (protocol JSON)',str(protocol_default),disabled=not (use_nls or use_forward))
    st.caption('NLS uses uniform prior ranges as parameter bounds, three initialisations and unweighted squared residuals. It assumes a linearly interpolated input, zero input at time zero, and a held final value to cover the final frame. For frame-averaged AIFs this interpolation is an approximation.')
    acknowledged=st.checkbox('These input interpolation assumptions are suitable for this NLS comparison',value=False,disabled=not (use_nls or use_forward))
    if use_nls and Path(protocol_source).exists():
        with st.expander('Review or edit NLS parameter bounds'):
            try:
                nls_cfg=load_protocol(protocol_source)
                bounds_text=st.text_area('NLS kinetic_priors',json.dumps(nls_cfg['kinetic_priors'],indent=2),height=180)
            except Exception as e: st.error(str(e));bounds_text=None
    else: bounds_text=None
    if st.button('Predict posterior and compare',type='primary',disabled=payload is None or cfg is None):
        try:
            if st.session_state.jobs.running(): raise ValueError('Wait for or stop the active job first')
            t,duration,y,columns,_=payload;index=columns.tolist().index(selected)
            stamp=datetime.now().strftime('%Y%m%d-%H%M%S-%f')
            folder=workspace/'analyses'/stamp;folder.mkdir(parents=True,exist_ok=False)
            chosen=folder/'input.npz'
            np.savez_compressed(chosen,tac=y[index:index+1,:len(t)],aif=y[index,len(t):],t_meas=t,
                                frame_durations_min=duration,voxel_columns=np.asarray([selected]))
            args=['analyse_tac.py','--run',run,'--input',chosen,'--posterior-output',folder/'posterior.npz',
                  '--output',folder/'nls.json','--checkpoint',checkpoint,'--samples',samples,'--device',device,
                  '--steps',steps,'--sample-chunk',chunk,'--seed',seed]
            if comparison_run: args+=['--comparison-run',comparison_run,'--comparison-checkpoint',comparison_checkpoint]
            if not use_nls: args+=['--skip-nls']
            if use_nls or use_forward:
                if not acknowledged: raise ValueError('Confirm the displayed input assumptions or turn off NLS')
                source_path=Path(protocol_source).expanduser()
                if not source_path.is_absolute(): source_path=root/source_path
                protocol=load_protocol(source_path)
                if bounds_text is not None: protocol['kinetic_priors']=json.loads(bounds_text)
                if protocol['parameter_names']!=cfg['parameter_names']: raise ValueError('NLS and CM parameter names/order must match')
                if protocol['model']!=cfg.get('kinetic_model','2TC'): raise ValueError('NLS and CM kinetic models must match')
                validate_protocol(protocol);load_model(protocol)
                nls_protocol=folder/'protocol.json';nls_protocol.write_text(json.dumps(protocol,indent=2)+'\n', encoding='utf-8')
                args+=['--protocol',nls_protocol]
            if launch('Single-TAC posterior and forward comparison',args):
                # Point at the new request immediately; never fall back to an old voxel.
                project_state.remember('current_analysis',str(folder))
        except Exception as e: st.error(str(e))
    choices=sorted((workspace/'analyses').glob('*/analysis.json')) if (workspace/'analyses').exists() else []
    current=project_state.get('current_analysis')
    if choices:
        with st.expander('Open an earlier completed analysis'):
            labels=[str(p.parent) for p in choices]
            historical=st.selectbox('Saved analysis',labels)
            if st.button('Show this saved analysis'):
                project_state.remember('current_analysis',historical);current=historical
    if current:
        result_folder=Path(current)
        if not (result_folder/'analysis.json').exists():
            jobs=st.session_state.jobs
            if jobs.process is not None and str(result_folder/'posterior.npz') in jobs.arguments and jobs.process.poll() is not None:
                st.warning('This analysis did not complete. Review the job log, start a new analysis or explicitly open a saved result.')
            else: st.info('The new voxel analysis is pending. Results will appear here when it finishes.')
            return run
        with np.load(result_folder/'posterior.npz',allow_pickle=False) as data:
            draws=data['posterior_samples'][0];names=data['parameter_names'].tolist()
        nls=json.loads((result_folder/'nls.json').read_text(encoding='utf-8')) if (result_folder/'nls.json').exists() else None
        with np.load(result_folder/'input.npz',allow_pickle=False) as selected_input:
            st.caption('Showing completed analysis for TAC '+str(selected_input['voxel_columns'][0]))
        st.subheader('Kinetic parameter posteriors')
        comparison_draws=None
        if (result_folder/'cm_posterior.npz').exists():
            with np.load(result_folder/'cm_posterior.npz',allow_pickle=False) as baseline: comparison_draws=baseline['posterior_samples'][0]
        fig=posterior_figure(draws,names,nls,comparison_draws)
        st.pyplot(fig);image_bytes=io.BytesIO();fig.savefig(image_bytes,format='png',dpi=180);plt.close(fig)
        st.download_button('Download posterior figure',image_bytes.getvalue(),file_name='posteriors.png')
        st.download_button('Download joint posterior samples',pd.DataFrame(draws,columns=names).to_csv(index=False),file_name='posterior_samples.csv')
        summary=pd.DataFrame({'Parameter':names,'Posterior mean':draws.mean(axis=0),'Posterior median':np.median(draws,axis=0),
                              '2.5% quantile':np.quantile(draws,.025,axis=0),'97.5% quantile':np.quantile(draws,.975,axis=0)})
        if nls is not None:
            if nls['success']: summary['NLS estimate']=nls['parameters']
            else: st.warning('NLS did not converge; no dashed estimate is displayed. Review bounds and fitting diagnostics.')
            with st.expander('NLS diagnostics'): st.json(nls)
        with np.load(result_folder/'input.npz',allow_pickle=False) as data:
            aif=data['aif']
            curves={'Measured TAC':data['tac'][0],'Input function':aif[0] if aif.ndim==2 else aif}
            if nls is not None: curves['NLS fitted TAC']=nls['fitted_tac']
            if (result_folder/'forward.json').exists():
                forward=json.loads((result_folder/'forward.json').read_text(encoding='utf-8'))
                for item in forward['curves']:
                    label='LoRACM' if item['model']=='lora' else 'CM from scratch'
                    if item['fitted_tac'] is not None: curves[label+' median forward TAC']=item['fitted_tac']
                    else: st.warning(label+' median forward curve could not be simulated: '+item.get('error','Invalid median parameters'))
                st.caption('Forward curves use the componentwise posterior medians and the selected kinetic model; parameters are not clipped to fitting bounds.')
            st.subheader('Measured and forward-simulated TACs')
            st.line_chart(pd.DataFrame(curves,index=data['t_meas']),x_label='Time (minutes)',y_label='Activity (input units)')
            st.download_button('Download TAC comparison CSV',pd.DataFrame(dict(time_min=data['t_meas'],**curves)).to_csv(index=False),file_name='tac_comparison.csv')
            st.caption('Saved result for TAC '+str(data['voxel_columns'][0]))
        if comparison_draws is not None: summary['Scratch CM median']=np.median(comparison_draws,axis=0)
        st.dataframe(summary,hide_index=True)
        st.caption('Dashed lines are NLS point estimates, not ground truth. Posterior quantiles describe the selected model’s samples; validate calibration and fit quality on independent data before interpretation.')
    return run
