"""Single-curve import, posterior viewing and nonlinear least-squares comparison."""
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


def render(root, workspace, launch):
    st.title('5 · One TAC, full posterior, and an NLS comparison')
    st.write('Load a tissue time–activity curve and its input function. The trained consistency model returns joint posterior samples; nonlinear least squares supplies a bounded point estimate from the same kinetic model.')
    default_run=workspace/'runs'/'adaptation'
    if st.session_state.get('active_run') and workspace in Path(st.session_state.active_run).parents:
        default_run=Path(st.session_state.active_run)
    run = Path(st.text_input('Model run folder',str(default_run))).expanduser()
    if not run.is_absolute(): run=root/run
    source = st.text_input('Single-TAC CSV or prediction NPZ/HDF5',str(workspace/'single_tac.csv'))
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
            payload=read_prediction_input(input_path,cfg['x_dim'],scale['t_meas'],scale.get('frame_durations_min'))
            t,duration,y,columns,_=payload
            selected=st.selectbox('TAC identifier',columns.tolist())
            index=columns.tolist().index(selected)
            st.line_chart(pd.DataFrame({'TAC':y[index,:len(t)],'Input function':y[index,len(t):]},index=t),x_label='Time (minutes)')
    except Exception as e: st.info(f'Choose a compatible run and input: {e}')
    a,b,c=st.columns(3)
    samples=a.number_input('Posterior draws',min_value=2,max_value=100000,value=1000)
    checkpoint=b.selectbox('Single-TAC checkpoint',['best.pth','last.pth'])
    device=c.selectbox('Single-TAC device',['auto','cpu','cuda:0','mps'])
    protocol_default=run/'protocol.json' if (run/'protocol.json').exists() else workspace/'protocol.json'
    use_nls=st.checkbox('Compare with nonlinear least squares',value=protocol_default.exists())
    protocol_source=st.text_input('Kinetic model and fitting bounds (protocol JSON)',str(protocol_default),disabled=not use_nls)
    st.caption('NLS uses uniform prior ranges as parameter bounds, three initialisations and unweighted squared residuals. It assumes a linearly interpolated input, zero input at time zero, and a held final value to cover the final frame. For frame-averaged AIFs this interpolation is an approximation.')
    acknowledged=st.checkbox('These input interpolation assumptions are suitable for this NLS comparison',value=False,disabled=not use_nls)
    if use_nls and Path(protocol_source).exists():
        with st.expander('Review or edit NLS parameter bounds'):
            try:
                nls_cfg=load_protocol(protocol_source)
                bounds_text=st.text_area('NLS kinetic_priors',json.dumps(nls_cfg['kinetic_priors'],indent=2),height=180)
            except Exception as e: st.error(str(e));bounds_text=None
    else: bounds_text=None
    if st.button('Predict posterior and compare',type='primary',disabled=payload is None or cfg is None):
        try:
            t,duration,y,columns,_=payload;index=columns.tolist().index(selected)
            stamp=datetime.now().strftime('%Y%m%d-%H%M%S-%f')
            folder=workspace/'analyses'/stamp;folder.mkdir(parents=True,exist_ok=False)
            chosen=folder/'input.npz'
            np.savez_compressed(chosen,tac=y[index:index+1,:len(t)],aif=y[index,len(t):],t_meas=t,
                                frame_durations_min=duration,voxel_columns=np.asarray([selected]))
            args=['analyse_tac.py','--run',run,'--input',chosen,'--posterior-output',folder/'posterior.npz',
                  '--output',folder/'nls.json','--checkpoint',checkpoint,'--samples',samples,'--device',device]
            if use_nls:
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
            st.session_state['analysis_'+str(workspace)]=str(folder)
            launch('Single-TAC posterior and NLS',args)
        except Exception as e: st.error(str(e))
    choices=sorted((workspace/'analyses').glob('*/posterior.npz')) if (workspace/'analyses').exists() else []
    if choices:
        default=st.session_state.get('analysis_'+str(workspace))
        labels=[str(p.parent) for p in choices]
        result_folder=Path(st.selectbox('Saved analysis',labels,index=labels.index(default) if default in labels else len(labels)-1))
        with np.load(result_folder/'posterior.npz',allow_pickle=False) as data:
            draws=data['posterior_samples'][0];names=data['parameter_names'].tolist()
        nls=json.loads((result_folder/'nls.json').read_text(encoding='utf-8')) if (result_folder/'nls.json').exists() else None
        st.subheader('Kinetic parameter posteriors')
        fig=posterior_figure(draws,names,nls)
        st.pyplot(fig);image_bytes=io.BytesIO();fig.savefig(image_bytes,format='png',dpi=180);plt.close(fig)
        st.download_button('Download posterior figure',image_bytes.getvalue(),file_name='posteriors.png')
        st.download_button('Download joint posterior samples',pd.DataFrame(draws,columns=names).to_csv(index=False),file_name='posterior_samples.csv')
        summary=pd.DataFrame({'Parameter':names,'Posterior mean':draws.mean(axis=0),'Posterior median':np.median(draws,axis=0),
                              '2.5% quantile':np.quantile(draws,.025,axis=0),'97.5% quantile':np.quantile(draws,.975,axis=0)})
        if nls is not None:
            if nls['success']: summary['NLS estimate']=nls['parameters']
            else: st.warning('NLS did not converge; no dashed estimate is displayed. Review bounds and fitting diagnostics.')
            with np.load(result_folder/'input.npz',allow_pickle=False) as data:
                st.line_chart(pd.DataFrame({'Measured TAC':data['tac'][0],'NLS fitted TAC':nls['fitted_tac']},index=data['t_meas']))
            with st.expander('NLS diagnostics'): st.json(nls)
        st.dataframe(summary,hide_index=True)
        st.caption('Dashed lines are NLS point estimates, not ground truth. Posterior quantiles describe the adapted model’s samples; validate calibration and fit quality on independent data before interpretation.')
    return run
