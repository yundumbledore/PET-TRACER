"""Interactive image on the left, identity-matched TAC and posterior on the right."""
import base64
import hashlib
import io
import json
from pathlib import Path
import uuid
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from streamlit.components.v1 import declare_component
from common import file_sha256
from parametric_imaging import load_image,plane,display_limits,map_figure,parameter_samples,density_figure,STATISTICS
from protocol import read_prediction_input
from inspect_voxel import inspect

picker=declare_component('pet_voxel_picker',path=str(Path(__file__).parent/'voxel_picker'))


@st.cache_data(show_spinner=False)
def file_hash(filename,modified,size): return file_sha256(filename)


def checked_hash(path):
    path=Path(path);stat=path.stat();return file_hash(str(path),stat.st_mtime_ns,stat.st_size)


@st.cache_data(show_spinner=False)
def input_curves(filename,modified,size,dim,t,d):
    return read_prediction_input(filename,dim,t,d)


def render(root,workspace,run_default,input_default,upload_or_path):
    st.title('6 · Total-body parametric imaging')
    st.caption('Choose a parameter or its 95% credible-interval width. Click a voxel to inspect its measured curve and posterior.')
    prediction_default=st.session_state.get('prediction_'+str(workspace),str(run_default/'predictions.npz'))
    with st.expander('Prediction, model and geometry files'):
        predictions=upload_or_path('Voxel predictions NPZ',prediction_default,['npz'],'imaging_predictions')
        geometry=upload_or_path('Voxel geometry NPZ',str(root/'data/real/Sub001_slice75_geometry.npz'),['npz'],'imaging_geometry')
        metadata=upload_or_path('Geometry metadata JSON',str(root/'data/real/Sub001_slice75_geometry.json'),['json'],'imaging_metadata')
        run=Path(st.text_input('Imaging model run',str(run_default))).expanduser()
        if not run.is_absolute(): run=root/run
        source=upload_or_path('Imaging TACs and input function',input_default,['npz','h5','hdf5'],'imaging_input')
        draws=st.number_input('Selected-voxel posterior draws',min_value=2,max_value=100000,value=1000)
        device=st.selectbox('Voxel inspection device',['auto','cpu','cuda:0','mps'])
    if not predictions.exists():
        st.info('Finish step 4 using the UC Davis patient slice, then return here. Simulated test pairs have no anatomical voxel geometry.')
        return
    try:
        initial=load_image(predictions,geometry,metadata)
        names=initial['available'];identity=hashlib.sha256((str(predictions.resolve())+str(geometry.resolve())).encode()).hexdigest()[:12]
        a,b=st.columns(2)
        parameter=a.selectbox('Parameter to image',names,key='map_parameter_'+identity)
        statistic=STATISTICS[b.selectbox('Map value',list(STATISTICS),key='map_statistic_'+identity)]
        result=initial if parameter==initial['parameter'] and statistic=='mean' else load_image(predictions,geometry,metadata,parameter,statistic)
    except Exception as e:
        st.info(f'Imaging files need attention: {e}');return
    st.caption('UC Davis provides one complete coronal slice through the body. Custom 2D slices and 3D masks use the same explicit voxel-ID mapping.')
    if 'small' in result['provenance'].get('config',{}).get('exp_name','').lower() or 'smoke' in result['provenance'].get('config',{}).get('exp_name','').lower():
        st.warning('Walkthrough model: the image demonstrates the software and is not a scientifically validated parametric result.')
    axis=1;index=None
    if result['image'].ndim==3:
        a,b=st.columns(2);axis=a.selectbox('Volume slice axis',[0,1,2]);index=b.slider('Slice index',0,result['image'].shape[axis]-1,result['image'].shape[axis]//2)
    values=plane(result['image'],axis,index);ids=plane(result['ids'],axis,index)
    try: low,high=display_limits(values)
    except ValueError as e: st.info(str(e)+'. Choose another parameter or review posterior validity.');return
    with st.expander('Colour scale'):
        a,b=st.columns(2);colourkey=identity+'_'+parameter+'_'+statistic
        low=a.number_input('Colour minimum',value=low,format='%.6g',key=colourkey+'_low')
        high=b.number_input('Colour maximum',value=high,format='%.6g',key=colourkey+'_high')
        st.caption('The initial scale uses the 2nd/98th percentiles. This changes colours only; no filtering or clipping is applied to exported values.')
    if high<=low: st.error('Colour maximum must exceed minimum');return
    left,right=st.columns([.9,1.1],gap='large')
    context=identity+'_'+str(axis)+'_'+str(index)
    with left:
        st.subheader(parameter+(' · 95% CI width' if statistic=='ci95_width' else ' · '+statistic))
        raster=values;click_ids=ids
        if result['metadata'].get('origin','lower')=='lower': raster=np.flipud(raster);click_ids=np.flipud(click_ids)
        image=io.BytesIO();plt.imsave(image,raster,cmap='inferno',vmin=low,vmax=high,format='png',origin='upper')
        colours=[plt.matplotlib.colors.to_hex(c) for c in plt.get_cmap('inferno')(np.linspace(0,1,32))]
        selected=picker(image='data:image/png;base64,'+base64.b64encode(image.getvalue()).decode(),
                        width=raster.shape[1],height=raster.shape[0],aspect=result['metadata'].get('pixel_aspect_ratio',1.),
                        ids=click_ids.ravel().tolist(),colours=colours,low=f'{low:.4g}',high=f'{high:.4g}',
                        context=context,key='voxel_picker_'+context,default=None)
        fig=map_figure(result,axis,index,low,high);buffer=io.BytesIO();fig.savefig(buffer,format='png',dpi=180);plt.close(fig)
        st.download_button('Download map figure',buffer.getvalue(),file_name=f'{parameter}_{statistic}.png')
        buffer=io.BytesIO();np.savez_compressed(buffer,image=result['image'],voxel_ids=result['ids'],mask=result['mask'])
        st.download_button('Download map values',buffer.getvalue(),file_name=f'{parameter}_{statistic}_map.npz')
    with right:
        key='selected_voxel_'+context
        if selected and selected.get('context')==context and selected!=st.session_state.get('last_click_'+context):
            st.session_state['last_click_'+context]=selected
            st.session_state[key]=str(selected['voxel_id'])
        voxel=st.text_input('Voxel ID (or click the image)',key=key)
        if not voxel: st.info('Click a foreground voxel on the left. You can also enter its voxel ID here.');return
        if voxel not in set(ids.ravel()): st.error('That voxel is not in the displayed image plane');return
        try:
            cfg=json.loads((run/'model_params.json').read_text(encoding='utf-8'))
            scale=json.loads((run/'scaling_params.json').read_text(encoding='utf-8'))
            if cfg.get('parameter_names',['K1','k2','k3','k4','Vb'])!=result['parameter_names']:
                raise ValueError('The selected model parameter names differ from this map')
            provenance=result['provenance'];checkpoint=Path(provenance.get('checkpoint','best.pth')).name
            for field,path in [('input_sha256',source),('checkpoint_sha256',run/checkpoint)]:
                if provenance.get(field) and checked_hash(path)!=provenance[field]:
                    raise ValueError('Use the same input data and model checkpoint that produced this map')
            stat=source.stat();t,d,y,column_ids,_=input_curves(str(source),stat.st_mtime_ns,stat.st_size,cfg['x_dim'],scale['t_meas'],scale.get('frame_durations_min'))
            matching=np.flatnonzero(column_ids==voxel)
            if len(matching)!=1: raise ValueError('Voxel identity is absent from the measured TAC file')
            i=int(matching[0]);fig,ax=plt.subplots(figsize=(6,2.7))
            ax.plot(t,y[i,:len(t)],color='#247b84',label='Measured TAC');ax.plot(t,y[i,len(t):],color='#ac744e',label='Input function')
            ax.set(xlabel='Time (minutes)',ylabel='Activity (input units)',title=f'Voxel {voxel}');ax.legend(fontsize=8);fig.tight_layout();st.pyplot(fig);plt.close(fig)
            if st.session_state.jobs.running(): st.info('Voxel sampling will be available when the active job finishes.');return
            seed=int(provenance.get('seed',0));steps=len(provenance.get('sampling_times',[1.,2/3,1/3]))
            digest=hashlib.sha256(json.dumps([checked_hash(run/checkpoint),checked_hash(source),voxel,draws,seed,steps,device]).encode()).hexdigest()[:20]
            cache=workspace/'imaging_voxels'/digest/'posterior.npz'
            marker=cache.parent/'completed.json'
            if marker.exists():
                completed=json.loads(marker.read_text(encoding='utf-8'))['filename']
                if Path(completed).name!=completed: raise ValueError('Invalid voxel-cache filename')
                cache=cache.parent/completed
            if not cache.exists():
                # Failed partial attempts get a new directory; finished draws are reused.
                if cache.with_suffix('.input.npz').exists(): cache=cache.parent/(uuid.uuid4().hex[:8]+'.npz')
                with st.spinner('Sampling the selected voxel posterior…'):
                    inspect(run,source,voxel,cache,draws,checkpoint,device,seed,steps)
                marker.write_text(json.dumps({'filename':cache.name}),encoding='utf-8')
            with np.load(cache,allow_pickle=False) as data:
                samples=parameter_samples(data['posterior_samples'][0],data['parameter_names'].tolist(),parameter)
                fig=density_figure(samples,parameter);st.pyplot(fig);plt.close(fig)
                lo,hi=np.quantile(samples,[.025,.975]);st.caption(f'Median {np.median(samples):.5g} · 95% interval [{lo:.5g}, {hi:.5g}] · width {hi-lo:.5g}')
                st.download_button('Download selected voxel samples',cache.read_bytes(),file_name=f'voxel_{voxel}_posterior.npz')
            st.caption('Fresh joint samples are generated only for the selected voxel and cached. Small Monte Carlo differences from the bulk map are expected. KDE is a display estimate; interval widths use empirical percentiles.')
        except Exception as e: st.info(f'Voxel inspection needs attention: {e}')
