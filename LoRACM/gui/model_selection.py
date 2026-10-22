"""Browser version of the notebook's linked model-selection and posterior viewer."""
import base64
from datetime import datetime
import io
import json
import re
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb
from scipy.stats import gaussian_kde
import streamlit as st
from streamlit.components.v1 import declare_component
from gui.imaging import picker
from gui.navigation import go,workspace_base,project_folder,WORKFLOW_PAGES,sidebar_logo
from gui import state as project_state
from runtime import write_json_atomic
from model_selection.core import ASSETS,MODELS,COLORS,PARAMETERS,TITLES,ACTIVE,load_demo,load_result,conditional_samples,forward_posterior

model_picker=declare_component('pet_model_probability_picker',path=str(Path(__file__).parent/'model_picker'))
DEFAULTS=dict(samples=200,steps=3,batch_size=16,sample_chunk=64,device='auto',seed=42,export_npz=False)


@st.cache_resource(show_spinner='Loading P001 slice 116…')
def demo_data():
    data=load_demo()
    for value in data.values():
        if isinstance(value,np.ndarray):value.flags.writeable=False
    return data


@st.cache_resource(show_spinner='Opening the posterior arrays…')
def result_data(folder,modified):
    result=load_result(folder)
    data=demo_data()
    if not np.array_equal(result['mask'],data['mask']):raise ValueError('Result mask does not match P001 slice 116')
    ids=result['voxel_indices']
    if ids.ndim!=1 or len(np.unique(ids))!=len(ids) or np.any(ids<0) or np.any(ids>=len(data['tac'])):
        raise ValueError('Invalid voxel identities in result')
    return result


def png(figure):
    buffer=io.BytesIO();figure.savefig(buffer,format='png',dpi=180,bbox_inches='tight',facecolor='white')
    return buffer.getvalue()


@st.fragment(run_every=2)
def job_monitor(jobs,workspace):
    if jobs.process is None or jobs.workspace!=workspace:return
    st.divider();st.subheader('Inference progress')
    code=jobs.process.poll();progress=jobs.progress()
    if code is None:
        st.info('P001 inference is running. You can leave this page and return to the same project.')
        if st.button('Stop inference',disabled=bool(jobs.stop_path and jobs.stop_path.exists())):jobs.stop()
    elif code==0:st.success('Inference completed. Open Explore results in the left menu.')
    elif jobs.stop_path and jobs.stop_path.exists():st.info('Inference stopped. Incomplete results were discarded.')
    else:st.error('Inference did not complete. Review the saved log below.')
    if progress:
        st.progress(min(1.,progress['completed']/max(1,progress['total'])),text=f"{progress['completed']:,} / {progress['total']:,} voxels")
        st.caption(f"Device: {progress.get('device','preparing')} · {progress.get('elapsed_seconds',0):.0f} seconds")
    with st.expander('Inference log',expanded=code not in (None,0)):
        st.text(jobs.tail())
        st.download_button('Download inference log',jobs.full_log(),file_name=jobs.log_path.name)
    if code is not None and st.session_state.get('_selection_finished_pid')!=jobs.process.pid:
        st.session_state['_selection_finished_pid']=jobs.process.pid;st.rerun()


def explore(workspace):
    runs=sorted((p for p in (workspace/'results').glob('*/metadata.json') if not p.parent.name.startswith('.')),reverse=True)
    if not runs:
        st.info('Run P001 inference first. Completed results will appear here automatically.');return
    folder=Path(st.selectbox('Completed inference run',[str(p.parent) for p in runs],format_func=lambda x:Path(x).name))
    metadata=json.loads((folder/'metadata.json').read_text(encoding='utf-8'))
    result=result_data(str(folder),(folder/'selection.npz').stat().st_mtime_ns)
    data=demo_data();ids=result['voxel_indices'];context=str(folder.resolve())
    if not metadata.get('full_slice'):st.info(f'Installation-check result: {len(ids):,} of {len(data["tac"]):,} voxels were processed.')
    st.caption(f"P001 · coronal slice 116 · {metadata['draws']} joint draws per voxel · {metadata['steps']} reverse sampling steps · epoch 1169")
    state_key='_selection_voxel_'+context
    if state_key not in st.session_state:
        coords=data['coordinates'][ids]
        st.session_state[state_key]=int(ids[np.argmin(np.square(coords-np.array(data['mask'].shape)/2).sum(1))])
    opacity=st.checkbox('Opacity shows model confidence',value=True,key='opacity_'+context)
    left,right=st.columns([.9,1.7],gap='large')
    with left:
        st.subheader('Model selection map')
        rgba=np.ones((*data['mask'].shape,4),dtype=float)
        valid=np.zeros(data['mask'].shape,dtype=np.int32)
        coordinates=data['coordinates'][ids];rows,cols=coordinates.T
        colors=np.asarray([to_rgb(c) for c in COLORS])[result['map_models']]
        alpha=result['confidences'][:,None] if opacity else np.ones((len(ids),1))
        rgba[rows,cols,:3]=colors*alpha+(1-alpha)
        valid[rows,cols]=ids+1
        # Notebook imshow(...).invert_yaxis(): original row zero is at bottom.
        from PIL import Image
        b=io.BytesIO();Image.fromarray(np.uint8(np.flipud(rgba)*255)).save(b,format='PNG')
        click=picker(image='data:image/png;base64,'+base64.b64encode(b.getvalue()).decode(),
            width=data['mask'].shape[1],height=data['mask'].shape[0],aspect=float(data['spacing'][1]/data['spacing'][0]),
            ids=np.flipud(valid).ravel().tolist(),context=context,colours=list(COLORS),low='0TC',high='3TC-Dual',key='selection_map_'+context)
        if click and click.get('context')==context and click.get('event')!=st.session_state.get('_selection_last_click'):
            st.session_state[state_key]=int(click['voxel_id'])-1
            st.session_state['_selection_last_click']=click['event']
            yy,xx=data['coordinates'][st.session_state[state_key]]
            st.session_state['sx_'+context]=int(xx);st.session_state['sy_'+context]=int(yy)
        legend=' · '.join(f'{i}: {name}' for i,name in enumerate(MODELS))
        st.caption(legend)
        yy,xx=data['coordinates'][st.session_state[state_key]]
        a,b=st.columns(2)
        x=a.number_input('Voxel x',min_value=0,max_value=data['mask'].shape[1]-1,value=int(xx),key='sx_'+context)
        y=b.number_input('Voxel y',min_value=0,max_value=data['mask'].shape[0]-1,value=int(yy),key='sy_'+context)
        voxel=int(data['ids'][int(y),int(x)])
        if voxel not in ids:
            st.warning('Select a foreground voxel included in this inference run.');return
        st.session_state[state_key]=voxel
        st.download_button('Download model selection CSV',(folder/'Model_Selection.csv').read_bytes(),file_name='P001_slice116_Model_Selection.csv')
        fig,ax=plt.subplots(figsize=(4.5,6.5))
        ax.imshow(rgba,origin='lower',aspect=float(data['spacing'][1]/data['spacing'][0]));ax.axis('off')
        ax.set_title('P001 · Bayesian model selection')
        st.download_button('Download map PNG',png(fig),file_name='P001_slice116_Model_Selection.png');plt.close(fig)
    position=int(np.flatnonzero(ids==voxel)[0]);draws=result['posteriors'][position]
    p=result['probabilities'][position];winner=int(np.argmax(p))
    model_key='_selection_model_'+context
    previous_key='_selection_model_voxel_'+context
    if st.session_state.get(previous_key)!=voxel:
        st.session_state[model_key]=winner;st.session_state[previous_key]=voxel
    with right:
        st.subheader(f'Voxel x={int(x)}, y={int(y)}')
        st.caption(f'MAP model: {MODELS[winner]} · posterior probability {p[winner]:.1%}. Click any model bar to inspect its conditional parameter posterior.')
        top,bottom=st.columns([1.25,1])
        with bottom:
            chosen=model_picker(probabilities=p.tolist(),labels=list(MODELS),colours=list(COLORS),
                selected=int(st.session_state[model_key]),context=f'{context}:{voxel}',key='model_bars_'+context)
            if chosen and chosen.get('context')==f'{context}:{voxel}':st.session_state[model_key]=int(chosen['model'])
        model=int(st.session_state[model_key]);subset=conditional_samples(draws,model)
        simulation=None;error=None
        if len(subset):
            try:simulation=forward_posterior(draws,model,data['times'],data['aif'])
            except ValueError as e:error=str(e)
        with top:
            fig,ax=plt.subplots(figsize=(6,4))
            tac=data['tac'][voxel]
            ax.plot(data['times'],tac,color='#202020',marker='o',markersize=3,label='Observed TAC',zorder=4)
            input_ax=ax.twinx()
            input_ax.plot(data['times'],data['aif'],color='#D93280',linestyle='--',label='Aorta input')
            input_ax.set_ylabel('Aorta activity (Bq/mL)',color='#D93280')
            input_ax.tick_params(axis='y',labelcolor='#D93280');input_ax.spines['top'].set_visible(False)
            if simulation is not None:
                ax.fill_between(data['times'],simulation['lower'],simulation['upper'],color=COLORS[model],alpha=.2,linewidth=0,label='95% credible band',zorder=1)
                ax.plot(data['times'],simulation['mean'],color=COLORS[model],linewidth=2.3,label=MODELS[model]+' posterior mean',zorder=3)
            # Scale the tissue axis to the observed voxel, independently of the AIF and fit.
            low=min(0.,float(tac.min()));high=max(0.,float(tac.max()));pad=.08*max(high-low,1.)
            ax.set_ylim(low-pad,high+pad)
            ax.set(xlabel='Time (min)',ylabel='Activity (Bq/mL)',title='TAC and forward simulation')
            ax.spines[['top','right']].set_visible(False);ax.grid(alpha=.15)
            handles,labels=ax.get_legend_handles_labels();input_handles,input_labels=input_ax.get_legend_handles_labels()
            ax.legend(handles+input_handles,labels+input_labels,fontsize=9)
            fig.tight_layout();st.pyplot(fig);plt.close(fig)
        st.subheader(MODELS[model]+' · kinetic parameter posteriors')
        if not len(subset):st.info('No posterior draws selected this model for this voxel. Choose another model.');return
        st.caption(f'{len(subset)} / {len(draws)} draws select this model. Each joint draw is forward-simulated with its own kinetics and time delay in minutes. The line is the mean curve; shading is the pointwise 95% credible band, without additional observation noise.')
        if error:st.warning(error)
        if simulation is not None and simulation['count']<simulation['total']:
            st.warning(f"{simulation['total']-simulation['count']} / {simulation['total']} selected-model draws produced non-finite curves and were omitted from the curve summary. Posterior samples were not clipped.")
        if simulation is not None and simulation['nonphysical']:
            st.warning(f"{simulation['nonphysical']} simulated draws have kinetic parameters outside physical bounds. Their finite curves are included without clipping.")
        active=list(ACTIVE[model])+[10];ncols=min(4,len(active));nrows=int(np.ceil(len(active)/ncols))
        fig,axes=plt.subplots(nrows,ncols,figsize=(11,2.6*nrows),squeeze=False)
        for ax,param in zip(axes.ravel(),active):
            values=subset[:,param]
            ax.hist(values,bins=min(24,max(3,int(np.sqrt(len(values))))),density=True,color=COLORS[model],alpha=.55,edgecolor='white')
            if len(values)>2 and np.ptp(values)>0:
                try:
                    grid=np.linspace(values.min(),values.max(),180)
                    ax.plot(grid,gaussian_kde(values)(grid),color=COLORS[model],linewidth=1.7)
                except (ValueError,np.linalg.LinAlgError):pass
            ax.axvline(np.median(values),color='#17344D',linestyle='--',linewidth=1.4)
            ax.set_title(TITLES[param],fontsize=15);ax.ticklabel_format(axis='x',style='plain',useOffset=False)
            ax.spines[['top','right']].set_visible(False);ax.set_yticks([])
        for ax in axes.ravel()[len(active):]:ax.set_visible(False)
        fig.tight_layout(pad=1.8);st.pyplot(fig)
        st.download_button('Download parameter posteriors PNG',png(fig),file_name=f'P001_x{x}_y{y}_{MODELS[model]}_posteriors.png');plt.close(fig)
        with st.expander('Parameter summaries and selected-voxel data'):
            summary=pd.DataFrame({'Parameter':[PARAMETERS[i] for i in active],
                'Median':np.median(subset[:,active],axis=0),'2.5%':np.quantile(subset[:,active],.025,axis=0),'97.5%':np.quantile(subset[:,active],.975,axis=0)})
            st.dataframe(summary,hide_index=True)
            st.download_button('Download selected-voxel joint samples',pd.DataFrame(draws,columns=PARAMETERS).to_csv(index=False),file_name=f'P001_x{x}_y{y}_samples.csv')
            st.caption('Full posterior arrays remain on disk: '+str(folder/'posteriors.npy'))


def render(root,jobs):
    with st.sidebar:
        sidebar_logo(root);st.caption('Bayesian Model Selection')
        if st.button('← All modules',key='selection_home'):go('home')
        name=st.text_input('Project name',st.query_params.get('project','my_project'),key='_selection_project_name')
        if not re.fullmatch(r'[A-Za-z0-9_-]+',name):st.error('Use letters, digits, underscores or hyphens.');st.stop()
        st.query_params['project']=name
        workspace=project_folder(workspace_base(root),name,'selection')
        page=st.radio('Model-selection workflow',WORKFLOW_PAGES['selection'],key='selection_page')
        st.caption('Pretrained P001 demonstration. Training will be added in a future release.')
        with st.expander('Project folder'):st.code(str(workspace),language='text')
    settings_file=workspace/'inference_settings.json'
    cfg=dict(DEFAULTS)
    if settings_file.exists():cfg.update(json.loads(settings_file.read_text(encoding='utf-8')))
    st.title(page)
    if page=='1 · P001 demonstration':
        st.write('Infer a joint posterior over seven kinetic models, their kinetic parameters and input-function delay using the supplied V8TD epoch-1169 model.')
        a,b,c=st.columns(3);a.metric('Subject','P001');b.metric('Coronal slice',116);c.metric('Tracer','FDG')
        st.image(str(ASSETS/'preview.png'),width=700)
        st.caption('Reference output from the supplied trained model. New inference runs have Monte Carlo variation.')
        st.dataframe(pd.DataFrame({'Index':range(7),'Candidate model':MODELS}),hide_index=True)
        st.write('Use the left menu to configure sampling, run inference and explore the results. This release includes P001 slice 116 only; training and additional subjects are planned.')
    elif page=='2 · Inference settings':
        project_state.context(workspace,'Selection settings')
        a,b,c=st.columns(3)
        cfg['samples']=a.number_input('Posterior draws per voxel',min_value=2,max_value=100000,value=int(cfg['samples']))
        cfg['steps']=b.number_input('Reverse sampling steps',min_value=1,max_value=100,value=int(cfg['steps']),help='Alternating CM denoising/noise-injection evaluations; the original inference script uses three.')
        cfg['batch_size']=c.number_input('Voxel curve batch size',min_value=1,max_value=10000,value=int(cfg['batch_size']),help='Number of different voxel TACs processed together. Lower values reduce device memory use.')
        a,b,c=st.columns(3)
        cfg['sample_chunk']=a.number_input('Posterior sample chunk',min_value=1,max_value=100000,value=int(cfg['sample_chunk']),help='Draws per voxel processed together. Expanded model batch is approximately curve batch × sample chunk.')
        devices=['auto','cpu','cuda:0','mps']
        cfg['device']=b.selectbox('Inference device',devices,index=devices.index(cfg['device']))
        cfg['seed']=c.number_input('Inference random seed',min_value=0,max_value=2**32-1,value=int(cfg['seed']))
        cfg['export_npz']=st.checkbox('Also export notebook-compatible Posteriors.npz',value=bool(cfg['export_npz']),help='Additional disk space and compression time. The viewer reads memory-mapped posteriors.npy without loading every draw into RAM.')
        st.caption('The supplied checkpoint, 35-frame schedule, normalisation, seven model definitions and delay convention stay fixed. These controls affect inference only.')
        data=demo_data();gib=len(data['tac'])*cfg['samples']*11*4/1024**3
        st.info(f'Full slice: {len(data["tac"]):,} voxels. Raw posterior array: approximately {gib:.2f} GiB on disk. Expanded model batch: {cfg["batch_size"]*min(cfg["sample_chunk"],cfg["samples"]):,} draws.')
        write_json_atomic(settings_file,cfg)
        st.success('Settings are saved automatically for this model-selection project.')
    elif page=='3 · Run inference':
        st.write('Run the supplied pretrained CM on every foreground voxel in P001 coronal slice 116.')
        st.json(cfg)
        if st.button('Run P001 model selection',type='primary',disabled=jobs.running()):
            try:
                stamp=datetime.now().strftime('%Y%m%d-%H%M%S-%f')
                target=workspace/'results'/('P001_slice116_'+stamp)
                args=[root/'model_selection/infer.py','--output',target,'--samples',cfg['samples'],'--steps',cfg['steps'],
                      '--batch-size',cfg['batch_size'],'--sample-chunk',cfg['sample_chunk'],'--device',cfg['device'],'--seed',cfg['seed']]
                if cfg['export_npz']:args.append('--export-npz')
                jobs.start('P001 Bayesian model selection',args,workspace)
                st.success('Inference started. Progress appears below; completed results are preserved in a new folder.')
            except (ValueError,OSError) as e:st.error(str(e))
    else:
        try:explore(workspace)
        except (OSError,ValueError,KeyError) as e:st.error('Cannot open this result: '+str(e))
    st.session_state['_draft_context']=None
    job_monitor(jobs,workspace)
