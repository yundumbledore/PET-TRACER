"""Local browser interface for the documented PET-TRACER workflows."""
import json
from pathlib import Path
import re
import sys
import os
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib
matplotlib.use('Agg')  # Browser rendering must not open native GUI windows.

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from protocol import validate_protocol, validate_file, read_prediction_input, load_model, load_measured_inputs
from gui.jobs import Jobs
from gui import state as project_state
from gui.state import checkpoint_picker, model_run_picker
from runtime import select_device, write_json_atomic
from ucdavis import defaults as workflow_defaults, PRESETS, testing_from_dataset

project_state.install()
st.session_state['_draft_context'] = None
st.set_page_config(page_title='PET-TRACER',page_icon='◈',layout='wide')
st.markdown('''<style>
.stApp {background: #f7fafc;}
h1,h2,h3 {color:#143c50; letter-spacing:-0.025em;}
h1 {font-size:2.25rem !important;} h2 {font-size:1.45rem !important;}
[data-testid="stSidebar"] {background:#eaf2f6;}
.stButton>button, .stDownloadButton>button {border-radius:8px;}
[data-testid="stMetric"] {background:white; border:1px solid #dce7ec; border-radius:10px; padding:14px;}
[data-testid="stExpander"] {border-color:#dce7ec;}
[data-testid="stMainBlockContainer"] {max-width:1280px; padding-top:3rem;}

</style>''',unsafe_allow_html=True)


from gui.navigation import MODULES,WORKFLOW_PAGES,CM_LABELS,go,home,module_overview,workspace_base,project_folder,sidebar_logo
if 'jobs' not in st.session_state:
    st.session_state.jobs=Jobs(ROOT)
module=st.query_params.get('module','home')
if module not in MODULES:
    home(ROOT)
    st.stop()
if module=='selection':
    from gui.model_selection import render as render_model_selection
    render_model_selection(ROOT,st.session_state.jobs)
    st.stop()


def path(value):
    p = Path(value).expanduser()
    return p if p.is_absolute() else ROOT/p


def write_json(target, config):
    write_json_atomic(target,config)


def launch(label, args, run=None):
    try:
        st.session_state.jobs.start(label,args,workspace)
        if run is not None: st.session_state.active_run=str(run)
        st.success(f'{label} started. Progress appears below.')
        return True
    except (ValueError,OSError) as e:
        st.error(str(e))
        return False


def upload_or_path(label, default, suffixes, key):
    remembered = project_state.get('file:'+key, str(default))
    value = st.text_input(label,value=remembered,key=key+'_'+project_name+'_path')
    upload = st.file_uploader('Or select a local file',type=suffixes,key=key+'_'+project_name+'_upload')
    if upload is not None:
        import hashlib
        digest = hashlib.sha256(upload.getbuffer()).hexdigest()[:12]
        name = Path(upload.name).name
        destination = workspace/'inputs'/(digest+'-'+name)
        destination.parent.mkdir(parents=True,exist_ok=True)
        if not destination.exists():
            destination.write_bytes(upload.getbuffer())
        project_state.remember(page+':text_input:'+key+'_'+project_name+'_path',str(destination))
        project_state.remember('file:'+key,str(destination))
        return destination
    project_state.remember('file:'+key,str(path(value)))
    return path(value)


with st.sidebar:
    sidebar_logo(ROOT)
    st.caption(MODULES[module])
    if st.button('← All modules',key='module_home'):go('home')
    project_name = st.text_input('Project name',st.query_params.get('project','my_project'),key='_project_name',help='Saved outside new downloads by default. Use a new name for an independent experiment.')
    if not re.fullmatch(r'[A-Za-z0-9_-]+',project_name):
        st.error('Use letters, numbers, underscores or hyphens.'); st.stop()
    st.query_params['project']=project_name
    workspace = project_folder(workspace_base(ROOT),project_name,module)
    from gui.release import sync_release
    sync_release(ROOT,workspace)
    project_state.context(workspace, 'Project')
    with st.expander('Project defaults',expanded=not (workspace/'protocol.json').exists()):
        previous_defaults={}
        for saved_file in ('protocol.json','train.json'):
            if (workspace/saved_file).exists(): previous_defaults.update(json.loads((workspace/saved_file).read_text(encoding='utf-8')))
        previous_preset=previous_defaults.get('workflow_preset',PRESETS[0])
        locked=any((workspace/name).exists() for name in ('protocol.json','train.json','train_cm.json','dataset_selection.json')) or any(not k.startswith('Project:') for k in st.session_state['_project_drafts_'+str(workspace)])
        preset = st.selectbox('Example setup',PRESETS,index=PRESETS.index(previous_preset) if previous_preset in PRESETS else 0,key='preset_'+project_name,disabled=locked,
            help='Initial values for a new project: the supplied UC Davis study, or generic settings for your own study. All settings can be edited in the workflow.')
        size = st.selectbox('Walkthrough scale',['Full Scenario 1','Quick installation check'],index=1 if previous_defaults.get('walkthrough_size')=='small' else 0,disabled=preset!=PRESETS[0] or locked,key='size_'+project_name,
            help='Full: 100,000 training / 100,000 validation / 100 test pairs and 5,000 epochs. Quick: 32 / 16 / 8 pairs and 2 epochs. These are initial values, not limits.')
        st.caption('Defaults initialise this project. Once you begin configuring it, use a new project name to choose fresh defaults. Edits in any workflow page are retained automatically.')
    small = preset==PRESETS[0] and size=='Quick installation check'
    profile=project_name
    enable_cm=False
    pages=WORKFLOW_PAGES[module]
    # Navigation is session-local; project settings live independently of widgets.
    st.session_state['_draft_context']=None
    if st.session_state.get('workflow_page') not in pages: st.session_state['workflow_page']=pages[0]
    page = st.radio('Your workflow',pages,key='workflow_page',format_func=lambda p:CM_LABELS.get(p,p) if module=='cm' else ('Overview' if p=='Welcome' else p))
    with st.expander('Project folder'):
        st.code(str(workspace),language='text')
    st.markdown('[User manual](https://github.com/yundumbledore/PET-TRACER/blob/main/LoRACM/README.md) · [Legacy framework](https://github.com/yundumbledore/PET-TRACER/blob/main/docs/legacy-framework.md)')

project_state.context(workspace,page)
if 'jobs' not in st.session_state:
    st.session_state.jobs = Jobs(ROOT)
protocol_path = workspace/'protocol.json'
dataset_path = workspace/'dataset.npz'
default_protocol,default_training=workflow_defaults(workspace,preset,small)
run_path = workspace/'runs'/default_training['exp_name']
if (workspace/'train.json').exists():
    run_path=workspace/'runs'/json.loads((workspace/'train.json').read_text(encoding='utf-8'))['exp_name']
if module=='cm':
    cm_settings=workspace/'train_cm.json'
    cm_name=json.loads(cm_settings.read_text(encoding='utf-8'))['exp_name'] if cm_settings.exists() else 'cm_scratch'
    run_path=workspace/'runs'/cm_name
input_default=str(ROOT/'data/real/Sub001_slice75_adjusted.h5') if preset==PRESETS[0] else str(workspace/'testing.npz')
if st.session_state.get('active_run'):
    active_run=Path(st.session_state.active_run)
    if workspace in active_run.parents: run_path=active_run

if page == 'Welcome':
    module_overview(module)

elif page == '1 · Model & protocol':
    st.title('1 · Define kinetics, input functions and timing')
    saved = json.loads(protocol_path.read_text(encoding='utf-8')) if protocol_path.exists() else default_protocol.copy()
    if preset==PRESETS[0] and (not protocol_path.exists() or saved.get('workflow_preset')==PRESETS[0]):
        st.caption('UC Davis defaults: FDG 2TC, four original AIFs from UCDavis_1H35.csv, 35 Sub001 frames over 60 minutes and half-life 109.8 minutes.')
    elif protocol_path.exists():
        st.caption('Using this project’s saved protocol. Choose a fresh project name to start from an unchanged template.')
    model_choice = st.selectbox('Kinetic model',['2TC','custom'],index=0 if saved.get('model')=='2TC' else 1,key='model_'+profile)
    model_file = st.text_input('Custom model Python file',saved.get('model_file','examples/custom_1tc.py'),disabled=model_choice=='2TC')
    st.caption('Custom models run your local Python code. Use only code you trust. The working 1TC template documents the batched rhs/observe interface; multiple input channels are outside this GUI contract.')
    if model_choice == 'custom' and st.button('Load the working 1TC template'):
        saved.update(model='custom',model_file=str(ROOT/'examples/custom_1tc.py'),
                     parameter_names=['K1','k2','Vb'],kinetic_priors={k:saved['kinetic_priors'].get(k,{'type':'uniform','min':.001,'max':1.}) for k in ['K1','k2','Vb']})
        project_state.remember(page+':text_input:Custom model Python file',saved['model_file'])
        project_state.remember(page+':text_input:Parameter names, in order',', '.join(saved['parameter_names']))
        project_state.remember(page+':text_area:Parameter priors (uniform ranges)',json.dumps(saved['kinetic_priors'],indent=2))
        write_json(protocol_path,saved); st.rerun()
    names = st.text_input('Parameter names, in order',', '.join(saved['parameter_names']))
    prior_text = st.text_area('Parameter priors (uniform ranges)',json.dumps(saved['kinetic_priors'],indent=2),height=220,help='Keys must match the parameter names in the same order. Rates are per minute; Vb is a fraction.')
    schedule_default = pd.DataFrame({'time_min':saved.get('prediction_t_meas',saved['t_meas']),'duration_min':saved['frame_durations_min']})
    st.subheader('Time sampling schedule')
    schedule_upload = st.file_uploader('Load a schedule CSV with time_min,duration_min',type=['csv'])
    if schedule_upload is not None:
        schedule_default = pd.read_csv(schedule_upload)
    schedule = st.data_editor(schedule_default,num_rows='dynamic',hide_index=True,key='schedule_'+profile)
    c1,c2 = st.columns(2)
    half_life = c1.number_input('Isotope half-life (minutes)',min_value=.0001,value=float(saved['half_life_min']))
    c1.caption(f'Decay constant: {np.log(2)/half_life:.6g} min⁻¹')
    step = c2.number_input('Integration step (minutes)',min_value=.00001,value=float(saved.get('integration_step_min',.01)),format='%.5f')
    st.caption('Check convergence by reducing the integration step. Use consistent activity units and decay correction for measured AIFs, TACs and training data. One input is used for both plasma and blood in the built-in 2TC.')
    aif_mode = st.selectbox('Input function source',['measured','feng_cohort'],index=0 if saved.get('aif_mode')=='measured' else 1)
    if aif_mode == 'measured':
        aif_source = upload_or_path('Measured AIF CSV',saved.get('aif_csv','data/aif/UCDavis_1H35.csv'),['csv'],'aif')
        st.caption('First column: time in minutes; remaining columns: one nonnegative AIF each. Each simulated sample uses one measured curve. Activities are interpolated to your measurement times when required.')
        try:
            preview = pd.read_csv(aif_source); st.line_chart(preview.set_index(preview.columns[0]))
            st.caption(f'Original AIF samples: {float(preview.iloc[0,0]):.6g}–{float(preview.iloc[-1,0]):.6g} minutes · {len(preview.columns)-1} input functions')
        except (OSError,ValueError,IndexError):
            st.caption('Choose an existing CSV to preview it.')
    else:
        aif_source = upload_or_path('Fitted Feng cohort parameter CSV',saved.get('feng_parameters_csv',str(workspace/'feng.csv')),['csv'],'feng')
        st.write('First fit a cohort below. Simulation resamples whole fitted parameter rows, then adds optional log-space variation. Inspect fit errors and curves before using the cohort.')
        fit_source = upload_or_path('Cohort AIF CSV to fit','data/aif/UCDavis_1H35.csv',['csv'],'fit')
        if st.button('Fit cohort AIFs to Feng'):
            launch('Feng fitting',['fit_feng.py','--input',fit_source,'--output',workspace/'feng.csv'])
        fits = workspace/'feng.fits.csv'
        if fits.exists():
            f = pd.read_csv(fits); st.line_chart(f.set_index('time_min'))
            st.dataframe(pd.read_csv(workspace/'feng.csv'))
    jitter = st.number_input('Feng log-space variation (0 = row resampling only)',min_value=0.,max_value=.5,value=float(saved.get('feng_log_jitter',.05)),disabled=aif_mode=='measured')
    def current_protocol():
        cfg = dict(saved,model=model_choice,model_file=str(path(model_file)),parameter_names=[n.strip() for n in names.split(',')],
                   kinetic_priors=json.loads(prior_text),t_meas=schedule['time_min'].tolist(),frame_durations_min=schedule['duration_min'].tolist(),
                   observation='midpoint',half_life_min=half_life,integration_step_min=step,aif_mode=aif_mode,
                   aif_csv=str(aif_source) if aif_mode=='measured' else saved.get('aif_csv'),
                   feng_parameters_csv=str(aif_source) if aif_mode=='feng_cohort' else saved.get('feng_parameters_csv'),
                   feng_log_jitter=jitter,aif_before_first='hpc_first_sample',aif_after_last='hold_last',output=str(dataset_path),
                   workflow_preset=preset,walkthrough_size=default_protocol['walkthrough_size'],
                   simulation_mode='hpc_euler',random_stream='jax_hpc',aif_sampling='interpolate_to_schedule')
        cfg.pop('prediction_t_meas',None)
        # Retain the supplied UC Davis observation labels only when both the
        # source CSV and patient frame schedule are unchanged.
        if aif_mode=='measured' and aif_source.is_file():
            from common import file_sha256
            reference=ROOT/'data/aif/UCDavis_1H35.csv'
            uc=json.loads((ROOT/'configs/protocol_ucdavis.json').read_text(encoding='utf-8'))
            same_schedule=(len(cfg['t_meas'])==len(uc['prediction_t_meas']) and
                           np.allclose(cfg['t_meas'],uc['prediction_t_meas'],atol=1e-8,rtol=0) and
                           np.allclose(cfg['frame_durations_min'],uc['frame_durations_min'],atol=1e-8,rtol=0))
            if same_schedule and file_sha256(aif_source)==file_sha256(reference):
                cfg.update(t_meas=uc['t_meas'],prediction_t_meas=schedule['time_min'].tolist(),aif_sampling='source_samples')
        cfg.pop('input_alignment',None)
        return cfg
    if aif_mode=='measured':
        try: load_measured_inputs(current_protocol())
        except (OSError,ValueError,KeyError,IndexError) as error: st.warning(str(error))
    if st.button('Save and validate protocol',type='primary'):
        try:
            cfg=current_protocol()
            validate_protocol(cfg); load_model(cfg)
            if aif_mode == 'measured': load_measured_inputs(cfg)
            write_json(protocol_path,cfg)
            st.success('Protocol saved. Continue to Dataset to choose size and create pairs.')
        except Exception as e:
            st.error(f'Protocol needs attention: {e}')

elif page in ('2 · Dataset','CM · Dataset'):
    from gui.dataset import render as render_dataset
    selected_dataset=render_dataset(ROOT,workspace,launch,upload_or_path,mode='cm' if page=='CM · Dataset' else 'lora',small=small)
    if page=='2 · Dataset': dataset_path=selected_dataset

elif page in ('3 · Fine-tune','CM from scratch'):
    from gui.training import render as render_training
    run_path=render_training(ROOT,workspace,launch,upload_or_path,mode='lora' if page=='3 · Fine-tune' else 'cm',defaults=default_training)

elif page in ('4 · Predict','CM · Predict'):
    from gui.prediction import render as render_prediction
    prediction_mode='cm' if page=='CM · Predict' else 'lora'
    default_prediction_run=run_path
    if prediction_mode=='cm':
        saved_cm=workspace/'train_cm.json'
        cm_name=json.loads(saved_cm.read_text(encoding='utf-8'))['exp_name'] if saved_cm.exists() else 'cm_scratch'
        default_prediction_run=workspace/'runs'/cm_name
    run_path=render_prediction(ROOT,workspace,launch,upload_or_path,default_prediction_run,preset,small,mode=prediction_mode)

elif page == '5 · Single TAC & NLS':
    from gui.single_tac import render
    run_path = render(ROOT,workspace,launch,default_run=run_path,default_input=input_default,enable_cm=False,mode=module)

elif page == '6 · Parametric imaging':
    from gui.imaging import render as render_imaging
    render_imaging(ROOT,workspace,run_path,st.session_state.get('prediction_input_'+str(workspace),input_default),upload_or_path,launch=launch,mode=module)

elif page == 'Compare models':
    from gui.comparison import render as render_comparison
    render_comparison(ROOT,workspace,launch,upload_or_path)

elif page == 'User manual':
    st.title(MODULES[module]+' user manual')
    sections={'Overview':'README.md','GUI and project settings':'docs/WORKBENCH.md','Installation':'docs/INSTALLATION.md','Models, inputs and protocols':'docs/MODELS_AND_PROTOCOLS.md',
              'Prepare your dataset':'docs/PREPARE_DATA.md','Fine-tuning':'docs/FINE_TUNING.md','CM from scratch and comparison':'docs/TRAIN_CM.md','Prediction and single-TAC NLS':'docs/PREDICTION.md',
              'Parametric imaging':'docs/PARAMETRIC_IMAGING.md','UC Davis demonstration':'docs/UCDAVIS.md'}
    if module=='cm':
        sections={'CM training workflow':'docs/TRAIN_CM.md',**{k:v for k,v in sections.items() if k not in ('Fine-tuning','UC Davis demonstration')}}
    section=st.selectbox('Manual section',list(sections))
    document = ROOT/sections[section]
    content = document.read_text(encoding='utf-8')
    def manual_link(match):
        label, target = match.group(1), match.group(2)
        if '://' in target or target.startswith('#'): return match.group(0)
        file, _, anchor = target.partition('#')
        from urllib.parse import quote
        relative = (document.parent/file).resolve().relative_to(ROOT.parent)
        return '['+label+'](https://github.com/yundumbledore/PET-TRACER/blob/main/'+quote(relative.as_posix())+('#'+anchor if anchor else '')+')'
    st.markdown(re.sub(r'\[([^\]]+)\]\(([^)]+)\)',manual_link,content))

elif page == 'UC Davis demonstration':
    st.title('UC Davis · Reproduce the paper example')
    st.write('Use the original epoch-2755 LoRACM checkpoint and normalization. The default example preserves the original measured AIF values, three reverse steps, 1,000 posterior draws, the paper’s Ki calculation and 2×2 median-filtered display.')
    st.info('To learn dataset creation and retraining, follow numbered steps 1 → 6 in a fresh UC Davis project. To reproduce the paper model, use the supplied checkpoint here or in step 3.')
    demo_device=st.selectbox('Paper-example compute device',['auto','cpu','cuda:0','mps'])
    st.caption('Auto selects CUDA, Apple MPS, then CPU. Whole-slice CPU inference can take substantially longer.')
    if st.button('Reproduce paper Ki map',type='primary',disabled=st.session_state.jobs.running()):
        try:
            from ucdavis import prepare_paper_run
            from gui.prediction import remember_prediction,available_output
            paper_run=prepare_paper_run(workspace)
            destination=available_output(workspace/'paper_predictions.npz')
            if launch('UC Davis paper reproduction',['reproduce_ucdavis.py','--workspace',workspace,'--device',demo_device],run=paper_run):
                remember_prediction(workspace,paper_run,destination,ROOT/'data/real/Sub001_slice75_adjusted.h5','lora',
                                    ROOT/'data/real/Sub001_slice75_geometry.npz',ROOT/'data/real/Sub001_slice75_geometry.json')
        except Exception as error: st.error(str(error))
    reference=ROOT/'paper/ucdavis/reference_ki.png'
    if reference.exists():
        with st.expander('Archived HPC reference map'):
            st.image(str(reference),caption='Archived Ki summaries, rendered with the original 2×2 median filter and 0–0.06 scale. Fresh posterior sampling has Monte Carlo variation.')
    maps=sorted(workspace.glob('paper_predictions*_Ki.png'))
    if maps:
        st.image(str(maps[-1]),caption='Completed paper-model prediction. Open step 6 for voxel inspection and other parameters.')



@st.fragment(run_every=2)
def monitor():
    jobs = st.session_state.jobs
    if jobs.process is not None and jobs.workspace==workspace:
        st.divider(); st.subheader('Job monitor')
        code = jobs.process.poll()
        if code is None:
            st.info(f'{jobs.label} is running')
            if st.button('Stop active job',disabled=bool(jobs.stop_path and jobs.stop_path.exists())):
                jobs.stop(); st.warning('Stop requested. Training keeps the best and last completed epochs; an incomplete epoch is discarded.')
        elif code == 0:
            if jobs.progress() and jobs.progress().get('state')=='stopped': st.info(f'{jobs.label} stopped; completed checkpoints are available.')
            else: st.success(f'{jobs.label} completed')
        else:
            st.error(f'{jobs.label} stopped with exit code {code}. Review the log below.')
        progress=jobs.progress()
        if progress:
            fraction=progress['completed']/max(1,progress['total'])
            st.progress(min(1.,fraction),text=f"{progress['stage']} · {progress['completed']:,} / {progress['total']:,} {progress['unit']}")
            details=[str(progress[k]) for k in ('phase','device') if k in progress]
            if 'epoch' in progress: details.insert(0,f"Epoch {progress['epoch']} / {progress.get('epochs','')}")
            st.caption(' · '.join(details)+f" · {progress.get('elapsed_seconds',0):.0f} seconds")
            if 'learning_rate' in progress:
                st.caption(f"Learning rate {progress['learning_rate']:.8g} · current training loss {progress.get('train_loss',float('nan')):.6g}"+
                           (f" · current validation loss {progress['val_loss']:.6g}" if 'val_loss' in progress else ''))
        else:
            st.caption('Preparing inputs and model…' if code is None else 'Job finished.')
        with st.expander('Job log',expanded=code is not None and code!=0):
            log_text=jobs.tail();st.text(log_text)
            st.caption('Saved log: '+str(jobs.log_path))
            st.download_button('Download job log',jobs.full_log(),file_name=jobs.log_path.name,key='log_'+str(jobs.log_path))
        if code is not None and st.session_state.get('finished_pid') != jobs.process.pid:
            st.session_state.finished_pid = jobs.process.pid
            st.rerun()
    displayed_run=run_path
    if not (displayed_run/'metrics.csv').exists() and st.session_state.get('active_run') and workspace in Path(st.session_state.active_run).parents:
        displayed_run=Path(st.session_state.active_run)
    metrics_path = displayed_run/'metrics.csv'
    if metrics_path.exists() and page in ('3 · Fine-tune','CM from scratch','UC Davis demonstration'):
        metrics = pd.read_csv(metrics_path)
        if len(metrics):
            st.subheader('Training progress')
            resolved=json.loads((displayed_run/'model_params.json').read_text(encoding='utf-8'))
            epoch=int(metrics.iloc[-1]['epoch'])
            st.progress(min(1.,epoch/resolved['epochs']),text=f"Epoch {epoch:,} / {resolved['epochs']:,} · {resolved.get('training_mode','lora').upper()}")
            a,b,c=st.columns(3)
            a.metric('Training loss',f"{metrics.iloc[-1]['train_loss']:.6g}")
            b.metric('Validation loss',f"{metrics.iloc[-1]['val_loss']:.6g}")
            c.metric('Learning rate',f"{metrics.iloc[-1]['learning_rate']:.8g}")
            loss_tab,lr_tab=st.tabs(['Loss','Learning rate'])
            loss_tab.line_chart(metrics.set_index('epoch')[['train_loss','val_loss']])
            lr_tab.line_chart(metrics.set_index('epoch')[['learning_rate']])
            st.caption(f"Lowest validation loss at epoch {int(metrics.loc[metrics.val_loss.idxmin(),'epoch'])}. Select best.pth or a saved epoch in prediction.")
            with st.expander('Epoch metrics'):
                st.dataframe(metrics.tail(5),hide_index=True)
            st.caption('These are consistency/self-consistency losses; assess posterior accuracy on held-out labelled data separately.')

monitor()
