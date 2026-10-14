"""Local browser interface for the documented PET-TRACER workflows."""
import json
from pathlib import Path
import re
import sys
import os
import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from protocol import validate_protocol, validate_file, read_prediction_input, load_model, load_measured_inputs
from gui.jobs import Jobs
from gui import state as project_state
from gui.state import checkpoint_picker, model_run_picker
from runtime import select_device
from ucdavis import defaults as workflow_defaults, PRESETS, testing_from_dataset

project_state.install()
st.session_state['_draft_context'] = None
st.set_page_config(page_title='PET-TRACER · LoRACM',page_icon='◈',layout='wide')
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


def path(value):
    p = Path(value).expanduser()
    return p if p.is_absolute() else ROOT/p


def write_json(target, config):
    target.parent.mkdir(parents=True,exist_ok=True)
    target.write_text(json.dumps(config,indent=2)+'\n', encoding='utf-8')


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
    st.title('PET-TRACER')
    st.caption('Adapt a consistency model to your PET protocol')
    project_name = st.text_input('Project name',st.query_params.get('project','my_project'),key='_project_name',help='Saved under LoRACM/workspaces/. Use a new name for an independent experiment.')
    if not re.fullmatch(r'[A-Za-z0-9_-]+',project_name):
        st.error('Use letters, numbers, underscores or hyphens.'); st.stop()
    st.query_params['project']=project_name
    workspace = Path(os.environ.get('PET_TRACER_WORKSPACES',str(ROOT/'workspaces')))/project_name
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
            help='Full: 50,000 training / 100,000 validation / 100 test pairs and 5,000 epochs. Quick: 32 / 16 / 8 pairs and 2 epochs. These are initial values, not limits.')
        st.caption('Defaults initialise this project. Once you begin configuring it, use a new project name to choose fresh defaults. Edits in any workflow page are retained automatically.')
    small = preset==PRESETS[0] and size=='Quick installation check'
    profile=project_name
    enable_cm=st.checkbox('Enable CM from scratch comparison',value=False,help='Optional: train a baseline, compare its posterior in step 5, and select either model for imaging.')
    pages=['Welcome','1 · Model & protocol','2 · Dataset','3 · Fine-tune']
    if enable_cm: pages+=['CM from scratch']
    pages+=['4 · Predict','5 · Single TAC & NLS','6 · Parametric imaging']
    if enable_cm: pages+=['Compare models']
    pages+=['UC Davis demonstration','User manual']
    # Navigation is session-local; project settings live independently of widgets.
    st.session_state['_draft_context']=None
    if st.session_state.get('workflow_page') not in pages: st.session_state['workflow_page']=pages[0]
    page = st.radio('Your workflow',pages,key='workflow_page')
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
input_default=str(ROOT/'data/real/Sub001_slice75_adjusted.h5') if preset==PRESETS[0] else str(workspace/'testing.npz')
if st.session_state.get('active_run'):
    active_run=Path(st.session_state.active_run)
    if workspace in active_run.parents: run_path=active_run

if page == 'Welcome':
    st.title('PET kinetic modelling for your study')
    st.write('Estimate kinetic parameter posteriors from tissue curves and input functions. Adapt the foundation model with LoRACM, or train a consistency model from scratch for comparison.')
    st.info('**Start with the UC Davis walkthrough**\n\nFollow steps 1 → 6 with the supplied 2TC model, four AIFs, Sub001 frame schedule and coronal slice. Each step is editable when you are ready to use your own project. Choose Quick installation check for a short practice run.')
    a,b = st.columns(2)
    a.info('**Simulate training pairs**\n\nDefine kinetics and timing → choose measured or Feng AIFs → generate labelled pairs → fine-tune → predict.')
    b.info('**Bring your own training pairs**\n\nOpen Dataset → validate your NPZ → fine-tune → predict. You can skip simulation.')
    with st.container(border=True):
        st.markdown('**Two training routes, one prediction workflow**')
        st.write('LoRACM transfers pretrained weights. CM from scratch learns from random initialization. Use the same dataset and held-out curves to compare them.')
    st.subheader('Start with a small check')
    st.write('UC Davis uses Scenario 1 priors and rank 4 / alpha 8. My own project keeps defaults of 100,000 training pairs, rank 8 and alpha 16. Dataset size counts training pairs; validation and testing are additional.')
    st.caption('This application runs locally. Computation uses your Python environment and hardware. Stopping the application can interrupt jobs; retain project configs and run outputs.')

elif page == '1 · Model & protocol':
    st.title('1 · Define kinetics, input functions and timing')
    saved = json.loads(protocol_path.read_text(encoding='utf-8')) if protocol_path.exists() else default_protocol.copy()
    if preset==PRESETS[0] and (not protocol_path.exists() or saved.get('workflow_preset')==PRESETS[0]):
        st.caption('UC Davis defaults: FDG 2TC, four original AIFs from UCDavis_1H35.csv, 35 Sub001 frames over 60 minutes and half-life 109.8 minutes. The simulator uses the original AIF time points with linear interpolation, zero at the origin and the last value held where needed.')
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
    schedule_default = pd.DataFrame({'time_min':saved['t_meas'],'duration_min':saved['frame_durations_min']})
    st.subheader('Time sampling schedule')
    schedule_upload = st.file_uploader('Load a schedule CSV with time_min,duration_min',type=['csv'])
    if schedule_upload is not None:
        schedule_default = pd.read_csv(schedule_upload)
    schedule = st.data_editor(schedule_default,num_rows='dynamic',hide_index=True,key='schedule_'+profile)
    observation = st.selectbox('How are your measured curves sampled?',['midpoint','frame_average'],index=0 if saved.get('observation')=='midpoint' else 1,
                               help='Frame averages integrate both TAC and AIF over each frame. Midpoint samples evaluate at the supplied times.')
    c1,c2 = st.columns(2)
    half_life = c1.number_input('Isotope half-life (minutes)',min_value=.0001,value=float(saved['half_life_min']))
    c1.caption(f'Decay constant: {np.log(2)/half_life:.6g} min⁻¹')
    step = c2.number_input('Integration step (minutes)',min_value=.00001,value=float(saved.get('integration_step_min',.01)),format='%.5f')
    st.caption('Check convergence by reducing the integration step. Use consistent activity units and decay correction for measured AIFs, TACs and training data. One input is used for both plasma and blood in the built-in 2TC.')
    aif_mode = st.selectbox('Input function source',['measured','feng_cohort'],index=0 if saved.get('aif_mode')=='measured' else 1)
    if aif_mode == 'measured':
        aif_source = upload_or_path('Measured AIF CSV',saved.get('aif_csv','data/aif/UCDavis_1H35.csv'),['csv'],'aif')
        st.caption('First column: time in minutes; remaining columns: one nonnegative AIF each. Each simulated sample uses one measured curve. Original samples stay unchanged; interpolation and the endpoint assumptions below define the simulation input.')
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
    origin = st.checkbox('If the first AIF sample is after zero, assume zero activity at time zero',value=saved.get('aif_before_first')=='zero_at_origin',disabled=aif_mode!='measured')
    tail_options = ['require_coverage', 'hold_last']
    tail_default = saved.get('aif_after_last',default_protocol.get('aif_after_last','require_coverage'))
    tail = st.selectbox('After the final AIF sample',tail_options,index=tail_options.index(tail_default) if tail_default in tail_options else 0,
                        format_func=lambda value: {'require_coverage':'Require measured AIF coverage', 'hold_last':'Hold the last measured AIF value'}[value],
                        disabled=aif_mode!='measured',help='Midpoint observations need coverage through the last midpoint; frame averages need coverage through the last frame end. Holding the last value is an explicit approximation, saved in the protocol.')
    if aif_mode == 'measured':
        try:
            load_measured_inputs(dict(saved,aif_csv=str(aif_source),t_meas=schedule['time_min'].tolist(),
                                      frame_durations_min=schedule['duration_min'].tolist(),observation=observation,
                                      aif_before_first='zero_at_origin' if origin else 'require_zero',aif_after_last=tail))
        except (OSError,ValueError,KeyError,IndexError) as e:
            st.warning(str(e))
    if st.button('Save and validate protocol',type='primary'):
        try:
            cfg = dict(saved,model=model_choice,model_file=str(path(model_file)),parameter_names=[n.strip() for n in names.split(',')],
                       kinetic_priors=json.loads(prior_text),t_meas=schedule['time_min'].tolist(),frame_durations_min=schedule['duration_min'].tolist(),
                       observation=observation,half_life_min=half_life,integration_step_min=step,aif_mode=aif_mode,
                       aif_csv=str(aif_source) if aif_mode=='measured' else saved.get('aif_csv'),
                       feng_parameters_csv=str(aif_source) if aif_mode=='feng_cohort' else saved.get('feng_parameters_csv'),
                       feng_log_jitter=jitter,aif_before_first='zero_at_origin' if origin else 'require_zero',aif_after_last=tail,output=str(dataset_path),
                       workflow_preset=preset,walkthrough_size=default_protocol['walkthrough_size'])
            cfg.pop('input_alignment',None)
            validate_protocol(cfg); load_model(cfg)
            if aif_mode == 'measured': load_measured_inputs(cfg)
            write_json(protocol_path,cfg)
            st.success('Protocol saved. Continue to Dataset to choose size and create pairs.')
        except Exception as e:
            st.error(f'Protocol needs attention: {e}')

elif page == '2 · Dataset':
    st.title('2 · Prepare your fine-tuning dataset')
    route = st.radio('Where will training pairs come from?',['Simulate my protocol','Use my own NPZ dataset'])
    if route == 'Simulate my protocol':
        if not protocol_path.exists():
            st.info('Save your model and protocol in step 1 first.')
        else:
            cfg = json.loads(protocol_path.read_text(encoding='utf-8'))
            a,b,c = st.columns(3)
            n_train = a.number_input('Training pairs',min_value=2,value=int(cfg['n_train']),step=1000,key='n_train_'+profile)
            n_val = b.number_input('Validation pairs',min_value=1,value=int(cfg['n_val']),step=1000,key='n_val_'+profile)
            n_test = c.number_input('Test pairs',min_value=1,value=int(cfg['n_test']),step=100,key='n_test_'+profile)
            a,b,c = st.columns(3)
            low = a.number_input('Noise amplitude minimum',min_value=0.,value=float(cfg['noise_prior']['min']))
            high = b.number_input('Noise amplitude maximum',min_value=0.,value=float(cfg['noise_prior']['max']))
            chunk = c.number_input('Simulation chunk size',min_value=1,value=int(cfg['chunk_size']))
            seed = st.number_input('Simulation random seed',min_value=0,value=int(cfg['seed']))
            st.caption('Noise follows the released heteroscedastic Gaussian equation, using your explicit frame durations. Negative noisy TAC values are retained. Match noise amplitude to your acquisition and reconstruction.')
            if st.button('Generate dataset',type='primary'):
                try:
                    cfg.update(n_train=n_train,n_val=n_val,n_test=n_test,noise_prior={'type':'uniform','min':low,'max':high},chunk_size=chunk,seed=seed)
                    validate_protocol(cfg)
                    if cfg.get('aif_mode','measured') == 'measured': load_measured_inputs(cfg)
                    if dataset_path.exists(): raise ValueError('Dataset already exists. Choose a new project or move it aside.')
                    write_json(protocol_path,cfg)
                    launch('Dataset simulation',['simulate_protocol.py','--config',protocol_path])
                except Exception as e: st.error(str(e))
    else:
        dataset_path = upload_or_path('Fine-tuning NPZ file',st.session_state.get('dataset_'+project_name,str(dataset_path)),['npz'],'dataset')
        st.markdown('Required: `x_train/x_val/x_test` = parameters then noise amplitude; `y_train/y_val/y_test` = TAC then AIF; `t_meas`. Custom parameters also require `parameter_names`. Explicit durations are recommended.')
        st.caption('Fine-tuning needs labelled parameter–curve pairs. Unlabelled clinical TACs alone are prediction inputs and cannot replace supervised training pairs.')
    if st.button('Validate and select dataset'):
        try:
            report = validate_file(dataset_path)
            st.session_state['dataset_'+project_name] = str(dataset_path)
            write_json(workspace/'dataset_selection.json',{'path':str(dataset_path),'report':report})
            for training_page,mode in [('3 · Fine-tune','lora'),('CM from scratch','cm')]:
                project_state.remember(training_page+':text_input:'+mode+'_'+workspace.name+'_data_'+workspace.name,str(dataset_path))
            st.success('Dataset accepted. Dimensions, parameter order, finite values and training scales checked.')
            st.json(report)
        except Exception as e: st.error(f'Dataset rejected: {e}')

elif page in ('3 · Fine-tune','CM from scratch'):
    from gui.training import render as render_training
    run_path=render_training(ROOT,workspace,launch,upload_or_path,mode='lora' if page=='3 · Fine-tune' else 'cm',defaults=default_training)

elif page == '4 · Predict':
    st.title('4 · Predict posteriors on your testing curves')
    run_path = model_run_picker(workspace,run_path)
    run_path = path(st.text_input('Model run folder',str(run_path),key='predict_run_'+str(run_path)))
    route=st.selectbox('Prediction input',['UC Davis patient slice','Held-out simulated pairs','My own file'],index=0 if preset==PRESETS[0] else 2)
    if route=='Held-out simulated pairs':
        input_default=str(workspace/'testing.npz')
        if st.button('Prepare held-out testing file'):
            try:
                selection=workspace/'dataset_selection.json'
                source=json.loads(selection.read_text(encoding='utf-8'))['path'] if selection.exists() else dataset_path
                testing_from_dataset(source,workspace/'testing.npz');st.success('Testing file prepared from the dataset’s held-out pairs.')
            except Exception as e: st.error(str(e))
        st.caption('These pairs can measure accuracy when truth is known; they do not have anatomical geometry for step 6.')
    elif route=='UC Davis patient slice':
        input_default=str(ROOT/'data/real/Sub001_slice75_adjusted.h5')
        st.caption('Predict all 24,237 foreground voxel TACs to make the supplied coronal map in step 6.')
    input_path = upload_or_path('Testing dataset',input_default,['npz','h5','hdf5'],'testing_'+route)
    st.markdown('NPZ: `tac [N,T]`, `aif [T]` or `[N,T]`, `t_meas [T]`, `frame_durations_min [T]`; optional `voxel_columns [N]` and `truth [N,P]`. Legacy pandas HDF5 is also accepted.')
    a,b,c = st.columns(3)
    samples = a.number_input('Posterior samples per curve',min_value=2,value=8 if small else 1000,key='prediction_samples_'+profile)
    batch = b.number_input('Curve batch size',min_value=1,value=64 if small else 16,key='prediction_batch_'+profile)
    sample_chunk = c.number_input('Sample chunk',min_value=1,value=8 if small else 64,key='prediction_chunk_'+profile)
    if small: st.caption('Eight draws per voxel are sufficient for an installation check; use a much larger sample count for credible-interval assessment.')
    a,b,c=st.columns(3)
    steps=a.number_input('Reverse sampling steps',min_value=1,value=3,help='Consistency sampling evaluations, evenly spaced from 1 towards 0. More steps do not necessarily improve a consistency model.')
    seed=b.number_input('Inference random seed',min_value=0,value=0)
    device=c.selectbox('Prediction device',['auto','cpu','cuda:0','mps'])
    st.caption('Auto selects CUDA when available, then Apple MPS, then CPU. MPS expanded batches are automatically capped and checked against CPU.')
    checkpoint_name = checkpoint_picker(run_path)
    output = path(st.text_input('Prediction output',str(run_path/'predictions.npz'),key='predict_output_'+str(run_path)))
    if st.button('Validate input'):
        try:
            model_cfg = json.loads((run_path/'model_params.json').read_text(encoding='utf-8'))
            scale = json.loads((run_path/'scaling_params.json').read_text(encoding='utf-8'))
            _,_,conditioning,_,_ = read_prediction_input(input_path,model_cfg['x_dim'],scale['t_meas'],scale.get('frame_durations_min'))
            st.success(f'Accepted {len(conditioning)} curves on the training schedule.')
        except Exception as e: st.error(f'Input rejected: {e}')
    if st.button('Run prediction',type='primary'):
        try:
            if not (run_path/checkpoint_name).exists(): raise ValueError('Checkpoint missing; finish training or select last.pth')
            cfg = json.loads((run_path/'model_params.json').read_text(encoding='utf-8')); scale=json.loads((run_path/'scaling_params.json').read_text(encoding='utf-8'))
            read_prediction_input(input_path,cfg['x_dim'],scale['t_meas'],scale.get('frame_durations_min'))
            st.session_state['prediction_'+str(workspace)]=str(output)
            st.session_state['prediction_input_'+str(workspace)]=str(input_path)
            launch('Posterior prediction',['infer.py','--run',run_path,'--checkpoint',checkpoint_name,'--input',input_path,'--output',output,
                                          '--samples',samples,'--batch-size',batch,'--sample-chunk',sample_chunk,'--steps',steps,'--seed',seed,'--device',device])
            key='imaging_predictions_'+str(run_path)
            project_state.remember('file:'+key,str(output))
            project_state.remember('6 · Parametric imaging:text_input:'+key+'_'+project_name+'_path',str(output))
        except Exception as e: st.error(str(e))
    if output.exists():
        with np.load(output,allow_pickle=False) as data:
            st.subheader('Posterior summaries')
            names = data['parameter_names'].tolist()
            st.dataframe(pd.DataFrame(data['params_mean'][:100],columns=names,index=data['voxel_columns'][:100]))
            if 'Ki_mean' in data: st.caption('2TC Kᵢ statistics are calculated from joint draws. Inspect negative-draw fractions and Kᵢ tails.')
        st.download_button('Download predictions',output.read_bytes(),file_name=output.name)
        if output.with_suffix('.json').exists(): st.json(json.loads(output.with_suffix('.json').read_text(encoding='utf-8')))

elif page == '5 · Single TAC & NLS':
    from gui.single_tac import render
    run_path = render(ROOT,workspace,launch,default_run=run_path,default_input=input_default,enable_cm=enable_cm)

elif page == '6 · Parametric imaging':
    from gui.imaging import render as render_imaging
    render_imaging(ROOT,workspace,run_path,st.session_state.get('prediction_input_'+str(workspace),input_default),upload_or_path,launch=launch)

elif page == 'Compare models':
    from gui.comparison import render as render_comparison
    render_comparison(ROOT,workspace,launch,upload_or_path)

elif page == 'User manual':
    st.title('LoRACM user manual')
    sections={'Overview':'README.md','GUI and project settings':'docs/WORKBENCH.md','Installation':'docs/INSTALLATION.md','Models, inputs and protocols':'docs/MODELS_AND_PROTOCOLS.md',
              'Prepare your dataset':'docs/PREPARE_DATA.md','Fine-tuning':'docs/FINE_TUNING.md','CM from scratch and comparison':'docs/TRAIN_CM.md','Prediction and single-TAC NLS':'docs/PREDICTION.md',
              'Parametric imaging':'docs/PARAMETRIC_IMAGING.md','UC Davis demonstration':'docs/UCDAVIS.md'}
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
    st.title('UC Davis · Learn the complete workflow')
    st.info('The default UC Davis route is now available in numbered steps 1–6, including patient-slice prediction, parametric maps and voxel inspection. The shortcuts below retain the earlier AIF-grid example and matched simulated testing.')
    st.write('The supplied four measured AIFs adapt the FDG 2TC foundation model to the uEXPLORER protocol. Choose an installation check or the original Scenario 1 settings.')
    mode = st.radio('Demonstration',['Small CPU installation check','Original Scenario 1'])
    is_smoke = mode.startswith('Small')
    demo = workspace/('demo_smoke' if is_smoke else 'demo_scenario1')
    demo_data = demo/'dataset.npz'; demo_run=demo/'runs'/'demo'
    config_suffix = 'smoke' if is_smoke else 'scenario1'
    st.table(pd.DataFrame({'Setting':['Training / validation / test pairs','LoRA rank / alpha','Epochs'],
                           'Value':['32 / 16 / 8' if is_smoke else '50,000 / 100,000 / 100','4 / 8','2' if is_smoke else '5,000']}))
    st.caption('These preserved demonstration settings differ from new-project defaults. The small run checks installation and produces no scientifically useful model.')
    if st.button('1. Prepare supplied demonstration pairs'):
        try:
            if demo_data.exists(): raise ValueError('Demo dataset exists; select a new project or use the next step')
            cfg = json.loads((ROOT/f'configs/dataset_{config_suffix}.json').read_text(encoding='utf-8'))
            demo.mkdir(parents=True,exist_ok=True)
            with np.load(ROOT/'data/simulated/scenario1.npz',allow_pickle=False) as data:
                subset={'t_meas':data['t_meas']}
                for split in ('train','val','test'):
                    for prefix in ('x','y','aif_index'):
                        key=prefix+'_'+split
                        if key in data: subset[key]=data[key][:cfg['n_'+split]]
                np.savez_compressed(demo_data,**subset)
            write_json(demo/'dataset.json',dict(cfg,source='Supplied Scenario 1 pairs; deterministic leading subset',output=str(demo_data)))
            st.success('Demonstration dataset prepared without JAX. Continue to fine-tuning.')
        except Exception as e: st.error(str(e))
    if st.button('2. Fine-tune demonstration model',disabled=not demo_data.exists()):
        cfg = json.loads((ROOT/f'configs/train_{config_suffix}.json').read_text(encoding='utf-8'))
        cfg.update(pretrained=str(ROOT/cfg['pretrained']),dataset_dir=str(demo_data),save_dir=str(demo/'runs'),exp_name='demo')
        write_json(demo/'train.json',cfg); launch('UC Davis fine-tuning',['train.py','--config',demo/'train.json'])
    st.write('3. Test on held-out simulated pairs with the same schedule. This avoids the timing mismatch in the supplied patient-slice file.')
    if st.button('3. Predict demonstration test set',disabled=not (demo_run/'last.pth').exists()):
        try:
            with np.load(demo_data,allow_pickle=False) as data:
                t = data['t_meas']; y=data['y_test']; size=len(t)
                # Duration is absent in the research NPZ; legacy inference does not use it.
                test_file=demo/'testing.npz'
                np.savez_compressed(test_file,tac=y[:,:size],aif=y[:,size:],t_meas=t,
                                    frame_durations_min=np.r_[t[0],np.diff(t)],truth=data['x_test'][:,:5])
            launch('UC Davis prediction',['infer.py','--run',demo_run,'--checkpoint','last.pth' if is_smoke else 'best.pth',
                                          '--input',test_file,'--output',demo_run/'predictions.npz','--samples','8' if is_smoke else '1000',
                                          '--batch-size','2' if is_smoke else '16','--sample-chunk','4' if is_smoke else '64','--device','cpu' if is_smoke else 'auto'])
        except Exception as e: st.error(str(e))
    st.warning('The released patient-slice time grid differs from the adaptation AIF grid. Clinical inference must pass the timing check after appropriate adaptation or verified resampling. The imaging notebook also requires author-verified voxel mapping and orientation.')
    run_path = demo_run


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
            st.text(jobs.tail())
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
