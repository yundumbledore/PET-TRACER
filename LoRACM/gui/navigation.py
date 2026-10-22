"""Separate module navigation and release-independent project locations."""
import os
from pathlib import Path
import streamlit as st

MODULES={'cm':'CM Posterior Estimation','lora':'LoRACM Adaptation','selection':'Bayesian Model Selection'}
WORKFLOW_PAGES={
    'cm':['Welcome','1 · Model & protocol','CM · Dataset','CM from scratch','CM · Predict','5 · Single TAC & NLS','6 · Parametric imaging','User manual'],
    'lora':['Welcome','1 · Model & protocol','2 · Dataset','3 · Fine-tune','4 · Predict','5 · Single TAC & NLS','6 · Parametric imaging','UC Davis demonstration','User manual'],
    'selection':['1 · P001 demonstration','2 · Inference settings','3 · Run inference','4 · Explore results']}
CM_LABELS={'Welcome':'Overview','CM · Dataset':'2 · Dataset','CM from scratch':'3 · Train CM','CM · Predict':'4 · Predict',
           '5 · Single TAC & NLS':'5 · Single TAC & NLS','6 · Parametric imaging':'6 · Parametric imaging'}


def go(module):
    st.session_state['_draft_context']=None
    st.query_params['module']=module
    st.session_state.pop('workflow_page',None)
    st.session_state.pop('active_run',None)
    st.rerun()


def workspace_base(root):
    configured=os.environ.get('PET_TRACER_WORKSPACES')
    if configured:return Path(configured).expanduser()
    # Keep existing projects discoverable in an older checkout. New installs use
    # a persistent home-folder location, independent of the downloaded release.
    legacy=Path(root)/'workspaces'
    if legacy.exists() and any(p.is_dir() and not p.name.startswith('.') for p in legacy.iterdir()):return legacy
    return Path.home()/'.pet-tracer'/'workspaces'


def project_folder(base,name,module):
    # LoRACM keeps its original project paths; CM projects are fully isolated.
    return Path(base)/('cm' if module=='cm' else 'model_selection' if module=='selection' else '')/name


def installation_help(root):
    st.subheader('Install once, keep your projects when upgrading')
    st.write('The GUI runs locally in your browser. Python 3.11 and the reusable PET-TRACER environment support Windows, Linux and Apple Silicon macOS.')
    st.markdown('**First installation:** On Windows, open **Setup PET-TRACER.cmd**, then **Launch PET-TRACER.cmd**. On macOS/Linux, run:')
    st.code('python3.11 setup_environment.py\npython3 launch_gui.py',language='bash')
    st.markdown('**Next release:** stop the running app, download and extract the new release into a new folder, then use its launcher. The existing Python environment is reused. If that release changes dependencies, rerun its setup command in the same environment.')
    st.caption('Project location: '+str(workspace_base(root)))
    st.write('New installations keep projects under ~/.pet-tracer/workspaces. To use another stable folder, set PET_TRACER_WORKSPACES before launch. Existing checkout-local projects remain available; copy them to a stable external folder before replacing that checkout. References to byte-identical bundled files are reconnected when a saved project opens in a new download.')
    st.markdown('[Installation and updates](https://github.com/yundumbledore/PET-TRACER/blob/main/LoRACM/docs/INSTALLATION.md)')


def sidebar_logo(root):
    st.image(str(Path(root).parent/'Assets/PET-TRACER_logo.png'),width=240)


def home(root):
    root=Path(root)
    with st.sidebar:
        sidebar_logo(root)
        st.caption('Bayesian kinetic modelling for dynamic PET')
        st.markdown('Choose a module to open its workflow.')
        st.markdown('[User manual](https://github.com/yundumbledore/PET-TRACER)')
    st.title('PET-TRACER')
    st.subheader('Bayesian kinetic modelling for dynamic PET')
    st.write('Estimate kinetic parameter posteriors, adapt a pretrained consistency model, or explore Bayesian kinetic model selection.')
    st.markdown('''<style>
[data-testid="stMainBlockContainer"] [data-testid="stImage"] img {height:180px;object-fit:contain}
/* Stretch the welcome cards to the tallest module and anchor their buttons. */
[data-testid="stHorizontalBlock"]:has([data-testid="stImage"]) > [data-testid="stColumn"] {
    align-self:stretch;display:flex;flex-direction:column;
    border:1px solid #dce7ec;border-radius:10px;padding:1rem;
}
[data-testid="stHorizontalBlock"]:has([data-testid="stImage"]) > [data-testid="stColumn"] > div {
    flex:1;display:flex;flex-direction:column;
}
[data-testid="stHorizontalBlock"]:has([data-testid="stImage"]) > [data-testid="stColumn"] [data-testid="stVerticalBlock"] {flex:1}
[data-testid="stHorizontalBlock"]:has([data-testid="stImage"]) > [data-testid="stColumn"] [data-testid="stVerticalBlock"] > :last-child {margin-top:auto}
</style>''',unsafe_allow_html=True)
    columns=st.columns(3)
    items=[('cm','1 · TRAIN A NEW CM FOR KINETIC POSTERIOR ESTIMATION','Train a consistency model from scratch for your kinetic model and acquisition protocol.',
            root.parent/'Assets/Welcome_CM_Workflow.png','Open CM workflow'),
           ('lora','2 · ADAPT A PRE-TRAINED CM TO A NEW SCENARIO WITH LORA','Fine-tune a pretrained consistency model with LoRA to handle a new study or protocol.',
            root.parent/'Assets/Welcome_LoRACM_Workflow.png','Open LoRACM workflow'),
           ('selection','3 · USE CM FOR BAYESIAN KINETIC MODEL SELECTION','Run the pretrained seven-model FDG example and inspect model probabilities, kinetic posteriors and fitted TACs.',
            root.parent/'Assets/Welcome_Model_Selection_Workflow.png','Open model-selection workflow')]
    for col,(module,title,description,image,label) in zip(columns,items):
        with col:
            if image.exists():st.image(str(image),width=300)
            st.subheader(title)
            st.write(description)
            st.caption('Use your data or simulate your own model.' if module!='selection' else 'P001, coronal slice 116. Training planned for a future release.')
            if st.button(label,key='open_'+module,type='primary'):go(module)
    st.divider()
    st.subheader('The team')
    st.markdown('**Yun Zhao and Steven Meikle** · The University of Sydney, Australia  \n**Andrew Reader** · King’s College London, UK  \n**Yanan Fan** · CSIRO’s Data61, Australia')
    st.markdown('**Contact:** [yun.zhao@sydney.edu.au](mailto:yun.zhao@sydney.edu.au)')
    st.caption('Runs locally on Windows, Linux and macOS. CUDA and Apple MPS acceleration are available with a compatible PyTorch installation; CPU remains supported.')
    with st.expander('Installation, updates and project storage'):installation_help(root)
    if st.session_state.get('jobs') and st.session_state.jobs.running():st.info('A computation is running in the background. Return to its module to view the job monitor.')


def module_overview(module):
    st.title(MODULES[module])
    if module=='cm':
        st.write('Train a new consistency model from random initialisation, then estimate kinetic parameter posteriors and make parametric images.')
        st.info('Start with the supplied UC Davis model, AIFs and protocol. Choose Quick installation check for a short run, or use your own data and a custom kinetic model.')
        st.markdown('1. Define the kinetic model and acquisition protocol.\n2. Simulate labelled pairs or upload your dataset.\n3. Configure the U-Net and train the CM.\n4. Predict on your own data.\n5. Inspect a TAC, parameter posteriors and an NLS reference.\n6. Create and inspect parametric images.')
    else:
        st.write('Use low-rank adaptation to fine-tune the pretrained consistency model for a new protocol or study.')
        st.info('UC Davis remains the default example. Reproduce its supplied paper checkpoint or follow the workflow to simulate data and fine-tune. Switch to My own project to start from your own data and kinetics.')
        st.markdown('1. Define the kinetic model and acquisition protocol.\n2. Simulate labelled pairs or upload your dataset.\n3. Configure LoRA and fine-tune.\n4. Predict on your own data.\n5. Inspect a TAC, posteriors and an NLS reference.\n6. Create and inspect parametric images.')
