"""Paired held-out evaluation and optional posterior overlays."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from plot_posteriors import posterior_figure


def render(root, workspace, launch, upload_or_path):
    st.title('Compare LoRACM and CM')
    st.caption('Evaluate both models on the same held-out TACs. Include known parameter truth for an accuracy comparison.')
    def previous_run(filename,expected,fallback):
        file=workspace/filename
        if file.exists():
            cfg=json.loads(file.read_text(encoding='utf-8'))
            if cfg.get('training_mode','lora')==expected: return str(workspace/'runs'/cfg['exp_name'])
        return str(workspace/'runs'/fallback)
    a,b=st.columns(2)
    lora=a.text_input('LoRACM run',previous_run('train.json','lora','adaptation'))
    cm=b.text_input('Scratch CM run',previous_run('train_cm.json','cm','cm_scratch'))
    testing=upload_or_path('Shared testing NPZ or HDF5',str(workspace/'testing.npz'),['npz','h5','hdf5'],'compare_test')
    a,b,c=st.columns(3)
    samples=a.number_input('Comparison posterior samples',min_value=2,value=1000)
    device=b.selectbox('Comparison device',['auto','cpu','cuda:0','mps'])
    name=c.text_input('Comparison name','comparison')
    retain=st.checkbox('Keep posterior draws for overlays (small curve sets)',value=False)
    with st.expander('Inference settings'):
        a,b=st.columns(2)
        lora_checkpoint=a.selectbox('LoRACM checkpoint',['best.pth','last.pth'])
        cm_checkpoint=b.selectbox('CM checkpoint',['best.pth','last.pth'])
        a,b,c=st.columns(3)
        batch=a.number_input('Comparison curve batch',min_value=1,value=16)
        chunk=b.number_input('Comparison sample chunk',min_value=1,value=64)
        seed=c.number_input('Comparison seed',min_value=0,value=0)
    output=workspace/'comparisons'/name
    if st.button('Compare models',type='primary'):
        try:
            import re
            if not re.fullmatch(r'[A-Za-z0-9_-]+',name): raise ValueError('Use a simple comparison name with letters, numbers, underscores or hyphens')
            def local(value):
                p=Path(value).expanduser();return p if p.is_absolute() else root/p
            if output.exists(): raise ValueError('Comparison folder exists; choose a new name')
            for run,expected in [(local(lora),'lora'),(local(cm),'cm')]:
                cfg=json.loads((run/'model_params.json').read_text(encoding='utf-8'))
                if cfg.get('training_mode','lora')!=expected: raise ValueError('Select the correct LoRACM and scratch CM run folders')
            args=['compare_models.py','--lora-run',local(lora),'--cm-run',local(cm),'--input',testing,'--output',output,
                  '--samples',samples,'--device',device,'--batch-size',batch,'--sample-chunk',chunk,'--seed',seed,
                  '--lora-checkpoint',lora_checkpoint,'--cm-checkpoint',cm_checkpoint]
            if retain: args+=['--save-samples']
            launch('Model comparison',args)
        except Exception as e: st.error(str(e))
    report_path=output/'comparison.json'
    if report_path.exists():
        report=json.loads(report_path.read_text(encoding='utf-8'));metrics=pd.read_csv(output/'metrics.csv')
        st.subheader('Held-out results')
        if 'rmse' in metrics:
            st.bar_chart(metrics.pivot(index='parameter',columns='model',values='rmse'))
            st.caption('RMSE of posterior means, in each parameter’s units. Compare the two models within each parameter.')
        else:
            st.info('No parameter truth was supplied. Uncertainty summaries and posterior overlays describe model outputs, not accuracy.')
        st.dataframe(metrics,hide_index=True)
        if report['same_training_dataset'] is False: st.warning('These models used different target training datasets. Treat the comparison accordingly.')
        if report['training_configuration_differences']:
            with st.expander('Training settings that differ'): st.json(report['training_configuration_differences'])
        with st.expander('Training and inference details'): st.json(report)
        st.download_button('Download comparison metrics',(output/'metrics.csv').read_bytes(),file_name='comparison_metrics.csv')
        with np.load(output/'loracm.npz',allow_pickle=False) as lora_data, np.load(output/'cm.npz',allow_pickle=False) as cm_data:
            if 'posterior_samples' in lora_data and 'posterior_samples' in cm_data:
                ids=lora_data['voxel_columns'].tolist();chosen=st.selectbox('Posterior overlay · TAC',ids)
                i=ids.index(chosen)
                fig=posterior_figure(lora_data['posterior_samples'][i],lora_data['parameter_names'].tolist(),comparison_draws=cm_data['posterior_samples'][i])
                st.pyplot(fig);plt.close(fig)
