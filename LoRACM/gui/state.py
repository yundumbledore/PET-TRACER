"""Project-owned drafts survive Streamlit's cleanup of off-page widget state."""
import functools
import hashlib
import json
import inspect
from pathlib import Path
import pandas as pd
import streamlit as st
from streamlit.delta_generator import DeltaGenerator
from runtime import write_json_atomic

_installed = False


def context(workspace, page):
    workspace = Path(workspace)
    store_key = '_project_drafts_' + str(workspace)
    if store_key not in st.session_state:
        filename = workspace/'gui_state.json'
        st.session_state[store_key] = json.loads(filename.read_text(encoding='utf-8')) if filename.exists() else {}
    st.session_state['_draft_context'] = (workspace, page, store_key)


def remember(name, value):
    workspace, page, store_key = st.session_state['_draft_context']
    store = st.session_state[store_key]
    if store.get(name) != value:
        store[name] = value
        write_json_atomic(workspace/'gui_state.json', store)


def get(name, default=None):
    return st.session_state[st.session_state['_draft_context'][2]].get(name, default)


def _encode_frame(value):
    return {'_dataframe': value.to_json(orient='split')}


def _wrapper(original, method, bound=False):
    @functools.wraps(original)
    def widget(*args, **kwargs):
        if not st.session_state.get('_draft_context'): return original(*args, **kwargs)
        supplied = args[1:] if bound else args
        label = supplied[0] if supplied else kwargs.get('label', '')
        if method == 'data_editor': label = kwargs.get('key', 'table')
        workspace, page, store_key = st.session_state['_draft_context']
        def save(value):
            store = st.session_state[store_key]
            if store.get(name) != value:
                store[name] = value
                write_json_atomic(workspace/'gui_state.json', store)
        name = page + ':' + method + ':' + str(kwargs.get('key', label))
        token = hashlib.sha256((str(workspace)+name).encode()).hexdigest()[:24]
        key = '_widget_' + token
        kwargs['key'] = key
        store = st.session_state[store_key]
        old_callback = kwargs.pop('on_change', None)
        old_args = kwargs.pop('args', ()) or ()
        old_kwargs = kwargs.pop('kwargs', {}) or {}
        if method == 'data_editor':
            data = supplied[0] if supplied else kwargs['data']
            if name in store:
                import io
                data = pd.read_json(io.StringIO(store[name]['_dataframe']), orient='split').reset_index(drop=True)
            call_args = list(args)
            if supplied: call_args[1 if bound else 0] = data
            else: kwargs['data'] = data
            def changed():
                edits = st.session_state.get(key, {})
                frame = data.copy()
                for row, cells in edits.get('edited_rows', {}).items():
                    for column, value in cells.items(): frame.loc[frame.index[int(row)], column] = value
                frame = frame.drop(frame.index[edits.get('deleted_rows', [])])
                if edits.get('added_rows'): frame = pd.concat([frame, pd.DataFrame(edits['added_rows'])], ignore_index=True)
                save(_encode_frame(frame))
                # The next render uses the committed draft, not the old edit delta.
                st.session_state.pop(key, None)
                if old_callback: old_callback(*old_args, **old_kwargs)
            kwargs['on_change'] = changed
            value = original(*call_args, **kwargs)
            save(_encode_frame(value))
            return value
        # Rehydrate before rendering; widget keys may have been removed on another page.
        if name in store:
            value = store[name]
            if method == 'slider' and isinstance(value,list): value = tuple(value)
            if method=='number_input' and value is not None:
                if kwargs.get('max_value') is not None: value=min(value,kwargs['max_value'])
                if kwargs.get('min_value') is not None: value=max(value,kwargs['min_value'])
            if method in ('selectbox','radio'):
                options = supplied[1] if len(supplied)>1 else kwargs['options']
                if value not in list(options): value = None
                if value is None:
                    store.pop(name, None)
                    st.session_state.pop(key,None)
            if name in store:
                # Use a saved draft as the widget's initial value, rather than
                # creating an off-page widget through the Session State API.
                call = inspect.signature(original).bind_partial(*args, **kwargs)
                if method in ('selectbox','radio'): call.arguments['index'] = list(options).index(value)
                else: call.arguments['value'] = value
                args, kwargs = call.args, call.kwargs
                if key in st.session_state and st.session_state[key] != value:
                    st.session_state[key] = value
                    if method in ('text_input','text_area','number_input'):
                        call.arguments['value'] = None
                        args, kwargs = call.args, call.kwargs
        def changed():
            save(st.session_state[key])
            if old_callback: old_callback(*old_args, **old_kwargs)
        kwargs['on_change'] = changed
        value = original(*args, **kwargs)
        # All supported values are JSON primitives (slider ranges use lists).
        save(list(value) if isinstance(value, tuple) else value)
        return value
    return widget


def install():
    global _installed
    if _installed: return
    for method in ('text_input','text_area','number_input','selectbox','radio','checkbox','slider','data_editor'):
        setattr(st, method, _wrapper(getattr(st, method), method))
        setattr(DeltaGenerator, method, _wrapper(getattr(DeltaGenerator, method), method, bound=True))
    _installed = True


def checkpoint_picker(run, label='Checkpoint'):
    run = Path(run)
    choices = ['best.pth','last.pth'] + [p.relative_to(run).as_posix() for p in sorted((run/'epochs').glob('epoch_*.pth'))]
    index_file = run/'checkpoints.json'
    info = json.loads(index_file.read_text(encoding='utf-8')) if index_file.exists() else {}
    def description(name):
        item = info.get(name)
        if item: return f"{name} · epoch {item['epoch']} · val {item['val_loss']:.6g}"
        return name + (' · lowest validation loss' if name=='best.pth' else '')
    return st.selectbox(label, choices, key=label+'_'+str(run), index=0 if (run/'best.pth').exists() or not (run/'last.pth').exists() else 1, format_func=description)


def model_run_picker(workspace, default, label='Model to use'):
    runs = [p for p in sorted((Path(workspace)/'runs').glob('*')) if (p/'model_params.json').exists()]
    if not runs: return Path(default)
    labels = {str(p):('LoRACM' if json.loads((p/'model_params.json').read_text(encoding='utf-8')).get('training_mode','lora')=='lora' else 'CM from scratch')+' · '+p.name for p in runs}
    options = list(labels)
    selected = st.selectbox(label, options, index=options.index(str(default)) if str(default) in options else 0, format_func=lambda p:labels[p])
    return Path(selected)
