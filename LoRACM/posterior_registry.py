"""Identifiers and user-facing names shared by saved runs and prediction tools."""
import json
from pathlib import Path

METHOD_NAMES = {'cm': 'CM', 'imf': 'iMF', 'scm': 'sCM'}
DEFAULT_STEPS = {'cm': 3, 'imf': 1, 'scm': 2}


def posterior_method(config):
    method = config.get('posterior_method', 'cm')
    if not isinstance(method, str) or method not in METHOD_NAMES:
        raise ValueError(f'Unknown posterior_method {method!r}; choose cm, imf or scm')
    if method != 'cm' and config.get('training_mode', 'lora') != 'cm':
        raise ValueError('iMF and sCM are available only for Module 1 training runs')
    return method


def run_model_label(config):
    method = posterior_method(config)
    if config.get('training_mode', 'lora') == 'lora':
        return 'LoRACM'
    return 'CM from scratch' if method == 'cm' else f'{METHOD_NAMES[method]} from scratch'


def run_method(run):
    """A new or missing run displays the legacy defaults until its config exists."""
    path = Path(run) / 'model_params.json'
    return posterior_method(json.loads(path.read_text(encoding='utf-8'))) if path.is_file() else 'cm'


def posterior_label(data, default):
    if 'metadata_json' in data:
        config = json.loads(str(data['metadata_json'])).get('config')
        if config:
            return run_model_label(config)
    return default
