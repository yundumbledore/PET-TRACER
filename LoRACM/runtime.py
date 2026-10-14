"""Shared device selection and machine-readable progress for CLI and GUI jobs."""
import json
import os
from pathlib import Path
import sys
import time


def select_device(requested='auto'):
    import torch
    if requested == 'auto':
        requested = 'cuda:0' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    device = torch.device(requested)
    if device.type == 'mps' and not torch.backends.mps.is_available():
        raise ValueError('MPS is unavailable in this Python environment. Install native arm64 PyTorch on an Apple M-series Mac, or select CPU.')
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise ValueError('CUDA is unavailable in this Python environment. Install CUDA-enabled PyTorch, or select CPU.')
    return device


def write_json_atomic(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n', encoding='utf-8')
    temporary.replace(path)


class Progress:
    def __init__(self, stage, total, unit='items'):
        self.stage, self.total, self.unit = stage, int(total), unit
        self.started = time.monotonic()
        self.filename = os.environ.get('PET_TRACER_PROGRESS_FILE')
        self.bar = None
        if not self.filename and sys.stderr.isatty():
            from tqdm import tqdm
            self.bar = tqdm(total=total, desc=stage, unit=unit)
        self.update(0)

    def update(self, completed, state='running', **details):
        self.completed = int(completed)
        if self.filename:
            write_json_atomic(self.filename, dict(stage=self.stage, completed=self.completed,
                total=self.total, unit=self.unit, state=state,
                elapsed_seconds=time.monotonic()-self.started, **details))
        if self.bar:
            self.bar.update(max(0, self.completed-self.bar.n))
            if state != 'running': self.bar.close()


def stop_requested():
    filename = os.environ.get('PET_TRACER_STOP_FILE')
    return bool(filename and Path(filename).exists())


def check_cancelled():
    if stop_requested():
        raise RuntimeError('Job cancelled by the user; no completed result was published.')
