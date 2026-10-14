"""Shared device selection and machine-readable progress for CLI and GUI jobs."""
import json
import os
from pathlib import Path
import sys
import time
import tempfile


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
    # Close the file before replacing it: Windows readers/virus scanners may
    # briefly hold the destination, and concurrent writes need unique names.
    descriptor, name = tempfile.mkstemp(prefix=path.name+'.',suffix='.tmp',dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(descriptor,'w',encoding='utf-8') as stream:
            stream.write(json.dumps(value, indent=2)+'\n')
        for attempt in range(7):
            try:
                os.replace(temporary,path)
                break
            except PermissionError as error:
                if attempt == 6:
                    raise PermissionError(f'Cannot update {path}. Check that this project folder is writable and the file is not locked by another application. Original error: {error}') from error
                time.sleep(.02*2**attempt)
    finally:
        temporary.unlink(missing_ok=True)


def probe_device(device):
    """Exercise actual compute/gradient kernels before creating a training run."""
    import torch
    device = torch.device(device)
    details = {'device':str(device),'torch_version':str(torch.__version__),'cuda_build':torch.version.cuda}
    if device.type == 'cuda':
        details.update(gpu=torch.cuda.get_device_name(device),capability=list(torch.cuda.get_device_capability(device)),
                       compiled_cuda_architectures=torch.cuda.get_arch_list())
    try:
        with torch.enable_grad():
            values=torch.ones((8,8),device=device,requires_grad=True)
            result=(values@values).sin().sum()
            result.backward()
            if device.type == 'cuda': torch.cuda.synchronize(device)
            if not torch.isfinite(result).item() or not torch.isfinite(values.grad).all().item():
                raise RuntimeError('Non-finite compute check')
    except RuntimeError as error:
        raise ValueError(f'Compute check failed on {device}: {error}. PyTorch {details["torch_version"]}, '
                         f'CUDA build {details["cuda_build"]}, GPU {details.get("gpu","not CUDA")}. '
                         'Use the official PyTorch installation selector for a build supporting your GPU/driver: '
                         'https://pytorch.org/get-started/locally/. Select CPU to check the remaining workflow.') from error
    return details


class Progress:
    def __init__(self, stage, total, unit='items'):
        self.stage, self.total, self.unit = stage, int(total), unit
        self.started = time.monotonic()
        self.filename = os.environ.get('PET_TRACER_PROGRESS_FILE')
        self.bar = None
        self.status_warning = False
        if not self.filename and sys.stderr.isatty():
            from tqdm import tqdm
            self.bar = tqdm(total=total, desc=stage, unit=unit)
        self.update(0)

    def update(self, completed, state='running', **details):
        self.completed = int(completed)
        if self.filename:
            try:
                write_json_atomic(self.filename, dict(stage=self.stage, completed=self.completed,
                    total=self.total, unit=self.unit, state=state,
                    elapsed_seconds=time.monotonic()-self.started, **details))
                self.status_warning = False
            except OSError as error:
                # A locked UI status file must not discard a long computation.
                if not self.status_warning:
                    print(f'Progress display could not update {self.filename}: {error}. Computation continues; results and diagnostics remain in the job log.',file=sys.stderr,flush=True)
                    self.status_warning = True
        if self.bar:
            self.bar.update(max(0, self.completed-self.bar.n))
            if state != 'running': self.bar.close()


def stop_requested():
    filename = os.environ.get('PET_TRACER_STOP_FILE')
    return bool(filename and Path(filename).exists())


def check_cancelled():
    if stop_requested():
        raise RuntimeError('Job cancelled by the user; no completed result was published.')
