"""Locate a reusable Python environment without binding it to a checkout."""
import os
from pathlib import Path
import subprocess
import sys


def environment_python(folder,platform=None):
    return Path(folder)/('Scripts/python.exe' if (platform or sys.platform)=='win32' else 'bin/python')


def default_environment():
    return Path.home()/'.venvs/pet-tracer'


def candidates(root,environ=None,home=None,platform=None,executable=None):
    environ=os.environ if environ is None else environ
    platform=platform or sys.platform
    if environ.get('PET_TRACER_PYTHON'):
        return [Path(environ['PET_TRACER_PYTHON']).expanduser()]
    folders=[Path(environ[name]) for name in ('VIRTUAL_ENV','CONDA_PREFIX') if environ.get(name)]
    folders += [(Path.home() if home is None else Path(home))/'.venvs/pet-tracer',Path(root)/'.venv']
    paths=[environment_python(folder,platform) for folder in folders]
    paths.append(Path(executable or sys.executable))
    return list(dict.fromkeys(paths))


def find_gui_python(root):
    checked=[]
    for python in candidates(root):
        if not python.is_file(): continue
        try:
            check=subprocess.run([str(python),'-c','import streamlit, torch, numpy, pandas, scipy, tables; assert tuple(int(v) for v in torch.__version__.split(".")[:2]) >= (2,6), "PyTorch >=2.6 is required"'],capture_output=True,text=True)
        except OSError as error:
            checked.append(str(python)+': '+str(error))
            continue
        if check.returncode==0: return python
        detail=check.stderr.strip().splitlines()[-1] if check.stderr.strip() else 'Dependency check failed'
        checked.append(str(python)+': '+detail)
    raise RuntimeError('No complete GUI environment found. Run setup_environment.py once, or activate your existing PET environment.\n'+'\n'.join(checked))
