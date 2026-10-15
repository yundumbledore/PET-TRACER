"""Create/reuse a shared PET-TRACER environment outside the downloaded repo."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import venv
from pet_environment import default_environment,environment_python


def main(args):
    if sys.version_info[:2]!=(3,11):
        raise ValueError('Use Python 3.11: python3.11 setup_environment.py, or py -3.11 setup_environment.py on Windows')
    target=Path(args.prefix).expanduser().resolve()
    python=environment_python(target)
    if not python.exists():
        if target.exists() and any(target.iterdir()):
            raise ValueError('Environment directory is not empty and has no Python; select another --prefix')
        print('Creating reusable environment:',target,flush=True)
        venv.EnvBuilder(with_pip=True).create(target)
    elif not (target/'pyvenv.cfg').is_file():
        raise ValueError('Target is not a virtual environment; activate an existing Conda environment instead')
    else:
        print('Reusing environment:',target,flush=True)
    version=subprocess.run([str(python),'-c','import sys; print("%s.%s" % sys.version_info[:2])'],capture_output=True,text=True,check=True).stdout.strip()
    if version!='3.11':
        raise ValueError('Existing environment uses Python '+version+'; choose another --prefix for Python 3.11')
    if not args.no_install:
        subprocess.run([str(python),'-m','pip','install','--upgrade','pip'],check=True)
        if args.pytorch_index:
            subprocess.run([str(python),'-m','pip','install','--upgrade','torch','--index-url',args.pytorch_index],check=True)
        requirements=Path(__file__).resolve().parent/'LoRACM'/('requirements.txt' if args.terminal_only else 'requirements-gui.txt')
        subprocess.run([str(python),'-m','pip','install','-r',str(requirements)],check=True)
        subprocess.run([str(python),'-m','pip','check'],check=True)
        versions=subprocess.run([str(python),'-m','pip','freeze'],capture_output=True,text=True,check=True).stdout
        (target/'pet-tracer-installed.txt').write_text(versions,encoding='utf-8')
    (target/'pet-tracer-environment.json').write_text(json.dumps({'python':str(python),'source_checkout':str(Path(__file__).resolve().parent),
        'reusable':True,'installation_skipped':args.no_install},indent=2)+'\n',encoding='utf-8')
    print('Ready. New downloads can use the same environment without reinstalling.')
    print('Python:',python)
    if target!=default_environment().resolve():
        print('For GUI discovery, set PET_TRACER_PYTHON to this Python path, or activate this environment.')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prefix',default=str(default_environment()),help='Default ~/.venvs/pet-tracer')
    parser.add_argument('--pytorch-index',help='Official PyTorch wheel index chosen using pytorch.org/get-started/locally/')
    parser.add_argument('--terminal-only',action='store_true')
    parser.add_argument('--no-install',action='store_true',help='Create/reuse only; install PyTorch and requirements yourself')
    try: main(parser.parse_args())
    except (OSError,ValueError,subprocess.CalledProcessError) as error:
        print('Setup stopped:',error,file=sys.stderr)
        raise SystemExit(1)
