"""Nonblocking local subprocess jobs; no shell interpolation."""
import json
import os
from pathlib import Path
import subprocess
import sys
import re
from datetime import datetime
import uuid


class Jobs:
    def __init__(self, root):
        self.root = Path(root)
        self.process = None
        self.log_path = None
        self.label = None
        self.workspace = None
        self.progress_path = None
        self.stop_path = None
        self.arguments = []

    def running(self):
        return self.process is not None and self.process.poll() is None

    def start(self, label, arguments, workspace):
        if self.running():
            raise ValueError('A job is already running; wait or stop it first')
        folder = Path(workspace)/'jobs'
        folder.mkdir(parents=True,exist_ok=True)
        stamp = datetime.now().strftime('%Y%m%d-%H%M%S')+'-'+uuid.uuid4().hex[:6]
        self.log_path = folder/(stamp+'.log')
        self.label = label
        self.workspace = Path(workspace)
        self.progress_path = folder/(stamp+'.progress.json')
        self.stop_path = folder/(stamp+'.stop')
        command = [sys.executable,'-u',*map(str,arguments)]
        self.arguments = list(map(str,arguments))
        environment=dict(os.environ,PYTHONUTF8='1',PYTHONIOENCODING='utf-8',
                         PET_TRACER_PROGRESS_FILE=str(self.progress_path),PET_TRACER_STOP_FILE=str(self.stop_path))
        with self.log_path.open('w',encoding='utf-8') as log:
            self.process = subprocess.Popen(command,cwd=self.root,stdout=log,stderr=subprocess.STDOUT,env=environment)
        self.log_path.with_suffix('.json').write_text(json.dumps({'label':label,'command':command,'pid':self.process.pid},indent=2), encoding='utf-8')

    def stop(self):
        if self.running():
            self.stop_path.touch()

    def progress(self):
        if self.progress_path and self.progress_path.exists():
            try: return json.loads(self.progress_path.read_text(encoding='utf-8'))
            except (OSError,json.JSONDecodeError): pass
        return None

    def tail(self):
        if self.log_path is None:
            return ''
        with self.log_path.open('rb') as f:
            f.seek(max(0,self.log_path.stat().st_size-12000))
            raw=f.read().decode('utf-8',errors='replace')
            # Keep only the latest carriage-return update of historical tqdm logs.
            return re.sub(r'\x1b\[[0-9;]*m','', '\n'.join(line.split('\r')[-1] for line in raw.split('\n')))
