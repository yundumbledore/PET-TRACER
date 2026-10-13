"""Nonblocking local subprocess jobs; no shell interpolation."""
import json
import os
from pathlib import Path
import subprocess
import sys
from datetime import datetime
import uuid


class Jobs:
    def __init__(self, root):
        self.root = Path(root)
        self.process = None
        self.log_path = None
        self.label = None

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
        command = [sys.executable,'-u',*map(str,arguments)]
        environment=dict(os.environ,PYTHONUTF8='1',PYTHONIOENCODING='utf-8')
        with self.log_path.open('w',encoding='utf-8') as log:
            self.process = subprocess.Popen(command,cwd=self.root,stdout=log,stderr=subprocess.STDOUT,env=environment)
        self.log_path.with_suffix('.json').write_text(json.dumps({'label':label,'command':command,'pid':self.process.pid},indent=2), encoding='utf-8')

    def stop(self):
        if self.running():
            self.process.terminate()

    def tail(self):
        if self.log_path is None:
            return ''
        with self.log_path.open('rb') as f:
            f.seek(max(0,self.log_path.stat().st_size-12000))
            return f.read().decode('utf-8',errors='replace')
