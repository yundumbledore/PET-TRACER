"""Nonblocking local subprocess jobs; no shell interpolation."""
import json
import os
from pathlib import Path
import subprocess
import sys
import re
from datetime import datetime
import uuid
import platform
import traceback
from runtime import write_json_atomic


class Jobs:
    def __init__(self, root):
        self.root = Path(root).resolve()
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
        executable=Path(sys.executable)
        if executable.name.lower()=='pythonw.exe' and executable.with_name('python.exe').exists():
            executable=executable.with_name('python.exe')
        command = [str(executable),'-u','-X','faulthandler',str(self.root/'job_runner.py'),*map(str,arguments)]
        self.arguments = list(map(str,arguments))
        environment=dict(os.environ,PYTHONUTF8='1',PYTHONIOENCODING='utf-8',PYTHONFAULTHANDLER='1',
                         PET_TRACER_PROGRESS_FILE=str(self.progress_path),PET_TRACER_STOP_FILE=str(self.stop_path))
        self.process=None
        details={'label':label,'command':command,'platform':platform.platform(),'python_version':sys.version}
        write_json_atomic(self.log_path.with_suffix('.json'),details)
        try:
            with self.log_path.open('w',encoding='utf-8') as log:
                log.write('Launching '+label+'\n'+json.dumps(details,indent=2)+'\n');log.flush()
                self.process = subprocess.Popen(command,cwd=self.root,stdout=log,stderr=subprocess.STDOUT,env=environment)
        except OSError as error:
            if self.log_path.exists():
                with self.log_path.open('a',encoding='utf-8') as log: log.write(traceback.format_exc())
            raise OSError(f'Could not launch {label}: {error}. Log: {self.log_path}') from error
        # A diagnostic sidecar must not turn a successfully started job into
        # a reported launch failure if a Windows scanner briefly locks it.
        try: write_json_atomic(self.log_path.with_suffix('.json'),dict(details,pid=self.process.pid))
        except PermissionError:
            with self.log_path.open('a',encoding='utf-8') as log:
                log.write('Could not update the job metadata sidecar; the job started successfully.\n')

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
        try:
            with self.log_path.open('rb') as f:
                f.seek(max(0,self.log_path.stat().st_size-12000))
                raw=f.read().decode('utf-8',errors='replace')
                # Keep only the latest carriage-return update of historical tqdm logs.
                result=re.sub(r'\x1b\[[0-9;]*m','', '\n'.join(line.split('\r')[-1] for line in raw.split('\n')))
        except OSError as error:
            result=f'Job log could not be read yet: {self.log_path}. {error}. The monitor will try again.'
        if self.process is not None and self.process.poll() not in (None,0):
            result+='\nProcess exit code: '+str(self.process.returncode)+'\n'
            if 'Traceback' not in result:
                result+='The process exited without a Python traceback. Run Check compute setup and share this log; this can indicate a Python/dependency or native-library startup failure.\n'
        return result

    def full_log(self):
        try: return self.log_path.read_bytes()
        except OSError: return self.tail().encode('utf-8')
