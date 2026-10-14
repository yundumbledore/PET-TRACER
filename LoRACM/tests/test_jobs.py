"""Windows-style file locks and child startup failures keep useful diagnostics."""
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from runtime import write_json_atomic
from gui.jobs import Jobs

ROOT=Path(__file__).resolve().parents[1]


class JobTests(unittest.TestCase):
    def test_atomic_json_retries_transient_permission_errors(self):
        with tempfile.TemporaryDirectory() as folder:
            output=Path(folder)/'settings.json';output.write_text('{"old":true}')
            replace=os.replace
            attempts=[]
            def locked(source,target):
                attempts.append(Path(source))
                if len(attempts)<3: raise PermissionError('[WinError 5] Access is denied')
                return replace(source,target)
            with patch('runtime.os.replace',side_effect=locked),patch('runtime.time.sleep'):
                write_json_atomic(output,{'saved':'tracér'})
            self.assertEqual(json.loads(output.read_text()),{'saved':'tracér'})
            self.assertEqual(len(attempts),3)
            self.assertFalse(list(Path(folder).glob('*.tmp')))

    def test_permanent_lock_preserves_previous_json_and_reports_path(self):
        with tempfile.TemporaryDirectory() as folder:
            output=Path(folder)/'settings.json';output.write_text('{"old":true}')
            with patch('runtime.os.replace',side_effect=PermissionError('locked')),patch('runtime.time.sleep'):
                with self.assertRaisesRegex(PermissionError,'Cannot update .*settings.json'):
                    write_json_atomic(output,{'new':True})
            self.assertEqual(json.loads(output.read_text()),{'old':True})
            self.assertFalse(list(Path(folder).glob('*.tmp')))

    def test_failed_job_captures_arguments_and_original_traceback(self):
        with tempfile.TemporaryDirectory() as folder:
            folder=Path(folder);script=folder/'script with spaces.py'
            script.write_text("import sys\nprint(sys.argv[1],flush=True)\nraise ValueError('fixture startup failure')\n")
            job=Jobs(ROOT);job.start('Failure fixture',[script,'tracér input with spaces'],folder)
            self.assertEqual(job.process.wait(timeout=20),1)
            log=job.tail()
            self.assertIn('PET-TRACER job starting',log);self.assertIn('fixture startup failure',log)
            self.assertIn('tracér input with spaces',log);self.assertIn('Traceback',log)
            self.assertIn('Process exit code: 1',log)

    def test_native_exit_still_has_startup_log(self):
        with tempfile.TemporaryDirectory() as folder:
            folder=Path(folder);script=folder/'native.py';script.write_text('import os\nos._exit(1)\n')
            job=Jobs(ROOT);job.start('Native exit',[script],folder)
            self.assertEqual(job.process.wait(timeout=20),1)
            self.assertIn('Launching Native exit',job.tail())
            self.assertIn('without a Python traceback',job.tail())

    def test_launch_permission_failure_does_not_reuse_previous_process(self):
        with tempfile.TemporaryDirectory() as folder:
            job=Jobs(ROOT)
            with patch('gui.jobs.subprocess.Popen',side_effect=PermissionError('[WinError 5] Access denied')):
                with self.assertRaisesRegex(OSError,'Could not launch .*Log:'):
                    job.start('Denied',['simulate_protocol.py'],folder)
            self.assertIsNone(job.process);self.assertIn('WinError 5',job.tail())


if __name__=='__main__': unittest.main()
