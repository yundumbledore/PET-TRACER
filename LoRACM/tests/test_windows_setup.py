"""Exercise the double-click installer's PowerShell 5.1 paths on Windows CI."""
import os
from pathlib import Path
import shutil
import subprocess
import sys
import unittest


@unittest.skipUnless(os.name == 'nt', 'Windows PowerShell installer checks require Windows')
class WindowsSetupTests(unittest.TestCase):
    def test_setup_success_failure_bootstrap_and_paths(self):
        shell = shutil.which('powershell.exe')
        self.assertIsNotNone(shell, 'Windows PowerShell 5.1 must be available')
        script = Path(__file__).with_name('windows_setup_checks.ps1')
        completed = subprocess.run(
            [shell, '-NoProfile', '-ExecutionPolicy', 'Bypass', '-File', str(script), '-TestPython', sys.executable],
            capture_output=True, text=True, timeout=90,
        )
        self.assertEqual(completed.returncode, 0, completed.stdout + '\n' + completed.stderr)
        self.assertIn('Windows setup checks passed', completed.stdout)
