"""Start the local GUI from any directory on macOS, Windows or Linux."""
from pathlib import Path
import subprocess
import sys

root = Path(__file__).resolve().parent
venv = root/'.venv'/('Scripts/python.exe' if sys.platform=='win32' else 'bin/python')
python = str(venv) if venv.exists() else sys.executable
check = subprocess.run([python,'-c','import streamlit'],capture_output=True,text=True)
if check.returncode:
    print('GUI dependencies are missing. Follow README.md installation, then run launch_gui.py again.')
    raise SystemExit(1)
raise SystemExit(subprocess.call([python,'-m','streamlit','run',str(root/'LoRACM/gui/app.py'),
                                 '--server.address','127.0.0.1','--browser.gatherUsageStats','false'],cwd=root))
