"""Launch this checkout with a reusable environment on all supported systems."""
from pathlib import Path
import subprocess
from pet_environment import find_gui_python

def main():
    root=Path(__file__).resolve().parent
    try:
        python=find_gui_python(root)
    except (OSError,RuntimeError) as error:
        print(str(error))
        return 1
    print('PET-TRACER Python:',python,flush=True)
    return subprocess.call([str(python),'-m','streamlit','run',str(root/'LoRACM/gui/app.py'),
                            '--server.address','127.0.0.1','--browser.gatherUsageStats','false'],cwd=root)


if __name__=='__main__': raise SystemExit(main())
