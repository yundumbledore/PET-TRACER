"""Run a GUI job with diagnostics before importing scientific dependencies."""
import faulthandler
import json
from pathlib import Path
import platform
import runpy
import sys
import traceback


def main():
    faulthandler.enable()
    arguments=sys.argv[1:]
    print('PET-TRACER job starting',flush=True)
    print(json.dumps({'python':sys.executable,'python_version':sys.version,'platform':platform.platform(),
                      'working_directory':str(Path.cwd()),'arguments':arguments},indent=2),flush=True)
    if not arguments: raise ValueError('A job script is required')
    script=Path(arguments[0]).resolve()
    sys.argv=[str(script),*arguments[1:]]
    sys.path.insert(0,str(script.parent))
    try:
        runpy.run_path(str(script),run_name='__main__')
    except SystemExit as error:
        print(f'Job requested exit: {error.code}',flush=True)
        raise
    except BaseException as error:
        traceback.print_exc()
        if 'out of memory' in str(error).lower():
            print('Memory limit reached. Reduce the training batch size (try 64) or the prediction curve batch/sample chunk, then use a new run/output.',flush=True)
        print('Job failed; the traceback above contains the original error.',flush=True)
        raise SystemExit(1)
    print('PET-TRACER job completed',flush=True)


if __name__=='__main__': main()
