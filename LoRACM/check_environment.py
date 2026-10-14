"""Report the active Python/PyTorch environment and test the chosen device."""
import argparse
import json
import platform
import sys
import traceback
from runtime import select_device, probe_device, write_json_atomic


def main(args):
    report={'python':sys.executable,'python_version':sys.version,'platform':platform.platform(),'requested_device':args.device}
    try:
        import torch
        report.update(torch_version=str(torch.__version__),cuda_build=torch.version.cuda,
                      cuda_available=torch.cuda.is_available(),mps_available=torch.backends.mps.is_available())
        report['compute']=probe_device(select_device(args.device))
        report['state']='passed'
    except Exception as error:
        report.update(state='failed',error=str(error),traceback=traceback.format_exc())
    write_json_atomic(args.output,report)
    print(json.dumps(report,indent=2),flush=True)
    if report['state']!='passed': raise SystemExit(1)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device',default='auto',choices=['auto','cpu','cuda:0','mps'])
    parser.add_argument('--output',default='environment-check.json')
    main(parser.parse_args())
