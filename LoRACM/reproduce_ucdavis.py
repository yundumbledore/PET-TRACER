"""Reproduce the UC Davis Ki example using the supplied original HPC checkpoint."""
import argparse
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import matplotlib.pyplot as plt
from ucdavis import ROOT,prepare_paper_run
from infer import main as infer
from parametric_imaging import load_image,map_figure,display_plane
from common import next_prediction_output


def main(args):
    workspace=Path(args.workspace).resolve()
    run=prepare_paper_run(workspace)
    if args.prepare_only:
        print('Original paper checkpoint and scaling ready:',run)
        return
    output=next_prediction_output(workspace/'paper_predictions.npz')
    infer(SimpleNamespace(run=str(run),checkpoint='best.pth',input=str(ROOT/'data/real/Sub001_slice75_adjusted.h5'),
                          output=str(output),samples=args.samples,steps=args.steps,seed=args.seed,device=args.device,
                          batch_size=args.batch_size,sample_chunk=args.sample_chunk,multi_gpu=False,max_voxels=None,
                          allow_time_mismatch=False,save_samples=False))
    result=load_image(output,ROOT/'data/real/Sub001_slice75_geometry.npz',ROOT/'data/real/Sub001_slice75_geometry.json','Ki')
    image,display=display_plane(result,style='ucdavis_hpc')
    fig=map_figure(result,style='ucdavis_hpc');figure=output.with_name(output.stem+'_Ki.png')
    fig.savefig(figure,dpi=180);plt.close(fig)
    np.savez_compressed(output.with_name(output.stem+'_Ki_map.npz'),image=result['image'],display_image=image,voxel_ids=result['ids'],mask=result['mask'])
    output.with_name(output.stem+'_map.json').write_text(json.dumps(dict(display=display,predictions=str(output),
        checkpoint=str(run/'best.pth'),sampling='Fresh posterior draws; device-dependent Monte Carlo variation is expected',
        raw_values_preserved=True),indent=2)+'\n',encoding='utf-8')
    print('Saved paper-example map:',figure,flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workspace',default='workspaces/ucdavis_paper')
    parser.add_argument('--prepare-only',action='store_true')
    parser.add_argument('--device',choices=['auto','cpu','mps','cuda:0'],default='auto')
    parser.add_argument('--samples',type=int,default=1000)
    parser.add_argument('--steps',type=int,default=3)
    parser.add_argument('--seed',type=int,default=0)
    parser.add_argument('--batch-size',type=int,default=16)
    parser.add_argument('--sample-chunk',type=int,default=64)
    main(parser.parse_args())
