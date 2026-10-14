"""Export measured TAC/AIF, posterior-median forward curves and an optional NLS fit."""
import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from protocol import read_prediction_input


def main(args):
    output=Path(args.output)
    if output.exists(): raise FileExistsError('Figure exists; choose a new output')
    forward=json.loads(Path(args.forward).read_text(encoding='utf-8'))
    if not forward['curves']: raise ValueError('Forward file has no model results')
    names=forward['curves'][0]['parameter_names']
    t,d,y,ids,_=read_prediction_input(args.input,len(names))
    if len(y)!=1: raise ValueError('TAC figure requires exactly one curve')
    fig,ax=plt.subplots(figsize=(8,4.5),layout='constrained')
    ax.plot(t,y[0,:len(t)],'o-',markersize=3,color='#247b84',label='Measured TAC')
    ax.plot(t,y[0,len(t):],color='#9cabb4',label='Input function')
    for item in forward['curves']:
        if item['parameter_names']!=names: raise ValueError('Forward model parameter orders differ')
        if item['fitted_tac'] is not None:
            label='LoRACM' if item['model']=='lora' else 'CM from scratch'
            ax.plot(t,item['fitted_tac'],label=label+' posterior-median forward TAC')
    if args.nls:
        nls=json.loads(Path(args.nls).read_text(encoding='utf-8'))
        if nls['parameter_names']!=names: raise ValueError('NLS parameter order differs')
        ax.plot(t,nls['fitted_tac'],'--',color='#243d56',label='NLS fitted TAC')
    ax.set(xlabel='Time (minutes)',ylabel='Activity (input units)',title=f'TAC {ids[0]}')
    ax.spines[['top','right']].set_visible(False);ax.legend(fontsize=8)
    output.parent.mkdir(parents=True,exist_ok=True);fig.savefig(output,dpi=180);plt.close(fig)
    print(f'Saved {output}',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',required=True)
    parser.add_argument('--forward',required=True)
    parser.add_argument('--nls')
    parser.add_argument('--output',required=True)
    main(parser.parse_args())
