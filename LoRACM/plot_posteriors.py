"""Posterior histograms with optional dashed nonlinear least-squares estimates."""
import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def posterior_figure(draws, names, nls=None, comparison_draws=None):
    draws = np.asarray(draws)
    if draws.ndim != 2 or draws.shape[1] != len(names) or not np.isfinite(draws).all():
        raise ValueError('Expected finite [samples, parameters] posterior draws')
    if nls is not None and nls['parameter_names'] != list(names):
        raise ValueError('NLS and posterior parameter names/order differ')
    if comparison_draws is not None:
        comparison_draws=np.asarray(comparison_draws)
        if comparison_draws.ndim!=2 or comparison_draws.shape[1]!=len(names) or not np.isfinite(comparison_draws).all():
            raise ValueError('Comparison draws must be finite [samples, parameters]')
    cols = min(3,len(names)); rows = (len(names)+cols-1)//cols
    fig,axes = plt.subplots(rows,cols,figsize=(4.3*cols,3.1*rows),squeeze=False,constrained_layout=True)
    for i,(ax,name) in enumerate(zip(axes.flat,names)):
        ax.hist(draws[:,i],bins=min(45,max(8,int(np.sqrt(len(draws))))),density=True,
                color='#2d8993',alpha=.65,label='LoRACM' if comparison_draws is not None else 'CM posterior samples')
        if comparison_draws is not None:
            ax.hist(comparison_draws[:,i],bins=min(45,max(8,int(np.sqrt(len(comparison_draws))))),density=True,
                    histtype='step',color='#a2613f',linewidth=1.7,label='CM from scratch')
        if nls is not None and nls['success']:
            ax.axvline(nls['parameters'][i],color='#243d56',linestyle='--',linewidth=1.8,label='Nonlinear least squares')
        ax.set(title=name,xlabel='Parameter value',ylabel='Density')
        ax.spines[['top','right']].set_visible(False)
        ax.legend(fontsize=8)
    for ax in list(axes.flat)[len(names):]: ax.set_visible(False)
    return fig


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--posterior',required=True);p.add_argument('--nls');p.add_argument('--output',required=True)
    p.add_argument('--comparison',help='Scratch CM posterior NPZ, same single TAC and parameter order')
    a=p.parse_args();output=Path(a.output)
    if output.exists(): raise FileExistsError('Figure output exists; choose another path')
    with np.load(a.posterior,allow_pickle=False) as data:
        if len(data['posterior_samples']) != 1: raise ValueError('Select a single TAC for posterior plotting')
        nls=json.loads(Path(a.nls).read_text(encoding='utf-8')) if a.nls else None
        comparison=None
        if a.comparison:
            with np.load(a.comparison,allow_pickle=False) as baseline:
                if not np.array_equal(data['parameter_names'],baseline['parameter_names']) or not np.array_equal(data['voxel_columns'],baseline['voxel_columns']):
                    raise ValueError('Comparison parameter order or curve identities differ')
                comparison=baseline['posterior_samples'][0]
        fig=posterior_figure(data['posterior_samples'][0],data['parameter_names'].tolist(),nls,comparison)
    output.parent.mkdir(parents=True,exist_ok=True);fig.savefig(output,dpi=180);plt.close(fig)
