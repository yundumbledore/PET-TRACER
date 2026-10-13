"""Geometry-checked parameter and 95% credible-interval maps, in 2D or 3D."""
import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

STATISTICS = {'Posterior mean':'mean', 'Posterior median':'median', '95% credible-interval width':'ci95_width'}


def load_image(predictions, geometry, metadata, parameter=None, statistic='mean', allow_partial=False):
    meta=json.loads(Path(metadata).read_text(encoding='utf-8'))
    if not meta.get('mapping_verified') or not meta.get('orientation_and_spacing_verified'):
        raise ValueError('Geometry mapping and display convention must be verified by the data author')
    with np.load(geometry,allow_pickle=False) as source:
        mask=source['mask'].copy();indices=source['voxel_indices'].copy();ids=source['voxel_columns'].astype(str)
    if mask.ndim not in (2,3) or not np.isin(mask,[0,1]).all(): raise ValueError('Geometry needs a binary 2D or 3D mask')
    if indices.shape!=ids.shape or indices.ndim!=1 or not np.issubdtype(indices.dtype,np.integer):
        raise ValueError('Geometry indices and column IDs must be matching 1D arrays')
    if len(np.unique(ids))!=len(ids) or len(np.unique(indices))!=len(indices): raise ValueError('Duplicate geometry identities or indices')
    if np.any(indices<0) or np.any(indices>=mask.size) or not np.all(mask.ravel()[indices]==1):
        raise ValueError('Geometry index lies outside the foreground mask')
    if set(indices.tolist())!=set(np.flatnonzero(mask.ravel()).tolist()): raise ValueError('Geometry must identify every foreground voxel')
    if 'shape' in meta and list(mask.shape)!=meta['shape']: raise ValueError('Geometry metadata shape differs from the mask')
    if meta.get('origin','lower') not in ('lower','upper'): raise ValueError('Image origin must be lower or upper')
    aspect=float(meta.get('pixel_aspect_ratio',1.))
    if not np.isfinite(aspect) or aspect<=0: raise ValueError('Pixel aspect ratio must be positive')
    with np.load(predictions,allow_pickle=False) as source:
        names=source['parameter_names'].tolist();prediction_ids=source['voxel_columns'].astype(str)
        available=(['Ki'] if 'Ki_mean' in source else [])+names
        parameter=parameter or available[0]
        if parameter not in available: raise ValueError('Parameter is unavailable in these predictions')
        if statistic not in STATISTICS.values(): raise ValueError('Choose mean, median or ci95_width')
        key=('Ki_' if parameter=='Ki' and 'Ki_mean' in source else 'params_')+statistic
        if key not in source: raise ValueError('These predictions do not contain 95% credible intervals. Run prediction again with the updated code.')
        values=source[key].copy()
        if key.startswith('params_'): values=values[:,names.index(parameter)]
        provenance=json.loads(str(source['metadata_json'])) if 'metadata_json' in source else {}
    if values.shape!=prediction_ids.shape or len(np.unique(prediction_ids))!=len(prediction_ids):
        raise ValueError('Prediction values and unique voxel identities must match')
    lookup=dict(zip(ids.tolist(),indices.tolist()))
    if any(v not in lookup for v in prediction_ids): raise ValueError('Prediction contains voxel IDs absent from this geometry')
    if not allow_partial and set(prediction_ids)!=set(ids): raise ValueError('Complete imaging requires every foreground voxel; partial predictions are not a complete map')
    image=np.full(mask.size,np.nan);id_image=np.full(mask.size,'',dtype=ids.dtype)
    positions=np.asarray([lookup[v] for v in prediction_ids])
    image[positions]=values;id_image[positions]=prediction_ids
    return dict(image=image.reshape(mask.shape),ids=id_image.reshape(mask.shape),mask=mask,
                parameter=parameter,available=available,parameter_names=names,statistic=statistic,metadata=meta,
                provenance=provenance,coverage=len(prediction_ids)/len(ids))


def plane(image,axis=1,index=None):
    if image.ndim==2: return image
    if axis not in (0,1,2): raise ValueError('Slice axis must be 0, 1 or 2')
    index=image.shape[axis]//2 if index is None else index
    if not 0<=index<image.shape[axis]: raise ValueError('Slice index is out of range')
    return np.take(image,index,axis=axis)


def display_limits(image):
    values=image[np.isfinite(image)]
    if not len(values): raise ValueError('No finite voxel values to display')
    low,high=np.quantile(values,[.02,.98])
    if low==high: high=low+max(abs(low)*.01,1e-8)
    return float(low),float(high)


def map_figure(result,axis=1,index=None,vmin=None,vmax=None):
    image=plane(result['image'],axis,index)
    low,high=display_limits(image)
    fig,ax=plt.subplots(figsize=(5,8))
    shown=ax.imshow(image,origin=result['metadata'].get('origin','lower'),aspect=result['metadata'].get('pixel_aspect_ratio',1.),
                    cmap='inferno',vmin=low if vmin is None else vmin,vmax=high if vmax is None else vmax,interpolation='nearest')
    label=result['parameter']+(' · 95% CI width' if result['statistic']=='ci95_width' else ' · '+result['statistic'])
    ax.set_title(label);ax.set_xlabel('Column (pixels)');ax.set_ylabel('Row (pixels)')
    fig.colorbar(shown,ax=ax,label=label);fig.tight_layout()
    return fig


def parameter_samples(draws,names,parameter):
    if parameter=='Ki' and names==['K1','k2','k3','k4','Vb']:
        denominator=draws[:,1]+draws[:,2]
        valid=np.abs(denominator)>1e-12
        return draws[valid,0]*draws[valid,2]/denominator[valid]
    return draws[:,names.index(parameter)]


def density_figure(values,parameter):
    values=np.asarray(values);values=values[np.isfinite(values)]
    if len(values)<2: raise ValueError('Need at least two finite posterior samples')
    fig,ax=plt.subplots(figsize=(6,2.7))
    low,high=np.quantile(values,[.025,.975]);spread=np.std(values)
    if spread>0:
        x=np.linspace(values.min()-spread*.25,values.max()+spread*.25,350)
        try:
            density=gaussian_kde(values)(x);ax.plot(x,density,color='#247b84');ax.fill_between(x,density,color='#247b84',alpha=.12)
        except np.linalg.LinAlgError: ax.hist(values,bins='auto',density=True,color='#247b84',alpha=.55)
    else: ax.axvline(values[0],color='#247b84')
    ax.axvspan(low,high,color='#247b84',alpha=.12,label='95% equal-tailed interval')
    ax.axvline(np.median(values),color='#153d50',linestyle='--',label='Median')
    ax.set(xlabel=parameter,ylabel='Density',title=f'{parameter} posterior');ax.legend(fontsize=8);fig.tight_layout()
    return fig


def main():
    root=Path(__file__).resolve().parent
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--predictions',required=True)
    parser.add_argument('--geometry',default=str(root/'data/real/Sub001_slice75_geometry.npz'))
    parser.add_argument('--geometry-metadata',default=str(root/'data/real/Sub001_slice75_geometry.json'))
    parser.add_argument('--parameter',default=None);parser.add_argument('--statistic',choices=list(STATISTICS.values()),default='mean')
    parser.add_argument('--axis',type=int,default=1);parser.add_argument('--slice',type=int)
    parser.add_argument('--output',required=True);args=parser.parse_args()
    output=Path(args.output)
    if output.suffix.lower()!='.png': raise ValueError('Figure output must end in .png')
    if any(p.exists() for p in [output,output.with_suffix('.npz'),output.with_suffix('.json')]): raise FileExistsError('Map output exists')
    result=load_image(args.predictions,args.geometry,args.geometry_metadata,args.parameter,args.statistic)
    fig=map_figure(result,args.axis,args.slice);output.parent.mkdir(parents=True,exist_ok=True)
    fig.savefig(output,dpi=180);plt.close(fig)
    np.savez_compressed(output.with_suffix('.npz'),image=result['image'],voxel_ids=result['ids'],mask=result['mask'])
    report=dict(predictions=args.predictions,geometry=args.geometry,parameter=result['parameter'],statistic=args.statistic,
                display_limits='2nd/98th percentiles for colour scale only; map values are unmodified',
                credible_interval='97.5th minus 2.5th percentile, not standard deviation',geometry_metadata=result['metadata'])
    output.with_suffix('.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    print(f'Saved {output}')


if __name__=='__main__': main()
