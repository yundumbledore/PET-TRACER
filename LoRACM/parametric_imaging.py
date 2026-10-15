"""Geometry-checked parameter and 95% credible-interval maps, in 2D or 3D."""
import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from scipy.ndimage import median_filter
from common import ki_samples

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
    if meta.get('input_sha256') and meta['input_sha256']!=provenance.get('input_sha256'):
        raise ValueError('This mask/mapping belongs to a different patient input file. Select the geometry prepared for the scan that produced these predictions.')
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


def display_plane(result,axis=1,index=None,style='auto',vmin=None,vmax=None):
    """Display filtering is explicit; raw posterior summaries are preserved."""
    image=plane(result['image'],axis,index).copy()
    if style == 'auto':
        style=result.get('provenance',{}).get('config',{}).get('map_display',result['metadata'].get('display_profile','raw'))
    if style not in ('raw','ucdavis_hpc'): raise ValueError('Choose raw or ucdavis_hpc display')
    filtered=style=='ucdavis_hpc'
    if filtered:
        valid=np.isfinite(image)
        image=median_filter(np.where(valid,image,0.),size=2)
        image[~valid]=np.nan
    low,high=display_limits(image)
    if filtered and result['parameter']=='Ki' and result['statistic']!='ci95_width': low,high=0.,.06
    low=low if vmin is None else float(vmin)
    high=high if vmax is None else float(vmax)
    if not np.isfinite([low,high]).all() or high<=low:
        raise ValueError('Colourbar limits must be finite, with maximum greater than minimum')
    return image,dict(style=style,median_filter_size=2 if filtered else 0,vmin=low,vmax=high,
                      note='Display only; raw voxel estimates and posterior samples are unchanged')


def map_figure(result,axis=1,index=None,vmin=None,vmax=None,style='auto'):
    image,settings=display_plane(result,axis,index,style,vmin,vmax)
    low,high=settings['vmin'],settings['vmax']
    fig,ax=plt.subplots(figsize=(5,8))
    cmap=plt.get_cmap('inferno').copy()
    if settings['style']=='ucdavis_hpc': cmap.set_bad(cmap(0.))
    shown=ax.imshow(image,origin=result['metadata'].get('origin','lower'),aspect=result['metadata'].get('pixel_aspect_ratio',1.),
                    cmap=cmap,vmin=low,vmax=high,interpolation='nearest')
    label=result['parameter']+(' · 95% CI width' if result['statistic']=='ci95_width' else ' · '+result['statistic'])
    ax.set_title(label);ax.set_xlabel('Column (pixels)');ax.set_ylabel('Row (pixels)')
    if settings['style']=='ucdavis_hpc': ax.set_axis_off()
    fig.colorbar(shown,ax=ax,label=label);fig.tight_layout()
    return fig


def parameter_samples(draws,names,parameter,ki_policy='raw'):
    if parameter=='Ki' and names==['K1','k2','k3','k4','Vb']:
        values=ki_samples(draws,ki_policy)
        return values[np.isfinite(values)]
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
    parser.add_argument('--display',choices=['auto','ucdavis_hpc','raw'],default='auto',help='UC Davis: original 2x2 median filter and Ki scale 0–0.06; raw: no filtering')
    parser.add_argument('--vmin',type=float,help='Colourbar minimum (otherwise use the display default)')
    parser.add_argument('--vmax',type=float,help='Colourbar maximum (otherwise use the display default)')
    parser.add_argument('--output',required=True);args=parser.parse_args()
    output=Path(args.output)
    if output.suffix.lower()!='.png': raise ValueError('Figure output must end in .png')
    if any(p.exists() for p in [output,output.with_suffix('.npz'),output.with_suffix('.json')]): raise FileExistsError('Map output exists')
    result=load_image(args.predictions,args.geometry,args.geometry_metadata,args.parameter,args.statistic)
    displayed,settings=display_plane(result,args.axis,args.slice,args.display,args.vmin,args.vmax)
    fig=map_figure(result,args.axis,args.slice,args.vmin,args.vmax,style=args.display);output.parent.mkdir(parents=True,exist_ok=True)
    fig.savefig(output,dpi=180);plt.close(fig)
    np.savez_compressed(output.with_suffix('.npz'),image=result['image'],display_image=displayed,display_settings=json.dumps(settings),voxel_ids=result['ids'],mask=result['mask'])
    report=dict(predictions=args.predictions,geometry=args.geometry,parameter=result['parameter'],statistic=args.statistic,
                display=settings,raw_values_preserved=True,
                credible_interval='97.5th minus 2.5th percentile, not standard deviation',geometry_metadata=result['metadata'])
    output.with_suffix('.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    print(f'Saved {output}')


if __name__=='__main__': main()
