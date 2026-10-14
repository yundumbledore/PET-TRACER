"""Bind a patient's TAC slice to its mask and verified voxel mapping."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from common import file_sha256
from protocol import read_prediction_input
from runtime import write_json_atomic


def load_mask(filename,axis=1,index=None):
    filename=Path(filename)
    if filename.suffix.lower()=='.npy': mask=np.load(filename,allow_pickle=False)
    elif filename.suffix.lower()=='.npz':
        with np.load(filename,allow_pickle=False) as data: mask=data['mask'].copy()
    else: raise ValueError('Supply a NumPy mask: .npy, or .npz containing mask')
    if mask.ndim==3:
        if axis not in (0,1,2) or index is None or not 0<=index<mask.shape[axis]:
            raise ValueError('For a 3D mask, choose its array slice axis and slice index')
        mask=np.take(mask,index,axis=axis)
    if mask.ndim!=2 or not np.isin(mask,[0,1]).all() or not np.any(mask):
        raise ValueError('The selected coronal mask must be a nonempty binary 2D array')
    return mask


def prepare(input_path,mask_path,output,parameter_dim=5,expected_times=None,expected_durations=None,
            verified=False,mapping_csv=None,origin='lower',aspect=1.,axis=1,index=None):
    if not verified: raise ValueError('Verify the TAC-to-mask voxel order, image orientation and pixel aspect ratio before preparing this patient scan')
    if origin not in ('lower','upper') or not np.isfinite(aspect) or aspect<=0:
        raise ValueError('Choose upper/lower image origin and a finite positive pixel aspect ratio')
    _,_,_,columns,_=read_prediction_input(input_path,parameter_dim,expected_times,expected_durations)
    mask=load_mask(mask_path,axis,index)
    foreground=np.flatnonzero(mask.ravel(order='C'))
    if len(columns)!=len(foreground):
        raise ValueError(f'Patient slice contains {len(columns)} TACs but its mask contains {len(foreground)} foreground pixels; provide the corresponding slice and mask')
    if mapping_csv is None:
        indices=foreground
        convention='Input TAC columns follow mask foreground pixels in NumPy C order; confirmed by user'
    else:
        table=pd.read_csv(mapping_csv,dtype={'voxel_columns':str})
        ids=table['voxel_columns'].to_numpy(dtype=str)
        raw=table['voxel_indices'].to_numpy(dtype=float)
        if len(ids)!=len(columns) or len(set(ids))!=len(ids) or set(ids)!=set(columns):
            raise ValueError('Mapping CSV must list every unique input voxel_columns identifier exactly once')
        if not np.isfinite(raw).all() or not np.equal(raw,np.floor(raw)).all(): raise ValueError('Mapping voxel_indices must be finite integers')
        lookup=dict(zip(ids,raw.astype(np.int64)))
        indices=np.asarray([lookup[value] for value in columns])
        if len(np.unique(indices))!=len(indices) or set(indices)!=set(foreground):
            raise ValueError('Mapping must identify every foreground C-order flat index exactly once')
        convention='Explicit voxel_columns/voxel_indices CSV; confirmed by user'
    output=Path(output)
    output.mkdir(parents=True,exist_ok=False)
    geometry=output/'geometry.npz';metadata=output/'geometry.json'
    with geometry.open('xb') as stream: np.savez_compressed(stream,mask=mask,voxel_indices=indices,voxel_columns=columns)
    info={'mapping_verified':True,'orientation_and_spacing_verified':True,'mapping_source':convention,
          'shape':list(mask.shape),'origin':origin,'pixel_aspect_ratio':float(aspect),'foreground_voxels':len(columns),
          'input':str(Path(input_path).resolve()),'input_sha256':file_sha256(input_path),
          'mask_source':str(Path(mask_path).resolve()),'mask_sha256':file_sha256(mask_path),
          'mask_axis':axis if index is not None else None,'mask_slice_index':index}
    if mapping_csv is not None: info['mapping_sha256']=file_sha256(mapping_csv)
    write_json_atomic(metadata,info)
    return geometry,metadata


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',required=True);parser.add_argument('--mask',required=True);parser.add_argument('--output',required=True)
    parser.add_argument('--parameter-dim',type=int,default=5)
    parser.add_argument('--mapping-csv',help='Explicit voxel_columns,voxel_indices mapping; otherwise foreground C order')
    parser.add_argument('--verified',action='store_true',help='Confirm voxel mapping, image orientation and pixel aspect ratio')
    parser.add_argument('--origin',choices=['lower','upper'],default='lower');parser.add_argument('--aspect',type=float,default=1.)
    parser.add_argument('--axis',type=int,default=1);parser.add_argument('--slice',type=int)
    args=parser.parse_args()
    files=prepare(args.input,args.mask,args.output,args.parameter_dim,verified=args.verified,mapping_csv=args.mapping_csv,
                  origin=args.origin,aspect=args.aspect,axis=args.axis,index=args.slice)
    print('Prepared patient geometry: '+', '.join(map(str,files)))


if __name__=='__main__': main()
