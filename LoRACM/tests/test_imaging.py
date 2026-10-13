"""Voxel identities, empirical interval widths and geometry are scientific contracts."""
import json
from pathlib import Path
import sys
import tempfile
import unittest
import numpy as np
import pandas as pd
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from common import summarize_samples
from parametric_imaging import load_image,plane,parameter_samples
from ucdavis import defaults
from protocol import validate_protocol

ROOT=Path(__file__).resolve().parents[1]


class ImagingTests(unittest.TestCase):
    def test_empirical_interval_widths_include_joint_ki(self):
        draws=np.array([[[1,1,1,.1,.1],[2,3,1,.2,.2],[9,1,3,.3,.3]]],dtype=float)
        summary=summarize_samples(draws)
        expected=np.quantile(draws,[.025,.975],axis=1)
        np.testing.assert_allclose(summary['params_ci95_width'],expected[1]-expected[0])
        ki=np.array([.5,.5,6.75]);interval=np.quantile(ki,[.025,.975])
        self.assertAlmostEqual(summary['Ki_ci95_width'][0],interval[1]-interval[0])
        np.testing.assert_array_equal(parameter_samples(draws[0],['K1','k2','k3','k4','Vb'],'Ki'),ki)

    def test_mapping_uses_identity_not_prediction_row_order(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);geometry=root/'g.npz';metadata=root/'g.json';prediction=root/'p.npz'
            np.savez(geometry,mask=np.array([[0,1],[1,1]]),voxel_indices=[1,2,3],voxel_columns=['a','b','c'])
            meta={'mapping_verified':True,'orientation_and_spacing_verified':True,'origin':'lower','pixel_aspect_ratio':1.}
            metadata.write_text(json.dumps(meta))
            np.savez(prediction,parameter_names=['rate'],voxel_columns=['c','a','b'],params_mean=[[8],[3],[5]],params_ci95_width=[[.8],[.3],[.5]])
            result=load_image(prediction,geometry,metadata,'rate','ci95_width')
            np.testing.assert_allclose(result['image'][1],[.5,.8]);self.assertEqual(result['image'][0,1],.3)
            self.assertEqual(result['ids'][1,1],'c');self.assertTrue(np.isnan(result['image'][0,0]))
            meta['mapping_verified']=False;metadata.write_text(json.dumps(meta))
            with self.assertRaisesRegex(ValueError,'verified'): load_image(prediction,geometry,metadata)

    def test_full_maps_reject_missing_or_unknown_voxels(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);g=root/'g.npz';m=root/'g.json';p=root/'p.npz'
            np.savez(g,mask=np.ones((1,2)),voxel_indices=[0,1],voxel_columns=['a','b'])
            m.write_text(json.dumps({'mapping_verified':True,'orientation_and_spacing_verified':True}))
            for identity in ('a','z'):
                np.savez(p,parameter_names=['rate'],voxel_columns=[identity],params_mean=[[1]])
                with self.assertRaises(ValueError): load_image(p,g,m)

    def test_volume_slice_preserves_coordinates(self):
        volume=np.arange(24).reshape(2,3,4)
        np.testing.assert_array_equal(plane(volume,1,2),volume[:,2,:])
        with self.assertRaises(ValueError): plane(volume,1,3)

    def test_ucdavis_defaults_match_measured_prediction_schedule(self):
        cfg,training=defaults(Path('workspaces/test_fixture'))
        t,d=validate_protocol(cfg)
        patient=pd.read_hdf(ROOT/'data/real/Sub001_slice75_adjusted.h5')
        np.testing.assert_allclose(t,patient.iloc[:,1],atol=1e-12)
        np.testing.assert_allclose(d,patient.iloc[:,0],atol=1e-12)
        aif=pd.read_csv(cfg['aif_csv'])
        self.assertEqual(aif.shape[1],5);self.assertEqual(aif.iloc[0,0],0);self.assertEqual(aif.iloc[-1,0],60)
        self.assertEqual((cfg['n_train'],cfg['n_val'],cfg['n_test']),(50000,100000,100))
        self.assertEqual((training['lora_r'],training['lora_alpha']),(4,8))
        small,short=defaults(Path('workspaces/test_fixture'),small=True)
        self.assertEqual((small['n_train'],short['epochs'],short['checkpoint_start_epoch']),(32,2,1))


if __name__=='__main__': unittest.main()
