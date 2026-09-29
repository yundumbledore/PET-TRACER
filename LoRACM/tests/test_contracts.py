import sys
from pathlib import Path
import unittest
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from common import summarize_samples, reconstruct_map, validate_dataset

class Contracts(unittest.TestCase):
    def test_joint_ki_and_statistic_labels(self):
        draws=np.array([[[1.,1.,1.,.1,.1],[2.,3.,1.,.2,.2],[9.,1.,3.,.3,.3]]])
        result=summarize_samples(draws)
        self.assertAlmostEqual(result['params_mean'][0,0],4.)
        self.assertAlmostEqual(result['params_median'][0,0],2.)
        expected=np.array([.5,.5,6.75])
        self.assertAlmostEqual(result['Ki_mean'][0],expected.mean())
        self.assertAlmostEqual(result['Ki_median'][0],.5)
        self.assertAlmostEqual(result['Ki_std'][0],expected.std())

    def test_near_zero_denominator_reported(self):
        result=summarize_samples(np.array([[[1.,0.,0.,0.,0.],[1.,1.,1.,0.,0.]]]))
        self.assertEqual(result['Ki_valid_fraction'][0],.5)
        self.assertEqual(result['Ki_mean'][0],.5)

    def test_explicit_mapping_preserves_order(self):
        mask=np.array([[0,1],[1,1]])
        image=reconstruct_map([8,3,5],mask,np.array([3,1,2]))
        np.testing.assert_array_equal(image[mask==1],[3,5,8])
        self.assertTrue(np.isnan(image[0,0]))

    def test_duplicate_and_background_indices_rejected(self):
        mask=np.array([[0,1],[1,1]])
        for indices in [np.array([1,1]),np.array([0,2])]:
            with self.assertRaises(ValueError): reconstruct_map([1,2],mask,indices)

    def test_partial_map_has_missing_values(self):
        image=reconstruct_map([9],np.ones((2,2)),np.array([2]))
        self.assertEqual(np.isfinite(image).sum(),1)
        self.assertEqual(image[1,0],9)

    def test_bad_conditioning_width_rejected(self):
        data={'t_meas':np.arange(1.,36.)}
        for split in ('train','val','test'):
            data['x_'+split]=np.ones((2,6));data['y_'+split]=np.ones((2,69))
        with self.assertRaises(ValueError):validate_dataset(data)

if __name__=='__main__':unittest.main()
