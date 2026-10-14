"""Patient masks, identities and image-source checksums remain inseparable."""
import json
from pathlib import Path
import sys
import tempfile
import unittest
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from common import file_sha256
from prepare_scan import prepare, load_mask
from parametric_imaging import load_image


class PatientScanTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.root=Path(self.temp.name)
        self.source=self.root/'patient.npz';self.mask=self.root/'mask.npy'
        np.savez(self.source,tac=np.ones((3,2)),aif=np.ones(2),t_meas=[.5,1.5],frame_durations_min=[1,1],voxel_columns=['a','b','c'])
        np.save(self.mask,np.array([[0,1],[1,1]]))

    def tearDown(self): self.temp.cleanup()

    def prediction(self,input_hash):
        file=self.root/'prediction.npz'
        np.savez(file,parameter_names=['K1','k2','k3','k4','Vb'],voxel_columns=['c','a','b'],
                 params_mean=np.array([[8]*5,[3]*5,[5]*5]),params_ci95_width=np.array([[.8]*5,[.3]*5,[.5]*5]),
                 Ki_mean=[80,30,50],Ki_ci95_width=[8,3,5],metadata_json=json.dumps({'input_sha256':input_hash}))
        return file

    def test_new_patient_maps_parameters_ki_and_uncertainty_by_id(self):
        geometry,metadata=prepare(self.source,self.mask,self.root/'scan',verified=True)
        prediction=self.prediction(file_sha256(self.source))
        for parameter,stat,expected in [('Ki','mean',30),('Ki','ci95_width',3),('K1','mean',3),('K1','ci95_width',.3)]:
            result=load_image(prediction,geometry,metadata,parameter,stat)
            self.assertAlmostEqual(result['image'][0,1],expected)
            self.assertEqual(result['ids'][1,1],'c')

    def test_same_voxel_ids_do_not_allow_a_different_patient(self):
        geometry,metadata=prepare(self.source,self.mask,self.root/'scan',verified=True)
        prediction=self.prediction('different-patient-hash')
        with self.assertRaisesRegex(ValueError,'different patient input file'):
            load_image(prediction,geometry,metadata)

    def test_mask_mismatch_and_unverified_mapping_are_rejected(self):
        with self.assertRaisesRegex(ValueError,'Verify the TAC-to-mask'):
            prepare(self.source,self.mask,self.root/'scan')
        np.save(self.mask,np.ones((2,2)))
        with self.assertRaisesRegex(ValueError,'3 TACs.*4 foreground'):
            prepare(self.source,self.mask,self.root/'scan',verified=True)
        self.assertFalse((self.root/'scan').exists())

    def test_explicit_mapping_and_3d_slice(self):
        volume=self.root/'volume.npy';np.save(volume,np.stack([np.zeros((2,2)),np.array([[0,1],[1,1]])],axis=1))
        np.testing.assert_array_equal(load_mask(volume,axis=1,index=1),np.load(self.mask))
        with self.assertRaises(ValueError):load_mask(volume)
        mapping=self.root/'mapping.csv';mapping.write_text('voxel_columns,voxel_indices\nc,1\na,3\nb,2\n')
        geometry,_=prepare(self.source,volume,self.root/'scan',verified=True,mapping_csv=mapping,axis=1,index=1)
        with np.load(geometry) as data: np.testing.assert_array_equal(data['voxel_indices'],[3,2,1])


if __name__=='__main__': unittest.main()
