"""Independent numeric and contract checks for the reusable workflow."""
import json
from pathlib import Path
import sys
import tempfile
import unittest
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from protocol import validate_protocol, validate_file, read_prediction_input, load_model, feng_curve, load_measured_inputs
from simulate_protocol import integrate, observe_frames, main as simulate
from analyse_tac import fit, predict_curve
from common import summarize_samples, validate_dataset
from ucdavis import defaults

ROOT=Path(__file__).resolve().parents[1]


def config():
    return json.loads((ROOT/'configs/dataset_custom.json').read_text())


class ProtocolTests(unittest.TestCase):
    def test_overlapping_frames_rejected(self):
        cfg=config();cfg['frame_durations_min'][1]=3
        with self.assertRaisesRegex(ValueError,'overlap'):validate_protocol(cfg)

    def test_infinite_times_rejected(self):
        cfg=config();cfg['t_meas'][-1]=np.inf
        with self.assertRaises(ValueError):validate_protocol(cfg)

    def test_duration_noise_and_names_contract(self):
        with tempfile.TemporaryDirectory() as folder:
            arrays={'t_meas':np.array([.5,1.5]),'frame_durations_min':np.ones(2),
                    'parameter_names':np.array(['a','b'])}
            for split in ('train','val','test'):
                arrays['x_'+split]=np.array([[1,2,0],[2,3,.1]])
                arrays['y_'+split]=np.array([[1,2,3,4],[2,3,4,5]])
            path=Path(folder)/'pairs.npz';np.savez(path,**arrays)
            self.assertEqual(validate_file(path)['x_dim'],2)
            arrays['x_train'][0,-1]=-1
            with self.assertRaisesRegex(ValueError,'noise'):validate_dataset(arrays,2,4)

    def test_custom_model_rk4_matches_analytical_1tc(self):
        cfg=config();cfg.update(model='custom',model_file=str(ROOT/'examples/custom_1tc.py'),parameter_names=['K1','k2','Vb'])
        model=load_model(cfg)
        grid=np.linspace(0,3,301)
        p=np.array([[.2,.4,.1],[.5,.2,.2]])
        inputs=np.full((2,len(grid)),10.)
        result=integrate(model,p,inputs,grid)
        expected=p[:,2:3]*10+(1-p[:,2:3])*p[:,0:1]*10/p[:,1:2]*(1-np.exp(-p[:,1:2]*grid))
        np.testing.assert_allclose(result,expected,atol=1e-9,rtol=1e-9)

    def test_frame_average_exact_for_linear_signal(self):
        grid=np.linspace(0,3,301)
        result=observe_frames((2*grid+1)[None,:],grid,np.array([.5,1.5,2.5]),np.ones(3),'frame_average')
        np.testing.assert_allclose(result,[[2,4,6]],atol=1e-12)

    def test_prediction_time_and_duration_mismatch_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'test.npz'
            np.savez(path,tac=np.ones((1,2)),aif=np.ones(2),t_meas=[.5,1.5],frame_durations_min=[1,1])
            with self.assertRaisesRegex(ValueError,'schedule'):read_prediction_input(path,5,[.6,1.6])
            with self.assertRaisesRegex(ValueError,'durations'):read_prediction_input(path,5,[.5,1.5],[.5,.5])

    def test_custom_samples_skip_ki(self):
        result=summarize_samples(np.ones((1,4,3)),['a','b','c'],calculate_ki=False)
        self.assertNotIn('Ki_mean',result)
        self.assertEqual(result['params_mean'].shape,(1,3))

    def test_nls_recovers_noise_free_custom_parameters(self):
        cfg=config();cfg.update(model='custom',model_file=str(ROOT/'examples/custom_1tc.py'),parameter_names=['K1','k2','Vb'],
                                kinetic_priors={k:cfg['kinetic_priors'][k] for k in ['K1','k2','Vb']},integration_step_min=.05)
        t=np.array(cfg['t_meas']);d=np.array(cfg['frame_durations_min']);aif=np.array([100,60,35,25,20.])
        truth=np.array([.3,.4,.12]);tac=predict_curve(truth,t,d,aif,cfg)
        result=fit(t,d,tac,aif,cfg,starts=1,max_nfev=120)
        self.assertTrue(result['success']);self.assertLess(result['rmse'],1e-5)
        np.testing.assert_allclose(result['parameters'],truth,atol=1e-5)

    def test_feng_delay_and_origin(self):
        p=np.array([[300,10,2,3,.1,.01,.5]])
        y=feng_curve(np.array([0,.5,1,2]),p)[0]
        self.assertEqual(y[0],0);self.assertAlmostEqual(y[1],0);self.assertTrue(np.all(y[2:]>0))

    def test_simulation_format_and_overwrite_guard(self):
        with tempfile.TemporaryDirectory() as folder:
            folder=Path(folder);aif=folder/'aif.csv'
            aif.write_text('t,a\n0,0\n1,10\n3,5\n5,2\n')
            cfg=config();cfg.update(aif_csv=str(aif),output=str(folder/'data.npz'),n_train=4,n_val=2,n_test=2,chunk_size=2,integration_step_min=.05)
            simulate(cfg)
            self.assertEqual(validate_file(folder/'data.npz')['counts']['train'],4)
            with self.assertRaises(FileExistsError):simulate(cfg)

    def test_original_ucdavis_samples_preserved_with_explicit_boundaries(self):
        import pandas as pd
        cfg,_=defaults(Path('workspaces/test_fixture'),small=True)
        raw=pd.read_csv(cfg['aif_csv'])
        for mode,end in [('midpoint',57.5),('frame_average',60.)]:
            with self.subTest(mode=mode):
                cfg['observation']=mode
                t,pool=load_measured_inputs(cfg)
                np.testing.assert_array_equal(t[1:-1],raw.iloc[:,0])
                np.testing.assert_array_equal(pool[:,1:-1],raw.iloc[:,1:].to_numpy().T)
                self.assertEqual(t[0],0);self.assertEqual(t[-1],end)
                np.testing.assert_array_equal(pool[:,0],np.zeros(4))
                np.testing.assert_array_equal(pool[:,-1],raw.iloc[-1,1:])

    def test_missing_aif_coverage_reports_actual_observation_boundary(self):
        cfg,_=defaults(Path('workspaces/test_fixture'),small=True)
        cfg['aif_after_last']='require_coverage'
        with self.assertRaisesRegex(ValueError,'last measurement midpoint is at 57.5 min'):
            load_measured_inputs(cfg)
        cfg['observation']='frame_average'
        with self.assertRaisesRegex(ValueError,'end of the last frame is at 60 min'):
            load_measured_inputs(cfg)
        cfg['aif_after_last']='hold_last';cfg['aif_before_first']='require_zero'
        with self.assertRaisesRegex(ValueError,'first sample is at .*zero_at_origin'):
            load_measured_inputs(cfg)

    def test_invalid_boundary_policy_rejected_even_with_complete_coverage(self):
        cfg=config();cfg['aif_csv']=str(ROOT/cfg['aif_csv'])
        cfg['aif_after_last']='extrapolate'
        with self.assertRaisesRegex(ValueError,'aif_after_last must be'):
            load_measured_inputs(cfg)
        cfg['aif_after_last']='require_coverage';cfg['aif_before_first']='hold_first'
        with self.assertRaisesRegex(ValueError,'aif_before_first must be'):
            load_measured_inputs(cfg)

    def test_ucdavis_original_csv_simulates_both_observation_modes(self):
        import hashlib
        import pandas as pd
        with tempfile.TemporaryDirectory() as folder:
            cfg,_=defaults(Path(folder),small=True)
            source=Path(cfg['aif_csv']);original=source.read_bytes()
            raw=pd.read_csv(source)
            for mode in ('midpoint','frame_average'):
                with self.subTest(mode=mode):
                    cfg.update(observation=mode,output=str(Path(folder)/(mode+'.npz')))
                    simulate(cfg)
                    self.assertEqual(validate_file(cfg['output'])['counts'],{'train':32,'val':16,'test':8})
                    metadata=json.loads(Path(cfg['output']).with_suffix('.json').read_text())
                    self.assertEqual(metadata['source_sha256'],hashlib.sha256(original).hexdigest())
                    self.assertEqual(metadata['input_boundary_policy']['after_last'],'hold_last')
                    with np.load(cfg['output'],allow_pickle=False) as data:
                        self.assertTrue(np.isfinite(data['y_train']).all())
                        if mode=='midpoint':
                            expected=np.array([np.interp(cfg['t_meas'],np.r_[0.,raw.iloc[:,0]],np.r_[0.,raw.iloc[:,i+1]])
                                               for i in data['aif_index_train']])
                            np.testing.assert_allclose(data['y_train'][:,len(cfg['t_meas']):],expected,rtol=1e-12)
            self.assertEqual(source.read_bytes(),original)


if __name__=='__main__':unittest.main()
