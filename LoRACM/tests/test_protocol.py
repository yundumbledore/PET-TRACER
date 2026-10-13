"""Independent numeric and contract checks for the reusable workflow."""
import json
from pathlib import Path
import sys
import tempfile
import unittest
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from protocol import validate_protocol, validate_file, read_prediction_input, load_model, feng_curve
from simulate_protocol import integrate, observe_frames, main as simulate
from analyse_tac import fit, predict_curve
from common import summarize_samples, validate_dataset

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


if __name__=='__main__':unittest.main()
