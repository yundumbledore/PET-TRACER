"""Historical solver, time labels, paper weights and display are contracts."""
import json
from pathlib import Path
import sys
import tempfile
import unittest
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
sys.path.insert(0,str(ROOT.parent))
from hpc_compat import euler_observations,noise_standard_deviation
from protocol import load_model,read_prediction_input,prediction_times
from ucdavis import defaults,prepare_paper_run
from common import summarize_samples,file_sha256
from parametric_imaging import display_plane
from pet_environment import candidates


class HPCReproductionTests(unittest.TestCase):
    def test_euler_first_update_and_rightward_observations(self):
        class OneTC:
            STATE_SIZE=1
            @staticmethod
            def rhs(t,state,p,aif): return p[:,0:1]*aif[:,None]-p[:,1:2]*state
            @staticmethod
            def observe(state,p,aif): return state[:,0]
        t=np.array([.04,.115,.2]);p=np.array([[.3,.7]])
        actual=euler_observations(OneTC,p,np.full((1,3),5.),t,.03)[0]
        grid=np.arange(t[0],t[-1]+1e-8,.03)
        if grid[-1]<t[-1]: grid=np.r_[grid,t[-1]]
        delta=np.r_[grid[0],np.diff(grid)]
        expected=(.3*5/.7)*(1-np.cumprod(1-.7*delta))
        np.testing.assert_allclose(actual,expected[np.searchsorted(grid,t)],atol=1e-14)
        self.assertAlmostEqual(actual[0],.3*5*.04)

    def test_historical_noise_uses_sample_spacing(self):
        clean=np.full((1,3),100.);t=np.array([.05,.15,.5]);amplitude=np.array([[2.]])
        decay=np.log(2)/109.8
        expected=2*np.sqrt(100*np.exp(decay*t)/np.array([.05,.1,.35]))
        np.testing.assert_allclose(noise_standard_deviation(clean,amplitude,t,109.8)[0],expected)
        self.assertFalse(np.allclose(expected,noise_standard_deviation(clean,amplitude,t,109.8,np.ones(3))[0]))

    def test_defaults_and_unchanged_aif_conditioning(self):
        from simulate_protocol import main
        with tempfile.TemporaryDirectory() as folder:
            cfg,training=defaults(folder,small=True)
            self.assertEqual(training['lr_scheduler_patience'],50)
            self.assertEqual(cfg['kinetic_priors']['k4']['max'],.05)
            self.assertEqual(cfg['kinetic_priors']['K1']['max'],1.5)
            main(cfg)
            raw=pd.read_csv(cfg['aif_csv']).iloc[:,1:].to_numpy().T
            with np.load(cfg['output']) as data:
                indices=data['aif_index_train']
                np.testing.assert_array_equal(data['y_train'][:,35:],raw[indices])
                self.assertEqual(str(data['simulation_mode']),'hpc_euler')
                np.testing.assert_array_equal(data['prediction_t_meas'],cfg['prediction_t_meas'])

    def test_only_explicit_hpc_time_contract_accepts_patient_labels(self):
        cfg,_=defaults('unused')
        patient=ROOT/'data/real/Sub001_slice75_adjusted.h5'
        with self.assertRaisesRegex(ValueError,'schedule'):
            read_prediction_input(patient,5,cfg['t_meas'],cfg['frame_durations_min'])
        payload=read_prediction_input(patient,5,prediction_times(cfg),cfg['frame_durations_min'])
        self.assertEqual(len(payload[2]),24237)
        wrong=dict(cfg,prediction_t_meas=(np.asarray(cfg['prediction_t_meas'])+.1).tolist())
        with self.assertRaisesRegex(ValueError,'schedule'):
            read_prediction_input(patient,5,prediction_times(wrong),cfg['frame_durations_min'])
        unmarked=dict(cfg,simulation_mode='rk4')
        with self.assertRaisesRegex(ValueError,'schedule'):
            read_prediction_input(patient,5,prediction_times(unmarked),cfg['frame_durations_min'])

    def test_paper_ki_transform_is_explicit_and_preserves_parameter_draws(self):
        draws=np.array([[[-1,-2,1,.01,.1],[1,2,1,.01,.1],[2,1,3,.01,.1]]],dtype=float)
        before=draws.copy()
        summary=summarize_samples(draws,ki_policy='hpc_floor_1e-4')
        expected=np.array([1e-4/(1+1e-4),1/3,1.5])
        self.assertAlmostEqual(summary['Ki_mean'][0],expected.mean())
        self.assertAlmostEqual(summary['Ki_median'][0],np.median(expected))
        self.assertAlmostEqual(summary['Ki_ci95_width'][0],np.quantile(expected,.975)-np.quantile(expected,.025))
        np.testing.assert_array_equal(before,draws)

    def test_paper_display_filters_only_rendered_values(self):
        image=np.array([[np.nan,.01,.02],[.03,.2,.04],[.01,.02,np.nan]])
        original=image.copy()
        result=dict(image=image,metadata={'display_profile':'ucdavis_hpc'},parameter='Ki',statistic='mean')
        displayed,settings=display_plane(result)
        self.assertEqual((settings['vmin'],settings['vmax'],settings['median_filter_size']),(0.,.06,2))
        self.assertTrue(np.isnan(displayed[0,0]))
        self.assertFalse(np.allclose(displayed,image,equal_nan=True))
        np.testing.assert_array_equal(image,original)
        raw,settings=display_plane(result,style='raw')
        np.testing.assert_array_equal(raw,image)
        self.assertEqual(settings['median_filter_size'],0)

    def test_exact_paper_weights_scaling_and_idempotent_preparation(self):
        with tempfile.TemporaryDirectory() as folder:
            run=prepare_paper_run(folder)
            self.assertEqual(file_sha256(run/'best.pth'),'f4eef814a56b78e639a4398fcf6eaa29f2526f3afb018b5fbd55b986ae04b420')
            self.assertEqual(prepare_paper_run(folder),run)
            scaling=json.loads((run/'scaling_params.json').read_text())
            self.assertAlmostEqual(scaling['x_std'][3],.014392484220825261)
            cfg=json.loads((run/'model_params.json').read_text())
            self.assertEqual(cfg['ki_policy'],'hpc_floor_1e-4')
            (run/'best.pth').write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError,'changed'): prepare_paper_run(folder)

    def test_environment_discovery_survives_new_checkout_and_handles_windows(self):
        options=candidates('/new/download',environ={},home='/user',platform='linux',executable='/system/python')
        self.assertEqual(str(options[0]),'/user/.venvs/pet-tracer/bin/python')
        win=candidates('/new/download',environ={},home='/user',platform='win32',executable='/system/python')
        self.assertTrue(str(win[0]).endswith('pet-tracer/Scripts/python.exe'))
        explicit=candidates('/new/download',environ={'PET_TRACER_PYTHON':'/custom/python'},home='/user')
        self.assertEqual(explicit,[Path('/custom/python')])


if __name__=='__main__': unittest.main()
