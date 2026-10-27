"""Independent distribution checks and end-to-end measurement-noise contracts."""
import json
from pathlib import Path
import sys
import tempfile
import unittest
import numpy as np
from scipy.special import ndtri
from scipy.stats import lognorm

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from measurement_noise import apply_measurement_noise, noise_settings, noise_metadata, read_noise_metadata
from protocol import validate_protocol, validate_file
from hpc_compat import noise_standard_deviation
from ucdavis import defaults, scratch_protocol, testing_from_dataset


class MeasurementNoiseTests(unittest.TestCase):
    def test_matches_independent_scipy_quantiles_and_moments(self):
        means=np.array([25.,250.,2500.])
        t=np.array([.05,.3,1.5]);duration=np.array([.1,.4,2.])
        amplitude=4.
        probabilities=np.array([.001,.025,.25,.5,.75,.975,.999])
        z=np.broadcast_to(ndtri(probabilities)[:,None],(len(probabilities),len(t)))
        actual=apply_measurement_noise(np.broadcast_to(means,z.shape),amplitude,t,109.8,z,frame_durations=duration)
        # Independent reference: original unsimplified sigma and SciPy's CDF.
        decay=np.log(2)/109.8
        sigma=amplitude*np.sqrt(means*np.exp(-decay*t)/duration)*np.exp(decay*t)
        eta=np.sqrt(np.log1p((sigma/means)**2))
        reference=lognorm(s=eta,scale=means*np.exp(-eta**2/2))
        np.testing.assert_allclose(actual,reference.ppf(probabilities[:,None]),rtol=4e-14)
        np.testing.assert_allclose(reference.mean(),means,rtol=1e-14)
        np.testing.assert_allclose(reference.var(),sigma**2,rtol=1e-13)
        np.testing.assert_allclose(actual[3],means*np.exp(-eta**2/2),rtol=1e-14)

    def test_stratified_draws_recover_target_mean_and_variance(self):
        # Equal-probability Gaussian quantiles avoid a flaky random MC assertion.
        n=200000
        z=np.broadcast_to(ndtri((np.arange(n)+.5)/n)[:,None],(n,2))
        means=np.broadcast_to([100.,1000.],z.shape)
        amplitude=np.broadcast_to([[4.]],(n,1))
        t=np.array([.5,3.]);duration=np.array([1.,2.])
        noisy=apply_measurement_noise(means,amplitude,t,109.8,z,frame_durations=duration)
        variance=16*means[0]*np.exp(np.log(2)*t/109.8)/duration
        np.testing.assert_allclose(noisy.mean(axis=0),means[0],rtol=2e-5)
        np.testing.assert_allclose(noisy.var(axis=0),variance,rtol=5e-4)

    def test_actual_duration_and_decay_correction_determine_variance(self):
        clean=np.full((1,3),1000.);t=np.array([.1,1.,2.]);duration=np.array([.2,.4,1.])
        z=np.ones_like(clean)
        actual=apply_measurement_noise(clean,5.,t,10.,z,frame_durations=duration)
        sigma=5*np.sqrt(clean*np.exp(np.log(2)*t/10.)/duration)
        eta=np.sqrt(np.log1p((sigma/clean)**2))
        np.testing.assert_allclose(actual,clean*np.exp(eta-.5*eta**2),rtol=1e-14)
        wrong_duration=np.r_[t[0],np.diff(t)]
        self.assertFalse(np.allclose(actual,apply_measurement_noise(clean,5.,t,10.,z,frame_durations=wrong_duration)))

    def test_floor_does_not_clip_draws_and_zero_amplitude_is_exact(self):
        clean=np.array([[0.,1e-20,4.]])
        t=np.array([.5,1.5,2.5]);duration=np.ones(3)
        noisy=apply_measurement_noise(clean,4.,t,109.8,np.zeros_like(clean),frame_durations=duration)
        self.assertTrue(np.all(noisy>0));self.assertTrue(np.all(noisy[0,:2]<1e-8))
        exact=apply_measurement_noise(clean,0.,t,109.8,np.ones_like(clean),frame_durations=duration)
        np.testing.assert_array_equal(exact,np.maximum(clean,1e-8))

    def test_numerical_underflow_is_positive_and_overflow_is_rejected(self):
        clean=np.ones((1,2));t=np.array([.5,1.5])
        tiny=apply_measurement_noise(clean,5.,t,109.8,np.full_like(clean,-1e5),frame_durations=np.ones(2))
        np.testing.assert_array_equal(tiny,np.full_like(clean,np.nextafter(0.,1.)))
        with self.assertRaisesRegex(ValueError,'float64 range'):
            apply_measurement_noise(clean,5.,t,109.8,np.full_like(clean,1e5),frame_durations=np.ones(2))
        # Squaring sigma/mean directly overflows at this valid float64 amplitude.
        finite=apply_measurement_noise(clean,1e200,t,109.8,np.zeros_like(clean),frame_durations=np.ones(2))
        self.assertTrue(np.all(np.isfinite(finite)));self.assertTrue(np.all(finite>0))

    def test_invalid_noise_inputs_are_rejected(self):
        args=dict(clean=np.ones((2,2)),amplitude=np.ones((2,1)),times=[.5,1.5],half_life=109.8,
                  standard_normals=np.zeros((2,2)),frame_durations=[1.,1.])
        for change in (dict(amplitude=-1.),dict(amplitude=[1.,2.]),dict(half_life=0.),
                       dict(times=[.5,.5]),dict(frame_durations=[1.,0.]),dict(frame_durations=None),
                       dict(standard_normals=np.ones((1,2))),dict(clean=np.full((2,2),-1.)),
                       dict(signal_floor=0.),dict(model='poisson'),dict(timing='sample_spacing'),
                       dict(standard_normals=np.full((2,2),np.nan))):
            with self.subTest(change=change),self.assertRaises(ValueError):
                apply_measurement_noise(**{**args,**change})

    def test_legacy_gaussian_samples_and_implicit_timing_are_unchanged(self):
        clean=np.array([[0.,100.,200.],[1.,2.,3.]])
        amplitude=np.array([[5.],[2.]])
        t=np.array([.05,.15,.5]);z=np.array([[-1.,0.,1.],[.5,-2.,3.]])
        actual=apply_measurement_noise(clean,amplitude,t,109.8,z,model='gaussian',timing='sample_spacing')
        np.testing.assert_array_equal(actual,clean+noise_standard_deviation(clean,amplitude,t,109.8)*z)
        self.assertLess(actual[0,0],0.)
        self.assertEqual(noise_settings({'simulation_mode':'hpc_euler'})['noise_timing'],'sample_spacing')
        self.assertEqual(noise_settings({})['noise_model'],'gaussian')
        self.assertEqual(noise_settings({})['noise_timing'],'frame_duration')

    def test_new_templates_and_scratch_defaults(self):
        for name in ('protocol_ucdavis','dataset_scenario1','dataset_custom','dataset_smoke'):
            cfg=json.loads((ROOT/'configs'/f'{name}.json').read_text())
            self.assertEqual(noise_settings(cfg),dict(noise_model='lognormal',noise_timing='frame_duration',noise_epsilon=1e-8))
            self.assertEqual(cfg['noise_prior'],dict(type='uniform',min=3.,max=7.))
            validate_protocol(cfg)
        cfg,_=defaults('unused')
        self.assertEqual(noise_metadata(cfg),noise_metadata(scratch_protocol(cfg,'unused')))
        cfg['noise_timing']='sample_spacing'
        with self.assertRaisesRegex(ValueError,'actual frame durations'):validate_protocol(cfg)

    def test_unknown_import_provenance_and_invalid_metadata(self):
        self.assertIsNone(read_noise_metadata({}))
        cfg,_=defaults('unused')
        metadata=noise_metadata(cfg)
        self.assertEqual(read_noise_metadata({'measurement_noise_json':np.asarray(json.dumps(metadata))}),metadata)
        for value in ('not json','[]',json.dumps({**metadata,'noise_epsilon':-1}),
                      json.dumps({**metadata,'half_life_min':0}),json.dumps({**metadata,'noise_timing':'sample_spacing'})):
            with self.subTest(value=value),self.assertRaisesRegex(ValueError,'measurement_noise_json'):
                read_noise_metadata({'measurement_noise_json':np.asarray(value)})

    def test_generated_log_normal_pairs_match_research_generator_and_chunking(self):
        import jax
        import jax.numpy as jnp
        import pandas as pd
        from simulate_protocol import main
        from simulation import generate_dataset,TwoTissueCompartment,get_central_priors
        with tempfile.TemporaryDirectory() as folder:
            cfg,_=defaults(folder,small=True)
            cfg.update(n_train=5,n_val=3,n_test=2,chunk_size=2)
            main(cfg)
            with np.load(cfg['output']) as data:
                first={key:data[key].copy() for key in data.files}
            csv=pd.read_csv(cfg['aif_csv']);t=csv.iloc[:,0].to_numpy();pool=csv.iloc[:,1:].to_numpy().T
            assigned=np.random.RandomState(cfg['seed']).choice(len(pool),size=10,replace=True)
            offset=0
            for split,key in zip(('train','val','test'),jax.random.split(jax.random.PRNGKey(cfg['seed']),3)):
                count=cfg['n_'+split];selected=assigned[offset:offset+count];offset+=count
                priors=get_central_priors(cfg['kinetic_priors'],cfg['test_prior_scale']) if split=='test' else cfg['kinetic_priors']
                y,x=generate_dataset(count,TwoTissueCompartment,priors,{},cfg['noise_prior'],jnp.asarray(t),key,
                                     cfg['half_life_min'],external_input_curves=pool[selected],return_clean=False,
                                     noise_model='lognormal',frame_durations=cfg['frame_durations_min'])
                np.testing.assert_array_equal(first['x_'+split],x)
                np.testing.assert_allclose(first['y_'+split],y,atol=2e-8,rtol=4e-12)
                np.testing.assert_array_equal(first['y_'+split][:,len(t):],pool[selected])
                self.assertTrue(np.all(first['y_'+split][:,:len(t)]>0))
            report=validate_file(cfg['output'])
            self.assertEqual(report['measurement_noise'],noise_metadata(cfg))
            sidecar=json.loads(Path(cfg['output']).with_suffix('.json').read_text())
            self.assertEqual(sidecar['measurement_noise'],report['measurement_noise'])
            cfg.update(chunk_size=4,output=str(Path(folder)/'chunks.npz'))
            main(cfg)
            with np.load(cfg['output']) as data:
                for name,values in first.items():np.testing.assert_array_equal(data[name],values,err_msg=name)
            testing=Path(folder)/'testing.npz';testing_from_dataset(cfg['output'],testing)
            with np.load(testing) as data:self.assertEqual(str(data['measurement_noise_json']),str(first['measurement_noise_json']))

    def test_legacy_cli_wrapper_honours_explicit_log_normal_config(self):
        from create_dataset import main
        with tempfile.TemporaryDirectory() as folder:
            cfg,_=defaults(folder,small=True)
            cfg.pop('simulation_mode')
            cfg.update(n_train=3,n_val=2,n_test=2,chunk_size=2)
            main(cfg)
            self.assertEqual(validate_file(cfg['output'])['measurement_noise']['noise_model'],'lognormal')
            with np.load(cfg['output']) as data:self.assertTrue(np.all(data['y_train'][:,:35]>0))

    def test_training_rejects_conflicting_noise_provenance_before_creating_run(self):
        from simulate_protocol import main as simulate
        from train import main as train
        with tempfile.TemporaryDirectory() as folder:
            cfg,training=defaults(folder,small=True)
            cfg.update(n_train=8,n_val=4,n_test=2,chunk_size=4)
            simulate(cfg)
            protocol=Path(training['simulation_protocol'])
            cfg['noise_model']='gaussian'
            protocol.write_text(json.dumps(cfg))
            with self.assertRaisesRegex(ValueError,'measurement noise must match'):
                train(training)
            self.assertFalse((Path(training['save_dir'])/training['exp_name']).exists())


if __name__=='__main__':unittest.main()
