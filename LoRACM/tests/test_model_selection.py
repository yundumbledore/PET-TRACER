"""Inference and coordinate contracts of the pretrained seven-model release."""
from pathlib import Path
import json
import os
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import torch
from model_selection.core import ASSETS,ACTIVE,decode,probabilities,load_demo,forward_median,forward_posterior,load_result
from model_selection.infer import parser,run,load_network,inference_chunks


class SelectionTests(unittest.TestCase):
    def test_decoding_preserves_model_specific_kinetics_and_delay(self):
        values=np.tile(np.arange(1.,12.),(7,1));values[:,9]=np.arange(7)+.1
        actual=decode(values)
        np.testing.assert_array_equal(actual[:,9],np.arange(7))
        np.testing.assert_array_equal(actual[:,10],values[:,10])
        for m,active in ACTIVE.items():
            np.testing.assert_array_equal(actual[m,list(active)],values[m,list(active)])
            self.assertTrue(all(actual[m,i]==0 for i in range(9) if i not in active))
        self.assertEqual(decode(np.array([[-1.]*9+[-9.,.2]]))[0,9],0)
        self.assertEqual(decode(np.array([[1.]*9+[9.,.2]]))[0,9],6)

    def test_probabilities_use_model_votes_not_parameter_medians(self):
        draws=np.zeros((2,4,11));draws[0,:,9]=[1,3,3,3];draws[1,:,9]=[2,2,4,4]
        p=probabilities(draws)
        np.testing.assert_allclose(p.sum(1),1)
        self.assertEqual(p[0,3],.75);self.assertEqual(p[1,4],.5)

    def test_mask_order_and_browser_flip_preserve_original_voxel_coordinates(self):
        data=load_demo();self.assertEqual(data['tac'].shape,(83608,35))
        rows,cols=data['coordinates'].T
        np.testing.assert_array_equal(data['ids'][rows,cols],np.arange(83608))
        flipped=np.flipud(data['ids'])
        for idx in (0,1200,83607):
            y,x=data['coordinates'][idx]
            self.assertEqual(flipped[len(flipped)-1-y,x],idx)

    def test_forward_uses_selected_model_delay_and_medians(self):
        times=np.array([.1,.2,.4,.8]);aif=np.array([10.,20.,15.,8.])
        draws=np.zeros((4,11));draws[:,9]=[0,0,3,3]
        draws[:2,6]=[.3,.5];draws[:2,10]=.1
        draws[2:,6]=.9;draws[2:,10]=1.5
        actual,median,count=forward_median(draws,0,times,aif)
        # The notebook takes the first dense-grid sample at/after each frame.
        np.testing.assert_allclose(actual,[0.,4.4,6.9,3.9],atol=1e-10)
        self.assertEqual(count,2);self.assertEqual(median[10],.1)
        with self.assertRaisesRegex(ValueError,'No posterior'):forward_median(draws,1,times,aif)

    def test_inference_publishes_complete_portable_result_and_never_overwrites(self):
        with tempfile.TemporaryDirectory() as folder:
            args=parser().parse_args(['--output',str(Path(folder)/'demo'),'--device','cpu','--limit','3',
                '--samples','7','--batch-size','2','--sample-chunk','3','--steps','3','--export-npz'])
            output=run(args);result=load_result(output)
            self.assertEqual(result['posteriors'].shape,(3,7,11))
            np.testing.assert_allclose(result['probabilities'].sum(1),1)
            np.testing.assert_allclose(probabilities(result['posteriors']),result['probabilities'],atol=1e-7)
            with np.load(output/'Posteriors.npz') as f:np.testing.assert_array_equal(f['posteriors'],result['posteriors'])
            metadata=json.loads((output/'metadata.json').read_text())
            np.testing.assert_allclose(metadata['sigmas'],[1.,2/3,1/3])
            self.assertFalse(metadata['full_slice'])
            with self.assertRaisesRegex(ValueError,'already exists'):run(args)
            result['posteriors']._mmap.close()
            metadata['input_sha256']='another scan'
            (output/'metadata.json').write_text(json.dumps(metadata))
            with self.assertRaisesRegex(ValueError,'bundled data/model'):load_result(output)

    def test_forward_posterior_matches_individual_draws_for_all_seven_models(self):
        times=np.array([.1,.2,.4,.8]);aif=np.array([10.,20.,15.,8.])
        base=np.array([.4,.3,.07,.01,.03,.02,.08,.6,.25,0.,0.])
        for model,active in ACTIVE.items():
            with self.subTest(model=model):
                draws=np.tile(base,(4,1));draws[:,9]=model
                draws[:3,list(active)]=base[list(active)][None,:]*np.array([.8,1.,1.2])[:,None]
                draws[:3,10]=[0.,.1,-.02]
                draws[3,9]=(model+1)%7;draws[3,0]=np.nan
                original=draws.copy()
                summary=forward_posterior(draws,model,times,aif)
                expected=np.stack([forward_median(draws[i:i+1],model,times,aif)[0] for i in range(3)])
                np.testing.assert_allclose(summary['curves'],expected,rtol=1e-12,atol=1e-12)
                np.testing.assert_allclose(summary['mean'],expected.mean(axis=0))
                np.testing.assert_allclose(summary['lower'],np.quantile(expected,.025,axis=0))
                np.testing.assert_allclose(summary['upper'],np.quantile(expected,.975,axis=0))
                self.assertEqual((summary['count'],summary['total'],summary['nonphysical']),(3,3,0))
                np.testing.assert_array_equal(draws,original)

    def test_forward_posterior_uses_every_draw_across_chunks_and_handles_one_draw(self):
        times=np.array([.1,.2,.4,.8]);aif=np.array([10.,20.,15.,8.])
        draws=np.zeros((513,11));draws[:,6]=np.linspace(.1,.9,513);draws[:,10]=np.linspace(0.,.1,513)
        summary=forward_posterior(draws,0,times,aif)
        expected=np.stack([forward_median(draws[i:i+1],0,times,aif)[0] for i in range(len(draws))])
        self.assertEqual(summary['count'],513)
        np.testing.assert_allclose(summary['curves'],expected)
        single=forward_posterior(draws[:1],0,times,aif)
        np.testing.assert_array_equal(single['lower'],single['upper'])
        np.testing.assert_array_equal(single['mean'],single['lower'])
        with self.assertRaisesRegex(ValueError,'No posterior'):forward_posterior(draws,1,times,aif)

    def test_forward_posterior_retains_finite_nonphysical_draws_without_clipping(self):
        times=np.array([.1,.2,.4,.8]);aif=np.array([10.,20.,15.,8.])
        draws=np.zeros((2,11));draws[:,6]=[-.1,.3]
        summary=forward_posterior(draws,0,times,aif)
        positive=forward_median(draws[1:],0,times,aif)[0]
        np.testing.assert_allclose(summary['curves'][0],-positive/3)
        self.assertEqual(summary['nonphysical'],1);self.assertEqual(summary['count'],2)
        self.assertEqual(draws[0,6],-.1)

    def test_forward_posterior_reports_nonfinite_or_unstable_curves(self):
        times=np.array([.1,.2,.4,.8]);aif=np.array([10.,20.,15.,8.])
        draws=np.zeros((3,11));draws[:,9]=1;draws[:,0]=.3;draws[:,1]=.2;draws[:,6]=.08
        draws[1,0]=np.nan;draws[2,1]=-1e9
        summary=forward_posterior(draws,1,times,aif)
        self.assertEqual((summary['count'],summary['total']),(1,3))
        np.testing.assert_allclose(summary['mean'],forward_median(draws[:1],1,times,aif)[0])
        with self.assertRaisesRegex(ValueError,'None of the 2'):
            forward_posterior(draws[1:],1,times,aif)

    def test_failure_removes_partial_run(self):
        with tempfile.TemporaryDirectory() as folder:
            args=parser().parse_args(['--output',str(Path(folder)/'demo'),'--device','cpu','--limit','2','--samples','2'])
            with patch('model_selection.infer.check_cancelled',side_effect=RuntimeError('cancelled')):
                with self.assertRaisesRegex(RuntimeError,'cancelled'):run(args)
            self.assertEqual(list(Path(folder).iterdir()),[])

    def test_checkpoint_loads_strictly_and_mps_batch_is_bounded(self):
        network,cfg=load_network(torch.device('cpu'))
        with torch.inference_mode():
            output=network(torch.zeros(2,11),torch.zeros(2,70),torch.ones(2))
        self.assertEqual(output.shape,(2,11));self.assertTrue(torch.isfinite(output).all())
        a,b=inference_chunks(5000,100,torch.device('mps'));self.assertLessEqual(a*b,4096)
        self.assertEqual(inference_chunks(25,64,torch.device('cpu')),(25,64))


if __name__=='__main__':unittest.main()
