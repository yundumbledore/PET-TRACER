"""Scratch initialization, schema inference and paired evaluation contracts."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from training_config import resolve_training_config
from protocol import validate_file
from train_cm import main as train_cm
from train import main as train_lora
from model_factory import build_model
from compare_models import main as compare
from analyse_tac import main as analyse, predict_curve

ROOT=Path(__file__).resolve().parents[1]


class TrainingTests(unittest.TestCase):
    def test_repeated_prediction_cli_keeps_old_samples_and_settings(self):
        torch.set_num_threads(2)
        from infer import main as infer
        from common import file_sha256
        run=train_cm(self.cfg)
        source=self.root/'testing.npz'
        np.savez(source,tac=np.ones((1,2)),aif=np.ones(2),t_meas=[.5,1.5],frame_durations_min=[1,1])
        output=self.root/'predictions.npz'
        args=SimpleNamespace(run=run,input=source,output=output,checkpoint='best.pth',samples=4,batch_size=1,sample_chunk=2,
                             steps=2,seed=0,device='cpu',multi_gpu=False,max_voxels=None,allow_time_mismatch=False,save_samples=True,new_output=True)
        infer(args);original=file_sha256(output)
        args.samples=8;args.steps=3;infer(args)
        self.assertEqual(file_sha256(output),original)
        newer=self.root/'predictions_002.npz'
        with np.load(newer) as data: self.assertEqual(data['posterior_samples'].shape,(1,8,3))
        self.assertEqual(json.loads(newer.with_suffix('.json').read_text())['samples'],8)
        self.assertEqual(len(json.loads(newer.with_suffix('.json').read_text())['sampling_times']),3)
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory()
        self.root=Path(self.temp.name)
        rng=np.random.default_rng(42)
        arrays={'t_meas':np.array([.5,1.5]),'frame_durations_min':np.ones(2),
                'parameter_names':np.array(['K1','k2','Vb']),'observation':np.asarray('midpoint')}
        for split in ('train','val','test'):
            arrays['x_'+split]=rng.uniform(.1,1,(4,4))
            arrays['y_'+split]=np.column_stack([rng.uniform(20,80,(4,2)),np.full((4,2),100.)])
        self.data=self.root/'data.npz';np.savez_compressed(self.data,**arrays)
        self.report=validate_file(self.data)
        self.cfg={'dataset_dir':str(self.data),'epochs':1,'batch_size':2,'embed_dim':32,'device':'cpu',
                  'save_dir':str(self.root/'runs'),'exp_name':'cm','seed':4}

    def tearDown(self):
        self.temp.cleanup()

    def test_inferred_dimensions_and_early_architecture_errors(self):
        cfg=resolve_training_config(self.cfg,self.report,'cm')
        self.assertEqual((cfg['x_dim'],cfg['y_dim']),(3,4))
        self.assertEqual(cfg['loss_weights'],[1.,1.,1.])
        self.assertEqual(cfg['parameter_names'],['K1','k2','Vb'])
        for override in [{'x_dim':5},{'y_dim':70},{'embed_dim':33},{'channels':[32,63,128,256]},
                         {'lr':float('nan')},{'ema_decay':1.},{'pretrained':'checkpoints/foundation.pth'}]:
            with self.subTest(override=override),self.assertRaises(ValueError):
                resolve_training_config(dict(self.cfg,**override),self.report,'cm')

    def test_stop_keeps_lowest_validation_checkpoint_before_legacy_warmup(self):
        torch.set_num_threads(2)
        from runtime import Progress
        original=Progress.update
        def request_after_two(progress,completed,state='running',**details):
            original(progress,completed,state,**details)
            if details.get('phase')=='Validation' and details.get('epoch')==2:
                (self.root/'stop').touch()
        cfg=dict(self.cfg,epochs=8,checkpoint_start_epoch=151,checkpoint_interval=1)
        with patch.dict(os.environ,{'PET_TRACER_STOP_FILE':str(self.root/'stop')}),patch.object(Progress,'update',request_after_two):
            run=train_cm(cfg)
        import pandas as pd
        metrics=pd.read_csv(run/'metrics.csv')
        self.assertEqual(metrics.epoch.tolist(),[1,2])
        best=torch.load(run/'best.pth',map_location='cpu',weights_only=True)
        last=torch.load(run/'last.pth',map_location='cpu',weights_only=True)
        self.assertEqual(best['epoch'],int(metrics.loc[metrics.val_loss.idxmin(),'epoch']))
        self.assertEqual(last['epoch'],2)
        index=json.loads((run/'checkpoints.json').read_text())
        self.assertIn('epochs/epoch_000001.pth',index)
        self.assertIn('epochs/epoch_000002.pth',index)
        self.assertTrue(json.loads((run/'training_summary.json').read_text())['stopped_early'])

    def test_unicode_configuration_without_utf8_locale(self):
        config=self.root/'unicode.json'
        expected='tracér/🧪 dataset.npz'
        config.write_text(json.dumps({'dataset_dir':expected},ensure_ascii=False),encoding='utf-8')
        environment=dict(os.environ,PYTHONUTF8='0',PYTHONCOERCECLOCALE='0',LC_ALL='C')
        code='import json,sys; from common import load_config; print(json.dumps(load_config(sys.argv[1])))'
        result=subprocess.run([sys.executable,'-c',code,str(config)],cwd=ROOT,env=environment,
                              capture_output=True,check=True)
        self.assertEqual(json.loads(result.stdout)['dataset_dir'],expected)

    def test_scratch_training_and_paired_inference(self):
        torch.set_num_threads(2)
        resolved=resolve_training_config(self.cfg,self.report,'cm')
        torch.manual_seed(self.cfg['seed'])
        initial=build_model(resolved,torch.device('cpu'),training=True)
        initial_conv=initial.conv1.weight.detach().clone()
        # No scratch path may read foundation weights, including indirectly.
        with patch('model_lora.torch.load',side_effect=AssertionError('Scratch training loaded a checkpoint')):
            cm_run=train_cm(self.cfg)
        checkpoint=torch.load(cm_run/'best.pth',map_location='cpu',weights_only=True)
        self.assertFalse(torch.equal(initial_conv,checkpoint['model_state_dict']['conv1.weight']))
        self.assertFalse(any('.wrapped.' in key or key.endswith(('.A','.B')) for key in checkpoint['model_state_dict']))
        summary=json.loads((cm_run/'training_summary.json').read_text())
        self.assertEqual(summary['training_mode'],'cm')
        self.assertEqual(summary['total_parameters']-summary['trainable_parameters'],16)
        lora_cfg=dict(self.cfg,pretrained=str(ROOT/'checkpoints/foundation.pth'),exp_name='lora')
        lora_run=train_lora(lora_cfg)
        testing=self.root/'testing.npz'
        np.savez_compressed(testing,tac=np.array([[30.,40.]]),aif=np.array([100.,100.]),
                            t_meas=np.array([.5,1.5]),frame_durations_min=np.ones(2),truth=np.array([[.3,.4,.1]]))
        output=self.root/'comparison'
        compare(SimpleNamespace(lora_run=str(lora_run),cm_run=str(cm_run),input=str(testing),output=str(output),
                                lora_checkpoint='best.pth',cm_checkpoint='best.pth',samples=8,batch_size=1,
                                sample_chunk=4,steps=3,seed=0,device='cpu',save_samples=True))
        report=json.loads((output/'comparison.json').read_text())
        self.assertTrue(report['same_training_dataset'])
        self.assertEqual(report['training_configuration_differences'],{})
        with np.load(output/'cm.npz',allow_pickle=False) as data:
            self.assertEqual(data['posterior_samples'].shape,(1,8,3))
            self.assertNotIn('Ki_mean',data)
        self.assertIn('rmse',(output/'metrics.csv').read_text())
        # The single-TAC worker must publish a median forward curve for both
        # estimators and mark the new analysis complete only after both finish.
        protocol=json.loads((ROOT/'configs/dataset_custom.json').read_text())
        protocol.update(model='custom',model_file=str(ROOT/'examples/custom_1tc.py'),
                        parameter_names=['K1','k2','Vb'],t_meas=[.5,1.5],frame_durations_min=[1.,1.],
                        observation='midpoint',integration_step_min=.1,
                        kinetic_priors={k:{'type':'uniform','min':.001,'max':1.} for k in ['K1','k2','Vb']})
        protocol_path=self.root/'protocol.json';protocol_path.write_text(json.dumps(protocol))
        folder=self.root/'single'
        analyse(SimpleNamespace(run=str(lora_run),comparison_run=str(cm_run),comparison_checkpoint='best.pth',
            protocol=str(protocol_path),input=str(testing),output=str(folder/'nls.json'),posterior_output=str(folder/'posterior.npz'),
            checkpoint='best.pth',samples=8,steps=2,sample_chunk=4,seed=17,device='cpu',skip_nls=True))
        forward=json.loads((folder/'forward.json').read_text())
        self.assertEqual([c['model'] for c in forward['curves']],['lora','cm'])
        for item,file in zip(forward['curves'],['posterior.npz','cm_posterior.npz']):
            with np.load(folder/file) as data: median=np.median(data['posterior_samples'][0],axis=0)
            np.testing.assert_allclose(item['posterior_median'],median)
            expected=predict_curve(median,np.array([.5,1.5]),np.ones(2),np.array([100.,100.]),protocol)
            np.testing.assert_allclose(item['fitted_tac'],expected)
        self.assertEqual(json.loads((folder/'analysis.json').read_text())['state'],'complete')


if __name__=='__main__':unittest.main()
