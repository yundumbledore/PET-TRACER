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

ROOT=Path(__file__).resolve().parents[1]


class TrainingTests(unittest.TestCase):
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


if __name__=='__main__':unittest.main()
