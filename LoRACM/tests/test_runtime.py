"""Device routing, adapter folding, and the expanded MPS batch safety contract."""
import copy
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from runtime import select_device, Progress, check_cancelled
from infer import inference_chunks, MPS_MAX_EXPANDED_BATCH
from lora import inject_lora_adapters, merge_for_inference


class RuntimeTests(unittest.TestCase):
    def test_auto_routes_cuda_then_mps_then_cpu_and_rejects_unavailable_device(self):
        with patch('torch.cuda.is_available',return_value=False),patch('torch.backends.mps.is_available',return_value=True):
            self.assertEqual(str(select_device()),'mps')
        with patch('torch.cuda.is_available',return_value=True): self.assertEqual(str(select_device()),'cuda:0')
        with patch('torch.cuda.is_available',return_value=False),patch('torch.backends.mps.is_available',return_value=False):
            self.assertEqual(str(select_device()),'cpu')
            with self.assertRaises(ValueError): select_device('mps')
            with self.assertRaises(ValueError): select_device('cuda:0')

    def test_mac_user_batch_cannot_exceed_expanded_limit(self):
        for curves,chunk in [(512,200),(100000,64),(1,100000),(16,64)]:
            batch,draws=inference_chunks(curves,chunk,torch.device('mps'))
            self.assertLessEqual(batch*draws,MPS_MAX_EXPANDED_BATCH)
            self.assertGreater(min(batch,draws),0)
        self.assertEqual(inference_chunks(512,200,torch.device('cuda:0')),(512,200))

    def test_adapter_folding_matches_nonzero_adapters_for_every_layer_type(self):
        torch.manual_seed(42)
        for layer,shape in [(torch.nn.Linear(5,7),(4,1,5)),(torch.nn.Conv1d(3,4,3),(4,3,9)),
                            (torch.nn.ConvTranspose1d(3,4,3,stride=2),(4,3,9))]:
            model=inject_lora_adapters(torch.nn.Sequential(layer),r=2,alpha=4,dropout=.2)
            with torch.no_grad(): model[0].B.normal_(std=.2)
            model.eval();x=torch.randn(shape)
            expected=model(x)
            merged=merge_for_inference(copy.deepcopy(model))
            torch.testing.assert_close(merged(x),expected,atol=1e-6,rtol=1e-5)
            self.assertFalse(any(key.endswith(('.A','.B')) for key in merged.state_dict()))

    def test_dynamic_progress_and_cooperative_cancel(self):
        with tempfile.TemporaryDirectory() as folder:
            status=Path(folder)/'status.json';stop=Path(folder)/'stop'
            with patch.dict(os.environ,{'PET_TRACER_PROGRESS_FILE':str(status),'PET_TRACER_STOP_FILE':str(stop)}):
                progress=Progress('Prediction',100,'draws');progress.update(25,device='mps')
                self.assertEqual(json.loads(status.read_text())['completed'],25)
                progress.update(100,state='complete')
                self.assertEqual(json.loads(status.read_text())['state'],'complete')
                stop.touch()
                with self.assertRaises(RuntimeError): check_cancelled()

