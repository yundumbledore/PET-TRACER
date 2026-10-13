"""Parameter semantics must reset heads even when shapes remain compatible."""
from pathlib import Path
import sys
import unittest
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from model_lora import build_lora_unet


class TransferTests(unittest.TestCase):
    def test_same_width_custom_parameters_reset_heads_keep_backbone(self):
        checkpoint=Path(__file__).resolve().parents[1]/'checkpoints/foundation.pth'
        state=torch.load(checkpoint,map_location='cpu',weights_only=True)
        state=state.get('model_state_dict',state.get('ema_state_dict',state))
        model=build_lora_unet(5,70,1024,pretrained_ckpt=str(checkpoint),reset_parameter_heads=True)
        self.assertTrue(model.encodex.wrapped.weight.requires_grad)
        self.assertTrue(model.decodex.weight.requires_grad)
        self.assertFalse(torch.equal(model.encodex.wrapped.weight,state['encodex.weight']))
        self.assertTrue(torch.equal(model.conv1.wrapped.weight,state['conv1.weight']))
        self.assertFalse(model.conv1.wrapped.weight.requires_grad)


if __name__=='__main__':unittest.main()
