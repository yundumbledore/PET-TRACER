"""New scratch methods survive the real training, checkpoint and imaging routes."""
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import PARAMETERS
from infer import main as infer
from inspect_voxel import inspect
from parametric_imaging import load_image
from train_cm import main as train_cm


class PosteriorWorkflowTests(unittest.TestCase):
    def test_new_methods_train_predict_and_image(self):
        threads = torch.get_num_threads()
        torch.set_num_threads(2)
        self.addCleanup(torch.set_num_threads, threads)
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            times = [.5, 1.5]
            duration = [1., 1.]
            pairs = dict(t_meas=times, frame_durations_min=duration, parameter_names=PARAMETERS)
            rng = np.random.default_rng(49)
            for split in ('train', 'val', 'test'):
                pairs['x_'+split] = rng.uniform(.2, 1., (6, 6))
                pairs['y_'+split] = rng.uniform(20., 100., (6, 4))
            dataset = root/'pairs.npz'
            np.savez(dataset, **pairs)
            source = root/'observations.npz'
            y = pairs['y_test'][:2]
            np.savez(source, tac=y[:,:2], aif=y[:,2:], t_meas=times,
                     frame_durations_min=duration, voxel_columns=['a','b'])
            geometry, geometry_info = root/'geometry.npz', root/'geometry.json'
            np.savez(geometry, mask=[[1,1]], voxel_indices=[0,1], voxel_columns=['a','b'])
            geometry_info.write_text(json.dumps(dict(mapping_verified=True, orientation_and_spacing_verified=True)))
            for method, default in [('imf',1), ('scm',2)]:
                with self.subTest(method=method):
                    run = train_cm(dict(dataset_dir=str(dataset), epochs=1, batch_size=2, embed_dim=32,
                                        channels=[32,32,32,32], save_dir=str(root), exp_name=method,
                                        posterior_method=method, device='cpu'))
                    destination = run/'prediction.npz'
                    args = SimpleNamespace(run=run, input=source, output=destination, steps=None,
                                           samples=6, batch_size=2, sample_chunk=3, seed=19, device='cpu',
                                           multi_gpu=False, max_voxels=None, allow_time_mismatch=False,
                                           save_samples=True, checkpoint='best.pth')
                    infer(args)
                    metadata = json.loads(destination.with_suffix('.json').read_text())
                    self.assertEqual(metadata['posterior_method'], method)
                    self.assertEqual(metadata['sampling_steps'], default)
                    self.assertEqual(len(metadata['sampling_times']), default)
                    with np.load(destination, allow_pickle=False) as result:
                        draws = result['posterior_samples']
                        self.assertEqual(draws.shape, (2,6,5))
                        self.assertTrue(np.isfinite(draws).all())
                        self.assertGreater(float(draws.std(axis=1).max()), 0.)
                    image = load_image(destination, geometry, geometry_info, 'Ki', 'ci95_width')
                    self.assertEqual(image['image'].shape, (1,2))
                    self.assertTrue(np.isfinite(image['image']).all())
                    voxel = run/'selected.npz'
                    inspect(run, source, 'b', voxel, samples=4, device='cpu', steps=3)
                    with np.load(voxel, allow_pickle=False) as result:
                        self.assertEqual(result['voxel_columns'].tolist(), ['b'])
                        self.assertEqual(result['posterior_samples'].shape, (1,4,5))
                    self.assertEqual(json.loads(voxel.with_suffix('.json').read_text())['sampling_steps'], 3)
                    checkpoint = torch.load(run/'best.pth', weights_only=True, map_location='cpu')
                    checkpoint['posterior_method'] = 'cm'
                    torch.save(checkpoint, run/'wrong_method.pth')
                    args.output, args.checkpoint = run/'should_not_exist.npz', 'wrong_method.pth'
                    with self.assertRaisesRegex(ValueError, 'Checkpoint posterior method'):
                        infer(args)
                    self.assertFalse(args.output.exists())


if __name__ == '__main__':
    unittest.main()
