# Original UC Davis LoRACM checkpoint

[Reproduction guide](../../docs/UCDAVIS.md) · [User manual](../../README.md)

`best.pth` is the unchanged HPC checkpoint originally named:

```text
valbest_epoch2755_trainloss0.00092387_valloss0.00079129.pth
```

Its SHA-256 is `f4eef814a56b78e639a4398fcf6eaa29f2526f3afb018b5fbd55b986ae04b420`.

Keep **best.pth**, **model_params.json** and **scaling_params.json** together. Original x/y normalization is unchanged. Added metadata records the explicit simulation/patient timestamp labels, parameter order, paper Ki policy and display convention. `protocol.json` retains the corresponding forward model for single-TAC NLS and posterior-median TAC overlays.

In the GUI's prediction, Single TAC & NLS or parametric imaging model selector, choose **Paper LoRACM · UC Davis · epoch 2755**. No fine-tuning or new dataset is required. The GUI copies the model to your project so analyses do not write into these supplied assets.

From `PET-TRACER/LoRACM`:

```bash
python reproduce_ucdavis.py --workspace workspaces/ucdavis_paper
```

`reference_ki.npz` contains archived HPC Ki mean/std maps indexed by the supplied voxel IDs. `reference_ki.png` renders the original mean map with a 2×2 median filter and fixed 0–0.06 scale. It is a reference artifact, not a regenerated posterior. Fresh draws can differ because of Monte Carlo sampling. The original summary archive does not record its Ki transformation; the release explicitly uses the per-draw parameter flooring from the HPC selected-voxel code. See `manifest.json` for asset hashes and provenance.
