# Adapt to your own measured input functions

1. Copy `configs/dataset_scenario1.json` to a new configuration. Point `aif_csv` at a CSV using the documented [format](DATA_FORMATS.md). Confirm units, decay correction, blood/plasma convention and acquisition timing against your voxel TAC data.
2. Choose scientifically appropriate parameter priors and noise amplitude bounds. Change the output path and seed. Choose split sizes suitable for your study; the supplied generator samples each split from the same AIF pool. For subject-held-out evaluation, extend the split logic to use distinct AIF pools.
3. Generate the NPZ and inspect the provenance JSON and curves. The selected model is FDG–2TCM. A different input curve does not by itself turn this demo into SRTM, dual-input 2TCM or AATH support.
4. Copy the training configuration, set `dataset_dir`, a new `exp_name`, and `y_dim = 2 * number_of_frames`. Keep `x_dim = 5` for this implementation. The model builder can transfer compatible shapes, but changed timing, dimensions and priors require renewed validation.
5. Fine-tune and retain the generated scaling file with its checkpoint. Run inference on TACs and AIFs using the same acquisition grid and units. Do not reuse another run's normalization file.
6. Supply a binary 2D mask and an explicit mapping from HDF5 voxel column identifiers to C-order positions in that mask. Confirm anatomical orientation and pixel aspect ratio before displaying maps.

The portable generator intentionally retains the source script's numerical solver, midpoint observation convention and noise delta-time definition. If you change those choices, record the change as a new simulation protocol and validate it rather than describing it as an exact reproduction. New kinetic models need their own forward equations, parameter schema, target dimensions, derived summaries and validation.
