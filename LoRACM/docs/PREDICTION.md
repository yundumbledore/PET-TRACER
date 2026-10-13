# Predict on your own project

[Manual](../README.md) · [Prepare training data](PREPARE_DATA.md)

Use your LoRACM or scratch-CM model to estimate **joint kinetic parameter samples**, then inspect marginal histograms, numerical summaries or maps. Prediction uses the selected run’s scaling and parameter order and selects its model architecture automatically. See [paired LoRACM/CM comparisons](TRAIN_CM.md#compare-the-two-models).

## Before predicting

You need an adapted run folder containing `model_params.json`, `scaling_params.json` and `best.pth` (or `last.pth`). Your testing TACs and input functions must match the training frame schedule, activity units, decay/metabolite correction, kinetic-model convention and midpoint/frame-average sampling. The program rejects mismatched times and recorded durations. Equal frame counts alone do not establish compatibility.

A new protocol generally needs adaptation on that protocol. Scientifically justified resampling is a study decision; the CLI timing override is for installation/debugging checks and is not offered as a GUI prediction shortcut.

## Batch prediction input

Portable NPZ:

```python
np.savez_compressed(
    "testing.npz",
    tac=tacs,                         # [N,T]
    aif=input_function,                # [T] shared, or [N,T]
    t_meas=times_min,                   # [T]
    frame_durations_min=durations_min,  # [T]
    voxel_columns=curve_ids,            # optional unique [N] strings
    truth=known_parameters,            # optional [N,P], for test evaluation
)
```

A legacy pandas HDF5 table is also supported: rows are frames; columns are duration, midpoint, AIF, then TACs. Negative **noisy TAC** values are accepted; AIF values must be nonnegative. Non-finite values are rejected.

GUI: **4 · Predict**, load/choose a file, validate it, select checkpoint/device and posterior sample count, then run. Summaries and the prediction download appear after completion.

Terminal, from `LoRACM/`:

```bash
python infer.py --run workspaces/my_project/runs/adaptation --input workspaces/my_project/testing.npz --output workspaces/my_project/predictions.npz --samples 1000 --batch-size 16 --sample-chunk 64 --device cpu
```

For an Apple GPU use `--device mps`; for a supported CUDA GPU use `--device cuda:0`. Reduce `--batch-size` or `--sample-chunk` to control memory. On one node with multiple visible CUDA devices, add `--multi-gpu`.

## Single TAC with posterior samples and NLS

GUI: **5 · Single TAC & NLS**. Load a CSV with:

```csv
time_min,duration_min,tac,aif
0.5,1,120,400
1.5,1,180,250
2.5,1,200,150
```

This is a **format example**, not a complete input for the pretrained 35-frame model. Supply every frame expected by your adapted run. You can also load a batch NPZ/HDF5 and select one TAC by identifier.

1. Select your adapted run and checkpoint, then load the TAC and input function.
2. Choose the number of posterior draws (default 1,000).
3. Enable **Compare with nonlinear least squares** and select the matching protocol JSON. Generated projects preserve it; for imported datasets, define the same kinetic model and protocol separately.
4. Review/edit the NLS parameter bounds and acknowledge the displayed input interpolation assumptions.
5. Choose **Predict posterior and compare**. The page plots one posterior histogram per parameter, overlays a **dashed vertical NLS estimate**, shows posterior summaries and the fitted TAC, and provides figure/joint-sample downloads.

**How to read the figure.** The histogram is the marginal distribution of samples from your adapted consistency model. The dashed line is a bounded NLS point estimate, not ground truth or an uncertainty interval. Agreement is useful, but both estimators depend on the forward model, input and data conventions. Inspect convergence, active bounds, residual fit, posterior tails and independent truth when available. No line is drawn for a nonconverged NLS fit.

NLS uses three deterministic initializations and **unweighted** squared TAC residuals. Bounds default to the configured uniform kinetic-prior ranges. It uses the same plugin/built-in kinetics and observation mode as the protocol, with RK4 integration. Supplied sampled AIF values are linearly interpolated, with zero inserted at the origin and the last input held through the final frame. For frame-averaged AIFs, this is an explicit continuous-input approximation. The preserved research dataset used forward Euler; its NLS comparison is not an exact solver reproduction.

### Terminal single-TAC analysis

Save one curve as a prediction NPZ (`tac` shape `[1,T]`). The GUI exports the selected `input.npz`, so a GUI analysis can also be repeated in a terminal. CSV users can convert with:

```python
import pandas as pd
import numpy as np
f = pd.read_csv("single_tac.csv")
np.savez_compressed("single_tac.npz", tac=f.tac.to_numpy()[None,:],
                   aif=f.aif.to_numpy(), t_meas=f.time_min.to_numpy(),
                   frame_durations_min=f.duration_min.to_numpy())
```

Run prediction, retain joint samples and fit NLS:

```bash
python analyse_tac.py --run workspaces/my_project/runs/adaptation --input single_tac.npz --protocol workspaces/my_project/runs/adaptation/protocol.json --posterior-output workspaces/my_project/posterior.npz --output workspaces/my_project/nls.json --samples 1000 --device cpu

python plot_posteriors.py --posterior workspaces/my_project/posterior.npz --nls workspaces/my_project/nls.json --output workspaces/my_project/posteriors.png
```

Omit `--protocol` for posterior-only analysis. Use `--checkpoint last.pth` after a short run if `best.pth` was not eligible. To run NLS without the CM, omit `--run` and supply `--protocol`, `--input` and `--output`. `infer.py --save-samples` can retain joint draws for small batches, capped at two million stored parameter values to avoid accidental large outputs.

## Outputs and maps

Prediction NPZ contains `params_mean`, `params_median`, `params_std`, negative-draw fractions, parameter names, curve identifiers, time vectors and metadata. `posterior_samples` is included when requested, shaped `[N, samples, P]`. Optional truth yields parameter RMSE/bias of posterior means in metadata; it does not test posterior calibration.

For **built-in 2TC only**, Kᵢ is calculated for each joint draw as `K1*k3/(k2+k3)`, then summarized. No positivity clipping is applied; near-zero denominators are excluded and a valid fraction is recorded. Custom models do not automatically get a Kᵢ interpretation.

Spatial mapping needs verified voxel indices, ordering, orientation and spacing. The [UC Davis imaging notebook](UCDAVIS.md#4-make-parametric-images) handles the supplied coronal slice once its geometry is verified; it is not a general 3D image importer. A list of TACs or equal voxel count alone does not establish spatial correspondence.
