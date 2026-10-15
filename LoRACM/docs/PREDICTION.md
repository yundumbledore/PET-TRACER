# Predict on your own project

[Manual](../README.md) · [Prepare training data](PREPARE_DATA.md) · [Parametric imaging](PARAMETRIC_IMAGING.md)

Use your LoRACM or scratch-CM model to estimate **joint kinetic parameter samples**, then inspect marginal histograms, numerical summaries or maps. Prediction uses the selected run’s scaling and parameter order and selects its model architecture automatically. See [paired LoRACM/CM comparisons](TRAIN_CM.md#compare-the-two-models).

In the default UC Davis walkthrough, step 4 selects the supplied patient-slice file and the model from step 3. Predict the complete foreground before opening step 6. New outputs include equal-tailed 95% credible intervals and widths for parameters and per-draw 2TC Kᵢ; whole-image draws do not need to be saved. The [imaging page](PARAMETRIC_IMAGING.md) displays maps and retrieves the clicked voxel's TAC/AIF and posterior KDE.

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

GUI: **4 · Predict** for LoRACM, or **CM · Predict** for a trained scratch CM. Each page lists its own model type. Choose **My own file** to reuse the model on another compatible testing file, then validate it, select the checkpoint, device, posterior samples, reverse sampling steps, seed, curve batch and sample chunk. Summaries and a download appear after completion.

For another patient's anatomical maps, choose **My own patient scan** and provide both the coronal TAC/AIF file and the corresponding mask. See [patient scan and mask preparation](PARAMETRIC_IMAGING.md#your-own-patient-scan-and-mask). A different patient does not require retraining when the trained protocol remains suitable and testing conventions match; a changed acquisition protocol still requires appropriate adaptation/training.

Repeated GUI predictions automatically preserve earlier results under numbered filenames. **Open an earlier prediction** deliberately retrieves a completed result. CLI users add `--new-output` to choose an unused numbered name; without that flag, the terminal retains its explicit overwrite guard.

Terminal, from `LoRACM/`:

```bash
python infer.py --run workspaces/my_project/runs/adaptation --input workspaces/my_project/testing.npz --output workspaces/my_project/predictions.npz --samples 1000 --batch-size 16 --sample-chunk 64 --steps 3 --seed 0 --device auto
```

For another inference setting without replacing the first prediction:

```bash
python infer.py --run workspaces/my_project/runs/adaptation --input workspaces/my_project/testing.npz --output workspaces/my_project/predictions.npz --new-output --samples 2000 --steps 5 --device auto
```

The terminal prints the actual saved filename. Each result retains its input/checkpoint checksum and inference settings.

`auto` prefers CUDA, then Apple MPS, then CPU and records the actual device in prediction metadata. Inference merges LoRA adapter weights into the base layers once; original checkpoints are unchanged. For an Apple GPU use `--device mps`; for a supported CUDA GPU use `--device cuda:0`. Reduce `--batch-size` or `--sample-chunk` to control memory. On one node with multiple visible CUDA devices, add `--multi-gpu`.

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
2. Choose posterior draws (default 1,000), reverse sampling steps (3), sample chunk (64), random seed (0) and device (`auto`). The steps are consistency-sampling evaluations, not training noise intervals; more steps do not guarantee better estimates.
3. Enable **Compare with nonlinear least squares** and select the matching protocol JSON. Generated projects preserve it; for imported datasets, define the same kinetic model and protocol separately.
4. Review/edit the NLS parameter bounds and acknowledge the displayed input interpolation assumptions.
5. Choose **Predict posterior and compare**. The page plots one posterior histogram per parameter, overlays a **dashed vertical NLS estimate**, shows posterior summaries and a graph with the measured TAC, AIF, NLS fitted TAC and forward TAC evaluated at the componentwise posterior median, and provides figure/joint-sample downloads.

Enable **Overlay forward TAC from posterior median** to simulate without NLS. A matching protocol is still required. With the optional scratch route enabled, **Overlay CM from scratch posterior** adds its distribution and median forward TAC for the same voxel and inference settings. New predictions replace the active display immediately; earlier analyses are available only through the explicit saved-analysis selector.

The median forward curve is `forward_model(median(parameters))`. It is not the median of posterior-predictive TACs, because the forward model is nonlinear. Parameters are not clipped to NLS bounds.

**How to read the figure.** The histogram is the marginal distribution of samples from your adapted consistency model. The dashed line is a bounded NLS point estimate, not ground truth or an uncertainty interval. Agreement is useful, but both estimators depend on the forward model, input and data conventions. Inspect convergence, active bounds, residual fit, posterior tails and independent truth when available. No line is drawn for a nonconverged NLS fit.

NLS uses three deterministic initializations and **unweighted** squared TAC residuals. Bounds default to the configured uniform kinetic-prior ranges. It uses the same plugin/built-in kinetics and observation mode as the protocol: the original Euler scheme for the UC Davis paper recipe, or RK4 for custom protocols. Supplied sampled AIF values are linearly interpolated, with zero inserted at the origin and the last input held through the final frame. For frame-averaged AIFs, this is an explicit continuous-input approximation. UC Davis retains the original CSV sample values and Euler observation times for both NLS and the posterior-median forward overlay; its patient-label convention is documented in [the UC Davis guide](UCDAVIS.md).

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
python analyse_tac.py --run workspaces/my_project/runs/adaptation --input single_tac.npz --protocol workspaces/my_project/runs/adaptation/protocol.json --posterior-output workspaces/my_project/posterior.npz --output workspaces/my_project/nls.json --samples 1000 --steps 3 --sample-chunk 64 --seed 0 --device auto

python plot_posteriors.py --posterior workspaces/my_project/posterior.npz --nls workspaces/my_project/nls.json --output workspaces/my_project/posteriors.png

python plot_tac.py --input single_tac.npz --forward workspaces/my_project/forward.json --nls workspaces/my_project/nls.json --output workspaces/my_project/tac_comparison.png
```

Omit `--protocol` for posterior-only analysis. `--checkpoint best.pth` selects the lowest validation-loss completed epoch even after early stopping; use `last.pth` or `epochs/epoch_000025.pth` to choose another saved epoch. Add `--skip-nls` to keep median forward simulation without NLS; add `--comparison-run workspaces/my_project/runs/cm_scratch` and optional `--comparison-checkpoint` to analyse both models. To run NLS without the CM, omit `--run` and supply `--protocol`, `--input` and `--output`. `infer.py --save-samples` can retain joint draws for small batches, capped at two million stored parameter values to avoid accidental large outputs.

## Apple MPS batch safety

Each network call expands the curve batch by the sample chunk. A setting of 512 curves × 200 draws creates 102,400 network rows, which exceeds a [reported MPS linear-layer limit](https://github.com/pytorch/pytorch/issues/189495). The Mac map investigated for this update showed errors precisely in each expanded batch’s tail above 65,536 rows; evaluating the same checkpoint on CPU removed those jumps. This supports an inference/backend explanation rather than an incorrect voxel mapping.

The updated path caps expanded MPS calls at 4,096 rows, uses CPU-generated random noise with MPS model execution, and checks selected rows from the effective batch against CPU using identical tensors. It stops if that check fails. Requested and effective batches, the numerical check and random-noise device are recorded in metadata. Old oversized MPS maps are flagged in step 6. Regenerate those predictions with a new filename; their values cannot be repaired by changing colour limits. GPU results may differ slightly from CPU due to floating-point and sampling differences.

## Outputs and maps

Prediction NPZ contains `params_mean`, `params_median`, `params_std`, negative-draw fractions, parameter names, curve identifiers, time vectors and metadata. `posterior_samples` is included when requested, shaped `[N, samples, P]`. Optional truth yields parameter RMSE/bias of posterior means in metadata; it does not test posterior calibration.

Single-TAC analysis also saves `forward.json` with the posterior medians, simulated curves and fit RMSE for each selected model, and `analysis.json` after all requested work finishes. The GUI exports a TAC-comparison CSV; `plot_tac.py` exports the same comparison as a scientific figure.

For **built-in 2TC only**, Kᵢ is calculated for each joint draw as `K1*k3/(k2+k3)`, then summarized. The UC Davis paper recipe explicitly floors K1/k2/k3 to 1e-4 per draw first (`ki_policy=hpc_floor_1e-4`); general custom projects use the raw transform. Near-zero denominators in raw mode are excluded and a valid fraction is recorded. Custom models do not automatically get a Kᵢ interpretation.

Spatial mapping needs verified voxel indices, ordering, orientation and spacing. The [UC Davis imaging notebook](UCDAVIS.md#4-make-parametric-images) handles the supplied coronal slice once its geometry is verified; it is not a general 3D image importer. A list of TACs or equal voxel count alone does not establish spatial correspondence.
